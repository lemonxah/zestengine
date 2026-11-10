//! Closest-triangle skin transfer. A BVH bounds the search; barycentric weights
//! are merged by joint, truncated to four influences, and normalized.
use anyhow::{Result,ensure,bail};
use glam::Vec3;
use zestengine_render::{MeshData,Vertex};
#[derive(Debug)]
pub struct ClothSkinTransferReport{pub vertices:usize,pub max_distance:f32,pub mean_distance:f32}
struct Node{lo:Vec3,hi:Vec3,children:Option<(Box<Node>,Box<Node>)>,triangles:Vec<usize>}
impl Node{
    fn build(mesh:&MeshData,mut triangles:Vec<usize>)->Self{
        let mut lo=Vec3::splat(f32::INFINITY);let mut hi=Vec3::splat(f32::NEG_INFINITY);
        for &i in &triangles{for &v in &mesh.indices[i*3..i*3+3]{let p=Vec3::from(mesh.vertices[v as usize].position);lo=lo.min(p);hi=hi.max(p);}}
        let children=if triangles.len()>8{let span=hi-lo;let axis=if span.x>=span.y&&span.x>=span.z{0}else if span.y>=span.z{1}else{2};let center=|i:usize|mesh.indices[i*3..i*3+3].iter().map(|&v|mesh.vertices[v as usize].position[axis]).sum::<f32>();triangles.sort_by(|a,b|center(*a).total_cmp(&center(*b)));let right=triangles.split_off(triangles.len()/2);Some((Box::new(Self::build(mesh,std::mem::take(&mut triangles))),Box::new(Self::build(mesh,right))))}else{None};Self{lo,hi,children,triangles}
    }
    fn distance(&self,p:Vec3)->f32{(p-p.clamp(self.lo,self.hi)).length_squared()}
    fn nearest(&self,mesh:&MeshData,p:Vec3,best:&mut(f32,usize,[f32;3])){
        if self.distance(p)>best.0{return;}
        if let Some((a,b))=&self.children{let(first,second)=if a.distance(p)<=b.distance(p){(a,b)}else{(b,a)};first.nearest(mesh,p,best);second.nearest(mesh,p,best);return;}
        for &i in &self.triangles{let tri=&mesh.indices[i*3..i*3+3];let pts:[Vec3;3]=std::array::from_fn(|k|Vec3::from(mesh.vertices[tri[k] as usize].position));let(q,w)=closest(p,pts[0],pts[1],pts[2]);let d=q.distance_squared(p);if d<best.0||(d==best.0&&i<best.1){*best=(d,i,w);}}
    }
}
fn closest(p:Vec3,a:Vec3,b:Vec3,c:Vec3)->(Vec3,[f32;3]){
    let ab=b-a;let ac=c-a;let ap=p-a;let d1=ab.dot(ap);let d2=ac.dot(ap);if d1<=0.0&&d2<=0.0{return(a,[1.0,0.0,0.0]);}
    let bp=p-b;let d3=ab.dot(bp);let d4=ac.dot(bp);if d3>=0.0&&d4<=d3{return(b,[0.0,1.0,0.0]);}
    let vc=d1*d4-d3*d2;if vc<=0.0&&d1>=0.0&&d3<=0.0{let v=d1/(d1-d3);return(a+ab*v,[1.0-v,v,0.0]);}
    let cp=p-c;let d5=ab.dot(cp);let d6=ac.dot(cp);if d6>=0.0&&d5<=d6{return(c,[0.0,0.0,1.0]);}
    let vb=d5*d2-d1*d6;if vb<=0.0&&d2>=0.0&&d6<=0.0{let w=d2/(d2-d6);return(a+ac*w,[1.0-w,0.0,w]);}
    let va=d3*d6-d5*d4;if va<=0.0&&(d4-d3)>=0.0&&(d5-d6)>=0.0{let w=(d4-d3)/((d4-d3)+(d5-d6));return(b+(c-b)*w,[0.0,1.0-w,w]);}
    let inv=1.0/(va+vb+vc);let(v,w)=(vb*inv,vc*inv);(a+ab*v+ac*w,[1.0-v-w,v,w])
}
pub fn transfer_clothing_weights(target:&mut[Vertex],source:&MeshData,joint_count:usize,max_distance:f32)->Result<ClothSkinTransferReport>{
    ensure!(joint_count>0&&max_distance.is_finite()&&max_distance>0.0,"Skin transfer needs a skeleton and a positive finite distance limit");
    ensure!(source.indices.len()%3==0&&source.indices.iter().all(|&i|(i as usize)<source.vertices.len()),"Invalid skin source topology");
    ensure!(source.vertices.iter().all(|v|v.position.iter().all(|x|x.is_finite())&&v.weights.iter().zip(v.joints).all(|(w,j)|w.is_finite()&&*w>=0.0&&(*w==0.0||(j as usize)<joint_count))),"Invalid skin source weights or positions");
    ensure!(target.iter().all(|v|v.position.iter().all(|x|x.is_finite())),"Invalid garment position");
    let triangles=(0..source.indices.len()/3).filter(|&i|{let t=&source.indices[i*3..i*3+3];let p:[Vec3;3]=std::array::from_fn(|k|Vec3::from(source.vertices[t[k] as usize].position));(p[1]-p[0]).cross(p[2]-p[0]).length_squared()>1e-16}).collect::<Vec<_>>();ensure!(!triangles.is_empty(),"Skin source has no nondegenerate triangles");
    let tree=Node::build(source,triangles);let mut output=Vec::with_capacity(target.len());let mut maximum=0.0f32;let mut sum=0.0;
    for (i,v) in target.iter().enumerate(){let mut best=(max_distance*max_distance,usize::MAX,[0.0;3]);tree.nearest(source,Vec3::from(v.position),&mut best);if best.1==usize::MAX{bail!("Garment vertex {i} is farther than {max_distance} m from the skin source; align the bind meshes or increase the limit");}
        let mut weights=std::collections::BTreeMap::<u32,f32>::new();for (&vertex,bary) in source.indices[best.1*3..best.1*3+3].iter().zip(best.2){let source=&source.vertices[vertex as usize];for (joint,w) in source.joints.into_iter().zip(source.weights){if w>0.0{*weights.entry(joint).or_default()+=w*bary;}}}
        let mut weights=weights.into_iter().filter(|(_,w)|*w>1e-8).collect::<Vec<_>>();weights.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));weights.truncate(4);let total: f32=weights.iter().map(|(_,w)|*w).sum();ensure!(total>1e-8,"Skin source at garment vertex {i} has no bone influences");
        let mut joints=[0;4];let mut normalized=[0.0;4];for (k,(joint,w)) in weights.into_iter().enumerate(){joints[k]=joint;normalized[k]=w/total;}output.push((joints,normalized));let distance=best.0.sqrt();maximum=maximum.max(distance);sum+=distance;
    }
    for (v,(j,w)) in target.iter_mut().zip(output){v.joints=j;v.weights=w;}
    Ok(ClothSkinTransferReport{vertices:target.len(),max_distance:maximum,mean_distance:sum/target.len().max(1) as f32})
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn cloth_skin_transfer_interpolates_normalizes_and_is_atomic_on_failure(){
        let source=MeshData{vertices:vec![Vertex{position:[0.0,0.0,0.0],joints:[0,0,0,0],weights:[1.0,0.0,0.0,0.0],..Default::default()},Vertex{position:[1.0,0.0,0.0],joints:[1,0,0,0],weights:[1.0,0.0,0.0,0.0],..Default::default()},Vertex{position:[0.0,1.0,0.0],joints:[2,0,0,0],weights:[1.0,0.0,0.0,0.0],..Default::default()}],indices:vec![0,1,2],has_lightmap_uv:false};
        let mut target=vec![Vertex{position:[0.25,0.25,0.05],uv:[0.4,0.7],..Default::default()}];let r=transfer_clothing_weights(&mut target,&source,3,0.1).unwrap();assert_eq!(r.vertices,1);assert!((r.max_distance-0.05).abs()<1e-6);assert_eq!(target[0].uv,[0.4,0.7]);
        assert_eq!(target[0].joints,[0,1,2,0]);assert_eq!(target[0].weights,[0.5,0.25,0.25,0.0]);
        target.push(Vertex{position:[10.0;3],..Default::default()});let previous=target[0].weights;assert!(transfer_clothing_weights(&mut target,&source,3,0.1).is_err());assert_eq!(target[0].weights,previous);
        assert!(transfer_clothing_weights(&mut target[..1],&source,1,0.1).is_err());
    }
}
