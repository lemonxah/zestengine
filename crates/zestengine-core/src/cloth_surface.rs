//! Continuous body contacts across cloth faces, including their interiors.
use super::*;

pub struct ClothSurfaceContact {
    pub weights:[f32;3],
    pub push:Vec3,
    pub normal:Vec3,
    pub carried:Vec3,
}

impl ClothColliderQuery<'_> {
    /// Spheres/capsules use closest features of the complete triangle.
    /// Concave fields use a covering hierarchy of swept spheres; unresolved
    /// leaves stay conservative instead of leaving gaps between point probes.
    /// Returns one constraint; the solver iterates for simultaneous contacts.
    pub fn triangle(&self,start:[Vec3;3],end:[Vec3;3],radius:f32)->Option<ClothSurfaceContact>{
        if start.iter().chain(&end).any(|p|!p.is_finite())||!radius.is_finite(){return None;}
        let radius=radius.max(0.0);
        if self.field.is_some(){return self.field_triangle(start,end,radius);}
        if matches!(self.motion.end,ClothShapeWorld::Disabled){return None;}
        let sample=|time:f32|{
            let triangle=std::array::from_fn(|i|start[i].lerp(end[i],time));
            match self.motion.at(time){
                ClothShapeWorld::Sphere{c,r}=>{let(q,w)=closest_triangle(c,triangle);(q.distance(c)-r.max(0.0),w)},
                ClothShapeWorld::Capsule{a,b,ra,rb}=>{
                    let edge=b-a;let dr=rb-ra;
                    let derivative=|u:f32|{let center=a+edge*u;let(q,_)=closest_triangle(center,triangle);edge.dot((center-q).normalize_or_zero())-dr};
                    let u=if derivative(0.0)>=0.0{0.0}else if derivative(1.0)<=0.0{1.0}else{
                        let(mut lo,mut hi)=(0.0,1.0);for _ in 0..32{let m=(lo+hi)*0.5;if derivative(m)<0.0{lo=m}else{hi=m}}(lo+hi)*0.5
                    };
                    let center=a+edge*u;let(q,w)=closest_triangle(center,triangle);(q.distance(center)-(ra+dr*u).max(0.0),w)
                },
                ClothShapeWorld::Plane{p,n}=>{
                    let n=n.normalize_or(Vec3::Y);let i=(0..3).min_by(|&i,&j|(triangle[i]-p).dot(n).total_cmp(&(triangle[j]-p).dot(n))).unwrap();
                    let mut w=[0.0;3];w[i]=1.0;((triangle[i]-p).dot(n),w)
                },
                _=>(f32::INFINITY,[0.0;3]),
            }
        };
        let speed=(0..3).map(|i|self.motion.speed_bound(start[i],end[i])).fold(0.0,f32::max);
        let(time,conservative)=advance(|t|sample(t).0,speed,radius)?;let(_,weights)=sample(time);
        let p0=(0..3).map(|i|start[i]*weights[i]).sum::<Vec3>();let p1=(0..3).map(|i|end[i]*weights[i]).sum::<Vec3>();
        let normal=shape_sample(p0.lerp(p1,time),self.motion.at(time)).1;
        let(out,normal,carried)=respond_to_collider(p0,p1,radius,self.motion.clone(),ClothSweep{time,normal,conservative})?;
        Some(ClothSurfaceContact{weights,push:out-p1,normal,carried})
    }

    fn field_triangle(&self,start:[Vec3;3],end:[Vec3;3],radius:f32)->Option<ClothSurfaceContact>{
        let field=self.field.as_ref().unwrap();let point=|p:&[Vec3;3],w:Vec3|p[0]*w.x+p[1]*w.y+p[2]*w.z;
        if field.triangle_separated(start,end,radius){return None;}
        // Refine the earliest possible impact first. A depth-first walk can
        // spend its budget on one harmless patch, then report a very large
        // unrefined neighbour as an artificial early collision.
        let bound=|patch:[Vec3;3],depth:u32|{
            if depth>0&&field.triangle_separated(patch.map(|w|point(&start,w)),patch.map(|w|point(&end,w)),radius){return None;}
            let weights=(patch[0]+patch[1]+patch[2])/3.0;
            let p0=point(&start,weights);let p1=point(&end,weights);
            let half=patch.iter().map(|&w|point(&start,w).distance(p0).max(point(&end,w).distance(p1))).fold(0.0,f32::max);
            field.first_contact(p0,p1,radius+half).map(|hit|(hit.time,half,patch,depth))
        };
        let mut pending=vec![bound([Vec3::X,Vec3::Y,Vec3::Z],0)?];let mut visited=0;
        let tolerance=(radius*0.125).clamp(0.00025,0.001);
        while !pending.is_empty(){
            let first=(0..pending.len()).min_by(|&a,&b|pending[a].0.total_cmp(&pending[b].0).then(pending[a].1.total_cmp(&pending[b].1))).unwrap();
            let(_,half,patch,depth)=pending.swap_remove(first);visited+=1;
            let weights=(patch[0]+patch[1]+patch[2])/3.0;let p0=point(&start,weights);let p1=point(&end,weights);
            // Prefer a real contact witness to the enclosing patch. Subsequent
            // iterations revisit the remaining surface after this constraint
            // is satisfied. This also avoids exhaustively refining a broad,
            // already contacting flat region down to submillimetre patches.
            if let Some((out,normal,carried))=field.resolve(p0,p1,radius).filter(|(out,_,_)|out.distance_squared(p1)>1e-16){
                return Some(ClothSurfaceContact{weights:weights.to_array(),push:out-p1,normal,carried});
            }
            if half>tolerance&&depth<20&&visited<512{
                let (i,j)=[(0,1),(1,2),(2,0)].into_iter().max_by(|&(i,j),&(k,l)|{
                    let length=|i,j|point(&start,patch[i]).distance_squared(point(&start,patch[j])).max(point(&end,patch[i]).distance_squared(point(&end,patch[j])));
                    length(i,j).total_cmp(&length(k,l))
                }).unwrap();let middle=(patch[i]+patch[j])*0.5;
                let mut a=patch;let mut b=patch;a[j]=middle;b[i]=middle;
                if let Some(node)=bound(a,depth+1){pending.push(node);}if let Some(node)=bound(b,depth+1){pending.push(node);}continue;
            }
            if let Some((out,normal,carried))=field.resolve(p0,p1,radius+half){
                return Some(ClothSurfaceContact{weights:weights.to_array(),push:out-p1,normal,carried});
            }
        }
        None
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn moving_bodies_hit_face_interiors_when_every_vertex_and_edge_misses(){
        let face=[Vec3::new(-1.0,-1.0,0.0),Vec3::new(1.0,-1.0,0.0),Vec3::new(0.0,1.0,0.0)];
        for capsule in [false,true]{
            let shape=|z|if capsule{ClothShapeWorld::Capsule{a:Vec3::new(-0.05,0.0,z),b:Vec3::new(0.05,0.0,z),ra:0.03,rb:0.05}}else{ClothShapeWorld::Sphere{c:Vec3::Z*z,r:0.05}};
            let motion=ClothColliderMotion{start:shape(-1.0),end:shape(1.0)};let query=motion.prepare();
            for &p in &face{assert!(query.resolve(p,p,0.01).is_none());}
            for(i,j)in [(0,1),(1,2),(2,0)]{assert!(query.segment([face[i],face[j]],[face[i],face[j]],[0.01;2]).is_none());}
            let c=query.triangle(face,face,0.01).expect("the body crosses the face interior");
            assert!(c.weights.iter().all(|&w|w>0.0));assert!(c.push.z>=1.0599&&c.normal.z>0.99);assert!(c.carried.z>=1.9999);
        }
    }
    #[test]
    fn a_small_moving_level_set_cannot_pass_between_face_probes(){
        use crate::cloth_sdf::{ClothSdf,ClothSdfShape};
        let mut samples=Vec::new();for z in 0..5{for y in 0..5{for x in 0..5{
            let p=Vec3::new(x as f32,y as f32,z as f32)*0.05-Vec3::splat(0.1);let q=p.abs()-Vec3::splat(0.04);
            samples.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
        }}}
        let field=ClothSdf::from_samples([5;3],Vec3::splat(-0.1),Vec3::splat(0.05),samples).unwrap();
        let shape=|z|ClothShapeWorld::LevelSet(std::sync::Arc::new(ClothSdfShape::new(field.clone(),glam::Mat4::from_translation(Vec3::new(0.137,0.231,z))).unwrap()));
        let face=[Vec3::new(-1.0,-1.0,0.0),Vec3::new(1.0,-1.0,0.0),Vec3::new(0.0,1.0,0.0)];
        let motion=ClothColliderMotion{start:shape(-1.0),end:shape(1.0)};let query=motion.prepare();
        for &p in &face{assert!(query.resolve(p,p,0.01).is_none());}
        let c=query.triangle(face,face,0.01).expect("covering bounds must catch the small off-center solid");
        let witness=(0..3).map(|i|face[i]*c.weights[i]).sum::<Vec3>();
        assert!((c.weights.iter().sum::<f32>()-1.0).abs()<1e-6);assert!(c.weights.iter().all(|&w|w>0.0));
        assert!(c.push.dot(c.normal)>0.01&&c.carried.distance(Vec3::Z*2.0)<1e-5);
        // A corner witness may slide sideways; its correction must clear the
        // field, rather than assuming every valid contact normal is vertical.
        assert!(cloth_shape_distance(witness+c.push,shape(1.0)).0>=0.0099);
        let far=face.map(|p|p+Vec3::X*5.0);assert!(query.triangle(far,far,0.01).is_none());
    }
}
