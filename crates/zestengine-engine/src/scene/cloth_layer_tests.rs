//! Coupled garment, level-set and LOD regression. Check the visible surface,
//! not just solver particles: residual detail must not hide penetrations.
use super::*;
use zestengine_core::{ClothCollider,ClothColliders,ClothShape,ObjectRef,ClothLod,ClothLodLevel};

fn tube(radius:f32,top:f32,bottom:f32)->MeshData{
    let ring=12u32;let rows=6u32;let mut vertices=Vec::new();let mut indices=Vec::new();
    for row in 0..=rows{for k in 0..ring{
        let a=std::f32::consts::TAU*k as f32/ring as f32;let normal=Vec3::new(a.cos(),0.0,a.sin());
        vertices.push(Vertex{position:[radius*a.cos(),top+(bottom-top)*row as f32/rows as f32,radius*a.sin()],normal:normal.to_array(),
            uv:[k as f32/ring as f32,row as f32/rows as f32],color:[if row==0{1.0}else{0.0},1.0,1.0,1.0],..Default::default()});
    }}
    for row in 0..rows{for k in 0..ring{let a=row*ring+k;let b=row*ring+(k+1)%ring;indices.extend([a,b,a+ring,b,b+ring,a+ring]);}}
    MeshData{vertices,indices,has_lightmap_uv:false}
}
fn body_field()->zestengine_core::cloth_sdf::ClothSdf{
    // Closed extrusion of a concave hip/two-leg outline. The gap between legs
    // is real empty space rather than the convex hull of the body.
    let outline=[[-0.35,1.2],[0.35,1.2],[0.35,-0.4],[0.12,-0.4],[0.12,0.6],[-0.12,0.6],[-0.12,-0.4],[-0.35,-0.4]];
    let p:Vec<_>=[-0.22,0.22].into_iter().flat_map(|z|outline.map(|[x,y]|Vec3::new(x,y,z))).collect();
    let mut indices=Vec::new();for t in [[0,1,4],[0,4,5],[1,2,4],[2,3,4],[0,5,7],[5,6,7]]{indices.extend(t);indices.extend([t[2]+8,t[1]+8,t[0]+8]);}
    for i in 0..8{let j=(i+1)%8;indices.extend([i,i+8,j,j,i+8,j+8]);}
    let field=zestengine_core::cloth_sdf::ClothSdf::from_mesh(&p,&indices,33).unwrap();
    assert!(field.sample(Vec3::new(0.0,0.0,0.0)).0>0.08,"leg gap stays open");
    assert!(field.sample(Vec3::new(0.23,0.0,0.0)).0<0.0,"legs stay solid");field
}
fn body_distance(p:Vec3)->f32{
    // Independent analytic distance to the same three-box union.
    [(Vec3::new(0.0,0.9,0.0),Vec3::new(0.35,0.3,0.22)),(Vec3::new(-0.235,0.1,0.0),Vec3::new(0.115,0.5,0.22)),(Vec3::new(0.235,0.1,0.0),Vec3::new(0.115,0.5,0.22))]
        .into_iter().map(|(c,h)|{let q=(p-c).abs()-h;q.max(Vec3::ZERO).length()+q.max_element().min(0.0)}).fold(f32::INFINITY,f32::min)
}
fn face_penetrates_body(face:[Vec3;3],tolerance:f64)->bool{
    // Clip the entire triangle against each analytic solid, inset by the
    // allowed contact tolerance. This also catches a body corner penetrating
    // a face when all three cloth vertices remain outside the collider.
    for (center,half) in [([0.0,0.9,0.0],[0.35,0.3,0.22]),([-0.235,0.1,0.0],[0.115,0.5,0.22]),([0.235,0.1,0.0],[0.115,0.5,0.22])]{
        let mut polygon=face.map(|p|p.as_dvec3()).to_vec();
        for axis in 0..3{for sign in [-1.0,1.0]{
            let input=std::mem::take(&mut polygon);if input.is_empty(){break;}
            let distance=|p:glam::DVec3|sign*(p[axis]-center[axis])-(half[axis]-tolerance);
            let mut previous=*input.last().unwrap();let mut previous_distance=distance(previous);
            for point in input{
                let d=distance(point);
                if (d<=0.0)!=(previous_distance<=0.0){polygon.push(previous+(point-previous)*(previous_distance/(previous_distance-d)));}
                if d<=0.0{polygon.push(point);}
                previous=point;previous_distance=d;
            }
        }}
        if !polygon.is_empty(){return true;}
    }false
}
#[test]
fn body_reference_detects_face_interior_penetration(){
    let face=[Vec3::new(-0.6,0.9,0.0),Vec3::new(0.6,0.9,0.0),Vec3::new(0.0,0.9,0.6)];
    assert!(face.iter().all(|&p|body_distance(p)>0.0));assert!(face_penetrates_body(face,0.001));
    assert!(!face_penetrates_body(face.map(|p|p+Vec3::Y),0.001));
    assert!(!face_penetrates_body([Vec3::new(-0.1,0.0,0.0),Vec3::new(0.1,0.0,0.0),Vec3::new(0.0,0.5,0.0)],0.001),"the concave leg gap is empty");
}
fn rendered(s:&LoadedScene,source:&AssetScene,i:usize)->Vec<Vec3>{
    let mesh=&source.meshes[i-1];let mut p=mesh.vertices.iter().map(|v|Vec3::from(v.position)).collect::<Vec<_>>();let mut normals=vec![Vec3::ZERO;p.len()];
    cloth(s,i).runtime.sim.as_ref().unwrap().apply(&mut p,&mut normals,s.world_transforms()[i]);p
}
fn crossings(a:&[Vec3],ai:&[u32],b:&[Vec3],bi:&[u32])->usize{
    let mut count=0;
    for(p,indices,q,triangles)in [(a,ai,b,bi),(b,bi,a,ai)]{
        let mut edges=std::collections::BTreeSet::new();for t in indices.as_chunks::<3>().0{for [i,j] in [[t[0],t[1]],[t[1],t[2]],[t[2],t[0]]]{edges.insert((i.min(j),i.max(j)));}}
        for(i,j)in edges{let start=p[i as usize].as_dvec3();let end=p[j as usize].as_dvec3();for t in triangles.as_chunks::<3>().0{
            let v=q[t[0]as usize].as_dvec3();let u=q[t[1]as usize].as_dvec3()-v;let w=q[t[2]as usize].as_dvec3()-v;let n=u.cross(w).normalize_or_zero();
            let da=(start-v).dot(n);let db=(end-v).dot(n);if da*db>=0.0||da.abs().min(db.abs())<1e-5{continue;}
            let d=end-start;let h=d.cross(w);let det=u.dot(h);if det.abs()<1e-20{continue;}
            let offset=start-v;let x=offset.dot(h)/det;let r=offset.cross(u);let y=d.dot(r)/det;
            if x>=-1e-8&&y>=-1e-8&&x+y<=1.0+1e-8{count+=1;}
        }}
    }count
}

fn surface_gap(a:&[Vec3],ai:&[u32],b:&[Vec3],bi:&[u32])->f64{
    let mut best=f64::INFINITY;
    for(p,q,triangles)in [(a,b,bi),(b,a,ai)]{for p in p{let p=p.as_dvec3();for t in triangles.as_chunks::<3>().0{
        let v=t.iter().map(|&i|q[i as usize].as_dvec3()).collect::<Vec<_>>();let u=v[1]-v[0];let w=v[2]-v[0];let normal=u.cross(w);let square=normal.length_squared();
        if square>1e-24{
            let projection=p-normal*((p-v[0]).dot(normal)/square);
            let weights=[(v[1]-projection).cross(v[2]-projection).dot(normal)/square,(v[2]-projection).cross(v[0]-projection).dot(normal)/square,(v[0]-projection).cross(v[1]-projection).dot(normal)/square];
            if weights.iter().all(|w|*w>=0.0){best=best.min(p.distance(projection));}
        }
        for(i,j)in [(0,1),(1,2),(2,0)]{let d=v[j]-v[i];let t=if d.length_squared()>1e-24{((p-v[i]).dot(d)/d.length_squared()).clamp(0.0,1.0)}else{0.0};best=best.min(p.distance(v[i]+d*t));}
    }}}
    let edges=|indices:&[u32]|{let mut e=std::collections::BTreeSet::new();for t in indices.as_chunks::<3>().0{for(i,j)in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]{e.insert((i.min(j)as usize,i.max(j)as usize));}}e};
    let ae=edges(ai);let be=edges(bi);
    for &(i,j)in &ae{for &(k,l)in &be{
        let origin=a[i].as_dvec3();let other=b[k].as_dvec3();let u=a[j].as_dvec3()-origin;let v=b[l].as_dvec3()-other;let w=origin-other;
        let aa=u.dot(u);let bb=u.dot(v);let cc=v.dot(v);let dd=u.dot(w);let ee=v.dot(w);let det=aa*cc-bb*bb;
        if aa<1e-24||cc<1e-24{continue;}
        let mut s=if det>1e-24{((bb*ee-cc*dd)/det).clamp(0.0,1.0)}else{0.0};let mut t=(bb*s+ee)/cc;
        if t<0.0{t=0.0;s=(-dd/aa).clamp(0.0,1.0);}else if t>1.0{t=1.0;s=((bb-dd)/aa).clamp(0.0,1.0);}
        best=best.min((origin+u*s).distance(other+v*t));
    }}best
}

#[test]
fn layered_coat_and_skirt_follow_a_concave_body_through_lod_changes(){
    for reversed in [false,true]{layered_coat_and_skirt(reversed);}
}
fn layered_coat_and_skirt(reversed:bool){
    let mut source=asset();source.meshes=vec![tube(0.447,1.3,0.15),tube(0.44,0.75,-0.3)];
    source.instances=vec![AssetInstance::single("Coat",0,0,Mat4::IDENTITY),AssetInstance::single("Skirt",1,0,Mat4::IDENTITY)];
    let mut s=loaded(&source);s.objects.set_id(1, ObjectId::from_raw(if reversed{2}else{1})).expect("unique saved object ID");s.objects.set_id(2, ObjectId::from_raw(if reversed{1}else{2})).expect("unique saved object ID");
    let field=body_field();let body_id=ObjectId::new();let body=s.objects.len();
    let mut object=SceneObject::new_empty(body_id,"Concave body");let mut collider=ClothCollider{shape:ClothShape::LevelSet,..Default::default()};collider.level_set.field=Some(field);
    object.components.push(Box::new(ClothColliders{colliders:vec![collider],..Default::default()}));s.objects.push(object);
    for i in [1,2]{s.objects[i].components.push(Box::new(MeshCloth{thickness:0.018,radius:0.008,iterations:6,damping:0.12,self_collision:true,collider_objects:vec![ObjectRef(Some(body_id))],
        lod:ClothLod{enabled:true,hysteresis:0.5,transition:0.2,levels:vec![ClothLodLevel{distance:5.0,particle_spacing:0.28,frequency:60.0,iterations:6}]},..Default::default()}));}
    s.set_cloth_viewer(Some(Vec3::ZERO));s.pose_skeletons(0.0);
    let fine=[cloth(&s,1).runtime.sim.as_ref().unwrap().particles.len(),cloth(&s,2).runtime.sim.as_ref().unwrap().particles.len()];
    let mut timings=Vec::new();let mut max_pop=0.0f32;let mut minimum_body=f32::INFINITY;let mut minimum_gap=[f64::INFINITY;2];
    for frame in 0..120{
        let t=frame as f32/60.0;let move_by=Vec3::new(0.15*(t*8.0).sin(),0.0,0.1*(t*5.0).sin());let rotation=Quat::from_rotation_y(0.2*(t*3.0).sin());
        if [20,50,80].contains(&frame){
            let before=[rendered(&s,&source,1),rendered(&s,&source,2)];s.set_cloth_viewer(Some(if frame==50{Vec3::ZERO}else{Vec3::Z*10.0}));s.pose_skeletons(0.0);
            for i in [1,2]{let after=rendered(&s,&source,i);let pop=before[i-1].iter().zip(after).map(|(a,b)|a.distance(b)).fold(0.0,f32::max);max_pop=max_pop.max(pop);assert!(pop<0.02,"LOD switch moves a garment more than its contact margin: frame={frame}, garment={i}, {pop} m");
                let particles=cloth(&s,i).runtime.sim.as_ref().unwrap().particles.len();if frame==50{assert_eq!(particles,fine[i-1]);}else{assert!(particles<fine[i-1],"LOD must actually reduce resolution");}
            }
        }
        for i in [1,2,body]{s.objects[i].transform_mut().translation=move_by;s.objects[i].transform_mut().rotation=rotation;}
        let started=std::time::Instant::now();s.pose_skeletons(1.0/60.0);timings.push(started.elapsed().as_secs_f64()*1000.0);
        let p=[rendered(&s,&source,1),rendered(&s,&source,2)];
        for i in 0..2{for(v,point)in p[i].iter().enumerate(){
            assert!(point.is_finite());if source.meshes[i].vertices[v].color[0]==1.0{assert!(point.distance(Vec3::from(source.meshes[i].vertices[v].position))<1e-5,"pins stay attached");}
            let clearance=body_distance(*point);minimum_body=minimum_body.min(clearance);
            if clearance< -0.001{
                let c=cloth(&s,i+1);let world=s.world_transforms()[i+1];let sim=c.runtime.sim.as_ref().unwrap();
                let particle_min=sim.particles.iter().map(|p|body_distance(world.inverse().transform_point3(p.pos))).fold(f32::INFINITY,f32::min);
                let sdf=c.runtime.collider_history.values().find_map(|shape|if let zestengine_core::ClothShapeWorld::LevelSet(s)=shape{Some(s.sample(world.transform_point3(*point)).0)}else{None});
                eprintln!("body miss: point={point:?}, particles={particle_min}, sdf={sdf:?}, LOD={}",c.runtime.lod_level);
            }
            assert!(clearance>=-0.001,"visible cloth penetrates the body: frame={frame}, garment={i}, vertex={v}, clearance={clearance}");
        }}
        for (i, i_value) in p.iter().enumerate() {for (face,t) in source.meshes[i].indices.as_chunks::<3>().0.iter().enumerate(){
            assert!(!face_penetrates_body([i_value[t[0]as usize],i_value[t[1]as usize],i_value[t[2]as usize]],0.001),"body penetrates garment face: frame={frame}, garment={i}, face={face}");
        }}
        let gap=surface_gap(&p[0],&source.meshes[0].indices,&p[1],&source.meshes[1].indices);let tier=cloth(&s,1).runtime.lod_level;minimum_gap[tier]=minimum_gap[tier].min(gap);
        assert!(gap>=0.0178,"garment contact loses its 18 mm thickness: frame={frame}, LOD={tier}, gap={gap} m");
        assert_eq!(crossings(&p[0],&source.meshes[0].indices,&p[1],&source.meshes[1].indices),0,"visible garments cross at frame {frame}");
    }
    timings.sort_by(f64::total_cmp);eprintln!("layered cloth (reversed={reversed}): median {:.3} ms, p95 {:.3} ms, max LOD displacement {max_pop:.6} m, minimum body distance {minimum_body:.6} m, garment gap by tier {minimum_gap:?}",timings[60],timings[114]);
}
