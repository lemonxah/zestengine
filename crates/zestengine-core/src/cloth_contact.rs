//! Swept cloth contact primitives. Conservative advancement uses a bound on
//! relative feature speed, not a fixed number of time samples. Queries that
//! exhaust their iteration budget return a conservative contact at the last
//! proven-safe time rather than silently accepting a possible crossing.
use glam::{Vec3, Quat};
use crate::ClothShapeWorld;
#[path="cloth_contact_mapping.rs"]
mod mapping;
pub use mapping::ClothContactMapping;
#[path="cloth_surface.rs"]
mod surface;
pub use surface::ClothSurfaceContact;
#[path="cloth_untangle.rs"]
mod untangling;

#[derive(Clone, Debug)]
pub struct ClothColliderMotion {
    pub start: ClothShapeWorld,
    pub end: ClothShapeWorld,
}
impl ClothColliderMotion {
    pub fn stationary(shape:ClothShapeWorld)->Self{Self{start:shape.clone(),end:shape}}
    /// Prepare field motion once per collider/substep rather than per particle.
    pub fn prepare(&self)->ClothColliderQuery<'_>{ClothColliderQuery{motion:self,field:self.level_set()}}
    fn level_set(&self)->Option<crate::cloth_sdf::ClothSdfMotion>{
        let ClothShapeWorld::LevelSet(end)=&self.end else{return None};
        let start=if let ClothShapeWorld::LevelSet(start)=&self.start{start.clone()}else{end.clone()};
        // A handedness flip cannot be swept through a nonsingular frame.
        // Treat it as replacement geometry and still enforce its final body.
        crate::cloth_sdf::ClothSdfMotion::new(start,end.clone()).or_else(|_|crate::cloth_sdf::ClothSdfMotion::new(end.clone(),end.clone())).ok()
    }
    pub fn at(&self,t:f32)->ClothShapeWorld {
        use ClothShapeWorld::*;
        match (self.start.clone(),self.end.clone()) {
            (Sphere{c:a,r:ra},Sphere{c:b,r:rb})=>Sphere{c:a.lerp(b,t),r:ra+(rb-ra)*t},
            (Capsule{a:a0,b:b0,ra:ra0,rb:rb0},Capsule{a:a1,b:b1,ra:ra1,rb:rb1})=>Capsule{a:a0.lerp(a1,t),b:b0.lerp(b1,t),ra:ra0+(ra1-ra0)*t,rb:rb0+(rb1-rb0)*t},
            (Plane{p:a,n:na},Plane{p:b,n:nb})=>{
                let na=na.normalize_or(Vec3::Y);let nb=nb.normalize_or(Vec3::Y);
                let n=Quat::IDENTITY.slerp(Quat::from_rotation_arc(na,nb),t)*na;
                Plane{p:a.lerp(b,t),n}
            },
            (LevelSet(a),LevelSet(b))=>{
                let world=crate::cloth_sdf::ClothSdfMotion::new(a,b.clone()).map(|m|m.at(t)).unwrap_or(b.transform());
                crate::cloth_sdf::ClothSdfShape::new(b.field.clone(),world).map(|s|LevelSet(std::sync::Arc::new(s))).unwrap_or(Disabled)
            },
            // A replacement is a new collider, not a sweep through unrelated geometry.
            _=>self.end.clone(),
        }
    }
    pub fn interval(&self,a:f32,b:f32)->Self{Self{start:self.at(a),end:self.at(b)}}
    fn speed_bound(&self,p0:Vec3,p1:Vec3)->f32 {
        use ClothShapeWorld::*;
        let v=p1-p0;
        match (self.start.clone(),self.end.clone()) {
            (Sphere{c:a,r:ra},Sphere{c:b,r:rb})=>(v-(b-a)).length()+(rb-ra).abs(),
            (Capsule{a:a0,b:b0,ra:ra0,rb:rb0},Capsule{a:a1,b:b1,ra:ra1,rb:rb1})=>
                (v-(a1-a0)).length().max((v-(b1-b0)).length())+(ra1-ra0).abs().max((rb1-rb0).abs()),
            (Plane{p:a,n:na},Plane{p:b,n:nb})=>{
                let angle=na.normalize_or(Vec3::Y).dot(nb.normalize_or(Vec3::Y)).clamp(-1.0,1.0).acos();
                (v-(b-a)).length()+angle*(p0-a).length().max((p1-b).length())
            },
            _=>v.length(),
        }
    }
}

pub struct ClothColliderQuery<'a>{motion:&'a ClothColliderMotion,field:Option<crate::cloth_sdf::ClothSdfMotion>}
impl ClothColliderQuery<'_>{
    pub fn end(&self)->&ClothShapeWorld{&self.motion.end}
    pub fn resolve(&self,p0:Vec3,p1:Vec3,radius:f32)->Option<(Vec3,Vec3,Vec3)>{
        if let Some(field)=&self.field{field.resolve(p0,p1,radius)}else{resolve_cloth_collider(p0,p1,radius,self.motion.clone())}
    }
    pub fn segment(&self,start:[Vec3;2],end:[Vec3;2],radii:[f32;2])->Option<(f32,Vec3,Vec3,Vec3)>{
        if let Some(field)=&self.field{field.resolve_segment(start,end,radii)}else{resolve_cloth_segment(start,end,radii,self.motion.clone())}
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClothSweep {
    pub time:f32,
    /// Outward normal from the obstacle/second feature to the first feature.
    pub normal:Vec3,
    /// A conservative bound was reached before an exact proximity was found.
    pub conservative:bool,
}
#[derive(Clone, Copy, Debug)]
pub struct ClothFeatureContact {
    pub sweep:ClothSweep,
    /// Coefficients of first feature minus second feature (sum is zero).
    pub coefficients:[f32;4],
}

/// Returns the first proximity along a continuous path. Distance is Lipschitz
/// with `speed`; all skipped intervals are therefore proven separated.
fn advance(mut distance:impl FnMut(f32)->f32,speed:f32,margin:f32)->Option<(f32,bool)> {
    let tolerance=(margin*1e-4).max(1e-7);
    let mut t=0.0;
    for _ in 0..192 {
        let gap=distance(t)-margin;
        if !gap.is_finite(){return None;}
        if gap<=tolerance{return Some((t,false));}
        if speed<=1e-12{return None;}
        let safe=gap/speed;
        if safe>1.0-t{return None;}
        let next=t+safe*0.9;
        if next<=t{return Some((t,true));}
        t=next;
    }
    Some((t,true))
}

/// Signed distance to the union of linearly tapered spheres. Unlike projecting
/// onto the axis, this finds the true closest taper parameter.
fn shape_sample(p:Vec3,shape:ClothShapeWorld)->(f32,Vec3,f32) {
    match shape {
        ClothShapeWorld::Disabled=>(f32::INFINITY,Vec3::Y,0.0),
        ClothShapeWorld::LevelSet(sdf)=>{let(d,n)=sdf.sample(p);(d,n,0.0)},
        ClothShapeWorld::Sphere{c,r}=>{let d=p-c;(d.length()-r.max(0.0),d.normalize_or(Vec3::Y),0.0)},
        ClothShapeWorld::Capsule{a,b,ra,rb}=>{
            let e=b-a;let length=e.length();let dr=rb-ra;
            let t=if length<1e-8 {if rb>ra{1.0}else{0.0}}
                else if dr.abs()>=length{if dr>0.0{1.0}else{0.0}}
                else{let axis=e/length;let x=(p-a).dot(axis);let h=((p-a)-axis*x).length();((x+dr*h/(length*length-dr*dr).sqrt())/length).clamp(0.0,1.0)};
            let d=p-a.lerp(b,t);(d.length()-(ra+dr*t).max(0.0),d.normalize_or(e.normalize_or(Vec3::Y).any_orthonormal_vector()),t)
        },
        ClothShapeWorld::Plane{p:o,n}=>{let n=n.normalize_or(Vec3::Y);((p-o).dot(n),n,0.0)},
    }
}
pub fn cloth_shape_distance(p:Vec3,shape:ClothShapeWorld)->(f32,Vec3){let(d,n,_)=shape_sample(p,shape);(d,n)}

pub fn sweep_cloth_collider(p0:Vec3,p1:Vec3,radius:f32,motion:ClothColliderMotion)->Option<ClothSweep>{
    if !p0.is_finite()||!p1.is_finite(){return None;}
    if let Some(sdf)=motion.level_set(){return sdf.first_contact(p0,p1,radius);}
    let(t,conservative)=advance(|t|shape_sample(p0.lerp(p1,t),motion.at(t)).0,motion.speed_bound(p0,p1),radius.max(0.0))?;
    Some(ClothSweep{time:t,normal:shape_sample(p0.lerp(p1,t),motion.at(t)).1,conservative})
}

/// Position response against a moving primitive, and its contact displacement
/// over the whole interval (used for friction in the collider's moving frame).
pub fn resolve_cloth_collider(p0:Vec3,p1:Vec3,radius:f32,motion:ClothColliderMotion)->Option<(Vec3,Vec3,Vec3)> {
    if let Some(sdf)=motion.level_set(){return sdf.resolve(p0,p1,radius);}
    let hit=sweep_cloth_collider(p0,p1,radius,motion.clone())?;
    respond_to_collider(p0,p1,radius,motion,hit)
}

fn respond_to_collider(p0:Vec3,p1:Vec3,radius:f32,motion:ClothColliderMotion,hit:ClothSweep)->Option<(Vec3,Vec3,Vec3)> {
    let at=p0.lerp(p1,hit.time);let (_,normal,u)=shape_sample(at,motion.at(hit.time));
    let surface=|shape| match shape {
        ClothShapeWorld::Disabled | ClothShapeWorld::LevelSet(_) => at,
        ClothShapeWorld::Sphere{c,r}=>c+normal*(r+radius),
        ClothShapeWorld::Capsule{a,b,ra,rb}=>a.lerp(b,u)+normal*(ra+(rb-ra)*u+radius),
        ClothShapeWorld::Plane{p,n}=>at-n.normalize_or(Vec3::Y)*((at-p).dot(n.normalize_or(Vec3::Y))-radius),
    };
    let support=surface(motion.end.clone());let displacement=surface(motion.end.clone())-surface(motion.start.clone());
    let mut out=p1;
    let depth=(out-support).dot(normal);
    if depth<0.0{out-=normal*depth;}
    let (distance,end_normal)=cloth_shape_distance(out,motion.end);
    if distance<radius{out+=end_normal*(radius-distance);}
    if out.distance_squared(p1)<1e-18 && (p1-p0-displacement).dot(normal)>=0.0{return None;}
    Some((out,normal,displacement))
}

/// Swept contact of the whole segment, including its interior. The radius may
/// vary linearly along a bone chain. Convex distance along the segment is
/// minimized by its derivative; no fixed spatial samples can miss a thin body.
pub fn resolve_cloth_segment(start:[Vec3;2],end:[Vec3;2],radii:[f32;2],motion:ClothColliderMotion)->Option<(f32,Vec3,Vec3,Vec3)> {
    if start.iter().chain(&end).any(|p|!p.is_finite()){return None;}
    if let Some(sdf)=motion.level_set(){return sdf.resolve_segment(start,end,radii);}
    let radii=radii.map(|r|r.max(0.0));
    let sample=|time:f32|{
        let a=start[0].lerp(end[0],time);let b=start[1].lerp(end[1],time);let edge=b-a;let dr=radii[1]-radii[0];let shape=motion.at(time);
        let derivative=|u:f32|edge.dot(shape_sample(a+edge*u,shape.clone()).1)-dr;
        let u=if derivative(0.0)>=0.0{0.0}else if derivative(1.0)<=0.0{1.0}else{
            let (mut lo,mut hi)=(0.0,1.0);for _ in 0..32{let mid=(lo+hi)*0.5;if derivative(mid)<0.0{lo=mid}else{hi=mid}}
            (lo+hi)*0.5
        };
        let (distance,normal,_)=shape_sample(a+edge*u,shape.clone());
        (distance-radii[0]-dr*u,normal,u)
    };
    let speed=motion.speed_bound(start[0],end[0]).max(motion.speed_bound(start[1],end[1]));
    let (time,conservative)=advance(|t|sample(t).0,speed,0.0)?;
    let (_,normal,u)=sample(time);let p0=start[0].lerp(start[1],u);let p1=end[0].lerp(end[1],u);let radius=radii[0]+(radii[1]-radii[0])*u;
    let (out,normal,carried)=respond_to_collider(p0,p1,radius,motion,ClothSweep{time,normal,conservative})?;
    Some((u,out-p1,normal,carried))
}

/// Closest triangle point, including edges and degenerate triangles.
pub(crate) fn closest_triangle(p:Vec3,t:[Vec3;3])->(Vec3,[f32;3]) {
    let [a,b,c]=t;let ab=b-a;let ac=c-a;
    let denom=ab.length_squared()*ac.length_squared()-ab.dot(ac).powi(2);
    if denom>1e-20 {
        let ap=p-a;let u=(ap.dot(ab)*ac.length_squared()-ap.dot(ac)*ab.dot(ac))/denom;
        let v=(ap.dot(ac)*ab.length_squared()-ap.dot(ab)*ab.dot(ac))/denom;
        if u>=0.0&&v>=0.0&&u+v<=1.0{return(a+ab*u+ac*v,[1.0-u-v,u,v]);}
    }
    let mut best=(a,[1.0,0.0,0.0]);let mut distance=p.distance_squared(a);
    for (i,j) in [(0,1),(1,2),(2,0)]{let e=t[j]-t[i];let u=if e.length_squared()>1e-20{((p-t[i]).dot(e)/e.length_squared()).clamp(0.0,1.0)}else{0.0};let q=t[i]+e*u;let d=p.distance_squared(q);
        if d<distance{distance=d;let mut w=[0.0;3];w[i]=1.0-u;w[j]=u;best=(q,w);}
    }best
}
fn closest_edges(a:[Vec3;2],b:[Vec3;2])->(Vec3,Vec3,f32,f32){
    let u=a[1]-a[0];let v=b[1]-b[0];let r=a[0]-b[0];let aa=u.dot(u);let bb=v.dot(v);let uv=u.dot(v);let ur=u.dot(r);let vr=v.dot(r);
    let mut s;let mut t;
    if aa<1e-20{s=0.0;t=if bb<1e-20{0.0}else{(vr/bb).clamp(0.0,1.0)};}
    else if bb<1e-20{s=(-ur/aa).clamp(0.0,1.0);t=0.0;}
    else{let d=aa*bb-uv*uv;s=if d>1e-20{((uv*vr-ur*bb)/d).clamp(0.0,1.0)}else{0.0};t=(uv*s+vr)/bb;
        if t<0.0{t=0.0;s=(-ur/aa).clamp(0.0,1.0);}else if t>1.0{t=1.0;s=((uv-ur)/aa).clamp(0.0,1.0);}}
    (a[0]+u*s,b[0]+v*t,s,t)
}

pub fn sweep_cloth_vertex_face(start:[Vec3;4],end:[Vec3;4],thickness:f32)->Option<ClothFeatureContact>{
    let velocity: [Vec3;4]=std::array::from_fn(|i|end[i]-start[i]);
    let speed=(1..4).map(|i|(velocity[i]-velocity[0]).length()).fold(0.0,f32::max);
    let sample=|time|{let p: [Vec3;4]=std::array::from_fn(|i|start[i].lerp(end[i],time));let(q,w)=closest_triangle(p[0],[p[1],p[2],p[3]]);(p,p[0]-q,w)};
    let (time,conservative)=advance(|t|sample(t).1.length(),speed,thickness.max(1e-7))?;
    let (p,d,w)=sample(time);let normal=d.try_normalize().unwrap_or_else(||{
        let n=(p[2]-p[1]).cross(p[3]-p[1]).normalize_or(Vec3::Y);
        if (start[0]-start[1]).dot(n)<0.0{-n}else{n}
    });
    Some(ClothFeatureContact{sweep:ClothSweep{time,normal,conservative},coefficients:[1.0,-w[0],-w[1],-w[2]]})
}
pub fn sweep_cloth_edges(start:[Vec3;4],end:[Vec3;4],thickness:f32)->Option<ClothFeatureContact>{
    let velocity: [Vec3;4]=std::array::from_fn(|i|end[i]-start[i]);
    let speed=(0..2).flat_map(|i|(2..4).map(move |j|(velocity[i]-velocity[j]).length())).fold(0.0,f32::max);
    let sample=|time|{let p: [Vec3;4]=std::array::from_fn(|i|start[i].lerp(end[i],time));let(a,b,s,t)=closest_edges([p[0],p[1]],[p[2],p[3]]);(p,a-b,s,t)};
    let(time,conservative)=advance(|t|sample(t).1.length(),speed,thickness.max(1e-7))?;
    let(p,d,s,t)=sample(time);let normal=d.try_normalize().unwrap_or_else(||{
        let n=(p[1]-p[0]).cross(p[3]-p[2]).normalize_or((p[1]-p[0]).normalize_or(Vec3::X).any_orthonormal_vector());
        if (start[0].lerp(start[1],s)-start[2].lerp(start[3],t)).dot(n)<0.0{-n}else{n}
    });
    Some(ClothFeatureContact{sweep:ClothSweep{time,normal,conservative},coefficients:[1.0-s,s,-(1.0-t),-t]})
}

/// Mass-weighted contact projection. Fixed particles never move. The contact
/// normal is retained from impact so an end-of-step crossing cannot flip sides.
pub fn project_cloth_contact(positions:&mut[Vec3],inverse_mass:&[f32],ids:[usize;4],contact:ClothFeatureContact,thickness:f32)->f32 {
    let c=contact.coefficients;let n=contact.sweep.normal;
    let gap=(0..4).map(|k|positions[ids[k]]*c[k]).sum::<Vec3>().dot(n);
    let denom=(0..4).map(|k|inverse_mass[ids[k]].max(0.0)*c[k]*c[k]).sum::<f32>();
    if denom<1e-20||gap>=thickness{return 0.0;}
    let correction=(thickness-gap)/denom;
    for k in 0..4 {positions[ids[k]]+=n*(correction*inverse_mass[ids[k]].max(0.0)*c[k]);}
    thickness-gap
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ClothContactReport {pub vertex_face:usize,pub edge_edge:usize,pub untangled:usize,pub conservative:usize}

#[derive(Clone, Debug, Default)]
pub struct ClothContactTopology {
    pub triangles:Vec<[usize;3]>,
    pub edges:Vec<[usize;2]>,
    adjacent:std::collections::HashSet<(usize,usize)>,
    excluded:std::collections::HashSet<(usize,usize)>,
}
impl ClothContactTopology {
    pub fn new(triangles:Vec<[usize;3]>)->Self {
        let mut edges=std::collections::BTreeSet::new();
        for t in &triangles {for k in 0..3 {let(a,b)=(t[k],t[(k+1)%3]);if a!=b{edges.insert([a.min(b),a.max(b)]);}}}
        let adjacent=edges.iter().map(|e|(e[0],e[1])).collect();
        Self{triangles,edges:edges.into_iter().collect(),adjacent,excluded:Default::default()}
    }
    pub fn exclude_pair(&mut self,a:usize,b:usize){let pair=(a.min(b),a.max(b));self.adjacent.insert(pair);self.excluded.insert(pair);}
    fn neighbours(&self,a:usize,b:usize)->bool{a==b||self.adjacent.contains(&(a.min(b),a.max(b)))}
    /// Swept broad phase covers the complete interval. The same routine handles
    /// a combined mesh of several garments: no adjacency crosses their islands.
    pub fn solve(&self,positions:&mut[Vec3],previous:&[Vec3],inverse_mass:&[f32],thickness:f32,continuous:bool,untangle:bool)->ClothContactReport {
        self.solve_filtered(positions,previous,inverse_mass,thickness,continuous,untangle,None,None,None)
    }
    /// Self-contact on a visible mesh whose vertices share simulation drivers.
    /// Topological neighbours still use the original surface connectivity.
    pub fn solve_mapped(&self,positions:&mut[Vec3],previous:&[Vec3],inverse_mass:&[f32],thickness:f32,drivers:&ClothContactMapping,continuous:bool,untangle:bool)->ClothContactReport{
        if !drivers.valid_weights(inverse_mass){return ClothContactReport::default();}
        if drivers.trivial(){return self.solve(positions,previous,inverse_mass,thickness,continuous,untangle);}
        self.solve_filtered(positions,previous,inverse_mass,thickness,continuous,untangle,None,None,Some(drivers))
    }
    /// Solve only between distinct garments. Radii are half the desired cloth
    /// thickness; pair gaps add the two garments' radii rather than inflating
    /// every garment to the thickest member of the group.
    pub fn solve_between(&self,positions:&mut[Vec3],previous:&[Vec3],inverse_mass:&[f32],owners:&[usize],radii:&[f32],continuous:bool,untangle:bool)->ClothContactReport {
        if owners.len()!=positions.len()||radii.len()!=positions.len(){return ClothContactReport::default();}
        let thickness=radii.iter().copied().fold(0.0,f32::max)*2.0;
        self.solve_filtered(positions,previous,inverse_mass,thickness,continuous,untangle,Some(owners),Some(radii),None)
    }
    /// Contact the visible surface while maintaining its simulation drivers.
    /// Each vertex carries its driver's inverse mass, not a separate mass.
    pub fn solve_between_mapped(&self,positions:&mut[Vec3],previous:&[Vec3],inverse_mass:&[f32],owners:&[usize],radii:&[f32],drivers:&ClothContactMapping,continuous:bool,untangle:bool)->ClothContactReport{
        if !drivers.valid(inverse_mass,owners)||radii.len()!=positions.len(){return ClothContactReport::default();}
        if drivers.trivial(){return self.solve_between(positions,previous,inverse_mass,owners,radii,continuous,untangle);}
        let thickness=radii.iter().copied().fold(0.0,f32::max)*2.0;
        self.solve_filtered(positions,previous,inverse_mass,thickness,continuous,untangle,Some(owners),Some(radii),Some(drivers))
    }
    #[allow(clippy::too_many_arguments)]
    fn solve_filtered(&self,positions:&mut[Vec3],previous:&[Vec3],inverse_mass:&[f32],thickness:f32,continuous:bool,untangle:bool,owners:Option<&[usize]>,radii:Option<&[f32]>,mapping:Option<&ClothContactMapping>)->ClothContactReport {
        let mut report=ClothContactReport::default();
        if thickness<=0.0||!thickness.is_finite()||positions.len()!=previous.len()||positions.len()!=inverse_mass.len()
            || self.triangles.iter().flatten().chain(self.edges.iter().flatten()).any(|&i|i>=positions.len())
            || positions.iter().chain(previous).any(|p|!p.is_finite())||inverse_mass.iter().any(|w|!w.is_finite())
            || radii.is_some_and(|r|r.iter().any(|r|!r.is_finite()||*r<0.0)){return report;}
        let start=if continuous{previous.to_vec()}else{positions.to_vec()};
        let end=positions.to_vec();
        let point_bounds:Vec<_>=start.iter().zip(&end).map(|(a,b)|Bounds::points(&[*a,*b]).expanded(thickness)).collect();
        let face_bounds:Vec<_>=self.triangles.iter().map(|t|t.iter().map(|&i|point_bounds[i]).reduce(Bounds::union).unwrap()).collect();
        let face_tree=Tree::new(&face_bounds);
        let mut candidates=Vec::new();
        for i in 0..positions.len(){candidates.clear();face_tree.query(point_bounds[i],&mut candidates);
            for &ti in &candidates {let t=self.triangles[ti];if owners.is_some_and(|o|o[i]==o[t[0]])||t.iter().any(|&j|self.neighbours(i,j)){continue;}
                let ids=[i,t[0],t[1],t[2]];if ids.iter().all(|&j|inverse_mass[j]<=0.0){continue;}
                let gap=radii.map_or(thickness,|r|r[i]+t.iter().map(|&j|r[j]).fold(0.0,f32::max));
                if let Some(contact)=sweep_cloth_vertex_face(ids.map(|i|start[i]),ids.map(|i|end[i]),gap){
                    if match mapping{Some(m)=>m.project(positions,inverse_mass,ids,contact,gap),None=>project_cloth_contact(positions,inverse_mass,ids,contact,gap)}>0.0{report.vertex_face+=1;report.conservative+=usize::from(contact.sweep.conservative);}
                }
            }
        }
        let edge_bounds:Vec<_>=self.edges.iter().map(|e|point_bounds[e[0]].union(point_bounds[e[1]])).collect();
        let edge_tree=Tree::new(&edge_bounds);
        for (ei,&a) in self.edges.iter().enumerate(){candidates.clear();edge_tree.query(edge_bounds[ei],&mut candidates);
            for &ej in &candidates{if ej<=ei{continue;}let b=self.edges[ej];if owners.is_some_and(|o|o[a[0]]==o[b[0]])||a.iter().any(|&i|b.iter().any(|&j|self.neighbours(i,j))){continue;}
                let ids=[a[0],a[1],b[0],b[1]];if ids.iter().all(|&j|inverse_mass[j]<=0.0){continue;}
                let gap=radii.map_or(thickness,|r|r[a[0]].max(r[a[1]])+r[b[0]].max(r[b[1]]));
                if let Some(contact)=sweep_cloth_edges(ids.map(|i|start[i]),ids.map(|i|end[i]),gap){
                    if match mapping{Some(m)=>m.project(positions,inverse_mass,ids,contact,gap),None=>project_cloth_contact(positions,inverse_mass,ids,contact,gap)}>0.0{report.edge_edge+=1;report.conservative+=usize::from(contact.sweep.conservative);}
                }
            }
        }
        let before=mapping.map(|_|positions.to_vec());
        if untangle {report.untangled=self.untangle_filtered(positions,inverse_mass,thickness,owners,radii);}
        untangling::project_fixed(self,positions,inverse_mass,thickness,owners,radii);
        if let Some(m)=mapping{m.cohere(before.as_ref().unwrap(),positions);}
        let coherent_previous=mapping.map(|m|{let mut origin=previous.to_vec();m.cohere(positions,&mut origin);origin});
        if continuous&&untangling::preserve_valid_pose(self,positions,coherent_previous.as_deref().unwrap_or(previous),inverse_mass,owners){report.conservative+=1;}
        report
    }
    /// Resolve intersections already present at the start (for example an
    /// imported garment intersecting itself). Partition along intersected edges
    /// and choose a coherent movable side; fixed particles retain zero weight.
    pub fn untangle(&self,p:&mut[Vec3],weights:&[f32],thickness:f32)->usize {
        self.untangle_filtered(p,weights,thickness,None,None)
    }
    fn untangle_filtered(&self,p:&mut[Vec3],weights:&[f32],thickness:f32,owners:Option<&[usize]>,radii:Option<&[f32]>)->usize {
        untangling::recover(self,p,weights,thickness,owners,radii)
    }
}

#[derive(Clone, Copy, Debug)]
struct Bounds{lo:Vec3,hi:Vec3}
impl Bounds {
    fn points(p:&[Vec3])->Self{Self{lo:p.iter().copied().fold(Vec3::splat(f32::INFINITY),Vec3::min),hi:p.iter().copied().fold(Vec3::splat(f32::NEG_INFINITY),Vec3::max)}}
    fn union(self,b:Self)->Self{Self{lo:self.lo.min(b.lo),hi:self.hi.max(b.hi)}}
    fn expanded(self,r:f32)->Self{Self{lo:self.lo-Vec3::splat(r),hi:self.hi+Vec3::splat(r)}}
    fn overlaps(self,b:Self)->bool{self.lo.cmple(b.hi).all()&&b.lo.cmple(self.hi).all()}
}
struct Node{bounds:Bounds,children:Option<[usize;2]>,items:Vec<usize>}
struct Tree{nodes:Vec<Node>,items:Vec<Bounds>}
impl Tree {
    fn new(bounds:&[Bounds])->Self {let mut tree=Self{nodes:Vec::new(),items:bounds.to_vec()};if !bounds.is_empty(){tree.build(bounds,(0..bounds.len()).collect());}tree}
    fn build(&mut self,bounds:&[Bounds],mut items:Vec<usize>)->usize{
        let b=items.iter().map(|&i|bounds[i]).reduce(Bounds::union).unwrap();let index=self.nodes.len();
        self.nodes.push(Node{bounds:b,children:None,items:Vec::new()});
        if items.len()<=8{self.nodes[index].items=items;}else{
            let extent=b.hi-b.lo;let axis=if extent.x>=extent.y&&extent.x>=extent.z{0}else if extent.y>=extent.z{1}else{2};
            items.sort_unstable_by(|&i,&j|(bounds[i].lo[axis]+bounds[i].hi[axis]).total_cmp(&(bounds[j].lo[axis]+bounds[j].hi[axis])).then(i.cmp(&j)));
            let right=items.split_off(items.len()/2);let a=self.build(bounds,items);let b=self.build(bounds,right);self.nodes[index].children=Some([a,b]);
        }index
    }
    fn query(&self,bounds:Bounds,out:&mut Vec<usize>){if !self.nodes.is_empty(){self.query_node(0,bounds,out);}}
    fn query_node(&self,index:usize,bounds:Bounds,out:&mut Vec<usize>){
        let node=&self.nodes[index];if !node.bounds.overlaps(bounds){return;}
        if let Some([a,b])=node.children{self.query_node(a,bounds,out);self.query_node(b,bounds,out);}
        else{out.extend(node.items.iter().copied().filter(|&i|self.items[i].overlaps(bounds)));}
    }
}

#[cfg(test)]
#[path="cloth_contact_tests.rs"]
mod tests;
