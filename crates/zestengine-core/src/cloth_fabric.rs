//! Independent XPBD fabric model. Constraints accumulate multipliers within a
//! substep (Macklin et al., 2016); material rest metrics never follow animation.
//! UV U selects warp, its in-surface perpendicular selects weft. Missing or
//! degenerate UVs use the first triangle edge instead.
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use super::MeshParticle;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClothFabric {
    /// Mass per undeformed square metre (kg/m²).
    pub density: f32,
    /// In-plane elastic stiffness in N/m (UV U, perpendicular, and shear).
    pub warp: f32,
    pub weft: f32,
    pub shear: f32,
    /// Angular fold stiffness (N m/rad). Zero disables bending.
    pub bend: f32,
    /// Multiplier on fold stiffness after buckling_angle degrees of folding.
    pub buckling: f32,
    pub buckling_angle: f32,
    /// Follow animation with an implicit spring (Hz); zero disables the drive.
    pub animation_hz: f32,
    /// Damping ratio of the animation drive; one is critically damped.
    pub animation_damping: f32,
    pub drag: f32,
    pub lift: f32,
    /// Signed normal pressure (Pa); does not require a Wind Zone.
    pub pressure: f32,
    /// Animated normal backstop sphere (m); zero radius disables it.
    pub backstop_radius: f32,
    /// Gap allowed behind the animated surface (m).
    pub backstop_distance: f32,
    /// Expensive self-contact passes per substep, distributed across the solver
    /// with one always at the end. Independent of material solver iterations.
    pub self_collision_passes: u32,
}
impl Default for ClothFabric {
    fn default()->Self { Self {density:0.2,warp:1500.0,weft:1000.0,shear:500.0,bend:0.02,
        buckling:0.3,buckling_angle:45.0,animation_hz:0.0,animation_damping:1.0,
        drag:1.0,lift:0.3,pressure:0.0,backstop_radius:0.0,backstop_distance:0.0,self_collision_passes:1} }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClothFabricPreset { Silk, Cotton, Denim, Leather }
impl ClothFabricPreset {
    pub const ALL:[Self;4]=[Self::Silk,Self::Cotton,Self::Denim,Self::Leather];
    pub fn label(self)->&'static str {match self {Self::Silk=>"Silk",Self::Cotton=>"Cotton",Self::Denim=>"Denim",Self::Leather=>"Leather"}}
    /// Art starting points, not measured material samples.
    pub fn settings(self)->ClothFabric {match self {
        Self::Silk=>ClothFabric {density:0.07,warp:800.0,weft:600.0,shear:80.0,bend:0.002,..Default::default()},
        Self::Cotton=>ClothFabric::default(),
        Self::Denim=>ClothFabric {density:0.4,warp:5000.0,weft:3000.0,shear:1200.0,bend:0.08,..Default::default()},
        Self::Leather=>ClothFabric {density:0.8,warp:8000.0,weft:7000.0,shear:3000.0,bend:0.3,buckling:0.7,..Default::default()},
    }}
}

#[derive(Clone, Debug)]
struct Triangle { ids:[usize;3], warp:Vec3, u:[f32;3], v:[f32;3], area:f32, lambda:[f32;3], material:[f32;3] }
#[derive(Clone, Debug)]
struct Hinge {ids:[usize;4], rest:f32, lambda:f32, material:f32}
#[derive(Clone, Debug, Default)]
pub(super) struct FabricState {
    triangles:Vec<Triangle>, hinges:Vec<Hinge>,
    pub weights:Vec<f32>,
    /// Mesh-edge geodesic length; disconnected islands cannot borrow pins.
    pub tethers:Vec<(usize,usize,f32)>,
    pub normals:Vec<Vec3>,
    contacts:crate::cloth_contact::ClothContactTopology,
}
impl FabricState {
    pub fn contact_triangles(&self)->Vec<[usize;3]>{self.triangles.iter().map(|t|t.ids).collect()}
    pub fn build(rest:&[Vec3],owner:&[u32],indices:&[u32],uv:&[Vec2],edges:&[(u32,u32)],pins:&[f32])->Self {
        let mut out=Self::default();
        let mut adjacent:HashMap<(usize,usize),Vec<usize>>=HashMap::new();
        let mut seen=std::collections::HashSet::new();
        for tri in indices.as_chunks::<3>().0 {
            let ids=[owner[tri[0]as usize]as usize,owner[tri[1]as usize]as usize,owner[tri[2]as usize]as usize];
            if ids[0]==ids[1]||ids[1]==ids[2]||ids[0]==ids[2]{continue;}
            let mut key=ids;key.sort_unstable();if !seen.insert(key){continue;}
            let e=rest[ids[1]]-rest[ids[0]];let f=rest[ids[2]]-rest[ids[0]];
            if e.cross(f).length_squared()<1e-16{continue;}
            let mut warp=e.normalize();
            if uv.len()==owner.len(){let a=uv[tri[1]as usize]-uv[tri[0]as usize];let b=uv[tri[2]as usize]-uv[tri[0]as usize];let det=a.perp_dot(b);
                if det.is_finite()&&det.abs()>1e-10 {let u=(e*b.y-f*a.y)/det;if u.is_finite()&&u.length_squared()>1e-16{warp=u.normalize();}}
            }
            out.triangles.push(Triangle{ids,warp,u:[0.0;3],v:[0.0;3],area:0.0,lambda:[0.0;3],material:[1.0;3]});
            for k in 0..3{let(a,b)=(ids[k],ids[(k+1)%3]);adjacent.entry((a.min(b),a.max(b))).or_default().push(ids[(k+2)%3]);}
        }
        let mut keys:Vec<_>=adjacent.keys().copied().collect();keys.sort_unstable();
        for (a,b) in keys {let opposite=&adjacent[&(a,b)];if opposite.len()==2{out.hinges.push(Hinge {ids:[a,b,opposite[0],opposite[1]],rest:0.0,lambda:0.0,material:1.0});}}
        out.contacts=crate::cloth_contact::ClothContactTopology::new(out.triangles.iter().map(|t|t.ids).collect());
        out.rebuild_tethers(rest,edges,pins);out
    }
    pub fn exclude_seams(&mut self,pairs:&std::collections::HashSet<(u32,u32)>){self.contacts=crate::cloth_contact::ClothContactTopology::new(self.contact_triangles());for &(a,b) in pairs{self.contacts.exclude_pair(a as usize,b as usize);}}
    pub fn set_material_masks(&mut self,masks:&[[f32;7]]){
        for t in &mut self.triangles{for channel in 0..3{t.material[channel]=t.ids.iter().map(|&i|masks.get(i).map_or(1.0,|m|m[channel])).sum::<f32>()/3.0;}}
        for h in &mut self.hinges{h.material=h.ids.iter().map(|&i|masks.get(i).map_or(1.0,|m|m[3])).sum::<f32>()/4.0;}
    }
    pub fn rebuild_tethers(&mut self,rest:&[Vec3],edges:&[(u32,u32)],pins:&[f32]) {
        let mut links=vec![Vec::new();rest.len()];
        for &(a,b) in edges{let(a,b)=(a as usize,b as usize);let d=rest[a].distance(rest[b]);links[a].push((b,d));links[b].push((a,d));}
        let mut distance=vec![f32::INFINITY;rest.len()];let mut anchor=vec![usize::MAX;rest.len()];let mut queue=BinaryHeap::new();
        for i in 0..rest.len(){if pins[i]>=0.999{distance[i]=0.0;anchor[i]=i;queue.push(Reverse((0u32,i,i)));}}
        while let Some(Reverse((bits,root,i)))=queue.pop(){let d=f32::from_bits(bits);if d!=distance[i]||root!=anchor[i]{continue;}
            for &(j,length) in &links[i]{let next=d+length;if next<distance[j]||(next==distance[j]&&root<anchor[j]){distance[j]=next;anchor[j]=root;queue.push(Reverse((next.to_bits(),root,j)));}}
        }
        self.tethers=(0..rest.len()).filter(|&i|pins[i]<0.999&&anchor[i]!=usize::MAX).map(|i|(i,anchor[i],distance[i])).collect();
    }
    pub fn prepare(&mut self,rest:&[Vec3],world:Mat4,p:&[MeshParticle],cfg:&ClothFabric) {
        let rest:Vec<_>=rest.iter().map(|x|world.transform_point3(*x)).collect();
        self.weights=vec![0.0;p.len()];self.normals=vec![Vec3::ZERO;p.len()];
        for t in &mut self.triangles {let [a,b,c]=t.ids;let e=rest[b]-rest[a];let f=rest[c]-rest[a];let cross=e.cross(f);t.area=cross.length()*0.5;
            let u=world.transform_vector3(t.warp).normalize_or_zero();let v=cross.normalize_or_zero().cross(u).normalize_or_zero();
            let det=e.dot(u)*f.dot(v)-f.dot(u)*e.dot(v);
            if det.abs()<1e-12{t.u=[0.0;3];t.v=[0.0;3];continue;}
            let (ub,uc,vb,vc)=(f.dot(v)/det,-e.dot(v)/det,-f.dot(u)/det,e.dot(u)/det);
            t.u=[-ub-uc,ub,uc];t.v=[-vb-vc,vb,vc];
            for i in t.ids{self.weights[i]+=t.area/3.0;}
        }
        for (i,w) in self.weights.iter_mut().enumerate(){*w=if *w>1e-10 {(1.0-p[i].pin.clamp(0.0,1.0))/(*w*cfg.density.clamp(0.001,100.0))}else{0.0};}
        for hinge in &mut self.hinges{hinge.rest=dihedral(hinge.ids.map(|i|rest[i])).map_or(0.0,|(a,_)|a);}
    }
    pub fn begin_step(&mut self,target:&[Vec3]) {
        for t in &mut self.triangles{t.lambda=[0.0;3];}
        for h in &mut self.hinges{h.lambda=0.0;}
        self.normals.fill(Vec3::ZERO);
        for t in &self.triangles {let[a,b,c]=t.ids;let n=(target[b]-target[a]).cross(target[c]-target[a]);for i in t.ids{self.normals[i]+=n;}}
        for n in &mut self.normals{*n=n.normalize_or_zero();}
    }
    pub fn solve(&mut self,p:&mut [MeshParticle],cfg:&ClothFabric,h:f32) {
        for t in &mut self.triangles {if t.area<=1e-10{continue;}
            for (channel,stiffness) in [cfg.warp,cfg.weft,cfg.shear].into_iter().enumerate(){let stiffness=stiffness*t.material[channel];if stiffness<=0.0{continue;}
                let u=(0..3).map(|k|p[t.ids[k]].pos*t.u[k]).sum::<Vec3>();let v=(0..3).map(|k|p[t.ids[k]].pos*t.v[k]).sum::<Vec3>();
                let (c,g)=match channel {0=>(u.length()-1.0,t.u.map(|w|u.normalize_or_zero()*w)),1=>(v.length()-1.0,t.v.map(|w|v.normalize_or_zero()*w)),_=>(u.dot(v),std::array::from_fn(|k|v*t.u[k]+u*t.v[k]))};
                project(p,&self.weights,t.ids,g,c,1.0/(stiffness.max(1e-6)*t.area*h*h),&mut t.lambda[channel]);
            }
        }
        if cfg.bend>0.0 {for hinge in &mut self.hinges {
            if hinge.material<=0.0{continue;}
            let Some((angle,g))=dihedral(hinge.ids.map(|i|p[i].pos))else{continue};
            let c=wrap_angle(angle-hinge.rest);
            let strength=if c.abs()>cfg.buckling_angle.clamp(0.0,180.0).to_radians(){cfg.buckling.clamp(0.001,1.0)}else{1.0};
            project(p,&self.weights,hinge.ids,g,c,1.0/(cfg.bend*hinge.material*strength*h*h),&mut hinge.lambda);
        }}
    }
    pub fn forces(&self,p:&[MeshParticle],cfg:&ClothFabric,winds:&[crate::cloth::WindSample],wind:f32,time:f32)->Vec<Vec3> {
        let mut force=vec![Vec3::ZERO;p.len()];
        for t in &self.triangles {let[a,b,c]=t.ids;let center=(p[a].pos+p[b].pos+p[c].pos)/3.0;let cross=(p[b].pos-p[a].pos).cross(p[c].pos-p[a].pos);let area=cross.length()*0.5;
            if area<1e-10{continue;}let n=cross.normalize();let velocity=(p[a].vel+p[b].vel+p[c].vel)/3.0;
            let air=winds.iter().map(|w|w.at(center,time)).sum::<Vec3>()*wind.max(0.0);let relative=(air-velocity).clamp_length_max(100.0);let dir=relative.normalize_or_zero();let cosine=n.dot(dir);
            // Drag follows relative flow; lift is perpendicular to it. Both
            // sides of the sheet receive the same force under reversed winding.
            let q=0.5*1.225*relative.length_squared()*area;
            let f=q*(cfg.drag.clamp(0.0,10.0)*cosine.abs()*dir+cfg.lift.clamp(0.0,10.0)*cosine*(n-dir*cosine))+n*(cfg.pressure.clamp(-10000.0,10000.0)*area);
            for i in t.ids{force[i]+=f/3.0;}
        }
        force
    }
    pub fn recover_intersections(&self,p:&mut[MeshParticle],thickness:f32){
        let mut positions:Vec<_>=p.iter().map(|p|p.pos).collect();
        for _ in 0..4{if self.contacts.untangle(&mut positions,&self.weights,thickness)==0{break;}}
        for (p,position) in p.iter_mut().zip(positions){p.pos=position;}
    }
    #[cfg(test)]
    pub fn self_collide(&self,p:&mut[MeshParticle],previous:&[Vec3],thickness:f32){
        self.self_collide_swept(p,previous,thickness,true,true);
    }
    pub fn self_collide_swept(&self,p:&mut[MeshParticle],previous:&[Vec3],thickness:f32,continuous:bool,untangle:bool){
        let mut positions:Vec<_>=p.iter().map(|p|p.pos).collect();
        self.contacts.solve(&mut positions,previous,&self.weights,thickness,continuous,untangle);
        for (p,position) in p.iter_mut().zip(positions){p.pos=position;}
    }
}

fn project<const N:usize>(p:&mut[MeshParticle],w:&[f32],ids:[usize;N],g:[Vec3;N],c:f32,alpha:f32,lambda:&mut f32){
    let denom=alpha+(0..N).map(|k|w[ids[k]]*g[k].length_squared()).sum::<f32>();
    if denom<=1e-12||!denom.is_finite()||!c.is_finite(){return;}
    let delta=(-c-alpha* *lambda)/denom;*lambda+=delta;
    for k in 0..N{p[ids[k]].pos+=g[k]*(w[ids[k]]*delta);}
}
fn wrap_angle(a:f32)->f32{(a+std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)-std::f32::consts::PI}
/// Signed fold angle and its analytic derivatives. The two faces use opposite
/// edge directions, so a flat sheet has angle zero.
fn dihedral(p:[Vec3;4])->Option<(f32,[Vec3;4])>{
    let [a,b,c,d]=p;let e=b-a;let l=e.length();let n0=e.cross(c-a);let n1=(d-a).cross(e);
    if l<1e-8||n0.length_squared()<1e-16||n1.length_squared()<1e-16{return None;}
    let u=n0.normalize();let v=n1.normalize();let angle=(e/l).dot(u.cross(v)).atan2(u.dot(v));
    let gc=-n0*(l/n0.length_squared());let gd=-n1*(l/n1.length_squared());
    let gb=-gc*((c-a).dot(e)/(l*l))-gd*((d-a).dot(e)/(l*l));
    Some((angle,[-gb-gc-gd,gb,gc,gd]))
}

#[cfg(test)]
#[path = "cloth_fabric_tests.rs"]
mod tests;
