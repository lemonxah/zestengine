//! Portable cloth panels and sewing definitions. No renderer or file I/O.
use glam::{Vec2,Vec3,Quat,EulerRot};
use serde::{Serialize,Deserialize};
#[path="cloth_pattern_edit.rs"]
mod edit;
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize,Default)]
pub enum ClothPanelSide{#[default]Left,Right,Bottom,Top}
impl ClothPanelSide{
    pub const ALL:[Self;4]=[Self::Left,Self::Right,Self::Bottom,Self::Top];
    pub fn label(self)->&'static str{match self{Self::Left=>"Left",Self::Right=>"Right",Self::Bottom=>"Bottom",Self::Top=>"Top"}}
}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default)]
pub struct ClothPanel{
    pub name:String,
    pub size:[f32;2],
    pub segments:[u32;2],
    pub translation:[f32;3],
    pub rotation:[f32;3],
    /// Optional custom 2D panel, in metres. Empty uses a rectangular grid.
    pub vertices:Vec<[f32;2]>,
    pub triangles:Vec<u32>,
}
impl Default for ClothPanel{fn default()->Self{Self{name:"Panel".into(),size:[1.0,1.0],segments:[12,12],translation:[0.0;3],rotation:[0.0;3],vertices:Vec::new(),triangles:Vec::new()}}}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ClothPanelEdge{pub panel:usize,pub side:ClothPanelSide,/// Explicit ordered boundary vertices for a custom panel.
pub vertices:Vec<u32>}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default)]
pub struct ClothSeam{pub a:ClothPanelEdge,pub b:ClothPanelEdge,pub reverse:bool,/// XPBD compliance in m/N. Zero = hard seam.
pub compliance:f32}
impl Default for ClothSeam{fn default()->Self{Self{a:Default::default(),b:ClothPanelEdge{panel:1,..Default::default()},reverse:false,compliance:0.0}}}
#[derive(Clone,Debug,Serialize,Deserialize,Default)]
#[serde(default)]
pub struct ClothPattern{pub panels:Vec<ClothPanel>,pub seams:Vec<ClothSeam>}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub struct ClothStitch{pub vertex:u32,pub edge:[u32;2],pub weight:f32,pub compliance:f32}
pub struct AssembledCloth{pub positions:Vec<Vec3>,pub uv:Vec<Vec2>,pub indices:Vec<u32>,pub stitches:Vec<ClothStitch>}
impl ClothPanel{
    pub fn mesh(&self)->Result<(Vec<Vec2>,Vec<u32>),String>{
        if self.translation.iter().chain(&self.rotation).any(|v|!v.is_finite()){return Err("Non-finite panel transform".into());}
        if !self.vertices.is_empty(){
            if self.vertices.len()>250_000||self.triangles.len()>1_500_000||self.vertices.iter().flatten().any(|v|!v.is_finite())||!self.triangles.len().is_multiple_of(3)||self.triangles.len()<3||self.triangles.iter().any(|&v|v as usize>=self.vertices.len()){return Err("Invalid custom panel geometry".into());}
            return Ok((self.vertices.iter().copied().map(Vec2::from).collect(),self.triangles.clone()));
        }
        let [x,y]=self.segments;if x==0||y==0||x>256||y>256||self.size.iter().any(|v|!v.is_finite()||*v<=0.0){return Err("Panel size must be positive; subdivisions must be 1–256".into());}
        let mut p=Vec::new();let mut t=Vec::new();
        for j in 0..=y{for i in 0..=x{p.push(Vec2::new(i as f32/x as f32*self.size[0],j as f32/y as f32*self.size[1]));}}
        for j in 0..y{for i in 0..x{let a=j*(x+1)+i;t.extend([a,a+1,a+x+1,a+1,a+x+2,a+x+1]);}}
        Ok((p,t))
    }
    fn edge(&self,edge:&ClothPanelEdge,vertices:usize,triangles:&[u32])->Result<Vec<u32>,String>{
        let chain=if !edge.vertices.is_empty(){edge.vertices.clone()}else{
            if !self.vertices.is_empty(){return Err("Custom panels need an ordered seam vertex chain".into());}
            let [x,y]=self.segments;match edge.side{ClothPanelSide::Left=>(0..=y).map(|j|j*(x+1)).collect(),ClothPanelSide::Right=>(0..=y).map(|j|j*(x+1)+x).collect(),ClothPanelSide::Bottom=>(0..=x).collect(),ClothPanelSide::Top=>(0..=x).map(|i|y*(x+1)+i).collect()}
        };
        if chain.len()<2||chain.iter().any(|&i|i as usize>=vertices){return Err("Invalid seam edge vertices".into());}
        let mut counts=std::collections::HashMap::new();for t in triangles.as_chunks::<3>().0{for i in 0..3{let(a,b)=(t[i],t[(i+1)%3]);*counts.entry((a.min(b),a.max(b))).or_insert(0)+=1;}}
        if chain.windows(2).any(|e|counts.get(&(e[0].min(e[1]),e[0].max(e[1])))!=Some(&1)){return Err("Seams must follow consecutive boundary edges".into());}
        Ok(chain)
    }
}
impl ClothPattern{
    pub fn assemble(&self)->Result<AssembledCloth,String>{
        if self.panels.is_empty()||self.panels.len()>128||self.seams.len()>512{return Err("Clothing needs 1–128 panels and at most 512 seams".into());}
        let mut out=AssembledCloth{positions:Vec::new(),uv:Vec::new(),indices:Vec::new(),stitches:Vec::new()};let mut meshes=Vec::new();let mut starts=Vec::new();
        for panel in &self.panels{
            let(p,t)=panel.mesh()?;starts.push(out.positions.len() as u32);let offset=out.positions.len() as u32;
            let r=panel.rotation.map(f32::to_radians);let rotation=Quat::from_euler(EulerRot::XYZ,r[0],r[1],r[2]);let translation=Vec3::from(panel.translation);
            out.positions.extend(p.iter().map(|v|rotation*v.extend(0.0)+translation));out.uv.extend(p.iter().copied());out.indices.extend(t.iter().map(|&i|i+offset));meshes.push((p,t));
            if out.positions.len()>250_000{return Err("Clothing exceeds 250,000 vertices".into());}
        }
        for seam in &self.seams{
            if !seam.compliance.is_finite()||seam.compliance<0.0{return Err("Seam compliance must be finite and nonnegative".into());}
            let mut chains=Vec::new();for edge in [&seam.a,&seam.b]{let panel=self.panels.get(edge.panel).ok_or("Missing seam panel")?;let(p,t)=&meshes[edge.panel];chains.push(panel.edge(edge,p.len(),t)?.iter().map(|v|v+starts[edge.panel]).collect::<Vec<_>>());}
            if seam.reverse{chains[1].reverse();}
            if chains[0].iter().any(|v|chains[1].contains(v)){return Err("A seam cannot stitch an edge to itself".into());}
            for side in 0..2{
                let a=&chains[side];let b=&chains[1-side];
                let distances=|chain:&[u32]|{let mut total=vec![0.0];for e in chain.windows(2){total.push(total.last().unwrap()+out.positions[e[0]as usize].distance(out.positions[e[1]as usize]));}total};
                let da=distances(a);let db=distances(b);let la=*da.last().unwrap();let lb=*db.last().unwrap();if la<1e-6||lb<1e-6{return Err("Zero-length seam".into());}
                for (i,&vertex) in a.iter().enumerate(){let distance=da[i]/la*lb;let j=db.partition_point(|&d|d<distance).saturating_sub(1).min(b.len()-2);let span=db[j+1]-db[j];if span<1e-8{return Err("Zero-length seam segment".into());}
                    out.stitches.push(ClothStitch{vertex,edge:[b[j],b[j+1]],weight:((distance-db[j])/span).clamp(0.0,1.0),compliance:seam.compliance});
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{MeshCloth,MeshClothSim,ClothFabric};use glam::Mat4;
    fn jacket()->ClothPattern{ClothPattern{panels:vec![ClothPanel{segments:[3,4],..Default::default()},ClothPanel{segments:[3,7],translation:[1.2,0.0,0.0],..Default::default()}],seams:vec![ClothSeam{a:ClothPanelEdge{side:ClothPanelSide::Right,..Default::default()},b:ClothPanelEdge{panel:1,side:ClothPanelSide::Left,..Default::default()},..Default::default()}]}}
    #[test]
    fn cloth_panels_with_unequal_edges_sew_and_keep_pins(){
        let pattern=jacket();let g=pattern.assemble().unwrap();assert_eq!(g.stitches.len(),13);
        let pins:Vec<_>=(0..g.positions.len()).map(|i|if i<20{1.0}else{0.0}).collect();
        let cloth=MeshCloth{gravity:[0.0;3],iterations:12,stitches:g.stitches.clone(),self_collision:true,fabric:Some(ClothFabric{drag:0.0,lift:0.0,..Default::default()}),..Default::default()};
        for spacing in [0.0,0.3]{
            let mut sim=MeshClothSim::build(&g.positions,&g.indices,&pins,spacing,Mat4::IDENTITY).unwrap();sim.set_texcoords(&g.uv);sim.set_stitches(&g.stitches).unwrap();
            for _ in 0..240{sim.step(&cloth,&g.positions,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
            let mut rendered=g.positions.clone();let mut normals=vec![Vec3::Z;rendered.len()];sim.apply(&mut rendered,&mut normals,Mat4::IDENTITY);
            for (p,want) in rendered[..20].iter().zip(&g.positions){assert!(p.distance(*want)<1e-5);}
            let error=g.stitches.iter().map(|s|rendered[s.vertex as usize].distance(rendered[s.edge[0]as usize].lerp(rendered[s.edge[1]as usize],s.weight))).fold(0.0f32,f32::max);
            eprintln!("cloth seam spacing {spacing}: max render separation {error}");assert!(error<0.01,"spacing {spacing}, seam error {error}");assert!(sim.particles.iter().all(|p|p.pos.is_finite()));
        }
    }
    #[test]
    fn cloth_panel_validation_rejects_bad_topology_and_nonboundary_seams(){
        let mut p=jacket();p.panels[0].segments=[0,3];assert!(p.assemble().is_err());p=jacket();p.seams[0].a.vertices=vec![0,2];assert!(p.assemble().is_err());p=jacket();p.seams[0].b.panel=0;p.seams[0].b.side=ClothPanelSide::Right;assert!(p.assemble().is_err());
        p=jacket();p.panels[0].vertices=vec![[0.0,0.0],[1.0,0.0],[0.0,1.0]];p.panels[0].triangles=vec![0,1,2];p.seams[0].a.vertices=vec![0,2];assert!(p.assemble().is_ok());p.panels[0].triangles[2]=9;assert!(p.assemble().is_err());
    }
    #[test]
    fn cloth_seams_keep_their_shape_through_rotated_teleports(){
        let g=jacket().assemble().unwrap();let pins:Vec<_>=(0..g.positions.len()).map(|i|if i<20{1.0}else{0.0}).collect();
        let settings=MeshCloth{gravity:[0.0;3],iterations:16,teleport:crate::ClothTeleport::Keep,stitches:g.stitches.clone(),fabric:Some(ClothFabric{drag:0.0,lift:0.0,..Default::default()}),..Default::default()};
        let mut a=MeshClothSim::build(&g.positions,&g.indices,&pins,0.3,Mat4::IDENTITY).unwrap();a.set_texcoords(&g.uv);a.set_stitches(&g.stitches).unwrap();
        for _ in 0..240{a.step(&settings,&g.positions,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
        let mut b=a.clone();let world=Mat4::from_rotation_translation(Quat::from_rotation_z(2.0),Vec3::new(7.0,3.0,1.0));
        a.step(&settings,&g.positions,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);b.step(&settings,&g.positions,world,&[],&[],0.0,1.0/60.0);
        for (a,b) in a.particles.iter().zip(&b.particles){
            assert!(world.transform_point3(a.pos).distance(b.pos)<0.0001,"seams must not pull toward the old orientation");
            assert!(world.transform_vector3(a.vel).distance(b.vel)<0.01,"teleport must not inject seam motion");
        }
        b.set_stitches(&[]).unwrap();assert!(b.particles.iter().all(|p|p.pos.is_finite()));
    }
}
