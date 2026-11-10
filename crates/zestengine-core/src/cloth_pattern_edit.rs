//! Topology-preserving panel edits and triangulation for new outline panels.
use super::*;
use std::collections::BTreeMap;

fn orient(a:glam::DVec2,b:glam::DVec2,c:glam::DVec2)->f64{(b-a).perp_dot(c-a)}
impl ClothPanel{
    /// Create a simple, possibly concave panel. Subdivision shares edge
    /// midpoints, so the resulting mesh has no cracks or duplicate seam edges.
    pub fn from_outline(outline:&[[f32;2]],subdivisions:u32)->Result<Self,String>{
        if !(3..=512).contains(&outline.len())||subdivisions>6||outline.iter().flatten().any(|v|!v.is_finite()){
            return Err("An outline needs 3–512 finite points and 0–6 subdivisions".into());
        }
        let mut p:Vec<_>=outline.iter().map(|&p|Vec2::from(p).as_dvec2()).collect();
        let extent=p.iter().copied().fold(glam::DVec2::splat(f64::NEG_INFINITY),glam::DVec2::max)-p.iter().copied().fold(glam::DVec2::splat(f64::INFINITY),glam::DVec2::min);
        let epsilon=extent.length_squared()*1e-12;
        if epsilon<=1e-24{return Err("The outline has no area".into());}
        // Closing the outline with a repeated first point is harmless.
        if p[0].distance_squared(*p.last().unwrap())<=epsilon{p.pop();}
        if p.len()<3||p.iter().enumerate().any(|(i,p0)|p0.distance_squared(p[(i+1)%p.len()])<=epsilon){return Err("Outline points must be distinct".into());}
        for i in 0..p.len(){for j in i+1..p.len(){
            if j==i+1||(i==0&&j+1==p.len()){continue;}
            let(a,b,c,d)=(p[i],p[(i+1)%p.len()],p[j],p[(j+1)%p.len()]);
            if a.min(b).cmple(c.max(d)+glam::DVec2::splat(epsilon.sqrt())).all()&&c.min(d).cmple(a.max(b)+glam::DVec2::splat(epsilon.sqrt())).all()
                &&orient(a,b,c)*orient(a,b,d)<=epsilon*epsilon&&orient(c,d,a)*orient(c,d,b)<=epsilon*epsilon{
                return Err("The outline crosses or touches itself".into());
            }
        }}
        // Collinear clicks do not need their own simulation vertex; generated
        // boundary subdivision supplies the requested edge density instead.
        loop{let redundant=(0..p.len()).find(|&i|orient(p[(i+p.len()-1)%p.len()],p[i],p[(i+1)%p.len()]).abs()<=epsilon);
            if let Some(i)=redundant{if p.len()<=3{return Err("The outline has no area".into());}p.remove(i);}else{break;}
        }
        let area=(0..p.len()).map(|i|p[i].perp_dot(p[(i+1)%p.len()])).sum::<f64>();
        if area.abs()<=epsilon{return Err("The outline has no area".into());}if area<0.0{p.reverse();}
        if (p.len()-2)*4usize.pow(subdivisions)>500_000{return Err("The subdivided panel exceeds 500,000 triangles".into());}
        let mut polygon=(0..p.len()).collect::<Vec<_>>();let mut triangles=Vec::new();
        while polygon.len()>3{
            let ear=(0..polygon.len()).find(|&i|{
                let ids=[polygon[(i+polygon.len()-1)%polygon.len()],polygon[i],polygon[(i+1)%polygon.len()]];let[a,b,c]=ids.map(|i|p[i]);
                orient(a,b,c)>epsilon&&!polygon.iter().any(|j|!ids.contains(j)&&orient(a,b,p[*j])>=-epsilon&&orient(b,c,p[*j])>=-epsilon&&orient(c,a,p[*j])>=-epsilon)
            }).ok_or("The outline cannot be triangulated; remove overlapping or very thin edges")?;
            triangles.extend([polygon[(ear+polygon.len()-1)%polygon.len()]as u32,polygon[ear]as u32,polygon[(ear+1)%polygon.len()]as u32]);polygon.remove(ear);
        }
        triangles.extend(polygon.into_iter().map(|i|i as u32));let mut vertices=p.into_iter().map(|p|p.as_vec2().to_array()).collect::<Vec<_>>();
        for _ in 0..subdivisions{
            let mut midpoints=BTreeMap::new();let mut refined=Vec::with_capacity(triangles.len()*4);
            for t in triangles.chunks_exact(3){let mut edge=[0;3];for i in 0..3{let(a,b)=(t[i],t[(i+1)%3]);let key=(a.min(b),a.max(b));
                edge[i]=*midpoints.entry(key).or_insert_with(||{let i=vertices.len()as u32;vertices.push(((Vec2::from(vertices[a as usize])+Vec2::from(vertices[b as usize]))*0.5).to_array());i});
            }
            let[a,b,c]=[t[0],t[1],t[2]];let[ab,bc,ca]=edge;refined.extend([a,ab,ca,ab,b,bc,ca,bc,c,ab,bc,ca]);}
            if vertices.len()>250_000{return Err("The subdivided panel exceeds 250,000 vertices".into());}triangles=refined;
        }
        Ok(Self{name:"Outline panel".into(),vertices,triangles,..Default::default()})
    }
    /// Move one custom vertex without changing its identity, painted masks or
    /// seam references. Reject collapsed/inverted incident triangles atomically.
    pub fn move_vertex(&mut self,index:usize,position:[f32;2])->Result<(),String>{
        if index>=self.vertices.len()||position.iter().any(|v|!v.is_finite()){return Err("Choose a valid vertex and finite position".into());}
        let target=Vec2::from(position).as_dvec2();
        for t in self.triangles.chunks_exact(3).filter(|t|t.contains(&(index as u32))){
            let mut p=[glam::DVec2::ZERO;3];for k in 0..3{p[k]=Vec2::from(*self.vertices.get(t[k]as usize).ok_or("Invalid panel triangle")?).as_dvec2();}
            let old=orient(p[0],p[1],p[2]);let extent=(p[1]-p[0]).length_squared().max((p[2]-p[0]).length_squared());
            for k in 0..3{if t[k]as usize==index{p[k]=target;}}
            let new=orient(p[0],p[1],p[2]);if old*new<=0.0||new.abs()<=extent*1e-8{return Err("That move would collapse or invert a triangle".into());}
        }
        self.vertices[index]=position;Ok(())
    }
    pub fn boundary_loops(&self)->Result<Vec<Vec<u32>>,String>{
        let (_,triangles)=self.mesh()?;let mut counts=BTreeMap::new();
        for t in triangles.chunks_exact(3){for i in 0..3{let(a,b)=(t[i],t[(i+1)%3]);*counts.entry((a.min(b),a.max(b))).or_insert(0usize)+=1;}}
        if counts.values().any(|&n|n>2){return Err("Panel edges must be manifold".into());}
        let mut adjacent:BTreeMap<u32,Vec<u32>>=BTreeMap::new();for (&(a,b),&count)in &counts{if count==1{adjacent.entry(a).or_default().push(b);adjacent.entry(b).or_default().push(a);}}
        if adjacent.values().any(|v|v.len()!=2){return Err("Panel boundaries must form simple loops".into());}
        let mut remaining=adjacent.keys().copied().collect::<std::collections::BTreeSet<_>>();let mut loops=Vec::new();
        while let Some(&first)=remaining.first(){let mut chain=vec![first];remaining.remove(&first);let mut last=first;let mut next=adjacent[&first][0];
            while next!=first{if !remaining.remove(&next){return Err("Panel boundary overlaps itself".into());}chain.push(next);let choices=&adjacent[&next];let after=if choices[0]==last{choices[1]}else{choices[0]};last=next;next=after;}
            loops.push(chain);
        }Ok(loops)
    }
    /// Ordered seam vertices between two boundary picks. The default is the
    /// shorter boundary arc in metres; alternate selects the other arc.
    pub fn boundary_path(&self,start:u32,end:u32,alternate:bool)->Result<Vec<u32>,String>{
        if start==end{return Err("Choose different seam endpoints".into());}
        let chain=self.boundary_loops()?.into_iter().find(|c|c.contains(&start)&&c.contains(&end)).ok_or("Seam endpoints must be on the same boundary")?;
        let index=chain.iter().position(|&i|i==start).unwrap();let mut paths=[Vec::new(),Vec::new()];
        for side in 0..2{for k in 0..chain.len(){let at=if side==0{(index+k)%chain.len()}else{(index+chain.len()-k)%chain.len()};let v=chain[at];paths[side].push(v);if v==end{break;}}}
        let(p,_)=self.mesh()?;let length=|path:&[u32]|path.windows(2).map(|e|p[e[0]as usize].distance(p[e[1]as usize])).sum::<f32>();
        let shortest=usize::from(length(&paths[1])<length(&paths[0]));Ok(paths[shortest^usize::from(alternate)].clone())
    }
}
impl ClothPattern{
    /// Turn a generated grid into editable vertices without invalidating its
    /// side-based seams or changing any existing vertex/stitch identity.
    pub fn make_panel_editable(&mut self,index:usize)->Result<(),String>{
        let panel=self.panels.get(index).ok_or("Missing panel")?;if !panel.vertices.is_empty(){return Ok(());}
        let(vertices,triangles)=panel.mesh()?;let mut seams=self.seams.clone();
        for seam in &mut seams{for edge in [&mut seam.a,&mut seam.b]{if edge.panel==index{edge.vertices=panel.edge(edge,vertices.len(),&triangles)?;}}}
        self.seams=seams;self.panels[index].vertices=vertices.into_iter().map(|p|p.to_array()).collect();self.panels[index].triangles=triangles;Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn concave_outlines_keep_area_boundaries_and_winding_after_subdivision(){
        let outline=[[0.0,0.0],[2.0,0.0],[2.0,1.0],[1.0,1.0],[1.0,2.0],[0.0,2.0]];
        for reverse in [false,true]{let mut points=outline.to_vec();if reverse{points.reverse();}
            let panel=ClothPanel::from_outline(&points,3).unwrap();let(p,t)=panel.mesh().unwrap();
            let area=t.chunks_exact(3).map(|t|{let a=orient(p[t[0]as usize].as_dvec2(),p[t[1]as usize].as_dvec2(),p[t[2]as usize].as_dvec2())*0.5;assert!(a>0.0);a}).sum::<f64>();assert!((area-3.0).abs()<1e-6);
            let loops=panel.boundary_loops().unwrap();assert_eq!(loops.len(),1);assert_eq!(loops[0].len(),outline.len()*8);
            for other in [false,true]{let chain=panel.boundary_path(0,1,other).unwrap();assert_eq!(chain[0],0);assert_eq!(*chain.last().unwrap(),1);panel.edge(&ClothPanelEdge{vertices:chain,..Default::default()},p.len(),&t).unwrap();}
        }
        assert!(ClothPanel::from_outline(&[[0.0,0.0],[1.0,1.0],[0.0,1.0],[1.0,0.0]],1).is_err());
        assert!(ClothPanel::from_outline(&[[0.0,0.0],[1.0,0.0],[2.0,0.0]],1).is_err());
    }
    #[test]
    fn editable_panels_preserve_seams_and_reject_inverted_edits(){
        let mut pattern=ClothPattern{panels:vec![ClothPanel{segments:[2,2],..Default::default()},ClothPanel{translation:[1.0,0.0,0.0],segments:[2,2],..Default::default()}],seams:vec![ClothSeam{a:ClothPanelEdge{side:ClothPanelSide::Right,..Default::default()},..Default::default()}]};
        let before=pattern.assemble().unwrap();pattern.make_panel_editable(0).unwrap();let after=pattern.assemble().unwrap();
        assert_eq!(before.positions,after.positions);assert_eq!(before.indices,after.indices);assert_eq!(before.stitches,after.stitches);
        let panel=&mut pattern.panels[0];let before=panel.vertices.clone();assert!(panel.move_vertex(0,[2.0,2.0]).is_err());assert_eq!(before,panel.vertices);
        panel.move_vertex(0,[-0.1,0.0]).unwrap();assert_eq!(before[1..],panel.vertices[1..]);pattern.assemble().unwrap();
    }
    #[test]
    fn custom_outline_panels_sew_at_different_resolutions_in_the_live_solver(){
        use crate::{MeshCloth,MeshClothSim,ClothFabric};use glam::Mat4;
        let outline=[[0.0,0.0],[2.0,0.0],[2.0,1.0],[1.0,1.0],[1.0,2.0],[0.0,2.0]];
        let a=ClothPanel::from_outline(&outline,1).unwrap();let mut b=ClothPanel::from_outline(&outline,2).unwrap();b.rotation=[0.0,180.0,0.0];b.translation=[-0.2,0.0,0.0];
        let seam=ClothSeam{a:ClothPanelEdge{vertices:a.boundary_path(0,5,false).unwrap(),..Default::default()},b:ClothPanelEdge{panel:1,vertices:b.boundary_path(0,5,false).unwrap(),..Default::default()},..Default::default()};
        assert_ne!(seam.a.vertices.len(),seam.b.vertices.len());let pinned=a.vertices.len();
        let geometry=ClothPattern{panels:vec![a,b],seams:vec![seam]}.assemble().unwrap();let pins=(0..geometry.positions.len()).map(|i|if i<pinned{1.0}else{0.0}).collect::<Vec<_>>();
        let cfg=MeshCloth{gravity:[0.0;3],iterations:12,stitches:geometry.stitches.clone(),self_collision:true,fabric:Some(ClothFabric{drag:0.0,lift:0.0,..Default::default()}),..Default::default()};
        let mut sim=MeshClothSim::build(&geometry.positions,&geometry.indices,&pins,0.0,Mat4::IDENTITY).unwrap();sim.set_texcoords(&geometry.uv);sim.set_stitches(&geometry.stitches).unwrap();
        for _ in 0..180{sim.step(&cfg,&geometry.positions,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
        let mut p=geometry.positions.clone();let mut normals=vec![Vec3::Z;p.len()];sim.apply(&mut p,&mut normals,Mat4::IDENTITY);
        assert_eq!(&p[..pinned],&geometry.positions[..pinned]);
        let gap=geometry.stitches.iter().map(|s|p[s.vertex as usize].distance(p[s.edge[0]as usize].lerp(p[s.edge[1]as usize],s.weight))).fold(0.0,f32::max);
        assert!(gap<0.005,"custom-panel seam failed to close: {gap} m");
    }
}
