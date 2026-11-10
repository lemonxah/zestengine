//! XPBD seams sampled along panel boundaries, including unequal tessellations.
use super::*;
#[derive(Clone,Debug)]
pub(super) struct ParticleStitch{pub terms:Vec<(usize,f32)>,pub compliance:f32,pub lambda:f32,vertices:[u32;3],coefficients:[f32;3],start:Vec3,end:Vec3}
impl MeshClothSim{
    pub fn set_stitches(&mut self,stitches:&[crate::ClothStitch])->Result<(),String>{
        let mut seams=Vec::new();let mut neighbours=std::collections::HashSet::new();
        for seam in stitches{
            if [seam.vertex,seam.edge[0],seam.edge[1]].iter().any(|&v|v as usize>=self.vertex_count)||!seam.weight.is_finite()||!(0.0..=1.0).contains(&seam.weight)||!seam.compliance.is_finite()||seam.compliance<0.0{return Err("Invalid cloth stitch".into());}
            let ids=[self.owner[seam.vertex as usize],self.owner[seam.edge[0] as usize],self.owner[seam.edge[1] as usize]];
            let coefficients=[1.0,-(1.0-seam.weight),-seam.weight];let mut terms:Vec<(usize,f32)>=Vec::new();
            for (id,c) in ids.into_iter().zip(coefficients){if let Some((_,weight))=terms.iter_mut().find(|(i,_)|*i==id as usize){*weight+=c;}else{terms.push((id as usize,c));}}
            terms.retain(|(_,w)|w.abs()>1e-6);
            for &a in &ids{for &b in &ids{if a!=b{neighbours.insert((a.min(b),a.max(b)));}}}
            if terms.len()>1{seams.push(ParticleStitch{terms,compliance:seam.compliance,lambda:0.0,vertices:[seam.vertex,seam.edge[0],seam.edge[1]],coefficients,start:Vec3::ZERO,end:Vec3::ZERO});}
        }
        self.sewing=seams;self.seam_neighbours=neighbours;self.fabric.exclude_seams(&self.seam_neighbours);self.rebuild_surface_contacts();Ok(())
    }
    pub(super) fn prepare_seams(&mut self,skinned:&[Vec3],world:Mat4,anim:&[Vec3]){
        if self.sewing.is_empty(){return;}
        let previous=if self.anim.len()==self.particles.len(){self.anim.clone()}else{self.animated(&self.animated_vertices,self.last_world)};
        for seam in &mut self.sewing{
            seam.start=seam.vertices.iter().zip(seam.coefficients).map(|(&v,c)|(self.last_world.transform_point3(self.animated_vertices[v as usize])-previous[self.owner[v as usize] as usize])*c).sum();
            seam.end=seam.vertices.iter().zip(seam.coefficients).map(|(&v,c)|(world.transform_point3(skinned[v as usize])-anim[self.owner[v as usize] as usize])*c).sum();
        }
    }
    pub(super) fn carry_seams(&mut self,delta:Mat4){
        // The bias is a vector (the coefficients sum to zero). Teleport Keep
        // has already transported particles and their animation targets.
        for seam in &mut self.sewing{seam.start=delta.transform_vector3(seam.start);}
    }
    pub(super) fn solve_seams(&mut self,physical:bool,h:f32,fraction:f32){
        for seam in &mut self.sewing{
            let delta=seam.terms.iter().map(|&(i,c)|self.particles[i].pos*c).sum::<Vec3>()+seam.start.lerp(seam.end,fraction);let len=delta.length();if len<1e-8{continue;}
            let mass=|i:usize|if physical{self.fabric.weights.get(i).copied().unwrap_or(0.0)}else{1.0-self.particles[i].pin.clamp(0.0,1.0)};
            let mut weights=[0.0;3];for (k,&(i,_)) in seam.terms.iter().enumerate(){weights[k]=mass(i);}
            let alpha=seam.compliance/(h*h).max(1e-12);let denominator=seam.terms.iter().zip(&weights).map(|((_,c),w)|c*c*w).sum::<f32>()+alpha;
            if denominator<1e-10{continue;}let dl=(-len-alpha*seam.lambda)/denominator;seam.lambda+=dl;let normal=delta/len;
            for (&(i,c),w) in seam.terms.iter().zip(weights){self.particles[i].pos+=normal*(dl*c*w);}
        }
    }
}
