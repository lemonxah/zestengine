//! Body contacts for the rendered cloth surface at every simulation resolution.
//! Vertices and face interiors can hit bodies even when coarse particles miss.
use super::*;
use crate::cloth_contact::ClothColliderQuery;

impl MeshClothSim {
    /// Body and self constraints can oppose each other. Sweep each body repair
    /// through self-contact on the next pass, with a bounded convergence budget.
    pub(super) fn collide_visible_constraints(&mut self,cloth:&MeshCloth,queries:&[ClothColliderQuery<'_>],previous:Option<&[Vec3]>,check_self:bool,dt:f32){
        if !check_self||!cloth.self_collision{
            self.collide_visible_surface(cloth,queries,previous,dt);return;
        }
        let mut history=previous.map(<[Vec3]>::to_vec);
        let stationary:Vec<_>=queries.iter().map(|q|ClothColliderMotion::stationary(q.end().clone())).collect();
        let final_queries:Vec<_>=stationary.iter().map(ClothColliderMotion::prepare).collect();
        let (_,_,iterations)=cloth.lod.settings(cloth.runtime.lod_level,cloth.particle_spacing,cloth.frequency,cloth.iterations);
        let tolerance=(cloth.thickness*crate::cloth::max_scale(self.last_world)*0.005).max(1e-6);
        for pass in 0..iterations.clamp(1,8)*4{
            let before=self.visible_collision_positions();
            self.collide_visible_self(cloth,history.is_some(),history.as_deref(),dt);
            let separated=self.visible_collision_positions();
            // Only the first pass sweeps the moving body over the frame. Later
            // passes sweep new self corrections against its final pose; using
            // the original interval again freezes obsolete contact features.
            self.collide_visible_surface(cloth,if pass==0{queries}else{&final_queries},if pass==0{previous}else{Some(&before)},dt);
            let repaired=self.visible_collision_positions();
            let residual=before.iter().zip(&separated).chain(separated.iter().zip(&repaired)).map(|(a,b)|a.distance_squared(*b)).fold(0.0,f32::max);
            history=Some(separated);
            if residual<=tolerance*tolerance{break;}
        }
    }
    pub(super) fn has_collision_detail(&self)->bool{
        self.particles.len()!=self.vertex_count||!self.transition_offsets.is_empty()
    }
    pub(super) fn visible_collision_positions(&self)->Vec<Vec3>{self.collision_render_positions(self.last_world)}
    pub fn collision_render_positions(&self,world:Mat4)->Vec<Vec3>{
        let fallback;
        let anim=if self.anim.len()==self.particles.len(){&self.anim}else{fallback=self.animated(&self.animated_vertices,self.last_world);&fallback};
        self.animated_vertices.iter().enumerate().map(|(v,&p)|{
            let owner=self.owner[v]as usize;
            world.transform_point3(p)+self.particles[owner].pos-anim[owner]+world.transform_vector3(self.transition_offset(v))
        }).collect()
    }
    /// Visible collision topology and the translational driver of each vertex.
    /// This keeps shared garment contacts faithful at every simulation LOD.
    pub fn render_collision_geometry(&self,physical:bool,world:Mat4)->(ClothCollisionGeometry,Vec<usize>){
        let drivers:Vec<_>=self.owner.iter().map(|&i|i as usize).collect();
        let previous=if self.visible_frame_previous.len()==self.vertex_count{self.visible_frame_previous.clone()}else{drivers.iter().map(|&i|self.frame_previous[i]).collect()};
        let inverse_mass=drivers.iter().map(|&i|if physical&&self.fabric.weights.len()==self.particles.len(){self.fabric.weights[i]}else{1.0-self.particles[i].pin.clamp(0.0,1.0)}).collect();
        (ClothCollisionGeometry{positions:self.collision_render_positions(world),previous,inverse_mass,triangles:self.indices.as_chunks::<3>().0.iter().map(|t|[t[0]as usize,t[1]as usize,t[2]as usize]).collect()},drivers)
    }
    /// Commit mapped vertex contacts back to the actual solver and reproject
    /// against body/animation constraints before the next shared contact pass.
    pub fn apply_render_contact_positions(&mut self,cloth:&MeshCloth,positions:&[Vec3],world:Mat4,colliders:&[ClothShapeWorld],dt:f32){
        if positions.len()!=self.vertex_count||positions.iter().any(|p|!p.is_finite()){return;}
        let visible=self.collision_render_positions(world);
        let positions:Vec<_>=self.members.iter().enumerate().map(|(i,members)|{
            let delta=members.iter().map(|&v|(positions[v as usize]-visible[v as usize]).as_dvec3()).sum::<glam::DVec3>()/members.len()as f64;
            self.particles[i].pos+delta.as_vec3()
        }).collect();
        self.apply_contact_positions(cloth,&positions,colliders,dt);
    }
    /// Project visible vertices and whole faces, feeding corrections back to
    /// their simulation drivers. Rendering, inertia and shared garment contacts
    /// consequently agree on the repaired surface.
    pub(super) fn collide_visible_surface(&mut self,cloth:&MeshCloth,queries:&[ClothColliderQuery<'_>],previous:Option<&[Vec3]>,dt:f32){
        if !cloth.collide||queries.is_empty()||self.anim.len()!=self.particles.len(){return;}
        let radius=cloth.radius.max(0.0)*crate::cloth::max_scale(self.last_world);
        let offsets:Vec<_>=self.animated_vertices.iter().enumerate().map(|(v,&p)|{
            self.last_world.transform_point3(p)-self.anim[self.owner[v]as usize]+self.last_world.transform_vector3(self.transition_offset(v))
        }).collect();
        for _ in 0..3{
            let mut changed=false;
            for (v,&offset) in offsets.iter().enumerate(){
                let owner=self.owner[v]as usize;if self.particles[owner].pin>=1.0||(offset.length_squared()<1e-14&&previous.is_none()){continue;}
                for query in queries{
                    let point=self.particles[owner].pos+offset;
                    let contact=if cloth.continuous_collision&&previous.is_some_and(|p|p.len()==self.vertex_count){
                        query.resolve(previous.unwrap()[v],point,radius).map(|(out,n,_)|(out-point,n))
                    }else{penetration(point,radius,query.end())};
                    if let Some((push,normal))=contact.filter(|(push,_)|push.is_finite()&&push.length_squared()>1e-16){
                        let p=&mut self.particles[owner];p.pos+=push;p.contact=Some(normal);
                        if dt>1e-6{p.vel=(p.vel+push/dt).clamp_length_max(30.0);}changed=true;
                    }
                }
            }
            for t in self.indices.as_chunks::<3>().0{
                let vertices=[t[0]as usize,t[1]as usize,t[2]as usize];let owners=vertices.map(|v|self.owner[v]as usize);
                if owners.iter().all(|&i|self.particles[i].pin>=1.0){continue;}
                for query in queries{
                    // A linear plane cannot intersect the interior if all
                    // vertices clear it; their exact checks above suffice.
                    if matches!(query.end(),ClothShapeWorld::Plane{..}|ClothShapeWorld::Disabled){continue;}
                    let end=std::array::from_fn(|i|self.particles[owners[i]].pos+offsets[vertices[i]]);
                    let start=if cloth.continuous_collision{previous.filter(|p|p.len()==self.vertex_count).map_or(end,|p|vertices.map(|v|p[v]))}else{end};
                    let Some(contact)=query.triangle(start,end,radius)else{continue;};
                    if !contact.push.is_finite()||contact.push.length_squared()<1e-16{continue;}
                    let mut weights=[0.0;3];
                    for k in 0..3{let first=(0..=k).find(|&j|owners[j]==owners[k]).unwrap();weights[first]+=contact.weights[k];}
                    let mass=owners.map(|i|if self.particles[i].pin>=1.0{0.0}else if cloth.fabric.is_some()&&self.fabric.weights.len()==self.particles.len(){self.fabric.weights[i]}else{1.0-self.particles[i].pin.clamp(0.0,1.0)});
                    let denominator=(0..3).map(|i|mass[i]*weights[i]*weights[i]).sum::<f32>();if denominator<1e-20{continue;}
                    for k in 0..3{if weights[k]==0.0||mass[k]==0.0{continue;}
                        let push=contact.push*(mass[k]*weights[k]/denominator);let p=&mut self.particles[owners[k]];
                        p.pos+=push;p.contact=Some(contact.normal);if dt>1e-6{p.vel=(p.vel+push/dt).clamp_length_max(30.0);}
                    }
                    changed=true;
                }
            }
            if !changed{break;}
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn artistic_solver_honours_surface_ccd_and_untangling_controls(){
        let fixed=[Vec3::new(-2.0,-2.0,0.0),Vec3::new(2.0,-2.0,0.0),Vec3::new(0.0,2.0,0.0)];
        let moving=[Vec3::new(-0.3,-0.3,1.0),Vec3::new(0.3,-0.3,1.0),Vec3::new(0.0,0.3,1.0)];
        let rest=fixed.into_iter().chain(moving).collect::<Vec<_>>();let pins=[1.0,1.0,1.0,0.0,0.0,0.0];
        let mut cfg=MeshCloth{fabric:None,self_collision:true,continuous_collision:true,untangle:false,gravity:[0.0;3],damping:0.0,frequency:10.0,iterations:1,stiffness:0.0,bending:0.0,shape_stiffness:0.0,thickness:0.02,..Default::default()};
        for continuous in [true,false]{
            cfg.continuous_collision=continuous;let mut sim=MeshClothSim::build(&rest,&[0,1,2,3,4,5],&pins,0.0,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
            for p in &mut sim.particles[3..]{p.vel=Vec3::NEG_Z*25.0;}
            sim.step(&cfg,&rest,Mat4::IDENTITY,&[],&[],0.0,0.1);
            for(i,p)in sim.particles[..3].iter().enumerate(){assert_eq!(p.pos,rest[i]);}
            if continuous{assert!(sim.particles[3..].iter().all(|p|p.pos.z>=0.0199),"artistic cloth must sweep faces, not just particles");}
            else{assert!(sim.particles[3..].iter().all(|p|p.pos.z< -1.0),"the test must actually cross the sheet without CCD");}
        }
        let mut rest=rest;rest[3].z = -0.5;rest[4].z = -0.5;rest[5].z=0.5;
        cfg.untangle=true;cfg.continuous_collision=true;cfg.damping=1.0;
        let mut sim=MeshClothSim::build(&rest,&[0,1,2,3,4,5],&pins,0.0,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
        for _ in 0..8{sim.step(&cfg,&rest,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
        assert!(sim.particles[3..].iter().all(|p|p.pos.z>=0.0199)||sim.particles[3..].iter().all(|p|p.pos.z<=-0.0199),"artistic cloth must repair an initially crossing face");
    }
    #[test]
    fn whole_face_contacts_reach_fine_and_reduced_simulation_particles(){
        use crate::cloth_sdf::{ClothSdf,ClothSdfShape};
        let rest=[Vec3::ZERO,Vec3::X*2.0,Vec3::new(1.0,2.0,0.0)];
        let mut samples=Vec::new();for z in 0..9{for y in 0..9{for x in 0..9{
            let p=Vec3::new(x as f32,y as f32,z as f32)*0.025-Vec3::splat(0.1);let q=p.abs()-Vec3::splat(0.05);
            samples.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
        }}}
        let field=ClothSdf::from_samples([9;3],Vec3::splat(-0.1),Vec3::splat(0.025),samples).unwrap();
        for spacing in [0.0,10.0]{for sdf in [false,true]{for center in [Vec3::new(1.0,1.0,0.2),Vec3::new(1.137,1.231,0.2)]{
            let mut sim=MeshClothSim::build(&rest,&[0,1,2],&[0.0;3],spacing,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
            if spacing>0.0{assert_eq!(sim.particles.len(),1);}
            let shape=|c|if sdf{ClothShapeWorld::LevelSet(std::sync::Arc::new(ClothSdfShape::new(field.clone(),Mat4::from_translation(c)).unwrap()))}else{ClothShapeWorld::Sphere{c,r:0.05}};
            let motion=ClothColliderMotion{start:shape(center-Vec3::Z*0.4),end:shape(center)};
            let cfg=MeshCloth{gravity:[0.0;3],damping:1.0,fabric:None,shape_stiffness:0.0,stiffness:0.0,bending:0.0,tether_slack:100.0,radius:0.01,..Default::default()};
            sim.step_moving(&cfg,&rest,Mat4::IDENTITY,&[motion],&[],0.0,1.0/60.0);
            let p=sim.collision_render_positions(Mat4::IDENTITY);let face=[p[0],p[1],p[2]];
            let closest=crate::cloth_contact::closest_triangle(center,face).0;let normal=(p[1]-p[0]).cross(p[2]-p[0]).normalize();
            assert!(closest.distance(center)>=0.0599,"face intersects the body at spacing {spacing}, SDF={sdf}: {p:?}");
            assert!((center-p[0]).dot(normal)<0.0,"the body must stay on its original side of the face");
            if spacing>0.0{for i in 1..3{assert!((p[i]-p[0]).distance(rest[i]-rest[0])<1e-5,"body contact must preserve the reduced cluster's shape");}}
        }}}
    }
    #[test]
    fn coarse_cloth_sweeps_visible_vertices_when_their_particle_misses_a_body(){
        let mut rest=Vec::new();let mut pins=Vec::new();let mut indices=Vec::new();let n=8u32;
        for y in 0..=n{for x in 0..=n{rest.push(Vec3::new(x as f32/n as f32,y as f32/n as f32,0.0));pins.push(if y==n{1.0}else{0.0});}}
        for y in 0..n{for x in 0..n{let a=y*(n+1)+x;indices.extend([a,a+1,a+n+1,a+1,a+n+2,a+n+1]);}}
        let mut sim=MeshClothSim::build(&rest,&indices,&pins,0.3,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
        assert!(sim.particles.len()<rest.len()/2);
        assert!(sim.particles.iter().all(|p|p.pos.truncate().length()>0.045),"all particle paths miss the small moving sphere");
        let cfg=MeshCloth{gravity:[0.0;3],damping:1.0,fabric:None,shape_stiffness:0.0,stiffness:0.0,bending:0.0,tether_slack:100.0,radius:0.005,..Default::default()};
        let motion=ClothColliderMotion{start:ClothShapeWorld::Sphere{c:Vec3::NEG_Z*0.15,r:0.04},end:ClothShapeWorld::Sphere{c:Vec3::Z*0.15,r:0.04}};
        sim.step_moving(&cfg,&rest,Mat4::IDENTITY,&[motion],&[],0.0,1.0/60.0);
        let mut rendered=rest.clone();let mut normals=vec![Vec3::Z;rest.len()];sim.apply(&mut rendered,&mut normals,Mat4::IDENTITY);
        assert!(rendered[0].z>=0.1949,"the moving sphere must carry the visible corner: {:?}",rendered[0]);
        for(i,&pin)in pins.iter().enumerate(){if pin==1.0{assert_eq!(rendered[i],rest[i]);}}
    }
    #[test]
    fn the_last_lod_blend_frame_still_sweeps_its_visible_motion(){
        let rest=[Vec3::ZERO,Vec3::X,Vec3::Y];
        let mut sim=MeshClothSim::build(&rest,&[0,1,2],&[0.0,1.0,1.0],0.0,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
        sim.transition_offsets=vec![Vec3::X*0.2,Vec3::ZERO,Vec3::ZERO];sim.transition_duration=0.05;sim.transition_remaining=0.05;
        let cfg=MeshCloth{gravity:[0.0;3],damping:1.0,fabric:None,shape_stiffness:0.0,stiffness:0.0,bending:0.0,tether_slack:100.0,radius:0.005,..Default::default()};
        let body=ClothShapeWorld::Sphere{c:Vec3::X*0.1,r:0.03};
        sim.step(&cfg,&rest,Mat4::IDENTITY,&[body],&[],0.0,0.1);
        let mut output=rest.to_vec();sim.apply(&mut output,&mut [Vec3::Z;3],Mat4::IDENTITY);
        assert!(sim.transition_offsets.is_empty());assert!(output[0].x>=0.1349,"fading visual detail must not pass through the body: {:?}",output[0]);
        assert_eq!(output[1],rest[1]);assert_eq!(output[2],rest[2]);
    }

}
