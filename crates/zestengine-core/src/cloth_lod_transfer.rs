//! State transfer through the common render topology. Simulation tiers never
//! replace UVs, materials or render triangles.
use super::*;
impl MeshClothSim {
    pub(super) fn transition_offset(&self,vertex:usize)->Vec3{
        let t=if self.transition_duration>0.0{(self.transition_remaining/self.transition_duration).clamp(0.0,1.0)}else{0.0};
        // Smoothstep has zero slope at both ends of a transition.
        self.transition_offsets.get(vertex).copied().unwrap_or(Vec3::ZERO)*(t*t*(3.0-2.0*t))
    }
    /// Import displacement and velocity from another resolution of the SAME
    /// render mesh. State stays in the previous frame, so normal inertia,
    /// animation and teleport handling still happen exactly once on the step.
    /// Rejects incompatible topology. Callers must also check mesh source identity.
    pub fn transfer_resolution(&mut self,old:&Self,seconds:f32)->bool{
        if self.vertex_count!=old.vertex_count||self.indices!=old.indices||old.anim.len()!=old.particles.len(){return false;}
        self.reset(&old.animated_vertices,old.last_world);
        let inv=old.last_world.inverse();
        let previous_offsets:Vec<_>=old.owner.iter().map(|&owner|{let i=owner as usize;old.particles[i].pos-old.anim[i]}).collect();
        for (i,members) in self.members.iter().enumerate(){
            let p=&mut self.particles[i];
            if p.pin>=1.0{continue;}
            p.pos+=members.iter().map(|&v|previous_offsets[v as usize]).sum::<Vec3>()/members.len() as f32;
            p.vel=members.iter().map(|&v|old.particles[old.owner[v as usize] as usize].vel).sum::<Vec3>()/members.len() as f32;
        }
        self.frame_previous=self.particles.iter().map(|p|p.pos).collect();
        self.transition_duration=if seconds.is_finite(){seconds.clamp(0.0,10.0)}else{0.0};
        self.transition_remaining=self.transition_duration;
        if self.transition_duration>0.0 {
            self.transition_offsets=(0..self.vertex_count).map(|v|{
                let i=self.owner[v] as usize;
                if self.particles[i].pin>=1.0{return Vec3::ZERO;}
                inv.transform_vector3(previous_offsets[v]-(self.particles[i].pos-self.anim[i]))+old.transition_offset(v)
            }).collect();
        }
        self.visible_frame_previous=self.visible_collision_positions();
        true
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn grid()->(Vec<Vec3>,Vec<u32>,Vec<f32>){grid_size(8)}
    fn grid_size(n:u32)->(Vec<Vec3>,Vec<u32>,Vec<f32>){
        let mut p=Vec::new();let mut t=Vec::new();let mut pins=Vec::new();
        for y in 0..=n{for x in 0..=n{p.push(Vec3::new(x as f32/n as f32,y as f32/n as f32,0.0));pins.push(if y==n{1.0}else{0.0});}}
        for y in 0..n{for x in 0..n{let a=y*(n+1)+x;t.extend([a,a+1,a+n+1,a+1,a+n+2,a+n+1]);}}
        (p,t,pins)
    }
    #[test]
    fn cloth_lod_transfer_keeps_render_pose_velocity_and_pins_then_fades(){
        let (p,t,pins)=grid();let world=Mat4::from_scale_rotation_translation(Vec3::splat(2.0),glam::Quat::from_rotation_y(0.4),Vec3::new(3.0,2.0,1.0));
        let mut fine=MeshClothSim::build(&p,&t,&pins,0.0,world).unwrap();fine.reset(&p,world);
        for (i,point) in fine.particles.iter_mut().enumerate().filter(|(_,p)|p.pin<1.0){point.pos+=Vec3::Z*(0.05*(i as f32).sin());point.vel=Vec3::new(0.2,0.1,0.3);}
        let mut before=p.clone();let mut normal=vec![Vec3::Z;p.len()];fine.apply(&mut before,&mut normal,world);
        let mut coarse=MeshClothSim::build(&p,&t,&pins,0.3,world).unwrap();assert!(coarse.particles.len()<fine.particles.len()/2);
        assert!(coarse.transfer_resolution(&fine,0.25));
        let mut after=p.clone();let mut after_normal=vec![Vec3::Z;p.len()];coarse.apply(&mut after,&mut after_normal,world);
        for ((a,b),(an,bn)) in before.iter().zip(&after).zip(normal.iter().zip(&after_normal)){assert!((*a-*b).length()<1e-5);assert!((*an-*bn).length()<1e-4);}
        for (i,point) in coarse.particles.iter().enumerate(){
            if point.pin>=1.0{assert!((point.pos-coarse.anim[i]).length()<1e-6);}else{assert!((point.vel-Vec3::new(0.2,0.1,0.3)).length()<1e-6);}
        }
        // A second switch midway includes the first switch's remaining detail.
        coarse.transition_remaining=0.125;
        let mut mid=p.clone();coarse.apply(&mut mid,&mut normal,world);
        let mut finer=MeshClothSim::build(&p,&t,&pins,0.0,world).unwrap();assert!(finer.transfer_resolution(&coarse,0.25));
        let mut switched=p.clone();finer.apply(&mut switched,&mut normal,world);
        assert!(mid.iter().zip(switched).all(|(a,b)|(*a-b).length()<1e-5));
        let cloth=MeshCloth{gravity:[0.0;3],fabric:None,shape_stiffness:0.0,stiffness:0.0,bending:0.0,tether_slack:100.0,..Default::default()};
        for _ in 0..30{finer.step(&cloth,&p,world,&[],&[],0.0,1.0/60.0);}
        assert!(finer.transition_offsets.is_empty());assert!(finer.particles.iter().all(|p|p.pos.is_finite()));
        // Resetting after teleport/cull discards residual detail.
        finer.transfer_resolution(&coarse,0.25);finer.reset(&p,world);assert!(finer.transition_offsets.is_empty());
    }
    #[test]
    #[ignore = "explicit performance measurement; prints workstation timings"]
    fn cloth_lod_profile_self_contact(){
        let (p,t,pins)=grid_size(24);let mut cloth=MeshCloth{self_collision:true,iterations:8,gravity:[0.0,0.0,-9.81],..Default::default()};
        cloth.lod.enabled=true;cloth.lod.levels=vec![crate::ClothLodLevel{distance:10.0,particle_spacing:0.125,frequency:30.0,iterations:3}];
        let mut near=MeshClothSim::build(&p,&t,&pins,0.0,Mat4::IDENTITY).unwrap();near.reset(&p,Mat4::IDENTITY);
        for _ in 0..10{near.step(&cloth,&p,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
        let mut far=MeshClothSim::build(&p,&t,&pins,0.125,Mat4::IDENTITY).unwrap();assert!(far.transfer_resolution(&near,0.25));
        for (level,sim) in [(0,&mut near),(1,&mut far)]{
            cloth.runtime.lod_level=level;let mut durations=Vec::new();
            for frame in 0..80{let start=std::time::Instant::now();sim.step(&cloth,&p,Mat4::IDENTITY,&[],&[],frame as f32/60.0,1.0/60.0);durations.push(start.elapsed().as_secs_f64()*1000.0);}
            durations.sort_by(f64::total_cmp);eprintln!("cloth LOD {level}: {} particles, self-contact median {:.3} ms, p95 {:.3} ms",sim.particles.len(),durations[40],durations[76]);
            assert!(sim.particles.iter().all(|p|p.pos.is_finite()));
        }
    }
    #[test]
    fn cloth_lod_does_not_merge_pins_with_free_vertices(){
        let (p,t,pins)=grid();let coarse=MeshClothSim::build(&p,&t,&pins,0.3,Mat4::IDENTITY).unwrap();
        for m in &coarse.members{assert!(m.iter().all(|&v|pins[v as usize]==pins[m[0] as usize]));}
    }
}
