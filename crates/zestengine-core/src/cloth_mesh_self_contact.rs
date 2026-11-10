//! Contact coverage for visible detail on reduced cloth meshes.
use super::*;

pub(super) fn topology(indices:&[u32],welds:&[usize])->crate::cloth_contact::ClothContactTopology{
    let mut topology=crate::cloth_contact::ClothContactTopology::new(indices.chunks_exact(3).map(|t|[t[0]as usize,t[1]as usize,t[2]as usize]).collect());
    let mut groups=HashMap::<usize,Vec<usize>>::new();
    for (v,&weld)in welds.iter().enumerate(){groups.entry(weld).or_default().push(v);}
    // UV duplicates share adjacency even if their vertex indices differ.
    for vertices in groups.values(){for &a in vertices{for &b in vertices{if a!=b{topology.exclude_pair(a,b);}}}}
    for edge in topology.edges.clone(){for &a in &groups[&welds[edge[0]]]{for &b in &groups[&welds[edge[1]]]{if a!=b{topology.exclude_pair(a,b);}}}}
    topology
}

impl MeshClothSim{
    pub(super) fn rebuild_surface_contacts(&mut self){
        self.surface_contacts=topology(&self.indices,&self.normal_owner);
        for &(a,b)in &self.seam_neighbours{for &a in &self.members[a as usize]{for &b in &self.members[b as usize]{self.surface_contacts.exclude_pair(a as usize,b as usize);}}}
    }
    pub(super) fn collide_visible_self(&mut self,cloth:&MeshCloth,sweep:bool,previous:Option<&[Vec3]>,dt:f32){
        let (mut geometry,_)=self.render_collision_geometry(cloth.fabric.is_some(),self.last_world);
        let before=geometry.positions.clone();
        if let Some(previous)=previous.filter(|p|p.len()==before.len()){geometry.previous=previous.to_vec();}
        if !sweep||geometry.previous.len()!=before.len(){geometry.previous.clone_from(&before);}
        let thickness=cloth.thickness*crate::cloth::max_scale(self.last_world);
        // The shared body/self loop owns the iteration budget and refreshes
        // the sweep after every correction. Do not multiply that budget here.
        self.surface_contacts.solve_mapped(&mut geometry.positions,&geometry.previous,&geometry.inverse_mass,thickness,&self.surface_mapping,cloth.continuous_collision&&sweep,cloth.untangle);
        for (p,members)in self.particles.iter_mut().zip(&self.members){
            if p.pin>=1.0{continue;}
            let delta=(members.iter().map(|&v|(geometry.positions[v as usize]-before[v as usize]).as_dvec3()).sum::<glam::DVec3>()/members.len()as f64).as_vec3();
            p.pos+=delta;if dt>1e-6{p.vel=(p.vel+delta/dt).clamp_length_max(30.0);}
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn reduced_self_collision_covers_detail_that_moves_between_particles(){
        let n=8u32;let mut rest=Vec::new();let mut indices=Vec::new();let mut pins=Vec::new();
        for layer in 0..2{let offset=rest.len()as u32;
            for y in 0..=n{for x in 0..=n{rest.push(Vec3::new(x as f32/n as f32,y as f32/n as f32,layer as f32*0.06));pins.push(if layer==0{1.0}else{0.0});}}
            for y in 0..n{for x in 0..n{let a=offset+y*(n+1)+x;indices.extend([a,a+1,a+n+1,a+1,a+n+2,a+n+1]);}}
        }
        let upper=((n+1)*(n+1))as usize;let mut animated=rest.clone();animated[upper+40].z=-0.18;
        let cfg=MeshCloth{fabric:None,self_collision:true,continuous_collision:true,untangle:true,gravity:[0.0;3],damping:1.0,frequency:60.0,iterations:1,stiffness:0.0,bending:0.0,shape_stiffness:0.0,thickness:0.02,..Default::default()};
        for physical in [false,true]{for enabled in [false,true]{for spacing in [0.0,0.3]{
            let mut cfg=cfg.clone();cfg.self_collision=enabled;
            if physical{cfg.fabric=Some(ClothFabric{warp:0.0,weft:0.0,shear:0.0,bend:0.0,drag:0.0,lift:0.0,..Default::default()});}
            let mut sim=MeshClothSim::build(&rest,&indices,&pins,spacing,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
            if spacing>0.0{assert!(sim.particles.len()<rest.len()/2);}
            sim.step(&cfg,&animated,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);
            let mut visible=animated.clone();sim.apply(&mut visible,&mut vec![Vec3::Z;rest.len()],Mat4::IDENTITY);
            assert_eq!(&visible[..upper],&rest[..upper]);
            let minimum=visible[upper..].iter().map(|p|p.z).fold(f32::INFINITY,f32::min);
            if enabled{assert!(minimum>=0.0199,"visible self-collision tunnels at spacing {spacing}, physical {physical}: minimum z={minimum}");}
            else if spacing>0.0{assert!(minimum<0.0,"fixture must cross without self-collision");}
        }}}
    }
    #[test]
    fn body_and_self_contacts_converge_after_a_shared_correction(){
        let rest=vec![Vec3::new(-2.0,-2.0,0.0),Vec3::new(2.0,-2.0,0.0),Vec3::new(0.0,2.0,0.0),Vec3::new(-0.02,-0.02,0.04),Vec3::new(0.02,-0.02,0.04),Vec3::new(0.0,0.02,0.04)];
        let cfg=MeshCloth{fabric:None,self_collision:true,continuous_collision:true,thickness:0.02,radius:0.005,iterations:8,..Default::default()};
        let body=ClothShapeWorld::Sphere{c:Vec3::new(0.08,0.0,0.12),r:0.15};
        for spacing in [0.0,0.5]{
            let mut sim=MeshClothSim::build(&rest,&[0,1,2,3,4,5],&[1.0,1.0,1.0,0.0,0.0,0.0],spacing,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
            let mut proposed=rest.clone();for p in &mut proposed[3..]{p.z=0.03;}
            sim.apply_render_contact_positions(&cfg,&proposed,Mat4::IDENTITY,&[body.clone()],1.0/60.0);
            let visible=sim.visible_collision_positions();assert_eq!(&visible[..3],&rest[..3]);
            assert!(visible[3..].iter().all(|p|p.z>=0.0198),"body repair crossed own cloth: {:?}",&visible[3..]);
            for a in 0..=10{for b in 0..=10-a{
                let p=visible[3]*(a as f32/10.0)+visible[4]*(b as f32/10.0)+visible[5]*((10-a-b)as f32/10.0);
                assert!(crate::cloth_contact::cloth_shape_distance(p,body.clone()).0>=0.0048,"self repair penetrated body at {p:?}");
            }}
        }
    }
    #[test]
    fn shared_corrections_cannot_push_a_garment_through_its_own_fixed_layer(){
        let rest=vec![Vec3::new(-2.0,-2.0,0.0),Vec3::new(2.0,-2.0,0.0),Vec3::new(0.0,2.0,0.0),Vec3::new(-0.2,-0.2,0.1),Vec3::new(0.2,-0.2,0.1),Vec3::new(0.0,0.2,0.1)];
        let cfg=MeshCloth{fabric:None,self_collision:true,continuous_collision:true,thickness:0.02,..Default::default()};
        for spacing in [0.0,0.5]{
            let mut sim=MeshClothSim::build(&rest,&[0,1,2,3,4,5],&[1.0,1.0,1.0,0.0,0.0,0.0],spacing,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
            let mut proposed=rest.clone();for p in &mut proposed[3..]{p.z=-0.1;}
            sim.apply_render_contact_positions(&cfg,&proposed,Mat4::IDENTITY,&[],1.0/60.0);
            let visible=sim.visible_collision_positions();assert_eq!(&visible[..3],&rest[..3]);
            assert!(visible[3..].iter().all(|p|p.z>=0.0199),"shared correction crossed own cloth at spacing {spacing}: {:?}",&visible[3..]);
        }
    }
    #[test]
    fn the_final_detail_fade_frame_cannot_cross_a_pinned_cloth_surface(){
        let rest=vec![Vec3::new(-2.0,-2.0,0.0),Vec3::new(2.0,-2.0,0.0),Vec3::new(0.0,2.0,0.0),Vec3::new(-0.2,-0.2,-0.1),Vec3::new(0.2,-0.2,-0.1),Vec3::new(0.0,0.2,-0.1)];
        let mut sim=MeshClothSim::build(&rest,&[0,1,2,3,4,5],&[1.0,1.0,1.0,0.0,0.0,0.0],0.0,Mat4::IDENTITY).unwrap();sim.reset(&rest,Mat4::IDENTITY);
        sim.transition_offsets=vec![Vec3::ZERO,Vec3::ZERO,Vec3::ZERO,Vec3::Z*0.2,Vec3::Z*0.2,Vec3::Z*0.2];sim.transition_duration=0.05;sim.transition_remaining=0.05;
        let cfg=MeshCloth{fabric:None,self_collision:true,continuous_collision:true,gravity:[0.0;3],damping:1.0,stiffness:0.0,bending:0.0,shape_stiffness:0.0,thickness:0.02,..Default::default()};
        sim.step(&cfg,&rest,Mat4::IDENTITY,&[],&[],0.0,0.1);
        let visible=sim.visible_collision_positions();
        assert!(sim.transition_offsets.is_empty());assert_eq!(&visible[..3],&rest[..3]);
        assert!(visible[3..].iter().all(|p|p.z>=0.0199),"fading detail crossed a fixed face: {:?}",&visible[3..]);
    }
    #[test]
    fn a_coarse_cluster_does_not_weld_opposite_sides_of_a_fold(){
        let mut rest=Vec::new();let mut indices=Vec::new();let n=8u32;
        // Hairpin cross-section, connected only around the far fold.
        let rows=(0..=n).map(|y|(y as f32/n as f32,0.0))
            .chain([(1.03,0.03)])
            .chain((0..=n).rev().map(|y|(y as f32/n as f32,0.06))).collect::<Vec<_>>();
        for &(y,z)in &rows{for x in 0..=n{rest.push(Vec3::new(x as f32/n as f32,y,z));}}
        for row in 0..rows.len()as u32-1{for x in 0..n{let a=row*(n+1)+x;indices.extend([a,a+1,a+n+1,a+1,a+n+2,a+n+1]);}}
        let sim=MeshClothSim::build(&rest,&indices,&vec![0.0;rest.len()],0.3,Mat4::IDENTITY).unwrap();assert!(sim.particles.len()<rest.len());
        for members in &sim.members{
            if members.iter().any(|&i|rest[i as usize].y>0.7){continue;}
            let sides=members.iter().map(|&i|rest[i as usize].z>0.03).collect::<std::collections::HashSet<_>>();
            assert!(sides.len()==1,"one cell merged disconnected local layers of a folded sheet");
        }
    }
}
