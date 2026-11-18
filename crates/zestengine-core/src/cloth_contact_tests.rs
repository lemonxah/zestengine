use super::*;
#[test]
fn sphere_sweeps_match_analytic_roots_even_when_both_endpoints_are_outside(){
    let mut seed=7u32;let mut rand=||{seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);seed as f32/u32::MAX as f32};
    for _ in 0..1000 {
        let radius=0.001+rand()*0.4;let y=(rand()*2.0-1.0)*radius*0.99;
        let c0=Vec3::new(rand()-0.5,rand()-0.5,rand()-0.5);let motion=Vec3::new(rand(),rand(),rand());
        let start=c0+Vec3::new(-2.0,y,0.0);let end=c0+motion+Vec3::new(2.0,y,0.0);
        let collider=ClothColliderMotion{start:ClothShapeWorld::Sphere{c:c0,r:radius},end:ClothShapeWorld::Sphere{c:c0+motion,r:radius}};
        let hit=sweep_cloth_collider(start,end,0.0,collider.clone()).unwrap();let exact=(2.0-(radius*radius-y*y).sqrt())/4.0;
        assert!(!hit.conservative);assert!((hit.time-exact).abs()<2e-5,"{} vs {exact}",hit.time);
        let(out,_,_)=resolve_cloth_collider(start,end,0.0,collider.clone()).unwrap();assert!(cloth_shape_distance(out,collider.end).0>=-1e-5);
    }
}
#[test]
fn moving_sphere_transfers_motion_and_stationary_contact_can_slide_away(){
    let motion=ClothColliderMotion{start:ClothShapeWorld::Sphere{c:Vec3::NEG_X,r:0.1},end:ClothShapeWorld::Sphere{c:Vec3::X,r:0.1}};
    let (p,n,carried)=resolve_cloth_collider(Vec3::ZERO,Vec3::ZERO,0.01,motion.clone()).unwrap();
    assert!(p.x>=1.1099&&n.x>0.99);assert!(carried.distance(Vec3::X*2.0)<1e-6);
    let sphere=ClothColliderMotion::stationary(ClothShapeWorld::Sphere{c:Vec3::ZERO,r:1.0});
    assert!(resolve_cloth_collider(Vec3::X,Vec3::X*2.0,0.0,sphere.clone()).is_none());
}
#[test]
fn thin_tapered_capsules_and_rotating_planes_are_swept(){
    let capsule=ClothColliderMotion::stationary(ClothShapeWorld::Capsule{a:Vec3::NEG_Y,b:Vec3::Y,ra:0.001,rb:0.1});
    let(p,_,_)=resolve_cloth_collider(Vec3::new(-1.0,-0.9,0.0),Vec3::new(1.0,-0.9,0.0),0.002,capsule.clone()).unwrap();
    assert!(p.x<0.0&&cloth_shape_distance(p,capsule.end).0>0.0019);
    let plane=ClothColliderMotion{start:ClothShapeWorld::Plane{p:Vec3::ZERO,n:Vec3::Y},end:ClothShapeWorld::Plane{p:Vec3::ZERO,n:Vec3::X}};
    let p=Vec3::new(-1.0,1.0,0.0);let hit=sweep_cloth_collider(p,p,0.0,plane).unwrap();assert!((hit.time-0.5).abs()<1e-5);
    let degenerate=ClothShapeWorld::Capsule{a:Vec3::ZERO,b:Vec3::ZERO,ra:0.1,rb:0.2};
    assert!((cloth_shape_distance(Vec3::X,degenerate).0-0.8).abs()<1e-6);
}
#[test]
fn vertex_face_ccd_preserves_side_when_entire_step_crosses_the_face(){
    let start=[Vec3::new(0.2,0.2,2.0),Vec3::ZERO,Vec3::X,Vec3::Y];let mut end=start;end[0].z = -2.0;
    let hit=sweep_cloth_vertex_face(start,end,0.001).unwrap();assert!((hit.sweep.time-0.49975).abs()<1e-5);
    assert!(hit.sweep.normal.z>0.99);assert!((hit.coefficients.iter().sum::<f32>()).abs()<1e-6);
    let fixed=end[1..].to_vec();project_cloth_contact(&mut end,&[1.0,0.0,0.0,0.0],[0,1,2,3],hit,0.001);
    assert!(end[0].z>=0.00099);assert_eq!(&end[1..],fixed);
}
#[test]
fn edge_edge_ccd_finds_midsegment_crossings_and_conserves_center_of_mass(){
    let start=[Vec3::new(-1.0,0.0,1.0),Vec3::new(1.0,0.0,1.0),Vec3::new(0.0,-1.0,0.0),Vec3::new(0.0,1.0,0.0)];
    let mut end=start;end[0].z = -1.0;end[1].z = -1.0;
    let hit=sweep_cloth_edges(start,end,0.002).unwrap();assert!((hit.sweep.time-0.499).abs()<1e-5);
    assert!(hit.coefficients.iter().all(|c|(c.abs()-0.5).abs()<1e-5));
    let center=end.iter().sum::<Vec3>();project_cloth_contact(&mut end,&[1.0;4],[0,1,2,3],hit,0.002);
    assert!(end.iter().sum::<Vec3>().distance(center)<1e-6);assert!(end[0].z-end[2].z>=0.00199);
}
#[test]
fn swept_broad_phase_does_not_only_query_the_final_cell(){
    let start=[Vec3::new(0.2,0.2,5.0),Vec3::new(0.3,0.2,5.0),Vec3::new(0.2,0.3,5.0),Vec3::ZERO,Vec3::X,Vec3::Y];
    let mut p=start;for q in &mut p[..3]{q.z = -5.0;}
    let topology=ClothContactTopology::new(vec![[0,1,2],[3,4,5]]);
    let report=topology.solve(&mut p,&start,&[1.0,1.0,1.0,0.0,0.0,0.0],0.01,true,false);
    assert!(report.vertex_face>=3);assert!(p[..3].iter().all(|p|p.z>=0.0099));
}
#[test]
fn existing_intersections_are_untangled_without_moving_fixed_face(){
    let mut p=[Vec3::ZERO,Vec3::X,Vec3::Y,Vec3::new(0.2,0.2,-0.1),Vec3::new(0.2,0.2,0.3),Vec3::new(0.4,0.2,0.3)];
    let original=p;let topology=ClothContactTopology::new(vec![[0,1,2],[3,4,5]]);let weights=[0.0,0.0,0.0,1.0,1.0,1.0];
    let mut count=0;for _ in 0..8{count+=topology.untangle(&mut p,&weights,0.01);}
    assert!(count>0);assert_eq!(&p[..3],&original[..3]);assert!(p[3..].iter().all(|p|p.z>=0.0099),"{p:?}");
}
#[test]
fn common_translation_does_not_change_cloth_contact(){
    let start=[Vec3::new(0.2,0.2,1.0),Vec3::ZERO,Vec3::X,Vec3::Y];let mut end=start;end[0].z = -1.0;
    let reference=sweep_cloth_vertex_face(start,end,0.01).unwrap();let shift=Vec3::new(100.0,-50.0,20.0);
    let moved=sweep_cloth_vertex_face(start,end.map(|p|p+shift),0.01).unwrap();
    assert!((reference.sweep.time-moved.sweep.time).abs()<1e-5);assert!(reference.sweep.normal.distance(moved.sweep.normal)<1e-3);
}

#[test]
fn full_mesh_step_sweeps_fast_motion_instead_of_only_testing_the_end(){
    use crate::{MeshCloth,MeshClothSim,ClothFabric};use glam::Mat4;
    // Check separation along the entire resulting trajectory. A sphere permits
    // tangential deflection around it; requiring a negative final X incorrectly
    // rejects valid sliding. Work in the collider's translating frame so the
    // same independent analytic check also covers a moving body.
    let clearance=|start:Vec3,end:Vec3,c0:Vec3,c1:Vec3|{
        let a=start-c0;let b=end-c1;let d=b-a;
        let t=if d.length_squared()>1e-12{(-a.dot(d)/d.length_squared()).clamp(0.0,1.0)}else{0.0};
        (a+d*t).length()
    };
    let rest=[Vec3::new(-1.0,-0.02,0.0),Vec3::new(-1.0,0.02,0.0),Vec3::new(-1.0,0.0,0.02)];
    let mut cloth=MeshCloth{frequency:10.0,gravity:[0.0;3],damping:0.0,fabric:Some(ClothFabric{drag:0.0,lift:0.0,..Default::default()}),..Default::default()};
    let mut sim=MeshClothSim::build(&rest,&[0,1,2],&[0.0;3],0.0,Mat4::IDENTITY).unwrap();
    for p in &mut sim.particles{p.vel=Vec3::X*20.0;}
    let mut discrete=sim.clone();cloth.continuous_collision=false;
    discrete.step(&cloth,&rest,Mat4::IDENTITY,&[ClothShapeWorld::Sphere{c:Vec3::ZERO,r:0.05}],&[],0.0,0.1);
    assert!(discrete.particles.iter().zip(rest).all(|(p,start)|clearance(start,p.pos,Vec3::ZERO,Vec3::ZERO)<0.05));
    cloth.continuous_collision=true;
    sim.step(&cloth,&rest,Mat4::IDENTITY,&[ClothShapeWorld::Sphere{c:Vec3::ZERO,r:0.05}],&[],0.0,0.1);
    assert!(sim.particles.iter().zip(rest).all(|(p,start)|clearance(start,p.pos,Vec3::ZERO,Vec3::ZERO)>=0.05+cloth.radius-1e-5),"{:?}",sim.particles);
    let rest=rest.map(|p|p+Vec3::X);let mut sim=MeshClothSim::build(&rest,&[0,1,2],&[0.0;3],0.0,Mat4::IDENTITY).unwrap();
    let motion=ClothColliderMotion{start:ClothShapeWorld::Sphere{c:Vec3::NEG_X,r:0.05},end:ClothShapeWorld::Sphere{c:Vec3::X,r:0.05}};
    sim.step_moving(&cloth,&rest,Mat4::IDENTITY,&[motion],&[],0.0,0.1);
    assert!(sim.particles.iter().zip(rest).all(|(p,start)|clearance(start,p.pos,Vec3::NEG_X,Vec3::X)>=0.05+cloth.radius-1e-5),"moving collider carries cloth out of its swept path: {:?}",sim.particles);
}
#[test]
fn mutual_contact_uses_both_radii_and_excludes_same_garment(){
    let start=[Vec3::ZERO,Vec3::X,Vec3::Y,Vec3::new(0.2,0.2,0.01),Vec3::new(0.3,0.2,0.01),Vec3::new(0.2,0.3,0.01)];
    let topology=ClothContactTopology::new(vec![[0,1,2],[3,4,5]]);let weights=[0.0,0.0,0.0,1.0,1.0,1.0];let radii=[0.01,0.01,0.01,0.02,0.02,0.02];
    let mut p=start;let report=topology.solve_between(&mut p,&start,&weights,&[0,0,0,1,1,1],&radii,true,true);
    assert!(report.vertex_face>0);assert!(p[3..].iter().all(|p|(p.z-0.03).abs()<1e-5));
    let mut p=start;topology.solve_between(&mut p,&start,&weights,&[0;6],&radii,true,true);assert_eq!(p,start);
}

#[test]
fn swept_segment_finds_thin_mid_edge_obstacles_with_clear_endpoints(){
    let start=[Vec3::new(-1.0,-1.0,0.0),Vec3::new(-1.0,1.0,0.0)];let end=start.map(|p|p+Vec3::X*2.0);
    for shape in [ClothShapeWorld::Sphere{c:Vec3::ZERO,r:0.001},ClothShapeWorld::Capsule{a:Vec3::NEG_Z,b:Vec3::Z,ra:0.001,rb:0.004}] {
        let motion=ClothColliderMotion::stationary(shape.clone());
        for i in 0..2{assert!(sweep_cloth_collider(start[i],end[i],0.001,motion.clone()).is_none());}
        let (u,push,n,_)=resolve_cloth_segment(start,end,[0.001;2],motion.clone()).expect("segment interior sweeps through the obstacle");
        assert!((u-0.5).abs()<1e-5&&n.x< -0.99);
        assert!(end[0].x+push.x<0.0);
        assert!(cloth_shape_distance(end[0].lerp(end[1],u)+push,shape).0>=0.00099);
    }
    let plane=ClothColliderMotion::stationary(ClothShapeWorld::Plane{p:Vec3::ZERO,n:Vec3::NEG_X});
    let (_,push,_,_)=resolve_cloth_segment(start,end,[0.001,0.2],plane).unwrap();
    assert!(end[1].x+push.x<= -0.1999,"varying chain radii use the largest penetrating endpoint");
}

#[test]
fn bone_chain_sweeps_a_body_crossing_between_both_particles(){
    use crate::{BoneCloth,BoneClothSim};use glam::Mat4;
    let models=[Mat4::from_translation(Vec3::Y),Mat4::from_translation(Vec3::NEG_Y)];
    let c=BoneCloth{roots:vec!["root".into()],gravity:[0.0;3],stiffness:[0.0;2],end_tail:false,frequency:10.0,damping:0.0,radius:[0.01;2],..Default::default()};
    let mut sim=BoneClothSim::build(&c,&[None,Some(0)],&["root".into(),"tip".into()],&models,Mat4::IDENTITY).unwrap();
    let body=ClothColliderMotion{start:ClothShapeWorld::Sphere{c:Vec3::NEG_Z,r:0.05},end:ClothShapeWorld::Sphere{c:Vec3::Z,r:0.05}};
    let before:Vec<_>=sim.particles.iter().map(|p|p.pos).collect();
    assert!(before.iter().all(|&p|sweep_cloth_collider(p,p,0.01,body.clone()).is_none()));
    sim.step_moving(&c,&models,Mat4::IDENTITY,&[body],&[],0.0,0.1);
    assert_eq!(sim.particles[0].pos,before[0],"root stays anchored");
    assert!(sim.particles[1].pos.z>1.0,"segment contact moves the free tip");
    for k in 0..=1000{
        let t=k as f32/1000.0;let a=before[0].lerp(sim.particles[0].pos,t);let b=before[1].lerp(sim.particles[1].pos,t);
        let center=Vec3::NEG_Z.lerp(Vec3::Z,t);let edge=b-a;let u=((center-a).dot(edge)/edge.length_squared()).clamp(0.0,1.0);
        assert!(center.distance(a+edge*u)>=0.0599,"swept chain stays outside moving sphere at {t}");
    }
}
