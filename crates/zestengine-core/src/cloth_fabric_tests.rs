use super::*;
use super::super::{MeshCloth,MeshClothSim};
fn particle(pos:Vec3)->MeshParticle{MeshParticle{pos,vel:Vec3::ZERO,pin:0.0,max_distance:f32::INFINITY,prev_anim:pos,contact:None}}
fn sheet(n:usize)->(Vec<Vec3>,Vec<u32>,Vec<f32>){
    let mut p=Vec::new();let mut idx=Vec::new();let mut pins=Vec::new();
    for y in 0..=n{for x in 0..=n{p.push(Vec3::new(x as f32/n as f32,0.0,y as f32/n as f32));pins.push(if y==0{1.0}else{0.0});}}
    for y in 0..n{for x in 0..n{let a=(y*(n+1)+x)as u32;let w=(n+1)as u32;idx.extend([a,a+1,a+w,a+1,a+w+1,a+w]);}}
    (p,idx,pins)
}
#[test]
fn fold_derivatives_match_numerical_energy_gradient(){
    for points in [[Vec3::ZERO,Vec3::X,Vec3::Y,Vec3::new(0.2,-1.0,0.4)],
        [Vec3::new(2.0,1.0,3.0),Vec3::new(2.6,1.2,3.4),Vec3::new(2.2,1.8,3.7),Vec3::new(2.3,0.4,2.7)]]{
        let(_,gradient)=dihedral(points).unwrap();
        assert!(gradient.iter().sum::<Vec3>().length()<1e-5);
        for i in 0..4{for axis in 0..3{let mut a=points;let mut b=points;a[i][axis]+=0.0002;b[i][axis]-=0.0002;
            let numeric=wrap_angle(dihedral(a).unwrap().0-dihedral(b).unwrap().0)/0.0004;
            assert!((numeric-gradient[i][axis]).abs()<0.003,"{i}/{axis}: {numeric} != {}",gradient[i][axis]);
        }}
    }
}
#[test]
fn xpbd_multiplier_makes_a_single_constraint_iteration_independent(){
    for steps in [1,4,32]{let mut p=[particle(Vec3::ZERO),particle(Vec3::X*1.2)];let mut lambda=0.0;
        for _ in 0..steps{let c=p[1].pos.x-p[0].pos.x-1.0;project(&mut p,&[0.0,2.0],[0,1],[Vec3::NEG_X,Vec3::X],c,3.0,&mut lambda);}
        // Closed-form compliant solve: correction = -0.2 * 2 / (2+3).
        assert!((p[1].pos.x-1.12).abs()<1e-6);assert_eq!(p[0].pos,Vec3::ZERO);
    }
}
#[test]
fn geodesic_tethers_follow_folded_paths_and_do_not_cross_islands(){
    let rest=[Vec3::ZERO,Vec3::X,Vec3::new(1.0,1.0,0.0),Vec3::new(0.0,1.0,0.0),Vec3::new(0.0,0.1,0.0),Vec3::Z];
    let mut state=FabricState::default();state.rebuild_tethers(&rest,&[(0,1),(1,2),(2,3),(3,4)],&[1.0,0.0,0.0,0.0,0.0,0.0]);
    let last=state.tethers.iter().find(|t|t.0==4).unwrap();assert_eq!(last.1,0);assert!((last.2-3.9).abs()<1e-6);
    assert!(!state.tethers.iter().any(|t|t.0==5));
    state.rebuild_tethers(&rest,&[(0,1),(1,2),(2,3),(3,4)],&[0.4,0.0,0.0,0.0,0.0,0.0]);assert!(state.tethers.is_empty(),"soft animation pins aren't hard attachment anchors");
}
#[test]
fn uv_axes_select_directional_stiffness(){
    let rest=[Vec3::ZERO,Vec3::X,Vec3::Y];let cfg=ClothFabric{warp:10.0,weft:10000.0,shear:0.0,bend:0.0,..Default::default()};
    let solve=|uv:&[Vec2]|{let mut p=rest.map(particle);p[0].pin=1.0;p[2].pin=1.0;p[1].pos.x=1.2;
        let mut state=FabricState::build(&rest,&[0,1,2],&[0,1,2],uv,&[(0,1),(0,2),(1,2)],&[1.0,0.0,1.0]);state.prepare(&rest,Mat4::IDENTITY,&p,&cfg);state.begin_step(&rest);
        for _ in 0..12{state.solve(&mut p,&cfg,1.0/60.0);}p[1].pos.x-1.0};
    let warp=solve(&[Vec2::ZERO,Vec2::X,Vec2::Y]);let weft=solve(&[Vec2::ZERO,Vec2::Y,Vec2::X]);
    assert!(warp>weft*10.0,"warp {warp}, weft {weft}");
}
#[test]
fn aerodynamic_forces_are_area_weighted_two_sided_and_edge_on_quiet(){
    let rest=[Vec3::ZERO,Vec3::X,Vec3::Y];let cfg=ClothFabric{lift:0.0,..Default::default()};
    let solve=|indices:&[u32],velocity:Vec3|{let mut p=rest.map(particle);for point in &mut p{point.vel=velocity;}
        let mut state=FabricState::build(&rest,&[0,1,2],indices,&[],&[],&[0.0;3]);state.prepare(&rest,Mat4::IDENTITY,&p,&cfg);
        state.forces(&p,&cfg,&[],1.0,0.0).iter().sum::<Vec3>()};
    let f=solve(&[0,1,2],Vec3::NEG_Z*2.0);let reverse=solve(&[0,2,1],Vec3::NEG_Z*2.0);
    assert!((f.z-1.225).abs()<1e-5);assert!(f.distance(reverse)<1e-6);assert!(solve(&[0,1,2],Vec3::X*2.0).length()<1e-6);
}
#[test]
fn fabric_hangs_with_stable_size_across_frame_rates_and_solver_counts(){
    let (rest,indices,pins)=sheet(6);let mut heights=Vec::new();
    for (fps,iterations) in [(30,8),(60,8),(120,8),(60,16)]{
        let cloth=MeshCloth{iterations,frequency:120.0,damping:0.05,fabric:Some(ClothFabric{bend:0.002,..Default::default()}),..Default::default()};
        let mut sim=MeshClothSim::build(&rest,&indices,&pins,0.0,Mat4::IDENTITY).unwrap();
        sim.set_texcoords(&rest.iter().map(|p|Vec2::new(p.x,p.z)).collect::<Vec<_>>());
        for frame in 0..fps*4 {sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],frame as f32/fps as f32,1.0/fps as f32);}
        for (i,p) in sim.particles.iter().enumerate(){assert!(p.pos.is_finite());if pins[i]==1.0{assert!(p.pos.distance(rest[i])<1e-6);}}
        let low=sim.particles.iter().map(|p|p.pos.y).fold(0.0,f32::min);assert!(low< -0.7&&low> -1.1,"{fps}/{iterations}: {low}");heights.push(low);
    }
    let span=heights.iter().copied().fold(f32::NEG_INFINITY,f32::max)-heights.iter().copied().fold(f32::INFINITY,f32::min);
    assert!(span<0.04,"frame/iteration variation {span}: {heights:?}");
}
#[test]
fn surface_contacts_stop_a_particle_in_the_middle_of_a_coarse_face(){
    let rest=[Vec3::ZERO,Vec3::X,Vec3::Y,Vec3::new(0.3,0.3,0.001),Vec3::new(0.31,0.3,0.1),Vec3::new(0.3,0.31,0.1)];
    let mut p=rest.map(particle);for v in &mut p[..3]{v.pin=1.0;}
    let mut state=FabricState::build(&rest,&[0,1,2,3,4,5],&[0,1,2,3,4,5],&[],&[],&[1.0,1.0,1.0,0.0,0.0,0.0]);state.prepare(&rest,Mat4::IDENTITY,&p,&ClothFabric::default());
    state.self_collide(&mut p,&rest,0.02);assert!(p[3].pos.z>=0.0199,"{}",p[3].pos.z);assert_eq!(p[0].pos,rest[0]);
}
#[test]
fn invalid_meshes_are_rejected_and_fabric_settings_round_trip(){
    assert!(MeshClothSim::build(&[Vec3::ZERO],&[0,1,2],&[],0.0,Mat4::IDENTITY).is_none());
    assert!(MeshClothSim::build(&[Vec3::NAN;3],&[0,1,2],&[],0.0,Mat4::IDENTITY).is_none());
    for preset in ClothFabricPreset::ALL{let c=MeshCloth{fabric:Some(preset.settings()),..Default::default()};let back:MeshCloth=ron::from_str(&ron::to_string(&c).unwrap()).unwrap();assert_eq!(c.fabric,back.fabric);}
}
#[test]
fn teleport_reset_drops_old_render_offsets_and_rescales_painted_limits(){
    let (rest,indices,pins)=sheet(2);let cloth=MeshCloth::default();
    let mut sim=MeshClothSim::build(&rest,&indices,&pins,0.0,Mat4::IDENTITY).unwrap();sim.set_limits(Some(&[0.1;9]));
    for _ in 0..30{sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
    let world=Mat4::from_scale_rotation_translation(Vec3::splat(2.0),glam::Quat::from_rotation_y(1.0),Vec3::new(10.0,0.0,0.0));
    sim.step(&cloth,&rest,world,&[],&[],0.0,1.0/60.0);
    let mut positions=rest.clone();let mut normals=vec![Vec3::Y;rest.len()];sim.apply(&mut positions,&mut normals,world);
    assert!(positions.iter().zip(&rest).all(|(a,b)|a.distance(*b)<1e-6));
    assert!(sim.particles.iter().all(|p|p.vel==Vec3::ZERO&&(p.max_distance-0.2).abs()<1e-6));
}
#[test]
fn backstop_follows_animated_normals_and_drive_converges(){
    let (rest,idx,_)=sheet(2);let mut sim=MeshClothSim::build(&rest,&idx,&[0.0;9],0.0,Mat4::IDENTITY).unwrap();
    // This sheet's winding points down; upward gravity pushes into its backstop.
    let cloth=MeshCloth{gravity:[0.0,9.81,0.0],fabric:Some(ClothFabric{backstop_radius:0.2,..Default::default()}),..Default::default()};
    for _ in 0..120{sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
    assert!(sim.particles.iter().all(|p|p.pos.y<1e-5));
    for p in &mut sim.particles{p.pos+=Vec3::Y*0.4;}
    let cloth=MeshCloth{gravity:[0.0;3],fabric:Some(ClothFabric{animation_hz:3.0,..Default::default()}),..Default::default()};
    for _ in 0..120{sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
    assert!(sim.particles.iter().zip(rest).all(|(p,a)|p.pos.distance(a)<0.001));
}

#[test]
#[ignore = "explicit cloth CPU budget and preset geometry capture"]
fn profile_and_capture_fabric_presets(){
    for n in [12,24]{for self_collision in [false,true]{
        let(rest,idx,pins)=sheet(n);let cloth=MeshCloth{self_collision,iterations:8,..Default::default()};
        let mut sim=MeshClothSim::build(&rest,&idx,&pins,0.0,Mat4::IDENTITY).unwrap();
        let mut timings=Vec::new();
        for frame in 0..80{let start=std::time::Instant::now();sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],frame as f32/60.0,1.0/60.0);
            if frame>=60{timings.push(start.elapsed().as_secs_f64()*1000.0);}}
        timings.sort_by(f64::total_cmp);
        eprintln!("[cloth budget] {} particles, {} triangles, self collision {self_collision}: {:.3} ms/frame median; two substeps x eight passes",rest.len(),idx.len()/3,timings[timings.len()/2]);
        assert!(sim.particles.iter().all(|p|p.pos.is_finite()&&p.pos.length()<3.0));
    }}
    let Some(dir)=std::env::var_os("ZESTENGINE_CLOTH_DUMP_DIR")else{return};let dir=std::path::Path::new(&dir);std::fs::create_dir_all(dir).unwrap();
    let (rest,indices,pins)=sheet(20);
    let wind=crate::ClothWind{strength:5.0,turbulence:0.2,..Default::default()};
    let winds=[crate::WindSample::new(&wind,Mat4::from_rotation_x(0.8))];
    for preset in ClothFabricPreset::ALL {
        let cloth=MeshCloth{fabric:Some(preset.settings()),iterations:8,..Default::default()};
        let mut sim=MeshClothSim::build(&rest,&indices,&pins,0.0,Mat4::IDENTITY).unwrap();
        sim.set_texcoords(&rest.iter().map(|p|Vec2::new(p.x,p.z)).collect::<Vec<_>>());
        for frame in 0..180{sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&winds,frame as f32/60.0,1.0/60.0);}
        let mut positions=rest.clone();let mut normals=vec![Vec3::Y;rest.len()];sim.apply(&mut positions,&mut normals,Mat4::IDENTITY);
        let mut obj=String::from("# Independent ZestEngine fabric solver, 3 seconds in 5 m/s wind\n");
        for p in positions{obj.push_str(&format!("v {} {} {}\n",p.x,p.y,p.z));}
        for t in indices.chunks_exact(3){obj.push_str(&format!("f {} {} {}\n",t[0]+1,t[1]+1,t[2]+1));}
        std::fs::write(dir.join(format!("{}.obj",preset.label())),obj).unwrap();
    }
}

#[test]
fn painted_stiffness_controls_warp_weft_shear_and_folds_independently(){
    for channel in 0..3{
        let rest=[Vec3::ZERO,Vec3::X,Vec3::Y];let cfg=ClothFabric{warp:if channel==0{10000.0}else{0.0},weft:if channel==1{10000.0}else{0.0},shear:if channel==2{10000.0}else{0.0},bend:0.0,..Default::default()};
        let run=|mask:f32|{let mut p=rest.map(particle);p[1].pos=if channel==0{Vec3::X*1.3}else{Vec3::X};p[2].pos=if channel==1{Vec3::Y*1.3}else if channel==2{Vec3::Y+Vec3::X*0.3}else{Vec3::Y};
            let start=p.iter().map(|p|p.pos).collect::<Vec<_>>();let mut state=FabricState::build(&rest,&[0,1,2],&[0,1,2],&[Vec2::ZERO,Vec2::X,Vec2::Y],&[],&[0.0;3]);
            let mut masks=[[1.0;7];3];for m in &mut masks{m[channel]=mask;}state.set_material_masks(&masks);state.prepare(&rest,Mat4::IDENTITY,&p,&cfg);state.begin_step(&rest);
            for _ in 0..12{state.solve(&mut p,&cfg,1.0/60.0);}p.iter().zip(start).map(|(p,a)|p.pos.distance(a)).sum::<f32>()};
        assert!(run(0.0)<1e-6,"zero map disables channel {channel}");assert!(run(1.0)>0.05,"full map restores channel {channel}");assert!(run(0.1)<run(1.0));
    }
    let rest=[Vec3::ZERO,Vec3::X,Vec3::Y,Vec3::new(1.0,1.0,0.0)];let cfg=ClothFabric{warp:0.0,weft:0.0,shear:0.0,bend:1.0,..Default::default()};
    let run=|mask:f32|{let mut p=rest.map(particle);p[3].pos.z=0.4;let start=p[3].pos;let mut state=FabricState::build(&rest,&[0,1,2,3],&[0,1,2,1,3,2],&[],&[],&[0.0;4]);let mut masks=[[1.0;7];4];for m in &mut masks{m[3]=mask;}state.set_material_masks(&masks);state.prepare(&rest,Mat4::IDENTITY,&p,&cfg);for _ in 0..60{state.begin_step(&rest);for _ in 0..12{state.solve(&mut p,&cfg,1.0/60.0);}}p[3].pos.distance(start)};
    assert!(run(0.0)<1e-6);let folded=run(1.0);assert!(folded>0.01,"full-map fold correction {folded}");
}

#[test]
fn painted_drive_and_backstops_act_locally_and_survive_shared_contacts(){
    use crate::{ClothPaint,PaintChannel};
    let p=[Vec3::ZERO,Vec3::X,Vec3::Y,Vec3::X*2.0,Vec3::X*3.0,Vec3::new(2.0,1.0,0.0)];let idx=[0,1,2,3,4,5];
    let mut paint=ClothPaint::seed(&p,&idx,&[0.0;6]);paint.set_values(PaintChannel::AnimationDrive,&[0.0,0.0,0.0,1.0,1.0,1.0]);
    let mut sim=MeshClothSim::build(&p,&idx,&[0.0;6],0.0,Mat4::IDENTITY).unwrap();sim.reset(&p,Mat4::IDENTITY);sim.set_painted_maps(&paint);
    let cloth=MeshCloth{gravity:[0.0;3],damping:0.0,fabric:Some(ClothFabric{warp:0.0,weft:0.0,shear:0.0,bend:0.0,drag:0.0,lift:0.0,animation_hz:3.0,..Default::default()}),..Default::default()};
    for point in &mut sim.particles{point.pos+=Vec3::Z*0.3;}
    for _ in 0..120{sim.step(&cloth,&p,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
    for i in 0..3{assert!((sim.particles[i].pos.z-0.3).abs()<1e-5);assert!(sim.particles[i+3].pos.z.abs()<0.001);}
    // Radius mask disables one island. A gap mask moves the other island's boundary.
    paint.set_values(PaintChannel::BackstopRadius,&[0.0,0.0,0.0,1.0,1.0,1.0]);paint.set_values(PaintChannel::BackstopDistance,&[1.0,1.0,1.0,0.5,0.5,0.5]);sim.reset(&p,Mat4::IDENTITY);sim.set_painted_maps(&paint);
    let cloth=MeshCloth{gravity:[0.0;3],fabric:Some(ClothFabric{backstop_radius:0.2,backstop_distance:0.1,warp:0.0,weft:0.0,shear:0.0,bend:0.0,drag:0.0,lift:0.0,..Default::default()}),..Default::default()};
    let pushed:Vec<_>=p.iter().map(|p|*p-Vec3::Z*0.08).collect();
    for (point,target) in sim.particles.iter_mut().zip(&pushed){point.pos=*target;point.vel=Vec3::ZERO;}
    sim.step(&cloth,&p,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);
    // The collision impulse may carry the point outward on the second substep;
    // the invariant is exclusion from the painted backstop, not exact rest on it.
    for i in 0..3{assert!((sim.particles[i].pos.z+0.08).abs()<1e-5);assert!(sim.particles[i+3].pos.z>= -0.1*128.0/255.0-1e-5);}
    sim.apply_contact_positions(&cloth,&pushed,&[],1.0/60.0);
    for i in 0..3{assert!((sim.particles[i].pos.z+0.08).abs()<1e-5);assert!((sim.particles[i+3].pos.z+0.1*128.0/255.0).abs()<1e-5);}
    paint.remove_channel(PaintChannel::BackstopRadius);sim.set_painted_maps(&paint);sim.apply_contact_positions(&cloth,&pushed,&[],1.0/60.0);
    // First island's full gap allows 0.08 m; change that gap to zero live.
    paint.set_values(PaintChannel::BackstopDistance,&[0.0;6]);sim.set_painted_maps(&paint);sim.apply_contact_positions(&cloth,&pushed,&[],1.0/60.0);
    assert!(sim.particles.iter().all(|p|p.pos.z.abs()<1e-5));
}

#[test]
fn painted_material_maps_follow_lod_resolution_without_changing_pins(){
    use crate::{ClothPaint,PaintChannel};
    let (rest,idx,_)=sheet(8);let pins=vec![0.0;rest.len()];
    let mut paint=ClothPaint::seed(&rest,&idx,&pins);paint.set_values(PaintChannel::AnimationDrive,&rest.iter().map(|p|if p.x<0.5{0.0}else{1.0}).collect::<Vec<_>>());
    let mut fine=MeshClothSim::build(&rest,&idx,&pins,0.0,Mat4::IDENTITY).unwrap();fine.reset(&rest,Mat4::IDENTITY);
    for p in &mut fine.particles{p.pos.y=0.3;}
    let mut coarse=MeshClothSim::build(&rest,&idx,&pins,0.3,Mat4::IDENTITY).unwrap();coarse.transfer_resolution(&fine,0.25);coarse.set_painted_maps(&paint);
    assert!(coarse.particles.len()<fine.particles.len()/2);
    let cloth=MeshCloth{gravity:[0.0;3],damping:0.0,fabric:Some(ClothFabric{warp:0.0,weft:0.0,shear:0.0,bend:0.0,drag:0.0,lift:0.0,animation_hz:3.0,..Default::default()}),..Default::default()};
    for _ in 0..120{coarse.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
    assert!(coarse.particles.iter().filter(|p|p.pos.x<0.2).all(|p|(p.pos.y-0.3).abs()<1e-5));
    assert!(coarse.particles.iter().filter(|p|p.pos.x>0.7).all(|p|p.pos.y.abs()<0.001));
    assert!(coarse.particles.iter().all(|p|p.pin==0.0));
}
