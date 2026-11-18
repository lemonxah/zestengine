use super::*;
use glam::{Mat4,Quat};

// Independent segment/triangle crossing check. A zero count alone does not
// prove clearance at touching vertices; the plane fixtures also check sides
// and the normal contact solver's final thickness explicitly.
fn intersections(top:&ClothContactTopology,p:&[Vec3])->usize{
    let mut count=0;
    for e in &top.edges{for t in &top.triangles{
        if e.iter().any(|i|t.contains(i)){continue;}
        let a=p[t[0]].as_dvec3();let u=p[t[1]].as_dvec3()-a;let v=p[t[2]].as_dvec3()-a;
        let n=u.cross(v).normalize_or_zero();let start=p[e[0]].as_dvec3();let end=p[e[1]].as_dvec3();
        let da=(start-a).dot(n);let db=(end-a).dot(n);
        // Ten micrometres is below the collision margin and above f32 roundoff
        // in these translated fixtures. Near-coplanar contact is checked by
        // separate side/margin assertions, not counted as a strict crossing.
        if da*db>=0.0||da.abs().min(db.abs())<1e-5{continue;}
        let direction=end-start;let h=direction.cross(v);let det=u.dot(h);if det.abs()<1e-20{continue;}
        let offset=start-a;let s=offset.dot(h)/det;let q=offset.cross(u);let r=direction.dot(q)/det;let time=v.dot(q)/det;
        if s>=-1e-8&&r>=-1e-8&&s+r<=1.0+1e-8&&time>0.0&&time<1.0{count+=1;}
    }}count
}

fn sheets(n:usize)->(ClothContactTopology,Vec<Vec3>,Vec<f32>,usize){
    let(mut p,mut triangles,mut weights)=(Vec::new(),Vec::new(),Vec::new());
    for vertical in [false,true]{
        let first=p.len();
        for y in 0..=n{for x in 0..=n{
            let u=x as f32/n as f32*2.0-1.0;let v=y as f32/n as f32*2.0-1.0;
            p.push(if vertical{Vec3::new(u,0.1,v)}else{Vec3::new(u*2.0,v*2.0,0.0)});
            weights.push(if !vertical||y==n{0.0}else{1.0});
        }}
        for y in 0..n{for x in 0..n{let a=first+y*(n+1)+x;triangles.extend([[a,a+1,a+n+1],[a+1,a+n+2,a+n+1]]);}}
    }
    (ClothContactTopology::new(triangles),p,weights,(n+1)*(n+1))
}

#[test]
fn pinned_intersecting_sheets_recover_across_tessellation_orientation_and_winding(){
    for n in [3,6,12]{for turn in [0.0,0.7,2.4]{
        let(top,initial,weights,first)=sheets(n);
        let transform=Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::new(1.0,2.0,3.0).normalize(),turn),Vec3::new(4.0,-2.0,7.0));
        let start:Vec<_>=initial.iter().map(|p|transform.transform_point3(*p)).collect();
        for reversed in [false,true]{
            let mut top=top.clone();if reversed{for (i,t) in top.triangles.iter_mut().enumerate(){if i%3==0{t.swap(1,2);}}}
            let mut p=start.clone();let mut count=0;
            for _ in 0..n*64{let changed=top.untangle(&mut p,&weights,0.01);count+=changed;if changed==0{break;}}
            assert!(count>0);
            for (i,q) in p.iter().enumerate(){
                if weights[i]==0.0{assert_eq!(*q,start[i],"pins cannot move");}
                if i>=first{let local=transform.inverse().transform_point3(*q);assert!(local.z>=0.0099,"n={n}, turn={turn}, winding={reversed}, vertex={i}, position={local:?}");}
            }
            let finished=p.clone();let changed=top.untangle(&mut p,&weights,0.01);assert_eq!(changed,0,"recovery settles: n={n}, turn={turn}, winding={reversed}, maximum move={}",p.iter().zip(&finished).map(|(a,b)|a.distance(*b)).fold(0.0,f32::max));assert_eq!(p,finished,"recovery settles instead of continuing to push");
            assert_eq!(intersections(&top,&p),0,"repair must not leave new self-intersections in the lifted sheet: n={n} turn={turn} winding={reversed}");
        }
    }}
}

#[test]
fn region_recovery_respects_garment_filters_and_both_contact_radii(){
    let(top,mut p,w,first)=sheets(3);let initial=p.clone();
    let radii:Vec<_>=(0..p.len()).map(|i|if i<first{0.02}else{0.03}).collect();
    let one=vec![0;p.len()];assert_eq!(recover(&top,&mut p,&w,0.06,Some(&one),Some(&radii)),0);assert_eq!(p,initial);
    let owners:Vec<_>=(0..p.len()).map(|i|usize::from(i>=first)).collect();
    for _ in 0..24{recover(&top,&mut p,&w,0.06,Some(&owners),Some(&radii));}
    assert!(p[first..].iter().all(|p|p.z>=0.0499));
    assert_eq!(&p[..first],&initial[..first]);
}

#[test]
fn recovery_preserves_mass_weighted_center_when_both_sheets_can_move(){
    let(top,mut p,_,first)=sheets(3);
    let w:Vec<_>=(0..p.len()).map(|i|if i<first{0.5}else{2.0}).collect();
    let center=|p:&[Vec3]|p.iter().zip(&w).map(|(p,w)|*p / *w).sum::<Vec3>();
    let before=center(&p);let mut count=0;
    for _ in 0..128{count+=top.untangle(&mut p,&w,0.01);}
    assert!(count>0);assert!(center(&p).distance(before)<1e-4,"recovery preserves the mass-weighted center");
    assert!(p.iter().all(|p|p.is_finite()));
    assert_eq!(intersections(&top,&p),0,"both free sheets must actually separate");
}

#[test]
fn invalid_or_already_separated_recovery_inputs_do_not_change_geometry(){
    let(top,mut p,w,first)=sheets(3);for p in &mut p[first..]{p.z+=2.0;}let original=p.clone();
    assert_eq!(top.untangle(&mut p,&w,0.01),0);assert_eq!(p,original);
    let mut invalid=w.clone();invalid[0]=f32::NAN;
    assert_eq!(top.untangle(&mut p,&invalid,0.01),0);assert_eq!(p,original);
    assert_eq!(top.untangle(&mut p,&w,f32::NAN),0);assert_eq!(p,original);
    let mut top=top;top.edges.push([0,usize::MAX]);
    assert_eq!(top.untangle(&mut p,&w,0.01),0);assert_eq!(p,original);
}

#[test]
fn mesh_simulation_recovers_a_trapped_sheet_with_material_and_ccd_enabled(){
    use crate::{ClothFabric,MeshCloth,MeshClothSim};
    for (n,angle) in [(3,0.0),(12,0.7)]{
    let(top,rest,w,first)=sheets(n);
    let transform=Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::new(1.0,2.0,3.0).normalize(),angle),Vec3::new(4.0,-2.0,7.0));
    let rest=rest.iter().map(|p|transform.transform_point3(*p)).collect::<Vec<_>>();
    let indices=top.triangles.iter().flatten().map(|i|*i as u32).collect::<Vec<_>>();
    let pins=w.iter().map(|w|if *w==0.0{1.0}else{0.0}).collect::<Vec<_>>();
    let cloth=MeshCloth{gravity:[0.0;3],damping:0.5,self_collision:true,continuous_collision:true,untangle:true,
        fabric:Some(ClothFabric{drag:0.0,lift:0.0,..Default::default()}),..Default::default()};
    let mut sim=MeshClothSim::build(&rest,&indices,&pins,0.0,Mat4::IDENTITY).unwrap();
    for _ in 0..180{sim.step(&cloth,&rest,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
    let mut positions=rest.clone();let mut normals=vec![Vec3::Z;rest.len()];sim.apply(&mut positions,&mut normals,Mat4::IDENTITY);
    for i in 0..rest.len(){
        assert!(positions[i].is_finite());
        if w[i]==0.0{assert_eq!(positions[i],rest[i]);}
        if i>=first{let local=transform.inverse().transform_point3(positions[i]);assert!(local.z>=-1e-5,"cloth must remain untangled with elasticity and CCD: n={n}, {i} {local:?}");}
    }
    assert_eq!(intersections(&top,&positions),0,"the material solver must preserve recovery: n={n}, angle={angle}");
    let stretch=top.edges.iter().map(|e|positions[e[0]].distance(positions[e[1]])/rest[e[0]].distance(rest[e[1]])).fold(0.0,f32::max);
    assert!(stretch<1.2,"recovery must let the material relax instead of freezing a stretched sheet: n={n}, stretch={stretch}");
    }
}

#[test]
fn a_single_folded_strip_recovers_without_releasing_its_pinned_ends(){
    let(mut p,mut triangles,mut weights)=(Vec::new(),Vec::new(),Vec::new());
    let path=[[-1.0,-1.0],[1.0,1.0],[-1.0,1.0],[1.0,-1.0]];
    let cols=4;let rows=21;
    for row in 0..=rows{
        let t=row as f32/rows as f32*3.0;let segment=(t as usize).min(2);let f=t-segment as f32;
        let y=path[segment][0]*(1.0-f)+path[segment+1][0]*f;
        let z=path[segment][1]*(1.0-f)+path[segment+1][1]*f;
        for x in 0..=cols{p.push(Vec3::new(x as f32/cols as f32*2.0-1.0,y,z));weights.push(if row==0||row==rows{0.0}else{1.0});}
    }
    for row in 0..rows{for x in 0..cols{let a=row*(cols+1)+x;triangles.extend([[a,a+1,a+cols+1],[a+1,a+cols+2,a+cols+1]]);}}
    let top=ClothContactTopology::new(triangles);let initial=p.clone();assert!(intersections(&top,&p)>0);
    for _ in 0..128{top.untangle(&mut p,&weights,0.01);}
    assert_eq!(intersections(&top,&p),0,"one garment's self-intersection must recover too");
    for (i,p) in p.iter().enumerate(){assert!(p.is_finite());if weights[i]==0.0{assert_eq!(*p,initial[i]);}}
}

#[test]
fn contact_solver_rejects_invalid_public_geometry_without_mutation(){
    let(top,initial,w,_)=sheets(3);
    let mut p=initial.clone();let mut invalid=top.clone();invalid.edges.push([0,usize::MAX]);
    assert_eq!(invalid.solve(&mut p,&initial,&w,0.01,true,true).untangled,0);assert_eq!(p,initial);
    let mut weights=w.clone();weights[0]=f32::NAN;
    assert_eq!(top.solve(&mut p,&initial,&weights,0.01,true,true).untangled,0);assert_eq!(p,initial);
    let mut previous=initial.clone();previous[0].x=f32::INFINITY;
    assert_eq!(top.solve(&mut p,&previous,&w,0.01,true,true).untangled,0);assert_eq!(p,initial);
    let owners=vec![0;initial.len()];let mut radii=vec![0.01;initial.len()];radii[0]=f32::NAN;
    assert_eq!(top.solve_between(&mut p,&initial,&w,&owners,&radii,true,true).untangled,0);assert_eq!(p,initial);
}

#[test]
fn crossing_guard_keeps_pins_and_retains_part_of_a_safe_motion(){
    let top=ClothContactTopology::new(vec![[0,1,2],[3,4,5]]);
    let previous=vec![Vec3::new(-2.0,-2.0,0.0),Vec3::new(2.0,-2.0,0.0),Vec3::new(0.0,2.0,0.0),
        Vec3::new(-0.5,0.0,1.0),Vec3::new(0.5,0.0,1.0),Vec3::new(0.0,0.0,2.0)];
    let weights=[0.0,0.0,0.0,1.0,1.0,1.0];
    let mut p=previous.clone();p[3].z = -1.0;
    assert!(intersections(&top,&p)>0);
    assert!(preserve_valid_pose(&top,&mut p,&previous,&weights,None));
    assert_eq!(intersections(&top,&p),0);assert_eq!(&p[..3],&previous[..3]);
    assert!(p[3].z<previous[3].z&&p[3].z>=0.0,"retain safe motion rather than unconditionally freezing");
    let mut clean=previous.clone();clean[3].x-=0.1;let proposed=clean.clone();
    assert!(!preserve_valid_pose(&top,&mut clean,&previous,&weights,None));assert_eq!(clean,proposed);
    let mut invalid=previous;invalid[3].z = -1.0;let mut current=invalid.clone();current[3].x-=0.1;
    let proposed=current.clone();assert!(!preserve_valid_pose(&top,&mut current,&invalid,&weights,None));assert_eq!(current,proposed,"never restore a known invalid origin");
}
