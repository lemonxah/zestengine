//! Same-frame garment contacts. Skinning queues cloth uploads until every
//! garment has simulated; shared contacts then update both garments before
//! either GPU patches or CPU vertex streams are submitted.
use glam::Mat4;
use zestengine_core::{MeshCloth,cloth_contact::{ClothContactTopology,ClothContactMapping}};
use zestengine_render::{MeshHandle,Renderer,Vertex};
use crate::scene::SceneObject;

pub(crate) struct ClothRenderWork {
    pub mesh:usize,
    pub object:usize,
    pub slot:usize,
    pub handle:MeshHandle,
    pub animated:Vec<Vertex>,
    pub world:Mat4,
    pub gpu:bool,
}

fn cloth(objects:&[SceneObject],object:usize,slot:usize)->Option<zestengine_core::ecs::ComponentRef<'_, MeshCloth>>{
    objects.get(object)?.components.iter_typed::<MeshCloth>().find(|c|c.slot==slot)
}
fn cloth_mut(objects:&[SceneObject],object:usize,slot:usize)->Option<zestengine_core::ecs::ComponentMut<'_, MeshCloth>>{
    objects.get(object)?.components.iter_typed_mut::<MeshCloth>().find(|c|c.slot==slot)
}

pub(crate) fn solve_groups(objects:&[SceneObject],entries:&[(usize,usize,Mat4)],dt:f32){
    let mut groups:std::collections::BTreeMap<u32,Vec<(usize,usize,Mat4)>>=Default::default();
    let mut seen=std::collections::HashSet::new();
    for &(object,slot,world) in entries {
        if !seen.insert((object,slot)){continue;}
        let Some(c)=cloth(objects,object,slot)else{continue};
        if c.collision_group==0||c.runtime.cache.playing||c.runtime.culled||c.runtime.hold||c.runtime.sim.is_none(){continue;}
        groups.entry(c.collision_group).or_default().push((object,slot,world));
    }
    for entries in groups.values_mut(){
        if entries.len()<2{continue;}
        entries.sort_by_key(|&(object,slot,_)|(objects[object].id().raw(),slot));
        let mut positions=Vec::new();let mut previous=Vec::new();let mut weights=Vec::new();let mut radii=Vec::new();let mut owners=Vec::new();let mut triangles=Vec::new();let mut spans=Vec::new();let mut drivers=Vec::new();let mut driver_base=0;
        let mut continuous=false;let mut untangle=false;let mut passes=1;
        for (owner,&(object,slot,world)) in entries.iter().enumerate(){
            let c=cloth(objects,object,slot).unwrap();let sim=c.runtime.sim.as_ref().unwrap();let (geometry,mapping)=sim.render_collision_geometry(c.fabric.is_some(),world);
            drivers.extend(mapping.into_iter().map(|i|i+driver_base));driver_base+=sim.particles.len();
            let first=positions.len();let count=geometry.positions.len();
            triangles.extend(geometry.triangles.iter().map(|t|t.map(|i|i+first)));
            positions.extend(geometry.positions);previous.extend(geometry.previous);weights.extend(geometry.inverse_mass);
            radii.extend(std::iter::repeat_n(c.thickness.max(0.0)*zestengine_core::max_scale(world)*0.5,count));owners.extend(std::iter::repeat_n(owner,count));spans.push(first..first+count);
            continuous|=c.continuous_collision;untangle|=c.untangle;passes=passes.max(c.lod.settings(c.runtime.lod_level,c.particle_spacing,c.frequency,c.iterations).2.clamp(1,8));
        }
        let topology=ClothContactTopology::new(triangles);let mapping=ClothContactMapping::new(&drivers);
        #[cfg(test)] let profile=std::env::var_os("ZESTENGINE_CLOTH_PROFILE").is_some();
        #[cfg(test)] let mut contact_time=std::time::Duration::ZERO;
        #[cfg(test)] let mut body_time=std::time::Duration::ZERO;
        // Shared and body constraints can oppose one another (a skirt pushed
        // into its body by a coat). Allow bounded convergence, with an early
        // exit once the coupled corrections fall below the contact tolerance.
        let tolerance=radii.iter().copied().filter(|r|*r>0.0).fold(f32::INFINITY,f32::min)*0.005;
        for pass in 0..passes*4 {
            #[cfg(test)] let started=std::time::Instant::now();
            let before=positions.clone();
            let report=topology.solve_between_mapped(&mut positions,&previous,&weights,&owners,&radii,&mapping,continuous,untangle);
            // The next pass sweeps body reprojection from this corrected pose.
            // Reusing the original frame pose forever also reuses its contact
            // barycentrics/normals, which can leave a new closest feature thin
            // even when every old constraint reports zero residual.
            previous.clone_from(&positions);
            #[cfg(test)] {contact_time+=started.elapsed();}
            #[cfg(test)] let started=std::time::Instant::now();
            let mut residual=positions.iter().zip(&before).map(|(a,b)|a.distance_squared(*b)).fold(0.0,f32::max);
            for ((object,slot,world),span) in entries.iter().zip(&spans){
                if positions[span.clone()]==before[span.clone()]{continue;}
                let mut c=cloth_mut(objects,*object,*slot).unwrap();let mut sim=c.runtime.sim.take().unwrap();
                let mut shapes:Vec<_>=c.runtime.collider_history.iter().collect();shapes.sort_by(|a,b|a.0.cmp(b.0));
                let shapes:Vec<_>=shapes.into_iter().map(|(_,s)|s.clone()).collect();
                sim.apply_render_contact_positions(&c,&positions[span.clone()],*world,&shapes,dt);
                let repaired=sim.collision_render_positions(*world);
                residual=residual.max(positions[span.clone()].iter().zip(&repaired).map(|(a,b)|a.distance_squared(*b)).fold(0.0,f32::max));
                positions[span.clone()].copy_from_slice(&repaired);
                c.runtime.sim=Some(sim);
            }
            #[cfg(test)] {body_time+=started.elapsed();}
            let settled=pass>0&&report.untangled==0&&residual<=tolerance*tolerance;
            #[cfg(test)] if profile&&(settled||pass+1==passes*4){eprintln!("cloth group: passes={}, contact_ms={:.3}, body_ms={:.3}, residual_mm={:.4}",pass+1,contact_time.as_secs_f64()*1000.0,body_time.as_secs_f64()*1000.0,residual.sqrt()*1000.0);}
            let _=pass;
            if settled{break;}
        }
    }
}

pub(crate) fn finish(objects:&[SceneObject],work:Vec<ClothRenderWork>,mut renderer:Option<&mut Renderer>,dt:f32,patch:&mut Vec<f32>)->Vec<(usize,usize,(glam::Vec3,glam::Vec3))>{
    solve_groups(objects,&work.iter().map(|w|(w.object,w.slot,w.world)).collect::<Vec<_>>(),dt);
    let mut bounds=Vec::with_capacity(work.len());
    for w in work {
        let mut simulated=w.animated.clone();
        if let Some(mut c)=cloth_mut(objects,w.object,w.slot).filter(|c|!c.runtime.culled&&!c.runtime.hold)
            && let Some(sim)=&c.runtime.sim {
                let mut positions:Vec<_>=simulated.iter().map(|v|glam::Vec3::from(v.position)).collect();let mut normals:Vec<_>=simulated.iter().map(|v|glam::Vec3::from(v.normal)).collect();
                sim.apply(&mut positions,&mut normals,w.world);
                for ((v,p),n) in simulated.iter_mut().zip(&positions).zip(normals){v.position=p.to_array();v.normal=n.to_array();}
                if c.runtime.paint_mode{c.runtime.posed=positions;}
            }
        if let Some(mut c)=cloth_mut(objects,w.object,w.slot){
            crate::cloth_cache::apply_playback(&mut c,&mut simulated);
            if !c.runtime.hold&&!c.runtime.culled{crate::cloth_cache::capture(&mut c,&simulated,dt);}
        }
        if !simulated.is_empty(){
            let (lo,hi)=simulated.iter().map(|v|glam::Vec3::from(v.position)).fold((glam::Vec3::splat(f32::INFINITY),glam::Vec3::splat(f32::NEG_INFINITY)),|(lo,hi),p|(lo.min(p),hi.max(p)));
            bounds.push((w.object,w.slot,(lo,hi)));
        }
        if let Some(renderer)=renderer.as_deref_mut().filter(|_|w.handle!=MeshHandle::NONE){
            if w.gpu {let count=zestengine_render::pack_cloth_patch(&w.animated,&simulated,1e-5,patch);renderer.set_mesh_cloth_patch(w.handle,patch,count);}
            else {renderer.update_mesh_vertices(w.handle,&simulated);}
        }
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use zestengine_core::{MeshClothSim,ObjectId};
    fn garment(z:f32,pinned:bool)->SceneObject{
        let mut o=SceneObject::new_empty(ObjectId::new(),"garment");
        let rest=[Vec3::new(0.0,0.0,z),Vec3::new(1.0,0.0,z),Vec3::new(0.0,1.0,z)];
        let mut c=MeshCloth{thickness:0.04,..Default::default()};
        c.runtime.sim=MeshClothSim::build(&rest,&[0,1,2],&[if pinned{1.0}else{0.0};3],0.0,Mat4::IDENTITY);
        o.components.push(Box::new(c));o
    }
    #[test]
    fn independent_garments_share_contacts_in_the_same_frame_and_respect_groups(){
        let objects=vec![garment(0.0,true),garment(0.01,false)];
        let entries=[(0,0,Mat4::IDENTITY),(1,0,Mat4::IDENTITY)];
        solve_groups(&objects,&entries,1.0/60.0);
        assert!(cloth(&objects,0,0).unwrap().runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z==0.0));
        assert!(cloth(&objects,1,0).unwrap().runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.0399));
        let objects=vec![garment(0.0,true),garment(0.01,false)];cloth_mut(&objects,1,0).unwrap().collision_group=2;
        solve_groups(&objects,&entries,1.0/60.0);
        assert!(cloth(&objects,1,0).unwrap().runtime.sim.as_ref().unwrap().particles.iter().all(|p|(p.pos.z-0.01).abs()<1e-6));
    }
    #[test]
    fn reversing_scene_scan_order_preserves_the_shared_result(){
        let a=vec![garment(0.0,true),garment(0.01,false)];
        let b=vec![garment(0.0,true),garment(0.01,false)];for i in 0..2{b[i].components.set_detached_id(a[i].id());}
        solve_groups(&a,&[(0,0,Mat4::IDENTITY),(1,0,Mat4::IDENTITY)],1.0/60.0);
        solve_groups(&b,&[(1,0,Mat4::IDENTITY),(0,0,Mat4::IDENTITY)],1.0/60.0);
        let points=|o:&[SceneObject]|cloth(o,1,0).unwrap().runtime.sim.as_ref().unwrap().particles.iter().map(|p|p.pos).collect::<Vec<_>>();
        assert_eq!(points(&a),points(&b));
    }
}
