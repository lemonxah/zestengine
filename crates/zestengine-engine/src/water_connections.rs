//! Resolve endpoint links against active bodies without modifying authored
//! spline points. Recursive resolution supports chains and rejects cycles.
use glam::{Mat4,Vec2,Vec3,Vec4};
use zestengine_core::{Water,WaterBody,ObjectId,ResolvedWaterConnection};
use crate::scene::LoadedScene;

pub fn sync(scene:&mut LoadedScene) {
    crate::water_custom::sync(scene);
    if !scene.objects.iter().flat_map(|o|o.components.iter_typed::<Water>()).any(|w|w.connections.iter().any(|c|c.target.id().is_some())||w.resolved_connections.iter().any(Option::is_some)){return;}
    let active=scene.active_flags();let world=scene.world_transforms();
    let waters:Vec<_>=scene.objects.iter().enumerate().map(|(i,o)|{
        if !active[i]{return None;}
        o.components.get_typed::<Water>().map(|water| water.surface_snapshot())
    }).collect();
    let ids:Vec<_>=scene.objects.iter().map(|o|o.id()).collect();let mut resolved=vec![None;waters.len()];let mut cycles=vec![false;waters.len()];
    for i in 0..waters.len(){resolve(i,&waters,&ids,&world,&mut resolved,&mut Vec::new(),&mut cycles);}
    for(i,o)in scene.objects.iter_mut().enumerate(){for mut w in o.components.iter_typed_mut::<Water>(){
        w.resolved_connections=resolved[i].as_ref().map_or([None;2],|w|w.resolved_connections);
        w.connected_surfaces=resolved[i].as_ref().map_or([None,None],|w|w.connected_surfaces.clone());
    }}
}

fn resolve(i:usize,waters:&[Option<Water>],ids:&[ObjectId],world:&[Mat4],resolved:&mut[Option<Water>],stack:&mut Vec<usize>,cycles:&mut[bool])->Option<Water> {
    if let Some(w)=&resolved[i]{return Some(w.clone());}
    if let Some(start)=stack.iter().position(|&j|j==i){for &j in &stack[start..]{cycles[j]=true;}return None;}
    let mut water=waters[i].as_ref()?.clone();
    if water.body!=WaterBody::River || water.river.len()<2 || !world[i].is_finite() || world[i].determinant().abs()<1e-8{resolved[i]=Some(water.clone());return Some(water);}
    stack.push(i);
    for end in 0..2 {
        let connection=water.connections[end];let Some(target)=connection.target.id().and_then(|id|ids.iter().position(|&j|j==id))else{continue;};
        let Some(other)=resolve(target,waters,ids,world,resolved,stack,cycles)else{continue;};
        if !world[target].is_finite() || world[target].determinant().abs()<1e-8{continue;}
        let point=water.river[if end==0{0}else{water.river.len()-1}];
        let to_target=world[target].inverse()*world[i];let p=to_target.transform_point3(point.position.into());
        let Some(column)=other.column(Vec2::new(p.x,p.z))else{continue;};
        let plane=to_target.transpose()*Vec4::new(0.0,1.0,0.0,-column.surface);
        if plane.y.abs()<1e-6{continue;}
        let to_self=world[i].inverse()*world[target];
        water.resolved_connections[end]=Some(ResolvedWaterConnection{plane:plane.to_array(),flow:to_self.transform_vector3(column.flow),
            depth:column.depth*to_self.transform_vector3(Vec3::Y).length(),length:if connection.length.is_finite(){connection.length.clamp(0.01,10000.0)}else{5.0}});
        water.connected_surfaces[end]=Some(zestengine_core::water_connection::WaterSurfaceConnection{to_target,water:std::sync::Arc::new(other)});
    }
    stack.pop();if cycles[i]{water.resolved_connections=[None;2];water.connected_surfaces=[None,None];}
    resolved[i]=Some(water.clone());Some(water)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    #[test]
    fn connected_waves_follow_transformed_receiver_and_cycles_restore_authored_water(){
        let mut scene=LoadedScene::empty();let mut target=SceneObject::new_empty(ObjectId::new(),"Target");
        target.transform_mut().translation=Vec3::new(2.0,3.0,1.0);target.transform_mut().scale=Vec3::splat(1.4);target.transform_mut().rotation=glam::Quat::from_rotation_y(0.7);
        target.components.push(Box::new(Water{size:[100.0;2],shore_depth:2.0,shore:Some(std::sync::Arc::new(zestengine_core::ShoreDepth::new(4,20.0,vec![0.6;16]))),..Default::default()}));
        let mut river=SceneObject::new_empty(ObjectId::new(),"River");let mut w=Water{body:WaterBody::River,river:vec![zestengine_core::RiverPoint{position:[-10.0,5.0,0.0],..Default::default()},zestengine_core::RiverPoint{position:[0.0,1.0,0.0],..Default::default()}],..Default::default()};w.connections[1].target=target.id().into();river.components.push(Box::new(w));scene.objects.extend([target,river]);sync(&mut scene);
        let world=scene.world_transforms();let receiving=scene.objects[0].components.get_typed_at::<Water>(0).unwrap();let river=scene.objects[1].components.get_typed_at::<Water>(0).unwrap();
        for t in [0.0,0.4,7.0] {let p=world[0].inverse().transform_point3(Vec3::new(0.0,3.0,0.0));let(d,n)=receiving.displacement(Vec2::new(p.x,p.z),t);let(rd,rn)=river.displacement(Vec2::ZERO,t);
            assert!(rd.distance(world[0].transform_vector3(d))<1e-5);assert!(rn.distance(world[0].inverse().transpose().transform_vector3(n).normalize())<1e-5);} drop(receiving);
        let target_id=scene.objects[0].id();let river_id=scene.objects[1].id();
        drop(river);let mut w=scene.objects[0].components.get_typed_at_mut::<Water>(0).unwrap();w.body=WaterBody::River;w.river=vec![zestengine_core::RiverPoint{position:[-10.0,0.0,0.0],width:50.0,..Default::default()},zestengine_core::RiverPoint{position:[10.0,0.0,0.0],width:50.0,..Default::default()}];w.connections[0].target=river_id.into();drop(w);sync(&mut scene);
        for o in &scene.objects{let w=o.components.get_typed_at::<Water>(0).unwrap();assert!(w.resolved_connections.iter().all(Option::is_none),"cycle on {:?}",o.id());assert!(w.connected_surfaces.iter().all(Option::is_none)); drop(w);}
        assert_ne!(target_id,river_id);
    }
    #[test]
    fn river_joins_follow_receiving_height_flow_and_depth_without_editing_spline(){
        let mut scene=LoadedScene::empty();let mut lake=SceneObject::new_empty(ObjectId::new(),"Lake");
        lake.transform_mut().translation.y=3.0;lake.components.push(Box::new(Water{size:[40.0;2],depth:7.0,current:[0.4,0.0,0.7],..Default::default()}));
        let mut river=SceneObject::new_empty(ObjectId::new(),"River");let mut water=Water{body:WaterBody::River,river:vec![zestengine_core::RiverPoint{position:[-10.0,5.0,0.0],..Default::default()},zestengine_core::RiverPoint{position:[0.0,1.0,0.0],..Default::default()}],..Default::default()};
        let authored=water.river.clone();water.connections[1].target=lake.id().into();river.components.push(Box::new(water));scene.objects.extend([lake,river]);sync(&mut scene);
        let w=scene.objects[1].components.get_typed_at::<Water>(0).unwrap();let end=w.column(Vec2::ZERO).unwrap();assert_eq!(w.river,authored); drop(w);assert!((end.surface-3.0).abs()<1e-5);assert!((end.depth-7.0).abs()<1e-5);assert!(end.flow.distance(Vec3::new(0.4,0.0,0.7))<1e-5);
        scene.objects[0].transform_mut().translation.y=4.0;sync(&mut scene);let w=scene.objects[1].components.get_typed_at::<Water>(0).unwrap();assert!((w.column(Vec2::ZERO).unwrap().surface-4.0).abs()<1e-5); drop(w);
        *scene.objects[0].enabled_mut()=false;sync(&mut scene);let w=scene.objects[1].components.get_typed_at::<Water>(0).unwrap();assert_eq!(w.column(Vec2::ZERO).unwrap().surface,1.0); drop(w);
    }
}
