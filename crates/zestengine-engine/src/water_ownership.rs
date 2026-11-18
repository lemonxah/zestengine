//! One deterministic overlap order for rendering and every gameplay query.
use glam::Mat4;
use zestengine_core::Water;
use crate::scene::LoadedScene;
pub fn sync(scene:&mut LoadedScene,world:&[Mat4]) {
    let active=scene.active_flags();
    let mut waters:Vec<_>=scene.objects.iter().enumerate().filter(|(i,_)|active[*i]&&world[*i].is_finite()&&world[*i].determinant().abs()>1e-8).filter_map(|(i,o)|{
        let w=o.components.get_typed::<Water>()?;
        let key=(w.ownership_key(),std::cmp::Reverse(o.id().raw()));
        let mut snapshot=Water{body:w.body,custom_mesh:w.custom_mesh.clone(),size:w.size,shape:w.shape.clone(),smooth_shape:w.smooth_shape,river:w.river.clone(),depth:w.depth,current:w.current,
            resolved_connections:w.resolved_connections,river_cache:w.river_cache.clone(),outline_cache:w.outline_cache.clone(),exclusions:w.exclusions.clone(),waves:Vec::new(),..Default::default()};
        snapshot.overlap_priority=w.overlap_priority;
        let padding=w.resolved_waves().iter().map(|w|w.amplitude).sum::<f32>()*2.0+w.ocean_settings().map_or(0.0,|o|o.sanitized().significant_height)*2.0;
        Some((i,key,std::sync::Arc::new(snapshot),padding))
    }).collect();
    waters.sort_by_key(|(_,key,_,_)|std::cmp::Reverse(*key));
    for(i,o)in scene.objects.iter_mut().enumerate(){let id=o.id();for mut w in o.components.iter_typed_mut::<Water>(){
        w.overlaps.clear();if !active[i]{continue;}
        let key=(w.ownership_key(),std::cmp::Reverse(id.raw()));
        for(j,other_key,other,padding)in &waters {if *other_key<=key{break;}
            w.overlaps.push(zestengine_core::water_connection::WaterOverlap{to_target:world[*j].inverse()*world[i],water:other.clone(),wave_padding:*padding});
        }
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use zestengine_core::{WaterBody,ObjectId,WaterExclusionRegion};
    use crate::scene::SceneObject;
    fn water(scene:&LoadedScene,i:usize)->zestengine_core::ecs::ComponentRef<'_, Water>{scene.objects[i].components.get_typed_at::<Water>(0).unwrap()}
    fn water_mut(scene:&mut LoadedScene,i:usize)->zestengine_core::ecs::ComponentMut<'_, Water>{scene.objects[i].components.get_typed_at_mut::<Water>(0).unwrap()}
    #[test]
    fn priority_is_stable_and_dry_or_elevated_owners_preserve_underlying_water(){
        let mut scene=LoadedScene::empty();
        for (name,body)in[("Ocean",WaterBody::Ocean),("Lake",WaterBody::Lake)]{let mut o=SceneObject::new_empty(ObjectId::new(),name);o.components.push(Box::new(Water{body,size:[6.0;2],waves:vec![],depth:2.0,..Default::default()}));scene.objects.push(o);}
        sync(&mut scene,&[Mat4::IDENTITY;2]);assert!(water(&scene,0).unavailable(Vec3::ZERO));assert!(!water(&scene,1).unavailable(Vec3::ZERO));
        assert!(!water(&scene,0).unavailable(Vec3::new(4.0,0.0,0.0)));assert!(!water(&scene,0).unavailable(Vec3::new(0.0,-3.0,0.0)));
        water_mut(&mut scene,1).exclusions.push(WaterExclusionRegion{to_unit:Mat4::IDENTITY});sync(&mut scene,&[Mat4::IDENTITY;2]);assert!(!water(&scene,0).unavailable(Vec3::ZERO));
        water_mut(&mut scene,1).exclusions.clear();scene.objects[1].transform_mut().translation.y=5.0;let world=scene.world_transforms();sync(&mut scene,&world);assert!(!water(&scene,0).unavailable(Vec3::ZERO));
        scene.objects[1].transform_mut().translation.y=0.0;water_mut(&mut scene,0).overlap_priority=10;sync(&mut scene,&[Mat4::IDENTITY;2]);assert!(water(&scene,1).unavailable(Vec3::ZERO));
        *scene.objects[0].enabled_mut()=false;sync(&mut scene,&[Mat4::IDENTITY;2]);assert!(!water(&scene,1).unavailable(Vec3::ZERO));
        *scene.objects[0].enabled_mut()=true;water_mut(&mut scene,0).overlap_priority=0;water_mut(&mut scene,0).body=WaterBody::Lake;sync(&mut scene,&[Mat4::IDENTITY;2]);
        let owner=if water(&scene,0).unavailable(Vec3::ZERO){scene.objects[1].id()}else{scene.objects[0].id()};scene.objects.swap(0,1);sync(&mut scene,&[Mat4::IDENTITY;2]);
        let owner2=if water(&scene,0).unavailable(Vec3::ZERO){scene.objects[1].id()}else{scene.objects[0].id()};assert_eq!(owner,owner2);
    }
}
