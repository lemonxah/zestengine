//! Compose reversible water-carving and island layers over authored terrain.
//! Shared by editor/render hosts and headless physics; the saved base grid is
//! never modified. Removing/disabling a brush restores the underlying layer.
use std::hash::{Hash,Hasher};
use glam::{Mat4,Vec2,Vec3};
use zestengine_core::{Terrain, TerrainWaterLayer, Water, WaterIsland, WaterTerrainBrush, WaterTerrainBlend, ObjectId, ObjectRef, component_enabled};
use crate::scene::LoadedScene;
struct Brush {water:Water, world:Mat4, inverse:Mat4, target:ObjectRef, blend:WaterTerrainBlend,falloff:f32,depth:f32,offset:f32,key:u64,order:(bool,i32,u128)}
fn floats(hash:&mut impl Hasher, values:impl IntoIterator<Item=f32>){for v in values{v.to_bits().hash(hash);}}

pub fn sync(scene:&mut LoadedScene)->Vec<ObjectId> {
    let world=scene.world_transforms();let active=scene.active_flags();let mut brushes=Vec::new();
    for (i,object) in scene.objects.iter().enumerate().filter(|(i,_)|active[*i]) {
        if !world[i].is_finite() || world[i].determinant().abs()<1e-8 {continue;}
        let water=object.components.get_typed::<Water>();
        for component in object.components.iter().filter(|c|component_enabled(c.as_ref())) {
            let (footprint,target,blend,falloff,depth,offset,order,island)=if let Some(brush)=component.as_any().downcast_ref::<WaterTerrainBrush>() {
                let Some(ref w)=water else {continue};
                (Water {body:w.body,custom_mesh:w.custom_mesh.clone(),size:w.size,shape:w.shape.clone(),smooth_shape:w.smooth_shape,river:w.river.clone(),depth:w.depth,current:w.current,resolved_connections:w.resolved_connections,
                    waves:Vec::new(),river_cache:w.river_cache.clone(),outline_cache:w.outline_cache.clone(),..Default::default()},
                    brush.terrain,brush.blend,brush.falloff,brush.depth_scale,brush.bed_offset,brush.order,false)
            }else if let Some(island)=component.as_any().downcast_ref::<WaterIsland>() {
                (island.footprint(),island.terrain,WaterTerrainBlend::Raise,island.falloff,0.0,0.0,island.order,true)
            }else{continue};
            let mut h=std::collections::hash_map::DefaultHasher::new();
            zestengine_render::water_coverage_key(&footprint).hash(&mut h);floats(&mut h,footprint.size);footprint.depth.to_bits().hash(&mut h);
            floats(&mut h,world[i].to_cols_array());floats(&mut h,[falloff,depth,offset]);
            object.id().raw().hash(&mut h);target.id().map(|id|id.raw()).hash(&mut h);(blend as u8).hash(&mut h);order.hash(&mut h);island.hash(&mut h);
            brushes.push(Brush {water:footprint,world:world[i],inverse:world[i].inverse(),target,blend,falloff:falloff.max(0.0),depth:depth.max(0.0),offset,
                key:h.finish(),order:(island,order,object.id().raw())});
        }
    }
    brushes.sort_by_key(|b|b.order);
    let mut changed=Vec::new();
    for (i,object) in scene.objects.iter_mut().enumerate() {
        let Some(mut terrain)=object.components.get_typed_mut::<Terrain>() else {continue};
        let relevant:Vec<_>=brushes.iter().filter(|b|b.target.id().is_none_or(|id|id==object.id())).collect();
        if !terrain.water_layers || relevant.is_empty() || !active[i] || world[i].determinant().abs()<1e-8 {
            if terrain.water_layer.take().is_some(){changed.push(object.id());}continue;
        }
        let n=terrain.resolution as usize+1;
        if terrain.heights.len()!=n*n || terrain.size<=0.0 {terrain.water_layer=None;continue;}
        let mut hash=std::collections::hash_map::DefaultHasher::new();
        floats(&mut hash,terrain.heights.iter().copied());floats(&mut hash,world[i].to_cols_array());terrain.size.to_bits().hash(&mut hash);terrain.resolution.hash(&mut hash);
        for b in &relevant{b.key.hash(&mut hash);}let key=hash.finish();
        if terrain.water_layer.as_ref().is_some_and(|layer|layer.key==key){continue;}
        let inverse=world[i].inverse();let mut heights=terrain.heights.clone();
        for z in 0..n {for x in 0..n {
            let index=z*n+x;let local=Vec3::new((x as f32/terrain.resolution.max(1) as f32-0.5)*terrain.size,terrain.heights[index],(z as f32/terrain.resolution.max(1) as f32-0.5)*terrain.size);
            let at=world[i].transform_point3(local);
            for brush in &relevant {
                let p=brush.inverse.transform_point3(at);
                let Some((column,bank))=brush.water.shore_sample(Vec2::new(p.x,p.z))else{continue};
                let weight=zestengine_core::water_terrain::terrain_brush_weight(bank,brush.falloff);
                if weight<=0.0{continue;}
                let target=brush.world.transform_point3(Vec3::new(p.x,column.surface-column.depth*brush.depth+brush.offset,p.z));
                let target=inverse.transform_point3(target).y;
                if target.is_finite(){heights[index]=zestengine_core::water_terrain::blend_terrain_height(heights[index],target,weight,brush.blend);}
            }
        }}
        terrain.water_layer=Some(TerrainWaterLayer {key,heights});changed.push(object.id());
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{Component,TerrainCollider};
    fn add(scene:&mut LoadedScene,at:Vec3,components:Vec<Box<dyn Component>>)->usize {
        let mut object=crate::scene::SceneObject::new_empty(ObjectId::new(),"terrain layer test");object.transform_mut().translation=at;object.components.replace_all(components);
        let index=scene.objects.len();scene.objects.push(object);index
    }
    #[test]
    fn carving_is_reversible_targeted_and_islands_apply_last(){
        let mut scene=LoadedScene::empty();
        let terrain=Terrain {size:20.0,resolution:20,heights:vec![3.0;441],..Default::default()};
        let t=add(&mut scene,Vec3::ZERO,vec![Box::new(terrain.clone()),Box::new(TerrainCollider::default())]);
        let other=add(&mut scene,Vec3::ZERO,vec![Box::new(terrain)]);
        let target=ObjectRef::new(scene.objects[t].id());
        let water=add(&mut scene,Vec3::new(0.0,1.0,0.0),vec![Box::new(Water {size:[8.0;2],depth:2.0,..Default::default()}),Box::new(WaterTerrainBrush {terrain:target,falloff:2.0,..Default::default()})]);
        let island=add(&mut scene,Vec3::new(0.0,2.0,0.0),vec![Box::new(WaterIsland {terrain:target,size:[2.0;2],falloff:1.0,..Default::default()})]);
        let sample=|s:&LoadedScene,i:usize,x:usize,z:usize|s.objects[i].components.get_typed_at::<Terrain>(0).unwrap().effective_heights()[z*21+x];
        assert_eq!(sync(&mut scene),vec![scene.objects[t].id()]);
        assert_eq!(sample(&scene,t,10,10),2.0);assert_eq!(sample(&scene,t,13,10),-1.0);assert_eq!(sample(&scene,t,15,10),1.0);
        assert_eq!(sample(&scene,other,10,10),3.0);assert!(sync(&mut scene).is_empty());
        *scene.objects[island].enabled_mut()=false;sync(&mut scene);assert_eq!(sample(&scene,t,10,10),-1.0);
        scene.objects[water].transform_mut().translation.x=8.0;sync(&mut scene);assert_eq!(sample(&scene,t,10,10),3.0);
        *scene.objects[water].enabled_mut()=false;sync(&mut scene);
        let terrain=scene.objects[t].components.get_typed_at::<Terrain>(0).unwrap();assert!(terrain.water_layer.is_none());assert!(terrain.heights.iter().all(|h|*h==3.0)); drop(terrain);
    }
    #[test]
    fn terrain_collision_follows_layer_edits_and_restores_without_replacing_bodies(){
        use zestengine_core::{PhysicsQuery,RayFilter};
        let mut scene=LoadedScene::empty();
        let t=add(&mut scene,Vec3::ZERO,vec![Box::new(Terrain {size:10.0,resolution:10,heights:vec![3.0;121],..Default::default()}),Box::new(TerrainCollider::default())]);
        let w=add(&mut scene,Vec3::ZERO,vec![Box::new(Water {size:[6.0;2],depth:2.0,..Default::default()}),Box::new(WaterTerrainBrush::default())]);
        let mut physics=crate::physics::PhysicsWorld::build(&scene);
        physics.step();
        let height=|p:&crate::physics::PhysicsWorld|p.raycast(Vec3::new(0.0,10.0,0.0),Vec3::NEG_Y,30.0,&RayFilter::default()).unwrap().point.y;
        assert!((height(&physics)-3.0).abs()<1e-4);
        sync(&mut scene);physics.sync_structure(&scene);physics.step();assert!((height(&physics)+2.0).abs()<1e-4);
        scene.objects[w].transform_mut().translation.y=1.0;sync(&mut scene);physics.sync_structure(&scene);physics.step();assert!((height(&physics)+1.0).abs()<1e-4);
        let mut terrain=scene.objects[t].components.get_typed_at_mut::<Terrain>(0).unwrap();
        let saved=ron::to_string(&*terrain).unwrap();let loaded:Terrain=ron::from_str(&saved).unwrap();
        assert!(loaded.water_layer.is_none());assert!(loaded.heights.iter().all(|h|*h==3.0));
        terrain.water_layers=false; drop(terrain);sync(&mut scene);physics.sync_structure(&scene);physics.step();assert!((height(&physics)-3.0).abs()<1e-4);
    }

}
