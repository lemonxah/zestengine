//! Resolve authored dry regions before physics and render updates. These
//! runtime regions live on Water so all existing water query consumers share
//! the same exclusion rules, including headless games.
use glam::Mat4;
use zestengine_core::{Water, WaterExclusionVolume, component_enabled};
use crate::scene::LoadedScene;

pub fn sync(scene: &mut LoadedScene, world: &[Mat4]) {
    let active = scene.active_flags();
    let regions: Vec<_> = scene.objects.iter().enumerate().filter(|(i, _)| active[*i]).flat_map(|(i, o)| {
        o.components.iter().filter(|c| component_enabled(c.as_ref())).filter_map(|c| {
            let volume = c.as_any().downcast_ref::<WaterExclusionVolume>()?;
            Some((volume.clone(), volume.region(world[i])?))
        }).collect::<Vec<_>>()
    }).collect();
    for (i, object) in scene.objects.iter_mut().enumerate() {
        let id = object.id();
        for mut component in &mut object.components {
            if let Some(water) = component.as_any_mut().downcast_mut::<Water>() {
                water.exclusions.clear();
                water.exclusions.extend(regions.iter().filter(|(v, _)| v.affects(id))
                    .map(|(_, r)| zestengine_core::WaterExclusionRegion { to_unit: r.to_unit * world[i] }));
            }
        }
    }
    crate::water_ownership::sync(scene,world);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Quat};
    use zestengine_core::{Component, ObjectId, ObjectRef};
    fn object(scene: &mut LoadedScene, at: Vec3, components: Vec<Box<dyn Component>>) -> usize {
        let index = scene.objects.len();
        scene.objects.push(crate::scene::SceneObject::from_components(zestengine_core::ecs::ComponentSet::from_vec(components)).with_parent(None).with_extra_render(Vec::new()).with_source(zestengine_assets::ObjectSource::Empty).with_component_roles(Default::default()).with_render(None).with_static_geometry(false).with_layer(0).with_net_role(Default::default()).with_id(ObjectId::new()).with_name(String::new()).with_enabled(true).with_transform(zestengine_core::Transform { translation: at, rotation: Quat::IDENTITY, scale: Vec3::ONE }));
        index
    }
    fn update(scene: &mut LoadedScene) { let world = scene.world_transforms(); sync(scene, &world); }
    fn water(scene: &LoadedScene, i: usize) -> zestengine_core::ecs::ComponentRef<'_, Water> { scene.objects[i].components.get_typed_at(0).unwrap() }

    #[test]
    fn moving_parent_target_filter_disable_and_delete_refresh_all_queries() {
        let mut scene = LoadedScene::empty();
        let a = object(&mut scene, Vec3::ZERO, vec![Box::new(Water::default())]);
        let b = object(&mut scene, Vec3::ZERO, vec![Box::new(Water::default())]);
        let parent = object(&mut scene, Vec3::ZERO, Vec::new());
        let target = scene.objects[a].id();
        let dry = object(&mut scene, Vec3::ZERO, vec![Box::new(WaterExclusionVolume {
            waters: vec![ObjectRef::new(target)], ..Default::default()
        })]);
        scene.objects[dry].set_parent(Some(parent));
        update(&mut scene);
        assert!(water(&scene,a).excluded(Vec3::ZERO));
        assert!(!water(&scene,b).excluded(Vec3::ZERO));
        scene.objects[parent].transform_mut().translation = Vec3::new(3.0,0.0,0.0);
        update(&mut scene);
        assert!(!water(&scene,a).excluded(Vec3::ZERO));
        assert!(water(&scene,a).excluded(Vec3::new(3.0,0.0,0.0)));
        scene.objects[a].transform_mut().translation = Vec3::new(2.0,0.0,0.0);
        update(&mut scene);
        assert!(water(&scene,a).excluded(Vec3::new(1.0,0.0,0.0)));
        assert!(!water(&scene,a).excluded(Vec3::new(3.0,0.0,0.0)));
        scene.objects[dry].components.set_enabled(0, false);
        update(&mut scene);
        assert!(water(&scene,a).exclusions.is_empty());
        scene.objects[dry].components.set_enabled(0, true);
        *scene.objects[parent].enabled_mut() = false;
        update(&mut scene);
        assert!(water(&scene,a).exclusions.is_empty());
        *scene.objects[parent].enabled_mut() = true;
        update(&mut scene);
        assert!(!water(&scene,a).exclusions.is_empty());
        scene.objects.pop();
        update(&mut scene);
        assert!(water(&scene,a).exclusions.is_empty());
    }
}
