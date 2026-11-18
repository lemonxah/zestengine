use super::*;
use zestengine_core::{Water,WaterBody};
#[test]
fn custom_water_follows_hidden_mesh_templates_and_self_meshes_headlessly(){
    let mut scene=LoadedScene::empty();let material=scene.ensure_default_material(None).unwrap();let mesh=scene.ensure_primitive_mesh(None,zestengine_assets::PrimitiveShape::Cube).unwrap();
    scene.mesh_geometry[mesh]=(vec![Vec3::ZERO,Vec3::new(2.0,1.0,0.0),Vec3::new(0.0,0.0,2.0)],vec![0,1,2]);
    let source=SceneObject::new_empty(ObjectId::new(),"Hidden water template");*source.enabled_mut()=false;source.transform_mut().translation=Vec3::new(3.0,2.0,0.0);source.transform_mut().scale=Vec3::splat(2.0);*source.render_mut()=Some(RenderInfo{mesh,material});
    let mut object=SceneObject::new_empty(ObjectId::new(),"Custom water");object.components.push(Box::new(Water{body:WaterBody::Custom,mesh_source:source.id().into(),waves:Vec::new(),..Default::default()}));scene.objects.extend([source,object]);
    crate::water_custom::sync(&mut scene);
    let get=|s:&LoadedScene|s.objects[1].components.get_typed_at::<Water>(0).unwrap().custom_mesh.clone().unwrap();let first=get(&scene);
    assert!((first.column(glam::Vec2::new(4.0,1.0)).unwrap()-2.5).abs()<1e-5);assert!(first.column(glam::Vec2::new(6.8,3.8)).is_none());
    crate::water_custom::sync(&mut scene);assert!(std::sync::Arc::ptr_eq(&first,&get(&scene)),"unchanged template reuses its BVH");
    scene.objects[0].transform_mut().translation.y=4.0;crate::water_custom::sync(&mut scene);assert!((get(&scene).column(glam::Vec2::new(4.0,1.0)).unwrap()-4.5).abs()<1e-5);
    *scene.objects[1].render_mut()=Some(RenderInfo{mesh,material});scene.objects[1].components.get_typed_at_mut::<Water>(0).unwrap().mesh_source=zestengine_core::ObjectRef::NONE;
    crate::water_custom::sync(&mut scene);assert!((get(&scene).column(glam::Vec2::new(0.5,0.5)).unwrap()-0.25).abs()<1e-5);
    let w=scene.objects[1].components.get_typed_at::<Water>(0).unwrap();let normal=w.geometry_height_at(Mat4::IDENTITY,Vec3::new(0.5,0.0,0.5),0.0).unwrap().1; drop(w);
    assert!(normal.distance(Vec3::new(-0.5,1.0,0.0).normalize())<1e-4);
}
