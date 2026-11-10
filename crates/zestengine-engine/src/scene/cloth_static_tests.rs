use super::*;
use zestengine_assets::{Instance as AssetInstance,Scene as AssetScene,SceneMaterial};
use zestengine_core::{ClothPaint,PinSource};
use zestengine_render::Vertex;

#[cfg(all(feature="client",target_os="linux"))]
#[path="cloth_static_gpu.rs"]
mod gpu;

fn sheet()->MeshData{
    let vertices=[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0]].into_iter().enumerate().map(|(i,position)|Vertex{
        position,normal:[0.0,0.0,1.0],uv:[i as f32*0.13,0.37],uv1:[0.21,i as f32*0.12],color:[1.0,0.2,0.4,0.8],tangent:[1.0,0.0,0.0,-1.0],..Default::default()
    }).collect();
    MeshData{vertices,indices:vec![0,1,2,1,3,2],has_lightmap_uv:true}
}
fn asset()->AssetScene{
    AssetScene{meshes:vec![sheet()],morphs:vec![],springs:Default::default(),textures:vec![],materials:vec![SceneMaterial{name:"Fabric".into(),params:Default::default(),features:Default::default(),albedo:None,normal:None,orm:None,emission:None}],
        instances:vec![AssetInstance::single("flag",0,0,Mat4::IDENTITY),AssetInstance::single("rigid copy",0,0,Mat4::from_translation(Vec3::X*3.0))],
        skeletons:vec![],animations:vec![],collision:vec![],identity:Default::default()}
}
fn loaded(asset:&AssetScene)->LoadedScene{let mut s=LoadedScene::empty();s.add_asset_scene_with(None,asset,None).unwrap();s}
fn cloth(s:&LoadedScene,i:usize)->zestengine_core::ecs::ComponentRef<'_, MeshCloth>{s.objects[i].components.get_typed::<MeshCloth>().unwrap()}
fn cloth_mut(s:&mut LoadedScene,i:usize)->zestengine_core::ecs::ComponentMut<'_, MeshCloth>{s.objects[i].components.get_typed_mut::<MeshCloth>().unwrap()}

#[test]
fn static_flag_runs_without_skeleton_and_does_not_deform_other_instances(){
    let asset=asset();let mut s=loaded(&asset);
    s.objects[1].components.push(Box::new(MeshCloth{pin:PinSource::Top,gravity:[0.0,0.0,-9.81],..Default::default()}));
    assert!(s.skeletons.is_empty()&&s.skinned_meshes.is_empty());
    for _ in 0..30{s.pose_skeletons(1.0/60.0);}
    let c=cloth(&s,1);let sim=c.runtime.sim.as_ref().expect("static cloth advances in the scene pass");
    assert!(sim.particles.iter().filter(|p|p.pin==1.0).all(|p|p.pos.z.abs()<1e-6));
    assert!(sim.particles.iter().any(|p|p.pos.z< -0.01)); drop(c);
    assert_eq!(s.static_cloth.instances.len(),1,"only the flag owns a deformable instance");
    assert!(s.mesh_geometry[0].0.iter().all(|p|p.z==0.0),"shared imported geometry stays rigid");
    let data=s.static_cloth_source(0).unwrap();
    for (got,want) in data.vertices.iter().zip(&asset.meshes[0].vertices){
        assert_eq!(got.uv,want.uv);assert_eq!(got.uv1,want.uv1);assert_eq!(got.color,want.color);assert_eq!(got.tangent,want.tangent);
    }
    assert!(data.has_lightmap_uv);assert_eq!(s.objects[1].render().unwrap().material,s.objects[2].render().unwrap().material);
    let (lo,hi)=s.static_cloth.bounds[&(s.objects[1].id(),0)];assert!(lo.z< -0.01&&hi.z.abs()<1e-5,"bounds follow deformed vertices");
    assert!(s.static_cloth.bounds.get(&(s.objects[2].id(),0)).is_none());
    s.sync_draws(None,0.0);
    let draw=s.draws.iter().find(|d|d.gi_source.is_some_and(|source|source.object==s.objects[1].id().raw())).unwrap();
    assert!(!draw.static_geometry&&draw.lightmap_layer==-1);assert!(draw.mesh_center[2]<0.0);
}

#[test]
fn static_and_skinned_cloth_collide_in_the_same_frame(){
    use zestengine_assets::{Joint,Skeleton};
    let mut asset=asset();let mut skinned=sheet();for v in &mut skinned.vertices{v.weights=[1.0,0.0,0.0,0.0];v.color=[0.0;4];}
    asset.meshes.push(skinned);asset.instances[1]=AssetInstance::single("skinned garment",1,0,Mat4::from_translation(Vec3::Z*0.01));
    asset.skeletons.push(Skeleton{joints:vec![Joint{name:"root".into(),parent:None,inverse_bind:Mat4::IDENTITY,rest_translation:Vec3::ZERO,rest_rotation:Quat::IDENTITY,rest_scale:Vec3::ONE}],..Default::default()});
    let mut s=loaded(&asset);
    for i in [1,2]{s.objects[i].components.push(Box::new(MeshCloth{gravity:[0.0;3],thickness:0.04,..Default::default()}));}
    s.pose_skeletons(1.0/60.0);
    assert_eq!(s.skinned_meshes.len(),1);
    assert!(cloth(&s,1).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z==0.0));
    assert!(cloth(&s,2).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.0399));
}

#[test]
fn static_paint_and_extra_slots_survive_serialization_and_component_removal(){
    let mut s=loaded(&asset());let slot=s.objects[1].render().unwrap();(*s.objects[1].extra_render_mut()).push(slot);
    let positions=s.mesh_geometry[0].0.clone();let pins=[1.0,0.0,1.0,0.0];
    let c=MeshCloth{slot:1,pin:PinSource::Painted,paint:ClothPaint::seed(&positions,&s.mesh_geometry[0].1,&pins),gravity:[0.0,0.0,-9.81],..Default::default()};
    let serialized=ron::to_string(&c).unwrap();s.objects[1].components.push(Box::new(ron::from_str::<MeshCloth>(&serialized).unwrap()));
    cloth_mut(&mut s,1).runtime.paint_mode=true;
    for _ in 0..10{s.pose_skeletons(1.0/60.0);}
    let c=cloth(&s,1);assert_eq!(c.runtime.source_pins.len(),0);assert_eq!(c.paint.pin_weights(),pins);
    assert_eq!(c.runtime.posed.len(),4);assert!(c.runtime.posed[1].z<0.0&&c.runtime.posed[0].z==0.0);
    let key=(s.objects[1].id(),1);assert!(s.static_cloth.instances.contains_key(&key));assert!(!s.static_cloth.instances.contains_key(&(key.0,0)));
    drop(c);s.objects[1].components.retain(|c|!c.as_any().is::<MeshCloth>());s.pose_skeletons(1.0/60.0);
    assert!(s.static_cloth.instances.is_empty());assert!(s.static_cloth.bounds.is_empty());assert!(s.static_cloth_source(0).is_none());
}

#[test]
fn static_cloth_replacement_rebuilds_even_with_the_same_vertex_count(){
    let mut s=loaded(&asset());s.objects[1].components.push(Box::new(MeshCloth{pin:PinSource::Top,..Default::default()}));
    s.pose_skeletons(1.0/60.0);let before=cloth(&s,1).runtime.source_key;
    let mut replacement=sheet();for v in &mut replacement.vertices{v.position[0]*=2.0;v.uv[0]+=0.5;}
    // Same registration path used after importer replacement.
    s.static_cloth.imported(0,&replacement,None,0);
    s.mesh_geometry[0]=(replacement.vertices.iter().map(|v|Vec3::from(v.position)).collect(),replacement.indices.clone());
    s.pose_skeletons(1.0/60.0);
    assert_ne!(before,cloth(&s,1).runtime.source_key);assert_eq!(cloth(&s,1).runtime.bind[1].x,2.0);
    assert_eq!(s.static_cloth_source(0).unwrap().vertices[1].uv,replacement.vertices[1].uv);
}

#[test]
fn unavailable_static_sources_report_an_error_instead_of_using_incomplete_vertices(){
    let mut s=loaded(&asset());s.static_cloth.sources.clear();s.objects[1].components.push(Box::new(MeshCloth::default()));
    s.pose_skeletons(1.0/60.0);
    assert!(cloth(&s,1).runtime.sim.is_none());assert!(cloth(&s,1).runtime.build_error.contains("vertex source"));
    assert!(s.static_cloth.instances.is_empty());
}

struct Project(PathBuf);
impl Project{
    fn new()->Self{let p=std::env::temp_dir().join(format!("zest_static_cloth_{}",ObjectId::new()));std::fs::create_dir_all(&p).unwrap();Self(p)}
    fn write_flag(&self,width:f32){
        let mut bytes=Vec::new();
        for p in [[0.0f32,0.0,0.0],[width,0.0,0.0],[0.0,1.0,0.0],[width,1.0,0.0]]{for x in p{bytes.extend(x.to_le_bytes());}}
        for _ in 0..4{for x in [0.0f32,0.0,1.0]{bytes.extend(x.to_le_bytes());}}
        for uv in [[0.0f32,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]{for x in uv{bytes.extend(x.to_le_bytes());}}
        for i in [0u32,1,2,1,3,2]{bytes.extend(i.to_le_bytes());}
        assert_eq!(bytes.len(),152);std::fs::write(self.0.join("flag.bin"),bytes).unwrap();
        let json=r#"{"asset":{"version":"2.0"},"buffers":[{"uri":"flag.bin","byteLength":152}],
            "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":48},{"buffer":0,"byteOffset":96,"byteLength":32},{"buffer":0,"byteOffset":128,"byteLength":24}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[$WIDTH,1,0]},
                {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},{"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},{"bufferView":3,"componentType":5125,"count":6,"type":"SCALAR"}],
            "materials":[{"name":"Fabric"}],"meshes":[{"name":"Flag","primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0}]}],
            "nodes":[{"name":"Flag","mesh":0}],"scenes":[{"nodes":[0]}],"scene":0}"#.replace("$WIDTH",&width.to_string());
        std::fs::write(self.0.join("flag.gltf"),json).unwrap();
    }
}
impl Drop for Project{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
fn flag_entry()->SceneEntry{
    SceneEntry{id:ObjectId::new().to_string(),name:"flag".into(),source:ObjectSource::Model{path:"flag.gltf".into(),mesh:0,extra_meshes:vec![],model:None,mesh_id:None,extra_mesh_ids:vec![]},
        enabled:true,static_geometry:false,lightmap_scale:1.0,cast_shadows:Default::default(),layer:0,material:MaterialRef::Model,extra_materials:vec![],renderer_split:false,parent:None,
        components:vec![zestengine_assets::ComponentData::of(&MeshCloth{pin:PinSource::Top,..Default::default()})],translation:[0.0;3],rotation:[0.0,0.0,0.0,1.0],scale:[1.0;3],model:None,model_part:None,prefab:None,prefab_parent:None,net_role:Default::default(),component_roles:Default::default()}
}

#[test]
fn imported_static_cloth_resolves_project_paths_and_reimports_live_geometry(){
    let project=Project::new();project.write_flag(1.0);
    let mut s=LoadedScene::empty();let registry=ComponentRegistry::with_builtins();let mut shaders=ShaderLibrary::default();
    s.append_entries_with(None,&mut shaders,&project.0,&registry,&[flag_entry()]).unwrap();
    let mesh=s.objects[0].render().unwrap().mesh;
    assert!(s.static_cloth_source(mesh).is_none(),"unrequested models have no duplicate vertex attributes");
    s.pose_skeletons(1.0/60.0);assert!(cloth(&s,0).runtime.build_error.is_empty());
    assert_eq!(s.static_cloth_source(mesh).unwrap().vertices.len(),4);let key=cloth(&s,0).runtime.source_key;
    project.write_flag(2.0);let (replacement,_)=zestengine_assets::load_model_cached(project.0.join("flag.gltf"),true).unwrap();
    assert!(s.reimport_model(None,&mut shaders,&project.0,"flag.gltf",&replacement).unwrap().is_some());
    s.pose_skeletons(1.0/60.0);
    assert!(cloth(&s,0).runtime.build_error.is_empty());assert_ne!(key,cloth(&s,0).runtime.source_key);
    assert!(cloth(&s,0).runtime.bind.iter().any(|p|(p.x-2.0).abs()<1e-5));
    let saved=s.to_scene_file(&project.0,&shaders);
    let mut again=LoadedScene::load_scene_file(None,&saved,&project.0,&registry,&mut shaders).unwrap();
    assert!(again.static_cloth_source(mesh).is_some(),"scene load warms authored cloth from the already parsed model");
    again.pose_skeletons(1.0/60.0);assert!(cloth(&again,0).runtime.sim.is_some());
}

#[test]
fn primitive_mesh_cloth_needs_no_imported_model_or_skeleton(){
    let mut entry=flag_entry();entry.source=ObjectSource::Primitive{shape:zestengine_assets::PrimitiveShape::Plane};
    let file=SceneFile{entries:vec![entry],..Default::default()};
    let mut s=LoadedScene::load_scene_file(None,&file,Path::new("."),&ComponentRegistry::with_builtins(),&mut ShaderLibrary::default()).unwrap();
    s.pose_skeletons(1.0/60.0);assert!(cloth(&s,0).runtime.sim.is_some());assert!(s.skeletons.is_empty());
}

#[test]
fn clothing_asset_placement_adds_runtime_cloth_and_survives_scene_reload(){
    let project=Project::new();let path=project.0.join("flag.cloth");let mut asset=zestengine_assets::cloth_file::ClothingAsset::default();asset.pattern.panels[0].segments=[4,4];asset.settings.gravity=[0.0,0.0,-9.81];
    zestengine_assets::cloth_file::save_clothing(&path,&asset).unwrap();let model=zestengine_assets::load_model_with_meta(&path).unwrap();let mut scene=LoadedScene::empty();scene.add_asset_scene_with(None,&model,Some(Path::new("flag.cloth"))).unwrap();
    assert_eq!(scene.objects[1].components.iter().filter(|c|c.as_any().is::<MeshCloth>()).count(),1);
    for _ in 0..20{scene.pose_skeletons(1.0/60.0);}
    assert!(cloth(&scene,1).runtime.sim.as_ref().unwrap().particles.iter().any(|p|p.pos.z< -0.01));
    assert!(!cloth(&scene,1).garment.contains(&project.0.to_string_lossy().to_string()),"garment reference stays portable");
    let file=scene.to_scene_file(&project.0,&ShaderLibrary::default());
    let mut restored=LoadedScene::load_scene_file(None,&file,&project.0,&ComponentRegistry::with_builtins(),&mut ShaderLibrary::default()).unwrap();restored.pose_skeletons(1.0/60.0);
    assert!(restored.objects.iter().flat_map(|o|o.components.iter_typed::<MeshCloth>()).any(|c|c.runtime.sim.is_some()));
}

#[test]
fn clothing_reimport_updates_seams_and_preserves_instance_settings(){
    use zestengine_core::{ClothPanel,ClothPanelEdge,ClothPanelSide,ClothSeam};
    let project=Project::new();let path=project.0.join("jacket.cloth");let mut asset=zestengine_assets::cloth_file::ClothingAsset::default();
    asset.pattern.panels[0].segments=[3,4];
    asset.pattern.panels.push(ClothPanel{segments:[3,7],translation:[1.2,0.0,0.0],..Default::default()});
    asset.pattern.seams.push(ClothSeam{a:ClothPanelEdge{side:ClothPanelSide::Right,..Default::default()},..Default::default()});
    zestengine_assets::cloth_file::save_clothing(&path,&asset).unwrap();
    let model=zestengine_assets::load_model_with_meta(&path).unwrap();let mut scene=LoadedScene::empty();
    scene.add_asset_scene_with(None,&model,Some(Path::new("jacket.cloth"))).unwrap();
    scene.add_asset_scene_with(None,&model,Some(Path::new("jacket.cloth"))).unwrap();
    assert_eq!(scene.objects[1].render().unwrap().mesh,scene.objects[3].render().unwrap().mesh,"placements share the reimported source");
    cloth_mut(&mut scene,1).fabric.as_mut().unwrap().warp=9876.0;
    scene.pose_skeletons(1.0/60.0);assert!(cloth(&scene,1).runtime.build_error.is_empty());
    let old_key=cloth(&scene,1).runtime.source_key;let old_stitches=cloth(&scene,1).stitches.clone();
    asset.pattern.panels[1].segments=[4,10];zestengine_assets::cloth_file::save_clothing(&path,&asset).unwrap();
    let model=zestengine_assets::load_model_with_meta(&path).unwrap();
    let expected=zestengine_assets::cloth_file::component_from_clothing(&path).unwrap().stitches;
    // The parsed model is the import snapshot, even if its source changes again
    // before the editor installs this result on the main thread.
    asset.pattern.seams.clear();zestengine_assets::cloth_file::save_clothing(&path,&asset).unwrap();
    let mut shaders=ShaderLibrary::default();
    assert!(scene.reimport_model(None,&mut shaders,&project.0,"jacket.cloth",&model).unwrap().is_some());
    assert_ne!(old_stitches,expected);assert_eq!(cloth(&scene,1).stitches,expected);
    assert_eq!(cloth(&scene,3).stitches,expected);
    assert_eq!(cloth(&scene,1).fabric.as_ref().unwrap().warp,9876.0);
    scene.pose_skeletons(1.0/60.0);let c=cloth(&scene,1);
    assert!(c.runtime.build_error.is_empty(),"{}",c.runtime.build_error);assert!(c.runtime.sim.is_some());assert_ne!(old_key,c.runtime.source_key);
    assert_eq!(c.runtime.bind.len(),75); drop(c);
    assert_eq!(cloth(&scene,3).runtime.bind.len(),75);
}

#[test]
fn static_cloth_uses_a_moving_external_level_set_and_releases_disabled_fields(){
    use zestengine_core::{ClothCollider,ClothColliders,ClothShape,ObjectRef};
    use zestengine_core::cloth_sdf::ClothSdf;
    let mut source=asset();for v in &mut source.meshes[0].vertices{v.color=[0.0;4];}
    let mut scene=loaded(&source);let body=scene.objects.len();let id=ObjectId::new();
    let proxy=zestengine_assets::primitive_mesh(zestengine_assets::PrimitiveShape::Cube);
    let points=proxy.vertices.iter().map(|v|Vec3::from(v.position)*Vec3::new(4.0,4.0,0.2)).collect::<Vec<_>>();
    let field=ClothSdf::from_mesh(&points,&proxy.indices,17).unwrap();
    let mut collider=ClothCollider{shape:ClothShape::LevelSet,center:[0.5,0.5,0.0],..Default::default()};collider.level_set.field=Some(field);
    let mut object=SceneObject::new_empty(id,"Level-set body");object.transform_mut().translation.z=-0.2;object.components.push(Box::new(ClothColliders{colliders:vec![collider],..Default::default()}));scene.objects.push(object);
    scene.objects[1].components.push(Box::new(MeshCloth{gravity:[0.0;3],collider_objects:vec![ObjectRef(Some(id))],..Default::default()}));
    scene.pose_skeletons(1.0/60.0);assert!(cloth(&scene,1).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z.abs()<1e-5));
    scene.objects[body].transform_mut().translation.z=0.2;scene.pose_skeletons(1.0/60.0);
    assert!(cloth(&scene,1).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.3099),"moving field must carry the cloth above its surface");
    *scene.objects[body].enabled_mut()=false;scene.pose_skeletons(1.0/60.0);assert!(cloth(&scene,1).runtime.collider_history.is_empty());
    let colliders=scene.objects[body].components.get_typed_at::<ClothColliders>(0).unwrap();let saved=ron::to_string(&*colliders).unwrap();
    let restored:ClothColliders=ron::from_str(&saved).unwrap();assert!(restored.colliders[0].level_set.field.is_some(),"baked samples ship with the component"); drop(colliders);
}

#[test]
fn static_cloth_records_and_replays_through_the_scene_pass(){
    let dir=Project::new();let mut s=loaded(&asset());s.cloth_cache_assets.root=dir.0.clone();
    let mut c=MeshCloth{pin:PinSource::Top,gravity:[0.0,0.0,-9.81],..Default::default()};
    c.cache.asset="flag.clothcache".into();c.cache.frame_rate=10;c.cache.record_seconds=0.2;c.runtime.cache.record_requested=true;
    s.objects[1].components.push(Box::new(c));
    for dt in [0.016,0.07,0.17,0.03]{s.pose_skeletons(dt);}
    let data=zestengine_assets::cloth_cache_file::load_cloth_cache(dir.0.join("flag.clothcache")).unwrap();assert_eq!(data.frames.len(),3);
    assert!(data.frames.last().unwrap().positions.iter().any(|p|p[2]< -0.01));
    let mut c=cloth_mut(&mut s,1);c.cache.playback=true;c.cache.speed=0.0;c.cache.offset=0.2;c.gravity=[0.0,0.0,100.0]; drop(c);
    s.pose_skeletons(0.1);assert!(cloth(&s,1).runtime.cache.playing);
    let want=data.sample(0.2,false).unwrap();let low=want.positions.iter().map(|p|p[2]).fold(f32::INFINITY,f32::min);
    assert!((s.static_cloth.bounds[&(s.objects[1].id(),0)].0.z-low).abs()<1e-5,"cached bounds use recorded vertices, not current gravity");
    let frozen=cloth(&s,1).runtime.sim.as_ref().unwrap().particles.iter().map(|p|p.pos).collect::<Vec<_>>();
    s.pose_skeletons(0.1);assert_eq!(frozen,cloth(&s,1).runtime.sim.as_ref().unwrap().particles.iter().map(|p|p.pos).collect::<Vec<_>>(),"playback bypasses live simulation");
    cloth_mut(&mut s,1).cache.playback=false;s.pose_skeletons(0.1);assert!(!cloth(&s,1).runtime.cache.playing);
    assert_ne!(frozen,cloth(&s,1).runtime.sim.as_ref().unwrap().particles.iter().map(|p|p.pos).collect::<Vec<_>>(),"disabling playback resumes live cloth");
}

#[path="cloth_layer_tests.rs"]
mod layered;
