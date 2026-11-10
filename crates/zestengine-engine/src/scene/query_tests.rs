use super::*;
use zestengine_core::{AudioListener, AudioSource, Disabled, Health, LightComponent, ReflectionProbe};

fn object(name: &str, components: Vec<Box<dyn Component>>) -> SceneObject {
    let mut object = SceneObject::new_empty(ObjectId::new(), name);
    object.components.replace_all(components);
    object
}

#[test]
fn typed_extraction_keeps_scene_order_and_first_enabled_instance_after_edits() {
    let mut scene = LoadedScene::empty();
    scene.objects.push(object("A", vec![Box::new(LightComponent { intensity: 1.0, ..Default::default() })]));
    scene.objects.push(object("Plain", vec![Box::new(Health::default())]));
    scene.objects.push(object("B", vec![Box::new(Health::default()), Box::new(Disabled(Box::new(LightComponent::default()))),
        Box::new(LightComponent { intensity: 2.0, ..Default::default() }), Box::new(LightComponent { intensity: 3.0, ..Default::default() })]));
    let a = scene.objects[0].id();
    let b = scene.objects[2].id();
    let values = |scene: &LoadedScene| scene.gather_lights_all().iter().map(|light| light.intensity).collect::<Vec<_>>();
    assert_eq!(values(&scene), [1.0, 2.0]);
    scene.objects.swap(0, 2);
    assert_eq!(scene.index_of(a), Some(2));
    assert_eq!(scene.index_of(b), Some(0));
    assert_eq!(values(&scene), [2.0, 1.0]);
    scene.objects[0].components.set_enabled(2, false);
    assert_eq!(values(&scene), [3.0, 1.0]);
    let renamed = ObjectId::new();
    let entity = scene.objects.world.entity(b).unwrap();
    scene.objects.set_id(0, renamed).unwrap();
    assert_eq!(scene.index_of(b), None);
    assert_eq!(scene.index_of(renamed), Some(0));
    assert_eq!(scene.objects.world.entity(renamed), Some(entity));
    scene.objects.world.with_world(|world| {
        world.get_mut::<LightComponent>(entity).unwrap().intensity = 8.0;
    });
    assert_eq!(values(&scene), [8.0, 1.0], "renderer reads native values directly");
    scene.remove_object(1);
    assert_eq!(scene.index_of(a), Some(1));
    assert_eq!(values(&scene), [8.0, 1.0]);
}

#[test]
fn audio_and_probe_queries_obey_activation_parent_space_and_first_listener_order() {
    let mut scene = LoadedScene::empty();
    let mut first = object("First", vec![Box::new(AudioListener::default()), Box::new(AudioSource::default()), Box::new(ReflectionProbe::default())]);
    first.transform_mut().translation = Vec3::new(1.0, 2.0, 3.0);
    first.transform_mut().rotation = Quat::from_rotation_y(0.5);
    let mut child = object("Child", vec![Box::new(AudioListener::default()), Box::new(ReflectionProbe::default())]);
    child.set_parent(Some(0));
    child.transform_mut().translation = Vec3::Y;
    scene.objects.extend([first, child, object("Last", vec![Box::new(AudioListener::default())])]);
    assert_eq!(scene.audio_listener(), Some(Vec3::new(1.0, 2.0, 3.0)));
    assert_eq!(scene.reflection_probe_captures().len(), 2);
    let cues = scene.gather_audio();
    assert_eq!(cues.len(), 1);
    assert!(cues[0].rotation.abs_diff_eq(Quat::from_rotation_y(0.5), 1e-5));
    *scene.objects[0].enabled_mut() = false;
    assert_eq!(scene.audio_listener(), Some(Vec3::ZERO));
    assert!(scene.gather_audio().is_empty());
    assert!(scene.reflection_probe_captures().is_empty());
    *scene.objects[0].enabled_mut() = true;
    scene.objects.swap(0, 2);
    scene.objects[1].set_parent(Some(2));
    assert_eq!(scene.audio_listener(), Some(Vec3::ZERO));
    let captures = scene.reflection_probe_captures();
    assert!(captures[0].1.abs_diff_eq(Vec3::new(1.0, 3.0, 3.0), 1e-5));
}

#[test]
#[ignore = "CPU ECS workload benchmark: run with --ignored --nocapture"]
fn sparse_component_extraction_workload() {
    use std::{hint::black_box, time::Instant};
    let mut scene = LoadedScene::empty();
    for i in 0..8192 {
        let mut object = object("Object", vec![Box::new(Health::default())]);
        if i % 64 == 0 { object.components.push(Box::new(LightComponent { intensity: i as f32, ..Default::default() })); }
        scene.objects.push(object);
    }
    let scan = || scene.objects.iter().filter_map(|object| {
        object.components.get_typed::<LightComponent>().map(|light| (object.id(), light.intensity))
    }).collect::<Vec<_>>();
    let query = || scene.objects.map_components::<LightComponent, _>(|i, light, _ecs| Some((scene.objects[i].id_in(_ecs), light.intensity)));
    assert_eq!(scan(), query());
    let frames = 500;
    let started = Instant::now();
    for _ in 0..frames { black_box(scan()); }
    let scan_time = started.elapsed();
    let started = Instant::now();
    for _ in 0..frames { black_box(query()); }
    let query_time = started.elapsed();
    eprintln!("8192 objects / 128 lights / {frames} extractions: object scan {:.3} ms, typed query {:.3} ms ({:.2}x)",
        scan_time.as_secs_f64() * 1000.0, query_time.as_secs_f64() * 1000.0, scan_time.as_secs_f64() / query_time.as_secs_f64());
}

#[test]
fn mirror_and_water_layers_share_scene_order_deduplicate_instances_and_respect_capacity() {
    use zestengine_core::{PlanarReflection, SurfaceReflection, Water};
    let mut scene = LoadedScene::empty();
    scene.objects.push(object("Water", vec![Box::new(Water { reflection: SurfaceReflection::Planar, ..Default::default() }),
        Box::new(Water { reflection: SurfaceReflection::Planar, ..Default::default() })]));
    let mut mirror = object("Mirror", vec![Box::new(PlanarReflection { max_roughness: 0.25 })]);
    *mirror.render_mut() = Some(RenderInfo { mesh: 0, material: 0 });
    mirror.transform_mut().translation.x = 5.0;
    scene.objects.push(mirror);
    let surfaces = scene.planar_surfaces();
    assert_eq!(surfaces.len(), 2, "duplicate water components use one object layer");
    assert_eq!(surfaces[0].point[0], 5.0);
    assert_eq!(surfaces[0].max_roughness, 0.25);
    assert_eq!(scene.water_planar_index(0), 1);
    scene.objects[1].components.set_enabled(0, false);
    assert_eq!(scene.water_planar_index(0), 0);
    assert_eq!(scene.planar_surfaces().len(), 1);
    for _ in 0..zestengine_render::MAX_PLANAR {
        let mut mirror = object("Mirror", vec![Box::new(PlanarReflection::default())]);
        *mirror.render_mut() = Some(RenderInfo { mesh: 0, material: 0 });
        scene.objects.push(mirror);
    }
    assert_eq!(scene.planar_surfaces().len(), zestengine_render::MAX_PLANAR);
    assert_eq!(scene.water_planar_index(0), -1, "mirror layers occupy the full capacity");
}

#[test]
fn volume_query_keeps_priority_weight_and_local_transform_rules() {
    use zestengine_core::VolumeComponent;
    let mut scene = LoadedScene::empty();
    let mut global = zestengine_assets::PostFxProfile::default();
    global.ambient_occlusion.enabled = true;
    global.ambient_occlusion.intensity = 1.0;
    let mut local = global;
    local.ambient_occlusion.intensity = 4.0;
    scene.postfx_cache.insert("global.postfx".into(), global);
    scene.postfx_cache.insert("local.postfx".into(), local);
    scene.objects.push(object("Global", vec![Box::new(VolumeComponent { profile: "global.postfx".into(), global: true, priority: 0.0, ..Default::default() })]));
    let mut local = object("Local", vec![Box::new(VolumeComponent { profile: "local.postfx".into(), global: false, priority: 10.0, half_extents: [1.0; 3], ..Default::default() })]);
    local.transform_mut().translation.x = 10.0;
    scene.objects.push(local);
    assert_eq!(scene.effective_postfx(Vec3::X * 10.0, Path::new("."), 0.0).ao.intensity, 4.0);
    assert_eq!(scene.postfx_overrides.iter().map(|(name, _, _)| name.as_str()).collect::<Vec<_>>(), ["Global", "Local"]);
    assert_eq!(scene.effective_postfx(Vec3::X * 20.0, Path::new("."), 0.0).ao.intensity, 1.0);
    scene.objects[1].components.set_enabled(0, false);
    assert_eq!(scene.effective_postfx(Vec3::X * 10.0, Path::new("."), 0.0).ao.intensity, 1.0);
}
