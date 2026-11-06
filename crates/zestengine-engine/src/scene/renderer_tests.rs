//! The renderer split on the CPU: legacy scenes load into `Mesh Renderer` components, the
//! components (not loose fields) drive what is drawn and saved, and nothing is duplicated or
//! dropped by load -> save -> load. No GPU: materials stay the model's template materials, so
//! these tests cover the data path; material resolution is covered by the migration golden
//! tests and needs a real display for the visual check.

use std::path::Path;

use glam::Vec3;
use zestengine_assets::{MaterialRef, ObjectSource, PrimitiveShape, SceneEntry, SceneFile};
use zestengine_core::{ComponentRegistry, MaterialSlot, MeshRenderer, RendererData, ShadowCasting, component_enabled, renderer_of};

use super::*;

fn legacy_entry(name: &str, source: ObjectSource) -> SceneEntry {
    SceneEntry {
        id: String::new(),
        name: name.into(),
        source,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: ShadowCasting::On,
        layer: 0,
        material: MaterialRef::Model,
        extra_materials: Vec::new(),
        renderer_split: false,
        parent: None,
        components: Vec::new(),
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: None,
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    }
}

fn cube(name: &str) -> SceneEntry {
    legacy_entry(name, ObjectSource::Primitive { shape: PrimitiveShape::Cube })
}

fn load(file: &SceneFile) -> LoadedScene {
    LoadedScene::load_scene_file(None, file, Path::new("."), &ComponentRegistry::with_builtins(), &mut ShaderLibrary::default()).expect("loads without a renderer")
}

fn renderer_data(o: &SceneObject) -> MeshRenderer {
    let r = o.renderer_any().expect("a renderer");
    MeshRenderer { cast_shadows: r.cast_shadows(), lightmap_scale: r.lightmap_scale(), materials: r.materials().to_vec(), ..Default::default() }
}

#[test]
fn a_legacy_scene_loads_with_the_renderer_in_a_component_and_round_trips_unchanged() {
    let mut a = cube("A");
    a.cast_shadows = ShadowCasting::TwoSided;
    a.lightmap_scale = 2.0;
    a.material = MaterialRef::File("materials/a.material".into());
    let b = cube("B");
    let empty = legacy_entry("Empty", ObjectSource::Empty);
    // Written the old way (loose fields) and read back through the file loader.
    let text = ron::to_string(&SceneFile { entries: vec![a, b, empty], ..Default::default() }).unwrap();
    assert!(text.contains("cast_shadows") && text.contains("lightmap_scale"), "the old form carries loose fields: {text}");
    let file: SceneFile = ron::from_str(&text).unwrap();

    let scene = load(&file);
    let (oa, ob, oe) = (&scene.objects[0], &scene.objects[1], &scene.objects[2]);
    assert_eq!((oa.cast_shadows(), oa.lightmap_scale()), (ShadowCasting::TwoSided, 2.0));
    assert_eq!((ob.cast_shadows(), ob.lightmap_scale()), (ShadowCasting::On, 1.0));
    assert!(oe.renderer_any().is_none(), "an empty never gets a renderer");
    assert_eq!(oa.components.iter().filter(|c| renderer_of(c.as_ref()).is_some()).count(), 1);
    // Without a GPU the slot is kept as authored.
    assert_eq!(renderer_data(oa).materials, vec![MaterialSlot::File("materials/a.material".into())]);

    let saved = scene.to_scene_file(Path::new("."), &ShaderLibrary::default());
    assert_eq!(saved.format_version, zestengine_assets::RENDERER_FORMAT);
    assert!(saved.entries.iter().all(|e| matches!(e.material, MaterialRef::Model) && e.extra_materials.is_empty() && e.cast_shadows.is_on() && e.lightmap_scale == 1.0));
    let text = ron::ser::to_string_pretty(&saved, ron::ser::PrettyConfig::default()).unwrap();
    assert!(!text.contains("extra_materials"));

    let again = load(&ron::from_str(&text).unwrap());
    assert_eq!(again.objects.len(), 3, "nothing duplicated or dropped");
    for (x, y) in scene.objects.iter().zip(&again.objects) {
        assert_eq!(x.components.len(), y.components.len(), "{}: no second renderer", x.name);
    }
    assert_eq!(renderer_data(&again.objects[0]), renderer_data(&scene.objects[0]));
    let twice = again.to_scene_file(Path::new("."), &ShaderLibrary::default());
    assert_eq!(ron::to_string(&twice).unwrap(), ron::to_string(&saved).unwrap(), "saving again is stable");
}

#[test]
fn a_mesh_object_without_an_enabled_renderer_is_not_drawn() {
    let file = SceneFile { entries: vec![cube("A"), cube("B")], ..Default::default() };
    let mut scene = load(&file);
    scene.sync_draws(None, 0.0);
    assert_eq!(scene.draws.len(), 2, "both cubes draw");

    // A renderer that is switched off draws nothing but keeps its settings.
    let at = scene.objects[0].components.iter().position(|c| renderer_of(c.as_ref()).is_some()).unwrap();
    zestengine_core::set_component_enabled(&mut scene.objects[0].components[at], false);
    assert!(!component_enabled(scene.objects[0].components[at].as_ref()));
    scene.objects[0].set_cast_shadows(ShadowCasting::Off);
    assert_eq!(scene.objects[0].cast_shadows(), ShadowCasting::Off, "a disabled renderer's settings stay editable");
    scene.sync_draws(None, 0.0);
    assert_eq!(scene.draws.len(), 1);

    // No renderer at all (removed on purpose): the mesh stays (collision, picking) but nothing draws.
    scene.objects[1].components.clear();
    scene.sync_draws(None, 0.0);
    assert_eq!(scene.draws.len(), 0);
    assert!(scene.objects[1].render.is_some(), "the Mesh Filter data is untouched");

    // Adding the renderer back makes it draw again, every slot inheriting the model.
    scene.objects[1].ensure_renderer();
    scene.sync_draws(None, 0.0);
    assert_eq!(scene.draws.len(), 1);
}

#[test]
fn a_renderer_removed_on_purpose_survives_save_and_load() {
    let file = SceneFile { entries: vec![cube("A")], ..Default::default() };
    let mut scene = load(&file);
    scene.objects[0].components.clear();
    let saved = scene.to_scene_file(Path::new("."), &ShaderLibrary::default());
    assert!(saved.entries[0].components.is_empty());
    let again = load(&ron::from_str(&ron::to_string(&saved).unwrap()).unwrap());
    assert!(again.objects[0].renderer_any().is_none(), "a version-2 file is not migrated again");
    let mut again = again;
    again.sync_draws(None, 0.0);
    assert_eq!(again.draws.len(), 0);
}

fn mannequin(tag: &str) -> Option<(std::path::PathBuf, usize)> {
    let sample = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../animations/kaykit/Mannequin_Medium.glb");
    if !sample.exists() {
        eprintln!("sample model missing; skipped");
        return None;
    }
    let root = std::env::temp_dir().join(format!("zest_renderer_split_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::copy(&sample, root.join("m.glb")).unwrap();
    let asset = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
    Some((root, asset.meshes.len()))
}

fn model_entry(extra: &[usize]) -> SceneEntry {
    legacy_entry("Body", ObjectSource::Model { path: "m.glb".into(), mesh: 0, extra_meshes: extra.to_vec(), model: None, mesh_id: None, extra_mesh_ids: Vec::new() })
}

fn append(root: &Path, entries: &[SceneEntry]) -> LoadedScene {
    let mut scene = LoadedScene::empty();
    scene.append_entries_with(None, &mut ShaderLibrary::default(), root, &ComponentRegistry::with_builtins(), entries).unwrap();
    scene
}

#[test]
fn slot_lists_shorter_and_longer_than_the_mesh_keep_their_data() {
    let Some((root, meshes)) = mannequin("slots") else { return };
    assert!(meshes >= 3);
    // Three slots; the legacy list names only slot 1 (shorter than the mesh).
    let mut e = model_entry(&[1, 2]);
    e.extra_materials = vec![MaterialRef::File("materials/arm.material".into())];
    let scene = append(&root, &[e]);
    let o = &scene.objects[0];
    assert_eq!(o.render_slots().count(), 3);
    let r = renderer_data(o);
    assert_eq!(
        r.materials,
        vec![MaterialSlot::Model, MaterialSlot::File("materials/arm.material".into()), MaterialSlot::Model],
        "slot 0 was never set (Model) and the tail inherits"
    );

    // A renderer whose list is longer than the mesh keeps the extras and saves them.
    let mut long = model_entry(&[1]);
    long.components.push(zestengine_assets::ComponentData::new("Mesh Renderer", "(materials:[Model,File(\"b.material\"),File(\"extra.material\")])"));
    long.renderer_split = true;
    let scene = append(&root, &[long]);
    let o = &scene.objects[0];
    assert_eq!(o.render_slots().count(), 2, "the extra slot has no submesh to draw");
    let saved = scene.to_scene_file(&root, &ShaderLibrary::default());
    let data = &saved.entries[0].components[0].data;
    assert!(data.contains("extra.material"), "kept: {data}");

    // A short list pads with Model in the saved data too.
    let mut short = model_entry(&[1, 2]);
    short.components.push(zestengine_assets::ComponentData::new("Mesh Renderer", "(materials:[File(\"a.material\")])"));
    short.renderer_split = true;
    let scene = append(&root, &[short]);
    assert_eq!(renderer_data(&scene.objects[0]).slot(2), MaterialSlot::Model);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn restoring_a_slot_updates_the_component_and_the_cache_together() {
    let Some((root, _)) = mannequin("restore") else { return };
    let scene_file = SceneFile { entries: vec![model_entry(&[1])], ..Default::default() };
    let mut scene = LoadedScene::empty();
    scene.append_entries_with(None, &mut ShaderLibrary::default(), &root, &ComponentRegistry::with_builtins(), &scene_file.entries).unwrap();
    let before = scene.slot_material(0, 1).unwrap();
    scene.restore_renderer_slot(0, 1, before + 7, MaterialSlot::File("materials/x.material".into()));
    assert_eq!(scene.slot_material(0, 1), Some(before + 7));
    assert_eq!(scene.renderer_slot_value(0, 1), MaterialSlot::File("materials/x.material".into()));
    assert_eq!(scene.renderer_slot_value(0, 0), MaterialSlot::Model, "other slots untouched");
    scene.restore_renderer_slot(0, 1, before, MaterialSlot::Model);
    assert_eq!(scene.renderer_slot_value(0, 1), MaterialSlot::Model);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn material_user_counts_count_objects_not_slots() {
    let Some((root, _)) = mannequin("users") else { return };
    let a = model_entry(&[1]);
    let mut b = model_entry(&[1]);
    b.name = "Body 2".into();
    let scene = append(&root, &[a, b]);
    let counts = scene.material_user_counts();
    let m0 = scene.slot_material(0, 0).unwrap();
    assert_eq!(counts.get(&m0), Some(&2), "two objects share the model's material");
    let _ = (Vec3::ZERO, std::fs::remove_dir_all(&root));
}

#[test]
fn duplicating_an_object_copies_its_renderer_exactly_once() {
    let mut e = cube("A");
    e.cast_shadows = ShadowCasting::Off;
    let mut scene = load(&SceneFile { entries: vec![e], ..Default::default() });
    let copy = scene.duplicate_object(0, &ComponentRegistry::with_builtins()).unwrap();
    assert_eq!(scene.objects[copy].components.iter().filter(|c| renderer_of(c.as_ref()).is_some()).count(), 1);
    assert_eq!(scene.objects[copy].cast_shadows(), ShadowCasting::Off);
}
