//! The built-in environment sun became a `Directional Light` object. These
//! tests pin the migration (old scenes light the same), the key-light rule and
//! the "no env sun" editor panel. CPU only.

use glam::Vec3;
use zestengine_assets::{ObjectSource, SceneFile};
use zestengine_core::ComponentRegistry;
use zestengine_render::{LightInstance, LightKind};

use super::{LoadedScene, key_light_data, main_directional};
use crate::shaders::ShaderLibrary;

const OLD_SUN: &str = r#"(entries:[(name:"Marker",source:Empty,material:Model,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0))],environment:(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,sun_enabled:true,sun_color:(1.0,0.98,0.92),sun_intensity:3.0,sun_direction:(-0.4,-1.0,-0.3),skybox_enabled:true,realtime_gi:false))"#;

fn load(file: &SceneFile) -> LoadedScene {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    LoadedScene::load_scene_file(None, file, &root, &ComponentRegistry::with_builtins(), &mut ShaderLibrary::default()).expect("scene loads without a renderer")
}

/// What the renderer got from the old environment sun, built the way the
/// removed `runtime.rs` / `editor_app.rs` code did.
fn old_sun_instance(color: [f32; 3], intensity: f32, direction: [f32; 3]) -> LightInstance {
    LightInstance {
        kind: LightKind::Directional,
        position: Vec3::ZERO,
        direction: Vec3::from(direction).normalize_or(Vec3::NEG_Y),
        color,
        intensity,
        range: 0.0,
        spot_inner_deg: 0.0,
        spot_outer_deg: 0.0,
        cast_shadows: true,
        soft_shadows: true, radius: 0.0,
        shadow_bias: 0.003,
        baked: false,
    }
}

fn dir(kind: LightKind, intensity: f32, color: [f32; 3]) -> LightInstance {
    LightInstance { kind, color, intensity, ..old_sun_instance([1.0; 3], 1.0, [0.0, -1.0, 0.0]) }
}

#[test]
fn migrated_scene_light_list_equals_the_old_sun_light() {
    let file: SceneFile = ron::from_str(OLD_SUN).unwrap();
    let scene = load(&file);
    assert!(scene.legacy_sun_migrated);
    assert!(!scene.environment.sun_enabled);
    let lights = scene.gather_lights();
    assert_eq!(lights.len(), 1, "exactly the old sun, as a light object");
    let (new, old) = (&lights[0], old_sun_instance([1.0, 0.98, 0.92], 3.0, [-0.4, -1.0, -0.3]));
    assert_eq!(new.kind, old.kind);
    assert_eq!(new.color, old.color);
    assert_eq!(new.intensity, old.intensity);
    assert!(new.direction.distance(old.direction) < 1e-5, "{:?} vs {:?}", new.direction, old.direction);
    assert_eq!((new.cast_shadows, new.soft_shadows, new.baked), (old.cast_shadows, old.soft_shadows, old.baked));
    assert_eq!(new.shadow_bias, old.shadow_bias);
    // The key light handed to the renderer matches the old world light too.
    let key = key_light_data(&lights, [0.1; 3], 1.0);
    assert_eq!((key.color, key.intensity), ([1.0, 0.98, 0.92], 3.0));
    assert!(key.direction.distance(old.direction) < 1e-5);
}

#[test]
fn scene_without_a_sun_stays_dark_and_the_key_light_is_off() {
    let file: SceneFile = ron::from_str(&OLD_SUN.replace("sun_enabled:true", "sun_enabled:false")).unwrap();
    let scene = load(&file);
    assert!(!scene.legacy_sun_migrated);
    assert!(scene.gather_lights().is_empty());
    let key = key_light_data(&[], [0.1; 3], 1.0);
    assert_eq!(key.intensity, 0.0);
}

#[test]
fn existing_directional_light_wins_over_the_old_sun() {
    let mut file: SceneFile = ron::from_str(OLD_SUN).unwrap();
    let mut own = zestengine_assets::default_directional_light_entry();
    own.components = vec![zestengine_assets::ComponentData::new("Light", "(kind:Directional,color:(0.2,0.3,0.4),intensity:7.0,range:10.0,spot_angle:45.0,spot_blend:0.15)")];
    file.entries.push(own);
    let scene = load(&file);
    let lights = scene.gather_lights();
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].intensity, 7.0);
    assert!(!scene.environment.sun_enabled, "the env sun is ignored");
}

#[test]
fn load_save_load_does_not_duplicate_the_light() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let shaders = ShaderLibrary::default();
    let mut file: SceneFile = ron::from_str(OLD_SUN).unwrap();
    for round in 0..3 {
        let scene = load(&file);
        assert_eq!(scene.legacy_sun_migrated, round == 0, "only the first load migrates");
        assert_eq!(scene.gather_lights().len(), 1, "round {round}");
        let saved = scene.to_scene_file(&root, &shaders);
        let text = ron::ser::to_string_pretty(&saved, ron::ser::PrettyConfig::default()).unwrap();
        assert!(!text.contains("sun_enabled"), "the migrated scene does not write the legacy sun");
        file = ron::from_str(&text).unwrap();
    }
}

#[test]
fn prefab_and_additive_entry_loads_never_add_a_light() {
    // Instantiating entries (prefabs, additive loads) goes through push_entries,
    // which never reads the environment: a legacy-sun file used as a prefab
    // source adds its entries only.
    let file: SceneFile = ron::from_str(OLD_SUN).unwrap();
    let mut scene = LoadedScene::empty();
    let registry = ComponentRegistry::with_builtins();
    scene.push_entries(None, &mut ShaderLibrary::default(), std::path::Path::new("."), &registry, &file.entries).unwrap();
    assert_eq!(scene.objects.len(), 1);
    assert!(scene.gather_lights().is_empty());
    assert!(!scene.legacy_sun_migrated);
    assert!(matches!(scene.objects[0].source, ObjectSource::Empty));
}

#[test]
fn new_editor_scene_starts_with_the_default_light() {
    let mut scene = LoadedScene::empty();
    scene.add_default_directional_light();
    let lights = scene.gather_lights();
    assert_eq!(lights.len(), 1);
    assert_eq!((lights[0].color, lights[0].intensity), ([1.0, 0.98, 0.92], 3.0));
    assert_eq!(scene.objects[0].name, "Directional Light");
}

#[test]
fn main_directional_is_the_brightest_first_of_equals() {
    let lights = [
        dir(LightKind::Point, 99.0, [1.0; 3]),
        dir(LightKind::Directional, 2.0, [1.0; 3]),
        dir(LightKind::Directional, 5.0, [1.0, 0.0, 0.0]), // luminance 0.21 * 5 ~ 1.06
        dir(LightKind::Directional, 2.0, [1.0; 3]),
    ];
    assert_eq!(main_directional(&lights), Some(1));
    assert_eq!(main_directional(&lights[..1]), None);
    let mut baked = dir(LightKind::Directional, 3.0, [1.0; 3]);
    baked.baked = true;
    assert_eq!(key_light_data(&[baked], [0.0; 3], 1.0).intensity, 0.0, "baked key light: in the lightmaps");
}

#[test]
fn environment_panel_has_no_sun_controls() {
    let src = include_str!("../../../zestengine-editor/src/editor_app.rs");
    let start = src.find("fn environment_ui").expect("environment_ui");
    let body = &src[start..];
    let end = body[10..].find("\n    fn ").map_or(body.len(), |e| e + 10);
    let body = &body[..end];
    for banned in ["sun_", "Sun", "World Light"] {
        assert!(!body.contains(banned), "environment_ui still mentions {banned:?}");
    }
}
