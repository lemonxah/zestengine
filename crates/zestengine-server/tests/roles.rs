//! NET-3 on the server: content roles are applied on load, a registry given to
//! `Server::open_with` supplies the project's components, and the `server.ron`
//! the Dedicated server build writes is accepted by the server's own parser.

use serde::{Deserialize, Serialize};
use zestengine_core::{ComponentCtx, ComponentRegistry, NetRole, TypedComponent};
use zestengine_server::{Server, ServerConfig};

/// A project component: counts ticks on the server only.
#[derive(Serialize, Deserialize, Default)]
struct Marker {
    #[serde(default)]
    ticks: u32,
}
impl TypedComponent for Marker {
    const NAME: &'static str = "Roles Test Marker";
    fn update(&mut self, _ctx: &mut ComponentCtx) {
        self.ticks += 1;
    }
}

/// A project component that exists for clients only by its own declaration.
#[derive(Serialize, Deserialize, Default)]
struct Sparkle;
impl TypedComponent for Sparkle {
    const NAME: &'static str = "Roles Test Sparkle";
    const NET_ROLE: NetRole = NetRole::Client;
}

fn register(r: &mut ComponentRegistry) {
    r.register::<Marker>();
    r.register::<Sparkle>();
}

#[test]
fn server_hosts_a_scene_with_project_components_and_strips_client_content() {
    let parent = std::env::temp_dir().join(format!("zestengine-server-roles-{}-{}", std::process::id(), zestengine_core::ObjectId::new()));
    std::fs::create_dir_all(&parent).unwrap();
    let root = zestengine_engine::scaffold_project(&parent, "vr_social", "vr_social").unwrap();
    let project = zestengine_assets::load_project_file(&root).unwrap();
    let boot = project.boot_scene.clone().unwrap();
    let mut file = zestengine_assets::load_scene_file(root.join(&boot)).unwrap();
    let reg = {
        let mut r = ComponentRegistry::with_builtins();
        register(&mut r);
        r
    };
    let mut entry = file.entries[0].clone();
    entry.id = zestengine_core::ObjectId::new().to_string();
    entry.name = "Marked".into();
    entry.parent = None;
    entry.components = vec![
        zestengine_assets::ComponentData::of(reg.create("Roles Test Marker").unwrap().as_ref()),
        zestengine_assets::ComponentData::of(reg.create("Roles Test Sparkle").unwrap().as_ref()),
    ];
    let mut fx = entry.clone();
    fx.id = zestengine_core::ObjectId::new().to_string();
    fx.name = "Client Fx".into();
    fx.components.clear();
    fx.net_role = NetRole::Client;
    file.entries.push(entry);
    file.entries.push(fx);
    zestengine_assets::save_scene_file(root.join(&boot), &file).unwrap();

    let cfg = ServerConfig {
        project: root.display().to_string(),
        port: Some(0),
        lan_discovery: false,
        console: false,
        stats_interval: 0.0,
        ..Default::default()
    };
    let mut server = Server::open_with(cfg, register).expect("server opens with the project registry");
    assert_eq!(server.host().scene().net_side, NetRole::Server);
    for _ in 0..20 {
        server.tick(1.0 / 30.0);
    }
    let scene = server.host().scene();
    assert!(!scene.objects.iter().any(|o| o.name == "Client Fx"), "client-only object is not on the server");
    let marked = scene.objects.iter().find(|o| o.name == "Marked").expect("project object loaded");
    let kinds: Vec<&str> = marked.components.iter().map(|c| c.type_name()).collect();
    assert_eq!(kinds, ["Roles Test Marker"], "the type-declared client-only component is stripped: {kinds:?}");
    let ticks = marked.components.iter().find_map(|c| c.as_any().downcast_ref::<Marker>()).unwrap().ticks;
    assert!(ticks >= 20, "the project component ran {ticks} updates");
    assert!(scene.objects.iter().all(|o| o.components.iter().all(|c| !matches!(c.type_name(), "Light" | "UI Text" | "Audio Source"))));
    let _ = std::fs::remove_dir_all(&parent);
}

#[test]
fn the_generated_server_ron_is_a_valid_config() {
    let text = zestengine_engine::server_build::server_ron_for("Arena", Some("scenes/main.scene"));
    let cfg = ServerConfig::from_ron_str(&text).unwrap_or_else(|e| panic!("{e:#}\n{text}"));
    assert_eq!(cfg.project, "assets");
    assert_eq!(cfg.scene.as_deref(), Some("scenes/main.scene"));
    assert_eq!(cfg.server_name, "Arena");
    assert!((cfg.tick_rate - 30.0).abs() < f32::EPSILON);
    let no_scene = ServerConfig::from_ron_str(&zestengine_engine::server_build::server_ron_for("A", None)).unwrap();
    assert!(no_scene.scene.is_none());
}

fn scaffold_with_marker(tag: &str) -> (std::path::PathBuf, std::path::PathBuf, String) {
    let parent = std::env::temp_dir().join(format!("zestengine-server-{tag}-{}-{}", std::process::id(), zestengine_core::ObjectId::new()));
    std::fs::create_dir_all(&parent).unwrap();
    let root = zestengine_engine::scaffold_project(&parent, "vr_social", "vr_social").unwrap();
    let project = zestengine_assets::load_project_file(&root).unwrap();
    let boot = project.boot_scene.clone().unwrap();
    let mut file = zestengine_assets::load_scene_file(root.join(&boot)).unwrap();
    let mut reg = ComponentRegistry::with_builtins();
    register(&mut reg);
    let mut entry = file.entries[0].clone();
    entry.id = zestengine_core::ObjectId::new().to_string();
    entry.name = "Marked".into();
    entry.parent = None;
    entry.components = vec![zestengine_assets::ComponentData::of(reg.create("Roles Test Marker").unwrap().as_ref())];
    file.entries.push(entry);
    zestengine_assets::save_scene_file(root.join(&boot), &file).unwrap();
    (parent, root, boot)
}

fn open_cfg(project: String) -> ServerConfig {
    ServerConfig { project, port: Some(0), lan_discovery: false, console: false, stats_interval: 0.0, ..Default::default() }
}

/// NET-3: a project-wide override moves a component type to the other side without touching scenes.
#[test]
fn project_role_override_strips_a_type_on_the_server() {
    let (parent, root, _boot) = scaffold_with_marker("override");
    // Without the override the Both-role marker is kept.
    let server = Server::open_with(open_cfg(root.display().to_string()), register).unwrap();
    assert!(server.host().scene().objects.iter().any(|o| o.name == "Marked" && o.components.iter().any(|c| c.type_name() == "Roles Test Marker")));
    drop(server);
    let mut project = zestengine_assets::load_project_file(&root).unwrap();
    project.net_role_overrides.insert("Roles Test Marker".into(), NetRole::Client);
    zestengine_assets::save_project_file(&root, &project).unwrap();
    let reloaded = zestengine_assets::load_project_file(&root).unwrap();
    assert_eq!(reloaded.net_role_overrides.get("Roles Test Marker"), Some(&NetRole::Client), "round-trips through project.zestengine");
    let server = Server::open_with(open_cfg(root.display().to_string()), register).unwrap();
    let marked = server.host().scene().objects.iter().find(|o| o.name == "Marked").expect("object stays");
    assert!(marked.components.iter().all(|c| c.type_name() != "Roles Test Marker"), "the override stripped the type");
    let _ = std::fs::remove_dir_all(&parent);
}

/// NET-3: the server runs from an `assets.pak` (unpacked once per pak id into its state folder).
#[test]
fn server_runs_from_an_assets_pak() {
    let (parent, root, boot) = scaffold_with_marker("pak");
    let out = parent.join("build-server");
    let mut reg = ComponentRegistry::with_builtins();
    register(&mut reg);
    let cook = zestengine_engine::server_build::cook_server_assets(&root, &out.join("assets"), &reg, &mut |_| {}).unwrap();
    assert!(cook.files > 0);
    let pak = out.join("assets.pak");
    let n = zestengine_assets::game_pak::pack_dir(&out.join("assets"), &pak).unwrap();
    assert_eq!(n, cook.files);
    let state = parent.join("state");
    for project in [pak.display().to_string(), out.display().to_string()] {
        let mut cfg = open_cfg(project.clone());
        cfg.scene = Some(boot.clone());
        cfg.state_dir = state.display().to_string();
        let mut server = Server::open_with(cfg, register).unwrap_or_else(|e| panic!("{project}: {e:#}"));
        for _ in 0..3 {
            server.tick(1.0 / 30.0);
        }
        assert!(server.host().scene().objects.iter().any(|o| o.name == "Marked"), "{project}");
    }
    assert!(state.join("assets").is_dir(), "extracted under the state folder");
    // A missing scene in the pak is a clear error.
    let mut cfg = open_cfg(pak.display().to_string());
    cfg.scene = Some("scenes/nope.scene".into());
    cfg.state_dir = state.display().to_string();
    let err = Server::open_with(cfg, register).err().expect("fails").to_string();
    assert!(err.contains("scenes/nope.scene"), "{err}");
    // Default state folder sits beside the pak, not inside the extraction.
    let c = open_cfg(pak.display().to_string());
    assert_eq!(c.state_path(), out.join(".server"));
    let _ = std::fs::remove_dir_all(&parent);
}
