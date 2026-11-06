//! NET-3: the dedicated server loads project component plugins (`plugins: [..]` in `server.ron`),
//! validates their ABI stamp and refuses to start, with a clear error, when one is unusable.
//! Uses the workspace's own plugin crate `plugins/components` (the scaffolded `Orbit` component).

use std::path::PathBuf;
use zestengine_core::ComponentRegistry;
use zestengine_server::{Server, ServerConfig};

/// Build the workspace plugin as a cdylib (same profile as this test, so the ABI stamp matches)
/// and return the library path.
fn built_plugin() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap().to_path_buf(); // target/<profile>/deps/x -> target/<profile>
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut cmd = std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(&workspace).args(["build", "-p", "zestengine-project-components", "--lib", "--locked"]);
    if !cfg!(debug_assertions) {
        cmd.arg("--release");
    }
    if let Some(t) = std::env::var_os("CARGO_TARGET_DIR") {
        cmd.env("CARGO_TARGET_DIR", t);
    }
    let out = cmd.output().expect("run cargo");
    assert!(out.status.success(), "plugin build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let lib = profile_dir.join(if cfg!(windows) { "zestengine_project_components.dll" } else if cfg!(target_os = "macos") { "libzestengine_project_components.dylib" } else { "libzestengine_project_components.so" });
    assert!(lib.is_file(), "{} not built", lib.display());
    lib
}

fn project_with_orbit(tag: &str, lib: &std::path::Path) -> (PathBuf, PathBuf) {
    let parent = std::env::temp_dir().join(format!("zestengine-server-{tag}-{}-{}", std::process::id(), zestengine_core::ObjectId::new()));
    std::fs::create_dir_all(&parent).unwrap();
    let root = zestengine_engine::scaffold_project(&parent, "vr_social", "vr_social").unwrap();
    let boot = zestengine_assets::load_project_file(&root).unwrap().boot_scene.unwrap();
    let mut file = zestengine_assets::load_scene_file(root.join(&boot)).unwrap();
    let plugins = zestengine_engine::RuntimePlugins::load_all(&[lib.to_path_buf()]).expect("the editor-side loader accepts the plugin");
    let mut reg = ComponentRegistry::with_builtins();
    plugins.register_all(&mut reg);
    let mut entry = file.entries[0].clone();
    entry.id = zestengine_core::ObjectId::new().to_string();
    entry.name = "Orbiter".into();
    entry.parent = None;
    entry.components = vec![zestengine_assets::ComponentData::of(reg.create("Orbit").expect("Orbit is registered by the plugin").as_ref())];
    file.entries.push(entry);
    zestengine_assets::save_scene_file(root.join(&boot), &file).unwrap();
    (parent, root)
}

fn cfg(root: &std::path::Path, plugins: Vec<String>) -> ServerConfig {
    ServerConfig { project: root.display().to_string(), port: Some(0), lan_discovery: false, console: false, stats_interval: 0.0, plugins, ..Default::default() }
}

#[test]
fn a_listed_plugin_supplies_the_scenes_components() {
    let lib = built_plugin();
    let (parent, root) = project_with_orbit("plugin-ok", &lib);
    let ron = format!("(project: {:?}, port: Some(0), plugins: [{:?}])", root.display().to_string(), lib.display().to_string());
    let from_ron = ServerConfig::from_ron_str(&ron).unwrap();
    assert_eq!(from_ron.plugins.len(), 1);

    let mut server = Server::open(ServerConfig { lan_discovery: false, console: false, stats_interval: 0.0, ..from_ron }).expect("starts with the plugin");
    let start = server.host().scene().objects.iter().find(|o| o.name == "Orbiter").unwrap().translation;
    for _ in 0..10 {
        server.tick(1.0 / 30.0);
    }
    let o = server.host().scene().objects.iter().find(|o| o.name == "Orbiter").unwrap();
    assert!(o.components.iter().any(|c| c.type_name() == "Orbit"), "the plugin component was instantiated on the server");
    assert!((o.translation - start).length() > 1e-3, "the plugin's update ran on the server");
    drop(server);

    // Without the plugin the same scene still opens (the component is simply unknown): the
    // plugin list is what supplies it, which is the point.
    let bare = Server::open(cfg(&root, vec![])).expect("opens without");
    assert!(!bare.host().scene().objects.iter().find(|o| o.name == "Orbiter").unwrap().components.iter().any(|c| c.type_name() == "Orbit"));
    let _ = std::fs::remove_dir_all(&parent);
}

#[test]
fn an_unusable_plugin_refuses_to_start_with_a_clear_error() {
    let lib = built_plugin();
    let (parent, root) = project_with_orbit("plugin-bad", &lib);
    let err = |plugins: Vec<String>| Server::open(cfg(&root, plugins)).err().map(|e| format!("{e:#}")).expect("must refuse to start");

    let e = err(vec![root.join("nope.so").display().to_string()]);
    assert!(e.contains("plugins") && e.contains("does not exist"), "{e}");

    let junk = root.join("junk.so");
    std::fs::write(&junk, b"not an elf").unwrap();
    let e = err(vec![junk.display().to_string()]);
    assert!(e.contains("junk.so"), "{e}");

    // A real shared library with no ABI stamp (any system library will do).
    #[cfg(target_os = "linux")]
    if let Some(sys) = ["/usr/lib/libm.so.6", "/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib64/libm.so.6"].iter().find(|p| std::path::Path::new(p).is_file()) {
        let e = err(vec![(*sys).into()]);
        assert!(e.contains("no ABI stamp") && e.contains("export_plugin_abi"), "{e}");
    }

    // A good plugin followed by a bad one still refuses (nothing half-loaded is served).
    let e = err(vec![lib.display().to_string(), junk.display().to_string()]);
    assert!(e.contains("junk.so"), "{e}");

    // Config validation: empty and duplicate entries.
    let e = cfg(&root, vec![String::new()]).validate().unwrap_err().to_string();
    assert!(e.contains("plugins[0] is empty"), "{e}");
    let e = cfg(&root, vec!["a.so".into(), "a.so".into()]).validate().unwrap_err().to_string();
    assert!(e.contains("listed twice"), "{e}");
    let _ = std::fs::remove_dir_all(&parent);
}
