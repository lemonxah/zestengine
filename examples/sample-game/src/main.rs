//! Generated game entry point (prototype of what `New Project` will scaffold).
//!
//! Boots the zestengine runtime, registers this project's statically-linked
//! components, and loads the first scene. The boot scene is passed here from
//! the project config; edit this file directly for full control over startup
//! (splash screen, pick a scene from a save file, etc.).

use std::path::PathBuf;

use anyhow::Result;

fn main() -> Result<()> {
    zestengine_engine::init_logging();

    // In a real bundle, assets sit next to the executable. For this in-repo
    // test, point at the zestengine repo root so the existing scenes resolve.
    let assets_root = std::env::var("ZESTENGINE_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")));

    let config = zestengine_engine::GameConfig {
        assets_root,
        boot_scene: "scenes/world.scene".into(),
        title: "Sample Game".into(),
        width: 1280.0,
        height: 720.0,
        ..Default::default()
    };

    zestengine_engine::run_game(config, |registry| {
        zestengine_project_components::zestengine_register(registry);
    })
}
