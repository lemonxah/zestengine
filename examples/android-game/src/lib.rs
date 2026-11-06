//! Android entry point for a zestengine game. The APK carries the game's assets
//! (built by tools/android/build_apk.sh); they're extracted to internal storage
//! on first run and the boot scene comes from their project.zestengine. Packaged
//! with `--quest`, the APK carries the OpenXR loader and VR starts on its own
//! (`XrMode::Auto`); a phone build has no loader and runs flat.

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(app: zestengine_engine::AndroidApp) {
    zestengine_engine::init_logging();
    let root = match zestengine_engine::android_assets_root(&app) {
        Ok(root) => root,
        Err(e) => {
            tracing::error!("extracting game assets: {e:#}");
            return;
        }
    };
    let config = match zestengine_engine::GameConfig::from_project_dir(root) {
        Ok(config) => config,
        Err(e) => {
            tracing::error!("reading project.zestengine: {e:#}");
            return;
        }
    };
    if let Err(e) = zestengine_engine::run_game_android(app, config, |_| {}) {
        tracing::error!("game exited: {e:#}");
    }
}
