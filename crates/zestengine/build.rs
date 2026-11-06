//! Steam Audio ships as a shared library (libphonon), downloaded by the
//! audionimbus-sys build script into its OUT_DIR. Copy it next to the editor
//! binary and point the binary's rpath at its own directory, so the editor
//! starts when launched directly (desktop entry), not only via `cargo run`.

use std::path::{Path, PathBuf};

fn lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "phonon.dll"
    } else if cfg!(target_os = "macos") {
        "libphonon.dylib"
    } else {
        "libphonon.so"
    }
}

/// The most recently built copy under `<profile>/build/audionimbus-sys-*/out/lib`.
fn newest_build(profile_dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(profile_dir.join("build"))
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("audionimbus-sys-"))
        .map(|e| e.path().join("out/lib").join(lib_name()))
        .filter(|p| p.is_file())
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok())
}

fn main() {
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target.as_str() {
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
        "macos" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path"),
        _ => {}
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // OUT_DIR = <target>/<profile>/build/zestengine-<hash>/out
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    let dest = profile_dir.join(lib_name());
    let mut found = newest_build(profile_dir);
    // Cargo does not order this script after audionimbus-sys's (it is not a build
    // dependency), so on a clean target the SDK can still be downloading. Wait for it
    // instead of leaving the editor without its library. Only with Steam Audio on and
    // more than one job: a single job would starve the download and deadlock.
    let jobs = std::env::var("NUM_JOBS").ok().and_then(|j| j.parse::<u32>().ok()).unwrap_or(1);
    if found.is_none() && std::env::var_os("CARGO_FEATURE_STEAM_AUDIO").is_some() && jobs > 1 {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
        while found.is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_secs(2));
            found = newest_build(profile_dir);
        }
    }
    match found {
        Some(src) => {
            println!("cargo:rerun-if-changed={}", src.display());
            let stale = std::fs::metadata(&dest).map(|m| m.len()).ok() != std::fs::metadata(&src).map(|m| m.len()).ok();
            if stale && let Err(e) = std::fs::copy(&src, &dest) {
                println!("cargo:warning=copying {} next to the editor: {e}", lib_name());
            }
        }
        None => {
            // First build: audionimbus-sys may not have run yet. A missing
            // rerun-if-changed path makes the next build run this again.
            println!("cargo:rerun-if-changed={}", dest.display());
            println!(
                "cargo:warning=Steam Audio library not built yet; build again to place {} next to the editor",
                lib_name()
            );
        }
    }
}
