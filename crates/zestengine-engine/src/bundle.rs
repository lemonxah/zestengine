//! Project scaffolding (`New Project`) and game bundling (`Build Game`).
//!
//! A zestengine project is a standalone cargo workspace: a game-binary package at
//! the root (`src/main.rs`) plus a `plugins/components` crate the editor
//! hot-loads and the game links statically. `scaffold_project` writes that
//! layout; `build_game` compiles it `--release` and copies the executable plus
//! the referenced assets into `build/`, ready to ship. Android and Meta Quest
//! builds generate a small `android_main` cdylib crate under
//! `target/zestengine-android/` and package it with `tools/android/build_apk.sh`.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use zestengine_assets::ProjectFile;

/// Asset subfolders every project gets.
const ASSET_DIRS: &[&str] = &["scenes", "materials", "shaders", "textures", "models", "audio", "prefabs"];

/// Top-level project folders that are code or build output, never game content.
const NON_ASSET_DIRS: &[&str] = &["build", "build-windows", "build-android", "build-quest", "build-server", "target", "src", "plugins"];

/// Which platform `build_game` produces an executable for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildTarget {
    /// The platform the editor runs on (output in `build/`).
    Host,
    /// 64-bit Windows (MSVC ABI). From Linux/macOS this cross-compiles with
    /// `cargo xwin`; output in `build-windows/`.
    Windows,
    /// Android phone/tablet (arm64-v8a APK, flat). Output in `build-android/`.
    Android,
    /// Meta Quest 3 / 3S / Pro standalone (arm64-v8a APK with the OpenXR
    /// loader and the VR manifest). Output in `build-quest/`.
    Quest,
    /// Dedicated server (NET-3): the server binary for the host triple
    /// (`ZESTENGINE_SERVER_TARGET` picks another), stripped collision-only
    /// assets and a ready `server.ron`. Output in `build-server/`.
    Server,
}

impl BuildTarget {
    pub const ALL: [BuildTarget; 5] =
        [BuildTarget::Host, BuildTarget::Windows, BuildTarget::Android, BuildTarget::Quest, BuildTarget::Server];

    pub fn label(self) -> &'static str {
        match self {
            BuildTarget::Host if cfg!(windows) => "Windows (this machine)",
            BuildTarget::Host if cfg!(target_os = "macos") => "macOS (this machine)",
            BuildTarget::Host => "Linux (this machine)",
            BuildTarget::Windows => "Windows x64",
            BuildTarget::Android => "Android (arm64)",
            BuildTarget::Quest => "Meta Quest 3 / 3S / Pro",
            BuildTarget::Server => "Dedicated server",
        }
    }

    /// Parse a CLI name (`host`, `windows`, `android`, `quest`, `server`).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "host" | "native" => Some(BuildTarget::Host),
            "windows" | "win" | "win64" | "x86_64-pc-windows-msvc" => Some(BuildTarget::Windows),
            "android" | "aarch64-linux-android" => Some(BuildTarget::Android),
            "quest" | "meta-quest" | "quest3" | "questpro" => Some(BuildTarget::Quest),
            "server" | "dedicated" | "dedicated-server" | "linux-server" => Some(BuildTarget::Server),
            _ => None,
        }
    }

    /// The targets offered on this machine: no Windows cross-build from
    /// Windows itself, and the APK packager is a bash script (no Windows host).
    pub fn available() -> impl Iterator<Item = BuildTarget> {
        Self::ALL.into_iter().filter(|t| match t {
            BuildTarget::Host => true,
            BuildTarget::Windows | BuildTarget::Android | BuildTarget::Quest => !cfg!(windows),
            BuildTarget::Server => true,
        })
    }

    /// True when this target needs a cross toolchain from the current host.
    fn is_cross(self) -> bool {
        self == BuildTarget::Windows && !cfg!(windows)
    }

    fn is_windows(self) -> bool {
        match self {
            BuildTarget::Host => cfg!(windows),
            BuildTarget::Windows => true,
            BuildTarget::Android | BuildTarget::Quest => false,
            BuildTarget::Server => cfg!(windows),
        }
    }

    /// Builds an APK (Android or Meta Quest).
    pub fn is_android(self) -> bool {
        matches!(self, BuildTarget::Android | BuildTarget::Quest)
    }

    fn output_dir(self) -> &'static str {
        match self {
            BuildTarget::Android => "build-android",
            BuildTarget::Quest => "build-quest",
            BuildTarget::Server => "build-server",
            _ if self.is_cross() => "build-windows",
            _ => "build",
        }
    }

    /// Where the build lands, relative to the project root.
    pub fn output_hint(self) -> &'static str {
        self.output_dir()
    }
}

const WINDOWS_TRIPLE: &str = "x86_64-pc-windows-msvc";

/// Whether `dir` is a zestengine checkout (the workspace that holds `crates/zestengine-engine`).
fn is_checkout(dir: &Path) -> bool {
    dir.join("crates/zestengine-engine/Cargo.toml").is_file()
}

/// The checkout this binary was built from (`crates/zestengine-engine/../..` at compile time), if it
/// still exists. Lets an installed editor (`/usr/lib/zestengine`, from `just install`), which is not
/// inside the checkout, still find the engine crates that generated projects path-depend on.
fn build_checkout() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?;
    is_checkout(dir).then(|| dir.to_path_buf())
}

/// Locate the zestengine checkout so generated projects can path-depend on its
/// crates: `$ZESTENGINE_ROOT` if it points at one, else the first checkout above
/// the running executable (a build tree), else the checkout the binary was built from.
pub fn zestengine_root() -> Result<PathBuf> {
    if let Ok(root) = std::env::var("ZESTENGINE_ROOT") {
        let root = PathBuf::from(root);
        if is_checkout(&root) {
            return Ok(root);
        }
    }
    let exe = std::env::current_exe().context("locating editor executable")?;
    let mut dir = exe.parent();
    while let Some(d) = dir {
        if is_checkout(d) {
            return Ok(d.to_path_buf());
        }
        dir = d.parent();
    }
    if let Some(root) = build_checkout() {
        return Ok(root);
    }
    bail!(
        "could not find the zestengine source checkout (looked above {} and at {}, where it was built); \
         set $ZESTENGINE_ROOT to the checkout",
        exe.display(),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../..")
    )
}

/// A cargo-legal crate name derived from a project name.
pub(crate) fn crate_name(project_name: &str) -> String {
    let mut s: String = project_name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    s = s.trim_matches('-').to_string();
    if s.is_empty() || s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        s = format!("game-{s}");
    }
    s
}

/// Create a fresh project under `parent/<name>` from a project template
/// (`crate::templates::TEMPLATES` id, e.g. "empty", "fps") and return its root
/// path. Errors if a project already exists there.
pub fn scaffold_project(parent: &Path, name: &str, template: &str) -> Result<PathBuf> {
    let root = parent.join(name);
    if root.join("project.zestengine").exists() {
        bail!("a zestengine project already exists at {}", root.display());
    }
    if crate::templates::find(template).is_none() {
        bail!(
            "unknown project template {template:?} (expected one of: {})",
            crate::templates::TEMPLATES.iter().map(|t| t.id).collect::<Vec<_>>().join(", ")
        );
    }
    let zestengine = zestengine_root()?;
    let pkg = crate_name(name);

    for dir in ASSET_DIRS {
        std::fs::create_dir_all(root.join(dir))
            .with_context(|| format!("creating {dir}/"))?;
    }
    std::fs::create_dir_all(root.join("src"))?;

    // Root Cargo.toml: the game binary + a workspace whose dependency table the
    // plugins/components crate inherits (`zestengine-*.workspace = true`).
    let zestengine_disp = zestengine.display();
    std::fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "{pkg}"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "{pkg}"
path = "src/main.rs"

# Standalone workspace: the zestengine crates are path-depended from your local
# checkout. Move/clone the engine elsewhere and update these paths.
[workspace]
members = ["plugins/components"]

[workspace.dependencies]
zestengine-core = {{ path = "{zestengine_disp}/crates/zestengine-core" }}
zestengine-editor = {{ path = "{zestengine_disp}/crates/zestengine-editor" }}
# Runtime only: default-features = false drops the editor (no zestengine-editor,
# egui_dock, gizmos, LSP) from the shipped game; `client` keeps the renderer, audio
# and input (a dedicated server build leaves it off). It must be set HERE — a member
# inheriting with `workspace = true` can't override default-features.
zestengine-engine = {{ path = "{zestengine_disp}/crates/zestengine-engine", default-features = false, features = ["client", "steam-audio"] }}
egui = "0.33"
serde = {{ version = "1", features = ["derive"] }}
glam = "0.30"

[dependencies]
zestengine-engine.workspace = true
# The component crate, linked statically (no editor feature -> no editor/egui).
zestengine-project-components = {{ path = "plugins/components" }}
anyhow = "1"

# Lean shipping binary: link-time optimization, one codegen unit, stripped
# symbols, and abort-on-panic (no unwinding tables). ~40% smaller than the
# default release profile.
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
panic = "abort"
"#
        ),
    )
    .context("writing Cargo.toml")?;

    // Editable game entry point: reads the boot scene from project.zestengine and
    // registers this project's components. Edit for custom startup.
    std::fs::write(
        root.join("src/main.rs"),
        format!(
            r#"//! Game entry point. The boot scene comes from project.zestengine
//! (Project Settings -> Starting scene); edit this file for custom startup.

use anyhow::Result;

fn main() -> Result<()> {{
    zestengine_engine::init_logging();

    // Shipped builds carry assets.pak next to the executable, which
    // GameConfig::from_project_dir unpacks and uses; for `cargo run` during
    // development the assets live in the project folder.
    let exe_assets = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("assets")))
        .filter(|p| p.join("project.zestengine").is_file());
    let assets_root = exe_assets
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    let config = zestengine_engine::GameConfig::from_project_dir(assets_root)?;
    zestengine_engine::run_game(config, |registry| {{
        {pkg_us}_components::zestengine_register(registry);
    }})
}}
"#,
            pkg_us = "zestengine_project"
        ),
    )
    .context("writing src/main.rs")?;

    // The component crate (cdylib for editor hot-load + rlib for the game).
    crate::plugins::create_template(&root).context("scaffolding plugins/components")?;
    // create_template defaults to cdylib only; a bundled game links it as rlib.
    add_rlib_crate_type(&root.join("plugins/components/Cargo.toml"))?;

    // Template content (scenes, materials, prefabs, sample assets) + project file.
    let (boot, bindings) =
        crate::templates::write_content(&root, Some(&zestengine), template).context("writing template content")?;
    let project = ProjectFile {
        name: name.to_string(),
        last_scene: Some(boot.clone()),
        boot_scene: Some(boot),
        bindings,
        vr: zestengine_assets::project_templates::find(template)
            .is_some_and(|t| t.category == zestengine_assets::project_templates::TemplateCategory::Vr),
        ..Default::default()
    };
    zestengine_assets::save_project_file(&root, &project).context("writing project.zestengine")?;

    Ok(root)
}

/// Ensure the component crate also produces an `rlib` (for static game linking),
/// not just the editor's `cdylib`.
fn add_rlib_crate_type(manifest: &Path) -> Result<()> {
    let text = std::fs::read_to_string(manifest).context("reading components Cargo.toml")?;
    if text.contains("\"rlib\"") {
        return Ok(());
    }
    let patched = text.replace(
        r#"crate-type = ["cdylib"]"#,
        r#"crate-type = ["cdylib", "rlib"]"#,
    );
    std::fs::write(manifest, patched).context("patching components crate-type")?;
    Ok(())
}

/// Compile the project `--release` and assemble a self-contained `build/`
/// folder: the executable plus `assets.pak`, the cooked referenced assets and
/// `project.zestengine` in one archive (Android packs them into its APK instead).
/// Returns the path of the produced executable.
/// `log` receives human-readable progress lines.
pub fn build_game(
    project_root: &Path,
    project: &ProjectFile,
    target: BuildTarget,
    mut log: impl FnMut(String),
) -> Result<PathBuf> {
    if target != BuildTarget::Server && (!project_root.join("src/main.rs").is_file() || !project_root.join("Cargo.toml").is_file()) {
        bail!(
            "this project has no game entry ({}/src/main.rs). Use File -> New Project to scaffold one.",
            project_root.display()
        );
    }
    if target.is_android() {
        return build_android(project_root, project, target, log);
    }
    let pkg = crate_name(&project.name);
    if target == BuildTarget::Server {
        // The server needs no game entry (src/main.rs); it has its own crate and cook.
        return crate::server_build::build_server(project_root, project, &pkg, log);
    }

    // `cargo rustc` passes the rpath to the game binary only (no dependency
    // rebuild), so it finds the Steam Audio library shipped next to it.
    // Windows loads DLLs from the executable's folder without one.
    let mut args: Vec<&str> = Vec::new();
    if target.is_cross() {
        check_xwin()?;
        args.extend(["xwin", "rustc", "--release", "--target", WINDOWS_TRIPLE]);
    } else {
        args.extend(["rustc", "--release"]);
    }
    args.extend(["--bin", &pkg]);
    if !target.is_windows() {
        if cfg!(target_os = "linux") {
            args.extend(["--", "-C", "link-arg=-Wl,-rpath,$ORIGIN"]);
        } else if cfg!(target_os = "macos") {
            args.extend(["--", "-C", "link-arg=-Wl,-rpath,@executable_path"]);
        }
    }
    log(format!("cargo build --release ({pkg}, {})…", target.label()));
    let mut cargo = std::process::Command::new("cargo");
    // Link the MSVC C/C++ runtime statically (cc builds the C++ deps with /MT
    // to match), so players need no Visual C++ Redistributable installed.
    if target.is_windows() {
        let var = "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS";
        let existing = std::env::var(var).unwrap_or_default();
        cargo.env(var, format!("{existing} -C target-feature=+crt-static").trim());
    }
    let output = cargo
        .current_dir(project_root)
        .args(&args)
        .output()
        .context("running cargo build")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("cargo build failed:\n{}", stderr.trim());
    }

    let exe_name = if target.is_windows() {
        format!("{pkg}.exe")
    } else {
        pkg.clone()
    };
    let profile_dir = if target.is_cross() {
        project_root.join("target").join(WINDOWS_TRIPLE).join("release")
    } else {
        project_root.join("target/release")
    };
    let built = profile_dir.join(&exe_name);
    if !built.is_file() {
        bail!("build succeeded but {} is missing", built.display());
    }

    let build_dir = project_root.join(target.output_dir());
    let assets_dir = build_dir.join("assets");
    // Fresh output each time so deleted assets don't linger.
    let _ = std::fs::remove_dir_all(&build_dir);
    std::fs::create_dir_all(&assets_dir).context("creating build/assets")?;

    log("copying executable…".into());
    let out_exe = build_dir.join(&exe_name);
    std::fs::copy(&built, &out_exe)
        .with_context(|| format!("copying {}", built.display()))?;
    if let Some(lib) = steam_audio_library(&profile_dir, target) {
        log("copying Steam Audio library…".into());
        let name = lib.file_name().expect("library file name");
        std::fs::copy(&lib, build_dir.join(name))
            .with_context(|| format!("copying {}", lib.display()))?;
        // Apache-2.0 requires the licence (and third-party notices) to ship with it.
        std::fs::write(build_dir.join("STEAM_AUDIO_LICENSE.txt"), STEAM_AUDIO_LICENSE)
            .context("writing the Steam Audio licence")?;
        if let Some(notices) = lib
            .parent()
            .and_then(|out_lib| out_lib.parent())
            .map(|out| out.join("steam_audio_cache/steamaudio_core/steamaudio/THIRDPARTY.md"))
            .filter(|p| p.is_file())
        {
            std::fs::copy(&notices, build_dir.join("STEAM_AUDIO_THIRDPARTY.md"))
                .with_context(|| format!("copying {}", notices.display()))?;
        }
    }

    stage_assets(project_root, &assets_dir, target, &mut log)?;
    log("packing assets…".into());
    let packed = zestengine_assets::game_pak::write_game_pak(&assets_dir, &build_dir.join(zestengine_assets::game_pak::GAME_PAK_NAME))?;
    std::fs::remove_dir_all(&assets_dir).context("removing the staged assets")?;
    log(format!("packed {packed} file(s) into {}", zestengine_assets::game_pak::GAME_PAK_NAME));
    log(format!("done: {}", out_exe.display()));
    Ok(out_exe)
}

/// Cook the project's content into `assets_dir`: the referenced files,
/// precompiled shaders and keyword variants, and `project.zestengine`. Android /
/// Quest cooks compile shaders for Vulkan 1.1 and pre-encode ASTC texture
/// caches ([`cook_astc_textures`]).
fn stage_assets(project_root: &Path, assets_dir: &Path, target: BuildTarget, log: &mut impl FnMut(String)) -> Result<()> {
    let _shader_env = target
        .is_android()
        .then(|| zestengine_assets::scoped_shader_target_env(zestengine_assets::ShaderTargetEnv::Vulkan1_1));
    log("cooking assets…".into());
    let cooked = cook_assets(project_root)?;
    for rel in &cooked.files {
        let (from, to) = (project_root.join(rel), assets_dir.join(rel));
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
    }
    for w in &cooked.warnings {
        log(format!("warning: {w}"));
    }
    log(format!(
        "cooked {} file(s), {:.1} MB (left out {} unreferenced file(s), {:.1} MB)",
        cooked.files.len(),
        cooked.bytes as f64 / 1e6,
        cooked.skipped,
        cooked.skipped_bytes as f64 / 1e6
    ));
    if target.is_android() {
        cook_astc_textures(project_root, &cooked.files, assets_dir, log);
        cook_geocaches(project_root, &cooked.files, assets_dir, log);
        cook_sprite_atlases(project_root, &cooked.files, assets_dir, log);
    }
    // Builds run from the editor; the probe writer is editor-only.
        if target == BuildTarget::Quest {
        let n = cook_astc_hdr_bakes(assets_dir, log);
        if n > 0 {
            log(format!("re-encoded {n} baked lighting file(s) as ASTC HDR"));
        }
    }
    log("precompiling shaders…".into());
    precompile_shaders(assets_dir, log);
    let baked = crate::shaders::bake_project_variants(assets_dir, log);
    log(format!("baked {baked} shader keyword variant(s)"));
    // The game reads boot_scene/title from project.zestengine at startup.
    std::fs::copy(
        project_root.join("project.zestengine"),
        assets_dir.join("project.zestengine"),
    )
    .context("copying project.zestengine")?;
    Ok(())
}

/// Meta Quest: re-encode the shipped BC6H baked lighting (lightmap layers,
/// prefiltered reflection-probe chains) as ASTC 4x4 HDR, which its GPU
/// samples directly; BC6H would otherwise decode to uncompressed formats on
/// the device. Uncompressed bakes stay as they are. Failures are logged and
/// leave that file BC6H. Returns the files re-encoded.
fn cook_astc_hdr_bakes(assets_dir: &Path, log: &mut impl FnMut(String)) -> usize {
    use zestengine_assets::hdr_codec::{transcode_array, transcode_cube};
    use zestengine_render::HdrTextureFormat::{AstcHdr4x4, Bc6h};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "lightmap" || e == "reflprobe") {
                out.push(path);
            }
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut files = Vec::new();
    walk(assets_dir, &mut files);
    let mut done = 0;
    for path in files {
        let result: anyhow::Result<bool> = (|| {
            if path.extension().is_some_and(|e| e == "lightmap") {
                let mut layers = zestengine_assets::read_lightmap_layers(&path)?;
                if layers.array.format != Bc6h {
                    return Ok(false);
                }
                layers.array = transcode_array(layers.array, AstcHdr4x4, threads)?;
                zestengine_assets::save_lightmap_layers(&path, &layers)?;
                return Ok(true);
            }
            let mut set = crate::refl_slots::read_reflprobes_stored(&path);
            let mut changed = false;
            for cube in set.entries.values_mut().chain(set.legacy.as_mut()) {
                if let crate::refl_slots::ProbeCube::Chain(chain) = cube
                    && chain.format == Bc6h
                {
                    let empty = zestengine_render::HdrCube { format: Bc6h, size: 0, faces: Default::default() };
                    *chain = transcode_cube(std::mem::replace(chain, empty), AstcHdr4x4, threads)?;
                    changed = true;
                }
            }
            if changed {
                crate::refl_slots::write_reflprobes(&path, &set)?;
            }
            Ok(changed)
        })();
        match result {
            Ok(true) => done += 1,
            Ok(false) => {}
            Err(e) => log(format!("warning: {} stays BC6H: {e:#}", path.display())),
        }
    }
    done
}

/// Android / Quest: pre-encode the ASTC import cache of every (texture,
/// sRGB) pair the cooked scenes, prefabs and materials use, so the device
/// never encodes on first load. The blobs go under
/// [`zestengine_assets::package::COOKED_TEXCACHE_DIR`] (the APK packager drops
/// the hidden per-file `.zestengine_texcache/`) and `android_assets_root`
/// installs them after extraction. Failures are logged, not fatal (that
/// texture is then encoded on the device). Returns the caches written.
fn cook_astc_textures(project_root: &Path, files: &[String], assets_dir: &Path, log: &mut impl FnMut(String)) -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let files: std::collections::BTreeSet<String> = files.iter().cloned().collect();
    let mut warnings = Vec::new();
    let refs: Vec<(String, bool)> =
        zestengine_assets::package::material_texture_refs(project_root, &files, &[], &mut warnings).into_iter().collect();
    for w in warnings {
        log(format!("warning: {w}"));
    }
    if refs.is_empty() {
        return 0;
    }
    let blocks = zestengine_assets::astc_blocks();
    log(format!(
        "encoding {} ASTC texture(s) (colour {b}x{b}, data {d}x{d})…",
        refs.len(),
        b = blocks.color,
        d = blocks.data
    ));
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(refs.len());
    let encode = |(rel, srgb): &(String, bool)| -> Result<u64> {
        let blob = zestengine_assets::astc_cache_blob(project_root.join(rel), *srgb, blocks).with_context(|| rel.clone())?;
        let out = assets_dir.join(zestengine_assets::package::cooked_texcache_rel(rel, *srgb));
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&out, &blob).with_context(|| format!("writing {}", out.display()))?;
        Ok(blob.len() as u64)
    };
    let results: Vec<Result<u64>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    while let Some(r) = refs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        done.push(encode(r));
                    }
                    done
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
    });
    let (mut written, mut bytes) = (0, 0u64);
    for r in results {
        match r {
            Ok(n) => {
                written += 1;
                bytes += n;
            }
            Err(e) => log(format!("warning: ASTC texture not cooked: {e:#}")),
        }
    }
    if written < refs.len() {
        log(format!("warning: {} texture(s) will be encoded on the device", refs.len() - written));
    }
    log(format!("cooked {written} ASTC texture cache(s), {:.1} MB", bytes as f64 / 1e6));
    written
}

/// Android / Quest: ship the cluster DAG ([`zestengine_assets::geo_cache`]) of
/// every cooked model with a mesh big enough for one, so the device never
/// builds it on first load. The blobs go under
/// [`zestengine_assets::geo_cache::COOKED_GEOCACHE_DIR`] (the APK packager drops
/// the hidden per-model `.zestengine_geocache/`) and `android_assets_root`
/// installs them after extraction with `seed_cooked_geocaches`. A model that
/// fails to import is skipped with a warning (the device builds it, or
/// renders it without a DAG). Returns the caches written.
fn cook_geocaches(project_root: &Path, files: &[String], assets_dir: &Path, log: &mut impl FnMut(String)) -> usize {
    use zestengine_assets::{game_pak::MODEL_EXTENSIONS, geo_cache};
    let models: Vec<&String> = files
        .iter()
        .filter(|rel| rel.rsplit_once('.').is_some_and(|(_, e)| MODEL_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str())))
        .collect();
    let (mut written, mut bytes) = (0, 0u64);
    for rel in models {
        let source = project_root.join(rel);
        let scene = match zestengine_assets::load_model_with_meta(&source) {
            Ok(s) => s,
            Err(e) => {
                log(format!("warning: Zestlet data for {rel} not cooked: {e:#}"));
                continue;
            }
        };
        // Reads the project's own geometry cache when it is current, else builds (and stores) it.
        let Some((_, blob)) = geo_cache::cook(&source, &scene) else { continue };
        let out = assets_dir.join(geo_cache::cooked_geocache_rel(rel));
        let result = out
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&out, &blob));
        match result {
            Ok(()) => {
                written += 1;
                bytes += blob.len() as u64;
            }
            Err(e) => log(format!("warning: Zestlet data for {rel} not written: {e}")),
        }
    }
    if written > 0 {
        log(format!("cooked Zestlet data for {written} model(s), {:.1} MB", bytes as f64 / 1e6));
    }
    written
}

/// Android / Quest: pack the project's sprite atlases for the Quest (ASTC pages)
/// and ship them under [`zestengine_assets::sprite_atlas::COOKED_ATLAS_DIR`]
/// (the APK packager drops the hidden `.zestengine_atlascache/`);
/// `android_assets_root` installs them with `seed_cooked_atlases`. An atlas
/// with a member outside the cooked file list is skipped with a warning (the
/// device packs it, or draws its sprites from their own textures). Returns the
/// atlases written.
fn cook_sprite_atlases(project_root: &Path, files: &[String], assets_dir: &Path, log: &mut impl FnMut(String)) -> usize {
    use zestengine_assets::sprite_atlas::{AtlasDb, cooked_atlas_rel};
    let shipped: std::collections::HashSet<&str> = files.iter().map(String::as_str).collect();
    let (blobs, warnings) = AtlasDb::scan(project_root).cooked_blobs(zestengine_assets::Platform::Quest, &|rel| shipped.contains(rel));
    for w in warnings {
        log(format!("warning: sprite atlas {w}"));
    }
    let (mut written, mut bytes) = (0, 0u64);
    for (rel, blob) in blobs {
        let out = assets_dir.join(cooked_atlas_rel(&rel));
        match out.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&out, &blob)) {
            Ok(()) => {
                written += 1;
                bytes += blob.len() as u64;
            }
            Err(e) => log(format!("warning: sprite atlas {rel} not written: {e}")),
        }
    }
    if written > 0 {
        log(format!("cooked {written} sprite atlas(es), {:.1} MB", bytes as f64 / 1e6));
    }
    written
}

const ANDROID_TRIPLE: &str = "aarch64-linux-android";

/// The Android SDK pieces the APK packager needs, found up front so a
/// missing one is a clear error instead of a script failure halfway in.
#[derive(Debug)]
struct AndroidSdk {
    sdk: PathBuf,
    ndk: PathBuf,
}

/// Newest entry (by version-aware name order) of `dir` passing `ok`.
fn newest_child(dir: &Path, ok: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| ok(p)).collect();
    let key = |p: &PathBuf| -> Vec<u64> {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        name.split(|c: char| !c.is_ascii_digit()).filter_map(|n| n.parse().ok()).collect()
    };
    found.sort_by_key(key);
    found.pop()
}

/// Locate the SDK, NDK, build-tools and a platform from `env` (ANDROID_HOME,
/// ANDROID_SDK_ROOT, ANDROID_NDK_HOME, HOME), falling back to /opt/android-sdk
/// and ~/Android/Sdk.
fn find_android_sdk(env: &dyn Fn(&str) -> Option<String>) -> Result<AndroidSdk> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.extend(env("ANDROID_HOME").map(PathBuf::from));
    candidates.extend(env("ANDROID_SDK_ROOT").map(PathBuf::from));
    candidates.push(PathBuf::from("/opt/android-sdk"));
    candidates.extend(env("HOME").map(|h| PathBuf::from(h).join("Android/Sdk")));
    let Some(sdk) = candidates.into_iter().find(|p| p.is_dir()) else {
        bail!("Android SDK not found: install it (build-tools, platforms;android-34, ndk) and set ANDROID_HOME");
    };
    let ndk = match env("ANDROID_NDK_HOME").map(PathBuf::from) {
        Some(ndk) if ndk.is_dir() => ndk,
        Some(ndk) => bail!("ANDROID_NDK_HOME={} does not exist", ndk.display()),
        None => newest_child(&sdk.join("ndk"), |p| p.join("toolchains/llvm").is_dir()).with_context(|| {
            format!("Android NDK not found under {}/ndk: install it with `sdkmanager --install \"ndk;29.0.14206865\"` or set ANDROID_NDK_HOME", sdk.display())
        })?,
    };
    newest_child(&sdk.join("build-tools"), |p| p.join("aapt2").is_file() && p.join("apksigner").is_file()).with_context(|| {
        format!("Android build-tools (aapt2, zipalign, apksigner) not found under {}/build-tools: `sdkmanager --install \"build-tools;35.0.0\"`", sdk.display())
    })?;
    newest_child(&sdk.join("platforms"), |p| p.join("android.jar").is_file()).with_context(|| {
        format!("no Android platform under {}/platforms: `sdkmanager --install \"platforms;android-34\"`", sdk.display())
    })?;
    Ok(AndroidSdk { sdk, ndk })
}

/// The Rust target and the JDK `keytool` (debug signing) must be installed.
fn check_android_host() -> Result<()> {
    let targets = std::process::Command::new("rustup").args(["target", "list", "--installed"]).output();
    if let Ok(out) = targets
        && out.status.success()
        && !String::from_utf8_lossy(&out.stdout).lines().any(|l| l.trim() == ANDROID_TRIPLE)
    {
        bail!("the Rust {ANDROID_TRIPLE} target is missing: run `rustup target add {ANDROID_TRIPLE}`");
    }
    if std::env::var_os("ZESTENGINE_KEYSTORE").is_none()
        && std::process::Command::new("keytool").arg("-help").output().is_err()
    {
        bail!("`keytool` (from a JDK) is needed to make the debug signing key; install a JDK or set ZESTENGINE_KEYSTORE");
    }
    Ok(())
}

/// A Quest build ships the OpenXR loader: `OPENXR_LOADER_SO`, or the pinned
/// Khronos release the packager fetches with curl.
fn check_openxr_loader(env: &dyn Fn(&str) -> Option<String>, curl_available: bool) -> Result<()> {
    if let Some(so) = env("OPENXR_LOADER_SO") {
        if !Path::new(&so).is_file() {
            bail!("OPENXR_LOADER_SO={so} does not exist");
        }
        return Ok(());
    }
    if env("ZESTENGINE_OPENXR_FETCH").as_deref() == Some("0") {
        bail!(
            "a Meta Quest build needs the OpenXR loader: set OPENXR_LOADER_SO to an arm64-v8a libopenxr_loader.so \
             (from the Khronos openxr_loader_for_android AAR), or unset ZESTENGINE_OPENXR_FETCH to download the pinned release"
        );
    }
    if !curl_available {
        bail!("a Meta Quest build needs the OpenXR loader: install curl (to fetch the pinned Khronos release) or set OPENXR_LOADER_SO");
    }
    Ok(())
}

/// A Java-legal application id from the project name, unless
/// `ZESTENGINE_ANDROID_PACKAGE` overrides it: `com.zestengine.<name>`.
fn android_package_id(project_name: &str) -> String {
    let mut seg: String = project_name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if seg.is_empty() {
        seg = "game".into();
    }
    if seg.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        seg.insert(0, 'g');
    }
    format!("com.zestengine.{seg}")
}

/// The generated `android_main` crate: links the engine runtime and the
/// project's components, and boots from the APK's extracted assets.
fn android_crate_sources(zestengine: &Path, project_root: &Path, pkg: &str, target: BuildTarget) -> (String, String) {
    let lib = pkg.replace('-', "_");
    let manifest = format!(
        r#"# Generated by zestengine Build Game ({label}); rewritten on every build.
[package]
name = "{pkg}-android"
version = "0.1.0"
edition = "2024"

[workspace]

[lib]
name = "{lib}"
crate-type = ["cdylib"]
path = "src/lib.rs"

[dependencies]
zestengine-engine = {{ path = "{zestengine}/crates/zestengine-engine", default-features = false, features = ["client"] }}
zestengine-project-components = {{ path = "{components}" }}
tracing = "0.1"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
panic = "abort"
"#,
        label = target.label(),
        zestengine = zestengine.display(),
        components = project_root.join("plugins/components").display(),
    );
    let xr = if target == BuildTarget::Quest { "Required" } else { "Auto" };
    let source = format!(
        r#"//! Generated by zestengine Build Game ({label}); rewritten on every build.

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(app: zestengine_engine::AndroidApp) {{
    zestengine_engine::init_logging();
    let root = match zestengine_engine::android_assets_root(&app) {{
        Ok(root) => root,
        Err(e) => {{
            tracing::error!("extracting game assets: {{e:#}}");
            return;
        }}
    }};
    let mut config = match zestengine_engine::GameConfig::from_project_dir(root) {{
        Ok(config) => config,
        Err(e) => {{
            tracing::error!("reading project.zestengine: {{e:#}}");
            return;
        }}
    }};
    config.xr = zestengine_engine::XrMode::{xr};
    if let Err(e) = zestengine_engine::run_game_android(app, config, zestengine_project_components::zestengine_register) {{
        tracing::error!("game exited: {{e:#}}");
    }}
}}
"#,
        label = target.label(),
    );
    (manifest, source)
}

/// Android / Meta Quest: cook the assets, generate the `android_main` crate
/// and run `tools/android/build_apk.sh`. Returns the signed APK.
fn build_android(
    project_root: &Path,
    project: &ProjectFile,
    target: BuildTarget,
    mut log: impl FnMut(String),
) -> Result<PathBuf> {
    if cfg!(windows) {
        bail!("Android builds need a Linux or macOS host (the APK packager is a bash script)");
    }
    let env = |k: &str| std::env::var(k).ok();
    let sdk = find_android_sdk(&env)?;
    check_android_host()?;
    if target == BuildTarget::Quest {
        let curl = std::process::Command::new("curl").arg("--version").output().is_ok_and(|o| o.status.success());
        check_openxr_loader(&env, curl)?;
    }
    let components = project_root.join("plugins/components/Cargo.toml");
    if !components.is_file() {
        bail!("{} is missing; the game's components crate is linked into the APK", components.display());
    }
    let zestengine = zestengine_root()?;
    let script = zestengine.join("tools/android/build_apk.sh");
    if !script.is_file() {
        bail!("APK packager not found at {}", script.display());
    }
    let pkg = crate_name(&project.name);

    let build_dir = project_root.join(target.output_dir());
    let assets_dir = build_dir.join("assets");
    let _ = std::fs::remove_dir_all(&assets_dir);
    std::fs::create_dir_all(&assets_dir).with_context(|| format!("creating {}", assets_dir.display()))?;
    stage_assets(project_root, &assets_dir, target, &mut log)?;

    let crate_dir = project_root.join("target/zestengine-android");
    std::fs::create_dir_all(crate_dir.join("src")).context("creating the Android crate")?;
    let (manifest, source) = android_crate_sources(&zestengine, project_root, &pkg, target);
    std::fs::write(crate_dir.join("Cargo.toml"), manifest).context("writing the Android Cargo.toml")?;
    std::fs::write(crate_dir.join("src/lib.rs"), source).context("writing the Android lib.rs")?;
    // Seed the lockfile from the engine's so the APK builds against the same
    // dependency versions as the editor.
    let lock = crate_dir.join("Cargo.lock");
    if !lock.is_file() && zestengine.join("Cargo.lock").is_file() {
        std::fs::copy(zestengine.join("Cargo.lock"), &lock).context("seeding the Android Cargo.lock")?;
    }

    let apk = build_dir.join(format!("{pkg}.apk"));
    let package_id = env("ZESTENGINE_ANDROID_PACKAGE").unwrap_or_else(|| android_package_id(&project.name));
    log(format!("building {} ({package_id}) for {}… (the first build compiles the engine for {ANDROID_TRIPLE})", apk.display(), target.label()));
    let mut cmd = std::process::Command::new("bash");
    cmd.arg(&script);
    if target == BuildTarget::Quest {
        cmd.arg("--quest");
    }
    cmd.arg(&crate_dir)
        .arg(&assets_dir)
        .arg(&apk)
        .arg(&package_id)
        .env("ANDROID_HOME", &sdk.sdk)
        .env("ANDROID_NDK_HOME", &sdk.ndk)
        .env("CARGO_TARGET_DIR", project_root.join("target"))
        .env("ZESTENGINE_APP_LABEL", &project.name);
    let output = cmd.output().context("running tools/android/build_apk.sh")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.trim().lines().rev().take(60).collect();
        bail!("APK build failed:\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
    }
    if !apk.is_file() {
        bail!("the packager succeeded but {} is missing", apk.display());
    }
    log(format!("done: {} (install: adb install -r {})", apk.display(), apk.display()));
    Ok(apk)
}

/// Steam Audio's licence text, shipped next to `libphonon` in game builds.
const STEAM_AUDIO_LICENSE: &str = include_str!("../assets/licenses/steam-audio-LICENSE.txt");

/// The Steam Audio shared library built for this profile (newest
/// `build/audionimbus-sys-*/out/lib` copy), if the game uses it.
fn steam_audio_library(profile_dir: &Path, target: BuildTarget) -> Option<PathBuf> {
    let name = if target.is_windows() {
        "phonon.dll"
    } else if cfg!(target_os = "macos") {
        "libphonon.dylib"
    } else {
        "libphonon.so"
    };
    std::fs::read_dir(profile_dir.join("build"))
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("audionimbus-sys-"))
        .map(|e| e.path().join("out/lib").join(name))
        .filter(|p| p.is_file())
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok())
}

/// Cross-compiling to Windows needs `cargo-xwin` (downloads the MSVC CRT and
/// Windows SDK on first use) and the rustup target.
fn check_xwin() -> Result<()> {
    let xwin = std::process::Command::new("cargo")
        .args(["xwin", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !xwin {
        bail!(
            "Windows builds from this platform need cargo-xwin: run `cargo install --locked cargo-xwin` \
             and `rustup target add {WINDOWS_TRIPLE}`"
        );
    }
    let targets = std::process::Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output();
    if let Ok(out) = targets
        && out.status.success()
        && !String::from_utf8_lossy(&out.stdout).lines().any(|l| l.trim() == WINDOWS_TRIPLE)
    {
        bail!("the Rust {WINDOWS_TRIPLE} target is missing: run `rustup target add {WINDOWS_TRIPLE}`");
    }
    Ok(())
}

/// Compile every custom `.frag` under `dir` to a `.frag.spv` sidecar so the
/// shipped game never needs `glslc`. Failures are logged, not fatal (the game
/// then falls back to runtime compilation for that shader).
fn precompile_shaders(dir: &Path, log: &mut impl FnMut(String)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            precompile_shaders(&path, log);
        } else if zestengine_assets::is_shader_path(&path)
            && let Err(e) = zestengine_assets::precompile_shader_file(&path) {
                log(format!("warning: shader {} not precompiled: {e:#}", path.display()));
            }
    }
}

/// Every top-level content folder of a project: all subdirectories except
/// code / build output and hidden ones (`.git`, `.zestengine` caches).
pub(crate) fn asset_dirs(project_root: &Path) -> Result<Vec<String>> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(project_root)
        .with_context(|| format!("reading {}", project_root.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || NON_ASSET_DIRS.contains(&name.as_str()) {
            continue;
        }
        dirs.push(name);
    }
    dirs.sort();
    Ok(dirs)
}

/// Build a project given only its root: loads `project.zestengine` and logs to
/// stdout. Convenience for the CLI (`zestengine --build <dir>`).
pub fn build_project_dir(project_root: &Path, target: BuildTarget) -> Result<PathBuf> {
    let project =
        zestengine_assets::load_project_file(project_root).context("reading project.zestengine")?;
    build_game(project_root, &project, target, |m| println!("build: {m}"))
}

/// What a build ships: every file reachable from the project's scenes,
/// prefabs, its `resources/` folder and paths named in its code.
pub struct CookedAssets {
    pub files: Vec<String>,
    pub bytes: u64,
    pub skipped: usize,
    pub skipped_bytes: u64,
    pub warnings: Vec<String>,
}

/// Work out which content files a build needs. Roots: every `.scene` and
/// `.prefab`, everything under `resources/` (loaded by path at run time),
/// and any project file whose path appears as a string literal in `src/`.
/// Dependencies are followed through every RON asset, model sidecar and
/// external model file (the content-package scanner).
pub fn cook_assets(project_root: &Path) -> Result<CookedAssets> {
    let mut all = Vec::new();
    for dir in asset_dirs(project_root)? {
        list_files(project_root, &project_root.join(&dir), &mut all);
    }
    let mut roots: Vec<String> = all
        .iter()
        .filter(|r| r.ends_with(".scene") || r.ends_with(".prefab") || r.starts_with("resources/"))
        .cloned()
        .collect();
    let mut code = String::new();
    let mut src_files = Vec::new();
    list_files(project_root, &project_root.join("src"), &mut src_files);
    for f in src_files.iter().filter(|f| f.ends_with(".rs")) {
        code.push_str(&std::fs::read_to_string(project_root.join(f)).unwrap_or_default());
    }
    roots.extend(all.iter().filter(|r| code.contains(&format!("\"{r}\""))).cloned());
    let mut keep = std::collections::BTreeSet::new();
    let mut warnings = Vec::new();
    for root in roots {
        let text = std::fs::read_to_string(project_root.join(&root)).unwrap_or_default();
        let (deps, w) = zestengine_assets::package::collect_dependencies(project_root, &root, &text);
        keep.extend(deps);
        keep.insert(root);
        warnings.extend(w.into_iter().filter(|w| !w.contains("not allowed in content packages")));
    }
    // No scene names a sprite atlas: ship each in-build atlas with its member textures.
    keep.extend(zestengine_assets::sprite_atlas::AtlasDb::scan(project_root).build_files());
    let size = |r: &String| std::fs::metadata(project_root.join(r)).map(|m| m.len()).unwrap_or(0);
    let files: Vec<String> = keep.into_iter().filter(|r| project_root.join(r).is_file()).collect();
    let skipped: Vec<&String> = all.iter().filter(|r| !files.contains(r)).collect();
    Ok(CookedAssets {
        bytes: files.iter().map(size).sum(),
        skipped: skipped.len(),
        skipped_bytes: skipped.iter().map(|r| size(r)).sum(),
        files,
        warnings,
    })
}

/// Project-relative paths of every file under `dir` (hidden entries skipped).
pub(crate) fn list_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            list_files(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

#[cfg(test)]
mod android_tests {
    use super::*;

    #[test]
    fn build_targets_parse_and_place_output() {
        assert_eq!(BuildTarget::parse("android"), Some(BuildTarget::Android));
        assert_eq!(BuildTarget::parse("aarch64-linux-android"), Some(BuildTarget::Android));
        assert_eq!(BuildTarget::parse("Quest"), Some(BuildTarget::Quest));
        assert_eq!(BuildTarget::parse("meta-quest"), Some(BuildTarget::Quest));
        assert_eq!(BuildTarget::parse("win64"), Some(BuildTarget::Windows));
        assert_eq!(BuildTarget::parse("ps5"), None);
        assert!(BuildTarget::Android.is_android() && BuildTarget::Quest.is_android());
        assert!(!BuildTarget::Host.is_android() && !BuildTarget::Windows.is_android());
        assert!(!BuildTarget::Quest.is_windows() && !BuildTarget::Quest.is_cross());
        assert_eq!(BuildTarget::Android.output_dir(), "build-android");
        assert_eq!(BuildTarget::Quest.output_dir(), "build-quest");
        for t in BuildTarget::ALL {
            assert!(NON_ASSET_DIRS.contains(&t.output_dir()), "{t:?} output would be cooked as content");
        }
        let labels: std::collections::BTreeSet<_> = BuildTarget::ALL.iter().map(|t| t.label()).collect();
        assert_eq!(labels.len(), BuildTarget::ALL.len());
        let available: Vec<_> = BuildTarget::available().collect();
        assert!(available.contains(&BuildTarget::Host));
        assert_eq!(available.contains(&BuildTarget::Quest), !cfg!(windows));
    }

    #[test]
    fn package_ids_are_java_legal() {
        assert_eq!(android_package_id("My Game"), "com.zestengine.my_game");
        assert_eq!(android_package_id("  VR--Shooter 2! "), "com.zestengine.vr_shooter_2");
        assert_eq!(android_package_id("3D Maze"), "com.zestengine.g3d_maze");
        assert_eq!(android_package_id("***"), "com.zestengine.game");
    }

    #[test]
    fn android_sdk_detection_reports_what_is_missing() {
        let root = std::env::temp_dir().join(format!("zestengine-android-sdk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sdk = root.join("sdk");
        std::fs::create_dir_all(&sdk).unwrap();
        let home = root.join("home");
        let sdk_s = sdk.to_string_lossy().into_owned();
        let home_s = home.to_string_lossy().into_owned();
        let env = |k: &str| match k {
            "ANDROID_HOME" => Some(sdk_s.clone()),
            "HOME" => Some(home_s.clone()),
            _ => None,
        };
        let err = format!("{:#}", find_android_sdk(&env).unwrap_err());
        assert!(err.contains("NDK"), "{err}");
        for ndk in ["27.1.12297006", "29.0.14206865"] {
            std::fs::create_dir_all(sdk.join("ndk").join(ndk).join("toolchains/llvm")).unwrap();
        }
        let err = format!("{:#}", find_android_sdk(&env).unwrap_err());
        assert!(err.contains("build-tools"), "{err}");
        let bt = sdk.join("build-tools/35.0.0");
        std::fs::create_dir_all(&bt).unwrap();
        std::fs::write(bt.join("aapt2"), "").unwrap();
        std::fs::write(bt.join("apksigner"), "").unwrap();
        let err = format!("{:#}", find_android_sdk(&env).unwrap_err());
        assert!(err.contains("platform"), "{err}");
        std::fs::create_dir_all(sdk.join("platforms/android-34")).unwrap();
        std::fs::write(sdk.join("platforms/android-34/android.jar"), "").unwrap();
        let found = find_android_sdk(&env).unwrap();
        assert_eq!(found.sdk, sdk);
        assert!(found.ndk.ends_with("29.0.14206865"), "{:?}", found.ndk);
        let bad_ndk = |k: &str| match k {
            "ANDROID_HOME" => Some(sdk_s.clone()),
            "ANDROID_NDK_HOME" => Some("/nonexistent/ndk".into()),
            _ => None,
        };
        assert!(format!("{:#}", find_android_sdk(&bad_ndk).unwrap_err()).contains("ANDROID_NDK_HOME"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn openxr_loader_sources() {
        let none = |_: &str| None;
        assert!(check_openxr_loader(&none, true).is_ok());
        assert!(format!("{:#}", check_openxr_loader(&none, false).unwrap_err()).contains("curl"));
        let no_fetch = |k: &str| (k == "ZESTENGINE_OPENXR_FETCH").then(|| "0".to_string());
        assert!(format!("{:#}", check_openxr_loader(&no_fetch, true).unwrap_err()).contains("OPENXR_LOADER_SO"));
        let missing = |k: &str| (k == "OPENXR_LOADER_SO").then(|| "/nonexistent/libopenxr_loader.so".to_string());
        assert!(check_openxr_loader(&missing, true).is_err());
    }

    #[test]
    fn generated_android_crate_links_components_and_picks_xr() {
        let (manifest, quest) = android_crate_sources(Path::new("/src/zestengine"), Path::new("/p/game"), "my-game", BuildTarget::Quest);
        assert!(manifest.contains(r#"name = "my_game""#), "{manifest}");
        assert!(manifest.contains(r#"crate-type = ["cdylib"]"#));
        assert!(manifest.contains("/src/zestengine/crates/zestengine-engine"));
        assert!(manifest.contains("default-features = false"));
        assert!(manifest.contains("/p/game/plugins/components"));
        assert!(manifest.contains("[workspace]"));
        assert!(quest.contains("fn android_main"));
        assert!(quest.contains("XrMode::Required"));
        assert!(quest.contains("zestengine_project_components::zestengine_register"));
        let (_, phone) = android_crate_sources(Path::new("/src/zestengine"), Path::new("/p/game"), "my-game", BuildTarget::Android);
        assert!(phone.contains("XrMode::Auto"));
    }

    /// The packager's manifests (`--manifest-only`), checked against what
    /// Meta Quest and the Khronos loader require.
    #[cfg(unix)]
    #[test]
    fn apk_manifests() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let script = repo.join("tools/android/build_apk.sh");
        let example = repo.join("examples/android-game");
        // `tools/` is git-ignored, so a checkout may not have the packager.
        if !script.is_file() {
            eprintln!("skipping apk_manifests: {} is not in this checkout", script.display());
            return;
        }
        let manifest = |quest: bool| -> Option<String> {
            let mut cmd = std::process::Command::new("bash");
            cmd.arg(&script).arg("--manifest-only");
            if quest {
                cmd.arg("--quest");
            }
            let out = cmd.arg(&example).arg("unused").arg("unused.apk").arg("com.zestengine.test").output().ok()?;
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            Some(String::from_utf8(out.stdout).unwrap())
        };
        let Some(quest) = manifest(true) else {
            return; // no bash
        };
        for needle in [
            r#"package="com.zestengine.test""#,
            r#"android:minSdkVersion="29""#,
            r#"android:targetSdkVersion="32""#,
            r#"<category android:name="com.oculus.intent.category.VR" />"#,
            r#"<category android:name="android.intent.category.LAUNCHER" />"#,
            r#"android:name="android.hardware.vr.headtracking" android:version="1" android:required="true""#,
            r#"android:name="com.oculus.supportedDevices" android:value="quest3|quest3s|questpro""#,
            r#"android:name="oculus.software.handtracking" android:required="false""#,
            r#"android:name="oculus.software.eye_tracking" android:required="false""#,
            r#"android:name="oculus.software.face_tracking" android:required="false""#,
            "com.oculus.permission.HAND_TRACKING",
            "com.oculus.permission.EYE_TRACKING",
            "com.oculus.permission.FACE_TRACKING",
            "org.khronos.openxr.permission.OPENXR",
            "org.khronos.openxr.runtime_broker",
            r#"android:name="com.oculus.vr.focusaware" android:value="true""#,
            r#"android:screenOrientation="landscape""#,
            r#"android:name="android.app.lib_name" android:value="android_game""#,
        ] {
            assert!(quest.contains(needle), "Quest manifest lacks {needle}:\n{quest}");
        }
        let phone = manifest(false).unwrap();
        assert!(!phone.contains("oculus"), "{phone}");
        assert!(phone.contains("android.intent.category.LAUNCHER"));
        assert!(phone.contains(r#"android:minSdkVersion="30""#));
    }

    #[test]
    fn quest_cook_reencodes_bc6h_bakes_as_astc_hdr() {
        use zestengine_render::HdrTextureFormat::{AstcHdr4x4, Bc6h, Rgba32f};
        let dir = std::env::temp_dir().join(format!("zestengine-quest-bakes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("scenes")).unwrap();
        let texels: Vec<u8> = (0..64).flat_map(|_| [1.25f32, 0.5, 0.25, 1.0]).flat_map(f32::to_le_bytes).collect();
        let bc6h = zestengine_assets::hdr_codec::transcode_level(&texels, Rgba32f, Bc6h, 8, 8, 1).unwrap();
        let lightmap = dir.join("scenes/main.lightmap");
        let layers = zestengine_assets::LightmapLayers {
            objects: vec![1],
            sizes: vec![8],
            array: zestengine_render::HdrTextureArray { format: Bc6h, size: 8, layers: vec![bc6h.clone()] },
        };
        zestengine_assets::save_lightmap_layers(&lightmap, &layers).unwrap();
        let probes = dir.join("scenes/main.reflprobe");
        let mip = |s: u32| {
            let texels: Vec<u8> = (0..s * s).flat_map(|_| [1.25f32, 0.5, 0.25, 1.0]).flat_map(f32::to_le_bytes).collect();
            zestengine_assets::hdr_codec::transcode_level(&texels, Rgba32f, Bc6h, s, s, 1).unwrap()
        };
        let faces: [Vec<Vec<u8>>; 6] = std::array::from_fn(|_| [8, 4, 2, 1].map(mip).to_vec());
        let mut set = crate::refl_slots::ReflProbeSet::default();
        set.entries.insert(7, crate::refl_slots::ProbeCube::Chain(zestengine_render::HdrCube { format: Bc6h, size: 8, faces }));
        crate::refl_slots::write_reflprobes(&probes, &set).unwrap();

        assert_eq!(crate::refl_slots::read_reflprobes_stored(&probes).entries[&7].format(), Bc6h, "a valid BC6H chain");
        let mut warnings = Vec::new();
        let mut log = |w: String| warnings.push(w);
        let n = cook_astc_hdr_bakes(&dir, &mut log);
        assert_eq!(n, 2, "{warnings:?}");
        assert_eq!(zestengine_assets::read_lightmap_layers(&lightmap).unwrap().array.format, AstcHdr4x4);
        assert_eq!(crate::refl_slots::read_reflprobes_stored(&probes).entries[&7].format(), AstcHdr4x4);
        assert_eq!(cook_astc_hdr_bakes(&dir, &mut log), 0, "already ASTC: left alone");
        let _ = std::fs::remove_dir_all(&dir);
    }
}