//! The dedicated-server build target (NET-3): a stripped cook, a project server
//! crate and a ready-to-run folder.
//!
//! `build_server` produces `build-server/`:
//!
//! ```text
//! zestengine-server     the server binary (the project's own, when it has components)
//! server.ron            ready settings (project: "assets", boot scene, name)
//! run.sh                cd into the folder and start with server.ron
//! deploy/               systemd units, production server.ron, Dockerfile, compose, firewall,
//!                       logrotate, install.sh (`server_deploy`, NET-7)
//! assets/               the stripped cook (loose files, read directly by the server)
//! build-report.txt      sizes, what was dropped, reference warnings
//! ```
//!
//! The cook ([`cook_server_assets`]) keeps scenes and prefabs with Client-role
//! content removed (`zestengine_assets::net_strip`), replaces models with
//! collision-only blobs (`zestengine_assets::server_model`), and drops every file
//! a server never reads (textures, audio, shaders, materials, fonts, baked
//! lighting, caches). Reachability is recomputed from the stripped scenes, so a
//! model used only by a stripped object is not shipped either.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use zestengine_assets::net_strip::{self, is_client_only_file};
use zestengine_assets::package::{collect_dependencies, normalize_rel, visit_ron_strings};
use zestengine_assets::{ProjectFile, server_model};
use zestengine_core::{ComponentRegistry, NetRole};

/// File name of the binary in the output folder.
pub const SERVER_BINARY: &str = "zestengine-server";
/// Folder (inside the output folder) holding the stripped assets.
pub const ASSETS_DIR: &str = "assets";
pub const REPORT_NAME: &str = "build-report.txt";

/// What the cook did.
#[derive(Debug, Default)]
pub struct ServerCook {
    /// Files / bytes a client build of the same project ships.
    pub client_files: usize,
    pub client_bytes: u64,
    /// Files / bytes in the server's assets.
    pub files: usize,
    pub bytes: u64,
    /// Dropped files by class: (class, files, bytes of the originals).
    pub dropped: Vec<(&'static str, usize, u64)>,
    /// Models rewritten to collision blobs: count, source bytes, blob bytes.
    pub models: (usize, u64, u64),
    /// Per scene or prefab: (path, object branches removed, components removed).
    pub stripped: Vec<(String, usize, usize)>,
    /// Kept content that references stripped content, one line each.
    pub lints: Vec<String>,
    pub warnings: Vec<String>,
}

fn size_of(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

/// Which report class a dropped file belongs to.
fn class_of(rel: &str) -> &'static str {
    let lower = rel.to_ascii_lowercase();
    let ext = lower.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if lower.split('/').any(|p| p.starts_with(".zestengine_")) {
        return "caches";
    }
    match ext {
        "png" | "jpg" | "jpeg" | "tga" | "bmp" | "gif" | "webp" | "exr" | "hdr" | "dds" | "ktx" | "ktx2" | "psd" | "tif" | "tiff" | "astc" | "atlas" => "textures",
        "wav" | "ogg" | "mp3" | "flac" | "opus" | "aac" | "m4a" | "aif" | "aiff" => "audio",
        "frag" | "vert" | "comp" | "glsl" | "hlsl" | "wgsl" | "spv" | "shadergraph" => "shaders",
        "material" | "matcap" | "postfx" => "materials",
        "ttf" | "otf" | "woff" | "woff2" => "fonts",
        "lightmap" | "lightdata" | "occlusion" | "reflprobe" | "audiobake" | "lmuv" => "baked lighting",
        "meta" => "sidecars of dropped files",
        _ => "other",
    }
}

const MODEL_EXT: &[&str] = &["glb", "gltf", "vrm", "fbx"];

fn ext_of(rel: &str) -> String {
    rel.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default()
}

/// Project files a RON text references (plain paths and `@id` asset refs).
fn direct_refs(project_root: &Path, text: &str, out: &mut BTreeSet<String>) {
    visit_ron_strings(text, &mut |s| {
        let resolved;
        let s = if zestengine_core::asset_ref::is_ref(s) {
            match zestengine_core::asset_ref::resolve(project_root, s) {
                Some(r) => {
                    resolved = r;
                    resolved.as_str()
                }
                None => return,
            }
        } else {
            s
        };
        if let Some(rel) = normalize_rel(s)
            && project_root.join(&rel).is_file()
        {
            out.insert(rel);
        }
    });
}

/// Rewrite a scene for the server: prefab links expanded, Client content stripped.
fn strip_scene(project_root: &Path, rel: &str, registry: &ComponentRegistry, cook: &mut ServerCook) -> Result<String> {
    let mut file = zestengine_assets::load_scene_file(project_root.join(rel))?;
    crate::prefab::expand_scene_file(&mut file, project_root, &Default::default());
    for e in &mut file.entries {
        e.prefab = None;
        e.prefab_parent = None;
    }
    let report = net_strip::strip_entries(&mut file.entries, NetRole::Server, registry);
    note_strip(cook, rel, &report, &file.entries);
    file.skybox = None;
    file.environment.skybox_faces = None;
    file.editor_camera = None;
    file.collapsed.clear();
    file.strip_editor_view();
    ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).context("serialising the stripped scene")
}

fn strip_prefab(project_root: &Path, rel: &str, registry: &ComponentRegistry, cook: &mut ServerCook) -> Result<String> {
    let mut prefab = crate::prefab::Prefab::load(project_root.join(rel))?;
    crate::prefab::expand_entries(&mut prefab.entries, project_root);
    for e in &mut prefab.entries {
        e.prefab = None;
        e.prefab_parent = None;
    }
    let report = net_strip::strip_entries(&mut prefab.entries, NetRole::Server, registry);
    note_strip(cook, rel, &report, &prefab.entries);
    Ok(prefab.to_ron())
}

fn note_strip(cook: &mut ServerCook, rel: &str, report: &net_strip::StripReport, kept: &[zestengine_assets::SceneEntry]) {
    cook.stripped.push((rel.to_owned(), report.objects.len(), report.components.len()));
    for d in report.dangling.iter().cloned().chain(net_strip::lint_asset_refs(kept)) {
        cook.lints.push(format!("{rel}: {}", d.describe()));
    }
}

/// Cook the project's server assets into `out`. `registry` supplies the default
/// roles of component types (the built-ins, plus the project's when the caller
/// has them loaded).
pub fn cook_server_assets(project_root: &Path, out: &Path, registry: &ComponentRegistry, log: &mut impl FnMut(String)) -> Result<ServerCook> {
    let mut cook = ServerCook::default();
    let client = crate::bundle::cook_assets(project_root)?;
    cook.client_files = client.files.len();
    cook.client_bytes = client.bytes;

    let mut all = Vec::new();
    for dir in crate::bundle::asset_dirs(project_root)? {
        crate::bundle::list_files(project_root, &project_root.join(&dir), &mut all);
    }
    all.sort();

    // Scenes and prefabs: rewritten, never copied.
    let mut rewritten: BTreeMap<String, String> = BTreeMap::new();
    for rel in all.iter().filter(|r| r.ends_with(".scene")) {
        match strip_scene(project_root, rel, registry, &mut cook) {
            Ok(text) => {
                rewritten.insert(rel.clone(), text);
            }
            Err(e) => cook.warnings.push(format!("{rel}: kept unstripped ({e:#})")),
        }
    }
    for rel in all.iter().filter(|r| r.ends_with(".prefab")) {
        match strip_prefab(project_root, rel, registry, &mut cook) {
            Ok(text) => {
                rewritten.insert(rel.clone(), text);
            }
            Err(e) => cook.warnings.push(format!("{rel}: kept unstripped ({e:#})")),
        }
    }

    // Roots: the rewritten files, resources/, and paths named in code.
    let mut direct = BTreeSet::new();
    for text in rewritten.values() {
        direct_refs(project_root, text, &mut direct);
    }
    let mut roots: BTreeSet<String> = all.iter().filter(|r| r.starts_with("resources/")).cloned().collect();
    let mut code = String::new();
    for dir in ["src", "plugins/components/src"] {
        let mut files = Vec::new();
        crate::bundle::list_files(project_root, &project_root.join(dir), &mut files);
        for f in files.iter().filter(|f| f.ends_with(".rs")) {
            code.push_str(&std::fs::read_to_string(project_root.join(f)).unwrap_or_default());
        }
    }
    roots.extend(all.iter().filter(|r| code.contains(&format!("\"{r}\""))).cloned());
    direct.extend(roots.iter().cloned());

    // Transitive closure through the files that are copied (animation clips,
    // controllers, data tables, models' sidecars ...).
    let mut keep: BTreeSet<String> = rewritten.keys().cloned().collect();
    let mut externals: BTreeSet<String> = BTreeSet::new();
    for d in &direct {
        if rewritten.contains_key(d) || d.ends_with(".scene") || d.ends_with(".prefab") {
            continue;
        }
        let text = if d.ends_with(".meta") || zestengine_assets::package::ALLOWED_EXTENSIONS.contains(&ext_of(d).as_str()) && !MODEL_EXT.contains(&ext_of(d).as_str()) {
            std::fs::read_to_string(project_root.join(d)).unwrap_or_default()
        } else {
            String::new()
        };
        let (deps, _) = collect_dependencies(project_root, d, &text);
        if MODEL_EXT.contains(&ext_of(d).as_str()) {
            externals.extend(deps.iter().filter(|x| *x != d && !x.ends_with(".meta")).cloned());
        }
        keep.extend(deps);
    }
    // Buffers and textures a model file pulls in are inside its blob.
    for e in &externals {
        if !direct.contains(e) {
            keep.remove(e);
        }
    }
    let (kept, dropped): (Vec<String>, Vec<String>) = all.iter().cloned().partition(|r| {
        let class_drop = is_client_only_file(r);
        keep.contains(r) && !class_drop
    });
    // Rewritten files outside asset dirs would not be in `all`; `keep` only holds project files.
    let mut by_class: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
    for r in &dropped {
        let slot = by_class.entry(class_of(r)).or_default();
        slot.0 += 1;
        slot.1 += size_of(&project_root.join(r));
    }
    cook.dropped = by_class.into_iter().map(|(c, (n, b))| (c, n, b)).collect();

    let _ = std::fs::remove_dir_all(out);
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    for rel in &kept {
        let dst = out.join(rel);
        if let Some(dir) = dst.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if let Some(text) = rewritten.get(rel) {
            std::fs::write(&dst, text).with_context(|| format!("writing {rel}"))?;
        } else if MODEL_EXT.contains(&ext_of(rel).as_str()) {
            match server_model::cook_model(&project_root.join(rel), &dst) {
                Ok((src, blob)) => {
                    cook.models.0 += 1;
                    cook.models.1 += src;
                    cook.models.2 += blob;
                }
                Err(e) => {
                    cook.warnings.push(format!("{rel}: shipped whole, the collision cook failed ({e:#})"));
                    std::fs::copy(project_root.join(rel), &dst).with_context(|| format!("copying {rel}"))?;
                }
            }
        } else {
            std::fs::copy(project_root.join(rel), &dst).with_context(|| format!("copying {rel}"))?;
        }
    }
    std::fs::copy(project_root.join("project.zestengine"), out.join("project.zestengine")).context("copying project.zestengine")?;
    let mut written = Vec::new();
    crate::bundle::list_files(out, out, &mut written);
    cook.files = written.len();
    cook.bytes = written.iter().map(|r| size_of(&out.join(r))).sum();
    log(format!(
        "server assets: {} file(s), {:.2} MB (a client build ships {} file(s), {:.2} MB)",
        cook.files,
        mb(cook.bytes),
        cook.client_files,
        mb(cook.client_bytes)
    ));
    for l in &cook.lints {
        log(format!("warning: {l}"));
    }
    Ok(cook)
}

/// `build-report.txt` for a finished cook.
pub fn report_text(project_name: &str, cook: &ServerCook, binary: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!("ZestEngine dedicated server build: {project_name}\n"));
    s.push_str(&format!("Binary: {binary}\n\n"));
    let pct = if cook.client_bytes > 0 { cook.bytes as f64 * 100.0 / cook.client_bytes as f64 } else { 0.0 };
    s.push_str("Asset sizes\n");
    s.push_str(&format!("  client build : {:>6} file(s) {:>10.2} MB\n", cook.client_files, mb(cook.client_bytes)));
    s.push_str(&format!("  server build : {:>6} file(s) {:>10.2} MB ({pct:.1}% of the client's)\n", cook.files, mb(cook.bytes)));
    if cook.models.0 > 0 {
        s.push_str(&format!("  models       : {} collision-only blob(s), {:.2} MB -> {:.2} MB\n", cook.models.0, mb(cook.models.1), mb(cook.models.2)));
    }
    s.push_str("\nDropped from the server\n");
    if cook.dropped.is_empty() {
        s.push_str("  (nothing)\n");
    }
    for (class, n, b) in &cook.dropped {
        s.push_str(&format!("  {class:<26} {n:>5} file(s) {:>10.2} MB\n", mb(*b)));
    }
    s.push_str("\nScenes and prefabs (client-role content removed)\n");
    for (path, objects, comps) in &cook.stripped {
        s.push_str(&format!("  {path}: {objects} object branch(es), {comps} component(s)\n"));
    }
    s.push_str(&format!("\nReference warnings: {}\n", cook.lints.len()));
    for l in &cook.lints {
        s.push_str(&format!("  - {l}\n"));
    }
    if !cook.warnings.is_empty() {
        s.push_str("\nOther warnings\n");
        for w in &cook.warnings {
            s.push_str(&format!("  - {w}\n"));
        }
    }
    s
}

/// `server.ron` for the build folder: relative to the folder the server runs in.
pub fn server_ron(project: &ProjectFile) -> String {
    server_ron_for(&project.name, project.boot_scene.as_deref().or(project.last_scene.as_deref()))
}

/// [`server_ron`] from a name and the scene to host.
pub fn server_ron_for(name: &str, scene: Option<&str>) -> String {
    let name = name.replace(['"', '\\'], "");
    let scene = scene.map(|s| format!("    scene: \"{s}\",\n")).unwrap_or_default();
    format!(
        "// Generated by Build Game (Dedicated server). Run from this folder (run.sh does).\n\
         // Every field is optional; see docs/dedicated-server.md.\n\
         (\n    project: \"{ASSETS_DIR}\",\n{scene}    tick_rate: 30.0,\n    max_players: 16,\n    server_name: \"{name}\",\n    log_path: \"logs/server.log\",\n    log_level: \"info\",\n)\n"
    )
}

/// The launcher next to the binary: the server's relative paths resolve against its folder.
pub fn run_script() -> String {
    format!("#!/bin/sh\ncd \"$(dirname \"$0\")\" || exit 1\nexec ./{SERVER_BINARY} --config server.ron \"$@\"\n")
}

/// The package name of the project's component crate, from its manifest.
fn components_package(project_root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(project_root.join("plugins/components/Cargo.toml")).ok()?;
    let mut in_package = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package && let Some(rest) = line.strip_prefix("name") {
            return rest.trim_start().strip_prefix('=').map(|v| v.trim().trim_matches('"').to_owned());
        }
    }
    None
}

/// Generated project server crate: `(Cargo.toml, src/main.rs)`. The crate links
/// `zestengine-server` (library, no client features) and the project's component
/// crate without its `editor` feature, and starts the server with the project's
/// own `zestengine_register`.
pub fn server_crate_sources(zestengine: &Path, project_root: &Path, pkg: &str, components_pkg: &str) -> (String, String) {
    let manifest = format!(
        r#"# Generated by Build Game (Dedicated server); rewritten on every build.
[package]
name = "{pkg}-server"
version = "0.1.0"
edition = "2024"

# Its own workspace root: this crate lives under the project's target/ folder.
[workspace]

[[bin]]
name = "{SERVER_BINARY}"
path = "src/main.rs"

[dependencies]
zestengine-server = {{ path = "{zestengine}/crates/zestengine-server" }}
# default-features = false: no `editor` feature, so neither the editor UI crate nor egui.
{components_pkg} = {{ path = "{components}", default-features = false }}

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
panic = "abort"
"#,
        zestengine = zestengine.display(),
        components = project_root.join("plugins/components").display(),
    );
    let source = format!(
        r#"//! Generated by Build Game (Dedicated server); rewritten on every build.
//! The stock zestengine-server with this project's components registered.

fn main() -> std::process::ExitCode {{
    zestengine_server::run_main(|registry| {lib}::zestengine_register(registry))
}}
"#,
        lib = components_pkg.replace('-', "_"),
    );
    (manifest, source)
}

fn run_cargo(cmd: &mut std::process::Command, what: &str) -> Result<()> {
    let output = cmd.output().with_context(|| format!("running cargo ({what})"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.trim().lines().rev().take(60).collect();
        bail!("cargo build failed ({what}):\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
    }
    Ok(())
}

/// Build the server binary: the project's own crate when it has components, else
/// the engine's `zestengine-server`. `ZESTENGINE_SERVER_TARGET` selects a target
/// triple for a cross build (the default is the host's).
fn build_binary(project_root: &Path, pkg: &str, log: &mut impl FnMut(String)) -> Result<PathBuf> {
    let zestengine = crate::bundle::zestengine_root()?;
    let triple = std::env::var("ZESTENGINE_SERVER_TARGET").ok().filter(|t| !t.is_empty());
    let mut cmd = std::process::Command::new("cargo");
    let (profile_dir, bin_name);
    let target_dir = |base: PathBuf| match &triple {
        Some(t) => base.join(t).join("release"),
        None => base.join("release"),
    };
    let windows = triple.as_deref().map_or(cfg!(windows), |t| t.contains("windows"));
    bin_name = if windows { format!("{SERVER_BINARY}.exe") } else { SERVER_BINARY.to_owned() };
    if let Some(components) = components_package(project_root) {
        let crate_dir = project_root.join("target/zestengine-server");
        std::fs::create_dir_all(crate_dir.join("src")).context("creating the server crate")?;
        let (manifest, source) = server_crate_sources(&zestengine, project_root, pkg, &components);
        std::fs::write(crate_dir.join("Cargo.toml"), manifest).context("writing the server Cargo.toml")?;
        std::fs::write(crate_dir.join("src/main.rs"), source).context("writing the server main.rs")?;
        // Same dependency versions as the engine's own build.
        let lock = crate_dir.join("Cargo.lock");
        if !lock.is_file() && zestengine.join("Cargo.lock").is_file() {
            std::fs::copy(zestengine.join("Cargo.lock"), &lock).context("seeding the server Cargo.lock")?;
        }
        let target = project_root.join("target");
        profile_dir = target_dir(target.clone());
        cmd.current_dir(&crate_dir).env("CARGO_TARGET_DIR", &target).args(["build", "--release"]);
        log(format!("cargo build --release (project server crate with {components}{})…", triple.as_deref().map(|t| format!(", {t}")).unwrap_or_default()));
    } else {
        let target = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| zestengine.join("target"));
        profile_dir = target_dir(target);
        cmd.current_dir(&zestengine).args(["build", "--release", "-p", "zestengine-server"]);
        log(format!("cargo build --release (zestengine-server, no project components{})…", triple.as_deref().map(|t| format!(", {t}")).unwrap_or_default()));
    }
    if let Some(t) = &triple {
        cmd.args(["--target", t]);
    }
    run_cargo(&mut cmd, "server")?;
    let built = profile_dir.join(&bin_name);
    if !built.is_file() {
        bail!("build succeeded but {} is missing", built.display());
    }
    Ok(built)
}

/// Options of the Dedicated server build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServerBuildOptions {
    /// Treat every cook warning and every kept-content-references-stripped-content lint as an
    /// error (`--strict-roles`, or `ZESTENGINE_STRICT_ROLES=1`). The build stops before compiling.
    pub strict_roles: bool,
    /// Also pack the cooked assets into `assets.pak` and point the generated `server.ron` at it
    /// (`--pak`, or `ZESTENGINE_SERVER_PAK=1`). The server unpacks it into its state folder.
    pub pak: bool,
}

impl ServerBuildOptions {
    /// The defaults with the environment switches applied.
    pub fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"));
        Self { strict_roles: on("ZESTENGINE_STRICT_ROLES"), pak: on("ZESTENGINE_SERVER_PAK") }
    }
}

/// `--strict-roles`: the cook's warnings and lints as one error, `Ok` when there are none.
pub fn strict_roles_check(cook: &ServerCook) -> Result<()> {
    let problems: Vec<&String> = cook.lints.iter().chain(cook.warnings.iter()).collect();
    if problems.is_empty() {
        return Ok(());
    }
    let list = problems.iter().map(|p| format!("  - {p}")).collect::<Vec<_>>().join("\n");
    bail!("--strict-roles: {} problem(s) in the server cook:\n{list}", problems.len())
}

/// Build the Dedicated server target: cook, binary, `server.ron`, launcher and report.
pub fn build_server(project_root: &Path, project: &ProjectFile, pkg: &str, log: impl FnMut(String)) -> Result<PathBuf> {
    build_server_with(project_root, project, pkg, ServerBuildOptions::from_env(), log)
}

/// Build the server of the project at `project_root` with explicit options, logging to stdout
/// (the `zestengine-build-server` binary).
pub fn build_server_dir(project_root: &Path, project: &ProjectFile, opts: ServerBuildOptions) -> Result<PathBuf> {
    build_server_with(project_root, project, &crate::bundle::crate_name(&project.name), opts, |m| println!("build: {m}"))
}

/// [`build_server`] with explicit [`ServerBuildOptions`].
pub fn build_server_with(project_root: &Path, project: &ProjectFile, pkg: &str, opts: ServerBuildOptions, mut log: impl FnMut(String)) -> Result<PathBuf> {
    let build_dir = project_root.join("build-server");
    let _ = std::fs::remove_dir_all(&build_dir);
    std::fs::create_dir_all(&build_dir).context("creating build-server/")?;
    log("cooking server assets (client-role content stripped)…".into());
    let mut registry = ComponentRegistry::with_builtins();
    project.apply_role_overrides(&mut registry);
    let cook = cook_server_assets(project_root, &build_dir.join(ASSETS_DIR), &registry, &mut log)?;
    if opts.strict_roles {
        strict_roles_check(&cook)?;
    }
    let built = build_binary(project_root, pkg, &mut log)?;
    let out_exe = build_dir.join(built.file_name().expect("binary name"));
    std::fs::copy(&built, &out_exe).with_context(|| format!("copying {}", built.display()))?;
    let mut ron_text = server_ron(project);
    if opts.pak {
        let pak = build_dir.join(zestengine_assets::game_pak::GAME_PAK_NAME);
        let n = zestengine_assets::game_pak::pack_dir(&build_dir.join(ASSETS_DIR), &pak)?;
        log(format!("packed {n} file(s) into {} ({:.2} MB)", zestengine_assets::game_pak::GAME_PAK_NAME, mb(size_of(&pak))));
        ron_text = ron_text.replace(&format!("project: \"{ASSETS_DIR}\""), &format!("project: \"{}\"", zestengine_assets::game_pak::GAME_PAK_NAME));
    }
    std::fs::write(build_dir.join("server.ron"), ron_text).context("writing server.ron")?;
    let run = build_dir.join("run.sh");
    std::fs::write(&run, run_script()).context("writing run.sh")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let _ = std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o755));
    }
    let scene = project.boot_scene.as_deref().or(project.last_scene.as_deref());
    let params = crate::server_deploy::DeployParams::new(&project.name, scene, crate::server_deploy::deploy_game_port(project_root, scene));
    crate::server_deploy::write_deploy(&build_dir, &params)?;
    log(format!("deploy/ written (systemd, Docker, firewall; state_dir {})", params.state_dir()));
    let report = report_text(&project.name, &cook, &out_exe.display().to_string());
    std::fs::write(build_dir.join(REPORT_NAME), &report).context("writing build-report.txt")?;
    log(format!(
        "server {:.2} MB of assets vs {:.2} MB for a client; {} reference warning(s); see {REPORT_NAME}",
        mb(cook.bytes),
        mb(cook.client_bytes),
        cook.lints.len()
    ));
    log(format!("done: {}", out_exe.display()));
    Ok(out_exe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::ObjectId;

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine-server-build-{tag}-{}-{}", std::process::id(), ObjectId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tree(dir: &Path) -> Vec<String> {
        let mut out = Vec::new();
        crate::bundle::list_files(dir, dir, &mut out);
        out.sort();
        out
    }

    /// A scaffolded template with a client-only light group, a server-only object, and a gameplay
    /// component naming a sound and a texture (so the cook has something to lint and drop).
    fn tagged_project(parent: &Path, template: &str) -> (PathBuf, String) {
        let root = crate::bundle::scaffold_project(parent, template, template).unwrap();
        let project = zestengine_assets::load_project_file(&root).unwrap();
        let boot = project.boot_scene.clone().unwrap();
        let mut file = zestengine_assets::load_scene_file(root.join(&boot)).unwrap();
        let reg = ComponentRegistry::with_builtins();
        let mut stage = file.entries[0].clone();
        stage.id = ObjectId::new().to_string();
        stage.parent = None;
        stage.components.clear();
        stage.net_role = NetRole::Client;
        stage.name = "Stage Fx".into();
        let fx_id = stage.id.clone();
        file.entries.push(stage);
        let mut keeper = file.entries[0].clone();
        keeper.id = ObjectId::new().to_string();
        keeper.name = "Score Keeper".into();
        keeper.parent = None;
        keeper.net_role = NetRole::Server;
        // A kept component pointing at the client-only object, a sound and a texture.
        keeper.components = vec![zestengine_assets::ComponentData::new("Health", format!("(fx: \"{fx_id}\", sound: \"audio/boom.wav\", icon: \"textures/icon.png\")"))];
        file.entries.push(keeper);
        zestengine_assets::save_scene_file(root.join(&boot), &file).unwrap();
        for (rel, bytes) in [("audio/boom.wav", 4000usize), ("textures/icon.png", 3000), ("shaders/glow.frag", 500)] {
            std::fs::write(root.join(rel), vec![7u8; bytes]).unwrap();
        }
        // Reachable only from the stripped scene text? no: referenced by the kept component above.
        let _ = reg;
        (root, boot)
    }

    #[test]
    fn server_cook_strips_a_scaffolded_project_and_the_result_runs() {
        let parent = temp("cook");
        let (root, boot) = tagged_project(&parent, "vr_social");
        let out = parent.join("out");
        let mut log = Vec::new();
        let cook = cook_server_assets(&root, &out, &ComponentRegistry::with_builtins(), &mut |m| log.push(m)).unwrap();
        let files = tree(&out);

        // No client-only file types survive, and no cache folders.
        for f in &files {
            assert!(!is_client_only_file(f), "client-only file shipped: {f}");
            assert!(!matches!(ext_of(f).as_str(), "png" | "jpg" | "wav" | "ogg" | "frag" | "spv" | "ttf" | "material"), "{f}");
        }
        assert!(files.contains(&boot) && files.contains(&"project.zestengine".to_owned()), "{files:?}");
        // Models are collision-only blobs at their own paths.
        let models: Vec<&String> = files.iter().filter(|f| f.ends_with(".glb")).collect();
        assert!(!models.is_empty(), "{files:?}");
        for m in &models {
            assert!(server_model::is_blob(&out.join(m)), "{m} is a collision blob");
        }
        assert!(cook.models.0 == models.len() && cook.models.2 < cook.models.1, "blobs are smaller: {:?}", cook.models);
        // The server folder is smaller than the client's cook.
        assert!(cook.bytes < cook.client_bytes, "server {} vs client {}", cook.bytes, cook.client_bytes);
        eprintln!("vr_social: client {} files / {} B, server {} files / {} B, models {:?}", cook.client_files, cook.client_bytes, cook.files, cook.bytes, cook.models);
        // Dropped classes include textures.
        assert!(cook.dropped.iter().any(|(c, n, _)| *c == "textures" && *n > 0), "{:?}", cook.dropped);

        // Scenes: client-role content removed, collision and the server-only object kept.
        let stripped: zestengine_assets::SceneFile = zestengine_assets::load_scene_file(out.join(&boot)).unwrap();
        let names: Vec<&str> = stripped.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(!names.contains(&"Stage Fx"), "{names:?}");
        assert!(names.contains(&"Score Keeper"));
        assert!(stripped.skybox.is_none());
        let kinds: Vec<&str> = stripped.entries.iter().flat_map(|e| e.components.iter().map(|c| c.kind.as_str())).collect();
        for gone in ["Light", "UI Text", "UI Panel", "UI Button", "UI Canvas", "Planar Reflection", "Audio Source", "Camera"] {
            assert!(!kinds.contains(&gone), "{gone} survived: {kinds:?}");
        }
        assert!(kinds.contains(&"Box Collider"), "colliders stay");
        assert!(stripped.entries.iter().all(|e| e.prefab.is_none()), "prefab links are expanded");

        // Lints name the holder and the stripped target.
        let lints = cook.lints.join("\n");
        assert!(lints.contains("Score Keeper") && lints.contains("Stage Fx"), "{lints}");
        assert!(lints.contains("audio/boom.wav") && lints.contains("textures/icon.png"), "{lints}");

        // The stripped project loads on a server and ticks, and still has collision.
        let mut host = crate::server::HeadlessHost::open(
            crate::server::HeadlessConfig { assets_root: out.clone(), scene: Some(boot.clone()), physics: None, net_side: NetRole::Server },
            |_| {},
        )
        .unwrap();
        assert!(host.scene().objects.iter().any(|o| o.name == "Score Keeper"));
        for _ in 0..60 {
            host.tick(1.0 / 60.0);
        }
        assert!(host.scene().objects.iter().all(|o| o.translation.is_finite()));
        let bodies = host.scene().objects.iter().filter(|o| o.components.iter().any(|c| matches!(c.type_name(), "Box Collider" | "Mesh Collider" | "Rigid Body"))).count();
        assert!(bodies > 3, "{bodies} physics objects");

        // The report carries sizes, classes and warnings.
        let report = report_text("vr_social", &cook, "zestengine-server");
        for needle in ["client build", "server build", "textures", "Reference warnings: ", "audio/boom.wav", "Stage Fx"] {
            assert!(report.contains(needle), "report lacks {needle:?}:\n{report}");
        }
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn topdown_cook_is_smaller_and_loads() {
        let parent = temp("topdown");
        let root = crate::bundle::scaffold_project(&parent, "topdown", "topdown").unwrap();
        let project = zestengine_assets::load_project_file(&root).unwrap();
        let out = parent.join("out");
        let cook = cook_server_assets(&root, &out, &ComponentRegistry::with_builtins(), &mut |_| {}).unwrap();
        eprintln!("topdown: client {} files / {} B, server {} files / {} B, models {:?}", cook.client_files, cook.client_bytes, cook.files, cook.bytes, cook.models);
        assert!(cook.bytes < cook.client_bytes);
        let mut host = crate::server::HeadlessHost::open(
            crate::server::HeadlessConfig { assets_root: out, scene: project.boot_scene, physics: None, net_side: NetRole::Server },
            |_| {},
        )
        .unwrap();
        for _ in 0..30 {
            host.tick(1.0 / 60.0);
        }
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn strict_roles_turns_lints_and_warnings_into_an_error() {
        let mut cook = ServerCook::default();
        assert!(strict_roles_check(&cook).is_ok());
        cook.warnings.push("scenes/a.scene: kept unstripped (boom)".into());
        cook.lints.push("Hud/Score: references sounds/x.ogg".into());
        let e = strict_roles_check(&cook).unwrap_err().to_string();
        assert!(e.contains("2 problem(s)") && e.contains("kept unstripped") && e.contains("sounds/x.ogg"), "{e}");
        assert!(!ServerBuildOptions::default().strict_roles && !ServerBuildOptions::default().pak);
    }

    #[test]
    fn server_ron_and_launcher_are_ready_to_run() {
        let project = ProjectFile { name: "Arena \"X\"".into(), boot_scene: Some("scenes/main.scene".into()), ..Default::default() };
        let ron = server_ron(&project);
        assert!(ron.contains("project: \"assets\"") && ron.contains("scene: \"scenes/main.scene\""), "{ron}");
        assert!(ron.contains("server_name: \"Arena X\""), "quotes are dropped: {ron}");
        let cfg: ron::Value = ron::from_str(&ron).expect("valid RON");
        let _ = cfg;
        let run = run_script();
        assert!(run.starts_with("#!/bin/sh") && run.contains("--config server.ron") && run.contains("./zestengine-server"));
    }

    #[test]
    fn project_server_crate_links_the_components_without_editor_features() {
        let (manifest, main) = server_crate_sources(Path::new("/opt/zest"), Path::new("/p/game"), "game", "zestengine-project-components");
        assert!(manifest.contains("name = \"game-server\"") && manifest.contains("name = \"zestengine-server\""));
        assert!(manifest.contains("zestengine-server = { path = \"/opt/zest/crates/zestengine-server\" }"), "{manifest}");
        assert!(manifest.contains("zestengine-project-components = { path = \"/p/game/plugins/components\", default-features = false }"), "{manifest}");
        assert!(manifest.contains("\n[workspace]\n"), "its own workspace root");
        assert!(!manifest.contains("zestengine-editor") && !manifest.contains("\"client\""), "{manifest}");
        assert!(main.contains("zestengine_server::run_main(|registry| zestengine_project_components::zestengine_register(registry))"), "{main}");
        // The scaffolded component crate has the `editor` feature off by default and the register entry.
        let parent = temp("scaffold");
        let root = crate::bundle::scaffold_project(&parent, "g", "empty").unwrap();
        assert_eq!(components_package(&root).as_deref(), Some("zestengine-project-components"));
        let comps = std::fs::read_to_string(root.join("plugins/components/Cargo.toml")).unwrap();
        assert!(comps.contains("default = []") && comps.contains("editor = [\"dep:zestengine-editor\""), "{comps}");
        let lib = std::fs::read_to_string(root.join("plugins/components/src/lib.rs")).unwrap();
        assert!(lib.contains("pub fn zestengine_register(registry: &mut ComponentRegistry)"));
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn target_and_recipe_are_wired() {
        use crate::bundle::BuildTarget;
        assert_eq!(BuildTarget::parse("server"), Some(BuildTarget::Server));
        assert_eq!(BuildTarget::parse("dedicated"), Some(BuildTarget::Server));
        assert_eq!(BuildTarget::Server.output_hint(), "build-server");
        assert!(BuildTarget::available().any(|t| t == BuildTarget::Server));
        assert_eq!(BuildTarget::Server.label(), "Dedicated server");
        let just = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../justfile")).unwrap();
        assert!(just.contains("build-server project out=\"\":") && just.contains("zestengine-build-server"), "justfile recipe");
        // The cook never walks its own output.
        assert!(crate::bundle::asset_dirs(&std::env::temp_dir().join("zestengine-no-such")).is_err());
    }
}
