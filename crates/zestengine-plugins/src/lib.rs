#![allow(dead_code)]
//! Rust component plugins: crates under `plugins/`, built by the editor
//! (`cargo build -p <name>`) and hot-loaded as dylibs.
//!
//! Plugins are workspace members, so their dependency versions are pinned by
//! the same Cargo.lock as the editor. That (plus the same rustc) is what
//! makes sharing `dyn Component` across the dylib boundary safe in practice.
//! Each plugin exports `zestengine_register(&mut ComponentRegistry)`.
//!
//! Loaded libraries are NEVER unloaded: scene objects can hold component
//! instances whose code lives in an older build. Reloading registers the new
//! build's types over the old names and the engine re-instantiates every
//! object's components from serialized state.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;

use anyhow::{Context as _, Result, bail};
use zestengine_core::{ComponentRegistry, PluginAbi};

/// One cargo clippy / rustc message, simplified for display.
#[derive(Clone, Debug)]
pub struct CodeDiagnostic {
    /// "error" or "warning".
    pub level: String,
    /// Project-relative file.
    pub file: String,
    pub line: u32,
    pub message: String,
}

pub fn check_shader(path: &Path) -> Vec<CodeDiagnostic> {
    let output = Command::new("glslc").arg("-o").arg("-").arg(path).output();

    let mut diags = Vec::new();
    match output {
        Ok(o) if !o.status.success() => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            for (i, line) in stderr.lines().enumerate() {
                if line.contains("error:") || line.contains("ERROR:") {
                    diags.push(CodeDiagnostic {
                        level: "error".into(),
                        file: path.to_string_lossy().into_owned(),
                        line: (i + 1) as u32,
                        message: line.trim().to_owned(),
                    });
                } else if line.contains("warning:") || line.contains("WARNING:") {
                    diags.push(CodeDiagnostic {
                        level: "warning".into(),
                        file: path.to_string_lossy().into_owned(),
                        line: (i + 1) as u32,
                        message: line.trim().to_owned(),
                    });
                }
            }
            if diags.is_empty() {
                diags.push(CodeDiagnostic {
                    level: "error".into(),
                    file: path.to_string_lossy().into_owned(),
                    line: 1,
                    message: "glslc failed (see full output in terminal)".into(),
                });
            }
        }
        Err(e) => {
            diags.push(CodeDiagnostic {
                level: "error".into(),
                file: path.to_string_lossy().into_owned(),
                line: 1,
                message: format!("failed to run glslc: {e}"),
            });
        }
        _ => {}
    }
    diags
}

#[derive(Default)]
pub struct PluginHost {
    /// Intentionally leaked for the app's lifetime (see module docs).
    libs: Vec<libloading::Library>,
    /// Suffix for copied dylibs so every reload loads a fresh path.
    generation: u32,
    /// `(plugin name, zestengine_register)` of every loaded plugin, a reload replacing its name's
    /// entry. The function pointers stay valid because libraries are never unloaded, so another
    /// thread (the "Play as Server" thread) can register the same components.
    registrations: Vec<PluginRegistration>,
    /// Raw `zestengine_register_editor` symbols (one per loaded plugin), cast
    /// back to `fn(&mut EditorComponents)` by the editor crate.
    editor_registrars: Vec<*const ()>,
}

/// One loaded plugin: its name plus the registration entry point.
type PluginRegistration = (String, fn(&mut ComponentRegistry));

/// Compare the plugin's exported ABI stamp (`zestengine_core::export_plugin_abi!()`) with this
/// build's. A plugin without the export is `Ok(false)` when `require` is off (every plugin built
/// before the stamp existed) and an error when it is on (the dedicated server).
fn check_abi(lib: &libloading::Library, what: &str, require: bool) -> Result<bool> {
    // SAFETY: `extern "C"` returning plain `#[repr(C)]` data: safe to call across compilers.
    let stamp = unsafe { lib.get::<extern "C" fn() -> PluginAbi>(b"zestengine_plugin_abi").ok().map(|f| f()) };
    match stamp {
        Some(s) => {
            s.check_against(&PluginAbi::current()).map_err(|e| anyhow::anyhow!("{what}: {e}"))?;
            Ok(true)
        }
        None if require => bail!("{what} has no ABI stamp: add `zestengine_core::export_plugin_abi!();` to the plugin crate and rebuild it"),
        None => Ok(false),
    }
}

/// Plugins loaded by a process that has no editor: the dedicated server (NET-3). The paths are
/// built `cdylib`s named in `server.ron`; each is ABI-checked (a stamp is required), must export
/// `zestengine_register`, and is kept mapped for the life of the process.
#[derive(Default)]
pub struct RuntimePlugins {
    registrations: Vec<PluginRegistration>,
    /// Raw `zestengine_register_editor` symbols (one per loaded plugin), cast
    /// back to `fn(&mut EditorComponents)` by the editor crate.
    editor_registrars: Vec<*const ()>,
}

impl RuntimePlugins {
    /// The loaded plugins' `zestengine_register_editor` symbols, type-erased.
    pub fn editor_registrars(&self) -> &[*const ()] {
        &self.editor_registrars
    }
}

impl PluginHost {
    /// The loaded plugins' `zestengine_register_editor` symbols, type-erased.
    pub fn editor_registrars(&self) -> &[*const ()] {
        &self.editor_registrars
    }
}

impl RuntimePlugins {
    /// Load every library in `paths`; the first failure is returned (the server refuses to start).
    pub fn load_all(paths: &[PathBuf]) -> Result<Self> {
        let mut out = Self::default();
        for path in paths {
            out.load_one(path)?;
        }
        Ok(out)
    }

    fn load_one(&mut self, path: &Path) -> Result<()> {
        let what = format!("plugin {}", path.display());
        if !path.is_file() {
            bail!("{what} does not exist (build it with the project's plugin build and list the cdylib in `plugins`)");
        }
        // SAFETY: loading a library runs its initialisers; it is the operator's own plugin. RTLD_NOW as in the editor.
        let lib: libloading::Library = unsafe {
            #[cfg(unix)]
            {
                use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
                Library::open(Some(path), RTLD_NOW | RTLD_LOCAL).with_context(|| format!("loading {what}"))?.into()
            }
            #[cfg(not(unix))]
            {
                libloading::Library::new(path).with_context(|| format!("loading {what}"))?
            }
        };
        check_abi(&lib, &what, true)?;
        // SAFETY: a Rust-ABI function pointer, valid for as long as the library stays mapped (forever).
        let register = unsafe { *lib.get::<fn(&mut ComponentRegistry)>(b"zestengine_register").with_context(|| format!("{what} has no `zestengine_register` export"))? };
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        self.registrations.push((name, register));
        // Never unloaded: scene objects hold component instances whose vtables live in the library.
        std::mem::forget(lib);
        tracing::info!("loaded component plugin {}", path.display());
        Ok(())
    }

    /// Plugin names in load order.
    pub fn names(&self) -> Vec<&str> {
        self.registrations.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Register every plugin's components.
    pub fn register_all(&self, registry: &mut ComponentRegistry) {
        for (_, f) in &self.registrations {
            f(registry);
        }
    }

    /// The registration functions, to move onto another thread (valid while `self` lives).
    pub fn registrations(&self) -> Vec<fn(&mut ComponentRegistry)> {
        self.registrations.iter().map(|(_, f)| *f).collect()
    }
}

/// (directory, package name) for each crate under `plugins/`.
pub fn discover(project_root: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(project_root.join("plugins")) else {
        return out;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let manifest = dir.join("Cargo.toml");
        if let Ok(text) = std::fs::read_to_string(&manifest)
            && let Some(name) = package_name(&text)
        {
            out.push((dir, name));
        }
    }
    out.sort();
    out
}

/// Minimal `name = "…"` extraction (avoids a toml dependency).
fn package_name(manifest: &str) -> Option<String> {
    manifest.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("name")?.trim_start();
        let rest = rest.strip_prefix('=')?.trim();
        Some(rest.trim_matches('"').to_owned())
    })
}

impl PluginHost {
    /// Registration functions of the loaded plugins, for a thread that builds its own registry
    /// (Play as Server). Libraries are never unloaded, so the pointers stay valid.
    pub fn registrations(&self) -> Vec<fn(&mut ComponentRegistry)> {
        self.registrations.iter().map(|(_, f)| *f).collect()
    }

    pub fn any_plugins(project_root: &Path) -> bool {
        !discover(project_root).is_empty()
    }

    /// Build every plugin crate and load + register it. Returns the loaded
    /// plugin names; the first failure aborts with the compiler output.
        pub fn build_and_load(&mut self, project_root: &Path, registry: &mut ComponentRegistry) -> Result<Vec<String>> {
    let names = Self::build_all(project_root)?;
    self.load_built(project_root, &names, registry)?;
    Ok(names)
}

    /// Run `cargo build` for every plugin crate. Touches no editor state, so it
    /// can run on a worker; [`Self::load_built`] registers the results.
        pub fn build_all(project_root: &Path) -> Result<Vec<String>> {
        Self::build_all_with(project_root, &|_| {})
    }

    /// [`Self::build_all`] that reports a one-line status as it goes (the
    /// splash shows it): elapsed seconds while cargo runs, "up to date" when
    /// the fingerprint skip applies, the first error line when a build fails.
        pub fn build_all_with(project_root: &Path, report: &dyn Fn(&str)) -> Result<Vec<String>> {
        Self::build_all_with_diags(project_root, report).0
    }

    /// [`Self::build_all_with`] plus the compiler diagnostics of every plugin
    /// crate, parsed from cargo's JSON output (for the editor's inspectors and
    /// file badges).
    pub fn build_all_with_diags(project_root: &Path, report: &dyn Fn(&str)) -> (Result<Vec<String>>, Vec<CodeDiagnostic>) {
        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let result = Self::build_all_inner(project_root, report, &sink);
        let diags = std::sync::Arc::try_unwrap(sink).map(|m| m.into_inner().unwrap()).unwrap_or_default();
        (result, diags)
    }

    fn build_all_inner(
        project_root: &Path,
        report: &dyn Fn(&str),
        diag_sink: &std::sync::Arc<std::sync::Mutex<Vec<CodeDiagnostic>>>,
    ) -> Result<Vec<String>> {
        // Build into the ENGINE's target dir (resolved from the workspace's
        // `zestengine-engine` path dep), so the editor's already-compiled
        // dependency tree (egui, the editor crate, ...) is reused and a plugin
        // build compiles only the plugin itself - not a fresh copy of the
        // whole engine into the project's own target dir.
        let target_dir = shared_target_dir(project_root);
        let mut built = Vec::new();
        for (dir, name) in discover(project_root) {
            // Nothing changed since the last successful build (sources of the
            // plugin and of the workspace crates it links, lockfile, this editor
            // binary): skip cargo, whose no-op run still resolves and
            // fingerprints the whole workspace (seconds on a big one).
            let t_total = std::time::Instant::now();
            let print = build_fingerprint(project_root, &dir);
            let dylib = target_dir.join(PROFILE_DIR).join(dylib_name(&name));
            let stamp = project_root.join("target/zestengine-plugins").join(format!("{name}.stamp"));
            if std::fs::read_to_string(&stamp).ok().and_then(|t| t.trim().parse::<u64>().ok()) == Some(print) && dylib.is_file() {
                tracing::info!("component plugin {name} unchanged, skipping cargo build ({:?} to check)", t_total.elapsed());
                report(&status_up_to_date(&name));
                built.push(name);
                continue;
            }
            let fingerprint_time = t_total.elapsed();
            tracing::info!("building component plugin {name}… (fingerprint {fingerprint_time:?})");
            // `--features editor`: the editor needs each plugin's inspector +
            // gizmo (and its `zestengine_register_editor` export). A shipped game
            // builds plugins without it, so zestengine-editor/egui never link in.
            report(&status_building(&name, 0));
            let mut child = Command::new("cargo")
                .current_dir(project_root)
                .env("CARGO_TARGET_DIR", &target_dir)
                .args(plugin_build_args(project_root, &name))
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .context("running cargo")?;
            // Stream every cargo line into the project log so the build can be
            // watched live; parse cargo's JSON compiler messages into
            // diagnostics for the editor's inspectors and file badges. Drain
            // on threads so a chatty build never blocks on a full pipe.
            let mut pipe = child.stderr.take();
            let name_err = name.clone();
            let reader = std::thread::spawn(move || {
                let mut text = Vec::new();
                if let Some(p) = pipe.as_mut() {
                    use std::io::{BufRead, BufReader};
                    for line in BufReader::new(p).lines().map_while(Result::ok) {
                        tracing::info!("{name_err}: {line}");
                        text.extend_from_slice(line.as_bytes());
                        text.push(b'\n');
                    }
                }
                String::from_utf8_lossy(&text).into_owned()
            });
            let mut out_pipe = child.stdout.take();
            let name_out = name.clone();
            let sink = diag_sink.clone();
            let proj = project_root.to_owned();
            std::thread::spawn(move || {
                if let Some(p) = out_pipe.as_mut() {
                    use std::io::{BufRead, BufReader};
                    for line in BufReader::new(p).lines().map_while(Result::ok) {
                        // JSON compiler message: show the rendered text in the
                        // log and collect the structured diagnostic.
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line)
                            && let Some(rendered) = v.get("rendered").and_then(|r| r.as_str())
                        {
                            tracing::info!("{name_out}: {rendered}");
                            if v.get("reason").and_then(|r| r.as_str()) == Some("compiler-message")
                                && let Some(msg) = v.get("message")
                            {
                                let level = msg
                                    .get("level")
                                    .and_then(|l| l.as_str())
                                    .unwrap_or("error");
                                let message = msg
                                    .get("message")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("")
                                    .to_owned();
                                let span = msg
                                    .get("spans")
                                    .and_then(|s| s.as_array())
                                    .and_then(|s| s.first())
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                let file = span
                                    .get("file_name")
                                    .and_then(|f| f.as_str())
                                    .map(std::path::PathBuf::from)
                                    .map(|f| {
                                        f.strip_prefix(&proj)
                                            .map(|r| r.to_string_lossy().into_owned())
                                            .unwrap_or_else(|_| f.to_string_lossy().into_owned())
                                    })
                                    .unwrap_or_default();
                                let line_no = span
                                    .get("line_start")
                                    .and_then(|l| l.as_u64())
                                    .map(|l| l as u32 + 1)
                                    .unwrap_or(0);
                                sink.lock().unwrap().push(CodeDiagnostic {
                                    file,
                                    line: line_no,
                                    message,
                                    level: level.to_owned(),
                                });
                            }
                        } else {
                            tracing::info!("{name_out}: {line}");
                        }
                    }
                }
            });
            let started = std::time::Instant::now();
            let mut shown = 0;
            let status = loop {
                if let Some(st) = child.try_wait().context("waiting for cargo")? {
                    break st;
                }
                let secs = started.elapsed().as_secs();
                if secs != shown {
                    shown = secs;
                    report(&status_building(&name, secs));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            };
            let stderr = reader.join().unwrap_or_default();
            if !status.success() {
                report(&status_failed(&name, &stderr));
                bail!("building {name} failed:\n{}", stderr.trim());
            }
            // Surface rustc/clippy warnings from a successful build in the log.
            if stderr.contains("warning:") {
                tracing::warn!("plugin {name} built with warnings:\n{}", stderr.trim());
            }
            tracing::info!("component plugin {name} cargo build took {:?}", t_total.elapsed() - fingerprint_time);
            // The fingerprint was taken before the build, so an edit made while
            // it ran leaves a stale stamp and rebuilds next time.
            if let Some(dir) = stamp.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&stamp, print.to_string());
            built.push(name);
        }
        Ok(built)
    }

    /// Load + register plugins already built by [`Self::build_all`].
        pub fn load_built(&mut self, project_root: &Path, names: &[String], registry: &mut ComponentRegistry) -> Result<()> {
    for name in names {
        self.load(project_root, name, registry)
                .with_context(|| format!("loading plugin {name}"))?;
        }
        Ok(())
    }

        fn load(&mut self, project_root: &Path, name: &str, registry: &mut ComponentRegistry) -> Result<()> {
        let file = dylib_name(name);
        // The dylib lives in the SHARED engine target dir (see
        // `build_all_with`); the per-process copies stay project-local so two
        // editor instances never overwrite each other's mmapped file.
        let built = shared_target_dir(project_root).join(PROFILE_DIR).join(&file);
        // Copy to a unique path: the next `cargo build` overwrites the
        // original, and dlopen caches by path. The filename includes this
        // process's PID so two editor instances open on the same project don't
        // overwrite each other's mmapped copy (which SIGSEGVs the other editor).
        let dir = project_root.join("target/zestengine-plugins");
        std::fs::create_dir_all(&dir)?;
        let copy = dir.join(format!(
            "{name}-{}-{}.{DYLIB_EXT}",
            std::process::id(),
            self.generation
        ));
        self.generation += 1;
        std::fs::copy(&built, &copy)
            .with_context(|| format!("copying {} for loading", built.display()))?;
        unsafe {
            // RTLD_NOW: resolve every symbol at load time. With the default
            // lazy binding a missing/incompatible symbol becomes a deferred
            // null-pointer call (a segfault from inside ld.so on first use,
            // hard to trace). RTLD_NOW turns that into a clean load error.
            // RTLD_LOCAL keeps the plugin's symbols out of the global scope.
            #[cfg(unix)]
            let lib: libloading::Library = {
                use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
                Library::open(Some(&copy), RTLD_NOW | RTLD_LOCAL)
                    .with_context(|| format!("loading {}", copy.display()))?
                    .into()
            };
            #[cfg(not(unix))]
            let lib = libloading::Library::new(&copy)
                .with_context(|| format!("loading {}", copy.display()))?;
            check_abi(&lib, &format!("plugin {name}"), false)?;
            let register: libloading::Symbol<fn(&mut ComponentRegistry)> = lib
                .get(b"zestengine_register")
                .context("plugin has no `zestengine_register` export")?;
            register(registry);
            let f = *register;
            match self.registrations.iter_mut().find(|(n, _)| n == name) {
                Some(slot) => slot.1 = f,
                None => self.registrations.push((name.to_string(), f)),
            }
            // Editor-only inspector/gizmo registration (optional; a runtime
            // game build of a plugin won't export it).
            // The editor registers the plugin's inspector/gizmo later: the raw
            // symbol is stored type-erased (this crate must not depend on the
            // editor crate) and cast back by the editor.
            if let Ok(register_editor) = lib.get::<unsafe extern "C" fn()>(b"zestengine_register_editor") {
                self.editor_registrars.push(*register_editor as *const ());
            }
            register_import_hooks(name, &lib);
            self.libs.push(lib); // keep mapped forever
        }
        tracing::info!("loaded component plugin {name}");
        Ok(())
    }
}

/// Model import hooks (IMP-6) from a plugin's optional exports
/// `zestengine_import_hook_api_version` and `zestengine_register_import_hooks`.
/// A plugin without them (every plugin built before IMP-6) loads unchanged. This
/// is the only place hooks get enabled: the editor's plugin loader, so a
/// standalone player (which never loads dylibs) never runs a hook. A reload
/// replaces the plugin's previous hooks, even when the new build has none.
fn register_import_hooks(name: &str, lib: &libloading::Library) {
    use zestengine_assets::import_hooks::{self, ImportHookRegistry};
    // SAFETY: Rust-ABI function pointers, like `zestengine_register`: same rustc and lockfile.
    let (version, register) = unsafe {
        (
            lib.get::<fn() -> u32>(b"zestengine_import_hook_api_version").ok().map(|f| f()),
            lib.get::<fn(&mut ImportHookRegistry)>(b"zestengine_register_import_hooks").ok(),
        )
    };
    let register = register.as_deref().map(|f| move |r: &mut ImportHookRegistry| f(r));
    let n = import_hooks::register_plugin(name, version, register.as_ref().map(|f| f as &dyn Fn(&mut ImportHookRegistry)));
    // The editor has loaded trusted project plugins: hooks may run (never for UGC paths).
    import_hooks::set_editor_hooks_enabled(true);
    if n > 0 {
        tracing::info!("plugin {name} registered {n} import hook(s)");
    }
}

/// Hash of everything a plugin build depends on: file sizes and mtimes under
/// the plugin crate and `crates/` (the workspace crates it links), the root
/// manifest and lockfile, this editor binary and the profile. Contents are not
/// read, so it costs a directory walk, not a build. `target`, hidden and
/// `node_modules` directories are skipped.
pub fn build_fingerprint(project_root: &Path, plugin_dir: &Path) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    fn stamp(path: &Path, mix: &mut dyn FnMut(&[u8])) {
        if let Ok(m) = std::fs::metadata(path) {
            mix(path.as_os_str().as_encoded_bytes());
            mix(&m.len().to_le_bytes());
            let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
            mix(&t.to_le_bytes());
        }
    }
    fn walk(dir: &Path, mix: &mut dyn FnMut(&[u8])) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().into_owned();
            let path = e.path();
            if path.is_dir() {
                if !name.starts_with('.') && name != "target" && name != "node_modules" {
                    walk(&path, mix);
                }
            } else {
                stamp(&path, mix);
            }
        }
    }
    mix(PROFILE_DIR.as_bytes());
    mix(plugin_build_args(project_root, "").join(" ").as_bytes());
    if let Ok(exe) = std::env::current_exe() {
        stamp(&exe, &mut mix);
    }
    for f in ["Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml"] {
        stamp(&project_root.join(f), &mut mix);
    }
    walk(plugin_dir, &mut mix);
    walk(&project_root.join("crates"), &mut mix);
    h
}

/// The plugin must be built exactly like the editor that loads it: it shares
/// egui types with the host, and egui's structs change layout with
/// `debug_assertions` and with features (`accesskit`). A dev-built plugin in a
/// release editor corrupted egui's per-frame state (free(): invalid pointer
/// when its inspector drew).
const PROFILE_DIR: &str = if cfg!(debug_assertions) { "debug" } else { "release" };

/// Splash line while cargo builds plugin `name`.
fn status_building(name: &str, secs: u64) -> String {
    if secs == 0 { format!("Building component plugins… ({name})") } else { format!("Building component plugins… {secs} s ({name})") }
}

fn status_up_to_date(name: &str) -> String {
    format!("Component plugins up to date ({name})")
}

/// First `error…` line of cargo/rustc output (skipping the closing
/// "could not compile" summary), trimmed to a splash-sized line.
pub fn first_error_line(stderr: &str) -> Option<String> {
    let mut summary = None;
    for line in stderr.lines().map(str::trim) {
        if line.starts_with("error") {
            if line.contains("could not compile") {
                summary.get_or_insert(line);
                continue;
            }
            return Some(clip(line));
        }
    }
    summary.map(clip)
}

fn clip(line: &str) -> String {
    const MAX: usize = 140;
    if line.chars().count() <= MAX { line.to_owned() } else { format!("{}…", line.chars().take(MAX).collect::<String>()) }
}

fn status_failed(name: &str, stderr: &str) -> String {
    match first_error_line(stderr) {
        Some(l) => format!("Component plugin {name} failed: {l}"),
        None => format!("Component plugin {name} failed to build"),
    }
}

/// The engine checkout's target dir, resolved from the workspace's
/// `zestengine-engine` path dep via `cargo metadata`. Plugin builds target
/// this (and the loader reads dylibs from it) so the editor's own compiled
/// dependency tree is reused instead of rebuilt into the project's target.
/// Falls back to the project's target dir when the metadata is unavailable.
fn shared_target_dir(project_root: &Path) -> PathBuf {
    let fallback = || project_root.join("target");
    let Ok(out) = Command::new("cargo")
        .current_dir(project_root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
    else {
        return fallback();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    // No serde_json dep here: find the engine package's manifest path with a
    // bounded scan of the metadata JSON.
    let Some(at) = text.find("\"name\":\"zestengine-engine\"") else {
        return fallback();
    };
    let Some(mp) = text[at..].find("\"manifest_path\":\"") else {
        return fallback();
    };
    let start = at + mp + "\"manifest_path\":\"".len();
    let Some(path) = text[start..].split('"').next() else {
        return fallback();
    };
    Path::new(path)
        .parent()
        .and_then(Path::parent)
        .map(|checkout| checkout.join("target"))
        .unwrap_or_else(fallback)
}

/// `cargo build` arguments for a plugin: the editor's profile, and when the
/// editor binary was built from this same workspace, that package too, so
/// cargo unifies features into the identical egui build the editor uses.
fn plugin_build_args(project_root: &Path, name: &str) -> Vec<String> {
    let mut args = vec!["build".to_owned()];
    if !cfg!(debug_assertions) {
        args.push("--release".to_owned());
        // The project's own `[profile.release]` is a lean SHIPPING profile
        // (lto, codegen-units=1, strip, panic=abort) for its game binary. A
        // plugin must match the EDITOR's build instead: different profile
        // settings are different artifacts in cargo, so inheriting the lean
        // profile forces a full recompile of the shared dependency tree (and
        // LTO makes that even slower) - and a panic=abort dylib inside the
        // unwinding editor aborts the whole process. Override back to cargo's
        // default release profile, which is what the editor binary was built
        // with.
        args.extend([
            "--config".to_owned(),
            "profile.release.lto=false".to_owned(),
            "--config".to_owned(),
            "profile.release.codegen-units=16".to_owned(),
            "--config".to_owned(),
            "profile.release.strip=false".to_owned(),
            "--config".to_owned(),
            "profile.release.panic=\"unwind\"".to_owned(),
        ]);
    }
    args.extend(["-p".to_owned(), name.to_owned()]);
    // JSON compiler messages so the editor can parse the build's diagnostics
    // for its inspectors and file badges (rendered text still streams to the
    // log).
    args.extend([
        "--message-format".to_owned(),
        "json-render-diagnostics".to_owned(),
    ]);
    let host = std::env::current_exe().ok().filter(|exe| {
        let target = project_root.join("target");
        exe.canonicalize().ok().zip(target.canonicalize().ok()).is_some_and(|(e, t)| e.starts_with(t))
    });
    if let Some(stem) = host.as_ref().and_then(|exe| exe.file_stem()).map(|s| s.to_string_lossy().into_owned()) {
        args.extend(["-p".to_owned(), stem, "--features".to_owned(), format!("{name}/editor")]);
    } else {
        args.extend(["--features".to_owned(), "editor".to_owned()]);
    }
    args
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const DYLIB_EXT: &str = "so";
#[cfg(target_os = "macos")]
const DYLIB_EXT: &str = "dylib";
#[cfg(target_os = "windows")]
const DYLIB_EXT: &str = "dll";

fn dylib_name(package: &str) -> String {
    let stem = package.replace('-', "_");
    #[cfg(target_os = "windows")]
    return format!("{stem}.dll");
    #[cfg(not(target_os = "windows"))]
    format!("lib{stem}.{DYLIB_EXT}")
}

/// Starter plugin written by Tools → Create Component Plugin.
pub fn create_template(project_root: &Path) -> Result<PathBuf> {
    let dir = project_root.join("plugins/components");
    if dir.join("Cargo.toml").exists() {
        return Ok(dir);
    }
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(
        dir.join("Cargo.toml"),
        r#"[package]
name = "zestengine-project-components"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[features]
# The editor builds plugins with `--features editor` (inspector + gizmo + egui);
# a shipped game builds without it, so the editor crate and egui never link in.
default = []
editor = ["dep:zestengine-editor", "dep:egui"]

[dependencies]
zestengine-core.workspace = true
zestengine-editor = { workspace = true, optional = true }
egui = { workspace = true, optional = true }
serde.workspace = true
glam.workspace = true
"#,
    )?;
    std::fs::write(
        dir.join("src/lib.rs"),
        r#"//! Project components, hot-loaded by the zestengine editor (Tools → Build &
//! Reload Components).
//!
//! Every BUILT-IN component type (LightComponent, the colliders, RigidBody,
//! CameraComponent, water bodies, audio, sprites, navmesh, particles, ...) is a
//! `zestengine_core` type too, so plugins read and mutate them directly:
//! `ctx.get::<LightComponent>(target)` and
//! `ctx.modify::<LightComponent>(target, |l| l.intensity = 2.0)` (deferred to
//! after the update pass).
//!
//! WORLD CONTROL, all deferred through the same ctx (applied after the pass):
//! - cameras: `ctx.set_active_camera(target)`;
//! - teleport / player position: `ctx.set_world_position_of(target, pos)` or
//!   `ctx.set_world_transform_of(target, Some(pos), Some(rot))` (world space;
//!   the engine converts through the parent chain);
//! - shaders via materials: `ctx.material_params(target, slot)` reads the live
//!   `zestengine_core::MaterialParams` (slot 0 = the main material) and
//!   `ctx.modify_material(target, slot, |m| m.emission_intensity = 2.0)` edits
//!   them - so a flicker is one line on the material's emissiveness and one
//!   on the light's intensity. `ctx.assign_material(target, slot,
//!   "materials/x.material")` swaps in a `.material` file at runtime;
//! - events are pub/sub: a component declares what it wants in
//!   `fn subscribed_events(&self, sink: &mut dyn FnMut(std::any::TypeId))`
//!   (`sink(TypeId::of::<MyEvent>());`) and receives only those in
//!   `fn on_event(&mut self, ctx, event: &dyn Any)` (downcast the event,
//!   delivered before `update`). `ctx.emit(MyEvent { .. })` publishes,
//!   readable by everyone next frame (`ctx.events::<MyEvent>()` polls
//!   instead, if a component prefers that);
//! - physics, nav, audio, animator, save data, scene loads, spawns
//!   and networking all have ctx methods (`add_force`, `set_destination`,
//!   `play_sound_at`, `set_animator_float`, `emit`, `save_value`,
//!   `spawn_prefab`, `request_ownership`, ...).
//!
//! A component's RUNTIME behaviour is `zestengine_core::TypedComponent` (no UI) and
//! always compiles. Its EDITOR inspector (`zestengine_editor::Inspect`) and gizmo
//! (`zestengine_editor::Gizmo`) are gated behind the `editor` feature, so a shipped
//! game (built without that feature) links neither zestengine-editor nor egui.

mod orbit;

use zestengine_core::{ComponentRegistry, TypedComponent};
#[cfg(feature = "editor")]
use zestengine_editor::EditorComponents;

// ABI stamp: lets a host (the editor, a dedicated server loading this cdylib) refuse a build from a
// different rustc / engine version instead of crashing.
zestengine_core::export_plugin_abi!();

/// Register runtime behaviour (called by the editor and a shipped game).
#[unsafe(no_mangle)]
pub fn zestengine_register(registry: &mut ComponentRegistry) {
    registry.register::<orbit::Orbit>();
    // zestengine: new components register here
}

/// Register editor-only traits; compiled only with `--features editor`.
#[cfg(feature = "editor")]
#[unsafe(no_mangle)]
pub fn zestengine_register_editor(editor: &mut EditorComponents) {
    editor.register::<orbit::Orbit>();
    // zestengine: new components register their editor traits here
}
"#,
    )?;
    std::fs::write(
        dir.join("rust-analyzer.toml"),
        "# rust-analyzer per-crate config: the plugin's editor-only deps (egui,\n# zestengine-editor) are gated behind the `editor` feature, which no build\n# enables by default. Enabling it here puts egui and the editor crate into\n# rust-analyzer's crate graph so `egui::…` / `zestengine_editor::…` complete.\n[cargo]\nfeatures = [\"editor\"]\n",
    )?;
    std::fs::write(
        dir.join("src/orbit.rs"),
        r#"//! Example component: circles around the object's authored position.

use serde::{Deserialize, Serialize};
use zestengine_core::{ComponentCtx, TypedComponent};
#[cfg(feature = "editor")]
use zestengine_editor::{Gizmo, Inspect, InspectCtx};

/// Circle around the object's authored position.
#[derive(Serialize, Deserialize)]
pub struct Orbit {
    pub degrees_per_second: f32,
    pub radius: f32,
    #[serde(skip)]
    applied: glam::Vec3,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            degrees_per_second: 90.0,
            radius: 1.5,
            applied: glam::Vec3::ZERO,
        }
    }
}

impl TypedComponent for Orbit {
    const NAME: &'static str = "Orbit";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let angle = ctx.time * self.degrees_per_second.to_radians();
        let offset = glam::Vec3::new(angle.cos(), 0.0, angle.sin()) * self.radius;
        *ctx.translation += offset - self.applied;
        self.applied = offset;
    }
}

#[cfg(feature = "editor")]
impl Inspect for Orbit {
    fn inspector_ui(&mut self, ui: &mut egui::Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.degrees_per_second, -360.0..=360.0).text("Speed (°/s)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.radius, 0.0..=10.0).text("Radius"))
            .changed();
        changed
    }
}

#[cfg(feature = "editor")]
impl Gizmo for Orbit {}
"#,
    )?;
    Ok(dir)
}

/// Run `cargo clippy` over every plugin crate on a background thread;
/// rustc errors and clippy lints arrive on the returned channel as
/// simplified diagnostics. (rust-analyzer LSP integration is the eventual
/// upgrade; clippy already includes all compiler errors.)
pub fn run_check(project_root: &Path) -> mpsc::Receiver<Vec<CodeDiagnostic>> {
    let (tx, rx) = mpsc::channel();
    let root = project_root.to_owned();
    std::thread::spawn(move || {
        let mut diagnostics = Vec::new();
        for (_dir, name) in discover(&root) {
            let output = Command::new("cargo")
                .current_dir(&root)
                .args(["clippy", "-p", &name, "--message-format=json"])
                .output();
            match output {
                Ok(output) => parse_cargo_messages(&output.stdout, &mut diagnostics),
                Err(e) => diagnostics.push(CodeDiagnostic {
                    level: "error".into(),
                    file: String::new(),
                    line: 0,
                    message: format!("running cargo clippy: {e}"),
                }),
            }
        }
        let _ = tx.send(diagnostics);
    });
    rx
}

fn parse_cargo_messages(stdout: &[u8], out: &mut Vec<CodeDiagnostic>) {
    for line in stdout.split(|&b| b == b'\n') {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(line) else {
            continue;
        };
        if value["reason"] != "compiler-message" {
            continue;
        }
        let message = &value["message"];
        let level = message["level"].as_str().unwrap_or_default();
        if level != "error" && level != "warning" {
            continue;
        }
        let text = message["message"].as_str().unwrap_or_default();
        // Primary span, if any (some messages are crate-level).
        let (file, line_no) = message["spans"]
            .as_array()
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|s| s["is_primary"] == true)
                    .or_else(|| spans.first())
            })
            .map(|s| {
                (
                    s["file_name"].as_str().unwrap_or_default().to_owned(),
                    s["line_start"].as_u64().unwrap_or(0) as u32,
                )
            })
            .unwrap_or_default();
        out.push(CodeDiagnostic {
            level: level.to_owned(),
            file,
            line: line_no,
            message: text.to_owned(),
        });
    }
}

const REGISTER_MARKER: &str = "// zestengine: new components register here";
const EDITOR_REGISTER_MARKER: &str =
    "// zestengine: new components register their editor traits here";

/// Files → Create → New Component: a fresh component module in the plugin
/// crate (created on demand), prefilled with every engine-called hook, and
/// registered in `zestengine_register`. Returns the new `.rs` path.
pub fn create_component(project_root: &Path) -> Result<PathBuf> {
    let dir = create_template(project_root)?;
    let src = dir.join("src");

    let (mod_name, file) = (0..1000)
        .map(|n| {
            let name = if n == 0 {
                "new_component".to_owned()
            } else {
                format!("new_component_{n}")
            };
            let file = src.join(format!("{name}.rs"));
            (name, file)
        })
        .find(|(_, file)| !file.exists())
        .context("no free component file name")?;
    let struct_name: String = mod_name
        .split('_')
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect();

    std::fs::write(&file, component_template(&struct_name))?;

    // Wire it into lib.rs: the `mod` declaration goes at the TOP (after the
    // file's `//!` doc lines, which inner attributes pin to the very top),
    // and the registrations follow the template markers.
    let lib = src.join("lib.rs");
    let mut text = std::fs::read_to_string(&lib).context("reading plugin lib.rs")?;
    let mut insert_at = 0usize;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("//!") || t.is_empty() {
            insert_at += line.len() + 1;
        } else {
            break;
        }
    }
    text.insert_str(insert_at, &format!("mod {mod_name};\n"));
    let registration = format!("    registry.register::<{mod_name}::{struct_name}>();\n");
    if let Some(marker) = text.find(REGISTER_MARKER) {
        let line_start = text[..marker].rfind('\n').map_or(0, |i| i + 1);
        text.insert_str(line_start, &registration);
    } else if let Some(at) = text.find("fn zestengine_register") {
        if let Some(brace) = text[at..].find('{') {
            text.insert_str(at + brace + 1, &format!("\n{registration}"));
        }
    } else {
        bail!("plugin lib.rs has no zestengine_register function to extend");
    }
    // Editor-trait registration (best-effort; gated behind the `editor` feature
    // in the template).
    let editor_reg = format!("    editor.register::<{mod_name}::{struct_name}>();\n");
    if let Some(marker) = text.find(EDITOR_REGISTER_MARKER) {
        let line_start = text[..marker].rfind('\n').map_or(0, |i| i + 1);
        text.insert_str(line_start, &editor_reg);
    }
    std::fs::write(&lib, text)?;
    Ok(file)
}

/// Starter component: runtime behaviour always compiles; the editor inspector
/// + gizmo are gated behind the `editor` feature.
fn component_template(name: &str) -> String {
    format!(
        r#"//! {name} — describe what it does.
//!
//! Rename the struct + NAME, then Tools → Build & Reload Components.

use zestengine_core::{{ComponentCtx, TypedComponent}};
#[cfg(feature = "editor")]
use zestengine_editor::{{Gizmo, Inspect, InspectCtx}};
use serde::{{Deserialize, Serialize}};

/// Your component's authored data: serialized into the .scene and shown in
/// the inspector. Add fields here (and in `Default`).
#[derive(Serialize, Deserialize)]
pub struct {name} {{}}

impl Default for {name} {{
    fn default() -> Self {{
        Self {{}}
    }}
}}

impl TypedComponent for {name} {{
    /// Shown in the Add Component menu.
    const NAME: &'static str = "{name}";

    /// Called every frame while playing. `ctx` is the whole world-control
    /// surface: `ctx.get_self::<LightComponent>()`, `ctx.modify::<T>`,
    /// `ctx.material_params`, `ctx.set_world_position_of`, ...
    fn update(&mut self, ctx: &mut ComponentCtx) {{
        let _ = ctx;
    }}
}}

/// Editor inspector (editor build only): draw your fields here.
#[cfg(feature = "editor")]
impl Inspect for {name} {{
    fn inspector_ui(&mut self, ui: &mut egui::Ui, _ctx: &InspectCtx) -> bool {{
        ui.label(egui::RichText::new("no fields yet").weak());
        false
    }}
}}

/// Editor viewport gizmo (editor build only). Override `draw_gizmo` to draw.
#[cfg(feature = "editor")]
impl Gizmo for {name} {{}}
"#
    )
}

#[cfg(test)]
mod status_tests {
    use super::*;

    #[test]
    fn first_error_skips_warnings_and_the_summary() {
        let out = "   Compiling x v0.1.0\nwarning: unused\nerror[E0425]: cannot find value `y` in this scope\n --> src/lib.rs:3:5\nerror: could not compile `x` (lib) due to 1 previous error\n";
        assert_eq!(first_error_line(out).as_deref(), Some("error[E0425]: cannot find value `y` in this scope"));
        assert_eq!(first_error_line("error: could not compile `x`").as_deref(), Some("error: could not compile `x`"));
        assert_eq!(first_error_line("warning: only\n"), None);
        assert!(first_error_line(&format!("error: {}", "z".repeat(500))).unwrap().chars().count() <= 141);
    }

    #[test]
    fn status_lines() {
        assert_eq!(status_building("p", 0), "Building component plugins… (p)");
        assert_eq!(status_building("p", 12), "Building component plugins… 12 s (p)");
        assert!(status_up_to_date("p").contains("up to date"));
        assert!(status_failed("p", "error: boom").ends_with("error: boom"));
        assert_eq!(status_failed("p", ""), "Component plugin p failed to build");
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;

    #[test]
    fn fingerprint_changes_with_plugin_or_crate_sources_only() {
        let root = std::env::temp_dir().join(format!("zest_plugin_fp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let plugin = root.join("plugins/p");
        std::fs::create_dir_all(plugin.join("src")).unwrap();
        std::fs::create_dir_all(root.join("crates/core/src")).unwrap();
        std::fs::create_dir_all(root.join("target/debug")).unwrap();
        std::fs::write(plugin.join("src/lib.rs"), "a").unwrap();
        std::fs::write(root.join("crates/core/src/lib.rs"), "a").unwrap();
        let a = build_fingerprint(&root, &plugin);
        assert_eq!(a, build_fingerprint(&root, &plugin), "stable");
        // Build output must not invalidate itself.
        std::fs::write(root.join("target/debug/out.so"), "x").unwrap();
        assert_eq!(a, build_fingerprint(&root, &plugin));
        std::fs::write(plugin.join("src/lib.rs"), "changed length").unwrap();
        let b = build_fingerprint(&root, &plugin);
        assert_ne!(a, b, "plugin source edit");
        std::fs::write(root.join("crates/core/src/lib.rs"), "changed length").unwrap();
        assert_ne!(b, build_fingerprint(&root, &plugin), "linked crate edit");
        let _ = std::fs::remove_dir_all(&root);
    }
}
