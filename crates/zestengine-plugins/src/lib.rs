#![allow(dead_code)]
//! Rust component plugins: crates under `plugins/`, built by the editor
//! (`cargo build -p <name>`) and hot-loaded as dylibs.
//!
//! Plugins of this checkout are workspace members, so their dependency versions
//! are pinned by the same Cargo.lock as the editor. A game project is its own
//! workspace: the editor stages that plugin into this checkout and builds it
//! with `--workspace`, the same graph as `just build`. That (plus the same
//! rustc and profile) is what makes sharing `dyn Component` across the dylib
//! boundary safe in practice.
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

impl Drop for PluginHost {
    fn drop(&mut self) {
        // ECS component types, queued destruction operations and registration
        // callbacks can outlive the host that loaded them. Keep their code
        // mapped through process shutdown, as the runtime-only host already does.
        for library in self.libs.drain(..) { std::mem::forget(library); }
    }
}

/// Compare the plugin's exported ABI stamp (`zestengine_core::export_plugin_abi!()`) with this
/// build's before passing Rust-ABI types across the library boundary. Unstamped
/// plugins must be rebuilt: their component vtables may predate ECS support.
fn check_abi(lib: &libloading::Library, what: &str) -> Result<()> {
    // SAFETY: `extern "C"` returning plain `#[repr(C)]` data: safe to call across compilers.
    let stamp = unsafe { lib.get::<extern "C" fn() -> PluginAbi>(b"zestengine_plugin_abi").ok().map(|f| f()) };
    match stamp {
        Some(s) => {
            s.check_against(&PluginAbi::current()).map_err(|e| anyhow::anyhow!("{what}: {e}"))?;
            Ok(())
        }
        None => bail!("{what} has no ABI stamp: add `zestengine_core::export_plugin_abi!();` to the plugin crate and rebuild it"),
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
        check_abi(&lib, &what)?;
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
        for (name, f) in &self.registrations {
            registry.register_plugin(name, *f);
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
        // External projects get their own lockfile and `[profile.release]`.
        // Pointing `CARGO_TARGET_DIR` at this checkout is not enough: cargo's
        // unit id still differs, so `TypeId`s of `zestengine-core` differ and
        // the host refuses the dylib. Stage the plugin into this checkout and
        // build `--workspace` (same graph as `just build`) instead.
        let checkout = engine_checkout(project_root);
        let external = checkout.as_ref().is_some_and(|c| !same_path(c, project_root));
        let target_dir = checkout
            .as_ref()
            .map(|root| root.join("target"))
            .unwrap_or_else(|| project_root.join("target"));
        let mut built = Vec::new();
        let mut pending_external: Vec<PendingPlugin> = Vec::new();
        for (dir, name) in discover(project_root) {
            // Nothing changed since the last successful build (sources of the
            // plugin and of the workspace crates it links, lockfile, this editor
            // binary): skip cargo, whose no-op run still resolves and
            // fingerprints the whole workspace (seconds on a big one).
            let t_total = std::time::Instant::now();
            let print = build_fingerprint(project_root, &dir);
            let dylib = if external {
                external_plugin_dir(checkout.as_ref().unwrap(), project_root).join(dylib_name(&name))
            } else {
                target_dir.join(PROFILE_DIR).join(dylib_name(&name))
            };
            let stamp = project_root.join("target/zestengine-plugins").join(format!("{name}.stamp"));
            if std::fs::read_to_string(&stamp).ok().and_then(|t| t.trim().parse::<u64>().ok()) == Some(print) && dylib.is_file() {
                tracing::info!("component plugin {name} unchanged, skipping cargo build ({:?} to check)", t_total.elapsed());
                report(&status_up_to_date(&name));
                built.push(name);
                continue;
            }
            if external {
                report(&status_building(&name, 0));
                pending_external.push(PendingPlugin {
                    dir,
                    name,
                    print,
                    stamp,
                    started: t_total,
                });
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
        if !pending_external.is_empty() {
            let checkout = checkout.expect("external plugin build has an engine checkout");
            build_external_plugins(&checkout, project_root, &pending_external, report, diag_sink)?;
            for plugin in &pending_external {
                built.push(plugin.name.clone());
            }
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
        // External projects publish into a side directory of the engine target
        // (`built_dylib`). A later `cargo build --workspace` of this checkout
        // rebuilds the sample plugin at `target/<profile>/lib<name>.so` and
        // must not clobber the game's dylib. Per-process copies stay
        // project-local so two editor instances never overwrite each other's
        // mmapped file.
        let built = built_dylib(project_root, name);
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
            check_abi(&lib, &format!("plugin {name}"))?;
            let register: libloading::Symbol<fn(&mut ComponentRegistry)> = lib
                .get(b"zestengine_register")
                .context("plugin has no `zestengine_register` export")?;
            registry.register_plugin(name, *register);
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

/// One plugin cargo still has to compile. `print` is the fingerprint taken
/// before the build; an edit during the build leaves a stale stamp.
struct PendingPlugin {
    dir: PathBuf,
    name: String,
    print: u64,
    stamp: PathBuf,
    started: std::time::Instant,
}

struct CargoJob {
    cwd: PathBuf,
    target_dir: PathBuf,
    args: Vec<String>,
    log_name: String,
    /// When non-empty, compiler warnings outside these directories are not
    /// recorded. A `--workspace` build would otherwise badge every engine file.
    keep_files_under: Vec<String>,
}

/// Restores a file's exact bytes when dropped, including on cargo failure.
struct RestoreBytes {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl Drop for RestoreBytes {
    fn drop(&mut self) {
        let _ = std::fs::write(&self.path, &self.bytes);
    }
}

/// A manifest swapped into the engine workspace for one plugin build.
struct StagedManifest {
    path: PathBuf,
    /// Previous contents. `None` when `remove_dir` was created for this build.
    original: Option<Vec<u8>>,
    remove_dir: Option<PathBuf>,
}

impl Drop for StagedManifest {
    fn drop(&mut self) {
        if let Some(bytes) = self.original.take() {
            let _ = std::fs::write(&self.path, bytes);
        } else if let Some(dir) = self.remove_dir.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn same_path(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn json_string_field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{key}\":\"");
    let at = text.find(&pattern)?;
    let start = at + pattern.len();
    text[start..].split('"').next()
}

/// Crate directory of the first `zestengine-engine` dependency object.
///
/// `cargo metadata --no-deps` lists that dependency (`"path"` is the crate
/// directory) before any package `"manifest_path"`. The next manifest in an
/// external project is the project's own `Cargo.toml`, whose grandparent is
/// not the engine checkout.
fn engine_crate_dir_from_metadata(text: &str) -> Option<&str> {
    let key = "\"name\":\"zestengine-engine\"";
    let at = text.find(key)?;
    let rest = &text[at + key.len()..];
    let end = rest.find('}')?;
    json_string_field(&rest[..end], "path")
}

fn checkout_of_engine_crate(crate_dir: &Path) -> Option<PathBuf> {
    // `.../crates/zestengine-engine` -> checkout root.
    crate_dir.parent()?.parent().map(|dir| dir.to_path_buf())
}

/// This checkout, when `project_root`'s metadata path-depends on it.
fn engine_checkout(project_root: &Path) -> Option<PathBuf> {
    let out = Command::new("cargo")
        .current_dir(project_root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let crate_dir = engine_crate_dir_from_metadata(&text)?;
    let checkout = checkout_of_engine_crate(Path::new(crate_dir))?;
    (checkout.join("Cargo.toml").is_file() && checkout.join("crates/zestengine-engine/Cargo.toml").is_file())
        .then_some(checkout)
}

/// `target/<profile>/zest-plugins/<project>-<hash>/`, not the sample plugin's
/// `lib<package>.so` (a workspace build of this checkout overwrites that name).
fn external_plugin_dir(checkout: &Path, project_root: &Path) -> PathBuf {
    let name = project_root.file_name().and_then(|s| s.to_str()).unwrap_or("project");
    let mut hash = 0xcbf29ce484222325u64;
    for b in project_root.as_os_str().as_encoded_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    checkout
        .join("target")
        .join(PROFILE_DIR)
        .join("zest-plugins")
        .join(format!("{name}-{hash:x}"))
}

fn built_dylib(project_root: &Path, name: &str) -> PathBuf {
    let file = dylib_name(name);
    match engine_checkout(project_root) {
        Some(checkout) if !same_path(&checkout, project_root) => {
            external_plugin_dir(&checkout, project_root).join(file)
        }
        Some(checkout) => checkout.join("target").join(PROFILE_DIR).join(file),
        None => project_root.join("target").join(PROFILE_DIR).join(file),
    }
}

fn toml_table<'a>(manifest: &'a str, header: &str) -> Option<&'a str> {
    let start = manifest.find(header)? + header.len();
    let rest = &manifest[start..];
    let end = rest.find("\n[").unwrap_or(rest.len());
    Some(&rest[..end])
}

fn plugin_lib_rs(plugin_dir: &Path, manifest: &str) -> PathBuf {
    let rel = toml_table(manifest, "[lib]").and_then(|table| {
        table.lines().find_map(|line| {
            let rest = line.trim().strip_prefix("path")?.trim_start();
            let rest = rest.strip_prefix('=')?.trim().trim_matches('"');
            Some(PathBuf::from(rest))
        })
    });
    match rel {
        Some(path) if path.is_absolute() => path,
        Some(path) => plugin_dir.join(path),
        None => plugin_dir.join("src/lib.rs"),
    }
}

fn inject_lib_path(manifest: &str, lib_rs: &Path) -> String {
    let path_line = format!("path = \"{}\"", lib_rs.display());
    let Some(table_at) = manifest.find("[lib]") else {
        return format!("[lib]\n{path_line}\ncrate-type = [\"cdylib\", \"rlib\"]\n\n{manifest}");
    };
    let table = toml_table(manifest, "[lib]").unwrap_or("");
    if let Some(line) = table.lines().find(|line| line.trim_start().starts_with("path")) {
        return manifest.replacen(line, &path_line, 1);
    }
    let insert_at = table_at + "[lib]".len();
    let mut out = String::new();
    out.push_str(&manifest[..insert_at]);
    out.push('\n');
    out.push_str(&path_line);
    out.push_str(&manifest[insert_at..]);
    out
}

/// The editor loads inspectors, so the staged build enables `editor`. It is
/// the package default for that one `cargo build --workspace`: `--features`
/// on a virtual workspace is not the plugin's feature.
fn enable_editor_default(manifest: &str) -> String {
    let Some(features) = toml_table(manifest, "[features]") else {
        return manifest.to_owned();
    };
    let defines_editor = features.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("editor") && trimmed.trim_start_matches("editor").trim_start().starts_with('=')
    });
    if !defines_editor {
        return manifest.to_owned();
    }
    if let Some(line) = features.lines().find(|line| line.trim_start().starts_with("default")) {
        if line.contains("editor") {
            return manifest.to_owned();
        }
        if let Some(open) = line.find('[') {
            let mut owned = line.to_owned();
            if line.trim().ends_with("[]") {
                owned.insert_str(open + 1, "\"editor\"");
            } else {
                owned.insert_str(open + 1, "\"editor\", ");
            }
            return manifest.replacen(line, &owned, 1);
        }
    }
    manifest.replacen("[features]", "[features]\ndefault = [\"editor\"]", 1)
}

fn manifest_built_in_engine(original: &str, lib_rs: &Path) -> String {
    enable_editor_default(&inject_lib_path(original, lib_rs))
}

fn member_manifest(checkout: &Path, package: &str) -> Option<PathBuf> {
    for dir in ["plugins", "crates"] {
        let Ok(entries) = std::fs::read_dir(checkout.join(dir)) else { continue };
        for entry in entries.flatten() {
            let manifest = entry.path().join("Cargo.toml");
            if let Ok(text) = std::fs::read_to_string(&manifest)
                && package_name(&text).as_deref() == Some(package)
            {
                return Some(manifest);
            }
        }
    }
    None
}

fn stage_plugin(checkout: &Path, plugin_dir: &Path, package: &str) -> Result<StagedManifest> {
    let original_text = std::fs::read_to_string(plugin_dir.join("Cargo.toml"))
        .with_context(|| format!("reading {package} manifest"))?;
    let lib_rs = plugin_lib_rs(plugin_dir, &original_text);
    if !lib_rs.is_file() {
        bail!("{package} has no library at {}", lib_rs.display());
    }
    let staged_text = manifest_built_in_engine(&original_text, &lib_rs);
    if let Some(dest) = member_manifest(checkout, package) {
        let original = std::fs::read(&dest).with_context(|| format!("reading {}", dest.display()))?;
        if let Err(err) = std::fs::write(&dest, staged_text) {
            let _ = std::fs::write(&dest, &original);
            return Err(err).context(format!("staging {package} into {}", dest.display()));
        }
        return Ok(StagedManifest { path: dest, original: Some(original), remove_dir: None });
    }
    let dir = checkout.join("plugins").join(format!("zest-external-{package}"));
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join("Cargo.toml");
    std::fs::write(&dest, staged_text).with_context(|| format!("writing {}", dest.display()))?;
    Ok(StagedManifest { path: dest, original: None, remove_dir: Some(dir) })
}

fn external_workspace_build_args() -> Vec<String> {
    let mut args = vec!["build".to_owned(), "--workspace".to_owned()];
    if !cfg!(debug_assertions) {
        args.push("--release".to_owned());
    }
    // No `[profile.release]` overrides. `just build` uses this checkout's
    // release profile. Overrides are a different cargo unit even when the
    // values match the defaults, and a rebuilt zestengine-core does not match
    // the editor that is already installed.
    args.extend(["--message-format".to_owned(), "json-render-diagnostics".to_owned()]);
    args
}

fn publish_external_dylib(checkout: &Path, project_root: &Path, package: &str) -> Result<()> {
    let produced = checkout.join("target").join(PROFILE_DIR).join(dylib_name(package));
    if !produced.is_file() {
        bail!("building {package} produced no {}", produced.display());
    }
    let dest_dir = external_plugin_dir(checkout, project_root);
    std::fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(dylib_name(package));
    std::fs::copy(&produced, &dest).with_context(|| format!("copying {} to {}", produced.display(), dest.display()))?;
    Ok(())
}

fn build_external_plugins(
    checkout: &Path,
    project_root: &Path,
    pending: &[PendingPlugin],
    report: &dyn Fn(&str),
    diag_sink: &std::sync::Arc<std::sync::Mutex<Vec<CodeDiagnostic>>>,
) -> Result<()> {
    let lock_path = checkout.join("Cargo.lock");
    let lock_bytes = std::fs::read(&lock_path).with_context(|| format!("reading {}", lock_path.display()))?;
    // Plugin-only crates (rand, …) are not in this lock. Cargo adds them for
    // the build; put the lock back so opening a project does not dirty it.
    let _lock_restore = RestoreBytes { path: lock_path, bytes: lock_bytes };
    let mut staged = Vec::new();
    for plugin in pending {
        staged.push(stage_plugin(checkout, &plugin.dir, &plugin.name)?);
    }
    let log_name = pending.iter().map(|plugin| plugin.name.as_str()).collect::<Vec<_>>().join(", ");
    let keep_files_under = pending.iter().map(|plugin| plugin.dir.display().to_string()).collect();
    run_plugin_cargo(
        &CargoJob {
            cwd: checkout.to_path_buf(),
            target_dir: checkout.join("target"),
            args: external_workspace_build_args(),
            log_name,
            keep_files_under,
        },
        project_root,
        report,
        diag_sink,
    )?;
    for plugin in pending {
        publish_external_dylib(checkout, project_root, &plugin.name)?;
        if let Some(dir) = plugin.stamp.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&plugin.stamp, plugin.print.to_string());
        tracing::info!("component plugin {} cargo build took {:?}", plugin.name, plugin.started.elapsed());
    }
    drop(staged);
    Ok(())
}

fn run_plugin_cargo(
    job: &CargoJob,
    project_root: &Path,
    report: &dyn Fn(&str),
    diag_sink: &std::sync::Arc<std::sync::Mutex<Vec<CodeDiagnostic>>>,
) -> Result<()> {
    let mut child = Command::new("cargo")
        .current_dir(&job.cwd)
        .env("CARGO_TARGET_DIR", &job.target_dir)
        .args(&job.args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("running cargo")?;
    let mut pipe = child.stderr.take();
    let name_err = job.log_name.clone();
    let keep_err = job.keep_files_under.clone();
    let reader = std::thread::spawn(move || {
        let mut text = Vec::new();
        if let Some(p) = pipe.as_mut() {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(p).lines().map_while(Result::ok) {
                // A workspace build prints every fresh crate's warnings. Keep
                // them in the failure text, but don't stream engine noise.
                let interesting = keep_err.is_empty()
                    || line.contains("error")
                    || keep_err.iter().any(|dir| line.contains(dir.as_str()));
                if interesting {
                    tracing::info!("{name_err}: {line}");
                }
                text.extend_from_slice(line.as_bytes());
                text.push(b'\n');
            }
        }
        String::from_utf8_lossy(&text).into_owned()
    });
    let mut out_pipe = child.stdout.take();
    let name_out = job.log_name.clone();
    let sink = diag_sink.clone();
    let proj = project_root.to_owned();
    let keep = job.keep_files_under.clone();
    std::thread::spawn(move || {
        if let Some(p) = out_pipe.as_mut() {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(p).lines().map_while(Result::ok) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line)
                    && let Some(rendered) = v.get("rendered").and_then(|r| r.as_str())
                {
                    let span = v
                        .get("message")
                        .and_then(|msg| msg.get("spans"))
                        .and_then(|s| s.as_array())
                        .and_then(|s| s.first())
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let file_raw = span.get("file_name").and_then(|f| f.as_str()).unwrap_or("");
                    let ours = keep.is_empty() || keep.iter().any(|dir| file_raw.contains(dir.as_str()));
                    if !ours && v.get("reason").and_then(|r| r.as_str()) == Some("compiler-message") {
                        continue;
                    }
                    tracing::info!("{name_out}: {rendered}");
                    if ours
                        && v.get("reason").and_then(|r| r.as_str()) == Some("compiler-message")
                        && let Some(msg) = v.get("message")
                    {
                        let level = msg.get("level").and_then(|l| l.as_str()).unwrap_or("error");
                        let message = msg.get("message").and_then(|m| m.as_str()).unwrap_or("").to_owned();
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
                        let line_no = span.get("line_start").and_then(|l| l.as_u64()).map(|l| l as u32 + 1).unwrap_or(0);
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
            report(&status_building(&job.log_name, secs));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    let stderr = reader.join().unwrap_or_default();
    if !status.success() {
        report(&status_failed(&job.log_name, &stderr));
        bail!("building {} failed:\n{}", job.log_name, stderr.trim());
    }
    if job.keep_files_under.is_empty() && stderr.contains("warning:") {
        tracing::warn!("plugin {} built with warnings:\n{}", job.log_name, stderr.trim());
    }
    Ok(())
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
    std::fs::create_dir_all(dir.join("src/components"))?;
    std::fs::create_dir_all(dir.join("src/systems"))?;
    std::fs::create_dir_all(dir.join("src/entities"))?;
    std::fs::write(dir.join("src/lib.rs"), LIB_TEMPLATE)?;
    std::fs::write(
        dir.join("rust-analyzer.toml"),
        "# rust-analyzer per-crate config: the plugin's editor-only deps (egui,\n# zestengine-editor) are gated behind the `editor` feature, which no build\n# enables by default. Enabling it here puts egui and the editor crate into\n# rust-analyzer's crate graph so `egui::…` / `zestengine_editor::…` complete.\n[cargo]\nfeatures = [\"editor\"]\n",
    )?;
    std::fs::write(dir.join("src/components/orbit.rs"), ORBIT_COMPONENT)?;
    std::fs::write(dir.join("src/systems/orbit.rs"), ORBIT_SYSTEM)?;
    std::fs::write(dir.join("src/entities/orbiter.rs"), ORBITER_ENTITY)?;
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
const SYSTEM_REGISTER_MARKER: &str = "// zestengine: new systems register here";
const ENTITY_REGISTER_MARKER: &str = "// zestengine: new entities register here";
const EDITOR_REGISTER_MARKER: &str =
    "// zestengine: new components register their editor traits here";
const COMPONENT_MODULE_MARKER: &str = "// zestengine: new component modules here";
const SYSTEM_MODULE_MARKER: &str = "// zestengine: new system modules here";
const ENTITY_MODULE_MARKER: &str = "// zestengine: new entity modules here";

/// What Files → New Rust can create in the project's plugin crate. Each kind
/// gets its own folder under `src/` and is registered in `lib.rs`, which
/// consolidates all three into the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustItem {
    /// Serializable data an object carries (shown in the inspector).
    Component,
    /// Behaviour that runs every frame over component data.
    System,
    /// A ready-made object: a mesh shape plus the components it starts with.
    Entity,
}

impl RustItem {
    /// Every kind, in menu order.
    pub const ALL: [RustItem; 3] = [Self::Component, Self::System, Self::Entity];

    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Component => "Component",
            Self::System => "System",
            Self::Entity => "Entity",
        }
    }

    /// Menu tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Component => "Data an object carries: fields, inspector and gizmo (src/components/)",
            Self::System => "Behaviour that runs every frame over component data (src/systems/)",
            Self::Entity => "A ready-made object, a mesh shape plus starting components, listed under Create > Entity (src/entities/)",
        }
    }

    fn folder(self) -> &'static str {
        match self {
            Self::Component => "components",
            Self::System => "systems",
            Self::Entity => "entities",
        }
    }

    fn stem(self) -> &'static str {
        match self {
            Self::Component => "new_component",
            Self::System => "new_system",
            Self::Entity => "new_entity",
        }
    }

    fn module_marker(self) -> &'static str {
        match self {
            Self::Component => COMPONENT_MODULE_MARKER,
            Self::System => SYSTEM_MODULE_MARKER,
            Self::Entity => ENTITY_MODULE_MARKER,
        }
    }

    fn register_marker(self) -> &'static str {
        match self {
            Self::Component => REGISTER_MARKER,
            Self::System => SYSTEM_REGISTER_MARKER,
            Self::Entity => ENTITY_REGISTER_MARKER,
        }
    }

    fn template(self) -> &'static str {
        match self {
            Self::Component => COMPONENT_TEMPLATE,
            Self::System => SYSTEM_TEMPLATE,
            Self::Entity => ENTITY_TEMPLATE,
        }
    }
}

/// Insert `line` (newline included) on its own line before the line holding
/// `marker`. False when the marker is absent.
fn insert_before_marker(text: &mut String, marker: &str, line: &str) -> bool {
    let Some(at) = text.find(marker) else { return false };
    let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
    text.insert_str(line_start, line);
    true
}

/// `new_component_1` → `NewComponent1`.
fn struct_name(module: &str) -> String {
    module
        .split('_')
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// `new_component_1` → `New Component 1` (the name scenes, menus and
/// `spawn_entity` use).
fn display_name(module: &str) -> String {
    module
        .split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Files → New Rust → Component / System / Entity: a new module with starter
/// code for that kind, registered in the plugin's `lib.rs` (the one place that
/// hands every component, system and entity to the game). The plugin crate is
/// created on demand. Returns the new `.rs` path.
///
/// Each module exports its own `register`, so lib.rs never names the types
/// inside and renaming a struct does not break it. Projects made before the
/// `components/ systems/ entities/` layout keep flat modules next to lib.rs.
pub fn create_rust_item(project_root: &Path, kind: RustItem) -> Result<PathBuf> {
    let dir = create_template(project_root)?;
    let src = dir.join("src");
    let lib = src.join("lib.rs");
    let mut text = std::fs::read_to_string(&lib).context("reading plugin lib.rs")?;
    let Some(register_at) = text.find("fn zestengine_register") else {
        bail!("plugin lib.rs has no zestengine_register function to extend");
    };
    let Some(brace) = text[register_at..].find('{') else {
        bail!("plugin lib.rs has no zestengine_register function body to extend");
    };
    let body_start = register_at + brace + 1;

    let nested = text.contains(kind.module_marker());
    let folder = if nested { src.join(kind.folder()) } else { src.clone() };
    std::fs::create_dir_all(&folder)?;
    let (mod_name, file) = (0..1000)
        .map(|n| {
            let name = if n == 0 { kind.stem().to_owned() } else { format!("{}_{n}", kind.stem()) };
            let file = folder.join(format!("{name}.rs"));
            (name, file)
        })
        .find(|(name, file)| !file.exists() && !text.contains(&format!("mod {name};")))
        .context("no free file name")?;

    let contents = kind
        .template()
        .replace("@MOD@", &mod_name)
        .replace("@IDENT@", &struct_name(&mod_name))
        .replace("@DISPLAY@", &display_name(&mod_name));
    std::fs::write(&file, contents)?;

    // The module declaration: inside the kind's block in the new layout, at the
    // top (after the file's `//!` doc lines, which inner attributes pin to the
    // very top) for a flat legacy lib.rs.
    let path = if nested {
        insert_before_marker(&mut text, kind.module_marker(), &format!("    pub mod {mod_name};\n"));
        format!("{}::{mod_name}", kind.folder())
    } else {
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
        mod_name.clone()
    };

    // The registration: before the kind's marker, or at the top of
    // `zestengine_register` when a legacy lib.rs has no marker for it.
    let registration = format!("    {path}::register(registry);\n");
    if !insert_before_marker(&mut text, kind.register_marker(), &registration) {
        // The module line above shifted the body by its own length when flat.
        let shift = if nested { 0 } else { format!("mod {mod_name};\n").len() };
        text.insert_str(body_start + shift, &format!("\n{registration}"));
    }
    if kind == RustItem::Component {
        // Best-effort: the editor registration only exists behind the `editor` feature.
        insert_before_marker(&mut text, EDITOR_REGISTER_MARKER, &format!("    {path}::register_editor(editor);\n"));
    }
    std::fs::write(&lib, text)?;
    Ok(file)
}

/// Files → New Rust → Component, kept for callers from before systems and entities.
pub fn create_component(project_root: &Path) -> Result<PathBuf> {
    create_rust_item(project_root, RustItem::Component)
}

/// The plugin crate's `lib.rs`: the single place that consolidates every
/// component, system and entity of the project into the game.
const LIB_TEMPLATE: &str = r#"//! Project gameplay code, compiled and hot-loaded by the zestengine editor
//! (Tools → Build & Reload Components) and linked into the shipped game.
//!
//! Three kinds of things live here, each in its own folder:
//!
//! - `components/`: data an object carries. Serializable, shown in the inspector.
//! - `systems/`: behaviour. Functions that query component data every frame.
//! - `entities/`: ready-made objects (a mesh shape plus starting components),
//!   listed under Create > Entity and spawned from code with `spawn_entity`.
//!
//! Files → New Rust → Component / System / Entity writes a starter file and
//! registers it below. `zestengine_register` is what the editor, the game and a
//! dedicated server call to bring all of it in.

use zestengine_core::ComponentRegistry;
#[cfg(feature = "editor")]
use zestengine_editor::EditorComponents;

mod components {
    pub mod orbit;
    // zestengine: new component modules here
}
mod systems {
    pub mod orbit;
    // zestengine: new system modules here
}
mod entities {
    pub mod orbiter;
    // zestengine: new entity modules here
}

// ABI stamp: lets a host (the editor, a dedicated server loading this cdylib) refuse a build from a
// different rustc / engine version instead of crashing.
zestengine_core::export_plugin_abi!();

/// Register everything the project adds (called by the editor and a shipped game).
#[unsafe(no_mangle)]
pub fn zestengine_register(registry: &mut ComponentRegistry) {
    // Components: data attached to objects.
    components::orbit::register(registry);
    // zestengine: new components register here

    // Systems: behaviour that runs every frame over component data.
    systems::orbit::register(registry);
    // zestengine: new systems register here

    // Entities: ready-made objects (Create > Entity, `spawn_entity`).
    entities::orbiter::register(registry);
    // zestengine: new entities register here
}

/// Register editor-only traits (inspector + gizmo); compiled only with
/// `--features editor` and absent from a shipped game.
#[cfg(feature = "editor")]
#[unsafe(no_mangle)]
pub fn zestengine_register_editor(editor: &mut EditorComponents) {
    components::orbit::register_editor(editor);
    // zestengine: new components register their editor traits here
}
"#;

const ORBIT_COMPONENT: &str = r#"//! Orbit: a component. It only holds data; `systems/orbit.rs` is the behaviour
//! that reads it. `register` is what lib.rs calls.

use serde::{Deserialize, Serialize};
use zestengine_core::{ComponentRegistry, TypedComponent};
#[cfg(feature = "editor")]
use zestengine_editor::{EditorComponents, Gizmo, Inspect, InspectCtx};

/// Circle around the object's authored position.
#[derive(Serialize, Deserialize)]
pub struct Orbit {
    pub degrees_per_second: f32,
    pub radius: f32,
    /// Offset applied so far, so each frame moves by the difference. Runtime
    /// state: `#[serde(skip)]` keeps it out of the saved scene.
    #[serde(skip)]
    pub applied: glam::Vec3,
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
}

/// Called by lib.rs: makes the component available to scenes and systems.
pub fn register(registry: &mut ComponentRegistry) {
    registry.register_ecs::<Orbit>();
}

/// Called by lib.rs in the editor build: adds the inspector and gizmo.
#[cfg(feature = "editor")]
pub fn register_editor(editor: &mut EditorComponents) {
    editor.register::<Orbit>();
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
"#;

const ORBIT_SYSTEM: &str = r#"//! Orbit system: moves every active object that carries an `Orbit` component.
//! `register` is what lib.rs calls.

use zestengine_core::{ComponentRegistry, Transform};
use zestengine_core::ecs::{Owner, Stage, SystemContext, SystemWorld};

use crate::components::orbit::Orbit;

/// `register_system_for` skips the system while no object has an `Orbit`.
pub fn register(registry: &mut ComponentRegistry) {
    registry.register_system_for::<Orbit>("project::orbit", Stage::Update, update);
}

pub fn update(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for (_, (owner, orbit)) in world.query::<(&Owner, &mut Orbit)>().iter() {
        // Disabled objects (and anything under a disabled parent) do not run.
        if !ctx.is_active(world, owner.0) { continue; }
        let angle = ctx.time * orbit.degrees_per_second.to_radians();
        let offset = glam::Vec3::new(angle.cos(), 0.0, angle.sin()) * orbit.radius;
        let mut transform = world.get_mut::<Transform>(owner.0).expect("orbit owner");
        transform.translation += offset - orbit.applied;
        orbit.applied = offset;
    }
}
"#;

const ORBITER_ENTITY: &str = r#"//! Orbiter: an entity, a ready-made object. Create > Entity > Orbiter places a
//! sphere that already carries an `Orbit`, and a system can spawn one while the
//! game runs with `host.spawn_entity("Orbiter", position, rotation)`.

use zestengine_core::ComponentRegistry;
use zestengine_core::ecs::{EntityShape, EntityTemplate};

use crate::components::orbit::Orbit;

/// Called by lib.rs. The name is what the menu and `spawn_entity` use.
pub fn register(registry: &mut ComponentRegistry) {
    registry.register_entity("Orbiter", build);
}

/// Runs for every new instance, so each one gets its own component values.
fn build() -> EntityTemplate {
    EntityTemplate::new(EntityShape::Sphere).with(Orbit::default())
}
"#;

/// Starter component. Placeholders: `@IDENT@` struct, `@DISPLAY@` registered name.
const COMPONENT_TEMPLATE: &str = r#"//! @DISPLAY@: a component, the data an object carries (shown in the inspector
//! and saved in the .scene). It holds no behaviour: write a system
//! (Files → New Rust → System) that queries this type.
//!
//! lib.rs calls `register` below rather than naming the struct, so you can
//! rename the struct freely.

use serde::{Deserialize, Serialize};
use zestengine_core::{ComponentRegistry, TypedComponent};
#[cfg(feature = "editor")]
use zestengine_editor::{EditorComponents, Gizmo, Inspect, InspectCtx};

/// Authored data: add fields here (and in `Default`). Fields marked
/// `#[serde(skip)]` are runtime state: kept while playing, never saved.
#[derive(Serialize, Deserialize)]
pub struct @IDENT@ {}

impl Default for @IDENT@ {
    fn default() -> Self {
        Self {}
    }
}

impl TypedComponent for @IDENT@ {
    /// The stable identity shown in the Add Component menu and stored in
    /// scenes and event bindings. Renaming it orphans saved data.
    const NAME: &'static str = "@DISPLAY@";
}

/// Called by lib.rs: makes the component available to scenes and systems.
pub fn register(registry: &mut ComponentRegistry) {
    registry.register_ecs::<@IDENT@>();
}

/// Called by lib.rs in the editor build: adds the inspector and gizmo.
#[cfg(feature = "editor")]
pub fn register_editor(editor: &mut EditorComponents) {
    editor.register::<@IDENT@>();
}

/// Editor inspector (editor build only): draw your fields here.
#[cfg(feature = "editor")]
impl Inspect for @IDENT@ {
    fn inspector_ui(&mut self, ui: &mut egui::Ui, _ctx: &InspectCtx) -> bool {
        ui.label(egui::RichText::new("no fields yet").weak());
        false
    }
}

/// Editor viewport gizmo (editor build only). Override `draw_gizmo` to draw.
#[cfg(feature = "editor")]
impl Gizmo for @IDENT@ {}
"#;

/// Starter system. Placeholder: `@MOD@` the module (and system) name.
const SYSTEM_TEMPLATE: &str = r#"//! @DISPLAY@: a system, behaviour that runs every frame over component data.
//!
//! A system is a plain function. It queries the components it cares about and
//! edits them directly; anything that reaches beyond (spawning, destroying,
//! sounds, scene loads) is queued as a command the engine applies after the
//! stage. Components live in `components/`, ready-made objects in `entities/`.

use zestengine_core::{ComponentRegistry, Transform};
use zestengine_core::ecs::{Owner, Stage, SystemContext, SystemWorld};

/// Called by lib.rs. The stage decides when the system runs each frame:
/// `FixedUpdate` (60 Hz, before physics), `Start` (once per new instance),
/// `Update`, `Gameplay` (after movement has resolved), `LateUpdate` (after
/// animation: cameras, HUDs) and `Events` (contacts, animation markers).
pub fn register(registry: &mut ComponentRegistry) {
    // Use `register_system_for::<YourComponent>` to skip the system while no
    // object carries that component.
    registry.register_system("project::@MOD@", Stage::Update, update);
}

pub fn update(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    // 1. Query the data you need. Replace `Transform` with your own component,
    //    for example `(&Owner, &mut Health)`; `Owner` is the object the data is on.
    for (_, (owner, transform)) in world.query::<(&Owner, &Transform)>().iter() {
        // 2. Skip disabled objects (and anything under a disabled parent).
        if !ctx.is_active(world, owner.0) { continue; }
        // 3. Your logic. `ctx.dt` is the frame time and `ctx.input` the player's
        //    actions; edit another component with `world.get_mut::<T>(entity)`.
        let _ = (transform, ctx.dt);
    }
}
"#;

/// Starter entity. Placeholder: `@DISPLAY@` the registered name.
const ENTITY_TEMPLATE: &str = r#"//! @DISPLAY@: an entity, a ready-made object (a mesh shape plus the components
//! it starts with). It is listed under Create > Entity in the editor, and a
//! system can spawn it while the game runs with
//! `host.spawn_entity("@DISPLAY@", position, rotation)`.

use zestengine_core::ComponentRegistry;
use zestengine_core::ecs::{EntityShape, EntityTemplate};

/// Called by lib.rs. The name is what the menu and `spawn_entity` use.
pub fn register(registry: &mut ComponentRegistry) {
    registry.register_entity("@DISPLAY@", build);
}

/// Runs for every new instance, so each one gets its own component values.
fn build() -> EntityTemplate {
    // Shape: Empty, Cube, Sphere, Capsule or Plane. Add components with `.with(..)`,
    // for example `.with(crate::components::my_component::MyComponent::default())`.
    EntityTemplate::new(EntityShape::Cube)
}
"#;

#[cfg(test)]
mod checkout_tests {
    use std::path::{Path, PathBuf};

    use super::*;

    #[test]
    fn engine_path_is_not_the_project_manifest() {
        // External metadata: the engine name is a path dependency. The next
        // manifest_path is the game's own Cargo.toml (what the old scan used).
        let text = concat!(
            r#"{"packages":[{"name":"game","manifest_path":"/projects/LongDark/Cargo.toml","dependencies":["#,
            r#"{"name":"other","path":"/projects/LongDark/plugins/components"},"#,
            r#"{"name":"zestengine-engine","source":null,"path":"/repo/crates/zestengine-engine"}]}"#,
            r#"],"manifest_path":"/projects/LongDark/Cargo.toml"}"#
        );
        assert_eq!(engine_crate_dir_from_metadata(text), Some("/repo/crates/zestengine-engine"));
        assert_eq!(
            checkout_of_engine_crate(Path::new("/repo/crates/zestengine-engine")).unwrap(),
            PathBuf::from("/repo")
        );
    }

    #[test]
    fn engine_workspace_metadata_uses_the_same_path_object() {
        let text = concat!(
            r#"{"dependencies":[{"name":"zestengine-editor","path":"/repo/crates/zestengine-editor"},"#,
            r#"{"name":"zestengine-engine","source":null,"path":"/repo/crates/zestengine-engine"}],"targets":[]}"#,
            r#","manifest_path":"/repo/crates/zestengine/Cargo.toml""#
        );
        assert_eq!(engine_crate_dir_from_metadata(text), Some("/repo/crates/zestengine-engine"));
    }

    #[test]
    fn staged_manifest_points_at_the_external_lib_and_enables_editor() {
        let original = r#"[package]
name = "zestengine-project-components"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = []
editor = ["dep:zestengine-editor", "dep:egui"]

[dependencies]
zestengine-core.workspace = true
rand = "0.10.3"
"#;
        let lib_rs = Path::new("/projects/LongDark/plugins/components/src/lib.rs");
        let staged = manifest_built_in_engine(original, lib_rs);
        assert!(staged.contains("path = \"/projects/LongDark/plugins/components/src/lib.rs\""));
        assert!(staged.contains("default = [\"editor\"]"));
        assert!(staged.contains("rand = \"0.10.3\""));
        assert_eq!(
            plugin_lib_rs(Path::new("/projects/LongDark/plugins/components"), original),
            PathBuf::from("/projects/LongDark/plugins/components/src/lib.rs")
        );
    }
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

#[cfg(test)]
mod scaffold_tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zest-scaffold-{name}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn before(text: &str, first: &str, second: &str) -> bool {
        matches!((text.find(first), text.find(second)), (Some(a), Some(b)) if a < b)
    }

    #[test]
    fn names_are_derived_from_the_module() {
        assert_eq!(struct_name("new_component_1"), "NewComponent1");
        assert_eq!(display_name("new_component_1"), "New Component 1");
        assert_eq!(display_name("new_entity"), "New Entity");
    }

    #[test]
    fn new_projects_register_components_systems_and_entities_in_lib_rs() {
        let root = scratch("new");
        create_template(&root).unwrap();
        let src = root.join("plugins/components/src");
        for starter in ["components/orbit.rs", "systems/orbit.rs", "entities/orbiter.rs"] {
            assert!(src.join(starter).is_file(), "{starter}");
        }

        let component = create_rust_item(&root, RustItem::Component).unwrap();
        let system = create_rust_item(&root, RustItem::System).unwrap();
        let entity = create_rust_item(&root, RustItem::Entity).unwrap();
        let second = create_rust_item(&root, RustItem::Component).unwrap();
        assert_eq!(component, src.join("components/new_component.rs"));
        assert_eq!(system, src.join("systems/new_system.rs"));
        assert_eq!(entity, src.join("entities/new_entity.rs"));
        assert_eq!(second, src.join("components/new_component_1.rs"), "a taken name gets a numeric suffix");

        let body = std::fs::read_to_string(&component).unwrap();
        assert!(body.contains("pub struct NewComponent {}") && body.contains("const NAME: &'static str = \"New Component\";"));
        assert!(std::fs::read_to_string(&second).unwrap().contains("pub struct NewComponent1 {}"));
        let body = std::fs::read_to_string(&system).unwrap();
        assert!(body.contains("registry.register_system(\"project::new_system\", Stage::Update, update);"));
        let body = std::fs::read_to_string(&entity).unwrap();
        assert!(body.contains("registry.register_entity(\"New Entity\", build);"));

        let lib = std::fs::read_to_string(src.join("lib.rs")).unwrap();
        for (line, marker) in [
            ("    pub mod new_component;", COMPONENT_MODULE_MARKER),
            ("    pub mod new_component_1;", COMPONENT_MODULE_MARKER),
            ("    pub mod new_system;", SYSTEM_MODULE_MARKER),
            ("    pub mod new_entity;", ENTITY_MODULE_MARKER),
            ("    components::new_component::register(registry);", REGISTER_MARKER),
            ("    components::new_component_1::register(registry);", REGISTER_MARKER),
            ("    systems::new_system::register(registry);", SYSTEM_REGISTER_MARKER),
            ("    entities::new_entity::register(registry);", ENTITY_REGISTER_MARKER),
            ("    components::new_component::register_editor(editor);", EDITOR_REGISTER_MARKER),
        ] {
            assert_eq!(lib.matches(line).count(), 1, "{line}");
            assert!(before(&lib, line, marker), "{line} goes before its marker so later items follow it");
            assert_eq!(lib.matches(marker).count(), 1, "markers survive for the next item: {marker}");
        }
        assert!(before(&lib, "components::orbit::register(registry);", "systems::orbit::register(registry);"));
        assert!(before(&lib, "systems::orbit::register(registry);", "entities::orbiter::register(registry);"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn projects_from_before_the_three_folder_layout_still_register_new_items() {
        let root = scratch("legacy");
        let src = root.join("plugins/components/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(root.join("plugins/components/Cargo.toml"), "[package]\nname = \"legacy\"\n").unwrap();
        std::fs::write(
            src.join("lib.rs"),
            "//! Old plugin.\n\nmod orbit;\n\nuse zestengine_core::ComponentRegistry;\n\npub fn zestengine_register(registry: &mut ComponentRegistry) {\n    registry.register_ecs::<orbit::Orbit>();\n    // zestengine: new components register here\n}\n",
        )
        .unwrap();

        let system = create_rust_item(&root, RustItem::System).unwrap();
        let component = create_rust_item(&root, RustItem::Component).unwrap();
        assert_eq!(system, src.join("new_system.rs"), "legacy plugins keep flat modules next to lib.rs");
        assert_eq!(component, src.join("new_component.rs"));
        let lib = std::fs::read_to_string(src.join("lib.rs")).unwrap();
        assert!(before(&lib, "//! Old plugin.", "mod new_system;"), "module lines follow the inner doc comment");
        assert!(lib.contains("mod new_component;"));
        assert!(before(&lib, "pub fn zestengine_register", "    new_system::register(registry);"), "no marker for systems: registered at the top of the function");
        assert!(before(&lib, "    new_component::register(registry);", REGISTER_MARKER));
        assert!(lib.contains("registry.register_ecs::<orbit::Orbit>();"), "existing registrations are untouched");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_lib_rs_without_a_register_function_is_rejected_before_any_file_is_written() {
        let root = scratch("broken");
        let src = root.join("plugins/components/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(root.join("plugins/components/Cargo.toml"), "[package]\nname = \"broken\"\n").unwrap();
        std::fs::write(src.join("lib.rs"), "//! nothing here\n").unwrap();
        let error = create_rust_item(&root, RustItem::Entity).unwrap_err();
        assert!(error.to_string().contains("zestengine_register"));
        assert!(!src.join("new_entity.rs").exists() && !src.join("entities").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
