//! User-generated content at runtime: load `.zestenginepak` worlds, avatars and
//! props into a running game without recompiling it.
//!
//! [`ContentManager`] is driven by `ctx.load_content` / `spawn_content` /
//! `spawn_avatar` / `enter_world` (see `zestengine_core::ContentCommand`). Each
//! request runs on a worker thread: resolve the source (content id, local
//! path, or URL through a [`ContentFetcher`]), validate the package against the
//! [`ContentLimits`] of its kind, extract it into the hash-keyed
//! [`ContentCache`], then decode models / textures and upload them on the
//! transfer queue exactly like the scene loader. The main thread installs the
//! result: worlds go through [`SceneFlow`]'s seamless switch, props and avatars
//! are appended to the current scene (tagged with an origin so they can be
//! unloaded). Progress and results come back to gameplay as
//! [`ContentEvent`]s. The design and safety model are in `docs/user-content.md`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use zestengine_assets::package::{self, ContentCache, ExtractOptions, ExtractedPackage, PackageReader};
use zestengine_assets::{ObjectSource, SceneFile};
use zestengine_core::{
    AvatarDescriptor, ComponentRegistry, ContentCommand, ContentEvent, ContentInfo, ContentKind, ContentPlacement,
    ContentRequest, ObjectId,
};
use zestengine_render::Renderer;
use glam::{Quat, Vec3};

use crate::resource_tracker::{ResourceSet, ResourceTracker};
use crate::scene::LoadedScene;
use crate::scene_loader::{LoadStage, LoadedSceneData, SceneFlow};
use crate::shaders::ShaderLibrary;

pub use zestengine_assets::package::{ContentLimits, PackageManifest, PackageStats};

/// Fetches remote packages. The engine ships a plain-HTTP fetcher; games that
/// need HTTPS (or a CDN / auth layer) install their own with
/// [`ContentManager::set_fetcher`].
pub trait ContentFetcher: Send + Sync {
    /// Download `url` into `dest`, failing past `max_bytes`.
    fn fetch(&self, url: &str, dest: &Path, max_bytes: u64, progress: &dyn Fn(u64, Option<u64>)) -> Result<(), String>;
}

/// `http://` downloads over `std::net` (no TLS client is compiled into the
/// engine; `https://` needs a custom fetcher).
pub struct HttpFetcher;

impl ContentFetcher for HttpFetcher {
    fn fetch(&self, url: &str, dest: &Path, max_bytes: u64, progress: &dyn Fn(u64, Option<u64>)) -> Result<(), String> {
        if url.starts_with("https://") {
            return Err("https:// needs a TLS-capable fetcher: install one with ContentManager::set_fetcher \
                        (the engine has no TLS client compiled in)"
                .into());
        }
        let mut url = url.to_owned();
        for _ in 0..4 {
            match http_get(&url, dest, max_bytes, progress)? {
                HttpOutcome::Done => return Ok(()),
                HttpOutcome::Redirect(next) => url = next,
            }
        }
        Err("too many redirects".into())
    }
}

enum HttpOutcome {
    Done,
    Redirect(String),
}

fn http_get(url: &str, dest: &Path, max_bytes: u64, progress: &dyn Fn(u64, Option<u64>)) -> Result<HttpOutcome, String> {
    use std::io::{BufRead, BufReader, Read, Write};
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("unsupported URL {url:?}"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let addr = if authority.contains(':') { authority.to_owned() } else { format!("{authority}:80") };
    let host = authority.split(':').next().unwrap_or(authority);
    let sock = std::net::ToSocketAddrs::to_socket_addrs(&addr)
        .map_err(|e| format!("resolving {authority}: {e}"))?
        .next()
        .ok_or_else(|| format!("no address for {authority}"))?;
    let timeout = std::time::Duration::from_secs(20);
    let mut stream = std::net::TcpStream::connect_timeout(&sock, timeout).map_err(|e| format!("connecting {authority}: {e}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    write!(stream, "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: zestengine-content\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status).map_err(|e| e.to_string())?;
    let code: u16 = status.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or("malformed HTTP response")?;
    let (mut length, mut chunked, mut location) = (None, false, None);
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            match k.as_str() {
                "content-length" => length = v.parse::<u64>().ok(),
                "transfer-encoding" => chunked = v.eq_ignore_ascii_case("chunked"),
                "location" => location = Some(v.to_owned()),
                _ => {}
            }
        }
    }
    if matches!(code, 301 | 302 | 303 | 307 | 308) {
        let loc = location.ok_or("redirect without a location")?;
        let next = if loc.starts_with("http://") || loc.starts_with("https://") {
            loc
        } else {
            format!("http://{authority}{}", if loc.starts_with('/') { loc } else { format!("/{loc}") })
        };
        return Ok(HttpOutcome::Redirect(next));
    }
    if code != 200 {
        return Err(format!("HTTP {code} for {url}"));
    }
    if length.is_some_and(|l| l > max_bytes) {
        return Err(format!("download is {} bytes (limit {max_bytes})", length.unwrap_or(0)));
    }
    let mut out = std::fs::File::create(dest).map_err(|e| format!("creating {}: {e}", dest.display()))?;
    let mut total = 0u64;
    let mut buf = vec![0u8; 64 * 1024];
    let mut copy = |reader: &mut dyn Read, limit: Option<u64>, out: &mut std::fs::File| -> Result<(), String> {
        let mut left = limit;
        loop {
            let want = left.map_or(buf.len(), |l| (l as usize).min(buf.len()));
            if want == 0 {
                return Ok(());
            }
            let n = reader.read(&mut buf[..want]).map_err(|e| e.to_string())?;
            if n == 0 {
                return if left.is_some_and(|l| l > 0) { Err("connection closed early".into()) } else { Ok(()) };
            }
            total += n as u64;
            if total > max_bytes {
                return Err(format!("download exceeds {max_bytes} bytes"));
            }
            out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            left = left.map(|l| l - n as u64);
            progress(total, length);
        }
    };
    if chunked {
        loop {
            let mut size_line = String::new();
            reader.read_line(&mut size_line).map_err(|e| e.to_string())?;
            let size = u64::from_str_radix(size_line.trim().split(';').next().unwrap_or("0"), 16)
                .map_err(|_| "malformed chunked encoding")?;
            if size == 0 {
                break;
            }
            copy(&mut reader, Some(size), &mut out)?;
            let mut crlf = String::new();
            reader.read_line(&mut crlf).map_err(|e| e.to_string())?;
        }
    } else {
        copy(&mut reader, length, &mut out)?;
    }
    Ok(HttpOutcome::Done)
}

/// Default per-user content cache: `<data dir>/<game title>/content`.
pub fn content_cache_dir(title: &str) -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").or_else(|| std::env::var_os("APPDATA")).map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    let name: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    base.unwrap_or_else(|| PathBuf::from(".")).join(name).join("content")
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

fn is_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

// -------------------------------------------------------------------- worker

/// What a worker job hands back to the main thread.
pub struct LoadedContent {
    pub extracted: ExtractedPackage,
    pub data: LoadedSceneData,
    /// Measured (not claimed) resource use.
    pub stats: PackageStats,
}

#[derive(Clone)]
struct WorkerCtx {
    cache: ContentCache,
    fetcher: Arc<dyn ContentFetcher>,
    limits: Arc<dyn Fn(ContentKind) -> ContentLimits + Send + Sync>,
    root: PathBuf,
    known: Arc<HashSet<String>>,
    fingerprint: String,
}

enum WorkerMsg {
    Progress(String),
    Done(Box<Result<LoadedContent, String>>),
}

/// Resolve, validate, extract and decode one package. Runs off the main
/// thread; `uploader` (when the GPU has a transfer queue) uploads meshes and
/// textures here too. Public for headless tools and tests.
#[allow(clippy::too_many_arguments)]
pub fn load_package_on_worker(
    source: &str,
    ctx_cache: &ContentCache,
    fetcher: &dyn ContentFetcher,
    limits: &dyn Fn(ContentKind) -> ContentLimits,
    root: &Path,
    known: &dyn Fn(&str) -> bool,
    fingerprint: &str,
    expect: &[ContentKind],
    uploader: Option<&zestengine_render::Uploader>,
    extra_models: &[String],
    status: &dyn Fn(&str),
) -> Result<LoadedContent, String> {
    let err = |e: anyhow::Error| format!("{e:#}");
    let extracted = if is_hash(source) {
        ctx_cache
            .lookup(source, fingerprint)
            .ok_or_else(|| format!("content {} is not in the cache", &source[..12]))?
    } else {
        let cached = is_url(source)
            .then(|| ctx_cache.lookup_source(source))
            .flatten()
            .and_then(|h| ctx_cache.lookup(&h, fingerprint));
        match cached {
            Some(ex) => ex,
            None => {
                let path = if is_url(source) {
                    status("Downloading…");
                    let dir = ctx_cache.downloads_dir();
                    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                    let dest = dir.join(format!("{}.{}", &package::sha256_hex(source.as_bytes())[..24], package::PACKAGE_EXTENSION));
                    let max = ContentLimits::outer().max_package_bytes;
                    fetcher.fetch(source, &dest, max, &|done, total| {
                        if let Some(t) = total.filter(|t| *t > 0) {
                            status(&format!("Downloading… {}%", done * 100 / t));
                        }
                    })?;
                    dest
                } else {
                    let p = PathBuf::from(source);
                    if p.is_absolute() || p.exists() { p } else { root.join(source) }
                };
                status("Validating…");
                let mut reader = PackageReader::open(&path, limits).map_err(err)?;
                check_kind(reader.manifest.kind, expect)?;
                status("Unpacking…");
                let ex = ctx_cache
                    .extract(
                        &mut reader,
                        &ExtractOptions {
                            known_component: known,
                            components_fingerprint: fingerprint.to_owned(),
                            server_side: zestengine_core::process_role() == zestengine_core::NetRole::Server,
                        },
                    )
                    .map_err(err)?;
                if is_url(source) {
                    ctx_cache.remember_source(source, &ex.manifest.content_hash);
                    let _ = std::fs::remove_file(&path);
                }
                ex
            }
        }
    };
    check_kind(extracted.manifest.kind, expect)?;
    let kind_limits = limits(extracted.manifest.kind);
    // Everything under the package is untrusted: its WASM modules get the
    // sandbox's tight limits, no `invoke`, and expose only ugc_safe actions
    // (when the content limits allow module actions at all).
    zestengine_core::wasm::register_untrusted_root(
        &extracted.dir.join("files"),
        zestengine_core::wasm::UntrustedPolicy {
            limits: zestengine_core::wasm::SandboxLimits {
                max_module_bytes: kind_limits.max_wasm_bytes.min(usize::MAX as u64) as usize,
                ..zestengine_core::wasm::SandboxLimits::untrusted()
            },
            allow_actions: kind_limits.allow_wasm_actions,
        },
    );
    for c in &extracted.manifest.components {
        if !known(c) {
            tracing::warn!("content {}: component {c:?} is not available in this game", extracted.manifest.short_id());
        }
    }

    let measured = std::sync::Mutex::new(PackageStats::default());
    let check = |stage: LoadStage| -> Result<(), String> {
        let mut m = measured.lock().unwrap();
        match stage {
            LoadStage::Models(file, models) => {
                m.triangles = package::count_triangles(file, models);
                m.objects = file.entries.len() as u32;
                m.texture_bytes = models.values().map(package::model_texture_bytes).sum();
                (m.materials, m.bones, m.blendshapes) = package::model_complexity(models.values());
                // Material textures are estimated from their headers here so a
                // over-budget package fails before anything is uploaded.
                let mut est = m.clone();
                for (abs, _) in LoadedScene::collect_material_texture_refs(file, root) {
                    if let Some((w, h, bytes)) = package::estimate_texture_bytes(&abs) {
                        if w.max(h) > kind_limits.max_texture_dim {
                            return Err(format!(
                                "texture {} is {w}x{h} (limit {})",
                                abs.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                                kind_limits.max_texture_dim
                            ));
                        }
                        est.texture_bytes += bytes;
                    }
                }
                kind_limits.check_stats(&est)
            }
            LoadStage::Textures(textures) => {
                for t in textures {
                    if t.width.max(t.height) > kind_limits.max_texture_dim {
                        return Err(format!("a {}x{} texture exceeds {}", t.width, t.height, kind_limits.max_texture_dim));
                    }
                    m.texture_bytes += t.mips.iter().map(|x| x.len() as u64).sum::<u64>();
                }
                kind_limits.check_stats(&m)
            }
        }
    };
    let extra: &[String] = if extracted.manifest.kind == ContentKind::World { extra_models } else { &[] };
    let data = crate::scene_loader::load_on_worker_checked(&extracted.entry, root, uploader, extra, status, &check)?;
    let mut stats = measured.into_inner().unwrap();
    stats.file_bytes = extracted.manifest.stats.file_bytes;
    stats.shaders = data.shaders.len() as u32;
    if data.shaders.len() > kind_limits.max_shaders {
        return Err(format!("{} custom shaders (limit {})", data.shaders.len(), kind_limits.max_shaders));
    }
    for (rel, compiled) in &data.shaders {
        if let Err(e) = compiled {
            tracing::warn!("content shader {rel} failed to compile (falls back to standard): {e}");
        }
    }
    Ok(LoadedContent { extracted, data, stats })
}

fn check_kind(kind: ContentKind, expect: &[ContentKind]) -> Result<(), String> {
    if expect.is_empty() || expect.contains(&kind) {
        Ok(())
    } else {
        let want: Vec<&str> = expect.iter().map(|k| k.label()).collect();
        Err(format!("package is a {} but a {} was requested", kind.label(), want.join(" / ")))
    }
}

// ------------------------------------------------------------------- manager

/// What to do once a package is loaded.
#[derive(Clone, Debug)]
enum After {
    Nothing,
    Spawn { root_id: ObjectId, placement: ContentPlacement },
    EnterWorld,
}

struct Job {
    source: String,
    followers: Vec<(ContentRequest, After)>,
    rx: Receiver<WorkerMsg>,
}

struct Package {
    info: ContentInfo,
    extracted: ExtractedPackage,
    stats: PackageStats,
    /// The installed scene (absolute asset paths), the template for spawns.
    file: SceneFile,
    /// Worker output not installed yet (consumed by the first spawn / switch).
    pending: Option<LoadedSceneData>,
    /// Models / textures live in the current scene's renderer resources.
    installed: bool,
    /// Origins of the live instances.
    instances: Vec<String>,
}

/// Scene cache entries a content origin added; removed again when it unloads
/// so a later spawn re-installs fresh resources instead of reusing freed ones.
#[derive(Default)]
struct OriginCaches {
    models: Vec<String>,
    material_files: Vec<PathBuf>,
    texture_files: Vec<(PathBuf, bool)>,
}

/// Host access for [`ContentManager::update`].
pub struct ContentHost<'a> {
    /// None on a dedicated server: packages load CPU-side (collision, skeletons, scripts).
    pub renderer: Option<&'a mut Renderer>,
    pub registry: &'a ComponentRegistry,
    pub shaders: &'a mut ShaderLibrary,
    pub root: &'a Path,
    pub scene: &'a mut LoadedScene,
    pub flow: &'a mut SceneFlow,
}

/// Result of one [`ContentManager::update`].
#[derive(Default)]
pub struct ContentUpdate {
    /// Deliver to gameplay (`PlaySession::post_event`).
    pub events: Vec<ContentEvent>,
    /// Objects were added / removed (tell physics / nav).
    pub structure_changed: bool,
}

/// Loads, spawns and unloads user-generated content packages at runtime.
pub struct ContentManager {
    cache: ContentCache,
    fetcher: Arc<dyn ContentFetcher>,
    limits: Arc<dyn Fn(ContentKind) -> ContentLimits + Send + Sync>,
    queued: Vec<ContentCommand>,
    jobs: Vec<Job>,
    packages: HashMap<String, Package>,
    /// Source (path / URL) -> content id.
    sources: HashMap<String, String>,
    events: Vec<ContentEvent>,
    /// Player object -> (origin, root) of its avatar instance.
    avatars: HashMap<ObjectId, (String, ObjectId)>,
    /// Player object -> content id of the avatar it wears.
    worn: HashMap<ObjectId, String>,
    /// Remote players whose replicated avatar was requested: player -> content id.
    remote_requests: HashMap<ObjectId, String>,
    /// World switch requested through SceneFlow: (content id, request).
    pending_world: Option<(String, ContentRequest)>,
    /// Renderer resource indices each live origin references.
    resources: ResourceTracker,
    /// Every renderer index allocated by a content spawn (safe-to-free set is
    /// the intersection of `free_set_for` with this, so base-scene resources
    /// are never freed).
    content_owned: ResourceSet,
    /// Scene cache entries each origin installed.
    origin_caches: HashMap<String, OriginCaches>,
    counter: u64,
}

impl ContentManager {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache: ContentCache::new(cache_dir),
            fetcher: Arc::new(HttpFetcher),
            limits: Arc::new(ContentLimits::for_kind),
            queued: Vec::new(),
            jobs: Vec::new(),
            packages: HashMap::new(),
            sources: HashMap::new(),
            events: Vec::new(),
            avatars: HashMap::new(),
            worn: HashMap::new(),
            remote_requests: HashMap::new(),
            pending_world: None,
            resources: ResourceTracker::new(),
            content_owned: ResourceSet::default(),
            origin_caches: HashMap::new(),
            counter: 0,
        }
    }

    /// [`Self::new`] plus loads queued for `sources` (`;`-separated paths /
    /// URLs, e.g. the `ZESTENGINE_CONTENT` environment variable).
    pub fn with_startup_sources(cache_dir: impl Into<PathBuf>, sources: Option<String>) -> Self {
        let mut m = Self::new(cache_dir);
        for s in sources.iter().flat_map(|s| s.split(';')).map(str::trim).filter(|s| !s.is_empty()) {
            m.queue(ContentCommand::Load { request: ContentRequest::next(), source: s.to_owned() });
        }
        m
    }

    pub fn cache(&self) -> &ContentCache {
        &self.cache
    }

    /// Replace the download backend (HTTPS client, CDN, authentication).
    pub fn set_fetcher(&mut self, fetcher: Arc<dyn ContentFetcher>) {
        self.fetcher = fetcher;
    }

    /// Override the per-kind resource limits.
    pub fn set_limits(&mut self, limits: impl Fn(ContentKind) -> ContentLimits + Send + Sync + 'static) {
        self.limits = Arc::new(limits);
    }

    /// Queue a gameplay request; applied by the next [`Self::update`].
    pub fn queue(&mut self, command: ContentCommand) {
        self.queued.push(command);
    }

    /// Loaded packages (content id, info, live instance count).
    pub fn loaded(&self) -> Vec<(String, ContentInfo, usize)> {
        let mut v: Vec<_> = self
            .packages
            .iter()
            .map(|(id, p)| (id.clone(), p.info.clone(), p.instances.len()))
            .collect();
        v.sort_by(|a, b| a.1.name.cmp(&b.1.name));
        v
    }

    pub fn stats(&self, id: &str) -> Option<&PackageStats> {
        self.packages.get(id).map(|p| &p.stats)
    }

    /// Requests still downloading / decoding.
    pub fn busy(&self) -> usize {
        self.jobs.len()
    }

    fn resolve(&self, id_or_source: &str) -> Option<String> {
        if self.packages.contains_key(id_or_source) {
            return Some(id_or_source.to_owned());
        }
        self.sources.get(id_or_source).cloned()
    }

    /// The scene was replaced (any switch): instances and installed GPU
    /// resources are gone. Call with the new scene's path after a switch.
    pub fn scene_replaced(&mut self, path: &str) {
        for p in self.packages.values_mut() {
            p.installed = false;
            p.instances.clear();
        }
        self.avatars.clear();
        self.worn.clear();
        self.remote_requests.clear();
        self.resources = ResourceTracker::new();
        self.content_owned = ResourceSet::default();
        self.origin_caches.clear();
        if let Some((id, request)) = self.pending_world.take() {
            if let Some(p) = self.packages.get_mut(&id)
                && p.extracted.entry.to_string_lossy() == path
            {
                p.installed = true;
                self.events.push(ContentEvent::WorldEntered { request, info: p.info.clone() });
            } else {
                self.pending_world = None;
            }
        }
    }

    /// Process queued requests, finished worker jobs and pending installs.
    pub fn update(&mut self, host: &mut ContentHost) -> ContentUpdate {
        let mut out = ContentUpdate::default();
        self.follow_remote_avatars(host.scene);
        for command in std::mem::take(&mut self.queued) {
            self.apply(command, host, &mut out);
        }
        let mut i = 0;
        while i < self.jobs.len() {
            let mut finished = None;
            while let Ok(msg) = self.jobs[i].rx.try_recv() {
                match msg {
                    WorkerMsg::Progress(stage) => {
                        for (request, _) in &self.jobs[i].followers {
                            out.events.push(ContentEvent::Progress { request: *request, stage: stage.clone() });
                        }
                    }
                    WorkerMsg::Done(r) => {
                        finished = Some(r);
                        break;
                    }
                }
            }
            match finished {
                Some(result) => {
                    let job = self.jobs.remove(i);
                    self.finish_job(job, *result, host, &mut out);
                }
                None => i += 1,
            }
        }
        out.events.append(&mut self.events);
        out
    }

    fn apply(&mut self, command: ContentCommand, host: &mut ContentHost, out: &mut ContentUpdate) {
        match command {
            ContentCommand::Load { request, source } => match self.resolve(&source) {
                Some(id) => out.events.push(ContentEvent::Loaded { request, info: self.packages[&id].info.clone() }),
                None => self.start_job(request, &source, After::Nothing, host),
            },
            ContentCommand::Spawn { request, source, root_id, placement } => {
                let after = After::Spawn { root_id, placement };
                match self.resolve(&source) {
                    Some(id) if self.packages[&id].installed || self.packages[&id].pending.is_some() => {
                        self.run_after(&id, request, after, host, out)
                    }
                    Some(id) => self.start_job(request, &id, after, host),
                    None => self.start_job(request, &source, after, host),
                }
            }
            ContentCommand::EnterWorld { request, source } => match self.resolve(&source) {
                Some(id) if self.packages[&id].pending.is_some() => self.run_after(&id, request, After::EnterWorld, host, out),
                Some(id) => self.start_job(request, &id, After::EnterWorld, host),
                None => self.start_job(request, &source, After::EnterWorld, host),
            },
            ContentCommand::Unload(id) => {
                if let Some(id) = self.resolve(&id) {
                    self.unload(&id, host, out);
                }
            }
        }
    }

    fn start_job(&mut self, request: ContentRequest, source: &str, after: After, host: &mut ContentHost) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.source == source) {
            job.followers.push((request, after));
            return;
        }
        let expect: Vec<ContentKind> = match &after {
            After::Nothing => Vec::new(),
            After::EnterWorld => vec![ContentKind::World],
            After::Spawn { placement: ContentPlacement::AvatarOf(_), .. } => vec![ContentKind::Avatar],
            After::Spawn { .. } => vec![ContentKind::Prop, ContentKind::Avatar],
        };
        let known: HashSet<String> = host.registry.names().map(str::to_owned).collect();
        let fingerprint = package::components_fingerprint(known.iter().map(String::as_str));
        let ctx = WorkerCtx {
            cache: self.cache.clone(),
            fetcher: self.fetcher.clone(),
            limits: self.limits.clone(),
            root: host.root.to_owned(),
            known: Arc::new(known),
            fingerprint,
        };
        let extra = match after {
            After::EnterWorld => host.flow.kept_model_paths(host.scene, host.root, host.shaders),
            _ => Vec::new(),
        };
        let uploader = host.renderer.as_deref().and_then(Renderer::uploader);
        let (tx, rx) = channel();
        let src = source.to_owned();
        std::thread::Builder::new()
            .name("zestengine-content".into())
            .spawn(move || run_worker(src, ctx, expect, uploader, extra, tx))
            .ok();
        self.jobs.push(Job {
            source: source.to_owned(),
            followers: vec![(request, after)],
            rx,
        });
    }

    fn finish_job(&mut self, job: Job, result: Result<LoadedContent, String>, host: &mut ContentHost, out: &mut ContentUpdate) {
        let loaded = match result {
            Ok(l) => l,
            Err(error) => {
                tracing::error!("content {}: {error}", job.source);
                for (request, _) in job.followers {
                    out.events.push(ContentEvent::Failed { request, source: job.source.clone(), error: error.clone() });
                }
                return;
            }
        };
        let m = &loaded.extracted.manifest;
        let id = m.content_hash.clone();
        let info = ContentInfo {
            id: id.clone(),
            kind: m.kind,
            name: m.name.clone(),
            author: m.author.clone(),
            version: m.version.clone(),
            warnings: loaded.extracted.warnings.clone(),
        };
        tracing::info!(
            "content loaded: {:?} by {:?} ({:?}, {} tris, {:.1} MiB textures)",
            info.name,
            info.author,
            info.kind,
            loaded.stats.triangles,
            loaded.stats.texture_bytes as f64 / (1024.0 * 1024.0)
        );
        if !is_hash(&job.source) {
            self.sources.insert(job.source.clone(), id.clone());
        }
        match self.packages.get_mut(&id) {
            Some(p) if p.installed || p.pending.is_some() => {
                // Already usable (a parallel request won); this upload is surplus.
                tracing::debug!("content {}: duplicate load ignored", m.short_id());
            }
            Some(p) => {
                p.file = loaded.data.file.clone();
                p.pending = Some(loaded.data);
            }
            None => {
                self.packages.insert(
                    id.clone(),
                    Package {
                        info: info.clone(),
                        file: loaded.data.file.clone(),
                        extracted: loaded.extracted,
                        stats: loaded.stats,
                        pending: Some(loaded.data),
                        installed: false,
                        instances: Vec::new(),
                    },
                );
            }
        }
        for (request, after) in job.followers {
            out.events.push(ContentEvent::Loaded { request, info: info.clone() });
            self.run_after(&id, request, after, host, out);
        }
    }

    fn run_after(&mut self, id: &str, request: ContentRequest, after: After, host: &mut ContentHost, out: &mut ContentUpdate) {
        let info = self.packages[id].info.clone();
        let fail = |out: &mut ContentUpdate, error: String| {
            out.events.push(ContentEvent::Failed { request, source: id.to_owned(), error });
        };
        match after {
            After::Nothing => {}
            After::EnterWorld => {
                if info.kind != ContentKind::World {
                    return fail(out, format!("{} is a {}, not a World", info.name, info.kind.label()));
                }
                let p = self.packages.get_mut(id).unwrap();
                match p.pending.take() {
                    Some(data) => {
                        let path = p.extracted.entry.to_string_lossy().into_owned();
                        host.flow.switch_to_loaded(&path, data);
                        self.pending_world = Some((id.to_owned(), request));
                    }
                    None => self.start_job(request, id, After::EnterWorld, host),
                }
            }
            After::Spawn { root_id, placement } => {
                if info.kind == ContentKind::World {
                    return fail(out, format!("{} is a World; use enter_world", info.name));
                }
                if matches!(placement, ContentPlacement::AvatarOf(_)) && info.kind != ContentKind::Avatar {
                    return fail(out, format!("{} is a {}, not an Avatar", info.name, info.kind.label()));
                }
                match self.spawn(id, root_id, &placement, host) {
                    Ok(true) => {
                        out.structure_changed = true;
                        out.events.push(ContentEvent::Spawned { request, info, root: root_id });
                    }
                    Ok(false) => self.start_job(request, id, After::Spawn { root_id, placement }, host),
                    Err(e) => fail(out, format!("{e:#}")),
                }
            }
        }
    }

    /// Add one instance. Ok(false) when the package must be decoded again
    /// (its GPU data went with a previous scene).
    fn spawn(&mut self, id: &str, root_id: ObjectId, placement: &ContentPlacement, host: &mut ContentHost) -> anyhow::Result<bool> {
        let p = self.packages.get_mut(id).unwrap();
        if !p.installed && p.pending.is_none() {
            return Ok(false);
        }
        self.counter += 1;
        let origin = format!("content:{}:{}", &id[..12], self.counter);
        let mut entries = remap_ids(&p.file, root_id);
        let root_local = entries.iter().position(|e| e.parent.is_none()).unwrap_or(0);
        match placement {
            ContentPlacement::World { translation, rotation } => {
                entries[root_local].translation = translation.to_array();
                entries[root_local].rotation = rotation.to_array();
            }
            ContentPlacement::AvatarOf(_) => {
                entries[root_local].translation = [0.0; 3];
                entries[root_local].rotation = Quat::IDENTITY.to_array();
            }
        }
        let counts = |h: &ContentHost| h.renderer.as_deref().map_or((0, 0, 0), |r| (r.mesh_count(), r.texture_count(), r.material_count()));
        let (mesh_base, texture_base, material_base) = counts(host);
        let candidate_models: Vec<String> = match &p.pending {
            Some(data) => data.models.keys().cloned().collect(),
            None => p.file.model_paths(),
        };
        let new_models: Vec<String> = candidate_models
            .into_iter()
            .filter(|path| !host.scene.has_model_template(path))
            .collect();
        let material_cache_before: HashSet<PathBuf> = host.scene.material_cache_keys().into_iter().collect();
        let texture_cache_before: HashSet<(PathBuf, bool)> = host.scene.texture_cache_keys().into_iter().collect();

        match p.pending.take() {
            Some(mut data) => {
                data.file.entries = entries;
                host.scene.append_loaded(host.renderer.as_deref_mut(), host.shaders, host.root, host.registry, data, &origin)?;
            }
            None => {
                let first = host.scene.append_entries_with(host.renderer.as_deref_mut(), host.shaders, host.root, host.registry, &entries)?;
                let ids: Vec<ObjectId> = host.scene.objects[first..].iter().map(|o| o.id).collect();
                for oid in ids {
                    host.scene.object_origin.insert(oid, origin.clone());
                }
            }
        }
        p.installed = true;
        p.instances.push(origin.clone());

        // Record what this origin references and which renderer indices it
        // newly allocated, so `free_origin` can release them later.
        let (mesh_end, texture_end, material_end) = counts(host);
        let newly = ResourceSet {
            meshes: (mesh_base..mesh_end).collect(),
            textures: (texture_base..texture_end).collect(),
            materials: (material_base..material_end).collect(),
        };
        let mut set = Self::resources_for(host.scene, &origin);
        set.meshes.extend(&newly.meshes);
        set.textures.extend(&newly.textures);
        set.materials.extend(&newly.materials);
        self.content_owned.meshes.extend(&newly.meshes);
        self.content_owned.textures.extend(&newly.textures);
        self.content_owned.materials.extend(&newly.materials);
        self.resources.record(&origin, set);

        let material_cache_after: HashSet<PathBuf> = host.scene.material_cache_keys().into_iter().collect();
        let texture_cache_after: HashSet<(PathBuf, bool)> = host.scene.texture_cache_keys().into_iter().collect();
        self.origin_caches.insert(
            origin.clone(),
            OriginCaches {
                models: new_models,
                material_files: material_cache_after.difference(&material_cache_before).cloned().collect(),
                texture_files: texture_cache_after.difference(&texture_cache_before).cloned().collect(),
            },
        );

        if let ContentPlacement::AvatarOf(player) = placement {
            if host.scene.index_of(*player).is_none() {
                if let Some(pkg) = self.packages.get_mut(id) {
                    pkg.instances.retain(|o| *o != origin);
                }
                self.free_origin(&origin, host.renderer.as_deref_mut(), host.scene);
                anyhow::bail!("player object not found");
            }
            if let Some((old, _)) = self.avatars.remove(player) {
                self.free_origin(&old, host.renderer.as_deref_mut(), host.scene);
                for pk in self.packages.values_mut() {
                    pk.instances.retain(|o| *o != old);
                }
            }
            let (Some(player_index), Some(root_index)) = (host.scene.index_of(*player), host.scene.index_of(root_id)) else {
                anyhow::bail!("avatar root vanished while attaching");
            };
            host.scene.objects[root_index].parent = Some(player_index);
            attach_avatar(host.scene, player_index, root_index);
            self.avatars.insert(*player, (origin, root_id));
            self.worn.insert(*player, id.to_owned());
            let url = self
                .sources
                .iter()
                .find(|(src, v)| v.as_str() == id && (src.starts_with("http://") || src.starts_with("https://")))
                .map(|(src, _)| src.clone())
                .unwrap_or_default();
            if let Some(av) = host.scene.objects[player_index]
                .components
                .iter_mut()
                .find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::NetworkAvatar>())
                && !av.remote
            {
                av.wearing = Some(zestengine_core::WornAvatar { id: id.to_owned(), url });
            }
        }
        Ok(true)
    }

    /// Wear the avatar each remote player announced on their Network Avatar
    /// (once per package; a failed load isn't retried until they change it).
    fn follow_remote_avatars(&mut self, scene: &LoadedScene) {
        for o in &scene.objects {
            let Some(av) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::NetworkAvatar>()) else {
                continue;
            };
            let Some(w) = av.wearing.as_ref().filter(|_| av.remote) else { continue };
            if self.worn.get(&o.id) == Some(&w.id) || self.remote_requests.get(&o.id) == Some(&w.id) {
                continue;
            }
            self.remote_requests.insert(o.id, w.id.clone());
            let source = if self.packages.contains_key(&w.id) { w.id.clone() } else { w.source().to_owned() };
            self.queued.push(ContentCommand::Spawn {
                request: ContentRequest::next(),
                source,
                root_id: ObjectId::new(),
                placement: ContentPlacement::AvatarOf(o.id),
            });
        }
    }

    fn unload(&mut self, id: &str, host: &mut ContentHost, out: &mut ContentUpdate) {
        let Some(mut p) = self.packages.remove(id) else { return };
        if let Some(mut pending) = p.pending.take()
            && let Some(renderer) = host.renderer.as_deref_mut() {
                free_pending_gpu(&mut pending, renderer);
            }
        for origin in &p.instances {
            if self.free_origin(origin, host.renderer.as_deref_mut(), host.scene) > 0 {
                out.structure_changed = true;
            }
        }
        self.avatars.retain(|_, (o, _)| !p.instances.contains(o));
        self.worn.retain(|_, w| w != id);
        self.sources.retain(|_, v| v != id);
        out.events.push(ContentEvent::Unloaded { id: id.to_owned() });
    }

    /// The renderer resource indices an origin's live objects reference.
    fn resources_for(scene: &LoadedScene, origin: &str) -> ResourceSet {
        let mut set = ResourceSet::default();
        for object in &scene.objects {
            if scene.object_origin.get(&object.id).map(String::as_str) != Some(origin) {
                continue;
            }
            for render in object.render_slots() {
                if let Some(mesh) = scene.mesh_handle(render.mesh) {
                    set.meshes.insert(mesh.index());
                }
                if let Some(material) = scene.materials.get(render.material) {
                    set.materials.insert(material.handle.index());
                    for texture in material.bound_textures.iter().flatten() {
                        set.textures.insert(texture.index());
                    }
                }
            }
        }
        set
    }

    /// Remove an origin's objects and free the GPU resources it alone owns.
    /// Returns the number of objects removed.
    fn free_origin(&mut self, origin: &str, renderer: Option<&mut Renderer>, scene: &mut LoadedScene) -> usize {
        let removed = scene.unload_origin(origin);
        let referenced = self.resources.free_set_for(origin);
        let free = ResourceSet {
            meshes: referenced.meshes.intersection(&self.content_owned.meshes).copied().collect(),
            textures: referenced.textures.intersection(&self.content_owned.textures).copied().collect(),
            materials: referenced.materials.intersection(&self.content_owned.materials).copied().collect(),
        };
        if let Some(renderer) = renderer {
            let _ = renderer.free_resources(&free.meshes, &free.textures, &free.materials);
        }
        for mesh in &free.meshes {
            self.content_owned.meshes.remove(mesh);
        }
        for texture in &free.textures {
            self.content_owned.textures.remove(texture);
        }
        for material in &free.materials {
            self.content_owned.materials.remove(material);
        }
        self.resources.clear(origin);
        if let Some(caches) = self.origin_caches.remove(origin) {
            for path in caches.models {
                scene.remove_model_template(&path);
            }
            for path in caches.material_files {
                scene.remove_material_cache(&path);
            }
            for key in caches.texture_files {
                scene.remove_texture_cache(&key);
            }
        }
        scene.clear_lazy_caches();
        removed
    }

    /// Remove every instance from the scene and forget installed state (the
    /// editor calls this when Play stops). Packages stay loadable from cache.
        pub fn clear(&mut self) {
        self.packages.clear();
        self.sources.clear();
        self.avatars.clear();
        self.worn.clear();
        self.remote_requests.clear();
        self.pending_world = None;
        self.queued.clear();
        self.resources = ResourceTracker::new();
        self.content_owned = ResourceSet::default();
        self.origin_caches.clear();
    }
}

fn free_pending_gpu(data: &mut LoadedSceneData, renderer: &mut Renderer) {
    if let Some(prepared) = data.prepared.take() {
        for (_, scene) in prepared {
            renderer.free_prepared(scene);
        }
    }
    if let Some((_, prepared)) = data.material_textures.take() {
        renderer.free_prepared(prepared);
    }
}

fn run_worker(
    source: String,
    ctx: WorkerCtx,
    expect: Vec<ContentKind>,
    uploader: Option<zestengine_render::Uploader>,
    extra: Vec<String>,
    tx: Sender<WorkerMsg>,
) {
    crate::sw_gi::lower_compute_priority();
    let progress = tx.clone();
    let known = ctx.known.clone();
    let result = load_package_on_worker(
        &source,
        &ctx.cache,
        ctx.fetcher.as_ref(),
        ctx.limits.as_ref(),
        &ctx.root,
        &|k| known.contains(k),
        &ctx.fingerprint,
        &expect,
        uploader.as_ref(),
        &extra,
        &|s| {
            let _ = progress.send(WorkerMsg::Progress(s.to_owned()));
        },
    );
    let _ = tx.send(WorkerMsg::Done(Box::new(result)));
}

/// Copy of a package scene's entries with fresh object ids (the first root
/// gets `root_id`); id references inside component data follow.
fn remap_ids(file: &SceneFile, root_id: ObjectId) -> Vec<zestengine_assets::SceneEntry> {
    let root_index = file.entries.iter().position(|e| e.parent.is_none()).unwrap_or(0);
    let map: Vec<(String, String)> = file
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let new = if i == root_index { root_id } else { ObjectId::new() };
            (e.id.clone(), new.to_string())
        })
        .collect();
    file.entries
        .iter()
        .zip(&map)
        .map(|(e, (_, new))| {
            let mut e = e.clone();
            e.id = new.clone();
            for c in &mut e.components {
                for (old, new) in map.iter().filter(|(o, _)| !o.is_empty()) {
                    if c.data.contains(old.as_str()) {
                        c.data = c.data.replace(old.as_str(), new);
                    }
                }
            }
            e
        })
        .collect()
}

/// Fit the player to its new avatar: the player's camera moves to the
/// descriptor's eye point and its character capsule to the descriptor's body.
fn attach_avatar(scene: &mut LoadedScene, player: usize, avatar_root: usize) {
    // Skinned parts get full-body IK: the wearer's trackers locally, the
    // replicated trackers on other players' copies.
    let in_avatar = |scene: &LoadedScene, mut i: usize| loop {
        if i == avatar_root {
            return true;
        }
        match scene.objects[i].parent {
            Some(p) => i = p,
            None => return false,
        }
    };
    for i in 0..scene.objects.len() {
        if in_avatar(scene, i)
            && scene.has_skinned_mesh(i)
            && !scene.objects[i].components.iter().any(|c| c.as_any().is::<zestengine_core::HumanoidIk>())
        {
            scene.objects[i].components.push(Box::new(zestengine_core::HumanoidIk::default()));
        }
        if in_avatar(scene, i)
            && scene.has_blendshapes(i)
            && !scene.objects[i].components.iter().any(|c| c.as_any().is::<zestengine_core::FaceTracking>())
        {
            scene.objects[i].components.push(Box::new(zestengine_core::FaceTracking::default()));
        }
        if in_avatar(scene, i)
            && scene.has_blendshapes(i)
            && !scene.objects[i].components.iter().any(|c| c.as_any().is::<zestengine_core::LipSync>())
        {
            scene.objects[i].components.push(Box::new(zestengine_core::LipSync::default()));
        }
    }
    let Some(desc) = scene.objects[avatar_root]
        .components
        .iter()
        .find_map(|c| c.as_any().downcast_ref::<AvatarDescriptor>())
        .cloned()
    else {
        return;
    };
    let eye = Vec3::from(desc.view_position);
    let cameras: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| i != avatar_root && matches!(scene.objects[i].source, ObjectSource::Camera))
        .filter(|&i| {
            let mut p = scene.objects[i].parent;
            while let Some(pi) = p {
                if pi == player {
                    return true;
                }
                if pi == avatar_root {
                    return false;
                }
                p = scene.objects[pi].parent;
            }
            false
        })
        .collect();
    for c in cameras {
        if scene.objects[c].parent == Some(player) {
            scene.objects[c].translation = eye;
        }
    }
    if desc.collider_height > 0.0 && desc.collider_radius > 0.0 {
        for c in scene.objects[player].components.iter_mut() {
            if let Some(cc) = c.as_any_mut().downcast_mut::<zestengine_core::CharacterController>() {
                cc.height = desc.collider_height;
                cc.radius = desc.collider_radius;
                cc.center = [0.0, desc.collider_height * 0.5, 0.0];
            }
        }
    }
}

/// Humanoid / descriptor checks for an avatar about to be packaged.
pub fn avatar_warnings(root: &Path, file: &SceneFile) -> Vec<String> {
    let mut out = Vec::new();
    let desc = file
        .entries
        .iter()
        .find(|e| e.parent.is_none())
        .and_then(|e| e.components.iter().find(|c| c.kind == <AvatarDescriptor as zestengine_core::TypedComponent>::NAME))
        .and_then(|c| ron::from_str::<AvatarDescriptor>(&c.data).ok())
        .unwrap_or_default();
    if !(0.3..=4.0).contains(&desc.eye_height()) {
        out.push(format!("eye height {:.2} m looks wrong (expected 0.3..4 m)", desc.eye_height()));
    }
    let mut humanoid = false;
    let mut any_skeleton = false;
    let mut seen = HashSet::new();
    for e in &file.entries {
        if let ObjectSource::Model { path, .. } = &e.source
            && seen.insert(path.clone())
            && let Ok(model) = zestengine_assets::load_model_with_meta(root.join(path))
        {
            for skel in &model.skeletons {
                any_skeleton = true;
                let (rig, problems) = crate::humanoid::HumanoidRig::map_with_overrides(skel, &desc.bone_map);
                out.extend(problems);
                out.extend(crate::humanoid::rig_problems(skel).into_iter().map(|m| format!("rig: {m}")));
                humanoid |= rig.is_humanoid();
            }
        }
    }
    if !any_skeleton {
        out.push("no skinned skeleton found: the avatar won't animate or follow VR trackers".into());
    } else if !humanoid {
        out.push("skeleton is not recognised as humanoid (needs hips + head): add a bone map on the Avatar Descriptor for VR IK".into());
    }
    out
}

// ------------------------------------------------------------------- overlay

/// Developer overlay for shipped games (F8 when enabled): lists packages in
/// the content inbox folders and the loaded ones, with buttons to load,
/// spawn, wear, enter and unload. Draws with the game-UI painter and its own
/// hit testing (the runtime feeds egui no raw events).
#[cfg(feature = "client")]
#[derive(Default)]
pub struct ContentOverlay {
    pub open: bool,
    pub dirs: Vec<PathBuf>,
    files: Vec<PathBuf>,
    last_scan: Option<std::time::Instant>,
    log: Vec<String>,
}

/// What the overlay asks the host to do this frame.
#[cfg(feature = "client")]
pub struct OverlayInput {
    pub pointer: Option<egui::Pos2>,
    pub clicked: bool,
    /// Where props spawn (in front of the camera).
    pub spawn_at: (Vec3, Quat),
    /// Player object avatars attach to (the possessed Pawn), if any.
    pub player: Option<ObjectId>,
}

#[cfg(feature = "client")]
impl ContentOverlay {
    pub fn new(dirs: Vec<PathBuf>) -> Self {
        Self { dirs, ..Default::default() }
    }

    /// Record events for the log panel.
    pub fn note_events(&mut self, events: &[ContentEvent]) {
        for e in events {
            let line = match e {
                ContentEvent::Progress { stage, .. } => format!("… {stage}"),
                ContentEvent::Loaded { info, .. } => format!("loaded {} ({})", info.name, info.kind.label()),
                ContentEvent::Spawned { info, .. } => format!("spawned {}", info.name),
                ContentEvent::WorldEntered { info, .. } => format!("entered {}", info.name),
                ContentEvent::Failed { error, .. } => format!("FAILED: {error}"),
                ContentEvent::Unloaded { id } => format!("unloaded {}", &id[..12.min(id.len())]),
            };
            if let ContentEvent::Loaded { info, .. } = e {
                for w in &info.warnings {
                    self.log.push(format!("  warning: {w}"));
                }
            }
            self.log.push(line);
        }
        let excess = self.log.len().saturating_sub(10);
        self.log.drain(..excess);
    }

    fn rescan(&mut self) {
        if self.last_scan.is_some_and(|t| t.elapsed().as_secs_f32() < 2.0) {
            return;
        }
        self.last_scan = Some(std::time::Instant::now());
        self.files.clear();
        for d in &self.dirs {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().and_then(|x| x.to_str()) == Some(package::PACKAGE_EXTENSION) {
                        self.files.push(p);
                    }
                }
            }
        }
        self.files.sort();
    }

    /// Draw the panel; returns true while the pointer is over it.
    pub fn show(&mut self, ctx: &egui::Context, input: &OverlayInput, manager: &mut ContentManager) -> bool {
        if !self.open {
            return false;
        }
        self.rescan();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("zestengine-content-overlay")));
        let font = egui::FontId::proportional(14.0);
        let small = egui::FontId::proportional(12.0);
        let width = 520.0;
        let row = 24.0;
        let mut y = 16.0;
        let x = 16.0;
        let rows = 4 + self.files.len().max(1) + manager.loaded().len().max(1) + self.log.len();
        let panel = egui::Rect::from_min_size(egui::pos2(x - 8.0, y - 8.0), egui::vec2(width + 16.0, rows as f32 * row + 40.0));
        painter.rect_filled(panel, 6.0, egui::Color32::from_rgba_unmultiplied(16, 16, 22, 230));
        let text = |y: f32, s: &str, f: &egui::FontId, c: egui::Color32| {
            painter.text(egui::pos2(x, y), egui::Align2::LEFT_TOP, s, f.clone(), c);
        };
        let button = |bx: f32, y: f32, label: &str| -> bool {
            let rect = egui::Rect::from_min_size(egui::pos2(bx, y - 2.0), egui::vec2(72.0, row - 4.0));
            let hover = input.pointer.is_some_and(|p| rect.contains(p));
            let fill = if hover { egui::Color32::from_rgb(90, 80, 150) } else { egui::Color32::from_rgb(52, 48, 80) };
            painter.rect_filled(rect, 4.0, fill);
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, label, small.clone(), egui::Color32::WHITE);
            hover && input.clicked
        };
        let busy = manager.busy();
        text(
            y,
            &format!("User content (F8){}", if busy > 0 { format!("  -  {busy} loading") } else { String::new() }),
            &font,
            egui::Color32::from_rgb(250, 200, 90),
        );
        y += row;
        text(y, "Inbox", &small, egui::Color32::GRAY);
        y += row;
        if self.files.is_empty() {
            let dirs: Vec<String> = self.dirs.iter().map(|d| d.display().to_string()).collect();
            text(y, &format!("drop .zestenginepak files into {}", dirs.join(" or ")), &small, egui::Color32::GRAY);
            y += row;
        }
        let mut commands = Vec::new();
        for f in self.files.clone() {
            let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            text(y, &name, &small, egui::Color32::WHITE);
            let src = f.to_string_lossy().into_owned();
            let bx = x + width - 3.0 * 78.0;
            if button(bx, y, "Spawn") {
                let (t, r) = input.spawn_at;
                commands.push(ContentCommand::Spawn {
                    request: ContentRequest::next(),
                    source: src.clone(),
                    root_id: ObjectId::new(),
                    placement: ContentPlacement::World { translation: t, rotation: r },
                });
            }
            if button(bx + 78.0, y, "Wear") {
                match input.player {
                    Some(player) => commands.push(ContentCommand::Spawn {
                        request: ContentRequest::next(),
                        source: src.clone(),
                        root_id: ObjectId::new(),
                        placement: ContentPlacement::AvatarOf(player),
                    }),
                    None => self.log.push("no possessed Pawn to wear the avatar".into()),
                }
            }
            if button(bx + 156.0, y, "Enter") {
                commands.push(ContentCommand::EnterWorld { request: ContentRequest::next(), source: src.clone() });
            }
            y += row;
        }
        text(y, "Loaded", &small, egui::Color32::GRAY);
        y += row;
        let loaded = manager.loaded();
        if loaded.is_empty() {
            text(y, "nothing loaded", &small, egui::Color32::GRAY);
            y += row;
        }
        for (id, info, count) in loaded {
            text(
                y,
                &format!("{} ({}, {} by {}) x{count}", info.name, info.kind.label(), info.version, info.author),
                &small,
                egui::Color32::WHITE,
            );
            let mut bx = x + width - 78.0;
            if button(bx, y, "Unload") {
                commands.push(ContentCommand::Unload(id.clone()));
            }
            let actions: &[&str] = match info.kind {
                ContentKind::World => &["Enter"],
                ContentKind::Avatar => &["Wear", "Spawn"],
                ContentKind::Prop => &["Spawn"],
            };
            for action in actions {
                bx -= 78.0;
                if !button(bx, y, action) {
                    continue;
                }
                let (request, root_id) = (ContentRequest::next(), ObjectId::new());
                match *action {
                    "Enter" => commands.push(ContentCommand::EnterWorld { request, source: id.clone() }),
                    "Spawn" => {
                        let (t, r) = input.spawn_at;
                        commands.push(ContentCommand::Spawn {
                            request,
                            source: id.clone(),
                            root_id,
                            placement: ContentPlacement::World { translation: t, rotation: r },
                        });
                    }
                    _ => match input.player {
                        Some(player) => commands.push(ContentCommand::Spawn {
                            request,
                            source: id.clone(),
                            root_id,
                            placement: ContentPlacement::AvatarOf(player),
                        }),
                        None => self.log.push("no possessed Pawn to wear the avatar".into()),
                    },
                }
            }
            y += row;
        }
        for line in &self.log {
            let color = if line.starts_with("FAILED") { egui::Color32::from_rgb(240, 110, 100) } else { egui::Color32::LIGHT_GRAY };
            text(y, line, &small, color);
            y += row;
        }
        for c in commands {
            manager.queue(c);
        }
        input.pointer.is_some_and(|p| panel.contains(p))
    }
}

// ------------------------------------------------------------------- builder

/// Tools > Build Content Package (editor).
#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::package::{BuildOptions, ComponentStatus};
    use zestengine_assets::{ComponentData, MaterialRef, SceneEntry};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = std::env::temp_dir().join(format!("zestengine-content-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn entry(name: &str, source: ObjectSource, material: MaterialRef, components: Vec<ComponentData>) -> SceneEntry {
        SceneEntry {
            id: ObjectId::new().to_string(),
            name: name.into(),
            source,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material,
            extra_materials: vec![],
            renderer_split: false,
            parent: None,
            components,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    fn classify(k: &str) -> ComponentStatus {
        if ComponentRegistry::with_builtins().names().any(|n| n == k) {
            ComponentStatus::Builtin
        } else {
            ComponentStatus::Unknown
        }
    }

    /// A world made from repo assets: the glass-table model, a primitive with
    /// a material using the repo's ferris texture, and a Spin component.
    fn build_world(limits: ContentLimits) -> (PathBuf, PathBuf, package::PackagePlan) {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = temp_dir("proj");
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::create_dir_all(root.join("textures")).unwrap();
        std::fs::copy(repo.join("models/modern_glass_table.glb"), root.join("models/table.glb")).unwrap();
        std::fs::copy(repo.join("crates/zestengine-editor/assets/ferris.png"), root.join("textures/ferris.png")).unwrap();
        let material = MaterialRef::Inline {
            params: Default::default(),
            features: Default::default(),
            shader: "standard".into(),
            custom: Default::default(),
            render_queue: None,
            textures: zestengine_assets::MaterialTextures {
                albedo: Some("textures/ferris.png".into()),
                ..Default::default()
            },
        };
        let spin = ComponentData::new("Spin", "(axis:(0.0,1.0,0.0),degrees_per_second:90.0)");
        let scene = SceneFile {
            entries: vec![
                entry(
                    "Table",
                    ObjectSource::Model { path: "models/table.glb".into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] },
                    MaterialRef::File(String::new()),
                    vec![],
                ),
                entry("Box", ObjectSource::Primitive { shape: zestengine_assets::PrimitiveShape::Cube }, material, vec![spin]),
            ],
            ..Default::default()
        };
        let opts = BuildOptions {
            kind: ContentKind::World,
            name: "Table World".into(),
            author: "tests".into(),
            version: "1".into(),
            description: String::new(),
            project_root: root.clone(),
            scene,
            entry: "scenes/table.scene".into(),
            thumbnail: None,
            classify: &classify,
            strip_plugin_components: false,
            include_texture_cache: false,
            limits,
        };
        let plan = package::plan_package(&opts).unwrap();
        assert!(plan.is_ok(), "{:?}", plan.errors);
        let out = root.join("table.zestenginepak");
        package::write_package(&plan, &out).unwrap();
        (root, out, plan)
    }

    fn load(
        path: &str,
        cache: &ContentCache,
        limits: &dyn Fn(ContentKind) -> ContentLimits,
        expect: &[ContentKind],
        game_root: &Path,
    ) -> Result<LoadedContent, String> {
        let registry = ComponentRegistry::with_builtins();
        let known: HashSet<String> = registry.names().map(str::to_owned).collect();
        let fp = package::components_fingerprint(known.iter().map(String::as_str));
        load_package_on_worker(path, cache, &HttpFetcher, limits, game_root, &|k| known.contains(k), &fp, expect, None, &[], &|_| {})
    }

    #[test]
    fn remote_players_announced_avatar_is_requested_once() {
        let mut scene = LoadedScene::empty();
        let player = ObjectId::new();
        scene.objects.push(crate::scene::SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: player,
            name: String::new(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: vec![Box::new(zestengine_core::NetworkAvatar {
                remote: true,
                wearing: Some(zestengine_core::WornAvatar { id: "abc".into(), url: "http://example.com/a.zestenginepak".into() }),
                ..Default::default()
            })],
        });
        let mut m = ContentManager::new(temp_dir("remote-avatar-cache"));
        m.follow_remote_avatars(&scene);
        m.follow_remote_avatars(&scene);
        assert_eq!(m.queued.len(), 1, "requested once");
        let ContentCommand::Spawn { source, placement: ContentPlacement::AvatarOf(p), .. } = &m.queued[0] else { panic!() };
        assert_eq!((source.as_str(), *p), ("http://example.com/a.zestenginepak", player));
    }

    #[test]
    fn world_package_loads_headless_and_reloads_from_cache() {
        let (root, out, plan) = build_world(ContentLimits::for_kind(ContentKind::World));
        let game = temp_dir("game");
        let cache = ContentCache::new(temp_dir("cache"));
        let loaded = load(&out.to_string_lossy(), &cache, &ContentLimits::for_kind, &[], &game).unwrap();
        assert_eq!(loaded.extracted.manifest.content_hash, plan.manifest.content_hash);
        assert!(loaded.stats.triangles > 0 && loaded.stats.triangles == plan.manifest.stats.triangles);
        let ObjectSource::Model { path, .. } = &loaded.data.file.entries[0].source else { panic!() };
        assert!(Path::new(path).is_absolute() && Path::new(path).starts_with(&cache.root), "{path}");
        assert!(loaded.data.models.contains_key(path), "model decoded on the worker");
        let Some(MaterialRef::Inline { textures, .. }) = zestengine_assets::material_refs(&loaded.data.file.entries[1]).into_iter().next() else { panic!() };
        assert!(textures.albedo.as_ref().unwrap().is_file());
        assert!(loaded.data.file.entries[1].components.iter().any(|c| c.kind == "Spin"));

        // By content id: straight from the cache, no package file needed.
        std::fs::remove_file(&out).unwrap();
        let again = load(&plan.manifest.content_hash, &cache, &ContentLimits::for_kind, &[], &game).unwrap();
        assert_eq!(again.extracted.entry, loaded.extracted.entry);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&cache.root);
    }

    #[test]
    fn wrong_kind_and_real_limits_are_rejected() {
        let (root, out, mut plan) = build_world(ContentLimits::for_kind(ContentKind::World));
        let game = temp_dir("game");
        let cache = ContentCache::new(temp_dir("cache"));
        let err = load(&out.to_string_lossy(), &cache, &ContentLimits::for_kind, &[ContentKind::Avatar], &game)
            .err()
            .unwrap();
        assert!(err.contains("requested"), "{err}");

        // The manifest under-reports its triangles; the measured count still fails.
        plan.manifest.stats.triangles = 0;
        let lying = root.join("lying.zestenginepak");
        package::write_package(&plan, &lying).unwrap();
        let strict = |k: ContentKind| ContentLimits { max_triangles: 10, ..ContentLimits::for_kind(k) };
        let err = load(&lying.to_string_lossy(), &cache, &strict, &[], &game).err().unwrap();
        assert!(err.contains("triangles"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&cache.root);
    }

    #[test]
    fn spawned_copies_get_fresh_ids_and_follow_references() {
        let root = entry("Root", ObjectSource::Empty, MaterialRef::File(String::new()), vec![]);
        let mut child = entry("Child", ObjectSource::Empty, MaterialRef::File(String::new()), vec![]);
        child.parent = Some(0);
        child.components.push(ComponentData::new("Ref", format!("(target:\"{}\")", root.id)));
        let file = SceneFile { entries: vec![root.clone(), child], ..Default::default() };
        let root_id = ObjectId::new();
        let copy = remap_ids(&file, root_id);
        assert_eq!(copy[0].id, root_id.to_string());
        assert_ne!(copy[1].id, file.entries[1].id);
        assert!(copy[1].components[0].data.contains(&root_id.to_string()));
        assert_eq!(copy[1].parent, Some(0));
    }

    #[test]
    fn avatar_checks_flag_unskinned_models() {
        let (root, _, _) = build_world(ContentLimits::for_kind(ContentKind::World));
        let mut table = entry(
            "Table",
            ObjectSource::Model { path: "models/table.glb".into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] },
            MaterialRef::File(String::new()),
            vec![],
        );
        table.components.push(ComponentData::new("Avatar Descriptor", "(view_position:(0.0,9.0,0.0))"));
        let warnings = avatar_warnings(&root, &SceneFile { entries: vec![table], ..Default::default() });
        assert!(warnings.iter().any(|w| w.contains("eye height")), "{warnings:?}");
        assert!(warnings.iter().any(|w| w.contains("skeleton")), "{warnings:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    fn serve_once(response: Vec<u8>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut s, _)) = listener.accept() {
                // Read the whole request header first: closing with unread
                // request bytes makes the kernel send a reset instead of a FIN.
                let mut req = Vec::new();
                let mut buf = [0u8; 1024];
                while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                    match s.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => req.extend_from_slice(&buf[..n]),
                    }
                }
                let _ = s.write_all(&response);
                let _ = s.shutdown(std::net::Shutdown::Write);
                // Let the client finish reading (or give up at the size limit)
                // before the socket drops.
                let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(2)));
                while matches!(s.read(&mut buf), Ok(n) if n > 0) {}
            }
        });
        format!("http://{addr}/pkg.zestenginepak")
    }

    #[test]
    fn http_fetcher_downloads_plain_and_chunked() {
        let dir = temp_dir("http");
        let body = b"hello content".to_vec();
        let mut plain = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
        plain.extend_from_slice(&body);
        let dest = dir.join("a.bin");
        HttpFetcher.fetch(&serve_once(plain), &dest, 1024, &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);

        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n8\r\n content\r\n0\r\n\r\n".to_vec();
        HttpFetcher.fetch(&serve_once(chunked), &dest, 1024, &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);

        let mut big = b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n".to_vec();
        big.extend(std::iter::repeat_n(0u8, 10));
        assert!(HttpFetcher.fetch(&serve_once(big), &dest, 1024, &|_, _| {}).unwrap_err().contains("limit"));
        assert!(HttpFetcher.fetch("https://example.com/x", &dest, 1024, &|_, _| {}).unwrap_err().contains("TLS"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
