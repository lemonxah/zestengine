//! Stored cluster DAGs (GEO-3), beside the model like the texture and model
//! import caches: `<dir>/.zestengine_geocache/<file>.zgeo`.
//!
//! One file per model holds the DAG of every mesh big enough to want one
//! ([`GEO_MIN_TRIANGLES`]). The key is a hash of the meshes' vertex, index and
//! blendshape data, the builder settings and [`GEO_BUILDER_VERSION`]; it is
//! stored inside the file, not in its name, so a file cooked into a package
//! can be dropped in place without knowing the importer's own key. A file is
//! used only when its checksum, its key and every DAG's structural validation
//! ([`ClusterDag::validate`]) pass; anything else is rebuilt. Bad DAG data can
//! hang the GPU, so nothing unvalidated is ever handed out.
//!
//! Layout: `ZGEO`, u32 [`GEO_FILE_FORMAT`], u32 builder version, u128 key,
//! u32 mesh slots, per slot a u64 length (0 = no DAG) and the DAG bytes, then a
//! u128 checksum over everything before it.
//!
//! Format 2 (GEO-6) lets a slot hold a *paged* mesh (`ZPG1`, see
//! [`crate::cluster_pages`]) instead of a plain DAG (`ZDG1`): big meshes are
//! stored page by page so the renderer's streamer can read single pages from the
//! file. Format 1 files (plain slots only) still load; they simply have no page
//! directory, and the engine splits their DAG in memory when it streams.
//!
//! Building is heavy (about 2.5 s per million triangles on 32 threads, 20 s on
//! one; the default build uses half the cores, spread over the meshes of the
//! model biggest first): call [`load_or_build`] from a worker, never from the frame loop.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::Scene;
use crate::cluster_dag::{CANCELLED, ClusterDag, DagOptions, GEO_BUILDER_VERSION, GEO_MIN_TRIANGLES, build_cluster_dag_ctl};
use crate::cluster_pages::{PageOptions, PagePlan, PagedFile, decode_paged, encode_paged, is_paged};
use crate::import_cache::Hasher128;

pub const GEOCACHE_DIR: &str = ".zestengine_geocache";
pub const GEO_EXTENSION: &str = "zgeo";
/// Bump when the file layout changes.
pub const GEO_FILE_FORMAT: u32 = 2;
/// Oldest container format [`decode`] reads.
pub const GEO_FILE_FORMAT_MIN: u32 = 1;
/// The part of the cache key that stands for the DAG data itself (unchanged by the
/// container, so format 1 files stay valid for the same meshes).
const GEO_KEY_FORMAT: u32 = 1;
/// Largest cache file read or written.
pub const MAX_FILE_BYTES: u64 = 1 << 30;
/// Android / Quest builds ship the file here instead (the APK packager drops dot-prefixed paths).
pub const COOKED_GEOCACHE_DIR: &str = "zestengine_geocache";
const MAGIC: &[u8; 4] = b"ZGEO";

/// The DAGs of one model, one slot per mesh of the scene (`None` for meshes
/// under the threshold or that failed to build).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelGeometry {
    pub key: u128,
    pub meshes: Vec<Option<ClusterDag>>,
}

impl ModelGeometry {
    pub fn is_empty(&self) -> bool {
        self.meshes.iter().all(Option::is_none)
    }

    pub fn clusters(&self) -> usize {
        self.meshes.iter().flatten().map(|d| d.clusters.len()).sum()
    }

    /// Serialised size of all the DAGs.
    pub fn dag_bytes(&self) -> usize {
        self.meshes.iter().flatten().map(ClusterDag::byte_len).sum()
    }
}

fn enabled() -> bool {
    std::env::var_os("ZESTENGINE_NO_GEO_CACHE").is_none()
}

/// Whether a mesh gets a DAG.
pub fn wants_dag(mesh: &zestengine_render::MeshData) -> bool {
    mesh.indices.len() / 3 >= GEO_MIN_TRIANGLES
}

/// `<dir>/.zestengine_geocache/<file>.zgeo`.
pub fn geo_path(source: &Path) -> Option<PathBuf> {
    let name = source.file_name()?.to_string_lossy();
    Some(source.parent()?.join(GEOCACHE_DIR).join(format!("{name}.{GEO_EXTENSION}")))
}

/// Package / APK path of a model's cooked file: `zestengine_geocache/<model path>.zgeo`.
pub fn cooked_geocache_rel(model_rel: &str) -> String {
    format!("{COOKED_GEOCACHE_DIR}/{model_rel}.{GEO_EXTENSION}")
}

/// Inverse of [`cooked_geocache_rel`].
pub fn parse_cooked_geocache_rel(rel: &str) -> Option<&str> {
    let rest = rel.strip_prefix(COOKED_GEOCACHE_DIR)?.strip_prefix('/')?;
    let model = rest.strip_suffix(GEO_EXTENSION)?.strip_suffix('.')?;
    crate::package::normalize_rel(model).filter(|m| m == model)?;
    Some(model)
}

/// The meshes and blendshapes of a model: everything the geometry cook reads.
#[derive(Clone, Copy)]
pub struct GeoView<'a> {
    pub meshes: &'a [zestengine_render::MeshData],
    /// Blendshapes per mesh, parallel to `meshes` (shorter or empty = none).
    pub morphs: &'a [Vec<crate::MorphTarget>],
}

impl<'a> GeoView<'a> {
    pub fn of(scene: &'a Scene) -> Self {
        Self { meshes: &scene.meshes, morphs: &scene.morphs }
    }
}

/// An owned [`GeoView`] that keeps only what the DAG needs: the meshes big
/// enough for one (the rest are empty placeholders, so mesh numbers line up)
/// and their blendshapes. The engine takes one when a model is installed, so
/// the cluster worker builds from the meshes the load already decoded instead
/// of importing the model a second time. Its key and DAGs are identical to the
/// full scene's.
pub struct GeoSource {
    meshes: Vec<zestengine_render::MeshData>,
    morphs: Vec<Vec<crate::MorphTarget>>,
}

impl GeoSource {
    pub fn of(scene: &Scene) -> Self {
        let meshes = scene
            .meshes
            .iter()
            .map(|m| {
                if wants_dag(m) {
                    zestengine_render::MeshData { vertices: m.vertices.clone(), indices: m.indices.clone(), has_lightmap_uv: m.has_lightmap_uv }
                } else {
                    zestengine_render::MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false }
                }
            })
            .collect::<Vec<_>>();
        let morphs = (0..scene.meshes.len())
            .map(|i| if wants_dag(&scene.meshes[i]) { scene.morphs.get(i).cloned().unwrap_or_default() } else { Vec::new() })
            .collect();
        Self { meshes, morphs }
    }

    pub fn view(&self) -> GeoView<'_> {
        GeoView { meshes: &self.meshes, morphs: &self.morphs }
    }

    /// [`load_or_build`] from this copy.
    pub fn load_or_build(&self, source: &Path) -> ModelGeometry {
        load_or_build_view(source, self.view(), &DagOptions::default())
    }

    /// [`load_or_build`](Self::load_or_build) that reports through, and can be cancelled by, `progress`.
    pub fn load_or_build_tracked(&self, source: &Path, progress: &BuildProgress) -> ModelGeometry {
        load_or_build_tracked_view(source, self.view(), &DagOptions::default(), progress)
    }

    /// [`load_or_build`] that leaves paged meshes of a stored file on disk
    /// ([`LazyGeometry`]); a missing, stale or unusable file is rebuilt (and
    /// stored) as usual and comes back fully decoded.
    pub fn load_or_build_lazy(&self, source: &Path) -> LazyGeometry {
        self.load_or_build_lazy_tracked(source, &BuildProgress::default())
    }

    /// [`load_or_build_lazy`](Self::load_or_build_lazy) with a watchable, cancellable build.
    pub fn load_or_build_lazy_tracked(&self, source: &Path, progress: &BuildProgress) -> LazyGeometry {
        let opts = DagOptions::default();
        let key = model_key_view(self.view(), &opts);
        if enabled()
            && let Some(l) = load_lazy(source, key)
            && fits(&l.geo, self.view())
        {
            return l;
        }
        LazyGeometry::eager(load_or_build_tracked_view(source, self.view(), &opts, progress))
    }
}

/// Cache key of a scene's geometry under `opts`.
pub fn model_key(scene: &Scene, opts: &DagOptions) -> u128 {
    model_key_view(GeoView::of(scene), opts)
}

fn model_key_view(scene: GeoView, opts: &DagOptions) -> u128 {
    let mut h = Hasher128::new();
    h.field(&GEO_BUILDER_VERSION.to_le_bytes());
    h.field(&GEO_KEY_FORMAT.to_le_bytes());
    h.field(&(GEO_MIN_TRIANGLES as u64).to_le_bytes());
    h.field(&opts.fingerprint());
    h.field(&(scene.meshes.len() as u64).to_le_bytes());
    for (i, m) in scene.meshes.iter().enumerate().filter(|(_, m)| wants_dag(m)) {
        h.field(&(i as u64).to_le_bytes());
        h.field(bytemuck::cast_slice(&m.vertices));
        h.field(bytemuck::cast_slice(&m.indices));
        h.field(&[u8::from(m.has_lightmap_uv)]);
        for t in scene.morphs.get(i).into_iter().flatten() {
            h.field(t.name.as_bytes());
            for (v, dp, dn) in &t.deltas {
                h.update(&v.to_le_bytes());
                for x in dp.iter().chain(dn) {
                    h.update(&x.to_bits().to_le_bytes());
                }
            }
        }
    }
    h.finish()
}

/// Live progress of a model's DAG build, shared between the build (any number
/// of worker threads) and whoever shows it (the editor's task list). Plain
/// atomics: reading never blocks the build or the frame. `cancel` stops the
/// build between groups; a cancelled build stores nothing.
#[derive(Debug, Default)]
pub struct BuildProgress {
    meshes_total: AtomicU32,
    meshes_done: AtomicU32,
    tris_total: AtomicU64,
    tris_done: AtomicU64,
    cancel: AtomicBool,
}

/// A consistent-enough reading of a [`BuildProgress`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildSnapshot {
    pub meshes_done: u32,
    pub meshes_total: u32,
    pub tris_done: u64,
    pub tris_total: u64,
}

impl BuildSnapshot {
    /// 0..=1, weighted by triangles (0 before the build has started).
    pub fn fraction(&self) -> f32 {
        if self.tris_total == 0 { 0.0 } else { (self.tris_done.min(self.tris_total) as f64 / self.tris_total as f64) as f32 }
    }

    /// "12/180 meshes, 3.2M/20.1M triangles".
    pub fn label(&self) -> String {
        let m = |n: u64| -> String {
            if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) } else if n >= 10_000 { format!("{:.0}k", n as f64 / 1e3) } else { n.to_string() }
        };
        format!("{}/{} meshes, {}/{} triangles", self.meshes_done, self.meshes_total, m(self.tris_done.min(self.tris_total)), m(self.tris_total))
    }
}

impl BuildProgress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Start (or restart) counting a build of `meshes` meshes and `tris` triangles.
    pub fn begin(&self, meshes: u32, tris: u64) {
        self.meshes_total.store(meshes, Ordering::Relaxed);
        self.tris_total.store(tris, Ordering::Relaxed);
        self.meshes_done.store(0, Ordering::Relaxed);
        self.tris_done.store(0, Ordering::Relaxed);
    }

    /// Triangles of the build finished (partial progress inside a mesh counts).
    pub fn add_tris(&self, n: u64) {
        self.tris_done.fetch_add(n, Ordering::Relaxed);
    }

    pub fn mesh_done(&self) {
        self.meshes_done.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> BuildSnapshot {
        BuildSnapshot {
            meshes_done: self.meshes_done.load(Ordering::Relaxed),
            meshes_total: self.meshes_total.load(Ordering::Relaxed),
            tris_done: self.tris_done.load(Ordering::Relaxed),
            tris_total: self.tris_total.load(Ordering::Relaxed),
        }
    }

    /// Ask the build to stop as soon as it can.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Build the DAG of every qualifying mesh. Meshes are built concurrently,
/// biggest first, by a pool of `opts` threads that pull the next mesh when they
/// finish one (work stealing by triangle count); a mesh with a big share of the
/// triangles also spreads its own levels over a matching share of the threads.
/// The output does not depend on any of this. A mesh that fails to build gets no DAG.
pub fn build_model(scene: &Scene, opts: &DagOptions) -> ModelGeometry {
    build_model_view(GeoView::of(scene), opts)
}

fn build_model_view(scene: GeoView, opts: &DagOptions) -> ModelGeometry {
    let progress = BuildProgress::default();
    let built = build_meshes(scene, opts, &progress, &|| false, &|_| {});
    ModelGeometry { key: model_key_view(scene, opts), meshes: built.dags }
}

struct Built {
    dags: Vec<Option<ClusterDag>>,
    secs: Vec<f32>,
    /// Stopped before every mesh was built.
    cancelled: bool,
}

/// How many threads a mesh of `tris` out of `total` triangles gets inside the pool of `threads`.
fn inner_threads(threads: usize, tris: usize, total: usize) -> usize {
    ((threads as u128 * tris as u128).div_ceil(total.max(1) as u128) as usize).clamp(1, threads.max(1))
}

/// The worker pool behind [`build_model`], [`load_or_build_tracked`] and [`rebuild`].
/// `report` gets the triangle-weighted fraction (monotonic) as the build advances.
fn build_meshes(scene: GeoView, opts: &DagOptions, progress: &BuildProgress, cancelled: &(dyn Fn() -> bool + Sync), report: &(dyn Fn(f32) + Sync)) -> Built {
    let n = scene.meshes.len();
    let mut jobs: Vec<(usize, usize)> = scene.meshes.iter().enumerate().filter(|(_, m)| wants_dag(m)).map(|(i, m)| (i, m.indices.len() / 3)).collect();
    let total: usize = jobs.iter().map(|j| j.1).sum();
    progress.begin(jobs.len() as u32, total as u64);
    // Biggest first (ties by mesh number): the long jobs start at once, the small ones fill the gaps.
    jobs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let threads = crate::cluster_dag::thread_count(opts);
    let stop = || cancelled() || progress.is_cancelled();
    let last = AtomicU32::new(0);
    let note = || {
        let f = (progress.snapshot().fraction() * 1_000_000.0) as u32;
        if last.fetch_max(f, Ordering::Relaxed) < f {
            report(f as f32 / 1_000_000.0);
        }
    };
    let out: Mutex<(Vec<Option<ClusterDag>>, Vec<f32>)> = Mutex::new(((0..n).map(|_| None).collect(), vec![0.0; n]));
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(jobs.len()).max(1) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= jobs.len() || stop() {
                        break;
                    }
                    let (i, tris) = jobs[k];
                    let morphs = scene.morphs.get(i).map_or(&[][..], Vec::as_slice);
                    let mesh_opts = DagOptions { section: i as u32, threads: inner_threads(threads, tris, total), ..opts.clone() };
                    let counted = AtomicU64::new(0);
                    let t = std::time::Instant::now();
                    let built = build_cluster_dag_ctl(&scene.meshes[i], morphs, &mesh_opts, &|f| {
                        let upto = (tris as f64 * f.clamp(0.0, 1.0) as f64) as u64;
                        let before = counted.fetch_max(upto, Ordering::Relaxed);
                        if upto > before {
                            progress.add_tris(upto - before);
                            note();
                        }
                    }, &stop);
                    let secs = t.elapsed().as_secs_f32();
                    let dag = match built {
                        Ok(d) => Some(d),
                        Err(e) => {
                            if e != CANCELLED {
                                tracing::warn!("cluster DAG for mesh {i}: {e}");
                            }
                            None
                        }
                    };
                    // The mesh counts as fully done whatever the callback saw last.
                    progress.add_tris((tris as u64).saturating_sub(counted.load(Ordering::Relaxed)));
                    progress.mesh_done();
                    note();
                    let mut guard = out.lock().unwrap_or_else(|e| e.into_inner());
                    guard.0[i] = dag;
                    guard.1[i] = secs;
                }
            });
        }
    });
    let (dags, secs) = out.into_inner().unwrap_or_else(|e| e.into_inner());
    let done = progress.snapshot().meshes_done as usize;
    Built { dags, secs, cancelled: stop() || done < jobs.len() }
}

pub fn encode(geo: &ModelGeometry) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&GEO_FILE_FORMAT.to_le_bytes());
    b.extend_from_slice(&GEO_BUILDER_VERSION.to_le_bytes());
    b.extend_from_slice(&geo.key.to_le_bytes());
    b.extend_from_slice(&(geo.meshes.len() as u32).to_le_bytes());
    for m in &geo.meshes {
        match m {
            Some(d) => {
                // Big static meshes are stored page by page (the streamer reads single pages).
                let plan = PagePlan::split(d, &PageOptions::default());
                let bytes = if plan.is_single() { d.to_bytes() } else { encode_paged(d, &plan) };
                b.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                b.extend_from_slice(&bytes);
            }
            None => b.extend_from_slice(&0u64.to_le_bytes()),
        }
    }
    let mut h = Hasher128::new();
    h.update(&b);
    b.extend_from_slice(&h.finish().to_le_bytes());
    b
}

/// Parse and fully validate. `Err` for a bad checksum, another format or
/// builder version, truncation, or any DAG that fails [`ClusterDag::validate`].
pub fn decode(bytes: &[u8]) -> Result<ModelGeometry, String> {
    if bytes.len() < 4 + 4 + 4 + 16 + 4 + 16 || bytes.len() as u64 > MAX_FILE_BYTES {
        return Err("geometry cache has the wrong size".into());
    }
    let (body, sum) = bytes.split_at(bytes.len() - 16);
    let mut h = Hasher128::new();
    h.update(body);
    if h.finish().to_le_bytes() != sum {
        return Err("geometry cache checksum differs".into());
    }
    if &body[..4] != MAGIC {
        return Err("not a geometry cache".into());
    }
    let u32_at = |at: usize| u32::from_le_bytes(body[at..at + 4].try_into().unwrap());
    if !(GEO_FILE_FORMAT_MIN..=GEO_FILE_FORMAT).contains(&u32_at(4)) || u32_at(8) != GEO_BUILDER_VERSION {
        return Err("geometry cache is from another builder version".into());
    }
    let key = u128::from_le_bytes(body[12..28].try_into().unwrap());
    let slots = u32_at(28) as usize;
    let mut at = 32;
    if slots > body.len() {
        return Err("geometry cache slot count is absurd".into());
    }
    let mut meshes = Vec::with_capacity(slots);
    for _ in 0..slots {
        let len = body.get(at..at + 8).ok_or("truncated geometry cache")?;
        let len = u64::from_le_bytes(len.try_into().unwrap()) as usize;
        at += 8;
        if len == 0 {
            meshes.push(None);
            continue;
        }
        let slot = body.get(at..at.checked_add(len).ok_or("bad length")?).ok_or("truncated geometry cache")?;
        let dag = if is_paged(slot) { decode_paged(slot)?.0 } else { ClusterDag::from_bytes(slot)? };
        at += len;
        dag.validate()?;
        meshes.push(Some(dag));
    }
    if at != body.len() {
        return Err("trailing bytes in geometry cache".into());
    }
    Ok(ModelGeometry { key, meshes })
}

/// Whether `geo` was built for `scene`'s meshes: slot per mesh, the DAG's
/// vertex count equals the mesh's, and a DAG exists exactly for qualifying meshes.
fn fits(geo: &ModelGeometry, scene: GeoView) -> bool {
    geo.meshes.len() == scene.meshes.len()
        && geo.meshes.iter().zip(scene.meshes).all(|(d, m)| match d {
            Some(d) => d.header.source_vertices as usize == m.vertices.len(),
            None => !wants_dag(m),
        })
}

/// The stored geometry for `source`, only if it matches `key` and validates.
pub fn load(source: &Path, key: u128) -> Option<ModelGeometry> {
    let path = geo_path(source)?;
    if std::fs::metadata(&path).ok()?.len() > MAX_FILE_BYTES {
        return None;
    }
    match decode(&std::fs::read(&path).ok()?) {
        Ok(g) if g.key == key => Some(g),
        Ok(_) => None,
        Err(e) => {
            tracing::debug!("geometry cache {} unusable: {e}", path.display());
            None
        }
    }
}

/// Store `geo` for `source`. Failures are ignored: it is rebuilt next time.
pub fn store(source: &Path, geo: &ModelGeometry) {
    let Some(path) = geo_path(source) else { return };
    let bytes = encode(geo);
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return;
    }
    if let Err(e) = crate::write_atomic(&path, &bytes) {
        tracing::debug!("geometry cache not written: {e:#}");
    }
}

/// The geometry of `scene` (imported from `source`), from the stored copy when
/// its key and validation hold, else built and stored. Heavy: call on a
/// worker. `ZESTENGINE_NO_GEO_CACHE` builds without touching the disk.
pub fn load_or_build(source: &Path, scene: &Scene) -> ModelGeometry {
    load_or_build_with(source, scene, &DagOptions::default())
}

/// [`load_or_build`] with explicit builder settings (`threads` is free to
/// differ: it is not part of the key).
pub fn load_or_build_with(source: &Path, scene: &Scene, opts: &DagOptions) -> ModelGeometry {
    load_or_build_view(source, GeoView::of(scene), opts)
}

/// One lock per model file: two loads of the same model at once (a scene reload, a preview
/// beside the scene) must not both spend the build; the second waits and reads the stored copy.
fn build_lock(source: &Path) -> std::sync::Arc<std::sync::Mutex<()>> {
    static LOCKS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, std::sync::Arc<std::sync::Mutex<()>>>>> = std::sync::OnceLock::new();
    let mut map = LOCKS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    map.entry(source.to_path_buf()).or_default().clone()
}

fn load_or_build_view(source: &Path, scene: GeoView, opts: &DagOptions) -> ModelGeometry {
    load_or_build_tracked_view(source, scene, opts, &BuildProgress::default())
}

/// [`load_or_build`] whose build can be watched and cancelled through `progress`
/// (a cache hit never touches it). A cancelled build stores nothing and returns
/// geometry without DAGs.
pub fn load_or_build_tracked(source: &Path, scene: &Scene, progress: &BuildProgress) -> ModelGeometry {
    load_or_build_tracked_view(source, GeoView::of(scene), &DagOptions::default(), progress)
}

fn load_or_build_tracked_view(source: &Path, scene: GeoView, opts: &DagOptions, progress: &BuildProgress) -> ModelGeometry {
    let lock = build_lock(source);
    let _building = lock.lock().unwrap_or_else(|e| e.into_inner());
    let key = model_key_view(scene, opts);
    if enabled()
        && let Some(g) = load(source, key)
        && fits(&g, scene)
    {
        return g;
    }
    let t = std::time::Instant::now();
    let built = build_meshes(scene, opts, progress, &|| false, &|_| {});
    if built.cancelled {
        tracing::info!("model {}: cluster build cancelled", source.display());
        return ModelGeometry { key, meshes: (0..scene.meshes.len()).map(|_| None).collect() };
    }
    let geo = ModelGeometry { key, meshes: built.dags };
    if !geo.is_empty() {
        tracing::info!("model {}: {} cluster(s), {:.1} MB of cluster data built in {:.1} s", source.display(), geo.clusters(), geo.dag_bytes() as f64 / 1e6, t.elapsed().as_secs_f32());
        if enabled() {
            store(source, &geo);
        }
    }
    geo
}

/// The bytes to ship in a package for `source`: the stored geometry or a fresh
/// build, `None` when the model has nothing big enough for a DAG.
pub fn cook(source: &Path, scene: &Scene) -> Option<(ModelGeometry, Vec<u8>)> {
    // An offline step: use every core.
    let all = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let geo = load_or_build_with(source, scene, &DagOptions { threads: all, ..DagOptions::default() });
    (!geo.is_empty()).then(|| {
        let bytes = encode(&geo);
        (geo, bytes)
    })
}

/// The paged meshes of the stored geometry file of `source` (one slot per mesh,
/// `None` for plain / absent DAGs), opened for reading single pages: only the
/// headers, skeletons and page tables are read. Not validated beyond the page
/// format's own checksums: use it for a model whose geometry [`load_or_build`]
/// just returned (that validated the file); `key` must equal the geometry's key.
pub fn open_paged(source: &Path, key: u128) -> Option<Vec<Option<PagedFile>>> {
    use std::io::{Read, Seek, SeekFrom};
    let path = geo_path(source)?;
    let total = std::fs::metadata(&path).ok()?.len();
    if total > MAX_FILE_BYTES {
        return None;
    }
    let mut f = std::fs::File::open(&path).ok()?;
    let mut head = [0u8; 32];
    f.read_exact(&mut head).ok()?;
    let u32_at = |at: usize| u32::from_le_bytes(head[at..at + 4].try_into().unwrap());
    if &head[..4] != MAGIC || !(GEO_FILE_FORMAT_MIN..=GEO_FILE_FORMAT).contains(&u32_at(4)) || u32_at(8) != GEO_BUILDER_VERSION || u128::from_le_bytes(head[12..28].try_into().unwrap()) != key {
        return None;
    }
    let slots = u32_at(28) as usize;
    if slots as u64 > total {
        return None;
    }
    let mut at = 32u64;
    let mut out = Vec::with_capacity(slots);
    for _ in 0..slots {
        f.seek(SeekFrom::Start(at)).ok()?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).ok()?;
        let len = u64::from_le_bytes(len);
        at += 8;
        if at.checked_add(len)? > total {
            return None;
        }
        let mut magic = [0u8; 4];
        if len >= 4 {
            f.read_exact(&mut magic).ok()?;
        }
        out.push(if len >= 4 && &magic == b"ZPG1" { PagedFile::open_file(&path, at, len).ok() } else { None });
        at += len;
    }
    Some(out)
}

/// A model's geometry loaded for streaming (GEO-6): the plain DAGs are decoded
/// as usual, but a paged mesh is NOT assembled. Its slot in `geo` holds the
/// skeleton alone (clusters, links, groups, BVH; no vertices or triangles, so
/// it cannot be drawn or uploaded as a DAG) and `paged` its open file; pages
/// are read from disk when the renderer asks for them. Peak memory while loading
/// is the skeletons (96 bytes per cluster) instead of the whole payload.
pub struct LazyGeometry {
    pub geo: ModelGeometry,
    /// Per slot of `geo.meshes`: the paged file whose skeleton sits there.
    pub paged: Vec<Option<PagedFile>>,
}

impl LazyGeometry {
    /// Wrap fully decoded geometry (nothing is lazy).
    pub fn eager(geo: ModelGeometry) -> Self {
        let paged = geo.meshes.iter().map(|_| None).collect();
        Self { geo, paged }
    }

    /// Turn slot `i` into a full, validated DAG (a paged slot is assembled from
    /// its file). `None` when the pages do not assemble or validate; the slot is
    /// then emptied.
    pub fn make_full(&mut self, i: usize) -> Option<&ClusterDag> {
        if let Some(file) = self.paged.get_mut(i).and_then(Option::take) {
            match file.assemble().and_then(|d| d.validate().map(|()| d)) {
                Ok(d) => self.geo.meshes[i] = Some(d),
                Err(e) => {
                    tracing::warn!("paged cluster mesh {i} could not be assembled: {e}");
                    self.geo.meshes[i] = None;
                }
            }
        }
        self.geo.meshes.get(i)?.as_ref()
    }
}

/// [`load`] without decoding paged meshes (see [`LazyGeometry`]). The whole file
/// is still checksummed (read in chunks, never held), plain DAGs are fully
/// validated, and each paged mesh's header, skeleton and directory are validated
/// now and every page when it is read. `None` for anything unusable: the caller
/// falls back to [`load`] / a rebuild.
pub fn load_lazy(source: &Path, key: u128) -> Option<LazyGeometry> {
    use std::io::{Read, Seek, SeekFrom};
    let path = geo_path(source)?;
    let total = std::fs::metadata(&path).ok()?.len();
    if !(4 + 4 + 4 + 16 + 4 + 16..=MAX_FILE_BYTES).contains(&total) {
        return None;
    }
    let mut f = std::fs::File::open(&path).ok()?;
    // Whole-file checksum in chunks.
    let mut h = Hasher128::new();
    let mut left = total - 16;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let n = left.min(buf.len() as u64) as usize;
        f.read_exact(&mut buf[..n]).ok()?;
        h.update(&buf[..n]);
        left -= n as u64;
    }
    let mut sum = [0u8; 16];
    f.read_exact(&mut sum).ok()?;
    if h.finish().to_le_bytes() != sum {
        return None;
    }
    f.seek(SeekFrom::Start(0)).ok()?;
    let mut head = [0u8; 32];
    f.read_exact(&mut head).ok()?;
    let u32_at = |at: usize| u32::from_le_bytes(head[at..at + 4].try_into().unwrap());
    if &head[..4] != MAGIC || !(GEO_FILE_FORMAT_MIN..=GEO_FILE_FORMAT).contains(&u32_at(4)) || u32_at(8) != GEO_BUILDER_VERSION {
        return None;
    }
    if u128::from_le_bytes(head[12..28].try_into().unwrap()) != key {
        return None;
    }
    let slots = u32_at(28) as usize;
    if slots as u64 > total {
        return None;
    }
    let (mut at, end) = (32u64, total - 16);
    let mut meshes = Vec::with_capacity(slots);
    let mut paged = Vec::with_capacity(slots);
    for _ in 0..slots {
        f.seek(SeekFrom::Start(at)).ok()?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).ok()?;
        let len = u64::from_le_bytes(len);
        at += 8;
        if at.checked_add(len)? > end {
            return None;
        }
        if len == 0 {
            meshes.push(None);
            paged.push(None);
            continue;
        }
        let mut magic = [0u8; 4];
        if len >= 4 {
            f.read_exact(&mut magic).ok()?;
        }
        if len >= 4 && &magic == b"ZPG1" {
            let file = PagedFile::open_file(&path, at, len).ok()?;
            meshes.push(Some(file.skeleton.clone()));
            paged.push(Some(file));
        } else {
            f.seek(SeekFrom::Start(at)).ok()?;
            let mut bytes = vec![0u8; len as usize];
            f.read_exact(&mut bytes).ok()?;
            let dag = ClusterDag::from_bytes(&bytes).ok()?;
            dag.validate().ok()?;
            meshes.push(Some(dag));
            paged.push(None);
        }
        at += len;
    }
    (at == end).then_some(LazyGeometry { geo: ModelGeometry { key, meshes }, paged })
}

/// Install a cooked file next to `source` after full validation. Returns what
/// it holds; `Err` leaves nothing on disk.
pub fn install(source: &Path, blob: &[u8]) -> Result<ModelGeometry, String> {
    let geo = decode(blob)?;
    let path = geo_path(source).ok_or("no cache path")?;
    crate::write_atomic(&path, blob).map_err(|e| format!("{e:#}"))?;
    Ok(geo)
}

/// Install every cooked file under `assets_root/`[`COOKED_GEOCACHE_DIR`] beside
/// its model, then delete the shipped copies (the Android counterpart of
/// `package::seed_cooked_texture_caches`). Returns (files installed, warnings).
pub fn seed_cooked_geocaches(assets_root: &Path) -> (usize, Vec<String>) {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        for e in read.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else if let Ok(rel) = p.strip_prefix(base) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let dir = assets_root.join(COOKED_GEOCACHE_DIR);
    let mut found = Vec::new();
    walk(&dir, &dir, &mut found);
    let (mut seeded, mut warnings) = (0, Vec::new());
    for shipped in found {
        let rel = format!("{COOKED_GEOCACHE_DIR}/{shipped}");
        let Some(model) = parse_cooked_geocache_rel(&rel) else {
            warnings.push(format!("ignoring unknown cooked geometry cache {rel}"));
            continue;
        };
        match std::fs::read(dir.join(&shipped)).map_err(|e| e.to_string()).and_then(|b| install(&assets_root.join(model), &b)) {
            Ok(_) => seeded += 1,
            Err(e) => warnings.push(format!("cooked geometry cache for {model}: {e}")),
        }
    }
    if dir.exists()
        && let Err(e) = std::fs::remove_dir_all(&dir)
    {
        warnings.push(format!("removing {}: {e}", dir.display()));
    }
    (seeded, warnings)
}

/// What a rebuild did: the geometry, and seconds spent per mesh slot (0 for
/// meshes without a DAG).
pub struct Rebuilt {
    pub geo: ModelGeometry,
    pub mesh_secs: Vec<f32>,
}

/// [`build_model`] that reports progress (0..=1, weighted by triangles, from
/// the worker threads) and stops once `cancelled` says so (`None`).
pub fn build_model_progress(scene: &Scene, opts: &DagOptions, progress: &(dyn Fn(f32) + Sync), cancelled: &(dyn Fn() -> bool + Sync)) -> Option<Rebuilt> {
    let view = GeoView::of(scene);
    let built = build_meshes(view, opts, &BuildProgress::default(), cancelled, progress);
    if built.cancelled {
        return None;
    }
    progress(1.0);
    Some(Rebuilt { geo: ModelGeometry { key: model_key_view(view, opts), meshes: built.dags }, mesh_secs: built.secs })
}

/// Rebuild and store the geometry of `source` regardless of what is cached
/// (an explicit user request). A model with nothing big enough for a DAG has
/// its stale file removed. `None` when cancelled; nothing is written then.
pub fn rebuild(source: &Path, scene: &Scene, opts: &DagOptions, progress: &(dyn Fn(f32) + Sync), cancelled: &(dyn Fn() -> bool + Sync)) -> Option<Rebuilt> {
    let r = build_model_progress(scene, opts, progress, cancelled)?;
    if r.geo.is_empty() {
        if let Some(p) = geo_path(source) {
            let _ = std::fs::remove_file(p);
        }
    } else {
        store(source, &r.geo);
    }
    Some(r)
}

/// Whether the stored file for a model can be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheState {
    /// No file (nothing to cache, or not built yet).
    Missing,
    /// Present, valid, and matches the model's current geometry.
    Valid,
    /// Present and well formed, but built for other geometry or settings.
    Stale,
    /// Present but damaged / unreadable (the reason).
    Invalid(String),
}

/// One mesh slot as the inspector shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshGeoInfo {
    pub triangles: usize,
    /// Big enough for a DAG ([`GEO_MIN_TRIANGLES`]).
    pub wants_dag: bool,
    pub lightmapped: bool,
    /// From the stored file when it is usable.
    pub clusters: usize,
    pub dag_bytes: usize,
    pub levels: u32,
    pub skinned: bool,
    pub blendshapes: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeoStatus {
    pub path: Option<PathBuf>,
    pub state: CacheState,
    pub file_bytes: u64,
    pub meshes: Vec<MeshGeoInfo>,
}

/// Read-only report on the stored geometry of `source` (a file read and a key
/// hash: run it off the frame loop).
pub fn inspect(source: &Path, scene: &Scene) -> GeoStatus {
    let path = geo_path(source);
    let mut meshes: Vec<MeshGeoInfo> = scene
        .meshes
        .iter()
        .map(|m| MeshGeoInfo {
            triangles: m.indices.len() / 3,
            wants_dag: wants_dag(m),
            lightmapped: m.has_lightmap_uv,
            clusters: 0,
            dag_bytes: 0,
            levels: 0,
            skinned: false,
            blendshapes: 0,
        })
        .collect();
    let mut status = GeoStatus { path: path.clone(), state: CacheState::Missing, file_bytes: 0, meshes: Vec::new() };
    if let Some(p) = &path
        && let Ok(md) = std::fs::metadata(p)
    {
        status.file_bytes = md.len();
        status.state = if md.len() > MAX_FILE_BYTES {
            CacheState::Invalid("file is larger than the allowed maximum".into())
        } else {
            match std::fs::read(p).map_err(|e| e.to_string()).and_then(|b| decode(&b)) {
                Err(e) => CacheState::Invalid(e),
                Ok(g) if g.key == model_key(scene, &DagOptions::default()) && fits(&g, GeoView::of(scene)) => {
                    for (info, dag) in meshes.iter_mut().zip(&g.meshes) {
                        if let Some(d) = dag {
                            info.clusters = d.clusters.len();
                            info.dag_bytes = d.byte_len();
                            info.levels = d.header.levels;
                            info.skinned = d.header.flags & crate::cluster_dag::FLAG_SKINNED != 0;
                            info.blendshapes = d.header.morph_targets;
                        }
                    }
                    CacheState::Valid
                }
                Ok(_) => CacheState::Stale,
            }
        };
    }
    status.meshes = meshes;
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_of(meshes: Vec<zestengine_render::MeshData>) -> Scene {
        Scene {
            meshes,
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        }
    }

    fn big_scene() -> Scene {
        // The engine's UV sphere is above the threshold; a cube below it.
        scene_of(vec![crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere), crate::procedural::primitive_mesh(crate::PrimitiveShape::Cube)])
    }

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zest_geo_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_geo_source_gives_the_same_key_and_dags_as_the_full_scene() {
        let mut scene = big_scene();
        // A blendshape on the big mesh and one on the small mesh (ignored by the cook).
        let target = crate::MorphTarget { name: "smile".into(), deltas: vec![(3, [0.1, 0.0, 0.0], [0.0; 3])] };
        scene.morphs = vec![vec![target.clone()], vec![target]];
        let source = GeoSource::of(&scene);
        let opts = DagOptions::default();
        assert_eq!(model_key_view(source.view(), &opts), model_key(&scene, &opts));
        let (full, partial) = (build_model(&scene, &opts), build_model_view(source.view(), &opts));
        assert!(full.meshes[0].is_some() && partial.meshes[1].is_none());
        assert_eq!(encode(&full), encode(&partial), "identical DAG output");
        assert!(fits(&partial, GeoView::of(&scene)), "and it fits the real scene");
        // The stored copy loads back through the same source.
        let dir = temp("source");
        let model = dir.join("m.glb");
        store(&model, &partial);
        assert_eq!(source.load_or_build(&model), full);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn only_big_meshes_get_a_dag_and_the_file_round_trips() {
        let scene = big_scene();
        let geo = build_model(&scene, &DagOptions::default());
        assert!(geo.meshes[0].is_some() && geo.meshes[1].is_none());
        geo.meshes[0].as_ref().unwrap().validate_against(&scene.meshes[0]).unwrap();
        assert_eq!(geo.meshes[0].as_ref().unwrap().clusters.iter().map(|c| c.section).max(), Some(0));
        let bytes = encode(&geo);
        assert_eq!(decode(&bytes).unwrap(), geo);
        assert_eq!(geo.dag_bytes(), geo.meshes[0].as_ref().unwrap().to_bytes().len());
    }

    #[test]
    fn rebuild_reports_progress_and_inspect_sees_the_file() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let dir = temp("rebuild");
        let src = dir.join("m.glb");
        let scene = big_scene();
        let st = inspect(&src, &scene);
        assert_eq!(st.state, CacheState::Missing);
        assert!(st.meshes[0].wants_dag && !st.meshes[1].wants_dag);
        let top = AtomicU32::new(0);
        let r = rebuild(&src, &scene, &DagOptions::default(), &|f| {
            top.fetch_max((f * 1000.0) as u32, Ordering::Relaxed);
        }, &|| false)
        .expect("not cancelled");
        assert_eq!(top.load(Ordering::Relaxed), 1000, "progress reaches 100%");
        assert!(r.mesh_secs[0] >= 0.0 && r.geo.meshes[0].is_some());
        let st = inspect(&src, &scene);
        assert_eq!(st.state, CacheState::Valid);
        assert!(st.file_bytes > 0 && st.meshes[0].clusters > 0 && st.meshes[0].levels > 0 && st.meshes[0].dag_bytes > 0);
        // Damaged file: reported, never trusted.
        let p = geo_path(&src).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[20] ^= 0x5a;
        std::fs::write(&p, bytes).unwrap();
        assert!(matches!(inspect(&src, &scene).state, CacheState::Invalid(_)));
        // Cancelled: nothing written over the damaged file.
        assert!(rebuild(&src, &scene, &DagOptions::default(), &|_| {}, &|| true).is_none());
        assert!(matches!(inspect(&src, &scene).state, CacheState::Invalid(_)));
        // Other geometry: stale.
        rebuild(&src, &scene, &DagOptions::default(), &|_| {}, &|| false).unwrap();
        let other = scene_of(vec![crate::procedural::primitive_mesh(crate::PrimitiveShape::Cube), crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere)]);
        assert_eq!(inspect(&src, &other).state, CacheState::Stale);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A closed, slightly bumpy sphere of 2 * n * n triangles.
    fn bumpy_sphere(n: u32, seed: f32) -> zestengine_render::MeshData {
        let mut vertices = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                let (u, v) = (std::f32::consts::PI * i as f32 / n as f32, std::f32::consts::TAU * j as f32 / n as f32);
                let r = 1.0 + 0.04 * (7.0 * u + seed).sin() * (5.0 * v).cos();
                let p = [r * u.sin() * v.cos(), r * u.cos(), r * u.sin() * v.sin()];
                vertices.push(zestengine_render::Vertex { position: p, normal: p, ..Default::default() });
            }
        }
        let mut indices = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let a = i * (n + 1) + j;
                indices.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false }
    }

    fn hash_of(bytes: &[u8]) -> u128 {
        let mut h = Hasher128::new();
        h.update(bytes);
        h.finish()
    }

    /// Meshes of very different sizes, some under the threshold.
    fn mixed_scene() -> Scene {
        scene_of(vec![bumpy_sphere(40, 0.0), bumpy_sphere(12, 1.0), bumpy_sphere(70, 2.0), bumpy_sphere(5, 3.0), bumpy_sphere(33, 4.0), bumpy_sphere(48, 5.0)])
    }

    #[test]
    fn the_parallel_build_is_byte_identical_to_the_serial_one_on_any_thread_count() {
        let scene = mixed_scene();
        let opts = DagOptions::default();
        // The previous implementation: one mesh after the other.
        let serial: Vec<Option<ClusterDag>> = scene
            .meshes
            .iter()
            .enumerate()
            .map(|(i, m)| wants_dag(m).then(|| crate::cluster_dag::build_cluster_dag(m, &[], &DagOptions { section: i as u32, threads: 2, ..opts.clone() }).unwrap()))
            .collect();
        let want = encode(&ModelGeometry { key: model_key(&scene, &opts), meshes: serial });
        assert!(want.len() > 1000);
        for threads in [1, 2, 3, 4] {
            let got = encode(&build_model(&scene, &DagOptions { threads, ..opts.clone() }));
            assert_eq!(hash_of(&got), hash_of(&want), "{threads} threads");
            assert_eq!(got, want);
        }
    }

    #[test]
    fn inner_threads_follow_the_triangle_share() {
        assert_eq!(inner_threads(16, 1_000_000, 1_000_000), 16);
        assert_eq!(inner_threads(16, 500_000, 1_000_000), 8);
        assert_eq!(inner_threads(16, 100, 1_000_000), 1, "small meshes get one thread");
        assert_eq!(inner_threads(1, 5, 10), 1);
        assert_eq!(inner_threads(4, 0, 0), 1, "an empty set does not divide by zero");
    }

    #[test]
    fn progress_counts_meshes_and_triangles_and_ends_complete() {
        let scene = mixed_scene();
        let want_tris: u64 = scene.meshes.iter().filter(|m| wants_dag(m)).map(|m| (m.indices.len() / 3) as u64).sum();
        let want_meshes = scene.meshes.iter().filter(|m| wants_dag(m)).count() as u32;
        assert!(want_meshes >= 4 && want_meshes < scene.meshes.len() as u32, "one mesh is under the threshold");
        let progress = BuildProgress::default();
        assert_eq!(progress.snapshot().fraction(), 0.0);
        let seen = Mutex::new(Vec::new());
        let built = build_meshes(GeoView::of(&scene), &DagOptions { threads: 4, ..Default::default() }, &progress, &|| false, &|f| seen.lock().unwrap().push(f));
        assert!(!built.cancelled);
        let snap = progress.snapshot();
        assert_eq!((snap.meshes_done, snap.meshes_total, snap.tris_done, snap.tris_total), (want_meshes, want_meshes, want_tris, want_tris));
        assert_eq!(snap.fraction(), 1.0);
        assert!(snap.label().starts_with(&format!("{want_meshes}/{want_meshes} meshes")), "{}", snap.label());
        let seen = seen.into_inner().unwrap();
        assert!(!seen.is_empty() && seen.iter().all(|f| (0.0..=1.0).contains(f)));
        assert_eq!(*seen.last().unwrap(), 1.0);
        assert!(built.secs.iter().zip(&scene.meshes).all(|(s, m)| wants_dag(m) == (*s > 0.0)));
    }

    #[test]
    fn a_fraction_is_weighted_by_triangles_and_clamped() {
        let s = BuildSnapshot { meshes_done: 1, meshes_total: 3, tris_done: 250, tris_total: 1000 };
        assert_eq!(s.fraction(), 0.25);
        assert_eq!(BuildSnapshot { tris_done: 5000, tris_total: 1000, ..s }.fraction(), 1.0);
        assert_eq!(BuildSnapshot::default().fraction(), 0.0);
        assert_eq!(BuildSnapshot { meshes_done: 12, meshes_total: 180, tris_done: 3_200_000, tris_total: 20_100_000 }.label(), "12/180 meshes, 3.2M/20.1M triangles");
    }

    #[test]
    fn cancelling_stops_the_build_and_stores_nothing() {
        let dir = temp("cancel");
        let src = dir.join("m.glb");
        let scene = mixed_scene();
        let progress = BuildProgress::default();
        progress.cancel();
        let geo = load_or_build_tracked(&src, &scene, &progress);
        assert!(geo.is_empty(), "cancelled before the first mesh");
        assert!(!geo_path(&src).unwrap().exists(), "a cancelled build is not stored");
        assert!(progress.snapshot().meshes_done < progress.snapshot().meshes_total);
        // Cancelled by the hook while running: stops early, and a fresh handle builds normally.
        let stop_after_first = AtomicU32::new(0);
        let built = build_meshes(GeoView::of(&scene), &DagOptions { threads: 1, ..Default::default() }, &BuildProgress::default(), &|| stop_after_first.fetch_add(1, Ordering::Relaxed) >= 2, &|_| {});
        assert!(built.cancelled && built.dags.iter().flatten().count() < scene.meshes.iter().filter(|m| wants_dag(m)).count());
        let fresh = BuildProgress::default();
        let geo = load_or_build_tracked(&src, &scene, &fresh);
        assert!(!geo.is_empty() && geo_path(&src).unwrap().is_file());
        assert_eq!(fresh.snapshot().fraction(), 1.0);
        // A cache hit leaves the progress alone.
        let hit = BuildProgress::default();
        assert_eq!(load_or_build_tracked(&src, &scene, &hit), geo);
        assert_eq!(hit.snapshot(), BuildSnapshot::default());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `ZESTENGINE_COLOUR_CENSUS="a.fbx;b.glb" cargo test -p zestengine-assets --lib -- --ignored --nocapture colour_census`
    /// Prints, per model, the triangle share that would stay off the cluster path for colour / lightmap streams
    /// (the measurement behind docs/research/cluster-colour-streams.md). Read only.
    #[test]
    #[ignore]
    fn colour_census() {
        let Some(list) = std::env::var_os("ZESTENGINE_COLOUR_CENSUS") else { return };
        for path in list.to_string_lossy().split(';').filter(|p| !p.is_empty()) {
            let Ok(scene) = crate::load_model(path) else {
                eprintln!("[census] {path}: not loadable");
                continue;
            };
            let (mut all, mut big, mut colour, mut colour_big, mut old_big, mut lm, mut skinned) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
            let mut meshes_big = 0;
            for m in &scene.meshes {
                let t = m.indices.len() / 3;
                all += t;
                let col = m.vertices.iter().any(|v| v.color != [1.0; 4]);
                let sk = m.vertices.iter().any(|v| v.weights.iter().any(|&w| w > 0.0));
                if col {
                    colour += t;
                }
                if m.has_lightmap_uv {
                    lm += t;
                }
                if sk {
                    skinned += t;
                }
                if wants_dag(m) {
                    big += t;
                    meshes_big += 1;
                    if col || m.has_lightmap_uv {
                        old_big += t;
                    }
                    if col {
                        colour_big += t;
                    }
                }
            }
            eprintln!(
                "[census] {path}: {} meshes ({meshes_big} DAG-sized), {all} tris; coloured {colour} ({:.0}%), lightmap-UV {lm} ({:.0}%), skinned {skinned}; of DAG-sized tris {big}: {colour_big} refused for streams now ({:.0}%; builder version 2: only real vertex colours refuse); before (version 1, colours or a lightmap UV set): {old_big} ({:.0}%)",
                scene.meshes.len(),
                100.0 * colour as f64 / all.max(1) as f64,
                100.0 * lm as f64 / all.max(1) as f64,
                100.0 * colour_big as f64 / big.max(1) as f64,
                100.0 * old_big as f64 / big.max(1) as f64
            );
        }
    }

    /// `cargo test -p zestengine-assets --release -- --ignored --nocapture geo_build_bench`
    /// Optional: `ZESTENGINE_GEO_BENCH_MODEL=<fbx path>` times a real model too (read only, nothing stored).
    #[test]
    #[ignore]
    fn geo_build_bench() {
        let time = |label: &str, scene: &Scene| {
            let tris: usize = scene.meshes.iter().filter(|m| wants_dag(m)).map(|m| m.indices.len() / 3).sum();
            // Capped: the bench must not load the machine.
            let opts = DagOptions { threads: 4, ..DagOptions::default() };
            let t = std::time::Instant::now();
            let serial: Vec<Option<ClusterDag>> = scene
                .meshes
                .iter()
                .enumerate()
                .map(|(i, m)| wants_dag(m).then(|| crate::cluster_dag::build_cluster_dag(m, scene.morphs.get(i).map_or(&[][..], Vec::as_slice), &DagOptions { section: i as u32, ..opts.clone() }).unwrap()))
                .collect();
            let before = t.elapsed().as_secs_f32();
            let before_bytes = encode(&ModelGeometry { key: model_key(scene, &opts), meshes: serial });
            let t = std::time::Instant::now();
            let geo = build_model(scene, &opts);
            let after = t.elapsed().as_secs_f32();
            let after_bytes = encode(&geo);
            eprintln!(
                "[geo-bench] {label}: {} meshes, {:.2}M triangles, {} threads: serial-over-meshes {before:.2} s -> work-stealing {after:.2} s ({:.2}x); identical bytes: {} (hash {:032x})",
                scene.meshes.iter().filter(|m| wants_dag(m)).count(),
                tris as f64 / 1e6,
                crate::cluster_dag::thread_count(&opts),
                before / after.max(1e-6),
                before_bytes == after_bytes,
                hash_of(&after_bytes)
            );
            assert_eq!(before_bytes, after_bytes);
        };
        // 5M triangles: one 1M, four 500k, forty 50k.
        let mut meshes = vec![bumpy_sphere(707, 0.0)];
        meshes.extend((0..4).map(|k| bumpy_sphere(500, k as f32)));
        meshes.extend((0..40).map(|k| bumpy_sphere(158, 10.0 + k as f32)));
        time("synthetic", &scene_of(meshes));
        if let Some(path) = std::env::var_os("ZESTENGINE_GEO_BENCH_MODEL") {
            let t = std::time::Instant::now();
            let scene = crate::fbx_loader::load_fbx(&path).expect("model");
            eprintln!("[geo-bench] imported in {:.1} s", t.elapsed().as_secs_f32());
            time(&path.to_string_lossy(), &scene);
        }
    }

    #[test]
    fn building_twice_gives_identical_files() {
        let scene = big_scene();
        let a = encode(&build_model(&scene, &DagOptions::default()));
        let b = encode(&build_model(&scene, &DagOptions { threads: 3, ..Default::default() }));
        assert_eq!(a, b);
    }

    #[test]
    fn corrupt_truncated_and_foreign_files_are_rejected() {
        let geo = build_model(&big_scene(), &DagOptions::default());
        let bytes = encode(&geo);
        assert!(decode(&[]).is_err());
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(decode(&bytes[..bytes.len() / 2]).is_err());
        for at in [0, 5, 9, 14, 40, bytes.len() / 3, bytes.len() / 2, bytes.len() - 20] {
            let mut bad = bytes.clone();
            bad[at] ^= 0x5a;
            assert!(decode(&bad).is_err(), "flip at {at}");
        }
        // A format change inside a valid checksum is still refused.
        let mut other = bytes[..bytes.len() - 16].to_vec();
        other[4] ^= 1;
        let mut h = Hasher128::new();
        h.update(&other);
        other.extend_from_slice(&h.finish().to_le_bytes());
        assert!(decode(&other).unwrap_err().contains("version"));
    }

    #[test]
    fn well_formed_but_invalid_dags_are_rejected_by_validation() {
        let mut geo = build_model(&big_scene(), &DagOptions::default());
        let ok = encode(&geo);
        assert!(decode(&ok).is_ok());
        // Each of these keeps the checksum valid (it is computed over the bad data).
        let d = geo.meshes[0].as_mut().unwrap();
        d.clusters[0].triangle_count = 129;
        assert!(decode(&encode(&geo)).is_err(), "too many triangles");
        let d = geo.meshes[0].as_mut().unwrap();
        d.clusters[0].triangle_count = 1;
        let last = d.clusters.len() - 1;
        d.clusters[last].parent_lod_error = 0.0;
        d.clusters[last].lod_error = 1.0;
        assert!(decode(&encode(&geo)).is_err(), "error shrinks going up");
        let mut geo = build_model(&big_scene(), &DagOptions::default());
        let d = geo.meshes[0].as_mut().unwrap();
        let o = d.clusters[0].triangle_offset as usize;
        d.triangles[o] = 250;
        assert!(decode(&encode(&geo)).is_err(), "triangle index beyond the cluster's vertices");
        let mut geo = build_model(&big_scene(), &DagOptions::default());
        let d = geo.meshes[0].as_mut().unwrap();
        d.links[3].parent_group = d.links[3].group;
        assert!(decode(&encode(&geo)).is_err(), "a cluster consumed by its own group");
    }

    #[test]
    fn stored_then_loaded_by_key_and_rebuilt_when_the_mesh_changes() {
        let dir = temp("store");
        let src = dir.join("m.glb");
        std::fs::write(&src, b"x").unwrap();
        let mut scene = big_scene();
        let key = model_key(&scene, &DagOptions::default());
        assert!(load(&src, key).is_none());
        let first = load_or_build(&src, &scene);
        assert!(geo_path(&src).unwrap().is_file(), "{}", geo_path(&src).unwrap().display());
        assert_eq!(load(&src, key).as_ref(), Some(&first));
        // Served from disk the second time.
        assert_eq!(load_or_build(&src, &scene), first);
        // Another mesh: another key, rebuilt and replaced.
        scene.meshes[0].vertices[5].position[0] += 0.25;
        let key2 = model_key(&scene, &DagOptions::default());
        assert_ne!(key, key2);
        assert!(load(&src, key2).is_none());
        let second = load_or_build(&src, &scene);
        assert_ne!(second.key, first.key);
        assert_eq!(load(&src, key2).as_ref(), Some(&second));
        // A damaged file is a miss and is rebuilt.
        let path = geo_path(&src).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[100] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(load(&src, key2).is_none());
        assert_eq!(load_or_build(&src, &scene).key, key2);
        assert!(load(&src, key2).is_some());
        // Changed builder settings are another key.
        assert_ne!(model_key(&scene, &DagOptions { group_clusters: 12, ..Default::default() }), key2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cooked_file_installs_beside_its_model_and_bad_ones_do_not() {
        let dir = temp("cook");
        let src = dir.join("m.glb");
        let scene = big_scene();
        let (geo, blob) = cook(&src, &scene).expect("a big mesh");
        let other = dir.join("sub").join("n.glb");
        installed_ok(&other, &blob, &geo);
        let mut bad = blob.clone();
        bad[60] ^= 1;
        let third = dir.join("o.glb");
        assert!(install(&third, &bad).is_err());
        assert!(!geo_path(&third).unwrap().exists());
        // The Android folder.
        let root = dir.join("apk");
        let shipped = root.join(cooked_geocache_rel("models/a.glb"));
        std::fs::create_dir_all(shipped.parent().unwrap()).unwrap();
        std::fs::write(&shipped, &blob).unwrap();
        let junk = root.join(COOKED_GEOCACHE_DIR).join("stray.bin");
        std::fs::write(&junk, b"?").unwrap();
        let (n, warnings) = seed_cooked_geocaches(&root);
        assert_eq!(n, 1);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(geo_path(&root.join("models/a.glb")).unwrap().is_file());
        assert!(!root.join(COOKED_GEOCACHE_DIR).exists());
        assert_eq!(parse_cooked_geocache_rel("zestengine_geocache/a/b.glb.zgeo"), Some("a/b.glb"));
        assert_eq!(parse_cooked_geocache_rel("zestengine_geocache/../b.glb.zgeo"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn installed_ok(path: &Path, blob: &[u8], geo: &ModelGeometry) {
        assert_eq!(&install(path, blob).unwrap(), geo);
        assert_eq!(std::fs::read(geo_path(path).unwrap()).unwrap(), blob);
    }

    /// A cache cooked by an older builder (a previous version) is rejected by every
    /// reader and rebuilt once, never mis-read.
    #[test]
    fn caches_from_the_previous_builder_version_are_rejected_and_rebuilt() {
        assert_eq!(GEO_BUILDER_VERSION, 5, "the seam-relaxed retry / piece pruning / triangle stall detection bumped the builder version");
        assert_eq!(GEO_MIN_TRIANGLES as u32, zestengine_render::ZESTLET_MIN_TRIANGLES, "the render crate's coverage view mirrors the DAG threshold");
        let scene = big_scene();
        let dir = temp("oldver");
        let src = dir.join("m.glb");
        let geo = load_or_build(&src, &scene);
        let path = geo_path(&src).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[8..12], &GEO_BUILDER_VERSION.to_le_bytes());
        // The same file as a version 1 builder wrote it: valid checksum, old version number.
        let body = bytes.len() - 16;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        let mut h = Hasher128::new();
        h.update(&bytes[..body]);
        bytes[body..].copy_from_slice(&h.finish().to_le_bytes());
        assert!(decode(&bytes).is_err(), "decode refuses the old builder version");
        std::fs::write(&path, &bytes).unwrap();
        assert!(!matches!(inspect(&src, &scene).state, CacheState::Valid), "the inspector reports it as not valid");
        // The next load rebuilds and stores the current version.
        let rebuilt = load_or_build(&src, &scene);
        assert_eq!(rebuilt, geo);
        assert_eq!(&std::fs::read(&path).unwrap()[8..12], &GEO_BUILDER_VERSION.to_le_bytes());
        assert_eq!(inspect(&src, &scene).state, CacheState::Valid);
        let _ = std::fs::remove_dir_all(dir);
    }
}
