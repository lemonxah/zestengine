//! Import artifact cache for models: the result of running an importer
//! (parse, triangulate, convert units, extract collision, lightmap-UV marker)
//! is stored next to the source and reused while nothing that could change it
//! has changed, so loading a scene stops re-parsing every FBX / glTF.
//!
//! The key is a 128-bit hash of (source bytes, the importer-affecting
//! [`ModelImport`] settings, [`IMPORTER_VERSION`], the registered import-hook
//! versions, the lightmap-UV marker). Files the importer read besides the
//! source (glTF buffers and images, external FBX textures) are recorded in the
//! entry with their hashes and re-checked on a hit, so editing a texture that
//! sits beside a model also invalidates it. Everything applied *after* the
//! importer (the `.meta` rig, sub-asset ids) is deliberately outside the key:
//! editing those does not re-parse the file.
//!
//! Layout: `<dir>/.zestengine_modelcache/<file>.<key>.zmc`, one entry per
//! source (writing a new one removes the stale ones). A missing, corrupt or
//! unwritable cache is never an error; the model is simply imported.

use std::cell::RefCell;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use glam::{Mat4, Quat, Vec3};
use zestengine_render::{MaterialFeatures, MaterialParams, MeshData, TextureData, Vertex};

use crate::collision::{CollisionKind, CollisionPiece};
use crate::skeleton::{AnimChannel, AnimationClip, ChannelExtra, ChannelPath, Joint, PackedValues, Skeleton, Tangents};
use crate::vrm::{SpringChain, SpringImport};
use crate::{Instance, MeshSlot, ModelImport, MorphTarget, Scene, SceneMaterial};

/// Bump when an importer (FBX, glTF, collision extraction, lightmap UVs, ...)
/// changes what it produces, so old entries are not reused.
/// v2: the mesh pipeline (indexed, welded, optimised FBX; tangents; glTF honours
/// ModelImport), source-frame-rate FBX animation, cameras and lights.
/// v3: MikkTSpace tangents, normals mode and weighting, UV mask, mesh
/// compression, quad tables, forced 16 bit splitting, FBX node visibility.
/// v4: FBX takes cut into clips by the Animation settings (reduction, loop
/// pose), blendshape curves and in-between shapes, blendshape normals.
/// v5: animation-only FBX files (skeleton from the node hierarchy), clip
/// playback settings, initial blendshape weights.
/// v6: glTF `KHR_materials_transmission` / `KHR_materials_volume` on imported materials.
/// v7: Bake Animations, Keep Tangents, quantised curves, constraints and custom properties.
/// v8: the non-bone ancestors of the bones (an `Armature` node) become container joints of the skeleton.
/// v9: skinned instances keep the file's own transform (the container basis is taken out of `Skeleton::world_matrices`).
pub const IMPORTER_VERSION: u32 = 9;
/// Version of the on-disk entry layout below.
/// v2: textures are zstd-compressed (v1 stored raw RGBA, hundreds of MB per model).
/// v3: imported cameras and lights.
/// v4: the hook signature the artifact was built with, and hook-assigned materials.
/// v7: the file's takes and the per-clip import info (blendshape curves, loop data).
/// v8: the clip playback settings and the initial blendshape weights.
/// v10: Hermite tangents and 16-bit packed channels, shared channel times, constraints / custom properties.
/// v11: the skeleton's container joints.
const FORMAT_VERSION: u32 = 11;
/// zstd level for texture pixels: fast to write, much faster to read back.
const TEXTURE_ZSTD_LEVEL: i32 = 1;
/// An entry larger than this is not written (the model just re-imports).
const MAX_ENTRY_BYTES: u64 = 512 << 20;
/// Total size of a cache directory; the oldest entries of other sources are
/// pruned past it. `ZESTENGINE_MODEL_CACHE_MAX_MB` overrides.
const MAX_TOTAL_BYTES: u64 = 2 << 30;
const MAGIC: &[u8; 4] = b"ZMC1";
/// Directory (beside the source) holding the entries.
pub const CACHE_DIR: &str = ".zestengine_modelcache";
/// Entry file extension.
pub const CACHE_EXTENSION: &str = "zmc";

/// What happened to the cache for one load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOutcome {
    /// The stored artifact was reused; the importer did not run.
    Hit,
    /// No usable entry; the importer ran and a new entry was written.
    Miss,
    /// The caller asked for a rebuild (explicit reimport); the importer ran.
    Rebuilt,
    /// Caching is off (`ZESTENGINE_NO_MODEL_CACHE`) or the path has no directory.
    Disabled,
}

/// Outcome and timings of one cached load, for logs and tests.
#[derive(Clone, Debug)]
pub struct ImportReport {
    pub outcome: CacheOutcome,
    /// Hex key the entry was looked up under (empty when disabled).
    pub key: String,
    /// Hashing the inputs (source, settings, deps).
    pub key_time: Duration,
    /// Reading the entry (hit) or running the importer (otherwise).
    pub load_time: Duration,
}

impl ImportReport {
    pub fn total(&self) -> Duration {
        self.key_time + self.load_time
    }
}

// --- hooks ------------------------------------------------------------------
//
// The hooks an import runs (`import_hooks::hooks_for`) are part of every key:
// their `(name#order, version)` list, in run order. Adding, removing, reordering
// or version-bumping a hook re-imports; a closed gate (player, UGC path) hashes
// as no hooks.

// --- dependency recording ---------------------------------------------------

thread_local! {
    static DEPS: RefCell<Option<Vec<PathBuf>>> = const { RefCell::new(None) };
}

/// Called by an importer when it reads a file besides the source (an external
/// texture). No-op outside a cached load.
pub fn record_dependency(path: &Path) {
    DEPS.with(|d| {
        if let Some(list) = d.borrow_mut().as_mut() {
            list.push(path.to_path_buf());
        }
    });
}

// --- hashing ----------------------------------------------------------------

/// Fast non-cryptographic 128-bit hash (two multiply-rotate lanes over 8-byte
/// words). Collision resistance against accidents is all a cache key needs.
pub(crate) struct Hasher128 {
    a: u64,
    b: u64,
    pending: [u8; 8],
    npending: usize,
    len: u64,
}

const P1: u64 = 0x9E37_79B9_7F4A_7C15;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;

impl Hasher128 {
    pub(crate) fn new() -> Self {
        Self { a: 0x243F_6A88_85A3_08D3, b: 0x1319_8A2E_0370_7344, pending: [0; 8], npending: 0, len: 0 }
    }

    fn word(&mut self, w: u64) {
        self.a = (self.a ^ w).wrapping_mul(P1).rotate_left(29) ^ self.b;
        self.b = (self.b.wrapping_add(w)).wrapping_mul(P2).rotate_left(31) ^ self.a;
    }

    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        if self.npending > 0 {
            let take = (8 - self.npending).min(bytes.len());
            self.pending[self.npending..self.npending + take].copy_from_slice(&bytes[..take]);
            self.npending += take;
            bytes = &bytes[take..];
            if self.npending == 8 {
                let w = u64::from_le_bytes(self.pending);
                self.word(w);
                self.npending = 0;
            }
        }
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks.iter() {
            self.word(u64::from_le_bytes(*c));
        }
        self.pending[..rest.len()].copy_from_slice(rest);
        self.npending += rest.len();
    }

    /// A length-prefixed field, so adjacent fields can't run together.
    pub(crate) fn field(&mut self, bytes: &[u8]) {
        self.update(&(bytes.len() as u64).to_le_bytes());
        self.update(bytes);
    }

    pub(crate) fn finish(mut self) -> u128 {
        let mut tail = [0u8; 8];
        tail[..self.npending].copy_from_slice(&self.pending[..self.npending]);
        self.word(u64::from_le_bytes(tail) ^ self.len);
        self.word(self.len);
        (u128::from(self.a) << 64) | u128::from(self.b)
    }
}

fn hash_file(path: &Path) -> Option<u64> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = Hasher128::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Some(h.finish() as u64)
}

/// The importer-affecting part of the settings, canonicalised. The rig, the id
/// table and the material / texture remap tables are applied after import and
/// are not part of the artifact.
fn settings_string(m: &ModelImport) -> String {
    let mut m = m.clone();
    m.validate();
    format!(
        "scale={:08x};materials={};flip_uv={};recalc_normals={};smooth={:08x};bake_axis={};blend_shapes={};cameras={};lights={};weld={};quads={};optimize={};tangents={:?};lightmap_uv={};normals={:?};weighting={:?};visibility={};uv_mask={};compression={:?};index_format={:?};animation={};blend_shapes={}",
        m.scale.to_bits(),
        m.import_materials,
        m.flip_uv,
        m.recalculate_normals,
        m.smoothing_angle.to_bits(),
        m.bake_axis_conversion,
        m.import_blend_shapes,
        m.import_cameras,
        m.import_lights,
        m.weld_vertices,
        m.keep_quads,
        m.optimize_mesh,
        m.tangents,
        m.generate_lightmap_uvs,
        m.normals,
        m.normal_weighting,
        m.import_visibility,
        m.uv_mask,
        m.mesh_compression,
        // Only a forced 16 bit format changes the artifact (it splits big meshes).
        m.index_format == crate::IndexFormat::UInt16,
        serde_json::to_string(&m.animation).unwrap_or_default(),
        serde_json::to_string(&m.blend_shapes).unwrap_or_default(),
    )
}

fn marker_state(path: &Path) -> Vec<u8> {
    std::fs::read(crate::lightmap_uv_marker_path(path)).map(|mut b| {
        b.insert(0, 1);
        b
    }).unwrap_or_else(|_| vec![0])
}

/// Hex cache key for `path` imported under `model`. `None` when the source
/// can't be read.
pub fn cache_key(path: &Path, model: &ModelImport) -> Option<String> {
    key_with_hooks(path, model, &crate::import_hooks::key_entries_for(path))
}

fn key_with_hooks(path: &Path, model: &ModelImport, hooks: &[(String, u32)]) -> Option<String> {
    let mut h = Hasher128::new();
    h.field(&IMPORTER_VERSION.to_le_bytes());
    h.field(&FORMAT_VERSION.to_le_bytes());
    h.field(path.extension()?.to_string_lossy().to_lowercase().as_bytes());
    h.field(settings_string(model).as_bytes());
    h.field(&marker_state(path));
    for (name, version) in hooks {
        h.field(name.as_bytes());
        h.field(&version.to_le_bytes());
    }
    h.field(&hash_file(path)?.to_le_bytes());
    Some(format!("{:032x}", h.finish()))
}

/// The import-hook signature the stored artifact of `source` was built with
/// (`Some("")` = no hooks), read from the entry's header without loading it.
/// `None` when there is no readable entry. When several exist the newest wins.
pub fn stored_hook_signature(source: &Path) -> Option<String> {
    let dir = source.parent()?.join(CACHE_DIR);
    let prefix = format!("{}.", source.file_name()?.to_string_lossy());
    let newest = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(&prefix) && n.ends_with(CACHE_EXTENSION)
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH))?;
    let mut head = Vec::new();
    std::fs::File::open(newest.path()).ok()?.take(64 * 1024).read_to_end(&mut head).ok()?;
    if head.len() < 16 || &head[..4] != MAGIC || u32::from_le_bytes(head[4..8].try_into().ok()?) != FORMAT_VERSION {
        return None;
    }
    let n = u64::from_le_bytes(head[8..16].try_into().ok()?) as usize;
    String::from_utf8(head.get(16..16usize.checked_add(n)?)?.to_vec()).ok()
}

/// `<dir>/.zestengine_modelcache/<file>.<key>.zmc`.
pub fn cache_path(path: &Path, key: &str) -> Option<PathBuf> {
    let name = path.file_name()?.to_string_lossy();
    Some(path.parent()?.join(CACHE_DIR).join(format!("{name}.{key}.{CACHE_EXTENSION}")))
}

fn enabled() -> bool {
    std::env::var_os("ZESTENGINE_NO_MODEL_CACHE").is_none()
}

// --- the cached load --------------------------------------------------------

/// Run `import` for `path` through the cache. With `force` the stored entry is
/// ignored and replaced (an explicit reimport).
pub fn load_cached(
    path: &Path,
    model: &ModelImport,
    force: bool,
    import: impl FnOnce() -> Result<Scene>,
) -> Result<(Scene, ImportReport)> {
    load_cached_with(path, model, force, &crate::import_hooks::hooks_for(path), import)
}

/// [`load_cached`] for an explicit hook set: its entries are keyed and its
/// signature is stored in the entry. `import` must run exactly these hooks.
pub fn load_cached_with(
    path: &Path,
    model: &ModelImport,
    force: bool,
    hooks: &crate::import_hooks::HookSet,
    import: impl FnOnce() -> Result<Scene>,
) -> Result<(Scene, ImportReport)> {
    let t0 = Instant::now();
    let key = if enabled() { key_with_hooks(path, model, &hooks.key_entries()) } else { None };
    let Some(key) = key else {
        let t1 = Instant::now();
        let scene = import()?;
        let report = ImportReport { outcome: CacheOutcome::Disabled, key: String::new(), key_time: t1 - t0, load_time: t1.elapsed() };
        return Ok((scene, report));
    };
    let key_time = t0.elapsed();
    let entry = cache_path(path, &key);
    let t1 = Instant::now();
    if !force
        && let Some(entry) = &entry
        && let Ok(bytes) = std::fs::read(entry)
    {
        match decode(&bytes, path) {
            Ok(scene) => {
                let report = ImportReport { outcome: CacheOutcome::Hit, key, key_time, load_time: t1.elapsed() };
                tracing::info!("model {}: import cache hit ({:.1} ms, key {:.1} ms)", path.display(), ms(report.load_time), ms(key_time));
                return Ok((scene, report));
            }
            Err(e) => tracing::debug!("model cache entry {} unusable: {e:#}", entry.display()),
        }
    }
    DEPS.with(|d| *d.borrow_mut() = Some(Vec::new()));
    let imported = import();
    let mut deps = DEPS.with(|d| d.borrow_mut().take()).unwrap_or_default();
    let scene = imported?;
    let load_time = t1.elapsed();
    deps.extend(gltf_dependencies(path));
    if let Some(entry) = &entry {
        store(entry, path, &scene, &deps, &hooks.signature());
    }
    let outcome = if force { CacheOutcome::Rebuilt } else { CacheOutcome::Miss };
    tracing::info!("model {}: imported ({outcome:?}, {:.1} ms, key {:.1} ms)", path.display(), ms(load_time), ms(key_time));
    Ok((scene, ImportReport { outcome, key, key_time, load_time }))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Write `scene` as the entry for `source`, dropping older entries of the same
/// source. Failures (read-only install, full disk) are ignored.
fn store(entry: &Path, source: &Path, scene: &Scene, deps: &[PathBuf], hooks: &str) {
    let bytes = encode(scene, source, deps, hooks);
    if bytes.len() as u64 > MAX_ENTRY_BYTES {
        tracing::info!("model cache: {} entry is {:.0} MB, over the {} MB cap; not stored", source.display(), bytes.len() as f64 / 1e6, MAX_ENTRY_BYTES >> 20);
        return;
    }
    if let Some(dir) = entry.parent()
        && let Some(prefix) = source.file_name().map(|n| format!("{}.", n.to_string_lossy()))
        && let Ok(read) = std::fs::read_dir(dir)
    {
        for old in read.flatten() {
            let name = old.file_name().to_string_lossy().into_owned();
            if name.starts_with(&prefix) && name.ends_with(CACHE_EXTENSION) && old.path() != entry {
                let _ = std::fs::remove_file(old.path());
            }
        }
    }
    if let Err(e) = crate::write_atomic(entry, &bytes) {
        tracing::debug!("model cache not written: {e:#}");
        return;
    }
    if let Some(dir) = entry.parent() {
        let max = std::env::var("ZESTENGINE_MODEL_CACHE_MAX_MB").ok().and_then(|v| v.parse::<u64>().ok()).map_or(MAX_TOTAL_BYTES, |mb| mb << 20);
        prune_dir(dir, entry, max);
    }
}

/// Delete the oldest entries in `dir` (by modification time) until the total is
/// within `max_total`. `keep` (the entry just written) is never removed.
fn prune_dir(dir: &Path, keep: &Path, max_total: u64) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = read
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == CACHE_EXTENSION))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, path) in files {
        if total <= max_total {
            break;
        }
        if path != keep && std::fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }
}

/// Files a `.gltf` references besides itself (`.glb` / `.vrm` embed theirs).
fn gltf_dependencies(path: &Path) -> Vec<PathBuf> {
    if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gltf")) {
        return Vec::new();
    }
    let Ok(text) = std::fs::read(path) else { return Vec::new() };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&text) else { return Vec::new() };
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for list in ["buffers", "images"] {
        for item in json.get(list).and_then(|v| v.as_array()).into_iter().flatten() {
            let Some(uri) = item.get("uri").and_then(|u| u.as_str()) else { continue };
            if uri.starts_with("data:") {
                continue;
            }
            let p = dir.join(percent_decode(uri));
            if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            out.push(v);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// --- encoding ---------------------------------------------------------------

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn len(&mut self, n: usize) {
        self.u64(n as u64);
    }
    fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.0.extend_from_slice(b);
    }
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.f32(*x);
        }
    }
    fn opt(&mut self, v: Option<usize>) {
        match v {
            Some(i) => {
                self.u8(1);
                self.u64(i as u64);
            }
            None => self.u8(0),
        }
    }
    fn mat4(&mut self, m: &Mat4) {
        self.f32s(&m.to_cols_array());
    }
    fn vec3(&mut self, v: Vec3) {
        self.f32s(&v.to_array());
    }
    fn quat(&mut self, q: Quat) {
        self.f32s(&q.to_array());
    }
    fn json<T: serde::Serialize>(&mut self, v: &T) {
        self.str(&serde_json::to_string(v).unwrap_or_default());
    }
}

fn kind_code(k: CollisionKind) -> u8 {
    match k {
        CollisionKind::Convex => 0,
        CollisionKind::Box => 1,
        CollisionKind::Sphere => 2,
        CollisionKind::Capsule => 3,
    }
}

fn path_code(p: ChannelPath) -> u8 {
    match p {
        ChannelPath::Translation => 0,
        ChannelPath::Rotation => 1,
        ChannelPath::Scale => 2,
    }
}

pub(crate) fn encode(scene: &Scene, source: &Path, deps: &[PathBuf], hooks: &str) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.0.extend_from_slice(MAGIC);
    w.u32(FORMAT_VERSION);
    w.str(hooks);

    // Dependencies: (relative-to-source-dir flag, path, content hash).
    let dir = source.parent().unwrap_or(Path::new("."));
    let mut seen = std::collections::HashSet::new();
    let deps: Vec<&PathBuf> = deps.iter().filter(|d| seen.insert((*d).clone())).collect();
    w.len(deps.len());
    for d in deps {
        match d.strip_prefix(dir) {
            Ok(rel) => {
                w.bool(true);
                w.str(&rel.to_string_lossy());
            }
            Err(_) => {
                w.bool(false);
                w.str(&d.to_string_lossy());
            }
        }
        w.u64(hash_file(d).unwrap_or(0));
    }

    w.len(scene.meshes.len());
    for m in &scene.meshes {
        w.bytes(bytemuck::cast_slice(&m.vertices));
        w.bytes(bytemuck::cast_slice(&m.indices));
        w.bool(m.has_lightmap_uv);
    }
    w.len(scene.morphs.len());
    for targets in &scene.morphs {
        w.len(targets.len());
        for t in targets {
            w.str(&t.name);
            w.len(t.deltas.len());
            for (v, p, n) in &t.deltas {
                w.u32(*v);
                w.f32s(p);
                w.f32s(n);
            }
        }
    }
    w.len(scene.springs.chains.len());
    for c in &scene.springs.chains {
        w.str(&c.name);
        w.len(c.roots.len());
        for r in &c.roots {
            w.str(r);
        }
        w.f32(c.stiffness);
        w.f32(c.damping);
        w.f32s(&c.gravity);
        w.f32(c.radius);
        w.len(c.colliders.len());
        for r in &c.colliders {
            w.str(r);
        }
    }
    w.len(scene.springs.colliders.len());
    for c in &scene.springs.colliders {
        w.json(c);
    }
    w.len(scene.textures.len());
    for t in &scene.textures {
        w.u32(t.width);
        w.u32(t.height);
        w.bool(t.srgb);
        w.bool(t.hdr);
        w.u64(t.pixels.len() as u64);
        match zstd::bulk::compress(&t.pixels, TEXTURE_ZSTD_LEVEL) {
            Ok(c) => w.bytes(&c),
            // Can't happen for in-memory input; an empty blob decodes as a miss.
            Err(_) => w.bytes(&[]),
        }
    }
    w.len(scene.materials.len());
    for m in &scene.materials {
        w.str(&m.name);
        w.json(&m.params);
        w.json(&m.features);
        w.opt(m.albedo);
        w.opt(m.normal);
        w.opt(m.orm);
        w.opt(m.emission);
    }
    w.len(scene.instances.len());
    for i in &scene.instances {
        w.str(&i.name);
        w.mat4(&i.transform);
        w.len(i.slots.len());
        for s in &i.slots {
            w.u64(s.mesh as u64);
            w.u64(s.material as u64);
        }
    }
    w.len(scene.skeletons.len());
    for s in &scene.skeletons {
        w.len(s.joints.len());
        for j in &s.joints {
            w.str(&j.name);
            w.opt(j.parent);
            w.mat4(&j.inverse_bind);
            w.vec3(j.rest_translation);
            w.quat(j.rest_rotation);
            w.vec3(j.rest_scale);
        }
        w.len(s.containers.len());
        for &c in &s.containers {
            w.u64(c as u64);
        }
    }
    w.len(scene.animations.len());
    for a in &scene.animations {
        w.str(&a.name);
        w.f32(a.duration);
        w.len(a.channels.len());
        let mut prev_times: Option<&Vec<f32>> = None;
        for c in &a.channels {
            w.u64(c.joint as u64);
            w.u8(path_code(c.path));
            // Resampled channels of a clip share one time array: write it once.
            if !c.times.is_empty() && prev_times == Some(&c.times) {
                w.u8(1);
            } else {
                w.u8(0);
                w.len(c.times.len());
                w.f32s(&c.times);
            }
            prev_times = Some(&c.times);
            w.len(c.vec_values.len());
            for v in &c.vec_values {
                w.vec3(*v);
            }
            w.len(c.quat_values.len());
            for q in &c.quat_values {
                w.quat(*q);
            }
            let tangents = c.tangents();
            let packed = c.extra.as_ref().and_then(|e| e.packed.as_ref());
            w.u8(u8::from(tangents.is_some()) | u8::from(packed.is_some()) << 1);
            if let Some(t) = tangents {
                w.len(t.tan_in.len());
                for v in t.tan_in.iter().chain(&t.tan_out) {
                    w.vec3(*v);
                }
            }
            if let Some(p) = packed {
                w.u8(p.lanes);
                w.f32s(&p.min);
                w.f32s(&p.extent);
                w.len(p.codes.len());
                for code in &p.codes {
                    w.0.extend_from_slice(&code.to_le_bytes());
                }
            }
        }
    }
    w.len(scene.collision.len());
    for c in &scene.collision {
        w.str(&c.name);
        w.u8(kind_code(c.kind));
        w.opt(c.target);
        w.len(c.points.len());
        for p in &c.points {
            w.vec3(*p);
        }
    }
    w.len(scene.identity.mesh_names.len());
    for n in &scene.identity.mesh_names {
        w.str(n);
    }
    w.len(scene.identity.cameras.len());
    for c in &scene.identity.cameras {
        w.json(c);
    }
    w.len(scene.identity.lights.len());
    for l in &scene.identity.lights {
        w.json(l);
    }
    w.len(scene.identity.texture_names.len());
    for n in &scene.identity.texture_names {
        w.str(n);
    }
    w.len(scene.identity.quads.len());
    for per_mesh in &scene.identity.quads {
        w.len(per_mesh.len());
        for q in per_mesh {
            for v in q {
                w.u32(*v);
            }
        }
    }
    w.len(scene.identity.hidden_instances.len());
    for i in &scene.identity.hidden_instances {
        w.u64(*i as u64);
    }
    w.len(scene.identity.hook_material_assign.len());
    for a in &scene.identity.hook_material_assign {
        match a {
            Some(p) => {
                w.u8(1);
                w.str(p);
            }
            None => w.u8(0),
        }
    }
    w.json(&scene.identity.takes);
    w.json(&scene.identity.clip_info);
    w.json(&scene.identity.morph_weights);
    w.json(&scene.identity.constraints);
    w.json(&scene.identity.custom_props);
    w.json(&scene.identity.import_warnings);
    w.json(&scene.identity.optimize);
    w.len(scene.identity.material_volume_maps.len());
    for m in &scene.identity.material_volume_maps {
        w.opt(*m);
    }

    let mut h = Hasher128::new();
    h.update(&w.0);
    w.u64(h.finish() as u64);
    w.0
}

// --- decoding ---------------------------------------------------------------

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.b.len());
        let Some(end) = end else { bail!("truncated entry") };
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    /// A count, sanity-limited by the bytes left (every element is >= 1 byte).
    fn len(&mut self) -> Result<usize> {
        let n = self.u64()? as usize;
        if n > self.b.len() {
            bail!("implausible count {n}");
        }
        Ok(n)
    }
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.len()?;
        self.take(n)
    }
    fn string(&mut self) -> Result<String> {
        Ok(String::from_utf8(self.bytes()?.to_vec())?)
    }
    fn arr<const N: usize>(&mut self) -> Result<[f32; N]> {
        let mut a = [0.0; N];
        for x in &mut a {
            *x = self.f32()?;
        }
        Ok(a)
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let raw = self.take(n.checked_mul(4).ok_or_else(|| anyhow::anyhow!("overflow"))?)?;
        Ok(raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
    fn opt(&mut self) -> Result<Option<usize>> {
        Ok(if self.u8()? != 0 { Some(self.u64()? as usize) } else { None })
    }
    fn mat4(&mut self) -> Result<Mat4> {
        Ok(Mat4::from_cols_array(&self.arr::<16>()?))
    }
    fn vec3(&mut self) -> Result<Vec3> {
        Ok(Vec3::from(self.arr::<3>()?))
    }
    fn quat(&mut self) -> Result<Quat> {
        Ok(Quat::from_array(self.arr::<4>()?))
    }
    fn json<T: serde::de::DeserializeOwned>(&mut self) -> Result<T> {
        Ok(serde_json::from_str(&self.string()?)?)
    }
}

type TextureBlob<'a> = (u32, u32, bool, bool, usize, &'a [u8]);

fn inflate_textures(blobs: &[TextureBlob]) -> Result<Vec<TextureData>> {
    let one = |&(width, height, srgb, hdr, raw_len, c): &TextureBlob| -> Result<TextureData> {
        let pixels = zstd::bulk::decompress(c, raw_len)?;
        if pixels.len() != raw_len {
            bail!("texture size mismatch");
        }
        Ok(TextureData { width, height, srgb, hdr, pixels })
    };
    let big = blobs.iter().map(|b| b.4).sum::<usize>() > (1 << 20);
    if !big || blobs.len() < 2 {
        return blobs.iter().map(one).collect();
    }
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get()).min(8).min(blobs.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<Result<TextureData>>> = (0..blobs.len()).map(|_| None).collect();
    let results = Mutex::new(&mut out);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(b) = blobs.get(i) else { break };
                    let t = one(b);
                    results.lock().unwrap()[i] = Some(t);
                }
            });
        }
    });
    out.into_iter().map(|t| t.unwrap_or_else(|| bail!("texture not decoded"))).collect()
}

pub(crate) fn decode(bytes: &[u8], source: &Path) -> Result<Scene> {
    if bytes.len() < 16 + 8 || &bytes[..4] != MAGIC {
        bail!("not a model cache entry");
    }
    let (body, tail) = bytes.split_at(bytes.len() - 8);
    let mut h = Hasher128::new();
    h.update(body);
    if h.finish() as u64 != u64::from_le_bytes(tail.try_into()?) {
        bail!("checksum mismatch");
    }
    let mut r = R { b: body, at: 4 };
    if r.u32()? != FORMAT_VERSION {
        bail!("entry format version differs");
    }
    r.string()?; // the hook signature (see `stored_hook_signature`)

    let dir = source.parent().unwrap_or(Path::new("."));
    for _ in 0..r.len()? {
        let relative = r.bool()?;
        let p = r.string()?;
        let path = if relative { dir.join(p) } else { PathBuf::from(p) };
        let want = r.u64()?;
        if hash_file(&path).unwrap_or(0) != want {
            bail!("dependency {} changed", path.display());
        }
    }

    let mut scene = Scene {
        meshes: Vec::new(),
        morphs: Vec::new(),
        springs: SpringImport::default(),
        textures: Vec::new(),
        materials: Vec::new(),
        instances: Vec::new(),
        skeletons: Vec::new(),
        animations: Vec::new(),
        collision: Vec::new(),
        identity: Default::default(),
    };
    for _ in 0..r.len()? {
        let vertices: Vec<Vertex> = bytemuck::pod_collect_to_vec(r.bytes()?);
        let indices: Vec<u32> = bytemuck::pod_collect_to_vec(r.bytes()?);
        scene.meshes.push(MeshData { vertices, indices, has_lightmap_uv: r.bool()? });
    }
    for _ in 0..r.len()? {
        let mut targets = Vec::new();
        for _ in 0..r.len()? {
            let name = r.string()?;
            let mut deltas = Vec::new();
            for _ in 0..r.len()? {
                deltas.push((r.u32()?, r.arr::<3>()?, r.arr::<3>()?));
            }
            targets.push(MorphTarget { name, deltas });
        }
        scene.morphs.push(targets);
    }
    for _ in 0..r.len()? {
        let name = r.string()?;
        let mut roots = Vec::new();
        for _ in 0..r.len()? {
            roots.push(r.string()?);
        }
        let (stiffness, damping, gravity, radius) = (r.f32()?, r.f32()?, r.arr::<3>()?, r.f32()?);
        let mut colliders = Vec::new();
        for _ in 0..r.len()? {
            colliders.push(r.string()?);
        }
        scene.springs.chains.push(SpringChain { name, roots, stiffness, damping, gravity, radius, colliders });
    }
    for _ in 0..r.len()? {
        scene.springs.colliders.push(r.json()?);
    }
    // Read the compressed blobs, then inflate them across threads.
    let mut blobs = Vec::new();
    for _ in 0..r.len()? {
        let (width, height, srgb, hdr) = (r.u32()?, r.u32()?, r.bool()?, r.bool()?);
        let raw_len = r.u64()? as usize;
        if raw_len > (1 << 32) {
            bail!("implausible texture size {raw_len}");
        }
        blobs.push((width, height, srgb, hdr, raw_len, r.bytes()?));
    }
    scene.textures = inflate_textures(&blobs)?;
    for _ in 0..r.len()? {
        let name = r.string()?;
        let params: MaterialParams = r.json()?;
        let features: MaterialFeatures = r.json()?;
        scene.materials.push(SceneMaterial { name, params, features, albedo: r.opt()?, normal: r.opt()?, orm: r.opt()?, emission: r.opt()? });
    }
    for _ in 0..r.len()? {
        let (name, transform) = (r.string()?, r.mat4()?);
        let mut slots = Vec::new();
        for _ in 0..r.len()? {
            slots.push(MeshSlot { mesh: r.u64()? as usize, material: r.u64()? as usize });
        }
        scene.instances.push(Instance { name, transform, slots });
    }
    for _ in 0..r.len()? {
        let mut joints = Vec::new();
        for _ in 0..r.len()? {
            joints.push(Joint {
                name: r.string()?,
                parent: r.opt()?,
                inverse_bind: r.mat4()?,
                rest_translation: r.vec3()?,
                rest_rotation: r.quat()?,
                rest_scale: r.vec3()?,
            });
        }
        let mut containers = Vec::new();
        for _ in 0..r.len()? {
            containers.push(r.u64()? as usize);
        }
        scene.skeletons.push(Skeleton { joints, rig: None, containers });
    }
    for _ in 0..r.len()? {
        let (name, duration) = (r.string()?, r.f32()?);
        let mut channels = Vec::new();
        for _ in 0..r.len()? {
            let joint = r.u64()? as usize;
            let path = match r.u8()? {
                0 => ChannelPath::Translation,
                1 => ChannelPath::Rotation,
                2 => ChannelPath::Scale,
                other => bail!("bad channel path {other}"),
            };
            let times = if r.u8()? == 1 {
                channels.last().map(|c: &AnimChannel| c.times.clone()).ok_or_else(|| anyhow::anyhow!("time reference without a channel"))?
            } else {
                let n = r.len()?;
                r.f32s(n)?
            };
            let mut vec_values = Vec::new();
            for _ in 0..r.len()? {
                vec_values.push(r.vec3()?);
            }
            let mut quat_values = Vec::new();
            for _ in 0..r.len()? {
                quat_values.push(r.quat()?);
            }
            let flags = r.u8()?;
            let mut extra = ChannelExtra::default();
            if flags & 1 != 0 {
                let n = r.len()?;
                let mut tan_in = Vec::with_capacity(n);
                let mut tan_out = Vec::with_capacity(n);
                for _ in 0..n {
                    tan_in.push(r.vec3()?);
                }
                for _ in 0..n {
                    tan_out.push(r.vec3()?);
                }
                extra.tangents = Some(Tangents { tan_in, tan_out });
            }
            if flags & 2 != 0 {
                let lanes = r.u8()?;
                if !(3..=4).contains(&lanes) {
                    bail!("bad packed lane count {lanes}");
                }
                let min = r.arr::<4>()?;
                let extent = r.arr::<4>()?;
                let n = r.len()?;
                let raw = r.take(n.checked_mul(2).ok_or_else(|| anyhow::anyhow!("overflow"))?)?;
                let codes = raw.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                extra.packed = Some(PackedValues { lanes, min, extent, codes });
            }
            let extra = (flags != 0).then(|| Box::new(extra));
            channels.push(AnimChannel { joint, path, times, vec_values, quat_values, extra });
        }
        scene.animations.push(AnimationClip { name, duration, channels, morph_tracks: Vec::new(), playback: None });
    }
    for _ in 0..r.len()? {
        let name = r.string()?;
        let kind = match r.u8()? {
            0 => CollisionKind::Convex,
            1 => CollisionKind::Box,
            2 => CollisionKind::Sphere,
            3 => CollisionKind::Capsule,
            other => bail!("bad collision kind {other}"),
        };
        let target = r.opt()?;
        let mut points = Vec::new();
        for _ in 0..r.len()? {
            points.push(r.vec3()?);
        }
        scene.collision.push(CollisionPiece { name, kind, target, points });
    }
    for _ in 0..r.len()? {
        scene.identity.mesh_names.push(r.string()?);
    }
    for _ in 0..r.len()? {
        scene.identity.cameras.push(r.json()?);
    }
    for _ in 0..r.len()? {
        scene.identity.lights.push(r.json()?);
    }
    for _ in 0..r.len()? {
        scene.identity.texture_names.push(r.string()?);
    }
    for _ in 0..r.len()? {
        let mut per_mesh = Vec::new();
        for _ in 0..r.len()? {
            per_mesh.push([r.u32()?, r.u32()?, r.u32()?, r.u32()?]);
        }
        scene.identity.quads.push(per_mesh);
    }
    for _ in 0..r.len()? {
        scene.identity.hidden_instances.push(r.u64()? as usize);
    }
    for _ in 0..r.len()? {
        scene.identity.hook_material_assign.push(if r.u8()? != 0 { Some(r.string()?) } else { None });
    }
    scene.identity.takes = r.json()?;
    scene.identity.clip_info = r.json()?;
    scene.identity.morph_weights = r.json()?;
    scene.identity.constraints = r.json()?;
    scene.identity.custom_props = r.json()?;
    scene.identity.import_warnings = r.json()?;
    scene.identity.optimize = r.json()?;
    for _ in 0..r.len()? {
        scene.identity.material_volume_maps.push(r.opt()?);
    }
    for (clip, info) in scene.animations.iter_mut().zip(&scene.identity.clip_info) {
        clip.morph_tracks = info.morph_tracks.clone();
        clip.playback = info.playback.clone().map(Box::new);
    }
    if r.at != body.len() {
        bail!("trailing bytes");
    }
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest_import_cache_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn hasher_depends_on_every_byte_and_length() {
        let h = |b: &[u8]| {
            let mut x = Hasher128::new();
            x.update(b);
            x.finish()
        };
        assert_eq!(h(b"abcdefghij"), h(b"abcdefghij"));
        assert_ne!(h(b"abcdefghij"), h(b"abcdefghik"));
        assert_ne!(h(b"abcdefgh"), h(b"abcdefgh\0"));
        // Chunked updates match one-shot.
        let mut x = Hasher128::new();
        x.update(b"abc");
        x.update(b"defghij");
        assert_eq!(x.finish(), h(b"abcdefghij"));
    }

    #[test]
    fn decode_rejects_garbage_and_corruption() {
        let src = Path::new("x.glb");
        assert!(decode(b"nope", src).is_err());
        let scene = Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        };
        let mut bytes = encode(&scene, src, &[], "");
        assert!(decode(&bytes, src).is_ok());
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(decode(&bytes, src).is_err());
    }

    fn repo_file(rel: &str) -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
        p.exists().then_some(p)
    }

    /// Two scenes are the same artifact when they encode to the same bytes.
    fn same(a: &Scene, b: &Scene, src: &Path) -> bool {
        encode(a, src, &[], "") == encode(b, src, &[], "")
    }

    #[test]
    fn hit_miss_and_invalidations() {
        let Some(sample) = repo_file("animations/kaykit/Mannequin_Medium.glb") else {
            eprintln!("sample model missing; skipped");
            return;
        };
        let dir = tmp_dir("keys");
        let src = dir.join("m.glb");
        std::fs::copy(&sample, &src).unwrap();
        let model = ModelImport::default();

        let (first, r1) = load_cached(&src, &model, false, || crate::load_gltf(&src)).unwrap();
        assert_eq!(r1.outcome, CacheOutcome::Miss);
        let (second, r2) = load_cached(&src, &model, false, || panic!("importer must not run on a hit")).unwrap();
        assert_eq!(r2.outcome, CacheOutcome::Hit);
        assert_eq!(r1.key, r2.key);
        assert!(same(&first, &second, &src), "a hit reproduces the imported scene exactly");
        assert!(!first.meshes.is_empty() && !first.skeletons.is_empty());

        // Settings that affect the importer change the key; the rig / id table do not.
        let scaled = ModelImport { scale: 0.5, ..Default::default() };
        let (_, r3) = load_cached(&src, &scaled, false, || crate::load_gltf(&src)).unwrap();
        assert_eq!(r3.outcome, CacheOutcome::Miss);
        assert_ne!(r3.key, r1.key);
        let mut with_table = ModelImport { scale: 0.5, ..Default::default() };
        with_table.id_table.entries.clear();
        with_table.id_scheme = 1;
        assert_eq!(cache_key(&src, &with_table), Some(r3.key.clone()), "id table / scheme are outside the key");
        let (_, r4) = load_cached(&src, &with_table, false, || panic!("same key must hit")).unwrap();
        assert_eq!(r4.outcome, CacheOutcome::Hit);

        // Only one entry is kept per source.
        let entries = std::fs::read_dir(dir.join(CACHE_DIR)).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == CACHE_EXTENSION)).count();
        assert_eq!(entries, 1, "stale entries are pruned");

        // Hook versions, importer version inputs and the lightmap marker are part of the key.
        let base = key_with_hooks(&src, &scaled, &[]).unwrap();
        assert_eq!(base, r3.key);
        assert_ne!(key_with_hooks(&src, &scaled, &[("h".into(), 1)]).unwrap(), base);
        assert_ne!(key_with_hooks(&src, &scaled, &[("h".into(), 1)]).unwrap(), key_with_hooks(&src, &scaled, &[("h".into(), 2)]).unwrap());
        crate::set_lightmap_uv_marker(&src, true).unwrap();
        assert_ne!(cache_key(&src, &scaled).unwrap(), base);
        crate::set_lightmap_uv_marker(&src, false).unwrap();
        assert_eq!(cache_key(&src, &scaled).unwrap(), base);

        // Forced rebuild runs the importer and replaces the entry.
        let (_, r5) = load_cached(&src, &scaled, true, || crate::load_gltf(&src)).unwrap();
        assert_eq!(r5.outcome, CacheOutcome::Rebuilt);

        // Changing the source changes the key.
        if let Some(other) = repo_file("animations/kaykit/Rig_Medium/Rig_Medium_General.glb") {
            std::fs::copy(&other, &src).unwrap();
            let (_, r6) = load_cached(&src, &scaled, false, || crate::load_gltf(&src)).unwrap();
            assert_eq!(r6.outcome, CacheOutcome::Miss);
            assert_ne!(r6.key, r1.key);
        }

        // A corrupt entry is a miss, not an error.
        let (_, r7) = load_cached(&src, &scaled, false, || crate::load_gltf(&src)).unwrap();
        let entry = cache_path(&src, &r7.key).unwrap();
        let mut bytes = std::fs::read(&entry).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x55;
        std::fs::write(&entry, bytes).unwrap();
        let (_, r8) = load_cached(&src, &scaled, false, || crate::load_gltf(&src)).unwrap();
        assert_eq!(r8.outcome, CacheOutcome::Miss);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Meta-driven load: a hit keeps the `.meta` GUID, ids and rig applied after the cache.
    #[test]
    fn model_load_applies_meta_after_the_cache() {
        let Some(sample) = repo_file("animations/kaykit/Mannequin_Medium.glb") else { return };
        let dir = tmp_dir("meta");
        let src = dir.join("m.glb");
        std::fs::copy(&sample, &src).unwrap();
        let meta = crate::load_or_create_asset_meta(&src).unwrap();
        let (a, ra) = crate::load_model_cached(&src, false).unwrap();
        let (b, rb) = crate::load_model_cached(&src, false).unwrap();
        assert_eq!((ra.outcome, rb.outcome), (CacheOutcome::Miss, CacheOutcome::Hit));
        assert_eq!(b.identity.guid, Some(meta.id));
        assert_eq!(a.identity.mesh_ids, b.identity.mesh_ids);
        assert_eq!(b.identity.mesh_ids.len(), b.meshes.len());
        // Registering ids rewrites the meta only; the artifact stays valid.
        crate::register_model_ids(&src).unwrap();
        let (c, rc) = crate::load_model_cached(&src, false).unwrap();
        assert_eq!(rc.outcome, CacheOutcome::Hit, "id table edits do not invalidate");
        assert_eq!(c.identity.mesh_ids, a.identity.mesh_ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Prints load time with the cache cold and warm on real models. Timing is
    /// machine-dependent, so this logs rather than asserts.
    #[test]
    fn logs_load_time_cold_and_warm() {
        for rel in ["models/DamagedHelmet.glb", "animations/kaykit/Mannequin_Medium.glb", "example-game/lemonjuice.fbx", "models/leisure_area.FBX"] {
            let Some(sample) = repo_file(rel) else { continue };
            let dir = tmp_dir("timing");
            let src = dir.join(sample.file_name().unwrap());
            std::fs::copy(&sample, &src).unwrap();
            let (_, cold) = crate::load_model_cached(&src, false).unwrap();
            let (_, warm) = crate::load_model_cached(&src, false).unwrap();
            let size = std::fs::metadata(cache_path(&src, &warm.key).unwrap()).map(|m| m.len()).unwrap_or(0);
            eprintln!(
                "import cache {rel}: cold {:?} -> warm {:?} (key {:?}, entry {:.1} MB)",
                cold.total(),
                warm.total(),
                warm.key_time,
                size as f64 / 1e6
            );
            assert_eq!((cold.outcome, warm.outcome), (CacheOutcome::Miss, CacheOutcome::Hit));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    struct TestHook(&'static str, u32, i32);
    impl crate::import_hooks::ModelImportHook for TestHook {
        fn name(&self) -> &str {
            self.0
        }
        fn version(&self) -> u32 {
            self.1
        }
        fn order(&self) -> i32 {
            self.2
        }
    }

    fn hook_set(hooks: &[(&'static str, u32, i32)]) -> crate::import_hooks::HookSet {
        crate::import_hooks::HookSet::new("test", hooks.iter().map(|h| Box::new(TestHook(h.0, h.1, h.2)) as Box<dyn crate::import_hooks::ModelImportHook>).collect())
    }

    #[test]
    fn the_key_follows_the_hook_set_and_the_entry_records_it() {
        let dir = tmp_dir("hook_key");
        let src = dir.join("m.glb");
        std::fs::write(&src, b"not really a model").unwrap();
        let model = ModelImport::default();
        let key = |hooks: &crate::import_hooks::HookSet| key_with_hooks(&src, &model, &hooks.key_entries()).unwrap();
        let none = hook_set(&[]);
        let a1 = hook_set(&[("a", 1, 0)]);
        assert_ne!(key(&a1), key(&none), "adding a hook");
        assert_eq!(key(&a1), key(&hook_set(&[("a", 1, 0)])), "the same set is the same key");
        assert_ne!(key(&a1), key(&hook_set(&[("a", 2, 0)])), "version bump");
        assert_ne!(key(&a1), key(&hook_set(&[("a", 1, 3)])), "order change");
        assert_ne!(key(&hook_set(&[("a", 1, 0), ("b", 1, 0)])), key(&a1), "another hook");
        assert_ne!(key(&hook_set(&[("a", 1, 0), ("b", 1, 0)])), key(&hook_set(&[("b", 1, 0)])), "removing one");
        // With the gate closed (the default here) the global key carries no hooks.
        assert_eq!(cache_key(&src, &model).unwrap(), key(&none));

        // The entry records the signature it was built under, and a different set re-imports.
        let empty = || Ok(scene_with_textures(Vec::new()));
        let (_, r1) = load_cached_with(&src, &model, false, &a1, empty).unwrap();
        assert_eq!(r1.outcome, CacheOutcome::Miss);
        assert_eq!(stored_hook_signature(&src).as_deref(), Some("a#0@1"));
        let (_, r2) = load_cached_with(&src, &model, false, &a1, || panic!("same hooks must hit")).unwrap();
        assert_eq!(r2.outcome, CacheOutcome::Hit);
        let (_, r3) = load_cached_with(&src, &model, false, &hook_set(&[("a", 2, 0)]), empty).unwrap();
        assert_eq!(r3.outcome, CacheOutcome::Miss);
        assert_eq!(stored_hook_signature(&src).as_deref(), Some("a#0@2"));
        let (_, r4) = load_cached_with(&src, &model, false, &none, empty).unwrap();
        assert_eq!(r4.outcome, CacheOutcome::Miss);
        assert_eq!(stored_hook_signature(&src).as_deref(), Some(""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn scene_with_textures(textures: Vec<TextureData>) -> Scene {
        Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures,
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        }
    }

    /// A smooth gradient with some structure, like a real albedo.
    fn gradient(w: u32, h: u32) -> TextureData {
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.extend_from_slice(&[(x / 4) as u8, (y / 4) as u8, ((x + y) / 8) as u8, 255]);
            }
        }
        TextureData { width: w, height: h, srgb: true, hdr: false, pixels }
    }

    #[test]
    fn textures_round_trip_and_shrink() {
        let src = Path::new("x.glb");
        let textures = vec![gradient(512, 512), gradient(300, 200), TextureData { width: 1, height: 1, srgb: false, hdr: true, pixels: vec![1, 2, 3, 4, 5, 6, 7, 8] }];
        let raw: usize = textures.iter().map(|t| t.pixels.len()).sum();
        let scene = scene_with_textures(textures);
        let bytes = encode(&scene, src, &[], "");
        assert!(bytes.len() * 4 < raw, "entry {} should be far smaller than raw {raw}", bytes.len());
        let back = decode(&bytes, src).unwrap();
        assert_eq!(back.textures.len(), 3);
        for (a, b) in scene.textures.iter().zip(&back.textures) {
            assert_eq!((a.width, a.height, a.srgb, a.hdr), (b.width, b.height, b.srgb, b.hdr));
            assert_eq!(a.pixels, b.pixels);
        }
    }

    #[test]
    fn truncated_or_lying_texture_is_a_miss() {
        let src = Path::new("x.glb");
        let bytes = encode(&scene_with_textures(vec![gradient(64, 64)]), src, &[], "");
        assert!(decode(&bytes[..bytes.len() - 20], src).is_err());
    }

    #[test]
    fn cap_prunes_oldest_and_keeps_the_new_entry() {
        let dir = tmp_dir("prune");
        let mk = |name: &str, age: u64| {
            let p = dir.join(name);
            std::fs::write(&p, vec![0u8; 1000]).unwrap();
            let f = std::fs::File::options().write(true).open(&p).unwrap();
            f.set_modified(std::time::SystemTime::now() - Duration::from_secs(age)).unwrap();
            p
        };
        let old = mk("a.glb.1.zmc", 300);
        let mid = mk("b.glb.2.zmc", 200);
        let new = mk("c.glb.3.zmc", 100);
        let other = dir.join("note.txt");
        std::fs::write(&other, b"x").unwrap();
        prune_dir(&dir, &new, 2500);
        assert!(!old.exists() && mid.exists() && new.exists(), "oldest goes first");
        prune_dir(&dir, &new, 0);
        assert!(!mid.exists() && new.exists(), "the just-written entry is never pruned");
        assert!(other.exists(), "non-entries are untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dependency_edit_invalidates_the_entry() {
        let dir = tmp_dir("deps");
        let src = dir.join("m.glb");
        let dep = dir.join("tex.png");
        std::fs::write(&src, b"x").unwrap();
        std::fs::write(&dep, b"one").unwrap();
        let scene = Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        };
        let bytes = encode(&scene, &src, std::slice::from_ref(&dep), "");
        assert!(decode(&bytes, &src).is_ok());
        std::fs::write(&dep, b"two").unwrap();
        assert!(decode(&bytes, &src).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod extras_tests {
    use super::*;
    use crate::{ImportedCamera, ImportedLight};

    #[test]
    fn imported_cameras_and_lights_survive_the_cache() {
        let src = Path::new("x.fbx");
        let mut scene = Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        };
        scene.identity.cameras.push(ImportedCamera { name: "Cam".into(), transform: glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)).to_cols_array(), fov_deg: 50.0, near: 0.2, far: 300.0 });
        scene.identity.lights.push(ImportedLight {
            name: "Sun".into(),
            transform: glam::Mat4::IDENTITY.to_cols_array(),
            kind: zestengine_core::LightKind::Spot,
            color: [1.0, 0.5, 0.25],
            intensity: 2.0,
            range: 12.0,
            spot_angle_deg: 33.0,
        });
        scene.identity.set_quads(1, vec![[0, 1, 2, 3], [4, 5, 6, 7]]);
        scene.identity.hidden_instances = vec![0, 3];
        let back = decode(&encode(&scene, src, &[], ""), src).unwrap();
        assert_eq!(back.identity.cameras, scene.identity.cameras);
        assert_eq!(back.identity.lights, scene.identity.lights);
        assert_eq!(back.identity.quads, vec![Vec::new(), vec![[0, 1, 2, 3], [4, 5, 6, 7]]], "the quad table survives, per mesh");
        assert_eq!(back.identity.hidden_instances, vec![0, 3]);
    }

    #[test]
    fn takes_and_blendshape_curves_survive_the_cache_and_change_the_key() {
        let src = Path::new("x.fbx");
        let mut scene = Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: vec![AnimationClip { name: "Talk".into(), duration: 1.0, channels: Vec::new(), morph_tracks: Vec::new(), playback: None }],
            collision: Vec::new(),
            identity: Default::default(),
        };
        let mut track = crate::AnimTrack::prop("Body", crate::TrackProperty::BlendShape, "Smile", 0);
        track.keys = vec![crate::Keyframe { time: 0.0, value: [0.0; 4], ..Default::default() }, crate::Keyframe { time: 1.0, value: [1.0, 0.0, 0.0, 0.0], interp: crate::KeyInterp::Linear, ..Default::default() }];
        scene.identity.takes = vec![crate::TakeInfo { name: "Take".into(), start: 0.0, stop: 1.0, frame_rate: 30.0, first_frame: 0.0, last_frame: 30.0 }];
        let pb = crate::ClipPlayback { loop_time: true, cycle_offset: 0.25, bake_y: true, body_yaw: 0.5, events: vec![crate::ClipEventDef { time: 0.5, name: "Step".into(), float: 2.0, ..Default::default() }], ..Default::default() };
        scene.identity.clip_info = vec![crate::ClipImportInfo { take: "Take".into(), loop_time: true, morph_tracks: vec![track.clone()], playback: Some(pb.clone()), ..Default::default() }];
        scene.identity.morph_weights = vec![(0, "Smile".into(), 0.4)];
        let back = decode(&encode(&scene, src, &[], ""), src).unwrap();
        assert_eq!(back.identity.takes, scene.identity.takes);
        assert_eq!(back.identity.clip_info, scene.identity.clip_info);
        assert_eq!(back.animations[0].morph_tracks, vec![track], "the clip gets its curves back");
        assert_eq!(back.animations[0].playback.as_deref(), Some(&pb), "and its playback settings");
        assert_eq!(back.identity.morph_weights, scene.identity.morph_weights);
        // The Animation / BlendShape options are part of the key.
        let a = ModelImport::default();
        let mut b = ModelImport::default();
        b.animation.rotation_error = 2.0;
        let mut c = ModelImport::default();
        c.blend_shapes.delta_threshold = 0.1;
        assert_ne!(settings_string(&a), settings_string(&b));
        assert_ne!(settings_string(&a), settings_string(&c));
    }

    #[test]
    fn importer_settings_change_the_key_but_the_remap_tables_do_not() {
        let dir = std::env::temp_dir().join(format!("zest_key_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.glb");
        std::fs::write(&src, b"not really a model").unwrap();
        let base = ModelImport::default();
        let key = |m: &ModelImport| cache_key(&src, m).unwrap();
        let k0 = key(&base);
        for changed in [
            ModelImport { weld_vertices: false, ..base.clone() },
            ModelImport { keep_quads: true, ..base.clone() },
            ModelImport { optimize_mesh: false, ..base.clone() },
            ModelImport { import_blend_shapes: false, ..base.clone() },
            ModelImport { import_cameras: true, ..base.clone() },
            ModelImport { import_lights: true, ..base.clone() },
            ModelImport { bake_axis_conversion: true, ..base.clone() },
            ModelImport { tangents: crate::TangentMode::Calculate, ..base.clone() },
            ModelImport { generate_lightmap_uvs: true, ..base.clone() },
            ModelImport { tangents: crate::TangentMode::Legacy, ..base.clone() },
            ModelImport { normals: crate::NormalsMode::Calculate, ..base.clone() },
            ModelImport { normal_weighting: crate::NormalWeighting::Area, ..base.clone() },
            ModelImport { import_visibility: false, ..base.clone() },
            ModelImport { uv_mask: 0b01, ..base.clone() },
            ModelImport { mesh_compression: crate::MeshCompression::Low, ..base.clone() },
            // A forced 16 bit format splits big meshes.
            ModelImport { index_format: crate::IndexFormat::UInt16, ..base.clone() },
        ] {
            assert_ne!(key(&changed), k0);
        }
        // The older toggle and the new Normals setting are one setting.
        assert_eq!(key(&ModelImport { recalculate_normals: true, ..base.clone() }), key(&ModelImport { normals: crate::NormalsMode::Calculate, ..base.clone() }));
        // Applied when the model is placed, not part of the artifact.
        for same in [
            ModelImport { generate_colliders: true, ..base.clone() },
            ModelImport { read_write: true, ..base.clone() },
            ModelImport { rig_pose_scale: 0.5, ..base.clone() },
        ] {
            assert_eq!(key(&same), k0);
        }
        let mut remapped = base.clone();
        remapped.material_remap.insert("A".into(), "a.material".into());
        remapped.texture_remap.insert("A/albedo".into(), "a.png".into());
        remapped.source_materials = vec!["A".into()];
        remapped.extract_textures_dir = "tex".into();
        remapped.extract_materials_dir = "mats".into();
        remapped.index_format = crate::IndexFormat::UInt32;
        assert_eq!(key(&remapped), k0, "remaps, source list and index format are applied after the import");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
