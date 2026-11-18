//! Device/driver-specific Vulkan pipeline persistence. This is separate from
//! the live material-variant map in `pipeline.rs`.
use ash::{vk, vk::Handle};
use std::{collections::HashMap, path::PathBuf, sync::{Arc, Mutex, OnceLock, RwLock, atomic::{AtomicBool, Ordering}}, time::Instant};

const MAGIC: &[u8; 8] = b"ZESTVK01";
const MAX_BYTES: usize = 256 * 1024 * 1024;
/// A Vulkan cache only grows: every shader that was ever edited keeps its
/// pipelines in it. A saved cache this close to the budget has no room for
/// another session's pipelines, and one over the budget is not written at all,
/// so every start would compile again whatever it lacks. Such a file is
/// discarded and rebuilt from the pipelines in use (one start without a cache).
const FULL_BYTES: usize = MAX_BYTES / 8 * 7;
type Registry = Mutex<HashMap<u64, Arc<Cache>>>;
static CACHES: OnceLock<Registry> = OnceLock::new();
static PROGRESS: Mutex<Progress> = Mutex::new(Progress { started: None, ready: 0, loaded: false });
struct Progress { started: Option<Instant>, ready: usize, loaded: bool }
/// Vulkan synchronizes pipeline creation and data readback on a cache itself
/// (this one is not created `EXTERNALLY_SYNCHRONIZED`); only its destruction
/// needs the caller's exclusion. So creations hold `building` shared, for
/// `unregister` to wait on, and never wait for each other: a pipeline the frame
/// creates is not held up by a material variant on the compiler thread.
struct Cache {
    handle: vk::PipelineCache,
    path: Option<PathBuf>,
    dirty: AtomicBool,
    /// Grew past the budget in this session: its file is gone and it is not
    /// read back again; the next session starts it afresh.
    full: AtomicBool,
    /// Time of the last write to disk. Held across a write: one at a time.
    saved: Mutex<Instant>,
    building: RwLock<()>,
}

/// Splash text uses real elapsed time and completed work, never an estimated ETA.
pub fn shader_build_status() -> Option<String> {
    let p = PROGRESS.lock().unwrap();
    let seconds = p.started?.elapsed().as_secs();
    Some(format!("Preparing shaders - {}m {:02}s - {} ready{}", seconds / 60, seconds % 60,
        p.ready, if p.loaded { " (saved cache loaded)" } else { "" }))
}

pub(crate) fn reset_progress() {
    *PROGRESS.lock().unwrap() = Progress { started: None, ready: 0, loaded: false };
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3))
}
fn encode(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + bytes.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum(bytes).to_le_bytes());
    out.extend_from_slice(bytes);
    out
}
fn decode<'a>(bytes: &'a [u8], props: &vk::PhysicalDeviceProperties) -> Option<&'a [u8]> {
    if bytes.len() < 56 || &bytes[..8] != MAGIC { return None; }
    let len = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let data = &bytes[24..];
    if len != data.len() as u64 || data.len() > MAX_BYTES
        || checksum(data) != u64::from_le_bytes(bytes[16..24].try_into().ok()?) { return None; }
    // Vulkan cache header v1 is explicitly little endian, including on BE hosts.
    let word = |offset| u32::from_le_bytes(data[offset..offset+4].try_into().unwrap());
    if word(0) != 32 || word(4) != 1 || word(8) != props.vendor_id
        || word(12) != props.device_id || data[16..32] != props.pipeline_cache_uuid { return None; }
    Some(data)
}
fn path(props: &vk::PhysicalDeviceProperties) -> Option<PathBuf> {
    let root = std::env::var_os("ZESTENGINE_PIPELINE_CACHE_DIR").map(PathBuf::from).or_else(|| {
        let base = if cfg!(target_os = "windows") { std::env::var_os("LOCALAPPDATA").map(PathBuf::from) }
        else if cfg!(target_os = "macos") { std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Caches")) }
        else { std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache"))) };
        base.map(|p| p.join("zestengine/pipelines"))
    })?;
    let uuid: String = props.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
    Some(root.join(format!("v1-{:x}-{:x}-{:x}-{}-{uuid}.bin", props.vendor_id, props.device_id, props.driver_version, usize::BITS)))
}

pub(crate) fn register(device: &ash::Device, props: &vk::PhysicalDeviceProperties) {
    let path = path(props);
    let bytes = path.as_ref().and_then(|p| {
        use std::io::Read;
        let file = std::fs::File::open(p).ok()?;
        let mut bytes = Vec::new();
        file.take((MAX_BYTES + 25) as u64).read_to_end(&mut bytes).ok()?;
        Some(bytes)
    });
    let mut initial = bytes.as_deref().and_then(|b| decode(b, props)).unwrap_or_default();
    if initial.len() > FULL_BYTES {
        tracing::warn!(bytes = initial.len(), budget = MAX_BYTES, path = ?path,
            "saved Vulkan pipeline cache has reached its budget; rebuilding it from the pipelines in use");
        if let Some(p) = &path { let _ = std::fs::remove_file(p); }
        initial = &[];
    }
    let created = unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(initial), None) };
    let (created, loaded) = match created {
        Ok(cache) => (Ok(cache), !initial.is_empty()),
        Err(error) if !initial.is_empty() => {
            tracing::warn!(?error, "saved pipeline cache rejected; rebuilding");
            (unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }, false)
        }
        Err(error) => (Err(error), false),
    };
    match created {
        Ok(handle) => {
            PROGRESS.lock().unwrap().loaded = loaded;
            tracing::info!(loaded, bytes = initial.len(), path = ?path, "Vulkan pipeline cache");
            CACHES.get_or_init(Default::default).lock().unwrap().insert(device.handle().as_raw(),
                Arc::new(Cache { handle, path, dirty: AtomicBool::new(false), full: AtomicBool::new(false),
                    saved: Mutex::new(Instant::now()), building: RwLock::new(()) }));
        }
        Err(error) => tracing::warn!(?error, "pipeline persistence unavailable; continuing without cache"),
    }
}
fn lookup(device: &ash::Device) -> Option<Arc<Cache>> {
    CACHES.get()?.lock().unwrap().get(&device.handle().as_raw()).cloned()
}
fn save(device: &ash::Device, cache: &Cache, saved: &mut Instant) {
    *saved = Instant::now();
    // Cleared first: a pipeline created during the readback marks it again.
    if !cache.dirty.swap(false, Ordering::AcqRel) || cache.full.load(Ordering::Acquire) { return; }
    let Some(path) = &cache.path else { return; };
    let result = (|| -> anyhow::Result<usize> {
        let bytes = unsafe { device.get_pipeline_cache_data(cache.handle)? };
        if bytes.len() > MAX_BYTES {
            // Not an error to retry: reading it back again cannot make it smaller.
            cache.full.store(true, Ordering::Release);
            let _ = std::fs::remove_file(path);
            tracing::warn!(bytes = bytes.len(), budget = MAX_BYTES,
                "Vulkan pipeline cache outgrew its budget; the next start rebuilds it from the pipelines in use");
            return Ok(0);
        }
        std::fs::create_dir_all(path.parent().unwrap())?;
        // Different editor/game processes cannot expose partially written data.
        let temp = path.with_extension(format!("{}-{}.tmp", std::process::id(), cache.handle.as_raw()));
        let result = std::fs::write(&temp, encode(&bytes)).and_then(|_| std::fs::rename(&temp, path));
        if result.is_err() { let _ = std::fs::remove_file(&temp); }
        result?;
        Ok(bytes.len())
    })();
    match result {
        Ok(0) => {}
        Ok(bytes) => tracing::debug!(bytes, "saved Vulkan pipeline cache"),
        Err(error) => {
            cache.dirty.store(true, Ordering::Release);
            tracing::warn!(%error, "could not save Vulkan pipeline cache");
        }
    }
}
pub(crate) fn checkpoint(device: &ash::Device) {
    if let Some(cache) = lookup(device) { save(device, &cache, &mut cache.saved.lock().unwrap()); }
}
pub(crate) fn unregister(device: &ash::Device) {
    crate::shader_tasks::forget(device.handle().as_raw());
    let cache = CACHES.get().and_then(|r| r.lock().unwrap().remove(&device.handle().as_raw()));
    if let Some(cache) = cache {
        let _idle = cache.building.write().unwrap();
        save(device, &cache, &mut cache.saved.lock().unwrap());
        unsafe { device.destroy_pipeline_cache(cache.handle, None); }
    }
}

type PipelineResult = Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>;
/// Same contracts as ash's pipeline creation functions. Creations run side by
/// side on the shared cache (see [`Cache`]).
pub(crate) trait PipelineDeviceExt {
    unsafe fn create_cached_compute_pipelines(&self, fallback: vk::PipelineCache, infos: &[vk::ComputePipelineCreateInfo<'_>], allocator: Option<&vk::AllocationCallbacks<'_>>) -> PipelineResult;
    unsafe fn create_cached_graphics_pipelines(&self, fallback: vk::PipelineCache, infos: &[vk::GraphicsPipelineCreateInfo<'_>], allocator: Option<&vk::AllocationCallbacks<'_>>) -> PipelineResult;
}
#[track_caller]
fn build(device: &ash::Device, fallback: vk::PipelineCache, kind: &'static str, make: impl FnOnce(vk::PipelineCache) -> PipelineResult) -> PipelineResult {
    let _scope = crate::hitch::scope(kind);
    let caller = std::panic::Location::caller();
    // Emit before entering the driver: a stuck creation otherwise leaves no
    // evidence until it returns, and all of its time appears under rr_main.
    tracing::info!(source = caller.file(), line = caller.line(), kind, "preparing shader pipeline");
    let cache = lookup(device);
    let _building = cache.as_ref().map(|c| c.building.read().unwrap());
    let started = Instant::now();
    PROGRESS.lock().unwrap().started.get_or_insert(started);
    let result = make(cache.as_ref().map_or(fallback, |c| c.handle));
    if let Ok(pipelines) = &result { PROGRESS.lock().unwrap().ready += pipelines.len(); }
    let elapsed = started.elapsed();
    if elapsed.as_millis() >= 50 {
        tracing::info!(source = caller.file(), line = caller.line(), milliseconds = elapsed.as_millis() as u64,
            success = result.is_ok(), "shader pipeline prepared");
    }
    if let Some(cache) = &cache {
        cache.dirty.store(true, Ordering::Release);
        // Preserve first-use pipelines too, even if the app is later terminated.
        // Throttle readback/write work; never perform it on ordinary draw frames,
        // and leave it to the other thread when one is already writing.
        if let Ok(mut saved) = cache.saved.try_lock()
            && saved.elapsed().as_secs() >= 10 { save(device, cache, &mut saved); }
    }
    result
}
impl PipelineDeviceExt for ash::Device {
    #[track_caller]
    unsafe fn create_cached_compute_pipelines(&self, fallback: vk::PipelineCache, infos: &[vk::ComputePipelineCreateInfo<'_>], allocator: Option<&vk::AllocationCallbacks<'_>>) -> PipelineResult {
        build(self, fallback, "compile:compute_pipeline", |cache| unsafe { self.create_compute_pipelines(cache, infos, allocator) })
    }
    #[track_caller]
    unsafe fn create_cached_graphics_pipelines(&self, fallback: vk::PipelineCache, infos: &[vk::GraphicsPipelineCreateInfo<'_>], allocator: Option<&vk::AllocationCallbacks<'_>>) -> PipelineResult {
        build(self, fallback, "compile:graphics_pipeline", |cache| unsafe { self.create_graphics_pipelines(cache, infos, allocator) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disk_cache_rejects_corruption_truncation_and_foreign_devices() {
        let p = vk::PhysicalDeviceProperties { vendor_id: 123, device_id: 456, pipeline_cache_uuid: [7;16], ..Default::default() };
        let mut raw = Vec::new();
        for v in [32u32, 1, p.vendor_id, p.device_id] { raw.extend_from_slice(&v.to_le_bytes()); }
        raw.extend_from_slice(&p.pipeline_cache_uuid); raw.extend_from_slice(b"opaque driver data");
        let bytes = encode(&raw);
        assert_eq!(decode(&bytes, &p), Some(raw.as_slice()));
        for end in 0..bytes.len() { assert!(decode(&bytes[..end], &p).is_none()); }
        for index in 0..bytes.len() {
            let mut damaged = bytes.clone(); damaged[index] ^= 1;
            assert!(decode(&damaged, &p).is_none());
        }
        let mut foreign = p; foreign.pipeline_cache_uuid[0] ^= 1;
        assert!(decode(&bytes, &foreign).is_none());
        foreign = p; foreign.device_id += 1; assert!(decode(&bytes, &foreign).is_none());
    }
}
