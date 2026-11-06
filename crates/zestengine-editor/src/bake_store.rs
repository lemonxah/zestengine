//! Saving baked lighting off the main thread. After a bake the editor hands
//! the raw lightmaps / captured probe faces to a background task (see
//! `tasks.rs`) that BC6H-encodes them, writes the `.lightmap` / `.reflprobe`
//! sidecars atomically, and returns GPU-ready data for the main thread to
//! swap in. Also the on-disk / VRAM size stats for the Baker's Man tab.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use zestengine_render::{BakedLightmap, HdrFormatSupport, HdrTextureArray, HdrTextureFormat};

use zestengine_engine::refl_slots::{Faces, ProbeCube, ReflProbeSet};
use crate::tasks::{TaskPayload, TaskProgress};

/// Serializes read-merge-write of each sidecar kind across save tasks.
static LIGHTMAP_WRITE: Mutex<()> = Mutex::new(());
static REFL_WRITE: Mutex<()> = Mutex::new(());
/// Newest bake serial handed to a lightmap save; older saves skip writing.
static LATEST_LIGHTMAP_SERIAL: AtomicU64 = AtomicU64::new(0);
/// Order of reflection writes, so a late result never replaces a newer set.
static REFL_WRITE_SEQ: AtomicU64 = AtomicU64::new(0);
/// Save tasks still running (the editor waits for these before quitting).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

struct InFlight;
impl InFlight {
    fn new() -> Self {
        IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
        Self
    }
}
impl Drop for InFlight {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Block (quit path only) until running sidecar saves finish, up to `timeout`.
pub fn wait_for_saves(timeout: std::time::Duration) {
    let start = std::time::Instant::now();
    while IN_FLIGHT.load(Ordering::SeqCst) > 0 && start.elapsed() < timeout {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Sizes of one sidecar kind.
#[derive(Clone, Copy, Debug)]
pub struct SidecarStats {
    pub count: usize,
    /// Stored format (None = mixed / legacy RGBA8).
    pub format: Option<HdrTextureFormat>,
    pub disk_bytes: u64,
    /// The same data uncompressed (RGBA32F lightmaps / RGBA16F probe faces).
    pub raw_bytes: u64,
    pub vram_bytes: u64,
    pub vram_format: Option<HdrTextureFormat>,
    /// Mean relative RGB error of the stored data against the raw bake,
    /// measured when it was compressed (None = not compressed / not measured).
    pub error: Option<f32>,
}

impl SidecarStats {
    /// One line for the Baker's Man tab.
    pub fn describe(&self, what: &str) -> String {
        let fmt = |f: Option<HdrTextureFormat>| f.map_or("mixed", HdrTextureFormat::label);
        let mut line = format!(
            "{what}: {} · {} · {} on disk",
            self.count,
            fmt(self.format),
            human_bytes(self.disk_bytes)
        );
        if self.raw_bytes > self.disk_bytes {
            line += &format!(" (uncompressed {}, {:.1}x)", human_bytes(self.raw_bytes), self.raw_bytes as f64 / self.disk_bytes.max(1) as f64);
        }
        line += &format!(" · {} VRAM ({})", human_bytes(self.vram_bytes), fmt(self.vram_format));
        if let Some(e) = self.error {
            line += &format!(" · compression error {:.1}%", e * 100.0);
        }
        line
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BakeSizes {
    pub lightmaps: Option<SidecarStats>,
    pub reflections: Option<SidecarStats>,
}

impl BakeSizes {
    /// Whether either sidecar is stored uncompressed (so "Compress" has work).
    pub fn has_uncompressed(&self) -> bool {
        [self.lightmaps, self.reflections]
            .iter()
            .flatten()
            .any(|s| s.format != Some(HdrTextureFormat::Bc6h))
    }
}

pub fn human_bytes(b: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    if b as f64 >= MB {
        format!("{:.1} MB", b as f64 / MB)
    } else {
        format!("{:.0} KB", (b as f64 / 1024.0).ceil())
    }
}

/// Stats from the sidecar headers (`base` = scene path without extension)
/// plus what is currently uploaded.
pub fn measure(base: &Path, baked: Option<&zestengine_engine::scene::BakedData>, refl: &ReflProbeSet) -> BakeSizes {
    let lightmaps = zestengine_assets::lightmap_file_info(base.with_extension("lightmap")).ok().map(|info| SidecarStats {
        count: info.sizes.len(),
        format: Some(info.format),
        disk_bytes: info.file_bytes,
        raw_bytes: info.raw_bytes(),
        vram_bytes: baked.map_or(0, |b| b.lightmap_vram_bytes()),
        vram_format: baked.map(|b| b.lightmap_format()),
        error: None,
    });
    let reflections = zestengine_engine::refl_slots::reflprobe_file_info(&base.with_extension("reflprobe")).ok().map(|info| {
        let mut formats: Vec<Option<HdrTextureFormat>> = info.cubes.iter().map(|c| c.0).collect();
        formats.dedup();
        let mut vram = refl.formats();
        vram.dedup();
        SidecarStats {
            count: info.cubes.len(),
            format: if formats.len() == 1 { formats[0] } else { None },
            disk_bytes: info.file_bytes,
            raw_bytes: info.raw_bytes(),
            vram_bytes: refl.vram_bytes(),
            vram_format: (vram.len() == 1).then(|| vram[0]),
            error: None,
        }
    });
    BakeSizes { lightmaps, reflections }
}

/// A finished lightmap save, for the main thread to swap in.
pub struct LightmapsSaved {
    pub serial: u64,
    pub base: PathBuf,
    /// Object → layer of `array` (layers are written in this order).
    pub object_lightmap: HashMap<usize, usize>,
    /// Upload-ready layers (this device's format for what was stored).
    pub array: HdrTextureArray,
    pub stats: SidecarStats,
}

/// A finished reflection-probe save.
pub struct ReflectionsSaved {
    pub path: PathBuf,
    pub seq: u64,
    /// The whole merged sidecar, prepared for upload.
    pub set: ReflProbeSet,
    pub stats: SidecarStats,
}

#[derive(Default)]
pub struct BakeSaved {
    pub lightmaps: Option<LightmapsSaved>,
    pub reflections: Option<ReflectionsSaved>,
}

/// Mark `serial` as the newest lightmap save (call on the main thread when
/// spawning it).
pub fn begin_lightmap_save(serial: u64) {
    LATEST_LIGHTMAP_SERIAL.fetch_max(serial, Ordering::SeqCst);
}

/// Write an empty `.lightmap` (a bake with no lit objects), superseding any
/// older file and any older save still encoding.
pub fn write_empty_lightmaps(base: &Path) {
    let _lock = LIGHTMAP_WRITE.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = zestengine_assets::save_lightmaps(base.with_extension("lightmap"), &zestengine_assets::LightmapFile::default()) {
        tracing::error!("saving .lightmap: {e:#}");
    }
}

fn set_progress(progress: &Mutex<TaskProgress>, done: f32, label: &str) {
    if let Ok(mut p) = progress.lock() {
        *p = TaskProgress::Fraction { done, label: label.into() };
    }
}

/// Worker: store a fresh bake's lightmaps (`maps` in layer order) as BC6H, or
/// as v1 RGBA32F when `compress` is off or the task is cancelled (so a bake is
/// Mean relative RGB error of compressed lightmap layers against the raw bake
/// (|decoded - raw| / max(raw, 0.05), over each map's own texels).
fn lightmap_error(maps: &[(u32, BakedLightmap)], array: &HdrTextureArray) -> Option<f32> {
    let (mut sum, mut n) = (0.0f64, 0u64);
    for ((_, map), layer) in maps.iter().zip(&array.layers) {
        let decoded =
            zestengine_assets::hdr_codec::transcode_level(layer, array.format, HdrTextureFormat::Rgba32f, array.size, array.size, 1).ok()?;
        let decoded: Vec<f32> = decoded.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        for y in 0..map.size as usize {
            for x in 0..map.size as usize {
                let raw = &map.pixels[(y * map.size as usize + x) * 4..][..3];
                let got = &decoded[(y * array.size as usize + x) * 4..][..3];
                for c in 0..3 {
                    sum += ((got[c] - raw[c]).abs() / raw[c].abs().max(0.05)) as f64;
                    n += 1;
                }
            }
        }
    }
    (n > 0).then(|| (sum / n as f64) as f32)
}

/// never lost), and return the upload-ready array.
pub fn save_lightmaps_job(
    serial: u64,
    base: PathBuf,
    maps: Vec<(u32, BakedLightmap)>,
    compress: bool,
    cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<TaskProgress>>,
) -> Result<TaskPayload, String> {
    let _flight = InFlight::new();
    zestengine_engine::sw_gi::lower_compute_priority();
    let threads = zestengine_engine::sw_gi::bake_worker_count();
    let path = base.with_extension("lightmap");
    let refs: Vec<&BakedLightmap> = maps.iter().map(|(_, m)| m).collect();
    let objects: Vec<u32> = maps.iter().map(|&(o, _)| o).collect();
    let sizes: Vec<u32> = maps.iter().map(|(_, m)| m.size).collect();
    let label = if compress { "Compressing lightmaps" } else { "Saving lightmaps" };
    let encoded = compress
        .then(|| {
            zestengine_assets::hdr_codec::encode_lightmaps(&refs, HdrTextureFormat::Bc6h, threads, &cancel, &|f| {
                set_progress(&progress, f * 0.95, label)
            })
        })
        .flatten();
    let _lock = LIGHTMAP_WRITE.lock().unwrap_or_else(|e| e.into_inner());
    if LATEST_LIGHTMAP_SERIAL.load(Ordering::SeqCst) > serial {
        return Ok(TaskPayload::None); // a newer bake owns the file
    }
    let (stored, disk_bytes) = match encoded {
        Some(array) => {
            let layers = zestengine_assets::LightmapLayers { objects: objects.clone(), sizes: sizes.clone(), array };
            let bytes = zestengine_assets::save_lightmap_layers(&path, &layers).map_err(|e| format!("{e:#}"))?;
            (layers.array, bytes)
        }
        None => {
            let file = zestengine_assets::LightmapFile {
                entries: maps
                    .iter()
                    .map(|(o, m)| zestengine_assets::LightmapEntry { object: *o, size: m.size, pixels: m.pixels.clone() })
                    .collect(),
            };
            zestengine_assets::save_lightmaps(&path, &file).map_err(|e| format!("{e:#}"))?;
            let array = zestengine_assets::hdr_codec::encode_lightmaps(&refs, HdrTextureFormat::Rgba32f, 1, &AtomicBool::new(false), &|_| {})
                .unwrap_or(HdrTextureArray { format: HdrTextureFormat::Rgba32f, size: 1, layers: Vec::new() });
            (array, std::fs::metadata(&path).map_or(0, |m| m.len()))
        }
    };
    set_progress(&progress, 1.0, label);
    let format = stored.format;
    let stored_copy = stored.clone();
    let to = HdrFormatSupport::current().upload_format(format);
    let array = zestengine_assets::hdr_codec::transcode_array(stored, to, threads).map_err(|e| format!("{e:#}"))?;
    let stats = SidecarStats {
        count: maps.len(),
        format: Some(format),
        disk_bytes,
        raw_bytes: sizes.iter().map(|&s| s as u64 * s as u64 * 16).sum(),
        vram_bytes: array.byte_size(),
        vram_format: Some(array.format),
        error: (format != HdrTextureFormat::Rgba32f).then(|| lightmap_error(&maps, &stored_copy)).flatten(),
    };
    tracing::info!("wrote {}: {}", path.display(), stats.describe("lightmaps"));
    Ok(TaskPayload::BakeSaved(Box::new(BakeSaved {
        lightmaps: Some(LightmapsSaved {
            serial,
            base,
            object_lightmap: objects.iter().enumerate().map(|(layer, &o)| (o as usize, layer)).collect(),
            array,
            stats,
        }),
        reflections: None,
    })))
}

/// Merge `new` probe cubes into `path` and rewrite it. With `compress`, every
/// raw cube in the file (old ones too) is BC6H-encoded.
fn save_reflections(path: &Path, new: Vec<(u128, Faces)>, compress: bool, threads: usize) -> Result<ReflectionsSaved, String> {
    let _lock = REFL_WRITE.lock().unwrap_or_else(|e| e.into_inner());
    let mut set = zestengine_engine::refl_slots::read_reflprobes_stored(path);
    // A legacy single cube is superseded once probes bake their own.
    if !new.is_empty() {
        set.legacy = None;
    }
    for (id, faces) in new {
        set.entries.insert(id, ProbeCube::Faces(faces));
    }
    if compress {
        for cube in set.entries.values_mut().chain(set.legacy.as_mut()) {
            if matches!(cube, ProbeCube::Faces(_)) {
                let raw = std::mem::replace(cube, ProbeCube::Chain(zestengine_render::HdrCube {
                    format: HdrTextureFormat::Bc6h,
                    size: 0,
                    faces: Default::default(),
                }));
                *cube = raw.compressed(threads).map_err(|e| format!("{e:#}"))?;
            }
        }
    }
    let disk_bytes = zestengine_engine::refl_slots::write_reflprobes(path, &set).map_err(|e| format!("{e:#}"))?;
    let seq = REFL_WRITE_SEQ.fetch_add(1, Ordering::SeqCst) + 1;
    let raw_bytes: u64 = set
        .entries
        .values()
        .chain(set.legacy.as_ref())
        .map(|c| match c {
            ProbeCube::Faces(f) => 6 * f[0].width as u64 * f[0].width as u64 * 8,
            ProbeCube::Chain(c) => 6 * c.size as u64 * c.size as u64 * 8,
        })
        .sum();
    let formats = set.formats();
    let count = set.entries.len() + usize::from(set.legacy.is_some());
    let set = set.prepare_for(HdrFormatSupport::current());
    let vram = set.formats();
    let stats = SidecarStats {
        count,
        format: (formats.len() == 1).then(|| formats[0]),
        disk_bytes,
        raw_bytes,
        vram_bytes: set.vram_bytes(),
        vram_format: (vram.len() == 1).then(|| vram[0]),
        error: None,
    };
    tracing::info!("wrote {}: {}", path.display(), stats.describe("reflection probes"));
    Ok(ReflectionsSaved { path: path.to_owned(), seq, set, stats })
}

/// Worker: merge freshly captured probe faces into the scene's `.reflprobe`.
pub fn save_reflections_job(path: PathBuf, new: Vec<(u128, Faces)>, compress: bool) -> Result<TaskPayload, String> {
    let _flight = InFlight::new();
    zestengine_engine::sw_gi::lower_compute_priority();
    let saved = save_reflections(&path, new, compress, zestengine_engine::sw_gi::bake_worker_count())?;
    Ok(TaskPayload::BakeSaved(Box::new(BakeSaved { lightmaps: None, reflections: Some(saved) })))
}

/// Worker: re-encode an existing (uncompressed) bake as BC6H in place.
pub fn recompress_job(serial: u64, base: PathBuf, progress: Arc<Mutex<TaskProgress>>) -> Result<TaskPayload, String> {
    let _flight = InFlight::new();
    zestengine_engine::sw_gi::lower_compute_priority();
    let threads = zestengine_engine::sw_gi::bake_worker_count();
    let mut out = BakeSaved::default();
    let lm_path = base.with_extension("lightmap");
    if let Ok(stored) = zestengine_assets::read_lightmap_layers(&lm_path)
        && stored.array.format != HdrTextureFormat::Bc6h
    {
        set_progress(&progress, 0.1, "Compressing lightmaps");
        let raw_bytes = stored.raw_bytes();
        let array = zestengine_assets::hdr_codec::transcode_array(stored.array, HdrTextureFormat::Bc6h, threads).map_err(|e| format!("{e:#}"))?;
        let layers = zestengine_assets::LightmapLayers { objects: stored.objects, sizes: stored.sizes, array };
        let disk_bytes = {
            let _lock = LIGHTMAP_WRITE.lock().unwrap_or_else(|e| e.into_inner());
            zestengine_assets::save_lightmap_layers(&lm_path, &layers).map_err(|e| format!("{e:#}"))?
        };
        let array = zestengine_assets::hdr_codec::transcode_array(
            layers.array,
            HdrFormatSupport::current().upload_format(HdrTextureFormat::Bc6h),
            threads,
        )
        .map_err(|e| format!("{e:#}"))?;
        out.lightmaps = Some(LightmapsSaved {
            serial,
            base: base.clone(),
            object_lightmap: layers.objects.iter().enumerate().map(|(layer, &o)| (o as usize, layer)).collect(),
            stats: SidecarStats {
                count: layers.objects.len(),
                format: Some(HdrTextureFormat::Bc6h),
                disk_bytes,
                raw_bytes,
                vram_bytes: array.byte_size(),
                vram_format: Some(array.format),
                error: None,
            },
            array,
        });
    }
    let refl_path = base.with_extension("reflprobe");
    if refl_path.exists() {
        set_progress(&progress, 0.5, "Compressing reflection probes");
        out.reflections = Some(save_reflections(&refl_path, Vec::new(), true, threads)?);
    }
    set_progress(&progress, 1.0, "Compressing bake");
    Ok(TaskPayload::BakeSaved(Box::new(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_line_reports_before_and_after() {
        let s = SidecarStats {
            count: 3,
            format: Some(HdrTextureFormat::Bc6h),
            disk_bytes: 3 * 1024 * 1024,
            raw_bytes: 48 * 1024 * 1024,
            vram_bytes: 3 * 1024 * 1024,
            vram_format: Some(HdrTextureFormat::Bc6h),
            error: None,
        };
        assert_eq!(
            s.describe("Lightmaps"),
            "Lightmaps: 3 · BC6H · 3.0 MB on disk (uncompressed 48.0 MB, 16.0x) · 3.0 MB VRAM (BC6H)"
        );
        let sizes = BakeSizes { lightmaps: Some(s), reflections: None };
        assert!(!sizes.has_uncompressed());
    }

    /// Recompressing a legacy (v1 + v2 single cube) bake upgrades both files
    /// in place and reports the shrink.
    #[test]
    fn recompress_upgrades_legacy_sidecars() {
        let dir = std::env::temp_dir().join(format!("zestengine-recompress-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("scene");
        let px: Vec<f32> = (0..64 * 64).flat_map(|i| [0.5 + (i % 64) as f32 * 0.02, 1.0, 0.75, 1.0]).collect();
        zestengine_assets::save_lightmaps(
            base.with_extension("lightmap"),
            &zestengine_assets::LightmapFile { entries: vec![zestengine_assets::LightmapEntry { object: 2, size: 64, pixels: px }] },
        )
        .unwrap();
        let face = zestengine_render::rgba32f_to_rgba16f(&[1.5f32, 0.75, 0.5, 1.0].repeat(32 * 32));
        let mut v2 = b"CITRSRP2".to_vec();
        v2.extend_from_slice(&32u32.to_le_bytes());
        v2.extend_from_slice(&6u32.to_le_bytes());
        for _ in 0..6 {
            v2.extend_from_slice(&face);
        }
        std::fs::write(base.with_extension("reflprobe"), v2).unwrap();

        let before = measure(&base, None, &ReflProbeSet::default());
        assert!(before.has_uncompressed());
        let progress = Arc::new(Mutex::new(TaskProgress::Indeterminate));
        let Ok(TaskPayload::BakeSaved(saved)) = recompress_job(1, base.clone(), progress) else { panic!() };
        let lm = saved.lightmaps.unwrap();
        assert_eq!(lm.stats.format, Some(HdrTextureFormat::Bc6h));
        assert_eq!(lm.stats.raw_bytes, 64 * 64 * 16);
        assert!(lm.stats.disk_bytes < lm.stats.raw_bytes / 10, "{:?}", lm.stats);
        assert_eq!(lm.object_lightmap.get(&2), Some(&0));
        let after = measure(&base, None, &ReflProbeSet::default());
        assert_eq!(after.lightmaps.unwrap().format, Some(HdrTextureFormat::Bc6h));
        // The legacy single cube stays the fallback for every probe, now BC6H.
        let refl = saved.reflections.unwrap();
        assert_eq!(refl.stats.format, Some(HdrTextureFormat::Bc6h));
        assert!(refl.stats.disk_bytes < refl.stats.raw_bytes / 4, "{:?}", refl.stats);
        assert!(refl.set.cube(5).is_some());
        assert!(!measure(&base, None, &ReflProbeSet::default()).has_uncompressed());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compression_error_is_measured_against_the_raw_bake() {
        let size = 16u32;
        let pixels: Vec<f32> = (0..size * size).flat_map(|i| [0.2 + (i % size) as f32 * 0.1, 1.5, 0.05 * (i / size) as f32, 1.0]).collect();
        let maps = vec![(3u32, BakedLightmap { size, pixels })];
        let refs: Vec<&BakedLightmap> = maps.iter().map(|(_, m)| m).collect();
        let cancel = AtomicBool::new(false);
        let bc6h = zestengine_assets::hdr_codec::encode_lightmaps(&refs, HdrTextureFormat::Bc6h, 1, &cancel, &|_| {}).unwrap();
        let e = lightmap_error(&maps, &bc6h).unwrap();
        assert!(e > 0.0 && e < 0.05, "BC6H error {e}");
        let raw = zestengine_assets::hdr_codec::encode_lightmaps(&refs, HdrTextureFormat::Rgba32f, 1, &cancel, &|_| {}).unwrap();
        assert!(lightmap_error(&maps, &raw).unwrap() < 1e-6, "raw data has no error");
        let stats = SidecarStats { count: 1, format: Some(HdrTextureFormat::Bc6h), disk_bytes: 1, raw_bytes: 16, vram_bytes: 1, vram_format: None, error: Some(e) };
        assert!(stats.describe("Lightmaps").contains("compression error"));
    }
}