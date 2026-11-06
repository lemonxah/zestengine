//! Baked-lighting asset files (compact little-endian binary; HDR float data
//! is too large for RON).
//!
//! - `.lightmap`: per static object, an HDR irradiance lightmap (the baked
//!   GI sampled by the standard shader / GI pipeline). v1 (`CITRSLM1`) holds
//!   each object's RGBA32F map at its own size; v2 (`CITRSLM2`) holds the
//!   GPU-ready array (every layer at the common size) in an
//!   [`HdrTextureFormat`], BC6H by default.
//! - `.lightdata`: light-probe volumes + SH-L1 irradiance, used to light
//!   dynamically moving objects (or leave them unlit, per how it was baked).
//!
//! Both sit next to their `.scene` (same stem) and load with it.

use std::io::{Cursor, Read, Write};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use zestengine_render::{HdrFormatSupport, HdrTextureArray, HdrTextureFormat};

pub const LIGHTMAP_EXTENSION: &str = "lightmap";
pub const LIGHTDATA_EXTENSION: &str = "lightdata";

const LIGHTMAP_MAGIC: &[u8; 8] = b"CITRSLM1";
const LIGHTMAP_MAGIC_V2: &[u8; 8] = b"CITRSLM2";
const LIGHTDATA_MAGIC: &[u8; 8] = b"CITRSLD2";
/// A `.lightdata` whose volumes also carry the Light Volume blend distance and weight.
const LIGHTDATA_MAGIC_V3: &[u8; 8] = b"CITRSLD3";

/// One static object's baked lightmap (size×size RGBA32F).
pub struct LightmapEntry {
    /// Scene object index this lightmap belongs to.
    pub object: u32,
    pub size: u32,
    pub pixels: Vec<f32>,
}

#[derive(Default)]
pub struct LightmapFile {
    pub entries: Vec<LightmapEntry>,
}

/// A probe volume's placement + which SH range it owns.
pub struct ProbeVolumeData {
    /// World → volume-local, column-major (glam `to_cols_array`).
    pub world_to_local: [f32; 16],
    pub size: [f32; 3],
    pub counts: [u32; 3],
    pub sh_base: u32,
    /// True for FluxVoxel voxel volumes (build-time baked from FluxVoxel Lights).
    pub flux_voxel: bool,
    /// Light Volume mode: edge fade distance in metres (0 = a classic cascade volume).
    pub blend_distance: f32,
    /// Light Volume blend weight (1 = default).
    pub weight: f32,
}

#[derive(Default)]
pub struct LightDataFile {
    pub volumes: Vec<ProbeVolumeData>,
    /// SH-L1 per probe: 4 coefficients × RGB = 12 floats.
    pub probes: Vec<[f32; 12]>,
}

// ---- little-endian helpers ------------------------------------------------

fn wr_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn wr_f32s(out: &mut Vec<u8>, v: &[f32]) {
    out.extend_from_slice(bytemuck::cast_slice(v));
}

struct Reader<'a>(Cursor<&'a [u8]>);
impl Reader<'_> {
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.0.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        self.0.read_exact(&mut bytes)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }
    fn array<const N: usize>(&mut self) -> Result<[f32; N]> {
        let v = self.f32s(N)?;
        let mut out = [0.0; N];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

// ---- .lightmap ------------------------------------------------------------

pub fn save_lightmaps(path: impl AsRef<Path>, file: &LightmapFile) -> Result<()> {
    let path = path.as_ref();
    let mut out = Vec::new();
    out.extend_from_slice(LIGHTMAP_MAGIC);
    wr_u32(&mut out, file.entries.len() as u32);
    for e in &file.entries {
        wr_u32(&mut out, e.object);
        wr_u32(&mut out, e.size);
        wr_f32s(&mut out, &e.pixels);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&out))
        .with_context(|| format!("writing {}", path.display()))
}

pub fn load_lightmaps(path: impl AsRef<Path>) -> Result<LightmapFile> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut magic = [0u8; 8];
    if bytes.len() < 8 {
        bail!("{} is too short", path.display());
    }
    magic.copy_from_slice(&bytes[..8]);
    if &magic != LIGHTMAP_MAGIC {
        bail!("{} is not a zestengine .lightmap", path.display());
    }
    let mut r = Reader(Cursor::new(&bytes[8..]));
    let count = r.u32()? as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let object = r.u32()?;
        let size = r.u32()?;
        let pixels = r.f32s((size * size * 4) as usize)?;
        entries.push(LightmapEntry {
            object,
            size,
            pixels,
        });
    }
    Ok(LightmapFile { entries })
}

/// A `.lightmap` as the GPU array it uploads to: layer `i` belongs to scene
/// object `objects[i]`, whose bake was `sizes[i]` square before resampling to
/// the array size.
pub struct LightmapLayers {
    pub objects: Vec<u32>,
    pub sizes: Vec<u32>,
    pub array: HdrTextureArray,
}

impl LightmapLayers {
    /// What the bake's own RGBA32F maps take (the v1 on-disk payload).
    pub fn raw_bytes(&self) -> u64 {
        self.sizes.iter().map(|&s| s as u64 * s as u64 * 16).sum()
    }
}

/// Write `bytes` to `path` via a sibling temp file + rename, so a reader (or a
/// crash mid-write) never sees a half-written sidecar.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

/// Write a v2 `.lightmap`: magic, format code, array size, count, then per
/// layer (object, baked size, byte length, bytes). Returns the file size.
pub fn save_lightmap_layers(path: impl AsRef<Path>, layers: &LightmapLayers) -> Result<u64> {
    let a = &layers.array;
    if layers.objects.len() != a.layers.len() || layers.sizes.len() != a.layers.len() {
        bail!("lightmap layer / object count mismatch");
    }
    let mut out = Vec::with_capacity(a.byte_size() as usize + 20 + a.layers.len() * 12);
    out.extend_from_slice(LIGHTMAP_MAGIC_V2);
    wr_u32(&mut out, a.format.code());
    wr_u32(&mut out, a.size);
    wr_u32(&mut out, a.layers.len() as u32);
    for ((layer, &object), &size) in a.layers.iter().zip(&layers.objects).zip(&layers.sizes) {
        wr_u32(&mut out, object);
        wr_u32(&mut out, size);
        wr_u32(&mut out, layer.len() as u32);
        out.extend_from_slice(layer);
    }
    write_atomic(path.as_ref(), &out)?;
    Ok(out.len() as u64)
}

/// Read a v1 or v2 `.lightmap` in its stored format. v1 maps are resampled to
/// the common array size (as the renderer's raw path does) as RGBA32F.
pub fn read_lightmap_layers(path: impl AsRef<Path>) -> Result<LightmapLayers> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() >= 8 && &bytes[..8] == LIGHTMAP_MAGIC {
        let file = load_lightmaps(path)?;
        let size = zestengine_render::lightmap_array_size(file.entries.iter().map(|e| e.size));
        return Ok(LightmapLayers {
            objects: file.entries.iter().map(|e| e.object).collect(),
            sizes: file.entries.iter().map(|e| e.size).collect(),
            array: HdrTextureArray {
                format: HdrTextureFormat::Rgba32f,
                size,
                layers: file
                    .entries
                    .iter()
                    .map(|e| bytemuck::cast_slice(&zestengine_render::resample_rgba32f(&e.pixels, e.size, size)).to_vec())
                    .collect(),
            },
        });
    }
    if bytes.len() < 20 || &bytes[..8] != LIGHTMAP_MAGIC_V2 {
        bail!("{} is not a zestengine .lightmap", path.display());
    }
    let mut r = Reader(Cursor::new(&bytes[8..]));
    let code = r.u32()?;
    let Some(format) = HdrTextureFormat::from_code(code) else {
        bail!("{}: unknown lightmap format {code}", path.display());
    };
    let size = r.u32()?;
    if size == 0 || size > 16384 {
        bail!("{}: bad lightmap size {size}", path.display());
    }
    let count = r.u32()? as usize;
    let layer_len = format.level_bytes(size, size);
    let mut out = LightmapLayers {
        objects: Vec::with_capacity(count),
        sizes: Vec::with_capacity(count),
        array: HdrTextureArray { format, size, layers: Vec::with_capacity(count) },
    };
    for _ in 0..count {
        out.objects.push(r.u32()?);
        out.sizes.push(r.u32()?);
        let len = r.u32()? as usize;
        if len != layer_len {
            bail!("{}: layer is {len} bytes, expected {layer_len}", path.display());
        }
        let mut layer = vec![0u8; len];
        r.0.read_exact(&mut layer)?;
        out.array.layers.push(layer);
    }
    Ok(out)
}

/// Header summary of a `.lightmap` (payloads skipped, for size stats).
pub struct LightmapFileInfo {
    /// 1 = raw per-object RGBA32F, 2 = GPU-ready array.
    pub version: u32,
    pub format: HdrTextureFormat,
    /// Common layer size (v1: the largest map, as uploaded).
    pub array_size: u32,
    /// Each map's baked size.
    pub sizes: Vec<u32>,
    pub file_bytes: u64,
}

impl LightmapFileInfo {
    /// Bytes of the same maps as raw RGBA32F (the v1 payload).
    pub fn raw_bytes(&self) -> u64 {
        self.sizes.iter().map(|&s| s as u64 * s as u64 * 16).sum()
    }
}

/// Read only the headers of a `.lightmap`, seeking over the texel payloads.
pub fn lightmap_file_info(path: impl AsRef<Path>) -> Result<LightmapFileInfo> {
    use std::io::{Seek, SeekFrom};
    let path = path.as_ref();
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let file_bytes = f.get_ref().metadata()?.len();
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let read = |f: &mut std::io::BufReader<std::fs::File>| -> Result<u32> {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    };
    let mut sizes = Vec::new();
    if &magic == LIGHTMAP_MAGIC {
        let count = read(&mut f)?;
        for _ in 0..count {
            let _object = read(&mut f)?;
            let size = read(&mut f)?;
            f.seek(SeekFrom::Current(size as i64 * size as i64 * 16))?;
            sizes.push(size);
        }
        let array_size = zestengine_render::lightmap_array_size(sizes.iter().copied());
        return Ok(LightmapFileInfo { version: 1, format: HdrTextureFormat::Rgba32f, array_size, sizes, file_bytes });
    }
    if &magic != LIGHTMAP_MAGIC_V2 {
        bail!("{} is not a zestengine .lightmap", path.display());
    }
    let code = read(&mut f)?;
    let format = HdrTextureFormat::from_code(code).with_context(|| format!("unknown lightmap format {code}"))?;
    let array_size = read(&mut f)?;
    let count = read(&mut f)?;
    for _ in 0..count {
        let _object = read(&mut f)?;
        sizes.push(read(&mut f)?);
        let len = read(&mut f)?;
        f.seek(SeekFrom::Current(len as i64))?;
    }
    Ok(LightmapFileInfo { version: 2, format, array_size, sizes, file_bytes })
}

/// Read a `.lightmap` and transcode it to what `support` samples (see
/// [`HdrFormatSupport::upload_format`]). Loader-thread work.
pub fn load_lightmap_layers(path: impl AsRef<Path>, support: HdrFormatSupport, threads: usize) -> Result<LightmapLayers> {
    let mut l = read_lightmap_layers(path)?;
    let to = support.upload_format(l.array.format);
    l.array = crate::hdr_codec::transcode_array(l.array, to, threads)?;
    Ok(l)
}

// ---- .lightdata -----------------------------------------------------------

pub fn save_lightdata(path: impl AsRef<Path>, file: &LightDataFile) -> Result<()> {
    let path = path.as_ref();
    let mut out = Vec::new();
    // Files without Light Volumes keep the original layout (older builds still read them).
    let v3 = file.volumes.iter().any(|v| v.blend_distance > 0.0);
    out.extend_from_slice(if v3 { LIGHTDATA_MAGIC_V3 } else { LIGHTDATA_MAGIC });
    wr_u32(&mut out, file.volumes.len() as u32);
    for v in &file.volumes {
        wr_f32s(&mut out, &v.world_to_local);
        wr_f32s(&mut out, &v.size);
        for c in v.counts {
            wr_u32(&mut out, c);
        }
        wr_u32(&mut out, v.sh_base);
        wr_u32(&mut out, v.flux_voxel as u32);
        if v3 {
            wr_f32s(&mut out, &[v.blend_distance, v.weight]);
        }
    }
    wr_u32(&mut out, file.probes.len() as u32);
    for p in &file.probes {
        wr_f32s(&mut out, p);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&out))
        .with_context(|| format!("writing {}", path.display()))
}

pub fn load_lightdata(path: impl AsRef<Path>) -> Result<LightDataFile> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let v3 = bytes.len() >= 8 && &bytes[..8] == LIGHTDATA_MAGIC_V3;
    if bytes.len() < 8 || (&bytes[..8] != LIGHTDATA_MAGIC && !v3) {
        bail!("{} is not a zestengine .lightdata", path.display());
    }
    let mut r = Reader(Cursor::new(&bytes[8..]));
    let volume_count = r.u32()? as usize;
    let mut volumes = Vec::with_capacity(volume_count);
    for _ in 0..volume_count {
        let world_to_local = r.array::<16>()?;
        let size = r.array::<3>()?;
        let counts = [r.u32()?, r.u32()?, r.u32()?];
        let sh_base = r.u32()?;
        let flux_voxel = r.u32()? != 0;
        let (blend_distance, weight) = if v3 {
            let [b, w] = r.array::<2>()?;
            (b, w)
        } else {
            (0.0, 1.0)
        };
        volumes.push(ProbeVolumeData {
            world_to_local,
            size,
            counts,
            sh_base,
            flux_voxel,
            blend_distance,
            weight,
        });
    }
    let probe_count = r.u32()? as usize;
    let mut probes = Vec::with_capacity(probe_count);
    for _ in 0..probe_count {
        probes.push(r.array::<12>()?);
    }
    Ok(LightDataFile { volumes, probes })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine-lightmap-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("scene.lightmap")
    }

    fn gradient(size: u32) -> Vec<f32> {
        (0..size * size).flat_map(|i| [(i % size) as f32 / size as f32 * 3.0, 0.5, 0.25, 1.0]).collect()
    }

    fn volume(blend: f32, weight: f32) -> ProbeVolumeData {
        ProbeVolumeData { world_to_local: [0.0; 16], size: [4.0, 3.0, 2.0], counts: [2, 2, 2], sh_base: 8, flux_voxel: false, blend_distance: blend, weight }
    }

    /// Light Volume settings round-trip through `.lightdata`; a file with only classic volumes
    /// keeps the original layout (and loads as cascades: blend 0, weight 1).
    #[test]
    fn lightdata_carries_light_volume_settings_and_old_files_load_as_cascades() {
        let dir = std::env::temp_dir().join(format!("zestengine-lightdata-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lv.lightdata");
        let probes = vec![[0.5; 12]; 16];
        save_lightdata(&path, &LightDataFile { volumes: vec![volume(0.0, 1.0), volume(1.5, 2.5)], probes: probes.clone() }).unwrap();
        let back = load_lightdata(&path).unwrap();
        assert_eq!((back.volumes[1].blend_distance, back.volumes[1].weight), (1.5, 2.5));
        assert_eq!((back.volumes[0].blend_distance, back.volumes[0].weight), (0.0, 1.0));
        assert_eq!(back.probes.len(), 16);
        let classic = dir.join("classic.lightdata");
        save_lightdata(&classic, &LightDataFile { volumes: vec![volume(0.0, 1.0)], probes }).unwrap();
        assert_eq!(&std::fs::read(&classic).unwrap()[..8], LIGHTDATA_MAGIC, "no Light Volumes: the original layout");
        let old = load_lightdata(&classic).unwrap();
        assert_eq!((old.volumes[0].blend_distance, old.volumes[0].weight), (0.0, 1.0));
    }

    #[test]
    fn v1_files_still_load_and_upgrade_formats() {
        let path = tmp("v1");
        let file = LightmapFile {
            entries: vec![
                LightmapEntry { object: 4, size: 16, pixels: gradient(16) },
                LightmapEntry { object: 9, size: 8, pixels: vec![2.0; 8 * 8 * 4] },
            ],
        };
        save_lightmaps(&path, &file).unwrap();
        let stored = read_lightmap_layers(&path).unwrap();
        assert_eq!(stored.objects, vec![4, 9]);
        assert_eq!(stored.sizes, vec![16, 8]);
        assert_eq!(stored.array.format, HdrTextureFormat::Rgba32f);
        assert_eq!(stored.array.size, 16);
        assert_eq!(stored.raw_bytes(), (16 * 16 + 8 * 8) * 16);
        let info = lightmap_file_info(&path).unwrap();
        assert_eq!((info.version, info.array_size, info.sizes.clone()), (1, 16, vec![16, 8]));
        assert_eq!(info.raw_bytes(), stored.raw_bytes());
        let support = HdrFormatSupport { bc6h: true, rgb9e5: true };
        let loaded = load_lightmap_layers(&path, support, 1).unwrap();
        assert_eq!(loaded.array.format, HdrTextureFormat::Rgba16f);
        let f = zestengine_render::rgba16f_to_rgba32f(&loaded.array.layers[1]);
        assert!(f.chunks(4).all(|t| t[0] == 2.0));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn v2_bc6h_round_trips_and_falls_back() {
        let path = tmp("v2");
        let map = zestengine_render::BakedLightmap { size: 30, pixels: gradient(30) };
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let array = crate::hdr_codec::encode_lightmaps(&[&map], HdrTextureFormat::Bc6h, 2, &cancel, &|_| {}).unwrap();
        let layers = LightmapLayers { objects: vec![3], sizes: vec![30], array };
        let written = save_lightmap_layers(&path, &layers).unwrap();
        assert_eq!(written, std::fs::metadata(&path).unwrap().len());
        // 30 is not a multiple of 4: 8x8 blocks.
        assert_eq!(written, 20 + 12 + 8 * 8 * 16);
        let info = lightmap_file_info(&path).unwrap();
        assert_eq!((info.version, info.format, info.array_size, info.file_bytes), (2, HdrTextureFormat::Bc6h, 30, written));
        let bc = load_lightmap_layers(&path, HdrFormatSupport { bc6h: true, rgb9e5: true }, 1).unwrap();
        assert_eq!((bc.array.format, bc.objects[0], bc.sizes[0]), (HdrTextureFormat::Bc6h, 3, 30));
        assert_eq!(bc.array.layers[0], layers.array.layers[0]);
        let e5 = load_lightmap_layers(&path, HdrFormatSupport { bc6h: false, rgb9e5: true }, 1).unwrap();
        assert_eq!(e5.array.format, HdrTextureFormat::Rgb9e5);
        assert_eq!(e5.array.layers[0].len(), 30 * 30 * 4);
        let f16 = load_lightmap_layers(&path, HdrFormatSupport { bc6h: false, rgb9e5: false }, 1).unwrap();
        assert_eq!(f16.array.format, HdrTextureFormat::Rgba16f);
        let px = zestengine_render::rgba16f_to_rgba32f(&f16.array.layers[0]);
        let want = gradient(30);
        let err = px.iter().zip(&want).enumerate().filter(|(i, _)| i % 4 != 3).map(|(_, (a, b))| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 0.1, "max abs error {err}");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn astc_hdr_lightmaps_load_with_or_without_device_support() {
        let path = tmp("astc_hdr");
        let format = HdrTextureFormat::AstcHdr4x4;
        let half: Vec<u8> = (0..30 * 30).flat_map(|_| [1.5f32, 0.75, 0.25, 1.0]).flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect();
        let layer = crate::hdr_codec::transcode_level(&half, HdrTextureFormat::Rgba16f, format, 30, 30, 1).unwrap();
        let layers = LightmapLayers {
            objects: vec![1],
            sizes: vec![30],
            array: HdrTextureArray { format, size: 30, layers: vec![layer.clone()] },
        };
        let written = save_lightmap_layers(&path, &layers).unwrap();
        assert_eq!(written, 20 + 12 + 8 * 8 * 16);
        assert_eq!(lightmap_file_info(&path).unwrap().format, format);
        let stored = read_lightmap_layers(&path).unwrap();
        assert_eq!((stored.array.format, &stored.array.layers[0]), (format, &layer));
        // A device without ASTC HDR gets the data decoded to its compact format.
        let loaded = load_lightmap_layers(&path, HdrFormatSupport { bc6h: true, rgb9e5: true }, 1).unwrap();
        assert_eq!(loaded.array.format, HdrTextureFormat::Rgb9e5);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn truncated_or_unknown_files_are_rejected() {
        let path = tmp("bad");
        std::fs::write(&path, b"CITRSLM2\x07\0\0\0\x04\0\0\0\x01\0\0\0").unwrap();
        assert!(read_lightmap_layers(&path).is_err(), "unknown format code");
        let mut v = b"CITRSLM2".to_vec();
        for x in [3u32, 4, 1, 0, 4, 999] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(&path, v).unwrap();
        assert!(read_lightmap_layers(&path).is_err(), "wrong layer length");
        std::fs::write(&path, b"nope").unwrap();
        assert!(read_lightmap_layers(&path).is_err());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
