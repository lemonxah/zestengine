//! Reflection-probe slots. The renderer blends the highest-priority probes
//! around the camera per pixel (slot 0 = binding 7, then bindings 9/11/12 for
//! additional probes). This keeps each slot's cube matching its probe as the
//! camera moves: baked probes upload their faces from the scene's `.reflprobe`,
//! realtime probes (and baked ones without a cube yet) capture live.

use std::collections::HashMap;
use std::path::Path;

use zestengine_render::{HdrCube, HdrFormatSupport, HdrTextureFormat, ReflectionProbeBox, Renderer, TextureData};

pub type Faces = [TextureData; 6];

/// One baked probe's cube as stored.
pub enum ProbeCube {
    /// Raw mip-0 faces (v1-v3 files, uncompressed bakes): the renderer builds
    /// the prefiltered chain on upload.
    Faces(Faces),
    /// The prefiltered chain in an [`HdrTextureFormat`] (compressed bakes,
    /// BC6H on disk; transcoded at load for GPUs without BC).
    Chain(HdrCube),
}

impl ProbeCube {
    /// Bytes of the GPU cube it uploads as (raw faces become an RGBA16F chain).
    pub fn vram_bytes(&self) -> u64 {
        match self {
            Self::Faces(f) => {
                let size = f[0].width.clamp(1, zestengine_render::MAX_BAKED_CUBE_SIZE);
                (0..zestengine_render::cube_mip_count(size)).map(|m| 6 * (size as u64 >> m).max(1).pow(2) * 8).sum()
            }
            Self::Chain(c) => c.byte_size(),
        }
    }

    pub fn format(&self) -> HdrTextureFormat {
        match self {
            Self::Faces(_) => HdrTextureFormat::Rgba16f,
            Self::Chain(c) => c.format,
        }
    }

    /// BC6H-encode raw faces (prefiltering on the CPU first, exactly as the
    /// upload path would; legacy RGBA8 faces convert too). Worker-thread work.
    pub fn compressed(self, threads: usize) -> anyhow::Result<Self> {
        match self {
            Self::Faces(f) => Ok(Self::Chain(zestengine_assets::hdr_codec::encode_cube(
                zestengine_render::reflection_cube_chain(&f),
                threads,
            )?)),
            other => Ok(other),
        }
    }
}

/// Baked probe cubes from a scene's `.reflprobe` sidecar.
#[derive(Default)]
pub struct ReflProbeSet {
    /// One cube per probe object id (format v3+).
    pub entries: HashMap<u128, ProbeCube>,
    /// A v1/v2 single-cube file: used by any baked probe without its own entry.
    pub legacy: Option<ProbeCube>,
}

impl ReflProbeSet {
    pub fn cube(&self, id: u128) -> Option<&ProbeCube> {
        self.entries.get(&id).or(self.legacy.as_ref())
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.legacy.is_none()
    }

    fn all(&self) -> impl Iterator<Item = &ProbeCube> {
        self.entries.values().chain(self.legacy.as_ref())
    }

    pub fn vram_bytes(&self) -> u64 {
        self.all().map(ProbeCube::vram_bytes).sum()
    }

    /// Distinct formats present (for the bake stats line).
    pub fn formats(&self) -> Vec<HdrTextureFormat> {
        let mut f: Vec<HdrTextureFormat> = self.all().map(ProbeCube::format).collect();
        f.sort_by_key(|f| f.code());
        f.dedup();
        f
    }

    /// Transcode chains to what `support` samples (BC6H decodes to E5B9G9R9 /
    /// RGBA16F on GPUs without BC). Loader-thread work.
    pub fn prepare_for(mut self, support: HdrFormatSupport) -> Self {
        for cube in self.entries.values_mut().chain(self.legacy.as_mut()) {
            if let ProbeCube::Chain(c) = cube {
                let to = support.upload_format(c.format);
                if to != c.format {
                    let src = std::mem::replace(c, HdrCube { format: to, size: 0, faces: Default::default() });
                    match zestengine_assets::hdr_codec::transcode_cube(src, to, 1) {
                        Ok(t) => *c = t,
                        Err(e) => tracing::error!("decoding baked reflection probe: {e:#}"),
                    }
                }
            }
        }
        self
    }
}

fn read_faces(bytes: &[u8], at: &mut usize, size: u32, hdr: bool) -> Option<Faces> {
    let face_len = (size as usize) * (size as usize) * if hdr { 8 } else { 4 };
    if bytes.len() < *at + face_len * 6 {
        return None;
    }
    let faces: Vec<TextureData> = (0..6)
        .map(|i| TextureData {
            width: size,
            height: size,
            pixels: bytes[*at + i * face_len..*at + (i + 1) * face_len].to_vec(),
            srgb: false,
            hdr,
        })
        .collect();
    *at += face_len * 6;
    faces.try_into().ok()
}

fn read_chain(bytes: &[u8], at: &mut usize, format: HdrTextureFormat, size: u32, mips: u32) -> Option<HdrCube> {
    if mips != zestengine_render::cube_mip_count(size) {
        return None;
    }
    let mut faces: [Vec<Vec<u8>>; 6] = Default::default();
    for face in &mut faces {
        for m in 0..mips {
            let s = (size >> m).max(1);
            let len = format.level_bytes(s, s);
            face.push(bytes.get(*at..*at + len)?.to_vec());
            *at += len;
        }
    }
    Some(HdrCube { format, size, faces })
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// v4 format code for raw RGBA8 faces (v1 files re-saved); not an HDR format.
const RGBA8_FACES: u32 = 0xFF;
/// v4 entry id of the legacy fallback cube (never a v4 UUID: wrong version bits).
const LEGACY_ID: u128 = u128::MAX;

/// Parse a `.reflprobe` sidecar for upload on this device (see
/// [`ReflProbeSet::prepare_for`]). Empty when missing or malformed.
pub fn read_reflprobes(path: &Path) -> ReflProbeSet {
    read_reflprobes_stored(path).prepare_for(HdrFormatSupport::current())
}

/// Parse a `.reflprobe` in its stored formats. `CITRSRP1` (RGBA8) /
/// `CITRSRP2` (RGBA16F) hold one cube; `CITRSRP3` holds `count` entries of
/// (probe id u128, face size, hdr flag, six faces); `CITRSRP4` holds `count`
/// entries of (probe id u128, format code, face size, mip count, then per face
/// each mip level), where one mip of RGBA16F / RGBA8 means raw faces and a
/// full chain is a prefiltered (usually BC6H) cube.
pub fn read_reflprobes_stored(path: &Path) -> ReflProbeSet {
    let mut set = ReflProbeSet::default();
    let Ok(bytes) = std::fs::read(path) else {
        return set;
    };
    if bytes.len() < 12 {
        return set;
    }
    match &bytes[..8] {
        magic @ (b"CITRSRP1" | b"CITRSRP2") => {
            let hdr = magic == b"CITRSRP2";
            let (Some(size), Some(count)) = (u32_at(&bytes, 8), u32_at(&bytes, 12)) else {
                return set;
            };
            let mut at = 16;
            if count == 6 {
                set.legacy = read_faces(&bytes, &mut at, size, hdr).map(ProbeCube::Faces);
            }
        }
        b"CITRSRP3" => {
            let Some(count) = u32_at(&bytes, 8) else {
                return set;
            };
            let mut at = 12;
            for _ in 0..count {
                let Some(id) = bytes.get(at..at + 16).and_then(|b| b.try_into().ok()).map(u128::from_le_bytes) else {
                    break;
                };
                let (Some(size), Some(hdr)) = (u32_at(&bytes, at + 16), u32_at(&bytes, at + 20)) else {
                    break;
                };
                at += 24;
                match read_faces(&bytes, &mut at, size, hdr != 0) {
                    Some(f) => {
                        set.entries.insert(id, ProbeCube::Faces(f));
                    }
                    None => break,
                }
            }
        }
        b"CITRSRP4" => {
            let Some(count) = u32_at(&bytes, 8) else {
                return set;
            };
            let mut at = 12;
            for _ in 0..count {
                let Some(id) = bytes.get(at..at + 16).and_then(|b| b.try_into().ok()).map(u128::from_le_bytes) else {
                    break;
                };
                let (Some(code), Some(size), Some(mips)) = (u32_at(&bytes, at + 16), u32_at(&bytes, at + 20), u32_at(&bytes, at + 24)) else {
                    break;
                };
                at += 28;
                if size == 0 || size > 8192 {
                    break;
                }
                let format = HdrTextureFormat::from_code(code);
                let cube = match (code, format) {
                    (RGBA8_FACES, _) if mips == 1 => read_faces(&bytes, &mut at, size, false).map(ProbeCube::Faces),
                    (_, Some(HdrTextureFormat::Rgba16f)) if mips == 1 => {
                        read_faces(&bytes, &mut at, size, true).map(ProbeCube::Faces)
                    }
                    (_, Some(f)) => read_chain(&bytes, &mut at, f, size, mips).map(ProbeCube::Chain),
                    _ => None,
                };
                match cube {
                    Some(c) if id == LEGACY_ID => set.legacy = Some(c),
                    Some(c) => {
                        set.entries.insert(id, c);
                    }
                    None => break,
                }
            }
        }
        _ => {}
    }
    set
}

/// Header summary of a `.reflprobe` (payloads skipped, for size stats).
pub struct ReflProbeFileInfo {
    /// Per cube: stored format (None = raw RGBA8 faces), face size, mip count.
    pub cubes: Vec<(Option<HdrTextureFormat>, u32, u32)>,
    pub file_bytes: u64,
}

impl ReflProbeFileInfo {
    /// Bytes of the same cubes as raw RGBA16F mip-0 faces (the v3 payload).
    pub fn raw_bytes(&self) -> u64 {
        self.cubes.iter().map(|&(_, s, _)| 6 * s as u64 * s as u64 * 8).sum()
    }
}

/// Read only the headers of a `.reflprobe`, seeking over the payloads.
pub fn reflprobe_file_info(path: &Path) -> std::io::Result<ReflProbeFileInfo> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let file_bytes = f.get_ref().metadata()?.len();
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let read = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    };
    let mut cubes = Vec::new();
    match &magic {
        b"CITRSRP1" | b"CITRSRP2" => {
            let hdr = &magic == b"CITRSRP2";
            let size = read(&mut f)?;
            cubes.push((hdr.then_some(HdrTextureFormat::Rgba16f), size, 1));
        }
        b"CITRSRP3" => {
            for _ in 0..read(&mut f)? {
                f.seek(SeekFrom::Current(16))?;
                let size = read(&mut f)?;
                let hdr = read(&mut f)? != 0;
                f.seek(SeekFrom::Current(6 * size as i64 * size as i64 * if hdr { 8 } else { 4 }))?;
                cubes.push((hdr.then_some(HdrTextureFormat::Rgba16f), size, 1));
            }
        }
        b"CITRSRP4" => {
            for _ in 0..read(&mut f)? {
                f.seek(SeekFrom::Current(16))?;
                let (code, size, mips) = (read(&mut f)?, read(&mut f)?, read(&mut f)?);
                let format = HdrTextureFormat::from_code(code);
                let bytes: usize = (0..mips)
                    .map(|m| {
                        let s = (size >> m).max(1);
                        format.map_or(s as usize * s as usize * 4, |f| f.level_bytes(s, s))
                    })
                    .sum();
                f.seek(SeekFrom::Current(6 * bytes as i64))?;
                cubes.push((format, size, mips));
            }
        }
        _ => return Err(std::io::Error::other("not a zestengine .reflprobe")),
    }
    Ok(ReflProbeFileInfo { cubes, file_bytes })
}

/// Serialize `set` as a v4 `.reflprobe` (callers drop the legacy cube once
/// per-probe entries exist; a kept one is written under [`LEGACY_ID`]).
/// Written atomically; returns the file size.
pub fn write_reflprobes(path: &Path, set: &ReflProbeSet) -> anyhow::Result<u64> {
    let mut out = Vec::new();
    out.extend_from_slice(b"CITRSRP4");
    let mut ids: Vec<&u128> = set.entries.keys().filter(|&&id| id != LEGACY_ID).collect();
    ids.sort();
    let legacy = set.legacy.as_ref().map(|c| (&LEGACY_ID, c));
    out.extend_from_slice(&((ids.len() + usize::from(legacy.is_some())) as u32).to_le_bytes());
    for (id, cube) in ids.into_iter().map(|id| (id, &set.entries[id])).chain(legacy) {
        out.extend_from_slice(&id.to_le_bytes());
        match cube {
            ProbeCube::Faces(faces) => {
                let code = if faces[0].hdr { HdrTextureFormat::Rgba16f.code() } else { RGBA8_FACES };
                for v in [code, faces[0].width, 1] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                for f in faces {
                    out.extend_from_slice(&f.pixels);
                }
            }
            ProbeCube::Chain(c) => {
                for v in [c.format.code(), c.size, c.mips()] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                for level in c.faces.iter().flatten() {
                    out.extend_from_slice(level);
                }
            }
        }
    }
    zestengine_assets::write_atomic(path, &out)?;
    Ok(out.len() as u64)
}

/// Which probe's cube each renderer slot holds.
#[derive(Default)]
pub struct ReflectionSlots {
    baked: ReflProbeSet,
    held: [Option<u128>; zestengine_render::MAX_REFLECTION_PROBES],
    /// Slot waiting for a live capture (realtime probe, or baked without a cube).
    needs_capture: [bool; zestengine_render::MAX_REFLECTION_PROBES],
    /// Slot's cube belongs to its probe (uploaded, or its capture finished).
    ready: [bool; zestengine_render::MAX_REFLECTION_PROBES],
    /// Slot whose live capture is in flight.
    capturing: Option<usize>,
    /// Realtime refreshes alternate between the slots.
    next_refresh: usize,
}

impl ReflectionSlots {
    /// The baked cubes currently in use (for bake stats).
        pub fn baked(&self) -> &ReflProbeSet {
        &self.baked
    }

    /// New baked cubes (scene load / after a bake); every slot reloads.
    pub fn set_baked(&mut self, baked: ReflProbeSet) {
        self.baked = baked;
        self.reset();
    }

    /// Forget what the slots hold so the next `update` reloads / recaptures.
    pub fn reset(&mut self) {
        self.held = [None; zestengine_render::MAX_REFLECTION_PROBES];
        self.needs_capture = [false; zestengine_render::MAX_REFLECTION_PROBES];
        self.ready = [false; zestengine_render::MAX_REFLECTION_PROBES];
        self.capturing = None;
    }

    /// Per frame: fit the slots to `probes` (best first, from
    /// `LoadedScene::active_reflection_probes`) and return the probe list for
    /// `FrameInput::reflection_probes`. `allow_capture` is false while a bake
    /// owns the capture.
    pub fn update(
        &mut self,
        renderer: &mut Renderer,
        probes: [Option<ReflectionProbeBox>; zestengine_render::MAX_REFLECTION_PROBES],
        allow_capture: bool,
    ) -> [Option<ReflectionProbeBox>; zestengine_render::MAX_REFLECTION_PROBES] {
        let want = probes.map(|p| p.map(|p| p.id));
        for s in 0..zestengine_render::MAX_REFLECTION_PROBES {
            if want[s] == self.held[s] {
                continue;
            }
            self.held[s] = want[s];
            self.ready[s] = false;
            self.needs_capture[s] = false;
            if self.capturing == Some(s) {
                self.capturing = None;
            }
            let Some(p) = probes[s] else {
                continue;
            };
            let upload = match (!p.realtime).then(|| self.baked.cube(p.id)).flatten() {
                Some(ProbeCube::Faces(faces)) => Some(renderer.set_reflection_faces_slot(s as u8, faces)),
                Some(ProbeCube::Chain(cube)) => Some(renderer.set_reflection_cube_slot(s as u8, cube)),
                None => None,
            };
            match upload {
                Some(r) => match r {
                    Ok(()) => self.ready[s] = true,
                    Err(e) => {
                        tracing::error!("uploading baked reflection probe: {e:#}");
                        self.needs_capture[s] = true;
                    }
                },
                None => self.needs_capture[s] = true,
            }
        }
        if let Some(s) = self.capturing
            && !renderer.reflection_capture_in_flight()
        {
            self.ready[s] = true;
            self.capturing = None;
        }
        if allow_capture && self.capturing.is_none() && !renderer.reflection_capture_in_flight() {
            // A slot still waiting for its first cube goes first; then realtime
            // probes refresh, alternating so both track dynamic lighting.
            let first = (0..zestengine_render::MAX_REFLECTION_PROBES)
                .find(|&s| self.needs_capture[s] && probes[s].is_some());
            let refresh = || {
                (0..zestengine_render::MAX_REFLECTION_PROBES)
                    .map(|i| (self.next_refresh + i) % zestengine_render::MAX_REFLECTION_PROBES)
                    .find(|&s| probes[s].is_some_and(|p| p.realtime))
            };
            if let Some(s) = first.or_else(refresh) {
                let p = probes[s].unwrap();
                renderer.request_reflection_capture_slot(
                    glam::Vec3::from(p.capture),
                    p.resolution,
                    s as u8,
                    if p.update_hz > 0.0 { 1.0 / p.update_hz } else { 0.0 },
                    p.time_slicing,
                    p.include_dynamic,
                );
                self.needs_capture[s] = false;
                self.capturing = Some(s);
                self.next_refresh = (s + 1) % zestengine_render::MAX_REFLECTION_PROBES;
            }
        }
        // Slot 0 is the fallback probe; extra slots are only used once their
        // own cube is ready.
        let mut out = probes;
        for (s, slot) in out.iter_mut().enumerate().skip(1) {
            if !self.ready[s] {
                *slot = None;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn faces(size: u32, v: u8) -> Faces {
        std::array::from_fn(|_| TextureData {
            width: size,
            height: size,
            pixels: vec![v; (size * size * 8) as usize],
            srgb: false,
            hdr: true,
        })
    }

    /// Faces of a smooth HDR gradient (0.25..~6) in RGBA16F.
    fn hdr_faces(size: u32) -> Faces {
        std::array::from_fn(|f| TextureData {
            width: size,
            height: size,
            pixels: zestengine_render::rgba32f_to_rgba16f(
                &(0..size * size)
                    .flat_map(|i| {
                        let t = (i % size + i / size) as f32 / size as f32;
                        [0.25 + t * 3.0 + f as f32 * 0.1, 0.5 + t, 1.0, 1.0]
                    })
                    .collect::<Vec<f32>>(),
            ),
            srgb: false,
            hdr: true,
        })
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine-reflprobe-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("scene.reflprobe")
    }

    fn raw(c: Option<&ProbeCube>) -> &Faces {
        match c {
            Some(ProbeCube::Faces(f)) => f,
            _ => panic!("expected raw faces"),
        }
    }

    #[test]
    fn reflprobe_v4_round_trips_raw_per_probe_cubes() {
        let path = tmp("raw");
        let mut set = ReflProbeSet::default();
        set.entries.insert(7, ProbeCube::Faces(faces(4, 1)));
        set.entries.insert(u128::MAX - 3, ProbeCube::Faces(faces(8, 2)));
        let written = write_reflprobes(&path, &set).unwrap();
        assert_eq!(written, std::fs::metadata(&path).unwrap().len());
        let back = read_reflprobes(&path);
        assert_eq!(back.entries.len(), 2);
        assert_eq!(raw(back.cube(7))[0].width, 4);
        assert_eq!(raw(back.cube(u128::MAX - 3))[5].pixels[0], 2);
        assert!(back.cube(99).is_none(), "no legacy cube to fall back to");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn v3_files_still_load() {
        let path = tmp("v3");
        let mut bytes = b"CITRSRP3".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&42u128.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend(std::iter::repeat_n(9u8, 2 * 2 * 8 * 6));
        std::fs::write(&path, bytes).unwrap();
        let set = read_reflprobes(&path);
        assert_eq!(raw(set.cube(42))[3].pixels[0], 9);
        let info = reflprobe_file_info(&path).unwrap();
        assert_eq!(info.cubes, vec![(Some(HdrTextureFormat::Rgba16f), 2, 1)]);
        assert_eq!(info.raw_bytes(), 2 * 2 * 8 * 6);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn legacy_single_cube_serves_any_baked_probe() {
        let path = tmp("legacy");
        let mut bytes = b"CITRSRP2".to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend(std::iter::repeat_n(5u8, 2 * 2 * 8 * 6));
        std::fs::write(&path, bytes).unwrap();
        let set = read_reflprobes(&path);
        assert!(set.entries.is_empty());
        assert_eq!(raw(set.cube(12345))[0].pixels[0], 5);
        // Re-saved (e.g. compressed in place), it stays the fallback cube.
        write_reflprobes(&path, &set).unwrap();
        let again = read_reflprobes(&path);
        assert!(again.entries.is_empty());
        assert_eq!(raw(again.cube(7))[0].pixels[0], 5);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn compressed_cubes_store_every_mip_and_decode_without_bc() {
        let path = tmp("bc6h");
        // 30 is not a multiple of 4: every level pads to whole blocks.
        let cube = ProbeCube::Faces(hdr_faces(30)).compressed(2).unwrap();
        let ProbeCube::Chain(c) = &cube else { panic!("expected a chain") };
        assert_eq!((c.format, c.size, c.mips()), (HdrTextureFormat::Bc6h, 30, 5));
        assert!(c.is_well_formed());
        let c = c.clone();
        let mut set = ReflProbeSet::default();
        set.entries.insert(1, cube);
        let written = write_reflprobes(&path, &set).unwrap();
        let raw_bytes = 6 * 30 * 30 * 8;
        assert!(written < raw_bytes as u64 / 4, "{written} vs raw {raw_bytes}");

        let info = reflprobe_file_info(&path).unwrap();
        assert_eq!(info.cubes, vec![(Some(HdrTextureFormat::Bc6h), 30, 5)]);
        assert_eq!(info.file_bytes, written);
        let stored = read_reflprobes_stored(&path);
        let Some(ProbeCube::Chain(s)) = stored.cube(1) else { panic!() };
        assert_eq!(s.faces, c.faces);
        let bc = stored.prepare_for(HdrFormatSupport { bc6h: true, rgb9e5: true });
        assert_eq!(bc.cube(1).unwrap().format(), HdrTextureFormat::Bc6h);

        for (support, want) in [
            (HdrFormatSupport { bc6h: false, rgb9e5: true }, HdrTextureFormat::Rgb9e5),
            (HdrFormatSupport { bc6h: false, rgb9e5: false }, HdrTextureFormat::Rgba16f),
        ] {
            let set = read_reflprobes_stored(&path).prepare_for(support);
            let Some(ProbeCube::Chain(d)) = set.cube(1) else { panic!() };
            assert_eq!(d.format, want);
            assert!(d.is_well_formed());
        }
        // Decoded mip 0 stays close to the CPU prefilter of the source.
        let set = read_reflprobes_stored(&path).prepare_for(HdrFormatSupport { bc6h: false, rgb9e5: false });
        let Some(ProbeCube::Chain(d)) = set.cube(1) else { panic!() };
        let want = zestengine_render::rgba16f_to_rgba32f(&zestengine_render::reflection_cube_chain(&hdr_faces(30)).faces[2][0]);
        let got = zestengine_render::rgba16f_to_rgba32f(&d.faces[2][0]);
        let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs() / a.max(1.0)).fold(0.0, f32::max);
        assert!(err < 0.1, "max relative error {err}");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn large_captures_clamp_to_the_baked_cube_size() {
        let f = faces(2048, 0);
        let cube = ProbeCube::Faces(f);
        let mips = zestengine_render::cube_mip_count(1024) as u64;
        assert!(cube.vram_bytes() > 6 * 1024 * 1024 * 8 && mips == 11);
        assert!(cube.vram_bytes() < 6 * 2048 * 2048 * 8);
    }
}
