//! MIRRORS of small pure items that sit beside Vulkan code in `zestengine-render`
//! (sampler_cache.rs, impostor.rs, lib.rs render queues). Parity: tests/parity.rs.

#[allow(unused_imports)]
use crate::types::{AlphaMode, TextureData, TextureHandle, Vertex};
use crate::hdr_format::{HdrCube, HdrTextureFormat, MAX_BAKED_CUBE_SIZE};
use std::sync::Arc;
use crate::cube_mip_count;

// ---- sampler_cache.rs ----
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    Point,
    Bilinear,
    #[default]
    Trilinear,
}

/// What a texture asks of its sampler. `Default` matches the renderer's shared
/// material sampler (repeat, trilinear, 8x anisotropy, no bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerKey {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub filter: Filter,
    /// Requested anisotropy (1 = off); clamped to the device limit.
    pub aniso: u8,
    /// Mip LOD bias in 1/8 steps (positive = blurrier).
    pub mip_bias_eighths: i8,
}

impl Default for SamplerKey {
    fn default() -> Self {
        Self { wrap_u: Wrap::Repeat, wrap_v: Wrap::Repeat, filter: Filter::Trilinear, aniso: 8, mip_bias_eighths: 0 }
    }
}

impl SamplerKey {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Point sampling and one-texel clamps make anisotropy meaningless; fold it
    /// out so equal samplers share one `VkSampler`.
    fn normalized(mut self) -> Self {
        if self.filter == Filter::Point {
            self.aniso = 1;
        }
        self.aniso = self.aniso.max(1);
        self
    }
}


// ---- impostor.rs ----
pub const MAX_LAYERS: u32 = 256;

/// A captured impostor as raw data (`zestengine_assets::impostor::Impostor`).
#[derive(Clone, Copy, Debug)]
pub struct ImpostorUpload<'a> {
    /// Views per side.
    pub grid: u32,
    /// Texels per view side.
    pub tile: u32,
    /// Centre and radius of the bounding sphere every view frames (model space).
    pub centre: [f32; 3],
    pub radius: f32,
    /// `(grid * tile)^2` RGBA8 texels, views laid out row by row: albedo (rgb colour, a coverage).
    pub albedo: &'a [u8],
    /// Same layout: aux (rg octahedral normal, b depth, a coverage).
    pub aux: &'a [u8],
}

impl ImpostorUpload<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.grid == 0 || self.tile == 0 || 2 * self.grid * self.grid > MAX_LAYERS {
            return Err(format!("impostor grid {} (tile {}) needs {} layers, at most {MAX_LAYERS}", self.grid, self.tile, 2 * self.grid * self.grid));
        }
        let size = (self.grid * self.tile) as usize;
        let want = size * size * 4;
        if self.albedo.len() != want || self.aux.len() != want {
            return Err(format!("impostor atlas is {} / {} bytes, expected {want}", self.albedo.len(), self.aux.len()));
        }
        if !(self.radius.is_finite() && self.radius > 0.0 && self.centre.iter().all(|c| c.is_finite())) {
            return Err("impostor bounds are not finite".into());
        }
        Ok(())
    }
}

/// Handle of an uploaded impostor atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImpostorHandle(pub(crate) usize);

impl ImpostorHandle {
    pub fn index(self) -> usize {
        self.0
    }
}


// ---- lib.rs ----
/// Unity-style render queue: draws are ordered by this number, with named
/// breakpoints and gaps for fine-tuning. Transparent (≥ TRANSPARENT) sorts
/// back-to-front within the queue.
pub const RENDER_QUEUE_GEOMETRY: i32 = 2000;
pub const RENDER_QUEUE_ALPHA_TEST: i32 = 2450;
pub const RENDER_QUEUE_TRANSPARENT: i32 = 3000;

/// Default render queue for a material given its alpha mode.
pub fn default_render_queue(alpha: AlphaMode) -> i32 {
    match alpha {
        AlphaMode::Opaque => RENDER_QUEUE_GEOMETRY,
        AlphaMode::Cutout => RENDER_QUEUE_ALPHA_TEST,
        AlphaMode::Blend | AlphaMode::Transclipping => RENDER_QUEUE_TRANSPARENT,
    }
}


// ---- skin.rs ----
/// Floats per cloth patch entry: vertex index (u32 bits), position, normal, pad.
pub const CLOTH_PATCH_FLOATS: usize = 8;

/// Select the cloth-affected vertices and pack them for `cloth_override.comp`.
///
/// `animated` is the pose the mesh would have without cloth (skinned and
/// morphed), `simulated` the same vertices after the cloth step. Vertices whose
/// position or normal moved by more than `eps` are cloth-driven and get an
/// entry; pinned vertices (unchanged) stay on plain GPU skinning. `out` is
/// cleared and reused (no per-frame allocation once warm). Returns the entry
/// count.
pub fn pack_cloth_patch(animated: &[Vertex], simulated: &[Vertex], eps: f32, out: &mut Vec<f32>) -> u32 {
    out.clear();
    let mut count = 0;
    for (i, (a, s)) in animated.iter().zip(simulated).enumerate() {
        let moved = (0..3).any(|k| (a.position[k] - s.position[k]).abs() > eps || (a.normal[k] - s.normal[k]).abs() > eps);
        if !moved {
            continue;
        }
        out.push(f32::from_bits(i as u32));
        out.extend_from_slice(&s.position);
        out.extend_from_slice(&s.normal);
        out.push(0.0);
        count += 1;
    }
    count
}

// ---- lib.rs ----
/// Most planar-reflection surfaces rendered per frame (each costs a full
/// mirrored scene pass); extras fall back to the reflection probe.
pub const MAX_PLANAR: usize = 4;


// ---- water_sim.rs ----
/// Displacers one body takes per frame.
pub const MAX_WATER_DISPLACERS: usize = 128;

/// The dispersive layer's constants for a frame (`zestengine_core::EwaveParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSimEwave {
    /// Height per metre of body occupancy change.
    pub gain: f32,
    pub depth: f32,
    /// Damping of every mode (1/s).
    pub damping: f32,
    /// Wave number below which modes fade to the pipe sim (0 = none).
    pub split: f32,
    /// Uniform current carrying the waves (local XZ, m/s).
    pub current: [f32; 2],
    /// Share of a body's occupancy change that starts waves.
    pub body_gain: f32,
    /// Seconds a body's occupancy change is spread over (0 = at once).
    pub smoothing: f32,
    /// Damping (1/s) of the grid's finest waves.
    pub grid_damping: f32,
    /// Share of the layer that shows (0..1).
    pub mix: f32,
}

/// One water body's simulation input for a frame.
#[derive(Clone, Debug)]
pub struct WaterSimBody {
    pub key: u64,
    /// The body's ripple texture ([`crate::Renderer::create_ripple_texture`]).
    pub texture: TextureHandle,
    pub n: u32,
    pub cell: f32,
    /// The window's first cell, in whole cells of the water's local XZ.
    pub origin: [i32; 2],
    /// Edges are walls (the window covers the whole body).
    pub closed: bool,
    /// Absorbing border width in cells.
    pub sponge: u32,
    /// Start again from rest.
    pub reset: bool,
    /// Nothing moving: keep the last result without stepping.
    pub idle: bool,
    pub dt: f32,
    pub substeps: u32,
    pub gravity: f32,
    /// Pipe flow kept per substep after damping.
    pub keep: f32,
    pub momentum: f32,
    pub level_return: f32,
    pub wave_depth: f32,
    /// Scales the bodies' occupancy (not the pushes').
    pub displace: f32,
    /// Kinematic viscosity (m^2/s).
    pub viscosity: f32,
    /// Eddy viscosity rate (1/s; nu_eddy = eddy * cell^2).
    pub eddy: f32,
    /// The dispersive layer, when it runs (`n` a power of two).
    pub ewave: Option<WaterSimEwave>,
    /// Per-cell rest depth, base flow x/z and kind for the window at
    /// `field_origin`; uploaded when `field_version` changes.
    pub field_version: u64,
    pub field_origin: [i32; 2],
    pub field: Arc<Vec<[f32; 4]>>,
    pub flowing: bool,
    pub default_depth: f32,
    /// `zestengine_core::WaterDisplacer::to_gpu` per displacer.
    pub displacers: Vec<[f32; 12]>,
}

/// A finished frame of a body's simulation: per window cell (row-major from
/// `origin`) the height offset, the water velocity x/z and the drag weight.
#[derive(Clone, Debug, Default)]
pub struct WaterSimReadback {
    pub serial: u64,
    pub origin: [i32; 2],
    pub n: u32,
    pub cell: f32,
    pub cells: Vec<[f32; 4]>,
}


// ---- lib.rs: the CPU reflection-cube prefilter (`ReflectionEnv` helpers, as free fns) ----

/// Linear RGBA of texel `i` of `src`: f16 for HDR sources, sRGB-decoded or
/// raw UNORM for 8-bit ones.
fn texel(src: &TextureData, i: usize) -> [f32; 4] {
    if src.hdr {
        let o = i * 8;
        let Some(b) = src.pixels.get(o..o + 8) else { return [0.0, 0.0, 0.0, 1.0] };
        let h = |k: usize| half::f16::from_le_bytes([b[k], b[k + 1]]).to_f32();
        [h(0), h(2), h(4), h(6)]
    } else {
        let o = i * 4;
        let Some(b) = src.pixels.get(o..o + 4) else { return [0.0, 0.0, 0.0, 1.0] };
        let c = |v: u8| {
            let x = v as f32 / 255.0;
            if !src.srgb {
                x
            } else if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        };
        [c(b[0]), c(b[1]), c(b[2]), b[3] as f32 / 255.0]
    }
}

/// 2x2 box downsample of one face.
fn downsample(src: &[[f32; 4]], src_size: u32) -> Vec<[f32; 4]> {
    let dst_size = (src_size / 2).max(1);
    let mut out = Vec::with_capacity((dst_size * dst_size) as usize);
    for y in 0..dst_size {
        for x in 0..dst_size {
            let mut sum = [0.0f32; 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = (x * 2 + dx).min(src_size - 1);
                    let sy = (y * 2 + dy).min(src_size - 1);
                    let p = src[(sy * src_size + sx) as usize];
                    for c in 0..4 {
                        sum[c] += p[c];
                    }
                }
            }
            out.push(sum.map(|v| v * 0.25));
        }
    }
    out
}

/// Resample a source image to a `size×size` cube face (nearest, linear RGBA).
fn resample_face(src: &TextureData, size: u32) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let sx = (x * src.width / size).min(src.width.saturating_sub(1));
            let sy = (y * src.height / size).min(src.height.saturating_sub(1));
            out.push(texel(src, (sy * src.width + sx) as usize));
        }
    }
    out
}

/// CPU prefilter: RGBA16F mip chain per face (mip 0 from `face_mip0`, then
/// 2x2 box downsamples to 1x1).
fn cpu_chain(size: u32, face_mip0: impl Fn(usize, u32) -> Vec<[f32; 4]>) -> HdrCube {
    let mips = cube_mip_count(size);
    let faces = std::array::from_fn(|face| {
        let mut cur = face_mip0(face, size);
        let mut s = size;
        let mut levels = Vec::with_capacity(mips as usize);
        for _ in 0..mips {
            let mut bytes = Vec::with_capacity(cur.len() * 8);
            for p in &cur {
                for c in p {
                    bytes.extend_from_slice(&half::f16::from_f32(*c).to_le_bytes());
                }
            }
            levels.push(bytes);
            if s > 1 {
                cur = downsample(&cur, s);
                s /= 2;
            }
        }
        levels
    });
    HdrCube { format: HdrTextureFormat::Rgba16f, size, faces }
}

/// The CPU prefiltered chain (RGBA16F) for 6 captured faces: mip 0 is the faces
/// resampled to at most [`MAX_BAKED_CUBE_SIZE`], rougher mips box-downsample.
pub fn reflection_cube_chain(faces: &[TextureData; 6]) -> HdrCube {
    let size = faces[0].width.clamp(1, MAX_BAKED_CUBE_SIZE);
    cpu_chain(size, |face, size| resample_face(&faces[face], size))
}
