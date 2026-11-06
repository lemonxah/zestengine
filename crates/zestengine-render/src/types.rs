//! Public data types of the renderer API: CPU-side asset data, material
//! definitions, handles, and per-frame input.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
// `egui` here is the crate's own alias, so the render-free `zestengine-render-lite`
// (which compiles this file too, with an opaque stand-in) needs no egui.
use crate::egui_compat as egui;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// xyz tangent, w handedness (+1/-1), glTF convention.
    pub tangent: [f32; 4],
    /// Second UV set: lightmap / baked-lighting coordinates. Read from the
    /// model's second UV channel, or generated when absent. Kept at the end of
    /// the struct so existing vertex attributes keep their offsets.
    pub uv1: [f32; 2],
    /// Skinning: up to 4 skeleton joint indices influencing this vertex. All
    /// zero on static meshes (the skinned pipeline variant is only used when the
    /// mesh has a skeleton, so static meshes ignore these). Appended last to keep
    /// existing attribute offsets stable.
    pub joints: [u32; 4],
    /// Skinning: the 4 joint weights (normalized). All zero on static meshes.
    pub weights: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
            color: [1.0; 4],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv1: [0.0; 2],
            joints: [0; 4],
            weights: [0.0; 4],
        }
    }
}

pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// True when `uv1` is a real, non-overlapping lightmap UV set — the model's
    /// own TEXCOORD_1, a primitive's generated atlas, or a generated unwrap.
    /// False when `uv1` is just a copy of `uv0` (no dedicated lightmap UV), which
    /// must NOT be baked (overlapping charts → garbage lightmap). Gates the bake
    /// (offer to generate) and the UV-checker preview. `#[serde(default)]`-style
    /// default is `false` so older callers stay conservative.
    pub has_lightmap_uv: bool,
}

/// Pixel data ready for upload. RGBA8 by default; when `hdr` is set, `pixels`
/// holds RGBA **f16** (8 bytes/texel) and uploads as R16G16B16A16_SFLOAT — the
/// native HDR-float path for EXR/HDR sources (no LDR clamp, linear).
#[derive(Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    /// True for color data (albedo, emission), false for data maps
    /// (normal, ORM). Ignored when `hdr` (float is always linear).
    pub srgb: bool,
    /// `pixels` is RGBA f16 linear (R16G16B16A16_SFLOAT) rather than RGBA8.
    pub hdr: bool,
}

/// Pixel format of a (possibly block-compressed) prepared texture. Every
/// block format stores 16 bytes per block: BC uses 4x4 blocks (desktop), ASTC
/// LDR the named footprint (mobile / standalone VR, which lack BC). Raw
/// variants are the fallback for sizes BC can't tile and for devices with
/// neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexFormat {
    /// BC7 sRGB — color maps (albedo, emission).
    Bc7Srgb,
    /// BC7 UNORM — linear data maps (normal, ORM, roughness, AO, metal).
    Bc7Unorm,
    /// BC6H unsigned float — HDR sources (EXR).
    Bc6h,
    /// Uncompressed RGBA8 sRGB.
    RgbaSrgb,
    /// Uncompressed RGBA8 linear.
    RgbaUnorm,
    /// Uncompressed RGBA f16 linear (8 bytes/texel).
    RgbaF16,
    Astc4x4Srgb,
    Astc4x4Unorm,
    Astc5x5Srgb,
    Astc5x5Unorm,
    Astc6x6Srgb,
    Astc6x6Unorm,
    Astc8x8Srgb,
    Astc8x8Unorm,
    /// ASTC 4x4 HDR (float) — HDR sources on mobile GPUs with ASTC HDR support.
    Astc4x4Hdr,
    /// BC1 sRGB (8 bytes per 4x4 block) — opaque colour maps at half BC7's size.
    Bc1Srgb,
    /// BC1 UNORM — opaque linear data at half BC7's size.
    Bc1Unorm,
    /// BC4 UNORM (8 bytes per block) — single-channel maps; samples as (r, 0, 0, 1).
    Bc4Unorm,
    /// BC5 UNORM (16 bytes per block) — two-channel normal maps (RG); samples
    /// as (r, g, 0, 1), so shaders rebuild Z.
    Bc5Unorm,
}

impl TexFormat {
    /// ASTC LDR block edges with a `TexFormat` variant (square footprints).
    pub const ASTC_BLOCKS: [u32; 4] = [4, 5, 6, 8];

    /// The ASTC LDR format for a square `block` footprint (one of
    /// [`Self::ASTC_BLOCKS`]).
    pub fn astc(block: u32, srgb: bool) -> Option<Self> {
        Some(match (block, srgb) {
            (4, true) => Self::Astc4x4Srgb,
            (4, false) => Self::Astc4x4Unorm,
            (5, true) => Self::Astc5x5Srgb,
            (5, false) => Self::Astc5x5Unorm,
            (6, true) => Self::Astc6x6Srgb,
            (6, false) => Self::Astc6x6Unorm,
            (8, true) => Self::Astc8x8Srgb,
            (8, false) => Self::Astc8x8Unorm,
            _ => return None,
        })
    }

    pub fn is_bc(self) -> bool {
        matches!(self, Self::Bc7Srgb | Self::Bc7Unorm | Self::Bc6h | Self::Bc1Srgb | Self::Bc1Unorm | Self::Bc4Unorm | Self::Bc5Unorm)
    }

    pub fn is_astc(self) -> bool {
        self.block_dims() != (1, 1) && !self.is_bc()
    }

    /// Texel footprint of one block (1x1 for uncompressed formats).
    pub fn block_dims(self) -> (u32, u32) {
        match self {
            Self::Bc7Srgb
            | Self::Bc7Unorm
            | Self::Bc6h
            | Self::Bc1Srgb
            | Self::Bc1Unorm
            | Self::Bc4Unorm
            | Self::Bc5Unorm
            | Self::Astc4x4Srgb
            | Self::Astc4x4Unorm
            | Self::Astc4x4Hdr => (4, 4),
            Self::Astc5x5Srgb | Self::Astc5x5Unorm => (5, 5),
            Self::Astc6x6Srgb | Self::Astc6x6Unorm => (6, 6),
            Self::Astc8x8Srgb | Self::Astc8x8Unorm => (8, 8),
            Self::RgbaSrgb | Self::RgbaUnorm | Self::RgbaF16 => (1, 1),
        }
    }

    /// Bytes per block (per texel for uncompressed formats).
    pub fn block_bytes(self) -> usize {
        match self {
            Self::RgbaSrgb | Self::RgbaUnorm => 4,
            Self::RgbaF16 => 8,
            Self::Bc1Srgb | Self::Bc1Unorm | Self::Bc4Unorm => 8,
            _ => 16,
        }
    }

    /// Exact byte size of one `w`×`h` level; partial edge blocks count whole,
    /// matching Vulkan's buffer layout for block formats.
    pub fn level_bytes(self, w: u32, h: u32) -> usize {
        let (bw, bh) = self.block_dims();
        w.max(1).div_ceil(bw) as usize * h.max(1).div_ceil(bh) as usize * self.block_bytes()
    }
}

/// A texture decoded + (optionally) block-compressed off the main thread, with a
/// full mip chain, ready to upload with no further CPU work. Produced by the
/// asset import cache; consumed by `GpuTexture::upload_compressed`.
pub struct CompressedTexture {
    pub format: TexFormat,
    pub width: u32,
    pub height: u32,
    /// Mip levels, largest first; each is the exact byte payload for that level.
    pub mips: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Cutout,
    /// Alpha test below the cutoff AND alpha blending: soft, dithered or
    /// feathered cutout edges (foliage) that still blend like transparent.
    Transclipping,
    Blend,
}

/// How a transparent (`AlphaMode::Blend`) material combines with what is behind it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    /// `src.rgb * a + dst * (1 - a)`.
    #[default]
    Alpha,
    /// `src.rgb * a + dst` (glow, light shafts).
    Additive,
    /// `src.rgb * dst` (shadows, tint films; custom shaders fade by alpha themselves).
    Multiply,
    /// `src.rgb + dst * (1 - a)` for colour already multiplied by alpha.
    Premultiplied,
}

/// Stencil comparison (Unity `Comp`): `(reference & read_mask) <op> (stored & read_mask)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StencilCompare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    #[default]
    Always,
}

/// What a stencil outcome does to the stored value (Unity `Pass` / `Fail` / `ZFail`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

/// Compare + the three outcome ops of one face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct StencilFace {
    pub compare: StencilCompare,
    /// Stencil test passed and depth test passed.
    pub pass_op: StencilOp,
    /// Stencil test failed.
    pub fail_op: StencilOp,
    /// Stencil passed, depth failed.
    pub zfail_op: StencilOp,
}

/// Per-material stencil state (Unity's `Stencil { Ref .. Comp .. Pass .. }`
/// block). The default (`Always`, every op `Keep`) is inactive: the pipeline
/// has stencil testing off and the draw is byte-identical to a material
/// without it.
///
/// Draw order is the whole contract between writers and readers: a material
/// that writes the stencil (`Replace` / `Increment` / `Invert`) must draw
/// BEFORE the materials that test it, so give it a lower render queue
/// (Unity: `Queue = Geometry-1`, or `Geometry+1` for the readers). Within one
/// queue the draw order is the renderer's sort order and carries no such
/// guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct StencilState {
    /// Reference value (`Ref`).
    pub reference: u8,
    /// Mask applied to both the reference and the stored value in the compare.
    pub read_mask: u8,
    /// Bits the pass / fail / zfail ops may change.
    pub write_mask: u8,
    pub front: StencilFace,
    /// Back faces; equal to `front` unless the material set `CompBack` etc.
    pub back: StencilFace,
    /// Unity `ColorMask` as RGBA bits (R = 1, G = 2, B = 4, A = 8; 15 = all).
    /// A pure stencil writer draws with 0: no colour, no G-buffer, no velocity.
    pub color_mask: u8,
    /// Unity `ZWrite`: off for a stencil writer that must not occlude what is
    /// drawn after it (eyes over hair, portals). Default on. A material with it
    /// off is left out of the depth prepass and the GPU-driven batches, so the
    /// queue order alone decides what it covers.
    pub depth_write: bool,
}

impl Default for StencilState {
    fn default() -> Self {
        Self { reference: 0, read_mask: 255, write_mask: 255, front: StencilFace::default(), back: StencilFace::default(), color_mask: 15, depth_write: true }
    }
}

impl StencilState {
    /// Whether the state does anything (otherwise the pipeline skips the stencil stage).
    pub fn is_active(&self) -> bool {
        let live = |f: &StencilFace| {
            f.compare != StencilCompare::Always || f.pass_op != StencilOp::Keep || f.fail_op != StencilOp::Keep || f.zfail_op != StencilOp::Keep
        };
        live(&self.front) || live(&self.back)
    }

    /// Both faces the same (`Comp` / `Pass` / `Fail` / `ZFail` without Front / Back suffix).
    pub fn both(face: StencilFace, reference: u8, read_mask: u8, write_mask: u8) -> Self {
        Self { reference, read_mask, write_mask, front: face, back: face, color_mask: 15, depth_write: true }
    }
}

/// Feature toggles: each combination selects a shader/pipeline variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialFeatures {
    pub toon: bool,
    pub normal_map: bool,
    pub emission: bool,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    /// Water surface shading (engine-generated water grids).
    #[serde(default)]
    pub water: bool,
    /// Per-instance material properties (Unity per-instance colour + one
    /// parameter vec4). When on, the forward pass takes the base colour and
    /// the metallic/roughness/toon parameters from the instance stream so
    /// material variants can share a batch. Off leaves the draw byte-identical.
    #[serde(default)]
    pub per_instance: bool,
    /// Transparent blend equation; only used when `alpha_mode` is `Blend`.
    #[serde(default)]
    pub blend: BlendMode,
    /// Cull front faces (only the inside of a mesh shows). Ignored when
    /// `double_sided`.
    #[serde(default)]
    pub cull_front: bool,
    /// Reads the opaque scene colour (`zestengine_grab` in custom shaders:
    /// refraction, blur, distortion). The main pass then copies the opaque
    /// colour + depth before this material's draw, like it does for water.
    #[serde(default)]
    pub grab: bool,
    /// Stencil test / write (Unity `Stencil` block). Inactive by default; see
    /// [`StencilState`] for the draw-order rule between writers and readers.
    #[serde(default)]
    pub stencil: StencilState,
}

pub use zestengine_core::{MaterialParams, DetailParams, TessParams, TransmissionParams};




/// Per-material "FX" uniform block (std140); the GPU layout of the extended
/// [`MaterialParams`] fields. Five vec4s = 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialFx {
    /// rgb rim colour, w rim power.
    pub rim: [f32; 4],
    /// x rim strength, y ramp smoothness, z emission pulse, w unused.
    pub toon: [f32; 4],
    /// xy base UV scroll, zw emission UV scroll.
    pub scroll: [f32; 4],
    /// xyz matcap layer strengths, w unused.
    pub matcap: [f32; 4],
    /// xyz matcap layer blend modes (0 Add / 1 Multiply / 2 Replace), w unused.
    pub matcap_blend: [f32; 4],
    /// Per-map UV transform: xy = tiling (scale), zw = offset.
    pub albedo_st: [f32; 4],
    pub normal_st: [f32; 4],
    pub orm_st: [f32; 4],
    pub emission_st: [f32; 4],
    /// Invert a split AO/Roughness/Metallic map: x = ao, y = roughness,
    /// z = metallic (0 = as-is, 1 = 1 - value). w unused.
    pub orm_invert: [f32; 4],
    /// Parallax occlusion mapping: x = displacement scale (0 = off). yzw unused.
    pub parallax: [f32; 4],
    /// x = water body slot in the frame UBO (water materials). yzw unused.
    pub water: [f32; 4],
    /// Detail layer: x albedo strength, y normal strength, z ORM strength, w blend mode.
    pub detail: [f32; 4],
    /// Detail layer tiling (xy) and offset (zw).
    pub detail_st: [f32; 4],
    /// Translucency: x strength, y power, z distortion, w map-is-thickness flag.
    pub trans0: [f32; 4],
    /// rgb translucency colour, w see-through transmission.
    pub trans1: [f32; 4],
    /// rgb transmission tint, w refraction strength.
    pub trans2: [f32; 4],
    /// x rough-transmission blur, y thickness, z map G holds transmission, w unused.
    pub trans3: [f32; 4],
    /// Tessellation: x displacement scale, y mid level, z target edge length (px), w max factor.
    pub tess0: [f32; 4],
    /// x Phong smoothing, y 1 when the material is tessellated, zw unused.
    pub tess1: [f32; 4],
}

impl MaterialFx {
    pub fn from_params(p: &MaterialParams) -> Self {
        Self {
            rim: [p.rim_color[0], p.rim_color[1], p.rim_color[2], p.rim_power],
            toon: [p.rim_strength, p.ramp_smoothness, p.emission_pulse, 0.0],
            scroll: [
                p.base_scroll[0],
                p.base_scroll[1],
                p.emission_scroll[0],
                p.emission_scroll[1],
            ],
            matcap: [
                p.matcap_strength[0],
                p.matcap_strength[1],
                p.matcap_strength[2],
                0.0,
            ],
            matcap_blend: [
                p.matcap_blend[0],
                p.matcap_blend[1],
                p.matcap_blend[2],
                0.0,
            ],
            albedo_st: [
                p.albedo_tiling[0],
                p.albedo_tiling[1],
                p.albedo_offset[0],
                p.albedo_offset[1],
            ],
            normal_st: [
                p.normal_tiling[0],
                p.normal_tiling[1],
                p.normal_offset[0],
                p.normal_offset[1],
            ],
            orm_st: [
                p.orm_tiling[0],
                p.orm_tiling[1],
                p.orm_offset[0],
                p.orm_offset[1],
            ],
            emission_st: [
                p.emission_tiling[0],
                p.emission_tiling[1],
                p.emission_offset[0],
                p.emission_offset[1],
            ],
            orm_invert: [
                p.ao_invert as u32 as f32,
                p.roughness_invert as u32 as f32,
                p.metallic_invert as u32 as f32,
                0.0,
            ],
            // y = per-material reflection strength (scales env-cube + SSR/RT).
            // z = screen-space/RT reflection toggle (1 = on, 0 = cube-only).
            // w = per-material reflection mode (0 env, 2 probes, 3 off).
            parallax: [
                // Real (tessellated) displacement replaces the parallax illusion.
                if p.tess.enabled { 0.0 } else { p.displacement_scale },
                p.reflection_intensity,
                if p.screen_reflections { 1.0 } else { 0.0 },
                p.reflection_mode as f32,
            ],
            water: [0.0; 4],
            detail: [
                p.detail.albedo_strength,
                p.detail.normal_strength,
                p.detail.orm_strength,
                p.detail.blend,
            ],
            detail_st: [p.detail.tiling[0], p.detail.tiling[1], p.detail.offset[0], p.detail.offset[1]],
            trans0: [
                p.transmission.translucency,
                p.transmission.power,
                p.transmission.distortion,
                p.transmission.map_is_thickness as u32 as f32,
            ],
            trans1: [
                p.transmission.translucency_color[0],
                p.transmission.translucency_color[1],
                p.transmission.translucency_color[2],
                p.transmission.transmission,
            ],
            trans2: [p.transmission.tint[0], p.transmission.tint[1], p.transmission.tint[2], p.transmission.refraction],
            trans3: [
                p.transmission.blur,
                p.transmission.thickness,
                p.transmission.map_has_transmission as u32 as f32,
                0.0,
            ],
            tess0: [p.tess.scale, p.tess.midlevel, p.tess.edge_px.max(1.0), p.tess.max_factor.clamp(1.0, 64.0)],
            tess1: [p.tess.phong.clamp(0.0, 1.0), p.tess.enabled as u32 as f32, 0.0, 0.0],
        }
    }
}

/// Fur shells (UNITY-12): the mesh drawn `layers` more times, each pushed out along its
/// normals, keeping only the pixels inside a procedural strand grid. Opaque and
/// alpha-tested; shaded by a wrapped key-light term, not the full PBR loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FurParams {
    /// Shell count (clamped to 2..=48 when drawn).
    pub layers: u32,
    /// How far the outermost shell sits from the skin, in world units.
    pub length: f32,
    /// Strands per UV unit along each axis.
    pub density: f32,
    /// Linear colour at the strand tip and root.
    pub tip_color: [f32; 3],
    pub root_color: [f32; 3],
}

/// Texture slots of a set-1 material set: the 16 original ones, then detail
/// albedo / normal / ORM / mask and the translucency (thickness) map.
pub const MATERIAL_SLOTS: usize = 21;

/// Water bodies the scene storage buffer holds (a water material picks one by slot).
pub const MAX_WATER_BODIES: usize = 16;
/// Reflection-probe cubes blended per pixel (slot 0 + extra slots).
pub const MAX_REFLECTION_PROBES: usize = 4;
/// Churned foam spots per water body (waterfall plunges, whirlpools).
pub const MAX_WATER_FOAM: usize = 4;
/// Waves summed per water body.
pub const MAX_WATER_WAVES: usize = 32;

/// One water surface's waves and look, as the standard shader's water path
/// reads them (std430, `GpuWater` in standard_common.glsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuWater {
    /// Per wave: xy direction (local XZ), z wave number, w angular speed.
    pub wave_a: [[f32; 4]; MAX_WATER_WAVES],
    /// Per wave: x horizontal amplitude (Q*A), y amplitude, z the normal's
    /// y slope term. The eight w lanes pack absorption RGB + IOR, then
    /// scattering RGB + anisotropy; the remaining wave lanes are reserved.
    pub wave_b: [[f32; 4]; MAX_WATER_WAVES],
    /// rgb shallow colour, w absorption depth.
    pub shallow: [f32; 4],
    /// rgb deep colour, w clarity.
    pub deep: [f32; 4],
    /// x roughness, y ripple strength, z ripple scale, w foam width.
    pub look0: [f32; 4],
    /// x crest foam, y refraction, z wave count, w summed amplitude.
    pub look1: [f32; 4],
    /// Interactive ripple window: xy centre (local XZ), z width (0 = none).
    /// The field is the water material's slot-15 texture (r height, gb its
    /// local XZ gradient).
    pub ripple: [f32; 4],
    /// World to the body's local space (caustics on what lies under it).
    pub to_local: [[f32; 4]; 4],
    /// x, y half extents in local XZ (0 = endless), z rest surface (local
    /// y), w caustic strength (0 = none).
    pub bounds: [f32; 4],
    /// Churned spots: xy centre (local XZ), z radius, w foam amount.
    pub foam: [[f32; 4]; MAX_WATER_FOAM],
    /// x: lights under the surface throw caustics out of it (pools); y:
    /// shore depth (m) the waves calm under (0 = off), zw: half extents
    /// (local XZ) of the shore-depth texture (material slot 14).
    pub extra: [f32; 4],
    /// x: reflection mode (0 environment, 1 planar, 2 probes only, 3 off);
    /// yz: the body's current (local m/s), so the caustics drift with it.
    pub reflect: [f32; 4],
}

impl GpuWater {
    pub fn set_optics(&mut self, absorption: [f32; 4], scattering: [f32; 4]) {
        for i in 0..4 { self.wave_b[i][3] = absorption[i]; self.wave_b[i + 4][3] = scattering[i]; }
    }
    pub fn absorption(&self) -> [f32; 4] { std::array::from_fn(|i| self.wave_b[i][3]) }
    pub fn scattering(&self) -> [f32; 4] { std::array::from_fn(|i| self.wave_b[i + 4][3]) }
}

impl Default for GpuWater {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshHandle(pub(crate) usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureHandle(pub(crate) usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialHandle(pub(crate) usize);

impl MeshHandle {
    /// Placeholder for a scene built without a renderer (a headless server):
    /// never drawn, never a valid renderer index.
    pub const NONE: Self = Self(usize::MAX);

    /// The renderer-internal resource index (used by resource liveness
    /// tracking to free exactly the resources an origin owns).
    pub fn index(self) -> usize {
        self.0
    }
}

impl TextureHandle {
    /// Placeholder for a scene built without a renderer (a headless server):
    /// never drawn, never a valid renderer index.
    pub const NONE: Self = Self(usize::MAX);

    /// The renderer-internal resource index.
    pub fn index(self) -> usize {
        self.0
    }
}

impl MaterialHandle {
    /// Placeholder for a scene built without a renderer (a headless server):
    /// never drawn, never a valid renderer index.
    pub const NONE: Self = Self(usize::MAX);

    /// The renderer-internal resource index.
    pub fn index(self) -> usize {
        self.0
    }
}
/// A registered custom fragment shader (compiled SPIR-V).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderId(pub(crate) usize);

/// Fallback bound to an unassigned custom-shader texture property.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CustomTextureDefault {
    #[default]
    White,
    Black,
    Normal,
    Gray,
}

#[derive(Default)]
pub struct MaterialDesc {
    pub name: String,
    pub params: MaterialParams,
    pub features: MaterialFeatures,
    pub albedo: Option<TextureHandle>,
    pub normal: Option<TextureHandle>,
    pub orm: Option<TextureHandle>,
    pub emission: Option<TextureHandle>,
    // Extended texture slots (bindings 5-12).
    pub opacity: Option<TextureHandle>,
    pub emission_mask: Option<TextureHandle>,
    pub matcap: [Option<TextureHandle>; 3],
    pub matcap_mask: [Option<TextureHandle>; 3],
    /// Split AO / Roughness / Metallic maps (bindings 13-15). When assigned each
    /// multiplies the corresponding packed-ORM channel (default white = no-op),
    /// so a material can use a packed ORM, separate maps, or a mix.
    pub ao: Option<TextureHandle>,
    pub roughness: Option<TextureHandle>,
    pub metallic: Option<TextureHandle>,
    /// Height / displacement map (binding 16) for parallax occlusion mapping.
    pub displacement: Option<TextureHandle>,
    /// Detail albedo / normal / ORM / mask and the translucency (thickness) map
    /// (bindings 17-21). Unassigned = a neutral 1x1.
    pub detail: [Option<TextureHandle>; 5],
    /// Render with the error swirl shader (broken/missing material).
    pub error: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraData {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
}

/// Scene-wide lighting that isn't tied to a single light object: the ambient
/// fill and the "key" directional light exposed to custom shaders through the
/// `u_light_dir` / `u_light_color` macros. The full per-object light set is
/// passed separately as [`LightInstance`]s.
#[derive(Clone, Copy, Debug)]
pub struct LightData {
    /// Direction the key light travels (from the light toward the scene).
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub ambient: [f32; 3],
    /// Environment / skylight intensity that scales the fallback skybox IBL
    /// (the specular reflection of the environment cubemap on metallic/smooth
    /// surfaces). 0 = the skybox does NOT light the scene, so with no analytic
    /// lights and no ambient the scene is black instead of showing skybox
    /// reflections on metals. Explicit reflection probes set their own intensity
    /// and are unaffected. Defaults to 1 (full skybox IBL).
    pub env_intensity: f32,
}

impl Default for LightData {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.4, -1.0, -0.3),
            color: [1.0, 0.98, 0.92],
            intensity: 3.0,
            ambient: [0.13, 0.14, 0.18],
            env_intensity: 1.0,
        }
    }
}

/// What shape of light a [`LightInstance`] emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

/// One light gathered from the scene for a frame. Directional lights use only
/// `direction`; point lights only `position`; spot lights both.
#[derive(Clone, Copy, Debug)]
pub struct LightInstance {
    pub kind: LightKind,
    /// World-space position (point/spot).
    pub position: Vec3,
    /// World-space travel direction, normalized (directional/spot).
    pub direction: Vec3,
    /// Linear RGB color (not yet scaled by intensity).
    pub color: [f32; 3],
    pub intensity: f32,
    /// Distance at which point/spot reach zero.
    pub range: f32,
    /// Spot inner cone full angle (degrees); full brightness inside it.
    pub spot_inner_deg: f32,
    /// Spot outer cone full angle (degrees); zero brightness outside it.
    pub spot_outer_deg: f32,
    /// Render shadows for this light (depth map from its POV).
    pub cast_shadows: bool,
    /// Soft (PCF kernel) vs hard (single tap) shadow edge. Ignored when
    /// `cast_shadows` is false.
    pub soft_shadows: bool,
    /// Source radius for realtime GI shadow sampling (world units for local
    /// lights, angular spread for directional lights). Zero is a point source.
    pub radius: f32,
    /// Light-clip-space depth-compare bias.
    pub shadow_bias: f32,
    /// This light's contribution is BAKED into lightmaps (Baked/Mixed mode + a
    /// bake exists). It's kept in the realtime pass so **non-lightmapped** objects
    /// (dynamic objects, anything added after the bake) still receive it; the
    /// shader skips it for lightmapped objects, which already have it baked in. So
    /// a baked scene's dynamic objects aren't black. `false` = realtime (all objs).
    pub baked: bool,
}

/// Shape of an area light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaShape {
    Rect,
    Disc,
}

/// One area light (rectangle or disc emitter) for a frame, shaded with linearly
/// transformed cosines: diffuse and specular, no shadows. Emits towards
/// `cross(right, up)`. `intensity` is the emitted radiance (the colour a white
/// diffuse surface facing the light fully would reflect at unit albedo), not an
/// irradiance. At most [`MAX_AREA_LIGHTS`] (the nearest to the camera) are drawn.
#[derive(Clone, Copy, Debug)]
pub struct AreaLightInstance {
    pub shape: AreaShape,
    pub position: Vec3,
    /// Unit axes of the emitter plane; `cross(right, up)` is the emitting side.
    pub right: Vec3,
    pub up: Vec3,
    /// Half width / half height (disc: radii along right / up).
    pub half_size: [f32; 2],
    pub color: [f32; 3],
    pub intensity: f32,
    /// Distance at which the light reaches zero.
    pub range: f32,
    /// Lights the back too.
    pub two_sided: bool,
    /// Optional emission texture (modulates the colour; a camera or video texture
    /// updates live). Copied into the area-light atlas every frame.
    pub texture: Option<TextureHandle>,
}

/// Area lights drawn per frame (the nearest to the camera win).
pub const MAX_AREA_LIGHTS: usize = 8;

/// How a draw takes part in shadows (the object's "Cast Shadows" setting).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShadowCasting {
    #[default]
    On,
    /// Never drawn into a shadow map; renders normally.
    Off,
    /// Casts from both faces. The shadow pass already rasterizes with culling
    /// off, so this casts exactly like `On`.
    TwoSided,
    /// Drawn into shadow maps only, never by a camera.
    ShadowsOnly,
}

impl ShadowCasting {
    /// Drawn into shadow maps (and blocks bake / RT shadow rays).
    pub fn casts(self) -> bool {
        self != Self::Off
    }
    /// Drawn by cameras: main pass, prepass, captures, mirrors, previews.
    pub fn visible(self) -> bool {
        self != Self::ShadowsOnly
    }
}

/// Stable scene identity shared by raster draws and software GI geometry.
/// Material values, draw order and transforms are deliberately not identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GiSource {
    pub object: u128,
    pub slot: u32,
}

#[derive(Clone, Copy)]
pub struct DrawCmd {
    /// None for standalone geometry without an associated scene object.
    pub gi_source: Option<GiSource>,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub transform: Mat4,
    /// Editor selection glow, 0.0 = off. Per-draw, not part of the material.
    pub highlight: f32,
    /// Mesh AABB center in object space; the outline pass inflates radially
    /// from it so hard-edged meshes stay watertight (no corner gaps).
    pub mesh_center: Vec3,
    /// Object-space bounding-sphere radius (AABB half-diagonal) around
    /// `mesh_center`, scaled by the transform to frustum-cull the main camera
    /// pass. 0 disables culling for this draw (always drawn).
    pub bound_radius: f32,
    /// Baked-lightmap array layer for this object's static GI, or -1 when the
    /// object has no lightmap (sample probes / flat ambient instead).
    pub lightmap_layer: i32,
    /// This object's lightmap resolution (texels/side) for the UV-checker
    /// preview — its native baked size, or the would-be size from bake settings
    /// for an un-baked static object. 0 = not lightmapped.
    pub lightmap_size: u32,
    /// Marked static in the scene. A BAKED reflection-probe capture draws only
    /// static geometry (Unity semantics); dynamic objects stay out of the cube.
    pub static_geometry: bool,
    /// Index into `FrameInput::planar` when this draw IS a planar-reflective
    /// surface (it samples its mirrored render and is excluded from its own
    /// planar pass), or -1.
    pub planar_index: i32,
    /// Hidden by baked occlusion culling from the culling camera's cell: skipped
    /// in the main camera pass, still drawn into shadows / probes / captures.
    pub occluded: bool,
    /// Which passes draw it: shadow maps, cameras, or both.
    pub shadow_casting: ShadowCasting,
    /// Per-instance base colour override (`None` = the material's base colour).
    /// Only read by the forward pass when the material opts in
    /// (`MaterialFeatures::per_instance`).
    pub instance_color: Option<[f32; 4]>,
    /// Per-instance material parameters override (metallic, roughness, toon
    /// steps, toon/PBR blend; `None` = the material's values). Only read when
    /// the material opts in.
    pub instance_param: Option<[f32; 4]>,
}

/// A world-space box-projector decal. The renderer projects its texture onto
/// opaque surfaces inside the box (see the deferred decal pass).
#[derive(Clone, Copy, Debug)]
pub struct Decal {
    pub texture: TextureHandle,
    /// World -> projector-local box transform (inverse of the object world).
    pub world_to_local: Mat4,
    /// World-space projector +Z direction (the direction the projector faces).
    pub forward: Vec3,
    /// Box width/height in projector-local units.
    pub size: [f32; 2],
    /// Projection depth along the projector's -Z.
    pub depth: f32,
    /// Fade-out fraction at the far end of the box.
    pub fade: f32,
    /// UV tiling over the box.
    pub tiling: [f32; 2],
    /// Linear RGBA tint multiplied with the decal texture.
    pub tint: [f32; 4],
}

pub struct EguiDraw {
    pub pixels_per_point: f32,
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
}

// ---------------------------------------------------------------- baking

/// One static mesh instance fed to the lighting bake. The bake rasters this
/// object's lightmap in `uv1` space and stores incoming light per texel.
#[derive(Clone, Copy, Debug)]
pub struct BakeInstance {
    pub mesh: MeshHandle,
    pub transform: Mat4,
    /// Square lightmap resolution for this object (texels per side).
    pub lightmap_size: u32,
    /// Diffuse albedo for indirect bounces (material base color, linear).
    pub albedo: [f32; 3],
    /// Emitted radiance (color × intensity, linear); surfaces glow into GI.
    pub emission: [f32; 3],
    /// Metalness [0,1]. The diffuse bounce uses albedo·(1−metallic): a metal has
    /// no diffuse albedo (its energy is specular/F0), so it absorbs the diffuse
    /// GI bounce instead of re-radiating it — what stops light pooling/growing in
    /// metal-on-metal contact cavities.
    pub metallic: f32,
    /// Roughness [0,1]. Carried for the surface cache / specular GI; the diffuse
    /// bounce magnitude is roughness-independent (Lambert), so it only modulates
    /// the (future) specular trace, not the diffuse albedo.
    pub roughness: f32,
    /// `Off` instances don't block shadow rays; `ShadowsOnly` ones block only
    /// shadow rays (invisible to bounce rays).
    pub shadow_casting: ShadowCasting,
}

/// A light contributing to the bake. Baked lights contribute their full
/// direct term; Mixed lights contribute only the indirect/shadow that the
/// realtime path can't do (the engine decides which to include).
#[derive(Clone, Copy, Debug)]
pub struct BakeLight {
    pub kind: LightKind,
    pub position: Vec3,
    pub direction: Vec3,
    /// Linear RGB × intensity (already scaled).
    pub color: [f32; 3],
    pub range: f32,
    pub spot_inner_deg: f32,
    pub spot_outer_deg: f32,
    /// Light source radius (world units) for soft shadows; 0 = hard.
    pub radius: f32,
}

/// Everything the GPU bake needs in one renderer-agnostic bundle.
pub struct BakeInput<'a> {
    pub instances: &'a [BakeInstance],
    pub lights: &'a [BakeLight],
    /// World-space probe centers (flattened across all volumes).
    pub probes: &'a [Vec3],
    /// Constant sky/ambient radiance for rays that escape the scene.
    pub sky_color: [f32; 3],
    /// Indirect bounces per path (0 = direct + sky only).
    pub bounces: u32,
    /// Paths traced per lightmap texel / per probe direction.
    pub samples: u32,
    /// Skip per-instance lightmap tracing and bake only the probes (the
    /// instances still act as occluders/bouncers). Used by the realtime-GI
    /// preview, which only needs probe SH each update.
    pub probes_only: bool,
    /// Fraction of each GPU bake submission's duration to then idle the GPU, so the
    /// desktop compositor gets a share instead of the bake hogging the GPU (which
    /// froze the machine). 0 = no throttle (the realtime-GI preview uses 0 so it
    /// stays responsive); the offline bake passes the user's `gpu_throttle`.
    pub gpu_idle_frac: f32,
}

/// One baked lightmap: `size`×`size` RGBA32F (rgb = irradiance, a = validity).
#[derive(Clone, Debug)]
pub struct BakedLightmap {
    pub size: u32,
    pub pixels: Vec<f32>,
}

/// SH-L1 irradiance for one probe: 4 coefficients × RGB, plus an SH-L1 of the
/// directional distance-to-geometry (4 scalar coefficients) used for DDGI-style
/// visibility weighting at sample time. `dist` left zero (the bake path) disables
/// the visibility test; the standard shader then weights probes by trilinear
/// only. The software march fills it so probes occluded from a fragment (e.g.
/// behind a wall) are down-weighted, killing light leaks.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeSh {
    pub coeffs: [[f32; 3]; 4],
    pub dist: [f32; 4],
    /// SH-L1 of the SQUARED first-hit distance (second moment), for a two-moment
    /// Chebyshev visibility test (DDGI-style) — smooth occlusion instead of a
    /// hard threshold. Zero = no data (the test falls back to single-moment).
    pub dist2: [f32; 4],
    /// World offset of the probe from its grid position (FluxVoxel relocation
    /// moves probes out of geometry); the visibility test uses the moved spot.
    pub offset: [f32; 3],
}

/// One Light Volume (a LightProbeVolume in Light Volume mode): an oriented box with a probe grid
/// of baked SH, blended with the other Light Volumes by `weight` x edge fade for objects that
/// have no lightmap.
#[derive(Clone, Debug)]
pub struct LightVolumeData {
    /// World to the volume's local space (the grid spans -size/2 .. +size/2).
    pub world_to_local: Mat4,
    pub size: [f32; 3],
    pub counts: [u32; 3],
    /// The volume fades out over this distance (m) inward from its faces. Must be > 0.
    pub blend_distance: f32,
    pub weight: f32,
    /// `counts[0] * counts[1] * counts[2]` probes, x fastest then y then z.
    pub probes: Vec<ProbeSh>,
}

/// Result of a bake: a lightmap per instance (same order) and SH per probe.
#[derive(Clone, Debug, Default)]
pub struct BakeOutput {
    pub lightmaps: Vec<BakedLightmap>,
    pub probes: Vec<ProbeSh>,
}

/// Per-frame render statistics (last completed frame). Categories grow as
/// render passes are added (reflections, probes, shadows…).
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    /// Total scene draw calls this frame (excluding egui).
    pub draw_calls: u32,
    /// Draw calls that drew more than one object (GPU instancing).
    pub instanced_draws: u32,
    /// Opaque + cutout draws.
    pub opaque_draws: u32,
    /// Extra draws caused by transparency (alpha-blended, sorted pass).
    pub transparent_draws: u32,
    /// Extra draws caused by the selection outline pass.
    pub outline_draws: u32,
    /// Draws using the error/missing-material swirl shader.
    pub error_draws: u32,
    /// Pipeline binds (state switches) this frame.
    pub pipeline_binds: u32,
    /// Distinct materials referenced by this frame's draws.
    pub materials_drawn: u32,
    /// Compiled shader-variant pipelines in the cache.
    pub pipeline_variants: u32,
    /// Total GPU time for the last completed frame (ms), from timestamp queries.
    /// 0 when the device/queue does not support timestamps.
    pub gpu_frame_ms: f32,
    /// GPU time in Flux GI (depth prepass, gather, denoise and integrate), in
    /// milliseconds. Zero when that view's realtime GI pass did not run.
    pub gpu_gi_ms: f32,
    /// GPU per-pass breakdown (ms) for the active path, from timestamp zones.
    /// A pass that didn't run this frame reads 0.
    pub gpu_shadows_ms: f32,
    pub gpu_scene_ms: f32,
    pub gpu_reflect_ms: f32,
    pub gpu_post_ms: f32,
    pub gpu_egui_ms: f32,
    pub gpu_cam_preview_ms: f32,
    /// Flux trace sub-passes (ms): screen-probe gather, denoise, integrate.
    pub gpu_gi_gather_ms: f32,
    pub gpu_gi_denoise_ms: f32,
    pub gpu_gi_integrate_ms: f32,
    /// Persistent surface-lighting update within the main view's Flux pass.
    pub gpu_gi_surface_ms: f32,
    /// Camera preview has independent zones: it can run in the same submission
    /// as Scene, or alone while the Scene tab is hidden.
    pub gpu_camera_gi_ms: f32,
    pub gpu_camera_gi_gather_ms: f32,
    pub gpu_camera_gi_denoise_ms: f32,
    pub gpu_camera_gi_integrate_ms: f32,
    pub gpu_camera_gi_surface_ms: f32,
    /// Screen-space ambient occlusion (prep, AO, denoise, upsample).
    pub gpu_ao_ms: f32,
    /// Anti-aliasing resolve (TAA; FXAA runs inside the post pass).
    pub gpu_aa_ms: f32,
    /// GPU-driven culling is on: the main camera's opaque / alpha-test draws,
    /// the shadow views and VR eyes are culled by compute and drawn indirect.
    pub gpu_culling: bool,
    /// Indirect draw calls recorded since the previous frame (one per batch).
    pub indirect_draws: u32,
    /// Main-camera GPU cull counts, read back from the frame slot's previous
    /// submission (no CPU wait): instances tested, drawn, rejected by the
    /// frustum / distance / size tests, and rejected by Hi-Z occlusion (after
    /// the late pass).
    pub gpu_cull_tested: u32,
    pub gpu_cull_visible: u32,
    pub gpu_cull_frustum: u32,
    pub gpu_cull_occluded: u32,
    /// Of the visible ones: instances last frame's depth hid that the late
    /// Hi-Z pass found visible in this frame's depth and drew the same frame
    /// (disocclusions a one-phase cull would have drawn a frame late).
    pub gpu_cull_recovered: u32,
    /// Shadow-caster instances drawn by the GPU-culled shadow views.
    pub gpu_cull_shadow_visible: u32,
    /// Instances drawn by the last GPU-culled VR eye.
    pub gpu_cull_eye_visible: u32,
    /// GPU time of the cull dispatches (early + late pass, the late pass's
    /// pyramid included) and of the next frame's Hi-Z pyramid build (ms).
    pub gpu_cull_ms: f32,
    pub gpu_hiz_ms: f32,
    /// GPU time of the skinning dispatches and of the blendshape (morph)
    /// dispatches that feed them (ms). 0 when nothing skinned/morphed this
    /// frame or timestamp zones are inactive (profiler window / frame stats /
    /// `ZESTENGINE_BENCH_AVATARS`).
    pub gpu_skin_ms: f32,
    pub gpu_morph_ms: f32,
    /// GPU cluster path counters and timings (GEO-4).
    pub cluster: crate::ClusterStats,
}

/// Spatial / temporal upscaler applied between the HDR resolve and tonemap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Upscaler {
    /// Native resolution, no upscaling.
    #[default]
    Off,
    /// AMD FSR 3.1 temporal upscaling (frame generation is a separate toggle).
    Fsr3,
}

/// FSR quality presets, each a render scale (fraction of the output the scene
/// renders at before the upscaler reconstructs it), following AMD's standard
/// per-quality resolution ratios.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpscaleQuality {
    /// Native resolution (the upscaler is off).
    Native,
    #[default]
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl UpscaleQuality {
    pub const ALL: [UpscaleQuality; 5] = [
        Self::Native,
        Self::Quality,
        Self::Balanced,
        Self::Performance,
        Self::UltraPerformance,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "Native",
            Self::Quality => "Quality",
            Self::Balanced => "Balanced",
            Self::Performance => "Performance",
            Self::UltraPerformance => "Ultra Performance",
        }
    }

    /// The render scale for this quality (AMD's standard ratios: 1.0 / 1.5 /
    /// 1.7 / 2.0 / 3.0).
    pub fn render_scale(self) -> f32 {
        match self {
            Self::Native => 1.0,
            Self::Quality => 1.0 / 1.5,
            Self::Balanced => 1.0 / 1.7,
            Self::Performance => 0.5,
            Self::UltraPerformance => 1.0 / 3.0,
        }
    }

    /// The preset whose render scale is closest to `scale` (`Native` at 1.0,
    /// i.e. the upscaler off).
    pub fn from_render_scale(scale: f32) -> Self {
        if scale >= 1.0 {
            return Self::Native;
        }
        Self::ALL
            .iter()
            .filter(|q| **q != Self::Native)
            .min_by(|a, b| (a.render_scale() - scale).abs().total_cmp(&(b.render_scale() - scale).abs()))
            .copied()
            .unwrap_or(Self::Quality)
    }
}

pub struct FrameInput<'a> {
    pub clear_color: [f32; 4],
    pub camera: CameraData,
    /// Ambient + key-light fallback (the latter feeds custom-shader macros).
    pub light: LightData,
    /// Every light gathered from the scene this frame. When empty, the
    /// standard shader falls back to the single key directional in `light`.
    pub lights: &'a [LightInstance],
    /// Lights for the reflection-probe cube capture — the FULL set including
    /// Baked/Mixed lights that `lights` drops once a bake exists. The cube
    /// renders the scene without lightmaps, so it needs the analytic lights or
    /// it captures dark. Empty → fall back to `lights`.
    pub capture_lights: &'a [LightInstance],
    /// FluxVoxel: moving emitters as point lights (colour already includes the
    /// GI gain), shaded per pixel with the grid's injection kernel as indirect
    /// light instead of being injected into the probe grid.
    pub voxel_emitters: &'a [BakeLight],
    /// When set, the scene is also rendered from this camera into the offscreen
    /// preview target (the editor's Camera tab). `None` skips that pass.
    pub camera_preview: Option<CameraData>,
    /// A model-only offscreen view (the rig Configure window's 3D figure),
    /// rendered while it is `Some` and freed shortly after it goes `None`.
    pub model_preview: Option<ModelPreviewInput<'a>>,
    /// Draw the skybox behind the scene. When false the clear color shows.
    pub draw_skybox: bool,
    /// PCF tap spacing in shadow-UV units (softness / shadow_resolution).
    pub shadow_pcf_texel: f32,
    /// Receiver depth-bias strength (point/spot: 1500 is 1x, sub-texel default)
    /// (Environment `shadow_bias_scale`; 0 = the 1500 default).
    pub shadow_bias_scale: f32,
    /// Directional shadow normal-offset in shadow texels (Environment
    /// `shadow_normal_offset`; 0 = the 2.0 default).
    pub shadow_normal_offset: f32,
    /// Directional shadow coverage in world units (ortho box fit ahead of the
    /// camera). Smaller = sharper.
    pub shadow_distance: f32,
    /// Cascades for the main directional light (Environment `shadow_cascades`;
    /// 0 = the tier's maximum, otherwise clamped to 1..=that maximum).
    pub shadow_cascades: u32,
    /// Blend of the cascade split scheme, 0 = uniform .. 1 = logarithmic
    /// (Environment `shadow_cascade_lambda`; 0 = the default).
    pub shadow_cascade_lambda: f32,
    /// Seconds since app start; drives animated shader effects.
    pub time: f32,
    /// Water clock (s): the play clock while playing, so the rendered waves
    /// match the ones floating bodies ride.
    pub water_time: f32,
    /// Water bodies this frame, indexed by the water materials' slots (at
    /// most `MAX_WATER_BODIES` are used).
    pub water: &'a [GpuWater],
    pub draws: &'a [DrawCmd],
    /// World-space projected decals to composite over the opaque scene.
    pub decals: &'a [Decal],
    /// Debug: render objects as a lightmap-UV checkerboard (cell size tracks
    /// each object's lightmap resolution) instead of their material.
    pub lightmap_preview: bool,
    /// GI debug view: 0 = off, 1 = world normals, 2 = indirect/GI term only.
    /// Lets you isolate the probe-grid blockiness on screen.
    pub gi_debug: u32,
    /// Whether the editor viewport is visible this frame. When false (e.g. only
    /// the Camera tab is shown) the main scene draws + the viewport Flux trace
    /// are skipped to save GPU; the swapchain is still cleared and egui drawn.
    /// Always true for the game runtime.
    pub render_viewport: bool,
    /// Editor only: when set, render the viewport 3D into an offscreen texture of
    /// this pixel size (the viewport dock rect) instead of the full swapchain, so
    /// it isn't rasterized under the dock panels. `None` for the game runtime,
    /// which renders straight to the swapchain.
    pub viewport_extent: Option<[u32; 2]>,
    /// Resolved post-processing parameters (from the camera's blended Volume
    /// profiles). Per-pixel effects applied in the surface shaders.
    pub postfx: PostFx,
    /// Post-processing for the camera preview (resolved at the preview camera).
    pub preview_postfx: PostFx,
    /// Active reflection-probe zone (the placed `ReflectionProbe` whose box
    /// contains the camera, or the nearest). Drives box-projected parallax +
    /// intensity for the environment reflection. `None` = treat the env cube as
    /// distant/infinite (no parallax, intensity 1).
    pub reflection_probe: Option<ReflectionProbeBox>,
    /// Reflection probes in priority order (best first), blended per pixel.
    /// Slot 0 is always used as the fallback; additional slots are mixed in by
    /// their box influence.
    pub reflection_probes: [Option<ReflectionProbeBox>; MAX_REFLECTION_PROBES],
    /// Second probe (slot 1, binding 9), blended per pixel with `reflection_probe`.
    pub reflection_probe_b: Option<ReflectionProbeBox>,
    /// Which slot wins where both probe boxes cover a pixel (higher importance,
    /// then the smaller box). Outside both boxes slot 0 is used.
    pub reflection_primary: u8,
    /// Distance + height exponential fog (atmospheric depth). `None` = no fog.
    pub fog: Option<FogParams>,
    /// FluxVoxel specular-from-volume: metallic/rough surfaces sample the probe
    /// volume in the reflection direction for emissive/voxel-light bounce (VXGI-style
    /// glossy approximation). Off = reflection cube only.
    pub voxel_specular: bool,
    /// Active planar-reflection surfaces (floors, mirrors). Each gets a mirrored
    /// scene render from the main camera into the planar RT array; surfaces whose
    /// `DrawCmd.planar_index` matches sample their layer instead of the probe.
    /// At most `MAX_PLANAR` are rendered; extras fall back to the probe.
    pub planar: &'a [PlanarSurface],
    /// Cameras rendering into textures this frame (CCTV / TV screens), recorded
    /// before the main pass. Their targets come from `Renderer::camera_target`.
    pub camera_targets: &'a [CameraTargetView],
    pub egui: Option<EguiDraw>,
    /// World-space UI canvases to paint into their targets before the scene
    /// pass samples them.
    pub world_ui: Vec<WorldUiDraw>,
}

/// One world-space UI canvas frame: egui output for a target made by
/// `Renderer::create_ui_target`. Each target must be fed by its own egui
/// context (its texture deltas are applied to that target's renderer only).
pub struct WorldUiDraw {
    pub target: TextureHandle,
    pub extent: [u32; 2],
    pub draw: EguiDraw,
}

/// One offscreen preview of a few of the frame's draws (see `FrameInput::model_preview`).
#[derive(Clone, Copy, Debug)]
pub struct ModelPreviewInput<'a> {
    pub camera: CameraData,
    /// Pixel size of the target (re-created when it changes).
    pub size: [u32; 2],
    /// Indices into `FrameInput::draws` of the draws to render; nothing else is drawn.
    pub draws: &'a [usize],
}

/// A camera rendering into its texture target this frame.
#[derive(Clone, Copy, Debug)]
pub struct CameraTargetView {
    /// The camera object's id (as passed to `Renderer::camera_target`).
    pub id: u128,
    pub camera: CameraData,
    /// Wanted texture size; the target is resized when it changes.
    pub size: [u32; 2],
}

/// One planar-reflection surface: the world plane the scene is mirrored about.
#[derive(Clone, Copy, Debug)]
pub struct PlanarSurface {
    /// A point on the plane (the owning object's world position).
    pub point: [f32; 3],
    /// Unit plane normal (the owning object's world +Y).
    pub normal: [f32; 3],
    /// Roughness above which the planar reflection has fully faded to the
    /// probe/env fallback (sharp mirrors ~0.2-0.4).
    pub max_roughness: f32,
}

/// Exponential distance + height fog parameters.
#[derive(Clone, Copy, Debug)]
pub struct FogParams {
    pub color: [f32; 3],
    /// Fog buildup per world unit of view distance.
    pub density: f32,
    /// Height falloff: fog thins above `height_ref` at this rate (0 = uniform).
    pub height_falloff: f32,
    /// World Y where fog is at full density.
    pub height_ref: f32,
    /// View distance (world units) before fog starts accumulating.
    pub start_distance: f32,
}

/// A placed reflection-probe zone in world space, for box-projected reflections.
#[derive(Clone, Copy, Debug)]
pub struct ReflectionProbeBox {
    /// Box center (world) = probe position + box offset. The parallax proxy.
    pub center: [f32; 3],
    /// Capture position (world) = the probe object's position: where the cube
    /// is rendered from and the anchor for the parallax re-vector.
    pub capture: [f32; 3],
    pub half_extents: [f32; 3],
    pub intensity: f32,
    /// Box-projected parallax (Unity-style) vs. infinite/distant sampling.
    pub box_projection: bool,
    /// Captured cubemap face resolution for this probe.
    pub resolution: u32,
    /// Realtime mode: the engine keeps re-capturing the cube continuously so
    /// dynamic lighting tracks in reflections; Baked loads a sidecar once.
    pub realtime: bool,
    /// The probe object's id (which cube belongs in which slot).
    pub id: u128,
    /// Thickness of the shell just inside the box faces over which this probe
    /// fades in when blended with a second probe.
    pub blend_distance: f32,
    /// Overlap priority: where two probe boxes overlap, the higher importance
    /// (then the smaller box) wins.
    pub importance: i32,
    /// Realtime refreshes per second (0 = every frame).
    pub update_hz: f32,
    /// Realtime capture one face per frame instead of all six at once.
    pub time_slicing: bool,
    /// Realtime capture draws non-static objects too.
    pub include_dynamic: bool,
}

/// Hardware ray-hit shading policy. Default inherits the environment in a profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum RayLightingMode {
    #[default]
    Default = 0,
    SurfaceCache = 1,
    HitLightingForReflections = 2,
    HitLighting = 3,
}
impl RayLightingMode {
    pub const ALL: [Self; 4] = [Self::Default, Self::SurfaceCache, Self::HitLightingForReflections, Self::HitLighting];
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Environment Default",
            Self::SurfaceCache => "Surface Cache",
            Self::HitLightingForReflections => "Hit Lighting for Reflections",
            Self::HitLighting => "Hit Lighting for GI and Reflections",
        }
    }
}

/// Per-property GI/reflection overrides in a camera's post-process volumes.
/// None inherits the environment. These do not mutate another camera's settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FluxPostFx {
    pub ray_lighting_mode: Option<RayLightingMode>,
    /// Shared surface-cache update budget multiplier (0.5..16), separate from final-gather history.
    pub scene_lighting_update_speed: Option<f32>,
    pub final_gather_quality: Option<f32>,
    pub final_gather_update_speed: Option<f32>,
    pub max_trace_distance: Option<f32>,
    pub reflection_quality: Option<f32>,
    pub max_roughness_to_trace: Option<f32>,
    pub max_reflection_bounces: Option<f32>,
}

impl FluxPostFx {
    /// Blend independent overrides. Quality and speed multiply the environment;
    /// their neutral baseline is 1. Absolute settings inherit an explicit base.
    /// Switching from automatic trace distance to an explicit distance is
    /// discrete, since an automatic distance has no numeric value to interpolate.
    pub fn blend(self, other: Self, weight: f32) -> Self {
        fn field(a: Option<f32>, b: Option<f32>, t: f32, neutral: Option<f32>) -> Option<f32> {
            let Some(b) = b.filter(|v| v.is_finite()) else { return a; };
            if t <= 0.0 { return a; }
            match a.or(neutral) {
                Some(a) => Some(a + (b - a) * t),
                None => if t >= 0.5 { Some(b) } else { None },
            }
        }
        let t = if weight.is_finite() { weight.clamp(0.0, 1.0) } else { 0.0 };
        Self {
            ray_lighting_mode: if t >= 0.5 { other.ray_lighting_mode.or(self.ray_lighting_mode) } else { self.ray_lighting_mode },
            scene_lighting_update_speed: field(self.scene_lighting_update_speed, other.scene_lighting_update_speed, t, Some(1.0)),
            final_gather_quality: field(self.final_gather_quality, other.final_gather_quality, t, Some(1.0)),
            final_gather_update_speed: field(self.final_gather_update_speed, other.final_gather_update_speed, t, Some(1.0)),
            max_trace_distance: field(self.max_trace_distance, other.max_trace_distance, t, None),
            reflection_quality: field(self.reflection_quality, other.reflection_quality, t, Some(1.0)),
            max_roughness_to_trace: field(self.max_roughness_to_trace, other.max_roughness_to_trace, t, None),
            max_reflection_bounces: field(self.max_reflection_bounces, other.max_reflection_bounces, t, None),
        }
    }
}

/// Histogram metering of the physical HDR image, before display exposure.
/// EV100 uses the standard 1.2 luminance calibration and 18% middle grey.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoExposure {
    pub enabled: bool,
    pub min_ev100: f32,
    pub max_ev100: f32,
    /// Stops per second when entering brighter / darker surroundings.
    pub speed_up: f32,
    pub speed_down: f32,
    /// Retained histogram percentile interval, in percent.
    pub low_percent: f32,
    pub high_percent: f32,
    pub histogram_min_ev100: f32,
    pub histogram_max_ev100: f32,
}
impl Default for AutoExposure {
    fn default() -> Self {
        Self {enabled:false,min_ev100:-10.0,max_ev100:20.0,speed_up:3.0,speed_down:1.0,
            low_percent:70.0,high_percent:90.0,histogram_min_ev100:-10.0,histogram_max_ev100:20.0}
    }
}
impl AutoExposure {
    pub fn blend(self, other: Self, t: f32) -> Self {
        let t=t.clamp(0.0,1.0);let mix=|a:f32,b:f32|a+(b-a)*t;
        Self {enabled:if t>=0.5 {other.enabled} else {self.enabled},
            min_ev100:mix(self.min_ev100,other.min_ev100),max_ev100:mix(self.max_ev100,other.max_ev100),
            speed_up:mix(self.speed_up,other.speed_up),speed_down:mix(self.speed_down,other.speed_down),
            low_percent:mix(self.low_percent,other.low_percent),high_percent:mix(self.high_percent,other.high_percent),
            histogram_min_ev100:mix(self.histogram_min_ev100,other.histogram_min_ev100),
            histogram_max_ev100:mix(self.histogram_max_ev100,other.histogram_max_ev100)}
    }
}

/// Resolved per-frame post-processing parameters (the blended profile flattened
/// for the GPU). Per-pixel effects (exposure, tonemap, color grading, vignette)
/// are applied in the surface shaders; chromatic aberration + bloom need a
/// a fullscreen pass (follow-up).
#[derive(Clone, Copy, Debug)]
pub struct PostFx {
    pub flux: FluxPostFx,
    pub auto_exposure: AutoExposure,
    /// 0 = none, 1 = Reinhard, 2 = ACES.
    pub tonemap: u32,
    /// Tonemap exposure in stops (EV); color ×= 2^exposure before the operator.
    pub exposure: f32,
    pub grading_enabled: bool,
    pub grade_exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub vignette_enabled: bool,
    pub vignette_intensity: f32,
    pub vignette_smoothness: f32,
    pub vignette_color: [f32; 3],
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub bloom_radius: f32,
    pub ca_enabled: bool,
    pub ca_intensity: f32,
    /// Screen-space ambient occlusion (applied to indirect light only).
    pub ao: AoSettings,
    /// The camera is in (or at the surface of) a water body.
    pub underwater: Option<UnderwaterFx>,
}

/// Underwater look for a camera inside a water body's column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnderwaterFx {
    /// World height of the water surface over the camera. Pixels whose near
    /// plane point is below it are under water (a clean waterline when the
    /// near plane straddles the surface).
    pub surface: f32,
    /// Water colour looking up toward the surface and down into the deep
    /// (linear, unlit: the renderer lights them by the sun + ambient).
    pub shallow_color: [f32; 3],
    pub deep_color: [f32; 3],
    /// Extinction per metre of view distance through the water.
    pub density: f32,
    /// Extinction per metre of depth for light coming down from the surface.
    pub light_falloff: f32,
    /// Screen wobble strength (0 = none).
    pub distortion: f32,
    /// Blur radius below the waterline (px at 1080p; 0 = none).
    pub blur: f32,
    /// Caustic light pattern strength (0 = none).
    pub caustics: f32,
    /// The water body's waves and transform, so the waterline, the fog's
    /// surface cut and the depth darkening follow the moving surface.
    pub water_to_world: glam::Mat4,
    pub body: GpuWater,
    /// Rest surface height in the water's local space at the camera (a
    /// river's follows its course).
    pub rest_surface: f32,
    /// The water clock the waves animate with.
    pub clock: f32,
}

impl Default for PostFx {
    fn default() -> Self {
        Self {
            flux: FluxPostFx::default(),
            auto_exposure: AutoExposure::default(),
            tonemap: 2, // ACES, matches the previous hardcoded behavior
            exposure: 0.0,
            grading_enabled: false,
            grade_exposure: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
            tint: 0.0,
            vignette_enabled: false,
            vignette_intensity: 0.4,
            vignette_smoothness: 0.4,
            vignette_color: [0.0, 0.0, 0.0],
            bloom_enabled: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.5,
            bloom_radius: 0.5,
            ca_enabled: false,
            ca_intensity: 0.3,
            ao: AoSettings::default(),
            underwater: None,
        }
    }
}

/// The views the last rendered frame drew, for culling that runs before the
/// next frame is built (the scene's animators: a skinned mesh no view can see
/// is not posed or skinned).
#[derive(Clone, Debug, Default)]
pub struct ViewCullInfo {
    /// View-projection of every shadow map rendered (cascades, spot and point
    /// faces). Casters are culled against these without their near plane.
    pub shadow_view_projs: Vec<Mat4>,
    /// A view the host cannot describe as a camera frustum was drawn: a planar
    /// mirror, a render-texture camera, the camera preview, or a reflection
    /// capture that includes dynamic objects. Nothing can be culled then.
    pub secondary: bool,
}

/// Viewport debug view (`FrameInput::gi_debug`) showing the AO term alone.
pub const VIEW_AO: u32 = 12;

/// Screen-space AO algorithm (Project Settings > Rendering).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AoQuality {
    /// Scalable Ambient Obscurance, half resolution, spatial denoise only.
    Low,
    /// Ground-truth AO (GTAO), half resolution, spatial + temporal denoise.
    #[default]
    High,
}

impl AoQuality {
    pub const ALL: [AoQuality; 2] = [Self::Low, Self::High];

    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low (SAO)",
            Self::High => "High (GTAO)",
        }
    }
}

/// Per-camera AO parameters (from the blended post-processing profile).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AoSettings {
    pub enabled: bool,
    /// Strength: GTAO raises the visibility to this power, SAO scales the
    /// obscurance by it. 0 = no effect, 1 = physically based (GTAO).
    pub intensity: f32,
    /// World-space radius occluders are searched in (metres).
    pub radius: f32,
    /// Fraction of the radius over which occluders fade out toward its edge.
    pub falloff: f32,
    /// Multi-bounce approximation (Jimenez 2016): occlusion on bright
    /// surfaces is lightened and tinted by the albedo.
    pub multi_bounce: bool,
}

impl Default for AoSettings {
    fn default() -> Self {
        Self { enabled: false, intensity: 1.0, radius: 0.6, falloff: 0.6, multi_bounce: true }
    }
}

/// Anti-aliasing mode (Project Settings > Rendering; a camera may override it).
///
/// The default is [`AntiAliasing::Fxaa`]: one cheap post-process, no extra
/// targets, no jitter, so it never changes motion or the lighting look except
/// for softening aliased edges. [`Renderer::new`] itself starts with
/// [`AntiAliasing::None`] (headless tools and tests keep exact pixels); the
/// editor and the player apply the project's setting each frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AntiAliasing {
    None,
    /// FXAA 3.11 inside the post pass (sub-pixel UV offset of the scene sample).
    #[default]
    Fxaa,
    /// Temporal AA: sub-pixel camera jitter, motion-vector / depth reprojection,
    /// variance-clipped history (`aa.rs`).
    Taa,
    /// 2x MSAA. Not implemented on desktop (see Features.md); renders as FXAA.
    Msaa2,
    /// 4x MSAA. Not implemented on desktop (see Features.md); renders as FXAA.
    Msaa4,
    /// FSR3 temporal upscaling (game path). The editor viewport has no FSR and
    /// falls back to TAA.
    Fsr,
}

impl AntiAliasing {
    pub const ALL: [AntiAliasing; 6] = [Self::None, Self::Fxaa, Self::Taa, Self::Msaa2, Self::Msaa4, Self::Fsr];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Fxaa => "FXAA",
            Self::Taa => "TAA",
            Self::Msaa2 => "MSAA 2x (not available)",
            Self::Msaa4 => "MSAA 4x (not available)",
            Self::Fsr => "FSR 3 (temporal upscaling)",
        }
    }

    /// Modes the desktop renderer can actually run (the UI disables the rest).
    pub fn available(self) -> bool {
        !matches!(self, Self::Msaa2 | Self::Msaa4)
    }
}
