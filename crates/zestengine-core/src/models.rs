//! Egui-free data models shared by the runtime (scene/shader handling) and the
//! editor inspector. These describe material and shader state as plain data;
//! the editor builds its egui UI on top of them, the runtime reads them when
//! loading scenes. Neither needs the other's crate.

/// How a material's transparency is treated when rendering (mirrors the
/// render crate's alpha mode).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AlphaModeModel {
    /// Fully opaque; alpha is ignored.
    Opaque,
    /// Alpha-tested: pixels with alpha below `alpha_cutoff` are discarded.
    Cutout,
    /// Clipped then blended: fragments are alpha-tested first, survivors blend.
    Transclipping,
    /// Full alpha blending.
    Blend,
}

/// How a transparent material combines with the scene behind it (mirrors the
/// render crate's `BlendMode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BlendModeModel {
    /// Standard alpha blend.
    #[default]
    Alpha,
    /// Adds to the scene behind (glow / light effects).
    Additive,
    /// Multiplies with the scene behind.
    Multiply,
    /// Premultiplied alpha blend.
    Premultiplied,
}

impl BlendModeModel {
    /// Every blend mode, in declaration order.
    pub const ALL: [BlendModeModel; 4] = [Self::Alpha, Self::Additive, Self::Multiply, Self::Premultiplied];

    /// Inspector label of this blend mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Alpha => "Alpha",
            Self::Additive => "Additive",
            Self::Multiply => "Multiply",
            Self::Premultiplied => "Premultiplied",
        }
    }
}

/// Stencil comparison (mirrors the render crate's `StencilCompare`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum StencilCompareModel {
    /// Test always fails.
    Never,
    /// Passes when the reference value is less than the stored stencil value.
    Less,
    /// Passes when both values are equal.
    Equal,
    /// Passes when the reference value is less than or equal to the stored value.
    LessEqual,
    /// Passes when the reference value is greater than the stored value.
    Greater,
    /// Passes when both values differ.
    NotEqual,
    /// Passes when the reference value is greater than or equal to the stored value.
    GreaterEqual,
    /// Test always passes.
    #[default]
    Always,
}

impl StencilCompareModel {
    /// Every comparison, in declaration order.
    pub const ALL: [StencilCompareModel; 8] =
        [Self::Always, Self::Never, Self::Less, Self::Equal, Self::LessEqual, Self::Greater, Self::NotEqual, Self::GreaterEqual];

    /// Inspector label of this comparison.
    pub fn label(self) -> &'static str {
        match self {
            Self::Never => "Never",
            Self::Less => "Less",
            Self::Equal => "Equal",
            Self::LessEqual => "Less or equal",
            Self::Greater => "Greater",
            Self::NotEqual => "Not equal",
            Self::GreaterEqual => "Greater or equal",
            Self::Always => "Always",
        }
    }
}

/// Stencil outcome op (mirrors the render crate's `StencilOp`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum StencilOpModel {
    /// Leave the stored stencil value unchanged.
    #[default]
    Keep,
    /// Set the stored value to 0.
    Zero,
    /// Replace the stored value with the reference value.
    Replace,
    /// Add 1, clamped at the maximum stencil value.
    IncrementClamp,
    /// Subtract 1, clamped at 0.
    DecrementClamp,
    /// Bitwise-invert the stored value.
    Invert,
    /// Add 1, wrapping from the maximum to 0.
    IncrementWrap,
    /// Subtract 1, wrapping from 0 to the maximum.
    DecrementWrap,
}

impl StencilOpModel {
    /// Every operation, in declaration order.
    pub const ALL: [StencilOpModel; 8] =
        [Self::Keep, Self::Zero, Self::Replace, Self::IncrementClamp, Self::DecrementClamp, Self::Invert, Self::IncrementWrap, Self::DecrementWrap];

    /// Inspector label of this operation.
    pub fn label(self) -> &'static str {
        match self {
            Self::Keep => "Keep",
            Self::Zero => "Zero",
            Self::Replace => "Replace",
            Self::IncrementClamp => "Increment (clamp)",
            Self::DecrementClamp => "Decrement (clamp)",
            Self::Invert => "Invert",
            Self::IncrementWrap => "Increment (wrap)",
            Self::DecrementWrap => "Decrement (wrap)",
        }
    }
}

/// Stencil operations for one face (front or back): what to compare and what
/// to do with the stored value on each test outcome.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct StencilFaceModel {
    /// Comparison between the reference value and the stored stencil value.
    pub compare: StencilCompareModel,
    /// Operation applied when both the stencil and depth tests pass.
    pub pass_op: StencilOpModel,
    /// Operation applied when the stencil test fails.
    pub fail_op: StencilOpModel,
    /// Operation applied when the stencil test passes but the depth test fails.
    pub zfail_op: StencilOpModel,
}

/// A material's stencil state as plain data (mirrors the render crate's
/// `StencilState`; the engine converts). Inactive by default.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilModel {
    /// Value compared with the stored stencil value (bits masked by `read_mask`).
    pub reference: u8,
    /// Bits of both values that take part in the comparison (255 = all).
    pub read_mask: u8,
    /// Bits of the stored value that can be modified (255 = all).
    pub write_mask: u8,
    /// Operations for front-facing fragments.
    pub front: StencilFaceModel,
    /// Operations for back-facing fragments (mirrors `front` while `two_sided` is off).
    pub back: StencilFaceModel,
    /// Unity `ColorMask` bits (R 1, G 2, B 4, A 8).
    pub color_mask: u8,
    /// Unity `ZWrite`: off for stencil writers that must not occlude later draws.
    pub depth_write: bool,
    /// Separate back-face state in the inspector (off = back mirrors front).
    pub two_sided: bool,
}

impl Default for StencilModel {
    fn default() -> Self {
        Self {
            reference: 0,
            read_mask: 255,
            write_mask: 255,
            front: StencilFaceModel::default(),
            back: StencilFaceModel::default(),
            color_mask: 15,
            depth_write: true,
            two_sided: false,
        }
    }
}

impl AlphaModeModel {
    /// Inspector label of this alpha mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Opaque => "Opaque",
            Self::Cutout => "Cutout",
            Self::Transclipping => "Transclipping",
            Self::Blend => "Transparent",
        }
    }

    /// Default render queue for this alpha mode (Unity-style breakpoints).
    pub fn default_render_queue(self) -> i32 {
        match self {
            Self::Opaque => 2000, // Geometry
            Self::Cutout => 2450, // AlphaTest
            Self::Transclipping => 3000, // Transparent (clips and blends)
            Self::Blend => 3000,  // Transparent
        }
    }
}

/// Project-relative texture paths for a material's 12 sampler slots. Mirrors
/// `zestengine_assets::MaterialTextures`; kept here (egui-free) so the inspector can
/// edit texture assignments as plain data and the engine can resolve them to
/// GPU handles at apply time. `None` = use the material's import-embedded
/// texture for that slot (or the 1×1 default if there is none).
#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct MaterialTexturePaths {
    /// Albedo / base-colour map (slot 0).
    pub albedo: Option<std::path::PathBuf>,
    /// Normal map (slot 1).
    pub normal: Option<std::path::PathBuf>,
    /// Combined occlusion / roughness / metallic map (slot 2).
    pub orm: Option<std::path::PathBuf>,
    /// Emission map (slot 3).
    pub emission: Option<std::path::PathBuf>,
    /// Opacity map (slot 4).
    pub opacity: Option<std::path::PathBuf>,
    /// Mask for the emission map (slot 5).
    pub emission_mask: Option<std::path::PathBuf>,
    /// Three additive matcap textures (slots 6, 8, 10).
    pub matcap: [Option<std::path::PathBuf>; 3],
    /// Masks for the three matcaps (slots 7, 9, 11).
    pub matcap_mask: [Option<std::path::PathBuf>; 3],
    /// Split AO / Roughness / Metallic maps (slots 12/13/14).
    pub ao: Option<std::path::PathBuf>,
    /// Split roughness map (slot 13).
    pub roughness: Option<std::path::PathBuf>,
    /// Split metallic map (slot 14).
    pub metallic: Option<std::path::PathBuf>,
    /// Height / displacement map (slot 15) for parallax occlusion mapping.
    pub displacement: Option<std::path::PathBuf>,
    /// Detail layer maps (pseudo-slots 1000..1004 in [`Self::slot_mut`]).
    pub detail_albedo: Option<std::path::PathBuf>,
    /// Detail-layer normal map (pseudo-slot 1001).
    pub detail_normal: Option<std::path::PathBuf>,
    /// Detail-layer ORM map (pseudo-slot 1002).
    pub detail_orm: Option<std::path::PathBuf>,
    /// Detail-layer mask (pseudo-slot 1003).
    pub detail_mask: Option<std::path::PathBuf>,
    /// Translucency / thickness map (pseudo-slot 1004).
    pub translucency: Option<std::path::PathBuf>,
    /// Custom-shader texture properties (slots 16..), in the shader's texture
    /// declaration order: (property name, path). Synced to the shader layout
    /// by the engine; saved by name.
    pub shader: Vec<(String, Option<std::path::PathBuf>)>,
}

/// First pseudo-slot index for the detail / translucency maps in
/// [`MaterialTexturePaths::slot_mut`] (and the editor's texture-drop reports);
/// 16.. stays the custom-shader texture range.
pub const DETAIL_SLOT_BASE: usize = 1000;

/// Detail layer of the standard shader (mirrors `zestengine_render::DetailParams`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DetailModel {
    /// How strongly the detail albedo shows (0 = off).
    pub albedo_strength: f32,
    /// Detail normal scale (0 = off).
    pub normal_strength: f32,
    /// How strongly the detail ORM modulates occlusion / roughness / metallic.
    pub orm_strength: f32,
    /// 0 Multiply x2, 1 Overlay, 2 Add, 3 Replace.
    pub blend: u32,
    /// UV scale for the detail maps (default 1).
    pub tiling: [f32; 2],
    /// UV shift for the detail maps (default 0).
    pub offset: [f32; 2],
}

impl Default for DetailModel {
    fn default() -> Self {
        Self { albedo_strength: 0.0, normal_strength: 0.0, orm_strength: 0.0, blend: 0, tiling: [1.0; 2], offset: [0.0; 2] }
    }
}

impl DetailModel {
    /// Inspector labels for `blend`, in order 0..3.
    pub const BLEND_LABELS: [&'static str; 4] = ["Multiply x2", "Overlay", "Add", "Replace"];
}

/// Translucency and transmission (mirrors `zestengine_render::TransmissionParams`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TransmissionModel {
    /// Back-lit translucency strength (0 = off).
    pub translucency: f32,
    /// Colour of the light that comes through (linear).
    pub translucency_color: [f32; 3],
    /// Sharpness of the transmitted glow (higher = tighter around the light).
    pub power: f32,
    /// How much the surface normal bends the transmitted light.
    pub distortion: f32,
    /// The translucency map holds thickness (1 = thick) instead of translucency.
    pub map_is_thickness: bool,
    /// See-through amount 0..1.
    pub transmission: f32,
    /// Colour the scene behind takes on passing through (linear).
    pub tint: [f32; 3],
    /// Screen-space refraction offset, in view-normal units.
    pub refraction: f32,
    /// Rough-transmission blur strength (0 = sharp).
    pub blur: f32,
    /// Volume thickness 0..1 scaling the blur.
    pub thickness: f32,
    /// The translucency / thickness map's green channel scales the transmission.
    pub map_has_transmission: bool,
}

impl Default for TransmissionModel {
    fn default() -> Self {
        Self {
            translucency: 0.0,
            translucency_color: [1.0; 3],
            power: 4.0,
            distortion: 0.2,
            map_is_thickness: false,
            transmission: 0.0,
            tint: [1.0; 3],
            refraction: 0.05,
            blur: 1.0,
            thickness: 1.0,
            map_has_transmission: false,
        }
    }
}

/// Displacement through hardware tessellation (mirrors `zestengine_render::TessParams`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TessellationModel {
    /// Whether tessellation displacement is active for this material.
    pub enabled: bool,
    /// Displacement at height 1 above `midlevel`, in world units.
    pub scale: f32,
    /// Height-map value that stays put (0.5 = half the map raises, half sinks).
    pub midlevel: f32,
    /// Target on-screen edge length of the generated triangles (pixels).
    pub edge_px: f32,
    /// Largest subdivision level.
    pub max_factor: f32,
    /// Phong smoothing 0..1.
    pub phong: f32,
}

impl Default for TessellationModel {
    fn default() -> Self {
        Self { enabled: false, scale: 0.05, midlevel: 0.5, edge_px: 12.0, max_factor: 32.0, phong: 0.5 }
    }
}

/// Display labels for the 15 texture slots, in binding order. Index matches
/// [`MaterialTexturePaths::slot_mut`].
pub const TEXTURE_SLOT_LABELS: [&str; 16] = [
    "Albedo",
    "Normal Map",
    "ORM (Occl/Rough/Metal)",
    "Emission",
    "Opacity",
    "Emission Mask",
    "Matcap 1",
    "Matcap 1 Mask",
    "Matcap 2",
    "Matcap 2 Mask",
    "Matcap 3",
    "Matcap 3 Mask",
    "Ambient Occlusion",
    "Roughness",
    "Metallic",
    "Displacement",
];

impl MaterialTexturePaths {
    /// Mutable access to slot `i` (0..15) in binding order; `None` if out of range.
    pub fn slot_mut(&mut self, i: usize) -> Option<&mut Option<std::path::PathBuf>> {
        if i >= DETAIL_SLOT_BASE {
            return Some(match i - DETAIL_SLOT_BASE {
                0 => &mut self.detail_albedo,
                1 => &mut self.detail_normal,
                2 => &mut self.detail_orm,
                3 => &mut self.detail_mask,
                4 => &mut self.translucency,
                _ => return None,
            });
        }
        if i >= 16 {
            return self.shader.get_mut(i - 16).map(|(_, p)| p);
        }
        Some(match i {
            0 => &mut self.albedo,
            1 => &mut self.normal,
            2 => &mut self.orm,
            3 => &mut self.emission,
            4 => &mut self.opacity,
            5 => &mut self.emission_mask,
            6 => &mut self.matcap[0],
            7 => &mut self.matcap_mask[0],
            8 => &mut self.matcap[1],
            9 => &mut self.matcap_mask[1],
            10 => &mut self.matcap[2],
            11 => &mut self.matcap_mask[2],
            12 => &mut self.ao,
            13 => &mut self.roughness,
            14 => &mut self.metallic,
            15 => &mut self.displacement,
            _ => return None,
        })
    }
}

/// How a matcap layer combines with the shaded colour beneath it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MatcapBlend {
    /// Add the matcap on top (classic sphere-map highlight).
    #[default]
    Add,
    /// Multiply with the base colour (tint / shading).
    Multiply,
    /// Replace the base colour (masked overlay).
    Replace,
}

impl MatcapBlend {
    /// Inspector label of this blend mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Multiply => "Multiply",
            Self::Replace => "Replace",
        }
    }

    /// Every blend mode, in declaration order.
    pub const ALL: [MatcapBlend; 3] = [Self::Add, Self::Multiply, Self::Replace];

    /// Shader encoding of the mode: 0 Add, 1 Multiply, 2 Replace.
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Add => 0.0,
            Self::Multiply => 1.0,
            Self::Replace => 2.0,
        }
    }

    /// Decode a shader value; anything other than 1 (Multiply) or 2 (Replace)
    /// becomes Add.
    pub fn from_f32(v: f32) -> Self {
        match v as i32 {
            1 => Self::Multiply,
            2 => Self::Replace,
            _ => Self::Add,
        }
    }
}

/// View of one material (standard or custom shader) as plain values.
#[derive(Clone, PartialEq)]
pub struct MaterialModel {
    /// Material name (the project-relative asset name).
    pub name: String,
    /// "standard" or a project-relative `.frag` custom shader path.
    pub shader: String,
    /// Custom-shader property values, packed by property offset (16 floats
    /// once initialized; empty = take the shader's defaults).
    pub custom_values: Vec<f32>,
    /// Base albedo, linear RGBA.
    pub base_color: [f32; 4],
    /// Metalness 0 (dielectric) .. 1 (metal).
    pub metallic: f32,
    /// Perceptual roughness 0 (mirror) .. 1 (diffuse).
    pub roughness: f32,
    /// Ambient-occlusion map strength.
    pub occlusion_strength: f32,
    /// Whether toon (cel) shading is on.
    pub toon_enabled: bool,
    /// Toon band count (1 = flat).
    pub toon_steps: f32,
    /// PBR / toon blend, 0 = PBR .. 1 = toon.
    pub pbr_toon_blend: f32,
    /// Whether emission is on.
    pub emission_enabled: bool,
    /// Emissive colour, linear RGB.
    pub emission_color: [f32; 3],
    /// Emission multiplier (0 = off).
    pub emission_intensity: f32,
    /// How transparency is treated.
    pub alpha_mode: AlphaModeModel,
    /// Alpha-test threshold 0..1 in the Cutout mode.
    pub alpha_cutoff: f32,
    /// A normal map is assigned to this material.
    pub has_normal_texture: bool,
    /// Normal mapping is active.
    pub normal_map_enabled: bool,
    /// Normal-map intensity.
    pub normal_strength: f32,
    /// Render both faces (no back-face culling).
    pub double_sided: bool,
    /// Transparent blend equation (only used in the Transparent mode).
    pub blend_mode: BlendModeModel,
    /// Cull front faces (shows the inside of a mesh); ignored when double sided.
    pub cull_front: bool,
    /// Samples the opaque scene colour (refraction / blur / distortion shaders).
    pub grab: bool,
    /// Stencil test / write (Unity `Stencil` block); inactive by default.
    pub stencil: StencilModel,
    // Extended "FX" params (per-material UBO): Toon rim + animated emission.
    /// Toon rim-light colour.
    pub rim_color: [f32; 3],
    /// Toon rim Fresnel exponent (higher = tighter edge).
    pub rim_power: f32,
    /// Toon rim strength (0 = off).
    pub rim_strength: f32,
    /// Toon cel-ramp edge softness.
    pub ramp_smoothness: f32,
    /// Animated UV scroll for the emission map (uv units / second).
    pub emission_scroll: [f32; 2],
    /// Emission pulse speed (0 = steady); brightness oscillates.
    pub emission_pulse: f32,
    /// Invert a split AO / Roughness / Metallic map (e.g. smoothness→roughness).
    pub ao_invert: bool,
    /// Invert the roughness channel of a split map (e.g. smoothness -> roughness).
    pub roughness_invert: bool,
    /// Invert the metallic channel of a split map.
    pub metallic_invert: bool,
    /// Parallax occlusion mapping strength for the displacement map (0 = off).
    pub displacement_scale: f32,
    /// Per-material reflection strength (1 = full, 0 = matte/no reflections).
    /// Scales BOTH the environment-cube probe reflection and the screen-space /
    /// ray-traced reflection for this material, so reflection technique + look can
    /// be mixed and matched per material without changing the global model.
    pub reflection_intensity: f32,
    /// Per-material screen-space / ray-traced reflection toggle. When false this
    /// material still gets the environment / reflection-probe CUBE reflection but
    /// the deferred SSR/RT resolve skips it (reflectivity 0 in the G-buffer) — so
    /// reflection TECHNIQUE can be mixed per material (cube-only vs cube + SSR/RT).
    pub screen_reflections: bool,
    /// Per-material reflection mode: the environment's global technique, baked
    /// probes only, or none. (Planar mirrors come from the object's Planar
    /// Reflection component, not from the material.)
    pub reflection_mode: crate::water::SurfaceReflection,
    /// Per-texture UV tiling (xy scale, default 1) + offset (default 0) for the
    /// main maps; masks follow their parent map.
    /// UV scale for the albedo map (default 1).
    pub albedo_tiling: [f32; 2],
    /// UV shift for the albedo map (default 0).
    pub albedo_offset: [f32; 2],
    /// UV scale for the normal map.
    pub normal_tiling: [f32; 2],
    /// UV shift for the normal map.
    pub normal_offset: [f32; 2],
    /// UV scale for the ORM map.
    pub orm_tiling: [f32; 2],
    /// UV shift for the ORM map.
    pub orm_offset: [f32; 2],
    /// UV scale for the emission map.
    pub emission_tiling: [f32; 2],
    /// UV shift for the emission map.
    pub emission_offset: [f32; 2],
    /// Additive matcap layer strengths (0 = off). Matcap textures are assigned
    /// via the `.material` file's texture slots.
    pub matcap_strength: [f32; 3],
    /// Per-layer matcap blend mode (how each layer combines with the colour).
    pub matcap_blend: [MatcapBlend; 3],
    /// Draw-order priority (Unity render queue): Geometry 2000, AlphaTest
    /// 2450, Transparent 3000, Overlay 4000.
    pub render_queue: i32,
    /// Detail albedo / normal / ORM layer.
    pub detail: DetailModel,
    /// Thin translucency + see-through transmission.
    pub transmission: TransmissionModel,
    /// Tessellated displacement (the Displacement texture is the height map).
    pub tess: TessellationModel,
    /// Per-slot texture assignments (project-relative). Editable in the
    /// inspector; resolved to GPU handles when the material is applied.
    pub textures: MaterialTexturePaths,
}

/// Reflected custom-shader property kinds (mirrors the declarations parsed
/// by zestengine-assets; the engine converts between the two).
#[derive(Clone, Debug, PartialEq)]
pub enum ShaderPropKindUi {
    /// Slider.
    Float {
        /// Minimum value shown at the slider's left end.
        min: f32,
        /// Maximum value shown at the slider's right end.
        max: f32,
    },
    /// Unbounded drag value.
    Drag,
    /// Integer; `min == max` = unbounded.
    Int {
        /// Slider minimum; when equal to `max` the field is unbounded.
        min: i32,
        /// Slider maximum; when equal to `min` the field is unbounded.
        max: i32,
    },
    /// Boolean checkbox.
    Toggle,
    /// Dropdown of the given option labels.
    Enum(Vec<String>),
    /// Colour picker; `hdr` allows values above 1.
    Color {
        /// Whether the picker allows HDR values above 1.
        hdr: bool,
    },
    /// Three-component colour picker; `hdr` allows values above 1.
    Color3 {
        /// Whether the picker allows HDR values above 1.
        hdr: bool,
    },
    /// Vector of `n` numeric drag fields.
    Vector(u8),
    /// Texture picker for `MaterialTexturePaths` slot `slot`; the property's
    /// four values are its tiling (xy) and offset (zw).
    Texture {
        /// Texture slot in `MaterialTexturePaths` to pick for.
        slot: usize,
        /// Default texture path shown when none is assigned.
        default: &'static str,
    },
}

/// Value-comparison operators used by shader property visibility conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderCmpUi {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// Visibility condition: OR of AND-groups of (value offset, op, value).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderCondUi(pub Vec<Vec<(usize, ShaderCmpUi, f32)>>);

impl ShaderCondUi {
    /// True when the condition holds for the given property values: an empty
    /// condition is always visible, otherwise any AND-group in which every
    /// comparison passes. Equality compares with a 0.5 epsilon.
    pub fn eval(&self, values: &[f32]) -> bool {
        self.0.is_empty()
            || self.0.iter().any(|all| {
                all.iter().all(|&(at, op, b)| {
                    let a = values.get(at).copied().unwrap_or(0.0);
                    match op {
                        ShaderCmpUi::Eq => (a - b).abs() < 0.5,
                        ShaderCmpUi::Ne => (a - b).abs() >= 0.5,
                        ShaderCmpUi::Lt => a < b,
                        ShaderCmpUi::Le => a <= b,
                        ShaderCmpUi::Gt => a > b,
                        ShaderCmpUi::Ge => a >= b,
                    }
                })
            })
    }
}

/// One editable property of a custom shader's material UI.
#[derive(Clone, Debug)]
pub struct ShaderPropUi {
    /// Shader-side property name.
    pub name: String,
    /// Display label shown in the inspector.
    pub label: String,
    /// How the property is edited.
    pub kind: ShaderPropKindUi,
    /// Flat float offset into `MaterialModel::custom_values`.
    pub offset: usize,
    /// Specialization-constant keyword (changing it selects another variant).
    pub keyword: bool,
    /// Index into `ShaderUiInfo::groups`.
    pub group: Option<usize>,
    /// Optional section heading rendered above the property.
    pub header: Option<String>,
    /// Visibility condition (empty = always shown).
    pub show_if: ShaderCondUi,
    /// Optional hover tooltip text.
    pub tooltip: Option<String>,
}

/// A labelled group of properties; the header can carry a toggle checkbox
/// bound to one property.
#[derive(Clone, Debug)]
pub struct ShaderGroupUi {
    /// Group heading.
    pub label: String,
    /// Header checkbox: index into `ShaderUiInfo::props`.
    pub toggle: Option<usize>,
    /// Visibility condition for the whole group (empty = always shown).
    pub show_if: ShaderCondUi,
}

/// A named preset of custom-shader property values.
#[derive(Clone, Debug)]
pub struct ShaderPresetUi {
    /// Preset name shown in the preset dropdown.
    pub name: String,
    /// (value offset, values).
    pub values: Vec<(usize, Vec<f32>)>,
}

/// Everything the inspector needs to draw a custom shader's material UI.
#[derive(Clone, Debug, Default)]
pub struct ShaderUiInfo {
    /// Shader name shown in the inspector.
    pub display_name: String,
    /// "GLSL" or "HLSL".
    pub language: &'static str,
    /// Editable properties, in shader declaration order.
    pub props: Vec<ShaderPropUi>,
    /// Property groups for the group headers.
    pub groups: Vec<ShaderGroupUi>,
    /// Value presets offered by the shader.
    pub presets: Vec<ShaderPresetUi>,
    /// Compile/parse error; shown instead of properties.
    pub error: Option<String>,
}

impl ShaderUiInfo {
    /// Apply preset `i` to a material's values.
    pub fn apply_preset(&self, i: usize, values: &mut [f32]) {
        if let Some(p) = self.presets.get(i) {
            for (at, v) in &p.values {
                if at + v.len() <= values.len() {
                    values[*at..at + v.len()].copy_from_slice(v);
                }
            }
        }
    }

    /// Index of the preset whose assignments all match `values`.
    pub fn matching_preset(&self, values: &[f32]) -> Option<usize> {
        self.presets.iter().position(|p| {
            p.values
                .iter()
                .all(|(at, v)| values.get(*at..at + v.len()).is_some_and(|cur| cur == v.as_slice()))
        })
    }

    /// Enabled keyword count (variant summary).
    pub fn active_keywords(&self, values: &[f32]) -> usize {
        self.props
            .iter()
            .filter(|p| p.keyword && values.get(p.offset).is_some_and(|v| *v != 0.0))
            .count()
    }
}
