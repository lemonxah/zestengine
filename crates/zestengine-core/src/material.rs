//! Continuously editable material parameters: the single block of data
//! that describes how a surface shades. Plugins read and write materials
//! through this type (see `ComponentCtx::material_params` and
//! `ComponentCtx::modify_material`), so it lives in core where plugins link.

use serde::{Deserialize, Serialize};

/// Continuously editable material parameters; delivered to the shader as
/// push constants every draw, so edits are live with zero sync hazards.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialParams {
    /// Base albedo, linear RGBA; the alpha is used by transparent materials.
    pub base_color: [f32; 4],
    /// Emissive colour, linear RGB (only meaningful when the emission map or intensity makes it visible).
    pub emission_color: [f32; 3],
    /// Multiplier on the emission colour / map (0 = off).
    pub emission_intensity: f32,
    /// Mean LINEAR RGB of the emission map (1,1,1 = no map). Scales the per-
    /// instance emission fed to the RT reflection + RT GI gather so a mostly-black
    /// map (e.g. only a glowing visor) doesn't make the whole mesh reflect/bounce
    /// as a uniform light. The per-pixel forward emission is unaffected (it
    /// samples the actual map).
    pub emission_map_mean: [f32; 3],
    /// Metalness 0 (dielectric) .. 1 (metal).
    pub metallic: f32,
    /// Perceptual roughness 0 (mirror) .. 1 (diffuse).
    pub roughness: f32,
    /// Toon cel-shading band count (1 = flat).
    pub toon_steps: f32,
    /// Blend between PBR and toon shading, 0 = PBR .. 1 = toon.
    pub pbr_toon_blend: f32,
    /// Alpha-test threshold 0..1 for cutout materials.
    pub alpha_cutoff: f32,
    /// Normal-map intensity (0 = off).
    pub normal_strength: f32,
    /// Ambient-occlusion map strength (0 = off).
    pub occlusion_strength: f32,
    // --- extended "FX" params, delivered via a per-material UBO (set 1, binding
    // 4) so the built-in Standard/Toon shaders can have options beyond the
    // full 128-byte push block. ---
    /// Toon rim-light colour.
    pub rim_color: [f32; 3],
    /// Toon rim Fresnel exponent (higher = tighter edge).
    pub rim_power: f32,
    /// Toon rim strength (0 = off).
    pub rim_strength: f32,
    /// Toon cel-ramp edge softness.
    pub ramp_smoothness: f32,
    /// Animated UV scroll for the base maps (uv units / second).
    pub base_scroll: [f32; 2],
    /// Animated UV scroll for the emission map.
    pub emission_scroll: [f32; 2],
    /// Emission pulse speed (0 = steady); brightness oscillates.
    pub emission_pulse: f32,
    /// Additive matcap strengths (3 layers; 0 = off). Each layer samples its own
    /// matcap texture, multiplied by its mask.
    pub matcap_strength: [f32; 3],
    /// Per-layer matcap blend mode encoded as a float: 0 = Add, 1 = Multiply,
    /// 2 = Replace.
    pub matcap_blend: [f32; 3],
    // --- Per-texture UV transform for the main maps. `*_tiling` scales the UVs
    // (Unity tiling; default 1), `*_offset` shifts them (default 0). Mask slots
    // follow their parent map (opacity→albedo, emission_mask→emission). ---
    /// UV scale for the albedo map (Unity tiling; default 1).
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
    // Invert a split AO / Roughness / Metallic map (e.g. a smoothness map used in
    // the roughness slot). Only meaningful when the matching map is assigned.
    /// Invert the AO channel of a split ORM map.
    pub ao_invert: bool,
    /// Invert the roughness channel (e.g. a smoothness map).
    pub roughness_invert: bool,
    /// Invert the metallic channel.
    pub metallic_invert: bool,
    /// Parallax occlusion mapping strength (uv shift at full height). 0 = off.
    pub displacement_scale: f32,
    /// Per-material reflection strength (1 = full, 0 = matte). Scales the env-cube
    /// probe reflection AND the screen-space/RT reflection for this material.
    pub reflection_intensity: f32,
    /// Per-material screen-space/RT reflection toggle (true = on). Off keeps the
    /// env-cube reflection but skips the deferred SSR/RT resolve for this material.
    pub screen_reflections: bool,
    /// Per-material reflection mode: 0 = the environment's global technique,
    /// 2 = baked probes only (no SSR / ray-traced resolve), 3 = none.
    /// Mirrors `zestengine_core::ReflectionMode` for the shader (fx.parallax.w).
    #[serde(default)]
    pub reflection_mode: u8,
    /// Detail layer (albedo / normal / ORM tiled over the base maps).
    pub detail: DetailParams,
    /// Thin-surface translucency and see-through transmission.
    pub transmission: TransmissionParams,
    /// Tessellation + height-map displacement (hardware tessellation stages).
    pub tess: TessParams,
}

/// Displacement through hardware tessellation (UNITY-12): the mesh's triangles are
/// subdivided by their screen-space edge length and the new vertices pushed along
/// the (Phong-smoothed) normal by the material's height map (the Displacement
/// texture slot, white = peak). Off by default; a material without it pays
/// nothing (its pipelines have no tessellation stages). Needs a device with
/// `tessellationShader`; elsewhere the material draws flat (use parallax mapping).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TessParams {
    /// Whether tessellation displacement is active for this material.
    pub enabled: bool,
    /// Displacement at height 1 (above `midlevel`), in world units along the normal.
    pub scale: f32,
    /// Height-map value that stays put (0.5: half the map raises, half sinks).
    pub midlevel: f32,
    /// Target on-screen edge length of the generated triangles, in pixels.
    pub edge_px: f32,
    /// Largest subdivision level (the device caps it at 64).
    pub max_factor: f32,
    /// Phong smoothing of the subdivided surface, 0 (flat triangles) to 1.
    pub phong: f32,
}

impl Default for TessParams {
    fn default() -> Self {
        Self { enabled: false, scale: 0.05, midlevel: 0.5, edge_px: 12.0, max_factor: 32.0, phong: 0.5 }
    }
}

/// Standard-shader detail layer: a second set of albedo, normal and ORM maps
/// sampled at their own tiling, gated by a mask. Every strength is 0 by default,
/// which switches the layer off (old materials load unchanged).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetailParams {
    /// How strongly the detail albedo shows (0 = off).
    pub albedo_strength: f32,
    /// Detail normal scale (0 = off).
    pub normal_strength: f32,
    /// How strongly the detail ORM modulates occlusion / roughness / metallic.
    pub orm_strength: f32,
    /// How the detail albedo combines: 0 Multiply x2 (Unity), 1 Overlay, 2 Add, 3 Replace.
    pub blend: f32,
    /// UV scale for the detail maps (Unity tiling; default 1).
    pub tiling: [f32; 2],
    /// UV shift for the detail maps (default 0).
    pub offset: [f32; 2],
}

impl Default for DetailParams {
    fn default() -> Self {
        Self {
            albedo_strength: 0.0,
            normal_strength: 0.0,
            orm_strength: 0.0,
            blend: 0.0,
            tiling: [1.0, 1.0],
            offset: [0.0, 0.0],
        }
    }
}

impl DetailParams {
    /// True when any of the three detail strengths is above 0, i.e. the
    /// detail layer is active rather than fully switched off.
    pub fn is_on(&self) -> bool {
        self.albedo_strength > 0.0 || self.normal_strength > 0.0 || self.orm_strength > 0.0
    }
}

/// Standard-shader translucency (light through a thin surface, leaves / paper /
/// cloth) and transmission (the scene behind a see-through surface, glass /
/// thin plastic). Both default to 0 = off.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TransmissionParams {
    /// Back-lit translucency strength (0 = off).
    pub translucency: f32,
    /// Colour of the light that comes through (linear).
    pub translucency_color: [f32; 3],
    /// Sharpness of the transmitted glow (higher = tighter around the light).
    pub power: f32,
    /// How much the surface normal bends the transmitted light.
    pub distortion: f32,
    /// The translucency map holds thickness (1 = thick, passes less) instead of translucency.
    pub map_is_thickness: bool,
    /// See-through amount 0..1 (glTF `KHR_materials_transmission`); needs "Scene Colour Read".
    pub transmission: f32,
    /// Colour the scene behind takes on passing through (linear); glTF volume attenuation.
    pub tint: [f32; 3],
    /// Screen-space refraction offset, in view-normal units.
    pub refraction: f32,
    /// Rough transmission: how strongly roughness blurs the scene seen through
    /// the surface (0 = always sharp, 1 = the full roughness x thickness blur).
    pub blur: f32,
    /// Volume thickness 0..1 that scales the blur (times the translucency /
    /// thickness map's red channel when `map_is_thickness`). glTF
    /// `KHR_materials_volume` thickness, normalised by the importer.
    pub thickness: f32,
    /// The translucency / thickness map's GREEN channel also scales the
    /// transmission amount (glTF `transmissionTexture`).
    pub map_has_transmission: bool,
}

impl Default for TransmissionParams {
    fn default() -> Self {
        Self {
            translucency: 0.0,
            translucency_color: [1.0, 1.0, 1.0],
            power: 4.0,
            distortion: 0.2,
            map_is_thickness: false,
            transmission: 0.0,
            tint: [1.0, 1.0, 1.0],
            refraction: 0.05,
            blur: 1.0,
            thickness: 1.0,
            map_has_transmission: false,
        }
    }
}


impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            emission_color: [0.0; 3],
            emission_intensity: 1.0,
            emission_map_mean: [1.0; 3],
            metallic: 0.0,
            roughness: 0.7,
            toon_steps: 3.0,
            pbr_toon_blend: 1.0,
            alpha_cutoff: 0.5,
            normal_strength: 1.0,
            occlusion_strength: 1.0,
            rim_color: [1.0, 1.0, 1.0],
            rim_power: 4.0,
            rim_strength: 0.0,
            ramp_smoothness: 0.12,
            base_scroll: [0.0, 0.0],
            emission_scroll: [0.0, 0.0],
            emission_pulse: 0.0,
            matcap_strength: [0.0, 0.0, 0.0],
            matcap_blend: [0.0, 0.0, 0.0],
            ao_invert: false,
            roughness_invert: false,
            metallic_invert: false,
            displacement_scale: 0.0,
            reflection_intensity: 1.0,
            screen_reflections: true,
            reflection_mode: 0,
            albedo_tiling: [1.0, 1.0],
            albedo_offset: [0.0, 0.0],
            normal_tiling: [1.0, 1.0],
            normal_offset: [0.0, 0.0],
            orm_tiling: [1.0, 1.0],
            orm_offset: [0.0, 0.0],
            emission_tiling: [1.0, 1.0],
            emission_offset: [0.0, 0.0],
            detail: DetailParams::default(),
            transmission: TransmissionParams::default(),
            tess: TessParams::default(),
        }
    }
}

impl TransmissionParams {
    /// True when the see-through part needs the opaque scene copy.
    pub fn needs_grab(&self) -> bool {
        self.transmission > 0.0
    }
}