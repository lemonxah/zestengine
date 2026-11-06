//! Shader-variant pipeline cache for the zestengine standard shader.
//!
//! Each material's feature set maps to a `PipelineKey`; variants are built
//! on demand via SPIR-V specialization constants, so disabled features
//! compile out of the fragment shader entirely.

use std::collections::HashMap;
use std::io::Cursor;

use anyhow::{Context as _, Result};
use ash::vk;

use crate::vertex_hook::{self, HookStage};
use crate::types::{AlphaMode, MaterialFeatures, StencilCompare, StencilFace, StencilOp, StencilState, Vertex};

/// Bytes per instance in the instance vertex stream: one column-major mat4,
/// then the per-instance base colour and one parameter vec4.
pub const INSTANCE_STRIDE: u32 = 96;

/// One instance record in the instance-rate vertex stream (binding 1).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub param: [f32; 4],
}

impl InstanceData {
    /// Model matrix only; the unused per-instance lanes default to identity.
    pub fn from_model(model: [[f32; 4]; 4]) -> Self {
        Self { model, color: [1.0; 4], param: [0.0; 4] }
    }
}

const _: () = assert!(std::mem::size_of::<InstanceData>() == INSTANCE_STRIDE as usize);
const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/standard.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/standard.frag.spv"));
const ERROR_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/error.frag.spv"));
const OUTLINE_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/outline.vert.spv"));
const OUTLINE_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/outline.frag.spv"));
const FUR_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fur.vert.spv"));
const FUR_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fur.frag.spv"));
const SKYBOX_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.vert.spv"));
const SKYBOX_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.frag.spv"));
const SHADOW_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
/// The shadow pass drawing from the cluster pool (cluster shadows, GEO-5).
const SHADOW_VERT_CLUSTER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.cluster.spv"));
const VERT_CLUSTER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/standard.vert.cluster.spv"));
/// The clustered vertex shader under multiview (both eyes from one visible list; GEO-5).
const VERT_CLUSTER_MV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/standard.vert.cluster.mv.spv"));
const VERT_MV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/standard.vert.mv.spv"));
const FRAG_MV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/standard.frag.mv.spv"));
const SKYBOX_VERT_MV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.vert.mv.spv"));
// Tessellated (displaced) materials: control / evaluation stages for the camera passes and the shadow pass.
const TESC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tess.tesc.spv"));
const TESE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tess.tese.spv"));
const TESE_MV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tess.tese.mv.spv"));
const SHADOW_TESC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tess.tesc.shadow.spv"));
const SHADOW_TESE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tess.tese.shadow.spv"));
const SHADOW_VERT_TESS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.tess.spv"));

pub(crate) const PUSH_CONSTANT_SIZE: u32 = 128;
/// The stages of a tessellated draw. They read the frame UBO and the material's set 1 but
/// never the push constants (the range stays vertex + fragment), so every existing
/// `cmd_push_constants` call is unchanged.
const TESS_STAGES: vk::ShaderStageFlags =
    vk::ShaderStageFlags::from_raw(vk::ShaderStageFlags::TESSELLATION_CONTROL.as_raw() | vk::ShaderStageFlags::TESSELLATION_EVALUATION.as_raw());
/// Motion-vector attachment format (`velocity` pipeline variants): two 16-bit
/// floats holding the clip-space reprojection difference.
pub(crate) const VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
/// Reactive-mask attachment format (`velocity` variants): single-channel R8
/// marking transparent/reactive pixels the upscaler must not accumulate.
pub(crate) const REACTIVE_FORMAT: vk::Format = vk::Format::R8_UNORM;
/// Keyword specialization constants per custom shader (ids 0..N).
pub const CUSTOM_KEYWORDS: usize = 32;
/// Custom-shader texture properties (set 2, bindings 1..=N).
pub const CUSTOM_TEXTURES: usize = 24;
/// Specialization constant id of `zestengine_alpha_mode` in custom shaders.
const CUSTOM_ALPHA_CONSTANT_ID: u32 = 63;
/// Specialization constant id of `zestengine_blend_mode` in custom shaders.
const CUSTOM_BLEND_CONSTANT_ID: u32 = 62;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PipelineKey {
    pub toon: bool,
    pub normal_map: bool,
    pub emission: bool,
    pub alpha_mode: u32,
    pub double_sided: bool,
    /// Water surface: Gerstner displacement in the vertex stage, the water
    /// shading path in the fragment stage.
    pub water: bool,
    /// Per-instance material properties (base colour + one parameter vec4).
    pub per_instance: bool,
    /// Error/missing-material fallback: pink swirl, ignores other flags.
    pub error: bool,
    /// Selection outline: inverted hull, front-face culled flat purple.
    pub outline: bool,
    /// Fur shell: the mesh pushed out along its normals, strand-masked, opaque.
    pub fur: bool,
    /// Depth-only prepass (no color writes); masks outline interiors for
    /// transparent objects, which never write depth in the main pass.
    pub depth_only: bool,
    /// Fullscreen skybox background (no vertex buffer, depth-test only).
    pub skybox: bool,
    /// Shadow depth pass: vertex-only, depth into a shadow map (no color).
    pub shadow: bool,
    /// Deferred-SSR G-buffer variant: adds a second color attachment
    /// (reflectance.rgb + roughness.a) so a fullscreen resolve pass can do
    /// current-frame screen-space reflections. Off = single color attachment.
    pub gbuf: bool,
    /// Motion-vector variant: adds an RG16F color attachment writing the
    /// reprojected screen-space motion (velocity = cur_clip - prev_clip).
    pub velocity: bool,
    /// 0 = standard shader; otherwise a custom fragment shader
    /// (`ShaderId` + 1). Custom variants skip specialization constants.
    pub shader: u32,
    /// Mirrored passes (reflection cube faces, planar mirrors) reverse the
    /// on-screen winding: clockwise is the front face, so culling and
    /// `gl_FrontFacing` (which flips the shading normal) stay correct.
    pub mirrored: bool,
    /// Color attachment format of the pass this variant draws into. Filled by
    /// [`PipelineCache::get`] from the current pass target; part of the key so
    /// HDR, LDR and capture passes each get a matching pipeline.
    pub color_format: vk::Format,
    /// Depth prepass: standard vertex shader (invariant position, so depths
    /// match the main pass exactly), no color attachments; vertex-only for
    /// opaque draws, with the fragment shader for cutout / custom draws so
    /// discarded texels stay empty.
    pub prepass: bool,
    /// Main pass over a prepass depth: compare EQUAL, no depth write.
    pub depth_equal: bool,
    /// Custom-shader keyword values (specialization constant id = index).
    /// Each distinct set is its own lazily built, cached pipeline in which the
    /// driver folds the constants and strips disabled feature code.
    pub spec: [u8; CUSTOM_KEYWORDS],
    /// Multiview pass (view mask 0b11, both XR eyes) and its sample count.
    /// Filled by [`PipelineCache::get`] from the current pass target.
    pub multiview: bool,
    pub samples: u8,
    /// Clustered variant (GEO-4): the vertex stage pulls its vertices from the
    /// cluster pool (`standard.vert -DCLUSTER`); no vertex buffers.
    pub cluster: bool,
    /// Transparent blend equation (BlendMode as u8; only read when alpha_mode is 2 or 3).
    pub blend: u8,
    /// Cull front faces instead of back faces (Unity `Cull Front`); ignored when double sided.
    pub cull_front: bool,
    /// Material stencil state. Only honoured in colour passes over a depth
    /// format with stencil (`scene_pipeline_key` clears it for prepass /
    /// shadow variants, `build_variant` for formats without a stencil aspect).
    pub stencil: StencilState,
    /// Displaced material: tessellation control / evaluation stages after the vertex stage
    /// (patch topology). The one extra key bit tessellation costs; materials without a
    /// Displacement feature never set it.
    pub tess: bool,
}

impl PipelineKey {
    pub fn from_features(f: &MaterialFeatures) -> Self {
        Self {
            toon: f.toon,
            normal_map: f.normal_map,
            emission: f.emission,
            alpha_mode: match f.alpha_mode {
                AlphaMode::Opaque => 0,
                AlphaMode::Cutout => 1,
                AlphaMode::Blend => 2,
                AlphaMode::Transclipping => 3,
            },
            double_sided: f.double_sided,
            water: f.water,
            per_instance: f.per_instance,
            error: false,
            outline: false,
            fur: false,
            depth_only: false,
            skybox: false,
            shadow: false,
            gbuf: false,
            velocity: false,
            shader: 0,
            mirrored: false,
            color_format: vk::Format::UNDEFINED,
            prepass: false,
            depth_equal: false,
            spec: [0; CUSTOM_KEYWORDS],
            multiview: false,
            samples: 1,
            cluster: false,
            blend: f.blend as u8,
            cull_front: f.cull_front,
            stencil: f.stencil,
            tess: false,
        }
    }

    pub fn error() -> Self {
        Self {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: 0,
            double_sided: true,
            water: false,
            per_instance: false,
            error: true,
            outline: false,
            fur: false,
            depth_only: false,
            skybox: false,
            shadow: false,
            gbuf: false,
            velocity: false,
            shader: 0,
            mirrored: false,
            color_format: vk::Format::UNDEFINED,
            prepass: false,
            depth_equal: false,
            spec: [0; CUSTOM_KEYWORDS],
            multiview: false,
            samples: 1,
            cluster: false,
            blend: 0,
            cull_front: false,
            stencil: StencilState::default(),
            tess: false,
        }
    }

    /// A fur shell pass: the outline's vertex stream and attachments, its own shaders.
    pub fn fur() -> Self {
        Self { outline: false, fur: true, ..Self::outline() }
    }

    pub fn outline() -> Self {
        Self {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: 0,
            double_sided: false,
            water: false,
            per_instance: false,
            error: false,
            outline: true,
            fur: false,
            depth_only: false,
            skybox: false,
            shadow: false,
            gbuf: false,
            velocity: false,
            shader: 0,
            mirrored: false,
            color_format: vk::Format::UNDEFINED,
            prepass: false,
            depth_equal: false,
            spec: [0; CUSTOM_KEYWORDS],
            multiview: false,
            samples: 1,
            cluster: false,
            blend: 0,
            cull_front: false,
            stencil: StencilState::default(),
            tess: false,
        }
    }

    pub fn depth_only() -> Self {
        Self {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: 0,
            double_sided: false,
            water: false,
            per_instance: false,
            error: false,
            outline: false,
            fur: false,
            depth_only: true,
            skybox: false,
            shadow: false,
            gbuf: false,
            velocity: false,
            shader: 0,
            mirrored: false,
            color_format: vk::Format::UNDEFINED,
            prepass: false,
            depth_equal: false,
            spec: [0; CUSTOM_KEYWORDS],
            multiview: false,
            samples: 1,
            cluster: false,
            blend: 0,
            cull_front: false,
            stencil: StencilState::default(),
            tess: false,
        }
    }

    pub fn skybox() -> Self {
        Self {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: 0,
            double_sided: false,
            water: false,
            per_instance: false,
            error: false,
            outline: false,
            fur: false,
            depth_only: false,
            skybox: true,
            shadow: false,
            gbuf: false,
            velocity: false,
            shader: 0,
            mirrored: false,
            color_format: vk::Format::UNDEFINED,
            prepass: false,
            depth_equal: false,
            spec: [0; CUSTOM_KEYWORDS],
            multiview: false,
            samples: 1,
            cluster: false,
            blend: 0,
            cull_front: false,
            stencil: StencilState::default(),
            tess: false,
        }
    }

    /// The shadow pass's clustered variant: vertex pulling from the cluster
    /// pool (`shadow.vert -DCLUSTER`), no vertex buffers.
    pub fn shadow_cluster() -> Self {
        Self { cluster: true, ..Self::shadow() }
    }

    pub fn shadow() -> Self {
        Self {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: 0,
            double_sided: false,
            water: false,
            per_instance: false,
            error: false,
            outline: false,
            fur: false,
            depth_only: false,
            skybox: false,
            shadow: true,
            gbuf: false,
            velocity: false,
            shader: 0,
            mirrored: false,
            color_format: vk::Format::UNDEFINED,
            prepass: false,
            depth_equal: false,
            spec: [0; CUSTOM_KEYWORDS],
            multiview: false,
            samples: 1,
            cluster: false,
            blend: 0,
            cull_front: false,
            stencil: StencilState::default(),
            tess: false,
        }
    }
}

/// Vulkan stencil state of one face of a material's [`StencilState`].
pub(crate) fn vk_stencil_face(state: &StencilState, face: &StencilFace) -> vk::StencilOpState {
    let op = |o: StencilOp| match o {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Zero => vk::StencilOp::ZERO,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOp::Invert => vk::StencilOp::INVERT,
        StencilOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    };
    let compare = match face.compare {
        StencilCompare::Never => vk::CompareOp::NEVER,
        StencilCompare::Less => vk::CompareOp::LESS,
        StencilCompare::Equal => vk::CompareOp::EQUAL,
        StencilCompare::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        StencilCompare::Greater => vk::CompareOp::GREATER,
        StencilCompare::NotEqual => vk::CompareOp::NOT_EQUAL,
        StencilCompare::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        StencilCompare::Always => vk::CompareOp::ALWAYS,
    };
    vk::StencilOpState::default()
        .fail_op(op(face.fail_op))
        .pass_op(op(face.pass_op))
        .depth_fail_op(op(face.zfail_op))
        .compare_op(compare)
        .compare_mask(state.read_mask as u32)
        .write_mask(state.write_mask as u32)
        .reference(state.reference as u32)
}

/// See [`PipelineCache::pass_formats`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct PassFormats {
    pub color: vk::Format,
    pub depth: vk::Format,
    pub gbuf: vk::Format,
    pub multiview: bool,
    pub samples: u8,
}

pub(crate) struct PipelineCache {
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    error_frag: vk::ShaderModule,
    outline_vert: vk::ShaderModule,
    outline_frag: vk::ShaderModule,
    fur_vert: vk::ShaderModule,
    fur_frag: vk::ShaderModule,
    skybox_vert: vk::ShaderModule,
    skybox_frag: vk::ShaderModule,
    shadow_vert: vk::ShaderModule,
    shadow_vert_cluster: vk::ShaderModule,
    /// Multiview (gl_ViewIndex) variants of the standard / skybox shaders.
    vert_mv: vk::ShaderModule,
    /// Clustered vertex-pulling variant of the standard vertex shader (GEO-4).
    vert_cluster: vk::ShaderModule,
    vert_cluster_mv: vk::ShaderModule,
    frag_mv: vk::ShaderModule,
    skybox_vert_mv: vk::ShaderModule,
    /// Runtime-registered custom fragment shaders, indexed by `ShaderId`.
    custom_frags: Vec<vk::ShaderModule>,
    /// Vertex hooks of the custom shaders (indexed like `custom_frags`): the GLSL chunk and the
    /// hooked vertex modules built from it so far (one per [`HookStage`]).
    hooks: Vec<Option<HookEntry>>,
    /// Tessellation stage modules; None when the device has no `tessellationShader`.
    tess: Option<TessModules>,
    /// Tessellated materials also draw in the XR multiview pass (`multiviewTessellationShader`).
    tess_multiview: bool,
    pub set0_layout: vk::DescriptorSetLayout,
    pub set1_layout: vk::DescriptorSetLayout,
    /// Set 2: custom-shader material block (binding 0) + texture properties.
    pub set2_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    /// Set 2 of the clustered pipelines: the cluster pool, instances and visible list.
    pub cluster_set_layout: vk::DescriptorSetLayout,
    /// `layout` with the cluster set at index 2.
    pub cluster_layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    /// Second color attachment format for `gbuf` variants (deferred SSR).
    gbuf_format: vk::Format,
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    /// Color format of the pass being recorded (see `set_target_format`).
    target_format: vk::Format,
    /// Multiview + sample count of the pass being recorded (`set_target_views`).
    target_multiview: bool,
    target_samples: u8,
}

/// A custom shader's vertex hook and the hooked vertex modules built from it.
struct HookEntry {
    glsl: String,
    modules: [Option<vk::ShaderModule>; HookStage::COUNT],
}

/// Tessellation stages of displaced materials (UNITY-12).
struct TessModules {
    tesc: vk::ShaderModule,
    tese: vk::ShaderModule,
    tese_mv: vk::ShaderModule,
    shadow_tesc: vk::ShaderModule,
    shadow_tese: vk::ShaderModule,
    shadow_vert: vk::ShaderModule,
}

impl PipelineCache {
    pub fn new(
        device: &ash::Device,
        color_format: vk::Format,
        depth_format: vk::Format,
        gbuf_format: vk::Format,
    ) -> Result<Self> {
        let vert_code = ash::util::read_spv(&mut Cursor::new(VERT_SPV))?;
        let frag_code = ash::util::read_spv(&mut Cursor::new(FRAG_SPV))?;
        let error_code = ash::util::read_spv(&mut Cursor::new(ERROR_FRAG_SPV))?;
        let vert = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vert_code),
                None,
            )?
        };
        let frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code),
                None,
            )?
        };
        let error_frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&error_code),
                None,
            )?
        };
        let outline_vert_code = ash::util::read_spv(&mut Cursor::new(OUTLINE_VERT_SPV))?;
        let outline_frag_code = ash::util::read_spv(&mut Cursor::new(OUTLINE_FRAG_SPV))?;
        let fur_vert_code = ash::util::read_spv(&mut Cursor::new(FUR_VERT_SPV))?;
        let fur_frag_code = ash::util::read_spv(&mut Cursor::new(FUR_FRAG_SPV))?;
        let fur_vert = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fur_vert_code), None)? };
        let fur_frag = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fur_frag_code), None)? };
        let outline_vert = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&outline_vert_code),
                None,
            )?
        };
        let outline_frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&outline_frag_code),
                None,
            )?
        };
        let skybox_vert_code = ash::util::read_spv(&mut Cursor::new(SKYBOX_VERT_SPV))?;
        let skybox_frag_code = ash::util::read_spv(&mut Cursor::new(SKYBOX_FRAG_SPV))?;
        let skybox_vert = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&skybox_vert_code),
                None,
            )?
        };
        let skybox_frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&skybox_frag_code),
                None,
            )?
        };
        let shadow_vert_code = ash::util::read_spv(&mut Cursor::new(SHADOW_VERT_SPV))?;
        let shadow_vert = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&shadow_vert_code),
                None,
            )?
        };
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            let code = ash::util::read_spv(&mut Cursor::new(spv))?;
            Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
        };
        let shadow_vert_cluster = module(SHADOW_VERT_CLUSTER_SPV)?;
        let vert_mv = module(VERT_MV_SPV)?;
        let vert_cluster = module(VERT_CLUSTER_SPV)?;
        let vert_cluster_mv = module(VERT_CLUSTER_MV_SPV)?;
        let frag_mv = module(FRAG_MV_SPV)?;
        let skybox_vert_mv = module(SKYBOX_VERT_MV_SPV)?;

        // set 0: per-frame UBO (binding 0), shadow-map array (binding 1), and
        // the baked light-probe SH storage buffer (binding 2, runtime GI).
        let frame_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                // The tessellation stages read the camera (edge lengths, view-projection).
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | TESS_STAGES),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Baked lightmap array (static-object GI), sampled by uv1 + layer.
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Screen-space GI gather result (full-res indirect irradiance).
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Water: depth of the opaque scene, copied before the transparent draws.
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Water: HDR colour of the opaque scene (refraction).
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Reflection probe: prefiltered environment cubemap (samplerCube).
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Planar reflections: per-plane mirrored scene renders (2D array,
            // one layer per active PlanarReflection surface).
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Second reflection probe (two-probe blending): its own cube.
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Third reflection probe cube.
            vk::DescriptorSetLayoutBinding::default()
                .binding(11)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Fourth reflection probe cube.
            vk::DescriptorSetLayoutBinding::default()
                .binding(12)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Screen-space ambient occlusion (full-res visibility, ao.rs); read
            // only while frame.refl_blend2.w > 0.
            vk::DescriptorSetLayoutBinding::default()
                .binding(13)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Each water body's ripple field (by body slot): the caustics on
            // what lies under it.
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(crate::MAX_WATER_BODIES as u32)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let set0_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&frame_bindings),
                None,
            )?
        };

        // set 1: material textures (albedo, normal, ORM, emission)
        // Texture samplers at bindings 0-3 (albedo/normal/orm/emission), 5-12
        // (opacity, emission mask, 3 matcaps + 3 matcap masks), and 13-15 (split
        // AO/Roughness/Metallic). Binding 4 is the FX uniform block. Every
        // material set writes all of these (create_material and the skybox set),
        // so they're always valid.
        // 17-21 are the detail albedo / normal / ORM / mask and the translucency
        // (thickness) map, appended so every existing binding keeps its number.
        // 22 is the area-light emission atlas (one shared image, written into every set);
        // 23 the LTC matrix table (likewise shared).
        let sampler_bindings: [u32; 23] =
            [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];
        let mut tex_bindings: Vec<_> = sampler_bindings
            .iter()
            .map(|&i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    // 16 doubles as the water ripple field and 15 as its shore
                    // depth, read by the vertex stage.
                    .stage_flags(if i == 16 {
                        // ... and 16 is the displacement height map the tessellation stages sample.
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | TESS_STAGES
                    } else if i == 15 {
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
                    } else {
                        vk::ShaderStageFlags::FRAGMENT
                    })
            })
            .collect();
        // Binding 4: per-material "FX" uniform block (rim, animated emission,
        // matcap strengths). Holds the extended params beyond the 128-byte
        // push block.
        tex_bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | TESS_STAGES),
        );
        let set1_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&tex_bindings),
                None,
            )?
        };

        // set 2: custom-shader properties. Binding 0 is the generated std140
        // block, bindings 1..=CUSTOM_TEXTURES the declared texture properties.
        // Only custom-shader draws bind it.
        let mut custom_bindings = vec![vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            // The vertex stage reads it too when the shader has a vertex hook.
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
        for i in 1..=CUSTOM_TEXTURES as u32 {
            custom_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            );
        }
        let set2_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&custom_bindings),
                None,
            )?
        };

        let set_layouts = [set0_layout, set1_layout, set2_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(PUSH_CONSTANT_SIZE)];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )?
        };

        let cluster_set_layout = crate::cluster::create_set_layout(device)?;
        let cluster_layouts = [set0_layout, set1_layout, cluster_set_layout];
        let cluster_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&cluster_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )?
        };

        Ok(Self {
            vert,
            frag,
            error_frag,
            outline_vert,
            outline_frag,
            fur_vert,
            fur_frag,
            skybox_vert,
            skybox_frag,
            shadow_vert,
            shadow_vert_cluster,
            vert_mv,
            vert_cluster,
            vert_cluster_mv,
            frag_mv,
            skybox_vert_mv,
            custom_frags: Vec::new(),
            hooks: Vec::new(),
            tess: None,
            tess_multiview: false,
            set0_layout,
            set1_layout,
            set2_layout,
            layout,
            cluster_set_layout,
            cluster_layout,
            color_format,
            depth_format,
            gbuf_format,
            pipelines: HashMap::new(),
            target_format: color_format,
            target_multiview: false,
            target_samples: 1,
        })
    }

    /// Attachment formats and view layout of the pass being recorded, for
    /// pipelines built outside this cache (the text pipeline).
    pub fn pass_formats(&self) -> PassFormats {
        PassFormats {
            color: self.target_format,
            depth: self.depth_format,
            gbuf: self.gbuf_format,
            multiview: self.target_multiview,
            samples: self.target_samples,
        }
    }

    pub fn variant_count(&self) -> u32 {
        self.pipelines.len() as u32
    }

    /// Register a runtime-compiled custom fragment shader; returns its
    /// index. Modules live until the cache is destroyed (hot reload
    /// registers a replacement rather than destroying in-flight modules).
    pub fn register_custom(&mut self, device: &ash::Device, spirv: &[u8]) -> Result<usize> {
        let code =
            ash::util::read_spv(&mut Cursor::new(spirv)).context("reading custom shader SPIR-V")?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };
        self.custom_frags.push(module);
        self.hooks.push(None);
        Ok(self.custom_frags.len() - 1)
    }

    /// [`Self::register_custom`] for a shader with a vertex hook (`hook` is the GLSL chunk the
    /// assets crate assembled). The main-pass vertex module is compiled right away so a
    /// hook that does not compile is reported at registration; the other passes' modules are
    /// built when a pipeline first needs them.
    pub fn register_custom_with_hook(&mut self, device: &ash::Device, spirv: &[u8], hook: &str) -> Result<usize> {
        let spv = vertex_hook::compile(HookStage::Main, hook)?;
        let id = self.register_custom(device, spirv)?;
        let code = ash::util::read_spv(&mut Cursor::new(&spv[..])).context("reading vertex hook SPIR-V")?;
        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? };
        let mut modules = [None; HookStage::COUNT];
        modules[HookStage::Main.index()] = Some(module);
        self.hooks[id] = Some(HookEntry { glsl: hook.to_owned(), modules });
        Ok(id)
    }

    /// Whether custom shader `index` carries a vertex hook (its casters then need their own
    /// shadow draws).
    pub fn has_hook(&self, index: usize) -> bool {
        self.hooks.get(index).is_some_and(|h| h.is_some())
    }

    /// Build the tessellation stage modules (call once; the device must have been created with
    /// `tessellationShader`). `multiview` says `multiviewTessellationShader` is on too.
    pub fn enable_tessellation(&mut self, device: &ash::Device, multiview: bool) -> Result<()> {
        if self.tess.is_some() {
            return Ok(());
        }
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            let code = ash::util::read_spv(&mut Cursor::new(spv))?;
            Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
        };
        self.tess = Some(TessModules {
            tesc: module(TESC_SPV)?,
            tese: module(TESE_SPV)?,
            tese_mv: module(TESE_MV_SPV)?,
            shadow_tesc: module(SHADOW_TESC_SPV)?,
            shadow_tese: module(SHADOW_TESE_SPV)?,
            shadow_vert: module(SHADOW_VERT_TESS_SPV)?,
        });
        self.tess_multiview = multiview;
        Ok(())
    }

    /// Whether displaced materials get tessellation stages on this device.
    pub fn tessellation(&self) -> bool {
        self.tess.is_some()
    }

    /// Make sure the hooked vertex module `key` needs exists (compiling it on first use).
    fn ensure_hook_module(&mut self, device: &ash::Device, key: &PipelineKey) -> Result<()> {
        if key.shader == 0 || key.skybox || key.fur || key.outline || key.cluster || key.error || key.depth_only {
            return Ok(());
        }
        let index = key.shader as usize - 1;
        let stage = if key.shadow {
            if key.tess { HookStage::ShadowTess } else { HookStage::Shadow }
        } else if key.multiview {
            HookStage::MainMultiview
        } else {
            HookStage::Main
        };
        let Some(Some(entry)) = self.hooks.get(index) else {
            return Ok(());
        };
        if entry.modules[stage.index()].is_some() {
            return Ok(());
        }
        let spv = vertex_hook::compile(stage, &entry.glsl)?;
        let code = ash::util::read_spv(&mut Cursor::new(&spv[..])).context("reading vertex hook SPIR-V")?;
        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? };
        if let Some(Some(entry)) = self.hooks.get_mut(index) {
            entry.modules[stage.index()] = Some(module);
        }
        Ok(())
    }

    /// The hooked vertex module of `key`, when its shader has a hook.
    fn hook_module(&self, key: &PipelineKey) -> Option<vk::ShaderModule> {
        if key.shader == 0 || key.skybox || key.fur || key.outline || key.cluster || key.error || key.depth_only {
            return None;
        }
        let entry = self.hooks.get(key.shader as usize - 1)?.as_ref()?;
        let stage = if key.shadow {
            if key.tess { HookStage::ShadowTess } else { HookStage::Shadow }
        } else if key.multiview {
            HookStage::MainMultiview
        } else {
            HookStage::Main
        };
        entry.modules[stage.index()]
    }

    /// Set the color format of the pass about to be recorded; every variant
    /// fetched afterwards targets it.
    pub fn set_target_format(&mut self, format: vk::Format) {
        self.target_format = format;
    }

    /// Set the view layout of the pass about to be recorded: multiview (both
    /// XR eyes, view mask 0b11) and the colour/depth sample count. Reset to
    /// `(false, 1)` once that pass ends.
    pub fn set_target_views(&mut self, multiview: bool, samples: u8) {
        self.target_multiview = multiview;
        self.target_samples = samples.max(1);
    }

    pub fn get(&mut self, device: &ash::Device, mut key: PipelineKey) -> Result<vk::Pipeline> {
        if key.shadow || key.prepass {
            key.color_format = vk::Format::UNDEFINED;
        } else if key.color_format == vk::Format::UNDEFINED {
            key.color_format = self.target_format;
        }
        if !key.shadow && !key.prepass {
            key.multiview = self.target_multiview;
            key.samples = self.target_samples;
        }
        // No tessellation on this device (or not under multiview): the material draws flat.
        if key.tess && (self.tess.is_none() || key.skybox || key.fur || key.outline || key.cluster || key.error || key.depth_only || (key.multiview && !self.tess_multiview)) {
            key.tess = false;
        }
        if let Some(&pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline);
        }
        self.ensure_hook_module(device, &key)?;
        tracing::debug!(?key, "compiling pipeline variant");
        let pipeline = self.create_variant(device, key)?;
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    fn create_variant(&self, device: &ash::Device, key: PipelineKey) -> Result<vk::Pipeline> {
        let spec_data: [u32; 7] = [
            key.toon as u32,
            key.normal_map as u32,
            key.emission as u32,
            key.alpha_mode,
            key.water as u32,
            key.velocity as u32,
            key.per_instance as u32,
        ];
        let spec_entries: Vec<_> = (0..7u32)
            .map(|id| vk::SpecializationMapEntry {
                constant_id: id,
                offset: id * 4,
                size: 4,
            })
            .collect();
        let spec = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(bytemuck::cast_slice(&spec_data));
        // Custom shaders: keyword constants 0..N plus the alpha mode. Entries
        // for ids a module doesn't declare are ignored by Vulkan.
        let mut custom_data = [0u32; CUSTOM_KEYWORDS + 2];
        for (i, v) in key.spec.iter().enumerate() {
            custom_data[i] = *v as u32;
        }
        custom_data[CUSTOM_KEYWORDS] = key.alpha_mode;
        custom_data[CUSTOM_KEYWORDS + 1] = key.blend as u32;
        let custom_entries: Vec<_> = (0..=CUSTOM_KEYWORDS as u32 + 1)
            .map(|i| vk::SpecializationMapEntry {
                constant_id: match i as usize {
                    n if n == CUSTOM_KEYWORDS => CUSTOM_ALPHA_CONSTANT_ID,
                    n if n == CUSTOM_KEYWORDS + 1 => CUSTOM_BLEND_CONSTANT_ID,
                    _ => i,
                },
                offset: i * 4,
                size: 4,
            })
            .collect();
        let custom_spec = vk::SpecializationInfo::default()
            .map_entries(&custom_entries)
            .data(bytemuck::cast_slice(&custom_data));
        // A hooked standard.vert: the custom keywords plus FEAT_WATER / FEAT_PER_INSTANCE moved
        // out of the keyword id range (see vertex_hook::HOOK_ID_*).
        let mut hook_data = [0u32; CUSTOM_KEYWORDS + 2];
        hook_data[..CUSTOM_KEYWORDS].copy_from_slice(&custom_data[..CUSTOM_KEYWORDS]);
        hook_data[CUSTOM_KEYWORDS] = key.water as u32;
        hook_data[CUSTOM_KEYWORDS + 1] = key.per_instance as u32;
        let hook_entries: Vec<_> = (0..CUSTOM_KEYWORDS as u32 + 2)
            .map(|i| vk::SpecializationMapEntry {
                constant_id: match i as usize {
                    n if n == CUSTOM_KEYWORDS => vertex_hook::HOOK_ID_WATER,
                    n if n == CUSTOM_KEYWORDS + 1 => vertex_hook::HOOK_ID_PER_INSTANCE,
                    _ => i,
                },
                offset: i * 4,
                size: 4,
            })
            .collect();
        let hook_spec = vk::SpecializationInfo::default().map_entries(&hook_entries).data(bytemuck::cast_slice(&hook_data));

        let hooked = self.hook_module(&key);
        let tess = if key.tess { self.tess.as_ref() } else { None };
        let vert_module = if let Some(h) = hooked {
            h
        } else if key.shadow {
            if key.cluster {
                self.shadow_vert_cluster
            } else if let Some(t) = tess {
                t.shadow_vert
            } else {
                self.shadow_vert
            }
        } else if key.skybox {
            if key.multiview { self.skybox_vert_mv } else { self.skybox_vert }
        } else if key.fur {
            self.fur_vert
        } else if key.outline {
            self.outline_vert
        } else if key.cluster {
            if key.multiview { self.vert_cluster_mv } else { self.vert_cluster }
        } else if key.multiview {
            self.vert_mv
        } else {
            self.vert
        };
        let frag_stage = if key.skybox {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.skybox_frag)
                .name(c"main")
        } else if key.depth_only {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.error_frag)
                .name(c"main")
        } else if key.fur {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fur_frag)
                .name(c"main")
        } else if key.outline {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.outline_frag)
                .name(c"main")
        } else if key.error {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.error_frag)
                .name(c"main")
        } else if key.shader > 0 {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.custom_frags[key.shader as usize - 1])
                .name(c"main")
                .specialization_info(&custom_spec)
        } else {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(if key.multiview { self.frag_mv } else { self.frag })
                .name(c"main")
                .specialization_info(&spec)
        };
        let mut vertex_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main");
        if vert_module == self.vert || vert_module == self.vert_mv || vert_module == self.vert_cluster || vert_module == self.vert_cluster_mv {
            vertex_stage = vertex_stage.specialization_info(&spec);
        } else if hooked.is_some() && !key.shadow {
            vertex_stage = vertex_stage.specialization_info(&hook_spec);
        } else if hooked.is_some() {
            // The shadow hook shader declares only the custom keywords.
            vertex_stage = vertex_stage.specialization_info(&custom_spec);
        }
        // The shadow pass is depth-only: vertex stage alone, no fragment. So is
        // the prepass for plain opaque draws (nothing can discard).
        let prepass_vertex_only = key.prepass && key.alpha_mode != 1 && key.alpha_mode != 3 && key.shader == 0 && !key.error;
        let mut stages: Vec<vk::PipelineShaderStageCreateInfo> = if key.shadow || prepass_vertex_only {
            vec![vertex_stage]
        } else {
            vec![vertex_stage, frag_stage]
        };
        if let Some(t) = tess {
            let (tesc, tese) = if key.shadow {
                (t.shadow_tesc, t.shadow_tese)
            } else {
                (t.tesc, if key.multiview { t.tese_mv } else { t.tese })
            };
            stages.insert(
                1,
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::TESSELLATION_CONTROL).module(tesc).name(c"main"),
            );
            stages.insert(
                2,
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION).module(tese).name(c"main"),
            );
        }

        let mut bindings = vec![
            vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
            // Per-instance model matrix, base colour and one parameter vec4
            // (standard / outline vertex shaders).
            vk::VertexInputBindingDescription::default()
                .binding(1)
                .stride(INSTANCE_STRIDE)
                .input_rate(vk::VertexInputRate::INSTANCE),
        ];
        let mut instance_attributes: Vec<_> = (0..4)
            .map(|c| vk::VertexInputAttributeDescription {
                location: 8 + c,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: c * 16,
            })
            .collect();
        // Per-instance colour (offset 64) and parameter vec4 (offset 80).
        instance_attributes.push(vk::VertexInputAttributeDescription {
            location: 13,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 64,
        });
        instance_attributes.push(vk::VertexInputAttributeDescription {
            location: 14,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 80,
        });
        let mut attributes = vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 48,
            },
            // uv1 (lightmap UVs), offset 64, after tangent.
            vk::VertexInputAttributeDescription {
                location: 5,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 64,
            },
            // Skinning joint indices (uvec4), offset 72.
            vk::VertexInputAttributeDescription {
                location: 6,
                binding: 0,
                format: vk::Format::R32G32B32A32_UINT,
                offset: 72,
            },
            // Skinning weights (vec4), offset 88.
            vk::VertexInputAttributeDescription {
                location: 7,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 88,
            },
        ];
        // Velocity variants also read the previous frame's skinned position.
        if key.velocity {
            bindings.push(
                vk::VertexInputBindingDescription::default()
                    .binding(2)
                    .stride(size_of::<Vertex>() as u32)
                    .input_rate(vk::VertexInputRate::VERTEX),
            );
            attributes.push(vk::VertexInputAttributeDescription {
                location: 12,
                binding: 2,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            });
        }
        // The shared vertex module exports previous clip position even when
        // the fragment variant does not write velocity. Supply current position
        // in that case; an absent vertex attribute is undefined Vulkan input.
        if !key.velocity {
            attributes.push(vk::VertexInputAttributeDescription {
                location: 12, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0,
            });
        }
        let instanced_attributes: Vec<_> = attributes.iter().chain(&instance_attributes).copied().collect();
        // The skybox is a vertex-buffer-less fullscreen triangle; every other
        // variant (shadow included) reads the instance stream.
        let vertex_input = if key.skybox || key.cluster {
            vk::PipelineVertexInputStateCreateInfo::default()
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&bindings)
                .vertex_attribute_descriptions(&instanced_attributes)
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(if tess.is_some() { vk::PrimitiveTopology::PATCH_LIST } else { vk::PrimitiveTopology::TRIANGLE_LIST });
        let tessellation_state = vk::PipelineTessellationStateCreateInfo::default().patch_control_points(3);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(if key.outline {
                // Inverted-hull outline keeps back faces only.
                vk::CullModeFlags::FRONT
            } else if key.shadow || key.double_sided || key.skybox || key.fur {
                // Shadow pass renders both faces so the closest (light-facing)
                // surface lays the depth. Culling front faces (second-depth)
                // instead leaks a lit line along a caster's light terminator;
                // a slope-scaled bias in the sampler handles the acne.
                vk::CullModeFlags::NONE
            } else if key.cull_front {
                // Unity `Cull Front`: only the back faces of the mesh show.
                vk::CullModeFlags::FRONT
            } else {
                vk::CullModeFlags::BACK
            })
            // Counter-clockwise like glTF; the scene viewport flips Y with
            // negative height, preserving winding. Mirrored passes reverse it.
            .front_face(if key.mirrored { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE })
            .line_width(1.0);

        let samples = match key.samples {
            4 => vk::SampleCountFlags::TYPE_4,
            2 => vk::SampleCountFlags::TYPE_2,
            _ => vk::SampleCountFlags::TYPE_1,
        };
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);

        let transparent = key.alpha_mode >= 2; // 2 = blend, 3 = transclipping (test and blend)
        // Water is blended but writes its surface's depth in the colour
        // passes: the deferred reflection resolve (SSR / RT) starts its rays
        // from the depth buffer, and the water's G-buffer texel describes the
        // surface, not the floor under it. Its refraction and the occlusion
        // culling read the opaque copy taken before it, so neither sees it.
        // Not in the XR eyes: they take no opaque copy (their culling would
        // see the surface and drop what lies under it) and run no resolve.
        let water_depth = key.water && !key.shadow && !key.depth_only && !key.multiview;
        let mut depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            // Skybox tests against the cleared far depth but never writes, so
            // geometry drawn afterwards always wins.
            .depth_write_enable(
                key.prepass
                    || (!key.depth_equal && !key.skybox && ((!transparent && !key.outline) || key.depth_only || water_depth) && key.stencil.depth_write),
            )
            .depth_compare_op(if key.depth_equal { vk::CompareOp::EQUAL } else { vk::CompareOp::LESS_OR_EQUAL });
        if key.stencil.is_active() && !key.shadow && !key.prepass && crate::sprite_gpu::has_stencil(self.depth_format) {
            depth_stencil = depth_stencil
                .stencil_test_enable(true)
                .front(vk_stencil_face(&key.stencil, &key.stencil.front))
                .back(vk_stencil_face(&key.stencil, &key.stencil.back));
        }

        let blend_attachment = if key.depth_only {
            // No color writes; only the depth buffer is touched.
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::empty())
        } else if transparent {
            // Alpha 0 / Additive 1 / Multiply 2 / Premultiplied 3 (`BlendMode`).
            let (src, dst) = match key.blend {
                1 => (vk::BlendFactor::SRC_ALPHA, vk::BlendFactor::ONE),
                2 => (vk::BlendFactor::DST_COLOR, vk::BlendFactor::ZERO),
                3 => (vk::BlendFactor::ONE, vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
                _ => (vk::BlendFactor::SRC_ALPHA, vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
            };
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(src)
                .dst_color_blend_factor(dst)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        };
        // A stencil writer (`ColorMask` 0) draws no colour at all: not the HDR
        // target, nor the G-buffer / velocity / reactive attachments.
        let mask = vk::ColorComponentFlags::from_raw((key.stencil.color_mask & 15) as u32);
        let blend_attachment = if key.stencil.color_mask & 15 != 15 { blend_attachment.color_write_mask(mask) } else { blend_attachment };
        // The G-buffer attachment (deferred SSR) always writes RGBA so the
        // resolve pass reads valid reflectance/roughness; it never alpha-blends.
        let gbuf_blend = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        // The velocity attachment is RG16F (two channels).
        let velocity_blend = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G);
        // The reactive mask is single-channel R8 (red only).
        let reactive_blend = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R);
        let (gbuf_blend, velocity_blend, reactive_blend) = if key.stencil.color_mask & 15 == 0 {
            let off = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::empty());
            (off, off, off)
        } else {
            (gbuf_blend, velocity_blend, reactive_blend)
        };
        // The shadow pass has no color attachment; gbuf / velocity add more.
        let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = match (key.gbuf, key.velocity) {
            (true, true) => vec![blend_attachment, gbuf_blend, velocity_blend, reactive_blend],
            (true, false) => vec![blend_attachment, gbuf_blend],
            (false, true) => vec![blend_attachment, velocity_blend, reactive_blend],
            (false, false) => vec![blend_attachment],
        };
        let color_blend = if key.shadow || key.prepass {
            vk::PipelineColorBlendStateCreateInfo::default()
        } else {
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments)
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let cf = if key.color_format == vk::Format::UNDEFINED { self.color_format } else { key.color_format };
        let color_formats: Vec<vk::Format> = match (key.gbuf, key.velocity) {
            (true, true) => vec![cf, self.gbuf_format, VELOCITY_FORMAT, REACTIVE_FORMAT],
            (true, false) => vec![cf, self.gbuf_format],
            (false, true) => vec![cf, VELOCITY_FORMAT, REACTIVE_FORMAT],
            (false, false) => vec![cf],
        };
        // Shadow maps stay depth-only (D32); every other pass shares the scene
        // depth(-stencil) format and, when it has stencil, declares the stencil
        // attachment too (all of them attach it: see `stencil_attachment`).
        let depth_fmt = if key.shadow { vk::Format::D32_SFLOAT } else { self.depth_format };
        let stencil_fmt = if crate::sprite_gpu::has_stencil(depth_fmt) { depth_fmt } else { vk::Format::UNDEFINED };
        let mut rendering = if key.shadow || key.prepass {
            // Depth-only target: no color attachments.
            vk::PipelineRenderingCreateInfo::default()
                .depth_attachment_format(depth_fmt)
                .stencil_attachment_format(stencil_fmt)
        } else {
            vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&color_formats)
                .depth_attachment_format(depth_fmt)
                .stencil_attachment_format(stencil_fmt)
                .view_mask(if key.multiview { 0b11 } else { 0 })
        };

        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(if key.cluster { self.cluster_layout } else { self.layout })
            .push_next(&mut rendering);
        if tess.is_some() {
            info = info.tessellation_state(&tessellation_state);
        }

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)
        }
        .context("creating standard shader pipeline variant")?[0];
        Ok(pipeline)
    }

    /// Hot-reload the standard fragment shader from fresh SPIR-V (dev convenience).
    /// Swaps in the new module and drops every cached pipeline so each variant
    /// rebuilds lazily against the new shader on its next use. The caller MUST have
    /// waited for the device to idle first (no in-flight use of the old pipelines).
    pub fn reload_standard_frag(&mut self, device: &ash::Device, frag_spirv: &[u8]) -> Result<()> {
        let frag_code = ash::util::read_spv(&mut Cursor::new(frag_spirv))
            .context("reading recompiled standard.frag SPIR-V")?;
        let new_frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code),
                None,
            )?
        };
        unsafe {
            // Drop all cached pipelines (they reference the old frag module). Only
            // the frag changed; each variant rebuilds lazily with its own modules.
            for (_, pipeline) in self.pipelines.drain() {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_shader_module(self.frag, None);
        }
        self.frag = new_frag;
        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for (_, pipeline) in self.pipelines.drain() {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline_layout(self.cluster_layout, None);
            device.destroy_descriptor_set_layout(self.cluster_set_layout, None);
            device.destroy_descriptor_set_layout(self.set0_layout, None);
            device.destroy_descriptor_set_layout(self.set1_layout, None);
            device.destroy_descriptor_set_layout(self.set2_layout, None);
            device.destroy_shader_module(self.vert, None);
            device.destroy_shader_module(self.frag, None);
            device.destroy_shader_module(self.error_frag, None);
            device.destroy_shader_module(self.outline_vert, None);
            device.destroy_shader_module(self.outline_frag, None);
            device.destroy_shader_module(self.fur_vert, None);
            device.destroy_shader_module(self.fur_frag, None);
            device.destroy_shader_module(self.skybox_vert, None);
            device.destroy_shader_module(self.skybox_frag, None);
            device.destroy_shader_module(self.shadow_vert, None);
            device.destroy_shader_module(self.shadow_vert_cluster, None);
            device.destroy_shader_module(self.vert_mv, None);
            device.destroy_shader_module(self.vert_cluster, None);
            device.destroy_shader_module(self.vert_cluster_mv, None);
            device.destroy_shader_module(self.frag_mv, None);
            device.destroy_shader_module(self.skybox_vert_mv, None);
            for module in self.custom_frags.drain(..) {
                device.destroy_shader_module(module, None);
            }
            for entry in self.hooks.drain(..).flatten() {
                for module in entry.modules.into_iter().flatten() {
                    device.destroy_shader_module(module, None);
                }
            }
            if let Some(t) = self.tess.take() {
                for module in [t.tesc, t.tese, t.tese_mv, t.shadow_tesc, t.shadow_tese, t.shadow_vert] {
                    device.destroy_shader_module(module, None);
                }
            }
        }
    }
}
