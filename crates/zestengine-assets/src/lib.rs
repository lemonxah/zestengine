//! zestengine-assets: CPU-side scene loading.
//!
//! M2: glTF 2.0 worlds/props + a procedural test scene. M3 adds VRM avatar
//! parsing (humanoid rig, expressions, spring bones) on top of the glTF
//! loader.

// Without the `client` feature the render-free stand-in answers to the same crate name.
#[cfg(not(feature = "client"))]
extern crate zestengine_render_lite as zestengine_render;

mod anim_clip;
pub mod anim_import;
pub mod anim_pack;
pub mod fbx_extras;
#[cfg(test)]
mod fbx_test_util;
#[cfg(test)]
mod fbx_extras_tests;
#[cfg(test)]
mod skeleton_chain_tests;
#[cfg(test)]
mod real_avatar_tests;
mod fbx_anim;
pub use fbx_anim::reference_bone_positions;
mod anim_controller;
mod asset_meta;
mod bake_file;
mod behaviour_tree;
mod collision;
mod fbx_loader;
pub mod auto_map;
pub mod rig_import;
pub mod avatar_mask;
pub mod avatar_build;
pub mod human_template;
pub mod human_rig;
pub mod ref_pose;
pub mod muscle;
pub mod rig_biped;
pub mod skin_rebind;
pub mod rig_config;
mod gltf_loader;
pub mod import_cache;
pub mod import_hooks;
pub mod hdr_codec;
pub mod mtoon;
pub mod vrm;
pub mod simplify;
mod lightmap_uv;
mod material_file;
pub mod water_file;
mod meshlet;
pub mod cluster_dag;
pub mod cluster_pages;
#[cfg(feature = "client")]
pub mod cluster_stream;
pub mod geo_cache;
pub mod impostor;
mod mesh_normals;
pub mod mesh_pipeline;
pub mod material_remap;
// Unity material conversion (Poiyomi / Mochie / lilToon / MToon / Standard -> ZestEngine).
pub mod unity_mat;
pub mod unity_convert;
pub mod unity_package;
// Megascans / Fab / MetaHuman package import (IMP-8).
pub mod pack_zip;
pub mod tex_roles;
pub mod megascans;
pub mod pack_scan;
pub mod pack_convert;
pub mod mhpkg;
mod post_file;
pub mod audio_file;
pub mod profile_file;
pub use audio_file::{AUDIOFX_EXTENSION, AUDIO_SETTINGS_EXTENSION, load_audiofx, save_audiofx,
    load_audio_settings, save_audio_settings, project_audio_settings};
mod procedural;
mod prefab_link;
mod project_file;
mod scene_file;
mod renderer_migration;
mod sequencer;
mod shader_file;
mod shader_props;
mod shader_graph;
mod skeleton;
pub mod aseprite;
pub mod atlas_pack;
pub mod ik2d;
pub mod layered_import;
pub mod layered_prefab;
pub mod psd;
pub mod rig2d;
pub mod rig2d_edit;
pub mod sprite_library;
pub mod sprite;
pub mod sprite_atlas;
pub mod sprite_edit;
pub mod sprite_mesh;
pub mod sprite_slice;
pub mod sub_asset;
mod tex_cache;
mod texture_import;
mod texture_pipeline;
pub mod package;
pub mod net_strip;
pub mod server_model;
pub mod perf_rank;
pub mod auto_lod;
pub mod lod_cache;
pub mod game_pak;
// Project Hub support: template catalogue, recent projects, sibling executables.
pub mod launcher;
pub mod project_templates;
pub mod recent_projects;

pub use asset_meta::{
    AssetMeta, ImporterSettings, IndexFormat, META_EXT, MaterialNaming, MaterialSearch, MeshCompression, ModelImport, NormalWeighting, NormalsMode,
    TEXTURE_REMAP_SLOTS, TangentMode, UV_ALL, load_asset_meta, load_or_create_asset_meta, meta_path, save_asset_meta, texture_remap_key,
};
pub use anim_import::{AnimCompression, AnimImport, BlendNormals, BlendShapeImport, ClipDef, ClipEventDef, ClipImportInfo, ClipPlayback, TakeInfo};
pub use anim_clip::{
    ANIM_CLIP_EXTENSION, AnimClip, AnimTrack, ClipEvent, ClipKind, EventArgs, KeyInterp, Keyframe, SourceBone, TrackProperty,
    load_anim_clip, save_anim_clip,
};
pub use avatar_mask::{AvatarMask, MASK_PARTS, MaskElement, MaskPart, load_avatar_mask, save_avatar_mask};
pub use anim_controller::{
    ANIM_CONTROLLER_EXTENSION, AnimCondition, AnimLayer, AnimParamKind, AnimParameter, AnimState,
    AnimTransition, AnimatorController, CondOp, InterruptionSource, LayerBlend, Motion, MotionOverride, StateMachine,
    load_anim_controller, save_anim_controller,
};
pub use collision::{
    CollisionKind, CollisionPiece, extract_collision, match_collision_target, parse_collision_name,
};
pub use fbx_loader::{load_fbx, load_fbx_with};
pub use gltf_loader::{load_gltf, load_gltf_with};
pub use lightmap_uv::generate_lightmap_uv;
pub use skeleton::{
    AnimChannel, AnimationClip, ChannelExtra, ChannelPath, Joint, PackedValues, Skeleton, Tangents, skin_direction, skin_position,
};
pub use material_file::{
    MATERIAL_EXTENSION, MaterialFile, MaterialTextures, load_material_file, load_texture_file,
    save_material_file,
};
pub use meshlet::{Meshlet, build_lod_meshlets, build_lod_meshlets_mapped, build_meshlets};
pub use bake_file::{
    LIGHTDATA_EXTENSION, LIGHTMAP_EXTENSION, LightDataFile, LightmapEntry, LightmapFile, LightmapFileInfo, LightmapLayers,
    ProbeVolumeData, load_lightdata, load_lightmap_layers, lightmap_file_info, load_lightmaps, read_lightmap_layers, save_lightdata,
    save_lightmap_layers, save_lightmaps, write_atomic,
};
pub use behaviour_tree::{
    BEHAVIOUR_TREE_EXTENSION, BehaviourTree, load_behaviour_tree, save_behaviour_tree,
};
pub use post_file::{
    AmbientOcclusion, Bloom, ChromaticAberration, ColorGrading, FluxPostFx, POSTFX_EXTENSION, PostFxProfile, TonemapMode,
    Vignette, blend_over, blend_profiles, load_postfx, save_postfx,
};
pub use procedural::{BUCKET_FLOOR, BUCKET_WALL, heightfield_mesh, heightfield_mesh_with_weights, import_heightmap, primitive_mesh, test_scene};
pub use project_file::{
    PROJECT_FILE_NAME, ProjectFile, ProjectSettings, load_project_file, save_project_file,
};
pub use prefab_link::{
    ApplyLevel, Change, FieldChange, IdMode, InstanceInfo, ObjectDiff, PrefabLink, PrefabOverride, PrefabScope, apply_prefab, asset_image, compact_prefabs, expand_prefabs,
    instance_info, live_id, remerge_prefabs, revert_prefab, component_keys,
};
pub use renderer_migration::{RENDERER_FORMAT, has_renderer, material_refs, material_ref_of, material_slot_of, migrate_entries, migrate_link, slot_count_of};
pub use scene_file::{
    ComponentData, EditorCamera, FluxQuality, GiMode, MaterialRef, ModelInstanceRef, ModelPart, ObjectSource, PartRef, PrimitiveShape,
    ProbeFallback, RealtimeGi, SCENE_EXTENSION, SceneEntry, BakeSettings, SceneFile,
    VoxelGridMode, WorldEnvironment, DIRECTIONAL_LIGHT_NAME, default_directional_light_entry, directional_light_entry, load_scene_file, save_scene_file,
};
pub use sequencer::{
    SEQUENCER_EXTENSION, SeqChannel, SeqCurve, SeqInterp, SeqKey, SeqTrack, SeqTrackKind, Sequencer,
    load_sequencer, save_sequencer,
};
pub use tex_cache::{
    ASTC_CACHE_EXTENSION, AstcBlocks, TexturePreview, astc_blocks, astc_cache_blob, compress_embedded_texture, encode_astc_level, import_cache_path, load_texture_bc,
    load_texture_import, preview_texture, seed_astc_cache, seed_texture_cache, set_astc_blocks, texture_cache_blob, texture_mean,
    texture_mean_linear, texture_vram_bytes, validate_compressed,
};
pub use texture_import::{
    AlphaSource, Channel, Compression, FilterMode, FormatOverride, MipFilter, MipOptions, NormalFilter, NormalOptions, NpotScale, Platform,
    PlatformOverride, ResizeFilter, SamplerOptions, SlotExpect, SlotIssue, SlotIssueKind, SlotUse, SourceInfo, TargetFormat, TextureImport,
    TextureKind, TextureWarning, WarnLevel, WrapMode, apply_expectation, conflicting_uses, duplicate_path, import_for_expectation,
    ensure_first_import_for_slot, name_suggests_normal_map, resolve_import, slot_expectation, slot_issues,
};
pub use shader_file::{
    ALPHA_MODE_CONSTANT_ID, HLSL_SHADER_EXTENSION, ShaderCondOp, ShaderCondTerm, ShaderCondition,
    HLSL_SHADER_TEMPLATE, KeywordSet, MAX_SHADER_KEYWORDS, MAX_SHADER_TEXTURES, PropStorage,
    SHADER_EXTENSION, SHADER_PROP_FLOATS, SHADER_TEMPLATE, SHADER_TEXTURE_SLOT_BASE,
    SHADER_UBO_MAX_BYTES, SHADER_VALUE_BASE, ShaderGroup, ShaderLang, ShaderPreset, ShaderProp,
    ShaderPropKind, ShaderSource, TextureDefault, assemble_shader, bake_shader_variant,
    compile_shader, compile_shader_variant, is_shader_path, keyword_hash, load_baked_variant,
    load_shader_file, parse_shader_source, parse_shader_source_lang, precompile_shader_file,
    scoped_shader_target_env, shader_lang_for, shader_target_env, variant_path, ShaderTargetEnv,
    ShaderTargetEnvGuard, VertexSection, split_vertex_section, vertex_hook_glsl,
};
pub use shader_graph::{
    SHADER_GRAPH_EXTENSION, GraphNode, GraphNodeKind, GraphTextureSlot, GraphValueType,
    ShaderGraph, default_graph, generate_glsl, graph_shader_source, infer_types,
    is_shader_graph_path, load_graph, load_shader_graph, save_shader_graph, topo_order, validate,
};

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use zestengine_render::{MaterialFeatures, MaterialParams, MeshData, TextureData};
use glam::Mat4;

/// One (mesh, material) sub-part of an instance.
#[derive(Clone, Copy)]
pub struct MeshSlot {
    pub mesh: usize,
    pub material: usize,
}

/// One renderable placement at a transform, with one or more material slots.
/// A single-material node has one slot; a multi-material mesh has one slot per
/// material (all drawn at the same transform, presented as one scene object).
pub struct Instance {
    pub name: String,
    pub transform: Mat4,
    pub slots: Vec<MeshSlot>,
}

impl Instance {
    /// A single-material instance (the common case).
    pub fn single(name: impl Into<String>, mesh: usize, material: usize, transform: Mat4) -> Self {
        Self {
            name: name.into(),
            transform,
            slots: vec![MeshSlot { mesh, material }],
        }
    }
}

/// A material referencing scene-local texture indices; the engine maps
/// these to renderer handles at upload.
pub struct SceneMaterial {
    pub name: String,
    pub params: MaterialParams,
    pub features: MaterialFeatures,
    pub albedo: Option<usize>,
    pub normal: Option<usize>,
    pub orm: Option<usize>,
    pub emission: Option<usize>,
}

/// One blendshape (morph target) of a mesh: sparse per-vertex offsets.
#[derive(Clone, Debug, Default)]
pub struct MorphTarget {
    pub name: String,
    /// (vertex index, position offset, normal offset).
    pub deltas: Vec<(u32, [f32; 3], [f32; 3])>,
}

pub struct Scene {
    pub meshes: Vec<MeshData>,
    /// Blendshapes per mesh, parallel to `meshes` (shorter or empty = none).
    pub morphs: Vec<Vec<MorphTarget>>,
    /// VRM spring bones and colliders (empty for other models).
    pub springs: vrm::SpringImport,
    pub textures: Vec<TextureData>,
    pub materials: Vec<SceneMaterial>,
    pub instances: Vec<Instance>,
    /// Imported armatures (one per glTF skin / FBX skin deformer). Vertex joint
    /// indices are skin-local and line up with `skeletons[mesh's skin].joints`.
    pub skeletons: Vec<Skeleton>,
    /// Imported skeletal animation clips (shared across the file's skeletons).
    pub animations: Vec<AnimationClip>,
    /// Authored collision pieces (`UCX_` / `UBX_` / `USP_` / `UCP_` objects),
    /// filled by [`extract_collision`] on load; not in `instances`.
    pub collision: Vec<CollisionPiece>,
    /// Stable ids of the meshes, so scenes can reference them by id instead of index.
    pub identity: ModelIdentity,
}

/// A camera found in the model file (when Import Cameras is on).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImportedCamera {
    pub name: String,
    /// Column-major world transform.
    pub transform: [f32; 16],
    /// Vertical field of view, degrees.
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

/// A light found in the model file (when Import Lights is on).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImportedLight {
    pub name: String,
    /// Column-major world transform (lights shine down -Z).
    pub transform: [f32; 16],
    pub kind: zestengine_core::LightKind,
    /// Linear RGB.
    pub color: [f32; 3],
    pub intensity: f32,
    /// 0 = unbounded.
    pub range: f32,
    /// Full cone angle for spot lights, degrees.
    pub spot_angle_deg: f32,
}

/// Identity of an imported model's sub-assets (see [`sub_asset`]), plus the
/// import-side extras that travel with it. Importers fill `mesh_names`,
/// `cameras` and `lights`; [`load_model_with_meta`] fills the rest.
#[derive(Clone, Debug, Default)]
pub struct ModelIdentity {
    /// The model's `.meta` GUID (`None` when imported without a meta).
    pub guid: Option<zestengine_core::ObjectId>,
    /// Unique name of each mesh (parallel to `Scene::meshes`), e.g. `Root/Body#part0`.
    pub mesh_names: Vec<String>,
    /// Stable id of each mesh (parallel to `Scene::meshes`); empty until assigned.
    pub mesh_ids: Vec<sub_asset::SubAssetId>,
    /// Unique name / stable id of each material, skeleton and clip (parallel to
    /// `Scene::materials` / `skeletons` / `animations`); empty until assigned.
    pub material_names: Vec<String>,
    pub material_ids: Vec<sub_asset::SubAssetId>,
    pub skeleton_names: Vec<String>,
    pub skeleton_ids: Vec<sub_asset::SubAssetId>,
    pub clip_names: Vec<String>,
    pub clip_ids: Vec<sub_asset::SubAssetId>,
    /// Source file stem of each texture (parallel to `Scene::textures`; may be
    /// shorter or hold empty names), for Search and Remap by texture name.
    pub texture_names: Vec<String>,
    /// The `.material` each material is remapped to (parallel to `materials`),
    /// from the model's remap table; filled by [`load_model_with_meta`].
    pub material_remap: Vec<Option<String>>,
    /// Remapped texture paths per material, in albedo / normal / orm / emission order.
    pub texture_remap: Vec<[Option<String>; 4]>,
    /// Cameras and lights the file carries (empty unless the import settings ask for them).
    pub cameras: Vec<ImportedCamera>,
    pub lights: Vec<ImportedLight>,
    /// A `.material` an import hook assigned to each material (parallel to
    /// `materials`; empty = none). Part of the cached artifact; the model's own
    /// remap table wins over it.
    pub hook_material_assign: Vec<Option<String>>,
    /// Quads kept by Keep Quads, per mesh (parallel to `Scene::meshes`, may be
    /// shorter): each is four vertex indices in polygon order, the triangles
    /// `(a, b, c)` and `(a, c, d)` of the mesh. Empty without Keep Quads.
    pub quads: Vec<Vec<[u32; 4]>>,
    /// Instances (indices into `Scene::instances`) the file hides (FBX node
    /// visibility, with Import Visibility): placed disabled.
    pub hidden_instances: Vec<usize>,
    /// The Generate Colliders setting: every placed part gets a Mesh Collider.
    pub generate_colliders: bool,
    /// The Read/Write setting.
    pub read_write: bool,
    /// The takes (animation stacks) the file carries, for the clip list.
    pub takes: Vec<TakeInfo>,
    /// What the importer knows about each clip (parallel to `Scene::animations`).
    pub clip_info: Vec<ClipImportInfo>,
    /// Initial blendshape weights from the file (`DeformPercent`, Import Deform
    /// Percent): (mesh index, shape name, weight). Placed on the instance at spawn.
    pub morph_weights: Vec<(usize, String, f32)>,
    /// glTF transmission / volume maps repacked into one texture per material
    /// (parallel to `materials`, may be empty): red = thickness, green =
    /// transmission. Index into `Scene::textures`; rides in the translucency /
    /// thickness slot.
    pub material_volume_maps: Vec<Option<usize>>,
    /// The file's constraints (Import Constraints): placed as Constraint
    /// components on the parts they constrain.
    pub constraints: Vec<zestengine_core::ConstraintSpec>,
    /// User properties of the file's nodes (Animated Custom Properties):
    /// `(node name, [(property, initial value)])`, placed as Custom Properties.
    pub custom_props: Vec<(String, Vec<(String, zestengine_core::CustomValue)>)>,
    /// Problems the import noticed and the inspector shows (vertices over the
    /// bone limit, constraints that cannot be solved, ...).
    pub import_warnings: Vec<String>,
    /// The Optimize Game Objects settings the model was imported with.
    pub optimize: OptimizeInfo,
}

/// Optimize Game Objects as recorded by the import: the bones the skeleton
/// asset keeps to itself and the transforms that stay addressable.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OptimizeInfo {
    pub enabled: bool,
    /// Extra Expose Transforms that name a joint or node of the model.
    pub exposed: Vec<String>,
    /// Joints that are internal to the skeleton asset (everything not exposed).
    pub internal_joints: usize,
}

impl ModelIdentity {
    /// Record the quad table of mesh `mesh` (a no-op for an empty one).
    pub fn set_quads(&mut self, mesh: usize, quads: Vec<[u32; 4]>) {
        if quads.is_empty() {
            return;
        }
        if self.quads.len() <= mesh {
            self.quads.resize(mesh + 1, Vec::new());
        }
        self.quads[mesh] = quads;
    }
}

/// True if the extension is an importable model format.
pub fn is_model_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase)
            .as_deref(),
        Some("gltf" | "glb" | "vrm" | "fbx")
    )
}

/// Sidecar marker that records "generate lightmap UVs for this model's meshes".
/// Its presence (content unused) means each mesh lacking a real UV1 gets a
/// deterministic generated unwrap on load — so the generated UVs persist across
/// sessions without storing the whole rebuilt mesh.
pub fn lightmap_uv_marker_path(model: impl AsRef<Path>) -> PathBuf {
    model.as_ref().with_extension("lmuv")
}

/// Create (or remove) the lightmap-UV marker for a model. Called by the editor
/// when the user opts to generate lightmap UVs at bake time.
pub fn set_lightmap_uv_marker(model: impl AsRef<Path>, on: bool) -> Result<()> {
    let p = lightmap_uv_marker_path(&model);
    if on {
        std::fs::write(&p, b"zestengine-lightmap-uv\n")
            .with_context(|| format!("writing lightmap-UV marker {}", p.display()))?;
    } else if p.exists() {
        std::fs::remove_file(&p)
            .with_context(|| format!("removing lightmap-UV marker {}", p.display()))?;
    }
    Ok(())
}

/// Apply the lightmap-UV marker after loading: replace every mesh that has no
/// real UV1 with a generated non-overlapping unwrap. No-op without the marker.
fn apply_lightmap_uv_marker(path: &Path, scene: &mut Scene) {
    if !lightmap_uv_marker_path(path).exists() {
        return;
    }
    let rendered = collision::rendered_meshes(scene);
    for (mesh, rendered) in scene.meshes.iter_mut().zip(rendered) {
        if rendered && !mesh.has_lightmap_uv {
            *mesh = generate_lightmap_uv(mesh);
        }
    }
}

/// Import any supported model format, dispatching on extension.
pub fn load_model(path: impl AsRef<Path>) -> Result<Scene> {
    let path = path.as_ref();
    let mut scene = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("gltf" | "glb" | "vrm") => load_gltf(path),
        Some("fbx") => load_fbx(path),
        other => bail!("unsupported model format {other:?} (gltf, glb, fbx)"),
    }?;
    extract_collision(&mut scene);
    apply_lightmap_uv_marker(path, &mut scene);
    assign_identity(&mut scene, None, &mut Default::default(), false);
    Ok(scene)
}

/// Fill `scene.identity`: the meta GUID and a stable id per mesh from its name,
/// looked up in `table` (recording new names when `record`).
/// Returns whether `table` changed.
fn assign_identity(
    scene: &mut Scene,
    guid: Option<zestengine_core::ObjectId>,
    table: &mut sub_asset::IdTable,
    record: bool,
) -> bool {
    if scene.identity.mesh_names.len() != scene.meshes.len() {
        // An importer that doesn't name its meshes: fall back to the index.
        scene.identity.mesh_names = (0..scene.meshes.len()).map(|i| format!("mesh{i}")).collect();
    }
    scene.identity.guid = guid;
    let parts = part_names(scene);
    let mut changed = false;
    for (kind, names) in parts {
        let (ids, c) = sub_asset::assign_ids(kind, &names, table, record);
        changed |= c;
        match kind {
            sub_asset::SubAssetKind::Mesh => (scene.identity.mesh_names, scene.identity.mesh_ids) = (names, ids),
            sub_asset::SubAssetKind::Material => (scene.identity.material_names, scene.identity.material_ids) = (names, ids),
            sub_asset::SubAssetKind::Skeleton => (scene.identity.skeleton_names, scene.identity.skeleton_ids) = (names, ids),
            sub_asset::SubAssetKind::Animation => (scene.identity.clip_names, scene.identity.clip_ids) = (names, ids),
            // Sprites belong to textures, never to a model scene.
            sub_asset::SubAssetKind::Sprite => {}
            // Parts are derived from mesh ids ([`ModelPart::of`]), not recorded.
            sub_asset::SubAssetKind::Part => {}
        }
    }
    changed
}

/// The unique name of every mesh, material, skeleton and clip of `scene`, per
/// kind, in scene order. These are what the sub-asset ids hash.
fn part_names(scene: &Scene) -> [(sub_asset::SubAssetKind, Vec<String>); 4] {
    let meshes = if scene.identity.mesh_names.len() == scene.meshes.len() {
        scene.identity.mesh_names.clone()
    } else {
        (0..scene.meshes.len()).map(|i| format!("mesh{i}")).collect()
    };
    [
        (sub_asset::SubAssetKind::Mesh, meshes),
        (sub_asset::SubAssetKind::Material, unique_names(scene.materials.iter().map(|m| m.name.as_str()), "material")),
        (
            sub_asset::SubAssetKind::Skeleton,
            unique_names(
                scene.skeletons.iter().map(|s| s.motion_root().map_or("", |j| s.joints[j].name.as_str())),
                "skeleton",
            ),
        ),
        (sub_asset::SubAssetKind::Animation, unique_names(scene.animations.iter().map(|c| c.name.as_str()), "clip")),
    ]
}

/// Make a list of names unique (later duplicates get `~1`, `~2`, ...), naming
/// empty ones `fallback`. Deterministic in list order, like the mesh names.
fn unique_names<'a>(names: impl Iterator<Item = &'a str>, fallback: &str) -> Vec<String> {
    let mut namer = sub_asset::SiblingNamer::default();
    names.map(|n| namer.unique("", if n.is_empty() { fallback } else { n })).collect()
}

/// The remap table of `model` laid out against `scene`'s materials.
fn apply_remap_tables(scene: &mut Scene, model: &ModelImport) {
    let hook_assigned = scene.identity.hook_material_assign.clone();
    scene.identity.material_remap = scene
        .identity
        .material_names
        .iter()
        .enumerate()
        .map(|(i, n)| model.material_remap.get(n).cloned().or_else(|| hook_assigned.get(i).cloned().flatten()))
        .collect();
    scene.identity.texture_remap = scene
        .identity
        .material_names
        .iter()
        .map(|n| TEXTURE_REMAP_SLOTS.map(|slot| model.texture_remap.get(&texture_remap_key(n, slot)).cloned()))
        .collect();
}

/// Import a model and record its sub-asset names in its `.meta` id table
/// (creating the meta if needed), so later renames through the importer keep
/// the ids. Returns whether the meta was rewritten. Editor-side: the game
/// itself only reads ([`load_model_with_meta`]).
pub fn register_model_ids(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();
    // Goes through the import cache, so this is cheap once the model has loaded once.
    let scene = load_model_with_meta(path)?;
    record_model_ids(path, &scene)
}

/// [`register_model_ids`] for a model that is already imported (the editor
/// calls this for every model it loads): records the mesh names of `scene` in
/// the `.meta` id table, creating the meta if needed. No re-import.
pub fn record_model_ids(path: impl AsRef<Path>, scene: &Scene) -> Result<bool> {
    let path = path.as_ref();
    let mut meta = load_or_create_asset_meta(path)?;
    let Some(model) = meta.importer.as_model_mut() else { return Ok(false) };
    let mut table = std::mem::take(&mut model.id_table);
    // Record all four kinds, named the way `assign_identity` names them.
    let parts = part_names(scene);
    let mut changed = false;
    for (kind, names) in &parts {
        changed |= sub_asset::assign_ids(*kind, names, &mut table, true).1;
    }
    model.id_table = table;
    // The remap table's rows: the materials this import produced.
    let materials = &parts[1].1;
    if model.source_materials != *materials {
        model.source_materials = materials.clone();
        changed = true;
    }
    if changed {
        save_asset_meta(path, &meta)?;
    }
    Ok(changed)
}

/// Rename a model's mesh through the importer: the new name gets the old
/// name's id, so scenes referencing it by id keep resolving once the source is
/// re-exported with the new name. Errors if the name is taken or the asset isn't a model.
pub fn rename_model_sub_asset(path: impl AsRef<Path>, kind: sub_asset::SubAssetKind, old: &str, new: &str) -> Result<sub_asset::SubAssetId> {
    let path = path.as_ref();
    let mut meta = load_or_create_asset_meta(path)?;
    let model = meta.importer.as_model_mut().context("not a model asset")?;
    let id = model
        .id_table
        .rename(kind, old, new)
        .with_context(|| format!("'{new}' is already a sub-asset name"))?;
    save_asset_meta(path, &meta)?;
    Ok(id)
}

/// Load a model applying its `.meta` sidecar import settings if the sidecar
/// exists (read-only, safe in a shipped game; the editor creates/edits the meta
/// separately). FBX honours the settings; glTF uses defaults for now.
pub fn load_model_with_meta(path: impl AsRef<Path>) -> Result<Scene> {
    load_model_cached(path, false).map(|(scene, _)| scene)
}

/// [`load_model_with_meta`] plus the model's cluster DAGs ([`geo_cache`]): one
/// per mesh at or above [`cluster_dag::GEO_MIN_TRIANGLES`] triangles, read from
/// the model's `.zestengine_geocache/` when its key and validation hold, else
/// built and stored. A cache miss costs seconds on a big model (about 2.5 s per
/// million triangles on 32 threads), so call this from a worker, never from the
/// frame loop; `ZESTENGINE_NO_GEO_CACHE` builds without touching the disk.
pub fn load_model_with_geometry(path: impl AsRef<Path>) -> Result<(Scene, geo_cache::ModelGeometry)> {
    load_model_with_geometry_tracked(path, &geo_cache::BuildProgress::default())
}

/// [`load_model_with_geometry`] whose DAG build reports through, and can be
/// cancelled by, `progress`.
pub fn load_model_with_geometry_tracked(path: impl AsRef<Path>, progress: &geo_cache::BuildProgress) -> Result<(Scene, geo_cache::ModelGeometry)> {
    let path = path.as_ref();
    let scene = load_model_with_meta(path)?;
    let geo = geo_cache::load_or_build_tracked(path, &scene, progress);
    Ok((scene, geo))
}

/// [`load_model_with_meta`] through the import artifact cache ([`import_cache`]),
/// also returning what the cache did. `force` rebuilds the artifact even if a
/// valid one exists (an explicit reimport).
pub fn load_model_cached(path: impl AsRef<Path>, force: bool) -> Result<(Scene, import_cache::ImportReport)> {
    let path = path.as_ref();
    let meta = load_asset_meta(path)?;
    let model = meta.as_ref().and_then(|m| m.importer.as_model().cloned()).unwrap_or_default();
    let hooks = import_hooks::hooks_for(path);
    // A dedicated-server cook ships collision-only blobs at the model's own path.
    let (mut scene, report) = if server_model::is_blob(path) {
        let t = std::time::Instant::now();
        let scene = server_model::decode(&std::fs::read(path)?)?;
        let report = import_cache::ImportReport {
            outcome: import_cache::CacheOutcome::Disabled,
            key: String::new(),
            key_time: Default::default(),
            load_time: t.elapsed(),
        };
        (scene, report)
    } else {
        import_cache::load_cached_with(path, &model, force, &hooks, || import_model_artifact(path, &model, &hooks))?
    };
    // Copy From Other Avatar takes the source model's rig; when it does not fit, the stored one applies.
    let copied = matches!(&model.rig.avatar_setup, human_rig::AvatarSetup::CopyFrom(src) if rig_import::apply_copy_from(path, src, &mut scene.skeletons));
    if !copied && model.rig.is_configured() {
        for skel in &mut scene.skeletons {
            skel.rig = Some(model.rig.clone());
        }
    }
    // Humanoid Oversampling travels with the rig to the muscle bake.
    for rig in scene.skeletons.iter_mut().filter_map(|s| s.rig.as_mut()) {
        rig.humanoid_oversampling = model.animation.humanoid_oversampling;
    }
    let mut table = model.id_table.clone();
    assign_identity(&mut scene, meta.as_ref().map(|m| m.id), &mut table, false);
    apply_remap_tables(&mut scene, &model);
    // Applied when the model is placed; not part of the cached artifact.
    scene.identity.generate_colliders = model.generate_colliders;
    scene.identity.read_write = model.read_write;
    Ok((scene, report))
}

/// What the import cache stores for a model: the format reader, collision and
/// lightmap-UV markers, then the import hooks ([`import_hooks`], none outside
/// the editor), so the artifact includes their effects.
pub(crate) fn import_model_artifact(path: &Path, model: &ModelImport, hooks: &import_hooks::HookSet) -> Result<Scene> {
    let ctx = (!hooks.is_empty()).then(|| import_hooks::ImportContext::new(path, model));
    let mut effective = model.clone();
    if let Some(ctx) = &ctx {
        hooks.preprocess(ctx, &mut effective);
    }
    let mut scene = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("gltf" | "glb" | "vrm") => load_gltf_with(path, &effective),
        Some("fbx") => load_fbx_with(path, &effective),
        other => bail!("unsupported model format {other:?} (gltf, glb, fbx)"),
    }?;
    extract_collision(&mut scene);
    apply_lightmap_uv_marker(path, &mut scene);
    fbx_loader::record_optimize(&mut scene, &effective);
    if let Some(ctx) = &ctx {
        hooks.process_scene(ctx, &mut scene);
    }
    Ok(scene)
}

#[cfg(test)]
mod import_tests {
    use super::*;

    fn sample(name: &str) -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(name);
        p.exists().then_some(p)
    }

    /// A copy of a sample model in a fresh temp dir (so its `.meta` is ours).
    fn project_copy(tag: &str, rel: &str) -> Option<(PathBuf, PathBuf)> {
        let src = sample(rel)?;
        let dir = std::env::temp_dir().join(format!("zest_import_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dst = dir.join(src.file_name().unwrap());
        std::fs::copy(&src, &dst).unwrap();
        Some((dir, dst))
    }

    #[test]
    fn materials_skeletons_and_clips_get_stable_ids() {
        let Some((dir, model)) = project_copy("ids", "animations/kaykit/Mannequin_Medium.glb") else { return };
        let first = load_model_with_meta(&model).unwrap();
        record_model_ids(&model, &first).unwrap();
        let a = load_model_with_meta(&model).unwrap();
        let id = &a.identity;
        assert_eq!(id.material_ids.len(), a.materials.len());
        assert_eq!(id.skeleton_ids.len(), a.skeletons.len());
        assert_eq!(id.clip_ids.len(), a.animations.len());
        assert!(!a.materials.is_empty() && !a.skeletons.is_empty());
        let all: Vec<_> = id.mesh_ids.iter().chain(&id.material_ids).chain(&id.skeleton_ids).chain(&id.clip_ids).collect();
        let unique: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), all.len(), "ids are unique across kinds");
        assert!(all.iter().all(|i| !i.is_none()));

        // The meta now records every kind and the material names for the remap table.
        let meta = load_asset_meta(&model).unwrap().unwrap();
        let m = meta.importer.as_model().unwrap();
        assert_eq!(m.source_materials, id.material_names);
        for (n, i) in id.material_names.iter().zip(&id.material_ids) {
            assert_eq!(m.id_table.find(sub_asset::SubAssetKind::Material, n), Some(*i));
        }
        for (n, i) in id.skeleton_names.iter().zip(&id.skeleton_ids) {
            assert_eq!(m.id_table.find(sub_asset::SubAssetKind::Skeleton, n), Some(*i));
        }

        // A second import gives the same ids, and reordering the file's materials moves nothing.
        let mut b = load_model_with_meta(&model).unwrap();
        assert_eq!(b.identity.material_ids, id.material_ids);
        let by_name: std::collections::HashMap<_, _> = id.material_names.iter().cloned().zip(id.material_ids.clone()).collect();
        b.materials.reverse();
        let mut table = m.id_table.clone();
        assign_identity(&mut b, None, &mut table, false);
        for (n, i) in b.identity.material_names.iter().zip(&b.identity.material_ids) {
            assert_eq!(by_name[n], *i, "{n} keeps its id when the file lists it elsewhere");
        }
        // Adding a material hands out a new id and leaves the others alone.
        let extra = SceneMaterial { name: "Brand New".into(), params: MaterialParams::default(), features: MaterialFeatures::default(), albedo: None, normal: None, orm: None, emission: None };
        b.materials.push(extra);
        assign_identity(&mut b, None, &mut table, false);
        let new_id = *b.identity.material_ids.last().unwrap();
        assert!(!by_name.values().any(|i| *i == new_id));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remap_tables_are_laid_out_against_the_materials() {
        let Some((dir, model)) = project_copy("remap", "models/DamagedHelmet.glb") else { return };
        let plain = load_model_with_meta(&model).unwrap();
        assert!(plain.identity.material_remap.iter().all(Option::is_none));
        let name = plain.identity.material_names[0].clone();
        let mut meta = load_or_create_asset_meta(&model).unwrap();
        let m = meta.importer.as_model_mut().unwrap();
        m.material_remap.insert(name.clone(), "mats/helmet.material".into());
        m.texture_remap.insert(texture_remap_key(&name, "normal"), "tex/n.png".into());
        save_asset_meta(&model, &meta).unwrap();
        let (scene, report) = load_model_cached(&model, false).unwrap();
        assert_eq!(scene.identity.material_remap[0].as_deref(), Some("mats/helmet.material"));
        assert_eq!(scene.identity.texture_remap[0][1].as_deref(), Some("tex/n.png"));
        assert_eq!(scene.identity.texture_remap[0][0], None);
        // Editing only the remap does not re-run the importer.
        assert_eq!(report.outcome, import_cache::CacheOutcome::Hit);
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct ScaleUp;
    impl import_hooks::ModelImportHook for ScaleUp {
        fn name(&self) -> &str {
            "scale-up"
        }
        fn version(&self) -> u32 {
            1
        }
        fn preprocess(&mut self, _: &import_hooks::ImportContext, s: &mut ModelImport) {
            s.scale = 2.0;
        }
    }

    struct Quiet;
    impl import_hooks::ModelImportHook for Quiet {
        fn name(&self) -> &str {
            "quiet"
        }
        fn version(&self) -> u32 {
            1
        }
    }

    #[test]
    fn hooks_run_inside_the_importer_and_no_op_hooks_change_nothing() {
        let Some((dir, model)) = project_copy("hooks", "models/DamagedHelmet.glb") else { return };
        let settings = ModelImport::default();
        let encode = |s: &Scene| import_cache::encode(s, &model, &[], "");
        let none = import_hooks::HookSet::default();
        let plain = import_model_artifact(&model, &settings, &none).unwrap();
        // A hook that overrides nothing: byte-identical artifact.
        let quiet = import_hooks::HookSet::new("t", vec![Box::new(Quiet)]);
        assert_eq!(encode(&import_model_artifact(&model, &settings, &quiet).unwrap()), encode(&plain));
        // preprocess reaches the format reader: the geometry doubles.
        let up = import_hooks::HookSet::new("t", vec![Box::new(ScaleUp)]);
        let big = import_model_artifact(&model, &settings, &up).unwrap();
        let far = |s: &Scene| s.meshes[0].vertices.iter().map(|v| v.position[0].abs()).fold(0.0f32, f32::max);
        assert!((far(&big) - far(&plain) * 2.0).abs() < 1e-3, "{} vs {}", far(&big), far(&plain));
        // And the cache stores the hooked artifact under its own key, then reuses it.
        let (a, ra) = import_cache::load_cached_with(&model, &settings, true, &up, || import_model_artifact(&model, &settings, &up)).unwrap();
        let (b, rb) = import_cache::load_cached_with(&model, &settings, false, &up, || panic!("must hit")).unwrap();
        assert_eq!(rb.outcome, import_cache::CacheOutcome::Hit);
        assert_ne!(ra.key, import_cache::cache_key(&model, &settings).unwrap());
        assert_eq!(encode(&a), encode(&b));
        assert!((far(&b) - far(&plain) * 2.0).abs() < 1e-3);
        // The default gate is closed, so the public loader ignores installed hooks.
        assert!(!import_hooks::hooks_apply(&model));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gltf_honours_the_model_import_settings() {
        let Some(path) = sample("models/DamagedHelmet.glb") else { return };
        let base = load_gltf_with(&path, &ModelImport::default()).unwrap();
        // Scale reaches vertices and node transforms.
        let small = load_gltf_with(&path, &ModelImport { scale: 0.01, ..Default::default() }).unwrap();
        assert!((small.meshes[0].vertices[0].position[0] - base.meshes[0].vertices[0].position[0] * 0.01).abs() < 1e-5
            || small.meshes[0].vertices.len() == base.meshes[0].vertices.len());
        let far = |s: &Scene| s.meshes[0].vertices.iter().map(|v| v.position[0].abs()).fold(0.0f32, f32::max);
        assert!((far(&small) - far(&base) * 0.01).abs() < 1e-4, "{} vs {}", far(&small), far(&base));
        // Flip UV (V).
        let flipped = load_gltf_with(&path, &ModelImport { flip_uv: true, ..Default::default() }).unwrap();
        let (a, b) = (&base.meshes[0].vertices, &flipped.meshes[0].vertices);
        assert_eq!(a.len(), b.len());
        let mut checked = 0;
        for (x, y) in a.iter().zip(b).take(50) {
            assert!((x.uv[1] + y.uv[1] - 1.0).abs() < 1e-5);
            assert_eq!(x.uv[0], y.uv[0]);
            checked += 1;
        }
        assert!(checked > 0);
        // Import materials off strips textures.
        assert!(base.materials.iter().any(|m| m.albedo.is_some()));
        let bare = load_gltf_with(&path, &ModelImport { import_materials: false, ..Default::default() }).unwrap();
        assert!(bare.materials.iter().all(|m| m.albedo.is_none() && m.normal.is_none()));
        // Weld / optimise: the unwelded mesh has one vertex per index and is never smaller.
        let raw = load_gltf_with(&path, &ModelImport { weld_vertices: false, optimize_mesh: false, ..Default::default() }).unwrap();
        assert!(raw.meshes.iter().all(|m| m.vertices.len() == m.indices.len()));
        assert!(raw.meshes[0].vertices.len() >= base.meshes[0].vertices.len());
        assert_eq!(raw.meshes[0].indices.len(), base.meshes[0].indices.len(), "same triangles");
        // Recalculated normals are unit length.
        let recalc = load_gltf_with(&path, &ModelImport { recalculate_normals: true, ..Default::default() }).unwrap();
        assert!(recalc.meshes[0].vertices.iter().take(200).all(|v| (glam::Vec3::from(v.normal).length() - 1.0).abs() < 1e-3));
    }

    #[test]
    fn gltf_blendshapes_import_toggle() {
        // Any sample with morph targets: find one among the avatars.
        let Some(dir) = sample("models/avatars") else { return };
        let Some(model) = std::fs::read_dir(&dir).ok().into_iter().flatten().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "vrm" || e == "glb")) else { return };
        let with = load_gltf_with(&model, &ModelImport::default()).unwrap();
        if with.morphs.iter().all(Vec::is_empty) {
            return;
        }
        let without = load_gltf_with(&model, &ModelImport { import_blend_shapes: false, ..Default::default() }).unwrap();
        assert!(without.morphs.iter().all(Vec::is_empty));
        assert_eq!(with.meshes.len(), without.meshes.len());
    }
}
