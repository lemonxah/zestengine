//! Per-asset `.meta` sidecar files (Unity/Godot-style). Each imported asset
//! `foo.fbx` gets a companion `foo.fbx.meta` (RON) holding a stable asset
//! GUID (so references survive renames/moves) plus that asset's importer
//! settings. Living next to the asset means the data travels with it in
//! version control and scales to thousands of assets without a central hot file.
//!
//! `#[serde(default)]` everywhere so old `.meta` files keep loading as the
//! importer settings grow.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use zestengine_core::ObjectId;
use serde::{Deserialize, Serialize};

/// Extension appended to the full asset filename: `mesh.fbx` -> `mesh.fbx.meta`.
pub const META_EXT: &str = "meta";

/// One asset's sidecar metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AssetMeta {
    /// Stable, path-independent asset identity. Scenes/materials can reference an
    /// asset by this id so moving or renaming the file doesn't break links.
    pub id: ObjectId,
    /// Per-importer settings for this asset's type.
    pub importer: ImporterSettings,
}

impl Default for AssetMeta {
    fn default() -> Self {
        Self {
            id: ObjectId::new(),
            importer: ImporterSettings::Generic,
        }
    }
}

impl AssetMeta {
    /// A fresh meta whose importer settings default to the right type for the
    /// asset's extension (models get model-import defaults, etc.).
    pub fn new_for(asset: &Path) -> Self {
        Self {
            id: ObjectId::new(),
            importer: ImporterSettings::for_extension(asset),
        }
    }
}

/// Settings keyed by asset kind. Extend with new variants as importers gain
/// options (textures: sRGB/mips; audio: streaming; etc.).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[derive(Default)]
pub enum ImporterSettings {
    #[default]
    Generic,
    Model(ModelImport),
    Texture(crate::texture_import::TextureImport),
}


impl ImporterSettings {
    /// Default settings variant for an asset path's extension.
    pub fn for_extension(asset: &Path) -> Self {
        match asset
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("fbx" | "glb" | "gltf" | "vrm" | "obj") => Self::Model(ModelImport::default()),
            Some("png" | "jpg" | "jpeg" | "tga" | "bmp" | "exr" | "hdr" | "webp" | "gif") => {
                Self::Texture(crate::texture_import::TextureImport::default())
            }
            _ => Self::Generic,
        }
    }

    /// The model-import settings if this is a model asset.
    pub fn as_model(&self) -> Option<&ModelImport> {
        match self {
            Self::Model(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_model_mut(&mut self) -> Option<&mut ModelImport> {
        match self {
            Self::Model(m) => Some(m),
            _ => None,
        }
    }

    /// The texture-import settings if this is a texture asset.
    pub fn as_texture(&self) -> Option<&crate::texture_import::TextureImport> {
        match self {
            Self::Texture(t) => Some(t),
            _ => None,
        }
    }
    pub fn as_texture_mut(&mut self) -> Option<&mut crate::texture_import::TextureImport> {
        match self {
            Self::Texture(t) => Some(t),
            _ => None,
        }
    }
}

/// Index width of imported meshes (Unity's Index Format). Auto and 32 bit keep a
/// mesh whole; 16 bit splits a mesh with more than 65535 vertices into meshes
/// that fit. The renderer uploads 16 bit indices for every mesh that fits when
/// ray tracing is off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexFormat {
    /// 16-bit when the mesh has at most 65535 vertices, else 32-bit.
    #[default]
    Auto,
    UInt16,
    UInt32,
}

impl IndexFormat {
    /// The width in bits a mesh of `vertex_count` vertices has. A forced 16-bit
    /// format splits a mesh that does not fit at import (so the pieces are 16
    /// bit); here such a mesh, as it is before splitting, reports 32.
    pub fn resolve(self, vertex_count: usize) -> u32 {
        match self {
            Self::UInt32 => 32,
            Self::Auto | Self::UInt16 => {
                if vertex_count <= 65535 {
                    16
                } else {
                    32
                }
            }
        }
    }
}

/// Where a mesh's tangents come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TangentMode {
    /// Use the file's tangents; calculate them when the file has none.
    #[default]
    Import,
    /// Always calculate, ignoring the file's.
    Calculate,
    /// Leave the constant default tangent (no normal mapping).
    None,
    /// The accumulate-and-orthogonalise tangents imports used before MikkTSpace
    /// (kept so a model baked against them can be matched).
    Legacy,
}

/// Where a mesh's normals come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalsMode {
    /// Use the file's normals; calculate them when the file has none.
    #[default]
    Import,
    /// Always calculate, with the smoothing angle and weighting below.
    Calculate,
    /// No normals: every vertex gets the constant +Z (unlit uses).
    None,
}

/// How the faces around a vertex are weighted when normals are calculated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalWeighting {
    /// By the angle the face makes at the vertex.
    #[default]
    Angle,
    /// By the face's area.
    Area,
    /// Area times angle.
    AreaAndAngle,
    /// Every face counts the same.
    Unweighted,
}

/// Lossy vertex quantisation applied at import (Unity's Mesh Compression). The
/// bits are kept per position axis (on the mesh's bounds), per UV axis (on the
/// 0..1 range around the mesh's UV bounds) and per normal / tangent component.
/// Welding runs after it, so vertices that quantise to the same values merge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshCompression {
    #[default]
    Off,
    /// 16 bit positions and UVs, 12 bit normals.
    Low,
    /// 14 bit positions, 12 bit UVs, 10 bit normals.
    Medium,
    /// 12 bit positions, 10 bit UVs, 8 bit normals.
    High,
}

impl MeshCompression {
    /// `(position, uv, normal)` bits, or None when off.
    pub fn bits(self) -> Option<(u32, u32, u32)> {
        match self {
            Self::Off => None,
            Self::Low => Some((16, 16, 12)),
            Self::Medium => Some((14, 12, 10)),
            Self::High => Some((12, 10, 8)),
        }
    }
}

/// Which UV sets are kept: bit 0 = UV0, bit 1 = UV1.
pub const UV_ALL: u8 = 0b11;

/// How `Search and Remap` names the `.material` it looks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialNaming {
    /// `<material>.material`.
    #[default]
    BasedOnMaterialName,
    /// `<model>-<material>.material`.
    BasedOnModelNameAndMaterialName,
    /// A `.material` whose textures have the same names as the ones the source
    /// material uses (same file stem, or the same base name / a long shared
    /// prefix: `wood_albedo` and `wood_normal` both belong to `wood`).
    BasedOnTextureName,
}

/// How far `Search and Remap` looks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialSearch {
    /// The model's folder only.
    Local,
    /// The model's folder, then each parent up to the project root.
    #[default]
    RecursiveUp,
    /// Anywhere in the project.
    Everywhere,
}

/// The slots of an imported material a texture can be remapped for.
pub const TEXTURE_REMAP_SLOTS: [&str; 4] = ["albedo", "normal", "orm", "emission"];

/// Key of a texture-remap entry: `<material>/<slot>`.
pub fn texture_remap_key(material: &str, slot: &str) -> String {
    format!("{material}/{slot}")
}

/// Import options shared by mesh formats (FBX / glTF / OBJ).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelImport {
    /// Uniform scale applied to imported geometry (e.g. 0.01 for cm→m sources).
    pub scale: f32,
    /// Import the file's materials (off = assign the default material).
    pub import_materials: bool,
    /// Flip the V texture coordinate (DirectX vs OpenGL convention mismatch).
    pub flip_uv: bool,
    /// Recompute smooth normals instead of using the file's.
    pub recalculate_normals: bool,
    /// Crease angle (degrees) for recomputed normals.
    pub smoothing_angle: f32,
    /// Where normals come from. `recalculate_normals` is the older form of
    /// `Calculate` and is migrated into this on load.
    pub normals: NormalsMode,
    /// How faces are weighted when normals are calculated.
    pub normal_weighting: NormalWeighting,
    /// Honour the file's per-node visibility (FBX): hidden nodes are placed
    /// disabled.
    pub import_visibility: bool,
    /// UV sets kept ([`UV_ALL`]): a cleared set is zeroed (UV1 falls back to UV0).
    pub uv_mask: u8,
    /// Vertex quantisation.
    pub mesh_compression: MeshCompression,
    /// Give every placed part a Mesh Collider.
    pub generate_colliders: bool,
    /// Keep the CPU copy of the meshes (readable at runtime). Meshes whose
    /// data nothing else needs can drop it after upload when this is off.
    pub read_write: bool,
    /// Version of the sub-asset id scheme the `id_table` was written with
    /// (see `sub_asset.rs`). Old metas without it load as the current scheme.
    pub id_scheme: u32,
    /// Recorded `(kind, name) -> id` pairs for this model's meshes, so scenes can
    /// reference a mesh by id and renames made through the importer keep it.
    pub id_table: crate::sub_asset::IdTable,
    /// Rig description (animation type, human bone map, reference pose),
    /// computed once and read by the runtime instead of re-mapping bone names.
    pub rig: crate::human_rig::RigImport,
    /// The `scale` the rig's reference pose translations are in (0 = not
    /// recorded yet: the pose is taken to be in the current scale). Changing
    /// `scale` rescales the pose by the ratio ([`Self::sync_rig_pose_scale`]).
    pub rig_pose_scale: f32,
    /// Bake the file's axis conversion into geometry and animation instead of
    /// leaving it as a rotation on the root (ufbx `ModifyGeometry`). FBX only.
    pub bake_axis_conversion: bool,
    /// Import blendshapes (morph targets).
    pub import_blend_shapes: bool,
    /// Skip the hidden full-mesh copies some exporters leave in the file as blendshape
    /// targets (hidden nodes named like the shapes): the shapes come from the deformer, the
    /// copies would only add objects and triangles. FBX only.
    pub skip_blend_shape_targets: bool,
    /// Import the file's cameras (placed as Camera objects). Off by default: a
    /// model's stray camera must not become the scene's main camera.
    pub import_cameras: bool,
    /// Import the file's lights (placed as Light objects).
    pub import_lights: bool,
    /// Merge corners that share a position and every attribute into one vertex.
    pub weld_vertices: bool,
    /// Keep quads through the mesh passes (they are emitted as consecutive
    /// triangle pairs; there is no quad consumer yet).
    pub keep_quads: bool,
    /// Index width of the imported meshes.
    pub index_format: IndexFormat,
    /// Reorder triangles and vertices for the GPU's vertex cache.
    pub optimize_mesh: bool,
    /// Where tangents come from.
    pub tangents: TangentMode,
    /// Unwrap a lightmap UV set (UV1) for meshes that have none.
    pub generate_lightmap_uvs: bool,
    /// Names for `Search and Remap`.
    pub material_naming: MaterialNaming,
    /// Scope for `Search and Remap`.
    pub material_search: MaterialSearch,
    /// Material names the last import saw (the rows of the remap table), so
    /// the inspector can list them without parsing the model.
    pub source_materials: Vec<String>,
    /// Remapped materials: source material name -> project-relative `.material`.
    /// Applied after the import (not part of the cached artifact).
    pub material_remap: std::collections::BTreeMap<String, String>,
    /// Remapped textures: `<material>/<slot>` -> project-relative image.
    pub texture_remap: std::collections::BTreeMap<String, String>,
    /// Where Extract Textures writes: a project-relative folder, empty = the
    /// `Textures` folder next to the model. Remembered per model.
    pub extract_textures_dir: String,
    /// Where Extract Materials writes: a project-relative folder, empty = the
    /// `Materials` folder next to the model.
    pub extract_materials_dir: String,
    /// Generate screen-size LODs for this model (GEO-2). Off = the model always
    /// draws its full mesh. Applied at placement, so not part of the cached artifact.
    #[serde(default = "default_auto_lod")]
    pub auto_lod: bool,
    /// The Animation tab: which clips are made and how their curves are thinned.
    pub animation: crate::anim_import::AnimImport,
    /// Import BlendShapes options (normals, empty-shape threshold, curves).
    pub blend_shapes: crate::anim_import::BlendShapeImport,
    /// Import Constraints (FBX): the file's parent / position / rotation /
    /// scale / aim constraints become Constraint components on the placed parts.
    pub import_constraints: bool,
    /// Skin Weights: the most bones that may influence a vertex (1..=4, the
    /// vertex format's limit). Vertices over the limit keep their strongest
    /// influences, renormalised, and the import records a warning.
    pub max_bones_per_vertex: u8,
    /// Optimize Game Objects: bones stay inside the skeleton asset and are not
    /// placed as objects (they never are in this engine's model placement; the
    /// flag is recorded and the exposed list below is honoured by the
    /// transforms that do get placed).
    pub optimize_game_objects: bool,
    /// Extra Expose Transforms: bone or node names that stay addressable
    /// objects under Optimize Game Objects.
    pub extra_expose_transforms: Vec<String>,
}

fn default_auto_lod() -> bool {
    true
}

impl Default for ModelImport {
    fn default() -> Self {
        Self {
            scale: 1.0,
            import_materials: true,
            flip_uv: false,
            recalculate_normals: false,
            smoothing_angle: 60.0,
            normals: NormalsMode::Import,
            normal_weighting: NormalWeighting::Angle,
            import_visibility: true,
            skip_blend_shape_targets: true,
            uv_mask: UV_ALL,
            mesh_compression: MeshCompression::Off,
            generate_colliders: false,
            read_write: false,
            id_scheme: crate::sub_asset::ID_SCHEME,
            id_table: Default::default(),
            rig: Default::default(),
            rig_pose_scale: 0.0,
            bake_axis_conversion: false,
            import_blend_shapes: true,
            import_cameras: false,
            import_lights: false,
            weld_vertices: true,
            keep_quads: false,
            index_format: IndexFormat::Auto,
            optimize_mesh: true,
            tangents: TangentMode::Import,
            generate_lightmap_uvs: false,
            material_naming: MaterialNaming::default(),
            material_search: MaterialSearch::default(),
            source_materials: Vec::new(),
            material_remap: Default::default(),
            texture_remap: Default::default(),
            extract_textures_dir: String::new(),
            extract_materials_dir: String::new(),
            auto_lod: true,
            animation: Default::default(),
            blend_shapes: Default::default(),
            import_constraints: false,
            max_bones_per_vertex: 4,
            optimize_game_objects: false,
            extra_expose_transforms: Vec::new(),
        }
    }
}

impl ModelImport {
    /// Largest accepted `scale` magnitude.
    pub const MAX_SCALE: f32 = 1e10;
    /// What a scale of 0 (or a non-finite value) becomes, so geometry never collapses to a point.
    pub const MIN_SCALE: f32 = 0.0001;

    /// Keep the rig's reference pose in step with `scale`: translations are in
    /// the units of the scale the pose was stored at, so a new scale moves them
    /// by `scale / rig_pose_scale` (rotations and scales are unitless). A pose
    /// with no recorded scale is taken to be in the current one. Returns true
    /// if the pose was rescaled or the scale recorded.
    pub fn sync_rig_pose_scale(&mut self) -> bool {
        if !self.scale.is_finite() || self.scale == 0.0 {
            return false;
        }
        if self.rig_pose_scale == 0.0 || !self.rig_pose_scale.is_finite() {
            self.rig_pose_scale = self.scale;
            return true;
        }
        if (self.rig_pose_scale - self.scale).abs() <= f32::EPSILON * self.scale.abs() {
            return false;
        }
        let k = self.scale / self.rig_pose_scale;
        for bone in &mut self.rig.reference_pose {
            for t in &mut bone.translation {
                *t *= k;
            }
        }
        self.rig_pose_scale = self.scale;
        true
    }

    /// Where normals come from after migrating the older toggle.
    pub fn normals_mode(&self) -> NormalsMode {
        if self.normals == NormalsMode::Import && self.recalculate_normals { NormalsMode::Calculate } else { self.normals }
    }

    /// Settings validation, run on every load (the counterpart of Unity's
    /// `CheckConsistency`): clamps `scale` to +-1e10 and turns 0 / NaN into
    /// 0.0001, clamps the crease angle to 0..=180, and repairs the id table
    /// (unknown scheme, id 0, duplicate ids or names). Returns true if
    /// anything was changed.
    pub fn validate(&mut self) -> bool {
        let before = (self.scale.to_bits(), self.smoothing_angle.to_bits(), self.id_scheme, self.id_table.entries.len());
        if !self.scale.is_finite() || self.scale == 0.0 {
            self.scale = Self::MIN_SCALE;
        }
        self.scale = self.scale.clamp(-Self::MAX_SCALE, Self::MAX_SCALE);
        if !self.smoothing_angle.is_finite() {
            self.smoothing_angle = 60.0;
        }
        self.smoothing_angle = self.smoothing_angle.clamp(0.0, 180.0);
        if self.id_scheme == 0 {
            self.id_scheme = crate::sub_asset::ID_SCHEME;
        }
        self.material_remap.retain(|k, v| !k.is_empty() && !v.is_empty());
        self.texture_remap.retain(|k, v| !k.is_empty() && !v.is_empty());
        let mut seen_ids = std::collections::HashSet::new();
        let mut seen_names = std::collections::HashSet::new();
        self.id_table
            .entries
            .retain(|e| !e.id.is_none() && seen_ids.insert(e.id) && seen_names.insert((e.kind, e.name.clone())));
        // Metas from before the Normals setting: "Recalculate Normals" became Calculate.
        if self.recalculate_normals {
            if self.normals == NormalsMode::Import {
                self.normals = NormalsMode::Calculate;
            }
            self.recalculate_normals = false;
        }
        self.uv_mask &= UV_ALL;
        self.max_bones_per_vertex = self.max_bones_per_vertex.clamp(1, 4);
        self.extra_expose_transforms.retain(|n| !n.trim().is_empty());
        self.extra_expose_transforms.sort();
        self.extra_expose_transforms.dedup();
        self.animation.validate();
        if !self.blend_shapes.delta_threshold.is_finite() || self.blend_shapes.delta_threshold < 0.0 {
            self.blend_shapes.delta_threshold = crate::anim_import::BlendShapeImport::default().delta_threshold;
        }
        // The pose follows the scale in memory; it is stored with the next save.
        self.sync_rig_pose_scale();
        before != (self.scale.to_bits(), self.smoothing_angle.to_bits(), self.id_scheme, self.id_table.entries.len())
    }
}

/// The `.meta` path for an asset (`mesh.fbx` -> `mesh.fbx.meta`).
pub fn meta_path(asset: impl AsRef<Path>) -> PathBuf {
    let mut s = asset.as_ref().as_os_str().to_os_string();
    s.push(".");
    s.push(META_EXT);
    PathBuf::from(s)
}

/// Load an asset's `.meta`, or `None` if it doesn't exist yet.
pub fn load_asset_meta(asset: impl AsRef<Path>) -> Result<Option<AssetMeta>> {
    let path = meta_path(&asset);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut meta: AssetMeta = ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if let Some(m) = meta.importer.as_model_mut() {
        m.validate();
    }
    Ok(Some(meta))
}

/// Load an asset's `.meta`, creating + writing a default (with a fresh GUID and
/// type-appropriate importer settings) if it's missing. Mirrors how Unity
/// auto-generates a `.meta` the first time it sees an asset.
pub fn load_or_create_asset_meta(asset: impl AsRef<Path>) -> Result<AssetMeta> {
    if let Some(meta) = load_asset_meta(&asset)? {
        return Ok(meta);
    }
    let meta = AssetMeta::new_for(asset.as_ref());
    save_asset_meta(&asset, &meta)?;
    Ok(meta)
}

/// Write an asset's `.meta` sidecar.
pub fn save_asset_meta(asset: impl AsRef<Path>, meta: &AssetMeta) -> Result<()> {
    let path = meta_path(&asset);
    let text = ron::ser::to_string_pretty(meta, ron::ser::PrettyConfig::default())?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbx_meta_defaults_to_model_import() {
        let meta = AssetMeta::new_for(Path::new("/x/mesh.FBX"));
        assert!(meta.importer.as_model().is_some(), "fbx → model importer");
        assert!(!meta.id.is_nil(), "fresh meta gets a real GUID");
    }

    #[test]
    fn png_meta_defaults_to_unconfigured_texture_import() {
        let meta = AssetMeta::new_for(Path::new("/x/tex.png"));
        let t = meta.importer.as_texture().expect("png → texture importer");
        assert!(!t.configured, "fresh metas keep the legacy slot-derived loading until applied");
        assert!(matches!(AssetMeta::new_for(Path::new("/x/readme.txt")).importer, ImporterSettings::Generic));
    }

    #[test]
    fn meta_path_appends_extension() {
        assert_eq!(meta_path("a/b/mesh.fbx"), PathBuf::from("a/b/mesh.fbx.meta"));
    }

    #[test]
    fn validation_clamps_scale_and_angle() {
        let mut m = ModelImport { scale: 0.0, smoothing_angle: 400.0, ..Default::default() };
        assert!(m.validate());
        assert_eq!(m.scale, ModelImport::MIN_SCALE, "0 becomes 0.0001");
        assert_eq!(m.smoothing_angle, 180.0);
        let mut m = ModelImport { scale: 1e30, ..Default::default() };
        m.validate();
        assert_eq!(m.scale, 1e10);
        let mut m = ModelImport { scale: -1e30, ..Default::default() };
        m.validate();
        assert_eq!(m.scale, -1e10, "negative scale is kept, only clamped");
        let mut m = ModelImport { scale: f32::NAN, smoothing_angle: f32::NAN, ..Default::default() };
        m.validate();
        assert_eq!((m.scale, m.smoothing_angle), (ModelImport::MIN_SCALE, 60.0));
        let mut ok = ModelImport { scale: 0.01, ..Default::default() };
        assert!(!ok.validate(), "valid settings are left alone");
    }

    #[test]
    fn changing_the_scale_rescales_the_rig_reference_pose() {
        let bone = |name: &str, t: [f32; 3]| crate::human_rig::PoseBone { name: name.into(), translation: t, ..Default::default() };
        let mut m = ModelImport { scale: 1.0, ..Default::default() };
        m.rig.reference_pose = vec![bone("Hips", [0.0, 100.0, 0.0]), bone("Spine", [0.0, 20.0, 5.0])];
        // First look: the pose is taken to be in the current scale.
        assert!(m.sync_rig_pose_scale());
        assert_eq!(m.rig_pose_scale, 1.0);
        assert_eq!(m.rig.reference_pose[0].translation, [0.0, 100.0, 0.0]);
        assert!(!m.sync_rig_pose_scale(), "nothing changed");
        // Centimetres to metres.
        m.scale = 0.01;
        assert!(m.sync_rig_pose_scale());
        let t = m.rig.reference_pose[1].translation;
        assert!((t[1] - 0.2).abs() < 1e-6 && (t[2] - 0.05).abs() < 1e-6, "{t:?}");
        assert_eq!(m.rig.reference_pose[0].rotation, [0.0, 0.0, 0.0, 1.0], "rotations are untouched");
        // And back, and through a save.
        let text = ron::to_string(&m).unwrap();
        let mut back: ModelImport = ron::from_str(&text).unwrap();
        assert!(!back.sync_rig_pose_scale(), "the recorded scale survives the meta");
        back.scale = 1.0;
        back.sync_rig_pose_scale();
        assert!((back.rig.reference_pose[0].translation[1] - 100.0).abs() < 1e-3);
        // An old meta (no recorded scale) loads as it is, then is stamped.
        let mut old: ModelImport = ron::from_str("(scale: 0.01, rig: (reference_pose: [(name: \"Hips\", translation: (0.0, 1.0, 0.0))]))").unwrap();
        assert_eq!(old.rig_pose_scale, 0.0);
        old.validate();
        assert_eq!((old.rig_pose_scale, old.rig.reference_pose[0].translation[1]), (0.01, 1.0));
    }

    #[test]
    fn validation_repairs_the_id_table() {
        use crate::sub_asset::{IdEntry, IdTable, SubAssetId, SubAssetKind};
        let e = |name: &str, id: u64| IdEntry { kind: SubAssetKind::Mesh, name: name.into(), id: SubAssetId(id) };
        let mut m = ModelImport {
            id_scheme: 0,
            id_table: IdTable { entries: vec![e("a", 1), e("b", 1), e("c", 0), e("a", 2), e("d", 3)] },
            ..Default::default()
        };
        assert!(m.validate());
        assert_eq!(m.id_scheme, crate::sub_asset::ID_SCHEME);
        let names: Vec<&str> = m.id_table.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["a", "d"], "duplicate ids/names and id 0 dropped");
    }

    #[test]
    fn old_meta_without_id_fields_loads_with_defaults_and_validates() {
        let dir = std::env::temp_dir().join(format!("zest_meta_old_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let asset = dir.join("m.fbx");
        let text = "(id:\"00000000-0000-4000-8000-000000000001\",importer:Model((scale:0.0,import_materials:true,flip_uv:false)))";
        std::fs::write(meta_path(&asset), text).unwrap();
        let meta = load_asset_meta(&asset).unwrap().unwrap();
        let m = meta.importer.as_model().unwrap();
        assert_eq!(m.scale, ModelImport::MIN_SCALE, "validated on load");
        assert_eq!(m.id_scheme, crate::sub_asset::ID_SCHEME);
        assert!(m.id_table.entries.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trips_ron() {
        let mut meta = AssetMeta::new_for(Path::new("m.fbx"));
        if let Some(m) = meta.importer.as_model_mut() {
            m.scale = 0.01;
            m.flip_uv = true;
        }
        let s = ron::ser::to_string_pretty(&meta, Default::default()).unwrap();
        let back: AssetMeta = ron::from_str(&s).unwrap();
        assert_eq!(back.id, meta.id);
        assert_eq!(back.importer.as_model().unwrap().scale, 0.01);
    }

    #[test]
    fn index_format_resolves_by_vertex_count() {
        assert_eq!(IndexFormat::Auto.resolve(65535), 16);
        assert_eq!(IndexFormat::Auto.resolve(65536), 32);
        assert_eq!(IndexFormat::UInt32.resolve(10), 32);
        assert_eq!(IndexFormat::UInt16.resolve(70000), 32, "a mesh that does not fit keeps 32 bit");
    }

    #[test]
    fn new_settings_round_trip_and_old_values_default() {
        let m = ModelImport {
            index_format: IndexFormat::UInt16,
            tangents: TangentMode::Calculate,
            material_naming: MaterialNaming::BasedOnModelNameAndMaterialName,
            material_search: MaterialSearch::Everywhere,
            keep_quads: true,
            bake_axis_conversion: true,
            ..Default::default()
        };
        let text = ron::ser::to_string(&m).unwrap();
        let back: ModelImport = ron::from_str(&text).unwrap();
        assert_eq!((back.index_format, back.tangents, back.material_search), (IndexFormat::UInt16, TangentMode::Calculate, MaterialSearch::Everywhere));
        assert!(back.keep_quads && back.bake_axis_conversion);
        let d = ModelImport::default();
        assert!(d.weld_vertices && d.optimize_mesh && d.import_blend_shapes && !d.import_cameras && !d.import_lights && !d.keep_quads);
    }
}
