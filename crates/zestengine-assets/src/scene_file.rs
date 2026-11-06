//! `.scene` asset files (RON): a list of placed objects with sources and
//! material references, so multiple scenes can be saved and reloaded.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use zestengine_render::{MaterialFeatures, MaterialParams};
use serde::{Deserialize, Serialize};

use crate::material_file::MaterialTextures;
use crate::sub_asset::{SubAssetId, SubAssetKind};
use zestengine_core::ObjectId;

pub const SCENE_EXTENSION: &str = "scene";

/// Where an object's mesh came from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ObjectSource {
    /// A model file in the project; `mesh` is the flattened primitive index
    /// (slot 0) produced by the importer (stable for an unchanged file).
    /// `extra_meshes` holds the flattened indices of any additional material
    /// slots on the same object (empty for single-material meshes).
    Model {
        path: String,
        mesh: usize,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extra_meshes: Vec<usize>,
        /// The model's `.meta` GUID. With `mesh_id` this is the real reference:
        /// on load the mesh is looked up by id, and `mesh` / `extra_meshes` are
        /// only the fallback for an id that can't be found. Absent in scenes
        /// saved before stable ids (migrated on load, see
        /// [`SceneFile::resolve_model_refs`]).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<ObjectId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mesh_id: Option<SubAssetId>,
        /// Ids of `extra_meshes`, in the same order.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extra_mesh_ids: Vec<SubAssetId>,
    },
    /// A built-in test-scene mesh (sphere/cube/plane by index).
    Builtin { mesh: usize },
    /// A generated primitive shape.
    Primitive { shape: PrimitiveShape },
    /// A transform-only grouping node.
    Empty,
    /// A camera placement.
    Camera,
    /// A light placement. The light's kind and settings live in its
    /// `Light` component; this only marks the object as a light (icon, default
    /// component, gathering).
    Light,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveShape {
    Cube,
    Sphere,
    Capsule,
    Plane,
    /// Open-top box with walls (a bucket / tub).
    Bucket,
}

impl PrimitiveShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::Sphere => "Sphere",
            Self::Capsule => "Capsule",
            Self::Plane => "Plane",
            Self::Bucket => "Bucket",
        }
    }
}

fn default_shader() -> String {
    "standard".into()
}

/// Which material an object uses. The inline snapshot is large, but a scene
/// holds few of these and boxing it would change every match site.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum MaterialRef {
    /// A `.material` asset file (project-relative path).
    File(String),
    /// The object's own material exactly as its model (or primitive) brings
    /// it: nothing stored, so unedited model materials cost no scene space.
    Model,
    /// Snapshot of parameters applied over whatever textures the object's
    /// imported material carried.
    Inline {
        params: MaterialParams,
        features: MaterialFeatures,
        /// "standard" or a project-relative `.frag` path.
        #[serde(default = "default_shader")]
        shader: String,
        /// Custom-shader property values by name.
        #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
        custom: std::collections::BTreeMap<String, Vec<f32>>,
        /// Draw-order priority (Unity render queue); None = derive from alpha.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        render_queue: Option<i32>,
        /// User-assigned texture slots (project-relative). Empty = none / use
        /// the object's import-embedded textures.
        #[serde(default, skip_serializing_if = "MaterialTextures::is_empty")]
        textures: MaterialTextures,
    },
}

/// One serialized component: registry name + its RON-encoded state, and
/// whether it is switched on (omitted when it is).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentData {
    pub kind: String,
    pub data: String,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub enabled: bool,
}

fn is_true(v: &bool) -> bool {
    *v
}

impl ComponentData {
    /// A switched-on component.
    pub fn new(kind: impl Into<String>, data: impl Into<String>) -> Self {
        Self { kind: kind.into(), data: data.into(), enabled: true }
    }

    /// `c` saved, switched off when it is wrapped in [`zestengine_core::Disabled`].
    pub fn of(c: &dyn zestengine_core::Component) -> Self {
        Self { kind: c.type_name().to_owned(), data: c.save(), enabled: zestengine_core::component_enabled(c) }
    }

    /// Rebuilt through `registry`, switched off when saved off.
    pub fn load(&self, registry: &zestengine_core::ComponentRegistry) -> Option<Box<dyn zestengine_core::Component>> {
        let c = registry.load(&self.kind, &self.data)?;
        Some(if self.enabled { c } else { Box::new(zestengine_core::Disabled(c)) })
    }
}

fn default_true() -> bool {
    true
}

fn default_lightmap_scale() -> f32 {
    1.0
}

fn is_default_lightmap_scale(v: &f32) -> bool {
    (*v - 1.0).abs() < f32::EPSILON
}

fn default_material_ref() -> MaterialRef {
    MaterialRef::Model
}

impl MaterialRef {
    /// The default: the model's own material (nothing stored).
    pub fn is_model(&self) -> bool {
        matches!(self, MaterialRef::Model)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneEntry {
    /// Stable object id (UUID string). Empty in legacy scenes; the engine
    /// assigns a fresh one on load.
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub source: ObjectSource,
    /// Whether the object renders / its light contributes. Disabled objects
    /// stay in the scene but are skipped at draw time.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Marks the object as non-moving so the lighting bake includes it as a
    /// lightmapped surface + ray-trace occluder ("Contribute GI" in the editor).
    #[serde(default)]
    pub static_geometry: bool,
    /// Per-object lightmap-resolution multiplier (Unity's "Scale In Lightmap").
    /// 1.0 = the scene texel density; raise for a sharper surface, lower to save
    /// texels.
    #[serde(default = "default_lightmap_scale", skip_serializing_if = "is_default_lightmap_scale")]
    pub lightmap_scale: f32,
    /// "Cast Shadows" (Unity's MeshRenderer setting). Omitted when On, so
    /// older scenes load as On.
    #[serde(default, skip_serializing_if = "zestengine_core::ShadowCasting::is_on")]
    pub cast_shadows: zestengine_core::ShadowCasting,
    /// Object layer index (0..32, Unity-style): camera culling mask (rendering)
    /// + layer-collision matrix (physics). 0 = "Default".
    #[serde(default)]
    pub layer: u8,
    /// LEGACY (read-only): slot 0 of a renderer before the `Mesh Renderer` component existed.
    /// Loading moves it into the component ([`SceneFile::migrate_renderers`]) and clears it, so
    /// it is only ever non-default on an entry built in memory by old-style code.
    #[serde(default = "default_material_ref", skip_serializing_if = "MaterialRef::is_model")]
    pub material: MaterialRef,
    /// LEGACY (read-only): slots 1.. before the component existed (see `material`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_materials: Vec<MaterialRef>,
    /// Runtime-only: the entry's renderer is described by its components (a `Mesh Renderer`, or
    /// none on purpose). False on an entry built in memory in the old loose-field style: the
    /// engine then derives a renderer from the legacy fields. Never serialized.
    #[serde(skip)]
    pub renderer_split: bool,
    /// Index of the parent entry in this file, if any. Transforms are local
    /// to the parent.
    #[serde(default)]
    pub parent: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<ComponentData>,
    pub translation: [f32; 3],
    /// Quaternion xyzw.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    /// On a model's root: the model whose untouched parts aren't stored (they
    /// are rebuilt from the model file on load).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelInstanceRef>,
    /// On a stored model part: which of the model's parts it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_part: Option<PartRef>,
    /// On a prefab instance's root: the asset it comes from plus what the
    /// instance changed (the rest is rebuilt from the asset on load).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefab: Option<crate::prefab_link::PrefabLink>,
    /// On an object added to a prefab instance whose parent is one of the
    /// instance's own objects: that parent's id inside the prefab (`parent`
    /// then points at the instance root).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefab_parent: Option<String>,
    /// Where the object exists on a networked game (NET-3). Omitted when
    /// `Both`, so old scenes load unchanged.
    #[serde(default, skip_serializing_if = "zestengine_core::NetRole::is_both")]
    pub net_role: zestengine_core::NetRole,
    /// Per-component role overrides, keyed like prefab component keys (`Kind`,
    /// `Kind#1` for a second component of a kind). A component without an entry
    /// takes its type's default role.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub component_roles: std::collections::BTreeMap<String, zestengine_core::NetRole>,
}

/// A model placed in a scene, stored as a reference plus what changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelInstanceRef {
    pub path: String,
    /// The model's `.meta` GUID (informational; stamped on save).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ObjectId>,
    /// Parts deleted from this instance. Written as part ids; scenes saved
    /// before ids hold indices into the model's parts, which load migrates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<PartRef>,
}

/// Names one part of a model: by its stable id, or (scenes saved before ids
/// existed) by its position in the model's part list. Positions are read, never
/// written; [`SceneFile::expand_models`] and [`SceneFile::compact_models`]
/// migrate them to ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PartRef {
    Index(u32),
    Id(SubAssetId),
}

impl PartRef {
    /// The part's position in `parts` as the model imports now.
    pub fn resolve(self, parts: &[ModelPart]) -> Option<usize> {
        match self {
            PartRef::Index(k) => ((k as usize) < parts.len()).then_some(k as usize),
            PartRef::Id(id) => parts.iter().position(|p| p.part_id == id),
        }
    }
}

impl Serialize for PartRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            PartRef::Index(k) => s.serialize_u32(*k),
            PartRef::Id(id) => s.serialize_str(&format!("p{:016x}", id.0)),
        }
    }
}

impl<'de> Deserialize<'de> for PartRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = PartRef;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a part id (\"p<hex>\") or a part index")
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<PartRef, E> {
                u32::try_from(v).map(PartRef::Index).map_err(E::custom)
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<PartRef, E> {
                u32::try_from(v).map(PartRef::Index).map_err(E::custom)
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<PartRef, E> {
                v.strip_prefix('p')
                    .and_then(|h| u64::from_str_radix(h, 16).ok())
                    .map(|h| PartRef::Id(SubAssetId(h)))
                    .ok_or_else(|| E::custom(format!("not a part id: {v}")))
            }
        }
        d.deserialize_any(V)
    }
}

/// A model's mesh index <-> id pairs, from its parts.
struct MeshIds {
    model: Option<ObjectId>,
    by_index: HashMap<usize, SubAssetId>,
    by_id: HashMap<SubAssetId, usize>,
}

impl MeshIds {
    fn of(parts: &[ModelPart]) -> Self {
        let mut me = Self { model: parts.iter().find_map(|p| p.model), by_index: HashMap::new(), by_id: HashMap::new() };
        for p in parts {
            let all = std::iter::once((p.mesh, p.mesh_id)).chain(p.extra_meshes.iter().copied().zip(p.extra_mesh_ids.iter().copied()));
            for (index, id) in all {
                if !id.is_none() {
                    me.by_index.insert(index, id);
                    me.by_id.insert(id, index);
                }
            }
        }
        me
    }
    fn id_of(&self, index: usize) -> Option<SubAssetId> {
        self.by_index.get(&index).copied()
    }
    fn index_of(&self, id: SubAssetId) -> Option<usize> {
        self.by_id.get(&id).copied()
    }
}

/// One part (instance) of a model as imported: what a pristine scene object
/// for it looks like.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPart {
    pub name: String,
    pub mesh: usize,
    pub extra_meshes: Vec<usize>,
    /// The model's `.meta` GUID, if it has one.
    pub model: Option<ObjectId>,
    /// Stable ids of `mesh` / `extra_meshes` ([`SubAssetId::NONE`] if unassigned).
    pub mesh_id: SubAssetId,
    pub extra_mesh_ids: Vec<SubAssetId>,
    /// Stable id of the part: its primary mesh's id and which occurrence of
    /// that mesh it is (so reordering parts, or adding others, keeps it).
    pub part_id: SubAssetId,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl ModelPart {
    /// The parts of a loaded model, in the order its objects are created.
    pub fn of(model: &crate::Scene) -> Vec<ModelPart> {
        let mut seen: HashMap<String, u32> = HashMap::new();
        model
            .instances
            .iter()
            .filter_map(|inst| {
                let slot0 = inst.slots.first()?;
                let (s, r, t) = inst.transform.to_scale_rotation_translation();
                let mesh_id = model.identity.mesh_ids.get(slot0.mesh).copied().unwrap_or(SubAssetId::NONE);
                // Parts of one mesh are told apart by order; a model without
                // mesh ids falls back to the instance name.
                let base = if mesh_id.is_none() { format!("n:{}", inst.name) } else { format!("m:{:016x}", mesh_id.0) };
                let nth = seen.entry(base.clone()).and_modify(|n| *n += 1).or_insert(0);
                let part_id = crate::sub_asset::hashed_id(SubAssetKind::Part, &format!("{base}#{nth}"));
                Some(ModelPart {
                    part_id,
                    name: inst.name.clone(),
                    mesh: slot0.mesh,
                    extra_meshes: inst.slots[1..].iter().map(|x| x.mesh).collect(),
                    model: model.identity.guid,
                    mesh_id,
                    extra_mesh_ids: inst.slots[1..]
                        .iter()
                        .map(|x| model.identity.mesh_ids.get(x.mesh).copied().unwrap_or(SubAssetId::NONE))
                        .collect(),
                    translation: t.to_array(),
                    rotation: r.to_array(),
                    scale: s.to_array(),
                })
            })
            .collect()
    }
}

/// No components, or only a renderer with default settings whose slots all inherit: what an
/// untouched imported part has (and what is not worth storing).
fn untouched_components(components: &[ComponentData]) -> bool {
    match components {
        [] => true,
        [c] if zestengine_core::is_renderer_kind(&c.kind) && c.enabled => {
            ron::from_str::<zestengine_core::MeshRenderer>(&c.data).is_ok_and(|r| r.is_untouched())
        }
        _ => false,
    }
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

/// Stable id of the object for `part` of the model instance whose root has id `root`.
fn part_entry_id(root: &str, part: SubAssetId) -> String {
    let mut h1 = 0xcbf2_9ce4_8422_2325u64;
    let mut h2 = 0x8422_2325_cbf2_9ce4u64;
    for b in root.bytes().chain(part.0.to_le_bytes()) {
        h1 = (h1 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        h2 = (h2 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
    }
    let b = (((h1 as u128) << 64) | h2 as u128).to_be_bytes();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

impl SceneFile {
    /// Drop the editor-only view state (camera, collapsed rows, scene-view
    /// hide / lock). Run on every runtime-bound copy: packages, server builds.
    pub fn strip_editor_view(&mut self) {
        self.editor_camera = None;
        self.collapsed.clear();
        self.scene_hidden.clear();
        self.scene_locked.clear();
    }

    /// A new scene: just the default `Directional Light` (the editor's New
    /// Scene and the file browser's Create > Scene).
    pub fn new_default() -> Self {
        Self { entries: vec![default_directional_light_entry()], ..Default::default() }
    }

    /// Whether an enabled object carries an enabled Directional `Light`.
    pub fn has_directional_light(&self) -> bool {
        self.entries.iter().any(|e| {
            e.enabled
                && e.components.iter().any(|c| {
                    c.enabled
                        && c.kind == "Light"
                        && ron::from_str::<zestengine_core::LightComponent>(&c.data)
                            .is_ok_and(|l| l.kind == zestengine_core::LightKind::Directional)
                })
        })
    }

    /// Legacy-sun migration, run once when a scene is loaded. Scenes saved
    /// before the sun became an object carry `environment.sun_*`; an enabled
    /// sun becomes a `Directional Light` object with the same colour,
    /// intensity and direction (Mixed mode, shadow bias 0.003, radius 0.03: how
    /// the sun was lit and baked), and the environment sun is switched off.
    ///
    /// If the scene already has a directional light object, the explicit object
    /// wins: no light is added and the old sun is dropped. A scene whose sun was
    /// off stays as it was (no light). Idempotent: once the sun is off a second
    /// call changes nothing. Returns true when the file changed, so the caller
    /// can mark the scene dirty and the next save writes the new form.
    pub fn migrate_environment_sun(&mut self) -> bool {
        let env = &mut self.environment;
        if !env.sun_enabled {
            return false;
        }
        let (color, intensity, direction) = (env.sun_color, env.sun_intensity, env.sun_direction);
        env.sun_enabled = false;
        if !self.has_directional_light() {
            self.entries.push(directional_light_entry(color, intensity, direction, zestengine_core::LightMode::Mixed));
        } else {
            tracing::warn!("scene has a directional light object; the legacy environment sun is ignored");
        }
        true
    }

    /// Model paths this file needs, including ones only referenced by a
    /// compacted model root.
    pub fn model_paths(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for e in &self.entries {
            let path = match (&e.source, &e.model) {
                (_, Some(m)) => Some(&m.path),
                (ObjectSource::Model { path, .. }, None) => Some(path),
                _ => None,
            };
            if let Some(p) = path
                && !out.contains(p)
            {
                out.push(p.clone());
            }
        }
        out
    }

    /// Find models that moved or were renamed: a model entry whose file is gone
    /// but whose stored meta GUID matches another `.meta` in the project gets
    /// that file's path (in every entry that used the old path). Returns the
    /// `(old, new)` pairs. Scans the project only when something is missing.
    pub fn relocate_models(&mut self, project_root: &Path) -> Vec<(String, String)> {
        let mut moved: Vec<(String, String)> = Vec::new();
        let mut index: Option<HashMap<ObjectId, String>> = None;
        for e in &self.entries {
            let (path, guid) = match (&e.source, &e.model) {
                (_, Some(m)) => (&m.path, m.model),
                (ObjectSource::Model { path, model, .. }, None) => (path, *model),
                _ => continue,
            };
            let Some(guid) = guid else { continue };
            if project_root.join(path).exists() || moved.iter().any(|m| &m.0 == path) {
                continue;
            }
            let found = index.get_or_insert_with(|| model_guid_index(project_root)).get(&guid).cloned();
            if let Some(new) = found.filter(|n| n != path) {
                moved.push((path.clone(), new));
            }
        }
        for e in &mut self.entries {
            if let ObjectSource::Model { path, .. } = &mut e.source
                && let Some((_, new)) = moved.iter().find(|m| &m.0 == path)
            {
                *path = new.clone();
            }
            if let Some(m) = &mut e.model
                && let Some((_, new)) = moved.iter().find(|x| x.0 == m.path)
            {
                m.path = new.clone();
            }
        }
        moved
    }

    /// Record, on every model object, which model (meta GUID) and which mesh
    /// (stable id) it uses, from the current flattened indices. Run on save so
    /// the file references meshes by identity rather than position.
    pub fn stamp_model_refs(&mut self, parts_of: &dyn Fn(&str) -> Option<Vec<ModelPart>>) {
        let mut cache: HashMap<String, Option<MeshIds>> = HashMap::new();
        for e in &mut self.entries {
            let ObjectSource::Model { path, mesh, extra_meshes, model, mesh_id, extra_mesh_ids } = &mut e.source else { continue };
            let ids = cache.entry(path.clone()).or_insert_with(|| parts_of(path).map(|p| MeshIds::of(&p)));
            let Some(ids) = ids else { continue };
            *model = ids.model;
            *mesh_id = ids.id_of(*mesh);
            let extra: Option<Vec<SubAssetId>> = extra_meshes.iter().map(|m| ids.id_of(*m)).collect();
            *extra_mesh_ids = extra.unwrap_or_default();
        }
    }

    /// Point every model object at its mesh by id. Objects saved before ids
    /// existed (no `mesh_id`) are migrated: the id is taken from their stored
    /// index in the model as it imports now. An id that no longer exists in the
    /// model (mesh deleted or renamed outside the importer) keeps the stored
    /// index as a fallback and logs a warning instead of silently showing the
    /// wrong mesh for a model whose indices shifted.
    pub fn resolve_model_refs(&mut self, parts_of: &dyn Fn(&str) -> Option<Vec<ModelPart>>) {
        let mut cache: HashMap<String, Option<MeshIds>> = HashMap::new();
        for e in &mut self.entries {
            let ObjectSource::Model { path, mesh, extra_meshes, model, mesh_id, extra_mesh_ids } = &mut e.source else { continue };
            let ids = cache.entry(path.clone()).or_insert_with(|| parts_of(path).map(|p| MeshIds::of(&p)));
            let Some(ids) = ids else { continue };
            match *mesh_id {
                Some(id) => match ids.index_of(id) {
                    Some(i) => *mesh = i,
                    None => tracing::warn!("{}: mesh id {} is no longer in {path}; using stored index {mesh}", e.name, id.0),
                },
                None => *mesh_id = ids.id_of(*mesh), // legacy scene: migrate
            }
            if extra_mesh_ids.len() == extra_meshes.len() {
                for (slot, id) in extra_meshes.iter_mut().zip(extra_mesh_ids.iter()) {
                    match ids.index_of(*id) {
                        Some(i) => *slot = i,
                        None => tracing::warn!("{}: extra mesh id {} is no longer in {path}; using stored index {slot}", e.name, id.0),
                    }
                }
            } else {
                let extra: Option<Vec<SubAssetId>> = extra_meshes.iter().map(|m| ids.id_of(*m)).collect();
                *extra_mesh_ids = extra.unwrap_or_default();
            }
            if model.is_none() {
                *model = ids.model;
            } else if ids.model.is_some() && *model != ids.model {
                tracing::warn!("{}: {path} has a different meta GUID than when the scene was saved; matching meshes by id", e.name);
            }
        }
    }

    /// Store each model placed in the scene as its root plus what changed:
    /// parts still exactly as imported (and not referenced by anything) are
    /// dropped, deleted parts are listed, edited ones stay as entries tagged
    /// with their part index. `parts_of` lists a model's parts.
    pub fn compact_models(&mut self, parts_of: &dyn Fn(&str) -> Option<Vec<ModelPart>>) {
        self.stamp_model_refs(parts_of);
        let n = self.entries.len();
        let mut has_children = vec![false; n];
        for e in &self.entries {
            if let Some(p) = e.parent.filter(|p| *p < n) {
                has_children[p] = true;
            }
        }
        let all_data: String = self.entries.iter().flat_map(|e| e.components.iter().map(|c| c.data.as_str())).collect::<Vec<_>>().join(" ");
        let mut drop = vec![false; n];
        for r in 0..n {
            if self.entries[r].parent.is_some() || !matches!(self.entries[r].source, ObjectSource::Empty) || self.entries[r].model.is_some() {
                continue;
            }
            let children: Vec<usize> = (0..n).filter(|&c| self.entries[c].parent == Some(r)).collect();
            let model_children: Vec<usize> = children.iter().copied().filter(|&c| matches!(self.entries[c].source, ObjectSource::Model { .. })).collect();
            let Some(path) = model_children.first().and_then(|&c| match &self.entries[c].source {
                ObjectSource::Model { path, .. } => Some(path.clone()),
                _ => None,
            }) else {
                continue;
            };
            if !model_children.iter().all(|&c| matches!(&self.entries[c].source, ObjectSource::Model { path: p, .. } if *p == path)) {
                continue;
            }
            let Some(parts) = parts_of(&path) else { continue };
            // Match each model child to a distinct part: same meshes, preferring
            // the same name, in order.
            let mut claimed = vec![false; parts.len()];
            let mut assign = Vec::new();
            for &c in &model_children {
                let ObjectSource::Model { mesh, extra_meshes, .. } = &self.entries[c].source else { unreachable!() };
                let fits = |k: usize| !claimed[k] && parts[k].mesh == *mesh && parts[k].extra_meshes == *extra_meshes;
                let name = &self.entries[c].name;
                let k = (0..parts.len()).find(|&k| fits(k) && parts[k].name == *name).or_else(|| (0..parts.len()).find(|&k| fits(k)));
                match k {
                    Some(k) => {
                        claimed[k] = true;
                        assign.push((c, k));
                    }
                    None => break,
                }
            }
            if assign.len() != model_children.len() {
                continue;
            }
            for &(c, k) in &assign {
                let e = &self.entries[c];
                let p = &parts[k];
                let pristine = e.name == p.name
                    && e.enabled
                    && !e.static_geometry
                    && e.lightmap_scale == 1.0
                    && e.cast_shadows.is_on()
                    && e.layer == 0
                    && matches!(e.material, MaterialRef::Model)
                    && e.extra_materials.iter().all(|m| matches!(m, MaterialRef::Model))
                    && e.extra_materials.len() == p.extra_meshes.len()
                    && untouched_components(&e.components)
                    && !has_children[c]
                    && (e.id.is_empty() || !all_data.contains(&e.id))
                    && close(&e.translation, &p.translation)
                    && close(&e.rotation, &p.rotation)
                    && close(&e.scale, &p.scale);
                if pristine {
                    drop[c] = true;
                } else {
                    self.entries[c].model_part = Some(PartRef::Id(p.part_id));
                }
            }
            let removed = (0..parts.len()).filter(|&k| !claimed[k]).map(|k| PartRef::Id(parts[k].part_id)).collect();
            let model = parts.iter().find_map(|p| p.model);
            self.entries[r].model = Some(ModelInstanceRef { path, model, removed });
        }
        if !drop.iter().any(|d| *d) {
            return;
        }
        let mut new_index = vec![None; n];
        let mut next = 0;
        for i in 0..n {
            if !drop[i] {
                new_index[i] = Some(next);
                next += 1;
            }
        }
        let old = std::mem::take(&mut self.entries);
        self.entries = old
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !drop[*i])
            .map(|(_, mut e)| {
                e.parent = e.parent.and_then(|p| new_index.get(p).copied().flatten());
                e
            })
            .collect();
    }

    /// Rebuild the parts compacted away by [`Self::compact_models`]: every
    /// part of a model root that wasn't removed and isn't stored as an entry
    /// is appended as a pristine child with a stable id.
    pub fn expand_models(&mut self, parts_of: &dyn Fn(&str) -> Option<Vec<ModelPart>>) {
        self.resolve_model_refs(parts_of);
        let n = self.entries.len();
        for r in 0..n {
            let Some(m) = self.entries[r].model.clone() else { continue };
            let Some(parts) = parts_of(&m.path) else { continue };
            // Migrate positions saved before part ids to ids, so the next save writes ids.
            for c in 0..n {
                if self.entries[c].parent == Some(r)
                    && let Some(PartRef::Index(k)) = self.entries[c].model_part
                    && let Some(p) = parts.get(k as usize)
                {
                    self.entries[c].model_part = Some(PartRef::Id(p.part_id));
                }
            }
            let migrated: Vec<PartRef> = m.removed.iter().map(|x| x.resolve(&parts).map_or(*x, |k| PartRef::Id(parts[k].part_id))).collect();
            if let Some(model) = &mut self.entries[r].model {
                model.removed = migrated;
            }
            let removed: Vec<usize> = m.removed.iter().filter_map(|x| x.resolve(&parts)).collect();
            let stored: Vec<usize> = (0..n).filter(|&c| self.entries[c].parent == Some(r)).filter_map(|c| self.entries[c].model_part).filter_map(|x| x.resolve(&parts)).collect();
            let root_id = self.entries[r].id.clone();
            for (k, p) in parts.iter().enumerate() {
                if removed.contains(&k) || stored.contains(&k) {
                    continue;
                }
                self.entries.push(SceneEntry {
                    id: part_entry_id(&root_id, p.part_id),
                    name: p.name.clone(),
                    source: ObjectSource::Model {
                        path: m.path.clone(),
                        mesh: p.mesh,
                        extra_meshes: p.extra_meshes.clone(),
                        model: p.model,
                        mesh_id: (!p.mesh_id.is_none()).then_some(p.mesh_id),
                        extra_mesh_ids: if p.extra_mesh_ids.iter().all(|i| !i.is_none()) { p.extra_mesh_ids.clone() } else { Vec::new() },
                    },
                    enabled: true,
                    static_geometry: false,
                    lightmap_scale: 1.0,
                    cast_shadows: zestengine_core::ShadowCasting::On,
                    layer: 0,
                    material: MaterialRef::Model,
                    extra_materials: vec![MaterialRef::Model; p.extra_meshes.len()],
                    renderer_split: true,
                    parent: Some(r),
                    components: vec![ComponentData::new(
                        zestengine_core::MESH_RENDERER,
                        ron::to_string(&zestengine_core::MeshRenderer::inheriting(1 + p.extra_meshes.len())).unwrap_or_default(),
                    )],
                    translation: p.translation,
                    rotation: p.rotation,
                    scale: p.scale,
                    model: None,
                    model_part: Some(PartRef::Id(p.part_id)),
                    prefab: None,
                    prefab_parent: None,
                    net_role: Default::default(),
                    component_roles: Default::default(),
                });
            }
        }
    }
}

/// Scene-level environment: ambient fill, skybox, fog, shadows, GI. The sun is
/// no longer here: it is a `Directional Light` object (the legacy `sun_*`
/// fields are only read to migrate old scenes, see
/// [`SceneFile::migrate_environment_sun`]). Configured in the editor's
/// Environment window.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldEnvironment {
    /// Linear ambient fill color.
    pub ambient: [f32; 3],
    /// Ambient multiplier.
    pub ambient_intensity: f32,
    /// LEGACY (read for loading only): the built-in world sun of older scenes.
    /// The sun is now an ordinary `Directional Light` object; on load
    /// [`SceneFile::migrate_environment_sun`] turns an enabled sun into that
    /// object and sets this to false. New scenes never enable it, and the
    /// renderer ignores it. Not written when off.
    #[serde(default, skip_serializing_if = "is_false")]
    pub sun_enabled: bool,
    /// LEGACY, see `sun_enabled`.
    #[serde(default = "legacy_sun_color", skip_serializing_if = "is_legacy_sun_color")]
    pub sun_color: [f32; 3],
    /// LEGACY, see `sun_enabled`.
    #[serde(default = "legacy_sun_intensity", skip_serializing_if = "is_legacy_sun_intensity")]
    pub sun_intensity: f32,
    /// LEGACY, see `sun_enabled`. World-space travel direction (normalized on use).
    #[serde(default = "legacy_sun_direction", skip_serializing_if = "is_legacy_sun_direction")]
    pub sun_direction: [f32; 3],
    /// Draw the skybox behind the scene (off = clear-color/black background).
    pub skybox_enabled: bool,
    /// Shadow-map resolution per layer (pixels). Common values 512–4096.
    #[serde(default = "default_shadow_resolution")]
    pub shadow_resolution: u32,
    /// PCF kernel softness multiplier (1.0 = one texel spacing).
    #[serde(default = "default_shadow_softness")]
    pub shadow_softness: f32,
    /// Shadow receiver depth-bias strength (see Features.md, "Shadow bias").
    /// Point/spot: 1500 is 1x with a sub-texel default; directional keeps slope scaling.
    /// Raise it if shadows show acne, lower it if thin casters detach from
    /// their contact point. 1500 is the tuned default.
    #[serde(default = "default_shadow_bias_scale")]
    pub shadow_bias_scale: f32,
    /// Directional shadow normal-offset, in shadow texels: the receiver moves
    /// along its surface normal by this many texels x sin(angle to the light).
    /// Cures acne at low sun / grazing angles without detaching thin casters
    /// (a depth bias big enough would). Raise for acne on steep slopes, lower
    /// if shadows start a little late at contact points. 2 is the tuned default.
    /// Point/spot lights compare receiver-plane depths without normal displacement.
    #[serde(default = "default_shadow_normal_offset")]
    pub shadow_normal_offset: f32,
    /// Cascades for the sun's shadow (1..=4; a mobile tier caps it lower).
    /// Fewer = more shadow-map resolution per cascade but a coarser spread
    /// with distance, and fewer shadow passes.
    #[serde(default = "default_shadow_cascades")]
    pub shadow_cascades: u32,
    /// Cascade split blend: 0 = uniform splits, 1 = logarithmic (more
    /// resolution close to the camera). 0.7 is the default.
    #[serde(default = "default_shadow_cascade_lambda")]
    pub shadow_cascade_lambda: f32,
    /// Directional shadow coverage (world units). The ortho box is fit to
    /// this size ahead of the camera. Smaller = sharper, less coverage.
    #[serde(default = "default_shadow_distance")]
    pub shadow_distance: f32,
    /// Lighting-bake settings (Bakery-style: texels-per-meter density).
    #[serde(default)]
    pub bake: BakeSettings,
    /// Realtime-GI settings: when enabled (and no bake exists) the engine
    /// continuously re-traces light probes from the realtime lights so surfaces
    /// show live indirect bounce, both in the editor and in a shipped game.
    #[serde(default, deserialize_with = "de_realtime_gi")]
    pub realtime_gi: RealtimeGi,
    /// Always-applied global post-processing (the implicit "global volume").
    /// Local `VolumeComponent`s blend on top of this by priority/weight. Defaults
    /// to ACES tonemap so a fresh scene is tonemapped without needing a volume.
    #[serde(default)]
    pub postfx: crate::PostFxProfile,
    /// Exponential distance + height fog (atmospheric depth).
    #[serde(default)]
    pub fog_enabled: bool,
    #[serde(default = "default_fog_color")]
    pub fog_color: [f32; 3],
    #[serde(default = "default_fog_density")]
    pub fog_density: f32,
    /// Height falloff (fog thins above `fog_height_ref`; 0 = uniform).
    #[serde(default)]
    pub fog_height_falloff: f32,
    #[serde(default)]
    pub fog_height_ref: f32,
    /// View distance before fog starts.
    #[serde(default)]
    pub fog_start_distance: f32,
    /// Cubemap skybox: 6 project-relative face image paths in +X,-X,+Y,-Y,+Z,-Z
    /// order. Takes precedence over the equirect skybox when set.
    #[serde(default)]
    pub skybox_faces: Option<[String; 6]>,
}

fn default_fog_color() -> [f32; 3] {
    [0.6, 0.68, 0.78]
}
fn default_fog_density() -> f32 {
    0.02
}

/// Accept both the legacy `realtime_gi: bool` (just the enable toggle) and the
/// current struct form, so scenes saved before the settings landed still load.
/// Uses a Visitor instead of `#[serde(untagged)]` so the struct path delegates
/// to the typed `RealtimeGi` deserialize. Untagged would buffer into a
/// self-describing `Content`, which can't replay RON enum variants (the `mode`
/// field), breaking the valid struct form.
fn de_realtime_gi<'de, D>(d: D) -> Result<RealtimeGi, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = RealtimeGi;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a bool (legacy) or a RealtimeGi struct")
        }
        fn visit_bool<E: serde::de::Error>(self, enabled: bool) -> Result<RealtimeGi, E> {
            Ok(RealtimeGi {
                enabled,
                ..Default::default()
            })
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            map: A,
        ) -> Result<RealtimeGi, A::Error> {
            RealtimeGi::deserialize(serde::de::value::MapAccessDeserializer::new(map))
        }
    }
    d.deserialize_any(V)
}

/// How the realtime-GI probe trace is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GiMode {
    /// FluxRT: hardware ray-query (RT cores) against the scene BVH. Most accurate.
    #[default]
    FluxRT,
    /// Flux: software ray-marching of per-mesh signed distance fields (no RT
    /// cores, runs anywhere).
    Flux,
    /// FluxVoxel: analytic voxel light volume (no ray tracing). Injects lights +
    /// emissive into an SH-L1 probe grid every frame — cheapest backend, built
    /// for VR. Static + dynamic meshes read it; dynamic lights mix in live.
    FluxVoxel,
}

impl GiMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::FluxRT => "FluxRT (hardware)",
            Self::Flux => "Flux (software)",
            Self::FluxVoxel => "FluxVoxel",
        }
    }
    pub const ALL: [GiMode; 3] = [Self::FluxRT, Self::Flux, Self::FluxVoxel];

    // --- Per-backend capability flags ---
    // Each backend declares which settings it actually consumes, so the
    // Environment UI can render only the relevant controls (modular settings)
    // and the engine can skip work a backend ignores.

    /// Diffuse bounce mode (FluxRT / Flux). FluxVoxel injects direct-only.
    pub fn uses_bounces(self) -> bool {
        matches!(self, Self::FluxRT | Self::Flux)
    }
    /// Rays-per-probe quality preset (the ray-marching/ray-query backends).
    pub fn uses_quality(self) -> bool {
        matches!(self, Self::FluxRT | Self::Flux)
    }
    /// Per-probe sample count + firefly clamp (the sampling backends).
    pub fn uses_samples(self) -> bool {
        matches!(self, Self::FluxRT | Self::Flux)
    }
    /// Signed-distance-field resolution (software march only).
    pub fn uses_gdf(self) -> bool {
        matches!(self, Self::Flux)
    }
    /// March distance (the software march; RT uses the BVH, Voxel uses volumes).
    pub fn uses_march_distance(self) -> bool {
        matches!(self, Self::Flux)
    }
    /// Voxel-grid density (the FluxVoxel analytic volume only).
    pub fn uses_voxel_density(self) -> bool {
        matches!(self, Self::FluxVoxel)
    }
    /// DDGI-style per-probe visibility (currently the FluxVoxel voxel-light path).
    pub fn uses_ddgi_occlusion(self) -> bool {
        matches!(self, Self::FluxVoxel)
    }
    /// LPV-style propagation bounce (FluxVoxel only — RT/SDF bounce via their march).
    pub fn uses_propagation(self) -> bool {
        matches!(self, Self::FluxVoxel)
    }
    /// VXGI-style specular-from-volume (FluxVoxel only — RT/SDF reflect via cube/BVH).
    pub fn uses_voxel_specular(self) -> bool {
        matches!(self, Self::FluxVoxel)
    }
}

/// Flux quality preset. Drives per-frame samples-per-probe and the march step
/// cap; the temporal accumulator does the rest, so even Performance stays smooth
/// once settled. Probe density (screen-probe spacing) is fixed for now.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FluxQuality {
    /// Fewest rays: cheapest, slightly noisier while moving.
    Performance,
    #[default]
    Balanced,
    High,
    /// Most rays: sharpest contact GI, highest cost.
    Ultra,
}

impl FluxQuality {
    pub fn label(self) -> &'static str {
        match self {
            Self::Performance => "Performance",
            Self::Balanced => "Balanced",
            Self::High => "High",
            Self::Ultra => "Ultra",
        }
    }
    pub const ALL: [FluxQuality; 4] =
        [Self::Performance, Self::Balanced, Self::High, Self::Ultra];
    /// Rays per screen probe per frame (temporal accumulation does the rest).
    /// One probe serves SCREEN_PROBE_DIV² pixels (16 at DIV=4), so the per-pixel
    /// cost is amortized — match ~64 rays/probe at the top preset rather
    /// than the old per-pixel-era counts (which badly undersampled the probes).
    pub fn samples(self) -> u32 {
        match self {
            Self::Performance => 16,
            Self::Balanced => 32,
            Self::High => 48,
            Self::Ultra => 64,
        }
    }
}

/// World-probe fallback policy. Flux drives the main view; the legacy world-probe
/// DDGI grid only feeds the in-game camera + off-screen fallback, so it doesn't
/// need to re-march every frame (that was a redundant ~6ms CPU cost).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ProbeFallback {
    /// No world-probe march while Flux is active (lowest CPU).
    Off,
    /// March the world probes at a low cadence for the fallback paths.
    #[default]
    Throttled,
}

impl ProbeFallback {
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Throttled => "Throttled",
        }
    }
    pub const ALL: [ProbeFallback; 2] = [Self::Off, Self::Throttled];
}

/// How the auto FluxVoxel grid is laid out when there are no author-placed
/// FluxVolumes. Trades coverage of empty space against probe count (= per-frame
/// cost). Only consulted by the FluxVoxel backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum VoxelGridMode {
    /// One grid spanning the whole static-scene AABB. Simplest, but the probe
    /// count scales with level size — empty space you never visit is computed too.
    #[default]
    WholeScene,
    /// A fixed-size grid centred on the camera (snapped to the probe spacing so it
    /// scrolls in discrete steps). Constant probe count regardless of level size:
    /// space you never go to is never computed. Best for large levels.
    CameraClipmap,
    /// Whole-scene bounds, but probes far from any geometry are dropped (left dark)
    /// so open air between surfaces costs nothing to inject/propagate.
    Occupancy,
    /// Tight grids hugging clusters of objects; the gaps between clusters aren't
    /// covered. Sparse by construction. Capped at a few clusters (shader volume cap).
    PerObject,
}

impl VoxelGridMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::WholeScene => "Whole scene",
            Self::CameraClipmap => "Camera clipmap",
            Self::Occupancy => "Occupancy culled",
            Self::PerObject => "Per object",
        }
    }
    pub const ALL: [VoxelGridMode; 4] = [
        Self::WholeScene,
        Self::CameraClipmap,
        Self::Occupancy,
        Self::PerObject,
    ];
}

/// Realtime global-illumination (probe) settings. Drives the live probe re-trace
/// that lets realtime lights cast soft indirect bounce without a bake.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RealtimeGi {
    pub enabled: bool,
    /// Flux quality preset (samples-per-probe + march step cap).
    pub quality: FluxQuality,
    /// 1 = single diffuse bounce; 2..4 enable iterative surface radiosity with
    /// no fixed bounce limit. Also bounds the finite path used for cache misses.
    pub bounces: u32,
    /// GI strength multiplier on the indirect term.
    pub intensity: f32,
    /// Temporal smoothing 0..1: higher = smoother but more motion lag; lower =
    /// sharper/more responsive but noisier while the camera moves.
    pub smoothing: f32,
    // --- Advanced ---
    /// GDF resolution (occlusion + bounce detail): 64 / 128 / 256.
    pub gdf_resolution: u32,
    /// Max trace distance in world units (0 = auto from scene size).
    pub march_distance: f32,
    /// Firefly clamp on bounce samples; caps bright outliers (lower = calmer).
    pub firefly_clamp: f32,
    /// Multiplier on how much light emissive objects throw into the GI (their
    /// on-screen brightness is unchanged). 1 = physical; small emitters at
    /// emission intensity ~1 are dim light sources, so raise this (like
    /// Lumen's emissive boost) for stronger coloured pools.
    pub emissive_boost: f32,
    /// World-probe fallback policy for the in-game camera / off-screen surfaces.
    pub probe_fallback: ProbeFallback,
    /// Screen-space reflections: trace specular rays against the depth prepass +
    /// last frame's colour. Only active while Flux runs (it owns the depth prepass).
    pub ssr_enabled: bool,
    /// SSR reflection strength multiplier.
    pub ssr_intensity: f32,
    /// SSR max ray distance in view-space units.
    pub ssr_max_distance: f32,
    /// SSR roughness cutoff: surfaces rougher than this skip the march.
    pub ssr_roughness_cutoff: f32,
    /// Reflection model: 0 = environment cube only, 1 = screen-space (SSR),
    /// 2 = ray-traced (needs GPU ray-query support, falls back to SSR).
    pub reflection_mode: u8,
    /// Ray-traced reflections: rays per pixel. Rough surfaces spread them over
    /// their glossy lobe, so more rays = smoother glossy reflections.
    pub rt_samples: u32,
    /// Ray-traced reflections: bounces (2+ shows mirrors / metals inside reflections).
    pub rt_bounces: u32,
    pub ray_lighting_mode: zestengine_render::RayLightingMode,
    // --- FluxVoxel backend ---
    /// Probes per world meter for the auto scene-covering FluxVoxel grid (used
    /// when there are 0 author-placed FluxVolumes). Higher = finer GI, more cost.
    pub voxel_density: f32,
    /// Auto-build a scene-covering FluxVoxel grid when no FluxVolumes exist, so
    /// the whole scene gets voxel GI (not just inside author-placed volumes).
    /// Author-placed FluxVolumes and the auto grid are mutually exclusive: placing
    /// any FluxVolume turns the auto grid off (the author then controls coverage).
    pub voxel_auto_grid: bool,
    /// Layout strategy for the auto grid (see `VoxelGridMode`). Lets large levels
    /// keep a constant probe count (camera clipmap) or skip empty space.
    pub voxel_grid_mode: VoxelGridMode,
    /// Half-extent (metres) of the camera-clipmap auto grid: the grid is a cube of
    /// side `2 * extent` centred on the camera. Larger = more GI range, more cost.
    pub voxel_clipmap_extent: f32,
    /// DDGI-style occlusion for FluxVoxel voxel lights: per-probe directional
    /// distance moments + a Chebyshev visibility test so voxel lights are blocked
    /// by geometry (cheap shadows). Toggle off on low-end/VR to save the moment
    /// write + test. Expensive-ish; ON by default.
    pub voxel_ddgi_occlusion: bool,
    /// FluxVoxel light propagation (LPV-style ≥1 diffuse bounce): spread injected
    /// radiance through the probe grid so light fills shadowed pockets. Toggle off
    /// on low-end/VR for direct-only (cheapest). ON by default.
    pub voxel_propagation: bool,
    /// FluxVoxel specular GI: metallic/rough surfaces sample the probe volume in the
    /// reflection direction (VXGI-style glossy approx) so they pick up emissive +
    /// voxel-light bounce, not just the reflection cube. ON by default.
    pub voxel_specular: bool,
    // --- Internal: not shown in the UI. The world-probe DDGI fallback path still
    // uses these; samples/mode are derived from `quality`/Flux on load. ---
    #[doc(hidden)]
    pub mode: GiMode,
    #[doc(hidden)]
    pub samples: u32,
    #[doc(hidden)]
    pub probe_spacing: f32,
    #[doc(hidden)]
    pub temporal_blend: f32,
    #[doc(hidden)]
    pub update_interval: f32,
}

impl Default for RealtimeGi {
    fn default() -> Self {
        // Flux is the realtime GI path: soft, gently-settling realtime-GI look. The
        // internal world-probe fields stay only for the throttled fallback.
        Self {
            enabled: false,
            quality: FluxQuality::Balanced,
            bounces: 2,
            intensity: 1.0,
            smoothing: 0.5,
            gdf_resolution: 128,
            march_distance: 0.0, // auto
            firefly_clamp: 4.0,
            emissive_boost: 1.0,
            // Default: Flux only (no world-probe march). The DDGI fallback is
            // opt-in for projects using the in-game camera / off-screen GI.
            probe_fallback: ProbeFallback::Off,
            ssr_enabled: true,
            ssr_intensity: 1.0,
            ssr_max_distance: 40.0,
            ssr_roughness_cutoff: 0.6,
            reflection_mode: 1,
            rt_samples: 1,
            rt_bounces: 1,
            ray_lighting_mode: zestengine_render::RayLightingMode::SurfaceCache,
            voxel_density: 1.0,
            voxel_auto_grid: true,
            voxel_grid_mode: VoxelGridMode::WholeScene,
            voxel_clipmap_extent: 16.0,
            voxel_ddgi_occlusion: true,
            voxel_propagation: true,
            voxel_specular: true,
            mode: GiMode::Flux,
            samples: 64,
            probe_spacing: 1.0,
            temporal_blend: 0.12,
            // 0 = the GDF + emitter feed (what Flux samples) refreshes every
            // frame so moving emitters track without lag.
            update_interval: 0.0,
        }
    }
}

/// Lighting-bake parameters, authored per scene. Resolution is a texel
/// density (Bakery / Unity style); each static object's lightmap size is
/// `density × world-AABB size`, clamped to `max_lightmap`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BakeSettings {
    /// Lightmap texels per world meter.
    pub texel_density: f32,
    /// Indirect bounces per path (0 = direct + sky only).
    pub bounces: u32,
    /// Paths traced per texel / per probe.
    pub samples: u32,
    /// Upper clamp on a single object's lightmap resolution.
    pub max_lightmap: u32,
    /// GPU throttle: fraction of each GPU bake dispatch's time to then idle the GPU
    /// so the desktop stays responsive while baking. 0 = max speed (bake hogs the
    /// GPU), 1 = ~50% duty cycle (≈2× slower bake, system stays usable).
    #[serde(default = "default_gpu_throttle")]
    pub gpu_throttle: f32,
    /// Store baked lightmaps as BC6H (~16x smaller on disk and in VRAM than
    /// RGBA32F). Off writes the raw v1 format.
    pub compress_lightmaps: bool,
    /// Store baked reflection-probe cubes (full mip chain) as BC6H.
    pub compress_reflections: bool,
}

fn default_gpu_throttle() -> f32 {
    1.0
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            // Detailed by default so lightmaps aren't pixelated/blocky: 40 texels/m
            // with a 2048 per-object cap (was 16/m capped at 512, which clamped any
            // sizeable surface to visible texel squares). Bakes take longer, but the
            // result is sharp. Crank texel_density (slider to 1024) + Max Lightmap
            // (up to 4096) for even finer detail.
            texel_density: 40.0,
            // High-quality multi-bounce by default (the FluxBaker slider goes to 8):
            // 6 bounces + plenty of paths per texel so the indirect settles to a
            // clean, good-looking lightmap rather than a noisy 1–2-bounce preview.
            bounces: 6,
            samples: 512,
            max_lightmap: 2048,
            gpu_throttle: default_gpu_throttle(),
            compress_lightmaps: true,
            compress_reflections: true,
        }
    }
}

fn default_shadow_resolution() -> u32 {
    2048
}

fn default_shadow_softness() -> f32 {
    1.0
}

fn default_shadow_bias_scale() -> f32 {
    1500.0
}

fn default_shadow_normal_offset() -> f32 {
    2.0
}

fn default_shadow_cascades() -> u32 {
    4
}

fn is_false(v: &bool) -> bool {
    !*v
}
fn legacy_sun_color() -> [f32; 3] {
    [1.0, 0.98, 0.92]
}
fn legacy_sun_intensity() -> f32 {
    3.0
}
fn legacy_sun_direction() -> [f32; 3] {
    [-0.4, -1.0, -0.3]
}
fn is_legacy_sun_color(v: &[f32; 3]) -> bool {
    *v == legacy_sun_color()
}
fn is_legacy_sun_intensity(v: &f32) -> bool {
    *v == legacy_sun_intensity()
}
fn is_legacy_sun_direction(v: &[f32; 3]) -> bool {
    *v == legacy_sun_direction()
}

/// Name of the light object a default scene (and a migrated legacy sun) holds.
pub const DIRECTIONAL_LIGHT_NAME: &str = "Directional Light";

/// A `Directional Light` scene entry (a `Light` component set to Directional)
/// travelling along `direction`, as the default scenes and the legacy-sun
/// migration write it. `mode` is Mixed so the light is baked like the old
/// environment sun was.
pub fn directional_light_entry(color: [f32; 3], intensity: f32, direction: [f32; 3], mode: zestengine_core::LightMode) -> SceneEntry {
    let dir = glam::Vec3::from(direction).normalize_or(glam::Vec3::NEG_Y);
    let rotation = glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, dir);
    let light = zestengine_core::LightComponent {
        kind: zestengine_core::LightKind::Directional,
        mode,
        color,
        intensity,
        shadow_bias: 0.003,
        radius: 0.03,
        ..Default::default()
    };
    SceneEntry {
        id: zestengine_core::ObjectId::new().to_string(),
        name: DIRECTIONAL_LIGHT_NAME.into(),
        source: ObjectSource::Light,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: zestengine_core::ShadowCasting::On,
        layer: 0,
        material: MaterialRef::Model,
        extra_materials: Vec::new(),
        renderer_split: false,
        parent: None,
        components: vec![ComponentData::of(&light)],
        translation: [0.0, 10.0, 0.0],
        rotation: rotation.to_array(),
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: None,
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    }
}

/// The default scene's key light: colour (1, 0.98, 0.92), intensity 3,
/// travelling along (-0.4, -1, -0.3).
pub fn default_directional_light_entry() -> SceneEntry {
    directional_light_entry(legacy_sun_color(), legacy_sun_intensity(), legacy_sun_direction(), zestengine_core::LightMode::Mixed)
}

fn default_shadow_cascade_lambda() -> f32 {
    0.7
}

fn default_shadow_distance() -> f32 {
    25.0
}

impl Default for WorldEnvironment {
    fn default() -> Self {
        Self {
            ambient: [0.13, 0.14, 0.18],
            ambient_intensity: 1.0,
            sun_enabled: false,
            sun_color: legacy_sun_color(),
            sun_intensity: legacy_sun_intensity(),
            sun_direction: legacy_sun_direction(),
            skybox_enabled: true,
            shadow_resolution: default_shadow_resolution(),
            shadow_softness: default_shadow_softness(),
            shadow_bias_scale: default_shadow_bias_scale(),
            shadow_normal_offset: default_shadow_normal_offset(),
            shadow_cascades: default_shadow_cascades(),
            shadow_cascade_lambda: default_shadow_cascade_lambda(),
            shadow_distance: default_shadow_distance(),
            bake: BakeSettings::default(),
            realtime_gi: RealtimeGi::default(),
            postfx: crate::PostFxProfile::default(),
            fog_enabled: false,
            fog_color: default_fog_color(),
            fog_density: default_fog_density(),
            fog_height_falloff: 0.0,
            fog_height_ref: 0.0,
            fog_start_distance: 0.0,
            skybox_faces: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(from = "SceneFileWire")]
pub struct SceneFile {
    pub entries: Vec<SceneEntry>,
    /// Project-relative path to the equirectangular skybox image, if any.
    /// `None` uses the procedural gradient sky.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skybox: Option<String>,
    /// Scene environment / world lighting.
    #[serde(default)]
    pub environment: WorldEnvironment,
    /// Layer names + collision matrix (Unity-style). Drives physics collision
    /// filtering and per-camera render culling masks.
    #[serde(default)]
    pub layers: zestengine_core::LayerSettings,
    /// Last editor (fly-cam) viewpoint, restored on open so the scene reopens
    /// framed the same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor_camera: Option<EditorCamera>,
    /// Object ids whose hierarchy row was COLLAPSED in the scene tree, so the
    /// tree reopens in the same expanded/collapsed state. (Stored as collapsed
    /// since the default is expanded.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collapsed: Vec<String>,
    /// Object ids hidden in the editor's scene view (the hierarchy eye). Editor
    /// view state only: saved so the scene reopens the same way, never honoured
    /// by Play, runtime, server or builds (stripped from them), and not part of
    /// prefabs. A hidden object is still enabled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scene_hidden: Vec<String>,
    /// Object ids locked in the editor's scene view (not pickable by a click or
    /// box select). Same rules as `scene_hidden`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scene_locked: Vec<String>,
    /// Layout version of the file: 0/1 = renderer settings are loose entry fields
    /// (`material`, `cast_shadows`...), 2 = mesh objects carry a `Mesh Renderer` component
    /// (see `renderer_migration`). Kept last: it only changes through migration.
    #[serde(default, skip_serializing_if = "is_zero_version")]
    pub format_version: u32,
}

fn is_zero_version(v: &u32) -> bool {
    *v == 0
}

/// What a `.scene` is parsed into; [`SceneFile`] is built from it, moving the loose renderer
/// fields of an older file into `Mesh Renderer` components on the way (every loader gets the
/// migration, none can forget it).
#[derive(Deserialize)]
struct SceneFileWire {
    entries: Vec<SceneEntry>,
    #[serde(default)]
    skybox: Option<String>,
    #[serde(default)]
    environment: WorldEnvironment,
    #[serde(default)]
    layers: zestengine_core::LayerSettings,
    #[serde(default)]
    editor_camera: Option<EditorCamera>,
    #[serde(default)]
    collapsed: Vec<String>,
    #[serde(default)]
    scene_hidden: Vec<String>,
    #[serde(default)]
    scene_locked: Vec<String>,
    #[serde(default)]
    format_version: u32,
}

impl From<SceneFileWire> for SceneFile {
    fn from(w: SceneFileWire) -> Self {
        let mut file = SceneFile::from_wire(w);
        file.migrate_renderers();
        file
    }
}

impl SceneFile {
    /// Parse a scene WITHOUT the renderer migration (tests of the migration itself).
    #[cfg(test)]
    pub(crate) fn parse_unmigrated(text: &str) -> SceneFile {
        SceneFile::from_wire(ron::from_str::<SceneFileWire>(text).expect("parses"))
    }

    fn from_wire(w: SceneFileWire) -> SceneFile {
        SceneFile {
            entries: w.entries,
            skybox: w.skybox,
            environment: w.environment,
            layers: w.layers,
            editor_camera: w.editor_camera,
            collapsed: w.collapsed,
            scene_hidden: w.scene_hidden,
            scene_locked: w.scene_locked,
            format_version: w.format_version,
        }
    }
}

/// Saved editor fly-camera pose (mirrors `FlyCamera`'s position/yaw/pitch).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EditorCamera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

#[cfg(test)]
mod net_role_tests {
    use super::*;
    use zestengine_core::NetRole;

    const OLD_ENTRY: &str = r#"(name: "Old", source: Empty, material: Model, translation: (0.0, 0.0, 0.0), rotation: (0.0, 0.0, 0.0, 1.0), scale: (1.0, 1.0, 1.0))"#;

    #[test]
    fn old_scenes_load_as_both_and_both_is_not_written() {
        let e: SceneEntry = ron::from_str(OLD_ENTRY).unwrap();
        assert_eq!(e.net_role, NetRole::Both);
        assert!(e.component_roles.is_empty());
        let text = ron::to_string(&e).unwrap();
        assert!(!text.contains("net_role") && !text.contains("component_roles"), "{text}");
    }

    #[test]
    fn roles_round_trip() {
        let mut e: SceneEntry = ron::from_str(OLD_ENTRY).unwrap();
        e.net_role = NetRole::Server;
        e.component_roles.insert("Audio Source#1".into(), NetRole::Both);
        let back: SceneEntry = ron::from_str(&ron::to_string(&e).unwrap()).unwrap();
        assert_eq!(back.net_role, NetRole::Server);
        assert_eq!(back.component_roles.get("Audio Source#1"), Some(&NetRole::Both));
    }
}

#[cfg(test)]
mod rgi_compat_tests {
    use super::*;

    #[test]
    fn component_enabled_flag_is_saved_only_when_off() {
        let on = ComponentData::new("Grabbable", "()");
        assert!(!ron::to_string(&on).unwrap().contains("enabled"), "omitted while on");
        let off = ComponentData { enabled: false, ..on.clone() };
        let text = ron::to_string(&off).unwrap();
        assert!(text.contains("enabled:false"), "{text}");
        assert_eq!(ron::from_str::<ComponentData>(&text).unwrap(), off);
        let old: ComponentData = ron::from_str("(kind:\"Grabbable\",data:\"()\")").unwrap();
        assert!(old.enabled, "old scenes load switched on");
    }

    #[test]
    fn voxel_grid_mode_all_covers_every_variant_with_labels() {
        assert_eq!(VoxelGridMode::ALL.len(), 4);
        for m in VoxelGridMode::ALL {
            assert!(!m.label().is_empty());
        }
        // Default is the simplest whole-scene grid (current behaviour).
        assert_eq!(VoxelGridMode::default(), VoxelGridMode::WholeScene);
    }

    #[test]
    fn realtime_gi_default_has_grid_mode_and_clipmap_extent() {
        let gi = RealtimeGi::default();
        assert_eq!(gi.voxel_grid_mode, VoxelGridMode::WholeScene);
        assert!(gi.voxel_clipmap_extent > 0.0);
    }

    #[test]
    fn legacy_bool_and_struct_both_parse() {
        let legacy: WorldEnvironment =
            ron::from_str("(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,sun_enabled:true,sun_color:(1.0,1.0,1.0),sun_intensity:3.0,sun_direction:(0.0,-1.0,0.0),skybox_enabled:true,realtime_gi:false)")
                .expect("legacy bool form must parse");
        assert!(!legacy.realtime_gi.enabled);

        let full: WorldEnvironment =
            ron::from_str("(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,sun_enabled:true,sun_color:(1.0,1.0,1.0),sun_intensity:3.0,sun_direction:(0.0,-1.0,0.0),skybox_enabled:true,realtime_gi:(enabled:true,bounces:3))")
                .expect("struct form must parse");
        assert!(full.realtime_gi.enabled);
        assert_eq!(full.realtime_gi.bounces, 3);

        // The struct form with the `mode` enum field. This is what broke the
        // untagged shim (RON enum can't replay through serde's Content buffer).
        let with_mode: WorldEnvironment =
            ron::from_str("(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,sun_enabled:true,sun_color:(1.0,1.0,1.0),sun_intensity:3.0,sun_direction:(0.0,-1.0,0.0),skybox_enabled:true,realtime_gi:(enabled:true,mode:Flux,bounces:2,samples:64,intensity:1.0,probe_spacing:2.0,temporal_blend:0.12,update_interval:0.2))")
                .expect("struct form with mode enum must parse");
        assert_eq!(with_mode.realtime_gi.mode, GiMode::Flux);
    }
}

pub fn load_scene_file(path: impl AsRef<Path>) -> Result<SceneFile> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing scene file {}", path.display()))
}

pub fn save_scene_file(path: impl AsRef<Path>, scene: &SceneFile) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(scene, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod model_instance_tests {
    use super::*;

    fn parts() -> Vec<ModelPart> {
        (0..4)
            .map(|k| ModelPart {
                name: format!("Part{k}"),
                mesh: k,
                extra_meshes: Vec::new(),
                model: None,
                mesh_id: SubAssetId::NONE,
                extra_mesh_ids: Vec::new(),
                part_id: crate::sub_asset::hashed_id(SubAssetKind::Part, &format!("Part{k}")),
                translation: [k as f32, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            })
            .collect()
    }

    const GUID: &str = "11111111-2222-4333-8444-555555555555";

    /// The house model as it imports after a re-export: the same four parts
    /// (ids 100..104 named after the parts) but listed in `order`, so flattened
    /// indices no longer match the ones a scene saved earlier recorded.
    fn parts_reordered(order: &[usize]) -> Vec<ModelPart> {
        order
            .iter()
            .enumerate()
            .map(|(index, &k)| ModelPart {
                name: format!("Part{k}"),
                mesh: index,
                extra_meshes: Vec::new(),
                model: Some(GUID.to_string().try_into().unwrap()),
                mesh_id: SubAssetId(100 + k as u64),
                extra_mesh_ids: Vec::new(),
                part_id: crate::sub_asset::hashed_id(SubAssetKind::Part, &format!("Part{k}")),
                translation: [k as f32, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            })
            .collect()
    }

    fn mesh_of(file: &SceneFile, name: &str) -> (usize, Option<SubAssetId>) {
        match &file.entries.iter().find(|e| e.name == name).unwrap().source {
            ObjectSource::Model { mesh, mesh_id, .. } => (*mesh, *mesh_id),
            _ => panic!("not a model"),
        }
    }

    #[test]
    fn legacy_scene_migrates_to_mesh_ids_on_load() {
        // A scene saved before ids: path + flattened index only.
        let text = r#"(entries:[(name:"Door",source:Model(path:"models/house.glb",mesh:2),material:Model,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0))])"#;
        let mut file: SceneFile = ron::from_str(text).unwrap();
        assert_eq!(mesh_of(&file, "Door"), (2, None));
        file.resolve_model_refs(&|_| Some(parts_reordered(&[0, 1, 2, 3])));
        assert_eq!(mesh_of(&file, "Door"), (2, Some(SubAssetId(102))), "id filled from the stored index");
        let ObjectSource::Model { model, .. } = &file.entries[0].source else { panic!() };
        assert_eq!(model.map(|g| g.to_string()), Some(GUID.to_string()), "GUID recorded");
        // And it round-trips through RON in the new format.
        let out = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let back: SceneFile = ron::from_str(&out).unwrap();
        assert_eq!(mesh_of(&back, "Door"), (2, Some(SubAssetId(102))));
    }

    #[test]
    fn meshes_follow_their_id_when_the_model_is_reordered() {
        let mut file = SceneFile::default();
        file.entries.push(entry("Door", model(2), None, [0.0; 3]));
        file.entries.push(entry("Roof", model(3), None, [1.0, 0.0, 0.0]));
        // Saved against the original order: the ids get stamped.
        file.stamp_model_refs(&|_| Some(parts_reordered(&[0, 1, 2, 3])));
        assert_eq!(mesh_of(&file, "Door"), (2, Some(SubAssetId(102))));
        // The model is re-exported with parts in another order.
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let mut loaded: SceneFile = ron::from_str(&text).unwrap();
        loaded.resolve_model_refs(&|_| Some(parts_reordered(&[3, 2, 1, 0])));
        assert_eq!(mesh_of(&loaded, "Door").0, 1, "Part2 is now mesh 1");
        assert_eq!(mesh_of(&loaded, "Roof").0, 0, "Part3 is now mesh 0");
    }

    #[test]
    fn a_missing_id_keeps_the_stored_index() {
        let mut file = SceneFile::default();
        file.entries.push(entry("Door", model(2), None, [0.0; 3]));
        file.entries[0].source = ObjectSource::Model {
            path: "models/house.glb".into(),
            mesh: 2,
            extra_meshes: Vec::new(),
            model: None,
            mesh_id: Some(SubAssetId(999)),
            extra_mesh_ids: Vec::new(),
        };
        file.resolve_model_refs(&|_| Some(parts_reordered(&[0, 1, 2, 3])));
        assert_eq!(mesh_of(&file, "Door"), (2, Some(SubAssetId(999))), "unknown id: stored index kept, id untouched");
    }

    #[test]
    fn extra_slots_resolve_by_id_too() {
        let mut file = SceneFile::default();
        file.entries.push(entry("Door", model(0), None, [0.0; 3]));
        file.entries[0].source = ObjectSource::Model {
            path: "models/house.glb".into(),
            mesh: 0,
            extra_meshes: vec![1],
            model: None,
            mesh_id: Some(SubAssetId(100)),
            extra_mesh_ids: vec![SubAssetId(101)],
        };
        file.resolve_model_refs(&|_| Some(parts_reordered(&[1, 0])));
        let ObjectSource::Model { mesh, extra_meshes, .. } = &file.entries[0].source else { panic!() };
        assert_eq!((*mesh, extra_meshes.clone()), (1, vec![0]));
    }

    #[test]
    fn compacted_model_root_records_the_guid() {
        let mut file = SceneFile::default();
        file.entries.push(entry("House", ObjectSource::Empty, None, [0.0; 3]));
        file.entries.push(entry("Part0", model(0), Some(0), [0.0, 0.0, 0.0]));
        file.compact_models(&|_| Some(parts_reordered(&[0, 1, 2, 3])));
        let m = file.entries[0].model.as_ref().unwrap();
        assert_eq!(m.model.map(|g| g.to_string()), Some(GUID.to_string()));
    }

    #[test]
    fn a_moved_model_is_found_by_its_guid() {
        let dir = std::env::temp_dir().join(format!("zest_relocate_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("props/new")).unwrap();
        // The model now lives at props/new/house.glb with its meta beside it.
        let moved = dir.join("props/new/house.glb");
        std::fs::write(&moved, b"x").unwrap();
        let meta = crate::AssetMeta {
            importer: crate::ImporterSettings::Model(Default::default()),
            id: zestengine_core::ObjectId::new(),
        };
        crate::save_asset_meta(&moved, &meta).unwrap();

        let mut file = SceneFile::default();
        file.entries.push(entry("House", ObjectSource::Empty, None, [0.0; 3]));
        file.entries.push(entry("Part0", model(0), Some(0), [0.0; 3]));
        let ObjectSource::Model { model, .. } = &mut file.entries[1].source else { panic!() };
        *model = Some(meta.id);
        file.entries[0].model = Some(ModelInstanceRef { path: "models/house.glb".into(), model: Some(meta.id), removed: Vec::new() });

        let moves = file.relocate_models(&dir);
        assert_eq!(moves, vec![("models/house.glb".to_string(), "props/new/house.glb".to_string())]);
        let ObjectSource::Model { path, .. } = &file.entries[1].source else { panic!() };
        assert_eq!(path, "props/new/house.glb");
        assert_eq!(file.entries[0].model.as_ref().unwrap().path, "props/new/house.glb");
        assert!(file.relocate_models(&dir).is_empty(), "nothing missing the second time");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn entry(name: &str, source: ObjectSource, parent: Option<usize>, t: [f32; 3]) -> SceneEntry {
        SceneEntry {
            id: format!("00000000-0000-0000-0000-{:012}", name.len() * 1000 + t[0] as usize),
            name: name.into(),
            source,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: MaterialRef::Model,
            extra_materials: Vec::new(),
            renderer_split: false,
            parent,
            components: Vec::new(),
            translation: t,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    fn model(k: usize) -> ObjectSource {
        ObjectSource::Model { path: "models/house.glb".into(), mesh: k, extra_meshes: Vec::new(), model: None, mesh_id: None, extra_mesh_ids: Vec::new() }
    }

    #[test]
    fn part_refs_round_trip_and_read_old_positions() {
        let id = PartRef::Id(SubAssetId(0xdead_beef_0123));
        let text = ron::to_string(&vec![id, PartRef::Index(3)]).unwrap();
        let back: Vec<PartRef> = ron::from_str(&text).unwrap();
        assert_eq!(back, vec![id, PartRef::Index(3)]);
        // What an old scene holds: plain numbers.
        let old: Vec<PartRef> = ron::from_str("[0, 2, 7]").unwrap();
        assert_eq!(old, vec![PartRef::Index(0), PartRef::Index(2), PartRef::Index(7)]);
        assert!(ron::from_str::<PartRef>("\"nonsense\"").is_err());
    }

    #[test]
    fn a_scene_saved_with_part_positions_migrates_to_ids_and_survives_a_reorder() {
        // The old format: removed = [2], an edited part tagged with its index 1.
        let old = r#"(entries:[
            (id:"root",name:"House",source:Empty,material:Model,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0),
             model:Some((path:"models/house.glb",removed:[2]))),
            (id:"moved",name:"Part1",source:Model(path:"models/house.glb",mesh:1),material:Model,parent:Some(0),translation:(1.0,5.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0),model_part:Some(1)),
        ])"#;
        let mut file: SceneFile = ron::from_str(old).unwrap();
        file.expand_models(&|_| Some(parts()));
        let part = |k: usize| PartRef::Id(parts()[k].part_id);
        assert_eq!(file.entries[0].model.as_ref().unwrap().removed, vec![part(2)], "removed positions became ids");
        assert_eq!(file.entries.iter().find(|e| e.name == "Part1").unwrap().model_part, Some(part(1)));
        assert_eq!(file.entries.len(), 4, "root, the stored Part1, and the pristine Part0 and Part3");
        assert!(file.entries.iter().all(|e| e.name != "Part2"));
        // Saved again it holds ids, not positions.
        let text = ron::to_string(&file).unwrap();
        assert!(!text.contains("removed:[2]") && !text.contains("model_part:Some(1)"), "{text}");
        // The model is re-exported with its parts in another order: the same parts stay removed and edited.
        let reordered: Vec<ModelPart> = {
            let mut p = parts();
            p.reverse();
            for (i, x) in p.iter_mut().enumerate() {
                x.mesh = i;
            }
            p
        };
        let mut back: SceneFile = ron::from_str(&text).unwrap();
        back.entries.truncate(2);
        back.expand_models(&|_| Some(reordered.clone()));
        assert!(back.entries.iter().all(|e| e.name != "Part2"), "Part2 is still the removed one");
        assert_eq!(back.entries.iter().filter(|e| e.name == "Part1").count(), 1, "the edited part is not duplicated");
        assert_eq!(back.entries.len(), 4);
    }

    #[test]
    fn pristine_parts_drop_out_and_come_back() {
        let mut file = SceneFile::default();
        file.entries.push(entry("House", ObjectSource::Empty, None, [0.0; 3]));
        file.entries.push(entry("Part0", model(0), Some(0), [0.0, 0.0, 0.0]));
        file.entries.push(entry("Part1", model(1), Some(0), [1.0, 5.0, 0.0])); // moved
        // Part2 deleted by the user.
        file.entries.push(entry("Part3", model(3), Some(0), [3.0, 0.0, 0.0]));
        file.entries.push(entry("Lamp", ObjectSource::Empty, Some(0), [0.0; 3]));
        // Something references Part3, so it must keep its id and stay stored.
        let part3_id = file.entries[3].id.clone();
        file.entries[4].components.push(ComponentData::new("Spin", format!("(target:(Some(\"{part3_id}\")))")));

        let full = file.clone();
        file.compact_models(&|_| Some(parts()));
        assert_eq!(file.entries.len(), 4, "Part0 dropped");
        let part = |k: usize| PartRef::Id(parts()[k].part_id);
        assert_eq!(file.entries[0].model, Some(ModelInstanceRef { path: "models/house.glb".into(), model: None, removed: vec![part(2)] }));
        assert_eq!(file.entries[1].model_part, Some(part(1)));
        assert!(file.entries.iter().all(|e| e.parent.is_none_or(|p| p < file.entries.len())));

        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let mut back: SceneFile = ron::from_str(&text).unwrap();
        back.expand_models(&|_| Some(parts()));
        back.expand_models(&|_| Some(parts()));
        assert_eq!(back.entries.len(), full.entries.len(), "expansion is idempotent and restores the part");
        let p0 = back.entries.iter().find(|e| e.name == "Part0").unwrap();
        assert_eq!((p0.translation, p0.parent), ([0.0; 3], Some(0)));
        assert!(back.entries.iter().all(|e| e.name != "Part2"), "removed stays removed");
        let moved = back.entries.iter().find(|e| e.name == "Part1").unwrap();
        assert_eq!(moved.translation, [1.0, 5.0, 0.0]);
        let ids: Vec<&str> = back.entries.iter().map(|e| e.id.as_str()).collect();
        let mut again = back.clone();
        again.entries.truncate(4);
        again.expand_models(&|_| Some(parts()));
        assert_eq!(again.entries.last().unwrap().id, *ids.last().unwrap(), "rebuilt parts keep stable ids");
    }

    #[test]
    fn edited_materials_and_components_keep_a_part_stored() {
        let mut file = SceneFile::default();
        file.entries.push(entry("House", ObjectSource::Empty, None, [0.0; 3]));
        for k in 0..4 {
            file.entries.push(entry(&format!("Part{k}"), model(k), Some(0), [k as f32, 0.0, 0.0]));
        }
        file.entries[2].material = MaterialRef::File("materials/brick.material".into());
        file.entries[3].components.push(ComponentData::new("Spin", "()"));
        file.compact_models(&|_| Some(parts()));
        assert_eq!(file.entries.len(), 3, "root + the two edited parts");
    }

    #[test]
    fn cast_shadows_keeps_a_part_stored() {
        let mut file = SceneFile::default();
        file.entries.push(entry("House", ObjectSource::Empty, None, [0.0; 3]));
        for k in 0..4 {
            file.entries.push(entry(&format!("Part{k}"), model(k), Some(0), [k as f32, 0.0, 0.0]));
        }
        file.entries[2].cast_shadows = zestengine_core::ShadowCasting::ShadowsOnly;
        file.compact_models(&|_| Some(parts()));
        assert_eq!(file.entries.len(), 2, "root + the part that no longer casts like the model");
        assert_eq!(file.entries[1].cast_shadows, zestengine_core::ShadowCasting::ShadowsOnly);
    }

    #[test]
    fn expanded_parts_carry_a_default_renderer_and_compaction_drops_it_again() {
        let mut file = SceneFile::default();
        let mut root = entry("House", ObjectSource::Empty, None, [0.0; 3]);
        root.model = Some(ModelInstanceRef { path: "models/house.glb".into(), model: None, removed: Vec::new() });
        file.entries.push(root);
        file.expand_models(&|_| Some(parts()));
        assert_eq!(file.entries.len(), 5);
        for part in &file.entries[1..] {
            assert!(part.renderer_split && part.components.len() == 1 && part.components[0].kind == "Mesh Renderer");
        }
        file.entries[0].model = None; // the live scene (entry_of) does not carry the marker
        file.compact_models(&|_| Some(parts()));
        assert_eq!(file.entries.len(), 1, "untouched parts, renderer included, are not stored");
    }

/// One renderer edit applied to an expanded model part: (slot index, edit).
type RendererEdit = (usize, Box<dyn Fn(&mut zestengine_core::MeshRenderer)>);

    #[test]
    fn an_edited_renderer_keeps_its_part_stored() {
        let mut file = SceneFile::default();
        let mut root = entry("House", ObjectSource::Empty, None, [0.0; 3]);
        root.model = Some(ModelInstanceRef { path: "models/house.glb".into(), model: None, removed: Vec::new() });
        file.entries.push(root);
        file.expand_models(&|_| Some(parts()));
        let edits: [RendererEdit; 3] = [
            (1, Box::new(|r| r.cast_shadows = zestengine_core::ShadowCasting::Off)),
            (2, Box::new(|r| r.lightmap_scale = 2.0)),
            (3, Box::new(|r| r.materials[0] = zestengine_core::MaterialSlot::File("m/a.material".into()))),
        ];
        for (at, edit) in &edits {
            let mut r: zestengine_core::MeshRenderer = ron::from_str(&file.entries[*at].components[0].data).unwrap();
            edit(&mut r);
            file.entries[*at].components[0].data = ron::to_string(&r).unwrap();
        }
        file.entries[0].model = None;
        file.compact_models(&|_| Some(parts()));
        assert_eq!(file.entries.len(), 4, "root + the three parts whose renderer was edited; part 4 is dropped");
        // Round trip: nothing duplicated.
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let mut back: SceneFile = ron::from_str(&text).unwrap();
        back.expand_models(&|_| Some(parts()));
        assert_eq!(back.entries.len(), 5);
        assert!(back.entries.iter().skip(1).all(|e| e.components.iter().filter(|c| c.kind == "Mesh Renderer").count() == 1));
    }

    #[test]
    fn a_model_with_hundreds_of_untouched_parts_stays_small() {
        let many: Vec<ModelPart> = (0..500)
            .map(|k| ModelPart {
                name: format!("Part{k}"),
                mesh: k,
                extra_meshes: vec![k + 1000],
                model: None,
                mesh_id: SubAssetId::NONE,
                extra_mesh_ids: Vec::new(),
                part_id: crate::sub_asset::hashed_id(SubAssetKind::Part, &format!("Part{k}")),
                translation: [k as f32, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            })
            .collect();
        let mut file = SceneFile::default();
        let mut root = entry("Town", ObjectSource::Empty, None, [0.0; 3]);
        root.model = Some(ModelInstanceRef { path: "models/town.glb".into(), model: None, removed: Vec::new() });
        file.entries.push(root);
        file.expand_models(&|_| Some(many.clone()));
        file.entries[0].model = None;
        file.compact_models(&|_| Some(many.clone()));
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let mut bare = SceneFile::default();
        bare.entries.push(entry("Town", ObjectSource::Empty, None, [0.0; 3]));
        let bare_len = ron::ser::to_string_pretty(&bare, ron::ser::PrettyConfig::default()).unwrap().len();
        assert!(text.len() < bare_len + 300, "500 untouched parts add only the model reference: {} vs {bare_len} bytes", text.len());
    }

    #[test]
    fn renderer_materials_are_found_by_the_dependency_scan_of_a_build() {
        let mut file = SceneFile::default();
        let mut e = entry("Hero", ObjectSource::Primitive { shape: PrimitiveShape::Cube }, None, [0.0; 3]);
        e.set_renderer_materials(&[MaterialRef::File("materials/hero.material".into())]);
        file.entries.push(e);
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let mut seen = Vec::new();
        crate::package::visit_ron_strings(&text, &mut |s| seen.push(s.to_owned()));
        assert!(seen.iter().any(|s| s == "materials/hero.material"), "the .material is reachable from the scene text only through the component: {seen:?}");
        // An inline snapshot's texture is reachable too (a string inside a string inside the component).
        let inline: MaterialRef = ron::from_str("Inline(params: (base_color: (1.0, 1.0, 1.0, 1.0), emission_color: (0.0, 0.0, 0.0), emission_intensity: 1.0, metallic: 0.0, roughness: 0.5, toon_steps: 3.0, pbr_toon_blend: 1.0, alpha_cutoff: 0.5, normal_strength: 1.0, occlusion_strength: 1.0), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false), shader: \"standard\", textures: (albedo: Some(\"textures/hero.png\")))").unwrap();
        file.entries[0].set_renderer_materials(std::slice::from_ref(&inline));
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let mut seen = Vec::new();
        crate::package::visit_ron_strings(&text, &mut |s| seen.push(s.to_owned()));
        assert!(seen.iter().any(|s| s == "textures/hero.png"), "{seen:?}");
        assert!(matches!(crate::renderer_migration::material_refs(&file.entries[0])[0], MaterialRef::Inline { .. }));
    }

    #[test]
    fn cast_shadows_is_omitted_when_on_and_round_trips() {
        use zestengine_core::{ShadowCasting, renderer_of};
        let reg = zestengine_core::ComponentRegistry::with_builtins();
        let cast_of = |e: &SceneEntry| {
            let c = e.components.iter().find(|c| zestengine_core::is_renderer_kind(&c.kind)).and_then(|c| c.load(&reg)).expect("a mesh entry has a renderer");
            renderer_of(c.as_ref()).unwrap().cast_shadows()
        };
        let mut file = SceneFile::default();
        file.entries.push(entry("Plane", ObjectSource::Primitive { shape: PrimitiveShape::Plane }, None, [0.0; 3]));
        // The loose field is never written when On (older scenes look like this text), and a mesh
        // entry gets its renderer when loaded.
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        assert!(!text.contains("cast_shadows"));
        let back: SceneFile = ron::from_str(&text).unwrap();
        assert_eq!(cast_of(&back.entries[0]), ShadowCasting::On);

        // An older file with a loose `cast_shadows` moves it into the renderer on load.
        for mode in [ShadowCasting::Off, ShadowCasting::TwoSided, ShadowCasting::ShadowsOnly] {
            file.entries[0].cast_shadows = mode;
            let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
            assert!(text.contains("cast_shadows"));
            let back: SceneFile = ron::from_str(&text).unwrap();
            assert_eq!(cast_of(&back.entries[0]), mode);
            assert!(back.entries[0].cast_shadows.is_on(), "the loose field is cleared");
            // ...and the migrated file round-trips through the component alone.
            let again: SceneFile = ron::from_str(&ron::ser::to_string_pretty(&back, ron::ser::PrettyConfig::default()).unwrap()).unwrap();
            assert_eq!(cast_of(&again.entries[0]), mode);
            assert_eq!(again.entries[0].components.len(), 1, "no duplicate renderer");
        }
    }
}

/// Meta GUID -> project-relative path (forward slashes) of every model with a
/// `.meta` under `root`. Skips dot-directories and `target`.
fn model_guid_index(root: &Path) -> HashMap<ObjectId, String> {
    fn walk(dir: &Path, root: &Path, out: &mut HashMap<ObjectId, String>) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        for entry in read.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                if !name.starts_with('.') && name != "target" {
                    walk(&path, root, out);
                }
            } else if let Some(model) = name.strip_suffix(".meta").map(|m| path.with_file_name(m))
                && crate::is_model_file(&model)
                && let Ok(Some(meta)) = crate::load_asset_meta(&model)
                && let Ok(rel) = model.strip_prefix(root)
            {
                out.insert(meta.id, rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut out = HashMap::new();
    walk(root, root, &mut out);
    out
}

#[cfg(test)]
mod legacy_sun_tests {
    use super::*;
    use zestengine_core::{LightComponent, LightKind, LightMode};

    const OLD_SCENE: &str = r#"(entries:[(name:"Cube",source:Primitive(shape:Cube),material:Model,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0))],environment:(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,sun_enabled:true,sun_color:(1.0,0.5,0.25),sun_intensity:4.5,sun_direction:(0.3,-1.0,0.2),skybox_enabled:true,realtime_gi:false))"#;

    fn lights(f: &SceneFile) -> Vec<(usize, LightComponent)> {
        f.entries
            .iter()
            .enumerate()
            .flat_map(|(i, e)| e.components.iter().filter(|c| c.kind == "Light").map(move |c| (i, ron::from_str::<LightComponent>(&c.data).unwrap())))
            .collect()
    }

    #[test]
    fn enabled_sun_becomes_a_directional_light_object() {
        let mut f: SceneFile = ron::from_str(OLD_SCENE).unwrap();
        assert!(f.environment.sun_enabled);
        assert!(f.migrate_environment_sun());
        assert!(!f.environment.sun_enabled);
        let l = lights(&f);
        assert_eq!(l.len(), 1);
        let (i, light) = &l[0];
        assert_eq!(f.entries[*i].name, "Directional Light");
        assert!(matches!(f.entries[*i].source, ObjectSource::Light));
        assert_eq!(light.kind, LightKind::Directional);
        assert_eq!(light.color, [1.0, 0.5, 0.25]);
        assert_eq!(light.intensity, 4.5);
        assert_eq!(light.mode, LightMode::Mixed);
        // The rotation points -Z along the old sun direction.
        let q = glam::Quat::from_array(f.entries[*i].rotation);
        let want = glam::Vec3::new(0.3, -1.0, 0.2).normalize();
        assert!((q * glam::Vec3::NEG_Z).distance(want) < 1e-5);
    }

    #[test]
    fn scene_without_sun_stays_lightless() {
        let text = OLD_SCENE.replace("sun_enabled:true", "sun_enabled:false");
        let mut f: SceneFile = ron::from_str(&text).unwrap();
        assert!(!f.migrate_environment_sun());
        assert!(lights(&f).is_empty());
        assert_eq!(f.entries.len(), 1);
    }

    #[test]
    fn explicit_directional_light_wins_over_the_sun() {
        let mut f: SceneFile = ron::from_str(OLD_SCENE).unwrap();
        f.entries.push(default_directional_light_entry());
        assert!(f.migrate_environment_sun(), "the sun is dropped, so the file changed");
        assert!(!f.environment.sun_enabled);
        assert_eq!(lights(&f).len(), 1, "no second light");
        assert_eq!(f.entries.len(), 2);
    }

    #[test]
    fn disabled_directional_light_does_not_count() {
        let mut f: SceneFile = ron::from_str(OLD_SCENE).unwrap();
        let mut e = default_directional_light_entry();
        e.enabled = false;
        f.entries.push(e);
        assert!(f.migrate_environment_sun());
        assert_eq!(lights(&f).len(), 2);
    }

    #[test]
    fn point_light_does_not_stop_the_sun_migrating() {
        let mut f: SceneFile = ron::from_str(OLD_SCENE).unwrap();
        let mut e = default_directional_light_entry();
        e.components = vec![ComponentData::of(&LightComponent { kind: LightKind::Point, ..Default::default() })];
        f.entries.push(e);
        assert!(f.migrate_environment_sun());
        assert_eq!(lights(&f).len(), 2);
    }

    #[test]
    fn repeated_load_save_does_not_duplicate_the_light() {
        let mut f: SceneFile = ron::from_str(OLD_SCENE).unwrap();
        for _ in 0..3 {
            f.migrate_environment_sun();
            let text = ron::ser::to_string_pretty(&f, ron::ser::PrettyConfig::default()).unwrap();
            assert!(!text.contains("sun_enabled"), "migrated scenes do not write the legacy sun");
            f = ron::from_str(&text).unwrap();
        }
        assert_eq!(lights(&f).len(), 1);
        assert_eq!(f.entries.len(), 2);
        assert!(!f.migrate_environment_sun());
    }

    #[test]
    fn environment_serde_old_and_new() {
        // New form: no sun fields at all.
        let new: WorldEnvironment = ron::from_str("(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,skybox_enabled:true,realtime_gi:false)").unwrap();
        assert!(!new.sun_enabled);
        assert_eq!(new.sun_color, [1.0, 0.98, 0.92]);
        // Old form keeps its values.
        let old: WorldEnvironment = ron::from_str("(ambient:(0.1,0.1,0.1),ambient_intensity:1.0,sun_enabled:true,sun_color:(0.5,0.5,0.5),sun_intensity:2.0,sun_direction:(0.0,-1.0,0.0),skybox_enabled:true,realtime_gi:false)").unwrap();
        assert!(old.sun_enabled && old.sun_intensity == 2.0 && old.sun_color == [0.5; 3]);
        // The default environment has no sun and does not write one.
        let d = WorldEnvironment::default();
        assert!(!d.sun_enabled);
        let text = ron::to_string(&d).unwrap();
        assert!(!text.contains("sun_"), "{text}");
    }

    #[test]
    fn new_default_scene_has_the_default_light() {
        let f = SceneFile::new_default();
        assert!(!f.environment.sun_enabled);
        let l = lights(&f);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].1.color, [1.0, 0.98, 0.92]);
        assert_eq!(l[0].1.intensity, 3.0);
        assert!(f.has_directional_light());
        let mut g = f.clone();
        assert!(!g.migrate_environment_sun());
    }
}

#[cfg(test)]
mod scene_view_tests {
    use super::*;

    #[test]
    fn old_files_load_and_default_flags_are_not_written() {
        let f: SceneFile = ron::from_str("(entries: [])").unwrap();
        assert!(f.scene_hidden.is_empty() && f.scene_locked.is_empty());
        let text = ron::to_string(&f).unwrap();
        assert!(!text.contains("scene_hidden") && !text.contains("scene_locked"), "{text}");
    }

    #[test]
    fn hide_and_lock_round_trip_and_builds_strip_them() {
        let mut f = SceneFile { scene_hidden: vec!["a".into()], scene_locked: vec!["b".into()], collapsed: vec!["c".into()], ..Default::default() };
        let back: SceneFile = ron::from_str(&ron::to_string(&f).unwrap()).unwrap();
        assert_eq!((back.scene_hidden, back.scene_locked), (vec!["a".to_owned()], vec!["b".to_owned()]));
        f.strip_editor_view();
        let text = ron::to_string(&f).unwrap();
        assert!(!text.contains("scene_hidden") && !text.contains("scene_locked") && !text.contains("collapsed"), "{text}");
    }

    #[test]
    fn entries_carry_no_view_flags_so_prefabs_cannot_hold_them() {
        let e: SceneEntry = ron::from_str(r#"(name: "O", source: Empty, material: Model, translation: (0.0, 0.0, 0.0), rotation: (0.0, 0.0, 0.0, 1.0), scale: (1.0, 1.0, 1.0))"#).unwrap();
        let text = ron::to_string(&e).unwrap();
        assert!(!text.contains("hidden") && !text.contains("locked"), "{text}");
    }
}
