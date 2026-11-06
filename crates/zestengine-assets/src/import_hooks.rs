//! Model import hooks (IMP-6): the Rust counterpart of Unity's
//! `AssetPostprocessor`, registered from trusted editor plugins.
//!
//! A [`ModelImportHook`] gets ordered callbacks while a model is imported
//! (settings, per material, hierarchy, per-node user properties, per clip, the
//! finished model) and may edit the small `Imported*` views it is handed. The
//! views are plain std data (strings, vectors, arrays), so nothing of the
//! engine's internals crosses the plugin boundary, like the `Component` trait.
//!
//! Rules:
//! - **Editor only, never UGC.** Hooks are native code from a project plugin.
//!   They run only after the editor's plugin loader enabled them
//!   ([`set_editor_hooks_enabled`]), never in the standalone player, never for
//!   a path under an untrusted root ([`mark_untrusted_root`]; content-package
//!   caches register theirs) and never inside [`suppress_hooks`].
//! - **Deterministic.** The result is cached (the hooks run inside the importer,
//!   before the artifact is stored) under a key that holds every hook's
//!   `(name, order, version)`. A hook must depend only on the callback
//!   arguments and on files read through [`ImportContext::read_project_file`]
//!   (those are recorded as cache dependencies). Time, randomness, environment
//!   and global state are out. Bump [`ModelImportHook::version`] whenever the
//!   hook's behaviour changes, or stale artifacts are reused.
//! - **Isolated.** Each callback runs under `catch_unwind` on a copy of its
//!   view; a panic logs, discards that callback's edits and skips the hook for
//!   the rest of that import. The model still imports.
//! - **Ordered** by `(order, name)`. Unity runs `order <= 0` before its import
//!   blocks and `order >= 1` after; ZestEngine has no import blocks yet, so the
//!   order is a plain sort and every hook sees the edits of the earlier ones.
//!
//! The views describe the model *after* the mesh pipeline (indexed, welded
//! meshes), because that is what the artifact stores.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use glam::{Mat4, Quat, Vec3};

use crate::skeleton::{AnimChannel, AnimationClip, ChannelPath};
use crate::{ImportedCamera, ImportedLight, ModelImport, Scene};

/// Version of the plugin-facing hook API (trait and views). A plugin may export
/// `zestengine_import_hook_api_version() -> u32`; the host registers its hooks
/// only when it equals this. Bump when the trait or a view changes shape.
pub const IMPORT_HOOK_API_VERSION: u32 = 1;

// --- the trait ----------------------------------------------------------------

/// Which optional data a hook wants gathered (Unity's "Has..." probes), so work
/// nobody listens to is skipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HookMask(pub u32);

impl HookMask {
    pub const NONE: Self = Self(0);
    /// The hook implements [`ModelImportHook::user_properties`]: read the
    /// file's custom node properties (glTF `extras`) and call it.
    pub const USER_PROPERTIES: Self = Self(1);

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// An import callback set. Every method but the identity ones is optional.
/// Hooks are created by a plugin, boxed and registered through
/// [`ImportHookRegistry::register_import_hook`].
pub trait ModelImportHook: Send {
    /// Stable unique name (shown in the inspector, part of the cache key).
    fn name(&self) -> &str;
    /// Bump when the behaviour changes: it is part of the cache key.
    fn version(&self) -> u32;
    /// Run order, ascending; ties break by name.
    fn order(&self) -> i32 {
        0
    }
    /// Optional data this hook needs gathered.
    fn wants(&self) -> HookMask {
        HookMask::NONE
    }

    /// Before the file is read: edit the effective import settings.
    fn preprocess(&mut self, _ctx: &ImportContext, _settings: &mut ModelImport) {}
    /// Per material, before it is assigned: return overrides for its values.
    fn describe_material(&mut self, _ctx: &ImportContext, _source: &SourceMaterial) -> Option<MaterialDescription> {
        None
    }
    /// Per material: change its name or default `.material` assignment.
    fn assign_material(&mut self, _ctx: &ImportContext, _source: &SourceMaterial, _current: &mut MaterialAssignment) {}
    /// Once, over every node: rename, retransform or remove.
    fn postprocess_hierarchy(&mut self, _ctx: &ImportContext, _nodes: &mut ImportedHierarchy) {}
    /// Per node that has custom properties (needs [`HookMask::USER_PROPERTIES`]).
    fn user_properties(&mut self, _ctx: &ImportContext, _node: &mut ImportedNode, _props: &UserProps) {}
    /// Per animation clip.
    fn postprocess_clip(&mut self, _ctx: &ImportContext, _clip: &mut ImportedClip) {}
    /// Last: the finished model's names, cameras and lights.
    fn postprocess(&mut self, _ctx: &ImportContext, _model: &mut ImportedModel) {}
}

/// What a plugin fills from its `zestengine_register_import_hooks` export.
#[derive(Default)]
pub struct ImportHookRegistry {
    hooks: Vec<Box<dyn ModelImportHook>>,
}

impl ImportHookRegistry {
    pub fn register_import_hook(&mut self, hook: Box<dyn ModelImportHook>) {
        self.hooks.push(hook);
    }

    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}

// --- context ------------------------------------------------------------------

/// Severity of a hook log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// One line a hook logged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HookLog {
    pub hook: String,
    pub level: LogLevel,
    pub message: String,
}

/// What a callback may know about the import.
pub struct ImportContext {
    asset_path: PathBuf,
    project_root: Option<PathBuf>,
    settings: ModelImport,
    current: RefCell<String>,
    log: RefCell<Vec<HookLog>>,
    disabled: RefCell<HashSet<usize>>,
}

impl ImportContext {
    pub fn new(asset_path: &Path, settings: &ModelImport) -> Self {
        let project_root = asset_path
            .ancestors()
            .skip(1)
            .find(|d| d.join(crate::project_file::PROJECT_FILE_NAME).is_file())
            .map(Path::to_path_buf);
        Self {
            asset_path: asset_path.to_path_buf(),
            project_root,
            settings: settings.clone(),
            current: RefCell::new(String::new()),
            log: RefCell::new(Vec::new()),
            disabled: RefCell::new(HashSet::new()),
        }
    }

    /// The model file being imported.
    pub fn asset_path(&self) -> &Path {
        &self.asset_path
    }

    /// The model's path relative to the project root, with `/` separators.
    pub fn project_path(&self) -> Option<String> {
        let rel = self.asset_path.strip_prefix(self.project_root.as_ref()?).ok()?;
        Some(rel.to_string_lossy().replace('\\', "/"))
    }

    /// The model's `.meta` settings as saved (before any `preprocess` edit).
    pub fn settings(&self) -> &ModelImport {
        &self.settings
    }

    /// Read a file by project-relative path. Absolute paths and `..` are
    /// refused. The file becomes a cache dependency: editing it re-imports.
    pub fn read_project_file(&self, rel: &str) -> std::io::Result<Vec<u8>> {
        let denied = |why: &str| std::io::Error::new(std::io::ErrorKind::PermissionDenied, why.to_owned());
        let root = self.project_root.as_ref().ok_or_else(|| denied("the model is not inside a project"))?;
        let rel_path = Path::new(rel);
        if rel_path.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
            return Err(denied("path must be project-relative without `..`"));
        }
        let full = root.join(rel_path);
        crate::import_cache::record_dependency(&full);
        std::fs::read(full)
    }

    pub fn log_info(&self, message: impl Into<String>) {
        self.push_log(LogLevel::Info, message.into());
    }
    pub fn log_warning(&self, message: impl Into<String>) {
        self.push_log(LogLevel::Warn, message.into());
    }
    pub fn log_error(&self, message: impl Into<String>) {
        self.push_log(LogLevel::Error, message.into());
    }

    fn push_log(&self, level: LogLevel, message: String) {
        let hook = self.current.borrow().clone();
        match level {
            LogLevel::Info => tracing::info!("import hook {hook} ({}): {message}", self.asset_path.display()),
            LogLevel::Warn => tracing::warn!("import hook {hook} ({}): {message}", self.asset_path.display()),
            LogLevel::Error => tracing::error!("import hook {hook} ({}): {message}", self.asset_path.display()),
        }
        self.log.borrow_mut().push(HookLog { hook, level, message });
    }

    /// Everything the hooks logged during this import.
    pub fn logs(&self) -> Vec<HookLog> {
        self.log.borrow().clone()
    }
}

// --- the views ----------------------------------------------------------------

/// A material as the file has it (read-only).
#[derive(Clone, Debug, PartialEq)]
pub struct SourceMaterial {
    pub index: usize,
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emission_color: [f32; 3],
    pub emission_intensity: f32,
    pub has_albedo_texture: bool,
    pub has_normal_texture: bool,
    pub has_orm_texture: bool,
    pub has_emission_texture: bool,
}

/// Overrides a hook returns from `describe_material`; `None` keeps the value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialDescription {
    pub name: Option<String>,
    pub base_color: Option<[f32; 4]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
    pub emission_color: Option<[f32; 3]>,
    pub emission_intensity: Option<f32>,
}

/// What a material becomes. `name` is the (unique-ified later) material name;
/// `material_path` is a project-relative `.material` used instead of the
/// imported one, unless the model's own remap table names one for it.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialAssignment {
    pub name: String,
    pub material_path: Option<String>,
}

/// One scene node (an instance), parallel to the model's render list.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedNode {
    pub name: String,
    /// Column-major local-to-model transform.
    pub transform: [f32; 16],
    /// Mesh/material slots on the node (read-only).
    pub mesh_slots: usize,
    /// Drop the node from the model.
    pub remove: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedHierarchy {
    pub nodes: Vec<ImportedNode>,
}

/// A custom node property (glTF `extras`).
#[derive(Clone, Debug, PartialEq)]
pub enum PropValue {
    Bool(bool),
    Number(f64),
    Text(String),
}

/// A node's custom properties, in file order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserProps(pub Vec<(String, PropValue)>);

impl UserProps {
    pub fn get(&self, key: &str) -> Option<&PropValue> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelKind {
    Translation,
    Rotation,
    Scale,
}

/// One animated joint property. `values` has one `[x, y, z, w]` per key
/// (`w` unused for translation and scale).
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedChannel {
    /// Index into the model's first skeleton's joints.
    pub joint: usize,
    pub kind: ChannelKind,
    pub times: Vec<f32>,
    pub values: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<ImportedChannel>,
}

/// The finished model's names and extras.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedModel {
    /// Per mesh part, parallel to the model's meshes. Renaming changes the
    /// sub-asset ids derived from the name.
    pub mesh_names: Vec<String>,
    /// Per material, parallel to the model's materials.
    pub material_names: Vec<String>,
    pub cameras: Vec<ImportedCamera>,
    pub lights: Vec<ImportedLight>,
}

// --- views over the scene -------------------------------------------------------

fn source_materials(scene: &Scene) -> Vec<SourceMaterial> {
    scene
        .materials
        .iter()
        .enumerate()
        .map(|(index, m)| SourceMaterial {
            index,
            name: m.name.clone(),
            base_color: m.params.base_color,
            metallic: m.params.metallic,
            roughness: m.params.roughness,
            emission_color: m.params.emission_color,
            emission_intensity: m.params.emission_intensity,
            has_albedo_texture: m.albedo.is_some(),
            has_normal_texture: m.normal.is_some(),
            has_orm_texture: m.orm.is_some(),
            has_emission_texture: m.emission.is_some(),
        })
        .collect()
}

fn hierarchy_view(scene: &Scene) -> ImportedHierarchy {
    ImportedHierarchy {
        nodes: scene
            .instances
            .iter()
            .map(|i| ImportedNode { name: i.name.clone(), transform: i.transform.to_cols_array(), mesh_slots: i.slots.len(), remove: false })
            .collect(),
    }
}

/// Write an edited hierarchy back. Removed nodes leave the render list; authored
/// collision pieces that targeted one are dropped and the others re-pointed.
fn apply_hierarchy(scene: &mut Scene, view: &ImportedHierarchy) {
    if view.nodes.len() != scene.instances.len() {
        tracing::warn!("import hook changed the node count ({} -> {}); hierarchy edit ignored", scene.instances.len(), view.nodes.len());
        return;
    }
    for (inst, node) in scene.instances.iter_mut().zip(&view.nodes) {
        inst.name.clone_from(&node.name);
        inst.transform = Mat4::from_cols_array(&node.transform);
    }
    if view.nodes.iter().any(|n| n.remove) {
        let mut remap = Vec::with_capacity(view.nodes.len());
        let mut next = 0usize;
        for n in &view.nodes {
            remap.push((!n.remove).then(|| {
                next += 1;
                next - 1
            }));
        }
        let mut keep = view.nodes.iter().map(|n| !n.remove);
        scene.instances.retain(|_| keep.next().unwrap_or(true));
        scene.collision.retain_mut(|c| match c.target {
            None => true,
            Some(t) => match remap.get(t).copied().flatten() {
                Some(new) => {
                    c.target = Some(new);
                    true
                }
                None => false,
            },
        });
    }
}

fn clip_view(clip: &AnimationClip) -> ImportedClip {
    ImportedClip {
        name: clip.name.clone(),
        duration: clip.duration,
        channels: clip
            .channels
            .iter()
            .map(|c| match c.path {
                ChannelPath::Rotation => ImportedChannel { joint: c.joint, kind: ChannelKind::Rotation, times: c.times.clone(), values: (0..c.len()).filter_map(|i| c.quat_at(i)).map(|q| q.to_array()).collect() },
                path => ImportedChannel {
                    joint: c.joint,
                    kind: if path == ChannelPath::Translation { ChannelKind::Translation } else { ChannelKind::Scale },
                    times: c.times.clone(),
                    values: (0..c.len()).filter_map(|i| c.vec_at(i)).map(|v| [v.x, v.y, v.z, 0.0]).collect(),
                },
            })
            .collect(),
    }
}

fn apply_clip(clip: &mut AnimationClip, view: &ImportedClip) {
    clip.name.clone_from(&view.name);
    clip.duration = view.duration;
    clip.channels = view
        .channels
        .iter()
        .filter(|c| c.times.len() == c.values.len())
        .map(|c| {
            let mut ch = AnimChannel {
                joint: c.joint,
                path: match c.kind {
                    ChannelKind::Translation => ChannelPath::Translation,
                    ChannelKind::Rotation => ChannelPath::Rotation,
                    ChannelKind::Scale => ChannelPath::Scale,
                },
                times: c.times.clone(),
                vec_values: Vec::new(),
                quat_values: Vec::new(), extra: None
            };
            if c.kind == ChannelKind::Rotation {
                ch.quat_values = c.values.iter().map(|v| Quat::from_array(*v)).collect();
            } else {
                ch.vec_values = c.values.iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect();
            }
            ch
        })
        .collect();
}

fn model_view(scene: &Scene) -> ImportedModel {
    ImportedModel {
        mesh_names: scene.identity.mesh_names.clone(),
        material_names: scene.materials.iter().map(|m| m.name.clone()).collect(),
        cameras: scene.identity.cameras.clone(),
        lights: scene.identity.lights.clone(),
    }
}

fn apply_model(scene: &mut Scene, view: &ImportedModel) {
    if view.mesh_names.len() == scene.meshes.len() {
        scene.identity.mesh_names.clone_from(&view.mesh_names);
    }
    if view.material_names.len() == scene.materials.len() {
        for (m, n) in scene.materials.iter_mut().zip(&view.material_names) {
            m.name.clone_from(n);
        }
    }
    scene.identity.cameras.clone_from(&view.cameras);
    scene.identity.lights.clone_from(&view.lights);
}

/// glTF node `extras` by node name, flat scalar entries only. Best effort: a
/// file that does not parse, or has no extras, yields nothing.
fn gltf_node_props(path: &Path) -> HashMap<String, UserProps> {
    let mut out = HashMap::new();
    let is_gltf = path.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "gltf" | "glb" | "vrm"));
    if !is_gltf {
        return out;
    }
    let Ok(doc) = gltf::Gltf::open(path) else { return out };
    for node in doc.nodes() {
        let (Some(name), Some(extras)) = (node.name(), node.extras().as_ref()) else { continue };
        let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(extras.get()) else { continue };
        let props: Vec<(String, PropValue)> = map
            .into_iter()
            .filter_map(|(k, v)| {
                let v = match v {
                    serde_json::Value::Bool(b) => PropValue::Bool(b),
                    serde_json::Value::Number(n) => PropValue::Number(n.as_f64()?),
                    serde_json::Value::String(s) => PropValue::Text(s),
                    _ => return None,
                };
                Some((k, v))
            })
            .collect();
        if !props.is_empty() {
            out.insert(name.to_owned(), UserProps(props));
        }
    }
    out
}

// --- the active set -------------------------------------------------------------

struct HookSlot {
    owner: String,
    name: String,
    version: u32,
    order: i32,
    wants: HookMask,
    hook: Mutex<Box<dyn ModelImportHook>>,
}

/// An ordered set of hooks: what one import runs.
#[derive(Clone, Default)]
pub struct HookSet {
    slots: Vec<Arc<HookSlot>>,
}

/// Name, version and order of an installed hook (the inspector's list).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HookInfo {
    pub owner: String,
    pub name: String,
    pub version: u32,
    pub order: i32,
}

impl HookSet {
    /// A set from loose hooks (`owner` labels them), ordered by `(order, name)`.
    pub fn new(owner: &str, hooks: Vec<Box<dyn ModelImportHook>>) -> Self {
        let mut slots: Vec<Arc<HookSlot>> = hooks
            .into_iter()
            .map(|h| {
                Arc::new(HookSlot {
                    owner: owner.to_owned(),
                    name: h.name().to_owned(),
                    version: h.version(),
                    order: h.order(),
                    wants: h.wants(),
                    hook: Mutex::new(h),
                })
            })
            .collect();
        slots.sort_by(|a, b| (a.order, &a.name).cmp(&(b.order, &b.name)));
        Self { slots }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn infos(&self) -> Vec<HookInfo> {
        self.slots.iter().map(|s| HookInfo { owner: s.owner.clone(), name: s.name.clone(), version: s.version, order: s.order }).collect()
    }

    /// `(name#order, version)` in run order: what the cache key hashes.
    pub fn key_entries(&self) -> Vec<(String, u32)> {
        self.slots.iter().map(|s| (format!("{}#{}", s.name, s.order), s.version)).collect()
    }

    /// Readable form of [`Self::key_entries`], stored in the artifact.
    pub fn signature(&self) -> String {
        self.slots.iter().map(|s| format!("{}#{}@{}", s.name, s.order, s.version)).collect::<Vec<_>>().join(";")
    }

    fn wants(&self, mask: HookMask) -> bool {
        self.slots.iter().any(|s| s.wants.contains(mask))
    }

    /// Run one callback of hook `i` on `view`, a copy that only replaces the
    /// real one when the callback returned. A panic logs, drops the edits and
    /// disables the hook for the rest of the import.
    fn call<V: Clone + PartialEq, R>(&self, i: usize, ctx: &ImportContext, view: &mut V, f: impl FnOnce(&mut dyn ModelImportHook, &mut V) -> R) -> Option<R> {
        if ctx.disabled.borrow().contains(&i) {
            return None;
        }
        let slot = &self.slots[i];
        *ctx.current.borrow_mut() = slot.name.clone();
        let mut copy = view.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut hook = slot.hook.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut **hook, &mut copy)
        }));
        match result {
            Ok(r) => {
                if copy != *view {
                    *view = copy;
                }
                Some(r)
            }
            Err(payload) => {
                let why = payload.downcast_ref::<&str>().map(|s| (*s).to_owned()).or_else(|| payload.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into());
                ctx.log_error(format!("panicked ({why}); its edits are discarded and it is skipped for this import"));
                ctx.disabled.borrow_mut().insert(i);
                None
            }
        }
    }

    /// Run `preprocess` for every hook on the effective settings.
    pub fn preprocess(&self, ctx: &ImportContext, settings: &mut ModelImport) {
        let mut view = SettingsBox(settings.clone());
        for i in 0..self.slots.len() {
            self.call(i, ctx, &mut view, |h, s| h.preprocess(ctx, &mut s.0));
        }
        *settings = view.0;
    }

    /// Run every scene callback in Unity's order: materials (describe, assign),
    /// then per-node user properties and the hierarchy, then clips, then the
    /// whole model.
    pub fn process_scene(&self, ctx: &ImportContext, scene: &mut Scene) {
        if self.slots.is_empty() {
            return;
        }
        let props = if self.wants(HookMask::USER_PROPERTIES) { gltf_node_props(ctx.asset_path()) } else { HashMap::new() };
        self.process_scene_with_props(ctx, scene, &props);
    }

    /// [`Self::process_scene`] with the nodes' custom properties given (by node name).
    pub(crate) fn process_scene_with_props(&self, ctx: &ImportContext, scene: &mut Scene, props: &HashMap<String, UserProps>) {
        // Materials.
        let sources = source_materials(scene);
        let mut assign = vec![None; scene.materials.len()];
        for (mi, src) in sources.iter().enumerate() {
            let mut src = src.clone();
            let mut current = MaterialAssignment { name: src.name.clone(), material_path: None };
            for i in 0..self.slots.len() {
                let mut desc_in = src.clone();
                if let Some(Some(d)) = self.call(i, ctx, &mut desc_in, |h, s| h.describe_material(ctx, s)) {
                    let m = &mut scene.materials[mi];
                    if let Some(v) = d.base_color {
                        m.params.base_color = v;
                    }
                    if let Some(v) = d.metallic {
                        m.params.metallic = v;
                    }
                    if let Some(v) = d.roughness {
                        m.params.roughness = v;
                    }
                    if let Some(v) = d.emission_color {
                        m.params.emission_color = v;
                    }
                    if let Some(v) = d.emission_intensity {
                        m.params.emission_intensity = v;
                    }
                    if let Some(n) = d.name {
                        m.name.clone_from(&n);
                        current.name = n;
                    }
                    src = source_materials(scene).swap_remove(mi);
                }
                let mut view = (src.clone(), current.clone());
                self.call(i, ctx, &mut view, |h, v| {
                    let (source, current) = v;
                    h.assign_material(ctx, source, current)
                });
                current = view.1;
            }
            scene.materials[mi].name.clone_from(&current.name);
            assign[mi] = current.material_path;
        }
        if assign.iter().any(Option::is_some) {
            scene.identity.hook_material_assign = assign;
        }

        // Nodes: user properties, then the hierarchy.
        let mut nodes = hierarchy_view(scene);
        if self.wants(HookMask::USER_PROPERTIES) {
            for n in 0..nodes.nodes.len() {
                let Some(p) = props.get(&nodes.nodes[n].name) else { continue };
                for i in 0..self.slots.len() {
                    if self.slots[i].wants.contains(HookMask::USER_PROPERTIES) {
                        self.call(i, ctx, &mut nodes.nodes[n], |h, node| h.user_properties(ctx, node, p));
                    }
                }
            }
        }
        for i in 0..self.slots.len() {
            self.call(i, ctx, &mut nodes, |h, v| h.postprocess_hierarchy(ctx, v));
        }
        if nodes != hierarchy_view(scene) {
            apply_hierarchy(scene, &nodes);
        }

        // Clips.
        for c in 0..scene.animations.len() {
            let mut view = clip_view(&scene.animations[c]);
            let before = view.clone();
            for i in 0..self.slots.len() {
                self.call(i, ctx, &mut view, |h, v| h.postprocess_clip(ctx, v));
            }
            if view != before {
                apply_clip(&mut scene.animations[c], &view);
            }
        }

        // The whole model.
        let mut model = model_view(scene);
        let before = model.clone();
        for i in 0..self.slots.len() {
            self.call(i, ctx, &mut model, |h, v| h.postprocess(ctx, v));
        }
        if model != before {
            apply_model(scene, &model);
        }
    }
}

/// `ModelImport` has no `PartialEq`; compare through its JSON text.
#[derive(Clone)]
struct SettingsBox(ModelImport);

impl PartialEq for SettingsBox {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_string(&self.0).ok() == serde_json::to_string(&other.0).ok()
    }
}

// --- the global registry ----------------------------------------------------------

static REGISTRY: RwLock<Vec<Arc<HookSlot>>> = RwLock::new(Vec::new());
static EDITOR_HOOKS: AtomicBool = AtomicBool::new(false);
static UNTRUSTED_ROOTS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

thread_local! {
    static SUPPRESSED: RefCell<u32> = const { RefCell::new(0) };
}

/// Allow hooks to run in this process. Only the editor's plugin loader calls
/// this; a standalone player never does, so hooks never run there.
pub fn set_editor_hooks_enabled(on: bool) {
    EDITOR_HOOKS.store(on, Ordering::SeqCst);
}

pub fn editor_hooks_enabled() -> bool {
    EDITOR_HOOKS.load(Ordering::SeqCst)
}

/// Never run hooks for files under `root` (a content-package cache: user
/// generated content must not reach native plugin code).
pub fn mark_untrusted_root(root: &Path) {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut roots = UNTRUSTED_ROOTS.lock().unwrap_or_else(|e| e.into_inner());
    if !roots.contains(&root) {
        roots.push(root);
    }
}

/// Scope guard: no hooks run on this thread while it lives.
pub struct HooksSuppressed(());

pub fn suppress_hooks() -> HooksSuppressed {
    SUPPRESSED.with(|s| *s.borrow_mut() += 1);
    HooksSuppressed(())
}

impl Drop for HooksSuppressed {
    fn drop(&mut self) {
        SUPPRESSED.with(|s| *s.borrow_mut() -= 1);
    }
}

fn is_untrusted(path: &Path) -> bool {
    let roots = UNTRUSTED_ROOTS.lock().unwrap_or_else(|e| e.into_inner());
    if roots.is_empty() {
        return false;
    }
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    roots.iter().any(|r| path.starts_with(r))
}

/// Whether the hook gate is open for `path` right now. The single place that
/// decides "editor only, never UGC".
pub fn hooks_apply(path: &Path) -> bool {
    gate_open(editor_hooks_enabled(), path)
}

pub(crate) fn gate_open(enabled: bool, path: &Path) -> bool {
    enabled && SUPPRESSED.with(|s| *s.borrow()) == 0 && !is_untrusted(path)
}

/// The hooks an import of `path` runs: empty unless the gate is open.
pub fn hooks_for(path: &Path) -> HookSet {
    if !hooks_apply(path) {
        return HookSet::default();
    }
    HookSet { slots: REGISTRY.read().unwrap_or_else(|e| e.into_inner()).clone() }
}

/// The hooks that would run for `path`, as `(name#order, version)` for the
/// cache key. Empty when the gate is closed.
pub(crate) fn key_entries_for(path: &Path) -> Vec<(String, u32)> {
    hooks_for(path).key_entries()
}

/// Signature of the hooks that run for `path` (empty = none).
pub fn signature_for(path: &Path) -> String {
    hooks_for(path).signature()
}

/// Every installed hook in run order, whatever the gate says (the inspector's list).
pub fn active_hooks() -> Vec<HookInfo> {
    HookSet { slots: REGISTRY.read().unwrap_or_else(|e| e.into_inner()).clone() }.infos()
}

/// Replace `owner`'s hooks with `registry`'s. The cache key reads the registry
/// through [`hooks_for`], so the change invalidates affected artifacts.
pub fn install_plugin_hooks(owner: &str, registry: ImportHookRegistry) {
    let new = HookSet::new(owner, registry.hooks).slots;
    let mut all = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    all.retain(|s| s.owner != owner);
    all.extend(new);
    all.sort_by(|a, b| (a.order, &a.name).cmp(&(b.order, &b.name)));
}

/// Drop every hook `owner` registered (plugin reload or unload).
pub fn remove_plugin_hooks(owner: &str) -> usize {
    let mut all = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    let before = all.len();
    all.retain(|s| s.owner != owner);
    before - all.len()
}

/// Load a plugin's hooks through two optional exports. `api_version` is the
/// plugin's `zestengine_import_hook_api_version()` and `register` its
/// `zestengine_register_import_hooks`; an old plugin has neither and simply
/// contributes none. A plugin built against another API version is refused
/// (its trait layout may differ), the rest of the plugin still loads.
/// Returns how many hooks were installed. Previous hooks of `owner` go first,
/// so a reload replaces them even when the new build has none.
pub fn register_plugin(owner: &str, api_version: Option<u32>, register: Option<&dyn Fn(&mut ImportHookRegistry)>) -> usize {
    remove_plugin_hooks(owner);
    let Some(register) = register else { return 0 };
    if api_version != Some(IMPORT_HOOK_API_VERSION) {
        tracing::warn!("plugin {owner}: import hook API {api_version:?} does not match the editor's ({IMPORT_HOOK_API_VERSION}); its import hooks are not registered");
        return 0;
    }
    let mut reg = ImportHookRegistry::default();
    if catch_unwind(AssertUnwindSafe(|| register(&mut reg))).is_err() {
        tracing::error!("plugin {owner}: zestengine_register_import_hooks panicked; its import hooks are not registered");
        return 0;
    }
    let n = reg.len();
    install_plugin_hooks(owner, reg);
    n
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instance, MeshSlot, SceneMaterial, skeleton::{Joint, Skeleton}};
    use zestengine_render::{MaterialFeatures, MaterialParams, MeshData};

    type Trace = Arc<Mutex<Vec<String>>>;

    fn scene() -> Scene {
        let mat = |n: &str| SceneMaterial { name: n.into(), params: MaterialParams::default(), features: MaterialFeatures::default(), albedo: None, normal: None, orm: None, emission: None };
        let mut scene = Scene {
            meshes: (0..2).map(|_| MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false }).collect(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: vec![mat("Skin"), mat("Cloth")],
            instances: vec![
                Instance { name: "Body".into(), transform: Mat4::IDENTITY, slots: vec![MeshSlot { mesh: 0, material: 0 }] },
                Instance { name: "Body_LOD1".into(), transform: Mat4::IDENTITY, slots: vec![MeshSlot { mesh: 1, material: 1 }] },
                Instance { name: "Prop".into(), transform: Mat4::IDENTITY, slots: Vec::new() },
            ],
            skeletons: vec![Skeleton {
                joints: vec![Joint { name: "Hips".into(), parent: None, inverse_bind: Mat4::IDENTITY, rest_translation: Vec3::ZERO, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE }],
                rig: None,
                ..Default::default()
            }],
            animations: vec![AnimationClip {
                name: "Walk".into(),
                duration: 1.0,
                channels: vec![AnimChannel { joint: 0, path: ChannelPath::Translation, times: vec![0.0, 1.0], vec_values: vec![Vec3::ZERO, Vec3::X], quat_values: Vec::new(), extra: None }],
                morph_tracks: Vec::new(), playback: None,
            }],
            collision: vec![
                crate::collision::CollisionPiece { name: "UCX_Body".into(), kind: crate::collision::CollisionKind::Convex, target: Some(0), points: Vec::new() },
                crate::collision::CollisionPiece { name: "UCX_Lod".into(), kind: crate::collision::CollisionKind::Convex, target: Some(1), points: Vec::new() },
                crate::collision::CollisionPiece { name: "UCX_Prop".into(), kind: crate::collision::CollisionKind::Convex, target: Some(2), points: Vec::new() },
            ],
            identity: Default::default(),
        };
        scene.identity.mesh_names = vec!["Body#part0".into(), "Body_LOD1#part0".into()];
        scene
    }

    fn ctx() -> ImportContext {
        ImportContext::new(Path::new("/nonexistent/m.glb"), &ModelImport::default())
    }

    fn bytes(s: &Scene) -> Vec<u8> {
        crate::import_cache::encode(s, Path::new("/nonexistent/m.glb"), &[], "")
    }

    /// Records its name when `preprocess` runs.
    struct Order(&'static str, i32, Trace);
    impl ModelImportHook for Order {
        fn name(&self) -> &str {
            self.0
        }
        fn version(&self) -> u32 {
            1
        }
        fn order(&self) -> i32 {
            self.1
        }
        fn preprocess(&mut self, _: &ImportContext, _: &mut ModelImport) {
            self.2.lock().unwrap().push(self.0.to_owned());
        }
    }

    #[test]
    fn hooks_run_by_order_then_name() {
        let trace: Trace = Default::default();
        let set = HookSet::new("t", vec![
            Box::new(Order("b", 5, trace.clone())),
            Box::new(Order("z", -1, trace.clone())),
            Box::new(Order("a", 5, trace.clone())),
            Box::new(Order("m", 0, trace.clone())),
        ]);
        set.preprocess(&ctx(), &mut ModelImport::default());
        assert_eq!(*trace.lock().unwrap(), ["z", "m", "a", "b"]);
        assert_eq!(set.signature(), "z#-1@1;m#0@1;a#5@1;b#5@1");
    }

    struct Noop;
    impl ModelImportHook for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn version(&self) -> u32 {
            1
        }
        fn wants(&self) -> HookMask {
            HookMask::USER_PROPERTIES
        }
    }

    #[test]
    fn default_hooks_leave_the_import_byte_identical() {
        let set = HookSet::new("t", vec![Box::new(Noop)]);
        let (mut a, b) = (scene(), scene());
        let mut settings = ModelImport::default();
        set.preprocess(&ctx(), &mut settings);
        assert_eq!(serde_json::to_string(&settings).unwrap(), serde_json::to_string(&ModelImport::default()).unwrap());
        let mut props = HashMap::new();
        props.insert("Body".to_owned(), UserProps(vec![("k".into(), PropValue::Bool(true))]));
        set.process_scene_with_props(&ctx(), &mut a, &props);
        assert_eq!(bytes(&a), bytes(&b));
    }

    /// Edits through every callback.
    struct Everything;
    impl ModelImportHook for Everything {
        fn name(&self) -> &str {
            "everything"
        }
        fn version(&self) -> u32 {
            3
        }
        fn wants(&self) -> HookMask {
            HookMask::USER_PROPERTIES
        }
        fn preprocess(&mut self, _: &ImportContext, s: &mut ModelImport) {
            s.scale = 2.0;
        }
        fn describe_material(&mut self, _: &ImportContext, src: &SourceMaterial) -> Option<MaterialDescription> {
            (src.name == "Skin").then(|| MaterialDescription { roughness: Some(0.25), base_color: Some([1.0, 0.0, 0.0, 1.0]), ..Default::default() })
        }
        fn assign_material(&mut self, _: &ImportContext, src: &SourceMaterial, cur: &mut MaterialAssignment) {
            if src.name == "Cloth" {
                cur.name = "Fabric".into();
                cur.material_path = Some("mats/fabric.material".into());
            }
        }
        fn postprocess_hierarchy(&mut self, _: &ImportContext, h: &mut ImportedHierarchy) {
            for n in &mut h.nodes {
                if n.name.contains("_LOD") {
                    n.remove = true;
                }
            }
        }
        fn user_properties(&mut self, _: &ImportContext, node: &mut ImportedNode, props: &UserProps) {
            if let Some(PropValue::Text(t)) = props.get("tag") {
                node.name = format!("{}@{t}", node.name);
            }
        }
        fn postprocess_clip(&mut self, _: &ImportContext, clip: &mut ImportedClip) {
            clip.name = format!("{}_edited", clip.name);
            clip.channels[0].values[1] = [0.0, 5.0, 0.0, 0.0];
        }
        fn postprocess(&mut self, _: &ImportContext, m: &mut ImportedModel) {
            m.mesh_names[0] = "Renamed#part0".into();
        }
    }

    #[test]
    fn each_callback_edits_its_view_and_the_model_reflects_it() {
        let set = HookSet::new("t", vec![Box::new(Everything)]);
        let mut settings = ModelImport::default();
        set.preprocess(&ctx(), &mut settings);
        assert_eq!(settings.scale, 2.0);

        let mut s = scene();
        let mut props = HashMap::new();
        props.insert("Prop".to_owned(), UserProps(vec![("tag".into(), PropValue::Text("x".into()))]));
        set.process_scene_with_props(&ctx(), &mut s, &props);
        assert_eq!(s.materials[0].params.roughness, 0.25);
        assert_eq!(s.materials[0].params.base_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.materials[1].name, "Fabric");
        assert_eq!(s.identity.hook_material_assign, [None, Some("mats/fabric.material".to_owned())]);
        // The _LOD node is gone, the property rename landed, collision followed.
        assert_eq!(s.instances.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(), ["Body", "Prop@x"]);
        assert_eq!(s.collision.iter().map(|c| (c.name.as_str(), c.target)).collect::<Vec<_>>(), [("UCX_Body", Some(0)), ("UCX_Prop", Some(1))]);
        assert_eq!(s.animations[0].name, "Walk_edited");
        assert_eq!(s.animations[0].channels[0].vec_values[1], Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(s.identity.mesh_names[0], "Renamed#part0");
    }

    #[test]
    fn hook_assigned_materials_survive_the_cache_and_yield_to_the_remap_table() {
        let set = HookSet::new("t", vec![Box::new(Everything)]);
        let mut s = scene();
        set.process_scene_with_props(&ctx(), &mut s, &HashMap::new());
        let src = Path::new("/nonexistent/m.glb");
        let mut back = crate::import_cache::decode(&crate::import_cache::encode(&s, src, &[], &set.signature()), src).unwrap();
        assert_eq!(back.identity.hook_material_assign, s.identity.hook_material_assign);
        crate::assign_identity(&mut back, None, &mut Default::default(), false);
        crate::apply_remap_tables(&mut back, &ModelImport::default());
        assert_eq!(back.identity.material_remap[1].as_deref(), Some("mats/fabric.material"));
        let mut own = ModelImport::default();
        own.material_remap.insert("Fabric".into(), "mine.material".into());
        crate::apply_remap_tables(&mut back, &own);
        assert_eq!(back.identity.material_remap[1].as_deref(), Some("mine.material"));
    }

    struct Panics(Trace);
    impl ModelImportHook for Panics {
        fn name(&self) -> &str {
            "panics"
        }
        fn version(&self) -> u32 {
            1
        }
        fn order(&self) -> i32 {
            -10
        }
        fn postprocess_hierarchy(&mut self, _: &ImportContext, h: &mut ImportedHierarchy) {
            h.nodes[0].name = "half-done".into();
            self.0.lock().unwrap().push("hierarchy".into());
            panic!("boom");
        }
        fn postprocess_clip(&mut self, _: &ImportContext, _: &mut ImportedClip) {
            self.0.lock().unwrap().push("clip".into());
        }
    }

    #[test]
    fn a_panicking_hook_is_isolated_and_skipped() {
        let trace: Trace = Default::default();
        let set = HookSet::new("t", vec![Box::new(Panics(trace.clone())), Box::new(Everything)]);
        let c = ctx();
        let mut s = scene();
        set.process_scene_with_props(&c, &mut s, &HashMap::new());
        // Its half-done rename never landed; the later hook still ran.
        assert_eq!(s.instances[0].name, "Body");
        assert_eq!(s.instances.len(), 2, "Everything still removed the LOD node");
        assert_eq!(s.animations[0].name, "Walk_edited");
        // Skipped after the panic: the clip callback never ran.
        assert_eq!(*trace.lock().unwrap(), ["hierarchy"]);
        let logs = c.logs();
        assert!(logs.iter().any(|l| l.hook == "panics" && l.level == LogLevel::Error && l.message.contains("boom")));
    }

    #[test]
    fn the_gate_is_closed_for_players_ugc_and_suppressed_scopes() {
        let path = Path::new("/definitely/not/there/m.glb");
        assert!(!gate_open(false, path), "never enabled (standalone player)");
        assert!(gate_open(true, path));
        {
            let _off = suppress_hooks();
            assert!(!gate_open(true, path), "extraction scope");
        }
        assert!(gate_open(true, path), "scope ended");
        let ugc = std::env::temp_dir().join(format!("zest_ugc_root_{}", std::process::id()));
        mark_untrusted_root(&ugc);
        assert!(!gate_open(true, &ugc.join("abc/models/m.glb")), "content-package cache");
        assert!(!editor_hooks_enabled(), "a fresh process has hooks off");
        assert!(hooks_for(path).is_empty());
    }

    #[test]
    fn content_package_caches_are_untrusted() {
        let root = std::env::temp_dir().join(format!("zest_ugc_cache_{}", std::process::id()));
        let cache = crate::package::ContentCache::new(&root);
        assert!(!gate_open(true, &cache.dir_for("abc").join("models/m.glb")));
        assert!(gate_open(true, Path::new("/some/project/models/m.glb")));
    }

    #[test]
    fn plugin_registration_replaces_removes_and_tolerates_old_plugins() {
        let owner = format!("plugin-{}", std::process::id());
        let reg = |hooks: fn() -> Vec<Box<dyn ModelImportHook>>| move |r: &mut ImportHookRegistry| {
            for h in hooks() {
                r.register_import_hook(h);
            }
        };
        let mine = |_: &()| active_hooks().iter().filter(|h| h.owner == owner).map(|h| h.name.clone()).collect::<Vec<_>>();
        // An old plugin exports neither symbol: loads, contributes nothing.
        assert_eq!(register_plugin(&owner, None, None), 0);
        assert!(mine(&()).is_empty());
        // A plugin built against another API is refused.
        let one = reg(|| vec![Box::new(Noop)]);
        assert_eq!(register_plugin(&owner, Some(IMPORT_HOOK_API_VERSION + 1), Some(&one)), 0);
        assert!(mine(&()).is_empty());
        // The right version installs.
        assert_eq!(register_plugin(&owner, Some(IMPORT_HOOK_API_VERSION), Some(&one)), 1);
        assert_eq!(mine(&()), ["noop"]);
        // Reload with a build that has no hooks removes the old ones.
        assert_eq!(register_plugin(&owner, None, None), 0);
        assert!(mine(&()).is_empty());
        // Unload.
        register_plugin(&owner, Some(IMPORT_HOOK_API_VERSION), Some(&one));
        assert_eq!(remove_plugin_hooks(&owner), 1);
        assert!(mine(&()).is_empty());
        // A registration function that panics installs nothing.
        let bad = |_: &mut ImportHookRegistry| panic!("bad plugin");
        assert_eq!(register_plugin(&owner, Some(IMPORT_HOOK_API_VERSION), Some(&bad)), 0);
    }

    #[test]
    fn project_files_are_confined_to_the_project() {
        let root = std::env::temp_dir().join(format!("zest_hook_ctx_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::write(root.join(crate::project_file::PROJECT_FILE_NAME), "x").unwrap();
        std::fs::write(root.join("rules.txt"), "hello").unwrap();
        let c = ImportContext::new(&root.join("models/m.glb"), &ModelImport::default());
        assert_eq!(c.project_path().as_deref(), Some("models/m.glb"));
        assert_eq!(c.read_project_file("rules.txt").unwrap(), b"hello");
        assert!(c.read_project_file("../etc/passwd").is_err());
        assert!(c.read_project_file("/etc/passwd").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
