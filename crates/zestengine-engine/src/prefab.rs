//! Prefab / reusable-object template (ENGINE_FEATURE_CHECKLIST T0 #7).
//!
//! A `.prefab` (RON) is a self-contained object subtree (`SceneEntry` list with
//! parent indices local to the prefab; roots have `parent = None`). The editor
//! makes one from a selection; `instantiate` clones it into a scene at a given
//! transform with FRESH ids (so instances don't collide), applying per-instance
//! root transform overrides — Unity/Godot's core "designer-scale content" workflow,
//! replacing the previous "duplicate only".

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use zestengine_assets::{PrefabLink, SceneEntry, SceneFile};
use zestengine_core::ObjectId;
use serde::{Deserialize, Serialize};

/// A reusable object template.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    /// Subtree entries. `parent` indices are local to this list; root entries (the
    /// instantiation anchors) have `parent: None`.
    pub entries: Vec<SceneEntry>,
    /// Layout version (see `SceneFile::format_version`): below 2 the renderer settings are
    /// loose entry fields and loading moves them into `Mesh Renderer` components.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub format_version: u32,
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

impl Prefab {
    /// Build a prefab from a set of scene entries (already re-indexed so parent
    /// references are local to `entries`).
    pub fn new(mut entries: Vec<SceneEntry>) -> Self {
        // Entries built in the old loose-field style are brought to the component form.
        zestengine_assets::migrate_entries(&mut entries, false);
        Self { entries, format_version: zestengine_assets::RENDERER_FORMAT }
    }

    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        let mut p: Self = ron::from_str(s)?;
        p.migrate_renderers();
        Ok(p)
    }

    /// Move the loose renderer fields of an older prefab into `Mesh Renderer` components (and
    /// rewrite overrides that name them). Idempotent; returns true when anything changed.
    pub fn migrate_renderers(&mut self) -> bool {
        let legacy = self.format_version < zestengine_assets::RENDERER_FORMAT;
        let mut changed = zestengine_assets::migrate_entries(&mut self.entries, legacy);
        if legacy {
            self.format_version = zestengine_assets::RENDERER_FORMAT;
            changed = true;
        }
        changed
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap_or_default()
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        std::fs::write(path, self.to_ron())?;
        Ok(())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Ok(Self::from_ron(&std::fs::read_to_string(path)?)?)
    }

    /// Root entry indices (those with no parent within the prefab).
    pub fn roots(&self) -> Vec<usize> {
        (0..self.entries.len())
            .filter(|&i| self.entries[i].parent.is_none())
            .collect()
    }

    /// Clone the prefab into a fresh set of entries ready to add to a scene:
    /// - **fresh ids** (instances never share an id with the prefab or each
    ///   other); object references inside component data that point at other
    ///   prefab objects are rewritten to the new ids,
    /// - root entries moved to `translation` (the per-instance override). Parent
    ///   indices stay local; the caller offsets them by the scene's current length.
    ///
    /// Returns the new entries (root-relative parent indices preserved).
    pub fn instantiate(&self, translation: [f32; 3]) -> Vec<SceneEntry> {
        self.instantiate_rooted(translation, None)
    }

    /// [`instantiate`](Self::instantiate) with a caller-chosen id for the first
    /// root (e.g. the id `ctx.spawn_prefab` already handed to gameplay code).
    pub fn instantiate_rooted(&self, translation: [f32; 3], root_id: Option<ObjectId>) -> Vec<SceneEntry> {
        let first_root = self.entries.iter().position(|e| e.parent.is_none());
        let new_ids: Vec<String> = (0..self.entries.len())
            .map(|i| match root_id {
                Some(id) if Some(i) == first_root => id.to_string(),
                _ => ObjectId::new().to_string(),
            })
            .collect();
        let mut out = self.entries.clone();
        for (i, e) in out.iter_mut().enumerate() {
            e.id = new_ids[i].clone();
            if self.entries[i].parent.is_none() {
                // Per-instance override: place the root(s) at `translation`.
                e.translation = translation;
            }
            for c in &mut e.components {
                for (old, new) in self.entries.iter().map(|e| e.id.as_str()).zip(&new_ids) {
                    if !old.is_empty() && c.data.contains(old) {
                        c.data = c.data.replace(old, new);
                    }
                }
            }
        }
        out
    }

    /// Like [`instantiate`] but also overrides each root's rotation + scale.
    pub fn instantiate_with(
        &self,
        translation: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    ) -> Vec<SceneEntry> {
        let mut out = self.instantiate(translation);
        for (i, e) in out.iter_mut().enumerate() {
            if self.entries[i].parent.is_none() {
                e.rotation = rotation;
                e.scale = scale;
            }
        }
        out
    }
}

/// `abs` as the project-relative, forward-slash path a [`PrefabLink`] stores.
pub fn link_path(project_root: &Path, abs: &Path) -> String {
    abs.strip_prefix(project_root).unwrap_or(abs).to_string_lossy().replace('\\', "/")
}

/// Prefab assets read so far, as the raw entries linked instances are built
/// from. Keeps the copy an open scene was merged against, so a changed asset
/// can be compared with what the instances were built from.
#[derive(Default)]
pub struct PrefabCache {
    assets: Mutex<HashMap<String, Vec<SceneEntry>>>,
    /// Files that failed to read and when, so a broken one is retried (and
    /// logged) every couple of seconds, not every frame.
    failed: Mutex<HashMap<String, std::time::Instant>>,
    /// The project the cached assets were read for (what `set` canonicalises against).
    root: Mutex<Option<std::path::PathBuf>>,
}

/// Write every `.material` reference of `entries` in the form the live scene
/// writes them (`@id` once the file has a `.meta`, else the project-relative
/// path). An asset saved by an older build, or by hand, names the same file by
/// path while an instance reads back as `@id`, which would show as a `material`
/// override on an untouched instance.
pub fn canonical_material_refs(project_root: &Path, entries: &mut [SceneEntry]) {
    let canon_path = |p: &mut String| {
        let abs = zestengine_core::asset_ref::resolve_path(project_root, p);
        let rel = link_path(project_root, &abs);
        *p = zestengine_core::asset_ref::reference(project_root, &rel);
    };
    let canon = |m: &mut zestengine_assets::MaterialRef| {
        if let zestengine_assets::MaterialRef::File(p) = m {
            canon_path(p);
        }
    };
    for e in entries {
        canon(&mut e.material);
        e.extra_materials.iter_mut().for_each(canon);
        for c in e.components.iter_mut().filter(|c| zestengine_core::is_renderer_kind(&c.kind)) {
            let Ok(mut r) = ron::from_str::<zestengine_core::MeshRenderer>(&c.data) else { continue };
            let mut touched = false;
            for slot in &mut r.materials {
                if let zestengine_core::MaterialSlot::File(p) = slot {
                    let before = p.clone();
                    canon_path(p);
                    touched |= *p != before;
                }
            }
            if touched {
                c.data = ron::to_string(&r).unwrap_or_default();
            }
        }
    }
}

impl PrefabCache {
    /// The asset's entries, read from disk on first use. A missing or broken
    /// file is not cached, so fixing it takes effect on the next call.
    pub fn get(&self, project_root: &Path, path: &str) -> Option<Vec<SceneEntry>> {
        let mut assets = self.assets.lock().ok()?;
        if let Some(e) = assets.get(path) {
            return Some(e.clone());
        }
        if self.failed.lock().ok()?.get(path).is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(2)) {
            return None;
        }
        let abs = zestengine_core::asset_ref::resolve_path(project_root, path);
        match Prefab::load(&abs) {
            Ok(mut p) => {
                canonical_material_refs(project_root, &mut p.entries);
                if let Ok(mut r) = self.root.lock() {
                    *r = Some(project_root.to_path_buf());
                }
                assets.insert(path.to_owned(), p.entries.clone());
                Some(p.entries)
            }
            Err(e) => {
                tracing::warn!("prefab {path}: {e:#}");
                self.failed.lock().ok()?.insert(path.to_owned(), std::time::Instant::now());
                None
            }
        }
    }

    /// The file on disk holds different objects than the cached copy (so
    /// instances were built from something else). False when the file can't be
    /// read or nothing is cached for it.
    pub fn differs_from_disk(&self, project_root: &Path, path: &str) -> bool {
        let Some(cached) = self.assets.lock().ok().and_then(|a| a.get(path).cloned()) else { return false };
        let abs = zestengine_core::asset_ref::resolve_path(project_root, path);
        let Ok(mut disk) = Prefab::load(&abs) else { return false };
        canonical_material_refs(project_root, &mut disk.entries);
        let text = |e: &Vec<SceneEntry>| ron::to_string(e).unwrap_or_default();
        text(&cached) != text(&disk.entries)
    }

    /// Replace the cached copy (after the asset was written); returns the old one.
    pub fn set(&self, path: &str, mut entries: Vec<SceneEntry>) -> Option<Vec<SceneEntry>> {
        zestengine_assets::migrate_entries(&mut entries, false);
        if let Some(root) = self.root.lock().ok().and_then(|r| r.clone()) {
            canonical_material_refs(&root, &mut entries);
        }
        self.assets.lock().ok()?.insert(path.to_owned(), entries)
    }

    /// Forget the cached copy; the next read goes to disk.
    pub fn forget(&self, path: &str) -> Option<Vec<SceneEntry>> {
        self.assets.lock().ok()?.remove(path)
    }

    /// Paths currently cached.
    pub fn paths(&self) -> Vec<String> {
        self.assets.lock().map(|a| a.keys().cloned().collect()).unwrap_or_default()
    }
}

/// Build the prefab links of a loaded scene file into objects, so the rest of
/// loading sees plain entries. Scenes without links are untouched.
pub fn expand_scene_file(file: &mut SceneFile, project_root: &Path, cache: &PrefabCache) {
    if file.entries.iter().any(|e| e.prefab.is_some()) {
        zestengine_assets::expand_prefabs(&mut file.entries, &|p| cache.get(project_root, p));
    }
}

/// [`expand_scene_file`] for a bare entry list (a runtime prefab spawn whose
/// asset nests linked prefabs, e.g. a variant). Entries without links are untouched.
pub fn expand_entries(entries: &mut Vec<SceneEntry>, project_root: &Path) {
    if entries.iter().any(|e| e.prefab.is_some()) {
        let cache = PrefabCache::default();
        zestengine_assets::expand_prefabs(entries, &|p| cache.get(project_root, p));
    }
}

/// The root entry of a new linked instance of `path`, placed at `translation`.
/// Expanding it (see [`expand_scene_file`]) produces the instance's objects.
pub fn link_root(path: &str, name: &str, translation: [f32; 3]) -> SceneEntry {
    SceneEntry {
        id: ObjectId::new().to_string(),
        name: name.to_owned(),
        source: zestengine_assets::ObjectSource::Empty,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: zestengine_core::ShadowCasting::On,
        layer: 0,
        material: zestengine_assets::MaterialRef::Model,
        extra_materials: Vec::new(),
        renderer_split: false,
        parent: None,
        components: Vec::new(),
        translation,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: Some(PrefabLink::new(path)),
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    }
}

/// "Create Prefab from Layers": write `<stem>.prefab` next to a PSD's `.layers` file, holding the
/// character as objects (sprite layers with the PSD stack order, bones as Sprite Bones, Sprite
/// Skins on the skinned layers; see `zestengine_assets::layered_prefab`). An existing prefab is
/// replaced (it is derived data: re-import, then create again).
pub fn create_from_layers(project_root: &Path, layers: &Path) -> anyhow::Result<(std::path::PathBuf, zestengine_assets::layered_prefab::LayersPrefabStats)> {
    let (entries, stats) = zestengine_assets::layered_prefab::layers_prefab_entries(project_root, layers)?;
    let path = zestengine_assets::layered_prefab::layers_prefab_path(layers);
    Prefab::new(entries).save(&path)?;
    Ok((path, stats))
}

/// Write a variant of the prefab at `base` next to it: a prefab whose only
/// object is a link to `base`, so it is the base plus whatever it overrides
/// later, and edits to the base reach it. Returns the new file.
pub fn create_variant(project_root: &Path, base: &Path) -> anyhow::Result<std::path::PathBuf> {
    let source = Prefab::load(base)?;
    let name = source.roots().first().map_or_else(|| "Prefab".to_owned(), |&r| source.entries[r].name.clone());
    let stem = base.file_stem().map_or_else(|| "Prefab".to_owned(), |s| s.to_string_lossy().into_owned());
    let dir = base.parent().unwrap_or(project_root);
    let mut path = dir.join(format!("{stem} Variant.prefab"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem} Variant {n}.prefab"));
        n += 1;
    }
    let root = link_root(&link_path(project_root, base), &name, [0.0; 3]);
    Prefab::new(vec![root]).save(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{MaterialRef, ObjectSource};

    fn entry(name: &str, parent: Option<usize>) -> SceneEntry {
        SceneEntry {
            id: format!("orig-{name}"),
            name: name.into(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: MaterialRef::File(String::new()),
            extra_materials: vec![],
            renderer_split: false,
            parent,
            components: vec![],
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    fn turret() -> Prefab {
        // root "Base" + child "Gun".
        Prefab::new(vec![entry("Base", None), entry("Gun", Some(0))])
    }

    #[test]
    fn instantiate_gives_fresh_ids_and_moves_root() {
        let p = turret();
        let inst = p.instantiate([5.0, 0.0, 2.0]);
        let again = p.instantiate([5.0, 0.0, 2.0]);
        assert_eq!(inst.len(), 2);
        assert!(inst.iter().all(|e| !e.id.is_empty() && !e.id.starts_with("orig-")), "fresh ids");
        assert_ne!(inst[0].id, again[0].id, "instances never share ids");
        assert_eq!(inst[0].translation, [5.0, 0.0, 2.0]); // root moved
        assert_eq!(inst[1].parent, Some(0)); // child parent preserved (local)
        assert_eq!(inst[1].translation, [0.0, 0.0, 0.0]); // child unchanged
    }

    #[test]
    fn instantiate_remaps_internal_references() {
        let gun_id = ObjectId::new().to_string();
        let mut base = entry("Base", None);
        base.components.push(zestengine_assets::ComponentData::new("Turret", format!("(barrel: Some(\"{gun_id}\"))")));
        let mut gun = entry("Gun", Some(0));
        gun.id = gun_id.clone();
        let p = Prefab::new(vec![base, gun]);
        let root = ObjectId::new();
        let inst = p.instantiate_rooted([0.0; 3], Some(root));
        assert_eq!(inst[0].id, root.to_string());
        assert!(inst[0].components[0].data.contains(&inst[1].id), "reference follows the new id");
        assert!(!inst[0].components[0].data.contains(&gun_id));
    }

    #[test]
    fn a_prefab_from_psd_layers_instantiates_with_its_skin_bones_remapped() {
        use zestengine_assets::layered_import::{LayerTree, LayerTreeNode};
        use zestengine_assets::rig2d::{Rig2D, SkinMesh2D, SkinPart, save_rig2d};
        let dir = std::env::temp_dir().join(format!("zest_layers_prefab_engine_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tree = LayerTree {
            canvas: [8, 8],
            ppu: 10.0,
            nodes: vec![
                LayerTreeNode { name: "body".into(), parent: None, sprite: "hero.png#0000000000000001".into(), position: [0.0, 0.2], sorting_order: 1 },
                LayerTreeNode { name: "arm".into(), parent: None, sprite: "hero.png#0000000000000002".into(), position: [0.3, 0.1], sorting_order: 2 },
            ],
        };
        std::fs::write(dir.join("hero.layers"), ron::ser::to_string_pretty(&tree, Default::default()).unwrap()).unwrap();
        let mut rig = Rig2D::default();
        rig.skeleton.add_bone(None, [0.0, 0.0], [0.0, 0.3], "Root");
        rig.skeleton.add_bone(Some(0), [0.3, 0.1], [0.5, 0.1], "Arm");
        rig.parts.push(SkinPart { sprite_id: 2, name: "arm".into(), mesh: SkinMesh2D::new(vec![[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]], vec![0, 1, 2], 1), bone_indices: vec![] });
        save_rig2d(dir.join("hero.rig2d"), &rig).unwrap();
        let (path, st) = create_from_layers(&dir, &dir.join("hero.layers")).unwrap();
        assert_eq!(path, dir.join("hero.prefab"));
        assert_eq!((st.sprites, st.bones, st.skinned), (2, 2, 1));
        let prefab = Prefab::load(&path).unwrap();
        assert_eq!(prefab.roots().len(), 1);
        let inst = prefab.instantiate([5.0, 0.0, 0.0]);
        assert_eq!(inst.len(), prefab.entries.len());
        assert_eq!(inst[prefab.roots()[0]].translation, [5.0, 0.0, 0.0]);
        // The skin of the arm names the instance's own bone objects, not the prefab's.
        let skin = inst.iter().flat_map(|e| &e.components).find(|c| c.kind == "Sprite Skin").expect("a skin");
        for e in &prefab.entries {
            assert!(!skin.data.contains(&e.id), "no reference to the prefab's own ids");
        }
        let bone_ids: Vec<&String> = inst.iter().filter(|e| e.components.iter().any(|c| c.kind == "Sprite Bone")).map(|e| &e.id).collect();
        assert_eq!(bone_ids.len(), 2);
        assert!(bone_ids.iter().all(|id| skin.data.contains(id.as_str())), "every bone of the rig is bound to an instance bone object");
        assert!(skin.data.contains(&inst[0].id), "the character root is the instance root");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_variant_is_a_link_to_its_base_and_expands_to_it() {
        let dir = std::env::temp_dir().join(format!("zest-variant-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("turret.prefab");
        turret().save(&base).unwrap();
        let v1 = create_variant(&dir, &base).unwrap();
        let v2 = create_variant(&dir, &base).unwrap();
        assert_eq!(v1.file_name().unwrap(), "turret Variant.prefab");
        assert_eq!(v2.file_name().unwrap(), "turret Variant 2.prefab", "never overwrites");
        let variant = Prefab::load(&v1).unwrap();
        assert_eq!(variant.entries.len(), 1);
        assert_eq!(variant.entries[0].prefab.as_ref().unwrap().path, "turret.prefab");
        // Placing the variant gives the base's objects.
        let cache = PrefabCache::default();
        let mut placed = vec![link_root("turret Variant.prefab", "Turret", [0.0; 3])];
        zestengine_assets::expand_prefabs(&mut placed, &|p| cache.get(&dir, p));
        assert_eq!(placed.len(), 2, "Base and Gun");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cache_notices_an_asset_edited_behind_its_back() {
        let dir = std::env::temp_dir().join(format!("zest-cache-diff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        turret().save(dir.join("t.prefab")).unwrap();
        let cache = PrefabCache::default();
        assert!(!cache.differs_from_disk(&dir, "t.prefab"), "nothing cached yet");
        cache.get(&dir, "t.prefab").unwrap();
        assert!(!cache.differs_from_disk(&dir, "t.prefab"), "just read");
        // Our own write goes through `set`, so it is not a change.
        let mut mine = turret();
        mine.entries[1].layer = 3;
        mine.save(dir.join("t.prefab")).unwrap();
        cache.set("t.prefab", mine.entries.clone());
        assert!(!cache.differs_from_disk(&dir, "t.prefab"));
        // Somebody else's write is.
        let mut theirs = turret();
        theirs.entries[1].layer = 9;
        theirs.save(dir.join("t.prefab")).unwrap();
        assert!(cache.differs_from_disk(&dir, "t.prefab"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_material_file_named_by_path_reads_back_in_the_form_the_scene_writes() {
        let dir = std::env::temp_dir().join(format!("zest-canon-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("red.material"), "()").unwrap();
        let mut p = turret();
        p.entries[1].source = ObjectSource::Model { path: "m.glb".into(), mesh: 0, extra_meshes: vec![1], model: None, mesh_id: None, extra_mesh_ids: Vec::new() };
        p.entries[1].set_renderer_materials(&[
            zestengine_assets::MaterialRef::File("red.material".into()),
            zestengine_assets::MaterialRef::File("./red.material".into()),
        ]);
        p.save(dir.join("t.prefab")).unwrap();
        let cache = PrefabCache::default();
        let got = cache.get(&dir, "t.prefab").unwrap();
        // What the live scene writes for that material (`@id` once the file has a sidecar).
        let live = zestengine_core::asset_ref::reference(&dir, "red.material");
        assert!(live.starts_with('@'), "{live}");
        let refs = zestengine_assets::material_refs(&got[1]);
        let zestengine_assets::MaterialRef::File(m) = &refs[0] else { panic!("a file ref") };
        assert_eq!(m, &live);
        let zestengine_assets::MaterialRef::File(x) = &refs[1] else { panic!("a file ref") };
        assert_eq!(x, &live, "another spelling of the same file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roots_are_parentless_entries() {
        assert_eq!(turret().roots(), vec![0]);
    }

    #[test]
    fn ron_round_trips_and_overrides_apply() {
        let p = turret();
        let s = p.to_ron();
        let back = Prefab::from_ron(&s).unwrap();
        let inst = back.instantiate_with([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]);
        assert_eq!(inst[0].translation, [1.0, 2.0, 3.0]);
        assert_eq!(inst[0].scale, [2.0, 2.0, 2.0]);
    }
}
