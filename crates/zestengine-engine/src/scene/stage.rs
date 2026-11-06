//! Prefab Mode inside the live scene.
//!
//! The scene is never reloaded: the open prefab's objects are appended to it
//! (ids as in the asset), the rest is hidden (isolated) or left on screen but
//! locked (in context, with the edited instance itself hidden), and closing
//! the stage removes them again. Object lists, render slots and the undo
//! history of the scene therefore stay valid across Prefab Mode.

use std::collections::HashSet;

use super::*;

/// How the rest of the scene shows while a prefab is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageMode {
    /// Only the prefab, at its own origin.
    Isolated,
    /// The scene stays visible (locked) and the prefab sits at the transform
    /// of the instance it was opened from.
    InContext,
}

/// Translation, rotation, scale.
pub type Trs = ([f32; 3], [f32; 4], [f32; 3]);

/// The prefab open in the scene.
#[derive(Clone, Debug)]
pub struct StageView {
    pub mode: StageMode,
    /// The open prefab's objects.
    pub members: HashSet<ObjectId>,
    /// Objects that are not drawn or lit while the stage is open.
    pub hidden: HashSet<ObjectId>,
    /// The scene instance the prefab was opened from (hidden in context).
    pub instance: Option<ObjectId>,
    /// The asset root's own transform, which a save writes back (the
    /// placement over the instance is only for looking at it).
    pub asset_root: Trs,
}

impl StageView {
    pub fn locked(&self, id: ObjectId) -> bool {
        !self.members.contains(&id)
    }
}

fn trs_of(m: Mat4) -> Trs {
    let (s, r, t) = m.to_scale_rotation_translation();
    (t.to_array(), r.to_array(), s.to_array())
}

impl LoadedScene {
    /// Whether `index` is outside the open prefab (not pickable, not editable).
    pub fn is_stage_locked(&self, index: usize) -> bool {
        self.stage.as_ref().is_some_and(|s| self.objects.get(index).is_some_and(|o| s.locked(o.id)))
    }

    /// Objects at and below `root`.
    fn subtree_ids(&self, root: usize) -> HashSet<ObjectId> {
        let mut ids = HashSet::new();
        let mut order = vec![root];
        let mut i = 0;
        while i < order.len() {
            let p = order[i];
            ids.insert(self.objects[p].id);
            for c in 0..self.objects.len() {
                if self.objects[c].parent == Some(p) && !order.contains(&c) {
                    order.push(c);
                }
            }
            i += 1;
        }
        ids
    }

    /// Open the prefab at `asset` (project-relative) in the scene. `place`
    /// puts its root there (the instance's world transform, in context); None
    /// leaves it at the asset's own transform. Returns the index of the root.
    #[allow(clippy::too_many_arguments)]
    pub fn open_stage(
        &mut self,
        renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        registry: &ComponentRegistry,
        asset: &str,
        mode: StageMode,
        instance: Option<usize>,
        place: Option<Trs>,
    ) -> Result<usize> {
        if self.stage.is_some() {
            anyhow::bail!("a prefab is already open");
        }
        let mut entries = self.prefabs.get(project_root, asset).with_context(|| format!("reading {asset}"))?;
        for e in &mut entries {
            if e.id.parse::<ObjectId>().is_err() {
                e.id = ObjectId::new().to_string();
            }
        }
        let taken: HashSet<String> = self.objects.iter().map(|o| o.id.to_string()).collect();
        if let Some(e) = entries.iter().find(|e| taken.contains(&e.id)) {
            anyhow::bail!("{asset} shares an object id with the scene ({}); open it from Files in a scene without that object", e.name);
        }
        let root = entries.iter().position(|e| e.parent.is_none()).context("the prefab has no root object")?;
        let asset_root: Trs = (entries[root].translation, entries[root].rotation, entries[root].scale);
        if let Some((t, r, s)) = place {
            entries[root].translation = t;
            entries[root].rotation = r;
            entries[root].scale = s;
        }
        let instance_id = instance.and_then(|i| self.objects.get(i)).map(|o| o.id);
        let mut hidden: HashSet<ObjectId> = match (mode, instance) {
            (StageMode::InContext, Some(i)) if i < self.objects.len() => self.subtree_ids(i),
            (StageMode::InContext, _) => HashSet::new(),
            // Isolated: everything that draws; lights and the rest keep working.
            (StageMode::Isolated, _) => self.objects.iter().filter(|o| o.render.is_some()).map(|o| o.id).collect(),
        };
        {
            let cache = &self.prefabs;
            zestengine_assets::expand_prefabs(&mut entries, &|p| cache.get(project_root, p));
        }
        let base = self.append_entries_with(renderer, shaders, project_root, registry, &entries)?;
        let members: HashSet<ObjectId> = self.objects[base..].iter().map(|o| o.id).collect();
        hidden.retain(|id| !members.contains(id));
        self.stage = Some(StageView { mode, members, hidden, instance: instance_id, asset_root });
        Ok(base + root.min(self.objects.len() - base - 1))
    }

    /// Take the open prefab's objects out of the scene. Returns their ids.
    pub fn close_stage(&mut self) -> HashSet<ObjectId> {
        let Some(view) = self.stage.take() else { return HashSet::new() };
        while let Some(i) = self.objects.iter().position(|o| view.members.contains(&o.id)) {
            self.remove_object(i);
        }
        view.members
    }

    /// The scene instance (and what is hidden with it) after objects moved:
    /// re-merges rebuild an instance's objects, which can bring ids the stage
    /// has not hidden yet.
    pub fn refresh_stage_hidden(&mut self) {
        let Some(view) = &self.stage else { return };
        if view.mode != StageMode::InContext {
            return;
        }
        let Some(inst) = view.instance.and_then(|id| self.objects.iter().position(|o| o.id == id)) else { return };
        let mut ids = self.subtree_ids(inst);
        let view = self.stage.as_mut().expect("checked above");
        ids.retain(|id| !view.members.contains(id));
        view.hidden = ids;
    }

    /// The open prefab as the entries of its `.prefab`: its objects (nested
    /// instances stored as their changes), the root at the asset's own
    /// transform.
    pub fn stage_entries(&self, project_root: &Path, shaders: &ShaderLibrary) -> Vec<SceneEntry> {
        let Some(view) = &self.stage else { return Vec::new() };
        let picked: Vec<usize> = (0..self.objects.len()).filter(|&i| view.members.contains(&self.objects[i].id)).collect();
        let local: HashMap<usize, usize> = picked.iter().enumerate().map(|(k, &o)| (o, k)).collect();
        let mut entries: Vec<SceneEntry> = picked
            .iter()
            .map(|&o| {
                let mut e = self.entry_of(&self.objects[o], project_root, shaders);
                e.parent = e.parent.and_then(|p| local.get(&p).copied());
                e
            })
            .collect();
        if let Some(root) = entries.iter_mut().find(|e| e.parent.is_none()) {
            (root.translation, root.rotation, root.scale) = view.asset_root;
        }
        self.compact_prefab_links(&mut entries, project_root);
        entries
    }

    /// Write the open prefab (`entries` from [`stage_entries`](Self::stage_entries))
    /// to `asset` and rebuild every instance of it in the scene. Returns the
    /// file change (for undo) and where each old object went.
    pub fn save_stage(
        &mut self,
        renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        registry: &ComponentRegistry,
        asset: &str,
        entries: Vec<SceneEntry>,
    ) -> Result<(PrefabFileChange, Vec<Option<usize>>)> {
        let abs = zestengine_core::asset_ref::resolve_path(project_root, asset);
        let before = std::fs::read(&abs).ok();
        crate::prefab::Prefab::new(entries).save(&abs).with_context(|| format!("writing {asset}"))?;
        let after = std::fs::read(&abs).with_context(|| format!("reading back {asset}"))?;
        // The cache still holds the previous version, which the instances were
        // built from: the re-merge compares against it and then reads the file.
        let remap = self.prefab_asset_changed(renderer, shaders, project_root, registry, asset)?;
        self.refresh_stage_hidden();
        Ok((PrefabFileChange { rel: asset.to_owned(), path: abs, before, after }, remap))
    }

    /// World transform of an object as a TRS (where to place a stage over it).
    pub fn world_trs(&self, index: usize) -> Trs {
        trs_of(self.world_transform(index))
    }
}

impl ObjectsSnapshot {
    /// The snapshot without the given objects (and what is below them), parents
    /// renumbered. Used to cut a stage's objects out of old undo steps.
    pub fn without(&self, gone: &HashSet<ObjectId>) -> ObjectsSnapshot {
        let n = self.objects.len();
        let mut drop: Vec<bool> = self.objects.iter().map(|o| gone.contains(&o.id)).collect();
        for _ in 0..n {
            let mut changed = false;
            for i in 0..n {
                if !drop[i] && self.objects[i].parent.is_some_and(|p| p < n && drop[p]) {
                    drop[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut at = vec![None; n];
        let mut next = 0;
        for i in 0..n {
            if !drop[i] {
                at[i] = Some(next);
                next += 1;
            }
        }
        let objects = self
            .objects
            .iter()
            .enumerate()
            .filter(|(i, _)| !drop[*i])
            .map(|(_, o)| {
                let mut o = o.clone();
                o.parent = o.parent.and_then(|p| at.get(p).copied().flatten());
                o
            })
            .collect::<Vec<_>>();
        let alive: HashSet<ObjectId> = objects.iter().map(|o| o.id).collect();
        ObjectsSnapshot {
            object_lightmap: self.object_lightmap.as_ref().map(|m| m.iter().filter_map(|(&k, &v)| at.get(k).copied().flatten().map(|nk| (nk, v))).collect()),
            prefab_links: self.prefab_links.iter().filter(|(id, _)| alive.contains(id)).map(|(k, v)| (*k, v.clone())).collect(),
            objects,
        }
    }

    /// Ids of the objects in the snapshot.
    pub fn ids(&self) -> Vec<ObjectId> {
        self.objects.iter().map(|o| o.id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::ObjectSource;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zest-stage-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(name: &str, parent: Option<usize>, source: ObjectSource) -> SceneEntry {
        SceneEntry {
            id: ObjectId::new().to_string(),
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
            translation: [0.0; 3],
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

    /// A scene holding a cube, a turret instance (Base > Gun) and a lone light object.
    fn scene_with_instance(root: &Path) -> (LoadedScene, usize) {
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let mut scene = LoadedScene::empty();
        let mut s = LoadedScene::empty();
        s.append_entries_with(None, &mut shaders, root, &registry, &[entry("Base", None, ObjectSource::Empty), entry("Gun", Some(0), ObjectSource::Empty)]).unwrap();
        let mut prefab = s.prefab_from_object(0, root, &shaders).unwrap();
        prefab.entries[0].translation = [1.0, 0.0, 0.0];
        prefab.save(root.join("turret.prefab")).unwrap();
        scene.append_entries_with(None, &mut shaders, root, &registry, &[entry("Cube", None, ObjectSource::Primitive { shape: zestengine_assets::PrimitiveShape::Cube })]).unwrap();
        let mut link = vec![crate::prefab::link_root("turret.prefab", "Turret", [10.0, 0.0, 0.0])];
        zestengine_assets::expand_prefabs(&mut link, &|p| scene.prefabs.get(root, p));
        let at = scene.append_entries_with(None, &mut shaders, root, &registry, &link).unwrap();
        (scene, at)
    }

    #[test]
    fn opening_in_context_places_the_prefab_over_the_instance_and_closing_leaves_the_scene_alone() {
        let root = temp_root("context");
        let (mut scene, inst) = scene_with_instance(&root);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let before = scene.snapshot_objects();
        let place = scene.world_trs(inst);
        let stage_root = scene.open_stage(None, &mut shaders, &root, &registry, "turret.prefab", StageMode::InContext, Some(inst), Some(place)).unwrap();
        assert_eq!(scene.objects[stage_root].translation, Vec3::new(10.0, 0.0, 0.0), "sits at the instance");
        let view = scene.stage.clone().unwrap();
        assert_eq!(view.members.len(), 2);
        assert_eq!(view.hidden.len(), 2, "the instance's own Base and Gun are hidden");
        assert!(view.hidden.contains(&scene.objects[inst].id));
        assert!(!scene.is_active(inst) || scene.stage.as_ref().unwrap().hidden.contains(&scene.objects[inst].id));
        // The scene's cube is visible but locked; the stage objects are editable.
        let cube = scene.objects.iter().position(|o| o.name == "Cube").unwrap();
        assert!(scene.is_stage_locked(cube) && !scene.is_stage_locked(stage_root));
        // Saving writes the asset's own root transform, not the placement.
        let saved = scene.stage_entries(&root, &shaders);
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].translation, [1.0, 0.0, 0.0]);
        // Closing puts everything back.
        let members = scene.close_stage();
        assert_eq!(members.len(), 2);
        assert!(scene.stage.is_none());
        assert_eq!(scene.objects.len(), before.ids().len());
        assert_eq!(scene.snapshot_objects().ids(), before.ids());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn isolated_hides_what_draws_and_refuses_clashing_ids() {
        let root = temp_root("isolated");
        let (mut scene, inst) = scene_with_instance(&root);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        scene.open_stage(None, &mut shaders, &root, &registry, "turret.prefab", StageMode::Isolated, None, None).unwrap();
        let view = scene.stage.clone().unwrap();
        let cube = scene.objects.iter().position(|o| o.name == "Cube").unwrap();
        assert!(view.hidden.contains(&scene.objects[cube].id), "the cube draws, so it is hidden");
        assert!(!view.hidden.contains(&scene.objects[inst].id), "an Empty has nothing to draw");
        assert!(scene.open_stage(None, &mut shaders, &root, &registry, "turret.prefab", StageMode::Isolated, None, None).is_err(), "one stage at a time");
        scene.close_stage();
        // A scene that already holds the asset's own objects can't also show them.
        let mut s2 = LoadedScene::empty();
        let prefab = crate::prefab::Prefab::load(root.join("turret.prefab")).unwrap();
        s2.append_entries_with(None, &mut shaders, &root, &registry, &prefab.entries).unwrap();
        let err = s2.open_stage(None, &mut shaders, &root, &registry, "turret.prefab", StageMode::Isolated, None, None).unwrap_err();
        assert!(format!("{err:#}").contains("shares an object id"), "{err:#}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_snapshot_loses_the_stage_objects_and_keeps_the_rest_consistent() {
        let root = temp_root("snapshot");
        let (mut scene, inst) = scene_with_instance(&root);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let plain = scene.snapshot_objects();
        scene.open_stage(None, &mut shaders, &root, &registry, "turret.prefab", StageMode::InContext, Some(inst), None).unwrap();
        let during = scene.snapshot_objects();
        assert!(during.ids().len() > plain.ids().len());
        let members = scene.stage.as_ref().unwrap().members.clone();
        let cut = during.without(&members);
        assert_eq!(cut.ids(), plain.ids());
        // Restoring the cut snapshot gives back the scene without the stage.
        scene.restore_objects(&cut, &registry);
        assert_eq!(scene.objects.len(), plain.ids().len());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn saving_the_stage_writes_the_asset_and_rebuilds_the_instance_in_place() {
        let root = temp_root("save");
        let (mut scene, inst) = scene_with_instance(&root);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let inst_id = scene.objects[inst].id;
        let place = scene.world_trs(inst);
        scene.open_stage(None, &mut shaders, &root, &registry, "turret.prefab", StageMode::InContext, Some(inst), Some(place)).unwrap();
        // Edit the open prefab's Gun.
        let gun = scene.objects.iter().position(|o| o.name == "Gun" && scene.stage.as_ref().unwrap().members.contains(&o.id)).unwrap();
        scene.objects[gun].layer = 7;
        let before_bytes = std::fs::read(root.join("turret.prefab")).unwrap();
        let entries = scene.stage_entries(&root, &shaders);
        let (change, _remap) = scene.save_stage(None, &mut shaders, &root, &registry, "turret.prefab", entries).unwrap();
        assert_eq!(change.before.as_deref(), Some(before_bytes.as_slice()));
        assert_ne!(change.after, before_bytes);
        // The instance now shows the change, kept its id, and is still hidden under the stage.
        let inst = scene.objects.iter().position(|o| o.id == inst_id).expect("instance kept its id");
        let inst_gun = scene.objects.iter().position(|o| o.name == "Gun" && o.parent.is_some() && !scene.stage.as_ref().unwrap().members.contains(&o.id)).unwrap();
        assert_eq!(scene.objects[inst_gun].layer, 7);
        let view = scene.stage.clone().unwrap();
        assert!(view.hidden.contains(&scene.objects[inst].id) && view.hidden.contains(&scene.objects[inst_gun].id));
        assert!(!scene.is_active(inst_gun));
        // Closing leaves the instance (merged) and nothing of the stage.
        scene.close_stage();
        assert_eq!(scene.objects.iter().filter(|o| o.name == "Gun").count(), 1);
        assert!(scene.prefab_info(scene.objects.iter().position(|o| o.id == inst_id).unwrap(), &root, &shaders).unwrap().diffs.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
