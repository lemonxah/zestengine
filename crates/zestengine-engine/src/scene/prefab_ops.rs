//! Prefab instances in the live scene: registering links on load, compacting
//! them for save, and the operations behind the editor (revert, apply, unpack,
//! re-merge when an asset changes).
//!
//! The scene's objects are plain objects whose ids come from the instance
//! root; the only extra state is `LoadedScene::prefab_links`. Every operation
//! turns the object list into entries, lets `zestengine_assets::prefab_link`
//! compact/expand them, and syncs the objects back **by id**, so identity,
//! selection and incoming references survive.

use std::path::PathBuf;

use zestengine_assets::{ApplyLevel, InstanceInfo, PrefabLink, PrefabScope};

use super::*;

/// A `.prefab` an operation wrote: for undo, which restores `before`
/// (None = the file didn't exist) and redo, which restores `after`.
#[derive(Clone, Debug)]
pub struct PrefabFileChange {
    /// Project-relative path (what links name).
    pub rel: String,
    pub path: PathBuf,
    pub before: Option<Vec<u8>>,
    pub after: Vec<u8>,
}

/// Which prefab link, if any, an object belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefabRole {
    /// Not part of a prefab instance.
    None,
    /// An object inside an instance.
    Member,
    /// The root of an instance nested inside another.
    NestedRoot,
    /// The root of an outermost instance.
    Root,
}

impl LoadedScene {
    /// Remember the links of freshly created objects, and read the assets they
    /// (and any prefab nested in them) came from so a later change to an asset
    /// can be compared with what the instances were built from.
    pub(crate) fn register_prefab_links(&mut self, entries: &[SceneEntry]) {
        for e in entries {
            if let Some(link) = &e.prefab
                && let Ok(id) = e.id.parse::<ObjectId>()
            {
                self.prefab_links.insert(id, link.clone());
            }
        }
    }

    /// Fill the asset cache with every prefab the links name, nested ones included.
    pub fn warm_prefab_cache(&self, project_root: &Path) {
        let mut paths: Vec<&str> = self.prefab_links.values().map(|l| l.path.as_str()).collect();
        paths.sort_unstable();
        paths.dedup();
        for p in paths {
            let _ = zestengine_assets::asset_image(p, &|q| self.prefabs.get(project_root, q), 0);
        }
    }

    /// Entries for the whole scene with the instances' links attached (what
    /// the prefab operations work on).
    pub(crate) fn linked_entries(&self, project_root: &Path, shaders: &ShaderLibrary) -> Vec<SceneEntry> {
        let mut entries: Vec<SceneEntry> = self.objects.iter().map(|o| self.entry_of(o, project_root, shaders)).collect();
        self.attach_links(&mut entries);
        entries
    }

    fn attach_links(&self, entries: &mut [SceneEntry]) {
        for e in entries {
            if let Ok(id) = e.id.parse::<ObjectId>()
                && let Some(link) = self.prefab_links.get(&id)
            {
                let mut link = link.clone();
                link.expanded = true;
                e.prefab = Some(link);
            }
        }
    }

    /// Store the instances in `entries` as their changes (for saving).
    pub(crate) fn compact_prefab_links(&self, entries: &mut Vec<SceneEntry>, project_root: &Path) {
        if self.prefab_links.is_empty() {
            return;
        }
        self.attach_links(entries);
        zestengine_assets::compact_prefabs(entries, &|p| self.prefabs.get(project_root, p));
    }

    /// Role of every object, for the Hierarchy markers.
    pub fn prefab_roles(&self) -> Vec<PrefabRole> {
        let n = self.objects.len();
        let mut roles = vec![PrefabRole::None; n];
        if self.prefab_links.is_empty() {
            return roles;
        }
        for (i, role) in roles.iter_mut().enumerate() {
            // Nearest linked ancestor-or-self decides; the outermost one names the root.
            let mut linked = None;
            let mut outer = None;
            let mut cur = Some(i);
            let mut steps = 0;
            while let Some(c) = cur {
                if steps > n {
                    break;
                }
                steps += 1;
                if self.prefab_links.contains_key(&self.objects[c].id) {
                    linked.get_or_insert(c);
                    outer = Some(c);
                }
                cur = self.objects[c].parent.filter(|p| *p < n);
            }
            *role = match (linked, outer) {
                (Some(l), Some(o)) if l == i && o == i => PrefabRole::Root,
                (Some(l), _) if l == i => PrefabRole::NestedRoot,
                (Some(_), _) => PrefabRole::Member,
                _ => PrefabRole::None,
            };
        }
        roles
    }

    /// The instance root an object belongs to: the outermost linked ancestor-or-self
    /// (`outermost`), else the nearest.
    pub fn prefab_root_of(&self, index: usize, outermost: bool) -> Option<usize> {
        let n = self.objects.len();
        let mut found = None;
        let mut cur = Some(index);
        let mut steps = 0;
        while let Some(c) = cur.filter(|c| *c < n) {
            if steps > n {
                break;
            }
            steps += 1;
            if self.prefab_links.contains_key(&self.objects[c].id) {
                found = Some(c);
                if !outermost {
                    break;
                }
            }
            cur = self.objects[c].parent;
        }
        found
    }

    /// An instance's objects as entries (root first, parents local), with the
    /// instance's links attached.
    fn instance_entries(&self, root: usize, project_root: &Path, shaders: &ShaderLibrary) -> Vec<SceneEntry> {
        let n = self.objects.len();
        let mut order = vec![root];
        let mut i = 0;
        while i < order.len() {
            let p = order[i];
            for c in 0..n {
                if self.objects[c].parent == Some(p) && !order.contains(&c) {
                    order.push(c);
                }
            }
            i += 1;
        }
        let local: HashMap<usize, usize> = order.iter().enumerate().map(|(k, &o)| (o, k)).collect();
        let mut entries: Vec<SceneEntry> = order
            .iter()
            .map(|&o| {
                let mut e = self.entry_of(&self.objects[o], project_root, shaders);
                e.parent = e.parent.and_then(|p| local.get(&p).copied());
                e
            })
            .collect();
        self.attach_links(&mut entries);
        entries
    }

    /// How the instance rooted at `root` differs from its asset. None when
    /// `root` isn't an instance or its asset can't be read.
    pub fn prefab_info(&self, root: usize, project_root: &Path, shaders: &ShaderLibrary) -> Option<InstanceInfo> {
        let id = self.objects.get(root)?.id.to_string();
        if !self.prefab_links.contains_key(&self.objects[root].id) {
            return None;
        }
        let entries = self.instance_entries(root, project_root, shaders);
        zestengine_assets::instance_info(&entries, &id, &|p| self.prefabs.get(project_root, p))
    }

    /// The stored link of an instance root (overrides as of the last load or operation).
    pub fn prefab_link(&self, root: usize) -> Option<&PrefabLink> {
        self.prefab_links.get(&self.objects.get(root)?.id)
    }

    /// The scene as the entries of a `.prefab` (Prefab Mode saves): objects as
    /// they are, nested instances stored as their changes. Model parts stay
    /// expanded, so the file compares against what instances expand to.
    pub fn to_prefab_entries(&self, project_root: &Path, shaders: &ShaderLibrary) -> Vec<SceneEntry> {
        let mut entries: Vec<SceneEntry> = self.objects.iter().map(|o| self.entry_of(o, project_root, shaders)).collect();
        self.compact_prefab_links(&mut entries, project_root);
        entries
    }

    /// Undo (`undo`) or redo an Apply: put the written `.prefab` files and the
    /// object list back as they were before or after it.
    pub fn restore_prefab_edit(
        &mut self,
        files: &[PrefabFileChange],
        snapshot: &ObjectsSnapshot,
        undo: bool,
        registry: &ComponentRegistry,
    ) -> Result<()> {
        for f in files {
            let bytes = if undo { f.before.as_ref() } else { Some(&f.after) };
            match bytes {
                Some(b) => {
                    std::fs::write(&f.path, b).with_context(|| format!("restoring {}", f.rel))?;
                    // Instances now match this content; keep the cache in step.
                    match std::str::from_utf8(b).ok().and_then(|t| crate::prefab::Prefab::from_ron(t).ok()) {
                        Some(p) => {
                            self.prefabs.set(&f.rel, p.entries);
                        }
                        None => {
                            self.prefabs.forget(&f.rel);
                        }
                    }
                }
                None => {
                    let _ = std::fs::remove_file(&f.path);
                    self.prefabs.forget(&f.rel);
                }
            }
        }
        self.restore_objects(snapshot, registry);
        Ok(())
    }

    /// Detach an instance (and with `completely`, every instance inside it)
    /// from its asset, keeping the objects as they are.
    pub fn unpack_prefab(&mut self, root: usize, completely: bool) -> bool {
        let Some(id) = self.objects.get(root).map(|o| o.id) else { return false };
        if self.prefab_links.remove(&id).is_none() {
            return false;
        }
        if completely {
            let n = self.objects.len();
            let mut inside = vec![false; n];
            inside[root] = true;
            for _ in 0..n {
                let mut changed = false;
                for i in 0..n {
                    if !inside[i]
                        && let Some(p) = self.objects[i].parent
                        && p < n
                        && inside[p]
                    {
                        inside[i] = true;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            for (i, is_inside) in inside.iter().enumerate() {
                if *is_inside {
                    self.prefab_links.remove(&self.objects[i].id);
                }
            }
        }
        true
    }

    /// Make the objects of the subtree at `root` an instance of the prefab
    /// just saved from it (`Create Prefab` then connects the selection). The
    /// objects get the ids the instance derives from the root's, and
    /// references to them from anywhere in the scene follow.
    pub fn connect_prefab(&mut self, root: usize, path: &str, registry: &ComponentRegistry) -> bool {
        let n = self.objects.len();
        if root >= n || self.prefab_links.contains_key(&self.objects[root].id) {
            return false;
        }
        let root_id = self.objects[root].id.to_string();
        let mut inside = vec![root];
        let mut i = 0;
        while i < inside.len() {
            let p = inside[i];
            for c in 0..n {
                if self.objects[c].parent == Some(p) && !inside.contains(&c) {
                    inside.push(c);
                }
            }
            i += 1;
        }
        let mut renames: Vec<(String, String)> = Vec::new();
        // Instances inside the selection become part of the new prefab's own objects.
        for &o in inside.iter().skip(1) {
            self.prefab_links.remove(&self.objects[o].id);
        }
        for &o in inside.iter().skip(1) {
            let old = self.objects[o].id.to_string();
            let new = zestengine_assets::live_id(&root_id, &root_id, &old);
            if let Ok(id) = new.parse::<ObjectId>() {
                self.objects[o].id = id;
                renames.push((old, new));
            }
        }
        for o in &mut self.objects {
            let mut saved = None;
            for (k, c) in o.components.iter().enumerate() {
                let data = c.save();
                if renames.iter().any(|(old, _)| data.contains(old.as_str())) {
                    saved.get_or_insert_with(|| o_save(&o.components))[k].data = renames.iter().fold(data, |d, (old, new)| d.replace(old.as_str(), new));
                }
            }
            if let Some(saved) = saved {
                o.load_components(&saved, registry);
            }
        }
        let mut link = PrefabLink::new(path);
        link.expanded = true;
        let id = self.objects[root].id;
        self.prefab_links.insert(id, link);
        true
    }

    /// Put the objects in line with `entries` (the whole scene, parent indices
    /// as in the list): objects are matched by id and updated in place, new
    /// ones are created, missing ones dropped. Returns where each old object
    /// went, so selection can follow.
    pub(crate) fn sync_entries(
        &mut self,
        mut renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        registry: &ComponentRegistry,
        entries: &[SceneEntry],
    ) -> Result<Vec<Option<usize>>> {
        let backup = self.snapshot_objects();
        let old_entries: HashMap<String, SceneEntry> = self.objects.iter().map(|o| (o.id.to_string(), self.entry_of(o, project_root, shaders))).collect();
        let old_ids: Vec<ObjectId> = self.objects.iter().map(|o| o.id).collect();
        let render_key = |e: &SceneEntry| ron::to_string(&(&e.source, zestengine_assets::material_refs(e))).unwrap_or_default();
        let mut pool: HashMap<ObjectId, SceneObject> = std::mem::take(&mut self.objects).into_iter().map(|o| (o.id, o)).collect();
        let mut made = Vec::with_capacity(entries.len());
        let result = (|| -> Result<()> {
            for e in entries {
                let id = e.id.parse::<ObjectId>().unwrap_or_else(|_| ObjectId::new());
                let mut fresh = |this: &mut Self, renderer: Option<&mut Renderer>| -> Result<SceneObject> {
                    this.append_entries_with(renderer, shaders, project_root, registry, std::slice::from_ref(e))?;
                    let mut o = this.objects.pop().context("object was not created")?;
                    o.id = id;
                    Ok(o)
                };
                let object = match pool.remove(&id) {
                    Some(mut o) => {
                        if old_entries.get(&e.id).is_none_or(|old| render_key(old) != render_key(e)) {
                            let f = fresh(self, renderer.as_deref_mut())?;
                            o.render = f.render;
                            o.extra_render = f.extra_render;
                            o.source = f.source;
                        }
                        o.name = e.name.clone();
                        o.enabled = e.enabled;
                        o.static_geometry = e.static_geometry;
                        o.layer = e.layer;
                        o.translation = Vec3::from(e.translation);
                        o.rotation = Quat::from_array(e.rotation);
                        o.scale = Vec3::from(e.scale);
                        if o.save_components() != e.components {
                            o.load_components(&e.components, registry);
                        }
                        o
                    }
                    None => fresh(self, renderer.as_deref_mut())?,
                };
                made.push(object);
            }
            Ok(())
        })();
        if let Err(err) = result {
            self.restore_objects(&backup, registry);
            return Err(err);
        }
        self.objects = made;
        let n = self.objects.len();
        for (k, e) in entries.iter().enumerate() {
            self.objects[k].parent = e.parent.filter(|p| *p < n && *p != k);
        }
        // Objects whose renderer component was replaced name materials the cache may not hold yet.
        if let Some(renderer) = renderer {
            for i in 0..n {
                self.refresh_renderer_materials(renderer, shaders, project_root, i);
            }
        }
        let new_index: HashMap<ObjectId, usize> = self.objects.iter().enumerate().map(|(k, o)| (o.id, k)).collect();
        let remap: Vec<Option<usize>> = old_ids.iter().map(|id| new_index.get(id).copied()).collect();
        if let Some(baked) = self.baked.as_mut() {
            baked.object_lightmap = baked.object_lightmap.drain().filter_map(|(old, layer)| remap.get(old).copied().flatten().map(|new| (new, layer))).collect();
        }
        if let Some(cam) = self.active_camera_override
            && !new_index.contains_key(&cam)
        {
            self.active_camera_override = None;
        }
        self.prefab_links.clear();
        self.register_prefab_links(entries);
        Ok(remap)
    }

    /// Throw away the changes `scope` covers on the instance at `root` and
    /// rebuild it from its asset. Returns where each old object went.
    pub fn revert_prefab_instance(
        &mut self,
        renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        registry: &ComponentRegistry,
        root: usize,
        scope: &PrefabScope,
    ) -> Result<Vec<Option<usize>>> {
        let id = self.objects.get(root).context("no such object")?.id.to_string();
        let mut entries = self.linked_entries(project_root, shaders);
        let load = |p: &str| self.prefabs.get(project_root, p);
        if !zestengine_assets::revert_prefab(&mut entries, &id, scope, &load) {
            anyhow::bail!("not a prefab instance");
        }
        self.sync_entries(renderer, shaders, project_root, registry, &entries)
    }

    /// Write the changes `scope` covers into the prefab asset at `level`,
    /// then rebuild every instance. Returns the file written (for undo) and
    /// where each old object went.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_prefab_instance(
        &mut self,
        renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        registry: &ComponentRegistry,
        root: usize,
        scope: &PrefabScope,
        level: &ApplyLevel,
    ) -> Result<(PrefabFileChange, Vec<Option<usize>>)> {
        let id = self.objects.get(root).context("no such object")?.id.to_string();
        let mut entries = self.linked_entries(project_root, shaders);
        let load = |p: &str| self.prefabs.get(project_root, p);
        let (path, raw) = zestengine_assets::apply_prefab(&mut entries, &id, scope, level, &load).context("nothing to apply")?;
        let change = self.write_prefab(project_root, &path, raw)?;
        let load = |p: &str| self.prefabs.get(project_root, p);
        zestengine_assets::expand_prefabs(&mut entries, &load);
        let remap = self.sync_entries(renderer, shaders, project_root, registry, &entries)?;
        Ok((change, remap))
    }

    /// Save `entries` as the prefab at `path` and make it the cached copy.
    pub(super) fn write_prefab(&self, project_root: &Path, path: &str, entries: Vec<SceneEntry>) -> Result<PrefabFileChange> {
        let abs = zestengine_core::asset_ref::resolve_path(project_root, path);
        let before = std::fs::read(&abs).ok();
        let prefab = crate::prefab::Prefab::new(entries.clone());
        prefab.save(&abs).with_context(|| format!("writing {path}"))?;
        let after = std::fs::read(&abs).with_context(|| format!("reading back {path}"))?;
        self.prefabs.set(path, entries);
        Ok(PrefabFileChange { rel: path.to_owned(), path: abs, before, after })
    }

    /// The prefab at `path` changed on disk (Prefab Mode save, an external
    /// edit, undo of an Apply): re-merge every instance of it, or of anything
    /// nesting it. Compacts against the copy the instances were built from,
    /// then expands against the file as it is now.
    pub fn prefab_asset_changed(
        &mut self,
        renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        registry: &ComponentRegistry,
        path: &str,
    ) -> Result<Vec<Option<usize>>> {
        let mut entries = self.linked_entries(project_root, shaders);
        let old: HashMap<String, Vec<SceneEntry>> = self.prefabs.paths().into_iter().filter_map(|p| self.prefabs.get(project_root, &p).map(|e| (p, e))).collect();
        zestengine_assets::compact_prefabs(&mut entries, &|p| old.get(p).cloned());
        self.prefabs.forget(path);
        let load = |p: &str| self.prefabs.get(project_root, p);
        zestengine_assets::expand_prefabs(&mut entries, &load);
        self.sync_entries(renderer, shaders, project_root, registry, &entries)
    }

    /// Duplicate `index` and its subtree. Prefab instances in the subtree are
    /// copied as **linked** instances: the copy keeps the link (overrides,
    /// removed objects and all), its objects get the ids the new root derives,
    /// and references between the copies follow. The rest of the subtree is
    /// copied as [`duplicate_object`](LoadedScene::duplicate_object) does.
    /// Duplicating a plain member of an instance gives an added object.
    pub fn duplicate_object_linked(
        &mut self,
        index: usize,
        project_root: &Path,
        shaders: &ShaderLibrary,
        registry: &ComponentRegistry,
    ) -> Option<usize> {
        let n = self.objects.len();
        if index >= n {
            return None;
        }
        let mut subtree = vec![index];
        let mut i = 0;
        while i < subtree.len() {
            let p = subtree[i];
            for c in 0..n {
                if self.objects[c].parent == Some(p) && !subtree.contains(&c) {
                    subtree.push(c);
                }
            }
            i += 1;
        }
        let linked: Vec<usize> = subtree.iter().copied().filter(|&o| self.prefab_links.contains_key(&self.objects[o].id)).collect();
        // Member tables of the instances that sit highest in the subtree.
        let outer: Vec<(usize, Vec<(String, String)>)> = linked
            .iter()
            .copied()
            .filter(|&o| !linked.iter().any(|&q| q != o && self.is_below(o, q)))
            .filter_map(|o| self.prefab_info(o, project_root, shaders).map(|info| (o, info.members)))
            .collect();
        let old_ids: Vec<String> = subtree.iter().map(|&o| self.objects[o].id.to_string()).collect();
        let base = self.duplicate_object(index, registry)?;
        if linked.is_empty() {
            return Some(base);
        }
        let pos_of: HashMap<&str, usize> = old_ids.iter().enumerate().map(|(k, id)| (id.as_str(), k)).collect();
        let mut renames: Vec<(String, String)> = Vec::new();
        // Plain objects keep the fresh ids the copy gave them.
        for (k, old) in old_ids.iter().enumerate() {
            renames.push((old.clone(), self.objects[base + k].id.to_string()));
        }
        // Members of an instance take the ids the copy's root derives.
        for (root, members) in &outer {
            let Some(rk) = subtree.iter().position(|o| o == root) else { continue };
            let new_root = self.objects[base + rk].id.to_string();
            let Some(root_src) = members.first().map(|(img, _)| img.clone()) else { continue };
            for (img, old_live) in members.iter().skip(1) {
                let Some(&k) = pos_of.get(old_live.as_str()) else { continue };
                let new = zestengine_assets::live_id(&root_src, &new_root, img);
                if let Ok(id) = new.parse::<ObjectId>() {
                    self.objects[base + k].id = id;
                    renames[k].1 = new;
                }
            }
        }
        // Links follow their instance; override values that name members follow too.
        for &o in &linked {
            let Some(k) = subtree.iter().position(|&x| x == o) else { continue };
            let Some(mut link) = self.prefab_links.get(&self.objects[o].id).cloned() else { continue };
            for ov in &mut link.overrides {
                for (old, new) in &renames {
                    if old != new && ov.value.contains(old.as_str()) {
                        ov.value = ov.value.replace(old.as_str(), new);
                    }
                }
            }
            self.prefab_links.insert(self.objects[base + k].id, link);
        }
        // Components of the copies that point at objects of the subtree point at the copies.
        for k in 0..subtree.len() {
            let o = &mut self.objects[base + k];
            let mut saved = None;
            for (ci, c) in o.components.iter().enumerate() {
                let data = c.save();
                if renames.iter().any(|(old, new)| old != new && data.contains(old.as_str())) {
                    saved.get_or_insert_with(|| o_save(&o.components))[ci].data =
                        renames.iter().filter(|(old, new)| old != new).fold(data, |d, (old, new)| d.replace(old.as_str(), new));
                }
            }
            if let Some(saved) = saved {
                o.load_components(&saved, registry);
            }
        }
        Some(base)
    }

    /// Whether `o` hangs below `ancestor`.
    fn is_below(&self, o: usize, ancestor: usize) -> bool {
        let mut cur = self.objects[o].parent;
        for _ in 0..=self.objects.len() {
            match cur {
                Some(p) if p == ancestor => return true,
                Some(p) if p < self.objects.len() => cur = self.objects[p].parent,
                _ => return false,
            }
        }
        false
    }

    /// Drop the overrides of an instance whose target is gone from the asset.
    pub fn remove_unused_prefab_overrides(&mut self, root: usize) -> usize {
        let Some(id) = self.objects.get(root).map(|o| o.id) else { return 0 };
        let Some(link) = self.prefab_links.get_mut(&id) else { return 0 };
        let before = link.overrides.len();
        link.overrides.retain(|o| !o.unused);
        before - link.overrides.len()
    }
}

fn o_save(components: &[Box<dyn Component>]) -> Vec<zestengine_assets::ComponentData> {
    components.iter().map(|c| zestengine_assets::ComponentData::of(c.as_ref())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::ObjectSource;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zest-prefab-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(name: &str, parent: Option<usize>) -> SceneEntry {
        SceneEntry {
            id: ObjectId::new().to_string(),
            name: name.into(),
            source: ObjectSource::Empty,
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

    /// A scene holding Base > Gun > Cap, saved as `turret.prefab`.
    fn make_asset(root: &Path) -> String {
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let mut scene = LoadedScene::empty();
        let entries = [entry("Base", None), entry("Gun", Some(0)), entry("Cap", Some(1))];
        scene.append_entries_with(None, &mut shaders, root, &registry, &entries).unwrap();
        let prefab = scene.prefab_from_object(0, root, &shaders).unwrap();
        prefab.save(root.join("turret.prefab")).unwrap();
        "turret.prefab".to_owned()
    }

    fn place(scene: &mut LoadedScene, root: &Path, path: &str) -> usize {
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let mut entries = vec![crate::prefab::link_root(path, "Turret", [1.0, 2.0, 3.0])];
        zestengine_assets::expand_prefabs(&mut entries, &|p| scene.prefabs.get(root, p));
        scene.append_entries_with(None, &mut shaders, root, &registry, &entries).unwrap()
    }

    fn by_name(scene: &LoadedScene, name: &str) -> usize {
        scene.objects.iter().position(|o| o.name == name).unwrap_or_else(|| panic!("no {name}"))
    }

    #[test]
    fn a_placed_instance_saves_as_its_changes_and_loads_back_identical() {
        let root = temp_root("roundtrip");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        let base = place(&mut scene, &root, &asset);
        assert_eq!(scene.objects.len(), 3);
        assert!(scene.prefab_links.contains_key(&scene.objects[base].id));
        let roles = scene.prefab_roles();
        assert_eq!(roles[base], PrefabRole::Root);
        assert_eq!(roles[by_name(&scene, "Gun")], PrefabRole::Member);

        // Untouched: the scene stores just the root.
        let shaders = ShaderLibrary::default();
        let file = scene.to_compact_scene_file(&root, &shaders);
        assert_eq!(file.entries.len(), 1);
        assert!(file.entries[0].prefab.is_some());

        // Edit Gun, delete Cap, add Flag; save; load into a fresh scene.
        let gun = by_name(&scene, "Gun");
        scene.objects[gun].layer = 4;
        let cap = by_name(&scene, "Cap");
        scene.remove_object(cap);
        let registry = ComponentRegistry::with_builtins();
        let mut sh = ShaderLibrary::default();
        let gun = by_name(&scene, "Gun");
        let mut flag = entry("Flag", Some(gun));
        flag.layer = 7;
        scene.append_entries_with(None, &mut sh, &root, &registry, &[flag]).unwrap();
        let flag_idx = by_name(&scene, "Flag");
        scene.objects[flag_idx].parent = Some(gun);
        let file = scene.to_compact_scene_file(&root, &shaders);
        assert_eq!(file.entries.len(), 2, "root + the added Flag");
        let path = root.join("level.scene");
        zestengine_assets::save_scene_file(&path, &file).unwrap();
        let mut loaded = zestengine_assets::load_scene_file(&path).unwrap();
        crate::prefab::expand_scene_file(&mut loaded, &root, &crate::prefab::PrefabCache::default());

        let mut fresh = LoadedScene::empty();
        fresh.append_entries_with(None, &mut sh, &root, &registry, &loaded.entries).unwrap();
        let ids = |s: &LoadedScene| {
            let mut v: Vec<(String, String, u8)> = s.objects.iter().map(|o| (o.id.to_string(), o.name.clone(), o.layer)).collect();
            v.sort();
            v
        };
        assert_eq!(ids(&scene), ids(&fresh), "the reloaded scene has the same objects and ids");
        assert_eq!(fresh.objects[by_name(&fresh, "Gun")].layer, 4);
        assert!(fresh.objects.iter().all(|o| o.name != "Cap"));
        let flag = &fresh.objects[by_name(&fresh, "Flag")];
        assert_eq!(flag.parent, Some(by_name(&fresh, "Gun")));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The editor / player load path (`load_on_worker` + `install`) must build
    /// the objects of a saved compact instance, not just its root.
    #[test]
    fn worker_load_expands_saved_linked_instances() {
        let root = temp_root("workerload");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        let base = place(&mut scene, &root, &asset);
        let root_id = scene.objects[base].id;
        let gun = by_name(&scene, "Gun");
        scene.objects[gun].layer = 4;
        let ids_before: std::collections::HashMap<String, ObjectId> = scene.objects.iter().map(|o| (o.name.clone(), o.id)).collect();
        let shaders = ShaderLibrary::default();
        let file = scene.to_compact_scene_file(&root, &shaders);
        assert_eq!(file.entries.len(), 1, "stored compact: root + link");
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        let path = root.join("scenes/level.scene");
        zestengine_assets::save_scene_file(&path, &file).unwrap();

        let data = crate::scene_loader::load_on_worker(&path, &root, None, &[], &|_| {}).unwrap();
        let registry = ComponentRegistry::with_builtins();
        let mut sh = ShaderLibrary::default();
        let (loaded, _post) = crate::scene_loader::install(data, None, &registry, &mut sh, &root, |_| {}).unwrap();

        assert_eq!(loaded.objects.len(), 3, "all the instance's objects exist");
        for name in ["Turret", "Gun", "Cap"] {
            assert_eq!(loaded.objects[by_name(&loaded, name)].id, ids_before[name], "{name} keeps its derived id");
        }
        assert_eq!(loaded.objects[by_name(&loaded, "Turret")].id, root_id);
        assert_eq!(loaded.objects[by_name(&loaded, "Gun")].layer, 4, "override applied");
        assert_eq!(loaded.objects[by_name(&loaded, "Cap")].parent, Some(by_name(&loaded, "Gun")));
        assert!(loaded.prefab_links.contains_key(&root_id), "link kept for later prefab ops");
        let roles = loaded.prefab_roles();
        assert_eq!(roles[by_name(&loaded, "Turret")], PrefabRole::Root);
        assert_eq!(roles[by_name(&loaded, "Gun")], PrefabRole::Member);
        // And it saves back compact.
        assert_eq!(loaded.to_compact_scene_file(&root, &sh).entries.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scenes_without_prefab_links_load_unchanged() {
        let root = temp_root("legacy");
        let mut file = zestengine_assets::SceneFile { entries: vec![entry("A", None), entry("B", Some(0))], ..Default::default() };
        let before: Vec<String> = file.entries.iter().map(|e| e.id.clone()).collect();
        crate::prefab::expand_scene_file(&mut file, &root, &crate::prefab::PrefabCache::default());
        assert_eq!(file.entries.iter().map(|e| e.id.clone()).collect::<Vec<_>>(), before);
        // A placement from before linked prefabs: plain objects, no link, and saving keeps them.
        let mut scene = LoadedScene::empty();
        let registry = ComponentRegistry::with_builtins();
        let mut sh = ShaderLibrary::default();
        scene.append_entries_with(None, &mut sh, &root, &registry, &file.entries).unwrap();
        assert!(scene.prefab_links.is_empty());
        let saved = scene.to_compact_scene_file(&root, &sh);
        assert_eq!(saved.entries.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn revert_is_one_undo_step_and_keeps_ids() {
        let root = temp_root("revert");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        let base = place(&mut scene, &root, &asset);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let gun = by_name(&scene, "Gun");
        let gun_id = scene.objects[gun].id;
        scene.objects[gun].layer = 9;
        let before = scene.snapshot_objects();
        let info = scene.prefab_info(base, &root, &shaders).unwrap();
        assert_eq!(info.diffs.len(), 1);
        scene.revert_prefab_instance(None, &mut shaders, &root, &registry, base, &PrefabScope::All).unwrap();
        let after = scene.snapshot_objects();
        let gun = by_name(&scene, "Gun");
        assert_eq!(scene.objects[gun].layer, 0);
        assert_eq!(scene.objects[gun].id, gun_id);
        assert!(scene.prefab_info(base, &root, &shaders).unwrap().diffs.is_empty());
        // Undo / redo restore both states from the two snapshots.
        scene.restore_objects(&before, &registry);
        assert_eq!(scene.objects[by_name(&scene, "Gun")].layer, 9);
        scene.restore_objects(&after, &registry);
        assert_eq!(scene.objects[by_name(&scene, "Gun")].layer, 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn apply_writes_the_asset_and_undo_puts_the_file_back() {
        let root = temp_root("apply");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        let base = place(&mut scene, &root, &asset);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let mut other_entries = vec![crate::prefab::link_root(&asset, "Second", [9.0, 0.0, 0.0])];
        zestengine_assets::expand_prefabs(&mut other_entries, &|p| scene.prefabs.get(&root, p));
        scene.append_entries_with(None, &mut shaders, &root, &registry, &other_entries).unwrap();

        let gun = by_name(&scene, "Gun");
        scene.objects[gun].layer = 6;
        let file_before = std::fs::read(root.join(&asset)).unwrap();
        let before = scene.snapshot_objects();
        let (change, _) = scene
            .apply_prefab_instance(None, &mut shaders, &root, &registry, base, &PrefabScope::All, &ApplyLevel::outermost())
            .unwrap();
        let after = scene.snapshot_objects();
        assert_ne!(std::fs::read(root.join(&asset)).unwrap(), file_before, "the asset changed");
        // Both instances now show the applied value, and neither has a diff left.
        let layers: Vec<u8> = scene.objects.iter().filter(|o| o.name == "Gun").map(|o| o.layer).collect();
        assert_eq!(layers, vec![6, 6]);
        for r in 0..scene.objects.len() {
            if scene.prefab_links.contains_key(&scene.objects[r].id) {
                assert!(scene.prefab_info(r, &root, &shaders).unwrap().diffs.is_empty());
            }
        }

        // Undo: file and objects go back as one step; redo reapplies.
        scene.restore_prefab_edit(std::slice::from_ref(&change), &before, true, &registry).unwrap();
        assert_eq!(std::fs::read(root.join(&asset)).unwrap(), file_before);
        let layers: Vec<u8> = scene.objects.iter().filter(|o| o.name == "Gun").map(|o| o.layer).collect();
        assert_eq!(layers.len(), 2);
        assert!(layers.contains(&6) && layers.contains(&0));
        assert!(scene.prefab_info(base, &root, &shaders).unwrap().diffs.len() == 1, "the override is back on the instance");
        scene.restore_prefab_edit(std::slice::from_ref(&change), &after, false, &registry).unwrap();
        assert_eq!(std::fs::read(root.join(&asset)).unwrap(), change.after);
        assert_eq!(scene.objects.iter().filter(|o| o.name == "Gun").map(|o| o.layer).collect::<Vec<_>>(), vec![6, 6]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_changed_asset_remerges_live_instances_in_place() {
        let root = temp_root("remerge");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        place(&mut scene, &root, &asset);
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let gun = by_name(&scene, "Gun");
        let gun_id = scene.objects[gun].id;
        let cap = by_name(&scene, "Cap");
        scene.objects[cap].layer = 3; // an override on something about to be deleted
        scene.objects[gun].translation = Vec3::new(0.0, 5.0, 0.0);

        // The asset changes: Cap is deleted, Barrel is new, Gun gets a scale.
        let mut p = crate::prefab::Prefab::load(root.join(&asset)).unwrap();
        let gi = p.entries.iter().position(|e| e.name == "Gun").unwrap();
        p.entries[gi].scale = [2.0; 3];
        p.entries.retain(|e| e.name != "Cap");
        let mut barrel = entry("Barrel", Some(gi));
        barrel.material = p.entries[gi].material.clone();
        p.entries.push(barrel);
        p.save(root.join(&asset)).unwrap();
        scene.prefab_asset_changed(None, &mut shaders, &root, &registry, &asset).unwrap();

        let gun = by_name(&scene, "Gun");
        assert_eq!(scene.objects[gun].id, gun_id, "identity kept");
        assert_eq!(scene.objects[gun].translation, Vec3::new(0.0, 5.0, 0.0), "the instance's override kept");
        assert_eq!(scene.objects[gun].scale, Vec3::splat(2.0), "the asset's change arrived");
        assert!(scene.objects.iter().any(|o| o.name == "Barrel") && scene.objects.iter().all(|o| o.name != "Cap"));
        let base = scene.prefab_root_of(gun, true).unwrap();
        assert_eq!(scene.prefab_link(base).unwrap().unused().count(), 1, "the override on the deleted Cap is kept as unused");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The repository root, where the sample models live.
    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
    }

    #[test]
    fn an_untouched_instance_with_a_textured_model_has_no_material_override() {
        // Real model-bearing prefab: the helmet has textured materials. Created
        // from the scene, placed twice, nothing touched: no diff at all.
        let project = repo_root();
        let out = temp_root("model-material");
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let mut scene = LoadedScene::empty();
        let mut helmet = entry("Helmet", None);
        helmet.source = ObjectSource::Model { path: "models/DamagedHelmet.glb".into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] };
        scene.append_entries_with(None, &mut shaders, &project, &registry, &[helmet, entry("Child", Some(0))]).unwrap();
        let prefab = scene.prefab_from_object(0, &project, &shaders).unwrap();
        prefab.save(out.join("helmet.prefab")).unwrap();
        // The prefab lives in a temp dir but names a model of the project: link it by absolute path.
        let rel = out.join("helmet.prefab").to_string_lossy().into_owned();
        let mut s2 = LoadedScene::empty();
        let mut entries = vec![crate::prefab::link_root(&rel, "Helmet", [0.0; 3])];
        zestengine_assets::expand_prefabs(&mut entries, &|p| s2.prefabs.get(&project, p));
        s2.append_entries_with(None, &mut shaders, &project, &registry, &entries).unwrap();
        let info = s2.prefab_info(0, &project, &shaders).expect("instance info");
        let paths: Vec<String> = info
            .diffs
            .iter()
            .flat_map(|d| match &d.change {
                zestengine_assets::Change::Modified { fields, .. } => fields.iter().map(|f| f.path.clone()).collect::<Vec<_>>(),
                _ => vec!["<added/removed>".to_owned()],
            })
            .collect();
        assert!(paths.is_empty(), "untouched instance reports {paths:?}");
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn duplicating_a_linked_instance_gives_a_linked_copy() {
        let root = temp_root("duplicate");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        let base = place(&mut scene, &root, &asset);
        let registry = ComponentRegistry::with_builtins();
        let shaders = ShaderLibrary::default();
        let gun = by_name(&scene, "Gun");
        scene.objects[gun].layer = 5; // an override that must come along
        let cap = by_name(&scene, "Cap");
        scene.remove_object(cap); // and a removed object
        let copy = scene.duplicate_object_linked(base, &root, &shaders, &registry).unwrap();
        assert_ne!(scene.objects[copy].id, scene.objects[base].id);
        assert!(scene.prefab_links.contains_key(&scene.objects[copy].id), "the copy is linked");
        assert_eq!(scene.prefab_roles()[copy], PrefabRole::Root);
        // Its objects carry ids derived from its own root: no diff against the asset beyond the copied overrides.
        let copy_info = scene.prefab_info(copy, &root, &shaders).unwrap();
        let orig_info = scene.prefab_info(base, &root, &shaders).unwrap();
        assert_eq!(copy_info.diffs.len(), orig_info.diffs.len(), "{:?} vs {:?}", copy_info.diffs, orig_info.diffs);
        assert_eq!(copy_info.members.len(), 2, "Cap stays removed in the copy");
        let guns: Vec<usize> = (0..scene.objects.len()).filter(|&i| scene.objects[i].name == "Gun").collect();
        assert_eq!(guns.len(), 2);
        assert!(guns.iter().all(|&g| scene.objects[g].layer == 5));
        // The copy saves as a small linked entry and loads back the same.
        let file = scene.to_compact_scene_file(&root, &shaders);
        assert_eq!(file.entries.len(), 2, "two roots, nothing else stored");
        let mut loaded = file.clone();
        crate::prefab::expand_scene_file(&mut loaded, &root, &crate::prefab::PrefabCache::default());
        let mut fresh = LoadedScene::empty();
        let mut sh = ShaderLibrary::default();
        fresh.append_entries_with(None, &mut sh, &root, &registry, &loaded.entries).unwrap();
        let ids = |s: &LoadedScene| {
            let mut v: Vec<String> = s.objects.iter().map(|o| o.id.to_string()).collect();
            v.sort();
            v
        };
        assert_eq!(ids(&scene), ids(&fresh));
        // Duplicating a member of an instance is an added object, not a second link.
        let gun = guns[0];
        let before = scene.prefab_links.len();
        let dup = scene.duplicate_object_linked(gun, &root, &shaders, &registry).unwrap();
        assert_eq!(scene.prefab_links.len(), before);
        assert!(scene.prefab_info(base, &root, &shaders).unwrap().diffs.iter().any(|d| matches!(d.change, zestengine_assets::Change::AddedObject)) || scene.prefab_info(copy, &root, &shaders).unwrap().diffs.iter().any(|d| matches!(d.change, zestengine_assets::Change::AddedObject)));
        let _ = dup;
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unpack_detaches_and_connect_makes_an_instance() {
        let root = temp_root("unpack");
        let asset = make_asset(&root);
        let mut scene = LoadedScene::empty();
        let base = place(&mut scene, &root, &asset);
        let registry = ComponentRegistry::with_builtins();
        assert!(scene.unpack_prefab(base, true));
        assert!(scene.prefab_links.is_empty());
        assert_eq!(scene.prefab_roles(), vec![PrefabRole::None; 3]);

        // Connect a plain subtree to the asset it was saved as.
        let mut plain = LoadedScene::empty();
        let mut shaders = ShaderLibrary::default();
        let entries = [entry("Base", None), entry("Gun", Some(0))];
        plain.append_entries_with(None, &mut shaders, &root, &registry, &entries).unwrap();
        let prefab = plain.prefab_from_object(0, &root, &shaders).unwrap();
        prefab.save(root.join("small.prefab")).unwrap();
        let gun_before = plain.objects[1].id;
        assert!(plain.connect_prefab(0, "small.prefab", &registry));
        assert_ne!(plain.objects[1].id, gun_before, "members take the ids the instance derives");
        assert_eq!(plain.prefab_roles()[1], PrefabRole::Member);
        assert!(plain.prefab_info(0, &root, &shaders).unwrap().diffs.is_empty(), "the connected instance matches its asset");
        let _ = std::fs::remove_dir_all(&root);
    }
}
