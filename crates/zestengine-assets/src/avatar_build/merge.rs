//! Passes 7 and 9: `Merge Animator` and `Merge Armature`.

use super::*;
use glam::{Mat4, Quat, Vec3};
use zestengine_core::MergePathMode;

fn entry_matrix(e: &SceneEntry) -> Mat4 {
    Mat4::from_scale_rotation_translation(Vec3::from_array(e.scale), Quat::from_array(e.rotation), Vec3::from_array(e.translation))
}

impl Builder<'_> {
    // ------------------------------------------------------ 7 merge animators

    /// Rewrite one motion's clip paths for the avatar. Returns the new motion.
    fn rebase_motion(&mut self, m: &Motion, rebase: &str, who: &str, cache: &mut BTreeMap<String, String>) -> Motion {
        match m {
            Motion::Clip(c) => Motion::Clip(self.rebase_clip_ref(c, rebase, who, cache)),
            Motion::BlendTree(t) => {
                let mut t = t.clone();
                self.rebase_tree(&mut t, rebase, who, cache);
                Motion::BlendTree(t)
            }
        }
    }

    fn rebase_tree(&mut self, t: &mut zestengine_core::BlendTree, rebase: &str, who: &str, cache: &mut BTreeMap<String, String>) {
        for m in &mut t.motions {
            if is_clip_path(&m.clip) {
                m.clip = self.rebase_clip_ref(&m.clip, rebase, who, cache);
            }
            if let Some(sub) = m.tree.as_deref_mut() {
                self.rebase_tree(sub, rebase, who, cache);
            }
        }
    }

    /// An authored clip used by a merged controller, rebased onto the avatar
    /// (its object paths gain `rebase`, and lose the Animator's own prefix).
    /// Clips that need no change keep their path; imported clip names stay.
    fn rebase_clip_ref(&mut self, clip: &str, rebase: &str, who: &str, cache: &mut BTreeMap<String, String>) -> String {
        if !is_clip_path(clip) {
            return clip.to_owned();
        }
        if let Some(done) = cache.get(clip) {
            return done.clone();
        }
        let Some(mut c) = self.load_clip(clip, who) else {
            return clip.to_owned();
        };
        let before = c.clone();
        if !rebase.is_empty() {
            c.rebase_targets(rebase);
        }
        let out = if c == before {
            clip.to_owned()
        } else {
            let stem = std::path::Path::new(clip).file_stem().map(|s| sanitise(&s.to_string_lossy())).unwrap_or_else(|| "clip".into());
            let rel = format!("{}/merged/{}_{stem}.anim", self.out_dir, short_hash(&format!("{who}|{clip}|{rebase}"), 8));
            self.clips.insert(rel.clone(), c);
            rel
        };
        cache.insert(clip.to_owned(), out.clone());
        out
    }

    pub(super) fn merge_animators(&mut self) {
        struct Item {
            order: usize,
            entry: usize,
            m: MergeAnimator,
        }
        let mut items: Vec<Item> = Vec::new();
        for (order, &i) in self.sub.clone().iter().enumerate() {
            for m in comps::<MergeAnimator>(&self.scene.entries[i]) {
                items.push(Item { order, entry: i, m });
            }
        }
        // Replace first, then by layer, priority and hierarchy order.
        items.sort_by_key(|it| (it.m.mode != zestengine_core::MergeMode::Replace, it.m.layer, it.m.priority, it.order));
        let base_wd = {
            let states: Vec<bool> = self.base.layers.iter().flat_map(|l| l.machine.states.iter().map(|s| s.write_defaults)).collect();
            states.is_empty() || states.iter().filter(|&&w| w).count() * 2 >= states.len()
        };
        for it in items {
            let who = format!("{} (Merge Animator)", if self.paths[it.entry].is_empty() { self.name_of(it.entry).to_owned() } else { self.paths[it.entry].clone() });
            if it.m.controller.is_empty() {
                self.warn(format!("{who}: no controller"));
                continue;
            }
            let path = zestengine_core::asset_ref::resolve_path(self.env.project_root, &it.m.controller);
            let mut ctrl = match crate::load_anim_controller(&path) {
                Ok(c) => c,
                Err(e) => {
                    self.warn(format!("{who}: controller {} could not be read ({e:#}); skipped", it.m.controller));
                    continue;
                }
            };
            // Parameters follow the scopes in effect where the merge sits.
            let mut chain: Vec<(String, String)> = self
                .remaps
                .iter()
                .filter(|r| self.is_within(it.entry, r.at))
                .map(|r| (r.from.clone(), r.to.clone(), ))
                .collect();
            chain.sort();
            for (from, to) in &chain {
                ctrl.rename_parameter(from, to);
            }
            // Clip paths: relative to the merge object, or already to the root.
            let to_anim_prefix = match it.m.path_mode {
                MergePathMode::Relative => match self.to_anim(&self.paths[it.entry]) {
                    Some(p) => p,
                    None => {
                        self.warn(format!("{who}: the object is outside the Animator's object; clip paths are used as written"));
                        String::new()
                    }
                },
                MergePathMode::Absolute => String::new(),
            };
            let mut cache = BTreeMap::new();
            for layer in &mut ctrl.layers {
                let machine = std::mem::take(&mut layer.machine);
                let mut machine = machine;
                let mut states = std::mem::take(&mut machine.states);
                for s in &mut states {
                    let m = s.motion.clone();
                    s.motion = self.rebase_motion(&m, &to_anim_prefix, &who, &mut cache);
                    if it.m.match_write_defaults {
                        s.write_defaults = base_wd;
                    }
                }
                machine.states = states;
                layer.machine = machine;
                for o in &mut layer.motion_overrides {
                    let m = o.motion.clone();
                    o.motion = self.rebase_motion(&m, &to_anim_prefix, &who, &mut cache);
                }
            }
            if it.m.mode == zestengine_core::MergeMode::Replace {
                self.replace_base = true;
            }
            for p in ctrl.parameters {
                if !self.ctrl_params.iter().any(|q| q.name == p.name) {
                    self.ctrl_params.push(p);
                }
            }
            for l in &mut ctrl.layers {
                l.name = format!("[merge] {}", l.name);
            }
            self.merged.push(MergedSet { layer: it.m.layer, layers: ctrl.layers });
            self.out.layers.push(GeneratedLayer { name: format!("{} ({})", it.m.controller, it.m.layer.label()), source: who });
        }
        let _ = &self.merged;
    }

    // -------------------------------------------------------- 9 merge armature

    fn world_matrix(&self, i: usize) -> Mat4 {
        let mut m = entry_matrix(&self.scene.entries[i]);
        let mut k = self.scene.entries[i].parent;
        let mut guard = 0;
        while let Some(p) = k {
            m = entry_matrix(&self.scene.entries[p]) * m;
            guard += 1;
            if guard > 256 {
                break;
            }
            k = self.scene.entries[p].parent;
        }
        m
    }

    fn children_of(&self, p: usize) -> Vec<usize> {
        (0..self.scene.entries.len()).filter(|&i| self.scene.entries[i].parent == Some(p) && i != p).collect()
    }

    fn strip_affixes(&self, name: &str, prefix: &str, suffix: &str) -> String {
        let n = name.strip_prefix(prefix).unwrap_or(name);
        n.strip_suffix(suffix).unwrap_or(n).to_owned()
    }

    /// Zip the children of `outfit` into the children of `avatar`.
    #[allow(clippy::too_many_arguments)]
    fn zip_bones(&mut self, outfit: usize, avatar: usize, cfg: &MergeArmature, removed: &mut BTreeSet<usize>, who: &str) {
        for c in self.children_of(outfit) {
            let old_path = self.paths[c].clone();
            let key = self.strip_affixes(&self.scene.entries[c].name, &cfg.prefix, &cfg.suffix);
            let partner = self
                .children_of(avatar)
                .into_iter()
                .find(|&t| !removed.contains(&t) && !self.is_within(t, outfit) && self.scene.entries[t].name == key);
            match partner {
                Some(t) => {
                    self.renames.push((old_path, self.paths[t].clone()));
                    self.zip_bones(c, t, cfg, removed, who);
                    // The outfit bone's own components move to the avatar bone.
                    let moved: Vec<ComponentData> = std::mem::take(&mut self.scene.entries[c].components);
                    for comp in moved {
                        if !self.scene.entries[t].components.iter().any(|x| x.kind == comp.kind) {
                            self.scene.entries[t].components.push(comp);
                        }
                    }
                    removed.insert(c);
                }
                None => {
                    // No partner: keep the bone, under the avatar's bone, in place.
                    let world = self.world_matrix(c);
                    let parent_world = self.world_matrix(avatar);
                    let local = parent_world.inverse() * world;
                    let (s, r, t) = local.to_scale_rotation_translation();
                    let taken = self.children_of(avatar).into_iter().any(|o| o != c && self.scene.entries[o].name == self.scene.entries[c].name);
                    let mut new_name = self.scene.entries[c].name.clone();
                    if taken && cfg.mangle_names {
                        new_name = format!("{new_name}.{}", short_hash(&old_path, 6));
                    } else if taken {
                        self.warn(format!("{who}: bone '{old_path}' has no partner and its name is taken under '{}'", self.paths[avatar]));
                    }
                    let e = &mut self.scene.entries[c];
                    e.parent = Some(avatar);
                    e.name = new_name.clone();
                    e.translation = t.to_array();
                    e.rotation = r.to_array();
                    e.scale = s.to_array();
                    let new_path = if self.paths[avatar].is_empty() { new_name } else { format!("{}/{new_name}", self.paths[avatar]) };
                    if new_path != old_path {
                        self.renames.push((old_path, new_path));
                    }
                }
            }
        }
    }

    pub(super) fn merge_armature(&mut self) {
        let mut jobs: Vec<(usize, MergeArmature)> = Vec::new();
        for &i in &self.sub {
            for m in comps::<MergeArmature>(&self.scene.entries[i]) {
                jobs.push((i, m));
            }
        }
        if jobs.is_empty() {
            return;
        }
        if !self.env.structural {
            self.warn("Merge Armature moves objects, so it only runs when a package is built; the outfit stays separate here");
            return;
        }
        if self.scene.entries.iter().any(|e| e.model.is_some() || e.model_part.is_some()) {
            self.warn("Merge Armature was skipped: the scene stores model instances in compact form (expand them first)");
            return;
        }
        let mut removed: BTreeSet<usize> = BTreeSet::new();
        for (m, cfg) in jobs {
            let who = format!("{} (Merge Armature)", self.paths[m]);
            let target = if cfg.target.is_empty() {
                self.resolve_path("Armature").unwrap_or(self.root)
            } else {
                match self.resolve_path(&cfg.target) {
                    Some(t) => t,
                    None => {
                        self.error(format!("{who}: no object at '{}' to merge into", cfg.target));
                        continue;
                    }
                }
            };
            if target == m || self.is_within(target, m) {
                self.error(format!("{who}: the merge target is inside the object being merged"));
                continue;
            }
            // Rebinding reads the outfit's place in the hierarchy, so it runs before the bones move.
            self.rebind_skins(m, &cfg, &who);
            self.zip_bones(m, target, &cfg, &mut removed, &who);
            // The component's own object is spent once its bones moved.
            if self.children_of(m).iter().all(|c| removed.contains(c)) {
                removed.insert(m);
            }
        }
        // Rebuild the entry list without the removed ones, parents first.
        let n = self.scene.entries.len();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut placed = vec![false; n];
        let keep: Vec<usize> = (0..n).filter(|i| !removed.contains(i)).collect();
        let parent_of = |scene: &SceneFile, mut i: usize| {
            // A removed parent hands its children to the next kept ancestor.
            let mut guard = 0;
            loop {
                match scene.entries[i].parent {
                    Some(p) if removed.contains(&p) && guard < 256 => {
                        i = p;
                        guard += 1;
                    }
                    other => return other,
                }
            }
        };
        let mut progress = true;
        while progress {
            progress = false;
            for &i in &keep {
                if placed[i] {
                    continue;
                }
                let ready = parent_of(self.scene, i).is_none_or(|p| placed[p]);
                if ready {
                    placed[i] = true;
                    order.push(i);
                    progress = true;
                }
            }
        }
        let mut new_index = vec![usize::MAX; n];
        for (new, &old) in order.iter().enumerate() {
            new_index[old] = new;
        }
        let parents: Vec<Option<usize>> = order.iter().map(|&o| parent_of(self.scene, o).map(|p| new_index[p])).collect();
        let mut entries: Vec<SceneEntry> = order.iter().map(|&o| self.scene.entries[o].clone()).collect();
        for (e, p) in entries.iter_mut().zip(parents) {
            e.parent = p;
        }
        self.scene.entries = entries;
        self.root = new_index[self.root];
        self.animator = self.animator.map(|a| new_index[a]).filter(|&a| a != usize::MAX);
        self.refresh_paths();

        // Clips and recorded properties follow the moved objects: specific
        // (deep) renames first so a parent's rename does not hide them.
        let mut renames = std::mem::take(&mut self.renames);
        renames.reverse();
        for (old, new) in &renames {
            let (old_a, new_a) = (self.to_anim(old), self.to_anim(new));
            let (Some(old_a), Some(new_a)) = (old_a, new_a) else { continue };
            for clip in self.clips.values_mut() {
                clip.rename_target(&old_a, &new_a);
            }
            let moved: Vec<(PropKey, Option<[f32; 4]>)> = self
                .keys
                .iter()
                .filter(|(k, _)| k.target == old_a || k.target.starts_with(&format!("{old_a}/")))
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            for (k, v) in moved {
                self.keys.remove(&k);
                let rest = k.target.strip_prefix(&old_a).unwrap_or("").to_owned();
                self.keys.insert(PropKey { target: format!("{new_a}{rest}"), ..k }, v);
            }
        }
        self.renames = renames;
    }
}
