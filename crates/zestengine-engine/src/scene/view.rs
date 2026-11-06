//! Editor scene-view hide / lock (the hierarchy eye and lock).
//!
//! View state only, keyed by object id: it never changes an object's `enabled`
//! flag, and it only acts while `edit` is set (the editor sets it while not
//! playing). Play, the runtime, the server and builds leave `edit` false, so
//! the flags are ignored there entirely (and the file strip drops them).
//!
//! Hidden = not drawn in the viewport (meshes, sprites, tilemaps, shapes,
//! gizmos) and not pickable; it also casts no shadow in the editor view. Its
//! light, physics and scripts keep working: hiding is visual only, so a hidden
//! light still lights the scene. Locked = drawn as usual, skipped by viewport
//! click / box select (still selectable from the hierarchy). Both apply to
//! descendants. They are not prefab overrides and not copied into prefabs.

use std::collections::HashSet;

use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneView {
    pub hidden: HashSet<ObjectId>,
    pub locked: HashSet<ObjectId>,
    /// The flags act only while true (editor, not playing).
    pub edit: bool,
}

/// Whether a draw is skipped: only in the editor view, only if hidden.
#[allow(dead_code)] // view helpers used by editor-side callers only
pub fn draw_filtered(edit: bool, hidden: bool) -> bool {
    edit && hidden
}

/// Whether a viewport click / box select may pick the object.
pub fn pickable(edit: bool, hidden: bool, locked: bool) -> bool {
    !(edit && (hidden || locked))
}

/// `own[i]` lifted down the tree: true if the object or any ancestor has it.
#[allow(dead_code)] // view helpers used by editor-side callers only
pub fn inherit(parents: &[Option<usize>], own: &[bool]) -> Vec<bool> {
    let mut out = vec![false; own.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut at = Some(i);
        let mut guard = 0;
        while let Some(k) = at {
            if own[k] {
                *slot = true;
                break;
            }
            guard += 1;
            at = parents.get(k).copied().flatten().filter(|_| guard <= own.len());
        }
    }
    out
}

impl LoadedScene {
    fn view_has(&self, index: usize, set: &HashSet<ObjectId>) -> bool {
        if set.is_empty() {
            return false;
        }
        let mut at = Some(index);
        for _ in 0..=self.objects.len() {
            let Some(k) = at else { return false };
            let Some(o) = self.objects.get(k) else { return false };
            if set.contains(&o.id) {
                return true;
            }
            at = o.parent;
        }
        false
    }

    /// Hidden by its own flag or an ancestor's (regardless of `edit`).
    pub fn view_hidden_flag(&self, index: usize) -> bool {
        self.view_has(index, &self.scene_view.hidden)
    }

    /// Locked by its own flag or an ancestor's (regardless of `edit`).
    pub fn view_locked_flag(&self, index: usize) -> bool {
        self.view_has(index, &self.scene_view.locked)
    }

    /// Skip this object's visuals right now (editor view only).
    pub fn view_hidden(&self, index: usize) -> bool {
        self.scene_view.edit && self.view_hidden_flag(index)
    }

    /// [`active_flags`](Self::active_flags) with scene-hidden objects off: what
    /// the visual passes (particles, world text / UI, water) draw. Equal to
    /// `active_flags` outside the editor view.
    pub fn visible_flags(&self) -> Vec<bool> {
        let mut flags = self.active_flags();
        if self.scene_view.edit && !self.scene_view.hidden.is_empty() {
            for (i, f) in flags.iter_mut().enumerate() {
                if *f && self.view_hidden_flag(i) {
                    *f = false;
                }
            }
        }
        flags
    }

    /// May a viewport click / box select pick it right now.
    pub fn view_pickable(&self, index: usize) -> bool {
        !self.scene_view.edit || pickable(true, self.view_hidden_flag(index), self.view_locked_flag(index))
    }

    /// Set / clear hide on `index` (own flag). Returns true if it changed.
    pub fn set_view_hidden(&mut self, index: usize, on: bool) -> bool {
        let id = self.objects[index].id;
        if on { self.scene_view.hidden.insert(id) } else { self.scene_view.hidden.remove(&id) }
    }

    pub fn set_view_locked(&mut self, index: usize, on: bool) -> bool {
        let id = self.objects[index].id;
        if on { self.scene_view.locked.insert(id) } else { self.scene_view.locked.remove(&id) }
    }

    /// Alt-click isolate: which objects stay shown (the row, its ancestors and
    /// descendants).
    pub fn isolate_keep(&self, i: usize) -> Vec<bool> {
        (0..self.objects.len()).map(|j| j == i || self.is_within(j, i) || self.is_within(i, j)).collect()
    }

    /// A duplicate keeps its source's view state: walks both subtrees in the
    /// same child order and copies hide / lock onto the matching copies.
    pub fn copy_view_flags(&mut self, src_root: usize, dst_root: usize) {
        let children = |s: &Self, p: usize| -> Vec<usize> { (0..s.objects.len()).filter(|&c| s.objects[c].parent == Some(p)).collect() };
        let mut stack = vec![(src_root, dst_root)];
        let mut guard = 0;
        while let Some((a, b)) = stack.pop() {
            guard += 1;
            if guard > self.objects.len() * 2 + 8 || a >= self.objects.len() || b >= self.objects.len() {
                break;
            }
            let (ia, ib) = (self.objects[a].id, self.objects[b].id);
            if self.scene_view.hidden.contains(&ia) {
                self.scene_view.hidden.insert(ib);
            }
            if self.scene_view.locked.contains(&ia) {
                self.scene_view.locked.insert(ib);
            }
            stack.extend(children(self, a).into_iter().zip(children(self, b)));
        }
    }

    /// Ids for the scene file (objects that still exist, in object order).
    pub fn view_ids(&self) -> (Vec<String>, Vec<String>) {
        let pick = |set: &HashSet<ObjectId>| self.objects.iter().filter(|o| set.contains(&o.id)).map(|o| o.id.to_string()).collect();
        (pick(&self.scene_view.hidden), pick(&self.scene_view.locked))
    }

    /// Load the saved ids (unknown ones are dropped).
    pub fn set_view_ids(&mut self, hidden: &[String], locked: &[String]) {
        let by_id: HashMap<String, ObjectId> = self.objects.iter().map(|o| (o.id.to_string(), o.id)).collect();
        self.scene_view.hidden = hidden.iter().filter_map(|s| by_id.get(s).copied()).collect();
        self.scene_view.locked = locked.iter().filter_map(|s| by_id.get(s).copied()).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_act_only_in_the_editor_view() {
        assert!(draw_filtered(true, true) && !draw_filtered(false, true) && !draw_filtered(true, false));
        assert!(!pickable(true, true, false) && !pickable(true, false, true) && pickable(true, false, false));
        assert!(pickable(false, true, true), "play / runtime ignore hide and lock");
    }

    #[test]
    fn inherit_walks_down_and_is_independent_per_flag() {
        let parents = [None, Some(0), Some(1), None];
        let hidden = inherit(&parents, &[true, false, false, false]);
        assert_eq!(hidden, [true, true, true, false]);
        let locked = inherit(&parents, &[false, false, true, false]);
        assert_eq!(locked, [false, false, true, false]);
    }

    fn obj(name: &str, parent: Option<usize>) -> SceneObject {
        SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: name.into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: Vec::new(),
        }
    }

    /// root -> child -> grand, plus a sibling root.
    fn tree() -> LoadedScene {
        let mut s = LoadedScene::empty();
        s.objects.push(obj("root", None));
        s.objects.push(obj("child", Some(0)));
        s.objects.push(obj("grand", Some(1)));
        s.objects.push(obj("other", None));
        s
    }

    #[test]
    fn hide_inherits_down_and_never_touches_enabled() {
        let mut s = tree();
        s.scene_view.edit = true;
        assert!(s.set_view_hidden(0, true));
        assert!(s.view_hidden(0) && s.view_hidden(1) && s.view_hidden(2) && !s.view_hidden(3));
        assert!(s.objects.iter().all(|o| o.enabled), "hiding does not disable");
        assert!((0..4).all(|i| s.is_active(i)), "hidden objects stay active (lights, physics, scripts)");
        // And disabling does not hide.
        s.set_view_hidden(0, false);
        s.objects[1].enabled = false;
        assert!(!s.view_hidden(1) && !s.view_hidden(2) && !s.is_active(2));
    }

    #[test]
    fn play_and_runtime_ignore_hide_and_lock() {
        let mut s = tree();
        s.set_view_hidden(0, true);
        s.set_view_locked(3, true);
        s.scene_view.edit = false;
        assert!((0..4).all(|i| !s.view_hidden(i) && s.view_pickable(i)));
        assert_eq!(s.visible_flags(), s.active_flags());
        s.scene_view.edit = true;
        assert!(!s.view_pickable(0) && !s.view_pickable(2) && !s.view_pickable(3));
        assert_eq!(s.visible_flags(), vec![false, false, false, true]);
        assert_eq!(s.active_flags(), vec![true; 4]);
    }

    #[test]
    fn locked_draws_but_is_not_pickable_and_inherits() {
        let mut s = tree();
        s.scene_view.edit = true;
        s.set_view_locked(1, true);
        assert!(!s.view_hidden(1));
        assert!(s.view_pickable(0) && !s.view_pickable(1) && !s.view_pickable(2));
        assert!(s.view_locked_flag(2) && !s.view_hidden_flag(2));
    }

    #[test]
    fn view_ids_round_trip_and_drop_unknown_ids() {
        let mut s = tree();
        s.set_view_hidden(1, true);
        s.set_view_locked(3, true);
        let (h, l) = s.view_ids();
        assert_eq!((h.len(), l.len()), (1, 1));
        let mut t = LoadedScene::empty();
        t.objects = std::mem::take(&mut s.objects);
        t.set_view_ids(&h, &["not-an-id".to_owned()]);
        assert!(t.view_hidden_flag(1) && t.view_hidden_flag(2) && t.scene_view.locked.is_empty());
    }

    #[test]
    fn a_duplicate_keeps_the_view_state_of_its_source() {
        let mut s = tree();
        s.set_view_hidden(1, true);
        s.set_view_locked(2, true);
        // Copy of the root's subtree appended as 4,5,6.
        s.objects.push(obj("root2", None));
        s.objects.push(obj("child2", Some(4)));
        s.objects.push(obj("grand2", Some(5)));
        s.copy_view_flags(0, 4);
        assert!(s.view_hidden_flag(5) && s.view_hidden_flag(6) && !s.view_hidden_flag(4));
        assert!(s.view_locked_flag(6) && !s.view_locked_flag(5));
    }

    #[test]
    fn isolate_keeps_ancestors_and_descendants() {
        let s = tree();
        assert_eq!(s.isolate_keep(1), vec![true, true, true, false]);
    }

}
