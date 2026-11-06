//! Undo/redo history.
//!
//! Entries are recorded by diffing a pre-frame snapshot of the selection
//! against the post-frame state, so every edit path (gizmo drags, inspector
//! fields, material sliders, drag & drop assignment) is captured without
//! per-widget bookkeeping. Continuous gestures coalesce into single entries.
//!
//! Structural edits (delete, group, reorder, re-parent) shift object indices,
//! so they're recorded as whole object-list snapshots instead
//! ([`UndoEntry::Structure`]).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::MaterialModel;
use glam::{Quat, Vec3};

/// How long after the last merge a gesture stays "open" for coalescing.
const MERGE_WINDOW: Duration = Duration::from_millis(700);

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectState {
    pub name: String,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// The object's network role (NET-3), so a role edit can be undone.
    pub net_role: zestengine_core::NetRole,
    /// Per-component role overrides, by component key.
    pub component_roles: std::collections::BTreeMap<String, zestengine_core::NetRole>,
    /// Components as (registry name, RON). Diffable and restorable.
    pub components: Vec<zestengine_assets::ComponentData>,
}

/// The skybox image sources. Kept as their own state because the equirect 360°
/// image lives on the scene (`LoadedScene::skybox`) while the 6-face cubemap
/// lives on the environment (`WorldEnvironment::skybox_faces`).
#[derive(Clone, Debug, PartialEq)]
pub struct SkyboxState {
    /// Project-relative equirectangular image (None = procedural sky).
    pub equirect: Option<String>,
    /// Cubemap faces in +X,-X,+Y,-Y,+Z,-Z order (overrides the equirect image
    /// once all six are set).
    pub faces: Option<[String; 6]>,
}

#[derive(Clone)]
pub enum UndoEntry {
    /// Transform / rename of a scene object.
    Object {
        index: usize,
        before: ObjectState,
        after: ObjectState,
    },
    /// Property edit of a scene material.
    Material {
        index: usize,
        before: Box<MaterialModel>,
        after: Box<MaterialModel>,
    },
    /// Material slot assignment on an object.
    Assign {
        object: usize,
        /// Which material slot (0 = primary, 1.. = extra slots).
        slot: usize,
        before: usize,
        after: usize,
        /// What the renderer component held for the slot before / after.
        before_slot: zestengine_core::MaterialSlot,
        after_slot: zestengine_core::MaterialSlot,
    },
    /// Property edit of a `.material` file opened in the Inspector.
    FileMaterial {
        path: PathBuf,
        before: Box<MaterialModel>,
        after: Box<MaterialModel>,
    },
    /// An Environment tab edit (sun, ambient, fog, sky, bake settings...).
    Environment {
        before: Box<zestengine_assets::WorldEnvironment>,
        after: Box<zestengine_assets::WorldEnvironment>,
    },
    /// A skybox image change (equirect 360° image or cubemap faces).
    Skybox {
        before: Box<SkyboxState>,
        after: Box<SkyboxState>,
    },
    /// A structural edit: the whole object list before and after.
    Structure {
        before: Box<zestengine_engine::scene::ObjectsSnapshot>,
        after: Box<zestengine_engine::scene::ObjectsSnapshot>,
    },
    /// A prefab Apply: the `.prefab` files it wrote plus the object list
    /// before and after (every instance of the asset re-merged). Undo writes
    /// the old files back and restores the objects, as one step.
    Prefab {
        files: Vec<zestengine_engine::scene::PrefabFileChange>,
        before: Box<zestengine_engine::scene::ObjectsSnapshot>,
        after: Box<zestengine_engine::scene::ObjectsSnapshot>,
    },
    /// One atomic multi-object edit (a multi-select gesture): all sub-entries are
    /// undone/redone together as a single history step, and coalesce as a unit so a
    /// continuous multi-object drag is ONE entry, not hundreds of interleaved ones.
    Group(Vec<UndoEntry>),
    /// Scene-view hide / lock sets before and after (the hierarchy eye / lock, `H`).
    View { before: Box<zestengine_engine::scene::SceneView>, after: Box<zestengine_engine::scene::SceneView> },
    /// Active flag changes from the Inspector: (object, before, after).
    Enabled(Vec<(zestengine_core::ObjectId, bool, bool)>),
    /// One tile paint stroke: the changed cells as (cell, old, new) diffs on a
    /// tilemap object. Never a snapshot of the map.
    Tiles { object: zestengine_core::ObjectId, edits: Vec<zestengine_core::CellEdit> },
}

impl UndoEntry {
    /// Same gesture target? (used for coalescing)
    fn same_target(&self, other: &UndoEntry) -> bool {
        match (self, other) {
            (UndoEntry::Object { index: a, .. }, UndoEntry::Object { index: b, .. }) => a == b,
            (UndoEntry::Material { index: a, .. }, UndoEntry::Material { index: b, .. }) => a == b,
            (UndoEntry::Environment { .. }, UndoEntry::Environment { .. }) => true,
            (UndoEntry::Skybox { .. }, UndoEntry::Skybox { .. }) => true,
            (UndoEntry::FileMaterial { path: a, .. }, UndoEntry::FileMaterial { path: b, .. }) => {
                a == b
            }
            // Same gesture if the groups cover the same targets in the same order
            // (record_edits builds them stably: anchor first, then the others).
            (UndoEntry::Group(a), UndoEntry::Group(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same_target(y))
            }
            _ => false,
        }
    }

    /// Merge `newer` into self (keep self.before, take newer.after).
    fn absorb(&mut self, newer: UndoEntry) {
        match (self, newer) {
            (UndoEntry::Object { after, .. }, UndoEntry::Object { after: new, .. }) => {
                *after = new;
            }
            (UndoEntry::Material { after, .. }, UndoEntry::Material { after: new, .. }) => {
                *after = new;
            }
            (UndoEntry::Environment { after, .. }, UndoEntry::Environment { after: new, .. }) => {
                *after = new;
            }
            (UndoEntry::Skybox { after, .. }, UndoEntry::Skybox { after: new, .. }) => {
                *after = new;
            }
            (UndoEntry::FileMaterial { after, .. }, UndoEntry::FileMaterial { after: new, .. }) => {
                *after = new;
            }
            // Absorb sub-entry by sub-entry (same structure, guaranteed by
            // `same_target`): each keeps its own `before`, takes the newest `after`.
            (UndoEntry::Group(a), UndoEntry::Group(new)) => {
                for (x, y) in a.iter_mut().zip(new) {
                    x.absorb(y);
                }
            }
            _ => {}
        }
    }
}

#[derive(Default)]
pub struct UndoStack {
    undo: Vec<UndoEntry>,
    redo: Vec<UndoEntry>,
    last_push: Option<Instant>,
    /// A paint stroke is open: its edits coalesce however long it pauses.
    stroke: bool,
}

impl UndoStack {
    /// Record an edit. Coalesces with the previous entry only while a continuous
    /// gesture is active (`gesture_active`, e.g. a pointer-held slider/handle drag)
    /// AND it targets the same thing within the merge window — so one drag is one
    /// entry, but two DISCRETE edits (typed value, separate clicks) each get their
    /// own entry instead of silently merging (which made them un-undoable). `Assign`
    /// entries never coalesce.
    pub fn record(&mut self, entry: UndoEntry, gesture_active: bool) {
        self.redo.clear();
        let now = Instant::now();
        let in_window = self
            .last_push
            .is_some_and(|t| self.stroke || now.duration_since(t) < MERGE_WINDOW);
        let coalesce = gesture_active
            && !matches!(entry, UndoEntry::Assign { .. })
            && in_window
            && self
                .undo
                .last()
                .is_some_and(|last| last.same_target(&entry));
        if coalesce {
            self.undo.last_mut().unwrap().absorb(entry);
        } else {
            self.undo.push(entry);
            if self.undo.len() > 256 {
                self.undo.remove(0);
            }
        }
        self.last_push = Some(now);
    }

    /// Open or close a paint stroke. Opening one never merges into the
    /// previous entry, and its edits then coalesce into one entry until it
    /// closes, so one stroke is one undo step.
    pub fn set_stroke(&mut self, stroking: bool) {
        if stroking && !self.stroke {
            self.last_push = None;
        }
        self.stroke = stroking;
    }

    pub fn pop_undo(&mut self) -> Option<UndoEntry> {
        let entry = self.undo.pop()?;
        self.redo.push(entry.clone());
        self.last_push = None; // never coalesce across an undo
        Some(entry)
    }

    pub fn pop_redo(&mut self) -> Option<UndoEntry> {
        let entry = self.redo.pop()?;
        self.undo.push(entry.clone());
        self.last_push = None;
        Some(entry)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Prefab Mode opens: park this history and start an empty one for the
    /// stage. The scene is not reloaded, so the parked steps stay valid and
    /// come back with [`end_stage`](Self::end_stage).
    pub fn begin_stage(&mut self) -> UndoStack {
        std::mem::take(self)
    }

    /// The stage's history ends (the prefab closed or another one opens):
    /// returns its Prefab saves with the stage's objects cut out of their
    /// snapshots, and empties the stack. Other steps edited objects that are
    /// about to leave the scene and are dropped.
    pub fn take_stage_saves(&mut self, stage_objects: &std::collections::HashSet<zestengine_core::ObjectId>) -> Vec<UndoEntry> {
        let saves = std::mem::take(&mut self.undo)
            .into_iter()
            .filter_map(|e| match e {
                UndoEntry::Prefab { files, before, after } => Some(UndoEntry::Prefab {
                    files,
                    before: Box::new(before.without(stage_objects)),
                    after: Box::new(after.without(stage_objects)),
                }),
                _ => None,
            })
            .collect();
        self.clear();
        saves
    }

    /// Prefab Mode closed: the parked history returns, and the stage's saves
    /// become the newest steps (so they can still be undone from the scene).
    pub fn end_stage(&mut self, parked: UndoStack, saves: Vec<UndoEntry>) {
        *self = parked;
        if !saves.is_empty() {
            self.redo.clear();
            self.undo.extend(saves);
            let over = self.undo.len().saturating_sub(256);
            self.undo.drain(..over);
        }
        self.last_push = None;
    }

    /// After a stage save re-merged the instances, object indices moved:
    /// keep only the saves, whose snapshots are by id.
    pub fn keep_only_saves(&mut self) {
        self.undo.retain(|e| matches!(e, UndoEntry::Prefab { .. }));
        self.redo.clear();
        self.last_push = None;
    }

    /// Drop everything (scene reload invalidates indices).
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.last_push = None;
    }
}

/// Undo history of one document a tab edits on its own (an animation clip,
/// an animator controller): snapshots of the document before each change.
/// A drag (pointer held) is one step.
pub struct DocHistory<T> {
    undo: Vec<T>,
    redo: Vec<T>,
    /// A held gesture already pushed its starting state.
    open: bool,
}

impl<T> Default for DocHistory<T> {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), open: false }
    }
}

impl<T: Clone + PartialEq> DocHistory<T> {
    /// After a frame: `before` is the document as the frame began, `now` as
    /// it ended; `gesture` while the pointer is held.
    pub fn observe(&mut self, before: T, now: &T, gesture: bool) {
        if before == *now {
            self.open &= gesture;
            return;
        }
        if !(gesture && self.open) {
            self.undo.push(before);
            if self.undo.len() > 128 {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
        self.open = gesture;
    }

    /// Step back (`undo`) or forward through the history. False when there
    /// is nothing to step to.
    pub fn step(&mut self, doc: &mut T, undo: bool) -> bool {
        let (from, to) = if undo { (&mut self.undo, &mut self.redo) } else { (&mut self.redo, &mut self.undo) };
        let Some(state) = from.pop() else { return false };
        to.push(std::mem::replace(doc, state));
        self.open = false;
        true
    }

    /// Another document was opened.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.open = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(pins: &str) -> ObjectState {
        ObjectState {
            name: "Skirt".into(),
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            net_role: Default::default(),
            component_roles: Default::default(),
            components: vec![zestengine_assets::ComponentData::new("Mesh Cloth", format!("(paint: (pins: \"{pins}\"))"))],
        }
    }

    fn edit(before: &str, after: &str) -> UndoEntry {
        UndoEntry::Object { index: 3, before: state(before), after: state(after) }
    }

    fn after_of(e: &UndoEntry) -> &str {
        match e {
            UndoEntry::Object { after, .. } => &after.components[0].data,
            _ => panic!("object entry"),
        }
    }

    #[test]
    fn a_paint_stroke_is_one_undo_step() {
        let mut stack = UndoStack::default();
        stack.record(edit("a", "b"), false);
        stack.set_stroke(true);
        stack.record(edit("b", "c"), true);
        std::thread::sleep(MERGE_WINDOW + Duration::from_millis(50));
        stack.record(edit("c", "d"), true);
        stack.record(edit("d", "e"), true);
        stack.set_stroke(false);
        // A second stroke right after starts its own entry.
        stack.set_stroke(true);
        stack.record(edit("e", "f"), true);
        stack.set_stroke(false);

        let second = stack.pop_undo().unwrap();
        assert!(after_of(&second).contains("\"f\""));
        let stroke = stack.pop_undo().unwrap();
        match &stroke {
            UndoEntry::Object { before, after, .. } => {
                assert_eq!(before, &state("b"), "the stroke undoes to its start, despite the pause");
                assert_eq!(after, &state("e"));
            }
            _ => panic!(),
        }
        assert!(stack.can_undo(), "the edit before the stroke stays its own step");
        stack.pop_undo();
        assert!(!stack.can_undo());
        assert!(after_of(&stack.pop_redo().unwrap()).contains("\"b\""));
    }

    /// NET-3: object and per-component network roles are part of the diffed state, so changing
    /// them records an entry whose undo carries the old roles back.
    #[test]
    fn role_edits_are_undo_steps() {
        use zestengine_core::NetRole;
        let before = state("a");
        let mut after = before.clone();
        after.net_role = NetRole::Server;
        after.component_roles.insert("Light".into(), NetRole::Client);
        assert_ne!(before, after, "a role change is a change");
        let mut stack = UndoStack::default();
        stack.record(UndoEntry::Object { index: 1, before: before.clone(), after: after.clone() }, false);
        match stack.pop_undo().unwrap() {
            UndoEntry::Object { before: b, after: a, .. } => {
                assert_eq!((b.net_role, b.component_roles.len()), (NetRole::Both, 0));
                assert_eq!((a.net_role, a.component_roles.get("Light")), (NetRole::Server, Some(&NetRole::Client)));
            }
            _ => panic!("object entry"),
        }
        assert!(matches!(stack.pop_redo(), Some(UndoEntry::Object { .. })));
    }

    #[test]
    fn structural_edits_are_single_steps_that_never_merge() {
        let mut stack = UndoStack::default();
        let snap = || Box::new(zestengine_engine::scene::LoadedScene::empty().snapshot_objects());
        stack.record(UndoEntry::Structure { before: snap(), after: snap() }, true);
        stack.record(UndoEntry::Structure { before: snap(), after: snap() }, true);
        assert!(matches!(stack.pop_undo(), Some(UndoEntry::Structure { .. })));
        assert!(matches!(stack.pop_undo(), Some(UndoEntry::Structure { .. })), "two edits, two steps");
        assert!(!stack.can_undo() && stack.can_redo());
    }

    #[test]
    fn placing_a_prefab_undoes_and_redoes_with_the_same_ids() {
        use zestengine_core::ObjectId;
        let registry = zestengine_core::ComponentRegistry::with_builtins();
        let mut scene = zestengine_engine::scene::LoadedScene::empty();
        let before = scene.snapshot_objects();
        let ids = [ObjectId::new(), ObjectId::new()];
        for (k, id) in ids.iter().enumerate() {
            scene.objects.push(zestengine_engine::scene::SceneObject {
                net_role: Default::default(),
                component_roles: Default::default(),
                id: *id,
                name: format!("Placed {k}"),
                render: None,
                extra_render: Vec::new(),
                source: zestengine_assets::ObjectSource::Builtin { mesh: 0 },
                enabled: true,
                static_geometry: false,
                layer: 0,
                parent: (k == 1).then_some(0),
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: Vec::new(),
            });
        }
        let after = scene.snapshot_objects();
        let mut stack = UndoStack::default();
        stack.record(UndoEntry::Structure { before: Box::new(before), after: Box::new(after) }, false);

        let Some(UndoEntry::Structure { before, after }) = stack.pop_undo() else { panic!("structure entry") };
        scene.restore_objects(&before, &registry);
        assert!(scene.objects.is_empty(), "undo removes the placed objects");
        scene.restore_objects(&after, &registry);
        let got: Vec<ObjectId> = scene.objects.iter().map(|o| o.id).collect();
        assert_eq!(got, ids, "redo restores the same ids");
        assert_eq!(scene.objects[1].parent, Some(0));
    }

    fn snap_with(ids: &[zestengine_core::ObjectId]) -> zestengine_engine::scene::ObjectsSnapshot {
        let mut scene = zestengine_engine::scene::LoadedScene::empty();
        for (k, id) in ids.iter().enumerate() {
            scene.objects.push(zestengine_engine::scene::SceneObject {
                net_role: Default::default(),
                component_roles: Default::default(),
                id: *id,
                name: format!("O{k}"),
                render: None,
                extra_render: Vec::new(),
                source: zestengine_assets::ObjectSource::Empty,
                enabled: true,
                static_geometry: false,
                layer: 0,
                parent: None,
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: Vec::new(),
            });
        }
        scene.snapshot_objects()
    }

    #[test]
    fn prefab_mode_keeps_the_scene_history_and_leaves_its_saves_as_steps() {
        use zestengine_core::ObjectId;
        let (a, b, stage_obj) = (ObjectId::new(), ObjectId::new(), ObjectId::new());
        let mut stack = UndoStack::default();
        // The scene's own history before Prefab Mode: two steps, one undone.
        stack.record(UndoEntry::Structure { before: Box::new(snap_with(&[a])), after: Box::new(snap_with(&[a, b])) }, false);
        stack.record(UndoEntry::Structure { before: Box::new(snap_with(&[a, b])), after: Box::new(snap_with(&[a])) }, false);
        stack.pop_undo();
        assert!(stack.can_undo() && stack.can_redo());

        let parked = stack.begin_stage();
        assert!(!stack.can_undo() && !stack.can_redo(), "the stage starts with its own history");
        // In the stage: an edit, then a save, then another edit.
        stack.record(UndoEntry::Structure { before: Box::new(snap_with(&[a, stage_obj])), after: Box::new(snap_with(&[a, stage_obj, b])) }, false);
        stack.record(UndoEntry::Prefab { files: Vec::new(), before: Box::new(snap_with(&[a, stage_obj])), after: Box::new(snap_with(&[a, stage_obj])) }, false);
        stack.keep_only_saves();
        stack.record(UndoEntry::Structure { before: Box::new(snap_with(&[a, stage_obj])), after: Box::new(snap_with(&[a])) }, false);
        assert!(stack.can_undo());

        let stage: std::collections::HashSet<_> = [stage_obj].into();
        let saves = stack.take_stage_saves(&stage);
        assert_eq!(saves.len(), 1, "only the save survives");
        assert!(!stack.can_undo());
        let UndoEntry::Prefab { before, after, .. } = &saves[0] else { panic!("a save") };
        assert_eq!(before.ids(), vec![a], "the stage's objects are cut out of the snapshots");
        assert_eq!(after.ids(), vec![a]);

        stack.end_stage(parked, saves);
        // The scene's history is back (one undoable step under the save, and the save on top).
        assert!(!stack.can_redo(), "a new step ends the redo line");
        assert!(matches!(stack.pop_undo(), Some(UndoEntry::Prefab { .. })), "the save is the newest step");
        assert!(matches!(stack.pop_undo(), Some(UndoEntry::Structure { .. })), "the scene's first step is still there");
        assert!(!stack.can_undo());
    }

    #[test]
    fn leaving_prefab_mode_without_saving_restores_the_history_untouched() {
        use zestengine_core::ObjectId;
        let (a, s) = (ObjectId::new(), ObjectId::new());
        let mut stack = UndoStack::default();
        stack.record(UndoEntry::Structure { before: Box::new(snap_with(&[])), after: Box::new(snap_with(&[a])) }, false);
        stack.pop_undo();
        let parked = stack.begin_stage();
        stack.record(UndoEntry::Structure { before: Box::new(snap_with(&[a])), after: Box::new(snap_with(&[a, s])) }, false);
        let saves = stack.take_stage_saves(&[s].into());
        assert!(saves.is_empty());
        stack.end_stage(parked, saves);
        assert!(stack.can_redo(), "the undone step can still be redone");
        assert!(!stack.can_undo());
    }

    #[test]
    fn doc_history_steps_back_and_forth_and_a_drag_is_one_step() {
        let mut h = DocHistory::default();
        let mut doc = 0;
        // A drag: 0 -> 1 -> 2 -> 3 with the pointer held.
        for v in 1..=3 {
            let before = doc;
            doc = v;
            h.observe(before, &doc, true);
        }
        h.observe(doc, &doc, false);
        // A separate click: 3 -> 10.
        let before = doc;
        doc = 10;
        h.observe(before, &doc, false);
        assert!(h.step(&mut doc, true));
        assert_eq!(doc, 3);
        assert!(h.step(&mut doc, true));
        assert_eq!(doc, 0, "the whole drag was one step");
        assert!(!h.step(&mut doc, true));
        assert!(h.step(&mut doc, false));
        assert_eq!(doc, 3);
    }
}

#[cfg(test)]
mod migrated {
    use super::*;

    /// The engine's `scene::view` test for the `View` undo entry, moved here
    /// with the undo module: it swaps hidden-state sets through the stack.
    #[test]
    fn view_undo_entry_swaps_the_sets() {
        let before = zestengine_engine::scene::SceneView::default();
        let mut after = before.clone();
        after.hidden.insert(zestengine_core::ObjectId::from_raw(7));
        let mut stack = UndoStack::default();
        stack.record(UndoEntry::View { before: Box::new(before.clone()), after: Box::new(after.clone()) }, false);
        let Some(UndoEntry::View { before: b, .. }) = stack.pop_undo() else { panic!("view entry") };
        assert_eq!(*b, before);
        let Some(UndoEntry::View { after: a, .. }) = stack.pop_redo() else { panic!("view entry") };
        assert_eq!(*a, after);
    }
}
