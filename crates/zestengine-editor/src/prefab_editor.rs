//! Editor side of prefabs: what the Hierarchy and Inspector show about a
//! prefab instance, the Overrides dropdown's actions, and Prefab Mode.
//!
//! Prefab Mode never reloads the scene: the open prefab's objects are added to
//! it (see `scene::stage`), either alone (isolated) or over the instance they
//! were opened from (in context, the rest of the scene locked). The scene's
//! undo history is parked while the prefab is open and comes back when it
//! closes, with the prefab's saves as the newest steps. Saving writes the
//! `.prefab` (by the Save button, or automatically once edits settle) and
//! re-merges every instance of it in place.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use zestengine_assets::{ApplyLevel, Change, InstanceInfo, PrefabScope};
use zestengine_core::ObjectId;
use crate::prefab_ui::{
    self, OverridesState, PrefabChangeRow, PrefabComponent, PrefabFieldRow, PrefabInspect, PrefabPanelData, PrefabRemovedComponent, PrefabRequest,
    PrefabRowKind, PrefabTarget,
};
use crate::{PrefabMark, PrefabMenu, PrefabRole};

use super::{EditorAction, EngineApp, Selection, UndoEntry};
use zestengine_engine::scene::{self, LoadedScene, StageMode, Trs};
use crate::undo::UndoStack;

/// A diff is recomputed at most this often after an edit.
const INFO_MIN_AGE: Duration = Duration::from_millis(150);
/// Instances nobody is looking at are re-diffed this often (one per frame).
const INFO_IDLE_AGE: Duration = Duration::from_millis(1500);
/// Even the selected instance is re-diffed after this long without an edit hint.
const INFO_MAX_AGE: Duration = Duration::from_secs(5);
/// Edits must be this old before Prefab Mode saves them by itself.
const AUTOSAVE_DELAY: Duration = Duration::from_millis(700);

/// Where Prefab Mode goes next.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum StageGo {
    /// Open this prefab (project-relative) on top of the current stage. In
    /// context needs the scene object it is an instance of.
    Open(String, StageMode, Option<ObjectId>),
    /// Leave the current stage.
    Back,
    /// Jump to a crumb (0 = the scene).
    Crumb(usize),
}

/// One prefab of the Prefab Mode breadcrumb.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Level {
    pub asset: String,
    pub mode: StageMode,
    /// The scene object the prefab was opened from.
    pub instance: Option<ObjectId>,
    /// Where the prefab sits (in context): the instance's world transform
    /// when it was opened.
    pub place: Option<Trs>,
}

/// The breadcrumb after `go`. In context only makes sense over an instance;
/// without one the prefab opens isolated. Opening what is already open changes
/// nothing.
pub(super) fn next_levels(current: &[Level], go: &StageGo) -> Vec<Level> {
    let mut next = current.to_vec();
    match go {
        StageGo::Open(path, mode, instance) => {
            if next.last().map(|l| &l.asset) != Some(path) {
                let mode = if instance.is_some() { *mode } else { StageMode::Isolated };
                next.push(Level { asset: path.clone(), mode, instance: *instance, place: None });
            }
        }
        StageGo::Back => {
            next.pop();
        }
        StageGo::Crumb(i) => next.truncate(*i),
    }
    next
}

pub(super) enum PrefabAction {
    /// A Hierarchy arrow / context-menu command on an object.
    Menu(usize, PrefabMenu),
    /// Requests the Inspector or the Overrides dropdown raised.
    Requests(Vec<PrefabRequest>),
    Stage(StageGo),
    /// The Prefab Mode prompt was answered: save first (or not), then go.
    StageConfirmed(StageGo, bool),
}

#[derive(Default)]
pub(super) struct Stage {
    /// Open prefabs, outermost first; the last is in the scene. Empty = no Prefab Mode.
    pub levels: Vec<Level>,
    /// The scene Prefab Mode started from (it stays loaded).
    pub main_scene: Option<PathBuf>,
    pub autosave: bool,
    /// Since when the stage has had unsaved edits.
    dirty_since: Option<Instant>,
    /// Waiting for an answer about unsaved edits.
    pending: Option<StageGo>,
    /// The scene's undo history while Prefab Mode is open.
    parked_undo: Option<UndoStack>,
    /// Whether the scene had unsaved edits when Prefab Mode opened.
    parked_dirty: bool,
    /// Saves made at levels already closed, for the scene's history.
    carried: Vec<UndoEntry>,
}

impl Stage {
    pub fn active(&self) -> bool {
        !self.levels.is_empty()
    }
}

struct CachedInfo {
    /// None: the asset can't be read.
    info: Option<InstanceInfo>,
    at: Instant,
}

pub(super) struct PrefabEditor {
    pub overrides: OverridesState,
    infos: HashMap<ObjectId, CachedInfo>,
    /// Hierarchy markers per object, rebuilt each frame from the cached infos.
    pub marks: Vec<PrefabMark>,
    /// What the Inspector shows for the selected object, if it is in an instance.
    pub inspect: Option<PrefabInspect>,
    /// The Overrides dropdown for the selected instance root.
    pub panel: Option<PrefabPanelData>,
    /// The instance root the inspect/panel data was built for.
    root: Option<ObjectId>,
    /// Apply levels behind the indexes in `inspect.levels` / `panel.levels`.
    object_levels: Vec<ApplyLevel>,
    pub stage: Stage,
}

impl Default for PrefabEditor {
    fn default() -> Self {
        Self {
            overrides: OverridesState::default(),
            infos: HashMap::new(),
            marks: Vec::new(),
            inspect: None,
            panel: None,
            root: None,
            object_levels: Vec::new(),
            stage: Stage { autosave: true, ..Default::default() },
        }
    }
}

fn stem(path: &str) -> String {
    std::path::Path::new(path).file_stem().map_or_else(|| path.to_owned(), |s| s.to_string_lossy().into_owned())
}

/// Labels the Inspector's rows use for an object field.
fn entry_labels(path: &str) -> &'static [&'static str] {
    match path {
        "translation" => &["translation", "position"],
        "rotation" => &["rotation"],
        "scale" => &["scale"],
        "name" => &["name"],
        "enabled" => &["enabled", "active"],
        "static_geometry" => &["staticgeometry", "static", "contributegi"],
        "lightmap_scale" => &["lightmapscale", "scaleinlightmap"],
        "cast_shadows" => &["castshadows"],
        "layer" => &["layer"],
        "material" | "extra_materials" => &["material"],
        _ => &[],
    }
}

impl PrefabEditor {
    /// A scene load replaced the scene Prefab Mode was open in: forget it.
    pub fn stage_closed_by_load(&mut self) {
        let autosave = self.stage.autosave;
        self.stage = Stage { autosave, ..Default::default() };
        self.invalidate();
    }

    /// Forget cached diffs (after anything that restructures instances).
    pub fn invalidate(&mut self) {
        self.infos.clear();
        self.root = None;
        self.inspect = None;
        self.panel = None;
    }

    /// The cached diff of the instance at `root`. Recomputed when missing,
    /// when an edit may have happened (`hint`) and the copy is older than
    /// [`INFO_MIN_AGE`], or when older than `max_age`. An unreadable asset is
    /// retried at the same pace.
    fn info(&mut self, scene: &LoadedScene, root: usize, project_root: &std::path::Path, shaders: &zestengine_engine::shaders::ShaderLibrary, hint: bool, max_age: Duration) -> Option<&InstanceInfo> {
        let id = scene.objects.get(root)?.id;
        if self.infos.get(&id).is_none_or(|c| c.at.elapsed() > max_age || (hint && c.at.elapsed() > INFO_MIN_AGE)) {
            let info = scene.prefab_info(root, project_root, shaders);
            self.infos.insert(id, CachedInfo { info, at: Instant::now() });
        }
        self.infos.get(&id).and_then(|c| c.info.as_ref())
    }

    /// Recompute the Hierarchy markers and the Inspector data for the
    /// selected object. Cheap when nothing is selected in an instance: the
    /// diff of the selected instance is refreshed at most every 400 ms and
    /// one other instance per call.
    pub fn refresh(&mut self, scene: &LoadedScene, selected: Option<usize>, project_root: &std::path::Path, shaders: &zestengine_engine::shaders::ShaderLibrary, hint: bool) {
        self.infos.retain(|id, _| scene.prefab_links.contains_key(id));
        let roles = scene.prefab_roles();
        self.marks = vec![PrefabMark::default(); roles.len()];
        if scene.prefab_links.is_empty() {
            self.inspect = None;
            self.panel = None;
            self.root = None;
            return;
        }
        let outer_roots: Vec<usize> = (0..roles.len()).filter(|&i| roles[i] == scene::PrefabRole::Root).collect();
        // The selected instance is kept fresh; one other each call.
        let selected_root = selected.and_then(|s| scene.prefab_root_of(s, true));
        if let Some(r) = selected_root {
            self.info(scene, r, project_root, shaders, hint, INFO_MAX_AGE);
        }
        if let Some(&r) = outer_roots.iter().filter(|&&r| Some(r) != selected_root).min_by_key(|&&r| self.infos.get(&scene.objects[r].id).map(|c| c.at)) {
            self.info(scene, r, project_root, shaders, false, INFO_IDLE_AGE);
        }
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); roles.len()];
        for (i, o) in scene.objects.iter().enumerate() {
            if let Some(p) = o.parent.filter(|p| *p < children.len() && *p != i) {
                children[p].push(i);
            }
        }
        // Markers.
        for (i, role) in roles.iter().enumerate() {
            let mark = &mut self.marks[i];
            mark.role = match role {
                scene::PrefabRole::None => PrefabRole::None,
                scene::PrefabRole::Member => PrefabRole::Member,
                scene::PrefabRole::NestedRoot => PrefabRole::NestedRoot,
                scene::PrefabRole::Root => PrefabRole::Root,
            };
        }
        for &r in &outer_roots {
            let Some(c) = self.infos.get(&scene.objects[r].id) else { continue };
            let Some(info) = &c.info else {
                self.marks[r].missing = true;
                continue;
            };
            let live: std::collections::HashSet<&str> = info.members.iter().map(|(_, l)| l.as_str()).collect();
            let changed = info.diffs.len();
            let removed = info.diffs.iter().filter(|d| matches!(d.change, Change::RemovedObject)).count();
            let link_unused = scene.prefab_link(r).map_or(0, |l| l.unused().count());
            self.marks[r].changed = changed + link_unused > 0;
            self.marks[r].removed = removed as u16;
            let mut stack = vec![r];
            let mut seen = 0;
            while let Some(p) = stack.pop() {
                seen += 1;
                if seen > roles.len() {
                    break;
                }
                stack.extend(children[p].iter().copied());
                if p != r && !live.contains(scene.objects[p].id.to_string().as_str()) {
                    self.marks[p].added = true;
                }
            }
        }
        // Inspector data for the selected object.
        self.inspect = None;
        self.panel = None;
        self.root = None;
        let (Some(sel), Some(root)) = (selected, selected_root) else { return };
        let Some(info) = self.infos.get(&scene.objects[root].id).and_then(|c| c.info.as_ref()) else { return };
        let live_id = scene.objects[sel].id.to_string();
        let image_id = info.members.iter().find(|(_, l)| *l == live_id).map(|(i, _)| i.clone());
        self.root = Some(scene.objects[root].id);

        // Apply levels for the selected object.
        let load = |p: &str| scene.prefabs.get(project_root, p);
        let outer_name = stem(&info.path);
        let mut levels: Vec<(String, ApplyLevel)> = Vec::new();
        let mut nested: Vec<(String, String)> = Vec::new(); // (prefix, asset path)
        if let Some(img) = &image_id
            && img.contains('/')
            && let Some(outer) = zestengine_assets::asset_image(&info.path, &load, 1)
        {
            let parts: Vec<&str> = img.split('/').collect();
            for k in 1..parts.len() {
                let prefix = parts[..k].join("/");
                if let Some(path) = outer.iter().find(|e| e.id == prefix).and_then(|e| e.prefab.as_ref()).map(|l| l.path.clone()) {
                    nested.push((prefix, path));
                }
            }
        }
        if nested.is_empty() {
            levels.push((format!("Apply to Prefab '{outer_name}'"), ApplyLevel::outermost()));
        } else {
            levels.push((format!("Apply as Override in Prefab '{outer_name}'"), ApplyLevel::outermost()));
            let last = nested.len() - 1;
            for (k, (prefix, path)) in nested.into_iter().enumerate() {
                let label = if k == last { format!("Apply to Prefab '{}'", stem(&path)) } else { format!("Apply as Override in Prefab '{}'", stem(&path)) };
                levels.push((label, ApplyLevel { nested: Some(prefix) }));
            }
        }
        self.object_levels = levels.iter().map(|(_, l)| l.clone()).collect();

        let mut inspect = PrefabInspect {
            target: image_id.clone().unwrap_or_default(),
            levels: levels.iter().map(|(s, _)| s.clone()).collect(),
            added: image_id.is_none(),
            is_root: roles[sel] != scene::PrefabRole::Member,
            ..Default::default()
        };
        for d in info.diffs.iter().filter(|d| d.live_id == live_id) {
            let Change::Modified { fields, .. } = &d.change else { continue };
            for f in fields {
                match prefab_ui::split_path(&f.path) {
                    (Some(key), field) => {
                        let comp = match inspect.components.iter().position(|c| c.key == key) {
                            Some(i) => i,
                            None => {
                                inspect.components.push(PrefabComponent { key: key.to_owned(), ..Default::default() });
                                inspect.components.len() - 1
                            }
                        };
                        let comp = &mut inspect.components[comp];
                        match field {
                            Some(field) => {
                                let field = field.trim_start_matches('@');
                                // The row may be labelled after something other than the field.
                                for row in prefab_ui::rows_of_field(key.split('#').next().unwrap_or(key), field) {
                                    comp.fields.push((row, f.path.clone()));
                                }
                            }
                            None if f.pristine.is_empty() => comp.added = true,
                            None => comp.whole_path = Some(f.path.clone()),
                        }
                    }
                    _ => {
                        for label in entry_labels(&f.path) {
                            inspect.entry.push(((*label).to_owned(), f.path.clone()));
                        }
                    }
                }
            }
            if let Change::Modified { removed_components, .. } = &d.change {
                for key in removed_components {
                    inspect.removed_components.push(PrefabRemovedComponent { key: key.clone(), kind: key.split('#').next().unwrap_or(key).to_owned() });
                }
            }
        }
        // Whole-component changes also colour every field of that component.
        self.inspect = Some(inspect);

        // The Overrides dropdown belongs to outermost roots.
        if sel == root && roles[sel] == scene::PrefabRole::Root {
            let rows = info
                .diffs
                .iter()
                .map(|d| match &d.change {
                    Change::Modified { fields, removed_components } => PrefabChangeRow {
                        kind: PrefabRowKind::Modified,
                        name: d.name.clone(),
                        id: d.image_id.clone(),
                        fields: fields.iter().map(|f| PrefabFieldRow { path: f.path.clone(), pristine: f.pristine.clone(), live: f.live.clone() }).collect(),
                        removed_components: removed_components.clone(),
                    },
                    Change::AddedObject => PrefabChangeRow { kind: PrefabRowKind::Added, name: d.name.clone(), id: d.live_id.clone(), fields: Vec::new(), removed_components: Vec::new() },
                    Change::RemovedObject => PrefabChangeRow { kind: PrefabRowKind::RemovedObject, name: d.name.clone(), id: d.image_id.clone(), fields: Vec::new(), removed_components: Vec::new() },
                })
                .collect();
            self.panel = Some(PrefabPanelData {
                asset_name: format!("{outer_name}.prefab"),
                asset_path: info.path.clone(),
                levels: vec![format!("Apply to Prefab '{outer_name}'")],
                rows,
                unused: scene.prefab_link(root).map_or(0, |l| l.unused().count()),
                missing: false,
            });
        }
    }
}

fn to_scope(t: &PrefabTarget) -> PrefabScope {
    match t {
        PrefabTarget::All => PrefabScope::All,
        PrefabTarget::Object(o) => PrefabScope::Object(o.clone()),
        PrefabTarget::Property { target, path } => PrefabScope::Property { target: target.clone(), path: path.clone() },
        PrefabTarget::Added(id) => PrefabScope::Added(id.clone()),
        PrefabTarget::RemovedObject(id) => PrefabScope::RemovedObject(id.clone()),
        PrefabTarget::RemovedComponent { target, key } => PrefabScope::RemovedComponent { target: target.clone(), key: key.clone() },
    }
}

fn flatten(reqs: Vec<PrefabRequest>, out: &mut Vec<PrefabRequest>) {
    for r in reqs {
        match r {
            PrefabRequest::Batch(inner) => flatten(inner, out),
            other => out.push(other),
        }
    }
}

impl EngineApp {
    /// Once per frame: act on what the prefab widgets raised, run the
    /// Prefab Mode auto-save, and show the leave prompt.
    pub(super) fn prefab_tick(&mut self, ctx: &egui::Context) {
        let requests = prefab_ui::take_requests(ctx);
        if !requests.is_empty() {
            self.actions.push(EditorAction::Prefab(PrefabAction::Requests(requests)));
        }
        // Auto-save once edits have settled (no drag, no text field in use).
        if self.prefab_editor.stage.active() {
            if self.scene_dirty {
                let since = *self.prefab_editor.stage.dirty_since.get_or_insert_with(Instant::now);
                let busy = ctx.input(|i| i.pointer.any_down()) || ctx.memory(|m| m.focused().is_some());
                if self.prefab_editor.stage.autosave && !busy && since.elapsed() >= AUTOSAVE_DELAY && !self.playing {
                    self.save_prefab_stage();
                } else if self.prefab_editor.stage.autosave {
                    ctx.request_repaint_after(AUTOSAVE_DELAY);
                }
            } else {
                self.prefab_editor.stage.dirty_since = None;
            }
        }
        if let Some(go) = self.prefab_editor.stage.pending.clone() {
            let mut answer = None;
            egui::Window::new("Unsaved prefab changes")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label("This has unsaved changes. Save before leaving?");
                    crate::widgets::dialog_footer(ui, |ui| {
                        if crate::widgets::primary_button(ui, None, "Save").clicked() {
                            answer = Some(Some(true));
                        }
                        if crate::widgets::secondary_button(ui, None, "Discard").clicked() {
                            answer = Some(Some(false));
                        }
                        if crate::widgets::secondary_button(ui, None, "Cancel").clicked() {
                            answer = Some(None);
                        }
                    });
                });
            match answer {
                Some(Some(save)) => {
                    self.prefab_editor.stage.pending = None;
                    self.actions.push(EditorAction::Prefab(PrefabAction::StageConfirmed(go, save)));
                }
                Some(None) => self.prefab_editor.stage.pending = None,
                None => {}
            }
        }
    }

    /// The header bar over the dock while a prefab is open.
    pub(super) fn prefab_stage_bar(&mut self, ctx: &egui::Context) {
        if !self.prefab_editor.stage.active() {
            return;
        }
        let scene_name = self
            .prefab_editor
            .stage
            .main_scene
            .as_ref()
            .and_then(|p| p.file_stem())
            .map_or_else(|| "Scene".to_owned(), |s| s.to_string_lossy().into_owned());
        let mut crumbs = vec![scene_name];
        crumbs.extend(self.prefab_editor.stage.levels.iter().map(|l| format!("{}.prefab", stem(&l.asset))));
        let mut autosave = self.prefab_editor.stage.autosave;
        let dirty = self.scene_dirty;
        let mut out = prefab_ui::StageBarResponse::default();
        egui::TopBottomPanel::top("zestengine-prefab-stage-bar").show_separator_line(true).show(ctx, |ui| {
            out = prefab_ui::stage_bar(ui, prefab_ui::StageBar { crumbs: &crumbs, dirty, autosave: &mut autosave });
        });
        self.prefab_editor.stage.autosave = autosave;
        if out.save {
            self.save_prefab_stage();
        }
        if out.back {
            self.actions.push(EditorAction::Prefab(PrefabAction::Stage(StageGo::Back)));
        } else if let Some(i) = out.crumb {
            self.actions.push(EditorAction::Prefab(PrefabAction::Stage(StageGo::Crumb(i))));
        }
    }

    /// Write the open prefab (its objects, nested instances stored as their
    /// changes), re-merge every instance of it in the scene, and record the
    /// save as one undo step.
    pub(super) fn save_prefab_stage(&mut self) {
        let Some(level) = self.prefab_editor.stage.levels.last().cloned() else { return };
        let entries = self.scene.stage_entries(&self.project_root, &self.shaders);
        if entries.iter().filter(|e| e.parent.is_none()).count() > 1 {
            self.tasks.notify("A prefab has one root object: extra top-level objects are not part of it", crate::tasks::NotifyLevel::Warn);
        }
        let before = self.scene.snapshot_objects();
        let result = self.scene.save_stage(self.renderer.as_mut(), &mut self.shaders, &self.project_root, &self.components, &level.asset, entries);
        match result {
            Ok((change, remap)) => {
                tracing::info!("prefab saved to {}", change.path.display());
                let map = |i: usize| remap.get(i).copied().flatten();
                self.selection = match &self.selection {
                    Selection::Object(i) => map(*i).map_or(Selection::None, Selection::Object),
                    other => other.clone(),
                };
                let moved: std::collections::BTreeSet<usize> = self.multi_objects.iter().filter_map(|&i| map(i)).collect();
                self.multi_objects = moved.into_iter().collect();
                let after = self.scene.snapshot_objects();
                // Indices moved with the re-merge: older edits would hit other objects.
                self.undo_stack.keep_only_saves();
                self.undo_stack.record(UndoEntry::Prefab { files: vec![change], before: Box::new(before), after: Box::new(after) }, false);
                self.suppress_undo_record = true;
                self.scene_dirty = false;
                self.prefab_editor.stage.dirty_since = None;
                self.prefab_editor.invalidate();
                self.rt_gi.invalidate();
            }
            Err(e) => {
                tracing::error!("saving prefab: {e:#}");
                self.tasks.notify(format!("Saving the prefab failed: {e}"), crate::tasks::NotifyLevel::Warn);
            }
        }
    }

    /// Where the project file should say the open scene is. The scene stays
    /// loaded in Prefab Mode, so it is simply the scene's own path.
    pub(super) fn scene_path_for_project(&self) -> Option<PathBuf> {
        if self.prefab_editor.stage.active() { self.prefab_editor.stage.main_scene.clone() } else { self.current_scene_path.clone() }
    }

    /// Open, switch or leave Prefab Mode, asking first about unsaved edits to
    /// the prefab that is open. The scene itself is never reloaded.
    fn stage_go(&mut self, go: StageGo, confirmed: Option<bool>) {
        if self.playing {
            self.set_status("Stop Play mode to edit a prefab", false);
            return;
        }
        let active = self.prefab_editor.stage.active();
        let mut next = next_levels(&self.prefab_editor.stage.levels, &go);
        if next == self.prefab_editor.stage.levels {
            return;
        }
        if active && self.scene_dirty {
            let silent = self.prefab_editor.stage.autosave;
            match confirmed {
                None if !silent => {
                    self.prefab_editor.stage.pending = Some(go);
                    return;
                }
                Some(false) => {}
                _ => self.save_prefab_stage(),
            }
        }
        // A prefab opened over an instance sits at that instance's transform,
        // read while the instance is still in the scene.
        let opened = next.len() > self.prefab_editor.stage.levels.len();
        if opened && let Some(level) = next.last_mut() {
            level.place = match (level.mode, level.instance.and_then(|id| self.scene.objects.iter().position(|o| o.id == id))) {
                (StageMode::InContext, Some(i)) => Some(self.scene.world_trs(i)),
                _ => None,
            };
        }
        if active {
            let members = self.scene.close_stage();
            let saves = self.undo_stack.take_stage_saves(&members);
            self.prefab_editor.stage.carried.extend(saves);
        } else {
            self.prefab_editor.stage.parked_undo = Some(self.undo_stack.begin_stage());
            self.prefab_editor.stage.parked_dirty = self.scene_dirty;
            self.prefab_editor.stage.main_scene = self.current_scene_path.clone();
        }
        self.prefab_editor.stage.levels = next.clone();
        self.prefab_editor.stage.dirty_since = None;
        self.prefab_editor.invalidate();
        self.selection = Selection::None;
        self.multi_objects.clear();
        self.scene_dirty = false;
        if let Some(level) = next.last() {
            let instance = level.instance.and_then(|id| self.scene.objects.iter().position(|o| o.id == id));
            match self.scene.open_stage(self.renderer.as_mut(), &mut self.shaders, &self.project_root, &self.components, &level.asset, level.mode, instance, level.place) {
                Ok(_) => {}
                Err(e) => {
                    tracing::error!("opening {}: {e:#}", level.asset);
                    self.tasks.notify(format!("Could not open {}: {e}", level.asset), crate::tasks::NotifyLevel::Warn);
                    self.prefab_editor.stage.levels.clear();
                }
            }
        }
        if !self.prefab_editor.stage.active() {
            self.leave_prefab_mode();
        }
    }

    /// Prefab Mode is over: the scene's history comes back with the stage's
    /// saves on top, and the scene's own unsaved state with it.
    fn leave_prefab_mode(&mut self) {
        let stage = &mut self.prefab_editor.stage;
        stage.main_scene = None;
        if let Some(parked) = stage.parked_undo.take() {
            let saves = std::mem::take(&mut stage.carried);
            self.undo_stack.end_stage(parked, saves);
        }
        self.scene_dirty = std::mem::take(&mut self.prefab_editor.stage.parked_dirty);
        self.scene.refresh_stage_hidden();
    }

    pub(super) fn run_prefab_action(&mut self, action: PrefabAction) {
        match action {
            PrefabAction::Stage(go) => self.stage_go(go, None),
            PrefabAction::StageConfirmed(go, save) => self.stage_go(go, Some(save)),
            PrefabAction::Menu(index, cmd) => self.prefab_menu(index, cmd),
            PrefabAction::Requests(reqs) => {
                let mut flat = Vec::new();
                flatten(reqs, &mut flat);
                self.prefab_requests(flat);
            }
        }
    }

    fn prefab_menu(&mut self, index: usize, cmd: PrefabMenu) {
        if index >= self.scene.objects.len() {
            return;
        }
        let root = self.scene.prefab_root_of(index, false);
        match cmd {
            PrefabMenu::SelectRoot => {
                if let Some(r) = self.scene.prefab_root_of(index, true) {
                    self.select_object(Some(r));
                }
            }
            PrefabMenu::Open | PrefabMenu::OpenIsolated | PrefabMenu::SelectAsset => {
                let Some(path) = root.and_then(|r| self.scene.prefab_link(r)).map(|l| l.path.clone()) else { return };
                if cmd == PrefabMenu::Open || cmd == PrefabMenu::OpenIsolated {
                    let mode = if cmd == PrefabMenu::Open { StageMode::InContext } else { StageMode::Isolated };
                    let instance = root.and_then(|r| self.scene.objects.get(r)).map(|o| o.id);
                    self.actions.push(EditorAction::Prefab(PrefabAction::Stage(StageGo::Open(path, mode, instance))));
                } else {
                    let abs = zestengine_core::asset_ref::resolve_path(&self.project_root, &path);
                    self.actions.push(EditorAction::SelectFile(abs));
                }
            }
            PrefabMenu::Unpack | PrefabMenu::UnpackCompletely => {
                let Some(root) = root else { return };
                let before = self.scene.snapshot_objects();
                if self.scene.unpack_prefab(root, cmd == PrefabMenu::UnpackCompletely) {
                    self.record_structure(before);
                    self.scene_dirty = true;
                    self.prefab_editor.invalidate();
                }
            }
            PrefabMenu::RevertAll | PrefabMenu::ApplyAll => {
                let Some(root) = root else { return };
                self.select_object(Some(root));
                self.prefab_editor.root = Some(self.scene.objects[root].id);
                self.prefab_editor.object_levels = vec![ApplyLevel::outermost()];
                let req = if cmd == PrefabMenu::RevertAll { PrefabRequest::Revert { target: PrefabTarget::All } } else { PrefabRequest::Apply { target: PrefabTarget::All, level: 0 } };
                self.prefab_requests(vec![req]);
            }
        }
    }

    /// Run Revert / Apply requests on the selected instance as one undo step.
    fn prefab_requests(&mut self, reqs: Vec<PrefabRequest>) {
        let Some(root_id) = self.prefab_editor.root else { return };
        let Some(mut root) = self.scene.objects.iter().position(|o| o.id == root_id) else { return };
        let before = self.scene.snapshot_objects();
        let mut files: Vec<scene::PrefabFileChange> = Vec::new();
        let mut changed = false;
        let mut track: Vec<usize> = match &self.selection {
            Selection::Object(i) => vec![*i],
            _ => Vec::new(),
        };
        for req in reqs {
            let step = match req {
                PrefabRequest::Revert { target } => {
                    self.scene
                        .revert_prefab_instance(self.renderer.as_mut(), &mut self.shaders, &self.project_root, &self.components, root, &to_scope(&target))
                        .map(|remap| (None, remap))
                }
                PrefabRequest::Apply { target, level } => {
                    let level = self.prefab_editor.object_levels.get(level).cloned().unwrap_or_else(ApplyLevel::outermost);
                    self.scene
                        .apply_prefab_instance(self.renderer.as_mut(), &mut self.shaders, &self.project_root, &self.components, root, &to_scope(&target), &level)
                        .map(|(file, remap)| (Some(file), remap))
                }
                PrefabRequest::RemoveUnused => {
                    changed |= self.scene.remove_unused_prefab_overrides(root) > 0;
                    continue;
                }
                PrefabRequest::OpenAsset => {
                    self.prefab_menu(root, PrefabMenu::Open);
                    continue;
                }
                PrefabRequest::SelectAsset => {
                    self.prefab_menu(root, PrefabMenu::SelectAsset);
                    continue;
                }
                PrefabRequest::Unpack => {
                    self.prefab_menu(root, PrefabMenu::Unpack);
                    continue;
                }
                PrefabRequest::UnpackCompletely => {
                    self.prefab_menu(root, PrefabMenu::UnpackCompletely);
                    continue;
                }
                PrefabRequest::Batch(_) => continue,
            };
            match step {
                Ok((file, remap)) => {
                    changed = true;
                    files.extend(file);
                    let map = |i: usize| remap.get(i).copied().flatten();
                    match map(root) {
                        Some(r) => root = r,
                        None => break,
                    }
                    track = track.iter().filter_map(|&i| map(i)).collect();
                }
                Err(e) => {
                    tracing::error!("prefab: {e:#}");
                    self.tasks.notify(format!("Prefab operation failed: {e}"), crate::tasks::NotifyLevel::Warn);
                    break;
                }
            }
        }
        if !changed {
            return;
        }
        let after = self.scene.snapshot_objects();
        let entry = if files.is_empty() {
            UndoEntry::Structure { before: Box::new(before), after: Box::new(after) }
        } else {
            UndoEntry::Prefab { files, before: Box::new(before), after: Box::new(after) }
        };
        self.undo_stack.record(entry, false);
        self.suppress_undo_record = true;
        self.scene_dirty = true;
        self.rt_gi.invalidate();
        self.prefab_editor.invalidate();
        self.multi_objects.clear();
        self.selection = match track.first() {
            Some(&i) => Selection::Object(i),
            None => Selection::None,
        };
    }

    /// The `.prefab` at `rel` changed on disk behind the editor's back (another
    /// program, version control): re-merge the open scene's instances of it, and
    /// of anything nesting it, without reloading the scene. Overrides, object ids
    /// and selection stay. True when instances were rebuilt.
    pub(super) fn remerge_prefab(&mut self, rel: &str) -> bool {
        // The prefab being edited in Prefab Mode is the scene itself.
        if self.prefab_editor.stage.levels.last().is_some_and(|l| l.asset == rel) {
            return false;
        }
        if !self.scene.prefabs.differs_from_disk(&self.project_root, rel) {
            return false;
        }
        match self.scene.prefab_asset_changed(self.renderer.as_mut(), &mut self.shaders, &self.project_root, &self.components, rel) {
            Ok(remap) => {
                let map = |i: usize| remap.get(i).copied().flatten();
                self.selection = match &self.selection {
                    Selection::Object(i) => map(*i).map_or(Selection::None, Selection::Object),
                    other => other.clone(),
                };
                let moved: std::collections::BTreeSet<usize> = self.multi_objects.iter().filter_map(|&i| map(i)).collect();
                self.multi_objects = moved.into_iter().collect();
                // Older steps hold objects built from the previous asset.
                self.undo_stack.clear();
                self.rt_gi.invalidate();
                self.prefab_editor.invalidate();
                true
            }
            Err(e) => {
                tracing::error!("re-merging {rel}: {e:#}");
                self.tasks.notify(format!("Updating instances of {rel} failed: {e}"), crate::tasks::NotifyLevel::Warn);
                false
            }
        }
    }

    /// "Create Prefab from Selection" saved `path`: make the selection an
    /// instance of it, so edits to the prefab reach it.
    pub(super) fn connect_prefab(&mut self, root: usize, path: &std::path::Path) {
        let rel = zestengine_engine::prefab::link_path(&self.project_root, path);
        let before = self.scene.snapshot_objects();
        if self.scene.connect_prefab(root, &rel, &self.components) {
            self.scene.warm_prefab_cache(&self.project_root);
            self.record_structure(before);
            self.scene_dirty = true;
            self.prefab_editor.invalidate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(path: &str, mode: StageMode, inst: Option<ObjectId>) -> StageGo {
        StageGo::Open(path.to_owned(), mode, inst)
    }

    fn assets(levels: &[Level]) -> Vec<&str> {
        levels.iter().map(|l| l.asset.as_str()).collect()
    }

    #[test]
    fn prefab_mode_opens_nests_and_unwinds() {
        let inst = ObjectId::new();
        let mut levels = Vec::new();
        levels = next_levels(&levels, &open("ship.prefab", StageMode::InContext, Some(inst)));
        assert_eq!(assets(&levels), ["ship.prefab"]);
        assert_eq!(levels[0].mode, StageMode::InContext);
        assert_eq!(levels[0].instance, Some(inst));
        // Opening a nested prefab goes one level deeper; opening the open one does nothing.
        let nested = ObjectId::new();
        levels = next_levels(&levels, &open("gun.prefab", StageMode::InContext, Some(nested)));
        assert_eq!(assets(&levels), ["ship.prefab", "gun.prefab"]);
        assert_eq!(next_levels(&levels, &open("gun.prefab", StageMode::Isolated, None)), levels);
        // Back goes up one; a crumb jumps; crumb 0 is the scene.
        assert_eq!(assets(&next_levels(&levels, &StageGo::Back)), ["ship.prefab"]);
        assert_eq!(assets(&next_levels(&levels, &StageGo::Crumb(1))), ["ship.prefab"]);
        assert!(next_levels(&levels, &StageGo::Crumb(0)).is_empty());
        assert!(next_levels(&next_levels(&levels, &StageGo::Back), &StageGo::Back).is_empty());
        assert!(next_levels(&[], &StageGo::Back).is_empty(), "Back with nothing open stays out");
    }

    #[test]
    fn in_context_without_an_instance_falls_back_to_isolated() {
        let levels = next_levels(&[], &open("a.prefab", StageMode::InContext, None));
        assert_eq!(levels[0].mode, StageMode::Isolated);
        assert_eq!(levels[0].instance, None);
        let inst = ObjectId::new();
        let both = next_levels(&levels, &open("b.prefab", StageMode::Isolated, Some(inst)));
        assert_eq!(both[1].mode, StageMode::Isolated, "a request for isolated stays isolated");
        assert_eq!(both[1].instance, Some(inst));
    }
}
