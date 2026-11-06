//! Asset placement driver: connects the pure state machine in `zestengine_engine::placement` to the
//! editor. A child module of `editor_app` (declared there with `#[path]`) so it can use
//! the app's private fields. See `docs/design/asset-placement.md`.
//!
//! Hooks in `editor_app.rs`: `placement_capture_input` and `placement_frame` run in the
//! egui pass, `placement_hierarchy_drop` serves the Scene panel's drops,
//! `placement_before_import` / `placement_after_import` bracket `apply_imported_model`.

use super::*;
use zestengine_engine::placement::{self, Arrival, AssetKind, Cmd, FrameInput, HierarchyDrop, Placement, Purpose, Ray, SnapCache, SurfaceHit, SurfaceKind};
use crate::placement_ui;
use zestengine_core::ObjectId;

/// Placement state held by the app.
#[derive(Default)]
pub(super) struct Run {
    session: Placement,
    cache: SnapCache,
    /// Scene before the spawn, for the single undo step.
    before: Option<scene::ObjectsSnapshot>,
    object: Option<ObjectId>,
    prev_selection: Option<Selection>,
    path: Option<PathBuf>,
    /// Bounds of the live object in its own space (None = no mesh).
    bounds: Option<(Vec3, Vec3)>,
    /// Newest ray result; the loading placeholder and a pending commit use it.
    last_hit: Option<SurfaceHit>,
    last_pointer: Option<egui::Pos2>,
    /// Wheel notches / R presses collected before the dock consumed the scroll.
    pending_ticks: i32,
    pending_fine: bool,
    /// What the user sees: Alt-adjusted alignment, ring radius.
    align_now: bool,
}

impl Run {
    /// A placement is running (or its Esc latch is on): R rotates the item, not the tool.
    pub(super) fn session_captures_input(&self) -> bool {
        self.session.captures_input()
    }
}

/// What `apply_imported_model` must do with the model that just arrived.
pub(super) enum ImportHook {
    /// Not ours: the normal import.
    Normal,
    /// The session that asked for it ended: drop the data without touching the scene.
    Discard,
    Tracked { arrival: Arrival, first_new: usize, before: scene::ObjectsSnapshot },
}

fn project_relative(root: &Path, path: &Path) -> PathBuf {
    let rel = root.join(zestengine_engine::scene::relative_to(path, root));
    PathBuf::from(zestengine_engine::scene::relative_to(&rel, root))
}

impl EngineApp {
    /// Wheel / R presses over the viewport while a spawnable file is dragged belong to the
    /// placement (they rotate the item instead of dollying the camera). Runs first in the
    /// egui pass, before the viewport tab reads the scroll.
    pub(super) fn placement_capture_input(&mut self, ctx: &egui::Context) {
        let dragging = egui::DragAndDrop::payload::<PathBuf>(ctx).is_some_and(|p| placement::asset_kind(&p).is_some());
        if !dragging && !self.placement.session.is_active() {
            return;
        }
        let Some(pos) = ctx.input(|i| i.pointer.latest_pos()) else { return };
        if !self.viewport_rect.contains(pos) {
            return;
        }
        let (scroll, shift) = ctx.input(|i| (i.raw_scroll_delta.y, i.modifiers.shift));
        if scroll != 0.0 {
            self.placement.pending_ticks += scroll.signum() as i32;
            self.placement.pending_fine |= shift;
            ctx.input_mut(|i| {
                i.raw_scroll_delta = egui::Vec2::ZERO;
                i.smooth_scroll_delta = egui::Vec2::ZERO;
            });
        }
    }

    /// One placement step per frame: read the drag, advance the state machine, run its
    /// commands, follow the cursor, paint the overlay.
    pub(super) fn placement_frame(&mut self, ctx: &egui::Context) {
        let drag = egui::DragAndDrop::payload::<PathBuf>(ctx).map(|p| (*p).clone());
        if self.playing {
            if self.placement.session.is_active() {
                self.placement_run(Cmd::Cancel);
                self.placement.session = Placement::default();
            }
            return;
        }
        let (pos, released, esc, r_key, shift, alt, ctrl) = ctx.input(|i| {
            (
                i.pointer.latest_pos(),
                i.pointer.any_released(),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::R),
                i.modifiers.shift,
                i.modifiers.alt,
                i.modifiers.command || i.modifiers.ctrl,
            )
        });
        let over_overlay = |p: egui::Pos2| {
            let hosts_view: Option<egui::LayerId> = ctx.data(|d| d.get_temp(egui::Id::new(VIEW_LAYER_KEY)));
            ctx.layer_id_at(p).is_some_and(|top| top.order != egui::Order::Background && Some(top) != hosts_view)
        };
        let inside = self.viewport_visible && pos.is_some_and(|p| self.viewport_rect.contains(p) && !over_overlay(p));
        let input = FrameInput { drag: drag.clone(), inside, released, esc };
        if esc && self.placement.session.is_active() {
            egui::DragAndDrop::clear_payload(ctx);
        }
        let cmds = self.placement.session.frame(&input);
        let started = cmds.iter().any(|c| matches!(c, Cmd::Spawn { .. }));
        for c in cmds {
            self.placement_run(c);
        }
        if self.placement.session.is_active() || started {
            // Placement moves the object every frame: not an undoable edit by itself.
            self.suppress_undo_record = true;
        }
        if !self.placement.session.is_active() {
            self.placement.pending_ticks = 0;
            self.placement.pending_fine = false;
            return;
        }
        // Yaw: wheel notches collected before the dock, plus R.
        let mut ticks = std::mem::take(&mut self.placement.pending_ticks);
        let fine = std::mem::take(&mut self.placement.pending_fine) || shift;
        if r_key {
            ticks += 1;
        }
        if ticks != 0 {
            self.placement.session.yaw_deg = placement::step_yaw(self.placement.session.yaw_deg, ticks, fine);
        }
        self.placement.align_now = self.gizmo.align_to_surface != alt;
        if let Some(p) = pos.filter(|_| inside) {
            self.placement.last_pointer = Some(p);
            if self.placement.session.is_following() || self.placement.session.is_loading() {
                self.placement_follow(p, ctrl);
            }
        }
        self.placement_overlay(ctx);
        // A big mesh BVH may still be building: keep repainting so the snap upgrades itself.
        ctx.request_repaint();
    }

    /// Cast the cursor ray, snap, and move the live object (or remember the point while a
    /// model still loads).
    fn placement_follow(&mut self, pointer: egui::Pos2, grid: bool) {
        let Some((origin, dir)) = self.cursor_ray(pointer) else { return };
        let ray = Ray::new(origin, dir);
        let index = self.placement.object.and_then(|id| self.scene.index_of(id));
        let targets = placement::collect_targets(&self.scene, index);
        let mut hit = placement::resolve_surface(&ray, &targets, &self.scene, &mut self.placement.cache, true);
        if grid && self.gizmo.grid_size > 0.0 {
            hit.point = placement::snap_point_to_grid(hit.point, hit.normal, self.gizmo.grid_size);
        }
        self.placement.last_hit = Some(hit);
        if let Some(index) = index {
            self.placement_pose(index, hit);
        }
    }

    fn placement_pose(&mut self, index: usize, hit: SurfaceHit) {
        let rot = placement::align_rotation(hit.normal, self.placement.session.yaw_deg, self.placement.align_now);
        let scale = self.scene.objects[index].scale;
        let pos = placement::placed_position(hit.point, hit.normal, rot, scale, self.placement.bounds);
        placement::set_root_pose(&mut self.scene, index, pos, rot);
    }

    fn placement_run(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Spawn { ticket, path, kind } => {
                self.placement.before = Some(self.scene.snapshot_objects());
                self.placement.prev_selection = Some(self.selection.clone());
                self.placement.path = Some(path.clone());
                self.placement.object = None;
                self.placement.bounds = None;
                self.placement.last_hit = None;
                let at = Vec3::ZERO;
                match kind {
                    AssetKind::Model => {
                        let source = project_relative(&self.project_root, &path);
                        self.placement.session.expect_model(source, Purpose::Session(ticket));
                        self.actions.push(EditorAction::ImportModel(path));
                        // `placement_after_import` reports back when it lands.
                    }
                    AssetKind::Prefab | AssetKind::Audio => {
                        let made = if kind == AssetKind::Prefab { self.placement_spawn_prefab(&path, at) } else { self.placement_spawn_audio(&path, at) };
                        match made {
                            Ok(index) => {
                                self.placement_adopt(index);
                                let cmds = self.placement.session.spawned(ticket, true);
                                for c in cmds {
                                    self.placement_run(c);
                                }
                            }
                            Err(e) => {
                                self.set_status(format!("Could not place {}: {e:#}", path.display()), false);
                                self.placement.session.spawned(ticket, false);
                                self.placement.before = None;
                            }
                        }
                    }
                }
            }
            Cmd::Reveal => {
                if let Some(i) = self.placement.object.and_then(|id| self.scene.index_of(id)) {
                    self.scene.objects[i].enabled = true;
                    self.select_object(Some(i));
                }
            }
            Cmd::Park => {
                if let Some(i) = self.placement.object.and_then(|id| self.scene.index_of(id)) {
                    self.scene.objects[i].enabled = false;
                }
                let prev = self.placement.prev_selection.clone().unwrap_or(Selection::None);
                self.restore_selection(prev);
            }
            Cmd::Commit => {
                let index = self.placement.object.and_then(|id| self.scene.index_of(id));
                let (Some(index), Some(before)) = (index, self.placement.before.take()) else {
                    self.placement_clear();
                    return;
                };
                // Last pose from the final ray, so a model that arrived after the release lands
                // where the pointer left it.
                if let Some(hit) = self.placement.last_hit {
                    self.placement_pose(index, hit);
                }
                self.scene.objects[index].enabled = true;
                self.record_structure(before);
                self.select_object(Some(index));
                self.gizmo.tool = GizmoTool::Move;
                self.scene_dirty = true;
                self.prefab_editor.invalidate();
                self.placement_clear();
            }
            Cmd::Cancel => {
                if let Some(i) = self.placement.object.and_then(|id| self.scene.index_of(id)) {
                    self.scene.remove_object(i);
                    self.prefab_editor.invalidate();
                }
                let prev = self.placement.prev_selection.clone().unwrap_or(Selection::None);
                self.restore_selection(prev);
                self.placement_clear();
                self.suppress_undo_record = true;
            }
        }
    }

    fn placement_clear(&mut self) {
        self.placement.before = None;
        self.placement.object = None;
        self.placement.path = None;
        self.placement.bounds = None;
        self.placement.prev_selection = None;
        self.placement.last_hit = None;
    }

    /// Selection restored after a cancel / park; stale object indices fall back to none.
    fn restore_selection(&mut self, prev: Selection) {
        match prev {
            Selection::Object(i) if i >= self.scene.objects.len() => self.select_object(None),
            Selection::Object(i) => self.select_object(Some(i)),
            other => {
                self.selection = other;
                self.multi_objects.clear();
            }
        }
    }

    /// Take over a freshly spawned root: remember it, measure it, select it.
    fn placement_adopt(&mut self, index: usize) {
        self.placement.object = Some(self.scene.objects[index].id);
        self.placement.bounds = placement::subtree_bounds(&self.scene, index);
        self.select_object(Some(index));
        if let Some(hit) = self.placement.last_hit {
            self.placement_pose(index, hit);
        } else if let Some(p) = self.placement.last_pointer {
            self.placement_follow(p, false);
        }
    }

    fn placement_spawn_prefab(&mut self, path: &Path, at: Vec3) -> Result<usize> {
        let prefab = zestengine_engine::prefab::Prefab::load(path)?;
        let name = prefab.roots().first().map_or_else(|| "Prefab".to_owned(), |&r| prefab.entries[r].name.clone());
        let mut entries = vec![zestengine_engine::prefab::link_root(&zestengine_engine::prefab::link_path(&self.project_root, path), &name, at.to_array())];
        let cache = &self.scene.prefabs;
        zestengine_assets::expand_prefabs(&mut entries, &|p| cache.get(&self.project_root, p));
        if entries.len() == 1 && entries[0].prefab.as_ref().is_some_and(|l| !l.expanded || cache.get(&self.project_root, &l.path).is_none()) {
            anyhow::bail!("{} could not be read", path.display());
        }
        let Some(renderer) = self.renderer.as_mut() else { anyhow::bail!("no renderer") };
        self.scene.append_entries(renderer, &mut self.shaders, &self.project_root, &self.components, &entries)
    }

    fn placement_spawn_audio(&mut self, path: &Path, at: Vec3) -> Result<usize> {
        let name = path.file_stem().map_or_else(|| "Audio Source".to_owned(), |s| s.to_string_lossy().into_owned());
        let Some(renderer) = self.renderer.as_mut() else { anyhow::bail!("no renderer") };
        let index = self.scene.spawn(renderer, zestengine_assets::ObjectSource::Empty, name, at)?;
        let clip = zestengine_engine::scene::relative_to(path, &self.project_root).replace('\\', "/");
        self.scene.objects[index].components.push(Box::new(zestengine_core::AudioSource { clip, ..Default::default() }));
        Ok(index)
    }

    /// The Scene panel got a file dropped: spawn at world (0,0,0), or assign a material to
    /// the row. One undo step, selected.
    pub(super) fn placement_hierarchy_drop(&mut self, path: PathBuf, row: Option<usize>) {
        let row = row.filter(|&r| r < self.scene.objects.len());
        match placement::classify_hierarchy_drop(&path, row.is_some()) {
            HierarchyDrop::AssignMaterial => {
                if let Some(r) = row {
                    self.actions.push(EditorAction::AssignMaterialToObject(r, 0, path));
                }
            }
            HierarchyDrop::Ignore => {}
            HierarchyDrop::Spawn(kind) => {
                let parent = row.map(|r| self.scene.objects[r].id);
                if kind == AssetKind::Model {
                    let source = project_relative(&self.project_root, &path);
                    self.placement.session.expect_model(source, Purpose::Origin { parent });
                    self.actions.push(EditorAction::ImportModel(path));
                    return;
                }
                let before = self.scene.snapshot_objects();
                let made = if kind == AssetKind::Prefab { self.placement_spawn_prefab(&path, Vec3::ZERO) } else { self.placement_spawn_audio(&path, Vec3::ZERO) };
                match made {
                    Ok(index) => self.placement_finish_at_origin(index, parent, before),
                    Err(e) => self.set_status(format!("Could not place {}: {e:#}", path.display()), false),
                }
            }
        }
    }

    /// Put `index` at world origin under `parent`, select it and record one undo step.
    fn placement_finish_at_origin(&mut self, index: usize, parent: Option<ObjectId>, before: scene::ObjectsSnapshot) {
        let parent_index = parent.and_then(|id| self.scene.index_of(id));
        if !placement::place_at_world_origin(&mut self.scene, index, parent_index) {
            placement::place_at_world_origin(&mut self.scene, index, None);
        }
        self.record_structure(before);
        self.select_object(Some(index));
        self.scene_dirty = true;
        self.prefab_editor.invalidate();
    }

    /// Before `apply_imported_model` uploads a model: is it one placement asked for?
    pub(super) fn placement_before_import(&mut self, source: &Path) -> ImportHook {
        match self.placement.session.model_arrived(source) {
            Arrival::Unrelated => ImportHook::Normal,
            Arrival::Discard => ImportHook::Discard,
            arrival => ImportHook::Tracked { arrival, first_new: self.scene.objects.len(), before: self.scene.snapshot_objects() },
        }
    }

    /// After the model went into the scene (`ok`), finish the placement it was for.
    pub(super) fn placement_after_import(&mut self, hook: ImportHook, ok: bool) {
        let ImportHook::Tracked { arrival, first_new, before } = hook else { return };
        let root = first_new;
        match arrival {
            Arrival::Session(ticket) => {
                if !ok || root >= self.scene.objects.len() {
                    for c in self.placement.session.spawned(ticket, false) {
                        self.placement_run(c);
                    }
                    self.placement.before = None;
                    return;
                }
                // The session's own "before" (taken at the spawn) already precedes the model.
                drop(before);
                self.placement_adopt(root);
                for c in self.placement.session.spawned(ticket, true) {
                    self.placement_run(c);
                }
            }
            Arrival::Origin { parent } => {
                if ok && root < self.scene.objects.len() {
                    self.placement_finish_at_origin(root, parent, before);
                }
            }
            Arrival::Discard | Arrival::Unrelated => {}
        }
    }

    fn placement_world_to_screen(&self, world: Vec3) -> Option<egui::Pos2> {
        let rect = self.viewport_rect;
        let aspect = rect.width() / rect.height().max(1.0);
        let clip = self.camera.proj(aspect) * self.camera.view() * world.extend(1.0);
        if clip.w <= 0.001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(egui::pos2(rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(), rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height()))
    }

    fn placement_overlay(&mut self, ctx: &egui::Context) {
        if !self.placement.session.hint_visible() {
            return;
        }
        let mut o = placement_ui::Overlay::new(self.viewport_rect, self.placement.align_now);
        if let Some(hit) = self.placement.last_hit {
            let radius = self
                .placement
                .bounds
                .map_or(0.4, |(mn, mx)| ((mx - mn).x.max((mx - mn).z) * 0.5).clamp(0.25, 3.0));
            o.ring = placement_ui::ring_points(hit.point + hit.normal * 0.01, hit.normal, radius, 40).into_iter().filter_map(|p| self.placement_world_to_screen(p)).collect();
            if o.ring.len() != 40 {
                o.ring.clear();
            }
            if hit.kind != SurfaceKind::Air
                && let (Some(a), Some(b)) = (self.placement_world_to_screen(hit.point), self.placement_world_to_screen(hit.point + hit.normal * 0.6))
            {
                o.tick = Some((a, b));
            }
        }
        if self.placement.session.is_loading()
            && let Some(p) = self.placement.last_pointer
        {
            let name = self.placement.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            o.loading = Some((p, format!("Loading {name}…")));
        }
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("zestengine-placement-overlay")));
        placement_ui::paint(&painter, &o);
    }
}
