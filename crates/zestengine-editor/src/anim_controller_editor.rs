//! The Animator Controller tab: a node-graph editor for `.animctrl` state
//! machines (states, Any State, Entry, transitions), with Layers and
//! Parameters on the left, an inspector for the selection on the right, and a
//! live view of the selected object's Animator while it runs.

use std::path::{Path, PathBuf};

use zestengine_assets::{
    AnimCondition, AnimLayer, AnimParamKind, AnimParameter, AnimState, AnimTransition, AnimatorController, CondOp,
    InterruptionSource, LayerBlend, Motion, StateMachine,
};
use zestengine_core::{Animator, BlendMotion, BlendTree, BlendTreeKind};
use crate::egui_phosphor::regular as ph;
use crate::icons::Icon;
use crate::{theme, widgets};
use egui::{Color32, CornerRadius, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2};

use crate::anim_canvas::{self as cv, ArrowTint, TimelineDrag, TimelineGeom};
use crate::anim_clip_editor::AnimClipTab;
use zestengine_engine::clip_motion::{self as cm, ClipMotion};
use zestengine_engine::scene::LoadedScene;

const NODE_SIZE: Vec2 = Vec2::new(160.0, 40.0);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Node {
    State(usize),
    Any,
    Entry,
}

/// A timeline drag in progress. Visual only: the transition is written once,
/// when the mouse is released (one undo step).
#[derive(Clone, Copy)]
struct TlDrag {
    transition: usize,
    kind: TimelineDrag,
    start: TimelineGeom,
    cur: TimelineGeom,
    /// The axis frozen at mouse-down so the view does not rescale mid-drag.
    axis: cv::TimeAxis,
    origin_x: f32,
}

/// What the timeline needs to know about the selected transition's states.
struct TlInfo {
    src: String,
    dst: String,
    src_dur: f32,
    dst_dur: f32,
    src_loop: bool,
    dst_loop: bool,
    /// A length could not be read (imported clip, blend tree of unknown
    /// clips): 1 s stands in.
    estimated: bool,
}

/// A node drag: every dragged node's start position and the drag so far.
struct NodeDrag {
    starts: Vec<(Node, [f32; 2])>,
    total: Vec2,
}

/// A box selection in progress.
struct BoxSel {
    from: Pos2,
    prev: Vec<Node>,
    additive: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sel {
    None,
    Node(Node),
    Transition(usize),
}

pub struct AnimControllerTab {
    path: Option<String>,
    ctrl: AnimatorController,
    /// Undo history of the open document.
    history: crate::undo::DocHistory<AnimatorController>,
    dirty: bool,
    layer: usize,
    /// Sub-state machine being edited: state indices from the layer's machine
    /// down (empty = the layer itself).
    sub: Vec<usize>,
    sel: Sel,
    pan: Vec2,
    zoom: f32,
    /// Side panel widths. Only the drag handles on the graph's edges change
    /// them: a resizable egui panel adopts whatever width its contents report.
    left_width: f32,
    right_width: f32,
    dragging: Option<Node>,
    /// Drawing a new transition from this node (drag with Shift, or the
    /// context menu's Make Transition followed by a click).
    linking: Option<Node>,
    context: Option<(Option<Node>, Option<usize>, Pos2)>,
    /// Nodes selected together (valid while `multi_for` is the selection).
    multi: Vec<Node>,
    multi_for: Sel,
    node_drag: Option<NodeDrag>,
    box_sel: Option<BoxSel>,
    /// Esc cancelled the gesture in progress: ignore the rest of the drag.
    drag_cancelled: bool,
    /// Snap dragged nodes to the 10 unit grid.
    snap: bool,
    /// Follow the live Animator into sub-machines and keep its state in view.
    auto_follow: bool,
    last_follow: Option<Vec<String>>,
    last_live_state: Option<String>,
    /// Canvas size from the last frame, for framing.
    view_size: Vec2,
    tl_drag: Option<TlDrag>,
    /// Horizontal pan of the transition timeline, in pixels.
    tl_shift: f32,
    /// Asked the editor to show this tab.
    pub open_request: bool,
    /// Parameter values scrubbed in the blend tree editor while no Animator
    /// is running (preview only, never saved).
    bt_preview: std::collections::HashMap<String, f32>,
    /// The object whose running Animator the blend tree scrubbers drive.
    playing_obj: Option<usize>,
    last_file: Option<PathBuf>,
    files: Vec<String>,
    clip_files: Vec<String>,
    /// Pan and zoom last used in each machine (layer, sub-machine path), restored on return.
    views: std::collections::HashMap<(usize, Vec<usize>), (Vec2, f32)>,
    view_key: (usize, Vec<usize>),
}

impl Default for AnimControllerTab {
    fn default() -> Self {
        Self {
            path: None,
            ctrl: AnimatorController::default(),
            dirty: false,
            history: Default::default(),
            layer: 0,
            sub: Vec::new(),
            sel: Sel::None,
            pan: Vec2::ZERO,
            zoom: 1.0,
            left_width: 236.0,
            right_width: 290.0,
            dragging: None,
            linking: None,
            context: None,
            multi: Vec::new(),
            multi_for: Sel::None,
            node_drag: None,
            box_sel: None,
            drag_cancelled: false,
            snap: true,
            auto_follow: true,
            last_follow: None,
            last_live_state: None,
            view_size: Vec2::ZERO,
            tl_drag: None,
            tl_shift: 0.0,
            open_request: false,
            bt_preview: std::collections::HashMap::new(),
            playing_obj: None,
            last_file: None,
            files: Vec::new(),
            clip_files: Vec::new(),
            views: Default::default(),
            view_key: (0, Vec::new()),
        }
    }
}

// ------------------------------------------------------------ shared helpers

/// Project-relative paths (forward slashes) of files with extension `ext`.
/// A project-relative path for a new `.mask` named after `layer`, not yet taken.
fn unique_mask_path(root: &Path, layer: &str) -> String {
    let stem: String = layer.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let stem = if stem.trim_matches('_').is_empty() { "mask".to_owned() } else { stem };
    (0..)
        .map(|n| if n == 0 { format!("masks/{stem}.mask") } else { format!("masks/{stem}_{n}.mask") })
        .find(|rel| !root.join(rel).exists())
        .unwrap_or_default()
}

pub(crate) fn scan_files(root: &Path, ext: &str) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, ext: &str, out: &mut Vec<String>, depth: usize) {
        if depth > 12 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "target" || name == "build" {
                continue;
            }
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, ext, out, depth + 1);
            } else if p.extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case(ext))
                && let Ok(rel) = p.strip_prefix(root)
            {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, ext, &mut out, 0);
    out.sort();
    out
}

/// A free `<dir>/<stem>.<ext>` (numbered when taken).
pub(crate) fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let first = dir.join(format!("{stem}.{ext}"));
    if !first.exists() {
        return first;
    }
    (1..).map(|n| dir.join(format!("{stem} {n}.{ext}"))).find(|p| !p.exists()).unwrap()
}

pub(crate) fn rel_path(root: &Path, abs: &Path) -> String {
    abs.strip_prefix(root).unwrap_or(abs).to_string_lossy().replace('\\', "/")
}

/// Single-line text field that edits a buffer and reports the new text when
/// the edit is committed (Enter or focus lost) and differs from `current`.
pub(crate) fn commit_text(ui: &mut egui::Ui, id: egui::Id, current: &str, width: f32) -> Option<String> {
    let mut buf: String = ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| current.to_owned());
    let resp = ui.add(egui::TextEdit::singleline(&mut buf).id(id.with("edit")).desired_width(width));
    if resp.has_focus() || resp.changed() {
        ui.data_mut(|d| d.insert_temp(id, buf.clone()));
    }
    if resp.lost_focus() {
        ui.data_mut(|d| d.remove::<String>(id));
        if buf != current && !buf.trim().is_empty() {
            return Some(buf.trim().to_owned());
        }
    } else if !resp.has_focus() {
        ui.data_mut(|d| d.remove::<String>(id));
    }
    None
}

fn file_stem(path: &str) -> String {
    Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_owned())
}

/// Requests the editor should act on this frame: tabs to show (true =
/// controller, false = clip). Reads the Animator inspector's Open button too.
pub fn take_open_requests(
    ctx: &egui::Context,
    ctrl: &mut AnimControllerTab,
    clip: &mut AnimClipTab,
    root: &Path,
) -> Vec<bool> {
    let mut out = Vec::new();
    let key = egui::Id::new(crate::OPEN_ANIM_CONTROLLER_KEY);
    if let Some(path) = ctx.data_mut(|d| d.remove_temp::<String>(key)) {
        ctrl.open(root, &path);
        out.push(true);
    }
    if std::mem::take(&mut ctrl.open_request) {
        out.push(true);
    }
    if std::mem::take(&mut clip.open_request) {
        out.push(false);
    }
    out
}

/// Inspector body for a selected `.animctrl` / `.anim` file: opens it in its
/// tab (once per selection) and offers a button to reopen.
pub fn file_inspector(
    ui: &mut egui::Ui,
    path: &Path,
    root: &Path,
    ctrl: &mut AnimControllerTab,
    clip: &mut AnimClipTab,
) {
    let rel = rel_path(root, path);
    let is_ctrl = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("animctrl"));
    ui.label(egui::RichText::new(file_stem(&rel)).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
    ui.label(egui::RichText::new(&rel).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
    ui.add_space(8.0);
    if ctrl.last_file.as_deref() != Some(path) {
        ctrl.last_file = Some(path.to_owned());
        if is_ctrl {
            ctrl.open(root, &rel);
            ctrl.open_request = true;
        } else {
            clip.open(root, &rel);
            clip.open_request = true;
        }
    }
    let label = if is_ctrl { "Open in Animator Controller" } else { "Open in Animation" };
    if widgets::secondary_button(ui, Some(Icon::FileDoc), label).clicked() {
        if is_ctrl {
            ctrl.open(root, &rel);
            ctrl.open_request = true;
        } else {
            clip.open(root, &rel);
            clip.open_request = true;
        }
    }
    ui.add_space(6.0);
    widgets::info_note(
        ui,
        if is_ctrl {
            "Assign it to an Animator's Controller field (drag the file onto it)."
        } else {
            "Use it as an Animator's clip, or as a state's motion in a controller."
        },
    );
}

/// Create a new controller or clip in `dir` and open it.
pub fn create_asset(dir: &Path, controller: bool, root: &Path, ctrl: &mut AnimControllerTab, clip: &mut AnimClipTab) {
    if controller {
        let path = unique_path(dir, "New Controller", zestengine_assets::ANIM_CONTROLLER_EXTENSION);
        let mut c = AnimatorController::default();
        c.layers[0].machine.states.push(AnimState { name: "Idle".into(), ..Default::default() });
        c.layers[0].machine.default_state = "Idle".into();
        match zestengine_assets::save_anim_controller(&path, &c) {
            Ok(()) => {
                ctrl.open(root, &rel_path(root, &path));
                ctrl.open_request = true;
            }
            Err(e) => tracing::error!("creating animator controller: {e:#}"),
        }
    } else {
        let path = unique_path(dir, "New Clip", zestengine_assets::ANIM_CLIP_EXTENSION);
        match zestengine_assets::save_anim_clip(&path, &zestengine_assets::AnimClip::default()) {
            Ok(()) => {
                clip.open(root, &rel_path(root, &path));
                clip.open_request = true;
            }
            Err(e) => tracing::error!("creating animation clip: {e:#}"),
        }
    }
}

/// The machine shown in the graph: `sub` followed down from the layer's
/// machine (the layer's own when the path is stale).
fn machine_of<'a>(ctrl: &'a AnimatorController, layer: usize, sub: &[usize]) -> &'a StateMachine {
    let root = &ctrl.layers[layer].machine;
    root.nested(sub).unwrap_or(root)
}

fn machine_of_mut<'a>(ctrl: &'a mut AnimatorController, layer: usize, sub: &[usize]) -> &'a mut StateMachine {
    let root = &mut ctrl.layers[layer].machine;
    if root.nested(sub).is_some() { root.nested_mut(sub).unwrap() } else { root }
}

/// Whether the runtime's flattened state name `full` is local state `local`
/// of the level named `prefix`, or inside it when it is a sub-state machine.
fn under(full: &str, prefix: &str, local: &str) -> bool {
    full.strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix(local))
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

fn animator(scene: &LoadedScene, i: usize) -> Option<&Animator> {
    scene.objects.get(i)?.components.iter().find_map(|c| c.as_any().downcast_ref::<Animator>())
}

fn animator_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut Animator> {
    scene.objects.get_mut(i)?.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Animator>())
}

// ------------------------------------------------------------------ the tab

impl AnimControllerTab {
    pub fn open(&mut self, root: &Path, rel: &str) {
        if self.path.as_deref() == Some(rel) {
            return;
        }
        if self.dirty {
            self.save(root);
        }
        match zestengine_assets::load_anim_controller(root.join(rel)) {
            Ok(mut c) => {
                if c.layers.is_empty() {
                    c.layers.push(AnimLayer::default());
                }
                // Layers that reference a `.mask` show the asset's content.
                for p in c.resolve_mask_assets(root) {
                    tracing::warn!("{rel}: {p}");
                }
                self.ctrl = c;
                self.path = Some(rel.to_owned());
                self.dirty = false;
                self.layer = 0;
                self.sub.clear();
                self.sel = Sel::None;
                self.linking = None;
                self.dragging = None;
                self.views.clear();
                self.view_key = (0, Vec::new());
            }
            Err(e) => tracing::error!("opening animator controller {rel}: {e:#}"),
        }
    }

    fn save(&mut self, root: &Path) {
        let Some(path) = &self.path else {
            return;
        };
        match zestengine_assets::save_anim_controller(root.join(path), &self.ctrl) {
            Ok(()) => self.dirty = false,
            Err(e) => tracing::error!("saving animator controller: {e:#}"),
        }
    }

    fn changed(&mut self, scene: &mut LoadedScene) {
        self.dirty = true;
        if let Some(path) = &self.path {
            scene.anim_assets.set_controller(path, self.ctrl.clone());
        }
    }

    /// The object whose Animator runs this controller: the selection, else
    /// (while playing) the first object using it.
    fn live_object(&self, scene: &LoadedScene, selected: Option<usize>, playing: bool) -> Option<usize> {
        let path = self.path.as_deref()?;
        let uses = |i: usize| animator(scene, i).is_some_and(|a| scene.anim_assets.same_asset(&a.controller, path) && !a.state.layers.is_empty());
        if let Some(s) = selected.filter(|&s| uses(s)) {
            return Some(s);
        }
        if playing {
            return (0..scene.objects.len()).find(|&i| uses(i));
        }
        None
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, selected: Option<usize>, root: &Path, playing: bool) {
        let (before, path) = (self.ctrl.clone(), self.path.clone());
        self.ui_inner(ui, scene, selected, root, playing);
        if self.path != path {
            self.history.clear();
        } else {
            self.history.observe(before, &self.ctrl, ui.input(|i| i.pointer.any_down()));
        }
    }

    /// Ctrl+Z / Ctrl+Shift+Z while this tab has focus.
    pub fn history_step(&mut self, undo: bool) {
        if self.history.step(&mut self.ctrl, undo) {
            self.dirty = true;
            // Selections are indices into the document: drop them.
            self.sel = Sel::None;
            self.dragging = None;
            self.linking = None;
            self.layer = self.layer.min(self.ctrl.layers.len().saturating_sub(1));
        }
    }

    fn ui_inner(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, selected: Option<usize>, root: &Path, playing: bool) {
        scene.anim_assets.set_root(root);
        let selected = selected.filter(|&i| i < scene.objects.len());
        let live = self.live_object(scene, selected, playing);
        if self.ctrl.layers.is_empty() {
            self.ctrl.layers.push(AnimLayer::default());
        }
        if self.layer >= self.ctrl.layers.len() {
            self.layer = 0;
            self.sel = Sel::None;
        }
        // Undo or a delete can remove the sub-machine being edited.
        if self.ctrl.layers[self.layer].machine.nested(&self.sub).is_none() {
            self.sub.clear();
            self.sel = Sel::None;
        }

        self.live_follow(scene, live);
        self.playing_obj = live.filter(|_| playing);

        egui::TopBottomPanel::top("zestengine-animctrl-toolbar")
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(12, 6)))
            .show_inside(ui, |ui| self.toolbar(ui, scene, root, live));

        if self.path.is_none() {
            self.empty_state(ui, root, scene, selected);
            return;
        }

        egui::SidePanel::left("zestengine-animctrl-left")
            .resizable(false)
            .exact_width(self.left_width)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::same(12)))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("zestengine-animctrl-left-scroll").show(ui, |ui| {
                    self.layers_ui(ui, scene);
                    widgets::section_gap(ui);
                    self.params_ui(ui, scene, live.filter(|_| playing));
                });
            });
        egui::SidePanel::right("zestengine-animctrl-right")
            .resizable(false)
            .exact_width(self.right_width)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::same(12)))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("zestengine-animctrl-right-scroll").show(ui, |ui| {
                    self.inspector_ui(ui, scene, selected.or(live), root);
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG_VIEWPORT))
            .show_inside(ui, |ui| {
                let area = ui.max_rect();
                egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(10, 4)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    self.breadcrumb(ui);
                });
                self.canvas(ui, scene, live, root);
                self.panel_handles(ui, area);
            });

        // Save once an edit settles (no button held), like code tabs.
        if self.dirty && !ui.input(|i| i.pointer.any_down()) && !ui.ctx().wants_keyboard_input() {
            self.save(root);
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, root: &Path, live: Option<usize>) {
        ui.horizontal(|ui| {
            ui.set_min_height(theme::H_TOOLBAR);
            ui.label(egui::RichText::new(ph::FLOW_ARROW).size(15.0).color(theme::TEXT_MUTED));
            match &self.path {
                Some(p) => {
                    ui.label(egui::RichText::new(file_stem(p)).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG))
                        .on_hover_text(p.as_str());
                    widgets::chip(ui, if self.dirty { "Unsaved" } else { "Saved" });
                }
                None => {
                    ui.label(egui::RichText::new("No controller").color(theme::TEXT_MUTED));
                }
            }
            ui.add_space(8.0);
            let new = widgets::secondary_button(ui, Some(Icon::Plus), "New");
            if new.clicked() {
                let dir = root.join("animations");
                create_asset(&dir, true, root, self, &mut AnimClipTab::default());
                self.open_request = false;
            }
            let open = widgets::secondary_button(ui, Some(Icon::Folder), "Open");
            if open.clicked() {
                self.files = scan_files(root, zestengine_assets::ANIM_CONTROLLER_EXTENSION);
            }
            egui::Popup::menu(&open).show(|ui| {
                if self.files.is_empty() {
                    ui.label(egui::RichText::new("No .animctrl files in the project").color(theme::TEXT_MUTED));
                }
                for f in self.files.clone() {
                    if ui.selectable_label(self.path.as_deref() == Some(f.as_str()), &f).clicked() {
                        self.open(root, &f);
                        ui.close();
                    }
                }
            });
            if self.path.is_some() {
                if widgets::icon_button_sized(ui, Icon::Target, "Frame all states", false, Vec2::splat(theme::H_FIELD)).clicked() {
                    self.frame_all();
                }
                let jump = widgets::secondary_button(ui, None, "Jump To State");
                let mut target: Option<(usize, Vec<usize>, usize)> = None;
                egui::Popup::menu(&jump).show(|ui| {
                    widgets::menu_frame(ui);
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        for (li, layer) in self.ctrl.layers.iter().enumerate() {
                            if self.ctrl.layers.len() > 1 {
                                widgets::eyebrow(ui, &layer.name);
                            }
                            for (path, idx, name) in layer.machine.walk_states() {
                                if widgets::menu_item(ui, None, &name, None).clicked() {
                                    target = Some((li, path, idx));
                                    ui.close();
                                }
                            }
                        }
                    });
                });
                if let Some((li, path, idx)) = target {
                    self.layer = li;
                    self.sub = path;
                    let n = Node::State(idx);
                    self.set_selected_nodes(vec![n], Some(n));
                    self.frame_nodes(&[n], 1.0);
                }
                if let Some(i) = live {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        widgets::chip(ui, &format!("Live: {}", scene.objects[i].name));
                    });
                }
            }
        });
    }

    fn empty_state(&mut self, ui: &mut egui::Ui, root: &Path, scene: &mut LoadedScene, selected: Option<usize>) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_VIEWPORT)).show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.3);
                ui.label(egui::RichText::new("No Animator Controller open").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                ui.label(
                    egui::RichText::new("Select a .animctrl in Files, press Open on an Animator, or create one.")
                        .color(theme::TEXT_MUTED),
                );
                ui.add_space(10.0);
                if widgets::primary_button(ui, Some(Icon::Plus), "Create Controller").clicked() {
                    create_asset(&root.join("animations"), true, root, self, &mut AnimClipTab::default());
                    self.open_request = false;
                    // Hook it to the selected object's Animator when there is one.
                    if let (Some(i), Some(path)) = (selected, self.path.clone())
                        && let Some(a) = animator_mut(scene, i)
                        && a.controller.is_empty()
                    {
                        a.controller = path;
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------- left panel

    fn layers_ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene) {
        ui.horizontal(|ui| {
            widgets::eyebrow(ui, "Layers");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::icon_button(ui, Icon::Plus, "Add layer", false).clicked() {
                    let n = self.ctrl.layers.len();
                    self.ctrl.layers.push(AnimLayer { name: format!("Layer {n}"), ..Default::default() });
                    self.layer = n;
                    self.sub.clear();
                    self.sel = Sel::None;
                    self.changed(scene);
                }
            });
        });
        let mut pick = None;
        let mut weight_edit = false;
        for k in 0..self.ctrl.layers.len() {
            let l = &self.ctrl.layers[k];
            let text = if k == 0 {
                l.name.clone()
            } else {
                format!("{}  ({}, {:.0}%)", l.name, if l.blend == LayerBlend::Additive { "add" } else { "override" }, l.weight * 100.0)
            };
            if ui.add_sized([ui.available_width(), theme::H_ROW], egui::Button::selectable(self.layer == k, text)).clicked() {
                pick = Some(k);
            }
            // Weight bar: drag to set the layer's weight (the base layer is always 1).
            if k > 0 {
                let (bar, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 6.0), Sense::click_and_drag());
                ui.painter().rect_filled(bar, 2.0, theme::LINE_PANEL);
                let w = self.ctrl.layers[k].weight.clamp(0.0, 1.0);
                let mut fill = bar;
                fill.set_width(bar.width() * w);
                ui.painter().rect_filled(fill, 2.0, theme::STATUS_INFO);
                if (resp.dragged() || resp.clicked())
                    && let Some(p) = resp.interact_pointer_pos()
                {
                    self.ctrl.layers[k].weight = ((p.x - bar.left()) / bar.width().max(1.0)).clamp(0.0, 1.0);
                    weight_edit = true;
                }
                resp.on_hover_text("Layer weight: drag to change");
            }
        }
        if weight_edit {
            self.changed(scene);
        }
        if let Some(k) = pick {
            self.layer = k;
            self.sub.clear();
            self.sel = Sel::None;
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, live: Option<usize>) {
        let mut edited = false;
        ui.horizontal(|ui| {
            widgets::eyebrow(ui, if live.is_some() { "Parameters (live)" } else { "Parameters" });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let add = widgets::icon_button(ui, Icon::Plus, "Add parameter", false);
                egui::Popup::menu(&add).show(|ui| {
                    for kind in AnimParamKind::ALL {
                        if ui.button(kind.label()).clicked() {
                            let mut name = kind.label().to_owned();
                            let mut n = 1;
                            while self.ctrl.parameter(&name).is_some() {
                                name = format!("{} {n}", kind.label());
                                n += 1;
                            }
                            self.ctrl.parameters.push(AnimParameter { name, kind, default: 0.0 });
                            edited = true;
                            ui.close();
                        }
                    }
                });
            });
        });
        if self.ctrl.parameters.is_empty() {
            ui.label(egui::RichText::new("None yet. Parameters drive transitions and blend trees.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
        let mut remove = None;
        let mut rename: Option<(String, String)> = None;
        let count = self.ctrl.parameters.len();
        for k in 0..count {
            let p = self.ctrl.parameters[k].clone();
            ui.horizontal(|ui| {
                ui.set_min_height(theme::H_FIELD);
                ui.label(egui::RichText::new(kind_glyph(p.kind)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM))
                    .on_hover_text(p.kind.label());
                if let Some(new) = commit_text(ui, ui.id().with(("zestengine-param-name", k)), &p.name, 96.0)
                    && self.ctrl.parameter(&new).is_none()
                {
                    rename = Some((p.name.clone(), new));
                }
                let mut value = match live.and_then(|i| animator(scene, i)) {
                    Some(a) => a.get_float(&p.name),
                    None => p.default,
                };
                let before = value;
                match p.kind {
                    AnimParamKind::Float => {
                        ui.add(egui::DragValue::new(&mut value).speed(0.02).max_decimals(3));
                    }
                    AnimParamKind::Int => {
                        let mut v = value.round() as i32;
                        ui.add(egui::DragValue::new(&mut v));
                        value = v as f32;
                    }
                    AnimParamKind::Bool => {
                        let mut b = value > 0.5;
                        ui.checkbox(&mut b, "");
                        value = if b { 1.0 } else { 0.0 };
                    }
                    AnimParamKind::Trigger => {
                        if live.is_some() {
                            if ui.small_button("Set").on_hover_text("Fire the trigger").clicked() {
                                value = 1.0;
                            }
                        } else {
                            let mut b = value > 0.5;
                            ui.checkbox(&mut b, "");
                            value = if b { 1.0 } else { 0.0 };
                        }
                    }
                }
                if value != before {
                    match live.and_then(|i| animator_mut(scene, i)) {
                        Some(a) => a.set_float(&p.name, value),
                        None => {
                            self.ctrl.parameters[k].default = value;
                            edited = true;
                        }
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if widgets::icon_button(ui, Icon::Trash, &format!("Remove {}", p.name), false).clicked() {
                        remove = Some(k);
                    }
                });
            });
        }
        if let Some((old, new)) = rename {
            self.ctrl.rename_parameter(&old, &new);
            edited = true;
        }
        if let Some(k) = remove {
            self.ctrl.parameters.remove(k);
            edited = true;
        }
        if edited {
            self.changed(scene);
        }
    }

    // ------------------------------------------------------------ inspector

    fn inspector_ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, object: Option<usize>, root: &Path) {
        let mut edited = false;
        match self.sel {
            Sel::Node(Node::State(s)) if s < machine_of(&self.ctrl, self.layer, &self.sub).states.len() => {
                edited |= self.state_inspector(ui, scene, s, object, root);
            }
            Sel::Transition(t) if t < machine_of(&self.ctrl, self.layer, &self.sub).transitions.len() => {
                let info = self.timeline_info(scene, t);
                edited |= self.transition_inspector(ui, t, info);
            }
            Sel::Node(Node::Any) => {
                ui.label(egui::RichText::new("Any State").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                widgets::info_note(ui, "Transitions from Any State are checked first, from whatever state is playing. Right-click it and choose Make Transition.");
                edited |= self.transition_list(ui, None);
            }
            Sel::Node(Node::Entry) => {
                ui.label(egui::RichText::new("Entry").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                widgets::info_note(ui, "The layer starts in its default state (orange). Right-click a state and choose Set as Default, or drag from Entry to a state.");
            }
            _ => {
                edited |= self.layer_inspector(ui, scene, object, root);
            }
        }
        if edited {
            self.changed(scene);
        }
    }

    fn layer_inspector(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, object: Option<usize>, root: &Path) -> bool {
        let mut edited = false;
        let li = self.layer;
        ui.label(egui::RichText::new("Layer").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
        ui.add_space(4.0);
        widgets::property_row(ui, "Name", |ui| {
            if let Some(new) = commit_text(ui, ui.id().with(("zestengine-layer-name", li)), &self.ctrl.layers[li].name, ui.available_width()) {
                self.ctrl.layers[li].name = new;
                edited = true;
            }
        });
        if li == 0 {
            widgets::info_note(ui, "The base layer always plays at full weight.");
        } else {
            let layer = &mut self.ctrl.layers[li];
            widgets::property_row(ui, "Weight", |ui| {
                edited |= crate::widgets::slider(ui, egui::Slider::new(&mut layer.weight, 0.0..=1.0)).changed();
            });
            widgets::property_row(ui, "Blending", |ui| {
                edited |= widgets::segmented(ui, &mut layer.blend, &[(LayerBlend::Override, "Override"), (LayerBlend::Additive, "Additive")]).changed();
            });
        }
        if li > 0 {
            let lower: Vec<String> = self.ctrl.layers[..li].iter().map(|l| l.name.clone()).collect();
            let layer = &mut self.ctrl.layers[li];
            widgets::property_row(ui, "Sync with", |ui| {
                let shown = layer.sync_with.and_then(|k| lower.get(k).cloned()).unwrap_or_else(|| "None".into());
                egui::ComboBox::from_id_salt(("zestengine-layer-sync", li)).width(130.0).selected_text(shown).show_ui(ui, |ui| {
                    widgets::menu_frame(ui);
                    edited |= ui.selectable_value(&mut layer.sync_with, None, "None").changed();
                    for (k, name) in lower.iter().enumerate() {
                        edited |= ui.selectable_value(&mut layer.sync_with, Some(k), name).changed();
                    }
                });
            });
            if layer.sync_with.is_some() {
                widgets::property_row(ui, "Timing", |ui| {
                        let r = ui.checkbox(&mut layer.sync_timing, "").on_hover_text("This layer's motion lengths join the shared state timing.");
                        edited |= r.changed();
                    });
                widgets::info_note(ui, "A sync layer plays the source layer's state machine; its own graph is ignored.");
            }
        }
        if self.ctrl.layers[li].sync_with.is_some() {
            edited |= self.motion_overrides_ui(ui, scene, object, root);
        }
        widgets::property_row(ui, "IK pass", |ui| {
                let r = ui.checkbox(&mut self.ctrl.layers[li].ik_pass, "").on_hover_text("Run this layer's state behaviours' IK callbacks.");
                edited |= r.changed();
            });
        ui.add_space(6.0);
        widgets::eyebrow(ui, "Mask");
        ui.label(
            egui::RichText::new("Humanoid parts and transforms this layer affects. All parts on with no transforms = everything.")
                .size(theme::SIZE_META)
                .color(theme::TEXT_DIM),
        );
        let mask_before = (!self.ctrl.layers[li].mask_asset.is_empty()).then(|| self.ctrl.layers[li].mask.clone());
        edited |= self.mask_asset_row(ui, root);
        edited |= crate::mask_silhouette::silhouette_ui(ui, &mut self.ctrl.layers[li].mask, 150.0);
        ui.horizontal_wrapped(|ui| {
            for part in zestengine_assets::MASK_PARTS {
                let mut on = self.ctrl.layers[li].mask.has_part(part);
                if ui.checkbox(&mut on, part.label()).changed() {
                    self.ctrl.layers[li].mask.set_part(part, on);
                    edited = true;
                }
            }
        });
        let mut remove = None;
        for (k, m) in self.ctrl.layers[li].mask.elements.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&m.path).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_BODY));
                edited |= ui.add(egui::DragValue::new(&mut m.weight).speed(0.01).range(0.0..=1.0)).on_hover_text("Weight; above 0.5 counts as active").changed();
                edited |= ui.checkbox(&mut m.descendants, "children").changed();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if widgets::icon_button(ui, Icon::Close, &format!("Remove {}", m.path), false).clicked() {
                        remove = Some(k);
                    }
                });
            });
        }
        if let Some(k) = remove {
            self.ctrl.layers[li].mask.elements.remove(k);
            edited = true;
        }
        let own: Vec<(String, Option<usize>)> = match object {
            Some(i) => {
                let tree = scene.skeleton_tree(i);
                if tree.is_empty() {
                    let rig = zestengine_engine::anim_graph::object_rig(scene, i).0;
                    rig.names.into_iter().zip(rig.parents).collect()
                } else {
                    tree
                }
            }
            None => Vec::new(),
        };
        edited |= crate::mask_ui::transforms_button(ui, egui::Id::new(("zestengine-mask-tree", li)), &mut self.ctrl.layers[li].mask, &own, root);
        // A referenced `.mask` asset is edited in place: every layer using it follows.
        if let Some(before) = mask_before
            && before != self.ctrl.layers[li].mask
        {
            let path = zestengine_core::asset_ref::resolve_path(root, &self.ctrl.layers[li].mask_asset);
            if let Err(e) = zestengine_assets::avatar_mask::save_avatar_mask(&path, &self.ctrl.layers[li].mask) {
                tracing::warn!("saving mask {}: {e:#}", path.display());
            }
        }
        if li > 0 {
            ui.add_space(10.0);
            if widgets::secondary_button(ui, Some(Icon::Trash), "Remove layer").clicked() {
                self.ctrl.layers.remove(li);
                self.layer = li - 1;
                self.sub.clear();
                edited = true;
            }
        }
        ui.add_space(10.0);
        widgets::info_note(
            ui,
            "Right-click the canvas to create states or sub-state machines. Double-click a sub-state machine to edit inside it (the path above the graph goes back up). Drag with Shift from a state (or use Make Transition) to connect two states. Scroll to zoom, drag empty space to pan.",
        );
        edited
    }

    /// The layer's `.mask` asset: pick one, save the inline mask as an asset
    /// (referenced by id, so it survives moves), or detach back to an inline copy.
    fn mask_asset_row(&mut self, ui: &mut egui::Ui, root: &Path) -> bool {
        let li = self.layer;
        let mut edited = false;
        let files = scan_files(root, "mask");
        let current = self.ctrl.layers[li].mask_asset.clone();
        let shown = if current.is_empty() {
            "None (inline)".to_owned()
        } else {
            zestengine_core::asset_ref::resolve(root, &current).unwrap_or_else(|| "(missing)".into())
        };
        let mut pick: Option<String> = None;
        widgets::property_row(ui, "Mask asset", |ui| {
            egui::ComboBox::from_id_salt(("zestengine-layer-mask-asset", li)).width(ui.available_width()).selected_text(shown).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                if ui.selectable_label(current.is_empty(), "None (inline)").clicked() {
                    pick = Some(String::new());
                }
                for f in &files {
                    if ui.selectable_label(false, f).clicked() {
                        pick = Some(zestengine_core::asset_ref::reference(root, f));
                    }
                }
            });
        });
        if let Some(p) = pick {
            // Choosing an asset loads it; going back to None keeps the copy.
            if !p.is_empty() {
                let path = zestengine_core::asset_ref::resolve_path(root, &p);
                match zestengine_assets::avatar_mask::load_avatar_mask(&path) {
                    Ok(m) => self.ctrl.layers[li].mask = m,
                    Err(e) => tracing::warn!("mask {}: {e:#}", path.display()),
                }
            }
            self.ctrl.layers[li].mask_asset = p;
            edited = true;
        }
        if current.is_empty() && widgets::secondary_button(ui, None, "Save as .mask asset").clicked() {
            let layer_name = self.ctrl.layers[li].name.clone();
            let rel = unique_mask_path(root, &layer_name);
            match zestengine_assets::avatar_mask::save_avatar_mask(&root.join(&rel), &self.ctrl.layers[li].mask) {
                Ok(()) => {
                    self.ctrl.layers[li].mask_asset = zestengine_core::asset_ref::reference(root, &rel);
                    edited = true;
                }
                Err(e) => tracing::warn!("saving mask {rel}: {e:#}"),
            }
        }
        edited
    }

    /// A sync layer's per-state motions: every state of the source layer with
    /// an Override switch; an overridden state edits its own clip or blend tree.
    fn motion_overrides_ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, object: Option<usize>, root: &Path) -> bool {
        let li = self.layer;
        let Some(src) = self.ctrl.layers[li].sync_with.filter(|&k| k < li) else { return false };
        let mut edited = false;
        if self.clip_files.is_empty() {
            self.clip_files = scan_files(root, zestengine_assets::ANIM_CLIP_EXTENSION);
        }
        let choices = self.clip_choices(scene, object);
        let params = self.ctrl.parameters.clone();
        let flat = self.ctrl.layers[src].machine.clone();
        let names = flat.leaf_names();
        let gone = self.ctrl.layers[li].prune_overrides(&flat);
        edited |= gone > 0;
        ui.add_space(6.0);
        widgets::eyebrow(ui, "Motion overrides");
        ui.label(
            egui::RichText::new("States without an override play the source layer's motion.").size(theme::SIZE_META).color(theme::TEXT_DIM),
        );
        let values: std::collections::HashMap<String, f32> = params.iter().map(|p| (p.name.clone(), self.bt_preview.get(&p.name).copied().unwrap_or(p.default))).collect();
        let source_motion = |name: &str| self.ctrl.layers[src].machine.flattened_motion(name);
        let mut changes: Vec<(String, Option<Motion>)> = Vec::new();
        let mut scrubbed: Vec<(String, f32)> = Vec::new();
        for name in &names {
            let current = self.ctrl.layers[li].override_for(name).map(|o| o.motion.clone());
            ui.push_id(("zestengine-override", li, name), |ui| {
                let mut on = current.is_some();
                if ui.checkbox(&mut on, name).changed() {
                    changes.push((name.clone(), on.then(|| source_motion(name).unwrap_or_default())));
                }
                let Some(mut motion) = current else { return };
                let mut local = false;
                let mut is_tree = matches!(motion, Motion::BlendTree(_));
                widgets::property_row(ui, "Motion", |ui| {
                    if widgets::segmented(ui, &mut is_tree, &[(false, "Clip"), (true, "Blend Tree")]).changed() {
                        motion = if is_tree {
                            Motion::BlendTree(BlendTree {
                                name: name.clone(),
                                parameter: params.iter().find(|p| p.kind == AnimParamKind::Float).map(|p| p.name.clone()).unwrap_or_default(),
                                ..Default::default()
                            })
                        } else {
                            Motion::Clip(String::new())
                        };
                        local = true;
                    }
                });
                match &mut motion {
                    Motion::Clip(clip) => {
                        widgets::property_row(ui, "Clip", |ui| {
                            local |= clip_combo(ui, egui::Id::new(("zestengine-override-clip", li, name)), clip, &choices);
                        });
                    }
                    Motion::BlendTree(tree) => {
                        let (motions, lengths) = tree_clip_data(ui, scene, tree);
                        let mut env = BtEnv { params: &params, choices: &choices, values: &values, scrub: Vec::new(), motions: &motions, lengths: &lengths };
                        local |= blend_tree_ui(ui, tree, &mut env, egui::Id::new(("zestengine-override-tree", li, name)), 0);
                        scrubbed.extend(env.scrub);
                    }
                }
                if local {
                    changes.push((name.clone(), Some(motion)));
                }
            });
            widgets::divider(ui);
        }
        for (n, m) in changes {
            self.ctrl.layers[li].set_override(&n, m);
            edited = true;
        }
        for (n, v) in scrubbed {
            self.bt_preview.insert(n, v);
        }
        edited
    }

    fn clip_choices(&self, scene: &LoadedScene, object: Option<usize>) -> Vec<String> {
        let mut out: Vec<String> = object.and_then(|i| animator(scene, i)).map(|a| a.state.clips.clone()).unwrap_or_default();
        out.extend(self.clip_files.iter().cloned());
        let mut seen = std::collections::HashSet::new();
        out.retain(|c| seen.insert(c.clone()));
        out
    }

    fn state_inspector(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, s: usize, object: Option<usize>, root: &Path) -> bool {
        let mut edited = false;
        let li = self.layer;
        let params = self.ctrl.parameters.clone();
        if self.clip_files.is_empty() {
            self.clip_files = scan_files(root, zestengine_assets::ANIM_CLIP_EXTENSION);
        }
        let choices = self.clip_choices(scene, object);
        // Parameter values for the blend tree preview: the running Animator's,
        // else the values scrubbed here (starting from the defaults).
        let live_anim = self.playing_obj.and_then(|i| animator(scene, i));
        let values: std::collections::HashMap<String, f32> = params
            .iter()
            .map(|p| {
                let v = match live_anim {
                    Some(a) => a.get_float(&p.name),
                    None => self.bt_preview.get(&p.name).copied().unwrap_or(p.default),
                };
                (p.name.clone(), v)
            })
            .collect();
        let mut scrubbed: Vec<(String, f32)> = Vec::new();
        let machine = machine_of_mut(&mut self.ctrl, li, &self.sub);
        let is_default = machine.default_index() == Some(s);
        let is_sub = machine.states[s].is_sub_machine();
        ui.horizontal(|ui| {
            let title = if is_sub { "Sub-State Machine" } else { "State" };
            ui.label(egui::RichText::new(title).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
            if is_default {
                widgets::chip(ui, "Default");
            }
        });
        ui.add_space(4.0);
        let name = machine.states[s].name.clone();
        widgets::property_row(ui, "Name", |ui| {
            if let Some(new) = commit_text(ui, ui.id().with(("zestengine-state-name", li, s)), &name, ui.available_width())
                && machine.state_index(&new).is_none()
                && !new.contains('/')
            {
                machine.rename_state(&name, &new);
                edited = true;
            }
        });
        if let Some(inner) = machine.states[s].sub_machine.as_deref() {
            let entry = inner.default_index().map(|i| inner.states[i].name.clone()).unwrap_or_else(|| "nothing (empty)".into());
            ui.label(
                egui::RichText::new(format!("{} states, enters {entry}", inner.states.len()))
                    .size(theme::SIZE_META)
                    .color(theme::TEXT_MUTED),
            );
            widgets::info_note(
                ui,
                "A transition into this node enters its default state; a transition from it leaves from whichever inner state is playing. Its Any State applies only inside it. Scripts name inner states by path, e.g. play_state(\"Locomotion/Walk\").",
            );
            ui.add_space(6.0);
            let mut open = false;
            ui.horizontal(|ui| {
                open = widgets::secondary_button(ui, None, "Open").clicked();
                if !is_default && widgets::secondary_button(ui, None, "Set as Default").clicked() {
                    machine.default_state = name.clone();
                    edited = true;
                }
                if widgets::secondary_button(ui, Some(Icon::Trash), "Delete").clicked() {
                    machine.remove_state(s);
                    self.sel = Sel::None;
                    edited = true;
                }
            });
            if open {
                self.enter_sub(s);
            } else if let Sel::Node(Node::State(_)) = self.sel {
                widgets::section_gap(ui);
                edited |= self.transition_list(ui, Some(name));
            }
            return edited;
        }
        let st = &mut machine.states[s];
        let mut is_tree = matches!(st.motion, Motion::BlendTree(_));
        widgets::property_row(ui, "Motion", |ui| {
            if widgets::segmented(ui, &mut is_tree, &[(false, "Clip"), (true, "Blend Tree")]).changed() {
                st.motion = if is_tree {
                    Motion::BlendTree(BlendTree {
                        name: st.name.clone(),
                        parameter: params.iter().find(|p| p.kind == AnimParamKind::Float).map(|p| p.name.clone()).unwrap_or_default(),
                        ..Default::default()
                    })
                } else {
                    Motion::Clip(String::new())
                };
                edited = true;
            }
        });
        match &mut st.motion {
            Motion::Clip(clip) => {
                widgets::property_row(ui, "Clip", |ui| {
                    edited |= clip_combo(ui, egui::Id::new(("zestengine-state-clip", li, s)), clip, &choices);
                });
            }
            Motion::BlendTree(tree) => {
                let (motions, lengths) = tree_clip_data(ui, scene, tree);
                let mut env = BtEnv { params: &params, choices: &choices, values: &values, scrub: Vec::new(), motions: &motions, lengths: &lengths };
                edited |= blend_tree_ui(ui, tree, &mut env, egui::Id::new(("zestengine-tree", li, s)), 0);
                scrubbed = env.scrub;
            }
        }
        widgets::property_row(ui, "Speed", |ui| {
            edited |= ui.add(egui::DragValue::new(&mut st.speed).speed(0.02).range(-10.0..=10.0)).changed();
        });
        widgets::property_row(ui, "Speed param", |ui| {
            edited |= param_combo(ui, egui::Id::new(("zestengine-state-speedp", li, s)), &mut st.speed_parameter, &params, true, |k| {
                k == AnimParamKind::Float
            });
        });
        widgets::property_row(ui, "Loop", |ui| edited |= ui.checkbox(&mut st.looping, "").changed());
        widgets::property_row(ui, "Write defaults", |ui| {
                let r = ui.checkbox(&mut st.write_defaults, "").on_hover_text("Off: values this state does not animate keep their last value instead of returning to the defaults.");
                edited |= r.changed();
            });
        widgets::property_row(ui, "Loop pose", |ui| {
                let r = ui.checkbox(&mut st.loop_pose, "").on_hover_text("Spread the difference between the first and last frame over the cycle so the loop closes without a pop.");
                edited |= r.changed();
            });
        widgets::property_row(ui, "On enter", |ui| {
            edited |= ui.add(egui::TextEdit::singleline(&mut st.on_enter).hint_text("event name").desired_width(ui.available_width())).changed();
        });
        widgets::property_row(ui, "On exit", |ui| {
            edited |= ui.add(egui::TextEdit::singleline(&mut st.on_exit).hint_text("event name").desired_width(ui.available_width())).changed();
        });
        ui.label(
            egui::RichText::new("State events reach on_animation_event and AnimationEvent, like clip events.")
                .size(theme::SIZE_META)
                .color(theme::TEXT_DIM),
        );
        ui.add_space(6.0);
        edited |= crate::anim_behaviour_ui::state_behaviour_fields(ui, egui::Id::new(("zestengine-state-behaviours", li, s)), st, &params);
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if !is_default && widgets::secondary_button(ui, None, "Set as Default").clicked() {
                machine.default_state = name.clone();
                edited = true;
            }
            if widgets::secondary_button(ui, Some(Icon::Trash), "Delete").clicked() {
                machine.remove_state(s);
                self.sel = Sel::None;
                edited = true;
            }
        });
        if let Sel::Node(Node::State(_)) = self.sel {
            widgets::section_gap(ui);
            edited |= self.transition_list(ui, Some(name));
        }
        for (n, v) in scrubbed {
            match self.playing_obj.and_then(|i| animator_mut(scene, i)) {
                Some(a) => a.set_float(&n, v),
                None => {
                    self.bt_preview.insert(n, v);
                }
            }
        }
        edited
    }

    /// Outgoing transitions of a state (None = Any State), click to select.
    fn transition_list(&mut self, ui: &mut egui::Ui, from: Option<String>) -> bool {
        widgets::eyebrow(ui, "Transitions");
        let machine = machine_of(&self.ctrl, self.layer, &self.sub);
        let list: Vec<(usize, String)> = machine
            .transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| t.from == from)
            .map(|(k, t)| (k, t.to.clone()))
            .collect();
        if list.is_empty() {
            ui.label(egui::RichText::new("None. Shift-drag to a state to add one.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
        for (k, to) in list {
            if ui.add_sized([ui.available_width(), theme::H_ROW], egui::Button::selectable(false, format!("{}  {to}", ph::ARROW_RIGHT))).clicked() {
                self.sel = Sel::Transition(k);
            }
        }
        false
    }

    /// Lengths and names for the selected transition's timeline, or why there
    /// is none.
    fn timeline_info(&self, scene: &mut LoadedScene, t: usize) -> Result<TlInfo, &'static str> {
        let m = machine_of(&self.ctrl, self.layer, &self.sub);
        let tr = &m.transitions[t];
        let Some(from) = tr.from.as_deref() else {
            return Err("Transitions from Any State have no single source to draw. Select one that leaves a state to see its timeline.");
        };
        let (Some(si), Some(di)) = (m.state_index(from), m.state_index(&tr.to)) else {
            return Err("The source or target state no longer exists.");
        };
        let (src, dst) = (&m.states[si], &m.states[di]);
        if src.is_sub_machine() || dst.is_sub_machine() {
            return Err("The timeline needs a plain state at both ends. Pick the state inside the sub-machine.");
        }
        let mut estimated = false;
        let mut length = |s: &AnimState| {
            // Speed scales the clip: a faster state is shorter on the axis.
            match motion_length(scene, &s.motion) {
                Some(l) => l / s.speed.abs().max(0.01),
                None => {
                    estimated = true;
                    1.0
                }
            }
        };
        let (src_dur, dst_dur) = (length(src), length(dst));
        Ok(TlInfo { src: src.name.clone(), dst: dst.name.clone(), src_dur, dst_dur, src_loop: src.looping, dst_loop: dst.looping, estimated })
    }

    fn transition_inspector(&mut self, ui: &mut egui::Ui, t: usize, info: Result<TlInfo, &'static str>) -> bool {
        let mut edited = false;
        let params = self.ctrl.parameters.clone();
        let machine = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);

        // Every transition on this edge, with Solo / Mute (a muted one never
        // fires; while any is soloed only the soloed ones of this source do).
        let (from, to) = (machine.transitions[t].from.clone(), machine.transitions[t].to.clone());
        let siblings: Vec<usize> = (0..machine.transitions.len()).filter(|&k| machine.transitions[k].from == from && machine.transitions[k].to == to).collect();
        widgets::eyebrow(ui, "Transitions on this edge");
        for &k in &siblings {
            ui.horizontal(|ui| {
                let label = format!("{}  {}  {}", from.as_deref().unwrap_or("Any State"), ph::ARROW_RIGHT, to);
                let w = (ui.available_width() - 110.0).max(60.0);
                if ui.add_sized([w, theme::H_ROW], egui::Button::selectable(k == t, label)).clicked() {
                    self.sel = Sel::Transition(k);
                }
                let tk = &mut machine.transitions[k];
                widgets::trough(ui, |ui| {
                    if widgets::segment(ui, tk.solo, "S").on_hover_text("Solo: only soloed transitions from this source can fire").clicked() {
                        tk.solo = !tk.solo;
                        edited = true;
                    }
                    if widgets::segment(ui, tk.muted, "M").on_hover_text("Mute: this transition never fires").clicked() {
                        tk.muted = !tk.muted;
                        edited = true;
                    }
                });
            });
        }
        ui.add_space(6.0);
        let state_names: Vec<String> = machine.states.iter().map(|s| s.name.clone()).collect();
        let tr = &mut machine.transitions[t];
        let from = tr.from.clone().unwrap_or_else(|| "Any State".into());
        ui.label(egui::RichText::new("Transition").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
        ui.label(egui::RichText::new(format!("{from}  {}  {}", ph::ARROW_RIGHT, tr.to)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_BODY));
        ui.add_space(6.0);
        // Re-point the transition: Any State or a state as the source, any state as the destination.
        widgets::property_row(ui, "Source", |ui| {
            egui::ComboBox::from_id_salt(("zestengine-tr-source", t)).width(150.0).selected_text(from.clone()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                edited |= ui.selectable_value(&mut tr.from, None, "Any State").changed();
                for n in &state_names {
                    edited |= ui.selectable_value(&mut tr.from, Some(n.clone()), n).changed();
                }
            });
        });
        widgets::property_row(ui, "Destination", |ui| {
            egui::ComboBox::from_id_salt(("zestengine-tr-dest", t)).width(150.0).selected_text(tr.to.clone()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for n in &state_names {
                    edited |= ui.selectable_value(&mut tr.to, n.clone(), n).changed();
                }
            });
        });
        widgets::property_row(ui, "Has exit time", |ui| edited |= ui.checkbox(&mut tr.has_exit_time, "").changed());
        ui.add_enabled_ui(tr.has_exit_time, |ui| {
            widgets::property_row(ui, "Exit time", |ui| {
                edited |= ui.add(egui::DragValue::new(&mut tr.exit_time).speed(0.01).range(0.0..=100.0)).on_hover_text("Normalized time of the source state").changed();
            });
        });
        widgets::property_row(ui, "Duration (s)", |ui| {
            edited |= ui.add(egui::DragValue::new(&mut tr.duration).speed(0.01).range(0.0..=10.0)).changed();
        });
        widgets::property_row(ui, "Offset", |ui| {
            edited |= ui.add(egui::DragValue::new(&mut tr.offset).speed(0.01).range(0.0..=1.0)).on_hover_text("Normalized start time in the target state").changed();
        });
        widgets::property_row(ui, "Interruption", |ui| {
            let mut src = tr.interruption_source();
            let before = src;
            egui::ComboBox::from_id_salt(("zestengine-tr-interruption", t)).selected_text(src.label()).show_ui(ui, |ui| {
                for s in InterruptionSource::ALL {
                    ui.selectable_value(&mut src, s, s.label());
                }
            });
            if src != before {
                set_interruption(tr, src);
                edited = true;
            }
        });
        ui.add_enabled_ui(tr.interruption_source() != InterruptionSource::None, |ui| {
            widgets::property_row(ui, "Ordered interruption", |ui| {
                edited |= ui.checkbox(&mut tr.ordered_interruption, "").on_hover_text("Only transitions ahead of this one may interrupt it").changed();
            });
        });
        ui.add_space(6.0);
        match &info {
            Ok(info) => {
                if let Some((exit, duration, offset)) = timeline_widget(ui, info, tr, t, &mut self.tl_drag, &mut self.tl_shift) {
                    if tr.has_exit_time {
                        tr.exit_time = exit.min(100.0);
                    }
                    tr.duration = duration.min(10.0);
                    tr.offset = offset;
                    edited = true;
                }
                ui.label(
                    egui::RichText::new("Left handle: exit time. Right handle: duration. Lower bar: offset. Ruler: move the fade. Esc cancels.")
                        .size(theme::SIZE_META)
                        .color(theme::TEXT_DIM),
                );
                if info.estimated {
                    widgets::info_note(ui, "A clip length could not be read, so 1 s stands in for it. Imported clips and unsaved clips have no length here.");
                }
            }
            Err(why) => widgets::info_note(ui, why),
        }
        if tr.from.is_none() {
            widgets::property_row(ui, "To self", |ui| edited |= ui.checkbox(&mut tr.can_transition_to_self, "").changed());
        }
        ui.add_space(6.0);
        widgets::eyebrow(ui, "Conditions");
        let mut remove = None;
        for (k, c) in tr.conditions.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                edited |= param_combo(ui, egui::Id::new(("zestengine-cond-param", t, k)), &mut c.parameter, &params, false, |_| true);
                let kind = params.iter().find(|p| p.name == c.parameter).map(|p| p.kind).unwrap_or_default();
                let ops: &[CondOp] = match kind {
                    AnimParamKind::Float => &[CondOp::Greater, CondOp::Less],
                    AnimParamKind::Int => &[CondOp::Greater, CondOp::Less, CondOp::Equals, CondOp::NotEquals],
                    AnimParamKind::Bool => &[CondOp::If, CondOp::IfNot],
                    AnimParamKind::Trigger => &[CondOp::If],
                };
                if !ops.contains(&c.op) {
                    c.op = ops[0];
                    edited = true;
                }
                egui::ComboBox::from_id_salt(("zestengine-cond-op", t, k)).width(78.0).selected_text(c.op.label()).show_ui(ui, |ui| {
                    for &op in ops {
                        edited |= ui.selectable_value(&mut c.op, op, op.label()).changed();
                    }
                });
                if c.op.uses_threshold() {
                    let speed = if kind == AnimParamKind::Int { 1.0 } else { 0.02 };
                    edited |= ui.add(egui::DragValue::new(&mut c.threshold).speed(speed)).changed();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if widgets::icon_button(ui, Icon::Close, "Remove condition", false).clicked() {
                        remove = Some(k);
                    }
                });
            });
        }
        if let Some(k) = remove {
            tr.conditions.remove(k);
            edited = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add condition").clicked() {
            let first = params.first();
            tr.conditions.push(AnimCondition {
                parameter: first.map(|p| p.name.clone()).unwrap_or_default(),
                op: match first.map(|p| p.kind) {
                    Some(AnimParamKind::Float) | Some(AnimParamKind::Int) => CondOp::Greater,
                    _ => CondOp::If,
                },
                threshold: 0.0,
            });
            edited = true;
        }
        if !tr.has_exit_time && tr.conditions.is_empty() {
            ui.add_space(4.0);
            widgets::info_note(ui, "This transition never fires: add a condition or enable exit time.");
        }
        ui.add_space(10.0);
        if widgets::secondary_button(ui, Some(Icon::Trash), "Delete transition").clicked() {
            machine.transitions.remove(t);
            self.sel = Sel::None;
            edited = true;
        }
        edited
    }

    // ------------------------------------------------------ sub-state machines

    fn enter_sub(&mut self, state: usize) {
        self.sub.push(state);
        self.sel = Sel::None;
        self.dragging = None;
        self.linking = None;
        self.frame_all();
    }

    /// Canvas-space rect of a node.
    fn node_canvas_rect(&self, node: Node) -> Rect {
        let size = match node {
            Node::State(_) => NODE_SIZE,
            _ => Vec2::new(120.0, 32.0),
        };
        let c = self.node_pos(node);
        Rect::from_center_size(Pos2::new(c[0], c[1]), size)
    }

    /// Fit `nodes` in the view (zoom capped at `max_zoom`).
    fn frame_nodes(&mut self, nodes: &[Node], max_zoom: f32) {
        let Some(bounds) = cv::bounds_of(nodes.iter().map(|&n| self.node_canvas_rect(n))) else {
            return;
        };
        let view = if self.view_size.x > 1.0 && self.view_size.y > 1.0 { self.view_size } else { Vec2::new(800.0, 500.0) };
        let (pan, zoom) = cv::frame_bounds(bounds, view);
        self.pan = pan;
        self.zoom = zoom.min(max_zoom);
    }

    /// Fit the shown machine's nodes, never zooming in past 1x.
    fn frame_all(&mut self) {
        let nodes = self.nodes();
        self.frame_nodes(&nodes, 1.0);
    }

    /// F: the selection (a selected transition counts as both its ends),
    /// or everything when nothing is selected.
    fn frame_selected(&mut self) {
        let m = machine_of(&self.ctrl, self.layer, &self.sub);
        let mut nodes = self.selected_nodes();
        if let Sel::Transition(k) = self.sel
            && let Some(t) = m.transitions.get(k)
        {
            nodes.extend(t.from.as_deref().and_then(|f| m.state_index(f)).map(Node::State).or(Some(Node::Any)));
            nodes.extend(m.state_index(&t.to).map(Node::State));
        }
        if nodes.is_empty() {
            nodes = self.nodes();
        }
        self.frame_nodes(&nodes, cv::ZOOM_MAX);
    }

    /// The selected nodes (more than one after a box select or Ctrl-click).
    fn selected_nodes(&self) -> Vec<Node> {
        match self.sel {
            Sel::Node(n) if self.multi_for == self.sel && self.multi.contains(&n) => self.multi.clone(),
            Sel::Node(n) => vec![n],
            _ => Vec::new(),
        }
    }

    fn set_selected_nodes(&mut self, nodes: Vec<Node>, primary: Option<Node>) {
        let primary = primary.filter(|p| nodes.contains(p)).or(nodes.last().copied());
        self.sel = primary.map(Sel::Node).unwrap_or(Sel::None);
        self.multi = nodes;
        self.multi_for = self.sel;
    }

    /// Delete the selected states, or the selected transition.
    fn delete_selected(&mut self) -> bool {
        let mut states: Vec<usize> = self.selected_nodes().into_iter().filter_map(|n| if let Node::State(i) = n { Some(i) } else { None }).collect();
        states.sort_unstable_by(|a, b| b.cmp(a));
        states.dedup();
        let sel = self.sel;
        let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
        let mut edited = false;
        for i in states {
            if i < m.states.len() {
                m.remove_state(i);
                edited = true;
            }
        }
        if let Sel::Transition(k) = sel
            && k < m.transitions.len()
        {
            m.transitions.remove(k);
            edited = true;
        }
        self.sel = Sel::None;
        self.multi.clear();
        edited
    }

    /// Sub-machine state indices for a path of names from the layer root.
    fn sub_indices(&self, names: &[String]) -> Option<Vec<usize>> {
        let mut m = &self.ctrl.layers.get(self.layer)?.machine;
        let mut out = Vec::new();
        for n in names {
            let i = m.state_index(n)?;
            m = m.states[i].sub_machine.as_deref()?;
            out.push(i);
        }
        Some(out)
    }

    /// Auto Live Link: show the machine that holds the playing state,
    /// switching halfway through a transition between machines.
    fn live_follow(&mut self, scene: &LoadedScene, live: Option<usize>) {
        let lp = live.and_then(|i| animator(scene, i)).and_then(|a| a.state.layers.get(self.layer));
        let (true, Some(lp)) = (self.auto_follow, lp) else {
            self.last_follow = None;
            return;
        };
        let flat = self.ctrl.layers[self.layer].machine.flatten();
        let source = lp.previous.and_then(|(p, _)| flat.states.get(p)).map(|s| s.name.as_str());
        let from_any = lp.transition.and_then(|ft| flat.transitions.get(ft)).is_some_and(|t| t.from.is_none());
        let want = cv::follow_machine_path(&lp.state_name, source, lp.in_transition(), lp.transition_progress(), from_any);
        if self.last_follow.as_ref() == Some(&want) {
            return;
        }
        if let Some(path) = self.sub_indices(&want)
            && path != self.sub
        {
            self.sub = path;
            self.sel = Sel::None;
            self.dragging = None;
            self.node_drag = None;
            self.linking = None;
            self.frame_all();
        }
        self.last_follow = Some(want);
    }

    /// Runtime name prefix of the shown level ("" or "Outer/Inner/").
    fn level_prefix(&self) -> String {
        let mut m = &self.ctrl.layers[self.layer].machine;
        let mut out = String::new();
        for &i in &self.sub {
            let Some(s) = m.states.get(i) else { break };
            let Some(sub) = s.sub_machine.as_deref() else { break };
            out.push_str(&s.name);
            out.push('/');
            m = sub;
        }
        out
    }

    /// Layer > Sub > Sub path above the graph; click a step to go back up.
    fn breadcrumb(&mut self, ui: &mut egui::Ui) {
        let mut names = vec![self.ctrl.layers[self.layer].name.clone()];
        let mut m = &self.ctrl.layers[self.layer].machine;
        for &i in &self.sub {
            let Some(s) = m.states.get(i) else { break };
            let Some(sub) = s.sub_machine.as_deref() else { break };
            names.push(s.name.clone());
            m = sub;
        }
        let mut go = None;
        ui.horizontal(|ui| {
            ui.set_min_height(theme::H_ROW);
            for (k, n) in names.iter().enumerate() {
                if k > 0 {
                    ui.label(egui::RichText::new(ph::CARET_RIGHT).color(theme::TEXT_DIM));
                }
                let last = k + 1 == names.len();
                let text = egui::RichText::new(n).color(if last { theme::TEXT_STRONG } else { theme::TEXT_MUTED });
                let resp = ui.add(egui::Button::new(text).frame(false));
                if !last && resp.on_hover_text("Go up to this level").clicked() {
                    go = Some(k);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::icon_button(ui, Icon::Eye, "Auto Live Link: follow the playing state, into sub-machines too", self.auto_follow).clicked() {
                    self.auto_follow = !self.auto_follow;
                    self.last_follow = None;
                    self.last_live_state = None;
                }
                if widgets::icon_button(ui, Icon::Snap, "Snap nodes to a 10 unit grid", self.snap).clicked() {
                    self.snap = !self.snap;
                }
                if widgets::icon_button(ui, Icon::Target, "Frame selection, or all (F, A)", false).clicked() {
                    self.frame_selected();
                }
            });
        });
        if let Some(k) = go {
            self.sub.truncate(k);
            self.sel = Sel::None;
            self.dragging = None;
            self.linking = None;
            self.frame_all();
        }
    }

    /// Whether runtime transition `ft` (flattened) came from transition `k` of
    /// the shown level.
    fn transition_matches(&self, k: usize, ft: &AnimTransition, prefix: &str) -> bool {
        let Some(t) = machine_of(&self.ctrl, self.layer, &self.sub).transitions.get(k) else {
            return false;
        };
        under(&ft.to, prefix, &t.to)
            && match (&t.from, &ft.from) {
                (None, None) => prefix.is_empty(),
                (None, Some(f)) => !prefix.is_empty() && f.starts_with(prefix),
                (Some(a), Some(f)) => under(f, prefix, a),
                (Some(_), None) => false,
            }
    }

    // --------------------------------------------------------------- canvas

    fn to_screen(&self, rect: Rect, p: [f32; 2]) -> Pos2 {
        rect.center() + (self.pan + Vec2::from(p)) * self.zoom
    }

    fn node_pos(&self, node: Node) -> [f32; 2] {
        let m = machine_of(&self.ctrl, self.layer, &self.sub);
        match node {
            Node::State(i) => m.states.get(i).map(|s| s.position).unwrap_or_default(),
            Node::Any => m.any_state_position,
            Node::Entry => m.entry_position,
        }
    }

    fn node_rect(&self, rect: Rect, node: Node) -> Rect {
        let size = match node {
            Node::State(_) => NODE_SIZE,
            _ => Vec2::new(120.0, 32.0),
        } * self.zoom;
        Rect::from_center_size(self.to_screen(rect, self.node_pos(node)), size)
    }

    fn nodes(&self) -> Vec<Node> {
        let n = machine_of(&self.ctrl, self.layer, &self.sub).states.len();
        let mut v: Vec<Node> = vec![Node::Entry, Node::Any];
        v.extend((0..n).map(Node::State));
        v
    }

    fn node_at(&self, rect: Rect, p: Pos2) -> Option<Node> {
        self.nodes().into_iter().rev().find(|&n| self.node_rect(rect, n).contains(p))
    }

    /// One screen segment per edge: all transitions between a node pair.
    fn edge_draws(&self, rect: Rect) -> Vec<EdgeDraw> {
        let m = machine_of(&self.ctrl, self.layer, &self.sub);
        let pairs: Vec<Option<(Option<usize>, usize)>> = m
            .transitions
            .iter()
            .map(|t| -> Option<(Option<usize>, usize)> {
                let to = m.state_index(&t.to)?;
                let from = match &t.from {
                    None => None,
                    Some(f) => Some(m.state_index(f)?),
                };
                Some((from, to))
            })
            .collect();
        let groups = cv::group_edges(&pairs);
        groups
            .iter()
            .map(|g| {
                let from = g.from.map(Node::State).unwrap_or(Node::Any);
                let a = self.node_rect(rect, from).center();
                let b = self.node_rect(rect, Node::State(g.to)).center();
                let off = cv::edge_offset(a, b, cv::has_reverse(&groups, g), self.zoom);
                EdgeDraw { group: g.clone(), a: a + off, b: b + off }
            })
            .collect()
    }

    /// Which transition of an edge a click selects: the selected one if it is
    /// on the edge, else the first.
    fn pick_in_edge(&self, group: &cv::EdgeGroup) -> usize {
        match self.sel {
            Sel::Transition(k) if group.transitions.contains(&k) => k,
            _ => group.transitions[0],
        }
    }

    /// Drag strips on the graph's left and right edges (added after the graph
    /// so they take the pointer first): the only way the side panels resize.
    fn panel_handles(&mut self, ui: &mut egui::Ui, area: egui::Rect) {
        for (left, x) in [(true, area.left()), (false, area.right())] {
            let strip = egui::Rect::from_x_y_ranges(x - 3.0..=x + 3.0, area.y_range());
            let handle = ui.interact(strip, ui.id().with(("zestengine-animctrl-panel-handle", left)), Sense::drag());
            if handle.hovered() || handle.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            if handle.dragged() {
                let dx = handle.drag_delta().x;
                if left {
                    self.left_width = (self.left_width + dx).clamp(180.0, 420.0);
                } else {
                    self.right_width = (self.right_width - dx).clamp(240.0, 520.0);
                }
            }
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, live: Option<usize>, root: &Path) {
        let rect = ui.available_rect_before_wrap();
        self.view_size = rect.size();
        // Each machine remembers its own view: store this one, or restore the
        // one of the machine just switched to (a first visit keeps its framing).
        let key = (self.layer, self.sub.clone());
        if key == self.view_key {
            self.views.insert(key, (self.pan, self.zoom));
        } else {
            if let Some(&(pan, zoom)) = self.views.get(&key) {
                self.pan = pan;
                self.zoom = zoom;
            }
            self.view_key = key;
        }
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let pointer = ui.input(|i| i.pointer.hover_pos());
        let mut edited = false;

        // Zoom around the pointer.
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0
                && let Some(p) = pointer
            {
                let before = (p - rect.center()) / self.zoom - self.pan;
                self.zoom = (self.zoom * (1.0 + scroll * 0.0015)).clamp(cv::ZOOM_MIN, cv::ZOOM_MAX);
                self.pan = (p - rect.center()) / self.zoom - before;
            }
        }

        // Grid.
        let step = 32.0 * self.zoom;
        if step > 6.0 {
            let origin = rect.center() + self.pan * self.zoom;
            let mut x = rect.left() + (origin.x - rect.left()).rem_euclid(step);
            while x < rect.right() {
                painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0_f32, theme::with_alpha(theme::LINE_PANEL, 0.6)));
                x += step;
            }
            let mut y = rect.top() + (origin.y - rect.top()).rem_euclid(step);
            while y < rect.bottom() {
                painter.line_segment([Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)], Stroke::new(1.0_f32, theme::with_alpha(theme::LINE_PANEL, 0.6)));
                y += step;
            }
        }

        // Live playback of this layer.
        let lp = live.and_then(|i| animator(scene, i)).and_then(|a| a.state.layers.get(self.layer).cloned());
        // The runtime plays the flattened machine (`Outer/Inner` names, its
        // own indices): map those onto this level's nodes by name.
        let flat = lp.as_ref().map(|_| self.ctrl.layers[self.layer].machine.flatten());
        let flat_state = |i: usize| flat.as_ref().and_then(|f| f.states.get(i));
        let prefix = self.level_prefix();

        // Auto Live Link: when the playing state changes, scroll the least
        // amount that brings its node into view.
        match (&lp, self.auto_follow) {
            (Some(l), true) => {
                if self.last_live_state.as_deref() != Some(l.state_name.as_str()) {
                    self.last_live_state = Some(l.state_name.clone());
                    let m = machine_of(&self.ctrl, self.layer, &self.sub);
                    let node = (0..m.states.len()).find(|&i| under(&l.state_name, &prefix, &m.states[i].name)).map(Node::State);
                    if let Some(n) = node {
                        let scroll = cv::scroll_to_keep_visible(rect, self.node_rect(rect, n), cv::FRAME_BORDER);
                        self.pan += scroll / self.zoom;
                    }
                }
            }
            _ => self.last_live_state = None,
        }

        let hovered_node = pointer.filter(|p| rect.contains(*p)).and_then(|p| self.node_at(rect, p));
        let edges = self.edge_draws(rect);
        let gesture = self.node_drag.is_some() || self.box_sel.is_some();
        let hovered_edge = if hovered_node.is_none() && !gesture {
            pointer.and_then(|p| {
                edges
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (i, cv::dist_to_segment(p, e.a, e.b)))
                    .filter(|&(_, d)| d < 8.0)
                    .min_by(|x, y| x.1.total_cmp(&y.1))
                    .map(|(i, _)| i)
            })
        } else {
            None
        };
        let hovered_tr = hovered_edge.map(|i| self.pick_in_edge(&edges[i].group));

        // Transitions: one edge per node pair, three arrowheads when several
        // transitions share it, red / green for muted / soloed.
        for (ei, e) in edges.iter().enumerate() {
            let (a, b) = (e.a, e.b);
            let selected = matches!(self.sel, Sel::Transition(k) if e.group.transitions.contains(&k));
            let active = lp.as_ref().is_some_and(|l| {
                l.in_transition()
                    && l.transition
                        .and_then(|ft| flat.as_ref()?.transitions.get(ft))
                        .is_some_and(|ft| e.group.transitions.iter().any(|&k| self.transition_matches(k, ft, &prefix)))
            });
            let color = if selected || active {
                theme::accent()
            } else if hovered_edge == Some(ei) {
                theme::TEXT_SECONDARY
            } else {
                theme::TEXT_DIM
            };
            painter.line_segment([a, b], Stroke::new(if selected { 2.0_f32 } else { 1.5 }, color));
            if active && let Some(l) = &lp {
                let p = a + (b - a) * l.transition_progress();
                painter.line_segment([a, p], Stroke::new(3.0_f32, theme::accent()));
            }
            let machine = machine_of(&self.ctrl, self.layer, &self.sub);
            let flags: Vec<(bool, bool)> = e.group.transitions.iter().map(|&k| (machine.transitions[k].muted, machine.transitions[k].solo)).collect();
            let tints = cv::arrow_tints(&flags);
            for (&frac, tint) in cv::arrow_fractions(flags.len()).iter().zip(tints) {
                let c = match tint {
                    ArrowTint::Normal => color,
                    ArrowTint::Muted => theme::STATUS_ERROR,
                    ArrowTint::Solo => theme::STATUS_OK,
                };
                arrow_head_at(&painter, a, b, frac, c, self.zoom);
            }
        }

        // Entry -> default state.
        if let Some(d) = machine_of(&self.ctrl, self.layer, &self.sub).default_index() {
            let a = self.node_rect(rect, Node::Entry).center();
            let b = self.node_rect(rect, Node::State(d)).center();
            painter.line_segment([a, b], Stroke::new(1.5_f32, theme::with_alpha(theme::TYPE_LIGHT, 0.7)));
            arrow_head_at(&painter, a, b, 0.5, theme::with_alpha(theme::TYPE_LIGHT, 0.7), self.zoom);
        }

        // Nodes.
        let default = machine_of(&self.ctrl, self.layer, &self.sub).default_index();
        for node in self.nodes() {
            let r = self.node_rect(rect, node);
            if !rect.intersects(r) {
                continue;
            }
            let states = &machine_of(&self.ctrl, self.layer, &self.sub).states;
            let is_sub = matches!(node, Node::State(i) if states[i].is_sub_machine());
            let (label, tint) = match node {
                Node::Entry => ("Entry".to_owned(), theme::STATUS_OK),
                Node::Any => ("Any State".to_owned(), theme::TYPE_UI),
                Node::State(i) => (
                    states[i].name.clone(),
                    if default == Some(i) {
                        theme::TYPE_LIGHT
                    } else if is_sub {
                        theme::TYPE_UI
                    } else {
                        theme::TEXT_MUTED
                    },
                ),
            };
            let current = matches!(node, Node::State(i) if lp.as_ref().is_some_and(|l| under(&l.state_name, &prefix, &states[i].name)));
            let previous = matches!(node, Node::State(i) if lp.as_ref().is_some_and(|l| {
                l.previous.and_then(|(p, _)| flat_state(p)).is_some_and(|ps| under(&ps.name, &prefix, &states[i].name))
            }));
            let fill = if current {
                theme::accent_faint()
            } else if matches!(node, Node::State(_)) {
                theme::BG_BUTTON
            } else {
                theme::with_alpha(tint, 0.18)
            };
            let selected = self.selected_nodes().contains(&node);
            let stroke = if selected || current {
                Stroke::new(2.0_f32, theme::accent())
            } else if hovered_node == Some(node) {
                Stroke::new(1.0_f32, theme::TEXT_SECONDARY)
            } else if previous {
                Stroke::new(1.5_f32, theme::accent_soft())
            } else {
                Stroke::new(1.0_f32, theme::LINE_BUTTON)
            };
            if is_sub {
                // A stacked card: the node holds a machine of its own.
                let back = r.translate(Vec2::splat(4.0 * self.zoom));
                painter.rect(back, CornerRadius::same(theme::R_ROW), theme::BG_PANEL, Stroke::new(1.0_f32, theme::with_alpha(tint, 0.8)), StrokeKind::Inside);
            }
            // Opaque base first: Entry / Any State and the playing state use
            // translucent fills, and transition lines must stay under every node.
            painter.rect_filled(r, CornerRadius::same(theme::R_ROW), theme::BG_PANEL);
            painter.rect(r, CornerRadius::same(theme::R_ROW), fill, stroke, StrokeKind::Inside);
            if is_sub {
                painter.text(
                    r.right_center() - Vec2::new(10.0 * self.zoom, 0.0),
                    egui::Align2::RIGHT_CENTER,
                    ph::STACK,
                    egui::FontId::proportional((14.0 * self.zoom).clamp(7.0, 22.0)),
                    theme::with_alpha(tint, 0.9),
                );
            }
            if matches!(node, Node::State(_)) {
                painter.rect_filled(
                    Rect::from_min_size(r.min, Vec2::new(4.0 * self.zoom, r.height())),
                    CornerRadius { nw: theme::R_ROW, sw: theme::R_ROW, ne: 0, se: 0 },
                    tint,
                );
            }
            painter.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                &label,
                theme::semibold((theme::SIZE_BODY * self.zoom).clamp(8.0, 20.0)),
                theme::TEXT_STRONG,
            );
            if current && let Some(l) = &lp {
                let f = cv::node_progress(l.normalized_time, flat_state(l.state).is_some_and(|s| s.looping));
                let bar = Rect::from_min_size(Pos2::new(r.left() + 6.0, r.bottom() - 5.0), Vec2::new((r.width() - 12.0) * f, 2.0));
                painter.rect_filled(bar, CornerRadius::ZERO, theme::accent());
            }
        }

        // Interaction.
        let (shift, ctrl, alt) = ui.input(|i| (i.modifiers.shift, i.modifiers.command || i.modifiers.ctrl, i.modifiers.alt));
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            self.drag_cancelled = false;
            match self.node_at(rect, p) {
                Some(n) if shift || n == Node::Entry => self.linking = Some(n),
                Some(n) => {
                    let mut sel = self.selected_nodes();
                    if !sel.contains(&n) {
                        if ctrl {
                            sel.push(n);
                        } else {
                            sel = vec![n];
                        }
                        self.set_selected_nodes(sel, Some(n));
                    }
                    let starts = self.selected_nodes().into_iter().map(|m| (m, self.node_pos(m))).collect();
                    self.node_drag = Some(NodeDrag { starts, total: Vec2::ZERO });
                    self.dragging = Some(n);
                }
                None if !alt && self.linking.is_none() => {
                    self.box_sel = Some(BoxSel { from: p, prev: self.selected_nodes(), additive: ctrl || shift });
                }
                None => {}
            }
        }
        let primary_drag = response.dragged_by(egui::PointerButton::Primary) && !self.drag_cancelled;
        let pan_drag = response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
            || (primary_drag && alt && self.node_drag.is_none() && self.box_sel.is_none() && self.linking.is_none());
        if pan_drag {
            self.pan += response.drag_delta() / self.zoom;
        }
        if primary_drag {
            if let Some(nd) = &mut self.node_drag {
                nd.total += response.drag_delta() / self.zoom;
                let (starts, total) = (nd.starts.clone(), nd.total);
                let (snap, m) = (self.snap, machine_of_mut(&mut self.ctrl, self.layer, &self.sub));
                for (n, start) in starts {
                    let mut v = [start[0] + total.x, start[1] + total.y];
                    if snap {
                        v = cv::snap_pos(v, cv::GRID);
                    }
                    let slot = match n {
                        Node::State(i) => m.states.get_mut(i).map(|s| &mut s.position),
                        Node::Any => Some(&mut m.any_state_position),
                        Node::Entry => Some(&mut m.entry_position),
                    };
                    if let Some(slot) = slot {
                        *slot = v;
                        edited = true;
                    }
                }
            } else if let Some(bx) = &self.box_sel
                && let Some(p) = pointer
            {
                let (from, prev, additive) = (bx.from, bx.prev.clone(), bx.additive);
                let sel_rect = Rect::from_two_pos(from, p);
                let rects: Vec<(Node, Rect)> = self.nodes().into_iter().map(|n| (n, self.node_rect(rect, n))).collect();
                let hits = cv::nodes_in_rect(&rects, sel_rect);
                let now = cv::combine_selection(&prev, &hits, additive);
                self.set_selected_nodes(now, None);
                painter.rect(sel_rect, CornerRadius::same(theme::R_ROW), theme::accent_faint(), Stroke::new(1.0_f32, theme::accent_soft()), StrokeKind::Inside);
            }
        }
        if let Some(from) = self.linking
            && let Some(p) = pointer
        {
            let a = self.node_rect(rect, from).center();
            painter.line_segment([a, p], Stroke::new(2.0_f32, theme::accent()));
            arrow_head_at(&painter, a, p, 0.5, theme::accent(), self.zoom);
        }
        let released = response.drag_stopped_by(egui::PointerButton::Primary);
        let clicked = response.clicked_by(egui::PointerButton::Primary);
        if (released || clicked) && let Some(from) = self.linking {
            let target = response.interact_pointer_pos().or(pointer).and_then(|p| self.node_at(rect, p));
            match target {
                Some(Node::State(to)) if Some(Node::State(to)) != Some(from) || from == Node::Any => {
                    let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
                    let to_name = m.states[to].name.clone();
                    match from {
                        Node::Entry => m.default_state = to_name,
                        Node::Any => {
                            m.transitions.push(AnimTransition { from: None, to: to_name, ..Default::default() });
                            self.sel = Sel::Transition(m.transitions.len() - 1);
                        }
                        Node::State(f) => {
                            let from_name = m.states[f].name.clone();
                            // Unity's default: wait for the end of the source state.
                            m.transitions.push(AnimTransition {
                                from: Some(from_name),
                                to: to_name,
                                has_exit_time: true,
                                exit_time: 1.0,
                                ..Default::default()
                            });
                            self.sel = Sel::Transition(m.transitions.len() - 1);
                        }
                    }
                    edited = true;
                    self.linking = None;
                }
                Some(_) if released => self.linking = None,
                None => self.linking = None,
                _ => {}
            }
        } else if clicked {
            let additive = ctrl || shift;
            match (hovered_node, hovered_tr) {
                (Some(n), _) => {
                    let now = cv::click_selection(&self.selected_nodes(), n, additive);
                    self.set_selected_nodes(now, Some(n));
                }
                (None, Some(k)) => {
                    self.sel = Sel::Transition(k);
                    self.multi.clear();
                }
                _ if !additive => {
                    self.sel = Sel::None;
                    self.multi.clear();
                }
                _ => {}
            }
        }
        if released {
            self.dragging = None;
            self.node_drag = None;
            self.box_sel = None;
            self.drag_cancelled = false;
        }
        if response.double_clicked_by(egui::PointerButton::Primary)
            && self.linking.is_none()
            && let Some(Node::State(i)) = hovered_node
            && machine_of(&self.ctrl, self.layer, &self.sub).states.get(i).is_some_and(|s| s.is_sub_machine())
        {
            self.enter_sub(i);
        }
        if response.secondary_clicked() {
            self.linking = None;
            if let Some(p) = response.interact_pointer_pos() {
                let canvas = (p - rect.center()) / self.zoom - self.pan;
                self.context = Some((hovered_node, hovered_tr, Pos2::new(canvas.x, canvas.y)));
                if let Some(n) = hovered_node {
                    if !self.selected_nodes().contains(&n) {
                        self.set_selected_nodes(vec![n], Some(n));
                    }
                } else if let Some(k) = hovered_tr {
                    self.sel = Sel::Transition(k);
                }
            }
        }

        // Drop an animation clip from the Files panel: a new state playing it.
        if let Some(path) = response.dnd_hover_payload::<PathBuf>()
            && (is_clip_file(&path) || is_model_file(&path))
        {
            painter.rect_stroke(rect.shrink(2.0), CornerRadius::same(theme::R_ROW), Stroke::new(2.0_f32, theme::accent()), StrokeKind::Inside);
            if let Some(p) = pointer {
                let ghost = Rect::from_center_size(p, NODE_SIZE * self.zoom);
                painter.rect(ghost, CornerRadius::same(theme::R_ROW), theme::accent_faint(), Stroke::new(1.0_f32, theme::accent_soft()), StrokeKind::Inside);
            }
        }
        if let Some(path) = response.dnd_release_payload::<PathBuf>()
            && (is_clip_file(&path) || is_model_file(&path))
            && let Some(p) = pointer
        {
            let at = (p - rect.center()) / self.zoom - self.pan;
            let pos = if self.snap { cv::snap_pos([at.x, at.y], cv::GRID) } else { [at.x, at.y] };
            let rel = if path.is_absolute() { rel_path(root, &path) } else { path.to_string_lossy().replace('\\', "/") };
            // A clip makes one state; a model makes one per clip it carries
            // (read here, which can take a moment for a large file).
            let clips: Vec<(String, String)> = if is_clip_file(&path) {
                vec![(file_stem(&rel), rel)]
            } else {
                match zestengine_assets::load_model_with_meta(root.join(&rel)) {
                    // A model without a mesh is a clip library (Mixamo "Without Skin"): its
                    // clips are played on this avatar by bone name, so the state refers to the
                    // model's clip rather than to a clip of the avatar's own model.
                    // A file with one take (the usual download) names its state after the file.
                    Ok(model) if model.meshes.is_empty() => {
                        let one = model.animations.len() == 1;
                        model.animations.iter().map(|c| (if one { file_stem(&rel) } else { c.name.clone() }, zestengine_engine::clip_source::clip_ref(&rel, &c.name))).collect()
                    }
                    Ok(model) => model.animations.iter().map(|c| (c.name.clone(), c.name.clone())).collect(),
                    Err(e) => {
                        tracing::warn!("{rel}: cannot read clips: {e:#}");
                        Vec::new()
                    }
                }
            };
            let spots = cv::grid_positions(clips.len(), pos, 3, [NODE_SIZE.x + 20.0, NODE_SIZE.y + 24.0]);
            let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
            let mut made = Vec::new();
            for ((label, motion), spot) in clips.into_iter().zip(spots) {
                let name = m.unique_name(&label);
                m.states.push(AnimState { name: name.clone(), motion: Motion::Clip(motion), position: spot, ..Default::default() });
                if m.states.len() == 1 {
                    m.default_state = name;
                }
                made.push(Node::State(m.states.len() - 1));
            }
            if let Some(&last) = made.last() {
                self.set_selected_nodes(made, Some(last));
                edited = true;
            }
        }

        response.context_menu(|ui| {
            crate::widgets::menu_frame(ui);
            let Some((mut node, mut tr, at)) = self.context else {
                ui.close();
                return;
            };
            let m = machine_of(&self.ctrl, self.layer, &self.sub);
            if matches!(node, Some(Node::State(i)) if i >= m.states.len()) {
                node = None;
            }
            if tr.is_some_and(|k| k >= m.transitions.len()) {
                tr = None;
            }
            match (node, tr) {
                (Some(n @ Node::State(i)), _) => {
                    if m.states[i].is_sub_machine() && ui.button("Open Sub-State Machine").clicked() {
                        self.enter_sub(i);
                        ui.close();
                        return;
                    }
                    if ui.button("Make Transition").clicked() {
                        self.linking = Some(n);
                        ui.close();
                    }
                    if ui.button(if self.sub.is_empty() { "Set as Layer Default" } else { "Set as Default" }).clicked() {
                        let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
                        m.default_state = m.states[i].name.clone();
                        edited = true;
                        ui.close();
                    }
                    if ui.button("Duplicate").clicked() {
                        let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
                        let mut copy = m.states[i].clone();
                        copy.name = m.unique_name(&copy.name);
                        copy.position[0] += 24.0;
                        copy.position[1] += 56.0;
                        m.states.push(copy);
                        self.sel = Sel::Node(Node::State(m.states.len() - 1));
                        edited = true;
                        ui.close();
                    }
                    if ui.button("Frame Selected  (F)").clicked() {
                        self.frame_selected();
                        ui.close();
                    }
                    let count = self.selected_nodes().iter().filter(|n| matches!(n, Node::State(_))).count();
                    if ui.button(if count > 1 { format!("Delete {count} States") } else { "Delete".to_owned() }).clicked() {
                        edited |= self.delete_selected();
                        ui.close();
                    }
                }
                (Some(n @ Node::Any), _) | (Some(n @ Node::Entry), _) => {
                    let label = if n == Node::Any { "Make Transition" } else { "Set Default State" };
                    if ui.button(label).clicked() {
                        self.linking = Some(n);
                        ui.close();
                    }
                }
                (None, Some(k)) => {
                    if ui.button("Delete Transition").clicked() {
                        machine_of_mut(&mut self.ctrl, self.layer, &self.sub).transitions.remove(k);
                        self.sel = Sel::None;
                        edited = true;
                        ui.close();
                    }
                    if ui.button("Frame Selected  (F)").clicked() {
                        self.frame_selected();
                        ui.close();
                    }
                }
                (None, None) => {
                    if ui.button("Frame All  (A)").clicked() {
                        let all = self.nodes();
                        self.frame_nodes(&all, cv::ZOOM_MAX);
                        ui.close();
                    }
                    for (label, tree) in [("Create State", false), ("Create Blend Tree State", true)] {
                        if ui.button(label).clicked() {
                            let params = &self.ctrl.parameters;
                            let float = params.iter().find(|p| p.kind == AnimParamKind::Float).map(|p| p.name.clone()).unwrap_or_default();
                            let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
                            let name = m.unique_name(if tree { "Blend Tree" } else { "New State" });
                            let motion = if tree {
                                Motion::BlendTree(BlendTree { name: name.clone(), parameter: float, ..Default::default() })
                            } else {
                                Motion::Clip(String::new())
                            };
                            m.states.push(AnimState { name: name.clone(), motion, position: [at.x, at.y], ..Default::default() });
                            if m.states.len() == 1 {
                                m.default_state = name;
                            }
                            self.sel = Sel::Node(Node::State(m.states.len() - 1));
                            edited = true;
                            ui.close();
                        }
                    }
                    if ui.button("Create Sub-State Machine").clicked() {
                        let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
                        let name = m.unique_name("Sub-State Machine");
                        m.states.push(AnimState {
                            name: name.clone(),
                            position: [at.x, at.y],
                            sub_machine: Some(Box::default()),
                            ..Default::default()
                        });
                        if m.states.len() == 1 {
                            m.default_state = name;
                        }
                        self.sel = Sel::Node(Node::State(m.states.len() - 1));
                        edited = true;
                        ui.close();
                    }
                }
            }
        });

        // Keyboard: Delete removes the selection, Escape cancels the gesture
        // in progress (node drag, box select, linking), F / A frame.
        if response.hovered() || response.has_focus() {
            let (del, esc, key_f, key_a) = ui.input(|i| {
                let plain = !i.modifiers.command && !i.modifiers.ctrl && !i.modifiers.alt;
                (i.key_pressed(egui::Key::Delete), i.key_pressed(egui::Key::Escape), plain && i.key_pressed(egui::Key::F), plain && i.key_pressed(egui::Key::A))
            });
            let typing = ui.ctx().wants_keyboard_input();
            if esc && !typing {
                if let Some(nd) = self.node_drag.take() {
                    let m = machine_of_mut(&mut self.ctrl, self.layer, &self.sub);
                    for (n, start) in nd.starts {
                        let slot = match n {
                            Node::State(i) => m.states.get_mut(i).map(|s| &mut s.position),
                            Node::Any => Some(&mut m.any_state_position),
                            Node::Entry => Some(&mut m.entry_position),
                        };
                        if let Some(slot) = slot {
                            *slot = start;
                        }
                    }
                    self.dragging = None;
                    self.drag_cancelled = true;
                    edited = true;
                } else if let Some(bx) = self.box_sel.take() {
                    self.set_selected_nodes(bx.prev, None);
                    self.drag_cancelled = true;
                } else if self.linking.is_some() {
                    self.linking = None;
                } else if self.sub.pop().is_some() {
                    self.sel = Sel::None;
                    self.frame_all();
                }
            }
            if del && !typing {
                edited |= self.delete_selected();
            }
            if key_a && !typing {
                let all = self.nodes();
                self.frame_nodes(&all, cv::ZOOM_MAX);
            } else if key_f && !typing {
                self.frame_selected();
            }
        }

        let hint = if self.linking.is_some() {
            "Click a state to connect · Esc cancels"
        } else {
            "Right-click: create · Drag empty: box select · Shift-drag: transition · Middle / Alt-drag: pan · F / A: frame · Drop a .anim or model: new states · Esc: cancel / up"
        };
        painter.text(
            rect.left_bottom() + Vec2::new(10.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            hint,
            egui::FontId::proportional(theme::SIZE_META),
            theme::TEXT_DIM,
        );
        if self.linking.is_some() || self.dragging.is_some() || self.box_sel.is_some() {
            ui.ctx().request_repaint();
        }
        if lp.is_some() {
            ui.ctx().request_repaint();
        }
        if edited {
            self.changed(scene);
        }
    }
}

/// An edge ready to draw: its transitions and its screen endpoints.
struct EdgeDraw {
    group: cv::EdgeGroup,
    a: Pos2,
    b: Pos2,
}

fn kind_glyph(kind: AnimParamKind) -> &'static str {
    match kind {
        AnimParamKind::Float => "F",
        AnimParamKind::Int => "I",
        AnimParamKind::Bool => "B",
        AnimParamKind::Trigger => "T",
    }
}

/// Length of a motion in seconds when its clips are readable authored `.anim`
/// assets (a blend tree: the mean of its readable clips).
fn motion_length(scene: &mut LoadedScene, motion: &Motion) -> Option<f32> {
    fn clip_len(scene: &mut LoadedScene, name: &str) -> Option<f32> {
        if !zestengine_engine::anim_graph::is_clip_asset(name) {
            return None;
        }
        scene.anim_assets.ensure_clip(name);
        scene.anim_assets.clips.get(name).and_then(|c| c.as_ref()).map(|c| c.length)
    }
    match motion {
        Motion::Clip(name) => clip_len(scene, name),
        Motion::BlendTree(tree) => {
            let lens: Vec<Option<f32>> = tree.motions.iter().map(|m| clip_len(scene, &m.clip)).collect();
            cv::mean_known(lens)
        }
    }
}

/// Fill `r` with a left-to-right colour gradient.
fn gradient_rect(painter: &egui::Painter, r: Rect, left: Color32, right: Color32) {
    if r.width() <= 0.0 || r.height() <= 0.0 {
        return;
    }
    let mut mesh = egui::epaint::Mesh::default();
    mesh.colored_vertex(r.left_top(), left);
    mesh.colored_vertex(r.right_top(), right);
    mesh.colored_vertex(r.right_bottom(), right);
    mesh.colored_vertex(r.left_bottom(), left);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(Shape::mesh(mesh));
}

/// Small "NN%" readout next to a handle while dragging.
fn percent_badge(painter: &egui::Painter, at: Pos2, text: String) {
    let galley = painter.layout_no_wrap(text, theme::mono(theme::SIZE_TINY), theme::TEXT_STRONG);
    let r = Rect::from_center_size(at, galley.size() + Vec2::new(8.0, 4.0));
    painter.rect(r, CornerRadius::same(theme::R_SWATCH), theme::BG_APP, Stroke::new(1.0_f32, theme::LINE_BUTTON), StrokeKind::Inside);
    painter.galley(r.center() - galley.size() * 0.5, galley, theme::TEXT_STRONG);
}

/// The transition timeline: source and destination bars on a shared seconds
/// axis with the cross-fade between them. Drag the handles to edit; nothing is
/// written until the mouse is released, when the new `(exit_time, duration,
/// offset)` is returned (one undo step).
fn timeline_widget(
    ui: &mut egui::Ui,
    info: &TlInfo,
    tr: &AnimTransition,
    t: usize,
    drag: &mut Option<TlDrag>,
    shift: &mut f32,
) -> Option<(f32, f32, f32)> {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), cv::TL_HEIGHT), Sense::click_and_drag());
    if drag.as_ref().is_some_and(|d| d.transition != t) {
        *drag = None;
    }
    let stored = TimelineGeom::new(tr.exit_time, tr.duration, tr.offset, info.src_dur, info.dst_dur);
    let base = drag.map(|d| d.axis).unwrap_or_else(|| {
        let (lo, hi) = cv::timeline_extent(&stored);
        cv::fit_axis(lo, hi + (hi - lo) * 0.15, 0.0, rect.width(), 12.0)
    });
    let shifted = |shift: f32| cv::TimeAxis { x0: base.x0 + shift, ..base };
    let rel = |p: Pos2| Pos2::new(p.x - rect.left(), p.y - rect.top());

    // Input.
    let mut commit = None;
    if resp.drag_started_by(egui::PointerButton::Primary)
        && let Some(p) = resp.interact_pointer_pos()
    {
        let kind = cv::timeline_hit(&stored, &shifted(*shift), rel(p), tr.has_exit_time);
        *drag = Some(TlDrag { transition: t, kind, start: stored, cur: stored, axis: base, origin_x: p.x });
    }
    if resp.dragged_by(egui::PointerButton::Primary) {
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            *drag = None;
        } else if let Some(d) = drag.as_mut() {
            if d.kind == TimelineDrag::Pan {
                *shift += resp.drag_delta().x;
            } else if let Some(p) = ui.input(|i| i.pointer.latest_pos()) {
                d.cur = cv::apply_timeline_drag(d.kind, &d.start, (p.x - d.origin_x) / base.px_per_sec);
            }
        }
    }
    if resp.drag_stopped_by(egui::PointerButton::Primary)
        && let Some(d) = drag.take()
        && d.kind != TimelineDrag::Pan
    {
        commit = Some(d.cur.values());
    }
    let axis = shifted(*shift);
    let g = drag.map(|d| d.cur).unwrap_or(stored);
    let dragging = drag.map(|d| d.kind);
    if let Some(p) = resp.hover_pos().filter(|_| dragging.is_none()) {
        let icon = match cv::timeline_hit(&g, &axis, rel(p), tr.has_exit_time) {
            TimelineDrag::ExitTime | TimelineDrag::Duration | TimelineDrag::Region => egui::CursorIcon::ResizeHorizontal,
            TimelineDrag::Offset => egui::CursorIcon::Grab,
            TimelineDrag::Pan => egui::CursorIcon::Default,
        };
        ui.ctx().set_cursor_icon(icon);
    }

    // Paint.
    let painter = ui.painter_at(rect);
    let x = |time: f32| rect.left() + axis.x(time);
    painter.rect(rect, CornerRadius::same(theme::R_FIELD), theme::BG_FIELD, Stroke::new(1.0_f32, theme::LINE_FIELD), StrokeKind::Inside);
    painter.rect_filled(Rect::from_min_size(rect.min, Vec2::new(rect.width(), cv::TL_RULER_H)), CornerRadius::ZERO, theme::BG_BUTTON);
    let step = cv::tick_step(60.0 / axis.px_per_sec);
    let first = (axis.t(0.0) / step).floor() as i32;
    let last = (axis.t(rect.width()) / step).ceil() as i32;
    for k in first..=last.min(first + 400) {
        let time = k as f32 * step;
        let px = x(time);
        painter.line_segment([Pos2::new(px, rect.top() + cv::TL_RULER_H), Pos2::new(px, rect.bottom())], Stroke::new(1.0_f32, theme::LINE_PANEL));
        painter.text(Pos2::new(px + 3.0, rect.top() + 2.0), egui::Align2::LEFT_TOP, format!("{time:.2}s"), theme::mono(theme::SIZE_TINY), theme::TEXT_DIM);
    }

    // The cross-fade column and its ruler head.
    let fade = Rect::from_x_y_ranges(x(g.start)..=x(g.stop), rect.top()..=rect.bottom());
    painter.rect_filled(fade, CornerRadius::ZERO, theme::accent_faint());
    let head_color = if tr.has_exit_time { theme::accent_soft() } else { theme::with_alpha(theme::LINE_BUTTON, 0.6) };
    painter.rect_filled(Rect::from_x_y_ranges(fade.x_range(), rect.top()..=rect.top() + cv::TL_RULER_H), CornerRadius::ZERO, head_color);
    for px in [fade.left(), fade.right()] {
        painter.line_segment([Pos2::new(px, rect.top()), Pos2::new(px, rect.bottom())], Stroke::new(1.0_f32, theme::accent_soft()));
    }

    // Bars. Source: full before the fade, fading out across it. Destination
    // is the mirror. Looping states repeat to cover the fade.
    let bars = [
        (cv::TL_SRC_Y, theme::TYPE_MESH, &info.src, cv::bar_spans(0.0, info.src_dur, info.src_loop, g.stop), true),
        (cv::TL_DST_Y, theme::TYPE_UI, &info.dst, cv::bar_spans(g.dst_start, info.dst_dur, info.dst_loop, g.stop), false),
    ];
    for (y, color, name, spans, is_src) in bars {
        let (on, off) = (theme::with_alpha(color, 0.5), theme::with_alpha(color, 0.12));
        let (before, after) = if is_src { (on, off) } else { (off, on) };
        for (i, &(s0, s1)) in spans.iter().enumerate() {
            let bar = Rect::from_x_y_ranges(x(s0)..=x(s1), rect.top() + y..=rect.top() + y + cv::TL_BAR_H);
            painter.rect_filled(bar, CornerRadius::ZERO, off);
            let part = |a: f32, b: f32| Rect::from_x_y_ranges(x(a.max(s0))..=x(b.min(s1)), bar.y_range());
            gradient_rect(&painter, part(s0, g.start), before, before);
            gradient_rect(&painter, part(g.start, g.stop), before, after);
            gradient_rect(&painter, part(g.stop, s1), after, after);
            painter.rect_stroke(bar, CornerRadius::ZERO, Stroke::new(1.0_f32, theme::with_alpha(color, 0.7)), StrokeKind::Inside);
            if i == 0 && dragging.is_none() {
                painter.with_clip_rect(bar.intersect(rect)).text(
                    bar.left_center() + Vec2::new(6.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    name,
                    theme::semibold(theme::SIZE_BODY),
                    theme::TEXT_STRONG,
                );
            }
        }
    }

    // Handles.
    let hy = rect.top() + cv::TL_RULER_H + cv::TL_HANDLE_H * 0.5;
    for (px, enabled) in [(x(g.start), tr.has_exit_time), (x(g.stop), true)] {
        let h = Rect::from_center_size(Pos2::new(px, hy), Vec2::new(cv::TL_HANDLE_W, cv::TL_HANDLE_H));
        painter.rect(
            h,
            CornerRadius::same(theme::R_SWATCH),
            if enabled { theme::accent() } else { theme::BG_BUTTON },
            Stroke::new(1.0_f32, if enabled { theme::accent_soft() } else { theme::LINE_BUTTON }),
            StrokeKind::Inside,
        );
    }

    // Percent badges while dragging.
    if let Some(kind) = dragging {
        let at = match kind {
            TimelineDrag::Duration => Some(g.stop),
            TimelineDrag::ExitTime | TimelineDrag::Offset | TimelineDrag::Region => Some(g.start),
            TimelineDrag::Pan => None,
        };
        if let Some(time) = at {
            percent_badge(&painter, Pos2::new(x(time), rect.top() + cv::TL_SRC_Y - 10.0), format!("{:.0}%", g.src_percent(time)));
            percent_badge(&painter, Pos2::new(x(time), rect.top() + cv::TL_DST_Y + cv::TL_BAR_H + 10.0), format!("{:.0}%", g.dst_percent(time)));
        }
        ui.ctx().request_repaint();
    }
    commit
}

/// A model file whose clips can become states.
fn is_model_file(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| ["fbx", "glb", "gltf", "vrm"].iter().any(|m| e.eq_ignore_ascii_case(m)))
}

fn is_clip_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case(zestengine_assets::ANIM_CLIP_EXTENSION))
}

/// Triangle on the segment `a..b` at fraction `frac`, pointing from `a` to `b`.
fn arrow_head_at(painter: &egui::Painter, a: Pos2, b: Pos2, frac: f32, color: Color32, zoom: f32) {
    let dir = (b - a).normalized();
    if !dir.x.is_finite() {
        return;
    }
    let mid = a + (b - a) * frac;
    let s = 7.0 * zoom.clamp(0.6, 1.6);
    let side = Vec2::new(-dir.y, dir.x);
    let tip = mid + dir * s;
    painter.add(Shape::convex_polygon(vec![tip, mid - dir * s * 0.6 + side * s, mid - dir * s * 0.6 - side * s], color, Stroke::NONE));
}

/// Clip picker: known clips in a menu plus a free text field (any imported
/// clip name or a `.anim` path).
fn clip_combo(ui: &mut egui::Ui, id: egui::Id, value: &mut String, choices: &[String]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        // Fit the row: a wider one makes the resizable side panel grow to fit
        // it, every frame. Clip names come from the object's imported model
        // (and `.anim` files), so they are picked, never typed.
        let w = (ui.available_width() - ui.spacing().button_padding.x * 2.0 - 18.0).max(60.0);
        let missing = !value.is_empty() && !choices.contains(value);
        let shown = if value.is_empty() {
            egui::RichText::new("None").color(theme::TEXT_MUTED)
        } else if missing {
            egui::RichText::new(format!("{value} (missing)")).color(theme::STATUS_WARN_FG)
        } else {
            egui::RichText::new(value.as_str())
        };
        let combo = egui::ComboBox::from_id_salt(id).width(w).selected_text(shown).show_ui(ui, |ui| {
            changed |= ui.selectable_value(value, String::new(), "None").changed();
            if choices.is_empty() {
                ui.label(egui::RichText::new("No clips found").color(theme::TEXT_MUTED));
            }
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                for c in choices {
                    changed |= ui.selectable_value(value, c.clone(), c).changed();
                }
            });
        });
        if missing {
            combo.response.on_hover_text(
                "No clip with this name on the selected object's model or in the project's .anim files. \
                 Select an object that has this controller's animations, or pick another clip.",
            );
        }
        // Drop a .anim from the Files panel.
        let dragging: Option<String> = ui.data(|d| d.get_temp(egui::Id::new(crate::DRAG_FILE_KEY)));
        if let Some(p) = dragging
            && p.to_ascii_lowercase().ends_with(".anim")
            && ui.rect_contains_pointer(ui.min_rect())
            && ui.input(|i| i.pointer.any_released())
        {
            *value = p;
            changed = true;
        }
    });
    changed
}

fn param_combo(
    ui: &mut egui::Ui,
    id: egui::Id,
    value: &mut String,
    params: &[AnimParameter],
    allow_none: bool,
    filter: impl Fn(AnimParamKind) -> bool,
) -> bool {
    let mut changed = false;
    let shown = if value.is_empty() { "None".to_owned() } else { value.clone() };
    egui::ComboBox::from_id_salt(id).width(110.0).selected_text(shown).show_ui(ui, |ui| {
        if allow_none {
            changed |= ui.selectable_value(value, String::new(), "None").changed();
        }
        for p in params.iter().filter(|p| filter(p.kind)) {
            changed |= ui.selectable_value(value, p.name.clone(), &p.name).changed();
        }
    });
    changed
}

/// What the blend tree editor reads and writes besides the tree itself.
struct BtEnv<'a> {
    params: &'a [AnimParameter],
    choices: &'a [String],
    /// Current parameter values: the running Animator's, else the editor's
    /// preview values.
    values: &'a std::collections::HashMap<String, f32>,
    /// Parameter writes from the scrub handles; the caller applies them.
    scrub: Vec<(String, f32)>,
    /// Root motion of the tree's clips and their lengths, for Compute
    /// Thresholds and Adjust Time Scale (clips that could not be read are absent).
    motions: &'a std::collections::HashMap<String, ClipMotion>,
    lengths: &'a std::collections::HashMap<String, f32>,
}

impl BtEnv<'_> {
    fn value(&self, name: &str) -> f32 {
        match self.scrub.iter().find(|(n, _)| n == name) {
            Some((_, v)) => *v,
            None => self.values.get(name).copied().unwrap_or(0.0),
        }
    }

    fn set(&mut self, name: &str, v: f32) {
        if name.is_empty() {
            return;
        }
        match self.scrub.iter_mut().find(|(n, _)| n == name) {
            Some(e) => e.1 = v,
            None => self.scrub.push((name.to_owned(), v)),
        }
    }
}

/// Nested trees deeper than this are not drawn (the runtime stops at 16).
const BT_UI_DEPTH: usize = 6;
/// Marker for "scrubbing the parameter" in a graph's drag state.
const SCRUB: usize = usize::MAX;

fn first_float(params: &[AnimParameter], not: &str) -> String {
    let float = |p: &&AnimParameter| p.kind == AnimParamKind::Float;
    params.iter().filter(float).find(|p| p.name != not).or_else(|| params.iter().find(float)).map(|p| p.name.clone()).unwrap_or_default()
}

fn set_kind(tree: &mut BlendTree, kind: BlendTreeKind, params: &[AnimParameter]) {
    tree.kind = kind;
    // A 2D kind with no Y parameter reads back as a 1D tree.
    if kind.is_2d() && tree.parameter_y.is_empty() {
        tree.parameter_y = first_float(params, &tree.parameter);
    }
}

fn new_child_tree(params: &[AnimParameter]) -> BlendTree {
    BlendTree { name: "Blend Tree".into(), kind: BlendTreeKind::Simple1D, parameter: first_float(params, ""), ..Default::default() }
}

fn blend_tree_ui(ui: &mut egui::Ui, tree: &mut BlendTree, env: &mut BtEnv, id: egui::Id, depth: usize) -> bool {
    let mut edited = false;
    let kind = tree.effective_kind();
    let floats = |k: AnimParamKind| matches!(k, AnimParamKind::Float | AnimParamKind::Int);
    widgets::property_row(ui, "Blend type", |ui| {
        egui::ComboBox::from_id_salt(id.with("kind")).width(150.0).selected_text(kind.label()).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for k in BlendTreeKind::ALL {
                if ui.selectable_label(kind == k, k.label()).clicked() && kind != k {
                    set_kind(tree, k, env.params);
                    edited = true;
                }
            }
        });
    });
    let kind = tree.effective_kind();
    if kind != BlendTreeKind::Direct {
        widgets::property_row(ui, if kind.is_2d() { "Parameter X" } else { "Parameter" }, |ui| {
            edited |= param_combo(ui, id.with("p"), &mut tree.parameter, env.params, false, floats);
        });
    }
    if kind.is_2d() {
        widgets::property_row(ui, "Parameter Y", |ui| {
            edited |= param_combo(ui, id.with("py"), &mut tree.parameter_y, env.params, false, floats);
        });
    } else if kind == BlendTreeKind::Direct {
        widgets::property_row(ui, "Normalize", |ui| {
                let r = ui.checkbox(&mut tree.normalize_direct, "").on_hover_text("Divide the weights by their sum when it is above 1, so the blend never over-weights.");
                edited |= r.changed();
            });
    }

    let weights = tree.weights_for(&|name| env.value(name));
    let sel_id = id.with("sel");
    let n = tree.motions.len();
    let mut sel: Option<usize> = ui.data(|d| d.get_temp(sel_id)).filter(|&k| k < n);

    if kind == BlendTreeKind::Simple1D && n >= 2 {
        ui.add_space(4.0);
        edited |= graph_1d(ui, id, tree, env, &weights, &mut sel);
    } else if kind.is_2d() && n >= 2 {
        ui.add_space(4.0);
        edited |= graph_2d(ui, id, tree, env, &weights, &mut sel);
    }
    if kind != BlendTreeKind::Direct && n >= 1 {
        widgets::property_row(ui, "Preview", |ui| {
            let mut x = env.value(&tree.parameter);
            if ui.add(egui::DragValue::new(&mut x).speed(0.02).max_decimals(2)).changed() {
                env.set(&tree.parameter, x);
            }
            if kind.is_2d() {
                let mut y = env.value(&tree.parameter_y);
                if ui.add(egui::DragValue::new(&mut y).speed(0.02).max_decimals(2)).changed() {
                    env.set(&tree.parameter_y, y);
                }
            }
        });
    }
    ui.add_space(4.0);

    // Children.
    let mut remove = None;
    let mut reorder: Option<(usize, usize)> = None;
    let mut duplicate = None;
    for k in 0..n {
        let selected = sel == Some(k);
        let wt = weights.get(k).copied().unwrap_or(0.0);
        let fill = if selected { theme::BG_SEGMENT_ACTIVE } else { theme::BG_APP };
        egui::Frame::new().fill(fill).corner_radius(egui::CornerRadius::same(theme::R_ROW)).inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.push_id(id.with(("child", k)), |ui| {
                let m = &mut tree.motions[k];
                ui.horizontal(|ui| {
                    let tab = ui.selectable_label(selected, format!("{}", k + 1)).on_hover_text("Select. Right-click to reorder or duplicate.");
                    if tab.clicked() {
                        sel = Some(k);
                    }
                    egui::Popup::context_menu(&tab).show(|ui| {
                        widgets::menu_frame(ui);
                        if k > 0 && widgets::menu_item(ui, None, "Move up", None).clicked() {
                            reorder = Some((k, k - 1));
                            ui.close();
                        }
                        if k + 1 < n && widgets::menu_item(ui, None, "Move down", None).clicked() {
                            reorder = Some((k, k + 1));
                            ui.close();
                        }
                        if widgets::menu_item(ui, None, "Duplicate", None).clicked() {
                            duplicate = Some(k);
                            ui.close();
                        }
                        if widgets::menu_item(ui, None, "Remove", None).clicked() {
                            remove = Some(k);
                            ui.close();
                        }
                    });
                    let w = (ui.available_width() - 74.0).max(60.0);
                    ui.allocate_ui(Vec2::new(w, theme::H_FIELD), |ui| match m.tree.as_deref_mut() {
                        Some(t) => {
                            if let Some(new) = commit_text(ui, id.with(("tname", k)), &t.name, ui.available_width()) {
                                t.name = new;
                                edited = true;
                            }
                        }
                        None => edited |= clip_combo(ui, id.with(("clip", k)), &mut m.clip, env.choices),
                    });
                    ui.label(egui::RichText::new(format!("{:>3.0}%", wt * 100.0)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
                    if widgets::icon_button(ui, Icon::Close, "Remove motion", false).clicked() {
                        remove = Some(k);
                    }
                });
                // The weight bar.
                let (bar, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 2.0), Sense::hover());
                ui.painter().rect_filled(bar, 1.0, theme::LINE_PANEL);
                let mut fill = bar;
                fill.set_width(bar.width() * wt.clamp(0.0, 1.0));
                ui.painter().rect_filled(fill, 1.0, theme::STATUS_INFO);
                ui.horizontal(|ui| match kind {
                    BlendTreeKind::Simple1D => {
                        ui.label(egui::RichText::new("Threshold").size(theme::SIZE_META).color(theme::TEXT_DIM));
                        edited |= ui.add(egui::DragValue::new(&mut m.position[0]).speed(0.05).max_decimals(2)).changed();
                    }
                    BlendTreeKind::Direct => {
                        ui.label(egui::RichText::new("Parameter").size(theme::SIZE_META).color(theme::TEXT_DIM));
                        edited |= param_combo(ui, id.with(("dp", k)), &mut m.direct_parameter, env.params, true, floats);
                        let mut v = env.value(&m.direct_parameter);
                        if widgets::slider(ui, egui::Slider::new(&mut v, 0.0..=1.0).show_value(false)).changed() {
                            env.set(&m.direct_parameter, v);
                        }
                    }
                    _ => {
                        ui.label(egui::RichText::new("X").size(theme::SIZE_META).color(theme::TEXT_DIM));
                        edited |= ui.add(egui::DragValue::new(&mut m.position[0]).speed(0.05).max_decimals(2)).changed();
                        ui.label(egui::RichText::new("Y").size(theme::SIZE_META).color(theme::TEXT_DIM));
                        edited |= ui.add(egui::DragValue::new(&mut m.position[1]).speed(0.05).max_decimals(2)).changed();
                    }
                });
                if selected {
                    ui.add_space(2.0);
                    let mut nested = m.tree.is_some();
                    widgets::property_row(ui, "Motion", |ui| {
                        if widgets::segmented(ui, &mut nested, &[(false, "Clip"), (true, "Blend Tree")]).changed() {
                            m.tree = nested.then(|| Box::new(new_child_tree(env.params)));
                            edited = true;
                        }
                    });
                    widgets::property_row(ui, "Time scale", |ui| {
                        edited |= ui.add(egui::DragValue::new(&mut m.time_scale).speed(0.01).range(-10.0..=10.0).max_decimals(2)).changed();
                    });
                    widgets::property_row(ui, "Cycle offset", |ui| {
                        edited |= ui.add(egui::DragValue::new(&mut m.cycle_offset).speed(0.01).range(-1.0..=1.0).max_decimals(2)).changed();
                    });
                    widgets::property_row(ui, "Mirror", |ui| {
                        edited |= ui.checkbox(&mut m.mirror, "").changed();
                        ui.label(egui::RichText::new("humanoid rigs only").size(theme::SIZE_META).color(theme::TEXT_DIM))
                            .on_hover_text("Plays the child mirrored left to right (swaps left and right bones, reflects the pose and root motion). Ignored on rigs without a humanoid avatar.");
                    });
                    if m.tree.is_none() && m.clip.is_empty() {
                        ui.label(egui::RichText::new("This child has no motion assigned.").size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
                    }
                }
                if let Some(t) = m.tree.as_deref_mut() {
                    if depth + 1 >= BT_UI_DEPTH {
                        ui.label(egui::RichText::new("Nested too deep to edit here.").size(theme::SIZE_META).color(theme::TEXT_DIM));
                    } else {
                        egui::CollapsingHeader::new(egui::RichText::new("Nested tree").size(theme::SIZE_META).color(theme::TEXT_MUTED))
                            .id_salt(id.with(("nest", k)))
                            .show(ui, |ui| edited |= blend_tree_ui(ui, t, env, id.with(("tree", k)), depth + 1));
                    }
                }
            });
        });
        ui.add_space(3.0);
    }
    if let Some((from, to)) = reorder {
        cv::move_item(&mut tree.motions, from, to);
        sel = Some(to);
        edited = true;
    }
    if let Some(k) = duplicate {
        let copy = tree.motions[k].clone();
        tree.motions.insert(k + 1, copy);
        sel = Some(k + 1);
        edited = true;
    }
    if let Some(k) = remove {
        tree.motions.remove(k);
        sel = None;
        edited = true;
    }

    // Add.
    if widgets::dashed_button(ui, Some(Icon::Plus), "Add motion").clicked() {
        let next = tree.motions.iter().map(|m| m.position[0]).fold(f32::NEG_INFINITY, f32::max);
        let next = if next.is_finite() { next + 1.0 } else { 0.0 };
        let direct = if kind == BlendTreeKind::Direct { first_float(env.params, "") } else { String::new() };
        tree.motions.push(BlendMotion { position: [next, 0.0], direct_parameter: direct, ..Default::default() });
        sel = Some(tree.motions.len() - 1);
        edited = true;
    }
    if depth + 1 < BT_UI_DEPTH && widgets::secondary_button(ui, Some(Icon::Plus), "Add blend tree").clicked() {
        let m = BlendMotion { tree: Some(Box::new(new_child_tree(env.params))), ..Default::default() };
        tree.motions.push(m);
        sel = Some(tree.motions.len() - 1);
        edited = true;
    }

    // Compute and adjust.
    if kind != BlendTreeKind::Direct && !tree.motions.is_empty() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let compute = widgets::secondary_button(ui, None, if kind.is_2d() { "Compute Positions" } else { "Compute Thresholds" });
            egui::Popup::menu(&compute).show(|ui| {
                widgets::menu_frame(ui);
                edited |= compute_menu(ui, tree, kind, env);
            });
            if tree.motions.iter().all(|m| m.tree.is_none()) {
                let adjust = widgets::secondary_button(ui, None, "Adjust Time Scale");
                egui::Popup::menu(&adjust).show(|ui| {
                    widgets::menu_frame(ui);
                    let known = tree.motions.iter().any(|m| env.lengths.contains_key(&m.clip));
                    let homogeneous = ui
                        .add_enabled_ui(known, |ui| widgets::menu_item(ui, None, "Homogeneous Speed", None))
                        .inner
                        .on_hover_text("Scales each clip so all of them take the average clip length.")
                        .on_disabled_hover_text("No clip length could be read.");
                    if homogeneous.clicked() {
                        edited |= cm::apply_homogeneous(tree, env.lengths);
                        ui.close();
                    }
                    if widgets::menu_item(ui, None, "Reset Time Scale", None).clicked() {
                        for m in &mut tree.motions {
                            m.time_scale = 1.0;
                        }
                        edited = true;
                        ui.close();
                    }
                });
            }
        });
    }
    ui.data_mut(|d| d.insert_temp(sel_id, sel));
    edited
}

/// Compute Thresholds / Positions: the even spread or ring, then the sources
/// read from the clips' root motion.
fn compute_menu(ui: &mut egui::Ui, tree: &mut BlendTree, kind: BlendTreeKind, env: &BtEnv) -> bool {
    let n = tree.motions.len();
    let mut edited = false;
    if kind == BlendTreeKind::Simple1D {
        if widgets::menu_item(ui, None, "Spread evenly", None).clicked() {
            let th: Vec<f32> = tree.motions.iter().map(|m| m.position[0]).collect();
            let (mut lo, mut hi) = th.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
            if hi - lo < 1e-4 {
                (lo, hi) = (0.0, (n as f32 - 1.0).max(1.0));
            }
            tree.motions.sort_by(|a, b| a.position[0].total_cmp(&b.position[0]));
            for (m, v) in tree.motions.iter_mut().zip(cv::spread_thresholds(n, lo, hi)) {
                m.position[0] = v;
            }
            edited = true;
            ui.close();
        }
    } else {
        let reach = tree.motions.iter().map(|m| m.position[0].hypot(m.position[1])).fold(0.0_f32, f32::max);
        let radius = if reach > 1e-3 { reach } else { 1.0 };
        for (label, centre) in [("Ring around the origin", false), ("Ring with the first at the origin", true)] {
            if widgets::menu_item(ui, None, label, None).clicked() {
                for (m, p) in tree.motions.iter_mut().zip(cv::ring_positions(n, radius, centre)) {
                    m.position = p;
                }
                edited = true;
                ui.close();
            }
        }
    }
    widgets::menu_separator(ui);
    let sources: &[cm::ComputeFrom] = if kind.is_2d() { &cm::ComputeFrom::TWO_D } else { &cm::ComputeFrom::ONE_D };
    let have = tree.motions.iter().any(|m| env.motions.contains_key(&m.clip));
    for &from in sources {
        let item = ui.add_enabled_ui(have, |ui| widgets::menu_item(ui, None, from.label(), None)).inner;
        let item = item.on_disabled_hover_text("None of the clips has root motion to measure (an imported clip needs a moving root or Hips).");
        if item.clicked() {
            edited |= cm::apply_compute(tree, env.motions, from);
            ui.close();
        }
    }
    edited
}

/// Root motion and lengths of the clips `tree` plays, refreshed about once a
/// second (and when the tree's clips change) so the menus can read them.
fn tree_clip_data(
    ui: &egui::Ui,
    scene: &mut LoadedScene,
    tree: &BlendTree,
) -> (std::collections::HashMap<String, ClipMotion>, std::collections::HashMap<String, f32>) {
    type Data = (Vec<String>, f64, std::collections::HashMap<String, ClipMotion>, std::collections::HashMap<String, f32>);
    let mut names = Vec::new();
    cm::clip_names(tree, &mut names);
    let id = egui::Id::new("zestengine-tree-clip-data");
    let now = ui.input(|i| i.time);
    if let Some((n, at, m, l)) = ui.data_mut(|d| d.get_temp::<Data>(id))
        && n == names
        && now - at < 1.0
    {
        return (m, l);
    }
    let mut motions = std::collections::HashMap::new();
    let mut lengths = std::collections::HashMap::new();
    for n in &names {
        if let Some(l) = scene.clip_length(n) {
            lengths.insert(n.clone(), l);
        }
        if let Some(m) = scene.clip_motion(n) {
            motions.insert(n.clone(), m);
        }
    }
    ui.data_mut(|d| d.insert_temp::<Data>(id, (names, now, motions.clone(), lengths.clone())));
    (motions, lengths)
}

fn graph_1d(ui: &mut egui::Ui, id: egui::Id, tree: &mut BlendTree, env: &mut BtEnv, weights: &[f32], sel: &mut Option<usize>) -> bool {
    let mut edited = false;
    let th: Vec<f32> = tree.motions.iter().map(|m| m.position[0]).collect();
    // The range is frozen while dragging, or moving an end threshold would
    // rescale the strip under the pointer.
    let range_id = id.with("range1d");
    let frozen: Option<(f32, f32)> = ui.data(|d| d.get_temp(range_id));
    let (lo, hi) = frozen.unwrap_or_else(|| {
        let (lo, hi) = cv::range_1d(&th);
        let pad = (hi - lo) * 0.1;
        (lo - pad, hi + pad)
    });
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 48.0), Sense::click_and_drag());
    let inner = rect.shrink2(Vec2::new(10.0, 6.0));
    let per_unit = inner.width() / (hi - lo);
    let x_of = |v: f32| inner.left() + (v - lo) * per_unit;
    let v_of = |x: f32| lo + (x - inner.left()) / per_unit;
    let painter = ui.painter_at(rect);
    painter.rect(rect, egui::CornerRadius::same(theme::R_FIELD), theme::BG_FIELD, Stroke::new(1.0_f32, theme::LINE_FIELD), StrokeKind::Inside);

    let mut order: Vec<usize> = (0..th.len()).collect();
    order.sort_by(|&a, &b| th[a].total_cmp(&th[b]));
    let sorted: Vec<f32> = order.iter().map(|&k| th[k]).collect();
    for (i, &k) in order.iter().enumerate() {
        let (l, pk, r) = cv::triangle_1d(&sorted, i);
        let wt = weights.get(k).copied().unwrap_or(0.0);
        let a = (28.0 + 120.0 * wt.sqrt()) as u8;
        let tint = theme::STATUS_INFO;
        let pts = vec![Pos2::new(x_of(l), inner.bottom()), Pos2::new(x_of(pk), inner.top()), Pos2::new(x_of(r), inner.bottom())];
        painter.add(Shape::convex_polygon(pts.clone(), Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), a), Stroke::NONE));
        let on = *sel == Some(k);
        let line = Color32::from_white_alpha(if on { 150 } else { 100 });
        painter.add(Shape::closed_line(pts, Stroke::new(1.0_f32, line)));
        if on {
            painter.vline(x_of(pk), inner.y_range(), Stroke::new(1.0_f32, line));
        }
    }
    let px = env.value(&tree.parameter);
    let m = Rect::from_center_size(Pos2::new(x_of(px.clamp(lo, hi)), rect.center().y), Vec2::new(9.0, rect.height() - 4.0));
    painter.rect(m, egui::CornerRadius::same(2), theme::ACCENT_DEFAULT, Stroke::new(1.0_f32, theme::TEXT_ON_ACCENT), StrokeKind::Inside);

    let drag_id = id.with("drag1d");
    if (resp.drag_started() || resp.clicked())
        && let Some(pos) = resp.interact_pointer_pos()
    {
        let peak = order.iter().map(|&k| (k, (x_of(th[k]) - pos.x).abs())).min_by(|a, b| a.1.total_cmp(&b.1)).filter(|h| h.1 <= 7.0);
        let marker = (x_of(px.clamp(lo, hi)) - pos.x).abs();
        let target = match peak {
            Some((k, d)) if d <= marker || marker > 6.0 => k,
            _ => SCRUB,
        };
        if target != SCRUB {
            *sel = Some(target);
        }
        ui.data_mut(|d| {
            d.insert_temp(drag_id, target);
            d.insert_temp(range_id, (lo, hi));
        });
    }
    if (resp.dragged() || resp.clicked())
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some(target) = ui.data(|d| d.get_temp::<usize>(drag_id))
    {
        let raw = (v_of(pos.x) * 100.0).round() / 100.0;
        if target == SCRUB {
            env.set(&tree.parameter, raw.clamp(lo, hi));
        } else if target < tree.motions.len() {
            let others = th.iter().enumerate().filter(|&(i, _)| i != target).map(|(_, &v)| v);
            let v = cv::snap_threshold(raw, others, 3.0 / per_unit).clamp(-1e4, 1e4);
            tree.motions[target].position[0] = v;
            let now: Vec<f32> = tree.motions.iter().map(|m| m.position[0]).collect();
            let slot = cv::sorted_slot(&now, target);
            cv::move_item(&mut tree.motions, target, slot);
            *sel = Some(slot);
            ui.data_mut(|d| d.insert_temp(drag_id, slot));
            edited = true;
        }
    }
    if resp.drag_stopped() || (!resp.dragged() && !resp.is_pointer_button_down_on()) {
        ui.data_mut(|d| d.remove::<(f32, f32)>(range_id));
    }
    edited
}

fn graph_2d(ui: &mut egui::Ui, id: egui::Id, tree: &mut BlendTree, env: &mut BtEnv, weights: &[f32], sel: &mut Option<usize>) -> bool {
    let mut edited = false;
    let kind = tree.effective_kind();
    let pts: Vec<[f32; 2]> = tree.motions.iter().map(|m| m.position).collect();
    let view_id = id.with("view2d");
    let frozen: Option<([f32; 2], f32)> = ui.data(|d| d.get_temp(view_id));
    let directional = kind != BlendTreeKind::FreeformCartesian2D;
    let (centre, half) = frozen.unwrap_or_else(|| cv::view_2d(directional, &pts));
    let side = ui.available_width().min(235.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect(rect, egui::CornerRadius::same(theme::R_FIELD), theme::BG_FIELD, Stroke::new(1.0_f32, theme::LINE_FIELD), StrokeKind::Inside);
    let drag_id = id.with("drag2d");
    let target: Option<usize> = ui.data(|d| d.get_temp(drag_id));
    let dragging_dot = resp.dragged() && target.is_some_and(|t| t != SCRUB);

    // Weight field behind the dots: how mixed the blend is, or one child's
    // weight while its dot is dragged.
    let res = 24;
    let only = if dragging_dot { target } else { None };
    let field = cv::weight_field(res, centre, half, |q| {
        let w = tree.weights(q);
        match only {
            Some(k) => w.get(k).copied().unwrap_or(0.0),
            None => 1.0 - w.iter().copied().fold(0.0_f32, f32::max),
        }
    });
    let cell = side / res as f32;
    for (i, f) in field.iter().enumerate() {
        let (row, col) = (i / res, i % res);
        let r = Rect::from_min_size(rect.min + Vec2::new(col as f32 * cell, row as f32 * cell), Vec2::splat(cell + 0.5));
        painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(166, 191, 255, (f.clamp(0.0, 1.0) * 120.0) as u8));
    }
    let origin = cv::to_diagram([0.0, 0.0], centre, half, rect);
    let axis = Stroke::new(1.0_f32, theme::LINE_BUTTON);
    painter.hline(rect.x_range(), origin.y, axis);
    painter.vline(origin.x, rect.y_range(), axis);

    if !dragging_dot {
        for (k, p) in pts.iter().enumerate() {
            let wt = weights.get(k).copied().unwrap_or(0.0);
            if wt > 0.001 {
                let c = cv::to_diagram(*p, centre, half, rect);
                painter.circle(c, cv::weight_radius(wt), Color32::from_rgba_unmultiplied(160, 200, 255, 30), Stroke::new(1.0_f32, Color32::from_white_alpha(150)));
            }
        }
    }
    for (k, p) in pts.iter().enumerate() {
        let c = cv::to_diagram(*p, centre, half, rect);
        let empty = tree.motions[k].tree.is_none() && tree.motions[k].clip.is_empty();
        let col = if empty { theme::TEXT_HIDDEN } else { Color32::from_rgb(128, 179, 255) };
        painter.circle_filled(c, 5.5, col);
        if *sel == Some(k) {
            painter.circle_stroke(c, 8.0, Stroke::new(1.5_f32, Color32::WHITE));
        }
    }
    let p = [env.value(&tree.parameter), env.value(&tree.parameter_y)];
    painter.circle_filled(cv::to_diagram(p, centre, half, rect), 5.5, Color32::from_rgb(255, 102, 102));

    if (resp.drag_started() || resp.clicked())
        && let Some(pos) = resp.interact_pointer_pos()
    {
        let hit = pts
            .iter()
            .enumerate()
            .map(|(k, q)| (k, cv::to_diagram(*q, centre, half, rect).distance(pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|h| h.1 <= 9.0);
        let t = hit.map(|h| h.0).unwrap_or(SCRUB);
        if t != SCRUB {
            *sel = Some(t);
        }
        ui.data_mut(|d| {
            d.insert_temp(drag_id, t);
            d.insert_temp(view_id, (centre, half));
        });
    }
    if (resp.dragged() || resp.clicked())
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some(t) = ui.data(|d| d.get_temp::<usize>(drag_id))
    {
        let q = cv::from_diagram(pos, centre, half, rect);
        let round = |v: f32| ((v * 100.0).round() / 100.0).clamp(-1e4, 1e4);
        if t == SCRUB {
            env.set(&tree.parameter, round(q[0]));
            env.set(&tree.parameter_y, round(q[1]));
        } else if t < tree.motions.len() && resp.dragged() {
            tree.motions[t].position = [round(q[0]), round(q[1])];
            edited = true;
        }
    }
    if resp.drag_stopped() || (!resp.dragged() && !resp.is_pointer_button_down_on()) {
        ui.data_mut(|d| {
            d.remove::<([f32; 2], f32)>(view_id);
            d.remove::<usize>(drag_id);
        });
    }
    ui.label(egui::RichText::new(format!("Sampler ({:.2}, {:.2})", p[0], p[1])).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
    for w in cv::blend_warnings(kind, &pts) {
        ui.add(egui::Label::new(egui::RichText::new(w.text()).size(theme::SIZE_META).color(theme::STATUS_WARN_FG)).wrap());
    }
    edited
}

/// Set a transition's interruption source, retiring the legacy `interruptible`
/// flag (it would otherwise keep reading as `Destination` when set to None).
fn set_interruption(tr: &mut AnimTransition, src: InterruptionSource) {
    tr.interruptible = false;
    tr.interruption = src;
    if src == InterruptionSource::None {
        tr.ordered_interruption = false;
    }
}

#[cfg(test)]
mod interruption_tests {
    #[test]
    fn new_mask_assets_get_unique_names_from_the_layer() {
        let dir = std::env::temp_dir().join(format!("zest-mask-path-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("masks")).unwrap();
        assert_eq!(super::unique_mask_path(&dir, "Upper Body"), "masks/Upper_Body.mask");
        std::fs::write(dir.join("masks/Upper_Body.mask"), "x").unwrap();
        assert_eq!(super::unique_mask_path(&dir, "Upper Body"), "masks/Upper_Body_1.mask");
        assert_eq!(super::unique_mask_path(&dir, "***"), "masks/mask.mask");
    }

    use super::*;

    #[test]
    fn legacy_flag_migrates_on_edit() {
        let mut tr = AnimTransition { interruptible: true, ..Default::default() };
        assert_eq!(tr.interruption_source(), InterruptionSource::Destination);
        set_interruption(&mut tr, InterruptionSource::None);
        assert_eq!(tr.interruption_source(), InterruptionSource::None);
        set_interruption(&mut tr, InterruptionSource::Source);
        assert_eq!(tr.interruption_source(), InterruptionSource::Source);
        assert!(!tr.interruptible);
    }
}
