//! Scene hierarchy: a tree under a root "Scene" node with click-select,
//! collapse arrows, drag-to-reparent (drop a row onto another row to parent
//! it; drop onto the root or empty space to unparent), and a right-click
//! Add Object menu.

use std::collections::{HashMap, HashSet};

use egui::{FontId, RichText, ScrollArea, Sense, Ui};
use egui_phosphor::regular as ph;

use crate::icons::{self, Icon};
use crate::theme::{self, ObjectKind};
use crate::widgets;

pub struct SceneObjectRow {
    pub name: String,
    pub parent: Option<usize>,
    /// Type family: icon + colour.
    pub kind: ObjectKind,
    /// The object's own active flag. Shown (dimmed italic name) but only the
    /// Inspector changes it.
    pub enabled: bool,
    /// Hidden in the editor scene view by its own flag (the eye). Not `enabled`.
    pub scene_hidden: bool,
    /// Locked in the scene view by its own flag (the lock): no viewport picking.
    pub scene_locked: bool,
    /// Prefab instance markers (default: not part of one).
    pub prefab: PrefabMark,
    /// Where the object exists on a networked game (a `C` / `S` badge unless `Both`).
    pub net_role: zestengine_core::NetRole,
    /// The nearest ancestor's role when the object itself is `Both` but an
    /// ancestor is not (the object follows it; its badge is drawn dimmed).
    pub net_inherited: zestengine_core::NetRole,
    /// The bone this object is attached to (its Bone Attach): listed under that bone's row
    /// of its parent model instead of directly under the parent.
    pub bone: Option<String>,
}

impl SceneObjectRow {
    /// Hierarchy filter match: name, type label, or role (`client`, `server`, `both`).
    pub fn matches(&self, filter: &str) -> bool {
        self.name.to_lowercase().contains(filter)
            || self.kind.label().to_lowercase().contains(filter)
            || (!self.net_role.is_both() && self.net_role.label().to_lowercase().contains(filter))
    }
}

/// What an object is to the prefab instance it sits in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefabRole {
    #[default]
    None,
    /// An object inside an instance.
    Member,
    /// The root of an instance nested inside another.
    NestedRoot,
    /// The root of an outermost instance.
    Root,
}

/// Hierarchy markers of a prefab object.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefabMark {
    pub role: PrefabRole,
    /// Added to the instance (not in the asset).
    pub added: bool,
    /// An instance root with changes besides placement: draws the blue bar.
    pub changed: bool,
    /// Objects deleted from the instance (shown on its root).
    pub removed: u16,
    /// The prefab asset can't be read.
    pub missing: bool,
    /// Prefab Mode: the object is outside the open prefab. Shown dimmed, not
    /// selectable.
    pub locked: bool,
    /// Prefab Mode, isolated: the object is not listed at all.
    pub omit: bool,
}

/// A Prefab entry of a row's context menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefabMenu {
    /// Open in Prefab Mode over the instance.
    Open,
    /// Open in Prefab Mode alone.
    OpenIsolated,
    SelectAsset,
    /// Select the instance root the object belongs to.
    SelectRoot,
    Unpack,
    UnpackCompletely,
    RevertAll,
    ApplyAll,
}

/// What to add from the scene tree's context menu. The engine maps this to
/// its object sources (the editor crate stays renderer/asset-agnostic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnKind {
    Empty,
    Camera,
    Light(zestengine_core::LightKind),
    AreaLight,
    LightProbeVolume,
    PostFxVolume,
    GlobalPostFxVolume,
    ReflectionProbe,
    FluxVolume,
    AudioSource,
    AudioVolume,
    BoxCollider,
    SphereCollider,
    CapsuleCollider,
    /// A Mesh Collider on the selected mesh objects.
    MeshCollider,
    /// A 20 x 20 m lake.
    Water,
    /// A cube with a collider and a rigid body half as dense as water.
    FloatingCube,
    /// An open-top container with a rigid body that holds water.
    Bucket,
    Waterfall,
    Whirlpool,
    /// Wind for water waves, cloth and steam.
    WindZone,
    Cube,
    Sphere,
    Capsule,
    Plane,
    /// A flat heightfield landscape (100 x 100 m, 32 x 32 quads).
    Terrain,
    Ui(UiPreset),
}

/// Ready-made UI objects for the "UI" create menu.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UiPreset {
    Canvas,
    WorldCanvas,
    Panel,
    Text,
    Button,
    Image,
    Toggle,
    Slider,
    TextField,
    ProgressBar,
}

impl UiPreset {
    pub const ALL: [UiPreset; 10] = [
        Self::Canvas,
        Self::WorldCanvas,
        Self::Panel,
        Self::Text,
        Self::Button,
        Self::Image,
        Self::Toggle,
        Self::Slider,
        Self::TextField,
        Self::ProgressBar,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Canvas => "Canvas",
            Self::WorldCanvas => "World Canvas (VR)",
            Self::Panel => "Panel",
            Self::Text => "Text",
            Self::Button => "Button",
            Self::Image => "Image",
            Self::Toggle => "Toggle",
            Self::Slider => "Slider",
            Self::TextField => "Text Field",
            Self::ProgressBar => "Progress Bar",
        }
    }
}

impl SpawnKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Empty => "Empty",
            Self::Camera => "Camera",
            Self::Light(kind) => kind.label(),
            Self::AreaLight => "Area Light",
            Self::LightProbeVolume => "Light Probe Volume",
            Self::PostFxVolume => "Post FX Volume (Box)",
            Self::GlobalPostFxVolume => "Post FX Volume (Global)",
            Self::ReflectionProbe => "Reflection Probe",
            Self::FluxVolume => "Flux Volume",
            Self::AudioSource => "Audio Source",
            Self::AudioVolume => "Audio Volume",
            Self::BoxCollider => "Box Collider",
            Self::SphereCollider => "Sphere Collider",
            Self::CapsuleCollider => "Capsule Collider",
            Self::MeshCollider => "Mesh Collider",
            Self::Water => "Water",
            Self::FloatingCube => "Floating Cube",
            Self::Bucket => "Bucket",
            Self::Waterfall => "Waterfall",
            Self::Whirlpool => "Whirlpool",
            Self::WindZone => "Wind Zone",
            Self::Cube => "Cube",
            Self::Sphere => "Sphere",
            Self::Capsule => "Capsule",
            Self::Plane => "Plane",
            Self::Terrain => "Terrain",
            Self::Ui(p) => p.label(),
        }
    }
}

/// A scene-view hide / lock change asked of the hierarchy. Never touches `enabled`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewRequest {
    /// Set the own hide flag of these rows.
    Hide(Vec<usize>, bool),
    /// Set the own lock flag of these rows.
    Lock(Vec<usize>, bool),
    /// Alt-click on an eye: hide everything outside this row's tree (and its
    /// ancestors); a second Alt-click on the isolated row shows all again.
    Isolate(usize),
    ShowAll,
    UnlockAll,
}

/// What the `H` key does, decided from the focus / mode (pure, for tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKeyAction {
    /// Hide the selection (all hidden already: show it).
    ToggleHide,
    /// Show every hidden object.
    ShowAll,
}

/// `H` toggles scene-view hide on the selection; `Shift+H` or `Alt+H` shows
/// all. Only while the Scene view / Hierarchy has focus, no text is being
/// edited, and not in Play.
pub fn view_key_action(h_pressed: bool, shift: bool, alt: bool, ctrl: bool, text_editing: bool, playing: bool, focus_ok: bool) -> Option<ViewKeyAction> {
    if !h_pressed || ctrl || text_editing || playing || !focus_ok {
        return None;
    }
    Some(if shift || alt { ViewKeyAction::ShowAll } else { ViewKeyAction::ToggleHide })
}

/// The request for hiding `targets` given their own flags: if any is shown,
/// hide them all; if all are hidden, show them all.
pub fn toggle_hide_request(rows: &[SceneObjectRow], targets: &[usize]) -> Option<ViewRequest> {
    let targets: Vec<usize> = targets.iter().copied().filter(|&i| i < rows.len()).collect();
    if targets.is_empty() {
        return None;
    }
    let all_hidden = targets.iter().all(|&i| rows[i].scene_hidden);
    Some(ViewRequest::Hide(targets, !all_hidden))
}

#[derive(Default)]
pub struct ScenePanelResponse {
    pub selection_changed: bool,
    /// (child, new parent) reparent requests (keep Vec position).
    pub reparent: Vec<(usize, Option<usize>)>,
    /// (child, new parent, before-sibling) move/reorder requests. `before`
    /// None = append at the end of the new parent's children.
    pub moves: Vec<(usize, Option<usize>, Option<usize>)>,
    /// Object requested via the context menu.
    pub spawn: Option<SpawnKind>,
    /// Object index requested for deletion (context menu / Delete key).
    pub delete: Option<usize>,
    /// (index, new name) inline-rename commit (F2 in the tree).
    pub rename: Option<(usize, String)>,
    /// A model file dropped onto the tree (import it into the scene).
    pub import_model: Option<std::path::PathBuf>,
    /// A file dragged from the Files panel and dropped on the tree: (path, row it landed on;
    /// None = the Scene root or empty space). The host decides what it spawns or assigns.
    pub asset_drop: Option<(std::path::PathBuf, Option<usize>)>,
    /// The eye / lock / context menu asked for a scene-view hide or lock change.
    pub view: Option<ViewRequest>,
    /// "Group Into Empty": these objects go under a new Empty.
    pub group: Option<Vec<usize>>,
    /// Object requested to open in ZestSurface (context menu).
    pub edit_in_zestsurface: Option<usize>,
    /// "Add Toggle": make this object a menu toggle of the avatar it is under.
    pub add_toggle: Option<usize>,
    /// A Prefab command on an object (the row's arrow button or context menu).
    pub prefab: Option<(usize, PrefabMenu)>,
    /// (object, model root row, bone name): an object dropped on a bone row; attach it there.
    pub attach_bone: Vec<(usize, usize, String)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DropZone {
    /// Top of the row: drop as a sibling before it.
    Before,
    /// Middle of the row: drop as a child (reparent).
    Onto,
    /// Bottom of the row: drop as a sibling after it.
    After,
}

fn drop_zone(rect: egui::Rect, y: f32) -> DropZone {
    let t = (y - rect.top()) / rect.height().max(1.0);
    if t < 0.30 {
        DropZone::Before
    } else if t > 0.70 {
        DropZone::After
    } else {
        DropZone::Onto
    }
}

/// Next sibling of `index` (same parent, next in display order), if any.
fn next_sibling(rows: &[SceneObjectRow], index: usize) -> Option<usize> {
    let parent = rows[index].parent;
    (index + 1..rows.len()).find(|&j| rows[j].parent == parent)
}

/// Shared by the hierarchy Add/context menus and the top-level Object menu.
pub(crate) fn audio_spawn_menu(ui: &mut Ui) -> Option<SpawnKind> {
    let mut selected = None;
    ui.menu_button("Audio", |ui| {
        crate::widgets::menu_frame(ui);
        for kind in [SpawnKind::AudioSource, SpawnKind::AudioVolume] {
            if ui.button(kind.label()).clicked() {
                selected = Some(kind);
                ui.close();
            }
        }
    });
    selected
}

fn spawn_menu(ui: &mut Ui, response: &mut ScenePanelResponse) {
    ui.label(RichText::new("Add Object").small().weak());
    for kind in [SpawnKind::Empty, SpawnKind::Camera] {
        if ui.button(kind.label()).clicked() {
            response.spawn = Some(kind);
            ui.close();
        }
    }
    ui.menu_button("Water", |ui| {
        crate::widgets::menu_frame(ui);
        for kind in [SpawnKind::Water, SpawnKind::Waterfall, SpawnKind::Whirlpool, SpawnKind::FloatingCube, SpawnKind::Bucket] {
            if ui.button(kind.label()).clicked() {
                response.spawn = Some(kind);
                ui.close();
            }
        }
    });
    ui.menu_button("Light", |ui| {
        crate::widgets::menu_frame(ui);
        for kind in zestengine_core::LightKind::ALL {
            if ui.button(kind.label()).clicked() {
                response.spawn = Some(SpawnKind::Light(kind));
                ui.close();
            }
        }
        if ui.button("Area Light").clicked() {
            response.spawn = Some(SpawnKind::AreaLight);
            ui.close();
        }
    });
    ui.menu_button("Volume", |ui| {
        crate::widgets::menu_frame(ui);
        for kind in [
            SpawnKind::PostFxVolume,
            SpawnKind::GlobalPostFxVolume,
            SpawnKind::LightProbeVolume,
            SpawnKind::ReflectionProbe,
            SpawnKind::FluxVolume,
        ] {
            if ui.button(kind.label()).clicked() {
                response.spawn = Some(kind);
                ui.close();
            }
        }
    });
    ui.menu_button("3D Object", |ui| {
        crate::widgets::menu_frame(ui);
        for kind in [
            SpawnKind::Cube,
            SpawnKind::Sphere,
            SpawnKind::Capsule,
            SpawnKind::Plane,
            SpawnKind::Terrain,
        ] {
            if ui.button(kind.label()).clicked() {
                response.spawn = Some(kind);
                ui.close();
            }
        }
    });
    ui.menu_button("UI", |ui| {
        crate::widgets::menu_frame(ui);
        for preset in UiPreset::ALL {
            if ui.button(preset.label()).clicked() {
                response.spawn = Some(SpawnKind::Ui(preset));
                ui.close();
            }
        }
    });
    if let Some(kind) = audio_spawn_menu(ui) {
        response.spawn = Some(kind);
    }
    ui.menu_button("Physics", |ui| {
        crate::widgets::menu_frame(ui);
        if ui.button(SpawnKind::WindZone.label()).clicked() {
            response.spawn = Some(SpawnKind::WindZone);
            ui.close();
        }
    });
    ui.menu_button("Collider", |ui| {
        crate::widgets::menu_frame(ui);
        if ui.button("Box Collider").clicked() {
            response.spawn = Some(SpawnKind::BoxCollider);
            ui.close();
        }
        if ui.button("Sphere Collider").clicked() {
            response.spawn = Some(SpawnKind::SphereCollider);
            ui.close();
        }
        if ui.button("Capsule Collider").clicked() {
            response.spawn = Some(SpawnKind::CapsuleCollider);
            ui.close();
        }
        if ui.button("Mesh Collider").on_hover_text("Add a Mesh Collider to the selected mesh objects").clicked() {
            response.spawn = Some(SpawnKind::MeshCollider);
            ui.close();
        }
    });
}

/// A model's skeleton as the hierarchy lists it under the model root: virtual, read-only rows
/// (not scene objects). `bones[i].1` is the parent bone, `None` for the single top node.
#[derive(Default)]
struct ArmatureTree {
    /// Cache key: the engine's skeleton index and joint count.
    key: (usize, usize),
    bones: Vec<(String, Option<usize>)>,
    children: Vec<Vec<usize>>,
}

#[derive(Default)]
pub struct ScenePanel {
    filter: String,
    collapsed: HashSet<usize>,
    /// The virtual "Scene" root row that parents every top-level object.
    scene_collapsed: bool,
    /// Inline rename in progress: (object index, edit buffer, focus-on-first-frame).
    renaming: Option<(usize, String, bool)>,
    /// Full multi-selection (object indices). Synced from the engine's canonical
    /// selection each frame via [`set_multi`], mutated here on ctrl/shift clicks,
    /// and read back after `ui` when `selection_changed`. The `selected` anchor
    /// (last-clicked) is passed in/out separately and is always also in `multi`.
    multi: HashSet<usize>,
    /// The tree has keyboard focus (last click landed in it): arrow keys
    /// navigate.
    focused: bool,
    /// Row to scroll into view this frame (after keyboard navigation).
    scroll_to: Option<usize>,
    /// Selection as of last frame, to notice picks made elsewhere (viewport).
    last_selected: Option<usize>,
    /// Row being pinged (an `ObjectRef` chip was clicked) and when, for the flash.
    ping: Option<(usize, std::time::Instant)>,
    /// Skeleton trees of the model roots (root row -> tree), shown first under the root.
    armatures: HashMap<usize, ArmatureTree>,
    /// Expanded bone rows (root row, bone); bones start collapsed.
    armature_open: HashSet<(usize, usize)>,
    /// The bone row last clicked (root row, bone): highlight only.
    bone_selected: Option<(usize, usize)>,
    /// Objects attached to a bone, per (root row, bone): shown under that bone's row. Rebuilt
    /// each frame by `ui`.
    bone_kids: HashMap<(usize, usize), Vec<usize>>,
    /// Bone rows already opened for their attached objects, so the user can collapse them.
    attached_seen: HashSet<(usize, usize)>,
}

impl ScenePanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sync the multi-selection from the engine's canonical set before drawing.
    /// Set which model roots list a skeleton: `(root row, skeleton key)`. A tree is rebuilt by
    /// `build(key)` (names and parents, parents first) only when a root's key changed.
    pub fn sync_armatures(&mut self, wanted: &[(usize, (usize, usize))], build: impl Fn((usize, usize)) -> Vec<(String, Option<usize>)>) {
        self.armatures.retain(|root, t| wanted.iter().any(|(r, k)| r == root && *k == t.key));
        for &(root, key) in wanted {
            if self.armatures.contains_key(&root) {
                continue;
            }
            let bones = build(key);
            let mut children = vec![Vec::new(); bones.len()];
            for (i, (_, parent)) in bones.iter().enumerate() {
                if let Some(p) = parent.filter(|&p| p < i) {
                    children[p].push(i);
                }
            }
            self.armatures.insert(root, ArmatureTree { key, bones, children });
        }
    }

    pub fn set_multi(&mut self, set: impl IntoIterator<Item = usize>) {
        self.multi = set.into_iter().collect();
    }

    /// The current multi-selection (read after `ui` when `selection_changed`).
    pub fn multi(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.multi.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Indices of currently-collapsed rows (for persisting tree state).
    pub fn collapsed_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.collapsed.iter().copied()
    }

    /// Restore the collapsed-row set (from the saved scene tree state).
    pub fn set_collapsed(&mut self, set: impl IntoIterator<Item = usize>) {
        self.collapsed = set.into_iter().collect();
    }

    pub fn ui(
        &mut self,
        ui: &mut Ui,
        rows: &[SceneObjectRow],
        selected: &mut Option<usize>,
    ) -> ScenePanelResponse {
        let mut response = ScenePanelResponse::default();
        // Selected elsewhere (the viewport): open every collapsed parent above
        // it and scroll its row into view.
        if *selected != self.last_selected
            && let Some(sel) = *selected
            && sel < rows.len()
        {
            self.reveal(rows, sel);
        }
        // An `ObjectRef` chip in an inspector was clicked: show where that
        // object is (open its parents, scroll to it, flash the row) without
        // changing the selection.
        let ping_key = egui::Id::new(crate::PING_OBJECT_KEY);
        if let Some(idx) = ui.data(|d| d.get_temp::<usize>(ping_key)) {
            ui.data_mut(|d| d.remove::<usize>(ping_key));
            if idx < rows.len() {
                self.filter.clear();
                self.reveal(rows, idx);
                self.ping = Some((idx, std::time::Instant::now()));
            }
        }
        let panel_rect = ui.max_rect();
        if let Some(p) = ui.input(|i| i.pointer.any_pressed().then(|| i.pointer.interact_pos()).flatten()) {
            self.focused = panel_rect.contains(p);
        }
        // Header strip: title + count, Add menu, kebab.
        egui::Frame::new().inner_margin(egui::Margin::symmetric(theme::PAD as i8, 0)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.set_height(theme::H_TAB_STRIP);
                ui.label(RichText::new("Scene").font(theme::semibold(theme::SIZE_BODY)).color(theme::TEXT_STRONG));
                widgets::badge(ui, &rows.len().to_string());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let kebab = widgets::icon_button(ui, Icon::Kebab, "More", false);
                    egui::Popup::menu(&kebab).show(|ui| {
                        if ui.button("Expand all").clicked() {
                            self.collapsed.clear();
                            self.scene_collapsed = false;
                            ui.close();
                        }
                        if ui.button("Collapse all").clicked() {
                            self.collapsed = (0..rows.len()).collect();
                            ui.close();
                        }
                        ui.separator();
                        if ui.add_enabled(rows.iter().any(|r| r.scene_hidden), egui::Button::new("Show all")).on_hover_text("Un-hide every object in the scene view (Shift+H)").clicked() {
                            response.view = Some(ViewRequest::ShowAll);
                            ui.close();
                        }
                        if ui.add_enabled(rows.iter().any(|r| r.scene_locked), egui::Button::new("Unlock all")).clicked() {
                            response.view = Some(ViewRequest::UnlockAll);
                            ui.close();
                        }
                    });
                    let add = widgets::secondary_button_sized(ui, Some(Icon::Plus), "Add", theme::H_BUTTON_SMALL);
                    egui::Popup::menu(&add).show(|ui| spawn_menu(ui, &mut response));
                });
            });
        });
        widgets::divider(ui);
        egui::Frame::new().inner_margin(egui::Margin::same(8)).show(ui, |ui| {
            let w = ui.available_width();
            widgets::search_field(ui, &mut self.filter, "Filter by name or type", w);
        });
        let filter = self.filter.to_lowercase();

        // F2 starts an inline rename of the selected object, unless a text
        // field is already focused (e.g. the filter box).
        if self.renaming.is_none()
            && let Some(i) = *selected
            && i < rows.len()
            && ui.input(|inp| inp.key_pressed(egui::Key::F2))
            && !ui.memory(|m| m.focused().is_some())
        {
            self.renaming = Some((i, rows[i].name.clone(), true));
        }

        // children[i] = indices whose parent == i; roots have no parent.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); rows.len()];
        let mut roots: Vec<usize> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            match row.parent {
                Some(p) if p < rows.len() && p != i => children[p].push(i),
                _ => roots.push(i),
            }
        }

        // Attached objects move from under their model root to under their bone's row. A bone
        // seen with attachments for the first time opens (with the bones above it).
        self.bone_kids = split_attached(rows, &mut children, &self.armatures);
        self.attached_seen.retain(|k| self.bone_kids.contains_key(k));
        let keys: Vec<(usize, usize)> = self.bone_kids.keys().copied().collect();
        for (root, bone) in keys {
            if self.attached_seen.insert((root, bone)) {
                self.collapsed.remove(&root);
                self.open_bone(root, bone);
            }
        }

        if self.focused && self.renaming.is_none() && !ui.memory(|m| m.focused().is_some()) {
            self.keyboard_nav(ui, rows, &children, &roots, &filter, selected, &mut response);
        }

        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Background click sensor over the whole tree, registered BEFORE
                // the rows so the rows (drawn after) take hit priority — a click
                // that lands here (empty space) clears the selection. More
                // reliable than a trailing rect, which can be mis-sized.
                let deselect_bg =
                    ui.interact(ui.max_rect(), ui.id().with("tree-deselect-bg"), Sense::click());
                ui.spacing_mut().item_spacing.y = 0.0;
                let hidden = hidden_flags(rows);
                if filter.is_empty() {
                    if !rows.is_empty() {
                        self.scene_root_row(ui, rows.len(), &mut response);
                    }
                    if !self.scene_collapsed || rows.is_empty() {
                        let depth = usize::from(!rows.is_empty());
                        for &index in &roots {
                            self.tree_row(ui, rows, &children, &hidden, index, depth, selected, &mut response);
                        }
                    }
                } else {
                    // Filtered: flat list, matching name or type.
                    for (i, row) in rows.iter().enumerate() {
                        if row.matches(&filter) {
                            self.object_row(ui, rows, &children, &hidden, false, i, 0, selected, &mut response);
                        }
                    }
                }
                if rows.is_empty() {
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.add_space(theme::PAD);
                        ui.label(RichText::new("Scene is empty").color(theme::TEXT_DIM));
                    });
                }
                // Remaining empty space: drop target to unparent and
                // right-click target for the Add Object menu.
                let mut rest = ui.available_rect_before_wrap();
                rest.set_height(rest.height().max(24.0));
                let bg = ui.allocate_rect(rest, Sense::click());
                if let Some(payload) = bg.dnd_release_payload::<usize>() {
                    response.reparent.push((*payload, None));
                }
                // A file dragged from the Files panel onto empty tree space: the host spawns
                // it at world origin (models, prefabs, clips).
                if let Some(payload) = bg.dnd_release_payload::<std::path::PathBuf>() {
                    response.asset_drop = Some(((*payload).clone(), None));
                }
                // A left-click on empty tree space (not on a row) clears the
                // selection. `deselect_bg` covers the area around rows; `bg` is
                // the trailing rect that fills the (often large) area BELOW the
                // last row and sits on top there — so check both.
                if deselect_bg.clicked() || bg.clicked() {
                    *selected = None;
                    self.multi.clear();
                    response.selection_changed = true;
                }
                bg.context_menu(|ui| {
                    crate::widgets::menu_frame(ui);
                    spawn_menu(ui, &mut response);
                });
            });
        self.last_selected = *selected;
        response
    }

    /// Open the Scene root and every collapsed parent above `index`, and scroll
    /// its row into view.
    fn reveal(&mut self, rows: &[SceneObjectRow], index: usize) {
        self.scene_collapsed = false;
        self.reveal_bone(rows, index);
        let mut p = rows[index].parent;
        let mut guard = 0;
        while let Some(i) = p.filter(|&i| i < rows.len()) {
            self.collapsed.remove(&i);
            self.reveal_bone(rows, i);
            p = rows[i].parent;
            guard += 1;
            if guard > rows.len() {
                break;
            }
        }
        self.scroll_to = Some(index);
    }

    /// Open the bone row `rows[index]` is attached to (and the bones above it).
    fn reveal_bone(&mut self, rows: &[SceneObjectRow], index: usize) {
        if let (Some(name), Some(root)) = (&rows[index].bone, rows[index].parent)
            && let Some(bone) = self.armatures.get(&root).and_then(|t| t.bones.iter().position(|(n, _)| n == name))
        {
            self.open_bone(root, bone);
        }
    }

    /// Expand `bone` of `root`'s armature and every bone above it.
    fn open_bone(&mut self, root: usize, bone: usize) {
        let Some(tree) = self.armatures.get(&root) else { return };
        let mut chain = vec![bone];
        let mut b = bone;
        while let Some(p) = tree.bones.get(b).and_then(|(_, p)| *p).filter(|p| !chain.contains(p)) {
            chain.push(p);
            b = p;
        }
        self.armature_open.extend(chain.into_iter().map(|b| (root, b)));
    }

    /// Objects attached to `root`'s bones that are on screen, in row order (bones that are
    /// collapsed hide theirs).
    fn visible_attached(&self, root: usize) -> Vec<usize> {
        fn walk(panel: &ScenePanel, tree: &ArmatureTree, root: usize, bone: usize, out: &mut Vec<usize>, depth: usize) {
            if depth > 64 || !panel.armature_open.contains(&(root, bone)) {
                return;
            }
            for &k in &tree.children[bone] {
                walk(panel, tree, root, k, out, depth + 1);
            }
            out.extend(panel.bone_kids.get(&(root, bone)).into_iter().flatten());
        }
        let mut out = Vec::new();
        if let Some(tree) = self.armatures.get(&root) {
            walk(self, tree, root, 0, &mut out, 0);
        }
        out
    }

    /// Arrow keys: Up / Down step through the visible rows, Right expands (or
    /// steps into the first child), Left collapses (or steps out to the parent).
    #[allow(clippy::too_many_arguments)]
    fn keyboard_nav(
        &mut self,
        ui: &Ui,
        rows: &[SceneObjectRow],
        children: &[Vec<usize>],
        roots: &[usize],
        filter: &str,
        selected: &mut Option<usize>,
        response: &mut ScenePanelResponse,
    ) {
        let (up, down, left, right) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
            )
        });
        if !(up || down || left || right) || rows.is_empty() || (self.scene_collapsed && filter.is_empty()) {
            return;
        }
        let visible: Vec<usize> = if filter.is_empty() {
            let mut out = Vec::new();
            let mut stack: Vec<usize> = roots.iter().rev().copied().collect();
            while let Some(n) = stack.pop() {
                out.push(n);
                if !self.collapsed.contains(&n) {
                    stack.extend(children[n].iter().rev().copied());
                    // Attached objects sit above the plain children (under the armature rows).
                    stack.extend(self.visible_attached(n).into_iter().rev());
                }
            }
            out
        } else {
            (0..rows.len())
                .filter(|&i| rows[i].matches(filter))
                .collect()
        };
        if visible.is_empty() {
            return;
        }
        let cur = selected.filter(|i| *i < rows.len());
        let pos = cur.and_then(|c| visible.iter().position(|&v| v == c));
        let mut target = None;
        if up || down {
            target = Some(match pos {
                None => if down { visible[0] } else { visible[visible.len() - 1] },
                Some(p) if up => visible[p.saturating_sub(1)],
                Some(p) => visible[(p + 1).min(visible.len() - 1)],
            });
        } else if let Some(c) = cur {
            let has_children = !children[c].is_empty() && filter.is_empty();
            if right && has_children {
                if self.collapsed.remove(&c) {
                    self.scroll_to = Some(c);
                } else {
                    target = Some(children[c][0]);
                }
            } else if left {
                if has_children && !self.collapsed.contains(&c) {
                    self.collapsed.insert(c);
                    self.scroll_to = Some(c);
                } else if let Some(p) = rows[c].parent.filter(|p| *p < rows.len()) {
                    target = Some(p);
                }
            }
        }
        if let Some(t) = target.filter(|t| Some(*t) != cur) {
            *selected = Some(t);
            self.multi.clear();
            self.multi.insert(t);
            self.scroll_to = Some(t);
            response.selection_changed = true;
        }
    }

    /// The "Scene" root row above the tree. Click toggles it; Ctrl/Cmd/Alt+click
    /// collapses every object (and the root), or expands everything when the
    /// root is already collapsed. Dropping an object here unparents it.
    fn scene_root_row(&mut self, ui: &mut Ui, count: usize, response: &mut ScenePanelResponse) {
        let (rect, row) = ui.allocate_exact_size(egui::vec2(ui.available_width(), theme::H_ROW), Sense::click());
        let inner = rect.shrink2(egui::vec2(6.0, 1.0));
        let drop_hover = row.dnd_hover_payload::<usize>().is_some();
        if drop_hover {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::accent_faint());
        } else if row.hovered() {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::with_alpha(theme::BG_BUTTON, 0.6));
        }
        let x0 = inner.left() + 4.0;
        let cy = rect.center().y;
        crate::widgets::paint_caret(
            ui.painter(),
            egui::Rect::from_center_size(egui::pos2(x0 + 8.0, cy), egui::vec2(12.0, 12.0)),
            !self.scene_collapsed,
            theme::TEXT_DIM,
        );
        icons::paint(
            ui.painter(),
            egui::Rect::from_center_size(egui::pos2(x0 + 16.0 + 9.0, cy), egui::vec2(14.0, 14.0)),
            Icon::Folder,
            theme::TEXT_SECONDARY,
        );
        let name_left = x0 + 16.0 + 22.0;
        let galley = ui.painter().layout_no_wrap("Scene".to_owned(), theme::semibold(theme::SIZE_BODY), theme::TEXT_STRONG);
        let name_right = name_left + galley.size().x;
        ui.painter().galley(egui::pos2(name_left, cy - galley.size().y * 0.5), galley, theme::TEXT_STRONG);
        if self.scene_collapsed {
            let text = count.to_string();
            let galley = ui.painter().layout_no_wrap(text, theme::mono(theme::SIZE_TINY), theme::TEXT_MUTED);
            let badge = egui::Rect::from_center_size(
                egui::pos2(name_right + 12.0 + galley.size().x * 0.5, cy),
                egui::vec2(galley.size().x + 10.0, 16.0),
            );
            ui.painter().rect_filled(badge, theme::R_FIELD, theme::LINE_PANEL);
            ui.painter().galley(badge.center() - galley.size() * 0.5, galley, theme::TEXT_MUTED);
        }
        if row.clicked() {
            let cascade = ui.input(|i| i.modifiers.alt || i.modifiers.command);
            if cascade && !self.scene_collapsed {
                self.collapsed = (0..count).collect();
                self.scene_collapsed = true;
            } else if cascade {
                self.collapsed.clear();
                self.scene_collapsed = false;
            } else {
                self.scene_collapsed = !self.scene_collapsed;
            }
        }
        if let Some(payload) = row.dnd_release_payload::<usize>() {
            response.reparent.push((*payload, None));
        }
        if let Some(payload) = row.dnd_release_payload::<std::path::PathBuf>() {
            response.asset_drop = Some(((*payload).clone(), None));
        }
        row.context_menu(|ui| {
            crate::widgets::menu_frame(ui);
            spawn_menu(ui, response);
        });
    }

    /// Collapse (`collapse=true`) or expand the whole subtree rooted at
    /// `index`, inclusive. Used by Alt+click.
    fn cascade(&mut self, children: &[Vec<usize>], index: usize, collapse: bool) {
        let mut stack = vec![index];
        while let Some(n) = stack.pop() {
            if collapse {
                self.collapsed.insert(n);
            } else {
                self.collapsed.remove(&n);
            }
            stack.extend(children[n].iter().copied());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_row(
        &mut self,
        ui: &mut Ui,
        rows: &[SceneObjectRow],
        children: &[Vec<usize>],
        hidden: &[bool],
        index: usize,
        depth: usize,
        selected: &mut Option<usize>,
        response: &mut ScenePanelResponse,
    ) {
        if depth > 64 || rows[index].prefab.omit {
            return;
        }
        let has_children = !children[index].is_empty() || self.armatures.contains_key(&index);
        self.object_row(ui, rows, children, hidden, has_children, index, depth, selected, response);
        if has_children && !self.collapsed.contains(&index) {
            self.armature_rows(ui, rows, children, hidden, index, 0, depth + 1, selected, response);
            for &child in &children[index] {
                self.tree_row(ui, rows, children, hidden, child, depth + 1, selected, response);
            }
        }
    }

    /// The bone tree of model root `root` from bone `bone` down: read-only virtual rows, a click
    /// only highlights. Bones start collapsed (`armature_open` holds the expanded ones).
    #[allow(clippy::too_many_arguments)]
    fn armature_rows(
        &mut self,
        ui: &mut Ui,
        rows: &[SceneObjectRow],
        children: &[Vec<usize>],
        hidden: &[bool],
        root: usize,
        bone: usize,
        depth: usize,
        selected: &mut Option<usize>,
        response: &mut ScenePanelResponse,
    ) {
        let Some(tree) = self.armatures.get(&root) else { return };
        let Some((name, _)) = tree.bones.get(bone) else { return };
        if depth > 64 {
            return;
        }
        let name = name.clone();
        let kids = tree.children[bone].clone();
        let key = (root, bone);
        let attached = self.bone_kids.get(&key).cloned().unwrap_or_default();
        let open = self.armature_open.contains(&key);
        let is_selected = self.bone_selected == Some(key);
        let (rect, row) = ui.allocate_exact_size(egui::vec2(ui.available_width(), theme::H_ROW), Sense::click());
        let inner = rect.shrink2(egui::vec2(6.0, 1.0));
        // An object dropped on the bone attaches to it (a file drop is not handled here).
        let drop_hover = row.dnd_hover_payload::<usize>().is_some();
        if drop_hover {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::accent_faint());
            ui.painter().rect_stroke(row.rect, 3.0, egui::Stroke::new(1.5_f32, theme::accent()), egui::StrokeKind::Outside);
        } else if is_selected {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::accent_faint());
        } else if row.hovered() {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::with_alpha(theme::BG_BUTTON, 0.6));
        }
        if let Some(payload) = row.dnd_release_payload::<usize>() {
            response.attach_bone.push((*payload, root, name.clone()));
            self.attached_seen.insert(key);
            self.collapsed.remove(&root);
            self.open_bone(root, bone);
        }
        let cy = rect.center().y;
        let x0 = inner.left() + 4.0 + VIEW_COL_W + depth as f32 * theme::INDENT;
        let has_kids = !kids.is_empty() || !attached.is_empty();
        if has_kids {
            crate::widgets::paint_caret(ui.painter(), egui::Rect::from_center_size(egui::pos2(x0 + 8.0, cy), egui::vec2(12.0, 12.0)), open, theme::TEXT_DIM);
        }
        ui.painter().text(egui::pos2(x0 + 25.0, cy), egui::Align2::CENTER_CENTER, ph::BONE, FontId::proportional(14.0), theme::TEXT_DIM);
        let color = if is_selected { theme::TEXT_SELECTED } else { theme::TEXT_SECONDARY };
        let mut job = egui::text::LayoutJob::default();
        job.append(&name, 0.0, egui::text::TextFormat { font_id: FontId::proportional(theme::SIZE_BODY), color, ..Default::default() });
        let left = x0 + 38.0;
        job.wrap = egui::text::TextWrapping::truncate_at_width((inner.right() - 6.0 - left).max(0.0));
        job.wrap.max_rows = 1;
        let galley = ui.painter().layout_job(job);
        ui.painter().galley(egui::pos2(left, cy - galley.size().y * 0.5), galley, color);
        if row.clicked() {
            self.bone_selected = Some(key);
            // The caret column toggles; so does a double click anywhere on the row.
            let on_caret = row.interact_pointer_pos().is_some_and(|p| p.x < x0 + 16.0);
            if has_kids && (on_caret || row.double_clicked()) && !self.armature_open.remove(&key) {
                self.armature_open.insert(key);
            }
        }
        if open {
            for k in kids {
                self.armature_rows(ui, rows, children, hidden, root, k, depth + 1, selected, response);
            }
            for child in attached {
                self.tree_row(ui, rows, children, hidden, child, depth + 1, selected, response);
            }
        }
    }

    /// One 26 px row: caret (space kept on leaves), type icon, name, child
    /// count when collapsed, and the eye toggle on the right. Returns the row
    /// response and whether the eye was clicked.
    #[allow(clippy::too_many_arguments)]
    fn row_widget(
        &mut self,
        ui: &mut Ui,
        row_data: &SceneObjectRow,
        depth: usize,
        is_selected: bool,
        child_count: usize,
        collapsed: bool,
        hidden: bool,
        view: ViewState,
    ) -> (egui::Response, RowClicks) {
        let (rect, row) = ui.allocate_exact_size(egui::vec2(ui.available_width(), theme::H_ROW), Sense::click_and_drag());
        let inner = rect.shrink2(egui::vec2(6.0, 1.0));
        if is_selected {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::accent_faint());
        } else if row.hovered() {
            ui.painter().rect_filled(inner, theme::R_ROW, theme::with_alpha(theme::BG_BUTTON, 0.6));
        }
        // Fixed left column: eye (hide in scene view) and lock.
        let cy = rect.center().y;
        let eye_rect = egui::Rect::from_center_size(egui::pos2(inner.left() + 13.0, cy), egui::vec2(18.0, 22.0));
        let lock_rect = egui::Rect::from_center_size(egui::pos2(inner.left() + 31.0, cy), egui::vec2(18.0, 22.0));
        let x0 = inner.left() + 4.0 + VIEW_COL_W + depth as f32 * theme::INDENT;
        if child_count > 0 {
            crate::widgets::paint_caret(
                ui.painter(),
                egui::Rect::from_center_size(egui::pos2(x0 + 8.0, cy), egui::vec2(12.0, 12.0)),
                !collapsed,
                theme::TEXT_DIM,
            );
        }
        let icon_color = if hidden { theme::with_alpha(row_data.kind.color(), 0.45) } else { row_data.kind.color() };
        let mark = row_data.prefab;
        let in_prefab = mark.role != PrefabRole::None;
        let is_prefab_root = matches!(mark.role, PrefabRole::Root | PrefabRole::NestedRoot);
        let icon_center = egui::pos2(x0 + 16.0 + 9.0, cy);
        if is_prefab_root {
            let color = if mark.missing { theme::STATUS_ERROR } else { theme::PREFAB_NAME };
            ui.painter().text(
                icon_center,
                egui::Align2::CENTER_CENTER,
                ph::CUBE,
                FontId::proportional(14.0),
                if hidden { theme::with_alpha(color, 0.45) } else { color },
            );
        } else {
            icons::paint(ui.painter(), egui::Rect::from_center_size(icon_center, egui::vec2(14.0, 14.0)), row_data.kind.icon(), icon_color);
        }
        if mark.added {
            ui.painter().text(
                icon_center + egui::vec2(6.0, 6.0),
                egui::Align2::CENTER_CENTER,
                ph::PLUS_CIRCLE,
                FontId::proportional(9.0),
                crate::prefab_ui::added_color(),
            );
        }
        if mark.role == PrefabRole::Root && mark.changed {
            // The override bar: 2 px down the row's left edge.
            ui.painter().rect_filled(
                egui::Rect::from_min_max(egui::pos2(rect.left() + 1.0, rect.top() + 3.0), egui::pos2(rect.left() + 3.0, rect.bottom() - 3.0)),
                0.0,
                theme::PREFAB_OVERRIDE,
            );
        }
        let eye = ui.interact(eye_rect, row.id.with("eye"), Sense::click());
        let lock = ui.interact(lock_rect, row.id.with("lock"), Sense::click());
        let quiet = if is_selected { theme::TEXT_MUTED } else { theme::EYE_QUIET };
        let view_color = |hover: bool, own: bool, inherited: bool| {
            if own {
                theme::accent()
            } else if inherited {
                theme::with_alpha(theme::accent(), 0.45)
            } else if hover {
                theme::TEXT_STRONG
            } else {
                theme::with_alpha(quiet, 0.55)
            }
        };
        let (eye_own, eye_inh) = (view.hidden_own, view.hidden_inherited);
        icons::paint(
            ui.painter(),
            egui::Rect::from_center_size(eye_rect.center(), egui::vec2(14.0, 14.0)),
            if eye_own || eye_inh { Icon::EyeOff } else { Icon::Eye },
            view_color(eye.hovered(), eye_own, eye_inh),
        );
        let (lock_own, lock_inh) = (view.locked_own, view.locked_inherited);
        icons::paint(
            ui.painter(),
            egui::Rect::from_center_size(lock_rect.center(), egui::vec2(14.0, 14.0)),
            if lock_own || lock_inh { Icon::Lock } else { Icon::LockOpen },
            view_color(lock.hovered(), lock_own, lock_inh),
        );
        let eye_tip = if eye_inh && !eye_own { "Hidden by a parent. Hide in scene view (H); does not disable the object" } else { "Hide in scene view (H). Does not disable the object; Alt-click isolates" };
        let lock_tip = if lock_inh && !lock_own { "Locked by a parent. Lock: cannot be selected in the scene view" } else { "Lock: cannot be selected in the scene view" };
        let clicks = RowClicks {
            eye: eye.on_hover_text(eye_tip).clicked(),
            lock: lock.on_hover_text(lock_tip).clicked(),
            arrow: false,
        };
        let eye_rect = egui::Rect::from_min_max(egui::pos2(inner.right() - 2.0, rect.top()), egui::pos2(inner.right() - 2.0, rect.bottom()));

        let name_color = if hidden {
            theme::TEXT_HIDDEN
        } else if mark.missing && in_prefab {
            theme::STATUS_ERROR
        } else if in_prefab {
            theme::PREFAB_NAME
        } else if is_selected {
            theme::TEXT_SELECTED
        } else {
            theme::TEXT_SECONDARY
        };
        let name_left = x0 + 16.0 + 22.0;
        let mut name_right = eye_rect.left() - 6.0; // the row's right edge
        // The "open in Prefab Mode" arrow on instance roots.
        let mut arrow_clicked = false;
        if is_prefab_root {
            let arrow_rect = egui::Rect::from_center_size(egui::pos2(eye_rect.left() - 12.0, cy), egui::vec2(20.0, 22.0));
            let arrow = ui.interact(arrow_rect, row.id.with("prefab_open"), Sense::click());
            let color = if arrow.hovered() { theme::TEXT_STRONG } else { theme::PREFAB_NAME };
            ui.painter().text(arrow_rect.center(), egui::Align2::CENTER_CENTER, ph::CARET_RIGHT, FontId::proportional(14.0), color);
            arrow_clicked = arrow.on_hover_text("Open the prefab in Prefab Mode").clicked();
            name_right = arrow_rect.left() - 4.0;
            if mark.removed > 0 {
                let text = format!("{} {}", ph::MINUS_CIRCLE, mark.removed);
                let galley = ui.painter().layout_no_wrap(text, theme::mono(theme::SIZE_TINY), theme::STATUS_ERROR);
                let pos = egui::pos2(name_right - galley.size().x, cy - galley.size().y * 0.5);
                name_right = pos.x - 4.0;
                ui.painter().galley(pos, galley, theme::STATUS_ERROR);
            }
        }
        if collapsed && child_count > 0 {
            let text = child_count.to_string();
            let galley = ui.painter().layout_no_wrap(text, theme::mono(theme::SIZE_TINY), theme::TEXT_MUTED);
            let badge = egui::Rect::from_center_size(
                egui::pos2(name_right - galley.size().x * 0.5 - 5.0, cy),
                egui::vec2(galley.size().x + 10.0, 16.0),
            );
            ui.painter().rect_filled(badge, theme::R_FIELD, theme::LINE_PANEL);
            ui.painter().galley(badge.center() - galley.size() * 0.5, galley, theme::TEXT_MUTED);
            name_right = badge.left() - 6.0;
        }
        // Network role tag: `C` (client only) / `S` (server only); dimmed while inherited.
        let (tag_role, inherited) = if row_data.net_role.is_both() { (row_data.net_inherited, true) } else { (row_data.net_role, false) };
        if let Some(letter) = tag_role.badge()
            && let Some(color) = theme::net_role_color(tag_role)
        {
            let color = if hidden || inherited { theme::with_alpha(color, 0.45) } else { color };
            let tag = egui::Rect::from_center_size(egui::pos2(name_right - 7.0, cy), egui::vec2(14.0, 14.0));
            ui.painter().rect_stroke(tag, theme::R_FIELD, egui::Stroke::new(1.0_f32, color), egui::StrokeKind::Inside);
            ui.painter().text(tag.center(), egui::Align2::CENTER_CENTER, letter, theme::mono(theme::SIZE_TINY), color);
            ui.interact(tag, row.id.with("net_role"), Sense::hover()).on_hover_text(format!(
                "{}: {}",
                if inherited { format!("{} (from a parent)", tag_role.label()) } else { tag_role.label().to_owned() },
                if tag_role == zestengine_core::NetRole::Client { "removed from a dedicated server" } else { "removed from players' clients" }
            ));
            name_right = tag.left() - 4.0;
        }
        let font = FontId::proportional(theme::SIZE_BODY);
        let mut job = egui::text::LayoutJob::default();
        job.append(&row_data.name, 0.0, egui::text::TextFormat { font_id: font, color: name_color, italics: hidden, ..Default::default() });
        job.wrap = egui::text::TextWrapping::truncate_at_width((name_right - name_left).max(0.0));
        job.wrap.max_rows = 1;
        let galley = ui.painter().layout_job(job);
        ui.painter().galley(egui::pos2(name_left, cy - galley.size().y * 0.5), galley, name_color);
        (row, RowClicks { arrow: arrow_clicked, ..clicks })
    }

    #[allow(clippy::too_many_arguments)]
    fn object_row(
        &mut self,
        ui: &mut Ui,
        rows: &[SceneObjectRow],
        children: &[Vec<usize>],
        hidden: &[bool],
        has_children: bool,
        index: usize,
        depth: usize,
        selected: &mut Option<usize>,
        response: &mut ScenePanelResponse,
    ) {
        let row_data = &rows[index];
        // Highlight the anchor AND every multi-selected row.
        let is_selected = *selected == Some(index) || self.multi.contains(&index);
        let collapsed = self.collapsed.contains(&index);

        // Inline rename editor in place of the normal row.
        if matches!(&self.renaming, Some((i, _, _)) if *i == index) {
            let indent = 10.0 + depth as f32 * theme::INDENT + 38.0 + VIEW_COL_W;
            let mut commit = false;
            let mut cancel = false;
            ui.horizontal(|ui| {
                ui.add_space(indent);
                if let Some((_, text, focus)) = &mut self.renaming {
                    let edit = ui.add(egui::TextEdit::singleline(text).desired_width(f32::INFINITY));
                    if *focus {
                        edit.request_focus();
                        *focus = false;
                    }
                    if edit.lost_focus() {
                        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            cancel = true;
                        } else {
                            commit = true;
                        }
                    }
                }
            });
            if commit {
                if let Some((i, text, _)) = self.renaming.take() {
                    let name = text.trim().to_owned();
                    if !name.is_empty() && name != rows[i].name {
                        response.rename = Some((i, name));
                    }
                }
            } else if cancel {
                self.renaming = None;
            }
            return;
        }

        let disabled_here = hidden[index] || row_data.prefab.locked;
        let view = ViewState {
            hidden_own: row_data.scene_hidden,
            hidden_inherited: ancestor_flag(rows, index, |r| r.scene_hidden),
            locked_own: row_data.scene_locked,
            locked_inherited: ancestor_flag(rows, index, |r| r.scene_locked),
        };
        let (row, clicks) = self.row_widget(
            ui,
            row_data,
            depth,
            is_selected,
            if has_children { children[index].len() } else { 0 },
            collapsed,
            disabled_here,
            view,
        );
        let (eye_clicked, lock_clicked, arrow_clicked) = (clicks.eye, clicks.lock, clicks.arrow);
        if row_data.prefab.locked {
            // Outside the open prefab: drawn, but nothing happens when it is used.
            return;
        }
        if self.scroll_to == Some(index) {
            row.scroll_to_me(None);
            self.scroll_to = None;
        }
        // Ping flash: an accent outline and wash that fade out over ~1.6 s.
        if let Some((pinged, at)) = self.ping
            && pinged == index
        {
            const PING_SECS: f32 = 1.6;
            let t = at.elapsed().as_secs_f32();
            if t < PING_SECS {
                let fade = 1.0 - t / PING_SECS;
                let rect = row.rect.shrink2(egui::vec2(6.0, 1.0));
                ui.painter().rect_filled(rect, theme::R_ROW, theme::with_alpha(theme::accent(), 0.30 * fade));
                ui.painter().rect_stroke(
                    rect,
                    theme::R_ROW,
                    egui::Stroke::new(1.5_f32, theme::with_alpha(theme::accent(), fade)),
                    egui::StrokeKind::Inside,
                );
                ui.ctx().request_repaint();
            } else {
                self.ping = None;
            }
        }
        if eye_clicked || lock_clicked {
            // A click on a selected row applies to the whole selection.
            let targets: Vec<usize> = if is_selected && self.multi.len() > 1 { self.multi() } else { vec![index] };
            if eye_clicked {
                response.view = if ui.input(|i| i.modifiers.alt) {
                    Some(ViewRequest::Isolate(index))
                } else {
                    Some(ViewRequest::Hide(targets, !row_data.scene_hidden))
                };
            } else {
                response.view = Some(ViewRequest::Lock(targets, !row_data.scene_locked));
            }
            return;
        }
        if arrow_clicked {
            response.prefab = Some((index, PrefabMenu::Open));
            return;
        }
        // Where the caret column starts: the row's 6 px inset + 4 + the fixed eye/lock column,
        // the same x0 `row_widget` paints the caret at (the zone below ends 16 px after it).
        let indent = 6.0 + 4.0 + VIEW_COL_W + depth as f32 * theme::INDENT;

        row.dnd_set_drag_payload(index);
        // Also publish the dragged index into egui memory as a plain usize, so
        // ObjectRef drop boxes (incl. plugin inspectors with their own egui)
        // can read it without egui's DragAndDrop plugin. Cleared each frame the
        // pointer is up (engine, end of frame).
        if row.dragged() {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::DRAG_OBJECT_KEY), index));
            ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::DRAG_OBJECT_NAME_KEY), row_data.name.clone()));
            // Floating "ghost" of the dragged row, following the cursor, so the
            // drag has visual feedback (and the same chip lands in an ObjectRef
            // drop box). Painted on a top layer without any egui dnd context.
            if let Some(pos) = ui.ctx().pointer_interact_pos() {
                let painter = ui.ctx().layer_painter(egui::LayerId::new(
                    egui::Order::Tooltip,
                    egui::Id::new("zestengine-drag-ghost"),
                ));
                let font = egui::TextStyle::Body.resolve(ui.style());
                let galley =
                    painter.layout_no_wrap(row_data.name.clone(), font, egui::Color32::WHITE);
                let pad = egui::vec2(8.0, 4.0);
                let origin = pos + egui::vec2(14.0, 6.0);
                let rect = egui::Rect::from_min_size(origin, galley.size() + pad * 2.0);
                painter.rect_filled(rect, theme::R_ROW, theme::BG_SEGMENT_ACTIVE);
                painter.rect_stroke(rect, theme::R_ROW, egui::Stroke::new(1.0_f32, theme::accent()), egui::StrokeKind::Inside);
                painter.galley(origin + pad, galley, theme::TEXT_STRONG);
            }
        }
        if row.clicked() {
            let on_arrow = has_children
                && row
                    .interact_pointer_pos()
                    .is_some_and(|p| p.x < row.rect.left() + indent + 16.0);
            if on_arrow {
                let collapse = !self.collapsed.contains(&index);
                // Alt or Ctrl/Cmd: cascade the whole subtree; plain click
                // toggles one node.
                if ui.input(|i| i.modifiers.alt || i.modifiers.command) {
                    self.cascade(children, index, collapse);
                } else if collapse {
                    self.collapsed.insert(index);
                } else {
                    self.collapsed.remove(&index);
                }
            } else {
                let mods = ui.input(|i| i.modifiers);
                if mods.command {
                    // Ctrl/Cmd: toggle this row in/out of the multi-selection.
                    if self.multi.contains(&index) && self.multi.len() > 1 {
                        self.multi.remove(&index);
                        // Keep the anchor valid (any remaining member).
                        if *selected == Some(index) {
                            *selected = self.multi.iter().next().copied();
                        }
                    } else {
                        self.multi.insert(index);
                        *selected = Some(index);
                    }
                } else if mods.shift {
                    // Shift: select the index range between the anchor and here
                    // (scene-order approximation of "everything in between").
                    if let Some(a) = *selected {
                        let (lo, hi) = (a.min(index), a.max(index));
                        for i in lo..=hi {
                            self.multi.insert(i);
                        }
                    }
                    self.multi.insert(index);
                    *selected = Some(index);
                } else {
                    // Plain click: single-select (clear the multi-selection). A
                    // click on the lone selection toggles it off.
                    let only = self.multi.len() <= 1 && *selected == Some(index);
                    self.multi.clear();
                    if only {
                        *selected = None;
                    } else {
                        self.multi.insert(index);
                        *selected = Some(index);
                    }
                }
                response.selection_changed = true;
            }
        }
        // Drag-drop: top third inserts before (reorder), middle reparents,
        // bottom third inserts after (reorder).
        let cursor_y = ui.ctx().pointer_latest_pos().map(|p| p.y);
        if let Some(payload) = row.dnd_release_payload::<usize>() {
            let child = *payload;
            if child != index {
                let zone = cursor_y.map_or(DropZone::Onto, |y| drop_zone(row.rect, y));
                let parent = rows[index].parent;
                match zone {
                    DropZone::Onto => response.moves.push((child, Some(index), None)),
                    DropZone::Before => response.moves.push((child, parent, Some(index))),
                    DropZone::After => {
                        response
                            .moves
                            .push((child, parent, next_sibling(rows, index)))
                    }
                }
            }
        } else if let Some(path) = row.dnd_release_payload::<std::path::PathBuf>() {
            response.asset_drop = Some(((*path).clone(), Some(index)));
        } else if row.dnd_hover_payload::<std::path::PathBuf>().is_some() {
            ui.painter().rect_stroke(row.rect, 3.0, egui::Stroke::new(1.5_f32, theme::accent()), egui::StrokeKind::Outside);
        } else if row
            .dnd_hover_payload::<usize>()
            .is_some_and(|p| *p != index)
        {
            let color = theme::accent();
            let painter = ui.painter();
            match cursor_y.map_or(DropZone::Onto, |y| drop_zone(row.rect, y)) {
                DropZone::Onto => {
                    painter.rect_stroke(
                        row.rect,
                        3.0,
                        egui::Stroke::new(1.5_f32, color),
                        egui::StrokeKind::Outside,
                    );
                }
                DropZone::Before => {
                    painter.hline(
                        row.rect.left()..=row.rect.right(),
                        row.rect.top(),
                        egui::Stroke::new(2.5_f32, color),
                    );
                }
                DropZone::After => {
                    painter.hline(
                        row.rect.left()..=row.rect.right(),
                        row.rect.bottom(),
                        egui::Stroke::new(2.5_f32, color),
                    );
                }
            }
        }
        row.context_menu(|ui| {
            crate::widgets::menu_frame(ui);
            let selected: Vec<usize> = if self.multi.contains(&index) { self.multi.iter().copied().collect() } else { vec![index] };
            let hide_now = !rows[index].scene_hidden;
            if ui.button(if hide_now { "Hide in Scene View" } else { "Show in Scene View" }).on_hover_text("Scene view only (H); does not disable the object").clicked() {
                response.view = Some(ViewRequest::Hide(selected.clone(), hide_now));
                ui.close();
            }
            let lock_now = !rows[index].scene_locked;
            if ui.button(if lock_now { "Lock" } else { "Unlock" }).on_hover_text("Locked objects cannot be selected in the scene view").clicked() {
                response.view = Some(ViewRequest::Lock(selected.clone(), lock_now));
                ui.close();
            }
            ui.separator();
            if ui
                .button(format!("{} Group Into Empty", ph::FOLDER_SIMPLE_PLUS))
                .on_hover_text("Put the selected objects under a new empty at their centre")
                .clicked()
            {
                response.group = Some(selected);
                ui.close();
            }
            if rows[index].parent.is_some() && ui.button("Unparent").clicked() {
                response.reparent.push((index, None));
                ui.close();
            }
            if rows[index].parent.is_some()
                && ui
                    .button("Add Toggle")
                    .on_hover_text("Under an avatar: add a menu toggle that shows or hides this object")
                    .clicked()
            {
                response.add_toggle = Some(index);
                ui.close();
            }
            if ui
                .button(format!("{} Delete", ph::TRASH))
                .on_hover_text("Delete this object and its children")
                .clicked()
            {
                response.delete = Some(index);
                ui.close();
            }
            if ui
                .button("Edit textures in ZestSurface")
                .on_hover_text("Paint this object's textures in ZestSurface")
                .clicked()
            {
                response.edit_in_zestsurface = Some(index);
                ui.close();
            }
            let mark = rows[index].prefab;
            if mark.role != PrefabRole::None {
                ui.separator();
                ui.label(RichText::new("Prefab").small().weak());
                if matches!(mark.role, PrefabRole::Root | PrefabRole::NestedRoot) {
                    for (label, cmd) in [
                        ("Open Prefab", PrefabMenu::Open),
                        ("Open Prefab in Isolation", PrefabMenu::OpenIsolated),
                        ("Select Prefab Asset", PrefabMenu::SelectAsset),
                    ] {
                        if widgets::menu_item(ui, None, label, None).clicked() {
                            response.prefab = Some((index, cmd));
                            ui.close();
                        }
                    }
                    if mark.role == PrefabRole::Root && mark.changed {
                        for (label, cmd) in [("Revert All Overrides", PrefabMenu::RevertAll), ("Apply All Overrides", PrefabMenu::ApplyAll)] {
                            if widgets::menu_item(ui, None, label, None).clicked() {
                                response.prefab = Some((index, cmd));
                                ui.close();
                            }
                        }
                    }
                    for (label, cmd) in [("Unpack", PrefabMenu::Unpack), ("Unpack Completely", PrefabMenu::UnpackCompletely)] {
                        if widgets::menu_item(ui, None, label, None).clicked() {
                            response.prefab = Some((index, cmd));
                            ui.close();
                        }
                    }
                } else if widgets::menu_item(ui, None, "Select Prefab Root", None).clicked() {
                    response.prefab = Some((index, PrefabMenu::SelectRoot));
                    ui.close();
                }
            }
            ui.separator();
            spawn_menu(ui, response);
        });
    }
}

/// Pull the objects attached to a bone out of their parent's child list: `(root row, bone)` ->
/// the attached rows. Only a parent that lists an armature with that bone qualifies; any other
/// Bone Attach object stays an ordinary child.
fn split_attached(rows: &[SceneObjectRow], children: &mut [Vec<usize>], armatures: &HashMap<usize, ArmatureTree>) -> HashMap<(usize, usize), Vec<usize>> {
    let mut out: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        let (Some(name), Some(root)) = (&row.bone, row.parent) else { continue };
        let Some(bone) = armatures.get(&root).and_then(|t| t.bones.iter().position(|(n, _)| n == name)) else { continue };
        if let Some(list) = children.get_mut(root) {
            list.retain(|&c| c != i);
        }
        out.entry((root, bone)).or_default().push(i);
    }
    out
}

/// Width of the fixed left column holding the eye and lock buttons.
const VIEW_COL_W: f32 = 40.0;

/// Own + inherited scene-view state of a row, for drawing.
#[derive(Clone, Copy, Default)]
struct ViewState {
    hidden_own: bool,
    hidden_inherited: bool,
    locked_own: bool,
    locked_inherited: bool,
}

#[derive(Clone, Copy, Default)]
struct RowClicks {
    eye: bool,
    lock: bool,
    arrow: bool,
}

/// Whether `pick` holds for any ancestor of row `index` (not the row itself).
pub fn ancestor_flag(rows: &[SceneObjectRow], index: usize, pick: impl Fn(&SceneObjectRow) -> bool) -> bool {
    let mut k = rows.get(index).and_then(|r| r.parent);
    let mut depth = 0;
    while let Some(j) = k.filter(|&j| j < rows.len()) {
        if pick(&rows[j]) {
            return true;
        }
        depth += 1;
        if depth > 64 {
            break;
        }
        k = rows[j].parent;
    }
    false
}

/// Rows disabled by their own `enabled` flag or any disabled ancestor.
/// Independent of the scene-view hide / lock flags.
fn hidden_flags(rows: &[SceneObjectRow]) -> Vec<bool> {
    (0..rows.len()).map(|i| !rows[i].enabled || ancestor_flag(rows, i, |r| !r.enabled)).collect()
}

/// Rows hidden / locked in the scene view by their own flag or any ancestor's.
pub fn view_flags(rows: &[SceneObjectRow]) -> (Vec<bool>, Vec<bool>) {
    let hidden = (0..rows.len()).map(|i| rows[i].scene_hidden || ancestor_flag(rows, i, |r| r.scene_hidden)).collect();
    let locked = (0..rows.len()).map(|i| rows[i].scene_locked || ancestor_flag(rows, i, |r| r.scene_locked)).collect();
    (hidden, locked)
}

#[cfg(test)]
mod view_tests {
    use super::*;

    fn row(name: &str, parent: Option<usize>) -> SceneObjectRow {
        SceneObjectRow {
            name: name.into(),
            parent,
            kind: ObjectKind::Mesh,
            enabled: true,
            scene_hidden: false,
            scene_locked: false,
            prefab: Default::default(),
            net_role: zestengine_core::NetRole::Both,
            net_inherited: zestengine_core::NetRole::Both,
            bone: None,
        }
    }

    fn tree(bones: &[(&str, Option<usize>)]) -> ArmatureTree {
        let bones: Vec<(String, Option<usize>)> = bones.iter().map(|(n, p)| (n.to_string(), *p)).collect();
        let mut children = vec![Vec::new(); bones.len()];
        for (i, (_, p)) in bones.iter().enumerate() {
            if let Some(p) = p {
                children[*p].push(i);
            }
        }
        ArmatureTree { key: (0, bones.len()), bones, children }
    }

    #[test]
    fn attached_objects_leave_the_parents_children_for_their_bone() {
        let mut rows = vec![row("avatar", None), row("body", Some(0)), row("sword", Some(0)), row("orphan", Some(1)), row("other", None)];
        rows[2].bone = Some("hand".into());
        rows[3].bone = Some("hand".into()); // parent has no armature: stays a plain child
        rows[4].bone = Some("hand".into()); // no parent
        let mut children = vec![vec![1, 2], vec![3], vec![], vec![], vec![]];
        let armatures = HashMap::from([(0, tree(&[("hips", None), ("hand", Some(0))]))]);
        let kids = split_attached(&rows, &mut children, &armatures);
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[&(0, 1)], [2]);
        assert_eq!(children[0], [1], "indices of real objects stay real");
        assert_eq!(children[1], [3]);
        // A bone the skeleton does not have: still a plain child.
        rows[2].bone = Some("gone".into());
        let mut children = vec![vec![1, 2], vec![3], vec![], vec![], vec![]];
        assert!(split_attached(&rows, &mut children, &armatures).is_empty());
        assert_eq!(children[0], [1, 2]);
    }

    #[test]
    fn opening_a_bone_opens_the_bones_above_it() {
        let mut panel = ScenePanel::new();
        panel.armatures.insert(0, tree(&[("root", None), ("spine", Some(0)), ("hand", Some(1))]));
        panel.open_bone(0, 2);
        assert!(panel.armature_open.is_superset(&HashSet::from([(0, 0), (0, 1), (0, 2)])));
        panel.bone_kids.insert((0, 2), vec![7]);
        assert_eq!(panel.visible_attached(0), [7]);
        panel.armature_open.remove(&(0, 1));
        assert!(panel.visible_attached(0).is_empty(), "a collapsed bone above hides it");
    }

    #[test]
    fn hidden_locked_and_disabled_are_independent_and_inherit() {
        let mut rows = vec![row("a", None), row("b", Some(0)), row("c", Some(1)), row("d", None)];
        rows[0].scene_hidden = true;
        rows[1].scene_locked = true;
        rows[3].enabled = false;
        let (hidden, locked) = view_flags(&rows);
        assert_eq!(hidden, [true, true, true, false]);
        assert_eq!(locked, [false, true, true, false]);
        assert_eq!(hidden_flags(&rows), [false, false, false, true], "disabled is its own thing");
        rows[0].enabled = false;
        assert_eq!(hidden_flags(&rows), [true, true, true, true]);
        assert_eq!(view_flags(&rows).0, [true, true, true, false], "disabling does not change hide");
    }

    #[test]
    fn toggle_hides_all_when_mixed_and_shows_when_all_hidden() {
        let mut rows = vec![row("a", None), row("b", None)];
        rows[0].scene_hidden = true;
        assert_eq!(toggle_hide_request(&rows, &[0, 1]), Some(ViewRequest::Hide(vec![0, 1], true)));
        rows[1].scene_hidden = true;
        assert_eq!(toggle_hide_request(&rows, &[0, 1]), Some(ViewRequest::Hide(vec![0, 1], false)));
        assert_eq!(toggle_hide_request(&rows, &[]), None);
        assert_eq!(toggle_hide_request(&rows, &[9]), None);
    }

    #[test]
    fn h_key_routing() {
        // key, shift, alt, ctrl, text_editing, playing, focus_ok
        let k = |sh, alt, ctrl, text, play, focus| view_key_action(true, sh, alt, ctrl, text, play, focus);
        assert_eq!(k(false, false, false, false, false, true), Some(ViewKeyAction::ToggleHide));
        assert_eq!(k(true, false, false, false, false, true), Some(ViewKeyAction::ShowAll));
        assert_eq!(k(false, true, false, false, false, true), Some(ViewKeyAction::ShowAll));
        assert_eq!(k(false, false, true, false, false, true), None, "Ctrl+H is not ours");
        assert_eq!(k(false, false, false, true, false, true), None, "typing in a text field");
        assert_eq!(k(false, false, false, false, true, true), None, "not during Play");
        assert_eq!(k(false, false, false, false, false, false), None, "needs Scene view / Hierarchy focus");
        assert_eq!(view_key_action(false, false, false, false, false, false, true), None);
    }
}
