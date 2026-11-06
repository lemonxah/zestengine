//! Editor-side component UI: the [`Inspect`] (inspector) and [`Gizmo`]
//! (viewport drawing) traits, the [`EditorComponents`] dispatch registry that
//! maps a component name to its inspector/gizmo, the built-in impls, and the
//! `components_ui` inspector list.
//!
//! The runtime component API (the `Component`/`TypedComponent` traits, the
//! registry, the component structs, `ComponentCtx`, `Transform`) lives in
//! `zestengine-core` and is egui-free. This module is editor-only; nothing here
//! ships in a built game.

use std::collections::HashMap;

use zestengine_core::{
    Animator, JointKind, PortalMode, ScenePortal, NavAgent, NavMeshSurface, ParticleEmitter, PhysicsJoint, AudioListener, AudioRolloff, AudioSource, AcousticMaterial, AcousticPreset, AudioProbeVolume, Bob, BodyKind, BoxCollider, CameraComponent,
    ai::{BehaviourTreeAgent, PerceptionComponent},
    Character, Component, UiAlign, UiButton, UiCanvas, UiImage, UiPanel, UiRect, UiRenderMode,
    UiScaleMode, UiSlider, UiText, UiTextField, UiTextOverflow, UiTextRenderer, UiToggle, UiAction, Text3D, Text3DBillboard, UiFillDirection, UiPauseMenu, UiProgressBar, ComponentRegistry, ControlMode, FluxVolume, FluxVoxelLight, LightComponent, LightKind,
    AreaLight, LightMode, LightProbeVolume, LodComponent, LodLevel, MeshCollider, ObjectId, ObjectRef, Pawn,
    PlanarReflection, ReflectionProbe, RigidBody, ShadowType,
    Spin, AvatarDescriptor, CapsuleCollider, ColliderAxis, DecalProjector,
    SpawnPoint, SphereCollider, Sync, Terrain, TerrainCollider, TypedComponent, VolumeComponent,
    COLLISION_LAYERS,
};
use egui::{DragValue, RichText, ScrollArea, Sense, Ui};
use egui_phosphor::regular as ph;

#[path = "net_inspect.rs"]
mod net_inspect;

#[path = "physics2d_ui.rs"]
mod physics2d_ui;
#[path = "audio_volume_ui.rs"]
mod audio_volume_ui;

/// egui-memory key under which the Scene tree publishes the index of the object
/// currently being dragged (a `usize`, so its slot is shared across the
/// plugin/egui dylib boundary). `ObjectRef` drop targets read it.
pub const DRAG_OBJECT_KEY: &str = "zestengine-drag-object";
/// The name of the object being dragged (a `String`, published beside
/// [`DRAG_OBJECT_KEY`]) for drop targets that match by name, like the rig
/// Configure window's bone rows.
pub const DRAG_OBJECT_NAME_KEY: &str = "zestengine-drag-object-name";
/// egui-memory key holding the project-relative path of the file currently
/// dragged from the file browser (a `String`, shared across the dylib boundary;
/// same plugin-safe pattern as [`DRAG_OBJECT_KEY`]). Drop targets read it.
pub const DRAG_FILE_KEY: &str = "zestengine-drag-file";

/// egui-memory keys (`usize`, an index into the scene's object list) an
/// `ObjectRef` chip writes: a click pings the object in the Scene tree
/// (reveal + flash, the inspector stays put), a double click selects it. Same
/// plugin-safe pattern as [`DRAG_OBJECT_KEY`].
pub const PING_OBJECT_KEY: &str = "zestengine-ping-object";
pub const SELECT_OBJECT_KEY: &str = "zestengine-select-object";

/// egui-memory key the Animator inspector writes a `.animctrl` path into
/// (a `String`) to ask the editor to open it in the Animator Controller tab.
pub const OPEN_ANIM_CONTROLLER_KEY: &str = "zestengine-open-animctrl";

/// egui-memory key (`bool`) a Weapon Hold sets to open the dedicated pose
/// editor window.
pub const OPEN_POSE_EDITOR_KEY: &str = "zestengine-open-pose-editor";

/// egui-memory key the editor puts the multi-selection's differing fields
/// under: `Vec<(component name, Vec<normalized field name>)>` (see
/// `widgets::mixed_key`). Rows whose label matches show an em dash.
pub const MIXED_COMPONENTS_KEY: &str = "zestengine-mixed-components";

/// Context handed to a component's inspector: the scene's object list (for
/// `ObjectRef` pickers) plus a helper to draw the picker.
pub struct InspectCtx<'a> {
    /// (id, display name) of every object in the scene.
    pub objects: &'a [(ObjectId, String)],
    /// Per-layer display names (len = NUM_LAYERS). Lets layer pickers (camera
    /// culling mask) show the SAME layer set + names as the Layers window and the
    /// object's layer dropdown, instead of a hardcoded count.
    pub layer_names: &'a [String],
    /// Edits the asset file at a project-relative path in place (physics
    /// materials, post profiles), so asset slots can show it under the
    /// field. Returns None for files it can't edit.
    pub asset_editor: Option<&'a AssetEditor<'a>>,
    /// The object has a Rigid Body (its colliders take the body's physics
    /// material unless they override it).
    pub has_body: bool,
}

/// See [`InspectCtx::asset_editor`]: draw the editor for the file at the
/// path; Some(true) when it changed (and was saved).
pub type AssetEditor<'a> = dyn Fn(&mut Ui, &std::path::Path) -> Option<bool> + 'a;

impl InspectCtx<'_> {
    /// Draw an object-reference row as a drop target: drag an object from
    /// the Scene tree onto the box to set it; the ✕ clears it. Returns true if
    /// the reference changed.
    ///
    /// Deliberately avoids egui's drag-and-drop API: that goes through egui's
    /// `DragAndDrop` context plugin, which panics when called from a plugin's
    /// own (separately-linked) egui. Same `TypeId`-mismatch class as the
    /// label-selection crash. Instead the Scene tree publishes the dragged
    /// object index into egui memory as a `usize` (a std type, so its slot is
    /// shared across the dylib boundary), and we detect the drop with raw
    /// pointer state.
    pub fn object_ref(&self, ui: &mut Ui, label: &str, target: &mut ObjectRef) -> bool {
        self.object_ref_with_hint(ui, label, target, "(drag an object here)")
    }

    /// Object-reference picker with a field-specific empty-state hint.
    pub fn object_ref_with_hint(&self, ui: &mut Ui, label: &str, target: &mut ObjectRef, hint: &str) -> bool {
        let mut changed = false;
        let name = target
            .id()
            .and_then(|id| self.objects.iter().find(|(i, _)| *i == id))
            .map(|(_, n)| n.clone());
        let text = match &name {
            Some(n) => n.clone(),
            None if target.is_set() => "(missing)".to_owned(),
            None => hint.to_owned(),
        };
        let dragging: Option<usize> = ui.data(|d| d.get_temp(egui::Id::new(DRAG_OBJECT_KEY)));
        crate::widgets::property_row(ui, label, |ui| {
            let inner = egui::Frame::group(ui.style())
                .inner_margin(egui::Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.set_min_width(120.0);
                    let mut t = RichText::new(text);
                    if name.is_none() {
                        t = t.weak();
                    }
                    ui.label(t);
                });
            let rect = inner.response.rect;
            let hovering = ui.rect_contains_pointer(rect);
            // Click the chip to find the object in the Scene tree; double-click selects it.
            if let Some(index) = target.id().and_then(|id| self.objects.iter().position(|(i, _)| *i == id)) {
                let chip = ui
                    .interact(rect, ui.id().with(("objref-select", label)), egui::Sense::click())
                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                    .on_hover_text("Click: show in the Scene tree. Double-click: select");
                if chip.double_clicked() {
                    ui.data_mut(|d| d.insert_temp(egui::Id::new(SELECT_OBJECT_KEY), index));
                } else if chip.clicked() {
                    ui.data_mut(|d| d.insert_temp(egui::Id::new(PING_OBJECT_KEY), index));
                }
            }
            // Highlight while a drag is in flight and the pointer is over us.
            if dragging.is_some() && hovering {
                ui.painter().rect_stroke(
                    rect,
                    4.0,
                    egui::Stroke::new(2.0_f32, ui.visuals().selection.stroke.color),
                    egui::StrokeKind::Inside,
                );
            }
            // Drop = pointer released over the box with a dragged object pending.
            if hovering
                && ui.input(|i| i.pointer.any_released())
                && let Some(idx) = dragging
                && let Some((id, _)) = self.objects.get(idx)
            {
                *target = ObjectRef::new(*id);
                changed = true;
            }
            if target.is_set() && ui.small_button(ph::X).on_hover_text("Clear").clicked() {
                *target = ObjectRef::NONE;
                changed = true;
            }
        });
        changed
    }

    /// An asset slot (see [`asset_chip`]) on a labelled row, with the
    /// assigned asset's own editor inline under it when there is one.
    /// `exts` (lowercase, no dot) restricts which files fit; empty = any.
    /// Returns true if the reference changed.
    pub fn file_field(
        &self,
        ui: &mut Ui,
        label: &str,
        value: &mut String,
        hint: &str,
        exts: &[&str],
    ) -> bool {
        let changed = crate::widgets::property_row(ui, label, |ui| asset_chip(ui, label, value, hint, exts));
        // The assigned asset's own settings, editable right here.
        if let Some(editor) = self.asset_editor
            && let Some(rel) = resolve_slot(ui, value)
        {
            let path = std::path::PathBuf::from(rel);
            let id = ui.make_persistent_id(("zestengine-inline-asset", label));
            let name = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
            let open = state.is_open();
            let header = ui.horizontal(|ui| {
                ui.add_space(crate::theme::LABEL_WIDTH - 14.0);
                let text = if open { format!("Hide {name}") } else { format!("Edit {name}") };
                ui.add(egui::Label::new(RichText::new(text).size(crate::theme::SIZE_META).color(crate::theme::ACCENT_HOVER)).sense(Sense::click()))
                    .on_hover_cursor(egui::CursorIcon::PointingHand)
            });
            if header.inner.clicked() {
                state.toggle(ui);
            }
            state.store(ui.ctx());
            if state.is_open() {
                egui::Frame::new()
                    .inner_margin(egui::Margin { left: 10, right: 0, top: 2, bottom: 6 })
                    .stroke(egui::Stroke::new(1.0_f32, crate::theme::LINE_PANEL))
                    .corner_radius(crate::theme::R_FIELD)
                    .show(ui, |ui| {
                        ui.push_id(id, |ui| {
                            if editor(ui, &path).is_none() {
                                ui.label(RichText::new("Can't open this file").size(crate::theme::SIZE_META).color(crate::theme::TEXT_DIM));
                            }
                        });
                    });
            }
        }
        changed
    }
}

/// egui temp data (`String`, absolute path): an asset slot was clicked, so
/// the file browser shows and highlights that file.
pub const PING_FILE_KEY: &str = "zestengine-ping-file";
/// egui temp data (`String`, absolute path): an asset slot was double-clicked,
/// so the editor selects that file in the inspector.
pub const OPEN_FILE_KEY: &str = "zestengine-open-file";

/// The project-relative file a slot value names (`@<id>` or a path), or
/// None when it's empty or the id is gone.
pub fn resolve_slot(ui: &Ui, value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    match project_root(ui) {
        Some(root) => zestengine_core::asset_ref::resolve(std::path::Path::new(&root), value),
        None => (!zestengine_core::asset_ref::is_ref(value)).then(|| value.to_owned()),
    }
}

/// A reference to project-relative `rel` for storing in a slot: its `@<id>`
/// (so renames and moves keep it), or the path without a project.
fn slot_reference(ui: &Ui, rel: &str) -> String {
    match project_root(ui) {
        Some(root) => zestengine_core::asset_ref::reference(std::path::Path::new(&root), rel),
        None => rel.to_owned(),
    }
}

/// An asset slot: a read-only chip showing the asset's icon and name. Drag a
/// file from the file browser onto it, pick one with the browse button, or
/// make a new one with [+]. Click it to show the file in the file browser,
/// double-click to open it in the inspector. Stores an `@<id>` reference so
/// renaming or moving the file keeps it assigned. True when it changed.
pub fn asset_chip(ui: &mut Ui, salt: &str, value: &mut String, hint: &str, exts: &[&str]) -> bool {
    let mut changed = false;
    let dragging: Option<String> = ui.data(|d| d.get_temp(egui::Id::new(DRAG_FILE_KEY)));
    let droppable = dragging.as_deref().is_some_and(|p| ext_fits(exts, p));
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if let Some(rel) = asset_new_button(ui, exts) {
            *value = slot_reference(ui, &rel);
            changed = true;
        }
        let current = resolve_slot(ui, value).unwrap_or_default();
        if let Some(picked) = asset_browse_button(ui, salt, exts, &current) {
            *value = if picked.is_empty() { String::new() } else { slot_reference(ui, &picked) };
            changed = true;
        }
        let resolved = resolve_slot(ui, value);
        let missing = !value.trim().is_empty() && resolved.is_none();
        let (glyph, name, color) = match &resolved {
            Some(rel) => {
                let p = std::path::Path::new(rel);
                let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                (crate::file_browser::file_glyph(p), name, crate::theme::TEXT_STRONG)
            }
            None if missing => (egui_phosphor::regular::WARNING, "Missing".to_owned(), crate::theme::STATUS_WARN_FG),
            None => ("", format!("None ({hint})"), crate::theme::TEXT_DIM),
        };
        let size = egui::vec2(ui.available_width(), crate::theme::H_FIELD);
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
        let hovered = resp.hovered() || (droppable && ui.rect_contains_pointer(rect));
        ui.painter().rect(
            rect,
            crate::theme::R_FIELD,
            crate::theme::BG_FIELD,
            egui::Stroke::new(1.0_f32, if hovered { crate::theme::accent() } else { crate::theme::LINE_FIELD }),
            egui::StrokeKind::Inside,
        );
        let mut x = rect.left() + 8.0;
        if !glyph.is_empty() {
            ui.painter().text(egui::pos2(x, rect.center().y), egui::Align2::LEFT_CENTER, glyph, egui::FontId::proportional(13.0), crate::theme::TEXT_MUTED);
            x += 18.0;
        }
        let galley = ui.painter().layout_no_wrap(name, egui::FontId::proportional(crate::theme::SIZE_BODY), color);
        let clip = egui::Rect::from_min_max(egui::pos2(x, rect.top()), egui::pos2(rect.right() - 6.0, rect.bottom()));
        ui.painter().with_clip_rect(clip).galley(egui::pos2(x, rect.center().y - galley.size().y * 0.5), galley, color);
        let resp = match &resolved {
            Some(rel) => resp.on_hover_text(format!("{rel}\nClick: show in Files. Double-click: open in the Inspector")),
            None => resp,
        };
        if let (Some(rel), Some(root)) = (&resolved, project_root(ui)) {
            let abs = std::path::Path::new(&root).join(rel).to_string_lossy().into_owned();
            if resp.double_clicked() {
                ui.data_mut(|d| d.insert_temp(egui::Id::new(OPEN_FILE_KEY), abs));
            } else if resp.clicked() {
                ui.data_mut(|d| d.insert_temp(egui::Id::new(PING_FILE_KEY), abs));
            }
        }
        if droppable
            && ui.rect_contains_pointer(rect)
            && ui.input(|i| i.pointer.any_released())
            && let Some(p) = &dragging
        {
            *value = slot_reference(ui, p);
            changed = true;
        }
    });
    changed
}

/// egui temp data (`String`): the project root, published by the editor each
/// frame so asset fields can list and create files.
pub const PROJECT_ROOT_KEY: &str = "zestengine-project-root";

/// Egui temp-data key (bool) the editor sets while Play is running:
/// components can only be switched on or off in edit mode.
pub const PLAY_MODE_KEY: &str = "zestengine-play-mode";
/// egui temp data (`Vec<AssetTemplate>`): asset kinds an asset field's [+]
/// can create.
pub const ASSET_TEMPLATES_KEY: &str = "zestengine-asset-templates";
/// egui temp data (`Option<(usize, String)>`): the inspected object's authored
/// collision pieces (count, prefixes in use), published by the editor each
/// frame for the Mesh Collider inspector.
pub const AUTHORED_COLLISION_KEY: &str = "zestengine-authored-collision";

/// A new-asset recipe: written to `folder/New <name>.<ext>` (numbered when
/// taken).
#[derive(Clone, Debug)]
pub struct AssetTemplate {
    pub ext: String,
    pub name: String,
    /// Project-relative folder.
    pub folder: String,
    pub contents: String,
}

fn project_root(ui: &Ui) -> Option<String> {
    ui.data(|d| d.get_temp(egui::Id::new(PROJECT_ROOT_KEY)))
}

fn ext_fits(exts: &[&str], p: &str) -> bool {
    exts.is_empty()
        || std::path::Path::new(p)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| exts.contains(&e.to_ascii_lowercase().as_str()))
}

/// A browse button listing the project's files that fit `exts`; returns the
/// picked project-relative path ("" = None). Needs [`PROJECT_ROOT_KEY`].
pub fn asset_browse_button(ui: &mut Ui, salt: &str, exts: &[&str], current: &str) -> Option<String> {
    let root = project_root(ui)?;
    let list_id = ui.make_persistent_id(("zestengine-asset-browse", salt));
    let browse = crate::widgets::icon_button(ui, crate::icons::Icon::Folder, "Browse", false);
    let mut picked = None;
    let shown = egui::Popup::menu(&browse)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            // Listed once per opening.
            let files: Vec<String> = ui.data_mut(|d| {
                d.get_temp_mut_or_insert_with(list_id, || {
                    let mut out = Vec::new();
                    let r = std::path::Path::new(&root);
                    list_project_files(r, r, &|p| ext_fits(exts, p), &mut out);
                    out.sort();
                    out
                })
                .clone()
            });
            ui.set_min_width(220.0);
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                if ui.selectable_label(current.is_empty(), "None").clicked() {
                    picked = Some(String::new());
                    ui.close();
                }
                if files.is_empty() {
                    ui.label(RichText::new("No matching files in the project").weak());
                }
                for f in files {
                    let name = std::path::Path::new(&f).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if ui.selectable_label(current == f, name).on_hover_text(&f).clicked() {
                        picked = Some(f);
                        ui.close();
                    }
                }
            });
        });
    if shown.is_none() {
        ui.data_mut(|d| d.remove::<Vec<String>>(list_id));
    }
    picked
}

/// A [+] button that makes a new asset of the first of `exts` from the
/// editor's templates ([`ASSET_TEMPLATES_KEY`]); returns its project-relative
/// path. Draws nothing when there's no template for that kind.
pub fn asset_new_button(ui: &mut Ui, exts: &[&str]) -> Option<String> {
    let root = project_root(ui)?;
    let t = ui
        .data(|d| d.get_temp::<Vec<AssetTemplate>>(egui::Id::new(ASSET_TEMPLATES_KEY)))
        .and_then(|all| all.into_iter().find(|t| exts.first() == Some(&t.ext.as_str())))?;
    let new = crate::widgets::icon_button(ui, crate::icons::Icon::Plus, &format!("New {}", t.name), false);
    if new.clicked() { create_from_template(&root, &t) } else { None }
}

fn create_from_template(root: &str, t: &AssetTemplate) -> Option<String> {
    let dir = std::path::Path::new(root).join(&t.folder);
    let mut n = 1;
    let path = loop {
        let file = if n == 1 { format!("New {}.{}", t.name, t.ext) } else { format!("New {} ({n}).{}", t.name, t.ext) };
        let p = dir.join(file);
        if !p.exists() {
            break p;
        }
        n += 1;
    };
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::write(&path, &t.contents).ok()?;
    Some(path.strip_prefix(root).ok()?.to_string_lossy().replace('\\', "/"))
}

fn list_project_files(root: &std::path::Path, dir: &std::path::Path, fits: &dyn Fn(&str) -> bool, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            list_project_files(root, &path, fits, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            let rel = rel.to_string_lossy().replace('\\', "/");
            if fits(&rel) {
                out.push(rel);
            }
        }
    }
}

/// Editor inspector for a component: draw widgets, return true if anything
/// changed. Implemented per component in the editor (or a plugin), keeping the
/// runtime `Component` egui-free.
pub trait Inspect {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool;
}

/// A viewport gizmo a component declares. The editor owns drawing + interaction
/// for each kind (wireframe box with resize handles, an interior point cloud, a
/// range sphere); a component just lists which it wants. Coordinates are in the
/// owning object's LOCAL space (the editor applies the object world transform).
#[derive(Clone)]
pub enum GizmoSpec {
    /// Resizable box. `object_anchored` = box is centered on the object origin
    /// (resize moves the OBJECT to keep the opposite face fixed); otherwise it
    /// has its own local `center` (resize moves that center, e.g. a collider).
    /// `blend` draws a dimmer shell at that distance: positive = inside the box
    /// (reflection-probe blend), negative = outside it (post volume fade-in); 0 = none.
    Box {
        center: glam::Vec3,
        half_extents: glam::Vec3,
        object_anchored: bool,
        blend: f32,
        color: egui::Color32,
    },
    /// A cloud of interior points (probe grid), display-only.
    Points {
        positions: Vec<glam::Vec3>,
        color: egui::Color32,
    },
    /// A resizable range sphere (radius handle on the camera-right side).
    Range {
        center: glam::Vec3,
        radius: f32,
        color: egui::Color32,
    },
    /// A draggable point (moved in the camera-facing plane), drawn as a ring
    /// with a dashed line to `anchor` (e.g. a probe's capture point).
    Handle {
        position: glam::Vec3,
        anchor: glam::Vec3,
        color: egui::Color32,
    },
    /// Display-only line segments (capsules, axes, arcs, paths). Local space
    /// unless `world` is set (e.g. a nav agent's planned path).
    Lines {
        segments: Vec<(glam::Vec3, glam::Vec3)>,
        color: egui::Color32,
        world: bool,
    },
}

/// Circle polyline around `center` in the plane spanned by `a`, `b` (unit).
#[allow(clippy::too_many_arguments)]
fn circle_segments(out: &mut Vec<(glam::Vec3, glam::Vec3)>, center: glam::Vec3, a: glam::Vec3, b: glam::Vec3, r: f32, from: f32, to: f32, seg: usize) {
    let mut prev = center + (a * from.cos() + b * from.sin()) * r;
    for k in 1..=seg {
        let t = from + (to - from) * k as f32 / seg as f32;
        let q = center + (a * t.cos() + b * t.sin()) * r;
        out.push((prev, q));
        prev = q;
    }
}

/// Two unit vectors perpendicular to `n` and each other.
fn basis_of(n: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let n = n.normalize_or(glam::Vec3::Y);
    let up = if n.y.abs() < 0.99 { glam::Vec3::Y } else { glam::Vec3::X };
    let a = up.cross(n).normalize();
    (a, n.cross(a))
}

/// Wire capsule along +Y: rings at both cap joints, four side lines and the
/// two hemispheres' meridians.
fn capsule_segments(center: glam::Vec3, radius: f32, height: f32) -> Vec<(glam::Vec3, glam::Vec3)> {
    use glam::Vec3;
    use std::f32::consts::PI;
    let half = (height * 0.5 - radius).max(0.0);
    let (top, bot) = (center + Vec3::Y * half, center - Vec3::Y * half);
    let mut s = Vec::new();
    circle_segments(&mut s, top, Vec3::X, Vec3::Z, radius, 0.0, 2.0 * PI, 32);
    circle_segments(&mut s, bot, Vec3::X, Vec3::Z, radius, 0.0, 2.0 * PI, 32);
    for d in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
        s.push((top + d * radius, bot + d * radius));
    }
    for d in [Vec3::X, Vec3::Z] {
        circle_segments(&mut s, top, d, Vec3::Y, radius, 0.0, PI, 16);
        circle_segments(&mut s, bot, d, Vec3::Y, radius, PI, 2.0 * PI, 16);
    }
    s
}

/// The result of dragging a gizmo handle, applied back to the component via
/// [`Gizmo::apply_gizmo_edit`]. `index` is the position in the component's
/// `gizmos()` list. Values are in the object's LOCAL space.
#[derive(Clone, Copy, Debug)]
pub enum GizmoEdit {
    /// Box resize: new half-extents (+ new local center for non-object-anchored
    /// boxes; object-anchored boxes get `center` unchanged and the editor moves
    /// the object instead).
    Box {
        index: usize,
        half_extents: glam::Vec3,
        center: glam::Vec3,
    },
    /// Range-sphere resize: new radius.
    Range { index: usize, radius: f32 },
    /// Point handle moved: new local position.
    Handle { index: usize, position: glam::Vec3 },
}

/// Editor viewport gizmos for a component. Default = none. A component lists the
/// gizmos it wants via [`gizmos`](Gizmo::gizmos) and writes back resize edits via
/// [`apply_gizmo_edit`](Gizmo::apply_gizmo_edit). Plugins implement this to get
/// the same handles as the built-ins.
pub trait Gizmo {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        Vec::new()
    }
    fn apply_gizmo_edit(&mut self, _edit: GizmoEdit) {}
}

/// Probe/voxel point cloud for a `Points` gizmo, capped for display: strided
/// down to a budget so a DENSE volume still shows a representative grid instead
/// of nothing (the old hard cap just hid all points past ~4096). `None` only for
/// an empty or absurdly large grid (generation guard). `positions` is generated
/// lazily so the guard avoids building a huge vec.
fn display_points(count: usize, positions: impl FnOnce() -> Vec<glam::Vec3>) -> Option<Vec<glam::Vec3>> {
    const GEN_CAP: usize = 300_000; // don't even build a vec larger than this
    const BUDGET: usize = 16_384; // max points actually drawn
    if count == 0 || count > GEN_CAP {
        return None;
    }
    let mut p = positions();
    if p.len() > BUDGET {
        let stride = p.len().div_ceil(BUDGET);
        p = p.into_iter().step_by(stride).collect();
    }
    Some(p)
}

type InspectFn = fn(&mut dyn Component, &mut Ui, &InspectCtx) -> bool;
type GizmosFn = fn(&dyn Component) -> Vec<GizmoSpec>;
type GizmoEditFn = fn(&mut dyn Component, GizmoEdit);

/// Editor-side dispatch: component name -> inspector / gizmo. Built-ins
/// register in [`Self::with_builtins`]; plugins register through their
/// `zestengine_register_editor` export.
#[derive(Default)]
pub struct EditorComponents {
    inspect: HashMap<&'static str, InspectFn>,
    gizmos: HashMap<&'static str, GizmosFn>,
    gizmo_edit: HashMap<&'static str, GizmoEditFn>,
}

impl EditorComponents {
    pub fn with_builtins() -> Self {
        let mut e = Self::default();
        e.register::<CameraComponent>();
        e.register::<LightComponent>();
        e.register::<AreaLight>();
        e.register::<LightProbeVolume>();
        e.register::<ReflectionProbe>();
        e.register::<PlanarReflection>();
        e.register::<DecalProjector>();
        e.register::<FluxVoxelLight>();
        e.register::<FluxVolume>();
        e.register::<AudioSource>();
        e.register::<AudioListener>();
        e.register::<zestengine_core::AudioVolume>();
        e.register::<AcousticMaterial>();
        e.register::<AudioProbeVolume>();
        e.register::<BoxCollider>();
        e.register::<SphereCollider>();
        e.register::<CapsuleCollider>();
        e.register::<MeshCollider>();
        e.register::<Terrain>();
        e.register::<TerrainCollider>();
        e.register::<Spin>();
        e.register::<Bob>();
        e.register::<LodComponent>();
        e.register::<RigidBody>();
        e.register::<Character>();
        e.register::<Animator>();
        e.register::<PhysicsJoint>();
        e.register::<zestengine_core::Rigidbody2D>();
        e.register::<zestengine_core::BoxCollider2D>();
        e.register::<zestengine_core::CircleCollider2D>();
        e.register::<zestengine_core::CapsuleCollider2D>();
        e.register::<zestengine_core::PolygonCollider2D>();
        e.register::<zestengine_core::EdgeCollider2D>();
        e.register::<zestengine_core::CompositeCollider2D>();
        e.register::<zestengine_core::Joint2D>();
        e.register::<zestengine_core::Effector2D>();
        e.register::<NavMeshSurface>();
        e.register::<NavAgent>();
        e.register::<PerceptionComponent>();
        e.register::<BehaviourTreeAgent>();
        e.register::<ParticleEmitter>();
        e.register::<ScenePortal>();
        e.register::<AvatarDescriptor>();
        e.register::<UiCanvas>();
        e.register::<UiRect>();
        e.register::<UiPanel>();
        e.register::<UiImage>();
        e.register::<UiText>();
        e.register::<Text3D>();
        e.register::<UiButton>();
        e.register::<UiToggle>();
        e.register::<UiSlider>();
        e.register::<UiTextField>();
        e.register::<UiProgressBar>();
        e.register::<UiPauseMenu>();
        e.register::<VolumeComponent>();
        e.register::<Pawn>();
        e.register::<SpawnPoint>();
        e.register::<Sync>();
        e.register::<zestengine_core::NetworkObject>();
        e.register::<zestengine_core::NetworkVariables>();
        e.register::<zestengine_core::NetworkAnimator>();
        e.register::<zestengine_core::NetworkAvatar>();
        e.register::<zestengine_core::NetworkManager>();
        e.register::<zestengine_core::PersistentState>();
        crate::kit_ui::register(&mut e);
        crate::wasm_ui::register(&mut e);
        crate::cloth_ui::register(&mut e);
        crate::ik_ui::register(&mut e);
        crate::face_ui::register(&mut e);
        crate::constraint_ui::register(&mut e);
        crate::water_ui::register(&mut e);
        crate::waterfall_ui::register(&mut e);
        crate::water_container_ui::register(&mut e);
        crate::avatar_ui::register(&mut e);
        crate::sprite2d_ui::register(&mut e);
        crate::rig2d_ui::register(&mut e);
        crate::sprite_shape_ui::register(&mut e);
        crate::tilemap_ui::register(&mut e);
        e
    }

    /// Register a component's editor traits. Re-registering replaces (plugin
    /// hot-reload). Requires the runtime [`TypedComponent`] (for the name) plus
    /// the editor [`Inspect`] + [`Gizmo`] impls.
    pub fn register<T: TypedComponent + Inspect + Gizmo>(&mut self) {
        self.inspect.insert(T::NAME, |c, ui, ictx| {
            c.as_any_mut()
                .downcast_mut::<T>()
                .is_some_and(|t| t.inspector_ui(ui, ictx))
        });
        self.gizmos.insert(T::NAME, |c| {
            c.as_any()
                .downcast_ref::<T>()
                .map(|t| t.gizmos())
                .unwrap_or_default()
        });
        self.gizmo_edit.insert(T::NAME, |c, edit| {
            if let Some(t) = c.as_any_mut().downcast_mut::<T>() {
                t.apply_gizmo_edit(edit);
            }
        });
    }

    /// A component's declared viewport gizmos (empty if unregistered/none).
    pub fn gizmos(&self, component: &dyn Component) -> Vec<GizmoSpec> {
        match self.gizmos.get(component.type_name()) {
            Some(f) => f(component),
            None => Vec::new(),
        }
    }

    /// Apply a gizmo resize edit back to a component.
    pub fn apply_gizmo_edit(&self, component: &mut dyn Component, edit: GizmoEdit) {
        if let Some(f) = self.gizmo_edit.get(component.type_name()) {
            f(component, edit);
        }
    }

    /// Draw a component's inspector (no-op if unregistered).
    pub fn inspect(&self, component: &mut dyn Component, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        match self.inspect.get(component.type_name()) {
            Some(f) => f(component, ui, ctx),
            None => false,
        }
    }

}

#[derive(Default)]
pub struct ComponentsResponse {
    pub changed: bool,
    /// Registry name picked from the Add Component menu.
    pub add: Option<&'static str>,
    /// Index of a component whose ✕ was clicked.
    pub remove: Option<usize>,
    /// A component's Network Role was picked: (index, `None` = the type's default).
    pub role: Option<(usize, Option<zestengine_core::NetRole>)>,
}

/// Temp-data key under which the engine publishes, per component of the selected
/// object, `(explicit override, type default)` network roles
/// (`Vec<(Option<NetRole>, NetRole)>`). Absent: no Network Role menu.
pub const COMPONENT_ROLES_KEY: &str = "zestengine-component-roles";

/// Component list + Add Component button (object inspector). Inspectors are
/// dispatched through `editor` so the runtime `Component` stays egui-free.
pub fn components_ui(
    ui: &mut Ui,
    components: &mut [Box<dyn Component>],
    registry: &ComponentRegistry,
    editor: &EditorComponents,
    objects: &[(ObjectId, String)],
    layer_names: &[String],
    asset_editor: Option<&AssetEditor<'_>>,
) -> ComponentsResponse {
    let mut response = ComponentsResponse::default();
    let hide_blendshapes = ui.data_mut(|d| d.remove_temp::<bool>(egui::Id::new(crate::renderer_ui::HIDE_BLENDSHAPES_KEY))).unwrap_or(false);
    let has_body = components.iter().any(|c| c.type_name() == "Rigid Body" && zestengine_core::component_enabled(c.as_ref()));
    let inspect_ctx = InspectCtx { objects, layer_names, asset_editor, has_body };
    // Plugin components (cdylibs) statically link their own copy of egui, so a
    // selectable label drawn from a plugin's egui panics (mismatched TypeId for
    // egui's selection state). Disable selection for the whole inspector list.
    ui.style_mut().interaction.selectable_labels = false;
    // Keys that tell apart components of one kind (`Kind`, `Kind#1`), as prefab overrides name them.
    let roles: Option<Vec<(Option<zestengine_core::NetRole>, zestengine_core::NetRole)>> = ui.data(|d| d.get_temp(egui::Id::new(COMPONENT_ROLES_KEY)));
    let prefab_keys: Vec<String> = {
        let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        components
            .iter()
            .map(|c| {
                let n = seen.entry(c.type_name()).or_insert(0);
                *n += 1;
                if *n == 1 { c.type_name().to_owned() } else { format!("{}#{}", c.type_name(), *n - 1) }
            })
            .collect()
    };
    for (index, component) in components.iter_mut().enumerate() {
        // The renderer has its own sections (slots, lighting, materials) above this list.
        if zestengine_core::is_renderer_kind(component.type_name()) {
            continue;
        }
        if component.type_name() == "Blend Shapes" && hide_blendshapes {
            continue;
        }
        let id = ui.make_persistent_id(("zestengine-component", index, component.type_name()));
        crate::widgets::divider(ui);
        let prefab_mark = crate::prefab_ui::begin_component(ui, &prefab_keys[index]);
        let (icon, color) = component_icon(component.type_name());
        // The header's checkbox switches the component on and off in place;
        // a switched-off component stays listed, greyed out.
        let mut on = zestengine_core::component_enabled(component.as_ref());
        let playing = ui.data(|d| d.get_temp::<bool>(egui::Id::new(PLAY_MODE_KEY)).unwrap_or(false));
        let (open, toggled) = crate::widgets::section_header_toggle(ui, id, Some((icon, color)), component.type_name(), &mut on, playing, |ui| {
            let kebab = crate::widgets::icon_button(ui, crate::icons::Icon::Kebab, "Component options", false);
            egui::Popup::menu(&kebab).show(|ui| {
                if let Some((explicit, default)) = roles.as_ref().and_then(|r| r.get(index)).copied() {
                    let effective = explicit.unwrap_or(default);
                    ui.menu_button(format!("Network Role: {}", effective.label()), |ui| {
                        for role in zestengine_core::NetRole::ALL {
                            if ui.selectable_label(explicit == Some(role), role.label()).clicked() {
                                response.role = Some((index, Some(role)));
                                ui.close();
                            }
                        }
                        ui.separator();
                        if ui.selectable_label(explicit.is_none(), format!("Type default ({})", default.label())).clicked() {
                            response.role = Some((index, None));
                            ui.close();
                        }
                    });
                    ui.separator();
                }
                if ui.button("Remove component").clicked() {
                    response.remove = Some(index);
                    ui.close();
                }
            });
        });
        if open {
            // Fields that differ across a multi-selection, for this component
            // (published by the editor under MIXED_COMPONENTS_KEY).
            let mixed: Vec<String> = ui
                .data(|d| d.get_temp::<Vec<(String, Vec<String>)>>(egui::Id::new(MIXED_COMPONENTS_KEY)))
                .and_then(|all| all.into_iter().find(|(n, _)| n == component.type_name()).map(|(_, f)| f))
                .unwrap_or_default();
            ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::widgets::MIXED_FIELDS_KEY), mixed));
            egui::Frame::new()
                .inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 10 })
                .show(ui, |ui| {
                    ui.add_enabled_ui(on, |ui| {
                        response.changed |= editor.inspect(zestengine_core::component_inner_mut(component), ui, &inspect_ctx);
                    });
                });
            ui.data_mut(|d| d.remove::<Vec<String>>(egui::Id::new(crate::widgets::MIXED_FIELDS_KEY)));
        }
        crate::prefab_ui::end_component(ui, prefab_mark);
        if toggled && !playing {
            zestengine_core::set_component_enabled(component, on);
            response.changed = true;
        }
    }
    crate::widgets::divider(ui);

    ui.add_space(12.0);
    let add = crate::widgets::dashed_button(ui, Some(crate::icons::Icon::Plus), "Add component");
    if add.clicked() {
        ui.data_mut(|d| d.insert_temp(egui::Id::new("zestengine-add-component-focus"), true));
    }
    // Fixed size so filtering can't shrink the popup and flip its placement.
    egui::Popup::menu(&add)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .align(egui::RectAlign::BOTTOM_START)
        .align_alternatives(&[egui::RectAlign::TOP_START])
        .show(|ui| {
            let search_id = ui.make_persistent_id("zestengine-add-component-search");
            // Fresh search, focused, each time the menu opens (the flag is set
            // on the click in case the popup first draws a frame later).
            let focus_id = egui::Id::new("zestengine-add-component-focus");
            let focus = add.clicked() || ui.data_mut(|d| d.remove_temp::<bool>(focus_id)).unwrap_or(false);
            if focus {
                ui.data_mut(|d| d.remove::<String>(search_id));
            }
            let mut search: String = ui.data(|d| d.get_temp(search_id).unwrap_or_default());
            let width = add.rect.width().max(240.0);
            ui.set_min_width(width);
            ui.set_max_width(width);
            let pick_id = search_id.with("pick");
            // Arrow keys move the pick; taken before the field sees them.
            let (down, up, enter) = ui.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    i.key_pressed(egui::Key::Enter),
                )
            });
            let edit = crate::widgets::search_field(ui, &mut search, "Search components", width);
            if focus {
                edit.request_focus();
            }
            if edit.changed() {
                ui.data_mut(|d| d.insert_temp(search_id, search.clone()));
            }
            let needle = search.to_lowercase();
            // Camera is intrinsic to camera objects.
            let matches: Vec<&'static str> = registry
                .names()
                .filter(|n| *n != "Camera" && (needle.is_empty() || n.to_lowercase().contains(&needle)))
                .collect();
            let mut pick: usize = if focus || edit.changed() { 0 } else { ui.data(|d| d.get_temp(pick_id).unwrap_or(0)) };
            if down {
                pick += 1;
            }
            if up {
                pick = pick.saturating_sub(1);
            }
            pick = pick.min(matches.len().saturating_sub(1));
            ui.data_mut(|d| d.insert_temp(pick_id, pick));
            // Enter adds the picked entry (the top match unless the arrows moved it).
            if enter && (edit.has_focus() || edit.lost_focus())
                && let Some(name) = matches.get(pick)
            {
                response.add = Some(*name);
                ui.data_mut(|d| d.remove::<String>(search_id));
                ui.close();
            }
            ui.add_space(4.0);
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .min_scrolled_height(320.0)
                .max_height(320.0)
                .show(ui, |ui| {
                let any = !matches.is_empty();
                for (k, &name) in matches.iter().enumerate() {
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, crate::theme::H_TOOLBAR), Sense::click());
                    let picked = k == pick;
                    if picked {
                        ui.painter().rect_filled(rect, crate::theme::R_ROW, crate::theme::BG_SEGMENT_ACTIVE);
                        if down || up {
                            resp.scroll_to_me(None);
                        }
                    } else if resp.hovered() {
                        ui.painter().rect_filled(rect, crate::theme::R_ROW, crate::theme::BG_BUTTON);
                    }
                    let (icon, color) = component_icon(name);
                    crate::icons::paint(
                        ui.painter(),
                        egui::Rect::from_center_size(egui::pos2(rect.left() + 16.0, rect.center().y), egui::vec2(14.0, 14.0)),
                        icon,
                        color,
                    );
                    ui.painter().text(
                        egui::pos2(rect.left() + 32.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        name,
                        egui::FontId::proportional(crate::theme::SIZE_BODY),
                        if resp.hovered() || picked { crate::theme::TEXT_STRONG } else { crate::theme::TEXT_SECONDARY },
                    );
                    if resp.clicked() {
                        response.add = Some(name);
                        ui.data_mut(|d| d.remove::<String>(search_id));
                        ui.close();
                    }
                }
                if !any {
                    ui.label(RichText::new("No matches").color(crate::theme::TEXT_DIM));
                }
            });
        });
    response
}

/// Header icon and colour for a component type (editor spec §2.4).
pub fn component_icon(name: &str) -> (crate::icons::Icon, egui::Color32) {
    use crate::icons::Icon;
    use crate::theme;
    if let Some(kit) = crate::kit_ui::icon(name) {
        return kit;
    }
    match name {
        "Light" | "Area Light" => (Icon::LightPoint, theme::TYPE_LIGHT),
        "Camera" => (Icon::Camera, theme::TYPE_CAMERA),
        "UI Canvas" => (Icon::UiCanvas, theme::TYPE_UI),
        "UI Toggle" => (Icon::UiToggle, theme::TYPE_UI),
        "UI Text" | "UI Text Field" => (Icon::UiText, theme::TYPE_UI),
        n if n.starts_with("UI ") => (Icon::UiCanvas, theme::TYPE_UI),
        "Reflection Probe" | "Planar Reflection" => (Icon::ReflectionProbe, theme::TYPE_GI),
        "Flux Volume" | "Light Probe Volume" | "FluxVoxel Light" => (Icon::FluxVolume, theme::TYPE_GI),
        "Audio Source" | "Audio Listener" => (Icon::Audio, theme::TYPE_GROUP),
        "Animator" | "LOD Group" | "Bone Cloth" | "Mesh Cloth" => (Icon::Model, theme::TYPE_MESH),
        "Behaviour Tree" | "Perception" => (Icon::Target, theme::TYPE_GI),
        "Cloth Colliders" => (Icon::Target, theme::TYPE_MESH),
        "Wind Zone" => (Icon::FluxVolume, theme::TYPE_GROUP),
        "Humanoid IK" | "Face Tracking" | "Lip Sync" => (Icon::Target, theme::TYPE_CAMERA),
        "Voice Chat" => (Icon::Audio, theme::TYPE_GROUP),
        "Water" | "Buoyancy" | "Waterfall" | "Whirlpool" | "Wave Rider" => (Icon::Water, theme::TYPE_WATER),
        "Blend Shapes" => (Icon::Model, theme::TYPE_MESH),
        _ => (Icon::Gear, theme::TYPE_GROUP),
    }
}

// ----------------------------------------------------------------- helpers

fn property_row(
    ui: &mut Ui,
    label: &str,
    changed: &mut bool,
    widget: impl FnOnce(&mut Ui) -> egui::Response,
) {
    crate::widgets::property_row(ui, label, |ui| {
        if widget(ui).changed() {
            *changed = true;
        }
    });
}

fn axis_row(ui: &mut Ui, label: &str, axis: &mut [f32; 3], changed: &mut bool) {
    crate::widgets::property_row(ui, label, |ui| {
        *changed |= crate::widgets::vec3_field(ui, axis, 2, 0.02);
    });
}

fn collider_common_ui(
    ui: &mut Ui,
    ctx: &InspectCtx,
    material: &mut String,
    is_trigger: &mut bool,
    layer: &mut u32,
    changed: &mut bool,
) {
    // On a body the material is the Rigid Body's; a collider overrides it
    // only on request (a rubber grip on a steel hammer). Without a body
    // (static colliders) the collider's material is the surface's.
    if ctx.has_body {
        let open_id = ui.id().with("override-material");
        let mut open = !material.is_empty() || ui.data(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
        crate::widgets::property_row(ui, "Override material", |ui| {
            if ui
                .checkbox(&mut open, "")
                .on_hover_text("Give this collider its own physics material instead of the Rigid Body's (friction, bounce and, with Auto mass, density for this part)")
                .changed()
                && !open
                && !material.is_empty()
            {
                material.clear();
                *changed = true;
            }
        });
        ui.data_mut(|d| d.insert_temp(open_id, open));
        if open {
            *changed |= ctx.file_field(ui, "Physics Material", material, "Rigid Body's", &["physmat"]);
        }
    } else {
        *changed |= ctx.file_field(ui, "Physics Material", material, "project default", &["physmat"]);
    }
    property_row(ui, "Is Trigger", changed, |ui| {
        ui.checkbox(is_trigger, "")
            .on_hover_text("Detect overlaps without blocking (no collision response)")
    });
    property_row(ui, "Layer", changed, |ui| {
        let r = ui.add(egui::DragValue::new(layer).range(0..=COLLISION_LAYERS - 1));
        *layer = (*layer).min(COLLISION_LAYERS - 1);
        r
    });
}

// ----------------------------------------------------------- builtin inspect

impl Inspect for CameraComponent {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Camera ID", &mut changed, |ui| {
            ui.add_enabled(false, DragValue::new(&mut self.id))
        });
        property_row(ui, "Projection", &mut changed, |ui| {
            ui.horizontal(|ui| {
                let a = ui
                    .selectable_value(&mut self.projection, zestengine_core::CameraProjection::Perspective, "Perspective")
                    .on_hover_text("Perspective: field of view");
                let b = ui
                    .selectable_value(&mut self.projection, zestengine_core::CameraProjection::Orthographic, "Orthographic")
                    .on_hover_text("Orthographic: no foreshortening, sized by half-height (2D / isometric)");
                a | b
            })
            .inner
        });
        if self.projection == zestengine_core::CameraProjection::Orthographic {
            property_row(ui, "Size", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.ortho_size).speed(0.05).range(0.01..=100000.0))
                    .on_hover_text("Half of the visible height, in world units")
            });
        } else {
            property_row(ui, "Field of View", &mut changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut self.fov_deg, 10.0..=170.0).suffix("°"))
            });
        }
        property_row(ui, "Near", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.near).speed(0.01).range(0.001..=1000.0))
        });
        property_row(ui, "Far", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.far).speed(0.5).range(0.01..=100000.0))
        });
        if self.far <= self.near {
            self.far = self.near + 0.01;
        }
        property_row(ui, "Anti-aliasing", &mut changed, |ui| {
            let mut c = false;
            let mut resp = egui::ComboBox::from_id_salt("camera_aa")
                .selected_text(self.anti_aliasing.label())
                .show_ui(ui, |ui| {
                    for m in zestengine_core::CameraAntiAliasing::ALL {
                        c |= ui.selectable_value(&mut self.anti_aliasing, m, m.label()).changed();
                    }
                })
                .response
                .on_hover_text("Overrides the Project Settings anti-aliasing while this camera is the main camera");
            if c {
                resp.mark_changed();
            }
            resp
        });
        property_row(ui, "Render Texture", &mut changed, |ui| {
            ui.checkbox(&mut self.render_texture, "").on_hover_text(
                "Render into a texture instead of the screen. Materials show it through the texture path camera:<object id>",
            )
        });
        if self.render_texture {
            property_row(ui, "Texture Size", &mut changed, |ui| {
                ui.horizontal(|ui| {
                    let a = ui.add(DragValue::new(&mut self.texture_size[0]).range(16..=4096).suffix(" px"));
                    let b = ui.add(DragValue::new(&mut self.texture_size[1]).range(16..=4096).suffix(" px"));
                    a | b
                })
                .inner
            });
            property_row(ui, "Update Rate", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.update_hz).range(0.0..=240.0).suffix(" Hz"))
                    .on_hover_text("Refreshes per second (0 = every frame). Off-screen screens never render.")
            });
            if ui
                .button("Create Screen")
                .on_hover_text("Spawn a plane in front of the view showing this camera's feed (TV / CCTV monitor)")
                .clicked()
            {
                self.create_screen = true;
                changed = true;
            }
        }
        // Layer culling mask (Unity-style): which layers this camera renders.
        // Layer names + the collision matrix are edited in Tools → Layers.
        ui.collapsing("Culling Mask", |ui| {
            ui.horizontal(|ui| {
                if ui.button("Everything").clicked() {
                    self.culling_mask = u32::MAX;
                    changed = true;
                }
                if ui.button("Nothing").clicked() {
                    self.culling_mask = 0;
                    changed = true;
                }
            });
            // Show ALL layers (same as the object dropdown + collision matrix), with
            // their real names. Scrollable so 32 rows don't blow out the inspector.
            let shown = ctx.layer_names.len().max(8) as u32;
            egui::ScrollArea::vertical()
                .id_salt("cam-cull-scroll")
                .max_height(220.0)
                .show(ui, |ui| {
            egui::Grid::new("camera-culling-mask").show(ui, |ui| {
                for l in 0u32..shown {
                    let bit = 1u32 << l;
                    let mut on = self.culling_mask & bit != 0;
                    let name = ctx
                        .layer_names
                        .get(l as usize)
                        .filter(|s| !s.is_empty())
                        .cloned()
                        .unwrap_or_else(|| format!("Layer {l}"));
                    if ui.checkbox(&mut on, name).changed() {
                        if on {
                            self.culling_mask |= bit;
                        } else {
                            self.culling_mask &= !bit;
                        }
                        changed = true;
                    }
                    if l % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
            });
        });
        ui.label(
            RichText::new("Post-processing effects arrive with the post pipeline")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for CameraComponent {}

impl Inspect for LightComponent {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Type");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("zestengine-light-kind")
                    .selected_text(self.kind.label())
                    .show_ui(ui, |ui| {
                        for kind in LightKind::ALL {
                            changed |= ui
                                .selectable_value(&mut self.kind, kind, kind.label())
                                .changed();
                        }
                    });
            });
        });
        ui.horizontal(|ui| {
            ui.label("Mode");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("zestengine-light-mode")
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        for mode in LightMode::ALL {
                            changed |= ui
                                .selectable_value(&mut self.mode, mode, mode.label())
                                .changed();
                        }
                    });
            });
        });
        property_row(ui, "Color", &mut changed, |ui| {
            ui.color_edit_button_rgb(&mut self.color)
        });
        property_row(ui, "Intensity", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.intensity).speed(0.05).range(0.0..=100.0))
        });
        let (lm, unit) = self.approx_photometric();
        ui.horizontal(|ui| {
            ui.label("");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("≈ {lm:.0} {unit}")).weak().small())
                    .on_hover_text("Approximate photometric output (display only)");
            });
        });
        if self.kind != LightKind::Directional {
            property_row(ui, "Range", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.range).speed(0.1).range(0.01..=10000.0))
            });
        }
        if self.kind == LightKind::Spot {
            property_row(ui, "Spot Angle", &mut changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut self.spot_angle, 1.0..=179.0).suffix("°"))
            });
            property_row(ui, "Spot Blend", &mut changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut self.spot_blend, 0.0..=1.0))
            });
        }
        ui.horizontal(|ui| {
            ui.label("Shadows");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("zestengine-light-shadow")
                    .selected_text(self.shadow_type.label())
                    .show_ui(ui, |ui| {
                        for st in ShadowType::ALL {
                            changed |= ui
                                .selectable_value(&mut self.shadow_type, st, st.label())
                                .changed();
                        }
                    });
            });
        });
        if self.shadow_type.casts() {
            property_row(ui, "Shadow Bias", &mut changed, |ui| {
                ui.add(
                    DragValue::new(&mut self.shadow_bias)
                        .speed(0.0005)
                        .range(0.0..=0.1)
                        .max_decimals(4),
                )
            });
            property_row(ui, "Radius (soft)", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.radius).speed(0.02).range(0.0..=5.0))
                    .on_hover_text(
                        "Light source size for baked soft shadows — larger = softer \
                         penumbra (smooths jagged shadows at low texel density). 0 = hard.",
                    )
            });
        }
        changed
    }
}
impl Gizmo for LightComponent {}

impl Inspect for AreaLight {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Shape");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("zestengine-area-light-shape")
                    .selected_text(self.shape.label())
                    .show_ui(ui, |ui| {
                        for shape in zestengine_core::AreaLightShape::ALL {
                            changed |= ui.selectable_value(&mut self.shape, shape, shape.label()).changed();
                        }
                    });
            });
        });
        vec2_row(ui, "Size (m)", &mut self.size, 0.02, &mut changed);
        for s in &mut self.size {
            *s = s.max(0.01);
        }
        property_row(ui, "Color", &mut changed, |ui| ui.color_edit_button_rgb(&mut self.color));
        property_row(ui, "Intensity", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.intensity).speed(0.05).range(0.0..=1000.0))
                .on_hover_text("Emitted radiance: a white surface held right against the emitter shows this brightness")
        });
        property_row(ui, "Range", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.range).speed(0.1).range(0.01..=10000.0))
        });
        property_row(ui, "Two Sided", &mut changed, |ui| ui.checkbox(&mut self.two_sided, ""));
        changed |= ctx.file_field(ui, "Emission Texture", &mut self.texture, "texture/screen.png", &["png", "jpg", "jpeg", "tga", "bmp", "webp", "exr", "hdr"]);
        ui.label(RichText::new("Lights toward -Z. No shadows. The 8 nearest area lights are drawn; the texture (or a camera:<id> render texture) modulates the colour live.").small().weak());
        changed
    }
}
impl Gizmo for AreaLight {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let half = glam::Vec3::new(self.size[0].max(0.0) * 0.5, self.size[1].max(0.0) * 0.5, 0.01);
        vec![GizmoSpec::Box {
            center: glam::Vec3::ZERO,
            half_extents: half,
            object_anchored: true,
            blend: 0.0,
            color: egui::Color32::from_rgb(255, 220, 120),
        }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit {
            self.size = [half_extents.x.abs() * 2.0, half_extents.y.abs() * 2.0];
        }
    }
}

impl Inspect for VolumeComponent {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        // Drag a .postfx from the file browser onto this field to set it.
        changed |= ctx.file_field(
            ui,
            "Profile",
            &mut self.profile,
            "drag a .postfx here",
            &["postfx"],
        );
        property_row(ui, "Global", &mut changed, |ui| {
            ui.checkbox(&mut self.global, "")
                .on_hover_text("Affects the whole scene (else only within the box bounds)")
        });
        property_row(ui, "Priority", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.priority).speed(0.1))
                .on_hover_text("Higher blends last / wins ties")
        });
        property_row(ui, "Weight", &mut changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut self.weight, 0.0..=1.0))
        });
        if !self.global {
            property_row(ui, "Blend Distance", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.blend_distance).speed(0.05).range(0.0..=100.0))
                    .on_hover_text("World-units the effect fades in over, approaching the box")
            });
            axis_row(ui, "Half Extents", &mut self.half_extents, &mut changed);
        }
        changed
    }
}
impl Gizmo for VolumeComponent {
    /// Local volumes: the box (resizable like a reflection probe's) plus an
    /// outer shell at the blend distance, where the effect starts fading in.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        if self.global {
            return Vec::new();
        }
        vec![GizmoSpec::Box {
            center: glam::Vec3::ZERO,
            half_extents: glam::Vec3::from(self.half_extents).abs(),
            object_anchored: true,
            blend: -self.blend_distance.max(0.0),
            color: crate::theme::TYPE_POSTFX,
        }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit {
            self.half_extents = half_extents.to_array();
        }
    }
}

impl Inspect for LodComponent {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.label(
            egui::RichText::new("Swap to a cheaper mesh as the object recedes. Level 0 = nearest. Models without a LOD Group get an automatic screen-size chain; remove all levels here to opt out.")
                .small()
                .weak(),
        );
        property_row(ui, "By screen size", &mut changed, |ui| {
            ui.checkbox(&mut self.by_screen_size, "")
                .on_hover_text("Pick levels by the share of the screen height the object covers, so one setting suits any object size, FOV and resolution")
        });
        let mut remove: Option<usize> = None;
        let by_screen = self.by_screen_size;
        for (i, level) in self.levels.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("LOD {i}"));
                if by_screen {
                    ui.label("≥");
                    let mut pct = level.screen_size * 100.0;
                    if ui
                        .add(DragValue::new(&mut pct).speed(0.1).range(0.0..=100.0).suffix(" %"))
                        .on_hover_text("Used while the object covers at least this share of the screen height (0 = no minimum)")
                        .changed()
                    {
                        level.screen_size = pct / 100.0;
                        changed = true;
                    }
                } else {
                    ui.label("≤");
                    changed |= ui
                        .add(DragValue::new(&mut level.max_distance).speed(0.5).range(0.1..=10000.0).suffix(" m"))
                        .changed();
                }
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut level.model).desired_width(120.0).hint_text("model (blank = generate)"))
                    .changed();
                if level.model.is_empty() {
                    let mut pct = (level.keep * 100.0).round();
                    if ui
                        .add(DragValue::new(&mut pct).speed(1.0).range(0.0..=100.0).suffix(" %"))
                        .on_hover_text("Generate this level from the object's model, keeping this share of its triangles (0 = the base mesh)")
                        .changed()
                    {
                        level.keep = pct / 100.0;
                        changed = true;
                    }
                }
                if ui.small_button(ph::X).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.levels.remove(i);
            changed = true;
        }
        ui.horizontal(|ui| {
            if ui.button("+ Add level").clicked() {
                let d = self.levels.last().map(|l| l.max_distance * 2.0).unwrap_or(15.0);
                let keep = self.levels.last().map(|l| if l.keep > 0.0 { l.keep * 0.5 } else { 0.5 }).unwrap_or(0.0);
                let screen_size = self.levels.last().map(|l| l.screen_size * 0.4).unwrap_or(0.3);
                self.levels.push(LodLevel { max_distance: d, model: String::new(), keep, screen_size });
                changed = true;
            }
        });
        property_row(ui, "Cull distance", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.cull_distance).speed(0.5).range(0.0..=10000.0).suffix(" m"))
                .on_hover_text("Beyond this the object is culled entirely (0 = never)")
        });
        // Keep levels sorted ascending so selection is correct.
        if changed {
            if self.by_screen_size {
                // Most detailed (largest on screen) first.
                self.levels.sort_by(|a, b| b.screen_size.total_cmp(&a.screen_size));
            } else {
                self.levels.sort_by(|a, b| a.max_distance.total_cmp(&b.max_distance));
            }
        }
        changed
    }
}
impl Gizmo for LodComponent {}

impl Inspect for Spin {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Axis", &mut self.axis, &mut changed);
        property_row(ui, "Speed (°/s)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.degrees_per_second).speed(1.0))
        });
        changed
    }
}
impl Gizmo for Spin {}

impl Inspect for Bob {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Axis", &mut self.axis, &mut changed);
        property_row(ui, "Amplitude", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.amplitude).speed(0.02))
        });
        property_row(ui, "Frequency (Hz)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.frequency).speed(0.02))
        });
        changed
    }
}
impl Gizmo for Bob {}

impl Inspect for RigidBody {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Type", &mut changed, |ui| {
            let mut resp = egui::ComboBox::from_id_salt("zestengine-body-kind")
                .selected_text(self.kind.label())
                .show_ui(ui, |ui| {
                    let mut clicked = false;
                    for k in BodyKind::ALL {
                        clicked |= ui
                            .selectable_value(&mut self.kind, k, k.label())
                            .clicked();
                    }
                    clicked
                });
            if resp.inner == Some(true) {
                resp.response.mark_changed();
            }
            resp.response
        });
        if self.kind == BodyKind::Dynamic {
            let mut set = self.mass > 0.0;
            crate::widgets::property_row(ui, "Mass", |ui| {
                let r = crate::widgets::segmented(ui, &mut set, &[(false, "Auto"), (true, "Set")]).on_hover_text(
                    "Auto: the colliders' volume times the physics material's density (500 kg/m3 without one). Set: a fixed mass in kg, ignoring density.",
                );
                if r.changed() {
                    self.mass = if set { self.measured.map_or(1.0, |(m, _)| m.max(0.01)) } else { 0.0 };
                    changed = true;
                }
            });
            if set {
                property_row(ui, "Mass (kg)", &mut changed, |ui| ui.add(DragValue::new(&mut self.mass).speed(0.1).range(0.001..=1e6)));
                if !self.material.is_empty() {
                    ui.label(egui::RichText::new("A set mass overrides the material's density: this body's density is mass / volume.").small().weak());
                }
            }
            if let Some((m, v)) = self.measured.filter(|(_, v)| *v > 1e-6) {
                let density = m / v;
                let fate = if density < 1000.0 { "floats" } else { "sinks" };
                ui.label(
                    egui::RichText::new(format!("{m:.1} kg · {v:.3} m3 · {density:.0} kg/m3: {fate} in fresh water (1000)"))
                        .small()
                        .color(crate::theme::TEXT_DIM),
                );
            }
            property_row(ui, "Gravity scale", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.gravity_scale).speed(0.05))
            });
        }
        changed |= ctx.file_field(ui, "Physics Material", &mut self.material, "project default", &["physmat"]);
        if self.material.is_empty() {
            property_row(ui, "Restitution", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.restitution).speed(0.02).range(0.0..=1.0))
            });
            property_row(ui, "Friction", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.friction).speed(0.02).range(0.0..=2.0))
            });
        }
        changed
    }
}
impl Gizmo for RigidBody {
    /// A small diamond at the body origin, coloured by body kind.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        use glam::Vec3;
        let r = 0.12;
        let pts = [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z].map(|d| d * r);
        let (top, bot) = (Vec3::Y * r * 1.4, Vec3::NEG_Y * r * 1.4);
        let mut segments = Vec::new();
        for (k, p) in pts.iter().enumerate() {
            segments.push((*p, top));
            segments.push((*p, bot));
            let q = pts[[2, 3, 1, 0][k]];
            segments.push((*p, q));
        }
        let color = match self.kind {
            BodyKind::Dynamic => egui::Color32::from_rgb(255, 170, 80),
            BodyKind::Kinematic => egui::Color32::from_rgb(120, 190, 255),
            BodyKind::Fixed => egui::Color32::from_rgb(170, 170, 180),
        };
        vec![GizmoSpec::Lines { segments, color, world: false }]
    }
}

impl Inspect for Character {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ui.checkbox(&mut self.use_default_capsule, "Default movement capsule").changed();
        changed |= ui.checkbox(&mut self.capsule_affects_water, "Capsule affects water").changed();
        ui.small("For individual leg/foot wakes, turn off Capsule affects water and use bone-attached kinematic colliders. This works with desktop and VR avatars.");
        if !self.use_default_capsule {
            ui.small("Default movement collision is disabled. Custom colliders and movement must provide world collision.");
        }
        property_row(ui, "Radius", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.radius).speed(0.01).range(0.01..=10.0))
        });
        property_row(ui, "Height", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.height).speed(0.01).range(0.02..=20.0))
        });
        axis_row(ui, "Center", &mut self.center, &mut changed);
        property_row(ui, "Step height", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.step_height).speed(0.01).range(0.0..=2.0))
        });
        property_row(ui, "Max slope (deg)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.max_slope).speed(0.5).range(0.0..=89.0))
        });
        property_row(ui, "Snap to ground", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.snap_to_ground).speed(0.01).range(0.0..=2.0))
        });
        changed
    }
}
impl Gizmo for Character {
    /// The capsule, plus a ring at step height (the tallest step it climbs).
    fn gizmos(&self) -> Vec<GizmoSpec> {
        use glam::Vec3;
        let center = Vec3::from(self.center);
        let mut g = vec![GizmoSpec::Lines {
            segments: capsule_segments(center, self.radius.max(0.01), self.height.max(0.02)),
            color: egui::Color32::from_rgb(110, 230, 140),
            world: false,
        }];
        if self.step_height > 0.0 {
            let foot = center - Vec3::Y * (self.height * 0.5);
            let mut s = Vec::new();
            circle_segments(&mut s, foot + Vec3::Y * self.step_height, Vec3::X, Vec3::Z, self.radius, 0.0, std::f32::consts::TAU, 32);
            g.push(GizmoSpec::Lines { segments: s, color: egui::Color32::from_rgb(70, 140, 90), world: false });
        }
        g
    }
}

fn range_row(ui: &mut Ui, label: &str, v: &mut [f32; 2], speed: f64, changed: &mut bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            *changed |= ui.add(DragValue::new(&mut v[1]).speed(speed).prefix("max ")).changed();
            *changed |= ui.add(DragValue::new(&mut v[0]).speed(speed).prefix("min ")).changed();
        });
    });
}

impl Inspect for ParticleEmitter {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Play on start", &mut changed, |ui| ui.checkbox(&mut self.play_on_start, ""));
        property_row(ui, "Looping", &mut changed, |ui| ui.checkbox(&mut self.looping, ""));
        if !self.looping {
            property_row(ui, "Duration (s)", &mut changed, |ui| ui.add(DragValue::new(&mut self.duration).speed(0.05).range(0.0..=600.0)));
        }
        property_row(ui, "Rate (/s)", &mut changed, |ui| ui.add(DragValue::new(&mut self.rate).speed(0.5).range(0.0..=10000.0)));
        property_row(ui, "Burst", &mut changed, |ui| ui.add(DragValue::new(&mut self.burst).range(0..=10000)));
        property_row(ui, "Max particles", &mut changed, |ui| ui.add(DragValue::new(&mut self.max_particles).range(1..=20000)));
        range_row(ui, "Lifetime (s)", &mut self.lifetime, 0.02, &mut changed);
        range_row(ui, "Speed", &mut self.speed, 0.05, &mut changed);
        axis_row(ui, "Direction", &mut self.direction, &mut changed);
        property_row(ui, "Cone angle", &mut changed, |ui| ui.add(DragValue::new(&mut self.cone_angle).speed(0.5).range(0.0..=180.0)));
        property_row(ui, "Spawn radius", &mut changed, |ui| ui.add(DragValue::new(&mut self.shape_radius).speed(0.01).range(0.0..=100.0)));
        property_row(ui, "Gravity", &mut changed, |ui| ui.add(DragValue::new(&mut self.gravity).speed(0.02)));
        property_row(ui, "Drag", &mut changed, |ui| ui.add(DragValue::new(&mut self.drag).speed(0.02).range(0.0..=50.0)));
        range_row(ui, "Size (start/end)", &mut self.size, 0.01, &mut changed);
        rgba_row(ui, "Color start", &mut self.color_start, &mut changed);
        rgba_row(ui, "Color end", &mut self.color_end, &mut changed);
        property_row(ui, "Texture", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.texture).hint_text("soft dot"))
        });
        property_row(ui, "World space", &mut changed, |ui| ui.checkbox(&mut self.world_space, ""));
        ui.horizontal(|ui| {
            if ui.button("Restart").clicked() {
                self.control.clear = true;
                self.control.play = true;
            }
            ui.label(egui::RichText::new(format!("{} alive", self.control.alive)).small().weak());
        });
        changed
    }
}
impl Gizmo for ParticleEmitter {}

impl Inspect for ScenePortal {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Scene", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.scene).hint_text("scenes/level2.scene"))
        });
        enum_row(ui, "Mode", "zestengine-portal-mode", &mut self.mode, &PortalMode::ALL, PortalMode::label, &mut changed);
        changed |= ctx.object_ref(ui, "Activator", &mut self.activator);
        if !self.activator.is_set() {
            ui.label(egui::RichText::new("Unset: any object with a Pawn").small().weak());
        }
        if self.mode != PortalMode::Unload {
            property_row(ui, "Preload distance", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.preload_distance).speed(0.5).range(0.0..=10000.0))
            });
        }
        if self.mode == PortalMode::Switch {
            property_row(ui, "Keep activator", &mut changed, |ui| ui.checkbox(&mut self.keep_activator, ""));
        }
        property_row(ui, "One shot", &mut changed, |ui| ui.checkbox(&mut self.one_shot, ""));
        ui.label(
            egui::RichText::new("Needs a trigger collider on this object. Preloads while the activator approaches.")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for ScenePortal {}

impl Inspect for AvatarDescriptor {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        crate::widgets::property_row(ui, "View position", |ui| {
            changed |= crate::widgets::vec3_field(ui, &mut self.view_position, 2, 0.01);
        });
        property_row(ui, "Collider radius", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.collider_radius).speed(0.01).range(0.0..=5.0))
        });
        property_row(ui, "Collider height", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.collider_height).speed(0.01).range(0.0..=10.0))
        });
        changed |= crate::avatar_rig::ui(ui, &mut self.bone_map);
        crate::avatar_ui::generated_output_ui(ui);
        ui.label(
            RichText::new("Root of an Avatar content package. The player camera sits at the view position; the humanoid map comes from the model's Rig settings.")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for AvatarDescriptor {}

impl Inspect for NavMeshSurface {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Bounds size", &mut self.size, &mut changed);
        property_row(ui, "Cell size", &mut changed, |ui| ui.add(DragValue::new(&mut self.cell_size).speed(0.01).range(0.05..=4.0)));
        property_row(ui, "Agent radius", &mut changed, |ui| ui.add(DragValue::new(&mut self.agent_radius).speed(0.01).range(0.0..=10.0)));
        property_row(ui, "Agent height", &mut changed, |ui| ui.add(DragValue::new(&mut self.agent_height).speed(0.01).range(0.1..=20.0)));
        property_row(ui, "Max slope", &mut changed, |ui| ui.add(DragValue::new(&mut self.max_slope).speed(0.5).range(0.0..=89.0)));
        property_row(ui, "Step height", &mut changed, |ui| ui.add(DragValue::new(&mut self.step_height).speed(0.01).range(0.0..=5.0)));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_preview, "Show navmesh");
            if ui.button("Rebake").clicked() {
                self.rebake_preview = true;
                self.show_preview = true;
            }
        });
        ui.label(
            egui::RichText::new("Baked from colliders when play starts; ctx.rebake_navmesh() at runtime")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for NavMeshSurface {
    /// The bake box (resizable).
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Box {
            center: glam::Vec3::ZERO,
            half_extents: glam::Vec3::from(self.size).abs() * 0.5,
            object_anchored: true,
            blend: 0.0,
            color: egui::Color32::from_rgb(90, 220, 170),
        }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit {
            self.size = (half_extents * 2.0).to_array();
        }
    }
}

impl Inspect for NavAgent {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Speed", &mut changed, |ui| ui.add(DragValue::new(&mut self.speed).speed(0.05).range(0.0..=100.0)));
        property_row(ui, "Turn speed", &mut changed, |ui| ui.add(DragValue::new(&mut self.turn_speed).speed(5.0).range(0.0..=5000.0)));
        property_row(ui, "Stopping distance", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.stopping_distance).speed(0.01).range(0.0..=50.0))
        });
        ui.label(egui::RichText::new("Gameplay: ctx.set_destination(agent, point)").small().weak());
        changed
    }
}
impl Gizmo for NavAgent {
    /// Stopping-distance ring at the feet; in Play, the planned path (world)
    /// with a cross at the destination.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        use glam::Vec3;
        let mut ring = Vec::new();
        circle_segments(&mut ring, Vec3::ZERO, Vec3::X, Vec3::Z, self.stopping_distance.max(0.1), 0.0, std::f32::consts::TAU, 32);
        let mut g = vec![GizmoSpec::Lines { segments: ring, color: egui::Color32::from_rgb(90, 220, 170), world: false }];
        let st = &self.state;
        if !st.path.is_empty() {
            let lift = Vec3::Y * 0.05;
            let mut s: Vec<(Vec3, Vec3)> = st.path.windows(2).map(|w| (w[0] + lift, w[1] + lift)).collect();
            if let Some(d) = st.destination {
                let c = 0.25;
                s.push((d + Vec3::new(-c, 0.05, -c), d + Vec3::new(c, 0.05, c)));
                s.push((d + Vec3::new(-c, 0.05, c), d + Vec3::new(c, 0.05, -c)));
            }
            let color = if st.unreachable { egui::Color32::from_rgb(240, 90, 90) } else { egui::Color32::from_rgb(250, 230, 110) };
            g.push(GizmoSpec::Lines { segments: s, color, world: true });
        }
        g
    }
}

impl Inspect for PerceptionComponent {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Vision range", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.perception.vision_range).speed(0.5).range(0.0..=1000.0))
        });
        property_row(ui, "Vision angle", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.perception.vision_angle).speed(1.0).range(0.0..=180.0).suffix(" deg"))
        });
        property_row(ui, "Hearing range", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.perception.hearing_range).speed(0.5).range(0.0..=1000.0))
        });
        changed |= ctx.object_ref(ui, "Target", &mut self.target);
        ui.label(egui::RichText::new("Seeds blackboard: has_target, target_id, target_position").small().weak());
        changed
    }
}

impl Gizmo for PerceptionComponent {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        use glam::Vec3;
        let forward = Vec3::NEG_Z;
        let (a, b) = basis_of(forward);
        let half = self.perception.half_angle();
        let mut ring = Vec::new();
        circle_segments(&mut ring, Vec3::ZERO, a, b, self.perception.vision_range.max(0.1), -half, half, 32);
        let mut s = vec![GizmoSpec::Lines {
            segments: ring,
            color: egui::Color32::from_rgb(120, 210, 255),
            world: false,
        }];
        let tip = forward * self.perception.vision_range.max(0.1);
        for t in [-half, 0.0_f32, half] {
            let d = a * t.sin() + forward * t.cos();
            s.push(GizmoSpec::Lines {
                segments: vec![(Vec3::ZERO, d * self.perception.vision_range.max(0.1))],
                color: egui::Color32::from_rgb(120, 210, 255),
                world: false,
            });
        }
        s.push(GizmoSpec::Lines {
            segments: vec![(Vec3::ZERO, tip)],
            color: egui::Color32::from_rgb(120, 210, 255),
            world: false,
        });
        s
    }
}

impl Inspect for BehaviourTreeAgent {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Asset", &mut changed, |ui| ui.text_edit_singleline(&mut self.asset));
        ui.label(egui::RichText::new("Project-relative .behaviour path (authoring)").small().weak());
        changed |= edit_bt_node(ui, &mut self.root, 0);
        changed
    }
}

impl Gizmo for BehaviourTreeAgent {}

pub fn edit_bt_node(ui: &mut Ui, node: &mut zestengine_core::ai::BtNode, depth: usize) -> bool {
    use zestengine_core::ai::{BtAction, BtNode};
    let mut changed = false;
    ui.horizontal(|ui| {
        if depth > 0 {
            ui.label("↳");
        }
        let mut kind = node_kind(node);
        let mut resp = egui::ComboBox::from_id_salt(("zestengine-bt-kind", depth, kind))
            .selected_text(kind_label(kind))
            .show_ui(ui, |ui| {
                let mut clicked = false;
                for k in [0u8, 1, 2, 3, 4, 5] {
                    clicked |= ui.selectable_value(&mut kind, k, kind_label(k)).clicked();
                }
                clicked
            });
        if resp.inner == Some(true) {
            resp.response.mark_changed();
        }
        if kind != node_kind(node) {
            *node = match kind {
                0 => BtNode::Success,
                1 => BtNode::Failure,
                2 => BtNode::Selector { children: Vec::new() },
                3 => BtNode::Sequence { children: Vec::new() },
                4 => BtNode::Invert { child: Box::new(BtNode::Success) },
                5 => BtNode::Condition { key: "has_target".into(), value: true, invert: false },
                _ => BtNode::Action { action: BtAction::Idle, param: String::new() },
            };
            changed = true;
        }
    });

    match node {
        BtNode::Success | BtNode::Failure => {}
        BtNode::Invert { child } => {
            ui.indent(("zestengine-bt-indent", depth), |ui| {
                changed |= edit_bt_node(ui, child, depth + 1);
            });
        }
        BtNode::Condition { key, value, invert } => {
            changed |= ui.text_edit_singleline(key).changed();
            changed |= ui.checkbox(value, "value").changed();
            changed |= ui.checkbox(invert, "invert").changed();
        }
        BtNode::Action { action, param } => {
            let mut act = *action;
            let mut resp = egui::ComboBox::from_id_salt(("zestengine-bt-action", depth))
                .selected_text(act.label())
                .show_ui(ui, |ui| {
                    let mut clicked = false;
                    for k in BtAction::ALL {
                        clicked |= ui.selectable_value(&mut act, k, k.label()).clicked();
                    }
                    clicked
                });
            if resp.inner == Some(true) {
                resp.response.mark_changed();
            }
            if act != *action {
                *action = act;
                changed = true;
            }
            changed |= ui.text_edit_singleline(param).changed();
        }
        BtNode::Selector { children } | BtNode::Sequence { children } => {
            let mut remove = None;
            let mut up = None;
            let mut down = None;
            let child_count = children.len();
            for (i, child) in children.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    changed |= edit_bt_node(ui, child, depth + 1);
                    if ui.small_button("↑").clicked() && i > 0 {
                        up = Some(i);
                    }
                    if ui.small_button("↓").clicked() && i + 1 < child_count {
                        down = Some(i);
                    }
                    if ui.small_button("✕").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                children.remove(i);
                changed = true;
            }
            if let Some(i) = up {
                children.swap(i, i - 1);
                changed = true;
            }
            if let Some(i) = down {
                children.swap(i, i + 1);
                changed = true;
            }
            if ui.small_button("+ Add child").clicked() {
                children.push(BtNode::Action { action: BtAction::Idle, param: String::new() });
                changed = true;
            }
        }
    }
    changed
}

fn node_kind(node: &zestengine_core::ai::BtNode) -> u8 {
    use zestengine_core::ai::BtNode;
    match node {
        BtNode::Success => 0,
        BtNode::Failure => 1,
        BtNode::Selector { .. } => 2,
        BtNode::Sequence { .. } => 3,
        BtNode::Invert { .. } => 4,
        BtNode::Condition { .. } => 5,
        BtNode::Action { .. } => 6,
    }
}

fn kind_label(kind: u8) -> &'static str {
    match kind {
        0 => "Success",
        1 => "Failure",
        2 => "Selector",
        3 => "Sequence",
        4 => "Invert",
        5 => "Condition",
        _ => "Action",
    }
}

impl Inspect for PhysicsJoint {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        enum_row(ui, "Type", "zestengine-joint-kind", &mut self.kind, &JointKind::ALL, JointKind::label, &mut changed);
        changed |= ctx.object_ref(ui, "Connected body", &mut self.connected);
        if !self.connected.is_set() {
            ui.label(egui::RichText::new("Unset: attached to its start point in the world").small().weak());
        }
        axis_row(ui, "Anchor", &mut self.anchor, &mut changed);
        let k = self.kind;
        if matches!(k, JointKind::Hinge | JointKind::Slider) {
            axis_row(ui, "Axis", &mut self.axis, &mut changed);
            property_row(ui, "Limits", &mut changed, |ui| ui.checkbox(&mut self.limits_enabled, ""));
            if self.limits_enabled {
                let unit = if k == JointKind::Hinge { " deg" } else { " m" };
                ui.horizontal(|ui| {
                    ui.label("Range");
                    changed |= ui.add(DragValue::new(&mut self.limits[0]).speed(0.5).suffix(unit)).changed();
                    changed |= ui.add(DragValue::new(&mut self.limits[1]).speed(0.5).suffix(unit)).changed();
                });
            }
            property_row(ui, "Motor", &mut changed, |ui| ui.checkbox(&mut self.motor_enabled, ""));
            if self.motor_enabled {
                property_row(ui, "Motor speed", &mut changed, |ui| ui.add(DragValue::new(&mut self.motor_speed).speed(1.0)));
                property_row(ui, "Max force", &mut changed, |ui| {
                    ui.add(DragValue::new(&mut self.motor_max_force).speed(1.0).range(0.0..=1e7))
                });
            }
        }
        if matches!(k, JointKind::Spring | JointKind::Rope) {
            property_row(ui, "Rest length", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.rest_length).speed(0.01).range(0.0..=1000.0))
            });
            ui.label(egui::RichText::new("0 = distance at start").small().weak());
        }
        if k == JointKind::Spring {
            property_row(ui, "Stiffness", &mut changed, |ui| ui.add(DragValue::new(&mut self.stiffness).speed(0.5).range(0.0..=1e6)));
            property_row(ui, "Damping", &mut changed, |ui| ui.add(DragValue::new(&mut self.damping).speed(0.05).range(0.0..=1e4)));
        }
        property_row(ui, "Collide connected", &mut changed, |ui| ui.checkbox(&mut self.collide_connected, ""));
        ui.label(egui::RichText::new("Needs a collider + Rigid Body on this object").small().weak());
        changed
    }
}
impl Gizmo for PhysicsJoint {
    /// Anchor handle (drag to move it), the hinge / slider axis, the hinge's
    /// limit arc or the slider's limit ticks, and the spring / rope rest length.
    /// The line to the connected body is drawn by the editor (it resolves the
    /// reference).
    fn gizmos(&self) -> Vec<GizmoSpec> {
        use glam::Vec3;
        let anchor = Vec3::from(self.anchor);
        let axis = Vec3::from(self.axis).normalize_or(Vec3::Y);
        let color = egui::Color32::from_rgb(255, 200, 90);
        let mut g = vec![GizmoSpec::Handle { position: anchor, anchor: Vec3::ZERO, color }];
        let mut s = Vec::new();
        match self.kind {
            JointKind::Hinge => {
                s.push((anchor - axis * 0.5, anchor + axis * 0.5));
                let (a, b) = basis_of(axis);
                let (lo, hi) = if self.limits_enabled {
                    (self.limits[0].to_radians(), self.limits[1].to_radians())
                } else {
                    (0.0, std::f32::consts::TAU)
                };
                circle_segments(&mut s, anchor, a, b, 0.35, lo, hi, 32);
                if self.limits_enabled {
                    for t in [lo, hi] {
                        s.push((anchor, anchor + (a * t.cos() + b * t.sin()) * 0.35));
                    }
                }
            }
            JointKind::Slider => {
                let (lo, hi) = if self.limits_enabled { (self.limits[0], self.limits[1]) } else { (-0.5, 0.5) };
                s.push((anchor + axis * lo, anchor + axis * hi));
                let (a, _) = basis_of(axis);
                for t in [lo, hi] {
                    let p = anchor + axis * t;
                    s.push((p - a * 0.08, p + a * 0.08));
                }
            }
            JointKind::Ball => {
                let (a, b) = basis_of(axis);
                circle_segments(&mut s, anchor, a, b, 0.15, 0.0, std::f32::consts::TAU, 24);
                circle_segments(&mut s, anchor, a, axis, 0.15, 0.0, std::f32::consts::TAU, 24);
                circle_segments(&mut s, anchor, b, axis, 0.15, 0.0, std::f32::consts::TAU, 24);
            }
            JointKind::Spring | JointKind::Rope if self.rest_length > 0.0 => {
                let (a, b) = basis_of(axis);
                circle_segments(&mut s, anchor, a, b, self.rest_length, 0.0, std::f32::consts::TAU, 48);
            }
            _ => {}
        }
        if !s.is_empty() {
            g.push(GizmoSpec::Lines { segments: s, color, world: false });
        }
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Handle { position, .. } = edit {
            self.anchor = position.to_array();
        }
    }
}

impl Inspect for Animator {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let clips = self.state.clips.clone();
        changed |= ctx.file_field(ui, "Controller", &mut self.controller, "drop a .animctrl", &["animctrl"]);
        ui.horizontal(|ui| {
            ui.add_space(crate::theme::LABEL_WIDTH);
            if ui
                .add_enabled(!self.controller.is_empty(), egui::Button::new("Open"))
                .on_hover_text("Edit this controller in the Animator Controller tab")
                .clicked()
            {
                let path = self.controller.clone();
                ui.data_mut(|d| d.insert_temp(egui::Id::new(OPEN_ANIM_CONTROLLER_KEY), path));
            }
            if !self.controller.is_empty() && ui.button("Clear").clicked() {
                self.controller.clear();
                changed = true;
            }
        });
        let controlled = !self.controller.is_empty();
        if controlled {
            crate::widgets::info_note(ui, "States, transitions and parameters come from the controller.");
            for (layer, l) in self.state.layers.iter().enumerate() {
                ui.label(
                    egui::RichText::new(format!(
                        "Layer {layer}: {} ({:.2}){}",
                        l.state_name,
                        l.normalized_time,
                        if l.in_transition() { "  blending" } else { "" }
                    ))
                    .small()
                    .weak(),
                );
            }
        }
        let choices: Vec<String> = clips
            .iter()
            .cloned()
            .chain(self.blend_trees.iter().map(|t| t.name.clone()))
            .collect();
        if !controlled {
            property_row(ui, "Clip", &mut changed, |ui| {
                let shown = if self.clip.is_empty() { "(first clip)" } else { self.clip.as_str() };
                let mut resp = egui::ComboBox::from_id_salt("zestengine-animator-clip")
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        let mut clicked = ui.selectable_value(&mut self.clip, String::new(), "(first clip)").clicked();
                        for c in &choices {
                            clicked |= ui.selectable_value(&mut self.clip, c.clone(), c).clicked();
                        }
                        clicked
                    });
                if resp.inner == Some(true) {
                    resp.response.mark_changed();
                }
                resp.response
            });
            changed |= ctx.file_field(ui, "Clip asset", &mut self.clip, "or drop a .anim", &["anim"]);
            if clips.is_empty() && !self.clip.ends_with(".anim") {
                ui.label(egui::RichText::new("No clips found on this object's model").small().weak());
            }
        }
        property_row(ui, "Speed", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.speed).speed(0.02).range(-10.0..=10.0))
        });
        if !controlled {
            property_row(ui, "Loop", &mut changed, |ui| ui.checkbox(&mut self.looping, ""));
            property_row(ui, "Play on start", &mut changed, |ui| ui.checkbox(&mut self.play_on_start, ""));
        }
        property_row(ui, "Root motion", &mut changed, |ui| ui.checkbox(&mut self.root_motion, ""));
        property_row(ui, "GPU skinning", &mut changed, |ui| ui.checkbox(&mut self.gpu_skinning, ""));
        property_row(ui, "Update mode", &mut changed, |ui| {
            let mut c = false;
            let r = egui::ComboBox::from_id_salt("zestengine-animator-update-mode")
                .selected_text(self.update_mode.label())
                .show_ui(ui, |ui| {
                    for m in zestengine_core::AnimatorUpdateMode::ALL {
                        c |= ui.selectable_value(&mut self.update_mode, m, m.label()).changed();
                    }
                });
            let mut resp = r.response;
            if c {
                resp.mark_changed();
            }
            resp
        });
        property_row(ui, "Culling mode", &mut changed, |ui| {
            let mut c = false;
            let r = egui::ComboBox::from_id_salt("zestengine-animator-culling")
                .selected_text(self.culling.label())
                .show_ui(ui, |ui| {
                    for m in zestengine_core::AnimatorCullingMode::ALL {
                        c |= ui.selectable_value(&mut self.culling, m, m.label()).changed();
                    }
                });
            let mut resp = r.response.on_hover_text(
                "While no view sees the object (camera, its shadows, XR eyes), skinning, blendshapes and the GPU pose \
                 upload always stop.\nAlways Animate: the skeleton keeps being posed.\nCull Update Transforms: the state \
                 machine and events run on; bones are posed only for root motion, IK or a Bone Attach / Weapon Hold \
                 that follows them.\nCull Completely: nothing runs.",
            );
            if c {
                resp.mark_changed();
            }
            resp
        });

        let clip_combo = |ui: &mut Ui, salt: String, value: &mut String, clips: &[String]| -> bool {
            let mut clicked = false;
            egui::ComboBox::from_id_salt(salt)
                .selected_text(if value.is_empty() { "(clip)" } else { value.as_str() })
                .show_ui(ui, |ui| {
                    for c in clips {
                        clicked |= ui.selectable_value(value, c.clone(), c).clicked();
                    }
                });
            clicked
        };

        if !controlled {
            egui::CollapsingHeader::new(format!("Blend trees ({})", self.blend_trees.len()))
                .id_salt("zestengine-anim-trees")
                .show(ui, |ui| {
                    let mut remove = None;
                    for (t, tree) in self.blend_trees.iter_mut().enumerate() {
                        ui.separator();
                        property_row(ui, "Name", &mut changed, |ui| ui.text_edit_singleline(&mut tree.name));
                        property_row(ui, "Parameter", &mut changed, |ui| ui.text_edit_singleline(&mut tree.parameter));
                        property_row(ui, "Parameter Y (2D)", &mut changed, |ui| ui.text_edit_singleline(&mut tree.parameter_y));
                        let two_d = tree.is_2d();
                        let mut remove_motion = None;
                        for (m, motion) in tree.motions.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                changed |= clip_combo(ui, format!("zestengine-tree-{t}-{m}"), &mut motion.clip, &clips);
                                changed |= ui.add(DragValue::new(&mut motion.position[0]).speed(0.05)).changed();
                                if two_d {
                                    changed |= ui.add(DragValue::new(&mut motion.position[1]).speed(0.05)).changed();
                                }
                                if ui.small_button("x").clicked() {
                                    remove_motion = Some(m);
                                }
                            });
                        }
                        if let Some(m) = remove_motion {
                            tree.motions.remove(m);
                            changed = true;
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Add motion").clicked() {
                                tree.motions.push(Default::default());
                                changed = true;
                            }
                            if ui.button("Remove tree").clicked() {
                                remove = Some(t);
                            }
                        });
                    }
                    if let Some(t) = remove {
                        self.blend_trees.remove(t);
                        changed = true;
                    }
                    if ui.button("Add blend tree").clicked() {
                        self.blend_trees.push(zestengine_core::BlendTree {
                            name: format!("Blend {}", self.blend_trees.len() + 1),
                            parameter: "Speed".into(),
                            ..Default::default()
                        });
                        changed = true;
                    }
                });
        }

        egui::CollapsingHeader::new(format!("Parameter drivers ({})", self.drivers.len()))
            .id_salt("zestengine-anim-drivers")
            .show(ui, |ui| {
                crate::widgets::info_note(ui, "Feed a parameter every frame from movement, the ground check, or another Animator's parameter.");
                let mut remove = None;
                for (k, d) in self.drivers.iter_mut().enumerate() {
                    ui.push_id(("zestengine-anim-driver", k), |ui| {
                        crate::widgets::divider(ui);
                        property_row(ui, "Parameter", &mut changed, |ui| ui.text_edit_singleline(&mut d.parameter));
                        enum_row(ui, "Source", &format!("zestengine-driver-src-{k}"), &mut d.source, &zestengine_core::DriverSource::ALL, zestengine_core::DriverSource::label, &mut changed);
                        changed |= ctx.object_ref(ui, "From object", &mut d.object);
                        if d.source == zestengine_core::DriverSource::AnimatorParameter {
                            property_row(ui, "Its parameter", &mut changed, |ui| ui.text_edit_singleline(&mut d.source_parameter));
                        }
                        property_row(ui, "Scale", &mut changed, |ui| ui.add(DragValue::new(&mut d.scale).speed(0.02)));
                        if ui.small_button("Remove driver").clicked() {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    self.drivers.remove(k);
                    changed = true;
                }
                if ui.button("Add driver").clicked() {
                    self.drivers.push(zestengine_core::ParamDriver { parameter: "Speed".into(), ..Default::default() });
                    changed = true;
                }
            });

        egui::CollapsingHeader::new(format!("Events ({})", self.events.len()))
            .id_salt("zestengine-anim-events")
            .show(ui, |ui| {
                let mut remove = None;
                for (k, e) in self.events.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        changed |= clip_combo(ui, format!("zestengine-anim-ev-{k}"), &mut e.clip, &clips);
                        changed |= ui.add(DragValue::new(&mut e.time).speed(0.01).range(0.0..=1000.0).suffix(" s")).changed();
                        changed |= ui.add(egui::TextEdit::singleline(&mut e.name).desired_width(90.0)).changed();
                        if ui.small_button("x").clicked() {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    self.events.remove(k);
                    changed = true;
                }
                if ui.button("Add event").clicked() {
                    self.events.push(zestengine_core::AnimEvent {
                        clip: clips.first().cloned().unwrap_or_default(),
                        time: 0.0,
                        name: "Event".into(),
                    });
                    changed = true;
                }
            });
        if let Some(cur) = &self.state.current {
            ui.label(
                egui::RichText::new(format!("Playing: {cur} ({:.2}s / {:.2}s)", self.state.time, self.state.duration))
                    .small()
                    .weak(),
            );
        }
        changed
    }
}
impl Gizmo for Animator {}

fn rgba_row(ui: &mut Ui, label: &str, c: &mut [f32; 4], changed: &mut bool) {
    property_row(ui, label, changed, |ui| crate::widgets::color_hex(ui, c));
}

fn vec2_row(ui: &mut Ui, label: &str, v: &mut [f32; 2], speed: f64, changed: &mut bool) {
    crate::widgets::property_row(ui, label, |ui| {
        let w = ((ui.available_width() - 4.0) / 2.0).max(40.0);
        ui.spacing_mut().item_spacing.x = 4.0;
        for (a, value) in v.iter_mut().enumerate() {
            *changed |= crate::widgets::axis_field(ui, a, value, 2, speed, w).changed();
        }
    });
}

fn enum_row<T: Copy + PartialEq>(
    ui: &mut Ui,
    label: &str,
    salt: &str,
    value: &mut T,
    all: &[T],
    name: fn(T) -> &'static str,
    changed: &mut bool,
) {
    property_row(ui, label, changed, |ui| {
        let mut resp = egui::ComboBox::from_id_salt(salt)
            .width(ui.available_width())
            .selected_text(name(*value))
            .show_ui(ui, |ui| {
                let mut clicked = false;
                for &k in all {
                    clicked |= ui.selectable_value(value, k, name(k)).clicked();
                }
                clicked
            });
        if resp.inner == Some(true) {
            resp.response.mark_changed();
        }
        resp.response
    });
}

impl Inspect for UiCanvas {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        enum_row(ui, "Render mode", "zestengine-ui-render", &mut self.render_mode, &UiRenderMode::ALL, UiRenderMode::label, &mut changed);
        if self.render_mode == UiRenderMode::World {
            vec2_row(ui, "Size (m)", &mut self.world_size, 0.01, &mut changed);
            property_row(ui, "Pixels per meter", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.pixels_per_meter).speed(5.0).range(50.0..=4000.0))
            });
            let [w, h] = self.world_pixels();
            ui.label(
                egui::RichText::new(format!("{w} x {h} px panel in the object's XY plane, facing +Z"))
                    .small()
                    .weak(),
            );
        }
        enum_row(ui, "Scale mode", "zestengine-ui-scale", &mut self.scale_mode, &UiScaleMode::ALL, UiScaleMode::label, &mut changed);
        if self.scale_mode == UiScaleMode::ScaleWithScreen {
            vec2_row(ui, "Reference size", &mut self.reference_size, 1.0, &mut changed);
            property_row(ui, "Match width/height", &mut changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut self.match_height, 0.0..=1.0))
            });
        }
        property_row(ui, "Sort order", &mut changed, |ui| ui.add(DragValue::new(&mut self.sort_order)));
        ui.label(
            egui::RichText::new("Children with UI components are laid out inside this canvas.")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for UiCanvas {}

impl Inspect for UiRect {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        // Spec §8.3: 3x3 anchor preset grid (active anchor filled accent), plus
        // Stretch to fill the parent.
        crate::widgets::property_row(ui, "Anchor", |ui| {
            let cell = 18.0;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(cell * 3.0 + 8.0, cell * 3.0 + 8.0), Sense::hover());
            let p = ui.painter();
            p.rect(rect, crate::theme::R_FIELD, crate::theme::BG_FIELD, egui::Stroke::new(1.0_f32, crate::theme::LINE_FIELD), egui::StrokeKind::Inside);
            for row in 0..3 {
                for col in 0..3 {
                    let a = [col as f32 * 0.5, row as f32 * 0.5];
                    let r = egui::Rect::from_min_size(
                        rect.min + egui::vec2(4.0 + col as f32 * cell, 4.0 + row as f32 * cell),
                        egui::vec2(cell, cell),
                    )
                    .shrink(3.0);
                    let id = ui.id().with(("zestengine-anchor", row, col));
                    let resp = ui.interact(r, id, Sense::click());
                    let active = self.anchor_min == a && self.anchor_max == a;
                    let fill = if active {
                        crate::theme::accent()
                    } else if resp.hovered() {
                        crate::theme::TEXT_MUTED
                    } else {
                        crate::theme::BG_SEGMENT_ACTIVE
                    };
                    ui.painter().rect_filled(r, crate::theme::R_SWATCH, fill);
                    let name = UiRect::PRESETS.iter().find(|(_, x, y)| *x == a && *y == a).map_or("", |(n, _, _)| n);
                    if resp.on_hover_text(name).clicked() {
                        self.anchor_min = a;
                        self.anchor_max = a;
                        self.pivot = a;
                        changed = true;
                    }
                }
            }
            let stretch = self.anchor_min == [0.0, 0.0] && self.anchor_max == [1.0, 1.0];
            if crate::widgets::segment(ui, stretch, "Stretch").on_hover_text("Fill the parent").clicked() && !stretch {
                self.anchor_min = [0.0, 0.0];
                self.anchor_max = [1.0, 1.0];
                self.size = [0.0, 0.0];
                self.position = [0.0, 0.0];
                self.pivot = [0.5, 0.5];
                changed = true;
            }
        });
        vec2_row(ui, "Position", &mut self.position, 1.0, &mut changed);
        let stretch = self.anchor_min != self.anchor_max;
        vec2_row(ui, if stretch { "Size delta" } else { "Size" }, &mut self.size, 1.0, &mut changed);
        egui::CollapsingHeader::new("Anchors & pivot").id_salt("zestengine-ui-rect-adv").show(ui, |ui| {
            vec2_row(ui, "Anchor min", &mut self.anchor_min, 0.01, &mut changed);
            vec2_row(ui, "Anchor max", &mut self.anchor_max, 0.01, &mut changed);
            vec2_row(ui, "Pivot", &mut self.pivot, 0.01, &mut changed);
        });
        changed
    }
}
impl Gizmo for UiRect {}

impl Inspect for UiPanel {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        rgba_row(ui, "Color", &mut self.color, &mut changed);
        property_row(ui, "Corner radius", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.corner_radius).speed(0.2).range(0.0..=200.0))
        });
        property_row(ui, "Border width", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.border_width).speed(0.1).range(0.0..=50.0))
        });
        if self.border_width > 0.0 {
            rgba_row(ui, "Border color", &mut self.border_color, &mut changed);
        }
        changed
    }
}
impl Gizmo for UiPanel {}

impl Inspect for UiImage {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Texture", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.texture).hint_text("textures/icon.png"))
        });
        rgba_row(ui, "Tint", &mut self.tint, &mut changed);
        property_row(ui, "Preserve aspect", &mut changed, |ui| ui.checkbox(&mut self.preserve_aspect, ""));
        changed
    }
}
impl Gizmo for UiImage {}

impl Inspect for UiText {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.label("Text");
        if ui
            .add(egui::TextEdit::multiline(&mut self.text).desired_rows(2).desired_width(f32::INFINITY))
            .changed()
        {
            changed = true;
        }
        property_row(ui, "Localization key", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.loc_key).hint_text("use the text above"))
        });
        property_row(ui, "Rich text", &mut changed, |ui| ui.checkbox(&mut self.rich_text, ""));
        property_row(ui, "Size", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.size).speed(0.2).range(1.0..=500.0))
        });
        property_row(ui, "Auto size", &mut changed, |ui| ui.checkbox(&mut self.auto_size, ""));
        if self.auto_size {
            property_row(ui, "Auto size min", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.auto_size_min).speed(0.2).range(1.0..=500.0))
            });
            property_row(ui, "Auto size max", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.auto_size_max).speed(0.2).range(1.0..=500.0))
            });
        }
        rgba_row(ui, "Color", &mut self.color, &mut changed);
        property_row(ui, "Bold", &mut changed, |ui| ui.checkbox(&mut self.bold, ""));
        property_row(ui, "Italic", &mut changed, |ui| ui.checkbox(&mut self.italic, ""));
        enum_row(ui, "Align X", "zestengine-ui-ax", &mut self.align_x, &UiAlign::ALL, UiAlign::label, &mut changed);
        enum_row(ui, "Align Y", "zestengine-ui-ay", &mut self.align_y, &UiAlign::ALL, UiAlign::label, &mut changed);
        property_row(ui, "Wrap", &mut changed, |ui| ui.checkbox(&mut self.wrap, ""));
        property_row(ui, "Justify", &mut changed, |ui| ui.checkbox(&mut self.justify, ""));
        enum_row(
            ui, "Overflow", "zestengine-ui-overflow", &mut self.overflow, &UiTextOverflow::ALL, UiTextOverflow::label,
            &mut changed,
        );
        for (label, v) in [
            ("Character spacing", &mut self.char_spacing),
            ("Word spacing", &mut self.word_spacing),
            ("Line spacing", &mut self.line_spacing),
            ("Paragraph spacing", &mut self.paragraph_spacing),
        ] {
            property_row(ui, label, &mut changed, |ui| ui.add(DragValue::new(v).speed(0.5).range(-100.0..=400.0)));
        }
        for (label, i) in [("Margin left", 0), ("Margin top", 1), ("Margin right", 2), ("Margin bottom", 3)] {
            property_row(ui, label, &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.margins[i]).speed(0.5).range(0.0..=500.0))
            });
        }
        property_row(ui, "Outline width", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.outline_width).speed(0.1).range(0.0..=40.0))
        });
        if self.outline_width > 0.0 {
            rgba_row(ui, "Outline color", &mut self.outline_color, &mut changed);
        }
        property_row(ui, "Font", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.font).hint_text("built-in (.ttf .otf .zfont .zfontstack)"))
        });
        enum_row(
            ui, "Renderer", "zestengine-ui-renderer", &mut self.renderer, &UiTextRenderer::ALL, UiTextRenderer::label,
            &mut changed,
        );
        changed
    }
}
impl Gizmo for UiText {}

impl Inspect for Text3D {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.label("Text");
        if ui
            .add(egui::TextEdit::multiline(&mut self.text).desired_rows(2).desired_width(f32::INFINITY))
            .changed()
        {
            changed = true;
        }
        property_row(ui, "Localization key", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.loc_key).hint_text("use the text above"))
        });
        property_row(ui, "Rich text", &mut changed, |ui| ui.checkbox(&mut self.rich_text, ""));
        property_row(ui, "Size (m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.size).speed(0.005).range(0.001..=20.0))
        });
        property_row(ui, "Auto size", &mut changed, |ui| ui.checkbox(&mut self.auto_size, ""));
        if self.auto_size {
            property_row(ui, "Auto size min", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.auto_size_min).speed(0.002).range(0.001..=20.0))
            });
            property_row(ui, "Auto size max", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.auto_size_max).speed(0.002).range(0.001..=20.0))
            });
        }
        rgba_row(ui, "Color", &mut self.color, &mut changed);
        property_row(ui, "Bold", &mut changed, |ui| ui.checkbox(&mut self.bold, ""));
        property_row(ui, "Italic", &mut changed, |ui| ui.checkbox(&mut self.italic, ""));
        property_row(ui, "Width (m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.width).speed(0.01).range(0.0..=100.0))
        });
        property_row(ui, "Height (m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.height).speed(0.01).range(0.0..=100.0))
        });
        enum_row(ui, "Align X", "zestengine-t3d-ax", &mut self.align_x, &UiAlign::ALL, UiAlign::label, &mut changed);
        enum_row(ui, "Align Y", "zestengine-t3d-ay", &mut self.align_y, &UiAlign::ALL, UiAlign::label, &mut changed);
        property_row(ui, "Wrap", &mut changed, |ui| ui.checkbox(&mut self.wrap, ""));
        enum_row(
            ui, "Overflow", "zestengine-t3d-overflow", &mut self.overflow, &UiTextOverflow::ALL, UiTextOverflow::label,
            &mut changed,
        );
        property_row(ui, "Character spacing", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.char_spacing).speed(0.5).range(-100.0..=400.0))
        });
        property_row(ui, "Line spacing", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.line_spacing).speed(0.5).range(-100.0..=400.0))
        });
        property_row(ui, "Outline width", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.outline_width).speed(0.1).range(0.0..=40.0))
        });
        if self.outline_width > 0.0 {
            rgba_row(ui, "Outline color", &mut self.outline_color, &mut changed);
        }
        enum_row(
            ui, "Billboard", "zestengine-t3d-billboard", &mut self.billboard, &Text3DBillboard::ALL,
            Text3DBillboard::label, &mut changed,
        );
        property_row(ui, "Font", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.font).hint_text("built-in (.ttf .otf .zfont .zfontstack)"))
        });
        changed
    }
}
impl Gizmo for Text3D {}

impl Inspect for UiButton {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Interactable", &mut changed, |ui| ui.checkbox(&mut self.interactable, ""));
        rgba_row(ui, "Normal", &mut self.normal, &mut changed);
        rgba_row(ui, "Hovered", &mut self.hovered, &mut changed);
        rgba_row(ui, "Pressed", &mut self.pressed, &mut changed);
        rgba_row(ui, "Disabled", &mut self.disabled, &mut changed);
        ui.separator();
        ui.label("On click");
        let mut remove = None;
        for (k, action) in self.on_click.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("zestengine-ui-action", k))
                    .selected_text(action.label())
                    .width(110.0)
                    .show_ui(ui, |ui| {
                        for kind in UiAction::kinds() {
                            let same = std::mem::discriminant(&kind) == std::mem::discriminant(action);
                            if ui.selectable_label(same, kind.label()).clicked() && !same {
                                *action = action.with_kind_of(&kind);
                                changed = true;
                            }
                        }
                    });
                let hint = if matches!(action, UiAction::LoadScene(_)) { "scenes/level.scene" } else { "object name" };
                if let Some(target) = action.target_mut() {
                    let w = (ui.available_width() - 28.0).max(40.0);
                    changed |= ui.add(egui::TextEdit::singleline(target).hint_text(hint).desired_width(w)).changed();
                }
                if ui.small_button(ph::X).on_hover_text("Remove action").clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            self.on_click.remove(k);
            changed = true;
        }
        if ui.button(format!("{} Add action", ph::PLUS)).clicked() {
            self.on_click.push(UiAction::Show(String::new()));
            changed = true;
        }
        ui.separator();
        let fields: &[(&str, zestengine_core::ActionType)] = &[("object", zestengine_core::ActionType::Object)];
        changed |= crate::actions_ui::event_actions_ui(ui, ctx, "on-click-actions", "On click: call", &mut self.on_click_actions, fields);
        ui.label(egui::RichText::new("Code: ctx.ui_clicked(button_ref)").small().weak());
        changed
    }
}
impl Gizmo for UiButton {}

impl Inspect for UiToggle {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "On", &mut changed, |ui| ui.checkbox(&mut self.on, ""));
        property_row(ui, "Interactable", &mut changed, |ui| ui.checkbox(&mut self.interactable, ""));
        rgba_row(ui, "Box", &mut self.box_color, &mut changed);
        rgba_row(ui, "Check", &mut self.check_color, &mut changed);
        property_row(ui, "Inverts Y of", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.invert_y_action).hint_text("none (e.g. Look)"))
                .on_hover_text("An Invert Look setting: inverts this action's vertical axis in the player's bindings (every control scheme that has it)")
        });
        changed
    }
}
impl Gizmo for UiToggle {}

impl Inspect for UiSlider {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Min", &mut changed, |ui| ui.add(DragValue::new(&mut self.min).speed(0.1)));
        property_row(ui, "Max", &mut changed, |ui| ui.add(DragValue::new(&mut self.max).speed(0.1)));
        let (lo, hi) = (self.min.min(self.max), self.max.max(self.min));
        property_row(ui, "Value", &mut changed, |ui| crate::widgets::slider(ui, egui::Slider::new(&mut self.value, lo..=hi)));
        property_row(ui, "Whole numbers", &mut changed, |ui| ui.checkbox(&mut self.whole_numbers, ""));
        property_row(ui, "Interactable", &mut changed, |ui| ui.checkbox(&mut self.interactable, ""));
        rgba_row(ui, "Track", &mut self.track_color, &mut changed);
        rgba_row(ui, "Fill", &mut self.fill_color, &mut changed);
        rgba_row(ui, "Handle", &mut self.handle_color, &mut changed);
        property_row(ui, "Audio bus", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.audio_bus).hint_text("none (e.g. Music)"))
                .on_hover_text("Drives this Audio Mixer bus's volume (0..1 along the slider)")
        });
        changed
    }
}
impl Gizmo for UiSlider {}

impl Inspect for UiProgressBar {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Value", &mut changed, |ui| crate::widgets::slider(ui, egui::Slider::new(&mut self.value, 0.0..=1.0)));
        enum_row(ui, "Direction", "zestengine-ui-fill", &mut self.direction, &UiFillDirection::ALL, UiFillDirection::label, &mut changed);
        rgba_row(ui, "Background", &mut self.background, &mut changed);
        rgba_row(ui, "Fill", &mut self.fill, &mut changed);
        property_row(ui, "Corner radius", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.corner_radius).speed(0.2).range(0.0..=200.0))
        });
        ui.label(egui::RichText::new("Gameplay: ctx.modify::<UiProgressBar>(bar, |b| b.value = hp)").small().weak());
        changed
    }
}
impl Gizmo for UiProgressBar {}

impl Inspect for UiPauseMenu {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Panel", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.panel).hint_text("object name"))
        });
        property_row(ui, "Pause time", &mut changed, |ui| ui.checkbox(&mut self.pause_time, ""));
        property_row(ui, "Lock cursor in game", &mut changed, |ui| ui.checkbox(&mut self.lock_cursor_in_game, ""));
        ui.label(
            egui::RichText::new("Escape / gamepad back toggles the panel. Resume button: Hide <panel>.")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for UiPauseMenu {}

impl Inspect for UiTextField {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Text", &mut changed, |ui| ui.text_edit_singleline(&mut self.text));
        property_row(ui, "Placeholder", &mut changed, |ui| ui.text_edit_singleline(&mut self.placeholder));
        property_row(ui, "Max length", &mut changed, |ui| ui.add(DragValue::new(&mut self.max_length).range(0..=10000)));
        property_row(ui, "Size", &mut changed, |ui| ui.add(DragValue::new(&mut self.size).speed(0.2).range(1.0..=300.0)));
        rgba_row(ui, "Text color", &mut self.text_color, &mut changed);
        rgba_row(ui, "Placeholder color", &mut self.placeholder_color, &mut changed);
        property_row(ui, "Password", &mut changed, |ui| ui.checkbox(&mut self.password, ""));
        property_row(ui, "Interactable", &mut changed, |ui| ui.checkbox(&mut self.interactable, ""));
        ui.label(egui::RichText::new("Gameplay: ctx.ui_submitted(field) on Enter").small().weak());
        changed
    }
}
impl Gizmo for UiTextField {}

impl Inspect for LightProbeVolume {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Size", &mut self.size, &mut changed);
        property_row(ui, "Density (/m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.density).speed(0.05).range(0.05..=16.0))
        });
        property_row(ui, "Blend Distance (m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.blend_distance).speed(0.05).range(0.0..=50.0))
                .on_hover_text("0 = classic probe volume (cascade). Above 0 the volume is a Light Volume: baked light for non-lightmapped objects, fading out over this distance at its faces and blended by weight with overlapping Light Volumes.")
        });
        if self.blend_distance > 0.0 {
            property_row(ui, "Weight", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.weight).speed(0.02).range(0.0..=16.0))
                    .on_hover_text("Share of this Light Volume where several overlap (times its edge fade)")
            });
        }
        let [x, y, z] = self.counts();
        ui.horizontal(|ui| {
            ui.label("Probes");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("{x} × {y} × {z} = {}", x * y * z)).weak());
            });
        });
        changed
    }
}
impl Gizmo for LightProbeVolume {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut g = vec![GizmoSpec::Box {
            center: glam::Vec3::ZERO,
            half_extents: glam::Vec3::from(self.size) * 0.5,
            object_anchored: true,
            blend: self.blend_distance,
            color: egui::Color32::from_rgb(120, 200, 255),
        }];
        if let Some(positions) = display_points(self.probe_count(), || self.local_positions()) {
            g.push(GizmoSpec::Points {
                positions,
                color: egui::Color32::from_rgb(180, 225, 255),
            });
        }
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit {
            self.size = (half_extents * 2.0).to_array();
        }
    }
}

impl Inspect for ReflectionProbe {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Size", &mut self.size, &mut changed);
        // Unity's box offset: the box aligns to the room while the OBJECT stays
        // where reflections should look right from (eye height).
        axis_row(ui, "Box Offset", &mut self.box_offset, &mut changed);
        // Where the cube is rendered from, relative to the object; the pink
        // handle in the viewport drags it without moving the box.
        axis_row(ui, "Capture Offset", &mut self.capture_offset, &mut changed);
        property_row(ui, "Lock box in world", &mut changed, |ui| {
            ui.checkbox(&mut self.lock_box, "")
                .on_hover_text(
                    "Freeze the box in world space: moving the probe object then only \
                     repositions the capture point inside the box (Box Offset auto-compensates). \
                     Align the box to the room first, then enable this and place the capture \
                     point freely.",
                )
        });
        ui.horizontal(|ui| {
            ui.label("Resolution");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let res_label = |r: u32| match r {
                    8192 => "8K".to_string(),
                    4096 => "4K".to_string(),
                    2048 => "2K".to_string(),
                    1024 => "1K".to_string(),
                    other => format!("{other}"),
                };
                egui::ComboBox::from_id_salt("zestengine-refl-probe-res")
                    .selected_text(res_label(self.resolution))
                    .show_ui(ui, |ui| {
                        for r in [8192u32, 4096, 2048, 1024, 512, 256, 128, 64, 32] {
                            changed |= ui
                                .selectable_value(&mut self.resolution, r, res_label(r))
                                .changed();
                        }
                    });
            });
        });
        if self.resolution >= 4096 {
            ui.label(
                RichText::new("High resolutions cost a lot of VRAM and bake time (8K ≈ GBs).")
                    .small()
                    .color(egui::Color32::from_rgb(220, 170, 110)),
            );
        }
        ui.horizontal(|ui| {
            ui.label("Mode");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                use zestengine_core::ReflectionProbeMode as M;
                egui::ComboBox::from_id_salt("zestengine-refl-probe-mode")
                    .selected_text(match self.mode {
                        M::Realtime => "Realtime",
                        M::Baked => "Baked",
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut self.mode, M::Realtime, "Realtime")
                            .changed();
                        changed |= ui
                            .selectable_value(&mut self.mode, M::Baked, "Baked")
                            .changed();
                    });
            });
        });
        if self.mode == zestengine_core::ReflectionProbeMode::Realtime {
            property_row(ui, "Update rate", &mut changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut self.update_hz, 0.0..=120.0).suffix(" Hz"))
                    .on_hover_text("Fresh cubes per second; 0 = every frame")
            });
            property_row(ui, "Time slicing", &mut changed, |ui| {
                ui.checkbox(&mut self.time_slicing, "")
                    .on_hover_text("Render one face per frame: cheaper per frame, but moving objects step in the reflection")
            });
            property_row(ui, "Moving objects", &mut changed, |ui| {
                ui.checkbox(&mut self.include_dynamic, "").on_hover_text(
                    "Also capture non-static objects. A cubemap paints everything onto the box walls, \
                     so a moving object inside the room appears as a ghost on far surfaces; \
                     leave off and let SSR / ray-traced reflections show moving objects",
                )
            });
        }
        if ui
            .button(format!("{} Bake This Probe", ph::SPARKLE))
            .on_hover_text("Capture this probe's cubemap from the current scene and save it as a .reflprobe sidecar")
            .clicked()
        {
            self.bake_now = true;
            // Not a serialized edit, but return changed so the engine's per-frame
            // poll runs this frame.
            changed = true;
        }
        property_row(ui, "Intensity", &mut changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut self.intensity, 0.0..=2.0))
        });
        property_row(ui, "Blend distance", &mut changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut self.blend_distance, 0.0..=10.0).suffix(" m"))
        });
        property_row(ui, "Importance", &mut changed, |ui| {
            ui.add(egui::DragValue::new(&mut self.importance).range(-8..=8))
        });
        property_row(ui, "Box projection", &mut changed, |ui| {
            ui.checkbox(&mut self.box_projection, "")
        });
        ui.label(
            RichText::new(
                "Box-projected cubemap reflection. Blend distance fades the probe to the \
                 skybox (and to overlapping probes) inside the box edge; importance breaks \
                 overlap ties. Recapture/bake from FluxBaker after editing the scene.",
            )
            .small()
            .weak(),
        );
        changed
    }
}
impl Inspect for PlanarReflection {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Max roughness", &mut changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut self.max_roughness, 0.05..=0.95))
        });
        ui.label(
            RichText::new(
                "True mirror: the scene re-renders every frame from the camera \
                 mirrored about this object's local +Y (up) plane — put it on a floor \
                 or a wall-mirror quad. Fades back to the reflection probe above the \
                 roughness cutoff. Each active surface costs a scene pass (max 4).",
            )
            .small()
            .weak(),
        );
        changed
    }
}
impl Gizmo for PlanarReflection {}

impl Inspect for DecalProjector {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ctx.file_field(
            ui,
            "Texture",
            &mut self.texture,
            "texture/foo.png",
            &["png", "jpg", "jpeg", "tga", "bmp", "webp", "exr", "hdr"],
        );
        vec2_row(ui, "Size", &mut self.size, 0.02, &mut changed);
        for s in &mut self.size {
            *s = s.max(0.01);
        }
        property_row(ui, "Depth", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.depth).speed(0.02).range(0.01..=1000.0))
        });
        property_row(ui, "Fade", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.fade).speed(0.01).range(0.0..=1.0))
        });
        vec2_row(ui, "Tiling", &mut self.tiling, 0.02, &mut changed);
        for t in &mut self.tiling {
            *t = t.max(0.01);
        }
        rgba_row(ui, "Tint", &mut self.tint, &mut changed);
        changed
    }
}

impl Gizmo for DecalProjector {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let half = glam::Vec3::new(self.size[0].max(0.0) * 0.5, self.size[1].max(0.0) * 0.5, self.depth.max(0.0) * 0.5);
        let center = glam::Vec3::new(0.0, 0.0, -half.z);
        let corners = [
            center + glam::Vec3::new(-half.x, -half.y, -half.z),
            center + glam::Vec3::new(half.x, -half.y, -half.z),
            center + glam::Vec3::new(half.x, half.y, -half.z),
            center + glam::Vec3::new(-half.x, half.y, -half.z),
            center + glam::Vec3::new(-half.x, -half.y, half.z),
            center + glam::Vec3::new(half.x, -half.y, half.z),
            center + glam::Vec3::new(half.x, half.y, half.z),
            center + glam::Vec3::new(-half.x, half.y, half.z),
        ];
        let mut segments = Vec::new();
        for i in 0..4 {
            let j = (i + 1) & 3;
            segments.push((corners[i], corners[j]));
            segments.push((corners[4 + i], corners[4 + j]));
            segments.push((corners[i], corners[4 + i]));
        }
        vec![GizmoSpec::Lines {
            segments,
            color: egui::Color32::from_rgb(120, 200, 255),
            world: false,
        }]
    }
}

impl Gizmo for ReflectionProbe {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![
            GizmoSpec::Box {
                center: glam::Vec3::from(self.box_offset),
                half_extents: glam::Vec3::from(self.size) * 0.5,
                object_anchored: true,
                blend: self.blend_distance.max(0.0),
                color: egui::Color32::from_rgb(150, 220, 255),
            },
            GizmoSpec::Handle {
                position: glam::Vec3::from(self.capture_offset),
                anchor: glam::Vec3::from(self.box_offset),
                color: crate::theme::TYPE_GI,
            },
        ]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        match edit {
            GizmoEdit::Box { half_extents, .. } => self.size = (half_extents * 2.0).to_array(),
            // Keep the capture point inside the box.
            GizmoEdit::Handle { position, .. } => {
                let center = glam::Vec3::from(self.box_offset);
                let half = glam::Vec3::from(self.size) * 0.5;
                self.capture_offset = position.clamp(center - half, center + half).to_array();
            }
            _ => {}
        }
    }
}

impl Inspect for FluxVoxelLight {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Color");
            changed |= ui.color_edit_button_rgb(&mut self.color).changed();
        });
        property_row(ui, "Intensity", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.intensity).speed(0.05).range(0.0..=64.0))
        });
        property_row(ui, "Range (m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.range).speed(0.1).range(0.1..=256.0))
        });
        changed
    }
}
impl Gizmo for FluxVoxelLight {}

impl Inspect for FluxVolume {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Size", &mut self.size, &mut changed);
        property_row(ui, "Density (/m)", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.density).speed(0.05).range(0.05..=8.0))
        });
        let [x, y, z] = self.counts();
        ui.horizontal(|ui| {
            ui.label("Voxels");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("{x} × {y} × {z} = {}", x * y * z)).weak());
            });
        });
        changed
    }
}
impl Gizmo for FluxVolume {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut g = vec![GizmoSpec::Box {
            center: glam::Vec3::ZERO,
            half_extents: glam::Vec3::from(self.size) * 0.5,
            object_anchored: true,
            blend: 0.0,
            color: egui::Color32::from_rgb(170, 140, 255),
        }];
        if let Some(positions) = display_points(self.probe_count(), || self.local_positions()) {
            g.push(GizmoSpec::Points {
                positions,
                color: egui::Color32::from_rgb(200, 170, 255),
            });
        }
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit {
            self.size = (half_extents * 2.0).to_array();
        }
    }
}

impl Inspect for AudioSource {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ctx.file_field(ui, "Clip", &mut self.clip, "audio/foo.wav", &["wav", "flac", "mp3", "ogg"]);
        property_row(ui, "Mixer bus", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.bus).desired_width(110.0))
        });
        ui.label(
            RichText::new(".wav · .flac · .mp3 · .ogg (project-relative)")
                .small()
                .weak(),
        );
        property_row(ui, "Play on Start", &mut changed, |ui| {
            ui.checkbox(&mut self.play_on_start, "")
        });
        property_row(ui, "Loop", &mut changed, |ui| {
            ui.checkbox(&mut self.looping, "")
        });
        property_row(ui, "Volume", &mut changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut self.volume, 0.0..=2.0))
        });
        property_row(ui, "Pitch", &mut changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut self.pitch, 0.25..=4.0))
        });
        property_row(ui, "Spatial (3D)", &mut changed, |ui| {
            ui.checkbox(&mut self.spatial, "")
        });
        if self.spatial {
            property_row(ui, "Min Distance", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.min_distance).speed(0.1).range(0.0..=10000.0))
            });
            property_row(ui, "Max Distance", &mut changed, |ui| {
                ui.add(DragValue::new(&mut self.max_distance).speed(0.5).range(0.0..=10000.0))
            });
            if self.max_distance < self.min_distance + 0.01 {
                self.max_distance = self.min_distance + 0.01;
            }
            ui.horizontal(|ui| {
                ui.label("Rolloff");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_salt("zestengine-audio-rolloff")
                        .selected_text(self.rolloff.label())
                        .show_ui(ui, |ui| {
                            for r in AudioRolloff::ALL {
                                changed |= ui
                                    .selectable_value(&mut self.rolloff, r, r.label())
                                    .changed();
                            }
                        });
                });
            });
            property_row(ui, "HRTF (binaural)", &mut changed, |ui| {
                ui.checkbox(&mut self.hrtf, "").on_hover_text("Off: plain stereo panning")
            });
            property_row(ui, "Air absorption", &mut changed, |ui| ui.checkbox(&mut self.air_absorption, ""));
            property_row(ui, "Occlusion", &mut changed, |ui| {
                ui.checkbox(&mut self.occlusion, "")
                    .on_hover_text("Static geometry between source and listener muffles it")
            });
            property_row(ui, "Directivity", &mut changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut self.directivity, 0.0..=1.0))
                    .on_hover_text("0 omnidirectional, 0.5 cardioid, 1 dipole along the object's forward (-Z)")
            });
            if self.directivity > 0.0 {
                property_row(ui, "Sharpness", &mut changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut self.directivity_sharpness, 0.0..=8.0))
                });
            }
            property_row(ui, "Reflections", &mut changed, |ui| {
                ui.checkbox(&mut self.reflections, "")
                    .on_hover_text("Simulate this source's own reflections; off uses the listener reverb")
            });
            property_row(ui, "Pathing", &mut changed, |ui| {
                ui.checkbox(&mut self.pathing, "")
                    .on_hover_text("Sound bends around corners (needs a baked Audio Probe Volume)")
            });
            property_row(ui, "Via ambisonics", &mut changed, |ui| {
                ui.checkbox(&mut self.spatialize_via_ambisonics, "")
                    .on_hover_text("Encode to ambisonics and decode instead of direct HRTF")
            });
        }
        const ORDERS: [u8; 4] = [0, 1, 2, 3];
        enum_row(ui, "Ambisonic clip", "zestengine-audio-ambi", &mut self.ambisonic_order, &ORDERS, |o| match o {
            0 => "No",
            1 => "1st order",
            2 => "2nd order",
            _ => "3rd order",
        }, &mut changed);
        changed
    }
}
impl Gizmo for AudioSource {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![
            GizmoSpec::Range {
                center: glam::Vec3::ZERO,
                radius: self.min_distance,
                color: egui::Color32::from_rgb(120, 200, 255),
            },
            GizmoSpec::Range {
                center: glam::Vec3::ZERO,
                radius: self.max_distance,
                color: egui::Color32::from_rgb(90, 140, 200),
            },
        ]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { index, radius } = edit {
            let r = radius.max(0.0);
            match index {
                0 => self.min_distance = r.min(self.max_distance),
                1 => self.max_distance = r.max(self.min_distance),
                _ => {}
            }
        }
    }
}

impl Inspect for AudioListener {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        ui.label(
            RichText::new("Ears for spatial audio (put this on the main camera).")
                .small()
                .weak(),
        );
        false
    }
}
impl Gizmo for AudioListener {}

impl Inspect for BoxCollider {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Center", &mut self.center, &mut changed);
        axis_row(ui, "Size", &mut self.size, &mut changed);
        for s in &mut self.size {
            *s = s.max(0.01);
        }
        collider_common_ui(ui, ctx, &mut self.material, &mut self.is_trigger, &mut self.layer, &mut changed);
        changed
    }
}
impl Gizmo for BoxCollider {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Box {
            center: glam::Vec3::from(self.center),
            half_extents: glam::Vec3::from(self.size) * 0.5,
            object_anchored: false,
            blend: 0.0,
            color: egui::Color32::from_rgb(240, 220, 70),
        }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box {
            half_extents,
            center,
            ..
        } = edit
        {
            self.size = (half_extents * 2.0).to_array();
            self.center = center.to_array();
        }
    }
}

impl Inspect for TerrainCollider {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        collider_common_ui(ui, ctx, &mut self.material, &mut self.is_trigger, &mut self.layer, &mut changed);
        changed
    }
}
impl Gizmo for TerrainCollider {}

impl Inspect for SphereCollider {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Center", &mut self.center, &mut changed);
        property_row(ui, "Radius", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.radius).speed(0.05).range(0.01..=10000.0))
        });
        collider_common_ui(ui, ctx, &mut self.material, &mut self.is_trigger, &mut self.layer, &mut changed);
        changed
    }
}
impl Gizmo for SphereCollider {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Range {
            center: glam::Vec3::from(self.center),
            radius: self.radius,
            color: egui::Color32::from_rgb(240, 220, 70),
        }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { radius, .. } = edit {
            self.radius = radius.max(0.0);
        }
    }
}

impl Inspect for CapsuleCollider {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Center", &mut self.center, &mut changed);
        property_row(ui, "Radius", &mut changed, |ui| ui.add(DragValue::new(&mut self.radius).speed(0.01).range(0.01..=1000.0)));
        property_row(ui, "Height", &mut changed, |ui| ui.add(DragValue::new(&mut self.height).speed(0.01).range(0.02..=1000.0)));
        crate::widgets::property_row(ui, "Direction", |ui| {
            let opts: Vec<(ColliderAxis, &str)> = ColliderAxis::ALL.iter().map(|a| (*a, a.label())).collect();
            changed |= crate::widgets::segmented(ui, &mut self.direction, &opts).changed();
        });
        collider_common_ui(ui, ctx, &mut self.material, &mut self.is_trigger, &mut self.layer, &mut changed);
        changed
    }
}
impl Gizmo for CapsuleCollider {
    /// The capsule outline along its axis, plus a radius handle.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let center = glam::Vec3::from(self.center);
        // Build along +Y, then turn onto the chosen axis.
        let turn = match self.direction {
            ColliderAxis::X => glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
            ColliderAxis::Y => glam::Quat::IDENTITY,
            ColliderAxis::Z => glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        };
        let segments = capsule_segments(glam::Vec3::ZERO, self.radius.max(0.01), self.height.max(self.radius * 2.0))
            .into_iter()
            .map(|(a, b)| (center + turn * a, center + turn * b))
            .collect();
        vec![
            GizmoSpec::Lines { segments, color: egui::Color32::from_rgb(240, 220, 70), world: false },
            GizmoSpec::Range { center, radius: self.radius, color: egui::Color32::from_rgb(240, 220, 70) },
        ]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { radius, .. } = edit {
            self.radius = radius.max(0.01);
            self.height = self.height.max(self.radius * 2.0);
        }
    }
}

impl Inspect for MeshCollider {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let authored: Option<(usize, String)> =
            ui.data(|d| d.get_temp::<Option<(usize, String)>>(egui::Id::new(AUTHORED_COLLISION_KEY))).flatten().filter(|(n, _)| *n > 0);
        if let Some((count, kinds)) = &authored {
            property_row(ui, "Use authored", &mut changed, |ui| {
                ui.checkbox(&mut self.use_authored, "").on_hover_text(
                    "Collide with the collision pieces modelled with this mesh (UCX_ convex, UBX_ box, USP_ sphere, UCP_ capsule objects) instead of building a shape from the mesh. Applies to static and moving bodies.",
                )
            });
            if self.use_authored {
                let s = if *count == 1 { "" } else { "s" };
                ui.label(RichText::new(format!("Using {count} authored collision piece{s} ({kinds})")).small().weak());
            }
        }
        let from_mesh = !(authored.is_some() && self.use_authored);
        ui.add_enabled_ui(from_mesh, |ui| {
            property_row(ui, "Convex", &mut changed, |ui| {
                ui.checkbox(&mut self.convex, "")
                    .on_hover_text("One convex hull even when static. Off: exact triangles when static; moving bodies always use a hull, or pieces with Follow shape")
            });
            property_row(ui, "Detail", &mut changed, |ui| {
                let opts: Vec<(zestengine_core::ColliderDetail, &str)> = zestengine_core::ColliderDetail::ALL.iter().map(|d| (*d, d.label())).collect();
                crate::widgets::segmented(ui, &mut self.detail, &opts)
                    .on_hover_text("How closely moving-body hulls and pieces hug the mesh (hull corners: 64 / 128 / 256; Follow-shape pieces: 8 / 16 / 32). Ball-shaped meshes always become an exact sphere.")
            });
            property_row(ui, "Follow shape", &mut changed, |ui| {
                ui.add_enabled(!self.convex, egui::Checkbox::without_text(&mut self.follow_shape)).on_hover_text(
                    "Moving bodies: split into a few convex pieces that follow a concave shape (a couch's seat and arms) instead of one hull around it all. Slower to build.",
                )
            });
        });
        collider_common_ui(ui, ctx, &mut self.material, &mut self.is_trigger, &mut self.layer, &mut changed);
        let note = if from_mesh {
            "Uses this object's mesh: exact triangles when static, one simplified hull on a moving Rigid Body (or pieces with Follow shape). Shown in green."
        } else {
            "Uses the authored pieces as a compound shape, static or moving. Shown in green."
        };
        ui.label(RichText::new(note).small().weak());
        changed
    }
}
impl Gizmo for MeshCollider {}

impl Inspect for Pawn {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Mode");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("zestengine-pawn-mode")
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        for m in ControlMode::ALL {
                            changed |= ui.selectable_value(&mut self.mode, m, m.label()).changed();
                        }
                    });
            });
        });
        property_row(ui, "Possessed", &mut changed, |ui| {
            ui.checkbox(&mut self.possessed, "")
                .on_hover_text("The local player controls this pawn (receives input)")
        });
        changed |= ctx.object_ref(ui, "Camera", &mut self.camera);
        property_row(ui, "Spawn Tag", &mut changed, |ui| {
            ui.text_edit_singleline(&mut self.spawn_tag)
                .on_hover_text("Teleport to a SpawnPoint with this tag on Play start")
        });
        property_row(ui, "Move Speed", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.move_speed).speed(0.1).range(0.0..=100.0))
        });
        property_row(ui, "Accel", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.accel).speed(0.5).range(0.1..=500.0))
        });
        property_row(ui, "Decel", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.decel).speed(0.5).range(0.1..=500.0))
        });
        property_row(ui, "Jump Power", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.jump_power).speed(0.1).range(0.0..=50.0))
        });
        property_row(ui, "Gravity", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.gravity).speed(0.1).range(0.0..=100.0))
        });
        property_row(ui, "Look Sensitivity", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.look_sensitivity).speed(0.005).range(0.0..=5.0))
        });
        property_row(ui, "Eye Height", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.eye_height).speed(0.05).range(0.0..=10.0))
        });
        property_row(ui, "Arm Length", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.arm_length).speed(0.1).range(0.0..=50.0))
        });
        changed
    }
}
impl Gizmo for Pawn {}

impl Inspect for SpawnPoint {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Tag", &mut changed, |ui| {
            ui.text_edit_singleline(&mut self.tag)
                .on_hover_text("Group: \"player\", \"npc\", a team name…")
        });
        property_row(ui, "Index", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.index).range(0..=4096))
        });
        changed
    }
}
impl Gizmo for SpawnPoint {}

impl Inspect for zestengine_core::PersistentState {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Transform", &mut changed, |ui| {
            ui.checkbox(&mut self.transform, "").on_hover_text("A dedicated server saves and restores position, rotation and scale")
        });
        property_row(ui, "Variables", &mut changed, |ui| {
            ui.checkbox(&mut self.variables, "").on_hover_text("A dedicated server saves and restores the Network Variables")
        });
        property_row(ui, "Per player", &mut changed, |ui| {
            ui.checkbox(&mut self.per_player, "").on_hover_text("On a player object: its Network Variables are saved per player id and returned when the player joins again")
        });
        changed
    }
}
impl Gizmo for zestengine_core::PersistentState {}

impl Inspect for Sync {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Grabbable", &mut changed, |ui| {
            ui.checkbox(&mut self.grabbable, "")
                .on_hover_text("Any peer can take ownership by pressing Grab")
        });
        property_row(ui, "Smoothing", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.smoothing).speed(0.5).range(0.0..=60.0))
                .on_hover_text("Remote-update lerp rate (0 = snap)")
        });
        ui.label(
            RichText::new("Replicates this object's transform; the owner sends, others receive.")
                .small()
                .weak(),
        );
        changed
    }
}
impl Gizmo for Sync {}

impl Inspect for AcousticMaterial {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let before = self.preset;
        enum_row(ui, "Preset", "zestengine-acoustic-preset", &mut self.preset, &AcousticPreset::ALL, AcousticPreset::label, &mut changed);
        if self.preset != before
            && let Some((a, sc, t)) = before.values()
            && self.preset == AcousticPreset::Custom
        {
            // Start a custom material from the preset it replaces.
            (self.absorption, self.scattering, self.transmission) = (a, sc, t);
        }
        let custom = self.preset == AcousticPreset::Custom;
        let (mut a, mut sc, mut t) = self.values();
        ui.add_enabled_ui(custom, |ui| {
            for (label, band) in [("Absorption", &mut a), ("Transmission", &mut t)] {
                property_row(ui, label, &mut changed, |ui| {
                    let mut any = false;
                    let mut resp = ui.horizontal(|ui| {
                        for v in band.iter_mut() {
                            any |= ui.add(DragValue::new(v).speed(0.005).range(0.0..=1.0).max_decimals(3)).changed();
                        }
                    }).response;
                    if any {
                        resp.mark_changed();
                    }
                    resp
                });
            }
            property_row(ui, "Scattering", &mut changed, |ui| crate::widgets::slider(ui, egui::Slider::new(&mut sc, 0.0..=1.0)));
        });
        if custom {
            (self.absorption, self.scattering, self.transmission) = (a, sc, t);
        }
        ui.label(RichText::new("Bands: low / mid / high").small().weak());
        property_row(ui, "Moving occluder", &mut changed, |ui| {
            ui.checkbox(&mut self.dynamic, "")
                .on_hover_text("Occlude and reflect even when not Static (follows the object)")
        });
        changed
    }
}
impl Gizmo for AcousticMaterial {}

impl Inspect for AudioProbeVolume {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Size", &mut self.size, &mut changed);
        property_row(ui, "Spacing", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.spacing).speed(0.05).range(0.25..=50.0).suffix(" m"))
        });
        property_row(ui, "Height", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.height).speed(0.05).range(0.0..=20.0).suffix(" m"))
        });
        property_row(ui, "Bake reverb", &mut changed, |ui| ui.checkbox(&mut self.reverb, ""));
        property_row(ui, "Bake pathing", &mut changed, |ui| ui.checkbox(&mut self.pathing, ""));
        ui.label(RichText::new("Probes sit on floor geometry inside the box. Tools > Bake Audio.").small().weak());
        changed
    }
}
impl Gizmo for AudioProbeVolume {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let half = glam::Vec3::from(self.size) * 0.5;
        let mut g = vec![GizmoSpec::Box {
            center: glam::Vec3::ZERO,
            half_extents: half,
            object_anchored: true,
            blend: 0.0,
            color: crate::theme::TYPE_GI,
        }];
        // Preview of the floor grid (the bake snaps probes to real floors).
        let step = self.spacing.max(0.25);
        let (nx, nz) = (((half.x * 2.0) / step).floor() as usize + 1, ((half.z * 2.0) / step).floor() as usize + 1);
        if let Some(positions) = display_points(nx * nz, || {
            let mut p = Vec::with_capacity(nx * nz);
            for ix in 0..nx {
                for iz in 0..nz {
                    p.push(glam::Vec3::new(-half.x + ix as f32 * step, -half.y + self.height, -half.z + iz as f32 * step));
                }
            }
            p
        }) {
            g.push(GizmoSpec::Points { positions, color: crate::theme::with_alpha(crate::theme::TYPE_GI, 0.7) });
        }
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit {
            self.size = (half_extents * 2.0).to_array();
        }
    }
}
