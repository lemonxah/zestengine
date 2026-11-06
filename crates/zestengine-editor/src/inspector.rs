//! Unified Inspector panel: shows the selected object (transform, mesh,
//! material slots + material editor) or the selected file (.material editor,
//! .scene loader, generic file info).

use std::path::PathBuf;

use egui::{RichText, Ui};

use zestengine_core::{Component, ComponentRegistry, MaterialModel, ObjectId, ShaderUiInfo};

use crate::components::{EditorComponents, components_ui};
use crate::sections::material_editor_ui;
use egui_phosphor::regular as ph;

#[derive(Clone, PartialEq)]
pub struct TransformModel {
    pub translation: [f32; 3],
    /// Euler XYZ in degrees.
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
    /// Name of the parent object. Position / Rotation / Scale above are LOCAL,
    /// relative to it (`None` = root object: they are world values).
    pub parent: Option<String>,
    /// World `(position, rotation_deg, scale)` (read-only, for the optional World line).
    pub world: Option<([f32; 3], [f32; 3], [f32; 3])>,
}

/// Which header / transform values differ across a multi-object selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct MixedInfo {
    pub name: bool,
    pub enabled: bool,
    pub static_geometry: bool,
    pub cast_shadows: bool,
    pub translation: [bool; 3],
    pub rotation: [bool; 3],
    pub scale: [bool; 3],
}

pub struct ObjectInfoModel {
    pub mixed: MixedInfo,
    pub name: String,
    /// Whether the object renders / contributes light.
    pub enabled: bool,
    /// Non-moving: included in the lighting bake (lightmaps + occluder).
    pub static_geometry: bool,
    /// Stable key of the object (its id), so per-object UI state such as the open material
    /// sections is remembered per object.
    pub key: String,
    /// "Mesh" / "Empty" / "Camera" / "Primitive".
    pub kind: &'static str,
    pub transform: TransformModel,
    /// (vertices, triangles) for mesh objects.
    pub mesh: Option<(u32, u32)>,
    /// Set on the root object of an imported model: shows the Model section.
    pub model_root: Option<crate::renderer_ui::ModelRootInfo>,
    /// The Mesh Filter view (mesh objects only).
    pub filter: Option<crate::renderer_ui::MeshFilterInfo>,
    /// Slot data for the Mesh Renderer section (objects with a renderer component).
    pub renderer: Option<crate::renderer_ui::RendererInfo>,
    /// Every blendshape of the object's meshes (the inspector's Blendshapes
    /// sub-section; empty = none, no section).
    pub blendshapes: std::sync::Arc<[String]>,
}

pub enum InspectorContent<'a> {
    Empty,
    Object {
        info: &'a mut ObjectInfoModel,
        /// One section per material the object's renderer uses; a section's model is
        /// lent only while it is open.
        sections: Vec<crate::renderer_ui::MaterialSectionModel<'a>>,
        components: &'a mut Vec<Box<dyn Component>>,
        registry: &'a ComponentRegistry,
        /// Editor-side inspector/gizmo dispatch for the components.
        editor_components: &'a EditorComponents,
        /// (id, name) of every scene object, for `ObjectRef` picker dropdowns.
        objects: &'a [(ObjectId, String)],
        /// Layer display names, so layer pickers (camera culling mask) stay
        /// consistent with the Layers window.
        layer_names: &'a [String],
        /// Inline editor for assets assigned to component slots.
        asset_editor: Option<&'a crate::components::AssetEditor<'a>>,
    },
    MaterialFile {
        path: String,
        material: &'a mut MaterialModel,
        shader_info: Option<&'a ShaderUiInfo>,
        dirty: bool,
    },
    SceneFile {
        path: String,
    },
    File {
        path: String,
        size: Option<u64>,
    },
}

/// Result of the pinned object header row.
#[derive(Default)]
pub struct ObjectHeaderResponse {
    pub object_changed: bool,
}

#[derive(Default)]
pub struct InspectorResponse {
    pub object_changed: bool,
    pub material_changed: bool,
    pub reset_material: bool,
    pub save_material: bool,
    pub load_scene: bool,
    /// What the Mesh Renderer and the material sections asked for this frame.
    pub renderer_events: Vec<crate::renderer_ui::RendererEvent>,
    /// An image file was dropped on a texture slot: (slot index 0..12, absolute
    /// path). The engine converts it to project-relative and assigns it.
    pub texture_dropped: Option<(usize, PathBuf)>,
    /// Component picked from the Add Component menu.
    pub add_component: Option<&'static str>,
    /// Component index whose remove button was clicked.
    pub remove_component: Option<usize>,
    /// A component's Network Role was changed from its menu (`None` = type default).
    pub set_component_role: Option<(usize, Option<zestengine_core::NetRole>)>,
    /// The code editor's text changed / its save button was clicked.
    pub code_changed: bool,
    pub save_code: bool,
    /// Run cargo clippy over the plugin crates.
    pub run_check: bool,
}

pub use zestengine_plugins::CodeDiagnostic;

pub struct InspectorPanel {
    search: String,
    /// When set, the inspector keeps showing the locked selection even as the
    /// scene selection changes. The engine reads this to freeze its content.
    pub locked: bool,
}

impl Default for InspectorPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl InspectorPanel {
    pub fn new() -> Self {
        Self {
            search: String::new(),
            locked: false,
        }
    }

    /// Thin bar at the top-right of the inspector with the INSPECTOR lock (keep showing the
    /// current selection while you select other things, as Unity's lock does). Returns true
    /// when the lock was toggled this frame (the engine then snapshots the selection).
    pub fn lock_bar(&mut self, ui: &mut Ui) -> bool {
        use crate::{icons::Icon, theme, widgets};
        let mut changed = false;
        ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), 20.0), egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let tip = if self.locked { "Unlock: follow the selection again" } else { "Lock the Inspector to the current selection" };
            if widgets::icon_button(ui, if self.locked { Icon::Lock } else { Icon::LockOpen }, tip, self.locked).clicked() {
                self.locked = !self.locked;
                changed = true;
            }
            if self.locked {
                ui.label(RichText::new("Locked").size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        });
        changed
    }

    /// Pinned object header, laid out like Unity's: row 1 = type tile, Active checkbox, the
    /// name (with the type line under it) and the Static flag at the right end; row 2 = Layer
    /// and Network Role side by side.
    #[allow(clippy::too_many_arguments)]
    pub fn object_header(
        &mut self,
        ui: &mut Ui,
        kind: crate::theme::ObjectKind,
        type_line: &str,
        name: &mut String,
        enabled: &mut bool,
        static_geometry: &mut bool,
        layer: &mut u8,
        layer_names: &[String],
        net_role: &mut zestengine_core::NetRole,
        net_warnings: &[String],
    ) -> ObjectHeaderResponse {
        use crate::{theme, widgets};
        let mut r = ObjectHeaderResponse::default();
        ui.horizontal(|ui| {
            widgets::type_tile(ui, kind, 32.0);
            if ui.checkbox(enabled, "").on_hover_text("Active: enable / disable the object (skips updates, rendering and lighting)").changed() {
                r.object_changed = true;
            }
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                let w = (ui.available_width() - 74.0).max(60.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::TextEdit::singleline(name)
                                .frame(false)
                                .hint_text("\u{2014}")
                                .font(theme::semibold(theme::SIZE_TITLE))
                                .text_color(theme::TEXT_STRONG)
                                .desired_width(w),
                        )
                        .changed()
                    {
                        r.object_changed = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .checkbox(static_geometry, RichText::new("Static").color(theme::TEXT_SECONDARY))
                            .on_hover_text("Static: the object never moves, so it is included in the lighting bake (lightmapped + occluder)")
                            .changed()
                        {
                            r.object_changed = true;
                        }
                    });
                });
                ui.label(RichText::new(type_line).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            });
        });
        ui.add_space(2.0);
        ui.columns(2, |cols| {
            cols[0].horizontal(|ui| {
                ui.label(RichText::new("Layer").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                let current = layer_names.get(*layer as usize).cloned().unwrap_or_else(|| format!("Layer {layer}"));
                egui::ComboBox::from_id_salt("zestengine-object-layer").width(ui.available_width()).selected_text(current).show_ui(ui, |ui| {
                    for (l, n) in layer_names.iter().enumerate() {
                        if ui.selectable_value(layer, l as u8, n).changed() {
                            r.object_changed = true;
                        }
                    }
                });
            });
            cols[1].horizontal(|ui| {
                ui.label(RichText::new("Network").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                let color = theme::net_role_color(*net_role).unwrap_or(theme::TEXT_SECONDARY);
                egui::ComboBox::from_id_salt("zestengine-object-net-role")
                    .width(ui.available_width())
                    .selected_text(RichText::new(net_role.label()).color(color))
                    .show_ui(ui, |ui| {
                        for role in zestengine_core::NetRole::ALL {
                            if ui.selectable_value(net_role, role, role.label()).changed() {
                                r.object_changed = true;
                            }
                        }
                    })
                    .response
                    .on_hover_text("Network Role. Both: everywhere. Client only: removed from a dedicated server (visuals, audio, UI). Server only: removed from players' clients. Children follow their parent.");
            });
        });
        for w in net_warnings {
            ui.label(RichText::new(format!("{} {w}", ph::WARNING)).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        ui.add_space(2.0);
        r
    }

    pub fn ui(
        &mut self,
        ui: &mut Ui,
        content: InspectorContent<'_>,
        shaders: &[&str],
    ) -> InspectorResponse {
        let mut response = InspectorResponse::default();
        match content {
            InspectorContent::Empty => {
                ui.label(RichText::new("Nothing selected").weak());
                ui.label(
                    RichText::new("Select an object in the viewport or Scene panel,\nor a file in the Files panel.")
                        .small()
                        .weak(),
                );
            }
            InspectorContent::Object {
                info,
                sections,
                components,
                registry,
                editor_components,
                objects,
                layer_names,
                asset_editor,
            } => {
                self.object_ui(
                    ui,
                    info,
                    sections,
                    components,
                    registry,
                    editor_components,
                    objects,
                    layer_names,
                    asset_editor,
                    shaders,
                    &mut response,
                );
            }
            InspectorContent::MaterialFile {
                path,
                material,
                shader_info,
                dirty,
            } => {
                ui.heading(RichText::new(&material.name).strong());
                ui.label(RichText::new(&path).small().weak());
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(dirty, egui::Button::new(format!("{} Save", ph::FLOPPY_DISK)))
                        .clicked()
                    {
                        response.save_material = true;
                    }
                    if dirty {
                        ui.label(RichText::new("unsaved changes").small().weak());
                    }
                });
                ui.separator();
                response.material_changed |= material_editor_ui(
                    ui,
                    &mut self.search,
                    material,
                    shaders,
                    shader_info,
                    &mut response.texture_dropped,
                    &path,
                );
            }
            InspectorContent::SceneFile { path } => {
                ui.heading("Scene");
                ui.label(RichText::new(&path).small().weak());
                ui.separator();
                if ui.button("Load Scene").clicked() {
                    response.load_scene = true;
                }
            }
            InspectorContent::File { path, size } => {
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.clone());
                ui.heading(name);
                ui.label(RichText::new(&path).small().weak());
                if let Some(size) = size {
                    ui.label(format!("Size: {}", human_size(size)));
                }
            }
        }
        response
    }

    #[allow(clippy::too_many_arguments)]
    fn object_ui(
        &mut self,
        ui: &mut Ui,
        info: &mut ObjectInfoModel,
        sections: Vec<crate::renderer_ui::MaterialSectionModel<'_>>,
        components: &mut Vec<Box<dyn Component>>,
        registry: &ComponentRegistry,
        editor_components: &EditorComponents,
        objects: &[(ObjectId, String)],
        layer_names: &[String],
        asset_editor: Option<&crate::components::AssetEditor<'_>>,
        shaders: &[&str],
        response: &mut InspectorResponse,
    ) {
        // The object name / enabled / static row is drawn as the pinned header
        // (see `object_header`), so the body starts at Transform.
        crate::widgets::divider(ui);
        let mut reset = false;
        let parent_name = info.transform.parent.clone();
        let space_tip = space_tooltip(parent_name.as_deref());
        let (open, _) = crate::widgets::section_header(ui, ui.make_persistent_id("zestengine-transform"), None, "Transform", None, |ui| {
            reset = crate::widgets::icon_button(ui, crate::icons::Icon::Reset, "Reset transform (local)", false).clicked();
            if let Some(name) = &parent_name {
                // Muted caption: the space the values below are relative to.
                let caption = egui::RichText::new(format!("in {name}")).size(crate::theme::SIZE_TINY).color(crate::theme::TEXT_MUTED);
                ui.add(egui::Label::new(caption).truncate()).on_hover_text(&space_tip);
            }
        });
        let mixed = info.mixed;
        let t = &mut info.transform;
        if reset {
            t.translation = [0.0; 3];
            t.rotation_deg = [0.0; 3];
            t.scale = [1.0; 3];
            response.object_changed = true;
        }
        if open {
            egui::Frame::new().inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 10 }).show(ui, |ui| {
                response.object_changed |= transform_row(ui, "Position", &mut t.translation, 0.02, 2, mixed.translation, &space_tip);
                response.object_changed |= transform_row(ui, "Rotation", &mut t.rotation_deg, 0.5, 1, mixed.rotation, &space_tip);
                response.object_changed |= scale_row(ui, &mut t.scale, mixed.scale, &space_tip);
                if let Some((wp, wr, ws)) = t.world {
                    world_line(ui, wp, wr, ws);
                }
            });
        }

        if let Some(model) = &info.model_root {
            crate::renderer_ui::model_section(ui, &info.key, model, &mut response.renderer_events);
        }
        // Mesh Filter (the mesh), Mesh Renderer (slots + lighting) and one section per material.
        if let Some(filter) = &info.filter {
            crate::renderer_ui::mesh_filter_section(ui, &info.key, filter, &mut response.renderer_events);
        }
        if let Some(renderer) = &info.renderer {
            let r = crate::renderer_ui::mesh_renderer_section(
                ui,
                &info.key,
                renderer,
                components,
                &mut info.static_geometry,
                info.mixed.cast_shadows,
                &info.blendshapes,
                &mut response.renderer_events,
            );
            response.object_changed |= r.changed;
            response.remove_component = response.remove_component.or(r.remove);
            response.set_component_role = response.set_component_role.or(r.role);
            crate::renderer_ui::material_sections(ui, &info.key, &mut self.search, shaders, sections, &mut response.renderer_events);
        }

        // Components (Unity-style: list + Add Component at the bottom).
        let comp = components_ui(ui, components, registry, editor_components, objects, layer_names, asset_editor);
        response.object_changed |= comp.changed;
        response.add_component = comp.add;
        response.remove_component = response.remove_component.or(comp.remove);
        response.set_component_role = response.set_component_role.or(comp.role);
    }
}

fn transform_row(ui: &mut Ui, label: &str, values: &mut [f32; 3], speed: f64, decimals: usize, mixed: [bool; 3], space_tip: &str) -> bool {
    let start = ui.cursor().min;
    let changed = crate::widgets::property_row_w(ui, label, 64.0, |ui| crate::widgets::vec3_field_mixed(ui, values, decimals, speed, mixed));
    space_hint(ui, start, 64.0, label, space_tip);
    changed.iter().any(|c| *c)
}

/// Hover hint on a row's label: which space the values are in.
fn space_hint(ui: &mut Ui, row_start: egui::Pos2, width: f32, label: &str, space_tip: &str) {
    let rect = egui::Rect::from_min_size(row_start, egui::vec2(width, crate::theme::H_FIELD));
    ui.interact(rect, ui.id().with(("zestengine-space-hint", label)), egui::Sense::hover()).on_hover_text(space_tip);
}

/// Tooltip of the Position / Rotation / Scale rows.
pub fn space_tooltip(parent: Option<&str>) -> String {
    match parent {
        Some(p) => format!("Relative to the parent ({p})"),
        None => "World (no parent)".to_owned(),
    }
}

/// Collapsed read-only world position / rotation / scale line.
fn world_line(ui: &mut Ui, pos: [f32; 3], rot: [f32; 3], scale: [f32; 3]) {
    let id = ui.make_persistent_id("zestengine-transform-world");
    let mut open = ui.data_mut(|d| *d.get_persisted_mut_or_default::<bool>(id));
    let arrow = if open { "v World" } else { "> World" };
    if ui.add(egui::Label::new(RichText::new(arrow).size(crate::theme::SIZE_TINY).color(crate::theme::TEXT_MUTED)).sense(egui::Sense::click())).on_hover_text("World-space values (read-only)").clicked() {
        open = !open;
    }
    ui.data_mut(|d| d.insert_persisted(id, open));
    if open {
        let f = |v: [f32; 3]| format!("{:.2}, {:.2}, {:.2}", v[0] + 0.0, v[1] + 0.0, v[2] + 0.0);
        for (k, v) in [("Pos", pos), ("Rot", rot), ("Scale", scale)] {
            ui.label(RichText::new(format!("{k}  {}", f(v))).size(crate::theme::SIZE_TINY).family(egui::FontFamily::Monospace).color(crate::theme::TEXT_MUTED));
        }
    }
}

/// The Scale row with a link toggle: while linked, changing one axis scales
/// the other two by the same ratio (proportions kept).
fn scale_row(ui: &mut Ui, values: &mut [f32; 3], mixed: [bool; 3], space_tip: &str) -> bool {
    let lock_id = egui::Id::new("zestengine-scale-linked");
    let mut linked = ui.data_mut(|d| *d.get_persisted_mut_or_default::<bool>(lock_id));
    let before = *values;
    let row_start = ui.cursor().min;
    // The lock sits in the label column, right after "Scale", so the axis
    // fields line up with Position and Rotation above.
    let changed = crate::widgets::property_row_w(ui, "Scale", 64.0, |ui| {
        let row = ui.min_rect();
        let text = egui::WidgetText::from("Scale").into_galley(ui, Some(egui::TextWrapMode::Truncate), 64.0, egui::TextStyle::Body);
        let size = egui::Vec2::splat(crate::theme::H_TOGGLE);
        let at = egui::pos2(row.left() + text.size().x + 4.0, row.center().y - size.y * 0.5);
        let icon = if linked { crate::icons::Icon::Lock } else { crate::icons::Icon::LockOpen };
        let tip = if linked { "Scale linked: axes change together" } else { "Link the scale axes" };
        let mut cell = ui.new_child(egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(at, size)));
        if crate::widgets::icon_button(&mut cell, icon, tip, linked).clicked() {
            linked = !linked;
        }
        crate::widgets::vec3_field_mixed(ui, values, 2, 0.01, mixed)
    });
    ui.data_mut(|d| d.insert_persisted(lock_id, linked));
    space_hint(ui, row_start, 32.0, "Scale", space_tip);
    if linked && let Some(axis) = (0..3).find(|&k| changed[k]) {
        let (old, new) = (before[axis], values[axis]);
        for k in (0..3).filter(|&k| k != axis) {
            values[k] = if old.abs() > 1e-6 { before[k] * new / old } else { new };
        }
    }
    changed.iter().any(|c| *c)
}

fn human_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}
