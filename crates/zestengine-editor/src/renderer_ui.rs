//! Inspector sections of a mesh object (see `docs/design/mesh-renderer-split.md`):
//!
//! * **Mesh Filter**: a read-only view of which mesh the object draws (its `ObjectSource`).
//! * **Mesh Renderer** / **Skinned Mesh Renderer**: the slot list (one row per submesh, with a
//!   material picker that also takes drops from Files), the Lighting group and, for skinned
//!   meshes, the skeleton and the Blendshapes sub-section.
//! * one collapsible **material section per material** below the renderer, closed by default,
//!   whose body is only laid out (and whose model is only lent by the engine) while open.
//!
//! Row building is plain functions, tested without a UI.

use egui::{RichText, Ui};
use zestengine_core::{Component, MaterialModel, MaterialSlot, NetRole, ShaderUiInfo, ShadowCasting, renderer_of, renderer_of_mut};

use crate::{icons::Icon, theme, widgets};

/// egui temp data (`bool`), set while the renderer section shows the blendshapes: the component
/// list then skips the `Blend Shapes` component.
pub const HIDE_BLENDSHAPES_KEY: &str = "zestengine-hide-blendshapes-component";

/// What the engine knows about one render slot (one submesh of the object's mesh).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlotInfo {
    /// Submesh label (the model's mesh name, or "Submesh k").
    pub submesh: String,
    /// The model's own material for this submesh (what `Model` inherits).
    pub model_material: String,
    /// Display name of the material the slot draws with right now.
    pub material: String,
}

/// Read-only data of the Mesh Filter section.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshFilterInfo {
    /// Mesh name or primitive shape.
    pub label: String,
    /// Project-relative model file for a model mesh.
    pub model_path: Option<String>,
    pub vertices: u32,
    pub triangles: u32,
}

/// A child mesh object of an imported model's root.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelChild {
    /// Object index in the scene.
    pub object: usize,
    pub name: String,
    /// Display name of the material of each slot.
    pub materials: Vec<String>,
}

/// Data of the Model section shown on the root object of an imported model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelRootInfo {
    /// Project-relative model file.
    pub path: String,
    pub meshes: usize,
    pub materials: usize,
    pub clips: usize,
    /// "None", "Generic" or "Humanoid".
    pub rig: String,
    pub children: Vec<ModelChild>,
}

/// Everything the renderer sections need beyond the component itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RendererInfo {
    pub slots: Vec<SlotInfo>,
    /// The mesh has a skeleton: the skinned-mesh extras (joints, Blendshapes) are shown.
    pub skinned: bool,
    /// Joint names of the model skeleton (read-only list).
    pub joints: Vec<String>,
}

/// One row of the Materials list.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotRow {
    pub index: usize,
    pub label: String,
    /// What the slot holds.
    pub value: MaterialSlot,
    /// Shown dim when the slot is `Model`: the material it inherits.
    pub inherits: String,
    /// Past the mesh's submeshes: kept in the data, not drawn.
    pub beyond: bool,
}

/// The rows of the list: one per submesh (a shorter data list reads as `Model`), then the data
/// entries past the last submesh, greyed.
pub fn slot_rows(info: &RendererInfo, data: &[MaterialSlot]) -> Vec<SlotRow> {
    let mut rows: Vec<SlotRow> = info
        .slots
        .iter()
        .enumerate()
        .map(|(k, s)| SlotRow {
            index: k,
            label: if s.submesh.is_empty() { format!("Submesh {k}") } else { s.submesh.clone() },
            value: data.get(k).cloned().unwrap_or_default(),
            inherits: s.model_material.clone(),
            beyond: false,
        })
        .collect();
    for (k, v) in data.iter().enumerate().skip(info.slots.len()) {
        rows.push(SlotRow { index: k, label: "No submesh".into(), value: v.clone(), inherits: String::new(), beyond: true });
    }
    rows
}

/// What a material section shows about where its material comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialSource {
    /// The model's own material, shared by every instance of the model.
    Model { model: String },
    /// A `.material` asset (project-relative path).
    File(String),
    /// An edited snapshot of a model material.
    Edited,
}

/// One material section: a material used by one or more slots of the object.
pub struct MaterialSectionModel<'a> {
    /// The slots drawing with this material (a material used by two slots shows once).
    pub slots: Vec<usize>,
    pub material_index: usize,
    pub title: String,
    pub shader: String,
    /// Objects drawing with this material (including this one).
    pub shared_by: u32,
    pub source: MaterialSource,
    /// Imported material with no file: read-only until extracted.
    pub embedded: bool,
    /// The material's editable model. Only lent while the section is open.
    pub model: Option<&'a mut MaterialModel>,
    pub shader_info: Option<&'a ShaderUiInfo>,
    /// Import warnings of the material's texture slots.
    pub slot_warnings: Vec<crate::texture_slots::SlotWarning>,
}

/// Title of a section: `hair (slots 0, 3)` for a material used by several slots.
pub fn section_title(name: &str, slots: &[usize], slot_count: usize) -> String {
    match slots {
        [_] if slot_count <= 1 => name.to_owned(),
        [k] => format!("{name} (slot {k})"),
        many => format!("{name} (slots {})", many.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    }
}

/// Group an object's slots by the material they draw with, in slot order: `(material, slots)`.
pub fn group_slots(materials: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
    for (k, &m) in materials.iter().enumerate() {
        match out.iter_mut().find(|(x, _)| *x == m) {
            Some((_, slots)) => slots.push(k),
            None => out.push((m, vec![k])),
        }
    }
    out
}

/// "shared by N objects" hint text.
pub fn shared_hint(source: &MaterialSource, users: u32) -> String {
    match source {
        MaterialSource::Model { model } => format!("From the model {model}; shared by all its instances"),
        MaterialSource::File(path) => {
            let name = path.rsplit('/').next().unwrap_or(path);
            if users <= 1 { format!("Edits the asset {name}") } else { format!("Edits the asset {name}, used by {users} objects") }
        }
        MaterialSource::Edited => "An edited copy of the model material".to_owned(),
    }
}

/// Events the sections raise for the engine.
#[derive(Clone, Debug, PartialEq)]
pub enum RendererEvent {
    /// A slot was given a new value (picker, drop, reset to the model default).
    Assign { slot: usize, value: MaterialSlot },
    /// Make Unique on a slot: copy its material to a new `.material` and assign it.
    MakeUnique { slot: usize },
    /// A material's properties were edited (index into the scene's materials).
    Edited(usize),
    /// Reset a material to its loaded values.
    Reset(usize),
    /// Extract an imported material to a `.material` file.
    Extract(usize),
    /// A texture was dropped on a material's slot: (material, texture slot, path).
    Texture { material: usize, slot: usize, path: std::path::PathBuf },
    /// A texture-slot warning fix was chosen for a material.
    Fix { material: usize, slot: usize, fix: crate::texture_slots::SlotFix },
    /// Show a project file in Files.
    Reveal(String),
    /// Open a file's import settings in the Inspector.
    OpenImportSettings(String),
    /// Select a scene object (a child listed in the Model section).
    SelectObject(usize),
}

fn project_root(ui: &Ui) -> Option<String> {
    ui.data(|d| d.get_temp(egui::Id::new(crate::PROJECT_ROOT_KEY)))
}

// ---------------------------------------------------------------- Model

/// The Model section of an imported model's root: its file, what it brought, import settings
/// and a read-only list of the child mesh objects with their material slots.
pub fn model_section(ui: &mut Ui, key: &str, info: &ModelRootInfo, events: &mut Vec<RendererEvent>) {
    widgets::divider(ui);
    let id = egui::Id::new(("zestengine-model-root", key));
    let (open, _) = widgets::section_header(ui, id, Some((Icon::Model, theme::TYPE_MESH)), "Model", None, |_| {});
    if !open {
        return;
    }
    egui::Frame::new().inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 10 }).show(ui, |ui| {
        widgets::property_row(ui, "Source", |ui| {
            let r = ui.add(egui::Label::new(RichText::new(&info.path).color(theme::TEXT_STRONG)).truncate().sense(egui::Sense::click()));
            if r.on_hover_text("Click: show the model file in Files").clicked() {
                events.push(RendererEvent::Reveal(info.path.clone()));
            }
        });
        let stat = |ui: &mut Ui, label: &str, value: String| {
            widgets::property_row(ui, label, |ui| {
                ui.label(RichText::new(value).color(theme::TEXT_SECONDARY));
            });
        };
        stat(ui, "Meshes", info.meshes.to_string());
        stat(ui, "Materials", info.materials.to_string());
        stat(ui, "Rig", info.rig.clone());
        stat(ui, "Animation clips", info.clips.to_string());
        ui.horizontal(|ui| {
            if ui.button("Open Import Settings").clicked() {
                events.push(RendererEvent::OpenImportSettings(info.path.clone()));
            }
            if ui.button("Select in Project").clicked() {
                events.push(RendererEvent::Reveal(info.path.clone()));
            }
        });
        ui.add_space(4.0);
        ui.label(RichText::new(format!("Children ({})", info.children.len())).size(theme::SIZE_META).color(theme::TEXT_DIM));
        let rows = info.children.len();
        let row_h = 20.0;
        egui::ScrollArea::vertical().max_height(row_h * 8.0).id_salt(("model-children", key)).auto_shrink([false, true]).show_rows(ui, row_h, rows, |ui, range| {
            for child in &info.children[range] {
                ui.horizontal(|ui| {
                    let r = ui.add(egui::Label::new(RichText::new(&child.name).color(theme::TEXT_STRONG)).sense(egui::Sense::click()));
                    if r.on_hover_text("Click: select this mesh object").clicked() {
                        events.push(RendererEvent::SelectObject(child.object));
                    }
                    ui.label(RichText::new(child.materials.join(", ")).size(theme::SIZE_META).color(theme::TEXT_DIM));
                });
            }
        });
    });
}

// ---------------------------------------------------------------- Mesh Filter

/// The Mesh Filter section: a read-only view of the object's mesh source.
pub fn mesh_filter_section(ui: &mut Ui, key: &str, info: &MeshFilterInfo, events: &mut Vec<RendererEvent>) {
    widgets::divider(ui);
    let id = egui::Id::new(("zestengine-mesh-filter", key));
    let (open, _) = widgets::section_header(ui, id, Some((Icon::Model, theme::TYPE_MESH)), "Mesh Filter", None, |ui| {
        ui.label(RichText::new(format!("{} verts \u{b7} {} tris", info.vertices, info.triangles)).size(theme::SIZE_META).color(theme::TEXT_DIM));
    });
    if !open {
        return;
    }
    egui::Frame::new().inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 10 }).show(ui, |ui| {
        widgets::property_row(ui, "Mesh", |ui| {
            let text = match &info.model_path {
                Some(path) => {
                    let file = path.rsplit('/').next().unwrap_or(path);
                    format!("{file} / {}", info.label)
                }
                None => info.label.clone(),
            };
            let r = ui.add(egui::Label::new(RichText::new(text).color(theme::TEXT_STRONG)).truncate().sense(egui::Sense::click()));
            if let Some(path) = &info.model_path
                && r.on_hover_text(format!("{path}\nClick: show the model in Files")).clicked()
            {
                events.push(RendererEvent::Reveal(path.clone()));
            }
        });
    });
}

// ---------------------------------------------------------------- Mesh Renderer

/// Result of the renderer section.
#[derive(Default)]
pub struct RendererResponse {
    /// A setting of the component or the Static flag changed.
    pub changed: bool,
    pub remove: Option<usize>,
    pub role: Option<(usize, Option<NetRole>)>,
}

/// Index of the renderer component, if the object has one.
pub fn renderer_index(components: &[Box<dyn Component>]) -> Option<usize> {
    components.iter().position(|c| renderer_of(c.as_ref()).is_some())
}

/// The Mesh Renderer / Skinned Mesh Renderer section with its slot list and Lighting group.
/// Edits the component in place; slot changes come back as events.
#[allow(clippy::too_many_arguments)]
pub fn mesh_renderer_section(
    ui: &mut Ui,
    key: &str,
    info: &RendererInfo,
    components: &mut Vec<Box<dyn Component>>,
    static_geometry: &mut bool,
    mixed_shadows: bool,
    blendshapes: &[String],
    events: &mut Vec<RendererEvent>,
) -> RendererResponse {
    let mut response = RendererResponse::default();
    let Some(index) = renderer_index(components) else { return response };
    let kind = components[index].type_name();
    let title = if info.skinned { zestengine_core::SKINNED_MESH_RENDERER } else { kind };
    widgets::divider(ui);
    let prefab_mark = crate::prefab_ui::begin_component(ui, kind);
    let id = egui::Id::new(("zestengine-renderer", key));
    let roles: Option<Vec<(Option<NetRole>, NetRole)>> = ui.data(|d| d.get_temp(egui::Id::new(crate::COMPONENT_ROLES_KEY)));
    let playing = ui.data(|d| d.get_temp::<bool>(egui::Id::new(crate::PLAY_MODE_KEY)).unwrap_or(false));
    let mut on = zestengine_core::component_enabled(components[index].as_ref());
    let (open, toggled) = widgets::section_header_toggle(ui, id, Some((Icon::Model, theme::TYPE_MESH)), title, &mut on, playing, |ui| {
        let kebab = widgets::icon_button(ui, Icon::Kebab, "Renderer options", false);
        egui::Popup::menu(&kebab).show(|ui| {
            if let Some((explicit, default)) = roles.as_ref().and_then(|r| r.get(index)).copied() {
                let effective = explicit.unwrap_or(default);
                ui.menu_button(format!("Network Role: {}", effective.label()), |ui| {
                    for role in NetRole::ALL {
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
            if ui.button("Remove component").on_hover_text("The object keeps its mesh but stops drawing").clicked() {
                response.remove = Some(index);
                ui.close();
            }
        });
    });
    if toggled && !playing {
        zestengine_core::set_component_enabled(&mut components[index], on);
        response.changed = true;
    }
    if open {
        let data: Vec<MaterialSlot> = renderer_of(components[index].as_ref()).map(|r| r.materials().to_vec()).unwrap_or_default();
        egui::Frame::new().inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 10 }).show(ui, |ui| {
            ui.add_enabled_ui(on, |ui| {
                slot_list(ui, info, &data, events);
                ui.add_space(4.0);
                lighting_group(ui, key, components, index, static_geometry, mixed_shadows, &mut response);
                if info.skinned {
                    skeleton_group(ui, key, info);
                }
            });
        });
    }
    crate::prefab_ui::end_component(ui, prefab_mark);
    // Blendshapes live in exactly one place: under the renderer. The `Blend Shapes` component
    // is only their storage, so its own section is hidden from the component list.
    if !blendshapes.is_empty() {
        ui.data_mut(|d| d.insert_temp(egui::Id::new(HIDE_BLENDSHAPES_KEY), true));
        if crate::blendshapes_section(ui, blendshapes, components) {
            response.changed = true;
        }
    }
    response
}

/// "Materials" list: size line, then one row per slot.
fn slot_list(ui: &mut Ui, info: &RendererInfo, data: &[MaterialSlot], events: &mut Vec<RendererEvent>) {
    let rows = slot_rows(info, data);
    if rows.is_empty() {
        ui.label(RichText::new("No mesh slots: this object has no mesh to draw.").size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    }
    widgets::property_row(ui, "Materials", |ui| {
        ui.label(RichText::new(format!("{} slot{}", info.slots.len(), if info.slots.len() == 1 { "" } else { "s" })).color(theme::TEXT_MUTED))
            .on_hover_text("One slot per submesh of the mesh (set by the mesh)");
    });
    let extra = rows.iter().filter(|r| r.beyond).count();
    let list = |ui: &mut Ui, events: &mut Vec<RendererEvent>, range: std::ops::Range<usize>| {
        for row in &rows[range] {
            slot_row(ui, row, events);
        }
    };
    // A face or avatar can have dozens of slots: only the rows in view are laid out.
    if rows.len() > 12 {
        let row_h = theme::H_FIELD + ui.spacing().item_spacing.y + 2.0;
        egui::ScrollArea::vertical().max_height(row_h * 10.0).auto_shrink([false, true]).show_rows(ui, row_h, rows.len(), |ui, range| list(ui, events, range));
    } else {
        list(ui, events, 0..rows.len());
    }
    if extra > 0 {
        ui.label(RichText::new(format!("{} entr{} past the last submesh: kept, not drawn", extra, if extra == 1 { "y" } else { "ies" })).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    }
}

fn slot_row(ui: &mut Ui, row: &SlotRow, events: &mut Vec<RendererEvent>) {
    let total = ui.available_width();
    let h = theme::H_FIELD;
    ui.allocate_ui_with_layout(egui::vec2(total, h), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.add_enabled_ui(!row.beyond, |ui| {
            let left = |ui: &mut Ui, width: f32, text: RichText| {
                ui.allocate_ui_with_layout(egui::vec2(width, h), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(text).truncate());
                });
            };
            left(ui, 16.0, RichText::new(row.index.to_string()).color(theme::TEXT_DIM));
            left(ui, 70.0, RichText::new(&row.label).color(theme::TEXT_SECONDARY));
            let mut value = match &row.value {
                MaterialSlot::File(p) => p.clone(),
                _ => String::new(),
            };
            let hint = if matches!(row.value, MaterialSlot::Inline(_)) {
                "edited model material".to_owned()
            } else if row.inherits.is_empty() {
                "model".to_owned()
            } else {
                format!("model: {}", row.inherits)
            };
            let chip_w = (total - 16.0 - 70.0 - 24.0 - 3.0 * 4.0).max(60.0);
            ui.allocate_ui_with_layout(egui::vec2(chip_w, h), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(chip_w);
                if crate::components::asset_chip(ui, &format!("slot-{}", row.index), &mut value, &hint, &["material"]) {
                    let new = if value.trim().is_empty() { MaterialSlot::Model } else { MaterialSlot::File(value.clone()) };
                    events.push(RendererEvent::Assign { slot: row.index, value: new });
                }
            });
            ui.add_enabled_ui(!row.value.is_model(), |ui| {
                if widgets::icon_button(ui, Icon::Reset, "Reset to the model's material", false).clicked() {
                    events.push(RendererEvent::Assign { slot: row.index, value: MaterialSlot::Model });
                }
            });
        });
    });
}

/// Lighting group: Cast Shadows, Contribute GI, Scale In Lightmap.
fn lighting_group(
    ui: &mut Ui,
    key: &str,
    components: &mut [Box<dyn Component>],
    index: usize,
    static_geometry: &mut bool,
    mixed_shadows: bool,
    response: &mut RendererResponse,
) {
    let id = egui::Id::new(("zestengine-renderer-lighting", key));
    let (open, _) = widgets::section_header_with(ui, id, None, "Lighting", None, true, |_| {});
    if !open {
        return;
    }
    let Some(r) = renderer_of_mut(&mut components[index]) else { return };
    let mut cast = r.cast_shadows();
    if cast_shadows_row(ui, &mut cast, mixed_shadows) {
        r.set_cast_shadows(cast);
        response.changed = true;
    }
    widgets::property_row(ui, "Contribute GI", |ui| {
        if ui.checkbox(static_geometry, "").on_hover_text("Non-moving: included in the lighting bake (lightmapped + occluder). Same as the Static flag in the header.").changed() {
            response.changed = true;
        }
    });
    let mut scale = r.lightmap_scale();
    ui.add_enabled_ui(*static_geometry, |ui| {
        widgets::property_row(ui, "Scale In Lightmap", |ui| {
            if ui.add(egui::DragValue::new(&mut scale).speed(0.05).range(0.0..=16.0)).on_hover_text("Per-object lightmap-resolution multiplier (needs Contribute GI)").changed() {
                r.set_lightmap_scale(scale);
                response.changed = true;
            }
        });
    });
}

/// "Cast Shadows" dropdown. `mixed` shows an em dash until a value is picked.
pub(crate) fn cast_shadows_row(ui: &mut Ui, value: &mut ShadowCasting, mixed: bool) -> bool {
    let mut changed = false;
    widgets::property_row(ui, "Cast Shadows", |ui| {
        let text = if mixed { "\u{2014}" } else { value.label() };
        egui::ComboBox::from_id_salt("zestengine-object-cast-shadows")
            .width(ui.available_width())
            .selected_text(text)
            .show_ui(ui, |ui| {
                for mode in ShadowCasting::ALL {
                    // Picking the shown value over a mixed selection still applies it.
                    if ui.selectable_label(!mixed && *value == mode, mode.label()).clicked() {
                        changed |= mixed || *value != mode;
                        *value = mode;
                    }
                }
            })
            .response
            .on_hover_text("On: casts shadows. Off: casts none. Two Sided: casts from both faces. Shadows Only: invisible, but still casts (and occludes in the bake).");
    });
    changed
}

/// Read-only skeleton foldout of a skinned mesh (the model skeleton drives skinning).
fn skeleton_group(ui: &mut Ui, key: &str, info: &RendererInfo) {
    let id = egui::Id::new(("zestengine-renderer-skeleton", key));
    let (open, _) = widgets::section_header_with(ui, id, None, "Skeleton", None, false, |ui| {
        ui.label(RichText::new(format!("{} joints", info.joints.len())).size(theme::SIZE_META).color(theme::TEXT_DIM));
    });
    if !open {
        return;
    }
    widgets::property_row(ui, "Root Bone", |ui| {
        ui.label(RichText::new(info.joints.first().map_or("\u{2014}", String::as_str)).color(theme::TEXT_SECONDARY));
    });
    ui.label(RichText::new("Bones come from the model's skeleton.").size(theme::SIZE_META).color(theme::TEXT_DIM));
    egui::ScrollArea::vertical().max_height(120.0).id_salt(("joints", key)).show(ui, |ui| {
        for j in &info.joints {
            ui.label(RichText::new(j).size(theme::SIZE_META).color(theme::TEXT_MUTED));
        }
    });
}

// ---------------------------------------------------------------- material sections

/// Whether the section for the material first used by `first_slot` is open (the engine lends
/// the material's model only for open sections).
pub fn section_open(ctx: &egui::Context, key: &str, first_slot: usize) -> bool {
    egui::collapsing_header::CollapsingState::load_with_default_open(ctx, section_id(key, first_slot), false).is_open()
}

/// Open or close a material section from code (tests, "reveal this material").
pub fn set_section_open(ctx: &egui::Context, key: &str, first_slot: usize, open: bool) {
    let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(ctx, section_id(key, first_slot), false);
    state.set_open(open);
    state.store(ctx);
}

fn section_id(key: &str, first_slot: usize) -> egui::Id {
    egui::Id::new(("zestengine-matsec", key, first_slot))
}

/// One collapsible section per material below the renderer.
pub fn material_sections(
    ui: &mut Ui,
    key: &str,
    search: &mut String,
    shaders: &[&str],
    sections: Vec<MaterialSectionModel<'_>>,
    events: &mut Vec<RendererEvent>,
) {
    for section in sections {
        widgets::divider(ui);
        let first = section.slots.first().copied().unwrap_or(0);
        let hint = shared_hint(&section.source, section.shared_by);
        let shared = if section.shared_by > 1 { format!("shared by {}", section.shared_by) } else { String::new() };
        let (open, _) = widgets::section_header_with(ui, section_id(key, first), Some((Icon::Model, theme::TYPE_GI)), &section.title, None, false, |ui| {
            let text = if shared.is_empty() { section.shader.clone() } else { format!("{} \u{b7} {shared}", section.shader) };
            ui.label(RichText::new(text).size(theme::SIZE_META).color(theme::TEXT_DIM)).on_hover_text(&hint);
        });
        if !open {
            continue;
        }
        egui::Frame::new().inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 10 }).show(ui, |ui| {
            ui.label(RichText::new(&hint).size(theme::SIZE_META).color(theme::TEXT_DIM));
            ui.horizontal(|ui| {
                if ui.button("Make Unique").on_hover_text("Copy this material to a new .material file and assign the copy to this slot only").clicked() {
                    events.push(RendererEvent::MakeUnique { slot: first });
                }
                if !section.embedded && ui.button("Reset").on_hover_text("Reset this material to its loaded values").clicked() {
                    events.push(RendererEvent::Reset(section.material_index));
                }
                if section.embedded && ui.button(format!("{} Extract", egui_phosphor::regular::DOWNLOAD_SIMPLE)).on_hover_text("Write the imported material to a .material file so it can be edited").clicked() {
                    events.push(RendererEvent::Extract(section.material_index));
                }
            });
            match section.model {
                Some(model) if !section.embedded => {
                    let mut texture_dropped = None;
                    ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::SLOT_WARNINGS_KEY), section.slot_warnings.clone()));
                    if crate::sections::material_editor_ui(ui, search, model, shaders, section.shader_info, &mut texture_dropped, &format!("mat-{key}-{}", section.material_index)) {
                        events.push(RendererEvent::Edited(section.material_index));
                    }
                    if let Some((slot, path)) = texture_dropped {
                        events.push(RendererEvent::Texture { material: section.material_index, slot, path });
                    }
                    if let Some((slot, fix)) = ui.data_mut(|d| d.remove_temp::<(usize, crate::texture_slots::SlotFix)>(egui::Id::new(crate::SLOT_FIX_KEY))) {
                        events.push(RendererEvent::Fix { material: section.material_index, slot, fix });
                    }
                }
                _ if section.embedded => {
                    ui.label(RichText::new("Imported material: read-only until extracted or made unique.").size(theme::SIZE_META).color(theme::TEXT_DIM));
                }
                _ => {
                    ui.label(RichText::new("Loading\u{2026}").color(theme::TEXT_DIM));
                    ui.ctx().request_repaint();
                }
            }
        });
    }
}

/// The project root, re-exported for the engine side tests.
#[allow(dead_code)]
pub(crate) fn root_of(ui: &Ui) -> Option<String> {
    project_root(ui)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(n: usize) -> RendererInfo {
        RendererInfo {
            slots: (0..n).map(|k| SlotInfo { submesh: format!("Part{k}"), model_material: format!("Mat{k}"), material: format!("Mat{k}") }).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_shorter_list_reads_as_model_and_a_longer_one_is_greyed() {
        let rows = slot_rows(&info(3), &[MaterialSlot::File("a.material".into())]);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].value, MaterialSlot::File("a.material".into()));
        assert!(rows[1].value.is_model() && rows[2].value.is_model());
        assert!(rows.iter().all(|r| !r.beyond));
        assert_eq!(rows[1].inherits, "Mat1");

        let long = [MaterialSlot::Model, MaterialSlot::Model, MaterialSlot::File("x".into()), MaterialSlot::File("extra".into())];
        let rows = slot_rows(&info(2), &long);
        assert_eq!(rows.len(), 4);
        assert!(!rows[1].beyond && rows[2].beyond && rows[3].beyond);
        assert_eq!(rows[3].value, MaterialSlot::File("extra".into()), "the extras are kept");
    }

    #[test]
    fn slots_sharing_a_material_share_one_section() {
        assert_eq!(group_slots(&[4, 7, 4, 9]), vec![(4, vec![0, 2]), (7, vec![1]), (9, vec![3])]);
        assert_eq!(section_title("hair", &[0, 3], 4), "hair (slots 0, 3)");
        assert_eq!(section_title("skin", &[1], 4), "skin (slot 1)");
        assert_eq!(section_title("skin", &[0], 1), "skin");
    }

    #[test]
    fn the_shared_hint_says_who_is_affected() {
        let file = MaterialSource::File("materials/wood.material".into());
        assert_eq!(shared_hint(&file, 4), "Edits the asset wood.material, used by 4 objects");
        assert_eq!(shared_hint(&file, 1), "Edits the asset wood.material");
        assert!(shared_hint(&MaterialSource::Model { model: "hero.glb".into() }, 12).contains("all its instances"));
    }

    #[test]
    fn open_state_is_per_object_and_slot() {
        let ctx = egui::Context::default();
        assert!(!section_open(&ctx, "obj-a", 0), "closed by default");
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(&ctx, section_id("obj-a", 1), false);
        state.set_open(true);
        state.store(&ctx);
        assert!(section_open(&ctx, "obj-a", 1));
        assert!(!section_open(&ctx, "obj-a", 0));
        assert!(!section_open(&ctx, "obj-b", 1), "another object keeps its own");
    }
}
