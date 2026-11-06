//! The model inspector's import settings: the Scene, Meshes, Geometry and
//! Materials sections (the Rig section lives in `rig_editor.rs`).
//!
//! Everything here edits the inspector's copy of the model's [`ModelImport`], so
//! it follows the usual Apply / Revert session: nothing is saved or reimported
//! until Apply. The buttons that do work on files (Extract Textures, Extract
//! Materials, Search and Remap) are reported through [`ModelOut`]; the editor
//! runs them on a worker and writes the filled tables back into the same copy.
//!
//! The Remapped Materials table lists the material names the last import saw
//! (`ModelImport::source_materials`) and the `.material` each is remapped to; an
//! empty slot means the imported material is used.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};

use zestengine_assets::{IndexFormat, MaterialNaming, MaterialSearch, MeshCompression, ModelImport, NormalWeighting, NormalsMode, TEXTURE_REMAP_SLOTS, TangentMode, UV_ALL, texture_remap_key};
use zestengine_assets::material_remap::{list_subfolders, project_relative, resolve_extract_dir};
use crate::icons::Icon;
use crate::{PROJECT_ROOT_KEY, asset_chip, theme, widgets};

/// Work on files the inspector hands to a worker; the result fills the remap tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemapJob {
    ExtractTextures,
    ExtractMaterials,
    Search,
}

/// What the section asks the inspector to do this frame.
#[derive(Default)]
pub struct ModelOut {
    /// A setting changed (refresh the dirty flag).
    pub changed: bool,
    /// Write the embedded textures out and fill the texture table.
    pub extract_textures: bool,
    /// Write the materials out and fill the material table.
    pub extract_materials: bool,
    /// Fill the material table from `.material` files found in the project.
    pub search_remap: bool,
    /// The material names the model has, when the settings did not list them yet
    /// (adopt them without making the settings dirty).
    pub fill_sources: Option<Vec<String>>,
    /// Reimport the model now (the Import Hooks notice's button): the import
    /// hooks changed since the cached artifact was built.
    pub reimport: bool,
    /// Rebuild the cluster geometry cache now (the Geometry (clusters) section's
    /// button); the editor runs it on a task worker (`GeoJob::run`).
    pub geo_rebuild: Option<std::sync::Arc<crate::geometry_cache_ui::GeoJob>>,
}

/// What a parse of the model reported, for the info line and the tables.
struct Stats {
    meshes: usize,
    vertices: usize,
    max_vertices: usize,
    triangles: usize,
    materials: Vec<String>,
    /// `<material>/<slot>` of every embedded texture.
    embedded: Vec<String>,
    cameras: usize,
    lights: usize,
    clips: usize,
    blendshapes: usize,
    /// The file's takes, and per imported clip `(name, loop mismatch degrees)`.
    takes: Vec<zestengine_assets::TakeInfo>,
    clip_rows: Vec<(String, f32, usize)>,
    /// Import warnings, the constraints and user properties read, and the
    /// Optimize Game Objects record.
    warnings: Vec<String>,
    constraints: usize,
    #[allow(dead_code)] // user property count surfaced by the import UI
    custom_props: usize,
    optimize: zestengine_assets::OptimizeInfo,
    /// The first skeleton and the clips (with their stored sizes), for the clip preview.
    rig: Option<std::sync::Arc<RigPreview>>,
}

/// What the clip preview poses: the first skeleton and the clips.
pub struct RigPreview {
    pub skeleton: zestengine_assets::Skeleton,
    pub clips: Vec<zestengine_assets::AnimationClip>,
}

struct Loaded {
    path: PathBuf,
    stats: Result<Stats, String>,
}

/// Editor state of the model import sections.
#[derive(Default)]
pub struct ModelImportUi {
    loaded: Option<Loaded>,
    rx: Option<(PathBuf, Receiver<Result<Stats, String>>)>,
    /// Import hook signatures when the model was first read: `(path, cached
    /// artifact's, current)`. They differ when plugins changed the hook set
    /// since the artifact was built.
    hooks_seen: Option<(PathBuf, Option<String>, String)>,
    /// Why Search and Remap picked each row's `.material`: model -> source name ->
    /// (the `.material` it picked, the reason). Session only; a row whose
    /// remap was edited since no longer shows its reason.
    reasons: HashMap<PathBuf, HashMap<String, (String, String)>>,
    /// The Geometry (clusters) section.
    pub geo: crate::geometry_cache_ui::GeoCacheUi,
}

impl ModelImportUi {
    /// Forget the parsed stats (after a reimport) so the next frame reads them again.
    pub fn forget(&mut self) {
        self.loaded = None;
        self.hooks_seen = None;
        self.geo.forget();
    }

    /// Remember the reasons of a Search and Remap run (`(source, picked path, reason)`).
    pub fn set_match_reasons(&mut self, path: &Path, hits: Vec<(String, String, String)>) {
        let map = self.reasons.entry(path.to_path_buf()).or_default();
        for (source, picked, reason) in hits {
            map.insert(source, (picked, reason));
        }
    }

    fn ensure_loaded(&mut self, path: &Path) {
        if let Some((p, rx)) = &self.rx {
            match rx.try_recv() {
                Ok(stats) => {
                    self.loaded = Some(Loaded { path: p.clone(), stats });
                    self.rx = None;
                }
                Err(TryRecvError::Disconnected) => self.rx = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        let have = self.loaded.as_ref().is_some_and(|l| l.path == path);
        let pending = self.rx.as_ref().is_some_and(|(p, _)| p == path);
        if have || pending {
            return;
        }
        // Before the read below rebuilds a stale artifact: what was it built with?
        self.hooks_seen = Some((path.to_path_buf(), zestengine_assets::import_cache::stored_hook_signature(path), zestengine_assets::import_hooks::signature_for(path)));
        let (tx, rx) = std::sync::mpsc::channel();
        let owned = path.to_path_buf();
        std::thread::spawn(move || {
            let res = zestengine_assets::load_model_cached(&owned, false).map(|(s, _)| stats_of(&s)).map_err(|e| format!("{e:#}"));
            let _ = tx.send(res);
        });
        self.rx = Some((path.to_path_buf(), rx));
    }

    fn stats(&self, path: &Path) -> Option<&Stats> {
        self.loaded.as_ref().filter(|l| l.path == path).and_then(|l| l.stats.as_ref().ok())
    }

    /// The Scene, Meshes, Geometry and Materials sections.
    pub fn show(&mut self, ui: &mut egui::Ui, path: &Path, m: &mut ModelImport) -> ModelOut {
        self.ensure_loaded(path);
        let mut out = ModelOut::default();
        let is_fbx = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("fbx"));

        widgets::eyebrow(ui, "Scene");
        widgets::property_row_w(ui, "Scale Factor", LABEL_W, |ui| {
            let scale_edit = ui
                .add(egui::DragValue::new(&mut m.scale).speed(0.001).range(-1000.0..=1000.0))
                .on_hover_text("Uniform import scale, applied to geometry, node and bone transforms and animation (0.01 for centimetre sources). The rig's reference pose follows it.")
                .changed();
            if scale_edit {
                m.sync_rig_pose_scale();
            }
            out.changed |= scale_edit;
        });
        if is_fbx {
            toggle(ui, &mut out, "Bake Axis Conversion", &mut m.bake_axis_conversion, "Bake the file's axis conversion into geometry and animation instead of leaving it as a rotation on the root");
        }
        toggle(ui, &mut out, "Import BlendShapes", &mut m.import_blend_shapes, "Import morph targets");
        if m.import_blend_shapes {
            self.blend_shape_options(ui, m, &mut out);
        }
        toggle(ui, &mut out, "Import Cameras", &mut m.import_cameras, "Place the file's cameras as Camera objects when the model is added to a scene");
        toggle(ui, &mut out, "Import Lights", &mut m.import_lights, "Place the file's lights as Light objects when the model is added to a scene");
        if is_fbx {
            toggle(ui, &mut out, "Import Visibility", &mut m.import_visibility, "Place the nodes the file hides as disabled objects");
            toggle(
                ui,
                &mut out,
                "Import Constraints",
                &mut m.import_constraints,
                "Read the file's parent, position, rotation, scale and aim constraints and put a Constraint component on each part they constrain (sources are found among the parts by name). Constraints on bones are not components: use Bake Animations to put their effect into the clips. IK chains are not supported.",
            );
            if let Some(s) = self.stats(path).filter(|_| m.import_constraints) {
                widgets::info_note(ui, &format!("{} constraint(s) read from the file.", s.constraints));
            }
        }

        ui.add_space(6.0);
        widgets::eyebrow(ui, "Rig");
        self.skin_and_hierarchy(ui, path, m, &mut out);

        ui.add_space(6.0);
        widgets::eyebrow(ui, "Meshes");
        toggle(ui, &mut out, "Optimize Mesh", &mut m.optimize_mesh, "Reorder triangles and vertices for the GPU's vertex cache");
        widgets::property_row_w(ui, "Index Format", LABEL_W, |ui| {
            out.changed |= widgets::segmented(ui, &mut m.index_format, &[(IndexFormat::Auto, "Auto"), (IndexFormat::UInt16, "16 bit"), (IndexFormat::UInt32, "32 bit")])
                .on_hover_text("Auto and 32 bit keep each mesh whole; the renderer uses 16 bit indices for any mesh that fits (when ray tracing is off). 16 bit splits a mesh with more than 65535 vertices into pieces that fit.")
                .changed();
        });
        if let Some(s) = self.stats(path) {
            let bits = m.index_format.resolve(s.max_vertices);
            if m.index_format == IndexFormat::UInt16 && bits == 32 {
                widgets::info_note(ui, "A mesh has more than 65535 vertices: it is split into pieces that fit 16 bit indices (extra material slots on the same object).");
            } else if m.index_format == IndexFormat::UInt32 && bits == 32 && s.max_vertices > 65535 {
                widgets::info_note(ui, "A mesh has more than 65535 vertices and keeps 32 bit indices.");
            }
        }

        ui.add_space(6.0);
        widgets::eyebrow(ui, "Geometry");
        toggle(ui, &mut out, "Keep Quads", &mut m.keep_quads, "Keep quads through normals, tangents and welding and record them in the import artifact (a quad table). They are still drawn as triangle pairs.");
        toggle(ui, &mut out, "Weld Vertices", &mut m.weld_vertices, "Merge vertices that share a position and every attribute. Hard edges and UV seams stay split.");
        widgets::property_row_w(ui, "Normals", LABEL_W, |ui| {
            let mut mode = m.normals_mode();
            let r = widgets::segmented(ui, &mut mode, &[(NormalsMode::Import, "Import"), (NormalsMode::Calculate, "Calculate"), (NormalsMode::None, "None")])
                .on_hover_text("Import uses the file's normals and calculates missing ones. Calculate always recomputes them. None writes a constant +Z normal.");
            if r.changed() {
                m.normals = mode;
                m.recalculate_normals = false;
                out.changed = true;
            }
        });
        if m.normals_mode() == NormalsMode::Calculate {
            widgets::property_row_w(ui, "Smoothing Angle", LABEL_W, |ui| {
                out.changed |= widgets::slider(ui, egui::Slider::new(&mut m.smoothing_angle, 0.0..=180.0).suffix("\u{b0}")).changed();
            });
            widgets::property_row_w(ui, "Weighting", LABEL_W, |ui| {
                out.changed |= widgets::segmented(
                    ui,
                    &mut m.normal_weighting,
                    &[(NormalWeighting::Angle, "Angle"), (NormalWeighting::Area, "Area"), (NormalWeighting::AreaAndAngle, "Both"), (NormalWeighting::Unweighted, "Equal")],
                )
                .on_hover_text("How much each face around a vertex counts: by the angle it makes there, by its area, both, or equally")
                .changed();
            });
        }
        widgets::property_row_w(ui, "Tangents", LABEL_W, |ui| {
            out.changed |= widgets::segmented(ui, &mut m.tangents, &[(TangentMode::Import, "Import"), (TangentMode::Calculate, "Calculate"), (TangentMode::Legacy, "Legacy"), (TangentMode::None, "None")])
                .on_hover_text("Import uses the file's tangents and calculates them (MikkTSpace) when it has none. Calculate always does. Legacy is the older per-vertex method.")
                .changed();
        });
        toggle(ui, &mut out, "Lightmap UVs", &mut m.generate_lightmap_uvs, "Unwrap a second UV set for meshes that have none (meshes with blendshapes are skipped)");
        widgets::property_row_w(ui, "UV Channels", LABEL_W, |ui| {
            let mut uv0 = m.uv_mask & 1 != 0;
            let mut uv1 = m.uv_mask & 2 != 0;
            let a = ui.checkbox(&mut uv0, "UV0").on_hover_text("Keep the first UV set (off zeroes it)").changed();
            let b = ui.checkbox(&mut uv1, "UV1").on_hover_text("Keep the second (lightmap) UV set (off falls back to a copy of UV0)").changed();
            if a || b {
                m.uv_mask = (u8::from(uv0) | u8::from(uv1) << 1) & UV_ALL;
                out.changed = true;
            }
        });
        widgets::property_row_w(ui, "Mesh Compression", LABEL_W, |ui| {
            out.changed |= widgets::segmented(
                ui,
                &mut m.mesh_compression,
                &[(MeshCompression::Off, "Off"), (MeshCompression::Low, "Low"), (MeshCompression::Medium, "Medium"), (MeshCompression::High, "High")],
            )
            .on_hover_text("Snap positions, UVs, normals and tangents to a coarser grid (lossy). Vertices that land on the same values weld together.")
            .changed();
        });
        toggle(ui, &mut out, "Generate Colliders", &mut m.generate_colliders, "Give every part of the model a Mesh Collider when it is added to a scene");
        toggle(ui, &mut out, "Read/Write", &mut m.read_write, "Keep the meshes readable at runtime");
        toggle(ui, &mut out, "Flip UV (V)", &mut m.flip_uv, "Flip the V texture coordinate (DirectX vs OpenGL convention)");
        if let Some(s) = self.stats(path) {
            let bits = m.index_format.resolve(s.max_vertices);
            let mut line = format!("{} mesh{} \u{b7} {} vertices \u{b7} {} triangles \u{b7} {bits}-bit indices", s.meshes, if s.meshes == 1 { "" } else { "es" }, s.vertices, s.triangles);
            if s.blendshapes > 0 {
                line += &format!(" \u{b7} {} blendshapes", s.blendshapes);
            }
            if s.clips > 0 {
                line += &format!(" \u{b7} {} clips", s.clips);
            }
            ui.label(egui::RichText::new(line).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            if s.cameras + s.lights > 0 {
                ui.label(egui::RichText::new(format!("{} camera(s), {} light(s) imported", s.cameras, s.lights)).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            }
        }

        if is_fbx {
            ui.add_space(6.0);
            self.animation(ui, path, m, &mut out);
        }
        ui.add_space(6.0);
        self.materials(ui, path, m, &mut out);
        ui.add_space(6.0);
        out.geo_rebuild = self.geo.section(ui, path);
        ui.add_space(6.0);
        self.import_hooks(ui, path, &mut out);
        out
    }

    /// The Import Hooks section (collapsed): the plugin hooks that run on every
    /// import, read-only, and a notice when the cached artifact was built with
    /// another set.
    fn import_hooks(&mut self, ui: &mut egui::Ui, path: &Path, out: &mut ModelOut) {
        let hooks = zestengine_assets::import_hooks::active_hooks();
        let stale = self.hooks_seen.as_ref().filter(|(p, stored, now)| p == path && stored.as_ref().is_some_and(|s| s != now));
        let title = if hooks.is_empty() { "Import Hooks".to_owned() } else { format!("Import Hooks ({})", hooks.len()) };
        let (open, _) = widgets::section_header_with(ui, ui.make_persistent_id("model-import-hooks"), None, &title, None, false, |ui| {
            if stale.is_some() {
                widgets::warn_badge(ui, "Changed");
            }
        });
        if !open {
            return;
        }
        if let Some((_, stored, now)) = stale {
            widgets::info_note(
                ui,
                &format!(
                    "The cached import was built with a different set of import hooks ({} -> {}). It was rebuilt when opened; reimport to refresh instances already in the scene.",
                    stored.as_deref().filter(|s| !s.is_empty()).unwrap_or("none"),
                    if now.is_empty() { "none" } else { now },
                ),
            );
            if widgets::secondary_button(ui, None, "Reimport").on_hover_text("Rebuild this model with the current import hooks and update the open scene in place").clicked() {
                out.reimport = true;
            }
        }
        if hooks.is_empty() {
            ui.label(egui::RichText::new("No import hooks are registered. Plugins add them with register_import_hook; they run in the editor only.").size(theme::SIZE_META).color(theme::TEXT_MUTED));
            return;
        }
        for h in &hooks {
            widgets::property_row_w(ui, h.name.as_str(), 160.0, |ui| {
                ui.label(egui::RichText::new(format!("v{} \u{b7} order {} \u{b7} {}", h.version, h.order, h.owner)).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            });
        }
        let note = if zestengine_assets::import_hooks::hooks_apply(path) { "Hooks run when this model is imported, in this order, before the result is cached." } else { "Hooks do not run for this file (outside the editor's trusted project)." };
        ui.label(egui::RichText::new(note).size(theme::SIZE_META).color(theme::TEXT_MUTED));
        ui.label(egui::RichText::new("Changing a hook or its version reimports every model. Reimport after editing a hook's code.").size(theme::SIZE_META).color(theme::TEXT_MUTED));
    }

    fn materials(&mut self, ui: &mut egui::Ui, path: &Path, m: &mut ModelImport, out: &mut ModelOut) {
        widgets::eyebrow(ui, "Materials");
        widgets::property_row_w(ui, "Creation Mode", LABEL_W, |ui| {
            out.changed |= widgets::segmented(ui, &mut m.import_materials, &[(false, "None"), (true, "Import")])
                .on_hover_text("None uses the default material for every mesh")
                .changed();
        });
        if !m.import_materials {
            widgets::info_note(ui, "Meshes use the default material. Remapped materials below still apply.");
        }

        // The rows: the names the model has (the settings' list, else what the parse saw).
        let parsed: Vec<String> = self.stats(path).map(|s| s.materials.clone()).unwrap_or_default();
        if m.source_materials.is_empty() && !parsed.is_empty() {
            out.fill_sources = Some(parsed.clone());
        }
        let names: Vec<String> = if m.source_materials.is_empty() { parsed } else { m.source_materials.clone() };

        ui.add_space(2.0);
        ui.label(egui::RichText::new("Remapped Materials").font(theme::semibold(theme::SIZE_BODY)));
        if names.is_empty() {
            let msg = if self.loaded.as_ref().is_some_and(|l| l.path == path) { "This model has no materials." } else { "Reading the model's materials\u{2026}" };
            ui.label(egui::RichText::new(msg).size(theme::SIZE_META).color(theme::TEXT_MUTED));
        }
        for name in &names {
            let mut value = m.material_remap.get(name).cloned().unwrap_or_default();
            widgets::property_row_w(ui, name.as_str(), 120.0, |ui| {
                if asset_chip(ui, &format!("model-remap-{name}"), &mut value, "Embedded", &["material"]) {
                    if value.trim().is_empty() {
                        m.material_remap.remove(name);
                    } else {
                        m.material_remap.insert(name.clone(), value.clone());
                    }
                    out.changed = true;
                }
            });
            // Why Search and Remap chose it (only while the row still holds that pick).
            if let Some((_, reason)) = self.reasons.get(path).and_then(|r| r.get(name)).filter(|(picked, _)| m.material_remap.get(name) == Some(picked)) {
                ui.label(egui::RichText::new(format!("Matched: {reason}")).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            }
        }
        // Entries for materials the file no longer has: shown so they can be cleared.
        let stale: Vec<String> = m.material_remap.keys().filter(|k| !names.contains(k)).cloned().collect();
        for name in stale {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{name} (not in the file)")).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
                if widgets::ghost_button(ui, "Clear", theme::H_BUTTON_SMALL).clicked() {
                    m.material_remap.remove(&name);
                    out.changed = true;
                }
            });
        }

        let all_remapped = !names.is_empty() && names.iter().all(|n| m.material_remap.contains_key(n));
        let embedded: Vec<String> = self.stats(path).map(|s| s.embedded.clone()).unwrap_or_default();
        let textures_left = embedded.iter().any(|k| !m.texture_remap.contains_key(k));
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.add_enabled_ui(textures_left, |ui| {
                if widgets::secondary_button(ui, None, "Extract Textures")
                    .on_hover_text("Write the embedded textures as PNG files into the folder chosen below (default: a Textures folder next to the model), set their import kind from the slot they fill (normal maps become Normal Map textures) and remap them. Textures already remapped are skipped.")
                    .on_disabled_hover_text("Every embedded texture is already remapped (or the model has none)")
                    .clicked()
                {
                    out.extract_textures = true;
                }
            });
            ui.add_enabled_ui(!names.is_empty() && !all_remapped, |ui| {
                if widgets::secondary_button(ui, None, "Extract Materials")
                    .on_hover_text("Write each material that is not remapped as a .material file into the folder chosen below (default: a Materials folder next to the model, with its textures) and remap it")
                    .on_disabled_hover_text("Every material is already remapped")
                    .clicked()
                {
                    out.extract_materials = true;
                }
            });
        });
        out.changed |= folder_row(ui, "extract-tex-dir", "Textures to", &mut m.extract_textures_dir, path, "Textures");
        out.changed |= folder_row(ui, "extract-mat-dir", "Materials to", &mut m.extract_materials_dir, path, "Materials");
        if !m.texture_remap.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{} texture(s) remapped to project files", m.texture_remap.len())).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                if widgets::ghost_button(ui, "Clear", theme::H_BUTTON_SMALL).on_hover_text("Use the embedded textures again").clicked() {
                    m.texture_remap.clear();
                    out.changed = true;
                }
            });
        }

        ui.add_space(6.0);
        widgets::eyebrow(ui, "On Demand Remap");
        widgets::property_row_w(ui, "Naming", LABEL_W, |ui| {
            out.changed |= widgets::segmented(
                ui,
                &mut m.material_naming,
                &[(MaterialNaming::BasedOnMaterialName, "Material"), (MaterialNaming::BasedOnModelNameAndMaterialName, "Model + Material"), (MaterialNaming::BasedOnTextureName, "Texture")],
            )
            .on_hover_text("Material: <material>.material. Model + Material: <model>-<material>.material. Texture: a .material whose textures have the same names (or the same base name / a shared prefix) as the ones the material uses; the closest match wins, ties go to the nearer folder, then the smaller path.")
            .changed();
        });
        widgets::property_row_w(ui, "Search", LABEL_W, |ui| {
            out.changed |= widgets::segmented(
                ui,
                &mut m.material_search,
                &[(MaterialSearch::Local, "Local"), (MaterialSearch::RecursiveUp, "Up"), (MaterialSearch::Everywhere, "Everywhere")],
            )
            .changed();
        });
        ui.add_enabled_ui(!names.is_empty(), |ui| {
            if widgets::secondary_button(ui, None, "Search and Remap")
                .on_hover_text("Look for a .material named after each material (by the Naming rule, within the Search scope) and remap it")
                .clicked()
            {
                out.search_remap = true;
            }
        });
        let _ = (texture_remap_key("", ""), TEXTURE_REMAP_SLOTS);
    }
}

/// What a folder setting shows: its project-relative text and whether it is bad.
#[derive(Debug, PartialEq, Eq)]
struct FolderLabel {
    text: String,
    is_default: bool,
    error: Option<String>,
}

/// The text for a folder setting (`""` = the default beside the model).
fn folder_label(root: &Path, model: &Path, setting: &str, default_name: &str) -> FolderLabel {
    match resolve_extract_dir(root, model, setting, default_name) {
        Ok(dir) => FolderLabel { text: project_relative(root, &dir), is_default: setting.trim().is_empty(), error: None },
        Err(e) => FolderLabel { text: setting.to_owned(), is_default: false, error: Some(format!("{e:#}")) },
    }
}

/// `a/b/c` -> `a/b`; the root is `""`.
fn parent_folder(rel: &str) -> String {
    rel.rsplit_once('/').map(|(p, _)| p.to_owned()).unwrap_or_default()
}

fn child_folder(rel: &str, name: &str) -> String {
    if rel.is_empty() { name.to_owned() } else { format!("{rel}/{name}") }
}

/// `rel` or its nearest parent that exists as a folder, where the chooser opens.
fn nearest_existing_folder(root: &Path, rel: &str) -> String {
    let mut cur = rel.to_owned();
    while !cur.is_empty() && !root.join(&cur).is_dir() {
        cur = parent_folder(&cur);
    }
    cur
}

/// A "where do extracted files go" row: the folder as a project-relative path,
/// a chooser popup over the project's folders, and a Reset to the default.
/// `value` is the stored setting (`""` = `<model dir>/<default_name>`).
fn folder_row(ui: &mut egui::Ui, salt: &str, label: &str, value: &mut String, model: &Path, default_name: &str) -> bool {
    let Some(root) = ui.data(|d| d.get_temp::<String>(egui::Id::new(PROJECT_ROOT_KEY))).map(PathBuf::from) else { return false };
    let mut changed = false;
    let shown = folder_label(&root, model, value, default_name);
    let default_rel = folder_label(&root, model, "", default_name).text;
    widgets::property_row_w(ui, label, 120.0, |ui| {
        let resp = widgets::secondary_button(ui, Some(Icon::Folder), &shown.text).on_hover_text("Choose a folder inside the project");
        let cur_id = ui.make_persistent_id((salt, "browsing"));
        let popup = egui::Popup::menu(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            widgets::menu_frame(ui);
            let mut cur: String = ui.data(|d| d.get_temp(cur_id)).unwrap_or_else(|| nearest_existing_folder(&root, if value.is_empty() { &default_rel } else { value }));
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!cur.is_empty(), |ui| {
                    if widgets::icon_button(ui, Icon::ArrowUp, "Parent folder", false).clicked() {
                        cur = parent_folder(&cur);
                    }
                });
                ui.label(egui::RichText::new(if cur.is_empty() { "Project".to_owned() } else { cur.clone() }).font(theme::semibold(theme::SIZE_BODY)));
            });
            let subs = list_subfolders(&root, &cur);
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                if subs.is_empty() {
                    ui.label(egui::RichText::new("No sub-folders").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                }
                for name in &subs {
                    if ui.selectable_label(false, format!("{}  {name}", crate::egui_phosphor::regular::FOLDER)).clicked() {
                        cur = child_folder(&cur, name);
                    }
                }
            });
            if widgets::primary_button(ui, None, "Use this folder").clicked() {
                *value = if cur == default_rel { String::new() } else { cur.clone() };
                changed = true;
                ui.close();
            }
            ui.data_mut(|d| d.insert_temp(cur_id, cur));
        });
        if popup.is_none() {
            ui.data_mut(|d| d.remove::<String>(cur_id));
        }
        ui.add_enabled_ui(!value.is_empty(), |ui| {
            if widgets::icon_button(ui, Icon::Reset, "Reset to the folder next to the model", false).clicked() {
                value.clear();
                changed = true;
            }
        });
    });
    if let Some(e) = &shown.error {
        ui.label(egui::RichText::new(format!("{e}; extracting will fail until it is reset")).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    } else if shown.is_default {
        ui.label(egui::RichText::new("Default: next to the model").size(theme::SIZE_META).color(theme::TEXT_MUTED));
    }
    changed
}

/// A labelled checkbox row.
impl ModelImportUi {
    /// Max Bones per Vertex, Optimize Game Objects and Extra Expose Transforms.
    fn skin_and_hierarchy(&self, ui: &mut egui::Ui, path: &Path, m: &mut ModelImport, out: &mut ModelOut) {
        widgets::property_row_w(ui, "Max Bones / Vertex", LABEL_W, |ui| {
            let mut n = m.max_bones_per_vertex.clamp(1, 4);
            let r = widgets::segmented(ui, &mut n, &[(1, "1"), (2, "2"), (3, "3"), (4, "4")])
                .on_hover_text("The most bones that may influence one vertex (the vertex format holds 4). Vertices over the limit keep their strongest influences, renormalised; the import reports how many were cut.");
            if r.changed() {
                m.max_bones_per_vertex = n;
                out.changed = true;
            }
        });
        toggle(
            ui,
            out,
            "Optimize Game Objects",
            &mut m.optimize_game_objects,
            "Keep the bone hierarchy inside the skeleton asset instead of as objects. Model parts are placed flat under the model's root and bones are never objects here, so this records the choice and which transforms stay exposed; it changes nothing at placement.",
        );
        if m.optimize_game_objects {
            let mut remove = None;
            for (i, name) in m.extra_expose_transforms.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    out.changed |= ui.add(egui::TextEdit::singleline(name).hint_text("bone or node name").desired_width(160.0)).changed();
                    if ui.small_button("x").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                m.extra_expose_transforms.remove(i);
                out.changed = true;
            }
            if ui.small_button("Add Extra Expose Transform").on_hover_text("A bone or node that stays addressable (attachment points, hand targets)").clicked() {
                m.extra_expose_transforms.push(String::new());
                out.changed = true;
            }
            if let Some(s) = self.stats(path) {
                let o = &s.optimize;
                widgets::info_note(ui, &format!("{} joint(s) stay inside the skeleton asset; {} exposed name(s) match the model.", o.internal_joints, o.exposed.len()));
            }
        }
        if let Some(s) = self.stats(path) {
            for w in &s.warnings {
                ui.label(egui::RichText::new(format!("\u{26a0} {w}")).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
            }
        }
    }

    /// The options under Import BlendShapes.
    fn blend_shape_options(&self, ui: &mut egui::Ui, m: &mut ModelImport, out: &mut ModelOut) {
        use zestengine_assets::BlendNormals;
        let normals_off = m.normals_mode() == NormalsMode::None;
        widgets::property_row_w(ui, "Blend Shape Normals", LABEL_W, |ui| {
            ui.add_enabled_ui(!normals_off, |ui| {
                out.changed |= widgets::segmented(ui, &mut m.blend_shapes.normals, &[(BlendNormals::Import, "Import"), (BlendNormals::Calculate, "Calculate"), (BlendNormals::None, "None")])
                    .on_hover_text("Import uses the file's normal offsets. Calculate recomputes them from the displaced surface. None keeps the base normals.")
                    .changed();
            });
        });
        widgets::property_row_w(ui, "Delta Threshold", LABEL_W, |ui| {
            out.changed |= ui
                .add(egui::DragValue::new(&mut m.blend_shapes.delta_threshold).speed(1e-6).range(0.0..=1.0).max_decimals(7))
                .on_hover_text("Offsets this small or smaller are dropped; a shape left with none is dropped. Keeps large face meshes small.")
                .changed();
        });
        toggle(ui, out, "Import Curves", &mut m.blend_shapes.import_curves, "Import the blendshape weight curves (DeformPercent) of the file's takes into its clips");
        toggle(ui, out, "Legacy Normals", &mut m.blend_shapes.legacy_normals, "The old importer's smoothing-group normals: recompute them from the displaced mesh whatever Blend Shape Normals says");
        toggle(ui, out, "Deform Percent", &mut m.blend_shapes.import_deform_percent, "Take each shape's initial weight from the file and place it on the object's Blend Shapes when the model is added (off starts every shape at 0)");
    }

    /// The Animation section: import switches, compression, the take list and the clip list.
    fn animation(&self, ui: &mut egui::Ui, path: &Path, m: &mut ModelImport, out: &mut ModelOut) {
        use zestengine_assets::AnimCompression;
        widgets::eyebrow(ui, "Animation");
        toggle(ui, out, "Import Animation", &mut m.animation.import_animation, "Import the file's takes as clips. Off imports none.");
        if !m.animation.import_animation {
            return;
        }
        let a = &mut m.animation;
        toggle(ui, out, "Resample Curves", &mut a.resample_curves, "Sample every source frame. Off keeps only the authored key times (the clip's ends are always sampled). Keep Tangents implies off.");
        toggle(
            ui,
            out,
            "Keep Tangents",
            &mut a.keep_tangents,
            "Keep the authored key times and the shape of the curves between them (Hermite tangents) instead of one sample per frame. Segments are subdivided until the curve is within the errors below of the source. Translation and scale follow the source exactly; rotations use a quaternion Hermite approximation.",
        );
        toggle(
            ui,
            out,
            "Bake Animations",
            &mut a.bake_animations,
            "Evaluate the whole node transform stack per frame - pivots, offsets, inherit modes - and the file's constraints (parent, position, rotation, scale, aim), and store plain local curves. Needed for rigs driven by constraints. IK chains and driven keys are not solved.",
        );
        toggle(
            ui,
            out,
            "Custom Props",
            &mut a.animated_custom_properties,
            "Import the file's user properties: their values land on a Custom Properties component and the animated ones become property tracks of the clips.",
        );
        widgets::property_row_w(ui, "Oversampling", LABEL_W, |ui| {
            let mut n = a.humanoid_oversampling.max(1);
            if widgets::segmented(ui, &mut n, &[(1, "1x"), (2, "2x"), (4, "4x"), (8, "8x")])
                .on_hover_text("How densely a humanoid clip is sampled when it is turned into muscles (multiples of 30 Hz). More follows fast motion and kept tangents more closely, at the price of a larger muscle clip.")
                .changed()
            {
                a.humanoid_oversampling = n;
                out.changed = true;
            }
        });
        widgets::property_row_w(ui, "Anim. Compression", LABEL_W, |ui| {
            out.changed |= widgets::segmented(
                ui,
                &mut a.compression,
                &[
                    (AnimCompression::Off, "Off"),
                    (AnimCompression::KeyframeReduction, "Keyframes"),
                    (AnimCompression::KeyframeReductionAndCompression, "+ Compress"),
                    (AnimCompression::Optimal, "Optimal"),
                ],
            )
            .on_hover_text(
                "Off keeps every sample. Keyframes drops keys a straight line reproduces within the errors below. + Compress also stores the curves as 16 bit values over their own range (when that pays). Optimal picks per curve the smallest of constant, reduced, quantised or reduced and quantised that stays within the errors.",
            )
            .changed();
        });
        if a.compression.reduces() {
            for (label, value, tip) in [
                ("Rotation Error", &mut a.rotation_error, "Largest angle deviation in degrees"),
                ("Position Error", &mut a.position_error, "Largest deviation, percent of the curve's range"),
                ("Scale Error", &mut a.scale_error, "Largest deviation, percent of the curve's range"),
            ] {
                widgets::property_row_w(ui, label, LABEL_W, |ui| {
                    out.changed |= ui.add(egui::DragValue::new(value).speed(0.01).range(0.0..=100.0)).on_hover_text(tip).changed();
                });
            }
        }
        toggle(ui, out, "Remove Const Scale", &mut a.remove_constant_scale_curves, "Drop scale channels that never leave the rest scale");
        toggle(ui, out, "Unreferenced Takes", &mut a.import_unreferenced_takes, "With a custom clip list: also import a clip for each take no clip uses");

        let stats = self.stats(path);
        let takes: Vec<zestengine_assets::TakeInfo> = stats.map(|s| s.takes.clone()).unwrap_or_default();
        let row_info = |name: &str| stats.and_then(|s| s.clip_rows.iter().find(|r| r.0 == name).cloned());
        ui.add_space(4.0);
        if stats.is_some_and(|s| s.meshes == 0 && !s.takes.is_empty()) {
            widgets::info_note(
                ui,
                "Animation-only file: no mesh, its skeleton comes from the node hierarchy (metres, y-up). Drop it on an Animator controller to play its clips on any avatar \
                 by bone name (a humanoid retargets through the muscle model). Set Rig > Copy From Other Avatar to resolve its humanoid map from another model.",
            );
        }
        if a.clips.is_empty() {
            if takes.is_empty() {
                widgets::info_note(ui, if stats.is_some() { "The file has no animation takes." } else { "Reading the file's takes..." });
                return;
            }
            for t in &takes {
                let mut line = format!("{}  \u{b7}  frames {:.0}-{:.0}  \u{b7}  {:.2}s @ {:.0} fps", t.name, t.first_frame, t.last_frame, (t.stop - t.start).max(0.0), t.frame_rate);
                if let Some((_, deg, shapes)) = row_info(&t.name) {
                    line += &format!("  \u{b7}  loop mismatch {deg:.1}\u{b0}");
                    if shapes > 0 {
                        line += &format!("  \u{b7}  {shapes} blendshape curves");
                    }
                }
                ui.label(egui::RichText::new(line).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            }
            if ui.button("Edit Clips").on_hover_text("Cut the takes into your own clips: ranges, loops, events").clicked() {
                a.customize_from_takes(&takes);
                out.changed = true;
            }
            if let Some(rig) = stats.and_then(|s| s.rig.clone()) {
                clip_preview(ui, path, &rig);
            }
            return;
        }
        if let Some(rig) = stats.and_then(|s| s.rig.clone()) {
            clip_preview(ui, path, &rig);
        }
        let mut remove = None;
        for (i, clip) in a.clips.iter_mut().enumerate() {
            let title = if clip.name.is_empty() { "(unnamed)".to_owned() } else { clip.name.clone() };
            egui::CollapsingHeader::new(title).id_salt(("model_clip", i)).default_open(i == 0).show(ui, |ui| {
                widgets::property_row_w(ui, "Name", LABEL_W, |ui| out.changed |= ui.text_edit_singleline(&mut clip.name).changed());
                widgets::property_row_w(ui, "Take", LABEL_W, |ui| {
                    egui::ComboBox::from_id_salt(("model_clip_take", i)).selected_text(if clip.take.is_empty() { "(first take)" } else { clip.take.as_str() }).show_ui(ui, |ui| {
                        for t in &takes {
                            if ui.selectable_value(&mut clip.take, t.name.clone(), &t.name).changed() {
                                clip.first_frame = t.first_frame;
                                clip.last_frame = t.last_frame;
                                out.changed = true;
                            }
                        }
                    });
                });
                widgets::property_row_w(ui, "Start / End", LABEL_W, |ui| {
                    out.changed |= ui.add(egui::DragValue::new(&mut clip.first_frame).speed(0.5).prefix("start ")).changed();
                    out.changed |= ui.add(egui::DragValue::new(&mut clip.last_frame).speed(0.5).prefix("end ")).changed();
                });
                widgets::property_row_w(ui, "Loop Time", LABEL_W, |ui| {
                    out.changed |= ui.checkbox(&mut clip.loop_time, "").on_hover_text("Wrap past the end (on) or hold the last pose (off). The controller graph honours it per clip.").changed();
                    if let Some((_, deg, _)) = row_info(&clip.name) {
                        ui.label(egui::RichText::new(format!("start/end differ by {deg:.1}\u{b0}")).size(theme::SIZE_META).color(if deg < 5.0 { theme::TEXT_MUTED } else { theme::STATUS_WARN_FG }));
                    }
                });
                toggle(ui, out, "Loop Pose", &mut clip.loop_pose, "Blend the end pose into the start pose so the loop has no pop");
                widgets::property_row_w(ui, "Cycle Offset", LABEL_W, |ui| {
                    out.changed |= ui.add(egui::DragValue::new(&mut clip.cycle_offset).speed(0.005).range(0.0..=1.0)).on_hover_text("Where in the loop every pass starts (0..1 of the clip): the controller graph starts the clip this far in").changed();
                });
                toggle(ui, out, "Mirror", &mut clip.mirror, "Play the clip mirrored left/right (humanoid rigs; the controller graph mirrors the pose and the root motion)");
                // Bake Into Pose: what is baked stays in the skeleton; the rest drives the Animator's root motion.
                const BAKE: &str = "Baked: stays in the pose, so the object does not move for it. Not baked: it drives root motion (Apply Root Motion on the Animator).";
                toggle(ui, out, "Bake Root Rotation", &mut clip.lock_root_rotation, BAKE);
                if clip.lock_root_rotation {
                    toggle(ui, out, "  Based Upon Original", &mut clip.keep_original_orientation, "Keep the authored yaw (on) or turn the clip to face forward along its average body orientation (off)");
                    widgets::property_row_w(ui, "  Rotation Offset", LABEL_W, |ui| {
                        out.changed |= ui.add(egui::DragValue::new(&mut clip.root_rotation_offset).speed(0.5).suffix("\u{b0}")).changed();
                    });
                }
                toggle(ui, out, "Bake Root Height", &mut clip.lock_root_height_y, BAKE);
                if clip.lock_root_height_y {
                    toggle(ui, out, "  Based Upon Original", &mut clip.keep_original_position_y, "Keep the authored height (on) or stand the lowest foot of the first frame on the ground (off: Feet)");
                    widgets::property_row_w(ui, "  Height Offset", LABEL_W, |ui| {
                        out.changed |= ui.add(egui::DragValue::new(&mut clip.height_offset).speed(0.01)).changed();
                    });
                }
                toggle(ui, out, "Bake Root Position XZ", &mut clip.lock_root_position_xz, BAKE);
                if clip.lock_root_position_xz {
                    toggle(ui, out, "  Based Upon Original", &mut clip.keep_original_position_xz, "Keep the authored ground position (on) or centre the clip on its average position (off: Center of Mass)");
                }
                widgets::info_note(ui, "Events call a function name when playback passes their time (once per loop pass), with a float, int, string and object parameter. They reach on_animation_event and the Event/Action bindings.");
                let mut drop_event = None;
                for (k, e) in clip.events.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        out.changed |= ui.add(egui::DragValue::new(&mut e.time).speed(0.01).range(0.0..=3600.0).suffix(" s")).changed();
                        out.changed |= ui.text_edit_singleline(&mut e.name).changed();
                        if ui.small_button("x").clicked() {
                            drop_event = Some(k);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(12.0);
                        out.changed |= ui.add(egui::DragValue::new(&mut e.float).speed(0.01).prefix("float ")).changed();
                        out.changed |= ui.add(egui::DragValue::new(&mut e.int).prefix("int ")).changed();
                        out.changed |= ui.add(egui::TextEdit::singleline(&mut e.string).hint_text("string").desired_width(70.0)).changed();
                        out.changed |= ui.add(egui::TextEdit::singleline(&mut e.object).hint_text("object").desired_width(70.0)).changed();
                    });
                }
                if let Some(k) = drop_event {
                    clip.events.remove(k);
                    out.changed = true;
                }
                ui.horizontal(|ui| {
                    if ui.small_button("Add Event").clicked() {
                        clip.events.push(Default::default());
                        out.changed = true;
                    }
                    if ui.small_button("Remove Clip").clicked() {
                        remove = Some(i);
                    }
                });
            });
        }
        if let Some(i) = remove {
            a.clips.remove(i);
            out.changed = true;
        }
        if ui.button("Add Clip").clicked() {
            a.add_clip(&takes);
            out.changed = true;
        }
    }
}

/// The viewing yaw of the clip preview, remembered per model while the editor runs.
fn preview_yaw_id(path: &Path) -> egui::Id {
    egui::Id::new(("model_clip_preview_yaw", path))
}

/// Draw `skeleton` posed by `clip` at `t` as a stick figure into `rect`, seen
/// from `yaw` radians about the vertical. Pure CPU: the clip is sampled, the
/// joints composed, projected and drawn as lines. Returns the number of bones drawn.
pub(crate) fn draw_pose(painter: &egui::Painter, rect: egui::Rect, rig: &RigPreview, clip: usize, t: f32, yaw: f32, color: egui::Color32) -> usize {
    let skel = &rig.skeleton;
    let Some(clip) = rig.clips.get(clip) else { return 0 };
    let world = skel.world_matrices(&clip.sample(skel, t));
    let (sy, cy) = yaw.sin_cos();
    let project = |p: glam::Vec3| glam::Vec2::new(p.x * cy + p.z * sy, p.y);
    // Frame the figure at its rest extent over the whole clip so scrubbing does not rescale it.
    let (mut lo, mut hi) = (glam::Vec2::splat(f32::INFINITY), glam::Vec2::splat(f32::NEG_INFINITY));
    for k in 0..=8 {
        let w = skel.world_matrices(&clip.sample(skel, clip.duration * k as f32 / 8.0 * 0.999));
        for m in &w {
            let p = project(m.w_axis.truncate());
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return 0;
    }
    let size = (hi - lo).max(glam::Vec2::splat(1e-3));
    let scale = (rect.width() * 0.9 / size.x).min(rect.height() * 0.9 / size.y);
    let centre = (lo + hi) * 0.5;
    let to_screen = |p: glam::Vec3| {
        let q = (project(p) - centre) * scale;
        egui::pos2(rect.center().x + q.x, rect.center().y - q.y)
    };
    let mut drawn = 0;
    for (i, j) in skel.joints.iter().enumerate() {
        let here = to_screen(world[i].w_axis.truncate());
        if let Some(p) = j.parent {
            painter.line_segment([to_screen(world[p].w_axis.truncate()), here], egui::Stroke::new(1.5_f32, color));
            drawn += 1;
        }
        painter.circle_filled(here, 2.0, color);
    }
    drawn
}

/// The Clip Preview: a clip picker, a time slider and the skeleton posed by
/// the clip (a CPU-posed stick figure; the viewport model preview cannot pose
/// a skeleton outside a scene). Drag the figure to turn it.
fn clip_preview(ui: &mut egui::Ui, path: &Path, rig: &RigPreview) {
    egui::CollapsingHeader::new("Clip Preview").id_salt(("model_clip_preview", path)).default_open(false).show(ui, |ui| {
        let id = egui::Id::new(("model_clip_preview_state", path));
        let (mut clip, mut t): (usize, f32) = ui.data(|d| d.get_temp(id)).unwrap_or((0, 0.0));
        clip = clip.min(rig.clips.len().saturating_sub(1));
        let Some(c) = rig.clips.get(clip) else { return };
        egui::ComboBox::from_id_salt(("model_clip_preview_pick", path)).selected_text(c.name.clone()).show_ui(ui, |ui| {
            for (i, other) in rig.clips.iter().enumerate() {
                if ui.selectable_value(&mut clip, i, &other.name).changed() {
                    t = 0.0;
                }
            }
        });
        let c = &rig.clips[clip];
        t = t.clamp(0.0, c.duration);
        ui.add(egui::Slider::new(&mut t, 0.0..=c.duration.max(1e-3)).text("s").fixed_decimals(2));
        let keys: usize = c.channels.iter().map(|ch| ch.len()).sum();
        ui.label(egui::RichText::new(format!("{} channels \u{b7} {keys} keys \u{b7} {:.1} KB stored", c.channels.len(), c.stored_bytes() as f32 / 1024.0)).size(theme::SIZE_META).color(theme::TEXT_MUTED));
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(320.0), 220.0), egui::Sense::drag());
        let mut yaw: f32 = ui.data(|d| d.get_temp(preview_yaw_id(path))).unwrap_or(0.0);
        if resp.dragged() {
            yaw -= resp.drag_delta().x * 0.01;
        }
        ui.painter().rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        draw_pose(&ui.painter_at(rect), rect, rig, clip, t, yaw, theme::TEXT_BODY);
        ui.data_mut(|d| {
            d.insert_temp(id, (clip, t));
            d.insert_temp(preview_yaw_id(path), yaw);
        });
    });
}

/// Label column of this section: its Unity-style names are long.
const LABEL_W: f32 = 128.0;

fn toggle(ui: &mut egui::Ui, out: &mut ModelOut, label: &str, value: &mut bool, tip: &str) {
    widgets::property_row_w(ui, label, LABEL_W, |ui| {
        out.changed |= ui.checkbox(value, "").on_hover_text(tip).changed();
    });
}

fn stats_of(s: &zestengine_assets::Scene) -> Stats {
    let names = &s.identity.material_names;
    let mut embedded = Vec::new();
    for (i, mat) in s.materials.iter().enumerate() {
        let name = names.get(i).cloned().unwrap_or_else(|| mat.name.clone());
        for (slot, tex) in TEXTURE_REMAP_SLOTS.iter().zip([mat.albedo, mat.normal, mat.orm, mat.emission]) {
            if tex.is_some() {
                embedded.push(texture_remap_key(&name, slot));
            }
        }
    }
    Stats {
        meshes: s.meshes.len(),
        vertices: s.meshes.iter().map(|m| m.vertices.len()).sum(),
        max_vertices: s.meshes.iter().map(|m| m.vertices.len()).max().unwrap_or(0),
        triangles: s.meshes.iter().map(|m| m.indices.len() / 3).sum(),
        materials: if names.len() == s.materials.len() { names.clone() } else { s.materials.iter().map(|m| m.name.clone()).collect() },
        embedded,
        cameras: s.identity.cameras.len(),
        lights: s.identity.lights.len(),
        clips: s.animations.len(),
        blendshapes: s.morphs.iter().map(Vec::len).sum(),
        takes: s.identity.takes.clone(),
        clip_rows: s
            .animations
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), s.identity.clip_info.get(i).map(|k| k.loop_error_deg).unwrap_or(0.0), c.morph_tracks.len()))
            .collect(),
        warnings: s.identity.import_warnings.clone(),
        constraints: s.identity.constraints.len(),
        custom_props: s.identity.custom_props.iter().map(|(_, v)| v.len()).sum(),
        optimize: s.identity.optimize.clone(),
        rig: s.skeletons.first().filter(|_| !s.animations.is_empty()).map(|skeleton| std::sync::Arc::new(RigPreview { skeleton: skeleton.clone(), clips: s.animations.clone() })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_settings_show_a_relative_path_and_flag_bad_ones() {
        let root = Path::new("/proj");
        let model = Path::new("/proj/art/hero.fbx");
        let d = folder_label(root, model, "", "Textures");
        assert_eq!(d, FolderLabel { text: "art/Textures".into(), is_default: true, error: None });
        let c = folder_label(root, model, "shared/tex", "Textures");
        assert_eq!((c.text.as_str(), c.is_default, c.error.is_none()), ("shared/tex", false, true));
        let bad = folder_label(root, model, "../elsewhere", "Textures");
        assert!(bad.error.is_some() && !bad.is_default);
    }

    #[test]
    fn the_folder_chooser_navigates_by_relative_paths() {
        assert_eq!(parent_folder("a/b/c"), "a/b");
        assert_eq!(parent_folder("a"), "");
        assert_eq!(parent_folder(""), "");
        assert_eq!(child_folder("", "x"), "x");
        assert_eq!(child_folder("a", "x"), "a/x");
        let dir = std::env::temp_dir().join(format!("zest_folder_nav_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("art/sub")).unwrap();
        assert_eq!(nearest_existing_folder(&dir, "art/sub/Textures/deeper"), "art/sub", "opens at the nearest folder that exists");
        assert_eq!(nearest_existing_folder(&dir, "nope/x"), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_list_embedded_texture_slots_by_remap_key() {
        use zestengine_render::{MaterialFeatures, MaterialParams, TextureData};
        let mut s = zestengine_assets::test_scene();
        s.textures = vec![TextureData { width: 1, height: 1, pixels: vec![0; 4], srgb: true, hdr: false }];
        s.materials = vec![zestengine_assets::SceneMaterial {
            name: "Skin".into(),
            params: MaterialParams::default(),
            features: MaterialFeatures::default(),
            albedo: Some(0),
            normal: None,
            orm: None,
            emission: Some(0),
        }];
        s.identity.material_names = vec!["Skin".into()];
        let st = stats_of(&s);
        assert_eq!(st.embedded, ["Skin/albedo", "Skin/emission"]);
        assert_eq!(st.materials, ["Skin"]);
    }

    #[test]
    fn the_clip_preview_poses_the_skeleton_on_the_cpu() {
        use zestengine_assets::{AnimChannel, AnimationClip, Joint, Skeleton};
        let joint = |name: &str, parent: Option<usize>, t: glam::Vec3| Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY,
            rest_translation: t,
            rest_rotation: glam::Quat::IDENTITY,
            rest_scale: glam::Vec3::ONE,
        };
        let skeleton = Skeleton { joints: vec![joint("Root", None, glam::Vec3::ZERO), joint("Arm", Some(0), glam::Vec3::Y), joint("Hand", Some(1), glam::Vec3::Y)], rig: None, ..Default::default() };
        let clip = AnimationClip {
            name: "Wave".into(),
            duration: 1.0,
            channels: vec![AnimChannel::rot(1, vec![0.0, 1.0], vec![glam::Quat::IDENTITY, glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)])],
            morph_tracks: Vec::new(),
            playback: None,
        };
        let rig = RigPreview { skeleton, clips: vec![clip] };
        let ctx = egui::Context::default();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let mut counts = Vec::new();
        for t in [0.0f32, 0.9] {
            let out = ctx.run(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    counts.push(draw_pose(ui.painter(), rect, &rig, 0, t, 0.0, egui::Color32::WHITE));
                });
            });
            assert!(out.shapes.len() >= 5, "a line and a dot per bone: {}", out.shapes.len());
        }
        assert_eq!(counts, [2, 2], "two bones with a parent");
        assert_eq!(draw_pose(&egui::Painter::new(ctx.clone(), egui::LayerId::background(), rect), rect, &rig, 5, 0.0, 0.0, egui::Color32::WHITE), 0, "a clip that does not exist draws nothing");
        // The pose really moves: the hand's projected x differs between t = 0 (straight up) and the end.
        let hand_x = |t: f32| {
            let w = rig.skeleton.world_matrices(&rig.clips[0].sample(&rig.skeleton, t));
            w[2].w_axis.x
        };
        assert!(hand_x(0.0).abs() < 1e-5 && hand_x(0.99) < -0.9, "{} {}", hand_x(0.0), hand_x(0.99));
    }
}

#[cfg(test)]
#[path = "shots/model_import.rs"]
mod shots;
