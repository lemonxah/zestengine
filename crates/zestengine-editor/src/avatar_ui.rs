//! Inspectors for `Avatar Parameters` and `Avatar Menu Item`.

use egui::{DragValue, RichText, Ui};
use zestengine_core::{
    AvatarLayer, AvatarMenuData, AvatarMenuItem, AvatarParam, AvatarParamKind, AvatarParamScope, AvatarParamSync, AvatarParameters, BudgetLevel, BuiltinParam,
    MenuInstaller, MenuItemKind, MenuToggle, MergeAnimator, MergeArmature, MergeMode, SkinRebound, MergePathMode, ObjectToggle, ObjectToggleEntry, ParamBudget, ToggleAction, ToggleMode,
};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx};
use crate::icons::Icon;
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<AvatarParameters>();
    e.register::<AvatarMenuItem>();
    e.register::<MenuToggle>();
    e.register::<MenuInstaller>();
    e.register::<ObjectToggle>();
    e.register::<MergeAnimator>();
    e.register::<MergeArmature>();
    e.register::<SkinRebound>();
    e.register::<AvatarMenuData>();
}

/// egui temp-data key under which the editor hosts publish an [`AvatarInfo`]
/// for the selected object (the inspector has no scene access).
pub const AVATAR_INFO_KEY: &str = "zestengine-avatar-info";
/// Key the Menu Toggle inspector writes a `bool` to (`true` = start the live
/// preview of the selected toggle, `false` = stop it); the host consumes it.
pub const AVATAR_PREVIEW_KEY: &str = "zestengine-avatar-preview-request";

/// What the avatar around the selected object adds up to.
#[derive(Clone, Debug, Default)]
pub struct AvatarInfo {
    /// Synced bits of the declared parameters plus every toggle's own.
    pub used_bits: u32,
    pub max_bits: u32,
    /// `Menu Toggle`s under the avatar.
    pub toggles: usize,
    /// The selected toggle's preview is applied to the scene.
    pub preview_active: bool,
    /// Objects below the avatar root a toggle target can be picked from.
    pub picks: Vec<PickObject>,
}

/// One object below the avatar root, for the target pickers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PickObject {
    /// Index in the scene (what the Hierarchy drag carries).
    pub scene_index: usize,
    /// Path below the avatar root, the way actions address it.
    pub path: String,
    /// Component type names on it (the Component action's choices).
    pub components: Vec<String>,
    /// Blendshape names on its skinned meshes.
    pub blend_shapes: Vec<String>,
}

/// Paths matching `query` (case-insensitive substring), best first: prefix
/// matches of the object's own name, then other matches, each in path order.
pub fn filter_paths<'a>(picks: &'a [PickObject], query: &str) -> Vec<&'a PickObject> {
    let q = query.trim().to_lowercase();
    let leaf = |p: &PickObject| p.path.rsplit('/').next().unwrap_or("").to_lowercase();
    let mut hits: Vec<(u8, &PickObject)> = picks
        .iter()
        .filter(|p| !p.path.is_empty())
        .filter_map(|p| {
            if q.is_empty() {
                Some((1, p))
            } else if leaf(p).starts_with(&q) {
                Some((0, p))
            } else {
                p.path.to_lowercase().contains(&q).then_some((1, p))
            }
        })
        .collect();
    hits.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.path.cmp(&b.1.path)));
    hits.into_iter().map(|(_, p)| p).collect()
}

/// The pick for a Hierarchy drag of scene object `scene_index`.
pub fn pick_for_drop(picks: &[PickObject], scene_index: usize) -> Option<&PickObject> {
    picks.iter().find(|p| p.scene_index == scene_index && !p.path.is_empty())
}

/// What the last "Build preview" of the avatar produced, published by the
/// host under [`AVATAR_GENERATED_KEY`].
#[derive(Clone, Debug, Default)]
pub struct GeneratedOutput {
    /// (layer, where it came from)
    pub layers: Vec<(String, String)>,
    /// (parameter, "Bool" / "Int" / "Float", synced)
    pub params: Vec<(String, String, bool)>,
    /// Generated clip and file paths.
    pub files: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub layers_merged: usize,
    pub synced_bits: u32,
    pub max_bits: u32,
    pub out_dir: String,
}

/// Temp-data key: the [`GeneratedOutput`] to show on the Avatar Descriptor.
pub const AVATAR_GENERATED_KEY: &str = "zestengine-avatar-generated";
/// Temp-data key set (`true`) when "Build preview" is clicked; the host builds
/// the selected avatar without writing files and publishes the result.
pub const AVATAR_BUILD_REQUEST_KEY: &str = "zestengine-avatar-build-request";

/// The "Generated output" foldout of the Avatar Descriptor: layers, parameters
/// and files the build makes, plus its warnings (lints first).
pub fn generated_output_ui(ui: &mut Ui) {
    let out = ui.data(|d| d.get_temp::<GeneratedOutput>(egui::Id::new(AVATAR_GENERATED_KEY)));
    egui::CollapsingHeader::new("Generated output").id_salt("zestengine-avatar-generated-out").show(ui, |ui| {
        note(ui, "What the build makes from the Menu Toggles, Object Toggles and Merge components under this avatar. Nothing is written until Play or a package build.");
        if widgets::secondary_button(ui, None, "Build preview").on_hover_text("Run the build without writing files").clicked() {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(AVATAR_BUILD_REQUEST_KEY), true));
        }
        let Some(out) = out else {
            note(ui, "Not built yet.");
            return;
        };
        note(ui, &format!("{} layer(s) ({} folded), {} parameter(s), {} file(s), {} of {} synced bits.", out.layers.len(), out.layers_merged, out.params.len(), out.files.len(), out.synced_bits, out.max_bits));
        for e in &out.errors {
            ui.label(RichText::new(e).size(theme::SIZE_META).color(theme::STATUS_ERROR));
        }
        let (lints, others): (Vec<&String>, Vec<&String>) = out.warnings.iter().partition(|w| w.starts_with("lint:"));
        for w in lints.iter().chain(others.iter()) {
            ui.label(RichText::new(w.as_str()).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        egui::CollapsingHeader::new(format!("Layers ({})", out.layers.len())).id_salt("zestengine-gen-layers").show(ui, |ui| {
            for (name, src) in &out.layers {
                ui.label(RichText::new(name).size(theme::SIZE_META));
                ui.label(RichText::new(src).size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        });
        egui::CollapsingHeader::new(format!("Parameters ({})", out.params.len())).id_salt("zestengine-gen-params").show(ui, |ui| {
            for (name, kind, synced) in &out.params {
                ui.label(RichText::new(format!("{name}  {kind}{}", if *synced { "  synced" } else { "" })).size(theme::SIZE_META));
            }
        });
        egui::CollapsingHeader::new(format!("Files ({})", out.files.len())).id_salt("zestengine-gen-files").show(ui, |ui| {
            if !out.out_dir.is_empty() {
                note(ui, &out.out_dir);
            }
            for f in &out.files {
                ui.label(RichText::new(f).size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        });
    });
}

/// A path field with a searchable dropdown of the avatar's objects. A
/// Hierarchy drag dropped on it fills the path. `picks` empty = a plain field.
fn target_picker(ui: &mut Ui, salt: impl std::hash::Hash + Copy, target: &mut String, picks: &[PickObject]) -> bool {
    let mut changed = false;
    let id = egui::Id::new(("zestengine-target-picker", salt));
    ui.horizontal(|ui| {
        let w = (ui.available_width() - 26.0).max(60.0);
        let r = ui.add(egui::TextEdit::singleline(target).desired_width(w).hint_text("path below the avatar"));
        changed |= r.changed();
        // A Hierarchy drag dropped on the field.
        if r.hovered()
            && ui.input(|i| i.pointer.any_released())
            && let Some(idx) = ui.data(|d| d.get_temp::<usize>(egui::Id::new(crate::DRAG_OBJECT_KEY)))
            && let Some(p) = pick_for_drop(picks, idx)
        {
            *target = p.path.clone();
            changed = true;
        }
        if !picks.is_empty() {
            ui.menu_button("v", |ui| {
                widgets::menu_frame(ui);
                let qid = id.with("query");
                let mut q = ui.data(|d| d.get_temp::<String>(qid)).unwrap_or_default();
                ui.add(egui::TextEdit::singleline(&mut q).hint_text("Search"));
                ui.data_mut(|d| d.insert_temp(qid, q.clone()));
                egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    for p in filter_paths(picks, &q).into_iter().take(200) {
                        if ui.selectable_label(*target == p.path, &p.path).clicked() {
                            *target = p.path.clone();
                            changed = true;
                            ui.close();
                        }
                    }
                });
            });
        }
    });
    changed
}

/// A dropdown of `choices` writing into `value`, beside a free text field so
/// a name that is not listed (a component on a prefab not loaded) still works.
fn name_picker(ui: &mut Ui, salt: impl std::hash::Hash, value: &mut String, choices: &[String]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let w = (ui.available_width() - if choices.is_empty() { 0.0 } else { 26.0 }).max(60.0);
        changed |= ui.add(egui::TextEdit::singleline(value).desired_width(w)).changed();
        if !choices.is_empty() {
            egui::ComboBox::from_id_salt(("zestengine-name-picker", salt)).width(0.0).selected_text("").show_ui(ui, |ui| {
                for c in choices {
                    if ui.selectable_label(value == c, c).clicked() {
                        *value = c.clone();
                        changed = true;
                    }
                }
            });
        }
    });
    changed
}

/// Component choices on the object at `target`.
fn components_at(picks: &[PickObject], target: &str) -> Vec<String> {
    picks.iter().find(|p| p.path == target).map(|p| p.components.clone()).unwrap_or_default()
}

fn shapes_at(picks: &[PickObject], target: &str) -> Vec<String> {
    picks.iter().find(|p| p.path == target).map(|p| p.blend_shapes.clone()).unwrap_or_default()
}

fn info(ui: &Ui) -> Option<AvatarInfo> {
    ui.data(|d| d.get_temp::<AvatarInfo>(egui::Id::new(AVATAR_INFO_KEY)))
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

/// "used/max synced bits" with a thin bar: neutral, amber past half, red over.
pub fn budget_meter(ui: &mut Ui, budget: ParamBudget) {
    let color = match budget.level() {
        BudgetLevel::Ok => theme::TEXT_MUTED,
        BudgetLevel::Warn => theme::STATUS_WARN_FG,
        BudgetLevel::Over => theme::STATUS_ERROR,
    };
    ui.label(RichText::new(budget.label()).font(theme::mono(theme::SIZE_META)).color(color));
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 4.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 2.0, theme::LINE_PANEL);
    let fill = budget.fraction().clamp(0.0, 1.0);
    if fill > 0.0 {
        let mut r = rect;
        r.set_width(rect.width() * fill);
        ui.painter().rect_filled(r, 2.0, color);
    }
    if budget.is_over() {
        note(ui, &format!("{} bits over: the avatar cannot be built until parameters are removed or made local only.", budget.used - budget.max));
    }
}

fn kind_combo(ui: &mut Ui, salt: impl std::hash::Hash, kind: &mut AvatarParamKind) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(salt).width(60.0).selected_text(kind.label()).show_ui(ui, |ui| {
        for k in AvatarParamKind::ALL {
            changed |= ui.selectable_value(kind, k, k.label()).changed();
        }
    });
    changed
}

impl Inspect for AvatarParameters {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "The values this avatar exposes to its Animator and menu. Put it on the avatar root. Synced parameters cost bits; local-only ones are free.");
        widgets::property_row(ui, "Bit budget", |ui| {
            changed |= ui.add(DragValue::new(&mut self.budget_bits).range(0..=4096)).on_hover_text("Capped by the project's content limit").changed();
        });
        match info(ui) {
            Some(i) => {
                budget_meter(ui, ParamBudget { used: i.used_bits, max: i.max_bits.min(self.budget_bits.max(1)) });
                note(ui, &format!("Counts the declared parameters and the {} toggle(s) under this avatar.", i.toggles));
            }
            None => budget_meter(ui, self.budget(self.budget_bits)),
        }
        widgets::eyebrow(ui, "Parameters");
        let mut remove = None;
        for (i, p) in self.params.iter_mut().enumerate() {
            ui.push_id(("zestengine-avparam", i), |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(110.0).hint_text("Name")).changed();
                    if kind_combo(ui, "kind", &mut p.kind) {
                        p.default = p.coerce(p.default);
                        changed = true;
                    }
                    if widgets::icon_button(ui, Icon::Close, "Remove parameter", false).clicked() {
                        remove = Some(i);
                    }
                });
                ui.horizontal(|ui| {
                    match p.kind {
                        AvatarParamKind::Bool => {
                            let mut on = p.default != 0.0;
                            if ui.checkbox(&mut on, "Default on").changed() {
                                p.default = on as u8 as f32;
                                changed = true;
                            }
                        }
                        AvatarParamKind::Int => {
                            changed |= ui.add(DragValue::new(&mut p.default).range(0..=255).speed(0.2).prefix("Default ")).changed();
                            p.default = p.default.round();
                        }
                        AvatarParamKind::Float => {
                            changed |= ui.add(DragValue::new(&mut p.default).range(-1.0..=1.0).speed(0.01).prefix("Default ")).changed();
                        }
                    }
                    changed |= ui.checkbox(&mut p.saved, "Saved").on_hover_text("Remembered per user between sessions").changed();
                    changed |= scope_ui(ui, i, &mut p.scope);
                    let local = p.sync == AvatarParamSync::LocalOnly;
                    let mut synced = !local;
                    if ui.checkbox(&mut synced, "Synced").on_hover_text("Sent to other players; counts toward the bit budget").changed() {
                        p.sync = if synced { AvatarParamSync::Synced } else { AvatarParamSync::LocalOnly };
                        changed = true;
                    }
                });
            });
            widgets::divider(ui);
        }
        if let Some(i) = remove {
            self.params.remove(i);
            changed = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add parameter").clicked() {
            self.params.push(AvatarParam::new(&format!("Param{}", self.params.len() + 1), AvatarParamKind::Bool));
            changed = true;
        }
        for msg in self.problems() {
            ui.label(RichText::new(msg).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        egui::CollapsingHeader::new("Built-in parameters").id_salt("zestengine-avparam-builtins").show(ui, |ui| {
            note(ui, "Filled in by the engine; use them in controller conditions and drivers. Never saved or synced.");
            for b in BuiltinParam::ALL {
                ui.label(RichText::new(format!("{}  {}", b.name(), b.kind().label())).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
            }
        });
        if changed {
            self.ready = false;
        }
        changed
    }
}
impl Gizmo for AvatarParameters {}

impl Inspect for AvatarMenuItem {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Label", |ui| changed |= ui.text_edit_singleline(&mut self.label).changed());
        widgets::property_row(ui, "Icon", |ui| changed |= ui.text_edit_singleline(&mut self.icon).changed());
        widgets::property_row(ui, "Type", |ui| {
            egui::ComboBox::from_id_salt("zestengine-menu-kind").width(ui.available_width()).selected_text(self.kind.label()).show_ui(ui, |ui| {
                for k in MenuItemKind::ALL {
                    changed |= ui.selectable_value(&mut self.kind, k, k.label()).changed();
                }
            });
        });
        let drives = matches!(self.kind, MenuItemKind::Button | MenuItemKind::Toggle);
        let open_param = matches!(self.kind, MenuItemKind::SubMenu) || self.kind.is_puppet();
        if drives || open_param {
            let label = if drives { "Parameter" } else { "Open parameter" };
            widgets::property_row(ui, label, |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.parameter).hint_text("(auto)")).changed());
            widgets::property_row(ui, "Value", |ui| changed |= ui.add(DragValue::new(&mut self.value).speed(0.05)).changed());
        }
        let need = self.kind.sub_parameter_count();
        if need > 0 {
            self.sub_parameters.resize(need, String::new());
            let (lo, hi) = self.kind.sub_parameter_range();
            let names: &[&str] = match self.kind {
                MenuItemKind::RadialPuppet => &["Rotation"],
                MenuItemKind::TwoAxisPuppet => &["Horizontal", "Vertical"],
                _ => &["Up", "Right", "Down", "Left"],
            };
            for (i, sp) in self.sub_parameters.iter_mut().enumerate() {
                widgets::property_row(ui, names[i], |ui| changed |= ui.text_edit_singleline(sp).changed());
            }
            note(ui, &format!("Each sub-parameter is a Float from {lo} to {hi}."));
            if need == 4 || need == 2 {
                self.labels.resize(4, String::new());
                for (i, l) in self.labels.iter_mut().enumerate() {
                    let dir = ["Up label", "Right label", "Down label", "Left label"][i];
                    widgets::property_row(ui, dir, |ui| changed |= ui.text_edit_singleline(l).changed());
                }
            }
        }
        if self.kind == MenuItemKind::SubMenu {
            widgets::property_row(ui, "Menu asset", |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.sub_menu_asset).hint_text("(use children)")).changed());
            note(ui, "This object's children are the page, up to 8 per page.");
        }
        if drives {
            widgets::property_row(ui, "Saved", |ui| changed |= ui.checkbox(&mut self.saved, "").changed());
            widgets::property_row(ui, "Synced", |ui| changed |= ui.checkbox(&mut self.synced, "").changed());
            widgets::property_row(ui, "Default", |ui| changed |= ui.checkbox(&mut self.is_default, "").changed());
        }
        let known = |_: &str| true;
        for msg in self.problems(&known) {
            ui.label(RichText::new(msg).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        changed
    }
}
impl Gizmo for AvatarMenuItem {}


/// The parameter's scope when this component sits below the avatar root.
fn scope_ui(ui: &mut Ui, salt: usize, scope: &mut AvatarParamScope) -> bool {
    let mut changed = false;
    let label = match scope {
        AvatarParamScope::Global => "Global",
        AvatarParamScope::Internal => "Internal",
        AvatarParamScope::RenameTo(_) => "Rename",
    };
    egui::ComboBox::from_id_salt(("zestengine-avparam-scope", salt))
        .width(70.0)
        .selected_text(label)
        .show_ui(ui, |ui| {
            changed |= ui.selectable_value(scope, AvatarParamScope::Global, "Global").on_hover_text("Shared by name with the whole avatar").changed();
            changed |= ui.selectable_value(scope, AvatarParamScope::Internal, "Internal").on_hover_text("Private to this object's subtree (renamed with a hash)").changed();
            if ui.selectable_label(matches!(scope, AvatarParamScope::RenameTo(_)), "Rename").clicked() && !matches!(scope, AvatarParamScope::RenameTo(_)) {
                *scope = AvatarParamScope::RenameTo(String::new());
                changed = true;
            }
        })
        .response
        .on_hover_text("Only matters on an Avatar Parameters below the avatar root");
    if let AvatarParamScope::RenameTo(n) = scope {
        changed |= ui.add(egui::TextEdit::singleline(n).desired_width(80.0).hint_text("new name")).changed();
    }
    changed
}

fn tags_text(tags: &[String]) -> String {
    tags.join(", ")
}

fn parse_tags(text: &str) -> Vec<String> {
    text.split(',').map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()).collect()
}

fn action_row(ui: &mut Ui, i: usize, a: &mut ToggleAction, picks: &[PickObject]) -> (bool, bool) {
    let mut changed = false;
    let mut remove = false;
    ui.push_id(("zestengine-toggle-action", i), |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(a.label()).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            if widgets::icon_button(ui, Icon::Close, "Remove action", false).clicked() {
                remove = true;
            }
        });
        match a {
            ToggleAction::Active { target, on } => {
                widgets::property_row(ui, "Object", |ui| changed |= target_picker(ui, (i, 0), target, picks));
                widgets::property_row(ui, "Active", |ui| changed |= ui.checkbox(on, "while on").changed());
            }
            ToggleAction::ComponentEnabled { target, component, on } => {
                widgets::property_row(ui, "Object", |ui| changed |= target_picker(ui, (i, 1), target, picks));
                let choices = components_at(picks, target);
                widgets::property_row(ui, "Component", |ui| changed |= name_picker(ui, (i, 1), component, &choices));
                widgets::property_row(ui, "Enabled", |ui| changed |= ui.checkbox(on, "while on").changed());
            }
            ToggleAction::BlendShape { target, shape, value } => {
                widgets::property_row(ui, "Mesh", |ui| changed |= target_picker(ui, (i, 2), target, picks));
                let choices = shapes_at(picks, target);
                widgets::property_row(ui, "Shape", |ui| changed |= name_picker(ui, (i, 2), shape, &choices));
                widgets::property_row(ui, "Weight", |ui| changed |= widgets::slider(ui, egui::Slider::new(value, 0.0..=1.0)).changed());
            }
            ToggleAction::MaterialFloat { target, slot, prop, value, off } => {
                widgets::property_row(ui, "Object", |ui| changed |= target_picker(ui, (i, 3), target, picks));
                widgets::property_row(ui, "Slot", |ui| changed |= ui.add(DragValue::new(slot)).changed());
                widgets::property_row(ui, "Property", |ui| changed |= ui.add(egui::TextEdit::singleline(prop).hint_text("metallic")).changed());
                widgets::property_row(ui, "On", |ui| changed |= ui.add(DragValue::new(value).speed(0.01)).changed());
                widgets::property_row(ui, "Off", |ui| changed |= ui.add(DragValue::new(off).speed(0.01)).changed());
            }
            ToggleAction::MaterialColor { target, slot, prop, value, off } => {
                widgets::property_row(ui, "Object", |ui| changed |= target_picker(ui, (i, 4), target, picks));
                widgets::property_row(ui, "Slot", |ui| changed |= ui.add(DragValue::new(slot)).changed());
                widgets::property_row(ui, "Property", |ui| changed |= ui.add(egui::TextEdit::singleline(prop).hint_text("base_color")).changed());
                widgets::property_row(ui, "On", |ui| changed |= ui.color_edit_button_rgba_unmultiplied(value).changed());
                widgets::property_row(ui, "Off", |ui| changed |= ui.color_edit_button_rgba_unmultiplied(off).changed());
            }
            ToggleAction::Clip { path } => {
                widgets::property_row(ui, "Clip", |ui| changed |= ui.add(egui::TextEdit::singleline(path).hint_text("animations/x.anim")).changed());
            }
            ToggleAction::SetParam { name, value } => {
                widgets::property_row(ui, "Parameter", |ui| changed |= ui.text_edit_singleline(name).changed());
                widgets::property_row(ui, "Value", |ui| changed |= ui.add(DragValue::new(value).speed(0.05)).changed());
            }
        }
    });
    widgets::divider(ui);
    (changed, remove)
}

/// Rows of `actions` with remove buttons and the "+ Add action" menu. `salt`
/// separates two lists in one inspector.
fn actions_list(ui: &mut Ui, salt: usize, actions: &mut Vec<ToggleAction>, picks: &[PickObject]) -> bool {
    let mut changed = false;
    let mut remove = None;
    for (i, a) in actions.iter_mut().enumerate() {
        let (c, r) = action_row(ui, salt * 1000 + i, a, picks);
        changed |= c;
        if r {
            remove = Some(i);
        }
    }
    if let Some(i) = remove {
        actions.remove(i);
        changed = true;
    }
    ui.menu_button("+ Add action", |ui| {
        widgets::menu_frame(ui);
        let mut add = |ui: &mut Ui, a: ToggleAction| {
            actions.push(a);
            ui.close();
            true
        };
        if ui.button("Object").clicked() {
            changed |= add(ui, ToggleAction::Active { target: String::new(), on: true });
            }
        if ui.button("Blend shape").clicked() {
            changed |= add(ui, ToggleAction::BlendShape { target: String::new(), shape: String::new(), value: 1.0 });
            }
        if ui.button("Material value").clicked() {
            changed |= add(ui, ToggleAction::MaterialFloat { target: String::new(), slot: 0, prop: "metallic".into(), value: 1.0, off: 0.0 });
            }
        if ui.button("Material colour").clicked() {
            changed |= add(ui, ToggleAction::MaterialColor { target: String::new(), slot: 0, prop: "base_color".into(), value: [1.0; 4], off: [1.0; 4] });
            }
        if ui.button("Component").clicked() {
            changed |= add(ui, ToggleAction::ComponentEnabled { target: String::new(), component: String::new(), on: true });
            }
        if ui.button("Animation clip").clicked() {
            changed |= add(ui, ToggleAction::Clip { path: String::new() });
            }
        if ui.button("Set parameter").clicked() {
            changed |= add(ui, ToggleAction::SetParam { name: String::new(), value: 1.0 });
            }
    });
    changed
}

impl Inspect for MenuToggle {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "One component builds the parameter, menu entry, animation and controller layer when the avatar is built (Play, package). The object is hidden in the scene while it is a toggle's target.");
        widgets::property_row(ui, "Menu path", |ui| {
            changed |= ui.add(egui::TextEdit::singleline(&mut self.menu_path).hint_text("Outfit/Hat (default: object name)")).changed()
        });
        widgets::property_row(ui, "Icon", |ui| changed |= ui.text_edit_singleline(&mut self.icon).changed());
        widgets::property_row(ui, "Mode", |ui| {
            let label = match self.mode {
                ToggleMode::Toggle => "Toggle",
                ToggleMode::Hold => "Hold",
                ToggleMode::Slider { .. } => "Slider",
            };
            egui::ComboBox::from_id_salt("zestengine-toggle-mode").width(ui.available_width()).selected_text(label).show_ui(ui, |ui| {
                changed |= ui.selectable_value(&mut self.mode, ToggleMode::Toggle, "Toggle").changed();
                changed |= ui.selectable_value(&mut self.mode, ToggleMode::Hold, "Hold").changed();
                if ui.selectable_label(self.mode.is_slider(), "Slider").clicked() && !self.mode.is_slider() {
                    self.mode = ToggleMode::Slider { default: 0.0, inactive_at_zero: true };
                    changed = true;
                }
            });
        });
        if let ToggleMode::Slider { default, inactive_at_zero } = &mut self.mode {
            widgets::property_row(ui, "Default", |ui| changed |= widgets::slider(ui, egui::Slider::new(default, 0.0..=1.0)).changed());
            widgets::property_row(ui, "Off at zero", |ui| changed |= ui.checkbox(inactive_at_zero, "").changed());
        }
        widgets::property_row(ui, "Saved", |ui| changed |= ui.checkbox(&mut self.saved, "").changed());
        widgets::property_row(ui, "Synced", |ui| changed |= ui.checkbox(&mut self.synced, "").on_hover_text("Sent to other players; costs bits").changed());
        if !self.mode.is_slider() {
            widgets::property_row(ui, "Default on", |ui| changed |= ui.checkbox(&mut self.default_on, "").changed());
        }
        match info(ui) {
            Some(i) => {
                budget_meter(ui, ParamBudget { used: i.used_bits, max: i.max_bits });
                if self.synced_bit_cost() > 0 {
                    note(ui, &format!("This toggle adds {} synced bit(s).", self.synced_bit_cost()));
                }
            }
            None => note(ui, &format!("Adds {} synced bit(s) (it must sit under an Avatar Descriptor to be built).", self.synced_bit_cost())),
        }

        let picks = info(ui).map(|i| i.picks).unwrap_or_default();
        widgets::eyebrow(ui, "Actions");
        changed |= actions_list(ui, 0, &mut self.actions, &picks);

        egui::CollapsingHeader::new("Options").id_salt("zestengine-toggle-options").show(ui, |ui| {
            let mut tags = tags_text(&self.exclusive_tags);
            widgets::property_row(ui, "Exclusive tags", |ui| {
                if ui.add(egui::TextEdit::singleline(&mut tags).hint_text("outfit, headwear")).on_hover_text("Toggles sharing a tag switch each other off").changed() {
                    self.exclusive_tags = parse_tags(&tags);
                    changed = true;
                }
            });
            if !self.exclusive_tags.is_empty() {
                widgets::property_row(ui, "Always one on", |ui| {
                    changed |= ui.checkbox(&mut self.exclusive_off_state, "").on_hover_text("Turns itself on when everything it shares a tag with is off").changed()
                });
            }
            widgets::property_row(ui, "Global parameter", |ui| {
                changed |= ui.add(egui::TextEdit::singleline(&mut self.global_param).hint_text("(generated)")).on_hover_text("Use an existing parameter instead of a new one").changed()
            });
            let mut drives = tags_text(&self.drive_globals);
            widgets::property_row(ui, "Drive parameters", |ui| {
                if ui.add(egui::TextEdit::singleline(&mut drives).hint_text("Name, Name")).on_hover_text("Bool parameters set while on, cleared while off").changed() {
                    self.drive_globals = parse_tags(&drives);
                    changed = true;
                }
            });
            widgets::property_row(ui, "Invert rest", |ui| changed |= ui.checkbox(&mut self.invert_rest, "").on_hover_text("The avatar rests in the toggle's on look").changed());
            widgets::property_row(ui, "Selector", |ui| {
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut self.selector).hint_text("(none)"))
                    .on_hover_text("Toggles with the same selector are options of one Int parameter (8 synced bits however many options)")
                    .changed()
            });
            widgets::property_row(ui, "Fade in", |ui| changed |= ui.add(DragValue::new(&mut self.transition_in).speed(0.01).range(0.0..=5.0).suffix(" s")).changed());
            widgets::property_row(ui, "Fade out", |ui| changed |= ui.add(DragValue::new(&mut self.transition_out).speed(0.01).range(0.0..=5.0).suffix(" s")).changed());
        });
        egui::CollapsingHeader::new(format!("Local only actions ({})", self.local_actions.len())).id_salt("zestengine-toggle-local").show(ui, |ui| {
            note(ui, "Run only on your own avatar (IsLocal), never on remote copies: hide your head, show first-person arms.");
            changed |= actions_list(ui, 1, &mut self.local_actions, &picks);
        });

        let preview = info(ui).is_some_and(|i| i.preview_active);
        let label = if preview { "Stop preview" } else { "Preview in scene" };
        if widgets::dashed_button(ui, Some(Icon::Eye), label).clicked() {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(AVATAR_PREVIEW_KEY), !preview));
        }
        if preview {
            note(ui, "Previewing: the scene shows this toggle on. Stop the preview before saving.");
        }
        changed
    }
}
impl Gizmo for MenuToggle {}

impl Inspect for ObjectToggle {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "Follows the nearest Avatar Menu Item above it (or on this object): the listed objects are active while that item is on. Drop a prefab with a menu item and this under an avatar to install it.");
        let picks = info(ui).map(|i| i.picks).unwrap_or_default();
        let mut remove = None;
        for (i, o) in self.objects.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                changed |= target_picker(ui, ("obj-toggle", i), &mut o.target, &picks);
                changed |= ui.checkbox(&mut o.active, "active").changed();
                if widgets::icon_button(ui, Icon::Close, "Remove", false).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.objects.remove(i);
            changed = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add object").clicked() {
            self.objects.push(ObjectToggleEntry::default());
            changed = true;
        }
        widgets::property_row(ui, "Inverted", |ui| changed |= ui.checkbox(&mut self.inverted, "").changed());
        changed
    }
}
impl Gizmo for ObjectToggle {}

impl Inspect for MergeAnimator {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "Adds another animator controller's layers to the avatar's when it is built. Its parameters follow the Avatar Parameters scopes above this object.");
        widgets::property_row(ui, "Controller", |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.controller).hint_text("animations/x.animctrl")).changed());
        widgets::property_row(ui, "Layer", |ui| {
            egui::ComboBox::from_id_salt("zestengine-merge-layer").width(ui.available_width()).selected_text(self.layer.label()).show_ui(ui, |ui| {
                for l in AvatarLayer::ALL {
                    changed |= ui.selectable_value(&mut self.layer, l, l.label()).changed();
                }
            });
        });
        widgets::property_row(ui, "Paths", |ui| {
            egui::ComboBox::from_id_salt("zestengine-merge-paths")
                .width(ui.available_width())
                .selected_text(if self.path_mode == MergePathMode::Relative { "Relative to this object" } else { "Relative to the avatar" })
                .show_ui(ui, |ui| {
                    changed |= ui.selectable_value(&mut self.path_mode, MergePathMode::Relative, "Relative to this object").changed();
                    changed |= ui.selectable_value(&mut self.path_mode, MergePathMode::Absolute, "Relative to the avatar").changed();
                });
        });
        widgets::property_row(ui, "Mode", |ui| {
            egui::ComboBox::from_id_salt("zestengine-merge-mode")
                .width(ui.available_width())
                .selected_text(if self.mode == MergeMode::Append { "Append" } else { "Replace" })
                .show_ui(ui, |ui| {
                    changed |= ui.selectable_value(&mut self.mode, MergeMode::Append, "Append").changed();
                    changed |= ui.selectable_value(&mut self.mode, MergeMode::Replace, "Replace").on_hover_text("Drops the avatar's own layers first").changed();
                });
        });
        widgets::property_row(ui, "Priority", |ui| changed |= ui.add(DragValue::new(&mut self.priority)).on_hover_text("Lower merges earlier").changed());
        widgets::property_row(ui, "Match write defaults", |ui| changed |= ui.checkbox(&mut self.match_write_defaults, "").changed());
        changed
    }
}
impl Gizmo for MergeAnimator {}

impl Inspect for MergeArmature {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "Merges this object's bones into the avatar's when a package is built: matching names become one bone, the rest are re-parented in place.");
        widgets::property_row(ui, "Merge into", |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.target).hint_text("Armature (default)")).changed());
        widgets::property_row(ui, "Prefix", |ui| changed |= ui.text_edit_singleline(&mut self.prefix).changed());
        widgets::property_row(ui, "Suffix", |ui| changed |= ui.text_edit_singleline(&mut self.suffix).changed());
        widgets::property_row(ui, "Rename clashes", |ui| changed |= ui.checkbox(&mut self.mangle_names, "").changed());
        changed
    }
}
impl Gizmo for MergeArmature {}

impl Inspect for SkinRebound {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "Written by the avatar build for an outfit mesh that was merged: the mesh is re-skinned onto the avatar's skeleton when the scene loads. You rarely edit this by hand.");
        widgets::property_row(ui, "Avatar object id", |ui| changed |= ui.text_edit_singleline(&mut self.avatar).changed());
        widgets::property_row(ui, "Prefix", |ui| changed |= ui.text_edit_singleline(&mut self.prefix).changed());
        widgets::property_row(ui, "Suffix", |ui| changed |= ui.text_edit_singleline(&mut self.suffix).changed());
        widgets::property_row(ui, "Fold extra bones", |ui| changed |= ui.checkbox(&mut self.fold_extras, "").on_hover_text("Weights of bones with no partner go to their nearest matched ancestor").changed());
        widgets::property_row(ui, "Auto align", |ui| changed |= ui.checkbox(&mut self.auto_align, "").changed());
        changed
    }
}
impl Gizmo for SkinRebound {}

impl Inspect for MenuInstaller {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        note(ui, "Puts an authored .avmenu page into the avatar's menu when the avatar is built: no wiring needed, drop it with the prefab that owns the menu.");
        widgets::property_row(ui, "Menu asset", |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.menu).hint_text("menus/outfit.avmenu")).changed());
        widgets::property_row(ui, "Install to", |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.install_to).hint_text("Outfits/Extras (empty = menu root)")).changed());
        changed
    }
}
impl Gizmo for MenuInstaller {}

impl Inspect for AvatarMenuData {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        note(ui, &format!("The finished menu: {} entries. Written by the avatar build; edit Avatar Menu Items and Menu Toggles instead.", self.menu.count()));
        false
    }
}
impl Gizmo for AvatarMenuData {}

/// Objects below the avatar around `i` for the toggle target pickers: path,
/// component type names and blendshape names (the avatar root is left out).
pub fn pick_list(scene: &zestengine_engine::scene::LoadedScene, i: usize) -> Vec<PickObject> {
    let Some(root) = crate::avatar_edit::avatar_root_of(scene, i) else { return Vec::new() };
    (0..scene.objects.len())
        .filter(|&k| k != root)
        .filter_map(|k| {
            let path = crate::avatar_edit::path_below(scene, root, k)?;
            Some(PickObject {
                scene_index: k,
                path,
                components: scene.objects[k].components.iter().map(|c| c.type_name().to_owned()).collect(),
                blend_shapes: scene.morph_names(k),
            })
        })
        .collect()
}

/// Run the build of the avatar at `root` without writing anything and
/// summarise it for the "Generated output" foldout.
pub fn generated_output(
    scene: &zestengine_engine::scene::LoadedScene,
    root: usize,
    project_root: &std::path::Path,
    shaders: &zestengine_engine::shaders::ShaderLibrary,
    budget_cap: u32,
) -> GeneratedOutput {
    use zestengine_assets::avatar_build::{BuildEnv, avatar_roots, build_avatar_cloned, subtree};
    let file = scene.to_scene_file(project_root, shaders);
    // `root` is an index in the live scene; the file keeps the same order.
    let env = BuildEnv { project_root, write: false, structural: false, budget_cap };
    let target = if avatar_roots(&file).contains(&root) { root } else { return Default::default() };
    let _ = subtree(&file, target);
    let (_, b) = build_avatar_cloned(&file, target, &env);
    GeneratedOutput {
        layers: b.layers.iter().map(|l| (l.name.clone(), l.source.clone())).collect(),
        params: b.params.iter().map(|p| (p.name.clone(), p.kind.label().to_owned(), p.is_synced())).collect(),
        files: b.files.keys().cloned().collect(),
        warnings: b.warnings,
        errors: b.errors,
        layers_merged: b.layers_merged,
        synced_bits: b.budget.map_or(0, |x| x.used),
        max_bits: b.budget.map_or(0, |x| x.max),
        out_dir: b.out_dir,
    }
}

#[cfg(test)]
mod pick_tests {
    use super::*;

    fn pick(i: usize, path: &str) -> PickObject {
        PickObject { scene_index: i, path: path.into(), components: vec!["Light".into()], blend_shapes: vec!["Smile".into()] }
    }

    #[test]
    fn paths_filter_by_name_first_then_path() {
        let picks = vec![pick(1, "Body"), pick(2, "Outfit/Hat"), pick(3, "Hats/Box"), pick(4, "Hat"), pick(0, "")];
        let hits: Vec<&str> = filter_paths(&picks, "hat").iter().map(|p| p.path.as_str()).collect();
        // Own-name prefix matches first (Hat, Outfit/Hat), then path matches.
        assert_eq!(hits, vec!["Hat", "Outfit/Hat", "Hats/Box"]);
        assert_eq!(filter_paths(&picks, "").len(), 4, "the avatar root itself is never offered");
        assert!(filter_paths(&picks, "zzz").is_empty());
    }

    #[test]
    fn a_hierarchy_drop_resolves_to_the_objects_path() {
        let picks = vec![pick(5, "Outfit/Hat")];
        assert_eq!(pick_for_drop(&picks, 5).map(|p| p.path.as_str()), Some("Outfit/Hat"));
        assert!(pick_for_drop(&picks, 6).is_none());
        assert_eq!(components_at(&picks, "Outfit/Hat"), vec!["Light".to_owned()]);
        assert_eq!(shapes_at(&picks, "Nope"), Vec::<String>::new());
    }
}
