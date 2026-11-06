//! The model inspector's Rig section, the avatar Configure window and the
//! viewport skeleton overlay.
//!
//! The Rig section edits the [`RigImport`] stored in a model's `.meta`
//! (animation type, avatar setup, root node, bone map) and shows what
//! [`validate_rig`](zestengine_assets::human_rig::validate_rig) thinks of it.
//! Configure mode is a window over the editor: a 2D figure (Body, Head and
//! both hands) whose dots are the human bones, coloured by validation state,
//! next to a list of bone rows with joint pickers, a Mapping menu (Clear,
//! Biped, Automap, Load, Save) and a Pose menu (Reset, Sample Bind-Pose,
//! Enforce T-Pose). Edits go to the same `FileMeta` session as the other import
//! settings, so Apply / Revert follow the model inspector's dirty pattern.
//!
//! The skeleton is parsed on a worker thread (through the import cache) the
//! first time a model's Rig section is opened.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use zestengine_assets::auto_map::biped_map;
use zestengine_assets::human_rig::{ALL, AnimationType, AvatarSetup, HumanBone, RigImport};
use zestengine_assets::human_template::{HumanMap, apply_template, load_humanmap, load_unity_ht, save_humanmap};
use zestengine_assets::rig_config::{
    BoneReport, BoneState, Group, GroupState, assign_bone, bone_report, enforce_tpose, group_state, pose_errors, reset_pose, sample_bind, summary_line,
};
use zestengine_assets::Skeleton;
use crate::icons::Icon;
use crate::{theme, widgets};

/// Tooltip of the Force T-Pose button.
const FORCE_TPOSE_TIP: &str = "Retargeting and muscles treat this as the neutral pose; use it for A-pose or otherwise non-T-posed models. Sets the exact canonical T-pose (arms horizontal, legs straight down) with the model's own bone lengths; Revert undoes it.";

/// What the Rig section / Configure window asks the inspector to do.
#[derive(Default)]
pub struct RigOut {
    /// The rig was edited (refresh the dirty flag).
    pub changed: bool,
    /// Apply: save the settings and reimport the model.
    pub apply: bool,
    /// Revert: reload the saved settings.
    pub revert: bool,
}

/// The figure shown in Configure mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FigureView {
    #[default]
    Body,
    Head,
    LeftHand,
    RightHand,
}

impl FigureView {
    fn groups(self) -> &'static [Group] {
        match self {
            Self::Body => &[Group::Body, Group::LeftArm, Group::RightArm, Group::LeftLeg, Group::RightLeg],
            Self::Head => &[Group::Head],
            Self::LeftHand => &[Group::LeftHand],
            Self::RightHand => &[Group::RightHand],
        }
    }
}

struct Loaded {
    path: PathBuf,
    skel: Option<Skeleton>,
    /// Skeletons in the model (the stored rig applies to single-skeleton models).
    skeletons: usize,
    error: Option<String>,
}

type LoadResult = Result<(Option<Skeleton>, usize), String>;

/// Validation of the last rig seen, so painting does not re-validate every frame.
struct Cache {
    rig: RigImport,
    joints: usize,
    last_joint: String,
    report: BoneReport,
    /// Bones outside their T-pose tolerance in the displayed (file) pose.
    pose_bad: Vec<HumanBone>,
}

/// Editor state of the Rig section and Configure window.
#[derive(Default)]
pub struct RigUi {
    loaded: Option<Loaded>,
    rx: Option<(PathBuf, Receiver<LoadResult>)>,
    /// The Configure window is open.
    pub configure: bool,
    /// Draw the skeleton in the viewport for the selected skinned object.
    pub overlay: bool,
    view: FigureView,
    selected: Option<HumanBone>,
    filter: String,
    note: String,
    cache: Option<Cache>,
    /// The Muscles & Settings tab of Configure mode.
    pub muscles: crate::rig_muscles::MuscleUi,
    /// The 3D figure (offscreen render of the model being configured).
    pub preview: crate::model_preview::ModelPreview,
    /// The scene instance of the model being configured (its world matrix), set
    /// by the editor each frame; without one the figure is drawn with the painter.
    avatar_world: Option<glam::Mat4>,
    /// The pose the instance should show (reference pose, forced T-pose or the
    /// muscle sliders), taken by the editor each frame.
    pose_override: Option<Vec<glam::Mat4>>,
    /// The model the Configure window edits.
    model_path: Option<PathBuf>,
    /// What the camera was last framed for: the view and its (rounded) bounds.
    framed: Option<(FigureView, [i32; 6])>,
}

impl RigUi {
    /// The skeleton posed by the Muscles & Settings sliders, while that tab of Configure mode is open.
    pub fn muscle_preview(&mut self, skel: &Skeleton, rig: &RigImport) -> Option<Vec<glam::Mat4>> {
        if !self.configure {
            return None;
        }
        self.muscles.preview_locals(skel, rig)
    }

    /// The model file the open Configure window edits.
    pub fn configure_model(&self) -> Option<&Path> {
        self.model_path.as_deref().filter(|_| self.configure)
    }

    /// The scene instance of that model (world matrix), or `None` when no object
    /// in the scene uses it or there is no renderer: the painter figure is used then.
    pub fn set_avatar(&mut self, world: Option<glam::Mat4>) {
        self.avatar_world = world;
    }

    /// The pose the Configure window wants the scene instance to show this frame.
    pub fn take_pose_override(&mut self) -> Option<Vec<glam::Mat4>> {
        self.pose_override.take()
    }

    /// Start (or poll) the skeleton load for `path`.
    fn ensure_loaded(&mut self, path: &Path) {
        if let Some((p, rx)) = &self.rx {
            match rx.try_recv() {
                Ok(res) => {
                    let (skel, skeletons, error) = match res {
                        Ok((s, n)) => (s, n, None),
                        Err(e) => (None, 0, Some(e)),
                    };
                    self.loaded = Some(Loaded { path: p.clone(), skel, skeletons, error });
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
        let (tx, rx) = std::sync::mpsc::channel();
        let owned = path.to_path_buf();
        std::thread::spawn(move || {
            let res = zestengine_assets::load_model_cached(&owned, false)
                .map(|(scene, _)| {
                    let n = scene.skeletons.len();
                    (scene.skeletons.into_iter().next(), n)
                })
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(res);
        });
        self.rx = Some((path.to_path_buf(), rx));
    }

    /// Force the next [`Self::ensure_loaded`] to parse the model again (after a reimport).
    pub fn forget(&mut self) {
        self.loaded = None;
        self.cache = None;
    }

    /// The bone selected in Configure mode.
    pub fn selected_bone(&self) -> Option<HumanBone> {
        self.selected
    }

    fn skeleton(&self, path: &Path) -> Option<&Skeleton> {
        self.loaded.as_ref().filter(|l| l.path == path).and_then(|l| l.skel.as_ref())
    }

    /// Per-bone states of `rig` on `skel` (cached while neither changes).
    pub fn report(&mut self, rig: &RigImport, skel: &Skeleton) -> (&BoneReport, &[HumanBone]) {
        let last = skel.joints.last().map(|j| j.name.clone()).unwrap_or_default();
        let fresh = self.cache.as_ref().is_some_and(|c| c.rig == *rig && c.joints == skel.joints.len() && c.last_joint == last);
        if !fresh {
            self.cache = Some(Cache {
                rig: rig.clone(),
                joints: skel.joints.len(),
                last_joint: last,
                report: bone_report(skel, &rig.human),
                pose_bad: pose_errors(skel, rig, false),
            });
        }
        let c = self.cache.as_ref().unwrap();
        (&c.report, &c.pose_bad)
    }

    // ------------------------------------------------------------------ section

    /// The Rig section of the model inspector.
    pub fn section(&mut self, ui: &mut egui::Ui, path: &Path, rig: &mut RigImport, dirty: bool) -> RigOut {
        self.ensure_loaded(path);
        let mut out = RigOut::default();
        ui.add_space(6.0);
        widgets::eyebrow(ui, "Rig");

        let mut ty = rig.animation_type;
        widgets::property_row(ui, "Animation Type", |ui| {
            widgets::segmented(ui, &mut ty, &[(AnimationType::None, "None"), (AnimationType::Generic, "Generic"), (AnimationType::Humanoid, "Humanoid")]);
        });
        if ty != rig.animation_type {
            rig.animation_type = ty;
            rig.auto_generated = false;
            if ty == AnimationType::Humanoid && rig.human.is_empty() {
                self.automap_into(path, rig);
            }
            out.changed = true;
        }

        let copy = matches!(rig.avatar_setup, AvatarSetup::CopyFrom(_));
        let mut want_copy = copy;
        widgets::property_row(ui, "Avatar Definition", |ui| {
            widgets::segmented(ui, &mut want_copy, &[(false, "Create From This Model"), (true, "Copy From Other")]);
        });
        if want_copy != copy {
            rig.avatar_setup = if want_copy { AvatarSetup::CopyFrom(String::new()) } else { AvatarSetup::CreateFromThisModel };
            out.changed = true;
        }
        let mut copy_from: Option<String> = None;
        if let AvatarSetup::CopyFrom(src) = &mut rig.avatar_setup {
            widgets::property_row(ui, "Source", |ui| {
                let r = ui.add(egui::TextEdit::singleline(src).hint_text("models/hero.fbx").desired_width(f32::INFINITY));
                out.changed |= r.changed();
                if r.lost_focus() || r.changed() {
                    copy_from = Some(src.clone());
                }
            });
            widgets::info_note(ui, "The source model's bone map is copied by joint name each time this model loads; this model's own map is the fallback when it does not fit.");
        }
        // Show the copy as soon as the source names a model that fits.
        if let (Some(src), Some(skel)) = (copy_from, self.skeleton(path))
            && let Some(copied) = zestengine_assets::rig_import::copy_rig_from(path, &src, skel)
        {
            rig.human = copied.human;
            rig.reference_pose = copied.reference_pose;
            rig.root_node = copied.root_node;
            rig.animation_type = AnimationType::Humanoid;
            rig.auto_generated = false;
        }

        let skel = self.skeleton(path).cloned();
        if rig.animation_type == AnimationType::Generic {
            widgets::property_row(ui, "Root node", |ui| {
                let label = if rig.root_node.is_empty() { "(skeleton root)".to_string() } else { rig.root_node.clone() };
                let btn = widgets::secondary_button(ui, None, &label);
                egui::Popup::menu(&btn).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
                    widgets::menu_frame(ui);
                    ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Filter").desired_width(f32::INFINITY));
                    egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                        if ui.selectable_label(rig.root_node.is_empty(), "(skeleton root)").clicked() {
                            rig.root_node.clear();
                            out.changed = true;
                            ui.close();
                        }
                        for j in skel.iter().flat_map(|s| s.joints.iter()) {
                            if !self.filter.is_empty() && !j.name.to_lowercase().contains(&self.filter.to_lowercase()) {
                                continue;
                            }
                            if ui.selectable_label(rig.root_node == j.name, &j.name).clicked() {
                                rig.root_node = j.name.clone();
                                out.changed = true;
                                ui.close();
                            }
                        }
                    });
                });
            });
        }

        if rig.animation_type == AnimationType::Humanoid {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(summary_line(rig)).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                if rig.auto_generated {
                    widgets::badge(ui, if rig.from_file { "from file" } else { "automatic" });
                }
            });
            match &skel {
                Some(s) => {
                    let (report, _) = self.report(rig, s);
                    let n = report.error_count();
                    ui.horizontal(|ui| {
                        if n == 0 && rig.has_required_bones() {
                            widgets::status_pill(ui, "Valid humanoid", theme::STATUS_OK);
                        } else {
                            widgets::status_pill(ui, &format!("{n} issue{}", if n == 1 { "" } else { "s" }), theme::STATUS_ERROR);
                        }
                    });
                    let mut lines: Vec<&String> = report.general.iter().collect();
                    for b in ALL {
                        lines.extend(report.messages(b));
                    }
                    for l in lines.into_iter().take(12) {
                        ui.add(egui::Label::new(egui::RichText::new(l).size(theme::SIZE_META).color(theme::STATUS_ERROR)).wrap());
                    }
                }
                None => {
                    let msg = match self.loaded.as_ref().and_then(|l| l.error.as_ref()) {
                        Some(e) => format!("Couldn't read the skeleton: {e}"),
                        None if self.loaded.as_ref().is_some_and(|l| l.path == path) => "This model has no skeleton.".to_string(),
                        None => "Reading the skeleton...".to_string(),
                    };
                    ui.label(egui::RichText::new(msg).size(theme::SIZE_META).color(theme::TEXT_DIM));
                }
            }
            if self.loaded.as_ref().is_some_and(|l| l.path == path && l.skeletons > 1) {
                widgets::info_note(ui, "This model has several skeletons; the stored rig applies to the first one.");
            }
            ui.horizontal(|ui| {
                if widgets::secondary_button(ui, Some(Icon::Gear), "Configure...").on_hover_text("Edit the bone mapping").clicked() {
                    self.configure = true;
                }
                let force = ui.add_enabled(skel.is_some(), egui::Button::new("Force T-Pose")).on_hover_text(FORCE_TPOSE_TIP);
                if force.clicked()
                    && let Some(s) = &skel
                {
                    if crate::rig_extras::apply_biped_pose(rig, s) {
                        out.changed = true;
                    } else {
                        self.note = "Force T-Pose needs Hips mapped.".into();
                    }
                }
                ui.checkbox(&mut self.overlay, "Show skeleton").on_hover_text("Draw the skeleton in the viewport for the selected character");
            });
            if ui
                .checkbox(&mut rig.force_tpose, "Always force T-Pose on import")
                .on_hover_text("Reimporting recomputes the reference pose as the exact canonical T-pose instead of the closer of bind and file pose")
                .changed()
            {
                if rig.force_tpose
                    && let Some(s) = &skel
                {
                    crate::rig_extras::apply_biped_pose(rig, s);
                }
                rig.auto_generated = false;
                out.changed = true;
            }
        }

        self.model_path = Some(path.to_path_buf());
        if self.configure {
            let w = self.configure_window(ui.ctx(), path, rig, dirty);
            out.changed |= w.changed;
            out.apply |= w.apply;
            out.revert |= w.revert;
        }
        out
    }

    /// Run the auto-mapper into `rig` (does nothing until the skeleton has loaded).
    fn automap_into(&mut self, path: &Path, rig: &mut RigImport) {
        let Some(skel) = self.skeleton(path) else {
            self.note = "The skeleton is still loading; try again in a moment.".into();
            return;
        };
        let auto = zestengine_engine::humanoid::auto_rig_import(skel);
        rig.human = auto.human;
        rig.reference_pose = auto.reference_pose;
        rig.animation_type = AnimationType::Humanoid;
        rig.auto_generated = false;
        self.note = format!("Automap found {} bones.", rig.human.len());
    }

    // ---------------------------------------------------------------- configure

    fn configure_window(&mut self, ctx: &egui::Context, path: &Path, rig: &mut RigImport, dirty: bool) -> RigOut {
        let mut out = RigOut::default();
        let mut open = true;
        let mut done = false;
        let mut ask = false;
        let skel = self.skeleton(path).cloned();
        let confirm_id = egui::Id::new("zestengine-rig-confirm-done");
        egui::Window::new("Avatar Configuration")
            .open(&mut open)
            .default_size(Vec2::new(700.0, 560.0))
            .collapsible(false)
            .show(ctx, |ui| {
                let Some(skel) = skel.as_ref() else {
                    ui.label("Reading the skeleton...");
                    return;
                };
                widgets::segmented(ui, &mut self.muscles.active, &[(false, "Mapping"), (true, "Muscles & Settings")]);
                ui.add_space(4.0);
                if self.muscles.active {
                    self.muscles.tab(ui, skel, rig, &mut out);
                }
                if !self.muscles.active {
                self.menus(ui, path, skel, rig, &mut out);
                ui.add_space(4.0);
                let (report, pose_bad) = {
                    let (r, p) = self.report(rig, skel);
                    (r.clone(), p.to_vec())
                };
                ui.columns(2, |cols| {
                    self.figure_column(&mut cols[0], skel, rig, &report);
                    self.list_column(&mut cols[1], skel, rig, &report, &mut out);
                });
                out.changed |= crate::rig_extras::transform_section(ui, skel, rig, self.selected);
                if !pose_bad.is_empty() {
                    ui.label(egui::RichText::new("Character is not in T pose").color(theme::STATUS_ERROR).size(theme::SIZE_BODY));
                }
                if !self.note.is_empty() {
                    ui.label(egui::RichText::new(&self.note).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                }
                }
                widgets::dialog_footer(ui, |ui| {
                    ui.add_enabled_ui(dirty, |ui| {
                        if widgets::primary_button(ui, None, "Apply").on_hover_text("Save the mapping and reimport the model").clicked() {
                            out.apply = true;
                        }
                        if widgets::secondary_button(ui, None, "Revert").on_hover_text("Discard the edits and go back to the saved settings").clicked() {
                            out.revert = true;
                        }
                    });
                    if widgets::secondary_button(ui, None, "Done").on_hover_text("Close; asks what to do with unapplied changes").clicked() {
                        // Unapplied edits are never dropped silently.
                        if dirty {
                            ask = true;
                        } else {
                            done = true;
                        }
                    }
                });
                if ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
                    && !ui.ctx().wants_keyboard_input()
                    && let Some(b) = self.selected
                {
                    assign_bone(&mut rig.human, b, None);
                    rig.auto_generated = false;
                    out.changed = true;
                }
            });
        // Done with unapplied changes: Apply / Revert / Cancel.
        let mut asking = ctx.data_mut(|d| d.get_temp::<bool>(confirm_id)).unwrap_or(false) || ask;
        if asking && !dirty {
            asking = false;
        }
        if asking {
            egui::Window::new("Unapplied changes").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO).show(ctx, |ui| {
                ui.label("The bone mapping has changes that are not applied to the model.");
                widgets::dialog_footer(ui, |ui| {
                    if widgets::primary_button(ui, None, "Apply").on_hover_text("Save the mapping and reimport the model, then close").clicked() {
                        out.apply = true;
                        done = true;
                        asking = false;
                    }
                    if widgets::secondary_button(ui, None, "Revert").on_hover_text("Discard the edits, then close").clicked() {
                        out.revert = true;
                        done = true;
                        asking = false;
                    }
                    if widgets::secondary_button(ui, None, "Cancel").on_hover_text("Keep editing").clicked() {
                        asking = false;
                    }
                });
            });
        }
        ctx.data_mut(|d| d.insert_temp(confirm_id, asking));
        if !open || done {
            self.configure = false;
            self.preview.reset();
            self.framed = None;
            self.pose_override = None;
            ctx.data_mut(|d| d.insert_temp(confirm_id, false));
        }
        out
    }

    /// Mapping and Pose menus.
    fn menus(&mut self, ui: &mut egui::Ui, path: &Path, skel: &Skeleton, rig: &mut RigImport, out: &mut RigOut) {
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        let biped = biped_map(skel);
        ui.horizontal(|ui| {
            let mapping = widgets::secondary_button(ui, None, "Mapping");
            egui::Popup::menu(&mapping).show(|ui| {
                widgets::menu_frame(ui);
                if widgets::menu_item(ui, None, "Clear", None).clicked() {
                    rig.human.clear();
                    rig.auto_generated = false;
                    out.changed = true;
                    ui.close();
                }
                if let Some(b) = &biped
                    && widgets::menu_item(ui, None, "Biped", None).clicked()
                {
                    rig.human = b.to_human(skel);
                    rig.auto_generated = false;
                    out.changed = true;
                    ui.close();
                }
                if widgets::menu_item(ui, None, "Automap", None).clicked() {
                    self.automap_into(path, rig);
                    out.changed = true;
                    ui.close();
                }
                widgets::menu_separator(ui);
                let files = mapping_files(&dir);
                if files.is_empty() {
                    ui.label(egui::RichText::new("No .humanmap or .ht files next to the model").size(theme::SIZE_META).color(theme::TEXT_DIM));
                }
                for f in files {
                    let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if widgets::menu_item(ui, None, &format!("Load {name}"), None).clicked() {
                        match load_mapping(&f) {
                            Ok(map) => {
                                rig.human = apply_template(skel, &map);
                                rig.animation_type = AnimationType::Humanoid;
                                rig.auto_generated = false;
                                out.changed = true;
                                self.note = format!("Loaded {name}: {} of {} bones found in this model.", rig.human.len(), map.bones.len());
                            }
                            Err(e) => self.note = format!("{e:#}"),
                        }
                        ui.close();
                    }
                }
                widgets::menu_separator(ui);
                let target = dir.join(format!("{stem}.humanmap"));
                if widgets::menu_item(ui, None, &format!("Save {stem}.humanmap"), None).clicked() {
                    self.note = match save_humanmap(&target, &HumanMap::new(stem.clone(), &rig.human)) {
                        Ok(()) => format!("Saved {}", target.display()),
                        Err(e) => format!("{e:#}"),
                    };
                    ui.close();
                }
            });
            let pose = widgets::secondary_button(ui, None, "Pose");
            egui::Popup::menu(&pose).show(|ui| {
                widgets::menu_frame(ui);
                if widgets::menu_item(ui, None, "Reset", None).on_hover_text("Use the pose stored in the file").clicked() {
                    reset_pose(rig);
                    rig.auto_generated = false;
                    out.changed = true;
                    ui.close();
                }
                if widgets::menu_item(ui, None, "Sample Bind-Pose", None).clicked() {
                    if sample_bind(rig, skel) {
                        rig.auto_generated = false;
                        out.changed = true;
                    } else {
                        self.note = "This model has no skin bind data.".into();
                    }
                    ui.close();
                }
                if widgets::menu_item(ui, None, "Enforce T-Pose", None).clicked() {
                    enforce_tpose(rig, skel);
                    rig.auto_generated = false;
                    out.changed = true;
                    ui.close();
                }
                if widgets::menu_item(ui, None, "Force T-Pose", None).on_hover_text(FORCE_TPOSE_TIP).clicked() {
                    if crate::rig_extras::apply_biped_pose(rig, skel) {
                        out.changed = true;
                    } else {
                        self.note = "Force T-Pose needs Hips mapped.".into();
                    }
                    ui.close();
                }
                if widgets::menu_item(ui, None, "Biped Pose", None).on_hover_text("An exact T-pose from the mapping: spine up, arms along X, legs down").clicked() {
                    if crate::rig_extras::apply_biped_pose(rig, skel) {
                        out.changed = true;
                    } else {
                        self.note = "Biped Pose needs Hips mapped.".into();
                    }
                    ui.close();
                }
            });
        });
    }

    /// The figure with its view buttons: a 3D view of the model when the scene has
    /// an instance of it, else the painter silhouette.
    fn figure_column(&mut self, ui: &mut egui::Ui, skel: &Skeleton, rig: &RigImport, report: &BoneReport) {
        widgets::segmented(
            ui,
            &mut self.view,
            &[(FigureView::Body, "Body"), (FigureView::Head, "Head"), (FigureView::LeftHand, "Left Hand"), (FigureView::RightHand, "Right Hand")],
        );
        let size = Vec2::new(ui.available_width().min(300.0), 400.0);
        match self.avatar_world {
            Some(world) => self.figure_3d(ui, size, world, skel, rig, report),
            None => self.figure_painted(ui, size, rig, report),
        }
        ui.horizontal(|ui| {
            legend_dot(ui, true, "Required");
            legend_dot(ui, false, "Optional");
        });
        if self.avatar_world.is_none() {
            widgets::info_note(ui, "The model could not be shown in 3D here.");
        }
    }

    /// The pose the 3D figure shows: the muscle sliders, else the rig's reference pose.
    fn figure_pose(&mut self, skel: &Skeleton, rig: &RigImport) -> Vec<glam::Mat4> {
        self.muscles.preview_locals(skel, rig).unwrap_or_else(|| reference_locals_of(skel, rig))
    }

    fn figure_3d(&mut self, ui: &mut egui::Ui, size: Vec2, world: glam::Mat4, skel: &Skeleton, rig: &RigImport, report: &BoneReport) {
        let locals = self.figure_pose(skel, rig);
        let positions = overlay_joint_positions(skel, &locals, glam::Mat4::IDENTITY);
        self.pose_override = Some(locals);
        // Frame the view's bones when the view changes or the bones moved a lot (Force T-Pose, new mapping).
        let focus = focus_points(self.view, skel, rig, &positions);
        let bounds = bounds_key(&focus);
        if self.framed != Some((self.view, bounds)) && !focus.is_empty() {
            let (yaw, pitch) = match self.view {
                FigureView::Body => (0.0, 0.1),
                FigureView::Head => (0.0, 0.05),
                FigureView::LeftHand | FigureView::RightHand => (0.0, 80f32.to_radians()),
            };
            if let Some(goal) = crate::model_preview::frame_points(&focus, yaw, pitch, size.x / size.y) {
                self.preview.go_to(goal);
            }
            self.framed = Some((self.view, bounds));
        }
        let mut clicked = None;
        let selected = self.selected;
        let view = self.view;
        let waiting = !self.preview.has_texture();
        self.preview.ui(ui, size, world, |painter, proj, resp| {
            if waiting {
                painter.text(proj.rect().center(), egui::Align2::CENTER_CENTER, "Rendering...", theme::mono(theme::SIZE_META), theme::TEXT_DIM);
            }
            let mut items: Vec<(HumanBone, Pos2, f32)> = Vec::new();
            for group in view.groups() {
                for bone in group.bones() {
                    if matches!(view, FigureView::LeftHand | FigureView::RightHand) && matches!(bone, HumanBone::LeftHand | HumanBone::RightHand) {
                        continue;
                    }
                    let Some(j) = rig.joint_name(bone).and_then(|n| skel.joints.iter().position(|x| x.name == n)) else { continue };
                    if let Some((p, depth)) = proj.at(positions[j])
                        && proj.rect().contains(p)
                    {
                        items.push((bone, p, depth));
                    }
                }
            }
            let hover = resp.hover_pos().and_then(|at| crate::model_preview::pick(&items, at, 10.0));
            for (bone, p, _) in &items {
                draw_dot(painter, *p, *bone, report.state(*bone), true, selected == Some(*bone), hover == Some(*bone));
            }
            if let Some(b) = hover
                && let Some(at) = resp.hover_pos()
            {
                let text = format!("{}: {}", b.name(), rig.joint_name(b).unwrap_or(""));
                let galley = painter.layout_no_wrap(text, theme::mono(theme::SIZE_META), theme::TEXT_BODY);
                let r = Rect::from_min_size(at + Vec2::new(12.0, 12.0), galley.size() + Vec2::splat(8.0));
                painter.rect_filled(r, egui::CornerRadius::same(theme::R_ROW), theme::BG_FIELD);
                painter.galley(r.min + Vec2::splat(4.0), galley, theme::TEXT_BODY);
            }
            if resp.clicked()
                && let Some(at) = resp.interact_pointer_pos()
            {
                clicked = crate::model_preview::pick(&items, at, 10.0);
            }
        });
        if clicked.is_some() {
            self.selected = clicked;
        }
    }

    /// The painter silhouette (no renderer or no instance of the model in the scene).
    fn figure_painted(&mut self, ui: &mut egui::Ui, size: Vec2, rig: &RigImport, report: &BoneReport) {
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, egui::CornerRadius::same(theme::R_ROW), theme::BG_FIELD);
        let map = |p: (f32, f32)| Pos2::new(rect.center().x + p.0 * rect.height() * 0.5 * 0.85, rect.center().y - p.1 * rect.height() * 0.5 * 0.9);

        for shape in silhouette(self.view) {
            let color = tint(group_state(report, shape.group));
            match shape.kind {
                Shape2::Circle(c, r) => {
                    painter.circle_filled(map(c), r * rect.height() * 0.5, color);
                }
                Shape2::Seg(a, b, w) => thick_line(&painter, map(a), map(b), w * rect.height() * 0.5, color),
            }
        }

        let mut clicked = None;
        for (bone, p) in dots(self.view) {
            let c = map(p);
            let resp = ui.interact(Rect::from_center_size(c, Vec2::splat(18.0)), ui.id().with(("bone-dot", bone as usize)), Sense::click());
            draw_dot(&painter, c, bone, report.state(bone), rig.joint_name(bone).is_some(), self.selected == Some(bone), resp.hovered());
            let resp = resp.on_hover_text(match rig.joint_name(bone) {
                Some(j) => format!("{}: {j}", bone.name()),
                None => format!("{} (unassigned{})", bone.name(), if bone.is_required() { ", required" } else { "" }),
            });
            if resp.clicked() {
                clicked = Some(bone);
            }
        }
        if clicked.is_some() {
            self.selected = clicked;
        }
    }

    /// Bone rows grouped by body part.
    fn list_column(&mut self, ui: &mut egui::Ui, skel: &Skeleton, rig: &mut RigImport, report: &BoneReport, out: &mut RigOut) {
        egui::ScrollArea::vertical().max_height(430.0).show(ui, |ui| {
            for group in self.view.groups() {
                let state = group_state(report, *group);
                egui::CollapsingHeader::new(egui::RichText::new(group.label()).color(match state {
                    GroupState::Red => theme::STATUS_ERROR,
                    GroupState::Green => theme::STATUS_OK,
                    GroupState::Grey => theme::TEXT_BODY,
                }))
                .id_salt(("rig-group", group.label()))
                .default_open(true)
                .show(ui, |ui| {
                    for bone in group.bones() {
                        if matches!(self.view, FigureView::LeftHand | FigureView::RightHand) && matches!(bone, HumanBone::LeftHand | HumanBone::RightHand) {
                            continue;
                        }
                        self.bone_row(ui, skel, rig, report, bone, out);
                    }
                });
            }
        });
    }

    fn bone_row(&mut self, ui: &mut egui::Ui, skel: &Skeleton, rig: &mut RigImport, report: &BoneReport, bone: HumanBone, out: &mut RigOut) {
        let state = report.state(bone);
        ui.horizontal(|ui| {
            let (dot, resp) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::click());
            let color = match state {
                BoneState::Valid => theme::STATUS_OK,
                BoneState::Invalid => theme::STATUS_ERROR,
                BoneState::Unassigned => theme::TEXT_DIM,
            };
            if rig.joint_name(bone).is_some() {
                ui.painter().circle_filled(dot.center(), 4.0, color);
            }
            if bone.is_required() {
                ui.painter().circle_stroke(dot.center(), 5.5, Stroke::new(1.2_f32, color));
            } else {
                dotted_circle(ui.painter(), dot.center(), 5.5, color);
            }
            if resp.clicked() {
                self.selected = Some(bone);
            }
            let label = bone.name().trim_start_matches("Left").trim_start_matches("Right");
            let name_color = if self.selected == Some(bone) { theme::STATUS_INFO } else { theme::TEXT_MUTED };
            ui.add_sized([110.0, 20.0], egui::Label::new(egui::RichText::new(label).size(theme::SIZE_META).color(name_color)).truncate());
            let current = rig.joint_name(bone).unwrap_or("None").to_string();
            let btn = widgets::secondary_button(ui, None, &current);
            // A Hierarchy object dropped on the row assigns the joint of that name.
            if let Some(joint) = crate::rig_extras::dropped_joint(ui, skel, &btn) {
                assign_bone(&mut rig.human, bone, Some(&joint));
                rig.auto_generated = false;
                out.changed = true;
                self.selected = Some(bone);
            }
            egui::Popup::menu(&btn).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
                widgets::menu_frame(ui);
                ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Filter").desired_width(f32::INFINITY));
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    if ui.selectable_label(rig.joint_name(bone).is_none(), "None").clicked() {
                        assign_bone(&mut rig.human, bone, None);
                        rig.auto_generated = false;
                        out.changed = true;
                        ui.close();
                    }
                    let needle = self.filter.to_lowercase();
                    for j in &skel.joints {
                        if !needle.is_empty() && !j.name.to_lowercase().contains(&needle) {
                            continue;
                        }
                        if ui.selectable_label(rig.joint_name(bone) == Some(j.name.as_str()), &j.name).clicked() {
                            assign_bone(&mut rig.human, bone, Some(&j.name));
                            rig.auto_generated = false;
                            out.changed = true;
                            self.selected = Some(bone);
                            ui.close();
                        }
                    }
                });
            });
        });
        for m in report.messages(bone) {
            ui.add(egui::Label::new(egui::RichText::new(m).size(theme::SIZE_TINY).color(theme::STATUS_ERROR)).wrap());
        }
    }
}

/// `.humanmap` and `.ht` files in `dir`, sorted.
fn mapping_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "humanmap" || e == "ht"))
        .collect();
    v.sort();
    v
}

fn load_mapping(path: &Path) -> anyhow::Result<HumanMap> {
    if path.extension().is_some_and(|e| e == "ht") { load_unity_ht(path) } else { load_humanmap(path) }
}

fn tint(s: GroupState) -> Color32 {
    match s {
        GroupState::Green => theme::with_alpha(theme::STATUS_OK, 0.35),
        GroupState::Red => theme::with_alpha(theme::STATUS_ERROR, 0.35),
        GroupState::Grey => theme::with_alpha(theme::TEXT_DIM, 0.25),
    }
}

/// One bone dot: filled when assigned, solid ring for required bones and dotted for
/// optional ones, coloured by validation state (the selected bone light blue).
fn draw_dot(painter: &egui::Painter, c: Pos2, bone: HumanBone, state: BoneState, assigned: bool, selected: bool, hovered: bool) {
    let color = if selected {
        theme::STATUS_INFO
    } else {
        match state {
            BoneState::Valid => theme::STATUS_OK,
            BoneState::Invalid => theme::STATUS_ERROR,
            BoneState::Unassigned => theme::TEXT_DIM,
        }
    };
    let r = 6.0;
    if assigned {
        painter.circle_filled(c, r - 1.0, color);
    }
    if bone.is_required() {
        painter.circle_stroke(c, r, Stroke::new(1.5_f32, color));
    } else {
        dotted_circle(painter, c, r, color);
    }
    if hovered {
        painter.circle_stroke(c, r + 3.0, Stroke::new(1.0_f32, theme::TEXT_MUTED));
    }
}

/// Local transforms of `skel` in the rig's reference pose (the file pose for
/// joints it does not list).
fn reference_locals_of(skel: &Skeleton, rig: &RigImport) -> Vec<glam::Mat4> {
    (0..skel.joints.len())
        .map(|j| {
            let p = crate::rig_extras::joint_pose(rig, skel, j);
            glam::Mat4::from_scale_rotation_translation(glam::Vec3::from(p.scale), glam::Quat::from_array(p.rotation), glam::Vec3::from(p.translation))
        })
        .collect()
}

/// Model-space points a view frames: its mapped bones (for the head a crown
/// estimate above the head bone, for a hand the wrist too).
fn focus_points(view: FigureView, skel: &Skeleton, rig: &RigImport, positions: &[glam::Vec3]) -> Vec<glam::Vec3> {
    let at = |b: HumanBone| rig.joint_name(b).and_then(|n| skel.joints.iter().position(|j| j.name == n)).map(|j| positions[j]);
    let mut pts: Vec<glam::Vec3> = view.groups().iter().flat_map(|g| g.bones()).filter_map(at).collect();
    match view {
        FigureView::Head => {
            if let (Some(head), Some(neck)) = (at(HumanBone::Head), at(HumanBone::Neck)) {
                pts.push(head + (head - neck) * 1.6);
            }
        }
        FigureView::LeftHand => pts.extend(at(HumanBone::LeftHand)),
        FigureView::RightHand => pts.extend(at(HumanBone::RightHand)),
        FigureView::Body => {}
    }
    pts
}

/// The bounding box of `pts` in centimetres, rounded to 5 cm steps: changes only when the bones move a lot.
fn bounds_key(pts: &[glam::Vec3]) -> [i32; 6] {
    let Some(first) = pts.first().copied() else { return [0; 6] };
    let (lo, hi) = pts.iter().fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let q = |v: f32| (v * 20.0).round() as i32;
    [q(lo.x), q(lo.y), q(lo.z), q(hi.x), q(hi.y), q(hi.z)]
}

fn legend_dot(ui: &mut egui::Ui, required: bool, label: &str) {
    let (r, _) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::hover());
    if required {
        ui.painter().circle_stroke(r.center(), 5.0, Stroke::new(1.2_f32, theme::TEXT_DIM));
    } else {
        dotted_circle(ui.painter(), r.center(), 5.0, theme::TEXT_DIM);
    }
    ui.label(egui::RichText::new(label).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

fn dotted_circle(painter: &egui::Painter, c: Pos2, r: f32, color: Color32) {
    const N: usize = 10;
    for i in 0..N {
        let a0 = (i as f32 / N as f32) * std::f32::consts::TAU;
        let a1 = a0 + std::f32::consts::TAU / N as f32 * 0.5;
        painter.line_segment(
            [c + Vec2::angled(a0) * r, c + Vec2::angled(a1) * r],
            Stroke::new(1.5_f32, color),
        );
    }
}

fn thick_line(painter: &egui::Painter, a: Pos2, b: Pos2, width: f32, color: Color32) {
    painter.line_segment([a, b], Stroke::new(width, color));
    painter.circle_filled(a, width * 0.5, color);
    painter.circle_filled(b, width * 0.5, color);
}

// ------------------------------------------------------------------- figure data

enum Shape2 {
    Circle((f32, f32), f32),
    Seg((f32, f32), (f32, f32), f32),
}

struct Piece {
    group: Group,
    kind: Shape2,
}

fn mirror(p: (f32, f32)) -> (f32, f32) {
    (-p.0, p.1)
}

/// Screen layout: the avatar faces the viewer, so its left side is on the right.
fn silhouette(view: FigureView) -> Vec<Piece> {
    use Shape2::*;
    let p = |group, kind| Piece { group, kind };
    match view {
        FigureView::Body => {
            let mut v = vec![
                p(Group::Head, Circle((0.0, 0.84), 0.09)),
                p(Group::Head, Seg((0.0, 0.7), (0.0, 0.62), 0.05)),
                p(Group::Body, Seg((0.0, 0.55), (0.0, 0.0), 0.2)),
                p(Group::LeftHand, Circle((0.8, 0.5), 0.05)),
                p(Group::RightHand, Circle((-0.8, 0.5), 0.05)),
            ];
            for (side, left) in [(1.0f32, true), (-1.0, false)] {
                let arm = if left { Group::LeftArm } else { Group::RightArm };
                let leg = if left { Group::LeftLeg } else { Group::RightLeg };
                v.push(p(arm, Seg((0.12 * side, 0.52), (0.74 * side, 0.5), 0.05)));
                v.push(p(leg, Seg((0.1 * side, -0.05), (0.12 * side, -0.8), 0.07)));
                v.push(p(leg, Seg((0.12 * side, -0.86), (0.15 * side, -0.93), 0.06)));
            }
            v
        }
        FigureView::Head => vec![p(Group::Head, Circle((0.0, 0.0), 0.5)), p(Group::Head, Seg((0.0, -0.45), (0.0, -0.85), 0.2))],
        FigureView::LeftHand | FigureView::RightHand => {
            let g = if view == FigureView::LeftHand { Group::LeftHand } else { Group::RightHand };
            let m = |q: (f32, f32)| if view == FigureView::RightHand { mirror(q) } else { q };
            let mut v = vec![p(g, Seg(m((0.0, -0.75)), m((0.0, -0.3)), 0.5))];
            for (x, top) in [(-0.3, 0.45), (-0.05, 0.6), (0.2, 0.48), (0.43, 0.25)] {
                v.push(p(g, Seg(m((x, -0.1)), m((x, top)), 0.08)));
            }
            v.push(p(g, Seg(m((-0.5, -0.4)), m((-0.85, 0.1)), 0.09)));
            v
        }
    }
}

/// Bone dots of a view in figure space (x right, y up, both about -1..1).
fn dots(view: FigureView) -> Vec<(HumanBone, (f32, f32))> {
    use HumanBone as B;
    match view {
        FigureView::Body => {
            let mut v = vec![
                (B::Hips, (0.0, 0.0)),
                (B::Spine, (0.0, 0.15)),
                (B::Chest, (0.0, 0.3)),
                (B::UpperChest, (0.0, 0.43)),
            ];
            let side = |left: bool| {
                let s = if left { 1.0 } else { -1.0 };
                let b = |l: HumanBone, r: HumanBone| if left { l } else { r };
                vec![
                    (b(B::LeftShoulder, B::RightShoulder), (0.14 * s, 0.52)),
                    (b(B::LeftUpperArm, B::RightUpperArm), (0.3 * s, 0.5)),
                    (b(B::LeftLowerArm, B::RightLowerArm), (0.52 * s, 0.5)),
                    (b(B::LeftHand, B::RightHand), (0.74 * s, 0.5)),
                    (b(B::LeftUpperLeg, B::RightUpperLeg), (0.1 * s, -0.05)),
                    (b(B::LeftLowerLeg, B::RightLowerLeg), (0.11 * s, -0.45)),
                    (b(B::LeftFoot, B::RightFoot), (0.12 * s, -0.8)),
                    (b(B::LeftToes, B::RightToes), (0.15 * s, -0.93)),
                ]
            };
            v.extend(side(true));
            v.extend(side(false));
            v
        }
        FigureView::Head => vec![(B::Neck, (0.0, -0.75)), (B::Head, (0.0, -0.1)), (B::LeftEye, (0.22, 0.12)), (B::RightEye, (-0.22, 0.12)), (B::Jaw, (0.0, -0.33))],
        FigureView::LeftHand | FigureView::RightHand => {
            let left = view == FigureView::LeftHand;
            let m = |q: (f32, f32)| if left { q } else { mirror(q) };
            // thumb, index, middle, ring, little: three dots each
            let cols: [[(f32, f32); 3]; 5] = [
                [(-0.5, -0.4), (-0.67, -0.15), (-0.82, 0.08)],
                [(-0.3, -0.05), (-0.3, 0.2), (-0.3, 0.42)],
                [(-0.05, -0.0), (-0.05, 0.3), (-0.05, 0.57)],
                [(0.2, -0.05), (0.2, 0.22), (0.2, 0.45)],
                [(0.43, -0.12), (0.43, 0.05), (0.43, 0.22)],
            ];
            let base = if left { 24 } else { 39 };
            let mut v = Vec::new();
            for (f, joints) in cols.iter().enumerate() {
                for (k, q) in joints.iter().enumerate() {
                    v.push((ALL[base + f * 3 + k], m(*q)));
                }
            }
            v
        }
    }
}

// ------------------------------------------------------------------- avatar descriptor

/// The model under object `root` (itself or a descendant) that carries a
/// skeleton: its project-relative path and skeleton.
pub fn avatar_model(scene: &zestengine_engine::scene::LoadedScene, root: usize) -> Option<(String, &Skeleton)> {
    for (i, o) in scene.objects.iter().enumerate() {
        let under = {
            let mut cur = Some(i);
            let mut hit = false;
            let mut guard = 0;
            while let Some(c) = cur {
                if c == root {
                    hit = true;
                    break;
                }
                cur = scene.objects[c].parent;
                guard += 1;
                if guard > 512 {
                    break;
                }
            }
            hit
        };
        if !under {
            continue;
        }
        if let zestengine_assets::ObjectSource::Model { path, .. } = &o.source
            && let Some(skel) = scene.object_skeleton(i)
        {
            return Some((path.clone(), skel));
        }
    }
    None
}

/// What the Avatar Descriptor on object `root` shows about its model's rig.
pub fn avatar_rig_info(scene: &zestengine_engine::scene::LoadedScene, root: usize) -> Option<crate::avatar_rig::AvatarRigInfo> {
    let (model, skel) = avatar_model(scene, root)?;
    let rig = skel.rig.clone().unwrap_or_default();
    Some(crate::avatar_rig::AvatarRigInfo {
        model,
        kind: match rig.animation_type {
            AnimationType::Humanoid => "Humanoid",
            AnimationType::Generic => "Generic",
            AnimationType::None => "No rig",
        }
        .into(),
        required_mapped: ALL.iter().filter(|b| b.is_required() && rig.joint_name(**b).is_some()).count(),
        human: rig.human.iter().map(|(b, j)| (b.name().to_string(), j.clone())).collect(),
        auto_generated: rig.auto_generated,
    })
}

// ------------------------------------------------------------------- overlay

/// Where the overlay's joint transforms come from, best first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlayPose<'a> {
    /// The pose the Animator / skinner produced this frame.
    Live(&'a [glam::Mat4]),
    /// The Muscles & Settings sliders.
    Preview(&'a [glam::Mat4]),
    /// The skeleton's rest (file) pose.
    Rest,
}

impl<'a> OverlayPose<'a> {
    /// Live pose over muscle preview over rest.
    pub fn select(live: Option<&'a [glam::Mat4]>, preview: Option<&'a [glam::Mat4]>) -> Self {
        match (live, preview) {
            (Some(l), _) => Self::Live(l),
            (None, Some(p)) => Self::Preview(p),
            _ => Self::Rest,
        }
    }

    /// Local transforms of `skel` in this pose.
    pub fn locals(&self, skel: &Skeleton) -> Vec<glam::Mat4> {
        match self {
            Self::Live(l) | Self::Preview(l) => l.to_vec(),
            Self::Rest => skel.rest_locals(),
        }
    }

    /// The T-pose tolerance leaders and note only mean something while Configure
    /// is open and nothing animates the character.
    pub fn show_t_pose_leaders(&self, configure: bool) -> bool {
        configure && !matches!(self, Self::Live(_))
    }
}

/// Scene positions of every joint of `skel` in `locals` under the object's
/// `world` matrix (applied exactly once).
pub fn overlay_joint_positions(skel: &Skeleton, locals: &[glam::Mat4], world: glam::Mat4) -> Vec<glam::Vec3> {
    skel.world_matrices(locals).iter().map(|m| world.transform_point3(m.w_axis.truncate())).collect()
}

/// Draw `skel` over the viewport: bones coloured by mapping state (human valid
/// green, human invalid red, other grey, the selected bone light blue), bones out
/// of their T-pose tolerance named with a leader label, and a red
/// "Character is not in T pose" note when any is.
///
/// `world` places the skeleton's root in the scene, `view_proj` is the editor
/// camera, `rect` the viewport rect. `pose` picks the joint transforms; the
/// T-pose leaders show only when `configure` is open and the pose is not live.
#[allow(clippy::too_many_arguments)]
pub fn draw_overlay(
    painter: &egui::Painter,
    view_proj: glam::Mat4,
    rect: Rect,
    world: glam::Mat4,
    skel: &Skeleton,
    rig: &RigImport,
    report: &BoneReport,
    pose_bad: &[HumanBone],
    selected: Option<HumanBone>,
    pose: OverlayPose,
    configure: bool,
) {
    const W_EPS: f32 = 0.001;
    let leaders = pose.show_t_pose_leaders(configure);
    let pose_bad: &[HumanBone] = if leaders { pose_bad } else { &[] };
    let positions = overlay_joint_positions(skel, &pose.locals(skel), world);
    let project = |p: glam::Vec3| -> Option<Pos2> {
        let c = view_proj * p.extend(1.0);
        if c.w <= W_EPS {
            return None;
        }
        let ndc = c.truncate() / c.w;
        Some(Pos2::new(rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(), rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height()))
    };
    let joint_bone = |j: usize| rig.human.iter().find(|(_, n)| *n == skel.joints[j].name).map(|(b, _)| *b);
    let color_of = |j: usize| -> Color32 {
        match joint_bone(j) {
            Some(b) if Some(b) == selected => theme::STATUS_INFO,
            Some(b) if pose_bad.contains(&b) || report.state(b) == BoneState::Invalid => theme::STATUS_ERROR,
            Some(_) => theme::with_alpha(theme::STATUS_OK, 0.8),
            None => theme::with_alpha(theme::TEXT_DIM, 0.6),
        }
    };
    let pos: Vec<Option<Pos2>> = positions.iter().map(|p| project(*p)).collect();
    for (j, joint) in skel.joints.iter().enumerate() {
        let (Some(a), Some(p)) = (pos[j], joint.parent.and_then(|p| pos[p])) else { continue };
        if !rect.expand(40.0).contains(a) && !rect.expand(40.0).contains(p) {
            continue;
        }
        painter.line_segment([p, a], Stroke::new(2.0_f32, color_of(j)));
    }
    let mut labelled = 0;
    for (j, _) in skel.joints.iter().enumerate() {
        let Some(a) = pos[j] else { continue };
        if !rect.contains(a) {
            continue;
        }
        let col = color_of(j);
        let mapped = joint_bone(j);
        painter.circle_filled(a, if mapped.is_some() { 3.5 } else { 2.0 }, col);
        if let Some(b) = mapped
            && pose_bad.contains(&b)
            && labelled < 24
        {
            labelled += 1;
            let side = if a.x < rect.center().x { rect.left() + 8.0 } else { rect.right() - 8.0 };
            let align = if a.x < rect.center().x { egui::Align2::LEFT_CENTER } else { egui::Align2::RIGHT_CENTER };
            let end = Pos2::new(side, a.y);
            painter.line_segment([a, end], Stroke::new(1.0_f32, theme::with_alpha(theme::STATUS_ERROR, 0.6)));
            painter.text(end, align, b.name(), theme::mono(theme::SIZE_META), theme::STATUS_ERROR);
        }
    }
    if !pose_bad.is_empty() {
        painter.text(rect.left_top() + Vec2::new(12.0, 12.0), egui::Align2::LEFT_TOP, "Character is not in T pose", theme::semibold(20.0), theme::STATUS_ERROR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_view_dot_is_unique_and_hand_views_cover_all_fingers() {
        let mut seen = std::collections::HashSet::new();
        for view in [FigureView::Body, FigureView::Head, FigureView::LeftHand, FigureView::RightHand] {
            for (b, _) in dots(view) {
                assert!(seen.insert(b), "{b:?} drawn twice");
            }
        }
        // 15 left + 15 right finger bones, plus the body-part bones (no jaw twice).
        for b in &ALL[24..54] {
            assert!(seen.contains(b), "{b:?} has no dot");
        }
        for b in ALL.iter().filter(|b| b.is_required()) {
            assert!(seen.contains(b), "required {b:?} has no dot");
        }
        assert_eq!(seen.len(), 55, "all 55 bones have a dot");
    }

    #[test]
    fn view_groups_list_exactly_the_bones_with_dots() {
        for view in [FigureView::Body, FigureView::Head, FigureView::LeftHand, FigureView::RightHand] {
            let from_groups: std::collections::HashSet<HumanBone> = view
                .groups()
                .iter()
                .flat_map(|g| g.bones())
                .filter(|b| !(matches!(view, FigureView::LeftHand | FigureView::RightHand) && matches!(b, HumanBone::LeftHand | HumanBone::RightHand)))
                .collect();
            let from_dots: std::collections::HashSet<HumanBone> = dots(view).into_iter().map(|(b, _)| b).collect();
            // The Body view's hand dots live in the arm groups, so compare as sets.
            assert_eq!(from_groups, from_dots, "{view:?}");
        }
    }

    /// The Configure window's height after `frames` frames, switching tabs as `tab_at` says.
    fn configure_heights(frames: usize, tab_at: impl Fn(usize) -> bool) -> Vec<f32> {
        configure_run(frames, tab_at, false).0
    }

    /// Heights per frame, and whether the 3D figure asked for a render and a pose on the last frame.
    fn configure_run(frames: usize, tab_at: impl Fn(usize) -> bool, avatar: bool) -> (Vec<f32>, bool) {
        let skel = crate::test_support::skeleton();
        let mut rig = skel.rig.clone().unwrap();
        let path = PathBuf::from("hero.fbx");
        let mut ui = RigUi { configure: true, ..RigUi::default() };
        ui.loaded = Some(Loaded { path: path.clone(), skel: Some(skel), skeletons: 1, error: None });
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        theme::install_fonts(&mut fonts);
        ctx.set_fonts(fonts);
        theme::apply(&ctx);
        let mut heights = Vec::new();
        let mut asked = false;
        for f in 0..frames {
            ui.muscles.active = tab_at(f);
            ui.set_avatar(avatar.then_some(glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0))));
            let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 1000.0))), ..Default::default() };
            let _ = ctx.run(input, |ctx| {
                ui.configure_window(ctx, &path, &mut rig, true);
            });
            let rect = ctx.memory(|m| m.area_rect(egui::Id::new("Avatar Configuration"))).expect("window area");
            heights.push(rect.height());
            asked = ui.preview.take_request().is_some() && ui.take_pose_override().is_some();
        }
        (heights, asked)
    }

    #[test]
    fn the_3d_figure_is_height_stable_and_asks_for_a_render_and_a_pose() {
        let (h, asked) = configure_run(31, |f| f % 2 == 1, true);
        assert!(asked, "an instance in the scene: the figure renders it in the reference pose");
        // The two tabs differ in height; each must be stable across visits.
        for parity in 0..2 {
            let same: Vec<f32> = h.iter().copied().enumerate().filter(|(i, _)| *i >= 6 && i % 2 == parity).map(|(_, v)| v).collect();
            assert!(same.iter().all(|v| (v - same[0]).abs() < 0.5), "height keeps changing: {h:?}");
        }
        let (_, asked) = configure_run(5, |_| false, false);
        assert!(!asked, "no instance: the painter figure, no render, no pose override");
    }

    #[test]
    fn every_view_frames_its_bones_inside_the_figure() {
        let skel = crate::test_support::skeleton();
        let rig = skel.rig.clone().unwrap();
        let positions = overlay_joint_positions(&skel, &reference_locals_of(&skel, &rig), glam::Mat4::IDENTITY);
        let rect = Rect::from_min_size(Pos2::new(5.0, 5.0), Vec2::new(300.0, 400.0));
        for (view, pitch) in [(FigureView::Body, 0.1), (FigureView::Head, 0.05), (FigureView::LeftHand, 1.4), (FigureView::RightHand, 1.4)] {
            let pts = focus_points(view, &skel, &rig, &positions);
            assert!(!pts.is_empty(), "{view:?} has bones to frame");
            let orbit = crate::model_preview::frame_points(&pts, 0.0, pitch, rect.width() / rect.height()).unwrap();
            let (v, p, _) = orbit.camera(glam::Mat4::IDENTITY, rect.width() / rect.height());
            for pt in &pts {
                let (s, _) = crate::model_preview::project(p * v, rect, *pt).unwrap();
                assert!(rect.contains(s), "{view:?}: {pt:?} at {s:?}");
            }
        }
        // The key only moves for big changes.
        let head = focus_points(FigureView::Head, &skel, &rig, &positions);
        assert_eq!(bounds_key(&head), bounds_key(&head.iter().map(|p| *p + glam::Vec3::splat(0.001)).collect::<Vec<_>>()));
    }

    #[test]
    fn the_reference_pose_locals_follow_the_stored_pose() {
        let skel = crate::test_support::skeleton();
        let mut rig = skel.rig.clone().unwrap();
        let plain = reference_locals_of(&skel, &rig);
        assert!(plain.iter().zip(skel.rest_locals()).all(|(a, b)| a.abs_diff_eq(b, 1e-5)), "no stored pose = file pose");
        assert!(crate::rig_extras::apply_biped_pose(&mut rig, &skel));
        let forced = reference_locals_of(&skel, &rig);
        let mut with = skel.clone();
        with.rig = Some(rig);
        assert!(forced.iter().zip(zestengine_assets::ref_pose::reference_locals(&with)).all(|(a, b)| a.abs_diff_eq(b, 1e-5)));
    }

    #[test]
    fn the_muscles_tab_does_not_grow_the_configure_window() {
        let h = configure_heights(30, |_| true);
        assert!(h[5..].iter().all(|v| (v - h[5]).abs() < 0.5), "height keeps changing: {h:?}");
    }

    #[test]
    fn the_mapping_tab_height_is_stable() {
        let h = configure_heights(30, |_| false);
        assert!(h[5..].iter().all(|v| (v - h[5]).abs() < 0.5), "height keeps changing: {h:?}");
    }

    #[test]
    fn switching_tabs_repeatedly_does_not_grow_the_window() {
        let h = configure_heights(60, |f| (f / 7) % 2 == 1);
        let max_first = h[..14].iter().cloned().fold(0.0, f32::max);
        let max_last = h[30..].iter().cloned().fold(0.0, f32::max);
        assert!(max_last <= max_first + 0.5, "grew from {max_first} to {max_last}: {h:?}");
    }

    #[test]
    fn the_overlay_pose_source_is_live_then_preview_then_rest() {
        let live = [glam::Mat4::IDENTITY];
        let prev = [glam::Mat4::from_translation(glam::Vec3::X)];
        assert_eq!(OverlayPose::select(Some(&live), Some(&prev)), OverlayPose::Live(&live));
        assert_eq!(OverlayPose::select(None, Some(&prev)), OverlayPose::Preview(&prev));
        assert_eq!(OverlayPose::select(None, None), OverlayPose::Rest);
    }

    #[test]
    fn t_pose_leaders_need_configure_and_no_live_pose() {
        let l = [glam::Mat4::IDENTITY];
        assert!(OverlayPose::Rest.show_t_pose_leaders(true));
        assert!(OverlayPose::Preview(&l).show_t_pose_leaders(true));
        assert!(!OverlayPose::Rest.show_t_pose_leaders(false), "Show skeleton alone draws bones only");
        assert!(!OverlayPose::Live(&l).show_t_pose_leaders(true), "no leaders on an animated character");
    }

    #[test]
    fn overlay_joints_equal_the_posed_joints_under_the_world_transform_once() {
        let skel = crate::test_support::skeleton();
        let mut locals = skel.rest_locals();
        // An "animation": every joint turned a little about Z.
        for m in &mut locals {
            *m *= glam::Mat4::from_rotation_z(0.3);
        }
        let world = glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(2.0), glam::Quat::from_rotation_y(1.0), glam::Vec3::new(5.0, 0.0, -3.0));
        let pose = OverlayPose::select(Some(&locals), None);
        let got = overlay_joint_positions(&skel, &pose.locals(&skel), world);
        let model = skel.world_matrices(&locals);
        for (g, m) in got.iter().zip(&model) {
            assert!((*g - world.transform_point3(m.w_axis.truncate())).length() < 1e-5);
        }
        let rest = overlay_joint_positions(&skel, &OverlayPose::Rest.locals(&skel), world);
        assert!(got.iter().zip(&rest).skip(1).any(|(a, b)| (*a - *b).length() > 0.01), "the animated pose differs from rest");
    }
}
