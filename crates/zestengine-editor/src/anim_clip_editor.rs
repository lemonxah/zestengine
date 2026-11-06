//! The Animation tab: author `.anim` clips on a timeline. Tracks animate the
//! target object (the selection's nearest Animator, or the selection) and its
//! children by relative path, or bones of a skinned model. Keys are diamonds
//! (select, drag in time, delete, copy / paste, per-key interpolation); the
//! playhead scrubs and plays a preview in edit mode, and Record turns gizmo /
//! inspector transform edits on the hierarchy into keys at the playhead.

use std::collections::HashMap;
use std::path::Path;

use zestengine_assets::{AnimClip, AnimTrack, ClipEvent, KeyInterp, Keyframe, TrackProperty};
use zestengine_core::{Animator, LightComponent, ObjectId};
use crate::egui_phosphor::regular as ph;
use crate::icons::Icon;
use crate::{theme, widgets};
use egui::{CornerRadius, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use glam::{EulerRot, Quat, Vec3};

use crate::anim_controller_editor::{AnimControllerTab, create_asset, scan_files};
use zestengine_engine::anim_graph::{ClipRef, Pose, Rig, apply_pose_to_objects, object_rig};
use zestengine_engine::anim_props::{self, MaterialProp};
use zestengine_engine::scene::LoadedScene;

mod muscle_tracks;

const ROW_H: f32 = 24.0;
const RULER_H: f32 = 24.0;

/// An object property on a scene object: (object, kind, material slot, name).
type PropKey = (ObjectId, TrackProperty, u8, String);

/// A key drag: where it started (seconds) and the (track, key, time) it moves.
type KeyDrag = (f32, Vec<(usize, usize, f32)>);

pub struct AnimClipTab {
    path: Option<String>,
    clip: AnimClip,
    /// Undo history of the open document.
    history: crate::undo::DocHistory<AnimClip>,
    dirty: bool,
    target: Option<ObjectId>,
    playhead: f32,
    playing: bool,
    preview: bool,
    recording: bool,
    /// Transforms (and light intensity) before preview started.
    originals: HashMap<ObjectId, (Vec3, Quat, Vec3, f32)>,
    /// What preview last wrote, to spot edits made while recording.
    applied: HashMap<ObjectId, (Vec3, Quat, Vec3)>,
    /// Object properties (active, component enabled, blendshapes, materials)
    /// before preview wrote them, and as of the last preview write.
    prop_originals: HashMap<PropKey, [f32; 4]>,
    prop_applied: HashMap<PropKey, [f32; 4]>,
    /// The scene selection, whose properties recording watches.
    selected_object: Option<ObjectId>,
    selected: Vec<(usize, usize)>,
    clipboard: Vec<(usize, Keyframe)>,
    pps: f32,
    scroll: f32,
    vscroll: f32,
    drag: Option<KeyDrag>,
    scrubbing: bool,
    /// Value lane shown in the curve editor (0..`TrackProperty::lanes`).
    curve_lane: usize,
    /// Asked the editor to show this tab.
    pub open_request: bool,
    files: Vec<String>,
}

impl Default for AnimClipTab {
    fn default() -> Self {
        Self {
            path: None,
            clip: AnimClip::default(),
            dirty: false,
            history: Default::default(),
            target: None,
            playhead: 0.0,
            playing: false,
            preview: false,
            recording: false,
            originals: HashMap::new(),
            applied: HashMap::new(),
            prop_originals: HashMap::new(),
            prop_applied: HashMap::new(),
            selected_object: None,
            selected: Vec::new(),
            clipboard: Vec::new(),
            pps: 200.0,
            scroll: 0.0,
            vscroll: 0.0,
            drag: None,
            scrubbing: false,
            curve_lane: 0,
            open_request: false,
            files: Vec::new(),
        }
    }
}

fn has_animator(scene: &LoadedScene, i: usize) -> bool {
    scene.objects[i].components.iter().any(|c| c.as_any().is::<Animator>())
}

fn is_descendant(scene: &LoadedScene, mut i: usize, root: usize) -> bool {
    for _ in 0..256 {
        if i == root {
            return true;
        }
        match scene.objects[i].parent {
            Some(p) => i = p,
            None => return false,
        }
    }
    false
}

fn light_intensity(scene: &LoadedScene, i: usize) -> Option<f32> {
    scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<LightComponent>()).map(|l| l.intensity)
}

fn quat_to_euler_deg(q: [f32; 4]) -> [f32; 3] {
    let (x, y, z) = Quat::from_array(q).normalize().to_euler(EulerRot::XYZ);
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

fn euler_deg_to_quat(e: [f32; 3]) -> [f32; 4] {
    Quat::from_euler(EulerRot::XYZ, e[0].to_radians(), e[1].to_radians(), e[2].to_radians()).to_array()
}

/// Short label for one value lane of a track property, as shown in the curve
/// editor's lane selector.
fn lane_name(property: TrackProperty, lane: usize) -> &'static str {
    match property {
        TrackProperty::Rotation | TrackProperty::RootRotation => match lane {
            0 => "X",
            1 => "Y",
            2 => "Z",
            _ => "W",
        },
        TrackProperty::LightIntensity => "I",
        TrackProperty::MaterialColor => match lane {
            0 => "R",
            1 => "G",
            2 => "B",
            _ => "A",
        },
        TrackProperty::MaterialOffset => {
            if lane == 0 {
                "U"
            } else {
                "V"
            }
        }
        TrackProperty::Active | TrackProperty::ComponentEnabled | TrackProperty::BlendShape | TrackProperty::MaterialFloat | TrackProperty::Muscle | TrackProperty::SpriteSwap => "V",
        _ => match lane {
            0 => "X",
            1 => "Y",
            _ => "Z",
        },
    }
}

/// Name of a track in the track list: the property, plus what it addresses
/// for component, blendshape and material tracks.
fn track_property_label(track: &AnimTrack) -> String {
    match track.property {
        p if p.is_muscle() => muscle_tracks::label(track),
        TrackProperty::ComponentEnabled => format!("{} / {}", track.property.label(), track.name),
        TrackProperty::BlendShape => format!("{} / {}", track.property.label(), track.name),
        p if p.uses_slot() => {
            let prop = MaterialProp::from_name(&track.name).map(MaterialProp::label).unwrap_or(track.name.as_str());
            format!("Material {} / {prop}", track.slot as usize + 1)
        }
        p => p.label().to_owned(),
    }
}

/// A recorded or read property value with the "no weight yet" NaN cleaned.
fn clean_value(v: [f32; 4]) -> [f32; 4] {
    v.map(|x| if x.is_nan() { 0.0 } else { x })
}

/// Key an object property at `time`, adding its track when the clip lacks one.
fn key_prop(clip: &mut AnimClip, path: &str, kind: TrackProperty, name: &str, slot: u8, time: f32, value: [f32; 4]) {
    let k = match clip.prop_track_index(path, kind, name, slot) {
        Some(k) => k,
        None => {
            clip.tracks.push(AnimTrack::prop(path, kind, name, slot));
            clip.tracks.len() - 1
        }
    };
    clip.tracks[k].set_key(time, value);
    clip.length = clip.length.max(time);
}

/// Rig of the target hierarchy plus the scene index of each node.
struct Target {
    index: usize,
    rig: Rig,
    nodes: Vec<usize>,
    bones: Vec<(String, zestengine_engine::anim_graph::Trs)>,
}

impl AnimClipTab {
    pub fn open(&mut self, root: &Path, rel: &str) {
        if self.path.as_deref() == Some(rel) {
            return;
        }
        if self.dirty {
            self.save(root);
        }
        match zestengine_assets::load_anim_clip(root.join(rel)) {
            Ok(c) => {
                self.clip = c;
                self.path = Some(rel.to_owned());
                self.dirty = false;
                self.selected.clear();
                self.playhead = 0.0;
                self.playing = false;
            }
            Err(e) => tracing::error!("opening animation clip {rel}: {e:#}"),
        }
    }

    fn save(&mut self, root: &Path) {
        let Some(path) = &self.path else {
            return;
        };
        match zestengine_assets::save_anim_clip(root.join(path), &self.clip) {
            Ok(()) => self.dirty = false,
            Err(e) => tracing::error!("saving animation clip: {e:#}"),
        }
    }

    fn changed(&mut self, scene: &mut LoadedScene) {
        self.dirty = true;
        if let Some(path) = &self.path {
            scene.anim_assets.set_clip(path, self.clip.clone());
        }
    }

    fn snap(&self, t: f32, free: bool) -> f32 {
        let fps = self.clip.frame_rate.max(1.0);
        let t = t.max(0.0);
        if free { t } else { (t * fps).round() / fps }
    }

    fn end_preview(&mut self, scene: &mut LoadedScene) {
        let ids = scene.id_index();
        for (id, (t, r, s, f)) in self.originals.drain() {
            if let Some(&i) = ids.get(&id) {
                let o = &mut scene.objects[i];
                o.translation = t;
                o.rotation = r;
                o.scale = s;
                if let Some(l) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<LightComponent>()) {
                    l.intensity = f;
                }
            }
        }
        for ((id, kind, slot, name), v) in self.prop_originals.drain() {
            if let Some(&i) = ids.get(&id) {
                anim_props::write_prop(scene, i, kind, slot, &name, v);
            }
        }
        self.applied.clear();
        self.prop_applied.clear();
        self.preview = false;
        self.recording = false;
        self.playing = false;
    }

    fn resolve_target(&mut self, scene: &mut LoadedScene, selected: Option<usize>) -> Option<Target> {
        let current = self.target.and_then(|id| scene.index_of(id));
        if let Some(s) = selected {
            let keep = current.is_some_and(|c| is_descendant(scene, s, c));
            if !keep {
                let mut pick = s;
                let mut cur = Some(s);
                while let Some(c) = cur {
                    if has_animator(scene, c) {
                        pick = c;
                        break;
                    }
                    cur = scene.objects[c].parent;
                }
                let new = scene.objects[pick].id;
                if self.target != Some(new) {
                    if self.preview {
                        self.end_preview(scene);
                    }
                    self.target = Some(new);
                }
            }
        }
        let index = self.target.and_then(|id| scene.index_of(id))?;
        let (rig, nodes) = object_rig(scene, index);
        let bones = scene.skeleton_rest_pose(index);
        Some(Target { index, rig, nodes, bones })
    }

    /// Current value of a track's property on the target (object, bone rest
    /// or sampled curve).
    fn current_value(&self, scene: &LoadedScene, target: &Target, track: &AnimTrack) -> [f32; 4] {
        if let Some(n) = target.rig.index(&track.target) {
            let o = &scene.objects[target.nodes[n]];
            if track.property.is_object_prop() {
                let now = anim_props::read_prop(scene, target.nodes[n], track.property, track.slot, &track.name);
                return clean_value(now.unwrap_or([0.0; 4]));
            }
            return match track.property {
                TrackProperty::Translation => o.translation.extend(0.0).to_array(),
                TrackProperty::Rotation => o.rotation.to_array(),
                TrackProperty::Scale => o.scale.extend(0.0).to_array(),
                TrackProperty::LightIntensity => [light_intensity(scene, target.nodes[n]).unwrap_or(0.0), 0.0, 0.0, 0.0],
                _ => [0.0; 4],
            };
        }
        if !track.keys.is_empty() {
            return track.sample(self.clip.local_time(self.playhead, self.clip.looping));
        }
        let rest = target.bones.iter().find(|(n, _)| *n == track.target).map(|(_, t)| *t).unwrap_or_default();
        match track.property {
            TrackProperty::Translation => rest.t.extend(0.0).to_array(),
            TrackProperty::Rotation => rest.r.to_array(),
            TrackProperty::Scale => rest.s.extend(0.0).to_array(),
            _ => [0.0; 4],
        }
    }

    fn key_track(&mut self, track: usize, value: [f32; 4]) {
        let t = self.snap(self.playhead, false);
        self.clip.tracks[track].set_key(t, value);
        if t > self.clip.length {
            self.clip.length = t;
        }
    }

    fn add_track(&mut self, scene: &LoadedScene, target: &Target, path: &str, property: TrackProperty) {
        if self.clip.track_index(path, property).is_some() {
            return;
        }
        let mut track = AnimTrack::new(path, property);
        let v = self.current_value(scene, target, &track);
        track.set_key(self.snap(self.playhead, false), v);
        self.clip.tracks.push(track);
    }

    /// Add an object-property track (active, component, blendshape, material)
    /// keyed with the property's current value at the playhead.
    fn add_prop_track(&mut self, scene: &LoadedScene, target: &Target, path: &str, property: TrackProperty, name: &str, slot: u8) {
        if self.clip.prop_track_index(path, property, name, slot).is_some() {
            return;
        }
        let mut track = AnimTrack::prop(path, property, name, slot);
        let v = self.current_value(scene, target, &track);
        track.set_key(self.snap(self.playhead, false), v);
        self.clip.tracks.push(track);
    }

    /// The object-property entries of an object's "Add Property" menu.
    fn prop_menu(&mut self, ui: &mut egui::Ui, scene: &LoadedScene, target: &Target, node: usize, path: &str) -> bool {
        let obj = target.nodes[node];
        let mut added = false;
        widgets::menu_separator(ui);
        let active = TrackProperty::Active;
        let exists = self.clip.prop_track_index(path, active, "", 0).is_some();
        if ui.add_enabled(!exists, egui::Button::new(active.label())).clicked() {
            self.add_prop_track(scene, target, path, active, "", 0);
            added = true;
            ui.close();
        }
        let components = anim_props::component_names(scene, obj);
        if components.contains(&"Sprite Resolver") {
            let swap = TrackProperty::SpriteSwap;
            let exists = self.clip.prop_track_index(path, swap, "", 0).is_some();
            if ui.add_enabled(!exists, egui::Button::new(swap.label())).clicked() {
                self.add_prop_track(scene, target, path, swap, "", 0);
                added = true;
                ui.close();
            }
        }
        if !components.is_empty() {
            ui.menu_button(TrackProperty::ComponentEnabled.label(), |ui| {
                widgets::menu_frame(ui);
                for c in components {
                    let exists = self.clip.prop_track_index(path, TrackProperty::ComponentEnabled, c, 0).is_some();
                    if ui.add_enabled(!exists, egui::Button::new(c)).clicked() {
                        self.add_prop_track(scene, target, path, TrackProperty::ComponentEnabled, c, 0);
                        added = true;
                        ui.close();
                    }
                }
            });
        }
        let shapes = scene.morph_names(obj);
        if !shapes.is_empty() {
            ui.menu_button(TrackProperty::BlendShape.label(), |ui| {
                widgets::menu_frame(ui);
                egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                    for shape in &shapes {
                        let exists = self.clip.prop_track_index(path, TrackProperty::BlendShape, shape, 0).is_some();
                        if ui.add_enabled(!exists, egui::Button::new(shape.as_str())).clicked() {
                            self.add_prop_track(scene, target, path, TrackProperty::BlendShape, shape, 0);
                            added = true;
                            ui.close();
                        }
                    }
                });
            });
        }
        for slot in 0..anim_props::material_slots(scene, obj) {
            ui.menu_button(format!("Material {}", slot + 1), |ui| {
                widgets::menu_frame(ui);
                for p in MaterialProp::ALL {
                    let exists = self.clip.prop_track_index(path, p.track(), p.name(), slot as u8).is_some();
                    if ui.add_enabled(!exists, egui::Button::new(p.label())).clicked() {
                        self.add_prop_track(scene, target, path, p.track(), p.name(), slot as u8);
                        added = true;
                        ui.close();
                    }
                }
            });
        }
        added
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, selected: Option<usize>, root: &Path, playing_game: bool) {
        let (before, path) = (self.clip.clone(), self.path.clone());
        self.ui_inner(ui, scene, selected, root, playing_game);
        if self.path != path {
            self.history.clear();
        } else {
            self.history.observe(before, &self.clip, ui.input(|i| i.pointer.any_down()));
        }
    }

    /// Ctrl+Z / Ctrl+Shift+Z while this tab has focus.
    pub fn history_step(&mut self, undo: bool) {
        if self.history.step(&mut self.clip, undo) {
            self.dirty = true;
            // Key selections are indices into the document: drop them.
            self.selected.clear();
        }
    }

    fn ui_inner(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, selected: Option<usize>, root: &Path, playing_game: bool) {
        scene.anim_assets.set_root(root);
        let selected = selected.filter(|&i| i < scene.objects.len());
        self.selected_object = selected.map(|i| scene.objects[i].id);
        if playing_game && self.preview {
            self.end_preview(scene);
        }
        let target = self.resolve_target(scene, selected);
        let mut edited = false;

        egui::TopBottomPanel::top("zestengine-anim-clip-toolbar")
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(12, 6)))
            .show_inside(ui, |ui| edited |= self.toolbar(ui, scene, root, target.as_ref(), selected, playing_game));

        if self.path.is_none() {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_VIEWPORT)).show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.3);
                    ui.label(egui::RichText::new("No animation clip open").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                    ui.label(egui::RichText::new("Select a .anim in Files, or create a clip for the selected object.").color(theme::TEXT_MUTED));
                    ui.add_space(10.0);
                    if widgets::primary_button(ui, Some(Icon::Plus), "Create Clip").clicked() {
                        self.create_for_target(scene, root, target.as_ref());
                    }
                });
            });
            return;
        }

        egui::TopBottomPanel::bottom("zestengine-anim-clip-keys")
            .resizable(true)
            .default_height(300.0)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(12, 8)))
            .show_inside(ui, |ui| edited |= self.key_inspector(ui));

        egui::SidePanel::left("zestengine-anim-clip-tracks")
            .resizable(true)
            .default_width(250.0)
            .frame(egui::Frame::new().fill(theme::BG_PANEL))
            .show_inside(ui, |ui| edited |= self.track_list(ui, scene, target.as_ref()));

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG_VIEWPORT))
            .show_inside(ui, |ui| edited |= self.timeline(ui, scene, target.as_ref()));

        // Preview playback and recording.
        if self.playing {
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            self.playhead += dt;
            let len = self.clip.length.max(1e-3);
            if self.playhead > len {
                if self.clip.looping {
                    self.playhead = self.playhead.rem_euclid(len);
                } else {
                    self.playhead = len;
                    self.playing = false;
                }
            }
            ui.ctx().request_repaint();
        }
        if self.preview
            && let Some(t) = &target
        {
            if self.recording {
                edited |= self.record_edits(scene, t);
            }
            self.apply_preview(scene, t);
        }
        if edited {
            self.changed(scene);
        }
        if self.dirty && !ui.input(|i| i.pointer.any_down()) && !ui.ctx().wants_keyboard_input() {
            self.save(root);
        }
    }

    fn create_for_target(&mut self, scene: &mut LoadedScene, root: &Path, target: Option<&Target>) {
        create_asset(&root.join("animations"), false, root, &mut AnimControllerTab::default(), self);
        self.open_request = false;
        // Give the target an Animator playing it when it has none.
        if let (Some(t), Some(path)) = (target, self.path.clone()) {
            let o = &mut scene.objects[t.index];
            match o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Animator>()) {
                Some(a) if a.controller.is_empty() && a.clip.is_empty() => a.clip = path,
                Some(_) => {}
                None => o.components.push(Box::new(Animator { clip: path, ..Default::default() })),
            }
        }
    }

    fn toolbar(
        &mut self,
        ui: &mut egui::Ui,
        scene: &mut LoadedScene,
        root: &Path,
        target: Option<&Target>,
        selected: Option<usize>,
        playing_game: bool,
    ) -> bool {
        let mut edited = false;
        ui.horizontal(|ui| {
            ui.set_min_height(theme::H_TOOLBAR);
            ui.label(egui::RichText::new(ph::FILM_STRIP).size(15.0).color(theme::TEXT_MUTED));
            match &self.path {
                Some(p) => {
                    let stem = Path::new(p).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                    ui.label(egui::RichText::new(stem).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG)).on_hover_text(p.as_str());
                    widgets::chip(ui, if self.dirty { "Unsaved" } else { "Saved" });
                }
                None => {
                    ui.label(egui::RichText::new("No clip").color(theme::TEXT_MUTED));
                }
            }
            if widgets::secondary_button(ui, Some(Icon::Plus), "New").clicked() {
                self.create_for_target(scene, root, target);
            }
            let open = widgets::secondary_button(ui, Some(Icon::Folder), "Open");
            if open.clicked() {
                self.files = scan_files(root, zestengine_assets::ANIM_CLIP_EXTENSION);
            }
            egui::Popup::menu(&open).show(|ui| {
                if self.files.is_empty() {
                    ui.label(egui::RichText::new("No .anim files in the project").color(theme::TEXT_MUTED));
                }
                for f in self.files.clone() {
                    if ui.selectable_label(self.path.as_deref() == Some(f.as_str()), &f).clicked() {
                        self.open(root, &f);
                        ui.close();
                    }
                }
            });
            self.bake_menu(ui, scene, root, target);
            if self.path.is_none() {
                return;
            }
            ui.separator();
            ui.add_enabled_ui(!playing_game && target.is_some(), |ui| {
                if widgets::icon_button_sized(ui, Icon::Eye, "Preview the clip on the target in the scene", self.preview, Vec2::splat(theme::H_FIELD))
                    .clicked()
                {
                    if self.preview {
                        self.end_preview(scene);
                    } else {
                        self.preview = true;
                    }
                }
                let rec_color = if self.recording { theme::STATUS_ERROR } else { theme::TEXT_MUTED };
                if ui
                    .add(egui::Button::new(egui::RichText::new(ph::RECORD).size(15.0).color(rec_color)).selected(self.recording))
                    .on_hover_text("Record: moving, rotating or scaling the target hierarchy keys it at the playhead")
                    .clicked()
                {
                    self.recording = !self.recording;
                    if self.recording {
                        self.preview = true;
                    }
                }
                if ui.button(egui::RichText::new(ph::SKIP_BACK).size(14.0)).on_hover_text("Go to start").clicked() {
                    self.playhead = 0.0;
                }
                let (icon, tip) = if self.playing { (Icon::Pause, "Pause preview") } else { (Icon::Play, "Play preview") };
                if widgets::icon_button_sized(ui, icon, tip, self.playing, Vec2::splat(theme::H_FIELD)).clicked() {
                    self.playing = !self.playing;
                    self.preview |= self.playing;
                }
            });
            ui.add(egui::DragValue::new(&mut self.playhead).speed(0.01).range(0.0..=1000.0).max_decimals(3).suffix(" s"))
                .on_hover_text("Playhead");
            ui.separator();
            ui.label(egui::RichText::new("Length").color(theme::TEXT_MUTED));
            edited |= ui.add(egui::DragValue::new(&mut self.clip.length).speed(0.01).range(0.01..=600.0).max_decimals(3).suffix(" s")).changed();
            edited |= ui.checkbox(&mut self.clip.looping, "Loop").changed();
            ui.label(egui::RichText::new("FPS").color(theme::TEXT_MUTED));
            edited |= ui.add(egui::DragValue::new(&mut self.clip.frame_rate).speed(1.0).range(1.0..=240.0)).changed();
            ui.separator();
            if let Some(t) = target {
                if widgets::secondary_button(ui, Some(Icon::Plus), "Key")
                    .on_hover_text("Key the selected object's tracks (all tracks when the target is selected) at the playhead")
                    .clicked()
                {
                    let only = selected.and_then(|s| t.nodes.iter().position(|&n| n == s)).filter(|&n| n != 0).map(|n| t.rig.names[n].clone());
                    for k in 0..self.clip.tracks.len() {
                        if only.as_ref().is_some_and(|p| *p != self.clip.tracks[k].target) {
                            continue;
                        }
                        let v = self.current_value(scene, t, &self.clip.tracks[k]);
                        self.key_track(k, v);
                    }
                    edited = true;
                }
                let has_ctrl = scene.objects[t.index]
                    .components
                    .iter()
                    .find_map(|c| c.as_any().downcast_ref::<Animator>())
                    .is_some_and(|a| !a.controller.is_empty());
                if !has_ctrl
                    && let Some(path) = self.path.clone()
                    && widgets::secondary_button(ui, None, "Use on Animator")
                        .on_hover_text("Make the target's Animator play this clip")
                        .clicked()
                {
                    let o = &mut scene.objects[t.index];
                    match o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Animator>()) {
                        Some(a) => a.clip = path,
                        None => o.components.push(Box::new(Animator { clip: path, ..Default::default() })),
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    widgets::chip(ui, &format!("Target: {}", scene.objects[t.index].name));
                });
            }
        });
        edited
    }

    fn track_label(&self, scene: &LoadedScene, target: Option<&Target>, track: &AnimTrack) -> (String, bool) {
        if track.property.is_muscle() {
            return (track_property_label(track), true);
        }
        let name = if track.target.is_empty() {
            target.map(|t| scene.objects[t.index].name.clone()).unwrap_or_else(|| "(root)".into())
        } else {
            track.target.rsplit('/').next().unwrap_or(&track.target).to_owned()
        };
        let found = target.is_none_or(|t| t.rig.index(&track.target).is_some() || t.bones.iter().any(|(n, _)| *n == track.target));
        (format!("{name} : {}", track_property_label(track)), found)
    }

    fn track_list(&mut self, ui: &mut egui::Ui, scene: &LoadedScene, target: Option<&Target>) -> bool {
        let mut edited = false;
        let rect = ui.max_rect();
        let header = Rect::from_min_size(rect.min, Vec2::new(rect.width(), RULER_H + 4.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(header.shrink2(Vec2::new(8.0, 2.0))), |ui| {
            ui.horizontal_centered(|ui| {
                let add = widgets::secondary_button_sized(ui, Some(Icon::Plus), "Add Property", theme::H_BUTTON_SMALL);
                egui::Popup::menu(&add).show(|ui| {
                    if self.is_muscle_clip() {
                        let reference = self.target_reference(scene, target);
                        edited |= self.muscle_add_menu(ui, reference.as_ref());
                        return;
                    }
                    let Some(t) = target else {
                        ui.label(egui::RichText::new("Select an object to animate").color(theme::TEXT_MUTED));
                        return;
                    };
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        for (n, path) in t.rig.names.iter().enumerate() {
                            let label = if path.is_empty() { scene.objects[t.index].name.clone() } else { path.clone() };
                            let light = light_intensity(scene, t.nodes[n]).is_some();
                            ui.menu_button(label, |ui| {
                                crate::widgets::menu_frame(ui);
                                for p in TrackProperty::ALL {
                                    if p == TrackProperty::LightIntensity && !light {
                                        continue;
                                    }
                                    let exists = self.clip.track_index(path, p).is_some();
                                    if ui.add_enabled(!exists, egui::Button::new(p.label())).clicked() {
                                        self.add_track(scene, t, path, p);
                                        edited = true;
                                        ui.close();
                                    }
                                }
                                edited |= self.prop_menu(ui, scene, t, n, path);
                            });
                        }
                        if !t.bones.is_empty() {
                            ui.separator();
                            ui.menu_button("Bones", |ui| {
                                crate::widgets::menu_frame(ui);
                                egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                                    for (bone, _) in &t.bones {
                                        ui.menu_button(bone, |ui| {
                                            crate::widgets::menu_frame(ui);
                                            for p in [TrackProperty::Translation, TrackProperty::Rotation, TrackProperty::Scale] {
                                                let exists = self.clip.track_index(bone, p).is_some();
                                                if ui.add_enabled(!exists, egui::Button::new(p.label())).clicked() {
                                                    self.add_track(scene, t, bone, p);
                                                    edited = true;
                                                    ui.close();
                                                }
                                            }
                                        });
                                    }
                                });
                            });
                        }
                    });
                });
            });
        });
        let body = Rect::from_min_max(Pos2::new(rect.left(), header.bottom()), rect.max);
        let painter = ui.painter_at(body);
        let mut remove = None;
        for k in 0..self.clip.tracks.len() {
            let y = body.top() + k as f32 * ROW_H - self.vscroll;
            let row = Rect::from_min_size(Pos2::new(body.left(), y), Vec2::new(body.width(), ROW_H));
            if row.bottom() < body.top() || row.top() > body.bottom() {
                continue;
            }
            if k % 2 == 1 {
                painter.rect_filled(row, CornerRadius::ZERO, theme::with_alpha(theme::BG_APP, 0.5));
            }
            let selected = self.selected.iter().any(|(t, _)| *t == k);
            if selected {
                painter.rect_filled(row, CornerRadius::ZERO, theme::accent_faint());
            }
            let (label, found) = self.track_label(scene, target, &self.clip.tracks[k]);
            painter.text(
                Pos2::new(row.left() + 12.0, row.center().y),
                egui::Align2::LEFT_CENTER,
                label,
                egui::FontId::proportional(theme::SIZE_BODY),
                if found { theme::TEXT_BODY } else { theme::TEXT_HIDDEN },
            );
            let btn = Rect::from_center_size(Pos2::new(row.right() - 16.0, row.center().y), Vec2::splat(20.0));
            if body.contains_rect(btn) || body.intersects(btn) {
                let r = ui.put(btn, |ui: &mut egui::Ui| widgets::icon_button(ui, Icon::Trash, "Remove track", false));
                if r.clicked() {
                    remove = Some(k);
                }
            }
        }
        if let Some(k) = remove {
            self.clip.tracks.remove(k);
            self.selected.clear();
            edited = true;
        }
        if ui.rect_contains_pointer(body) {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            let max = (self.clip.tracks.len() as f32 * ROW_H - body.height()).max(0.0);
            self.vscroll = (self.vscroll - scroll).clamp(0.0, max);
        }
        if self.clip.tracks.is_empty() {
            painter.text(
                body.center_top() + Vec2::new(0.0, 24.0),
                egui::Align2::CENTER_TOP,
                "Add Property to animate the target or a child",
                egui::FontId::proportional(theme::SIZE_META),
                theme::TEXT_DIM,
            );
        }
        edited
    }

    fn time_to_x(&self, area: Rect, t: f32) -> f32 {
        area.left() + 10.0 + (t - self.scroll) * self.pps
    }

    fn x_to_time(&self, area: Rect, x: f32) -> f32 {
        (x - area.left() - 10.0) / self.pps + self.scroll
    }

    fn timeline(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, target: Option<&Target>) -> bool {
        let mut edited = false;
        let area = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(area, Sense::click_and_drag());
        let painter = ui.painter_at(area);
        let ruler = Rect::from_min_size(area.min, Vec2::new(area.width(), RULER_H + 4.0));
        let rows = Rect::from_min_max(Pos2::new(area.left(), ruler.bottom()), area.max);
        let pointer = ui.input(|i| i.pointer.hover_pos());
        let (mods, scroll) = ui.input(|i| (i.modifiers, i.smooth_scroll_delta));

        // Zoom (wheel) and pan (Shift + wheel, middle drag).
        if response.hovered() {
            if mods.shift {
                self.scroll = (self.scroll - (scroll.y + scroll.x) / self.pps).max(0.0);
            } else if scroll.y.abs() > 0.0
                && let Some(p) = pointer
            {
                let t = self.x_to_time(area, p.x);
                self.pps = (self.pps * (1.0 + scroll.y * 0.002)).clamp(20.0, 2000.0);
                self.scroll = (t - (p.x - area.left() - 10.0) / self.pps).max(0.0);
            } else if scroll.x.abs() > 0.0 {
                self.scroll = (self.scroll - scroll.x / self.pps).max(0.0);
            }
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            self.scroll = (self.scroll - response.drag_delta().x / self.pps).max(0.0);
        }

        // Past-the-end shading.
        let end_x = self.time_to_x(area, self.clip.length);
        if end_x < area.right() {
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(end_x.max(area.left()), rows.top()), area.max),
                CornerRadius::ZERO,
                theme::with_alpha(theme::BG_CHROME, 0.6),
            );
        }

        // Ruler.
        painter.rect_filled(ruler, CornerRadius::ZERO, theme::BG_PANEL);
        let steps = [1.0 / 60.0, 1.0 / 30.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
        let major = steps.iter().copied().find(|s| s * self.pps >= 64.0).unwrap_or(60.0);
        let minor = major / 5.0;
        let t0 = (self.scroll / minor).floor() * minor;
        let mut t = t0;
        let t_end = self.x_to_time(area, area.right());
        while t <= t_end {
            let x = self.time_to_x(area, t);
            let is_major = ((t / major).round() * major - t).abs() < minor * 0.1;
            let h = if is_major { 10.0 } else { 4.0 };
            painter.line_segment([Pos2::new(x, ruler.bottom() - h), Pos2::new(x, ruler.bottom())], Stroke::new(1.0_f32, theme::TEXT_DIM));
            if is_major {
                painter.line_segment([Pos2::new(x, rows.top()), Pos2::new(x, rows.bottom())], Stroke::new(1.0_f32, theme::with_alpha(theme::LINE_PANEL, 0.7)));
                painter.text(
                    Pos2::new(x + 3.0, ruler.top() + 3.0),
                    egui::Align2::LEFT_TOP,
                    format!("{:.2}", t),
                    theme::mono(theme::SIZE_TINY),
                    theme::TEXT_MUTED,
                );
            }
            t += minor;
        }
        painter.line_segment([Pos2::new(end_x, ruler.top()), Pos2::new(end_x, rows.bottom())], Stroke::new(1.0_f32, theme::TEXT_DIM));

        // Events (small flags on the ruler).
        for e in &self.clip.events {
            let x = self.time_to_x(area, e.time);
            painter.add(Shape::convex_polygon(
                vec![Pos2::new(x, ruler.bottom() - 12.0), Pos2::new(x + 6.0, ruler.bottom() - 9.0), Pos2::new(x, ruler.bottom() - 6.0)],
                theme::STATUS_INFO,
                Stroke::NONE,
            ));
        }

        // Rows + keys.
        let row_rect = |k: usize| Rect::from_min_size(Pos2::new(area.left(), rows.top() + k as f32 * ROW_H - self.vscroll), Vec2::new(area.width(), ROW_H));
        let mut hovered_key: Option<(usize, usize)> = None;
        for (k, track) in self.clip.tracks.iter().enumerate() {
            let r = row_rect(k);
            if r.bottom() < rows.top() || r.top() > rows.bottom() {
                continue;
            }
            if k % 2 == 1 {
                painter.rect_filled(r.intersect(rows), CornerRadius::ZERO, theme::with_alpha(theme::BG_APP, 0.5));
            }
            for (j, key) in track.keys.iter().enumerate() {
                let x = self.time_to_x(area, key.time);
                if x < area.left() - 8.0 || x > area.right() + 8.0 {
                    continue;
                }
                let c = Pos2::new(x, r.center().y);
                let hit = pointer.is_some_and(|p| (p - c).length() < 7.0 && rows.contains(p));
                if hit {
                    hovered_key = Some((k, j));
                }
                let selected = self.selected.contains(&(k, j));
                let fill = if selected {
                    theme::accent()
                } else if hit {
                    theme::TEXT_STRONG
                } else {
                    theme::TEXT_SECONDARY
                };
                let s = 5.5;
                let pts = match key.interp {
                    KeyInterp::Step => vec![c + Vec2::new(-s, -s), c + Vec2::new(s, -s), c + Vec2::new(s, s), c + Vec2::new(-s, s)],
                    _ => vec![c + Vec2::new(0.0, -s), c + Vec2::new(s, 0.0), c + Vec2::new(0.0, s), c + Vec2::new(-s, 0.0)],
                };
                painter.add(Shape::convex_polygon(pts, fill, Stroke::new(1.0_f32, theme::BG_CHROME)));
            }
        }

        // Interaction.
        let free = mods.alt;
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            if ruler.contains(p) {
                self.scrubbing = true;
            } else if let Some(hk) = hovered_key {
                if !self.selected.contains(&hk) {
                    if !mods.ctrl && !mods.command {
                        self.selected.clear();
                    }
                    self.selected.push(hk);
                }
                let orig = self
                    .selected
                    .iter()
                    .filter_map(|&(t, k)| self.clip.tracks.get(t).and_then(|tr| tr.keys.get(k)).map(|key| (t, k, key.time)))
                    .collect();
                self.drag = Some((self.x_to_time(area, p.x), orig));
            }
        }
        if response.dragged_by(egui::PointerButton::Primary)
            && let Some(p) = pointer
        {
            if self.scrubbing {
                self.playhead = self.snap(self.x_to_time(area, p.x), free);
                self.preview |= target.is_some();
            } else if let Some((start, orig)) = &self.drag {
                let delta = self.x_to_time(area, p.x) - start;
                for &(t, k, time) in orig {
                    let new = self.snap(time + delta, free);
                    if let Some(key) = self.clip.tracks.get_mut(t).and_then(|tr| tr.keys.get_mut(k)) {
                        key.time = new;
                    }
                }
                edited = true;
            }
        }
        if response.drag_stopped() {
            self.scrubbing = false;
            if self.drag.take().is_some() {
                self.resort_selection();
                edited = true;
            }
        }
        if response.clicked_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            if ruler.contains(p) {
                self.playhead = self.snap(self.x_to_time(area, p.x), free);
                self.preview |= target.is_some();
            } else if let Some(hk) = hovered_key {
                if mods.ctrl || mods.command {
                    if let Some(i) = self.selected.iter().position(|s| *s == hk) {
                        self.selected.remove(i);
                    } else {
                        self.selected.push(hk);
                    }
                } else {
                    self.selected = vec![hk];
                }
            } else {
                self.selected.clear();
            }
        }
        if response.double_clicked()
            && hovered_key.is_none()
            && let Some(p) = response.interact_pointer_pos()
            && rows.contains(p)
        {
            let k = ((p.y - rows.top() + self.vscroll) / ROW_H).floor() as usize;
            if k < self.clip.tracks.len() {
                let time = self.snap(self.x_to_time(area, p.x), free);
                let v = match target {
                    Some(t) if (time - self.playhead).abs() < 1e-4 => self.current_value(scene, t, &self.clip.tracks[k]),
                    _ => self.clip.tracks[k].sample(time),
                };
                let j = self.clip.tracks[k].set_key(time, v);
                self.selected = vec![(k, j)];
                edited = true;
            }
        }
        response.context_menu(|ui| {
            crate::widgets::menu_frame(ui);
            if self.selected.is_empty() {
                ui.label(egui::RichText::new("Select keys first").color(theme::TEXT_MUTED));
                return;
            }
            for interp in KeyInterp::ALL {
                if ui.button(interp.label()).clicked() {
                    self.set_interp(interp);
                    edited = true;
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Delete keys").clicked() {
                self.delete_selected();
                edited = true;
                ui.close();
            }
        });

        // Keyboard: Delete, Ctrl+C / Ctrl+V.
        if (response.hovered() || response.has_focus()) && !ui.ctx().wants_keyboard_input() {
            let (del, copy, paste) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::Delete),
                    i.events.iter().any(|e| matches!(e, egui::Event::Copy)),
                    i.events.iter().any(|e| matches!(e, egui::Event::Paste(_))) || (i.modifiers.command && i.key_pressed(egui::Key::V)),
                )
            });
            if del && !self.selected.is_empty() {
                self.delete_selected();
                edited = true;
            }
            if copy {
                self.copy_selected();
            }
            if paste && !self.clipboard.is_empty() {
                self.paste();
                edited = true;
            }
        }

        // Playhead.
        let x = self.time_to_x(area, self.playhead);
        if x >= area.left() && x <= area.right() {
            let color = if self.recording { theme::STATUS_ERROR } else { theme::accent() };
            painter.line_segment([Pos2::new(x, ruler.top()), Pos2::new(x, rows.bottom())], Stroke::new(1.5_f32, color));
            painter.add(Shape::convex_polygon(
                vec![Pos2::new(x - 5.0, ruler.top()), Pos2::new(x + 5.0, ruler.top()), Pos2::new(x, ruler.top() + 7.0)],
                color,
                Stroke::NONE,
            ));
        }
        if self.recording {
            painter.rect_stroke(area.shrink(1.0), CornerRadius::ZERO, Stroke::new(2.0_f32, theme::with_alpha(theme::STATUS_ERROR, 0.6)), egui::StrokeKind::Inside);
        }
        painter.text(
            area.left_bottom() + Vec2::new(10.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "Click ruler: scrub · Double-click row: add key · Drag keys · Del · Ctrl+C / Ctrl+V · Wheel: zoom · Shift+wheel: pan · Alt: no snap",
            egui::FontId::proportional(theme::SIZE_META),
            theme::TEXT_DIM,
        );
        if self.drag.is_some() || self.scrubbing {
            ui.ctx().request_repaint();
        }
        edited
    }

    fn set_interp(&mut self, interp: KeyInterp) {
        for &(t, k) in &self.selected {
            if let Some(key) = self.clip.tracks.get_mut(t).and_then(|tr| tr.keys.get_mut(k)) {
                key.interp = interp;
            }
        }
    }

    fn delete_selected(&mut self) {
        let mut sel = self.selected.clone();
        sel.sort_by(|a, b| b.cmp(a));
        for (t, k) in sel {
            if let Some(tr) = self.clip.tracks.get_mut(t)
                && k < tr.keys.len()
            {
                tr.keys.remove(k);
            }
        }
        self.selected.clear();
    }

    fn copy_selected(&mut self) {
        let keys: Vec<(usize, Keyframe)> = self
            .selected
            .iter()
            .filter_map(|&(t, k)| self.clip.tracks.get(t).and_then(|tr| tr.keys.get(k)).map(|key| (t, *key)))
            .collect();
        let first = keys.iter().map(|(_, k)| k.time).fold(f32::INFINITY, f32::min);
        self.clipboard = keys.into_iter().map(|(t, mut k)| {
            k.time -= first;
            (t, k)
        }).collect();
    }

    fn paste(&mut self) {
        let base = self.snap(self.playhead, false);
        self.selected.clear();
        for (t, k) in self.clipboard.clone() {
            if let Some(tr) = self.clip.tracks.get_mut(t) {
                let time = base + k.time;
                let j = tr.set_key(time, k.value);
                tr.keys[j].interp = k.interp;
                self.clip.length = self.clip.length.max(time);
            }
        }
    }

    /// After moving keys: sort tracks, merge duplicates, and keep the moved
    /// keys selected.
    fn resort_selection(&mut self) {
        let picked: Vec<(usize, f32)> = self
            .selected
            .iter()
            .filter_map(|&(t, k)| self.clip.tracks.get(t).and_then(|tr| tr.keys.get(k)).map(|key| (t, key.time)))
            .collect();
        for tr in &mut self.clip.tracks {
            tr.sort();
            tr.keys.dedup_by(|a, b| (a.time - b.time).abs() < 5e-4);
        }
        self.selected = picked
            .into_iter()
            .filter_map(|(t, time)| {
                self.clip.tracks[t].keys.iter().position(|k| (k.time - time).abs() < 5e-4).map(|k| (t, k))
            })
            .collect();
    }

    fn key_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut edited = false;
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width((ui.available_width() * 0.6).max(280.0));
                match self.selected.len() {
                    0 => {
                        widgets::eyebrow(ui, "Key");
                        ui.label(egui::RichText::new("Select a key to edit its time, value and curve.").size(theme::SIZE_META).color(theme::TEXT_DIM));
                    }
                    1 => {
                        let (t, k) = self.selected[0];
                        let property = match self.clip.tracks.get(t) {
                            Some(track) => track.property,
                            None => return,
                        };
                        {
                            let Some(track) = self.clip.tracks.get_mut(t) else {
                                return;
                            };
                            let Some(key) = track.keys.get_mut(k) else {
                                return;
                            };
                            widgets::eyebrow(ui, &format!("Key  {}", property.label()));
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Time").color(theme::TEXT_MUTED));
                                if ui.add(egui::DragValue::new(&mut key.time).speed(0.01).range(0.0..=600.0).max_decimals(3).suffix(" s")).changed() {
                                    edited = true;
                                }
                                ui.label(egui::RichText::new("Curve").color(theme::TEXT_MUTED));
                                egui::ComboBox::from_id_salt("zestengine-key-interp").selected_text(key.interp.label()).show_ui(ui, |ui| {
                                    for i in KeyInterp::ALL {
                                        edited |= ui.selectable_value(&mut key.interp, i, i.label()).changed();
                                    }
                                });
                            });
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Value").color(theme::TEXT_MUTED));
                                match property {
                                    TrackProperty::Rotation => {
                                        let mut e = quat_to_euler_deg(key.value);
                                        if widgets::vec3_field(ui, &mut e, 1, 0.5) {
                                            key.value = euler_deg_to_quat(e);
                                            edited = true;
                                        }
                                    }
                                    TrackProperty::LightIntensity => {
                                        edited |= ui.add(egui::DragValue::new(&mut key.value[0]).speed(0.05)).changed();
                                    }
                                    TrackProperty::Active | TrackProperty::ComponentEnabled => {
                                        let mut on = key.value[0] >= 0.5;
                                        let text = if on { "On" } else { "Off" };
                                        if ui.checkbox(&mut on, text).changed() {
                                            key.value[0] = if on { 1.0 } else { 0.0 };
                                            edited = true;
                                        }
                                    }
                                    TrackProperty::BlendShape | TrackProperty::MaterialFloat => {
                                        edited |= ui.add(egui::DragValue::new(&mut key.value[0]).speed(0.01).max_decimals(3)).changed();
                                    }
                                    TrackProperty::MaterialColor => {
                                        edited |= widgets::color_hex(ui, &mut key.value).changed();
                                    }
                                    TrackProperty::MaterialOffset => {
                                        for lane in 0..2 {
                                            edited |= ui.add(egui::DragValue::new(&mut key.value[lane]).speed(0.005).max_decimals(4)).changed();
                                        }
                                    }
                                    _ => {
                                        let mut v = [key.value[0], key.value[1], key.value[2]];
                                        if widgets::vec3_field(ui, &mut v, 3, 0.01) {
                                            key.value[..3].copy_from_slice(&v);
                                            edited = true;
                                        }
                                    }
                                }
                            });
                            // A Custom (bezier) key exposes its tangents (derivatives
                            // w.r.t. time). Rotation keeps automatic tangents (editing
                            // quaternion derivatives by hand is error-prone).
                            if key.interp == KeyInterp::Custom {
                                for (label, slot) in [("Tangent out", &mut key.tangent_out), ("Tangent in", &mut key.tangent_in)] {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new(label).color(theme::TEXT_MUTED));
                                        match property {
                                            TrackProperty::Rotation => {
                                                ui.label(egui::RichText::new("auto").color(theme::TEXT_DIM));
                                            }
                                            TrackProperty::Active | TrackProperty::ComponentEnabled => {
                                                ui.label(egui::RichText::new("switch").color(theme::TEXT_DIM));
                                            }
                                            TrackProperty::LightIntensity | TrackProperty::BlendShape | TrackProperty::MaterialFloat => {
                                                let mut t = slot.unwrap_or([0.0; 4]);
                                                if ui.add(egui::DragValue::new(&mut t[0]).speed(0.05)).changed() {
                                                    *slot = Some(t);
                                                    edited = true;
                                                }
                                            }
                                            _ => {
                                                let mut t = slot.unwrap_or([0.0; 4]);
                                                let mut v = [t[0], t[1], t[2]];
                                                if widgets::vec3_field(ui, &mut v, 3, 0.01) {
                                                    t[..3].copy_from_slice(&v);
                                                    *slot = Some(t);
                                                    edited = true;
                                                }
                                            }
                                        }
                                    });
                                }
                            }
                        }
                        edited |= self.curve_editor(ui, t, k);
                    }
                    n => {
                        widgets::eyebrow(ui, &format!("{n} keys"));
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Curve").color(theme::TEXT_MUTED));
                            for i in KeyInterp::ALL {
                                if ui.button(i.label()).clicked() {
                                    self.set_interp(i);
                                    edited = true;
                                }
                            }
                            if widgets::icon_button(ui, Icon::Trash, "Delete keys", false).clicked() {
                                self.delete_selected();
                                edited = true;
                            }
                        });
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    widgets::eyebrow(ui, "Events");
                    if widgets::icon_button(ui, Icon::Plus, "Add an event at the playhead", false).clicked() {
                        self.clip.events.push(ClipEvent { time: self.snap(self.playhead, false), name: "Event".into() });
                        edited = true;
                    }
                });
                let mut remove = None;
                egui::ScrollArea::vertical().id_salt("zestengine-clip-events").max_height(70.0).show(ui, |ui| {
                    for (k, e) in self.clip.events.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            edited |= ui.add(egui::DragValue::new(&mut e.time).speed(0.01).range(0.0..=600.0).max_decimals(3).suffix(" s")).changed();
                            edited |= ui.add(egui::TextEdit::singleline(&mut e.name).desired_width(110.0)).changed();
                            if widgets::icon_button(ui, Icon::Close, "Remove event", false).clicked() {
                                remove = Some(k);
                            }
                        });
                    }
                });
                if let Some(k) = remove {
                    self.clip.events.remove(k);
                    edited = true;
                }
            });
        });
        if edited {
            for tr in &mut self.clip.tracks {
                tr.sort();
            }
        }
        edited
    }

    /// Graphical curve editor for the selected key: plots the track around the
    /// key and, for Custom (bezier) non-rotation keys, exposes draggable
    /// incoming / outgoing tangent handles. The handles edit the same
    /// `Keyframe::tangent_in` / `tangent_out` values as the numeric fields.
    fn curve_editor(&mut self, ui: &mut egui::Ui, t: usize, k: usize) -> bool {
        let Some(property) = self.clip.tracks.get(t).map(|tr| tr.property) else {
            return false;
        };
        let interp = self.clip.tracks.get(t).and_then(|tr| tr.keys.get(k)).map(|key| key.interp);
        let lanes = property.lanes().max(1);
        let mut lane = self.curve_lane.min(lanes - 1);

        ui.horizontal(|ui| {
            widgets::eyebrow(ui, "Curve");
            ui.label(egui::RichText::new("lane").color(theme::TEXT_MUTED));
            egui::ComboBox::from_id_salt("zestengine-curve-lane")
                .selected_text(lane_name(property, lane))
                .show_ui(ui, |ui| {
                    for l in 0..lanes {
                        ui.selectable_value(&mut self.curve_lane, l, lane_name(property, l));
                    }
                });
        });
        lane = self.curve_lane.min(lanes - 1);

        let Some(track) = self.clip.tracks.get(t) else {
            return false;
        };
        let Some(key) = track.keys.get(k) else {
            return false;
        };
        let t_key = key.time;
        let key_v = key.value[lane];

        // Show the segment(s) touching this key, with a little context on the
        // open side when it is the first or last key.
        let prev_t = track.keys[..k].last().map(|p| p.time);
        let next_t = track.keys.get(k + 1).map(|n| n.time);
        let mut t_min = prev_t.unwrap_or(t_key - 0.5);
        let mut t_max = next_t.unwrap_or(t_key + 0.5);
        if (t_max - t_min).abs() < 1e-4 {
            t_min = t_key - 0.5;
            t_max = t_key + 0.5;
        }
        let pad = (t_max - t_min).max(1e-3) * 0.12;
        t_min -= pad;
        t_max += pad;
        let h_dt = (t_max - t_min) * 0.16;

        // Sample the window to get the value range and the plotted curve.
        let n = 96;
        let mut samples = Vec::with_capacity(n + 1);
        let mut v_min = f32::INFINITY;
        let mut v_max = f32::NEG_INFINITY;
        let mut note = |v: f32| {
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        };
        for i in 0..=n {
            let time = t_min + (t_max - t_min) * i as f32 / n as f32;
            let v = track.sample(time)[lane];
            samples.push((time, v));
            note(v);
        }
        note(key_v);
        let custom = key.interp == KeyInterp::Custom && property != TrackProperty::Rotation;
        // A tangent only shapes a segment when that segment uses Custom: the
        // outgoing one is driven by this key, the incoming one by the previous
        // key. Endpoint sides have no segment, so they expose no handle.
        let out_active = custom && next_t.is_some();
        let in_active = custom && prev_t.is_some() && track.keys[k - 1].interp == KeyInterp::Custom;
        let eps = (t_max - t_min) * 1e-3;
        let m_out = if out_active {
            (track.sample(t_key + eps)[lane] - track.sample(t_key)[lane]) / eps
        } else {
            0.0
        };
        let m_in = if in_active {
            (track.sample(t_key)[lane] - track.sample(t_key - eps)[lane]) / eps
        } else {
            0.0
        };
        if out_active {
            note(key_v + m_out * h_dt);
        }
        if in_active {
            note(key_v - m_in * h_dt);
        }
        if (v_max - v_min).abs() < 1e-4 {
            v_max = v_min + 1.0;
        }
        let v_pad = (v_max - v_min) * 0.15;
        v_min -= v_pad;
        v_max += v_pad;

        let height = (ui.available_height() - 8.0).clamp(96.0, 220.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
        let painter = ui.painter_at(rect).with_clip_rect(rect);
        painter.rect_filled(rect, CornerRadius::ZERO, theme::BG_VIEWPORT);
        painter.rect_stroke(rect, CornerRadius::ZERO, Stroke::new(1.0_f32, theme::LINE_PANEL), egui::StrokeKind::Inside);

        let inner = rect.shrink2(Vec2::new(12.0, 12.0));
        let x_of = |time: f32| inner.left() + (time - t_min) / (t_max - t_min) * inner.width();
        let y_of = |v: f32| inner.bottom() - (v - v_min) / (v_max - v_min) * inner.height();
        let y_to_v = |y: f32| v_min + (inner.bottom() - y) / inner.height() * (v_max - v_min);

        // Hint for keys whose tangents cannot be dragged here.
        if property == TrackProperty::Rotation {
            painter.text(
                inner.left_top() + Vec2::new(4.0, 2.0),
                egui::Align2::LEFT_TOP,
                "rotation tangents stay automatic",
                theme::mono(theme::SIZE_TINY),
                theme::TEXT_DIM,
            );
        } else if interp != Some(KeyInterp::Custom) {
            painter.text(
                inner.left_top() + Vec2::new(4.0, 2.0),
                egui::Align2::LEFT_TOP,
                "set Curve to Custom to drag tangents",
                theme::mono(theme::SIZE_TINY),
                theme::TEXT_DIM,
            );
        }

        // Grid: the selected key's time and the zero line.
        let xk = x_of(t_key);
        painter.line_segment([Pos2::new(xk, inner.top()), Pos2::new(xk, inner.bottom())], Stroke::new(1.0_f32, theme::with_alpha(theme::LINE_PANEL, 0.8)));
        if v_min < 0.0 && v_max > 0.0 {
            let y0 = y_of(0.0);
            painter.line_segment([Pos2::new(inner.left(), y0), Pos2::new(inner.right(), y0)], Stroke::new(1.0_f32, theme::with_alpha(theme::LINE_PANEL, 0.8)));
        }

        let points: Vec<Pos2> = samples.iter().map(|&(time, v)| Pos2::new(x_of(time), y_of(v))).collect();
        painter.add(Shape::line(points, Stroke::new(1.5_f32, theme::accent())));

        for (j, kk) in track.keys.iter().enumerate() {
            if kk.time < t_min || kk.time > t_max {
                continue;
            }
            let c = Pos2::new(x_of(kk.time), y_of(kk.value[lane]));
            let s = 4.0;
            let pts = vec![
                c + Vec2::new(0.0, -s),
                c + Vec2::new(s, 0.0),
                c + Vec2::new(0.0, s),
                c + Vec2::new(-s, 0.0),
            ];
            let fill = if j == k { theme::accent() } else { theme::TEXT_SECONDARY };
            painter.add(Shape::convex_polygon(pts, fill, Stroke::new(1.0_f32, theme::BG_CHROME)));
        }

        // Tangent handles for the active Custom segments. Each handle is a fixed
        // horizontal distance from the key, so its vertical position (and drag)
        // encodes the tangent slope dv/dt.
        let mut out_resp: Option<egui::Response> = None;
        let mut in_resp: Option<egui::Response> = None;
        if out_active || in_active {
            let key_pos = Pos2::new(xk, y_of(key_v));
            let style = |resp: &egui::Response, set: bool| -> (egui::Color32, bool) {
                if resp.dragged() || resp.hovered() {
                    (theme::accent(), true)
                } else if set {
                    (theme::TEXT_STRONG, true)
                } else {
                    (theme::TEXT_DIM, false)
                }
            };
            if out_active {
                let out_h = Pos2::new(x_of(t_key + h_dt), y_of(key_v + m_out * h_dt));
                painter.line_segment([key_pos, out_h], Stroke::new(1.0_f32, theme::with_alpha(theme::accent(), if key.tangent_out.is_some() { 1.0 } else { 0.45 })));

                let out_rect = Rect::from_center_size(out_h, Vec2::splat(14.0));
                let out_id = ui.make_persistent_id(("zestengine-curve-out", t, k, lane));
                let resp = ui.interact(out_rect, out_id, Sense::drag());
                let (fill, solid) = style(&resp, key.tangent_out.is_some());
                if solid {
                    painter.circle_filled(out_h, 4.5, fill);
                } else {
                    painter.circle_stroke(out_h, 4.5, Stroke::new(1.5_f32, fill));
                }
                out_resp = Some(resp);
            }
            if in_active {
                let in_h = Pos2::new(x_of(t_key - h_dt), y_of(key_v - m_in * h_dt));
                painter.line_segment([key_pos, in_h], Stroke::new(1.0_f32, theme::with_alpha(theme::accent(), if key.tangent_in.is_some() { 1.0 } else { 0.45 })));

                let in_rect = Rect::from_center_size(in_h, Vec2::splat(14.0));
                let in_id = ui.make_persistent_id(("zestengine-curve-in", t, k, lane));
                let resp = ui.interact(in_rect, in_id, Sense::drag());
                let (fill, solid) = style(&resp, key.tangent_in.is_some());
                if solid {
                    painter.circle_filled(in_h, 4.5, fill);
                } else {
                    painter.circle_stroke(in_h, 4.5, Stroke::new(1.5_f32, fill));
                }
                in_resp = Some(resp);
            }
        }

        let mut new_out: Option<f32> = None;
        let mut new_in: Option<f32> = None;
        if let Some(resp) = out_resp
            && resp.dragged() && let Some(p) = resp.interact_pointer_pos() {
                new_out = Some((y_to_v(p.y) - key_v) / h_dt);
            }
        if let Some(resp) = in_resp
            && resp.dragged() && let Some(p) = resp.interact_pointer_pos() {
                new_in = Some((key_v - y_to_v(p.y)) / h_dt);
            }

        let mut edited = false;
        if (new_out.is_some() || new_in.is_some())
            && let Some(track) = self.clip.tracks.get_mut(t)
                && let Some(key) = track.keys.get_mut(k) {
                    if let Some(m) = new_out {
                        let mut tg = key.tangent_out.unwrap_or([0.0; 4]);
                        tg[lane] = m;
                        key.tangent_out = Some(tg);
                    }
                    if let Some(m) = new_in {
                        let mut tg = key.tangent_in.unwrap_or([0.0; 4]);
                        tg[lane] = m;
                        key.tangent_in = Some(tg);
                    }
                    edited = true;
                }
        edited
    }

    /// While recording, key transform channels the user changed since the
    /// last preview write.
    fn record_edits(&mut self, scene: &LoadedScene, target: &Target) -> bool {
        let mut edited = false;
        let time = self.snap(self.playhead, false);
        for (n, &i) in target.nodes.iter().enumerate() {
            let o = &scene.objects[i];
            let Some(&(t, r, s)) = self.applied.get(&o.id) else {
                continue;
            };
            let path = target.rig.names[n].clone();
            let key = |prop: TrackProperty, v: [f32; 4], clip: &mut AnimClip| {
                let k = match clip.track_index(&path, prop) {
                    Some(k) => k,
                    None => {
                        clip.tracks.push(AnimTrack::new(&path, prop));
                        clip.tracks.len() - 1
                    }
                };
                clip.tracks[k].set_key(time, v);
                clip.length = clip.length.max(time);
            };
            if (o.translation - t).length() > 1e-5 {
                key(TrackProperty::Translation, o.translation.extend(0.0).to_array(), &mut self.clip);
                edited = true;
            }
            if o.rotation.angle_between(r) > 1e-4 {
                key(TrackProperty::Rotation, o.rotation.to_array(), &mut self.clip);
                edited = true;
            }
            if (o.scale - s).length() > 1e-5 {
                key(TrackProperty::Scale, o.scale.extend(0.0).to_array(), &mut self.clip);
                edited = true;
            }
        }
        edited |= self.record_props(scene, target, time);
        edited
    }

    /// The target's nodes whose object properties are watched: the selected
    /// object and every node a property track already drives.
    fn watched_nodes(&self, scene: &LoadedScene, target: &Target) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        let selected = self.selected_object.and_then(|id| target.nodes.iter().position(|&i| scene.objects[i].id == id));
        out.extend(selected);
        for t in self.clip.tracks.iter().filter(|t| t.property.is_object_prop()) {
            if let Some(n) = target.rig.index(&t.target)
                && !out.contains(&n)
            {
                out.push(n);
            }
        }
        out
    }

    /// While recording, key object properties changed since the last preview
    /// write: object active, component switches, blendshape weights and
    /// material values.
    fn record_props(&mut self, scene: &LoadedScene, target: &Target, time: f32) -> bool {
        let mut edited = false;
        for n in self.watched_nodes(scene, target) {
            let i = target.nodes[n];
            let id = scene.objects[i].id;
            let path = target.rig.names[n].clone();
            for (kind, slot, name) in anim_props::watched_props(scene, i, &scene.morph_names(i)) {
                let key = (id, kind, slot, name);
                let Some(&prev) = self.prop_applied.get(&key) else {
                    continue;
                };
                let Some(now) = anim_props::read_prop(scene, i, kind, slot, &key.3) else {
                    continue;
                };
                if anim_props::prop_changed(prev, now) {
                    key_prop(&mut self.clip, &path, kind, &key.3, slot, time, clean_value(now));
                    edited = true;
                }
            }
        }
        edited
    }

    /// Pose the target hierarchy at the playhead (originals kept for restore).
    fn apply_preview(&mut self, scene: &mut LoadedScene, target: &Target) {
        for &i in &target.nodes {
            let o = &scene.objects[i];
            self.originals
                .entry(o.id)
                .or_insert((o.translation, o.rotation, o.scale, light_intensity(scene, i).unwrap_or(0.0)));
        }
        for t in self.clip.tracks.iter().filter(|t| t.property.is_object_prop()) {
            let Some(n) = target.rig.index(&t.target) else {
                continue;
            };
            let i = target.nodes[n];
            let key = (scene.objects[i].id, t.property, if t.property.uses_slot() { t.slot } else { 0 }, t.name.clone());
            if !self.prop_originals.contains_key(&key)
                && let Some(v) = anim_props::read_prop(scene, i, t.property, key.2, &t.name)
            {
                self.prop_originals.insert(key, v);
            }
        }
        let mut pose = Pose::rest(&target.rig);
        ClipRef::Asset(&self.clip).sample(&target.rig, self.playhead, self.clip.looping, &mut pose);
        apply_pose_to_objects(scene, &target.nodes, &pose);
        anim_props::apply_pose_props(scene, &target.rig, &target.nodes, &pose);
        for &i in &target.nodes {
            let o = &scene.objects[i];
            self.applied.insert(o.id, (o.translation, o.rotation, o.scale));
        }
        if self.recording {
            self.prop_applied.clear();
            for n in self.watched_nodes(scene, target) {
                let i = target.nodes[n];
                let id = scene.objects[i].id;
                for (kind, slot, name) in anim_props::watched_props(scene, i, &scene.morph_names(i)) {
                    if let Some(v) = anim_props::read_prop(scene, i, kind, slot, &name) {
                        self.prop_applied.insert((id, kind, slot, name), v);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_tracks_are_labelled_by_what_they_address() {
        let label = |t: AnimTrack| track_property_label(&t);
        assert_eq!(label(AnimTrack::new("", TrackProperty::Translation)), "Position");
        assert_eq!(label(AnimTrack::prop("Hat", TrackProperty::Active, "", 0)), "Active");
        assert_eq!(label(AnimTrack::prop("Hat", TrackProperty::ComponentEnabled, "Light", 0)), "Component Enabled / Light");
        assert_eq!(label(AnimTrack::prop("Body", TrackProperty::BlendShape, "Smile", 0)), "Blend Shape / Smile");
        assert_eq!(label(AnimTrack::prop("Body", TrackProperty::MaterialColor, "base_color", 1)), "Material 2 / Base Colour");
        assert_eq!(label(AnimTrack::prop("Body", TrackProperty::MaterialFloat, "custom_thing", 0)), "Material 1 / custom_thing");
    }

    #[test]
    fn keying_a_property_creates_the_track_once_and_replaces_at_the_same_time() {
        let mut clip = AnimClip::default();
        key_prop(&mut clip, "Hat", TrackProperty::Active, "", 0, 0.5, [1.0, 0.0, 0.0, 0.0]);
        key_prop(&mut clip, "Hat", TrackProperty::Active, "", 0, 0.5, [0.0; 4]);
        key_prop(&mut clip, "Hat", TrackProperty::Active, "", 0, 1.5, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(clip.tracks.len(), 1);
        assert_eq!(clip.tracks[0].keys.len(), 2);
        assert_eq!(clip.tracks[0].keys[0].value[0], 0.0);
        assert_eq!(clip.tracks[0].keys[0].interp, KeyInterp::Step);
        assert_eq!(clip.length, 1.5);
        // Same property on another material slot is its own track.
        key_prop(&mut clip, "Body", TrackProperty::MaterialFloat, "metallic", 0, 0.0, [0.5; 4]);
        key_prop(&mut clip, "Body", TrackProperty::MaterialFloat, "metallic", 1, 0.0, [0.5; 4]);
        assert_eq!(clip.tracks.len(), 3);
    }

    #[test]
    fn absent_weights_read_as_zero_when_keyed() {
        assert_eq!(clean_value([f32::NAN, 1.0, 0.0, 0.0]), [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn lane_names_cover_the_new_properties() {
        assert_eq!(lane_name(TrackProperty::MaterialColor, 3), "A");
        assert_eq!(lane_name(TrackProperty::MaterialOffset, 1), "V");
        assert_eq!(lane_name(TrackProperty::BlendShape, 0), "V");
        assert_eq!(lane_name(TrackProperty::Translation, 2), "Z");
    }
}
