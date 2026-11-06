//! The Sequencer tab: author `.seq` cinematics on a timeline. Tracks are
//! scene-anchored by object id, so a sequence keeps driving the same object
//! across scene reloads. A transform track animates translation / rotation /
//! scale; a camera track adds an optional field-of-view curve. The playhead
//! scrubs and plays a preview in edit mode, and Play mode drives the scene
//! through the same sampler.

use std::collections::HashMap;
use std::path::Path;

use zestengine_assets::{
    SEQUENCER_EXTENSION, SeqChannel, SeqCurve, SeqInterp, SeqTrack, SeqTrackKind, Sequencer,
    load_sequencer, save_sequencer,
};
use zestengine_core::{CameraComponent, ObjectId};
use crate::egui_phosphor::regular as ph;
use crate::icons::Icon;
use crate::{theme, widgets};
use egui::{CornerRadius, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use glam::{EulerRot, Quat, Vec3};

use zestengine_engine::scene::LoadedScene;

const ROW_H: f32 = 24.0;
const RULER_H: f32 = 24.0;

type KeyDrag = (f32, Vec<(usize, usize, f32)>);

fn quat_to_euler_deg(q: [f32; 4]) -> [f32; 3] {
    let (x, y, z) = Quat::from_array(q).normalize().to_euler(EulerRot::XYZ);
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

fn euler_deg_to_quat(e: [f32; 3]) -> [f32; 4] {
    Quat::from_euler(EulerRot::XYZ, e[0].to_radians(), e[1].to_radians(), e[2].to_radians()).to_array()
}

pub struct SequencerTab {
    path: Option<String>,
    seq: Sequencer,
    history: crate::undo::DocHistory<Sequencer>,
    dirty: bool,
    playhead: f32,
    playing: bool,
    preview: bool,
    /// Transforms (and camera FOV) captured before preview / play.
    originals: HashMap<ObjectId, (Vec3, Quat, Vec3, Option<f32>)>,
    selected: Vec<(usize, usize)>,
    pps: f32,
    scroll: f32,
    vscroll: f32,
    drag: Option<KeyDrag>,
    scrubbing: bool,
    /// (track index, channel) rows this frame, shared by the track list and
    /// timeline so their rows line up.
    rows: Vec<(usize, SeqChannel)>,
    pub open_request: bool,
    files: Vec<String>,
}

impl Default for SequencerTab {
    fn default() -> Self {
        Self {
            path: None,
            seq: Sequencer::default(),
            history: Default::default(),
            dirty: false,
            playhead: 0.0,
            playing: false,
            preview: false,
            originals: HashMap::new(),
            selected: Vec::new(),
            pps: 200.0,
            scroll: 0.0,
            vscroll: 0.0,
            drag: None,
            scrubbing: false,
            rows: Vec::new(),
            open_request: false,
            files: Vec::new(),
        }
    }
}

/// Inspector body for a selected `.seq` file: opens it in the Sequencer tab.
pub fn file_inspector(ui: &mut egui::Ui, path: &Path, root: &Path, tab: &mut SequencerTab) {
    let rel = crate::anim_controller_editor::rel_path(root, path);
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Sequence".into());
    ui.label(egui::RichText::new(stem).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
    ui.label(egui::RichText::new(&rel).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
    ui.add_space(8.0);
    if tab.path.as_deref() != Some(rel.as_str()) {
        tab.open(root, &rel);
        tab.open_request = true;
    }
    if widgets::secondary_button(ui, Some(Icon::FileDoc), "Open in Sequencer").clicked() {
        tab.open(root, &rel);
        tab.open_request = true;
    }
    ui.add_space(6.0);
    widgets::info_note(ui, "Edit the timeline, then press Play to drive the scene.");
}

fn main_camera_index(scene: &LoadedScene) -> Option<usize> {
    scene.main_camera()
}

impl SequencerTab {
    pub fn is_open(&self) -> bool {
        self.path.is_some()
    }

    pub fn open(&mut self, root: &Path, rel: &str) {
        if self.path.as_deref() == Some(rel) {
            return;
        }
        if self.dirty {
            self.save(root);
        }
        match load_sequencer(root.join(rel)) {
            Ok(seq) => {
                self.seq = seq;
                self.path = Some(rel.to_owned());
                self.dirty = false;
                self.selected.clear();
                self.playhead = 0.0;
                self.playing = false;
                self.preview = false;
            }
            Err(e) => tracing::error!("opening sequencer {rel}: {e:#}"),
        }
    }

    fn save(&mut self, root: &Path) {
        let Some(path) = &self.path else {
            return;
        };
        match save_sequencer(root.join(path), &self.seq) {
            Ok(()) => self.dirty = false,
            Err(e) => tracing::error!("saving sequencer: {e:#}"),
        }
    }

    fn changed(&mut self) {
        self.dirty = true;
    }

    fn snap(&self, t: f32, free: bool) -> f32 {
        let fps = self.seq.frame_rate.max(1.0);
        let t = t.max(0.0);
        if free { t } else { (t * fps).round() / fps }
    }

    fn refresh_rows(&mut self) {
        self.rows = self
            .seq
            .tracks
            .iter()
            .enumerate()
            .flat_map(|(track, t)| t.channels().into_iter().map(move |channel| (track, channel)))
            .collect();
    }

    fn track(&self, index: usize) -> Option<&SeqTrack> {
        self.seq.tracks.get(index)
    }

    fn track_mut(&mut self, index: usize) -> Option<&mut SeqTrack> {
        self.seq.tracks.get_mut(index)
    }

    fn curve_mut(&mut self, track: usize, channel: SeqChannel) -> Option<&mut SeqCurve> {
        self.track_mut(track)?.curve_mut(channel)
    }

    /// Rebuild the originals map for every object the sequence touches.
    fn capture_originals(&mut self, scene: &LoadedScene) {
        let ids: Vec<ObjectId> = self.seq.tracks.iter().map(SeqTrack::target).collect();
        for id in ids {
            if self.originals.contains_key(&id) {
                continue;
            }
            let Some(&i) = scene.id_index().get(&id) else {
                continue;
            };
            let o = &scene.objects[i];
            let fov = o
                .components
                .iter()
                .find_map(|c| c.as_any().downcast_ref::<CameraComponent>())
                .map(|c| c.fov_deg);
            self.originals.insert(id, (o.translation, o.rotation, o.scale, fov));
        }
    }

    fn end_preview(&mut self, scene: &mut LoadedScene) {
        let ids = scene.id_index();
        for (id, (t, r, s, fov)) in self.originals.drain() {
            let Some(&i) = ids.get(&id) else {
                continue;
            };
            let o = &mut scene.objects[i];
            o.translation = t;
            o.rotation = r;
            o.scale = s;
            if let Some(fov) = fov
                && let Some(c) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<CameraComponent>())
            {
                c.fov_deg = fov;
            }
        }
        self.preview = false;
        self.playing = false;
    }

    /// Sample the sequence at the playhead and write it onto the scene.
    pub fn apply(&self, scene: &mut LoadedScene) {
        let t = self.seq.local_time(self.playhead, self.seq.looping);
        for track in &self.seq.tracks {
            let Some(&i) = scene.id_index().get(&track.target()) else {
                continue;
            };
            if i >= scene.objects.len() {
                continue;
            }
            let sample = |channel: SeqChannel| track.curve(channel).filter(|c| !c.keys.is_empty()).map(|c| c.sample(t));
            let o = &mut scene.objects[i];
            if let Some(v) = sample(SeqChannel::Translation) {
                o.translation = Vec3::from_array([v[0], v[1], v[2]]);
            }
            if let Some(v) = sample(SeqChannel::Rotation) {
                o.rotation = Quat::from_array(v).normalize();
            }
            if let Some(v) = sample(SeqChannel::Scale) {
                o.scale = Vec3::from_array([v[0], v[1], v[2]]);
            }
            if track.is_camera()
                && let Some(v) = sample(SeqChannel::Fov)
                && let Some(c) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<CameraComponent>())
            {
                c.fov_deg = v[0];
            }
        }
    }

    /// Advance (in play mode) and apply the current frame. The editor calls
    /// this after the play session ticks and before the draw build.
    pub fn tick(&mut self, scene: &mut LoadedScene, dt: f32) {
        if self.playing {
            self.playhead += dt.min(0.1);
            let len = self.seq.length.max(1e-3);
            if self.playhead > len {
                if self.seq.looping {
                    self.playhead = self.playhead.rem_euclid(len);
                } else {
                    self.playhead = len;
                    self.playing = false;
                }
            }
        }
        if self.preview {
            self.apply(scene);
        }
    }

    /// Enter play mode: drive the sequence from its start, capturing the
    /// touched objects for restore.
    pub fn begin_play(&mut self, scene: &LoadedScene) {
        self.playhead = 0.0;
        self.playing = true;
        self.preview = true;
        self.capture_originals(scene);
    }

    /// Leave play mode: restore what the sequence overwrote.
    pub fn end_play(&mut self, scene: &mut LoadedScene) {
        self.end_preview(scene);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, selected: Option<usize>, root: &Path, playing_game: bool) {
        let (before, path) = (self.seq.clone(), self.path.clone());
        self.refresh_rows();
        self.ui_inner(ui, scene, selected, root, playing_game);
        if self.path != path {
            self.history.clear();
        } else {
            self.history.observe(before, &self.seq, ui.input(|i| i.pointer.any_down()));
        }
    }

    pub fn history_step(&mut self, undo: bool) {
        if self.history.step(&mut self.seq, undo) {
            self.dirty = true;
            self.selected.clear();
        }
    }

    fn ui_inner(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, selected: Option<usize>, root: &Path, playing_game: bool) {
        let selected = selected.filter(|&i| i < scene.objects.len());
        if playing_game && self.preview && !self.playing {
            self.end_preview(scene);
        }
        let mut edited = false;

        egui::TopBottomPanel::top("zestengine-sequencer-toolbar")
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(12, 6)))
            .show_inside(ui, |ui| edited |= self.toolbar(ui, scene, selected, root, playing_game));

        if self.path.is_none() {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_VIEWPORT)).show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.3);
                    ui.label(egui::RichText::new("No sequence open").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                    ui.label(egui::RichText::new("Create a .seq to animate the scene over time.").color(theme::TEXT_MUTED));
                    ui.add_space(10.0);
                    if widgets::primary_button(ui, Some(Icon::Plus), "New Sequence").clicked() {
                        self.create(root);
                    }
                });
            });
            return;
        }

        egui::TopBottomPanel::bottom("zestengine-sequencer-keys")
            .resizable(true)
            .default_height(300.0)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(12, 8)))
            .show_inside(ui, |ui| edited |= self.key_inspector(ui));

        egui::SidePanel::left("zestengine-sequencer-tracks")
            .resizable(true)
            .default_width(260.0)
            .frame(egui::Frame::new().fill(theme::BG_PANEL))
            .show_inside(ui, |ui| edited |= self.track_list(ui, scene, selected));

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG_VIEWPORT))
            .show_inside(ui, |ui| edited |= self.timeline(ui, scene, selected));

        if self.playing && !playing_game {
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            self.playhead += dt;
            let len = self.seq.length.max(1e-3);
            if self.playhead > len {
                if self.seq.looping {
                    self.playhead = self.playhead.rem_euclid(len);
                } else {
                    self.playhead = len;
                    self.playing = false;
                }
            }
            ui.ctx().request_repaint();
        }
        if self.preview && !playing_game {
            self.apply(scene);
        }
        if edited {
            self.changed();
        }
        if self.dirty && !ui.input(|i| i.pointer.any_down()) && !ui.ctx().wants_keyboard_input() {
            self.save(root);
        }
    }

    fn create(&mut self, root: &Path) {
        let dir = root.join("sequences");
        let path = crate::anim_controller_editor::unique_path(&dir, "New Sequence", SEQUENCER_EXTENSION);
        match save_sequencer(&path, &Sequencer::default()) {
            Ok(()) => {
                let rel = crate::anim_controller_editor::rel_path(root, &path);
                self.open(root, &rel);
                self.open_request = true;
            }
            Err(e) => tracing::error!("creating sequencer: {e:#}"),
        }
    }

    fn add_track(&mut self, scene: &LoadedScene, selected: Option<usize>, camera: bool) {
        let target = selected
            .and_then(|i| scene.objects.get(i).map(|o| o.id))
            .or_else(|| main_camera_index(scene).map(|i| scene.objects[i].id));
        let Some(target) = target else {
            return;
        };
        let track = if camera {
            SeqTrack::new(SeqTrackKind::Camera, target)
        } else {
            SeqTrack::new(SeqTrackKind::Transform, target)
        };
        self.seq.tracks.push(track);
        self.changed();
    }

    fn toolbar(
        &mut self,
        ui: &mut egui::Ui,
        scene: &mut LoadedScene,
        _selected: Option<usize>,
        root: &Path,
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
                    ui.label(egui::RichText::new("No sequence").color(theme::TEXT_MUTED));
                }
            }
            if widgets::secondary_button(ui, Some(Icon::Plus), "New").clicked() {
                self.create(root);
            }
            let open = widgets::secondary_button(ui, Some(Icon::Folder), "Open");
            if open.clicked() {
                self.files = crate::anim_controller_editor::scan_files(root, SEQUENCER_EXTENSION);
            }
            egui::Popup::menu(&open).show(|ui| {
                if self.files.is_empty() {
                    ui.label(egui::RichText::new("No .seq files in the project").color(theme::TEXT_MUTED));
                }
                for f in self.files.clone() {
                    if ui.selectable_label(self.path.as_deref() == Some(f.as_str()), &f).clicked() {
                        self.open(root, &f);
                        ui.close();
                    }
                }
            });
            if self.path.is_none() {
                return;
            }
            ui.separator();
            ui.add_enabled_ui(!playing_game, |ui| {
                if widgets::icon_button_sized(ui, Icon::Eye, "Preview the sequence on the scene", self.preview, Vec2::splat(theme::H_FIELD))
                    .clicked()
                {
                    if self.preview {
                        self.end_preview(scene);
                    } else {
                        self.preview = true;
                        self.capture_originals(scene);
                    }
                }
                if ui.button(egui::RichText::new(ph::SKIP_BACK).size(14.0)).on_hover_text("Go to start").clicked() {
                    self.playhead = 0.0;
                }
                let (icon, tip) = if self.playing { (Icon::Pause, "Pause preview") } else { (Icon::Play, "Play preview") };
                if widgets::icon_button_sized(ui, icon, tip, self.playing, Vec2::splat(theme::H_FIELD)).clicked() {
                    self.playing = !self.playing;
                    self.preview |= self.playing;
                    if self.preview {
                        self.capture_originals(scene);
                    }
                }
            });
            ui.add(egui::DragValue::new(&mut self.playhead).speed(0.01).range(0.0..=1000.0).max_decimals(3).suffix(" s"))
                .on_hover_text("Playhead");
            ui.separator();
            ui.label(egui::RichText::new("Length").color(theme::TEXT_MUTED));
            edited |= ui.add(egui::DragValue::new(&mut self.seq.length).speed(0.01).range(0.01..=600.0).max_decimals(3).suffix(" s")).changed();
            edited |= ui.checkbox(&mut self.seq.looping, "Loop").changed();
            ui.label(egui::RichText::new("FPS").color(theme::TEXT_MUTED));
            edited |= ui.add(egui::DragValue::new(&mut self.seq.frame_rate).speed(1.0).range(1.0..=240.0)).changed();
        });
        edited
    }

    fn track_list(&mut self, ui: &mut egui::Ui, scene: &LoadedScene, selected: Option<usize>) -> bool {
        let mut edited = false;
        let rect = ui.max_rect();
        let header = Rect::from_min_size(rect.min, Vec2::new(rect.width(), RULER_H + 4.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(header.shrink2(Vec2::new(8.0, 2.0))), |ui| {
            ui.horizontal_centered(|ui| {
                let add = widgets::secondary_button_sized(ui, Some(Icon::Plus), "Add Track", theme::H_BUTTON_SMALL);
                egui::Popup::menu(&add).show(|ui| {
                    if ui.button("Transform Track").clicked() {
                        self.add_track(scene, selected, false);
                        edited = true;
                        ui.close();
                    }
                    if ui.button("Camera Track").clicked() {
                        self.add_track(scene, selected, true);
                        edited = true;
                        ui.close();
                    }
                    if self.seq.tracks.is_empty() {
                        ui.label(egui::RichText::new("Select an object first to target it").color(theme::TEXT_MUTED));
                    }
                });
            });
        });
        let body = Rect::from_min_max(Pos2::new(rect.left(), header.bottom()), rect.max);
        let painter = ui.painter_at(body);
        let mut remove = None;
        let mut toggle_fov = None;
        for (row, &(track, channel)) in self.rows.iter().enumerate() {
            let y = body.top() + row as f32 * ROW_H - self.vscroll;
            let rr = Rect::from_min_size(Pos2::new(body.left(), y), Vec2::new(body.width(), ROW_H));
            if rr.bottom() < body.top() || rr.top() > body.bottom() {
                continue;
            }
            if row % 2 == 1 {
                painter.rect_filled(rr, CornerRadius::ZERO, theme::with_alpha(theme::BG_APP, 0.5));
            }
            let is_first_channel = self.rows.iter().take(row).all(|&(t, _)| t != track);
            let name = self.track(track).and_then(|t| {
                scene.id_index().get(&t.target()).and_then(|&i| scene.objects.get(i)).map(|o| o.name.clone())
            });
            if is_first_channel {
                painter.rect_filled(Rect::from_min_size(Pos2::new(rr.left(), rr.top()), Vec2::new(4.0, ROW_H)), CornerRadius::ZERO, theme::accent());
            }
            let label = match (is_first_channel, name) {
                (true, Some(name)) => name.to_string(),
                (true, None) => "missing object".to_owned(),
                (false, _) => format!("  {}", channel.label()),
            };
            painter.text(
                Pos2::new(rr.left() + 14.0, rr.center().y),
                egui::Align2::LEFT_CENTER,
                label,
                egui::FontId::proportional(if is_first_channel { theme::SIZE_META } else { theme::SIZE_BODY }),
                if is_first_channel { theme::TEXT_STRONG } else { theme::TEXT_BODY },
            );
            let camera = self.track(track).is_some_and(|t| t.is_camera());
            if channel == SeqChannel::Translation
                && camera
            {
                let btn = Rect::from_center_size(Pos2::new(rr.right() - 16.0, rr.center().y), Vec2::splat(20.0));
                let active = self.track(track).and_then(|t| t.curve(SeqChannel::Fov)).is_some();
                let r = ui.put(btn, |ui: &mut egui::Ui| widgets::icon_button(ui, Icon::Camera, "Animate field of view", active));
                if r.clicked() {
                    toggle_fov = Some(track);
                }
            }
            let del_x = rr.right() - if is_first_channel && camera { 40.0 } else { 16.0 };
            let del = Rect::from_center_size(Pos2::new(del_x, rr.center().y), Vec2::splat(20.0));
            if is_first_channel && body.intersects(del) {
                let r = ui.put(del, |ui: &mut egui::Ui| widgets::icon_button(ui, Icon::Trash, "Remove track", false));
                if r.clicked() {
                    remove = Some(track);
                }
            }
        }
        if let Some(track) = remove {
            self.seq.tracks.remove(track);
            self.selected.clear();
            edited = true;
        }
        if let Some(track) = toggle_fov
            && let Some(t) = self.track_mut(track) {
                if t.curve(SeqChannel::Fov).is_some() {
                    t.remove_fov();
                } else {
                    t.ensure_fov();
                }
                edited = true;
            }
        if ui.rect_contains_pointer(body) {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            let max = (self.rows.len() as f32 * ROW_H - body.height()).max(0.0);
            self.vscroll = (self.vscroll - scroll).clamp(0.0, max);
        }
        if self.seq.tracks.is_empty() {
            painter.text(
                body.center_top() + Vec2::new(0.0, 24.0),
                egui::Align2::CENTER_TOP,
                "Add Track to animate a transform or camera",
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

    fn timeline(&mut self, ui: &mut egui::Ui, scene: &mut LoadedScene, _selected: Option<usize>) -> bool {
        let mut edited = false;
        let area = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(area, Sense::click_and_drag());
        let painter = ui.painter_at(area);
        let ruler = Rect::from_min_size(area.min, Vec2::new(area.width(), RULER_H + 4.0));
        let rows_rect = Rect::from_min_max(Pos2::new(area.left(), ruler.bottom()), area.max);
        let pointer = ui.input(|i| i.pointer.hover_pos());
        let (mods, scroll) = ui.input(|i| (i.modifiers, i.smooth_scroll_delta));

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

        let end_x = self.time_to_x(area, self.seq.length);
        if end_x < area.right() {
            painter.rect_filled(Rect::from_min_max(Pos2::new(end_x.max(area.left()), rows_rect.top()), area.max), CornerRadius::ZERO, theme::with_alpha(theme::BG_CHROME, 0.6));
        }

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
                painter.line_segment([Pos2::new(x, rows_rect.top()), Pos2::new(x, rows_rect.bottom())], Stroke::new(1.0_f32, theme::with_alpha(theme::LINE_PANEL, 0.7)));
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
        painter.line_segment([Pos2::new(end_x, ruler.top()), Pos2::new(end_x, rows_rect.bottom())], Stroke::new(1.0_f32, theme::TEXT_DIM));

        let mut hovered: Option<(usize, usize)> = None;
        for (row, &(track, channel)) in self.rows.iter().enumerate() {
            let Some(curve) = self.track(track).and_then(|t| t.curve(channel)) else {
                continue;
            };
            let y = rows_rect.top() + row as f32 * ROW_H - self.vscroll + ROW_H * 0.5;
            if y < rows_rect.top() - ROW_H || y > rows_rect.bottom() + ROW_H {
                continue;
            }
            if row % 2 == 1 {
                painter.rect_filled(Rect::from_min_size(Pos2::new(rows_rect.left(), y - ROW_H * 0.5), Vec2::new(rows_rect.width(), ROW_H)), CornerRadius::ZERO, theme::with_alpha(theme::BG_APP, 0.5));
            }
            for (j, key) in curve.keys.iter().enumerate() {
                let x = self.time_to_x(area, key.time);
                if x < area.left() - 8.0 || x > area.right() + 8.0 {
                    continue;
                }
                let c = Pos2::new(x, y);
                let hit = pointer.is_some_and(|p| (p - c).length() < 7.0 && rows_rect.contains(p));
                if hit {
                    hovered = Some((row, j));
                }
                let sel = self.selected.contains(&(row, j));
                let fill = if sel {
                    theme::accent()
                } else if hit {
                    theme::TEXT_STRONG
                } else {
                    theme::TEXT_SECONDARY
                };
                let s = 5.5;
                let pts = if key.interp == SeqInterp::Step {
                    vec![c + Vec2::new(-s, -s), c + Vec2::new(s, -s), c + Vec2::new(s, s), c + Vec2::new(-s, s)]
                } else {
                    vec![c + Vec2::new(0.0, -s), c + Vec2::new(s, 0.0), c + Vec2::new(0.0, s), c + Vec2::new(-s, 0.0)]
                };
                painter.add(Shape::convex_polygon(pts, fill, Stroke::new(1.0_f32, theme::BG_CHROME)));
            }
        }

        let free = mods.alt;
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            if ruler.contains(p) {
                self.scrubbing = true;
            } else if let Some(hk) = hovered {
                if !self.selected.contains(&hk) {
                    if !mods.ctrl && !mods.command {
                        self.selected.clear();
                    }
                    self.selected.push(hk);
                }
                let orig = self
                    .selected
                    .iter()
                    .filter_map(|&(row, k)| {
                        let (track, channel) = self.rows.get(row).copied()?;
                        self.track(track).and_then(|t| t.curve(channel)).and_then(|c| c.keys.get(k)).map(|key| (row, k, key.time))
                    })
                    .collect();
                self.drag = Some((self.x_to_time(area, p.x), orig));
            }
        }
        if response.dragged_by(egui::PointerButton::Primary)
            && let Some(p) = pointer
        {
            if self.scrubbing {
                self.playhead = self.snap(self.x_to_time(area, p.x), free);
                if !self.rows.is_empty() {
                    self.preview = true;
                    self.capture_originals(scene);
                }
            } else if let Some((start, orig)) = &self.drag {
                let delta = self.x_to_time(area, p.x) - start;
                let drags: Vec<(usize, usize, f32)> = orig.clone();
                for (row, k, time) in drags {
                    let new = self.snap(time + delta, free);
                    if let Some((track, channel)) = self.rows.get(row).copied()
                        && let Some(curve) = self.curve_mut(track, channel)
                        && let Some(key) = curve.keys.get_mut(k)
                    {
                        key.time = new;
                    }
                }
                edited = true;
            }
        }
        if response.drag_stopped() {
            self.scrubbing = false;
            if self.drag.take().is_some() {
                edited = true;
            }
        }
        if response.clicked_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            if ruler.contains(p) {
                self.playhead = self.snap(self.x_to_time(area, p.x), free);
                if !self.rows.is_empty() {
                    self.preview = true;
                    self.capture_originals(scene);
                }
            } else if let Some(hk) = hovered {
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
            && hovered.is_none()
            && let Some(p) = response.interact_pointer_pos()
            && rows_rect.contains(p)
        {
            let row = ((p.y - rows_rect.top() + self.vscroll) / ROW_H).floor() as usize;
            if let Some((track, channel)) = self.rows.get(row).copied() {
                let time = self.snap(self.x_to_time(area, p.x), free);
                let v = self.track(track)
                    .and_then(|t| t.curve(channel))
                    .map(|c| c.sample(time))
                    .unwrap_or([0.0; 4]);
                if let Some(curve) = self.curve_mut(track, channel) {
                    let j = curve.set_key(time, v);
                    self.selected = vec![(row, j)];
                    self.seq.length = self.seq.length.max(time);
                    edited = true;
                }
            }
        }
        response.context_menu(|ui| {
            crate::widgets::menu_frame(ui);
            if self.selected.is_empty() {
                ui.label(egui::RichText::new("Select keys first").color(theme::TEXT_MUTED));
                return;
            }
            for interp in SeqInterp::ALL {
                if ui.button(interp.label()).clicked() {
                    let keys: Vec<(usize, usize)> = self.selected.clone();
                    for (row, k) in keys {
                        if let Some((track, channel)) = self.rows.get(row).copied()
                            && let Some(curve) = self.curve_mut(track, channel)
                            && let Some(key) = curve.keys.get_mut(k)
                        {
                            key.interp = interp;
                        }
                    }
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

        if (response.hovered() || response.has_focus()) && !ui.ctx().wants_keyboard_input() {
            let del = ui.input(|i| i.key_pressed(egui::Key::Delete));
            if del && !self.selected.is_empty() {
                self.delete_selected();
                edited = true;
            }
        }

        let x = self.time_to_x(area, self.playhead);
        if x >= area.left() && x <= area.right() {
            painter.line_segment([Pos2::new(x, ruler.top()), Pos2::new(x, rows_rect.bottom())], Stroke::new(1.5_f32, theme::accent()));
            painter.add(Shape::convex_polygon(
                vec![Pos2::new(x - 5.0, ruler.top()), Pos2::new(x + 5.0, ruler.top()), Pos2::new(x, ruler.top() + 7.0)],
                theme::accent(),
                Stroke::NONE,
            ));
        }
        painter.text(
            area.left_bottom() + Vec2::new(10.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "Click ruler: scrub · Double-click row: add key · Drag keys · Del · Wheel: zoom · Shift+wheel: pan · Alt: no snap",
            egui::FontId::proportional(theme::SIZE_META),
            theme::TEXT_DIM,
        );
        if self.drag.is_some() || self.scrubbing {
            ui.ctx().request_repaint();
        }
        edited
    }

    fn delete_selected(&mut self) {
        let mut sel = self.selected.clone();
        sel.sort_by(|a, b| b.cmp(a));
        for (row, k) in sel {
            if let Some((track, channel)) = self.rows.get(row).copied()
                && let Some(curve) = self.curve_mut(track, channel)
                && k < curve.keys.len()
            {
                curve.keys.remove(k);
            }
        }
        self.selected.clear();
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
                        let (row, k) = self.selected[0];
                        let Some((track, channel)) = self.rows.get(row).copied() else {
                            return;
                        };
                        let is_camera = self.track(track).is_some_and(|t| t.is_camera());
                        let mut time = self.track(track)
                            .and_then(|t| t.curve(channel))
                            .and_then(|c| c.keys.get(k))
                            .map(|key| key.time);
                        let mut interp = self.track(track)
                            .and_then(|t| t.curve(channel))
                            .and_then(|c| c.keys.get(k))
                            .map(|key| key.interp);
                        let value = self.track(track)
                            .and_then(|t| t.curve(channel))
                            .and_then(|c| c.keys.get(k))
                            .map(|key| key.value);
                        widgets::eyebrow(ui, &format!("Key  {}", channel.label()));
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Time").color(theme::TEXT_MUTED));
                            if let Some(t) = &mut time
                                && ui.add(egui::DragValue::new(t).speed(0.01).range(0.0..=600.0).max_decimals(3).suffix(" s")).changed()
                            {
                                edited = true;
                            }
                            ui.label(egui::RichText::new("Curve").color(theme::TEXT_MUTED));
                            egui::ComboBox::from_id_salt("zestengine-seq-interp")
                                .selected_text(interp.map(|i| i.label()).unwrap_or(""))
                                .show_ui(ui, |ui| {
                                    for i in SeqInterp::ALL {
                                        if ui.selectable_label(interp == Some(i), i.label()).clicked() {
                                            interp = Some(i);
                                            edited = true;
                                        }
                                    }
                                });
                        });
                        if let Some(mut v) = value {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Value").color(theme::TEXT_MUTED));
                                match channel {
                                    SeqChannel::Rotation => {
                                        let mut e = quat_to_euler_deg(v);
                                        if widgets::vec3_field(ui, &mut e, 1, 0.5) {
                                            v = euler_deg_to_quat(e);
                                            edited = true;
                                        }
                                    }
                                    SeqChannel::Fov => {
                                        if ui.add(egui::DragValue::new(&mut v[0]).speed(0.1).range(1.0..=179.0).suffix(" deg")).changed() {
                                            edited = true;
                                        }
                                    }
                                    _ => {
                                        let mut e = [v[0], v[1], v[2]];
                                        if widgets::vec3_field(ui, &mut e, 3, 0.01) {
                                            v = [e[0], e[1], e[2], 0.0];
                                            edited = true;
                                        }
                                    }
                                }
                            });
                            if edited
                                && let Some(curve) = self.curve_mut(track, channel)
                                    && let Some(key) = curve.keys.get_mut(k)
                                {
                                    key.time = time.unwrap_or(key.time);
                                    key.interp = interp.unwrap_or(key.interp);
                                    key.value = v;
                                }
                        }
                        if is_camera && channel == SeqChannel::Translation {
                            ui.add_space(4.0);
                            let has_fov = self.track(track).and_then(|t| t.curve(SeqChannel::Fov)).is_some();
                            let mut animate_fov = has_fov;
                            if ui.checkbox(&mut animate_fov, "Animate field of view").changed()
                                && let Some(t) = self.track_mut(track) {
                                    if animate_fov {
                                        t.ensure_fov();
                                    } else {
                                        t.remove_fov();
                                    }
                                    edited = true;
                                }
                        }
                    }
                    n => {
                        widgets::eyebrow(ui, &format!("{n} keys"));
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Curve").color(theme::TEXT_MUTED));
                            for i in SeqInterp::ALL {
                                if ui.button(i.label()).clicked() {
                                    let keys: Vec<(usize, usize)> = self.selected.clone();
                                    for (row, k) in keys {
                                        if let Some((track, channel)) = self.rows.get(row).copied()
                                            && let Some(curve) = self.curve_mut(track, channel)
                                            && let Some(key) = curve.keys.get_mut(k)
                                        {
                                            key.interp = i;
                                        }
                                    }
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
                widgets::eyebrow(ui, "Tracks");
                if self.seq.tracks.is_empty() {
                    ui.label(egui::RichText::new("Add a track from the left panel.").size(theme::SIZE_META).color(theme::TEXT_DIM));
                }
                for (track, t) in self.seq.tracks.iter().enumerate() {
                    let kind = if t.is_camera() { "Camera" } else { "Transform" };
                    ui.label(egui::RichText::new(format!("{track}: {kind}")).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                }
            });
        });
        edited
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_engine::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    

    fn push_object(scene: &mut LoadedScene, id: ObjectId) {
        scene.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id,
            name: "cam".into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Camera,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: vec![Box::new(CameraComponent { fov_deg: 60.0, ..Default::default() })],
        });
    }

    #[test]
    fn apply_drives_transform_and_camera() {
        let target = ObjectId::new();
        let mut scene = LoadedScene::empty();
        push_object(&mut scene, target);

        let mut tab = SequencerTab::default();
        let mut track = SeqTrack::new(SeqTrackKind::Camera, target);
        track.translation.set_key(0.0, [0.0, 0.0, 0.0, 0.0]);
        track.translation.set_key(1.0, [3.0, 4.0, 5.0, 0.0]);
        if let Some(fov) = &mut track.fov {
            fov.set_key(0.0, [60.0, 0.0, 0.0, 0.0]);
            fov.set_key(1.0, [90.0, 0.0, 0.0, 0.0]);
        }
        tab.seq.length = 1.0;
        tab.seq.tracks.push(track);

        tab.playhead = 0.5;
        tab.apply(&mut scene);
        let o = &scene.objects[0];
        assert!((o.translation.x - 1.5).abs() < 1e-4);
        assert!((o.translation.y - 2.0).abs() < 1e-4);
        assert!((o.translation.z - 2.5).abs() < 1e-4);
        let fov = o.components.iter().find_map(|c| c.as_any().downcast_ref::<CameraComponent>()).unwrap().fov_deg;
        assert!((fov - 75.0).abs() < 1e-3);
    }
}
