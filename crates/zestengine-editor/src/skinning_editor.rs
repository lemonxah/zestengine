//! Skinning Editor window (2D-7): edits a `.rig2d` against the sprite sheet it
//! rigs. Opened by double-clicking a `.rig2d` in the Files panel
//! ([`request_open`]); the sheet is the `.png` with the same name next to it
//! (changeable in the window). The behaviour lives in
//! `zestengine_assets::rig2d_edit::SkinningModel` and `rig2d` (all unit
//! tested); this file is the egui shell and has not been seen on screen.
//!
//! Tools: **Select** (click a bone), **Create Bone** (drag head to tip; the
//! new bone is a child of the selected bone when it starts at that bone's
//! tip), **Paint** (brush add / subtract / smooth for the selected bone, one
//! undo step per stroke), **Weight Sliders** (click / Shift-click / drag a box to select
//! vertices, one slider per bone with the other influences renormalised, Copy / Paste Weights,
//! Mirror Weights across the pivot line with L / R bones swapped; each slider drag is one undo
//! step). Buttons: Auto Geometry (mesh the sprite's alpha),
//! Auto Weights, Save, Undo / Redo. The Weights overlay colours the mesh by the
//! selected bone's weight (blue 0 .. red 1).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use egui::{Color32, Pos2, Rect, Sense, Vec2};
use zestengine_assets::rig2d::{AutoWeightMode, AutoWeightParams, Brush, BrushMode, GeometryParams, Rig2D, load_rig2d, save_rig2d};
use zestengine_assets::rig2d_edit::SkinningModel;
use zestengine_assets::sprite::{SpriteDef, SpriteSheet, load_sprite_table};
use zestengine_assets::sprite_mesh::AlphaMap;
use crate::{theme, widgets};

static OPEN_REQUEST: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Ask the window to open a `.rig2d` (called by the Files panel).
pub fn request_open(path: PathBuf) {
    if let Ok(mut g) = OPEN_REQUEST.lock() {
        *g = Some(path);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    Select,
    CreateBone,
    Paint,
    /// Pick vertices and set their weights with per-bone sliders.
    Weights,
}

struct Loaded {
    alpha: AlphaMap,
    tex: egui::TextureHandle,
}

pub struct SkinningEditorWindow {
    open: bool,
    rig_path: Option<PathBuf>,
    sheet_path: String,
    model: Option<SkinningModel>,
    sheet: Option<SpriteSheet>,
    sprites: Vec<SpriteDef>,
    sprite_idx: usize,
    loaded: Option<Loaded>,
    tool: Tool,
    brush: Brush,
    geo: GeometryParams,
    weights: AutoWeightParams,
    show_weights: bool,
    show_mesh: bool,
    pan: Vec2,
    zoom: f32,
    drag_head: Option<[f32; 2]>,
    stroking: bool,
    /// Marquee start (screen) of a vertex box selection.
    marquee: Option<Pos2>,
    /// A slider drag is open (one undo entry per drag).
    slider_stroke: bool,
    /// Mirror from the right (x > axis) side to the left, or the reverse.
    mirror_from_right: bool,
    status: String,
}

impl Default for SkinningEditorWindow {
    fn default() -> Self {
        Self {
            open: false,
            rig_path: None,
            sheet_path: String::new(),
            model: None,
            sheet: None,
            sprites: Vec::new(),
            sprite_idx: 0,
            loaded: None,
            tool: Tool::Select,
            brush: Brush::default(),
            geo: GeometryParams::default(),
            weights: AutoWeightParams::default(),
            show_weights: true,
            show_mesh: true,
            pan: Vec2::ZERO,
            zoom: 3.0,
            drag_head: None,
            stroking: false,
            marquee: None,
            slider_stroke: false,
            mirror_from_right: true,
            status: String::new(),
        }
    }
}

fn col(c: Color32, a: f32) -> Color32 {
    theme::with_alpha(c, a)
}

impl SkinningEditorWindow {
    pub fn open(&mut self, rig: &Path) {
        let r = if rig.exists() { load_rig2d(rig) } else { Ok(Rig2D::default()) };
        match r {
            Ok(mut r) => {
                r.skeleton.ensure_ids();
                self.model = Some(SkinningModel::new(r));
                self.rig_path = Some(rig.to_path_buf());
                self.sheet_path = rig.with_extension("png").to_string_lossy().into_owned();
                self.status.clear();
                self.loaded = None;
                self.sprites.clear();
                self.open = true;
            }
            Err(e) => {
                self.status = format!("{e:#}");
                self.open = true;
            }
        }
    }

    fn load_sheet(&mut self, ctx: &egui::Context) {
        self.loaded = None;
        self.sprites.clear();
        let path = PathBuf::from(&self.sheet_path);
        let table = match load_sprite_table(&path) {
            Ok(Some(t)) => t,
            Ok(None) => {
                self.status = "The texture is not a Sprite texture".into();
                return;
            }
            Err(e) => {
                self.status = format!("{e:#}");
                return;
            }
        };
        match image::open(&path) {
            Ok(img) => {
                let img = img.to_rgba8();
                let alpha = AlphaMap::from_rgba8(img.width(), img.height(), img.as_raw());
                let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
                let tex = ctx.load_texture("skinning-editor", color, egui::TextureOptions::NEAREST);
                self.loaded = Some(Loaded { alpha, tex });
            }
            Err(e) => {
                self.status = format!("{e}");
                return;
            }
        }
        self.sprites = table.sprites;
        self.sheet = Some(table.sheet);
        self.sprite_idx = self.sprite_idx.min(self.sprites.len().saturating_sub(1));
        self.sync_sprite();
    }

    fn sync_sprite(&mut self) {
        if let (Some(m), Some(s)) = (self.model.as_mut(), self.sprites.get(self.sprite_idx)) {
            m.sprite_id = s.id.0;
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if let Some(p) = OPEN_REQUEST.lock().ok().and_then(|mut g| g.take()) {
            self.open(&p);
        }
        if !self.open {
            return;
        }
        if self.model.is_some() && self.loaded.is_none() && !self.sheet_path.is_empty() && Path::new(&self.sheet_path).exists() && self.status.is_empty() {
            self.load_sheet(ctx);
        }
        let mut open = true;
        egui::Window::new("Skinning Editor").id(egui::Id::new("skinning-editor-window")).open(&mut open).default_size([980.0, 640.0]).resizable(true).show(ctx, |ui| {
            self.contents(ui, ctx);
        });
        self.open = open;
    }

    fn contents(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(mut model) = self.model.take() else {
            ui.label(egui::RichText::new(if self.status.is_empty() { "No rig open".to_owned() } else { self.status.clone() }).color(theme::STATUS_ERROR));
            return;
        };
        // Toolbar.
        ui.horizontal(|ui| {
            if ui.button("Save").clicked()
                && let Some(p) = &self.rig_path {
                    match save_rig2d(p, &model.rig) {
                        Ok(()) => {
                            model.mark_saved();
                            self.status = "Saved".into();
                        }
                        Err(e) => self.status = format!("{e:#}"),
                    }
                }
            if ui.add_enabled(true, egui::Button::new("Undo")).clicked() {
                model.undo();
            }
            if ui.button("Redo").clicked() {
                model.redo();
            }
            ui.separator();
            for (t, label) in [(Tool::Select, "Select"), (Tool::CreateBone, "Create Bone"), (Tool::Paint, "Paint Weights"), (Tool::Weights, "Weight Sliders")] {
                if ui.selectable_label(self.tool == t, label).clicked() {
                    self.tool = t;
                }
            }
            ui.separator();
            ui.checkbox(&mut self.show_mesh, "Mesh");
            ui.checkbox(&mut self.show_weights, "Weights");
            if model.is_dirty() {
                ui.label(egui::RichText::new("unsaved").color(theme::STATUS_WARN_FG).size(theme::SIZE_META));
            }
            ui.label(egui::RichText::new(&self.status).color(theme::TEXT_DIM).size(theme::SIZE_META));
        });
        ui.horizontal(|ui| {
            ui.label("Sheet");
            let r = ui.add(egui::TextEdit::singleline(&mut self.sheet_path).desired_width(280.0));
            if r.lost_focus() && r.changed() || ui.button("Load").clicked() {
                self.status.clear();
                self.load_sheet(ctx);
            }
            if !self.sprites.is_empty() {
                let cur = self.sprites[self.sprite_idx].name.clone();
                egui::ComboBox::from_id_salt("skinning-sprite").selected_text(cur).show_ui(ui, |ui| {
                    for (i, s) in self.sprites.iter().enumerate() {
                        if ui.selectable_label(self.sprite_idx == i, &s.name).clicked() {
                            self.sprite_idx = i;
                        }
                    }
                });
                self.sync_sprite_for(&mut model);
            }
        });
        ui.separator();
        ui.horizontal_top(|ui| {
            // Left column: bones and actions.
            ui.vertical(|ui| {
                ui.set_width(220.0);
                self.side_panel(ui, &mut model);
            });
            ui.separator();
            self.canvas(ui, &mut model);
        });
        self.model = Some(model);
    }

    fn sync_sprite_for(&self, model: &mut SkinningModel) {
        if let Some(s) = self.sprites.get(self.sprite_idx) {
            model.sprite_id = s.id.0;
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, model: &mut SkinningModel) {
        ui.label(egui::RichText::new("Bones").strong());
        egui::ScrollArea::vertical().id_salt("skinning-bones").max_height(220.0).show(ui, |ui| {
            let n = model.rig.skeleton.bones.len();
            for i in 0..n {
                let b = &model.rig.skeleton.bones[i];
                let mut depth = 0;
                let mut p = b.parent;
                while p >= 0 && depth < 32 {
                    depth += 1;
                    p = model.rig.skeleton.bones[p as usize].parent;
                }
                let text = format!("{}{}", "  ".repeat(depth), b.name);
                if ui.selectable_label(model.active_bone == Some(i), text).clicked() {
                    model.active_bone = Some(i);
                }
            }
            if n == 0 {
                ui.label(egui::RichText::new("Use Create Bone and drag in the canvas").size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        });
        if let Some(i) = model.active_bone.filter(|&i| i < model.rig.skeleton.bones.len()) {
            let mut name = model.rig.skeleton.bones[i].name.clone();
            if ui.add(egui::TextEdit::singleline(&mut name).desired_width(200.0)).lost_focus() {
                model.rename_bone(i, &name);
            }
            ui.horizontal(|ui| {
                if ui.small_button("Split").clicked() {
                    model.split_bone(i, 0.5);
                }
                if ui.small_button("Unparent").clicked() {
                    model.reparent_bone(i, None);
                }
                if ui.small_button("Delete").clicked() {
                    model.delete_bone(i);
                }
            });
        }
        ui.separator();
        ui.label(egui::RichText::new("Geometry").strong());
        widgets::property_row(ui, "Spacing", |ui| {
            ui.add(egui::DragValue::new(&mut self.geo.spacing).speed(0.5).range(0.0..=256.0)).on_hover_text("Pixels between points; 0 = a tenth of the sprite");
        });
        widgets::property_row(ui, "Alpha >", |ui| {
            ui.add(egui::DragValue::new(&mut self.geo.alpha_threshold).range(0..=254));
        });
        if ui.button("Auto Geometry").on_hover_text("Mesh the sprite's alpha and bind it with Auto Weights").clicked() {
            self.auto_geometry(model);
        }
        ui.label(egui::RichText::new("Weights").strong());
        widgets::property_row(ui, "Mode", |ui| {
            egui::ComboBox::from_id_salt("skinning-wmode").selected_text(if self.weights.mode == AutoWeightMode::Heat { "Heat" } else { "Distance" }).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.weights.mode, AutoWeightMode::Heat, "Heat");
                ui.selectable_value(&mut self.weights.mode, AutoWeightMode::Distance, "Distance");
            });
        });
        widgets::property_row(ui, "Sharpness", |ui| {
            ui.add(egui::DragValue::new(&mut self.weights.sharpness).speed(0.05).range(0.5..=10.0));
        });
        if ui.button("Auto Weights").clicked() && !model.generate_weights(&self.weights) {
            self.status = "Generate geometry first".into();
        }
        if self.tool == Tool::Paint {
            ui.separator();
            ui.label(egui::RichText::new("Brush").strong());
            widgets::property_row(ui, "Mode", |ui| {
                for (m, l) in [(BrushMode::Add, "Add"), (BrushMode::Subtract, "Sub"), (BrushMode::Smooth, "Smooth")] {
                    ui.selectable_value(&mut self.brush.mode, m, l);
                }
            });
            widgets::property_row(ui, "Radius", |ui| {
                ui.add(egui::DragValue::new(&mut self.brush.radius).speed(0.005).range(0.01..=5.0));
            });
            widgets::property_row(ui, "Hardness", |ui| {
                ui.add(egui::Slider::new(&mut self.brush.hardness, 0.0..=1.0));
            });
            widgets::property_row(ui, "Strength", |ui| {
                ui.add(egui::Slider::new(&mut self.brush.strength, 0.01..=1.0));
            });
        }
        if self.tool == Tool::Weights {
            ui.separator();
            self.weight_sliders(ui, model);
        }
        let problems = model.rig.validate();
        for p in problems.iter().take(3) {
            ui.label(egui::RichText::new(p).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
    }

    /// Per-bone weight sliders for the selected vertices, copy / paste and mirror.
    fn weight_sliders(&mut self, ui: &mut egui::Ui, model: &mut SkinningModel) {
        ui.label(egui::RichText::new("Vertex Weights").strong());
        let count = model.selection.len();
        ui.label(egui::RichText::new(if count == 0 { "Click a vertex, Shift-click to add, drag a box to select".to_owned() } else { format!("{count} vertices selected (sliders show the mean)") }).size(theme::SIZE_META).color(theme::TEXT_DIM));
        ui.horizontal(|ui| {
            if ui.small_button("All").clicked() {
                let n = model.part().map_or(0, |p| p.mesh.positions.len() as u32);
                model.select_vertices(&(0..n).collect::<Vec<_>>(), false);
            }
            if ui.small_button("None").clicked() {
                model.select_vertices(&[], false);
            }
            if ui.add_enabled(count > 0, egui::Button::new("Copy").small()).on_hover_text("Remember the first selected vertex's weights").clicked() {
                self.status = if model.copy_weights() { "Weights copied".into() } else { "Select a vertex first".into() };
            }
            if ui.add_enabled(count > 0 && model.has_copied_weights(), egui::Button::new("Paste").small()).on_hover_text("Give every selected vertex the copied weights").clicked() {
                self.status = format!("Pasted onto {} vertices", model.paste_weights());
            }
        });
        let bones = model.usable_bones();
        let mut stroke_open = self.slider_stroke;
        egui::ScrollArea::vertical().id_salt("skinning-weight-sliders").max_height(220.0).show(ui, |ui| {
            for b in bones {
                let name = model.rig.skeleton.bones.get(b).map(|x| x.name.clone()).unwrap_or_default();
                let mut v = model.selection_weight(b);
                let r = ui.add_enabled(count > 0, egui::Slider::new(&mut v, 0.0..=1.0).text(name).fixed_decimals(2));
                if r.drag_started() || (r.changed() && !r.dragged() && !stroke_open) {
                    model.begin_stroke();
                    stroke_open = r.dragged();
                }
                if r.changed() {
                    model.active_bone = Some(b);
                    model.set_selection_weight(b, v);
                }
                if r.drag_stopped() {
                    stroke_open = false;
                }
            }
        });
        self.slider_stroke = stroke_open;
        ui.horizontal(|ui| {
            if ui.button("Mirror Weights").on_hover_text("Copy weights across the pivot line to the other side, swapping L / R bones; only selected vertices if any are selected").clicked() {
                let n = model.mirror_weights(0.0, self.mirror_from_right, 0.02);
                self.status = if n == 0 { "Nothing to mirror (no symmetric partner vertices)".into() } else { format!("Mirrored {n} vertices") };
            }
            ui.selectable_value(&mut self.mirror_from_right, true, "R > L");
            ui.selectable_value(&mut self.mirror_from_right, false, "L > R");
        });
    }

    fn auto_geometry(&mut self, model: &mut SkinningModel) {
        let (Some(l), Some(sheet), Some(def)) = (self.loaded.as_ref(), self.sheet.as_ref(), self.sprites.get(self.sprite_idx)) else {
            self.status = "Load a sprite sheet first".into();
            return;
        };
        if model.rig.skeleton.bones.is_empty() {
            self.status = "Create a bone first".into();
            return;
        }
        let ppu = def.effective_ppu(sheet);
        match model.generate_geometry(&l.alpha, def.rect, def.pivot_norm(), ppu, &def.name, &self.geo, &self.weights) {
            Some(n) => self.status = format!("{n} vertices"),
            None => self.status = "The sprite has no solid pixels".into(),
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui, model: &mut SkinningModel) {
        let size = ui.available_size().max(Vec2::splat(200.0));
        let (resp, painter) = ui.allocate_painter(size, Sense::click_and_drag());
        painter.rect_filled(resp.rect, 0.0, theme::BG_VIEWPORT);
        let ppu = match (self.sheet.as_ref(), self.sprites.get(self.sprite_idx)) {
            (Some(sh), Some(d)) => d.effective_ppu(sh),
            _ => 100.0,
        };
        // Scale: screen pixels per world unit.
        let scale = ppu * self.zoom;
        let origin = resp.rect.center() + self.pan;
        let to_screen = |p: [f32; 2]| Pos2::new(origin.x + p[0] * scale, origin.y - p[1] * scale);
        let to_world = |s: Pos2| [(s.x - origin.x) / scale, -(s.y - origin.y) / scale];
        // Zoom (cursor anchored) and pan (middle drag / two-finger scroll).
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let before = resp.hover_pos().map(to_world);
                self.zoom = (self.zoom * (1.0 + scroll * 0.002)).clamp(0.2, 60.0);
                if let (Some(b), Some(h)) = (before, resp.hover_pos()) {
                    let s2 = ppu * self.zoom;
                    let o2 = Pos2::new(h.x - b[0] * s2, h.y + b[1] * s2);
                    self.pan += o2 - origin;
                }
            }
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            self.pan += resp.drag_delta();
        }
        // The sprite image.
        if let (Some(l), Some(sh), Some(d)) = (self.loaded.as_ref(), self.sheet.as_ref(), self.sprites.get(self.sprite_idx)) {
            let pv = d.pivot_world(sh);
            let sz = d.size_world(sh);
            let (x0, y0) = (-pv[0], -pv[1]);
            let rect = Rect::from_min_max(to_screen([x0, y0 + sz[1]]), to_screen([x0 + sz[0], y0]));
            let (iw, ih) = (l.alpha.w as f32, l.alpha.h as f32);
            let uv = Rect::from_min_max(Pos2::new(d.rect.x as f32 / iw, d.rect.y as f32 / ih), Pos2::new((d.rect.x + d.rect.w) as f32 / iw, (d.rect.y + d.rect.h) as f32 / ih));
            painter.image(l.tex.id(), rect, uv, Color32::WHITE);
            painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0_f32, col(theme::TEXT_DIM, 0.4)), egui::StrokeKind::Outside);
        }
        // Mesh with the weight overlay.
        let active_w = model.active_bone.map(|b| model.weight_map(b)).unwrap_or_default();
        if let Some(part) = model.part() {
            let m = &part.mesh;
            if self.show_mesh {
                for t in m.indices.as_chunks::<3>().0 {
                    let pts: Vec<Pos2> = t.iter().map(|&i| to_screen(m.positions[i as usize])).collect();
                    if self.show_weights && !active_w.is_empty() {
                        let w = (active_w[t[0] as usize] + active_w[t[1] as usize] + active_w[t[2] as usize]) / 3.0;
                        let c = Color32::from_rgb((255.0 * w) as u8, (80.0 * (1.0 - (2.0 * w - 1.0).abs())) as u8, (255.0 * (1.0 - w)) as u8);
                        painter.add(egui::Shape::convex_polygon(pts.clone(), col(c, 0.45), egui::Stroke::NONE));
                    }
                    painter.add(egui::Shape::closed_line(pts, egui::Stroke::new(1.0_f32, col(Color32::WHITE, 0.35))));
                }
                for (i, p) in m.positions.iter().enumerate() {
                    if model.selection.contains(&(i as u32)) {
                        painter.circle_filled(to_screen(*p), 4.0, theme::accent());
                        painter.circle_stroke(to_screen(*p), 4.0, egui::Stroke::new(1.0_f32, Color32::WHITE));
                    } else {
                        painter.circle_filled(to_screen(*p), 2.0, col(Color32::WHITE, 0.6));
                    }
                }
            }
        }
        // Bones.
        let segs = model.rig.skeleton.rest_segments();
        for (i, (h, t)) in segs.iter().enumerate() {
            let on = model.active_bone == Some(i);
            let (a, b) = (to_screen(*h), to_screen(*t));
            let c = if on { theme::accent() } else { Color32::from_rgb(0xFF, 0xD0, 0x60) };
            let d = b - a;
            let n = if d.length() > 0.0 { Vec2::new(-d.y, d.x).normalized() * (d.length() * 0.12).clamp(2.0, 8.0) } else { Vec2::ZERO };
            let mid = a + d * 0.2;
            painter.add(egui::Shape::convex_polygon(vec![a, mid + n, b, mid - n], col(c, 0.55), egui::Stroke::new(1.5_f32, c)));
        }
        // Interaction.
        let ptr = resp.hover_pos().or(resp.interact_pointer_pos());
        match self.tool {
            Tool::Select => {
                if resp.clicked_by(egui::PointerButton::Primary)
                    && let Some(p) = ptr
                {
                    let w = to_world(p);
                    let near = segs
                        .iter()
                        .enumerate()
                        .map(|(i, (h, t))| (i, seg_distance(w, *h, *t) * scale))
                        .filter(|(_, d)| *d < 10.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1));
                    model.active_bone = near.map(|(i, _)| i);
                }
            }
            Tool::CreateBone => {
                if resp.drag_started_by(egui::PointerButton::Primary)
                    && let Some(p) = ptr
                {
                    self.drag_head = Some(to_world(p));
                }
                if let (Some(h), Some(p)) = (self.drag_head, ptr) {
                    painter.line_segment([to_screen(h), p], egui::Stroke::new(2.0_f32, theme::accent()));
                }
                if resp.drag_stopped()
                    && let (Some(h), Some(p)) = (self.drag_head.take(), ptr) {
                        let tip = to_world(p);
                        if ((tip[0] - h[0]).powi(2) + (tip[1] - h[1]).powi(2)).sqrt() * scale > 6.0 {
                            // A child of the selected bone when the drag starts at its tip.
                            let parent = model.active_bone.filter(|&i| i < segs.len() && ((segs[i].1[0] - h[0]).powi(2) + (segs[i].1[1] - h[1]).powi(2)).sqrt() * scale < 12.0);
                            model.create_bone(parent, h, tip);
                        }
                    }
            }
            Tool::Weights => {
                let shift = ui.input(|i| i.modifiers.shift);
                if resp.drag_started_by(egui::PointerButton::Primary) {
                    self.marquee = resp.interact_pointer_pos();
                }
                if let (Some(a), Some(b)) = (self.marquee, resp.interact_pointer_pos().or(ptr)) {
                    painter.rect_stroke(Rect::from_two_pos(a, b), 0.0, egui::Stroke::new(1.0_f32, theme::accent()), egui::StrokeKind::Inside);
                }
                if resp.drag_stopped() {
                    if let (Some(a), Some(b)) = (self.marquee.take(), resp.interact_pointer_pos().or(ptr)) {
                        let r = Rect::from_two_pos(a, b);
                        let inside: Vec<u32> = model.part().map_or(Vec::new(), |part| part.mesh.positions.iter().enumerate().filter(|(_, q)| r.contains(to_screen(**q))).map(|(i, _)| i as u32).collect());
                        model.select_vertices(&inside, shift);
                    }
                } else if resp.clicked_by(egui::PointerButton::Primary)
                    && let Some(p) = ptr
                {
                    let w = to_world(p);
                    match model.nearest_vertex(w, 12.0 / scale) {
                        Some(v) if shift => model.toggle_vertex(v),
                        Some(v) => model.select_vertices(&[v], false),
                        None if !shift => model.select_vertices(&[], false),
                        None => {}
                    }
                }
            }
            Tool::Paint => {
                if let Some(p) = ptr {
                    painter.circle_stroke(p, self.brush.radius * scale, egui::Stroke::new(1.0_f32, col(Color32::WHITE, 0.7)));
                }
                if resp.drag_started_by(egui::PointerButton::Primary) {
                    model.begin_stroke();
                    self.stroking = true;
                }
                if self.stroking && resp.dragged_by(egui::PointerButton::Primary)
                    && let Some(p) = resp.interact_pointer_pos() {
                        model.paint(to_world(p), &self.brush);
                    }
                if resp.drag_stopped() {
                    self.stroking = false;
                }
            }
        }
    }
}

fn seg_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (ab, ap) = ([b[0] - a[0], b[1] - a[1]], [p[0] - a[0], p[1] - a[1]]);
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if l2 > 1e-12 { ((ap[0] * ab[0] + ap[1] * ab[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    ((a[0] + ab[0] * t - p[0]).powi(2) + (a[1] + ab[1] * t - p[1]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_distance() {
        assert!((seg_distance([0.5, 1.0], [0.0, 0.0], [1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((seg_distance([2.0, 0.0], [0.0, 0.0], [1.0, 0.0]) - 1.0).abs() < 1e-6, "clamped past the end");
        assert!((seg_distance([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]) - 2f32.sqrt()).abs() < 1e-6, "zero-length bone");
    }

    #[test]
    fn open_request_is_taken_once() {
        request_open(PathBuf::from("x.rig2d"));
        assert_eq!(OPEN_REQUEST.lock().unwrap().take(), Some(PathBuf::from("x.rig2d")));
        assert!(OPEN_REQUEST.lock().unwrap().take().is_none());
    }
}

#[cfg(test)]
#[path = "shots/skinning_editor.rs"]
mod shots;
