//! The Sprite Editor window (2D-4), opened from the texture inspector's Sprite
//! section. It edits the same `TextureImport` draft the inspector's Apply /
//! Revert session owns (`FileMeta` in `editor_app.rs`), so there is one dirty
//! state: Apply asks the inspector to save and reimport, Revert returns to the
//! sheet as last applied, Revert All also throws away the inspector's other
//! unapplied edits.
//!
//! All decisions live in `zestengine_assets::{sprite_edit, sprite_slice}` (unit
//! tested, no egui); this file turns pointer events into those calls and draws
//! the result. The GUI has not been looked at by a human: the list of visual
//! checks is in Features.md.
//!
//! Undo is automatic: every frame in which the sheet changed pushes the sheet as
//! it was at the start of the frame, except while a pointer drag or a DragValue
//! drag continues (one entry per press). Whole-sheet snapshots, like Unity's.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use zestengine_assets::sprite::{PivotPreset, PixelRect, SliceMethod, SpriteDef, SpriteMode, SpriteSheet};
use zestengine_assets::sprite_atlas::AtlasDb;
use zestengine_assets::sprite_edit::{
    self as ed, BorderEdge, Handle, PointId, Selection, ShapeKind, SpriteEdit, View,
};
use zestengine_assets::sprite_mesh::{AlphaMap, P2};
use zestengine_assets::sprite_slice::{SliceSettings, SliceType, SlicePlan, apply_plan, plan_slice};
use zestengine_assets::{Platform, TextureImport, TextureKind};
use crate::{icons::Icon, theme, widgets};

// ---------------------------------------------------------------------------
// Atlas lookups for the inspector / Files panel badge
// ---------------------------------------------------------------------------

static ATLAS_ROOT: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);
static ATLAS_DB: std::sync::Mutex<Option<(std::time::Instant, Arc<AtlasDb>)>> = std::sync::Mutex::new(None);

/// Seconds a scanned [`AtlasDb`] is reused before the project is scanned again.
const ATLAS_DB_TTL: f64 = 5.0;

/// Tell the badge probe where the project is (call each frame, it is cheap).
pub fn set_atlas_root(root: &Path) {
    if let Ok(mut g) = ATLAS_ROOT.lock()
        && g.as_deref() != Some(root)
    {
        *g = Some(root.to_path_buf());
        if let Ok(mut db) = ATLAS_DB.lock() {
            *db = None;
        }
    }
}

fn atlas_db() -> Option<Arc<AtlasDb>> {
    let root = ATLAS_ROOT.lock().ok()?.clone()?;
    let mut g = ATLAS_DB.lock().ok()?;
    if let Some((at, db)) = g.as_ref()
        && at.elapsed().as_secs_f64() < ATLAS_DB_TTL
    {
        return Some(db.clone());
    }
    let db = Arc::new(AtlasDb::scan(&root));
    *g = Some((std::time::Instant::now(), db.clone()));
    Some(db)
}

/// Name of the atlas packing the texture at `abs`, for the "in atlas" badge.
pub fn atlas_name_probe(abs: &Path) -> Option<String> {
    let db = atlas_db()?;
    let rel = abs.strip_prefix(&db.root).ok()?.to_string_lossy().replace('\\', "/");
    db.atlas_name_of(&rel).map(str::to_owned)
}

/// Pack every atlas in the background (cached by content hash), so Play finds
/// them ready. Never blocks the UI; failures only reach the log.
pub fn warm_atlases(root: &Path) {
    let root = root.to_path_buf();
    std::thread::spawn(move || {
        let r = AtlasDb::scan(&root).cook_all(Platform::Desktop, false);
        for w in &r.warnings {
            tracing::warn!("sprite atlas: {w}");
        }
        if r.packed + r.from_cache > 0 {
            tracing::info!("sprite atlases ready: {} packed, {} from the cache", r.packed, r.from_cache);
        }
    });
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Sprites,
    Outline,
    Physics,
}

impl Mode {
    fn shape(self) -> Option<ShapeKind> {
        match self {
            Mode::Sprites => None,
            Mode::Outline => Some(ShapeKind::Outline),
            Mode::Physics => Some(ShapeKind::Physics),
        }
    }
}

struct Loaded {
    path: PathBuf,
    w: u32,
    h: u32,
    alpha: Arc<AlphaMap>,
    tex: egui::TextureHandle,
}

type LoadResult = (PathBuf, Result<(Arc<AlphaMap>, egui::ColorImage, u32, u32), String>);

/// The pointer gesture in progress. `orig` is the sheet when it began: each
/// frame re-applies the whole gesture to a copy, so a drag never accumulates
/// rounding and snapping stays exact.
enum Drag {
    None,
    Pan,
    Move { start: P2, orig: SpriteSheet },
    Resize { i: usize, handle: Handle, orig: SpriteSheet },
    Pivot { i: usize, orig: SpriteSheet },
    Border { i: usize, edge: BorderEdge, orig: SpriteSheet },
    Create { start: P2 },
    Marquee { start: P2 },
    Points { start: P2, ids: Vec<PointId>, orig: SpriteSheet },
    PointBox { start: P2 },
    Quad { start: P2 },
}

pub struct SpriteEditorWindow {
    open: bool,
    path: Option<PathBuf>,
    loaded: Option<Loaded>,
    job: Option<(PathBuf, Receiver<LoadResult>)>,
    error: Option<String>,
    edit: Option<SpriteEdit>,
    view: View,
    fitted: bool,
    mode: Mode,
    slice: SliceSettings,
    plan: Option<(SliceSettings, SlicePlan)>,
    snap_grid: u32,
    pivot_in_pixels: bool,
    drag: Drag,
    pt_sel: Vec<PointId>,
    outline_detail: f32,
    outline_alpha: u8,
    show_pieces: bool,
    /// A press that already pushed an undo entry is still held.
    coalescing: bool,
    status: String,
    req_apply: bool,
    req_revert_all: bool,
}

impl Default for SpriteEditorWindow {
    fn default() -> Self {
        Self {
            open: false,
            path: None,
            loaded: None,
            job: None,
            error: None,
            edit: None,
            view: View::default(),
            fitted: false,
            mode: Mode::Sprites,
            slice: SliceSettings::default(),
            plan: None,
            snap_grid: 1,
            pivot_in_pixels: false,
            drag: Drag::None,
            pt_sel: Vec::new(),
            outline_detail: 0.5,
            outline_alpha: 128,
            show_pieces: false,
            coalescing: false,
            status: String::new(),
            req_apply: false,
            req_revert_all: false,
        }
    }
}

/// What the window asks the inspector to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditorRequests {
    pub apply: bool,
    pub revert_all: bool,
}

fn load_image(path: &Path) -> Result<(Arc<AlphaMap>, egui::ColorImage, u32, u32), String> {
    let img = image::open(path).map_err(|e| e.to_string())?.to_rgba8();
    let (w, h) = (img.width(), img.height());
    let alpha = Arc::new(AlphaMap::from_rgba8(w, h, img.as_raw()));
    // Textures over 4096 are shown downscaled (the canvas maps UVs, not pixels).
    let shown = if w.max(h) > 4096 {
        let s = 4096.0 / w.max(h) as f32;
        image::imageops::resize(&img, ((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1), image::imageops::FilterType::Nearest)
    } else {
        img
    };
    Ok((alpha, egui::ColorImage::from_rgba_unmultiplied([shown.width() as usize, shown.height() as usize], shown.as_raw()), w, h))
}

fn color(c: egui::Color32, a: f32) -> egui::Color32 {
    theme::with_alpha(c, a)
}

impl SpriteEditorWindow {
    pub fn open(&mut self, path: &Path) {
        if self.path.as_deref() != Some(path) {
            self.loaded = None;
            self.job = None;
            self.edit = None;
            self.plan = None;
            self.fitted = false;
            self.drag = Drag::None;
            self.pt_sel.clear();
            self.error = None;
            self.path = Some(path.to_path_buf());
        }
        self.open = true;
    }

    /// Apply / Revert All requests since the last call.
    pub fn take_requests(&mut self) -> EditorRequests {
        EditorRequests { apply: std::mem::take(&mut self.req_apply), revert_all: std::mem::take(&mut self.req_revert_all) }
    }

    fn poll(&mut self, ctx: &egui::Context, path: &Path) {
        if self.loaded.as_ref().is_some_and(|l| l.path != path) {
            self.loaded = None;
        }
        if self.loaded.is_none() && self.job.as_ref().is_none_or(|(p, _)| p != path) {
            let (tx, rx) = std::sync::mpsc::channel();
            let p = path.to_path_buf();
            std::thread::spawn(move || {
                let r = load_image(&p);
                let _ = tx.send((p, r));
            });
            self.job = Some((path.to_path_buf(), rx));
            self.error = None;
        }
        if let Some((p, rx)) = &self.job {
            match rx.try_recv() {
                Ok((rp, r)) if rp == *p => {
                    match r {
                        Ok((alpha, color, w, h)) => {
                            let tex = ctx.load_texture("sprite-editor", color, egui::TextureOptions::NEAREST);
                            self.loaded = Some(Loaded { path: rp, w, h, alpha, tex });
                        }
                        Err(e) => self.error = Some(e),
                    }
                    self.job = None;
                }
                Ok(_) | Err(TryRecvError::Disconnected) => self.job = None,
                Err(TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(60)),
            }
        }
    }

    /// Show the window when it is open for `path` (the texture the inspector
    /// shows). `t` is the inspector's draft; `session_dirty` is its dirty flag.
    /// Returns true when the draft changed.
    pub fn show(&mut self, ctx: &egui::Context, path: &Path, t: &mut TextureImport, session_dirty: bool) -> bool {
        if !self.open {
            return false;
        }
        if self.path.as_deref() != Some(path) || t.kind != TextureKind::Sprite {
            self.open = false;
            return false;
        }
        self.poll(ctx, path);
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "sprite".into());
        let before = t.sprite.clone();
        let edit = self.edit.get_or_insert_with(|| SpriteEdit::new(&before));
        // The inspector applied or reverted behind our back: adopt its sheet.
        if !session_dirty && edit.is_dirty(&t.sprite) && !self.coalescing {
            edit.rebase(&t.sprite);
        }
        let mut open = true;
        let mut skip_undo = false;
        egui::Window::new("Sprite Editor").id(egui::Id::new("sprite-editor-window")).open(&mut open).default_size([1000.0, 660.0]).resizable(true).show(ctx, |ui| {
            skip_undo = self.contents(ui, &mut t.sprite, &stem, session_dirty);
        });
        self.open = open;
        let changed = t.sprite != before;
        let any_down = ctx.input(|i| i.pointer.any_down());
        if changed && !skip_undo {
            if !(any_down && self.coalescing)
                && let Some(e) = self.edit.as_mut() {
                    e.checkpoint(&before);
                }
            self.coalescing = any_down;
        }
        if !any_down {
            self.coalescing = false;
        }
        changed
    }

    // -----------------------------------------------------------------------

    /// The whole window body. Returns true when this frame's change was an
    /// undo / redo / revert (which must not push an undo entry of its own).
    fn contents(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, stem: &str, session_dirty: bool) -> bool {
        let mut skip_undo = false;
        let Some(loaded) = self.loaded.as_ref().map(|l| (l.w, l.h, l.alpha.clone())) else {
            match &self.error {
                Some(e) => {
                    ui.label(egui::RichText::new(format!("Could not load the texture: {e}")).color(theme::STATUS_ERROR));
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Loading pixels...");
                    });
                }
            }
            return false;
        };
        let (iw, ih, alpha) = loaded;
        if sheet.mode == SpriteMode::Single {
            // The one sprite covers the whole image.
            crate::sprite_inspector::single_def(sheet, iw, ih, stem);
        }
        skip_undo |= self.toolbar(ui, sheet, stem, session_dirty);
        ui.add_space(4.0);
        egui::SidePanel::right("sprite-editor-side").exact_width(theme::W_INSPECTOR).resizable(false).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                self.side_panel(ui, sheet, stem, iw, ih, &alpha);
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_VIEWPORT)).show_inside(ui, |ui| {
            self.canvas(ui, sheet, stem, iw, ih, &alpha);
        });
        // Keyboard.
        let ctx = ui.ctx().clone();
        let hovered = ui.rect_contains_pointer(ui.max_rect());
        if hovered && !ctx.wants_keyboard_input() {
            let (undo, redo, del, all, dup) = ctx.input(|i| {
                let cmd = i.modifiers.command;
                (
                    cmd && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                    (cmd && i.key_pressed(egui::Key::Z) && i.modifiers.shift) || (cmd && i.key_pressed(egui::Key::Y)),
                    i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                    cmd && i.key_pressed(egui::Key::A),
                    cmd && i.key_pressed(egui::Key::D),
                )
            });
            skip_undo |= self.run_keys(sheet, stem, iw, ih, undo, redo, del, all, dup);
        }
        skip_undo
    }

    #[allow(clippy::too_many_arguments)]
    fn run_keys(&mut self, sheet: &mut SpriteSheet, stem: &str, iw: u32, ih: u32, undo: bool, redo: bool, del: bool, all: bool, dup: bool) -> bool {
        let Some(edit) = self.edit.as_mut() else { return false };
        let multiple = sheet.mode == SpriteMode::Multiple;
        if undo {
            return edit.undo(sheet);
        }
        if redo {
            return edit.redo(sheet);
        }
        match self.mode {
            Mode::Sprites if multiple => {
                if del {
                    ed::delete_selected(sheet, &mut edit.sel);
                } else if all {
                    edit.sel.select_all(sheet.sprites.len());
                } else if dup {
                    ed::duplicate_selected(sheet, stem, &mut edit.sel, iw, ih);
                }
            }
            m => {
                if let (Some(kind), true) = (m.shape(), del)
                    && let Some(i) = edit.sel.primary()
                    && let Some(d) = def_mut(sheet, i)
                {
                    ed::delete_points(ed::shape_mut(d, kind), &self.pt_sel);
                    self.pt_sel.clear();
                }
            }
        }
        false
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, stem: &str, session_dirty: bool) -> bool {
        let mut skip = false;
        let dirty = session_dirty || self.edit.as_ref().is_some_and(|e| e.is_dirty(sheet));
        ui.horizontal(|ui| {
            widgets::segmented(ui, &mut self.mode, &[(Mode::Sprites, "Sprites"), (Mode::Outline, "Custom Outline"), (Mode::Physics, "Custom Physics Shape")]);
            ui.separator();
            let (can_undo, can_redo) = self.edit.as_ref().map(|e| (e.can_undo(), e.can_redo())).unwrap_or((false, false));
            ui.add_enabled_ui(can_undo, |ui| {
                if widgets::ghost_button(ui, "Undo", theme::H_FIELD).on_hover_text("Ctrl+Z").clicked()
                    && let Some(e) = self.edit.as_mut()
                {
                    skip |= e.undo(sheet);
                }
            });
            ui.add_enabled_ui(can_redo, |ui| {
                if widgets::ghost_button(ui, "Redo", theme::H_FIELD).on_hover_text("Ctrl+Shift+Z").clicked()
                    && let Some(e) = self.edit.as_mut()
                {
                    skip |= e.redo(sheet);
                }
            });
            ui.separator();
            ui.label(egui::RichText::new("Snap").size(theme::SIZE_META).color(theme::TEXT_DIM));
            widgets::segmented(ui, &mut self.snap_grid, &[(1, "1"), (2, "2"), (4, "4"), (8, "8"), (16, "16")]);
            ui.separator();
            ui.label(egui::RichText::new(format!("{:.0}%", self.view.zoom * 100.0)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_SECONDARY));
            if widgets::ghost_button(ui, "Fit", theme::H_FIELD).on_hover_text("Fit the texture in the window").clicked() {
                self.fitted = false;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(dirty, |ui| {
                    if widgets::primary_button(ui, Some(Icon::Check), "Apply").on_hover_text("Save the sprite table and reimport the texture").clicked() {
                        sheet.ensure_ids(stem);
                        if let Some(e) = self.edit.as_mut() {
                            e.mark_applied(sheet);
                        }
                        self.req_apply = true;
                    }
                    if widgets::secondary_button(ui, Some(Icon::Reset), "Revert All").on_hover_text("Discard every unapplied import edit, not only the sprite table").clicked() {
                        self.req_revert_all = true;
                    }
                    if widgets::secondary_button(ui, None, "Revert").on_hover_text("Go back to the sprite table as last applied").clicked()
                        && let Some(e) = self.edit.as_mut()
                    {
                        e.revert(sheet);
                        skip = true;
                    }
                });
                if dirty {
                    ui.label(egui::RichText::new("unapplied").small().weak());
                }
            });
        });
        skip
    }

    // -----------------------------------------------------------------------
    // Side panel
    // -----------------------------------------------------------------------

    fn side_panel(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, stem: &str, iw: u32, ih: u32, alpha: &AlphaMap) {
        let Some(sel) = self.edit.as_ref().map(|e| e.sel.clone()) else { return };
        let multiple = sheet.mode == SpriteMode::Multiple;
        if !multiple {
            widgets::info_note(ui, "Single mode: the whole texture is one sprite. Switch to Multiple to slice it into several.");
            if widgets::secondary_button(ui, None, "Switch to Multiple").clicked() {
                sheet.mode = SpriteMode::Multiple;
            }
            ui.add_space(6.0);
        }
        if self.mode == Mode::Sprites && multiple {
            self.slice_panel(ui, sheet, stem, iw, ih, alpha);
            ui.add_space(6.0);
            widgets::eyebrow(ui, "Sprites");
            let mut clicked: Option<(usize, bool)> = None;
            egui::ScrollArea::vertical().id_salt("sprite-editor-list").max_height(150.0).show(ui, |ui| {
                for (i, d) in sheet.sprites.iter().enumerate() {
                    let r = ui.selectable_label(sel.contains(i), format!("{i}  {}", d.name));
                    if r.clicked() {
                        clicked = Some((i, ui.input(|i| i.modifiers.shift || i.modifiers.command)));
                    }
                }
            });
            if let Some((i, add)) = clicked
                && let Some(e) = self.edit.as_mut()
            {
                e.sel.click(i, add);
            }
            ui.add_space(6.0);
        }
        let sel = self.edit.as_ref().map(|e| e.sel.clone()).unwrap_or_default();
        let target = if multiple { sel.primary() } else { Some(0) };
        let count = if multiple { sheet.sprites.len() } else { 1 };
        let Some(i) = target.filter(|&i| i < count) else {
            widgets::info_note(ui, "Select a sprite on the canvas or in the list.");
            return;
        };
        match self.mode {
            Mode::Sprites => self.sprite_props(ui, sheet, stem, i, &sel, iw, ih),
            Mode::Outline => self.shape_panel(ui, sheet, i, ShapeKind::Outline, alpha),
            Mode::Physics => self.shape_panel(ui, sheet, i, ShapeKind::Physics, alpha),
        }
        if !self.status.is_empty() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(&self.status).size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
    }

    fn slice_panel(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, stem: &str, iw: u32, ih: u32, alpha: &AlphaMap) {
        widgets::eyebrow(ui, "Slice");
        let s = &mut self.slice;
        let types: Vec<(SliceType, &str)> = SliceType::ALL.iter().map(|t| (*t, t.label())).collect();
        widgets::property_row(ui, "Type", |ui| {
            egui::ComboBox::from_id_salt("slice-type").width(ui.available_width()).selected_text(s.ty.label()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for (t, l) in &types {
                    if ui.selectable_label(s.ty == *t, *l).clicked() {
                        s.ty = *t;
                    }
                }
            });
        });
        let pair = |ui: &mut egui::Ui, label: &str, v: &mut [u32; 2], max: [u32; 2]| {
            widgets::property_row(ui, label, |ui| {
                ui.add(egui::DragValue::new(&mut v[0]).range(0..=max[0]).speed(0.5));
                ui.add(egui::DragValue::new(&mut v[1]).range(0..=max[1]).speed(0.5));
            });
        };
        match s.ty {
            SliceType::Automatic => {
                widgets::property_row(ui, "Minimum size", |ui| {
                    ui.add(egui::DragValue::new(&mut s.min_size).range(1..=iw.max(ih)).speed(0.25));
                });
            }
            SliceType::GridByCellSize => {
                pair(ui, "Pixel size", &mut s.cell, [iw.max(1), ih.max(1)]);
                pair(ui, "Offset", &mut s.offset, [iw, ih]);
                pair(ui, "Padding", &mut s.padding, [iw, ih]);
            }
            SliceType::GridByCellCount => {
                pair(ui, "Columns, rows", &mut s.count, [4096, 4096]);
                pair(ui, "Offset", &mut s.offset, [iw, ih]);
                pair(ui, "Padding", &mut s.padding, [iw, ih]);
            }
            SliceType::Isometric => {
                pair(ui, "Pixel size", &mut s.cell, [iw.max(2), ih.max(2)]);
                pair(ui, "Offset", &mut s.offset, [iw, ih]);
                widgets::property_row(ui, "Is alternate", |ui| {
                    ui.checkbox(&mut s.alternate, "");
                });
            }
        }
        if s.ty != SliceType::Automatic {
            widgets::property_row(ui, "Keep empty rects", |ui| {
                ui.checkbox(&mut s.keep_empty, "");
            });
        }
        let presets: Vec<(PivotPreset, &str)> = PivotPreset::ALL.iter().map(|p| (*p, p.label())).collect();
        widgets::property_row(ui, "Pivot", |ui| {
            egui::ComboBox::from_id_salt("slice-pivot").width(ui.available_width()).selected_text(s.pivot.label()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for (p, l) in &presets {
                    if ui.selectable_label(s.pivot == *p, *l).clicked() {
                        s.pivot = *p;
                    }
                }
            });
        });
        if s.pivot == PivotPreset::Custom {
            widgets::property_row(ui, "Custom pivot", |ui| {
                ui.add(egui::DragValue::new(&mut s.custom_pivot[0]).speed(0.005).fixed_decimals(3));
                ui.add(egui::DragValue::new(&mut s.custom_pivot[1]).speed(0.005).fixed_decimals(3));
            });
        }
        widgets::property_row(ui, "Existing sprites", |ui| {
            widgets::segmented(ui, &mut s.method, &[(SliceMethod::Smart, "Smart"), (SliceMethod::Safe, "Safe"), (SliceMethod::DeleteExisting, "Delete existing")]);
        });
        // Grid and isometric previews follow the settings; Automatic runs on demand.
        let threshold = sheet.alpha_tolerance;
        let settings = self.slice.clone();
        if settings.ty != SliceType::Automatic && self.plan.as_ref().is_none_or(|(k, _)| *k != settings) {
            self.plan = Some((settings.clone(), plan_slice(&settings, iw, ih, Some(alpha), threshold)));
        }
        let mut run = false;
        ui.horizontal(|ui| {
            if settings.ty == SliceType::Automatic && widgets::secondary_button(ui, None, "Preview").on_hover_text("Find the islands of solid pixels and show them in red").clicked() {
                self.plan = Some((settings.clone(), plan_slice(&settings, iw, ih, Some(alpha), threshold)));
            }
            if widgets::primary_button(ui, Some(Icon::Grid), "Slice").clicked() {
                run = true;
            }
        });
        if run {
            let plan = plan_slice(&settings, iw, ih, Some(alpha), threshold);
            let rep = apply_plan(sheet, stem, &plan, &settings);
            self.status = format!("Sliced into {} sprite(s): {} kept their identity, {} new", plan.rects.len(), rep.kept, rep.created.len());
            self.plan = None;
            if let Some(e) = self.edit.as_mut() {
                e.sel.clear();
            }
        } else if let Some((_, p)) = &self.plan {
            ui.label(egui::RichText::new(format!("{} rect(s) previewed in red", p.rects.len())).size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sprite_props(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, stem: &str, i: usize, sel: &Selection, iw: u32, ih: u32) {
        let multiple = sheet.mode == SpriteMode::Multiple;
        widgets::eyebrow(ui, if sel.items.len() > 1 { "Selected sprites (editing the first)" } else { "Sprite" });
        if multiple {
            // Name: edited in a buffer and committed on focus loss, so a half-typed
            // or duplicate name never reaches the table.
            let id = egui::Id::new(("sprite-editor-name", sheet.sprites[i].id.0));
            let mut buf = ui.data_mut(|m| m.get_temp::<String>(id)).unwrap_or_else(|| sheet.sprites[i].name.clone());
            let name_resp = widgets::property_row(ui, "Name", |ui| ui.add(egui::TextEdit::singleline(&mut buf).desired_width(ui.available_width())));
            if name_resp.lost_focus() {
                if buf != sheet.sprites[i].name && !sheet.rename_sprite(i, &buf) {
                    self.status = format!("'{}' is empty or already used", buf.trim());
                }
                ui.data_mut(|m| m.remove_temp::<String>(id));
            } else if name_resp.has_focus() {
                ui.data_mut(|m| m.insert_temp(id, buf));
            }
            // Position and size.
            let mut r = sheet.sprites[i].rect;
            let mut changed = false;
            for (label, fields) in [("Position", [crate::sprite_inspector::RectField::X, crate::sprite_inspector::RectField::Y]), ("Size", [crate::sprite_inspector::RectField::W, crate::sprite_inspector::RectField::H])] {
                widgets::property_row(ui, label, |ui| {
                    for f in fields {
                        let mut v = match f {
                            crate::sprite_inspector::RectField::X => r.x,
                            crate::sprite_inspector::RectField::Y => r.y,
                            crate::sprite_inspector::RectField::W => r.w,
                            crate::sprite_inspector::RectField::H => r.h,
                        };
                        if ui.add(egui::DragValue::new(&mut v).range(0..=iw.max(ih)).speed(0.5)).changed() {
                            crate::sprite_inspector::set_rect_field(&mut r, f, v, iw, ih);
                            changed = true;
                        }
                    }
                });
            }
            if changed {
                sheet.sprites[i].rect = r;
                sheet.sprites[i].border = sheet.sprites[i].border_clamped();
            }
        }
        // Pivot.
        let presets: Vec<(PivotPreset, &str)> = PivotPreset::ALL.iter().map(|p| (*p, p.label())).collect();
        let items: Vec<usize> = if multiple { sel.items.clone() } else { vec![0] };
        let current = def_ref(sheet, i).map(|d| d.pivot_preset).unwrap_or_default();
        widgets::property_row(ui, "Pivot", |ui| {
            egui::ComboBox::from_id_salt("sprite-editor-pivot").width(ui.available_width()).selected_text(current.label()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for (p, l) in &presets {
                    if ui.selectable_label(current == *p, *l).clicked() {
                        for &k in &items {
                            if let Some(d) = def_mut(sheet, k) {
                                match p.normalized() {
                                    Some(n) => {
                                        d.pivot = n;
                                        d.pivot_preset = *p;
                                    }
                                    None => d.pivot_preset = PivotPreset::Custom,
                                }
                            }
                        }
                    }
                }
            });
        });
        if current == PivotPreset::Custom {
            widgets::property_row(ui, "", |ui| {
                widgets::segmented(ui, &mut self.pivot_in_pixels, &[(false, "Normalized"), (true, "Pixels")]);
            });
            if let Some(d) = def_mut(sheet, i) {
                widgets::property_row(ui, "Custom pivot", |ui| {
                    if self.pivot_in_pixels {
                        let mut p = d.pivot_pixels();
                        let a = ui.add(egui::DragValue::new(&mut p[0]).speed(0.25)).changed();
                        let b = ui.add(egui::DragValue::new(&mut p[1]).speed(0.25)).changed();
                        if a | b {
                            d.pivot = [p[0] / d.rect.w.max(1) as f32, p[1] / d.rect.h.max(1) as f32];
                        }
                    } else {
                        ui.add(egui::DragValue::new(&mut d.pivot[0]).speed(0.005).fixed_decimals(3));
                        ui.add(egui::DragValue::new(&mut d.pivot[1]).speed(0.005).fixed_decimals(3));
                    }
                });
            }
        }
        // Border (applies to every selected sprite).
        if let Some(d) = def_ref(sheet, i) {
            let mut b = d.border;
            let mut any = false;
            widgets::property_row(ui, "Border L B R T", |ui| {
                for (k, v) in b.iter_mut().enumerate() {
                    let max = if k % 2 == 0 { d.rect.w } else { d.rect.h };
                    any |= ui.add(egui::DragValue::new(v).range(0..=max).speed(0.5)).changed();
                }
            });
            if any {
                for &k in &items {
                    if let Some(d) = def_mut(sheet, k) {
                        d.border = b;
                        d.border = d.border_clamped();
                    }
                }
            }
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if multiple {
                if widgets::secondary_button(ui, Some(Icon::Duplicate), "Duplicate").on_hover_text("Ctrl+D").clicked()
                    && let Some(e) = self.edit.as_mut()
                {
                    ed::duplicate_selected(sheet, stem, &mut e.sel, iw, ih);
                }
                if widgets::icon_button(ui, Icon::Trash, "Delete the selected sprites (Delete). Their ids stay reserved.", false).clicked()
                    && let Some(e) = self.edit.as_mut()
                {
                    ed::delete_selected(sheet, &mut e.sel);
                }
            }
        });
        if let Some(d) = def_ref(sheet, i) {
            ui.label(egui::RichText::new(crate::sprite_inspector::sprite_summary(d, sheet)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
        }
    }

    fn shape_panel(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, i: usize, kind: ShapeKind, alpha: &AlphaMap) {
        let Some(rect) = def_ref(sheet, i).map(|d| d.rect) else { return };
        let (title, hint) = match kind {
            ShapeKind::Outline => ("Custom outline", "The render mesh follows this outline instead of tracing the alpha."),
            ShapeKind::Physics => ("Custom physics shape", "The collider uses this shape instead of the generated one."),
        };
        widgets::eyebrow(ui, title);
        widgets::property_row(ui, "Detail", |ui| {
            widgets::slider(ui, egui::Slider::new(&mut self.outline_detail, 0.0..=1.0));
        });
        widgets::property_row(ui, "Alpha tolerance", |ui| {
            let mut a = self.outline_alpha as i32;
            if widgets::slider(ui, egui::Slider::new(&mut a, 0..=254)).changed() {
                self.outline_alpha = a as u8;
            }
        });
        let gen_one = |sheet: &mut SpriteSheet, k: usize, detail: f32, tol: u8| {
            if let Some(d) = def_mut(sheet, k) {
                let loops = match kind {
                    ShapeKind::Outline => ed::generate_outline(alpha, d.rect, tol, detail),
                    ShapeKind::Physics => ed::generate_physics(alpha, d.rect, tol, detail),
                };
                *ed::shape_mut(d, kind) = loops;
            }
        };
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, None, "Generate").on_hover_text("Trace the alpha of this sprite").clicked() {
                gen_one(sheet, i, self.outline_detail, self.outline_alpha);
                self.pt_sel.clear();
            }
            if widgets::secondary_button(ui, None, "Generate all").on_hover_text("Trace every sprite").clicked() {
                let n = if sheet.mode == SpriteMode::Multiple { sheet.sprites.len() } else { 1 };
                for k in 0..n {
                    gen_one(sheet, k, self.outline_detail, self.outline_alpha);
                }
                self.pt_sel.clear();
            }
            if widgets::ghost_button(ui, "Clear", theme::H_FIELD).on_hover_text("Go back to the automatic shape").clicked() {
                if let Some(d) = def_mut(sheet, i) {
                    ed::shape_mut(d, kind).clear();
                }
                self.pt_sel.clear();
            }
        });
        let other = if kind == ShapeKind::Outline { ShapeKind::Physics } else { ShapeKind::Outline };
        if widgets::ghost_button(ui, if kind == ShapeKind::Outline { "Copy to physics shape" } else { "Copy to outline" }, theme::H_FIELD).clicked()
            && let Some(d) = def_mut(sheet, i)
        {
            ed::copy_shape(d, kind, other);
        }
        if kind == ShapeKind::Physics {
            widgets::property_row(ui, "Convex pieces", |ui| {
                ui.checkbox(&mut self.show_pieces, "");
            });
        }
        let Some(d) = def_ref(sheet, i) else { return };
        let loops = ed::shape(d, kind);
        ui.label(egui::RichText::new(format!("{} path(s), {} point(s)", loops.len(), loops.iter().map(Vec::len).sum::<usize>())).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
        for p in ed::shape_problems(loops) {
            ui.label(egui::RichText::new(p).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        ui.label(egui::RichText::new(format!("{hint} Click an edge to add a point, drag points, Delete removes. Shift+drag selects points; dragging inside the sprite with nothing selected draws a quad. Rect {}x{}.", rect.w, rect.h)).size(theme::SIZE_META).color(theme::TEXT_DIM));
    }

    // -----------------------------------------------------------------------
    // Canvas
    // -----------------------------------------------------------------------

    fn canvas(&mut self, ui: &mut egui::Ui, sheet: &mut SpriteSheet, _stem: &str, iw: u32, ih: u32, _alpha: &AlphaMap) {
        let (resp, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
        let area = resp.rect;
        if !self.fitted {
            self.view = View::fit([area.width(), area.height()], [iw, ih]);
            self.fitted = true;
        }
        let multiple = sheet.mode == SpriteMode::Multiple;
        let origin = area.min;
        let to_screen = |v: &View, p: P2| -> egui::Pos2 {
            let c = v.to_canvas(p);
            egui::pos2(origin.x + c[0], origin.y + c[1])
        };
        let to_img = |v: &View, s: egui::Pos2| -> P2 { v.to_image([s.x - origin.x, s.y - origin.y]) };

        // Zoom (cursor anchored) and pan (middle / right drag, or Alt + left drag).
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0
                && let Some(p) = ui.input(|i| i.pointer.hover_pos())
            {
                self.view.zoom_at([p.x - origin.x, p.y - origin.y], (scroll * 0.002).exp());
            }
        }
        let (alt, shift, ctrl) = ui.input(|i| (i.modifiers.alt, i.modifiers.shift, i.modifiers.command));
        if resp.dragged_by(egui::PointerButton::Middle) || resp.dragged_by(egui::PointerButton::Secondary) || (alt && resp.dragged_by(egui::PointerButton::Primary)) {
            self.view.pan_by([resp.drag_delta().x, resp.drag_delta().y]);
            self.drag = Drag::Pan;
        }

        let view = self.view;
        let mut painter = painter;
        painter.rect_filled(area, 0.0, theme::BG_VIEWPORT);
        let img = egui::Rect::from_min_max(to_screen(&view, [0.0, 0.0]), to_screen(&view, [iw as f32, ih as f32]));
        let visible = img.intersect(area);
        paint_checker(&painter, visible, img.min);
        if let Some(l) = &self.loaded {
            painter.with_clip_rect(area).image(l.tex.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        }
        painter.rect_stroke(img, 0.0, egui::Stroke::new(1.0_f32, theme::LINE_BUTTON), egui::StrokeKind::Outside);
        painter = painter.with_clip_rect(area);

        let sel = self.edit.as_ref().map(|e| e.sel.clone()).unwrap_or_default();
        let defs: Vec<SpriteDef> = if multiple { sheet.sprites.clone() } else { vec![sheet.single.clone()] };
        let accent = theme::accent();
        let rect_screen = |r: PixelRect| egui::Rect::from_min_max(to_screen(&view, [r.x as f32, r.y as f32]), to_screen(&view, [(r.x + r.w) as f32, (r.y + r.h) as f32]));

        // Sprite rects (and names when there is room).
        for (i, d) in defs.iter().enumerate() {
            let selected = if multiple { sel.contains(i) } else { true };
            let r = rect_screen(d.rect);
            let (c, w) = if selected { (accent, 1.5_f32) } else { (color(theme::TEXT_STRONG, 0.35), 1.0_f32) };
            painter.rect_stroke(r, 0.0, egui::Stroke::new(w, c), egui::StrokeKind::Inside);
            if multiple && r.width() > 56.0 && r.height() > 16.0 {
                painter.text(r.min + egui::vec2(3.0, 2.0), egui::Align2::LEFT_TOP, &d.name, theme::mono(theme::SIZE_TINY), color(theme::TEXT_STRONG, 0.8));
            }
        }
        // The pending slice, in red.
        if self.mode == Mode::Sprites
            && multiple
            && let Some((_, plan)) = &self.plan
        {
            let red = color(theme::STATUS_ERROR, 0.85);
            for r in &plan.rects {
                painter.rect_stroke(rect_screen(*r), 0.0, egui::Stroke::new(1.0_f32, red), egui::StrokeKind::Inside);
            }
            for c in &plan.iso {
                let pts: Vec<egui::Pos2> = c.outline.iter().map(|p| to_screen(&view, [c.rect.x as f32 + p[0], c.rect.y as f32 + p[1]])).collect();
                painter.add(egui::Shape::closed_line(pts, egui::Stroke::new(1.0_f32, red)));
            }
        }

        let primary = if multiple { sel.primary() } else { Some(0) }.filter(|&i| i < defs.len());
        match self.mode {
            Mode::Sprites => self.sprites_mode(ui, &resp, sheet, &defs, primary, &sel, iw, ih, &view, &painter, origin, (shift, ctrl, alt), multiple),
            m => {
                if let Some(kind) = m.shape() {
                    self.shape_mode(ui, &resp, sheet, &defs, primary, kind, &view, &painter, origin, (shift, ctrl, alt), multiple, iw, ih);
                }
            }
        }
        // Marquee / create previews.
        let cur = ui.input(|i| i.pointer.hover_pos()).map(|p| to_img(&view, p));
        if let (Some(c), true) = (cur, resp.dragged()) {
            match &self.drag {
                Drag::Create { start } | Drag::Quad { start } => {
                    let r = egui::Rect::from_two_pos(to_screen(&view, *start), to_screen(&view, c));
                    painter.rect_stroke(r, 0.0, egui::Stroke::new(1.0_f32, accent), egui::StrokeKind::Middle);
                }
                Drag::Marquee { start } | Drag::PointBox { start } => {
                    let r = egui::Rect::from_two_pos(to_screen(&view, *start), to_screen(&view, c));
                    painter.rect_filled(r, 0.0, color(accent, 0.12));
                    painter.rect_stroke(r, 0.0, egui::Stroke::new(1.0_f32, color(accent, 0.7)), egui::StrokeKind::Middle);
                }
                _ => {}
            }
        }
        if resp.drag_stopped() || (!resp.dragged() && !matches!(self.drag, Drag::None)) {
            self.finish_drag(ui, sheet, &view, origin, iw, ih);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sprites_mode(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        sheet: &mut SpriteSheet,
        defs: &[SpriteDef],
        primary: Option<usize>,
        sel: &Selection,
        iw: u32,
        ih: u32,
        view: &View,
        painter: &egui::Painter,
        origin: egui::Pos2,
        mods: (bool, bool, bool),
        multiple: bool,
    ) {
        let mut sel = sel.clone();
        let (shift, ctrl, alt) = mods;
        let to_img = |s: egui::Pos2| -> P2 { view.to_image([s.x - origin.x, s.y - origin.y]) };
        let to_canvas = |s: egui::Pos2| -> P2 { [s.x - origin.x, s.y - origin.y] };
        let to_screen = |p: P2| -> egui::Pos2 {
            let c = view.to_canvas(p);
            egui::pos2(origin.x + c[0], origin.y + c[1])
        };
        let grid = self.snap_grid;
        let accent = theme::accent();

        // Handles, pivot and 9-slice lines of the primary sprite.
        if let Some(i) = primary {
            let d = &defs[i];
            let r = egui::Rect::from_min_max(to_screen([d.rect.x as f32, d.rect.y as f32]), to_screen([(d.rect.x + d.rect.w) as f32, (d.rect.y + d.rect.h) as f32]));
            if multiple {
                for h in Handle::ALL {
                    let f = h.frac();
                    let c = egui::pos2(r.min.x + f[0] * r.width(), r.min.y + f[1] * r.height());
                    let hr = egui::Rect::from_center_size(c, egui::vec2(7.0, 7.0));
                    painter.rect_filled(hr, 1.0, theme::BG_PANEL);
                    painter.rect_stroke(hr, 1.0, egui::Stroke::new(1.25_f32, accent), egui::StrokeKind::Inside);
                }
            }
            for (_, [a, b]) in ed::border_lines(d) {
                painter.line_segment(
                    [to_screen([d.rect.x as f32 + a[0], d.rect.y as f32 + a[1]]), to_screen([d.rect.x as f32 + b[0], d.rect.y as f32 + b[1]])],
                    egui::Stroke::new(1.0_f32, color(theme::STATUS_OK, 0.9)),
                );
            }
            let pv = d.pivot_pixels();
            let pc = to_screen([d.rect.x as f32 + pv[0], (d.rect.y + d.rect.h) as f32 - pv[1]]);
            painter.circle_stroke(pc, 5.0, egui::Stroke::new(1.75_f32, theme::STATUS_WARN_FG));
            painter.circle_filled(pc, 1.5, theme::STATUS_WARN_FG);
        }

        // Gestures.
        if resp.drag_started_by(egui::PointerButton::Primary) && !alt {
            let Some(press) = ui.input(|i| i.pointer.press_origin()) else { return };
            let c = to_canvas(press);
            let p = to_img(press);
            let mut next = Drag::None;
            if let Some(i) = primary {
                let d = &defs[i];
                if multiple && let Some(h) = ed::hit_handle(d.rect, view, c, 7.0) {
                    next = Drag::Resize { i, handle: h, orig: sheet.clone() };
                } else {
                    let pv = d.pivot_pixels();
                    let pc = view.to_canvas([d.rect.x as f32 + pv[0], (d.rect.y + d.rect.h) as f32 - pv[1]]);
                    if (pc[0] - c[0]).hypot(pc[1] - c[1]) <= 8.0 {
                        next = Drag::Pivot { i, orig: sheet.clone() };
                    } else if ctrl
                        && let Some(edge) = nearest_border(d, view, c, 6.0)
                    {
                        next = Drag::Border { i, edge, orig: sheet.clone() };
                    }
                }
            }
            if matches!(next, Drag::None) {
                if let Some(hit) = ed::pick_sprite(defs, p).filter(|_| multiple) {
                    if !sel.contains(hit) {
                        sel.click(hit, shift);
                        if let Some(e) = self.edit.as_mut() {
                            e.sel = sel.clone();
                        }
                    }
                    next = Drag::Move { start: p, orig: sheet.clone() };
                } else if multiple {
                    next = if shift { Drag::Marquee { start: p } } else { Drag::Create { start: p } };
                }
            }
            self.drag = next;
        }
        // Click selection.
        if resp.clicked_by(egui::PointerButton::Primary)
            && let Some(p) = ui.input(|i| i.pointer.interact_pos())
            && multiple
        {
            let img = to_img(p);
            if let Some(e) = self.edit.as_mut() {
                match ed::pick_sprite(defs, img) {
                    Some(i) => e.sel.click(i, shift || ctrl),
                    None if !shift && !ctrl => e.sel.clear(),
                    None => {}
                }
            }
        }
        // Continue the gesture.
        let cur = ui.input(|i| i.pointer.interact_pos()).map(&to_img);
        let Some(cur) = cur else { return };
        if !resp.dragged() {
            return;
        }
        let snap = |v: f32| ed::snap(v, grid);
        match &self.drag {
            Drag::Move { start, orig } => {
                let (dx, dy) = (snap(cur[0] - start[0]), snap(cur[1] - start[1]));
                let mut s = orig.clone();
                ed::move_selected(&mut s, &sel, dx, dy, iw, ih);
                *sheet = s;
            }
            Drag::Resize { i, handle, orig } => {
                let mut s = orig.clone();
                ed::resize_sprite(&mut s, *i, *handle, [snap(cur[0]), snap(cur[1])], iw, ih);
                *sheet = s;
            }
            Drag::Pivot { i, orig } => {
                let mut s = orig.clone();
                if let Some(d) = def_mut(&mut s, *i) {
                    // Rect-local pixels, y down.
                    let local = [cur[0] - d.rect.x as f32, cur[1] - d.rect.y as f32];
                    let p = ed::pivot_from_drag(d.rect, local, self.pivot_in_pixels, ctrl.then_some(6.0));
                    d.set_pivot(p);
                }
                *sheet = s;
            }
            Drag::Border { i, edge, orig } => {
                let mut s = orig.clone();
                if let Some(d) = def_mut(&mut s, *i) {
                    let local = [cur[0] - d.rect.x as f32, cur[1] - d.rect.y as f32];
                    let b = ed::border_from_drag(d, *edge, local);
                    d.border = b;
                }
                *sheet = s;
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn shape_mode(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        sheet: &mut SpriteSheet,
        defs: &[SpriteDef],
        primary: Option<usize>,
        kind: ShapeKind,
        view: &View,
        painter: &egui::Painter,
        origin: egui::Pos2,
        mods: (bool, bool, bool),
        multiple: bool,
        _iw: u32,
        _ih: u32,
    ) {
        let (shift, _ctrl, alt) = mods;
        let to_img = |s: egui::Pos2| -> P2 { view.to_image([s.x - origin.x, s.y - origin.y]) };
        let to_screen = |p: P2| -> egui::Pos2 {
            let c = view.to_canvas(p);
            egui::pos2(origin.x + c[0], origin.y + c[1])
        };
        let accent = theme::accent();
        // Clicking another sprite (outside the edited one) selects it.
        if resp.clicked_by(egui::PointerButton::Primary)
            && multiple
            && let Some(p) = ui.input(|i| i.pointer.interact_pos())
        {
            let img = to_img(p);
            let inside_primary = primary.is_some_and(|i| contains(defs[i].rect, img));
            if !inside_primary && let Some(e) = self.edit.as_mut() {
                match ed::pick_sprite(defs, img) {
                    Some(i) => e.sel.click(i, false),
                    None => e.sel.clear(),
                }
                self.pt_sel.clear();
            }
        }
        let Some(i) = primary else { return };
        let d = &defs[i];
        let (ox, oy) = (d.rect.x as f32, d.rect.y as f32);
        let size = [d.rect.w, d.rect.h];
        let loops = ed::shape(d, kind).clone();
        let line = if kind == ShapeKind::Outline { theme::STATUS_INFO } else { theme::STATUS_OK };
        if loops.is_empty() {
            painter.text(
                to_screen([ox + d.rect.w as f32 / 2.0, oy + d.rect.h as f32 / 2.0]),
                egui::Align2::CENTER_CENTER,
                "automatic: Generate or draw a quad",
                theme::mono(theme::SIZE_META),
                color(theme::TEXT_STRONG, 0.6),
            );
        }
        for l in &loops {
            let pts: Vec<egui::Pos2> = l.iter().map(|p| to_screen([ox + p[0], oy + p[1]])).collect();
            painter.add(egui::Shape::closed_line(pts, egui::Stroke::new(1.5_f32, line)));
        }
        if kind == ShapeKind::Physics && self.show_pieces {
            for piece in ed::convex_pieces_of(&loops) {
                let pts: Vec<egui::Pos2> = piece.iter().map(|p| to_screen([ox + p[0], oy + p[1]])).collect();
                painter.add(egui::Shape::convex_polygon(pts, color(theme::STATUS_OK, 0.12), egui::Stroke::new(1.0_f32, color(theme::STATUS_OK, 0.6))));
            }
        }
        for (li, l) in loops.iter().enumerate() {
            for (pi, p) in l.iter().enumerate() {
                let c = to_screen([ox + p[0], oy + p[1]]);
                let selected = self.pt_sel.contains(&(li, pi));
                let hr = egui::Rect::from_center_size(c, egui::vec2(7.0, 7.0));
                painter.rect_filled(hr, 1.0, if selected { accent } else { theme::BG_PANEL });
                painter.rect_stroke(hr, 1.0, egui::Stroke::new(1.25_f32, if selected { theme::TEXT_ON_ACCENT } else { accent }), egui::StrokeKind::Inside);
            }
        }

        let local_of = |s: egui::Pos2| -> P2 {
            let p = to_img(s);
            [p[0] - ox, p[1] - oy]
        };
        let tol_img = 8.0 / view.zoom;
        let pixel_snap = true;
        if resp.drag_started_by(egui::PointerButton::Primary) && !alt {
            let Some(press) = ui.input(|i| i.pointer.press_origin()) else { return };
            let lp = local_of(press);
            let mut next = Drag::None;
            if let Some(id) = ed::nearest_point(&loops, lp, tol_img) {
                if !self.pt_sel.contains(&id) {
                    if shift {
                        self.pt_sel.push(id);
                    } else {
                        self.pt_sel = vec![id];
                    }
                }
                next = Drag::Points { start: lp, ids: self.pt_sel.clone(), orig: sheet.clone() };
            } else if !shift {
                // A press on an edge adds a point there and drags it.
                let mut s = sheet.clone();
                let inserted = def_mut(&mut s, i).and_then(|d| ed::insert_point_on_edge(ed::shape_mut(d, kind), lp, 6.0 / view.zoom, size, pixel_snap));
                if let Some(id) = inserted {
                    self.pt_sel = vec![id];
                    next = Drag::Points { start: lp, ids: vec![id], orig: s.clone() };
                    *sheet = s;
                } else if contains_local(size, lp) {
                    next = Drag::Quad { start: to_img(press) };
                }
            } else {
                next = Drag::PointBox { start: to_img(press) };
            }
            self.drag = next;
        }
        if resp.clicked_by(egui::PointerButton::Primary) && !shift {
            // A plain click on empty space drops the point selection.
            if let Some(p) = ui.input(|i| i.pointer.interact_pos()) {
                let lp = local_of(p);
                if ed::nearest_point(&loops, lp, tol_img).is_none() {
                    self.pt_sel.clear();
                }
            }
        }
        if resp.dragged()
            && let Drag::Points { start, ids, orig } = &self.drag
            && let Some(cur) = ui.input(|i| i.pointer.interact_pos()).map(local_of)
        {
            let mut s = orig.clone();
            if let Some(d) = def_mut(&mut s, i) {
                let delta = [(cur[0] - start[0]).round(), (cur[1] - start[1]).round()];
                ed::move_points(ed::shape_mut(d, kind), ids, delta, size, pixel_snap);
            }
            *sheet = s;
        }
    }

    /// The pointer was released: commit creates and selections.
    fn finish_drag(&mut self, ui: &egui::Ui, sheet: &mut SpriteSheet, view: &View, origin: egui::Pos2, iw: u32, ih: u32) {
        let end = ui.input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos())).map(|s| view.to_image([s.x - origin.x, s.y - origin.y]));
        let grid = self.snap_grid;
        let drag = std::mem::replace(&mut self.drag, Drag::None);
        let Some(end) = end else { return };
        let stem = self.path.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "sprite".into());
        match drag {
            Drag::Create { start } => {
                let a = [ed::snap(start[0], grid), ed::snap(start[1], grid)];
                let b = [ed::snap(end[0], grid), ed::snap(end[1], grid)];
                if let (Some(r), Some(e)) = (ed::rect_from_points(a, b, iw, ih), self.edit.as_mut()) {
                    ed::create_sprite(sheet, &stem, r, &mut e.sel);
                }
            }
            Drag::Marquee { start } => {
                if let Some(e) = self.edit.as_mut() {
                    let hit = ed::sprites_in_box(&sheet.sprites, start, end);
                    let add = ui.input(|i| i.modifiers.command);
                    e.sel.set(hit, add);
                }
            }
            Drag::Quad { start } => {
                if let Some(e) = self.edit.as_ref()
                    && let (Some(i), Some(kind)) = (e.sel.primary(), self.mode.shape())
                    && let Some(d) = def_mut(sheet, i)
                {
                    let (ox, oy) = (d.rect.x as f32, d.rect.y as f32);
                    let size = [d.rect.w, d.rect.h];
                    ed::new_quad(ed::shape_mut(d, kind), [start[0] - ox, start[1] - oy], [end[0] - ox, end[1] - oy], size, true);
                }
            }
            Drag::PointBox { start } => {
                if let Some(e) = self.edit.as_ref()
                    && let (Some(i), Some(kind)) = (e.sel.primary(), self.mode.shape())
                    && let Some(d) = def_ref(sheet, i)
                {
                    let (ox, oy) = (d.rect.x as f32, d.rect.y as f32);
                    let ids = ed::points_in_box(ed::shape(d, kind), [start[0] - ox, start[1] - oy], [end[0] - ox, end[1] - oy]);
                    let add = ui.input(|i| i.modifiers.command);
                    self.pt_sel = if add { self.pt_sel.iter().copied().chain(ids).collect() } else { ids };
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn def_ref(sheet: &SpriteSheet, i: usize) -> Option<&SpriteDef> {
    match sheet.mode {
        SpriteMode::Multiple => sheet.sprites.get(i),
        SpriteMode::Single => (i == 0).then_some(&sheet.single),
    }
}

fn def_mut(sheet: &mut SpriteSheet, i: usize) -> Option<&mut SpriteDef> {
    match sheet.mode {
        SpriteMode::Multiple => sheet.sprites.get_mut(i),
        SpriteMode::Single => (i == 0).then_some(&mut sheet.single),
    }
}

fn contains(r: PixelRect, p: P2) -> bool {
    p[0] >= r.x as f32 && p[0] < (r.x + r.w) as f32 && p[1] >= r.y as f32 && p[1] < (r.y + r.h) as f32
}

fn contains_local(size: [u32; 2], p: P2) -> bool {
    p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= size[0] as f32 && p[1] <= size[1] as f32
}

/// The border edge of `d` whose line is within `tol` canvas pixels of `c`.
fn nearest_border(d: &SpriteDef, view: &View, c: P2, tol: f32) -> Option<BorderEdge> {
    ed::border_lines(d)
        .into_iter()
        .filter_map(|(edge, [a, b])| {
            let pa = view.to_canvas([d.rect.x as f32 + a[0], d.rect.y as f32 + a[1]]);
            let pb = view.to_canvas([d.rect.x as f32 + b[0], d.rect.y as f32 + b[1]]);
            let dist = match edge {
                BorderEdge::Left | BorderEdge::Right => (c[0] - pa[0]).abs().max(if c[1] < pa[1].min(pb[1]) - tol || c[1] > pa[1].max(pb[1]) + tol { f32::MAX } else { 0.0 }),
                BorderEdge::Top | BorderEdge::Bottom => (c[1] - pa[1]).abs().max(if c[0] < pa[0].min(pb[0]) - tol || c[0] > pa[0].max(pb[0]) + tol { f32::MAX } else { 0.0 }),
            };
            (dist <= tol).then_some((dist, edge))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, e)| e)
}

/// Checkerboard behind the texture: light base, darker 16 px squares.
fn paint_checker(painter: &egui::Painter, visible: egui::Rect, anchor: egui::Pos2) {
    if visible.width() <= 0.0 || visible.height() <= 0.0 {
        return;
    }
    let light = theme::BG_BUTTON;
    let dark = theme::BG_FIELD;
    painter.rect_filled(visible, 0.0, light);
    let tile = 16.0_f32;
    let x0 = ((visible.min.x - anchor.x) / tile).floor() as i32;
    let y0 = ((visible.min.y - anchor.y) / tile).floor() as i32;
    let x1 = ((visible.max.x - anchor.x) / tile).ceil() as i32;
    let y1 = ((visible.max.y - anchor.y) / tile).ceil() as i32;
    for ty in y0..y1 {
        for tx in x0..x1 {
            if (tx + ty).rem_euclid(2) == 1 {
                let r = egui::Rect::from_min_size(egui::pos2(anchor.x + tx as f32 * tile, anchor.y + ty as f32 * tile), egui::vec2(tile, tile)).intersect(visible);
                if r.width() > 0.0 && r.height() > 0.0 {
                    painter.rect_filled(r, 0.0, dark);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_mode_sprite_is_addressed_as_index_zero() {
        let mut s = SpriteSheet::default();
        s.single.name = "one".into();
        assert_eq!(def_ref(&s, 0).map(|d| d.name.as_str()), Some("one"));
        assert!(def_ref(&s, 1).is_none());
        def_mut(&mut s, 0).unwrap().border = [1, 0, 0, 0];
        assert_eq!(s.single.border, [1, 0, 0, 0]);
        s.mode = SpriteMode::Multiple;
        assert!(def_ref(&s, 0).is_none(), "an empty Multiple table has no sprite 0");
        s.sprites.push(SpriteDef::named("a", PixelRect::new(0, 0, 4, 4)));
        assert_eq!(def_ref(&s, 0).map(|d| d.name.as_str()), Some("a"));
    }

    #[test]
    fn border_lines_are_picked_by_distance() {
        let mut d = SpriteDef::named("a", PixelRect::new(10, 10, 40, 20));
        d.border = [10, 5, 10, 5];
        let v = View { zoom: 2.0, pan: [0.0, 0.0] };
        // The left line is at image x = 20 -> canvas x = 40, spanning y 20..60.
        assert_eq!(nearest_border(&d, &v, [41.0, 40.0], 4.0), Some(BorderEdge::Left));
        assert_eq!(nearest_border(&d, &v, [48.0, 40.0], 4.0), None);
        // The top line is at image y = 15 -> canvas y = 30.
        assert_eq!(nearest_border(&d, &v, [60.0, 31.0], 4.0), Some(BorderEdge::Top));
        // Far above the line segment's extent: no hit.
        assert_eq!(nearest_border(&d, &v, [41.0, 0.0], 4.0), None);
    }

    #[test]
    fn local_contains_and_modes() {
        assert!(contains_local([10, 10], [0.0, 10.0]));
        assert!(!contains_local([10, 10], [-1.0, 5.0]));
        assert!(contains(PixelRect::new(5, 5, 5, 5), [5.0, 9.9]));
        assert!(!contains(PixelRect::new(5, 5, 5, 5), [10.0, 9.0]));
        assert_eq!(Mode::Sprites.shape(), None);
        assert_eq!(Mode::Physics.shape(), Some(ShapeKind::Physics));
    }

    #[test]
    fn requests_are_taken_once() {
        let mut w = SpriteEditorWindow { req_apply: true, ..Default::default() };
        assert_eq!(w.take_requests(), EditorRequests { apply: true, revert_all: false });
        assert_eq!(w.take_requests(), EditorRequests::default());
        w.open(Path::new("/a.png"));
        assert!(w.open);
        w.open(Path::new("/b.png"));
        assert!(w.edit.is_none() && w.loaded.is_none());
    }
}

#[cfg(test)]
#[path = "shots/sprite_editor.rs"]
mod shots;
