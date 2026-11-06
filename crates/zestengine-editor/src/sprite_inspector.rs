//! Sprite section of the texture inspector (2D-2): sprite mode, pixels per
//! unit, mesh type, pivot, physics shape, a sprite table with numeric rects, a
//! simple grid slicer and a read-only preview of the rects, mesh and physics
//! outline over the texture. The full Sprite Editor window is 2D-4.
//!
//! Like `texture_inspector`, the decisions live in plain functions at the top
//! (grid slicing, rect field edits, overlay geometry) so they are unit tested;
//! the egui code below only lays them out. Edits go to the `TextureImport`
//! draft the Apply / Revert session owns, so nothing here saves anything.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use zestengine_assets::sprite::{
    PivotPreset, PixelRect, SliceMethod, SpriteDef, SpriteMeshType, SpriteMode, SpriteSheet, drop_empty, grid_rects, grid_rects_by_count, sprite_physics_pixels,
    sprite_pixel_mesh,
};
use zestengine_assets::sprite_mesh::{AlphaMap, P2};
use zestengine_assets::{TextureImport, TextureKind};
use crate::{icons::Icon, theme, widgets};

// ---------------------------------------------------------------------------
// Pure logic
// ---------------------------------------------------------------------------

/// Which rows the Sprite section shows for a mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpriteSections {
    /// Pivot and border of the one Single-mode sprite.
    pub single_sprite: bool,
    /// Grid slicer and the sprite table.
    pub table: bool,
    /// Extrude / detail / alpha tolerance (only a Tight mesh uses them).
    pub tight: bool,
}

pub fn sprite_sections(mode: SpriteMode, mesh: SpriteMeshType) -> SpriteSections {
    SpriteSections { single_sprite: mode == SpriteMode::Single, table: mode == SpriteMode::Multiple, tight: mesh == SpriteMeshType::Tight }
}

/// Grid slicing by cell size or by cell count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridBy {
    CellSize,
    CellCount,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSettings {
    pub by: GridBy,
    pub cell: [u32; 2],
    pub count: [u32; 2],
    pub offset: [u32; 2],
    pub padding: [u32; 2],
    pub keep_empty: bool,
    pub method: SliceMethod,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self { by: GridBy::CellSize, cell: [32, 32], count: [4, 4], offset: [0, 0], padding: [0, 0], keep_empty: false, method: SliceMethod::Smart }
    }
}

/// The rects the grid cuts out of a `w` x `h` image. Cells with no pixel above
/// `alpha_tolerance` are dropped unless `keep_empty` (or no pixels are loaded yet).
pub fn grid_cells(g: &GridSettings, w: u32, h: u32, alpha: Option<&AlphaMap>, alpha_tolerance: u8) -> Vec<PixelRect> {
    let rects = match g.by {
        GridBy::CellSize => grid_rects(w, h, g.cell, g.offset, g.padding),
        GridBy::CellCount => grid_rects_by_count(w, h, g.count[0], g.count[1], g.offset, g.padding),
    };
    match alpha {
        Some(a) if !g.keep_empty => drop_empty(rects, a, alpha_tolerance),
        _ => rects,
    }
}

/// Slice into the sheet's Multiple table. Returns (sprites after, sprites that
/// kept their identity).
pub fn apply_grid(sheet: &mut SpriteSheet, stem: &str, g: &GridSettings, w: u32, h: u32, alpha: Option<&AlphaMap>) -> (usize, usize) {
    let rects = grid_cells(g, w, h, alpha, sheet.alpha_tolerance);
    sheet.mode = SpriteMode::Multiple;
    let kept = sheet.apply_slice(stem, &rects, g.method);
    (sheet.sprites.len(), kept)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RectField {
    X,
    Y,
    W,
    H,
}

/// Set one field of a sprite rect, keeping it inside the `iw` x `ih` image and
/// at least 1 x 1 (position edits move the rect, size edits clip to the edge).
pub fn set_rect_field(rect: &mut PixelRect, field: RectField, value: u32, iw: u32, ih: u32) {
    match field {
        RectField::X => rect.x = value.min(iw.saturating_sub(rect.w.max(1))),
        RectField::Y => rect.y = value.min(ih.saturating_sub(rect.h.max(1))),
        RectField::W => rect.w = value.clamp(1, iw.saturating_sub(rect.x).max(1)),
        RectField::H => rect.h = value.clamp(1, ih.saturating_sub(rect.y).max(1)),
    }
}

/// The Single-mode sprite, made to cover the whole image and named after the file.
pub fn single_def<'a>(sheet: &'a mut SpriteSheet, w: u32, h: u32, stem: &str) -> &'a mut SpriteDef {
    sheet.ensure_ids(stem);
    sheet.single.rect = PixelRect::new(0, 0, w.max(1), h.max(1));
    &mut sheet.single
}

/// What the preview draws for one sprite, all in image pixels (y down).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    pub rect: PixelRect,
    /// Triangle edges of the render mesh.
    pub edges: Vec<[P2; 2]>,
    /// Physics polygons.
    pub physics: Vec<Vec<P2>>,
    pub tight: bool,
    pub triangles: usize,
}

/// Mesh and physics outline of `def` placed in image space.
pub fn build_overlay(def: &SpriteDef, sheet: &SpriteSheet, alpha: Option<&AlphaMap>) -> Overlay {
    let (mesh, tight) = sprite_pixel_mesh(def, sheet, alpha);
    let (ox, oy) = (def.rect.x as f32, def.rect.y as f32);
    let at = |p: P2| [p[0] + ox, p[1] + oy];
    let mut edges = Vec::new();
    for t in mesh.indices.as_chunks::<3>().0 {
        for k in 0..3 {
            edges.push([at(mesh.verts[t[k] as usize]), at(mesh.verts[t[(k + 1) % 3] as usize])]);
        }
    }
    let physics = sprite_physics_pixels(def, sheet, alpha).into_iter().map(|poly| poly.into_iter().map(at).collect()).collect();
    Overlay { rect: def.rect, edges, physics, tight, triangles: mesh.indices.len() / 3 }
}

/// Image pixel -> screen position inside `frame` (the drawn image).
pub fn image_to_screen(p: P2, frame: egui::Rect, w: u32, h: u32) -> egui::Pos2 {
    egui::pos2(frame.left() + p[0] / w.max(1) as f32 * frame.width(), frame.top() + p[1] / h.max(1) as f32 * frame.height())
}

/// One-line facts about a sprite: pixel size, world size and pivot.
pub fn sprite_summary(def: &SpriteDef, sheet: &SpriteSheet) -> String {
    let s = def.size_world(sheet);
    let p = def.pivot_pixels();
    format!("{}x{} px, {:.3} x {:.3} units, pivot ({:.1}, {:.1}) px", def.rect.w, def.rect.h, s[0], s[1], p[0], p[1])
}

/// Mesh facts for the preview caption.
pub fn mesh_caption(o: &Overlay, def: &SpriteDef, sheet: &SpriteSheet) -> String {
    let wanted = def.effective_mesh_type(sheet);
    let kind = if o.tight {
        "tight mesh"
    } else if wanted == SpriteMeshType::Tight {
        "quad (tight needs >= 32x32 and some alpha)"
    } else {
        "full rect"
    };
    format!("{kind}, {} triangles, {} physics shape{}", o.triangles, o.physics.len(), if o.physics.len() == 1 { "" } else { "s" })
}

/// Key under which a cached overlay stays valid.
type OverlayKey = (SpriteDef, SpriteMeshType, u32, u8, u32, bool, u8, u32);

fn overlay_key(def: &SpriteDef, s: &SpriteSheet) -> OverlayKey {
    (def.clone(), s.mesh_type, s.extrude, s.alpha_tolerance, s.outline_detail.to_bits(), s.generate_physics_shape, s.physics_alpha_tolerance, s.physics_detail.to_bits())
}

// ---------------------------------------------------------------------------
// egui
// ---------------------------------------------------------------------------

/// Decoded texture for the preview: a display copy (at most 512 px on the long
/// edge) and the full-resolution alpha plane the mesh generator reads.
struct Source {
    path: PathBuf,
    alpha: Arc<AlphaMap>,
    thumb: Option<egui::TextureHandle>,
}

type LoadResult = (PathBuf, Result<(AlphaMap, egui::ColorImage), String>);

/// UI state of the Sprite section.
#[derive(Default)]
pub struct SpriteUi {
    selected: usize,
    grid: GridSettings,
    pivot_in_pixels: bool,
    source: Option<Source>,
    job: Option<(PathBuf, Receiver<LoadResult>)>,
    error: Option<String>,
    overlay: Option<(OverlayKey, Overlay)>,
    status: String,
    /// The Sprite Editor window (2D-4), opened from the Sprite section.
    editor: crate::sprite_editor::SpriteEditorWindow,
    /// The inspector session has unapplied edits (set by the texture inspector).
    pub session_dirty: bool,
}

impl SpriteUi {
    /// Apply / Revert All the Sprite Editor asked for since the last call.
    pub fn take_editor_requests(&mut self) -> crate::sprite_editor::EditorRequests {
        self.editor.take_requests()
    }
}

fn load_preview(path: &Path) -> Result<(AlphaMap, egui::ColorImage), String> {
    let img = image::open(path).map_err(|e| e.to_string())?.to_rgba8();
    let (w, h) = (img.width(), img.height());
    let alpha = AlphaMap::from_rgba8(w, h, img.as_raw());
    let scale = (512.0 / w.max(h) as f32).min(1.0);
    let (tw, th) = (((w as f32 * scale) as u32).max(1), ((h as f32 * scale) as u32).max(1));
    let small = if scale < 1.0 { image::imageops::resize(&img, tw, th, image::imageops::FilterType::Nearest) } else { img };
    Ok((alpha, egui::ColorImage::from_rgba_unmultiplied([small.width() as usize, small.height() as usize], small.as_raw())))
}

/// Index of the sprite a Files-grid click picked (`pick` = absolute texture
/// path and sprite id), when it is for the texture at `path` and still exists.
pub fn picked_sprite(pick: Option<&(String, u64)>, path: &Path, sprites: &[SpriteDef]) -> Option<usize> {
    let (p, id) = pick?;
    if Path::new(p) != path {
        return None;
    }
    sprites.iter().position(|d| d.id.0 == *id)
}

fn drag_u32(ui: &mut egui::Ui, v: &mut u32, max: u32) -> bool {
    ui.add(egui::DragValue::new(v).range(0..=max).speed(0.5)).changed()
}

fn pair_row(ui: &mut egui::Ui, label: &str, v: &mut [u32; 2], max: [u32; 2]) -> bool {
    widgets::property_row(ui, label, |ui| {
        let a = drag_u32(ui, &mut v[0], max[0]);
        let b = drag_u32(ui, &mut v[1], max[1]);
        a | b
    })
}

impl SpriteUi {
    fn poll(&mut self, ctx: &egui::Context, path: &Path) {
        if self.source.as_ref().is_some_and(|s| s.path != path) {
            self.source = None;
            self.overlay = None;
            self.selected = 0;
        }
        if self.source.is_none() && self.job.as_ref().is_none_or(|(p, _)| p != path) {
            let (tx, rx) = std::sync::mpsc::channel();
            let p = path.to_path_buf();
            std::thread::spawn(move || {
                let r = load_preview(&p);
                let _ = tx.send((p, r));
            });
            self.job = Some((path.to_path_buf(), rx));
            self.error = None;
        }
        if let Some((p, rx)) = &self.job {
            match rx.try_recv() {
                Ok((rp, r)) if rp == *p => {
                    match r {
                        Ok((alpha, color)) => {
                            let thumb = ctx.load_texture("sprite-preview", color, egui::TextureOptions::NEAREST);
                            self.source = Some(Source { path: rp, alpha: Arc::new(alpha), thumb: Some(thumb) });
                        }
                        Err(e) => self.error = Some(e),
                    }
                    self.job = None;
                }
                Ok(_) | Err(TryRecvError::Disconnected) => self.job = None,
                Err(TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(80)),
            }
        }
    }

    /// The Sprite section for a texture of kind Sprite. `size` is the source
    /// image size (from the preview). Returns true when `t` changed.
    pub fn show(&mut self, ui: &mut egui::Ui, path: &Path, t: &mut TextureImport, size: Option<(u32, u32)>) -> bool {
        if t.kind != TextureKind::Sprite {
            return false;
        }
        self.poll(ui.ctx(), path);
        // A sprite clicked under this sheet in the Files grid becomes the selected one.
        let pick = ui.data_mut(|d| d.get_temp::<(String, u64)>(egui::Id::new(crate::SELECTED_SPRITE_KEY)));
        if let Some(i) = picked_sprite(pick.as_ref(), path, &t.sprite.sprites) {
            self.selected = i;
            ui.data_mut(|d| d.remove_temp::<(String, u64)>(egui::Id::new(crate::SELECTED_SPRITE_KEY)));
        }
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "sprite".into());
        let (w, h) = size.or_else(|| self.source.as_ref().map(|s| (s.alpha.w, s.alpha.h))).unwrap_or((0, 0));
        let mut changed = false;

        widgets::eyebrow(ui, "Sprite");
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, Some(Icon::Grid), "Open Sprite Editor").on_hover_text("Slice, edit rects, pivots, borders, outlines and physics shapes").clicked() {
                self.editor.open(path);
            }
            if let Some(name) = crate::sprite_editor::atlas_name_probe(path) {
                widgets::badge(ui, &format!("in atlas {name}")).on_hover_text("This texture's sprites are packed into an atlas when the game is built; the texture itself stays for the editor.");
            }
        });
        changed |= self.editor.show(ui.ctx(), path, t, self.session_dirty);
        {
            let s = &mut t.sprite;
            widgets::property_row(ui, "Sprite mode", |ui| {
                changed |= widgets::segmented(ui, &mut s.mode, &[(SpriteMode::Single, "Single"), (SpriteMode::Multiple, "Multiple")]).changed();
            });
            widgets::property_row(ui, "Pixels per unit", |ui| {
                changed |= ui.add(egui::DragValue::new(&mut s.ppu).range(0.001..=100_000.0).speed(0.5)).changed();
            });
            widgets::property_row(ui, "Mesh type", |ui| {
                changed |= widgets::segmented(ui, &mut s.mesh_type, &[(SpriteMeshType::FullRect, "Full Rect"), (SpriteMeshType::Tight, "Tight")]).changed();
            });
            let sec = sprite_sections(s.mode, s.mesh_type);
            if sec.tight {
                widgets::property_row(ui, "Extrude edges", |ui| {
                    let mut e = s.extrude as i32;
                    if widgets::slider(ui, egui::Slider::new(&mut e, 0..=32)).changed() {
                        s.extrude = e as u32;
                        changed = true;
                    }
                });
                widgets::property_row(ui, "Outline detail", |ui| {
                    changed |= widgets::slider(ui, egui::Slider::new(&mut s.outline_detail, 0.0..=1.0)).changed();
                });
                widgets::property_row(ui, "Alpha tolerance", |ui| {
                    let mut a = s.alpha_tolerance as i32;
                    if widgets::slider(ui, egui::Slider::new(&mut a, 0..=254)).changed() {
                        s.alpha_tolerance = a as u8;
                        changed = true;
                    }
                });
            }
            widgets::property_row(ui, "Physics shape", |ui| changed |= ui.checkbox(&mut s.generate_physics_shape, "").changed());
            if s.generate_physics_shape {
                widgets::property_row(ui, "Physics detail", |ui| {
                    changed |= widgets::slider(ui, egui::Slider::new(&mut s.physics_detail, 0.0..=1.0)).changed();
                });
            }
            widgets::property_row(ui, "Atlas padding", |ui| changed |= drag_u32(ui, &mut s.pack.padding, 64));
            widgets::property_row(ui, "Alpha dilation", |ui| changed |= ui.checkbox(&mut s.pack.alpha_dilate, "").changed());
            if t.sprite.mode == SpriteMode::Single && w > 0 {
                ui.add_space(4.0);
                let sheet_copy = t.sprite.clone();
                let def = single_def(&mut t.sprite, w, h, &stem);
                changed |= self.def_ui(ui, def, &sheet_copy);
            }
        }

        if t.sprite.mode == SpriteMode::Multiple {
            changed |= self.table_ui(ui, t, &stem, w, h);
        }
        self.preview_ui(ui, t, &stem, w, h);
        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
        changed
    }

    /// Pivot, border and the per-sprite physics reset of one sprite.
    fn def_ui(&mut self, ui: &mut egui::Ui, d: &mut SpriteDef, sheet: &SpriteSheet) -> bool {
        let mut changed = false;
        let presets: Vec<(PivotPreset, String)> = PivotPreset::ALL.iter().map(|p| (*p, p.label().to_owned())).collect();
        widgets::property_row(ui, "Pivot", |ui| {
            let current = d.pivot_preset;
            egui::ComboBox::from_id_salt("sprite-pivot").width(ui.available_width()).selected_text(current.label()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for (p, l) in &presets {
                    if ui.selectable_label(d.pivot_preset == *p, l).clicked() {
                        if let Some(n) = p.normalized() {
                            d.pivot = n;
                        }
                        d.pivot_preset = *p;
                        changed = true;
                    }
                }
            });
        });
        if d.pivot_preset == PivotPreset::Custom {
            widgets::property_row(ui, "", |ui| {
                changed |= widgets::segmented(ui, &mut self.pivot_in_pixels, &[(false, "Normalized"), (true, "Pixels")]).changed();
            });
            widgets::property_row(ui, "Custom pivot", |ui| {
                if self.pivot_in_pixels {
                    let mut p = d.pivot_pixels();
                    let a = ui.add(egui::DragValue::new(&mut p[0]).speed(0.25)).changed();
                    let b = ui.add(egui::DragValue::new(&mut p[1]).speed(0.25)).changed();
                    if a | b {
                        d.pivot = [p[0] / d.rect.w.max(1) as f32, p[1] / d.rect.h.max(1) as f32];
                        changed = true;
                    }
                } else {
                    let a = ui.add(egui::DragValue::new(&mut d.pivot[0]).speed(0.005).fixed_decimals(3)).changed();
                    let b = ui.add(egui::DragValue::new(&mut d.pivot[1]).speed(0.005).fixed_decimals(3)).changed();
                    changed |= a | b;
                }
            });
        }
        widgets::property_row(ui, "Border L B R T", |ui| {
            let mut b = d.border;
            let mut any = false;
            for (k, v) in b.iter_mut().enumerate() {
                let max = if k % 2 == 0 { d.rect.w } else { d.rect.h };
                any |= drag_u32(ui, v, max);
            }
            if any {
                d.border = b;
                d.border = d.border_clamped();
                changed = true;
            }
        });
        if !d.physics_shape.is_empty() {
            widgets::property_row(ui, "Physics shape", |ui| {
                ui.label(egui::RichText::new(format!("custom, {} polygon(s)", d.physics_shape.len())).size(theme::SIZE_META).color(theme::TEXT_SECONDARY));
                if widgets::ghost_button(ui, "Use auto", theme::H_FIELD).clicked() {
                    d.physics_shape.clear();
                    changed = true;
                }
            });
        }
        ui.label(egui::RichText::new(sprite_summary(d, sheet)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
        changed
    }

    fn table_ui(&mut self, ui: &mut egui::Ui, t: &mut TextureImport, stem: &str, w: u32, h: u32) -> bool {
        let mut changed = false;
        ui.add_space(4.0);
        widgets::eyebrow(ui, "Sprites");
        // ---- Grid slicer ----
        {
            let g = &mut self.grid;
            widgets::property_row(ui, "Grid by", |ui| {
                widgets::segmented(ui, &mut g.by, &[(GridBy::CellSize, "Cell size"), (GridBy::CellCount, "Cell count")]);
            });
            match g.by {
                GridBy::CellSize => {
                    pair_row(ui, "Cell size", &mut g.cell, [w.max(1), h.max(1)]);
                }
                GridBy::CellCount => {
                    pair_row(ui, "Columns, rows", &mut g.count, [4096, 4096]);
                }
            }
            pair_row(ui, "Offset", &mut g.offset, [w, h]);
            pair_row(ui, "Padding", &mut g.padding, [w, h]);
            widgets::property_row(ui, "Keep empty", |ui| {
                ui.checkbox(&mut g.keep_empty, "");
            });
            widgets::property_row(ui, "Existing sprites", |ui| {
                widgets::segmented(ui, &mut g.method, &[(SliceMethod::Smart, "Smart"), (SliceMethod::Safe, "Safe"), (SliceMethod::DeleteExisting, "Replace")]);
            });
        }
        ui.horizontal(|ui| {
            let can = w > 0 && h > 0;
            ui.add_enabled_ui(can, |ui| {
                if widgets::secondary_button(ui, Some(Icon::Grid), "Slice grid").on_hover_text("Cut the texture into a grid of sprites. Smart keeps names and ids of overlapping sprites.").clicked() {
                    let alpha = self.source.as_ref().map(|s| s.alpha.clone());
                    let (n, kept) = apply_grid(&mut t.sprite, stem, &self.grid, w, h, alpha.as_deref());
                    self.selected = self.selected.min(n.saturating_sub(1));
                    self.status = format!("Sliced into {n} sprite(s), {kept} kept their identity");
                    changed = true;
                }
                if widgets::secondary_button(ui, Some(Icon::Plus), "Add sprite").clicked() {
                    let i = t.sprite.add_sprite(stem, PixelRect::new(0, 0, w.clamp(1, 32), h.clamp(1, 32)));
                    self.selected = i;
                    changed = true;
                }
            });
        });
        ui.add_space(4.0);

        // ---- Table ----
        let sheet_copy = t.sprite.clone();
        let mut remove: Option<usize> = None;
        let mut rename: Option<(usize, String)> = None;
        for i in 0..t.sprite.sprites.len() {
            let selected = i == self.selected;
            ui.horizontal(|ui| {
                let d = &mut t.sprite.sprites[i];
                if ui.selectable_label(selected, format!("{i}")).clicked() {
                    self.selected = i;
                }
                // Edit in a buffer and commit on focus loss, so a half-typed or
                // duplicate name never reaches the table.
                let id = egui::Id::new(("sprite-name", d.id.0));
                let mut buf = ui.data_mut(|m| m.get_temp::<String>(id)).unwrap_or_else(|| d.name.clone());
                let r = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(96.0));
                if r.lost_focus() {
                    rename = Some((i, buf));
                    ui.data_mut(|m| m.remove_temp::<String>(id));
                } else if r.has_focus() {
                    ui.data_mut(|m| m.insert_temp(id, buf));
                }
                let mut rect = d.rect;
                let mut touched = false;
                for (f, v) in [(RectField::X, rect.x), (RectField::Y, rect.y), (RectField::W, rect.w), (RectField::H, rect.h)] {
                    let mut nv = v;
                    if drag_u32(ui, &mut nv, w.max(1)) {
                        set_rect_field(&mut rect, f, nv, w.max(1), h.max(1));
                        touched = true;
                    }
                }
                if touched {
                    d.rect = rect;
                    changed = true;
                }
                if widgets::icon_button(ui, Icon::Trash, "Remove this sprite (its id is retired, not reused)", false).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some((i, name)) = rename {
            // Only commit unique, non-empty names; the field keeps showing the old one otherwise.
            changed |= t.sprite.rename_sprite(i, &name);
        }
        if let Some(i) = remove {
            t.sprite.remove_sprite(i);
            self.selected = self.selected.min(t.sprite.sprites.len().saturating_sub(1));
            changed = true;
        }
        if t.sprite.sprites.is_empty() {
            widgets::info_note(ui, "No sprites yet. Slice a grid or add one by hand.");
        }
        for warn in t.sprite.warnings(w, h) {
            ui.label(egui::RichText::new(warn).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        // Selected sprite details.
        if self.selected < t.sprite.sprites.len() {
            ui.add_space(4.0);
            let sel = self.selected;
            let mut d = t.sprite.sprites[sel].clone();
            if self.def_ui(ui, &mut d, &sheet_copy) {
                t.sprite.sprites[sel] = d;
                changed = true;
            }
        }
        if changed {
            t.sprite.ensure_ids(stem);
        }
        changed
    }

    fn preview_ui(&mut self, ui: &mut egui::Ui, t: &TextureImport, stem: &str, w: u32, h: u32) {
        ui.add_space(6.0);
        widgets::eyebrow(ui, "Sprite preview");
        let Some(src) = &self.source else {
            match &self.error {
                Some(e) => {
                    ui.label(egui::RichText::new(format!("Preview failed: {e}")).color(theme::STATUS_ERROR).size(theme::SIZE_META));
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(egui::RichText::new("Loading pixels…").size(theme::SIZE_META).color(theme::TEXT_DIM));
                    });
                }
            }
            return;
        };
        let (w, h) = if w > 0 { (w, h) } else { (src.alpha.w, src.alpha.h) };
        let Some(thumb) = &src.thumb else { return };
        let defs = t.sprite.resolve(w, h, stem);
        let sel = self.selected.min(defs.len().saturating_sub(1));

        let avail = ui.available_width().min(320.0);
        let scale = (avail / w as f32).min(240.0 / h as f32);
        let (frame, _) = ui.allocate_exact_size(egui::vec2(w as f32 * scale, h as f32 * scale), egui::Sense::hover());
        let painter = ui.painter_at(frame);
        painter.rect_filled(frame, egui::CornerRadius::same(theme::R_ROW), theme::BG_FIELD);
        painter.image(thumb.id(), frame, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);

        let accent = theme::accent();
        for (i, d) in defs.iter().enumerate() {
            let r = egui::Rect::from_min_max(
                image_to_screen([d.rect.x as f32, d.rect.y as f32], frame, w, h),
                image_to_screen([(d.rect.x + d.rect.w) as f32, (d.rect.y + d.rect.h) as f32], frame, w, h),
            );
            let (stroke, kind) = if i == sel { (egui::Stroke::new(1.5_f32, accent), egui::StrokeKind::Inside) } else { (egui::Stroke::new(1.0_f32, theme::with_alpha(accent, 0.45)), egui::StrokeKind::Inside) };
            painter.rect_stroke(r, 0.0, stroke, kind);
        }
        if let Some(d) = defs.get(sel) {
            // Recompute the overlay only when its inputs changed, and not while
            // a slider is being dragged (the mesh of a big sprite is not free).
            let key = overlay_key(d, &t.sprite);
            let stale = self.overlay.as_ref().is_none_or(|(k, _)| *k != key);
            let dragging = ui.input(|i| i.pointer.any_down());
            if stale && (!dragging || self.overlay.is_none()) {
                let o = build_overlay(d, &t.sprite, Some(&src.alpha));
                self.overlay = Some((key, o));
            } else if stale {
                ui.ctx().request_repaint();
            }
            if let Some((_, o)) = &self.overlay {
                let mesh_col = theme::with_alpha(egui::Color32::from_rgb(90, 200, 255), 0.7);
                for e in &o.edges {
                    painter.line_segment([image_to_screen(e[0], frame, w, h), image_to_screen(e[1], frame, w, h)], egui::Stroke::new(0.75_f32, mesh_col));
                }
                for poly in &o.physics {
                    let pts: Vec<egui::Pos2> = poly.iter().map(|p| image_to_screen(*p, frame, w, h)).collect();
                    painter.add(egui::Shape::closed_line(pts, egui::Stroke::new(1.25_f32, egui::Color32::from_rgb(120, 230, 120))));
                }
                // Pivot marker.
                let pv = d.pivot_pixels();
                let c = image_to_screen([d.rect.x as f32 + pv[0], (d.rect.y + d.rect.h) as f32 - pv[1]], frame, w, h);
                painter.circle_stroke(c, 3.5, egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 200, 60)));
                let caption = mesh_caption(o, d, &t.sprite);
                ui.label(egui::RichText::new(caption).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_SECONDARY));
            }
        }
        ui.label(egui::RichText::new("blue: render mesh, green: physics shape, orange: pivot. Read-only here; edit outlines in the Sprite Editor.").size(theme::SIZE_META).color(theme::TEXT_DIM));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::sprite::PixelRect;

    fn solid(w: u32, h: u32, f: impl Fn(u32, u32) -> bool) -> AlphaMap {
        AlphaMap { w, h, alpha: (0..w * h).map(|i| if f(i % w, i / w) { 255 } else { 0 }).collect() }
    }

    #[test]
    fn files_grid_pick_selects_the_sprite_of_that_texture_only() {
        let mut s = SpriteSheet { mode: SpriteMode::Multiple, ..SpriteSheet::default() };
        for (i, n) in ["a", "b", "c"].iter().enumerate() {
            s.sprites.push(SpriteDef::named(n, PixelRect::new(i as u32 * 8, 0, 8, 8)));
        }
        s.ensure_ids("tex");
        let id = s.sprites[2].id.0;
        let path = Path::new("/p/sheet.png");
        assert_eq!(picked_sprite(Some(&("/p/sheet.png".into(), id)), path, &s.sprites), Some(2));
        assert_eq!(picked_sprite(Some(&("/p/other.png".into(), id)), path, &s.sprites), None);
        assert_eq!(picked_sprite(Some(&("/p/sheet.png".into(), 42)), path, &s.sprites), None, "removed sprite");
        assert_eq!(picked_sprite(None, path, &s.sprites), None);
    }

    #[test]
    fn sections_follow_mode_and_mesh_type() {
        let s = sprite_sections(SpriteMode::Single, SpriteMeshType::Tight);
        assert!(s.single_sprite && !s.table && s.tight);
        let s = sprite_sections(SpriteMode::Multiple, SpriteMeshType::FullRect);
        assert!(!s.single_sprite && s.table && !s.tight);
    }

    #[test]
    fn grid_by_size_and_by_count_agree() {
        let by_size = GridSettings { by: GridBy::CellSize, cell: [16, 32], keep_empty: true, ..GridSettings::default() };
        let by_count = GridSettings { by: GridBy::CellCount, count: [4, 2], keep_empty: true, ..GridSettings::default() };
        assert_eq!(grid_cells(&by_size, 64, 64, None, 0), grid_cells(&by_count, 64, 64, None, 0));
        assert_eq!(grid_cells(&by_size, 64, 64, None, 0).len(), 8);
    }

    #[test]
    fn empty_cells_are_dropped_unless_kept() {
        let a = solid(64, 32, |x, _| x < 32);
        let g = GridSettings { cell: [32, 32], ..GridSettings::default() };
        assert_eq!(grid_cells(&g, 64, 32, Some(&a), 0), vec![PixelRect::new(0, 0, 32, 32)]);
        let keep = GridSettings { keep_empty: true, ..g };
        assert_eq!(grid_cells(&keep, 64, 32, Some(&a), 0).len(), 2);
        // Without pixels loaded yet nothing is dropped.
        assert_eq!(grid_cells(&g, 64, 32, None, 0).len(), 2);
    }

    #[test]
    fn apply_grid_switches_to_multiple_and_reports() {
        let mut sheet = SpriteSheet::default();
        let g = GridSettings { cell: [32, 32], keep_empty: true, ..GridSettings::default() };
        let (n, kept) = apply_grid(&mut sheet, "tex", &g, 64, 64, None);
        assert_eq!((n, kept), (4, 0));
        assert_eq!(sheet.mode, SpriteMode::Multiple);
        assert_eq!(sheet.sprites[3].name, "tex_3");
        // Re-slicing the same grid keeps every identity.
        let ids: Vec<_> = sheet.sprites.iter().map(|s| s.id).collect();
        let (n, kept) = apply_grid(&mut sheet, "tex", &g, 64, 64, None);
        assert_eq!((n, kept), (4, 4));
        assert_eq!(sheet.sprites.iter().map(|s| s.id).collect::<Vec<_>>(), ids);
    }

    #[test]
    fn rect_field_edits_stay_inside_the_image() {
        let mut r = PixelRect::new(10, 10, 20, 20);
        set_rect_field(&mut r, RectField::X, 1000, 64, 64);
        assert_eq!(r.x, 44, "x clamps so the rect still fits");
        set_rect_field(&mut r, RectField::W, 1000, 64, 64);
        assert_eq!(r.w, 20, "width clips to the right edge");
        set_rect_field(&mut r, RectField::W, 0, 64, 64);
        assert_eq!(r.w, 1, "never empty");
        set_rect_field(&mut r, RectField::Y, 0, 64, 64);
        set_rect_field(&mut r, RectField::H, 64, 64, 64);
        assert_eq!((r.y, r.h), (0, 64));
    }

    #[test]
    fn single_def_covers_the_image_and_keeps_its_id() {
        let mut sheet = SpriteSheet::default();
        let id = single_def(&mut sheet, 128, 64, "hero").id;
        assert!(!id.is_none());
        assert_eq!(sheet.single.rect, PixelRect::new(0, 0, 128, 64));
        assert_eq!(sheet.single.name, "hero");
        // A different size later keeps the identity.
        assert_eq!(single_def(&mut sheet, 256, 64, "hero").id, id);
    }

    #[test]
    fn overlay_places_mesh_in_image_space() {
        let a = solid(128, 64, |x, y| (70..100).contains(&x) && (10..50).contains(&y));
        let sheet = SpriteSheet::default();
        let d = SpriteDef::named("a", PixelRect::new(64, 0, 64, 64));
        let o = build_overlay(&d, &sheet, Some(&a));
        assert!(o.tight);
        assert_eq!(o.triangles, 2);
        let xs: Vec<f32> = o.edges.iter().flat_map(|e| [e[0][0], e[1][0]]).collect();
        assert_eq!((xs.iter().cloned().fold(f32::MAX, f32::min), xs.iter().cloned().fold(f32::MIN, f32::max)), (70.0, 100.0));
        assert_eq!(o.physics.len(), 1);
        assert!(mesh_caption(&o, &d, &sheet).contains("tight mesh"));
        // Without pixels: a quad and no auto physics.
        let q = build_overlay(&d, &sheet, None);
        assert!(!q.tight && q.physics.is_empty());
        assert!(mesh_caption(&q, &d, &sheet).contains("quad"));
    }

    #[test]
    fn screen_mapping_and_summary() {
        let frame = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(100.0, 50.0));
        assert_eq!(image_to_screen([0.0, 0.0], frame, 200, 100), egui::pos2(10.0, 20.0));
        assert_eq!(image_to_screen([200.0, 100.0], frame, 200, 100), egui::pos2(110.0, 70.0));
        let sheet = SpriteSheet { ppu: 50.0, ..SpriteSheet::default() };
        let d = SpriteDef::named("a", PixelRect::new(0, 0, 100, 50));
        assert_eq!(sprite_summary(&d, &sheet), "100x50 px, 2.000 x 1.000 units, pivot (50.0, 25.0) px");
    }

    #[test]
    fn switching_kind_keeps_the_sprite_table() {
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        t.apply_kind(TextureKind::Sprite);
        assert_eq!(t.sampler.filter, zestengine_assets::FilterMode::Point, "sprites default to point filtering");
        apply_grid(&mut t.sprite, "tex", &GridSettings { keep_empty: true, ..GridSettings::default() }, 64, 64, None);
        let before = t.sprite.clone();
        t.apply_kind(TextureKind::Default);
        t.apply_kind(TextureKind::NormalMap);
        t.apply_kind(TextureKind::Sprite);
        assert_eq!(t.sprite, before, "kind switches never touch the table");
        // Re-applying the same kind does not reset a filter the user changed.
        t.sampler.filter = zestengine_assets::FilterMode::Bilinear;
        t.apply_kind(TextureKind::Sprite);
        assert_eq!(t.sampler.filter, zestengine_assets::FilterMode::Bilinear);
    }
}

#[cfg(test)]
#[path = "shots/sprite_inspector.rs"]
mod shots;
