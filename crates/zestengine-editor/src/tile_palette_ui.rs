//! The Tile Palette window (2D-5): palette selection and editing, tools,
//! brush, the active tilemap picker and "create palette from sprites".
//!
//! # Split of work
//!
//! This crate holds the window and the shared state ([`TilePaletteState`], kept
//! in egui memory like the other paint tools); the engine's `tile_edit.rs`
//! reads it every frame, turns scene-view pointer input into cell edits with
//! the algorithms in `zestengine_core::tile_paint`, applies them and records
//! per-cell undo diffs. Painting is active only while the window is open and a
//! Tilemap is the target, and the transform gizmo / picking stand back then.
//!
//! # Shortcuts (window focused or the scene view hovered, no text field focused)
//!
//! `S` select, `M` move, `B` paint, `U` box fill, `I` pick, `D` erase, `G`
//! flood fill; `[` / `]` rotate the brush counter clockwise / clockwise,
//! `Shift+[` / `Shift+]` flip it in X / Y. In the scene view, `Shift` while
//! painting erases, `Ctrl` picks, `Esc` cancels a box / selection, `Delete`
//! clears the selected cells, `Ctrl+C` / `Ctrl+V` copy and paste.

use std::path::{Path, PathBuf};

use egui::{Color32, RichText, Ui};
use glam::IVec2;
use zestengine_core::{
    Brush, CellEdit, CellLayout, CellState, ObjectId, RULETILE_EXTENSION, RuleTile, TILE_EXTENSION, PALETTE_EXTENSION, TileAsset, TileFile, TilePalette, TileSelection, TileTool,
    TileXform, rotate_brush_cw, save_palette, load_palette, tool_for_key,
};

use crate::{theme, widgets};

/// A key press the palette understands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PaletteKey {
    Tool(TileTool),
    RotateCw,
    RotateCcw,
    FlipX,
    FlipY,
}

/// Map a typed character (as egui reports `Key` names) and shift state to a palette key.
pub fn palette_key(c: char, shift: bool) -> Option<PaletteKey> {
    match (c, shift) {
        (']', false) => Some(PaletteKey::RotateCw),
        ('[', false) => Some(PaletteKey::RotateCcw),
        ('[', true) | ('{', _) => Some(PaletteKey::FlipX),
        (']', true) | ('}', _) => Some(PaletteKey::FlipY),
        (c, false) if c.is_ascii_alphabetic() => tool_for_key(c).map(PaletteKey::Tool),
        _ => None,
    }
}

/// The brush made of the palette cells in `sel` (inclusive corners): the
/// palette's y grows upward, so the brush rows follow it. Empty palette cells
/// stay `None`.
pub fn brush_from_palette(p: &TilePalette, sel: TileSelection) -> Brush {
    let size = sel.size();
    let mut cells = Vec::with_capacity((size.x * size.y) as usize);
    for c in sel.cells() {
        cells.push(p.tile_at(c.x, c.y).map(CellState::tile));
    }
    Brush { size, cells, pivot: IVec2::ZERO }
}

/// Apply a palette key to the state.
pub fn apply_key(st: &mut TilePaletteState, k: PaletteKey) {
    match k {
        PaletteKey::Tool(t) => st.tool = t,
        PaletteKey::RotateCw => st.brush = rotate_brush_cw(&st.brush),
        PaletteKey::RotateCcw => {
            for _ in 0..3 {
                st.brush = rotate_brush_cw(&st.brush);
            }
        }
        PaletteKey::FlipX => st.brush = st.brush.flipped_x(),
        PaletteKey::FlipY => st.brush = st.brush.flipped_y(),
    }
}

/// Tile assets and a palette for a list of sprites `(name, "path#id")`:
/// each becomes `<folder>/<name>.tile` and a palette cell, `columns` per row.
/// Names are made unique. Returns the files to write (relative path, asset)
/// and the palette (which refers to those paths).
pub fn palette_from_sprites(sprites: &[(String, String)], folder: &str, columns: usize) -> (Vec<(String, TileAsset)>, TilePalette) {
    let mut used = std::collections::HashSet::new();
    let mut files = Vec::new();
    for (name, sref) in sprites {
        let clean: String = name.chars().map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
        let base = if clean.is_empty() { "tile".to_string() } else { clean };
        let mut n = base.clone();
        let mut k = 2;
        while !used.insert(n.clone()) {
            n = format!("{base}_{k}");
            k += 1;
        }
        let rel = if folder.is_empty() { format!("{n}.{TILE_EXTENSION}") } else { format!("{}/{n}.{TILE_EXTENSION}", folder.trim_end_matches('/')) };
        files.push((rel, TileAsset::with_sprite(sref)));
    }
    let paths: Vec<String> = files.iter().map(|(p, _)| p.clone()).collect();
    (files, TilePalette::from_tiles(&paths, columns))
}

/// Write the files of [`palette_from_sprites`] under `root` plus
/// `<folder>/<palette_name>.palette`; existing tile files are kept. Returns the
/// palette's relative path.
pub fn write_palette(root: &Path, folder: &str, palette_name: &str, files: &[(String, TileAsset)], palette: &TilePalette) -> Result<String, String> {
    let dir = root.join(folder);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (rel, asset) in files {
        let abs = root.join(rel);
        if !abs.exists() {
            TileFile::Tile(asset.clone()).save(&abs)?;
        }
    }
    let rel = if folder.is_empty() { format!("{palette_name}.{PALETTE_EXTENSION}") } else { format!("{}/{palette_name}.{PALETTE_EXTENSION}", folder.trim_end_matches('/')) };
    save_palette(&root.join(&rel), palette)?;
    Ok(rel)
}

/// Window state, in egui memory (see [`TilePaletteState::get`]).
#[derive(Clone)]
pub struct TilePaletteState {
    pub open: bool,
    pub tool: TileTool,
    pub project_root: PathBuf,
    pub palette_rel: String,
    pub palette: TilePalette,
    loaded_rel: String,
    pub palette_dirty: bool,
    /// Selected palette cells (inclusive corners).
    pub palette_sel: Option<(IVec2, IVec2)>,
    /// What the Paint / Box / Flood tools put down.
    pub brush: Brush,
    /// Copied cells (Ctrl+C), pasted by Ctrl+V.
    pub clipboard: Option<Brush>,
    /// The tilemap object being painted.
    pub target: Option<ObjectId>,
    /// A stroke is in progress (one undo step).
    pub stroking: bool,
    /// Draw the grid lines at the active cell size in the scene view.
    pub show_grid: bool,
    /// Random brush: every painted cell gets one tile of the brush, picked by hashing the cell and `seed`.
    pub random: bool,
    pub seed: u32,
    /// Edits made this frame, drained by the engine into the undo stack.
    pub pending: Vec<(ObjectId, Vec<CellEdit>)>,
    pub status: String,
    create_texture: String,
    create_folder: String,
    /// Requests for the engine (it owns the sprite tables).
    pub create_request: Option<(String, String)>,
}

impl Default for TilePaletteState {
    fn default() -> Self {
        Self {
            open: false,
            tool: TileTool::Paint,
            project_root: PathBuf::new(),
            palette_rel: String::new(),
            palette: TilePalette::default(),
            loaded_rel: String::new(),
            palette_dirty: false,
            palette_sel: None,
            brush: Brush { size: IVec2::ONE, cells: vec![None], pivot: IVec2::ZERO },
            clipboard: None,
            target: None,
            stroking: false,
            show_grid: true,
            random: false,
            seed: 0,
            pending: Vec::new(),
            status: String::new(),
            create_texture: String::new(),
            create_folder: "Tiles".into(),
            create_request: None,
        }
    }
}

impl TilePaletteState {
    fn id() -> egui::Id {
        egui::Id::new("zestengine-tile-palette")
    }
    pub fn get(ctx: &egui::Context) -> Self {
        ctx.data(|d| d.get_temp(Self::id())).unwrap_or_default()
    }
    pub fn set(self, ctx: &egui::Context) {
        ctx.data_mut(|d| d.insert_temp(Self::id(), self));
    }
    /// Painting is live: the window is open and a tilemap is the target.
    pub fn painting(&self) -> bool {
        self.open && self.target.is_some()
    }
    /// The brush has something to put down.
    pub fn has_brush(&self) -> bool {
        !self.brush.is_empty()
    }
    /// Make a one-tile brush (a palette click or a pick).
    pub fn set_single(&mut self, tile: &str) {
        self.brush = Brush::tile(tile);
    }
    /// Select palette cells and rebuild the brush from them.
    pub fn select_palette(&mut self, a: IVec2, b: IVec2) {
        let sel = TileSelection::from_corners(a, b);
        self.palette_sel = Some((sel.min, sel.max));
        self.brush = brush_from_palette(&self.palette, sel);
    }
    /// Load the palette named by `palette_rel` when it changed.
    pub fn ensure_palette(&mut self) {
        if self.palette_rel == self.loaded_rel {
            return;
        }
        self.loaded_rel = self.palette_rel.clone();
        self.palette_sel = None;
        if self.palette_rel.is_empty() {
            self.palette = TilePalette::default();
            return;
        }
        match load_palette(&self.project_root.join(&self.palette_rel)) {
            Ok(p) => {
                self.palette = p;
                self.status.clear();
            }
            Err(e) => {
                self.palette = TilePalette::default();
                self.status = e;
            }
        }
    }
    pub fn save_palette(&mut self) {
        if self.palette_rel.is_empty() {
            return;
        }
        match save_palette(&self.project_root.join(&self.palette_rel), &self.palette) {
            Ok(()) => self.palette_dirty = false,
            Err(e) => self.status = e,
        }
    }
}

/// Create an empty palette file and select it.
pub fn new_palette(st: &mut TilePaletteState, folder: &str) {
    let mut n = 0;
    let rel = loop {
        let name = if n == 0 { "NewPalette".to_string() } else { format!("NewPalette{n}") };
        let rel = if folder.is_empty() { format!("{name}.{PALETTE_EXTENSION}") } else { format!("{folder}/{name}.{PALETTE_EXTENSION}") };
        if !st.project_root.join(&rel).exists() {
            break rel;
        }
        n += 1;
    };
    if let Some(dir) = st.project_root.join(&rel).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match save_palette(&st.project_root.join(&rel), &TilePalette::default()) {
        Ok(()) => st.palette_rel = rel,
        Err(e) => st.status = e,
    }
}

/// Create a new tile file in `folder` (a `.tile` or `.ruletile`) and add it to the palette's next free cell.
pub fn new_tile_file(st: &mut TilePaletteState, folder: &str, rule: bool) {
    let ext = if rule { RULETILE_EXTENSION } else { TILE_EXTENSION };
    let mut n = 0;
    let rel = loop {
        let name = if n == 0 { format!("NewTile.{ext}") } else { format!("NewTile{n}.{ext}") };
        let rel = if folder.is_empty() { name } else { format!("{folder}/{name}") };
        if !st.project_root.join(&rel).exists() {
            break rel;
        }
        n += 1;
    };
    let file = if rule { TileFile::Rule(RuleTile::default()) } else { TileFile::Tile(TileAsset::default()) };
    if let Some(dir) = st.project_root.join(&rel).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = file.save(&st.project_root.join(&rel)) {
        st.status = e;
        return;
    }
    let (mut x, mut y) = (0, 0);
    while st.palette.tile_at(x, y).is_some() {
        x += 1;
        if x > 7 {
            x = 0;
            y -= 1;
        }
    }
    st.palette.set(x, y, &rel);
    st.palette_dirty = true;
}

/// Something dragged from the Files panel onto the palette.
#[derive(Clone, Debug, PartialEq)]
pub enum Dropped {
    /// A sprite (`path#id`, from a sheet's children): becomes a tile that draws it.
    Sprite(String),
    /// A `.prefab` (project-relative): becomes a GameObject tile that puts an instance in each painted cell.
    Prefab(String),
}

/// The first free palette cell, filling rows of eight from the top-left like [`new_tile_file`].
fn next_free_cell(p: &TilePalette) -> (i32, i32) {
    let (mut x, mut y) = (0, 0);
    while p.tile_at(x, y).is_some() {
        x += 1;
        if x > 7 {
            x = 0;
            y -= 1;
        }
    }
    (x, y)
}

/// Make a `.tile` in `folder` for a dropped sprite or prefab and put it in the palette's next free cell.
/// A sprite or prefab that already has a tile in the palette is not added twice. Returns the tile's path.
pub fn add_dropped_tile(st: &mut TilePaletteState, folder: &str, dropped: &Dropped) -> Option<String> {
    let (stem, asset) = match dropped {
        Dropped::Sprite(sref) => {
            let (path, id) = sref.rsplit_once('#')?;
            let name = Path::new(path).file_stem()?.to_string_lossy().into_owned();
            let tail: String = id.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
            (format!("{name}_{tail}"), TileAsset::with_sprite(sref))
        }
        Dropped::Prefab(path) => {
            let name = Path::new(path).file_stem()?.to_string_lossy().into_owned();
            // A prefab has no sprite and no collider of its own: the instance brings its own.
            (name, TileAsset { game_object: path.clone(), collider: zestengine_core::TileColliderType::None, ..Default::default() })
        }
    };
    // Already in the palette?
    for (_, rel) in &st.palette.cells {
        if let Ok(TileFile::Tile(t)) = TileFile::load(&st.project_root.join(rel)) {
            let same = match dropped {
                Dropped::Sprite(s) => t.sprite == *s && t.game_object.is_empty(),
                Dropped::Prefab(p) => t.game_object == *p,
            };
            if same {
                st.status = format!("{} is already in the palette", tile_label(rel));
                return Some(rel.clone());
            }
        }
    }
    let clean: String = stem.chars().map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    let mut n = 0;
    let rel = loop {
        let name = if n == 0 { format!("{clean}.{TILE_EXTENSION}") } else { format!("{clean}_{n}.{TILE_EXTENSION}") };
        let rel = if folder.is_empty() { name } else { format!("{}/{name}", folder.trim_end_matches('/')) };
        if !st.project_root.join(&rel).exists() {
            break rel;
        }
        n += 1;
    };
    if let Some(dir) = st.project_root.join(&rel).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = TileFile::Tile(asset).save(&st.project_root.join(&rel)) {
        st.status = e;
        return None;
    }
    let (x, y) = next_free_cell(&st.palette);
    st.palette.set(x, y, &rel);
    st.palette_dirty = true;
    Some(rel)
}

fn tile_label(path: &str) -> String {
    Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}

/// Tilemap objects the picker offers: `(id, name)`.
pub type MapList<'a> = &'a [(ObjectId, String)];

/// Read shortcuts for the frame; returns true when a key was used. Skipped while
/// a text field has focus.
pub fn handle_shortcuts(ctx: &egui::Context, st: &mut TilePaletteState) -> bool {
    if !st.open || ctx.wants_keyboard_input() {
        return false;
    }
    let mut used = false;
    let events = ctx.input(|i| i.events.clone());
    for e in events {
        if let egui::Event::Text(t) = &e {
            let shift = ctx.input(|i| i.modifiers.shift);
            for c in t.chars() {
                if let Some(k) = palette_key(c, shift) {
                    apply_key(st, k);
                    used = true;
                }
            }
        }
    }
    used
}

/// The window. `maps` lists the scene's tilemap objects for the target picker.
pub fn tile_palette_window(ctx: &egui::Context, maps: MapList) {
    let mut st = TilePaletteState::get(ctx);
    if !st.open {
        return;
    }
    handle_shortcuts(ctx, &mut st);
    st.ensure_palette();
    let mut open = true;
    egui::Window::new("Tile Palette").open(&mut open).default_width(340.0).default_height(460.0).show(ctx, |ui| {
        palette_body(ui, &mut st, maps);
    });
    if !open {
        st.open = false;
        st.stroking = false;
    }
    st.set(ctx);
}

fn palette_body(ui: &mut Ui, st: &mut TilePaletteState, maps: MapList) {
    // Active tilemap.
    widgets::property_row(ui, "Active Tilemap", |ui| {
        let current = maps.iter().find(|(id, _)| Some(*id) == st.target).map(|(_, n)| n.clone()).unwrap_or_else(|| "None".to_string());
        egui::ComboBox::from_id_salt("tile-target").selected_text(current).show_ui(ui, |ui| {
            if ui.selectable_label(st.target.is_none(), "None").clicked() {
                st.target = None;
            }
            for (id, name) in maps {
                if ui.selectable_label(st.target == Some(*id), name).clicked() {
                    st.target = Some(*id);
                }
            }
        });
    });
    // Tools.
    ui.horizontal_wrapped(|ui| {
        for t in TileTool::ALL {
            let r = widgets::segment(ui, st.tool == t, &format!("{} ({})", t.label(), t.key()));
            if r.clicked() {
                st.tool = t;
            }
        }
    });
    ui.horizontal(|ui| {
        if widgets::ghost_button(ui, "Rotate ]", 22.0).clicked() {
            apply_key(st, PaletteKey::RotateCw);
        }
        if widgets::ghost_button(ui, "Flip X", 22.0).clicked() {
            apply_key(st, PaletteKey::FlipX);
        }
        if widgets::ghost_button(ui, "Flip Y", 22.0).clicked() {
            apply_key(st, PaletteKey::FlipY);
        }
        ui.checkbox(&mut st.show_grid, "Grid");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut st.random, "Random").on_hover_text("Paint, Line and Box Fill pick one tile of the selected cells at random per cell (stable for a cell and seed)");
        if st.random && widgets::ghost_button(ui, "Reroll", 22.0).on_hover_text("New seed: the picks change").clicked() {
            st.seed = st.seed.wrapping_add(1);
        }
    });
    ui.separator();
    // Palette picker.
    ui.horizontal(|ui| {
        let r = ui.add(egui::TextEdit::singleline(&mut st.palette_rel).hint_text("palette (.palette)").desired_width(ui.available_width() - 150.0));
        if r.lost_focus() {
            st.ensure_palette();
        }
        if ui.button("New").clicked() {
            new_palette(st, "Tiles");
        }
        if ui.add_enabled(st.palette_dirty, egui::Button::new("Save")).clicked() {
            st.save_palette();
        }
    });
    ui.horizontal(|ui| {
        if ui.button("+ Tile").on_hover_text("Create a .tile and add it to the palette").clicked() {
            new_tile_file(st, "Tiles", false);
        }
        if ui.button("+ Rule Tile").clicked() {
            new_tile_file(st, "Tiles", true);
        }
    });
    let zone = ui.scope(|ui| palette_grid(ui, st));
    // Drag sprites (a sheet's children) or prefabs from the Files panel onto the palette to make tiles.
    let drop = ui.interact(zone.response.rect.expand(2.0), ui.id().with("palette-drop"), egui::Sense::hover());
    let prefab_path = |p: &PathBuf| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("prefab"));
    let hovering = drop.dnd_hover_payload::<crate::file_browser::SpriteDrag>().is_some() || drop.dnd_hover_payload::<PathBuf>().is_some_and(|p| prefab_path(&p));
    if hovering {
        ui.painter().rect_stroke(zone.response.rect.expand(2.0), 4.0, egui::Stroke::new(2.0_f32, theme::ACCENT_DEFAULT), egui::StrokeKind::Outside);
    }
    if let Some(sprite) = drop.dnd_release_payload::<crate::file_browser::SpriteDrag>() {
        add_dropped_tile(st, "Tiles", &Dropped::Sprite(sprite.0.clone()));
    } else if let Some(path) = drop.dnd_release_payload::<PathBuf>()
        && prefab_path(&path)
    {
        let rel = path.strip_prefix(&st.project_root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
        add_dropped_tile(st, "Tiles", &Dropped::Prefab(rel));
    }
    if st.palette_rel.is_empty() && st.palette_dirty {
        st.status = "Drop made tiles: press New to create a palette file, then Save".into();
    }
    ui.separator();
    ui.label(RichText::new("Create palette from a sliced texture").size(theme::SIZE_META).color(theme::TEXT_DIM));
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut st.create_texture).hint_text("texture (sliced Sprite)").desired_width(150.0));
        ui.add(egui::TextEdit::singleline(&mut st.create_folder).hint_text("folder").desired_width(90.0));
        if ui.button("Create").clicked() && !st.create_texture.is_empty() {
            st.create_request = Some((st.create_texture.clone(), st.create_folder.clone()));
        }
    });
    if !st.status.is_empty() {
        ui.label(RichText::new(&st.status).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    }
    if st.target.is_none() {
        widgets::info_note(ui, "Pick an Active Tilemap to paint in the Scene view.");
    }
}

fn palette_grid(ui: &mut Ui, st: &mut TilePaletteState) {
    let Some((lo, hi)) = st.palette.bounds() else {
        widgets::info_note(ui, "Empty palette: create tiles with + Tile, or make one from a sliced texture.");
        return;
    };
    let cell = 56.0;
    let (w, h) = ((hi.x - lo.x + 1) as f32, (hi.y - lo.y + 1) as f32);
    egui::ScrollArea::both().max_height(220.0).show(ui, |ui| {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(w * cell, h * cell), egui::Sense::click_and_drag());
        let to_cell = |p: egui::Pos2| IVec2::new(lo.x + ((p.x - rect.left()) / cell).floor() as i32, hi.y - ((p.y - rect.top()) / cell).floor() as i32);
        let cell_rect = |c: IVec2| egui::Rect::from_min_size(egui::pos2(rect.left() + (c.x - lo.x) as f32 * cell, rect.top() + (hi.y - c.y) as f32 * cell), egui::vec2(cell, cell));
        let painter = ui.painter_at(rect);
        for ((x, y), tile) in &st.palette.cells {
            let c = IVec2::new(*x, *y);
            let r = cell_rect(c).shrink(2.0);
            painter.rect_filled(r, 4.0, theme::BG_BUTTON);
            painter.text(r.center(), egui::Align2::CENTER_CENTER, tile_label(tile), egui::FontId::proportional(11.0), theme::TEXT_BODY);
        }
        if let Some((a, b)) = st.palette_sel {
            let r = cell_rect(IVec2::new(a.x, b.y)).union(cell_rect(IVec2::new(b.x, a.y)));
            painter.rect_stroke(r, 4.0, egui::Stroke::new(2.0_f32, theme::ACCENT_DEFAULT), egui::StrokeKind::Inside);
        }
        if let Some(p) = resp.interact_pointer_pos() {
            let c = to_cell(p);
            if resp.drag_started() {
                ui.data_mut(|d| d.insert_temp(egui::Id::new("tile-palette-anchor"), c));
            }
            let anchor: IVec2 = ui.data(|d| d.get_temp(egui::Id::new("tile-palette-anchor"))).unwrap_or(c);
            if resp.dragged() || resp.clicked() {
                st.select_palette(anchor, c);
            }
        }
    });
    let _ = (Color32::WHITE, CellLayout::Rectangle, TileXform::IDENTITY);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(tag: &str) -> TilePaletteState {
        let root = std::env::temp_dir().join(format!("zest-palette-drop-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        TilePaletteState { project_root: root, ..Default::default() }
    }

    #[test]
    fn dropped_sprites_and_prefabs_become_tiles_in_the_next_free_cells() {
        let mut st = state("drop");
        let a = add_dropped_tile(&mut st, "Tiles", &Dropped::Sprite("art/grass.png#00000000000a1b2c".into())).unwrap();
        assert_eq!(a, "Tiles/grass_1b2c.tile");
        let TileFile::Tile(t) = TileFile::load(&st.project_root.join(&a)).unwrap() else { panic!("a plain tile") };
        assert_eq!(t.sprite, "art/grass.png#00000000000a1b2c");
        assert_eq!(st.palette.tile_at(0, 0), Some(a.as_str()));
        assert!(st.palette_dirty);
        // The same sprite again adds nothing; a different one takes the next cell.
        assert_eq!(add_dropped_tile(&mut st, "Tiles", &Dropped::Sprite("art/grass.png#00000000000a1b2c".into())).as_deref(), Some(a.as_str()));
        assert_eq!(st.palette.cells.len(), 1);
        let b = add_dropped_tile(&mut st, "Tiles", &Dropped::Sprite("art/grass.png#00000000000a1b2d".into())).unwrap();
        assert_ne!(a, b);
        assert_eq!(st.palette.tile_at(1, 0), Some(b.as_str()));
        // A prefab becomes a GameObject tile with no sprite and no collider.
        let c = add_dropped_tile(&mut st, "Tiles", &Dropped::Prefab("prefabs/crate.prefab".into())).unwrap();
        assert_eq!(c, "Tiles/crate.tile");
        let TileFile::Tile(t) = TileFile::load(&st.project_root.join(&c)).unwrap() else { panic!() };
        assert_eq!((t.game_object.as_str(), t.sprite.as_str(), t.collider), ("prefabs/crate.prefab", "", zestengine_core::TileColliderType::None));
        // A name clash with an unrelated existing file gets a suffix.
        std::fs::write(st.project_root.join("Tiles/rock.tile"), "(sprite: \"other#0000000000000001\")").unwrap();
        let d = add_dropped_tile(&mut st, "Tiles", &Dropped::Prefab("x/rock.prefab".into())).unwrap();
        assert_eq!(d, "Tiles/rock_1.tile");
        // Nine drops fill row one and wrap to the next row down.
        for i in 0..9 {
            add_dropped_tile(&mut st, "Tiles", &Dropped::Sprite(format!("s.png#{i:016x}")));
        }
        assert!(st.palette.cells.iter().any(|((_, y), _)| *y == -1));
        // Garbage references are refused.
        assert_eq!(add_dropped_tile(&mut st, "Tiles", &Dropped::Sprite("no-hash".into())), None);
        let _ = std::fs::remove_dir_all(&st.project_root);
    }

    #[test]
    fn shortcuts_map_to_tools_and_brush_ops() {
        assert_eq!(palette_key('b', false), Some(PaletteKey::Tool(TileTool::Paint)));
        assert_eq!(palette_key('U', false), Some(PaletteKey::Tool(TileTool::BoxFill)));
        assert_eq!(palette_key('g', false), Some(PaletteKey::Tool(TileTool::FloodFill)));
        assert_eq!(palette_key(']', false), Some(PaletteKey::RotateCw));
        assert_eq!(palette_key('[', true), Some(PaletteKey::FlipX));
        assert_eq!(palette_key(']', true), Some(PaletteKey::FlipY));
        assert_eq!(palette_key('z', false), None);
        assert_eq!(palette_key('B', true), None, "shift letters are not tool keys");
    }

    #[test]
    fn brush_from_palette_follows_the_selection() {
        let p = TilePalette::from_tiles(&["a".into(), "b".into(), "c".into(), "d".into()], 2);
        // a b  (y = 1)   c d  (y = 0)
        let b = brush_from_palette(&p, TileSelection::from_corners(IVec2::new(1, 1), IVec2::new(0, 0)));
        assert_eq!(b.size, IVec2::new(2, 2));
        assert_eq!(b.get(0, 1).unwrap().tile, "a");
        assert_eq!(b.get(1, 0).unwrap().tile, "d");
        let one = brush_from_palette(&p, TileSelection::from_corners(IVec2::new(1, 0), IVec2::new(1, 0)));
        assert_eq!(one.get(0, 0).unwrap().tile, "d");
        let gap = brush_from_palette(&p, TileSelection::from_corners(IVec2::new(5, 5), IVec2::new(5, 5)));
        assert!(gap.is_empty());
    }

    #[test]
    fn keys_change_tool_and_rotate_the_brush() {
        let mut st = TilePaletteState::default();
        apply_key(&mut st, PaletteKey::Tool(TileTool::Erase));
        assert_eq!(st.tool, TileTool::Erase);
        st.brush = Brush { size: IVec2::new(2, 1), cells: vec![Some(CellState::tile("a")), Some(CellState::tile("b"))], pivot: IVec2::ZERO };
        apply_key(&mut st, PaletteKey::RotateCw);
        assert_eq!(st.brush.size, IVec2::new(1, 2));
        apply_key(&mut st, PaletteKey::RotateCcw);
        assert_eq!(st.brush.size, IVec2::new(2, 1));
        assert_eq!(st.brush.get(0, 0).unwrap().tile, "a");
        assert!(st.brush.get(0, 0).unwrap().xform.is_identity());
    }

    #[test]
    fn palette_creation_from_a_sliced_texture() {
        let sprites: Vec<(String, String)> = (0..5).map(|i| (if i == 3 { "grass".to_string() } else { format!("sheet_{i}") }, format!("art/sheet.png#{:016x}", i + 1))).chain([("grass".to_string(), "art/sheet.png#00000000000000ff".to_string())]).collect();
        let (files, pal) = palette_from_sprites(&sprites, "Tiles/Sheet", 4);
        assert_eq!(files.len(), 6);
        let names: Vec<_> = files.iter().map(|(p, _)| p.as_str()).collect();
        assert!(names.contains(&"Tiles/Sheet/grass.tile") && names.contains(&"Tiles/Sheet/grass_2.tile"), "{names:?}");
        assert_eq!(files[0].1.sprite, "art/sheet.png#0000000000000001");
        assert_eq!(pal.cells.len(), 6);
        // Two rows of four: the first four on the top row.
        assert_eq!(pal.tile_at(0, 1), Some("Tiles/Sheet/sheet_0.tile"));
        assert_eq!(pal.tile_at(1, 0), Some("Tiles/Sheet/grass_2.tile"));
        // Written to disk and readable back.
        let dir = std::env::temp_dir().join(format!("zest_palette_{}", std::process::id()));
        let rel = write_palette(&dir, "Tiles/Sheet", "Sheet", &files, &pal).unwrap();
        assert_eq!(rel, "Tiles/Sheet/Sheet.palette");
        assert_eq!(load_palette(&dir.join(&rel)).unwrap(), pal);
        match TileFile::load(&dir.join("Tiles/Sheet/grass.tile")).unwrap() {
            TileFile::Tile(t) => assert_eq!(t.sprite, "art/sheet.png#0000000000000004"),
            _ => panic!("a tile"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_palette_and_tile_files_get_unique_names() {
        let dir = std::env::temp_dir().join(format!("zest_palette_new_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut st = TilePaletteState { project_root: dir.clone(), ..Default::default() };
        new_palette(&mut st, "Tiles");
        assert_eq!(st.palette_rel, "Tiles/NewPalette.palette");
        new_palette(&mut st, "Tiles");
        assert_eq!(st.palette_rel, "Tiles/NewPalette1.palette");
        new_tile_file(&mut st, "Tiles", false);
        new_tile_file(&mut st, "Tiles", true);
        assert!(dir.join("Tiles/NewTile.tile").exists() && dir.join("Tiles/NewTile.ruletile").exists());
        assert_eq!(st.palette.cells.len(), 2);
        assert!(st.palette_dirty);
        st.save_palette();
        assert!(!st.palette_dirty);
        st.palette_rel = "Tiles/NewPalette.palette".into();
        st.ensure_palette();
        assert!(st.palette.cells.is_empty(), "a different palette file loads its own cells");
        assert!(!st.painting());
        st.open = true;
        st.target = Some(ObjectId::new());
        assert!(st.painting());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
