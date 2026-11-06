//! Scene-view tile painting (2D-5): turns pointer input over the Scene view
//! into cell edits on the Tile Palette's target tilemap.
//!
//! Active only while the Tile Palette window is open and a Tilemap is its
//! target (not in Play). While active, left drags paint instead of orbiting or
//! moving the transform gizmo (`active()` is checked at the same places the
//! other paint tools are). Every stroke is one undo entry made of per-cell
//! diffs (`UndoEntry::Tiles`), never a snapshot of the map: edits apply to the
//! map immediately, accumulate in a `StrokeRecord`, and the finished stroke is
//! queued in `TilePaletteState::pending` for the app to push onto the undo
//! stack. The frame's generic component diff is suppressed (`take_edited`) so
//! the same change is not recorded twice.
//!
//! Tools: Paint (Shift erases, Ctrl picks), Box Fill (Shift = box erase), Pick,
//! Erase, Flood Fill, Select (Ctrl+C / Ctrl+V / Delete), Move (drag the
//! selection). Esc cancels a box / selection.

use std::path::Path;

use egui::Pos2;
use glam::{IVec2, Vec2, Vec3};
use zestengine_core::{
    Brush, CellEdit, ObjectId, StrokeRecord, TileSelection, TileTool, Tilemap, apply_edits, box_fill, box_fill_random, erase_cells, flood_fill, line_brush, line_cells, move_cells,
    paint_brush_mode, pick_brush,
};
use crate::TilePaletteState;

use crate::cloth_paint::{View, hint_bar};
use zestengine_engine::scene::LoadedScene;
use zestengine_engine::tile_db::grid_for;

/// Scene tilemaps for the target picker, in egui memory.
const MAPS_KEY: &str = "zestengine-tile-maps";
const EDITED_KEY: &str = "zestengine-tile-edited";

/// Per-gesture state that does not belong in the palette state.
#[derive(Clone, Default)]
struct Gesture {
    stroke: Option<StrokeRecordBox>,
    last_cell: Option<IVec2>,
    anchor: Option<IVec2>,
    selection: Option<(IVec2, IVec2)>,
    move_from: Option<IVec2>,
    #[allow(dead_code)] // a tool override hook set by the tile menus
    tool_override: Option<TileTool>,
    last_selected: Option<ObjectId>,
}

#[derive(Clone, Default)]
struct StrokeRecordBox(StrokeRecord);

fn gesture(ctx: &egui::Context) -> Gesture {
    ctx.data(|d| d.get_temp(egui::Id::new("zestengine-tile-gesture"))).unwrap_or_default()
}

fn set_gesture(ctx: &egui::Context, g: Gesture) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("zestengine-tile-gesture"), g));
}

fn tilemap_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut Tilemap> {
    scene.objects.get_mut(i)?.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Tilemap>())
}

fn tilemap(scene: &LoadedScene, i: usize) -> Option<&Tilemap> {
    scene.objects.get(i)?.components.iter().find_map(|c| c.as_any().downcast_ref::<Tilemap>())
}

/// Keep the palette's target valid and follow the selection; publish the map list.
pub(crate) fn sync(ctx: &egui::Context, scene: &LoadedScene, selected: Option<usize>, playing: bool, root: &Path) {
    let mut st = TilePaletteState::get(ctx);
    if !st.open {
        return;
    }
    let before_target = st.target;
    if st.project_root != root {
        st.project_root = root.to_path_buf();
    }
    let maps: Vec<(ObjectId, String)> = scene
        .objects
        .iter()
        .enumerate()
        .filter(|(i, _)| tilemap(scene, *i).is_some())
        .map(|(_, o)| (o.id, if o.name.is_empty() { "Tilemap".to_string() } else { o.name.clone() }))
        .collect();
    let mut g = gesture(ctx);
    let sel_id = selected.and_then(|i| scene.objects.get(i)).map(|o| o.id);
    if sel_id != g.last_selected {
        g.last_selected = sel_id;
        if let Some(id) = sel_id
            && maps.iter().any(|(m, _)| *m == id)
        {
            st.target = Some(id);
        }
    }
    if st.target.is_some_and(|t| !maps.iter().any(|(m, _)| *m == t)) {
        st.target = None;
    }
    if st.target != before_target || playing {
        st.stroking = false;
        g.stroke = None;
        g.selection = None;
        g.anchor = None;
    }
    set_gesture(ctx, g);
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(MAPS_KEY), maps));
    st.set(ctx);
}

/// The window itself (call inside the egui pass) plus the engine-side
/// "create palette from a texture" request.
pub(crate) fn window(ctx: &egui::Context) {
    let maps: Vec<(ObjectId, String)> = ctx.data(|d| d.get_temp(egui::Id::new(MAPS_KEY))).unwrap_or_default();
    crate::tile_palette_window(ctx, &maps);
    let mut st = TilePaletteState::get(ctx);
    if let Some((tex, folder)) = st.create_request.take() {
        match create_from_texture(&st.project_root, &tex, &folder) {
            Ok(rel) => {
                st.palette_rel = rel;
                st.status = format!("Created palette from {tex}");
            }
            Err(e) => st.status = e,
        }
        st.set(ctx);
    }
}

/// Slice a texture's sprites into tile assets plus a palette in `folder`.
pub(crate) fn create_from_texture(root: &Path, texture_rel: &str, folder: &str) -> Result<String, String> {
    let table = zestengine_assets::sprite::load_sprite_table(&root.join(texture_rel))
        .map_err(|e| format!("{texture_rel}: {e:#}"))?
        .ok_or_else(|| format!("{texture_rel} is not a Sprite texture"))?;
    if table.sprites.is_empty() {
        return Err(format!("{texture_rel} has no sprites: slice it in the Sprite Editor first"));
    }
    let sprites: Vec<(String, String)> =
        table.sprites.iter().map(|s| (s.name.clone(), zestengine_assets::sprite::sprite_ref_string(texture_rel, s.id))).collect();
    let (files, palette) = crate::palette_from_sprites(&sprites, folder, 8);
    let name = Path::new(texture_rel).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Palette".into());
    crate::write_palette(root, folder, &name, &files, &palette)
}

pub(crate) fn active(ctx: &egui::Context) -> bool {
    let st = TilePaletteState::get(ctx);
    st.painting()
}

#[allow(dead_code)] // stroke probe used by the editor's undo coalescing
pub(crate) fn stroking(ctx: &egui::Context) -> bool {
    TilePaletteState::get(ctx).stroking
}

/// Esc: cancel a box / selection first, else close nothing. True when used.
pub(crate) fn exit(ctx: &egui::Context) -> bool {
    let mut g = gesture(ctx);
    if g.anchor.is_none() && g.selection.is_none() && g.stroke.is_none() {
        return false;
    }
    g.anchor = None;
    g.selection = None;
    g.stroke = None;
    g.move_from = None;
    set_gesture(ctx, g);
    let mut st = TilePaletteState::get(ctx);
    st.stroking = false;
    st.set(ctx);
    true
}

/// True once if tile edits landed this frame (the app then skips its generic
/// component diff for the frame).
pub(crate) fn take_edited(ctx: &egui::Context) -> bool {
    ctx.data_mut(|d| d.remove_temp::<bool>(egui::Id::new(EDITED_KEY))).unwrap_or(false)
}

fn mark_edited(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(EDITED_KEY), true));
}

/// Finished strokes (`(tilemap object, per-cell edits)`) for the undo stack.
pub(crate) fn drain_strokes(ctx: &egui::Context) -> Vec<(ObjectId, Vec<CellEdit>)> {
    let mut st = TilePaletteState::get(ctx);
    if st.pending.is_empty() {
        return Vec::new();
    }
    let v = std::mem::take(&mut st.pending);
    st.set(ctx);
    v
}

/// Cell under a screen position on the tilemap's plane.
fn cell_under(view: &View, scene: &LoadedScene, i: usize, pos: Pos2) -> Option<IVec2> {
    let (o, d) = view.ray(pos)?;
    let world = scene.world_transform(i);
    let inv = world.inverse();
    let (lo, ld) = (inv.transform_point3(o), inv.transform_vector3(d));
    if ld.z.abs() < 1e-8 {
        return None;
    }
    let t = -lo.z / ld.z;
    let p = lo + ld * t;
    let c = grid_for(scene, i).local_to_cell(Vec3::new(p.x, p.y, 0.0));
    Some(IVec2::new(c.x, c.y))
}

fn outline(painter: &egui::Painter, view: &View, scene: &LoadedScene, i: usize, cell: IVec2, stroke: egui::Stroke, fill: Option<egui::Color32>) {
    let world = scene.world_transform(i);
    let pts: Vec<Pos2> = grid_for(scene, i).cell_corners(cell.extend(0)).iter().filter_map(|p| view.project(world.transform_point3(*p))).collect();
    if pts.len() >= 3 {
        painter.add(egui::Shape::convex_polygon(pts, fill.unwrap_or(egui::Color32::TRANSPARENT), stroke));
    }
}

fn apply_to_map(scene: &mut LoadedScene, i: usize, edits: &[CellEdit]) {
    if let Some(m) = tilemap_mut(scene, i) {
        apply_edits(m, edits, true);
    }
}

/// The per-frame handler. Returns true when the pointer was used (the caller
/// then skips picking). `cursor` is the pointer position over the viewport.
pub(crate) fn viewport(ui: &mut egui::Ui, scene: &mut LoadedScene, view: &View, response: &egui::Response, playing: bool) -> bool {
    let ctx = ui.ctx().clone();
    let mut st = TilePaletteState::get(&ctx);
    if !st.painting() || playing {
        return false;
    }
    let Some(i) = st.target.and_then(|t| scene.index_of(t)) else { return false };
    if tilemap(scene, i).is_none() {
        return false;
    }
    let id = scene.objects[i].id;
    let mut g = gesture(&ctx);
    let (shift, ctrl, alt) = ui.input(|inp| (inp.modifiers.shift, inp.modifiers.command, inp.modifiers.alt));
    let hover = response.hover_pos().filter(|_| !alt);
    let cell = hover.and_then(|p| cell_under(view, scene, i, p));
    let (pressed, down, released) = ui.input(|inp| (inp.pointer.primary_pressed(), inp.pointer.primary_down(), inp.pointer.primary_released()));
    let pressed = pressed && response.hovered() && !alt;
    let keys = response.hovered() && !ctx.wants_keyboard_input();
    let painter = ui.painter().clone();

    // Grid lines at the active cell size.
    if st.show_grid {
        let grid = grid_for(scene, i);
        let world = scene.world_transform(i);
        let inv = world.inverse();
        let corners = [view.rect.left_top(), view.rect.right_top(), view.rect.left_bottom(), view.rect.right_bottom()];
        let mut lo = Vec2::splat(f32::MAX);
        let mut hi = Vec2::splat(f32::MIN);
        let mut ok = true;
        for c in corners {
            match view.ray(c).and_then(|(o, d)| {
                let (lo_, ld) = (inv.transform_point3(o), inv.transform_vector3(d));
                (ld.z.abs() > 1e-8).then(|| lo_ + ld * (-lo_.z / ld.z))
            }) {
                Some(p) => {
                    lo = lo.min(Vec2::new(p.x, p.y));
                    hi = hi.max(Vec2::new(p.x, p.y));
                }
                None => ok = false,
            }
        }
        if ok {
            let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(28));
            for (a, b) in crate::grid_lines(&grid, lo, hi, 3000) {
                let (pa, pb) = (view.project(world.transform_point3(Vec3::new(a.x, a.y, 0.0))), view.project(world.transform_point3(Vec3::new(b.x, b.y, 0.0))));
                if let (Some(pa), Some(pb)) = (pa, pb) {
                    painter.line_segment([pa, pb], stroke);
                }
            }
        }
    }

    // Shortcut actions on the selection.
    let mut tool = st.tool;
    if ctrl && down && tool == TileTool::Paint {
        tool = TileTool::Pick;
    }
    if keys {
        let (copy, paste, delete, esc) = ui.input(|inp| (inp.modifiers.command && inp.key_pressed(egui::Key::C), inp.modifiers.command && inp.key_pressed(egui::Key::V), inp.key_pressed(egui::Key::Delete) || inp.key_pressed(egui::Key::Backspace), inp.key_pressed(egui::Key::Escape)));
        if let Some((a, b)) = g.selection {
            let sel = TileSelection::from_corners(a, b);
            if copy && let Some(m) = tilemap(scene, i) {
                st.clipboard = Some(pick_brush(m, sel));
                st.status = format!("Copied {} x {}", sel.size().x, sel.size().y);
            }
            if delete && let Some(m) = tilemap(scene, i) {
                let cells: Vec<IVec2> = sel.cells().collect();
                let edits = erase_cells(m, cells);
                if !edits.is_empty() {
                    apply_to_map(scene, i, &edits);
                    st.pending.push((id, edits));
                    mark_edited(&ctx);
                }
            }
        }
        if paste && let Some(c) = st.clipboard.clone() {
            st.brush = c;
            st.tool = TileTool::Paint;
            tool = TileTool::Paint;
        }
        if esc {
            g.anchor = None;
            g.selection = None;
        }
    }

    // Ghost: the brush footprint under the cursor.
    if let Some(c) = cell {
        let accent = egui::Stroke::new(2.0_f32, crate::theme::ACCENT_DEFAULT);
        let tint = Some(crate::theme::with_alpha(crate::theme::ACCENT_DEFAULT, 0.18));
        match tool {
            TileTool::Paint | TileTool::BoxFill | TileTool::Line if g.anchor.is_none() && st.has_brush() => {
                let origin = c - st.brush.pivot;
                for y in 0..st.brush.size.y {
                    for x in 0..st.brush.size.x {
                        if st.brush.get(x, y).is_some() {
                            outline(&painter, view, scene, i, origin + IVec2::new(x, y), accent, tint);
                        }
                    }
                }
            }
            _ => outline(&painter, view, scene, i, c, accent, tint),
        }
    }
    if let (Some(a), Some(c)) = (g.anchor, cell) {
        let cells: Vec<IVec2> = if tool == TileTool::Line { line_cells(a, c, false) } else { TileSelection::from_corners(a, c).cells().collect() };
        for p in cells {
            outline(&painter, view, scene, i, p, egui::Stroke::new(1.0_f32, egui::Color32::WHITE), Some(egui::Color32::from_white_alpha(20)));
        }
    }
    if let Some((a, b)) = g.selection {
        for p in TileSelection::from_corners(a, b).cells() {
            outline(&painter, view, scene, i, p, egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(90, 170, 255)), Some(egui::Color32::from_rgba_unmultiplied(90, 170, 255, 30)));
        }
    }
    hint_bar(&painter, view.rect, &format!("Tile Palette: {} - Shift erases, Ctrl picks, Esc cancels", tool.label()));

    let Some(c) = cell else {
        if released {
            finish_stroke(&mut st, &mut g, id);
        }
        set_gesture(&ctx, g);
        st.set(&ctx);
        return false;
    };

    // Tools.
    let erase_mode = shift && matches!(tool, TileTool::Paint | TileTool::BoxFill | TileTool::Line);
    let tool = if erase_mode && tool == TileTool::Paint { TileTool::Erase } else { tool };
    let mut used = false;
    match tool {
        TileTool::Paint | TileTool::Erase => {
            if pressed {
                g.stroke = Some(StrokeRecordBox::default());
                g.last_cell = None;
                st.stroking = true;
            }
            if down && let Some(stroke) = g.stroke.as_mut() {
                let path: Vec<IVec2> = match g.last_cell {
                    Some(l) if l != c => line_cells(l, c, false),
                    Some(_) => Vec::new(),
                    None => vec![c],
                };
                for cell in path {
                    let Some(m) = tilemap(scene, i) else { break };
                    let edits = if tool == TileTool::Erase { erase_cells(m, [cell]) } else if st.has_brush() { paint_brush_mode(m, &st.brush, cell, st.random, st.seed) } else { Vec::new() };
                    if !edits.is_empty() {
                        apply_to_map(scene, i, &edits);
                        stroke.0.push(edits);
                        mark_edited(&ctx);
                    }
                }
                g.last_cell = Some(c);
                used = true;
            }
            if released {
                finish_stroke(&mut st, &mut g, id);
            }
        }
        TileTool::BoxFill => {
            if pressed {
                g.anchor = Some(c);
                st.stroking = true;
            }
            if released && let Some(a) = g.anchor.take() {
                if let Some(m) = tilemap(scene, i) {
                    let edits = if shift {
                        erase_cells(m, TileSelection::from_corners(a, c).cells())
                    } else if st.has_brush() && st.random {
                        box_fill_random(m, a, c, &st.brush, st.seed)
                    } else if st.has_brush() {
                        box_fill(m, a, c, &st.brush)
                    } else {
                        Vec::new()
                    };
                    if !edits.is_empty() {
                        apply_to_map(scene, i, &edits);
                        st.pending.push((id, edits));
                        mark_edited(&ctx);
                    }
                }
                st.stroking = false;
            }
            used = down || released;
        }
        TileTool::Line => {
            if pressed {
                g.anchor = Some(c);
                st.stroking = true;
            }
            if released && let Some(a) = g.anchor.take() {
                if let Some(m) = tilemap(scene, i) {
                    let edits = if shift {
                        erase_cells(m, line_cells(a, c, false))
                    } else if st.has_brush() {
                        line_brush(m, a, c, &st.brush, st.random, st.seed)
                    } else {
                        Vec::new()
                    };
                    if !edits.is_empty() {
                        apply_to_map(scene, i, &edits);
                        st.pending.push((id, edits));
                        mark_edited(&ctx);
                    }
                }
                st.stroking = false;
            }
            used = down || released;
        }
        TileTool::Pick => {
            if pressed && let Some(m) = tilemap(scene, i) {
                let b = pick_brush(m, TileSelection::from_corners(c, c));
                if !b.is_empty() {
                    st.brush = b;
                    st.tool = TileTool::Paint;
                }
            }
            used = down;
        }
        TileTool::FloodFill => {
            if pressed && let Some(m) = tilemap(scene, i) {
                let limit = match m.bounds() {
                    Some((lo, hi)) => TileSelection { min: lo.min(c) - IVec2::splat(2), max: hi.max(c) + IVec2::splat(2) },
                    None => TileSelection { min: c - IVec2::splat(32), max: c + IVec2::splat(32) },
                };
                let edits = flood_fill(m, c, limit, 100_000, !ctrl, &st.brush);
                if !edits.is_empty() {
                    apply_to_map(scene, i, &edits);
                    st.pending.push((id, edits));
                    mark_edited(&ctx);
                }
            }
            used = down;
        }
        TileTool::Select => {
            if pressed {
                g.anchor = Some(c);
                g.selection = None;
            }
            if let Some(a) = g.anchor {
                if released {
                    g.selection = Some((a.min(c), a.max(c)));
                    g.anchor = None;
                }
                used = true;
            }
        }
        TileTool::Move => {
            if pressed {
                if g.selection.is_some_and(|(a, b)| TileSelection::from_corners(a, b).contains(c)) {
                    g.move_from = Some(c);
                    st.stroking = true;
                } else {
                    g.selection = None;
                }
            }
            if released && let (Some(from), Some((a, b))) = (g.move_from.take(), g.selection) {
                let delta = c - from;
                if let Some(m) = tilemap(scene, i) {
                    let edits = move_cells(m, TileSelection::from_corners(a, b), delta);
                    if !edits.is_empty() {
                        apply_to_map(scene, i, &edits);
                        st.pending.push((id, edits));
                        mark_edited(&ctx);
                    }
                    g.selection = Some((a + delta, b + delta));
                }
                st.stroking = false;
            }
            used = g.move_from.is_some() || released;
        }
    }
    set_gesture(&ctx, g);
    st.set(&ctx);
    used || pressed
}

fn finish_stroke(st: &mut TilePaletteState, g: &mut Gesture, id: ObjectId) {
    if let Some(s) = g.stroke.take() {
        let edits = s.0.finish();
        if !edits.is_empty() {
            st.pending.push((id, edits));
        }
    }
    g.last_cell = None;
    st.stroking = false;
}

#[allow(dead_code)]
fn _brush_is_used(_: &Brush) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creating_a_palette_needs_a_sprite_texture() {
        let dir = std::env::temp_dir().join(format!("zest_tile_edit_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // No such texture: a readable error, nothing written.
        let e = create_from_texture(&dir, "missing.png", "Tiles").unwrap_err();
        assert!(e.contains("missing.png"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
