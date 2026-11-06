//! Editor UI for tilemaps (2D-5): inspectors for `Grid`, `Tilemap`,
//! `TilemapRenderer` and `TilemapCollider2D`, the editors for `.tile` /
//! `.ruletile` assets (including the rule editor with its 3x3 neighbour grid
//! and a live match preview), and the pure helpers behind them.
//!
//! The logic worth testing lives in free functions ([`rule_slot_layout`],
//! [`preview_match`], [`move_rule`], [`grid_lines`] ...); the egui code only
//! lays them out. Nothing here was seen on screen: see the visual checks list
//! in the 2D-5 notes of `Features.md`.

use egui::{Color32, DragValue, RichText, Ui};
use glam::{IVec2, Vec2};
use zestengine_core::{
    CellLayout, CellSwizzle, CullingBounds, Grid, NeighborReq, RuleHit, RuleOutput, RuleTile, RuleTransform, TileAsset, TileColliderType, TileFile, TileSortOrder,
    Tilemap, TilemapCollider2D, TilemapMode, TilemapOrientation, TilemapRenderer, TilingRule, ring_spec, split_sprite_ref,
};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx};
use crate::file_browser::DRAG_SPRITE_KEY;
use crate::{theme, widgets};

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<Grid>();
    e.register::<Tilemap>();
    e.register::<TilemapRenderer>();
    e.register::<TilemapCollider2D>();
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Where ring slot `i` sits in the rule editor's 3x3-style grid, as
/// `(column, row)` with the centre at `(0, 0)` and rows growing upward. Hexagon
/// rings use the same cells (a flat hexagon drops the two side columns' middle).
pub fn rule_slot_layout(layout: CellLayout) -> Vec<(i32, i32)> {
    match layout {
        CellLayout::HexPointy => vec![(1, 0), (1, 1), (-1, 1), (-1, 0), (-1, -1), (1, -1)],
        CellLayout::HexFlat => vec![(1, 1), (0, 1), (-1, 1), (-1, -1), (0, -1), (1, -1)],
        _ => vec![(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
    }
}

/// Which rule (if any) and orientation the compiled table picks for a ring
/// pattern; `same[i]` = ring slot `i` holds this tile. The editor's live preview.
pub fn preview_match(tile: &RuleTile, same: &[bool]) -> RuleHit {
    let mask = same.iter().enumerate().fold(0u32, |m, (i, &s)| m | (s as u32) << i);
    tile.compile().lookup(mask)
}

/// A rule's neighbour list resized to the ring of `layout` (missing = Any).
pub fn fit_neighbors(rule: &mut TilingRule, layout: CellLayout) {
    rule.neighbors.resize(layout.neighbor_count(), NeighborReq::Any);
}

/// Move a rule from `from` to `to` (both clamped); returns the new index.
pub fn move_rule(rules: &mut Vec<TilingRule>, from: usize, to: usize) -> usize {
    if rules.is_empty() {
        return 0;
    }
    let from = from.min(rules.len() - 1);
    let to = to.min(rules.len() - 1);
    let r = rules.remove(from);
    rules.insert(to, r);
    to
}

/// Duplicate a rule right below itself; returns the copy's index.
pub fn duplicate_rule(rules: &mut Vec<TilingRule>, at: usize) -> usize {
    let at = at.min(rules.len().saturating_sub(1));
    if rules.is_empty() {
        rules.push(TilingRule::default());
        return 0;
    }
    let c = rules[at].clone();
    rules.insert(at + 1, c);
    at + 1
}

/// Delete a rule; returns the index to select afterwards.
pub fn delete_rule(rules: &mut Vec<TilingRule>, at: usize) -> usize {
    if at < rules.len() {
        rules.remove(at);
    }
    at.min(rules.len().saturating_sub(1))
}

/// Segments of the grid lines inside the local rectangle `lo..hi`, at most
/// `cap` of them: full lines for rectangle grids, cell outlines for the other
/// layouts (which have no straight lattice lines).
pub fn grid_lines(grid: &Grid, lo: Vec2, hi: Vec2, cap: usize) -> Vec<(Vec2, Vec2)> {
    let mut out = Vec::new();
    let st = Vec2::new(grid.cell_size[0] + grid.cell_gap[0], grid.cell_size[1] + grid.cell_gap[1]);
    if st.x <= 1e-6 || st.y <= 1e-6 || hi.x < lo.x || hi.y < lo.y {
        return out;
    }
    if grid.layout == CellLayout::Rectangle && grid.swizzle == CellSwizzle::XYZ {
        let (x0, x1) = ((lo.x / st.x).floor() as i32, (hi.x / st.x).ceil() as i32);
        let (y0, y1) = ((lo.y / st.y).floor() as i32, (hi.y / st.y).ceil() as i32);
        for x in x0..=x1 {
            if out.len() >= cap {
                return out;
            }
            out.push((Vec2::new(x as f32 * st.x, lo.y), Vec2::new(x as f32 * st.x, hi.y)));
        }
        for y in y0..=y1 {
            if out.len() >= cap {
                return out;
            }
            out.push((Vec2::new(lo.x, y as f32 * st.y), Vec2::new(hi.x, y as f32 * st.y)));
        }
        return out;
    }
    let a = grid.local_to_cell(glam::Vec3::new(lo.x, lo.y, 0.0));
    let b = grid.local_to_cell(glam::Vec3::new(hi.x, hi.y, 0.0));
    let c = grid.local_to_cell(glam::Vec3::new(lo.x, hi.y, 0.0));
    let d = grid.local_to_cell(glam::Vec3::new(hi.x, lo.y, 0.0));
    let (mn, mx) = (IVec2::new(a.x.min(b.x).min(c.x).min(d.x), a.y.min(b.y).min(c.y).min(d.y)) - IVec2::ONE, IVec2::new(a.x.max(b.x).max(c.x).max(d.x), a.y.max(b.y).max(c.y).max(d.y)) + IVec2::ONE);
    for y in mn.y..=mx.y {
        for x in mn.x..=mx.x {
            let pts = grid.cell_corners(glam::IVec3::new(x, y, 0));
            for k in 0..pts.len() {
                if out.len() >= cap {
                    return out;
                }
                let (p, q) = (pts[k], pts[(k + 1) % pts.len()]);
                out.push((Vec2::new(p.x, p.y), Vec2::new(q.x, q.y)));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Small widgets
// ---------------------------------------------------------------------------

fn combo<T: Copy + PartialEq>(ui: &mut Ui, label: &str, salt: &str, v: &mut T, all: &[T], name: fn(T) -> &'static str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(("zestengine-tilemap", salt)).selected_text(name(*v)).show_ui(ui, |ui| {
            for &o in all {
                if ui.selectable_label(*v == o, name(o)).clicked() && *v != o {
                    *v = o;
                    *changed = true;
                }
            }
        });
    });
}

fn vec3_row(ui: &mut Ui, label: &str, v: &mut [f32; 3], speed: f64, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        ui.horizontal(|ui| {
            for x in v.iter_mut() {
                *changed |= ui.add(DragValue::new(x).speed(speed).max_decimals(3)).changed();
            }
        });
    });
}

fn sprite_field(ui: &mut Ui, label: &str, value: &mut String, changed: &mut bool) {
    let dragging: Option<String> = ui.data(|d| d.get_temp(egui::Id::new(DRAG_SPRITE_KEY)));
    widgets::property_row(ui, label, |ui| {
        ui.horizontal(|ui| {
            let reserve = if value.is_empty() { 0.0 } else { 44.0 };
            let r = ui.add(egui::TextEdit::singleline(value).hint_text("drop a sprite").desired_width(ui.available_width() - reserve));
            *changed |= r.changed();
            if dragging.is_some() && ui.rect_contains_pointer(r.rect) && ui.input(|i| i.pointer.any_released())
                && let Some(t) = &dragging {
                    *value = t.clone();
                    *changed = true;
                }
            if !value.is_empty() && ui.small_button(egui_phosphor::regular::X).on_hover_text("Clear").clicked() {
                value.clear();
                *changed = true;
            }
        });
    });
    if !value.is_empty() && split_sprite_ref(value).is_none() {
        ui.label(RichText::new("Not a sprite reference (path#16-hex-id)").size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    }
}

// ---------------------------------------------------------------------------
// Component inspectors
// ---------------------------------------------------------------------------

impl Inspect for Grid {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        combo(ui, "Cell Layout", "layout", &mut self.layout, &CellLayout::ALL, CellLayout::label, &mut changed);
        vec3_row(ui, "Cell Size", &mut self.cell_size, 0.01, &mut changed);
        vec3_row(ui, "Cell Gap", &mut self.cell_gap, 0.01, &mut changed);
        combo(ui, "Cell Swizzle", "swizzle", &mut self.swizzle, &CellSwizzle::ALL, CellSwizzle::label, &mut changed);
        changed
    }
}
impl Gizmo for Grid {}

impl Inspect for Tilemap {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        combo(ui, "Orientation", "orient", &mut self.orientation, &TilemapOrientation::ALL, TilemapOrientation::label, &mut changed);
        widgets::property_row(ui, "Tile Anchor", |ui| {
            ui.horizontal(|ui| {
                for v in self.anchor.iter_mut() {
                    changed |= ui.add(DragValue::new(v).speed(0.01).range(-4.0..=4.0)).changed();
                }
            });
        });
        widgets::property_row(ui, "Color", |ui| changed |= ui.color_edit_button_rgba_unmultiplied(&mut self.color).changed());
        widgets::property_row(ui, "Anim Frame Rate", |ui| changed |= ui.add(DragValue::new(&mut self.animation_frame_rate).speed(0.05).range(0.0..=120.0)).changed());
        widgets::property_row(ui, "Anim Offset", |ui| changed |= ui.add(DragValue::new(&mut self.animation_time_offset).speed(0.05)).changed());
        let (cells, chunks) = (self.cell_count(), self.chunk_count());
        let bounds = self.bounds().map(|(lo, hi)| format!("{} x {}  ({},{}) to ({},{})", hi.x - lo.x + 1, hi.y - lo.y + 1, lo.x, lo.y, hi.x, hi.y)).unwrap_or_else(|| "empty".into());
        widgets::property_row(ui, "Bounds", |ui| ui.label(bounds));
        widgets::property_row(ui, "Tiles", |ui| ui.label(format!("{cells} cells, {chunks} chunks, {} tile assets", self.tile_table().len())));
        ui.horizontal(|ui| {
            if ui.button("Compress Bounds").on_hover_text("Shrink the bounds to the cells in use").clicked() {
                self.compress_bounds();
                changed = true;
            }
            if ui.button("Clear All Tiles").clicked() {
                self.clear();
                changed = true;
            }
        });
        if widgets::secondary_button(ui, None, "Open Tile Palette").clicked() {
            let mut st = crate::TilePaletteState::get(ui.ctx());
            st.open = true;
            st.set(ui.ctx());
        }
        ui.label(RichText::new("Paint in the Scene view while the Tile Palette is open and this tilemap is its active target.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        changed
    }
}
impl Gizmo for Tilemap {}

impl Inspect for TilemapRenderer {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let layers: Vec<(u32, String)> = ui.data(|d| d.get_temp(egui::Id::new(crate::SORTING_LAYERS_KEY))).unwrap_or_else(|| vec![(0, "Default".to_string())]);
        let current = layers.iter().find(|(i, _)| *i == self.sorting_layer.0).map(|(_, n)| n.clone()).unwrap_or_else(|| "(missing)".into());
        widgets::property_row(ui, "Sorting Layer", |ui| {
            egui::ComboBox::from_id_salt("zestengine-tilemap-layer").selected_text(current).show_ui(ui, |ui| {
                for (id, name) in &layers {
                    if ui.selectable_label(self.sorting_layer.0 == *id, name).clicked() && self.sorting_layer.0 != *id {
                        self.sorting_layer = zestengine_core::SortingLayerId(*id);
                        changed = true;
                    }
                }
            });
        });
        widgets::property_row(ui, "Order in Layer", |ui| {
            let mut x = self.order_in_layer as i32;
            if ui.add(DragValue::new(&mut x).speed(0.2).range(i16::MIN as i32..=i16::MAX as i32)).changed() {
                self.order_in_layer = x as i16;
                changed = true;
            }
        });
        combo(ui, "Mode", "mode", &mut self.mode, &TilemapMode::ALL, TilemapMode::label, &mut changed);
        combo(ui, "Sort Order", "sort", &mut self.sort_order, &TileSortOrder::ALL, TileSortOrder::label, &mut changed);
        combo(ui, "Mask Interaction", "mask", &mut self.mask_interaction, &zestengine_core::MaskInteraction2D::ALL, zestengine_core::MaskInteraction2D::label, &mut changed);
        let mut manual = self.detect_chunk_culling_bounds == CullingBounds::Manual;
        widgets::property_row(ui, "Chunk Culling", |ui| {
            let auto = !manual;
            if ui.checkbox(&mut manual, "Manual").on_hover_text(if auto { "Auto: the chunk's real tile geometry" } else { "Adds the margin below" }).changed() {
                self.detect_chunk_culling_bounds = if manual { CullingBounds::Manual } else { CullingBounds::Auto };
                changed = true;
            }
        });
        if manual {
            widgets::property_row(ui, "Culling Margin", |ui| changed |= ui.add(DragValue::new(&mut self.chunk_culling_margin).speed(0.05).range(0.0..=100.0)).changed());
        }
        widgets::property_row(ui, "Max Chunk Count", |ui| changed |= ui.add(DragValue::new(&mut self.max_chunk_count).speed(1.0).range(1..=4096)).changed());
        widgets::property_row(ui, "Max Frame Age", |ui| changed |= ui.add(DragValue::new(&mut self.max_frame_age).speed(1.0).range(1..=10_000)).changed());
        widgets::property_row(ui, "Material", |ui| changed |= ui.add(egui::TextEdit::singleline(&mut self.material).hint_text("default sprite material (not used yet)").desired_width(ui.available_width())).changed());
        changed
    }
}
impl Gizmo for TilemapRenderer {}

impl Inspect for TilemapCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Is Trigger", |ui| changed |= ui.checkbox(&mut self.is_trigger, "").changed());
        widgets::property_row(ui, "Used By Effector", |ui| changed |= ui.checkbox(&mut self.used_by_effector, "").changed());
        widgets::property_row(ui, "In Composite", |ui| changed |= ui.checkbox(&mut self.used_by_composite, "").on_hover_text("Merge every chunk into the sibling Composite Collider 2D's one collider").changed());
        changed |= ctx.file_field(ui, "Physics Material", &mut self.material, "body's or project default", &["physmat"]);
        widgets::property_row(ui, "Extrusion Factor", |ui| changed |= ui.add(DragValue::new(&mut self.extrusion_factor).speed(0.001).range(0.0..=1.0).max_decimals(4)).changed());
        widgets::property_row(ui, "Merge Grid Cells", |ui| changed |= ui.checkbox(&mut self.merge_cells, "").on_hover_text("Greedy-merge Grid colliders into rectangles").changed());
        widgets::property_row(ui, "Hollow Outlines", |ui| changed |= ui.checkbox(&mut self.outlines, "").on_hover_text("Grid colliders as boundary chains; fast bodies may tunnel inside").changed());
        widgets::property_row(ui, "Max Changes", |ui| changed |= ui.add(DragValue::new(&mut self.maximum_tile_change_count).speed(10.0).range(1..=1_000_000)).changed());
        ui.label(RichText::new("One compound collider per 16 x 16 chunk; edits rebuild only the changed chunks.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        changed
    }
}
impl Gizmo for TilemapCollider2D {}

// ---------------------------------------------------------------------------
// Tile asset editors
// ---------------------------------------------------------------------------

fn collider_combo(ui: &mut Ui, salt: &str, v: &mut TileColliderType, changed: &mut bool) {
    combo(ui, "Collider Type", salt, v, &TileColliderType::ALL, TileColliderType::label, changed);
}

/// Editor for a `.tile` (Tile and Animated Tile). True when changed.
pub fn tile_asset_ui(ui: &mut Ui, t: &mut TileAsset) -> bool {
    let mut changed = false;
    let animated = t.animation.is_some();
    let mut want_anim = animated;
    widgets::property_row(ui, "Animated", |ui| ui.checkbox(&mut want_anim, ""));
    if want_anim != animated {
        t.animation = want_anim.then(|| zestengine_core::TileAnimation { sprites: if t.sprite.is_empty() { Vec::new() } else { vec![t.sprite.clone()] }, ..Default::default() });
        changed = true;
    }
    if let Some(a) = &mut t.animation {
        let mut remove = None;
        for (i, s) in a.sprites.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{i}")).size(theme::SIZE_META).color(theme::TEXT_DIM));
                changed |= ui.add(egui::TextEdit::singleline(s).desired_width(ui.available_width() - 44.0)).changed();
                if ui.small_button(egui_phosphor::regular::TRASH).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            a.sprites.remove(i);
            changed = true;
        }
        let mut dropped = String::new();
        sprite_field(ui, "Add Frame", &mut dropped, &mut false);
        if split_sprite_ref(&dropped).is_some() {
            a.sprites.push(dropped);
            changed = true;
        }
        widgets::property_row(ui, "Speed (min / max)", |ui| {
            ui.horizontal(|ui| {
                changed |= ui.add(DragValue::new(&mut a.min_speed).speed(0.05).range(0.0..=100.0)).changed();
                changed |= ui.add(DragValue::new(&mut a.max_speed).speed(0.05).range(0.0..=100.0)).changed();
            });
        });
        widgets::property_row(ui, "Start Time", |ui| changed |= ui.add(DragValue::new(&mut a.start_time).speed(0.05)).changed());
        widgets::property_row(ui, "Loop Once", |ui| changed |= ui.checkbox(&mut a.loop_once, "").changed());
    } else {
        sprite_field(ui, "Sprite", &mut t.sprite, &mut changed);
    }
    widgets::property_row(ui, "Color", |ui| changed |= ui.color_edit_button_rgba_unmultiplied(&mut t.color).changed());
    collider_combo(ui, "tcol", &mut t.collider, &mut changed);
    widgets::property_row(ui, "Game Object", |ui| {
        changed |= ui.add(egui::TextEdit::singleline(&mut t.game_object).hint_text("prefabs/x.prefab").desired_width(ui.available_width())).on_hover_text("A prefab placed once in every cell that uses this tile (a child of the tilemap, removed when the cell is erased)").changed();
    });
    widgets::property_row(ui, "Lock Color", |ui| {
        let mut on = t.flags & zestengine_core::FLAG_LOCK_COLOR != 0;
        if ui.checkbox(&mut on, "").changed() {
            t.flags = (t.flags & !zestengine_core::FLAG_LOCK_COLOR) | (on as u8 * zestengine_core::FLAG_LOCK_COLOR);
            changed = true;
        }
    });
    widgets::property_row(ui, "Lock Transform", |ui| {
        let mut on = t.flags & zestengine_core::FLAG_LOCK_TRANSFORM != 0;
        if ui.checkbox(&mut on, "").changed() {
            t.flags = (t.flags & !zestengine_core::FLAG_LOCK_TRANSFORM) | (on as u8 * zestengine_core::FLAG_LOCK_TRANSFORM);
            changed = true;
        }
    });
    widgets::property_row(ui, "Transform", |ui| {
        ui.horizontal(|ui| {
            let mut quarter = t.transform.rot / 6;
            if ui.add(DragValue::new(&mut quarter).range(0..=3).suffix(" x 90"))
                .changed()
            {
                t.transform.rot = quarter * 6;
                changed = true;
            }
            changed |= ui.checkbox(&mut t.transform.mirror, "Mirror X").changed();
        });
    });
    changed
}

fn req_color(r: NeighborReq) -> Color32 {
    match r {
        NeighborReq::Any => theme::LINE_BUTTON,
        NeighborReq::This => Color32::from_rgb(60, 170, 90),
        NeighborReq::NotThis => Color32::from_rgb(200, 70, 70),
    }
}

fn req_glyph(r: NeighborReq) -> &'static str {
    match r {
        NeighborReq::Any => "",
        NeighborReq::This => egui_phosphor::regular::ARROW_UP,
        NeighborReq::NotThis => egui_phosphor::regular::X,
    }
}

/// The clickable neighbour grid of one rule: clicking a slot cycles
/// Any / This / Not This. The centre cell is the tile itself.
fn neighbor_grid(ui: &mut Ui, layout: CellLayout, rule: &mut TilingRule, changed: &mut bool) {
    fit_neighbors(rule, layout);
    let slots = rule_slot_layout(layout);
    let cell = 26.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(cell * 3.0 + 4.0, cell * 3.0 + 4.0), egui::Sense::hover());
    let at = |c: i32, r: i32| egui::Rect::from_min_size(rect.min + egui::vec2((c + 1) as f32 * cell + 2.0, (1 - r) as f32 * cell + 2.0), egui::vec2(cell - 2.0, cell - 2.0));
    ui.painter().rect_filled(at(0, 0), 3.0, theme::ACCENT_DEFAULT);
    for (i, &(c, r)) in slots.iter().enumerate() {
        let rc = at(c, r);
        let resp = ui.interact(rc, ui.id().with(("rule-nb", i)), egui::Sense::click());
        let req = rule.neighbors[i];
        ui.painter().rect_filled(rc, 3.0, theme::BG_BUTTON);
        ui.painter().rect_stroke(rc, 3.0, egui::Stroke::new(1.5_f32, req_color(req)), egui::StrokeKind::Inside);
        let g = req_glyph(req);
        if !g.is_empty() {
            ui.painter().text(rc.center(), egui::Align2::CENTER_CENTER, g, egui::FontId::proportional(14.0), req_color(req));
        }
        if resp.on_hover_text(req.label()).clicked() {
            rule.neighbors[i] = req.cycled();
            *changed = true;
        }
    }
}

/// Per-editor state (selected rule, preview pattern); keep it in egui memory.
#[derive(Clone, Default)]
struct RuleEditState {
    selected: usize,
    preview: Vec<bool>,
}

/// Editor for a `.ruletile`. True when changed.
pub fn rule_tile_ui(ui: &mut Ui, t: &mut RuleTile) -> bool {
    let mut changed = false;
    let id = ui.id().with("rule-tile-editor");
    let mut st: RuleEditState = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    let layouts = [CellLayout::Rectangle, CellLayout::Isometric, CellLayout::HexPointy, CellLayout::HexFlat];
    let before_layout = t.layout;
    combo(ui, "Grid Layout", "rlayout", &mut t.layout, &layouts, CellLayout::label, &mut changed);
    if t.layout != before_layout {
        for r in &mut t.rules {
            fit_neighbors(r, t.layout);
        }
    }
    sprite_field(ui, "Default Sprite", &mut t.default_sprite, &mut changed);
    collider_combo(ui, "rcol", &mut t.default_collider, &mut changed);
    widgets::property_row(ui, "Color", |ui| changed |= ui.color_edit_button_rgba_unmultiplied(&mut t.color).changed());
    widgets::eyebrow(ui, "Tiling Rules");
    st.selected = st.selected.min(t.rules.len().saturating_sub(1));
    let mut op: Option<(usize, u8)> = None; // 0 up 1 down 2 dup 3 delete
    for i in 0..t.rules.len() {
        let sel = i == st.selected;
        ui.horizontal(|ui| {
            if widgets::segment(ui, sel, &format!("Rule {}", i + 1)).clicked() {
                st.selected = i;
            }
            let n = t.rules.len();
            if ui.add_enabled(i > 0, egui::Button::new(egui_phosphor::regular::ARROW_UP).small()).clicked() {
                op = Some((i, 0));
            }
            if ui.add_enabled(i + 1 < n, egui::Button::new(egui_phosphor::regular::ARROW_DOWN).small()).clicked() {
                op = Some((i, 1));
            }
            if ui.small_button(egui_phosphor::regular::COPY).on_hover_text("Duplicate").clicked() {
                op = Some((i, 2));
            }
            if ui.small_button(egui_phosphor::regular::TRASH).on_hover_text("Delete").clicked() {
                op = Some((i, 3));
            }
        });
        if sel {
            ui.indent(("rule-body", i), |ui| {
                let layout = t.layout;
                let r = &mut t.rules[i];
                ui.horizontal(|ui| {
                    neighbor_grid(ui, layout, r, &mut changed);
                    ui.vertical(|ui| {
                        combo(ui, "Transform", "rxf", &mut r.transform, &RuleTransform::ALL, RuleTransform::label, &mut changed);
                        combo(ui, "Output", "rout", &mut r.output, &RuleOutput::ALL, RuleOutput::label, &mut changed);
                        collider_combo(ui, "rrc", &mut r.collider, &mut changed);
                    });
                });
                let mut remove = None;
                for (k, s) in r.sprites.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        changed |= ui.add(egui::TextEdit::singleline(s).desired_width(ui.available_width() - 44.0)).changed();
                        if ui.small_button(egui_phosphor::regular::TRASH).clicked() {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    r.sprites.remove(k);
                    changed = true;
                }
                let mut dropped = String::new();
                sprite_field(ui, if r.output == RuleOutput::Single { "Sprite" } else { "Add Sprite" }, &mut dropped, &mut false);
                if split_sprite_ref(&dropped).is_some() {
                    if r.output == RuleOutput::Single {
                        r.sprites.clear();
                    }
                    r.sprites.push(dropped);
                    changed = true;
                }
                match r.output {
                    RuleOutput::Animation => {
                        widgets::property_row(ui, "Speed (min / max)", |ui| {
                            ui.horizontal(|ui| {
                                changed |= ui.add(DragValue::new(&mut r.min_speed).speed(0.05)).changed();
                                changed |= ui.add(DragValue::new(&mut r.max_speed).speed(0.05)).changed();
                            });
                        });
                    }
                    RuleOutput::Random => {
                        widgets::property_row(ui, "Noise Scale", |ui| changed |= ui.add(DragValue::new(&mut r.random_scale).speed(0.01).range(0.001..=100.0)).changed());
                        widgets::property_row(ui, "Random Rotation", |ui| changed |= ui.checkbox(&mut r.random_rotation, "").changed());
                    }
                    RuleOutput::Single => {}
                }
            });
        }
    }
    if let Some((i, o)) = op {
        match o {
            0 => st.selected = move_rule(&mut t.rules, i, i.saturating_sub(1)),
            1 => st.selected = move_rule(&mut t.rules, i, i + 1),
            2 => st.selected = duplicate_rule(&mut t.rules, i),
            _ => st.selected = delete_rule(&mut t.rules, i),
        }
        changed = true;
    }
    if ui.button(format!("{} Add Rule", egui_phosphor::regular::PLUS)).clicked() {
        let mut r = TilingRule::default();
        fit_neighbors(&mut r, t.layout);
        t.rules.push(r);
        st.selected = t.rules.len() - 1;
        changed = true;
    }
    // Live preview: click the ring to place "same tile" neighbours, see which rule wins.
    widgets::eyebrow(ui, "Preview");
    let n = ring_spec(t.layout).n;
    st.preview.resize(n, false);
    let slots = rule_slot_layout(t.layout);
    let cell = 22.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(cell * 3.0 + 4.0, cell * 3.0 + 4.0), egui::Sense::hover());
    let at = |c: i32, r: i32| egui::Rect::from_min_size(rect.min + egui::vec2((c + 1) as f32 * cell + 2.0, (1 - r) as f32 * cell + 2.0), egui::vec2(cell - 2.0, cell - 2.0));
    ui.painter().rect_filled(at(0, 0), 3.0, theme::ACCENT_DEFAULT);
    for (i, &(c, r)) in slots.iter().enumerate() {
        let rc = at(c, r);
        let resp = ui.interact(rc, ui.id().with(("rule-pv", i)), egui::Sense::click());
        ui.painter().rect_filled(rc, 3.0, if st.preview[i] { theme::ACCENT_DEFAULT } else { theme::BG_BUTTON });
        if resp.clicked() {
            st.preview[i] = !st.preview[i];
        }
    }
    let hit = preview_match(t, &st.preview);
    let text = if hit.matched() {
        format!("Matches Rule {} (rotation {} x 15 deg{})", hit.rule as usize + 1, hit.xform.rot, if hit.xform.mirror { ", mirrored" } else { "" })
    } else {
        "No rule matches: the default sprite is used".to_string()
    };
    ui.label(RichText::new(text).size(theme::SIZE_META));
    ui.data_mut(|d| d.insert_temp(id, st));
    changed
}

/// Inspector body for a selected `.tile` / `.ruletile` file: loads, edits and
/// saves it. True when the file was rewritten.
pub fn tile_file_ui(ui: &mut Ui, path: &std::path::Path) -> bool {
    let mut file = TileFile::load(path).unwrap_or_else(|_| if path.extension().is_some_and(|e| e == "ruletile") { TileFile::Rule(RuleTile::default()) } else { TileFile::Tile(TileAsset::default()) });
    let changed = match &mut file {
        TileFile::Tile(t) => {
            ui.heading(if t.animation.is_some() { "Animated Tile" } else { "Tile" });
            tile_asset_ui(ui, t)
        }
        TileFile::Rule(r) => {
            ui.heading("Rule Tile");
            rule_tile_ui(ui, r)
        }
    };
    if changed && let Err(e) = file.save(path) {
        tracing::error!("saving tile: {e}");
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_slots_cover_the_ring_without_overlap() {
        for (layout, n) in [(CellLayout::Rectangle, 8), (CellLayout::Isometric, 8), (CellLayout::HexPointy, 6), (CellLayout::HexFlat, 6)] {
            let s = rule_slot_layout(layout);
            assert_eq!(s.len(), n);
            let set: std::collections::HashSet<_> = s.iter().collect();
            assert_eq!(set.len(), n);
            assert!(!set.contains(&(0, 0)) && s.iter().all(|&(c, r)| c.abs() <= 1 && r.abs() <= 1));
        }
    }

    #[test]
    fn rule_list_operations() {
        let mk = |n: usize| TilingRule { sprites: vec![format!("s{n}")], ..Default::default() };
        let mut r: Vec<_> = (0..4).map(mk).collect();
        assert_eq!(move_rule(&mut r, 3, 0), 0);
        assert_eq!(r[0].sprites[0], "s3");
        assert_eq!(move_rule(&mut r, 0, 99), 3, "clamped");
        let n = duplicate_rule(&mut r, 1);
        assert_eq!((n, r.len(), r[1].sprites == r[2].sprites), (2, 5, true));
        assert_eq!(delete_rule(&mut r, 4), 3);
        assert_eq!(r.len(), 4);
        let mut empty: Vec<TilingRule> = Vec::new();
        assert_eq!(duplicate_rule(&mut empty, 0), 0);
        assert_eq!(delete_rule(&mut empty, 5), 0);
    }

    #[test]
    fn preview_shows_the_winning_rule_and_orientation() {
        let mut edge = TilingRule { transform: RuleTransform::Rotated, ..Default::default() };
        edge.neighbors[6] = NeighborReq::This; // south
        edge.neighbors[2] = NeighborReq::NotThis; // north
        let t = RuleTile { rules: vec![edge], ..Default::default() };
        let mut same = vec![false; 8];
        same[6] = true;
        let h = preview_match(&t, &same);
        assert_eq!((h.rule, h.xform.rot), (0, 0));
        same[6] = false;
        same[0] = true; // east
        assert_eq!(preview_match(&t, &same).xform.rot, 6, "rotated a quarter turn");
        same[0] = false;
        assert!(!preview_match(&t, &same).matched() || same.iter().all(|s| !s));
    }

    #[test]
    fn fit_neighbors_resizes_to_the_ring() {
        let mut r = TilingRule::default();
        fit_neighbors(&mut r, CellLayout::HexPointy);
        assert_eq!(r.neighbors.len(), 6);
        fit_neighbors(&mut r, CellLayout::Rectangle);
        assert_eq!(r.neighbors.len(), 8);
    }

    #[test]
    fn grid_lines_follow_the_cell_size_and_respect_the_cap() {
        let g = Grid { cell_size: [2.0, 1.0, 1.0], ..Default::default() };
        let l = grid_lines(&g, Vec2::new(-3.0, -1.5), Vec2::new(3.0, 1.5), 1000);
        // x lines at -4,-2,0,2,4 and y lines at -2..2.
        assert_eq!(l.len(), 5 + 5);
        assert!(l.iter().any(|(a, b)| a.x == b.x && a.x == 2.0));
        assert_eq!(grid_lines(&g, Vec2::splat(-50.0), Vec2::splat(50.0), 7).len(), 7);
        let h = Grid { layout: CellLayout::HexPointy, ..Default::default() };
        let l = grid_lines(&h, Vec2::splat(-1.0), Vec2::splat(1.0), 10_000);
        assert!(!l.is_empty() && l.len().is_multiple_of(6));
        assert!(grid_lines(&g, Vec2::splat(1.0), Vec2::splat(0.0), 10).is_empty());
    }
}
