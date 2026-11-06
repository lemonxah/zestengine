//! Inspectors for the 2D sprite components (`SpriteRenderer`, `SortingGroup`,
//! `SpriteMask`) and the Sorting Layers list the Layers window shows.
//!
//! The sorting layers live in the scene (`LayerSettings.sorting`); the engine
//! publishes a copy into egui memory each frame ([`publish_sorting_layers`], a
//! plain `Vec<(u32, String)>` so it crosses plugin boundaries like the other
//! `*_KEY` slots) and the layer popups read it.

use egui::{DragValue, RichText, Ui};
use zestengine_core::{
    MaskInteraction2D, SortingGroup, SortingLayerId, SortingLayers, SpriteDrawMode, SpriteMask, SpriteMaskRange, SpriteRenderer, SpriteSortPoint, SpriteTileMode,
    split_sprite_ref,
};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx};
use crate::file_browser::DRAG_SPRITE_KEY;
use crate::{theme, widgets};

/// egui temp data `Vec<(u32, String)>`: the scene's sorting layers, back to front.
pub const SORTING_LAYERS_KEY: &str = "zestengine-sorting-layers";

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<SpriteRenderer>();
    e.register::<SortingGroup>();
    e.register::<SpriteMask>();
}

/// Publish the scene's sorting layers for the inspectors (call once per frame
/// before they draw).
pub fn publish_sorting_layers(ctx: &egui::Context, layers: &SortingLayers) {
    let list: Vec<(u32, String)> = layers.layers.iter().map(|l| (l.id.0, l.name.clone())).collect();
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(SORTING_LAYERS_KEY), list));
}

fn published(ui: &Ui) -> Vec<(u32, String)> {
    ui.data(|d| d.get_temp::<Vec<(u32, String)>>(egui::Id::new(SORTING_LAYERS_KEY))).unwrap_or_else(|| vec![(0, "Default".to_string())])
}

pub(crate) fn layer_row(ui: &mut Ui, label: &str, salt: &str, id: &mut SortingLayerId, changed: &mut bool) {
    let layers = published(ui);
    let current = layers.iter().find(|(i, _)| *i == id.0).map(|(_, n)| n.clone()).unwrap_or_else(|| format!("(missing {:08x})", id.0));
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(("zestengine-sorting-layer", salt)).selected_text(current).show_ui(ui, |ui| {
            for (lid, name) in &layers {
                if ui.selectable_label(id.0 == *lid, name).clicked() && id.0 != *lid {
                    *id = SortingLayerId(*lid);
                    *changed = true;
                }
            }
        });
    });
}

pub(crate) fn order_row(ui: &mut Ui, label: &str, v: &mut i16, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        let mut x = *v as i32;
        if ui.add(DragValue::new(&mut x).speed(0.2).range(i16::MIN as i32..=i16::MAX as i32)).changed() {
            *v = x as i16;
            *changed = true;
        }
    });
}

pub(crate) fn combo<T: Copy + PartialEq>(ui: &mut Ui, label: &str, salt: &str, v: &mut T, all: &[T], name: fn(T) -> &'static str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(("zestengine-sprite2d", salt)).selected_text(name(*v)).show_ui(ui, |ui| {
            for &o in all {
                if ui.selectable_label(*v == o, name(o)).clicked() && *v != o {
                    *v = o;
                    *changed = true;
                }
            }
        });
    });
}

/// A sprite reference slot: a text field holding `path#id`, a drop target for
/// the sprite children dragged out of the Files grid, and a clear button.
pub(crate) fn sprite_slot(ui: &mut Ui, label: &str, value: &mut String, changed: &mut bool) {
    let dragging: Option<String> = ui.data(|d| d.get_temp(egui::Id::new(DRAG_SPRITE_KEY)));
    widgets::property_row(ui, label, |ui| {
        ui.horizontal(|ui| {
            let reserve = if value.is_empty() { 0.0 } else { 44.0 };
            let r = ui.add(egui::TextEdit::singleline(value).hint_text("drop a sprite here").desired_width(ui.available_width() - reserve));
            if r.changed() {
                *changed = true;
            }
            if dragging.is_some() && ui.rect_contains_pointer(r.rect) {
                ui.painter().rect_stroke(r.rect, 4.0, egui::Stroke::new(2.0_f32, ui.visuals().selection.stroke.color), egui::StrokeKind::Inside);
                if ui.input(|i| i.pointer.any_released())
                    && let Some(text) = &dragging
                {
                    *value = text.clone();
                    *changed = true;
                }
            }
            if !value.is_empty() && ui.small_button(egui_phosphor::regular::X).on_hover_text("Clear").clicked() {
                value.clear();
                *changed = true;
            }
        });
    });
    if !value.is_empty() && split_sprite_ref(value).is_none() {
        ui.label(RichText::new("Not a sprite reference (expected path#16-hex-id)").size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    }
}

impl Inspect for SpriteRenderer {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        sprite_slot(ui, "Sprite", &mut self.sprite, &mut changed);
        widgets::property_row(ui, "Color", |ui| {
            changed |= ui.color_edit_button_rgba_unmultiplied(&mut self.color).changed();
        });
        widgets::property_row(ui, "Flip", |ui| {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut self.flip_x, "X").changed();
                changed |= ui.checkbox(&mut self.flip_y, "Y").changed();
            });
        });
        combo(ui, "Draw Mode", "draw", &mut self.draw_mode, &SpriteDrawMode::ALL, SpriteDrawMode::label, &mut changed);
        if self.draw_mode != SpriteDrawMode::Simple {
            widgets::property_row(ui, "Size", |ui| {
                ui.horizontal(|ui| {
                    for v in self.size.iter_mut() {
                        changed |= ui.add(DragValue::new(v).speed(0.02).range(0.0..=10_000.0).max_decimals(3)).changed();
                    }
                });
            });
            ui.label(RichText::new("Sliced / Tiled need a Full Rect sprite with a border (Sprite Editor).").size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
        if self.draw_mode == SpriteDrawMode::Tiled {
            combo(ui, "Tile Mode", "tile", &mut self.tile_mode, &SpriteTileMode::ALL, SpriteTileMode::label, &mut changed);
            if self.tile_mode == SpriteTileMode::Adaptive {
                widgets::property_row(ui, "Stretch Value", |ui| {
                    changed |= ui.add(DragValue::new(&mut self.adaptive_threshold).speed(0.01).range(0.0..=16.0)).changed();
                });
            }
        }
        combo(ui, "Mask Interaction", "mask", &mut self.mask_interaction, &MaskInteraction2D::ALL, MaskInteraction2D::label, &mut changed);
        combo(ui, "Sort Point", "sortpt", &mut self.sort_point, &SpriteSortPoint::ALL, SpriteSortPoint::label, &mut changed);
        layer_row(ui, "Sorting Layer", "sr", &mut self.sorting_layer, &mut changed);
        order_row(ui, "Order in Layer", &mut self.order_in_layer, &mut changed);
        widgets::property_row(ui, "Material", |ui| {
            changed |= ui.add(egui::TextEdit::singleline(&mut self.material).hint_text("default sprite material").desired_width(ui.available_width())).changed();
        });
        changed
    }
}
impl Gizmo for SpriteRenderer {}

impl Inspect for SortingGroup {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        layer_row(ui, "Sorting Layer", "sg", &mut self.sorting_layer, &mut changed);
        order_row(ui, "Order in Layer", &mut self.order_in_layer, &mut changed);
        widgets::property_row(ui, "Sort At Root", |ui| changed |= ui.checkbox(&mut self.sort_at_root, "").on_hover_text("Stay a root even when nested in another Sorting Group").changed());
        ui.label(RichText::new("The sprites below this object sort among themselves, and the group sorts as one.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        changed
    }
}
impl Gizmo for SortingGroup {}

impl Inspect for SpriteMask {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        sprite_slot(ui, "Sprite", &mut self.sprite, &mut changed);
        widgets::property_row(ui, "Alpha Cutoff", |ui| changed |= ui.add(DragValue::new(&mut self.alpha_cutoff).speed(0.01).range(0.0..=1.0)).changed());
        let mut custom = matches!(self.range, SpriteMaskRange::Custom { .. });
        widgets::property_row(ui, "Custom Range", |ui| {
            if ui.checkbox(&mut custom, "").changed() {
                self.range = if custom {
                    SpriteMaskRange::Custom { back_layer: self.sorting_layer, back_order: self.order_in_layer, front_layer: self.sorting_layer, front_order: self.order_in_layer.saturating_add(10) }
                } else {
                    SpriteMaskRange::Auto
                };
                changed = true;
            }
        });
        if let SpriteMaskRange::Custom { mut back_layer, mut back_order, mut front_layer, mut front_order } = self.range {
            let mut c = false;
            layer_row(ui, "Front Layer", "mf", &mut front_layer, &mut c);
            order_row(ui, "Front Order", &mut front_order, &mut c);
            layer_row(ui, "Back Layer", "mb", &mut back_layer, &mut c);
            order_row(ui, "Back Order", &mut back_order, &mut c);
            if c {
                self.range = SpriteMaskRange::Custom { back_layer, back_order, front_layer, front_order };
                changed = true;
            }
        }
        layer_row(ui, "Sorting Layer", "ms", &mut self.sorting_layer, &mut changed);
        order_row(ui, "Order in Layer", &mut self.order_in_layer, &mut changed);
        ui.label(RichText::new("Needs a stencil-capable depth format (off by default); see the 2D docs.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        changed
    }
}
impl Gizmo for SpriteMask {}

/// The Sorting Layers editor: a reorderable list (back to front) with rename,
/// lock and delete; Default is fixed. Returns true when anything changed.
pub fn sorting_layers_ui(ui: &mut Ui, layers: &mut SortingLayers) -> bool {
    let mut changed = false;
    ui.label(RichText::new("Back to front. A sprite's layer decides its draw order before its Order in Layer.").weak().small());
    let n = layers.layers.len();
    let mut mv: Option<(usize, usize)> = None;
    let mut del: Option<usize> = None;
    let default_at = layers.default_index();
    egui::Grid::new("sorting-layers").striped(true).show(ui, |ui| {
        let head = |ui: &mut Ui, t: &str| ui.label(RichText::new(t).size(theme::SIZE_TINY).color(theme::TEXT_DIM));
        head(ui, "");
        head(ui, "Name");
        head(ui, "Lock");
        head(ui, "Behind 3D");
        ui.end_row();
        for i in 0..n {
            let is_default = i == default_at;
            let value = i as i32 - default_at as i32;
            ui.label(RichText::new(format!("{value:+}")).weak().small());
            let l = &mut layers.layers[i];
            ui.add_enabled_ui(!is_default, |ui| {
                changed |= ui.add_sized([96.0, theme::H_FIELD], egui::TextEdit::singleline(&mut l.name)).changed();
            });
            changed |= ui.checkbox(&mut l.locked, "").on_hover_text("Locked: sprites on this layer cannot be picked in the viewport").changed();
            changed |= ui
                .checkbox(&mut l.behind_3d, "")
                .on_hover_text("Behind 3D transparents: this layer and every layer below it draw before the transparent 3D queue")
                .changed();
            if ui.add_enabled(i > 0, egui::Button::new(egui_phosphor::regular::ARROW_UP)).clicked() {
                mv = Some((i, i - 1));
            }
            if ui.add_enabled(i + 1 < n, egui::Button::new(egui_phosphor::regular::ARROW_DOWN)).clicked() {
                mv = Some((i, i + 1));
            }
            if ui.add_enabled(!is_default, egui::Button::new(egui_phosphor::regular::TRASH)).clicked() {
                del = Some(i);
            }
            ui.end_row();
        }
    });
    if let Some((a, b)) = mv {
        layers.move_layer(a, b);
        changed = true;
    }
    if let Some(i) = del {
        let id = layers.layers[i].id;
        layers.remove(id);
        changed = true;
    }
    if ui.button(format!("{} Add sorting layer", egui_phosphor::regular::PLUS)).clicked() {
        let name = format!("Layer {}", layers.layers.len());
        layers.add(&name);
        changed = true;
    }
    changed
}
