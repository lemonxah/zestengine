//! In-game UI: lays out the scene's UI components (`UiCanvas` trees), handles
//! pointer / keyboard / gamepad interaction, and paints through egui.
//!
//! Screen canvases are painted over a target rect: the standalone runtime feeds
//! its own egui context sized to the window, the editor feeds its context with
//! the viewport rect. World canvases each get their own egui context and render
//! target (see [`WorldUi`]); the target is drawn as an unlit quad in the scene
//! and pointed at with rays (mouse or VR controllers).
//!
//! Interaction is done here (not with egui widgets) so it never competes with
//! the editor's own widgets under the viewport.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use zestengine_core::{
    ObjectId, UiButton, UiButtonState, UiCanvas, UiFillDirection, UiImage, UiPanel, UiProgressBar, UiRect,
    UiRenderMode, UiScaleMode, UiSlider, UiText, UiTextField, UiTextOverflow, UiTextRenderer, UiToggle,
};
use zestengine_text::{
    Billboard, Effects, HAlign, LayoutParams, Overflow, TextDraw, TextHandle, TextRequest, TextSpace, VAlign,
    gpu::srgb_to_linear,
};
use egui::{Color32, CornerRadius, FontFamily, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};
use glam::{Mat4, Vec3};
use zestengine_core::ecs::{Enabled, Entity, Parent, SystemWorld};

use crate::scene::LoadedScene;

/// Pointer state for one UI frame, in egui points.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiPointer {
    pub pos: Option<Pos2>,
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
}

/// Keyboard / gamepad navigation edges for one UI frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiNav {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub submit: bool,
    pub cancel: bool,
}

impl UiNav {
    fn any_dir(&self) -> bool {
        self.up || self.down || self.left || self.right
    }
}

/// Text typed since the last UI frame (for focused text fields).
#[derive(Clone, Debug, Default)]
pub struct UiKeys {
    pub text: String,
    pub backspaces: u32,
    pub enter: bool,
    pub escape: bool,
}

/// Everything the UI reads in one frame.
#[derive(Clone, Debug, Default)]
pub struct UiInput {
    pub pointer: UiPointer,
    pub nav: UiNav,
    pub keys: UiKeys,
}

/// What the UI reports back to gameplay.
#[derive(Clone, Debug, Default)]
pub struct UiFrame {
    pub pointer_over_ui: bool,
    pub clicks: Vec<ObjectId>,
    pub submits: Vec<ObjectId>,
    pub text_editing: bool,
    pub cancel: bool,
}

impl UiFrame {
    pub fn merge(&mut self, other: UiFrame) {
        self.pointer_over_ui |= other.pointer_over_ui;
        self.clicks.extend(other.clicks);
        self.submits.extend(other.submits);
        self.text_editing |= other.text_editing;
        self.cancel |= other.cancel;
    }
}

/// Which canvases a [`GameUi::run`] covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiScope {
    /// Every screen-overlay canvas.
    Screen,
    /// One world-space canvas (object index).
    Canvas(usize),
}

#[derive(Default)]
pub struct GameUi {
    textures: HashMap<String, Option<egui::TextureHandle>>,
    fonts: HashSet<String>,
    /// Element the pointer went down on (click = release over the same one).
    active: Option<ObjectId>,
    /// Keyboard / gamepad focus.
    focused: Option<ObjectId>,
    /// Text field being edited.
    editing: Option<ObjectId>,
    /// Physical pixels per point to lay out with instead of the context's
    /// (the editor previews a target resolution in a smaller rect).
    pub virtual_ppp: Option<f32>,
    /// Layer to paint on (default: a Middle-order layer of its own).
    pub layer: Option<egui::LayerId>,
    /// Lay this object out even if it (or a parent) is disabled (editor).
    pub reveal: Option<usize>,
    /// GPU text (SDF/MTSDF). `None` paints `UiText` through egui instead
    /// (editor previews, tests, or when the GPU text path is unavailable).
    pub text: Option<TextHandle>,
    /// String tables for `UiText::loc_key`.
    pub strings: Option<std::sync::Arc<zestengine_core::Localization>>,
    /// World canvases: canvas-target pixels -> world, so text can draw
    /// straight into the scene pass instead of through the canvas target.
    pub world_xform: Option<Mat4>,
    /// Set by `run`: something other than `UiText` was painted through egui
    /// (a world canvas needs its render target only then).
    pub egui_content: bool,
    /// Draw order of GPU text within one `run`.
    text_seq: i32,
}

/// One laid-out element, in draw order.
struct Node {
    index: usize,
    entity: Entity,
    id: ObjectId,
    rect: Rect,
    /// Reference pixels -> egui points.
    scale: f32,
}

fn color(c: [f32; 4]) -> Color32 {
    let u = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgba_unmultiplied(u(c[0]), u(c[1]), u(c[2]), u(c[3]))
}

#[cfg(test)]
fn find<T: zestengine_core::Component>(scene: &LoadedScene, i: usize) -> Option<zestengine_core::ecs::ComponentRef<'_, T>> {
    scene.objects[i].components.get_typed::<T>()
}

fn interactive(world: &SystemWorld<'_>, entity: Entity) -> bool {
    world.get::<UiButton>(entity).is_some_and(|b| b.interactable)
        || world.get::<UiToggle>(entity).is_some_and(|t| t.interactable)
        || world.get::<UiSlider>(entity).is_some_and(|s| s.interactable)
        || world.get::<UiTextField>(entity).is_some_and(|t| t.interactable)
}

fn blocks_pointer(world: &SystemWorld<'_>, entity: Entity) -> bool {
    world.get::<UiPanel>(entity).is_some_and(|x| x.color[3] > 0.0) || world.get::<UiImage>(entity).is_some_and(|x| x.tint[3] > 0.0)
}

impl GameUi {
    /// Lay out, interact with (when `input` is Some) and paint the UI in
    /// `scope` over `view` (egui points). `selected` gets an outline (editor).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        ctx: &egui::Context,
        view: Rect,
        scene: &mut LoadedScene,
        assets_root: &Path,
        input: Option<&UiInput>,
        selected: Option<usize>,
        scope: UiScope,
    ) -> UiFrame {
        scene.objects.query(|world, order| self.run_in(ctx, view, scene, world, order, assets_root, input, selected, scope))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_in(&mut self, ctx: &egui::Context, view: Rect, scene: &LoadedScene, world: &SystemWorld<'_>, order: &HashMap<ObjectId, usize>, assets_root: &Path, input: Option<&UiInput>, selected: Option<usize>, scope: UiScope) -> UiFrame {
        let mut frame = UiFrame::default();
        if !view.is_finite() || view.width() < 1.0 || view.height() < 1.0 {
            return frame;
        }
        self.egui_content = false;
        self.text_seq = 0;
        if let (Some(h), Some(strings)) = (&self.text, &self.strings)
            && let Ok(mut sys) = h.system.lock()
        {
            sys.set_language(strings.current_language());
            crate::text3d::warm_language(&mut sys, strings);
        }
        let nodes = layout_in(scene, world, order, view, self.virtual_ppp.unwrap_or(ctx.pixels_per_point()), scope, self.reveal);
        // Saved IDs survive row moves, but interaction also depends on the
        // current enabled widget, not just the continued existence of its row.
        for slot in [&mut self.active, &mut self.focused] {
            if slot.is_some_and(|id| !nodes.iter().any(|n| n.id == id && interactive(world, n.entity))) {
                *slot = None;
            }
        }
        if let Some(id) = self.editing {
            let valid = nodes.iter().any(|n| n.id == id && world.get::<UiTextField>(n.entity).is_some_and(|field| field.interactable));
            if !valid {
                self.editing = None;
                if let Some(entity) = world.entity(id) && let Some(mut field) = world.get_mut::<UiTextField>(entity) { field.editing = false; }
            }
        }
        if nodes.is_empty() {
            return frame;
        }

        match input {
            Some(input) => {
                let was_editing = self.editing.is_some();
                self.interact(world, &nodes, &input.pointer, &mut frame);
                self.navigate(world, &nodes, &input.nav, &mut frame);
                self.edit_text(world, &nodes, &input.keys, &mut frame);
                frame.cancel |= input.nav.cancel && !was_editing;
            }
            None => {
                self.active = None;
                self.editing = None;
            }
        }
        for n in &nodes {
            let id = n.id;
            let editing = self.editing == Some(id);
            if let Some(mut f) = world.get_mut::<UiTextField>(n.entity) {
                f.editing = editing;
            }
        }
        frame.text_editing = self.editing.is_some();

        let layer = self
            .layer
            .unwrap_or(egui::LayerId::new(egui::Order::Middle, egui::Id::new("zestengine-game-ui")));
        let painter = ctx.layer_painter(layer).with_clip_rect(view);
        let blink = (ctx.input(|i| i.time) * 2.0) as i64 % 2 == 0;
        for n in &nodes {
            let id = n.id;
            self.paint(ctx, &painter, world, assets_root, n, blink);
            let outline = if selected == Some(n.index) {
                Some(Color32::from_rgb(255, 170, 40))
            } else if self.focused == Some(id) && input.is_some() {
                Some(Color32::from_rgb(255, 230, 120))
            } else {
                None
            };
            if let Some(c) = outline {
                self.egui_content = true;
                painter.rect_stroke(n.rect, CornerRadius::ZERO, Stroke::new(2.0_f32, c), StrokeKind::Outside);
            }
        }
        frame
    }

    fn interact(&mut self, world: &SystemWorld<'_>, nodes: &[Node], p: &UiPointer, frame: &mut UiFrame) {
        // Topmost interactive element under the pointer (last in draw order).
        let mut hover = None;
        if let Some(pos) = p.pos {
            for n in nodes {
                if !n.rect.contains(pos) {
                    continue;
                }
                if interactive(world, n.entity) {
                    hover = Some(n);
                    frame.pointer_over_ui = true;
                } else if blocks_pointer(world, n.entity) {
                    frame.pointer_over_ui = true;
                }
            }
        }
        let hover_id = hover.map(|n| n.id);
        if p.pressed {
            self.active = hover_id;
            self.focused = hover_id;
            // Clicking anywhere else ends text editing.
            if self.editing != hover_id {
                self.editing = None;
            }
        }

        if let Some(active) = self.active
            && let Some(n) = nodes.iter().find(|n| n.id == active)
        {
            if p.down
                && let (Some(pos), Some(mut slider)) = (p.pos, world.get_mut::<UiSlider>(n.entity))
                && slider.interactable
            {
                slider.set_fraction((pos.x - n.rect.left()) / n.rect.width().max(1.0));
            }
            if p.released {
                if hover_id == Some(active) {
                    self.activate(world, n, frame);
                }
                self.active = None;
            }
        } else if p.released || !p.down {
            self.active = None;
        }

        for n in nodes {
            let id = n.id;
            let state = if hover_id == Some(id) && self.active == Some(id) && p.down {
                UiButtonState::Pressed
            } else if (hover_id == Some(id) && (self.active.is_none() || self.active == Some(id)))
                || (hover_id.is_none() && self.focused == Some(id))
            {
                UiButtonState::Hovered
            } else {
                UiButtonState::Normal
            };
            if let Some(mut b) = world.get_mut::<UiButton>(n.entity) {
                b.state = state;
            }
        }
    }

    /// Click / submit on an element: buttons report a click, toggles flip,
    /// text fields start editing.
    fn activate(&mut self, world: &SystemWorld<'_>, node: &Node, frame: &mut UiFrame) {
        let id = node.id;
        if world.get::<UiTextField>(node.entity).is_some_and(|t| t.interactable) {
            self.editing = Some(id);
            return;
        }
        frame.clicks.push(id);
        if let Some(mut t) = world.get_mut::<UiToggle>(node.entity) {
            t.on = !t.on;
        }
    }

    /// Spatial focus navigation (arrows / D-pad / stick), submit and slider steps.
    fn navigate(&mut self, world: &SystemWorld<'_>, nodes: &[Node], nav: &UiNav, frame: &mut UiFrame) {
        if self.editing.is_some() {
            return;
        }
        let candidates: Vec<&Node> = nodes.iter().filter(|n| interactive(world, n.entity)).collect();
        if candidates.is_empty() {
            return;
        }
        let focused = self
            .focused
            .and_then(|id| candidates.iter().find(|n| n.id == id).copied());
        let Some(cur) = focused else {
            if nav.any_dir() || nav.submit {
                self.focused = Some(candidates[0].id);
            }
            return;
        };
        if nav.submit {
            self.activate(world, cur, frame);
            return;
        }
        // Sliders take left / right as value steps.
        if (nav.left || nav.right)
            && let Some(mut s) = world.get_mut::<UiSlider>(cur.entity)
            && s.interactable
        {
            let step = if s.whole_numbers { 1.0 } else { (s.max - s.min) / 10.0 };
            let v = s.value + if nav.right { step } else { -step };
            s.value = v.clamp(s.min.min(s.max), s.max.max(s.min));
            return;
        }
        let dir = match (nav.up, nav.down, nav.left, nav.right) {
            (true, _, _, _) => Vec2::new(0.0, -1.0),
            (_, true, _, _) => Vec2::new(0.0, 1.0),
            (_, _, true, _) => Vec2::new(-1.0, 0.0),
            (_, _, _, true) => Vec2::new(1.0, 0.0),
            _ => return,
        };
        let from = cur.rect.center();
        let best = candidates
            .iter()
            .filter(|n| n.index != cur.index)
            .filter_map(|n| {
                let d = n.rect.center() - from;
                let along = d.dot(dir);
                if along <= 1.0 {
                    return None;
                }
                let perp = (d - dir * along).length();
                Some((along + perp * 2.0, *n))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, n)) = best {
            self.focused = Some(n.id);
        }
    }

    fn edit_text(&mut self, world: &SystemWorld<'_>, nodes: &[Node], keys: &UiKeys, frame: &mut UiFrame) {
        let Some(id) = self.editing else {
            return;
        };
        let Some(n) = nodes.iter().find(|n| n.id == id) else {
            self.editing = None;
            return;
        };
        if let Some(mut field) = world.get_mut::<UiTextField>(n.entity) {
            field.type_text(&keys.text, keys.backspaces);
        }
        if keys.enter {
            frame.submits.push(id);
            self.editing = None;
        } else if keys.escape {
            self.editing = None;
        }
    }

    fn paint(&mut self, ctx: &egui::Context, painter: &egui::Painter, world: &SystemWorld<'_>, root: &Path, n: &Node, blink: bool) {
        let rect = n.rect;
        let s = n.scale;
        let panel = world.get::<UiPanel>(n.entity);
        let button = world.get::<UiButton>(n.entity);
        let field = world.get::<UiTextField>(n.entity);
        self.egui_content |= panel.is_some()
            || button.is_some()
            || field.is_some()
            || world.get::<UiImage>(n.entity).is_some_and(|i| !i.texture.is_empty())
            || world.get::<UiToggle>(n.entity).is_some()
            || world.get::<UiProgressBar>(n.entity).is_some()
            || world.get::<UiSlider>(n.entity).is_some();
        let radius = |r: f32| CornerRadius::same((r * s).round().clamp(0.0, 255.0) as u8);
        match (panel, button) {
            (Some(p), b) => {
                let fill = b.map(|b| b.color()).unwrap_or(p.color);
                let stroke = if p.border_width > 0.0 {
                    Stroke::new(p.border_width * s, color(p.border_color))
                } else {
                    Stroke::NONE
                };
                painter.rect(rect, radius(p.corner_radius), color(fill), stroke, StrokeKind::Inside);
            }
            (None, Some(b)) => {
                painter.rect_filled(rect, radius(6.0), color(b.color()));
            }
            (None, None) if field.is_some() => {
                painter.rect(
                    rect,
                    radius(4.0),
                    color([0.08, 0.08, 0.1, 0.9]),
                    Stroke::new(1.0_f32, color([1.0, 1.0, 1.0, 0.25])),
                    StrokeKind::Inside,
                );
            }
            (None, None) => {}
        }

        if let Some(img) = world.get::<UiImage>(n.entity)
            && !img.texture.is_empty()
            && let Some(tex) = self.texture(ctx, root, &img.texture)
        {
            let mut target = rect;
            if img.preserve_aspect {
                let [w, h] = tex.size();
                let aspect = w as f32 / (h as f32).max(1.0);
                let fit = if rect.width() / rect.height().max(1e-3) > aspect {
                    Vec2::new(rect.height() * aspect, rect.height())
                } else {
                    Vec2::new(rect.width(), rect.width() / aspect)
                };
                target = Rect::from_center_size(rect.center(), fit);
            }
            painter.image(tex.id(), target, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), color(img.tint));
        }

        if let Some(t) = world.get::<UiToggle>(n.entity) {
            let side = rect.height().min(rect.width());
            let bx = Rect::from_min_size(rect.min, Vec2::splat(side));
            painter.rect_filled(bx, radius(4.0), color(t.box_color));
            if t.on {
                painter.rect_filled(bx.shrink(side * 0.22), radius(2.0), color(t.check_color));
            }
        }

        if let Some(bar) = world.get::<UiProgressBar>(n.entity) {
            let r = radius(bar.corner_radius);
            painter.rect_filled(rect, r, color(bar.background));
            let v = bar.value.clamp(0.0, 1.0);
            let fill = match bar.direction {
                UiFillDirection::LeftToRight => Rect::from_min_size(rect.min, Vec2::new(rect.width() * v, rect.height())),
                UiFillDirection::RightToLeft => {
                    Rect::from_min_max(Pos2::new(rect.right() - rect.width() * v, rect.top()), rect.max)
                }
                UiFillDirection::BottomToTop => {
                    Rect::from_min_max(Pos2::new(rect.left(), rect.bottom() - rect.height() * v), rect.max)
                }
                UiFillDirection::TopToBottom => Rect::from_min_size(rect.min, Vec2::new(rect.width(), rect.height() * v)),
            };
            if v > 0.0 {
                painter.rect_filled(fill, r, color(bar.fill));
            }
        }

        if let Some(sl) = world.get::<UiSlider>(n.entity) {
            let h = rect.height();
            let track = Rect::from_center_size(rect.center(), Vec2::new(rect.width(), (h * 0.3).max(2.0)));
            let r = CornerRadius::same((track.height() * 0.5).min(255.0) as u8);
            painter.rect_filled(track, r, color(sl.track_color));
            let x = track.left() + track.width() * sl.fraction();
            painter.rect_filled(Rect::from_min_max(track.min, Pos2::new(x, track.max.y)), r, color(sl.fill_color));
            painter.circle_filled(Pos2::new(x, rect.center().y), h * 0.45, color(sl.handle_color));
        }

        if let Some(f) = field {
            let font = FontId::new((f.size * s).max(1.0), FontFamily::Proportional);
            let (shown, col) = if f.text.is_empty() && !f.editing {
                (f.placeholder.clone(), f.placeholder_color)
            } else {
                (f.display_text(), f.text_color)
            };
            let pad = 8.0 * s;
            let galley = ctx.fonts_mut(|fonts| fonts.layout_no_wrap(shown, font, color(col)));
            let pos = Pos2::new(rect.left() + pad, rect.center().y - galley.rect.height() * 0.5);
            let caret_x = if f.text.is_empty() { pos.x } else { pos.x + galley.rect.width() };
            let gh = galley.rect.height();
            painter.with_clip_rect(rect.shrink(2.0)).galley(pos, galley, color(col));
            if f.editing && blink {
                painter.line_segment(
                    [Pos2::new(caret_x + 1.0, pos.y), Pos2::new(caret_x + 1.0, pos.y + gh)],
                    Stroke::new((1.5 * s).max(1.0), color(f.text_color)),
                );
            }
        }

        if let Some(text) = world.get::<UiText>(n.entity) {
            let strings = self.strings.clone();
            let shown: &str = match &strings {
                Some(s) if !text.loc_key.is_empty() && s.entries.contains_key(&text.loc_key) => s.tr(&text.loc_key),
                _ => &text.text,
            };
            if !shown.is_empty() && !self.paint_text_gpu(ctx, &text, shown, n) {
                self.egui_content = true;
                let family = self.font_family(ctx, root, &text.font);
                let font = FontId::new((text.size * s).max(1.0), family);
                let wrap = if text.wrap { rect.width().max(1.0) } else { f32::INFINITY };
                let mut job = egui::text::LayoutJob::simple(shown.to_owned(), font, color(text.color), wrap);
                job.halign = match text.align_x {
                    zestengine_core::UiAlign::Start => egui::Align::LEFT,
                    zestengine_core::UiAlign::Center => egui::Align::Center,
                    zestengine_core::UiAlign::End => egui::Align::RIGHT,
                };
                let galley = ctx.fonts_mut(|f| f.layout_job(job));
                let x = rect.left() + rect.width() * text.align_x.factor();
                let y = rect.top() + (rect.height() - galley.rect.height()) * text.align_y.factor();
                painter.galley(Pos2::new(x, y), galley, color(text.color));
            }
        }
    }

    /// Lay `text` out on the GPU text path and queue its draw. False when the
    /// element uses the legacy renderer or no GPU text / usable font exists
    /// (the caller paints through egui instead).
    fn paint_text_gpu(&mut self, ctx: &egui::Context, text: &UiText, shown: &str, n: &Node) -> bool {
        if text.renderer == UiTextRenderer::Legacy {
            return false;
        }
        let Some(handle) = self.text.clone() else { return false };
        let s = n.scale.max(1e-4);
        let rect = n.rect;
        let align_h = match text.align_x {
            _ if text.justify && text.wrap => HAlign::Justified,
            zestengine_core::UiAlign::Start => HAlign::Start,
            zestengine_core::UiAlign::Center => HAlign::Center,
            zestengine_core::UiAlign::End => HAlign::End,
        };
        let params = LayoutParams {
            size: text.size,
            box_size: [rect.width() / s, rect.height() / s],
            wrap: text.wrap,
            halign: align_h,
            valign: match text.align_y {
                zestengine_core::UiAlign::Start => VAlign::Top,
                zestengine_core::UiAlign::Center => VAlign::Middle,
                zestengine_core::UiAlign::End => VAlign::Bottom,
            },
            overflow: match text.overflow {
                UiTextOverflow::Overflow => Overflow::Overflow,
                UiTextOverflow::Ellipsis => Overflow::Ellipsis,
                UiTextOverflow::Truncate => Overflow::Truncate,
                UiTextOverflow::Masking => Overflow::Masking,
                UiTextOverflow::Page => Overflow::Page,
            },
            auto_size: text.auto_size.then_some((text.auto_size_min, text.auto_size_max)),
            char_spacing: text.char_spacing,
            word_spacing: text.word_spacing,
            line_spacing: text.line_spacing,
            paragraph_spacing: text.paragraph_spacing,
            margins: text.margins,
            bold: text.bold,
            italic: text.italic,
            page: 0,
        };
        let oc = text.outline_color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        let req = TextRequest {
            text: shown,
            font: &text.font,
            rich: text.rich_text,
            params,
            effects: Effects { outline_em: text.outline_width / 100.0, outline_color: oc, ..Default::default() },
            rtl: None,
        };
        let Some(block) = handle.system.lock().ok().and_then(|mut sys| sys.block(&req)) else { return false };
        let local = Mat4::from_translation(Vec3::new(rect.min.x, rect.min.y, 0.0)) * Mat4::from_scale(Vec3::new(s, s, 1.0));
        let (transform, space) = match self.world_xform {
            Some(world) => (world * local, TextSpace::World),
            None => {
                let ppp = ctx.pixels_per_point();
                (Mat4::from_scale(Vec3::new(ppp, ppp, 1.0)) * local, TextSpace::Screen)
            }
        };
        self.text_seq += 1;
        handle.push(TextDraw {
            instances: block.instances.clone(),
            transform: transform.to_cols_array_2d(),
            space,
            billboard: Billboard::None,
            tint: srgb_to_linear(text.color),
            local_offset: [0.0; 2],
            clip: block.clip,
            supersample: space == TextSpace::World,
            order: self.text_seq,
        });
        true
    }

    fn texture(&mut self, ctx: &egui::Context, root: &Path, path: &str) -> Option<egui::TextureHandle> {
        self.textures
            .entry(path.to_owned())
            .or_insert_with(|| match zestengine_assets::load_texture_file(zestengine_core::asset_ref::resolve_path(root, path), true) {
                Ok(t) if !t.hdr => {
                    let img = egui::ColorImage::from_rgba_unmultiplied([t.width as usize, t.height as usize], &t.pixels);
                    Some(ctx.load_texture(format!("ui:{path}"), img, egui::TextureOptions::LINEAR))
                }
                Ok(_) => {
                    tracing::warn!("UI image {path}: HDR images aren't supported in UI");
                    None
                }
                Err(e) => {
                    tracing::warn!("UI image {path}: {e:#}");
                    None
                }
            })
            .clone()
    }

    /// Font family for a UiText font path; registers the font on first use
    /// and falls back to the built-in font until egui has loaded it.
    fn font_family(&mut self, ctx: &egui::Context, root: &Path, path: &str) -> FontFamily {
        if path.is_empty() {
            return FontFamily::Proportional;
        }
        let family = FontFamily::Name(path.into());
        if self.fonts.insert(path.to_owned()) {
            match std::fs::read(zestengine_core::asset_ref::resolve_path(root, path)) {
                Ok(bytes) => ctx.add_font(egui::epaint::text::FontInsert::new(
                    path,
                    egui::FontData::from_owned(bytes),
                    vec![egui::epaint::text::InsertFontFamily {
                        family: family.clone(),
                        priority: egui::epaint::text::FontPriority::Highest,
                    }],
                )),
                Err(e) => tracing::warn!("UI font {path}: {e}"),
            }
        }
        let ready = ctx.fonts(|f| f.definitions().families.contains_key(&family));
        if ready { family } else { FontFamily::Proportional }
    }
}

/// Laid-out rects of the UI in `scope` (object index, rect in points,
/// reference-pixel-to-point scale), in draw order. Used by the editor's
/// rect handles.
pub fn layout_rects(
    scene: &LoadedScene,
    view: Rect,
    pixels_per_point: f32,
    scope: UiScope,
    reveal: Option<usize>,
) -> Vec<(usize, Rect, f32)> {
    layout(scene, view, pixels_per_point, scope, reveal)
        .into_iter()
        .map(|n| (n.index, n.rect, n.scale))
        .collect()
}

/// UI elements in `scope`, in draw order (canvases by `sort_order`, then
/// depth-first in hierarchy order).
fn layout(scene: &LoadedScene, view: Rect, pixels_per_point: f32, scope: UiScope, reveal: Option<usize>) -> Vec<Node> {
    scene.objects.query(|world, order| layout_in(scene, world, order, view, pixels_per_point, scope, reveal))
}

/// Memoized inherited activation. Reveal affects the selected object's ancestor
/// chain only; unrelated disabled widgets remain inactive in editor previews.
fn layout_active(world: &SystemWorld<'_>, entity: Entity, revealed: &HashSet<Entity>, state: &mut HashMap<Entity, bool>, limit: usize) -> bool {
    let mut chain = Vec::new();
    let mut at = Some(entity);
    let active = loop {
        let Some(current) = at else { break true };
        if let Some(&active) = state.get(&current) { break active; }
        if chain.len() > limit || chain.contains(&current) { break false; }
        chain.push(current);
        if !world.get::<Enabled>(current).is_some_and(|enabled| enabled.0) && !revealed.contains(&current) { break false; }
        at = world.get::<Parent>(current).and_then(|parent| parent.0);
    };
    for current in chain { state.insert(current, active); }
    active
}

#[allow(clippy::too_many_arguments)]
fn layout_in(scene: &LoadedScene, world: &SystemWorld<'_>, order: &HashMap<ObjectId, usize>, view: Rect, pixels_per_point: f32, scope: UiScope, reveal: Option<usize>) -> Vec<Node> {
    let mut revealed = HashSet::new();
    let mut at = reveal.and_then(|index| scene.objects.get(index)).map(|object| object.entity());
    while let Some(entity) = at {
        if !revealed.insert(entity) || revealed.len() > order.len() { break; }
        at = world.get::<Parent>(entity).and_then(|parent| parent.0);
    }
    let mut state = HashMap::new();
    let mut children: HashMap<Entity, Vec<(usize, Entity, ObjectId)>> = HashMap::new();
    for (entity, (id, parent)) in world.query::<(&ObjectId, &Parent)>().iter() {
        if let Some(parent) = parent.0 && let Some(&index) = order.get(id) {
            children.entry(parent).or_default().push((index, entity, *id));
        }
    }
    for children in children.values_mut() { children.sort_unstable_by_key(|(index, _, _)| *index); }
    let mut canvases: Vec<_> = world.query::<(&ObjectId, &UiCanvas)>().iter().filter_map(|(entity, (id, canvas))| {
        let &index = order.get(id)?;
        let in_scope = match scope { UiScope::Screen => canvas.render_mode == UiRenderMode::Screen, UiScope::Canvas(target) => index == target };
        (in_scope && layout_active(world, entity, &revealed, &mut state, order.len())).then_some((canvas.sort_order, index, entity, *id))
    }).collect();
    canvases.sort_unstable_by_key(|(sort, index, _, _)| (*sort, *index));
    let mut out = Vec::new();
    let ppp = pixels_per_point.max(1e-3);
    for (_, canvas_index, canvas_entity, canvas_id) in canvases {
        let canvas = world.get::<UiCanvas>(canvas_entity).expect("selected canvas");
        let px = view.size() * ppp;
        let scale_px = match canvas.scale_mode {
            UiScaleMode::ConstantPixelSize => 1.0,
            UiScaleMode::ScaleWithScreen => {
                let sx = px.x / canvas.reference_size[0].max(1.0);
                let sy = px.y / canvas.reference_size[1].max(1.0);
                let mix = canvas.match_height.clamp(0.0, 1.0);
                sx.powf(1.0 - mix) * sy.powf(mix)
            }
        };
        let scale = scale_px / ppp;
        let mut stack = vec![(canvas_index, canvas_entity, canvas_id, view)];
        while let Some((index, entity, id, parent_rect)) = stack.pop() {
            let rect = if entity == canvas_entity { view } else {
                match world.get::<UiRect>(entity) {
                    Some(layout) => {
                        let (min, size) = layout.resolve([parent_rect.min.x, parent_rect.min.y], [parent_rect.width(), parent_rect.height()], scale);
                        Rect::from_min_size(Pos2::new(min[0], min[1]), Vec2::new(size[0], size[1]))
                    }
                    None => parent_rect,
                }
            };
            out.push(Node { index, entity, id, rect, scale });
            if let Some(children) = children.get(&entity) {
                for &(index, entity, id) in children.iter().rev() {
                    if layout_active(world, entity, &revealed, &mut state, order.len()) && world.get::<UiCanvas>(entity).is_none() {
                        stack.push((index, entity, id, rect));
                    }
                }
            }
        }
    }
    out
}

/// Derived world-canvas draw parameters only. Widget values remain in ECS.
fn world_canvas_instances(scene: &LoadedScene) -> Vec<(usize, ObjectId, [f32; 2], [u32; 2], Mat4)> {
    scene.objects.query(|world, order| {
        let mut canvases: Vec<_> = world.query::<(&ObjectId, &UiCanvas)>().iter().filter_map(|(entity, (id, canvas))| {
            let &index = order.get(id)?;
            if canvas.render_mode != UiRenderMode::World || !scene.visible_entity_in(world, entity) { return None; }
            Some((index, *id, canvas.world_size, canvas.world_pixels(), world.world_transform(entity)?))
        }).collect();
        canvases.sort_unstable_by_key(|(index, ..)| *index);
        canvases
    })
}

// ------------------------------------------------------------ world canvases

/// A pointing ray for world canvases (mouse through the camera, or a VR
/// controller) with its button edges.
#[derive(Clone, Copy, Debug)]
pub struct UiRay {
    pub origin: Vec3,
    pub dir: Vec3,
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
}

/// The mouse as a pointing ray for world canvases on a flat screen (no
/// headset). `uv` is the cursor in the game view, 0..1 from the top-left. A
/// locked cursor (the VR Player's desktop mouse-look) has no position, so the
/// ray goes through the view centre: the crosshair is the pointer and the
/// primary button clicks (`button` = down / pressed / released edges). None
/// when the cursor is free but outside the view.
pub fn desktop_ui_ray(view: Mat4, proj: Mat4, uv: Option<glam::Vec2>, cursor_locked: bool, button: [bool; 3]) -> Option<UiRay> {
    let uv = if cursor_locked { glam::Vec2::splat(0.5) } else { uv? };
    let ndc = glam::Vec2::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    let inv = (proj * view).inverse();
    let near = inv.project_point3(ndc.extend(0.0));
    let far = inv.project_point3(ndc.extend(1.0));
    let dir = (far - near).try_normalize()?;
    let [down, pressed, released] = button;
    Some(UiRay { origin: near, dir, down, pressed, released })
}

/// Where a ray hits a world canvas: target pixel coordinates + distance.
pub fn ray_canvas_hit(ray: &UiRay, canvas_world: Mat4, size_m: [f32; 2], pixels: [u32; 2]) -> Option<(Pos2, f32)> {
    let inv = canvas_world.inverse();
    let o = inv.transform_point3(ray.origin);
    let d = inv.transform_vector3(ray.dir);
    if d.z.abs() < 1e-6 {
        return None;
    }
    let t = -o.z / d.z;
    if t <= 0.0 {
        return None;
    }
    let p = o + d * t;
    let (w, h) = (size_m[0].max(1e-4), size_m[1].max(1e-4));
    if p.x.abs() > w * 0.5 || p.y.abs() > h * 0.5 {
        return None;
    }
    let u = p.x / w + 0.5;
    let v = 0.5 - p.y / h;
    let world_hit = canvas_world.transform_point3(p);
    Some((
        Pos2::new(u * pixels[0] as f32, v * pixels[1] as f32),
        world_hit.distance(ray.origin),
    ))
}

struct WorldCanvas {
    ctx: egui::Context,
    ui: GameUi,
    /// Render target + material, created the first time something besides
    /// text needs egui (panels, images, fields). Text-only canvases draw
    /// straight into the scene pass and never get one.
    target: Option<(zestengine_render::TextureHandle, zestengine_render::MaterialHandle)>,
    pixels: [u32; 2],
    /// egui texture changes from frames that skipped the target.
    pending_textures: egui::TexturesDelta,
}

/// Render targets + egui contexts for the scene's world-space canvases.
#[derive(Default)]
pub struct WorldUi {
    canvases: HashMap<ObjectId, WorldCanvas>,
    quad: Option<zestengine_render::MeshHandle>,
    /// String tables for `UiText::loc_key` (set by the host each frame).
    pub strings: Option<std::sync::Arc<zestengine_core::Localization>>,
}

impl WorldUi {
    /// Lay out / interact with / paint every active world canvas. Returns the
    /// renderer draws for the targets plus the merged interaction result, and
    /// queues the canvas quads on `scene.extra_draws`. Text goes straight to
    /// the renderer's text queue; a canvas only gets a target and a quad when
    /// it also has non-text elements.
    pub fn update(
        &mut self,
        renderer: &mut zestengine_render::Renderer,
        scene: &mut LoadedScene,
        root: &Path,
        rays: &[UiRay],
        nav: Option<&UiInput>,
    ) -> (Vec<zestengine_render::WorldUiDraw>, UiFrame) {
        let mut draws = Vec::new();
        let mut frame = UiFrame::default();
        crate::text_setup::init(renderer, root);
        if renderer.text_available() {
            crate::text3d::queue_scene(&renderer.text_handle(), scene, self.strings.as_deref());
        }
        let list = world_canvas_instances(scene);
        self.canvases.retain(|id, _| list.iter().any(|(_, canvas, ..)| canvas == id));
        if list.is_empty() {
            return (draws, frame);
        }
        if self.quad.is_none() {
            match renderer.upload_mesh(&quad_mesh()) {
                Ok(m) => self.quad = Some(m),
                Err(e) => {
                    tracing::error!("world UI quad: {e:#}");
                    return (draws, frame);
                }
            }
        }
        let quad = self.quad.unwrap();

        // Each ray points at its nearest canvas.
        let mut ray_for: HashMap<usize, (UiRay, Pos2)> = HashMap::new();
        for ray in rays {
            let best = list
                .iter()
                .filter_map(|&(i, _, size, px, matrix)| {
                    ray_canvas_hit(ray, matrix, size, px).map(|(p, d)| (i, p, d))
                })
                .min_by(|a, b| a.2.total_cmp(&b.2));
            if let Some((i, p, _)) = best {
                ray_for.entry(i).or_insert((*ray, p));
            }
        }

        let text_handle = renderer.text_available().then(|| renderer.text_handle());
        for &(i, id, size, px, matrix) in &list {
            if self.canvases.get(&id).is_none_or(|c| c.pixels != px) {
                self.canvases.insert(
                    id,
                    WorldCanvas {
                        ctx: egui::Context::default(),
                        ui: GameUi::default(),
                        target: None,
                        pixels: px,
                        pending_textures: egui::TexturesDelta::default(),
                    },
                );
            }
            let canvas = self.canvases.get_mut(&id).unwrap();
            canvas.ui.text = text_handle.clone();
            canvas.ui.strings = self.strings.clone();
            // Canvas target pixels (y down from the top-left) -> world metres.
            canvas.ui.world_xform = Some(
                matrix
                    * Mat4::from_translation(Vec3::new(-size[0] * 0.5, size[1] * 0.5, 0.0))
                    * Mat4::from_scale(Vec3::new(size[0] / px[0] as f32, -size[1] / px[1] as f32, 1.0)),
            );
            let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(px[0] as f32, px[1] as f32));
            let input = ray_for.get(&i).map(|(ray, pos)| UiInput {
                pointer: UiPointer {
                    pos: Some(*pos),
                    down: ray.down,
                    pressed: ray.pressed,
                    released: ray.released,
                },
                nav: nav.map(|n| n.nav).unwrap_or_default(),
                keys: nav.map(|n| n.keys.clone()).unwrap_or_default(),
            });
            let raw = egui::RawInput {
                screen_rect: Some(view),
                ..Default::default()
            };
            let ui = &mut canvas.ui;
            let mut f = UiFrame::default();
            let mut out = canvas.ctx.run(raw, |ctx| {
                f = ui.run(ctx, view, scene, root, input.as_ref(), None, UiScope::Canvas(i));
            });
            frame.merge(f);
            canvas.pending_textures.append(std::mem::take(&mut out.textures_delta));
            if !canvas.ui.egui_content {
                continue;
            }
            if canvas.target.is_none() {
                let made = renderer
                    .create_ui_target(px[0], px[1])
                    .and_then(|t| unlit_material(renderer, Some(t), "world ui").map(|m| (t, m)));
                match made {
                    Ok(t) => canvas.target = Some(t),
                    Err(e) => {
                        tracing::error!("world canvas: {e:#}");
                        continue;
                    }
                }
            }
            let (target, material) = canvas.target.unwrap();
            let primitives = canvas.ctx.tessellate(out.shapes, out.pixels_per_point);
            draws.push(zestengine_render::WorldUiDraw {
                target,
                extent: px,
                draw: zestengine_render::EguiDraw {
                    pixels_per_point: out.pixels_per_point,
                    primitives,
                    textures_delta: std::mem::take(&mut canvas.pending_textures),
                },
            });
            let transform = matrix * Mat4::from_scale(Vec3::new(size[0], size[1], 1.0));
            scene.extra_draws.push(zestengine_render::DrawCmd {
                gi_source: None,
                mesh: quad,
                material,
                transform,
                highlight: 0.0,
                mesh_center: Vec3::ZERO,
                bound_radius: 0.75,
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index: -1,
                occluded: false,
                shadow_casting: zestengine_render::ShadowCasting::On,
                instance_color: None,
                instance_param: None,
            });
        }
        (draws, frame)
    }
}

pub use crate::render_util::unlit_material;

/// Unit quad in the local XY plane facing +Z, uv (0,0) at the top-left.
fn quad_mesh() -> zestengine_render::MeshData {
    let v = |x: f32, y: f32, u: f32, w: f32| zestengine_render::Vertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [u, w],
        color: [1.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv1: [u, w],
        joints: [0; 4],
        weights: [0.0; 4],
    };
    zestengine_render::MeshData {
        vertices: vec![
            v(-0.5, 0.5, 0.0, 0.0),
            v(0.5, 0.5, 1.0, 0.0),
            v(0.5, -0.5, 1.0, 1.0),
            v(-0.5, -0.5, 0.0, 1.0),
        ],
        indices: vec![0, 2, 1, 0, 3, 2],
        has_lightmap_uv: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    use zestengine_core::Component;
    use glam::Quat;

    fn obj(scene: &mut LoadedScene, parent: Option<usize>, components: Vec<Box<dyn Component>>) -> usize {
        scene.objects.push(SceneObject::from_components(zestengine_core::ecs::ComponentSet::from_vec(components)).with_parent(parent).with_component_roles(Default::default()).with_extra_render(Vec::new()).with_source(ObjectSource::Empty).with_net_role(Default::default()).with_render(None).with_static_geometry(false).with_layer(0).with_id(ObjectId::new()).with_name(String::new()).with_enabled(true).with_transform(zestengine_core::Transform { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }));
        scene.objects.len() - 1
    }

    fn rect_at(y: f32) -> UiRect {
        UiRect {
            position: [0.0, y],
            size: [200.0, 50.0],
            ..Default::default()
        }
    }

    fn menu_scene() -> (LoadedScene, Vec<usize>) {
        let mut scene = LoadedScene::empty();
        let canvas = obj(
            &mut scene,
            None,
            vec![Box::new(UiCanvas {
                reference_size: [960.0, 540.0],
                ..Default::default()
            })],
        );
        let mut items = Vec::new();
        for (k, y) in [-100.0, 0.0, 100.0].into_iter().enumerate() {
            let comps: Vec<Box<dyn Component>> = if k == 2 {
                vec![Box::new(rect_at(y)), Box::new(UiTextField::default())]
            } else {
                vec![Box::new(rect_at(y)), Box::new(UiButton::default()), Box::new(UiText::default())]
            };
            items.push(obj(&mut scene, Some(canvas), comps));
        }
        (scene, items)
    }

    fn run_frame(ui: &mut GameUi, ctx: &egui::Context, scene: &mut LoadedScene, input: UiInput) -> UiFrame {
        let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0));
        let mut frame = UiFrame::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            frame = ui.run(ctx, view, scene, Path::new("."), Some(&input), None, UiScope::Screen);
        });
        frame
    }

    fn pointer(pos: Pos2, down: bool, pressed: bool, released: bool) -> UiInput {
        UiInput {
            pointer: UiPointer { pos: Some(pos), down, pressed, released },
            ..Default::default()
        }
    }

    /// A text system with egui's built-in faces (what the host registers). The
    /// glyph budget is lifted: the default 4 ms/frame cap defers rasterization
    /// (and drops instances) when the parallel test run starves the thread,
    /// which made the instance-count assertions flaky.
    fn text_handle() -> TextHandle {
        let cfg = zestengine_text::TextConfig { glyph_budget: std::time::Duration::from_secs(60), ..Default::default() };
        let mut sys = zestengine_text::TextSystem::new(cfg);
        let defs = egui::FontDefinitions::default();
        for name in defs.families[&egui::FontFamily::Proportional].clone() {
            sys.add_builtin_font(&name, defs.font_data[&name].font.to_vec());
        }
        zestengine_text::TextShared::new(sys)
    }

    fn text_scene(text: UiText) -> LoadedScene {
        let mut scene = LoadedScene::empty();
        let canvas = obj(&mut scene, None, vec![Box::new(UiCanvas { reference_size: [960.0, 540.0], ..Default::default() })]);
        obj(&mut scene, Some(canvas), vec![Box::new(rect_at(0.0)), Box::new(text)]);
        scene
    }

    fn run_once(ui: &mut GameUi, scene: &mut LoadedScene, scope: UiScope) {
        let ctx = egui::Context::default();
        let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(960.0, 540.0));
        let raw = egui::RawInput { screen_rect: Some(view), ..Default::default() };
        let _ = ctx.run(raw, |ctx| {
            ui.run(ctx, view, scene, Path::new("."), None, None, scope);
        });
    }

    #[test]
    fn sdf_text_queues_a_screen_draw_instead_of_egui_text() {
        let handle = text_handle();
        let mut scene = text_scene(UiText { text: "Hello".into(), ..Default::default() });
        let mut ui = GameUi { text: Some(handle.clone()), ..Default::default() };
        run_once(&mut ui, &mut scene, UiScope::Screen);
        let draws = handle.draws.lock().unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].space, TextSpace::Screen);
        assert_eq!(draws[0].instances.len(), 5);
        assert!(!ui.egui_content, "text-only UI needs no egui content");
        // The box origin lands inside the 960x540 view (centred 200x50 rect).
        assert!(draws[0].transform[3][0] > 300.0 && draws[0].transform[3][0] < 560.0);
    }

    #[test]
    fn legacy_renderer_and_missing_text_system_use_egui() {
        let handle = text_handle();
        let mut scene = text_scene(UiText { renderer: UiTextRenderer::Legacy, ..Default::default() });
        let mut ui = GameUi { text: Some(handle.clone()), ..Default::default() };
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert!(handle.draws.lock().unwrap().is_empty());
        assert!(ui.egui_content);
        // Default renderer but no GPU text (editor preview): egui too.
        let mut scene = text_scene(UiText::default());
        let mut ui = GameUi::default();
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert!(ui.egui_content);
        // A text system without any font also falls back instead of vanishing.
        let empty = zestengine_text::TextShared::new(zestengine_text::TextSystem::new(Default::default()));
        let mut ui = GameUi { text: Some(empty.clone()), ..Default::default() };
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert!(empty.draws.lock().unwrap().is_empty() && ui.egui_content);
    }

    #[test]
    fn localization_key_overrides_text_and_changes_language() {
        let mut loc = zestengine_core::Localization::from_ron(
            r#"Localization(languages: ["en", "fr"], entries: {"play": ["Play", "Jouer"]})"#,
        )
        .unwrap();
        let handle = text_handle();
        let mut scene = text_scene(UiText { text: "x".into(), loc_key: "play".into(), ..Default::default() });
        let mut ui = GameUi { text: Some(handle.clone()), strings: Some(std::sync::Arc::new(loc.clone())), ..Default::default() };
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert_eq!(handle.draws.lock().unwrap().last().unwrap().instances.len(), 4, "Play");
        handle.draws.lock().unwrap().clear();
        loc.set_language("fr");
        ui.strings = Some(std::sync::Arc::new(loc));
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert_eq!(handle.draws.lock().unwrap().last().unwrap().instances.len(), 5, "Jouer");
        assert_eq!(handle.system.lock().unwrap().language(), "fr");
        // An unknown key falls back to the literal text.
        handle.draws.lock().unwrap().clear();
        let mut scene = text_scene(UiText { text: "ab".into(), loc_key: "nope".into(), ..Default::default() });
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert_eq!(handle.draws.lock().unwrap().last().unwrap().instances.len(), 2);
    }

    #[test]
    fn world_canvas_text_draws_in_world_space() {
        let handle = text_handle();
        let mut scene = text_scene(UiText { text: "Hi".into(), ..Default::default() });
        let mut ui = GameUi { text: Some(handle.clone()), world_xform: Some(Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0))), ..Default::default() };
        run_once(&mut ui, &mut scene, UiScope::Screen);
        let draws = handle.draws.lock().unwrap();
        assert_eq!(draws[0].space, TextSpace::World);
        assert!(draws[0].supersample);
        // World matrix translation (+5 m) plus the rect origin (245 px) in canvas space.
        assert!((draws[0].transform[3][1] - 250.0).abs() < 0.01);
    }

    #[test]
    fn panels_still_need_egui_next_to_gpu_text() {
        let handle = text_handle();
        let mut scene = LoadedScene::empty();
        let canvas = obj(&mut scene, None, vec![Box::new(UiCanvas::default())]);
        obj(&mut scene, Some(canvas), vec![Box::new(rect_at(0.0)), Box::new(UiPanel::default()), Box::new(UiText::default())]);
        let mut ui = GameUi { text: Some(handle.clone()), ..Default::default() };
        run_once(&mut ui, &mut scene, UiScope::Screen);
        assert!(ui.egui_content, "the panel paints through egui");
        assert_eq!(handle.draws.lock().unwrap().len(), 1, "the text still goes to the GPU path");
    }

    #[test]
    fn native_canvas_selection_preserves_order_instances_scopes_and_editor_reveal() {
        use zestengine_core::{Disabled, ObjectId};
        let mut scene = LoadedScene::empty();
        let first = obj(&mut scene, None, vec![Box::new(UiCanvas { sort_order: 10, ..Default::default() })]);
        let second = obj(&mut scene, None, vec![
            Box::new(Disabled(Box::new(UiCanvas { sort_order: -100, ..Default::default() }))),
            Box::new(UiCanvas { sort_order: -1, ..Default::default() }),
            Box::new(UiCanvas { sort_order: 100, ..Default::default() }),
        ]);
        let child = obj(&mut scene, Some(first), vec![Box::new(UiRect::default())]);
        let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(960.0, 540.0));
        let indices = |scene: &LoadedScene, scope, reveal| layout(scene, view, 1.0, scope, reveal).into_iter().map(|node| node.index).collect::<Vec<_>>();
        assert_eq!(indices(&scene, UiScope::Screen, None), [second, first, child]);
        assert_eq!(indices(&scene, UiScope::Canvas(first), None), [first, child]);
        let id: ObjectId = scene.objects[first].id();
        scene.objects.query(|world, _| world.get_mut::<zestengine_core::ecs::Enabled>(world.entity(id).unwrap()).unwrap().0 = false);
        assert_eq!(indices(&scene, UiScope::Screen, None), [second]);
        assert_eq!(indices(&scene, UiScope::Screen, Some(child)), [second, first, child]);
    }

    #[test]
    fn native_widget_state_survives_reordering_and_reads_live_edits() {
        let (mut scene, items) = menu_scene();
        let button = scene.objects[items[1]].id();
        let field = scene.objects[items[2]].id();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        let at = Pos2::new(960.0, 540.0);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        scene.objects.swap(0, items[2]);
        assert_eq!(run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true)).clicks, [button]);
        let at = Pos2::new(960.0, 740.0);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true));
        scene.objects.query(|world, _| {
            let entity = world.entity(field).unwrap();
            world.get_mut::<UiTextField>(entity).unwrap().text = "native".into();
            world.get_mut::<UiRect>(entity).unwrap().position = [150.0, 100.0];
        });
        scene.objects.swap(0, items[1]);
        let frame = run_frame(&mut ui, &ctx, &mut scene, UiInput {
            keys: UiKeys { text: "!".into(), enter: true, ..Default::default() }, ..Default::default()
        });
        assert_eq!(frame.submits, [field]);
        assert!(!frame.text_editing);
        scene.objects.query(|world, _| {
            assert_eq!(world.get::<UiTextField>(world.entity(field).unwrap()).unwrap().text, "native!");
        });
        let nodes = layout(&scene, Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0)), 1.0, UiScope::Screen, None);
        assert_eq!(nodes.iter().find(|node| node.id == field).unwrap().rect.center(), Pos2::new(1260.0, 740.0));
        let row = scene.objects.index_of(field).unwrap();
        scene.objects[row].components.set_enabled(1, false);
        assert!(!run_frame(&mut ui, &ctx, &mut scene, pointer(Pos2::new(1260.0, 740.0), true, true, false)).pointer_over_ui);
        scene.objects.remove(row);
        assert!(run_frame(&mut ui, &ctx, &mut scene, UiInput { nav: UiNav { submit: true, ..Default::default() }, ..Default::default() }).submits.is_empty());
    }

    #[test]
    fn native_widget_mutations_preserve_duplicate_instances_and_slider_focus() {
        use zestengine_core::Disabled;
        let mut scene = LoadedScene::empty();
        let canvas = obj(&mut scene, None, vec![Box::new(UiCanvas { reference_size: [960.0, 540.0], ..Default::default() })]);
        let toggle = obj(&mut scene, Some(canvas), vec![Box::new(rect_at(-100.0)),
            Box::new(Disabled(Box::new(UiToggle::default()))), Box::new(UiToggle::default()), Box::new(UiToggle::default())]);
        let slider = obj(&mut scene, Some(canvas), vec![Box::new(rect_at(0.0)), Box::new(UiSlider::default())]);
        let toggle_id = scene.objects[toggle].id();
        let slider_id = scene.objects[slider].id();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        let at = Pos2::new(960.0, 340.0);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        assert_eq!(run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true)).clicks, [toggle_id]);
        assert!(scene.objects[toggle].components.get_typed::<UiToggle>().unwrap().on);
        scene.objects[toggle].components.set_enabled(2, false);
        assert!(!scene.objects[toggle].components.get_typed::<UiToggle>().unwrap().on, "only the primary enabled duplicate was toggled");
        scene.objects.query(|world, _| world.get_mut::<UiSlider>(world.entity(slider_id).unwrap()).unwrap().max = 20.0);
        run_frame(&mut ui, &ctx, &mut scene, pointer(Pos2::new(1060.0, 540.0), true, true, false));
        assert_eq!(scene.objects[slider].components.get_typed::<UiSlider>().unwrap().value, 15.0);
        scene.objects.swap(toggle, slider);
        run_frame(&mut ui, &ctx, &mut scene, UiInput { nav: UiNav { right: true, ..Default::default() }, ..Default::default() });
        scene.objects.query(|world, _| assert_eq!(world.get::<UiSlider>(world.entity(slider_id).unwrap()).unwrap().value, 17.0));
        assert_eq!(ui.focused, Some(slider_id));
    }

    #[test]
    fn native_text_editing_stops_when_the_field_or_its_parent_is_inactive() {
        let (mut scene, items) = menu_scene();
        let id = scene.objects[items[2]].id();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        let at = Pos2::new(960.0, 740.0);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        assert!(run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true)).text_editing);
        scene.objects[items[2]].components.set_enabled(1, false);
        let frame = run_frame(&mut ui, &ctx, &mut scene, UiInput { keys: UiKeys { text: "ignored".into(), enter: true, ..Default::default() }, ..Default::default() });
        assert!(frame.submits.is_empty(), "a disabled field cannot submit using a stale editing ID");
        assert!(!frame.text_editing);
        scene.objects[items[2]].components.set_enabled(1, true);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true));
        scene.objects[0].set_enabled(false);
        assert!(!run_frame(&mut ui, &ctx, &mut scene, UiInput::default()).text_editing);
        scene.objects.query(|world, _| assert!(!world.get::<UiTextField>(world.entity(id).unwrap()).unwrap().editing, "inactive parent clears the field's live editing flag too"));
        scene.objects[0].set_enabled(true);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true));
        scene.objects.query(|world, _| world.get_mut::<UiTextField>(world.entity(id).unwrap()).unwrap().interactable = false);
        let frame = run_frame(&mut ui, &ctx, &mut scene, UiInput { keys: UiKeys { text: "ignored".into(), enter: true, ..Default::default() }, ..Default::default() });
        assert!(frame.submits.is_empty() && !frame.text_editing);
        scene.objects.query(|world, _| {
            let field = world.get::<UiTextField>(world.entity(id).unwrap()).unwrap();
            assert!(field.text.is_empty() && !field.editing);
        });
    }

    #[test]
    fn native_layout_reveals_deep_ancestors_and_respects_canvas_boundaries() {
        let mut scene = LoadedScene::empty();
        let root = obj(&mut scene, None, Vec::new());
        scene.objects[root].set_enabled(false);
        let mut parent = root;
        for _ in 0..96 { parent = obj(&mut scene, Some(parent), Vec::new()); }
        let canvas = obj(&mut scene, Some(parent), vec![Box::new(UiCanvas::default())]);
        let child = obj(&mut scene, Some(canvas), vec![Box::new(UiRect::default())]);
        let sibling = obj(&mut scene, Some(canvas), vec![Box::new(UiRect::default())]);
        scene.objects[sibling].set_enabled(false);
        let nested = obj(&mut scene, Some(canvas), vec![Box::new(UiCanvas::default())]);
        obj(&mut scene, Some(nested), vec![Box::new(UiRect::default())]);
        let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(960.0, 540.0));
        assert!(layout(&scene, view, 1.0, UiScope::Canvas(canvas), None).is_empty());
        assert_eq!(layout(&scene, view, 1.0, UiScope::Canvas(canvas), Some(child)).iter().map(|node| node.index).collect::<Vec<_>>(), [canvas, child]);
        let child_id = scene.objects[child].id();
        let sibling_id = scene.objects[sibling].id();
        scene.objects.query(|world, _| {
            world.get_mut::<Enabled>(world.entity(sibling_id).unwrap()).unwrap().0 = true;
            world.get_mut::<Parent>(world.entity(child_id).unwrap()).unwrap().0 = world.entity(sibling_id);
        });
        let shown = layout(&scene, view, 1.0, UiScope::Canvas(canvas), Some(child));
        assert_eq!(shown.iter().map(|node| node.index).collect::<Vec<_>>(), [canvas, sibling, child]);
    }

    #[test]
    fn native_world_canvases_follow_current_hierarchy_visibility_and_primary_components() {
        use zestengine_core::{Disabled, Transform};
        let mut scene = LoadedScene::empty();
        let parent = obj(&mut scene, None, Vec::new());
        let canvas = obj(&mut scene, Some(parent), vec![
            Box::new(Disabled(Box::new(UiCanvas { render_mode: UiRenderMode::World, world_size: [9.0, 9.0], ..Default::default() }))),
            Box::new(UiCanvas { render_mode: UiRenderMode::World, world_size: [2.0, 1.0], ..Default::default() }),
            Box::new(UiCanvas { render_mode: UiRenderMode::World, world_size: [3.0, 1.5], ..Default::default() }),
        ]);
        let id = scene.objects[canvas].id();
        let parent_id = scene.objects[parent].id();
        scene.objects.query(|world, _| {
            world.get_mut::<Transform>(world.entity(parent_id).unwrap()).unwrap().translation = Vec3::new(3.0, 4.0, 0.0);
            world.get_mut::<Transform>(world.entity(id).unwrap()).unwrap().translation = Vec3::new(1.0, 2.0, -3.0);
        });
        let instances = world_canvas_instances(&scene);
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].1, id);
        assert_eq!(instances[0].2, [2.0, 1.0]);
        assert_eq!(instances[0].4.transform_point3(Vec3::ZERO), Vec3::new(4.0, 6.0, -3.0));
        scene.objects.swap(parent, canvas);
        scene.objects[0].components.set_enabled(1, false);
        assert_eq!(world_canvas_instances(&scene)[0].2, [3.0, 1.5]);
        scene.scene_view.edit = true;
        scene.set_view_hidden(1, true);
        assert!(world_canvas_instances(&scene).is_empty());
        scene.scene_view.edit = false;
        assert_eq!(world_canvas_instances(&scene)[0].0, 0);
        scene.objects.query(|world, _| world.get_mut::<Enabled>(world.entity(parent_id).unwrap()).unwrap().0 = false);
        assert!(world_canvas_instances(&scene).is_empty());
        scene.objects.query(|world, _| world.get_mut::<Parent>(world.entity(id).unwrap()).unwrap().0 = None);
        assert_eq!(world_canvas_instances(&scene)[0].4.transform_point3(Vec3::ZERO), Vec3::new(1.0, 2.0, -3.0));
    }

    #[test]
    fn layout_scales_with_screen() {
        let (scene, items) = menu_scene();
        let nodes = layout(&scene, Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0)), 1.0, UiScope::Screen, None);
        let n = nodes.iter().find(|n| n.index == items[1]).unwrap();
        assert_eq!(n.rect.size(), Vec2::new(400.0, 100.0));
        assert_eq!(n.rect.center(), Pos2::new(960.0, 540.0));
    }

    #[test]
    fn press_and_release_over_button_clicks_once() {
        let (mut scene, items) = menu_scene();
        let id = scene.objects[items[1]].id();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        let at = Pos2::new(960.0, 540.0);
        let f = run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        assert!(f.pointer_over_ui && f.clicks.is_empty());
        let f = run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true));
        assert_eq!(f.clicks, vec![id]);
        let f = run_frame(&mut ui, &ctx, &mut scene, pointer(Pos2::new(5.0, 5.0), false, false, false));
        assert!(!f.pointer_over_ui && f.clicks.is_empty());
    }

    #[test]
    fn release_off_the_button_cancels_click() {
        let (mut scene, _) = menu_scene();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        run_frame(&mut ui, &ctx, &mut scene, pointer(Pos2::new(960.0, 540.0), true, true, false));
        let f = run_frame(&mut ui, &ctx, &mut scene, pointer(Pos2::new(5.0, 5.0), false, false, true));
        assert!(f.clicks.is_empty());
    }

    #[test]
    fn gamepad_navigation_moves_focus_and_submits() {
        let (mut scene, items) = menu_scene();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        let nav = |nav: UiNav| UiInput { nav, ..Default::default() };
        run_frame(&mut ui, &ctx, &mut scene, nav(UiNav { down: true, ..Default::default() }));
        assert_eq!(ui.focused, Some(scene.objects[items[0]].id()), "first press focuses the first element");
        run_frame(&mut ui, &ctx, &mut scene, nav(UiNav { down: true, ..Default::default() }));
        assert_eq!(ui.focused, Some(scene.objects[items[1]].id()));
        let f = run_frame(&mut ui, &ctx, &mut scene, nav(UiNav { submit: true, ..Default::default() }));
        assert_eq!(f.clicks, vec![scene.objects[items[1]].id()]);
        run_frame(&mut ui, &ctx, &mut scene, nav(UiNav { up: true, ..Default::default() }));
        assert_eq!(ui.focused, Some(scene.objects[items[0]].id()));
    }

    #[test]
    fn text_field_edits_and_submits() {
        let (mut scene, items) = menu_scene();
        let field_id = scene.objects[items[2]].id();
        let ctx = egui::Context::default();
        let mut ui = GameUi::default();
        let at = Pos2::new(960.0, 540.0 + 200.0);
        run_frame(&mut ui, &ctx, &mut scene, pointer(at, true, true, false));
        let f = run_frame(&mut ui, &ctx, &mut scene, pointer(at, false, false, true));
        assert!(f.text_editing);
        let typing = UiInput {
            keys: UiKeys { text: "Neo1".into(), backspaces: 0, ..Default::default() },
            ..Default::default()
        };
        run_frame(&mut ui, &ctx, &mut scene, typing);
        let back = UiInput {
            keys: UiKeys { backspaces: 1, enter: true, ..Default::default() },
            ..Default::default()
        };
        let f = run_frame(&mut ui, &ctx, &mut scene, back);
        assert_eq!(f.submits, vec![field_id]);
        assert!(!f.text_editing);
        assert_eq!(find::<UiTextField>(&scene, items[2]).unwrap().text, "Neo");
    }

    #[test]
    fn ray_hits_world_canvas_at_expected_pixel() {
        let canvas = Mat4::from_translation(Vec3::new(0.0, 1.0, -2.0));
        let ray = UiRay {
            origin: Vec3::new(0.4, 1.2, 0.0),
            dir: Vec3::NEG_Z,
            down: false,
            pressed: false,
            released: false,
        };
        let (p, d) = ray_canvas_hit(&ray, canvas, [2.0, 1.0], [1000, 500]).unwrap();
        assert!((p.x - 700.0).abs() < 1e-3 && (p.y - 150.0).abs() < 1e-3, "{p:?}");
        assert!((d - 2.0).abs() < 1e-4);
        let miss = UiRay { origin: Vec3::new(3.0, 1.0, 0.0), ..ray };
        assert!(ray_canvas_hit(&miss, canvas, [2.0, 1.0], [1000, 500]).is_none());
    }

    /// Flat-screen world UI: with the VR Player's mouse-look locking the
    /// cursor, the view centre is the pointer and a click presses the button
    /// under the crosshair; a free cursor aims where it points.
    #[test]
    fn desktop_ray_clicks_a_world_canvas_under_the_crosshair() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, -1.0), Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let canvas = Mat4::from_translation(Vec3::new(0.0, 1.0, -2.0));
        let click = [true, true, false];
        let locked = desktop_ui_ray(view, proj, None, true, click).expect("locked cursor still aims");
        assert!(locked.pressed && locked.down && !locked.released);
        let (p, d) = ray_canvas_hit(&locked, canvas, [2.0, 1.0], [1000, 500]).expect("centre ray hits the board");
        assert!((p.x - 500.0).abs() < 1.0 && (p.y - 250.0).abs() < 1.0, "centre of the canvas: {p:?}");
        assert!((d - 1.9).abs() < 1e-2, "measured from the near plane: {d}");
        // A free cursor at the right edge of the view misses the 2 m wide board; unknown cursor, no ray.
        let free = desktop_ui_ray(view, proj, Some(glam::Vec2::new(0.95, 0.5)), false, [false; 3]).unwrap();
        assert!(ray_canvas_hit(&free, canvas, [2.0, 1.0], [1000, 500]).is_none());
        assert!(desktop_ui_ray(view, proj, None, false, [false; 3]).is_none());
        // Releasing reports the edge.
        let up = desktop_ui_ray(view, proj, None, true, [false, false, true]).unwrap();
        assert!(up.released && !up.down);
    }
}
