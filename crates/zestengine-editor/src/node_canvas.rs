//! A reusable node-graph canvas in the ZestBay patchbay look: rounded cards
//! with a category chip and title, port rows with pins centred on the card
//! edge, smooth cubic-bezier wires coloured by data type, dot grid, box and
//! multi-select, snapping and search dimming.
//!
//! The module has two layers. The geometry functions at the top are pure
//! (bezier control points, flattening, wire hit-testing, box-vs-wire tests,
//! pin positions, snapping, zoom anchoring) and unit-tested without egui
//! state. [`NodeCanvasState::show`] is the thin painter and interaction layer
//! on top, driven through [`NodeCanvasModel`], so any graph UI can adopt it by
//! implementing that trait for its node store.
//!
//! Coordinates: node positions are card *centres* in canvas units; a screen
//! point is `centre + (pan + p) * zoom`.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};

use egui::{
    Color32, CornerRadius, FontId, Key, Mesh, PointerButton, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui, Vec2,
    pos2, vec2,
};

use crate::theme;

// ------------------------------------------------------------------ metrics

/// Dot-grid pitch in canvas units.
pub const GRID: f32 = 24.0;
pub const CARD_WIDTH: f32 = 184.0;
pub const CARD_RADIUS: f32 = 10.0;
pub const HEADER_H: f32 = 34.0;
pub const ROW_H: f32 = 22.0;
/// Space above the first and below the last port row.
pub const PAD_TOP: f32 = 4.0;
pub const PAD_BOTTOM: f32 = 8.0;
pub const PIN_DIAMETER: f32 = 10.0;
pub const PIN_HOVER_DIAMETER: f32 = 12.0;
/// Half-width of the wire hit band in screen px (an 8 px band).
pub const WIRE_BAND: f32 = 4.0;
pub const WIRE_WIDTH: f32 = 1.75;
pub const WIRE_WIDTH_HOVER: f32 = 2.25;
pub const ZOOM_MIN: f32 = 0.25;
pub const ZOOM_MAX: f32 = 2.5;
/// Non-matching nodes fade to this opacity while a search is active.
pub const DIM_OPACITY: f32 = 0.35;
/// Flattening tolerance (screen px) for drawing and hit-testing.
const FLATTEN_TOL: f32 = 0.4;

// ----------------------------------------------------------------- geometry

/// Control-point offset for a wire from x1 to x2: `clamp(0.5 * |dx|, 32, 200)`,
/// horizontal only, so wires leave and enter pins horizontally.
pub fn wire_ctrl(x1: f32, x2: f32, scale: f32) -> f32 {
    (0.5 * (x2 - x1).abs()).clamp(32.0 * scale, 200.0 * scale)
}

/// The four bezier points of a wire from output pin `a` to input pin `b`.
/// `scale` is the zoom (the 32..200 clamp is in canvas units).
pub fn wire_bezier(a: Pos2, b: Pos2, scale: f32) -> [Pos2; 4] {
    let d = wire_ctrl(a.x, b.x, scale);
    [a, pos2(a.x + d, a.y), pos2(b.x - d, b.y), b]
}

pub fn bezier_at(c: &[Pos2; 4], t: f32) -> Pos2 {
    let u = 1.0 - t;
    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    pos2(
        w0 * c[0].x + w1 * c[1].x + w2 * c[2].x + w3 * c[3].x,
        w0 * c[0].y + w1 * c[1].y + w2 * c[2].y + w3 * c[3].y,
    )
}

/// Bounding box of the control points. A bezier lies inside its control hull,
/// so this is a conservative box for culling and early rejection.
pub fn bezier_bounds(c: &[Pos2; 4]) -> Rect {
    Rect::from_points(c)
}

/// Segment count that keeps the polyline within `tol` of the curve (Wang's
/// bound for cubics), at least 2, at most 128.
pub fn segment_count(c: &[Pos2; 4], tol: f32) -> usize {
    let v = |p: Pos2| p.to_vec2();
    let d1 = (v(c[0]) - 2.0 * v(c[1]) + v(c[2])).length();
    let d2 = (v(c[1]) - 2.0 * v(c[2]) + v(c[3])).length();
    ((0.75 * d1.max(d2) / tol.max(1e-3)).sqrt().ceil() as usize).clamp(2, 128)
}

/// Sample the curve into `out` (cleared first) with an adaptive segment count.
pub fn flatten(c: &[Pos2; 4], tol: f32, out: &mut Vec<Pos2>) {
    out.clear();
    let n = segment_count(c, tol);
    out.reserve(n + 1);
    for i in 0..=n {
        out.push(if i == n { c[3] } else { bezier_at(c, i as f32 / n as f32) });
    }
}

/// Visit the flattened curve's segments without allocating.
fn for_each_segment(c: &[Pos2; 4], tol: f32, mut f: impl FnMut(Pos2, Pos2) -> bool) {
    let n = segment_count(c, tol);
    let mut prev = c[0];
    for i in 1..=n {
        let next = if i == n { c[3] } else { bezier_at(c, i as f32 / n as f32) };
        if f(prev, next) {
            return;
        }
        prev = next;
    }
}

pub fn dist_point_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= f32::EPSILON {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Distance from `p` to the curve (distance to the flattened polyline).
pub fn dist_to_bezier(p: Pos2, c: &[Pos2; 4], tol: f32) -> f32 {
    let mut best = f32::INFINITY;
    for_each_segment(c, tol, |a, b| {
        best = best.min(dist_point_segment(p, a, b));
        false
    });
    best
}

/// Liang-Barsky: does the segment touch the rect?
pub fn segment_intersects_rect(a: Pos2, b: Pos2, r: Rect) -> bool {
    let (mut t0, mut t1) = (0.0_f32, 1.0_f32);
    let d = b - a;
    for (p, q) in [(-d.x, a.x - r.min.x), (d.x, r.max.x - a.x), (-d.y, a.y - r.min.y), (d.y, r.max.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return false;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

/// Does any part of the curve cross or lie inside `r` (box select of wires)?
pub fn bezier_intersects_rect(c: &[Pos2; 4], r: Rect, tol: f32) -> bool {
    if !bezier_bounds(c).intersects(r) {
        return false;
    }
    let mut hit = false;
    for_each_segment(c, tol, |a, b| {
        hit = segment_intersects_rect(a, b, r);
        hit
    });
    hit
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Input,
    Output,
}

impl Side {
    pub fn opposite(self) -> Self {
        match self {
            Self::Input => Self::Output,
            Self::Output => Self::Input,
        }
    }
}

/// Card size in canvas units for `rows` port rows plus a body of `body_h`.
pub fn card_size(rows: usize, body_h: f32) -> Vec2 {
    vec2(CARD_WIDTH, HEADER_H + PAD_TOP + ROW_H * rows as f32 + body_h + PAD_BOTTOM)
}

/// Offset from a card's top edge to the centre of port row `row`.
pub fn row_center_y(row: usize) -> f32 {
    HEADER_H + PAD_TOP + ROW_H * (row as f32 + 0.5)
}

/// Pin centre: exactly on the card's left (inputs) or right (outputs) edge.
pub fn pin_pos(card: Rect, side: Side, row: usize) -> Pos2 {
    let x = match side {
        Side::Input => card.left(),
        Side::Output => card.right(),
    };
    pos2(x, card.top() + row_center_y(row))
}

/// Pan/zoom. `screen = centre + (pan + p) * zoom`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub pan: Vec2,
    pub zoom: f32,
}

impl Default for View {
    fn default() -> Self {
        Self { pan: Vec2::ZERO, zoom: 1.0 }
    }
}

impl View {
    pub fn to_screen(&self, centre: Pos2, p: Pos2) -> Pos2 {
        centre + (self.pan + p.to_vec2()) * self.zoom
    }

    pub fn to_canvas(&self, centre: Pos2, s: Pos2) -> Pos2 {
        ((s - centre) / self.zoom - self.pan).to_pos2()
    }

    pub fn rect_to_screen(&self, centre: Pos2, r: Rect) -> Rect {
        Rect::from_min_max(self.to_screen(centre, r.min), self.to_screen(centre, r.max))
    }

    /// The canvas point at the middle of the viewport.
    pub fn center_canvas(&self) -> Pos2 {
        (-self.pan).to_pos2()
    }

    /// Scale by `factor` keeping the canvas point under `cursor` fixed.
    pub fn zoom_at(&mut self, centre: Pos2, cursor: Pos2, factor: f32) {
        let before = (cursor - centre) / self.zoom - self.pan;
        self.zoom = (self.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        self.pan = (cursor - centre) / self.zoom - before;
    }
}

/// A view that fits `bounds` (canvas units) in a viewport of `viewport` px.
pub fn fit_view(bounds: Rect, viewport: Vec2, margin: f32) -> View {
    let avail = (viewport - Vec2::splat(margin * 2.0)).max(Vec2::splat(1.0));
    let size = bounds.size().max(Vec2::splat(1.0));
    let zoom = (avail.x / size.x).min(avail.y / size.y).clamp(ZOOM_MIN, 1.5);
    View { pan: -bounds.center().to_vec2(), zoom }
}

pub fn snap_to_grid(v: f32, grid: f32) -> f32 {
    (v / grid).round() * grid
}

pub fn snap_pos(p: Pos2, grid: f32) -> Pos2 {
    pos2(snap_to_grid(p.x, grid), snap_to_grid(p.y, grid))
}

/// What the pointer is over. Pins beat nodes, nodes beat wires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    None,
    Pin(PortRef, Side),
    Node(usize),
    /// Index into the frame's link list.
    Wire(usize),
}

pub fn prioritise(pin: Option<(PortRef, Side)>, node: Option<usize>, wire: Option<usize>) -> Hit {
    if let Some((p, s)) = pin {
        Hit::Pin(p, s)
    } else if let Some(n) = node {
        Hit::Node(n)
    } else if let Some(w) = wire {
        Hit::Wire(w)
    } else {
        Hit::None
    }
}

// -------------------------------------------------------------------- model

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PortRef {
    pub node: usize,
    pub port: usize,
}

/// A wire from an output port to an input port. Identity ignores the colour.
#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub from: PortRef,
    pub to: PortRef,
    pub color: Color32,
}

impl PartialEq for Link {
    fn eq(&self, o: &Self) -> bool {
        self.from == o.from && self.to == o.to
    }
}
impl Eq for Link {}

pub struct NodeStyle {
    pub title: Cow<'static, str>,
    /// Category colour for the header chip.
    pub chip: Color32,
    /// Icon-font glyph drawn in the chip.
    pub glyph: &'static str,
}

pub struct PortDesc {
    pub label: Cow<'static, str>,
    /// Shown in the pin tooltip next to the full name.
    pub type_label: &'static str,
    /// Signal colour of the pin (and of wires leaving an output).
    pub color: Color32,
}

/// What the canvas needs from a graph. Node ids are indices `0..node_count()`.
pub trait NodeCanvasModel {
    fn node_count(&self) -> usize;
    /// Card centre in canvas units.
    fn position(&self, node: usize) -> Pos2;
    fn set_position(&mut self, node: usize, pos: Pos2);
    fn style(&self, node: usize) -> NodeStyle;
    fn input_count(&self, node: usize) -> usize;
    fn output_count(&self, node: usize) -> usize;
    fn input(&self, node: usize, port: usize) -> PortDesc;
    fn output(&self, node: usize, port: usize) -> PortDesc;
    /// Append every wire to `out` (it is cleared by the canvas).
    fn links(&self, out: &mut Vec<Link>);

    /// May `from` (an output) be wired into `to` (an input)?
    fn can_connect(&self, from: PortRef, to: PortRef) -> bool;
    /// Make the connection; true when the graph changed.
    fn connect(&mut self, from: PortRef, to: PortRef) -> bool;
    fn can_disconnect(&self, _link: &Link) -> bool {
        true
    }
    fn disconnect(&mut self, _link: &Link) -> bool {
        false
    }
    fn delete_node(&mut self, node: usize);

    /// Search/filter dimming: false fades the node to 35 %.
    fn matches_filter(&self, _node: usize) -> bool {
        true
    }
    /// Height of the node's body area (below the port rows), canvas units.
    fn body_height(&self, _node: usize) -> f32 {
        0.0
    }
    /// Draw editable widgets in the body. The ui is sized to the body rect
    /// and its fonts and spacing are already zoom-scaled. True on edit.
    fn draw_body(&mut self, _ui: &mut Ui, _node: usize) -> bool {
        false
    }
    /// Does this unconnected input show an inline default-value widget?
    fn has_port_default(&self, _node: usize, _input: usize) -> bool {
        false
    }
    /// Draw the inline default for an unconnected input (zoom-scaled ui).
    fn draw_port_default(&mut self, _ui: &mut Ui, _node: usize, _input: usize) -> bool {
        false
    }
}

/// What a right-click landed on; the host builds the menu.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CanvasTarget {
    Node(usize),
    Wire(Link),
    Canvas,
}

pub struct CanvasOutput {
    pub response: Response,
    /// The graph was edited (positions, wires, deleted nodes, widget values).
    pub changed: bool,
    /// Set on the frame of a right-click, with the canvas-space position.
    pub context: Option<(CanvasTarget, Pos2)>,
}

// ------------------------------------------------------------------- state

#[derive(Default)]
enum Drag {
    #[default]
    None,
    Nodes { start: Vec<(usize, Pos2)>, anchor: usize, accum: Vec2 },
    Wire { from: PortRef, side: Side, valid: HashSet<PortRef> },
    Box { origin: Pos2, base: BTreeSet<usize>, base_links: Vec<Link>, additive: bool },
    Pan,
}

#[derive(Clone, Copy)]
struct Meta {
    card: Rect,
    inputs: usize,
    outputs: usize,
    body_h: f32,
    dim: bool,
}

/// Per-canvas interaction state; own one per graph editor.
pub struct NodeCanvasState {
    pub view: View,
    pub selected: BTreeSet<usize>,
    pub selected_links: Vec<Link>,
    /// The most recently selected node (what an inspector shows).
    pub primary: Option<usize>,
    /// Snap dragged nodes to [`GRID`] (Shift inverts it while dragging).
    pub snap: bool,
    drag: Drag,
    frame_request: Option<bool>,
    meta: Vec<Meta>,
    links: Vec<Link>,
    order: Vec<usize>,
    in_connected: HashSet<PortRef>,
    out_connected: HashSet<PortRef>,
    scratch: Vec<Pos2>,
}

impl Default for NodeCanvasState {
    fn default() -> Self {
        Self {
            view: View::default(),
            selected: BTreeSet::new(),
            selected_links: Vec::new(),
            primary: None,
            snap: false,
            drag: Drag::None,
            frame_request: None,
            meta: Vec::new(),
            links: Vec::new(),
            order: Vec::new(),
            in_connected: HashSet::new(),
            out_connected: HashSet::new(),
            scratch: Vec::new(),
        }
    }
}

fn qfont(size: f32, semibold: bool, mono: bool) -> FontId {
    let s = (size * 2.0).round() / 2.0;
    if mono {
        theme::mono(s)
    } else if semibold {
        theme::semibold(s)
    } else {
        FontId::proportional(s)
    }
}

impl NodeCanvasState {
    /// Fit all nodes (or the selection) in view on the next frame.
    pub fn request_frame(&mut self, selection_only: bool) {
        self.frame_request = Some(selection_only);
    }

    pub fn clear_selection(&mut self) {
        self.selected.clear();
        self.selected_links.clear();
        self.primary = None;
    }

    pub fn select_only(&mut self, node: usize) {
        self.selected.clear();
        self.selected_links.clear();
        self.selected.insert(node);
        self.primary = Some(node);
    }

    /// Forget everything tied to a graph (call when another graph is opened).
    pub fn reset(&mut self) {
        *self = Self { snap: self.snap, ..Self::default() };
    }

    fn screen_card(&self, centre: Pos2, n: usize) -> Rect {
        self.view.rect_to_screen(centre, self.meta[n].card)
    }

    fn pin_screen(&self, centre: Pos2, p: PortRef, side: Side) -> Pos2 {
        self.view.to_screen(centre, pin_pos(self.meta[p.node].card, side, p.port))
    }

    fn link_curve(&self, centre: Pos2, l: &Link) -> [Pos2; 4] {
        wire_bezier(self.pin_screen(centre, l.from, Side::Output), self.pin_screen(centre, l.to, Side::Input), self.view.zoom)
    }

    fn pin_hit_radius(&self) -> f32 {
        (6.0 * self.view.zoom).max(8.0)
    }

    /// The pin under `p` on `side` (or either side when `None`), topmost node first.
    fn pin_at(&self, centre: Pos2, p: Pos2, only: Option<Side>) -> Option<(PortRef, Side)> {
        let r = self.pin_hit_radius();
        for &n in self.order.iter().rev() {
            let m = &self.meta[n];
            if !self.screen_card(centre, n).expand(r).contains(p) {
                continue;
            }
            for side in [Side::Output, Side::Input] {
                if only.is_some_and(|o| o != side) {
                    continue;
                }
                let count = if side == Side::Input { m.inputs } else { m.outputs };
                for port in 0..count {
                    let pr = PortRef { node: n, port };
                    if self.pin_screen(centre, pr, side).distance(p) <= r {
                        return Some((pr, side));
                    }
                }
            }
        }
        None
    }

    fn node_at(&self, centre: Pos2, p: Pos2) -> Option<usize> {
        self.order.iter().rev().copied().find(|&n| self.screen_card(centre, n).contains(p))
    }

    fn wire_at(&self, centre: Pos2, p: Pos2) -> Option<usize> {
        let mut best = (WIRE_BAND, None);
        for (i, l) in self.links.iter().enumerate() {
            let c = self.link_curve(centre, l);
            if !bezier_bounds(&c).expand(WIRE_BAND).contains(p) {
                continue;
            }
            let d = dist_to_bezier(p, &c, FLATTEN_TOL);
            if d <= best.0 {
                best = (d, Some(i));
            }
        }
        best.1
    }

    fn hit_at(&self, centre: Pos2, p: Pos2) -> Hit {
        let pin = self.pin_at(centre, p, None);
        let node = if pin.is_none() { self.node_at(centre, p) } else { None };
        let wire = if pin.is_none() && node.is_none() { self.wire_at(centre, p) } else { None };
        prioritise(pin, node, wire)
    }

    fn rebuild(&mut self, model: &dyn NodeCanvasModel) {
        let count = model.node_count();
        self.meta.clear();
        for n in 0..count {
            let (inputs, outputs) = (model.input_count(n), model.output_count(n));
            let body_h = model.body_height(n);
            let size = card_size(inputs.max(outputs).max(1), body_h);
            self.meta.push(Meta {
                card: Rect::from_center_size(model.position(n), size),
                inputs,
                outputs,
                body_h,
                dim: !model.matches_filter(n),
            });
        }
        self.selected.retain(|&n| n < count);
        if self.primary.is_some_and(|n| n >= count) {
            self.primary = None;
        }
        self.links.clear();
        model.links(&mut self.links);
        let meta = &self.meta;
        self.links.retain(|l| {
            meta.get(l.from.node).is_some_and(|m| l.from.port < m.outputs) && meta.get(l.to.node).is_some_and(|m| l.to.port < m.inputs)
        });
        let links = &self.links;
        self.selected_links.retain(|s| links.contains(s));
        self.in_connected.clear();
        self.out_connected.clear();
        for l in &self.links {
            self.out_connected.insert(l.from);
            self.in_connected.insert(l.to);
        }
        // Selected nodes draw (and pick) on top.
        self.order.clear();
        self.order.extend((0..count).filter(|n| !self.selected.contains(n)));
        self.order.extend(self.selected.iter().copied());
    }

    fn select_in_box(&mut self, centre: Pos2, screen_box: Rect, base: &BTreeSet<usize>, base_links: &[Link], additive: bool) {
        self.selected.clear();
        self.selected_links.clear();
        if additive {
            self.selected.extend(base.iter().copied());
            self.selected_links.extend_from_slice(base_links);
        }
        for n in 0..self.meta.len() {
            if self.screen_card(centre, n).intersects(screen_box) {
                self.selected.insert(n);
            }
        }
        for l in &self.links {
            if !self.selected_links.contains(l) && bezier_intersects_rect(&self.link_curve(centre, l), screen_box, FLATTEN_TOL) {
                self.selected_links.push(*l);
            }
        }
        self.primary = self.selected.iter().next_back().copied();
    }

    /// Draw the canvas into the rest of `ui` and process input.
    pub fn show(&mut self, ui: &mut Ui, model: &mut dyn NodeCanvasModel) -> CanvasOutput {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let centre = rect.center();
        let mut changed = false;
        let mut context = None;
        self.rebuild(model);

        if let Some(sel_only) = self.frame_request.take() {
            let mut bounds: Option<Rect> = None;
            for (n, m) in self.meta.iter().enumerate() {
                if sel_only && !self.selected.contains(&n) {
                    continue;
                }
                bounds = Some(bounds.map_or(m.card, |b| b.union(m.card)));
            }
            if let Some(b) = bounds.or_else(|| self.meta.iter().map(|m| m.card).reduce(|a, b| a.union(b))) {
                self.view = fit_view(b, rect.size(), 48.0);
            }
        }

        let (modifiers, space, pointer, press_origin) =
            ui.input(|i| (i.modifiers, i.key_down(Key::Space), i.pointer.hover_pos(), i.pointer.press_origin()));
        let ctrl = modifiers.command || modifiers.ctrl;

        // Zoom anchored at the cursor.
        if response.hovered()
            && let Some(p) = pointer
        {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (1.0 + scroll * 0.0015) * pinch;
            if (factor - 1.0).abs() > 1e-5 {
                self.view.zoom_at(centre, p, factor);
            }
        }
        if ui.input(|i| i.key_pressed(Key::Escape)) {
            self.drag = Drag::None;
        }

        // ---- hover (pin > node > wire)
        let idle = matches!(self.drag, Drag::None);
        let hover = match pointer {
            Some(p) if rect.contains(p) && (response.hovered() || !idle) => match &self.drag {
                Drag::None | Drag::Nodes { .. } => self.hit_at(centre, p),
                Drag::Wire { side, .. } => {
                    self.pin_at(centre, p, Some(side.opposite())).map_or(Hit::None, |(pr, s)| Hit::Pin(pr, s))
                }
                _ => Hit::None,
            },
            _ => Hit::None,
        };

        self.draw_grid(&painter, rect);
        self.draw_wires(&painter, rect, centre, hover);
        changed |= self.draw_nodes(ui, &painter, rect, centre, model, hover);
        self.draw_drag_overlay(&painter, centre, pointer, model);

        // ---- interaction
        let zoom = self.view.zoom;
        if response.drag_started_by(PointerButton::Primary) {
            let origin = press_origin.unwrap_or(centre);
            let hit = self.hit_at(centre, origin);
            if space {
                self.drag = Drag::Pan;
            } else {
                match hit {
                    Hit::Pin(pr, side) => {
                        let valid = self.valid_targets(model, pr, side);
                        self.drag = Drag::Wire { from: pr, side, valid };
                    }
                    Hit::Node(n) => {
                        if !self.selected.contains(&n) {
                            if !ctrl {
                                self.selected.clear();
                            }
                            self.selected_links.clear();
                            self.selected.insert(n);
                        }
                        self.primary = Some(n);
                        let start = self.selected.iter().map(|&s| (s, model.position(s))).collect();
                        self.drag = Drag::Nodes { start, anchor: n, accum: Vec2::ZERO };
                    }
                    Hit::Wire(_) | Hit::None => {
                        let base = if ctrl { self.selected.clone() } else { BTreeSet::new() };
                        let base_links = if ctrl { self.selected_links.clone() } else { Vec::new() };
                        self.drag = Drag::Box { origin: self.view.to_canvas(centre, origin), base, base_links, additive: ctrl };
                    }
                }
            }
        }
        if response.dragged_by(PointerButton::Middle) {
            self.view.pan += response.drag_delta() / zoom;
        }
        if response.dragged_by(PointerButton::Primary) {
            let delta = response.drag_delta();
            let snapping = self.snap != modifiers.shift;
            let cur = response.interact_pointer_pos().or(pointer).unwrap_or(centre);
            match &mut self.drag {
                Drag::Pan => self.view.pan += delta / zoom,
                Drag::Nodes { start, anchor, accum } => {
                    *accum += delta / zoom;
                    let offset = start.iter().find(|(n, _)| n == anchor).map(|(_, p)| *p).unwrap_or(Pos2::ZERO);
                    let half = self.meta.get(*anchor).map_or(Vec2::ZERO, |m| m.card.size() * 0.5);
                    let shift = if snapping {
                        // Snap the anchor's top-left corner; everyone moves by the same delta.
                        let tl = offset - half + *accum;
                        snap_pos(tl, GRID) - (offset - half)
                    } else {
                        *accum
                    };
                    for (n, p) in start.iter() {
                        model.set_position(*n, *p + shift);
                    }
                    changed = true;
                }
                Drag::Box { origin, base, base_links, additive } => {
                    let a = self.view.to_screen(centre, *origin);
                    let b = Rect::from_two_pos(a, cur);
                    let (base, base_links, additive) = (base.clone(), base_links.clone(), *additive);
                    self.select_in_box(centre, b, &base, &base_links, additive);
                }
                _ => {}
            }
        }
        if response.drag_stopped() {
            match std::mem::take(&mut self.drag) {
                Drag::Wire { from, side, .. } => {
                    let end = pointer.or(response.interact_pointer_pos());
                    if let Some(p) = end
                        && let Some((target, _)) = self.pin_at(centre, p, Some(side.opposite()))
                    {
                        let (out, inp) = if side == Side::Output { (from, target) } else { (target, from) };
                        if out.node != inp.node && model.can_connect(out, inp) && model.connect(out, inp) {
                            changed = true;
                        }
                    }
                }
                other => drop(other),
            }
        }

        if response.clicked_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.hit_at(centre, p) {
                Hit::Node(n) => {
                    self.selected_links.clear();
                    if ctrl {
                        if !self.selected.remove(&n) {
                            self.selected.insert(n);
                            self.primary = Some(n);
                        }
                    } else {
                        self.select_only(n);
                    }
                }
                Hit::Wire(i) => {
                    let l = self.links[i];
                    if !ctrl {
                        self.selected.clear();
                        self.primary = None;
                        self.selected_links.clear();
                    }
                    self.selected_links.push(l);
                }
                Hit::Pin(..) => {}
                Hit::None => {
                    if !ctrl {
                        self.clear_selection();
                    }
                }
            }
        }
        if response.secondary_clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            self.drag = Drag::None;
            let at = self.view.to_canvas(centre, p);
            let target = match self.hit_at(centre, p) {
                Hit::Node(n) | Hit::Pin(PortRef { node: n, .. }, _) => {
                    if !self.selected.contains(&n) {
                        self.select_only(n);
                    }
                    CanvasTarget::Node(n)
                }
                Hit::Wire(i) => {
                    let l = self.links[i];
                    self.selected.clear();
                    self.primary = None;
                    self.selected_links = vec![l];
                    CanvasTarget::Wire(l)
                }
                Hit::None => CanvasTarget::Canvas,
            };
            context = Some((target, at));
        }

        // ---- keyboard
        let typing = ui.ctx().wants_keyboard_input();
        if response.hovered() && !typing {
            let (del, frame) = ui.input(|i| (i.key_pressed(Key::Delete) || i.key_pressed(Key::Backspace), i.key_pressed(Key::F)));
            if del {
                changed |= self.delete_selection(model);
            }
            if frame {
                self.request_frame(!self.selected.is_empty());
            }
        }

        // ---- pin tooltip
        if let Hit::Pin(pr, side) = hover
            && idle
        {
            let desc = if side == Side::Input { model.input(pr.node, pr.port) } else { model.output(pr.node, pr.port) };
            response.clone().on_hover_ui_at_pointer(|ui| {
                ui.label(egui::RichText::new(desc.label.as_ref()).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_STRONG));
                ui.label(egui::RichText::new(desc.type_label).color(desc.color));
            });
        }

        CanvasOutput { response, changed, context }
    }

    /// Delete the selected wires then nodes (highest id first so ids stay valid).
    pub fn delete_selection(&mut self, model: &mut dyn NodeCanvasModel) -> bool {
        let mut changed = false;
        for l in std::mem::take(&mut self.selected_links) {
            if model.can_disconnect(&l) && model.disconnect(&l) {
                changed = true;
            }
        }
        let nodes: Vec<usize> = self.selected.iter().rev().copied().collect();
        for n in nodes {
            model.delete_node(n);
            changed = true;
        }
        self.clear_selection();
        changed
    }

    fn valid_targets(&self, model: &dyn NodeCanvasModel, from: PortRef, side: Side) -> HashSet<PortRef> {
        let mut out = HashSet::new();
        for (n, m) in self.meta.iter().enumerate() {
            if n == from.node {
                continue;
            }
            let count = if side == Side::Output { m.inputs } else { m.outputs };
            for port in 0..count {
                let other = PortRef { node: n, port };
                let ok = if side == Side::Output { model.can_connect(from, other) } else { model.can_connect(other, from) };
                if ok {
                    out.insert(other);
                }
            }
        }
        out
    }

    // ------------------------------------------------------------- painting

    fn draw_grid(&self, painter: &egui::Painter, rect: Rect) {
        let mut step = GRID * self.view.zoom;
        while step < 10.0 {
            step *= 2.0;
        }
        let origin = rect.center() + self.view.pan * self.view.zoom;
        let size = 1.5 * self.view.zoom.clamp(0.9, 1.6);
        let color = theme::LINE_DASHED;
        let mut mesh = Mesh::default();
        let mut x = rect.left() + (origin.x - rect.left()).rem_euclid(step);
        while x < rect.right() {
            let mut y = rect.top() + (origin.y - rect.top()).rem_euclid(step);
            while y < rect.bottom() {
                mesh.add_colored_rect(Rect::from_center_size(pos2(x, y), Vec2::splat(size)), color);
                y += step;
            }
            x += step;
        }
        painter.add(Shape::mesh(mesh));
    }

    fn draw_wires(&mut self, painter: &egui::Painter, rect: Rect, centre: Pos2, hover: Hit) {
        let hovered_link = if let Hit::Wire(i) = hover { Some(i) } else { None };
        let zoom_w = self.view.zoom.clamp(0.7, 1.5);
        let mut scratch = std::mem::take(&mut self.scratch);
        for pass in [false, true] {
            for (i, l) in self.links.iter().enumerate() {
                let selected_link = self.selected_links.contains(l);
                let on_selected = self.selected.contains(&l.from.node) || self.selected.contains(&l.to.node);
                let hot = hovered_link == Some(i);
                let emphasised = selected_link || on_selected || hot;
                if emphasised != pass {
                    continue;
                }
                let c = self.link_curve(centre, l);
                if !bezier_bounds(&c).expand(4.0).intersects(rect) {
                    continue;
                }
                let color = if selected_link || on_selected { theme::accent() } else { l.color };
                let dim = self.meta[l.from.node].dim || self.meta[l.to.node].dim;
                let color = if dim { theme::with_alpha(color, DIM_OPACITY) } else { color };
                let width = (if hot || selected_link { WIRE_WIDTH_HOVER } else { WIRE_WIDTH }) * zoom_w;
                flatten(&c, FLATTEN_TOL, &mut scratch);
                painter.add(Shape::line(scratch.clone(), Stroke::new(width, color)));
            }
        }
        self.scratch = scratch;
    }

    /// Cards, pins and inline widgets. True when a widget edited the graph.
    fn draw_nodes(&self, ui: &mut Ui, painter: &egui::Painter, rect: Rect, centre: Pos2, model: &mut dyn NodeCanvasModel, hover: Hit) -> bool {
        let mut changed = false;
        let z = self.view.zoom;
        let linking = if let Drag::Wire { side, valid, .. } = &self.drag { Some((side.opposite(), valid)) } else { None };
        for &n in &self.order {
            let m = self.meta[n];
            let card = self.screen_card(centre, n);
            if !card.expand(24.0).intersects(rect) {
                continue;
            }
            let mut p = painter.clone();
            if m.dim {
                p.multiply_opacity(DIM_OPACITY);
            }
            let selected = self.selected.contains(&n);
            let hovered = hover == Hit::Node(n) || matches!(hover, Hit::Pin(pr, _) if pr.node == n);
            let r = CornerRadius::same((CARD_RADIUS * z).round().clamp(2.0, 20.0) as u8);
            let style = model.style(n);

            if z > 0.4 {
                let shadow = egui::Shadow {
                    offset: [0, (4.0 * z) as i8],
                    blur: (14.0 * z) as u8,
                    spread: 0,
                    color: Color32::from_black_alpha(110),
                };
                p.add(shadow.as_shape(card, r));
            }
            let border = if selected {
                Stroke::new(1.5_f32, theme::accent())
            } else if hovered {
                Stroke::new(1.0_f32, theme::TEXT_DIM)
            } else {
                Stroke::new(1.0_f32, theme::LINE_BUTTON)
            };
            p.rect(card, r, theme::BG_PANEL, border, StrokeKind::Inside);
            if selected {
                p.rect_stroke(card.expand(1.5), r, Stroke::new(3.0_f32, theme::with_alpha(theme::accent(), 0.30)), StrokeKind::Outside);
            }

            // Header: chip + title, hairline below.
            let hy = card.top() + HEADER_H * z;
            p.hline((card.left() + 1.0)..=(card.right() - 1.0), hy, Stroke::new(1.0_f32, theme::LINE_PANEL));
            let chip = Rect::from_min_size(card.min + vec2(10.0, 8.0) * z, Vec2::splat(18.0 * z));
            p.rect_filled(chip, CornerRadius::same((5.0 * z).round().max(1.0) as u8), theme::with_alpha(style.chip, 0.22));
            if z > 0.45 {
                p.text(chip.center(), egui::Align2::CENTER_CENTER, style.glyph, FontId::proportional((12.0 * z).max(6.0)), style.chip);
                let tx = chip.right() + 8.0 * z;
                let mut job = egui::text::LayoutJob::simple_singleline(style.title.into_owned(), qfont(12.0 * z, true, false), theme::TEXT_STRONG);
                job.wrap = egui::text::TextWrapping::truncate_at_width((card.right() - 10.0 * z - tx).max(8.0));
                let galley = p.layout_job(job);
                p.galley(pos2(tx, card.top() + (HEADER_H * z - galley.size().y) * 0.5), galley, theme::TEXT_STRONG);
            }

            // Port rows.
            let rows = m.inputs.max(m.outputs).max(1);
            let label_font = qfont(11.0 * z, false, true);
            for row in 0..rows {
                let y = card.top() + row_center_y(row) * z;
                if row < m.inputs {
                    let pr = PortRef { node: n, port: row };
                    let desc = model.input(n, row);
                    let connected = self.in_connected.contains(&pr);
                    if z > 0.5 {
                        let g = p.layout_no_wrap(desc.label.to_string(), label_font.clone(), theme::TEXT_SECONDARY);
                        let lw = g.size().x;
                        p.galley(pos2(card.left() + 14.0 * z, y - g.size().y * 0.5), g, theme::TEXT_SECONDARY);
                        if !connected && z > 0.6 && model.has_port_default(n, row) {
                            let w = (card.width() * 0.5 - lw - 22.0 * z).max(20.0);
                            let wr = Rect::from_min_size(pos2(card.left() + 14.0 * z + lw + 6.0 * z, y - 9.0 * z), vec2(w, 18.0 * z));
                            let mut child = scaled_child(ui, ("port-default", n, row), wr, z);
                            if m.dim {
                                child.multiply_opacity(DIM_OPACITY);
                            }
                            changed |= model.draw_port_default(&mut child, n, row);
                        }
                    }
                    self.draw_pin(&p, pos2(card.left(), y), z, desc.color, connected, selected, hover == Hit::Pin(pr, Side::Input), linking.as_ref().map(|(s, v)| (*s == Side::Input, v.contains(&pr))));
                }
                if row < m.outputs {
                    let pr = PortRef { node: n, port: row };
                    let desc = model.output(n, row);
                    let connected = self.out_connected.contains(&pr);
                    if z > 0.5 {
                        let g = p.layout_no_wrap(desc.label.to_string(), label_font.clone(), theme::TEXT_SECONDARY);
                        p.galley(pos2(card.right() - 14.0 * z - g.size().x, y - g.size().y * 0.5), g, theme::TEXT_SECONDARY);
                    }
                    self.draw_pin(&p, pos2(card.right(), y), z, desc.color, connected, selected, hover == Hit::Pin(pr, Side::Output), linking.as_ref().map(|(s, v)| (*s == Side::Output, v.contains(&pr))));
                }
            }

            if m.body_h > 0.0 && z > 0.6 {
                let top = card.top() + (HEADER_H + PAD_TOP + ROW_H * rows as f32) * z;
                let br = Rect::from_min_size(pos2(card.left() + 10.0 * z, top), vec2(card.width() - 20.0 * z, (m.body_h - 4.0) * z));
                let mut child = scaled_child(ui, ("body", n), br, z);
                if m.dim {
                    child.multiply_opacity(DIM_OPACITY);
                }
                changed |= model.draw_body(&mut child, n);
            }
        }
        changed
    }

    /// `link_state`: `Some((is_target_side, valid))` while a wire is dragged.
    #[allow(clippy::too_many_arguments)]
    fn draw_pin(&self, p: &egui::Painter, at: Pos2, z: f32, color: Color32, connected: bool, selected: bool, hovered: bool, link_state: Option<(bool, bool)>) {
        let base = if hovered { PIN_HOVER_DIAMETER } else { PIN_DIAMETER };
        let radius = (base * 0.5 * z).max(2.5);
        let mut color = color;
        if let Some((target_side, valid)) = link_state
            && target_side
        {
            if valid {
                p.circle_stroke(at, radius + 3.0, Stroke::new(2.0_f32, theme::with_alpha(theme::accent(), 0.8)));
            } else {
                color = theme::with_alpha(color, 0.3);
            }
        }
        if connected {
            let fill = if selected { theme::accent() } else { color };
            p.circle(at, radius, fill, Stroke::new(1.0_f32, theme::BG_PANEL));
        } else {
            p.circle(at, radius, theme::BG_PANEL, Stroke::new(2.0_f32.min(radius), color));
        }
    }

    fn draw_drag_overlay(&self, painter: &egui::Painter, centre: Pos2, pointer: Option<Pos2>, model: &dyn NodeCanvasModel) {
        match &self.drag {
            Drag::Wire { from, side, valid } => {
                let Some(cursor) = pointer else { return };
                let pin = self.pin_screen(centre, *from, *side);
                // Snap the free end onto a valid target pin under the cursor.
                let target = self
                    .pin_at(centre, cursor, Some(side.opposite()))
                    .filter(|(pr, _)| valid.contains(pr))
                    .map(|(pr, _)| self.pin_screen(centre, pr, side.opposite()));
                let end = target.unwrap_or(cursor);
                let (a, b) = if *side == Side::Output { (pin, end) } else { (end, pin) };
                let color = if *side == Side::Output { model.output(from.node, from.port).color } else { model.input(from.node, from.port).color };
                let c = wire_bezier(a, b, self.view.zoom);
                let mut pts = Vec::new();
                flatten(&c, FLATTEN_TOL, &mut pts);
                painter.extend(Shape::dashed_line(&pts, Stroke::new(WIRE_WIDTH_HOVER, color), 4.0, 4.0));
            }
            Drag::Box { origin, .. } => {
                if let Some(cur) = pointer {
                    let a = self.view.to_screen(centre, *origin);
                    let r = Rect::from_two_pos(a, cur);
                    painter.rect(r, CornerRadius::same(2), theme::with_alpha(theme::accent(), 0.08), Stroke::new(1.0_f32, theme::with_alpha(theme::accent(), 0.7)), StrokeKind::Inside);
                }
            }
            _ => {}
        }
    }
}

/// A child ui over `rect` whose fonts and spacing follow the canvas zoom, for
/// inline widgets (drag fields, swatches) inside a scaled card.
fn scaled_child(ui: &mut Ui, salt: impl std::hash::Hash, rect: Rect, z: f32) -> Ui {
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(salt)
            .max_rect(rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    let style = child.style_mut();
    for font in style.text_styles.values_mut() {
        font.size = (font.size * z * 2.0).round() / 2.0;
    }
    style.spacing.interact_size = vec2(18.0 * z, rect.height());
    style.spacing.item_spacing *= z;
    style.spacing.button_padding *= z;
    child.set_clip_rect(rect.expand(2.0).intersect(ui.clip_rect()));
    child
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32) -> Pos2 {
        pos2(x, y)
    }

    #[test]
    fn control_offset_rule() {
        assert_eq!(wire_ctrl(0.0, 10.0, 1.0), 32.0, "short wires clamp up to 32");
        assert_eq!(wire_ctrl(0.0, 200.0, 1.0), 100.0, "half the distance");
        assert_eq!(wire_ctrl(0.0, 1000.0, 1.0), 200.0, "long wires clamp to 200");
        assert_eq!(wire_ctrl(500.0, 0.0, 1.0), 200.0, "backwards wires use |dx|");
        assert_eq!(wire_ctrl(0.0, 200.0, 2.0), 100.0, "half the screen distance");
        assert_eq!(wire_ctrl(0.0, 10.0, 2.0), 64.0, "clamp scales with zoom");
        let c = wire_bezier(p(0.0, 0.0), p(200.0, 50.0), 1.0);
        assert_eq!(c[1], p(100.0, 0.0), "no vertical offset out of the output");
        assert_eq!(c[2], p(100.0, 50.0), "no vertical offset into the input");
    }

    #[test]
    fn flatten_matches_analytic_points() {
        let c = wire_bezier(p(0.0, 0.0), p(300.0, 120.0), 1.0);
        let mut pts = Vec::new();
        flatten(&c, 0.25, &mut pts);
        assert_eq!(pts.first(), Some(&c[0]));
        assert_eq!(pts.last(), Some(&c[3]));
        // Every flattened vertex lies on the curve; the chord between two of
        // them stays within the tolerance of the analytic midpoint.
        let n = pts.len() - 1;
        for i in 0..n {
            let mid = bezier_at(&c, (i as f32 + 0.5) / n as f32);
            assert!(dist_point_segment(mid, pts[i], pts[i + 1]) <= 0.25 + 1e-3, "segment {i} off by more than tol");
        }
        // The analytic midpoint of a symmetric S-curve is its centre.
        let m = bezier_at(&c, 0.5);
        assert!((m - p(150.0, 60.0)).length() < 1e-3);
    }

    #[test]
    fn segment_count_adapts() {
        let short = wire_bezier(p(0.0, 0.0), p(20.0, 0.0), 1.0);
        let long = wire_bezier(p(0.0, 0.0), p(800.0, 600.0), 1.0);
        assert!(segment_count(&short, 0.4) <= segment_count(&long, 0.4));
        assert!(segment_count(&long, 0.4) <= 128);
        assert!(segment_count(&long, 0.1) > segment_count(&long, 1.0), "tighter tolerance, more segments");
    }

    #[test]
    fn hit_band_is_eight_px() {
        let c = wire_bezier(p(0.0, 0.0), p(200.0, 0.0), 1.0);
        // Straight horizontal wire: distance is the vertical offset.
        assert!(dist_to_bezier(p(100.0, 3.0), &c, 0.25) <= WIRE_BAND);
        assert!(dist_to_bezier(p(100.0, 6.0), &c, 0.25) > WIRE_BAND);
        assert!((dist_to_bezier(p(100.0, 3.0), &c, 0.25) - 3.0).abs() < 0.05);
        // Past the end it is the distance to the pin.
        assert!((dist_to_bezier(p(-10.0, 0.0), &c, 0.25) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn bezier_vs_rect() {
        let c = wire_bezier(p(0.0, 0.0), p(300.0, 100.0), 1.0);
        let on_curve = bezier_at(&c, 0.5);
        assert!(bezier_intersects_rect(&c, Rect::from_center_size(on_curve, vec2(6.0, 6.0)), 0.25));
        // A box fully inside the curve's bbox but off the curve misses it.
        assert!(!bezier_intersects_rect(&c, Rect::from_min_size(p(10.0, 70.0), vec2(30.0, 20.0)), 0.25));
        // A thin box crossing between samples is still caught (segments, not points).
        let thin = Rect::from_min_max(p(on_curve.x - 0.01, -50.0), p(on_curve.x + 0.01, 150.0));
        assert!(bezier_intersects_rect(&c, thin, 0.25));
        // Far away.
        assert!(!bezier_intersects_rect(&c, Rect::from_min_size(p(1000.0, 1000.0), vec2(10.0, 10.0)), 0.25));
        // Box containing the whole curve.
        assert!(bezier_intersects_rect(&c, Rect::from_min_max(p(-50.0, -50.0), p(400.0, 200.0)), 0.25));
    }

    #[test]
    fn segment_rect_cases() {
        let r = Rect::from_min_max(p(0.0, 0.0), p(10.0, 10.0));
        assert!(segment_intersects_rect(p(-5.0, 5.0), p(15.0, 5.0), r), "passes through");
        assert!(segment_intersects_rect(p(2.0, 2.0), p(3.0, 3.0), r), "inside");
        assert!(!segment_intersects_rect(p(-5.0, 11.0), p(15.0, 11.0), r), "parallel miss");
        assert!(!segment_intersects_rect(p(11.0, -5.0), p(11.0, 15.0), r));
    }

    #[test]
    fn pins_sit_on_the_card_edge_at_any_zoom() {
        let view0 = View::default();
        let centre = p(400.0, 300.0);
        let size = card_size(3, 0.0);
        for zoom in [0.3, 0.75, 1.0, 1.6, 2.5] {
            let view = View { pan: vec2(37.0, -12.0), zoom };
            let card = Rect::from_center_size(p(50.0, 20.0), size);
            let screen = view.rect_to_screen(centre, card);
            for row in 0..3 {
                let a = view.to_screen(centre, pin_pos(card, Side::Input, row));
                let b = view.to_screen(centre, pin_pos(card, Side::Output, row));
                assert!((a.x - screen.left()).abs() < 1e-3, "input pin on left edge at zoom {zoom}");
                assert!((b.x - screen.right()).abs() < 1e-3, "output pin on right edge at zoom {zoom}");
                let want = screen.top() + row_center_y(row) * zoom;
                assert!((a.y - want).abs() < 1e-3 && (b.y - want).abs() < 1e-3);
            }
        }
        assert_eq!(view0.zoom, 1.0);
    }

    #[test]
    fn card_height_follows_the_design_bible() {
        // 34 header + 4 + 22*rows + 8.
        assert_eq!(card_size(2, 0.0).y, 34.0 + 4.0 + 44.0 + 8.0);
        assert_eq!(card_size(1, 30.0).x, CARD_WIDTH);
    }

    #[test]
    fn snapping() {
        assert_eq!(snap_to_grid(11.9, 24.0), 0.0);
        assert_eq!(snap_to_grid(12.1, 24.0), 24.0);
        assert_eq!(snap_to_grid(-13.0, 24.0), -24.0);
        assert_eq!(snap_pos(p(25.0, 47.0), 24.0), p(24.0, 48.0));
    }

    #[test]
    fn zoom_is_anchored_at_the_cursor() {
        let centre = p(500.0, 400.0);
        let cursor = p(620.0, 310.0);
        for factor in [0.5, 0.9, 1.1, 2.0] {
            let mut v = View { pan: vec2(-40.0, 25.0), zoom: 1.0 };
            let before = v.to_canvas(centre, cursor);
            v.zoom_at(centre, cursor, factor);
            let after = v.to_canvas(centre, cursor);
            assert!((before - after).length() < 1e-3, "canvas point under cursor moved for factor {factor}");
            assert!((v.zoom - factor).abs() < 1e-6);
        }
        // Clamped zoom still keeps the anchor.
        let mut v = View::default();
        let before = v.to_canvas(centre, cursor);
        v.zoom_at(centre, cursor, 100.0);
        assert_eq!(v.zoom, ZOOM_MAX);
        assert!((v.to_canvas(centre, cursor) - before).length() < 1e-3);
    }

    #[test]
    fn screen_canvas_round_trip() {
        let v = View { pan: vec2(10.0, -4.0), zoom: 1.7 };
        let c = p(300.0, 200.0);
        let q = p(-55.0, 91.0);
        assert!((v.to_canvas(c, v.to_screen(c, q)) - q).length() < 1e-3);
    }

    #[test]
    fn fit_view_contains_bounds() {
        let b = Rect::from_min_max(p(-300.0, -100.0), p(500.0, 300.0));
        let vp = vec2(800.0, 600.0);
        let v = fit_view(b, vp, 40.0);
        let centre = p(400.0, 300.0);
        let s = v.rect_to_screen(centre, b);
        assert!(s.left() >= 0.0 && s.right() <= 800.0 && s.top() >= 0.0 && s.bottom() <= 600.0);
        assert!((s.center() - centre).length() < 1e-2);
    }

    #[test]
    fn pick_priority_is_pin_node_wire() {
        let pr = PortRef { node: 1, port: 0 };
        assert_eq!(prioritise(Some((pr, Side::Input)), Some(1), Some(0)), Hit::Pin(pr, Side::Input));
        assert_eq!(prioritise(None, Some(1), Some(0)), Hit::Node(1));
        assert_eq!(prioritise(None, None, Some(0)), Hit::Wire(0));
        assert_eq!(prioritise(None, None, None), Hit::None);
    }

    // A two-node model to exercise the stateful picking path.
    struct Two {
        pos: [Pos2; 2],
        linked: bool,
    }

    impl NodeCanvasModel for Two {
        fn node_count(&self) -> usize {
            2
        }
        fn position(&self, n: usize) -> Pos2 {
            self.pos[n]
        }
        fn set_position(&mut self, n: usize, p: Pos2) {
            self.pos[n] = p;
        }
        fn style(&self, _n: usize) -> NodeStyle {
            NodeStyle { title: "T".into(), chip: Color32::WHITE, glyph: "" }
        }
        fn input_count(&self, n: usize) -> usize {
            (n == 1) as usize
        }
        fn output_count(&self, n: usize) -> usize {
            (n == 0) as usize
        }
        fn input(&self, _n: usize, _p: usize) -> PortDesc {
            PortDesc { label: "in".into(), type_label: "Float", color: Color32::WHITE }
        }
        fn output(&self, _n: usize, _p: usize) -> PortDesc {
            PortDesc { label: "out".into(), type_label: "Float", color: Color32::WHITE }
        }
        fn links(&self, out: &mut Vec<Link>) {
            if self.linked {
                out.push(Link { from: PortRef { node: 0, port: 0 }, to: PortRef { node: 1, port: 0 }, color: Color32::WHITE });
            }
        }
        fn can_connect(&self, from: PortRef, to: PortRef) -> bool {
            from.node != to.node
        }
        fn connect(&mut self, _f: PortRef, _t: PortRef) -> bool {
            self.linked = true;
            true
        }
        fn delete_node(&mut self, _n: usize) {}
    }

    fn two() -> (NodeCanvasState, Two, Pos2) {
        let model = Two { pos: [p(0.0, 0.0), p(400.0, 0.0)], linked: true };
        let mut st = NodeCanvasState::default();
        st.rebuild(&model);
        (st, model, p(500.0, 300.0))
    }

    #[test]
    fn stateful_pick_priority() {
        let (st, _m, c) = two();
        let out_pin = st.pin_screen(c, PortRef { node: 0, port: 0 }, Side::Output);
        // On the pin (which sits on the card edge and the wire's start): pin wins.
        assert!(matches!(st.hit_at(c, out_pin), Hit::Pin(..)));
        // Inside the card but away from pins: node wins over any wire.
        assert_eq!(st.hit_at(c, st.screen_card(c, 0).center()), Hit::Node(0));
        // On the wire between the cards (middle of the gap).
        let curve = st.link_curve(c, &st.links[0]);
        let mid = bezier_at(&curve, 0.5);
        assert_eq!(st.hit_at(c, mid + vec2(0.0, 2.0)), Hit::Wire(0));
        assert_eq!(st.hit_at(c, mid + vec2(0.0, 30.0)), Hit::None);
    }

    #[test]
    fn box_select_picks_wires_and_nodes() {
        let (mut st, _m, c) = two();
        let curve = st.link_curve(c, &st.links[0]);
        let mid = bezier_at(&curve, 0.5);
        // A box over the gap touches the wire but neither card.
        let b = Rect::from_center_size(mid, vec2(10.0, 10.0));
        st.select_in_box(c, b, &BTreeSet::new(), &[], false);
        assert!(st.selected.is_empty());
        assert_eq!(st.selected_links.len(), 1);
        // A box over the first card selects it and (the wire starts at its edge) the wire.
        let b = st.screen_card(c, 0).shrink(4.0);
        st.select_in_box(c, b, &BTreeSet::new(), &[], false);
        assert!(st.selected.contains(&0) && !st.selected.contains(&1));
        // Far away selects nothing; additive keeps the base.
        let far = Rect::from_min_size(p(5000.0, 5000.0), vec2(10.0, 10.0));
        let base: BTreeSet<usize> = [1].into();
        st.select_in_box(c, far, &base, &[], true);
        assert_eq!(st.selected.iter().copied().collect::<Vec<_>>(), vec![1]);
        st.select_in_box(c, far, &base, &[], false);
        assert!(st.selected.is_empty() && st.selected_links.is_empty());
    }

    #[test]
    fn selected_nodes_pick_on_top() {
        let model = Two { pos: [p(0.0, 0.0), p(10.0, 0.0)], linked: false };
        let mut st = NodeCanvasState::default();
        st.selected.insert(0);
        st.rebuild(&model);
        let c = p(500.0, 300.0);
        // The cards overlap; selected node 0 draws and picks above node 1.
        let overlap = st.screen_card(c, 0).center() + vec2(20.0, 20.0);
        assert_eq!(st.hit_at(c, overlap), Hit::Node(0));
    }

    #[test]
    fn link_identity_ignores_colour() {
        let a = Link { from: PortRef { node: 0, port: 0 }, to: PortRef { node: 1, port: 0 }, color: Color32::RED };
        let b = Link { color: Color32::BLUE, ..a };
        assert_eq!(a, b);
    }
}
