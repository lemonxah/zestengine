//! Editor handles for UI elements: click to select, drag to move, drag the
//! eight handles to resize. Edits the element's `UiRect` in reference pixels,
//! so the result matches what the game lays out at any resolution. Shared by
//! the viewport's game-UI overlay and the UI Editor tab.

use zestengine_core::{UiCanvas, UiRect};
use egui::{Color32, CornerRadius, Pos2, Rect, Stroke, StrokeKind, Vec2};

use zestengine_engine::scene::LoadedScene;

const HANDLE: f32 = 7.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Grab {
    Move,
    /// Edge / corner: -1 = left/top, 1 = right/bottom, 0 = untouched axis.
    Resize(i8, i8),
}

#[derive(Clone, Copy)]
struct Drag {
    index: usize,
    grab: Grab,
    origin: Pos2,
    start: UiRect,
    scale: f32,
}

/// What a frame of [`UiRectEditor::interact`] did.
#[derive(Default)]
pub struct UiEditOutcome {
    /// An element was clicked (select it). `None` inside means empty space.
    pub clicked: Option<Option<usize>>,
    /// A move / resize drag is in progress (callers suppress their own drags).
    pub dragging: bool,
    /// The pointer is over a handle or the selected element.
    pub hovering: bool,
    pub changed: bool,
}

#[derive(Default)]
pub struct UiRectEditor {
    drag: Option<Drag>,
    hover: Option<usize>,
    /// Snap step in reference pixels (0 = off).
    pub snap: f32,
}

fn has<T: 'static>(scene: &LoadedScene, i: usize) -> bool {
    scene.objects[i].components.iter().any(|c| c.as_any().is::<T>())
}

fn rect_of(scene: &mut LoadedScene, i: usize) -> Option<&mut UiRect> {
    scene.objects[i]
        .components
        .iter_mut()
        .find_map(|c| c.as_any_mut().downcast_mut::<UiRect>())
}

/// Topmost non-canvas element under a point.
fn pick(scene: &LoadedScene, nodes: &[(usize, Rect, f32)], p: Pos2) -> Option<usize> {
    nodes
        .iter()
        .rev()
        .find(|(i, r, _)| r.contains(p) && !has::<UiCanvas>(scene, *i))
        .map(|n| n.0)
}

fn handles(r: Rect) -> [(Grab, Pos2); 8] {
    let (l, c, rt) = (r.left(), r.center().x, r.right());
    let (t, m, b) = (r.top(), r.center().y, r.bottom());
    [
        (Grab::Resize(-1, -1), Pos2::new(l, t)),
        (Grab::Resize(0, -1), Pos2::new(c, t)),
        (Grab::Resize(1, -1), Pos2::new(rt, t)),
        (Grab::Resize(-1, 0), Pos2::new(l, m)),
        (Grab::Resize(1, 0), Pos2::new(rt, m)),
        (Grab::Resize(-1, 1), Pos2::new(l, b)),
        (Grab::Resize(0, 1), Pos2::new(c, b)),
        (Grab::Resize(1, 1), Pos2::new(rt, b)),
    ]
}

fn cursor_for(grab: Grab) -> egui::CursorIcon {
    match grab {
        Grab::Move => egui::CursorIcon::Move,
        Grab::Resize(0, _) => egui::CursorIcon::ResizeVertical,
        Grab::Resize(_, 0) => egui::CursorIcon::ResizeHorizontal,
        Grab::Resize(x, y) if x == y => egui::CursorIcon::ResizeNwSe,
        Grab::Resize(..) => egui::CursorIcon::ResizeNeSw,
    }
}

/// New rect after dragging `grab` by `d` reference pixels. Resizing keeps the
/// opposite edge fixed whatever the pivot or anchors.
fn apply(start: UiRect, grab: Grab, d: Vec2, snap: f32) -> UiRect {
    let snap_to = |v: f32| if snap > 0.0 { (v / snap).round() * snap } else { v.round() };
    let mut r = start;
    let d = [d.x, d.y];
    match grab {
        Grab::Move => {
            for (a, p) in r.position.iter_mut().enumerate() {
                *p = snap_to(start.position[a] + d[a]);
            }
        }
        Grab::Resize(x, y) => {
            for (a, side) in [(0, x), (1, y)] {
                let delta = match side {
                    1 => snap_to(start.size[a] + d[a]) - start.size[a],
                    -1 => start.size[a] - snap_to(start.size[a] - d[a]),
                    _ => continue,
                };
                if side == 1 {
                    r.size[a] = start.size[a] + delta;
                    r.position[a] = start.position[a] + delta * start.pivot[a];
                } else {
                    r.size[a] = start.size[a] - delta;
                    r.position[a] = start.position[a] + delta * (1.0 - start.pivot[a]);
                }
            }
        }
    }
    r
}

impl UiRectEditor {
    /// Handle the pointer over `area` for the laid-out `nodes` (from
    /// `game_ui::layout_rects`). `drag_unselected` lets a drag on any element
    /// select and move it (the UI Editor); the viewport only moves the
    /// selection so other drags still orbit the camera.
    #[allow(clippy::too_many_arguments)]
    pub fn interact(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        scene: &mut LoadedScene,
        nodes: &[(usize, Rect, f32)],
        selected: Option<usize>,
        area: Rect,
        drag_unselected: bool,
    ) -> UiEditOutcome {
        let mut out = UiEditOutcome::default();
        let pointer = response.hover_pos().or_else(|| response.interact_pointer_pos());
        let node = |i: usize| nodes.iter().find(|n| n.0 == i).copied();
        let editable = selected
            .and_then(node)
            .filter(|(i, _, _)| has::<UiRect>(scene, *i) && !has::<UiCanvas>(scene, *i));

        // What the pointer would grab.
        let grab_at = |p: Pos2| -> Option<Grab> {
            let (_, r, _) = editable?;
            if let Some((g, _)) = handles(r).into_iter().find(|(_, h)| h.distance(p) <= HANDLE + 2.0) {
                return Some(g);
            }
            r.contains(p).then_some(Grab::Move)
        };

        self.hover = None;
        if let Some(p) = pointer.filter(|p| area.contains(*p)) {
            if let Some(g) = self.drag.map(|d| d.grab).or_else(|| grab_at(p)) {
                ui.ctx().set_cursor_icon(cursor_for(g));
                out.hovering = true;
            }
            if self.drag.is_none() {
                self.hover = pick(scene, nodes, p).filter(|h| Some(*h) != selected);
            }
        }

        // A drag that selects a new element starts moving it next frame, so the
        // editor's undo snapshot (taken per selected object) sees the whole move.
        let mut fresh = false;
        if response.drag_started_by(egui::PointerButton::Primary) {
            let origin = ui.input(|i| i.pointer.press_origin()).or(pointer);
            if let Some(origin) = origin.filter(|p| area.contains(*p)) {
                let target = match grab_at(origin) {
                    Some(g) => editable.map(|(i, _, s)| (i, g, s)),
                    None if drag_unselected => pick(scene, nodes, origin).and_then(node).and_then(|(i, _, s)| {
                        out.clicked = Some(Some(i));
                        fresh = Some(i) != selected;
                        has::<UiRect>(scene, i).then_some((i, Grab::Move, s))
                    }),
                    None => None,
                };
                if let Some((index, grab, scale)) = target
                    && let Some(start) = rect_of(scene, index).map(|r| *r)
                {
                    self.drag = Some(Drag { index, grab, origin, start, scale });
                }
            }
        }
        if let Some(d) = self.drag {
            out.dragging = true;
            if let Some(p) = pointer.filter(|_| !fresh) {
                let delta = (p - d.origin) / d.scale.max(1e-4);
                let snap = if ui.input(|i| i.modifiers.ctrl) && self.snap <= 0.0 { 10.0 } else { self.snap };
                let next = apply(d.start, d.grab, delta, snap);
                if let Some(r) = rect_of(scene, d.index) {
                    *r = next;
                    out.changed = true;
                }
            }
            if !ui.input(|i| i.pointer.primary_down()) {
                self.drag = None;
            }
        }

        if response.clicked_by(egui::PointerButton::Primary)
            && let Some(p) = pointer.filter(|p| area.contains(*p))
        {
            out.clicked = Some(pick(scene, nodes, p));
        }
        out
    }

    /// Hover outline, selection outline, anchor markers and resize handles.
    pub fn paint(&self, painter: &egui::Painter, scene: &LoadedScene, nodes: &[(usize, Rect, f32)], selected: Option<usize>) {
        let node = |i: usize| nodes.iter().find(|n| n.0 == i).copied();
        if let Some((_, r, _)) = self.hover.and_then(node) {
            painter.rect_stroke(r, CornerRadius::ZERO, Stroke::new(1.0_f32, crate::theme::accent_soft()), StrokeKind::Outside);
        }
        let Some((i, r, _)) = selected.and_then(node) else {
            return;
        };
        painter.rect_stroke(r, CornerRadius::ZERO, Stroke::new(1.5_f32, crate::theme::accent()), StrokeKind::Outside);
        if !has::<UiRect>(scene, i) || has::<UiCanvas>(scene, i) {
            return;
        }
        let rect = scene.objects[i]
            .components
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<UiRect>())
            .copied()
            .unwrap_or_default();
        if let Some((_, pr, _)) = scene.objects[i].parent.and_then(node) {
            for (ax, ay) in [
                (rect.anchor_min[0], rect.anchor_min[1]),
                (rect.anchor_max[0], rect.anchor_min[1]),
                (rect.anchor_min[0], rect.anchor_max[1]),
                (rect.anchor_max[0], rect.anchor_max[1]),
            ] {
                let p = Pos2::new(pr.left() + ax * pr.width(), pr.top() + ay * pr.height());
                painter.add(egui::Shape::convex_polygon(
                    vec![p, p + Vec2::new(-5.0, -9.0), p + Vec2::new(5.0, -9.0)],
                    Color32::WHITE,
                    Stroke::new(1.0_f32, Color32::BLACK),
                ));
            }
        }
        for (_, h) in handles(r) {
            let sq = Rect::from_center_size(h, Vec2::splat(HANDLE));
            painter.rect(sq, CornerRadius::ZERO, Color32::WHITE, Stroke::new(1.0_f32, crate::theme::accent()), StrokeKind::Inside);
        }
    }

    /// Arrow keys nudge the selected element by 1 reference pixel (10 with
    /// Shift). True when something moved.
    pub fn nudge(ui: &egui::Ui, scene: &mut LoadedScene, selected: usize) -> bool {
        let (d, shift) = ui.input(|i| {
            let mut d = Vec2::ZERO;
            for (key, v) in [
                (egui::Key::ArrowLeft, Vec2::new(-1.0, 0.0)),
                (egui::Key::ArrowRight, Vec2::new(1.0, 0.0)),
                (egui::Key::ArrowUp, Vec2::new(0.0, -1.0)),
                (egui::Key::ArrowDown, Vec2::new(0.0, 1.0)),
            ] {
                if i.key_pressed(key) {
                    d += v;
                }
            }
            (d, i.modifiers.shift)
        });
        if d == Vec2::ZERO || has::<UiCanvas>(scene, selected) {
            return false;
        }
        let step = if shift { 10.0 } else { 1.0 };
        match rect_of(scene, selected) {
            Some(r) => {
                r.position[0] += d.x * step;
                r.position[1] += d.y * step;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(r: &UiRect) -> ([f32; 2], [f32; 2]) {
        r.resolve([0.0, 0.0], [1000.0, 800.0], 1.0)
    }

    #[test]
    fn resize_keeps_the_opposite_edge_for_any_pivot() {
        for pivot in [[0.0, 0.0], [0.5, 0.5], [1.0, 0.25]] {
            let start = UiRect { pivot, size: [200.0, 100.0], ..Default::default() };
            let (min0, size0) = resolved(&start);
            let r = apply(start, Grab::Resize(1, 1), Vec2::new(30.0, -20.0), 0.0);
            let (min, size) = resolved(&r);
            assert!((min[0] - min0[0]).abs() < 1e-3 && (min[1] - min0[1]).abs() < 1e-3, "{pivot:?}");
            assert_eq!(size, [size0[0] + 30.0, size0[1] - 20.0]);

            let r = apply(start, Grab::Resize(-1, 0), Vec2::new(40.0, 0.0), 0.0);
            let (min, size) = resolved(&r);
            assert!((min[0] + size[0] - (min0[0] + size0[0])).abs() < 1e-3, "{pivot:?}");
            assert_eq!(size[0], size0[0] - 40.0);
        }
    }

    #[test]
    fn move_snaps_to_the_step() {
        let r = apply(UiRect::default(), Grab::Move, Vec2::new(13.0, 26.0), 10.0);
        assert_eq!(r.position, [10.0, 30.0]);
    }
}
