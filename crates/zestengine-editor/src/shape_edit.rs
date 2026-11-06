//! Scene-view spline editing for Sprite Shapes (2D-8), in the style of
//! `river_edit.rs` / `tile_edit.rs`. Active only while the selected object has a
//! `SpriteShapeRenderer`, the inspector's Edit Spline toggle is on and the editor
//! is not playing. While active, left drags edit the spline instead of orbiting
//! or moving the transform gizmo (`active()` is checked where the other paint
//! tools are).
//!
//! * Click on empty space: add a point (open shapes append, Ctrl prepends;
//!   closed shapes insert after the nearest segment).
//! * Click on the line: split the segment there (the curve keeps its shape);
//!   Shift+Click splits at the nearest point of the line wherever you click.
//! * Drag a point or, for the selected point, a tangent handle. Snapping uses the
//!   inspector's grid step. Dragging a handle of a Linear point makes it smooth.
//! * `1` / `2` / `3` set the selected point Linear / Continuous / Broken, `C`
//!   toggles its corner flag, Delete removes it (a spline keeps two points).
//!
//! Edits go through the pure operations of `zestengine_core::sprite_shape::edit`
//! (each has an exact inverse, tested there) and land on the scene during the UI
//! pass, so the frame's component diff records them as undo entries; a drag is one
//! entry because the undo stack coalesces while the pointer is down.

use egui::Pos2;
use glam::{Mat4, Vec2, Vec3};
use zestengine_core::sprite_shape::edit::{self, SplineOp};
use zestengine_core::sprite_shape::spline::{bezier_at, segment_bezier, tessellate};
use zestengine_core::{ShapePoint, ShapeTangentMode, SpriteShapeRenderer};
use crate::{SHAPE_HINT, ShapeEditState, theme};

use crate::cloth_paint::{View, hint_bar};
use zestengine_engine::scene::LoadedScene;

/// Screen distance (px) within which a press takes a handle.
const HANDLE_PX: f32 = 11.0;
/// ...and within which it counts as on the line.
const LINE_PX: f32 = 7.0;

fn shape(scene: &LoadedScene, i: usize) -> Option<&SpriteShapeRenderer> {
    scene.objects.get(i)?.components.iter().find_map(|c| c.as_any().downcast_ref::<SpriteShapeRenderer>())
}

fn shape_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut SpriteShapeRenderer> {
    scene.objects.get_mut(i)?.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<SpriteShapeRenderer>())
}

/// Keep the shared state on the selected object: selecting something else, a
/// missing shape or Play ends spline editing.
pub(crate) fn sync(ctx: &egui::Context, scene: &LoadedScene, selected: Option<usize>, playing: bool) {
    let mut st = ShapeEditState::get(ctx);
    let before = st;
    let id = selected.and_then(|i| scene.objects.get(i)).map(|o| o.id);
    if st.object != id {
        st = ShapeEditState { object: id, snap: st.snap, show_tessellation: st.show_tessellation, ..Default::default() };
    }
    let points = selected.and_then(|i| shape(scene, i)).map(|s| s.points.len());
    if points.is_none() || playing {
        st.active = false;
    }
    st.selected = st.selected.filter(|&k| points.is_some_and(|n| k < n));
    if st != before {
        st.set(ctx);
    }
}

/// The selected shape's spline is being edited.
pub(crate) fn active(ctx: &egui::Context) -> bool {
    ShapeEditState::get(ctx).active
}

/// Leave spline editing; false when it wasn't on.
pub(crate) fn exit(ctx: &egui::Context) -> bool {
    let mut st = ShapeEditState::get(ctx);
    if !st.active {
        return false;
    }
    st.active = false;
    st.set(ctx);
    true
}

/// What a press can grab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Handle {
    Point(usize),
    TangentIn(usize),
    TangentOut(usize),
}

impl Handle {
    pub fn point(self) -> usize {
        match self {
            Handle::Point(k) | Handle::TangentIn(k) | Handle::TangentOut(k) => k,
        }
    }
}

/// Handles with their local positions: every point, plus the tangent handles of
/// the selected point when it is not Linear.
pub fn handles(s: &SpriteShapeRenderer, selected: Option<usize>) -> Vec<(Handle, Vec2)> {
    let mut out: Vec<(Handle, Vec2)> = s.points.iter().enumerate().map(|(k, p)| (Handle::Point(k), Vec2::from(p.position))).collect();
    if let Some(k) = selected
        && let Some(p) = s.points.get(k)
        && p.mode != ShapeTangentMode::Linear
    {
        out.push((Handle::TangentIn(k), Vec2::from(p.position) + Vec2::from(p.tangent_in)));
        out.push((Handle::TangentOut(k), Vec2::from(p.position) + Vec2::from(p.tangent_out)));
    }
    out
}

/// The handle nearest `cursor` within `radius` px; tangent handles win ties
/// (they sit on top of their point when short).
pub fn hit_handle(screen: &[(Handle, Pos2)], cursor: Pos2, radius: f32) -> Option<Handle> {
    let mut best: Option<(f32, bool, Handle)> = None;
    for &(h, p) in screen {
        let d = p.distance(cursor);
        if d > radius {
            continue;
        }
        let tangent = !matches!(h, Handle::Point(_));
        if best.is_none_or(|(bd, bt, _)| d < bd - 0.5 || (d <= bd + 0.5 && tangent && !bt)) {
            best = Some((d, tangent, h));
        }
    }
    best.map(|b| b.2)
}

/// The operation a drag of `handle` to the local position `to` applies.
pub fn drag_op(s: &SpriteShapeRenderer, handle: Handle, to: Vec2, snap: f32) -> Option<SplineOp> {
    let to = edit::snap(to.to_array(), snap);
    match handle {
        Handle::Point(k) => edit::move_point(s, k, to),
        Handle::TangentOut(k) => edit::move_tangent(s, k, true, to),
        Handle::TangentIn(k) => edit::move_tangent(s, k, false, to),
    }
}

/// What a click does at local position `local`: the operation and the index of
/// the point to select afterwards. `on_line` is the (segment, t) the click hit
/// on the spline, if any.
pub fn click_op(s: &SpriteShapeRenderer, local: Vec2, on_line: Option<(usize, f32)>, shift: bool, ctrl: bool, snap: f32) -> Option<(SplineOp, usize)> {
    let n = s.points.len();
    let at = edit::snap(local.to_array(), snap);
    let landing = |seg: usize| if (seg + 1).is_multiple_of(n.max(1)) { n } else { seg + 1 };
    if n >= 2 {
        let line = on_line.or_else(|| shift.then(|| edit::nearest_on_spline(s, local).map(|(seg, t, _)| (seg, t))).flatten());
        if let Some((seg, t)) = line {
            return edit::split_segment(s, seg, t).map(|op| (op, landing(seg)));
        }
    }
    let mut p = ShapePoint::at(at[0], at[1]);
    if s.closed && n >= 2 {
        let (seg, _, _) = edit::nearest_on_spline(s, local)?;
        p.height = s.points[seg].height;
        let index = landing(seg);
        return Some((SplineOp::Insert { index, point: p }, index));
    }
    if ctrl && n > 0 {
        p.height = s.points[0].height;
        return Some((SplineOp::Insert { index: 0, point: p }, 0));
    }
    if let Some(last) = s.points.last() {
        p.height = last.height;
    }
    Some((SplineOp::Insert { index: n, point: p }, n))
}

fn plane_hit(view: &View, cursor: Pos2, world_inv: Mat4) -> Option<Vec2> {
    let (o, d) = view.ray(cursor)?;
    let (lo, ld) = (world_inv.transform_point3(o), world_inv.transform_vector3(d));
    if ld.z.abs() < 1e-8 {
        return None;
    }
    let p = lo + ld * (-lo.z / ld.z);
    Some(Vec2::new(p.x, p.y))
}

fn to_screen(view: &View, world: Mat4, local: Vec2) -> Option<Pos2> {
    view.project(world.transform_point3(Vec3::new(local.x, local.y, 0.0)))
}

/// The drag in flight, in egui memory.
fn drag_id() -> egui::Id {
    egui::Id::new("zestengine-shape-drag")
}

fn get_drag(ctx: &egui::Context) -> Option<Handle> {
    ctx.data(|d| d.get_temp::<Option<(u8, usize)>>(drag_id())).flatten().map(|(k, i)| match k {
        0 => Handle::Point(i),
        1 => Handle::TangentIn(i),
        _ => Handle::TangentOut(i),
    })
}

fn set_drag(ctx: &egui::Context, h: Option<Handle>) {
    let v = h.map(|h| match h {
        Handle::Point(i) => (0u8, i),
        Handle::TangentIn(i) => (1, i),
        Handle::TangentOut(i) => (2, i),
    });
    ctx.data_mut(|d| d.insert_temp::<Option<(u8, usize)>>(drag_id(), v));
}

/// Scene-view input and overlay for the selected shape (object `i`). Returns
/// true when the press / click was used (so it does not also pick objects).
pub(crate) fn viewport(ui: &mut egui::Ui, scene: &mut LoadedScene, i: usize, view: &View, response: &egui::Response) -> bool {
    let ctx = ui.ctx().clone();
    let mut st = ShapeEditState::get(&ctx);
    let Some(id) = scene.objects.get(i).map(|o| o.id) else { return false };
    if shape(scene, i).is_none() || st.object != Some(id) || !(st.active || st.show_tessellation) {
        return false;
    }
    let before = st;
    let world = scene.world_transform(i);
    let inv = world.inverse();
    let cursor = response.hover_pos().or_else(|| response.interact_pointer_pos());
    let (alt, shift, ctrl) = ui.input(|inp| (inp.modifiers.alt, inp.modifiers.shift, inp.modifiers.command));
    let (pressed, down, released) = ui.input(|inp| (inp.pointer.primary_pressed(), inp.pointer.primary_down(), inp.pointer.primary_released()));
    let mut used = false;

    if st.active {
        // Press: grab a handle, split the line or add a point.
        if pressed && response.hovered() && !alt && let Some(cur) = cursor {
            let s = shape(scene, i).expect("checked");
            let screen: Vec<(Handle, Pos2)> = handles(s, st.selected).into_iter().filter_map(|(h, p)| to_screen(view, world, p).map(|q| (h, q))).collect();
            if let Some(h) = hit_handle(&screen, cur, HANDLE_PX) {
                st.selected = Some(h.point());
                st.reveal = true;
                set_drag(&ctx, Some(h));
                used = true;
            } else if let Some(local) = plane_hit(view, cur, inv) {
                let on_line = edit::nearest_on_spline(s, local).and_then(|(seg, t, _)| {
                    let c = segment_bezier(&s.points[seg], &s.points[(seg + 1) % s.points.len()]);
                    let screen_pt = to_screen(view, world, bezier_at(&c, t))?;
                    (screen_pt.distance(cur) <= LINE_PX).then_some((seg, t))
                });
                if let Some((op, select)) = click_op(s, local, on_line, shift, ctrl, st.snap) {
                    if let Some(s) = shape_mut(scene, i) {
                        edit::apply_op(s, &op);
                    }
                    st.selected = Some(select);
                    st.reveal = true;
                    set_drag(&ctx, Some(Handle::Point(select)));
                }
                used = true;
            }
        }
        // Drag in flight.
        if let Some(h) = get_drag(&ctx) {
            if down && let Some(cur) = cursor {
                if let Some(local) = plane_hit(view, cur, inv)
                    && let Some(op) = shape(scene, i).and_then(|s| drag_op(s, h, local, st.snap))
                    && let Some(s) = shape_mut(scene, i)
                {
                    edit::apply_op(s, &op);
                }
                used = true;
            }
            if released || !down {
                set_drag(&ctx, None);
            }
        }
        // Keys.
        if response.hovered() && !ctx.wants_keyboard_input()
            && let Some(k) = st.selected
        {
            let (del, m1, m2, m3, corner) = ui.input(|inp| {
                (
                    inp.key_pressed(egui::Key::Delete) || inp.key_pressed(egui::Key::Backspace),
                    inp.key_pressed(egui::Key::Num1),
                    inp.key_pressed(egui::Key::Num2),
                    inp.key_pressed(egui::Key::Num3),
                    inp.key_pressed(egui::Key::C) && !inp.modifiers.command,
                )
            });
            let op = shape(scene, i).and_then(|s| {
                if del && s.points.len() > 2 {
                    Some(SplineOp::Remove { index: k })
                } else if m1 {
                    edit::set_mode(s, k, ShapeTangentMode::Linear)
                } else if m2 {
                    edit::set_mode(s, k, ShapeTangentMode::Continuous)
                } else if m3 {
                    edit::set_mode(s, k, ShapeTangentMode::Broken)
                } else if corner {
                    edit::toggle_corner(s, k)
                } else {
                    None
                }
            });
            if let Some(op) = op
                && let Some(s) = shape_mut(scene, i)
            {
                let removed = matches!(op, SplineOp::Remove { .. });
                edit::apply_op(s, &op);
                if removed {
                    st.selected = Some(k.saturating_sub(1).min(s.points.len().saturating_sub(1)));
                    st.reveal = true;
                }
            }
        }
    }

    draw(ui, scene, i, view, &st, cursor.filter(|_| st.active && !down), world, inv);
    if st != before {
        st.set(&ctx);
    }
    used
}

#[allow(clippy::too_many_arguments)]
fn draw(ui: &egui::Ui, scene: &LoadedScene, i: usize, view: &View, st: &ShapeEditState, hover: Option<Pos2>, world: Mat4, inv: Mat4) {
    let Some(s) = shape(scene, i) else { return };
    let painter = ui.painter_at(view.rect);
    let line = egui::Stroke::new(1.5_f32, theme::with_alpha(theme::accent(), 0.85));
    // The tessellated spline: what the generator cuts it into.
    let poly = tessellate(&s.points, s.closed, s.tess_mode(), s.max_points);
    let pts: Vec<Option<Pos2>> = poly.pts.iter().map(|p| to_screen(view, world, *p)).collect();
    let n = pts.len();
    for k in 0..poly.segment_count() {
        if let (Some(a), Some(b)) = (pts[k], pts[(k + 1) % n]) {
            painter.line_segment([a, b], line);
        }
    }
    if st.show_tessellation {
        for p in pts.iter().flatten() {
            painter.circle_filled(*p, 2.0, theme::with_alpha(theme::accent(), 0.7));
        }
    }
    if !st.active {
        return;
    }
    let font = egui::FontId::proportional(theme::SIZE_META);
    for (k, p) in s.points.iter().enumerate() {
        let Some(q) = to_screen(view, world, Vec2::from(p.position)) else { continue };
        let selected = st.selected == Some(k);
        let color = if selected { theme::accent() } else { theme::TEXT_SECONDARY };
        let r = egui::Rect::from_center_size(q, egui::vec2(9.0, 9.0));
        if p.mode == ShapeTangentMode::Linear {
            painter.rect_filled(r, 1.0, color);
        } else {
            painter.circle_filled(q, 5.0, color);
        }
        if !p.corner {
            painter.circle_stroke(q, 7.5, egui::Stroke::new(1.0_f32, theme::STATUS_WARN_FG));
        }
        if selected {
            painter.circle_stroke(q, 11.0, egui::Stroke::new(2.0_f32, theme::accent()));
        }
        painter.text(q + egui::vec2(9.0, -9.0), egui::Align2::LEFT_BOTTOM, format!("{}", k + 1), font.clone(), color);
    }
    if let Some(k) = st.selected
        && let Some(p) = s.points.get(k)
        && p.mode != ShapeTangentMode::Linear
        && let Some(c) = to_screen(view, world, Vec2::from(p.position))
    {
        for t in [p.tangent_in, p.tangent_out] {
            if let Some(h) = to_screen(view, world, Vec2::from(p.position) + Vec2::from(t)) {
                painter.line_segment([c, h], egui::Stroke::new(1.0_f32, theme::TEXT_DIM));
                painter.circle(h, 4.0, theme::BG_PANEL, egui::Stroke::new(1.5_f32, theme::accent()));
            }
        }
    }
    // Ghost where the next point would go.
    if let Some(cur) = hover.filter(|c| view.rect.contains(*c))
        && let Some(local) = plane_hit(view, cur, inv)
    {
        let screen: Vec<(Handle, Pos2)> = handles(s, st.selected).into_iter().filter_map(|(h, p)| to_screen(view, world, p).map(|q| (h, q))).collect();
        if hit_handle(&screen, cur, HANDLE_PX).is_none() {
            let at = edit::snap(local.to_array(), st.snap);
            if let Some(q) = to_screen(view, world, Vec2::from(at)) {
                let ghost = egui::Stroke::new(1.5_f32, theme::with_alpha(theme::accent(), 0.8));
                painter.circle(q, 6.0, theme::with_alpha(theme::accent(), 0.25), ghost);
                if !s.closed
                    && let Some(last) = s.points.last()
                    && let Some(l) = to_screen(view, world, Vec2::from(last.position))
                {
                    painter.extend(egui::Shape::dashed_line(&[l, q], ghost, 6.0, 4.0));
                }
            }
        }
    }
    hint_bar(&painter, view.rect, SHAPE_HINT);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_shape() -> SpriteShapeRenderer {
        SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(4.0, 0.0), ShapePoint::at(4.0, 4.0)], ..Default::default() }
    }

    #[test]
    fn handles_list_points_and_the_selected_tangents() {
        let mut s = open_shape();
        assert_eq!(handles(&s, Some(1)).len(), 3, "Linear points have no tangent handles");
        s.points[1].set_mode(ShapeTangentMode::Continuous, [1.0, 0.0], 1.5);
        let h = handles(&s, Some(1));
        assert_eq!(h.len(), 5);
        assert_eq!(h[3], (Handle::TangentIn(1), Vec2::new(2.5, 0.0)));
        assert_eq!(h[4], (Handle::TangentOut(1), Vec2::new(5.5, 0.0)));
        assert_eq!(handles(&s, Some(0)).len(), 3, "only the selected point shows its handles");
        assert_eq!(handles(&s, None).len(), 3);
    }

    #[test]
    fn hit_prefers_the_nearest_and_tangents_on_ties() {
        let screen = vec![
            (Handle::Point(0), Pos2::new(10.0, 10.0)),
            (Handle::Point(1), Pos2::new(40.0, 10.0)),
            (Handle::TangentOut(1), Pos2::new(40.5, 10.0)),
        ];
        assert_eq!(hit_handle(&screen, Pos2::new(12.0, 10.0), 11.0), Some(Handle::Point(0)));
        assert_eq!(hit_handle(&screen, Pos2::new(40.2, 10.0), 11.0), Some(Handle::TangentOut(1)), "a tangent on top of its point wins");
        assert_eq!(hit_handle(&screen, Pos2::new(25.0, 10.0), 11.0), None);
        let apart = vec![(Handle::Point(1), Pos2::new(40.0, 10.0)), (Handle::TangentOut(1), Pos2::new(50.0, 10.0))];
        assert_eq!(hit_handle(&apart, Pos2::new(42.0, 10.0), 11.0), Some(Handle::Point(1)), "clearly closer to the point");
        assert_eq!(hit_handle(&apart, Pos2::new(48.0, 10.0), 11.0), Some(Handle::TangentOut(1)));
        assert_eq!(Handle::TangentIn(3).point(), 3);
    }

    #[test]
    fn dragging_moves_points_and_tangents_with_snapping() {
        let mut s = open_shape();
        let op = drag_op(&s, Handle::Point(1), Vec2::new(4.31, 0.12), 0.5).unwrap();
        edit::apply_op(&mut s, &op).unwrap();
        assert_eq!(s.points[1].position, [4.5, 0.0]);
        let op = drag_op(&s, Handle::TangentOut(1), Vec2::new(6.0, 1.0), 0.0).unwrap();
        edit::apply_op(&mut s, &op).unwrap();
        let p = s.points[1];
        assert_eq!(p.mode, ShapeTangentMode::Continuous, "a Linear point turns smooth when its handle is dragged");
        assert_eq!((p.tangent_out, p.tangent_in), ([1.5, 1.0], [-1.5, -1.0]));
        let op = drag_op(&s, Handle::TangentIn(1), Vec2::new(3.0, 0.0), 0.0).unwrap();
        edit::apply_op(&mut s, &op).unwrap();
        assert_eq!(s.points[1].tangent_out, [1.5, 0.0], "continuous mirrors the other side");
        assert!(drag_op(&s, Handle::Point(9), Vec2::ZERO, 0.0).is_none());
    }

    #[test]
    fn clicks_add_insert_and_split() {
        let s = open_shape();
        // Open: append, Ctrl prepends.
        let (op, sel) = click_op(&s, Vec2::new(8.0, 4.0), None, false, false, 0.0).unwrap();
        assert_eq!((sel, &op), (3, &SplineOp::Insert { index: 3, point: ShapePoint::at(8.0, 4.0) }));
        let (op, sel) = click_op(&s, Vec2::new(-3.0, 0.0), None, false, true, 0.0).unwrap();
        assert!(sel == 0 && matches!(op, SplineOp::Insert { index: 0, .. }));
        // Snapped.
        let (op, _) = click_op(&s, Vec2::new(8.13, 4.4), None, false, false, 0.5).unwrap();
        assert!(matches!(op, SplineOp::Insert { point, .. } if point.position == [8.0, 4.5]));
        // On the line: a split at that parameter, selecting the new point.
        let (op, sel) = click_op(&s, Vec2::new(2.0, 0.0), Some((0, 0.5)), false, false, 0.0).unwrap();
        assert_eq!(sel, 1);
        let mut t = s.clone();
        edit::apply_op(&mut t, &op).unwrap();
        assert_eq!((t.points.len(), t.points[1].position), (4, [2.0, 0.0]));
        // Shift splits at the nearest point wherever you click.
        let (op, sel) = click_op(&s, Vec2::new(5.0, 2.0), None, true, false, 0.0).unwrap();
        assert_eq!(sel, 2);
        let mut t = s.clone();
        edit::apply_op(&mut t, &op).unwrap();
        assert!((Vec2::from(t.points[2].position) - Vec2::new(4.0, 2.0)).length() < 0.1);
        // Closed: insert after the nearest segment; the wrap-around segment appends.
        let mut c = s.clone();
        c.closed = true;
        let (op, sel) = click_op(&c, Vec2::new(1.0, 3.0), None, false, false, 0.0).unwrap();
        assert!(matches!(op, SplineOp::Insert { index: 3, .. }) && sel == 3, "nearest is the closing segment");
        let (op, sel) = click_op(&c, Vec2::new(2.0, -1.0), None, false, false, 0.0).unwrap();
        assert!(matches!(op, SplineOp::Insert { index: 1, .. }) && sel == 1);
        // Closed shapes split on the line too, including the wrap.
        let (op, sel) = click_op(&c, Vec2::new(2.0, 2.0), Some((2, 0.5)), false, false, 0.0).unwrap();
        assert_eq!(sel, 3);
        let mut t = c.clone();
        edit::apply_op(&mut t, &op).unwrap();
        assert_eq!((t.points.len(), t.points[3].position), (4, [2.0, 2.0]));
        // An empty shape takes its first point.
        let empty = SpriteShapeRenderer { points: vec![], ..Default::default() };
        let (op, sel) = click_op(&empty, Vec2::new(1.0, 1.0), None, false, false, 0.0).unwrap();
        assert_eq!((sel, matches!(op, SplineOp::Insert { index: 0, .. })), (0, true));
    }
}
