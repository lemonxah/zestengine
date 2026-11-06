//! Scene-view river path editing (the Water inspector's Edit Path): clicks lay
//! points on the scene, insert them along the river or split a segment, Delete
//! removes the selected point, and an overlay previews where the next point
//! goes. Edits land on the selected object during the UI pass, so the frame's
//! undo diff records each one.

use zestengine_core::{Water, WaterBody};
use crate::{Gizmo, GizmoSpec, RiverHandle, RiverPathEdit, river_handle, theme};
use glam::{Mat4, Vec2, Vec3};

use zestengine_engine::scene::LoadedScene;

/// Screen distance (px) within which a click takes a handle.
const HANDLE_PX: f32 = 12.0;
/// ...and within which it lands on the centre line.
const LINE_PX: f32 = 7.0;

fn river(scene: &LoadedScene, i: usize) -> Option<&Water> {
    scene.objects.get(i)?.components.iter().find_map(|c| c.as_any().downcast_ref::<Water>()).filter(|w| w.body == WaterBody::River)
}

fn river_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut Water> {
    scene.objects.get_mut(i)?.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Water>()).filter(|w| w.body == WaterBody::River)
}

/// Keep the shared state on the selected object: selecting something else,
/// a Water that is no longer a river, or playing ends path editing.
pub(crate) fn sync(ctx: &egui::Context, scene: &LoadedScene, selected: Option<usize>, playing: bool) {
    let mut edit = RiverPathEdit::get(ctx);
    let before = edit;
    let id = selected.and_then(|i| scene.objects.get(i)).map(|o| o.id);
    if edit.object != id {
        edit = RiverPathEdit { object: id, ..Default::default() };
    }
    let points = selected.and_then(|i| river(scene, i)).map(|w| w.river.len());
    if points.is_none() || playing {
        edit.active = false;
    }
    edit.selected = edit.selected.filter(|&k| points.is_some_and(|n| k < n));
    if edit != before {
        edit.set(ctx);
    }
}

/// The selected river's path is being built.
pub(crate) fn building(ctx: &egui::Context) -> bool {
    RiverPathEdit::get(ctx).active
}

/// Leave path editing; false when it wasn't on.
pub(crate) fn exit(ctx: &egui::Context) -> bool {
    let mut edit = RiverPathEdit::get(ctx);
    if !edit.active {
        return false;
    }
    edit.active = false;
    edit.set(ctx);
    true
}

/// The Scene view's camera, for projecting and casting rays.
pub(crate) struct View {
    view_proj: Mat4,
    rect: egui::Rect,
    eye: Vec3,
}

impl View {
    pub(crate) fn new(view: Mat4, proj: Mat4, rect: egui::Rect) -> Self {
        Self { view_proj: proj * view, rect, eye: view.inverse().w_axis.truncate() }
    }

    fn project(&self, p: Vec3) -> Option<egui::Pos2> {
        let clip = self.view_proj * p.extend(1.0);
        if clip.w <= 0.001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(egui::pos2(
            self.rect.left() + (ndc.x * 0.5 + 0.5) * self.rect.width(),
            self.rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * self.rect.height(),
        ))
    }

    fn ray(&self, pos: egui::Pos2) -> Option<(Vec3, Vec3)> {
        if self.rect.width() < 1.0 || self.rect.height() < 1.0 {
            return None;
        }
        let ndc_x = 2.0 * ((pos.x - self.rect.left()) / self.rect.width()) - 1.0;
        let ndc_y = 1.0 - 2.0 * ((pos.y - self.rect.top()) / self.rect.height());
        let inv = self.view_proj.inverse();
        let near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).try_normalize()?;
        Some((self.eye, dir))
    }
}

/// Where a click puts the next point.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Placement {
    /// On the centre line, `t` of the way along `segment`.
    Split { segment: usize, t: f32 },
    /// At `index`, at local `position`.
    At { index: usize, position: Vec3 },
}

impl Placement {
    /// The new point's index and local position.
    fn target(self, water: &Water) -> Option<(usize, Vec3)> {
        match self {
            Placement::Split { segment, t } => water.river_point_on_segment(segment, t).map(|p| (segment + 1, Vec3::from(p.position))),
            Placement::At { index, position } => Some((index, position)),
        }
    }

    /// Add the point; returns its index.
    fn apply(self, water: &mut Water) -> Option<usize> {
        match self {
            Placement::Split { segment, t } => {
                let p = water.river_point_on_segment(segment, t)?;
                water.river.insert(segment + 1, p);
                Some(segment + 1)
            }
            Placement::At { index, position } => {
                water.insert_river_point(index, position);
                Some(index.min(water.river.len() - 1))
            }
        }
    }
}

/// Screen distance from `p` to the segment `a`-`b`.
fn screen_segment_dist(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let (ab, ap) = (b - a, p - a);
    let len2 = ab.length_sq();
    let t = if len2 > 1e-6 { (ap.dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t).distance(p)
}

/// How far (0..1) along `a`-`b` the segment comes closest to the ray.
fn closest_on_segment(a: Vec3, b: Vec3, origin: Vec3, dir: Vec3) -> f32 {
    let u = b - a;
    let w = a - origin;
    let (uu, ud, uw, dw) = (u.dot(u), u.dot(dir), u.dot(w), dir.dot(w));
    let denom = uu - ud * ud;
    if uu < 1e-10 || denom.abs() < 1e-10 {
        return 0.0;
    }
    ((ud * dw - uw) / denom).clamp(0.0, 1.0)
}

/// Where the ray meets the plane through `point` facing `normal`, ahead of it.
fn ray_plane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = dir.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - origin).dot(normal) / denom;
    (t > 0.0).then(|| origin + dir * t)
}

/// The placement under `cursor` for the river on object `i`: the centre line
/// when the cursor is on it (unless Ctrl), else the scene surface (or, with
/// nothing under it, the plane at the nearest end's height) at the end, the
/// start (Ctrl) or along the river (Shift).
fn placement(ctx: &egui::Context, scene: &LoadedScene, i: usize, view: &View, cursor: egui::Pos2, mods: egui::Modifiers) -> Option<Placement> {
    let water = river(scene, i)?;
    let world = scene.world_transform(i);
    let (origin, dir) = view.ray(cursor)?;
    let to_start = mods.ctrl || mods.command;
    if !to_start {
        let mut best: Option<(f32, usize, f32)> = None;
        for (k, pair) in water.river.windows(2).enumerate() {
            let (a, b) = (world.transform_point3(Vec3::from(pair[0].position)), world.transform_point3(Vec3::from(pair[1].position)));
            let (Some(pa), Some(pb)) = (view.project(a), view.project(b)) else { continue };
            let d = screen_segment_dist(cursor, pa, pb);
            if d <= LINE_PX && best.is_none_or(|(bd, _, _)| d < bd) {
                best = Some((d, k, closest_on_segment(a, b, origin, dir)));
            }
        }
        // The very ends belong to the handles there.
        if let Some((_, segment, t)) = best.filter(|&(_, _, t)| t > 0.02 && t < 0.98) {
            return Some(Placement::Split { segment, t });
        }
    }
    let n = water.river.len();
    let reference = match (to_start, water.river.first(), water.river.last()) {
        (true, Some(p), _) | (false, _, Some(p)) => Vec3::from(p.position),
        _ => Vec3::ZERO,
    };
    let up = world.transform_vector3(Vec3::Y).normalize_or(Vec3::Y);
    let hit = cached_pick(ctx, scene, origin, dir)
        .filter(|&(obj, _)| obj != i)
        .map(|(_, t)| origin + dir * t)
        .or_else(|| ray_plane(origin, dir, world.transform_point3(reference), up))?;
    let position = world.inverse().transform_point3(hit);
    let index = if to_start {
        0
    } else if mods.shift {
        water.river_insert_index(Vec2::new(position.x, position.z))
    } else {
        n
    };
    Some(Placement::At { index, position })
}

/// `pick_hit` for the ray, reused while the ray holds still (the preview asks
/// every frame and big meshes are slow to test).
fn cached_pick(ctx: &egui::Context, scene: &LoadedScene, origin: Vec3, dir: Vec3) -> Option<(usize, f32)> {
    let id = egui::Id::new("zestengine-river-path-pick");
    let key = [origin.x, origin.y, origin.z, dir.x, dir.y, dir.z];
    if let Some((k, hit)) = ctx.data(|d| d.get_temp::<([f32; 6], Option<(usize, f32)>)>(id))
        && k == key
    {
        return hit;
    }
    let hit = scene.pick_hit(origin, dir);
    ctx.data_mut(|d| d.insert_temp(id, (key, hit)));
    hit
}

/// The river point whose handle (point or width) is under `cursor`.
fn handle_under(water: &Water, world: Mat4, view: &View, cursor: egui::Pos2) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (g, spec) in water.gizmos().into_iter().enumerate() {
        let GizmoSpec::Handle { position, .. } = spec else { continue };
        let Some(k) = river_handle(water, g).map(|h| match h {
            RiverHandle::Point(k) | RiverHandle::Width(k) | RiverHandle::Depth(k) | RiverHandle::Flow(k) => k,
        }) else {
            continue;
        };
        let Some(p) = view.project(world.transform_point3(position)) else { continue };
        let d = p.distance(cursor);
        if d <= HANDLE_PX && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, k));
        }
    }
    best.map(|(_, k)| k)
}

/// Scene-view input and overlay for the selected river (object `i`): point
/// selection from handle clicks and drags, and while building, click to add,
/// Delete / Backspace to remove and the placement preview. `new_drag` is the
/// (component, gizmo) of a handle drag that started this frame; `click` a
/// click the viewport may take. Returns true when the click was used.
#[allow(clippy::too_many_arguments)]
pub(crate) fn viewport(
    ui: &mut egui::Ui,
    scene: &mut LoadedScene,
    i: usize,
    view: &View,
    cursor: Option<egui::Pos2>,
    new_drag: Option<(usize, usize)>,
    click: Option<egui::Pos2>,
    dragging: bool,
) -> bool {
    let ctx = ui.ctx().clone();
    let mut edit = RiverPathEdit::get(&ctx);
    if river(scene, i).is_none() || edit.object != Some(scene.objects[i].id) {
        return false;
    }
    let before = edit;
    let world = scene.world_transform(i);
    let mut used = false;

    if let Some((component, gizmo)) = new_drag
        && let Some(w) = scene.objects[i].components.get(component).and_then(|c| c.as_any().downcast_ref::<Water>())
        && let Some(RiverHandle::Point(k) | RiverHandle::Width(k) | RiverHandle::Depth(k) | RiverHandle::Flow(k)) = river_handle(w, gizmo)
    {
        edit.selected = Some(k);
        edit.reveal = true;
    }

    if let Some(pos) = click {
        let hit = river(scene, i).and_then(|w| handle_under(w, world, view, pos));
        if let Some(k) = hit {
            edit.selected = Some(k);
            edit.reveal = true;
            used = true;
        } else if edit.active {
            let mods = ui.input(|inp| inp.modifiers);
            if let Some(place) = placement(&ctx, scene, i, view, pos, mods)
                && let Some(k) = river_mut(scene, i).and_then(|w| place.apply(w))
            {
                edit.selected = Some(k);
                edit.reveal = true;
            }
            used = true;
        }
    }

    if edit.active && !ctx.wants_keyboard_input() && ui.input(|inp| inp.key_pressed(egui::Key::Delete) || inp.key_pressed(egui::Key::Backspace))
        && let Some(k) = edit.selected
        && let Some(w) = river_mut(scene, i)
        && k < w.river.len()
    {
        w.river.remove(k);
        edit.selected = (!w.river.is_empty()).then(|| k.saturating_sub(1).min(w.river.len() - 1));
        edit.reveal = true;
    }

    let painter = ui.painter_at(view.rect);
    let water = river(scene, i).expect("checked above");
    let screen: Vec<Option<egui::Pos2>> = water.river.iter().map(|p| view.project(world.transform_point3(Vec3::from(p.position)))).collect();
    if let Some(p) = edit.selected.and_then(|k| screen.get(k).copied().flatten()) {
        painter.circle_stroke(p, 11.0, egui::Stroke::new(2.0_f32, theme::accent()));
        painter.circle_filled(p, 3.0, theme::accent());
    }
    if edit.active {
        let font = egui::FontId::proportional(theme::SIZE_META);
        for (k, p) in screen.iter().enumerate() {
            if let Some(p) = p {
                let color = if edit.selected == Some(k) { theme::accent() } else { theme::TEXT_SECONDARY };
                painter.text(*p + egui::vec2(9.0, -9.0), egui::Align2::LEFT_BOTTOM, format!("{}", k + 1), font.clone(), color);
            }
        }
        let hovering = cursor.filter(|c| view.rect.contains(*c) && !dragging);
        let over_handle = hovering.is_some_and(|c| handle_under(water, world, view, c).is_some());
        if let Some(c) = hovering.filter(|_| !over_handle) {
            let mods = ui.input(|inp| inp.modifiers);
            if let Some((place, (_, local))) = placement(&ctx, scene, i, view, c, mods).and_then(|p| Some((p, p.target(water)?)))
                && let Some(at) = view.project(world.transform_point3(local))
            {
                let ghost = egui::Stroke::new(1.5_f32, theme::with_alpha(theme::accent(), 0.8));
                let (prev, next) = match place {
                    Placement::Split { .. } => (None, None),
                    Placement::At { index, .. } => (index.checked_sub(1).and_then(|k| screen.get(k).copied().flatten()), screen.get(index).copied().flatten()),
                };
                for q in [prev, next].into_iter().flatten() {
                    painter.extend(egui::Shape::dashed_line(&[q, at], ghost, 6.0, 4.0));
                }
                painter.circle(at, 6.0, theme::with_alpha(theme::accent(), 0.25), ghost);
                let label = match place {
                    Placement::Split { segment, .. } => format!("split {}-{}", segment + 1, segment + 2),
                    Placement::At { index, .. } if index >= water.river.len() => format!("add {}", index + 1),
                    Placement::At { index, .. } => format!("insert {}", index + 1),
                };
                painter.text(at + egui::vec2(10.0, 4.0), egui::Align2::LEFT_TOP, label, font.clone(), theme::accent());
            }
        }
        hint_bar(&painter, view.rect);
    }

    if edit != before {
        edit.set(&ctx);
    }
    used
}

/// The key reminder along the bottom of the Scene view while building.
fn hint_bar(painter: &egui::Painter, rect: egui::Rect) {
    let text = "River path   Click add   Shift+Click insert   Ctrl+Click add at start   Click line split   Drag move   Del remove   Esc done";
    let galley = painter.layout_no_wrap(text.to_owned(), egui::FontId::proportional(theme::SIZE_META), theme::TEXT_SECONDARY);
    let size = galley.size() + egui::vec2(20.0, 10.0);
    let bar = egui::Rect::from_center_size(rect.center_bottom() - egui::vec2(0.0, size.y * 0.5 + 12.0), size);
    painter.rect(bar, theme::R_ROW, theme::with_alpha(theme::BG_PANEL, 0.92), egui::Stroke::new(1.0_f32, theme::accent_soft()), egui::StrokeKind::Inside);
    painter.galley(bar.center() - galley.size() * 0.5, galley, theme::TEXT_SECONDARY);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_on_a_segment_to_a_ray() {
        let (a, b) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let down = Vec3::NEG_Y;
        assert!((closest_on_segment(a, b, Vec3::new(3.0, 5.0, 0.0), down) - 0.3).abs() < 1e-5);
        assert!((closest_on_segment(a, b, Vec3::new(7.5, 5.0, 2.0), down) - 0.75).abs() < 1e-5, "offset to the side");
        assert_eq!(closest_on_segment(a, b, Vec3::new(-4.0, 5.0, 0.0), down), 0.0, "clamped to the ends");
        let slanted = (Vec3::new(5.0, -5.0, -5.0)).normalize();
        assert!((closest_on_segment(a, b, Vec3::new(0.0, 5.0, 5.0), slanted) - 0.5).abs() < 1e-4);
        assert_eq!(closest_on_segment(a, b, Vec3::new(0.0, 1.0, 0.0), Vec3::X), 0.0, "parallel");
    }

    #[test]
    fn rays_meet_planes_only_ahead() {
        let hit = ray_plane(Vec3::new(1.0, 10.0, 2.0), Vec3::NEG_Y, Vec3::new(0.0, 3.0, 0.0), Vec3::Y).unwrap();
        assert!((hit - Vec3::new(1.0, 3.0, 2.0)).length() < 1e-5);
        assert!(ray_plane(Vec3::new(1.0, 10.0, 2.0), Vec3::Y, Vec3::new(0.0, 3.0, 0.0), Vec3::Y).is_none(), "behind");
        assert!(ray_plane(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 3.0, 0.0), Vec3::Y).is_none(), "parallel");
    }

    #[test]
    fn screen_distance_to_a_segment() {
        let (a, b) = (egui::pos2(0.0, 0.0), egui::pos2(100.0, 0.0));
        assert_eq!(screen_segment_dist(egui::pos2(50.0, 6.0), a, b), 6.0);
        assert_eq!(screen_segment_dist(egui::pos2(103.0, 4.0), a, b), 5.0);
        assert_eq!(screen_segment_dist(egui::pos2(3.0, 4.0), a, a), 5.0);
    }

    fn straight() -> Water {
        let p = |x: f32, width: f32| zestengine_core::RiverPoint { position: [x, 0.0, 0.0], width, depth: 1.0, speed: 1.0 };
        Water { body: WaterBody::River, river: vec![p(0.0, 2.0), p(10.0, 6.0)], ..Default::default() }
    }

    #[test]
    fn placements_add_where_they_point() {
        let mut w = straight();
        let split = Placement::Split { segment: 0, t: 0.25 };
        let (index, at) = split.target(&w).unwrap();
        assert!(index == 1 && at.distance(Vec3::new(2.5, 0.0, 0.0)) < 1e-4, "on the curve: {at}");
        assert_eq!(split.apply(&mut w), Some(1));
        assert_eq!(w.river.len(), 3);
        assert_eq!(w.river[1].width, 3.0, "a split keeps the river's shape");
        let end = Placement::At { index: 3, position: Vec3::new(20.0, 1.0, 4.0) };
        assert_eq!(end.apply(&mut w), Some(3));
        assert_eq!(w.river[3].position, [20.0, 1.0, 4.0]);
        assert_eq!(w.river[3].width, 6.0);
        let start = Placement::At { index: 0, position: Vec3::new(-8.0, 0.0, 0.0) };
        assert_eq!(start.apply(&mut w), Some(0));
        assert_eq!(w.river[0].position, [-8.0, 0.0, 0.0]);
        assert_eq!(w.river.len(), 5);
    }
}
