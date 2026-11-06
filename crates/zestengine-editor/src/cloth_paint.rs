//! Scene-view pin painting for Mesh Cloth (the inspector's Paint Pins, or P
//! over the Scene view): a brush projected on the cloth's shown pose paints
//! pin weights or max distances, over a heat map of the painted values.
//! Strokes land on the selected object during the UI pass, so the frame's
//! undo diff records them, and the undo stack keeps each stroke as one entry.

use zestengine_core::{BrushMode, ClothPaint, Dab, MeshCloth, PaintChannel, PinSource, max_scale, mirror_x, ray_triangles};
use crate::{ClothPaintTool, theme};
use glam::{Mat4, Vec3};

use zestengine_engine::scene::LoadedScene;

/// Stroke dabs land every this share of the brush radius.
const DAB_SPACING: f32 = 0.2;
/// Overlay budget: past this many triangles the heat map skips some.
const MAX_OVERLAY_TRIANGLES: usize = 60_000;

fn cloth(scene: &LoadedScene, i: usize) -> Option<&MeshCloth> {
    scene.objects.get(i)?.components.iter().find_map(|c| c.as_any().downcast_ref::<MeshCloth>())
}

fn cloth_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut MeshCloth> {
    scene.objects.get_mut(i)?.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<MeshCloth>())
}

/// Keep the painter on the selected Mesh Cloth: selecting something else or
/// playing ends painting. Sets the cloth's paint flags for this frame and, on
/// the first painted frame, switches it to painted pins seeded from its
/// current ones.
pub(crate) fn sync(ctx: &egui::Context, scene: &mut LoadedScene, selected: Option<usize>, playing: bool) {
    let mut tool = ClothPaintTool::get(ctx);
    let before = tool;
    let selected = selected.filter(|&i| cloth(scene, i).is_some());
    let id = selected.map(|i| scene.objects[i].id);
    if tool.object != id {
        if let Some(old) = tool.object
            && let Some(k) = scene.objects.iter().position(|o| o.id == old)
            && let Some(mc) = cloth_mut(scene, k)
        {
            mc.runtime.paint_mode = false;
            mc.runtime.hold = false;
        }
        tool.object = id;
        tool.active = false;
    }
    if id.is_none() || playing {
        tool.active = false;
    }
    if !tool.active {
        tool.stroking = false;
    }
    if let Some(mc) = selected.and_then(|i| cloth_mut(scene, i)) {
        mc.runtime.paint_mode = tool.active;
        mc.runtime.hold = tool.active && !tool.simulate;
        if tool.active {
            ensure_painted(mc);
        }
    }
    if tool != before {
        tool.set(ctx);
    }
}

/// Switch to painted pins (seeded from the current source) and keep them
/// fitted to the simulated mesh. False while the cloth hasn't built yet.
fn ensure_painted(mc: &mut MeshCloth) -> bool {
    let n = mc.runtime.bind.len();
    if n == 0 {
        return false;
    }
    if mc.pin != PinSource::Painted {
        let seed = if mc.runtime.source_pins.len() == n { mc.runtime.source_pins.clone() } else { Vec::new() };
        mc.paint = ClothPaint::seed(&mc.runtime.bind, &mc.runtime.indices, &seed);
        mc.pin = PinSource::Painted;
        mc.runtime.retether = true;
    } else if !mc.paint.matches(n, mc.runtime.topology) {
        mc.paint.fit(&mc.runtime.bind, &mc.runtime.indices);
        mc.runtime.retether = true;
    }
    true
}

/// The painter is open.
pub(crate) fn active(ctx: &egui::Context) -> bool {
    ClothPaintTool::get(ctx).active
}

/// A stroke is in progress (the undo stack keeps it as one entry).
pub(crate) fn stroking(ctx: &egui::Context) -> bool {
    ClothPaintTool::get(ctx).stroking
}

/// Leave painting; false when it wasn't on.
pub(crate) fn exit(ctx: &egui::Context) -> bool {
    let mut tool = ClothPaintTool::get(ctx);
    if !tool.active {
        return false;
    }
    tool.active = false;
    tool.stroking = false;
    tool.set(ctx);
    true
}

/// The Scene view's camera, for projecting and casting rays.
pub(crate) struct View {
    view_proj: Mat4,
    pub(crate) rect: egui::Rect,
    eye: Vec3,
    right: Vec3,
}

impl View {
    pub(crate) fn new(view: Mat4, proj: Mat4, rect: egui::Rect) -> Self {
        let inv = view.inverse();
        Self { view_proj: proj * view, rect, eye: inv.w_axis.truncate(), right: inv.x_axis.truncate().normalize_or(Vec3::X) }
    }

    pub(crate) fn project(&self, p: Vec3) -> Option<egui::Pos2> {
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

    pub(crate) fn ray(&self, pos: egui::Pos2) -> Option<(Vec3, Vec3)> {
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

/// Where the cursor ray meets the cloth: the point on the shown pose and the
/// same surface point on the bind mesh (object space), and the shown normal.
struct Hit {
    posed: Vec3,
    bind: Vec3,
    normal: Vec3,
}

fn hit_cloth(view: &View, cursor: egui::Pos2, world: Mat4, posed: &[Vec3], bind: &[Vec3], indices: &[u32]) -> Option<Hit> {
    let (origin, dir) = view.ray(cursor)?;
    let inv = world.inverse();
    let (o, d) = (inv.transform_point3(origin), inv.transform_vector3(dir));
    let (_, tri, bary) = ray_triangles(o, d, posed, indices)?;
    let corners = [indices[tri * 3] as usize, indices[tri * 3 + 1] as usize, indices[tri * 3 + 2] as usize];
    let w = [bary.x, bary.y, bary.z];
    let at = |pts: &[Vec3]| corners.iter().zip(w).map(|(&c, w)| pts[c] * w).sum::<Vec3>();
    let normal = (posed[corners[1]] - posed[corners[0]]).cross(posed[corners[2]] - posed[corners[0]]).normalize_or(Vec3::Y);
    Some(Hit { posed: at(posed), bind: at(bind), normal })
}

fn heat(channel: PaintChannel, v: f32) -> [f32; 3] {
    let free = [18.0, 18.0, 22.0];
    let full = match channel {
        PaintChannel::Pin => [235.0, 40.0, 30.0],
        PaintChannel::MaxDistance => [60.0, 150.0, 255.0],
    };
    let v = v.clamp(0.0, 1.0);
    [free[0] + (full[0] - free[0]) * v, free[1] + (full[1] - free[1]) * v, free[2] + (full[2] - free[2]) * v]
}

fn mode_color(mode: BrushMode) -> egui::Color32 {
    match mode {
        BrushMode::Paint => egui::Color32::from_rgb(255, 110, 90),
        BrushMode::Erase => egui::Color32::from_rgb(170, 190, 220),
        BrushMode::Smooth => egui::Color32::from_rgb(120, 220, 140),
        BrushMode::Set => egui::Color32::from_rgb(250, 210, 90),
    }
}

/// Scene-view input and overlay of the painter on object `i` (the selection):
/// the P toggle, brush size keys, strokes, the heat map, pinned vertex dots,
/// max-distance spheres and the brush circle. Returns true when it took the
/// viewport click (no object picking while painting).
pub(crate) fn viewport(ui: &mut egui::Ui, scene: &mut LoadedScene, i: usize, view: &View, response: &egui::Response) -> bool {
    let ctx = ui.ctx().clone();
    let mut tool = ClothPaintTool::get(&ctx);
    if cloth(scene, i).is_none() || tool.object != Some(scene.objects[i].id) {
        return false;
    }
    let before = tool;
    let keys = response.hovered() && !ctx.wants_keyboard_input();
    if keys && ui.input(|inp| inp.modifiers.is_none() && inp.key_pressed(egui::Key::P)) {
        tool.active = !tool.active;
        tool.stroking = false;
    }
    if !tool.active {
        if tool != before {
            tool.set(&ctx);
        }
        return false;
    }
    if keys {
        let (smaller, larger) = ui.input(|inp| (inp.key_pressed(egui::Key::OpenBracket), inp.key_pressed(egui::Key::CloseBracket)));
        if smaller {
            tool.radius /= 1.15;
        }
        if larger {
            tool.radius *= 1.15;
        }
    }
    let (mods, scroll, primary_down) = ui.input(|inp| (inp.modifiers, inp.raw_scroll_delta.y, inp.pointer.primary_down()));
    if mods.command && scroll != 0.0 && response.contains_pointer() {
        tool.radius *= 1.1f32.powf(scroll / 50.0);
    }
    tool.radius = tool.radius.clamp(0.002, 5.0);

    let world = scene.world_transform(i);
    let painter = ui.painter_at(view.rect);
    let cursor = response.hover_pos().or_else(|| response.interact_pointer_pos()).filter(|c| view.rect.contains(*c));
    let mode = if mods.shift {
        BrushMode::Erase
    } else if mods.alt {
        BrushMode::Smooth
    } else {
        tool.mode
    };

    let mc = cloth(scene, i).expect("checked above");
    let ready = !mc.runtime.bind.is_empty() && mc.pin == PinSource::Painted && mc.paint.matches(mc.runtime.bind.len(), mc.runtime.topology);
    if !ready {
        hint_bar(&painter, view.rect, "Paint pins: waiting for the cloth to build (the object must be skinned and active)   Esc done");
        tool.stroking = false;
        if tool != before {
            tool.set(&ctx);
        }
        return response.clicked();
    }
    let bind = &mc.runtime.bind;
    let posed = if mc.runtime.posed.len() == bind.len() { &mc.runtime.posed } else { bind };
    let hit = cursor.and_then(|c| hit_cloth(view, c, world, posed, bind, &mc.runtime.indices));

    // Strokes: a press on the view opens one, dabs land as the brush moves
    // across the cloth, the release closes it.
    let scale = max_scale(world).max(1e-6);
    let radius = tool.radius / scale;
    let last_id = egui::Id::new("zestengine-cloth-paint-last-dab");
    if !tool.stroking && primary_down && response.is_pointer_button_down_on() {
        tool.stroking = true;
        ctx.data_mut(|d| d.remove::<Vec3>(last_id));
    }
    let mut painted = false;
    if tool.stroking && !primary_down {
        tool.stroking = false;
        if let Some(mc) = cloth_mut(scene, i) {
            mc.runtime.retether = true;
        }
    } else if tool.stroking
        && let Some(h) = &hit
    {
        let last: Option<Vec3> = ctx.data(|d| d.get_temp(last_id));
        if last.is_none_or(|l| (l - h.bind).length() >= radius * DAB_SPACING) {
            ctx.data_mut(|d| d.insert_temp(last_id, h.bind));
            let centers = if tool.mirror_x { vec![h.bind, mirror_x(h.bind)] } else { vec![h.bind] };
            let dab = Dab { centers, radius, strength: tool.strength, falloff: tool.falloff, mode, value: tool.value };
            if let Some(mc) = cloth_mut(scene, i) {
                let mut values = mc.paint.values(tool.channel);
                if let Some(weld) = mc.runtime.weld()
                    && weld.apply(&mut values, &dab)
                {
                    mc.paint.set_values(tool.channel, &values);
                    mc.runtime.pins_dirty = true;
                    painted = true;
                }
            }
        }
    }
    if painted {
        ctx.request_repaint();
    }
    if let Some(mc) = cloth_mut(scene, i) {
        mc.runtime.weld();
    }

    let mc = cloth(scene, i).expect("checked above");
    let bind = &mc.runtime.bind;
    let posed = if mc.runtime.posed.len() == bind.len() { &mc.runtime.posed } else { bind };
    draw_heat_map(&painter, view, world, posed, &mc.runtime.indices, &mc.paint.values(tool.channel), tool.channel);
    let weld = mc.runtime.weld.as_ref().filter(|w| w.vertex_count() == bind.len());
    if let Some(weld) = weld {
        let pins = mc.paint.pin_weights();
        let groups: Vec<&Vec<u32>> = weld.members.iter().filter(|m| m.first().is_some_and(|&v| pins[v as usize] >= 0.99)).collect();
        let stride = groups.len().div_ceil(3000).max(1);
        for m in groups.into_iter().step_by(stride) {
            if let Some(p) = view.project(world.transform_point3(posed[m[0] as usize])) {
                painter.circle(p, 2.2, egui::Color32::from_rgb(255, 70, 50), egui::Stroke::new(0.8_f32, egui::Color32::from_rgb(40, 8, 6)));
            }
        }
        if tool.show_distance && !mc.paint.max_distance.is_empty() {
            let limits = mc.paint.values(PaintChannel::MaxDistance);
            let stride = weld.members.len().div_ceil(400).max(1);
            let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(110, 180, 255, 150));
            for m in weld.members.iter().step_by(stride) {
                let at = world.transform_point3(posed[m[0] as usize]);
                let r = limits[m[0] as usize] * mc.max_distance * scale;
                if let (Some(c), Some(edge)) = (view.project(at), view.project(at + view.right * r)) {
                    painter.circle_stroke(c, c.distance(edge).max(1.0), stroke);
                }
            }
        }
    }
    if let Some(h) = &hit {
        let color = mode_color(mode);
        let n = world.transform_vector3(h.normal).normalize_or(Vec3::Y);
        brush_circle(&painter, view, world.transform_point3(h.posed), n, tool.radius, egui::Stroke::new(1.5_f32, color));
        if tool.mirror_x {
            let m = world.transform_vector3(mirror_x(h.normal)).normalize_or(Vec3::Y);
            brush_circle(&painter, view, world.transform_point3(mirror_x(h.posed)), m, tool.radius, egui::Stroke::new(1.0_f32, color.gamma_multiply(0.6)));
        }
        if let Some(c) = view.project(world.transform_point3(h.posed)) {
            painter.circle_filled(c, 2.0, color);
        }
    }
    let channel = match tool.channel {
        PaintChannel::Pin => "Pin weight",
        PaintChannel::MaxDistance => "Max distance",
    };
    hint_bar(
        &painter,
        view.rect,
        &format!(
            "Paint {channel}: {}   radius {:.3} m   Drag paint   Shift erase   Alt smooth   [ ] or Ctrl+Scroll size   P / Esc done",
            mode.label(),
            tool.radius
        ),
    );
    if tool != before {
        tool.set(&ctx);
    }
    response.clicked() || tool.stroking
}

/// A circle of world radius `r` around `center` in the plane facing `normal`.
pub(crate) fn brush_circle(painter: &egui::Painter, view: &View, center: Vec3, normal: Vec3, r: f32, stroke: egui::Stroke) {
    let up = if normal.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let a = up.cross(normal).normalize();
    let b = normal.cross(a);
    let pts: Vec<egui::Pos2> = (0..=48)
        .filter_map(|k| {
            let t = k as f32 / 48.0 * std::f32::consts::TAU;
            view.project(center + (a * t.cos() + b * t.sin()) * r)
        })
        .collect();
    if pts.len() > 2 {
        painter.add(egui::Shape::line(pts, stroke));
    }
}

/// The painted values as a heat map over the cloth's shown pose, depth
/// sorted back to front and shaded by facing so its shape reads.
fn draw_heat_map(painter: &egui::Painter, view: &View, world: Mat4, posed: &[Vec3], indices: &[u32], values: &[f32], channel: PaintChannel) {
    if values.len() != posed.len() {
        return;
    }
    let wp: Vec<Vec3> = posed.iter().map(|p| world.transform_point3(*p)).collect();
    let sp: Vec<Option<egui::Pos2>> = wp.iter().map(|p| view.project(*p)).collect();
    let tris = indices.as_chunks::<3>().0;
    let stride = tris.len().div_ceil(MAX_OVERLAY_TRIANGLES).max(1);
    let mut order: Vec<(f32, usize)> = tris
        .iter()
        .enumerate()
        .step_by(stride)
        .filter(|(_, t)| t.iter().all(|&v| sp.get(v as usize).is_some_and(Option::is_some)))
        .map(|(k, t)| (t.iter().map(|&v| (wp[v as usize] - view.eye).length_squared()).sum::<f32>(), k))
        .collect();
    order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let mut mesh = egui::Mesh::default();
    for (_, k) in order {
        let t = tris[k];
        let (a, b, c) = (wp[t[0] as usize], wp[t[1] as usize], wp[t[2] as usize]);
        let n = (b - a).cross(c - a).normalize_or_zero();
        let facing = n.dot((view.eye - (a + b + c) / 3.0).normalize_or_zero());
        let shade = (0.45 + 0.55 * facing.abs()) * if facing < 0.0 { 0.6 } else { 1.0 };
        let base = mesh.vertices.len() as u32;
        for v in t {
            let [r, g, bl] = heat(channel, values[v as usize]);
            let color = egui::Color32::from_rgba_unmultiplied((r * shade) as u8, (g * shade) as u8, (bl * shade) as u8, 225);
            mesh.colored_vertex(sp[v as usize].expect("filtered above"), color);
        }
        mesh.add_triangle(base, base + 1, base + 2);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// The key reminder along the bottom of the Scene view while painting.
pub(crate) fn hint_bar(painter: &egui::Painter, rect: egui::Rect, text: &str) {
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
    fn heat_runs_from_dark_to_the_channel_colour() {
        assert_eq!(heat(PaintChannel::Pin, 0.0), heat(PaintChannel::MaxDistance, 0.0));
        assert_eq!(heat(PaintChannel::Pin, 1.0), [235.0, 40.0, 30.0]);
        assert_eq!(heat(PaintChannel::Pin, 2.0), heat(PaintChannel::Pin, 1.0), "clamped");
        let mid = heat(PaintChannel::MaxDistance, 0.5);
        assert!(mid[2] > mid[0] && mid[2] < 255.0);
    }

    #[test]
    fn the_cursor_ray_finds_the_bind_point_under_the_posed_surface() {
        // A unit quad facing +Z; the posed copy is lifted 1 m along Y.
        let bind = vec![Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 0.0), Vec3::new(-0.5, 0.5, 0.0)];
        let posed: Vec<Vec3> = bind.iter().map(|p| *p + Vec3::Y).collect();
        let idx = [0, 1, 2, 0, 2, 3];
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let view = View::new(Mat4::look_at_rh(Vec3::new(0.2, 1.1, 5.0), Vec3::new(0.2, 1.1, 0.0), Vec3::Y), Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0), rect);
        let h = hit_cloth(&view, rect.center(), Mat4::IDENTITY, &posed, &bind, &idx).unwrap();
        assert!((h.posed - Vec3::new(0.2, 1.1, 0.0)).length() < 1e-3, "{}", h.posed);
        assert!((h.bind - Vec3::new(0.2, 0.1, 0.0)).length() < 1e-3, "{}", h.bind);
        assert!(h.normal.z.abs() > 0.99);
        // Through a moved object.
        let world = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let view = View::new(Mat4::look_at_rh(Vec3::new(3.2, 1.1, 5.0), Vec3::new(3.2, 1.1, 0.0), Vec3::Y), Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0), rect);
        let h = hit_cloth(&view, rect.center(), world, &posed, &bind, &idx).unwrap();
        assert!((h.bind - Vec3::new(0.2, 0.1, 0.0)).length() < 1e-3);
        assert!(hit_cloth(&view, egui::pos2(1.0, 1.0), world, &posed, &bind, &idx).is_none(), "off the cloth");
    }
}
