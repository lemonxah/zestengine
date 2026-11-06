//! Scene-view sculpting for `Terrain`: a Unity-style brush raises, lowers,
//! sets, or smooths the heightfield under the cursor. Strokes land on the
//! selected object during the UI pass so the frame's undo diff records them,
//! and the undo stack keeps each stroke as one entry.

use zestengine_core::{BrushFalloff, Terrain, max_scale, ray_triangles};
use crate::{TerrainBrushMode, TerrainPaintTool};
use glam::Vec3;

use crate::cloth_paint::{View, brush_circle, hint_bar};
use zestengine_engine::scene::LoadedScene;

/// Raise/lower rate in metres per second at full strength and weight.
const RAISE_SPEED: f32 = 0.6;
/// Set-height and smooth convergence rate (1/s) at full strength and weight.
const LERP_SPEED: f32 = 6.0;

fn terrain(scene: &LoadedScene, i: usize) -> Option<&Terrain> {
    scene
        .objects
        .get(i)?
        .components
        .iter()
        .find_map(|c| c.as_any().downcast_ref::<Terrain>())
}

fn terrain_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut Terrain> {
    scene
        .objects
        .get_mut(i)?
        .components
        .iter_mut()
        .find_map(|c| c.as_any_mut().downcast_mut::<Terrain>())
}

/// Keep the painter on the selected Terrain; selecting something else or
/// playing ends it.
pub(crate) fn sync(
    ctx: &egui::Context,
    scene: &LoadedScene,
    selected: Option<usize>,
    playing: bool,
) {
    let mut tool = TerrainPaintTool::get(ctx);
    let before = tool;
    let selected = selected.filter(|&i| terrain(scene, i).is_some());
    let id = selected.map(|i| scene.objects[i].id);
    if tool.object != id {
        tool.object = id;
        tool.active = false;
    }
    if id.is_none() || playing {
        tool.active = false;
    }
    if !tool.active {
        tool.stroking = false;
    }
    if tool != before {
        tool.set(ctx);
    }
}

pub(crate) fn active(ctx: &egui::Context) -> bool {
    TerrainPaintTool::get(ctx).active
}

pub(crate) fn stroking(ctx: &egui::Context) -> bool {
    TerrainPaintTool::get(ctx).stroking
}

pub(crate) fn exit(ctx: &egui::Context) -> bool {
    let mut tool = TerrainPaintTool::get(ctx);
    if !tool.active {
        return false;
    }
    tool.active = false;
    tool.stroking = false;
    tool.set(ctx);
    true
}

/// Object-space vertex positions and triangle indices for a heightfield,
/// mirroring `zestengine_assets::heightfield_mesh` (centred at the origin).
fn heightfield(terrain: &Terrain) -> (Vec<Vec3>, Vec<u32>) {
    let n = (terrain.resolution + 1) as usize;
    let step = terrain.size / terrain.resolution as f32;
    let half = terrain.size * 0.5;
    let mut positions = Vec::with_capacity(n * n);
    for z in 0..n {
        for x in 0..n {
            positions.push(Vec3::new(
                x as f32 * step - half,
                terrain.heights[z * n + x],
                z as f32 * step - half,
            ));
        }
    }
    let quads = terrain.resolution as usize;
    let mut indices = Vec::with_capacity(quads * quads * 6);
    for z in 0..quads {
        for x in 0..quads {
            let a = (z * n + x) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, c, d, b]);
        }
    }
    (positions, indices)
}

/// Apply one frame of sculpting at an object-space surface point. Returns true
/// when any vertex height changed. `dt` is the frame time in seconds.
#[allow(clippy::too_many_arguments)] // one brush dab: centre, radius, mode, falloff, strength, height.
pub(crate) fn apply_dab(
    terrain: &mut Terrain,
    center: Vec3,
    radius: f32,
    mode: TerrainBrushMode,
    falloff: BrushFalloff,
    strength: f32,
    height: f32,
    dt: f32,
) -> bool {
    let res = terrain.resolution as usize;
    let n = res + 1;
    if terrain.heights.len() != n * n || radius <= 0.0 || dt <= 0.0 {
        return false;
    }
    let step = terrain.size / terrain.resolution as f32;
    let half = terrain.size * 0.5;
    let x0 = (((center.x - radius + half) / step).floor() as i64).clamp(0, res as i64) as usize;
    let x1 = (((center.x + radius + half) / step).ceil() as i64).clamp(0, res as i64) as usize;
    let z0 = (((center.z - radius + half) / step).floor() as i64).clamp(0, res as i64) as usize;
    let z1 = (((center.z + radius + half) / step).ceil() as i64).clamp(0, res as i64) as usize;

    let mut hits: Vec<(usize, f32)> = Vec::new();
    for gz in z0..=z1 {
        for gx in x0..=x1 {
            let px = gx as f32 * step - half;
            let pz = gz as f32 * step - half;
            let d = ((px - center.x).powi(2) + (pz - center.z).powi(2)).sqrt();
            let w = falloff.weight(d, radius);
            if w > 0.0 {
                hits.push((gz * n + gx, w));
            }
        }
    }
    if hits.is_empty() {
        return false;
    }

    let strength = strength.clamp(0.0, 1.0);
    let dt = dt.min(0.1);
    let mut changed = false;
    match mode {
        TerrainBrushMode::Raise => {
            for (i, w) in &hits {
                let delta = RAISE_SPEED * strength * w * dt;
                if delta > 0.0 {
                    terrain.heights[*i] += delta;
                    changed = true;
                }
            }
        }
        TerrainBrushMode::Lower => {
            for (i, w) in &hits {
                let delta = RAISE_SPEED * strength * w * dt;
                if delta > 0.0 {
                    terrain.heights[*i] -= delta;
                    changed = true;
                }
            }
        }
        TerrainBrushMode::SetHeight => {
            for (i, w) in &hits {
                let t = (LERP_SPEED * strength * w * dt).clamp(0.0, 1.0);
                let h = terrain.heights[*i];
                let nh = h + (height - h) * t;
                if (nh - h).abs() > 1e-6 {
                    terrain.heights[*i] = nh;
                    changed = true;
                }
            }
        }
        TerrainBrushMode::Smooth => {
            let avg =
                hits.iter().map(|(i, _)| terrain.heights[*i]).sum::<f32>() / hits.len() as f32;
            for (i, w) in &hits {
                let t = (LERP_SPEED * strength * w * dt).clamp(0.0, 1.0);
                let h = terrain.heights[*i];
                let nh = h + (avg - h) * t;
                if (nh - h).abs() > 1e-6 {
                    terrain.heights[*i] = nh;
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Paint one texture-weight layer (RGBA = 4 layers) under the brush. The
/// selected layer approaches 1 and the others fade proportionally, so every
/// vertex stays normalized.
pub(crate) fn apply_texture_dab(
    terrain: &mut Terrain,
    center: Vec3,
    radius: f32,
    falloff: BrushFalloff,
    strength: f32,
    layer: usize,
    dt: f32,
) -> bool {
    let res = terrain.resolution as usize;
    let n = res + 1;
    if terrain.heights.len() != n * n || radius <= 0.0 || dt <= 0.0 || layer > 3 {
        return false;
    }
    if terrain.texture_weights.len() != n * n {
        terrain.texture_weights = vec![[1.0, 0.0, 0.0, 0.0]; n * n];
    }
    let step = terrain.size / terrain.resolution as f32;
    let half = terrain.size * 0.5;
    let x0 = (((center.x - radius + half) / step).floor() as i64).clamp(0, res as i64) as usize;
    let x1 = (((center.x + radius + half) / step).ceil() as i64).clamp(0, res as i64) as usize;
    let z0 = (((center.z - radius + half) / step).floor() as i64).clamp(0, res as i64) as usize;
    let z1 = (((center.z + radius + half) / step).ceil() as i64).clamp(0, res as i64) as usize;

    let strength = strength.clamp(0.0, 1.0);
    let dt = dt.min(0.1);
    let mut changed = false;
    for gz in z0..=z1 {
        for gx in x0..=x1 {
            let px = gx as f32 * step - half;
            let pz = gz as f32 * step - half;
            let d = ((px - center.x).powi(2) + (pz - center.z).powi(2)).sqrt();
            let w = falloff.weight(d, radius);
            if w <= 0.0 {
                continue;
            }
            let t = (LERP_SPEED * strength * w * dt).clamp(0.0, 1.0);
            if t <= 0.0 {
                continue;
            }
            let idx = gz * n + gx;
            let mut tw = terrain.texture_weights[idx];
            let before = tw;
            for (k, v) in tw.iter_mut().enumerate() {
                if k == layer {
                    *v = *v + (1.0 - *v) * t;
                } else {
                    *v *= 1.0 - t;
                }
            }
            let sum: f32 = tw.iter().copied().sum();
            if sum > 1e-6 {
                for v in &mut tw {
                    *v /= sum;
                }
            } else {
                tw = [1.0, 0.0, 0.0, 0.0];
            }
            if tw
                .iter()
                .zip(before)
                .any(|(a, b)| (*a - b).abs() > 1e-6)
            {
                terrain.texture_weights[idx] = tw;
                changed = true;
            }
        }
    }
    changed
}

fn mode_color(mode: TerrainBrushMode) -> egui::Color32 {
    match mode {
        TerrainBrushMode::Raise => egui::Color32::from_rgb(255, 150, 90),
        TerrainBrushMode::Lower => egui::Color32::from_rgb(170, 190, 220),
        TerrainBrushMode::SetHeight => egui::Color32::from_rgb(250, 210, 90),
        TerrainBrushMode::Smooth => egui::Color32::from_rgb(120, 220, 140),
    }
}

fn texture_layer_color(layer: usize) -> egui::Color32 {
    match layer {
        0 => egui::Color32::from_rgb(255, 120, 120),
        1 => egui::Color32::from_rgb(130, 230, 130),
        2 => egui::Color32::from_rgb(120, 170, 255),
        _ => egui::Color32::from_rgb(250, 220, 120),
    }
}

/// Scene-view input and overlay of the sculptor on object `i` (the selection).
/// Returns true when it took the viewport click (no object picking).
pub(crate) fn viewport(
    ui: &mut egui::Ui,
    scene: &mut LoadedScene,
    i: usize,
    view: &View,
    response: &egui::Response,
) -> bool {
    let ctx = ui.ctx().clone();
    let mut tool = TerrainPaintTool::get(&ctx);
    if terrain(scene, i).is_none() || tool.object != Some(scene.objects[i].id) {
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
        let (smaller, larger) = ui.input(|inp| {
            (
                inp.key_pressed(egui::Key::OpenBracket),
                inp.key_pressed(egui::Key::CloseBracket),
            )
        });
        if smaller {
            tool.radius /= 1.15;
        }
        if larger {
            tool.radius *= 1.15;
        }
    }
    let (mods, scroll, primary_down) = ui.input(|inp| {
        (
            inp.modifiers,
            inp.raw_scroll_delta.y,
            inp.pointer.primary_down(),
        )
    });
    if mods.command && scroll != 0.0 && response.contains_pointer() {
        tool.radius *= 1.1f32.powf(scroll / 50.0);
    }
    tool.radius = tool.radius.clamp(0.05, 1000.0);

    let painter = ui.painter_at(view.rect);
    let cursor = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos())
        .filter(|c| view.rect.contains(*c));

    let mode = if mods.alt && !tool.texture_mode {
        TerrainBrushMode::Smooth
    } else if mods.shift && !tool.texture_mode {
        match tool.mode {
            TerrainBrushMode::Raise => TerrainBrushMode::Lower,
            TerrainBrushMode::Lower => TerrainBrushMode::Raise,
            other => other,
        }
    } else {
        tool.mode
    };

    let world = scene.world_transform(i);
    let (positions, indices) = heightfield(terrain(scene, i).expect("checked above"));
    let hit = cursor.and_then(|c| {
        let (origin, dir) = view.ray(c)?;
        let inv = world.inverse();
        let (o, d) = (inv.transform_point3(origin), inv.transform_vector3(dir));
        let (_, tri, bary) = ray_triangles(o, d, &positions, &indices)?;
        let a = indices[tri * 3] as usize;
        let b = indices[tri * 3 + 1] as usize;
        let c = indices[tri * 3 + 2] as usize;
        let p = positions[a] * bary.x + positions[b] * bary.y + positions[c] * bary.z;
        let normal = (positions[b] - positions[a])
            .cross(positions[c] - positions[a])
            .normalize_or(Vec3::Y);
        Some((p, normal))
    });

    let scale = max_scale(world).max(1e-6);
    let radius = tool.radius / scale;
    if !tool.stroking && primary_down && response.is_pointer_button_down_on() {
        tool.stroking = true;
    }
    let dt = ui.input(|inp| inp.stable_dt).min(0.1);
    let mut painted = false;
    if tool.stroking && !primary_down {
        tool.stroking = false;
    } else if tool.stroking
        && let Some((p, _)) = &hit
        && let Some(t) = terrain_mut(scene, i) {
            painted = if tool.texture_mode {
                apply_texture_dab(
                    t,
                    *p,
                    radius,
                    tool.falloff,
                    tool.strength,
                    tool.texture_layer as usize,
                    dt,
                )
            } else {
                apply_dab(
                    t,
                    *p,
                    radius,
                    mode,
                    tool.falloff,
                    tool.strength,
                    tool.height,
                    dt,
                )
            };
        }
    if painted {
        ctx.request_repaint();
    }

    if let Some((p, normal)) = &hit {
        let wp = world.transform_point3(*p);
        let n = world.transform_vector3(*normal).normalize_or(Vec3::Y);
        brush_circle(
            &painter,
            view,
            wp,
            n,
            tool.radius,
            egui::Stroke::new(
                1.5_f32,
                if tool.texture_mode {
                    texture_layer_color(tool.texture_layer as usize)
                } else {
                    mode_color(mode)
                },
            ),
        );
        if let Some(c) = view.project(wp) {
            painter.circle_filled(
                c,
                2.0,
                if tool.texture_mode {
                    texture_layer_color(tool.texture_layer as usize)
                } else {
                    mode_color(mode)
                },
            );
        }
    }

    if tool.texture_mode {
        let labels = ["R", "G", "B", "A"];
        hint_bar(
            &painter,
            view.rect,
            &format!(
                "Paint terrain texture: layer {}   radius {:.2} m   Drag paint   [ ] or Ctrl+Scroll size   P / Esc done",
                labels[tool.texture_layer as usize],
                tool.radius
            ),
        );
    } else {
        hint_bar(
            &painter,
            view.rect,
            &format!(
                "Sculpt terrain: {}   radius {:.2} m   Drag sculpt   Shift reverse   Alt smooth   [ ] or Ctrl+Scroll size   P / Esc done",
                mode.label(),
                tool.radius
            ),
        );
    }
    if tool != before {
        tool.set(&ctx);
    }
    response.clicked() || tool.stroking
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn flat(resolution: u32, size: f32) -> Terrain {
        let n = (resolution as usize + 1).pow(2);
        Terrain {
            size,
            resolution,
            heights: vec![0.0; n],
            material: String::new(),
            texture_weights: Vec::new(),
            ..Default::default()
        }
    }

    #[test]
    fn the_brush_raises_and_lowers_the_grid() {
        let mut t = flat(4, 20.0);
        let center = 2 * 5 + 2;
        apply_dab(
            &mut t,
            Vec3::ZERO,
            3.0,
            TerrainBrushMode::Raise,
            BrushFalloff::Constant,
            1.0,
            0.0,
            0.1,
        );
        assert!(t.heights[center] > 0.0, "raise lifts the centre vertex");
        assert_eq!(t.heights[0], 0.0, "the far corner stays put");
        let raised = t.heights[center];
        apply_dab(
            &mut t,
            Vec3::ZERO,
            3.0,
            TerrainBrushMode::Lower,
            BrushFalloff::Constant,
            1.0,
            0.0,
            0.1,
        );
        assert!(t.heights[center] < raised, "lower digs back down");
    }

    #[test]
    fn set_height_converges_toward_the_target() {
        let mut t = flat(8, 40.0);
        for _ in 0..10 {
            apply_dab(
                &mut t,
                Vec3::ZERO,
                10.0,
                TerrainBrushMode::SetHeight,
                BrushFalloff::Constant,
                1.0,
                5.0,
                0.1,
            );
        }
        let mid = t.heights[4 * 9 + 4];
        assert!((mid - 5.0).abs() < 0.1, "centre {mid} converges to 5");
    }

    #[test]
    fn the_cursor_ray_finds_the_terrain_surface() {
        let mut t = flat(2, 10.0);
        t.heights[4] = 2.0;
        let (positions, indices) = heightfield(&t);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let view = View::new(
            Mat4::look_at_rh(Vec3::new(0.0, 4.0, 12.0), Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
            Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0),
            rect,
        );
        let (origin, dir) = view.ray(rect.center()).expect("ray through the viewport");
        let (t, _, _) = ray_triangles(origin, dir, &positions, &indices).expect("hits the field");
        assert!(t > 0.0);
    }

    #[test]
    fn texture_brush_paints_a_selected_layer_and_keeps_weights_normalized() {
        let mut t = flat(8, 40.0);
        assert!(apply_texture_dab(
            &mut t,
            Vec3::ZERO,
            10.0,
            BrushFalloff::Constant,
            1.0,
            2,
            0.1,
        ));
        let n = t.resolution as usize + 1;
        let mid = t.texture_weights[4 * n + 4];
        assert!(mid[2] > 0.4, "paint raises target layer: {:?}", mid);
        let sum: f32 = mid.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weights stay normalized: {sum}");
    }
}
