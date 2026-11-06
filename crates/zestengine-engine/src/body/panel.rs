//! The body menu: a world-space panel in VR (pointed at with either
//! controller's laser and the trigger, or poked with a controller) and the
//! same contents as an editor window. Height, guided calibration with prompts
//! and countdowns, tracker roles, and a live avatar preview with the trackers
//! drawn on it.

use zestengine_core::{TRACKER_SLOTS, TrackerTargets, VrRig, role_name};
use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use glam::{Mat4, Quat, Vec3};

use super::{BodyTracking, Flow, Step};
use crate::game_ui::{UiRay, ray_canvas_hit};
use crate::scene::{AvatarPose, LoadedScene};

const PANEL_PX: [u32; 2] = [1024, 720];
/// Panel size in tracking-space meters.
const PANEL_M: [f32; 2] = [0.8, 0.5625];
/// Pixels per egui point on the panel (big, legible type in a headset).
const ZOOM: f32 = 1.5;
/// A controller this close to the panel (m) pokes it.
const POKE: f32 = 0.025;

const ACCENT: Color32 = Color32::from_rgb(120, 230, 160);
const DIM: Color32 = Color32::from_rgb(150, 156, 170);
const BONE: Color32 = Color32::from_rgb(200, 205, 215);

/// Tracker slot colors (head, hands, then body roles).
fn slot_color(slot: usize) -> Color32 {
    match slot {
        0 => Color32::from_rgb(255, 255, 255),
        1 | 2 => Color32::from_rgb(110, 180, 255),
        3 | 6 => Color32::from_rgb(255, 200, 90),
        4 | 5 => Color32::from_rgb(120, 230, 160),
        7 | 8 => Color32::from_rgb(230, 130, 255),
        _ => Color32::from_rgb(255, 130, 120),
    }
}

/// Role choices a device cycles through: its own, each body role, ignored.
const ROLE_CHOICES: [Option<Option<usize>>; 10] =
    [None, Some(Some(3)), Some(Some(4)), Some(Some(5)), Some(Some(6)), Some(Some(7)), Some(Some(8)), Some(Some(9)), Some(Some(10)), Some(None)];

fn choice_label(c: Option<Option<usize>>, native: Option<usize>) -> String {
    match c {
        None => match native {
            Some(n) => format!("{} (own)", role_name(n)),
            None => "Unassigned".into(),
        },
        Some(Some(s)) => role_name(s).into(),
        Some(None) => "Ignored".into(),
    }
}

/// What the panel draws besides the body state.
pub(crate) struct PanelView {
    pose: Option<AvatarPose>,
    /// Live trackers (slot, world position).
    trackers: Vec<(usize, Vec3)>,
    /// The viewer's floor point and flat facing (front view of the preview).
    center: Vec3,
    fwd: Vec3,
}

impl PanelView {
    pub(crate) fn new(scene: &LoadedScene, rig: &VrRig, raw: &TrackerTargets) -> Self {
        let trackers: Vec<(usize, Vec3)> =
            raw.slots().iter().enumerate().filter_map(|(k, p)| p.map(|p| (k, rig.stage_to_world(p.0)))).collect();
        let (center, fwd) = match raw.head {
            Some(h) => {
                let (p, r) = rig.stage_pose_to_world(h);
                let f = r * Vec3::NEG_Z;
                (Vec3::new(p.x, rig.origin.y, p.z), Vec3::new(f.x, 0.0, f.z).normalize_or(Vec3::NEG_Z))
            }
            None => (rig.origin, rig.stage_rot_to_world(Quat::IDENTITY) * Vec3::NEG_Z),
        };
        let (center, fwd) = scene.calibration_pose.as_ref().and_then(|(id, _)| scene.index_of(*id)).map(|i| {
            let w = scene.world_transform(i);
            (w.w_axis.truncate(), w.transform_vector3(Vec3::Z).with_y(0.0).normalize_or(fwd))
        }).unwrap_or((center, fwd));
        Self { pose: scene.avatar_preview.clone(), trackers, center, fwd }
    }
}

struct PanelGpu {
    target: zestengine_render::TextureHandle,
    material: zestengine_render::MaterialHandle,
    quad: zestengine_render::MeshHandle,
    laser: zestengine_render::MeshHandle,
    laser_material: zestengine_render::MaterialHandle,
}

/// Render target, egui context and placement of the VR body panel.
#[derive(Default)]
pub struct BodyPanel {
    gpu: Option<PanelGpu>,
    ctx: egui::Context,
    /// World transform of the panel's center (uniformly scaled with the player).
    anchor: Option<Mat4>,
    down: bool,
    last_pos: Option<Pos2>,
    time: f64,
}

fn style(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    ctx.style_mut(|s| {
        use egui::TextStyle::*;
        s.text_styles.insert(Body, FontId::proportional(17.0));
        s.text_styles.insert(Button, FontId::proportional(18.0));
        s.text_styles.insert(Heading, FontId::proportional(26.0));
        s.text_styles.insert(Small, FontId::proportional(13.0));
        s.spacing.button_padding = Vec2::new(12.0, 8.0);
        s.spacing.item_spacing = Vec2::new(8.0, 8.0);
        s.spacing.interact_size.y = 34.0;
    });
}

fn create_gpu(renderer: &mut zestengine_render::Renderer) -> anyhow::Result<PanelGpu> {
    let target = renderer.create_ui_target(PANEL_PX[0], PANEL_PX[1])?;
    let material = crate::game_ui::unlit_material(renderer, Some(target), "body panel")?;
    let quad = renderer.upload_mesh(&strip_mesh(-0.5, [1.0; 4], [1.0; 4]))?;
    let laser = renderer.upload_mesh(&strip_mesh(0.0, [0.55, 0.9, 1.0, 0.85], [0.55, 0.9, 1.0, 0.05]))?;
    let laser_material = crate::game_ui::unlit_material(renderer, None, "body panel laser")?;
    Ok(PanelGpu { target, material, quad, laser, laser_material })
}

/// A unit quad in local XY facing +Z, from `y0` to `y0 + 1`, colored from
/// `bottom` to `top` (uv (0,0) at the top-left).
fn strip_mesh(y0: f32, bottom: [f32; 4], top: [f32; 4]) -> zestengine_render::MeshData {
    let v = |x: f32, y: f32, u: f32, w: f32, color: [f32; 4]| zestengine_render::Vertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [u, w],
        color,
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv1: [u, w],
        joints: [0; 4],
        weights: [0.0; 4],
    };
    zestengine_render::MeshData {
        vertices: vec![
            v(-0.5, y0 + 1.0, 0.0, 0.0, top),
            v(0.5, y0 + 1.0, 1.0, 0.0, top),
            v(0.5, y0, 1.0, 1.0, bottom),
            v(-0.5, y0, 0.0, 1.0, bottom),
        ],
        indices: vec![0, 2, 1, 0, 3, 2],
        has_lightmap_uv: false,
    }
}

fn draw(mesh: zestengine_render::MeshHandle, material: zestengine_render::MaterialHandle, transform: Mat4, radius: f32) -> zestengine_render::DrawCmd {
    zestengine_render::DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform,
        highlight: 0.0,
        mesh_center: Vec3::ZERO,
        bound_radius: radius,
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index: -1,
        occluded: false,
        shadow_casting: zestengine_render::ShadowCasting::Off,
        instance_color: None,
        instance_param: None,
    }
}

impl BodyTracking {
    /// The VR body panel for this frame: lays it out in front of the player
    /// when it opens, takes the controllers' pointing, queues its quad and
    /// lasers on `scene.extra_draws` and returns its UI draw. Call after the
    /// skinning update (the preview reads the posed avatar) and after the
    /// host cleared `extra_draws`.
    pub fn panel_frame(
        &mut self,
        renderer: &mut zestengine_render::Renderer,
        scene: &mut LoadedScene,
        rig: &VrRig,
        dt: f32,
    ) -> Option<zestengine_render::WorldUiDraw> {
        let raw = self.last_raw;
        if !self.panel_visible() || raw.head.is_none() {
            self.panel.anchor = None;
            if !self.window_open {
                scene.avatar_preview_for = None;
            }
            return None;
        }
        scene.avatar_preview_for = scene.local_avatar().map(|a| a.index);
        let mut panel = std::mem::take(&mut self.panel);
        if panel.gpu.is_none() {
            match create_gpu(renderer) {
                Ok(g) => {
                    style(&panel.ctx);
                    panel.gpu = Some(g);
                }
                Err(e) => {
                    tracing::error!("body panel: {e:#}");
                    self.panel = panel;
                    return None;
                }
            }
        }
        let scale = rig.scale.max(1e-3);
        let anchor = *panel.anchor.get_or_insert_with(|| {
            let (hp, hr) = raw.head.map(|h| rig.stage_pose_to_world(h)).unwrap_or((rig.origin, Quat::IDENTITY));
            let f = hr * Vec3::NEG_Z;
            let fwd = Vec3::new(f.x, 0.0, f.z).normalize_or(Vec3::NEG_Z);
            let pos = hp + fwd * 0.75 * scale - Vec3::Y * 0.22 * scale;
            Mat4::from_scale_rotation_translation(Vec3::splat(scale), Quat::from_rotation_y((-fwd.x).atan2(-fwd.z)), pos)
        });

        // Pointing: a poke wins, else the nearest laser hit.
        let hands: Vec<(usize, Vec3, Quat)> = [raw.left_hand, raw.right_hand]
            .iter()
            .enumerate()
            .filter_map(|(k, h)| h.map(|h| rig.stage_pose_to_world(h)).map(|(p, r)| (k, p, r)))
            .collect();
        let inv = anchor.inverse();
        let to_px = |l: Vec3| Pos2::new((l.x / PANEL_M[0] + 0.5) * PANEL_PX[0] as f32, (0.5 - l.y / PANEL_M[1]) * PANEL_PX[1] as f32);
        let inside = |l: Vec3| l.x.abs() <= PANEL_M[0] * 0.5 && l.y.abs() <= PANEL_M[1] * 0.5;
        let mut pointer: Option<(Pos2, bool)> = None;
        let mut best = f32::INFINITY;
        let mut laser_ends = Vec::new();
        for &(k, p, r) in &hands {
            let local = inv.transform_point3(p);
            if inside(local) && local.z.abs() < POKE {
                pointer = Some((to_px(local), true));
                best = -1.0;
            }
            let dir = (r * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
            let ray = UiRay { origin: p, dir, down: self.triggers_prev[k], pressed: false, released: false };
            let hit = ray_canvas_hit(&ray, anchor, PANEL_M, PANEL_PX);
            let len = hit.map(|h| h.1).unwrap_or(1.2 * scale);
            laser_ends.push((p, dir, len));
            if let Some((px, d)) = hit
                && d < best
            {
                best = d;
                pointer = Some((px, self.triggers_prev[k]));
            }
        }

        let to_points = |p: Pos2| Pos2::new(p.x / ZOOM, p.y / ZOOM);
        let mut events = Vec::new();
        match pointer {
            Some((px, down)) => {
                let pos = to_points(px);
                events.push(egui::Event::PointerMoved(pos));
                if down != panel.down {
                    events.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: down, modifiers: Default::default() });
                }
                panel.down = down;
                panel.last_pos = Some(pos);
            }
            None => {
                if panel.down
                    && let Some(pos) = panel.last_pos
                {
                    events.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
                }
                events.push(egui::Event::PointerGone);
                panel.down = false;
            }
        }
        panel.time += dt as f64;
        let mut input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(PANEL_PX[0] as f32 / ZOOM, PANEL_PX[1] as f32 / ZOOM))),
            events,
            time: Some(panel.time),
            ..Default::default()
        };
        input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ZOOM);
        let view = PanelView::new(scene, rig, &raw);
        let out = panel.ctx.run(input, |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::new().fill(Color32::from_rgba_unmultiplied(22, 24, 30, 240)).inner_margin(egui::Margin::same(18)))
                .show(ctx, |ui| self.panel_ui(ui, &view));
        });
        let primitives = panel.ctx.tessellate(out.shapes, out.pixels_per_point);
        let Some(gpu) = panel.gpu.as_ref() else {
            self.panel = panel;
            return None;
        };
        let ui_draw = zestengine_render::WorldUiDraw {
            target: gpu.target,
            extent: PANEL_PX,
            draw: zestengine_render::EguiDraw { pixels_per_point: out.pixels_per_point, primitives, textures_delta: out.textures_delta },
        };
        scene.extra_draws.push(draw(gpu.quad, gpu.material, anchor * Mat4::from_scale(Vec3::new(PANEL_M[0], PANEL_M[1], 1.0)), 0.75));
        let eye = raw.head.map(|h| rig.stage_to_world(h.0)).unwrap_or(rig.origin);
        for (origin, dir, len) in laser_ends {
            let to_eye = eye - origin;
            let normal = (to_eye - dir * to_eye.dot(dir)).normalize_or(Vec3::Y.cross(dir).normalize_or(Vec3::X));
            let side = dir.cross(normal).normalize_or(Vec3::X);
            let width = 0.004 * scale;
            let m = Mat4::from_cols((side * width).extend(0.0), (dir * len).extend(0.0), normal.extend(0.0), origin.extend(1.0));
            scene.extra_draws.push(draw(gpu.laser, gpu.laser_material, m, 0.75));
        }
        self.panel = panel;
        Some(ui_draw)
    }

    /// The editor's desktop Body Tracking window (also reachable on the
    /// editor's VR panels): the same contents as the VR panel.
        pub fn window(&mut self, ctx: &egui::Context, scene: &mut LoadedScene, rig: &VrRig) {
        if !self.window_open {
            return;
        }
        scene.avatar_preview_for = scene.local_avatar().map(|a| a.index);
        let view = PanelView::new(scene, rig, &self.last_raw);
        let mut open = self.window_open;
        egui::Window::new("Body Tracking").open(&mut open).default_width(640.0).show(ctx, |ui| self.panel_ui(ui, &view));
        self.window_open = open;
        if !open && !self.panel_visible() {
            scene.avatar_preview_for = None;
        }
    }

    /// The panel contents: calibration prompts while a step runs, else the
    /// menu (height, calibration, trackers) beside the avatar preview.
    pub(crate) fn panel_ui(&mut self, ui: &mut egui::Ui, view: &PanelView) {
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(380.0);
                if self.calibration.is_some() {
                    self.prompt_ui(ui);
                } else {
                    self.menu_ui(ui);
                }
            });
            ui.separator();
            let h = ui.available_height().clamp(200.0, 420.0);
            preview(ui, view, Vec2::new(ui.available_width().max(160.0), h));
        });
    }

    fn prompt_ui(&mut self, ui: &mut egui::Ui) {
        let Some(cal) = self.calibration.as_ref() else { return };
        let total = match cal.flow {
            Flow::Full => 3,
            Flow::Align | Flow::Height | Flow::Quick => 1,
        };
        let (n, title, text, count, progress) = match &cal.step {
            Step::Align(_) => (1, "Step into your avatar", "Match the T-pose in the mirror. Align the tracker markers, release both triggers, then squeeze both triggers together to accept.", None, None),
            Step::StandCountdown(t) => (1, "Stand up straight", "Stand tall, feet flat, and look straight ahead.", Some(*t), None),
            Step::Standing(s) => (1, "Hold still", "Measuring your height...", None, Some(s.progress())),
            Step::TPoseCountdown(t) => (
                if cal.flow == Flow::Full { 2 } else { 1 },
                "T-pose",
                "Stand straight, arms out to the sides at shoulder height, palms down.",
                Some(*t),
                None,
            ),
            Step::TPoseHold(_) => (if cal.flow == Flow::Full { 2 } else { 1 }, "Hold it", "Capturing the T-pose...", None, None),
            Step::Floor => (
                3,
                "Floor check (optional)",
                "Put a controller flat on the floor and pull its trigger, or skip.",
                None,
                None,
            ),
        };
        let floor = matches!(cal.step, Step::Floor);
        ui.label(RichText::new(format!("Step {n} of {total}")).color(DIM));
        ui.label(RichText::new(title).size(30.0).strong().color(ACCENT));
        ui.label(RichText::new(text).size(19.0));
        ui.add_space(12.0);
        if let Some(t) = count {
            ui.label(RichText::new(format!("{}", t.ceil().max(1.0) as i32)).size(84.0).strong());
        }
        if let Some(p) = progress {
            ui.add(egui::ProgressBar::new(p).desired_height(18.0));
        }
        ui.add_space(12.0);
        if let Some(status) = &self.status { ui.label(RichText::new(status).color(ACCENT)); }
        ui.label(format!("Tracked points: {}/11", self.last_raw.slots().iter().flatten().count()));
        ui.horizontal(|ui| {
            if floor && big_button(ui, "Skip").clicked() {
                self.skip_floor();
            }
            if big_button(ui, "Cancel").clicked() {
                self.cancel();
            }
        });
    }

    fn menu_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Body").size(28.0).strong());
        if let Some(msg) = &self.status {
            ui.label(RichText::new(msg).color(ACCENT));
        }
        let source = if self.profile.height_override_cm.is_some() {
            "entered"
        } else if self.profile.measured_eye_height.is_some() {
            "measured"
        } else {
            "default"
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Height {:.0} cm", self.profile.height() * 100.0)).size(22.0));
            ui.label(RichText::new(source).color(DIM));
        });
        ui.horizontal(|ui| {
            let cm = (self.profile.height() * 100.0).round();
            if big_button(ui, "-1 cm").clicked() {
                self.command(zestengine_core::BodyCommand::SetHeight(Some(cm - 1.0)));
            }
            if big_button(ui, "+1 cm").clicked() {
                self.command(zestengine_core::BodyCommand::SetHeight(Some(cm + 1.0)));
            }
            if self.profile.height_override_cm.is_some() && big_button(ui, "Use measured").clicked() {
                self.command(zestengine_core::BodyCommand::SetHeight(None));
            }
        });
        let avatar = match self.avatar {
            Some(m) => format!(
                "{}: avatar {:.2}x, play space {:.2}x (avatar eyes {:.2} model units)",
                self.mode.label(),
                self.fit.avatar,
                self.fit.tracking,
                m.eye_height
            ),
            None => "No humanoid avatar to fit".into(),
        };
        ui.label(RichText::new(avatar).color(DIM).size(14.0));
        if let Some(span) = self.profile.arm_span {
            ui.label(RichText::new(format!("Arm span {:.0} cm", span * 100.0)).color(DIM).size(14.0));
        }
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if big_button(ui, "Calibrate avatar").on_hover_text("Step into your worn avatar in the T-pose mirror, then accept with both triggers").clicked() {
                self.command(zestengine_core::BodyCommand::AlignAvatar);
            }
            if big_button(ui, "Guided setup").on_hover_text("Measure height, assign trackers, check floor").clicked() {
                self.command(zestengine_core::BodyCommand::Calibrate);
            }
            if big_button(ui, "Quick recalibrate").on_hover_text("T-pose only, keeps the tracker roles").clicked() {
                self.command(zestengine_core::BodyCommand::QuickCalibrate);
            }
            if big_button(ui, "Height").on_hover_text("Measure your standing height").clicked() {
                self.command(zestengine_core::BodyCommand::MeasureHeight);
            }
            if big_button(ui, "Done").clicked() {
                self.menu_open = false;
            }
        });
        ui.add_space(6.0);
        ui.label(RichText::new(if self.calibrated() { "Trackers calibrated" } else { "Trackers not calibrated" }).color(DIM));
        self.trackers_ui(ui);
        ui.horizontal(|ui| {
            let mut auto = self.profile.auto_assign;
            if ui.checkbox(&mut auto, "Assign roles from the T-pose").changed() {
                self.profile.auto_assign = auto;
                self.save();
            }
        });
        ui.horizontal(|ui| {
            if ui.small_button("Forget calibration").clicked() {
                self.clear_calibration();
            }
            if ui.small_button("Reset everything").clicked() {
                self.command(zestengine_core::BodyCommand::Reset);
            }
        });
    }

    /// Detected trackers (and ones saved earlier) with role pickers.
    fn trackers_ui(&mut self, ui: &mut egui::Ui) {
        let mut rows: Vec<(String, String, Option<usize>, bool)> =
            self.devices.iter().map(|d| (d.id.clone(), d.label.clone(), d.native, d.pose.is_some())).collect();
        for r in &self.profile.roles {
            if !rows.iter().any(|row| row.0 == r.device) {
                rows.push((r.device.clone(), r.device.clone(), None, false));
            }
        }
        if rows.is_empty() {
            ui.label(RichText::new("No body trackers detected (runtime trackers or SlimeVR)").color(DIM));
            return;
        }
        let mut taken = [0u8; TRACKER_SLOTS];
        for (id, _, native, _) in &rows {
            if let Some(s) = self.profile.role_of(id).unwrap_or(*native) {
                taken[s.min(TRACKER_SLOTS - 1)] += 1;
            }
        }
        egui::ScrollArea::vertical().max_height(190.0).show(ui, |ui| {
            for (id, label, native, live) in rows {
                ui.horizontal(|ui| {
                    let (dot, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                    ui.painter().circle_filled(dot.center(), 5.0, if live { ACCENT } else { Color32::from_gray(80) });
                    ui.add_sized([150.0, 24.0], egui::Label::new(RichText::new(label).size(15.0)).truncate());
                    let current = self.profile.role_of(&id);
                    let at = ROLE_CHOICES.iter().position(|c| *c == current).unwrap_or(0);
                    let step = |d: isize| ROLE_CHOICES[(at as isize + d).rem_euclid(ROLE_CHOICES.len() as isize) as usize];
                    let mut pick = None;
                    if ui.button("<").clicked() {
                        pick = Some(step(-1));
                    }
                    let slot = current.unwrap_or(native);
                    let dup = slot.is_some_and(|s| taken[s.min(TRACKER_SLOTS - 1)] > 1);
                    let text = RichText::new(choice_label(current, native)).color(match slot {
                        Some(_) if dup => Color32::from_rgb(255, 120, 100),
                        Some(s) => slot_color(s),
                        None => DIM,
                    });
                    ui.add_sized([130.0, 24.0], egui::Label::new(text));
                    if ui.button(">").clicked() {
                        pick = Some(step(1));
                    }
                    if let Some(c) = pick {
                        match c {
                            None => self.profile.roles.retain(|r| r.device != id),
                            Some(role) => self.profile.set_role(&id, role),
                        }
                        self.save();
                    }
                });
            }
        });
    }
}

fn big_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(egui::Button::new(RichText::new(text).size(18.0)).min_size(Vec2::new(64.0, 40.0)))
}

/// Front (mirror) view of the posed avatar with the live trackers on it.
fn preview(ui: &mut egui::Ui, view: &PanelView, size: Vec2) {
    let (resp, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = resp.rect;
    painter.rect_filled(rect, 8.0, Color32::from_rgb(14, 15, 19));
    let right = view.fwd.cross(Vec3::Y).normalize_or(Vec3::X);
    let mut points: Vec<Vec3> = view.trackers.iter().map(|t| t.1).collect();
    if let Some(p) = &view.pose {
        points.extend(&p.joints);
    }
    let top = points.iter().map(|p| p.y - view.center.y).fold(1.0f32, f32::max) * 1.12;
    let half_width = points.iter().map(|p| (*p - view.center).dot(right).abs()).fold(0.5_f32, f32::max);
    let k = ((rect.height() - 24.0) / top.max(0.1)).min((rect.width() - 24.0) / (half_width * 2.2));
    let to_screen = |p: Vec3| {
        let rel = p - view.center;
        Pos2::new(rect.center().x + rel.dot(right) * k, rect.bottom() - 12.0 - rel.y * k)
    };
    painter.line_segment(
        [Pos2::new(rect.left() + 8.0, rect.bottom() - 12.0), Pos2::new(rect.right() - 8.0, rect.bottom() - 12.0)],
        Stroke::new(1.0_f32, Color32::from_gray(60)),
    );
    match &view.pose {
        Some(pose) => {
            // A front-on mirror of the actual worn mesh, with tracking markers drawn over it.
            // Painter order resolves depth without depending on lighting or a scene mirror.
            let mut triangles: Vec<_> = pose.triangles.iter().collect();
            triangles.sort_by(|a, b| {
                let depth = |t: &&[Vec3; 3]| (t[0] + t[1] + t[2]).dot(view.fwd);
                depth(a).total_cmp(&depth(b))
            });
            let mut mesh = egui::Mesh::default();
            for triangle in triangles {
                let n = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).normalize_or(Vec3::Y);
                let light = (0.45 + 0.45 * n.dot((-view.fwd + Vec3::Y).normalize()).abs()).clamp(0.0, 1.0);
                let color = Color32::from_rgb((light * 175.0) as u8, (light * 190.0) as u8, (light * 210.0) as u8);
                let base = mesh.vertices.len() as u32;
                for p in triangle { mesh.colored_vertex(to_screen(*p), color); }
                mesh.add_triangle(base, base + 1, base + 2);
            }
            painter.add(egui::Shape::mesh(mesh));
            for (j, parent) in pose.parents.iter().enumerate() {
                if let (Some(p), Some(a)) = (parent, pose.joints.get(j))
                    && let Some(b) = pose.joints.get(*p)
                {
                    painter.line_segment([to_screen(*a), to_screen(*b)], Stroke::new(2.0_f32, BONE));
                }
            }
        }
        None => {
            painter.text(rect.center_top() + Vec2::new(0.0, 16.0), Align2::CENTER_TOP, "No avatar", FontId::proportional(14.0), DIM);
        }
    }
    for &(slot, p) in &view.trackers {
        let s = to_screen(p);
        painter.circle_filled(s, 6.0, slot_color(slot));
        painter.circle_stroke(s, 6.0, Stroke::new(1.0_f32, Color32::BLACK));
        let short: String = role_name(slot).split(' ').map(|w| w.chars().next().unwrap_or(' ')).collect();
        painter.text(s + Vec2::new(9.0, 0.0), Align2::LEFT_CENTER, short, FontId::proportional(11.0), slot_color(slot));
    }
}
