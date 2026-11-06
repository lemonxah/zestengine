//! Offscreen model preview: an orbit camera and an egui widget showing a model
//! the renderer draws into its own texture (`FrameInput::model_preview`).
//!
//! The widget keeps the orbit state (drag rotates, wheel zooms, middle / right
//! drag pans, view presets glide to a framing) and hands the editor a
//! [`PreviewRequest`] each frame it is on screen; the editor renders only the
//! model's draws with that camera and gives the texture back
//! ([`ModelPreview::set_texture`]). A closed window stops asking, so nothing is
//! rendered and the renderer frees the target. First user: the rig Configure
//! window; the clip, transition and blend tree inspectors reuse it.
//!
//! All camera maths is in the model's own space (the avatar's joints, y up,
//! yaw 0 in front of it) and mapped through the object's world matrix once when
//! the camera is built, so scale and rotation of the instance never matter.

use egui::{Pos2, Rect, Sense, Vec2};
use glam::{Mat4, Vec3};

/// Vertical field of view of the preview camera.
pub const FOV_Y: f32 = 35.0 * std::f32::consts::PI / 180.0;
/// The orbit never looks straight down / up (the view basis stays defined).
pub const MAX_PITCH: f32 = 85.0 * std::f32::consts::PI / 180.0;
/// Closest and farthest zoom as a share of the framed distance.
pub const ZOOM_RANGE: (f32, f32) = (0.15, 4.0);

/// An orbit camera around `target`, in model space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    pub target: Vec3,
    /// Radians about the up axis; 0 looks at the model from +Z (its front).
    pub yaw: f32,
    /// Radians above the horizon.
    pub pitch: f32,
    pub dist: f32,
}

impl Orbit {
    /// Camera position.
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(sy * cp, sp, cy * cp) * self.dist
    }

    /// Orbit by a pointer drag in points (`speed` radians per point).
    pub fn rotate(&mut self, delta: Vec2, speed: f32) {
        self.yaw -= delta.x * speed;
        self.pitch = (self.pitch + delta.y * speed).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Zoom by wheel `scroll` points, kept inside `home_dist * ZOOM_RANGE`.
    pub fn zoom(&mut self, scroll: f32, home_dist: f32) {
        self.dist = (self.dist * (-scroll * 0.0015).exp()).clamp(home_dist * ZOOM_RANGE.0, home_dist * ZOOM_RANGE.1);
    }

    /// Pan the target by a pointer drag in points over a view `height` points tall.
    pub fn pan(&mut self, delta: Vec2, height: f32) {
        let per_point = 2.0 * self.dist * (FOV_Y * 0.5).tan() / height.max(1.0);
        let (right, up) = self.basis();
        self.target += (-right * delta.x + up * delta.y) * per_point;
    }

    /// Camera right and up vectors.
    pub fn basis(&self) -> (Vec3, Vec3) {
        let fwd = (self.target - self.eye()).normalize_or_zero();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        (right, right.cross(fwd))
    }

    /// Move a fraction `t` (0..1) of the way to `goal` (yaw along the short way round).
    pub fn approach(&mut self, goal: &Orbit, t: f32) {
        let t = t.clamp(0.0, 1.0);
        let tau = std::f32::consts::TAU;
        let dyaw = (goal.yaw - self.yaw + std::f32::consts::PI).rem_euclid(tau) - std::f32::consts::PI;
        self.yaw += dyaw * t;
        self.pitch += (goal.pitch - self.pitch) * t;
        self.dist += (goal.dist - self.dist) * t;
        self.target += (goal.target - self.target) * t;
    }

    /// Whether [`Self::approach`] has arrived at `goal` for all practical purposes.
    pub fn reached(&self, goal: &Orbit) -> bool {
        (self.target - goal.target).length() < 1e-3 * goal.dist.max(1.0)
            && (self.dist - goal.dist).abs() < 1e-3 * goal.dist.max(1.0)
            && (self.yaw - goal.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI < 1e-3
            && (self.yaw - goal.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI > -1e-3
            && (self.pitch - goal.pitch).abs() < 1e-3
    }

    /// World-space camera for an instance placed by `world`: view, projection
    /// (right-handed, depth 0..1 like the editor camera) and position.
    pub fn camera(&self, world: Mat4, aspect: f32) -> (Mat4, Mat4, Vec3) {
        let scale = world.transform_vector3(Vec3::X).length().max(1e-4);
        let eye = world.transform_point3(self.eye());
        let target = world.transform_point3(self.target);
        let up = world.transform_vector3(Vec3::Y).normalize_or(Vec3::Y);
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh(FOV_Y, aspect.max(0.05), 0.02 * scale * self.dist.max(0.05), 200.0 * scale * self.dist.max(1.0));
        (view, proj, eye)
    }
}

/// The orbit that frames `points` (model space) from `yaw` / `pitch`: centred on
/// their bounding box, far enough for the bounding sphere to fit the shorter
/// side of a view of `aspect` (width / height) with some margin.
pub fn frame_points(points: &[Vec3], yaw: f32, pitch: f32, aspect: f32) -> Option<Orbit> {
    let first = *points.first()?;
    let (lo, hi) = points.iter().fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let target = (lo + hi) * 0.5;
    let radius = points.iter().map(|p| (*p - target).length()).fold(0.0, f32::max).max(0.02);
    let half_fov = (FOV_Y * 0.5).min(((FOV_Y * 0.5).tan() * aspect.clamp(0.2, 5.0)).atan());
    Some(Orbit { target, yaw, pitch, dist: radius * 1.15 / half_fov.sin() })
}

/// Screen position (points, y down) of a world point under `view_proj` in a
/// view drawn into `rect`, with its NDC depth; `None` behind the camera.
pub fn project(view_proj: Mat4, rect: Rect, p: Vec3) -> Option<(Pos2, f32)> {
    let c = view_proj * p.extend(1.0);
    if c.w <= 1e-3 {
        return None;
    }
    let ndc = c.truncate() / c.w;
    Some((Pos2::new(rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(), rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height()), ndc.z))
}

/// The item of `items` nearest to `at` within `radius` points (ties go to the
/// nearer camera, i.e. the smaller depth); `None` when none is that close.
pub fn pick<T: Copy>(items: &[(T, Pos2, f32)], at: Pos2, radius: f32) -> Option<T> {
    items
        .iter()
        .map(|(t, p, depth)| (*t, p.distance(at), *depth))
        .filter(|(_, d, _)| *d <= radius)
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
        .map(|(t, ..)| t)
}

/// What the editor needs to render the preview this frame.
#[derive(Clone, Copy, Debug)]
pub struct PreviewRequest {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
    /// Pixel size of the target.
    pub size: [u32; 2],
}

/// Projects a model-space point into the widget's rect (what an overlay draws with).
pub struct Projector {
    view_proj: Mat4,
    world: Mat4,
    rect: Rect,
}

impl Projector {
    /// Screen position and depth of a model-space point.
    pub fn at(&self, p: Vec3) -> Option<(Pos2, f32)> {
        project(self.view_proj, self.rect, self.world.transform_point3(p))
    }

    /// The widget's screen rect.
    pub fn rect(&self) -> Rect {
        self.rect
    }
}

/// The preview widget: orbit state, the glide towards a preset, and the
/// request / texture exchange with the renderer.
pub struct ModelPreview {
    orbit: Orbit,
    goal: Orbit,
    /// Distance of the last framing: the zoom range is relative to it.
    home_dist: f32,
    /// The last framing handed to [`Self::go_to`] / [`Self::restore`].
    home: Orbit,
    texture: Option<(egui::TextureId, [f32; 2])>,
    request: Option<PreviewRequest>,
    /// The camera the texture on screen was rendered with (one frame behind).
    shown: Option<Mat4>,
    framed: bool,
}

impl Default for ModelPreview {
    fn default() -> Self {
        let orbit = Orbit { target: Vec3::new(0.0, 1.0, 0.0), yaw: 0.0, pitch: 0.1, dist: 3.0 };
        Self { orbit, goal: orbit, home_dist: 3.0, home: orbit, texture: None, request: None, shown: None, framed: false }
    }
}

impl ModelPreview {
    /// The renderer's texture for the preview, as `Renderer::model_preview_texture` reports it.
    pub fn set_texture(&mut self, texture: Option<(egui::TextureId, [f32; 2])>) {
        self.texture = texture;
    }

    /// Whether the renderer has produced a texture to show.
    pub fn has_texture(&self) -> bool {
        self.texture.is_some()
    }

    /// The camera to render this frame; `None` unless [`Self::ui`] ran this frame.
    pub fn take_request(&mut self) -> Option<PreviewRequest> {
        self.request.take()
    }

    /// Forget the framing and the texture (the window closed).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Glide to `goal` (a framing from [`frame_points`]); the first one snaps.
    pub fn go_to(&mut self, goal: Orbit) {
        self.home = goal;
        self.home_dist = goal.dist;
        self.goal = goal;
        if !self.framed {
            self.orbit = goal;
            self.framed = true;
        }
    }

    /// Jump to `orbit` (a saved view) with `home` as the framing the zoom range is relative to.
    pub fn restore(&mut self, home: Orbit, orbit: Orbit) {
        self.home = home;
        self.home_dist = home.dist;
        self.orbit = orbit;
        self.goal = orbit;
        self.framed = true;
    }

    /// The framing last handed to `go_to` / `restore`.
    pub fn home(&self) -> Orbit {
        self.home
    }

    /// The current orbit (tests, overlays).
    pub fn orbit(&self) -> Orbit {
        self.orbit
    }

    /// Show the preview in a `size` area of `ui` for an instance placed by
    /// `world`, handle the camera input, then call `overlay` with the painter
    /// (clipped to the area) and a [`Projector`] for drawing over the image.
    pub fn ui(&mut self, ui: &mut egui::Ui, size: Vec2, world: Mat4, overlay: impl FnOnce(&egui::Painter, &Projector, &egui::Response)) -> egui::Response {
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        if resp.dragged_by(egui::PointerButton::Primary) {
            self.orbit.rotate(resp.drag_delta(), 0.008);
            self.goal = self.orbit;
        } else if resp.dragged_by(egui::PointerButton::Middle) || resp.dragged_by(egui::PointerButton::Secondary) {
            self.orbit.pan(resp.drag_delta(), rect.height());
            self.goal = self.orbit;
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.orbit.zoom(scroll, self.home_dist);
                self.goal = self.orbit;
                // The wheel belongs to the preview: take it so the inspector's scroll area
                // around it does not scroll too. (Only mutates inside `input_mut`: calling
                // other context accessors there would deadlock.)
                ui.ctx().input_mut(|i| {
                    i.smooth_scroll_delta = Vec2::ZERO;
                    i.raw_scroll_delta = Vec2::ZERO;
                });
            }
        }
        if !self.orbit.reached(&self.goal) {
            self.orbit.approach(&self.goal, 1.0 - (-dt * 9.0).exp());
            ui.ctx().request_repaint();
        }
        let aspect = rect.width() / rect.height().max(1.0);
        let (view, proj, position) = self.orbit.camera(world, aspect);
        let ppp = ui.ctx().pixels_per_point();
        self.request = Some(PreviewRequest { view, proj, position, size: [(rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32] });
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, egui::CornerRadius::same(4), egui::Color32::from_rgb(12, 13, 17));
        if let Some((id, _)) = self.texture {
            painter.image(id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), egui::Color32::WHITE);
        }
        // The image on screen was rendered with last frame's camera: project the dots with it.
        let view_proj = self.shown.unwrap_or(proj * view);
        self.shown = Some(proj * view);
        overlay(&painter, &Projector { view_proj, world, rect }, &resp);
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(300.0, 400.0))
    }

    #[test]
    fn framing_fits_the_points_and_looks_at_their_centre() {
        let pts = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 1.8, 0.0)];
        let o = frame_points(&pts, 0.0, 0.1, 0.75).unwrap();
        assert!((o.target - Vec3::new(0.0, 0.9, 0.0)).length() < 1e-5);
        let (view, proj, _) = o.camera(Mat4::IDENTITY, 0.75);
        for p in pts {
            let (s, _) = project(proj * view, rect(), p).expect("in front of the camera");
            assert!(rect().contains(s), "{p:?} projects outside: {s:?}");
        }
        assert!(frame_points(&[], 0.0, 0.0, 1.0).is_none());
        // A single point still gets a sane distance.
        assert!(frame_points(&[Vec3::ONE], 0.0, 0.0, 1.0).unwrap().dist > 0.0);
    }

    #[test]
    fn the_target_projects_to_the_centre_of_the_view() {
        let o = Orbit { target: Vec3::new(0.3, 1.0, -0.2), yaw: 0.7, pitch: 0.3, dist: 2.5 };
        let world = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), glam::Quat::from_rotation_y(1.1), Vec3::new(4.0, 0.0, 1.0));
        let (view, proj, _) = o.camera(world, 0.75);
        let (s, d) = project(proj * view, rect(), world.transform_point3(o.target)).unwrap();
        assert!((s - rect().center()).length() < 0.5, "{s:?}");
        assert!(d > 0.0 && d < 1.0);
        assert!(project(proj * view, rect(), world.transform_point3(o.eye() + (o.eye() - o.target))).is_none(), "behind the camera");
    }

    #[test]
    fn yaw_zero_looks_at_the_front_and_pitch_lifts_the_camera() {
        let o = Orbit { target: Vec3::ZERO, yaw: 0.0, pitch: 0.0, dist: 2.0 };
        assert!((o.eye() - Vec3::new(0.0, 0.0, 2.0)).length() < 1e-5);
        let up = Orbit { pitch: 1.0, ..o };
        assert!(up.eye().y > 1.5);
    }

    #[test]
    fn orbit_and_zoom_are_clamped() {
        let mut o = Orbit { target: Vec3::ZERO, yaw: 0.0, pitch: 0.0, dist: 3.0 };
        o.rotate(Vec2::new(0.0, 100_000.0), 0.01);
        assert!((o.pitch - MAX_PITCH).abs() < 1e-6);
        o.rotate(Vec2::new(0.0, -200_000.0), 0.01);
        assert!((o.pitch + MAX_PITCH).abs() < 1e-6);
        o.zoom(1.0e6, 3.0);
        assert!((o.dist - 3.0 * ZOOM_RANGE.0).abs() < 1e-5);
        o.zoom(-1.0e6, 3.0);
        assert!((o.dist - 3.0 * ZOOM_RANGE.1).abs() < 1e-5);
        // Wheel up zooms in.
        let mut o = Orbit { dist: 3.0, ..o };
        o.zoom(100.0, 3.0);
        assert!(o.dist < 3.0);
    }

    #[test]
    fn panning_moves_the_target_across_the_screen_plane() {
        let mut o = Orbit { target: Vec3::ZERO, yaw: 0.0, pitch: 0.0, dist: 2.0 };
        o.pan(Vec2::new(-50.0, 0.0), 400.0);
        assert!(o.target.x > 0.0 && o.target.y.abs() < 1e-6 && o.target.z.abs() < 1e-6, "dragging left slides the scene left: {:?}", o.target);
        o.pan(Vec2::new(0.0, 40.0), 400.0);
        assert!(o.target.y > 0.0);
    }

    #[test]
    fn approach_glides_the_short_way_round_and_arrives() {
        let mut o = Orbit { target: Vec3::ZERO, yaw: 3.0, pitch: 0.0, dist: 1.0 };
        let goal = Orbit { target: Vec3::new(1.0, 1.0, 0.0), yaw: -3.0, pitch: 0.5, dist: 2.0 };
        o.approach(&goal, 0.5);
        assert!(o.yaw > 3.0, "3 to -3 radians is 0.28 rad through pi, not 6 back: {}", o.yaw);
        for _ in 0..200 {
            o.approach(&goal, 0.2);
        }
        assert!(o.reached(&goal));
    }

    #[test]
    fn picking_takes_the_nearest_dot_within_the_radius_and_the_nearer_camera_on_ties() {
        let items = [(1, Pos2::new(100.0, 100.0), 0.5), (2, Pos2::new(108.0, 100.0), 0.5), (3, Pos2::new(100.0, 100.0), 0.2)];
        assert_eq!(pick(&items, Pos2::new(106.0, 100.0), 9.0), Some(2));
        assert_eq!(pick(&items, Pos2::new(100.0, 100.0), 9.0), Some(3), "same spot: the one in front");
        assert_eq!(pick(&items, Pos2::new(140.0, 100.0), 9.0), None);
        assert_eq!(pick::<i32>(&[], Pos2::ZERO, 9.0), None);
    }

    #[test]
    fn the_request_follows_the_widget_size_and_a_closed_widget_asks_for_nothing() {
        let ctx = egui::Context::default();
        let mut preview = ModelPreview::default();
        preview.go_to(Orbit { target: Vec3::new(0.0, 1.0, 0.0), yaw: 0.0, pitch: 0.0, dist: 3.0 });
        let _ = ctx.run(egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 800.0))), ..Default::default() }, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                preview.ui(ui, Vec2::new(300.0, 400.0), Mat4::IDENTITY, |_, p, _| {
                    let (s, _) = p.at(Vec3::new(0.0, 1.0, 0.0)).unwrap();
                    assert!((s - p.rect().center()).length() < 0.5);
                });
            });
        });
        let req = preview.take_request().expect("asked for a render");
        assert_eq!(req.size, [300, 400]);
        assert!(preview.take_request().is_none(), "a frame without ui() asks for nothing");
    }
}
