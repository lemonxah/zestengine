//! The on-screen feedback while an asset is placed in the viewport: a hint chip near the
//! bottom of the viewport, a footprint ring and normal tick at the snap point, and a
//! loading label while a model is still being read. Painted with egui's painter in the
//! editor theme; `placement_glue` projects the 3D points to screen space.

use egui::{Color32, CornerRadius, Painter, Pos2, Rect, Stroke, pos2, vec2};
use glam::Vec3;
use crate::theme;

/// Everything `paint` needs, already in screen space.
pub struct Overlay {
    /// The viewport rect the overlay is clipped to.
    pub viewport: Rect,
    /// Closed polyline around the hit point on the surface plane.
    pub ring: Vec<Pos2>,
    /// Surface-normal tick: (hit point, tip).
    pub tick: Option<(Pos2, Pos2)>,
    /// Text next to the cursor while the model loads.
    pub loading: Option<(Pos2, String)>,
    /// "Align to surface" as it applies right now (setting XOR Alt).
    pub align: bool,
}

impl Overlay {
    pub fn new(viewport: Rect, align: bool) -> Self {
        Self { viewport, ring: Vec::new(), tick: None, loading: None, align }
    }
}

/// `n` points of a circle of `radius` around `center` lying in the plane with `normal`.
pub fn ring_points(center: Vec3, normal: Vec3, radius: f32, n: usize) -> Vec<Vec3> {
    let nrm = normal.normalize_or(Vec3::Y);
    let helper = if nrm.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let u = nrm.cross(helper).normalize();
    let v = nrm.cross(u);
    (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            center + (u * a.cos() + v * a.sin()) * radius
        })
        .collect()
}

pub fn hint_text(align: bool) -> String {
    format!("Release to place · Wheel/R rotate · Alt toggle align ({}) · Ctrl grid · Esc cancel", if align { "on" } else { "off" })
}

pub fn paint(painter: &Painter, o: &Overlay) {
    let painter = painter.with_clip_rect(o.viewport);
    let accent = theme::accent();
    if o.ring.len() > 2 {
        let mut pts = o.ring.clone();
        pts.push(o.ring[0]);
        painter.add(egui::Shape::line(pts, Stroke::new(3.0_f32, theme::with_alpha(Color32::BLACK, 0.35))));
        painter.add(egui::Shape::line(o.ring.iter().copied().chain(o.ring.first().copied()).collect(), Stroke::new(1.5_f32, accent)));
    }
    if let Some((a, b)) = o.tick {
        painter.line_segment([a, b], Stroke::new(1.5_f32, accent));
        painter.circle_filled(b, 2.5, accent);
    }
    if let Some((at, text)) = &o.loading {
        let galley = painter.layout_no_wrap(text.clone(), theme::mono(theme::SIZE_META), theme::TEXT_MUTED);
        let r = Rect::from_min_size(*at + vec2(14.0, 14.0), galley.size() + vec2(14.0, 8.0));
        painter.rect_filled(r, CornerRadius::same(theme::R_FIELD), theme::with_alpha(theme::BG_APP, 0.9));
        painter.galley(r.min + vec2(7.0, 4.0), galley, theme::TEXT_MUTED);
    }
    // Hint chip, bottom centre.
    let galley = painter.layout_no_wrap(hint_text(o.align), theme::mono(theme::SIZE_META), theme::TEXT_MUTED);
    let size = galley.size() + vec2(20.0, 10.0);
    let r = Rect::from_center_size(pos2(o.viewport.center().x, o.viewport.bottom() - 18.0 - size.y * 0.5), size);
    painter.rect_filled(r, CornerRadius::same(theme::R_FIELD), theme::with_alpha(theme::BG_APP, 0.88));
    painter.rect_stroke(r, CornerRadius::same(theme::R_FIELD), Stroke::new(1.0_f32, theme::LINE_FIELD), egui::StrokeKind::Inside);
    painter.galley(r.min + vec2(10.0, 5.0), galley, theme::TEXT_MUTED);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shot::Shot;

    #[test]
    fn ring_lies_in_the_surface_plane() {
        for n in [Vec3::Y, Vec3::X, Vec3::new(1.0, 2.0, -0.5).normalize(), Vec3::NEG_Y] {
            let c = Vec3::new(1.0, 2.0, 3.0);
            for p in ring_points(c, n, 0.7, 16) {
                assert!((p - c).dot(n).abs() < 1e-4);
                assert!(((p - c).length() - 0.7).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn overlay_shot() {
        let mut shot = Shot::new(900.0, 420.0, 1.5);
        let vp = Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 420.0));
        let c = pos2(450.0, 250.0);
        let ring: Vec<Pos2> = (0..32).map(|i| {
            let a = i as f32 / 32.0 * std::f32::consts::TAU;
            c + vec2(a.cos() * 90.0, a.sin() * 30.0)
        }).collect();
        let o = Overlay { viewport: vp, ring, tick: Some((c, c + vec2(0.0, -60.0))), loading: Some((c, "Loading hangar.fbx…".into())), align: true };
        let img = shot.run(|ctx| {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_PANEL)).show(ctx, |_| {});
            paint(&ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("ov"))), &o);
        });
        shot.save_named(&img, "placement_overlay");
        assert!(img.non_background() > 500);
    }
}
