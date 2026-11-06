//! The Avatar Mask silhouette: a humanoid body drawn from simple shapes, one
//! per mask part, that you click to switch the part on or off (Unity's mask
//! picture). The geometry and hit test are pure so they are tested; the
//! drawing is not verified visually.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use zestengine_assets::avatar_mask::{AvatarMask, MaskPart};
use crate::theme;

/// A shape in the unit square (x right, y down, the figure faces the viewer,
/// so the character's left side is on the right of the picture).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Shape {
    Circle { c: [f32; 2], r: f32 },
    Rect { min: [f32; 2], max: [f32; 2] },
}

impl Shape {
    fn contains(self, p: [f32; 2]) -> bool {
        match self {
            Shape::Circle { c, r } => (p[0] - c[0]).hypot(p[1] - c[1]) <= r,
            Shape::Rect { min, max } => p[0] >= min[0] && p[0] <= max[0] && p[1] >= min[1] && p[1] <= max[1],
        }
    }
}

/// Parts with their shapes. Small shapes come first so they win a hit where
/// they overlap a bigger part (the IK dots sit on the hands and feet).
pub(crate) const SILHOUETTE: [(MaskPart, Shape); 13] = [
    (MaskPart::LeftHandIk, Shape::Circle { c: [0.93, 0.235], r: 0.04 }),
    (MaskPart::RightHandIk, Shape::Circle { c: [0.07, 0.235], r: 0.04 }),
    (MaskPart::LeftFootIk, Shape::Circle { c: [0.56, 0.93], r: 0.045 }),
    (MaskPart::RightFootIk, Shape::Circle { c: [0.44, 0.93], r: 0.045 }),
    (MaskPart::Head, Shape::Circle { c: [0.5, 0.1], r: 0.075 }),
    (MaskPart::LeftFingers, Shape::Rect { min: [0.79, 0.19], max: [0.88, 0.28] }),
    (MaskPart::RightFingers, Shape::Rect { min: [0.12, 0.19], max: [0.21, 0.28] }),
    (MaskPart::LeftArm, Shape::Rect { min: [0.61, 0.2], max: [0.79, 0.27] }),
    (MaskPart::RightArm, Shape::Rect { min: [0.21, 0.2], max: [0.39, 0.27] }),
    (MaskPart::Body, Shape::Rect { min: [0.39, 0.19], max: [0.61, 0.5] }),
    (MaskPart::LeftLeg, Shape::Rect { min: [0.5, 0.5], max: [0.61, 0.88] }),
    (MaskPart::RightLeg, Shape::Rect { min: [0.39, 0.5], max: [0.5, 0.88] }),
    (MaskPart::Root, Shape::Rect { min: [0.3, 0.965], max: [0.7, 1.0] }),
];

/// The part under `p` (unit-square coordinates).
pub(crate) fn part_at(p: [f32; 2]) -> Option<MaskPart> {
    SILHOUETTE.iter().find(|(_, s)| s.contains(p)).map(|(part, _)| *part)
}

/// Switch `part` and report whether the mask changed.
pub(crate) fn toggle_part(mask: &mut AvatarMask, part: MaskPart) -> bool {
    let on = !mask.has_part(part);
    mask.set_part(part, on);
    true
}

/// Draw the silhouette in a square of `size` and handle clicks. Returns
/// whether the mask changed. The hovered part's name is shown as a tooltip.
pub(crate) fn silhouette_ui(ui: &mut egui::Ui, mask: &mut AvatarMask, size: f32) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    let to_unit = |p: Pos2| [(p.x - rect.min.x) / rect.width(), (p.y - rect.min.y) / rect.height()];
    let to_screen = |u: [f32; 2]| Pos2::new(rect.min.x + u[0] * rect.width(), rect.min.y + u[1] * rect.height());
    let hovered = response.hover_pos().and_then(|p| part_at(to_unit(p)));
    let painter = ui.painter_at(rect);
    // Back to front: the big parts first so the small ones sit on top.
    for (part, shape) in SILHOUETTE.iter().rev() {
        let on = mask.has_part(*part);
        let fill = if on { theme::ACCENT_DEFAULT } else { Color32::from_gray(60) };
        let stroke = Stroke::new(if hovered == Some(*part) { 2.0_f32 } else { 1.0 }, if hovered == Some(*part) { Color32::WHITE } else { Color32::from_gray(110) });
        match *shape {
            Shape::Circle { c, r } => {
                painter.circle(to_screen(c), r * rect.width(), fill, stroke);
            }
            Shape::Rect { min, max } => {
                painter.rect(Rect::from_min_max(to_screen(min), to_screen(max)), 3.0, fill, stroke, egui::StrokeKind::Inside);
            }
        }
    }
    let mut changed = false;
    if let Some(part) = hovered {
        response.clone().on_hover_text(format!("{} ({})", part.label(), if mask.has_part(part) { "on" } else { "off" }));
        if response.clicked() {
            changed = toggle_part(mask, part);
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::avatar_mask::MASK_PARTS;

    #[test]
    fn every_part_has_one_shape_and_can_be_hit_at_its_centre() {
        for part in MASK_PARTS {
            let shapes: Vec<&Shape> = SILHOUETTE.iter().filter(|(p, _)| *p == part).map(|(_, s)| s).collect();
            assert_eq!(shapes.len(), 1, "{part:?}");
            let c = match *shapes[0] {
                Shape::Circle { c, .. } => c,
                Shape::Rect { min, max } => [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5],
            };
            assert_eq!(part_at(c), Some(part), "{part:?}");
        }
    }

    #[test]
    fn empty_space_hits_nothing_and_the_left_side_is_on_the_right() {
        assert_eq!(part_at([0.02, 0.6]), None);
        assert_eq!(part_at([0.7, 0.23]), Some(MaskPart::LeftArm));
        assert_eq!(part_at([0.3, 0.23]), Some(MaskPart::RightArm));
        // The hand IK dot sits clear of the fingers and wins over nothing else.
        assert_eq!(part_at([0.93, 0.235]), Some(MaskPart::LeftHandIk));
    }

    #[test]
    fn clicking_sets_the_flag_of_exactly_that_part() {
        let mut m = AvatarMask::default();
        assert!(toggle_part(&mut m, MaskPart::Head));
        assert!(!m.has_part(MaskPart::Head));
        assert_eq!(MASK_PARTS.iter().filter(|p| m.has_part(**p)).count(), 12);
        toggle_part(&mut m, MaskPart::Head);
        assert!(m.is_everything());
    }
}
