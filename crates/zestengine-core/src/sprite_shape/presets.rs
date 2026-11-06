//! Starter profiles and splines for Sprite Shapes. A preset fixes the *structure*
//! (angle ranges, open or closed, fit mode, corner behaviour) and leaves the
//! sprite slots empty: the sprites belong to the project, so the user drops
//! theirs onto the slots afterwards.

use super::{AngleRange, EdgeFit, ShapePoint, ShapeTangentMode, SpriteShapeProfile};

/// Starter spline / profile shapes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapePreset {
    /// An open strip of grass tufts along the top of a line.
    GrassStrip,
    /// Closed ground: grass on top, dirt sides and bottom, a tiled fill.
    GroundWithFill,
    /// A sagging rope or cable, one sprite all the way round.
    RopeCable,
    /// A road: one sprite fitted along an open path, with end caps.
    Road,
}

impl ShapePreset {
    /// Every preset, in declaration order.
    pub const ALL: [ShapePreset; 4] = [Self::GrassStrip, Self::GroundWithFill, Self::RopeCable, Self::Road];

    /// Inspector label of this preset.
    pub fn label(self) -> &'static str {
        match self {
            Self::GrassStrip => "Grass Strip",
            Self::GroundWithFill => "Ground / Dirt with Fill",
            Self::RopeCable => "Rope / Cable",
            Self::Road => "Road",
        }
    }

    /// A tip shown in the inspector: what the preset fixes and what it leaves
    /// for the user.
    pub fn hint(self) -> &'static str {
        match self {
            Self::GrassStrip => "Open spline. One range facing up; assign tuft sprites to it.",
            Self::GroundWithFill => "Closed spline, walk it clockwise. Top / sides / bottom ranges and a Repeat-wrap fill texture.",
            Self::RopeCable => "Open spline with smooth points. One sprite around the whole angle circle, tiled.",
            Self::Road => "Open spline. One sprite fitted along the path, optional start / end caps.",
        }
    }
}

/// Everything a preset sets.
#[derive(Clone, Debug, PartialEq)]
pub struct PresetData {
    /// The preset's profile.
    pub profile: SpriteShapeProfile,
    /// The preset's control points.
    pub points: Vec<ShapePoint>,
    /// Open or closed.
    pub closed: bool,
    /// How edge sprites lay along a run.
    pub fit: EdgeFit,
    /// Corner threshold (degrees).
    pub corner_angle_threshold: f32,
    /// Edge sprite offset along the normal (fraction of height).
    pub border_pivot: f32,
}

fn smooth(x: f32, y: f32, tx: f32, ty: f32) -> ShapePoint {
    ShapePoint { position: [x, y], tangent_out: [tx, ty], tangent_in: [-tx, -ty], mode: ShapeTangentMode::Continuous, ..Default::default() }
}

fn range(start: f32, end: f32, order: i32) -> AngleRange {
    AngleRange { start, end, order, sprites: Vec::new() }
}

/// Everything a preset sets.
pub fn preset_shape(p: ShapePreset) -> PresetData {
    match p {
        ShapePreset::GrassStrip => PresetData {
            profile: SpriteShapeProfile { angle_ranges: vec![range(-50.0, 50.0, 0)], ..Default::default() },
            points: vec![smooth(-3.0, 0.0, 1.0, 0.3), smooth(0.0, 0.5, 1.0, 0.0), smooth(3.0, 0.0, 1.0, -0.3)],
            closed: false,
            fit: EdgeFit::Stretch,
            corner_angle_threshold: 30.0,
            border_pivot: 0.3,
        },
        ShapePreset::GroundWithFill => PresetData {
            profile: SpriteShapeProfile {
                angle_ranges: vec![range(-45.0, 45.0, 2), range(45.0, 135.0, 1), range(135.0, 225.0, 1), range(225.0, 315.0, 1)],
                fill_pixels_per_unit: 100.0,
                fill_edge_offset: -0.1,
                ..Default::default()
            },
            // Clockwise from the top-left.
            points: vec![ShapePoint::at(-3.0, 1.0), ShapePoint::at(3.0, 1.0), ShapePoint::at(3.0, -2.0), ShapePoint::at(-3.0, -2.0)],
            closed: true,
            fit: EdgeFit::Stretch,
            corner_angle_threshold: 30.0,
            border_pivot: 0.0,
        },
        ShapePreset::RopeCable => PresetData {
            profile: SpriteShapeProfile { angle_ranges: vec![range(0.0, 360.0, 0)], ..Default::default() },
            points: vec![smooth(-3.0, 1.0, 1.2, -1.0), smooth(0.0, -0.6, 1.3, 0.0), smooth(3.0, 1.0, 1.2, 1.0)],
            closed: false,
            fit: EdgeFit::Tile,
            corner_angle_threshold: 80.0,
            border_pivot: 0.0,
        },
        ShapePreset::Road => PresetData {
            profile: SpriteShapeProfile { angle_ranges: vec![range(0.0, 360.0, 0)], ..Default::default() },
            points: vec![smooth(-4.0, -1.0, 1.5, 0.3), smooth(0.0, 0.5, 1.5, 0.0), smooth(4.0, -1.0, 1.5, -0.3)],
            closed: false,
            fit: EdgeFit::Fit,
            corner_angle_threshold: 45.0,
            border_pivot: 0.0,
        },
    }
}

/// Apply a preset's structure to an existing component (the profile is a
/// separate asset; the caller writes it).
pub fn apply_preset(shape: &mut super::SpriteShapeRenderer, data: &PresetData) {
    shape.points = data.points.clone();
    shape.closed = data.closed;
    shape.fit = data.fit;
    shape.corner_angle_threshold = data.corner_angle_threshold;
    shape.border_pivot = data.border_pivot;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite_shape::spline::{TessMode, signed_area, tessellate};
    use crate::sprite_shape::SpriteShapeRenderer;

    #[test]
    fn presets_are_well_formed() {
        for p in ShapePreset::ALL {
            let d = preset_shape(p);
            assert!(d.points.len() >= 3 && !d.profile.angle_ranges.is_empty(), "{p:?}");
            assert!(!p.label().is_empty() && !p.hint().is_empty());
            let poly = tessellate(&d.points, d.closed, TessMode::Adaptive { tolerance: 0.02 }, 2048);
            assert!(poly.pts.len() >= d.points.len());
            // The profile survives the file format.
            let back: SpriteShapeProfile = ron::from_str(&ron::ser::to_string_pretty(&d.profile, Default::default()).unwrap()).unwrap();
            assert_eq!(back, d.profile);
            let mut s = SpriteShapeRenderer::default();
            apply_preset(&mut s, &d);
            assert_eq!((s.closed, s.points.len(), s.fit), (d.closed, d.points.len(), d.fit));
        }
    }

    #[test]
    fn ground_is_clockwise_and_covers_every_direction() {
        let d = preset_shape(ShapePreset::GroundWithFill);
        let poly = tessellate(&d.points, true, TessMode::Fixed { detail: 4 }, 100);
        assert!(signed_area(&poly.pts) < 0.0, "clockwise");
        assert!(d.profile.uncovered_arcs().is_empty(), "top, both sides and bottom are all assigned");
        let rope = preset_shape(ShapePreset::RopeCable);
        assert!(rope.profile.uncovered_arcs().is_empty());
        let grass = preset_shape(ShapePreset::GrassStrip);
        assert_eq!(grass.profile.uncovered_arcs(), vec![(50.0, 310.0)]);
    }
}
