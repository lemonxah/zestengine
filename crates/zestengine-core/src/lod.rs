//! Screen-size level of detail: pure selection logic shared by the renderer's
//! mesh swap and the non-geometry distance LOD (animation rate, IK).
//!
//! "Coverage" is the share of the screen height an object's bounding sphere
//! covers. Unlike a distance it already accounts for the object's size, the
//! camera FOV and (through the projection) the window height, so one set of
//! thresholds works for a ring and a building, on a monitor and in a headset.

use crate::LodLevel;

/// Fraction of the screen height covered by a bounding sphere of `radius` at
/// `distance`, for a camera with `tan_half_fov` = tan(vertical FOV / 2).
/// An object the camera is inside of covers everything (`f32::MAX`).
pub fn screen_coverage(radius: f32, distance: f32, tan_half_fov: f32) -> f32 {
    if distance <= radius || tan_half_fov <= 0.0 {
        return f32::MAX;
    }
    // Projected diameter over the frustum height at that distance.
    radius / (distance * tan_half_fov)
}

/// Relative margin around a threshold inside which the current level holds.
pub const COVERAGE_HYSTERESIS: f32 = 0.1;

/// Level for `coverage` out of `thresholds` (the minimum coverage each level
/// serves, descending; level 0 is the most detailed). `previous` holds while
/// the coverage stays within [`COVERAGE_HYSTERESIS`] of its band so an object
/// does not flicker at a boundary. Empty thresholds give `None`.
pub fn select_by_coverage(thresholds: &[f32], coverage: f32, previous: Option<usize>) -> Option<usize> {
    if thresholds.is_empty() {
        return None;
    }
    let pick = |c: f32| thresholds.iter().position(|&t| c >= t).unwrap_or(thresholds.len() - 1);
    if let Some(prev) = previous.filter(|&p| p < thresholds.len()) {
        let (grown, shrunk) = (coverage * (1.0 + COVERAGE_HYSTERESIS), coverage * (1.0 - COVERAGE_HYSTERESIS));
        if pick(grown) == prev || pick(shrunk) == prev {
            return Some(prev);
        }
    }
    Some(pick(coverage))
}

/// The automatic chain every model gets: the share of triangles kept and the
/// minimum screen coverage that level serves.
pub const AUTO_LOD_STEPS: [(f32, f32); 4] = [(1.0, 0.30), (0.5, 0.12), (0.25, 0.05), (0.1, 0.0)];

/// Below this coverage (the last automatic level serves everything under 0.05)
/// a static mesh may draw as an octahedral impostor billboard instead.
pub const IMPOSTOR_COVERAGE: f32 = 0.015;

/// From this coverage down a mesh gets its impostor captured, so the atlas is
/// ready before the switch.
pub const IMPOSTOR_PREPARE_COVERAGE: f32 = 0.04;

/// Whether a mesh at `coverage` draws as its impostor; `was` (it did last
/// frame) holds it within [`COVERAGE_HYSTERESIS`] of the boundary.
pub fn use_impostor(coverage: f32, was: bool) -> bool {
    let limit = if was { IMPOSTOR_COVERAGE * (1.0 + COVERAGE_HYSTERESIS) } else { IMPOSTOR_COVERAGE };
    coverage < limit
}

/// Camera distance at which a bounding sphere of `radius` reaches [`IMPOSTOR_COVERAGE`].
pub fn impostor_distance(radius: f32, tan_half_fov: f32) -> f32 {
    radius / (IMPOSTOR_COVERAGE * tan_half_fov)
}

/// Models with fewer triangles than this get no automatic chain: the draw and
/// memory overhead of extra levels outweighs the saving.
pub const AUTO_LOD_MIN_TRIANGLES: usize = 1500;

impl LodLevel {
    /// A level selected by screen size, generated from the object's own model.
    pub fn by_screen(keep: f32, screen_size: f32) -> Self {
        Self { max_distance: 0.0, model: String::new(), keep, screen_size }
    }
}

/// Minimum coverages of an automatic chain (descending), for selection.
pub fn auto_thresholds() -> [f32; AUTO_LOD_STEPS.len()] {
    AUTO_LOD_STEPS.map(|s| s.1)
}

/// How often a far object's animation is re-evaluated, in frames (1 = every
/// frame). Cheap tiers only: the pose between updates is held, so the steps
/// stay small enough not to read as stutter at that size on screen.
pub fn animation_interval(coverage: f32) -> u32 {
    match coverage {
        c if c >= 0.15 => 1,
        c if c >= 0.06 => 2,
        c if c >= 0.025 => 4,
        _ => 8,
    }
}

/// Below this coverage humanoid IK of a non-local avatar is skipped.
pub const IK_MIN_COVERAGE: f32 = 0.02;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_falls_with_distance_and_grows_with_size() {
        let t = (60f32.to_radians() / 2.0).tan();
        let near = screen_coverage(1.0, 5.0, t);
        let far = screen_coverage(1.0, 50.0, t);
        assert!((near / far - 10.0).abs() < 1e-3);
        assert!(screen_coverage(2.0, 50.0, t) > far);
        // A narrower FOV (zoomed in) makes the same object cover more.
        assert!(screen_coverage(1.0, 50.0, (20f32.to_radians() / 2.0).tan()) > far);
        assert_eq!(screen_coverage(1.0, 0.5, t), f32::MAX);
    }

    #[test]
    fn impostor_switch_has_hysteresis_and_matches_its_distance() {
        assert!(!use_impostor(0.02, false) && use_impostor(0.01, false));
        // Just over the boundary holds an impostor already in use, clearly over drops it.
        assert!(use_impostor(IMPOSTOR_COVERAGE * 1.05, true) && !use_impostor(IMPOSTOR_COVERAGE * 1.05, false));
        assert!(!use_impostor(IMPOSTOR_COVERAGE * 1.2, true));
        let t = (60f32.to_radians() / 2.0).tan();
        let d = impostor_distance(2.0, t);
        assert!((screen_coverage(2.0, d, t) - IMPOSTOR_COVERAGE).abs() < 1e-6);
        // The impostor takes over below the last automatic level.
        assert!(IMPOSTOR_PREPARE_COVERAGE < AUTO_LOD_STEPS[AUTO_LOD_STEPS.len() - 2].1 && IMPOSTOR_COVERAGE < IMPOSTOR_PREPARE_COVERAGE);
    }

    #[test]
    fn selection_follows_thresholds_with_hysteresis() {
        let th = auto_thresholds();
        assert_eq!(select_by_coverage(&th, 0.5, None), Some(0));
        assert_eq!(select_by_coverage(&th, 0.2, None), Some(1));
        assert_eq!(select_by_coverage(&th, 0.06, None), Some(2));
        assert_eq!(select_by_coverage(&th, 0.001, None), Some(3));
        assert_eq!(select_by_coverage(&[], 0.5, None), None);
        // Just under the 0.30 boundary stays on level 0, clearly under switches.
        assert_eq!(select_by_coverage(&th, 0.29, Some(0)), Some(0));
        assert_eq!(select_by_coverage(&th, 0.2, Some(0)), Some(1));
        // Just over the boundary from below holds level 1.
        assert_eq!(select_by_coverage(&th, 0.31, Some(1)), Some(1));
        assert_eq!(select_by_coverage(&th, 0.4, Some(1)), Some(0));
        // A stale previous index out of range is ignored.
        assert_eq!(select_by_coverage(&th, 0.5, Some(9)), Some(0));
    }

    #[test]
    fn animation_rate_only_drops_with_size() {
        let mut last = 0;
        for c in [1.0, 0.1, 0.03, 0.001] {
            let n = animation_interval(c);
            assert!(n >= last);
            last = n;
        }
        assert_eq!(animation_interval(0.5), 1);
        assert_eq!(animation_interval(0.001), 8);
    }

    #[test]
    fn auto_chain_is_descending_and_ends_open() {
        let th = auto_thresholds();
        assert!(th.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(*th.last().unwrap(), 0.0);
        assert!(AUTO_LOD_STEPS.windows(2).all(|w| w[0].0 > w[1].0));
    }
}
