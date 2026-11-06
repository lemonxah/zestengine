//! Shared spatial-region selection, independent of rendering or audio.
use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct VolumeRegion {
    pub world: Mat4,
    pub half_extents: Vec3,
    pub global: bool,
    pub priority: f32,
}

impl VolumeRegion {
    pub fn contains(&self, point: Vec3) -> bool {
        if self.global { return true; }
        if !self.world.is_finite() || self.world.determinant().abs() < 1e-8 { return false; }
        let local = self.world.inverse().transform_point3(point);
        local.is_finite() && local.abs().cmple(self.half_extents.abs()).all()
    }
}

/// Highest priority wins; equal priority retains the first region in scene order.
pub fn select_volume(regions: impl IntoIterator<Item = VolumeRegion>, point: Vec3) -> Option<usize> {
    let mut selected: Option<(usize, f32)> = None;
    for (i, region) in regions.into_iter().enumerate() {
        if region.priority.is_finite() && region.contains(point)
            && selected.is_none_or(|(_, p)| region.priority > p)
        { selected = Some((i, region.priority)); }
    }
    selected.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transformed_bounds_priority_and_stable_ties() {
        let global = VolumeRegion { world: Mat4::IDENTITY, half_extents: Vec3::ONE, global: true, priority: -1.0 };
        let local = VolumeRegion { world: Mat4::from_scale_rotation_translation(Vec3::new(2.0, 1.0, 1.0), glam::Quat::from_rotation_y(1.0), Vec3::splat(10.0)), global: false, priority: 1.0, ..global };
        assert_eq!(select_volume([global, local, local], Vec3::splat(10.0)), Some(1));
        assert_eq!(select_volume([global, local], Vec3::ZERO), Some(0));
        assert!(local.contains(local.world.transform_point3(Vec3::new(0.9, 0.0, 0.0))));
        assert!(!local.contains(local.world.transform_point3(Vec3::new(1.1, 0.0, 0.0))));
        assert!(!VolumeRegion { world: Mat4::ZERO, ..local }.contains(Vec3::ZERO));
    }

}
