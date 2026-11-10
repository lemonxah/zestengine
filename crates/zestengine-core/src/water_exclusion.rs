//! Authored dry regions and their renderer-independent spatial queries.
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use crate::{ObjectId, ObjectRef, TypedComponent};

/// Removes water inside a transformed box. Parent this to a boat or building
/// to keep its interior dry without removing the water beneath its floor.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterExclusionVolume {
    pub half_extents: [f32; 3],
    /// Empty affects every water body; otherwise only the listed objects.
    pub waters: Vec<ObjectRef>,
}

impl Default for WaterExclusionVolume {
    fn default() -> Self { Self { half_extents: [1.0; 3], waters: Vec::new() } }
}
impl TypedComponent for WaterExclusionVolume { const NAME: &'static str = "Water Exclusion Volume"; }

impl WaterExclusionVolume {
    pub fn region(&self, world: Mat4) -> Option<WaterExclusionRegion> {
        let half = Vec3::from(self.half_extents);
        if !half.is_finite() || half.cmple(Vec3::ZERO).any() || !world.is_finite() || world.determinant().abs() < 1e-8 { return None; }
        Some(WaterExclusionRegion { to_unit: Mat4::from_scale(half.recip()) * world.inverse() })
    }
    pub fn affects(&self, water: ObjectId) -> bool { self.waters.is_empty() || self.waters.iter().any(|r| r.id() == Some(water)) }
}

/// Transform from the query's space to a unit box [-1,1]. Engine-resolved
/// copies on Water use water-local space; this has no rendering dependency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterExclusionRegion { pub to_unit: Mat4 }

impl WaterExclusionRegion {
    pub fn contains(&self, point: Vec3) -> bool {
        let p = self.to_unit.transform_point3(point);
        p.is_finite() && p.abs().cmple(Vec3::ONE).all()
    }

    /// Ray parameters inside the box. Direction need not be normalized: a
    /// transformed world ray retains world-metre parameters, even with scale.
    pub fn interval(&self, origin: Vec3, direction: Vec3, range: [f32; 2]) -> Option<[f32; 2]> {
        let p = self.to_unit.transform_point3(origin);
        let d = self.to_unit.transform_vector3(direction);
        if !p.is_finite() || !d.is_finite() { return None; }
        let [mut lo, mut hi] = range;
        for axis in 0..3 {
            if d[axis].abs() < 1e-12 {
                if p[axis].abs() > 1.0 { return None; }
            } else {
                let a = (-1.0 - p[axis]) / d[axis];
                let b = (1.0 - p[axis]) / d[axis];
                lo = lo.max(a.min(b)); hi = hi.min(a.max(b));
            }
        }
        (hi > lo).then_some([lo, hi])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transformed_boxes_clip_world_metres_and_keep_water_below() {
        let volume = WaterExclusionVolume { half_extents: [2.0, 1.0, 0.5], ..Default::default() };
        let world = Mat4::from_scale_rotation_translation(Vec3::new(2.0, 1.0, 3.0), glam::Quat::from_rotation_y(0.7), Vec3::new(4.0, 2.0, -3.0));
        let region = volume.region(world).unwrap();
        assert!(region.contains(world.transform_point3(Vec3::ZERO)));
        assert!(!region.contains(world.transform_point3(Vec3::new(0.0, -1.01, 0.0))));
        let origin = world.transform_point3(Vec3::new(-3.0, 0.0, 0.0));
        let dir = world.transform_vector3(Vec3::X).normalize();
        let hit = region.interval(origin, dir, [0.0, 100.0]).unwrap();
        assert!((hit[0]-2.0).abs() < 1e-5 && (hit[1]-10.0).abs() < 1e-5);
        assert!(volume.region(Mat4::ZERO).is_none());
    }

    #[test]
    fn dry_water_queries_leave_the_wet_floor_and_do_not_serialize_runtime_regions() {
        let volume = WaterExclusionVolume { half_extents: [1.0, 0.5, 1.0], ..Default::default() };
        let mut water = crate::Water { waves: Vec::new(), current: [2.0, 0.0, 0.0], ..Default::default() };
        water.exclusions.push(volume.region(Mat4::IDENTITY).unwrap());
        assert!(water.height_at(Mat4::IDENTITY, Vec3::ZERO, 0.0).is_none());
        assert_eq!(water.flow_at(Mat4::IDENTITY, Vec3::ZERO), Vec3::ZERO);
        assert!(water.height_at(Mat4::IDENTITY, Vec3::new(0.0,-1.0,0.0),0.0).is_some());
        assert_eq!(water.flow_at(Mat4::IDENTITY, Vec3::new(0.0,-1.0,0.0)), Vec3::new(2.0,0.0,0.0));
        let saved = ron::to_string(&water).unwrap();
        let loaded: crate::Water = ron::from_str(&saved).unwrap();
        assert!(loaded.exclusions.is_empty());
        let authored: WaterExclusionVolume = ron::from_str(&ron::to_string(&volume).unwrap()).unwrap();
        assert_eq!(authored.half_extents, volume.half_extents);
    }
}
