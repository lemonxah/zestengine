//! Custom properties of an object: named float / int / bool / vector values
//! that a model file (FBX user properties) or a script carries, and that
//! animation tracks can drive.
//!
//! A clip addresses a property by name; a vector's lanes are `Name.x`,
//! `Name.y` and `Name.z`. Writing a lane converts to the property's type
//! (ints round, bools switch at 0.5). A property the object lacks is created
//! as a float (a vector lane creates the vector).

use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// One value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CustomValue {
    /// A float.
    Float(f32),
    /// An integer.
    Int(i32),
    /// A boolean.
    Bool(bool),
    /// A 3D vector.
    Vec3([f32; 3]),
}

impl CustomValue {
    /// The value's type name ("float", "int", "bool", "vector").
    pub fn type_label(&self) -> &'static str {
        match self {
            Self::Float(_) => "float",
            Self::Int(_) => "int",
            Self::Bool(_) => "bool",
            Self::Vec3(_) => "vector",
        }
    }
    /// The first lane as a float.
    pub fn as_f32(&self) -> f32 {
        match self {
            Self::Float(v) => *v,
            Self::Int(v) => *v as f32,
            Self::Bool(v) => *v as u8 as f32,
            Self::Vec3(v) => v[0],
        }
    }
}

/// The lane suffixes of a vector property.
const LANES: [&str; 3] = ["x", "y", "z"];

/// Split `Name.x` into (`Name`, 0); `None` when `name` has no lane suffix.
fn lane_of(name: &str) -> Option<(&str, usize)> {
    let (base, suffix) = name.rsplit_once('.')?;
    let lane = LANES.iter().position(|l| *l == suffix)?;
    (!base.is_empty()).then_some((base, lane))
}

/// Named values on an object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomProperties {
    /// The named values, in insertion order.
    pub values: Vec<(String, CustomValue)>,
}

impl TypedComponent for CustomProperties {
    const NAME: &'static str = "Custom Properties";
}

impl CustomProperties {
    /// The named value.
    pub fn get(&self, name: &str) -> Option<CustomValue> {
        self.values.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    /// The first lane of the named value as a float (0 when absent).
    pub fn float(&self, name: &str) -> f32 {
        self.get(name).map_or(0.0, |v| v.as_f32())
    }

    /// Set (or create) a named value.
    pub fn set(&mut self, name: &str, value: CustomValue) {
        match self.values.iter_mut().find(|(n, _)| n == name) {
            Some((_, v)) => *v = value,
            None => self.values.push((name.to_owned(), value)),
        }
    }

    /// The scalar a track reads for `name` (`Prop` or a lane `Prop.y`).
    pub fn read_lane(&self, name: &str) -> Option<f32> {
        if let Some(v) = self.get(name) {
            return Some(v.as_f32());
        }
        let (base, lane) = lane_of(name)?;
        match self.get(base)? {
            CustomValue::Vec3(v) => Some(v[lane]),
            _ => None,
        }
    }

    /// Write the scalar a track produced into `name`, converting to the
    /// property's type.
    pub fn write_lane(&mut self, name: &str, value: f32) {
        if !value.is_finite() {
            return;
        }
        if let Some((_, v)) = self.values.iter_mut().find(|(n, _)| n == name) {
            match v {
                CustomValue::Float(f) => *f = value,
                CustomValue::Int(i) => *i = value.round() as i32,
                CustomValue::Bool(b) => *b = value >= 0.5,
                CustomValue::Vec3(a) => a[0] = value,
            }
            return;
        }
        if let Some((base, lane)) = lane_of(name) {
            match self.values.iter_mut().find(|(n, _)| n == base) {
                Some((_, CustomValue::Vec3(a))) => a[lane] = value,
                Some(_) => {}
                None => {
                    let mut a = [0.0; 3];
                    a[lane] = value;
                    self.values.push((base.to_owned(), CustomValue::Vec3(a)));
                }
            }
            return;
        }
        self.values.push((name.to_owned(), CustomValue::Float(value)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_convert_to_the_property_type() {
        let mut p = CustomProperties::default();
        p.set("Count", CustomValue::Int(0));
        p.set("On", CustomValue::Bool(false));
        p.set("Speed", CustomValue::Float(1.0));
        p.write_lane("Count", 2.6);
        p.write_lane("On", 0.7);
        p.write_lane("Speed", 4.5);
        assert_eq!(p.get("Count"), Some(CustomValue::Int(3)));
        assert_eq!(p.get("On"), Some(CustomValue::Bool(true)));
        assert_eq!(p.float("Speed"), 4.5);
        p.write_lane("On", 0.2);
        assert_eq!(p.get("On"), Some(CustomValue::Bool(false)));
    }

    #[test]
    fn vector_lanes_and_creation() {
        let mut p = CustomProperties::default();
        p.write_lane("Dir.y", 2.0);
        p.write_lane("Dir.z", -1.0);
        assert_eq!(p.get("Dir"), Some(CustomValue::Vec3([0.0, 2.0, -1.0])));
        assert_eq!(p.read_lane("Dir.z"), Some(-1.0));
        assert_eq!(p.read_lane("Dir.w"), None);
        p.write_lane("Fresh", 1.5);
        assert_eq!(p.get("Fresh"), Some(CustomValue::Float(1.5)));
        p.write_lane("Fresh", f32::NAN);
        assert_eq!(p.float("Fresh"), 1.5, "NaN never lands");
    }

    #[test]
    fn serde_defaults_load_old_data() {
        let p: CustomProperties = ron::from_str("()").unwrap();
        assert!(p.values.is_empty());
        let text = ron::ser::to_string(&CustomProperties { values: vec![("A".into(), CustomValue::Vec3([1.0, 2.0, 3.0]))] }).unwrap();
        assert_eq!(ron::from_str::<CustomProperties>(&text).unwrap().float("A"), 1.0);
    }
}
