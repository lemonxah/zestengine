//! Water-only proxies derived from the final humanoid pose. These do not
//! replace the character's world-collision capsule or add rigid bodies.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SkeletalWaterInteraction {
    pub hands: bool,
    pub feet: bool,
    pub arms: bool,
    pub legs: bool,
    /// Metres in the avatar's model space; scale follows the worn avatar.
    pub hand_radius: f32,
    pub foot_radius: f32,
    pub arm_radius: f32,
    pub leg_radius: f32,
    /// How strongly the limbs transfer horizontal momentum to water.
    pub momentum: f32,
}
impl Default for SkeletalWaterInteraction {
    fn default() -> Self {
        Self { hands: true, feet: true, arms: true, legs: true, hand_radius: 0.055, foot_radius: 0.09,
            arm_radius: 0.055, leg_radius: 0.085, momentum: 1.0 }
    }
}
impl crate::TypedComponent for SkeletalWaterInteraction {
    const NAME: &'static str = "Skeletal Water Interaction";
}
