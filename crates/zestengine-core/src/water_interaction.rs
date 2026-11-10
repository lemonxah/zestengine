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

/// Water-only capsule for paddles, tracked props, procedural limbs or boats.
/// Equal endpoints make a sphere; no rigid body or world collider is needed.
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default)]
pub struct WaterInteractionProxy {
    pub start:[f32;3],
    pub end:[f32;3],
    pub replace_colliders:bool,
    pub radius:f32,
    pub momentum:f32,
}
impl Default for WaterInteractionProxy {
    fn default()->Self{Self{start:[0.0,-0.25,0.0],end:[0.0,0.25,0.0],radius:0.1,momentum:1.0,replace_colliders:true}}
}
impl crate::TypedComponent for WaterInteractionProxy {const NAME:&'static str="Water Interaction Proxy";}

/// One-shot world-space impact for projectiles, rain or scripted effects.
#[derive(Clone,Copy,Debug)]
pub struct WaterImpact {
    pub position:glam::Vec3,
    pub velocity:glam::Vec3,
    pub radius:f32,
    pub strength:f32,
    /// None selects the highest intersected active surface.
    pub water:crate::ObjectRef,
}
impl Default for WaterImpact {
    fn default()->Self{Self{position:glam::Vec3::ZERO,velocity:glam::Vec3::NEG_Y*5.0,radius:0.1,strength:1.0,water:crate::ObjectRef::NONE}}
}
