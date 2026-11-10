//! Shared water mesh policy. The engine builds one adaptive topology per zone;
//! authored water bodies keep their geometry, material and simulation settings.
use serde::{Deserialize,Serialize};
use crate::TypedComponent;
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default)]
pub struct WaterZone {
    pub half_extents:[f32;2],
    /// Smallest tile width in zone-local metres.
    pub tile_size:f32,
    /// Cells along each tile edge (power of two at runtime).
    pub tile_quads:u32,
    /// Split tiles closer than this many tile widths to the camera.
    /// Runtime minimum 2 keeps successive geometric transition bands separate.
    pub lod_scale:f32,
    pub max_lod:u32,
    /// Higher priority zones own bodies in overlapping zones.
    pub priority:i32,
    /// Ocean coverage outside the detailed zone, in local metres (0 disables).
    pub far_distance:f32,
}
impl Default for WaterZone {
    fn default()->Self {Self {half_extents:[512.0;2],tile_size:16.0,tile_quads:8,lod_scale:2.0,max_lod:6,priority:0,far_distance:20000.0}}
}
impl TypedComponent for WaterZone {const NAME:&'static str="Water Zone";}
