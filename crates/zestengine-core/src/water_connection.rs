//! Authored river endpoints and their engine-resolved receiving surfaces.
use serde::{Deserialize,Serialize};
#[derive(Clone,Copy,Debug,PartialEq,Serialize,Deserialize)]
#[serde(default)]
pub struct WaterConnection {pub target:crate::ObjectRef,pub length:f32}
impl Default for WaterConnection {fn default()->Self{Self{target:crate::ObjectRef::NONE,length:5.0}}}
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct ResolvedWaterConnection {
    /// Receiving surface plane in river-local coordinates, normal.xyz + d.
    pub plane:[f32;4],
    pub flow:glam::Vec3,
    pub depth:f32,
    pub length:f32,
}
impl ResolvedWaterConnection {
    pub fn weight(&self,distance:f32)->f32 {let x=(distance/self.length.max(0.01)).clamp(0.0,1.0);1.0-x*x*(3.0-2.0*x)}
    pub fn height(&self,p:glam::Vec3)->f32 {if self.plane[1].abs()<1e-6{p.y}else{-(self.plane[0]*p.x+self.plane[2]*p.z+self.plane[3])/self.plane[1]}}
}

/// Acyclic receiving-body snapshot. Engine resolution owns this graph; authored
/// references are serialized separately and can never create an Arc cycle.
#[derive(Clone,Debug)]
pub struct WaterSurfaceConnection {
    pub to_target:glam::Mat4,
    pub water:std::sync::Arc<crate::Water>,
}

/// Higher-ranked intersecting water; an immutable geometry snapshot without
/// its own overlap list avoids recursive ownership tests.
#[derive(Clone,Debug)]
pub struct WaterOverlap {pub to_target:glam::Mat4,pub water:std::sync::Arc<crate::Water>,pub wave_padding:f32}
impl WaterOverlap {
    pub fn contains(&self,p:glam::Vec3)->bool {
        let p=self.to_target.transform_point3(p);
        if self.water.excluded(p){return false;}
        self.water.column(glam::Vec2::new(p.x,p.z)).is_some_and(|c|p.y>=c.surface-c.depth && p.y<=c.surface+self.wave_padding+0.05)
    }
}

impl crate::Water {
    /// Immutable surface inputs for connection resolution. Do not clone live
    /// ripple grids, pending simulation feeds, or the previous connection graph
    /// into every edge of the new graph on every frame.
    pub fn surface_snapshot(&self)->Self {
        Self{body:self.body,shape:self.shape.clone(),smooth_shape:self.smooth_shape,river:self.river.clone(),size:self.size,depth:self.depth,
            custom_mesh:self.custom_mesh.clone(),connections:self.connections,current:self.current,waves:self.waves.clone(),
            profile_waves:self.profile_waves.clone(),ocean:self.ocean.clone(),profile_ocean:self.profile_ocean.clone(),ocean_cache:self.ocean_cache.clone(),
            simulation_cache:self.simulation_cache.clone(),cache_data:self.cache_data.clone(),cache_clock:self.cache_clock.clone(),
            density:self.density,viscosity:self.viscosity,fluid_scale:self.fluid_scale,wind_response:self.wind_response,runtime:self.runtime,
            shore:self.shore.clone(),shore_depth:self.shore_depth,look:self.look,quality:self.quality,exclusions:self.exclusions.clone(),
            overlap_priority:self.overlap_priority,outline_cache:self.outline_cache.clone(),..Default::default()}
    }
}
