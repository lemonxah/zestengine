//! Water: a `Water` volume (surface at the object's height, Gerstner waves,
//! density, drag, current and its look) and `Buoyancy` for rigid bodies that
//! float in it. The wave function here is the one the water shader renders,
//! so floating objects ride the waves you see.

use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Wind's growth rate on a wave as a share of its angular frequency.
const WAVE_GROWTH: f32 = 0.03;
const GRAVITY: f32 = 9.81;

/// One Gerstner wave.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GerstnerWave {
    /// Travel direction in degrees around +Y (0 = +X).
    pub direction: f32,
    /// Crest to crest (m).
    pub wavelength: f32,
    /// Crest height above the mean (m).
    pub amplitude: f32,
    /// 0 = sine wave, 1 = sharp crests.
    pub steepness: f32,
    /// Multiplies the deep-water speed for this wavelength.
    pub speed: f32,
}

impl Default for GerstnerWave {
    fn default() -> Self {
        Self { direction: 0.0, wavelength: 8.0, amplitude: 0.15, steepness: 0.5, speed: 1.0 }
    }
}

/// Water's look in the water shader.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterLook {
    /// Physical volume coefficients. Missing in older scenes: derive from
    /// their existing colour/depth settings at the render boundary.
    #[serde(default)]
    pub optics: Option<crate::WaterOptics>,
    /// Colour where the water is thin (linear RGB).
    pub shallow_color: [f32; 3],
    /// Colour it fades to with depth.
    pub deep_color: [f32; 3],
    /// Depth (m) over which the water turns fully deep-coloured.
    pub absorption_depth: f32,
    /// How much of the bottom shows through (0 = opaque).
    pub clarity: f32,
    /// Surface roughness 0 (mirror) .. 1 (diffuse).
    pub roughness: f32,
    /// Strength of the small scrolling ripples on top of the waves.
    pub ripple_strength: f32,
    /// World-space scale of the small scrolling ripples (bigger = smaller ripples).
    pub ripple_scale: f32,
    /// Shoreline / intersection foam width (m) and crest foam amount.
    pub foam_width: f32,
    /// Foam on wave crests (0..1 amount).
    pub crest_foam: f32,
    /// How strongly the surface bends the view of what's underneath
    /// (1 = water's real refraction, 0 = none).
    pub refraction: f32,
    /// All foam (contact, crests, whitewater) as a multiplier.
    pub foam: f32,
    /// Foam drawn at all (keeps `foam` for when it's turned back on).
    #[serde(default = "default_true")]
    pub foam_on: bool,
}

fn default_true() -> bool {
    true
}

impl Default for WaterLook {
    fn default() -> Self {
        Self {
            optics: Some(crate::WaterOptics::default()),
            shallow_color: [0.10, 0.45, 0.48],
            deep_color: [0.01, 0.07, 0.12],
            absorption_depth: 4.0,
            clarity: 0.7,
            roughness: 0.06,
            ripple_strength: 0.35,
            ripple_scale: 1.0,
            foam_width: 0.35,
            crest_foam: 0.3,
            refraction: 1.0,
            foam: 1.0,
            foam_on: true,
        }
    }
}

/// How a reflective surface chooses its reflections: the environment's
/// global technique (the scene's Reflection setting), the surface's own
/// planar mirror render (water, or an object with a Planar Reflection
/// component), baked probes only (no SSR / ray-traced resolve), or none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SurfaceReflection {
    /// The scene's global reflection technique (Environment -> Reflections).
    #[default]
    Environment,
    /// A planar mirror render of the scene. Water renders its own; a mesh
    /// uses its Planar Reflection component.
    Planar,
    /// Baked reflection probes only: the deferred SSR / ray-traced resolve
    /// skips this surface.
    Probes,
    /// No reflections at all (the surface stays matte).
    Off,
}

impl SurfaceReflection {
    /// The u8 the shader's fx.parallax.w carries (0 env, 1 planar, 2 probes, 3 off).
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Planar,
            2 => Self::Probes,
            3 => Self::Off,
            _ => Self::Environment,
        }
    }
}

/// What shape of water a [`Water`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WaterBody {
    /// Endless in every direction.
    Ocean,
    /// The `size` rectangle, or the `shape` polygon when it has points.
    #[default]
    Lake,
    /// A channel along `river`: each point sets the width, depth and flow
    /// speed there, and the surface follows the points' heights downstream.
    River,
    /// A built pool, hot tub or spring: shaped like a lake, still and clear,
    /// with optional steam off warm water.
    Pool,
}

impl WaterBody {
    /// Every body, in declaration order.
    pub const ALL: [WaterBody; 4] = [WaterBody::Ocean, WaterBody::Lake, WaterBody::River, WaterBody::Pool];

    /// Shaped by the `size` rectangle or the `shape` polygon.
    pub fn basin(self) -> bool {
        matches!(self, WaterBody::Lake | WaterBody::Pool)
    }
    /// Inspector label of this body.
    pub fn label(self) -> &'static str {
        match self {
            WaterBody::Ocean => "Ocean",
            WaterBody::Lake => "Lake",
            WaterBody::River => "River",
            WaterBody::Pool => "Pool",
        }
    }
}

/// One control point of a river (local space).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RiverPoint {
    /// Local-space control point position.
    pub position: [f32; 3],
    /// River width at this point (m).
    pub width: f32,
    /// Water depth at this point (m).
    pub depth: f32,
    /// Flow speed downstream (m/s).
    pub speed: f32,
}

impl Default for RiverPoint {
    fn default() -> Self {
        Self { position: [0.0; 3], width: 4.0, depth: 1.5, speed: 1.5 }
    }
}

/// The water at one spot: its rest surface height and floor depth (local),
/// how it flows (local, m/s) and how far toward the bank it is (0 centre, 1 edge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterColumn {
    /// Rest surface height (local Y).
    pub surface: f32,
    /// Floor depth below the surface (m).
    pub depth: f32,
    /// Current at this spot (local, m/s).
    pub flow: Vec3,
    /// How far toward the bank: 0 centre, 1 edge.
    pub edge: f32,
}

/// A body of water: an ocean, a lake or a river. Its rest surface is the
/// object's local XZ plane (a river's follows its points), the volume
/// extends `depth` below it. That plane is authoritative: the editor's
/// basin box hangs from it (its top is the surface), the drawn surface
/// and the simulation rest on it, and ground baked under the water (which
/// may be shallower than `depth`) only changes how deep the columns are,
/// never where the surface sits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Water {
    /// What shape of water this is.
    pub body: WaterBody,
    /// Which reflection the surface uses: the environment's global technique,
    /// this water's own planar mirror render (warped by the ripples), baked
    /// probes only, or none.
    pub reflection: SurfaceReflection,
    /// Lake / pool outline (local XZ, in order); empty = the `size` rectangle.
    pub shape: Vec<[f32; 2]>,
    /// The outline is a smooth closed curve through the points rather than
    /// straight edges (hot springs, kidney pools).
    pub smooth_shape: bool,
    /// River control points, upstream first.
    pub river: Vec<RiverPoint>,
    /// Lake / pool rectangle extent (local XZ, m); ignored for oceans and rivers.
    pub size: [f32; 2],
    /// Volume depth below the surface (m).
    pub depth: f32,
    /// Authored Gerstner waves (the wind sea is added on top).
    pub waves: Vec<GerstnerWave>,
    /// Optional reusable .waterwaves asset; empty uses the authored waves.
    pub wave_profile: String,
    #[serde(skip)]
    pub profile_waves: Option<std::sync::Arc<Vec<GerstnerWave>>>,
    /// Density (kg/m^3): 1000 fresh water, 1025 sea. Sets what floats
    /// (Archimedes: lift is the weight of the fluid displaced).
    pub density: f32,
    /// Dynamic viscosity (Pa s): 0.001 water, ~0.1 oil, ~10 honey. Sets how
    /// thick the fluid is: drag on bodies (how fast they sink or rise to
    /// where their density settles them), how fast waves die, how sluggishly
    /// it flows.
    pub viscosity: f32,
    /// Stage the fluid as a 1:N scale model: it moves like the same fluid
    /// N times smaller (a jar of honey in a pool-sized scene). By Froude
    /// similarity that is viscosity times N^1.5. 1 = real size.
    pub fluid_scale: f32,
    /// Linear drag on submerged bodies (1/s).
    pub drag: f32,
    /// Angular drag on submerged bodies (1/s).
    pub angular_drag: f32,
    /// Current, in the water's local space (m/s).
    pub current: [f32; 3],
    /// Surface mesh quads per side.
    pub resolution: u32,
    /// How the surface shades.
    pub look: WaterLook,
    /// How strongly Wind Zones raise waves here (0 = ignore wind).
    pub wind_response: f32,
    /// Floating bodies and characters push the surface and leave ripples
    /// and wakes (simulated in a window around the player).
    pub interactive: bool,
    /// Width of that window (m) and its cells per side.
    pub ripple_area: f32,
    /// Cells per side of the interactive ripple window.
    pub ripple_resolution: u32,
    /// Share of ripple motion lost per second.
    pub ripple_damping: f32,
    /// Run the interactive layer as a GPU shallow-water simulation (mass,
    /// flow, displacement); off, or without a GPU, it falls back to CPU ripples.
    pub gpu_sim: bool,
    /// Deepest water the simulated waves travel as (m): deeper water still
    /// holds its full mass but its waves move like water this deep.
    pub sim_wave_depth: f32,
    /// Scales the water a body pushes aside as it enters or moves.
    pub sim_displacement: f32,
    /// How strongly moving bodies drag the water under them along.
    pub sim_momentum: f32,
    /// Strength of the dispersive wave layer over the simulation (wave
    /// trains, Kelvin wakes, chop; 0 = off). Runs weaker over shallow water.
    pub sim_dispersion: f32,
    /// How much of the dispersive ripple layer shows (0 = off, 1 = all of
    /// it); scales the ripples already there too, not only new ones.
    pub sim_ripple_mix: f32,
    /// Eddy viscosity on the simulation's short chop (0 = none, 1 = strong):
    /// turbulence drains short waves far faster than long swells, so chop
    /// from bodies and impacts dies out instead of bouncing round a pool.
    pub sim_chop_smoothing: f32,
    /// How strongly bodies moving through start dispersive waves (their
    /// wake); drops and splashes are not scaled by it.
    pub sim_wake_ripples: f32,
    /// Seconds a body's push is spread over: smooths out the frame-by-frame
    /// steps that otherwise ring as fine ripples all round it.
    pub sim_ripple_smoothing: f32,
    /// Damping (1/s) of the finest waves the grid holds, so grid-sized
    /// ripples die out instead of covering the surface in a checker.
    pub sim_fine_damping: f32,
    /// How it looks and sounds from below the surface.
    #[serde(default)]
    pub underwater: UnderwaterSettings,
    /// Steam rising off warm water (0 = none; hot tubs, hot springs).
    pub steam: f32,
    /// Lights under the surface throw caustics out of the water onto the
    /// walls and ceiling around it.
    pub project_caustics: bool,
    /// Brightness of that projected light (1 = physical).
    pub projection_strength: f32,
    /// Waves calm where the water is shallower than this (m) over the
    /// ground under it (baked from the static geometry): gone at the
    /// waterline, full from this depth on. 0 = full waves everywhere.
    pub shore_depth: f32,
    /// Engine-filled wind state, not saved.
    #[serde(skip)]
    pub runtime: WaterRuntime,
    /// The depth of water over the ground (engine-baked), for `shore_depth`.
    #[serde(skip)]
    pub shore: Option<std::sync::Arc<ShoreDepth>>,
    /// CPU ripple state, present while the interactive layer runs (not saved).
    #[serde(skip)]
    pub ripples: Option<crate::RippleSim>,
    /// What this frame feeds the GPU simulation (window, displacers, field).
    #[serde(skip)]
    pub sim_feed: crate::WaterSimFeed,
    /// The river's sampled curve, rebuilt when its points change.
    #[serde(skip)]
    pub river_cache: std::sync::Arc<std::sync::Mutex<RiverCache>>,
    /// The outline as drawn and tested (`outline`), rebuilt when it changes.
    #[serde(skip)]
    pub outline_cache: std::sync::Arc<std::sync::Mutex<OutlineCache>>,
    /// Asked (from the inspector) to fit the outline and depth to the basin
    /// around it; the editor does it and clears this.
    #[serde(skip)]
    pub fit_request: bool,
}

/// Depth baked where nothing lies under the water (m): open water.
pub const SHORE_NO_GROUND: f32 = 1.0e4;

/// The depth of water over the ground under a body, on an `n` x `n` grid of
/// cells spanning local XZ -`half`..`half` (row-major, x fastest; cell
/// centres at the texel centres the shader samples), clamped to
/// 0..`SHORE_NO_GROUND`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShoreDepth {
    /// Grid cells per side.
    pub n: usize,
    /// Grid half-extent (m).
    pub half: f32,
    /// Depth per cell (row-major, x fastest).
    pub depth: Vec<f32>,
    /// Changes with every bake (what the renderer re-uploads on).
    pub version: u64,
}

impl ShoreDepth {
    /// Build a grid, clamping NaN cells to open water and bumping the bake
    /// version so the renderer re-uploads it.
    pub fn new(n: usize, half: f32, mut depth: Vec<f32>) -> Self {
        static VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        for d in &mut depth {
            *d = if d.is_nan() { SHORE_NO_GROUND } else { d.clamp(0.0, SHORE_NO_GROUND) };
        }
        Self { n, half, depth, version: VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }
    }

    /// The depth under local `xz`, bilinear between cell centres (as the
    /// GPU filters it); None outside the grid.
    pub fn depth_at(&self, xz: Vec2) -> Option<f32> {
        if self.n == 0 || self.half <= 0.0 || self.depth.len() != self.n * self.n {
            return None;
        }
        let uv = xz / (2.0 * self.half) + 0.5;
        if !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
            return None;
        }
        let n = self.n;
        let p = (uv * n as f32 - 0.5).clamp(Vec2::ZERO, Vec2::splat((n - 1) as f32));
        let (x0, z0) = (p.x.floor() as usize, p.y.floor() as usize);
        let (x1, z1) = ((x0 + 1).min(n - 1), (z0 + 1).min(n - 1));
        let f = p - Vec2::new(x0 as f32, z0 as f32);
        let at = |x: usize, z: usize| self.depth[z * n + x];
        let a = at(x0, z0) + (at(x1, z0) - at(x0, z0)) * f.x;
        let b = at(x0, z1) + (at(x1, z1) - at(x0, z1)) * f.x;
        Some(a + (b - a) * f.y)
    }
}

/// Share of the waves left over water `depth` deep for a body's
/// `shore_depth`: smoothstep(0, shore_depth, depth); 1 when it is off.
pub fn shore_damping(depth: f32, shore_depth: f32) -> f32 {
    if shore_depth <= 0.0 {
        return 1.0;
    }
    let t = (depth / shore_depth).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Debug, Default)]
pub struct OutlineCache {
    key: (Vec<[f32; 2]>, bool),
    outline: std::sync::Arc<Vec<[f32; 2]>>,
}

/// Samples per edge of a smooth outline.
const OUTLINE_SAMPLES: usize = 8;

/// The closed Catmull-Rom curve through `pts`, `OUTLINE_SAMPLES` per edge.
fn smooth_outline(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = pts.len();
    let at = |k: usize| Vec2::from(pts[k % n]);
    let mut out = Vec::with_capacity(n * OUTLINE_SAMPLES);
    for k in 0..n {
        let (p0, p1, p2, p3) = (at(k + n - 1), at(k), at(k + 1), at(k + 2));
        for s in 0..OUTLINE_SAMPLES {
            let t = s as f32 / OUTLINE_SAMPLES as f32;
            let (t2, t3) = (t * t, t * t * t);
            let p = 0.5 * ((2.0 * p1) + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
            out.push(p.to_array());
        }
    }
    out
}

/// A fluid template: its physics (density, viscosity) and look, applied to
/// a water as a starting point to tune from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidPreset {
    /// Preset name.
    pub name: &'static str,
    /// kg/m^3.
    pub density: f32,
    /// Dynamic viscosity (Pa s).
    pub viscosity: f32,
    /// Shallow-water colour (linear RGB).
    pub shallow: [f32; 3],
    /// Deep-water colour (linear RGB).
    pub deep: [f32; 3],
    /// Metres light travels before taking the deep colour.
    pub absorption: f32,
    /// How much of the bottom shows through (0 = opaque).
    pub clarity: f32,
    /// Surface roughness 0 (mirror) .. 1 (diffuse).
    pub roughness: f32,
    /// Strength of the small scrolling ripples.
    pub ripples: f32,
    /// Overall foam amount.
    pub foam: f32,
    /// Refraction against water's (1 = n 1.33; oil and honey bend more).
    pub refraction: f32,
    /// Steam off warm water (0 = none).
    pub steam: f32,
}

impl FluidPreset {
    /// Set `water`'s fluid and look to this template (its shape, waves and
    /// simulation settings are kept).
    pub fn apply(&self, water: &mut Water) {
        water.density = self.density;
        water.viscosity = self.viscosity;
        water.steam = self.steam;
        let l = &mut water.look;
        l.shallow_color = self.shallow;
        l.deep_color = self.deep;
        l.absorption_depth = self.absorption;
        l.clarity = self.clarity;
        l.roughness = self.roughness;
        l.ripple_strength = self.ripples;
        l.foam = self.foam;
        l.refraction = self.refraction;
        l.optics = Some(crate::WaterOptics::from_legacy(l));
    }

    /// Whether `water`'s fluid is this one's (density and viscosity).
    pub fn matches(&self, water: &Water) -> bool {
        (water.density - self.density).abs() < 0.5 && (water.viscosity - self.viscosity).abs() <= self.viscosity * 0.01
    }
}

const fn fluid(name: &'static str, density: f32, viscosity: f32, shallow: [f32; 3], deep: [f32; 3], look: [f32; 7]) -> FluidPreset {
    let [absorption, clarity, roughness, ripples, foam, refraction, steam] = look;
    FluidPreset { name, density, viscosity, shallow, deep, absorption, clarity, roughness, ripples, foam, refraction, steam }
}

/// Fluid templates. Look: absorption, clarity, roughness, ripples, foam,
/// refraction, steam.
pub const FLUID_PRESETS: [FluidPreset; 9] = [
    fluid("Water", 1000.0, 0.001, [0.10, 0.45, 0.48], [0.01, 0.07, 0.12], [4.0, 0.7, 0.06, 0.35, 1.0, 1.0, 0.0]),
    fluid("Sea water", 1025.0, 0.00108, [0.06, 0.40, 0.44], [0.0, 0.05, 0.12], [6.0, 0.6, 0.07, 0.4, 1.0, 1.0, 0.0]),
    fluid("Hot spring", 1005.0, 0.0005, [0.35, 0.75, 0.80], [0.05, 0.35, 0.45], [5.0, 0.85, 0.04, 0.2, 0.3, 1.0, 1.0]),
    fluid("Swamp", 1010.0, 0.003, [0.25, 0.30, 0.12], [0.04, 0.06, 0.02], [0.6, 0.15, 0.1, 0.15, 0.4, 1.0, 0.0]),
    fluid("Oil", 900.0, 0.1, [0.25, 0.20, 0.08], [0.03, 0.025, 0.01], [0.4, 0.2, 0.03, 0.15, 0.0, 1.4, 0.0]),
    fluid("Syrup", 1330.0, 2.5, [0.65, 0.32, 0.08], [0.25, 0.08, 0.01], [0.5, 0.45, 0.02, 0.05, 0.0, 1.4, 0.0]),
    fluid("Honey", 1420.0, 10.0, [0.95, 0.60, 0.12], [0.55, 0.25, 0.02], [0.35, 0.55, 0.015, 0.03, 0.0, 1.5, 0.0]),
    fluid("Cold honey", 1430.0, 100.0, [0.90, 0.52, 0.10], [0.50, 0.20, 0.02], [0.3, 0.45, 0.02, 0.0, 0.0, 1.5, 0.0]),
    fluid("Mud", 1700.0, 50.0, [0.32, 0.24, 0.16], [0.18, 0.13, 0.08], [0.05, 0.0, 0.35, 0.05, 0.0, 0.0, 0.0]),
];

/// Ready-made outlines fitted to a `size` rectangle (local XZ, centred).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlinePreset {
    /// A circle inscribed in the rectangle.
    Circle,
    /// An oval fitted to the rectangle.
    Oval,
    /// A rounded rectangle.
    Rounded,
}

impl OutlinePreset {
    /// Every preset, in declaration order.
    pub const ALL: [OutlinePreset; 3] = [OutlinePreset::Circle, OutlinePreset::Oval, OutlinePreset::Rounded];

    /// Inspector label of this preset.
    pub fn label(self) -> &'static str {
        match self {
            OutlinePreset::Circle => "Circle",
            OutlinePreset::Oval => "Oval",
            OutlinePreset::Rounded => "Rounded",
        }
    }

    /// The points (straight edges) for a `size` rectangle.
    pub fn points(self, size: [f32; 2]) -> Vec<[f32; 2]> {
        let half = Vec2::from(size).max(Vec2::splat(0.2)) * 0.5;
        let ring = |r: Vec2, n: usize| -> Vec<[f32; 2]> {
            (0..n)
                .map(|k| {
                    let a = k as f32 / n as f32 * std::f32::consts::TAU;
                    [a.cos() * r.x, a.sin() * r.y]
                })
                .collect()
        };
        match self {
            OutlinePreset::Circle => ring(Vec2::splat(half.min_element()), 32),
            OutlinePreset::Oval => ring(half, 32),
            OutlinePreset::Rounded => {
                let r = half.min_element() * 0.35;
                let inner = half - Vec2::splat(r);
                let mut out = Vec::new();
                for (corner, start) in [(Vec2::new(1.0, 1.0), 0.0f32), (Vec2::new(-1.0, 1.0), 90.0), (Vec2::new(-1.0, -1.0), 180.0), (Vec2::new(1.0, -1.0), 270.0)] {
                    let c = inner * corner;
                    for s in 0..=6 {
                        let a = (start + s as f32 * 15.0).to_radians();
                        out.push([c.x + a.cos() * r, c.y + a.sin() * r]);
                    }
                }
                out
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct RiverCache {
    points: Vec<RiverPoint>,
    path: std::sync::Arc<Vec<RiverSample>>,
}

/// One sample along a river's smooth centre line (local space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverSample {
    pub pos: Vec3,
    pub width: f32,
    pub depth: f32,
    pub speed: f32,
    /// Unit direction downstream (down the slope too).
    pub dir: Vec3,
    /// Distance downstream (m).
    pub dist: f32,
    /// Position along the control points: segment index plus fraction.
    pub param: f32,
}

/// Centripetal Catmull-Rom through `p1`..`p2` (no cusps or loops).
fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let knot = |a: Vec3, b: Vec3| a.distance(b).max(1e-4).sqrt();
    let (t0, t1) = (0.0, knot(p0, p1));
    let t2 = t1 + knot(p1, p2);
    let t3 = t2 + knot(p2, p3);
    let u = t1 + (t2 - t1) * t;
    let lerp = |a: Vec3, b: Vec3, ta: f32, tb: f32| a * ((tb - u) / (tb - ta)) + b * ((u - ta) / (tb - ta));
    let a1 = lerp(p0, p1, t0, t1);
    let a2 = lerp(p1, p2, t1, t2);
    let a3 = lerp(p2, p3, t2, t3);
    let b1 = lerp(a1, a2, t0, t2);
    let b2 = lerp(a2, a3, t1, t3);
    lerp(b1, b2, t1, t2)
}

/// The river's centre line at `param` (segment index plus fraction).
fn river_curve(points: &[RiverPoint], param: f32) -> Vec3 {
    let n = points.len();
    let at = |i: usize| Vec3::from(points[i].position);
    if n < 2 {
        return points.first().map_or(Vec3::ZERO, |p| Vec3::from(p.position));
    }
    let k = (param.floor() as usize).min(n - 2);
    let t = (param - k as f32).clamp(0.0, 1.0);
    let (p1, p2) = (at(k), at(k + 1));
    if t <= 0.0 {
        return p1;
    }
    if t >= 1.0 {
        return p2;
    }
    let p0 = if k == 0 { p1 * 2.0 - p2 } else { at(k - 1) };
    let p3 = if k + 2 >= n { p2 * 2.0 - p1 } else { at(k + 2) };
    catmull_rom(p0, p1, p2, p3, t)
}

/// A camera or listener below a [`Water`]'s surface: the view fogs toward the
/// water's colour and the mix is muffled under its ambience.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UnderwaterSettings {
    /// Whether underwater effects apply.
    pub enabled: bool,
    /// Looping ambience while the listener is under (project-relative path).
    pub sound: String,
    /// Ambience volume while the listener is under the surface.
    pub volume: f32,
    /// How strongly everything else is muffled (0 = not at all).
    pub muffle: f32,
    /// How far you can see under water (m): the view is nearly all fog
    /// (95%) at this distance. Murky water: a couple of metres; a clear pool
    /// or tropical sea: 20-40 m.
    pub visibility: f32,
    /// Post-processing profile (`.postfx`) blended over the camera's own
    /// while it is under (colour grading, vignette, bloom...).
    pub postfx: String,
    /// Screen wobble strength.
    pub distortion: f32,
    /// Soft blur of the view below the surface (screen pixels at 1080p).
    pub blur: f32,
    /// Light focused by the moving surface onto what lies under the water
    /// (1 = as real water focuses it, 0 = none).
    pub caustics: f32,
}

impl Default for UnderwaterSettings {
    fn default() -> Self {
        Self { enabled: true, sound: String::new(), volume: 0.8, muffle: 0.85, visibility: 12.0, postfx: String::new(), distortion: 1.0, blur: 2.5, caustics: 1.0 }
    }
}

/// Engine-filled wind state of a [`Water`], not saved.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterRuntime {
    /// Wind direction across the surface (local XZ, unit) and the sea state's
    /// wind speed (m/s), which builds up and dies down over time.
    pub wind_dir: Vec2,
    /// The sea state's wind speed (m/s), which builds up and dies down over time.
    pub wind_speed: f32,
}

/// A wave ready to evaluate: direction (local XZ), wave number, angular
/// speed, horizontal (Gerstner) amplitude and height amplitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedWave {
    /// Travel direction (local XZ, unit).
    pub dir: Vec2,
    /// Wave number (radians per metre).
    pub k: f32,
    /// Angular frequency (radians per second).
    pub omega: f32,
    /// Horizontal (Gerstner) amplitude.
    pub qa: f32,
    /// Height amplitude (m).
    pub amplitude: f32,
}

impl Default for Water {
    fn default() -> Self {
        Self {
            body: WaterBody::Lake,
            reflection: SurfaceReflection::default(),
            shape: Vec::new(),
            smooth_shape: false,
            river: Vec::new(),
            size: [20.0, 20.0],
            depth: 5.0,
            wave_profile: String::new(),
            profile_waves: None,
            waves: vec![
                GerstnerWave { direction: 20.0, wavelength: 9.0, amplitude: 0.14, steepness: 0.5, speed: 1.0 },
                GerstnerWave { direction: -35.0, wavelength: 5.0, amplitude: 0.07, steepness: 0.45, speed: 1.0 },
                GerstnerWave { direction: 75.0, wavelength: 2.6, amplitude: 0.03, steepness: 0.4, speed: 1.0 },
            ],
            density: 1000.0,
            viscosity: 0.001,
            fluid_scale: 1.0,
            drag: 1.0,
            angular_drag: 1.0,
            current: [0.0; 3],
            resolution: 128,
            look: WaterLook::default(),
            wind_response: 1.0,
            interactive: true,
            ripple_area: 32.0,
            ripple_resolution: 256,
            ripple_damping: 0.8,
            gpu_sim: true,
            sim_wave_depth: 1.0,
            sim_displacement: 0.7,
            sim_momentum: 0.43,
            sim_dispersion: 1.0,
            sim_ripple_mix: 0.45,
            sim_chop_smoothing: 1.0,
            sim_wake_ripples: 0.2,
            sim_ripple_smoothing: 0.06,
            sim_fine_damping: 6.0,
            underwater: UnderwaterSettings::default(),
            steam: 0.0,
            project_caustics: false,
            projection_strength: 1.0,
            shore_depth: 2.0,
            runtime: WaterRuntime::default(),
            shore: None,
            ripples: None,
            sim_feed: Default::default(),
            river_cache: Default::default(),
            outline_cache: Default::default(),
            fit_request: false,
        }
    }
}

impl TypedComponent for Water {
    const NAME: &'static str = "Water";
}

impl Water {
    /// Most waves the surface sums (authored plus wind).
    pub const MAX_WAVES: usize = 8;

    /// The waves this surface shows: the authored ones plus the wind sea, with
    /// the depth-aware dispersion (waves slow down in shallow water).
    pub fn resolved_waves(&self) -> Vec<ResolvedWave> {
        // (direction, wavelength, amplitude, steepness, speed)
        let authored = self.profile_waves.as_deref().unwrap_or(&self.waves);
        let mut list: Vec<(Vec2, f32, f32, f32, f32)> = authored
            .iter()
            .map(|w| {
                let a = w.direction.to_radians();
                (Vec2::new(a.cos(), a.sin()), w.wavelength, w.amplitude, w.steepness, w.speed)
            })
            .collect();
        let u = self.runtime.wind_speed * self.wind_response.max(0.0);
        if u > 0.05 && self.runtime.wind_dir.length_squared() > 0.5 {
            // A developed wind sea: peak wavelength and height grow with the
            // square of the wind speed, limited by the fetch (the water's
            // size) and by depth (waves break past ~0.78 of the depth).
            let mut peak = 0.833 * u * u;
            let fetch = match (self.size[0] > 0.0, self.size[1] > 0.0) {
                (true, true) => self.size[0].min(self.size[1]),
                (true, false) => self.size[0],
                (false, true) => self.size[1],
                (false, false) => f32::MAX,
            };
            peak = peak.min(fetch * 0.5).max(0.2);
            let height = (0.0107 * u * u).min(0.39 * self.depth.max(0.05)).min(peak / 14.0);
            let base = self.runtime.wind_dir.y.atan2(self.runtime.wind_dir.x);
            for (angle, len, amp) in [(0.0f32, 1.0f32, 1.0f32), (25.0, 0.62, 0.5), (-30.0, 0.41, 0.3), (60.0, 0.27, 0.15)] {
                let a = base + angle.to_radians();
                list.push((Vec2::new(a.cos(), a.sin()), peak * len, height * amp, 0.55, 1.0));
            }
        }
        list.truncate(Self::MAX_WAVES);
        let n = list.len();
        let depth = self.depth.max(0.05);
        list.into_iter()
            .map(|(dir, wavelength, amp, steep, speed)| {
                let k = std::f32::consts::TAU / wavelength.max(0.05);
                let omega = (GRAVITY * k * (k * depth).tanh()).sqrt() * speed;
                let amp = amp * self.wave_share(k);
                // Steepness is shared out so the summed crests never loop over.
                let qa = (steep.clamp(0.0, 1.0) / (k * n.max(1) as f32)).min(amp.max(0.0));
                ResolvedWave { dir, k, omega, qa, amplitude: amp.max(0.0) }
            })
            .collect()
    }

    /// Surface displacement of the rest point `xz` (local) at time `t`, and
    /// the surface normal there.
    pub fn displacement(&self, xz: Vec2, t: f32) -> (Vec3, Vec3) {
        let (d, normal) = self.free_displacement(xz, t);
        let calm = self.shore_damping(xz);
        (d * calm, Vec3::Y.lerp(normal, calm).normalize_or(Vec3::Y))
    }

    /// How much of the waves shows at local `xz` (0..1): they calm over
    /// water shallower than `shore_depth` (the shader's `water_shore`).
    pub fn shore_damping(&self, xz: Vec2) -> f32 {
        match &self.shore {
            Some(s) if self.shore_depth > 0.0 => s.depth_at(xz).map_or(1.0, |d| shore_damping(d, self.shore_depth)),
            _ => 1.0,
        }
    }

    /// `displacement` without the shore calming (open water).
    fn free_displacement(&self, xz: Vec2, t: f32) -> (Vec3, Vec3) {
        let mut d = Vec3::ZERO;
        let mut normal = Vec3::Y;
        for w in self.resolved_waves() {
            let phase = w.k * w.dir.dot(xz) - w.omega * t;
            let (s, c) = phase.sin_cos();
            d.x += w.qa * w.dir.x * c;
            d.z += w.qa * w.dir.y * c;
            d.y += w.amplitude * s;
            let wa = w.k * w.amplitude;
            normal.x -= w.dir.x * wa * c;
            normal.z -= w.dir.y * wa * c;
            normal.y -= (w.qa / w.amplitude.max(1e-6)) * wa * s;
        }
        (d, normal.normalize_or(Vec3::Y))
    }

    /// Ease the sea state toward the wind `local_wind` (water-local XZ, m/s):
    /// seas build over ~10 s and settle over ~20 s.
    pub fn feel_wind(&mut self, local_wind: Vec2, dt: f32) {
        let target = local_wind.length();
        let rate = if target > self.runtime.wind_speed { 0.1 } else { 0.05 };
        let k = 1.0 - (-rate * dt.max(0.0)).exp();
        self.runtime.wind_speed += (target - self.runtime.wind_speed) * k;
        if target > 0.1 {
            let want = local_wind / target;
            let cur = if self.runtime.wind_dir.length_squared() > 0.5 { self.runtime.wind_dir } else { want };
            self.runtime.wind_dir = cur.lerp(want, (k * 2.0).min(1.0)).normalize_or(want);
        }
    }

    /// Height (local Y) and normal of the surface above local `xz`, solving
    /// for the rest point the horizontal wave motion carries there.
    pub fn surface(&self, xz: Vec2, t: f32) -> (f32, Vec3) {
        let mut rest = xz;
        for _ in 0..4 {
            let (d, _) = self.displacement(rest, t);
            rest = xz - Vec2::new(d.x, d.z);
        }
        let (d, n) = self.displacement(rest, t);
        (d.y, n)
    }

    /// Whether local `xz` lies over the water.
    pub fn covers(&self, xz: Vec2) -> bool {
        self.column(xz).is_some()
    }

    /// How readily the fluid splashes (0..1): 1 / (1 + 3 Oh), Oh the
    /// Ohnesorge number mu / sqrt(rho sigma L) at a 10 cm splash. Water
    /// splashes freely; honey and mud swallow an impact without throwing
    /// droplets.
    pub fn splashiness(&self) -> f32 {
        let oh = self.effective_viscosity() / (self.density.max(1.0) * 0.072 * 0.1).sqrt();
        1.0 / (1.0 + 3.0 * oh)
    }

    /// Dynamic viscosity (Pa s) the fluid acts with: `viscosity` scaled
    /// by `fluid_scale`^1.5.
    pub fn effective_viscosity(&self) -> f32 {
        self.viscosity.max(0.0) * self.fluid_scale.max(1.0).powf(1.5)
    }

    /// Kinematic viscosity (m^2/s) the fluid acts with.
    pub fn kinematic_viscosity(&self) -> f32 {
        self.effective_viscosity() / self.density.max(1.0)
    }

    /// Share of a wave of wave number `k` (rad/m) the fluid keeps (0..1): a
    /// wave lives only where wind feeds it (growth about 3% of its angular
    /// frequency, Miles' order for a strong wind) faster than viscosity
    /// drains it (2 nu k^2 in the body, plus the bed's boundary layer).
    /// 1 for water at every size; thick fluids keep only long waves.
    pub fn wave_share(&self, k: f32) -> f32 {
        let nu = self.kinematic_viscosity();
        if nu <= 0.0 {
            return 1.0;
        }
        let depth = self.depth.max(0.05);
        let kh = (k * depth).min(10.0);
        let omega = ((GRAVITY * k + 7.3e-5 * k * k * k) * kh.tanh()).max(1e-6).sqrt();
        let bulk = 2.0 * nu * k * k;
        let bed = k * (nu * omega * 0.5).sqrt() / (2.0 * kh).sinh().max(1e-3);
        (1.0 - (bulk + bed) / (WAVE_GROWTH * omega)).clamp(0.0, 1.0)
    }

    /// Make this a pool: still, clear, pool-blue water with no waves, wind
    /// sea or foam (kept: its shape and depth).
    pub fn make_pool(&mut self) {
        self.body = WaterBody::Pool;
        self.waves.clear();
        self.wind_response = 0.0;
        self.current = [0.0; 3];
        self.look.shallow_color = [0.35, 0.75, 0.85];
        self.look.deep_color = [0.02, 0.25, 0.42];
        self.look.absorption_depth = 8.0;
        self.look.clarity = 0.95;
        self.look.roughness = 0.03;
        self.look.ripple_strength = 0.15;
        self.look.foam = 0.0;
        self.project_caustics = true;
    }

    /// The water at local `xz`, or None when that spot is dry.
    pub fn column(&self, xz: Vec2) -> Option<WaterColumn> {
        let current = Vec3::from(self.current);
        let still = |depth: f32| WaterColumn { surface: 0.0, depth, flow: current, edge: 0.0 };
        match self.body {
            WaterBody::Ocean => Some(still(self.depth)),
            WaterBody::Lake | WaterBody::Pool if self.shape.len() >= 3 => point_in_polygon(xz, &self.outline()).then(|| still(self.depth)),
            WaterBody::Lake | WaterBody::Pool => {
                let inside = (self.size[0] <= 0.0 || xz.x.abs() <= self.size[0] * 0.5)
                    && (self.size[1] <= 0.0 || xz.y.abs() <= self.size[1] * 0.5);
                inside.then(|| still(self.depth))
            }
            WaterBody::River => {
                let path = self.river_path();
                river_column(&path, 0..path.len().saturating_sub(1), xz, current)
            }
        }
    }

    /// Local XZ bounds of the water, None when it is endless.
    pub fn extent(&self) -> Option<(Vec2, Vec2)> {
        match self.body {
            WaterBody::Ocean => None,
            WaterBody::Lake | WaterBody::Pool if self.shape.len() >= 3 => {
                let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
                for p in self.outline().iter() {
                    lo = lo.min(Vec2::from(*p));
                    hi = hi.max(Vec2::from(*p));
                }
                Some((lo, hi))
            }
            WaterBody::Lake | WaterBody::Pool => {
                let half = Vec2::from(self.size) * 0.5;
                (self.size[0] > 0.0 && self.size[1] > 0.0).then_some((-half, half))
            }
            WaterBody::River => {
                let path = self.river_path();
                if path.len() < 2 {
                    return None;
                }
                let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
                for p in path.iter() {
                    let c = Vec2::new(p.pos.x, p.pos.z);
                    lo = lo.min(c - Vec2::splat(p.width * 0.5));
                    hi = hi.max(c + Vec2::splat(p.width * 0.5));
                }
                Some((lo, hi))
            }
        }
    }

    /// The water column at every cell centre of a `n` x `n` grid of `cell`
    /// metres whose first cell starts at local `lo` (row-major, z rows);
    /// rivers only test the segments near the grid.
    pub fn columns_in(&self, lo: Vec2, n: usize, cell: f32) -> Vec<Option<WaterColumn>> {
        let at = |x: usize, z: usize| lo + Vec2::new((x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell);
        if self.body != WaterBody::River {
            return (0..n * n).map(|i| self.column(at(i % n, i / n))).collect();
        }
        let path = self.river_path();
        let hi = lo + Vec2::splat(n as f32 * cell);
        let near: Vec<usize> = (0..path.len().saturating_sub(1))
            .filter(|&k| {
                let (a, b) = (path[k], path[k + 1]);
                let r = a.width.max(b.width) * 0.5;
                let (smin, smax) = (Vec2::new(a.pos.x.min(b.pos.x), a.pos.z.min(b.pos.z)), Vec2::new(a.pos.x.max(b.pos.x), a.pos.z.max(b.pos.z)));
                smax.x + r >= lo.x && smin.x - r <= hi.x && smax.y + r >= lo.y && smin.y - r <= hi.y
            })
            .collect();
        let current = Vec3::from(self.current);
        (0..n * n).map(|i| river_column(&path, near.iter().copied(), at(i % n, i / n), current)).collect()
    }

    /// The surface height (local Y) over `xz` from the rest surface and the
    /// waves alone, without the interactive layer; None when dry.
    pub fn base_surface(&self, xz: Vec2, t: f32) -> Option<f32> {
        let col = self.column(xz)?;
        let (h, _) = self.surface(xz, t);
        Some(col.surface + h * (1.0 - col.edge * col.edge))
    }

    /// How the water moves at world `p` without the simulated layer: the
    /// current and a river's flow (world space, m/s).
    pub fn base_flow_at(&self, world: Mat4, p: Vec3) -> Vec3 {
        let local = world.inverse().transform_point3(p);
        self.column(Vec2::new(local.x, local.z)).map(|c| world.transform_vector3(c.flow)).unwrap_or(Vec3::ZERO)
    }

    /// The simulated layer's water velocity at world `p` (world space, m/s).
    pub fn sim_flow_at(&self, world: Mat4, p: Vec3) -> Vec3 {
        let Some(r) = self.ripples.as_ref() else { return Vec3::ZERO };
        let local = world.inverse().transform_point3(p);
        let f = r.flow(Vec2::new(local.x, local.z));
        if f == Vec2::ZERO {
            return Vec3::ZERO;
        }
        world.transform_vector3(Vec3::new(f.x, 0.0, f.y))
    }

    /// World surface height above a world point (None when outside the water
    /// horizontally or below its floor). `world` is the water's world matrix.
    pub fn height_at(&self, world: Mat4, p: Vec3, t: f32) -> Option<(f32, Vec3)> {
        let local = world.inverse().transform_point3(p);
        let xz = Vec2::new(local.x, local.z);
        let col = self.column(xz)?;
        if local.y < col.surface - col.depth {
            return None;
        }
        let (h, n) = self.surface(xz, t);
        let h = h + self.ripples.as_ref().map(|r| r.height(xz)).unwrap_or(0.0);
        // Waves calm toward a river's banks.
        let calm = 1.0 - col.edge * col.edge;
        let n = Vec3::Y.lerp(n, calm).normalize_or(Vec3::Y);
        let top = world.transform_point3(Vec3::new(local.x, col.surface + h * calm, local.z));
        Some((top.y, world.transform_vector3(n).normalize_or(Vec3::Y)))
    }

    /// How far world `p` is below the surface over it (negative above it),
    /// with the surface height; None outside the water's column.
    pub fn depth_below(&self, world: Mat4, p: Vec3, t: f32) -> Option<(f32, f32)> {
        self.height_at(world, p, t).map(|(h, _)| (h - p.y, h))
    }

    /// How the water moves at world `p` (world space, m/s): the current, a
    /// river's flow along its channel and the simulated layer's flow.
    pub fn flow_at(&self, world: Mat4, p: Vec3) -> Vec3 {
        let local = world.inverse().transform_point3(p);
        match self.column(Vec2::new(local.x, local.z)) {
            Some(c) => world.transform_vector3(c.flow) + self.sim_flow_at(world, p),
            None => Vec3::ZERO,
        }
    }

    /// The river as its surface mesh is built: a cross-section every ~half
    /// width along the points, and the section each point sits on (empty
    /// with fewer than two points).
    pub fn river_sections(&self) -> (Vec<RiverSection>, Vec<usize>) {
        let path = self.river_path();
        let sections = path
            .iter()
            .map(|p| {
                let t = Vec2::new(p.dir.x, p.dir.z).normalize_or(Vec2::X);
                RiverSection { center: p.pos, side: Vec3::new(-t.y, 0.0, t.x), width: p.width }
            })
            .collect();
        let at_point = path.iter().enumerate().filter(|(_, p)| p.param.fract() == 0.0).map(|(i, _)| i).collect();
        (sections, at_point)
    }

    /// The lake / pool outline as drawn and tested: the points, or the smooth
    /// curve through them (`smooth_shape`). Cached until they change.
    pub fn outline(&self) -> std::sync::Arc<Vec<[f32; 2]>> {
        let mut cache = self.outline_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.key.0 != self.shape || cache.key.1 != self.smooth_shape {
            cache.key = (self.shape.clone(), self.smooth_shape);
            let smooth = self.smooth_shape && self.shape.len() >= 3;
            cache.outline = std::sync::Arc::new(if smooth { smooth_outline(&self.shape) } else { self.shape.clone() });
        }
        cache.outline.clone()
    }

    /// The river's smooth centre line (a curve through every point), sampled
    /// about every half width: what the surface mesh, the physics and the
    /// editor all follow. Cached until the points change.
    pub fn river_path(&self) -> std::sync::Arc<Vec<RiverSample>> {
        let mut cache = self.river_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.points != self.river {
            cache.points = self.river.clone();
            cache.path = std::sync::Arc::new(build_river_path(&self.river));
        }
        cache.path.clone()
    }

    /// The river segment nearest local `xz`: (segment, unclamped position
    /// along it, horizontal distance to it).
    pub fn river_nearest_segment(&self, xz: Vec2) -> Option<(usize, f32, f32)> {
        let path = self.river_path();
        let last = path.len().checked_sub(2)?;
        let mut best: Option<(usize, f32, f32)> = None;
        for (i, pair) in path.windows(2).enumerate() {
            let a = Vec2::new(pair[0].pos.x, pair[0].pos.z);
            let seg = Vec2::new(pair[1].pos.x, pair[1].pos.z) - a;
            let len2 = seg.length_squared();
            let t = if len2 > 1e-8 { (xz - a).dot(seg) / len2 } else { 0.0 };
            let dist = (a + seg * t.clamp(0.0, 1.0)).distance(xz);
            if best.is_none_or(|(_, _, d)| dist < d) {
                best = Some((i, t, dist));
            }
        }
        let (i, t, dist) = best?;
        let (pa, pb) = (path[i].param, path[i + 1].param);
        // Past either end the fraction runs outside 0..1.
        let param = if (i == 0 && t < 0.0) || (i == last && t > 1.0) { pa + (pb - pa) * t } else { pa + (pb - pa) * t.clamp(0.0, 1.0) };
        let k = (param.max(0.0).floor() as usize).min(self.river.len().saturating_sub(2));
        Some((k, param - k as f32, dist))
    }

    /// Index a new river point at local `xz` goes in: before the first point
    /// or after the last when it lies past that end, else inside the nearest
    /// segment.
    pub fn river_insert_index(&self, xz: Vec2) -> usize {
        let n = self.river.len();
        match self.river_nearest_segment(xz) {
            None => n,
            Some((0, t, _)) if t < 0.0 => 0,
            Some((k, t, _)) if k + 2 == n && t > 1.0 => n,
            Some((k, _, _)) => k + 1,
        }
    }

    /// A point `t` (0..1) of the way along river segment `segment`, every
    /// value blended between its two ends.
    pub fn river_point_on_segment(&self, segment: usize, t: f32) -> Option<RiverPoint> {
        let (a, b) = (self.river.get(segment)?, self.river.get(segment + 1)?);
        let t = t.clamp(0.0, 1.0);
        let mix = |x: f32, y: f32| x + (y - x) * t;
        Some(RiverPoint {
            position: river_curve(&self.river, segment as f32 + t).to_array(),
            width: mix(a.width, b.width),
            depth: mix(a.depth, b.depth),
            speed: mix(a.speed, b.speed),
        })
    }

    /// A new river point at `index` (see [`Self::river_insert_index`]) placed at
    /// local `position`, its width, depth and flow taken from where it lands.
    pub fn insert_river_point(&mut self, index: usize, position: Vec3) {
        let n = self.river.len();
        let index = index.min(n);
        let template = if index == 0 || index == n {
            self.river.get(index.min(n.saturating_sub(1))).copied().unwrap_or_default()
        } else {
            let (a, b) = (self.river[index - 1], self.river[index]);
            let (a2, b2) = (Vec2::new(a.position[0], a.position[2]), Vec2::new(b.position[0], b.position[2]));
            let seg = b2 - a2;
            let t = if seg.length_squared() > 1e-8 { (Vec2::new(position.x, position.z) - a2).dot(seg) / seg.length_squared() } else { 0.5 };
            self.river_point_on_segment(index - 1, t).unwrap_or(a)
        };
        self.river.insert(index, RiverPoint { position: position.to_array(), ..template });
    }
}

/// One cross-section of a river (local space): its centre, the unit
/// horizontal vector across it and its width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverSection {
    /// Cross-section centre (local).
    pub center: Vec3,
    /// Unit horizontal vector across the river.
    pub side: Vec3,
    /// River width at this section (m).
    pub width: f32,
}

fn build_river_path(points: &[RiverPoint]) -> Vec<RiverSample> {
    let mut out: Vec<RiverSample> = Vec::new();
    for (k, pair) in points.windows(2).enumerate() {
        let (a, b) = (pair[0], pair[1]);
        let chord = Vec2::new(b.position[0] - a.position[0], b.position[2] - a.position[2]).length();
        let step = ((a.width + b.width) * 0.25).clamp(0.25, 4.0);
        let steps = ((chord * 1.1 / step).ceil() as usize).clamp(1, 4096);
        for s in (if k == 0 { 0 } else { 1 })..=steps {
            let t = s as f32 / steps as f32;
            let mix = |x: f32, y: f32| x + (y - x) * t;
            let param = if s == steps { (k + 1) as f32 } else { k as f32 + t };
            let pos = river_curve(points, param);
            let dist = out.last().map_or(0.0, |p| p.dist + Vec2::new(pos.x - p.pos.x, pos.z - p.pos.z).length());
            out.push(RiverSample { pos, width: mix(a.width, b.width).max(0.01), depth: mix(a.depth, b.depth), speed: mix(a.speed, b.speed), dir: Vec3::X, dist, param });
        }
    }
    // Tangent from the curve itself, so ends and corners point true.
    let end = points.len().saturating_sub(1) as f32;
    for p in &mut out {
        let (a, b) = ((p.param - 1e-3).max(0.0), (p.param + 1e-3).min(end));
        p.dir = (river_curve(points, b) - river_curve(points, a)).normalize_or(Vec3::X);
    }
    out
}

/// The river column at `xz` from the path segments `segments` (index of
/// each segment's first sample), the nearest one wins.
fn river_column(path: &[RiverSample], segments: impl Iterator<Item = usize>, xz: Vec2, current: Vec3) -> Option<WaterColumn> {
    let mut best: Option<(f32, WaterColumn)> = None;
    for k in segments {
        let (a, b) = (path[k], path[k + 1]);
        let (a2, b2) = (Vec2::new(a.pos.x, a.pos.z), Vec2::new(b.pos.x, b.pos.z));
        let seg = b2 - a2;
        let len2 = seg.length_squared();
        let t = if len2 > 1e-8 { ((xz - a2).dot(seg) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let dist = (a2 + seg * t).distance(xz);
        let half = (a.width + (b.width - a.width) * t).max(0.01) * 0.5;
        if dist > half || best.as_ref().is_some_and(|(d, _)| *d <= dist) {
            continue;
        }
        let mix = |x: f32, y: f32| x + (y - x) * t;
        let col = WaterColumn {
            surface: mix(a.pos.y, b.pos.y),
            depth: mix(a.depth, b.depth).max(0.0),
            flow: a.dir.lerp(b.dir, t).normalize_or(Vec3::X) * mix(a.speed, b.speed) + current,
            edge: dist / half,
        };
        best = Some((dist, col));
    }
    best.map(|(_, c)| c)
}

/// Even-odd point in polygon test.
fn point_in_polygon(p: Vec2, poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (Vec2::from(poly[i]), Vec2::from(poly[j]));
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Makes a Rigid Body float: buoyancy at sample points, drag and the
/// current while submerged.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Buoyancy {
    /// Lift at full submersion as a multiple of the body's weight: 2 floats
    /// it half under, 1 is neutral, below 1 sinks slowly.
    pub buoyancy: f32,
    /// Sample points in local space (empty = the corners and centre of the
    /// object's bounds).
    pub points: Vec<[f32; 3]>,
    /// Height of water each point stands for (m; 0 = half the bounds' height).
    pub point_depth: f32,
    /// Scales the water's drag for this body.
    pub drag_scale: f32,
    /// Follow the water's current.
    pub follow_current: bool,
    /// Engine-filled: share of the points under water.
    #[serde(skip)]
    pub submerged: f32,
}

impl Default for Buoyancy {
    fn default() -> Self {
        Self { buoyancy: 2.0, points: Vec::new(), point_depth: 0.0, drag_scale: 1.0, follow_current: true, submerged: 0.0 }
    }
}

impl TypedComponent for Buoyancy {
    const NAME: &'static str = "Buoyancy";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pools_take_preset_and_smooth_outlines() {
        let mut pool = Water { size: [4.0, 2.0], ..Default::default() };
        pool.make_pool();
        assert!(pool.covers(Vec2::new(1.9, 0.9)), "the Size rectangle by default");
        pool.shape = OutlinePreset::Rounded.points(pool.size);
        assert!(pool.covers(Vec2::ZERO) && pool.covers(Vec2::new(1.9, 0.0)));
        assert!(!pool.covers(Vec2::new(1.95, 0.95)), "rounded corners");
        pool.shape = OutlinePreset::Oval.points(pool.size);
        assert!(pool.covers(Vec2::new(1.9, 0.0)) && !pool.covers(Vec2::new(1.5, 0.8)));
        pool.shape = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        assert!(!pool.covers(Vec2::new(1.05, 0.0)));
        pool.smooth_shape = true;
        assert!(pool.covers(Vec2::new(1.05, 0.0)), "the curve bulges through the points");
        let (lo, hi) = pool.extent().unwrap();
        assert!(hi.x > 1.0 && lo.x < -1.0);
    }

    #[test]
    fn calm_water_is_flat_at_the_object() {
        let w = Water { waves: Vec::new(), ..Default::default() };
        let world = Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        let (h, n) = w.height_at(world, Vec3::new(3.0, 1.0, -4.0), 1.0).unwrap();
        assert!((h - 2.0).abs() < 1e-5 && (n - Vec3::Y).length() < 1e-5);
        assert!(w.height_at(world, Vec3::new(30.0, 1.0, 0.0), 1.0).is_none(), "outside the 20 m surface");
        assert!(w.height_at(world, Vec3::new(0.0, -10.0, 0.0), 1.0).is_none(), "below the floor");
    }

    #[test]
    fn waves_move_and_stay_within_their_amplitude() {
        let w = Water::default();
        let total: f32 = w.waves.iter().map(|w| w.amplitude).sum();
        let mut seen = (f32::MAX, f32::MIN);
        for k in 0..200 {
            let (h, n) = w.surface(Vec2::new(k as f32 * 0.37, 1.3), k as f32 * 0.05);
            assert!(h.abs() <= total + 1e-4);
            assert!(n.y > 0.5, "normals point up");
            seen = (seen.0.min(h), seen.1.max(h));
        }
        assert!(seen.1 - seen.0 > total * 0.8, "the surface actually moves");
    }

    #[test]
    fn wind_raises_a_sea_that_shallow_water_slows_and_caps() {
        let mut deep = Water { waves: Vec::new(), size: [0.0, 0.0], depth: 50.0, ..Default::default() };
        for _ in 0..600 {
            deep.feel_wind(Vec2::new(8.0, 0.0), 0.1);
        }
        assert!(deep.runtime.wind_speed > 7.5, "the sea built up: {}", deep.runtime.wind_speed);
        let waves = deep.resolved_waves();
        assert_eq!(waves.len(), 4);
        assert!(waves[0].dir.x > 0.99, "main wave runs downwind");
        assert!(waves[0].amplitude > 0.3, "8 m/s raises real waves: {}", waves[0].amplitude);
        let mut shallow = Water { depth: 0.4, ..deep.clone() };
        shallow.runtime = deep.runtime;
        let sw = shallow.resolved_waves();
        assert!(sw[0].amplitude <= 0.39 * 0.4 + 1e-6, "capped by depth");
        assert!(sw[0].omega < waves[0].omega, "shallow water is slower");
        for _ in 0..600 {
            deep.feel_wind(Vec2::ZERO, 0.1);
        }
        assert!(deep.runtime.wind_speed < 0.5, "and it settles when the wind drops");
    }

    #[test]
    fn lakes_rivers_and_oceans_cover_their_shapes() {
        let calm = |body| Water { body, waves: Vec::new(), ..Default::default() };
        let ocean = calm(WaterBody::Ocean);
        assert!(ocean.covers(Vec2::new(1e4, -1e4)));
        let mut lake = calm(WaterBody::Lake);
        lake.shape = vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        assert!(lake.covers(Vec2::new(2.0, 2.0)) && !lake.covers(Vec2::new(8.0, 8.0)));
        let mut river = calm(WaterBody::River);
        river.river = vec![
            RiverPoint { position: [0.0, 2.0, 0.0], width: 4.0, depth: 1.0, speed: 2.0 },
            RiverPoint { position: [20.0, 0.0, 0.0], width: 2.0, depth: 0.5, speed: 4.0 },
        ];
        let mid = river.column(Vec2::new(10.0, 0.5)).unwrap();
        assert!((mid.surface - 1.0).abs() < 1e-4, "surface follows the slope");
        assert!(mid.flow.x > 2.9 && mid.flow.y < 0.0, "flows downstream and downhill: {}", mid.flow);
        assert!(river.column(Vec2::new(10.0, 2.0)).is_none(), "outside the 3 m channel there");
        let (h, _) = river.height_at(Mat4::IDENTITY, Vec3::new(10.0, 0.5, 0.0), 0.0).unwrap();
        assert!((h - 1.0).abs() < 1e-4);
        assert!(river.height_at(Mat4::IDENTITY, Vec3::new(10.0, -1.0, 0.0), 0.0).is_none(), "below the river bed");
    }

    fn bend() -> Water {
        let p = |x: f32, z: f32, width: f32| RiverPoint { position: [x, 1.0, z], width, depth: 1.0, speed: 2.0 };
        Water { body: WaterBody::River, river: vec![p(0.0, 0.0, 4.0), p(20.0, 0.0, 4.0), p(20.0, 20.0, 8.0)], ..Default::default() }
    }

    #[test]
    fn river_sections_run_along_the_points() {
        let w = bend();
        let (sections, at_point) = w.river_sections();
        assert_eq!(at_point.len(), 3);
        for (k, &s) in at_point.iter().enumerate() {
            assert_eq!(sections[s].center, Vec3::from(w.river[k].position), "point {k} sits on a section");
            assert_eq!(sections[s].width, w.river[k].width);
        }
        assert_eq!(*at_point.last().unwrap(), sections.len() - 1);
        assert!((sections[0].side - Vec3::Z).length() < 1e-3, "across a +X flow");
        assert!((sections[sections.len() - 1].side - Vec3::NEG_X).length() < 1e-3, "across a +Z flow");
        let corner = sections[at_point[1]].side;
        assert!((corner.length() - 1.0).abs() < 1e-5 && corner.x < 0.0 && corner.z > 0.0, "mitred at the bend: {corner}");
        for pair in sections.windows(2) {
            let gap = pair[0].center.distance(pair[1].center);
            assert!(gap > 0.0 && gap <= 3.5, "about half a width apart: {gap}");
        }
        assert!(Water { river: vec![RiverPoint::default()], ..bend() }.river_sections().0.is_empty());
    }

    #[test]
    fn new_points_go_past_the_ends_or_into_the_nearest_segment() {
        let w = bend();
        assert_eq!(w.river_insert_index(Vec2::new(-5.0, 0.5)), 0, "before the source");
        assert_eq!(w.river_insert_index(Vec2::new(20.5, 30.0)), 3, "past the mouth");
        assert_eq!(w.river_insert_index(Vec2::new(10.0, -2.0)), 1, "beside the first segment");
        assert_eq!(w.river_insert_index(Vec2::new(22.0, 12.0)), 2, "beside the second");
        assert_eq!(Water { river: Vec::new(), ..bend() }.river_insert_index(Vec2::ZERO), 0);
        let (seg, t, d) = w.river_nearest_segment(Vec2::new(5.0, 3.0)).unwrap();
        assert_eq!(seg, 0);
        assert!((t - 0.25).abs() < 0.06 && (d - 3.0).abs() < 0.5, "{t} {d}");
    }

    #[test]
    fn inserted_points_blend_their_neighbours() {
        let mut w = bend();
        let mid = w.river_point_on_segment(1, 0.5).unwrap();
        let p = Vec3::from(mid.position);
        assert!((p.z - 10.0).abs() < 1.5 && (p.x - 20.0).abs() < 2.0 && (p.y - 1.0).abs() < 1e-4, "on the curve near the segment's middle: {p}");
        assert_eq!(mid.width, 6.0);
        w.insert_river_point(2, Vec3::new(21.0, 3.0, 15.0));
        assert_eq!(w.river.len(), 4);
        assert_eq!(w.river[2].position, [21.0, 3.0, 15.0], "keeps where it was placed");
        assert!((w.river[2].width - 7.0).abs() < 1e-4, "width from 3/4 along the segment");
        w.insert_river_point(99, Vec3::new(20.0, 0.0, 40.0));
        assert_eq!(w.river.len(), 5);
        assert_eq!(w.river[4].width, 8.0, "a new end copies the old end");
        let mut empty = Water { river: Vec::new(), ..bend() };
        empty.insert_river_point(0, Vec3::ONE);
        assert_eq!(empty.river[0].position, [1.0; 3]);
        assert_eq!(empty.river[0].width, RiverPoint::default().width);
    }

    #[test]
    fn surface_lookup_inverts_the_horizontal_motion() {
        let w = Water::default();
        let rest = Vec2::new(1.7, -2.2);
        let t = 3.4;
        let (d, _) = w.displacement(rest, t);
        let moved = rest + Vec2::new(d.x, d.z);
        let (h, _) = w.surface(moved, t);
        assert!((h - d.y).abs() < 1e-3, "{h} vs {}", d.y);
    }

    #[test]
    fn depth_below_the_surface() {
        let w = Water { waves: Vec::new(), ..Default::default() };
        let world = Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        let (d, h) = w.depth_below(world, Vec3::new(1.0, 0.5, 1.0), 0.0).unwrap();
        assert!((d - 1.5).abs() < 1e-5 && (h - 2.0).abs() < 1e-5);
        assert!(w.depth_below(world, Vec3::new(1.0, 3.0, 1.0), 0.0).unwrap().0 < 0.0, "above reads negative");
        assert!(w.depth_below(world, Vec3::new(50.0, 0.5, 0.0), 0.0).is_none());
    }

    #[test]
    fn old_water_files_get_underwater_defaults() {
        let w: Water = ron::from_str("(depth: 3.0)").unwrap();
        assert_eq!(w.underwater, UnderwaterSettings::default());
        assert!(w.underwater.enabled && w.underwater.sound.is_empty());
    }
}
