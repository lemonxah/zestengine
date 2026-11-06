//! Sprite Shape (2D-8): a spline-driven sprite mesh. A closed spline is filled
//! (a tiled texture or a stretched sprite) and every spline segment gets an
//! edge sprite chosen by the direction the segment faces; sharp corners get
//! corner sprites and an open spline can end in caps.
//!
//! * [`SpriteShapeRenderer`] (component): the spline (control points with
//!   in / out tangents, per-point height, corner flag and sprite variant),
//!   open or closed, the tessellation settings, sorting and the collider
//!   settings. It names a `.spriteshape` profile.
//! * [`SpriteShapeProfile`] (asset, RON, extension [`SPRITE_SHAPE_EXTENSION`]):
//!   angle ranges mapping a segment's normal angle to a list of edge sprites,
//!   corner and cap sprites, the fill texture.
//! * [`spline`]: pure geometry (bezier tessellation with an error bound and a
//!   point cap, ear-clipping triangulation, polygon offset, Douglas-Peucker).
//! * [`generate`]: spline + profile -> [`generate::ShapeMesh`] (position / UV /
//!   index groups, one per texture), and the collider outline.
//! * [`edit`]: the spline edit operations the Scene-view tool applies, each with
//!   an exact inverse (pure model, unit tested).
//! * [`presets`]: starter profiles and splines (grass strip, ground with fill,
//!   rope, road).
//!
//! The engine gathers shapes in `engine/src/scene/shape2d.rs`: it rebuilds a
//! shape only when [`SpriteShapeRenderer::content_hash`] or the profile changed
//! and writes the groups into rewritable meshes of the sprite pool, so a scene
//! with no shapes pays nothing. Behaviour follows the Unity Sprite Shape
//! package as documented in `docs/research/unity7/phase5b-2d-tilemaps-physics.md`
//! section 5; the code is our own.
//!
//! Conventions: object-local XY plane, y up. A segment's *normal* is the left of
//! its travel direction; a closed spline is walked clockwise (a counter-clockwise
//! one is reversed) so the normal points out of the shape. The *angle* of a
//! segment is the normal's angle from +Y, counter-clockwise, in degrees
//! `[0, 360)`: 0 faces up, 90 left, 180 down, 270 right.

use serde::{Deserialize, Serialize};

use crate::{MaskInteraction2D, SortingLayerId, TypedComponent};

pub mod edit;
pub mod generate;
pub mod presets;
pub mod spline;

pub use edit::{SplineOp, apply_op};
pub use generate::{ColliderOutline, FillTexture, GenInput, GroupKind, MeshGroup, ShapeMesh, ShapeStats, SpriteMetrics, collider_outline, generate_shape};
pub use presets::{PresetData, ShapePreset, apply_preset, preset_shape};
pub use spline::{Poly, TessMode, tessellate};

/// File extension of `.spriteshape` profile assets.
pub const SPRITE_SHAPE_EXTENSION: &str = "spriteshape";

/// How a control point's tangents behave.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum ShapeTangentMode {
    /// No tangents: both neighbouring segments leave / arrive straight.
    #[default]
    Linear,
    /// `tangent_in` is always the mirror of `tangent_out` (smooth curve).
    Continuous,
    /// The two tangents are independent (a cusp is possible).
    Broken,
}

impl ShapeTangentMode {
    /// Every mode, in declaration order.
    pub const ALL: [ShapeTangentMode; 3] = [Self::Linear, Self::Continuous, Self::Broken];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Continuous => "Continuous",
            Self::Broken => "Broken",
        }
    }
}

fn one() -> f32 {
    1.0
}
fn yes() -> bool {
    true
}

/// One control point of the spline. Tangents are offsets from `position` (the
/// bezier handles), in object-local units.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ShapePoint {
    /// Control point position (object-local).
    pub position: [f32; 2],
    /// Incoming tangent offset from `position`.
    pub tangent_in: [f32; 2],
    /// Outgoing tangent offset from `position`.
    pub tangent_out: [f32; 2],
    /// How the tangents behave.
    pub mode: ShapeTangentMode,
    /// Scales the edge sprites' thickness at this point (interpolated between
    /// points); 1 = the sprite's own height.
    #[serde(default = "one")]
    pub height: f32,
    /// Edge sprite variant used by the segment that starts at this point
    /// (index into the angle range's sprite list, wrapped).
    pub sprite_index: u32,
    /// Allowed to become a corner when the turn exceeds the shape's threshold.
    #[serde(default = "yes")]
    pub corner: bool,
}

impl Default for ShapePoint {
    fn default() -> Self {
        Self { position: [0.0; 2], tangent_in: [0.0; 2], tangent_out: [0.0; 2], mode: ShapeTangentMode::Linear, height: 1.0, sprite_index: 0, corner: true }
    }
}

impl ShapePoint {
    /// A point at (x, y) with default handles.
    pub fn at(x: f32, y: f32) -> Self {
        Self { position: [x, y], ..Default::default() }
    }

    /// The handle offset the bezier really uses (a Linear point has none).
    pub fn out_handle(&self) -> [f32; 2] {
        if self.mode == ShapeTangentMode::Linear { [0.0; 2] } else { self.tangent_out }
    }

    /// The incoming handle offset the bezier really uses (none when Linear).
    pub fn in_handle(&self) -> [f32; 2] {
        if self.mode == ShapeTangentMode::Linear { [0.0; 2] } else { self.tangent_in }
    }

    /// Set the outgoing tangent; a Continuous point mirrors it into the incoming one.
    pub fn set_tangent_out(&mut self, t: [f32; 2]) {
        self.tangent_out = t;
        if self.mode == ShapeTangentMode::Continuous {
            self.tangent_in = [-t[0], -t[1]];
        }
    }

    /// Set the incoming tangent; a Continuous point mirrors it into the
    /// outgoing one.
    pub fn set_tangent_in(&mut self, t: [f32; 2]) {
        self.tangent_in = t;
        if self.mode == ShapeTangentMode::Continuous {
            self.tangent_out = [-t[0], -t[1]];
        }
    }

    /// Change the mode. Leaving Linear gives the point default handles along
    /// `along` (the direction between its neighbours) so it actually curves;
    /// entering Continuous mirrors the outgoing tangent; entering Linear keeps
    /// the stored handles (they come back if the mode is switched again).
    pub fn set_mode(&mut self, mode: ShapeTangentMode, along: [f32; 2], handle_len: f32) {
        if self.mode == ShapeTangentMode::Linear && mode != ShapeTangentMode::Linear && self.tangent_in == [0.0; 2] && self.tangent_out == [0.0; 2] {
            let len = (along[0] * along[0] + along[1] * along[1]).sqrt();
            if len > 1e-6 {
                let d = [along[0] / len * handle_len, along[1] / len * handle_len];
                self.tangent_out = d;
                self.tangent_in = [-d[0], -d[1]];
            }
        }
        self.mode = mode;
        if mode == ShapeTangentMode::Continuous {
            self.tangent_in = [-self.tangent_out[0], -self.tangent_out[1]];
        }
    }
}

/// How the curved segments are turned into line pieces.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Tessellation {
    /// Subdivide until the polyline is within `tolerance` of the curve.
    #[default]
    Adaptive,
    /// `spline_detail` pieces per curved segment (Unity's quality detail: 16 / 8 / 4).
    Fixed,
}

impl Tessellation {
    /// Every mode, in declaration order.
    pub const ALL: [Tessellation; 2] = [Self::Adaptive, Self::Fixed];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Adaptive => "Adaptive",
            Self::Fixed => "Fixed",
        }
    }
}

/// How edge sprites are laid along a run of segments of one angle range.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum EdgeFit {
    /// A whole number of tiles (nearest), stretched or squeezed to fill the run exactly.
    #[default]
    Stretch,
    /// A whole number of tiles that never shrink (round down), widened to fill the run exactly.
    Fit,
    /// Tiles at their own width; the last one is cut off (its UVs too).
    Tile,
}

impl EdgeFit {
    /// Every fit, in declaration order.
    pub const ALL: [EdgeFit; 3] = [Self::Stretch, Self::Fit, Self::Tile];
    /// Inspector label of this fit.
    pub fn label(self) -> &'static str {
        match self {
            Self::Stretch => "Stretch",
            Self::Fit => "Fit (never shrink)",
            Self::Tile => "Tile (cut last)",
        }
    }
}

fn detail_default() -> u32 {
    16
}
fn tolerance_default() -> f32 {
    0.02
}
fn max_points_default() -> u32 {
    2048
}
fn corner_default() -> f32 {
    30.0
}
fn collider_opt_default() -> f32 {
    0.01
}

/// An inner closed spline that cuts a hole out of the fill of a closed shape (and gets its own
/// edge sprites, facing into the hole, and its own collider path). Same tessellation settings as
/// the outer spline. A hole should lie inside the outer outline and not overlap another hole.
#[derive(Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ShapeHole {
    /// The hole's control points (an inner closed spline).
    pub points: Vec<ShapePoint>,
}

impl ShapeHole {
    /// A regular `n`-gon (linear points) of radius `r` around `c`.
    pub fn ngon(c: [f32; 2], r: f32, n: usize) -> Self {
        let n = n.max(3);
        Self { points: (0..n).map(|i| {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            ShapePoint::at(c[0] + r * a.cos(), c[1] + r * a.sin())
        }).collect() }
    }
}

/// The Sprite Shape component: spline + settings of one shape. Needs a
/// `.spriteshape` profile for sprites; without one it draws nothing.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteShapeRenderer {
    /// Project-relative `.spriteshape` path.
    pub profile: String,
    /// The spline's control points.
    pub points: Vec<ShapePoint>,
    /// Inner closed splines subtracted from the fill of a closed shape.
    pub holes: Vec<ShapeHole>,
    /// Closed shapes are filled and walked clockwise; open ones are a strip.
    pub closed: bool,
    /// Draw the fill of a closed shape.
    pub fill: bool,
    /// How curved segments become line pieces.
    pub tessellation: Tessellation,
    /// Pieces per curved segment (`Fixed`).
    pub spline_detail: u32,
    /// Largest distance between the curve and its polyline, local units (`Adaptive`).
    pub tolerance: f32,
    /// Upper bound of tessellated points (the tolerance is relaxed to honour it).
    pub max_points: u32,
    /// A control point turning more than this (degrees) may become a corner.
    pub corner_angle_threshold: f32,
    /// How edge sprites lay along a run of one angle range.
    pub fit: EdgeFit,
    /// Walk the spline the other way: edge sprites go on the other side.
    pub flip_edge: bool,
    /// Shifts the edge sprites along the normal as a fraction of their height
    /// (0 = centred on the spline, 0.5 = fully outside).
    pub border_pivot: f32,
    /// Tint, linear RGBA.
    pub color: [f32; 4],
    /// Sorting layer.
    pub sorting_layer: SortingLayerId,
    /// Sort key inside the layer.
    pub order_in_layer: i16,
    /// How sprite masks affect the shape.
    pub mask_interaction: MaskInteraction2D,
    /// Write the outline into the Polygon (closed) / Edge (open) Collider 2D
    /// on this object whenever the shape changes.
    pub auto_update_collider: bool,
    /// Pieces per curved segment of the collider outline.
    pub collider_detail: u32,
    /// Outline offset (closed shapes), local units; positive grows the polygon.
    pub collider_offset: f32,
    /// Douglas-Peucker tolerance of the collider outline, 0 = keep every point.
    pub collider_optimize: f32,
}

impl Default for SpriteShapeRenderer {
    fn default() -> Self {
        Self {
            profile: String::new(),
            points: vec![ShapePoint::at(-2.0, 0.0), ShapePoint::at(2.0, 0.0)],
            holes: Vec::new(),
            closed: false,
            fill: true,
            tessellation: Tessellation::Adaptive,
            spline_detail: detail_default(),
            tolerance: tolerance_default(),
            max_points: max_points_default(),
            corner_angle_threshold: corner_default(),
            fit: EdgeFit::Stretch,
            flip_edge: false,
            border_pivot: 0.0,
            color: [1.0; 4],
            sorting_layer: crate::DEFAULT_SORTING_LAYER,
            order_in_layer: 0,
            mask_interaction: MaskInteraction2D::None,
            auto_update_collider: true,
            collider_detail: detail_default(),
            collider_offset: 0.0,
            collider_optimize: collider_opt_default(),
        }
    }
}

impl TypedComponent for SpriteShapeRenderer {
    const NAME: &'static str = "Sprite Shape Renderer";
}

/// FNV-1a over raw bits, the shape's "dirty version".
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn f(&mut self, v: f32) {
        self.bytes(&v.to_bits().to_le_bytes());
    }
    fn u(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
}

impl SpriteShapeRenderer {
    /// Hash of everything that changes the generated geometry (not the tint,
    /// sorting or mask settings, which are per-frame instance data). The engine
    /// rebuilds the mesh only when this or the profile changed.
    pub fn content_hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.bytes(self.profile.as_bytes());
        h.u(self.points.len() as u32);
        for p in &self.points {
            for v in p.position.iter().chain(&p.tangent_in).chain(&p.tangent_out) {
                h.f(*v);
            }
            h.u(p.mode as u32);
            h.f(p.height);
            h.u(p.sprite_index);
            h.u(p.corner as u32);
        }
        h.u(self.holes.len() as u32);
        for hole in &self.holes {
            h.u(hole.points.len() as u32);
            for p in &hole.points {
                for v in p.position.iter().chain(&p.tangent_in).chain(&p.tangent_out) {
                    h.f(*v);
                }
                h.u(p.mode as u32);
                h.f(p.height);
                h.u(p.sprite_index);
                h.u(p.corner as u32);
            }
        }
        h.u(self.closed as u32);
        h.u(self.fill as u32);
        h.u(self.tessellation as u32);
        h.u(self.spline_detail);
        h.f(self.tolerance);
        h.u(self.max_points);
        h.f(self.corner_angle_threshold);
        h.u(self.fit as u32);
        h.u(self.flip_edge as u32);
        h.f(self.border_pivot);
        h.0
    }

    /// Hash of the collider-relevant settings and the spline.
    pub fn collider_hash(&self) -> u64 {
        let mut h = Fnv(self.content_hash());
        h.u(self.collider_detail);
        h.f(self.collider_offset);
        h.f(self.collider_optimize);
        h.0
    }

    /// The tessellation settings as the geometry code takes them.
    pub fn tess_mode(&self) -> TessMode {
        match self.tessellation {
            Tessellation::Adaptive => TessMode::Adaptive { tolerance: self.tolerance.max(1e-5) },
            Tessellation::Fixed => TessMode::Fixed { detail: self.spline_detail.max(1) },
        }
    }
}

/// An angle range of a profile: segments whose normal angle falls in
/// `start..end` (degrees, counter-clockwise from up, wrapping allowed: a range
/// of `-40..40` is the top) use the range's sprites. Overlapping ranges: the
/// higher `order` wins, then the earlier in the list.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AngleRange {
    /// First angle of the range (degrees, counter-clockwise from up).
    pub start: f32,
    /// Last angle of the range (degrees).
    pub end: f32,
    /// Higher order wins when ranges overlap.
    pub order: i32,
    /// Edge sprite variants (`path#id` references); the segment's point picks one.
    pub sprites: Vec<String>,
}

impl Default for AngleRange {
    fn default() -> Self {
        Self { start: -40.0, end: 40.0, order: 0, sprites: Vec::new() }
    }
}

impl AngleRange {
    /// Whether `angle` (degrees, any value) lies in the range. A range whose
    /// span reaches 360 degrees covers everything; `start == end` covers nothing
    /// but that exact angle.
    pub fn contains(&self, angle: f32) -> bool {
        let span = self.end - self.start;
        if span >= 360.0 {
            return true;
        }
        if span < 0.0 {
            // Written with end before start (170..-170): the arc runs counter-clockwise
            // from start through the wrap (here 170 -> 190), the same as the range 170..190.
            let span = span.rem_euclid(360.0);
            return (angle - self.start).rem_euclid(360.0) <= span;
        }
        (angle - self.start).rem_euclid(360.0) <= span + 1e-4
    }
}

/// The angle range that draws a segment facing `angle`, or `None` (no edge
/// sprite there).
pub fn select_range(ranges: &[AngleRange], angle: f32) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, r) in ranges.iter().enumerate() {
        if r.contains(angle) && best.is_none_or(|b| r.order > ranges[b].order) {
            best = Some(i);
        }
    }
    best
}

fn ppu_default() -> f32 {
    100.0
}

/// The `.spriteshape` asset.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteShapeProfile {
    /// Which edge sprites each segment angle uses.
    pub angle_ranges: Vec<AngleRange>,
    /// Sprite for corners that turn away from the shape (a convex corner of a
    /// clockwise outline). Empty = a mitered join.
    pub outer_corner: String,
    /// Sprite for corners that turn into the shape (concave).
    pub inner_corner: String,
    /// Open shapes: sprite placed before the first / after the last point.
    pub start_cap: String,
    /// Open shapes: sprite placed after the last point.
    pub end_cap: String,
    /// Fill texture (project-relative path), tiled by world units: it repeats
    /// every `size_px / fill_pixels_per_unit` units (needs a Repeat wrap mode).
    pub fill_texture: String,
    /// Fill texture pixels per world unit (tiling scale).
    pub fill_pixels_per_unit: f32,
    /// Shifts the tiling pattern, world units.
    pub fill_uv_offset: [f32; 2],
    /// Fill used when there is no `fill_texture`: one sprite stretched over the
    /// shape's bounds (`path#id`).
    pub fill_sprite: String,
    /// Moves the fill outline along its normal, world units (negative tucks
    /// the fill under the edge sprites).
    pub fill_edge_offset: f32,
}

impl Default for SpriteShapeProfile {
    fn default() -> Self {
        Self {
            angle_ranges: vec![AngleRange { start: -40.0, end: 40.0, order: 0, sprites: Vec::new() }],
            outer_corner: String::new(),
            inner_corner: String::new(),
            start_cap: String::new(),
            end_cap: String::new(),
            fill_texture: String::new(),
            fill_pixels_per_unit: ppu_default(),
            fill_uv_offset: [0.0; 2],
            fill_sprite: String::new(),
            fill_edge_offset: 0.0,
        }
    }
}

impl SpriteShapeProfile {
    /// Hash of the whole profile (the engine compares it, not file times, for
    /// the live profile editor).
    pub fn content_hash(&self) -> u64 {
        let mut h = Fnv::new();
        for r in &self.angle_ranges {
            h.f(r.start);
            h.f(r.end);
            h.u(r.order as u32);
            for s in &r.sprites {
                h.bytes(s.as_bytes());
                h.bytes(&[0]);
            }
            h.bytes(&[1]);
        }
        for s in [&self.outer_corner, &self.inner_corner, &self.start_cap, &self.end_cap, &self.fill_texture, &self.fill_sprite] {
            h.bytes(s.as_bytes());
            h.bytes(&[2]);
        }
        h.f(self.fill_pixels_per_unit);
        h.f(self.fill_uv_offset[0]);
        h.f(self.fill_uv_offset[1]);
        h.f(self.fill_edge_offset);
        h.0
    }

    /// Every sprite reference the profile can draw (edge variants, corners, caps, fill).
    pub fn all_sprites(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.angle_ranges.iter().flat_map(|r| r.sprites.iter().map(String::as_str)).collect();
        v.extend([&self.outer_corner, &self.inner_corner, &self.start_cap, &self.end_cap, &self.fill_sprite].iter().map(|s| s.as_str()));
        v.retain(|s| !s.is_empty());
        v
    }

    /// The circle of angles split into arcs `(start, end, range)` by the range
    /// that wins there (`None` = no range covers it): a partition of
    /// `[0, 360)` at whole-degree resolution, in order, for the dial preview.
    pub fn resolved_arcs(&self) -> Vec<(f32, f32, Option<usize>)> {
        let pick = |d: u32| select_range(&self.angle_ranges, d as f32 + 0.5);
        let mut out: Vec<(f32, f32, Option<usize>)> = Vec::new();
        let mut start = 0u32;
        let mut cur = pick(0);
        for d in 1..=360u32 {
            let next = if d < 360 { pick(d) } else { None };
            if d == 360 || next != cur {
                out.push((start as f32, d as f32, cur));
                start = d;
                cur = next;
            }
        }
        out
    }

    /// Angles (degrees, `[0, 360)`) no range covers, as `(start, end)` arcs.
    pub fn uncovered_arcs(&self) -> Vec<(f32, f32)> {
        self.resolved_arcs().into_iter().filter(|a| a.2.is_none()).map(|a| (a.0, a.1)).collect()
    }
}

/// Read and parse a `.spriteshape` profile.
pub fn load_spriteshape(path: impl AsRef<std::path::Path>) -> Result<SpriteShapeProfile, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    ron::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

/// Serialize and write a `.spriteshape` profile.
pub fn save_spriteshape(path: impl AsRef<std::path::Path>, profile: &SpriteShapeProfile) -> Result<(), String> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(profile, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_files_load_with_defaults() {
        let s: SpriteShapeRenderer = ron::from_str("()").unwrap();
        assert_eq!(s.spline_detail, 16);
        assert_eq!(s.corner_angle_threshold, 30.0);
        assert!(s.fill && s.auto_update_collider && !s.closed);
        assert_eq!(s.points.len(), 2);
        let s: SpriteShapeRenderer = ron::from_str("(closed: true, points: [(position: (1.0, 2.0))])").unwrap();
        assert!(s.closed);
        assert_eq!(s.points[0].height, 1.0, "height defaults to 1, not 0");
        assert!(s.points[0].corner);
        let p: SpriteShapeProfile = ron::from_str("(fill_texture: \"a.png\")").unwrap();
        assert_eq!(p.fill_pixels_per_unit, 100.0);
        assert_eq!(p.angle_ranges.len(), 1);
        let r: AngleRange = ron::from_str("(sprites: [\"a#0000000000000001\"])").unwrap();
        assert_eq!((r.start, r.end), (-40.0, 40.0));
    }

    #[test]
    fn serialisation_round_trip() {
        let mut s = SpriteShapeRenderer { closed: true, profile: "p.spriteshape".into(), fit: EdgeFit::Tile, tessellation: Tessellation::Fixed, ..Default::default() };
        s.points.push(ShapePoint { position: [1.0, 3.0], tangent_in: [-0.5, 0.0], tangent_out: [0.5, 0.0], mode: ShapeTangentMode::Continuous, height: 1.5, sprite_index: 2, corner: false });
        let back: SpriteShapeRenderer = ron::from_str(&ron::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        let mut p = SpriteShapeProfile::default();
        p.angle_ranges.push(AngleRange { start: 100.0, end: 260.0, order: 3, sprites: vec!["x#0000000000000002".into()] });
        p.start_cap = "cap#0000000000000003".into();
        let back: SpriteShapeProfile = ron::from_str(&ron::ser::to_string_pretty(&p, Default::default()).unwrap()).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn file_round_trip() {
        let dir = std::env::temp_dir().join(format!("zest-spriteshape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.spriteshape");
        let p = presets::preset_shape(ShapePreset::GroundWithFill).profile;
        save_spriteshape(&path, &p).unwrap();
        assert_eq!(load_spriteshape(&path).unwrap(), p);
        assert!(load_spriteshape(dir.join("missing.spriteshape")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_changes_with_geometry_only() {
        let a = SpriteShapeRenderer::default();
        let mut b = a.clone();
        b.color = [0.5; 4];
        b.order_in_layer = 4;
        b.sorting_layer = SortingLayerId(9);
        assert_eq!(a.content_hash(), b.content_hash(), "tint and sorting are instance data");
        for edit in [
            |s: &mut SpriteShapeRenderer| s.points[0].position[0] += 0.001,
            |s: &mut SpriteShapeRenderer| s.points[1].height = 2.0,
            |s: &mut SpriteShapeRenderer| s.closed = true,
            |s: &mut SpriteShapeRenderer| s.tolerance = 0.5,
            |s: &mut SpriteShapeRenderer| s.profile = "x".into(),
            |s: &mut SpriteShapeRenderer| s.points[0].sprite_index = 1,
        ] {
            let mut c = a.clone();
            edit(&mut c);
            assert_ne!(a.content_hash(), c.content_hash());
        }
        let mut c = a.clone();
        c.collider_offset = 1.0;
        assert_eq!(a.content_hash(), c.content_hash());
        assert_ne!(a.collider_hash(), c.collider_hash());
    }

    #[test]
    fn angle_ranges_wrap_and_overlap() {
        let top = AngleRange { start: -40.0, end: 40.0, order: 0, sprites: vec![] };
        for (a, want) in [(0.0, true), (39.0, true), (41.0, false), (320.0, true), (319.0, false), (359.9, true), (-10.0, true), (400.0, true)] {
            assert_eq!(top.contains(a), want, "angle {a}");
        }
        let full = AngleRange { start: 0.0, end: 360.0, ..Default::default() };
        assert!(full.contains(123.0) && full.contains(-5.0));
        let wrapped = AngleRange { start: 170.0, end: -170.0, ..Default::default() };
        assert!(wrapped.contains(180.0) && wrapped.contains(175.0) && !wrapped.contains(0.0), "end before start wraps counter-clockwise");
        let ranges = vec![
            AngleRange { start: 0.0, end: 360.0, order: 0, sprites: vec![] }, // everything, lowest priority
            AngleRange { start: -40.0, end: 40.0, order: 2, sprites: vec![] },
            AngleRange { start: 30.0, end: 100.0, order: 2, sprites: vec![] },
            AngleRange { start: 20.0, end: 60.0, order: 5, sprites: vec![] },
        ];
        assert_eq!(select_range(&ranges, 180.0), Some(0));
        assert_eq!(select_range(&ranges, 0.0), Some(1));
        assert_eq!(select_range(&ranges, 35.0), Some(3), "higher order wins the overlap");
        assert_eq!(select_range(&ranges, 80.0), Some(2));
        assert_eq!(select_range(&ranges[1..2], 90.0), None);
        // Ties on order: the earlier range.
        assert_eq!(select_range(&ranges[1..3], 35.0), Some(0));
    }

    #[test]
    fn uncovered_arcs_report_the_gaps() {
        let mut p = SpriteShapeProfile::default();
        assert_eq!(p.uncovered_arcs(), vec![(40.0, 320.0)]);
        p.angle_ranges.push(AngleRange { start: 40.0, end: 320.0, order: 0, sprites: vec![] });
        assert!(p.uncovered_arcs().is_empty());
    }

    #[test]
    fn resolved_arcs_partition_the_circle() {
        let mut p = SpriteShapeProfile {
            angle_ranges: vec![
                AngleRange { start: -40.0, end: 40.0, order: 1, sprites: vec![] },
                AngleRange { start: 0.0, end: 360.0, order: 0, sprites: vec![] },
            ],
            ..Default::default()
        };
        let arcs = p.resolved_arcs();
        assert_eq!(arcs, vec![(0.0, 40.0, Some(0)), (40.0, 320.0, Some(1)), (320.0, 360.0, Some(0))], "the top range wraps through 0");
        assert!(arcs.windows(2).all(|w| w[0].1 == w[1].0) && arcs[0].0 == 0.0 && arcs.last().unwrap().1 == 360.0);
        p.angle_ranges.clear();
        assert_eq!(p.resolved_arcs(), vec![(0.0, 360.0, None)]);
    }

    #[test]
    fn point_modes_manage_tangents() {
        let mut p = ShapePoint::at(0.0, 0.0);
        assert_eq!(p.out_handle(), [0.0; 2]);
        p.set_mode(ShapeTangentMode::Continuous, [2.0, 0.0], 0.5);
        assert_eq!((p.tangent_out, p.tangent_in), ([0.5, 0.0], [-0.5, 0.0]));
        p.set_tangent_in([0.0, 1.0]);
        assert_eq!(p.tangent_out, [0.0, -1.0], "continuous mirrors");
        p.set_mode(ShapeTangentMode::Broken, [1.0, 0.0], 0.5);
        p.set_tangent_out([2.0, 2.0]);
        assert_eq!(p.tangent_in, [0.0, 1.0], "broken keeps them apart");
        p.set_mode(ShapeTangentMode::Linear, [1.0, 0.0], 0.5);
        assert_eq!((p.out_handle(), p.in_handle()), ([0.0; 2], [0.0; 2]));
        assert_eq!(p.tangent_out, [2.0, 2.0], "the handles survive a trip through Linear");
    }
}
