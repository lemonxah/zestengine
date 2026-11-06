//! Transform constraints: an object follows one or more source objects by
//! position, rotation, scale, a full parent link or by aiming at them.
//!
//! The maths ([`solve_world`]) is pure and works on world matrices, so the
//! same solver evaluates the [`Constraint`] component every frame at runtime
//! and bakes constraints from a model file into plain animation curves at
//! import (the importer reads FBX constraint nodes into [`ConstraintSpec`]s).
//!
//! Semantics (this engine's own, written from the FBX constraint attributes):
//!
//! * `Position`: the world position becomes the weighted mean of the sources'
//!   positions (each moved by its offset) plus the constraint's own offset.
//! * `Rotation`: the weighted mean of the sources' rotations, then the own offset.
//! * `Scale`: the weighted mean of the sources' world scales times the own offset.
//! * `Parent`: the object keeps the per-source offset it was authored with and
//!   follows the weighted blend of `source_world * offset` (translation,
//!   rotation and, if enabled, scale), then the own offset.
//! * `Aim`: the local aim axis turns towards the weighted mean source
//!   position; an up vector keeps the roll (scene up, a vector, towards a node
//!   or aligned with a node).
//!
//! Every constraint has a total `weight` (0 = off, 1 = fully constrained: the
//! result is blended from the unconstrained pose) and per-axis switches for
//! translation, rotation (as XYZ Euler angles) and scale.

use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ComponentCtx, ObjectRef, TypedComponent};

/// What a constraint does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Follow the sources as if parented to them (keeps the authored offsets).
    #[default]
    Parent,
    /// The world position becomes the weighted mean of the sources.
    Position,
    /// The rotation becomes the weighted mean of the sources.
    Rotation,
    /// The scale becomes the weighted mean of the sources.
    Scale,
    /// Turn the aim axis towards the sources.
    Aim,
}

impl ConstraintKind {
    /// Every kind, in declaration order.
    pub const ALL: [ConstraintKind; 5] = [Self::Parent, Self::Position, Self::Rotation, Self::Scale, Self::Aim];
    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Parent => "Parent",
            Self::Position => "Position",
            Self::Rotation => "Rotation",
            Self::Scale => "Scale",
            Self::Aim => "Aim",
        }
    }
}

/// How an aim constraint keeps the roll.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AimUp {
    /// The object's up axis follows the scene's +Y.
    #[default]
    Scene,
    /// The up axis points at the up object.
    ToObject,
    /// The up axis copies the up object's own up (Y) axis.
    AlignObject,
    /// The up axis follows a fixed world vector.
    Vector,
    /// No roll control (the shortest turn).
    None,
}

/// A translation / rotation / scale offset.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrsOffset {
    /// Translation offset.
    pub translation: [f32; 3],
    /// Quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Scale offset.
    pub scale: [f32; 3],
}

impl Default for TrsOffset {
    fn default() -> Self {
        Self { translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
    }
}

impl TrsOffset {
    /// Build an offset.
    pub fn new(t: Vec3, r: Quat, s: Vec3) -> Self {
        Self { translation: t.to_array(), rotation: r.to_array(), scale: s.to_array() }
    }
    /// The translation.
    pub fn t(&self) -> Vec3 {
        Vec3::from_array(self.translation)
    }
    /// The rotation (normalised).
    pub fn r(&self) -> Quat {
        Quat::from_array(self.rotation).normalize()
    }
    /// The scale.
    pub fn s(&self) -> Vec3 {
        Vec3::from_array(self.scale)
    }
    /// The offset as a transform matrix.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.s(), self.r(), self.t())
    }
    /// True when the offset is the identity.
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }
}

/// One source of a constraint, by node name (the importer's form, resolved
/// to objects when a model is placed).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceSpec {
    /// The source node's name.
    pub node: String,
    /// How much this source contributes (relative to the other sources).
    pub weight: f32,
    /// The authored offset from the source (parent constraints).
    pub offset: TrsOffset,
}

/// A constraint as a model file describes it, with nodes by name. Produced by
/// the FBX importer, cached with the model, turned into a [`Constraint`]
/// component on the constrained object (or baked into curves).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConstraintSpec {
    /// The constrained node.
    pub node: String,
    /// What the constraint does.
    pub kind: ConstraintKind,
    /// Overall strength 0..1.
    pub weight: f32,
    /// The constraint applies.
    pub active: bool,
    /// Per-axis translation switches.
    pub translation_axes: [bool; 3],
    /// Per-axis rotation switches (XYZ Euler).
    pub rotation_axes: [bool; 3],
    /// Per-axis scale switches.
    pub scale_axes: [bool; 3],
    /// The constraint's own offset.
    pub offset: TrsOffset,
    /// The sources.
    pub sources: Vec<SourceSpec>,
    /// Aim: the local axis that points at the sources.
    pub aim_axis: [f32; 3],
    /// How the aim roll is controlled.
    pub aim_up: AimUp,
    /// Aim: the local up axis, and for `AimUp::Vector` the world up vector.
    pub up_axis: [f32; 3],
    /// Aim: the world up vector (Vector mode).
    pub up_vector: [f32; 3],
    /// Aim: the up object's node name.
    pub up_node: String,
}

impl Default for ConstraintSpec {
    fn default() -> Self {
        Self {
            node: String::new(),
            kind: ConstraintKind::Parent,
            weight: 1.0,
            active: true,
            translation_axes: [true; 3],
            rotation_axes: [true; 3],
            scale_axes: [false; 3],
            offset: TrsOffset::default(),
            sources: Vec::new(),
            aim_axis: [1.0, 0.0, 0.0],
            aim_up: AimUp::Scene,
            up_axis: [0.0, 1.0, 0.0],
            up_vector: [0.0, 1.0, 0.0],
            up_node: String::new(),
        }
    }
}

impl ConstraintSpec {
    /// Every node the constraint reads (its sources and the up object).
    pub fn source_nodes(&self) -> impl Iterator<Item = &str> {
        self.sources.iter().map(|s| s.node.as_str()).chain((!self.up_node.is_empty()).then_some(self.up_node.as_str()))
    }
}

/// A source with its world matrix, for [`solve_world`].
#[derive(Clone, Copy, Debug)]
pub struct SourceWorld {
    /// The source's world matrix.
    pub world: Mat4,
    /// How much this source contributes.
    pub weight: f32,
    /// The authored offset from the source.
    pub offset: TrsOffset,
}

/// The settings [`solve_world`] reads (shared by the spec and the component).
#[derive(Clone, Copy, Debug)]
pub struct SolveParams {
    /// What the constraint does.
    pub kind: ConstraintKind,
    /// Overall strength 0..1.
    pub weight: f32,
    /// Per-axis translation switches.
    pub translation_axes: [bool; 3],
    /// Per-axis rotation switches (XYZ Euler).
    pub rotation_axes: [bool; 3],
    /// Per-axis scale switches.
    pub scale_axes: [bool; 3],
    /// The constraint's own offset.
    pub offset: TrsOffset,
    /// Aim: the local axis that points at the sources.
    pub aim_axis: Vec3,
    /// How the aim roll is controlled.
    pub aim_up: AimUp,
    /// Aim: the local up axis.
    pub up_axis: Vec3,
    /// Aim: the world up vector (Vector mode).
    pub up_vector: Vec3,
}

impl ConstraintSpec {
    /// The settings as [`SolveParams`].
    pub fn params(&self) -> SolveParams {
        SolveParams {
            kind: self.kind,
            weight: self.weight,
            translation_axes: self.translation_axes,
            rotation_axes: self.rotation_axes,
            scale_axes: self.scale_axes,
            offset: self.offset,
            aim_axis: Vec3::from_array(self.aim_axis),
            aim_up: self.aim_up,
            up_axis: Vec3::from_array(self.up_axis),
            up_vector: Vec3::from_array(self.up_vector),
        }
    }
}

fn blend_rotations(items: &[(Quat, f32)]) -> Quat {
    let mut acc = Quat::IDENTITY;
    let mut total = 0.0;
    for (q, w) in items {
        if *w <= 0.0 {
            continue;
        }
        if total == 0.0 {
            acc = *q;
            total = *w;
        } else {
            // Running weighted slerp keeps neighbours in one hemisphere.
            let t = w / (total + w);
            acc = acc.slerp(*q, t);
            total += w;
        }
    }
    acc.normalize()
}

/// `world` with the components `rest` has on the axes the switches turn off,
/// blended `weight` of the way from `rest`.
fn mix_axes(rest: &Mat4, target: &Mat4, p: &SolveParams) -> Mat4 {
    let (rs, rr, rt) = rest.to_scale_rotation_translation();
    let (ts, tr, tt) = target.to_scale_rotation_translation();
    let w = p.weight.clamp(0.0, 1.0);
    let pick = |axes: [bool; 3], rest: Vec3, target: Vec3| Vec3::new(
        if axes[0] { target.x } else { rest.x },
        if axes[1] { target.y } else { rest.y },
        if axes[2] { target.z } else { rest.z },
    );
    let t = rt.lerp(pick(p.translation_axes, rt, tt), w);
    let s = rs.lerp(pick(p.scale_axes, rs, ts), w);
    let rot = if p.rotation_axes.iter().all(|a| *a) {
        rr.slerp(if rr.dot(tr) < 0.0 { -tr } else { tr }, w)
    } else if p.rotation_axes.iter().all(|a| !*a) {
        rr
    } else {
        let (ex, ey, ez) = rr.to_euler(EulerRot::XYZ);
        let (tx, ty, tz) = tr.to_euler(EulerRot::XYZ);
        let e = pick(p.rotation_axes, Vec3::new(ex, ey, ez), Vec3::new(tx, ty, tz));
        let full = Quat::from_euler(EulerRot::XYZ, e.x, e.y, e.z);
        rr.slerp(if rr.dot(full) < 0.0 { -full } else { full }, w)
    };
    Mat4::from_scale_rotation_translation(s, rot.normalize(), t)
}

/// The world matrix a constrained object gets: `rest_world` is its own
/// unconstrained world matrix, `sources` the sources' current worlds, `up` the
/// aim up object's world (when the mode wants one). With no usable source (or
/// zero weight) the rest matrix comes back unchanged.
pub fn solve_world(p: &SolveParams, rest_world: Mat4, sources: &[SourceWorld], up: Option<Mat4>) -> Mat4 {
    let active: Vec<&SourceWorld> = sources.iter().filter(|s| s.weight > 0.0).collect();
    let total: f32 = active.iter().map(|s| s.weight).sum();
    if active.is_empty() || total <= 1e-9 || p.weight <= 0.0 {
        return rest_world;
    }
    let (rest_s, rest_r, rest_t) = rest_world.to_scale_rotation_translation();
    let wn = |s: &SourceWorld| s.weight / total;
    let target = match p.kind {
        ConstraintKind::Position => {
            let pos: Vec3 = active.iter().map(|s| (s.world * s.offset.t().extend(1.0)).truncate() * wn(s)).sum();
            Mat4::from_scale_rotation_translation(rest_s, rest_r, pos + p.offset.t())
        }
        ConstraintKind::Rotation => {
            let q = blend_rotations(&active.iter().map(|s| (s.world.to_scale_rotation_translation().1 * s.offset.r(), wn(s))).collect::<Vec<_>>());
            Mat4::from_scale_rotation_translation(rest_s, (q * p.offset.r()).normalize(), rest_t)
        }
        ConstraintKind::Scale => {
            let sc: Vec3 = active.iter().map(|s| s.world.to_scale_rotation_translation().0 * s.offset.s() * wn(s)).sum();
            Mat4::from_scale_rotation_translation(sc * p.offset.s(), rest_r, rest_t)
        }
        ConstraintKind::Parent => {
            let mut t = Vec3::ZERO;
            let mut sc = Vec3::ZERO;
            let mut rots = Vec::with_capacity(active.len());
            for s in &active {
                let m = s.world * s.offset.matrix();
                let (ms, mr, mt) = m.to_scale_rotation_translation();
                t += mt * wn(s);
                sc += ms * wn(s);
                rots.push((mr, wn(s)));
            }
            let blended = Mat4::from_scale_rotation_translation(sc, blend_rotations(&rots), t);
            blended * p.offset.matrix()
        }
        ConstraintKind::Aim => {
            let aim_pos: Vec3 = active.iter().map(|s| (s.world * s.offset.t().extend(1.0)).truncate() * wn(s)).sum();
            let dir = (aim_pos - rest_t).normalize_or_zero();
            if dir == Vec3::ZERO {
                return rest_world;
            }
            let a = p.aim_axis.normalize_or(Vec3::X);
            let up_world = match p.aim_up {
                AimUp::Scene => Some(Vec3::Y),
                AimUp::Vector => Some(p.up_vector.normalize_or(Vec3::Y)),
                AimUp::ToObject => up.map(|u| (u.w_axis.truncate() - rest_t).normalize_or_zero()).filter(|v| *v != Vec3::ZERO),
                AimUp::AlignObject => up.map(|u| u.transform_vector3(Vec3::Y).normalize_or(Vec3::Y)),
                AimUp::None => None,
            };
            let rot = aim_rotation(a, p.up_axis.normalize_or(Vec3::Y), dir, up_world) * p.offset.r();
            Mat4::from_scale_rotation_translation(rest_s, rot.normalize(), rest_t)
        }
    };
    mix_axes(&rest_world, &target, p)
}

/// A rotation taking the local `aim` axis to `dir`, and (when given) turning
/// the local `up` axis as close to `up_world` as the aim allows.
pub fn aim_rotation(aim: Vec3, up: Vec3, dir: Vec3, up_world: Option<Vec3>) -> Quat {
    let primary = Quat::from_rotation_arc(aim, dir);
    let Some(u) = up_world else { return primary };
    // Where the up axis ends up after the primary turn, against where it should be.
    let cur = (primary * up) - dir * (primary * up).dot(dir);
    let want = u - dir * u.dot(dir);
    if cur.length_squared() < 1e-12 || want.length_squared() < 1e-12 {
        return primary;
    }
    let (cur, want) = (cur.normalize(), want.normalize());
    let angle = cur.dot(want).clamp(-1.0, 1.0).acos();
    let sign = if cur.cross(want).dot(dir) < 0.0 { -1.0 } else { 1.0 };
    Quat::from_axis_angle(dir, angle * sign) * primary
}

// ----------------------------------------------------------------- component

/// A resolved source of a [`Constraint`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConstraintSource {
    /// The source object.
    pub target: ObjectRef,
    /// How much this source contributes.
    pub weight: f32,
    /// The authored offset from the source.
    pub offset: TrsOffset,
}

/// Makes the object follow other objects (see the module docs). Evaluated in
/// `late_update`, from the sources' transforms as of the start of the frame.
/// The object's own transform is its rest pose: capture happens at start, so
/// an object an animation drives should use weight 1.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Constraint {
    /// What the constraint does.
    pub kind: ConstraintKind,
    /// Overall strength 0..1 (0 = off).
    pub weight: f32,
    /// Per-axis translation switches.
    pub translation_axes: [bool; 3],
    /// Per-axis rotation switches (XYZ Euler).
    pub rotation_axes: [bool; 3],
    /// Per-axis scale switches.
    pub scale_axes: [bool; 3],
    /// The constraint's own offset.
    pub offset: TrsOffset,
    /// The sources.
    pub sources: Vec<ConstraintSource>,
    /// Aim: the local axis that points at the sources.
    pub aim_axis: [f32; 3],
    /// How the aim roll is controlled.
    pub aim_up: AimUp,
    /// Aim: the local up axis.
    pub up_axis: [f32; 3],
    /// Aim: the world up vector (Vector mode).
    pub up_vector: [f32; 3],
    /// Aim: the up object (ToObject / AlignObject modes).
    pub up_object: ObjectRef,
    /// Names of sources that could not be found when a model was placed
    /// (kept so the inspector can say why a constraint does nothing).
    pub unresolved: Vec<String>,
    #[serde(skip)]
    rest: Option<(Vec3, Quat, Vec3)>,
}

impl Default for Constraint {
    fn default() -> Self {
        let s = ConstraintSpec::default();
        Self {
            kind: s.kind,
            weight: 1.0,
            translation_axes: s.translation_axes,
            rotation_axes: s.rotation_axes,
            scale_axes: s.scale_axes,
            offset: TrsOffset::default(),
            sources: Vec::new(),
            aim_axis: s.aim_axis,
            aim_up: s.aim_up,
            up_axis: s.up_axis,
            up_vector: s.up_vector,
            up_object: ObjectRef::NONE,
            unresolved: Vec::new(),
            rest: None,
        }
    }
}

impl Constraint {
    /// A component from a spec; `resolve` finds the object of a node name.
    /// Sources it cannot find are dropped and listed in `unresolved`.
    pub fn from_spec(spec: &ConstraintSpec, resolve: impl Fn(&str) -> Option<crate::ObjectId>) -> Self {
        let mut c = Self {
            kind: spec.kind,
            weight: if spec.active { spec.weight } else { 0.0 },
            translation_axes: spec.translation_axes,
            rotation_axes: spec.rotation_axes,
            scale_axes: spec.scale_axes,
            offset: spec.offset,
            aim_axis: spec.aim_axis,
            aim_up: spec.aim_up,
            up_axis: spec.up_axis,
            up_vector: spec.up_vector,
            ..Default::default()
        };
        for s in &spec.sources {
            match resolve(&s.node) {
                Some(id) => c.sources.push(ConstraintSource { target: ObjectRef::new(id), weight: s.weight, offset: s.offset }),
                None => c.unresolved.push(s.node.clone()),
            }
        }
        if !spec.up_node.is_empty() {
            match resolve(&spec.up_node) {
                Some(id) => c.up_object = ObjectRef::new(id),
                None => c.unresolved.push(spec.up_node.clone()),
            }
        }
        c
    }

    fn params(&self) -> SolveParams {
        SolveParams {
            kind: self.kind,
            weight: self.weight,
            translation_axes: self.translation_axes,
            rotation_axes: self.rotation_axes,
            scale_axes: self.scale_axes,
            offset: self.offset,
            aim_axis: Vec3::from_array(self.aim_axis),
            aim_up: self.aim_up,
            up_axis: Vec3::from_array(self.up_axis),
            up_vector: Vec3::from_array(self.up_vector),
        }
    }
}

impl TypedComponent for Constraint {
    const NAME: &'static str = "Constraint";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.rest = Some((*ctx.translation, *ctx.rotation, *ctx.scale));
    }

    fn late_update(&mut self, ctx: &mut ComponentCtx) {
        let Some((t, r, s)) = self.rest else {
            self.rest = Some((*ctx.translation, *ctx.rotation, *ctx.scale));
            return;
        };
        let world_of = |id: Option<crate::ObjectId>| id.and_then(|id| ctx.id_index.get(&id)).and_then(|i| ctx.world_transforms.get(*i)).copied();
        let sources: Vec<SourceWorld> = self
            .sources
            .iter()
            .filter_map(|s| world_of(s.target.id()).map(|world| SourceWorld { world, weight: s.weight, offset: s.offset }))
            .collect();
        if sources.is_empty() {
            return;
        }
        let parent = ctx.parent_world.unwrap_or(Mat4::IDENTITY);
        let rest_world = parent * Mat4::from_scale_rotation_translation(s, r, t);
        let world = solve_world(&self.params(), rest_world, &sources, world_of(self.up_object.id()));
        let local = parent.inverse() * world;
        let (ls, lr, lt) = local.to_scale_rotation_translation();
        if lt.is_finite() && lr.is_finite() && ls.is_finite() {
            *ctx.translation = lt;
            *ctx.rotation = lr;
            *ctx.scale = ls;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(world: Mat4, weight: f32) -> SourceWorld {
        SourceWorld { world, weight, offset: TrsOffset::default() }
    }
    fn params(kind: ConstraintKind) -> SolveParams {
        let mut s = ConstraintSpec { kind, ..Default::default() };
        if kind == ConstraintKind::Scale {
            s.scale_axes = [true; 3];
        }
        s.params()
    }

    #[test]
    fn position_follows_the_weighted_mean_and_keeps_rotation() {
        let rest = Mat4::from_rotation_translation(Quat::from_rotation_y(0.5), Vec3::new(1.0, 1.0, 1.0));
        let a = src(Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)), 1.0);
        let b = src(Mat4::from_translation(Vec3::new(0.0, 10.0, 0.0)), 3.0);
        let out = solve_world(&params(ConstraintKind::Position), rest, &[a, b], None);
        let (_, r, t) = out.to_scale_rotation_translation();
        assert!((t - Vec3::new(2.5, 7.5, 0.0)).length() < 1e-5, "{t:?}");
        assert!(r.angle_between(Quat::from_rotation_y(0.5)) < 1e-5);
    }

    #[test]
    fn weight_blends_from_the_rest_pose_and_axes_mask() {
        let rest = Mat4::from_translation(Vec3::ZERO);
        let a = src(Mat4::from_translation(Vec3::new(10.0, 10.0, 10.0)), 1.0);
        let mut p = params(ConstraintKind::Position);
        p.weight = 0.5;
        p.translation_axes = [true, false, true];
        let t = solve_world(&p, rest, &[a], None).w_axis.truncate();
        assert!((t - Vec3::new(5.0, 0.0, 5.0)).length() < 1e-5, "{t:?}");
        p.weight = 0.0;
        assert_eq!(solve_world(&p, rest, &[a], None), rest, "weight 0 is the rest pose");
        // No usable source: unchanged.
        p.weight = 1.0;
        assert_eq!(solve_world(&p, rest, &[src(Mat4::IDENTITY, 0.0)], None), rest);
    }

    #[test]
    fn rotation_blends_sources_and_applies_the_offset() {
        let q1 = Quat::from_rotation_y(0.0);
        let q2 = Quat::from_rotation_y(1.0);
        let a = src(Mat4::from_quat(q1), 1.0);
        let b = src(Mat4::from_quat(q2), 1.0);
        let out = solve_world(&params(ConstraintKind::Rotation), Mat4::from_translation(Vec3::X), &[a, b], None);
        let (_, r, t) = out.to_scale_rotation_translation();
        assert!(r.angle_between(Quat::from_rotation_y(0.5)) < 1e-4);
        assert!((t - Vec3::X).length() < 1e-6, "rotation leaves the position alone");
        let mut p = params(ConstraintKind::Rotation);
        p.offset.rotation = Quat::from_rotation_x(0.25).to_array();
        let r2 = solve_world(&p, Mat4::IDENTITY, &[a], None).to_scale_rotation_translation().1;
        assert!(r2.angle_between(q1 * Quat::from_rotation_x(0.25)) < 1e-5);
    }

    #[test]
    fn scale_constraint_takes_the_source_scale() {
        let a = src(Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)), 1.0);
        let out = solve_world(&params(ConstraintKind::Scale), Mat4::IDENTITY, &[a], None);
        let s = out.to_scale_rotation_translation().0;
        assert!((s - Vec3::new(2.0, 3.0, 4.0)).length() < 1e-5);
    }

    #[test]
    fn parent_constraint_keeps_the_authored_offset() {
        // Source turned 90 degrees about Y at the origin; the child sits 1 m along the source's X.
        let source = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let s = SourceWorld { world: source, weight: 1.0, offset: TrsOffset::new(Vec3::X, Quat::IDENTITY, Vec3::ONE) };
        let out = solve_world(&params(ConstraintKind::Parent), Mat4::IDENTITY, &[s], None);
        let (_, r, t) = out.to_scale_rotation_translation();
        // Source X turned about Y by 90 degrees points along -Z.
        assert!((t - Vec3::NEG_Z).length() < 1e-5, "{t:?}");
        assert!(r.dot(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)).abs() > 0.99999);
        // Two sources at half weight each meet in the middle.
        let a = src(Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)), 1.0);
        let b = src(Mat4::from_translation(Vec3::new(-2.0, 4.0, 0.0)), 1.0);
        let mid = solve_world(&params(ConstraintKind::Parent), Mat4::IDENTITY, &[a, b], None).w_axis.truncate();
        assert!((mid - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn aim_points_the_axis_at_the_target_and_keeps_up() {
        let rest = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0));
        let target = src(Mat4::from_translation(Vec3::new(0.0, 1.0, -5.0)), 1.0);
        let p = params(ConstraintKind::Aim); // aim +X, up +Y, scene up
        let out = solve_world(&p, rest, &[target], None);
        let (_, r, _) = out.to_scale_rotation_translation();
        let aimed = r * Vec3::X;
        assert!((aimed - Vec3::NEG_Z).length() < 1e-5, "{aimed:?}");
        assert!(((r * Vec3::Y) - Vec3::Y).length() < 1e-5, "the up axis stays up");
        // Aiming up the sky still leaves a valid rotation.
        let above = src(Mat4::from_translation(Vec3::new(0.0, 9.0, 0.0)), 1.0);
        let r2 = solve_world(&p, rest, &[above], None).to_scale_rotation_translation().1;
        assert!(((r2 * Vec3::X) - Vec3::Y).length() < 1e-4 && r2.is_normalized());
        // Up object: the up axis turns towards it.
        let mut q = p;
        q.aim_up = AimUp::ToObject;
        let up_obj = Mat4::from_translation(Vec3::new(5.0, 1.0, 0.0));
        let r3 = solve_world(&q, rest, &[target], Some(up_obj)).to_scale_rotation_translation().1;
        assert!(((r3 * Vec3::Y) - Vec3::X).length() < 1e-4, "{:?}", r3 * Vec3::Y);
    }

    #[test]
    fn rotation_axis_mask_keeps_the_unmasked_euler_angle() {
        let rest = Mat4::from_quat(Quat::from_euler(EulerRot::XYZ, 0.2, 0.3, 0.4));
        let a = src(Mat4::from_quat(Quat::from_euler(EulerRot::XYZ, 1.0, 1.0, 1.0)), 1.0);
        let mut p = params(ConstraintKind::Rotation);
        p.rotation_axes = [false, true, false];
        let (x, y, z) = solve_world(&p, rest, &[a], None).to_scale_rotation_translation().1.to_euler(EulerRot::XYZ);
        assert!((x - 0.2).abs() < 1e-4 && (y - 1.0).abs() < 1e-4 && (z - 0.4).abs() < 1e-4, "{x} {y} {z}");
    }

    #[test]
    fn the_component_writes_the_local_transform_under_a_parent() {
        // The component's maths: world solved, then taken back into the parent's space.
        let parent = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let rest_world = parent * Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let a = src(Mat4::from_translation(Vec3::new(10.0, 5.0, 0.0)), 1.0);
        let world = solve_world(&params(ConstraintKind::Position), rest_world, &[a], None);
        let local = (parent.inverse() * world).w_axis.truncate();
        assert!((local - Vec3::new(0.0, 5.0, 0.0)).length() < 1e-5, "{local:?}");
    }

    #[test]
    fn specs_round_trip_and_resolve_by_name() {
        let spec = ConstraintSpec {
            node: "Hand".into(),
            kind: ConstraintKind::Aim,
            sources: vec![SourceSpec { node: "Target".into(), weight: 0.5, ..Default::default() }, SourceSpec { node: "Gone".into(), weight: 1.0, ..Default::default() }],
            up_node: "Up".into(),
            ..Default::default()
        };
        let text = ron::ser::to_string(&spec).unwrap();
        let back: ConstraintSpec = ron::from_str(&text).unwrap();
        assert_eq!(back, spec);
        let (t, u) = (crate::ObjectId::new(), crate::ObjectId::new());
        let c = Constraint::from_spec(&spec, |n| match n {
            "Target" => Some(t),
            "Up" => Some(u),
            _ => None,
        });
        assert_eq!(c.sources.len(), 1);
        assert_eq!(c.unresolved, vec!["Gone".to_string()]);
        assert_eq!(c.up_object.id(), Some(u));
        // An older file without any field loads with the defaults.
        let old: ConstraintSpec = ron::from_str("()").unwrap();
        assert!(old.active && old.weight == 1.0);
    }
}
