//! FBX extras beyond plain TRS curves: constraints, user (custom) properties
//! and the baked evaluation of the whole node transform stack.
//!
//! ufbx evaluates a node's transform (Lcl TRS, Pre/PostRotation, pivots,
//! offsets, inherit modes) but it does **not** solve constraints. This module
//! reads them into [`ConstraintSpec`]s, solves them with the engine's own
//! solver ([`zestengine_core::solve_world`]) on top of ufbx's evaluated scene,
//! and hands the result out as world matrices ([`Baker::frame`]) that the
//! importer turns into plain TRS curves (Bake Animations).
//!
//! Not solved: Single Chain IK constraints (counted in
//! [`ConstraintReport::unsupported`]) and driven keys (Maya exports them baked
//! or as expressions the FBX format has no record of).

use glam::{Mat4, Quat, Vec3};
use zestengine_core::{AimUp, ConstraintKind, ConstraintSpec, CustomValue, SolveParams, SourceSpec, SourceWorld, TrsOffset, solve_world};

use crate::fbx_loader::matrix_to_mat4;

fn v3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn offset_of(t: &ufbx::Transform) -> TrsOffset {
    let q = Quat::from_xyzw(t.rotation.x as f32, t.rotation.y as f32, t.rotation.z as f32, t.rotation.w as f32);
    TrsOffset::new(v3(t.translation), if q.length_squared() > 1e-12 { q.normalize() } else { Quat::IDENTITY }, v3(t.scale))
}

/// A file constraint with its nodes resolved to indices into `Scene::nodes`.
pub(crate) struct Resolved<'a> {
    pub spec: ConstraintSpec,
    pub node: usize,
    pub sources: Vec<usize>,
    pub up: Option<usize>,
    pub source: &'a ufbx::Constraint,
}

/// What reading the file's constraints found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstraintReport {
    /// Constraints turned into specs.
    pub read: usize,
    /// Constraints the importer cannot solve or place (IK chains, unknown
    /// types, no constrained node, no sources).
    pub unsupported: usize,
}

fn node_name(n: &ufbx::Node) -> String {
    if n.element.name.is_empty() { format!("node {}", n.element.element_id) } else { n.element.name.to_string() }
}

/// Every parent / position / rotation / scale / aim constraint of the file.
pub(crate) fn read_constraints(fbx: &ufbx::Scene) -> (Vec<Resolved<'_>>, ConstraintReport) {
    let mut out = Vec::new();
    let mut report = ConstraintReport::default();
    for c in &fbx.constraints {
        let kind = match c.type_ {
            ufbx::ConstraintType::Aim => ConstraintKind::Aim,
            ufbx::ConstraintType::Parent => ConstraintKind::Parent,
            ufbx::ConstraintType::Position => ConstraintKind::Position,
            ufbx::ConstraintType::Rotation => ConstraintKind::Rotation,
            ufbx::ConstraintType::Scale => ConstraintKind::Scale,
            _ => {
                report.unsupported += 1;
                continue;
            }
        };
        let Some(node) = c.node.as_ref() else {
            report.unsupported += 1;
            continue;
        };
        let mut spec = ConstraintSpec {
            node: node_name(node),
            kind,
            weight: c.weight as f32,
            active: c.active,
            translation_axes: c.constrain_translation,
            rotation_axes: c.constrain_rotation,
            scale_axes: c.constrain_scale,
            offset: offset_of(&c.transform_offset),
            ..Default::default()
        };
        let mut sources = Vec::new();
        for t in &c.targets {
            let tn = &t.node;
            spec.sources.push(SourceSpec { node: node_name(tn), weight: t.weight as f32, offset: offset_of(&t.transform) });
            sources.push(tn.element.typed_id as usize);
        }
        if spec.sources.is_empty() {
            report.unsupported += 1;
            continue;
        }
        let mut up = None;
        if kind == ConstraintKind::Aim {
            let a = v3(c.aim_vector);
            spec.aim_axis = if a.length_squared() > 1e-12 { a.to_array() } else { [1.0, 0.0, 0.0] };
            spec.aim_up = match c.aim_up_type {
                ufbx::ConstraintAimUpType::Scene => AimUp::Scene,
                ufbx::ConstraintAimUpType::ToNode => AimUp::ToObject,
                ufbx::ConstraintAimUpType::AlignNode => AimUp::AlignObject,
                ufbx::ConstraintAimUpType::Vector => AimUp::Vector,
                ufbx::ConstraintAimUpType::None => AimUp::None,
            };
            spec.up_vector = v3(c.aim_up_vector).to_array();
            if let Some(un) = c.aim_up_node.as_ref() {
                spec.up_node = node_name(un);
                up = Some(un.element.typed_id as usize);
            }
        }
        report.read += 1;
        out.push(Resolved { spec, node: node.element.typed_id as usize, sources, up, source: c });
    }
    (out, report)
}

/// The file's constraint specs without the node indices.
pub(crate) fn constraint_specs(fbx: &ufbx::Scene) -> (Vec<ConstraintSpec>, ConstraintReport) {
    let (resolved, report) = read_constraints(fbx);
    (resolved.into_iter().map(|r| r.spec).collect(), report)
}

/// Evaluates whole frames of the file: every node's world matrix as ufbx's
/// scene evaluation gives it (the complete transform stack), with the
/// constraints solved on top.
pub(crate) struct Baker<'a> {
    fbx: &'a ufbx::Scene,
    constraints: Vec<Resolved<'a>>,
    parent: Vec<Option<usize>>,
}

impl<'a> Baker<'a> {
    pub fn new(fbx: &'a ufbx::Scene, with_constraints: bool) -> Self {
        let constraints = if with_constraints { read_constraints(fbx).0 } else { Vec::new() };
        let parent = fbx.nodes.iter().map(|n| n.parent.as_ref().map(|p| p.element.typed_id as usize)).collect();
        Self { fbx, constraints, parent }
    }

    #[allow(dead_code)] // constraint introspection used by the import tests
    pub fn has_constraints(&self) -> bool {
        !self.constraints.is_empty()
    }

    fn weight_at(&self, c: &Resolved<'_>, anim: &ufbx::Anim, time: f64) -> f32 {
        if c.source.element.props.num_animated > 0 && c.source.element.props.find_prop("Weight").is_some() {
            let p = ufbx::evaluate_prop(anim, &c.source.element, "Weight", time);
            return (p.value_vec4.x / 100.0) as f32;
        }
        c.spec.weight
    }

    /// World matrix of every node (index = `Node::typed_id`) at `time` of `anim`:
    /// ufbx's own evaluation (pivots, offsets, inherit modes), with the
    /// constraints solved on top and their effect carried down to the
    /// constrained nodes' descendants.
    pub fn frame(&self, anim: &ufbx::Anim, time: f64) -> Option<Vec<Mat4>> {
        let ev = ufbx::evaluate_scene(self.fbx, anim, time, ufbx::EvaluateOpts::default()).ok()?;
        let base: Vec<Mat4> = ev.nodes.iter().map(|n| matrix_to_mat4(&n.node_to_world)).collect();
        if self.constraints.is_empty() {
            return Some(base);
        }
        let n = base.len();
        // Each node's matrix relative to its parent as the evaluation composed it.
        let rel: Vec<Mat4> = (0..n)
            .map(|i| match self.parent[i] {
                Some(p) => base[p].inverse() * base[i],
                None => base[i],
            })
            .collect();
        let mut over: Vec<Option<Mat4>> = vec![None; n];
        let worlds = |over: &[Option<Mat4>]| {
            let mut world = vec![Mat4::IDENTITY; n];
            for i in 0..n {
                world[i] = match (over[i], self.parent[i]) {
                    (Some(o), _) => o,
                    (None, Some(p)) => world[p] * rel[i],
                    (None, None) => rel[i],
                };
            }
            world
        };
        // Two passes settle chains of constraints that read each other.
        for _ in 0..2 {
            for c in &self.constraints {
                if !c.spec.active || c.node >= n {
                    continue;
                }
                let mut unconstrained = over.clone();
                unconstrained[c.node] = None;
                let world = worlds(&unconstrained);
                let sources: Vec<SourceWorld> = c
                    .sources
                    .iter()
                    .zip(&c.spec.sources)
                    .filter_map(|(&i, s)| worlds(&over).get(i).map(|w| SourceWorld { world: *w, weight: s.weight, offset: s.offset }))
                    .collect();
                let mut params: SolveParams = c.spec.params();
                params.weight = self.weight_at(c, anim, time);
                let up = c.up.and_then(|u| world.get(u).copied());
                over[c.node] = Some(solve_world(&params, world[c.node], &sources, up));
            }
        }
        Some(worlds(&over))
    }
}

// ---------------------------------------------------------------- user props

/// A file's user (custom) property of a node: its name and initial value.
#[derive(Clone, Debug, PartialEq)]
pub struct UserProp {
    pub name: String,
    pub value: CustomValue,
}

/// The user-defined properties of a node with a type a custom property can hold.
pub(crate) fn user_props(node: &ufbx::Node) -> Vec<UserProp> {
    let mut out = Vec::new();
    for p in &node.element.props.props {
        if !p.flags.has_any(ufbx::PropFlags::USER_DEFINED) {
            continue;
        }
        let v = p.value_vec4;
        let value = match p.type_ {
            ufbx::PropType::Boolean => CustomValue::Bool(p.value_int != 0),
            ufbx::PropType::Integer => CustomValue::Int(p.value_int.clamp(i32::MIN as i64, i32::MAX as i64) as i32),
            ufbx::PropType::Number | ufbx::PropType::Distance => CustomValue::Float(v.x as f32),
            ufbx::PropType::Vector | ufbx::PropType::Color | ufbx::PropType::Translation | ufbx::PropType::Rotation | ufbx::PropType::Scaling => {
                CustomValue::Vec3([v.x as f32, v.y as f32, v.z as f32])
            }
            _ => continue,
        };
        out.push(UserProp { name: p.name.to_string(), value });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_offsets_normalise() {
        let t = ufbx::Transform {
            translation: ufbx::Vec3 { x: 1.0, y: 2.0, z: 3.0 },
            rotation: ufbx::Quat { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            scale: ufbx::Vec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let o = offset_of(&t);
        assert_eq!(o.r(), Quat::IDENTITY, "a zero quaternion becomes the identity");
        assert_eq!(o.t(), Vec3::new(1.0, 2.0, 3.0));
    }
}
