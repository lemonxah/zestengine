//! AI behaviour trees and perception.
//!
//! The behaviour tree is a data-driven node graph evaluated against a shared
//! [`Blackboard`]. Starter node set: `Sequence`, `Selector`, `Condition` and
//! `Action` leaves (plus `Invert` and the constant `Success` / `Failure`
//! leaves). Action leaves call the engine API through [`ComponentCtx`], so the
//! tree runs entirely inside the [`BehaviourTreeAgent`] component's `update`.
//!
//! Perception is a separate [`Perception`] component: a vision cone plus a
//! ray-cast through the engine's physics queries, written into the blackboard
//! for conditions to read. The `.behaviour` asset format (RON) lives in
//! `zestengine-assets` and wraps the same [`BtNode`] graph.

use std::collections::HashMap;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::{ComponentCtx, ObjectId, ObjectRef, TypedComponent};

/// A behaviour-tree node status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BtStatus {
    /// Still in progress; the tree keeps ticking it next frame.
    Running,
    /// Finished successfully.
    Success,
    /// Failed.
    Failure,
}

/// A single value on the shared blackboard.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BtValue {
    /// A boolean.
    Bool(bool),
    /// A float.
    Float(f32),
    /// A 32-bit integer.
    Int(i32),
    /// A string.
    String(String),
    /// A 3D vector (xyz).
    Vec3([f32; 3]),
    /// An object id.
    Object(ObjectId),
}

impl BtValue {
    /// The value as a Vec3 (zero when it is not one).
    pub fn as_vec3(&self) -> Vec3 {
        match self {
            BtValue::Vec3(v) => Vec3::from(*v),
            _ => Vec3::ZERO,
        }
    }

    /// Wrap a Vec3 as a blackboard value.
    pub fn from_vec3(v: Vec3) -> Self {
        BtValue::Vec3(v.to_array())
    }
}

/// Shared blackboard a tree reads each tick. Conditions read values; the agent
/// seeds it from perception; action leaves use it for their parameters.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Blackboard {
    values: HashMap<String, BtValue>,
}

impl Blackboard {
    /// Set (or replace) the value under `key`.
    pub fn set(&mut self, key: &str, value: BtValue) {
        self.values.insert(key.to_owned(), value);
    }

    /// Remove `key` from the blackboard.
    pub fn remove(&mut self, key: &str) {
        self.values.remove(key);
    }

    /// The value under `key`, if present.
    pub fn value(&self, key: &str) -> Option<&BtValue> {
        self.values.get(key)
    }

    /// Set a boolean under `key`.
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.set(key, BtValue::Bool(value));
    }

    /// The boolean under `key` (false when absent or not a bool).
    pub fn get_bool(&self, key: &str) -> bool {
        matches!(self.values.get(key), Some(BtValue::Bool(v)) if *v)
    }

    /// Set a float under `key`.
    pub fn set_float(&mut self, key: &str, value: f32) {
        self.set(key, BtValue::Float(value));
    }

    /// The float under `key` (0 when absent).
    pub fn get_float(&self, key: &str) -> f32 {
        match self.values.get(key) {
            Some(BtValue::Float(v)) => *v,
            Some(BtValue::Int(v)) => *v as f32,
            _ => 0.0,
        }
    }

    /// Set a string under `key`.
    pub fn set_string(&mut self, key: &str, value: &str) {
        self.set(key, BtValue::String(value.to_owned()));
    }

    /// The string under `key`, if present.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(BtValue::String(v)) => Some(v),
            _ => None,
        }
    }

    /// Set an object id under `key`.
    pub fn set_object(&mut self, key: &str, id: ObjectId) {
        self.set(key, BtValue::Object(id));
    }

    /// The object id under `key` (None when absent or nil).
    pub fn get_object(&self, key: &str) -> Option<ObjectId> {
        match self.values.get(key) {
            Some(BtValue::Object(id)) if !id.is_nil() => Some(*id),
            _ => None,
        }
    }

    /// Set a Vec3 under `key`.
    pub fn set_vec3(&mut self, key: &str, value: Vec3) {
        self.set(key, BtValue::from_vec3(value));
    }

    /// The Vec3 under `key` (zero when absent).
    pub fn get_vec3(&self, key: &str) -> Vec3 {
        self.values.get(key).map(BtValue::as_vec3).unwrap_or(Vec3::ZERO)
    }
}

/// A behaviour-tree node. This is both the runtime graph and the serialized
/// `.behaviour` asset node (see `zestengine_assets::BehaviourTree`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BtNode {
    /// Succeeds immediately.
    Success,
    /// Fails immediately.
    Failure,
    /// Runs children left to right until one succeeds or runs.
    Selector {
        /// Children tried in order.
        children: Vec<BtNode>,
    },
    /// Runs children left to right until one fails or runs.
    Sequence {
        /// Children tried in order.
        children: Vec<BtNode>,
    },
    /// Inverts its child's success / failure.
    Invert {
        /// The child whose result is inverted.
        child: Box<BtNode>,
    },
    /// Succeeds when blackboard `key` equals `value` (inverted when `invert`).
    Condition {
        /// Blackboard key read for the comparison.
        key: String,
        /// The value `key` must equal.
        value: bool,
        /// Invert the result (succeed when the values differ).
        invert: bool,
    },
    /// Executes a built-in action leaf through [`ComponentCtx`].
    Action {
        /// Which built-in action to run.
        action: BtAction,
        /// Parameter: a blackboard key (or a literal the editor shows in
        /// parentheses) the action reads.
        param: String,
    },
}

impl Default for BtNode {
    fn default() -> Self {
        BtNode::Sequence { children: Vec::new() }
    }
}

/// The built-in action leaves. `param` in [`BtNode::Action`] is a blackboard
/// key (or a literal string the editor shows in parentheses) the action reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BtAction {
    /// Wait / succeed without touching the world (the `param` is unused).
    #[default]
    Idle,
    /// `ctx.set_destination` toward the `param` object key.
    Chase,
    /// `ctx.stop_agent` on the owning object.
    Stop,
    /// `ctx.play_animation` with the `param` clip key.
    PlayAnimation,
    /// `ctx.set_animator_bool` with `param` = "key=name".
    SetAnimatorBool,
    /// `ctx.emit` a `BlackboardEvent { name }` (param is the event name key).
    Emit,
}

impl BtAction {
    /// Every action leaf, in declaration order.
    pub const ALL: [BtAction; 6] = [
        BtAction::Idle,
        BtAction::Chase,
        BtAction::Stop,
        BtAction::PlayAnimation,
        BtAction::SetAnimatorBool,
        BtAction::Emit,
    ];

    /// Inspector label of this action.
    pub fn label(self) -> &'static str {
        match self {
            BtAction::Idle => "Idle",
            BtAction::Chase => "Chase target",
            BtAction::Stop => "Stop agent",
            BtAction::PlayAnimation => "Play animation",
            BtAction::SetAnimatorBool => "Set animator bool",
            BtAction::Emit => "Emit event",
        }
    }
}

/// A gameplay event an [`BtAction::Emit`] leaf broadcasts.
#[derive(Clone, Debug)]
pub struct BlackboardEvent {
    /// Event name.
    pub name: String,
}

/// Evaluate `node` against the blackboard, calling engine APIs for action
/// leaves. `self_index` is the owning object index (used for self-references).
pub fn tick(node: &BtNode, bb: &mut Blackboard, ctx: &mut ComponentCtx) -> BtStatus {
    match node {
        BtNode::Success => BtStatus::Success,
        BtNode::Failure => BtStatus::Failure,
        BtNode::Selector { children } => {
            for c in children {
                match tick(c, bb, ctx) {
                    BtStatus::Success => return BtStatus::Success,
                    BtStatus::Running => return BtStatus::Running,
                    BtStatus::Failure => {}
                }
            }
            BtStatus::Failure
        }
        BtNode::Sequence { children } => {
            for c in children {
                match tick(c, bb, ctx) {
                    BtStatus::Failure => return BtStatus::Failure,
                    BtStatus::Running => return BtStatus::Running,
                    BtStatus::Success => {}
                }
            }
            BtStatus::Success
        }
        BtNode::Invert { child } => match tick(child, bb, ctx) {
            BtStatus::Success => BtStatus::Failure,
            BtStatus::Failure => BtStatus::Success,
            BtStatus::Running => BtStatus::Running,
        },
        BtNode::Condition { key, value, invert } => {
            let met = bb.get_bool(key) == *value;
            if met != *invert {
                BtStatus::Success
            } else {
                BtStatus::Failure
            }
        }
        BtNode::Action { action, param } => run_action(*action, param, bb, ctx),
    }
}

fn run_action(action: BtAction, param: &str, bb: &Blackboard, ctx: &mut ComponentCtx) -> BtStatus {
    match action {
        BtAction::Idle => BtStatus::Success,
        BtAction::Chase => match bb.get_object(param) {
            Some(target) => {
                ctx.set_destination(target, bb.get_vec3("target_position"));
                BtStatus::Running
            }
            None => BtStatus::Failure,
        },
        BtAction::Stop => {
            ctx.stop_agent(ctx.self_id());
            BtStatus::Success
        }
        BtAction::PlayAnimation => match bb.get_string(param) {
            Some(clip) => {
                ctx.play_animation(ctx.self_id(), clip, 0.2);
                BtStatus::Success
            }
            None => BtStatus::Failure,
        },
        BtAction::SetAnimatorBool => {
            // param = "<animator param name>=<blackboard bool key>"
            let Some((name, key)) = param.split_once('=') else {
                return BtStatus::Failure;
            };
            let value = bb.get_bool(key);
            ctx.set_animator_bool(ctx.self_id(), name, value);
            BtStatus::Success
        }
        BtAction::Emit => match bb.get_string(param) {
            Some(name) => {
                ctx.emit(BlackboardEvent { name: name.to_owned() });
                BtStatus::Success
            }
            None => BtStatus::Failure,
        },
    }
}

/// Vision and hearing senses for AI perception.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Perception {
    /// Vision distance (metres).
    pub vision_range: f32,
    /// Vision half-angle in degrees about the forward axis.
    pub vision_angle: f32,
    /// Hearing distance (full sphere, metres).
    pub hearing_range: f32,
    /// Physics layers the vision ray-cast may hit.
    pub layer_mask: u32,
}

impl Default for Perception {
    fn default() -> Self {
        Self {
            vision_range: 20.0,
            vision_angle: 60.0,
            hearing_range: 15.0,
            layer_mask: u32::MAX,
        }
    }
}

impl Perception {
    /// The vision half-angle in radians.
    pub fn half_angle(&self) -> f32 {
        self.vision_angle.clamp(0.0, 180.0).to_radians() * 0.5
    }

    /// Whether `target` (world offset from the observer) is visible: within
    /// range, inside the vision cone about `forward`, and `line_of_sight` is
    /// clear. Kept free of `ComponentCtx` so tests can drive it directly.
    pub fn visible(&self, forward: Vec3, target: Vec3, line_of_sight: bool) -> bool {
        let d = target.length();
        if d > self.vision_range || !line_of_sight {
            return false;
        }
        if d <= 1e-4 {
            return true;
        }
        target.normalize().dot(forward.normalize_or(Vec3::Z)) >= self.half_angle().cos()
    }

    /// Whether `target` (world offset from the observer) is within hearing range.
    pub fn audible(&self, target: Vec3) -> bool {
        target.length() <= self.hearing_range
    }
}

/// The last target the owning [`Perception`] component detected.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerceptionState {
    /// Object currently seen (vision) or None.
    pub last_seen: Option<ObjectId>,
    /// World position of the last detected target (vision or hearing).
    pub last_target: Vec3,
}

/// Component that detects other objects through a vision cone (ray-cast
/// against the physics world) and a hearing sphere. The owning
/// [`BehaviourTreeAgent`] reads the result into its blackboard as
/// `has_target`, `target_id` and `target_position`.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerceptionComponent {
    /// The senses (ranges, cone, layers).
    pub perception: Perception,
    /// Restrict detection to this object (None = any object).
    pub target: ObjectRef,
    /// Last detected target (runtime, not saved).
    #[serde(skip)]
    pub state: PerceptionState,
}

impl Default for PerceptionComponent {
    fn default() -> Self {
        Self {
            perception: Perception::default(),
            target: ObjectRef::NONE,
            state: PerceptionState::default(),
        }
    }
}

impl PerceptionComponent {
    /// Run the vision cone + ray-cast. Returns the detected object, if any.
    fn detect(&mut self, ctx: &ComponentCtx) -> Option<ObjectId> {
        let self_t = ctx.self_transform();
        let origin = self_t.translation;
        let forward = self_t.forward();

        if let Some(id) = self.target.id() {
            let index = ctx.index_of(id)?;
            let pos = ctx.object_position(index)?;
            let offset = pos - origin;
            let seen = self.perception.visible(forward, offset, self.line_of_sight(ctx, origin, id));
            if seen || self.perception.audible(offset) {
                return Some(id);
            }
            return None;
        }

        let mut best: Option<(f32, ObjectId, Vec3)> = None;
        for (index, id) in ctx.object_ids.iter().copied().enumerate() {
            if index == ctx.self_index || id.is_nil() {
                continue;
            }
            let Some(pos) = ctx.object_position(index) else { continue };
            let offset = pos - origin;
            let dist = offset.length();
            if dist > self.perception.vision_range && dist > self.perception.hearing_range {
                continue;
            }
            let seen = dist <= self.perception.vision_range
                && self.perception.visible(forward, offset, self.line_of_sight(ctx, origin, id));
            if !seen && dist > self.perception.hearing_range {
                continue;
            }
            if best.is_none_or(|(d, _, _)| dist < d) {
                best = Some((dist, id, pos));
            }
        }
        best.map(|(_, id, pos)| {
            self.state.last_target = pos;
            id
        })
    }

    fn line_of_sight(&self, ctx: &ComponentCtx, origin: Vec3, target: ObjectId) -> bool {
        let target_pos = match ctx.index_of(target).and_then(|i| ctx.object_position(i)) {
            Some(p) => p,
            None => return false,
        };
        let dir = target_pos - origin;
        let dist = dir.length();
        if dist <= 1e-4 {
            return true;
        }
        let filter = crate::RayFilter {
            layer_mask: self.perception.layer_mask,
            exclude: Some(ctx.self_id()),
            ..Default::default()
        };
        match ctx.raycast_filtered(origin + dir.normalize() * 0.05, dir.normalize(), dist, &filter) {
            None => true,
            Some(hit) => hit.object == target,
        }
    }
}

impl TypedComponent for PerceptionComponent {
    const NAME: &'static str = "Perception";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        self.state.last_seen = None;
        self.state.last_target = Vec3::ZERO;
        let Some(id) = self.detect(ctx) else { return };
        self.state.last_seen = Some(id);
        self.state.last_target = ctx.object_position(ctx.index_of(id).unwrap_or(usize::MAX)).unwrap_or(self.state.last_target);
    }
}

/// Runs a behaviour tree every frame. The tree is stored inline (the same
/// `BtNode` RON the `.behaviour` asset wraps); `asset` is the project-relative
/// `.behaviour` path the editor opens, and a future increment reloads it.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviourTreeAgent {
    /// The tree, run from its root each frame.
    pub root: BtNode,
    /// Project-relative `.behaviour` asset (authoring hook; runtime uses `root`).
    pub asset: String,
    /// Shared blackboard, seeded by perception and read by conditions
    /// (not saved).
    #[serde(skip)]
    pub blackboard: Blackboard,
    /// The root's status from the last tick (not saved).
    #[serde(skip)]
    pub last_status: Option<BtStatus>,
}

impl Default for BehaviourTreeAgent {
    fn default() -> Self {
        Self {
            root: BtNode::Sequence { children: Vec::new() },
            asset: String::new(),
            blackboard: Blackboard::default(),
            last_status: None,
        }
    }
}

impl BehaviourTreeAgent {
    /// Seed the blackboard from the owning object's [`PerceptionComponent`].
    pub fn sync_perception(&mut self, ctx: &ComponentCtx) {
        let Some(perception) = ctx.get_self::<PerceptionComponent>() else { return };
        let has = perception.state.last_seen.is_some();
        self.blackboard.set_bool("has_target", has);
        if let Some(id) = perception.state.last_seen {
            self.blackboard.set_object("target_id", id);
            self.blackboard.set_vec3("target_position", perception.state.last_target);
        } else {
            self.blackboard.remove("target_id");
            self.blackboard.remove("target_position");
        }
    }
}

impl TypedComponent for BehaviourTreeAgent {
    const NAME: &'static str = "Behaviour Tree";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        self.sync_perception(ctx);
        self.last_status = Some(tick(&self.root, &mut self.blackboard, ctx));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(bools: &[(&str, bool)]) -> Blackboard {
        let mut b = Blackboard::default();
        for (k, v) in bools {
            b.set_bool(k, *v);
        }
        b
    }

    #[test]
    fn blackboard_typed_roundtrip() {
        let mut b = Blackboard::default();
        b.set_bool("a", true);
        b.set_float("b", 1.5);
        b.set_string("c", "clip");
        assert!(b.get_bool("a"));
        assert_eq!(b.get_float("b"), 1.5);
        assert_eq!(b.get_string("c"), Some("clip"));
        assert_eq!(b.get_object("missing"), None);
    }

    #[test]
    fn selector_runs_until_success_and_sequence_fails_fast() {
        let _ = bb(&[]);
        // These are pure structural nodes, so a default blackboard is enough.
        let tree = BtNode::Selector {
            children: vec![
                BtNode::Failure,
                BtNode::Success,
                BtNode::Failure,
            ],
        };
        // tick needs a ComponentCtx, so assert on the shape through a small
        // context-free evaluator used only by tests.
        assert_eq!(struct_tick(&tree), BtStatus::Success);

        let seq = BtNode::Sequence {
            children: vec![
                BtNode::Success,
                BtNode::Failure,
                BtNode::Success,
            ],
        };
        assert_eq!(struct_tick(&seq), BtStatus::Failure);
    }

    /// Structural-only evaluator mirroring `tick` without `ComponentCtx`, so
    /// control-flow nodes can be tested without constructing a full frame.
    fn struct_tick(node: &BtNode) -> BtStatus {
        match node {
            BtNode::Success => BtStatus::Success,
            BtNode::Failure => BtStatus::Failure,
            BtNode::Selector { children } => {
                for c in children {
                    match struct_tick(c) {
                        BtStatus::Success => return BtStatus::Success,
                        BtStatus::Running => return BtStatus::Running,
                        BtStatus::Failure => {}
                    }
                }
                BtStatus::Failure
            }
            BtNode::Sequence { children } => {
                for c in children {
                    match struct_tick(c) {
                        BtStatus::Failure => return BtStatus::Failure,
                        BtStatus::Running => return BtStatus::Running,
                        BtStatus::Success => {}
                    }
                }
                BtStatus::Success
            }
            BtNode::Invert { child } => match struct_tick(child) {
                BtStatus::Success => BtStatus::Failure,
                BtStatus::Failure => BtStatus::Success,
                BtStatus::Running => BtStatus::Running,
            },
            BtNode::Condition { .. } | BtNode::Action { .. } => BtStatus::Failure,
        }
    }

    #[test]
    fn perception_vision_needs_range_cone_and_los() {
        let p = Perception::default();
        let fwd = Vec3::Z;
        assert!(p.visible(fwd, Vec3::new(0.0, 0.0, 5.0), true));
        assert!(!p.visible(fwd, Vec3::new(0.0, 0.0, -5.0), true));
        assert!(!p.visible(fwd, Vec3::new(0.0, 0.0, 100.0), true));
        assert!(!p.visible(fwd, Vec3::new(0.0, 0.0, 5.0), false));
        assert!(p.audible(Vec3::new(10.0, 0.0, 0.0)));
        assert!(!p.audible(Vec3::new(100.0, 0.0, 0.0)));
    }

    #[test]
    fn bt_asset_ron_roundtrip() {
        let tree = BtNode::Sequence {
            children: vec![
                BtNode::Condition {
                    key: "has_target".into(),
                    value: true,
                    invert: false,
                },
                BtNode::Action {
                    action: BtAction::Chase,
                    param: "target_id".into(),
                },
            ],
        };
        let ron = ron::to_string(&tree).unwrap();
        let back: BtNode = ron::from_str(&ron).unwrap();
        assert_eq!(back, tree);
    }
}
