//! AI behaviour trees and perception.
//!
//! The behaviour tree is a data-driven node graph evaluated against a shared
//! [`Blackboard`]. Starter node set: `Sequence`, `Selector`, `Condition` and
//! `Action` leaves (plus `Invert` and the constant `Success` / `Failure`
//! leaves). Native systems update perception before ticking behaviour trees;
//! action leaves read and write typed ECS values and defer emitted events.
//!
//! Perception is a separate [`Perception`] component: a vision cone plus a
//! ray-cast through the engine's physics queries, written into the blackboard
//! for conditions to read. The `.behaviour` asset format (RON) lives in
//! `zestengine-assets` and wraps the same [`BtNode`] graph.

use std::collections::HashMap;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::{AnyBox, ComponentCommand, ObjectId, ObjectRef, TypedComponent};
use crate::ecs::{Entity, Owner, Stage, SystemContext, SystemWorld};

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
    /// Executes a built-in action leaf against native ECS data.
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
    /// Set an animator bool with `param` = "name=blackboard_key".
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
/// leaves. `owner` is the owning runtime entity; persistent references use saved IDs.
pub fn tick(node: &BtNode, bb: &mut Blackboard, owner: Entity, world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) -> BtStatus {
    match node {
        BtNode::Success => BtStatus::Success,
        BtNode::Failure => BtStatus::Failure,
        BtNode::Selector { children } => {
            for c in children {
                match tick(c, bb, owner, world, ctx) {
                    BtStatus::Success => return BtStatus::Success,
                    BtStatus::Running => return BtStatus::Running,
                    BtStatus::Failure => {}
                }
            }
            BtStatus::Failure
        }
        BtNode::Sequence { children } => {
            for c in children {
                match tick(c, bb, owner, world, ctx) {
                    BtStatus::Failure => return BtStatus::Failure,
                    BtStatus::Running => return BtStatus::Running,
                    BtStatus::Success => {}
                }
            }
            BtStatus::Success
        }
        BtNode::Invert { child } => match tick(child, bb, owner, world, ctx) {
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
        BtNode::Action { action, param } => run_action(*action, param, bb, owner, world, ctx),
    }
}

fn run_action(action: BtAction, param: &str, bb: &Blackboard, owner: Entity, world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) -> BtStatus {
    match action {
        BtAction::Idle => BtStatus::Success,
        BtAction::Chase => match bb.get_object(param) {
            Some(target) => {
                if world.entity(target).is_some() && let Some(mut agent) = world.get_mut::<crate::NavAgent>(owner) {
                    agent.set_destination(bb.get_vec3("target_position"));
                }
                BtStatus::Running
            }
            None => BtStatus::Failure,
        },
        BtAction::Stop => {
            if let Some(mut agent) = world.get_mut::<crate::NavAgent>(owner) { agent.stop(); }
            BtStatus::Success
        }
        BtAction::PlayAnimation => match bb.get_string(param) {
            Some(clip) => {
                if let Some(mut animator) = world.get_mut::<crate::Animator>(owner) { animator.play(clip, 0.2); }
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
            if let Some(mut animator) = world.get_mut::<crate::Animator>(owner) { animator.set_bool(name, value); }
            BtStatus::Success
        }
        BtAction::Emit => match bb.get_string(param) {
            Some(name) => {
                ctx.commands.push(ComponentCommand::Emit(AnyBox(Box::new(BlackboardEvent { name: name.to_owned() }))));
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
    /// clear. Independent of the native system so tests can drive it directly.
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

impl TypedComponent for PerceptionComponent {
    const NAME: &'static str = "Perception";
}

struct Target {
    entity: Entity,
    id: ObjectId,
    position: Vec3,
}

impl PerceptionComponent {
    fn detect(&self, owner: Entity, origin: Vec3, forward: Vec3, targets: &[Target], physics: Option<&dyn crate::PhysicsQuery>, self_id: ObjectId) -> Option<(ObjectId, Vec3)> {
        let detectable = |target: &Target| {
            let offset = target.position - origin;
            let distance = offset.length();
            if distance > self.perception.vision_range && distance > self.perception.hearing_range { return false; }
            let visible = self.perception.visible(forward, offset, if distance <= 1e-4 { true } else {
                let filter = crate::RayFilter { layer_mask: self.perception.layer_mask, exclude: Some(self_id), ..Default::default() };
                physics.and_then(|physics| physics.raycast(origin + offset.normalize() * 0.05, offset.normalize(), distance, &filter))
                    .is_none_or(|hit| hit.object == target.id)
            });
            visible || self.perception.audible(offset)
        };
        if let Some(id) = self.target.id() {
            return targets.iter().find(|target| target.id == id && detectable(target)).map(|target| (target.id, target.position));
        }
        let mut best: Option<(f32, &Target)> = None;
        for target in targets {
            if target.entity == owner || !detectable(target) { continue; }
            let distance = target.position.distance(origin);
            if best.is_none_or(|(previous, _)| distance < previous) { best = Some((distance, target)); }
        }
        best.map(|(_, target)| (target.id, target.position))
    }
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<PerceptionComponent>();
    registry.register_ecs::<BehaviourTreeAgent>();
    registry.register_system_for::<PerceptionComponent>("zestengine::perception", Stage::Update, perception);
    registry.register_system_for::<BehaviourTreeAgent>("zestengine::behaviour_trees", Stage::Update, behaviour_trees);
}

fn perception(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut PerceptionComponent)>();
    let mut sensors: Vec<_> = query.iter().filter(|(_, (owner, _))| ctx.is_active(world, owner.0)).collect();
    if sensors.is_empty() { return; }
    sensors.sort_unstable_by_key(|(_, (owner, _))| ctx.order_key(world, owner.0));
    let mut targets: Vec<_> = world.query::<(&ObjectId, &crate::Transform)>().iter().filter_map(|(entity, (id, _))| {
        if id.is_nil() || !ctx.is_visible(world, entity) { return None; }
        Some(Target { entity, id: *id, position: world.world_transform(entity)?.w_axis.truncate() })
    }).collect();
    targets.sort_unstable_by_key(|target| ctx.order_key(world, target.entity));
    for (_, (owner, sensor)) in sensors {
        sensor.state = PerceptionState::default();
        let Some(matrix) = world.world_transform(owner.0) else { continue };
        let transform = crate::Transform::from_matrix(matrix);
        let self_id = world.object_id(owner.0).expect("component owner");
        if let Some((id, position)) = sensor.detect(owner.0, transform.translation, transform.forward(), &targets, ctx.services.physics, self_id) {
            sensor.state = PerceptionState { last_seen: Some(id), last_target: position };
        }
    }
}

fn behaviour_trees(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut BehaviourTreeAgent)>();
    let mut agents: Vec<_> = query.iter().filter(|(_, (owner, _))| ctx.is_active(world, owner.0)).collect();
    agents.sort_unstable_by_key(|(_, (owner, _))| ctx.order_key(world, owner.0));
    for (_, (owner, agent)) in agents {
        if let Some(perception) = world.get::<PerceptionComponent>(owner.0) { agent.sync_perception(perception.state); }
        agent.last_status = Some(tick(&agent.root, &mut agent.blackboard, owner.0, world, ctx));
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
    pub fn sync_perception(&mut self, perception: PerceptionState) {
        let has = perception.last_seen.is_some();
        self.blackboard.set_bool("has_target", has);
        if let Some(id) = perception.last_seen {
            self.blackboard.set_object("target_id", id);
            self.blackboard.set_vec3("target_position", perception.last_target);
        } else {
            self.blackboard.remove("target_id");
            self.blackboard.remove("target_position");
        }
    }
}

impl TypedComponent for BehaviourTreeAgent {
    const NAME: &'static str = "Behaviour Tree";
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
        // The structural cases use the same evaluator as native gameplay.
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

    fn struct_tick(node: &BtNode) -> BtStatus {
        let mut scene = crate::ecs::SceneWorld::new();
        let owner = scene.spawn("Agent");
        let input = crate::InputState::default();
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&input, &mut commands);
        scene.with_world(|world| tick(node, &mut Blackboard::default(), owner, world, &mut ctx))
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
    fn native_tree_actions_write_owner_data_and_emit_without_legacy_deferred_edits() {
        use crate::{Animator, InputState, NavAgent, ecs::SceneWorld};
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Agent");
        let target = scene.spawn("Target");
        let target_id = scene.object_id(target).unwrap();
        scene.insert(owner, NavAgent::default()).unwrap();
        scene.insert(owner, Animator::default()).unwrap();
        scene.insert(target, NavAgent::default()).unwrap();
        let mut bb = Blackboard::default();
        bb.set_object("target", target_id);
        bb.set_vec3("target_position", Vec3::new(1.0, 2.0, 3.0));
        bb.set_string("clip", "Run");
        bb.set_bool("moving", true);
        bb.set_string("message", "Spotted");
        let input = InputState::default();
        let mut commands = Vec::new();
        scene.with_world(|world| {
            let mut ctx = SystemContext::new(&input, &mut commands);
            let mut action = |action, param: &str| tick(&BtNode::Action { action, param: param.into() }, &mut bb, owner, world, &mut ctx);
            assert_eq!(action(BtAction::Chase, "target"), BtStatus::Running);
            assert_eq!(world.get::<NavAgent>(owner).unwrap().state.destination, Some(Vec3::new(1.0, 2.0, 3.0)));
            assert_eq!(world.get::<NavAgent>(target).unwrap().state.destination, None, "Chase steers the observer's agent");
            assert_eq!(action(BtAction::PlayAnimation, "clip"), BtStatus::Success);
            assert_eq!(world.get::<Animator>(owner).unwrap().state.current.as_deref(), Some("Run"));
            assert_eq!(action(BtAction::SetAnimatorBool, "moving=moving"), BtStatus::Success);
            assert!(world.get::<Animator>(owner).unwrap().get_bool("moving"));
            assert_eq!(action(BtAction::SetAnimatorBool, "invalid"), BtStatus::Failure);
            assert_eq!(action(BtAction::Emit, "message"), BtStatus::Success);
            assert_eq!(action(BtAction::Stop, ""), BtStatus::Success);
            assert_eq!(world.get::<NavAgent>(owner).unwrap().state.destination, None);
        });
        assert_eq!(commands.len(), 1);
        let ComponentCommand::Emit(event) = &commands[0] else { panic!("native tree must not emit erased Modify closures") };
        assert_eq!(event.0.downcast_ref::<BlackboardEvent>().unwrap().name, "Spotted");
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
