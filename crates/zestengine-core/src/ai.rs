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

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &BtValue)> {
        self.values.iter().map(|(key, value)| (key.as_str(), value))
    }

    pub fn keys_sorted(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.values.keys().cloned().collect();
        keys.sort();
        keys
    }

    pub fn rename(&mut self, from: &str, to: &str) -> bool {
        if from == to || to.is_empty() || self.values.contains_key(to) {
            return false;
        }
        match self.values.remove(from) {
            Some(value) => {
                self.values.insert(to.to_owned(), value);
                true
            }
            None => false,
        }
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
    Parallel {
        children: Vec<BtNode>,
        require_all: bool,
    },
    Repeat {
        child: Box<BtNode>,
        times: u32,
    },
    UntilFail {
        child: Box<BtNode>,
    },
    Succeed {
        child: Box<BtNode>,
    },
    Cooldown {
        child: Box<BtNode>,
        seconds: f32,
    },
    Timeout {
        child: Box<BtNode>,
        seconds: f32,
    },
    Wait {
        seconds: f32,
    },
    Compare {
        key: String,
        op: BtCompare,
        value: f32,
    },
    Near {
        key: String,
        distance: f32,
    },
    HasKey {
        key: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BtCompare {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    #[default]
    Greater,
}

impl BtCompare {
    pub const ALL: [BtCompare; 6] = [Self::Less, Self::LessEqual, Self::Equal, Self::NotEqual, Self::GreaterEqual, Self::Greater];

    pub fn label(self) -> &'static str {
        match self {
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Equal => "=",
            Self::NotEqual => "!=",
            Self::GreaterEqual => ">=",
            Self::Greater => ">",
        }
    }

    pub fn test(self, a: f32, b: f32) -> bool {
        match self {
            Self::Less => a < b,
            Self::LessEqual => a <= b,
            Self::Equal => (a - b).abs() <= 1e-4,
            Self::NotEqual => (a - b).abs() > 1e-4,
            Self::GreaterEqual => a >= b,
            Self::Greater => a > b,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BtSlot {
    timer: f32,
    count: u32,
    armed: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BtMemory {
    slots: Vec<BtSlot>,
}

impl BtMemory {
    fn slot(&mut self, index: usize) -> &mut BtSlot {
        if self.slots.len() <= index {
            self.slots.resize(index + 1, BtSlot::default());
        }
        &mut self.slots[index]
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

impl BtNode {
    pub const KINDS: [&'static str; 17] = [
        "Success", "Failure", "Selector", "Sequence", "Invert", "Condition", "Action", "Parallel", "Repeat", "Until Fail", "Succeed", "Cooldown", "Timeout", "Wait",
        "Compare", "Near", "Has Key",
    ];

    pub fn kind(&self) -> usize {
        match self {
            BtNode::Success => 0,
            BtNode::Failure => 1,
            BtNode::Selector { .. } => 2,
            BtNode::Sequence { .. } => 3,
            BtNode::Invert { .. } => 4,
            BtNode::Condition { .. } => 5,
            BtNode::Action { .. } => 6,
            BtNode::Parallel { .. } => 7,
            BtNode::Repeat { .. } => 8,
            BtNode::UntilFail { .. } => 9,
            BtNode::Succeed { .. } => 10,
            BtNode::Cooldown { .. } => 11,
            BtNode::Timeout { .. } => 12,
            BtNode::Wait { .. } => 13,
            BtNode::Compare { .. } => 14,
            BtNode::Near { .. } => 15,
            BtNode::HasKey { .. } => 16,
        }
    }

    pub fn template(kind: usize) -> BtNode {
        let leaf = || Box::new(BtNode::Action { action: BtAction::Idle, param: String::new() });
        match kind {
            0 => BtNode::Success,
            1 => BtNode::Failure,
            2 => BtNode::Selector { children: Vec::new() },
            3 => BtNode::Sequence { children: Vec::new() },
            4 => BtNode::Invert { child: leaf() },
            5 => BtNode::Condition { key: "has_target".into(), value: true, invert: false },
            7 => BtNode::Parallel { children: Vec::new(), require_all: true },
            8 => BtNode::Repeat { child: leaf(), times: 0 },
            9 => BtNode::UntilFail { child: leaf() },
            10 => BtNode::Succeed { child: leaf() },
            11 => BtNode::Cooldown { child: leaf(), seconds: 2.0 },
            12 => BtNode::Timeout { child: leaf(), seconds: 5.0 },
            13 => BtNode::Wait { seconds: 1.0 },
            14 => BtNode::Compare { key: "health".into(), op: BtCompare::Less, value: 25.0 },
            15 => BtNode::Near { key: "target_id".into(), distance: 2.0 },
            16 => BtNode::HasKey { key: "target_id".into() },
            _ => BtNode::Action { action: BtAction::Idle, param: String::new() },
        }
    }

    pub fn size(&self) -> usize {
        1 + match self {
            BtNode::Selector { children } | BtNode::Sequence { children } | BtNode::Parallel { children, .. } => children.iter().map(BtNode::size).sum(),
            BtNode::Invert { child }
            | BtNode::Repeat { child, .. }
            | BtNode::UntilFail { child }
            | BtNode::Succeed { child }
            | BtNode::Cooldown { child, .. }
            | BtNode::Timeout { child, .. } => child.size(),
            _ => 0,
        }
    }
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
    MoveTo,
    Flee,
    Wander,
    SetBool,
    SetFloat,
    ClearKey,
    Invoke,
}

impl BtAction {
    /// Every action leaf, in declaration order.
    pub const ALL: [BtAction; 13] = [
        BtAction::Idle,
        BtAction::Chase,
        BtAction::Stop,
        BtAction::PlayAnimation,
        BtAction::SetAnimatorBool,
        BtAction::Emit,
        BtAction::MoveTo,
        BtAction::Flee,
        BtAction::Wander,
        BtAction::SetBool,
        BtAction::SetFloat,
        BtAction::ClearKey,
        BtAction::Invoke,
    ];

    pub fn hint(self) -> &'static str {
        match self {
            BtAction::Idle | BtAction::Stop => "",
            BtAction::Chase => "object key",
            BtAction::PlayAnimation => "key holding a clip name",
            BtAction::SetAnimatorBool => "parameter=bool key",
            BtAction::Emit => "key holding the event name",
            BtAction::MoveTo => "position or object key",
            BtAction::Flee => "threat key, or key:distance",
            BtAction::Wander => "radius in metres",
            BtAction::SetBool => "key=true",
            BtAction::SetFloat => "key=1.5",
            BtAction::ClearKey => "key",
            BtAction::Invoke => "Component.Action",
        }
    }

    /// Inspector label of this action.
    pub fn label(self) -> &'static str {
        match self {
            BtAction::Idle => "Idle",
            BtAction::Chase => "Chase target",
            BtAction::Stop => "Stop agent",
            BtAction::PlayAnimation => "Play animation",
            BtAction::SetAnimatorBool => "Set animator bool",
            BtAction::Emit => "Emit event",
            BtAction::MoveTo => "Move to",
            BtAction::Flee => "Flee from",
            BtAction::Wander => "Wander",
            BtAction::SetBool => "Set bool",
            BtAction::SetFloat => "Set number",
            BtAction::ClearKey => "Clear key",
            BtAction::Invoke => "Invoke component action",
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
    tick_with(node, &mut BtMemory::default(), bb, owner, world, ctx)
}

pub fn tick_with(node: &BtNode, memory: &mut BtMemory, bb: &mut Blackboard, owner: Entity, world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) -> BtStatus {
    run(node, 0, memory, bb, owner, world, ctx)
}

fn position_of(key: &str, bb: &Blackboard, world: &SystemWorld<'_>) -> Option<Vec3> {
    match bb.value(key)? {
        BtValue::Object(id) => {
            let entity = world.entity(*id)?;
            Some(world.world_transform(entity)?.w_axis.truncate())
        }
        BtValue::Vec3(v) => Some(Vec3::from(*v)),
        _ => None,
    }
}

fn owner_position(owner: Entity, world: &SystemWorld<'_>) -> Option<Vec3> {
    Some(world.world_transform(owner)?.w_axis.truncate())
}

fn run(node: &BtNode, at: usize, memory: &mut BtMemory, bb: &mut Blackboard, owner: Entity, world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) -> BtStatus {
    match node {
        BtNode::Success => BtStatus::Success,
        BtNode::Failure => BtStatus::Failure,
        BtNode::Selector { children } => {
            let mut child_at = at + 1;
            for c in children {
                match run(c, child_at, memory, bb, owner, world, ctx) {
                    BtStatus::Success => return BtStatus::Success,
                    BtStatus::Running => return BtStatus::Running,
                    BtStatus::Failure => {}
                }
                child_at += c.size();
            }
            BtStatus::Failure
        }
        BtNode::Sequence { children } => {
            let mut child_at = at + 1;
            for c in children {
                match run(c, child_at, memory, bb, owner, world, ctx) {
                    BtStatus::Failure => return BtStatus::Failure,
                    BtStatus::Running => return BtStatus::Running,
                    BtStatus::Success => {}
                }
                child_at += c.size();
            }
            BtStatus::Success
        }
        BtNode::Parallel { children, require_all } => {
            let (mut succeeded, mut failed) = (0, 0);
            let mut child_at = at + 1;
            for c in children {
                match run(c, child_at, memory, bb, owner, world, ctx) {
                    BtStatus::Success => succeeded += 1,
                    BtStatus::Failure => failed += 1,
                    BtStatus::Running => {}
                }
                child_at += c.size();
            }
            if *require_all {
                if failed > 0 {
                    BtStatus::Failure
                } else if succeeded == children.len() {
                    BtStatus::Success
                } else {
                    BtStatus::Running
                }
            } else if succeeded > 0 {
                BtStatus::Success
            } else if failed == children.len() {
                BtStatus::Failure
            } else {
                BtStatus::Running
            }
        }
        BtNode::Invert { child } => match run(child, at + 1, memory, bb, owner, world, ctx) {
            BtStatus::Success => BtStatus::Failure,
            BtStatus::Failure => BtStatus::Success,
            BtStatus::Running => BtStatus::Running,
        },
        BtNode::Succeed { child } => match run(child, at + 1, memory, bb, owner, world, ctx) {
            BtStatus::Running => BtStatus::Running,
            _ => BtStatus::Success,
        },
        BtNode::UntilFail { child } => match run(child, at + 1, memory, bb, owner, world, ctx) {
            BtStatus::Failure => BtStatus::Success,
            _ => BtStatus::Running,
        },
        BtNode::Repeat { child, times } => match run(child, at + 1, memory, bb, owner, world, ctx) {
            BtStatus::Running => BtStatus::Running,
            _ => {
                let slot = memory.slot(at);
                slot.count += 1;
                if *times != 0 && slot.count >= *times {
                    slot.count = 0;
                    BtStatus::Success
                } else {
                    BtStatus::Running
                }
            }
        },
        BtNode::Cooldown { child, seconds } => {
            let slot = memory.slot(at);
            if slot.armed {
                if ctx.time < slot.timer {
                    return BtStatus::Failure;
                }
                slot.armed = false;
            }
            let status = run(child, at + 1, memory, bb, owner, world, ctx);
            if status != BtStatus::Running {
                let slot = memory.slot(at);
                slot.armed = *seconds > 0.0;
                slot.timer = ctx.time + *seconds;
            }
            status
        }
        BtNode::Timeout { child, seconds } => match run(child, at + 1, memory, bb, owner, world, ctx) {
            BtStatus::Running => {
                let slot = memory.slot(at);
                slot.timer += ctx.dt;
                if slot.timer > *seconds {
                    slot.timer = 0.0;
                    BtStatus::Failure
                } else {
                    BtStatus::Running
                }
            }
            status => {
                memory.slot(at).timer = 0.0;
                status
            }
        },
        BtNode::Wait { seconds } => {
            let slot = memory.slot(at);
            slot.timer += ctx.dt;
            if slot.timer >= *seconds {
                slot.timer = 0.0;
                BtStatus::Success
            } else {
                BtStatus::Running
            }
        }
        BtNode::Condition { key, value, invert } => {
            let met = bb.get_bool(key) == *value;
            if met != *invert {
                BtStatus::Success
            } else {
                BtStatus::Failure
            }
        }
        BtNode::Compare { key, op, value } => {
            if op.test(bb.get_float(key), *value) { BtStatus::Success } else { BtStatus::Failure }
        }
        BtNode::Near { key, distance } => match (owner_position(owner, world), position_of(key, bb, world)) {
            (Some(here), Some(there)) if here.distance(there) <= *distance => BtStatus::Success,
            _ => BtStatus::Failure,
        },
        BtNode::HasKey { key } => {
            if bb.value(key).is_some() { BtStatus::Success } else { BtStatus::Failure }
        }
        BtNode::Action { action, param } => run_action(*action, param, bb, owner, world, ctx),
    }
}

const ARRIVE_DISTANCE: f32 = 0.3;
const FLEE_DISTANCE: f32 = 8.0;
const WANDER_ORIGIN: &str = "wander_origin";
const WANDER_TARGET: &str = "wander_target";
const WANDER_COUNT: &str = "wander_count";

fn steer(owner: Entity, world: &SystemWorld<'_>, target: Vec3) -> Option<BtStatus> {
    let here = owner_position(owner, world)?;
    let mut agent = world.get_mut::<crate::NavAgent>(owner)?;
    if here.distance(target) <= agent.stopping_distance.max(ARRIVE_DISTANCE) {
        return Some(BtStatus::Success);
    }
    let heading_there = agent.state.destination.is_some_and(|d| d.distance(target) <= 0.05);
    if heading_there && agent.state.unreachable {
        return Some(BtStatus::Failure);
    }
    if !heading_there {
        agent.set_destination(target);
    }
    Some(BtStatus::Running)
}

fn wander_point(origin: Vec3, radius: f32, seed: u64) -> Vec3 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    let angle = (x & 0xFFFF) as f32 / 65_536.0 * std::f32::consts::TAU;
    let reach = (((x >> 16) & 0xFFFF) as f32 / 65_535.0).sqrt() * radius;
    origin + Vec3::new(angle.cos() * reach, 0.0, angle.sin() * reach)
}

fn run_action(action: BtAction, param: &str, bb: &mut Blackboard, owner: Entity, world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) -> BtStatus {
    match action {
        BtAction::Idle => BtStatus::Success,
        BtAction::MoveTo => position_of(param, bb, world).and_then(|target| steer(owner, world, target)).unwrap_or(BtStatus::Failure),
        BtAction::Flee => {
            let (key, distance) = match param.split_once(':') {
                Some((key, distance)) => (key, distance.trim().parse().unwrap_or(FLEE_DISTANCE)),
                None => (param, FLEE_DISTANCE),
            };
            let (Some(here), Some(threat)) = (owner_position(owner, world), position_of(key.trim(), bb, world)) else {
                return BtStatus::Failure;
            };
            let offset = Vec3::new(here.x - threat.x, 0.0, here.z - threat.z);
            if offset.length() >= distance {
                return BtStatus::Success;
            }
            let away = if offset.length() > 1e-3 { offset.normalize() } else { Vec3::X };
            let Some(mut agent) = world.get_mut::<crate::NavAgent>(owner) else {
                return BtStatus::Failure;
            };
            let target = threat + away * (distance + 1.0);
            if agent.state.destination.is_none_or(|d| d.distance(target) > 1.0) {
                agent.set_destination(target);
            }
            BtStatus::Running
        }
        BtAction::Wander => {
            let radius = param.trim().parse::<f32>().ok().filter(|r| *r > 0.0).unwrap_or(5.0);
            let Some(here) = owner_position(owner, world) else {
                return BtStatus::Failure;
            };
            if bb.value(WANDER_ORIGIN).is_none() {
                bb.set_vec3(WANDER_ORIGIN, here);
            }
            if bb.value(WANDER_TARGET).is_none() {
                let count = bb.get_float(WANDER_COUNT);
                bb.set_float(WANDER_COUNT, count + 1.0);
                let seed = world.object_id(owner).map_or(0, |id| id.raw() as u64) ^ ((count as u64) << 32) ^ (ctx.time * 1000.0) as u64;
                bb.set_vec3(WANDER_TARGET, wander_point(bb.get_vec3(WANDER_ORIGIN), radius, seed));
            }
            match steer(owner, world, bb.get_vec3(WANDER_TARGET)) {
                Some(BtStatus::Running) => BtStatus::Running,
                Some(BtStatus::Success) => {
                    bb.remove(WANDER_TARGET);
                    BtStatus::Success
                }
                Some(BtStatus::Failure) => {
                    bb.remove(WANDER_TARGET);
                    BtStatus::Running
                }
                None => BtStatus::Failure,
            }
        }
        BtAction::SetBool => {
            let (key, value) = param.split_once('=').unwrap_or((param, "true"));
            if key.trim().is_empty() {
                return BtStatus::Failure;
            }
            bb.set_bool(key.trim(), !matches!(value.trim(), "false" | "0" | "off"));
            BtStatus::Success
        }
        BtAction::SetFloat => match param.split_once('=').and_then(|(key, value)| Some((key.trim(), value.trim().parse::<f32>().ok()?))) {
            Some((key, value)) if !key.is_empty() => {
                bb.set_float(key, value);
                BtStatus::Success
            }
            _ => BtStatus::Failure,
        },
        BtAction::ClearKey => {
            bb.remove(param.trim());
            BtStatus::Success
        }
        BtAction::Invoke => {
            let (Some((component, action)), Some(id)) = (param.rsplit_once('.'), world.object_id(owner)) else {
                return BtStatus::Failure;
            };
            if action.trim().is_empty() {
                return BtStatus::Failure;
            }
            let binding = crate::EventBinding { target: ObjectRef::new(id), component: component.trim().to_owned(), action: action.trim().to_owned(), ..Default::default() };
            let source = crate::FireSource { source: id, instigator: None, payload: &[] };
            crate::queue_actions(std::slice::from_ref(&binding), &source, ctx.commands);
            BtStatus::Success
        }
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
        let BehaviourTreeAgent { root, memory, blackboard, defaults, seeded, last_status, .. } = &mut *agent;
        if !*seeded {
            *seeded = true;
            for (key, value) in defaults.iter() {
                if blackboard.value(key).is_none() {
                    blackboard.set(key, value.clone());
                }
            }
        }
        *last_status = Some(tick_with(root, memory, blackboard, owner.0, world, ctx));
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
    #[serde(skip_serializing_if = "Blackboard::is_empty")]
    pub defaults: Blackboard,
    #[serde(skip)]
    pub memory: BtMemory,
    #[serde(skip)]
    pub seeded: bool,
}

impl Default for BehaviourTreeAgent {
    fn default() -> Self {
        Self {
            root: BtNode::Sequence { children: Vec::new() },
            asset: String::new(),
            blackboard: Blackboard::default(),
            last_status: None,
            defaults: Blackboard::default(),
            memory: BtMemory::default(),
            seeded: false,
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

    fn harness(dt: f32, run: impl FnOnce(&SystemWorld<'_>, &mut SystemContext<'_>, Entity, ObjectId)) -> Vec<ComponentCommand> {
        use crate::{InputState, NavAgent, ecs::SceneWorld};
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Agent");
        let other = scene.spawn("Other");
        scene.get_mut::<crate::Transform>(other).unwrap().translation = Vec3::new(10.0, 0.0, 0.0);
        scene.insert(owner, NavAgent::default()).unwrap();
        scene.propagate().unwrap();
        let other_id = scene.object_id(other).unwrap();
        let input = InputState::default();
        let mut commands = Vec::new();
        scene.with_world(|world| {
            let mut ctx = SystemContext::new(&input, &mut commands);
            ctx.dt = dt;
            ctx.time = 3.0;
            run(world, &mut ctx, owner, other_id);
        });
        commands
    }

    fn leaf(action: BtAction, param: &str) -> BtNode {
        BtNode::Action { action, param: param.into() }
    }

    #[test]
    fn timed_nodes_keep_their_own_state_between_ticks() {
        harness(0.5, |world, ctx, owner, _| {
            let tree = BtNode::Sequence {
                children: vec![BtNode::Wait { seconds: 1.0 }, BtNode::Cooldown { child: Box::new(leaf(BtAction::SetBool, "fired")), seconds: 1.2 }],
            };
            let (mut memory, mut bb) = (BtMemory::default(), Blackboard::default());
            let mut step = |bb: &mut Blackboard, memory: &mut BtMemory| {
                ctx.time += ctx.dt;
                tick_with(&tree, memory, bb, owner, world, ctx)
            };
            assert_eq!(step(&mut bb, &mut memory), BtStatus::Running, "0.5 s into the wait");
            assert!(!bb.get_bool("fired"));
            assert_eq!(step(&mut bb, &mut memory), BtStatus::Success, "the wait ends and the action runs");
            assert!(bb.get_bool("fired"));
            bb.set_bool("fired", false);
            assert_eq!(step(&mut bb, &mut memory), BtStatus::Running);
            assert_eq!(step(&mut bb, &mut memory), BtStatus::Failure, "the wait ended again, 1 s into a 1.2 s cooldown");
            assert!(!bb.get_bool("fired"));
            assert_eq!(step(&mut bb, &mut memory), BtStatus::Running);
            assert_eq!(step(&mut bb, &mut memory), BtStatus::Success, "the cooldown ran out while the wait was in the way");
            assert!(bb.get_bool("fired"));
        });
    }

    #[test]
    fn repeat_timeout_and_result_decorators() {
        harness(0.4, |world, ctx, owner, _| {
            let mut bb = Blackboard::default();
            let thrice = BtNode::Repeat { child: Box::new(BtNode::Failure), times: 3 };
            let mut memory = BtMemory::default();
            let results: Vec<_> = (0..4).map(|_| tick_with(&thrice, &mut memory, &mut bb, owner, world, ctx)).collect();
            assert_eq!(results, [BtStatus::Running, BtStatus::Running, BtStatus::Success, BtStatus::Running]);
            let forever = BtNode::Repeat { child: Box::new(BtNode::Success), times: 0 };
            assert!((0..10).all(|_| tick_with(&forever, &mut memory, &mut bb, owner, world, ctx) == BtStatus::Running));

            let slow = BtNode::Timeout { child: Box::new(BtNode::Wait { seconds: 100.0 }), seconds: 1.0 };
            let mut memory = BtMemory::default();
            let results: Vec<_> = (0..4).map(|_| tick_with(&slow, &mut memory, &mut bb, owner, world, ctx)).collect();
            assert_eq!(results, [BtStatus::Running, BtStatus::Running, BtStatus::Failure, BtStatus::Running]);

            let mut memory = BtMemory::default();
            assert_eq!(tick_with(&BtNode::Succeed { child: Box::new(BtNode::Failure) }, &mut memory, &mut bb, owner, world, ctx), BtStatus::Success);
            assert_eq!(tick_with(&BtNode::UntilFail { child: Box::new(BtNode::Success) }, &mut memory, &mut bb, owner, world, ctx), BtStatus::Running);
            assert_eq!(tick_with(&BtNode::UntilFail { child: Box::new(BtNode::Failure) }, &mut memory, &mut bb, owner, world, ctx), BtStatus::Success);
        });
    }

    #[test]
    fn parallel_policies_and_blackboard_conditions() {
        harness(0.1, |world, ctx, owner, other| {
            let mut bb = Blackboard::default();
            let mut go = |node: &BtNode, bb: &mut Blackboard| tick(node, bb, owner, world, ctx);
            let all = |children: Vec<BtNode>| BtNode::Parallel { children, require_all: true };
            let any = |children: Vec<BtNode>| BtNode::Parallel { children, require_all: false };
            let waiting = || BtNode::Wait { seconds: 9.0 };
            assert_eq!(go(&all(vec![BtNode::Success, waiting()]), &mut bb), BtStatus::Running);
            assert_eq!(go(&all(vec![BtNode::Success, BtNode::Success]), &mut bb), BtStatus::Success);
            assert_eq!(go(&all(vec![waiting(), BtNode::Failure]), &mut bb), BtStatus::Failure);
            assert_eq!(go(&any(vec![waiting(), BtNode::Success]), &mut bb), BtStatus::Success);
            assert_eq!(go(&any(vec![BtNode::Failure, waiting()]), &mut bb), BtStatus::Running);
            assert_eq!(go(&any(vec![BtNode::Failure, BtNode::Failure]), &mut bb), BtStatus::Failure);
            let both = all(vec![leaf(BtAction::SetBool, "a"), leaf(BtAction::SetBool, "b=false"), leaf(BtAction::SetFloat, "health=12.5")]);
            assert_eq!(go(&both, &mut bb), BtStatus::Success);
            assert!(bb.get_bool("a") && !bb.get_bool("b"), "every child of a parallel runs");

            let compare = |op, value| BtNode::Compare { key: "health".into(), op, value };
            assert_eq!(go(&compare(BtCompare::Less, 25.0), &mut bb), BtStatus::Success);
            assert_eq!(go(&compare(BtCompare::GreaterEqual, 25.0), &mut bb), BtStatus::Failure);
            assert_eq!(go(&compare(BtCompare::Equal, 12.5), &mut bb), BtStatus::Success);
            assert_eq!(go(&leaf(BtAction::SetFloat, "health=lots"), &mut bb), BtStatus::Failure);
            assert_eq!(go(&BtNode::HasKey { key: "health".into() }, &mut bb), BtStatus::Success);
            assert_eq!(go(&leaf(BtAction::ClearKey, "health"), &mut bb), BtStatus::Success);
            assert_eq!(go(&BtNode::HasKey { key: "health".into() }, &mut bb), BtStatus::Failure);

            bb.set_object("friend", other);
            bb.set_vec3("spot", Vec3::new(0.5, 0.0, 0.0));
            assert_eq!(go(&BtNode::Near { key: "friend".into(), distance: 5.0 }, &mut bb), BtStatus::Failure, "the other object is 10 m away");
            assert_eq!(go(&BtNode::Near { key: "friend".into(), distance: 11.0 }, &mut bb), BtStatus::Success);
            assert_eq!(go(&BtNode::Near { key: "spot".into(), distance: 1.0 }, &mut bb), BtStatus::Success);
            assert_eq!(go(&BtNode::Near { key: "nothing".into(), distance: 1.0 }, &mut bb), BtStatus::Failure);
        });
    }

    #[test]
    fn movement_actions_steer_the_nav_agent() {
        harness(0.1, |world, ctx, owner, other| {
            use crate::NavAgent;
            let mut bb = Blackboard::default();
            bb.set_object("friend", other);
            bb.set_vec3("here", Vec3::new(0.1, 0.0, 0.0));
            let mut go = |action, param: &str, bb: &mut Blackboard| tick(&leaf(action, param), bb, owner, world, ctx);
            assert_eq!(go(BtAction::MoveTo, "friend", &mut bb), BtStatus::Running);
            assert_eq!(world.get::<NavAgent>(owner).unwrap().state.destination, Some(Vec3::new(10.0, 0.0, 0.0)));
            world.get_mut::<NavAgent>(owner).unwrap().state.replan = false;
            assert_eq!(go(BtAction::MoveTo, "friend", &mut bb), BtStatus::Running);
            assert!(!world.get::<NavAgent>(owner).unwrap().state.replan, "an unchanged target does not ask for a new path");
            world.get_mut::<NavAgent>(owner).unwrap().state.unreachable = true;
            assert_eq!(go(BtAction::MoveTo, "friend", &mut bb), BtStatus::Failure, "no path to it");
            assert_eq!(go(BtAction::MoveTo, "here", &mut bb), BtStatus::Success, "already there");
            assert_eq!(go(BtAction::MoveTo, "missing", &mut bb), BtStatus::Failure);

            bb.set_vec3("threat", Vec3::new(2.0, 0.0, 0.0));
            assert_eq!(go(BtAction::Flee, "threat", &mut bb), BtStatus::Running);
            let away = world.get::<NavAgent>(owner).unwrap().state.destination.unwrap();
            assert!(away.x < -6.0, "flees to the far side of the threat: {away}");
            assert_eq!(go(BtAction::Flee, "threat:1.5", &mut bb), BtStatus::Success, "2 m is already outside a 1.5 m flee distance");

            assert_eq!(go(BtAction::Wander, "4", &mut bb), BtStatus::Running);
            let first = bb.get_vec3("wander_target");
            assert!(first.length() <= 4.0 + 1e-3 && first.y == 0.0, "inside the radius around where it started: {first}");
            assert_eq!(world.get::<NavAgent>(owner).unwrap().state.destination, Some(first));
            assert_eq!(go(BtAction::Wander, "4", &mut bb), BtStatus::Running);
            assert_eq!(bb.get_vec3("wander_target"), first, "the goal holds until it is reached");
            bb.set_vec3("wander_target", Vec3::new(0.05, 0.0, 0.0));
            assert_eq!(go(BtAction::Wander, "4", &mut bb), BtStatus::Success, "arrived");
            assert_eq!(go(BtAction::Wander, "4", &mut bb), BtStatus::Running);
            assert_ne!(bb.get_vec3("wander_target"), first, "a new goal after each arrival");
        });
    }

    #[test]
    fn invoke_queues_a_component_action_on_the_agent() {
        let commands = harness(0.1, |world, ctx, owner, _| {
            let mut bb = Blackboard::default();
            assert_eq!(tick(&leaf(BtAction::Invoke, "Sequence Player.Play"), &mut bb, owner, world, ctx), BtStatus::Success);
            assert_eq!(tick(&leaf(BtAction::Invoke, "NoDot"), &mut bb, owner, world, ctx), BtStatus::Failure);
        });
        assert_eq!(commands.len(), 1);
        let ComponentCommand::FireActions { bindings, .. } = &commands[0] else { panic!("an action call is queued") };
        assert_eq!((bindings[0].component.as_str(), bindings[0].action.as_str()), ("Sequence Player", "Play"));
    }

    #[test]
    fn authored_defaults_seed_the_blackboard_once_and_every_kind_has_a_template() {
        use crate::{InputState, ecs::SceneWorld};
        let mut defaults = Blackboard::default();
        defaults.set_string("clip", "Patrol");
        defaults.set_float("health", 100.0);
        let agent = BehaviourTreeAgent { root: BtNode::Compare { key: "health".into(), op: BtCompare::Greater, value: 50.0 }, defaults, ..Default::default() };
        let text = crate::Component::save(&agent);
        assert!(text.contains("defaults") && !text.contains("memory"));
        let restored: BehaviourTreeAgent = ron::from_str(&text).unwrap();
        assert_eq!(restored.defaults.get_string("clip"), Some("Patrol"));
        assert!(!crate::Component::save(&BehaviourTreeAgent::default()).contains("defaults"));

        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Agent");
        scene.insert(owner, restored).unwrap();
        scene.propagate().unwrap();
        let input = InputState::default();
        let mut commands = Vec::new();
        for round in 0..2 {
            let mut ctx = SystemContext::new(&input, &mut commands);
            ctx.dt = 0.1;
            crate::ComponentRegistry::with_builtins().run_systems(Stage::Update, &mut scene, &mut ctx);
            let mut agent = scene.get_mut::<BehaviourTreeAgent>(owner).unwrap();
            if round == 0 {
                assert_eq!(agent.last_status, Some(BtStatus::Success));
                agent.blackboard.set_float("health", 10.0);
            } else {
                assert_eq!(agent.last_status, Some(BtStatus::Failure), "the defaults are not applied again over runtime values");
            }
        }

        for kind in 0..BtNode::KINDS.len() {
            let node = BtNode::template(kind);
            assert_eq!(node.kind(), kind, "{}", BtNode::KINDS[kind]);
            let back: BtNode = ron::from_str(&ron::to_string(&node).unwrap()).unwrap();
            assert_eq!(back, node);
        }
        let tree = BtNode::Sequence { children: vec![BtNode::Invert { child: Box::new(BtNode::Wait { seconds: 1.0 }) }, BtNode::Parallel { children: vec![BtNode::Success, BtNode::Failure], require_all: true }] };
        assert_eq!(tree.size(), 6);
        assert!(BtAction::ALL.iter().all(|a| !a.label().is_empty()));
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
