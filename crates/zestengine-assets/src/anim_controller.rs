//! Animator Controller asset (`.animctrl`): typed parameters plus layers, each
//! a state machine of motion states joined by conditional transitions. Modelled
//! on Unity's Animator Controller (states, Any State, exit time, triggers,
//! layers with masks and additive blending), with Unreal-style state enter/exit
//! notifies so other systems can react. Evaluated by the engine's anim graph.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use zestengine_core::BlendTree;

pub const ANIM_CONTROLLER_EXTENSION: &str = "animctrl";

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnimParamKind {
    #[default]
    Float,
    Int,
    Bool,
    /// A bool that resets itself when a transition consumes it.
    Trigger,
}

impl AnimParamKind {
    pub const ALL: [AnimParamKind; 4] = [Self::Float, Self::Int, Self::Bool, Self::Trigger];
    pub fn label(self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Int => "Int",
            Self::Bool => "Bool",
            Self::Trigger => "Trigger",
        }
    }
}

/// A controller parameter. Every kind is stored as an f32 at runtime (ints
/// rounded, bools and triggers 0 or 1), so `default` covers all of them.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AnimParameter {
    pub name: String,
    pub kind: AnimParamKind,
    pub default: f32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LayerBlend {
    #[default]
    Override,
    /// Adds the layer pose's offset from the rest pose on top of the layers below.
    Additive,
}

/// What a state plays.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Motion {
    /// An imported clip name, or a project-relative `.anim` path.
    Clip(String),
    BlendTree(BlendTree),
}

impl Default for Motion {
    fn default() -> Self {
        Motion::Clip(String::new())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AnimState {
    pub name: String,
    pub motion: Motion,
    pub speed: f32,
    /// Float parameter multiplying `speed` (empty = none).
    pub speed_parameter: String,
    pub looping: bool,
    /// Graph-editor position (canvas units).
    pub position: [f32; 2],
    /// Event names fired when the state is entered / left (delivered like
    /// clip events: `on_animation_event` + `AnimationEvent`). Empty = none.
    pub on_enter: String,
    pub on_exit: String,
    /// Off = values this state does not animate hold their last-frame value
    /// (the previous pose joins the mix at full weight) instead of blending
    /// toward the defaults.
    pub write_defaults: bool,
    /// Loop Pose: spread the difference between the clips' first and last
    /// frame over the cycle, so a loop closes without a pop.
    pub loop_pose: bool,
    /// Names of [`zestengine_core::StateBehaviour`]s (registered with
    /// `register_state_behaviour`) that receive this state's callbacks.
    pub behaviours: Vec<String>,
    /// Per-behaviour settings as RON text, by behaviour name (a behaviour
    /// with none gets an empty string). Kept as text so the file shape does
    /// not change as behaviour types are added.
    pub behaviour_settings: BTreeMap<String, String>,
    /// Float parameter that sets the state's normalized time each frame
    /// (empty = the state's clock runs normally). The motion plays a pose
    /// picked by the parameter instead of a time; speed is ignored.
    pub time_parameter: String,
    /// Set = this is a sub-state machine node: it plays nothing itself, and
    /// `motion`, `speed`, `looping` and the events are ignored. See
    /// [`StateMachine::flatten`] for how it runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_machine: Option<Box<StateMachine>>,
}

impl AnimState {
    pub fn is_sub_machine(&self) -> bool {
        self.sub_machine.is_some()
    }
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            name: "State".into(),
            motion: Motion::default(),
            speed: 1.0,
            speed_parameter: String::new(),
            looping: true,
            position: [0.0, 0.0],
            on_enter: String::new(),
            on_exit: String::new(),
            write_defaults: true,
            loop_pose: false,
            behaviours: Vec::new(),
            behaviour_settings: BTreeMap::new(),
            time_parameter: String::new(),
            sub_machine: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CondOp {
    /// Bool / trigger is set.
    #[default]
    If,
    IfNot,
    Greater,
    Less,
    Equals,
    NotEquals,
}

/// Which transition lists may cut in while a cross-fade is running. Any State
/// is checked with every choice except `None`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterruptionSource {
    /// Nothing interrupts (not even Any State).
    #[default]
    None,
    /// The state being left.
    Source,
    /// The state being entered.
    Destination,
    SourceThenDestination,
    DestinationThenSource,
}

impl InterruptionSource {
    pub const ALL: [InterruptionSource; 5] =
        [Self::None, Self::Source, Self::Destination, Self::SourceThenDestination, Self::DestinationThenSource];
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Source => "Current State",
            Self::Destination => "Next State",
            Self::SourceThenDestination => "Current State Then Next State",
            Self::DestinationThenSource => "Next State Then Current State",
        }
    }
}

impl CondOp {
    pub const ALL: [CondOp; 6] = [Self::If, Self::IfNot, Self::Greater, Self::Less, Self::Equals, Self::NotEquals];
    pub fn label(self) -> &'static str {
        match self {
            Self::If => "true",
            Self::IfNot => "false",
            Self::Greater => "Greater",
            Self::Less => "Less",
            Self::Equals => "Equals",
            Self::NotEquals => "Not Equal",
        }
    }
    pub fn uses_threshold(self) -> bool {
        !matches!(self, Self::If | Self::IfNot)
    }
    /// Unity differs for floats here: its Equals / Not Equal are always false
    /// on a Float parameter (Equals is Int-only). We keep an epsilon compare so
    /// authored `speed == 1` style conditions work on floats (decision D2).
    /// Int, Bool and Trigger parameters match Unity.
    pub fn test(self, value: f32, threshold: f32) -> bool {
        match self {
            Self::If => value > 0.5,
            Self::IfNot => value <= 0.5,
            Self::Greater => value > threshold,
            Self::Less => value < threshold,
            Self::Equals => (value - threshold).abs() < 1e-4,
            Self::NotEquals => (value - threshold).abs() >= 1e-4,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AnimCondition {
    pub parameter: String,
    pub op: CondOp,
    pub threshold: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AnimTransition {
    /// Source state name; None = Any State.
    pub from: Option<String>,
    pub to: String,
    /// All must hold.
    pub conditions: Vec<AnimCondition>,
    pub has_exit_time: bool,
    /// Normalized time of the source state at which the transition may fire.
    pub exit_time: f32,
    /// Cross-fade length in seconds.
    pub duration: f32,
    /// Normalized start time in the target state.
    pub offset: f32,
    /// Legacy flag from before `interruption`: when `interruption` is None,
    /// `true` means `Destination`. Use [`AnimTransition::interruption_source`].
    pub interruptible: bool,
    /// Which transition lists may cut in while this one is blending.
    pub interruption: InterruptionSource,
    /// With an interruption source, only transitions ahead of this one in the
    /// owning list (and the lists before it) may cut in.
    pub ordered_interruption: bool,
    /// Any State only: may re-enter the state already playing.
    pub can_transition_to_self: bool,
    /// Editor debugging: a muted transition never fires.
    pub muted: bool,
    /// Editor debugging: while any transition leaving the same source is
    /// soloed, only the soloed ones fire.
    pub solo: bool,
}

impl Default for AnimTransition {
    fn default() -> Self {
        Self {
            from: None,
            to: String::new(),
            conditions: Vec::new(),
            has_exit_time: false,
            exit_time: 1.0,
            duration: 0.25,
            offset: 0.0,
            interruptible: false,
            interruption: InterruptionSource::None,
            ordered_interruption: false,
            can_transition_to_self: false,
            muted: false,
            solo: false,
        }
    }
}

impl AnimTransition {
    /// The effective interruption source (folds in the legacy `interruptible`).
    pub fn interruption_source(&self) -> InterruptionSource {
        match self.interruption {
            InterruptionSource::None if self.interruptible => InterruptionSource::Destination,
            i => i,
        }
    }
}

/// One layer's state machine.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct StateMachine {
    pub states: Vec<AnimState>,
    /// Entered on start (empty = the first state).
    pub default_state: String,
    pub transitions: Vec<AnimTransition>,
    pub entry_position: [f32; 2],
    pub any_state_position: [f32; 2],
}

impl Default for StateMachine {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            default_state: String::new(),
            transitions: Vec::new(),
            entry_position: [-260.0, -60.0],
            any_state_position: [-260.0, 60.0],
        }
    }
}

impl StateMachine {
    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.states.iter().position(|s| s.name == name)
    }

    pub fn default_index(&self) -> Option<usize> {
        self.state_index(&self.default_state).or(if self.states.is_empty() { None } else { Some(0) })
    }

    /// Rename a state and every reference to it.
    pub fn rename_state(&mut self, old: &str, new: &str) {
        for s in &mut self.states {
            if s.name == old {
                s.name = new.to_owned();
            }
        }
        if self.default_state == old {
            self.default_state = new.to_owned();
        }
        for t in &mut self.transitions {
            if t.from.as_deref() == Some(old) {
                t.from = Some(new.to_owned());
            }
            if t.to == old {
                t.to = new.to_owned();
            }
        }
    }

    /// Remove a state and the transitions touching it.
    pub fn remove_state(&mut self, index: usize) {
        if index >= self.states.len() {
            return;
        }
        let name = self.states.remove(index).name;
        self.transitions.retain(|t| t.to != name && t.from.as_deref() != Some(name.as_str()));
        if self.default_state == name {
            self.default_state = self.states.first().map(|s| s.name.clone()).unwrap_or_default();
        }
    }

    /// A state name not used yet, based on `base`.
    pub fn unique_name(&self, base: &str) -> String {
        if self.state_index(base).is_none() {
            return base.to_owned();
        }
        (1..).map(|n| format!("{base} {n}")).find(|n| self.state_index(n).is_none()).unwrap()
    }

    /// Every state at every depth as `(path of sub-machine indices to its
    /// machine, state index, qualified name)`; sub-machine states are listed
    /// too, followed by what is inside them. Names read `Outer/Inner`.
    pub fn walk_states(&self) -> Vec<(Vec<usize>, usize, String)> {
        fn go(m: &StateMachine, path: &mut Vec<usize>, prefix: &str, out: &mut Vec<(Vec<usize>, usize, String)>) {
            for (i, s) in m.states.iter().enumerate() {
                let name = format!("{prefix}{}", s.name);
                out.push((path.clone(), i, name.clone()));
                if let Some(inner) = s.sub_machine.as_deref() {
                    path.push(i);
                    go(inner, path, &format!("{name}/"), out);
                    path.pop();
                }
            }
        }
        let mut out = Vec::new();
        go(self, &mut Vec::new(), "", &mut out);
        out
    }

    /// The machine reached by following sub-machine state indices from here.
    pub fn nested(&self, path: &[usize]) -> Option<&StateMachine> {
        match path.split_first() {
            None => Some(self),
            Some((&i, rest)) => self.states.get(i)?.sub_machine.as_deref()?.nested(rest),
        }
    }

    pub fn nested_mut(&mut self, path: &[usize]) -> Option<&mut StateMachine> {
        match path.split_first() {
            None => Some(self),
            Some((&i, rest)) => self.states.get_mut(i)?.sub_machine.as_deref_mut()?.nested_mut(rest),
        }
    }

    /// This machine and every nested one, depth first.
    pub fn for_each_machine_mut(&mut self, f: &mut impl FnMut(&mut StateMachine)) {
        f(self);
        for s in &mut self.states {
            if let Some(sub) = s.sub_machine.as_deref_mut() {
                sub.for_each_machine_mut(f);
            }
        }
    }

    /// Whether any state is a sub-state machine.
    pub fn has_sub_machines(&self) -> bool {
        self.states.iter().any(|s| s.sub_machine.is_some())
    }

    /// The equivalent machine without sub-state machines, which is what the
    /// runtime plays. Every leaf state is named by its path, `Outer/Inner`
    /// (recursively, `A/B/C`); top-level states keep their names, so a
    /// machine without sub-machines flattens to itself. Within each machine
    /// references stay local (a sub-machine's transitions name its own states;
    /// the parent names the sub-machine node), so renames never cross levels.
    ///
    /// - Entering a sub-machine node (default state or transition target)
    ///   enters its default state, recursively.
    /// - A transition whose source is a sub-machine node applies from every
    ///   leaf inside it.
    /// - A sub-machine's Any State transitions apply from every leaf inside
    ///   that sub-machine (not from outside it); the top level's stay global.
    /// - Check order for a leaf: top-level Any State, then outer levels'
    ///   transitions before inner ones, and within a sub-machine its Any
    ///   State before the leaf's own transitions.
    /// - Transitions whose target cannot be resolved (missing or empty
    ///   sub-machine) are dropped.
    pub fn flatten(&self) -> StateMachine {
        if self.has_mute_solo() {
            return self.without_disabled().flatten_unfiltered();
        }
        self.flatten_unfiltered()
    }

    /// Whether this machine or any nested one mutes or solos a transition.
    pub fn has_mute_solo(&self) -> bool {
        self.transitions.iter().any(|t| t.muted || t.solo)
            || self.states.iter().any(|s| s.sub_machine.as_deref().is_some_and(|m| m.has_mute_solo()))
    }

    /// Whether the runtime must play [`StateMachine::flatten`] instead of this
    /// machine as authored.
    pub fn needs_flatten(&self) -> bool {
        self.has_sub_machines() || self.has_mute_solo()
    }

    /// A copy without the transitions that mute / solo disable, recursively.
    pub fn without_disabled(&self) -> StateMachine {
        let mut out = self.clone();
        let on = enabled_transitions(&out.transitions);
        let mut it = on.into_iter();
        out.transitions.retain(|_| it.next().unwrap_or(true));
        for s in &mut out.states {
            if let Some(sub) = s.sub_machine.as_deref_mut() {
                *sub = sub.without_disabled();
            }
        }
        out
    }

    fn flatten_unfiltered(&self) -> StateMachine {
        if !self.has_sub_machines() {
            return self.clone();
        }
        let mut out = StateMachine {
            states: Vec::new(),
            default_state: self.entry_leaf("").unwrap_or_default(),
            transitions: Vec::new(),
            entry_position: self.entry_position,
            any_state_position: self.any_state_position,
        };
        self.flatten_into("", true, &mut out);
        out
    }

    /// Full name of the leaf entered when this machine (named by `prefix`) starts.
    fn entry_leaf(&self, prefix: &str) -> Option<String> {
        self.enter(prefix, &self.states.get(self.default_index()?)?.name)
    }

    /// Full name of the leaf entered by targeting local state `name`.
    fn enter(&self, prefix: &str, name: &str) -> Option<String> {
        let s = &self.states[self.state_index(name)?];
        let full = format!("{prefix}{}", s.name);
        match s.sub_machine.as_deref() {
            None => Some(full),
            Some(sub) => sub.entry_leaf(&format!("{full}/")),
        }
    }

    /// The motion of the state with the flattened name `name` (`Sub/State`).
    pub fn flattened_motion(&self, name: &str) -> Option<Motion> {
        fn find(m: &StateMachine, prefix: &str, name: &str) -> Option<Motion> {
            for s in &m.states {
                let full = format!("{prefix}{}", s.name);
                match s.sub_machine.as_deref() {
                    None if full == name => return Some(s.motion.clone()),
                    Some(sub) => {
                        if let Some(r) = find(sub, &format!("{full}/"), name) {
                            return Some(r);
                        }
                    }
                    None => {}
                }
            }
            None
        }
        find(self, "", name)
    }

    /// Full names (`Sub/State`) of every state that is not a sub-machine: the
    /// names a sync layer's motion overrides use.
    pub fn leaf_names(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.leaves("", &mut v);
        v
    }

    /// Full names of every leaf in this machine.
    fn leaves(&self, prefix: &str, out: &mut Vec<String>) {
        for s in &self.states {
            let full = format!("{prefix}{}", s.name);
            match s.sub_machine.as_deref() {
                None => out.push(full),
                Some(sub) => sub.leaves(&format!("{full}/"), out),
            }
        }
    }

    fn flatten_into(&self, prefix: &str, top: bool, out: &mut StateMachine) {
        for s in &self.states {
            if s.sub_machine.is_none() {
                out.states.push(AnimState { name: format!("{prefix}{}", s.name), ..s.clone() });
            }
        }
        // Sub-machine Any State first, then the rest in order.
        let any_first = self.transitions.iter().filter(|t| t.from.is_none() && !top);
        let rest = self.transitions.iter().filter(|t| t.from.is_some() || top);
        for t in any_first.chain(rest) {
            let Some(to) = self.enter(prefix, &t.to) else {
                continue;
            };
            let sources: Option<Vec<String>> = match &t.from {
                None if top => None,
                None => {
                    let mut v = Vec::new();
                    self.leaves(prefix, &mut v);
                    // An Any State transition only re-enters its target when allowed.
                    v.retain(|l| t.can_transition_to_self || *l != to);
                    Some(v)
                }
                Some(f) => {
                    let Some(s) = self.state_index(f).map(|i| &self.states[i]) else {
                        continue;
                    };
                    let full = format!("{prefix}{}", s.name);
                    let mut v = Vec::new();
                    match s.sub_machine.as_deref() {
                        None => v.push(full),
                        Some(sub) => sub.leaves(&format!("{full}/"), &mut v),
                    }
                    Some(v)
                }
            };
            match sources {
                None => out.transitions.push(AnimTransition { to, ..t.clone() }),
                Some(v) => out.transitions.extend(v.into_iter().map(|from| AnimTransition {
                    from: Some(from),
                    to: to.clone(),
                    ..t.clone()
                })),
            }
        }
        for s in &self.states {
            if let Some(sub) = s.sub_machine.as_deref() {
                sub.flatten_into(&format!("{prefix}{}/", s.name), false, out);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AnimLayer {
    pub name: String,
    /// 0..1; the base layer is always applied at full weight.
    pub weight: f32,
    pub blend: LayerBlend,
    /// Which part of the rig this layer affects (humanoid parts plus weighted
    /// transforms). The default mask is everything; old controllers stored a
    /// list of bone names, which loads as descendant-inclusive elements.
    pub mask: crate::avatar_mask::AvatarMask,
    /// A `.mask` asset (an `@<id>` reference or a project path) that replaces
    /// `mask` when the controller is loaded for use
    /// ([`AnimatorController::resolve_mask_assets`]); empty = the inline mask.
    /// Editing the asset updates every layer that references it.
    pub mask_asset: String,
    pub machine: StateMachine,
    /// Index of the layer whose state machine and timing this layer shares
    /// (a sync layer: `machine` is ignored, `motion_overrides` replace the
    /// motions). Must be a lower layer.
    pub sync_with: Option<usize>,
    /// The sync layer's motion lengths join the shared state timing.
    pub sync_timing: bool,
    /// Per-state motions of a sync layer, by (flattened) state name. States
    /// without one play the source layer's motion.
    pub motion_overrides: Vec<MotionOverride>,
    /// The layer takes part in the IK pass (its states' `on_state_ik` runs).
    pub ik_pass: bool,
}

impl AnimLayer {
    /// The override this sync layer has for the flattened state `state`.
    pub fn override_for(&self, state: &str) -> Option<&MotionOverride> {
        self.motion_overrides.iter().find(|o| o.state == state)
    }

    /// Set (`Some`) or remove (`None`) the override for `state`.
    pub fn set_override(&mut self, state: &str, motion: Option<Motion>) {
        match (self.motion_overrides.iter().position(|o| o.state == state), motion) {
            (Some(i), Some(m)) => self.motion_overrides[i].motion = m,
            (None, Some(m)) => self.motion_overrides.push(MotionOverride { state: state.into(), motion: m }),
            (Some(i), None) => {
                self.motion_overrides.remove(i);
            }
            (None, None) => {}
        }
    }

    /// Drop overrides whose state no longer exists in `source` (the layer this
    /// one syncs with). Returns how many went.
    pub fn prune_overrides(&mut self, source: &StateMachine) -> usize {
        let names = source.leaf_names();
        let before = self.motion_overrides.len();
        self.motion_overrides.retain(|o| names.contains(&o.state));
        before - self.motion_overrides.len()
    }
}

/// A sync layer's motion for one state.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct MotionOverride {
    pub state: String,
    pub motion: Motion,
}

impl Default for AnimLayer {
    fn default() -> Self {
        Self {
            name: "Base Layer".into(),
            weight: 1.0,
            blend: LayerBlend::Override,
            mask: Default::default(),
            mask_asset: String::new(),
            machine: StateMachine::default(),
            sync_with: None,
            sync_timing: false,
            motion_overrides: Vec::new(),
            ik_pass: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AnimatorController {
    pub parameters: Vec<AnimParameter>,
    pub layers: Vec<AnimLayer>,
}

impl Default for AnimatorController {
    fn default() -> Self {
        Self { parameters: Vec::new(), layers: vec![AnimLayer::default()] }
    }
}

impl AnimatorController {
    pub fn parameter(&self, name: &str) -> Option<&AnimParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Rename a parameter and every condition / tree / speed reference to it.
    pub fn rename_parameter(&mut self, old: &str, new: &str) {
        for p in &mut self.parameters {
            if p.name == old {
                p.name = new.to_owned();
            }
        }
        let fix = |s: &mut String| {
            if s == old {
                *s = new.to_owned();
            }
        };
        for layer in &mut self.layers {
            layer.machine.for_each_machine_mut(&mut |m| {
                for t in &mut m.transitions {
                    for c in &mut t.conditions {
                        fix(&mut c.parameter);
                    }
                }
                for s in &mut m.states {
                    fix(&mut s.speed_parameter);
                    fix(&mut s.time_parameter);
                    if let Some(text) = s.behaviour_settings.get_mut(zestengine_core::PARAMETER_DRIVER)
                        && let Ok(mut d) = zestengine_core::ParamDriverSettings::from_ron(text)
                    {
                        d.rename_parameter(old, new);
                        *text = d.to_ron();
                    }
                    if let Motion::BlendTree(tree) = &mut s.motion {
                        tree.rename_parameter(old, new);
                    }
                }
            });
        }
    }

    /// Every layer's machine flattened ([`StateMachine::flatten`]): what the
    /// runtime evaluates. Parameters are shared by all levels.
    ///
    /// Sync layers are resolved here too: a layer with `sync_with` gets its
    /// source layer's (flattened) machine, with its `motion_overrides` swapped
    /// in, so both share state indices, transitions and timing.
    pub fn flattened(&self) -> AnimatorController {
        let mut c = self.clone();
        for l in &mut c.layers {
            l.machine = l.machine.flatten();
        }
        for i in 0..c.layers.len() {
            let Some(src) = c.layers[i].sync_with.filter(|&s| s < i && c.layers[s].sync_with.is_none()) else {
                continue;
            };
            let mut machine = c.layers[src].machine.clone();
            for o in &c.layers[i].motion_overrides {
                if let Some(st) = machine.states.iter_mut().find(|s| s.name == o.state) {
                    st.motion = o.motion.clone();
                }
            }
            c.layers[i].machine = machine;
        }
        c
    }
}

/// Which transitions of one list may fire given mute and solo: a muted one
/// never; while any transition leaving a source is soloed, only the soloed
/// ones of that source.
pub fn enabled_transitions(ts: &[AnimTransition]) -> Vec<bool> {
    ts.iter()
        .map(|t| !t.muted && (t.solo || !ts.iter().any(|o| o.solo && o.from == t.from)))
        .collect()
}

impl AnimatorController {
    /// Replace the inline mask of every layer that references a `.mask` asset
    /// with the asset's content. Returns one message per asset that could not
    /// be read (the layer keeps its inline mask then).
    pub fn resolve_mask_assets(&mut self, project_root: &Path) -> Vec<String> {
        let mut problems = Vec::new();
        for l in &mut self.layers {
            if l.mask_asset.is_empty() {
                continue;
            }
            let path = zestengine_core::asset_ref::resolve_path(project_root, &l.mask_asset);
            match crate::avatar_mask::load_avatar_mask(&path) {
                Ok(m) => l.mask = m,
                Err(e) => problems.push(format!("layer '{}': mask {} could not be read ({e:#})", l.name, l.mask_asset)),
            }
        }
        problems
    }
}

pub fn load_anim_controller(path: impl AsRef<Path>) -> Result<AnimatorController> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing animator controller {}", path.display()))
}

pub fn save_anim_controller(path: impl AsRef<Path>, ctrl: &AnimatorController) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(ctrl, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::BlendMotion;

    fn mtr(from: &str, to: &str, muted: bool, solo: bool) -> AnimTransition {
        AnimTransition { from: Some(from.into()), to: to.into(), muted, solo, ..Default::default() }
    }

    #[test]
    fn mute_and_solo_pick_the_transitions_that_can_fire() {
        // Muted never fires; no solo: the rest fire.
        assert_eq!(enabled_transitions(&[mtr("A", "B", true, false), mtr("A", "C", false, false)]), vec![false, true]);
        // A solo silences the other transitions of the same source only.
        let ts = [mtr("A", "B", false, true), mtr("A", "C", false, false), mtr("X", "Y", false, false)];
        assert_eq!(enabled_transitions(&ts), vec![true, false, true]);
        // Muted wins over solo.
        assert_eq!(enabled_transitions(&[mtr("A", "B", true, true)]), vec![false]);
    }

    #[test]
    fn walk_states_lists_every_depth_with_paths() {
        let leaf = |n: &str| AnimState { name: n.into(), ..Default::default() };
        let inner = StateMachine { states: vec![leaf("In1"), leaf("In2")], ..Default::default() };
        let outer = AnimState { name: "Outer".into(), sub_machine: Some(Box::new(inner)), ..Default::default() };
        let m = StateMachine { states: vec![leaf("Top"), outer], ..Default::default() };
        let walked = m.walk_states();
        let names: Vec<&str> = walked.iter().map(|w| w.2.as_str()).collect();
        assert_eq!(names, vec!["Top", "Outer", "Outer/In1", "Outer/In2"]);
        assert_eq!(walked[3].0, vec![1]);
        assert_eq!(walked[3].1, 1);
        assert!(walked[0].0.is_empty());
        assert_eq!(m.nested(&walked[2].0).unwrap().states[walked[2].1].name, "In1");
    }

    #[test]
    fn sync_layer_overrides_are_set_replaced_removed_and_pruned() {
        let mut inner = StateMachine::default();
        inner.states.push(AnimState { name: "Walk".into(), ..Default::default() });
        let mut outer = StateMachine::default();
        outer.states.push(AnimState { name: "Idle".into(), ..Default::default() });
        outer.states.push(AnimState { name: "Move".into(), sub_machine: Some(Box::new(inner)), ..Default::default() });
        assert_eq!(outer.leaf_names(), vec!["Idle".to_owned(), "Move/Walk".to_owned()]);
        outer.states[1].sub_machine.as_mut().unwrap().states[0].motion = Motion::Clip("walk.anim".into());
        assert_eq!(outer.flattened_motion("Move/Walk"), Some(Motion::Clip("walk.anim".into())));
        assert_eq!(outer.flattened_motion("Move"), None);
        let mut layer = AnimLayer::default();
        layer.set_override("Idle", Some(Motion::Clip("a.anim".into())));
        layer.set_override("Move/Walk", Some(Motion::Clip("w.anim".into())));
        layer.set_override("Idle", Some(Motion::Clip("b.anim".into())));
        assert_eq!(layer.motion_overrides.len(), 2);
        assert_eq!(layer.override_for("Idle").unwrap().motion, Motion::Clip("b.anim".into()));
        layer.set_override("Gone", Some(Motion::Clip("x.anim".into())));
        assert_eq!(layer.prune_overrides(&outer), 1);
        layer.set_override("Idle", None);
        assert!(layer.override_for("Idle").is_none() && layer.override_for("Move/Walk").is_some());
    }

    #[test]
    fn a_layer_can_reference_a_mask_asset() {
        let dir = std::env::temp_dir().join(format!("zest-mask-ref-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let arms = crate::avatar_mask::AvatarMask { parts: vec![crate::avatar_mask::MaskPart::LeftArm], ..Default::default() };
        crate::avatar_mask::save_avatar_mask(&dir.join("arm.mask"), &arms).unwrap();
        let mut ctrl = AnimatorController { parameters: Vec::new(), layers: vec![AnimLayer { mask_asset: "arm.mask".into(), ..Default::default() }, AnimLayer::default()] };
        assert!(ctrl.resolve_mask_assets(&dir).is_empty());
        assert_eq!(ctrl.layers[0].mask, arms);
        assert!(ctrl.layers[1].mask.is_everything(), "layers without an asset keep their inline mask");
        // A missing asset is reported and leaves the inline mask alone.
        ctrl.layers[1].mask_asset = "gone.mask".into();
        let problems = ctrl.resolve_mask_assets(&dir);
        assert_eq!(problems.len(), 1);
        assert!(ctrl.layers[1].mask.is_everything());
        // Old files without the field load, and the field survives a round trip.
        let old: AnimLayer = ron::from_str("(name: \"X\")").unwrap();
        assert!(old.mask_asset.is_empty());
        let text = ron::to_string(&ctrl).unwrap();
        assert_eq!(ron::from_str::<AnimatorController>(&text).unwrap().layers[0].mask_asset, "arm.mask");
    }

    #[test]
    fn old_layer_mask_lists_still_load() {
        let old: AnimLayer = ron::from_str("(name: \"Arms\", mask: [\"Spine\", \"LeftHand\"])").unwrap();
        assert_eq!(old.mask.elements.len(), 2);
        assert!(old.mask.elements.iter().all(|e| e.descendants && e.weight == 1.0));
        assert!(!old.mask.is_everything());
        let none: AnimLayer = ron::from_str("(name: \"Base\")").unwrap();
        assert!(none.mask.is_everything());
        let again: AnimLayer = ron::from_str(&ron::to_string(&old).unwrap()).unwrap();
        assert_eq!(again, old);
    }

    #[test]
    fn flatten_drops_disabled_transitions_even_without_sub_machines() {
        let mut m = StateMachine::default();
        for n in ["A", "B", "C"] {
            m.states.push(AnimState { name: n.into(), ..Default::default() });
        }
        m.transitions = vec![mtr("A", "B", true, false), mtr("A", "C", false, false)];
        assert!(m.has_mute_solo() && m.needs_flatten());
        let f = m.flatten();
        assert_eq!(f.transitions.len(), 1);
        assert_eq!(f.transitions[0].to, "C");
        // The authored machine keeps both (and the flags survive a round trip).
        assert_eq!(m.transitions.len(), 2);
        let ron = ron::to_string(&m.transitions[0]).unwrap();
        assert!(ron::from_str::<AnimTransition>(&ron).unwrap().muted);
        // Old files without the fields load as enabled.
        assert!(!ron::from_str::<AnimTransition>("(to: \"B\")").unwrap().muted);
    }

    #[test]
    fn round_trips_through_ron() {
        let mut c = AnimatorController::default();
        c.parameters.push(AnimParameter { name: "Speed".into(), kind: AnimParamKind::Float, default: 0.0 });
        c.parameters.push(AnimParameter { name: "Jump".into(), kind: AnimParamKind::Trigger, default: 0.0 });
        let m = &mut c.layers[0].machine;
        m.states.push(AnimState { name: "Idle".into(), motion: Motion::Clip("Idle".into()), ..Default::default() });
        m.states.push(AnimState {
            name: "Move".into(),
            motion: Motion::BlendTree(BlendTree {
                name: "Move".into(),
                parameter: "Speed".into(),
                parameter_y: String::new(),
                kind: Default::default(),
                normalize_direct: false,
                motions: vec![BlendMotion { clip: "Walk".into(), position: [1.0, 0.0], ..Default::default() }],
            }),
            ..Default::default()
        });
        m.transitions.push(AnimTransition {
            from: Some("Idle".into()),
            to: "Move".into(),
            conditions: vec![AnimCondition { parameter: "Speed".into(), op: CondOp::Greater, threshold: 0.1 }],
            ..Default::default()
        });
        let text = ron::ser::to_string_pretty(&c, ron::ser::PrettyConfig::default()).unwrap();
        let back: AnimatorController = ron::from_str(&text).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn layer_feature_fields_default_for_old_files() {
        let st: AnimState = ron::from_str("(name: \"A\")").unwrap();
        assert!(st.write_defaults, "Write Defaults is on unless a file turns it off");
        assert!(!st.loop_pose);
        assert!(st.behaviours.is_empty());
        let layer: AnimLayer = ron::from_str("(name: \"L\")").unwrap();
        assert_eq!(layer.sync_with, None);
        assert!(!layer.sync_timing && !layer.ik_pass && layer.motion_overrides.is_empty());
        let new: AnimLayer = ron::from_str(
            "(sync_with: Some(0), sync_timing: true, ik_pass: true, motion_overrides: [(state: \"A\", motion: Clip(\"b\"))])",
        )
        .unwrap();
        assert_eq!(new.sync_with, Some(0));
        assert_eq!(new.motion_overrides[0].motion, Motion::Clip("b".into()));
    }

    #[test]
    fn rename_parameter_reaches_nested_trees_and_direct_parameters() {
        let mut c = AnimatorController::default();
        let inner = BlendTree { parameter: "Speed".into(), ..Default::default() };
        let tree = BlendTree {
            parameter: "Speed".into(),
            motions: vec![BlendMotion { direct_parameter: "Speed".into(), tree: Some(Box::new(inner)), ..Default::default() }],
            ..Default::default()
        };
        c.layers[0].machine.states.push(AnimState { name: "T".into(), motion: Motion::BlendTree(tree), ..Default::default() });
        c.rename_parameter("Speed", "Pace");
        let Motion::BlendTree(t) = &c.layers[0].machine.states[0].motion else { unreachable!() };
        assert_eq!(t.parameter, "Pace");
        assert_eq!(t.motions[0].direct_parameter, "Pace");
        assert_eq!(t.motions[0].tree.as_ref().unwrap().parameter, "Pace");
    }

    #[test]
    fn interruption_defaults_and_legacy_flag() {
        // A file from before `interruption` loads with its old `interruptible`.
        let old: AnimTransition = ron::from_str("(to: \"B\", interruptible: true)").unwrap();
        assert_eq!(old.interruption, InterruptionSource::None);
        assert!(!old.ordered_interruption);
        assert_eq!(old.interruption_source(), InterruptionSource::Destination);
        let none: AnimTransition = ron::from_str("(to: \"B\")").unwrap();
        assert_eq!(none.interruption_source(), InterruptionSource::None);
        let new: AnimTransition = ron::from_str("(to: \"B\", interruptible: true, interruption: Source)").unwrap();
        assert_eq!(new.interruption_source(), InterruptionSource::Source);
    }

    #[test]
    fn float_equals_uses_an_epsilon_unlike_unity() {
        assert!(CondOp::Equals.test(0.5, 0.5 + 1e-6));
        assert!(!CondOp::Equals.test(0.5, 0.6));
        assert!(CondOp::NotEquals.test(0.5, 0.6));
        assert!(!CondOp::NotEquals.test(0.5, 0.5));
    }

    #[test]
    fn rename_and_remove_patch_references() {
        let mut m = StateMachine::default();
        m.states.push(AnimState { name: "A".into(), ..Default::default() });
        m.states.push(AnimState { name: "B".into(), ..Default::default() });
        m.default_state = "A".into();
        m.transitions.push(AnimTransition { from: Some("A".into()), to: "B".into(), ..Default::default() });
        m.rename_state("A", "Idle");
        assert_eq!(m.default_state, "Idle");
        assert_eq!(m.transitions[0].from.as_deref(), Some("Idle"));
        m.remove_state(1);
        assert!(m.transitions.is_empty());
    }

    fn leaf(name: &str) -> AnimState {
        AnimState { name: name.into(), ..Default::default() }
    }

    fn sub(name: &str, machine: StateMachine) -> AnimState {
        AnimState { name: name.into(), sub_machine: Some(Box::new(machine)), ..Default::default() }
    }

    fn tr(from: Option<&str>, to: &str) -> AnimTransition {
        AnimTransition { from: from.map(Into::into), to: to.into(), has_exit_time: true, ..Default::default() }
    }

    fn names(m: &StateMachine) -> Vec<&str> {
        m.states.iter().map(|s| s.name.as_str()).collect()
    }

    fn pairs(m: &StateMachine) -> Vec<(Option<&str>, &str)> {
        m.transitions.iter().map(|t| (t.from.as_deref(), t.to.as_str())).collect()
    }

    /// Idle -> Locomotion{Walk (default), Run}; Locomotion -> Idle.
    fn locomotion() -> StateMachine {
        StateMachine {
            states: vec![
                leaf("Idle"),
                sub(
                    "Locomotion",
                    StateMachine {
                        states: vec![leaf("Walk"), leaf("Run")],
                        default_state: "Walk".into(),
                        transitions: vec![tr(Some("Walk"), "Run"), tr(None, "Walk")],
                        ..Default::default()
                    },
                ),
            ],
            default_state: "Idle".into(),
            transitions: vec![tr(Some("Idle"), "Locomotion"), tr(Some("Locomotion"), "Idle"), tr(None, "Idle")],
            ..Default::default()
        }
    }

    #[test]
    fn flatten_without_sub_machines_is_identity() {
        let mut m = StateMachine::default();
        m.states.push(leaf("A"));
        m.transitions.push(tr(None, "A"));
        assert_eq!(m.flatten(), m);
    }

    #[test]
    fn flatten_enters_default_and_exits_from_every_inner_state() {
        let f = locomotion().flatten();
        assert_eq!(names(&f), ["Idle", "Locomotion/Walk", "Locomotion/Run"]);
        assert_eq!(f.default_state, "Idle");
        assert_eq!(
            pairs(&f),
            [
                // Into the sub-machine = into its default state.
                (Some("Idle"), "Locomotion/Walk"),
                // Out of the sub-machine = from each inner state.
                (Some("Locomotion/Walk"), "Idle"),
                (Some("Locomotion/Run"), "Idle"),
                // Top-level Any State stays global.
                (None, "Idle"),
                // Inner Any State: from every inner state but its target, first.
                (Some("Locomotion/Run"), "Locomotion/Walk"),
                (Some("Locomotion/Walk"), "Locomotion/Run"),
            ]
        );
    }

    #[test]
    fn flatten_nested_two_levels() {
        let inner = StateMachine {
            states: vec![leaf("Light"), leaf("Heavy")],
            default_state: "Heavy".into(),
            ..Default::default()
        };
        let middle = StateMachine {
            states: vec![leaf("Draw"), sub("Swing", inner)],
            default_state: "Swing".into(),
            transitions: vec![tr(Some("Swing"), "Draw")],
            ..Default::default()
        };
        let top = StateMachine {
            states: vec![leaf("Idle"), sub("Attack", middle)],
            default_state: "Attack".into(),
            transitions: vec![tr(Some("Idle"), "Attack"), tr(Some("Attack"), "Idle")],
            ..Default::default()
        };
        let f = top.flatten();
        assert_eq!(names(&f), ["Idle", "Attack/Draw", "Attack/Swing/Light", "Attack/Swing/Heavy"]);
        // Default resolves through both levels.
        assert_eq!(f.default_state, "Attack/Swing/Heavy");
        assert_eq!(
            pairs(&f),
            [
                (Some("Idle"), "Attack/Swing/Heavy"),
                (Some("Attack/Draw"), "Idle"),
                (Some("Attack/Swing/Light"), "Idle"),
                (Some("Attack/Swing/Heavy"), "Idle"),
                (Some("Attack/Swing/Light"), "Attack/Draw"),
                (Some("Attack/Swing/Heavy"), "Attack/Draw"),
            ]
        );
    }

    #[test]
    fn flatten_follows_renames_and_drops_dangling_targets() {
        let mut m = locomotion();
        m.rename_state("Locomotion", "Move");
        m.nested_mut(&[1]).unwrap().rename_state("Walk", "Jog");
        let f = m.flatten();
        assert_eq!(names(&f), ["Idle", "Move/Jog", "Move/Run"]);
        assert!(pairs(&f).contains(&(Some("Idle"), "Move/Jog")));
        assert!(pairs(&f).contains(&(Some("Move/Run"), "Idle")));
        // Removing the inner default falls back to the first inner state.
        m.nested_mut(&[1]).unwrap().remove_state(0);
        assert!(pairs(&m.flatten()).contains(&(Some("Idle"), "Move/Run")));
        // An empty sub-machine cannot be entered: transitions into it vanish.
        m.nested_mut(&[1]).unwrap().remove_state(0);
        assert!(!pairs(&m.flatten()).iter().any(|(f, _)| *f == Some("Idle")));
    }

    #[test]
    fn sub_machines_round_trip_and_old_files_load() {
        let mut c = AnimatorController::default();
        c.layers[0].machine = locomotion();
        let text = ron::ser::to_string_pretty(&c, ron::ser::PrettyConfig::default()).unwrap();
        assert_eq!(ron::from_str::<AnimatorController>(&text).unwrap(), c);
        // A plain state writes no sub_machine field, so old files are unchanged.
        let plain = ron::to_string(&leaf("Idle")).unwrap();
        assert!(!plain.contains("sub_machine"));
        assert_eq!(ron::from_str::<AnimState>(&plain).unwrap(), leaf("Idle"));
    }

    #[test]
    fn rename_parameter_reaches_nested_machines() {
        let mut c = AnimatorController::default();
        let mut m = locomotion();
        m.nested_mut(&[1]).unwrap().transitions[0].conditions.push(AnimCondition { parameter: "Speed".into(), ..Default::default() });
        c.layers[0].machine = m;
        c.rename_parameter("Speed", "Velocity");
        assert_eq!(c.layers[0].machine.nested(&[1]).unwrap().transitions[0].conditions[0].parameter, "Velocity");
    }
}
