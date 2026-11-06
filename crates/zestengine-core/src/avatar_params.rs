//! Avatar parameters and expression-menu data.
//!
//! [`AvatarParameters`] declares the typed values an avatar exposes (Bool / Int
//! / Float, with a default, a saved flag and synced-or-local) and owns the
//! runtime [`AvatarParamStore`]. The store is the avatar's one parameter
//! namespace: engine-filled built-ins ([`BuiltinParam`]) first, then the
//! declared ones. It pushes only the values that changed since the last flush
//! into the Animator's parameter map, so a quiet avatar costs nothing per
//! frame, and reads back what state-behaviour drivers wrote.
//! [`AvatarMenuItem`] is the menu data (the hierarchy is the menu); the radial
//! menu UI and the network sync are built on top of it elsewhere.
//!
//! Sync budget: a Bool costs 1 bit, an Int or Float 8; only `Synced`
//! parameters count (decision D6: 256 bits per avatar by default, set per
//! project through `ContentLimits::max_param_bits` and per avatar through
//! `AvatarParameters::budget_bits`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Synced-bit budget when nothing says otherwise.
pub const DEFAULT_PARAM_BUDGET_BITS: u32 = 256;
/// Controls one menu page holds.
pub const MAX_MENU_CONTROLS: usize = 8;
/// Bits a Bool costs on the wire.
pub const BOOL_BITS: u32 = 1;
/// Bits an Int or Float costs on the wire.
pub const NUMBER_BITS: u32 = 8;

/// What a declared parameter can hold.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AvatarParamKind {
    /// On / off (stored as 0 / 1).
    Bool,
    /// Whole number.
    Int,
    /// Continuous value.
    #[default]
    Float,
}

impl AvatarParamKind {
    /// Every kind, in declaration order.
    pub const ALL: [AvatarParamKind; 3] = [Self::Bool, Self::Int, Self::Float];
    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::Int => "Int",
            Self::Float => "Float",
        }
    }
    /// Bits this kind occupies in the synced payload.
    pub fn bits(self) -> u32 {
        match self {
            Self::Bool => BOOL_BITS,
            Self::Int | Self::Float => NUMBER_BITS,
        }
    }
    /// Round / threshold `v` to what this kind can hold (no range clamp).
    pub fn coerce_local(self, v: f32) -> f32 {
        match self {
            Self::Bool => (v != 0.0) as u8 as f32,
            Self::Int => v.round(),
            Self::Float => v,
        }
    }
}

/// Whether a parameter's value replicates to other players.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AvatarParamSync {
    /// Sent to other players; counts toward the bit budget.
    #[default]
    Synced,
    /// Lives on the wearer's machine only (menus and saved data still use it).
    LocalOnly,
}

impl AvatarParamSync {
    /// Every mode, in declaration order.
    pub const ALL: [AvatarParamSync; 2] = [Self::Synced, Self::LocalOnly];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Synced => "Synced",
            Self::LocalOnly => "Local only",
        }
    }
}

/// How a parameter declared below the avatar root is named in the finished
/// avatar (see `avatar_build`'s scoping pass). Parameters declared on the root
/// are always global.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
pub enum AvatarParamScope {
    /// Shared by name with the whole avatar (how an add-on binds to a
    /// parameter the avatar already has).
    #[default]
    Global,
    /// Private to the declaring object's subtree: renamed
    /// `name$<12 hex of a hash of the object's path>`, the same every build.
    Internal,
    /// Renamed to this name everywhere in the subtree.
    RenameTo(String),
}

/// One declared avatar parameter.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AvatarParam {
    /// Parameter name (also what controllers and menus reference).
    pub name: String,
    /// What the parameter holds.
    pub kind: AvatarParamKind,
    /// Value before the engine or the player writes anything.
    pub default: f32,
    /// Remembered per user across sessions.
    pub saved: bool,
    /// Whether the value replicates to other players.
    pub sync: AvatarParamSync,
    /// Only meaningful on an `AvatarParameters` below the avatar root.
    #[serde(skip_serializing_if = "is_global")]
    pub scope: AvatarParamScope,
}

fn is_global(s: &AvatarParamScope) -> bool {
    *s == AvatarParamScope::Global
}

impl Default for AvatarParam {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: AvatarParamKind::Float,
            default: 0.0,
            saved: false,
            sync: AvatarParamSync::Synced,
            scope: AvatarParamScope::Global,
        }
    }
}

impl AvatarParam {
    /// Declare a parameter of this kind with default values.
    pub fn new(name: &str, kind: AvatarParamKind) -> Self {
        Self { name: name.to_owned(), kind, ..Default::default() }
    }
    /// True when the value replicates to other players.
    pub fn is_synced(&self) -> bool {
        self.sync == AvatarParamSync::Synced
    }
    /// Bits this parameter costs against the budget (0 when local only).
    pub fn bit_cost(&self) -> u32 {
        if self.is_synced() { self.kind.bits() } else { 0 }
    }
    /// The value a write of `v` leaves: Bool 0/1, Int rounded, and synced
    /// values held to the wire range (Int 0..=255, Float -1..=1).
    pub fn coerce(&self, v: f32) -> f32 {
        let v = self.kind.coerce_local(v);
        match (self.kind, self.is_synced()) {
            (AvatarParamKind::Int, true) => v.clamp(0.0, 255.0),
            (AvatarParamKind::Float, true) => v.clamp(-1.0, 1.0),
            _ => v,
        }
    }
}

/// What a remote copy sees of a synced float: clamped to -1..1 and stored as a
/// signed byte, so 0.5 arrives as 0.504.
pub fn quantise_float(v: f32) -> f32 {
    (v.clamp(-1.0, 1.0) * 127.0).round() / 127.0
}

/// Parameters the engine fills in itself. They are always present, never
/// saved or synced, and a declared parameter of the same name is ignored.
/// Which ones the engine feeds today is listed in `Features.md`; the rest hold
/// their default until a source is wired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum BuiltinParam {
    /// 1 while the avatar is on the local player's own machine.
    IsLocal,
    /// 1 while the avatar is a menu / preview copy.
    PreviewMode,
    /// Index of the viseme the face is forming.
    Viseme,
    /// Voice activity (0 = silent).
    Voice,
    /// Left-hand gesture index.
    GestureLeft,
    /// Right-hand gesture index.
    GestureRight,
    /// Weight of the left-hand gesture (0..1).
    GestureLeftWeight,
    /// Weight of the right-hand gesture (0..1).
    GestureRightWeight,
    /// Turning speed about Y (degrees / second).
    AngularY,
    /// Velocity along X (m/s).
    VelocityX,
    /// Velocity along Y (m/s).
    VelocityY,
    /// Velocity along Z (m/s).
    VelocityZ,
    /// Speed (m/s).
    VelocityMagnitude,
    /// How upright the avatar stands (0..1).
    Upright,
    /// Standing on ground.
    Grounded,
    /// Seated / in a station.
    Seated,
    /// The player is away from keyboard.
    Afk,
    /// Sitting in an interaction station.
    InStation,
    /// Tracking method in use (engine-defined index).
    TrackingType,
    /// The player is in VR.
    VrMode,
    /// The player muted their microphone.
    MuteSelf,
    /// The player muted other players.
    Earmuffs,
    /// The avatar's animator is running.
    IsAnimatorEnabled,
    /// The player changed their avatar scale.
    ScaleModified,
    /// The player's avatar scale.
    ScaleFactor,
    /// 1 / the player's avatar scale.
    ScaleFactorInverse,
    /// Standing eye height (m).
    EyeHeightAsMeters,
    /// Eye height relative to the avatar's default.
    EyeHeightAsPercent,
    /// The player is on the local player's friends list.
    IsOnFriendsList,
    /// The avatar's version counter.
    AvatarVersion,
}

impl BuiltinParam {
    /// Every built-in, in slot order.
    pub const ALL: [BuiltinParam; 30] = [
        Self::IsLocal,
        Self::PreviewMode,
        Self::Viseme,
        Self::Voice,
        Self::GestureLeft,
        Self::GestureRight,
        Self::GestureLeftWeight,
        Self::GestureRightWeight,
        Self::AngularY,
        Self::VelocityX,
        Self::VelocityY,
        Self::VelocityZ,
        Self::VelocityMagnitude,
        Self::Upright,
        Self::Grounded,
        Self::Seated,
        Self::Afk,
        Self::InStation,
        Self::TrackingType,
        Self::VrMode,
        Self::MuteSelf,
        Self::Earmuffs,
        Self::IsAnimatorEnabled,
        Self::ScaleModified,
        Self::ScaleFactor,
        Self::ScaleFactorInverse,
        Self::EyeHeightAsMeters,
        Self::EyeHeightAsPercent,
        Self::IsOnFriendsList,
        Self::AvatarVersion,
    ];
    /// Parameter name (also what controllers and menus reference).
    pub fn name(self) -> &'static str {
        match self {
            Self::IsLocal => "IsLocal",
            Self::PreviewMode => "PreviewMode",
            Self::Viseme => "Viseme",
            Self::Voice => "Voice",
            Self::GestureLeft => "GestureLeft",
            Self::GestureRight => "GestureRight",
            Self::GestureLeftWeight => "GestureLeftWeight",
            Self::GestureRightWeight => "GestureRightWeight",
            Self::AngularY => "AngularY",
            Self::VelocityX => "VelocityX",
            Self::VelocityY => "VelocityY",
            Self::VelocityZ => "VelocityZ",
            Self::VelocityMagnitude => "VelocityMagnitude",
            Self::Upright => "Upright",
            Self::Grounded => "Grounded",
            Self::Seated => "Seated",
            Self::Afk => "AFK",
            Self::InStation => "InStation",
            Self::TrackingType => "TrackingType",
            Self::VrMode => "VRMode",
            Self::MuteSelf => "MuteSelf",
            Self::Earmuffs => "Earmuffs",
            Self::IsAnimatorEnabled => "IsAnimatorEnabled",
            Self::ScaleModified => "ScaleModified",
            Self::ScaleFactor => "ScaleFactor",
            Self::ScaleFactorInverse => "ScaleFactorInverse",
            Self::EyeHeightAsMeters => "EyeHeightAsMeters",
            Self::EyeHeightAsPercent => "EyeHeightAsPercent",
            Self::IsOnFriendsList => "IsOnFriendsList",
            Self::AvatarVersion => "AvatarVersion",
        }
    }
    /// The built-in's parameter kind.
    pub fn kind(self) -> AvatarParamKind {
        use AvatarParamKind::*;
        match self {
            Self::IsLocal | Self::Grounded | Self::Seated | Self::Afk | Self::InStation | Self::MuteSelf | Self::Earmuffs
            | Self::IsAnimatorEnabled | Self::ScaleModified | Self::IsOnFriendsList => Bool,
            Self::PreviewMode | Self::Viseme | Self::GestureLeft | Self::GestureRight | Self::TrackingType | Self::VrMode
            | Self::AvatarVersion => Int,
            _ => Float,
        }
    }
    /// Value before the engine writes anything.
    pub fn default_value(self) -> f32 {
        match self {
            Self::IsLocal | Self::IsAnimatorEnabled | Self::ScaleFactor | Self::ScaleFactorInverse => 1.0,
            _ => 0.0,
        }
    }
    /// The built-in called `name`.
    pub fn from_name(name: &str) -> Option<BuiltinParam> {
        Self::ALL.iter().copied().find(|b| b.name() == name)
    }
    /// True for built-ins that only exist on other players' copies of an avatar.
    pub fn remote_only(self) -> bool {
        self == Self::IsOnFriendsList
    }
}

/// The wire cost of `params` in bits: synced parameters only, each name once,
/// built-ins never.
pub fn synced_bits(params: &[AvatarParam]) -> u32 {
    let mut seen: HashSet<&str> = HashSet::with_capacity(params.len());
    params
        .iter()
        .filter(|p| !p.name.is_empty() && BuiltinParam::from_name(&p.name).is_none() && seen.insert(p.name.as_str()))
        .map(AvatarParam::bit_cost)
        .sum()
}

/// How full a budget is, for the meter (amber past half, red past the limit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetLevel {
    /// Within budget.
    Ok,
    /// Past half the budget.
    Warn,
    /// Past the budget.
    Over,
}

/// How much of a synced-bit budget a set of parameters uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamBudget {
    /// Bits used.
    pub used: u32,
    /// Bit limit.
    pub max: u32,
}

impl ParamBudget {
    /// Measure `params` against a bit limit.
    pub fn of(params: &[AvatarParam], max: u32) -> Self {
        Self { used: synced_bits(params), max }
    }
    /// Bits left before the limit.
    pub fn remaining(&self) -> u32 {
        self.max.saturating_sub(self.used)
    }
    /// True when the used bits exceed the limit.
    pub fn is_over(&self) -> bool {
        self.used > self.max
    }
    /// 0..1+ (above 1 = over budget).
    pub fn fraction(&self) -> f32 {
        if self.max == 0 { if self.used == 0 { 0.0 } else { f32::INFINITY } } else { self.used as f32 / self.max as f32 }
    }
    /// Ok / Warn / Over, for the budget meter.
    pub fn level(&self) -> BudgetLevel {
        if self.is_over() {
            BudgetLevel::Over
        } else if self.used * 2 > self.max {
            BudgetLevel::Warn
        } else {
            BudgetLevel::Ok
        }
    }
    /// "cost/MAX Synced Bits".
    pub fn label(&self) -> String {
        format!("{}/{} synced bits", self.used, self.max)
    }
    /// Whether adding `extra` still fits.
    pub fn fits(&self, extra: &AvatarParam) -> bool {
        self.used + extra.bit_cost() <= self.max
    }
}

/// Values worth keeping between sessions: the saved parameters, by name.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct SavedParams {
    /// Saved values by parameter name.
    pub values: BTreeMap<String, f32>,
}

impl SavedParams {
    /// Serialize to RON.
    pub fn to_ron(&self) -> String {
        ron::to_string(self).unwrap_or_default()
    }
    /// Parse from RON.
    pub fn from_ron(text: &str) -> Result<Self, String> {
        ron::from_str(text).map_err(|e| e.to_string())
    }
    /// `<dir>/avatar_params/<package hash>.ron` (the hash is reduced to
    /// file-name-safe characters).
    pub fn path_for(dir: &Path, package_hash: &str) -> PathBuf {
        let safe: String = package_hash.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
        dir.join("avatar_params").join(format!("{safe}.ron"))
    }
    /// Write to `path` (creating parent directories).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_ron())
    }
    /// Missing or unreadable files give an empty set (first run).
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|t| Self::from_ron(&t).ok()).unwrap_or_default()
    }
}

/// An avatar's live parameter values. Slots are resolved once (`index_of`);
/// reads and writes by slot are array accesses.
#[derive(Clone, Debug)]
pub struct AvatarParamStore {
    names: Vec<String>,
    /// Declared parameters (`None` for built-ins).
    decl: Vec<Option<AvatarParam>>,
    kinds: Vec<AvatarParamKind>,
    values: Vec<f32>,
    /// One bit per slot: changed since the last flush to the Animator.
    dirty: Vec<u64>,
    index: HashMap<String, usize>,
}

impl Default for AvatarParamStore {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl AvatarParamStore {
    /// Built-ins, then `params` (duplicates, empty names and built-in names
    /// are skipped), every value at its default.
    pub fn new(params: &[AvatarParam]) -> Self {
        let n = BuiltinParam::ALL.len();
        let mut s = Self {
            names: Vec::with_capacity(n + params.len()),
            decl: Vec::with_capacity(n + params.len()),
            kinds: Vec::with_capacity(n + params.len()),
            values: Vec::with_capacity(n + params.len()),
            dirty: Vec::new(),
            index: HashMap::with_capacity(n + params.len()),
        };
        for b in BuiltinParam::ALL {
            s.index.insert(b.name().to_owned(), s.names.len());
            s.names.push(b.name().to_owned());
            s.decl.push(None);
            s.kinds.push(b.kind());
            s.values.push(b.default_value());
        }
        for p in params {
            if p.name.is_empty() || s.index.contains_key(&p.name) {
                continue;
            }
            s.index.insert(p.name.clone(), s.names.len());
            s.names.push(p.name.clone());
            s.kinds.push(p.kind);
            s.values.push(p.coerce(p.default));
            s.decl.push(Some(p.clone()));
        }
        // Everything is new to the Animator: the first flush writes it all.
        s.dirty = vec![u64::MAX; s.names.len().div_ceil(64)];
        s
    }

    /// An allocation-free stand-in with no slots, for swapping a store out of
    /// a component for a moment. Not usable until the real one is put back.
    pub fn placeholder() -> Self {
        Self { names: Vec::new(), decl: Vec::new(), kinds: Vec::new(), values: Vec::new(), dirty: Vec::new(), index: HashMap::new() }
    }

    /// Number of slots (built-ins included).
    pub fn len(&self) -> usize {
        self.names.len()
    }
    /// True when the store has no slots.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
    /// The slot of the named parameter.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }
    /// The parameter name at `slot`.
    pub fn name(&self, slot: usize) -> &str {
        &self.names[slot]
    }
    /// The kind at `slot`.
    pub fn kind(&self, slot: usize) -> AvatarParamKind {
        self.kinds[slot]
    }
    /// The value at `slot`.
    pub fn get(&self, slot: usize) -> f32 {
        self.values[slot]
    }
    /// The built-in's current value.
    pub fn get_builtin(&self, b: BuiltinParam) -> f32 {
        self.values[b as usize]
    }
    /// True when `slot` is a built-in (not a declared parameter).
    pub fn is_builtin(&self, slot: usize) -> bool {
        slot < BuiltinParam::ALL.len()
    }
    /// The declaration behind `slot` (`None` for built-ins).
    pub fn declared(&self, slot: usize) -> Option<&AvatarParam> {
        self.decl.get(slot).and_then(Option::as_ref)
    }
    /// Slots of the synced declared parameters, in name order: the wire layout.
    pub fn synced_slots(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.decl.len()).filter(|&s| self.decl[s].as_ref().is_some_and(AvatarParam::is_synced)).collect();
        v.sort_by(|&a, &b| self.names[a].cmp(&self.names[b]));
        v
    }

    fn mark(&mut self, slot: usize) {
        self.dirty[slot / 64] |= 1 << (slot % 64);
    }

    /// Write a declared parameter, coerced to its kind and range.
    pub fn set(&mut self, slot: usize, v: f32) {
        if !v.is_finite() {
            return;
        }
        let v = match &self.decl[slot] {
            Some(p) => p.coerce(v),
            None => self.kinds[slot].coerce_local(v),
        };
        if self.values[slot] != v {
            self.values[slot] = v;
            self.mark(slot);
        }
    }
    /// Write a built-in, coerced to its kind.
    pub fn set_builtin(&mut self, b: BuiltinParam, v: f32) {
        self.set(b as usize, v);
    }

    /// Back to the declared defaults (built-ins are the engine's to set).
    pub fn reset_to_defaults(&mut self) {
        for slot in BuiltinParam::ALL.len()..self.names.len() {
            let d = self.decl[slot].as_ref().map_or(0.0, |p| p.default);
            self.set(slot, d);
        }
    }

    /// Write the slots that changed since the last flush into the Animator's
    /// parameter map, then clear the marks.
    pub fn flush_to(&mut self, map: &mut HashMap<String, f32>) {
        for w in 0..self.dirty.len() {
            let mut bits = self.dirty[w];
            while bits != 0 {
                let slot = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if slot >= self.names.len() {
                    break;
                }
                match map.get_mut(&self.names[slot]) {
                    Some(x) => *x = self.values[slot],
                    None => {
                        map.insert(self.names[slot].clone(), self.values[slot]);
                    }
                }
            }
            self.dirty[w] = 0;
        }
    }

    /// Take values state-behaviour drivers (or code) wrote into the Animator's
    /// map for declared parameters. A name lookup per declared parameter, so
    /// call it only when something may have written (`params_touched`).
    pub fn absorb_from(&mut self, map: &HashMap<String, f32>) {
        for slot in BuiltinParam::ALL.len()..self.names.len() {
            if let Some(&v) = map.get(&self.names[slot]) {
                self.set(slot, v);
            }
        }
    }

    /// Slots of the declared parameters flagged `saved` (never built-ins).
    pub fn saved_slots(&self) -> Vec<usize> {
        (0..self.decl.len()).filter(|&s| self.decl[s].as_ref().is_some_and(|p| p.saved)).collect()
    }

    /// The saved parameters' current values.
    pub fn saved_values(&self) -> SavedParams {
        let mut out = SavedParams::default();
        for (slot, d) in self.decl.iter().enumerate() {
            if let Some(p) = d
                && p.saved
            {
                out.values.insert(self.names[slot].clone(), self.values[slot]);
            }
        }
        out
    }

    /// Restore saved values. Names that are not (or no longer) saved
    /// parameters are ignored.
    pub fn load_saved(&mut self, saved: &SavedParams) {
        for (name, &v) in &saved.values {
            if let Some(slot) = self.index_of(name)
                && self.decl[slot].as_ref().is_some_and(|p| p.saved)
            {
                self.set(slot, v);
            }
        }
    }
}

/// The parameters one avatar (or one part of it) declares. Put it on the
/// avatar root; the store below is what Play runs on.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarParameters {
    /// The declared parameters.
    pub params: Vec<AvatarParam>,
    /// Synced-bit budget for this avatar; the project's content limit
    /// (`ContentLimits::max_param_bits`) caps it.
    pub budget_bits: u32,
    /// Live values (built from `params`; not saved).
    #[serde(skip)]
    pub store: AvatarParamStore,
    /// `store` was built from `params` (Play builds it on first use).
    #[serde(skip)]
    pub ready: bool,
    /// A puppet is open on this avatar's radial menu: its parameters sync at
    /// the fast rate (set by the menu, read by the network sync).
    #[serde(skip)]
    pub fast_sync: bool,
}

impl Default for AvatarParameters {
    fn default() -> Self {
        Self { params: Vec::new(), budget_bits: DEFAULT_PARAM_BUDGET_BITS, store: AvatarParamStore::default(), ready: false, fast_sync: false }
    }
}

impl AvatarParameters {
    /// Rebuild the store from `params` (Play start, or after an edit).
    pub fn rebuild_store(&mut self) {
        self.store = AvatarParamStore::new(&self.params);
        self.ready = true;
    }
    /// This avatar's synced-bit use against `max`.
    pub fn budget(&self, max: u32) -> ParamBudget {
        ParamBudget::of(&self.params, max)
    }
    /// The budget this avatar is held to, given the project's cap.
    pub fn effective_budget(&self, project_cap: u32) -> ParamBudget {
        ParamBudget::of(&self.params, self.budget_bits.min(project_cap))
    }
    /// Names declared more than once, empty, or shadowing a built-in.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for p in &self.params {
            if p.name.is_empty() {
                out.push("A parameter has no name.".to_owned());
            } else if BuiltinParam::from_name(&p.name).is_some() {
                out.push(format!("'{}' is a built-in parameter and is ignored.", p.name));
            } else if !seen.insert(p.name.as_str()) {
                out.push(format!("'{}' is declared more than once.", p.name));
            }
        }
        out
    }
}

impl TypedComponent for AvatarParameters {
    const NAME: &'static str = "Avatar Parameters";
}

/// What an expression-menu item does when used.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MenuItemKind {
    /// Sets `parameter` to `value` while held (at least 0.2 s), 0 on release.
    Button,
    /// Click sets `parameter` to `value`, click again sets 0.
    #[default]
    Toggle,
    /// Opens a page of the item's children. `parameter` (optional) is `value`
    /// while the page is open.
    SubMenu,
    /// One axis 0..1 from the stick angle.
    RadialPuppet,
    /// Two axes, -1..1 each.
    TwoAxisPuppet,
    /// Four directions, 0..1 each.
    FourAxisPuppet,
}

impl MenuItemKind {
    /// Every kind, in declaration order.
    pub const ALL: [MenuItemKind; 6] =
        [Self::Button, Self::Toggle, Self::SubMenu, Self::RadialPuppet, Self::TwoAxisPuppet, Self::FourAxisPuppet];
    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Button => "Button",
            Self::Toggle => "Toggle",
            Self::SubMenu => "Sub Menu",
            Self::RadialPuppet => "Radial Puppet",
            Self::TwoAxisPuppet => "Two-Axis Puppet",
            Self::FourAxisPuppet => "Four-Axis Puppet",
        }
    }
    /// Sub-parameters a puppet of this kind needs (0 for the others).
    pub fn sub_parameter_count(self) -> usize {
        match self {
            Self::RadialPuppet => 1,
            Self::TwoAxisPuppet => 2,
            Self::FourAxisPuppet => 4,
            _ => 0,
        }
    }
    /// True for the puppet kinds (they drive sub-parameters).
    pub fn is_puppet(self) -> bool {
        self.sub_parameter_count() > 0
    }
    /// Range of each sub-parameter.
    pub fn sub_parameter_range(self) -> (f32, f32) {
        if self == Self::TwoAxisPuppet { (-1.0, 1.0) } else { (0.0, 1.0) }
    }
}

/// One entry of the avatar's expression menu. The object hierarchy is the menu:
/// a Sub Menu item's children are its page, and the Sub Menu on the avatar
/// root is the main page.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AvatarMenuItem {
    /// Text shown in the menu.
    pub label: String,
    /// Project-relative icon (capped at 256 px when built).
    pub icon: String,
    /// What the item does when used.
    pub kind: MenuItemKind,
    /// Parameter this item drives (empty = let the build pass pick one).
    pub parameter: String,
    /// Value written for Button / Toggle, and the "page open" value of
    /// Sub Menu and puppets. 0 reads as "off", so use a non-zero value.
    pub value: f32,
    /// Puppet axes: 1, 2 or 4 parameter names by kind.
    pub sub_parameters: Vec<String>,
    /// Puppet direction labels (up, right, down, left).
    pub labels: Vec<String>,
    /// Sub Menu from a hand-authored `.avmenu` asset instead of children.
    pub sub_menu_asset: String,
    /// Remember the value between sessions.
    pub saved: bool,
    /// Send the value to other players.
    pub synced: bool,
    /// Starts selected.
    pub is_default: bool,
}

impl Default for AvatarMenuItem {
    fn default() -> Self {
        Self {
            label: "Item".into(),
            icon: String::new(),
            kind: MenuItemKind::Toggle,
            parameter: String::new(),
            value: 1.0,
            sub_parameters: Vec::new(),
            labels: Vec::new(),
            sub_menu_asset: String::new(),
            saved: true,
            synced: true,
            is_default: false,
        }
    }
}

impl AvatarMenuItem {
    /// What a click does to a Toggle whose parameter holds `current`.
    pub fn toggled(&self, current: f32) -> f32 {
        if (current - self.value).abs() < 1e-4 { 0.0 } else { self.value }
    }
    /// Whether the Toggle / Button is on at `current`.
    pub fn is_on(&self, current: f32) -> bool {
        (current - self.value).abs() < 1e-4
    }
    /// Declared parameters this item needs but `known` lacks, plus shape
    /// problems (wrong puppet axis count, value 0).
    pub fn problems(&self, known: &dyn Fn(&str) -> bool) -> Vec<String> {
        let mut out = Vec::new();
        let drives = matches!(self.kind, MenuItemKind::Button | MenuItemKind::Toggle);
        if drives && self.parameter.is_empty() {
            out.push(format!("'{}' has no parameter.", self.label));
        }
        if !self.parameter.is_empty() && !known(&self.parameter) {
            out.push(format!("'{}' uses '{}', which is not declared.", self.label, self.parameter));
        }
        if drives && self.value == 0.0 {
            out.push(format!("'{}' sets its parameter to 0, which cannot be told apart from off.", self.label));
        }
        let need = self.kind.sub_parameter_count();
        if self.sub_parameters.len() != need && self.kind.is_puppet() {
            out.push(format!("'{}' needs {need} sub-parameter(s), has {}.", self.label, self.sub_parameters.len()));
        }
        for sp in self.sub_parameters.iter().filter(|s| !s.is_empty()) {
            if !known(sp) {
                out.push(format!("'{}' uses '{}', which is not declared.", self.label, sp));
            }
        }
        out
    }
}

impl TypedComponent for AvatarMenuItem {
    const NAME: &'static str = "Avatar Menu Item";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str, kind: AvatarParamKind) -> AvatarParam {
        AvatarParam::new(name, kind)
    }

    #[test]
    fn bit_costs_and_budget_maths() {
        let mut params = vec![p("A", AvatarParamKind::Bool), p("B", AvatarParamKind::Int), p("C", AvatarParamKind::Float)];
        assert_eq!(synced_bits(&params), 1 + 8 + 8);
        params[2].sync = AvatarParamSync::LocalOnly;
        assert_eq!(synced_bits(&params), 9, "local-only parameters are free");
        // Duplicates count once; built-ins never.
        params.push(p("A", AvatarParamKind::Bool));
        params.push(p("Grounded", AvatarParamKind::Bool));
        params.push(p("", AvatarParamKind::Int));
        assert_eq!(synced_bits(&params), 9);

        let b = ParamBudget::of(&params, 256);
        assert_eq!((b.used, b.max, b.remaining()), (9, 256, 247));
        assert_eq!(b.level(), BudgetLevel::Ok);
        assert!(b.fits(&p("X", AvatarParamKind::Float)));

        let full: Vec<AvatarParam> = (0..32).map(|i| p(&format!("f{i}"), AvatarParamKind::Float)).collect();
        let b = ParamBudget::of(&full, 256);
        assert_eq!(b.used, 256);
        assert_eq!(b.level(), BudgetLevel::Warn);
        assert!(!b.is_over());
        assert!(!b.fits(&p("one_more", AvatarParamKind::Bool)));
        assert!(b.fits(&AvatarParam { sync: AvatarParamSync::LocalOnly, ..p("loc", AvatarParamKind::Float) }));
        let b = ParamBudget::of(&full, 128);
        assert!(b.is_over());
        assert_eq!(b.level(), BudgetLevel::Over);
        assert_eq!(b.remaining(), 0);
        assert_eq!(b.label(), "256/128 synced bits");
        assert_eq!(ParamBudget::of(&full, 0).fraction(), f32::INFINITY);
    }

    #[test]
    fn coerce_and_quantise() {
        let int = p("i", AvatarParamKind::Int);
        assert_eq!(int.coerce(300.4), 255.0);
        assert_eq!(int.coerce(-3.0), 0.0);
        assert_eq!(int.coerce(2.6), 3.0);
        let local = AvatarParam { sync: AvatarParamSync::LocalOnly, ..int.clone() };
        assert_eq!(local.coerce(300.0), 300.0, "local values are not held to the wire range");
        let f = p("f", AvatarParamKind::Float);
        assert_eq!(f.coerce(4.0), 1.0);
        assert_eq!(f.coerce(-4.0), -1.0);
        assert_eq!(p("b", AvatarParamKind::Bool).coerce(0.3), 1.0);
        assert_eq!(p("b", AvatarParamKind::Bool).coerce(0.0), 0.0);
        assert!((quantise_float(0.5) - 0.503937).abs() < 1e-4);
        assert_eq!(quantise_float(2.0), 1.0);
    }

    #[test]
    fn builtins_are_always_present_with_defaults() {
        let s = AvatarParamStore::default();
        for b in BuiltinParam::ALL {
            assert_eq!(s.index_of(b.name()), Some(b as usize), "{}", b.name());
            assert_eq!(s.get_builtin(b), b.default_value());
            assert_eq!(BuiltinParam::from_name(b.name()), Some(b));
        }
        assert_eq!(s.get_builtin(BuiltinParam::IsLocal), 1.0);
        assert_eq!(s.get_builtin(BuiltinParam::ScaleFactor), 1.0);
        assert_eq!(BuiltinParam::Grounded.kind(), AvatarParamKind::Bool);
        assert_eq!(BuiltinParam::Viseme.kind(), AvatarParamKind::Int);
        assert_eq!(BuiltinParam::Voice.kind(), AvatarParamKind::Float);
        assert_eq!(BuiltinParam::ALL.len(), 30);
        // Slots follow the enum order.
        assert!(BuiltinParam::ALL.iter().enumerate().all(|(i, b)| *b as usize == i));
    }

    #[test]
    fn builtin_writes_ignore_wire_ranges_and_round_ints() {
        let mut s = AvatarParamStore::default();
        s.set_builtin(BuiltinParam::VelocityZ, 7.5);
        assert_eq!(s.get_builtin(BuiltinParam::VelocityZ), 7.5);
        s.set_builtin(BuiltinParam::Viseme, 3.4);
        assert_eq!(s.get_builtin(BuiltinParam::Viseme), 3.0);
        s.set_builtin(BuiltinParam::Grounded, 0.9);
        assert_eq!(s.get_builtin(BuiltinParam::Grounded), 1.0);
        s.set_builtin(BuiltinParam::Voice, f32::NAN);
        assert_eq!(s.get_builtin(BuiltinParam::Voice), 0.0, "non-finite writes are dropped");
    }

    #[test]
    fn declared_parameters_start_at_default_and_shadowing_is_ignored() {
        let params = vec![
            AvatarParam { default: 1.0, ..p("Hat", AvatarParamKind::Bool) },
            AvatarParam { default: 3.0, ..p("Outfit", AvatarParamKind::Int) },
            p("Grounded", AvatarParamKind::Float),
            p("Hat", AvatarParamKind::Int),
            p("", AvatarParamKind::Bool),
        ];
        let s = AvatarParamStore::new(&params);
        assert_eq!(s.len(), BuiltinParam::ALL.len() + 2);
        assert_eq!(s.get(s.index_of("Hat").unwrap()), 1.0);
        assert_eq!(s.kind(s.index_of("Hat").unwrap()), AvatarParamKind::Bool);
        assert_eq!(s.get(s.index_of("Outfit").unwrap()), 3.0);
        assert_eq!(s.kind(s.index_of("Grounded").unwrap()), AvatarParamKind::Bool, "the built-in wins");
    }

    #[test]
    fn flush_writes_only_changed_values() {
        let mut s = AvatarParamStore::new(&[p("Hat", AvatarParamKind::Bool), p("Hue", AvatarParamKind::Float)]);
        let mut map = HashMap::new();
        s.flush_to(&mut map);
        assert_eq!(map.len(), s.len(), "the first flush writes everything");
        assert_eq!(map["IsLocal"], 1.0);

        // Nothing changed: a stale map value is left alone.
        map.insert("Hat".into(), 9.0);
        s.flush_to(&mut map);
        assert_eq!(map["Hat"], 9.0);

        let hat = s.index_of("Hat").unwrap();
        s.set(hat, 1.0);
        s.set(hat, 1.0);
        s.set_builtin(BuiltinParam::VelocityMagnitude, 2.0);
        s.flush_to(&mut map);
        assert_eq!(map["Hat"], 1.0);
        assert_eq!(map["VelocityMagnitude"], 2.0);
        assert_eq!(map["Hue"], 0.0);
        map.insert("Hat".into(), 5.0);
        s.flush_to(&mut map);
        assert_eq!(map["Hat"], 5.0, "a flush clears the marks");
    }

    #[test]
    fn absorb_takes_driver_writes_for_declared_parameters_only() {
        let mut s = AvatarParamStore::new(&[p("Outfit", AvatarParamKind::Int)]);
        let mut map = HashMap::new();
        s.flush_to(&mut map);
        map.insert("Outfit".into(), 2.0);
        map.insert("IsLocal".into(), 0.0);
        s.absorb_from(&map);
        assert_eq!(s.get(s.index_of("Outfit").unwrap()), 2.0);
        assert_eq!(s.get_builtin(BuiltinParam::IsLocal), 1.0, "built-ins belong to the engine");
        // Range rules apply on the way in.
        map.insert("Outfit".into(), 999.0);
        s.absorb_from(&map);
        assert_eq!(s.get(s.index_of("Outfit").unwrap()), 255.0);
    }

    #[test]
    fn save_and_load_round_trip_with_defaults() {
        let params = vec![
            AvatarParam { saved: true, default: 1.0, ..p("Hat", AvatarParamKind::Bool) },
            AvatarParam { saved: true, ..p("Outfit", AvatarParamKind::Int) },
            p("Temp", AvatarParamKind::Float),
        ];
        let mut s = AvatarParamStore::new(&params);
        let (hat, outfit, temp) = (s.index_of("Hat").unwrap(), s.index_of("Outfit").unwrap(), s.index_of("Temp").unwrap());
        s.set(hat, 0.0);
        s.set(outfit, 4.0);
        s.set(temp, 0.5);
        let saved = s.saved_values();
        assert_eq!(saved.values.len(), 2, "only saved parameters are kept");

        let text = saved.to_ron();
        let back = SavedParams::from_ron(&text).unwrap();
        assert_eq!(back, saved);

        let mut fresh = AvatarParamStore::new(&params);
        assert_eq!(fresh.get(hat), 1.0, "defaults before loading");
        fresh.load_saved(&back);
        assert_eq!(fresh.get(hat), 0.0);
        assert_eq!(fresh.get(outfit), 4.0);
        assert_eq!(fresh.get(temp), 0.0);

        // Unknown and unsaved names are ignored.
        let mut stray = SavedParams::default();
        stray.values.insert("Temp".into(), 0.9);
        stray.values.insert("Gone".into(), 1.0);
        fresh.load_saved(&stray);
        assert_eq!(fresh.get(temp), 0.0);

        fresh.reset_to_defaults();
        assert_eq!(fresh.get(hat), 1.0);
        assert_eq!(fresh.get(outfit), 0.0);
    }

    #[test]
    fn saved_file_round_trip_and_missing_file() {
        let dir = std::env::temp_dir().join(format!("zest-avparams-{}", std::process::id()));
        let path = SavedParams::path_for(&dir, "ab/../cd12");
        assert!(path.ends_with("avatar_params/abcd12.ron"), "{path:?}");
        assert!(SavedParams::load(&path).values.is_empty());
        let mut saved = SavedParams::default();
        saved.values.insert("Hat".into(), 1.0);
        saved.save(&path).unwrap();
        assert_eq!(SavedParams::load(&path), saved);
        let _ = std::fs::remove_dir_all(&dir);
        // Old files with fields we do not know yet, and empty ones, still load.
        assert!(SavedParams::from_ron("()").unwrap().values.is_empty());
    }

    #[test]
    fn component_serde_defaults_and_store_rebuild() {
        let text = "(params:[(name:\"Hat\")])";
        let mut c: AvatarParameters = ron::from_str(text).unwrap();
        assert_eq!(c.params[0].kind, AvatarParamKind::Float);
        assert!(c.params[0].is_synced());
        c.rebuild_store();
        assert!(c.store.index_of("Hat").is_some());
        assert!(!ron::to_string(&c).unwrap().contains("store"));
        let empty: AvatarParameters = ron::from_str("()").unwrap();
        assert!(empty.params.is_empty());
        assert_eq!(c.budget_bits, 256);
        c.budget_bits = 100;
        assert_eq!(c.effective_budget(256).max, 100);
        assert_eq!(c.effective_budget(64).max, 64, "the project cap wins");
        let dup = AvatarParameters {
            params: vec![p("A", AvatarParamKind::Bool), p("A", AvatarParamKind::Bool), p("IsLocal", AvatarParamKind::Bool), p("", AvatarParamKind::Bool)],
            ..Default::default()
        };
        assert_eq!(dup.problems().len(), 3);
        assert_eq!(dup.budget(256).used, 1);
    }

    #[test]
    fn menu_item_semantics() {
        let mut t = AvatarMenuItem { parameter: "Hat".into(), value: 1.0, ..Default::default() };
        assert_eq!(t.toggled(0.0), 1.0);
        assert_eq!(t.toggled(1.0), 0.0);
        t.value = 3.0;
        assert_eq!(t.toggled(1.0), 3.0, "another item's value on a shared Int turns this one on");
        assert!(t.is_on(3.0) && !t.is_on(1.0));

        assert_eq!(MenuItemKind::TwoAxisPuppet.sub_parameter_count(), 2);
        assert_eq!(MenuItemKind::TwoAxisPuppet.sub_parameter_range(), (-1.0, 1.0));
        assert_eq!(MenuItemKind::FourAxisPuppet.sub_parameter_range(), (0.0, 1.0));
        assert!(!MenuItemKind::SubMenu.is_puppet());

        let known = |n: &str| n == "Hat";
        assert!(t.problems(&known).is_empty());
        let bad = AvatarMenuItem { kind: MenuItemKind::TwoAxisPuppet, sub_parameters: vec!["X".into()], ..Default::default() };
        let msgs = bad.problems(&known);
        assert!(msgs.iter().any(|m| m.contains("needs 2")));
        assert!(msgs.iter().any(|m| m.contains("'X'")));
        let zero = AvatarMenuItem { parameter: "Hat".into(), value: 0.0, ..Default::default() };
        assert!(zero.problems(&known).iter().any(|m| m.contains("off")));
        let none = AvatarMenuItem::default();
        assert!(none.problems(&known).iter().any(|m| m.contains("no parameter")));
    }

    #[test]
    fn menu_item_serde_defaults_and_ron() {
        let m: AvatarMenuItem = ron::from_str("(label:\"Hat\",kind:Button)").unwrap();
        assert_eq!(m.kind, MenuItemKind::Button);
        assert_eq!(m.value, 1.0);
        assert!(m.saved && m.synced);
        let text = ron::to_string(&m).unwrap();
        assert_eq!(ron::from_str::<AvatarMenuItem>(&text).unwrap(), m);
    }
}
