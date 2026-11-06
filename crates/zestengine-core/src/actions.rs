//! Event/action bindings: "when this happens, call that function on that
//! component, with these values".
//!
//! A component declares the functions it exposes ([`ActionDesc`], through
//! `Component::actions`) and the values other things may read from it
//! ([`PropertyDesc`], through `Component::properties`). An event source
//! (Interactable, Trigger Events, UI Button, animation events) stores an
//! [`EventActions`] of [`EventBinding`]s. Firing a binding resolves the target
//! object and component, evaluates each parameter from its [`ArgSource`]
//! (constant, a property of another component, the instigator or the event
//! payload), checks types, and queues a [`crate::ComponentCommand::InvokeAction`]
//! that the engine applies after the pass, like every other mutation.
//!
//! Everything here is egui-free and uses std types only, so plugin components
//! can expose actions across the dylib boundary the same way they expose
//! their data. The declaration shape (name, typed parameters, typed
//! properties) is deliberately what a WASM export manifest can express (see
//! EVT-2 in IMPLEMENTATION_PLAN.md).

use std::cell::Cell;

use serde::{Deserialize, Serialize};

use crate::{Component, ComponentCommand, ComponentCtx, ObjectId, ObjectRef};

/// The type of an action parameter or property.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ActionType {
    /// A boolean.
    Bool,
    /// A signed 64-bit integer.
    Int,
    /// A 32-bit float.
    Float,
    /// A string.
    String,
    /// A 3D vector.
    Vec3,
    /// An object reference.
    Object,
}

impl ActionType {
    /// Every type, in declaration order.
    pub const ALL: [ActionType; 6] = [Self::Bool, Self::Int, Self::Float, Self::String, Self::Vec3, Self::Object];

    /// Inspector label of this type.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::Int => "Int",
            Self::Float => "Float",
            Self::String => "Text",
            Self::Vec3 => "Vector3",
            Self::Object => "Object",
        }
    }
}

/// A value passed to an action or read from a property.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ArgValue {
    /// A boolean.
    Bool(bool),
    /// A signed 64-bit integer.
    Int(i64),
    /// A 32-bit float.
    Float(f32),
    /// A string.
    String(String),
    /// A 3D vector (xyz).
    Vec3([f32; 3]),
    /// An object reference.
    Object(ObjectRef),
}

impl Default for ArgValue {
    fn default() -> Self {
        ArgValue::Float(0.0)
    }
}

impl ArgValue {
    /// The value's type.
    pub fn ty(&self) -> ActionType {
        match self {
            Self::Bool(_) => ActionType::Bool,
            Self::Int(_) => ActionType::Int,
            Self::Float(_) => ActionType::Float,
            Self::String(_) => ActionType::String,
            Self::Vec3(_) => ActionType::Vec3,
            Self::Object(_) => ActionType::Object,
        }
    }

    /// The zero value of a type (the default of a new constant).
    pub fn zero(ty: ActionType) -> Self {
        match ty {
            ActionType::Bool => Self::Bool(false),
            ActionType::Int => Self::Int(0),
            ActionType::Float => Self::Float(0.0),
            ActionType::String => Self::String(String::new()),
            ActionType::Vec3 => Self::Vec3([0.0; 3]),
            ActionType::Object => Self::Object(ObjectRef::NONE),
        }
    }

    /// Convert to `ty`: numbers and bools convert between each other, any
    /// value converts to text, everything else must already match.
    pub fn coerce(&self, ty: ActionType) -> Result<ArgValue, String> {
        if self.ty() == ty {
            return Ok(self.clone());
        }
        let number = match self {
            Self::Bool(b) => Some(*b as i64 as f64),
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f as f64),
            _ => None,
        };
        match (ty, number) {
            (ActionType::Float, Some(n)) => Ok(Self::Float(n as f32)),
            (ActionType::Int, Some(n)) => Ok(Self::Int(n.round() as i64)),
            (ActionType::Bool, Some(n)) => Ok(Self::Bool(n != 0.0)),
            (ActionType::String, _) => Ok(Self::String(self.display())),
            _ => Err(format!("expected {}, got {}", ty.label(), self.ty().label())),
        }
    }

    /// Human-readable form (also what a value turns into as text).
    pub fn display(&self) -> String {
        match self {
            Self::Bool(b) => b.to_string(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => f.to_string(),
            Self::String(s) => s.clone(),
            Self::Vec3(v) => format!("({}, {}, {})", v[0], v[1], v[2]),
            Self::Object(o) => o.id().map(|i| i.to_string()).unwrap_or_default(),
        }
    }

    /// The value as a bool: booleans pass through, non-zero numbers become
    /// true, anything else is false.
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            _ => false,
        }
    }
    /// The value as an f32: booleans become 0 / 1, ints are widened.
    pub fn as_f32(&self) -> f32 {
        match self {
            Self::Bool(b) => *b as i32 as f32,
            Self::Int(i) => *i as f32,
            Self::Float(f) => *f,
            _ => 0.0,
        }
    }
    /// The value as an i64: booleans become 0 / 1, floats round.
    pub fn as_i64(&self) -> i64 {
        match self {
            Self::Bool(b) => *b as i64,
            Self::Int(i) => *i,
            Self::Float(f) => f.round() as i64,
            _ => 0,
        }
    }
    /// The string when the value is one (empty otherwise).
    pub fn as_str(&self) -> &str {
        match self {
            Self::String(s) => s,
            _ => "",
        }
    }
    /// The Vec3 when the value is one (zero otherwise).
    pub fn as_vec3(&self) -> glam::Vec3 {
        match self {
            Self::Vec3(v) => glam::Vec3::from_array(*v),
            _ => glam::Vec3::ZERO,
        }
    }
    /// The object reference when the value is one (`NONE` otherwise).
    pub fn as_object(&self) -> ObjectRef {
        match self {
            Self::Object(o) => *o,
            _ => ObjectRef::NONE,
        }
    }
}

/// A parameter default. Copy and string-literal based so descriptor tables
/// can be `static`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ParamDefault {
    /// Defaults to the boolean.
    Bool(bool),
    /// Defaults to the integer.
    Int(i64),
    /// Defaults to the float.
    Float(f32),
    /// Defaults to the string.
    Str(&'static str),
    /// Defaults to the vector.
    Vec3([f32; 3]),
    /// Defaults to no object.
    Object,
}

impl ParamDefault {
    /// The default as an `ArgValue`.
    pub fn value(self) -> ArgValue {
        match self {
            Self::Bool(b) => ArgValue::Bool(b),
            Self::Int(i) => ArgValue::Int(i),
            Self::Float(f) => ArgValue::Float(f),
            Self::Str(s) => ArgValue::String(s.to_owned()),
            Self::Vec3(v) => ArgValue::Vec3(v),
            Self::Object => ArgValue::Object(ObjectRef::NONE),
        }
    }
}

/// One parameter of an action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamDesc {
    /// Parameter name.
    pub name: &'static str,
    /// Parameter type.
    pub ty: ActionType,
    /// Used when a binding supplies fewer arguments than the action has parameters.
    pub default: ParamDefault,
}

impl ParamDesc {
    /// A float parameter with this default.
    pub const fn float(name: &'static str, default: f32) -> Self {
        Self { name, ty: ActionType::Float, default: ParamDefault::Float(default) }
    }
    /// An int parameter with this default.
    pub const fn int(name: &'static str, default: i64) -> Self {
        Self { name, ty: ActionType::Int, default: ParamDefault::Int(default) }
    }
    /// A bool parameter with this default.
    pub const fn bool(name: &'static str, default: bool) -> Self {
        Self { name, ty: ActionType::Bool, default: ParamDefault::Bool(default) }
    }
    /// A string parameter defaulting to empty.
    pub const fn string(name: &'static str) -> Self {
        Self { name, ty: ActionType::String, default: ParamDefault::Str("") }
    }
    /// A Vec3 parameter with this default.
    pub const fn vec3(name: &'static str, default: [f32; 3]) -> Self {
        Self { name, ty: ActionType::Vec3, default: ParamDefault::Vec3(default) }
    }
    /// An object parameter defaulting to none.
    pub const fn object(name: &'static str) -> Self {
        Self { name, ty: ActionType::Object, default: ParamDefault::Object }
    }
}

/// A function a component exposes to bindings.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionDesc {
    /// Stable identifier (what files store). Never rename a shipped action.
    pub name: &'static str,
    /// Shown in the editor dropdown.
    pub label: &'static str,
    /// The parameters, in order; bindings supply one source per parameter.
    pub params: &'static [ParamDesc],
    /// Safe to bind from untrusted user-generated content (no scene changes,
    /// no damage to other players, nothing that leaves the world).
    pub ugc_safe: bool,
}

impl ActionDesc {
    /// Build a descriptor (usable in `static` tables).
    pub const fn new(name: &'static str, label: &'static str, params: &'static [ParamDesc], ugc_safe: bool) -> Self {
        Self { name, label, params, ugc_safe }
    }
}

/// A value a component lets other bindings read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropertyDesc {
    /// Property name.
    pub name: &'static str,
    /// Property type.
    pub ty: ActionType,
}

impl PropertyDesc {
    /// Build a descriptor (usable in `static` tables).
    pub const fn new(name: &'static str, ty: ActionType) -> Self {
        Self { name, ty }
    }
}

/// What a component gets when one of its actions is invoked. Mutations that
/// reach beyond the component go through `commands`, like in `ComponentCtx`.
pub struct ActionCtx<'a> {
    /// The object the component sits on.
    pub self_id: ObjectId,
    /// Whoever caused the event (the player who used a switch), if known.
    pub instigator: Option<ObjectId>,
    /// Queued mutations the action asked for (the engine applies them after
    /// the pass).
    pub commands: &'a mut Vec<ComponentCommand>,
    /// Read-only view of the world's properties (the same lookup a binding's
    /// argument sources use). `None` where the caller has no world (tests,
    /// tools); an action that needs it reports "not available".
    pub reader: Option<&'a dyn PropertyReader>,
}

/// Reads a property of any object's component, or of the object itself
/// (`Object` with `name` / `active` / `position`). Implemented by
/// [`ComponentCtx`] (inside an update) and by the engine over the scene (when
/// a queued action is applied), so a component reads the same way in both.
pub trait PropertyReader {
    /// Read `property` of `component` on `object` (empty `component` = the
    /// object itself).
    fn read(&self, object: ObjectId, component: &str, property: &str) -> Result<ArgValue, String>;
}

impl PropertyReader for ComponentCtx<'_> {
    fn read(&self, object: ObjectId, component: &str, property: &str) -> Result<ArgValue, String> {
        read_property(self, object, component, property)
    }
}

/// A property of the object itself, for [`PropertyReader`] implementations
/// that hold the object's fields rather than a [`ComponentCtx`].
pub fn object_property(object: ObjectId, name: &str, active: bool, position: glam::Vec3, property: &str) -> Result<ArgValue, String> {
    match property {
        "" => Ok(ArgValue::Object(object.into())),
        "name" => Ok(ArgValue::String(name.to_owned())),
        "active" => Ok(ArgValue::Bool(active)),
        "position" => Ok(ArgValue::Vec3(position.to_array())),
        _ => Err(format!("object has no property '{property}'")),
    }
}

/// Reads sibling components of the object whose component is running an
/// action (the target's own slot is mutably borrowed, so it is not readable),
/// and everything else through `outer`.
struct SiblingReader<'a> {
    self_id: ObjectId,
    before: &'a [Box<dyn Component>],
    after: &'a [Box<dyn Component>],
    outer: Option<&'a dyn PropertyReader>,
}

impl PropertyReader for SiblingReader<'_> {
    fn read(&self, object: ObjectId, component: &str, property: &str) -> Result<ArgValue, String> {
        if object == self.self_id && !is_object(component) {
            return match self.before.iter().chain(self.after).find(|c| c.type_name() == component) {
                Some(c) => c.read_property(property).ok_or_else(|| format!("{component} has no property '{property}'")),
                None => Err(format!("object has no {component} component (a component cannot read itself during its own action)")),
            };
        }
        match self.outer {
            Some(o) => o.read(object, component, property),
            None => Err("no world view is available here".into()),
        }
    }
}

impl ActionCtx<'_> {
    /// Broadcast an event, readable next frame with `ctx.events::<E>()`.
    pub fn emit<E: std::any::Any>(&mut self, event: E) {
        self.commands.push(ComponentCommand::Emit(crate::AnyBox(Box::new(event))));
    }
}

/// "Unknown action" error for a component's `invoke_action` fallthrough.
pub fn unknown_action(component: &str, name: &str) -> String {
    format!("{component} has no action '{name}'")
}

// ------------------------------------------------------------------ bindings

/// Where one argument of a binding comes from.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ArgSource {
    /// A fixed value.
    Const(ArgValue),
    /// A property of a component on `object` (unset = the object that owns
    /// the event source). An empty `component` reads the object itself.
    Property {
        /// The object to read from (unset = the event source's object).
        object: ObjectRef,
        /// Component type name (empty = the object itself).
        component: String,
        /// Property name.
        property: String,
    },
    /// A property of a component on whoever caused the event. Empty
    /// `component` = the instigator object itself.
    Instigator {
        /// Component type name (empty = the instigator object itself).
        component: String,
        /// Property name.
        property: String,
    },
    /// A field of the event: Interactable has `instigator`, `object`, `uses`,
    /// `on`; Trigger Events has `other`, `object`; animation events have `name`.
    Payload(String),
}

impl Default for ArgSource {
    fn default() -> Self {
        ArgSource::Const(ArgValue::default())
    }
}

/// The component name that means "the object itself" (no component needed).
pub const OBJECT_COMPONENT: &str = "Object";

/// Actions every object has, no component needed.
pub static OBJECT_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("SetActive", "Set Active", &[ParamDesc::bool("active", true)], true),
    ActionDesc::new("ToggleActive", "Toggle Active", &[], true),
    ActionDesc::new("PlaySound", "Play Sound", &[ParamDesc::string("clip"), ParamDesc::float("volume", 1.0)], true),
    ActionDesc::new("EmitEvent", "Emit Event", &[ParamDesc::string("name")], true),
    ActionDesc::new("LoadScene", "Load Scene", &[ParamDesc::string("path")], false),
];

/// Properties every object has.
pub static OBJECT_PROPERTIES: &[PropertyDesc] = &[
    PropertyDesc::new("name", ActionType::String),
    PropertyDesc::new("active", ActionType::Bool),
    PropertyDesc::new("position", ActionType::Vec3),
];

/// Broadcast by the built-in `EmitEvent` action; read with
/// `ctx.events::<ActionEvent>()`.
#[derive(Clone, Debug)]
pub struct ActionEvent {
    /// Event name.
    pub name: String,
    /// The object the binding's event source sits on.
    pub source: ObjectId,
    /// Whoever caused the event, if known.
    pub instigator: Option<ObjectId>,
}

/// Cached resolution and the log-once flag of a binding. Never saved; equal
/// to every other cache so it doesn't affect binding comparison.
#[derive(Default, Debug)]
pub struct BindingState {
    component_slot: Cell<u32>,
    action_index: Cell<u32>,
    warned: Cell<bool>,
}

impl Clone for BindingState {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for BindingState {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// One "call this function on that component" entry.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EventBinding {
    /// The object to call (unset = the object that owns the event source).
    pub target: ObjectRef,
    /// Component type name ("Health"), or "Object"/empty for the built-in
    /// object actions.
    pub component: String,
    /// Action name from the component's declaration.
    pub action: String,
    /// One source per action parameter, in order; missing ones use the default.
    pub args: Vec<ArgSource>,
    /// Which of several components of this type on the target (0 = the
    /// first; files from before this field mean the first).
    pub instance: u32,
    /// The picked component's [`Component::instance_label`] (a Wasm Script's
    /// module name), if it had one. Keeps the binding on the right component
    /// when components are reordered or one is removed.
    pub instance_label: String,
    /// Cached resolution state, rebuilt when needed (not saved).
    #[serde(skip)]
    pub state: BindingState,
}

impl Default for EventBinding {
    fn default() -> Self {
        Self {
            target: ObjectRef::NONE,
            component: OBJECT_COMPONENT.into(),
            action: String::new(),
            args: Vec::new(),
            instance: 0,
            instance_label: String::new(),
            state: BindingState::default(),
        }
    }
}

/// Which component of a type is meant when an object has several of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstanceSel<'a> {
    /// Position among the components of that type (0 = first).
    pub index: usize,
    /// Preferred [`Component::instance_label`]; empty = go by index only.
    pub label: &'a str,
}

impl InstanceSel<'_> {
    /// The plain "first component of the type" selection.
    pub const FIRST: InstanceSel<'static> = InstanceSel { index: 0, label: "" };

    /// True for the plain "first component of the type" case, where a cached
    /// slot can be trusted without counting.
    pub fn is_first(&self) -> bool {
        self.index == 0 && self.label.is_empty()
    }
}

/// Pick the component of type `name` that `sel` means from `(slot, component)`
/// pairs in slot order. A label that matches wins (the index is only a
/// fallback when no component carries it), so a binding follows its module
/// when components are reordered.
pub fn pick_instance<'a>(
    list: impl Iterator<Item = (usize, &'a dyn Component)>,
    name: &str,
    sel: InstanceSel,
) -> Option<(usize, &'a dyn Component)> {
    let mut nth = None;
    for (seen, (slot, c)) in list.filter(|(_, c)| c.type_name() == name).enumerate() {
        if sel.label.is_empty() {
            if seen == sel.index {
                return Some((slot, c));
            }
        } else {
            if c.instance_label().as_deref() == Some(sel.label) {
                return Some((slot, c));
            }
            if seen == sel.index {
                nth = Some((slot, c));
            }
        }
    }
    nth
}

/// What an event source hands to [`EventBinding::fire`].
pub struct FireSource<'a> {
    /// The object the event source sits on.
    pub source: ObjectId,
    /// Whoever caused the event, if known.
    pub instigator: Option<ObjectId>,
    /// Named event fields for [`ArgSource::Payload`].
    pub payload: &'a [(&'static str, ArgValue)],
}

/// An ordered list of bindings an event runs.
pub type EventActions = Vec<EventBinding>;

/// Fire every binding of a list, in order. Failures are logged once per
/// binding and never stop the rest.
pub fn fire_all(list: &[EventBinding], ctx: &mut ComponentCtx, src: &FireSource) {
    for binding in list {
        binding.fire(ctx, src);
    }
}

fn is_object(component: &str) -> bool {
    component.is_empty() || component == OBJECT_COMPONENT
}

impl EventBinding {
    /// The component instance this binding addresses.
    pub fn sel(&self) -> InstanceSel<'_> {
        InstanceSel { index: self.instance as usize, label: &self.instance_label }
    }

    /// Fire, logging the first failure (with the owning object's name).
    pub fn fire(&self, ctx: &mut ComponentCtx, src: &FireSource) {
        if let Err(e) = self.try_fire(ctx, src)
            && !self.state.warned.replace(true)
        {
            let owner = ctx.object_names.get(ctx.self_index).map(String::as_str).unwrap_or("?");
            tracing::warn!("action binding on {owner:?} ({}.{}): {e}", self.component, self.action);
        }
    }

    /// Resolve and queue the call; the error says what didn't resolve.
    pub fn try_fire(&self, ctx: &mut ComponentCtx, src: &FireSource) -> Result<(), String> {
        let target_id = self.target.id().unwrap_or(src.source);
        let index = ctx.index_of(target_id).ok_or("target object not found")?;
        if is_object(&self.component) {
            let desc = OBJECT_ACTIONS
                .iter()
                .find(|a| a.name == self.action)
                .ok_or_else(|| format!("no object action '{}'", self.action))?;
            let args = self.eval_args(ctx, src, desc)?;
            return run_object_action(ctx, src, target_id, desc.name, &args);
        }
        let lookup = ctx.lookup;
        let (slot, comp) = lookup
            .component_named(index, &self.component, self.state.component_slot.get() as usize, self.sel())
            .ok_or_else(|| format!("target has no {} component", self.component))?;
        self.state.component_slot.set(slot as u32);
        let actions = comp.actions();
        let cached = self.state.action_index.get() as usize;
        let ai = match actions.get(cached) {
            Some(a) if a.name == self.action => cached,
            _ => actions
                .iter()
                .position(|a| a.name == self.action)
                .ok_or_else(|| crate::actions::unknown_action(&self.component, &self.action))?,
        };
        self.state.action_index.set(ai as u32);
        let desc = &actions[ai];
        let args = self.eval_args(ctx, src, desc)?;
        ctx.commands.push(ComponentCommand::InvokeAction {
            id: target_id,
            slot: slot as u32,
            component: comp.type_name(),
            action: desc.name,
            args,
            instigator: src.instigator,
        });
        Ok(())
    }

    fn eval_args(&self, ctx: &ComponentCtx, src: &FireSource, desc: &ActionDesc) -> Result<Vec<ArgValue>, String> {
        let mut out = Vec::with_capacity(desc.params.len());
        for (i, p) in desc.params.iter().enumerate() {
            let value = match self.args.get(i) {
                Some(source) => resolve_arg(source, ctx, src),
                None => Ok(p.default.value()),
            }
            .map_err(|e| format!("parameter '{}': {e}", p.name))?;
            out.push(value.coerce(p.ty).map_err(|e| format!("parameter '{}': {e}", p.name))?);
        }
        Ok(out)
    }
}

/// Read a property of an object: a real component's, or the object's own.
pub(crate) fn read_property(ctx: &ComponentCtx, object: ObjectId, component: &str, property: &str) -> Result<ArgValue, String> {
    let index = ctx.index_of(object).ok_or("object not found")?;
    if is_object(component) {
        if property.is_empty() {
            return Ok(ArgValue::Object(object.into()));
        }
        let position = ctx.object_position(index).ok_or_else(|| "object has no position".to_string())?;
        return object_property(object, ctx.object_names.get(index).map_or("", String::as_str), ctx.active.get(index).copied().unwrap_or(false), position, property);
    }
    let (_, comp) = ctx
        .lookup
        .component_named(index, component, 0, InstanceSel::FIRST)
        .ok_or_else(|| format!("object has no {component} component"))?;
    comp.read_property(property).ok_or_else(|| format!("{component} has no property '{property}'"))
}

fn resolve_arg(source: &ArgSource, ctx: &ComponentCtx, src: &FireSource) -> Result<ArgValue, String> {
    match source {
        ArgSource::Const(v) => Ok(v.clone()),
        ArgSource::Property { object, component, property } => {
            read_property(ctx, object.id().unwrap_or(src.source), component, property)
        }
        ArgSource::Instigator { component, property } => {
            let who = src.instigator.ok_or("this event has no instigator")?;
            read_property(ctx, who, component, property)
        }
        ArgSource::Payload(field) => src
            .payload
            .iter()
            .find(|(n, _)| n == field)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("the event has no field '{field}'")),
    }
}

fn run_object_action(
    ctx: &mut ComponentCtx,
    src: &FireSource,
    target: ObjectId,
    action: &str,
    args: &[ArgValue],
) -> Result<(), String> {
    match action {
        "SetActive" => ctx.set_active(target, args[0].as_bool()),
        "ToggleActive" => {
            let on = ctx.is_active(target);
            ctx.set_active(target, !on);
        }
        "PlaySound" => {
            let clip = args[0].as_str();
            if !clip.is_empty() {
                let at = ctx.transform_of(target.into()).map(|t| t.translation).unwrap_or_default();
                ctx.play_sound_at(clip.to_owned(), at, args[1].as_f32());
            }
        }
        "EmitEvent" => ctx.emit(ActionEvent { name: args[0].as_str().to_owned(), source: src.source, instigator: src.instigator }),
        "LoadScene" => {
            if !args[0].as_str().is_empty() {
                ctx.load_scene(args[0].as_str().to_owned());
            }
        }
        _ => return Err(format!("no object action '{action}'")),
    }
    Ok(())
}

/// Apply a queued [`ComponentCommand::InvokeAction`] to the target object's
/// components (the engine calls this). `slot` is the cached position from
/// firing time; the name is checked, so a stale slot falls back to a search.
/// Follow-up commands the action issues are pushed to `out`.
#[allow(clippy::too_many_arguments)]
pub fn invoke_on(
    components: &mut [Box<dyn Component>],
    slot: u32,
    component: &str,
    action: &str,
    args: &[ArgValue],
    self_id: ObjectId,
    instigator: Option<ObjectId>,
    out: &mut Vec<ComponentCommand>,
) -> Result<(), String> {
    invoke_on_with(components, slot, component, action, args, self_id, instigator, out, None)
}

/// [`invoke_on`] with a world view: the action can read other objects'
/// properties through `reader` and its own object's other components (a Wasm
/// Script's `host.get_property` inside `zest_invoke`).
#[allow(clippy::too_many_arguments)]
pub fn invoke_on_with(
    components: &mut [Box<dyn Component>],
    slot: u32,
    component: &str,
    action: &str,
    args: &[ArgValue],
    self_id: ObjectId,
    instigator: Option<ObjectId>,
    out: &mut Vec<ComponentCommand>,
    reader: Option<&dyn PropertyReader>,
) -> Result<(), String> {
    let i = match components.get(slot as usize) {
        Some(c) if c.type_name() == component => slot as usize,
        _ => components
            .iter()
            .position(|c| c.type_name() == component)
            .ok_or_else(|| format!("target has no {component} component"))?,
    };
    let (before, rest) = components.split_at_mut(i);
    let (target, after) = rest.split_first_mut().expect("index checked above");
    let siblings = SiblingReader { self_id, before, after, outer: reader };
    target.invoke_action(&mut ActionCtx { self_id, instigator, commands: out, reader: Some(&siblings) }, action, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnyBox, ComponentLookup, Health, InputState, InteractMode, Interactable, LightComponent, Localization, NetView,
        SaveGame, ScreenInfo, TriggerEvents, TypedComponent, UiButton, UiText,
    };
    use glam::{Mat4, Quat, Vec3};
    use std::any::TypeId;
    use std::collections::HashMap;

    /// A tiny scene: objects with components, enough to build a `ComponentCtx`
    /// and apply the invoke commands the way the engine does.
    struct World {
        ids: Vec<ObjectId>,
        names: Vec<String>,
        objects: Vec<Vec<Box<dyn Component>>>,
        id_index: HashMap<ObjectId, usize>,
        active: Vec<bool>,
        world: Vec<Mat4>,
        commands: Vec<ComponentCommand>,
        t: Vec3,
        r: Quat,
        s: Vec3,
        input: InputState,
        net: NetView,
        screen: ScreenInfo,
        save: SaveGame,
        strings: Localization,
        events: Vec<AnyBox>,
    }

    struct Lookup<'a>(&'a [Vec<Box<dyn Component>>]);

    impl ComponentLookup for Lookup<'_> {
        fn component(&self, object: usize, type_id: TypeId) -> Option<&dyn Component> {
            self.0.get(object)?.iter().map(|c| c.as_ref()).find(|c| c.as_any().type_id() == type_id)
        }
        fn component_named(&self, object: usize, name: &str, hint: usize, sel: InstanceSel) -> Option<(usize, &dyn Component)> {
            let list = self.0.get(object)?;
            if sel.is_first()
                && let Some(c) = list.get(hint).filter(|c| c.type_name() == name)
            {
                return Some((hint, c.as_ref()));
            }
            pick_instance(list.iter().enumerate().map(|(s, c)| (s, c.as_ref())), name, sel)
        }
    }

    impl World {
        fn new(names: &[&str]) -> Self {
            let ids: Vec<ObjectId> = names.iter().map(|_| ObjectId::new()).collect();
            Self {
                id_index: ids.iter().enumerate().map(|(i, id)| (*id, i)).collect(),
                names: names.iter().map(|n| n.to_string()).collect(),
                objects: names.iter().map(|_| Vec::new()).collect(),
                active: vec![true; names.len()],
                world: vec![Mat4::IDENTITY; names.len()],
                ids,
                commands: Vec::new(),
                t: Vec3::ZERO,
                r: Quat::IDENTITY,
                s: Vec3::ONE,
                input: InputState::default(),
                net: NetView::default(),
                screen: ScreenInfo::default(),
                save: SaveGame::new(),
                strings: Localization::default(),
                events: Vec::new(),
            }
        }

        fn add(&mut self, object: usize, c: impl Component) {
            self.objects[object].push(Box::new(c));
        }

        /// Run `f` as the component at `object` would: with a context for it.
        fn with_ctx<R>(&mut self, object: usize, f: impl FnOnce(&mut ComponentCtx) -> R) -> R {
            let lookup = Lookup(&self.objects);
            let mut ctx = ComponentCtx {
                dt: 0.016,
                unscaled_dt: 0.016,
                time: 1.0,
                water_time: 0.0,
                translation: &mut self.t,
                rotation: &mut self.r,
                scale: &mut self.s,
                commands: &mut self.commands,
                world_transforms: &self.world,
                object_names: &self.names,
                object_ids: &self.ids,
                self_index: object,
                parent_world: None,
                input: &self.input,
                net: &self.net,
                spawn_points: &[],
                id_index: &self.id_index,
                active: &self.active,
                lookup: &lookup,
                physics: None,
                nav: None,
                materials: None,
                events: &self.events,
                time_scale: 1.0,
                screen: &self.screen,
                save: &self.save,
                strings: &self.strings,
            };
            f(&mut ctx)
        }

        /// Take the queued commands and apply the invoke ones, as the engine's
        /// `apply` does. Returns the errors and the commands left over.
        fn apply(&mut self) -> (Vec<String>, Vec<ComponentCommand>) {
            let mut errors = Vec::new();
            let mut rest = Vec::new();
            for c in std::mem::take(&mut self.commands) {
                match c {
                    ComponentCommand::InvokeAction { id, slot, component, action, args, instigator } => {
                        let i = self.id_index[&id];
                        let mut follow = Vec::new();
                        if let Err(e) = invoke_on(&mut self.objects[i], slot, component, action, &args, id, instigator, &mut follow) {
                            errors.push(e);
                        }
                        rest.extend(follow);
                    }
                    other => rest.push(other),
                }
            }
            (errors, rest)
        }

        fn get<T: Component>(&self, object: usize) -> &T {
            self.objects[object].iter().find_map(|c| c.as_any().downcast_ref::<T>()).unwrap()
        }
    }

    fn bind(target: ObjectId, component: &str, action: &str, args: Vec<ArgSource>) -> EventBinding {
        EventBinding { target: target.into(), component: component.into(), action: action.into(), args, ..Default::default() }
    }

    fn fire(w: &mut World, from: usize, b: &EventBinding, instigator: Option<ObjectId>, payload: &[(&'static str, ArgValue)]) {
        let src = FireSource { source: w.ids[from], instigator, payload };
        w.with_ctx(from, |ctx| b.fire(ctx, &src));
    }

    fn health(hp: f32) -> Health {
        Health { current: hp, max: 100.0, ..Default::default() }
    }

    fn update(w: &mut World, object: usize, i: &mut Interactable) {
        w.with_ctx(object, |ctx| TypedComponent::update(i, ctx));
    }

    #[test]
    fn calls_an_action_on_another_objects_component() {
        let mut w = World::new(&["switch", "target"]);
        w.add(1, health(80.0));
        let b = bind(w.ids[1], "Health", "Damage", vec![ArgSource::Const(ArgValue::Float(30.0))]);
        fire(&mut w, 0, &b, None, &[]);
        let (errors, _) = w.apply();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(w.get::<Health>(1).current, 50.0);
        // Missing arguments fall back to the declared default (10).
        let b = bind(w.ids[1], "Health", "Damage", vec![]);
        fire(&mut w, 0, &b, None, &[]);
        w.apply();
        assert_eq!(w.get::<Health>(1).current, 40.0);
    }

    #[test]
    fn argument_sources_read_properties_instigator_and_payload() {
        let mut w = World::new(&["switch", "lamp", "player", "tank"]);
        w.add(1, LightComponent { intensity: 1.0, ..Default::default() });
        w.add(2, health(25.0));
        w.add(3, health(60.0));
        // Property of another component on another object.
        let b = bind(
            w.ids[1],
            "Light",
            "SetIntensity",
            vec![ArgSource::Property { object: w.ids[3].into(), component: "Health".into(), property: "current".into() }],
        );
        fire(&mut w, 0, &b, None, &[]);
        w.apply();
        assert_eq!(w.get::<LightComponent>(1).intensity, 60.0);
        // Property of the instigator.
        let b = bind(
            w.ids[1],
            "Light",
            "SetIntensity",
            vec![ArgSource::Instigator { component: "Health".into(), property: "current".into() }],
        );
        let player = w.ids[2];
        fire(&mut w, 0, &b, Some(player), &[]);
        w.apply();
        assert_eq!(w.get::<LightComponent>(1).intensity, 25.0);
        // Event payload (an Int converts to the Float parameter).
        let b = bind(w.ids[1], "Light", "SetIntensity", vec![ArgSource::Payload("uses".into())]);
        fire(&mut w, 0, &b, None, &[("uses", ArgValue::Int(7))]);
        w.apply();
        assert_eq!(w.get::<LightComponent>(1).intensity, 7.0);
        // The instigator itself is an Object value; the object's own position is a Vec3.
        let b = bind(w.ids[1], "Object", "EmitEvent", vec![ArgSource::Property { object: w.ids[2].into(), component: String::new(), property: "name".into() }]);
        fire(&mut w, 0, &b, None, &[]);
        let (errors, rest) = w.apply();
        assert!(errors.is_empty());
        assert!(rest.iter().any(|c| matches!(c, ComponentCommand::Emit(_))));
    }

    #[test]
    fn bad_bindings_report_and_never_panic() {
        let mut w = World::new(&["switch", "target"]);
        w.add(1, health(80.0));
        let id = w.ids[1];
        let cases = [
            bind(ObjectId::new(), "Health", "Damage", vec![]),                                     // missing object
            bind(id, "Light", "SetIntensity", vec![]),                                             // missing component
            bind(id, "Health", "Explode", vec![]),                                                 // missing action
            bind(id, "Health", "Damage", vec![ArgSource::Const(ArgValue::String("lots".into()))]), // type mismatch
            bind(id, "Health", "Damage", vec![ArgSource::Payload("nope".into())]),                 // missing payload field
            bind(id, "Health", "Damage", vec![ArgSource::Instigator { component: String::new(), property: String::new() }]), // no instigator
            bind(id, "Health", "Damage", vec![ArgSource::Property { object: id.into(), component: "Health".into(), property: "nope".into() }]),
            bind(id, "Object", "Nope", vec![]),
        ];
        for b in &cases {
            let src = FireSource { source: w.ids[0], instigator: None, payload: &[] };
            assert!(w.with_ctx(0, |ctx| b.try_fire(ctx, &src)).is_err(), "{b:?}");
            fire(&mut w, 0, b, None, &[]); // logs once, returns normally
            assert!(b.state.warned.get());
        }
        assert!(w.commands.is_empty(), "failed bindings queue nothing");
        assert_eq!(w.get::<Health>(1).current, 80.0);
        // A stale queued call (component removed meanwhile) is an error, not a panic.
        let mut out = Vec::new();
        let err = invoke_on(&mut w.objects[0], 0, "Health", "Damage", &[ArgValue::Float(1.0)], id, None, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn resolution_is_cached_and_survives_a_reorder() {
        let mut w = World::new(&["switch", "target"]);
        w.add(1, LightComponent::default());
        w.add(1, health(80.0));
        let b = bind(w.ids[1], "Health", "Damage", vec![ArgSource::Const(ArgValue::Float(5.0))]);
        fire(&mut w, 0, &b, None, &[]);
        assert_eq!(b.state.component_slot.get(), 1);
        assert_eq!(b.state.action_index.get(), 0);
        w.apply();
        // Reorder the components: the stale slot is rechecked by name.
        w.objects[1].reverse();
        fire(&mut w, 0, &b, None, &[]);
        assert_eq!(b.state.component_slot.get(), 0, "re-resolved");
        let (errors, _) = w.apply();
        assert!(errors.is_empty());
        assert_eq!(w.get::<Health>(1).current, 70.0);
        // A queued call whose slot went stale between firing and applying also finds it.
        fire(&mut w, 0, &b, None, &[]);
        w.objects[1].reverse();
        let (errors, _) = w.apply();
        assert!(errors.is_empty());
        assert_eq!(w.get::<Health>(1).current, 65.0);
    }

    #[test]
    fn builtin_object_actions() {
        let mut w = World::new(&["switch", "door"]);
        let id = w.ids[1];
        fire(&mut w, 0, &bind(id, "Object", "SetActive", vec![ArgSource::Const(ArgValue::Bool(false))]), None, &[]);
        fire(&mut w, 0, &bind(id, "", "ToggleActive", vec![]), None, &[]);
        fire(&mut w, 0, &bind(id, "Object", "LoadScene", vec![ArgSource::Const(ArgValue::String("a.scene".into()))]), None, &[]);
        let (_, rest) = w.apply();
        assert!(matches!(rest[0], ComponentCommand::SetActive(i, false) if i == id));
        assert!(matches!(rest[1], ComponentCommand::SetActive(i, false) if i == id), "toggles the current (active) state");
        assert!(matches!(&rest[2], ComponentCommand::LoadScene(p) if p == "a.scene"));
    }

    #[test]
    fn unset_target_means_the_source_object() {
        let mut w = World::new(&["lamp"]);
        w.add(0, LightComponent::default());
        let mut b = bind(w.ids[0], "Light", "SetIntensity", vec![ArgSource::Const(ArgValue::Float(3.0))]);
        b.target = ObjectRef::NONE;
        fire(&mut w, 0, &b, None, &[]);
        w.apply();
        assert_eq!(w.get::<LightComponent>(0).intensity, 3.0);
    }

    #[test]
    fn interactable_trigger_mode_runs_on_interact_with_the_user_as_instigator() {
        let mut w = World::new(&["button", "target", "player"]);
        w.add(1, health(100.0));
        w.add(2, health(40.0));
        let mut i = Interactable {
            on_interact: vec![bind(
                w.ids[1],
                "Health",
                "Heal",
                vec![ArgSource::Instigator { component: "Health".into(), property: "current".into() }],
            )],
            ..Default::default()
        };
        w.objects[1][0] = Box::new(health(10.0));
        i.request(w.ids[2]);
        update(&mut w, 0, &mut i);
        w.apply();
        assert_eq!(w.get::<Health>(1).current, 50.0);
        assert_eq!(i.uses, 1);
        assert!(!i.on, "Trigger mode keeps no state");
    }

    #[test]
    fn interactable_toggle_flips_state_and_runs_on_and_off() {
        let mut w = World::new(&["lever", "lamp", "player"]);
        let set = |id: ObjectId, v: f32| bind(id, "Object", "SetActive", vec![ArgSource::Const(ArgValue::Bool(v > 0.0))]);
        let mut i = Interactable {
            mode: InteractMode::Toggle,
            cooldown: 0.0,
            on_on: vec![set(w.ids[1], 1.0)],
            on_off: vec![set(w.ids[1], 0.0)],
            ..Default::default()
        };
        let lamp = w.ids[1];
        for expect in [true, false, true] {
            i.request(w.ids[2]);
            update(&mut w, 0, &mut i);
            assert_eq!(i.on, expect);
            let (_, rest) = w.apply();
            assert!(matches!(rest[0], ComponentCommand::SetActive(id, on) if id == lamp && on == expect));
        }
        // The state is saved.
        let back: Interactable = ron::from_str(&ron::to_string(&i).unwrap()).unwrap();
        assert!(back.on);
        assert_eq!(back.mode, InteractMode::Toggle);
        assert_eq!(back.on_on.len(), 1);
    }

    #[test]
    fn interactable_hold_fires_press_then_release() {
        let mut w = World::new(&["plate", "door", "player"]);
        let door = w.ids[1];
        let mut i = Interactable {
            mode: InteractMode::Hold,
            cooldown: 0.0,
            on_press: vec![bind(door, "Object", "SetActive", vec![ArgSource::Const(ArgValue::Bool(false))])],
            on_release: vec![bind(door, "Object", "SetActive", vec![ArgSource::Const(ArgValue::Bool(true))])],
            ..Default::default()
        };
        // Releasing without a press does nothing.
        i.request_release(w.ids[2]);
        update(&mut w, 0, &mut i);
        assert!(w.apply().1.is_empty());
        i.request(w.ids[2]);
        update(&mut w, 0, &mut i);
        assert!(i.held);
        assert!(matches!(w.apply().1[0], ComponentCommand::SetActive(_, false)));
        i.request_release(w.ids[2]);
        update(&mut w, 0, &mut i);
        assert!(!i.held);
        assert!(matches!(w.apply().1[0], ComponentCommand::SetActive(_, true)));
    }

    #[test]
    fn bindings_pick_one_of_several_components_of_a_type() {
        let mut w = World::new(&["tank", "switch"]);
        w.add(0, health(10.0));
        w.add(0, health(20.0));
        let tank = w.ids[0];
        let mut b = bind(tank, "Health", "Heal", vec![ArgSource::Const(ArgValue::Float(5.0))]);
        // The first instance by default (and for files from before `instance`).
        fire(&mut w, 1, &b, None, &[]);
        w.apply();
        let healths = |w: &World| -> Vec<f32> {
            w.objects[0].iter().filter_map(|c| c.as_any().downcast_ref::<Health>()).map(|h| h.current).collect()
        };
        assert_eq!(healths(&w), [15.0, 20.0]);
        b.instance = 1;
        fire(&mut w, 1, &b, None, &[]);
        w.apply();
        assert_eq!(healths(&w), [15.0, 25.0]);
        b.instance = 5;
        let err = w.with_ctx(1, |ctx| b.try_fire(ctx, &FireSource { source: tank, instigator: None, payload: &[] }));
        assert!(err.is_err(), "no sixth Health");
        let old: EventBinding = ron::from_str(r#"(component: "Health", action: "Heal")"#).unwrap();
        assert_eq!((old.instance, old.instance_label.as_str()), (0, ""));
    }

    #[test]
    fn instance_labels_follow_the_module_when_components_move() {
        use crate::WasmScript;
        let list: Vec<Box<dyn Component>> = vec![
            Box::new(WasmScript::new("scripts/door.wasm")),
            Box::new(health(1.0)),
            Box::new(WasmScript::new("scripts/lamp.wasm")),
        ];
        let all = || list.iter().enumerate().map(|(s, c)| (s, c.as_ref()));
        let pick = |index, label| pick_instance(all(), "Wasm Script", InstanceSel { index, label }).map(|(s, _)| s);
        assert_eq!(pick(0, ""), Some(0));
        assert_eq!(pick(1, ""), Some(2));
        assert_eq!(pick(2, ""), None);
        // The label wins over a stale index, and falls back to it when absent.
        assert_eq!(pick(0, "lamp"), Some(2));
        assert_eq!(pick(1, "door"), Some(0));
        assert_eq!(pick(1, "gone"), Some(2));
        assert_eq!(list[0].instance_label().as_deref(), Some("door"));
        assert_eq!(Component::instance_label(&WasmScript::new("@abc")), None);
    }

    #[test]
    fn click_to_move_releases_a_held_interactable_when_the_press_ends() {
        use crate::kit::ClickToMove;
        let mut w = World::new(&["player", "plate"]);
        w.add(1, Interactable { mode: InteractMode::Hold, ..Default::default() });
        let plate = w.ids[1];
        let mut ctm = ClickToMove { pending_interact: Some(plate), ..Default::default() };
        let step = |w: &mut World, ctm: &mut ClickToMove| {
            w.commands.clear();
            w.with_ctx(0, |ctx| TypedComponent::update(ctm, ctx));
            w.commands.len()
        };
        // Arriving in range uses the plate and remembers it is held.
        assert!(step(&mut w, &mut ctm) > 0);
        assert_eq!(ctm.held, Some(plate));
        // The button is up: the next update asks for the release, once.
        assert!(step(&mut w, &mut ctm) > 0);
        assert_eq!(ctm.held, None);
        assert_eq!(step(&mut w, &mut ctm), 0);
        // Still pressed: kept; a Trigger plate is never marked held.
        ctm.held = Some(plate);
        assert_eq!(ctm.take_release(true), None);
        assert_eq!(ctm.held, Some(plate));
        w.objects[1][0] = Box::new(Interactable::default());
        ctm.pending_interact = Some(plate);
        ctm.held = None;
        step(&mut w, &mut ctm);
        assert_eq!(ctm.held, None);
    }

    #[test]
    fn interactable_cooldown_and_single_use_still_apply() {
        let mut w = World::new(&["chest", "player"]);
        let emit = bind(w.ids[0], "Object", "EmitEvent", vec![ArgSource::Const(ArgValue::String("opened".into()))]);
        let mut i = Interactable { cooldown: 1.0, on_interact: vec![emit.clone()], ..Default::default() };
        i.request(w.ids[1]);
        update(&mut w, 0, &mut i);
        i.request(w.ids[1]);
        update(&mut w, 0, &mut i); // inside the cooldown
        assert_eq!(i.uses, 1);
        i.cooldown_left = 0.0;
        i.request(w.ids[1]);
        update(&mut w, 0, &mut i);
        assert_eq!(i.uses, 2);
        let mut once = Interactable { single_use: true, cooldown: 0.0, on_interact: vec![emit], ..Default::default() };
        for _ in 0..3 {
            once.request(w.ids[1]);
            update(&mut w, 0, &mut once);
        }
        assert_eq!(once.uses, 1);
        let emits = w.apply().1.iter().filter(|c| matches!(c, ComponentCommand::Emit(_))).count();
        // Two uses of the first, one of the single-use, plus an InteractEvent each.
        assert_eq!(emits, 3 * 2);
    }

    #[test]
    fn interactable_exposes_its_state_and_can_be_driven_by_a_binding() {
        let mut w = World::new(&["button", "lever", "player"]);
        let mut lever = Interactable { mode: InteractMode::Toggle, cooldown: 0.0, ..Default::default() };
        lever.on = true;
        w.add(1, lever);
        // Call "Interact" on the lever as the player: it requests with the player as the user.
        let b = bind(w.ids[1], "Interactable", "Interact", vec![]);
        let player = w.ids[2];
        fire(&mut w, 0, &b, Some(player), &[]);
        w.apply();
        assert_eq!(w.get::<Interactable>(1).requested, Some(w.ids[2]));
        assert_eq!(Component::read_property(w.get::<Interactable>(1), "on"), Some(ArgValue::Bool(true)));
    }

    #[test]
    fn trigger_events_fire_enter_and_exit_with_the_visitor() {
        let mut w = World::new(&["zone", "light", "player", "crate"]);
        w.add(1, LightComponent::default());
        w.add(2, health(100.0));
        let light = w.ids[1];
        let set = |v: f32| bind(light, "Light", "SetIntensity", vec![ArgSource::Const(ArgValue::Float(v))]);
        let mut t = TriggerEvents {
            require_component: "Health".into(),
            on_enter: vec![set(5.0)],
            on_exit: vec![set(0.5)],
            ..Default::default()
        };
        let (player, krate) = (w.ids[2], w.ids[3]);
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_enter(&mut t, ctx, krate)); // no Health: ignored
        assert!(w.commands.is_empty());
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_enter(&mut t, ctx, player));
        w.apply();
        assert_eq!(w.get::<LightComponent>(1).intensity, 5.0);
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_exit(&mut t, ctx, player));
        w.apply();
        assert_eq!(w.get::<LightComponent>(1).intensity, 0.5);
        // `once` and first/last only.
        let mut once = TriggerEvents { once: true, on_enter: vec![set(9.0)], ..Default::default() };
        for _ in 0..2 {
            w.with_ctx(0, |ctx| TypedComponent::on_trigger_enter(&mut once, ctx, player));
        }
        assert_eq!(w.commands.len(), 1);
        w.apply();
        let mut edge = TriggerEvents { first_and_last_only: true, on_enter: vec![set(1.0)], on_exit: vec![set(2.0)], ..Default::default() };
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_enter(&mut edge, ctx, player));
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_enter(&mut edge, ctx, krate));
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_exit(&mut edge, ctx, player));
        assert_eq!(w.commands.len(), 1, "only the first enter so far");
        w.with_ctx(0, |ctx| TypedComponent::on_trigger_exit(&mut edge, ctx, krate));
        assert_eq!(w.commands.len(), 2, "and the last exit");
    }

    #[test]
    fn ui_text_set_text_and_button_bindings() {
        let mut w = World::new(&["button", "label"]);
        w.add(1, UiText::default());
        let mut button = UiButton {
            on_click_actions: vec![bind(w.ids[1], "UI Text", "SetText", vec![ArgSource::Const(ArgValue::String("Clicked".into()))])],
            ..Default::default()
        };
        w.screen.ui_clicks.push(w.ids[0]);
        w.with_ctx(0, |ctx| TypedComponent::update(&mut button, ctx));
        w.apply();
        assert_eq!(w.get::<UiText>(1).text, "Clicked");
        assert_eq!(Component::read_property(w.get::<UiText>(1), "text"), Some(ArgValue::String("Clicked".into())));
    }

    #[test]
    fn old_files_load_unchanged() {
        // An Interactable saved before event bindings existed.
        let old = r#"(prompt: "Open", range: 2.0, toggle: (Some("00000000-0000-0000-0000-000000000000")), rebake_navmesh: true, sound: "", cooldown: 0.5, single_use: false)"#;
        let i: Interactable = ron::from_str(old).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(i.prompt, "Open");
        assert!(i.rebake_navmesh);
        assert_eq!(i.mode, InteractMode::Trigger);
        assert!(i.on_interact.is_empty() && !i.on);
        let i: Interactable = ron::from_str("()").unwrap();
        assert_eq!(i.range, 1.8);
        // A UI Button saved with only the legacy action list.
        let b: UiButton = ron::from_str("(interactable: true, on_click: [Show(\"Menu\"), Quit])").unwrap();
        assert_eq!(b.on_click.len(), 2);
        assert!(b.on_click_actions.is_empty());
    }

    #[test]
    fn animation_event_bindings_match_by_name() {
        use crate::{AnimationEventBinding, AnimationEventBindings};
        let mut w = World::new(&["rig", "light"]);
        w.add(1, LightComponent::default());
        let mut a = AnimationEventBindings {
            events: vec![AnimationEventBinding {
                event: "Footstep".into(),
                actions: vec![bind(w.ids[1], "Light", "SetIntensity", vec![ArgSource::Const(ArgValue::Float(2.0))])],
            }],
        };
        w.with_ctx(0, |ctx| TypedComponent::on_animation_event(&mut a, ctx, "Other"));
        assert!(w.commands.is_empty());
        w.with_ctx(0, |ctx| TypedComponent::on_animation_event(&mut a, ctx, "Footstep"));
        w.apply();
        assert_eq!(w.get::<LightComponent>(1).intensity, 2.0);
    }

    #[test]
    fn coercion_rules() {
        assert_eq!(ArgValue::Int(3).coerce(ActionType::Float), Ok(ArgValue::Float(3.0)));
        assert_eq!(ArgValue::Float(2.6).coerce(ActionType::Int), Ok(ArgValue::Int(3)));
        assert_eq!(ArgValue::Bool(true).coerce(ActionType::Float), Ok(ArgValue::Float(1.0)));
        assert_eq!(ArgValue::Float(0.0).coerce(ActionType::Bool), Ok(ArgValue::Bool(false)));
        assert_eq!(ArgValue::Int(7).coerce(ActionType::String), Ok(ArgValue::String("7".into())));
        assert!(ArgValue::String("x".into()).coerce(ActionType::Float).is_err());
        assert!(ArgValue::Vec3([0.0; 3]).coerce(ActionType::Object).is_err());
    }

    #[test]
    fn binding_round_trips_and_ignores_the_cache() {
        let b = EventBinding {
            target: ObjectId::new().into(),
            component: "Health".into(),
            action: "Damage".into(),
            args: vec![
                ArgSource::Const(ArgValue::Float(5.0)),
                ArgSource::Property { object: ObjectRef::NONE, component: "Health".into(), property: "current".into() },
                ArgSource::Instigator { component: String::new(), property: String::new() },
                ArgSource::Payload("uses".into()),
                ArgSource::Const(ArgValue::Vec3([1.0, 2.0, 3.0])),
            ],
            instance: 1,
            instance_label: "door".into(),
            state: BindingState::default(),
        };
        b.state.component_slot.set(4);
        let text = ron::to_string(&b).unwrap();
        let back: EventBinding = ron::from_str(&text).unwrap();
        assert_eq!(b, back);
        assert_eq!(back.state.component_slot.get(), 0, "the cache is not saved");
        let empty: EventBinding = ron::from_str("()").unwrap();
        assert_eq!(empty.component, OBJECT_COMPONENT);
    }

    #[test]
    fn declarations_are_consistent() {
        // Every declared action of the example components has typed params and
        // every declared property reads back.
        let samples: Vec<Box<dyn Component>> = vec![
            Box::new(Health::default()),
            Box::new(LightComponent::default()),
            Box::new(UiText::default()),
            Box::new(Interactable::default()),
        ];
        for c in &samples {
            for p in c.properties() {
                assert!(c.read_property(p.name).is_some_and(|v| v.ty() == p.ty), "{}.{}", c.type_name(), p.name);
            }
            assert!(!c.actions().is_empty());
        }
        let mut registry_count = 0;
        for name in crate::ComponentRegistry::with_builtins().names() {
            let _ = name;
            registry_count += 1;
        }
        assert!(registry_count > 50);
    }

    // ---------------------------------------------------------- WASM components

    #[test]
    fn interactable_calls_a_wasm_action_with_values_read_from_other_components() {
        use crate::wasm::fixtures::*;
        use crate::WasmScript;
        let dir = temp_dir("evt");
        let module = write(&dir, "counter.wasm", &build(counter_manifest(), &counter(1)));
        let mut w = World::new(&["switch", "box", "player"]);
        w.add(1, WasmScript::new(module));
        w.add(2, health(40.0));
        let mut i = Interactable {
            cooldown: 0.0,
            on_interact: vec![
                // A constant, then the player's health, through the same dropdown path as a Rust component.
                bind(w.ids[1], "Wasm Script", "Add", vec![ArgSource::Const(ArgValue::Int(4))]),
                bind(
                    w.ids[1],
                    "Wasm Script",
                    "Add",
                    vec![ArgSource::Instigator { component: "Health".into(), property: "current".into() }],
                ),
                bind(w.ids[1], "Wasm Script", "Fail", vec![]),
                bind(w.ids[1], "Wasm Script", "Nope", vec![]),
            ],
            ..Default::default()
        };
        i.request(w.ids[2]);
        update(&mut w, 0, &mut i);
        let (errors, _) = w.apply();
        assert_eq!(errors, vec!["nope".to_string()], "the guest's failure reaches the engine; the rest ran");
        let count = Component::read_property(w.get::<WasmScript>(1), "count");
        assert_eq!(count, Some(ArgValue::Int(44)), "4 + the player's 40 health");
        // The catalog sees the module's actions like any component's.
        let api = Component::actions(w.get::<WasmScript>(1));
        assert!(api.iter().any(|a| a.name == "Add" && a.params.len() == 1));
    }

    #[test]
    fn a_wasm_property_feeds_another_components_action() {
        use crate::wasm::fixtures::*;
        use crate::WasmScript;
        let dir = temp_dir("evt-prop");
        let module = write(&dir, "counter.wasm", &build(counter_manifest(), &counter(1)));
        let mut w = World::new(&["switch", "box", "lamp"]);
        let mut counter = WasmScript::new(module);
        w.with_ctx(1, |_| ());
        let mut commands = Vec::new();
        TypedComponent::invoke_action(
            &mut counter,
            &mut ActionCtx { self_id: w.ids[1], instigator: None, commands: &mut commands, reader: None },
            "Add",
            &[ArgValue::Int(75)],
        )
        .unwrap();
        w.add(1, counter);
        w.add(2, LightComponent::default());
        let b = bind(
            w.ids[2],
            "Light",
            "SetIntensity",
            vec![ArgSource::Property { object: w.ids[1].into(), component: "Wasm Script".into(), property: "count".into() }],
        );
        fire(&mut w, 0, &b, None, &[]);
        let (errors, _) = w.apply();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(w.get::<LightComponent>(2).intensity, 75.0);
    }

    #[test]
    fn wasm_update_gets_frame_time_reads_properties_and_queues_invokes() {
        use crate::wasm::fixtures::*;
        use crate::WasmScript;
        let dir = temp_dir("update");
        let mut w = World::new(&["actor"]);
        w.add(0, health(80.0));
        let mut ticker = WasmScript::new(write(&dir, "u.wasm", &build(UPDATER_MANIFEST, UPDATER)));
        w.with_ctx(0, |ctx| TypedComponent::update(&mut ticker, ctx));
        assert_eq!(ticker.float("dt"), Some(0.016));
        assert_eq!(ticker.float("time"), Some(1.0));
        // get_property reads Health.current; invoke queues Health.Damage(25) like a binding would.
        let mut driver = WasmScript::new(write(&dir, "d.wasm", &build(DRIVER_MANIFEST, DRIVER)));
        w.with_ctx(0, |ctx| TypedComponent::update(&mut driver, ctx));
        assert_eq!(driver.float("hp"), Some(80.0));
        let (errors, _) = w.apply();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(w.get::<Health>(0).current, 55.0);
    }

    #[test]
    fn wasm_invoke_reads_properties_through_siblings_and_the_reader() {
        use crate::wasm::fixtures::*;
        use crate::wasm::sandbox::{ALL_CAPS, Instance, SandboxLimits};
        use crate::WasmScript;
        let dir = temp_dir("invoke-read");
        let file = write(&dir, "l.wasm", &build(LOOKER_MANIFEST, LOOKER));
        // Through the object's own sibling component (a binding fired it).
        let id = ObjectId::new();
        let mut comps: Vec<Box<dyn Component>> = vec![Box::new(health(80.0)), Box::new(WasmScript::new(&file))];
        let mut out = Vec::new();
        invoke_on_with(&mut comps, 1, "Wasm Script", "Look", &[], id, None, &mut out, None).unwrap();
        let script = comps[1].as_any().downcast_ref::<WasmScript>().unwrap();
        assert_eq!(script.float("hp"), Some(80.0));
        assert!(script.float("rc").is_some_and(|rc| rc > 0.0));
        // A sibling that is not there reads as not found, not as a crash.
        let mut alone: Vec<Box<dyn Component>> = vec![Box::new(WasmScript::new(&file))];
        invoke_on_with(&mut alone, 0, "Wasm Script", "Look", &[], id, None, &mut out, None).unwrap();
        assert_eq!(alone[0].as_any().downcast_ref::<WasmScript>().unwrap().float("rc"), Some(-1.0));
        // Straight at the sandbox: a reader answers, and without one the
        // import reports "unavailable" (-2).
        struct Fixed;
        impl PropertyReader for Fixed {
            fn read(&self, _o: ObjectId, c: &str, p: &str) -> Result<ArgValue, String> {
                assert_eq!((c, p), ("Health", "current"));
                Ok(ArgValue::Float(33.0))
            }
        }
        let bytes = build(LOOKER_MANIFEST, LOOKER);
        let (mut inst, _) = Instance::load(&bytes, SandboxLimits::trusted(), ALL_CAPS, "t").unwrap();
        inst.call_invoke(0, &[], id, None, Some(&Fixed)).unwrap();
        assert_eq!(inst.float("hp"), Some(33.0));
        inst.call_invoke(0, &[], id, None, None).unwrap();
        assert_eq!(inst.float("rc"), Some(-2.0));
        // The pointer does not outlive the call: a later call without a reader sees none.
        inst.call_invoke(0, &[], id, None, Some(&Fixed)).unwrap();
        inst.call_invoke(0, &[], id, None, None).unwrap();
        assert_eq!(inst.float("rc"), Some(-2.0));
    }
}
