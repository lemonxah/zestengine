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

use std::sync::{Arc, atomic::{AtomicBool, AtomicU32, Ordering}};

use serde::{Deserialize, Serialize};

use crate::{Component, ComponentCommand, ObjectId, ObjectRef};

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
/// reach beyond the component go through the native command queue.
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
/// native systems and by the engine over the scene (when
/// a queued action is applied), so a component reads the same way in both.
pub trait PropertyReader {
    /// Read `property` of `component` on `object` (empty `component` = the
    /// object itself).
    fn read(&self, object: ObjectId, component: &str, property: &str) -> Result<ArgValue, String>;
}

/// A property of the object itself, for [`PropertyReader`] implementations
/// that read authoritative native metadata.
pub fn object_property(object: ObjectId, name: &str, active: bool, position: glam::Vec3, property: &str) -> Result<ArgValue, String> {
    match property {
        "" => Ok(ArgValue::Object(object.into())),
        "name" => Ok(ArgValue::String(name.to_owned())),
        "active" => Ok(ArgValue::Bool(active)),
        "position" => Ok(ArgValue::Vec3(position.to_array())),
        _ => Err(format!("object has no property '{property}'")),
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
    cache: Arc<BindingCache>,
}

#[derive(Default, Debug)]
struct BindingCache {
    component_slot: AtomicU32,
    action_index: AtomicU32,
    warned: AtomicBool,
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

/// What an event source hands to [`EventBinding::fire_native`].
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

/// Native event sources defer resolution until their checked queries finish.
/// The request retains per-binding cache/warning state without copying payloads.
pub fn queue_actions(list: &[EventBinding], src: &FireSource<'_>, out: &mut Vec<ComponentCommand>) {
    if list.is_empty() { return; }
    out.push(ComponentCommand::FireActions { bindings: list.iter().map(EventBinding::for_dispatch).collect(), source: src.source,
        instigator: src.instigator, payload: src.payload.to_vec() });
}

fn is_object(component: &str) -> bool {
    component.is_empty() || component == OBJECT_COMPONENT
}

impl EventBinding {
    /// A dispatch request shares this binding's runtime cache and warn-once
    /// state. Ordinary Clone still resets them for duplication and prefabs.
    fn for_dispatch(&self) -> Self {
        Self { target: self.target, component: self.component.clone(), action: self.action.clone(), args: self.args.clone(), instance: self.instance,
            instance_label: self.instance_label.clone(), state: BindingState { cache: self.state.cache.clone() } }
    }

    pub fn fire_native(&self, reader: &crate::ecs::NativeProperties<'_, '_>, src: &FireSource<'_>, out: &mut Vec<ComponentCommand>) {
        if let Err(error) = self.try_fire_native(reader, src, out)
            && !self.state.cache.warned.swap(true, Ordering::Relaxed)
        {
            tracing::warn!("action binding on {} ({}.{}): {error}", src.source, self.component, self.action);
        }
    }

    pub fn try_fire_native(&self, reader: &crate::ecs::NativeProperties<'_, '_>, src: &FireSource<'_>, out: &mut Vec<ComponentCommand>) -> Result<(), String> {
        let target = self.target.id().unwrap_or(src.source);
        let owner = reader.object(target).ok_or("target object not found")?;
        if is_object(&self.component) {
            let desc = OBJECT_ACTIONS.iter().find(|action| action.name == self.action)
                .ok_or_else(|| format!("no object action '{}'", self.action))?;
            let args = self.eval_native_args(reader, src, desc)?;
            match desc.name {
                "SetActive" => out.push(ComponentCommand::SetActive(target, args[0].as_bool())),
                "ToggleActive" => out.push(ComponentCommand::SetActive(target, !reader.world.current_active(owner))),
                "PlaySound" => {
                    if !args[0].as_str().is_empty() {
                        let position = reader.world.world_transform(owner).map(|matrix| matrix.w_axis.truncate());
                        out.push(ComponentCommand::PlaySound { clip: args[0].as_str().to_owned(), position, volume: args[1].as_f32() });
                    }
                }
                "EmitEvent" => out.push(ComponentCommand::Emit(crate::AnyBox(Box::new(ActionEvent { name: args[0].as_str().to_owned(), source: src.source, instigator: src.instigator })))),
                "LoadScene" => { if !args[0].as_str().is_empty() { out.push(ComponentCommand::LoadScene(args[0].as_str().to_owned())); } }
                _ => return Err(format!("no object action '{}'", desc.name)),
            }
            return Ok(());
        }
        let (position, slot) = reader.component(owner, &self.component, self.sel())
            .ok_or_else(|| format!("target has no {} component", self.component))?;
        self.state.cache.component_slot.store(position as u32, Ordering::Relaxed);
        let actions = reader.with_component(slot, |value| value.actions()).ok_or("target component is gone")?;
        let cached = self.state.cache.action_index.load(Ordering::Relaxed) as usize;
        let ai = match actions.get(cached) {
            Some(action) if action.name == self.action => cached,
            _ => actions.iter().position(|action| action.name == self.action).ok_or_else(|| unknown_action(&self.component, &self.action))?,
        };
        self.state.cache.action_index.store(ai as u32, Ordering::Relaxed);
        let desc = &actions[ai];
        let args = self.eval_native_args(reader, src, desc)?;
        out.push(ComponentCommand::InvokeAction { id: target, slot: position as u32, component: slot.name(), action: desc.name, args, instigator: src.instigator });
        Ok(())
    }

    fn eval_native_args(&self, reader: &dyn PropertyReader, src: &FireSource<'_>, desc: &ActionDesc) -> Result<Vec<ArgValue>, String> {
        desc.params.iter().enumerate().map(|(i, parameter)| {
            let value = match self.args.get(i) {
                None => Ok(parameter.default.value()),
                Some(ArgSource::Const(value)) => Ok(value.clone()),
                Some(ArgSource::Property { object, component, property }) => reader.read(object.id().unwrap_or(src.source), component, property),
                Some(ArgSource::Instigator { component, property }) => src.instigator.ok_or("this event has no instigator".into()).and_then(|object| reader.read(object, component, property)),
                Some(ArgSource::Payload(field)) => src.payload.iter().find(|(name, _)| name == field).map(|(_, value)| value.clone()).ok_or_else(|| format!("the event has no field '{field}'")),
            }.map_err(|error| format!("parameter '{}': {error}", parameter.name))?;
            value.coerce(parameter.ty).map_err(|error| format!("parameter '{}': {error}", parameter.name))
        }).collect()
    }

    /// The component instance this binding addresses.
    pub fn sel(&self) -> InstanceSel<'_> {
        InstanceSel { index: self.instance as usize, label: &self.instance_label }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Health, InputState, InteractMode, Interactable, LightComponent, NetView, ScreenInfo, TypedComponent, UiButton, UiText};
    use crate::ecs::{ComponentRef, ComponentSet, Entity, NativeTestFixture};

    /// Action tests use the same permanent typed storage and native dispatch
    /// as gameplay. Only object order and pending commands are fixture data.
    struct World {
        fixture: NativeTestFixture,
        ids: Vec<ObjectId>,
        owners: Vec<Entity>,
        commands: Vec<ComponentCommand>,
    }
    impl World {
        fn new(names: &[&str]) -> Self {
            let mut fixture = NativeTestFixture::new();
            let owners: Vec<_> = names.iter().map(|name| fixture.add(name, Vec::new())).collect();
            let ids = owners.iter().map(|owner| fixture.id(*owner)).collect();
            Self { fixture, ids, owners, commands: Vec::new() }
        }
        fn add(&mut self, object: usize, value: impl Component) { self.components_mut(object).push(Box::new(value)); }
        fn components(&self, object: usize) -> &ComponentSet { self.fixture.components(self.owners[object]) }
        fn components_mut(&mut self, object: usize) -> &mut ComponentSet { self.fixture.components_mut(self.owners[object]) }
        fn reverse(&mut self, object: usize) {
            let values = self.components_mut(object);
            for position in 0..values.len() / 2 { values.swap(position, values.len() - 1 - position); }
        }
        fn try_fire(&mut self, binding: &EventBinding, src: &FireSource<'_>) -> Result<(), String> {
            self.commands.extend(self.fixture.fire(binding, src.source, src.instigator, src.payload)?);
            Ok(())
        }
        fn apply(&mut self) -> (Vec<String>, Vec<ComponentCommand>) { self.fixture.apply_errors(std::mem::take(&mut self.commands)) }
        fn get<T: Component>(&self, object: usize) -> ComponentRef<'_, T> { self.components(object).get_typed::<T>().unwrap() }
    }

    fn bind(target: ObjectId, component: &str, action: &str, args: Vec<ArgSource>) -> EventBinding {
        EventBinding { target: target.into(), component: component.into(), action: action.into(), args, ..Default::default() }
    }

    fn fire(w: &mut World, from: usize, b: &EventBinding, instigator: Option<ObjectId>, payload: &[(&'static str, ArgValue)]) {
        let src = FireSource { source: w.ids[from], instigator, payload };
        w.commands.extend(w.fixture.fire_logged(b, src.source, src.instigator, src.payload));
    }

    fn health(hp: f32) -> Health {
        Health { current: hp, max: 100.0, ..Default::default() }
    }


    #[test]
    fn queued_bindings_share_dispatch_state_but_duplication_resets_it() {
        let binding = EventBinding::default();
        let mut out = Vec::new();
        queue_actions(std::slice::from_ref(&binding), &FireSource { source: ObjectId::new(), instigator: None, payload: &[] }, &mut out);
        let ComponentCommand::FireActions { bindings, .. } = &out[0] else { panic!("native binding request"); };
        bindings[0].state.cache.warned.store(true, Ordering::Relaxed);
        bindings[0].state.cache.component_slot.store(4, Ordering::Relaxed);
        assert!(binding.state.cache.warned.load(Ordering::Relaxed));
        assert_eq!(binding.state.cache.component_slot.load(Ordering::Relaxed), 4);
        let duplicate = binding.clone();
        assert!(!duplicate.state.cache.warned.load(Ordering::Relaxed));
        assert_eq!(duplicate.state.cache.component_slot.load(Ordering::Relaxed), 0);
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
            assert!(w.try_fire(b, &src).is_err(), "{b:?}");
            fire(&mut w, 0, b, None, &[]); // logs once, returns normally
            assert!(b.state.cache.warned.load(Ordering::Relaxed));
        }
        assert!(w.commands.is_empty(), "failed bindings queue nothing");
        assert_eq!(w.get::<Health>(1).current, 80.0);
        // A stale queued call (component removed meanwhile) is an error, not a panic.
        fire(&mut w, 0, &bind(id, "Health", "Damage", vec![ArgSource::Const(ArgValue::Float(1.0))]), None, &[]);
        w.components_mut(1).clear();
        assert_eq!(w.apply().0.len(), 1, "a target component removed after resolution produces a dispatch error");
    }

    #[test]
    fn resolution_is_cached_and_survives_a_reorder() {
        let mut w = World::new(&["switch", "target"]);
        w.add(1, LightComponent::default());
        w.add(1, health(80.0));
        let b = bind(w.ids[1], "Health", "Damage", vec![ArgSource::Const(ArgValue::Float(5.0))]);
        fire(&mut w, 0, &b, None, &[]);
        assert_eq!(b.state.cache.component_slot.load(Ordering::Relaxed), 1);
        assert_eq!(b.state.cache.action_index.load(Ordering::Relaxed), 0);
        w.apply();
        // Reorder the components: the stale slot is rechecked by name.
        w.reverse(1);
        fire(&mut w, 0, &b, None, &[]);
        assert_eq!(b.state.cache.component_slot.load(Ordering::Relaxed), 0, "re-resolved");
        let (errors, _) = w.apply();
        assert!(errors.is_empty());
        assert_eq!(w.get::<Health>(1).current, 70.0);
        // A queued call whose slot went stale between firing and applying also finds it.
        fire(&mut w, 0, &b, None, &[]);
        w.reverse(1);
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
            w.components(0).iter_typed::<Health>().map(|health| health.current).collect()
        };
        assert_eq!(healths(&w), [15.0, 20.0]);
        b.instance = 1;
        fire(&mut w, 1, &b, None, &[]);
        w.apply();
        assert_eq!(healths(&w), [15.0, 25.0]);
        b.instance = 5;
        let err = w.try_fire(&b, &FireSource { source: tank, instigator: None, payload: &[] });
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
    fn ui_text_set_text_and_button_bindings() {
        let mut fixture = crate::ecs::NativeTestFixture::new();
        let label = fixture.add("label", vec![Box::new(UiText::default())]);
        let button = fixture.add("button", vec![Box::new(UiButton {
            on_click_actions: vec![bind(fixture.id(label), "UI Text", "SetText", vec![ArgSource::Const(ArgValue::String("Clicked".into()))])],
            ..Default::default()
        })]);
        let mut screen = ScreenInfo::default();
        screen.ui_clicks.push(fixture.id(button));
        let commands = fixture.frame(&InputState::default(), &NetView::default(), &screen);
        fixture.apply_output(commands).unwrap();
        let text = fixture.scene.get::<UiText>(label).unwrap();
        assert_eq!(text.text, "Clicked");
        assert_eq!(Component::read_property(&*text, "text"), Some(ArgValue::String("Clicked".into())));
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
        b.state.cache.component_slot.store(4, Ordering::Relaxed);
        let text = ron::to_string(&b).unwrap();
        let back: EventBinding = ron::from_str(&text).unwrap();
        assert_eq!(b, back);
        assert_eq!(back.state.cache.component_slot.load(Ordering::Relaxed), 0, "the cache is not saved");
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
    fn a_wasm_property_feeds_another_components_action() {
        use crate::wasm::fixtures::*;
        use crate::WasmScript;
        let dir = temp_dir("evt-prop");
        let module = write(&dir, "counter.wasm", &build(counter_manifest(), &counter(1)));
        let mut w = World::new(&["switch", "box", "lamp"]);
        let mut counter = WasmScript::new(module);
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
        let mut fixture = crate::ecs::NativeTestFixture::new();
        let owner = fixture.add("actor", vec![
            Box::new(health(80.0)),
            Box::new(WasmScript::new(write(&dir, "u.wasm", &build(UPDATER_MANIFEST, UPDATER)))),
            Box::new(WasmScript::new(write(&dir, "d.wasm", &build(DRIVER_MANIFEST, DRIVER)))),
        ]);
        let commands = fixture.stage(crate::ecs::Stage::Gameplay, &InputState::default(), &NetView::default(), &ScreenInfo::default(), 0.016, 1.0);
        assert_eq!(fixture.scene.get::<WasmScript>(owner).unwrap().float("dt"), Some(0.016));
        assert_eq!(fixture.scene.get::<WasmScript>(owner).unwrap().float("time"), Some(1.0));
        // The second script read Health.current and queued Damage(25).
        assert_eq!(fixture.scene.get::<Health>(owner).unwrap().current, 80.0);
        let (errors, _) = fixture.apply_errors(commands);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(fixture.scene.get::<Health>(owner).unwrap().current, 55.0);
    }

    #[test]
    fn mod_scripts_only_invoke_granted_actions_on_the_authoritative_peer() {
        use crate::wasm::fixtures::*;
        use crate::WasmScript;
        for granted in [false, true] { for client in [false, true] {
            let dir = temp_dir("mod-permissions");
            crate::wasm::register_mod_root(&dir, if granted { vec!["Health.Damage".into()] } else { Vec::new() });
            let mut fixture = crate::ecs::NativeTestFixture::new();
            let owner = fixture.add("actor", vec![Box::new(health(80.0)), Box::new(WasmScript::new(write(&dir, "mod.wasm", &build(DRIVER_MANIFEST, DRIVER))))]);
            let net = NetView { connected: client, ..Default::default() };
            let commands = fixture.frame(&InputState::default(), &net, &ScreenInfo::default());
            let (errors, _) = fixture.apply_errors(commands);
            assert!(errors.is_empty(), "{errors:?}");
            assert_eq!(fixture.scene.get::<Health>(owner).unwrap().current, if granted && !client { 55.0 } else { 80.0 });
            crate::wasm::unregister_mod_root(&dir);
            let _ = std::fs::remove_dir_all(dir);
        } }
    }

    #[test]
    fn wasm_invoke_reads_properties_through_siblings_and_the_reader() {
        use crate::wasm::fixtures::*;
        use crate::wasm::sandbox::{ALL_CAPS, Instance, SandboxLimits};
        use crate::WasmScript;
        let dir = temp_dir("invoke-read");
        let file = write(&dir, "l.wasm", &build(LOOKER_MANIFEST, LOOKER));
        // Through the object's own sibling component (a binding fired it).
        let mut fixture = NativeTestFixture::new();
        let owner = fixture.add("With sibling", vec![Box::new(health(80.0)), Box::new(WasmScript::new(&file))]);
        let id = fixture.id(owner);
        let binding = bind(id, "Wasm Script", "Look", vec![]);
        let commands = fixture.fire(&binding, id, None, &[]).unwrap();
        fixture.apply_output(commands).unwrap();
        assert_eq!(fixture.scene.get::<WasmScript>(owner).unwrap().float("hp"), Some(80.0));
        assert!(fixture.scene.get::<WasmScript>(owner).unwrap().float("rc").is_some_and(|rc| rc > 0.0));
        let alone = fixture.add("Alone", vec![Box::new(WasmScript::new(&file))]);
        let alone_id = fixture.id(alone);
        let commands = fixture.fire(&bind(alone_id, "Wasm Script", "Look", vec![]), alone_id, None, &[]).unwrap();
        fixture.apply_output(commands).unwrap();
        assert_eq!(fixture.scene.get::<WasmScript>(alone).unwrap().float("rc"), Some(-1.0));
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
