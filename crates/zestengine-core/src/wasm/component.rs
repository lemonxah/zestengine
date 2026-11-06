//! The `Wasm Script` component: hosts one module and publishes its manifest
//! through `Component::actions` / `invoke_action` / `properties` /
//! `read_property`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use super::abi::Manifest;
use super::sandbox::{ALL_CAPS, CallError, Instance, ModuleInfo, SandboxLimits, UNTRUSTED_CAPS};
use super::{UntrustedPolicy, untrusted_policy};
use crate::actions::{
    ActionCtx, ActionDesc, ActionEvent, ActionType, ArgSource, ArgValue, EventBinding, FireSource, ParamDefault,
    ParamDesc, PropertyDesc,
};
use crate::{ComponentCtx, ObjectId, TypedComponent};

/// How often the module file is checked for changes (hot reload).
const RELOAD_POLL: Duration = Duration::from_millis(500);
/// Distinct manifests that may be leaked into `'static` tables per process.
const MAX_INTERNED: usize = 512;

/// The `'static` descriptor tables of one manifest.
#[derive(Clone, Copy)]
struct Tables {
    actions: &'static [ActionDesc],
    properties: &'static [PropertyDesc],
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn default_of(ty: ActionType, v: &Option<ArgValue>) -> ParamDefault {
    match v.clone().unwrap_or_else(|| ArgValue::zero(ty)) {
        ArgValue::Bool(b) => ParamDefault::Bool(b),
        ArgValue::Int(i) => ParamDefault::Int(i),
        ArgValue::Float(f) => ParamDefault::Float(f),
        ArgValue::String(s) => ParamDefault::Str(leak(&s)),
        ArgValue::Vec3(v) => ParamDefault::Vec3(v),
        ArgValue::Object(_) => ParamDefault::Object,
    }
}

/// Descriptor tables for a manifest, shared by every component loading the
/// same manifest text.
fn intern(text: &str, m: &Manifest) -> Result<Tables, String> {
    static TABLES: Mutex<Option<HashMap<String, Tables>>> = Mutex::new(None);
    let mut guard = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(t) = map.get(text) {
        return Ok(*t);
    }
    if map.len() >= MAX_INTERNED {
        return Err(format!("too many distinct module manifests loaded ({MAX_INTERNED}); restart to load more"));
    }
    let actions: Vec<ActionDesc> = m
        .actions
        .iter()
        .map(|a| {
            let params: Vec<ParamDesc> = a
                .params
                .iter()
                .map(|p| ParamDesc { name: leak(&p.name), ty: p.ty, default: default_of(p.ty, &p.default) })
                .collect();
            ActionDesc {
                name: leak(&a.name),
                label: leak(if a.label.is_empty() { &a.name } else { &a.label }),
                params: Box::leak(params.into_boxed_slice()),
                ugc_safe: a.ugc_safe,
            }
        })
        .collect();
    let properties: Vec<PropertyDesc> =
        m.properties.iter().map(|p| PropertyDesc { name: leak(&p.name), ty: p.ty }).collect();
    let t = Tables { actions: Box::leak(actions.into_boxed_slice()), properties: Box::leak(properties.into_boxed_slice()) };
    map.insert(text.to_owned(), t);
    Ok(t)
}

#[derive(Clone, Copy, PartialEq)]
struct Stamp {
    modified: Option<SystemTime>,
    len: u64,
}

fn stamp_of(path: &std::path::Path) -> Option<Stamp> {
    let md = std::fs::metadata(path).ok()?;
    Some(Stamp { modified: md.modified().ok(), len: md.len() })
}

/// Everything that is not saved: the loaded module and its bookkeeping.
#[derive(Default)]
struct Runtime {
    /// The `module` value the current state was loaded for.
    loaded_for: Option<String>,
    path: PathBuf,
    stamp: Option<Stamp>,
    checked: Option<Instant>,
    inst: Option<Instance>,
    tables: Option<Tables>,
    info: Option<ModuleInfo>,
    policy: Option<UntrustedPolicy>,
    /// A load error, or why a trapped module was disabled.
    error: Option<String>,
    /// Bumped on every (re)load; the editor uses it to refresh.
    generation: u32,
}

fn resolve(module: &str) -> PathBuf {
    crate::asset_ref::resolve_path(std::path::Path::new("."), module)
}

impl Runtime {
    fn load(&mut self, module: &str) {
        let generation = self.generation + 1;
        *self = Runtime { loaded_for: Some(module.to_owned()), checked: Some(Instant::now()), generation, ..Runtime::default() };
        if module.trim().is_empty() {
            return;
        }
        self.path = resolve(module);
        self.stamp = stamp_of(&self.path);
        let result = std::fs::read(&self.path).map_err(|e| format!("{}: {e}", self.path.display())).and_then(|bytes| {
            self.policy = untrusted_policy(&self.path);
            let (limits, allowed) = match &self.policy {
                Some(p) => (p.limits, UNTRUSTED_CAPS),
                None => (SandboxLimits::trusted(), ALL_CAPS),
            };
            let (inst, info) = Instance::load(&bytes, limits, allowed, module)?;
            let tables = intern(&info.manifest_text, &info.manifest)?;
            Ok((inst, info, tables))
        });
        match result {
            Ok((inst, info, tables)) => {
                self.inst = Some(inst);
                self.info = Some(info);
                self.tables = Some(tables);
            }
            Err(e) => {
                tracing::error!("Wasm Script {module:?}: {e}");
                self.error = Some(e);
            }
        }
    }

    /// Load on first use, when the `module` field changed, and when the file
    /// changed on disk (checked at most every [`RELOAD_POLL`]; `force` skips the wait).
    fn refresh(&mut self, module: &str, force: bool) {
        if self.loaded_for.as_deref() != Some(module) {
            return self.load(module);
        }
        if module.trim().is_empty() {
            return;
        }
        let due = force || self.checked.is_none_or(|t| t.elapsed() >= RELOAD_POLL);
        if due {
            self.checked = Some(Instant::now());
            if stamp_of(&self.path) != self.stamp {
                self.load(module);
            }
        }
    }

    /// A trap ends the instance: one log line, then the component is inert.
    fn disable(&mut self, what: &str, why: &str) {
        let module = self.loaded_for.clone().unwrap_or_default();
        let msg = format!("{what}: {why}; the component is disabled until its module changes");
        tracing::error!("Wasm Script {module:?}: {msg}");
        self.inst = None;
        self.error = Some(msg);
    }

    /// The tables bindings may see (none when the policy hides the actions).
    fn visible_actions(&self) -> &'static [ActionDesc] {
        match (&self.tables, &self.policy) {
            (Some(_), Some(p)) if !p.allow_actions => &[],
            (Some(t), _) => t.actions,
            _ => &[],
        }
    }
}

/// What the inspector shows about a component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WasmStatus {
    /// The module's name (from its manifest).
    pub name: String,
    /// A load error, or why a trapped module was disabled.
    pub error: Option<String>,
    /// Declared actions.
    pub actions: usize,
    /// Declared properties.
    pub properties: usize,
    /// Capabilities the module requested.
    pub capabilities: Vec<String>,
    /// The module runs under an untrusted-content policy.
    pub untrusted: bool,
    /// Bumped on every (re)load.
    pub generation: u32,
}

/// A sandboxed WebAssembly component whose declared functions can be bound
/// to events (see [`crate::wasm`]).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct WasmScript {
    /// The `.wasm` module: a project path or an `@<id>` asset reference.
    pub module: String,
    #[serde(skip)]
    rt: RefCell<Runtime>,
}

impl Default for WasmScript {
    fn default() -> Self {
        Self { module: String::new(), rt: RefCell::new(Runtime::default()) }
    }
}

impl WasmScript {
    /// A component hosting this module.
    pub fn new(module: impl Into<String>) -> Self {
        Self { module: module.into(), ..Self::default() }
    }

    /// Run `f` on the up-to-date runtime; `None` if it is busy (re-entered
    /// through a lookup while the guest runs).
    fn with<R>(&self, f: impl FnOnce(&mut Runtime) -> R) -> Option<R> {
        let mut rt = self.rt.try_borrow_mut().ok()?;
        rt.refresh(&self.module, false);
        Some(f(&mut rt))
    }

    /// Check the file now and reload if it changed (tests, a reload button).
    pub fn reload_if_changed(&self) {
        if let Ok(mut rt) = self.rt.try_borrow_mut() {
            rt.refresh(&self.module, true);
        }
    }

    /// Reload even if the file did not change.
    pub fn force_reload(&self) {
        if let Ok(mut rt) = self.rt.try_borrow_mut() {
            rt.load(&self.module);
        }
    }

    /// What the inspector shows about this component.
    pub fn status(&self) -> WasmStatus {
        self.with(|rt| WasmStatus {
            name: rt.info.as_ref().map(|i| i.manifest.name.clone()).unwrap_or_default(),
            error: rt.error.clone(),
            actions: rt.tables.map_or(0, |t| t.actions.len()),
            properties: rt.tables.map_or(0, |t| t.properties.len()),
            capabilities: rt.info.as_ref().map(|i| i.manifest.capabilities.clone()).unwrap_or_default(),
            untrusted: rt.policy.is_some(),
            generation: rt.generation,
        })
        .unwrap_or_default()
    }

    /// A float the module stored with `host.set_float` (tests, debugging).
    pub fn float(&self, key: &str) -> Option<f32> {
        self.rt.try_borrow().ok()?.inst.as_ref()?.float(key)
    }
}

/// The module's file name without folders or extension ("door" for
/// `scripts/door.wasm`); none for an unset or `@id` reference.
pub(crate) fn module_label(module: &str) -> Option<String> {
    if module.is_empty() || module.starts_with('@') {
        return None;
    }
    std::path::Path::new(module).file_stem().map(|s| s.to_string_lossy().into_owned()).filter(|s| !s.is_empty())
}

impl TypedComponent for WasmScript {
    const NAME: &'static str = "Wasm Script";

    fn start(&mut self, _ctx: &mut ComponentCtx) {
        // Load now so a broken module reports at Play start, not on first use.
        self.with(|_| ());
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let Self { module, rt } = self;
        let rt = rt.get_mut();
        rt.refresh(module, false);
        let Some(inst) = rt.inst.as_mut() else { return };
        if !inst.has_update() && !inst.has_outputs() {
            return;
        }
        let id = ctx.object_ids.get(ctx.self_index).copied().unwrap_or(ObjectId::NIL);
        if inst.has_update()
            && let Err(e) = inst.call_update(ctx.dt, ctx.time, ctx, id)
        {
            match e {
                CallError::Trap(m) => return rt.disable("zest_update", &m),
                CallError::Failed(m) => tracing::warn!("Wasm Script zest_update: {m}"),
            }
        }
        let out = inst.take_outputs();
        for name in out.events {
            ctx.emit(ActionEvent { name, source: id, instigator: None });
        }
        for call in out.pending {
            let binding = EventBinding {
                target: call.object,
                component: call.component,
                action: call.action,
                args: call.args.into_iter().map(ArgSource::Const).collect(),
                ..EventBinding::default()
            };
            // Same resolution and type checks as a bound action; one warning per
            // failing binding because this one is rebuilt on every call.
            let src = FireSource { source: id, instigator: None, payload: &[] };
            if let Err(e) = binding.try_fire(ctx, &src) {
                tracing::warn!("Wasm Script invoke {}.{}: {e}", binding.component, binding.action);
            }
        }
    }

    fn actions(&self) -> &'static [ActionDesc] {
        self.with(|rt| rt.visible_actions()).unwrap_or(&[])
    }

    fn instance_label(&self) -> Option<String> {
        module_label(&self.module)
    }

    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        let Self { module, rt } = self;
        let rt = rt.get_mut();
        rt.refresh(module, false);
        if rt.tables.is_none() {
            return Err(format!("Wasm Script has no module loaded: {}", rt.error.as_deref().unwrap_or("no module set")));
        }
        let actions = rt.visible_actions();
        let index = actions.iter().position(|a| a.name == name).ok_or_else(|| crate::actions::unknown_action("Wasm Script", name))?;
        if let Some(p) = &rt.policy
            && (!p.allow_actions || !actions[index].ugc_safe)
        {
            return Err(format!("'{name}' is not allowed for untrusted content (needs ugc_safe and content limits that allow module actions)"));
        }
        // A disabled component (trapped earlier) is inert; the error was logged once.
        let Some(inst) = rt.inst.as_mut() else { return Ok(()) };
        let result = inst.call_invoke(index, args, ctx.self_id, ctx.instigator, ctx.reader);
        for event in inst.take_events() {
            ctx.emit(ActionEvent { name: event, source: ctx.self_id, instigator: ctx.instigator });
        }
        match result {
            Ok(()) => Ok(()),
            Err(CallError::Failed(m)) => Err(m),
            Err(CallError::Trap(m)) => {
                rt.disable(name, &m);
                Ok(())
            }
        }
    }

    fn properties(&self) -> &'static [PropertyDesc] {
        self.with(|rt| rt.tables.map_or(&[][..], |t| t.properties)).unwrap_or(&[])
    }

    fn read_property(&self, name: &str) -> Option<ArgValue> {
        let mut rt = self.rt.try_borrow_mut().ok()?;
        rt.refresh(&self.module, false);
        let props = rt.tables?.properties;
        let index = props.iter().position(|p| p.name == name)?;
        let ty = props[index].ty;
        match rt.inst.as_mut()?.call_read_property(index) {
            Ok(v) => v?.coerce(ty).ok(),
            Err(CallError::Trap(m)) => {
                rt.disable(name, &m);
                None
            }
            Err(CallError::Failed(m)) => {
                tracing::warn!("Wasm Script property '{name}': {m}");
                None
            }
        }
    }
}
