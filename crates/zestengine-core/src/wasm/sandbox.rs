//! Loading and running one WASM component module inside the sandbox: static
//! inspection (size, imports, memory, manifest section), the `host` import
//! set with capability gating, and bounded calls (fuel, stack, memory).

use std::collections::HashMap;
use std::sync::OnceLock;

use zestengine_scripting::wasmtime;
use wasmtime::{Caller, Extern, Linker, Memory, Module, Store, StoreLimits, TypedFunc};

pub use zestengine_scripting::SandboxLimits;

use super::abi::{self, Manifest};
use crate::actions::{ArgValue, PropertyReader};
use crate::{ObjectId, ObjectRef};

/// Name of the custom section that carries the manifest.
pub const MANIFEST_SECTION: &str = "zest_manifest";

/// Largest `get_property` / `invoke` string, and the per-call budgets that
/// keep one call from flooding the host.
const MAX_HOST_STRING: usize = 256;
const MAX_LOG_BYTES: usize = 512;
const MAX_LOGS_PER_CALL: u32 = 32;
const MAX_EVENTS_PER_CALL: usize = 16;
const MAX_PENDING_CALLS: usize = 16;
const MAX_FLOATS: usize = 64;

// --------------------------------------------------------------- capabilities

/// Capability bit: the `state` floats.
pub const CAP_STATE: u8 = 1;
/// Capability bit: emitting events.
pub const CAP_EVENTS: u8 = 2;
/// Capability bit: reading world properties.
pub const CAP_READ: u8 = 4;
/// Capability bit: invoking other components' actions.
pub const CAP_INVOKE: u8 = 8;

fn cap_bit(name: &str) -> u8 {
    match name {
        "state" => CAP_STATE,
        "events" => CAP_EVENTS,
        "read" => CAP_READ,
        "invoke" => CAP_INVOKE,
        _ => 0,
    }
}

/// Every import a module may use, with the capability it needs (0 = always allowed).
const IMPORTS: &[(&str, u8)] = &[
    ("log", 0),
    ("get_dt", 0),
    ("get_time", 0),
    ("get_self", 0),
    ("get_instigator", 0),
    ("get_float", CAP_STATE),
    ("set_float", CAP_STATE),
    ("emit_event", CAP_EVENTS),
    ("get_property", CAP_READ),
    ("invoke", CAP_INVOKE),
];

/// Capabilities untrusted content may be granted: no `invoke`, so it can't
/// reach actions it wasn't bound to.
pub const UNTRUSTED_CAPS: u8 = CAP_STATE | CAP_EVENTS | CAP_READ;
/// Every capability (project-authored modules may use all of them).
pub const ALL_CAPS: u8 = CAP_STATE | CAP_EVENTS | CAP_READ | CAP_INVOKE;

fn declared_caps(m: &Manifest) -> u8 {
    m.capabilities.iter().fold(0, |a, c| a | cap_bit(c))
}

// ----------------------------------------------------------------- inspection

/// What static inspection learned about a module (nothing was run).
#[derive(Clone, Debug)]
pub struct ModuleInfo {
    /// The parsed manifest.
    pub manifest: Manifest,
    /// The manifest text it came from.
    pub manifest_text: String,
}

fn manifest_text(bytes: &[u8]) -> Result<String, String> {
    let data = zestengine_scripting::custom_section(bytes, MANIFEST_SECTION)?
        .ok_or_else(|| format!("no {MANIFEST_SECTION:?} custom section (see the Wasm Script docs)"))?;
    String::from_utf8(data.to_vec()).map_err(|_| "manifest is not UTF-8".into())
}

/// Check a module against `limits` and `allowed` capabilities without running
/// any of it: size, `host`-only imports from the allowlist, one capped
/// memory, a valid manifest, and that every import's capability is declared
/// (and allowed). Package building and extraction call this.
pub fn inspect_module(bytes: &[u8], limits: &SandboxLimits, allowed: u8) -> Result<(ModuleInfo, Module), String> {
    zestengine_scripting::check_size(bytes, limits)?;
    let text = manifest_text(bytes)?;
    let manifest = Manifest::parse(&text)?;
    let declared = declared_caps(&manifest);
    if declared & !allowed != 0 {
        let denied: Vec<&str> =
            manifest.capabilities.iter().map(String::as_str).filter(|c| cap_bit(c) & !allowed != 0).collect();
        return Err(format!("capabilities not permitted here: {}", denied.join(", ")));
    }
    let module = zestengine_scripting::compile(bytes)?;
    zestengine_scripting::check_imports(&module, "host", |name| {
        let Some(&(_, need)) = IMPORTS.iter().find(|(n, _)| *n == name) else {
            return Err(format!("imports unknown host function {name:?}"));
        };
        if need != 0 && declared & need == 0 {
            return Err(format!(
                "imports host.{name} but the manifest does not declare the {:?} capability",
                CAPABILITIES_BY_BIT.iter().find(|(b, _)| *b == need).map(|(_, n)| *n).unwrap_or("?")
            ));
        }
        Ok(())
    })?;
    zestengine_scripting::check_memory(&module, limits)?;
    for (needed, why) in [("zest_invoke", !manifest.actions.is_empty()), ("zest_read_property", !manifest.properties.is_empty())] {
        if why && module.get_export(needed).is_none() {
            return Err(format!("the manifest declares {} but the module exports no {needed}", if needed == "zest_invoke" { "actions" } else { "properties" }));
        }
    }
    if !manifest.actions.is_empty() && module.get_export("zest_alloc").is_none() {
        return Err("the manifest declares actions but the module exports no zest_alloc".into());
    }
    Ok((ModuleInfo { manifest, manifest_text: text }, module))
}

const CAPABILITIES_BY_BIT: &[(u8, &str)] = &[(CAP_STATE, "state"), (CAP_EVENTS, "events"), (CAP_READ, "read"), (CAP_INVOKE, "invoke")];

// ------------------------------------------------------------------ host state

/// An invoke request a guest queued; the component runs it with the engine
/// context once the guest call returned.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingCall {
    /// The object to call on.
    pub object: ObjectRef,
    /// Component type name.
    pub component: String,
    /// Action name.
    pub action: String,
    /// The action's arguments.
    pub args: Vec<ArgValue>,
}

/// Raw pointer to the world view of the guest call in progress (`zest_update`
/// and `zest_invoke` read through the checked property reader the engine
/// passes). Set right before the call and cleared on return or unwind; host
/// functions only read through it.
#[derive(Clone, Copy)]
struct ReaderPtr(*const (dyn PropertyReader + 'static));
// SAFETY: only dereferenced on the thread that made the call, inside it.
unsafe impl Send for ReaderPtr {}
unsafe impl Sync for ReaderPtr {}

impl ReaderPtr {
    fn new(r: &dyn PropertyReader) -> Self {
        // SAFETY: the lifetime is erased only for the duration of the call
        // that stores it; `with_reader` clears it before the borrow ends.
        let r: &'static dyn PropertyReader = unsafe { std::mem::transmute::<&dyn PropertyReader, &'static dyn PropertyReader>(r) };
        Self(r as *const dyn PropertyReader)
    }
}

/// The guest call's host-side state: capabilities, limits and per-call budgets.
pub struct HostState {
    limits: StoreLimits,
    caps: u8,
    label: String,
    floats: HashMap<String, f32>,
    events: Vec<String>,
    pending: Vec<PendingCall>,
    self_id: ObjectId,
    instigator: Option<ObjectId>,
    dt: f32,
    time: f32,
    logs_left: u32,
    reader: Option<ReaderPtr>,
}

/// Return codes of the host functions that return a status.
pub mod code {
    /// The value asked for was not found.
    pub const NOT_FOUND: i32 = -1;
    /// The value exists but no world view is available right now.
    pub const UNAVAILABLE: i32 = -2;
    /// The capability for the call is not granted.
    pub const DENIED: i32 = -3;
    /// The output buffer was smaller than the value.
    pub const TOO_SMALL: i32 = -4;
    /// The arguments did not decode.
    pub const BAD_ARGS: i32 = -5;
}

type Res<T> = Result<T, wasmtime::Error>;

fn memory(caller: &mut Caller<'_, HostState>) -> Res<Memory> {
    match caller.get_export("memory") {
        Some(Extern::Memory(m)) => Ok(m),
        _ => Err(wasmtime::Error::msg("module has no memory export")),
    }
}

fn bad_ptr(what: &str) -> wasmtime::Error {
    wasmtime::Error::msg(format!("host call {what}: pointer outside guest memory"))
}

fn read_bytes(caller: &mut Caller<'_, HostState>, ptr: i32, len: i32, max: usize, what: &str) -> Res<Vec<u8>> {
    let mem = memory(caller)?;
    let (ptr, len) = (ptr as u32 as usize, len as u32 as usize);
    if len > max {
        return Err(wasmtime::Error::msg(format!("host call {what}: {len} bytes (limit {max})")));
    }
    let end = ptr.checked_add(len).ok_or_else(|| bad_ptr(what))?;
    mem.data(&*caller).get(ptr..end).map(<[u8]>::to_vec).ok_or_else(|| bad_ptr(what))
}

fn read_string(caller: &mut Caller<'_, HostState>, ptr: i32, len: i32, max: usize, what: &str) -> Res<String> {
    Ok(String::from_utf8_lossy(&read_bytes(caller, ptr, len, max, what)?).into_owned())
}

fn write_bytes(caller: &mut Caller<'_, HostState>, ptr: i32, bytes: &[u8], what: &str) -> Res<()> {
    let mem = memory(caller)?;
    let ptr = ptr as u32 as usize;
    let end = ptr.checked_add(bytes.len()).ok_or_else(|| bad_ptr(what))?;
    mem.data_mut(&mut *caller).get_mut(ptr..end).ok_or_else(|| bad_ptr(what))?.copy_from_slice(bytes);
    Ok(())
}

fn read_object(caller: &mut Caller<'_, HostState>, ptr: i32, what: &str) -> Res<ObjectRef> {
    let b = read_bytes(caller, ptr, 16, 16, what)?;
    Ok(abi::object_from_bytes(b.try_into().unwrap()))
}

fn shared_linker() -> Result<&'static Linker<HostState>, String> {
    static LINKER: OnceLock<Result<Linker<HostState>, String>> = OnceLock::new();
    LINKER.get_or_init(|| build_linker().map_err(|e| e.to_string())).as_ref().map_err(Clone::clone)
}

fn build_linker() -> Res<Linker<HostState>> {
    let mut l = Linker::new(zestengine_scripting::engine().map_err(wasmtime::Error::msg)?);
    l.func_wrap("host", "log", |mut c: Caller<'_, HostState>, ptr: i32, len: i32| -> Res<()> {
        if c.data().logs_left == 0 {
            return Ok(());
        }
        c.data_mut().logs_left -= 1;
        let msg = read_string(&mut c, ptr, len.min(MAX_LOG_BYTES as i32), MAX_LOG_BYTES, "log")?;
        tracing::info!(target: "wasm", "{}: {msg}", c.data().label);
        Ok(())
    })?;
    l.func_wrap("host", "get_dt", |c: Caller<'_, HostState>| c.data().dt)?;
    l.func_wrap("host", "get_time", |c: Caller<'_, HostState>| c.data().time)?;
    l.func_wrap("host", "get_self", |mut c: Caller<'_, HostState>, out: i32| -> Res<()> {
        let id = abi::object_to_bytes(ObjectRef::new(c.data().self_id));
        write_bytes(&mut c, out, &id, "get_self")
    })?;
    l.func_wrap("host", "get_instigator", |mut c: Caller<'_, HostState>, out: i32| -> Res<i32> {
        match c.data().instigator {
            Some(id) => {
                write_bytes(&mut c, out, &abi::object_to_bytes(ObjectRef::new(id)), "get_instigator")?;
                Ok(1)
            }
            None => Ok(0),
        }
    })?;
    l.func_wrap("host", "get_float", |mut c: Caller<'_, HostState>, ptr: i32, len: i32| -> Res<f32> {
        if c.data().caps & CAP_STATE == 0 {
            return Ok(0.0);
        }
        let key = read_string(&mut c, ptr, len, MAX_HOST_STRING, "get_float")?;
        Ok(c.data().floats.get(&key).copied().unwrap_or(0.0))
    })?;
    l.func_wrap("host", "set_float", |mut c: Caller<'_, HostState>, ptr: i32, len: i32, v: f32| -> Res<()> {
        if c.data().caps & CAP_STATE == 0 {
            return Ok(());
        }
        let key = read_string(&mut c, ptr, len, MAX_HOST_STRING, "set_float")?;
        let st = c.data_mut();
        if st.floats.len() < MAX_FLOATS || st.floats.contains_key(&key) {
            st.floats.insert(key, v);
        }
        Ok(())
    })?;
    l.func_wrap("host", "emit_event", |mut c: Caller<'_, HostState>, ptr: i32, len: i32| -> Res<()> {
        if c.data().caps & CAP_EVENTS == 0 {
            return Ok(());
        }
        let name = read_string(&mut c, ptr, len, MAX_HOST_STRING, "emit_event")?;
        let st = c.data_mut();
        if st.events.len() < MAX_EVENTS_PER_CALL {
            st.events.push(name);
        }
        Ok(())
    })?;
    l.func_wrap(
        "host",
        "get_property",
        |mut c: Caller<'_, HostState>, obj: i32, cp: i32, cl: i32, pp: i32, pl: i32, out: i32, cap: i32| -> Res<i32> {
            if c.data().caps & CAP_READ == 0 {
                return Ok(code::DENIED);
            }
            let object = read_object(&mut c, obj, "get_property")?;
            let component = read_string(&mut c, cp, cl, MAX_HOST_STRING, "get_property")?;
            let property = read_string(&mut c, pp, pl, MAX_HOST_STRING, "get_property")?;
            let Some(ptr) = c.data().reader else { return Ok(code::UNAVAILABLE) };
            // SAFETY: set by `call_update` / `call_invoke` for the duration of the call; read-only.
            let reader = unsafe { &*ptr.0 };
            let id = object.id().unwrap_or(c.data().self_id);
            let Ok(value) = reader.read(id, &component, &property) else {
                return Ok(code::NOT_FOUND);
            };
            let mut buf = Vec::new();
            abi::encode_value(&mut buf, &value);
            if buf.len() > cap.max(0) as usize {
                return Ok(code::TOO_SMALL);
            }
            write_bytes(&mut c, out, &buf, "get_property")?;
            Ok(buf.len() as i32)
        },
    )?;
    l.func_wrap(
        "host",
        "invoke",
        |mut c: Caller<'_, HostState>, obj: i32, cp: i32, cl: i32, ap: i32, al: i32, argp: i32, argl: i32| -> Res<i32> {
            if c.data().caps & CAP_INVOKE == 0 {
                return Ok(code::DENIED);
            }
            let object = read_object(&mut c, obj, "invoke")?;
            let component = read_string(&mut c, cp, cl, MAX_HOST_STRING, "invoke")?;
            let action = read_string(&mut c, ap, al, MAX_HOST_STRING, "invoke")?;
            let raw = read_bytes(&mut c, argp, argl, abi::MAX_ENCODED_BYTES, "invoke")?;
            let Ok(args) = abi::decode_args(&raw) else { return Ok(code::BAD_ARGS) };
            let st = c.data_mut();
            if st.pending.len() >= MAX_PENDING_CALLS {
                return Ok(code::TOO_SMALL);
            }
            st.pending.push(PendingCall { object, component, action, args });
            Ok(0)
        },
    )?;
    Ok(l)
}

// -------------------------------------------------------------------- instance

/// A call failed. A trap (fuel, stack, memory fault, `unreachable`) leaves the
/// instance unusable; a failure is the guest saying "no" and is harmless.
#[derive(Debug, PartialEq)]
pub enum CallError {
    /// The guest trapped (fuel, stack, memory fault, unreachable); the
    /// instance is unusable.
    Trap(String),
    /// The guest reported a failure (harmless).
    Failed(String),
}

impl CallError {
    /// The error text.
    pub fn message(&self) -> &str {
        match self {
            Self::Trap(m) | Self::Failed(m) => m,
        }
    }
}

fn trap_message(e: wasmtime::Error) -> CallError {
    CallError::Trap(zestengine_scripting::trap_text(&e))
}

/// What a call left behind for the engine to carry out.
#[derive(Default)]
pub struct Outputs {
    /// Events the guest emitted.
    pub events: Vec<String>,
    /// Actions the guest queued for the engine to run.
    pub pending: Vec<PendingCall>,
}

impl Outputs {
    /// True when the guest left nothing to carry out.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty() && self.pending.is_empty()
    }
}

/// One instantiated module. Entry points are resolved once at load.
pub struct Instance {
    store: Store<HostState>,
    memory: Memory,
    alloc: Option<TypedFunc<i32, i32>>,
    invoke: Option<TypedFunc<(i32, i32, i32), i32>>,
    read_property: Option<TypedFunc<i32, i64>>,
    update: Option<TypedFunc<f32, ()>>,
    last_error: Option<TypedFunc<(), i64>>,
    limits: SandboxLimits,
    scratch: Vec<u8>,
}

fn unpack(v: i64) -> (usize, usize) {
    (v as u32 as usize, (v >> 32) as u32 as usize)
}

impl Instance {
    /// Inspect, compile and instantiate. `allowed` bounds the capabilities the
    /// manifest may declare; `label` names the module in logs.
    pub fn load(bytes: &[u8], limits: SandboxLimits, allowed: u8, label: &str) -> Result<(Instance, ModuleInfo), String> {
        let (info, module) = inspect_module(bytes, &limits, allowed)?;
        let engine = zestengine_scripting::engine()?;
        let host = HostState {
            limits: limits.store_limits(2),
            caps: declared_caps(&info.manifest) & allowed,
            label: label.to_owned(),
            floats: HashMap::new(),
            events: Vec::new(),
            pending: Vec::new(),
            self_id: ObjectId::NIL,
            instigator: None,
            dt: 0.0,
            time: 0.0,
            logs_left: MAX_LOGS_PER_CALL,
            reader: None,
        };
        let mut store = Store::new(engine, host);
        store.limiter(|s| &mut s.limits);
        store.set_fuel(limits.fuel_per_call).map_err(|e| e.to_string())?;
        // The start function (if any) runs inside this budget.
        let instance = shared_linker()?.instantiate(&mut store, &module).map_err(|e| format!("instantiating: {}", trap_message(e).message()))?;
        let memory = instance.get_memory(&mut store, "memory").ok_or("no exported memory")?;
        let inst = Instance {
            alloc: instance.get_typed_func(&mut store, "zest_alloc").ok(),
            invoke: instance.get_typed_func(&mut store, "zest_invoke").ok(),
            read_property: instance.get_typed_func(&mut store, "zest_read_property").ok(),
            update: instance.get_typed_func(&mut store, "zest_update").ok(),
            last_error: instance.get_typed_func(&mut store, "zest_last_error").ok(),
            store,
            memory,
            limits,
            scratch: Vec::new(),
        };
        if !info.manifest.actions.is_empty() && (inst.invoke.is_none() || inst.alloc.is_none()) {
            return Err("zest_invoke / zest_alloc have the wrong signature (see the ABI docs)".into());
        }
        if !info.manifest.properties.is_empty() && inst.read_property.is_none() {
            return Err("zest_read_property has the wrong signature (see the ABI docs)".into());
        }
        Ok((inst, info))
    }

    /// Whether the module exports `zest_update`.
    pub fn has_update(&self) -> bool {
        self.update.is_some()
    }

    /// Whether the guest queued anything (cheap; checked every frame).
    pub fn has_outputs(&self) -> bool {
        let s = self.store.data();
        !s.events.is_empty() || !s.pending.is_empty()
    }

    /// Take only the events (what an action call hands back at once).
    /// Take only the events (what an action call hands back at once).
    pub fn take_events(&mut self) -> Vec<String> {
        std::mem::take(&mut self.store.data_mut().events)
    }

    pub fn take_outputs(&mut self) -> Outputs {
        let s = self.store.data_mut();
        Outputs { events: std::mem::take(&mut s.events), pending: std::mem::take(&mut s.pending) }
    }

    /// Floats the guest stored with `host.set_float`.
    /// Floats the guest stored with `host.set_float`.
    pub fn float(&self, key: &str) -> Option<f32> {
        self.store.data().floats.get(key).copied()
    }

    fn begin(&mut self, self_id: ObjectId, instigator: Option<ObjectId>) -> Result<(), CallError> {
        let s = self.store.data_mut();
        s.self_id = self_id;
        s.instigator = instigator;
        s.logs_left = MAX_LOGS_PER_CALL;
        self.store.set_fuel(self.limits.fuel_per_call).map_err(|e| CallError::Trap(e.to_string()))
    }

    fn guest_error(&mut self, status: i32) -> CallError {
        let detail = self.last_error.clone().and_then(|f| f.call(&mut self.store, ()).ok()).and_then(|v| {
            let (p, l) = unpack(v);
            let data = self.memory.data(&self.store);
            data.get(p..p.checked_add(l.min(MAX_LOG_BYTES))?).map(|b| String::from_utf8_lossy(b).into_owned())
        });
        CallError::Failed(detail.unwrap_or_else(|| format!("returned status {status}")))
    }

    /// The guest bridge stores a reader only during its synchronous call.
    /// Clear it on host unwinding too, before any checked native loans end.
    fn with_reader<R>(&mut self, reader: Option<&dyn PropertyReader>, call: impl FnOnce(&mut Self) -> R) -> R {
        use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
        self.store.data_mut().reader = reader.map(ReaderPtr::new);
        let result = catch_unwind(AssertUnwindSafe(|| call(self)));
        self.store.data_mut().reader = None;
        match result { Ok(value) => value, Err(panic) => resume_unwind(panic) }
    }

    /// Run `zest_invoke(action_index, args)`.
    pub fn call_invoke(
        &mut self,
        index: usize,
        args: &[ArgValue],
        self_id: ObjectId,
        instigator: Option<ObjectId>,
        reader: Option<&dyn PropertyReader>,
    ) -> Result<(), CallError> {
        let (Some(alloc), Some(invoke)) = (self.alloc.clone(), self.invoke.clone()) else {
            return Err(CallError::Failed("module exports no zest_invoke".into()));
        };
        abi::encode_args_into(&mut self.scratch, args);
        if self.scratch.len() > abi::MAX_ENCODED_BYTES {
            return Err(CallError::Failed("arguments are too large".into()));
        }
        self.begin(self_id, instigator)?;
        let len = self.scratch.len() as i32;
        let ptr = alloc.call(&mut self.store, len).map_err(trap_message)?;
        let start = ptr as u32 as usize;
        let dst = self.memory.data_mut(&mut self.store).get_mut(start..start + len as usize);
        match dst {
            Some(dst) => dst.copy_from_slice(&self.scratch),
            None => return Err(CallError::Trap("zest_alloc returned a pointer outside memory".into())),
        }
        let status = self.with_reader(reader, |this| invoke.call(&mut this.store, (index as i32, ptr, len)));
        let status = status.map_err(trap_message)?;
        if status != 0 { Err(self.guest_error(status)) } else { Ok(()) }
    }

    /// Run `zest_read_property(index)`; `Ok(None)` when the guest has no value.
    pub fn call_read_property(&mut self, index: usize) -> Result<Option<ArgValue>, CallError> {
        let Some(read) = self.read_property.clone() else { return Ok(None) };
        let self_id = self.store.data().self_id;
        self.begin(self_id, None)?;
        let packed = read.call(&mut self.store, index as i32).map_err(trap_message)?;
        if packed == 0 {
            return Ok(None);
        }
        let (p, l) = unpack(packed);
        let data = self.memory.data(&self.store);
        let bytes = p
            .checked_add(l)
            .and_then(|end| data.get(p..end))
            .ok_or_else(|| CallError::Failed("zest_read_property returned a range outside memory".into()))?;
        abi::decode_one(bytes).map(Some).map_err(CallError::Failed)
    }

    /// Run `zest_update(dt)` with the native properties visible to this call.
    pub fn call_update(&mut self, dt: f32, time: f32, reader: Option<&dyn PropertyReader>, self_id: ObjectId) -> Result<(), CallError> {
        let Some(update) = self.update.clone() else { return Ok(()) };
        self.begin(self_id, None)?;
        let s = self.store.data_mut();
        s.dt = dt;
        s.time = time;
        let result = self.with_reader(reader, |this| update.call(&mut this.store, dt));
        result.map_err(trap_message)
    }
}

#[cfg(test)]
mod reader_tests {
    use super::*;

    struct PanickingReader;
    impl PropertyReader for PanickingReader {
        fn read(&self, _: ObjectId, _: &str, _: &str) -> Result<ArgValue, String> { panic!("native property panic probe") }
    }

    #[test]
    fn borrowed_guest_reader_is_cleared_when_a_host_property_callback_unwinds() {
        use crate::wasm::fixtures::{DRIVER, DRIVER_MANIFEST, build};
        let (mut instance, _) = Instance::load(&build(DRIVER_MANIFEST, DRIVER), SandboxLimits::trusted(), ALL_CAPS, "panic probe").unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = instance.call_update(0.1, 1.0, Some(&PanickingReader), ObjectId::NIL);
        }));
        assert!(result.is_err());
        assert!(instance.store.data().reader.is_none(), "no guest call may retain a reader after its checked query unwinds");
        instance.call_update(0.1, 2.0, None, ObjectId::NIL).unwrap();
        assert!(instance.store.data().reader.is_none());
    }
}
