//! Loading and running one WASM component module inside the sandbox: static
//! inspection (size, imports, memory, manifest section), the `host` import
//! set with capability gating, and bounded calls (fuel, stack, memory).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use wasmtime::{Caller, Engine, Extern, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc};

use super::abi::{self, Manifest};
use crate::actions::{ArgValue, PropertyReader};
use crate::{ComponentCtx, ObjectId, ObjectRef};

/// Name of the custom section that carries the manifest.
pub const MANIFEST_SECTION: &str = "zest_manifest";

const WASM_PAGE: u64 = 65_536;
/// Largest `get_property` / `invoke` string, and the per-call budgets that
/// keep one call from flooding the host.
const MAX_HOST_STRING: usize = 256;
const MAX_LOG_BYTES: usize = 512;
const MAX_LOGS_PER_CALL: u32 = 32;
const MAX_EVENTS_PER_CALL: usize = 16;
const MAX_PENDING_CALLS: usize = 16;
const MAX_FLOATS: usize = 64;
/// Compiled modules kept for components that share one `.wasm`.
const MODULE_CACHE: usize = 32;

/// Resource limits for one module, chosen by trust (see [`SandboxLimits::trusted`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandboxLimits {
    /// Largest `.wasm` file.
    pub max_module_bytes: usize,
    /// Linear-memory cap, in 64 KiB pages (declared and grown).
    pub max_memory_pages: u32,
    /// Instructions one entry-point call may run (wasmtime fuel).
    pub fuel_per_call: u64,
    /// Guest call-stack cap in bytes; deep recursion traps instead of crashing the host.
    pub max_stack_bytes: usize,
}

impl SandboxLimits {
    /// Project-authored modules: 4 MiB file, 16 MiB memory, 5 M instructions per call.
    pub const fn trusted() -> Self {
        Self { max_module_bytes: 4 << 20, max_memory_pages: 256, fuel_per_call: 5_000_000, max_stack_bytes: 512 << 10 }
    }
    /// Modules from user-generated content: 1 MiB file, 4 MiB memory, 1 M instructions per call.
    pub const fn untrusted() -> Self {
        Self { max_module_bytes: 1 << 20, max_memory_pages: 64, fuel_per_call: 1_000_000, max_stack_bytes: 256 << 10 }
    }
}

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

fn leb_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let mut v = 0u64;
    for shift in (0..35).step_by(7) {
        let b = *bytes.get(*pos)?;
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return u32::try_from(v).ok();
        }
    }
    None
}

/// The text of the `zest_manifest` custom section. Reads the section table
/// only; a module without one (or with two) is an error.
fn manifest_text(bytes: &[u8]) -> Result<String, String> {
    if bytes.len() < 8 || &bytes[..4] != b"\0asm" {
        return Err("not a WebAssembly binary".into());
    }
    let mut pos = 8;
    let mut found: Option<&[u8]> = None;
    while pos < bytes.len() {
        let id = bytes[pos];
        pos += 1;
        let size = leb_u32(bytes, &mut pos).ok_or("truncated section header")? as usize;
        let end = pos.checked_add(size).filter(|e| *e <= bytes.len()).ok_or("section runs past the end of the file")?;
        if id == 0 {
            let mut p = pos;
            let name_len = leb_u32(bytes, &mut p).ok_or("bad custom section name")? as usize;
            let name_end = p.checked_add(name_len).filter(|e| *e <= end).ok_or("bad custom section name")?;
            if &bytes[p..name_end] == MANIFEST_SECTION.as_bytes() {
                if found.is_some() {
                    return Err(format!("more than one {MANIFEST_SECTION:?} section"));
                }
                found = Some(&bytes[name_end..end]);
            }
        }
        pos = end;
    }
    let data = found.ok_or_else(|| format!("no {MANIFEST_SECTION:?} custom section (see the Wasm Script docs)"))?;
    String::from_utf8(data.to_vec()).map_err(|_| "manifest is not UTF-8".into())
}

fn shared_engine() -> Result<&'static Engine, String> {
    static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let mut c = wasmtime::Config::new();
            c.consume_fuel(true);
            // Deterministic floats; no threads or relaxed SIMD (results differ per CPU).
            c.cranelift_nan_canonicalization(true);
            c.wasm_relaxed_simd(false);
            // The deepest limit any trust level uses; each store tightens it with fuel.
            c.max_wasm_stack(SandboxLimits::trusted().max_stack_bytes);
            Engine::new(&c).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Compile `bytes` (cached by content) after the cheap checks.
fn compile(bytes: &[u8]) -> Result<Module, String> {
    static CACHE: Mutex<Vec<(Arc<[u8]>, Module)>> = Mutex::new(Vec::new());
    let engine = shared_engine()?;
    {
        let cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, m)) = cache.iter().find(|(b, _)| **b == *bytes) {
            return Ok(m.clone());
        }
    }
    let module = Module::new(engine, bytes).map_err(|e| format!("invalid module: {e}"))?;
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= MODULE_CACHE {
        cache.remove(0);
    }
    cache.push((Arc::from(bytes), module.clone()));
    Ok(module)
}

/// Check a module against `limits` and `allowed` capabilities without running
/// any of it: size, `host`-only imports from the allowlist, one capped
/// memory, a valid manifest, and that every import's capability is declared
/// (and allowed). Package building and extraction call this.
pub fn inspect_module(bytes: &[u8], limits: &SandboxLimits, allowed: u8) -> Result<(ModuleInfo, Module), String> {
    if bytes.len() > limits.max_module_bytes {
        return Err(format!("module is {} bytes (limit {})", bytes.len(), limits.max_module_bytes));
    }
    let text = manifest_text(bytes)?;
    let manifest = Manifest::parse(&text)?;
    let declared = declared_caps(&manifest);
    if declared & !allowed != 0 {
        let denied: Vec<&str> =
            manifest.capabilities.iter().map(String::as_str).filter(|c| cap_bit(c) & !allowed != 0).collect();
        return Err(format!("capabilities not permitted here: {}", denied.join(", ")));
    }
    let module = compile(bytes)?;
    for import in module.imports() {
        if import.module() != "host" {
            return Err(format!("imports {}.{}; only the \"host\" namespace is available", import.module(), import.name()));
        }
        let Some(&(_, need)) = IMPORTS.iter().find(|(n, _)| *n == import.name()) else {
            return Err(format!("imports unknown host function {:?}", import.name()));
        };
        if !import.ty().func().is_some() {
            return Err(format!("host.{} must be imported as a function", import.name()));
        }
        if need != 0 && declared & need == 0 {
            return Err(format!(
                "imports host.{} but the manifest does not declare the {:?} capability",
                import.name(),
                CAPABILITIES_BY_BIT.iter().find(|(b, _)| *b == need).map(|(_, n)| *n).unwrap_or("?")
            ));
        }
    }
    let mut memories = 0;
    for export in module.exports() {
        if let wasmtime::ExternType::Memory(m) = export.ty() {
            memories += 1;
            if m.is_64() || m.is_shared() {
                return Err("memory must be 32-bit and not shared".into());
            }
            if m.minimum() > limits.max_memory_pages as u64 {
                return Err(format!(
                    "memory starts at {} pages ({} KiB); limit {} pages",
                    m.minimum(),
                    m.minimum() * WASM_PAGE / 1024,
                    limits.max_memory_pages
                ));
            }
        }
    }
    if memories != 1 {
        return Err("the module must export exactly one memory named \"memory\"".into());
    }
    if module.get_export("memory").is_none() {
        return Err("the module must export its memory as \"memory\"".into());
    }
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
/// reads through the update context, `zest_invoke` through the reader the
/// engine passes). Set right before the call and cleared right after; host
/// functions only read through it.
#[derive(Clone, Copy)]
struct ReaderPtr(*const (dyn PropertyReader + 'static));
// SAFETY: only dereferenced on the thread that made the call, inside it.
unsafe impl Send for ReaderPtr {}
unsafe impl Sync for ReaderPtr {}

impl ReaderPtr {
    fn new(r: &dyn PropertyReader) -> Self {
        // SAFETY: the lifetime is erased only for the duration of the call
        // that stores it; `clear_reader` runs before the borrow ends.
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
    let mut l = Linker::new(shared_engine().map_err(wasmtime::Error::msg)?);
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
    CallError::Trap(match e.downcast_ref::<wasmtime::Trap>() {
        Some(wasmtime::Trap::OutOfFuel) => "ran past its instruction budget (infinite loop?)".into(),
        Some(wasmtime::Trap::StackOverflow) => "call stack exhausted".into(),
        Some(t) => t.to_string(),
        None => format!("{e}").lines().next().unwrap_or("trap").to_owned(),
    })
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
        let engine = shared_engine()?;
        let host = HostState {
            limits: StoreLimitsBuilder::new()
                .memory_size(limits.max_memory_pages as usize * WASM_PAGE as usize)
                .memories(1)
                .tables(2)
                .table_elements(10_000)
                .instances(1)
                .trap_on_grow_failure(false)
                .build(),
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
        // `host.get_property` reads through `reader` while the guest runs.
        self.store.data_mut().reader = reader.map(ReaderPtr::new);
        let status = invoke.call(&mut self.store, (index as i32, ptr, len));
        self.store.data_mut().reader = None;
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

    /// Run `zest_update(dt)`. `ctx` lets the guest call `host.get_property`.
    pub fn call_update(&mut self, dt: f32, time: f32, ctx: &ComponentCtx, self_id: ObjectId) -> Result<(), CallError> {
        let Some(update) = self.update.clone() else { return Ok(()) };
        self.begin(self_id, None)?;
        let s = self.store.data_mut();
        s.dt = dt;
        s.time = time;
        s.reader = Some(ReaderPtr::new(ctx));
        let result = update.call(&mut self.store, dt);
        self.store.data_mut().reader = None;
        result.map_err(trap_message)
    }
}
