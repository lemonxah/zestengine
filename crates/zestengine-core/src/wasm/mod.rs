//! WASM components that expose callable functions (EVT-2): the
//! [`WasmScript`] component hosts a sandboxed `.wasm` module whose declared
//! actions and properties appear in the same event-binding dropdowns as the
//! built-in Rust components (see [`crate::actions`]).
//!
//! The older [`crate::Script`] component (`env` imports, `start` / `update`
//! hooks) is a separate prototype and keeps working; this is the component
//! ABI that bindings call into.
//!
//! # Authoring a module
//!
//! Compile for `wasm32-unknown-unknown`. The `guest/zest-guest` crate in the
//! repository wraps everything below (manifest section, value codec, host
//! imports, export glue) for Rust authors; the rest of this section is the
//! exact contract it implements.
//!
//! ## Manifest
//!
//! A custom section named `zest_manifest` holding UTF-8 RON (at most 64 KiB):
//!
//! ```ron
//! (
//!     name: "Door",                       // inspector title
//!     api: 1,                             // manifest version
//!     capabilities: ["state", "events"],  // host imports the module uses
//!     actions: [
//!         (name: "Open", label: "Open door", ugc_safe: true, params: [
//!             (name: "speed", ty: Float, default: Some(Float(2.0))),
//!         ]),
//!     ],
//!     properties: [(name: "state", ty: Int)],
//! )
//! ```
//!
//! `ty` is `Bool`, `Int` (i64), `Float` (f32), `String`, `Vec3` or `Object`.
//! A custom section was chosen over an exported function because the host,
//! the package builder and the extractor can read it without running any guest
//! code. The position of an action in `actions` is the `action_index` the
//! host passes to `zest_invoke`; the position of a property is the `index` of
//! `zest_read_property`. Names are what scenes store, so never rename a
//! shipped action; indices may change freely between builds.
//!
//! ## Value encoding
//!
//! All integers are little-endian. A value is a one-byte tag and a payload:
//!
//! | tag | type   | payload                                          |
//! |-----|--------|--------------------------------------------------|
//! | 0   | Bool   | 1 byte, 0 or 1                                   |
//! | 1   | Int    | i64, 8 bytes                                     |
//! | 2   | Float  | f32, 4 bytes                                     |
//! | 3   | String | u32 byte length (at most 16384), UTF-8 bytes     |
//! | 4   | Vec3   | 3 x f32, 12 bytes                                |
//! | 5   | Object | u128 object id, 16 bytes, all zero = none        |
//!
//! An argument block is a `u32` count (at most 32) followed by that many
//! values, in parameter order, at most 64 KiB in all. The host has already
//! coerced each value to the declared parameter type and filled in defaults,
//! so a guest can decode by position without checking tags. [`abi`] has the
//! encoder and decoder.
//!
//! ## Exports
//!
//! | export | signature | notes |
//! |--------|-----------|-------|
//! | `memory` | memory | exactly one, 32-bit, within the memory cap |
//! | `zest_alloc` | `(len: i32) -> i32` | returns a pointer to at least `len` writable bytes the host fills with an argument block; the same scratch buffer may be returned every time (the host uses it once per call). Needed when actions are declared |
//! | `zest_invoke` | `(action_index: i32, args_ptr: i32, args_len: i32) -> i32` | 0 = ok; anything else is a failure reported to the binding |
//! | `zest_last_error` | `() -> i64` | optional; after a non-zero status, `(len << 32) \| ptr` of a UTF-8 message |
//! | `zest_read_property` | `(index: i32) -> i64` | `(len << 32) \| ptr` of one encoded value in guest memory (valid until the next call), or 0 for "no value". Needed when properties are declared |
//! | `zest_update` | `(dt: f32)` | optional; once per frame while playing |
//!
//! ## Host imports (module `host`, nothing else)
//!
//! Pointers are `i32` offsets into guest memory; strings are UTF-8 `(ptr, len)`
//! pairs of at most 256 bytes. A bad pointer traps the call.
//!
//! | import | capability | notes |
//! |--------|------------|-------|
//! | `log(ptr, len)` | none | 32 lines per call, 512 bytes each |
//! | `get_dt() -> f32`, `get_time() -> f32` | none | frame time and game time; the only clock |
//! | `get_self(out_ptr)` | none | writes the owner's 16-byte object id |
//! | `get_instigator(out_ptr) -> i32` | none | 16-byte id of whoever caused the event; 0 if none |
//! | `get_float(ptr, len) -> f32`, `set_float(ptr, len, v)` | `state` | per-component named floats (64 keys) |
//! | `emit_event(ptr, len)` | `events` | broadcasts an `ActionEvent`, 16 per call |
//! | `get_property(obj_ptr, comp_ptr, comp_len, prop_ptr, prop_len, out_ptr, out_cap) -> i32` | `read` | encoded value written to `out_ptr`, returns its length; an all-zero object means this one. Inside `zest_update`, and inside `zest_invoke` when the engine passes a world view (always in Play; -2 elsewhere, e.g. a test calling an action without one); -1 not found, -4 buffer too small. In `zest_invoke` the module's own component is not readable, its sibling components and other objects are |
//! | `invoke(obj_ptr, comp_ptr, comp_len, act_ptr, act_len, args_ptr, args_len) -> i32` | `invoke` | queues a call of any component's action (same type checks as a binding); it runs after the guest call returns, in the frame's next command pass. 0 queued, -5 bad arguments, -4 queue full |
//!
//! A module may import only these. A host function whose capability the
//! manifest doesn't declare makes the module fail to load. There is no WASI,
//! wall clock or randomness.
//!
//! # Sandbox
//!
//! - The `.wasm` file is capped ([`SandboxLimits`]), its memory is capped in
//!   pages (declared and grown; growth past the cap fails with -1), and so is
//!   the guest call stack (deep recursion traps).
//! - Every entry-point call, and instantiation, gets a fuel budget; running
//!   out terminates the call. The instance is reused across calls.
//! - Any trap (fuel, stack, `unreachable`, bad pointer) disables the
//!   component: one error is logged, later calls do nothing, and the engine
//!   keeps running. Editing the file (hot reload) or changing the `module`
//!   field loads a fresh instance.
//! - Imports are checked against the allowlist before instantiation.
//!
//! # Trust
//!
//! Modules inside an extracted content package are untrusted
//! ([`register_untrusted_root`]). They get tighter limits, may not use the
//! `invoke` capability, and expose an action to bindings only if its manifest
//! flag `ugc_safe` is set and the package's content limits allow module
//! actions (`ContentLimits::allow_wasm_actions`). Project modules are trusted.
//! See `docs/user-content.md`.
//!
//! # Dynamic descriptors
//!
//! `Component::actions` returns `&'static [ActionDesc]`, so a module's
//! manifest is turned into such tables once per distinct manifest text and
//! leaked ([`component`] interns them; at most 512 distinct manifests per
//! process, then loading fails). Reloading an unchanged manifest reuses the
//! tables.

pub mod abi;
pub mod component;
pub mod sandbox;
#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::RwLock;

pub use abi::{Manifest, ManifestAction, ManifestParam, ManifestProperty};
pub use component::WasmScript;
pub use sandbox::{CallError, Instance, ModuleInfo, SandboxLimits, UNTRUSTED_CAPS, inspect_module};

/// How modules under an untrusted root behave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UntrustedPolicy {
    /// Resource limits for modules under the root.
    pub limits: SandboxLimits,
    /// Whether the module's actions can be bound at all (the content limits').
    pub allow_actions: bool,
}

static UNTRUSTED: RwLock<Vec<(PathBuf, UntrustedPolicy)>> = RwLock::new(Vec::new());

/// Mark everything under `root` (an extracted package's folder) as untrusted
/// content. The engine's content manager calls this when it extracts a package.
pub fn register_untrusted_root(root: &Path, policy: UntrustedPolicy) {
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut roots = UNTRUSTED.write().unwrap_or_else(|e| e.into_inner());
    match roots.iter_mut().find(|(r, _)| *r == root) {
        Some(entry) => entry.1 = policy,
        None => roots.push((root, policy)),
    }
}

/// The policy for a module file, if it sits in an untrusted root.
pub fn untrusted_policy(module: &Path) -> Option<UntrustedPolicy> {
    let path = std::fs::canonicalize(module).unwrap_or_else(|_| module.to_path_buf());
    let roots = UNTRUSTED.read().unwrap_or_else(|e| e.into_inner());
    roots.iter().find(|(r, _)| path.starts_with(r)).map(|(_, p)| *p)
}
