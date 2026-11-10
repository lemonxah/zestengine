# WASM Scripting

ZestEngine runs sandboxed WebAssembly (WASM) modules inside the game process. A module can expose named actions and readable properties that show up in the editor's event-binding dropdowns, react to every frame, emit events, and call other components' actions, all without being able to touch anything the engine did not hand it.

This guide covers writing a module, building it, attaching it to an object, binding its functions to events, and shipping it in a project, a content package or a mod.

## Overview

### Why WASM

ZestEngine is Rust-first. Gameplay that ships with your game is native Rust: ECS components linked into the game binary, with the full engine API and no sandbox (see [ECS.md](ECS.md) and [ENGINE_API.md](ENGINE_API.md)). That is the trusted path, and it is the right one for your own game code.

It cannot be the path for code you do not trust. A player-made world, a prop from a content package or a mod must not be able to run machine code in your process. WASM is the engine's only sandbox for that: a module runs inside wasmtime and can only call the functions the engine imports into it, each call bounded by an instruction budget, a memory cap and a stack cap. There is no WASI, no network, no filesystem, no wall clock and no randomness.

The same mechanism also works for trusted project modules, for example logic you want to write in another language.

### The two component types

The engine has two WASM-hosting components. They are independent; pick one per module.

| | `Wasm Script` (`WasmScript`) | `Script` (legacy) |
|---|---|---|
| Source | `crates/zestengine-core/src/wasm/` | `crates/zestengine-core/src/script.rs` |
| Import module | `host` | `env` |
| Entry points | `zest_invoke`, `zest_read_property`, optional `zest_update` | `start()`, `update()` |
| Declares itself | Yes: a `zest_manifest` custom section (actions, properties, capabilities) | No |
| Shows up in binding dropdowns | Yes | No |
| Reads or writes the transform | No (it can read properties and call actions) | Yes: local position, rotation, scale |
| State | Its own linear memory (not saved) | 8 `f32` props, saved in the scene |
| Hot reload | Yes (file polled every 0.5 s) | No |
| Content packages and mods | Yes | No (see [The legacy Script component](#the-legacy-script-component)) |
| Editor inspector | Yes | None |
| Per-call fuel | 5 M (trusted), 1 M (untrusted) | 100 k |

### Which one to use

Use `Wasm Script` for everything new. Its actions and properties appear in the same dropdowns as built-in components, it has capability gating and trust levels, and it is the only one that content packages and mods accept.

Use the legacy `Script` only for tiny trusted transform tweaks, such as a spinner or a bobbing platform, where direct transform access is convenient and you are happy to edit scene text by hand.

### The zestengine-scripting crate

`crates/zestengine-scripting` is not used by the engine. No crate in the workspace depends on it. It is a standalone prototype host (`ScriptHost`) with three imports (`log`, `get_float`, `set_float`), an exported `update(f32)`, and none of the fuel, memory, capability or manifest machinery. The live hosts are `zestengine-core::wasm` (`Wasm Script`) and `zestengine-core::script` (legacy `Script`). Its only mention outside its own folder is a README line calling it "WASM host (consolidation with core integration pending)".

## Quick start

This takes the repository's `guest/example-counter` crate from source to a working counter bound to a button. The build steps (1 to 3) were not run end to end when this guide was written; see the note at the top of [Toolchain and building](#toolchain-and-building).

**1. Install the WASM target (once).**

```bash
rustup target add wasm32-unknown-unknown
```

**2. Build the example.**

```bash
cd guest/example-counter
cargo build --release --target wasm32-unknown-unknown
```

The module is `guest/example-counter/target/wasm32-unknown-unknown/release/example_counter.wasm`. The guest crates are their own Cargo workspaces, so `target/` is inside the crate folder.

**3. Put it in your project.**

```bash
mkdir -p /path/to/my-project/logic
cp target/wasm32-unknown-unknown/release/example_counter.wasm /path/to/my-project/logic/counter.wasm
```

Use a folder other than `src/`, `target/`, `plugins/` or `build*`: the build pipeline does not treat those as asset folders.

**4. Add the component.** In the editor, select (or create) an object, click Add Component at the bottom of the inspector and pick `Wasm Script`. Set Module to `logic/counter.wasm` with the asset picker or by dragging the file onto the field. The inspector now shows what the module declared:

```text
Counter
Add(amount: Int)
Reset()
property count: Int
capabilities: events
```

If the module is rejected, the inspector shows the reason in red instead (see [Testing and debugging](#testing-and-debugging)).

**5. Bind an action.** On another object add an `Interactable` component. Under On interact click Add action, then set Target to the counter object, Component to `Wasm Script`, Function to `Add to count`, and the `amount (Int)` parameter to the Value `5`.

**6. Try it.** Enter Play mode. The row's play button (Test: run this action now) fires the binding immediately, or interact with the object normally. To see the number, add a `UI Text` whose `SetText` action takes its `text` from a Property source: object = the counter, component = `Wasm Script`, property = `count`.

**7. Iterate.** Rebuild and copy the file again. The engine notices the changed file within about half a second and loads a fresh instance (the counter restarts at 0). The inspector's Reload module button forces it.

## Toolchain and building

Verification status: the engine side (loading, ABI, sandbox, bindings) was verified by running the repository's unit tests (29 `wasm` and 8 `script` tests pass) against WAT modules, and every guest snippet in this guide was checked with `cargo check` on the host target (with the manifests and the WAT module also parsed and run by the engine's real loader), but no real `wasm32-unknown-unknown` build was produced because that target is not installed on the machine this guide was written on, so the build commands, the output path and the `link_section` manifest embedding below come from reading the `Cargo.toml` files and the guest macros.

### Prerequisites

- Rust 1.82 or newer (the guest code uses `#[unsafe(no_mangle)]`, `#[unsafe(link_section)]` and `unsafe extern` blocks). The host-target checks in this guide ran on rustc 1.98.1.
- The WASM target: `rustup target add wasm32-unknown-unknown`.
- Nothing else. No `wasm-bindgen`, no WASI SDK, no `wasm-pack`.

### Building a guest crate

`guest/zest-guest` and `guest/example-counter` each contain an empty `[workspace]` table, which makes them standalone workspaces rather than members of the engine workspace. Build from inside the crate folder:

```bash
cd guest/example-counter
cargo build --release --target wasm32-unknown-unknown
# -> target/wasm32-unknown-unknown/release/example_counter.wasm
```

The file name is the crate name with `-` turned into `_`. The crate must be a `cdylib`:

```toml
[lib]
crate-type = ["cdylib"]
```

You can also check a guest crate on your development machine with a plain `cargo check` or `cargo test` (no `--target`): `zest-guest` swaps its host imports for stubs when not compiling for `wasm32`. See [Testing and debugging](#testing-and-debugging) for what that does and does not cover.

### Release profile and size

`example-counter` ships this profile:

```toml
[profile.release]
opt-level = "s"
lto = true
panic = "abort"
```

`panic = "abort"` removes the unwinding machinery. `lto = true` lets the linker drop unused code. `opt-level = "s"` trades a little speed for size. Beyond that, the usual size costs are float formatting, `format!`-heavy error messages and large dependencies. Keep modules small: the file cap is 4 MiB for project modules and 1 MiB for packages and mods (512 KiB for Prop packages). `opt-level = "z"` and `codegen-units = 1` are standard Rust size options but were not tried here.

Do not strip custom sections. The engine needs the `zest_manifest` section; a tool that removes custom sections (some strip options and optimisers do) makes the module fail with `no "zest_manifest" custom section`. After any post-processing, check that the module still loads. `grep -c -a zest_manifest module.wasm` is a quick sanity check.

### Where the engine looks for the module

The `module` field of a `Wasm Script` is one of:

- a path relative to the process working directory (the editor sets it to the project root), for example `logic/door.wasm`;
- an absolute path;
- an `@<id>` asset reference. The editor's asset picker stores this form: it writes a `.meta` sidecar next to the `.wasm` (keep it under version control) so renaming or moving the file keeps the reference valid.

A built game ships every file reachable from its scenes, so a module named by a `Wasm Script` component is included. A module loaded some other way is included only if it sits under `resources/` or its path appears as a string literal in `src/`.

The file is polled for changes (modification time and size) about every 0.5 s. Editing the `module` field also loads a fresh instance. Each component gets its own instance with its own memory, even when several objects use the same `.wasm`; the compiled code and the manifest tables are shared.

### Content packages and mods

Packages (`.zestenginepak`) carry `.wasm` files under `files/<project-relative path>`. The builder (Tools > Build Content Package) and the extractor both validate each module without running it, so a bad module fails the build instead of a player's game. On extraction, strings in RON files that name packaged files, including a `Wasm Script`'s `module`, are rewritten to absolute paths inside the extraction folder, and that folder is registered as untrusted. A module's trust is its location. See [Limits and safety](#limits-and-safety).

Mods reuse the same archives. The mod mounts into a per-session snapshot, `module` references resolve inside it, and the module's trust and action permissions come from the mod. A mod can also replace a game's own `.wasm` by path; the replacement keeps the mod sandbox. See [Limits and safety](#limits-and-safety), and the game's mod settings under Project Settings > Mods in the editor.

### Writing modules in other languages

Anything that produces a core WebAssembly module with the right shape works. The engine only looks at the binary:

1. Imports, if any, come from the module `host` and name functions in [Host imports](#host-imports). No WASI, no other namespace, no imported memory, table or global.
2. One exported 32-bit, non-shared memory named `memory`, whose initial size is within the page cap.
3. Exactly one custom section named `zest_manifest` containing UTF-8 RON, at most 64 KiB.
4. The exports the manifest implies: `zest_alloc` and `zest_invoke` when it declares actions, `zest_read_property` when it declares properties. `zest_update` and `zest_last_error` are optional.
5. Every capability a host import needs is declared in the manifest.

This is a complete module in WAT. It was assembled with the `wat` crate and loaded by the engine's loader; it also passes the package validator's untrusted limits:

```wat
(module
  (@custom "zest_manifest" "(name: \"Flag\", actions: [(name: \"Raise\", ugc_safe: true)], properties: [(name: \"raised\", ty: Bool)])")
  (memory (export "memory") 1)
  (global $raised (mut i32) (i32.const 0))

  ;; Scratch buffer the host writes the argument block into.
  (func (export "zest_alloc") (param $len i32) (result i32)
    i32.const 1024)

  ;; Action 0 is Raise. Status 0 means success, anything else is a failure.
  (func (export "zest_invoke") (param $action i32) (param $ptr i32) (param $len i32) (result i32)
    (if (i32.eqz (local.get $action))
      (then (global.set $raised (i32.const 1)) (return (i32.const 0))))
    i32.const 1)

  ;; Property 0 is raised, encoded as one Bool value: tag 0, then one byte.
  (func (export "zest_read_property") (param $index i32) (result i64)
    (i32.store8 (i32.const 4096) (i32.const 0))
    (i32.store8 (i32.const 4097) (global.get $raised))
    ;; (len << 32) | ptr, with len = 2 and ptr = 4096
    (i64.or (i64.shl (i64.const 2) (i64.const 32)) (i64.const 4096))))
```

Other toolchains (C or C++ with clang, Zig, AssemblyScript, a post-link step that appends the section) should work if they produce the shape above, but none was tried for this guide. The one non-obvious part is embedding the custom section; `zest-guest` does it with `#[unsafe(link_section = "zest_manifest")]` on a `static`.

## The WasmScript component ABI

This is the exact contract between the engine and a `Wasm Script` module. `guest/zest-guest` implements all of it for Rust authors. All numbers below are copied from `crates/zestengine-core/src/wasm/{abi,sandbox,component}.rs`.

### Manifest

A custom section named `zest_manifest` holds RON text, at most 64 KiB (65536 bytes). It is read without running any guest code, which is why packages and tools can inspect a module cheaply.

```ron
(
    name: "Door",                         // shown in the inspector
    api: 1,                               // manifest version; 1 is the only one
    capabilities: ["events", "invoke"],   // host imports the module uses
    actions: [
        (name: "Open", label: "Open door", ugc_safe: false, params: [
            (name: "blocker", ty: Object),
            (name: "sound", ty: String, default: Some(String("audio/door_open.wav"))),
            (name: "offset", ty: Vec3, default: Some(Vec3((0.0, 1.0, 0.0)))),
        ]),
    ],
    properties: [(name: "is_open", ty: Bool)],
)
```

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Inspector title. Empty shows as `(unnamed)`. |
| `api` | u32 | Manifest version. Must be 1 or less (default 0 is accepted); write `1`. |
| `capabilities` | list of strings | Any of `state`, `events`, `read`, `invoke`. Unknown names fail the load. |
| `actions` | list | At most 64. The position in the list is the `action_index` passed to `zest_invoke`. |
| `actions[].name` | string | What bindings store in scenes. Never rename a shipped action. |
| `actions[].label` | string | Dropdown text; defaults to `name`. |
| `actions[].params` | list | At most 16, in the order the guest receives them. |
| `actions[].ugc_safe` | bool | Default false. Whether untrusted content may bind this action. |
| `params[].name`, `ty`, `default` | | `ty` is `Bool`, `Int` (i64), `Float` (f32), `String`, `Vec3` or `Object`. `default` is `Some(<Type>(value))` and must match `ty`; otherwise the zero value is used (an `Object` default is always none). |
| `properties` | list | At most 64. The position is the `index` passed to `zest_read_property`. |

Names are 1 to 64 bytes of letters, digits, space, `_`, `-` and `.`, and action names and property names must each be unique. Notes from testing the real parser:

- A `Vec3` default is written with a tuple: `Vec3((0.0, 1.0, 0.0))`. Square brackets fail with `Expected opening '('`.
- Unknown fields are ignored, not rejected. A typo such as `ugc_save: true` silently leaves the action not UGC-safe.
- Indices can change freely between builds; names are the stable identity. Keep your action numbers in constants next to the manifest.
- The engine builds its dropdown tables once per distinct manifest text and keeps them for the life of the process, at most 512 distinct manifests, after which loading fails until restart.

### Value encoding

All integers are little-endian. A value is a one-byte tag followed by a payload.

| Tag | Type | Payload |
|---|---|---|
| 0 | Bool | 1 byte, 0 or 1 (any non-zero byte reads as true) |
| 1 | Int | `i64`, 8 bytes |
| 2 | Float | `f32`, 4 bytes |
| 3 | String | `u32` byte length (at most 16384), then UTF-8 bytes |
| 4 | Vec3 | 3 x `f32`, 12 bytes |
| 5 | Object | `u128` object id, 16 bytes; all zero means none |

An argument block is a `u32` count (at most 32) followed by that many values, in parameter order, at most 64 KiB in all. The host has already coerced every argument to the declared parameter type and filled in defaults for missing ones, so a guest can decode by position without checking tags. Coercion is the engine's usual one: numbers and bools convert between each other (an `Int` from a `Float` rounds), any value converts to text, everything else must match.

Property values are a single tagged value (no count). The host coerces the result to the declared property type; a value that cannot be coerced reads as "no value".

### Guest exports

| Export | Signature | Required | Behaviour |
|---|---|---|---|
| `memory` | memory | Yes | Exactly one, 32-bit, not shared. |
| `zest_alloc` | `(len: i32) -> i32` | When actions are declared | Return a pointer to at least `len` writable bytes. The host copies the argument block there before each `zest_invoke`. Returning the same scratch buffer every time is fine. |
| `zest_invoke` | `(action_index: i32, args_ptr: i32, args_len: i32) -> i32` | When actions are declared | Run an action. Return 0 for success; any other value is a failure reported to the binding. |
| `zest_last_error` | `() -> i64` | No | After a non-zero status, return `(len << 32) \| ptr` of a UTF-8 message (cut to 512 bytes). Without it the error is `returned status N`. |
| `zest_read_property` | `(index: i32) -> i64` | When properties are declared | Return `(len << 32) \| ptr` of one encoded value in guest memory, valid until the next call, or 0 for "no value". |
| `zest_update` | `(dt: f32)` | No | Once per frame while playing, for active objects. |

Exports are looked up by name and signature. If `zest_invoke`, `zest_alloc` or `zest_read_property` is declared but has the wrong signature, the load fails. A `zest_update` or `zest_last_error` with the wrong signature is silently ignored, so "my update never runs" usually means a wrong signature (it must be `(f32) -> ()`).

### Host imports

A module may import only these, all from the module `host`. Pointers are `i32` offsets into the guest's memory. Strings are UTF-8 `(ptr, len)` pairs, at most 256 bytes (except `log`, which cuts at 512).

| Import | Wasm signature | Capability | Semantics |
|---|---|---|---|
| `log` | `(ptr, len)` | none | Write a line to the engine log (INFO, target `wasm`). At most 32 lines per entry-point call; text beyond 512 bytes is cut; invalid UTF-8 is replaced. |
| `get_dt` | `() -> f32` | none | Frame time passed to the most recent `zest_update` (the engine caps one frame at 0.25 s before applying the time scale). It is 0.0 until the first update, and inside `zest_invoke` it is the value from the last update, not a fresh one. |
| `get_time` | `() -> f32` | none | Game time at the most recent `zest_update`; same caveat. The only clock. |
| `get_self` | `(out_ptr)` | none | Write the owner object's 16-byte id to `out_ptr`. |
| `get_instigator` | `(out_ptr) -> i32` | none | Writes the 16-byte id of whoever caused the event (the player who used a switch) and returns 1, or returns 0 and writes nothing. Always 0 inside `zest_update`. |
| `get_float` | `(ptr, len) -> f32` | `state` | Read a named float from this component's store; 0.0 if unset. |
| `set_float` | `(ptr, len, v: f32)` | `state` | Store a named float. At most 64 keys; new keys beyond that are silently dropped. |
| `emit_event` | `(ptr, len)` | `events` | Broadcast an `ActionEvent`. At most 16 per entry-point call; extras are dropped. |
| `get_property` | `(obj_ptr, comp_ptr, comp_len, prop_ptr, prop_len, out_ptr, out_cap) -> i32` | `read` | Read a property of a component on an object (see below). |
| `invoke` | `(obj_ptr, comp_ptr, comp_len, act_ptr, act_len, args_ptr, args_len) -> i32` | `invoke` | Queue a call of any component's action (see below). |

`get_float` and `set_float` are a small host-side store keyed by name. It is not saved, not replicated and not visible to bindings (only to tests through `WasmScript::float`), and it dies with the instance. Your own linear memory persists across calls just as well, so you rarely need it.

`get_property` details:

- `obj_ptr` points at a 16-byte object id; all zero means this object.
- `comp_len` of 0, or the component `Object`, reads the built-in object properties `name` (String), `active` (Bool) and `position` (Vec3).
- The first instance of the component on that object is read.
- On success the encoded value is written to `out_ptr` and its length is returned (always more than 0). Negative returns are in the table below.
- It works inside `zest_update`, and inside `zest_invoke` when the engine passes a world view (always in Play). Inside `zest_read_property`, and in a test with no world view, it returns -2.
- Inside `zest_invoke` and `zest_update` the module's own component is not readable; its sibling components and other objects are. Sibling `Wasm Script` components on the same object are readable by their declared property names.

`invoke` details:

- The call runs after your function returns, when the owning component next drains its queue (the next gameplay pass). It never runs inside the guest call, so there is no return value from the target action.
- It goes through the same resolution and type checks as an editor binding: the object must exist, have a component of that name, which must declare that action, and the arguments are coerced to the declared parameter types. Problems after queueing are logged by the engine, not returned to you.
- `obj_ptr` all zero means this object. For built-in object actions (`SetActive`, `ToggleActive`, `PlaySound`, `EmitEvent`, `LoadScene`) use the component name `Object`.
- The argument block is the encoding above, at most 64 KiB.
- At most 16 calls can be waiting in the queue.

Return codes of the status-returning imports:

| Code | Name | Meaning |
|---|---|---|
| >= 0 | ok | `get_property`: encoded length; `invoke`: 0 = queued; `get_instigator`: 1 or 0 |
| -1 | not found | `get_property`: no such object, component or property (or reading your own component) |
| -2 | unavailable | `get_property`: no world view in this call |
| -3 | denied | The capability is not granted (cannot happen for a module that loaded) |
| -4 | too small / full | `get_property`: `out_cap` too small; `invoke`: queue full |
| -5 | bad arguments | `invoke`: the argument block did not decode |

A string longer than 256 bytes, or a pointer outside guest memory, is not a return code: the host function raises a trap and the component is disabled (see [Errors and traps](#errors-and-traps)).

### Capabilities

A capability is a name in the manifest that unlocks host imports. Importing a host function whose capability is not declared makes the module fail to load. Declaring one you never use is harmless.

| Capability | Unlocks | Project module | Package (untrusted) | Mod |
|---|---|---|---|---|
| none | `log`, `get_dt`, `get_time`, `get_self`, `get_instigator` | yes | yes | yes |
| `state` | `get_float`, `set_float` | yes | yes | yes |
| `events` | `emit_event` | yes | yes | yes |
| `read` | `get_property` | yes | yes | yes |
| `invoke` | `invoke` | yes | no, refused at load | yes, each call checked against the game's allow-list |

### Errors and traps

There are three outcomes.

- **Load error.** The file is missing, not a valid module, fails validation or instantiation. The component stays inert and shows the message in the inspector, and one `ERROR` line is logged. Fixing the file (or the `module` field) loads it again.
- **Guest failure.** `zest_invoke` returns a non-zero status. This is the module saying no; it does not disable anything. The message from `zest_last_error` (or `returned status N`) goes back to the binding, which logs it as a warning such as `action Wasm Script.Open: <message>`. The rest of a binding batch still runs. (The editor's Test button reports only problems resolving the binding, such as a missing action; a guest failure shows up in the log.)
- **Trap.** Running out of fuel, a stack overflow, `unreachable` (this includes every Rust panic with `panic = "abort"`), an out-of-bounds access, or a host call that rejects its arguments. The instance is dropped, one `ERROR` line is logged, and the component stays disabled (later calls do nothing, and bindings to it still resolve) until its module changes: an edited file, a changed `module` field or the inspector's Reload module button.

There is no rollback. The guest's memory goes away with the instance. Events a failed or trapping action already emitted are still delivered; events and queued calls from a trapping `zest_update` are discarded. A malformed `zest_read_property` result (wrong length, unknown tag) is not a trap: it is logged as `Wasm Script property '<name>': <reason>` and the property reads as no value.

## Writing guests with zest-guest

`guest/zest-guest` is a small crate with no dependencies. It implements the manifest section, the codec, the host imports and the export glue, so you write one `Module` implementation.

### Crate layout

```toml
[package]
name = "my-module"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
zest-guest = { path = "../zestengine/guest/zest-guest" }

[profile.release]
opt-level = "s"
lto = true
panic = "abort"

# Not part of any other workspace: it builds for wasm32.
[workspace]
```

This is `guest/example-counter/Cargo.toml` with the dependency path adjusted.

### The API at a glance

| Item | Notes |
|---|---|
| `manifest!(r#"( ... )"#)` | Embeds the manifest as the `zest_manifest` custom section. Use it once per crate. |
| `trait Module` | `invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String>` (required); `property(&mut self, index: u32) -> Option<Value>` (default `None`); `update(&mut self, dt: f32)` (default no-op). |
| `export!(init)` | Exports `zest_alloc`, `zest_invoke`, `zest_last_error` and `zest_read_property`. `init` is an expression that builds the module; it runs lazily, once, on the first call. |
| `export!(init, update)` | The same, plus `zest_update`, so `Module::update` runs every frame. Without `update` in the macro the method is never called. |
| `enum Value` | `Bool(bool)`, `Int(i64)`, `Float(f32)`, `Str(String)`, `Vec3([f32; 3])`, `Object([u8; 16])`. Helpers: `as_bool`, `as_i64`, `as_f32`, `as_str`, `as_vec3` (no object helper: match `Value::Object`). |
| `type ObjectId = [u8; 16]` | All zero means none, or "this object" in a host call. |
| `log(&str)` | No capability. |
| `dt()`, `time()` | No capability. See the `get_dt` caveat above. |
| `this_object() -> ObjectId`, `instigator() -> Option<ObjectId>` | No capability. |
| `get_float(key)`, `set_float(key, v)` | `state`. |
| `emit_event(name)` | `events`. |
| `get_property(object, component, property) -> Option<Value>` | `read`. Uses a 512-byte buffer, so a long String property does not fit. Every failure is `None`. |
| `invoke(object, component, action, args: &[Value]) -> i32` | `invoke`. Returns 0 when queued, else the status code. |
| `codec` | `encode_value`, `encode_args`, `decode_value`, `decode_args`, if you write your own exports. |

Guidelines that follow from the ABI:

- Match the manifest and the code. The host always passes every declared parameter, so `args[i]` is safe when the code agrees with the manifest. A mismatch panics, and a panic traps.
- Use the `as_*` helper that matches the declared type: the host has already converted the argument to that type, so the helper is exact. (The helpers' own conversions differ from the host's, for example `as_i64` truncates a float where the host rounds.)
- Return `Err(message)` for expected problems (bad input, wrong state). Reserve panics for bugs.
- State lives in your `Module` struct. It survives between calls, resets on hot reload, and is not saved with the scene.

### The counter example

`guest/example-counter/src/lib.rs`, minus its doc comment. It is the smallest useful module: two actions, one property and one event.

```rust
use zest_guest::{Module, Value};

zest_guest::manifest!(
    r#"(
    name: "Counter",
    api: 1,
    capabilities: ["events"],
    actions: [
        (name: "Add", label: "Add to count", ugc_safe: true, params: [
            (name: "amount", ty: Int, default: Some(Int(1))),
        ]),
        (name: "Reset", ugc_safe: true),
    ],
    properties: [(name: "count", ty: Int)],
)"#
);

#[derive(Default)]
struct Counter(i64);

impl Module for Counter {
    fn invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String> {
        match action {
            0 => self.0 += args[0].as_i64(),
            1 => self.0 = 0,
            _ => return Err(format!("no action {action}")),
        }
        zest_guest::emit_event("Changed");
        Ok(())
    }
    fn property(&mut self, index: u32) -> Option<Value> {
        (index == 0).then(|| Value::Int(self.0))
    }
}

zest_guest::export!(Counter::default());
```

Action 0 is `Add` and action 1 is `Reset` because that is their order in the manifest. Another object reads `count` through a Property source in a binding, and `Changed` is broadcast as an event (see [Events](#events)). It is `ugc_safe` because adding to a number touches nothing outside the module.

### A door that opens through an action

A door that plays a sound, switches off a blocking collider and reports its state. It uses `invoke` to call the built-in `Object.PlaySound` and `Object.SetActive` actions, so it declares the `invoke` capability and works as a project module, or as a mod whose game allow-lists those two actions (packages cannot use `invoke`). Checked with `cargo check` on the host target; not built for wasm32.

```rust
use zest_guest::{Module, ObjectId, Value};

zest_guest::manifest!(
    r#"(
    name: "Door",
    api: 1,
    capabilities: ["events", "invoke"],
    actions: [
        (name: "Open", label: "Open door", params: [
            (name: "blocker", ty: Object),
            (name: "sound", ty: String, default: Some(String("audio/door_open.wav"))),
        ]),
        (name: "Close", label: "Close door", params: [
            (name: "blocker", ty: Object),
        ]),
        (name: "Toggle", label: "Toggle door", params: [
            (name: "blocker", ty: Object),
            (name: "sound", ty: String, default: Some(String("audio/door_open.wav"))),
        ]),
    ],
    properties: [(name: "is_open", ty: Bool)],
)"#
);

const OPEN: u32 = 0;
const CLOSE: u32 = 1;
const TOGGLE: u32 = 2;

/// The all-zero id: "no object" in a parameter, "this object" in a host call.
const NONE: ObjectId = [0; 16];

#[derive(Default)]
struct Door {
    open: bool,
}

fn object(v: &Value) -> ObjectId {
    match v {
        Value::Object(id) => *id,
        _ => NONE,
    }
}

impl Door {
    fn set_open(&mut self, open: bool, blocker: ObjectId, sound: &str) -> Result<(), String> {
        if self.open == open {
            return Ok(()); // already there, nothing to do
        }
        // The blocker is a collider that is active while the door is shut. An
        // unset parameter would mean "this object" to the host, so skip it.
        if blocker != NONE {
            let status = zest_guest::invoke(blocker, "Object", "SetActive", &[Value::Bool(!open)]);
            if status != 0 {
                return Err(format!("could not queue SetActive (status {status})"));
            }
        }
        if open && !sound.is_empty() {
            zest_guest::invoke(NONE, "Object", "PlaySound", &[Value::Str(sound.to_owned()), Value::Float(1.0)]);
        }
        self.open = open;
        zest_guest::emit_event(if open { "DoorOpened" } else { "DoorClosed" });
        zest_guest::log(if open { "door opened" } else { "door closed" });
        Ok(())
    }
}

impl Module for Door {
    fn invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String> {
        match action {
            OPEN => self.set_open(true, object(&args[0]), args[1].as_str()),
            CLOSE => self.set_open(false, object(&args[0]), ""),
            TOGGLE => self.set_open(!self.open, object(&args[0]), args[1].as_str()),
            _ => Err(format!("no action {action}")),
        }
    }

    fn property(&mut self, index: u32) -> Option<Value> {
        (index == 0).then(|| Value::Bool(self.open))
    }
}

zest_guest::export!(Door::default());
```

To wire it up, put the module on the door object, create the blocker as a child collider, and on an `Interactable` add an On interact action: Target = the door, Component = `Wasm Script`, Function = `Toggle door`, `blocker` = the blocker object, `sound` = a sound in your project. Other objects can read `is_open` through a Property source.

### Calling a built-in action

`Hazard.Zap` damages whatever is passed to it. It reads `Health.dead` first (capability `read`) and queues `Health.Damage` (capability `invoke`). `Health.Damage` is not `ugc_safe`, so this is a project module, or a mod whose game lists `Health.Damage` in its allow-list. Checked with `cargo check` on the host target; not built for wasm32.

```rust
use zest_guest::{Module, ObjectId, Value};

zest_guest::manifest!(
    r#"(
    name: "Hazard",
    api: 1,
    capabilities: ["read", "invoke"],
    actions: [
        (name: "Zap", label: "Damage target", params: [
            (name: "target", ty: Object),
            (name: "amount", ty: Float, default: Some(Float(10.0))),
        ]),
    ],
    properties: [(name: "zaps", ty: Int)],
)"#
);

#[derive(Default)]
struct Hazard {
    zaps: i64,
}

impl Module for Hazard {
    fn invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String> {
        if action != 0 {
            return Err(format!("no action {action}"));
        }
        let target: ObjectId = match &args[0] {
            Value::Object(id) => *id,
            _ => [0; 16],
        };
        if target == [0; 16] {
            return Err("Zap needs a target".into());
        }
        let amount = args[1].as_f32();
        if amount <= 0.0 {
            return Err("amount must be positive".into());
        }
        // `read` capability: skip the call when the target is already dead.
        if let Some(Value::Bool(true)) = zest_guest::get_property(target, "Health", "dead") {
            return Ok(());
        }
        // `invoke` capability: queue Health.Damage(amount) on the target.
        let status = zest_guest::invoke(target, "Health", "Damage", &[Value::Float(amount)]);
        if status != 0 {
            return Err(format!("Health.Damage was not queued (status {status})"));
        }
        self.zaps += 1;
        Ok(())
    }

    fn property(&mut self, index: u32) -> Option<Value> {
        (index == 0).then(|| Value::Int(self.zaps))
    }
}

zest_guest::export!(Hazard::default());
```

Put it on a hazard volume that also has `Trigger Events`. In On enter add an action: Target = the volume, Component = `Wasm Script`, Function = `Damage target`, `target` from an Event source with field `other` (the object that entered), `amount` = a Value.

How the permission checks work for this call:

1. The manifest must declare `invoke`, and the module must be allowed it: project modules are, packages never are, mods are only for calls on the game's allow-list.
2. The call is queued and then resolved like any binding. If the target has no `Health` component, or `Damage` is misspelled, the engine logs `target has no Health component` or `Health has no action 'Damage'` and nothing happens. Your `invoke` still returned 0, because it only queued.
3. For mods, the call is dropped with a warning `Mod script denied action Health.Damage` unless `Health.Damage` is on the game's allow-list. The check compares the exact text you passed, so use `Object` (not an empty string) for built-in object actions.
4. `ugc_safe` on the target action is not checked on this path. It controls who may bind an action, not who may queue it.

### Per-frame logic and events

`Ticker` shows `update` and `events`: it counts time and emits a `Tick` event at an interval. It must be exported with `export!(init, update)`. Checked with `cargo check` on the host target; not built for wasm32.

```rust
use zest_guest::{Module, Value};

zest_guest::manifest!(
    r#"(
    name: "Ticker",
    api: 1,
    capabilities: ["events"],
    actions: [
        (name: "SetInterval", label: "Set tick interval", ugc_safe: true, params: [
            (name: "seconds", ty: Float, default: Some(Float(1.0))),
        ]),
    ],
    properties: [(name: "ticks", ty: Int)],
)"#
);

struct Ticker {
    interval: f32,
    since_tick: f32,
    ticks: i64,
}

impl Default for Ticker {
    fn default() -> Self {
        Self { interval: 1.0, since_tick: 0.0, ticks: 0 }
    }
}

impl Module for Ticker {
    fn invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String> {
        match action {
            0 => {
                let seconds = args[0].as_f32();
                if seconds < 0.05 {
                    return Err(format!("interval {seconds} is too short (minimum 0.05)"));
                }
                self.interval = seconds;
                Ok(())
            }
            _ => Err(format!("no action {action}")),
        }
    }

    fn property(&mut self, index: u32) -> Option<Value> {
        (index == 0).then(|| Value::Int(self.ticks))
    }

    fn update(&mut self, dt: f32) {
        self.since_tick += dt;
        while self.since_tick >= self.interval {
            self.since_tick -= self.interval;
            self.ticks += 1;
            zest_guest::emit_event("Tick");
        }
    }
}

zest_guest::export!(Ticker::default(), update);
```

The 0.05 s minimum keeps the loop under the 16 events per call cap for frames up to 0.8 s long (after time scaling); extra events are silently dropped.

### Events

A module can emit events but cannot receive them: there is no import that delivers events to a guest. `emit_event(name)` (capability `events`) broadcasts an `ActionEvent { name, source, instigator }` on the engine's event bus: `source` is the object the module sits on, and `instigator` is who triggered the action (always none from `zest_update`). Events from an action call are delivered right after it; events from `zest_update` after the update. Native Rust gameplay reads them on the next frame, for example (a sketch; see [ECS.md](ECS.md) for the system signatures):

```rust
for event in ctx.events::<ActionEvent>() {
    if event.name == "DoorOpened" {
        // event.source is the door's object id
    }
}
```

As of this writing no built-in component reacts to a named event from the editor. To react inside a module, expose an action and have a binding call it, or poll another component's property from `zest_update` with `get_property`.

### Panics and the engine log

With `panic = "abort"`, a panic becomes an `unreachable` trap and its message is lost (a `wasm32-unknown-unknown` module has no stderr). Route it to the engine log by installing a hook in the `init` expression of `export!`; this variant of the door was checked with `cargo check` on the host:

```rust
zest_guest::export!({
    // Route panic messages to the engine log (the default message is lost on wasm32).
    std::panic::set_hook(Box::new(|info| zest_guest::log(&info.to_string())));
    Door::default()
});
```

## The legacy Script component

`Script` (`crates/zestengine-core/src/script.rs`) is the older prototype. It still works and is unchanged by the `Wasm Script` work. It hosts a module that exports `start` and/or `update`, both `() -> ()` (at least one is required), and gives it a transform and eight numbers.

### Host API

Import module `env`:

| Import | Signature | Notes |
|---|---|---|
| `env.get_dt` | `() -> f32` | Frame time. |
| `env.get_time` | `() -> f32` | Game time. |
| `env.set_position` | `(f32, f32, f32)` | Owner's local translation. |
| `env.get_position` | `() -> (f32, f32, f32)` | Local translation (multi-value result). |
| `env.set_rotation` | `(f32, f32, f32, f32)` | Local rotation quaternion x, y, z, w, stored as given, so pass a unit quaternion. |
| `env.get_rotation` | `() -> (f32, f32, f32, f32)` | Multi-value result. |
| `env.set_scale` | `(f32, f32, f32)` | Local scale. |
| `env.get_scale` | `() -> (f32, f32, f32)` | Multi-value result. |
| `env.get_prop` | `(i32) -> f32` | Read props slot 0 to 7; out of range reads 0.0. |
| `env.set_prop` | `(i32, f32)` | Write a slot; out of range is ignored. |

### Behaviour

- The transform and `props` are copied into the sandbox before each hook and written back after it. A trap leaves the object exactly as it was (a rollback, unlike `Wasm Script`).
- `start` runs once per instance, `update` once per frame in the gameplay pass.
- The module is loaded once, from `module` read as a plain file path (the project root in the editor). It does not support `@<id>` references and does not hot reload.
- Limits: 100,000 fuel per hook call, 1 MiB of linear memory, one memory and one table. The epoch deadline is configured but the engine never advances the epoch, so fuel is the only time bound.
- Any failure (missing file, invalid module, trap, out of fuel) permanently disables that instance for the session. The error is stored in a private field; it is not logged and no inspector shows it, so a broken legacy script just stops.
- There is no inspector. Edit the component in the scene file. A component is saved as its registry name plus RON data:

```ron
(kind: "Script", data: "(module:\"scripts/spin.wasm\",props:(0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0))")
```

- Content packages and mods: a legacy module has no manifest and imports from `env`, so it cannot pass the package validator, and mod scenes explicitly reject the `Script` component.

### Examples

A minimal WAT module that counts calls in two props (from the unit tests in `crates/zestengine-core/src/script.rs`):

```wat
(module
  (import "env" "get_prop" (func $get (param i32) (result f32)))
  (import "env" "set_prop" (func $set (param i32 f32)))
  (func (export "start") (call $set (i32.const 0) (f32.add (call $get (i32.const 0)) (f32.const 1))))
  (func (export "update") (call $set (i32.const 1) (f32.add (call $get (i32.const 1)) (f32.const 1)))))
```

`crates/zestengine-core/examples/scripts/spin.wat` is a fuller example that moves, rotates and scales an object. The same idea in Rust, a spinner whose turns per second come from `props[0]` (checked with `cargo check` on the host target; not built for wasm32):

```rust
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn get_time() -> f32;
    fn get_prop(index: i32) -> f32;
    fn set_rotation(x: f32, y: f32, z: f32, w: f32);
}

#[unsafe(no_mangle)]
pub extern "C" fn update() {
    unsafe {
        let half = get_time() * get_prop(0) * std::f32::consts::PI;
        set_rotation(0.0, half.sin(), 0.0, half.cos());
    }
}
```

Two practical limits when using Rust. First, stable Rust has no way to declare an import with several results, so `get_position`, `get_rotation` and `get_scale` are out of reach; keep your own copy of the values in a prop or a `static`, or write that module in WAT. Second, a default `wasm32-unknown-unknown` build reserves a 1 MiB stack in linear memory, which probably puts the declared memory above the legacy 1 MiB cap; shrink it with `-C link-arg=-zstack-size=65536`. Neither point was tested.

## Limits and safety

### What a module can and cannot do

A module can compute, keep state in its own memory, read the properties the engine lets it read, emit events and queue actions. It cannot open a file or socket, read a clock other than `get_time`, get random numbers, load code, or see anything the capabilities do not give it. The import allowlist is checked before instantiation, so an unknown import fails the load rather than failing later. The engine also turns on NaN canonicalisation and turns off relaxed SIMD, so float results do not depend on the CPU.

### Limits by trust level

| Limit | Project module | Package or mod | Legacy `Script` |
|---|---|---|---|
| Module file | 4 MiB | 1 MiB (packages also obey the content kind: World 1 MiB, Prop 512 KiB; Avatars carry none) | none |
| Linear memory (declared and grown) | 256 pages (16 MiB) | 64 pages (4 MiB) | 1 MiB |
| Fuel per entry-point call | 5,000,000 | 1,000,000 | 100,000 |
| Guest call stack | 512 KiB | 512 KiB (see note) | wasmtime default |
| Capabilities | all | `state`, `events`, `read` (mods also `invoke`) | n/a |
| Modules per package | n/a | World 16, Prop 4, Avatar 0 | n/a |

Fuel is wasmtime's instruction counter: roughly one unit per executed instruction, so it is deterministic and independent of CPU speed. Each entry-point call starts with a full budget: `zest_alloc` plus `zest_invoke` share one, `zest_read_property` and `zest_update` get their own, and so does instantiation (a start function runs inside it). Running out ends the call with a trap. There is no wall-clock deadline for `Wasm Script`; the engine has no epoch timer for it. A frame with many scripts can still add up to many full budgets, so keep `zest_update` cheap.

The stack note: `SandboxLimits::untrusted()` declares 256 KiB, but the engine configures one stack limit for all modules, the trusted 512 KiB (`crates/zestengine-core/src/wasm/sandbox.rs`). Do not rely on the smaller number.

Why each limit exists:

- File size bounds compile time and package size.
- The memory cap stops a module allocating the host out of RAM. Growth past the cap returns -1 to the guest (Rust's allocator then aborts, which traps).
- Fuel stops an infinite loop from freezing the frame, since guest code runs synchronously on the game thread.
- The stack cap turns deep recursion into a trap instead of a host crash.
- Per-call caps (32 log lines, 16 events, 16 queued calls, 64 float keys, 256-byte strings) keep one call from flooding the host.

### Trust by location

The engine does not trust a manifest to say how trusted it is; trust follows where the file is.

- Anything under a project folder is trusted.
- Anything under an extracted content package folder is untrusted, with tighter limits. The content manager registers the package's `files` folder when it extracts it.
- Anything under a mod's folder in the session snapshot, or a `.wasm` a mod replaced, is untrusted with the mod rules below.

Untrusted modules cannot declare `invoke` (the load fails with `capabilities not permitted here: invoke`), and they can only expose actions whose manifest sets `ugc_safe: true`, and only when the content kind allows module actions at all (`allow_wasm_actions`: true for World and Prop, false for Avatar).

### ugc_safe

`ugc_safe` is an opt-in on each action: "binding this from user-generated content is acceptable". Mark an action safe only if it changes nothing outside the module, cannot hurt other players and cannot leave the world. Counting and toggling a flag are safe; dealing damage or loading a scene is not.

How it is enforced today:

- For `Wasm Script` actions in untrusted content, the runtime refuses a non-safe action with `'<name>' is not allowed for untrusted content (needs ugc_safe and content limits that allow module actions)`, and hides every action when the content kind disallows module actions.
- In the editor, when you edit an object that came from a content package or while the Content Builder is open, the action dropdowns offer only `ugc_safe` actions, and an existing binding to any other action is flagged `Not allowed in user-generated content`. This applies to built-in actions too, such as `Health.Damage`.
- For built-in actions, as of this writing no runtime or load-time check rejects a package scene's binding to an action that is not `ugc_safe`. For those, treat the editor filter as authoring help, not as a security boundary.

### Mod permissions and the action allow-list

A mod's modules get the untrusted limits but are allowed the `invoke` capability, because a game may want mods to do things like deal damage. The game decides which actions:

- In Project Settings > Mods the game lists exact `Component.Action` names in the allow-list (`mods.actions` in `project.zestengine`), for example `Health.Damage`.
- A mod's own manifest (`actions`) lists the ones it needs. The mod is refused when the profile is validated or mounted if the game has not exposed one of them.
- At run time every queued `invoke` is checked against the mod's list: a call whose `Component.Action` is not on it is dropped with a warning `Mod script denied action Component.Action`.
- A mod cannot grant itself anything: the game's list (Project Settings > Mods) is the ceiling.

Further mod rules in the code: a mod module cannot be hot reloaded (it runs from the session snapshot); a mod's scenes may not use the legacy `Script` component; a mod's `Wasm Script` components are assigned the Server network role; and on a connected client a mod module's `zest_update` does not run. A mod module's own actions still need `ugc_safe` to be callable from bindings.

### Networking and dedicated servers

A `Wasm Script` has the network role `Both` by default: it exists and runs on every peer, including a dedicated server, and each peer has its own instance. Its state is not replicated, and it is not saved with the scene. Choose where it runs from the component's options menu (Network Role: Both, Client or Server), or project-wide in Project Settings > Network roles. The dedicated-server build keeps `.wasm` files.

For gameplay that must be authoritative, give the module the Server role and put the outcome into the engine's replication tools, for example Network Variables or a spawned network object (see the networking components in the [ECS Guide](ECS.md) and the Network Variables and RPC calls in the [Engine API Guide](ENGINE_API.md#networking)). Module state is not shared between peers, so two peers running the same module will not stay in sync by themselves.

## Using WASM from the editor and from scenes

### Adding the component

Select an object, click Add Component and choose `Wasm Script`. The inspector shows:

- Module: an asset field for `.wasm` files (drag one from the file browser, or browse). It stores an `@<id>` reference.
- The module's name from its manifest, or a red error message if loading failed. A note appears when the module comes from untrusted content.
- One line per declared action, `Name(param: Type, ...)`, with `(not UGC-safe)` on actions without the flag, then `property name: Type` lines and the capability list.
- Reload module, which re-reads the file and replaces the instance. Opening the inspector also re-checks the file, so a rebuild shows up while it is open.

You can add several `Wasm Script` components to one object, each with its own module.

### Binding actions

The action editor appears on `Interactable` (On interact, When turned on, When turned off, On press, On release), `Trigger Events` (On enter, On exit), `UI Button` (On click: call) and `Animation Event Bindings` (Runs, one list per named event). Each row has:

- Target: which object (empty means the object that owns the event source).
- Component: `Object` (the built-in object actions) or a component on the target. Components with several instances are numbered, for example `Wasm Script #2 (door)`.
- Function: the action's label.
- One row per parameter, each with a source: Value (a constant), Property (a property of any component on any object, including a `Wasm Script` property), Instigator (a property of whoever triggered the event) or Event (a field of the event: Interactable offers `instigator`, `object`, `uses`, `on`; Trigger Events offers `other`, `object`; animation events offer `name`, `object`).
- A Test button (Play mode only) that fires the row as if the event had happened and shows `Fired` or the resolve error under the row (a failure inside the module is only logged), and buttons to reorder or remove the row. Warnings appear under a row when a source does not resolve or its type cannot convert to the parameter's type.

Missing arguments use the manifest defaults. Types convert as described in [Value encoding](#value-encoding). A binding that cannot resolve (target gone, action missing) logs once and does nothing.

In a scene file the binding is plain RON inside the source component's data, for example (copied from a serialised `Interactable`, abbreviated):

```ron
on_interact:[
  (target:(Some("37daca0c-5c38-460b-8d74-840effdf98fa")),component:"Wasm Script",action:"Add",args:[Const(Int(5))],instance:0,instance_label:""),
  (target:(Some("37daca0c-5c38-460b-8d74-840effdf98fa")),component:"Wasm Script",action:"Add",args:[Instigator(component:"Health",property:"current")],instance:1,instance_label:"counter"),
]
```

Scenes store names only (`component`, `action`), never action indices, so reordering actions in a module does not break existing bindings.

### Instance labels

When an object has several components of one type, a binding stores the position among them (`instance`) and a label (`instance_label`). For a `Wasm Script` the label is the module's file name without folders or extension (`door` for `scripts/door.wasm`). When resolving, a matching label wins over the position, so a binding keeps following its module when you reorder or remove components.

The label is empty when the `module` field holds an `@<id>` reference, which is what the editor's asset picker stores. Bindings to such components follow the position only. If you rely on labels, write the plain path in the scene.

## Testing and debugging

### Unit-testing guest logic on the host

`zest-guest` compiles on your development machine with stub host imports: they do nothing and return zero, except `get_property` and `invoke`, which report -2. So `cargo test` in the guest crate works for logic that does not depend on the host. A test added to the end of `guest/example-counter/src/lib.rs` (ran and passed on the host):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_reset() {
        let mut c = Counter::default();
        c.invoke(0, &[Value::Int(5)]).unwrap();
        c.invoke(0, &[Value::Int(2)]).unwrap();
        assert_eq!(c.property(0), Some(Value::Int(7)));
        c.invoke(1, &[]).unwrap();
        assert_eq!(c.property(0), Some(Value::Int(0)));
        assert!(c.invoke(9, &[]).is_err());
    }
}
```

### Testing a module in the real sandbox

The engine's tests exercise modules without a wasm32 toolchain by writing them in WAT and assembling them with the `wat` crate; helpers are in `crates/zestengine-core/src/wasm/fixtures.rs` (compiled only in the engine's own test builds, so copy what you need) and the tests in `crates/zestengine-core/src/wasm/tests.rs`. A `build(manifest, body)` helper wraps a body in `(module (@custom "zest_manifest" "...") ...)`, and fixtures such as `counter`, `DRIVER` (reads `Health.current` and queues `Health.Damage`) and `UPDATER` (logs, stores `dt` and `time`) show every host import in use.

You can do the same from your own crate, with `zestengine-core` and `wat` as dependencies. This test loads the `Flag` module from [Writing modules in other languages](#writing-modules-in-other-languages) (`FLAG_WAT`) the way the engine does, once with the package validator's untrusted limits and once as a project module:

```rust
use zestengine_core::actions::ArgValue;
use zestengine_core::wasm::sandbox::ALL_CAPS;
use zestengine_core::wasm::{inspect_module, Instance, SandboxLimits, UNTRUSTED_CAPS};
use zestengine_core::ObjectId;

#[test]
fn flag_module_works_and_passes_the_package_validator() {
    let wasm = wat::parse_str(FLAG_WAT).unwrap();
    // What the package builder checks: untrusted limits, no `invoke`.
    inspect_module(&wasm, &SandboxLimits::untrusted(), UNTRUSTED_CAPS).unwrap();
    let (mut inst, info) = Instance::load(&wasm, SandboxLimits::trusted(), ALL_CAPS, "flag").unwrap();
    assert_eq!(info.manifest.name, "Flag");
    assert_eq!(inst.call_read_property(0).unwrap(), Some(ArgValue::Bool(false)));
    inst.call_invoke(0, &[], ObjectId::NIL, None, None).unwrap();
    assert_eq!(inst.call_read_property(0).unwrap(), Some(ArgValue::Bool(true)));
}
```

To test a built module, read it from disk instead (`std::fs::read("logic/counter.wasm")`): the calls are the same, `call_update(dt, time, None, ObjectId::NIL)` runs `zest_update`, and `inst.float("key")` reads the `state` store.

The repository's own tests:

```bash
cargo test -p zestengine-core --lib --locked wasm      # 29 tests: ABI, sandbox, component, bindings
cargo test -p zestengine-core --lib --locked script    # legacy Script and mod-permission tests
```

### Logging

`zest_guest::log` (`host.log`) writes an INFO line with tracing target `wasm`, prefixed with the `module` field's value: `<module>: <message>`. Lines go to stdout (the level honours `RUST_LOG`, default `info`) and appear in the editor's Log tab. At most 32 lines per call, 512 bytes each. To watch a value without a debugger, log it, or bind a `UI Text.SetText` parameter to a module property in Play mode.

A module that traps leaves one `ERROR` line: `Wasm Script "<module>": <what>: <reason>; the component is disabled until its module changes`. Legacy `Script` failures are not logged at all.

### Common failures

Load errors appear in red in the inspector and as one `ERROR` line `Wasm Script "<module>": <message>`. The strings below are the engine's real ones.

| Message | Cause and fix |
|---|---|
| `<path>: No such file or directory (os error 2)` | The path is wrong or relative to the wrong folder. It is read relative to the working directory (the project root in the editor). |
| `not a WebAssembly binary` | The file is not a `.wasm` module (a text file, an HTML error page, a `.wat`). |
| `no "zest_manifest" custom section (see the Wasm Script docs)` | The section is missing or was stripped. Use `zest_guest::manifest!` or the WAT `@custom` form, and do not strip custom sections. |
| `more than one "zest_manifest" section` | The manifest was embedded twice (two `manifest!` calls, or a dependency embeds one). |
| `manifest does not parse: 1:2: Unexpected end of RON` | RON syntax error in the manifest. The position is within the manifest text. |
| `manifest api 9 is newer than this engine's 1` | Set `api: 1`. |
| `unknown capability "fs" (known: state, events, read, invoke)` | A capability name is misspelled or does not exist. |
| `action "A" is declared twice` / `property "p" is declared twice` | Names must be unique within their list. |
| `property name "a/b" has characters other than letters, digits, space, _ - .` | Names allow only those characters, 1 to 64 bytes (the same text appears for `action` and `parameter`). |
| `action "A" parameter "x": default is Float, not Int` | A `default` must be the same type as `ty`. |
| `imports env.x; only the "host" namespace is available` | The module imports from another namespace. WASI imports (`wasi_snapshot_preview1`) produce the same message; the target must be `wasm32-unknown-unknown`, not a WASI target. |
| `imports unknown host function "system"` | The name is not in the import list above. |
| `imports host.emit_event but the manifest does not declare the "events" capability` | Add the capability. |
| `capabilities not permitted here: invoke` | An untrusted module (a package) declared `invoke`. Remove it, or ship the logic as a project module or an approved mod. |
| `memory starts at 300 pages (19200 KiB); limit 256 pages` | The module's declared initial memory is over the cap (64 pages for untrusted). Reduce the initial memory or stack size. |
| `the module must export exactly one memory named "memory"` | The memory is not exported, or is exported twice. |
| `the manifest declares actions but the module exports no zest_invoke` | Also appears for `zest_alloc` and (for properties) `zest_read_property`. Add the export, or use `zest_guest::export!`. |
| `zest_invoke / zest_alloc have the wrong signature (see the ABI docs)` | An export exists with the wrong parameter or result types. |
| `module is N bytes (limit M)` | The file exceeds the size cap for its trust level. |
| `invalid module: <wasmtime message>` | The bytes start like a module but do not validate or compile: a truncated or corrupt file, or an unsupported feature. |
| `instantiating: ran past its instruction budget (infinite loop?)` | A start function ran out of fuel. |
| `too many distinct module manifests loaded (512); restart to load more` | Hit the process-wide manifest cap. |

Errors while running:

| Message | Cause and fix |
|---|---|
| `<action>: ran past its instruction budget (infinite loop?); the component is disabled ...` | The call exceeded the fuel cap. Remove the loop or cut the work per call. |
| `<action>: call stack exhausted; ...` | Runaway recursion. |
| ``<action>: wasm trap: wasm `unreachable` instruction executed; ...`` | A panic (with `panic = "abort"`), `unreachable!`, a failed assertion or `index out of bounds`. Install the panic hook above to see the message. |
| `<action>: wasm trap: out of bounds memory access; ...` | A bad pointer in the guest. |
| `<action>: error while executing at wasm backtrace:; ...` | A host import rejected its arguments: a string over 256 bytes (event names, keys, component and action names) or a pointer outside guest memory. The engine currently keeps only the first line of the error, so the cause is not shown. |
| `Wasm Script has no module loaded: <reason>` | A binding fired but the module failed to load. `no module set` means the field is empty. |
| `Wasm Script has no action 'X'` | The binding names an action the manifest does not declare (renamed or removed). |
| `'X' is not allowed for untrusted content (needs ugc_safe and content limits that allow module actions)` | See [ugc_safe](#ugc_safe). |
| `<message from zest_last_error>` or `returned status N` | The action returned a non-zero status. |
| `arguments are too large` | The encoded arguments are over 64 KiB. |
| `Mod script denied action Component.Action` | A mod called something the game's allow-list does not include. |
| `Wasm Script property '<name>': <reason>` | `zest_read_property` returned a malformed value (for example trailing bytes or an unknown tag). |

Two quiet mistakes: a module whose `zest_update` has the wrong signature is accepted and never updated, and a manifest typo in an optional field name is ignored. If something silently does nothing, check these first.

## Reference

The manifest fields and their limits are in [Manifest](#manifest). All imports below come from the module `host`, except the legacy `env` ones.

### Value tags

| Tag | Type | Payload |
|---|---|---|
| 0 | Bool | 1 byte |
| 1 | Int | i64 |
| 2 | Float | f32 |
| 3 | String | u32 length (at most 16384) + bytes |
| 4 | Vec3 | 3 x f32 |
| 5 | Object | 16 bytes, zero = none |

Argument block: `u32` count (at most 32), then values. At most 64 KiB.

### Wasm Script exports

| Export | Signature |
|---|---|
| `memory` | memory (32-bit, one) |
| `zest_alloc` | `(i32) -> i32` |
| `zest_invoke` | `(i32, i32, i32) -> i32` |
| `zest_last_error` | `() -> i64` (optional) |
| `zest_read_property` | `(i32) -> i64` |
| `zest_update` | `(f32)` (optional) |

### Wasm Script imports

| Import | Signature | Capability |
|---|---|---|
| `log` | `(i32, i32)` | none |
| `get_dt` | `() -> f32` | none |
| `get_time` | `() -> f32` | none |
| `get_self` | `(i32)` | none |
| `get_instigator` | `(i32) -> i32` | none |
| `get_float` | `(i32, i32) -> f32` | `state` |
| `set_float` | `(i32, i32, f32)` | `state` |
| `emit_event` | `(i32, i32)` | `events` |
| `get_property` | `(i32, i32, i32, i32, i32, i32, i32) -> i32` | `read` |
| `invoke` | `(i32, i32, i32, i32, i32, i32, i32) -> i32` | `invoke` |

### Legacy Script exports and imports

| Name | Signature |
|---|---|
| export `start`, `update` | `() -> ()` (at least one) |
| `env.get_dt`, `env.get_time` | `() -> f32` |
| `env.set_position`, `env.set_scale` | `(f32, f32, f32)` |
| `env.get_position`, `env.get_scale` | `() -> (f32, f32, f32)` |
| `env.set_rotation` | `(f32, f32, f32, f32)` |
| `env.get_rotation` | `() -> (f32, f32, f32, f32)` |
| `env.get_prop` | `(i32) -> f32` |
| `env.set_prop` | `(i32, f32)` |

### Per-call caps

| Cap | Value |
|---|---|
| Strings passed to host imports | 256 bytes (`log` 512, cut instead of trapping) |
| Log lines per entry-point call | 32 |
| Events per entry-point call | 16 |
| Queued `invoke` calls | 16 |
| `state` float keys | 64 |
| String value | 16384 bytes |
| Arguments per call | 32 |
| Argument block | 64 KiB |
| Manifest text | 64 KiB |
| Distinct manifests per process | 512 |

### Source map

| What | Where |
|---|---|
| ABI codec and manifest | `crates/zestengine-core/src/wasm/abi.rs` |
| Sandbox, imports, limits, capabilities | `crates/zestengine-core/src/wasm/sandbox.rs` |
| The `Wasm Script` component | `crates/zestengine-core/src/wasm/component.rs` |
| Trust roots, mod roots | `crates/zestengine-core/src/wasm/mod.rs` |
| Test fixtures and tests | `crates/zestengine-core/src/wasm/{fixtures,tests}.rs` |
| Legacy `Script` | `crates/zestengine-core/src/script.rs` |
| Bindings and actions | `crates/zestengine-core/src/actions.rs` |
| Mod contracts | `crates/zestengine-core/src/mods.rs` |
| Package validation | `crates/zestengine-assets/src/package.rs` (`validate_wasm_module`) |
| Editor inspector and binding UI | `crates/zestengine-editor/src/{wasm_ui,actions_ui}.rs` |
| Rust guest helper and example | `guest/zest-guest`, `guest/example-counter` |
