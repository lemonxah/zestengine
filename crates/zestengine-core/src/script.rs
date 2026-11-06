//! Sandboxed user-generated logic: the `Script` component.
//!
//! A `Script` hosts one WebAssembly module (compiled by `wasmtime`) and calls
//! its exported `start` / `update` hooks. The module sees only the narrow
//! imported-function API below; no WASI is linked, so guest code has no
//! file-system or network access. Execution is bounded per hook call with
//! fuel, and linear-memory growth is capped.
//!
//! # Host API (import module `env`)
//!
//! | import                        | signature                          |
//! |-------------------------------|------------------------------------|
//! | `env.get_dt`                  | `() -> f32`                        |
//! | `env.get_time`                | `() -> f32`                        |
//! | `env.set_position`            | `(f32, f32, f32) -> ()`            |
//! | `env.get_position`            | `() -> (f32, f32, f32)`            |
//! | `env.set_rotation`            | `(f32, f32, f32, f32) -> ()`       |
//! | `env.get_rotation`            | `() -> (f32, f32, f32, f32)`       |
//! | `env.set_scale`               | `(f32, f32, f32) -> ()`            |
//! | `env.get_scale`               | `() -> (f32, f32, f32)`            |
//! | `env.get_prop`                | `(i32) -> f32`                     |
//! | `env.set_prop`                | `(i32, f32) -> ()`                 |
//!
//! Position/rotation/scale are the owning object's **local** transform (the
//! same fields `ComponentCtx` exposes). The eight `props` are the component's
//! own numeric fields, so scripts have some per-object state. Reads happen
//! against a copy written into the store before each hook and written back
//! afterwards, so a trap leaves the object unchanged.
//!
//! # Module format
//!
//! The `module` field is a path to a `.wasm` binary. Content packages have
//! packaged paths rewritten to absolute extraction paths at load time
//! (see `docs/user-content.md`); base-game scenes resolve relative to the
//! process working directory. The module must export at least one of
//! `start` or `update` (both `() -> ()`).

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ComponentCtx, TypedComponent};

/// Per-hook fuel budget. Wasmtime consumes one unit per instruction, so this
/// is a hard bound on guest work per `start` / `update` call.
const FUEL_PER_CALL: u64 = 100_000;

/// Cap on guest linear-memory growth (1 MiB). A module whose declared or grown
/// memory exceeds this fails to instantiate / grow.
const MEMORY_LIMIT_BYTES: usize = 1 << 20;

/// Epoch-deadline backstop. The engine never increments the epoch, so this is
/// wired but not a wall-clock limit yet; fuel is the active per-frame bound.
const EPOCH_DEADLINE_TICKS: u64 = 1 << 40;

/// Number of `props` slots exposed through `env.get_prop` / `env.set_prop`.
pub const SCRIPT_PROP_COUNT: usize = 8;

/// The current host API version. Future package manifests can refuse modules
/// written against a newer ABI (see `docs/user-content.md`).
pub const SCRIPT_API_VERSION: u32 = 1;

#[derive(Clone, Copy)]
enum Hook {
    Start,
    Update,
}

/// Host state owned by a script's `wasmtime::Store`. Imported functions read
/// and write this; the component copies it in before a hook and out after it.
struct ScriptState {
    dt: f32,
    time: f32,
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
    props: [f32; SCRIPT_PROP_COUNT],
    limits: wasmtime::StoreLimits,
}

impl ScriptState {
    fn new() -> Self {
        Self {
            dt: 0.0,
            time: 0.0,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            props: [0.0; SCRIPT_PROP_COUNT],
            limits: wasmtime::StoreLimitsBuilder::new()
                .memory_size(MEMORY_LIMIT_BYTES)
                .memories(1)
                .instances(1)
                .tables(1)
                .build(),
        }
    }
}

/// A sandboxed WebAssembly behaviour component (see module docs).
#[derive(Serialize, Deserialize)]
pub struct Script {
    /// Path to the `.wasm` module (absolute after content extraction).
    pub module: String,
    /// Script-visible numeric state (`env.get_prop` / `env.set_prop`).
    pub props: [f32; SCRIPT_PROP_COUNT],
    #[serde(skip)]
    engine: Option<wasmtime::Engine>,
    #[serde(skip)]
    store: Option<wasmtime::Store<ScriptState>>,
    #[serde(skip)]
    update: Option<wasmtime::TypedFunc<(), ()>>,
    #[serde(skip)]
    start: Option<wasmtime::TypedFunc<(), ()>>,
    /// Last compile / instantiate / trap error; the script is disabled once set.
    #[serde(skip)]
    error: Option<String>,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            module: String::new(),
            props: [0.0; SCRIPT_PROP_COUNT],
            engine: None,
            store: None,
            update: None,
            start: None,
            error: None,
        }
    }
}

impl Script {
    /// Compile `bytes` and instantiate it against the `env` API. Exposed as a
    /// private helper so the loader (path) and the tests (raw bytes) share one
    /// code path.
    fn build_runtime(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        config.epoch_interruption(true);

        let engine = wasmtime::Engine::new(&config).map_err(|e| e.to_string())?;
        let module = wasmtime::Module::new(&engine, bytes).map_err(|e| e.to_string())?;
        let mut linker = wasmtime::Linker::<ScriptState>::new(&engine);

        linker
            .func_wrap("env", "get_dt", |caller: wasmtime::Caller<'_, ScriptState>| {
                caller.data().dt
            })
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap("env", "get_time", |caller: wasmtime::Caller<'_, ScriptState>| {
                caller.data().time
            })
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap(
                "env",
                "set_position",
                |mut caller: wasmtime::Caller<'_, ScriptState>, x: f32, y: f32, z: f32| {
                    caller.data_mut().translation = Vec3::new(x, y, z);
                },
            )
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap("env", "get_position", |caller: wasmtime::Caller<'_, ScriptState>| {
                let p = caller.data().translation;
                (p.x, p.y, p.z)
            })
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap(
                "env",
                "set_rotation",
                |mut caller: wasmtime::Caller<'_, ScriptState>, x: f32, y: f32, z: f32, w: f32| {
                    caller.data_mut().rotation = Quat::from_xyzw(x, y, z, w);
                },
            )
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap("env", "get_rotation", |caller: wasmtime::Caller<'_, ScriptState>| {
                let q = caller.data().rotation;
                (q.x, q.y, q.z, q.w)
            })
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap(
                "env",
                "set_scale",
                |mut caller: wasmtime::Caller<'_, ScriptState>, x: f32, y: f32, z: f32| {
                    caller.data_mut().scale = Vec3::new(x, y, z);
                },
            )
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap("env", "get_scale", |caller: wasmtime::Caller<'_, ScriptState>| {
                let s = caller.data().scale;
                (s.x, s.y, s.z)
            })
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap(
                "env",
                "get_prop",
                |caller: wasmtime::Caller<'_, ScriptState>, index: i32| {
                    caller.data().props.get(index as usize).copied().unwrap_or(0.0)
                },
            )
            .map_err(|e| e.to_string())?;
        linker
            .func_wrap(
                "env",
                "set_prop",
                |mut caller: wasmtime::Caller<'_, ScriptState>, index: i32, value: f32| {
                    if let Some(slot) = caller.data_mut().props.get_mut(index as usize) {
                        *slot = value;
                    }
                },
            )
            .map_err(|e| e.to_string())?;

        let mut store = wasmtime::Store::new(&engine, ScriptState::new());
        store.limiter(|state| &mut state.limits);
        store.epoch_deadline_trap();
        store.set_epoch_deadline(EPOCH_DEADLINE_TICKS);

        let instance = linker.instantiate(&mut store, &module).map_err(|e| e.to_string())?;
        let update = instance.get_typed_func::<(), ()>(&mut store, "update").ok();
        let start = instance.get_typed_func::<(), ()>(&mut store, "start").ok();
        if update.is_none() && start.is_none() {
            return Err("Script module exports neither `start` nor `update`".into());
        }

        self.engine = Some(engine);
        self.store = Some(store);
        self.update = update;
        self.start = start;
        Ok(())
    }

    /// Load and instantiate the module named by `self.module`, once.
    fn ensure_loaded(&mut self) {
        if self.engine.is_some() || self.error.is_some() {
            return;
        }
        if self.module.trim().is_empty() {
            self.error = Some("Script module path is empty".into());
            return;
        }
        let bytes = match std::fs::read(&self.module) {
            Ok(bytes) => bytes,
            Err(e) => {
                self.error = Some(format!("Script {}: {e}", self.module));
                return;
            }
        };
        if let Err(e) = self.build_runtime(&bytes) {
            self.error = Some(e);
        }
    }

    /// Run one exported hook (`start` / `update`) against the current frame.
    fn run_hook(&mut self, ctx: &mut ComponentCtx, hook: Hook) {
        if self.error.is_some() {
            return;
        }
        let func = match hook {
            Hook::Start => self.start.clone(),
            Hook::Update => self.update.clone(),
        };
        let Some(func) = func else { return };
        let Some(store) = self.store.as_mut() else { return };

        {
            let state = store.data_mut();
            state.dt = ctx.dt;
            state.time = ctx.time;
            state.translation = *ctx.translation;
            state.rotation = *ctx.rotation;
            state.scale = *ctx.scale;
            state.props = self.props;
        }

        if let Err(e) = store.set_fuel(FUEL_PER_CALL) {
            self.error = Some(format!("Script fuel: {e}"));
            return;
        }
        if let Err(e) = func.call(&mut *store, ()) {
            self.error = Some(e.to_string());
            return;
        }

        let state = store.data();
        *ctx.translation = state.translation;
        *ctx.rotation = state.rotation;
        *ctx.scale = state.scale;
        self.props = state.props;
    }
}

impl TypedComponent for Script {
    const NAME: &'static str = "Script";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.ensure_loaded();
        self.run_hook(ctx, Hook::Start);
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        self.ensure_loaded();
        self.run_hook(ctx, Hook::Update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnyBox, Component, ComponentCommand, ComponentLookup, InputState, Localization, NetView, ObjectId, SaveGame,
        ScreenInfo,
    };
    use glam::Mat4;
    use std::collections::HashMap;
    use std::any::TypeId;

    struct DummyLookup;
    impl ComponentLookup for DummyLookup {
        fn component(&self, _object: usize, _type_id: TypeId) -> Option<&dyn Component> {
            None
        }
    }

    struct Rig {
        commands: Vec<ComponentCommand>,
        translation: Vec3,
        rotation: Quat,
        scale: Vec3,
        id_index: HashMap<ObjectId, usize>,
        active: Vec<bool>,
        screen: ScreenInfo,
        save: SaveGame,
        strings: Localization,
        input: InputState,
        net: NetView,
        world_transforms: Vec<Mat4>,
        object_names: Vec<String>,
        object_ids: Vec<ObjectId>,
        spawn_points: Vec<(usize, String)>,
        events: Vec<AnyBox>,
        lookup: DummyLookup,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                commands: Vec::new(),
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                id_index: HashMap::new(),
                active: vec![true],
                screen: ScreenInfo::default(),
                save: SaveGame::new(),
                strings: Localization::default(),
                input: InputState::default(),
                net: NetView::default(),
                world_transforms: vec![Mat4::IDENTITY],
                object_names: vec![String::new()],
                object_ids: vec![ObjectId::NIL],
                spawn_points: Vec::new(),
                events: Vec::new(),
                lookup: DummyLookup,
            }
        }

        fn ctx(&mut self) -> ComponentCtx<'_> {
            ComponentCtx {
                dt: 1.0 / 60.0,
                unscaled_dt: 1.0 / 60.0,
                time: 1.0,
                water_time: 0.0,
                translation: &mut self.translation,
                rotation: &mut self.rotation,
                scale: &mut self.scale,
                commands: &mut self.commands,
                world_transforms: &self.world_transforms,
                object_names: &self.object_names,
                object_ids: &self.object_ids,
                self_index: 0,
                parent_world: None,
                input: &self.input,
                net: &self.net,
                spawn_points: &self.spawn_points,
                id_index: &self.id_index,
                active: &self.active,
                lookup: &self.lookup,
                physics: None,
                nav: None,
                materials: None,
                events: &self.events,
                time_scale: 1.0,
                screen: &self.screen,
                save: &self.save,
                strings: &self.strings,
            }
        }
    }

    const SPIN_WAT: &str = include_str!("../examples/scripts/spin.wat");

    #[test]
    fn spin_example_runs_start_and_update() {
        let mut script = Script {
            module: "spin.wat".into(),
            ..Default::default()
        };
        script.build_runtime(&wat::parse_str(SPIN_WAT).unwrap()).unwrap();

        let mut rig = Rig::new();
        TypedComponent::start(&mut script, &mut rig.ctx());

        assert_eq!(script.error, None);
        assert_eq!(rig.scale, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(rig.translation, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(script.props[0], 0.0);

        TypedComponent::update(&mut script, &mut rig.ctx());
        assert!(script.props[0] > 0.0, "update accumulated dt into props[0]");
        assert!(
            (script.props[0] - 1.0 / 60.0).abs() < 1e-5,
            "props[0] should be one frame of dt, got {}",
            script.props[0]
        );
        assert!(
            (rig.translation.x - 2.0 / 60.0).abs() < 1e-4,
            "update should advance x by 2*dt, got {:?}",
            rig.translation
        );
        assert_eq!(rig.translation.y, 1.0);
        assert_eq!(rig.translation.z, 0.0);
        assert_eq!(rig.rotation, Quat::from_xyzw(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn wasi_imports_are_rejected() {
        let wasi_guest = r#"(module
            (import "wasi_snapshot_preview1" "fd_write"
                (func (param i32 i32 i32 i32) (result i32)))
            (func (export "update")))"#;

        let mut script = Script::default();
        let err = script.build_runtime(&wat::parse_str(wasi_guest).unwrap()).unwrap_err();
        assert!(err.contains("unknown import"), "unexpected error: {err}");
    }

    #[test]
    fn oversized_memory_is_rejected() {
        // 64 pages = 4 MiB, above the 1 MiB store limit.
        let greedy = r#"(module (memory 64) (func (export "update")))"#;

        let mut script = Script::default();
        let err = script.build_runtime(&wat::parse_str(greedy).unwrap()).unwrap_err();
        assert!(err.to_lowercase().contains("memory"), "unexpected error: {err}");
    }
}
