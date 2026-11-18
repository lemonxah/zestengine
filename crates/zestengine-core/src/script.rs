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
//! fields stored in ECS). The eight `props` are the component's
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

use zestengine_scripting::{SandboxLimits, wasmtime};

use crate::{Transform, TypedComponent};
use crate::ecs::{ObjectContext, Stage};

const LIMITS: SandboxLimits = SandboxLimits::legacy_script();

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
            limits: LIMITS.store_limits(1),
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
    store: Option<wasmtime::Store<ScriptState>>,
    #[serde(skip)]
    update: Option<wasmtime::TypedFunc<(), ()>>,
    #[serde(skip)]
    start: Option<wasmtime::TypedFunc<(), ()>>,
    /// Last compile / instantiate / trap error; the script is disabled once set.
    #[serde(skip)]
    error: Option<String>,
    /// Startup belongs to this native instance, including late additions.
    #[serde(skip)]
    started: bool,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            module: String::new(),
            props: [0.0; SCRIPT_PROP_COUNT],
            store: None,
            update: None,
            start: None,
            error: None,
            started: false,
        }
    }
}

fn shared_linker() -> Result<&'static wasmtime::Linker<ScriptState>, String> {
    static LINKER: std::sync::OnceLock<Result<wasmtime::Linker<ScriptState>, String>> = std::sync::OnceLock::new();
    LINKER.get_or_init(build_linker).as_ref().map_err(Clone::clone)
}

fn build_linker() -> Result<wasmtime::Linker<ScriptState>, String> {
    let mut linker = wasmtime::Linker::<ScriptState>::new(zestengine_scripting::engine()?);
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
    Ok(linker)
}

impl Script {
    /// Compile `bytes` and instantiate it against the `env` API. Exposed as a
    /// private helper so the loader (path) and the tests (raw bytes) share one
    /// code path.
    fn build_runtime(&mut self, bytes: &[u8]) -> Result<(), String> {
        zestengine_scripting::check_size(bytes, &LIMITS)?;
        let engine = zestengine_scripting::engine()?;
        let module = zestengine_scripting::compile(bytes)?;
        let mut store = wasmtime::Store::new(engine, ScriptState::new());
        store.limiter(|state| &mut state.limits);
        store.set_fuel(LIMITS.fuel_per_call).map_err(|e| e.to_string())?;

        let instance = shared_linker()?.instantiate(&mut store, &module).map_err(|e| e.to_string())?;
        let update = instance.get_typed_func::<(), ()>(&mut store, "update").ok();
        let start = instance.get_typed_func::<(), ()>(&mut store, "start").ok();
        if update.is_none() && start.is_none() {
            return Err("Script module exports neither `start` nor `update`".into());
        }

        self.store = Some(store);
        self.update = update;
        self.start = start;
        Ok(())
    }

    /// Load and instantiate the module named by `self.module`, once.
    fn ensure_loaded(&mut self) {
        if self.store.is_some() || self.error.is_some() {
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
    fn run_hook(&mut self, ctx: &ObjectContext<'_, '_>, hook: Hook) {
        if self.error.is_some() {
            return;
        }
        let func = match hook {
            Hook::Start => self.start.clone(),
            Hook::Update => self.update.clone(),
        };
        let Some(func) = func else { return };
        let Some(store) = self.store.as_mut() else { return };

        let Some(mut transform) = ctx.world.get_mut::<Transform>(ctx.owner) else { return };
        {
            let state = store.data_mut();
            state.dt = ctx.dt;
            state.time = ctx.time;
            state.translation = transform.translation;
            state.rotation = transform.rotation;
            state.scale = transform.scale;
            state.props = self.props;
        }

        if let Err(e) = store.set_fuel(LIMITS.fuel_per_call) {
            self.error = Some(format!("Script fuel: {e}"));
            return;
        }
        if let Err(e) = func.call(&mut *store, ()) {
            self.error = Some(e.to_string());
            return;
        }

        let state = store.data();
        transform.translation = state.translation;
        transform.rotation = state.rotation;
        transform.scale = state.scale;
        self.props = state.props;
    }
}

impl TypedComponent for Script {
    const NAME: &'static str = "Script";
}

impl Script {
    fn start_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        if self.started { return; }
        self.started = true;
        self.ensure_loaded();
        self.run_hook(ctx, Hook::Start);
    }

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        self.start_native(ctx);
        self.ensure_loaded();
        self.run_hook(ctx, Hook::Update);
    }
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<Script>();
    registry.register_system_if("zestengine::script_start", Stage::Start,
        |world, ctx| crate::ecs::run_ordered(world, ctx, Script::start_native),
        |world| world.query::<&Script>().iter().any(|(_, script)| !script.started));
    registry.register_system_for::<Script>("zestengine::script_update", Stage::Gameplay,
        |world, ctx| crate::ecs::run_ordered(world, ctx, Script::update_native));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentRegistry, InputState};
    use crate::ecs::SceneWorld;

    fn run(scene: &mut SceneWorld, stage: Stage) {
        scene.propagate().unwrap();
        let input = InputState::default();
        let mut commands = Vec::new();
        let mut ctx = crate::ecs::SystemContext::new(&input, &mut commands);
        ctx.dt = 1.0 / 60.0;
        ctx.time = 1.0;
        ComponentRegistry::with_builtins().run_systems(stage, scene, &mut ctx);
        assert!(commands.is_empty(), "script values are committed directly to typed storage");
    }

    const SPIN_WAT: &str = include_str!("../examples/scripts/spin.wat");

    #[test]
    fn spin_example_runs_start_and_update() {
        let mut script = Script {
            module: "spin.wat".into(),
            ..Default::default()
        };
        script.build_runtime(&wat::parse_str(SPIN_WAT).unwrap()).unwrap();

        let mut scene = SceneWorld::new();
        let entity = scene.spawn("Scripted");
        scene.insert(entity, script).unwrap();
        run(&mut scene, Stage::Start);
        assert_eq!(scene.get::<Script>(entity).unwrap().error, None);
        assert_eq!(scene.get::<Transform>(entity).unwrap().scale, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(scene.get::<Transform>(entity).unwrap().translation, Vec3::Y);
        assert_eq!(scene.get::<Script>(entity).unwrap().props[0], 0.0);
        run(&mut scene, Stage::Gameplay);
        let script = scene.get::<Script>(entity).unwrap();
        let transform = scene.get::<Transform>(entity).unwrap();
        assert!((script.props[0] - 1.0 / 60.0).abs() < 1e-5);
        assert!((transform.translation.x - 2.0 / 60.0).abs() < 1e-4);
        assert_eq!((transform.translation.y, transform.translation.z), (1.0, 0.0));
        assert_eq!(transform.rotation, Quat::from_xyzw(0.0, 1.0, 0.0, 0.0));
    }

    fn compiled(wat: &str) -> Script {
        let mut script = Script::default();
        script.build_runtime(&wat::parse_str(wat).unwrap()).unwrap();
        script
    }

    #[test]
    fn native_script_start_tracks_new_instances_and_survives_disable_reenable() {
        use crate::ecs::Enabled;
        const COUNTS: &str = r#"(module
            (import "env" "get_prop" (func $get (param i32) (result f32)))
            (import "env" "set_prop" (func $set (param i32 f32)))
            (func (export "start") (call $set (i32.const 0) (f32.add (call $get (i32.const 0)) (f32.const 1))))
            (func (export "update") (call $set (i32.const 1) (f32.add (call $get (i32.const 1)) (f32.const 1)))))"#;
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Hidden parent");
        let entity = scene.spawn("Scripted");
        scene.set_parent(entity, Some(parent), false).unwrap();
        scene.get_mut::<Enabled>(parent).unwrap().0 = false;
        scene.insert(entity, compiled(COUNTS)).unwrap();
        run(&mut scene, Stage::Start);
        assert!(!scene.get::<Script>(entity).unwrap().started);
        scene.get_mut::<Enabled>(parent).unwrap().0 = true;
        for _ in 0..2 { run(&mut scene, Stage::Start); run(&mut scene, Stage::Gameplay); }
        assert_eq!(&scene.get::<Script>(entity).unwrap().props[..2], &[1.0, 2.0]);
        let script = scene.remove::<Script>(entity).unwrap();
        scene.insert_disabled(entity, script).unwrap();
        run(&mut scene, Stage::Start); run(&mut scene, Stage::Gameplay);
        let script = scene.remove_disabled::<Script>(entity).unwrap();
        assert_eq!(&script.props[..2], &[1.0, 2.0]);
        scene.insert(entity, script).unwrap();
        run(&mut scene, Stage::Start); run(&mut scene, Stage::Gameplay);
        assert_eq!(&scene.get::<Script>(entity).unwrap().props[..2], &[1.0, 3.0]);
        scene.remove::<Script>(entity).unwrap();
        scene.insert(entity, compiled(COUNTS)).unwrap();
        run(&mut scene, Stage::Start); run(&mut scene, Stage::Gameplay);
        assert_eq!(&scene.get::<Script>(entity).unwrap().props[..2], &[1.0, 1.0], "replacement instances get their own start");
        let saved = crate::Component::save(&*scene.get::<Script>(entity).unwrap());
        let restored: Script = ron::from_str(&saved).unwrap();
        assert!(!restored.started && restored.store.is_none() && restored.error.is_none());
    }

    #[test]
    fn native_script_traps_and_fuel_exhaustion_rollback_guest_transform_and_props() {
        for failure in ["unreachable", "(loop $forever (br $forever))"] {
            let code = format!(r#"(module
                (import "env" "set_position" (func $position (param f32 f32 f32)))
                (import "env" "set_prop" (func $prop (param i32 f32)))
                (func (export "update")
                    (call $position (f32.const 99) (f32.const 98) (f32.const 97))
                    (call $prop (i32.const 0) (f32.const 96)) {failure}))"#);
            let mut scene = SceneWorld::new();
            let entity = scene.spawn("Trapping script");
            scene.get_mut::<Transform>(entity).unwrap().translation = Vec3::new(1.0, 2.0, 3.0);
            let mut script = compiled(&code);
            script.props[0] = 42.0;
            scene.insert(entity, script).unwrap();
            run(&mut scene, Stage::Gameplay);
            assert_eq!(scene.get::<Transform>(entity).unwrap().translation, Vec3::new(1.0, 2.0, 3.0));
            assert_eq!(scene.get::<Script>(entity).unwrap().props[0], 42.0);
            let error = scene.get::<Script>(entity).unwrap().error.clone().expect("guest failure disables this instance");
            run(&mut scene, Stage::Gameplay);
            assert_eq!(scene.get::<Script>(entity).unwrap().error.as_deref(), Some(error.as_str()));
            assert_eq!(scene.get::<Script>(entity).unwrap().props[0], 42.0);
        }
    }

    #[test]
    fn native_script_reads_current_local_transform_under_parent_and_keeps_runtime_state() {
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Parent");
        scene.get_mut::<Transform>(parent).unwrap().translation = Vec3::X * 10.0;
        let entity = scene.spawn("Scripted");
        scene.set_parent(entity, Some(parent), false).unwrap();
        scene.insert(entity, compiled(SPIN_WAT)).unwrap();
        run(&mut scene, Stage::Start);
        for _ in 0..3 { run(&mut scene, Stage::Gameplay); }
        assert!((scene.get::<Script>(entity).unwrap().props[0] - 3.0 / 60.0).abs() < 1e-5);
        scene.get_mut::<Transform>(entity).unwrap().translation.x = 5.0;
        run(&mut scene, Stage::Gameplay);
        scene.with_world(|world| {
            assert!((world.world_transform(entity).unwrap().w_axis.x - (15.0 + 2.0 / 60.0)).abs() < 1e-4);
        });
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
        let greedy = r#"(module (memory 64) (func (export "update")))"#;

        let mut script = Script::default();
        let err = script.build_runtime(&wat::parse_str(greedy).unwrap()).unwrap_err();
        assert!(err.to_lowercase().contains("memory"), "unexpected error: {err}");
    }
}
