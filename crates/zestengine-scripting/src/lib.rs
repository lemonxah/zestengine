//! WASM scripting for user content: load and run WebAssembly modules against a
//! small host API (`log`, float read/write).

use std::collections::HashMap;

use wasmtime::{Caller, Engine, Linker, Module, Store, TypedFunc};

/// Per-script host state the host functions can read and write.
#[derive(Default)]
pub struct HostState {
    pub logs: Vec<String>,
    pub floats: HashMap<String, f32>,
}

/// A compiled-and-instantiated user script.
pub struct Script {
    store: Store<HostState>,
    update: TypedFunc<f32, ()>,
}

impl Script {
    /// Call the module's exported `update(f32)`.
    pub fn update(&mut self, dt: f32) -> Result<(), String> {
        self.update.call(&mut self.store, dt).map_err(|e| e.to_string())
    }

    /// Log lines the script wrote via `host.log`.
    pub fn logs(&self) -> &[String] {
        &self.store.data().logs
    }

    /// A float the script wrote via `host.set_float`.
    pub fn get_float(&self, key: &str) -> Option<f32> {
        self.store.data().floats.get(key).copied()
    }
}

/// Loads and instantiates user WASM modules against the host API.
pub struct ScriptHost {
    engine: Engine,
    linker: Linker<HostState>,
}

/// Read a UTF-8 string from the caller's linear memory at `[ptr, ptr+len)`.
fn read_str(caller: &mut Caller<'_, HostState>, ptr: i32, len: i32) -> String {
    let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) else {
        return String::new();
    };
    let data = mem.data(&*caller);
    let start = ptr.max(0) as usize;
    let end = (ptr + len).max(ptr) as usize;
    String::from_utf8_lossy(data.get(start..end).unwrap_or(&[])).into_owned()
}

fn log_impl(mut caller: Caller<'_, HostState>, ptr: i32, len: i32) {
    let msg = read_str(&mut caller, ptr, len);
    caller.data_mut().logs.push(msg);
}

fn get_float_impl(mut caller: Caller<'_, HostState>, ptr: i32, len: i32) -> f32 {
    let key = read_str(&mut caller, ptr, len);
    caller.data().floats.get(&key).copied().unwrap_or(0.0)
}

fn set_float_impl(mut caller: Caller<'_, HostState>, ptr: i32, len: i32, value: f32) {
    let key = read_str(&mut caller, ptr, len);
    caller.data_mut().floats.insert(key, value);
}

impl ScriptHost {
    pub fn new() -> Result<Self, String> {
        let engine = Engine::default();
        let mut linker = Linker::new(&engine);
        linker.func_wrap("host", "log", log_impl).map_err(|e| e.to_string())?;
        linker.func_wrap("host", "get_float", get_float_impl).map_err(|e| e.to_string())?;
        linker.func_wrap("host", "set_float", set_float_impl).map_err(|e| e.to_string())?;
        Ok(Self { engine, linker })
    }

    /// Compile and instantiate `wasm_bytes`, returning a script that can call
    /// `update(dt)`.
    pub fn load(&self, wasm_bytes: &[u8]) -> Result<Script, String> {
        let module = Module::new(&self.engine, wasm_bytes).map_err(|e| e.to_string())?;
        let mut store = Store::new(&self.engine, HostState::default());
        let instance = self.linker.instantiate(&mut store, &module).map_err(|e| e.to_string())?;
        let update = instance.get_typed_func::<f32, ()>(&mut store, "update").map_err(|e| e.to_string())?;
        Ok(Script { store, update })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULE: &str = r#"
(module
  (import "host" "log" (func $log (param i32 i32)))
  (import "host" "set_float" (func $set_float (param i32 i32 f32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "hello")
  (data (i32.const 16) "score")
  (func (export "update") (param f32)
    i32.const 0
    i32.const 5
    call $log
    i32.const 16
    i32.const 5
    f32.const 3.5
    call $set_float))
"#;

    #[test]
    fn script_calls_the_host_and_reads_back() {
        let wasm = wat::parse_str(MODULE).unwrap();
        let host = ScriptHost::new().unwrap();
        let mut script = host.load(&wasm).unwrap();
        script.update(0.016).unwrap();
        assert_eq!(script.logs(), &["hello".to_owned()]);
        assert_eq!(script.get_float("score"), Some(3.5));
    }

    #[test]
    fn host_rejects_malformed_wasm() {
        let host = ScriptHost::new().unwrap();
        assert!(host.load(&[0xde, 0xad, 0xbe, 0xef]).is_err());
    }
}
