use std::sync::{Arc, Mutex, OnceLock};

pub use wasmtime;
use wasmtime::{Engine, ExternType, Module, StoreLimits, StoreLimitsBuilder, Trap};

pub const WASM_PAGE_BYTES: u64 = 65_536;

const MODULE_CACHE: usize = 32;
const MAX_TABLE_ELEMENTS: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandboxLimits {
    pub max_module_bytes: usize,
    pub max_memory_pages: u32,
    pub fuel_per_call: u64,
    pub max_stack_bytes: usize,
}

impl SandboxLimits {
    pub const fn trusted() -> Self {
        Self { max_module_bytes: 4 << 20, max_memory_pages: 256, fuel_per_call: 5_000_000, max_stack_bytes: 512 << 10 }
    }

    pub const fn untrusted() -> Self {
        Self { max_module_bytes: 1 << 20, max_memory_pages: 64, fuel_per_call: 1_000_000, max_stack_bytes: 256 << 10 }
    }

    pub const fn legacy_script() -> Self {
        Self { max_module_bytes: 4 << 20, max_memory_pages: 16, fuel_per_call: 100_000, max_stack_bytes: 512 << 10 }
    }

    pub const fn memory_bytes(&self) -> usize {
        self.max_memory_pages as usize * WASM_PAGE_BYTES as usize
    }

    pub fn store_limits(&self, tables: usize) -> StoreLimits {
        StoreLimitsBuilder::new()
            .memory_size(self.memory_bytes())
            .memories(1)
            .tables(tables)
            .table_elements(MAX_TABLE_ELEMENTS)
            .instances(1)
            .trap_on_grow_failure(false)
            .build()
    }
}

pub fn engine() -> Result<&'static Engine, String> {
    static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let mut config = wasmtime::Config::new();
            config.consume_fuel(true);
            config.cranelift_nan_canonicalization(true);
            config.wasm_relaxed_simd(false);
            config.max_wasm_stack(SandboxLimits::trusted().max_stack_bytes);
            Engine::new(&config).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub fn compile(bytes: &[u8]) -> Result<Module, String> {
    static CACHE: Mutex<Vec<(Arc<[u8]>, Module)>> = Mutex::new(Vec::new());
    let engine = engine()?;
    {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = cache.iter().position(|(b, _)| **b == *bytes) {
            let hit = cache.remove(at);
            let module = hit.1.clone();
            cache.push(hit);
            return Ok(module);
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

pub fn check_size(bytes: &[u8], limits: &SandboxLimits) -> Result<(), String> {
    if bytes.len() > limits.max_module_bytes {
        return Err(format!("module is {} bytes (limit {})", bytes.len(), limits.max_module_bytes));
    }
    Ok(())
}

fn leb_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let mut value = 0u64;
    for shift in (0..35).step_by(7) {
        let byte = *bytes.get(*pos)?;
        *pos += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return u32::try_from(value).ok();
        }
    }
    None
}

pub fn custom_section<'a>(bytes: &'a [u8], name: &str) -> Result<Option<&'a [u8]>, String> {
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
            if &bytes[p..name_end] == name.as_bytes() {
                if found.is_some() {
                    return Err(format!("more than one {name:?} section"));
                }
                found = Some(&bytes[name_end..end]);
            }
        }
        pos = end;
    }
    Ok(found)
}

pub fn check_imports(module: &Module, namespace: &str, mut check: impl FnMut(&str) -> Result<(), String>) -> Result<(), String> {
    for import in module.imports() {
        if import.module() != namespace {
            return Err(format!("imports {}.{}; only the {namespace:?} namespace is available", import.module(), import.name()));
        }
        check(import.name())?;
        if import.ty().func().is_none() {
            return Err(format!("{namespace}.{} must be imported as a function", import.name()));
        }
    }
    Ok(())
}

pub fn check_memory(module: &Module, limits: &SandboxLimits) -> Result<(), String> {
    let mut memories = 0;
    for export in module.exports() {
        if let ExternType::Memory(memory) = export.ty() {
            memories += 1;
            if memory.is_64() || memory.is_shared() {
                return Err("memory must be 32-bit and not shared".into());
            }
            if memory.minimum() > limits.max_memory_pages as u64 {
                return Err(format!(
                    "memory starts at {} pages ({} KiB); limit {} pages",
                    memory.minimum(),
                    memory.minimum() * WASM_PAGE_BYTES / 1024,
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
    Ok(())
}

pub fn trap_text(error: &wasmtime::Error) -> String {
    match error.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel) => "ran past its instruction budget (infinite loop?)".into(),
        Some(Trap::StackOverflow) => "call stack exhausted".into(),
        Some(trap) => trap.to_string(),
        None => format!("{error}").lines().next().unwrap_or("trap").to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmtime::{Linker, Store};

    struct Host {
        limits: StoreLimits,
    }

    fn store(limits: SandboxLimits) -> Store<Host> {
        let mut store = Store::new(engine().unwrap(), Host { limits: limits.store_limits(1) });
        store.limiter(|host| &mut host.limits);
        store.set_fuel(limits.fuel_per_call).unwrap();
        store
    }

    #[test]
    fn one_engine_and_cached_modules_are_shared() {
        let bytes = wat::parse_str(r#"(module (memory (export "memory") 1) (func (export "run")))"#).unwrap();
        let first = compile(&bytes).unwrap();
        let second = compile(&bytes).unwrap();
        assert!(Engine::same(first.engine(), second.engine()));
        assert!(Engine::same(first.engine(), engine().unwrap()));
        assert!(compile(&[0xde, 0xad, 0xbe, 0xef]).unwrap_err().starts_with("invalid module"));
    }

    #[test]
    fn size_memory_and_import_checks_reject_without_running() {
        let limits = SandboxLimits::untrusted();
        assert!(check_size(&vec![0; limits.max_module_bytes + 1], &limits).is_err());
        let greedy = compile(&wat::parse_str(r#"(module (memory (export "memory") 65))"#).unwrap()).unwrap();
        assert!(check_memory(&greedy, &limits).unwrap_err().contains("limit 64 pages"));
        let unnamed = compile(&wat::parse_str(r#"(module (memory (export "heap") 1))"#).unwrap()).unwrap();
        assert!(check_memory(&unnamed, &limits).unwrap_err().contains("as \"memory\""));
        let none = compile(&wat::parse_str("(module)").unwrap()).unwrap();
        assert!(check_memory(&none, &limits).unwrap_err().contains("exactly one memory"));
        let wasi = compile(&wat::parse_str(r#"(module (import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32))))"#).unwrap()).unwrap();
        assert!(check_imports(&wasi, "host", |_| Ok(())).unwrap_err().contains("only the \"host\" namespace"));
        let unknown = compile(&wat::parse_str(r#"(module (import "host" "format_disk" (func)))"#).unwrap()).unwrap();
        let known = |name: &str| if name == "log" { Ok(()) } else { Err(format!("imports unknown host function {name:?}")) };
        assert!(check_imports(&unknown, "host", known).unwrap_err().contains("format_disk"));
        let global = compile(&wat::parse_str(r#"(module (import "host" "log" (global i32)))"#).unwrap()).unwrap();
        assert!(check_imports(&global, "host", known).unwrap_err().contains("must be imported as a function"));
    }

    #[test]
    fn custom_sections_are_read_from_the_section_table() {
        let bytes = wat::parse_str(r#"(module (@custom "zest_manifest" "hello") (@custom "other" "x"))"#).unwrap();
        assert_eq!(custom_section(&bytes, "zest_manifest").unwrap(), Some(&b"hello"[..]));
        assert_eq!(custom_section(&bytes, "missing").unwrap(), None);
        let twice = wat::parse_str(r#"(module (@custom "zest_manifest" "a") (@custom "zest_manifest" "b"))"#).unwrap();
        assert!(custom_section(&twice, "zest_manifest").unwrap_err().contains("more than one"));
        assert!(custom_section(b"nope", "zest_manifest").is_err());
        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() - 2);
        assert!(custom_section(&truncated, "zest_manifest").is_err());
    }

    #[test]
    fn fuel_stack_and_memory_growth_are_bounded() {
        let bytes = wat::parse_str(
            r#"(module
                (memory (export "memory") 1)
                (func (export "spin") (loop $forever (br $forever)))
                (func $dive (export "dive") (call $dive))
                (func (export "grow") (result i32) (memory.grow (i32.const 1024))))"#,
        )
        .unwrap();
        let module = compile(&bytes).unwrap();
        let limits = SandboxLimits::untrusted();
        let linker = Linker::<Host>::new(engine().unwrap());
        let mut spin_store = store(limits);
        let instance = linker.instantiate(&mut spin_store, &module).unwrap();
        let spin = instance.get_typed_func::<(), ()>(&mut spin_store, "spin").unwrap();
        assert_eq!(trap_text(&spin.call(&mut spin_store, ()).unwrap_err()), "ran past its instruction budget (infinite loop?)");
        let mut dive_store = store(SandboxLimits { fuel_per_call: u64::MAX / 2, ..limits });
        let instance = linker.instantiate(&mut dive_store, &module).unwrap();
        let dive = instance.get_typed_func::<(), ()>(&mut dive_store, "dive").unwrap();
        assert_eq!(trap_text(&dive.call(&mut dive_store, ()).unwrap_err()), "call stack exhausted");
        let mut grow_store = store(limits);
        let instance = linker.instantiate(&mut grow_store, &module).unwrap();
        let grow = instance.get_typed_func::<(), i32>(&mut grow_store, "grow").unwrap();
        assert_eq!(grow.call(&mut grow_store, ()).unwrap(), -1);
    }
}
