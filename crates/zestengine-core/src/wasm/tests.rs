use super::fixtures::*;
use super::sandbox::{ALL_CAPS, Instance, SandboxLimits, UNTRUSTED_CAPS, inspect_module};
use super::*;
use crate::actions::{ActionCtx, ArgValue};
use crate::{Component, ComponentCommand, ObjectId, ObjectRef, TypedComponent};

#[path = "../../../../guest/zest-guest/src/codec.rs"]
#[allow(dead_code)]
mod guest;

fn trusted(bytes: &[u8]) -> Result<Instance, String> {
    Instance::load(bytes, SandboxLimits::trusted(), ALL_CAPS, "test").map(|(i, _)| i)
}

fn invoke(c: &mut WasmScript, name: &str, args: &[ArgValue]) -> (Result<(), String>, Vec<ComponentCommand>) {
    let mut commands = Vec::new();
    let r = TypedComponent::invoke_action(
        c,
        &mut ActionCtx { self_id: ObjectId::new(), instigator: None, commands: &mut commands, reader: None },
        name,
        args,
    );
    (r, commands)
}

fn counter_file(dir: &std::path::Path, name: &str, factor: i64) -> String {
    write(dir, name, &build(counter_manifest(), &counter(factor)))
}

#[test]
fn every_arg_type_round_trips_through_zest_invoke() {
    let mut inst = trusted(&build(ECHO_MANIFEST, ECHO)).unwrap();
    let values = [
        ArgValue::Bool(true),
        ArgValue::Int(-1234567890123),
        ArgValue::Float(0.25),
        ArgValue::String("héllo".into()),
        ArgValue::Vec3([1.0, 2.0, -3.0]),
        ArgValue::Object(ObjectRef::new(ObjectId::new())),
        ArgValue::Object(ObjectRef::NONE),
    ];
    for v in values {
        inst.call_invoke(0, std::slice::from_ref(&v), ObjectId::NIL, None, None).unwrap();
        assert_eq!(inst.call_read_property(0).unwrap(), Some(v));
    }
}

#[test]
fn guest_side_codec_matches_the_host_codec() {
    let args = [
        ArgValue::Bool(true),
        ArgValue::Int(7),
        ArgValue::Float(1.5),
        ArgValue::String("x".into()),
        ArgValue::Vec3([1.0, 2.0, 3.0]),
        ArgValue::Object(ObjectRef::new(ObjectId::new())),
    ];
    let host_bytes = abi::encode_args(&args);
    let decoded = guest::decode_args(&host_bytes).unwrap();
    assert_eq!(guest::encode_args(&decoded), host_bytes);
}

#[test]
fn manifest_section_is_required_and_unique() {
    let plain = wat::parse_str(r#"(module (memory (export "memory") 1))"#).unwrap();
    assert!(inspect_module(&plain, &SandboxLimits::trusted(), ALL_CAPS).unwrap_err().contains("zest_manifest"));
    assert!(inspect_module(b"not wasm at all", &SandboxLimits::trusted(), ALL_CAPS).is_err());
    let two = wat::parse_str(r#"(module (@custom "zest_manifest" "()") (@custom "zest_manifest" "()") (memory (export "memory") 1))"#).unwrap();
    assert!(inspect_module(&two, &SandboxLimits::trusted(), ALL_CAPS).unwrap_err().contains("more than one"));
    // Declared actions need the exports that serve them.
    let m = build(ONE_ACTION, r#"(memory (export "memory") 1)"#);
    assert!(inspect_module(&m, &SandboxLimits::trusted(), ALL_CAPS).is_err());
}

#[test]
fn imports_outside_the_host_allowlist_are_rejected() {
    for import in [
        r#"(import "env" "x" (func))"#,
        r#"(import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32)))"#,
        r#"(import "host" "system" (func))"#,
        r#"(import "host" "log" (global i32))"#,
        r#"(import "host" "memory" (memory 1))"#,
    ] {
        let m = build("()", &format!(r#"{import} (memory (export "memory") 1)"#));
        let e = inspect_module(&m, &SandboxLimits::trusted(), ALL_CAPS).unwrap_err();
        assert!(e.contains("host") || e.contains("import"), "{import}: {e}");
    }
    let ok = build("()", r#"(import "host" "log" (func (param i32 i32))) (memory (export "memory") 1)"#);
    assert!(inspect_module(&ok, &SandboxLimits::trusted(), ALL_CAPS).is_ok());
}

#[test]
fn host_capabilities_are_gated_by_manifest_and_trust() {
    let invoke_import = r#"(import "host" "invoke" (func (param i32 i32 i32 i32 i32 i32 i32) (result i32))) (memory (export "memory") 1)"#;
    // Imported but not declared.
    let undeclared = build("()", invoke_import);
    assert!(inspect_module(&undeclared, &SandboxLimits::trusted(), ALL_CAPS).unwrap_err().contains("invoke"));
    // Declared, trusted: fine. Declared, untrusted: refused.
    let declared = build(r#"(capabilities: ["invoke"])"#, invoke_import);
    assert!(inspect_module(&declared, &SandboxLimits::trusted(), ALL_CAPS).is_ok());
    assert!(inspect_module(&declared, &SandboxLimits::untrusted(), UNTRUSTED_CAPS).unwrap_err().contains("not permitted"));
    // An unknown capability name fails the manifest.
    assert!(inspect_module(&build(r#"(capabilities: ["fs"])"#, r#"(memory (export "memory") 1)"#), &SandboxLimits::trusted(), ALL_CAPS).is_err());
}

#[test]
fn module_size_and_memory_caps_are_enforced() {
    let big_memory = build("()", r#"(memory (export "memory") 300)"#);
    assert!(inspect_module(&big_memory, &SandboxLimits::trusted(), ALL_CAPS).unwrap_err().contains("pages"));
    let tiny = SandboxLimits { max_module_bytes: 16, ..SandboxLimits::trusted() };
    assert!(inspect_module(&build("()", r#"(memory (export "memory") 1)"#), &tiny, ALL_CAPS).unwrap_err().contains("limit"));
    // Growth past the cap is refused (memory.grow returns -1), not a crash.
    let grow = build(ONE_ACTION, &with_invoke("", GROW));
    let mut inst = trusted(&grow).unwrap();
    inst.call_invoke(0, &[], ObjectId::NIL, None, None).unwrap();
}

#[test]
fn infinite_loop_runs_out_of_fuel_and_disables_the_component() {
    let dir = temp_dir("fuel");
    let path = write(&dir, "spin.wasm", &build(ONE_ACTION, &with_invoke("", SPIN)));
    let mut c = WasmScript::new(path);
    assert_eq!(TypedComponent::actions(&c).len(), 1);
    let (r, _) = invoke(&mut c, "Go", &[]);
    assert!(r.is_ok(), "the trap is absorbed: {r:?}");
    let status = c.status();
    assert!(status.error.as_deref().unwrap().contains("instruction budget"), "{status:?}");
    // Inert from now on: no panic, no second error.
    assert!(invoke(&mut c, "Go", &[]).0.is_ok());
    assert_eq!(TypedComponent::actions(&c).len(), 1, "bindings still resolve");
}

#[test]
fn deep_recursion_and_traps_disable_without_crashing_the_host() {
    let dir = temp_dir("trap");
    for (name, extra, body, expect) in [("rec", RECURSE_FN, RECURSE, "stack"), ("trap", "", TRAP, "unreachable")] {
        let path = write(&dir, &format!("{name}.wasm"), &build(ONE_ACTION, &with_invoke(extra, body)));
        let mut c = WasmScript::new(path);
        assert!(invoke(&mut c, "Go", &[]).0.is_ok());
        let e = c.status().error.unwrap();
        assert!(e.to_lowercase().contains(expect), "{name}: {e}");
    }
}

#[test]
fn component_publishes_actions_properties_and_runs_them() {
    let dir = temp_dir("counter");
    let mut c = WasmScript::new(counter_file(&dir, "c.wasm", 1));
    let actions = TypedComponent::actions(&c);
    assert_eq!(actions.iter().map(|a| a.name).collect::<Vec<_>>(), ["Add", "Fail", "Ping", "Secret"]);
    assert_eq!(actions[0].label, "Add to count");
    assert_eq!(actions[0].params[0].default.value(), ArgValue::Int(1));
    assert!(actions[0].ugc_safe && !actions[3].ugc_safe);
    assert_eq!(TypedComponent::properties(&c)[0].name, "count");
    assert_eq!(TypedComponent::read_property(&c, "count"), Some(ArgValue::Int(0)));
    assert!(invoke(&mut c, "Add", &[ArgValue::Int(5)]).0.is_ok());
    assert!(invoke(&mut c, "Add", &[ArgValue::Int(2)]).0.is_ok());
    assert_eq!(TypedComponent::read_property(&c, "count"), Some(ArgValue::Int(7)));
    assert_eq!(TypedComponent::read_property(&c, "nope"), None);
    // A guest failure is an error for the binding, and does not disable.
    assert_eq!(invoke(&mut c, "Fail", &[]).0, Err("nope".into()));
    assert!(c.status().error.is_none());
    assert!(invoke(&mut c, "Missing", &[]).0.is_err());
    // Events come back as commands.
    let (r, commands) = invoke(&mut c, "Ping", &[]);
    assert!(r.is_ok());
    assert!(matches!(&commands[..], [ComponentCommand::Emit(_)]));
    // The Component trait object sees the same surface (what the catalog uses).
    let boxed: Box<dyn Component> = Box::new(c);
    assert_eq!(boxed.type_name(), "Wasm Script");
    assert_eq!(boxed.actions().len(), 4);
}

#[test]
fn saves_only_the_module_path_and_missing_modules_report_once() {
    let c = WasmScript::new("nowhere/missing.wasm");
    let text = Component::save(&c);
    assert!(text.contains("missing.wasm"));
    let back: WasmScript = ron::from_str(&text).unwrap();
    assert_eq!(back.module, "nowhere/missing.wasm");
    assert!(TypedComponent::actions(&back).is_empty());
    assert!(back.status().error.is_some());
    let _: WasmScript = ron::from_str("()").unwrap();
}

#[test]
fn hot_reload_swaps_behaviour_when_the_file_changes() {
    let dir = temp_dir("reload");
    let path = counter_file(&dir, "c.wasm", 1);
    let mut c = WasmScript::new(path.clone());
    invoke(&mut c, "Add", &[ArgValue::Int(3)]).0.unwrap();
    assert_eq!(TypedComponent::read_property(&c, "count"), Some(ArgValue::Int(3)));
    let before = c.status().generation;
    // Same bytes again: nothing to do.
    c.reload_if_changed();
    assert_eq!(c.status().generation, before);
    // New build (adds x10); push the mtime so coarse clocks notice.
    std::fs::write(&path, build(counter_manifest(), &counter(10))).unwrap();
    let f = std::fs::File::options().write(true).open(&path).unwrap();
    f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5)).unwrap();
    c.reload_if_changed();
    assert!(c.status().generation > before);
    invoke(&mut c, "Add", &[ArgValue::Int(3)]).0.unwrap();
    assert_eq!(TypedComponent::read_property(&c, "count"), Some(ArgValue::Int(30)));
    // A broken build disables; fixing it brings it back.
    std::fs::write(&path, b"garbage").unwrap();
    let f = std::fs::File::options().write(true).open(&path).unwrap();
    f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(10)).unwrap();
    c.reload_if_changed();
    assert!(c.status().error.is_some() && TypedComponent::actions(&c).is_empty());
    std::fs::write(&path, build(counter_manifest(), &counter(1))).unwrap();
    let f = std::fs::File::options().write(true).open(&path).unwrap();
    f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(15)).unwrap();
    c.reload_if_changed();
    assert_eq!(TypedComponent::actions(&c).len(), 4);
}

#[test]
fn untrusted_content_only_exposes_ugc_safe_actions_the_limits_allow() {
    let dir = temp_dir("ugc");
    let path = counter_file(&dir, "c.wasm", 1);
    let policy = UntrustedPolicy { limits: SandboxLimits::untrusted(), allow_actions: true };
    register_untrusted_root(&dir, policy);
    let mut c = WasmScript::new(path.clone());
    assert!(c.status().untrusted);
    assert!(invoke(&mut c, "Add", &[ArgValue::Int(2)]).0.is_ok());
    let e = invoke(&mut c, "Secret", &[]).0.unwrap_err();
    assert!(e.contains("not allowed"), "{e}");
    assert_eq!(TypedComponent::read_property(&c, "count"), Some(ArgValue::Int(2)), "Secret never ran");
    // The package's content limits can switch module actions off entirely.
    register_untrusted_root(&dir, UntrustedPolicy { allow_actions: false, ..policy });
    c.force_reload();
    assert!(TypedComponent::actions(&c).is_empty());
    assert!(invoke(&mut c, "Add", &[ArgValue::Int(1)]).0.is_err());
    // A module declaring `invoke` doesn't load in untrusted content.
    let bad = write(&dir, "bad.wasm", &build(r#"(capabilities: ["invoke"])"#, r#"(import "host" "invoke" (func (param i32 i32 i32 i32 i32 i32 i32) (result i32))) (memory (export "memory") 1)"#));
    let c = WasmScript::new(bad);
    assert!(c.status().error.unwrap().contains("not permitted"));
}

#[test]
fn many_components_share_compiled_modules_and_descriptor_tables() {
    let dir = temp_dir("share");
    let path = counter_file(&dir, "c.wasm", 1);
    let a = WasmScript::new(path.clone());
    let b = WasmScript::new(path);
    assert!(std::ptr::eq(TypedComponent::actions(&a).as_ptr(), TypedComponent::actions(&b).as_ptr()));
}
