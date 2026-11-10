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

fn native_frame(fixture: &mut crate::ecs::NativeTestFixture) -> Vec<ComponentCommand> {
    fixture.frame(&crate::InputState::default(), &crate::NetView::default(), &crate::ScreenInfo::default())
}

const LOAN_COUNTER_MANIFEST: &str = r#"(name: "Loan counter", capabilities: ["read", "state"], properties: [(name: "count", ty: Int)])"#;
fn loan_counter(initial: i64, delta: i64, object: Option<ObjectId>) -> String {
    let object_data = object.map(|id| {
        let bytes: String = id.raw().to_le_bytes().iter().map(|byte| format!("\\{byte:02x}")).collect();
        format!(r#"(data (i32.const 100) "{bytes}")"#)
    }).unwrap_or_default();
    format!(r#"
        (import "host" "get_property" (func $get (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
        (import "host" "set_float" (func $set (param i32 i32 f32)))
        (memory (export "memory") 1)
        {object_data}
        (data (i32.const 200) "Wasm Script")
        (data (i32.const 220) "count")
        (data (i32.const 240) "other")
        (data (i32.const 250) "rc")
        (global $count (mut i64) (i64.const {initial}))
        (func (export "zest_read_property") (param i32) (result i64)
            (i32.store8 (i32.const 400) (i32.const 1))
            (i64.store offset=1 (i32.const 400) (global.get $count))
            (i64.or (i64.shl (i64.const 9) (i64.const 32)) (i64.const 400)))
        (func (export "zest_update") (param f32) (local $rc i32)
            (local.set $rc (call $get (i32.const 100) (i32.const 200) (i32.const 11) (i32.const 220) (i32.const 5) (i32.const 300) (i32.const 32)))
            (call $set (i32.const 250) (i32.const 2) (f32.convert_i32_s (local.get $rc)))
            (if (i32.gt_s (local.get $rc) (i32.const 0)) (then
                (call $set (i32.const 240) (i32.const 5) (f32.convert_i64_s (i64.load offset=1 (i32.const 300))))))
            (global.set $count (i64.add (global.get $count) (i64.const {delta}))))
    "#)
}

#[test]
fn native_wasm_updates_lend_same_type_siblings_in_authoring_order_and_exclude_self() {
    use crate::ecs::{NativeTestFixture, Owner};
    let dir = temp_dir("native-query-loans");
    let first = write(&dir, "first.wasm", &build(LOAN_COUNTER_MANIFEST, &loan_counter(10, 1, None)));
    let second = write(&dir, "second.wasm", &build(LOAN_COUNTER_MANIFEST, &loan_counter(20, 1, None)));
    let mut fixture = NativeTestFixture::new();
    let owner = fixture.add("Siblings", vec![Box::new(WasmScript::new(&first)), Box::new(WasmScript::new(&second))]);
    assert!(native_frame(&mut fixture).is_empty());
    fixture.scene.with_world(|world| {
        for (_, (actual_owner, script)) in world.query::<(&Owner, &WasmScript)>().iter() {
            assert_eq!(actual_owner.0, owner);
            let (other, count) = if script.module == first { (20.0, 11) } else { (11.0, 21) };
            assert_eq!(script.float("other"), Some(other));
            assert_eq!(TypedComponent::read_property(script, "count"), Some(ArgValue::Int(count)));
        }
    });
    let alone = fixture.add("Alone", vec![Box::new(WasmScript::new(&first))]);
    native_frame(&mut fixture);
    assert_eq!(fixture.scene.get::<WasmScript>(alone).unwrap().float("rc"), Some(-1.0), "the currently executing component is excluded from its own property reads");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn native_wasm_reads_another_object_in_the_same_archetype() {
    let dir = temp_dir("native-other-query");
    let mut fixture = crate::ecs::NativeTestFixture::new();
    let counter = fixture.add("Counter", vec![Box::new(WasmScript::new(write(&dir, "counter.wasm", &build(LOAN_COUNTER_MANIFEST, &loan_counter(30, 2, None)))))]);
    let reader = fixture.add("Reader", vec![Box::new(WasmScript::new(write(&dir, "reader.wasm", &build(LOAN_COUNTER_MANIFEST, &loan_counter(0, 0, Some(fixture.id(counter)))))))]);
    assert!(native_frame(&mut fixture).is_empty());
    assert_eq!(fixture.scene.get::<WasmScript>(reader).unwrap().float("other"), Some(32.0));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn native_wasm_loads_at_start_and_retains_or_reloads_runtime_without_replacing_components() {
    use crate::ecs::{Enabled, Stage};
    let dir = temp_dir("native-start-reload");
    let path = write(&dir, "counter.wasm", &build(LOAN_COUNTER_MANIFEST, &loan_counter(0, 1, None)));
    let mut fixture = crate::ecs::NativeTestFixture::new();
    let owner = fixture.add("Script", vec![Box::new(WasmScript::new(&path))]);
    assert!(fixture.stage(Stage::Start, &crate::InputState::default(), &crate::NetView::default(), &crate::ScreenInfo::default(), 0.1, 0.0).is_empty());
    let generation = fixture.scene.get::<WasmScript>(owner).unwrap().status().generation;
    for _ in 0..2 { native_frame(&mut fixture); }
    assert_eq!(TypedComponent::read_property(&*fixture.scene.get::<WasmScript>(owner).unwrap(), "count"), Some(ArgValue::Int(2)));
    fixture.scene.get_mut::<Enabled>(owner).unwrap().0 = false;
    native_frame(&mut fixture);
    fixture.scene.get_mut::<Enabled>(owner).unwrap().0 = true;
    native_frame(&mut fixture);
    let script = fixture.scene.get::<WasmScript>(owner).unwrap();
    assert_eq!(script.status().generation, generation);
    assert_eq!(TypedComponent::read_property(&*script, "count"), Some(ArgValue::Int(3)));
    drop(script);
    std::fs::write(&path, build(LOAN_COUNTER_MANIFEST, &loan_counter(0, 10, None))).unwrap();
    fixture.scene.get::<WasmScript>(owner).unwrap().force_reload();
    native_frame(&mut fixture);
    let script = fixture.scene.get::<WasmScript>(owner).unwrap();
    assert!(script.status().generation > generation);
    assert_eq!(TypedComponent::read_property(&*script, "count"), Some(ArgValue::Int(10)));
    drop(script);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn native_wasm_update_traps_discard_outputs_and_file_reload_recovers() {
    let dir = temp_dir("native-trap");
    let trapped = r#"
        (import "host" "emit_event" (func $emit (param i32 i32)))
        (memory (export "memory") 1)
        (data (i32.const 0) "discarded")
        (func (export "zest_update") (param f32) (call $emit (i32.const 0) (i32.const 9)) unreachable)
    "#;
    let path = write(&dir, "script.wasm", &build(r#"(capabilities: ["events"])"#, trapped));
    let mut fixture = crate::ecs::NativeTestFixture::new();
    let owner = fixture.add("Script", vec![Box::new(WasmScript::new(&path))]);
    assert!(native_frame(&mut fixture).is_empty());
    let error = fixture.scene.get::<WasmScript>(owner).unwrap().status().error.expect("update trap disables this module");
    assert!(error.contains("unreachable"));
    assert!(native_frame(&mut fixture).is_empty());
    std::fs::write(&path, build(UPDATER_MANIFEST, UPDATER)).unwrap();
    fixture.scene.get::<WasmScript>(owner).unwrap().force_reload();
    assert!(native_frame(&mut fixture).is_empty());
    assert!(fixture.scene.get::<WasmScript>(owner).unwrap().status().error.is_none());
    assert_eq!(fixture.scene.get::<WasmScript>(owner).unwrap().float("dt"), Some(0.1));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn native_wasm_queued_actions_resolve_after_query_and_can_invoke_their_own_component() {
    let dir = temp_dir("native-self-invoke");
    let manifest = r#"(capabilities: ["invoke", "events"], actions: [(name: "Bump", ugc_safe: true)], properties: [(name: "count", ty: Int)])"#;
    let code = r#"
        (import "host" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
        (import "host" "emit_event" (func $emit (param i32 i32)))
        (memory (export "memory") 1)
        (data (i32.const 200) "Wasm Script")
        (data (i32.const 220) "Bump")
        (data (i32.const 240) "tick")
        (global $count (mut i64) (i64.const 0))
        (func (export "zest_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "zest_invoke") (param i32 i32 i32) (result i32)
            (global.set $count (i64.add (global.get $count) (i64.const 1))) i32.const 0)
        (func (export "zest_read_property") (param i32) (result i64)
            (i32.store8 (i32.const 400) (i32.const 1))
            (i64.store offset=1 (i32.const 400) (global.get $count))
            (i64.or (i64.shl (i64.const 9) (i64.const 32)) (i64.const 400)))
        (func (export "zest_update") (param f32)
            (call $emit (i32.const 240) (i32.const 4))
            (drop (call $invoke (i32.const 100) (i32.const 200) (i32.const 11) (i32.const 220) (i32.const 4) (i32.const 300) (i32.const 4))))
    "#;
    let mut fixture = crate::ecs::NativeTestFixture::new();
    let owner = fixture.add("Script", vec![Box::new(WasmScript::new(write(&dir, "self.wasm", &build(manifest, code))))]);
    let commands = native_frame(&mut fixture);
    assert_eq!(TypedComponent::read_property(&*fixture.scene.get::<WasmScript>(owner).unwrap(), "count"), Some(ArgValue::Int(0)), "guest invokes remain deferred");
    let output = fixture.apply_output(commands).unwrap();
    assert_eq!(TypedComponent::read_property(&*fixture.scene.get::<WasmScript>(owner).unwrap(), "count"), Some(ArgValue::Int(1)));
    let event = output.iter().find_map(|command| match command { ComponentCommand::Emit(event) => event.0.downcast_ref::<crate::ActionEvent>(), _ => None }).unwrap();
    assert_eq!((event.name.as_str(), event.source, event.instigator), ("tick", fixture.id(owner), None));
    let _ = std::fs::remove_dir_all(dir);
}
