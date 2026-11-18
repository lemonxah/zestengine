use std::path::{Path, PathBuf};
use std::process::Command;

use zestengine_core::actions::{ActionCtx, ArgValue};
use zestengine_core::wasm::sandbox::ALL_CAPS;
use zestengine_core::wasm::{Instance, SandboxLimits, UNTRUSTED_CAPS, WasmScript, inspect_module};
use zestengine_core::{ObjectId, TypedComponent};

fn guest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../guest/example-counter")
}

fn wasm32_installed() -> bool {
    Command::new("rustc")
        .args(["--print", "target-libdir", "--target", "wasm32-unknown-unknown"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| Path::new(String::from_utf8_lossy(&out.stdout).trim()).is_dir())
        .unwrap_or(false)
}

fn build_counter() -> Option<Vec<u8>> {
    if !wasm32_installed() {
        eprintln!("skipped: rustup target add wasm32-unknown-unknown");
        return None;
    }
    let dir = guest_dir();
    let status = Command::new(env!("CARGO"))
        .args(["build", "--release", "--target", "wasm32-unknown-unknown", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .status()
        .expect("cargo runs");
    assert!(status.success(), "the example guest builds for wasm32");
    Some(std::fs::read(dir.join("target/wasm32-unknown-unknown/release/example_counter.wasm")).expect("built module"))
}

#[test]
fn real_guest_manifest_passes_untrusted_inspection() {
    let Some(bytes) = build_counter() else { return };
    let (info, _) = inspect_module(&bytes, &SandboxLimits::untrusted(), UNTRUSTED_CAPS).expect("a real wasm32 guest passes the UGC checks");
    assert_eq!(info.manifest.name, "Counter");
    assert_eq!(info.manifest.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["Add", "Reset"]);
    assert_eq!(info.manifest.properties.len(), 1);
    assert!(info.manifest.actions.iter().all(|a| a.ugc_safe));
}

#[test]
fn real_guest_runs_actions_properties_and_events_in_the_sandbox() {
    let Some(bytes) = build_counter() else { return };
    let (mut instance, _) = Instance::load(&bytes, SandboxLimits::untrusted(), UNTRUSTED_CAPS, "counter").unwrap();
    assert_eq!(instance.call_read_property(0).unwrap(), Some(ArgValue::Int(0)));
    instance.call_invoke(0, &[ArgValue::Int(5)], ObjectId::NIL, None, None).unwrap();
    instance.call_invoke(0, &[ArgValue::Int(-2)], ObjectId::NIL, None, None).unwrap();
    assert_eq!(instance.call_read_property(0).unwrap(), Some(ArgValue::Int(3)));
    assert_eq!(instance.take_events(), ["Changed", "Changed"]);
    instance.call_invoke(1, &[], ObjectId::NIL, None, None).unwrap();
    assert_eq!(instance.call_read_property(0).unwrap(), Some(ArgValue::Int(0)));
    let error = instance.call_invoke(7, &[], ObjectId::NIL, None, None).unwrap_err();
    assert_eq!(error.message(), "no action 7");
    instance.call_invoke(0, &[ArgValue::Int(1)], ObjectId::NIL, None, None).expect("a guest failure leaves the instance usable");
}

#[test]
fn real_guest_is_driven_through_the_wasm_script_component() {
    let Some(bytes) = build_counter() else { return };
    let dir = std::env::temp_dir().join(format!("zest-wasm-guest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("counter.wasm");
    std::fs::write(&path, &bytes).unwrap();
    let mut script = WasmScript::new(path.to_string_lossy());
    assert_eq!(TypedComponent::actions(&script).iter().map(|a| a.name).collect::<Vec<_>>(), ["Add", "Reset"]);
    let mut commands = Vec::new();
    let mut ctx = ActionCtx { self_id: ObjectId::new(), instigator: None, commands: &mut commands, reader: None };
    TypedComponent::invoke_action(&mut script, &mut ctx, "Add", &[]).unwrap();
    TypedComponent::invoke_action(&mut script, &mut ctx, "Add", &[ArgValue::Int(41)]).unwrap();
    assert_eq!(TypedComponent::read_property(&script, "count"), Some(ArgValue::Int(42)));
    assert_eq!(script.status().error, None);
    assert!(inspect_module(&bytes, &SandboxLimits::trusted(), ALL_CAPS).is_ok());
    std::fs::remove_dir_all(&dir).ok();
}
