//! Test modules written in WAT (no wasm32 target needed), plus helpers to put
//! them on disk. Test-only.

use std::path::{Path, PathBuf};

/// Assemble a module whose `zest_manifest` section holds `manifest`.
pub fn build(manifest: &str, body: &str) -> Vec<u8> {
    let hex: String = manifest.bytes().map(|b| format!("\\{b:02x}")).collect();
    let text = format!("(module (@custom \"zest_manifest\" \"{hex}\") {body})");
    wat::parse_str(&text).unwrap_or_else(|e| panic!("bad fixture: {e}\n{text}"))
}

/// A fresh temp directory for one test.
pub fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zest-wasm-{tag}-{}", crate::ObjectId::new()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn write(dir: &Path, name: &str, bytes: &[u8]) -> String {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}

/// Copies the one argument it is called with and hands it back as property 0:
/// a round trip of every value type through guest memory.
pub const ECHO_MANIFEST: &str =
    r#"(name: "Echo", actions: [(name: "Echo", ugc_safe: true)], properties: [(name: "last", ty: String)])"#;
pub const ECHO: &str = r#"
  (memory (export "memory") 2)
  (global $len (mut i32) (i32.const 0))
  (func (export "zest_alloc") (param i32) (result i32) i32.const 1024)
  (func (export "zest_invoke") (param i32 i32 i32) (result i32)
    (global.set $len (local.get 2))
    (memory.copy (i32.const 4096) (local.get 1) (local.get 2))
    i32.const 0)
  (func (export "zest_read_property") (param i32) (result i64)
    (i64.or
      (i64.shl (i64.extend_i32_u (i32.sub (global.get $len) (i32.const 4))) (i64.const 32))
      (i64.const 4100)))
"#;

/// An integer counter: `Add(amount)` (index 0), `Fail` (1), `Ping` (2, emits an
/// event), `Secret` (3, not ugc_safe); property `count`. `factor` scales `Add`.
pub fn counter_manifest() -> &'static str {
    r#"(
        name: "Counter",
        api: 1,
        capabilities: ["events"],
        actions: [
            (name: "Add", label: "Add to count", ugc_safe: true, params: [(name: "amount", ty: Int, default: Some(Int(1)))]),
            (name: "Fail", ugc_safe: true),
            (name: "Ping", ugc_safe: true),
            (name: "Secret"),
        ],
        properties: [(name: "count", ty: Int)],
    )"#
}

pub fn counter(factor: i64) -> String {
    format!(
        r#"
  (import "host" "emit_event" (func $emit (param i32 i32)))
  (memory (export "memory") 2)
  (global $count (mut i64) (i64.const 0))
  (data (i32.const 8192) "pinged")
  (data (i32.const 8200) "nope")
  (func (export "zest_alloc") (param i32) (result i32) i32.const 1024)
  (func (export "zest_invoke") (param $i i32) (param $p i32) (param $l i32) (result i32)
    (if (i32.eq (local.get $i) (i32.const 0)) (then
      (global.set $count (i64.add (global.get $count)
        (i64.mul (i64.load offset=5 (local.get $p)) (i64.const {factor}))))
      (return (i32.const 0))))
    (if (i32.eq (local.get $i) (i32.const 1)) (then (return (i32.const 5))))
    (if (i32.eq (local.get $i) (i32.const 2)) (then
      (call $emit (i32.const 8192) (i32.const 6))
      (return (i32.const 0))))
    (if (i32.eq (local.get $i) (i32.const 3)) (then
      (global.set $count (i64.const -1))
      (return (i32.const 0))))
    i32.const 99)
  (func (export "zest_last_error") (result i64)
    (i64.or (i64.shl (i64.const 4) (i64.const 32)) (i64.const 8200)))
  (func (export "zest_read_property") (param i32) (result i64)
    (i32.store8 (i32.const 4096) (i32.const 1))
    (i64.store offset=1 (i32.const 4096) (global.get $count))
    (i64.or (i64.shl (i64.const 9) (i64.const 32)) (i64.const 4096)))
"#
    )
}

pub const ONE_ACTION: &str = r#"(actions: [(name: "Go", ugc_safe: true)])"#;

/// Wraps `body` (a `zest_invoke` body returning i32) in the boilerplate.
pub fn with_invoke(extra: &str, invoke_body: &str) -> String {
    format!(
        r#"{extra}
  (memory (export "memory") 2)
  (func (export "zest_alloc") (param i32) (result i32) i32.const 1024)
  (func (export "zest_invoke") (param i32 i32 i32) (result i32) {invoke_body})"#
    )
}

pub const SPIN: &str = "(loop $l (br $l)) i32.const 0";
pub const TRAP: &str = "unreachable";
pub const RECURSE: &str = "(i32.add (call $r (i32.const 0)) (i32.const 1))";
pub const RECURSE_FN: &str = "(func $r (param i32) (result i32) (i32.add (call $r (local.get 0)) (i32.const 1)))";
/// Status 0 when the 1000-page grow is refused, 7 when it succeeded.
pub const GROW: &str = "(if (i32.eq (memory.grow (i32.const 1000)) (i32.const -1)) (then (return (i32.const 0)))) i32.const 7";

/// `zest_update` that logs, stores `dt` as float "dt" and the time as "time".
pub const UPDATER_MANIFEST: &str = r#"(name: "Updater", capabilities: ["state"])"#;
pub const UPDATER: &str = r#"
  (import "host" "log" (func $log (param i32 i32)))
  (import "host" "set_float" (func $setf (param i32 i32 f32)))
  (import "host" "get_time" (func $time (result f32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "hello")
  (data (i32.const 16) "dt")
  (data (i32.const 32) "time")
  (func (export "zest_update") (param f32)
    (call $log (i32.const 0) (i32.const 5))
    (call $setf (i32.const 16) (i32.const 2) (local.get 0))
    (call $setf (i32.const 32) (i32.const 4) (call $time)))
"#;

/// Reads `Health.current` of its own object into float "hp" and queues
/// `Health.Damage(25)` on its own object, every update.
pub const DRIVER_MANIFEST: &str = r#"(name: "Driver", capabilities: ["read", "invoke", "state"])"#;
pub const DRIVER: &str = r#"
  (import "host" "set_float" (func $setf (param i32 i32 f32)))
  (import "host" "get_property" (func $get (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (import "host" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 16) "hp")
  (data (i32.const 200) "Health")
  (data (i32.const 210) "current")
  (data (i32.const 220) "Damage")
  (data (i32.const 240) "\01\00\00\00\02\00\00\c8\41")
  (func (export "zest_update") (param f32)
    (drop (call $get (i32.const 100) (i32.const 200) (i32.const 6) (i32.const 210) (i32.const 7) (i32.const 300) (i32.const 32)))
    (call $setf (i32.const 16) (i32.const 2) (f32.load offset=1 (i32.const 300)))
    (drop (call $invoke (i32.const 100) (i32.const 200) (i32.const 6) (i32.const 220) (i32.const 6) (i32.const 240) (i32.const 9))))
"#;

/// `zest_invoke` that reads `Health.current` of its own object (an all-zero
/// object id): stores the status of the read as float "rc" and the value as
/// float "hp" when the read worked.
pub const LOOKER_MANIFEST: &str = r#"(name: "Looker", capabilities: ["read", "state"], actions: [(name: "Look", ugc_safe: true)])"#;
pub const LOOKER: &str = r#"
  (import "host" "set_float" (func $setf (param i32 i32 f32)))
  (import "host" "get_property" (func $get (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 16) "rc")
  (data (i32.const 24) "hp")
  (data (i32.const 200) "Health")
  (data (i32.const 210) "current")
  (func (export "zest_alloc") (param i32) (result i32) i32.const 1024)
  (func (export "zest_invoke") (param i32 i32 i32) (result i32)
    (local $rc i32)
    (local.set $rc (call $get (i32.const 100) (i32.const 200) (i32.const 6) (i32.const 210) (i32.const 7) (i32.const 300) (i32.const 32)))
    (call $setf (i32.const 16) (i32.const 2) (f32.convert_i32_s (local.get $rc)))
    (if (i32.gt_s (local.get $rc) (i32.const 0))
      (then (call $setf (i32.const 24) (i32.const 2) (f32.load offset=1 (i32.const 300)))))
    i32.const 0)
"#;
