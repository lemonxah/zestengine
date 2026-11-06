//! A counter other objects can drive from event bindings:
//! `Add(amount)` and `Reset` actions, a `count` property, and a `Changed`
//! event broadcast after every change.
//!
//! cargo build --release --target wasm32-unknown-unknown
//! -> target/wasm32-unknown-unknown/release/example_counter.wasm

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
