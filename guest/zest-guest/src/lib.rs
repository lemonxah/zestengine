//! Write a ZestEngine WASM component in Rust.
//!
//! ```ignore
//! use zest_guest::{Module, Value};
//!
//! zest_guest::manifest!(r#"(
//!     name: "Counter",
//!     capabilities: ["events"],
//!     actions: [(name: "Add", ugc_safe: true, params: [(name: "amount", ty: Int, default: Some(Int(1)))])],
//!     properties: [(name: "count", ty: Int)],
//! )"#);
//!
//! #[derive(Default)]
//! struct Counter(i64);
//!
//! impl Module for Counter {
//!     fn invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String> {
//!         match action {
//!             0 => self.0 += args[0].as_i64(),   // actions are numbered as in the manifest
//!             _ => return Err("unknown action".into()),
//!         }
//!         Ok(())
//!     }
//!     fn property(&mut self, index: u32) -> Option<Value> {
//!         (index == 0).then(|| Value::Int(self.0))
//!     }
//! }
//!
//! zest_guest::export!(Counter::default());
//! ```
//!
//! Build with `cargo build --release --target wasm32-unknown-unknown` and set
//! the `crate-type = ["cdylib"]` library type. The exact ABI is documented in
//! `crates/zestengine-core/src/wasm/mod.rs`.

pub mod codec;
pub use codec::Value;

/// Embed the manifest (RON text) as the `zest_manifest` custom section.
#[macro_export]
macro_rules! manifest {
    ($text:expr) => {
        #[used]
        #[unsafe(link_section = "zest_manifest")]
        static ZEST_MANIFEST: [u8; $text.len()] = {
            let src = $text.as_bytes();
            let mut out = [0u8; $text.len()];
            let mut i = 0;
            while i < src.len() {
                out[i] = src[i];
                i += 1;
            }
            out
        };
    };
}

/// The behaviour of a component. Indices follow the order in the manifest.
pub trait Module {
    /// Run action number `action` with its arguments (already typed and in
    /// parameter order). `Err` is reported to the binding that called it.
    fn invoke(&mut self, action: u32, args: &[Value]) -> Result<(), String>;
    /// Current value of property number `index`.
    fn property(&mut self, _index: u32) -> Option<Value> {
        None
    }
    /// Once per frame while playing. Only called when the module was exported
    /// with `export!(init, update)`.
    fn update(&mut self, _dt: f32) {}
}

/// Export the ABI entry points for a module value (an expression run once,
/// on the first call). `export!(init, update)` also exports `zest_update`,
/// so the host calls `Module::update` every frame.
#[macro_export]
macro_rules! export {
    ($init:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn zest_alloc(len: u32) -> u32 {
            $crate::rt::alloc(len)
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn zest_invoke(action: u32, ptr: u32, len: u32) -> i32 {
            $crate::rt::invoke(|| ::std::boxed::Box::new($init), action, ptr, len)
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn zest_last_error() -> i64 {
            $crate::rt::last_error()
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn zest_read_property(index: u32) -> i64 {
            $crate::rt::read_property(|| ::std::boxed::Box::new($init), index)
        }
    };
    ($init:expr, update) => {
        $crate::export!($init);
        #[unsafe(no_mangle)]
        pub extern "C" fn zest_update(dt: f32) {
            $crate::rt::update(|| ::std::boxed::Box::new($init), dt)
        }
    };
}

#[doc(hidden)]
pub mod rt {
    use super::{codec, Module};
    use std::cell::RefCell;

    #[derive(Default)]
    struct State {
        module: Option<Box<dyn Module>>,
        /// The buffer the host writes argument blocks into.
        scratch: Vec<u8>,
        /// The last property value / error message, kept alive for the host to read.
        out: Vec<u8>,
        error: String,
    }

    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State::default());
    }

    fn pack(bytes: &[u8]) -> i64 {
        ((bytes.len() as i64) << 32) | (bytes.as_ptr() as usize as u32 as i64)
    }

    pub fn alloc(len: u32) -> u32 {
        STATE.with(|s| {
            let s = &mut *s.borrow_mut();
            if s.scratch.len() < len as usize {
                s.scratch.resize(len as usize, 0);
            }
            s.scratch.as_ptr() as usize as u32
        })
    }

    fn module<'a>(s: &'a mut State, init: impl FnOnce() -> Box<dyn Module>) -> &'a mut dyn Module {
        s.module.get_or_insert_with(init).as_mut()
    }

    pub fn invoke(init: impl FnOnce() -> Box<dyn Module>, action: u32, ptr: u32, len: u32) -> i32 {
        STATE.with(|s| {
            let s = &mut *s.borrow_mut();
            // SAFETY: the host wrote `len` bytes at `ptr`, inside this module's memory.
            let bytes = unsafe { std::slice::from_raw_parts(ptr as usize as *const u8, len as usize) };
            let args = match codec::decode_args(bytes) {
                Ok(a) => a,
                Err(e) => {
                    s.error = e.to_owned();
                    return 1;
                }
            };
            match module(s, init).invoke(action, &args) {
                Ok(()) => 0,
                Err(e) => {
                    s.error = e;
                    1
                }
            }
        })
    }

    pub fn last_error() -> i64 {
        STATE.with(|s| {
            let s = &mut *s.borrow_mut();
            s.out = s.error.clone().into_bytes();
            pack(&s.out)
        })
    }

    pub fn read_property(init: impl FnOnce() -> Box<dyn Module>, index: u32) -> i64 {
        STATE.with(|s| {
            let s = &mut *s.borrow_mut();
            match module(s, init).property(index) {
                Some(v) => {
                    s.out.clear();
                    codec::encode_value(&mut s.out, &v);
                    pack(&s.out)
                }
                None => 0,
            }
        })
    }

    pub fn update(init: impl FnOnce() -> Box<dyn Module>, dt: f32) {
        STATE.with(|s| module(&mut s.borrow_mut(), init).update(dt));
    }
}

// ----------------------------------------------------------------- host calls

#[cfg(target_arch = "wasm32")]
mod sys {
    #[link(wasm_import_module = "host")]
    unsafe extern "C" {
        pub fn log(ptr: *const u8, len: u32);
        pub fn get_dt() -> f32;
        pub fn get_time() -> f32;
        pub fn get_self(out: *mut u8);
        pub fn get_instigator(out: *mut u8) -> i32;
        pub fn get_float(ptr: *const u8, len: u32) -> f32;
        pub fn set_float(ptr: *const u8, len: u32, v: f32);
        pub fn emit_event(ptr: *const u8, len: u32);
        pub fn get_property(obj: *const u8, cp: *const u8, cl: u32, pp: *const u8, pl: u32, out: *mut u8, cap: u32) -> i32;
        pub fn invoke(obj: *const u8, cp: *const u8, cl: u32, ap: *const u8, al: u32, args: *const u8, args_len: u32) -> i32;
    }
}

/// Off wasm32 the host functions don't exist; these keep the crate building
/// (and unit-testable) on the development machine.
#[cfg(not(target_arch = "wasm32"))]
mod sys {
    pub unsafe fn log(_: *const u8, _: u32) {}
    pub unsafe fn get_dt() -> f32 { 0.0 }
    pub unsafe fn get_time() -> f32 { 0.0 }
    pub unsafe fn get_self(_: *mut u8) {}
    pub unsafe fn get_instigator(_: *mut u8) -> i32 { 0 }
    pub unsafe fn get_float(_: *const u8, _: u32) -> f32 { 0.0 }
    pub unsafe fn set_float(_: *const u8, _: u32, _: f32) {}
    pub unsafe fn emit_event(_: *const u8, _: u32) {}
    pub unsafe fn get_property(_: *const u8, _: *const u8, _: u32, _: *const u8, _: u32, _: *mut u8, _: u32) -> i32 { -2 }
    pub unsafe fn invoke(_: *const u8, _: *const u8, _: u32, _: *const u8, _: u32, _: *const u8, _: u32) -> i32 { -2 }
}

/// An object id: 16 bytes, all zero = "this object" / none.
pub type ObjectId = [u8; 16];

pub fn log(msg: &str) {
    unsafe { sys::log(msg.as_ptr(), msg.len() as u32) }
}
pub fn dt() -> f32 {
    unsafe { sys::get_dt() }
}
pub fn time() -> f32 {
    unsafe { sys::get_time() }
}
pub fn this_object() -> ObjectId {
    let mut id = [0u8; 16];
    unsafe { sys::get_self(id.as_mut_ptr()) };
    id
}
pub fn instigator() -> Option<ObjectId> {
    let mut id = [0u8; 16];
    (unsafe { sys::get_instigator(id.as_mut_ptr()) } == 1).then_some(id)
}
/// Needs the `state` capability.
pub fn get_float(key: &str) -> f32 {
    unsafe { sys::get_float(key.as_ptr(), key.len() as u32) }
}
/// Needs the `state` capability.
pub fn set_float(key: &str, v: f32) {
    unsafe { sys::set_float(key.as_ptr(), key.len() as u32, v) }
}
/// Needs the `events` capability.
pub fn emit_event(name: &str) {
    unsafe { sys::emit_event(name.as_ptr(), name.len() as u32) }
}
/// Read a property of a component (`object` all zero = this object). Needs the
/// `read` capability and only works inside `Module::update`.
pub fn get_property(object: ObjectId, component: &str, property: &str) -> Option<Value> {
    let mut out = [0u8; 512];
    let n = unsafe {
        sys::get_property(
            object.as_ptr(), component.as_ptr(), component.len() as u32, property.as_ptr(), property.len() as u32,
            out.as_mut_ptr(), out.len() as u32,
        )
    };
    if n <= 0 {
        return None;
    }
    let mut pos = 0;
    codec::decode_value(&out[..n as usize], &mut pos).ok()
}
/// Queue a call of `component.action` on `object` (zero = this object). Needs
/// the `invoke` capability. Returns 0 when queued.
pub fn invoke(object: ObjectId, component: &str, action: &str, args: &[Value]) -> i32 {
    let block = codec::encode_args(args);
    unsafe {
        sys::invoke(
            object.as_ptr(), component.as_ptr(), component.len() as u32, action.as_ptr(), action.len() as u32,
            block.as_ptr(), block.len() as u32,
        )
    }
}
