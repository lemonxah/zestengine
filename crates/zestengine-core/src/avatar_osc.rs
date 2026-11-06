//! `/avatar/parameters` OSC bridge: the address mapping, the declared-only
//! input filter and the message encoder, with no sockets (the engine owns
//! those, see `face_input.rs`).
//!
//! The convention is VRChat's, so existing tools work: a parameter `Hat` is
//! `/avatar/parameters/Hat`; tools send to UDP 9000 and listen on 9001; the
//! bridge announces the worn avatar with `/avatar/change <id>`. Input is
//! limited to the parameters the avatar *declares*: engine built-ins
//! (`IsLocal`, `Grounded`, ...) and unknown names are ignored, so a tool cannot
//! fake what the engine measures. Each parameter accepts at most 60 writes a
//! second; the newest value that was held back is released on the next poll,
//! so a sweep always ends on its last value.

use std::collections::HashMap;

use crate::avatar_params::{AvatarParamKind, AvatarParamStore};

/// The address prefix all parameters live under.
pub const PARAM_PREFIX: &str = "/avatar/parameters/";
/// The address announcing the worn avatar.
pub const CHANGE_ADDRESS: &str = "/avatar/change";
/// Where tools listen by default.
pub const DEFAULT_OUT_PORT: u16 = 9001;
/// Most writes per parameter per second.
pub const MAX_RATE_HZ: f64 = 60.0;

/// `/avatar/parameters/<name>`.
pub fn param_address(name: &str) -> String {
    format!("{PARAM_PREFIX}{name}")
}

/// The parameter name in an address, if it is one (`None` for other
/// addresses and for an empty name).
pub fn param_name(address: &str) -> Option<&str> {
    address.strip_prefix(PARAM_PREFIX).filter(|n| !n.is_empty())
}

fn pad(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    out.push(0);
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

/// One OSC message setting `name` to `value`, typed by `kind` (Bool `T` / `F`,
/// Int `i`, Float `f`).
pub fn encode_param(name: &str, kind: AvatarParamKind, value: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + name.len());
    pad(&mut out, &param_address(name));
    match kind {
        AvatarParamKind::Bool => pad(&mut out, if value != 0.0 { ",T" } else { ",F" }),
        AvatarParamKind::Int => {
            pad(&mut out, ",i");
            out.extend_from_slice(&(value.round() as i32).to_be_bytes());
        }
        AvatarParamKind::Float => {
            pad(&mut out, ",f");
            out.extend_from_slice(&value.to_be_bytes());
        }
    }
    out
}

/// `/avatar/change <id>`.
pub fn encode_avatar_change(avatar_id: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + avatar_id.len());
    pad(&mut out, CHANGE_ADDRESS);
    pad(&mut out, ",s");
    pad(&mut out, avatar_id);
    out
}

struct Entry {
    slot: usize,
    last_accept: f64,
    /// A value held back by the rate limit, to release later.
    pending: Option<f32>,
}

/// Input filter and output tracker for one worn avatar.
#[derive(Default)]
pub struct OscBridge {
    by_name: HashMap<String, usize>,
    entries: Vec<Entry>,
    /// Values last sent out, parallel to `entries`.
    sent: Vec<f32>,
}

impl OscBridge {
    /// A bridge for the declared (non-built-in) parameters of `store`.
    pub fn new(store: &AvatarParamStore) -> Self {
        let mut b = Self::default();
        for slot in 0..store.len() {
            if store.declared(slot).is_some() {
                b.by_name.insert(store.name(slot).to_owned(), b.entries.len());
                b.entries.push(Entry { slot, last_accept: f64::NEG_INFINITY, pending: None });
                b.sent.push(f32::NAN);
            }
        }
        b
    }

    /// Number of declared parameters in the bridge.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// True when the bridge declares nothing.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// True when the avatar declares a parameter of this name.
    pub fn declares(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// An incoming message: the store slot and the value to write, or `None`
    /// when the address is not a declared parameter or the rate limit held the
    /// value back (see [`Self::poll_pending`]).
    pub fn accept(&mut self, store: &AvatarParamStore, address: &str, value: f32, now: f64) -> Option<(usize, f32)> {
        let name = param_name(address)?;
        let &i = self.by_name.get(name)?;
        if !value.is_finite() {
            return None;
        }
        let e = &mut self.entries[i];
        if now - e.last_accept < 1.0 / MAX_RATE_HZ {
            e.pending = Some(value);
            return None;
        }
        e.last_accept = now;
        e.pending = None;
        Some((e.slot, store.declared(e.slot)?.coerce(value)))
    }

    /// Values the rate limit held back whose interval has passed.
    pub fn poll_pending(&mut self, store: &AvatarParamStore, now: f64, out: &mut Vec<(usize, f32)>) {
        for e in &mut self.entries {
            if let Some(v) = e.pending
                && now - e.last_accept >= 1.0 / MAX_RATE_HZ
            {
                e.pending = None;
                e.last_accept = now;
                if let Some(p) = store.declared(e.slot) {
                    out.push((e.slot, p.coerce(v)));
                }
            }
        }
    }

    /// Declared parameters whose value differs from what was last sent, as
    /// OSC messages. Marks them sent.
    pub fn changed_messages(&mut self, store: &AvatarParamStore, out: &mut Vec<Vec<u8>>) {
        for (i, e) in self.entries.iter().enumerate() {
            let v = store.get(e.slot);
            if self.sent[i].to_bits() != v.to_bits() {
                self.sent[i] = v;
                out.push(encode_param(store.name(e.slot), store.kind(e.slot), v));
            }
        }
    }

    /// Forget what was sent (a new listener, or the avatar changed): the next
    /// [`Self::changed_messages`] sends everything.
    pub fn resend_all(&mut self) {
        self.sent.iter_mut().for_each(|s| *s = f32::NAN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avatar_params::{AvatarParam, AvatarParamSync};

    fn store() -> AvatarParamStore {
        AvatarParamStore::new(&[
            AvatarParam::new("Hat", AvatarParamKind::Bool),
            AvatarParam::new("Outfit", AvatarParamKind::Int),
            AvatarParam::new("Smile", AvatarParamKind::Float),
            AvatarParam { sync: AvatarParamSync::LocalOnly, ..AvatarParam::new("Wetness", AvatarParamKind::Float) },
        ])
    }

    #[test]
    fn addresses_map_both_ways() {
        assert_eq!(param_address("Hat"), "/avatar/parameters/Hat");
        assert_eq!(param_name("/avatar/parameters/Hat"), Some("Hat"));
        assert_eq!(param_name("/avatar/parameters/FT/v2/JawOpen"), Some("FT/v2/JawOpen"), "slashes stay in the name");
        assert_eq!(param_name("/avatar/parameters/"), None);
        assert_eq!(param_name("/avatar/change"), None);
        assert_eq!(param_name("/input/Jump"), None);
    }

    #[test]
    fn only_declared_parameters_pass() {
        let s = store();
        let mut b = OscBridge::new(&s);
        assert_eq!(b.len(), 4, "built-ins are not part of the bridge");
        assert!(b.declares("Hat") && !b.declares("IsLocal") && !b.declares("Grounded"));
        let (slot, v) = b.accept(&s, "/avatar/parameters/Smile", 0.5, 0.0).unwrap();
        assert_eq!((s.name(slot), v), ("Smile", 0.5));
        assert!(b.accept(&s, "/avatar/parameters/IsLocal", 0.0, 0.0).is_none(), "built-ins cannot be faked");
        assert!(b.accept(&s, "/avatar/parameters/Grounded", 1.0, 0.0).is_none());
        assert!(b.accept(&s, "/avatar/parameters/Nope", 1.0, 0.0).is_none());
        assert!(b.accept(&s, "/avatar/change", 1.0, 0.0).is_none());
        assert!(b.accept(&s, "/avatar/parameters/Hat", f32::NAN, 1.0).is_none());
    }

    #[test]
    fn values_are_clamped_to_the_synced_ranges() {
        let s = store();
        let mut b = OscBridge::new(&s);
        assert_eq!(b.accept(&s, "/avatar/parameters/Smile", 7.0, 0.0).unwrap().1, 1.0);
        assert_eq!(b.accept(&s, "/avatar/parameters/Outfit", 999.0, 0.0).unwrap().1, 255.0);
        assert_eq!(b.accept(&s, "/avatar/parameters/Outfit", 2.4, 1.0).unwrap().1, 2.0);
        assert_eq!(b.accept(&s, "/avatar/parameters/Hat", 0.3, 0.0).unwrap().1, 1.0, "any non-zero Bool is true");
        // Local-only floats keep their full range.
        assert_eq!(b.accept(&s, "/avatar/parameters/Wetness", 7.0, 0.0).unwrap().1, 7.0);
    }

    #[test]
    fn writes_are_limited_to_sixty_a_second_and_the_last_one_is_not_lost() {
        let s = store();
        let mut b = OscBridge::new(&s);
        assert!(b.accept(&s, "/avatar/parameters/Smile", 0.1, 0.000).is_some());
        assert!(b.accept(&s, "/avatar/parameters/Smile", 0.2, 0.005).is_none());
        assert!(b.accept(&s, "/avatar/parameters/Smile", 0.3, 0.010).is_none());
        // Another parameter has its own budget.
        assert!(b.accept(&s, "/avatar/parameters/Outfit", 1.0, 0.010).is_some());
        let mut out = Vec::new();
        b.poll_pending(&s, 0.012, &mut out);
        assert!(out.is_empty(), "too early");
        b.poll_pending(&s, 0.020, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 0.3, "the newest held value wins");
        out.clear();
        b.poll_pending(&s, 0.5, &mut out);
        assert!(out.is_empty(), "released once");
    }

    #[test]
    fn messages_encode_with_the_right_type_tag() {
        let m = encode_param("Hat", AvatarParamKind::Bool, 1.0);
        assert_eq!(&m[..24], b"/avatar/parameters/Hat\0\0");
        assert_eq!(&m[24..28], b",T\0\0");
        assert_eq!(m.len(), 28);
        assert_eq!(&encode_param("Hat", AvatarParamKind::Bool, 0.0)[24..28], b",F\0\0");
        let i = encode_param("Outfit", AvatarParamKind::Int, 3.0);
        assert_eq!(i.len() % 4, 0);
        assert_eq!(&i[i.len() - 4..], &3i32.to_be_bytes());
        let f = encode_param("Smile", AvatarParamKind::Float, 0.25);
        assert_eq!(&f[f.len() - 4..], &0.25f32.to_be_bytes());
        let c = encode_avatar_change("abc");
        assert!(c.starts_with(b"/avatar/change\0\0"));
        assert_eq!(&c[c.len() - 4..], b"abc\0");
    }

    #[test]
    fn only_changed_parameters_go_out() {
        let mut s = store();
        let mut b = OscBridge::new(&s);
        let mut out = Vec::new();
        b.changed_messages(&s, &mut out);
        assert_eq!(out.len(), 4, "the first pass sends every declared parameter");
        out.clear();
        b.changed_messages(&s, &mut out);
        assert!(out.is_empty());
        let hat = s.index_of("Hat").unwrap();
        s.set(hat, 1.0);
        b.changed_messages(&s, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out[0].starts_with(b"/avatar/parameters/Hat"));
        b.resend_all();
        out.clear();
        b.changed_messages(&s, &mut out);
        assert_eq!(out.len(), 4);
    }
}
