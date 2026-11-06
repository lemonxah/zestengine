//! Wire format for synced avatar parameters (`P_AVPARAM`).
//!
//! Every peer wearing an avatar has the same declared parameters (they ship in
//! the avatar package), so the packet carries no names: the *schema* is the
//! synced parameters sorted by name, identified by a 32-bit hash of their
//! names and kinds. A packet is `[schema hash: u32 LE][payload]`, and the
//! payload is one byte per Int / Float (Int 0..=255; Float -1..=1 as a signed
//! byte over 127, the same quantisation remote copies show) followed by the
//! Bools packed eight to a byte. A full 256-bit budget is therefore at most
//! 32 payload bytes. A receiver whose schema hash differs (another version of
//! the avatar) drops the packet instead of misreading it.
//!
//! The codec is pure: it reads values from, and writes them to, plain slices
//! or an [`AvatarParamStore`] through slots resolved once when the schema is
//! built. [`SyncClock`] holds the send cadence: changes go out at 10 Hz,
//! 30 Hz while a puppet is open, and the full state repeats every second so
//! a lost packet or a late joiner catches up.

use crate::avatar_params::{AvatarParam, AvatarParamKind, AvatarParamStore};

/// Seconds between packets while values are changing.
pub const SEND_INTERVAL: f64 = 0.1;
/// Seconds between packets while a puppet or radial control is being moved.
pub const FAST_INTERVAL: f64 = 1.0 / 30.0;
/// Seconds between repeats of an unchanged state.
pub const KEYFRAME_INTERVAL: f64 = 1.0;
/// Bytes of the schema hash at the start of every packet.
pub const HEADER_BYTES: usize = 4;

/// Why a synced avatar-parameter packet was dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncError {
    /// The sender's parameters differ from ours.
    SchemaMismatch {
        /// The sender's schema hash.
        got: u32,
        /// Our schema hash.
        want: u32,
    },
    /// The packet is shorter or longer than the schema needs.
    BadLength {
        /// The packet's byte length.
        got: usize,
        /// The byte length the schema needs.
        want: usize,
    },
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaMismatch { got, want } => write!(f, "avatar parameter schema {got:08x} does not match {want:08x}"),
            Self::BadLength { got, want } => write!(f, "avatar parameter packet is {got} bytes, expected {want}"),
        }
    }
}

/// One parameter in the wire schema.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemaEntry {
    /// Parameter name.
    pub name: String,
    /// Parameter kind.
    pub kind: AvatarParamKind,
    /// Store slot (`usize::MAX` when the schema was built from declarations).
    pub slot: usize,
}

/// The synced parameters of one avatar in wire order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AvParamSchema {
    entries: Vec<SchemaEntry>,
    hash: u32,
    /// Entries that are numbers (Int / Float), in schema order, as indices
    /// into `entries`; then the Bools.
    numbers: Vec<usize>,
    bools: Vec<usize>,
}

/// 32-bit FNV-1a over names and kinds.
fn schema_hash(entries: &[SchemaEntry]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut eat = |b: u8| {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    };
    for e in entries {
        e.name.bytes().for_each(&mut eat);
        eat(0);
        eat(match e.kind {
            AvatarParamKind::Bool => 1,
            AvatarParamKind::Int => 2,
            AvatarParamKind::Float => 3,
        });
        eat(b'\n');
    }
    h
}

impl AvParamSchema {
    fn from_entries(mut entries: Vec<SchemaEntry>) -> Self {
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries.dedup_by(|a, b| a.name == b.name);
        let hash = schema_hash(&entries);
        let numbers = (0..entries.len()).filter(|&i| entries[i].kind != AvatarParamKind::Bool).collect();
        let bools = (0..entries.len()).filter(|&i| entries[i].kind == AvatarParamKind::Bool).collect();
        Self { entries, hash, numbers, bools }
    }

    /// The synced parameters among `params` (local-only, nameless and
    /// duplicate ones are not part of the schema).
    pub fn from_params(params: &[AvatarParam]) -> Self {
        Self::from_entries(
            params
                .iter()
                .filter(|p| p.is_synced() && !p.name.is_empty())
                .map(|p| SchemaEntry { name: p.name.clone(), kind: p.kind, slot: usize::MAX })
                .collect(),
        )
    }

    /// The same schema with store slots filled in, for `encode` / `apply`.
    pub fn from_store(store: &AvatarParamStore) -> Self {
        Self::from_entries(
            store
                .synced_slots()
                .into_iter()
                .map(|slot| SchemaEntry { name: store.name(slot).to_owned(), kind: store.kind(slot), slot })
                .collect(),
        )
    }

    /// The schema's hash (identifies the parameter set).
    pub fn hash(&self) -> u32 {
        self.hash
    }
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// True when the schema has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// The entries in wire order.
    pub fn entries(&self) -> &[SchemaEntry] {
        &self.entries
    }
    /// Payload size in bytes (the header is extra).
    pub fn payload_bytes(&self) -> usize {
        self.numbers.len() + self.bools.len().div_ceil(8)
    }
    /// Whole packet size in bytes.
    pub fn packet_bytes(&self) -> usize {
        HEADER_BYTES + self.payload_bytes()
    }

    /// Append a packet for `values` (parallel to [`Self::entries`]).
    pub fn encode_values(&self, values: &[f32], out: &mut Vec<u8>) {
        out.extend_from_slice(&self.hash.to_le_bytes());
        for &i in &self.numbers {
            let v = values.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0);
            out.push(match self.entries[i].kind {
                AvatarParamKind::Int => v.round().clamp(0.0, 255.0) as u8,
                _ => ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8,
            });
        }
        let mut byte = 0u8;
        for (n, &i) in self.bools.iter().enumerate() {
            if values.get(i).copied().unwrap_or(0.0) != 0.0 {
                byte |= 1 << (n % 8);
            }
            if n % 8 == 7 {
                out.push(byte);
                byte = 0;
            }
        }
        if !self.bools.len().is_multiple_of(8) {
            out.push(byte);
        }
    }

    /// Read a packet into `out` (resized to one value per entry).
    pub fn decode_values(&self, data: &[u8], out: &mut Vec<f32>) -> Result<(), SyncError> {
        if data.len() < HEADER_BYTES {
            return Err(SyncError::BadLength { got: data.len(), want: self.packet_bytes() });
        }
        let got = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if got != self.hash {
            return Err(SyncError::SchemaMismatch { got, want: self.hash });
        }
        if data.len() != self.packet_bytes() {
            return Err(SyncError::BadLength { got: data.len(), want: self.packet_bytes() });
        }
        out.clear();
        out.resize(self.entries.len(), 0.0);
        let body = &data[HEADER_BYTES..];
        for (n, &i) in self.numbers.iter().enumerate() {
            out[i] = match self.entries[i].kind {
                AvatarParamKind::Int => body[n] as f32,
                _ => (body[n] as i8) as f32 / 127.0,
            };
        }
        let bits = &body[self.numbers.len()..];
        for (n, &i) in self.bools.iter().enumerate() {
            out[i] = ((bits[n / 8] >> (n % 8)) & 1) as f32;
        }
        Ok(())
    }

    /// Append a packet with the store's current values.
    pub fn encode(&self, store: &AvatarParamStore, out: &mut Vec<u8>) {
        let values: Vec<f32> = self.entries.iter().map(|e| if e.slot < store.len() { store.get(e.slot) } else { 0.0 }).collect();
        self.encode_values(&values, out);
    }

    /// Write decoded values into the store (through the normal coercion).
    pub fn apply(&self, values: &[f32], store: &mut AvatarParamStore) {
        for (e, &v) in self.entries.iter().zip(values) {
            if e.slot < store.len() {
                store.set(e.slot, v);
            }
        }
    }

    /// The values as a remote copy would see them: floats quantised, ints and
    /// bools rounded. What `encode_values` + `decode_values` give back.
    pub fn quantise(&self, values: &[f32]) -> Vec<f32> {
        let mut bytes = Vec::with_capacity(self.packet_bytes());
        self.encode_values(values, &mut bytes);
        let mut out = Vec::new();
        let _ = self.decode_values(&bytes, &mut out);
        out
    }
}

/// When to send: 10 Hz while values change (30 Hz with a puppet open), and
/// the full state once a second regardless.
#[derive(Clone, Copy, Debug, Default)]
pub struct SyncClock {
    last_send: f64,
    sent_once: bool,
}

impl SyncClock {
    /// Whether to send at `now` (seconds on any steady clock). `changed`: a
    /// synced value differs from what was last sent. `fast`: a puppet is open.
    pub fn poll(&mut self, now: f64, changed: bool, fast: bool) -> bool {
        let since = now - self.last_send;
        let due = !self.sent_once || since >= KEYFRAME_INTERVAL || (changed && since >= if fast { FAST_INTERVAL } else { SEND_INTERVAL });
        if due {
            self.last_send = now;
            self.sent_once = true;
        }
        due
    }
}

/// One step of a remote float catching up with the last received value over
/// the send interval, so radial changes look smooth between packets.
pub fn smooth_step(current: f32, target: f32, dt: f32, interval: f32) -> f32 {
    let k = if interval <= 1e-4 { 1.0 } else { (dt / interval).clamp(0.0, 1.0) };
    current + (target - current) * k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avatar_params::AvatarParamSync;

    fn p(name: &str, kind: AvatarParamKind) -> AvatarParam {
        AvatarParam::new(name, kind)
    }

    fn sample() -> Vec<AvatarParam> {
        vec![
            p("Hat", AvatarParamKind::Bool),
            p("Outfit", AvatarParamKind::Int),
            p("Smile", AvatarParamKind::Float),
            p("Cape", AvatarParamKind::Bool),
            AvatarParam { sync: AvatarParamSync::LocalOnly, ..p("Menu", AvatarParamKind::Bool) },
        ]
    }

    #[test]
    fn schema_is_sorted_synced_only_and_order_independent() {
        let a = AvParamSchema::from_params(&sample());
        assert_eq!(a.entries().iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), vec!["Cape", "Hat", "Outfit", "Smile"]);
        let mut rev = sample();
        rev.reverse();
        assert_eq!(AvParamSchema::from_params(&rev).hash(), a.hash());
        // 2 numbers + 2 bools in one byte.
        assert_eq!(a.payload_bytes(), 3);
        assert_eq!(a.packet_bytes(), 7);
    }

    #[test]
    fn round_trip_quantises_like_a_remote_copy() {
        let s = AvParamSchema::from_params(&sample());
        // Cape, Hat, Outfit, Smile
        let values = [1.0, 0.0, 200.0, 0.5];
        let mut bytes = Vec::new();
        s.encode_values(&values, &mut bytes);
        assert_eq!(bytes.len(), s.packet_bytes());
        let mut back = Vec::new();
        s.decode_values(&bytes, &mut back).unwrap();
        assert_eq!(back[0], 1.0);
        assert_eq!(back[1], 0.0);
        assert_eq!(back[2], 200.0);
        assert!((back[3] - 0.5).abs() < 0.005 && back[3] != 0.5, "0.5 arrives as 64/127: {}", back[3]);
        assert_eq!(s.quantise(&values), back);
        // Out-of-range values clamp to what the wire can hold.
        let mut b2 = Vec::new();
        s.encode_values(&[1.0, 1.0, 999.0, -7.0], &mut b2);
        s.decode_values(&b2, &mut back).unwrap();
        assert_eq!((back[2], back[3]), (255.0, -1.0));
        // Negative floats keep their sign.
        let mut b3 = Vec::new();
        s.encode_values(&[0.0, 0.0, 0.0, -0.25], &mut b3);
        s.decode_values(&b3, &mut back).unwrap();
        assert!((back[3] + 0.25).abs() < 0.005);
    }

    #[test]
    fn many_bools_pack_eight_to_a_byte() {
        let params: Vec<AvatarParam> = (0..256).map(|i| p(&format!("B{i:03}"), AvatarParamKind::Bool)).collect();
        let s = AvParamSchema::from_params(&params);
        assert_eq!(s.payload_bytes(), 32, "the whole 256-bit budget fits in 32 bytes");
        let values: Vec<f32> = (0..256).map(|i| (i % 3 == 0) as u8 as f32).collect();
        let mut bytes = Vec::new();
        s.encode_values(&values, &mut bytes);
        assert_eq!(bytes.len(), 36);
        let mut back = Vec::new();
        s.decode_values(&bytes, &mut back).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn schema_hash_mismatch_is_rejected() {
        let a = AvParamSchema::from_params(&sample());
        let mut other = sample();
        other.push(p("Extra", AvatarParamKind::Bool));
        let b = AvParamSchema::from_params(&other);
        assert_ne!(a.hash(), b.hash());
        let mut bytes = Vec::new();
        a.encode_values(&[0.0; 4], &mut bytes);
        let mut out = Vec::new();
        assert_eq!(b.decode_values(&bytes, &mut out), Err(SyncError::SchemaMismatch { got: a.hash(), want: b.hash() }));
        // A kind change alone also changes the hash.
        let mut kinds = sample();
        kinds[0].kind = AvatarParamKind::Int;
        assert_ne!(AvParamSchema::from_params(&kinds).hash(), a.hash());
        // Truncated and padded packets are refused too.
        let mut ok = Vec::new();
        a.encode_values(&[0.0; 4], &mut ok);
        assert!(matches!(a.decode_values(&ok[..ok.len() - 1], &mut out), Err(SyncError::BadLength { .. })));
        ok.push(0);
        assert!(matches!(a.decode_values(&ok, &mut out), Err(SyncError::BadLength { .. })));
        assert!(matches!(a.decode_values(&[1, 2], &mut out), Err(SyncError::BadLength { .. })));
    }

    #[test]
    fn store_round_trip_goes_through_slots() {
        let mut tx = AvatarParamStore::new(&sample());
        let mut rx = AvatarParamStore::new(&sample());
        let (a, b) = (AvParamSchema::from_store(&tx), AvParamSchema::from_store(&rx));
        assert_eq!(a.hash(), b.hash());
        let hat = tx.index_of("Hat").unwrap();
        let outfit = tx.index_of("Outfit").unwrap();
        tx.set(hat, 1.0);
        tx.set(outfit, 3.0);
        let mut bytes = Vec::new();
        a.encode(&tx, &mut bytes);
        let mut values = Vec::new();
        b.decode_values(&bytes, &mut values).unwrap();
        b.apply(&values, &mut rx);
        assert_eq!(rx.get(rx.index_of("Hat").unwrap()), 1.0);
        assert_eq!(rx.get(rx.index_of("Outfit").unwrap()), 3.0);
        assert_eq!(rx.get(rx.index_of("Cape").unwrap()), 0.0);
    }

    #[test]
    fn clock_sends_changes_at_ten_hz_puppets_faster_and_repeats_every_second() {
        let mut c = SyncClock::default();
        assert!(c.poll(0.0, false, false), "the first poll sends the state");
        assert!(!c.poll(0.05, true, false), "coalesced inside 100 ms");
        assert!(c.poll(0.11, true, false));
        assert!(!c.poll(0.15, false, false), "nothing changed");
        assert!(!c.poll(0.13, true, true), "fast mode still needs 33 ms");
        assert!(c.poll(0.11 + FAST_INTERVAL + 1e-6, true, true));
        let t = 0.11 + FAST_INTERVAL + 1e-6;
        assert!(!c.poll(t + 0.9, false, false));
        assert!(c.poll(t + 1.01, false, false), "unchanged state repeats once a second");
    }

    #[test]
    fn remote_floats_ease_toward_the_last_value() {
        assert_eq!(smooth_step(0.0, 1.0, 0.05, 0.1), 0.5);
        assert_eq!(smooth_step(0.0, 1.0, 0.2, 0.1), 1.0);
        assert_eq!(smooth_step(0.3, 0.8, 0.016, 0.0), 0.8);
    }
}
