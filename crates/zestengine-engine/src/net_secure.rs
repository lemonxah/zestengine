//! Session security: key stretching, per-connection keys, authenticated
//! encryption, replay protection, handshake cookies and packet-loss
//! accounting.
//!
//! The join secret is stretched once per session into the session key `K`:
//! Argon2id for passphrases (parameters chosen by the host and carried in its
//! CHALLENGE, defaults 64 MiB x 3 passes, 19 MiB x 2 on Android), plain
//! HKDF for random generated secrets (80+ bits, which offline guessing can't
//! touch) and open sessions. A client refuses a CHALLENGE that offers less
//! stretching than its own secret needs, so a forged CHALLENGE can't
//! downgrade a passphrase to a cheap KDF. Every connection key is
//! HKDF-SHA256(K, transcript), where the transcript hash covers the client
//! nonce, the host's stateless cookie and the whole CHALLENGE, so a tampered
//! CHALLENGE yields keys that don't match. Everything after the handshake is
//! sealed with ChaCha20-Poly1305:
//!
//! `[header (authenticated, clear)][counter u64][ciphertext][tag 16]`
//!
//! The 96-bit AEAD nonce is `[direction, 0, 0, 0, counter]`: unique per key
//! and direction because counters only go up. Receivers drop anything that
//! fails authentication or repeats a counter inside a 64-packet sliding
//! window, and estimate the loss rate from the counter gaps.

use chacha20poly1305::aead::AeadInPlace;
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, Tag};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

/// Counter + tag bytes a sealed packet adds to its header and payload.
pub const SEAL_OVERHEAD: usize = 8 + 16;
/// Client -> host.
pub const DIR_TO_HOST: u8 = 0;
/// Host -> client.
pub const DIR_TO_CLIENT: u8 = 1;

const CONN_LABEL: &[u8] = b"zestengine-net connection key v6";
const SESSION_LABEL: &[u8] = b"zestengine-net session key v6";
const RELAY_LABEL: &[u8] = b"zestengine-net relay join key v6";
const WINDOW: u64 = 64;

/// Weakest Argon2id a client accepts for a passphrase (KiB, passes).
pub const MIN_M_KIB: u32 = 8 * 1024;
pub const MIN_PASSES: u32 = 1;
/// Most a client agrees to spend on a host's parameters.
pub const MAX_M_KIB: u32 = 256 * 1024;
pub const MAX_PASSES: u32 = 10;

/// How the join secret becomes the session key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdfMode {
    /// No secret: anyone may join (still encrypted).
    Open = 0,
    /// A passphrase: Argon2id with the carried parameters.
    Stretched = 1,
    /// A random generated secret (80+ bits): HKDF only.
    Random = 2,
}

/// Key-stretching parameters, chosen by the host and carried in the
/// CHALLENGE (and in relay rooms, for joiners' relay proofs). The salt is
/// random per session and survives host migration: it doubles as the
/// session id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdfParams {
    pub mode: KdfMode,
    pub salt: [u8; 16],
    pub m_kib: u32,
    pub passes: u32,
    pub lanes: u8,
}

/// Wire size of [`KdfParams`].
pub const KDF_BYTES: usize = 26;

impl KdfParams {
    /// Defaults for a host with `secret` (Argon2id 64 MiB x 3; 19 MiB x 2 on Android).
    pub fn for_secret(secret: &str) -> Self {
        let mode = if secret.is_empty() {
            KdfMode::Open
        } else if is_generated_secret(secret) {
            KdfMode::Random
        } else {
            KdfMode::Stretched
        };
        let (m_kib, passes) = if cfg!(target_os = "android") { (19 * 1024, 2) } else { (64 * 1024, 3) };
        Self { mode, salt: random_bytes(), m_kib, passes, lanes: 1 }
    }

    pub fn encode(&self, b: &mut Vec<u8>) {
        b.push(self.mode as u8);
        b.extend_from_slice(&self.salt);
        b.extend_from_slice(&self.m_kib.to_le_bytes());
        b.extend_from_slice(&self.passes.to_le_bytes());
        b.push(self.lanes);
    }

    pub fn decode(d: &[u8]) -> Option<Self> {
        let d = d.get(..KDF_BYTES)?;
        let mode = match d[0] {
            0 => KdfMode::Open,
            1 => KdfMode::Stretched,
            2 => KdfMode::Random,
            _ => return None,
        };
        Some(Self {
            mode,
            salt: d[1..17].try_into().ok()?,
            m_kib: u32::from_le_bytes(d[17..21].try_into().ok()?),
            passes: u32::from_le_bytes(d[21..25].try_into().ok()?),
            lanes: d[25],
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(KDF_BYTES);
        self.encode(&mut b);
        b
    }

    /// Client side: whether these parameters are good enough for `secret`
    /// (and not absurdly expensive). A passphrase needs Argon2id at
    /// [`MIN_M_KIB`] / [`MIN_PASSES`] or more.
    pub fn acceptable_for(&self, secret: &str) -> Result<(), String> {
        if self.m_kib > MAX_M_KIB || self.passes > MAX_PASSES || !(1..=8).contains(&self.lanes) {
            return Err(format!("the host asked for excessive key stretching ({} MiB x {} passes)", self.m_kib / 1024, self.passes));
        }
        if secret.is_empty() || is_generated_secret(secret) {
            return Ok(());
        }
        if self.mode != KdfMode::Stretched || self.m_kib < MIN_M_KIB || self.passes < MIN_PASSES {
            return Err("the host offered weak key stretching for a passphrase (refusing to send a guessable handshake)".into());
        }
        Ok(())
    }
}

/// Whether `secret` has the shape of a generated one (4 groups of 4 from
/// the room-code alphabet, about 80 bits), which needs no stretching.
pub fn is_generated_secret(secret: &str) -> bool {
    const ALPHABET: &str = "abcdefghjklmnpqrstuvwxyz23456789";
    let groups: Vec<&str> = secret.split('-').collect();
    groups.len() == 4 && groups.iter().all(|g| g.len() == 4 && g.chars().all(|c| ALPHABET.contains(c)))
}

/// A random join secret that is easy to read out (16 characters, ~80 bits).
pub fn generate_secret() -> String {
    const ALPHABET: &[u8] = b"abcdefghjklmnpqrstuvwxyz23456789";
    let b = random_bytes::<16>();
    let chars: Vec<char> = b.iter().map(|v| ALPHABET[*v as usize % ALPHABET.len()] as char).collect();
    chars.chunks(4).map(|c| c.iter().collect::<String>()).collect::<Vec<_>>().join("-")
}

/// Stretch the join secret into the session key `K` (slow for passphrases:
/// run it off the frame path).
pub fn stretch(secret: &str, p: &KdfParams) -> [u8; 32] {
    let mut out = [0u8; 32];
    match p.mode {
        KdfMode::Stretched => {
            let params = argon2::Params::new(p.m_kib, p.passes, p.lanes as u32, Some(32));
            let ok = params.ok().is_some_and(|params| {
                argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params)
                    .hash_password_into(secret.as_bytes(), &p.salt, &mut out)
                    .is_ok()
            });
            if !ok {
                // Invalid parameters: a key nobody else derives (the join fails cleanly).
                out = random_bytes();
            }
        }
        KdfMode::Open | KdfMode::Random => {
            let hk = Hkdf::<Sha256>::new(Some(&p.salt), secret.as_bytes());
            hk.expand(SESSION_LABEL, &mut out).expect("32 bytes is a valid HKDF-SHA256 output length");
        }
    }
    out
}

/// Session key stretching on a worker thread (Argon2id takes a noticeable
/// fraction of a second).
pub struct KdfJob {
    params: KdfParams,
    rx: std::sync::mpsc::Receiver<[u8; 32]>,
}

impl KdfJob {
    pub fn start(secret: &str, params: KdfParams) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        if params.mode == KdfMode::Stretched {
            let owned = secret.to_owned();
            let worker = tx.clone();
            let spawned = std::thread::Builder::new().name("zestengine-net-kdf".into()).spawn(move || {
                let _ = worker.send(stretch(&owned, &params));
            });
            if spawned.is_err() {
                let _ = tx.send(stretch(secret, &params));
            }
        } else {
            let _ = tx.send(stretch(secret, &params));
        }
        Self { params, rx }
    }

    pub fn params(&self) -> KdfParams {
        self.params
    }

    /// The key once ready (`wait` blocks: deterministic tests).
    pub fn poll(&self, wait: bool) -> Option<[u8; 32]> {
        if wait { self.rx.recv().ok() } else { self.rx.try_recv().ok() }
    }
}

/// SHA-256 over the concatenated parts.
pub fn hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

/// HMAC-SHA256 truncated to 16 bytes.
pub fn mac16(key: &[u8], parts: &[&[u8]]) -> [u8; 16] {
    let mut m = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        m.update(p);
    }
    let full = m.finalize().into_bytes();
    let mut out = [0u8; 16];
    out.copy_from_slice(&full[..16]);
    out
}

/// The key for one connection: from the session key and the handshake transcript.
pub fn conn_key(session_key: &[u8; 32], transcript: &[u8; 32]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(Some(transcript), session_key);
    let mut key = [0u8; 32];
    hk.expand(CONN_LABEL, &mut key).expect("32 bytes is a valid HKDF-SHA256 output length");
    key
}

/// The relay join key (what joiners prove to the relay): one-way from the
/// session key, so the relay never learns anything that opens the session.
pub fn relay_join_key(session_key: &[u8; 32]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::from_prk(session_key).expect("32-byte PRK");
    let mut key = [0u8; 32];
    hk.expand(RELAY_LABEL, &mut key).expect("32 bytes is a valid HKDF-SHA256 output length");
    key
}

/// A token bucket (handshake rate limits).
#[derive(Clone, Copy, Debug)]
pub struct Bucket {
    tokens: f64,
    last: f64,
    rate: f64,
    burst: f64,
}

impl Bucket {
    pub fn new(rate: f64, burst: f64) -> Self {
        Self { tokens: burst, last: 0.0, rate, burst }
    }

    pub fn take(&mut self, now: f64) -> bool {
        self.tokens = (self.tokens + (now - self.last).max(0.0) * self.rate).min(self.burst);
        self.last = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    if getrandom::fill(&mut b).is_err() {
        // No OS randomness: fall back to the clock (keys stay per-connection).
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        for (i, v) in b.iter_mut().enumerate() {
            *v = (t >> ((i % 16) * 8)) as u8 ^ (i as u8).wrapping_mul(151);
        }
    }
    b
}

/// Sliding-window duplicate filter over packet counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayWindow {
    top: u64,
    bits: u64,
    any: bool,
}

impl ReplayWindow {
    /// Whether `ctr` is new (not seen and not older than the window).
    pub fn fresh(&self, ctr: u64) -> bool {
        if !self.any || ctr > self.top {
            return true;
        }
        let off = self.top - ctr;
        off < WINDOW && self.bits & (1 << off) == 0
    }

    pub fn mark(&mut self, ctr: u64) {
        if !self.any {
            *self = Self { top: ctr, bits: 1, any: true };
        } else if ctr > self.top {
            let shift = ctr - self.top;
            self.bits = if shift >= WINDOW { 0 } else { self.bits << shift };
            self.bits |= 1;
            self.top = ctr;
        } else if self.top - ctr < WINDOW {
            self.bits |= 1 << (self.top - ctr);
        }
    }
}

/// Incoming loss estimate from counter gaps, over roughly one-second windows.
#[derive(Clone, Copy, Debug, Default)]
pub struct LossMeter {
    window_start: f64,
    first: Option<u64>,
    highest: u64,
    received: u64,
    /// Smoothed fraction of packets lost (0..1).
    pub loss: f32,
    started: bool,
}

impl LossMeter {
    pub fn on_packet(&mut self, ctr: u64, now: f64) {
        if !self.started {
            self.started = true;
            self.window_start = now;
        }
        let first = *self.first.get_or_insert(ctr);
        if ctr >= first {
            self.received += 1;
            self.highest = self.highest.max(ctr);
        }
        if now - self.window_start >= 1.0 && self.highest >= first {
            let expected = (self.highest - first + 1) as f32;
            let sample = (1.0 - self.received as f32 / expected).clamp(0.0, 1.0);
            self.loss = if self.loss == 0.0 && sample == 0.0 { 0.0 } else { self.loss * 0.5 + sample * 0.5 };
            self.first = Some(self.highest + 1);
            self.received = 0;
            self.window_start = now;
        }
    }
}

/// One direction pair of a connection: seals what we send, opens what we get.
pub struct SecureChannel {
    cipher: ChaCha20Poly1305,
    send_dir: u8,
    recv_dir: u8,
    send_ctr: u64,
    window: ReplayWindow,
    pub loss: LossMeter,
}

fn nonce(dir: u8, ctr: u64) -> Nonce {
    let mut n = [0u8; 12];
    n[0] = dir;
    n[4..].copy_from_slice(&ctr.to_le_bytes());
    Nonce::from(n)
}

impl SecureChannel {
    /// `host_side`: this end is the host (sends to the client).
    pub fn new(key: &[u8; 32], host_side: bool) -> Self {
        let (send_dir, recv_dir) = if host_side { (DIR_TO_CLIENT, DIR_TO_HOST) } else { (DIR_TO_HOST, DIR_TO_CLIENT) };
        Self {
            cipher: ChaCha20Poly1305::new(key.into()),
            send_dir,
            recv_dir,
            send_ctr: 0,
            window: ReplayWindow::default(),
            loss: LossMeter::default(),
        }
    }

    /// Seal `plain` behind a clear, authenticated `header` with the next counter.
    pub fn seal(&mut self, header: &[u8], plain: &[u8]) -> Vec<u8> {
        let ctr = self.send_ctr;
        self.send_ctr += 1;
        self.seal_at(header, ctr, plain)
    }

    /// Seal with an explicit counter (the handshake's first packet resends
    /// byte-identical copies under counter 0).
    pub fn seal_at(&self, header: &[u8], ctr: u64, plain: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(header.len() + SEAL_OVERHEAD + plain.len());
        out.extend_from_slice(header);
        out.extend_from_slice(&ctr.to_le_bytes());
        let body = out.len();
        out.extend_from_slice(plain);
        let (aad, rest) = out.split_at_mut(body);
        let tag = self
            .cipher
            .encrypt_in_place_detached(&nonce(self.send_dir, ctr), &aad[..header.len()], rest)
            .expect("ChaCha20-Poly1305 seals any datagram-sized buffer");
        out.extend_from_slice(&tag);
        out
    }

    /// Make the next sealed packet use at least `ctr` (after a handshake
    /// packet used the low counters).
    pub fn skip_to(&mut self, ctr: u64) {
        self.send_ctr = self.send_ctr.max(ctr);
    }

    /// Authenticate and decrypt a packet whose first `header_len` bytes are
    /// its clear header. None for forgeries, corruption and replays.
    pub fn open(&mut self, data: &[u8], header_len: usize, now: f64) -> Option<Vec<u8>> {
        if data.len() < header_len + SEAL_OVERHEAD {
            return None;
        }
        let (header, rest) = data.split_at(header_len);
        let ctr = u64::from_le_bytes(rest[..8].try_into().ok()?);
        if !self.window.fresh(ctr) {
            return None;
        }
        let (body, tag) = rest[8..].split_at(rest.len() - 8 - 16);
        let mut plain = body.to_vec();
        self.cipher
            .decrypt_in_place_detached(&nonce(self.recv_dir, ctr), header, &mut plain, Tag::from_slice(tag))
            .ok()?;
        self.window.mark(ctr);
        self.loss.on_packet(ctr, now);
        Some(plain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(secret_a: &str, secret_b: &str) -> (SecureChannel, SecureChannel) {
        let p = KdfParams { mode: KdfMode::Random, ..KdfParams::for_secret("x") };
        let t = random_bytes::<32>();
        (SecureChannel::new(&conn_key(&stretch(secret_a, &p), &t), false), SecureChannel::new(&conn_key(&stretch(secret_b, &p), &t), true))
    }

    #[test]
    fn argon2_parameters_are_honored_and_downgrades_refused() {
        let fast = KdfParams { mode: KdfMode::Stretched, salt: [7; 16], m_kib: MIN_M_KIB, passes: 1, lanes: 1 };
        let a = stretch("correct horse", &fast);
        assert_eq!(a, stretch("correct horse", &fast), "deterministic");
        assert_ne!(a, stretch("correct horse", &KdfParams { passes: 2, ..fast }), "passes matter");
        assert_ne!(a, stretch("correct horse", &KdfParams { m_kib: MIN_M_KIB * 2, ..fast }), "memory matters");
        assert_ne!(a, stretch("correct horse", &KdfParams { salt: [8; 16], ..fast }), "salt matters");
        assert_ne!(a, stretch("correct horse", &KdfParams { mode: KdfMode::Random, ..fast }), "Argon2id differs from HKDF");
        assert_eq!(KdfParams::decode(&fast.to_bytes()), Some(fast));
        // A passphrase never goes through a weak or unstretched KDF.
        assert!(fast.acceptable_for("correct horse").is_ok());
        assert!(KdfParams { mode: KdfMode::Random, ..fast }.acceptable_for("correct horse").is_err());
        assert!(KdfParams { mode: KdfMode::Open, ..fast }.acceptable_for("correct horse").is_err());
        assert!(KdfParams { m_kib: 1024, ..fast }.acceptable_for("correct horse").is_err());
        assert!(KdfParams { m_kib: MAX_M_KIB * 2, ..fast }.acceptable_for("").is_err(), "absurd costs refused");
        // Generated secrets may skip stretching; the host picks that for them.
        let g = generate_secret();
        assert!(is_generated_secret(&g), "{g}");
        assert!(KdfParams { mode: KdfMode::Random, ..fast }.acceptable_for(&g).is_ok());
        assert_eq!(KdfParams::for_secret(&g).mode, KdfMode::Random);
        assert_eq!(KdfParams::for_secret("hunter2").mode, KdfMode::Stretched);
        assert_eq!(KdfParams::for_secret("").mode, KdfMode::Open);
        let d = KdfParams::for_secret("hunter2");
        assert!(d.m_kib >= 19 * 1024 && d.passes >= 2, "sane defaults: {d:?}");
    }

    #[test]
    fn seal_open_tamper_and_replay() {
        let (mut client, mut host) = pair("hunter2", "hunter2");
        let p = client.seal(&[0x20], b"hello host");
        assert_eq!(p.len(), 1 + SEAL_OVERHEAD + 10);
        assert_eq!(host.open(&p, 1, 0.0).as_deref(), Some(&b"hello host"[..]));
        assert_eq!(host.open(&p, 1, 0.0), None, "replayed packet dropped");
        let mut bad = client.seal(&[0x20], b"x");
        let last = bad.len() - 1;
        bad[last] ^= 1;
        assert_eq!(host.open(&bad, 1, 0.0), None, "tampered tag");
        let mut hdr = client.seal(&[0x20], b"y");
        hdr[0] = 0x21;
        assert_eq!(host.open(&hdr, 1, 0.0), None, "the clear header is authenticated");
        // Direction separation: the host's own packets don't open on the host.
        let echo = host.seal(&[0x20], b"to client");
        assert_eq!(host.open(&echo, 1, 0.0), None);
        assert_eq!(client.open(&echo, 1, 0.0).as_deref(), Some(&b"to client"[..]));

        let (mut c2, mut h2) = pair("right", "wrong");
        assert_eq!(h2.open(&c2.seal(&[0x20], b"z"), 1, 0.0), None, "wrong secret fails to authenticate");
    }

    #[test]
    fn replay_window_accepts_reordering_within_the_window() {
        let mut w = ReplayWindow::default();
        for c in [5u64, 3, 9, 4, 70, 8] {
            assert!(w.fresh(c), "{c} is new");
            w.mark(c);
        }
        assert!(!w.fresh(9) && !w.fresh(5), "duplicates");
        assert!(!w.fresh(6), "older than the 64-packet window behind 70");
        assert!(w.fresh(69) && w.fresh(71));
    }

    #[test]
    fn loss_meter_estimates_gaps() {
        let mut m = LossMeter::default();
        let mut t = 0.0;
        for ctr in 0..600u64 {
            t += 1.0 / 60.0;
            if ctr % 5 != 0 {
                m.on_packet(ctr, t);
            }
        }
        assert!((m.loss - 0.2).abs() < 0.05, "about 20% lost: {}", m.loss);
    }
}
