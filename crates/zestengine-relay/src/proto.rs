//! Relay wire protocol, shared by the server and the engine's relay client.
//!
//! Every relay packet starts with one tag byte in `0xF1..=0xFB`, which never
//! collides with a game packet (those start with a tag below `0x80`), so one
//! UDP socket carries relay control, relayed data and direct game traffic.
//!
//! | Packet | Direction | Body |
//! |---|---|---|
//! | REGISTER | peer -> relay | protocol, role, room code, nonce, LAN endpoints, cookie, role proof (padded to [`MIN_REGISTER`]) |
//! | COOKIE | relay -> peer | stateless cookie, the room's key-stretching parameters |
//! | REGISTERED | relay -> peer | nonce, slot, own punch token, room code, public endpoint, claim nonce |
//! | PEER | relay -> peer | slot, punch token, public endpoint, LAN endpoints |
//! | DATA | peer <-> relay | slot (destination going in, source coming out), payload |
//! | PUNCH / PUNCH_ACK | peer <-> peer | sender's punch token and slot |
//! | CLAIM | member -> relay | nonce, claim nonce, room-owner signature (host migration) |
//! | KEEPALIVE / LEAVE | peer -> relay | nonce |
//! | ERROR | relay -> peer | reason |
//!
//! Authentication (protocol 2): a REGISTER without a valid cookie only gets
//! a COOKIE back (no state is kept, and the reply is never larger than the
//! request). A host creating a room registers the room's owner key (Ed25519),
//! a join key and the key-stretching parameters, and with an operator token
//! configured proves it knows the token. A client proves it knows the
//! session secret with an HMAC over the cookie under the join key (derived
//! from the stretched secret; it never reveals the secret). A member taking
//! the room over after the host left signs a CLAIM with the room owner key
//! that the host shared inside the encrypted session.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use chacha20poly1305::aead::AeadInPlace;
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, Tag};
use curve25519_dalek::montgomery::MontgomeryPoint;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::Sha256;

/// Bumped when the relay wire format changes.
pub const RELAY_PROTOCOL: u16 = 3;
/// Default UDP port of a relay server.
pub const DEFAULT_PORT: u16 = 47800;
/// The room host's slot; clients get 1, 2, ...
pub const HOST_SLOT: u32 = 0;
/// Largest relayed payload (a sealed game datagram is well under this).
pub const MAX_PAYLOAD: usize = 1400;
/// DATA header bytes (tag + slot).
pub const DATA_HEADER: usize = 5;
/// REGISTER packets are padded to at least this size, so nothing the relay
/// answers an unauthenticated sender with is larger than what it got.
pub const MIN_REGISTER: usize = 160;
/// Cookie and proof sizes.
pub const COOKIE_LEN: usize = 16;
pub const PROOF_LEN: usize = 16;
/// Largest key-stretching parameter blob a room carries.
pub const MAX_KDF: usize = 64;
/// A join key sealed to the relay: ephemeral X25519 key, ciphertext, tag.
pub const SEALED_JOIN_KEY_LEN: usize = 32 + 32 + 16;

pub const R_REGISTER: u8 = 0xF1;
pub const R_REGISTERED: u8 = 0xF2;
pub const R_PEER: u8 = 0xF3;
pub const R_DATA: u8 = 0xF4;
pub const R_PUNCH: u8 = 0xF5;
pub const R_PUNCH_ACK: u8 = 0xF6;
pub const R_KEEPALIVE: u8 = 0xF7;
pub const R_ERROR: u8 = 0xF8;
pub const R_LEAVE: u8 = 0xF9;
pub const R_COOKIE: u8 = 0xFA;
pub const R_CLAIM: u8 = 0xFB;
/// Host: the room's ranked successors (relay slots), mirroring the session's
/// migration plan; the relay lets only the best-ranked live one claim.
pub const R_SUCCESSORS: u8 = 0xFC;
/// Most successor slots a SUCCESSORS packet carries.
pub const MAX_SUCCESSORS: usize = 255;

pub const ROLE_HOST: u8 = 0;
pub const ROLE_CLIENT: u8 = 1;

const CODE_ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const MAX_LOCALS: usize = 4;

/// Whether a datagram is relay control / relayed data rather than game traffic.
pub fn is_relay_packet(data: &[u8]) -> bool {
    data.first().is_some_and(|t| (R_REGISTER..=R_SUCCESSORS).contains(t))
}

/// A fresh 6-character room code (no easily confused letters or digits).
pub fn new_room_code() -> String {
    let mut b = [0u8; 6];
    let _ = getrandom::fill(&mut b);
    b.iter().map(|v| CODE_ALPHABET[*v as usize % CODE_ALPHABET.len()] as char).collect()
}

/// Room codes are case-insensitive and may be typed with spaces or dashes.
pub fn normalize_code(code: &str) -> String {
    code.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).take(16).collect()
}

/// Whether `s` looks like a room code (as opposed to a `host:port` address).
pub fn looks_like_code(s: &str) -> bool {
    let n = normalize_code(s);
    (4..=16).contains(&n.len()) && !s.contains(':') && !s.contains('.')
}

pub fn random_u64() -> u64 {
    let mut b = [0u8; 8];
    let _ = getrandom::fill(&mut b);
    u64::from_le_bytes(b)
}

fn mac16(key: &[u8], parts: &[&[u8]]) -> [u8; 16] {
    let mut m = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        m.update(p);
    }
    let full = m.finalize().into_bytes();
    let mut out = [0u8; 16];
    out.copy_from_slice(&full[..16]);
    out
}

/// Constant-time comparison of two short tags.
pub fn tags_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A client's proof that it knows the session secret: HMAC over the relay's
/// cookie, the room code and its nonce, under the room's join key.
pub fn join_proof(join_key: &[u8; 32], cookie: &[u8; COOKIE_LEN], code: &str, nonce: u64) -> [u8; PROOF_LEN] {
    mac16(join_key, &[b"zestengine relay join v2", cookie, normalize_code(code).as_bytes(), &nonce.to_le_bytes()])
}

/// A host's proof that it knows the relay operator token.
pub fn token_proof(token: &str, cookie: &[u8; COOKIE_LEN], code: &str, nonce: u64) -> [u8; PROOF_LEN] {
    mac16(token.as_bytes(), &[b"zestengine relay create v2", cookie, normalize_code(code).as_bytes(), &nonce.to_le_bytes()])
}

/// The relay's stateless cookie for `src` (rotates with `bucket`).
pub fn cookie_for(secret: &[u8; 32], src: SocketAddr, code: &str, bucket: u64) -> [u8; COOKIE_LEN] {
    let mut addr = Vec::with_capacity(20);
    put_addr(&mut addr, src);
    mac16(secret, &[b"zestengine relay cookie v2", &addr, normalize_code(code).as_bytes(), &bucket.to_le_bytes()])
}

fn claim_message(code: &str, slot: u32, nonce: u64, claim_nonce: u64) -> Vec<u8> {
    let mut m = b"zestengine relay claim v2".to_vec();
    m.extend_from_slice(normalize_code(code).as_bytes());
    m.extend_from_slice(&slot.to_le_bytes());
    m.extend_from_slice(&nonce.to_le_bytes());
    m.extend_from_slice(&claim_nonce.to_le_bytes());
    m
}

/// A member's signature taking over the host slot of `code` (room owner key).
pub fn sign_claim(owner_seed: &[u8; 32], code: &str, slot: u32, nonce: u64, claim_nonce: u64) -> [u8; 64] {
    SigningKey::from_bytes(owner_seed).sign(&claim_message(code, slot, nonce, claim_nonce)).to_bytes()
}

pub fn verify_claim(owner: &[u8; 32], code: &str, slot: u32, nonce: u64, claim_nonce: u64, sig: &[u8; 64]) -> bool {
    let Ok(key) = VerifyingKey::from_bytes(owner) else {
        return false;
    };
    key.verify(&claim_message(code, slot, nonce, claim_nonce), &Signature::from_bytes(sig)).is_ok()
}

/// The public half of a room owner key.
pub fn owner_public(owner_seed: &[u8; 32]) -> [u8; 32] {
    SigningKey::from_bytes(owner_seed).verifying_key().to_bytes()
}

/// The relay's X25519 key pair from a random secret.
pub fn relay_public(secret: &[u8; 32]) -> [u8; 32] {
    MontgomeryPoint::mul_base_clamped(*secret).to_bytes()
}

fn join_key_cipher(shared: &[u8; 32], ephemeral: &[u8; 32], relay_key: &[u8; 32], cookie: &[u8; COOKIE_LEN]) -> ChaCha20Poly1305 {
    let mut info = b"zestengine-relay sealed join key v3".to_vec();
    info.extend_from_slice(ephemeral);
    info.extend_from_slice(relay_key);
    let mut key = [0u8; 32];
    Hkdf::<Sha256>::new(Some(cookie), shared).expand(&info, &mut key).expect("32 bytes is a valid HKDF output");
    ChaCha20Poly1305::new(&key.into())
}

/// Host: seal the room's join key to the relay's key (from its COOKIE), so
/// someone watching the host's link to the relay can't learn it. A fresh
/// ephemeral key per seal; the key is bound to the cookie.
pub fn seal_join_key(relay_key: &[u8; 32], cookie: &[u8; COOKIE_LEN], join_key: &[u8; 32]) -> [u8; SEALED_JOIN_KEY_LEN] {
    let mut secret = [0u8; 32];
    getrandom::fill(&mut secret).expect("system randomness");
    let ephemeral = MontgomeryPoint::mul_base_clamped(secret).to_bytes();
    let shared = MontgomeryPoint(*relay_key).mul_clamped(secret).to_bytes();
    let mut ct = *join_key;
    let tag = join_key_cipher(&shared, &ephemeral, relay_key, cookie)
        .encrypt_in_place_detached(&Nonce::default(), &[], &mut ct)
        .expect("in-memory AEAD");
    let mut out = [0u8; SEALED_JOIN_KEY_LEN];
    out[..32].copy_from_slice(&ephemeral);
    out[32..64].copy_from_slice(&ct);
    out[64..].copy_from_slice(&tag);
    out
}

/// Relay: open a host's sealed join key with the relay's secret.
pub fn open_join_key(relay_secret: &[u8; 32], cookie: &[u8; COOKIE_LEN], sealed: &[u8; SEALED_JOIN_KEY_LEN]) -> Option<[u8; 32]> {
    let ephemeral: [u8; 32] = sealed[..32].try_into().ok()?;
    let shared = MontgomeryPoint(ephemeral).mul_clamped(*relay_secret).to_bytes();
    let mut key: [u8; 32] = sealed[32..64].try_into().ok()?;
    let tag = Tag::from_slice(&sealed[64..]);
    join_key_cipher(&shared, &ephemeral, &relay_public(relay_secret), cookie)
        .decrypt_in_place_detached(&Nonce::default(), &[], &mut key, tag)
        .ok()?;
    Some(key)
}

/// What a REGISTER proves, by role.
#[derive(Debug, Clone, PartialEq)]
pub enum RegisterAuth {
    /// Nothing yet (asks for a cookie).
    None,
    /// Room creation: the owner key, the join key clients prove (sealed to
    /// the relay's key, see [`seal_join_key`]), the key-stretching parameters
    /// handed to joiners, and the operator token proof.
    Host { owner: [u8; 32], sealed_join_key: [u8; SEALED_JOIN_KEY_LEN], kdf: Vec<u8>, token_proof: [u8; PROOF_LEN] },
    /// A joiner's proof of the session secret.
    Client { proof: [u8; PROOF_LEN] },
}

/// A decoded relay packet.
#[derive(Debug, Clone, PartialEq)]
pub enum Packet<'a> {
    Register { protocol: u16, role: u8, code: String, nonce: u64, locals: Vec<SocketAddr>, cookie: [u8; COOKIE_LEN], auth: RegisterAuth },
    /// `relay_key`: the relay's X25519 public key, which a host seals the
    /// room's join key to.
    Cookie { cookie: [u8; COOKIE_LEN], kdf: Vec<u8>, relay_key: [u8; 32] },
    Registered { nonce: u64, slot: u32, token: u64, code: String, public: SocketAddr, claim_nonce: u64 },
    Peer { slot: u32, token: u64, public: SocketAddr, locals: Vec<SocketAddr> },
    Data { slot: u32, payload: &'a [u8] },
    Punch { token: u64, slot: u32 },
    PunchAck { token: u64, slot: u32 },
    Claim { nonce: u64, claim_nonce: u64, sig: [u8; 64] },
    Keepalive { nonce: u64 },
    Successors { nonce: u64, slots: Vec<u32> },
    Error { reason: String },
    Leave { nonce: u64 },
}

impl Packet<'_> {
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64);
        match self {
            Packet::Register { protocol, role, code, nonce, locals, cookie, auth } => {
                b.push(R_REGISTER);
                b.extend_from_slice(&protocol.to_le_bytes());
                b.push(*role);
                put_str(&mut b, code);
                b.extend_from_slice(&nonce.to_le_bytes());
                put_addrs(&mut b, locals);
                b.extend_from_slice(cookie);
                match auth {
                    RegisterAuth::None => b.push(0),
                    RegisterAuth::Host { owner, sealed_join_key, kdf, token_proof } => {
                        b.push(1);
                        b.extend_from_slice(owner);
                        b.extend_from_slice(sealed_join_key);
                        let n = kdf.len().min(MAX_KDF);
                        b.push(n as u8);
                        b.extend_from_slice(&kdf[..n]);
                        b.extend_from_slice(token_proof);
                    }
                    RegisterAuth::Client { proof } => {
                        b.push(2);
                        b.extend_from_slice(proof);
                    }
                }
                if b.len() < MIN_REGISTER {
                    b.resize(MIN_REGISTER, 0);
                }
            }
            Packet::Cookie { cookie, kdf, relay_key } => {
                b.push(R_COOKIE);
                b.extend_from_slice(cookie);
                let n = kdf.len().min(MAX_KDF);
                b.push(n as u8);
                b.extend_from_slice(&kdf[..n]);
                b.extend_from_slice(relay_key);
            }
            Packet::Registered { nonce, slot, token, code, public, claim_nonce } => {
                b.push(R_REGISTERED);
                b.extend_from_slice(&nonce.to_le_bytes());
                b.extend_from_slice(&slot.to_le_bytes());
                b.extend_from_slice(&token.to_le_bytes());
                put_str(&mut b, code);
                put_addr(&mut b, *public);
                b.extend_from_slice(&claim_nonce.to_le_bytes());
            }
            Packet::Peer { slot, token, public, locals } => {
                b.push(R_PEER);
                b.extend_from_slice(&slot.to_le_bytes());
                b.extend_from_slice(&token.to_le_bytes());
                put_addr(&mut b, *public);
                put_addrs(&mut b, locals);
            }
            Packet::Data { slot, payload } => return data_packet(*slot, payload),
            Packet::Punch { token, slot } | Packet::PunchAck { token, slot } => {
                b.push(if matches!(self, Packet::Punch { .. }) { R_PUNCH } else { R_PUNCH_ACK });
                b.extend_from_slice(&token.to_le_bytes());
                b.extend_from_slice(&slot.to_le_bytes());
            }
            Packet::Claim { nonce, claim_nonce, sig } => {
                b.push(R_CLAIM);
                b.extend_from_slice(&nonce.to_le_bytes());
                b.extend_from_slice(&claim_nonce.to_le_bytes());
                b.extend_from_slice(sig);
            }
            Packet::Successors { nonce, slots } => {
                b.push(R_SUCCESSORS);
                b.extend_from_slice(&nonce.to_le_bytes());
                let slots = &slots[..slots.len().min(MAX_SUCCESSORS)];
                b.push(slots.len() as u8);
                for s in slots {
                    b.extend_from_slice(&s.to_le_bytes());
                }
            }
            Packet::Keepalive { nonce } | Packet::Leave { nonce } => {
                b.push(if matches!(self, Packet::Keepalive { .. }) { R_KEEPALIVE } else { R_LEAVE });
                b.extend_from_slice(&nonce.to_le_bytes());
            }
            Packet::Error { reason } => {
                b.push(R_ERROR);
                put_str(&mut b, reason);
            }
        }
        b
    }

    pub fn decode(data: &[u8]) -> Option<Packet<'_>> {
        let mut r = Reader { d: data, p: 1 };
        let p = match *data.first()? {
            R_REGISTER => {
                let protocol = r.u16()?;
                if protocol != RELAY_PROTOCOL {
                    // Other versions: enough to answer with a clear error.
                    return Some(Packet::Register {
                        protocol,
                        role: 0,
                        code: String::new(),
                        nonce: 0,
                        locals: Vec::new(),
                        cookie: [0; COOKIE_LEN],
                        auth: RegisterAuth::None,
                    });
                }
                let role = r.u8()?;
                let code = r.string()?;
                let nonce = r.u64()?;
                let locals = r.addrs()?;
                let cookie = r.arr::<COOKIE_LEN>()?;
                let auth = match r.u8()? {
                    0 => RegisterAuth::None,
                    1 => {
                        let owner = r.arr::<32>()?;
                        let sealed_join_key = r.arr::<SEALED_JOIN_KEY_LEN>()?;
                        let n = (r.u8()? as usize).min(MAX_KDF);
                        let kdf = r.take(n)?.to_vec();
                        RegisterAuth::Host { owner, sealed_join_key, kdf, token_proof: r.arr()? }
                    }
                    2 => RegisterAuth::Client { proof: r.arr()? },
                    _ => return None,
                };
                Packet::Register { protocol, role, code, nonce, locals, cookie, auth }
            }
            R_COOKIE => {
                let cookie = r.arr()?;
                let n = (r.u8()? as usize).min(MAX_KDF);
                let kdf = r.take(n)?.to_vec();
                Packet::Cookie { cookie, kdf, relay_key: r.arr()? }
            }
            R_REGISTERED => Packet::Registered {
                nonce: r.u64()?,
                slot: r.u32()?,
                token: r.u64()?,
                code: r.string()?,
                public: r.addr()?,
                claim_nonce: r.u64()?,
            },
            R_PEER => Packet::Peer { slot: r.u32()?, token: r.u64()?, public: r.addr()?, locals: r.addrs()? },
            R_DATA => Packet::Data { slot: r.u32()?, payload: data.get(DATA_HEADER..)? },
            R_PUNCH => Packet::Punch { token: r.u64()?, slot: r.u32()? },
            R_PUNCH_ACK => Packet::PunchAck { token: r.u64()?, slot: r.u32()? },
            R_CLAIM => Packet::Claim { nonce: r.u64()?, claim_nonce: r.u64()?, sig: r.arr()? },
            R_KEEPALIVE => Packet::Keepalive { nonce: r.u64()? },
            R_LEAVE => Packet::Leave { nonce: r.u64()? },
            R_SUCCESSORS => {
                let nonce = r.u64()?;
                let n = r.u8()? as usize;
                Packet::Successors { nonce, slots: (0..n).map(|_| r.u32()).collect::<Option<_>>()? }
            }
            R_ERROR => Packet::Error { reason: r.string()? },
            _ => return None,
        };
        Some(p)
    }
}

/// A DATA packet without building a [`Packet`] (the hot path).
pub fn data_packet(slot: u32, payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(DATA_HEADER + payload.len());
    b.push(R_DATA);
    b.extend_from_slice(&slot.to_le_bytes());
    b.extend_from_slice(payload);
    b
}

/// An ERROR no longer than `limit` bytes (the reason is cut to fit).
pub fn error_within(reason: &str, limit: usize) -> Option<Vec<u8>> {
    let room = limit.checked_sub(2)?;
    let mut end = reason.len().min(room).min(255);
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    Some(Packet::Error { reason: reason[..end].to_owned() }.encode())
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    let mut n = s.len().min(255);
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    b.push(n as u8);
    b.extend_from_slice(&s.as_bytes()[..n]);
}

pub fn put_addr(b: &mut Vec<u8>, a: SocketAddr) {
    match a.ip() {
        IpAddr::V4(ip) => {
            b.push(4);
            b.extend_from_slice(&ip.octets());
        }
        IpAddr::V6(ip) => {
            b.push(6);
            b.extend_from_slice(&ip.octets());
        }
    }
    b.extend_from_slice(&a.port().to_le_bytes());
}

fn put_addrs(b: &mut Vec<u8>, list: &[SocketAddr]) {
    let n = list.len().min(MAX_LOCALS);
    b.push(n as u8);
    for a in &list[..n] {
        put_addr(b, *a);
    }
}

/// Decode one address written by [`put_addr`] at `*at`, advancing it.
pub fn read_addr(d: &[u8], at: &mut usize) -> Option<SocketAddr> {
    let mut r = Reader { d, p: *at };
    let a = r.addr()?;
    *at = r.p;
    Some(a)
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.d.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }
    fn arr<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.arr()?))
    }
    fn string(&mut self) -> Option<String> {
        let n = self.u8()? as usize;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn addr(&mut self) -> Option<SocketAddr> {
        let ip = match self.u8()? {
            4 => IpAddr::V4(Ipv4Addr::from(self.arr::<4>()?)),
            6 => IpAddr::V6(Ipv6Addr::from(self.arr::<16>()?)),
            _ => return None,
        };
        Some(SocketAddr::new(ip, self.u16()?))
    }
    fn addrs(&mut self) -> Option<Vec<SocketAddr>> {
        let n = (self.u8()? as usize).min(MAX_LOCALS);
        (0..n).map(|_| self.addr()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packets_round_trip() {
        let a: SocketAddr = "203.0.113.7:40000".parse().unwrap();
        let b: SocketAddr = "[2001:db8::1]:9".parse().unwrap();
        let payload = [1u8, 2, 3];
        let all = [
            Packet::Register {
                protocol: RELAY_PROTOCOL,
                role: ROLE_CLIENT,
                code: "ABC234".into(),
                nonce: 7,
                locals: vec![a, b],
                cookie: [3; 16],
                auth: RegisterAuth::Client { proof: [9; 16] },
            },
            Packet::Register {
                protocol: RELAY_PROTOCOL,
                role: ROLE_HOST,
                code: String::new(),
                nonce: 8,
                locals: vec![],
                cookie: [0; 16],
                auth: RegisterAuth::Host { owner: [1; 32], sealed_join_key: [2; SEALED_JOIN_KEY_LEN], kdf: vec![5, 6, 7], token_proof: [4; 16] },
            },
            Packet::Cookie { cookie: [8; 16], kdf: vec![1, 2], relay_key: [3; 32] },
            Packet::Registered { nonce: 7, slot: 3, token: 11, code: "ABC234".into(), public: b, claim_nonce: 99 },
            Packet::Peer { slot: 0, token: u64::MAX, public: a, locals: vec![] },
            Packet::Data { slot: 9, payload: &payload },
            Packet::Punch { token: 5, slot: 1 },
            Packet::PunchAck { token: 6, slot: 0 },
            Packet::Claim { nonce: 1, claim_nonce: 2, sig: [7; 64] },
            Packet::Keepalive { nonce: 1 },
            Packet::Leave { nonce: 2 },
            Packet::Successors { nonce: 3, slots: vec![4, 2, 9] },
            Packet::Error { reason: "nope".into() },
        ];
        for p in all {
            let bytes = p.encode();
            assert!(is_relay_packet(&bytes));
            if matches!(p, Packet::Register { .. }) {
                assert!(bytes.len() >= MIN_REGISTER, "registers are padded");
            }
            assert_eq!(Packet::decode(&bytes), Some(p));
        }
        assert_eq!(Packet::decode(&[R_PEER, 1, 2]), None, "short packets are rejected");
        assert!(!is_relay_packet(&[0x20]));
        assert!(error_within("relay protocol mismatch", 10).unwrap().len() <= 10);
    }

    #[test]
    fn join_keys_are_sealed_to_the_relay() {
        let relay_secret = [3u8; 32];
        let cookie = [1u8; 16];
        let sealed = seal_join_key(&relay_public(&relay_secret), &cookie, &[9; 32]);
        assert!(!sealed.windows(32).any(|w| w == [9; 32]), "the join key isn't in the clear");
        assert_eq!(open_join_key(&relay_secret, &cookie, &sealed), Some([9; 32]));
        assert_eq!(open_join_key(&[4; 32], &cookie, &sealed), None, "another relay can't open it");
        assert_eq!(open_join_key(&relay_secret, &[2; 16], &sealed), None, "bound to the cookie");
        assert_ne!(sealed, seal_join_key(&relay_public(&relay_secret), &cookie, &[9; 32]), "fresh ephemeral key per seal");
    }

    #[test]
    fn proofs_and_claims() {
        let key = [7u8; 32];
        let cookie = [1u8; 16];
        let p = join_proof(&key, &cookie, "abc234", 5);
        assert_eq!(p, join_proof(&key, &cookie, "ABC-234", 5), "codes are normalized");
        assert_ne!(p, join_proof(&key, &cookie, "ABC234", 6));
        assert_ne!(p, join_proof(&[8; 32], &cookie, "ABC234", 5));
        let seed = [3u8; 32];
        let owner = owner_public(&seed);
        let sig = sign_claim(&seed, "ABC234", 2, 10, 11);
        assert!(verify_claim(&owner, "ABC234", 2, 10, 11, &sig));
        assert!(!verify_claim(&owner, "ABC234", 2, 10, 12, &sig), "bound to the claim nonce");
        assert!(!verify_claim(&owner_public(&[4; 32]), "ABC234", 2, 10, 11, &sig));
    }

    #[test]
    fn room_codes() {
        let c = new_room_code();
        assert_eq!(c.len(), 6);
        assert_eq!(normalize_code(" ab-c2 34 "), "ABC234");
        assert!(looks_like_code("abc-234"));
        assert!(!looks_like_code("127.0.0.1:9000"));
        assert!(!looks_like_code("localhost:9000"));
    }
}
