//! Client side of internet play: relay registration, UDP hole punching and
//! the per-peer route (direct or relayed).
//!
//! A [`RelayClient`] rides on the session's UDP socket (see `UdpTransport`).
//! Peers reached through the relay get a *virtual* address (`240.x.y.z`,
//! encoding their relay slot) so the session code above keeps working with
//! plain `SocketAddr`s; the transport resolves a virtual address to the
//! punched direct endpoint when there is one, else wraps the datagram in a
//! relay DATA packet.
//!
//! Punch flow, once the relay introduced a peer (public endpoint + LAN
//! endpoints + punch token): both sides send PUNCH to every candidate
//! endpoint every [`PUNCH_INTERVAL`] for up to [`PUNCH_WINDOW`]. A valid
//! PUNCH is answered with PUNCH_ACK; the first ACK received picks that
//! endpoint as the direct route. No ACK in time, or a direct route going
//! silent for [`DIRECT_STALE`], falls back to the relay and retries after
//! [`PUNCH_RETRY`]. Relayed packets are always accepted, so switching paths
//! never drops a connection.
//!
//! Registration is authenticated (relay protocol 2): the first REGISTER only
//! fetches a cookie; a host then registers its room's owner key, join key
//! and key-stretching parameters ([`HostAuth`], once the session key is
//! stretched), and a client proves the session secret with an HMAC under
//! the join key (the session derives it once the relay handed over the
//! room's stretching parameters). After a host migration the elected
//! successor signs a CLAIM with the room owner key and takes the host slot,
//! so every member keeps reaching "the host" at the same virtual address.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};

use zestengine_relay::proto::{self, COOKIE_LEN, HOST_SLOT, Packet, RELAY_PROTOCOL, ROLE_CLIENT, ROLE_HOST, RegisterAuth};

/// Seconds between punch bursts while punching.
pub const PUNCH_INTERVAL: f64 = 0.15;
/// How long one punch attempt lasts before falling back to the relay.
pub const PUNCH_WINDOW: f64 = 4.0;
/// A direct route with no traffic for this long falls back to the relay.
pub const DIRECT_STALE: f64 = 4.0;
/// Delay before punching again after a failed or lost direct route.
pub const PUNCH_RETRY: f64 = 20.0;
const REGISTER_INTERVAL: f64 = 0.5;
const REGISTER_GIVE_UP: f64 = 10.0;
const KEEPALIVE: f64 = 5.0;
/// A host keepalives faster: the relay hands its slot to a successor after
/// a second of silence.
const HOST_KEEPALIVE: f64 = 0.25;

/// The virtual address standing for relay slot `slot`.
pub fn virtual_addr(slot: u32) -> SocketAddr {
    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(240, (slot >> 16) as u8, (slot >> 8) as u8, slot as u8), proto::DEFAULT_PORT))
}

/// The relay slot a virtual address stands for.
pub fn slot_of(addr: SocketAddr) -> Option<u32> {
    match addr.ip() {
        IpAddr::V4(ip) if ip.octets()[0] == 240 => {
            let o = ip.octets();
            Some(((o[1] as u32) << 16) | ((o[2] as u32) << 8) | o[3] as u32)
        }
        _ => None,
    }
}

/// One hole-punch attempt toward one peer (pure; driven with explicit time).
#[derive(Clone, Debug)]
pub struct Punch {
    active: bool,
    started: f64,
    last_sent: f64,
    next_try: f64,
    /// Confirmed direct endpoint.
    pub direct: Option<SocketAddr>,
    last_direct: f64,
}

impl Punch {
    /// Start punching now.
    pub fn new(now: f64) -> Self {
        Self { active: true, started: now, last_sent: f64::NEG_INFINITY, next_try: 0.0, direct: None, last_direct: now }
    }

    /// Whether to send a punch burst now; also expires attempts and stale routes.
    pub fn tick(&mut self, now: f64) -> bool {
        if self.direct.is_some() {
            if now - self.last_direct <= DIRECT_STALE {
                return false;
            }
            self.direct = None;
            self.active = false;
            self.next_try = now + PUNCH_RETRY;
        }
        if !self.active {
            if now < self.next_try {
                return false;
            }
            self.active = true;
            self.started = now;
            self.last_sent = f64::NEG_INFINITY;
        }
        if now - self.started > PUNCH_WINDOW {
            self.active = false;
            self.next_try = now + PUNCH_RETRY;
            return false;
        }
        if now - self.last_sent >= PUNCH_INTERVAL - 1e-9 {
            self.last_sent = now;
            return true;
        }
        false
    }

    /// The peer acknowledged a punch from `addr`: the direct path works.
    pub fn on_ack(&mut self, addr: SocketAddr, now: f64) {
        self.direct = Some(addr);
        self.last_direct = now;
        self.active = false;
    }

    /// Traffic arrived over the direct path.
    pub fn on_direct(&mut self, now: f64) {
        self.last_direct = now;
    }

    pub fn punching(&self) -> bool {
        self.active
    }
}

/// What a room host registers with the relay.
#[derive(Clone, Debug, PartialEq)]
pub struct HostAuth {
    /// The room owner key's seed (shared with members for host migration).
    pub owner_seed: [u8; 32],
    /// What joiners prove (derived from the session key).
    pub join_key: [u8; 32],
    /// Key-stretching parameters handed to joiners.
    pub kdf: Vec<u8>,
    /// Relay operator token (empty = none).
    pub token: String,
}

/// Which end of the room this client is.
#[derive(Clone, Debug, PartialEq)]
pub enum RelayRole {
    /// Open a room (empty code = let the relay pick one).
    Host { code: String },
    Client { code: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelayPhase {
    Registering,
    Registered,
    Failed(String),
}

/// One peer's path, for the UI.
#[derive(Clone, Debug)]
pub struct PeerPath {
    pub slot: u32,
    pub direct: Option<SocketAddr>,
    pub punching: bool,
}

/// Relay state for the UI.
#[derive(Clone, Debug)]
pub struct RelayStatus {
    pub relay: SocketAddr,
    pub phase: RelayPhase,
    /// The room code (assigned by the relay for hosts).
    pub code: String,
    /// This peer's endpoint as the relay sees it.
    pub public: Option<SocketAddr>,
    pub peers: Vec<PeerPath>,
    /// This peer's slot in the room (0 = the host).
    pub slot: Option<u32>,
    /// Hosting the room (after a claim, too).
    pub host: bool,
}

/// Where a datagram for a peer goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Route {
    Direct(SocketAddr),
    /// Through the relay, to this slot.
    Relay(u32),
}

struct Remote {
    token: u64,
    candidates: Vec<SocketAddr>,
    punch: Punch,
}

/// Relay registration and routing for one socket.
pub struct RelayClient {
    pub relay: SocketAddr,
    role: RelayRole,
    nonce: u64,
    phase: RelayPhase,
    code: String,
    slot: Option<u32>,
    public: Option<SocketAddr>,
    locals: Vec<SocketAddr>,
    /// This client's punch token (the relay hands it to the other side).
    my_token: Option<u64>,
    remotes: HashMap<u32, Remote>,
    by_addr: HashMap<SocketAddr, u32>,
    punch_enabled: bool,
    started: Option<f64>,
    last_register: f64,
    last_keepalive: f64,
    now: f64,
    cookie: Option<[u8; COOKIE_LEN]>,
    /// The relay's sealing key (from its COOKIE): a host's join key goes to
    /// the relay sealed to it, never in the clear.
    relay_key: [u8; 32],
    /// The room's stretching parameters, from the relay's cookie (clients).
    room_kdf: Option<Vec<u8>>,
    join_key: Option<[u8; 32]>,
    host_auth: Option<HostAuth>,
    claim_nonce: u64,
    /// Taking over the host slot with this room owner seed.
    claiming: Option<[u8; 32]>,
    /// Host: the ranked successor slots last handed to [`Self::set_successors`],
    /// re-sent with each keepalive (UDP may drop one).
    successors: Vec<u32>,
    successors_dirty: bool,
}

impl RelayClient {
    /// `local_port` is the socket's port (LAN candidates are built from it);
    /// `punch` off keeps every peer on the relay.
    pub fn new(relay: SocketAddr, role: RelayRole, local_port: u16, punch: bool) -> Self {
        let mut locals = Vec::new();
        if let Some(ip) = local_ip_towards(relay) {
            locals.push(SocketAddr::new(ip, local_port));
        }
        let code = match &role {
            RelayRole::Host { code } | RelayRole::Client { code } => proto::normalize_code(code),
        };
        Self {
            relay,
            role,
            nonce: proto::random_u64(),
            phase: RelayPhase::Registering,
            code,
            slot: None,
            public: None,
            locals,
            my_token: None,
            remotes: HashMap::new(),
            by_addr: HashMap::new(),
            punch_enabled: punch,
            started: None,
            last_register: f64::NEG_INFINITY,
            last_keepalive: 0.0,
            now: 0.0,
            cookie: None,
            relay_key: [0; 32],
            room_kdf: None,
            join_key: None,
            host_auth: None,
            claim_nonce: 0,
            claiming: None,
            successors: Vec::new(),
            successors_dirty: false,
        }
    }

    /// Host: the room credentials (registration waits for them).
    pub fn set_host_auth(&mut self, auth: HostAuth) {
        if self.host_auth.as_ref() != Some(&auth) {
            self.host_auth = Some(auth);
            self.started = None;
            self.last_register = f64::NEG_INFINITY;
        }
    }

    /// Client: the join key once the session stretched the secret.
    pub fn set_join_key(&mut self, key: [u8; 32]) {
        if self.join_key != Some(key) {
            self.join_key = Some(key);
            self.started = None;
            self.last_register = f64::NEG_INFINITY;
        }
    }

    /// Client: the relay handed over the room's stretching parameters and
    /// waits for a join proof.
    pub fn needs_join_key(&self) -> Option<Vec<u8>> {
        if self.is_host() || self.join_key.is_some() {
            return None;
        }
        self.room_kdf.clone().filter(|k| !k.is_empty())
    }

    /// Take over the room's host slot (host migration), proving membership
    /// with the room owner key.
    pub fn claim_host(&mut self, owner_seed: [u8; 32]) {
        if self.is_host() && self.slot == Some(HOST_SLOT) {
            return;
        }
        self.claiming = Some(owner_seed);
        self.last_register = f64::NEG_INFINITY;
    }

    /// Host: the session's migration order as relay slots, so the relay only
    /// lets the best-ranked live member claim the room.
    pub fn set_successors(&mut self, slots: Vec<u32>) {
        if slots != self.successors {
            self.successors = slots;
            self.successors_dirty = true;
        }
    }

    pub fn is_host(&self) -> bool {
        matches!(self.role, RelayRole::Host { .. })
    }

    pub fn status(&self) -> RelayStatus {
        let mut peers: Vec<PeerPath> = self
            .remotes
            .iter()
            .map(|(&slot, r)| PeerPath { slot, direct: r.punch.direct, punching: r.punch.punching() })
            .collect();
        peers.sort_by_key(|p| p.slot);
        RelayStatus {
            relay: self.relay,
            phase: self.phase.clone(),
            code: self.code.clone(),
            public: self.public,
            peers,
            slot: self.slot,
            host: self.is_host() && self.slot == Some(HOST_SLOT),
        }
    }

    /// The next REGISTER, or None while credentials are missing.
    fn register_packet(&self) -> Option<Vec<u8>> {
        let role = if self.is_host() { ROLE_HOST } else { ROLE_CLIENT };
        let cookie = self.cookie.unwrap_or_default();
        let auth = match (&self.cookie, self.is_host()) {
            (None, _) => RegisterAuth::None,
            (Some(c), true) => {
                let h = self.host_auth.as_ref()?;
                let token_proof = if h.token.is_empty() { [0; proto::PROOF_LEN] } else { proto::token_proof(&h.token, c, &self.code, self.nonce) };
                let sealed_join_key = proto::seal_join_key(&self.relay_key, c, &h.join_key);
                RegisterAuth::Host { owner: proto::owner_public(&h.owner_seed), sealed_join_key, kdf: h.kdf.clone(), token_proof }
            }
            (Some(c), false) => RegisterAuth::Client { proof: proto::join_proof(&self.join_key?, c, &self.code, self.nonce) },
        };
        if self.is_host() && self.host_auth.is_none() {
            return None;
        }
        Some(Packet::Register { protocol: RELAY_PROTOCOL, role, code: self.code.clone(), nonce: self.nonce, locals: self.locals.clone(), cookie, auth }.encode())
    }

    fn claim_packet(&self) -> Option<Vec<u8>> {
        let seed = self.claiming?;
        let slot = self.slot?;
        let sig = proto::sign_claim(&seed, &self.code, slot, self.nonce, self.claim_nonce);
        Some(Packet::Claim { nonce: self.nonce, claim_nonce: self.claim_nonce, sig }.encode())
    }

    /// Periodic work: (re)registration, keepalives and punch bursts.
    pub fn poll(&mut self, now: f64, out: &mut Vec<(Vec<u8>, SocketAddr)>) {
        self.now = now;
        let started = *self.started.get_or_insert(now);
        match self.phase {
            RelayPhase::Failed(_) => return,
            RelayPhase::Registering => {
                // Waiting for credentials (the session is stretching its key): no clock.
                let waiting = (self.is_host() && self.host_auth.is_none())
                    || (!self.is_host() && self.cookie.is_some() && self.join_key.is_none());
                if waiting {
                    self.started = Some(now);
                } else if now - started > REGISTER_GIVE_UP {
                    self.phase = RelayPhase::Failed("relay server unreachable".into());
                    return;
                }
                if now - self.last_register >= REGISTER_INTERVAL
                    && let Some(p) = self.register_packet()
                {
                    self.last_register = now;
                    out.push((p, self.relay));
                }
            }
            RelayPhase::Registered => {
                if self.claiming.is_some() {
                    if now - self.last_register >= REGISTER_INTERVAL
                        && let Some(p) = self.claim_packet()
                    {
                        self.last_register = now;
                        out.push((p, self.relay));
                    }
                } else {
                    // A client re-registers until the host's introduction arrived
                    // (it, and ours to the host, may have been lost).
                    let introduced = self.remotes.contains_key(&HOST_SLOT);
                    if !self.is_host()
                        && !introduced
                        && now - self.last_register >= REGISTER_INTERVAL
                        && let Some(p) = self.register_packet()
                    {
                        self.last_register = now;
                        out.push((p, self.relay));
                    }
                }
                let interval = if self.slot == Some(HOST_SLOT) { HOST_KEEPALIVE } else { KEEPALIVE };
                let keepalive = now - self.last_keepalive >= interval;
                if keepalive {
                    self.last_keepalive = now;
                    out.push((Packet::Keepalive { nonce: self.nonce }.encode(), self.relay));
                }
                if self.slot == Some(HOST_SLOT) && (self.successors_dirty || (keepalive && !self.successors.is_empty())) {
                    self.successors_dirty = false;
                    out.push((Packet::Successors { nonce: self.nonce, slots: self.successors.clone() }.encode(), self.relay));
                }
            }
        }
        let (Some(slot), Some(token)) = (self.slot, self.my_token) else {
            return;
        };
        if !self.punch_enabled {
            return;
        }
        for r in self.remotes.values_mut() {
            if r.punch.tick(now) {
                let p = Packet::Punch { token, slot }.encode();
                for c in &r.candidates {
                    out.push((p.clone(), *c));
                }
            }
        }
    }

    /// Handle a relay-protocol datagram (from the relay or a punching peer).
    /// Returns the payload of relayed game data, with the sender's virtual address.
    pub fn on_packet<'a>(&mut self, data: &'a [u8], src: SocketAddr, out: &mut Vec<(Vec<u8>, SocketAddr)>) -> Option<(&'a [u8], SocketAddr)> {
        let now = self.now;
        let from_relay = src == self.relay;
        match Packet::decode(data)? {
            Packet::Data { slot, payload } if from_relay => return Some((payload, virtual_addr(slot))),
            Packet::Cookie { cookie, kdf, relay_key } if from_relay => {
                let fresh = self.cookie != Some(cookie) || self.relay_key != relay_key;
                self.cookie = Some(cookie);
                self.relay_key = relay_key;
                if !kdf.is_empty() {
                    self.room_kdf = Some(kdf);
                }
                if fresh && let Some(p) = self.register_packet() {
                    self.last_register = now;
                    out.push((p, self.relay));
                }
            }
            Packet::Registered { nonce, slot, token, code, public, claim_nonce } if from_relay && nonce == self.nonce => {
                if self.phase != RelayPhase::Registered || self.slot != Some(slot) {
                    tracing::info!("net: relay {} registered slot {slot} in room {code} (public {public})", self.relay);
                }
                if slot == HOST_SLOT && self.claiming.take().is_some() {
                    // We took the room over: the old host's routes are gone.
                    if let Some(old) = self.remotes.remove(&HOST_SLOT) {
                        self.by_addr.retain(|_, s| *s != HOST_SLOT);
                        drop(old);
                    }
                    self.role = RelayRole::Host { code: code.clone() };
                }
                self.phase = RelayPhase::Registered;
                self.slot = Some(slot);
                self.my_token = Some(token);
                self.code = code;
                self.public = Some(public);
                self.claim_nonce = claim_nonce;
                self.last_keepalive = now;
            }
            Packet::Peer { slot, token, public, locals } if from_relay => {
                let mut candidates = vec![public];
                for l in locals {
                    if !candidates.contains(&l) {
                        candidates.push(l);
                    }
                }
                match self.remotes.get_mut(&slot) {
                    Some(r) if r.token == token => r.candidates = candidates,
                    _ => {
                        self.remotes.insert(slot, Remote { token, candidates, punch: Punch::new(now) });
                    }
                }
            }
            Packet::Punch { token, slot } => {
                let r = self.remotes.get_mut(&slot)?;
                if r.token != token {
                    return None;
                }
                // This path reaches us; answer on it and accept its traffic.
                self.by_addr.insert(src, slot);
                if let (Some(me), Some(t)) = (self.slot, self.my_token) {
                    out.push((Packet::PunchAck { token: t, slot: me }.encode(), src));
                }
            }
            Packet::PunchAck { token, slot } => {
                let r = self.remotes.get_mut(&slot)?;
                if r.token != token {
                    return None;
                }
                if r.punch.direct.is_none() {
                    tracing::info!("net: direct path to relay slot {slot} via {src}");
                }
                r.punch.on_ack(src, now);
                self.by_addr.insert(src, slot);
            }
            Packet::Error { reason } if from_relay => {
                tracing::warn!("net: relay: {reason}");
                if self.phase == RelayPhase::Registering || reason.contains("room closed") {
                    self.phase = RelayPhase::Failed(reason);
                } else if reason.contains("successor") {
                    // Another member took the host slot: we are no longer in the room.
                    self.phase = RelayPhase::Failed(reason);
                }
            }
            _ => {}
        }
        None
    }

    /// A game datagram arrived directly from `src`: the virtual address of
    /// the punched peer it belongs to, if any.
    pub fn virtual_of(&mut self, src: SocketAddr) -> Option<SocketAddr> {
        let slot = *self.by_addr.get(&src)?;
        if let Some(r) = self.remotes.get_mut(&slot)
            && r.punch.direct == Some(src)
        {
            r.punch.on_direct(self.now);
        }
        Some(virtual_addr(slot))
    }

    /// How to reach `to` (virtual addresses resolve through the routes).
    pub fn route(&self, to: SocketAddr) -> Route {
        match slot_of(to) {
            Some(slot) => match self.remotes.get(&slot).and_then(|r| r.punch.direct) {
                Some(d) => Route::Direct(d),
                None => Route::Relay(slot),
            },
            None => Route::Direct(to),
        }
    }

    /// Route label for a peer address, for the UI.
    pub fn route_label(&self, addr: SocketAddr) -> &'static str {
        match self.route(addr) {
            Route::Direct(_) if slot_of(addr).is_some() => "direct",
            Route::Direct(_) => "LAN",
            Route::Relay(_) => "relayed",
        }
    }

    /// Tell the relay we are gone (best effort).
    pub fn leave_packet(&self) -> (Vec<u8>, SocketAddr) {
        (Packet::Leave { nonce: self.nonce }.encode(), self.relay)
    }
}

/// The local interface address used to reach `remote` (no packet is sent).
fn local_ip_towards(remote: SocketAddr) -> Option<IpAddr> {
    let bind: SocketAddr = if remote.is_ipv4() { "0.0.0.0:0".parse().ok()? } else { "[::]:0".parse().ok()? };
    let s = UdpSocket::bind(bind).ok()?;
    s.connect(remote).ok()?;
    let ip = s.local_addr().ok()?.ip();
    (!ip.is_unspecified()).then_some(ip)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELAY: &str = "198.51.100.1:47800";

    fn registered_client() -> (RelayClient, Vec<(Vec<u8>, SocketAddr)>) {
        let relay: SocketAddr = RELAY.parse().unwrap();
        let mut c = RelayClient::new(relay, RelayRole::Client { code: "abc234".into() }, 5000, true);
        let mut out = Vec::new();
        c.poll(0.0, &mut out);
        assert!(matches!(Packet::decode(&out[0].0), Some(Packet::Register { code, auth: RegisterAuth::None, .. }) if code == "ABC234"));
        out.clear();
        // The relay's cookie carries the room's stretching parameters; the
        // session derives the join key, then the proof goes out.
        c.on_packet(&Packet::Cookie { cookie: [4; 16], kdf: vec![1, 2, 3], relay_key: [0; 32] }.encode(), relay, &mut out);
        assert!(out.is_empty(), "no proof without the key");
        assert_eq!(c.needs_join_key(), Some(vec![1, 2, 3]));
        c.set_join_key([9; 32]);
        c.poll(0.1, &mut out);
        let expect = proto::join_proof(&[9; 32], &[4; 16], "ABC234", c.nonce);
        assert!(matches!(Packet::decode(&out[0].0), Some(Packet::Register { auth: RegisterAuth::Client { proof }, .. }) if proof == expect));
        out.clear();
        let reg = Packet::Registered { nonce: c.nonce, slot: 3, token: 33, code: "ABC234".into(), public: "203.0.113.5:6000".parse().unwrap(), claim_nonce: 5 }.encode();
        assert!(c.on_packet(&reg, relay, &mut out).is_none());
        // The host's introduction: one public and one LAN candidate.
        let peer = Packet::Peer {
            slot: HOST_SLOT,
            token: 77,
            public: "203.0.113.9:9000".parse().unwrap(),
            locals: vec!["192.168.1.20:9000".parse().unwrap()],
        }
        .encode();
        c.on_packet(&peer, relay, &mut out);
        assert_eq!(c.status().phase, RelayPhase::Registered);
        (c, out)
    }

    #[test]
    fn punch_times_out_to_the_relay_then_retries() {
        let (mut c, mut out) = registered_client();
        let host = virtual_addr(HOST_SLOT);
        assert_eq!(c.route(host), Route::Relay(HOST_SLOT), "relayed until a punch succeeds");
        // Punch bursts go to both candidates every PUNCH_INTERVAL for PUNCH_WINDOW.
        let mut t = 0.0;
        let mut bursts = 0;
        while t <= PUNCH_WINDOW + 1.0 {
            out.clear();
            c.poll(t, &mut out);
            let punches: Vec<&SocketAddr> = out.iter().filter(|(b, _)| matches!(Packet::decode(b), Some(Packet::Punch { token: 33, slot: 3 }))).map(|(_, a)| a).collect();
            if !punches.is_empty() {
                assert_eq!(punches.len(), 2, "every candidate");
                bursts += 1;
            }
            t += 0.05;
        }
        let expected = (PUNCH_WINDOW / PUNCH_INTERVAL) as i32;
        assert!((bursts - expected).abs() <= 2, "{bursts} bursts, expected about {expected}");
        // Nobody answered: still relayed, and quiet until the retry.
        assert_eq!(c.route(host), Route::Relay(HOST_SLOT));
        out.clear();
        c.poll(t + 1.0, &mut out);
        assert!(!out.iter().any(|(b, _)| matches!(Packet::decode(b), Some(Packet::Punch { .. }))));
        out.clear();
        c.poll(t + PUNCH_RETRY + 0.1, &mut out);
        assert!(out.iter().any(|(b, _)| matches!(Packet::decode(b), Some(Packet::Punch { .. }))), "punching again after PUNCH_RETRY");
        // Relayed data arrives as the host's virtual address.
        let relay: SocketAddr = RELAY.parse().unwrap();
        let data = zestengine_relay::proto::data_packet(HOST_SLOT, b"game");
        assert_eq!(c.on_packet(&data, relay, &mut out), Some((&b"game"[..], host)));
        // ...but DATA from anyone else is ignored.
        assert_eq!(c.on_packet(&data, "203.0.113.66:1".parse().unwrap(), &mut out), None);
    }

    #[test]
    fn punch_ack_goes_direct_and_stale_paths_fall_back() {
        let (mut c, mut out) = registered_client();
        let host = virtual_addr(HOST_SLOT);
        let lan: SocketAddr = "192.168.1.20:9000".parse().unwrap();
        c.poll(0.1, &mut out);
        // A forged ACK (wrong token) is ignored.
        c.on_packet(&Packet::PunchAck { token: 1, slot: HOST_SLOT }.encode(), lan, &mut out);
        assert_eq!(c.route(host), Route::Relay(HOST_SLOT));
        // The host's punch reaches us on the LAN path: we answer there.
        out.clear();
        c.on_packet(&Packet::Punch { token: 77, slot: HOST_SLOT }.encode(), lan, &mut out);
        assert!(matches!(out.as_slice(), [(b, a)] if *a == lan && matches!(Packet::decode(b), Some(Packet::PunchAck { token: 33, slot: 3 }))));
        // Its ACK confirms the path: direct from now on.
        c.on_packet(&Packet::PunchAck { token: 77, slot: HOST_SLOT }.encode(), lan, &mut out);
        assert_eq!(c.route(host), Route::Direct(lan));
        assert_eq!(c.route_label(host), "direct");
        assert_eq!(c.virtual_of(lan), Some(host), "direct traffic maps back to the peer");
        // Traffic keeps it alive; silence past DIRECT_STALE falls back to the relay.
        let mut t = 0.1;
        for _ in 0..20 {
            t += 0.5;
            c.poll(t, &mut out);
            c.virtual_of(lan);
        }
        assert_eq!(c.route(host), Route::Direct(lan));
        c.poll(t + DIRECT_STALE + 0.5, &mut out);
        assert_eq!(c.route(host), Route::Relay(HOST_SLOT), "stale direct path dropped");
    }

    #[test]
    fn virtual_addresses_round_trip() {
        for slot in [0u32, 1, 255, 256, 70_000, 0xFF_FFFF] {
            assert_eq!(slot_of(virtual_addr(slot)), Some(slot));
        }
        assert_eq!(slot_of("10.0.0.1:9000".parse().unwrap()), None);
    }
}
