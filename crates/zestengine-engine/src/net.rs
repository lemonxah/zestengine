//! Built-in networking (2G): transport, connections and the replicated
//! session registry.
//!
//! Topology: host-authoritative star. One peer hosts (peer id 1, either a
//! player-host or a dedicated server); clients join it and every packet goes
//! through the host, which validates writes (only an object's owner may move
//! it, write its variables or despawn it) and relays them. One code path
//! serves client-server games and "P2P-feel" social worlds on a LAN or over a
//! forwarded port.
//!
//! Transport: UDP (or an in-memory hub for tests) with two channels per
//! connection. Unreliable packets carry snapshots, avatar poses, animator
//! parameters, voice, predicted-pawn inputs / states and unreliable RPCs. A
//! reliable ordered channel (sequence numbers, batched acks, RTT-based
//! resend, fragmentation of large messages) carries everything that must
//! arrive: joins/leaves, ownership, spawns, variables, RPCs, chat, scene
//! changes and late-join catch-up.
//!
//! Handshake (protocol 7): HELLO (padded, client nonce) -> CHALLENGE (no
//! larger than the HELLO: key-stretching parameters, the host's identity,
//! epoch and peer id, a stateless cookie; the host keeps nothing) -> JOIN
//! (the cookie back, plus name, player id and a signature over the
//! transcript, sealed under the connection key: HKDF of the stretched secret
//! and the transcript) -> WELCOME (sealed, signed by the host). A wrong
//! secret gets a REJECT signed with the host's identity; other refusals are
//! sealed; unauthenticated rejects are only believed when no CHALLENGE turns
//! up. After that every packet is sealed with ChaCha20-Poly1305 and
//! replay-checked (`net_secure`), so stray or forged datagrams are dropped.
//! Connections ping for RTT, jitter and loss (both directions) and time out
//! after `NetConfig::timeout` of silence. Half-open joins are capped.
//!
//! Host migration: the host replicates a succession plan (members in join
//! order, their endpoints, the relay room and its owner key, bans). When the
//! host times out or hands off, every member walks the same list: the first
//! one takes over with the world it has (peer ids, ownership and spawns
//! stay), the others reconnect to it (directly, then through the relay),
//! resend what the old host never acknowledged, and resync. A promoted host
//! that gave up on a better-ranked one keeps probing it and steps down if it
//! turns out to be hosting too.
//!
//! Internet play: with a relay server configured (`NetConfig::relay`) the
//! host also opens a room there and clients join by room code; peers punch
//! a direct UDP path when they can and stay relayed when they can't
//! (`net_relay`, `zestengine-relay`). LAN clients keep connecting directly to the
//! same socket.
//!
//! Scene-level replication (which objects to send, interpolation, physics
//! hand-over) lives in `net_sync`, prediction in `net_predict`; this module
//! only moves state around.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use zestengine_core::{
    HOST_PEER, HitHistory, InterpolationMode, NetEvent, NetQuality, NetRpc, NetSpawnSource, NetValue, NetView,
    NetValidation, NetWorldOp, NetworkManager, ObjectId, OwnershipMode, PeerInfo, RpcTarget, TRACKER_SLOTS, Topology,
};
use zestengine_relay::proto::{self as relay_proto, HOST_SLOT};
use glam::{Quat, Vec3};

pub use crate::net_identity::{Identity, fingerprint, set_identity_dir, short_fingerprint};
pub use crate::net_relay::{HostAuth, RelayPhase, RelayRole, RelayStatus, virtual_addr};
pub use crate::net_secure::{KdfMode, KdfParams, generate_secret};
use crate::net_predict::{AdaptiveDelay, AuthState, InputCmd, InputQueue, JitterMeter, MAX_CMDS_PER_PACKET, Predictor};
use crate::net_relay::{RelayClient, Route};
use zestengine_core::{AntiCheatConfig, AntiCheatProvider, ClientStatus, Verdict};
use crate::net_validate::{REPORT_RPC, Claim, ClaimCtx, DenyWhy, Escalation, Kind, Limit, Reason, RpcVerdict, SpawnReq, Stamp, Validator};
use crate::net_secure::{Bucket, KdfJob, SEAL_OVERHEAD, SecureChannel, conn_key, hash, mac16, random_bytes, relay_join_key};

/// Bumped whenever the wire format changes; mismatched peers are rejected
/// with a message naming both versions.
/// Wire protocol. 7: upstream 32-bit ids (`P_SNAPB` from clients) and delta-baselined snapshots; an
/// older client is refused with a version error instead of silently receiving no snapshots.
pub const PROTOCOL: u16 = 7;
/// UDP port LAN discovery queries go to.
pub const DISCOVERY_PORT: u16 = 47777;
/// Largest game packet we build. Sealing adds [`SEAL_OVERHEAD`] + 1 and a
/// relay hop 5 more, so datagrams on the wire stay under [`MAX_DATAGRAM`]
/// (below the 1280-byte IPv6 minimum MTU; no IP fragmentation).
pub const MTU: usize = 1200;
pub const MAX_DATAGRAM: usize = MTU + 1 + SEAL_OVERHEAD + relay_proto::DATA_HEADER;
const FRAG: usize = MTU - 16;
const RECV_BUF: usize = 2048;
const PING_INTERVAL: f64 = 0.5;
const HELLO_INTERVAL: f64 = 0.25;
/// HELLOs are padded to this, so the CHALLENGE (and any pre-auth reply) is
/// never larger than what an unauthenticated sender sent.
pub const HELLO_SIZE: usize = 128;
/// How long an unauthenticated REJECT may stand before the client believes it.
const REJECT_GRACE: f64 = 2.0;
/// A JOIN unanswered this long moves on to the next spare CHALLENGE.
const JOIN_PATIENCE: f64 = 1.0;
/// Most other CHALLENGEs a client keeps while its JOIN is pending.
const MAX_SPARE_CHALLENGES: usize = 4;
/// Handshake cookies rotate this often (the previous one is accepted too).
const COOKIE_PERIOD: f64 = 30.0;
/// Accepted joins that haven't confirmed their key yet.
pub const MAX_HALF_OPEN: usize = 32;
/// Seconds a half-open join may stay unconfirmed.
const HALF_OPEN_TTL: f64 = 5.0;
/// Seconds between tie-break probes of better-ranked successors.
const PROBE_INTERVAL: f64 = 1.0;
const MAX_QUEUE: usize = 4096;
/// Pending world writes (variables, parents) wait this long for their object.
const PENDING_TTL: f64 = 10.0;
/// Snapshots kept per object.
const SNAP_KEEP: usize = 32;

// Packet tags.
const P_HELLO: u8 = 1;
const P_WELCOME: u8 = 2;
const P_REJECT: u8 = 3;
const P_PING: u8 = 4;
const P_PONG: u8 = 5;
const P_BYE: u8 = 6;
const P_REL: u8 = 7;
const P_ACK: u8 = 8;
const P_SNAP: u8 = 9;
const P_AVATAR: u8 = 10;
const P_ANIM: u8 = 11;
const P_VOICE: u8 = 12;
const P_URPC: u8 = 13;
const P_DISCOVER: u8 = 14;
const P_INFO: u8 = 15;
const P_FACE: u8 = 16;
const P_CHALLENGE: u8 = 17;
const P_JOIN: u8 = 18;
const P_INPUT: u8 = 19;
const P_PSTATE: u8 = 20;
const P_PROBE_ACK: u8 = 21;
/// Bit-packed avatar parameters (schema hash + payload, no names).
const P_AVPARAM: u8 = 22;
/// Host -> client: batched snapshots of many senders (interest management).
const P_SNAPB: u8 = 23;
/// Every packet after the handshake: `[P_SECURE][counter][sealed packet][tag]`.
const P_SECURE: u8 = 0x20;
/// JOIN header: tag + client nonce + cookie + mode (authenticated, clear).
const JOIN_HEADER: usize = 34;
const JOIN_MODE_JOIN: u8 = 0;
const JOIN_MODE_PROBE: u8 = 1;
// REJECT kinds (clear packets).
const REJECT_HINT: u8 = 0;
const REJECT_SIGNED: u8 = 1;
// BYE reasons (sealed).
const BYE_LEAVE: u8 = 0;
const BYE_HANDOFF: u8 = 1;
const BYE_CLOSED: u8 = 2;
const BYE_KICKED: u8 = 3;

// Reliable message tags.
const M_PEERS: u8 = 1;
const M_JOINED: u8 = 2;
const M_LEFT: u8 = 3;
const M_OWNER: u8 = 4;
const M_CLAIM: u8 = 5;
const M_RELEASE: u8 = 6;
const M_TRANSFER: u8 = 7;
const M_OWNER_REQ: u8 = 8;
const M_DENY: u8 = 9;
const M_SPAWN: u8 = 10;
const M_DESPAWN: u8 = 11;
const M_VAR: u8 = 12;
const M_ACTIVE: u8 = 13;
const M_PARENT: u8 = 14;
const M_RPC: u8 = 15;
const M_TEXT: u8 = 16;
const M_SCENE: u8 = 17;
const M_XFORM: u8 = 18;
const M_ANIM: u8 = 19;
const M_SUCCESSION: u8 = 20;
const M_RESYNC: u8 = 21;
const M_RESYNC_END: u8 = 22;
const M_VAR_SYNC: u8 = 23;
const M_ANCHOR: u8 = 24;
const M_AVPARAM: u8 = 25;
/// Host -> owner: a move was refused, put the object back.
const M_CORRECT: u8 = 26;
/// Host -> client: a 32-bit alias for a network id (interest management).
const M_NETID: u8 = 27;
/// Anti-cheat provider payload (either direction; SEC-2).
const M_ANTICHEAT: u8 = 28;

// ------------------------------------------------------------------ transport

/// A datagram transport. UDP in games; [`MemHub`] endpoints in tests.
pub trait Transport {
    fn send(&self, data: &[u8], to: SocketAddr);
    fn recv(&mut self, buf: &mut [u8]) -> Option<(usize, SocketAddr)>;
    fn local_addr(&self) -> Option<SocketAddr>;
    /// Periodic work (relay registration, hole punching, keepalives).
    fn poll(&mut self, _now: f64) {}
    /// Relay registration and routes, when going through a relay.
    fn relay_status(&self) -> Option<RelayStatus> {
        None
    }
    /// How datagrams reach `addr`: "direct", "relayed" or "LAN".
    fn route_label(&self, _addr: SocketAddr) -> &'static str {
        "LAN"
    }
    /// The session is closing (leave the relay room).
    fn close(&mut self) {}
    /// The punched direct endpoint of a relay peer (None for anything else).
    fn direct_route(&self, _addr: SocketAddr) -> Option<SocketAddr> {
        None
    }
    /// Host: the room credentials, once the session key is stretched.
    fn set_relay_host_auth(&mut self, _auth: HostAuth) {}
    /// Client: the relay handed over the room's stretching parameters and
    /// waits for the join proof.
    fn relay_needs_join_key(&self) -> Option<Vec<u8>> {
        None
    }
    fn set_relay_join_key(&mut self, _key: [u8; 32]) {}
    /// Host migration: take over the room's host slot.
    fn relay_claim_host(&mut self, _owner_seed: [u8; 32]) {}
    /// Host: the migration order as relay slots (see [`RelayClient::set_successors`]).
    fn relay_set_successors(&mut self, _slots: Vec<u32>) {}
    /// Start using a relay mid-session (migration fallback). False when
    /// this transport can't.
    fn attach_relay(&mut self, _relay: SocketAddr, _role: RelayRole, _punch: bool) -> bool {
        false
    }
}

/// UDP socket, optionally registered with a relay server: relay control and
/// relayed data, punched direct paths and plain LAN traffic share it.
pub struct UdpTransport {
    socket: UdpSocket,
    relay: Option<RelayClient>,
    out: Vec<(Vec<u8>, SocketAddr)>,
}

impl UdpTransport {
    /// The socket's bound address (hosts show this to joiners).
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.socket.local_addr().ok()
    }

    pub fn bind(port: u16) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(("0.0.0.0", port))?;
        socket.set_nonblocking(true)?;
        Ok(Self { socket, relay: None, out: Vec::new() })
    }

    /// Register with a relay too (a room host, or a client joining a room).
    /// `punch` off keeps every relay peer on the relayed path.
    pub fn with_relay(mut self, relay: SocketAddr, role: RelayRole, punch: bool) -> Self {
        let port = self.socket.local_addr().map(|a| a.port()).unwrap_or(0);
        self.relay = Some(RelayClient::new(relay, role, port, punch));
        self
    }

    fn flush(&mut self) {
        for (b, a) in self.out.drain(..) {
            let _ = self.socket.send_to(&b, a);
        }
    }
}

impl Transport for UdpTransport {
    fn send(&self, data: &[u8], to: SocketAddr) {
        match &self.relay {
            Some(r) => match r.route(to) {
                Route::Direct(a) => {
                    let _ = self.socket.send_to(data, a);
                }
                Route::Relay(slot) => {
                    let _ = self.socket.send_to(&relay_proto::data_packet(slot, data), r.relay);
                }
            },
            None => {
                let _ = self.socket.send_to(data, to);
            }
        }
    }
    fn recv(&mut self, buf: &mut [u8]) -> Option<(usize, SocketAddr)> {
        loop {
            let (n, src) = match self.socket.recv_from(buf) {
                Ok(v) => v,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return None,
                // ICMP port-unreachable surfaces as a reset on some platforms.
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => continue,
                Err(e) => {
                    tracing::debug!("net recv error: {e}");
                    return None;
                }
            };
            let Some(relay) = self.relay.as_mut() else {
                return Some((n, src));
            };
            if relay_proto::is_relay_packet(&buf[..n]) {
                let data = relay.on_packet(&buf[..n], src, &mut self.out).map(|(p, v)| (p.len(), v));
                self.flush();
                if let Some((len, virt)) = data {
                    buf.copy_within(relay_proto::DATA_HEADER..relay_proto::DATA_HEADER + len, 0);
                    return Some((len, virt));
                }
                continue;
            }
            return Some((n, relay.virtual_of(src).unwrap_or(src)));
        }
    }
    fn local_addr(&self) -> Option<SocketAddr> {
        self.socket.local_addr().ok()
    }
    fn poll(&mut self, now: f64) {
        if let Some(r) = self.relay.as_mut() {
            r.poll(now, &mut self.out);
        }
        self.flush();
    }
    fn relay_status(&self) -> Option<RelayStatus> {
        self.relay.as_ref().map(|r| r.status())
    }
    fn route_label(&self, addr: SocketAddr) -> &'static str {
        self.relay.as_ref().map_or("LAN", |r| r.route_label(addr))
    }
    fn close(&mut self) {
        if let Some(r) = self.relay.take() {
            let (b, a) = r.leave_packet();
            for _ in 0..2 {
                let _ = self.socket.send_to(&b, a);
            }
        }
    }
    fn direct_route(&self, addr: SocketAddr) -> Option<SocketAddr> {
        match self.relay.as_ref()?.route(addr) {
            Route::Direct(d) if crate::net_relay::slot_of(addr).is_some() => Some(d),
            _ => None,
        }
    }
    fn set_relay_host_auth(&mut self, auth: HostAuth) {
        if let Some(r) = self.relay.as_mut() {
            r.set_host_auth(auth);
        }
    }
    fn relay_needs_join_key(&self) -> Option<Vec<u8>> {
        self.relay.as_ref()?.needs_join_key()
    }
    fn set_relay_join_key(&mut self, key: [u8; 32]) {
        if let Some(r) = self.relay.as_mut() {
            r.set_join_key(key);
        }
    }
    fn relay_claim_host(&mut self, owner_seed: [u8; 32]) {
        if let Some(r) = self.relay.as_mut() {
            r.claim_host(owner_seed);
        }
    }
    fn relay_set_successors(&mut self, slots: Vec<u32>) {
        if let Some(r) = self.relay.as_mut() {
            r.set_successors(slots);
        }
    }
    fn attach_relay(&mut self, relay: SocketAddr, role: RelayRole, punch: bool) -> bool {
        let port = self.socket.local_addr().map(|a| a.port()).unwrap_or(0);
        self.relay = Some(RelayClient::new(relay, role, port, punch));
        true
    }
}

/// In-process packet hub: endpoints exchange datagrams through shared queues,
/// with optional random loss and severed links. Deterministic, so tests can
/// drive sessions with a manual clock.
#[derive(Clone)]
pub struct MemHub {
    inner: Arc<Mutex<HubInner>>,
}

impl Default for MemHub {
    fn default() -> Self {
        // Every hub gets its own address range, so the host identity pins
        // (trust on first use, keyed by address) of tests running side by side
        // in one process never meet.
        static NEXT_HUB: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = NEXT_HUB.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self { inner: Arc::new(Mutex::new(HubInner { base: n, ..Default::default() })) }
    }
}

#[derive(Default)]
struct HubInner {
    /// This hub's address range (the second and third octet).
    base: u32,
    /// (datagram, sender, delivery time)
    queues: HashMap<SocketAddr, VecDeque<(Vec<u8>, SocketAddr, f64)>>,
    next: u16,
    loss: f32,
    rng: u64,
    cut: HashSet<SocketAddr>,
    cut_links: HashSet<(SocketAddr, SocketAddr)>,
    time: f64,
    latency: f64,
    jitter: f64,
}

impl HubInner {
    fn rand(&mut self) -> f64 {
        if self.rng == 0 {
            self.rng = 0x9E37_79B9_7F4A_7C15;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % 10_000) as f64 / 10_000.0
    }
}

impl MemHub {
    pub fn new() -> Self {
        Self::default()
    }
    /// A new endpoint with its own address.
    pub fn endpoint(&self) -> MemTransport {
        let mut h = self.inner.lock().unwrap();
        h.next += 1;
        let addr: SocketAddr = format!("10.{}.{}.1:{}", (h.base >> 8) & 255, h.base & 255, 20000 + h.next).parse().unwrap();
        h.queues.insert(addr, VecDeque::new());
        MemTransport { addr, hub: self.clone() }
    }
    /// Drop this fraction (0..1) of packets.
    pub fn set_loss(&self, loss: f32) {
        let mut h = self.inner.lock().unwrap();
        h.loss = loss;
        h.rng = 0x9E37_79B9_7F4A_7C15;
    }
    /// Silently drop everything to and from `addr` (a vanished peer).
    pub fn cut(&self, addr: SocketAddr) {
        self.inner.lock().unwrap().cut.insert(addr);
    }
    /// Drop everything between `a` and `b` (both ways): a partition.
    pub fn cut_link(&self, a: SocketAddr, b: SocketAddr) {
        let mut h = self.inner.lock().unwrap();
        h.cut_links.insert((a, b));
        h.cut_links.insert((b, a));
    }
    /// Undo every cut.
    pub fn heal(&self) {
        let mut h = self.inner.lock().unwrap();
        h.cut.clear();
        h.cut_links.clear();
    }
    /// Deliver a datagram claiming to come from `from` (a spoofing attacker).
    pub fn inject(&self, from: SocketAddr, to: SocketAddr, data: &[u8]) {
        let mut h = self.inner.lock().unwrap();
        let at = h.time;
        if let Some(q) = h.queues.get_mut(&to) {
            q.push_back((data.to_vec(), from, at));
        }
    }
    /// Delay every datagram by `one_way` seconds plus up to `jitter` more
    /// (which reorders them). Time only moves with [`Self::advance`].
    pub fn set_latency(&self, one_way: f64, jitter: f64) {
        let mut h = self.inner.lock().unwrap();
        h.latency = one_way;
        h.jitter = jitter;
    }
    pub fn advance(&self, dt: f64) {
        self.inner.lock().unwrap().time += dt;
    }
}

pub struct MemTransport {
    addr: SocketAddr,
    hub: MemHub,
}

impl Transport for MemTransport {
    fn send(&self, data: &[u8], to: SocketAddr) {
        let mut h = self.hub.inner.lock().unwrap();
        if h.cut.contains(&self.addr) || h.cut.contains(&to) || h.cut_links.contains(&(self.addr, to)) {
            return;
        }
        if h.loss > 0.0 && (h.rand() as f32) < h.loss {
            return;
        }
        let at = h.time + h.latency + if h.jitter > 0.0 { h.rand() * h.jitter } else { 0.0 };
        if let Some(q) = h.queues.get_mut(&to) {
            q.push_back((data.to_vec(), self.addr, at));
        }
    }
    fn recv(&mut self, buf: &mut [u8]) -> Option<(usize, SocketAddr)> {
        let mut h = self.hub.inner.lock().unwrap();
        let now = h.time;
        let q = h.queues.get_mut(&self.addr)?;
        let i = q.iter().position(|(_, _, at)| *at <= now)?;
        let (data, from, _) = q.remove(i)?;
        let n = data.len().min(buf.len());
        buf[..n].copy_from_slice(&data[..n]);
        Some((n, from))
    }
    fn local_addr(&self) -> Option<SocketAddr> {
        Some(self.addr)
    }
}

mod interest;
use interest::Stream;
pub use interest::{Interest, InterestClientStats};

// ------------------------------------------------------------------ config / state

/// Session settings (from the scene's Network Manager when there is one).
#[derive(Clone, Debug)]
pub struct NetConfig {
    pub player_name: String,
    pub session_name: String,
    pub max_players: u32,
    pub timeout: f32,
    pub discovery: bool,
    pub discovery_port: u16,
    pub clients_can_spawn: bool,
    /// Join secret: keys the session encryption, and joiners must know it.
    /// Empty = an open session (still encrypted, but anyone may join).
    pub secret: String,
    /// Relay server (`host:port`, default port 47800) for internet play:
    /// hosting also opens a room there; room-code joins go through it.
    pub relay: String,
    /// Room code a host asks the relay for (empty = the relay picks one).
    pub room: String,
    /// Try UDP hole punching to relay peers (off = always relayed).
    pub punch: bool,
    /// Operator token for relays that require one to open rooms.
    pub relay_token: String,
    /// This install's player identity (signs the handshake).
    pub identity: Identity,
    /// Host: key-stretching parameters (None = the defaults for the secret).
    pub kdf: Option<KdfParams>,
    /// Keep the session going when the host leaves: the next in line takes over.
    pub migration: bool,
    /// This peer may be elected host after a migration.
    pub can_host: bool,
    /// Seconds a migrating client tries one successor over one path
    /// (direct, then through the relay).
    pub migrate_attempt: f32,
    /// Who runs the session. `Dedicated` turns host migration off, makes
    /// clients pin the server's identity key and applies the strict
    /// validation rules to every client.
    pub topology: Topology,
    /// Server-side validation limits (see `net_validate`).
    pub validation: NetValidation,
    /// Host: interest management for large worlds (off by default).
    pub interest: zestengine_core::InterestConfig,
    /// Host: persist bans made by the strike system (and `ban`) in this RON
    /// file, and refuse the player ids listed in it. `None` = in memory only.
    pub ban_file: Option<std::path::PathBuf>,
    /// How many of the next hosts' addresses a listen host shares with its
    /// members for host migration (0 = none: members reach a new host through
    /// the relay only). Relay sessions and dedicated servers share none.
    pub migration_addr_depth: u32,
    /// Host: only these player fingerprints may join (empty = anyone with the secret).
    pub allow: Vec<String>,
    /// Host: player fingerprints refused at the door.
    pub ban: Vec<String>,
    /// Client: remember the host's identity per address / room code (trust
    /// on first use) and refuse a different one later.
    pub pin_host: bool,
    /// Optional account name to log in as (empty = no account login).
    pub account_name: String,
    /// Password for `account_name` (sealed in JOIN).
    pub account_password: String,
    /// Host account list. Empty keeps identity+secret-only auth.
    pub accounts: Vec<NetAccount>,
    /// Anti-cheat provider and requirement (SEC-2). Default: off.
    pub anticheat: AntiCheatConfig,
}

/// One host-side login accepted by [`NetConfig::accounts`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetAccount {
    pub name: String,
    pub password: String,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            player_name: std::env::var("ZESTENGINE_NAME").unwrap_or_else(|_| "Player".to_string()),
            session_name: "ZestEngine Session".to_string(),
            max_players: 16,
            timeout: 10.0,
            discovery: true,
            discovery_port: DISCOVERY_PORT,
            clients_can_spawn: true,
            secret: std::env::var("ZESTENGINE_SECRET").unwrap_or_default(),
            relay: std::env::var("ZESTENGINE_RELAY").unwrap_or_default(),
            room: String::new(),
            punch: std::env::var("ZESTENGINE_NO_PUNCH").is_err(),
            relay_token: std::env::var("ZESTENGINE_RELAY_TOKEN").unwrap_or_default(),
            identity: Identity::install(),
            kdf: None,
            topology: Topology::ListenHost,
            validation: NetValidation::default(),
            interest: zestengine_core::InterestConfig::default(),
            ban_file: None,
            migration_addr_depth: 2,
            migration: true,
            can_host: true,
            migrate_attempt: 3.0,
            allow: Vec::new(),
            ban: Vec::new(),
            pin_host: false,
            account_name: std::env::var("ZESTENGINE_ACCOUNT").unwrap_or_default(),
            account_password: std::env::var("ZESTENGINE_PASSWORD").unwrap_or_default(),
            accounts: parse_accounts(&std::env::var("ZESTENGINE_ACCOUNTS").unwrap_or_default()),
            anticheat: AntiCheatConfig::default(),
        }
    }
}

impl NetConfig {
    pub fn from_manager(m: &NetworkManager) -> Self {
        Self {
            session_name: m.session_name.clone(),
            max_players: m.max_players.max(1),
            timeout: m.timeout.max(1.0),
            discovery: m.discovery,
            clients_can_spawn: m.clients_can_spawn,
            ..Self::default().with_manager_net(m)
        }
    }

    fn with_manager_net(mut self, m: &NetworkManager) -> Self {
        if !m.secret.is_empty() {
            self.secret = m.secret.clone();
        }
        if !m.relay.is_empty() {
            self.relay = m.relay.clone();
        }
        self.room = m.room_code.clone();
        self.migration = m.host_migration;
        self.pin_host = m.pin_host;
        self.topology = if m.is_dedicated() { Topology::Dedicated } else { Topology::ListenHost };
        self.validation = m.validation.clone();
        self.interest = m.interest.clone();
        self.anticheat = m.anticheat.clone();
        self.allow = m.allow_list.clone();
        self.ban = m.ban_list.clone();
        if !m.account_list.is_empty() {
            self.accounts = parse_accounts(&m.account_list.join("\n"));
        }
        self
    }
}

/// Where a join goes: a direct address, or a room on a relay.
#[derive(Clone, Debug, PartialEq)]
pub enum JoinTarget {
    Direct(SocketAddr),
    Room { code: String, relay: SocketAddr },
}

impl JoinTarget {
    /// `host:port` joins directly; `CODE` joins a room on `default_relay`;
    /// `CODE@relay:port` names the relay.
    pub fn parse(input: &str, default_relay: &str) -> std::io::Result<Self> {
        let input = input.trim();
        if let Some((code, relay)) = input.split_once('@') {
            return Ok(Self::Room { code: relay_proto::normalize_code(code), relay: parse_relay(relay)? });
        }
        if relay_proto::looks_like_code(input) {
            if default_relay.trim().is_empty() {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "joining by room code needs a relay server"));
            }
            return Ok(Self::Room { code: relay_proto::normalize_code(input), relay: parse_relay(default_relay)? });
        }
        Ok(Self::Direct(parse_addr(input)?))
    }
}

/// A relay address; the port defaults to 47800.
pub fn parse_relay(addr: &str) -> std::io::Result<SocketAddr> {
    let addr = addr.trim();
    let has_port = addr.rsplit_once(':').is_some_and(|(h, p)| p.parse::<u16>().is_ok() && (!h.contains(':') || h.ends_with(']')));
    if has_port {
        parse_addr(addr)
    } else {
        parse_addr(&format!("{addr}:{}", relay_proto::DEFAULT_PORT))
    }
}

fn parse_accounts(spec: &str) -> Vec<NetAccount> {
    let mut out = Vec::new();
    for entry in spec.split(['\n', ';', ',']) {
        let e = entry.trim();
        if e.is_empty() {
            continue;
        }
        let Some((name, password)) = e.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        out.push(NetAccount {
            name: name.to_string(),
            password: password.trim().to_string(),
        });
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConnState {
    Connecting,
    Connected,
    Disconnected(String),
}

/// Traffic counters for the Network window.
#[derive(Clone, Copy, Debug, Default)]
pub struct NetStats {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub packets_in: u64,
    pub packets_out: u64,
    pub resends: u64,
    /// Datagrams dropped for failing authentication / replay / cookie checks.
    pub rejected: u64,
    /// Joins turned away because too many handshakes were half-open.
    pub half_open_refused: u64,
    /// Host: client data refused by the validation layer (all reasons).
    pub violations: u64,
    /// Host: packets and messages ignored because their sender is over the
    /// drop-level strike score.
    pub muted: u64,
    /// Host: clients the anti-cheat provider rejected.
    pub ac_rejected: u64,
    /// Host, interest management: clients tracked, objects in the spatial
    /// hash, relevant objects per client (average), updates held back by a
    /// client's budget / merged by a distance tier (totals over all clients).
    pub interest_clients: u32,
    pub interest_objects: u32,
    pub interest_relevant_avg: f32,
    pub interest_dropped: u64,
    pub interest_throttled: u64,
    /// Bytes per second over the last full second.
    pub rate_in: f32,
    pub rate_out: f32,
    window_start: f64,
    window_in: u64,
    window_out: u64,
}

/// A replicated runtime spawn, kept by every peer for late-join catch-up.
#[derive(Clone, Debug)]
pub struct SpawnRecord {
    pub id: u128,
    pub source: NetSpawnSource,
    pub translation: Vec3,
    pub rotation: Quat,
    pub owner: u64,
    /// Its owner's player object (despawned when they leave).
    pub player: bool,
}

/// Latest known world transform of an object.
#[derive(Clone, Copy, Debug)]
pub struct XformState {
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
    pub vel: Vec3,
}

impl Default for XformState {
    fn default() -> Self {
        Self { pos: Vec3::ZERO, rot: Quat::IDENTITY, scale: Vec3::ONE, vel: Vec3::ZERO }
    }
}

/// The replicated registry: authoritative on the host, mirrored on clients.
#[derive(Default)]
pub struct NetWorld {
    /// netid -> (owner, seq). Owner 0 = unowned (host authority).
    pub owners: HashMap<u128, (u64, u64)>,
    pub spawns: Vec<SpawnRecord>,
    pub vars: HashMap<u128, Vec<(String, NetValue)>>,
    pub actives: HashMap<u128, bool>,
    pub parents: HashMap<u128, u128>,
    pub xforms: HashMap<u128, XformState>,
    pub anims: HashMap<u128, HashMap<String, f32>>,
    /// Last `P_AVPARAM` packet (schema hash + payload) per avatar.
    pub avparams: HashMap<u128, Vec<u8>>,
    /// Scene the host switched everyone to with `net_load_scene`.
    pub scene: Option<String>,
    // Host-side settings gathered from the scene each frame.
    pub modes: HashMap<u128, OwnershipMode>,
    pub var_anyone: HashSet<u128>,
    pub relevancy: HashMap<u128, f32>,
    /// Host, interest management: objects every client is told about.
    pub always_relevant: HashSet<u128>,
    pub despawn_on_leave: HashSet<u128>,
    /// Each peer's player object (relevancy origin, despawned on leave).
    pub player_obj: HashMap<u64, u128>,
    /// Host: predicted objects (the host simulates them from their owner's input).
    pub predicted: HashSet<u128>,
}

impl NetWorld {
    fn forget(&mut self, id: u128) {
        self.owners.remove(&id);
        self.spawns.retain(|s| s.id != id);
        self.vars.remove(&id);
        self.actives.remove(&id);
        self.parents.remove(&id);
        self.xforms.remove(&id);
        self.anims.remove(&id);
        self.avparams.remove(&id);
        self.despawn_on_leave.remove(&id);
        self.player_obj.retain(|_, o| *o != id);
    }
    fn set_var(&mut self, id: u128, key: &str, value: NetValue) {
        let vars = self.vars.entry(id).or_default();
        match vars.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => vars.push((key.to_owned(), value)),
        }
    }
    pub fn var(&self, id: u128, key: &str) -> Option<&NetValue> {
        self.vars.get(&id)?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// One transform sample (sender clock).
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    pub time: f64,
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
    pub vel: Vec3,
}

/// Received snapshots for one object, oldest first.
#[derive(Default)]
pub struct SnapBuffer {
    pub origin: u64,
    pub snaps: VecDeque<Snap>,
    /// Set by a late-join baseline: hold it until live snapshots arrive.
    pub baseline: bool,
}

/// One received avatar sample: world poses in wire order (head, hands, hips, feet).
#[derive(Clone, Copy, Debug)]
pub struct AvatarSnap {
    pub time: f64,
    pub poses: [Option<(Vec3, Quat)>; TRACKER_SLOTS],
}

#[derive(Default)]
pub struct AvatarBuffer {
    pub origin: u64,
    pub snaps: VecDeque<AvatarSnap>,
}

/// A transform entry on the wire. Omitted fields are unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct SnapEntry {
    pub id: u128,
    pub pos: Option<Vec3>,
    pub rot: Option<Quat>,
    pub scale: Option<Vec3>,
    pub vel: Option<Vec3>,
    pub quantized: bool,
}

/// Compat: latest received transform of an object.
#[derive(Clone, Copy)]
pub struct RemoteTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub seq: u64,
}

/// A received voice frame (mono PCM) tagged by sender + sequence for the jitter
/// buffer to reorder.
pub struct VoicePacket {
    pub from: u64,
    pub seq: u32,
    /// Codec payload (first byte names the codec, see `voice_codec`).
    pub payload: Vec<u8>,
}

/// A synced-variable write waiting for its object to exist in the scene.
pub struct PendingVar {
    pub id: u128,
    pub key: String,
    pub value: NetValue,
    pub expires: f64,
}

/// One peer in line to take over as host.
#[derive(Clone, Debug, PartialEq)]
pub struct Successor {
    pub peer: u64,
    /// Endpoints other members can try directly (empty for relay-only peers).
    pub addrs: Vec<SocketAddr>,
}

/// The host's succession plan, replicated to every member ahead of time:
/// who takes over (in order), and what the new host needs to carry on.
#[derive(Clone, Debug, Default)]
pub struct Succession {
    pub version: u32,
    pub epoch: u32,
    pub next_peer: u64,
    pub spawn_counter: u32,
    pub max_players: u32,
    pub clients_can_spawn: bool,
    pub session_name: String,
    /// Relay the session's room lives on, and its code ("" = none).
    pub relay: String,
    pub room: String,
    /// The room owner key members claim the room with.
    pub room_key: Option<[u8; 32]>,
    /// Candidates in election order (join order; peers that can't host left out).
    pub order: Vec<Successor>,
    pub ban: Vec<String>,
    pub allow: Vec<String>,
}

/// Where a migrating client is (for the UI).
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationStatus {
    /// The host that was lost.
    pub previous: u64,
    /// The successor being tried now.
    pub target: u64,
    /// Through the relay rather than directly.
    pub relayed: bool,
    pub seconds: f32,
}

struct Conn {
    addr: SocketAddr,
    peer: u64,
    name: String,
    last_recv: f64,
    last_ping: f64,
    rtt: f32,
    send_seq: u32,
    /// seq -> (packet, last sent, first fragment of its message).
    unacked: BTreeMap<u32, (Vec<u8>, f64, bool)>,
    recv_next: u32,
    recv_buf: BTreeMap<u32, (bool, Vec<u8>)>,
    frag: Vec<u8>,
    acks: Vec<u32>,
    sec: Option<SecureChannel>,
    client_nonce: [u8; 16],
    /// Smoothed round-trip variation (s).
    rtt_var: f32,
    /// Loss the other side measures on what we send.
    remote_loss: f32,
    /// The client's reported interpolation delay (s).
    view_delay: f32,
    /// Host: the client's verified player id and handshake transcript.
    identity: [u8; 32],
    /// Account login used for this connection (empty = none).
    account: String,
    transcript: [u8; 32],
    /// Host: the client has sent a sealed packet since its JOIN (until then
    /// it is a half-open join, not yet a member).
    confirmed: bool,
    created: f64,
    /// Host: a member coming back after a host migration (keeps its peer id).
    rejoined: bool,
    can_host: bool,
}

impl Conn {
    fn new(addr: SocketAddr, peer: u64, now: f64) -> Self {
        Self {
            sec: None,
            client_nonce: [0; 16],
            rtt_var: 0.0,
            remote_loss: 0.0,
            view_delay: 0.0,
            addr,
            peer,
            name: String::new(),
            last_recv: now,
            last_ping: now,
            rtt: 0.05,
            send_seq: 0,
            unacked: BTreeMap::new(),
            recv_next: 0,
            recv_buf: BTreeMap::new(),
            frag: Vec::new(),
            acks: Vec::new(),
            identity: [0; 32],
            account: String::new(),
            transcript: [0; 32],
            confirmed: true,
            created: now,
            rejoined: false,
            can_host: true,
        }
    }
    fn rto(&self) -> f64 {
        (self.rtt as f64 * 2.0 + 0.05).clamp(0.08, 1.0)
    }
    /// Reliable messages sent but not acknowledged, reassembled in order
    /// (resent to a new host after a migration).
    fn unacked_messages(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut cur: Option<Vec<u8>> = None;
        let mut seqs: Vec<&u32> = self.unacked.keys().collect();
        seqs.sort_by_key(|s| s.wrapping_sub(self.send_seq));
        for seq in seqs {
            let (p, _, first) = &self.unacked[seq];
            if p.len() < 6 {
                continue;
            }
            if *first {
                cur = Some(Vec::new());
            }
            if let Some(m) = cur.as_mut() {
                m.extend_from_slice(&p[6..]);
                if p[5] != 0 {
                    out.push(cur.take().unwrap_or_default());
                }
            }
        }
        out
    }
}

enum Clock {
    Real(Instant),
    Manual(f64),
}

/// Per-object outgoing replication state (owned by `net_sync`).
#[derive(Default)]
pub struct SendState {
    pub last_send: f64,
    pub last_key: f64,
    pub sent: Option<XformState>,
    pub prev_pos: Option<(Vec3, f64)>,
    pub parent: Option<u128>,
    pub active: Option<bool>,
    pub anim: HashMap<String, f32>,
    pub last_anim: f64,
    pub last_anim_full: f64,
    pub last_avatar: f64,
    pub seen: bool,
    pub missing_since: Option<f64>,
    /// Rest snapshots still to send after the object stopped moving.
    pub rest: u8,
}

/// A host's CHALLENGE as a client parsed it.
#[derive(Clone)]
struct Challenge {
    kdf: KdfParams,
    host_pk: [u8; 32],
    epoch: u32,
    host_peer: u64,
    cookie: [u8; 16],
    transcript: [u8; 32],
}

/// A client's handshake in progress (joining, or rejoining a new host).
struct Handshake {
    /// Where HELLOs go until a CHALLENGE picks one.
    addrs: Vec<SocketAddr>,
    addr: Option<SocketAddr>,
    nonce: [u8; 16],
    started: f64,
    last_hello: f64,
    /// Migration: the peer that must answer, and the player id the session
    /// knows it by.
    expect: Option<(u64, Option<[u8; 32]>)>,
    challenge: Option<Challenge>,
    /// Other answers to our HELLO, tried in turn when the chosen one's JOIN
    /// gets no WELCOME: an on-path attacker can race a forged CHALLENGE
    /// ahead of the host's, which would otherwise stall the join.
    spare: Vec<(Challenge, SocketAddr)>,
    join: Option<Vec<u8>>,
    /// When the JOIN was sealed.
    join_sent: f64,
    sec: Option<SecureChannel>,
    /// An unauthenticated rejection (believed only if nothing better arrives).
    hint: Option<(String, f64)>,
}

/// A client looking for the new host after losing the old one.
struct Migration {
    previous: u64,
    candidates: Vec<Successor>,
    index: usize,
    relayed: bool,
    attempt_started: Option<f64>,
    started: f64,
}

/// A promoted host checking whether a better-ranked successor is hosting too.
struct Probe {
    peer: u64,
    addr: SocketAddr,
    nonce: [u8; 16],
    sec: Option<SecureChannel>,
    join: Option<Vec<u8>>,
    last: f64,
}

/// A client re-learning the world from a new host.
#[derive(Default)]
struct Resync {
    spawns: HashSet<u128>,
    owners: HashSet<u128>,
    mine_before: Vec<SpawnRecord>,
}

/// A reliable RPC or chat line the new host replays to members that are
/// still reconnecting after a migration (they drop ones they already saw).
/// A player's report against another, queued on the host for the operators.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerReport {
    pub reporter: u64,
    pub reporter_name: String,
    pub reporter_fingerprint: String,
    pub target: u64,
    pub target_name: String,
    pub target_fingerprint: String,
    pub reason: String,
    /// Session clock when it arrived.
    pub at: f64,
}

/// Reports kept waiting for the operators at most.
const MAX_REPORTS: usize = 256;

/// Host: anti-cheat state of one client.
#[derive(Clone, Copy)]
struct AcPeer {
    since: f64,
    trusted: bool,
}

struct Backlog {
    sender: u64,
    msg: Vec<u8>,
}

// ------------------------------------------------------------------ session

/// One live networking session (host or client).
pub struct NetSession {
    transport: Box<dyn Transport>,
    discovery: Option<UdpSocket>,
    pub cfg: NetConfig,
    clock: Clock,
    is_host: bool,
    local_peer: u64,
    /// The current host's peer id (1 until a migration).
    host_peer: u64,
    /// Bumped by every host migration.
    epoch: u32,
    state: ConnState,
    conns: HashMap<u64, Conn>,
    addr_peer: HashMap<SocketAddr, u64>,
    /// Everyone known (id -> name), the local peer included.
    peers: BTreeMap<u64, String>,
    /// Verified player ids (the host checked their signatures).
    peer_ids: HashMap<u64, [u8; 32]>,
    next_peer: u64,
    claim_seq: u64,
    xform_seq: u64,
    pub world: NetWorld,
    pub stats: NetStats,
    /// Interest management state (host) and the aliases it taught us (client).
    pub interest: Interest,
    // Outputs drained by `view()`.
    inbox: Vec<(u64, bool, String)>,
    events: Vec<NetEvent>,
    rpcs: Vec<NetRpc>,
    world_ops: Vec<NetWorldOp>,
    hold_ops_until: f64,
    voice_in: Vec<VoicePacket>,
    // Replication inputs/outputs used by `net_sync`.
    pub pending_vars: Vec<PendingVar>,
    pub pending_parents: Vec<(u128, u128, f64)>,
    pub buffers: HashMap<u128, SnapBuffer>,
    pub offsets: HashMap<u64, f64>,
    pub anim_in: HashMap<u128, HashMap<String, f32>>,
    /// Newest avatar parameter packet per avatar and a counter that grows with each.
    pub avparam_in: HashMap<u128, (u64, Vec<u8>)>,
    pub avparam_seq: u64,
    pub avparam_state: HashMap<u128, crate::avatar_net::AvatarNetState>,
    pub avatar_in: HashMap<u128, AvatarBuffer>,
    /// Latest face frame per avatar: (sender time, frame).
    pub face_in: HashMap<u128, (f64, zestengine_core::FaceFrame)>,
    pub send_state: HashMap<u128, SendState>,
    pub remote_bodies: HashMap<ObjectId, Vec3>,
    pub reclaimed_bodies: HashMap<ObjectId, Vec3>,
    /// Peers whose player object the host still has to spawn.
    pub players_to_spawn: Vec<u64>,
    pub host_player_done: bool,
    pub spawn_counter: u32,
    /// The session's key stretching (host: chosen; client: from the CHALLENGE).
    kdf: KdfParams,
    /// Stretched session keys by parameters, and the one being computed.
    keys: Vec<(KdfParams, [u8; 32])>,
    kdf_job: Option<KdfJob>,
    /// Host: keys the stateless handshake cookies.
    cookie_key: [u8; 32],
    /// Host: client nonces already accepted (a replayed JOIN is ignored).
    seen_nonces: HashMap<[u8; 16], f64>,
    join_bucket: Bucket,
    legacy_bucket: Bucket,
    /// Client: the handshake in progress.
    hs: Option<Handshake>,
    /// What to pin the host's identity under (trust on first use).
    pin_target: Option<String>,
    closed: bool,
    /// Host: queued inputs of remote players' predicted objects.
    pub inputs: HashMap<u128, InputQueue>,
    /// Owner: newest authoritative state of each predicted object.
    pub auth_in: HashMap<u128, AuthState>,
    /// New host: owners' predicted states to continue from after a migration,
    /// and the objects waiting for one (deadline).
    anchors: HashMap<u128, AuthState>,
    anchor_wait: HashMap<u128, f64>,
    /// Owner: prediction history of each predicted object.
    pub predictors: HashMap<u128, Predictor>,
    /// Host: display offset of each remote predicted object, so it glides
    /// while its owner's inputs arrive in bursts.
    pub host_smoothing: HashMap<u128, Vec3>,
    /// Snapshot arrival jitter per origin peer.
    pub jitter: HashMap<u64, JitterMeter>,
    /// Per-object adaptive interpolation delay.
    pub delays: HashMap<u128, AdaptiveDelay>,
    /// How far behind remote objects render (reported to the host, used for lag compensation).
    pub view_delay: f32,
    /// Host: hitbox history for lag compensation (`net_sync` records it).
    pub history: Arc<Mutex<HitHistory>>,
    /// Host: validation of what clients send (limits, strikes, bans).
    pub val: Validator,
    /// Anti-cheat provider (SEC-2): the server half judges clients, the client half attests this one.
    ac: Box<dyn AntiCheatProvider>,
    /// Host: per-client anti-cheat state.
    ac_peers: HashMap<u64, AcPeer>,
    /// Host: player reports not yet taken by the server.
    reports: Vec<PlayerReport>,
    /// Host, anti-cheat `require`: clients admitted but held, with whether they are rejoining
    /// after a migration. They get the world (and a player object) once they are Trusted.
    ac_deferred: HashMap<u64, bool>,
    /// Host: the provider account (ban-list key) each trusted client was linked to, by peer.
    ac_links: HashMap<u64, String>,
    /// Client: the provider's client half was started.
    ac_started: bool,
    /// Owner: where the host says objects we move belong after it refused a
    /// move (`net_sync` puts them back): (object, position, rotation).
    pub corrections: Vec<(u128, Vec3, Quat)>,
    /// Client: the session's topology as the host announced it.
    topology: Topology,
    // Host migration.
    succession: Option<Succession>,
    succession_version: u32,
    migration: Option<Migration>,
    /// New host: members expected back, with their deadline.
    reconnecting: HashMap<u64, f64>,
    reconnect_expected: bool,
    rejoins: u32,
    /// Client: reliable messages for the host while there is none.
    outbox: Vec<Vec<u8>>,
    backlog: Vec<Backlog>,
    resync: Option<Resync>,
    probes: Vec<Probe>,
    probe_until: f64,
    /// Completed migrations (the UI shows the count).
    migrations: u32,
    // Duplicate suppression for messages resent across a migration.
    mid_out: u32,
    seen_mid: HashMap<u64, u32>,
    // Session clock: (local time, value) at the last pump, the host's
    // offset of the session clock over its own, and the host's as told.
    clock_value: Option<(f64, f64)>,
    clock_offset: f64,
    host_clock_offset: f64,
    /// Room owner key (host-generated, shared with members for migration).
    room_key: Option<[u8; 32]>,
    relay_addr: String,
    relay_auth_sent: bool,
}

impl NetSession {
    /// Start hosting on `port` (0 = OS-assigned). The host is peer id 1.
    pub fn host(port: u16) -> std::io::Result<Self> {
        Self::host_with(port, NetConfig::default())
    }

    pub fn host_with(port: u16, cfg: NetConfig) -> std::io::Result<Self> {
        let mut t = UdpTransport::bind(port)?;
        tracing::info!("net: hosting on {:?}", t.local_addr());
        let relay = if cfg.relay.trim().is_empty() { None } else { Some(parse_relay(&cfg.relay)?) };
        if let Some(relay) = relay {
            tracing::info!("net: opening a room on relay {relay}");
            t = t.with_relay(relay, RelayRole::Host { code: cfg.room.clone() }, cfg.punch);
        }
        let mut s = Self::host_on(Box::new(t), cfg);
        if let Some(relay) = relay {
            s.room_key = Some(random_bytes());
            s.relay_addr = relay.to_string();
        }
        if s.cfg.discovery {
            match bind_shared_udp(s.cfg.discovery_port).and_then(|d| d.set_nonblocking(true).map(|_| d)) {
                Ok(d) => s.discovery = Some(d),
                Err(e) => tracing::warn!("net: LAN discovery unavailable (port {}): {e}", s.cfg.discovery_port),
            }
        }
        Ok(s)
    }

    /// Answer LAN discovery queries on the shared `port` (see
    /// [`DISCOVERY_PORT`]). Call once after constructing a host; replies are
    /// served by [`NetSession::pump`]. No-op if it cannot bind.
    pub fn enable_discovery(&mut self, port: u16) {
        if self.discovery.is_some() {
            return;
        }
        match bind_shared_udp(port).and_then(|d| d.set_nonblocking(true).map(|_| d)) {
            Ok(d) => self.discovery = Some(d),
            Err(e) => tracing::warn!("net: LAN discovery unavailable (port {port}): {e}"),
        }
    }

    /// Host over any transport (tests use [`MemHub`]).
    pub fn host_on(transport: Box<dyn Transport>, cfg: NetConfig) -> Self {
        let mut s = Self::new(transport, cfg, true);
        s.local_peer = HOST_PEER;
        s.host_peer = HOST_PEER;
        s.state = ConnState::Connected;
        s.peers.insert(HOST_PEER, s.cfg.player_name.clone());
        s.peer_ids.insert(HOST_PEER, s.cfg.identity.public());
        s.kdf = s.cfg.kdf.unwrap_or_else(|| KdfParams::for_secret(&s.cfg.secret));
        s.kdf_job = Some(KdfJob::start(&s.cfg.secret, s.kdf));
        s.events.push(NetEvent::Connected { peer: HOST_PEER });
        s
    }

    /// Join a host at `addr` (e.g. "127.0.0.1:9000").
    pub fn join(addr: &str) -> std::io::Result<Self> {
        Self::join_with(addr, NetConfig::default())
    }

    /// Join `addr`: `host:port` directly, or a room code (`ABC123` on the
    /// configured relay, or `ABC123@relay:port`).
    pub fn join_with(addr: &str, cfg: NetConfig) -> std::io::Result<Self> {
        match JoinTarget::parse(addr, &cfg.relay)? {
            JoinTarget::Direct(host) => {
                let t = UdpTransport::bind(0)?;
                tracing::info!("net: joining {host}");
                Ok(Self::join_on(Box::new(t), host, cfg))
            }
            JoinTarget::Room { code, relay } => {
                tracing::info!("net: joining room {code} on relay {relay}");
                let t = UdpTransport::bind(0)?.with_relay(relay, RelayRole::Client { code: code.clone() }, cfg.punch);
                let mut s = Self::join_on(Box::new(t), virtual_addr(HOST_SLOT), cfg);
                s.pin_target = Some(format!("room:{code}@{relay}"));
                s.relay_addr = relay.to_string();
                Ok(s)
            }
        }
    }

    pub fn join_on(transport: Box<dyn Transport>, host: SocketAddr, cfg: NetConfig) -> Self {
        let mut s = Self::new(transport, cfg, false);
        s.pin_target = Some(host.to_string());
        s.start_handshake(vec![host], None);
        s
    }

    fn new(transport: Box<dyn Transport>, mut cfg: NetConfig, is_host: bool) -> Self {
        let kdf = KdfParams::for_secret(&cfg.secret);
        if cfg.topology == Topology::Dedicated {
            // A dedicated session ends with its server: nobody can take over.
            cfg.migration = false;
            cfg.can_host = false;
        }
        cfg.validation = std::mem::take(&mut cfg.validation).sanitized();
        cfg.anticheat = std::mem::take(&mut cfg.anticheat).sanitized();
        cfg.interest = std::mem::take(&mut cfg.interest).sanitized();
        let interest = Interest::with_config(cfg.interest.clone());
        let mut val = Validator::new(cfg.validation.clone(), cfg.topology);
        if is_host && let Some(path) = &cfg.ban_file {
            val.bans = crate::net_validate::BanList::open(path);
            val.bans.prune(crate::net_validate::unix_now());
        }
        let mut history = HitHistory::default();
        history.configure(val.settings().max_rewind, val.settings().future_tolerance, val.strict());
        let topology = cfg.topology;
        let ac = zestengine_core::create_provider(&cfg.anticheat);
        Self {
            transport,
            discovery: None,
            cfg,
            clock: Clock::Real(Instant::now()),
            is_host,
            local_peer: 0,
            host_peer: HOST_PEER,
            epoch: 0,
            state: ConnState::Connecting,
            conns: HashMap::new(),
            addr_peer: HashMap::new(),
            peers: BTreeMap::new(),
            peer_ids: HashMap::new(),
            next_peer: 2,
            claim_seq: 0,
            xform_seq: 0,
            world: NetWorld::default(),
            stats: NetStats::default(),
            interest,
            inbox: Vec::new(),
            events: Vec::new(),
            rpcs: Vec::new(),
            world_ops: Vec::new(),
            hold_ops_until: 0.0,
            voice_in: Vec::new(),
            pending_vars: Vec::new(),
            pending_parents: Vec::new(),
            buffers: HashMap::new(),
            offsets: HashMap::new(),
            anim_in: HashMap::new(),
            avparam_in: HashMap::new(),
            avparam_seq: 0,
            avparam_state: HashMap::new(),
            avatar_in: HashMap::new(),
            face_in: HashMap::new(),
            send_state: HashMap::new(),
            remote_bodies: HashMap::new(),
            reclaimed_bodies: HashMap::new(),
            players_to_spawn: Vec::new(),
            host_player_done: false,
            spawn_counter: 0,
            kdf,
            keys: Vec::new(),
            kdf_job: None,
            cookie_key: random_bytes(),
            seen_nonces: HashMap::new(),
            join_bucket: Bucket::new(64.0, 64.0),
            legacy_bucket: Bucket::new(10.0, 10.0),
            hs: None,
            pin_target: None,
            closed: false,
            inputs: HashMap::new(),
            auth_in: HashMap::new(),
            anchors: HashMap::new(),
            anchor_wait: HashMap::new(),
            predictors: HashMap::new(),
            host_smoothing: HashMap::new(),
            jitter: HashMap::new(),
            delays: HashMap::new(),
            view_delay: 0.0,
            history: Arc::new(Mutex::new(history)),
            val,
            ac,
            ac_peers: HashMap::new(),
            reports: Vec::new(),
            ac_deferred: HashMap::new(),
            ac_links: HashMap::new(),
            ac_started: false,
            corrections: Vec::new(),
            topology,
            succession: None,
            succession_version: 0,
            migration: None,
            reconnecting: HashMap::new(),
            reconnect_expected: false,
            rejoins: 0,
            outbox: Vec::new(),
            backlog: Vec::new(),
            resync: None,
            probes: Vec::new(),
            probe_until: 0.0,
            migrations: 0,
            mid_out: 0,
            seen_mid: HashMap::new(),
            clock_value: None,
            clock_offset: 0.0,
            host_clock_offset: 0.0,
            room_key: None,
            relay_addr: String::new(),
            relay_auth_sent: false,
        }
    }

    /// Drive time by hand (tests): `now()` only moves with [`Self::advance`].
    /// Key stretching then finishes before the handshake goes on, so
    /// handshakes take a fixed number of steps.
    pub fn use_manual_clock(&mut self) {
        let now = self.now();
        self.clock = Clock::Manual(now);
    }
    pub fn advance(&mut self, dt: f64) {
        if let Clock::Manual(t) = &mut self.clock {
            *t += dt;
        }
    }
    /// Seconds since the session started (this machine's clock).
    pub fn now(&self) -> f64 {
        match &self.clock {
            Clock::Real(start) => start.elapsed().as_secs_f64(),
            Clock::Manual(t) => *t,
        }
    }

    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.transport.local_addr()
    }
    pub fn is_host(&self) -> bool {
        self.is_host
    }
    pub fn local_peer(&self) -> u64 {
        self.local_peer
    }
    /// The current host's peer id (1 until a migration).
    pub fn host_peer(&self) -> u64 {
        self.host_peer
    }
    /// How many times the host changed hands.
    pub fn epoch(&self) -> u32 {
        self.epoch
    }
    /// The session's key-stretching parameters.
    pub fn kdf(&self) -> KdfParams {
        self.kdf
    }
    pub fn migrations(&self) -> u32 {
        self.migrations
    }
    pub fn state(&self) -> &ConnState {
        &self.state
    }
    pub fn is_connected(&self) -> bool {
        self.state == ConnState::Connected
    }
    pub fn peer_count(&self) -> usize {
        self.peers.len().max(1)
    }
    /// Round-trip time to the host in ms (0 on the host).
    pub fn rtt_ms(&self) -> f32 {
        if self.is_host {
            0.0
        } else {
            self.conns.get(&self.host_peer).map(|c| c.rtt * 1000.0).unwrap_or(0.0)
        }
    }

    /// This install's player identity.
    pub fn identity(&self) -> &Identity {
        &self.cfg.identity
    }
    /// A peer's verified player id.
    pub fn peer_identity(&self, peer: u64) -> Option<[u8; 32]> {
        self.peer_ids.get(&peer).copied()
    }
    pub fn peer_fingerprint(&self, peer: u64) -> Option<String> {
        self.peer_ids.get(&peer).map(fingerprint)
    }
    /// The replicated succession plan (who takes over if the host goes).
    pub fn succession(&self) -> Option<&Succession> {
        self.succession.as_ref()
    }
    /// Who would take over right now (None without a plan or migration off).
    pub fn next_host(&self) -> Option<u64> {
        if !self.cfg.migration {
            return None;
        }
        if self.is_host {
            return self.successors().first().map(|s| s.peer);
        }
        self.succession.as_ref()?.order.iter().map(|s| s.peer).find(|p| *p != self.host_peer)
    }
    /// Searching for a new host (None while settled).
    pub fn migration_status(&self) -> Option<MigrationStatus> {
        let m = self.migration.as_ref()?;
        let target = m.candidates.get(m.index).map_or(0, |c| c.peer);
        Some(MigrationStatus { previous: m.previous, target, relayed: m.relayed, seconds: (self.now() - m.started) as f32 })
    }
    /// New host: members still expected back after a migration.
    pub fn reconnecting(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.reconnecting.keys().copied().collect();
        v.sort();
        v
    }
    /// Host: joins accepted but not yet confirmed.
    pub fn half_open(&self) -> usize {
        self.conns.values().filter(|c| !c.confirmed).count()
    }

    /// Everyone in the session, local peer included.
    pub fn peer_infos(&self) -> Vec<PeerInfo> {
        self.peers
            .iter()
            .map(|(&id, name)| {
                let conn = if id == self.local_peer || (!self.is_host && id != self.host_peer) { None } else { self.conns.get(&id) };
                let identity = self.peer_ids.get(&id).copied();
                let mut info = PeerInfo {
                    id,
                    name: name.clone(),
                    is_host: id == self.host_peer,
                    fingerprint: identity.map(|k| fingerprint(&k)).unwrap_or_default(),
                    identity,
                    ..Default::default()
                };
                if let Some(c) = conn {
                    let q = self.conn_quality(c);
                    info.rtt_ms = q.rtt_ms;
                    info.jitter_ms = q.jitter_ms;
                    info.loss = 1.0 - (1.0 - q.loss_in) * (1.0 - q.loss_out);
                    info.view_delay = c.view_delay;
                }
                info
            })
            .collect()
    }

    fn conn_quality(&self, c: &Conn) -> NetQuality {
        NetQuality {
            rtt_ms: c.rtt * 1000.0,
            jitter_ms: c.rtt_var * 1000.0,
            loss_in: c.sec.as_ref().map_or(0.0, |s| s.loss.loss),
            loss_out: c.remote_loss,
            relayed: self.transport.route_label(c.addr) == "relayed",
        }
    }

    /// Connection quality to the host (zeros on the host).
    pub fn quality(&self) -> NetQuality {
        if self.is_host {
            return NetQuality::default();
        }
        self.conns.get(&self.host_peer).map(|c| self.conn_quality(c)).unwrap_or_default()
    }

    /// Incoming loss on the link packets from `origin` arrive over (the
    /// host's link to that client, or a client's link to the host).
    pub fn link_loss(&self, origin: u64) -> f32 {
        let peer = if self.is_host { origin } else { self.host_peer };
        self.conns.get(&peer).and_then(|c| c.sec.as_ref()).map_or(0.0, |s| s.loss.loss)
    }

    /// Relay registration and routes (None without a relay).
    pub fn relay_status(&self) -> Option<RelayStatus> {
        self.transport.relay_status()
    }

    /// The room code peers join with, once the relay assigned it.
    pub fn room_code(&self) -> Option<String> {
        self.relay_status().filter(|r| r.phase == RelayPhase::Registered).map(|r| r.code)
    }

    /// "direct", "relayed" or "LAN" for a connected peer.
    pub fn peer_route(&self, peer: u64) -> Option<&'static str> {
        self.conns.get(&peer).map(|c| self.transport.route_label(c.addr))
    }

    /// Owner peer of a network id (0 = unowned).
    pub fn owner_of_raw(&self, id: u128) -> u64 {
        self.world.owners.get(&id).map(|(o, _)| *o).unwrap_or(0)
    }
    pub fn owner_of(&self, id: ObjectId) -> u64 {
        self.owner_of_raw(id.raw())
    }

    /// Whether the local peer holds ownership of an object.
    pub fn owns(&self, id: ObjectId) -> bool {
        self.local_peer != 0 && self.owner_of(id) == self.local_peer
    }

    /// Whether the local peer simulates an object (owner, or host when
    /// unowned). Unowned objects have no authority while the host is being
    /// replaced.
    pub fn has_authority_raw(&self, id: u128) -> bool {
        if !self.is_connected() {
            return false;
        }
        match self.owner_of_raw(id) {
            0 => self.is_host,
            o => o == self.local_peer,
        }
    }

    /// Non-draining `(object, owner)` snapshot, used to position remote peers'
    /// voices at the object they own (spatial voice).
    pub fn owners_snapshot(&self) -> Vec<(ObjectId, u64)> {
        let mut out: Vec<(ObjectId, u64)> = self
            .world
            .player_obj
            .iter()
            .map(|(p, id)| (ObjectId::from_raw(*id), *p))
            .collect();
        out.extend(
            self.world
                .owners
                .iter()
                .filter(|(id, (o, _))| *o != 0 && !self.world.player_obj.values().any(|p| p == *id))
                .map(|(id, (o, _))| (ObjectId::from_raw(*id), *o)),
        );
        out
    }

    /// Build the per-frame [`NetView`] components see, draining received
    /// messages, events and RPCs into it (each is delivered once).
    pub fn view(&mut self) -> NetView {
        let owners = self
            .world
            .owners
            .iter()
            .filter(|(_, (o, _))| *o != 0)
            .map(|(id, (o, _))| (ObjectId::from_raw(*id), *o))
            .collect();
        // After a scene switch, hold later world ops until the new scene is in.
        let world_ops = if self.now() < self.hold_ops_until {
            match self.world_ops.iter().rposition(|op| matches!(op, NetWorldOp::LoadScene(_))) {
                Some(i) => self.world_ops.drain(..=i).collect(),
                None => Vec::new(),
            }
        } else {
            std::mem::take(&mut self.world_ops)
        };
        NetView {
            connected: self.is_connected(),
            is_server: self.is_host,
            local_peer: self.local_peer,
            host_peer: self.host_peer,
            migrating: self.migration.is_some(),
            owners,
            messages: std::mem::take(&mut self.inbox),
            peers: self.peer_infos(),
            events: std::mem::take(&mut self.events),
            rpcs: std::mem::take(&mut self.rpcs),
            world_ops,
            remote_bodies: self.remote_bodies.clone(),
            reclaimed_bodies: std::mem::take(&mut self.reclaimed_bodies),
            rtt_ms: self.rtt_ms(),
            quality: self.quality(),
            clock: self.shared_clock().unwrap_or(0.0),
            view_delay: self.view_delay,
            rewind: self.is_host.then(|| self.history.clone()),
        }
    }

    /// Drain voice packets received since the last call (for the jitter buffer).
    pub fn take_voice(&mut self) -> Vec<VoicePacket> {
        std::mem::take(&mut self.voice_in)
    }

    /// An empty hitbox history with this session's rewind window.
    fn fresh_history(&self) -> HitHistory {
        let mut h = HitHistory::default();
        h.configure(self.val.settings().max_rewind, self.val.settings().future_tolerance, self.val.strict());
        h
    }

    /// What this client has pinned for the host it joined (`host:port` or
    /// `room:CODE@relay`), for messages that tell the user what to clear.
    pub fn host_pin_target(&self) -> Option<&str> {
        self.pin_target.as_deref()
    }

    /// Forget the saved identity key of a host (`host:port`, or
    /// `room:CODE@relay`), so the next join pins whatever key it has now.
    /// What a user does after a server was legitimately reinstalled.
    pub fn forget_host_pin(target: &str) -> bool {
        crate::net_identity::forget_pin(target)
    }

    /// Whether an identity key is saved for the host `target`.
    pub fn host_is_pinned(target: &str) -> bool {
        crate::net_identity::is_pinned(target)
    }

    /// Register an RPC's permission and rate limit (host side; see
    /// `NetValidation::rpc_rules` for doing it from settings).
    pub fn register_rpc(&mut self, rule: zestengine_core::RpcRule) {
        self.val.register_rpc(rule);
    }

    /// Tooling / test hook: send `plain` to the host as one sealed datagram,
    /// bypassing every client-side check (a misbehaving client).
    #[doc(hidden)]
    pub fn inject_plain(&mut self, plain: &[u8]) {
        let host = self.host_peer;
        self.send_to_peer(host, plain);
    }

    /// Tooling / test hook: the sealed datagram for `plain`, not sent (to replay it).
    #[doc(hidden)]
    pub fn seal_plain(&mut self, plain: &[u8]) -> Option<Vec<u8>> {
        let host = self.host_peer;
        self.conns.get_mut(&host)?.sec.as_mut().map(|s| s.seal(&[P_SECURE], plain))
    }

    /// Tooling / test hook: send `msg` to the host on the reliable channel.
    #[doc(hidden)]
    pub fn inject_reliable(&mut self, msg: &[u8]) {
        let host = self.host_peer;
        self.send_reliable(host, msg);
    }

    /// Tooling / test hook: send the first fragment of a reliable message that
    /// never ends (`n` bytes), to test the host's buffer caps.
    #[doc(hidden)]
    pub fn inject_endless_fragments(&mut self, fragments: usize) {
        let host = self.host_peer;
        let now = self.now();
        let Some(c) = self.conns.get_mut(&host) else {
            return;
        };
        let addr = c.addr;
        let mut out = Vec::new();
        for _ in 0..fragments {
            let seq = c.send_seq;
            c.send_seq = c.send_seq.wrapping_add(1);
            let mut p = vec![P_REL];
            put_u32(&mut p, seq);
            p.push(0);
            p.extend_from_slice(&[7u8; FRAG]);
            c.unacked.insert(seq, (p.clone(), now, false));
            out.push(p);
        }
        for p in out {
            self.send_raw(&p, addr);
        }
    }

    /// The session's topology (a client learns it from the host's WELCOME).
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// A dedicated server simulates every remote player's controller itself (NET-4): an object
    /// with a predictable controller owned by a player is predicted whatever its Network Object
    /// says, so a client can never move its pawn by announcing positions. Server and clients
    /// reach the same answer from the session topology and the owner.
    pub fn forces_prediction(&self, owner: u64) -> bool {
        self.topology == Topology::Dedicated && owner != 0 && owner != self.host_peer
    }

    fn forget_prediction(&mut self, id: u128) {
        self.anchors.remove(&id);
        self.anchor_wait.remove(&id);
        self.inputs.remove(&id);
        self.auth_in.remove(&id);
        self.predictors.remove(&id);
        self.delays.remove(&id);
    }

    /// Forget all replicated world state (Play stopped in the editor).
    pub fn reset_world(&mut self) {
        self.world = NetWorld::default();
        self.interest.reset();
        self.inputs.clear();
        self.auth_in.clear();
        self.predictors.clear();
        self.delays.clear();
        *self.history.lock().unwrap() = self.fresh_history();
        self.buffers.clear();
        self.anim_in.clear();
        self.avparam_in.clear();
        self.avparam_state.clear();
        self.avatar_in.clear();
        self.face_in.clear();
        self.send_state.clear();
        self.pending_vars.clear();
        self.pending_parents.clear();
        self.world_ops.clear();
        self.remote_bodies.clear();
        self.reclaimed_bodies.clear();
        self.host_player_done = false;
        self.players_to_spawn = if self.is_host { self.world_members() } else { Vec::new() };
    }

    fn push_event(&mut self, e: NetEvent) {
        if self.events.len() >= MAX_QUEUE {
            self.events.remove(0);
        }
        self.events.push(e);
    }

    /// Host: every member's peer id.
    fn members(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.conns.keys().copied().collect();
        v.sort();
        v
    }

    /// Host: the members that get world state: everyone but the clients the anti-cheat layer
    /// still holds (SEC-2 `require`).
    fn world_members(&self) -> Vec<u64> {
        let mut v = self.members();
        if !self.ac_deferred.is_empty() {
            v.retain(|p| !self.ac_deferred.contains_key(p));
        }
        v
    }

    // ------------------------------------------------------------ anti-cheat (SEC-2)

    /// Install an anti-cheat provider (a game's own, or a binding built into
    /// this program) in place of the one `NetConfig::anticheat` named.
    pub fn set_anticheat(&mut self, provider: Box<dyn AntiCheatProvider>) {
        self.ac = provider;
        let held: Vec<u64> = self.ac_deferred.keys().copied().collect();
        self.ac_peers.clear();
        for p in held {
            self.ac_release(p);
        }
        self.ac_started = false;
    }

    /// Name of the active provider.
    pub fn anticheat_name(&self) -> &str {
        self.ac.name()
    }

    /// Client: state of this machine's anti-cheat half.
    pub fn anticheat_client_status(&self) -> ClientStatus {
        self.ac.client_status()
    }

    /// Host: a client's verdict (`None` = not tracked: the layer is off or the peer is unknown).
    pub fn anticheat_verdict(&self, peer: u64) -> Option<Verdict> {
        let p = self.ac_peers.get(&peer)?;
        Some(if p.trusted { Verdict::Trusted } else { self.ac.server_verdict(peer) })
    }

    /// Host: clients still waiting for a Trusted verdict.
    pub fn anticheat_pending(&self) -> usize {
        self.ac_peers.values().filter(|p| !p.trusted).count()
    }

    fn ac_active(&self) -> bool {
        self.cfg.anticheat.enabled() || self.cfg.anticheat.require
    }

    /// Host: with `require`, a client that is not Trusted yet is held out of gameplay.
    fn ac_held(&self, peer: u64) -> bool {
        self.cfg.anticheat.require && self.ac_peers.get(&peer).is_some_and(|p| !p.trusted)
    }

    fn ac_register(&mut self, peer: u64, key: &[u8; 32]) {
        if !self.ac_active() || peer == self.local_peer {
            return;
        }
        let now = self.now();
        self.ac_peers.insert(peer, AcPeer { since: now, trusted: false });
        self.ac.server_register(peer, key);
    }

    /// Host: link a newly trusted client to the provider's account id. A banned account under a
    /// fresh player key is a ban evader: its new key is banned too and it is removed.
    fn ac_link(&mut self, peer: u64) {
        let Some(id) = self.ac.server_provider_id(peer) else { return };
        let key = crate::net_identity::provider_fingerprint(self.ac.name(), &id);
        let now_unix = crate::net_validate::unix_now();
        if let Some(b) = self.val.bans.is_banned(&key, now_unix) {
            let why = format!("banned account{}", if b.reason.is_empty() { String::new() } else { format!(": {}", b.reason) });
            let secs = b.until.saturating_sub(now_unix);
            if let Some(fp) = self.peer_fingerprint(peer) {
                tracing::warn!("net: peer {peer} ({}) is on a banned {} account: banning this player id too", short_fingerprint(&fp), self.ac.name());
                self.val.bans.add(&fp, if b.until == 0 { 0 } else { secs.max(1) }, &why, now_unix);
            }
            self.kick(peer, &format!("banned: {why}"));
            return;
        }
        self.ac_links.insert(peer, key);
    }

    /// Host: the ban-list key of the provider account linked to `peer` (none without a trusted
    /// verdict or a provider id).
    pub fn provider_ban_key(&self, peer: u64) -> Option<String> {
        self.ac_links.get(&peer).cloned()
    }

    /// Host: ban the provider account linked to `peer` along with its player id, so a new key
    /// on the same account cannot get back in. Returns whether an account was linked.
    pub fn ban_provider_link(&mut self, peer: u64, secs: u64, reason: &str) -> bool {
        let Some(key) = self.ac_links.get(&peer).cloned() else { return false };
        self.val.bans.add(&key, secs, reason, crate::net_validate::unix_now());
        true
    }

    /// Host: let a held client into the world (it was trusted, or the layer was swapped out).
    fn ac_release(&mut self, peer: u64) {
        let Some(rejoined) = self.ac_deferred.remove(&peer) else { return };
        self.send_catchup(peer, rejoined);
        if rejoined {
            self.send_backlog(peer);
        } else {
            self.players_to_spawn.push(peer);
        }
    }

    /// Host: clients held out of the world by `require` and not yet released (tests, stats).
    pub fn anticheat_held(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.ac_deferred.keys().copied().collect();
        v.sort();
        v
    }

    fn ac_unregister(&mut self, peer: u64) {
        self.ac_deferred.remove(&peer);
        self.ac_links.remove(&peer);
        if self.ac_peers.remove(&peer).is_some() {
            self.ac.server_unregister(peer);
        }
    }

    fn on_anticheat(&mut self, from: u64, payload: &[u8]) {
        if payload.len() > zestengine_core::ANTICHEAT_MAX_PAYLOAD {
            if self.is_host && from != self.local_peer {
                self.violation(from, Reason::TooLarge, "anti-cheat payload");
            }
            return;
        }
        if self.is_host {
            if self.ac_peers.contains_key(&from) {
                self.ac.server_receive(from, payload);
            }
        } else if from == self.host_peer && self.ac_active() {
            self.ac.client_receive(payload);
        }
    }

    fn anticheat_tick(&mut self, now: f64) {
        if !self.ac_active() {
            return;
        }
        self.ac.tick(now);
        if self.is_host {
            let mut out = Vec::new();
            self.ac.server_poll(&mut out);
            for (peer, payload) in out {
                if payload.len() <= zestengine_core::ANTICHEAT_MAX_PAYLOAD && self.ac_peers.contains_key(&peer) {
                    let mut m = vec![M_ANTICHEAT];
                    m.extend_from_slice(&payload);
                    self.send_reliable(peer, &m);
                }
            }
            let require = self.cfg.anticheat.require;
            let timeout = self.cfg.anticheat.pending_timeout as f64;
            let pending: Vec<(u64, AcPeer)> = self.ac_peers.iter().filter(|(_, p)| !p.trusted).map(|(k, v)| (*k, *v)).collect();
            for (peer, st) in pending {
                match self.ac.server_verdict(peer) {
                    Verdict::Trusted => {
                        if let Some(p) = self.ac_peers.get_mut(&peer) {
                            p.trusted = true;
                        }
                        // The client was held back from the world: bring it in now.
                        self.ac_release(peer);
                        self.ac_link(peer);
                    }
                    Verdict::Rejected(why) => {
                        self.stats.ac_rejected += 1;
                        if require {
                            self.ac_reject(peer, &why);
                        } else {
                            tracing::warn!("net: anti-cheat rejected peer {peer}: {why} (not required: not removed)");
                            self.ac_peers.remove(&peer);
                            self.ac.server_unregister(peer);
                        }
                    }
                    Verdict::Pending => {
                        if require && now - st.since > timeout {
                            self.ac_reject(peer, "anti-cheat did not answer in time");
                        }
                    }
                }
            }
        } else if self.state == ConnState::Connected {
            if !self.ac_started {
                self.ac_started = true;
                if let Err(e) = self.ac.client_start() {
                    tracing::warn!("net: anti-cheat client could not start: {e}");
                }
            }
            let mut out = Vec::new();
            self.ac.client_poll(&mut out);
            let host = self.host_peer;
            for payload in out {
                if payload.len() <= zestengine_core::ANTICHEAT_MAX_PAYLOAD {
                    let mut m = vec![M_ANTICHEAT];
                    m.extend_from_slice(&payload);
                    self.send_reliable(host, &m);
                }
            }
        }
    }

    /// Host: the provider refused a client (or never answered): strike it and remove it.
    fn ac_reject(&mut self, peer: u64, why: &str) {
        let now = self.now();
        let text = format!("anti-cheat: {why}");
        tracing::warn!("net: removing peer {peer}: {text}");
        // Counts in the strike system: a repeat offender (by player id) ends up banned.
        let esc = self.val.report(peer, Reason::AntiCheat, why, now);
        self.stats.violations += 1;
        if esc == Escalation::Ban
            && let Some(fp) = self.peer_fingerprint(peer)
        {
            let secs = self.val.settings().strikes.ban_secs;
            self.val.bans.add(&fp, secs, &text, crate::net_validate::unix_now());
            self.ban_provider_link(peer, secs, &text);
            self.kick(peer, &format!("banned: {text}"));
        } else {
            self.kick(peer, &text);
        }
    }

    // ------------------------------------------------------------ validation (host)

    /// Host: a client's data was refused. Counts it, adds strike points to its
    /// player id and acts on the escalation (warning, kick, ban).
    fn violation(&mut self, peer: u64, reason: Reason, detail: &str) {
        self.stats.violations += 1;
        let now = self.now();
        match self.val.report(peer, reason, detail, now) {
            Escalation::None => {}
            Escalation::Warn => {
                self.send_message(Some(peer), "server: warning, your client keeps sending data the server refuses; further violations get you removed");
            }
            Escalation::Kick => {
                let why = format!("too many violations ({})", reason.label());
                tracing::warn!("net: kicking peer {peer}: {why}");
                self.kick(peer, &why);
            }
            Escalation::Ban => {
                let why = format!("repeated violations ({})", reason.label());
                let secs = self.val.settings().strikes.ban_secs;
                if let Some(fp) = self.peer_fingerprint(peer) {
                    tracing::warn!("net: banning peer {peer} ({}) for {}: {why}", short_fingerprint(&fp), if secs == 0 { "good".to_string() } else { format!("{secs} s") });
                    self.val.bans.add(&fp, secs, &why, crate::net_validate::unix_now());
                }
                self.ban_provider_link(peer, secs, &why);
                self.kick(peer, &format!("banned: {why}"));
            }
        }
    }

    /// Host: take a token of `limit` for a client; a dry bucket is a violation.
    fn gate(&mut self, peer: u64, limit: Limit) -> bool {
        let now = self.now();
        if self.val.allow(peer, limit, now) {
            return true;
        }
        self.violation(peer, Reason::RateLimited, limit.label());
        false
    }

    /// Host: the peer is over the drop-level strike score and is being ignored.
    fn muted(&mut self, peer: u64) -> bool {
        let muted = self.is_host && peer != self.local_peer && (self.val.dropping(peer, self.now()) || self.ac_held(peer));
        if muted {
            self.stats.muted += 1;
        }
        muted
    }

    /// Host: common checks for an unreliable packet a client relays through
    /// us: it names its own id as the origin, is not muted and has tokens left.
    fn admit_unreliable(&mut self, peer: u64, origin: u64, limit: Limit) -> bool {
        if origin != peer {
            self.violation(peer, Reason::Forged, "sender id");
            return false;
        }
        self.gate(peer, limit)
    }

    /// Host: where the peer's avatar is (its player object's last known position).
    fn avatar_pos(&self, peer: u64) -> Option<Vec3> {
        let obj = self.world.player_obj.get(&peer)?;
        self.world.xforms.get(obj).map(|x| x.pos)
    }

    /// Host: VR pose check for an avatar packet; `false` = drop it.
    fn validate_pose(&mut self, peer: u64, id: u128, time: f64, poses: &[Option<(Vec3, Quat)>; TRACKER_SLOTS]) -> bool {
        match self.val.stamp(peer, time) {
            Stamp::Fresh => {}
            Stamp::Stale => return false,
            Stamp::Bad => {
                self.violation(peer, Reason::BadTime, "avatar time");
                return false;
            }
        }
        let now = self.now();
        let root = self.world.xforms.get(&id).map(|x| x.pos);
        let scale = match self.world.var(id, zestengine_core::AVATAR_SCALE_VAR) {
            Some(NetValue::Float(f)) => *f,
            _ => 1.0,
        };
        match self.val.avatar_pose(peer, id, root, scale, poses, now) {
            Ok(()) => true,
            Err(reason) => {
                self.violation(peer, reason, "avatar pose");
                false
            }
        }
    }

    fn wait_for_keys(&self) -> bool {
        matches!(self.clock, Clock::Manual(_))
    }

    /// The stretched session key for `p` (None while it is being computed).
    fn session_key_for(&mut self, p: &KdfParams) -> Option<[u8; 32]> {
        if let Some((_, k)) = self.keys.iter().find(|(q, _)| q == p) {
            return Some(*k);
        }
        if self.kdf_job.as_ref().is_none_or(|j| j.params() != *p) {
            self.kdf_job = Some(KdfJob::start(&self.cfg.secret, *p));
        }
        let wait = self.wait_for_keys();
        let k = self.kdf_job.as_ref()?.poll(wait)?;
        self.kdf_job = None;
        self.keys.push((*p, k));
        Some(k)
    }

    fn next_mid(&mut self) -> u32 {
        self.mid_out = self.mid_out.wrapping_add(1).max(1);
        let me = self.local_peer;
        self.seen_mid.insert(me, self.mid_out);
        self.mid_out
    }

    /// Whether a message id from `sender` is new (0 = no id, always new).
    fn fresh_mid(&mut self, sender: u64, mid: u32) -> bool {
        if mid == 0 {
            return true;
        }
        let seen = self.seen_mid.entry(sender).or_insert(0);
        if mid <= *seen {
            return false;
        }
        *seen = mid;
        true
    }

    // ------------------------------------------------------------ raw send

    /// A handshake datagram, in the clear.
    fn send_clear(&mut self, data: &[u8], to: SocketAddr) {
        self.stats.bytes_out += data.len() as u64;
        self.stats.packets_out += 1;
        self.transport.send(data, to);
    }

    /// A game packet, sealed for the connection at `to` (dropped when there
    /// is no secure channel yet; the reliable channel resends later).
    fn send_raw(&mut self, data: &[u8], to: SocketAddr) {
        let sealed = self
            .addr_peer
            .get(&to)
            .and_then(|p| self.conns.get_mut(p))
            .and_then(|c| c.sec.as_mut())
            .map(|sec| sec.seal(&[P_SECURE], data));
        if let Some(p) = sealed {
            self.interest_account(to, p.len());
            self.send_clear(&p, to);
        }
    }

    fn send_to_peer(&mut self, peer: u64, data: &[u8]) {
        if let Some(addr) = self.conns.get(&peer).map(|c| c.addr) {
            self.send_raw(data, addr);
        }
    }

    /// Unreliable send to everyone else (host: all members; client: the host).
    fn fanout(&mut self, data: &[u8], except: Option<u64>) {
        let targets: Vec<SocketAddr> = self.conns.values().filter(|c| Some(c.peer) != except && !self.ac_deferred.contains_key(&c.peer)).map(|c| c.addr).collect();
        for a in targets {
            self.send_raw(data, a);
        }
    }

    // ------------------------------------------------------------ handshake (client)

    fn start_handshake(&mut self, addrs: Vec<SocketAddr>, expect: Option<(u64, Option<[u8; 32]>)>) {
        let now = self.now();
        self.hs = Some(Handshake {
            addrs,
            addr: None,
            nonce: random_bytes(),
            started: now,
            last_hello: f64::NEG_INFINITY,
            expect,
            challenge: None,
            spare: Vec::new(),
            join: None,
            join_sent: 0.0,
            sec: None,
            hint: None,
        });
        self.send_hello();
    }

    /// HELLO (padded) until a CHALLENGE arrives, then the sealed JOIN.
    fn send_hello(&mut self) {
        let now = self.now();
        let Some(hs) = self.hs.as_mut() else {
            return;
        };
        hs.last_hello = now;
        let out: Vec<(Vec<u8>, SocketAddr)> = match (&hs.join, hs.addr) {
            (Some(j), Some(a)) => vec![(j.clone(), a)],
            _ => {
                let b = hello_packet(&hs.nonce, JOIN_MODE_JOIN);
                hs.addrs.iter().map(|a| (b.clone(), *a)).collect()
            }
        };
        for (b, a) in out {
            self.send_clear(&b, a);
        }
    }

    fn hint(&mut self, why: String) {
        let now = self.now();
        if let Some(hs) = self.hs.as_mut()
            && hs.challenge.is_none()
            && hs.hint.is_none()
        {
            tracing::debug!("net: handshake hint: {why}");
            hs.hint = Some((why, now));
        }
    }

    /// Client: a host's CHALLENGE (unauthenticated: its contents are bound
    /// into the connection key, so tampering only breaks the join).
    fn on_challenge(&mut self, data: &[u8], src: SocketAddr) {
        let Some(hs) = self.hs.as_ref() else {
            return;
        };
        if !hs.addrs.contains(&src) {
            return;
        }
        let Some(ch) = parse_challenge(data, &hs.nonce) else {
            self.stats.rejected += 1;
            return;
        };
        if let Some((peer, _)) = hs.expect
            && (ch.host_peer != peer || ch.epoch < self.epoch)
        {
            self.hint(format!("peer {} (epoch {}) answered instead of peer {peer} as the new host", ch.host_peer, ch.epoch));
            return;
        }
        if let Err(why) = ch.kdf.acceptable_for(&self.cfg.secret) {
            self.hint(why);
            return;
        }
        if let Some(hs) = self.hs.as_mut() {
            if hs.challenge.is_some() {
                let seen = hs.challenge.iter().chain(hs.spare.iter().map(|(c, _)| c)).any(|c| c.transcript == ch.transcript);
                if !seen {
                    if hs.spare.len() == MAX_SPARE_CHALLENGES {
                        hs.spare.remove(0);
                    }
                    hs.spare.push((ch, src));
                }
                return;
            }
            hs.addr = Some(src);
            hs.challenge = Some(ch);
        }
        self.try_send_join();
    }

    /// Seal the JOIN once the session key for the CHALLENGE's parameters is ready.
    fn try_send_join(&mut self) {
        let Some(ch) = self.hs.as_ref().filter(|h| h.join.is_none()).and_then(|h| h.challenge.clone()) else {
            return;
        };
        let Some(k) = self.session_key_for(&ch.kdf) else {
            return;
        };
        let Some(nonce) = self.hs.as_ref().map(|h| h.nonce) else {
            return;
        };
        let rejoin = self.local_peer;
        let mut sec = SecureChannel::new(&conn_key(&k, &ch.transcript), false);
        let mut header = vec![P_JOIN];
        header.extend_from_slice(&nonce);
        header.extend_from_slice(&ch.cookie);
        header.push(JOIN_MODE_JOIN);
        let mut plain = Vec::new();
        put_u16(&mut plain, PROTOCOL);
        plain.push(self.cfg.can_host as u8);
        put_str(&mut plain, &self.cfg.player_name);
        plain.extend_from_slice(&self.cfg.identity.public());
        plain.extend_from_slice(&self.cfg.identity.sign(&join_message(&ch.transcript, rejoin)));
        put_u64(&mut plain, rejoin);
        put_u32(&mut plain, self.epoch);
        put_str(&mut plain, &self.cfg.account_name);
        put_str(&mut plain, &self.cfg.account_password);
        let join = sec.seal_at(&header, 0, &plain);
        sec.skip_to(1);
        self.kdf = ch.kdf;
        let now = self.now();
        if let Some(hs) = self.hs.as_mut() {
            hs.sec = Some(sec);
            hs.join = Some(join);
            hs.join_sent = now;
        }
        self.send_hello();
    }

    /// Client: a clear REJECT. Signed ones (the host's identity from its
    /// CHALLENGE) end the join; anything else is only a hint, believed if
    /// no CHALLENGE turns up within [`REJECT_GRACE`].
    fn on_clear_reject(&mut self, data: &[u8], src: SocketAddr) {
        let Some(hs) = self.hs.as_ref() else {
            return;
        };
        if !hs.addrs.contains(&src) {
            return;
        }
        let mut c = Cursor::new(data, 1);
        match c.u8() {
            REJECT_SIGNED => {
                let code = c.u8();
                let reason = c.string();
                let sig: [u8; 64] = c.arr();
                let valid = !c.bad
                    && hs.challenge.as_ref().is_some_and(|ch| {
                        crate::net_identity::verify(&ch.host_pk, &reject_message(&hs.nonce, &ch.cookie, code, &reason), &sig)
                    });
                if valid {
                    self.fail_handshake(&format!("rejected: {reason}"));
                } else {
                    self.stats.rejected += 1;
                }
            }
            REJECT_HINT => {
                let proto = c.u16();
                let reason = c.string();
                if !c.bad {
                    self.hint(if proto != PROTOCOL {
                        format!("the host speaks network protocol {proto}, this build speaks {PROTOCOL}: update both to the same version")
                    } else {
                        format!("rejected: {reason}")
                    });
                }
            }
            _ => {
                // Older hosts send a bare reason.
                let reason = Cursor::new(data, 1).string();
                self.hint(format!("the host runs an older network protocol ({reason}); this build speaks {PROTOCOL}"));
            }
        }
    }

    /// The handshake failed for good (an authenticated refusal).
    fn fail_handshake(&mut self, why: &str) {
        self.hs = None;
        if self.migration.is_some() {
            self.migration = None;
        }
        self.disconnect_local(why);
    }

    /// Client: an authenticated packet from the host we are handshaking with.
    fn on_handshake_plain(&mut self, data: &[u8]) {
        match data.first() {
            Some(&P_WELCOME) => self.on_welcome(data),
            Some(&P_REJECT) => {
                let reason = Cursor::new(data, 1).string();
                self.fail_handshake(&format!("rejected: {reason}"));
            }
            _ => {}
        }
    }

    fn on_welcome(&mut self, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let id = c.u64();
        let max = c.u32();
        let name = c.string();
        let host_peer = c.u64();
        let epoch = c.u32();
        let clock_offset = c.f64();
        let flags = c.u8();
        let sig: [u8; 64] = c.arr();
        let Some(mut hs) = self.hs.take() else {
            return;
        };
        let (Some(ch), Some(addr)) = (hs.challenge.clone(), hs.addr) else {
            self.hs = Some(hs);
            return;
        };
        if c.bad || host_peer != ch.host_peer || !crate::net_identity::verify(&ch.host_pk, &welcome_message(&ch.transcript, id), &sig) {
            self.stats.rejected += 1;
            self.hs = Some(hs);
            return;
        }
        if let Some((_, Some(known))) = hs.expect
            && known != ch.host_pk
        {
            self.fail_handshake(&format!("peer {host_peer}'s identity doesn't match the one the session knows (refusing an impostor host)"));
            return;
        }
        // A dedicated server's key is always pinned (trust on first use), and so is
        // any host this install pinned before: an impostor cannot dodge the check
        // by claiming to be a listen host.
        let topology = Topology::from_u8((flags >> 1) & 1);
        if self.migration.is_none()
            && let Some(target) = self.pin_target.clone()
            && (self.cfg.pin_host || topology == Topology::Dedicated || crate::net_identity::is_pinned(&target))
            && let Err(why) = crate::net_identity::check_pin(&target, &ch.host_pk)
        {
            // The host already admitted us: tell it we are leaving rather than let it wait for a timeout.
            if let Some(sec) = hs.sec.as_mut() {
                let bye = sec.seal(&[P_SECURE], &[P_BYE, BYE_LEAVE]);
                self.send_clear(&bye, addr);
            }
            self.fail_handshake(&why);
            return;
        }
        self.topology = topology;
        let now = self.now();
        let mut conn = Conn::new(addr, host_peer, now);
        conn.sec = hs.sec.take();
        conn.transcript = ch.transcript;
        conn.identity = ch.host_pk;
        self.conns.insert(host_peer, conn);
        self.addr_peer.insert(addr, host_peer);
        self.peer_ids.insert(host_peer, ch.host_pk);
        self.host_clock_offset = clock_offset;
        self.cfg.max_players = max;
        self.cfg.session_name = name;
        self.epoch = epoch;
        match self.migration.take() {
            Some(m) => {
                let previous = m.previous;
                self.host_peer = host_peer;
                // The aliases the old host taught us mean nothing to the new one.
                self.interest.reset();
                if id != self.local_peer {
                    // The new host didn't know us: we joined fresh.
                    self.peers.remove(&self.local_peer);
                    self.local_peer = id;
                    self.peers.insert(id, self.cfg.player_name.clone());
                }
                self.migrations += 1;
                self.push_event(NetEvent::HostChanged { host: host_peer, previous });
                tracing::info!("net: peer {host_peer} is the new host (epoch {epoch})");
                // Our reliable messages the old host may never have handled.
                let out = std::mem::take(&mut self.outbox);
                for m in out {
                    self.send_reliable(host_peer, &m);
                }
            }
            None => {
                self.local_peer = id;
                self.host_peer = host_peer;
                self.state = ConnState::Connected;
                self.peers.insert(id, self.cfg.player_name.clone());
                self.push_event(NetEvent::Connected { peer: id });
                tracing::info!("net: joined as peer {id}");
            }
        }
        // Confirm the key right away (the host admits us on this).
        let mut b = vec![P_PING];
        put_f64(&mut b, now);
        put_f32(&mut b, 0.0);
        put_f32(&mut b, self.view_delay);
        self.send_raw(&b, addr);
    }

    /// Client: handshake upkeep each pump (key stretching, HELLO resends,
    /// relay join proofs, timeouts and hints).
    fn handshake_tick(&mut self, now: f64) {
        if let Some(kdf) = self.transport.relay_needs_join_key()
            && let Some(p) = KdfParams::decode(&kdf)
        {
            match p.acceptable_for(&self.cfg.secret) {
                Ok(()) => {
                    if let Some(k) = self.session_key_for(&p) {
                        self.transport.set_relay_join_key(relay_join_key(&k));
                    }
                }
                Err(why) => {
                    if self.migration.is_none() && self.state == ConnState::Connecting {
                        self.disconnect_local(&format!("relay: {why}"));
                        return;
                    }
                }
            }
        }
        if let Some(hs) = self.hs.as_mut()
            && hs.join.is_some()
            && !hs.spare.is_empty()
            && now - hs.join_sent > JOIN_PATIENCE
        {
            let (ch, addr) = hs.spare.remove(0);
            tracing::debug!("net: no WELCOME for the chosen CHALLENGE; trying another answer");
            hs.challenge = Some(ch);
            hs.addr = Some(addr);
            hs.join = None;
            hs.sec = None;
        }
        let Some(hs) = self.hs.as_ref() else {
            return;
        };
        if hs.challenge.is_some() && hs.join.is_none() {
            self.try_send_join();
        }
        let Some(hs) = self.hs.as_ref() else {
            return;
        };
        let hinted = hs.challenge.is_none() && hs.hint.as_ref().is_some_and(|(_, t)| now - t > REJECT_GRACE);
        let resend = now - hs.last_hello > HELLO_INTERVAL;
        if self.migration.is_some() {
            if resend {
                self.send_hello();
            }
            return;
        }
        // The first join.
        let relay_failed = match self.transport.relay_status().map(|r| r.phase) {
            Some(RelayPhase::Failed(why)) => Some(why),
            _ => None,
        };
        let hint = hs.hint.as_ref().map(|(h, _)| h.clone());
        let timed_out = now - hs.started > self.cfg.timeout as f64;
        if let Some(why) = relay_failed {
            self.hs = None;
            self.disconnect_local(&format!("relay: {why}"));
        } else if hinted {
            self.hs = None;
            self.disconnect_local(&hint.unwrap_or_default());
        } else if timed_out {
            self.hs = None;
            self.disconnect_local(&hint.unwrap_or_else(|| "could not reach the host".into()));
        } else if resend {
            self.send_hello();
        }
    }

    // ------------------------------------------------------------ reliable channel

    fn send_reliable(&mut self, peer: u64, msg: &[u8]) {
        let now = self.now();
        let Some(c) = self.conns.get_mut(&peer) else {
            // Between hosts: keep it for the next one.
            if !self.is_host && self.migration.is_some() && peer == self.host_peer && self.outbox.len() < MAX_QUEUE {
                self.outbox.push(msg.to_vec());
            }
            return;
        };
        let addr = c.addr;
        let mut out = Vec::new();
        let n = msg.len().div_ceil(FRAG).max(1);
        for k in 0..n {
            let chunk = &msg[(k * FRAG).min(msg.len())..((k + 1) * FRAG).min(msg.len())];
            let seq = c.send_seq;
            c.send_seq = c.send_seq.wrapping_add(1);
            let mut p = Vec::with_capacity(chunk.len() + 6);
            p.push(P_REL);
            put_u32(&mut p, seq);
            p.push((k + 1 == n) as u8);
            p.extend_from_slice(chunk);
            c.unacked.insert(seq, (p.clone(), now, k == 0));
            out.push(p);
        }
        for p in out {
            self.send_raw(&p, addr);
        }
    }

    /// Reliable send to every member (host only), optionally skipping one.
    fn broadcast_reliable(&mut self, msg: &[u8], except: Option<u64>) {
        let interest = self.interest_on();
        let peers: Vec<u64> = self.world_members().into_iter().filter(|p| Some(*p) != except).collect();
        for p in peers {
            if interest && !self.interest_wants(p, msg) {
                continue;
            }
            self.send_reliable(p, msg);
        }
    }

    /// Host: to all members; client: to the host.
    fn upstream_or_broadcast(&mut self, msg: &[u8]) {
        if self.is_host {
            self.broadcast_reliable(msg, None);
        } else {
            let host = self.host_peer;
            self.send_reliable(host, msg);
        }
    }

    fn on_rel(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let seq = c.u32();
        let last = c.u8() != 0;
        if c.bad {
            return;
        }
        let payload = c.rest().to_vec();
        let mut ready = Vec::new();
        // A client cannot make the host buffer without bound: reorder window
        // and message size are capped per connection.
        let from_client = self.is_host && peer != self.local_peer;
        let (max_reorder, max_message) = (self.val.max_reorder(), self.val.max_message());
        let mut fault = None;
        {
            let Some(conn) = self.conns.get_mut(&peer) else {
                return;
            };
            conn.acks.push(seq);
            let ahead = seq.wrapping_sub(conn.recv_next);
            if ahead >= 1 << 20 || conn.recv_buf.contains_key(&seq) {
                return; // duplicate or ancient
            }
            if from_client && conn.recv_buf.len() >= max_reorder {
                fault = Some("reliable reorder buffer");
            } else {
                conn.recv_buf.insert(seq, (last, payload));
                while let Some((last, p)) = conn.recv_buf.remove(&conn.recv_next) {
                    conn.recv_next = conn.recv_next.wrapping_add(1);
                    conn.frag.extend_from_slice(&p);
                    if from_client && conn.frag.len() > max_message {
                        conn.frag = Vec::new();
                        fault = Some("reliable message size");
                    } else if last {
                        ready.push(std::mem::take(&mut conn.frag));
                    }
                }
            }
        }
        if let Some(what) = fault {
            self.violation(peer, Reason::TooLarge, what);
        }
        for m in ready {
            if !self.conns.contains_key(&peer) {
                break;
            }
            self.on_msg(peer, &m);
        }
    }

    fn flush_acks(&mut self) {
        let mut out = Vec::new();
        for c in self.conns.values_mut() {
            if c.acks.is_empty() {
                continue;
            }
            for chunk in c.acks.chunks((MTU - 4) / 4) {
                let mut b = vec![P_ACK];
                put_u16(&mut b, chunk.len() as u16);
                for s in chunk {
                    put_u32(&mut b, *s);
                }
                out.push((b, c.addr));
            }
            c.acks.clear();
        }
        for (b, a) in out {
            self.send_raw(&b, a);
        }
    }

    // ------------------------------------------------------------ pump

    /// Receive + process all pending packets, resend, ping and time out. Call
    /// once per frame (also while not playing, to keep connections alive).
    pub fn pump(&mut self) {
        let t = self.now();
        self.transport.poll(t);
        let mut buf = [0u8; RECV_BUF];
        while let Some((len, src)) = self.transport.recv(&mut buf) {
            let data = buf[..len].to_vec();
            self.stats.bytes_in += len as u64;
            self.stats.packets_in += 1;
            self.handle(&data, src);
        }
        self.serve_discovery();
        let now = self.now();
        self.handshake_tick(now);
        if self.is_host {
            self.host_tick(now);
            self.interest_tick(now);
        }
        self.anticheat_tick(now);
        self.migration_tick(now);
        // Resends, pings, timeouts.
        let mut resend = Vec::new();
        let mut ping = Vec::new();
        let mut dead = Vec::new();
        let connected = self.state == ConnState::Connected;
        for c in self.conns.values_mut() {
            let rto = c.rto();
            for (p, sent, _) in c.unacked.values_mut() {
                if now - *sent > rto {
                    *sent = now;
                    resend.push((p.clone(), c.addr));
                }
            }
            if connected && now - c.last_ping > PING_INTERVAL {
                c.last_ping = now;
                ping.push(c.addr);
            }
            if now - c.last_recv > self.cfg.timeout as f64 {
                dead.push(c.peer);
            }
        }
        self.stats.resends += resend.len() as u64;
        for (p, a) in resend {
            self.send_raw(&p, a);
        }
        let view_delay = self.view_delay;
        for a in ping {
            let loss = self.addr_peer.get(&a).and_then(|p| self.conns.get(p)).and_then(|c| c.sec.as_ref()).map_or(0.0, |s| s.loss.loss);
            let mut b = vec![P_PING];
            put_f64(&mut b, now);
            put_f32(&mut b, loss);
            put_f32(&mut b, view_delay);
            self.send_raw(&b, a);
        }
        for p in dead {
            if self.is_host {
                self.drop_peer(p, "timed out");
            } else if p == self.host_peer && self.state == ConnState::Connected {
                self.host_lost(None, "connection to the host timed out");
            }
        }
        self.flush_acks();
        let s = &mut self.stats;
        if now - s.window_start >= 1.0 {
            let span = (now - s.window_start) as f32;
            s.rate_in = (s.bytes_in - s.window_in) as f32 / span;
            s.rate_out = (s.bytes_out - s.window_out) as f32 / span;
            s.window_in = s.bytes_in;
            s.window_out = s.bytes_out;
            s.window_start = now;
        }
        self.pending_vars.retain(|p| p.expires > now);
        self.pending_parents.retain(|p| p.2 > now);
        self.update_clock(now);
    }

    fn serve_discovery(&mut self) {
        let Some(d) = self.discovery.as_ref() else {
            return;
        };
        let mut buf = [0u8; 64];
        let game_port = self.transport.local_addr().map(|a| a.port()).unwrap_or(0);
        while let Ok((n, src)) = d.recv_from(&mut buf) {
            if n < 1 || buf[0] != P_DISCOVER {
                continue;
            }
            let mut b = vec![P_INFO];
            put_u16(&mut b, PROTOCOL);
            put_u16(&mut b, game_port);
            put_u32(&mut b, self.peers.len() as u32);
            put_u32(&mut b, self.cfg.max_players);
            put_str(&mut b, &self.cfg.session_name);
            b.push(self.cfg.topology.to_u8());
            let _ = d.send_to(&b, src);
        }
    }

    fn disconnect_local(&mut self, reason: &str) {
        if matches!(self.state, ConnState::Disconnected(_)) {
            return;
        }
        tracing::info!("net: disconnected ({reason})");
        self.hs = None;
        self.migration = None;
        self.state = ConnState::Disconnected(reason.to_owned());
        self.push_event(NetEvent::Disconnected { reason: reason.to_owned() });
    }

    /// Leave the session. A host with members hands it to its successor
    /// (with the latest state) when migration is on; otherwise, and for
    /// clients, the others are told the peer left.
    pub fn disconnect(&mut self) {
        if self.is_host && self.cfg.migration && !self.members().is_empty() {
            self.hand_off();
        } else {
            let addrs: Vec<SocketAddr> = self.conns.values().map(|c| c.addr).collect();
            for a in addrs {
                for _ in 0..3 {
                    self.send_raw(&[P_BYE, BYE_LEAVE], a);
                }
            }
        }
        self.conns.clear();
        self.addr_peer.clear();
        self.close_transport();
        self.disconnect_local("disconnected");
    }

    /// Host: end the session for everyone (no migration).
    pub fn close_session(&mut self) {
        self.close_session_with("");
    }

    /// [`close_session`](Self::close_session) with a reason the members see
    /// ("the host closed the session: <reason>"), e.g. a dedicated server's
    /// shutdown notice. An empty reason sends the plain BYE older builds know.
    pub fn close_session_with(&mut self, reason: &str) {
        let mut bye = vec![P_BYE, BYE_CLOSED];
        if !reason.is_empty() {
            put_str(&mut bye, reason);
        }
        let addrs: Vec<SocketAddr> = self.conns.values().map(|c| c.addr).collect();
        for a in addrs {
            for _ in 0..3 {
                self.send_raw(&bye, a);
            }
        }
        self.conns.clear();
        self.addr_peer.clear();
        self.close_transport();
        self.disconnect_local("closed the session");
    }

    fn close_transport(&mut self) {
        if !self.closed {
            self.closed = true;
            self.transport.close();
        }
    }

    // ------------------------------------------------------------ packet dispatch

    fn handle(&mut self, data: &[u8], src: SocketAddr) {
        let Some(&tag) = data.first() else {
            return;
        };
        let host = self.is_host;
        match tag {
            P_SECURE => {
                let now = self.now();
                if let Some(&peer) = self.addr_peer.get(&src) {
                    match self.conns.get_mut(&peer).and_then(|c| c.sec.as_mut()).and_then(|s| s.open(data, 1, now)) {
                        Some(plain) => self.handle_plain(&plain, src, peer),
                        None => self.stats.rejected += 1,
                    }
                    return;
                }
                if self.hs.as_ref().is_some_and(|h| h.addr == Some(src)) {
                    match self.hs.as_mut().and_then(|h| h.sec.as_mut()).and_then(|s| s.open(data, 1, now)) {
                        Some(plain) => self.on_handshake_plain(&plain),
                        None => self.stats.rejected += 1,
                    }
                    return;
                }
                if let Some(i) = self.probes.iter().position(|p| p.addr == src) {
                    match self.probes[i].sec.as_mut().and_then(|s| s.open(data, 1, now)) {
                        Some(plain) => self.on_probe_ack(i, &plain),
                        None => self.stats.rejected += 1,
                    }
                    return;
                }
                self.stats.rejected += 1;
            }
            P_HELLO if host => self.on_hello(data, src),
            P_JOIN if host => self.on_join(data, src),
            // Members trying us as the next host before we took over: they retry.
            P_HELLO | P_JOIN => {}
            P_CHALLENGE if self.probes.iter().any(|p| p.addr == src) => self.on_probe_challenge(data, src),
            P_CHALLENGE if self.hs.is_some() => self.on_challenge(data, src),
            P_REJECT if self.hs.is_some() => self.on_clear_reject(data, src),
            _ => self.stats.rejected += 1,
        }
    }

    /// An authenticated packet from `peer`.
    fn handle_plain(&mut self, data: &[u8], src: SocketAddr, peer: u64) {
        let Some(&tag) = data.first() else {
            return;
        };
        let now = self.now();
        if let Some(c) = self.conns.get_mut(&peer) {
            c.last_recv = now;
            c.confirmed = true;
        }
        // Host: a client's packet is charged to its all-packets bucket before
        // anything is parsed, so a flood costs one bucket update each.
        if self.is_host && peer != self.local_peer {
            if data.len() > MTU + 16 {
                self.violation(peer, Reason::TooLarge, "oversized packet");
                return;
            }
            if !self.val.allow(peer, Limit::Packets, now) {
                self.violation(peer, Reason::RateLimited, "packets");
                return;
            }
            if self.muted(peer) && !matches!(tag, P_PING | P_PONG | P_BYE | P_REL | P_ACK) {
                return;
            }
        }
        let mut c = Cursor::new(data, 1);
        match tag {
            P_WELCOME => {}
            P_PING => {
                let t = c.f64();
                let their_loss = c.f32();
                let view_delay = c.f32();
                let rewind_cap = self.val.settings().max_rewind;
                let mut b = vec![P_PONG];
                put_f64(&mut b, t);
                if let Some(conn) = self.conns.get_mut(&peer) {
                    if !c.bad {
                        conn.remote_loss = if their_loss.is_finite() { their_loss.clamp(0.0, 1.0) } else { 0.0 };
                        // The rewind window caps the reported delay (a NaN or huge one counts as none).
                        conn.view_delay = if view_delay.is_finite() { view_delay.clamp(0.0, rewind_cap) } else { 0.0 };
                    }
                    put_f32(&mut b, conn.sec.as_ref().map_or(0.0, |s| s.loss.loss));
                }
                self.send_raw(&b, src);
            }
            P_PONG => {
                let t = c.f64();
                let their_loss = c.f32();
                if let Some(conn) = self.conns.get_mut(&peer)
                    && !c.bad
                    && t <= now
                {
                    let sample = (now - t) as f32;
                    conn.rtt_var = conn.rtt_var * 0.75 + (sample - conn.rtt).abs() * 0.25;
                    conn.rtt = conn.rtt * 0.8 + sample * 0.2;
                    conn.remote_loss = if their_loss.is_finite() { their_loss.clamp(0.0, 1.0) } else { 0.0 };
                }
            }
            P_BYE => self.on_bye(peer, data),
            P_REL => self.on_rel(peer, data),
            P_ACK => {
                let n = c.u16() as usize;
                if let Some(conn) = self.conns.get_mut(&peer) {
                    for _ in 0..n {
                        let s = c.u32();
                        if c.bad {
                            break;
                        }
                        conn.unacked.remove(&s);
                    }
                }
            }
            P_SNAP => self.on_snap(peer, data),
            P_SNAPB if !self.is_host && peer == self.host_peer => self.on_snapb(data),
            P_SNAPB if self.is_host && peer != self.local_peer => self.on_snap_up(peer, data),
            P_AVATAR => self.on_avatar(peer, data),
            P_FACE => self.on_face(peer, data),
            P_ANIM => self.on_anim(peer, data),
            P_AVPARAM => self.on_avparam(peer, data),
            P_VOICE => {
                let from = c.u64();
                let seq = c.u32();
                let n = c.u16() as usize;
                let at = 1 + 8 + 4 + 2;
                let Some(payload) = data.get(at..at + n) else {
                    return;
                };
                if c.bad {
                    return;
                }
                if self.is_host && peer != self.local_peer {
                    // The host relays voice as sent: the sender id must be the sender's own.
                    if from != peer {
                        self.violation(peer, Reason::Forged, "voice sender id");
                        return;
                    }
                    if !self.gate(peer, Limit::Voice) {
                        return;
                    }
                    if !self.val.voice_ok(payload.len()) {
                        self.violation(peer, Reason::TooLarge, "voice packet");
                        return;
                    }
                }
                if from != self.local_peer {
                    self.voice_in.push(VoicePacket { from, seq, payload: payload.to_vec() });
                    if self.voice_in.len() > 512 {
                        self.voice_in.remove(0);
                    }
                }
                if self.is_host {
                    self.fanout_voice(data, Some(peer), from);
                }
            }
            P_URPC => {
                let body = &data[1..];
                self.on_rpc(peer, body, false);
            }
            P_INPUT if self.is_host => self.on_input(peer, data),
            P_PSTATE if !self.is_host && peer == self.host_peer => self.on_pstate(data),
            _ => {}
        }
    }

    /// A sealed BYE: a member leaving (host side), or the host leaving,
    /// handing off, closing or kicking us (client side).
    fn on_bye(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let kind = if data.len() > 1 { c.u8() } else { BYE_CLOSED };
        if self.is_host {
            self.drop_peer(peer, "left");
            return;
        }
        if peer != self.host_peer {
            return;
        }
        match kind {
            BYE_HANDOFF => {
                let to = c.u64();
                let n = c.u8() as usize;
                let mut addrs = Vec::new();
                let mut at = 1 + 1 + 8 + 1;
                for _ in 0..n.min(4) {
                    match relay_proto::read_addr(data, &mut at) {
                        Some(a) => addrs.push(a),
                        None => break,
                    }
                }
                if c.bad {
                    return;
                }
                self.host_lost(Some((to, addrs)), "the host left");
            }
            BYE_KICKED => {
                let reason = c.string();
                self.conns.clear();
                self.addr_peer.clear();
                self.disconnect_local(&if reason.is_empty() { "kicked by the host".to_string() } else { format!("kicked by the host: {reason}") });
            }
            BYE_LEAVE => self.host_lost(None, "the host left"),
            BYE_CLOSED => {
                let reason = if data.len() > 2 { c.string() } else { String::new() };
                self.conns.clear();
                self.addr_peer.clear();
                self.disconnect_local(&if reason.is_empty() || c.bad { "the host closed the session".to_string() } else { format!("the host closed the session: {reason}") });
            }
            _ => self.disconnect_local("the host closed the session"),
        }
    }

    // ------------------------------------------------------------ handshake (host)

    fn cookie(&self, src: SocketAddr, nonce: &[u8; 16], bucket: u64) -> [u8; 16] {
        let mut addr = Vec::with_capacity(20);
        relay_proto::put_addr(&mut addr, src);
        mac16(&self.cookie_key, &[b"zestengine-net cookie v6", &addr, nonce, &bucket.to_le_bytes()])
    }

    fn cookie_valid(&self, src: SocketAddr, nonce: &[u8; 16], cookie: &[u8; 16]) -> bool {
        let bucket = (self.now() / COOKIE_PERIOD) as u64;
        [bucket, bucket.saturating_sub(1)].iter().any(|b| relay_proto::tags_equal(&self.cookie(src, nonce, *b), cookie))
    }

    /// The CHALLENGE bytes for `cookie` (deterministic, so the host can
    /// rebuild the transcript from a JOIN without having kept anything).
    fn challenge_bytes(&self, cookie: &[u8; 16]) -> Vec<u8> {
        let mut b = vec![P_CHALLENGE];
        put_u16(&mut b, PROTOCOL);
        self.kdf.encode(&mut b);
        b.extend_from_slice(&self.cfg.identity.public());
        put_u32(&mut b, self.epoch);
        put_u64(&mut b, self.local_peer);
        b.extend_from_slice(cookie);
        b
    }

    fn host_key(&mut self) -> Option<[u8; 32]> {
        let p = self.kdf;
        self.session_key_for(&p)
    }

    /// Host: answer a (padded) HELLO with a CHALLENGE no larger than it. No
    /// state is kept: the cookie lets the JOIN prove it came back from the
    /// same endpoint.
    fn on_hello(&mut self, data: &[u8], src: SocketAddr) {
        let proto = Cursor::new(data, 1).u16();
        if data.len() < 3 {
            return;
        }
        if proto != PROTOCOL {
            let why = format!("protocol version mismatch (host {PROTOCOL}, you {proto})");
            if data.len() < HELLO_SIZE {
                // An older client: its own reject format, rate-limited
                // (the only pre-auth reply that can exceed its request).
                let now = self.now();
                if self.legacy_bucket.take(now) {
                    let mut b = vec![P_REJECT];
                    put_str(&mut b, &why);
                    self.send_clear(&b, src);
                }
            } else {
                let mut b = vec![P_REJECT, REJECT_HINT];
                put_u16(&mut b, PROTOCOL);
                put_str(&mut b, &why);
                b.truncate(data.len());
                self.send_clear(&b, src);
            }
            return;
        }
        if data.len() < HELLO_SIZE {
            self.stats.rejected += 1;
            return;
        }
        let Some(nonce) = data.get(3..19).and_then(|n| <[u8; 16]>::try_from(n).ok()) else {
            return;
        };
        let bucket = (self.now() / COOKIE_PERIOD) as u64;
        let cookie = self.cookie(src, &nonce, bucket);
        let ch = self.challenge_bytes(&cookie);
        debug_assert!(ch.len() <= HELLO_SIZE);
        self.send_clear(&ch, src);
    }

    /// A REJECT for a JOIN the host couldn't open, signed with its identity
    /// (the client checks it against the CHALLENGE's key).
    fn signed_reject(&mut self, src: SocketAddr, nonce: &[u8; 16], cookie: &[u8; 16], code: u8, reason: &str, limit: usize) {
        let sig = self.cfg.identity.sign(&reject_message(nonce, cookie, code, reason));
        let mut b = vec![P_REJECT, REJECT_SIGNED, code];
        put_str(&mut b, reason);
        b.extend_from_slice(&sig);
        if b.len() <= limit {
            self.send_clear(&b, src);
        }
    }

    /// Host: a sealed JOIN. Cheap checks first (cookie, half-open cap, rate),
    /// then opening it proves the secret and the signature the identity.
    fn on_join(&mut self, data: &[u8], src: SocketAddr) {
        let now = self.now();
        let (Some(nonce), Some(cookie), Some(&mode)) = (
            data.get(1..17).and_then(|n| <[u8; 16]>::try_from(n).ok()),
            data.get(17..33).and_then(|n| <[u8; 16]>::try_from(n).ok()),
            data.get(33),
        ) else {
            return;
        };
        if !self.cookie_valid(src, &nonce, &cookie) {
            self.stats.rejected += 1;
            return;
        }
        let Some(k) = self.host_key() else {
            return; // still stretching the key: the client resends
        };
        let transcript = hash(&[b"zestengine-net handshake v6", &nonce, &self.challenge_bytes(&cookie)]);
        if mode == JOIN_MODE_PROBE {
            if self.join_bucket.take(now) {
                self.on_probe(data, src, &k, &transcript);
            }
            return;
        }
        if let Some(&peer) = self.addr_peer.get(&src) {
            if self.conns.get(&peer).is_some_and(|c| c.client_nonce == nonce) {
                // Duplicate JOIN: our WELCOME was lost.
                if let Some(c) = self.conns.get_mut(&peer) {
                    c.last_recv = now;
                }
                self.send_welcome(peer);
                return;
            }
            // A new session from the same address: the old one is gone.
            self.drop_peer(peer, "rejoined");
        }
        if self.seen_nonces.contains_key(&nonce) {
            self.stats.rejected += 1;
            return;
        }
        if self.half_open() >= MAX_HALF_OPEN {
            self.stats.half_open_refused += 1;
            return;
        }
        if !self.join_bucket.take(now) {
            self.stats.half_open_refused += 1;
            return;
        }
        let mut sec = SecureChannel::new(&conn_key(&k, &transcript), true);
        let Some(plain) = sec.open(data, JOIN_HEADER, now) else {
            self.stats.rejected += 1;
            self.signed_reject(src, &nonce, &cookie, 1, "wrong session secret", data.len());
            return;
        };
        let sealed_reject = |s: &mut Self, sec: &mut SecureChannel, why: &str| {
            let mut b = vec![P_REJECT];
            put_str(&mut b, why);
            let p = sec.seal(&[P_SECURE], &b);
            s.send_clear(&p, src);
        };
        let mut c = Cursor::new(&plain, 0);
        let proto = c.u16();
        let can_host = c.u8() != 0;
        let name = c.string();
        let pk: [u8; 32] = c.arr();
        let sig: [u8; 64] = c.arr();
        let rejoin = c.u64();
        let _their_epoch = c.u32();
        let account_name = if c.p < plain.len() { c.string() } else { String::new() };
        let account_password = if c.p < plain.len() { c.string() } else { String::new() };
        if c.bad || proto != PROTOCOL {
            sealed_reject(self, &mut sec, &format!("protocol version mismatch (host {PROTOCOL}, you {proto})"));
            return;
        }
        if !crate::net_identity::verify(&pk, &join_message(&transcript, rejoin), &sig) {
            self.stats.rejected += 1;
            sealed_reject(self, &mut sec, "invalid identity signature");
            return;
        }
        let fp = fingerprint(&pk);
        if self.cfg.ban.iter().any(|b| crate::net_identity::fingerprint_matches(b, &fp)) {
            sealed_reject(self, &mut sec, "banned from this session");
            tracing::info!("net: refused banned player {} from {src}", short_fingerprint(&fp));
            return;
        }
        if let Some(b) = self.val.bans.is_banned(&fp, crate::net_validate::unix_now()) {
            let why = if b.reason.is_empty() { "banned from this server".to_string() } else { format!("banned from this server: {}", b.reason) };
            sealed_reject(self, &mut sec, &why);
            tracing::info!("net: refused banned player {} from {src}", short_fingerprint(&fp));
            return;
        }
        if !self.cfg.allow.is_empty() && !self.cfg.allow.iter().any(|a| crate::net_identity::fingerprint_matches(a, &fp)) {
            sealed_reject(self, &mut sec, "not on this session's allow list");
            return;
        }
        if !self.cfg.accounts.is_empty() {
            if account_name.trim().is_empty() {
                sealed_reject(self, &mut sec, "account login required");
                return;
            }
            let Some(acc) = self.cfg.accounts.iter().find(|a| a.name == account_name) else {
                sealed_reject(self, &mut sec, "unknown account");
                return;
            };
            if acc.password != account_password {
                sealed_reject(self, &mut sec, "wrong account password");
                return;
            }
            if self.conns.values().any(|c| c.account == account_name && c.confirmed) {
                sealed_reject(self, &mut sec, "account already in use");
                return;
            }
        }
        let rejoined = rejoin != 0 && self.reconnecting.contains_key(&rejoin) && self.peer_ids.get(&rejoin) == Some(&pk);
        if !rejoined && self.peers.len() as u32 >= self.cfg.max_players {
            sealed_reject(self, &mut sec, "session full");
            return;
        }
        self.seen_nonces.insert(nonce, now);
        let peer = if rejoined {
            rejoin
        } else {
            let p = self.next_peer;
            self.next_peer += 1;
            p
        };
        if let Some(old) = self.conns.remove(&peer) {
            self.addr_peer.remove(&old.addr);
        }
        let mut conn = Conn::new(src, peer, now);
        conn.name = if name.is_empty() { format!("Player {peer}") } else { name };
        conn.sec = Some(sec);
        conn.client_nonce = nonce;
        conn.identity = pk;
        conn.account = account_name;
        conn.transcript = transcript;
        conn.confirmed = false;
        conn.rejoined = rejoined;
        conn.can_host = can_host;
        self.conns.insert(peer, conn);
        self.addr_peer.insert(src, peer);
        self.send_welcome(peer);
        self.admit(peer);
    }

    fn send_welcome(&mut self, peer: u64) {
        let Some(c) = self.conns.get(&peer) else {
            return;
        };
        let (addr, transcript, rejoined) = (c.addr, c.transcript, c.rejoined);
        let mut b = vec![P_WELCOME];
        put_u64(&mut b, peer);
        put_u32(&mut b, self.cfg.max_players);
        put_str(&mut b, &self.cfg.session_name);
        put_u64(&mut b, self.local_peer);
        put_u32(&mut b, self.epoch);
        put_f64(&mut b, self.clock_offset);
        // Bit 0: a member coming back after a migration. Bit 1: dedicated server.
        b.push(rejoined as u8 | (self.cfg.topology.to_u8() << 1));
        b.extend_from_slice(&self.cfg.identity.sign(&welcome_message(&transcript, peer)));
        self.send_raw(&b, addr);
    }

    /// Host: a joiner proved the secret and its identity: it becomes a
    /// member (or, after a migration, a returning one) and is brought up to
    /// date. Until its first sealed packet confirms the key it counts as a
    /// half-open join.
    fn admit(&mut self, peer: u64) {
        let Some(c) = self.conns.get(&peer) else {
            return;
        };
        let (name, rejoined, pk, addr) = (c.name.clone(), c.rejoined, c.identity, c.addr);
        self.peer_ids.insert(peer, pk);
        self.val.peer_joined(peer, pk, &name);
        self.ac_register(peer, &pk);
        if rejoined {
            self.reconnecting.remove(&peer);
            self.rejoins += 1;
            // Its predicted controllers wait for the owner's state (M_ANCHOR).
            let until = self.now() + 1.0;
            let mine: Vec<u128> =
                self.world.owners.iter().filter(|(id, (o, _))| *o == peer && self.world.predicted.contains(*id)).map(|(id, _)| *id).collect();
            for id in mine {
                self.anchor_wait.insert(id, until);
            }
            if self.ac_held(peer) {
                self.ac_deferred.insert(peer, true);
            } else {
                self.send_catchup(peer, true);
                self.send_backlog(peer);
            }
            tracing::info!("net: peer {peer} ({name}) is back after the host migration");
        } else {
            let mut j = vec![M_JOINED];
            put_u64(&mut j, peer);
            put_str(&mut j, &name);
            j.extend_from_slice(&pk);
            self.broadcast_reliable(&j, Some(peer));
            self.peers.insert(peer, name.clone());
            if self.ac_held(peer) {
                // `require`: no world state and no player object until the provider trusts it.
                self.ac_deferred.insert(peer, false);
            } else {
                self.send_catchup(peer, false);
                self.players_to_spawn.push(peer);
            }
            self.push_event(NetEvent::PeerJoined { id: peer, name: name.clone() });
            tracing::info!(
                "net: peer {peer} ({name}, {}) joined from {addr} ({})",
                short_fingerprint(&fingerprint(&pk)),
                self.transport.route_label(addr)
            );
        }
        self.broadcast_succession();
    }

    /// Everything a late joiner (or, wrapped in RESYNC, a member rejoining
    /// after a migration) needs, in dependency order.
    fn send_catchup(&mut self, peer: u64, resync: bool) {
        if resync {
            let mut m = vec![M_RESYNC];
            put_u32(&mut m, self.epoch);
            self.send_reliable(peer, &m);
        }
        let mut m = vec![M_PEERS];
        put_u32(&mut m, self.peers.len() as u32);
        for (id, name) in &self.peers {
            put_u64(&mut m, *id);
            put_str(&mut m, name);
            m.extend_from_slice(&self.peer_ids.get(id).copied().unwrap_or([0; 32]));
        }
        self.send_reliable(peer, &m);
        if let Some(scene) = self.world.scene.clone() {
            let mut m = vec![M_SCENE];
            put_str(&mut m, &scene);
            self.send_reliable(peer, &m);
        }
        let mut msgs = Vec::new();
        // With interest management the objects follow as they become relevant.
        let all = !self.interest_on();
        for s in self.world.spawns.iter().filter(|_| all) {
            msgs.push(enc_spawn(s));
        }
        for (id, (owner, seq)) in &self.world.owners {
            let mut m = vec![M_OWNER];
            put_u128(&mut m, *id);
            put_u64(&mut m, *owner);
            put_u64(&mut m, *seq);
            msgs.push(m);
        }
        for (id, parent) in &self.world.parents {
            let mut m = vec![M_PARENT];
            put_u128(&mut m, *id);
            put_u128(&mut m, *parent);
            msgs.push(m);
        }
        for (id, on) in &self.world.actives {
            let mut m = vec![M_ACTIVE];
            put_u128(&mut m, *id);
            m.push(*on as u8);
            msgs.push(m);
        }
        for (id, vars) in &self.world.vars {
            for (k, v) in vars {
                msgs.push(enc_var(*id, k, v, self.host_peer, 0));
            }
        }
        for (id, x) in &self.world.xforms {
            let mut m = vec![M_XFORM];
            enc_entry(
                &mut m,
                &SnapEntry { id: *id, pos: Some(x.pos), rot: Some(x.rot), scale: Some(x.scale), vel: None, quantized: false },
            );
            msgs.push(m);
        }
        for (id, params) in &self.world.anims {
            let mut m = vec![M_ANIM];
            put_u128(&mut m, *id);
            put_u16(&mut m, params.len() as u16);
            for (k, v) in params {
                put_str(&mut m, k);
                put_f32(&mut m, *v);
            }
            msgs.push(m);
        }
        for (id, packet) in &self.world.avparams {
            let mut m = vec![M_AVPARAM];
            put_u128(&mut m, *id);
            put_u16(&mut m, packet.len().min(u16::MAX as usize) as u16);
            m.extend_from_slice(&packet[..packet.len().min(u16::MAX as usize)]);
            msgs.push(m);
        }
        if !all {
            msgs.clear();
        }
        if resync {
            msgs.push(vec![M_RESYNC_END]);
        }
        for m in msgs {
            self.send_reliable(peer, &m);
        }
    }

    /// Host: a member left, timed out, was kicked, never finished joining
    /// or never came back after a migration.
    fn drop_peer(&mut self, peer: u64, reason: &str) {
        let conn = self.conns.remove(&peer);
        if let Some(c) = &conn {
            self.addr_peer.remove(&c.addr);
        }
        let expected = self.reconnecting.remove(&peer).is_some();
        if conn.is_none() && !expected {
            return;
        }
        self.peers.remove(&peer);
        self.peer_ids.remove(&peer);
        self.val.peer_left(peer);
        self.ac_unregister(peer);
        self.interest_drop_peer(peer);
        self.players_to_spawn.retain(|p| *p != peer);
        let mut m = vec![M_LEFT];
        put_u64(&mut m, peer);
        put_str(&mut m, reason);
        self.broadcast_reliable(&m, None);
        self.release_objects_of(peer);
        self.offsets.remove(&peer);
        self.jitter.remove(&peer);
        self.push_event(NetEvent::PeerLeft { id: peer, reason: reason.to_owned() });
        tracing::info!("net: peer {peer} {reason}");
        self.broadcast_succession();
    }

    /// Host: a departed peer's objects: despawn the flagged ones and its
    /// player object, release the rest to the host.
    fn release_objects_of(&mut self, peer: u64) {
        let owned: Vec<u128> = self.world.owners.iter().filter(|(_, (o, _))| *o == peer).map(|(id, _)| *id).collect();
        let player = self.world.player_obj.get(&peer).copied();
        for id in owned {
            if self.world.despawn_on_leave.contains(&id) || Some(id) == player {
                self.despawn_raw(id);
            } else {
                self.grant(id, 0);
            }
        }
        if let Some(p) = player
            && self.world.spawns.iter().any(|s| s.id == p)
        {
            self.despawn_raw(p);
        }
        self.world.player_obj.remove(&peer);
    }

    /// Host: kick a member (it is told why and does not migrate).
    pub fn kick(&mut self, peer: u64, reason: &str) {
        if !self.is_host || peer == self.local_peer {
            return;
        }
        if let Some(addr) = self.conns.get(&peer).map(|c| c.addr) {
            let mut b = vec![P_BYE, BYE_KICKED];
            put_str(&mut b, reason);
            for _ in 0..3 {
                self.send_raw(&b, addr);
            }
        }
        let why = if reason.is_empty() { "kicked".to_string() } else { format!("kicked: {reason}") };
        self.drop_peer(peer, &why);
    }

    /// Host: ban a member's player id (refused from now on) and kick it.
    pub fn ban(&mut self, peer: u64) {
        if let Some(fp) = self.peer_fingerprint(peer) {
            self.ban_fingerprint(&fp);
        }
        self.kick(peer, "banned");
    }

    pub fn ban_fingerprint(&mut self, fp: &str) {
        if !self.cfg.ban.iter().any(|b| crate::net_identity::fingerprint_matches(b, fp)) {
            self.cfg.ban.push(fp.to_owned());
            if self.cfg.ban_file.is_some() {
                // A server's bans survive a restart.
                self.val.bans.add(fp, 0, "banned by an admin", crate::net_validate::unix_now());
            }
            self.broadcast_succession();
        }
    }

    pub fn unban(&mut self, fp: &str) {
        let before = self.cfg.ban.len();
        self.cfg.ban.retain(|b| !crate::net_identity::fingerprint_matches(b, fp));
        let lifted = self.val.bans.remove(fp);
        if self.cfg.ban.len() != before || lifted {
            self.broadcast_succession();
        }
    }

    // ------------------------------------------------------------ host migration

    /// Host: who takes over, in order: members in join order (lowest peer
    /// id first), skipping ones that can't host. Every member computes the
    /// same list from the same broadcast, so the election needs no vote.
    ///
    /// Addresses are private: only the first `migration_addr_depth` candidates
    /// carry endpoints (the ones a member would actually try), and none do in a
    /// dedicated or relay-mode session. The others are listed by id only.
    fn successors(&self) -> Vec<Successor> {
        let share = if self.cfg.topology == Topology::Dedicated || self.room_key.is_some() { 0 } else { self.cfg.migration_addr_depth as usize };
        self.members()
            .into_iter()
            .filter_map(|p| self.conns.get(&p))
            .filter(|c| c.can_host)
            .enumerate()
            .map(|(i, c)| {
                let addrs = if i >= share {
                    Vec::new()
                } else {
                    match crate::net_relay::slot_of(c.addr) {
                        None => vec![c.addr],
                        Some(_) => self.transport.direct_route(c.addr).into_iter().collect(),
                    }
                };
                Successor { peer: c.peer, addrs }
            })
            .collect()
    }

    fn current_succession(&mut self) -> Succession {
        self.succession_version += 1;
        let slots = self
            .successors()
            .iter()
            .filter_map(|s| self.conns.get(&s.peer).and_then(|c| crate::net_relay::slot_of(c.addr)))
            .collect();
        self.transport.relay_set_successors(slots);
        let (relay, room) = match (self.room_key, self.transport.relay_status()) {
            (Some(_), Some(r)) if r.phase == RelayPhase::Registered => (self.relay_addr.clone(), r.code),
            _ => (String::new(), String::new()),
        };
        Succession {
            version: self.succession_version,
            epoch: self.epoch,
            next_peer: self.next_peer,
            spawn_counter: self.spawn_counter,
            max_players: self.cfg.max_players,
            clients_can_spawn: self.cfg.clients_can_spawn,
            session_name: self.cfg.session_name.clone(),
            room_key: if room.is_empty() { None } else { self.room_key },
            relay,
            room,
            order: self.successors(),
            ban: self.cfg.ban.clone(),
            allow: self.cfg.allow.clone(),
        }
    }

    /// Host: tell every member the current succession plan.
    fn broadcast_succession(&mut self) {
        if !self.is_host || !self.cfg.migration {
            return;
        }
        let s = self.current_succession();
        let m = enc_succession(&s);
        self.broadcast_reliable(&m, None);
    }

    /// Graceful host quit: the successor gets the latest state, everyone is
    /// told who takes over.
    fn hand_off(&mut self) {
        let order = self.successors();
        let Some(next) = order.first().cloned() else {
            return;
        };
        let s = self.current_succession();
        let m = enc_succession(&s);
        self.broadcast_reliable(&m, None);
        self.send_catchup(next.peer, true);
        self.flush_unacked_now();
        let mut b = vec![P_BYE, BYE_HANDOFF];
        put_u64(&mut b, next.peer);
        b.push(next.addrs.len().min(4) as u8);
        for a in next.addrs.iter().take(4) {
            relay_proto::put_addr(&mut b, *a);
        }
        let addrs: Vec<SocketAddr> = self.conns.values().map(|c| c.addr).collect();
        for a in addrs {
            for _ in 0..3 {
                self.send_raw(&b, a);
            }
        }
        tracing::info!("net: handing the session to peer {}", next.peer);
    }

    /// Send every unacknowledged reliable packet once more right now.
    fn flush_unacked_now(&mut self) {
        let out: Vec<(Vec<u8>, SocketAddr)> =
            self.conns.values().flat_map(|c| c.unacked.values().map(move |(p, _, _)| (p.clone(), c.addr))).collect();
        for (p, a) in out {
            self.send_raw(&p, a);
        }
    }

    /// Client: the host is gone (timed out, left, or handed off to `to`).
    /// Without a plan, or with migration off, the session ends.
    fn host_lost(&mut self, to: Option<(u64, Vec<SocketAddr>)>, reason: &str) {
        let previous = self.host_peer;
        let handoff = to.is_some();
        let plan = self.succession.clone().filter(|_| self.cfg.migration && self.state == ConnState::Connected);
        let Some(plan) = plan else {
            self.conns.remove(&previous);
            self.disconnect_local(if reason == "the host left" { "the host left the session" } else { reason });
            return;
        };
        let mut candidates: Vec<Successor> =
            plan.order.iter().filter(|c| c.peer != previous && self.peers.contains_key(&c.peer)).cloned().collect();
        if let Some((peer, addrs)) = to {
            let mut first = candidates.iter().position(|c| c.peer == peer).map(|i| candidates.remove(i)).unwrap_or(Successor { peer, addrs: Vec::new() });
            for a in addrs {
                if !first.addrs.contains(&a) {
                    first.addrs.push(a);
                }
            }
            candidates.insert(0, first);
        }
        let others = self.peers.keys().filter(|p| **p != previous && **p != self.local_peer).count();
        if candidates.is_empty() || (!handoff && others == 0) {
            self.conns.remove(&previous);
            self.disconnect_local(if handoff { "the host left and nobody can take over" } else { reason });
            return;
        }
        // Reliable messages the old host may not have handled go to the new one.
        let mut resend = self.conns.get(&previous).map(|c| c.unacked_messages()).unwrap_or_default();
        resend.append(&mut self.outbox);
        self.outbox = resend;
        if let Some(c) = self.conns.remove(&previous) {
            self.addr_peer.remove(&c.addr);
        }
        let now = self.now();
        tracing::info!("net: lost the host ({reason}); successors: {:?}", candidates.iter().map(|c| c.peer).collect::<Vec<_>>());
        self.migration = Some(Migration { previous, candidates, index: 0, relayed: false, attempt_started: None, started: now });
        self.push_event(NetEvent::HostMigrating { previous });
    }

    /// Client: walk the successor list: promote when it's our turn, else
    /// try the candidate directly, then through the relay, then the next.
    fn migration_tick(&mut self, now: f64) {
        let Some(m) = self.migration.as_ref() else {
            return;
        };
        let Some(target) = m.candidates.get(m.index).cloned() else {
            self.disconnect_local("lost the host: no successor could be reached");
            return;
        };
        if target.peer == self.local_peer {
            self.promote();
            return;
        }
        let attempt = self.cfg.migrate_attempt.max(0.5) as f64;
        let hinted = self.hs.as_ref().is_some_and(|h| h.challenge.is_none() && h.hint.as_ref().is_some_and(|(_, t)| now - t > REJECT_GRACE));
        let expired = m.attempt_started.is_some_and(|t| now - t > attempt) || hinted;
        if m.attempt_started.is_some() && !expired {
            return;
        }
        let relayed = m.relayed;
        let started = m.attempt_started.is_some();
        // Next path: direct first, then the relay, then the next candidate.
        let (relayed, advance) = match (started, relayed) {
            (false, _) => (false, false),
            (true, false) => (true, false),
            (true, true) => (false, true),
        };
        if let Some(m) = self.migration.as_mut() {
            if advance {
                m.index += 1;
            }
            m.relayed = relayed;
        }
        if advance {
            self.hs = None;
            if let Some(m) = self.migration.as_mut() {
                m.attempt_started = None;
            }
            return self.migration_tick(now);
        }
        let addrs: Vec<SocketAddr> = if relayed { self.relay_path().into_iter().collect() } else { target.addrs.clone() };
        if let Some(m) = self.migration.as_mut() {
            m.attempt_started = Some(now);
        }
        if addrs.is_empty() {
            // No such path: skip it right away.
            if let Some(m) = self.migration.as_mut() {
                m.attempt_started = Some(now - attempt - 1.0);
            }
            self.hs = None;
            return self.migration_tick(now);
        }
        tracing::info!("net: trying peer {} as the new host ({})", target.peer, if relayed { "relayed" } else { "direct" });
        let known = self.peer_ids.get(&target.peer).copied();
        self.start_handshake(addrs, Some((target.peer, known)));
    }

    /// The relay path to whoever holds the room's host slot (attaching the
    /// relay first when this peer joined directly).
    fn relay_path(&mut self) -> Option<SocketAddr> {
        let plan = self.succession.clone()?;
        if plan.room.is_empty() {
            return None;
        }
        let in_room = self.transport.relay_status().is_some_and(|r| !matches!(r.phase, RelayPhase::Failed(_)) && r.code == plan.room);
        if !in_room {
            let relay = parse_relay(&plan.relay).ok()?;
            if !self.transport.attach_relay(relay, RelayRole::Client { code: plan.room.clone() }, self.cfg.punch) {
                return None;
            }
            self.relay_addr = plan.relay.clone();
            let p = self.kdf;
            if let Some(k) = self.session_key_for(&p) {
                self.transport.set_relay_join_key(relay_join_key(&k));
            }
        }
        Some(virtual_addr(HOST_SLOT))
    }

    /// Client: our turn: take over as host with the world as we have it.
    fn promote(&mut self) {
        let Some(m) = self.migration.take() else {
            return;
        };
        let previous = m.previous;
        let now = self.now();
        let shared = self.shared_clock().unwrap_or(now);
        let rank = m.index;
        self.hs = None;
        self.resync = None;
        self.is_host = true;
        self.host_peer = self.local_peer;
        // Interest management starts afresh on the new host: no client baselines, no 32-bit ids
        // from the old host (or from an earlier stint of our own), so every member that comes
        // back is brought up to date from scratch.
        self.interest.reset();
        self.epoch += 1;
        self.claim_seq = (self.epoch as u64) << 40;
        self.clock_offset = shared - now;
        self.clock_value = Some((now, shared));
        // Lag-compensation history restarts on the new host. A crashed old
        // host leaves nothing to copy, and a successor that previously
        // hosted (then stepped down for a better-ranked one) must not keep
        // serving those stale hitboxes; seed the clock so rewinds clamp to
        // the handover moment before the first frame is recorded.
        self.history.lock().unwrap_or_else(|e| e.into_inner()).restart(shared);
        let plan = self.succession.clone().unwrap_or_default();
        let top = self.peers.keys().copied().max().unwrap_or(1);
        self.next_peer = self.next_peer.max(plan.next_peer).max(top + 1);
        self.spawn_counter = self.spawn_counter.max(plan.spawn_counter);
        if plan.max_players > 0 {
            self.cfg.max_players = plan.max_players;
            self.cfg.session_name = plan.session_name.clone();
            self.cfg.clients_can_spawn = plan.clients_can_spawn;
        }
        for b in &plan.ban {
            if !self.cfg.ban.contains(b) {
                self.cfg.ban.push(b.clone());
            }
        }
        if self.cfg.allow.is_empty() {
            self.cfg.allow = plan.allow.clone();
        }
        if plan.room_key.is_some() {
            self.room_key = plan.room_key;
            self.relay_addr = plan.relay.clone();
        }
        // The old host is gone; everyone else is expected back.
        self.peers.remove(&previous);
        self.peer_ids.remove(&previous);
        self.offsets.remove(&previous);
        self.jitter.remove(&previous);
        // Members reach us after giving up on everyone ranked above (two
        // paths each), some a little later than us: wait that long, plus slack.
        let grace = now + self.cfg.timeout as f64 + self.cfg.migrate_attempt.max(0.5) as f64 * 2.0 * (rank as f64 + 1.0) + 2.0;
        self.reconnecting = self.peers.keys().filter(|p| **p != self.local_peer).map(|p| (*p, grace)).collect();
        self.reconnect_expected = !self.reconnecting.is_empty();
        self.rejoins = 0;
        // Objects the old host owned; player objects come back from the spawn flags.
        self.world.player_obj.clear();
        for s in &self.world.spawns {
            if s.player {
                self.world.player_obj.insert(s.owner, s.id);
                self.world.despawn_on_leave.insert(s.id);
            }
        }
        self.release_objects_of(previous);
        self.players_to_spawn.clear();
        self.host_player_done = true;
        self.inputs.clear();
        // Our own messages the old host never acknowledged: act on them as the host.
        let out = std::mem::take(&mut self.outbox);
        for msg in out {
            self.replay_as_host(&msg);
        }
        // Take the relay room over (joining it first when we came in directly).
        if let Some(seed) = self.room_key
            && !plan.room.is_empty()
        {
            if self.transport.relay_status().is_none()
                && let Ok(relay) = parse_relay(&self.relay_addr)
                && self.transport.attach_relay(relay, RelayRole::Client { code: plan.room.clone() }, self.cfg.punch)
            {
                let p = self.kdf;
                if let Some(k) = self.session_key_for(&p) {
                    self.transport.set_relay_join_key(relay_join_key(&k));
                }
            }
            self.transport.relay_claim_host(seed);
        }
        // Better-ranked successors we gave up on might be hosting too: check.
        self.probes = m.candidates[..rank.min(m.candidates.len())]
            .iter()
            .flat_map(|c| c.addrs.iter().map(move |a| (c.peer, *a)))
            .map(|(peer, addr)| Probe { peer, addr, nonce: random_bytes(), sec: None, join: None, last: f64::NEG_INFINITY })
            .collect();
        self.probe_until = grace;
        self.migrations += 1;
        self.push_event(NetEvent::PeerLeft { id: previous, reason: "the host left".into() });
        self.push_event(NetEvent::HostChanged { host: self.local_peer, previous });
        tracing::info!("net: took over as host (epoch {}), expecting {} member(s) back", self.epoch, self.reconnecting.len());
    }

    /// New host: act on a reliable message this peer had sent the old host
    /// that it never acknowledged.
    fn replay_as_host(&mut self, msg: &[u8]) {
        let me = self.local_peer;
        let mut c = Cursor::new(msg, 1);
        match msg.first().copied() {
            Some(M_RPC) | Some(M_TEXT) => self.backlog.push(Backlog { sender: me, msg: msg.to_vec() }),
            Some(M_CLAIM) => {
                let id = c.u128();
                if !c.bad {
                    self.handle_claim(id, me);
                }
            }
            Some(M_RELEASE) => {
                let id = c.u128();
                if !c.bad && self.owner_of_raw(id) == me {
                    self.grant(id, 0);
                }
            }
            Some(M_TRANSFER) => {
                let id = c.u128();
                let to = c.u64();
                if !c.bad && self.owner_of_raw(id) == me {
                    self.grant(id, to);
                }
            }
            // Spawns, despawns, variables, parents and activation are in our
            // world already; returning members get them in the resync.
            _ => {}
        }
    }

    /// New host: replay the backlog to a member that just came back.
    fn send_backlog(&mut self, peer: u64) {
        let msgs: Vec<(u64, Vec<u8>)> = self.backlog.iter().map(|b| (b.sender, b.msg.clone())).collect();
        for (sender, msg) in msgs {
            if sender == peer {
                continue;
            }
            let to_peer = match msg.first() {
                Some(&M_RPC) => {
                    let mut c = Cursor::new(&msg, 1);
                    match dec_rpc(&mut c) {
                        Some((from, _, target, obj, _, _)) => self.rpc_recipient(from, target, obj, peer),
                        None => false,
                    }
                }
                Some(&M_TEXT) => {
                    let target = Cursor::new(&msg, 1).u64();
                    target == 0 || target == peer
                }
                _ => false,
            };
            if to_peer {
                self.send_reliable(peer, &msg);
            }
        }
    }

    /// Host upkeep: relay credentials, half-open joins, members that never
    /// came back after a migration, the backlog and tie-break probes.
    fn host_tick(&mut self, now: f64) {
        // Keep the key stretching moving; the room opens once it is done.
        let p = self.kdf;
        let key = self.session_key_for(&p);
        if !self.relay_auth_sent
            && self.migrations == 0
            && let (Some(seed), Some(k)) = (self.room_key, key)
            && self.transport.relay_status().is_some()
        {
            self.transport.set_relay_host_auth(HostAuth {
                owner_seed: seed,
                join_key: relay_join_key(&k),
                kdf: self.kdf.to_bytes(),
                token: self.cfg.relay_token.clone(),
            });
            self.relay_auth_sent = true;
        }
        // Another member took the relay room while we host: take it back.
        if let Some(seed) = self.room_key
            && self.migrations > 0
            && self.transport.relay_status().is_some_and(|r| matches!(&r.phase, RelayPhase::Failed(why) if why.contains("successor")))
            && let Ok(relay) = parse_relay(&self.relay_addr)
            && let Some(code) = self.succession.as_ref().map(|s| s.room.clone()).filter(|c| !c.is_empty())
            && self.transport.attach_relay(relay, RelayRole::Client { code }, self.cfg.punch)
        {
            let p = self.kdf;
            if let Some(k) = self.session_key_for(&p) {
                self.transport.set_relay_join_key(relay_join_key(&k));
            }
            self.transport.relay_claim_host(seed);
        }
        let stale: Vec<u64> = self.conns.values().filter(|c| !c.confirmed && now - c.created > HALF_OPEN_TTL).map(|c| c.peer).collect();
        for p in stale {
            self.drop_peer(p, "did not finish joining");
        }
        let gone: Vec<u64> = self.reconnecting.iter().filter(|(_, t)| now > **t).map(|(p, _)| *p).collect();
        for p in gone {
            self.drop_peer(p, "did not come back after the host migration");
        }
        if self.reconnect_expected && self.reconnecting.is_empty() {
            self.reconnect_expected = false;
            self.backlog.clear();
            if self.rejoins == 0 && self.members().is_empty() {
                // Everyone else stayed with another host: we were cut off.
                self.disconnect_local("lost the session: nobody reconnected after the host migration");
                return;
            }
        }
        self.seen_nonces.retain(|_, t| now - *t < COOKIE_PERIOD * 3.0);
        self.probe_tick(now);
    }

    fn probe_tick(&mut self, now: f64) {
        if self.probes.is_empty() {
            return;
        }
        if now > self.probe_until {
            self.probes.clear();
            return;
        }
        let mut out = Vec::new();
        for p in self.probes.iter_mut() {
            if now - p.last >= PROBE_INTERVAL {
                p.last = now;
                out.push((p.join.clone().unwrap_or_else(|| hello_packet(&p.nonce, JOIN_MODE_PROBE)), p.addr));
            }
        }
        for (b, a) in out {
            self.send_clear(&b, a);
        }
    }

    /// Prober: a better-ranked successor answered; prove we're a member.
    fn on_probe_challenge(&mut self, data: &[u8], src: SocketAddr) {
        let Some(i) = self.probes.iter().position(|p| p.addr == src && p.join.is_none()) else {
            return;
        };
        let Some(ch) = parse_challenge(data, &self.probes[i].nonce) else {
            return;
        };
        if ch.host_peer != self.probes[i].peer || ch.kdf != self.kdf {
            return;
        }
        let p = self.kdf;
        let Some(k) = self.session_key_for(&p) else {
            return;
        };
        let mut sec = SecureChannel::new(&conn_key(&k, &ch.transcript), false);
        let mut header = vec![P_JOIN];
        header.extend_from_slice(&self.probes[i].nonce);
        header.extend_from_slice(&ch.cookie);
        header.push(JOIN_MODE_PROBE);
        let mut plain = Vec::new();
        put_u64(&mut plain, self.local_peer);
        put_u32(&mut plain, self.epoch);
        let join = sec.seal_at(&header, 0, &plain);
        sec.skip_to(1);
        self.probes[i].sec = Some(sec);
        self.probes[i].join = Some(join.clone());
        self.send_clear(&join, src);
    }

    /// Host: a member's probe (sealed under the session key): say we host.
    fn on_probe(&mut self, data: &[u8], src: SocketAddr, k: &[u8; 32], transcript: &[u8; 32]) {
        let now = self.now();
        let mut sec = SecureChannel::new(&conn_key(k, transcript), true);
        if sec.open(data, JOIN_HEADER, now).is_none() {
            self.stats.rejected += 1;
            return;
        }
        let mut b = vec![P_PROBE_ACK];
        put_u64(&mut b, self.local_peer);
        put_u32(&mut b, self.epoch);
        put_u32(&mut b, self.members().len() as u32);
        let p = sec.seal(&[P_SECURE], &b);
        self.send_clear(&p, src);
    }

    /// Prober: a better-ranked successor is hosting: hand our members to it
    /// and join it ourselves (the deterministic tie-break).
    fn on_probe_ack(&mut self, i: usize, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        if data.first() != Some(&P_PROBE_ACK) {
            return;
        }
        let host = c.u64();
        if c.bad || host != self.probes[i].peer || !self.is_host {
            return;
        }
        let addr = self.probes[i].addr;
        self.epoch = self.epoch.saturating_sub(1);
        tracing::info!("net: peer {host} is hosting too and ranks higher: handing over");
        self.probes.clear();
        let mut b = vec![P_BYE, BYE_HANDOFF];
        put_u64(&mut b, host);
        b.push(1);
        relay_proto::put_addr(&mut b, addr);
        let addrs: Vec<SocketAddr> = self.conns.values().map(|c| c.addr).collect();
        for a in addrs {
            for _ in 0..3 {
                self.send_raw(&b, a);
            }
        }
        self.conns.clear();
        self.addr_peer.clear();
        self.is_host = false;
        self.reconnecting.clear();
        self.reconnect_expected = false;
        self.backlog.clear();
        self.players_to_spawn.clear();
        let me = self.local_peer;
        let mut candidates = vec![Successor { peer: host, addrs: vec![addr] }];
        if let Some(plan) = &self.succession {
            candidates.extend(plan.order.iter().filter(|c| c.peer != host && c.peer != me).cloned());
        }
        let now = self.now();
        self.migration = Some(Migration { previous: me, candidates, index: 0, relayed: false, attempt_started: None, started: now });
        self.host_peer = me;
    }

    // ------------------------------------------------------------ reliable messages

    fn on_msg(&mut self, from: u64, m: &[u8]) {
        let Some(&tag) = m.first() else {
            return;
        };
        let mut c = Cursor::new(m, 1);
        let host = self.is_host;
        if host && from != self.local_peer {
            // A client's reliable message: charged to its message bucket first, ignored
            // while it is over the drop-level strike score.
            if !self.gate(from, Limit::Messages) || (tag != M_ANTICHEAT && self.muted(from)) {
                return;
            }
        }
        match tag {
            M_PEERS if !host => {
                let n = c.u32();
                let mut list = BTreeMap::new();
                let mut ids = Vec::new();
                for _ in 0..n.min(4096) {
                    let id = c.u64();
                    let name = c.string();
                    let pk: [u8; 32] = c.arr();
                    list.insert(id, name);
                    ids.push((id, pk));
                }
                if c.bad {
                    return;
                }
                list.insert(self.local_peer, self.cfg.player_name.clone());
                for (id, name) in &list {
                    if *id != self.local_peer && !self.peers.contains_key(id) {
                        self.push_event(NetEvent::PeerJoined { id: *id, name: name.clone() });
                    }
                }
                let gone: Vec<u64> = self.peers.keys().copied().filter(|id| !list.contains_key(id)).collect();
                for id in gone {
                    self.peer_ids.remove(&id);
                    self.offsets.remove(&id);
                    self.push_event(NetEvent::PeerLeft { id, reason: "left".into() });
                }
                for (id, pk) in ids {
                    if pk != [0; 32] {
                        self.peer_ids.insert(id, pk);
                    }
                }
                self.peers = list;
            }
            M_JOINED if !host => {
                let id = c.u64();
                let name = c.string();
                let pk: [u8; 32] = c.arr();
                if !c.bad && id != self.local_peer {
                    self.peers.insert(id, name.clone());
                    self.peer_ids.insert(id, pk);
                    self.push_event(NetEvent::PeerJoined { id, name });
                }
            }
            M_LEFT if !host => {
                let id = c.u64();
                let reason = c.string();
                if self.peers.remove(&id).is_some() {
                    self.peer_ids.remove(&id);
                    self.offsets.remove(&id);
                    self.push_event(NetEvent::PeerLeft { id, reason });
                }
            }
            M_SUCCESSION if !host => {
                if let Some(s) = dec_succession(&mut c) {
                    self.succession = Some(s);
                }
            }
            M_RESYNC if !host => {
                let mine = self.world.spawns.iter().filter(|s| s.owner == self.local_peer).cloned().collect();
                self.resync = Some(Resync { mine_before: mine, ..Default::default() });
            }
            M_RESYNC_END if !host => self.finish_resync(),
            M_OWNER if !host => {
                let id = c.u128();
                let owner = c.u64();
                let seq = c.u64();
                if c.bad {
                    return;
                }
                if let Some(r) = self.resync.as_mut() {
                    // The new host's view wins, whatever the sequence numbers.
                    r.owners.insert(id);
                    let prev = self.owner_of_raw(id);
                    self.world.owners.insert(id, (owner, seq));
                    if prev != owner {
                        self.push_event(NetEvent::OwnershipChanged { object: ObjectId::from_raw(id), owner, previous: prev });
                    }
                } else {
                    self.set_owner_local(id, owner, seq);
                }
            }
            M_CLAIM if host => {
                let id = c.u128();
                if !c.bad && self.gate(from, Limit::Ownership) {
                    self.handle_claim(id, from);
                }
            }
            M_RELEASE if host => {
                let id = c.u128();
                if !c.bad && self.owner_of_raw(id) == from {
                    self.grant(id, 0);
                }
            }
            M_TRANSFER if host => {
                let id = c.u128();
                let to = c.u64();
                if !c.bad && self.owner_of_raw(id) == from {
                    // Only to somebody who is in the session (or to nobody).
                    if to == 0 || self.peers.contains_key(&to) {
                        self.grant(id, to);
                    } else {
                        self.violation(from, Reason::OwnershipDenied, "transfer to a peer that is not here");
                    }
                }
            }
            M_OWNER_REQ if !host => {
                let id = c.u128();
                let by = c.u64();
                self.push_event(NetEvent::OwnershipRequested { object: ObjectId::from_raw(id), by });
            }
            M_DENY if !host => {
                let id = c.u128();
                self.push_event(NetEvent::OwnershipDenied { object: ObjectId::from_raw(id) });
            }
            M_SPAWN => {
                let Some(mut rec) = dec_spawn(&mut c) else {
                    return;
                };
                if host {
                    if !self.cfg.clients_can_spawn {
                        let mut d = vec![M_DESPAWN];
                        put_u128(&mut d, rec.id);
                        self.send_reliable(from, &d);
                        return;
                    }
                    if self.world.spawns.iter().any(|s| s.id == rec.id) {
                        return;
                    }
                    if !self.allow_client_spawn(from, &rec) {
                        // The client already made it locally: have it removed again.
                        let mut d = vec![M_DESPAWN];
                        put_u128(&mut d, rec.id);
                        self.send_reliable(from, &d);
                        return;
                    }
                    if rec.owner != 0 {
                        rec.owner = from;
                    }
                    rec.player = false;
                    self.forward_spawn(rec, Some(from));
                } else {
                    if let Some(r) = self.resync.as_mut() {
                        r.spawns.insert(rec.id);
                        if let Some(s) = self.world.spawns.iter_mut().find(|s| s.id == rec.id) {
                            s.owner = rec.owner;
                            s.player = rec.player;
                            return;
                        }
                    }
                    self.apply_spawn(rec);
                }
            }
            M_DESPAWN => {
                let id = c.u128();
                if c.bad {
                    return;
                }
                if host {
                    let owner = self.owner_of_raw(id);
                    if owner != from && !(owner == 0 && from == self.host_peer) {
                        return;
                    }
                    // Nobody despawns a player object but the server (leaving does it).
                    if self.val.strict() && self.world.player_obj.values().any(|p| *p == id) {
                        self.violation(from, Reason::OwnershipDenied, "despawn of a player object");
                        return;
                    }
                    let mut d = vec![M_DESPAWN];
                    put_u128(&mut d, id);
                    self.broadcast_reliable(&d, Some(from));
                }
                self.apply_despawn(id);
            }
            M_VAR | M_VAR_SYNC => {
                let id = c.u128();
                let key = c.string();
                let value = c.value();
                let sender = c.u64();
                let mid = c.u32();
                if c.bad {
                    return;
                }
                let sender = if host { from } else { sender };
                if host && !self.admit_var(from, &key, &value) {
                    return;
                }
                if !self.fresh_mid(sender, mid) {
                    return;
                }
                if host {
                    let owner = self.owner_of_raw(id);
                    if owner != from && !self.world.var_anyone.contains(&id) {
                        tracing::debug!("net: dropped var write {key:?} from non-owner {from}");
                        return;
                    }
                    if tag == M_VAR_SYNC && self.world.var(id, &key) == Some(&value) {
                        return;
                    }
                    self.broadcast_reliable(&enc_var(id, &key, &value, from, mid), Some(from));
                } else if self.resync.is_some() && self.owner_of_raw(id) == self.local_peer {
                    // Ours: our copy is the newest; it is re-sent after the resync.
                    return;
                }
                self.apply_var(id, key, value, sender);
            }
            M_ACTIVE => {
                let id = c.u128();
                let on = c.u8() != 0;
                if c.bad || (host && self.owner_of_raw(id) != from) {
                    return;
                }
                if host {
                    self.broadcast_reliable(m, Some(from));
                }
                self.world.actives.insert(id, on);
                self.world_ops.push(NetWorldOp::SetActive(ObjectId::from_raw(id), on));
            }
            M_PARENT => {
                let id = c.u128();
                let parent = c.u128();
                if c.bad || (host && self.owner_of_raw(id) != from) {
                    return;
                }
                if host {
                    let verdict = self.val.parent_ok(from, self.host_peer, id, parent, |o| self.owner_of_raw(o), &self.world.parents);
                    if let Err(reason) = verdict {
                        self.violation(from, reason, "parent");
                        return;
                    }
                    self.broadcast_reliable(m, Some(from));
                }
                self.world.parents.insert(id, parent);
                let exp = self.now() + PENDING_TTL;
                self.pending_parents.push((id, parent, exp));
            }
            M_ANCHOR if host => {
                let id = c.u128();
                let Some(auth) = m.get(17..).and_then(crate::net_predict::decode_auth) else {
                    return;
                };
                if c.bad || self.owner_of_raw(id) != from {
                    return;
                }
                self.anchor_wait.remove(&id);
                self.inputs.entry(id).or_default().rebase(auth.ack);
                self.anchors.insert(id, auth);
            }
            M_RPC => self.on_rpc(from, &m[1..], true),
            M_TEXT => {
                let target = c.u64();
                let sender = c.u64();
                let mid = c.u32();
                let text = c.string();
                if c.bad {
                    return;
                }
                // The host knows who really sent it.
                let sender = if host { from } else { sender };
                if host {
                    if !self.val.chat_ok(&text) {
                        self.violation(from, Reason::TooLarge, "chat line");
                        return;
                    }
                    if !self.gate(from, Limit::Chat) {
                        return;
                    }
                    if target != 0 && !self.peers.contains_key(&target) {
                        self.violation(from, Reason::Malformed, "message to a peer that is not here");
                        return;
                    }
                }
                if !self.fresh_mid(sender, mid) {
                    return;
                }
                if host {
                    self.route_text(target, sender, mid, &text, Some(from));
                } else {
                    self.inbox.push((sender, target != 0, text));
                }
            }
            M_SCENE if !host => {
                let path = c.string();
                if c.bad {
                    return;
                }
                self.clear_world_for_scene(&path);
            }
            M_XFORM if !host => {
                let Some(e) = dec_entry(&mut c) else {
                    return;
                };
                if self.resync.is_some() && self.has_authority_raw(e.id) {
                    return;
                }
                let x = XformState {
                    pos: e.pos.unwrap_or(Vec3::ZERO),
                    rot: e.rot.unwrap_or(Quat::IDENTITY),
                    scale: e.scale.unwrap_or(Vec3::ONE),
                    vel: Vec3::ZERO,
                };
                self.world.xforms.insert(e.id, x);
                let b = self.buffers.entry(e.id).or_default();
                if b.snaps.is_empty() {
                    b.baseline = true;
                    b.snaps.push_back(Snap { time: 0.0, pos: x.pos, rot: x.rot, scale: x.scale, vel: Vec3::ZERO });
                }
            }
            M_AVPARAM if !host => {
                let id = c.u128();
                let n = c.u16() as usize;
                let packet = c.take(n).to_vec();
                if !c.bad {
                    self.avparam_seq += 1;
                    self.world.avparams.insert(id, packet.clone());
                    self.avparam_in.insert(id, (self.avparam_seq, packet));
                }
            }
            M_NETID if !host && from == self.host_peer => self.on_netid(&mut c),
            M_ANTICHEAT => self.on_anticheat(from, &m[1..]),
            M_CORRECT if !host && from == self.host_peer => {
                let id = c.u128();
                let pos = c.vec3();
                let rot = c.quat();
                if !c.bad && self.has_authority_raw(id) && pos.is_finite() && rot.is_finite() {
                    if let Some(x) = self.world.xforms.get_mut(&id) {
                        x.pos = pos;
                        x.rot = rot;
                    }
                    if self.corrections.len() < MAX_QUEUE {
                        self.corrections.push((id, pos, rot));
                    }
                }
            }
            M_ANIM if !host => {
                let id = c.u128();
                let n = c.u16();
                let mut params = HashMap::new();
                for _ in 0..n {
                    let k = c.string();
                    let v = c.f32();
                    params.insert(k, v);
                }
                if !c.bad {
                    self.world.anims.insert(id, params.clone());
                    self.anim_in.insert(id, params);
                }
            }
            _ => {}
        }
    }

    /// Client: the new host's world is in: drop what it doesn't know (ours
    /// excepted: re-sent), and re-assert the variables of what we own.
    fn finish_resync(&mut self) {
        let Some(r) = self.resync.take() else {
            return;
        };
        let me = self.local_peer;
        let unknown: Vec<SpawnRecord> = self.world.spawns.iter().filter(|s| !r.spawns.contains(&s.id)).cloned().collect();
        let mut reassert: Vec<u128> = Vec::new();
        for s in unknown {
            if s.owner == me || r.mine_before.iter().any(|m| m.id == s.id) {
                let rec = SpawnRecord { owner: me, ..s.clone() };
                self.world.owners.insert(rec.id, (me, 0));
                let host = self.host_peer;
                self.send_reliable(host, &enc_spawn(&rec));
                reassert.push(rec.id);
            } else {
                self.apply_despawn(s.id);
            }
        }
        let dropped: Vec<u128> = self.world.owners.keys().copied().filter(|id| !r.owners.contains(id) && !reassert.contains(id)).collect();
        for id in dropped {
            if let Some((prev, _)) = self.world.owners.remove(&id)
                && prev != 0
            {
                self.push_event(NetEvent::OwnershipChanged { object: ObjectId::from_raw(id), owner: 0, previous: prev });
            }
        }
        let owned: Vec<u128> = self.world.owners.iter().filter(|(_, (o, _))| *o == me).map(|(id, _)| *id).collect();
        let host = self.host_peer;
        // Predicted controllers we own: the new host continues from our state.
        let anchors: Vec<(u128, AuthState)> =
            self.predictors.iter().filter(|(id, _)| owned.contains(id)).filter_map(|(id, p)| p.newest().map(|a| (*id, a))).collect();
        for (id, a) in anchors {
            let mut m = vec![M_ANCHOR];
            put_u128(&mut m, id);
            crate::net_predict::encode_auth(&mut m, &a);
            self.send_reliable(host, &m);
        }
        for id in owned {
            let vars = self.world.vars.get(&id).cloned().unwrap_or_default();
            for (k, v) in vars {
                let mut m = enc_var(id, &k, &v, me, 0);
                m[0] = M_VAR_SYNC;
                self.send_reliable(host, &m);
            }
        }
        tracing::info!("net: resynced with host {host}");
    }

    fn clear_world_for_scene(&mut self, path: &str) {
        self.world = NetWorld { scene: Some(path.to_owned()), ..Default::default() };
        self.inputs.clear();
        self.auth_in.clear();
        self.predictors.clear();
        self.delays.clear();
        *self.history.lock().unwrap() = self.fresh_history();
        self.buffers.clear();
        self.anim_in.clear();
        self.avparam_in.clear();
        self.avparam_state.clear();
        self.avatar_in.clear();
        self.face_in.clear();
        self.send_state.clear();
        self.pending_vars.clear();
        self.pending_parents.clear();
        self.world_ops.retain(|op| matches!(op, NetWorldOp::LoadScene(_)));
        self.world_ops.push(NetWorldOp::LoadScene(path.to_owned()));
        self.hold_ops_until = self.now() + 0.5;
        if self.is_host {
            self.host_player_done = false;
            self.players_to_spawn = self.world_members();
        }
    }

    // ------------------------------------------------------------ ownership

    fn set_owner_local(&mut self, id: u128, owner: u64, seq: u64) {
        let prev = self.owner_of_raw(id);
        if let Some((_, s)) = self.world.owners.get(&id)
            && *s > seq
        {
            return;
        }
        self.world.owners.insert(id, (owner, seq));
        if prev != owner {
            self.val.forget_object(id);
            self.inputs.remove(&id);
            self.auth_in.remove(&id);
            self.predictors.remove(&id);
            self.push_event(NetEvent::OwnershipChanged { object: ObjectId::from_raw(id), owner, previous: prev });
        }
    }

    /// Host: assign ownership and tell everyone.
    fn grant(&mut self, id: u128, owner: u64) {
        self.claim_seq += 1;
        let seq = self.claim_seq;
        self.set_owner_local(id, owner, seq);
        let mut m = vec![M_OWNER];
        put_u128(&mut m, id);
        put_u64(&mut m, owner);
        put_u64(&mut m, seq);
        self.broadcast_reliable(&m, None);
    }

    fn handle_claim(&mut self, id: u128, by: u64) {
        let cur = self.owner_of_raw(id);
        let mode = self.world.modes.get(&id).copied().unwrap_or(OwnershipMode::Takeover);
        let verdict = if by == self.local_peer {
            // The host's own claims are the server's decision: the old rules.
            match mode {
                _ if cur == by => Claim::Grant,
                OwnershipMode::Takeover | OwnershipMode::Grabbable | OwnershipMode::Fixed => Claim::Grant,
                OwnershipMode::Request if cur == 0 => Claim::Grant,
                OwnershipMode::Request => Claim::Ask,
            }
        } else {
            let now = self.now();
            let known = self.world.modes.contains_key(&id) || self.world.owners.contains_key(&id) || self.world.spawns.iter().any(|s| s.id == id);
            let protected = self.world.predicted.contains(&id) || self.world.player_obj.values().any(|p| *p == id);
            let owner_silent = cur != 0 && cur != self.local_peer && self.conns.get(&cur).is_some_and(|c| now - c.last_recv > self.val.settings().owner_silent as f64);
            let ctx = ClaimCtx {
                id,
                by,
                cur,
                mode,
                host_peer: self.host_peer,
                known,
                protected,
                owner_silent,
                requester: self.avatar_pos(by),
                object: self.world.xforms.get(&id).map(|x| x.pos),
            };
            let v = self.val.claim(&ctx, now);
            if let Claim::Deny(why) = v
                && (self.val.strict() || mode == OwnershipMode::Grabbable)
            {
                let reason = if why == DenyWhy::Cooldown { Reason::RateLimited } else { Reason::OwnershipDenied };
                self.violation(by, reason, &format!("{why:?}"));
            }
            v
        };
        if verdict == Claim::Grant {
            self.grant(id, by);
            return;
        }
        if verdict == Claim::Ask {
            if cur == self.local_peer {
                self.push_event(NetEvent::OwnershipRequested { object: ObjectId::from_raw(id), by });
            } else {
                let mut m = vec![M_OWNER_REQ];
                put_u128(&mut m, id);
                put_u64(&mut m, by);
                self.send_reliable(cur, &m);
            }
        }
        if by == self.local_peer {
            self.push_event(NetEvent::OwnershipDenied { object: ObjectId::from_raw(id) });
        } else {
            let mut m = vec![M_DENY];
            put_u128(&mut m, id);
            self.send_reliable(by, &m);
        }
    }

    /// Host: whether a client may spawn this (a violation, and `false`, when not).
    fn allow_client_spawn(&mut self, from: u64, rec: &SpawnRecord) -> bool {
        // A spawn may not reuse the id of an object that exists (it would take it over).
        if self.world.owners.contains_key(&rec.id) || self.world.modes.contains_key(&rec.id) {
            self.violation(from, Reason::Forged, "spawn with the id of an existing object");
            return false;
        }
        let live = self.world.spawns.iter().filter(|s| s.owner == from && !s.player).count();
        let requester = self.avatar_pos(from);
        let prefab = match &rec.source {
            NetSpawnSource::Prefab(p) => Some(p.as_str()),
            NetSpawnSource::Clone(_) => None,
        };
        let same = |s: &&SpawnRecord| matches!((&s.source, prefab), (NetSpawnSource::Prefab(p), Some(q)) if p == q);
        let live_prefab = self.world.spawns.iter().filter(|s| s.owner == from && !s.player).filter(same).count();
        let live_prefab_total = self.world.spawns.iter().filter(|s| s.owner != self.host_peer && !s.player).filter(same).count();
        let req = SpawnReq { prefab, pos: rec.translation, rot: rec.rotation, live, live_prefab, live_prefab_total, requester };
        let now = self.now();
        match self.val.spawn(from, &req, now) {
            Ok(()) => true,
            Err(reason) => {
                let what = prefab.unwrap_or("clone");
                self.violation(from, reason, what);
                false
            }
        }
    }

    /// Host: size and content limits for a variable write a client sent.
    fn admit_var(&mut self, from: u64, key: &str, value: &NetValue) -> bool {
        let bytes = match value {
            NetValue::Str(s) => s.len(),
            NetValue::Bytes(b) => b.len(),
            _ => 0,
        };
        if !self.val.var_ok(key, bytes) {
            self.violation(from, Reason::TooLarge, "variable");
            return false;
        }
        let finite = match value {
            NetValue::Float(f) => f.is_finite(),
            NetValue::Vec3(v) => v.iter().all(|f| f.is_finite()),
            NetValue::Quat(q) => q.iter().all(|f| f.is_finite()),
            _ => true,
        };
        if !finite {
            self.violation(from, Reason::NotFinite, "variable");
            return false;
        }
        if !self.val.var_budget(from, key, bytes) {
            self.violation(from, Reason::TooLarge, "variable budget");
            return false;
        }
        self.gate(from, Limit::Var)
    }

    /// Local peer claims authority over an object.
    pub fn request_ownership(&mut self, id: ObjectId) {
        if !self.is_connected() {
            return;
        }
        if self.is_host {
            let me = self.local_peer;
            self.handle_claim(id.raw(), me);
        } else {
            let mut m = vec![M_CLAIM];
            put_u128(&mut m, id.raw());
            let host = self.host_peer;
            self.send_reliable(host, &m);
        }
    }

    /// Local peer releases authority over an object.
    pub fn release_ownership(&mut self, id: ObjectId) {
        if !self.is_connected() || self.owner_of(id) != self.local_peer {
            return;
        }
        if self.is_host {
            self.grant(id.raw(), 0);
        } else {
            let mut m = vec![M_RELEASE];
            put_u128(&mut m, id.raw());
            let host = self.host_peer;
            self.send_reliable(host, &m);
        }
    }

    /// Hand an object to `to` (current owner, or the host for anything).
    pub fn transfer_ownership(&mut self, id: ObjectId, to: u64) {
        if !self.is_connected() {
            return;
        }
        if self.is_host {
            self.grant(id.raw(), to);
        } else if self.owner_of(id) == self.local_peer {
            let mut m = vec![M_TRANSFER];
            put_u128(&mut m, id.raw());
            put_u64(&mut m, to);
            let host = self.host_peer;
            self.send_reliable(host, &m);
        }
    }

    // ------------------------------------------------------------ spawning

    /// Replicate a spawn the local play session already applied.
    pub fn spawn(&mut self, source: NetSpawnSource, root_id: ObjectId, translation: Vec3, rotation: Quat, owner: u64) {
        let rec = SpawnRecord { id: root_id.raw(), source, translation, rotation, owner, player: false };
        self.record_spawn(&rec);
        if self.is_host {
            self.claim_seq += 1;
            let seq = self.claim_seq;
            self.set_owner_local(rec.id, rec.owner, seq);
            self.broadcast_spawn(&rec, None);
            self.broadcast_owner(rec.id);
        } else if self.is_connected() {
            // Optimistic: the spawner simulates it until the host says otherwise.
            self.world.owners.insert(rec.id, (rec.owner, 0));
            let host = self.host_peer;
            self.send_reliable(host, &enc_spawn(&rec));
        }
        self.push_event(NetEvent::Spawned { object: root_id, owner: rec.owner });
    }

    /// Host: spawn a peer's player object on every peer, the host included.
    pub fn spawn_everywhere(&mut self, source: NetSpawnSource, translation: Vec3, rotation: Quat, owner: u64) -> ObjectId {
        let id = ObjectId::new();
        let rec = SpawnRecord { id: id.raw(), source, translation, rotation, owner, player: true };
        self.forward_spawn(rec, None);
        id
    }

    /// Host: spawn a prefab instance of the server's own (unowned, not a player object) on every
    /// peer, the host's scene included. For a server restoring persistent objects.
    pub fn spawn_unowned(&mut self, source: NetSpawnSource, translation: Vec3, rotation: Quat) -> ObjectId {
        let id = ObjectId::new();
        let rec = SpawnRecord { id: id.raw(), source, translation, rotation, owner: 0, player: false };
        self.forward_spawn(rec, None);
        id
    }

    fn broadcast_owner(&mut self, id: u128) {
        let (owner, seq) = self.world.owners.get(&id).copied().unwrap_or((0, 0));
        let mut m = vec![M_OWNER];
        put_u128(&mut m, id);
        put_u64(&mut m, owner);
        put_u64(&mut m, seq);
        self.broadcast_reliable(&m, None);
    }

    /// Host: accept a spawn (from a client or itself), apply it locally and
    /// relay it to everyone except `except` (who already has it).
    fn forward_spawn(&mut self, rec: SpawnRecord, except: Option<u64>) {
        self.broadcast_spawn(&rec, except);
        self.claim_seq += 1;
        let seq = self.claim_seq;
        let id = rec.id;
        self.apply_spawn_with_seq(rec, seq);
        self.broadcast_owner(id);
    }

    fn record_spawn(&mut self, rec: &SpawnRecord) {
        self.world.spawns.retain(|s| s.id != rec.id);
        self.world.spawns.push(rec.clone());
    }

    fn apply_spawn(&mut self, rec: SpawnRecord) {
        self.apply_spawn_with_seq(rec, 0);
    }

    fn apply_spawn_with_seq(&mut self, rec: SpawnRecord, seq: u64) {
        if self.world.spawns.iter().any(|s| s.id == rec.id) {
            return;
        }
        self.record_spawn(&rec);
        self.set_owner_local(rec.id, rec.owner, seq);
        self.world_ops.push(NetWorldOp::Spawn {
            source: rec.source.clone(),
            root_id: ObjectId::from_raw(rec.id),
            translation: rec.translation,
            rotation: rec.rotation,
            owner: rec.owner,
        });
        self.push_event(NetEvent::Spawned { object: ObjectId::from_raw(rec.id), owner: rec.owner });
    }

    /// Replicate a despawn the local play session already applied.
    pub fn despawn(&mut self, id: ObjectId) {
        let raw = id.raw();
        if !self.is_connected() {
            return;
        }
        let mut d = vec![M_DESPAWN];
        put_u128(&mut d, raw);
        self.upstream_or_broadcast(&d);
        self.world.forget(raw);
        self.buffers.remove(&raw);
        self.send_state.remove(&raw);
        self.push_event(NetEvent::Despawned { object: id });
    }

    /// Host: despawn on every peer, the host included.
    pub fn despawn_raw(&mut self, id: u128) {
        let mut d = vec![M_DESPAWN];
        put_u128(&mut d, id);
        self.broadcast_reliable(&d, None);
        self.apply_despawn(id);
    }

    fn apply_despawn(&mut self, id: u128) {
        self.val.forget_object(id);
        self.interest_forget(id);
        self.world.forget(id);
        self.forget_prediction(id);
        self.buffers.remove(&id);
        self.send_state.remove(&id);
        self.anim_in.remove(&id);
        self.avparam_in.remove(&id);
        self.avparam_state.remove(&id);
        self.avatar_in.remove(&id);
        self.face_in.remove(&id);
        self.world_ops.push(NetWorldOp::Despawn(ObjectId::from_raw(id)));
        self.push_event(NetEvent::Despawned { object: ObjectId::from_raw(id) });
    }

    // ------------------------------------------------------------ variables / state

    /// Replicate a synced-variable write the local play session already applied.
    pub fn set_var(&mut self, id: ObjectId, key: &str, value: NetValue) {
        let raw = id.raw();
        self.world.set_var(raw, key, value.clone());
        if self.is_connected() {
            let local = self.local_peer;
            let mid = self.next_mid();
            self.upstream_or_broadcast(&enc_var(raw, key, &value, local, mid));
        }
        self.push_event(NetEvent::VarChanged { object: id, key: key.to_owned(), value, from: self.local_peer });
    }

    fn apply_var(&mut self, id: u128, key: String, value: NetValue, from: u64) {
        self.world.set_var(id, &key, value.clone());
        let expires = self.now() + PENDING_TTL;
        self.pending_vars.retain(|p| !(p.id == id && p.key == key));
        self.pending_vars.push(PendingVar { id, key: key.clone(), value: value.clone(), expires });
        self.push_event(NetEvent::VarChanged { object: ObjectId::from_raw(id), key, value, from });
    }

    pub fn set_active(&mut self, id: u128, on: bool) {
        self.world.actives.insert(id, on);
        let mut m = vec![M_ACTIVE];
        put_u128(&mut m, id);
        m.push(on as u8);
        self.upstream_or_broadcast(&m);
    }

    /// `parent` 0 = root.
    pub fn set_parent(&mut self, id: u128, parent: u128) {
        self.world.parents.insert(id, parent);
        let mut m = vec![M_PARENT];
        put_u128(&mut m, id);
        put_u128(&mut m, parent);
        self.upstream_or_broadcast(&m);
    }

    /// Host: switch every peer (and late joiners) to a scene.
    pub fn load_scene(&mut self, path: &str) {
        if !self.is_host {
            return;
        }
        let mut m = vec![M_SCENE];
        put_str(&mut m, path);
        self.broadcast_reliable(&m, None);
        self.clear_world_for_scene(path);
        self.hold_ops_until = 0.0;
    }

    // ------------------------------------------------------------ RPC / text

    pub fn send_rpc(&mut self, object: Option<ObjectId>, name: &str, args: &[NetValue], target: RpcTarget, reliable: bool) {
        let me = self.local_peer;
        let obj = object.map(|o| o.raw()).unwrap_or(0);
        let mid = if reliable && self.is_connected() { self.next_mid() } else { 0 };
        let body = enc_rpc(me, mid, target, obj, name, args);
        let rpc = NetRpc { from: me, object, name: name.to_owned(), args: args.to_vec() };
        let owner = match self.owner_of_raw(obj) {
            0 => self.host_peer,
            o => o,
        };
        let local = match target {
            RpcTarget::All => true,
            RpcTarget::Others => false,
            RpcTarget::Server => self.is_host,
            RpcTarget::Owner => owner == me,
            RpcTarget::Peer(p) => p == me,
        };
        if local {
            self.rpcs.push(rpc);
        }
        if !self.is_connected() {
            return;
        }
        if self.is_host {
            self.route_rpc(me, target, obj, &body, reliable, None);
        } else if !local || target == RpcTarget::All {
            let host = self.host_peer;
            self.send_rpc_packet(host, &body, reliable);
        }
    }

    fn send_rpc_packet(&mut self, peer: u64, body: &[u8], reliable: bool) {
        if reliable {
            let mut m = Vec::with_capacity(body.len() + 1);
            m.push(M_RPC);
            m.extend_from_slice(body);
            self.send_reliable(peer, &m);
        } else {
            let mut p = Vec::with_capacity(body.len() + 1);
            p.push(P_URPC);
            p.extend_from_slice(body);
            self.send_to_peer(peer, &p);
        }
    }

    /// Host: whether a member (not the host itself) receives an RPC.
    fn rpc_recipient(&self, sender: u64, target: RpcTarget, obj: u128, peer: u64) -> bool {
        if peer == sender || peer == self.local_peer {
            return false;
        }
        match target {
            RpcTarget::All | RpcTarget::Others => true,
            RpcTarget::Server => false,
            RpcTarget::Owner => self.owner_of_raw(obj) == peer,
            RpcTarget::Peer(p) => p == peer,
        }
    }

    /// Host: deliver an RPC to the members its target names (the host itself
    /// was handled by the caller), skipping `origin`. Reliable ones are kept
    /// for members still reconnecting after a migration.
    fn route_rpc(&mut self, sender: u64, target: RpcTarget, obj: u128, body: &[u8], reliable: bool, origin: Option<u64>) {
        let to: Vec<u64> = self.world_members().into_iter().filter(|p| self.rpc_recipient(sender, target, obj, *p)).collect();
        for p in to {
            if Some(p) != origin {
                self.send_rpc_packet(p, body, reliable);
            }
        }
        if reliable && !self.reconnecting.is_empty() {
            let mut m = vec![M_RPC];
            m.extend_from_slice(body);
            self.backlog.push(Backlog { sender, msg: m });
        }
    }

    /// Host: permission, limits and interaction range of an RPC a client sent.
    fn admit_rpc(&mut self, from: u64, name: &str, args: &[NetValue], obj: u128, bytes: usize) -> bool {
        let now = self.now();
        let owner = self.owner_of_raw(obj);
        if self.muted(from) {
            return false;
        }
        match self.val.rpc(from, self.host_peer, name, args.len(), bytes, obj != 0, owner, now) {
            RpcVerdict::Allow => {}
            RpcVerdict::Deny(reason) => {
                self.violation(from, reason, name);
                return false;
            }
        }
        if self.val.strict() && self.val.needs_interaction(name) {
            // The user object named in the arguments must be the sender's own, and near.
            let by = match args.first() {
                Some(NetValue::Bytes(b)) if b.len() == 16 => b.as_slice().try_into().ok().map(u128::from_be_bytes),
                _ => None,
            };
            let ok = by.is_some_and(|by| {
                self.owner_of_raw(by) == from && self.val.interaction_ok(self.world.xforms.get(&by).map(|x| x.pos), self.world.xforms.get(&obj).map(|x| x.pos))
            });
            if !ok {
                self.violation(from, Reason::RpcDenied, name);
                return false;
            }
        }
        true
    }

    fn on_rpc(&mut self, from: u64, body: &[u8], reliable: bool) {
        let mut c = Cursor::new(body, 0);
        let Some((sender, mid, target, obj, name, args)) = dec_rpc(&mut c) else {
            return;
        };
        let sender = if self.is_host { from } else { sender };
        if reliable && !self.fresh_mid(sender, mid) {
            return;
        }
        if self.is_host && from != self.local_peer && !self.admit_rpc(from, &name, &args, obj, body.len()) {
            return;
        }
        if self.is_host && name == REPORT_RPC {
            // Consumed here: a report goes to the server's queue, never to other players.
            if from != self.local_peer {
                self.on_report(from, &args);
            }
            return;
        }
        let rpc = NetRpc {
            from: sender,
            object: (obj != 0).then(|| ObjectId::from_raw(obj)),
            name,
            args,
        };
        if self.is_host {
            let owner = match self.owner_of_raw(obj) {
                0 => self.host_peer,
                o => o,
            };
            let to_host = match target {
                RpcTarget::All | RpcTarget::Others | RpcTarget::Server => true,
                RpcTarget::Owner => owner == self.local_peer,
                RpcTarget::Peer(p) => p == self.local_peer,
            };
            let fwd = enc_rpc(sender, mid, target, obj, &rpc.name, &rpc.args);
            self.route_rpc(sender, target, obj, &fwd, reliable, Some(from));
            if to_host {
                self.rpcs.push(rpc);
            }
        } else {
            self.rpcs.push(rpc);
        }
        if self.rpcs.len() > MAX_QUEUE {
            self.rpcs.remove(0);
        }
    }

    /// Report another player to the server's operators (SEC): the reason is logged and queued
    /// for [`NetSession::take_reports`]. The host files it directly; a client sends the
    /// `zestengine.report` RPC, which any component can also call itself.
    pub fn report_player(&mut self, target: u64, reason: &str) {
        let args = [NetValue::Int(target as i64), NetValue::Str(reason.to_owned())];
        if self.is_host {
            let me = self.local_peer;
            self.file_report(me, &args);
        } else {
            self.send_rpc(None, crate::net_validate::REPORT_RPC, &args, RpcTarget::Server, true);
        }
    }

    /// Host: reports filed since the last call.
    pub fn take_reports(&mut self) -> Vec<PlayerReport> {
        std::mem::take(&mut self.reports)
    }

    fn on_report(&mut self, from: u64, args: &[NetValue]) {
        if !self.file_report(from, args) {
            self.violation(from, Reason::Malformed, "report");
        }
    }

    /// Validate a report and queue it. `false` = the arguments or the target are not valid.
    fn file_report(&mut self, from: u64, args: &[NetValue]) -> bool {
        let (Some(NetValue::Int(target)), Some(NetValue::Str(reason))) = (args.first(), args.get(1)) else {
            return false;
        };
        let target = *target as u64;
        let Some(target_name) = self.peers.get(&target).cloned().filter(|_| target != from) else {
            return false;
        };
        if !self.val.chat_ok(reason) {
            return false;
        }
        let report = PlayerReport {
            reporter: from,
            reporter_name: self.peers.get(&from).cloned().unwrap_or_default(),
            reporter_fingerprint: self.peer_fingerprint(from).unwrap_or_default(),
            target,
            target_name,
            target_fingerprint: self.peer_fingerprint(target).unwrap_or_default(),
            reason: reason.chars().filter(|c| !c.is_control()).collect(),
            at: self.now(),
        };
        tracing::warn!(
            "net: report by {} ({}) against {} ({}): {}",
            report.reporter_name,
            short_fingerprint(&report.reporter_fingerprint),
            report.target_name,
            short_fingerprint(&report.target_fingerprint),
            report.reason
        );
        if self.reports.len() >= MAX_REPORTS {
            self.reports.remove(0);
        }
        self.reports.push(report);
        true
    }

    /// Send a text message: `to = None` broadcasts (public), `Some(peer)` is
    /// private. Routed through the host; never echoed to the sender.
    pub fn send_message(&mut self, to: Option<u64>, text: &str) {
        if !self.is_connected() {
            return;
        }
        let target = to.unwrap_or(0);
        let mid = self.next_mid();
        if self.is_host {
            let me = self.local_peer;
            self.route_text(target, me, mid, text, None);
        } else {
            let mut m = vec![M_TEXT];
            put_u64(&mut m, target);
            put_u64(&mut m, self.local_peer);
            put_u32(&mut m, mid);
            put_str(&mut m, text);
            let host = self.host_peer;
            self.send_reliable(host, &m);
        }
    }

    fn route_text(&mut self, target: u64, sender: u64, mid: u32, text: &str, origin: Option<u64>) {
        let mut m = vec![M_TEXT];
        put_u64(&mut m, target);
        put_u64(&mut m, sender);
        put_u32(&mut m, mid);
        put_str(&mut m, text);
        if target == 0 {
            let to: Vec<u64> = self.world_members().into_iter().filter(|p| *p != sender && Some(*p) != origin).collect();
            for p in to {
                self.send_reliable(p, &m);
            }
            if sender != self.local_peer {
                self.inbox.push((sender, false, text.to_owned()));
            }
        } else if target == self.local_peer {
            self.inbox.push((sender, true, text.to_owned()));
        } else {
            self.send_reliable(target, &m);
        }
        if !self.reconnecting.is_empty() {
            self.backlog.push(Backlog { sender, msg: m });
        }
    }

    // ------------------------------------------------------------ unreliable state

    /// Send transform entries for objects the local peer has authority over.
    /// The host filters by relevancy per client.
    pub fn send_snapshot(&mut self, entries: &[SnapEntry]) {
        if entries.is_empty() || !self.is_connected() {
            return;
        }
        let now = self.now();
        let origin = self.local_peer;
        if self.interest_on() {
            self.interest_route(origin, now, entries, None, true);
            self.interest_flush(now);
        } else if self.is_host {
            for c in self.world_members() {
                let list: Vec<SnapEntry> = entries
                    .iter()
                    .copied()
                    .filter(|e| self.relevant(e.id, c) && !(self.world.predicted.contains(&e.id) && self.owner_of_raw(e.id) == c))
                    .collect();
                for p in build_snap_packets(origin, now, &list) {
                    self.send_to_peer(c, &p);
                }
            }
        } else {
            let host = self.host_peer;
            if self.interest.short_out.is_empty() {
                for p in build_snap_packets(origin, now, entries) {
                    self.send_to_peer(host, &p);
                }
            } else {
                // The host taught us 32-bit ids for what we can see: use them upstream too.
                let short = &self.interest.short_out;
                let mapped: Vec<SnapEntry> = entries
                    .iter()
                    .map(|e| SnapEntry { id: short.get(&e.id).map_or(e.id, |s| *s as u128), ..*e })
                    .collect();
                for p in build_upstream_packets(origin, now, &mapped) {
                    self.send_to_peer(host, &p);
                }
            }
        }
    }

    /// Host-side interest management: is `id` worth sending to `peer`?
    fn relevant(&self, id: u128, peer: u64) -> bool {
        let range = self.world.relevancy.get(&id).copied().unwrap_or(0.0);
        if range <= 0.0 || self.owner_of_raw(id) == peer {
            return true;
        }
        let Some(player) = self.world.player_obj.get(&peer) else {
            return true;
        };
        match (self.world.xforms.get(&id), self.world.xforms.get(player)) {
            (Some(a), Some(b)) => a.pos.distance(b.pos) <= range,
            _ => true,
        }
    }

    fn on_snap(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let origin = c.u64();
        let time = c.f64();
        let n = c.u8();
        let from_client = self.is_host && peer != self.local_peer;
        if from_client {
            if n as usize > self.val.max_entries() {
                self.violation(peer, Reason::TooLarge, "snapshot entries");
                return;
            }
            if !self.gate(peer, Limit::Snapshot) {
                return;
            }
        }
        let mut entries = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match dec_entry(&mut c) {
                Some(e) => entries.push(e),
                None => {
                    if from_client {
                        self.violation(peer, Reason::Malformed, "snapshot entry");
                    }
                    return;
                }
            }
        }
        self.snap_in(peer, origin, time, entries);
    }

    /// An upstream snapshot with 32-bit aliases (`P_SNAPB` from a client): one group, ids
    /// resolved against what the host taught this client, then the same path as `P_SNAP`.
    fn on_snap_up(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let groups = c.u8();
        let origin = c.u64();
        let time = c.f64();
        let n = c.u8();
        if c.bad || groups != 1 {
            self.violation(peer, Reason::Malformed, "upstream snapshot");
            return;
        }
        if n as usize > self.val.max_entries() {
            self.violation(peer, Reason::TooLarge, "snapshot entries");
            return;
        }
        if !self.gate(peer, Limit::Snapshot) {
            return;
        }
        let mut entries = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match dec_entry_b(&mut c) {
                Some(e) => entries.push(e),
                None => {
                    self.violation(peer, Reason::Malformed, "snapshot entry");
                    return;
                }
            }
        }
        self.resolve_upstream(peer, &mut entries);
        self.snap_in(peer, origin, time, entries);
    }

    /// The part of snapshot handling after the entries are decoded.
    fn snap_in(&mut self, peer: u64, origin: u64, time: f64, mut entries: Vec<SnapEntry>) {
        let from_client = self.is_host && peer != self.local_peer;
        let origin = if self.is_host { peer } else { origin };
        if self.is_host {
            // Only the owner may move an object, and nobody but the host a predicted one.
            entries.retain(|e| self.owner_of_raw(e.id) == peer && !self.world.predicted.contains(&e.id));
        }
        if from_client {
            match self.val.stamp(peer, time) {
                Stamp::Fresh => {}
                Stamp::Stale => return,
                Stamp::Bad => {
                    self.violation(peer, Reason::BadTime, "snapshot time");
                    return;
                }
            }
            self.validate_entries(peer, &mut entries);
        }
        self.snap_ingest(origin, time, &entries);
        if self.interest_on() && !entries.is_empty() {
            // Batched with everything else due this pump (`interest_tick`).
            self.interest_route(origin, time, &entries, Some(peer), false);
        } else if self.is_host && !entries.is_empty() {
            let clients: Vec<u64> = self.world_members().into_iter().filter(|p| *p != peer).collect();
            for cl in clients {
                let list: Vec<SnapEntry> = entries.iter().copied().filter(|e| self.relevant(e.id, cl)).collect();
                for p in build_snap_packets(origin, time, &list) {
                    self.send_to_peer(cl, &p);
                }
            }
        }
    }

    /// Take received snapshot entries into the buffers (jitter, clock offset, samples).
    fn snap_ingest(&mut self, origin: u64, time: f64, entries: &[SnapEntry]) {
        let now = self.now();
        self.jitter.entry(origin).or_default().on_packet(time, now);
        let sample = now - time;
        let off = self.offsets.entry(origin).or_insert(sample);
        if sample < *off {
            *off = sample;
        } else {
            *off += (sample - *off) * 0.02;
        }
        for e in entries {
            if self.has_authority_raw(e.id) {
                continue;
            }
            self.insert_snapshot(origin, time, e);
        }
    }

    /// Client: a batch from the host (groups of one origin and time each).
    fn on_snapb(&mut self, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let groups = c.u8();
        for _ in 0..groups {
            let origin = c.u64();
            let time = c.f64();
            let n = c.u8();
            if c.bad {
                return;
            }
            let mut entries = Vec::with_capacity(n as usize);
            for _ in 0..n {
                match dec_entry_b(&mut c) {
                    Some(e) => entries.push(e),
                    None => return,
                }
            }
            interest::resolve_ids(&self.interest.short_in, &mut entries);
            self.snap_ingest(origin, time, &entries);
        }
    }

    /// Host: drop the snapshot entries a client may not send (non-finite or
    /// out of bounds values, teleports, speed hacks) and tell the owner where
    /// the object really is.
    fn validate_entries(&mut self, peer: u64, entries: &mut Vec<SnapEntry>) {
        let now = self.now();
        let player = self.world.player_obj.get(&peer).copied();
        let mut i = 0;
        while i < entries.len() {
            let e = entries[i];
            let kind = if Some(e.id) == player { Kind::Avatar } else { Kind::Prop };
            let baseline = self.world.xforms.get(&e.id).map(|x| x.pos);
            match self.val.entry(peer, e.id, kind, e.pos, e.rot, e.scale, e.vel, baseline, now) {
                Ok(()) => i += 1,
                Err(rej) => {
                    entries.remove(i);
                    self.violation(peer, rej.reason, "snapshot");
                    if let Some(pos) = rej.correct_to {
                        let rot = self.world.xforms.get(&e.id).map_or(Quat::IDENTITY, |x| x.rot);
                        self.send_correction(peer, e.id, pos, rot);
                    }
                }
            }
        }
    }

    /// Host: put an owner's object back where the host last accepted it.
    fn send_correction(&mut self, peer: u64, id: u128, pos: Vec3, rot: Quat) {
        let mut m = vec![M_CORRECT];
        put_u128(&mut m, id);
        put_vec3(&mut m, pos);
        put_quat(&mut m, rot);
        self.send_reliable(peer, &m);
    }

    fn insert_snapshot(&mut self, origin: u64, time: f64, e: &SnapEntry) {
        let base = self.world.xforms.get(&e.id).copied().unwrap_or_default();
        let buf = self.buffers.entry(e.id).or_default();
        if buf.origin != origin || buf.baseline {
            buf.snaps.clear();
            buf.origin = origin;
            buf.baseline = false;
        }
        if buf.snaps.iter().any(|s| s.time == time)
            || (buf.snaps.len() >= SNAP_KEEP && buf.snaps.front().is_some_and(|s| time < s.time))
        {
            return;
        }
        let prev = buf.snaps.back().copied();
        let from = prev.map(|p| XformState { pos: p.pos, rot: p.rot, scale: p.scale, vel: p.vel }).unwrap_or(base);
        let s = Snap {
            time,
            pos: e.pos.unwrap_or(from.pos),
            rot: e.rot.unwrap_or(from.rot),
            scale: e.scale.unwrap_or(from.scale),
            vel: e.vel.unwrap_or(Vec3::ZERO),
        };
        let at = buf.snaps.iter().position(|x| x.time > time).unwrap_or(buf.snaps.len());
        buf.snaps.insert(at, s);
        while buf.snaps.len() > SNAP_KEEP {
            buf.snaps.pop_front();
        }
        let newest = *buf.snaps.back().unwrap();
        self.world.xforms.insert(e.id, XformState { pos: newest.pos, rot: newest.rot, scale: newest.scale, vel: newest.vel });
        self.xform_seq += 1;
    }

    /// Latest remote transform for an object we don't own.
    pub fn remote_transform(&self, id: ObjectId) -> Option<RemoteTransform> {
        let s = self.buffers.get(&id.raw())?.snaps.back()?;
        Some(RemoteTransform { translation: s.pos, rotation: s.rot, scale: s.scale, seq: self.xform_seq })
    }

    /// Send the transform of an object the local peer owns to all other peers.
    pub fn send_transform(&mut self, id: ObjectId, t: Vec3, r: Quat, s: Vec3) {
        self.world.xforms.insert(id.raw(), XformState { pos: t, rot: r, scale: s, vel: Vec3::ZERO });
        self.send_snapshot(&[SnapEntry { id: id.raw(), pos: Some(t), rot: Some(r), scale: Some(s), vel: None, quantized: false }]);
    }

    /// Animator parameters of an object the local peer has authority over.
    pub fn send_anim(&mut self, id: u128, params: &[(String, f32)]) {
        if params.is_empty() || !self.is_connected() {
            return;
        }
        let anims = self.world.anims.entry(id).or_default();
        for (k, v) in params {
            anims.insert(k.clone(), *v);
        }
        let mut b = vec![P_ANIM];
        put_u64(&mut b, self.local_peer);
        put_u128(&mut b, id);
        let n = params.len().min(64);
        b.push(n as u8);
        for (k, v) in &params[..n] {
            put_str(&mut b, k);
            put_f32(&mut b, *v);
        }
        if b.len() <= MTU {
            self.fanout_obj(&b, None, id, Stream::State);
        }
    }

    fn on_anim(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let origin = c.u64();
        let id = c.u128();
        let n = c.u8();
        let from_client = self.is_host && peer != self.local_peer;
        if from_client && !self.admit_unreliable(peer, origin, Limit::Anim) {
            return;
        }
        let mut params = Vec::new();
        for _ in 0..n {
            let k = c.string();
            let v = c.f32();
            if from_client && (!v.is_finite() || !self.val.string_ok(&k)) {
                self.violation(peer, if v.is_finite() { Reason::TooLarge } else { Reason::NotFinite }, "animator parameter");
                return;
            }
            params.push((k, v));
        }
        if c.bad || (self.is_host && self.owner_of_raw(id) != peer) || self.has_authority_raw(id) {
            return;
        }
        let stored = self.world.anims.entry(id).or_default();
        let live = self.anim_in.entry(id).or_default();
        for (k, v) in params {
            stored.insert(k.clone(), v);
            live.insert(k, v);
        }
        if self.is_host {
            self.fanout_obj(data, Some(peer), id, Stream::State);
        }
    }

    /// Avatar parameters of an avatar the local peer has authority over:
    /// `packet` is `AvParamSchema::encode` output (schema hash + payload).
    pub fn send_avparam(&mut self, id: u128, packet: &[u8]) {
        if packet.is_empty() || !self.is_connected() {
            return;
        }
        self.world.avparams.insert(id, packet.to_vec());
        let mut b = vec![P_AVPARAM];
        put_u64(&mut b, self.local_peer);
        put_u128(&mut b, id);
        b.extend_from_slice(packet);
        if b.len() <= MTU {
            self.fanout_obj(&b, None, id, Stream::State);
        } else {
            tracing::warn!("avatar parameter packet is {} bytes, over the {MTU}-byte limit; not sent (lower the avatar's synced bits)", b.len());
        }
    }

    fn on_avparam(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let origin = c.u64();
        let id = c.u128();
        if self.is_host && peer != self.local_peer && !self.admit_unreliable(peer, origin, Limit::AvParam) {
            return;
        }
        if c.bad || (self.is_host && self.owner_of_raw(id) != peer) || self.has_authority_raw(id) {
            return;
        }
        let Some(packet) = data.get(1 + 8 + 16..).filter(|p| !p.is_empty()) else {
            return;
        };
        self.avparam_seq += 1;
        self.world.avparams.insert(id, packet.to_vec());
        self.avparam_in.insert(id, (self.avparam_seq, packet.to_vec()));
        if self.is_host {
            self.fanout_obj(data, Some(peer), id, Stream::State);
        }
    }

    /// Avatar part poses (world space, wire order) of an object we have authority over.
    pub fn send_avatar(&mut self, id: u128, poses: &[Option<(Vec3, Quat)>; TRACKER_SLOTS]) {
        if !self.is_connected() {
            return;
        }
        let mut b = vec![P_AVATAR];
        put_u64(&mut b, self.local_peer);
        put_f64(&mut b, self.now());
        put_u128(&mut b, id);
        let mask = poses.iter().enumerate().fold(0u16, |m, (i, p)| if p.is_some() { m | (1 << i) } else { m });
        put_u16(&mut b, mask);
        for (p, r) in poses.iter().flatten() {
            put_vec3(&mut b, *p);
            put_rot32(&mut b, *r);
        }
        self.fanout_obj(&b, None, id, Stream::Pose);
    }

    fn on_avatar(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let origin = c.u64();
        let time = c.f64();
        let id = c.u128();
        let mask = c.u16();
        let mut poses = [None; TRACKER_SLOTS];
        for (i, slot) in poses.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                let p = c.vec3();
                let r = c.rot32();
                *slot = Some((p, r));
            }
        }
        let from_client = self.is_host && peer != self.local_peer;
        if from_client && !self.admit_unreliable(peer, origin, Limit::Avatar) {
            return;
        }
        if c.bad || (self.is_host && self.owner_of_raw(id) != peer) || self.has_authority_raw(id) {
            return;
        }
        if from_client && !self.validate_pose(peer, id, time, &poses) {
            return;
        }
        let origin = if self.is_host { peer } else { origin };
        let now = self.now();
        let sample = now - time;
        let off = self.offsets.entry(origin).or_insert(sample);
        if sample < *off {
            *off = sample;
        }
        let buf = self.avatar_in.entry(id).or_default();
        if buf.origin != origin {
            buf.snaps.clear();
            buf.origin = origin;
        }
        let at = buf.snaps.iter().position(|x| x.time > time).unwrap_or(buf.snaps.len());
        buf.snaps.insert(at, AvatarSnap { time, poses });
        while buf.snaps.len() > SNAP_KEEP {
            buf.snaps.pop_front();
        }
        if self.is_host {
            self.fanout_obj(data, Some(peer), id, Stream::Pose);
        }
    }

    /// Face tracking of an avatar we have authority over.
    pub fn send_face(&mut self, id: u128, face: &zestengine_core::FaceFrame) {
        if !self.is_connected() {
            return;
        }
        let mut b = vec![P_FACE];
        put_u64(&mut b, self.local_peer);
        put_f64(&mut b, self.now());
        put_u128(&mut b, id);
        face.encode(&mut b);
        if b.len() <= MTU {
            self.fanout_obj(&b, None, id, Stream::Face);
        }
    }

    fn on_face(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let origin = c.u64();
        let time = c.f64();
        let id = c.u128();
        if self.is_host && peer != self.local_peer && (!self.admit_unreliable(peer, origin, Limit::Face) || !time.is_finite()) {
            return;
        }
        if c.bad || (self.is_host && self.owner_of_raw(id) != peer) || self.has_authority_raw(id) {
            return;
        }
        let Some((frame, _)) = data.get(1 + 8 + 8 + 16..).and_then(zestengine_core::FaceFrame::decode) else {
            return;
        };
        if self.face_in.get(&id).is_some_and(|(t, _)| *t > time) {
            return;
        }
        self.face_in.insert(id, (time, frame));
        if self.is_host {
            self.fanout_obj(data, Some(peer), id, Stream::Face);
        }
    }

    /// Send one encoded voice frame to everyone (relayed by the host).
    pub fn send_voice(&mut self, seq: u32, payload: &[u8]) {
        if !self.is_connected() || payload.len() > MTU - 20 {
            return;
        }
        let mut b = vec![P_VOICE];
        put_u64(&mut b, self.local_peer);
        put_u32(&mut b, seq);
        put_u16(&mut b, payload.len() as u16);
        b.extend_from_slice(payload);
        self.fanout_voice(&b, None, self.local_peer);
    }

    // ------------------------------------------------------------ prediction

    /// Owner: send the unacknowledged input frames of a predicted object
    /// (newest last; as many as fit one packet).
    pub fn send_inputs(&mut self, id: u128, cmds: &[&InputCmd]) {
        if !self.is_connected() || self.is_host || cmds.is_empty() {
            return;
        }
        let mut b = vec![P_INPUT];
        put_u128(&mut b, id);
        let skip = cmds.len().saturating_sub(MAX_CMDS_PER_PACKET);
        crate::net_predict::encode_cmds(&mut b, &cmds[skip..], MTU);
        let host = self.host_peer;
        self.send_to_peer(host, &b);
    }

    fn on_input(&mut self, peer: u64, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let id = c.u128();
        if c.bad || self.owner_of_raw(id) != peer || !self.world.predicted.contains(&id) {
            return;
        }
        if !self.gate(peer, Limit::Input) {
            return;
        }
        let Some(cmds) = data.get(17..).and_then(crate::net_predict::decode_cmds) else {
            self.violation(peer, Reason::Malformed, "input packet");
            return;
        };
        // View angles, headset poses and teleport targets ride in `extra`: finite and sane.
        if cmds.iter().any(|cmd| !self.val.input_extras_ok(&cmd.extra) || !cmd.mouse.is_finite()) {
            self.violation(peer, Reason::NotFinite, "input extras");
            return;
        }
        let q = self.inputs.entry(id).or_default();
        for cmd in cmds {
            q.push(cmd);
        }
    }

    /// Host: the input frames of a remote player's predicted object to
    /// simulate this frame (`frame_dt` = host time elapsed).
    pub fn take_inputs(&mut self, id: u128, frame_dt: f32) -> Vec<InputCmd> {
        let now = self.now();
        if self.anchor_wait.get(&id).is_some_and(|t| now < *t) {
            return Vec::new();
        }
        self.anchor_wait.remove(&id);
        self.inputs.get_mut(&id).map(|q| q.take_ready(frame_dt, now)).unwrap_or_default()
    }

    /// Host: the authoritative state of a predicted object, to its owner.
    pub fn send_auth(&mut self, id: u128, owner: u64, auth: &AuthState) {
        let mut b = vec![P_PSTATE];
        put_u128(&mut b, id);
        crate::net_predict::encode_auth(&mut b, auth);
        self.send_to_peer(owner, &b);
    }

    fn on_pstate(&mut self, data: &[u8]) {
        let mut c = Cursor::new(data, 1);
        let id = c.u128();
        if c.bad || !self.has_authority_raw(id) {
            return;
        }
        let Some(auth) = data.get(17..).and_then(crate::net_predict::decode_auth) else {
            return;
        };
        if self.auth_in.get(&id).is_none_or(|a| auth.ack > a.ack) {
            self.auth_in.insert(id, auth);
        }
    }

    /// Owner: the newest authoritative state received for a predicted object.
    pub fn take_auth(&mut self, id: u128) -> Option<AuthState> {
        self.auth_in.remove(&id)
    }

    /// New host: the owner's state a predicted object continues from after a migration.
    pub fn take_anchor(&mut self, id: u128) -> Option<AuthState> {
        self.anchors.remove(&id)
    }

    /// Smoothed snapshot arrival jitter from `origin` (s).
    pub fn jitter_of(&self, origin: u64) -> f32 {
        self.jitter.get(&origin).map_or(0.0, |j| j.jitter)
    }

    /// The session clock as this peer estimates it: the host's clock (plus
    /// the offset a migrated host carries over), one-way delay corrected
    /// with half the round trip. A time base every peer agrees on, for
    /// things each client simulates itself, like the water. It runs on
    /// through a host migration and eases (at most half a frame per frame)
    /// onto the new host's estimate, so it never jumps or runs backwards.
    /// None until the host has been heard from.
    pub fn shared_clock(&self) -> Option<f64> {
        match self.clock_value {
            Some((at, v)) => Some(v + (self.now() - at).max(0.0)),
            None if self.is_host => Some(self.now() + self.clock_offset),
            None => None,
        }
    }

    fn clock_target(&self) -> Option<f64> {
        if self.is_host {
            return Some(self.now() + self.clock_offset);
        }
        let off = self.offsets.get(&self.host_peer)?;
        let conn = self.conns.get(&self.host_peer)?;
        Some(self.now() - off + conn.rtt as f64 * 0.5 + self.host_clock_offset)
    }

    fn update_clock(&mut self, now: f64) {
        let target = self.clock_target();
        self.clock_value = match (self.clock_value, target) {
            (None, t) => t.map(|t| (now, t)),
            (Some((at, v)), t) => {
                let dt = (now - at).max(0.0);
                let free = v + dt;
                match t {
                    Some(t) if (t - free).abs() > 1.0 => Some((now, t)),
                    Some(t) => Some((now, free + (t - free).clamp(-0.5 * dt, 0.5 * dt))),
                    None => Some((now, free)),
                }
            }
        };
    }

    /// Render time (sender clock) for a buffered object from `origin`.
    pub fn render_time(&self, origin: u64, delay: f32) -> f64 {
        let off = self.offsets.get(&origin).copied().unwrap_or(0.0);
        self.now() - off - delay as f64
    }
}

#[cfg(test)]
impl NetSession {
    /// Vanish without a word (a crashed process): nothing is sent, now or on drop.
    pub(crate) fn crash(&mut self) {
        self.closed = true;
        self.conns.clear();
        self.addr_peer.clear();
        self.state = ConnState::Disconnected("crashed".into());
    }
}

impl Drop for NetSession {
    fn drop(&mut self) {
        if self.is_connected() {
            self.disconnect();
        }
        self.close_transport();
    }
}

// ------------------------------------------------------------------ sampling

/// How long Interpolate mode keeps extrapolating when the buffer runs dry
/// (lost or late snapshots) before holding the newest pose.
pub const LOSS_EXTRAPOLATION: f64 = 0.1;

/// Transform at `render_time` from a snapshot buffer.
pub fn sample_snapshots(buf: &SnapBuffer, render_time: f64, mode: InterpolationMode) -> Option<Snap> {
    let newest = *buf.snaps.back()?;
    if buf.baseline || mode == InterpolationMode::None || buf.snaps.len() == 1 {
        return Some(newest);
    }
    if render_time >= newest.time {
        let ahead = render_time - newest.time;
        if mode == InterpolationMode::Extrapolate {
            let ahead = ahead.min(0.25) as f32;
            return Some(Snap { pos: newest.pos + newest.vel * ahead, ..newest });
        }
        // Starved: carry on briefly at the last known velocity (sent, or
        // from the last two snapshots; a stopped object sends a rest
        // snapshot, so it does not drift).
        let vel = if newest.vel != Vec3::ZERO {
            newest.vel
        } else {
            let prev = buf.snaps[buf.snaps.len() - 2];
            let span = (newest.time - prev.time) as f32;
            if span > 1e-4 { (newest.pos - prev.pos) / span } else { Vec3::ZERO }
        };
        let ahead = ahead.min(LOSS_EXTRAPOLATION) as f32;
        return Some(Snap { pos: newest.pos + vel * ahead, ..newest });
    }
    let oldest = buf.snaps[0];
    if render_time <= oldest.time {
        return Some(oldest);
    }
    let i = buf.snaps.iter().position(|s| s.time > render_time)?;
    let (a, b) = (buf.snaps[i - 1], buf.snaps[i]);
    let t = ((render_time - a.time) / (b.time - a.time).max(1e-6)) as f32;
    Some(Snap {
        time: render_time,
        pos: a.pos.lerp(b.pos, t),
        rot: a.rot.slerp(b.rot, t),
        scale: a.scale.lerp(b.scale, t),
        vel: a.vel.lerp(b.vel, t),
    })
}

/// Avatar poses at `render_time`.
pub fn sample_avatar(buf: &AvatarBuffer, render_time: f64) -> Option<[Option<(Vec3, Quat)>; TRACKER_SLOTS]> {
    let newest = buf.snaps.back()?;
    if render_time >= newest.time || buf.snaps.len() == 1 {
        return Some(newest.poses);
    }
    let Some(i) = buf.snaps.iter().position(|s| s.time > render_time) else {
        return Some(newest.poses);
    };
    if i == 0 {
        return Some(buf.snaps[0].poses);
    }
    let (a, b) = (&buf.snaps[i - 1], &buf.snaps[i]);
    let t = ((render_time - a.time) / (b.time - a.time).max(1e-6)) as f32;
    let mut out = [None; TRACKER_SLOTS];
    for ((slot, pa), pb) in out.iter_mut().zip(a.poses).zip(b.poses) {
        *slot = match (pa, pb) {
            (Some((pa, ra)), Some((pb, rb))) => Some((pa.lerp(pb, t), ra.slerp(rb, t))),
            (_, b) => b,
        };
    }
    Some(out)
}

// ------------------------------------------------------------------ LAN discovery

/// A session found on the LAN.
#[derive(Clone, Debug)]
pub struct LanSession {
    pub addr: SocketAddr,
    pub name: String,
    pub players: u32,
    pub max_players: u32,
    /// A dedicated server or a player-hosted session.
    pub topology: Topology,
    last_seen: Instant,
}

/// What [`probe_server`] learned from one unauthenticated round trip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ServerProbe {
    /// Round trip of a padded HELLO and the host's answer, in milliseconds.
    pub rtt_ms: f32,
    /// The network protocol the server speaks (compare with [`PROTOCOL`]).
    pub protocol: u16,
}

/// Ping a game server and read its protocol version without joining: a full-size HELLO that
/// claims protocol 0 is answered with a version hint (no state is kept on the server, no
/// larger than the request, so it is no amplifier). `None` = no answer in `timeout`.
pub fn probe_server(addr: SocketAddr, timeout: std::time::Duration) -> Option<ServerProbe> {
    let socket = UdpSocket::bind(if addr.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" }).ok()?;
    socket.connect(addr).ok()?;
    socket.set_read_timeout(Some(timeout)).ok()?;
    let mut hello = hello_packet(&[0; 16], JOIN_MODE_JOIN);
    hello[1..3].copy_from_slice(&0u16.to_le_bytes());
    let start = Instant::now();
    socket.send(&hello).ok()?;
    let mut buf = [0u8; 512];
    while start.elapsed() < timeout {
        let n = socket.recv(&mut buf).ok()?;
        if n >= 4 && buf[0] == P_REJECT && buf[1] == REJECT_HINT {
            return Some(ServerProbe { rtt_ms: start.elapsed().as_secs_f32() * 1000.0, protocol: u16::from_le_bytes([buf[2], buf[3]]) });
        }
    }
    None
}

/// Finds hosts on the local network by UDP broadcast.
pub struct LanBrowser {
    socket: UdpSocket,
    port: u16,
    sessions: Vec<LanSession>,
}

impl LanBrowser {
    pub fn new() -> std::io::Result<Self> {
        Self::with_port(DISCOVERY_PORT)
    }

    pub fn with_port(port: u16) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(("0.0.0.0", 0))?;
        socket.set_broadcast(true)?;
        socket.set_nonblocking(true)?;
        Ok(Self { socket, port, sessions: Vec::new() })
    }

    /// Broadcast a discovery query (call every couple of seconds while browsing).
    pub fn refresh(&mut self) {
        let mut b = vec![P_DISCOVER];
        put_u16(&mut b, PROTOCOL);
        // The limited broadcast leaves through one interface only, so also
        // ask each interface's own subnet. Re-enumerated every refresh so a
        // laptop that changes network keeps working.
        for (ip, mask) in local_ipv4_interfaces() {
            if let Some(bcast) = directed_broadcast(ip, mask) {
                let _ = self.socket.send_to(&b, (bcast, self.port));
            }
        }
        let _ = self.socket.send_to(&b, ("255.255.255.255", self.port));
        let _ = self.socket.send_to(&b, ("127.0.0.1", self.port));
    }

    /// Collect replies; returns every session seen in the last few seconds.
    pub fn poll(&mut self) -> &[LanSession] {
        let mut buf = [0u8; 512];
        while let Ok((n, src)) = self.socket.recv_from(&mut buf) {
            let mut c = Cursor::new(&buf[..n], 0);
            if c.u8() != P_INFO || c.u16() != PROTOCOL {
                continue;
            }
            let port = c.u16();
            let players = c.u32();
            let max_players = c.u32();
            let name = c.string();
            if c.bad {
                continue;
            }
            // Older hosts end the datagram after the name.
            let topology = if c.p < n { Topology::from_u8(c.u8()) } else { Topology::ListenHost };
            let addr = SocketAddr::new(src.ip(), port);
            self.sessions.retain(|s| s.addr != addr);
            self.sessions.push(LanSession { addr, name, players, max_players, topology, last_seen: Instant::now() });
        }
        self.sessions.retain(|s| s.last_seen.elapsed().as_secs_f32() < 6.0);
        &self.sessions
    }
}

/// Directed broadcast of an interface: `ip | !mask`. `None` for /31 and /32
/// masks, which have no broadcast address.
pub fn directed_broadcast(ip: std::net::Ipv4Addr, mask: std::net::Ipv4Addr) -> Option<std::net::Ipv4Addr> {
    let m = u32::from(mask);
    if m.count_zeros() < 2 {
        return None;
    }
    Some(std::net::Ipv4Addr::from(u32::from(ip) | !m))
}

/// Up, broadcast-capable, non-loopback, non-point-to-point IPv4 interfaces as
/// `(address, netmask)`. Empty where enumeration isn't implemented.
#[cfg(unix)]
fn local_ipv4_interfaces() -> Vec<(std::net::Ipv4Addr, std::net::Ipv4Addr)> {
    let mut out = Vec::new();
    unsafe {
        let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut head) != 0 {
            return out;
        }
        let mut cur = head;
        while !cur.is_null() {
            let ifa = &*cur;
            cur = ifa.ifa_next;
            let flags = ifa.ifa_flags as libc::c_int;
            if flags & libc::IFF_UP == 0
                || flags & libc::IFF_BROADCAST == 0
                || flags & (libc::IFF_LOOPBACK | libc::IFF_POINTOPOINT) != 0
                || ifa.ifa_addr.is_null()
                || ifa.ifa_netmask.is_null()
                || (*ifa.ifa_addr).sa_family as libc::c_int != libc::AF_INET
            {
                continue;
            }
            let ip = &*(ifa.ifa_addr as *const libc::sockaddr_in);
            let mask = &*(ifa.ifa_netmask as *const libc::sockaddr_in);
            out.push((
                std::net::Ipv4Addr::from(u32::from_be(ip.sin_addr.s_addr)),
                std::net::Ipv4Addr::from(u32::from_be(mask.sin_addr.s_addr)),
            ));
        }
        libc::freeifaddrs(head);
    }
    out
}

/// Netmask for a CIDR prefix length (0..=32).
#[cfg_attr(not(windows), allow(dead_code))]
fn mask_from_prefix(len: u8) -> std::net::Ipv4Addr {
    let len = len.min(32) as u32;
    std::net::Ipv4Addr::from(if len == 0 { 0 } else { u32::MAX << (32 - len) })
}

/// Windows: up, non-loopback, non-tunnel IPv4 unicast addresses with their
/// prefix length (`GetAdaptersAddresses`). Not run on CI/Linux; the code is
/// type-checked for `x86_64-pc-windows-gnu` only.
#[cfg(windows)]
fn local_ipv4_interfaces() -> Vec<(std::net::Ipv4Addr, std::net::Ipv4Addr)> {
    use windows_sys::Win32::Foundation::{ERROR_BUFFER_OVERFLOW, NO_ERROR};
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GAA_FLAG_SKIP_ANYCAST, GAA_FLAG_SKIP_DNS_SERVER, GAA_FLAG_SKIP_MULTICAST, GetAdaptersAddresses, IF_TYPE_SOFTWARE_LOOPBACK,
        IF_TYPE_TUNNEL, IP_ADAPTER_ADDRESSES_LH,
    };
    use windows_sys::Win32::NetworkManagement::Ndis::IfOperStatusUp;
    use windows_sys::Win32::Networking::WinSock::{AF_INET, SOCKADDR_IN};

    let flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
    // 8-byte aligned buffer: the adapter list is a chain of structs with pointers.
    let mut words: Vec<u64> = vec![0; 16 * 1024 / 8];
    let mut size = (words.len() * 8) as u32;
    let mut result = ERROR_BUFFER_OVERFLOW;
    for _ in 0..3 {
        result = unsafe {
            GetAdaptersAddresses(AF_INET as u32, flags, std::ptr::null(), words.as_mut_ptr() as *mut IP_ADAPTER_ADDRESSES_LH, &mut size)
        };
        if result != ERROR_BUFFER_OVERFLOW {
            break;
        }
        words = vec![0; (size as usize).div_ceil(8)];
    }
    let mut out = Vec::new();
    if result != NO_ERROR {
        return out;
    }
    unsafe {
        let mut adapter = words.as_ptr() as *const IP_ADAPTER_ADDRESSES_LH;
        while !adapter.is_null() {
            let a = &*adapter;
            adapter = a.Next;
            if a.OperStatus != IfOperStatusUp || a.IfType == IF_TYPE_SOFTWARE_LOOPBACK || a.IfType == IF_TYPE_TUNNEL {
                continue;
            }
            let mut u = a.FirstUnicastAddress;
            while !u.is_null() {
                let ua = &*u;
                u = ua.Next;
                let sa = ua.Address.lpSockaddr;
                if sa.is_null() || (*sa).sa_family != AF_INET {
                    continue;
                }
                let sin = &*(sa as *const SOCKADDR_IN);
                let ip = std::net::Ipv4Addr::from(u32::from_be(sin.sin_addr.S_un.S_addr));
                out.push((ip, mask_from_prefix(ua.OnLinkPrefixLength)));
            }
        }
    }
    out
}

#[cfg(not(any(unix, windows)))]
fn local_ipv4_interfaces() -> Vec<(std::net::Ipv4Addr, std::net::Ipv4Addr)> {
    Vec::new()
}

/// Bind `0.0.0.0:port` with `SO_REUSEADDR` (and `SO_REUSEPORT` where it
/// exists) so several hosts on one machine can all answer discovery queries.
#[cfg(unix)]
fn bind_shared_udp(port: u16) -> std::io::Result<UdpSocket> {
    use std::os::fd::FromRawFd;
    unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0);
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // Owned from here: closes the fd on any error path below.
        let sock = UdpSocket::from_raw_fd(fd);
        let on: libc::c_int = 1;
        let len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        let ptr = &on as *const libc::c_int as *const libc::c_void;
        if libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, ptr, len) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        #[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
        if libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, ptr, len) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut addr: libc::sockaddr_in = std::mem::zeroed();
        addr.sin_family = libc::AF_INET as libc::sa_family_t;
        addr.sin_port = port.to_be();
        addr.sin_addr.s_addr = 0;
        let alen = std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t;
        if libc::bind(fd, &addr as *const libc::sockaddr_in as *const libc::sockaddr, alen) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(sock)
    }
}

#[cfg(not(unix))]
fn bind_shared_udp(port: u16) -> std::io::Result<UdpSocket> {
    UdpSocket::bind(("0.0.0.0", port))
}

pub fn parse_addr(addr: &str) -> std::io::Result<SocketAddr> {
    use std::net::ToSocketAddrs;
    if let Ok(a) = addr.parse::<SocketAddr>() {
        return Ok(a);
    }
    addr.to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("bad address {addr:?}")))
}

// ------------------------------------------------------------------ message codecs

fn enc_spawn(s: &SpawnRecord) -> Vec<u8> {
    let mut m = vec![M_SPAWN];
    put_u128(&mut m, s.id);
    match &s.source {
        NetSpawnSource::Prefab(p) => {
            m.push(0);
            put_str(&mut m, p);
        }
        NetSpawnSource::Clone(id) => {
            m.push(1);
            put_u128(&mut m, id.raw());
        }
    }
    put_vec3(&mut m, s.translation);
    put_quat(&mut m, s.rotation);
    put_u64(&mut m, s.owner);
    m.push(s.player as u8);
    m
}

fn dec_spawn(c: &mut Cursor) -> Option<SpawnRecord> {
    let id = c.u128();
    let source = match c.u8() {
        0 => NetSpawnSource::Prefab(c.string()),
        _ => NetSpawnSource::Clone(ObjectId::from_raw(c.u128())),
    };
    let translation = c.vec3();
    let rotation = c.quat();
    let owner = c.u64();
    let player = c.u8() & 1 != 0;
    (!c.bad).then_some(SpawnRecord { id, source, translation, rotation, owner, player })
}

/// `mid`: the sender's message id (0 = none), so copies resent across a
/// host migration are recognized.
fn enc_var(id: u128, key: &str, v: &NetValue, from: u64, mid: u32) -> Vec<u8> {
    let mut m = vec![M_VAR];
    put_u128(&mut m, id);
    put_str(&mut m, key);
    put_value(&mut m, v);
    put_u64(&mut m, from);
    put_u32(&mut m, mid);
    m
}

fn enc_rpc(from: u64, mid: u32, target: RpcTarget, obj: u128, name: &str, args: &[NetValue]) -> Vec<u8> {
    let mut b = Vec::new();
    put_u64(&mut b, from);
    put_u32(&mut b, mid);
    match target {
        RpcTarget::All => b.push(0),
        RpcTarget::Others => b.push(1),
        RpcTarget::Server => b.push(2),
        RpcTarget::Owner => b.push(3),
        RpcTarget::Peer(p) => {
            b.push(4);
            put_u64(&mut b, p);
        }
    }
    put_u128(&mut b, obj);
    put_str(&mut b, name);
    b.push(args.len().min(255) as u8);
    for a in args.iter().take(255) {
        put_value(&mut b, a);
    }
    b
}

type RpcParts = (u64, u32, RpcTarget, u128, String, Vec<NetValue>);

fn dec_rpc(c: &mut Cursor) -> Option<RpcParts> {
    let from = c.u64();
    let mid = c.u32();
    let target = match c.u8() {
        0 => RpcTarget::All,
        1 => RpcTarget::Others,
        2 => RpcTarget::Server,
        3 => RpcTarget::Owner,
        _ => RpcTarget::Peer(c.u64()),
    };
    let obj = c.u128();
    let name = c.string();
    let n = c.u8();
    let mut args = Vec::with_capacity(n as usize);
    for _ in 0..n {
        args.push(c.value());
    }
    (!c.bad).then_some((from, mid, target, obj, name, args))
}

/// A padded HELLO: protocol, client nonce, mode.
fn hello_packet(nonce: &[u8; 16], mode: u8) -> Vec<u8> {
    let mut b = vec![P_HELLO];
    put_u16(&mut b, PROTOCOL);
    b.extend_from_slice(nonce);
    b.push(mode);
    b.resize(HELLO_SIZE, 0);
    b
}

/// A CHALLENGE, with the transcript the connection key binds.
fn parse_challenge(data: &[u8], nonce: &[u8; 16]) -> Option<Challenge> {
    let mut c = Cursor::new(data, 1);
    if c.u16() != PROTOCOL {
        return None;
    }
    let kdf = KdfParams::decode(c.take(crate::net_secure::KDF_BYTES))?;
    let host_pk: [u8; 32] = c.arr();
    let epoch = c.u32();
    let host_peer = c.u64();
    let cookie: [u8; 16] = c.arr();
    if c.bad {
        return None;
    }
    let transcript = hash(&[b"zestengine-net handshake v6", nonce, data]);
    Some(Challenge { kdf, host_pk, epoch, host_peer, cookie, transcript })
}

fn join_message(transcript: &[u8; 32], rejoin: u64) -> Vec<u8> {
    let mut m = b"zestengine-net join v6".to_vec();
    m.extend_from_slice(transcript);
    m.extend_from_slice(&rejoin.to_le_bytes());
    m
}

fn welcome_message(transcript: &[u8; 32], peer: u64) -> Vec<u8> {
    let mut m = b"zestengine-net welcome v6".to_vec();
    m.extend_from_slice(transcript);
    m.extend_from_slice(&peer.to_le_bytes());
    m
}

fn reject_message(nonce: &[u8; 16], cookie: &[u8; 16], code: u8, reason: &str) -> Vec<u8> {
    let mut m = b"zestengine-net reject v6".to_vec();
    m.extend_from_slice(nonce);
    m.extend_from_slice(cookie);
    m.push(code);
    m.extend_from_slice(reason.as_bytes());
    m
}

fn enc_succession(s: &Succession) -> Vec<u8> {
    let mut m = vec![M_SUCCESSION];
    put_u32(&mut m, s.version);
    put_u32(&mut m, s.epoch);
    put_u64(&mut m, s.next_peer);
    put_u32(&mut m, s.spawn_counter);
    put_u32(&mut m, s.max_players);
    m.push(s.clients_can_spawn as u8);
    put_str(&mut m, &s.session_name);
    put_str(&mut m, &s.relay);
    put_str(&mut m, &s.room);
    match &s.room_key {
        Some(k) => {
            m.push(1);
            m.extend_from_slice(k);
        }
        None => m.push(0),
    }
    put_u16(&mut m, s.order.len().min(4096) as u16);
    for c in s.order.iter().take(4096) {
        put_u64(&mut m, c.peer);
        m.push(c.addrs.len().min(4) as u8);
        for a in c.addrs.iter().take(4) {
            relay_proto::put_addr(&mut m, *a);
        }
    }
    for list in [&s.ban, &s.allow] {
        put_u16(&mut m, list.len().min(4096) as u16);
        for f in list.iter().take(4096) {
            put_str(&mut m, f);
        }
    }
    m
}

fn dec_succession(c: &mut Cursor) -> Option<Succession> {
    let mut s = Succession {
        version: c.u32(),
        epoch: c.u32(),
        next_peer: c.u64(),
        spawn_counter: c.u32(),
        max_players: c.u32(),
        clients_can_spawn: c.u8() != 0,
        session_name: c.string(),
        relay: c.string(),
        room: c.string(),
        ..Default::default()
    };
    if c.u8() == 1 {
        s.room_key = Some(c.arr());
    }
    let n = c.u16();
    for _ in 0..n {
        let peer = c.u64();
        let k = c.u8();
        let mut addrs = Vec::new();
        for _ in 0..k {
            let mut at = c.p;
            match relay_proto::read_addr(c.d, &mut at) {
                Some(a) => {
                    c.p = at;
                    addrs.push(a);
                }
                None => {
                    c.bad = true;
                    break;
                }
            }
        }
        s.order.push(Successor { peer, addrs });
    }
    for list in [&mut s.ban, &mut s.allow] {
        let n = c.u16();
        for _ in 0..n {
            list.push(c.string());
        }
    }
    (!c.bad).then_some(s)
}

const E_POS: u8 = 1;
const E_ROT: u8 = 2;
const E_SCALE: u8 = 4;
const E_VEL: u8 = 8;
const E_QUANT: u8 = 16;
/// The id is a 32-bit alias the host taught the client (`M_NETID`).
const E_SHORT: u8 = 32;

fn entry_flags(e: &SnapEntry) -> u8 {
    let mut flags = 0;
    if e.pos.is_some() {
        flags |= E_POS;
    }
    if e.rot.is_some() {
        flags |= E_ROT;
    }
    if e.scale.is_some() {
        flags |= E_SCALE;
    }
    if e.vel.is_some() {
        flags |= E_VEL;
    }
    if e.quantized {
        flags |= E_QUANT;
    }
    flags
}

/// The fields after the id and the flags byte.
fn enc_fields(b: &mut Vec<u8>, e: &SnapEntry) {
    if let Some(p) = e.pos {
        if e.quantized {
            for v in p.to_array() {
                put_i24(b, (v * 1000.0).round().clamp(-8_388_607.0, 8_388_607.0) as i32);
            }
        } else {
            put_vec3(b, p);
        }
    }
    if let Some(r) = e.rot {
        if e.quantized {
            put_rot32(b, r);
        } else {
            put_rot56(b, r);
        }
    }
    if let Some(s) = e.scale {
        put_vec3(b, s);
    }
    if let Some(v) = e.vel {
        put_vec3(b, v);
    }
}

fn dec_fields(c: &mut Cursor, id: u128, flags: u8) -> Option<SnapEntry> {
    let q = flags & E_QUANT != 0;
    let pos = (flags & E_POS != 0).then(|| {
        if q {
            Vec3::new(c.i24() as f32 / 1000.0, c.i24() as f32 / 1000.0, c.i24() as f32 / 1000.0)
        } else {
            c.vec3()
        }
    });
    let rot = (flags & E_ROT != 0).then(|| if q { c.rot32() } else { c.rot56() });
    let scale = (flags & E_SCALE != 0).then(|| c.vec3());
    let vel = (flags & E_VEL != 0).then(|| c.vec3());
    (!c.bad).then_some(SnapEntry { id, pos, rot, scale, vel, quantized: q })
}

pub fn enc_entry(b: &mut Vec<u8>, e: &SnapEntry) {
    put_u128(b, e.id);
    b.push(entry_flags(e));
    enc_fields(b, e);
}

pub fn dec_entry(c: &mut Cursor) -> Option<SnapEntry> {
    let id = c.u128();
    let flags = c.u8();
    dec_fields(c, id, flags)
}

/// Batch (`P_SNAPB`) entry: flags first, then a 128-bit id, or a 32-bit
/// alias when `e.id` is small (real ids are random 128-bit values).
fn enc_entry_b(b: &mut Vec<u8>, e: &SnapEntry) {
    if e.id <= u32::MAX as u128 {
        b.push(entry_flags(e) | E_SHORT);
        put_u32(b, e.id as u32);
    } else {
        b.push(entry_flags(e));
        put_u128(b, e.id);
    }
    enc_fields(b, e);
}

/// An alias comes back as a small id; `interest::resolve_ids` maps it.
fn dec_entry_b(c: &mut Cursor) -> Option<SnapEntry> {
    let flags = c.u8();
    let id = if flags & E_SHORT != 0 { c.u32() as u128 } else { c.u128() };
    dec_fields(c, id, flags)
}

fn entry_size(e: &SnapEntry) -> usize {
    17 + e.pos.map_or(0, |_| if e.quantized { 9 } else { 12 })
        + e.rot.map_or(0, |_| if e.quantized { 4 } else { 7 })
        + e.scale.map_or(0, |_| 12)
        + e.vel.map_or(0, |_| 12)
}

/// Size of a batch entry.
fn entry_size_b(e: &SnapEntry) -> usize {
    entry_size(e) - if e.id <= u32::MAX as u128 { 12 } else { 0 }
}

/// Batch entries into as few packets as fit the MTU.
pub fn build_snap_packets(origin: u64, time: f64, entries: &[SnapEntry]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let mut b = vec![P_SNAP];
        put_u64(&mut b, origin);
        put_f64(&mut b, time);
        let count_at = b.len();
        b.push(0);
        let mut n = 0u8;
        while i < entries.len() && n < 255 && b.len() + entry_size(&entries[i]) <= MTU {
            enc_entry(&mut b, &entries[i]);
            n += 1;
            i += 1;
        }
        if n == 0 {
            i += 1; // cannot happen with sane entries; avoid looping forever
            continue;
        }
        b[count_at] = n;
        out.push(b);
    }
    out
}

/// A client's snapshot entries as `P_SNAPB` packets (one group each): ids that are 32-bit
/// aliases go as four bytes instead of sixteen.
pub fn build_upstream_packets(origin: u64, time: f64, entries: &[SnapEntry]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let mut b = vec![P_SNAPB, 1];
        put_u64(&mut b, origin);
        put_f64(&mut b, time);
        let count_at = b.len();
        b.push(0);
        let mut n = 0u8;
        while i < entries.len() && n < 255 && b.len() + entry_size_b(&entries[i]) <= MTU {
            enc_entry_b(&mut b, &entries[i]);
            n += 1;
            i += 1;
        }
        if n == 0 {
            i += 1;
            continue;
        }
        b[count_at] = n;
        out.push(b);
    }
    out
}

// ------------------------------------------------------------------ wire helpers

fn put_u16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u128(b: &mut Vec<u8>, v: u128) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_i24(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes()[..3]);
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(u16::MAX as usize);
    put_u16(b, n as u16);
    b.extend_from_slice(&bytes[..n]);
}
fn put_vec3(b: &mut Vec<u8>, v: Vec3) {
    put_f32(b, v.x);
    put_f32(b, v.y);
    put_f32(b, v.z);
}
fn put_quat(b: &mut Vec<u8>, v: Quat) {
    put_f32(b, v.x);
    put_f32(b, v.y);
    put_f32(b, v.z);
    put_f32(b, v.w);
}

/// Smallest-three: drop the largest component (recomputed on decode).
fn smallest_three(r: Quat) -> (usize, [f32; 3]) {
    let mut q = r.normalize().to_array();
    let (big, _) = q.iter().enumerate().fold((0, 0.0f32), |(bi, bv), (i, v)| if v.abs() > bv { (i, v.abs()) } else { (bi, bv) });
    if q[big] < 0.0 {
        q = q.map(|v| -v);
    }
    let mut rest = [0.0; 3];
    let mut k = 0;
    for (i, v) in q.iter().enumerate() {
        if i != big {
            rest[k] = *v;
            k += 1;
        }
    }
    (big, rest)
}

fn from_smallest_three(big: usize, rest: [f32; 3]) -> Quat {
    let w = (1.0 - rest.iter().map(|v| v * v).sum::<f32>()).max(0.0).sqrt();
    let mut q = [0.0; 4];
    let mut k = 0;
    for (i, slot) in q.iter_mut().enumerate() {
        if i == big {
            *slot = w;
        } else {
            *slot = rest[k];
            k += 1;
        }
    }
    Quat::from_array(q).normalize()
}

const RMAX: f32 = std::f32::consts::FRAC_1_SQRT_2;

fn put_rot32(b: &mut Vec<u8>, r: Quat) {
    let (big, rest) = smallest_three(r);
    let mut v = big as u32;
    for c in rest {
        let q = (((c / RMAX).clamp(-1.0, 1.0) * 0.5 + 0.5) * 1023.0).round() as u32;
        v = (v << 10) | q;
    }
    put_u32(b, v);
}

fn put_rot56(b: &mut Vec<u8>, r: Quat) {
    let (big, rest) = smallest_three(r);
    b.push(big as u8);
    for c in rest {
        let q = ((c / RMAX).clamp(-1.0, 1.0) * 32767.0).round() as i16;
        b.extend_from_slice(&q.to_le_bytes());
    }
}

fn put_value(b: &mut Vec<u8>, v: &NetValue) {
    match v {
        NetValue::Bool(x) => {
            b.push(0);
            b.push(*x as u8);
        }
        NetValue::Int(x) => {
            b.push(1);
            b.extend_from_slice(&x.to_le_bytes());
        }
        NetValue::Float(x) => {
            b.push(2);
            put_f32(b, *x);
        }
        NetValue::Vec3(x) => {
            b.push(3);
            for c in x {
                put_f32(b, *c);
            }
        }
        NetValue::Quat(x) => {
            b.push(4);
            for c in x {
                put_f32(b, *c);
            }
        }
        NetValue::Str(s) => {
            b.push(5);
            put_str(b, s);
        }
        NetValue::Bytes(d) => {
            b.push(6);
            put_u32(b, d.len() as u32);
            b.extend_from_slice(d);
        }
    }
}

/// Bounds-checked reader: short reads yield zeros and set `bad`.
pub struct Cursor<'a> {
    d: &'a [u8],
    p: usize,
    pub bad: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(d: &'a [u8], p: usize) -> Self {
        Self { d, p, bad: false }
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        match self.d.get(self.p..self.p + n) {
            Some(s) => {
                self.p += n;
                s
            }
            None => {
                self.bad = true;
                self.p = self.d.len();
                &[]
            }
        }
    }
    fn rest(&mut self) -> &'a [u8] {
        let s = self.d.get(self.p..).unwrap_or(&[]);
        self.p = self.d.len();
        s
    }
    fn arr<const N: usize>(&mut self) -> [u8; N] {
        self.take(N).try_into().unwrap_or([0; N])
    }
    fn u8(&mut self) -> u8 {
        self.arr::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.arr())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.arr())
    }
    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.arr())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.arr())
    }
    fn u128(&mut self) -> u128 {
        u128::from_le_bytes(self.arr())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.arr())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.arr())
    }
    fn i24(&mut self) -> i32 {
        let b: [u8; 3] = self.arr();
        (i32::from_le_bytes([b[0], b[1], b[2], 0]) << 8) >> 8
    }
    fn string(&mut self) -> String {
        let n = self.u16() as usize;
        String::from_utf8_lossy(self.take(n)).into_owned()
    }
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.f32(), self.f32(), self.f32())
    }
    fn quat(&mut self) -> Quat {
        Quat::from_xyzw(self.f32(), self.f32(), self.f32(), self.f32())
    }
    fn rot32(&mut self) -> Quat {
        let v = self.u32();
        let big = (v >> 30) as usize & 3;
        let mut rest = [0.0; 3];
        for (k, slot) in rest.iter_mut().enumerate() {
            let q = (v >> (10 * (2 - k))) & 1023;
            *slot = ((q as f32 / 1023.0) * 2.0 - 1.0) * RMAX;
        }
        from_smallest_three(big, rest)
    }
    fn rot56(&mut self) -> Quat {
        let big = (self.u8() & 3) as usize;
        let mut rest = [0.0; 3];
        for slot in rest.iter_mut() {
            *slot = self.i16() as f32 / 32767.0 * RMAX;
        }
        from_smallest_three(big, rest)
    }
    fn value(&mut self) -> NetValue {
        match self.u8() {
            0 => NetValue::Bool(self.u8() != 0),
            1 => NetValue::Int(i64::from_le_bytes(self.arr())),
            2 => NetValue::Float(self.f32()),
            3 => NetValue::Vec3([self.f32(), self.f32(), self.f32()]),
            4 => NetValue::Quat([self.f32(), self.f32(), self.f32(), self.f32()]),
            5 => NetValue::Str(self.string()),
            6 => {
                let n = self.u32() as usize;
                NetValue::Bytes(self.take(n).to_vec())
            }
            _ => {
                self.bad = true;
                NetValue::Int(0)
            }
        }
    }
}

/// Hand-built wire messages, for tests and tools that play a misbehaving
/// client (see `NetSession::inject_plain` / `inject_reliable`).
#[doc(hidden)]
pub mod wire {
    use super::*;

    pub fn claim(id: u128) -> Vec<u8> {
        let mut m = vec![M_CLAIM];
        put_u128(&mut m, id);
        m
    }
    pub fn transfer(id: u128, to: u64) -> Vec<u8> {
        let mut m = vec![M_TRANSFER];
        put_u128(&mut m, id);
        put_u64(&mut m, to);
        m
    }
    pub fn despawn(id: u128) -> Vec<u8> {
        let mut m = vec![M_DESPAWN];
        put_u128(&mut m, id);
        m
    }
    pub fn parent(id: u128, parent: u128) -> Vec<u8> {
        let mut m = vec![M_PARENT];
        put_u128(&mut m, id);
        put_u128(&mut m, parent);
        m
    }
    pub fn spawn(id: u128, prefab: &str, pos: Vec3, owner: u64) -> Vec<u8> {
        enc_spawn(&SpawnRecord { id, source: NetSpawnSource::Prefab(prefab.to_owned()), translation: pos, rotation: Quat::IDENTITY, owner, player: false })
    }
    pub fn var(id: u128, key: &str, value: &NetValue, sender: u64) -> Vec<u8> {
        enc_var(id, key, value, sender, 0)
    }
    pub fn text(target: u64, sender: u64, text: &str) -> Vec<u8> {
        let mut m = vec![M_TEXT];
        put_u64(&mut m, target);
        put_u64(&mut m, sender);
        put_u32(&mut m, 0);
        put_str(&mut m, text);
        m
    }
    /// A reliable RPC message (`M_RPC`).
    pub fn rpc(sender: u64, target: RpcTarget, obj: u128, name: &str, args: &[NetValue]) -> Vec<u8> {
        let mut m = vec![M_RPC];
        m.extend_from_slice(&enc_rpc(sender, 0, target, obj, name, args));
        m
    }
    /// An unreliable RPC datagram (`P_URPC`).
    pub fn urpc(sender: u64, target: RpcTarget, obj: u128, name: &str, args: &[NetValue]) -> Vec<u8> {
        let mut m = vec![P_URPC];
        m.extend_from_slice(&enc_rpc(sender, 0, target, obj, name, args));
        m
    }
    pub fn voice(from: u64, seq: u32, payload: &[u8]) -> Vec<u8> {
        let mut b = vec![P_VOICE];
        put_u64(&mut b, from);
        put_u32(&mut b, seq);
        put_u16(&mut b, payload.len() as u16);
        b.extend_from_slice(payload);
        b
    }
    pub fn snapshot(origin: u64, time: f64, entries: &[SnapEntry]) -> Vec<Vec<u8>> {
        build_snap_packets(origin, time, entries)
    }
    pub fn avatar(origin: u64, time: f64, id: u128, poses: &[Option<(Vec3, Quat)>; TRACKER_SLOTS]) -> Vec<u8> {
        let mut b = vec![P_AVATAR];
        put_u64(&mut b, origin);
        put_f64(&mut b, time);
        put_u128(&mut b, id);
        let mask = poses.iter().enumerate().fold(0u16, |m, (i, p)| if p.is_some() { m | (1 << i) } else { m });
        put_u16(&mut b, mask);
        for (p, r) in poses.iter().flatten() {
            put_vec3(&mut b, *p);
            put_rot32(&mut b, *r);
        }
        b
    }
    pub fn avparam(origin: u64, id: u128, payload: &[u8]) -> Vec<u8> {
        let mut b = vec![P_AVPARAM];
        put_u64(&mut b, origin);
        put_u128(&mut b, id);
        b.extend_from_slice(payload);
        b
    }
    pub fn anim(origin: u64, id: u128, params: &[(&str, f32)]) -> Vec<u8> {
        let mut b = vec![P_ANIM];
        put_u64(&mut b, origin);
        put_u128(&mut b, id);
        b.push(params.len() as u8);
        for (k, v) in params {
            put_str(&mut b, k);
            put_f32(&mut b, *v);
        }
        b
    }
    /// A predicted-pawn input packet header followed by `body` (garbage allowed).
    pub fn input(id: u128, body: &[u8]) -> Vec<u8> {
        let mut b = vec![P_INPUT];
        put_u128(&mut b, id);
        b.extend_from_slice(body);
        b
    }
    /// A packet with an unknown tag and `len` bytes of filler.
    pub fn garbage(tag: u8, len: usize) -> Vec<u8> {
        let mut b = vec![tag];
        b.extend((0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)));
        b
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn eleven_tracker_slots_survive_avatar_wire_encoding() {
        use super::*;
        let poses = std::array::from_fn(|slot| Some((Vec3::new(slot as f32 * 0.1, 1.0, 0.0), Quat::from_rotation_y(slot as f32 * 0.03))));
        let bytes = wire::avatar(7, 2.0, 9, &poses);
        let mut c = Cursor::new(&bytes, 1);
        assert_eq!(c.u64(), 7); assert_eq!(c.f64(), 2.0); assert_eq!(c.u128(), 9);
        assert_eq!(c.u16(), (1 << 11) - 1);
        for pose in poses.into_iter().flatten() {
            assert!(c.vec3().distance(pose.0) < 1e-6);
            assert!(c.rot32().angle_between(pose.1) < 0.005);
        }
        assert!(!c.bad);
    }
    use super::*;

    const DT: f64 = 1.0 / 60.0;

    fn cfg(name: &str) -> NetConfig {
        NetConfig {
            player_name: name.to_string(),
            discovery: false,
            secret: String::new(),
            relay: String::new(),
            punch: true,
            account_name: String::new(),
            account_password: String::new(),
            accounts: Vec::new(),
            ..Default::default()
        }
    }

    fn host(hub: &MemHub) -> NetSession {
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), cfg("Host"));
        h.use_manual_clock();
        h
    }

    fn client(hub: &MemHub, host: &NetSession, name: &str) -> NetSession {
        let mut c = NetSession::join_on(Box::new(hub.endpoint()), host.local_addr().unwrap(), cfg(name));
        c.use_manual_clock();
        c
    }

    fn step(all: &mut [&mut NetSession], frames: usize) {
        for _ in 0..frames {
            for s in all.iter_mut() {
                s.advance(DT);
                s.pump();
            }
        }
    }

    fn events(s: &mut NetSession) -> Vec<NetEvent> {
        s.view().events
    }

    #[test]
    fn join_leave_timeout_and_player_list() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "Ann");
        let mut b = client(&hub, &h, "Bob");
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert_eq!((a.local_peer(), b.local_peer()), (2, 3));
        for s in [&h, &a, &b] {
            let names: Vec<String> = s.peer_infos().into_iter().map(|p| p.name).collect();
            assert_eq!(names, vec!["Host", "Ann", "Bob"], "everyone sees the same player list");
        }
        assert!(events(&mut h).iter().any(|e| matches!(e, NetEvent::PeerJoined { id: 3, name } if name == "Bob")));
        assert!(events(&mut a).iter().any(|e| matches!(e, NetEvent::PeerJoined { id: 3, .. })));

        // Clean leave.
        b.disconnect();
        step(&mut [&mut h, &mut a], 10);
        assert!(events(&mut h).iter().any(|e| matches!(e, NetEvent::PeerLeft { id: 3, .. })));
        assert!(events(&mut a).iter().any(|e| matches!(e, NetEvent::PeerLeft { id: 3, .. })));
        assert_eq!(a.peer_count(), 2);

        // A vanished client times out on the host, and notices itself.
        hub.cut(a.local_addr().unwrap());
        step(&mut [&mut h, &mut a], (11.0 / DT) as usize);
        assert!(events(&mut h).iter().any(|e| matches!(e, NetEvent::PeerLeft { id: 2, reason } if reason == "timed out")));
        assert!(matches!(a.state(), ConnState::Disconnected(_)));
        assert_eq!(h.peer_count(), 1);
    }

    #[test]
    fn full_session_rejects() {
        let hub = MemHub::new();
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), NetConfig { max_players: 2, ..cfg("Host") });
        h.use_manual_clock();
        let mut a = client(&hub, &h, "A");
        let mut b = client(&hub, &h, "B");
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert!(a.is_connected());
        assert!(matches!(b.state(), ConnState::Disconnected(r) if r.contains("full")));
    }

    #[test]
    fn host_accounts_require_valid_login() {
        let hub = MemHub::new();
        let mut host_cfg = cfg("Host");
        host_cfg.accounts = vec![
            NetAccount {
                name: "ann".into(),
                password: "pw1".into(),
            },
            NetAccount {
                name: "bob".into(),
                password: "pw2".into(),
            },
        ];
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), host_cfg);
        h.use_manual_clock();

        let mut missing = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), cfg("Missing"));
        missing.use_manual_clock();
        step(&mut [&mut h, &mut missing], 10);
        assert!(matches!(missing.state(), ConnState::Disconnected(r) if r.contains("account login required")));

        let mut wrong_cfg = cfg("Wrong");
        wrong_cfg.account_name = "ann".into();
        wrong_cfg.account_password = "nope".into();
        let mut wrong = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), wrong_cfg);
        wrong.use_manual_clock();
        step(&mut [&mut h, &mut wrong], 10);
        assert!(matches!(wrong.state(), ConnState::Disconnected(r) if r.contains("wrong account password")));

        let mut ok_cfg = cfg("Ann");
        ok_cfg.account_name = "ann".into();
        ok_cfg.account_password = "pw1".into();
        let mut ann = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), ok_cfg);
        ann.use_manual_clock();
        step(&mut [&mut h, &mut ann], 10);
        assert!(ann.is_connected());

        let mut reused_cfg = cfg("Ann2");
        reused_cfg.account_name = "ann".into();
        reused_cfg.account_password = "pw1".into();
        let mut ann2 = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), reused_cfg);
        ann2.use_manual_clock();
        step(&mut [&mut h, &mut ann, &mut ann2], 12);
        assert!(matches!(ann2.state(), ConnState::Disconnected(r) if r.contains("already in use")));
    }

    #[test]
    fn network_manager_account_list_populates_host_accounts() {
        let mgr = NetworkManager {
            account_list: vec!["ann:pw1".into(), "bob:pw2".into()],
            ..Default::default()
        };
        let cfg = NetConfig::from_manager(&mgr);
        assert_eq!(cfg.accounts.len(), 2);
        assert_eq!(cfg.accounts[0].name, "ann");
        assert_eq!(cfg.accounts[0].password, "pw1");
        assert_eq!(cfg.accounts[1].name, "bob");
        assert_eq!(cfg.accounts[1].password, "pw2");
    }

    #[test]
    fn ownership_takeover_request_and_fixed() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        let mut b = client(&hub, &h, "B");
        step(&mut [&mut h, &mut a, &mut b], 10);
        let id = ObjectId::new();

        // Takeover (default): last claim wins.
        a.request_ownership(id);
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert!(a.owns(id) && b.owner_of(id) == 2 && h.owner_of(id) == 2);
        b.request_ownership(id);
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert!(b.owns(id) && !a.owns(id));
        assert!(events(&mut a).iter().any(|e| matches!(e, NetEvent::OwnershipChanged { owner: 3, previous: 2, .. })));

        // Request: the owner is asked and decides.
        h.world.modes.insert(id.raw(), OwnershipMode::Request);
        a.request_ownership(id);
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert!(b.owns(id), "request mode does not steal");
        assert!(events(&mut b).iter().any(|e| matches!(e, NetEvent::OwnershipRequested { by: 2, .. })));
        assert!(events(&mut a).iter().any(|e| matches!(e, NetEvent::OwnershipDenied { .. })));
        b.transfer_ownership(id, 2);
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert!(a.owns(id), "owner handed it over");
        a.release_ownership(id);
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert_eq!(h.owner_of(id), 0);
        assert!(h.has_authority_raw(id.raw()), "unowned objects fall back to the host");

        // Fixed: clients can't claim.
        h.world.modes.insert(id.raw(), OwnershipMode::Fixed);
        a.request_ownership(id);
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert_eq!(h.owner_of(id), 0);
    }

    #[test]
    fn reliable_channel_survives_loss_and_fragments() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        step(&mut [&mut h, &mut a], 10);
        hub.set_loss(0.3);
        for i in 0..40 {
            a.send_message(None, &format!("m{i}"));
        }
        let big = NetValue::Bytes((0..5000u32).map(|i| (i % 251) as u8).collect());
        let id = ObjectId::new();
        a.request_ownership(id);
        step(&mut [&mut h, &mut a], 60);
        a.set_var(id, "blob", big.clone());
        step(&mut [&mut h, &mut a], 240);
        let v = h.view();
        let got: Vec<&str> = v.messages.iter().map(|(_, _, t)| t.as_str()).collect();
        let want: Vec<String> = (0..40).map(|i| format!("m{i}")).collect();
        assert_eq!(got, want.iter().map(|s| s.as_str()).collect::<Vec<_>>(), "all delivered, in order");
        assert!(v.events.iter().any(|e| matches!(e, NetEvent::VarChanged { value, .. } if *value == big)), "5 kB value reassembled");
        assert!(h.stats.resends + a.stats.resends > 0, "loss forced resends");
    }

    #[test]
    fn synced_variables_owner_write() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        let mut b = client(&hub, &h, "B");
        step(&mut [&mut h, &mut a, &mut b], 10);
        let id = ObjectId::new();
        a.request_ownership(id);
        step(&mut [&mut h, &mut a, &mut b], 10);
        a.set_var(id, "hp", NetValue::Int(75));
        step(&mut [&mut h, &mut a, &mut b], 10);
        for s in [&mut h, &mut b] {
            assert!(events(s).iter().any(|e| matches!(e, NetEvent::VarChanged { key, value: NetValue::Int(75), from: 2, .. } if key == "hp")));
            assert!(s.pending_vars.iter().any(|p| p.key == "hp"), "queued for the scene");
        }
        // A non-owner write is dropped by the host.
        b.set_var(id, "hp", NetValue::Int(0));
        step(&mut [&mut h, &mut a, &mut b], 10);
        assert!(!events(&mut a).iter().any(|e| matches!(e, NetEvent::VarChanged { value: NetValue::Int(0), .. })));
        assert_eq!(h.world.vars[&id.raw()][0].1, NetValue::Int(75));
    }

    #[test]
    fn rpc_targets() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        let mut b = client(&hub, &h, "B");
        step(&mut [&mut h, &mut a, &mut b], 10);
        let obj = ObjectId::new();
        b.request_ownership(obj);
        step(&mut [&mut h, &mut a, &mut b], 10);
        let names = |s: &mut NetSession| -> Vec<String> { s.view().rpcs.into_iter().map(|r| r.name).collect() };
        for s in [&mut h, &mut a, &mut b] {
            names(s);
        }
        a.send_rpc(None, "all", &[NetValue::Int(1)], RpcTarget::All, true);
        a.send_rpc(None, "others", &[], RpcTarget::Others, true);
        a.send_rpc(None, "server", &[], RpcTarget::Server, true);
        a.send_rpc(Some(obj), "owner", &[], RpcTarget::Owner, true);
        a.send_rpc(None, "peer3", &[], RpcTarget::Peer(3), false);
        h.send_rpc(None, "fromhost", &[NetValue::from("hi")], RpcTarget::Others, true);
        step(&mut [&mut h, &mut a, &mut b], 10);
        let sorted = |mut v: Vec<String>| {
            v.sort();
            v
        };
        let (hn, an, bn) = (sorted(names(&mut h)), sorted(names(&mut a)), sorted(names(&mut b)));
        assert_eq!(hn, vec!["all", "others", "server"]);
        assert_eq!(an, vec!["all", "fromhost"]);
        assert_eq!(bn, vec!["all", "fromhost", "others", "owner", "peer3"]);
    }

    #[test]
    fn spawn_and_late_join_catch_up() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        step(&mut [&mut h, &mut a], 10);
        let template = ObjectId::new();
        let hs = ObjectId::new();
        h.spawn(NetSpawnSource::Clone(template), hs, Vec3::X, Quat::IDENTITY, 0);
        let as_ = ObjectId::new();
        a.spawn(NetSpawnSource::Prefab("prefabs/ball.prefab".into()), as_, Vec3::Y, Quat::IDENTITY, 2);
        h.set_var(hs, "score", NetValue::Int(7));
        h.send_transform(hs, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);
        step(&mut [&mut h, &mut a], 10);
        let hv = h.view();
        assert!(hv.world_ops.iter().any(|op| matches!(op, NetWorldOp::Spawn { root_id, .. } if *root_id == as_)), "host spawns the client's object");
        let av = a.view();
        assert!(av.world_ops.iter().any(|op| matches!(op, NetWorldOp::Spawn { root_id, .. } if *root_id == hs)));
        assert_eq!(av.owners.get(&as_), Some(&2));

        // A late joiner gets both spawns, their owners, the variable and the last transform.
        let mut c = client(&hub, &h, "C");
        step(&mut [&mut h, &mut a, &mut c], 20);
        let cv = c.view();
        let spawned: Vec<ObjectId> = cv
            .world_ops
            .iter()
            .filter_map(|op| match op {
                NetWorldOp::Spawn { root_id, .. } => Some(*root_id),
                _ => None,
            })
            .collect();
        assert!(spawned.contains(&hs) && spawned.contains(&as_));
        assert_eq!(cv.owners.get(&as_), Some(&2));
        assert!(c.pending_vars.iter().any(|p| p.id == hs.raw() && p.value == NetValue::Int(7)));
        let x = c.remote_transform(hs).expect("baseline transform");
        assert!((x.translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);

        // Despawn by the owner reaches everyone.
        a.despawn(as_);
        step(&mut [&mut h, &mut a, &mut c], 10);
        assert!(c.view().world_ops.iter().any(|op| matches!(op, NetWorldOp::Despawn(id) if *id == as_)));
        assert!(h.world.spawns.iter().all(|s| s.id != as_.raw()));
    }

    #[test]
    fn owner_leaving_releases_or_despawns() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        step(&mut [&mut h, &mut a], 10);
        let kept = ObjectId::new();
        let temp = ObjectId::new();
        a.request_ownership(kept);
        a.spawn(NetSpawnSource::Clone(ObjectId::new()), temp, Vec3::ZERO, Quat::IDENTITY, 2);
        step(&mut [&mut h, &mut a], 10);
        h.world.despawn_on_leave.insert(temp.raw());
        a.disconnect();
        step(&mut [&mut h], 5);
        assert_eq!(h.owner_of(kept), 0, "released to the host");
        assert!(h.view().world_ops.iter().any(|op| matches!(op, NetWorldOp::Despawn(id) if *id == temp)));
    }

    #[test]
    fn transform_codec_and_interpolation() {
        let r = Quat::from_euler(glam::EulerRot::YXZ, 1.1, -0.4, 0.25);
        let e = SnapEntry { id: 9, pos: Some(Vec3::new(123.4567, -0.001, 77.0)), rot: Some(r), scale: None, vel: Some(Vec3::X), quantized: true };
        let mut b = Vec::new();
        enc_entry(&mut b, &e);
        assert_eq!(b.len(), entry_size(&e));
        let d = dec_entry(&mut Cursor::new(&b, 0)).unwrap();
        assert!((d.pos.unwrap() - e.pos.unwrap()).abs().max_element() <= 0.0005);
        assert!(d.rot.unwrap().angle_between(r).to_degrees() < 0.2);
        let e2 = SnapEntry { quantized: false, ..e };
        let mut b2 = Vec::new();
        enc_entry(&mut b2, &e2);
        let d2 = dec_entry(&mut Cursor::new(&b2, 0)).unwrap();
        assert!(d2.rot.unwrap().angle_between(r).to_degrees() < 0.01);
        assert!(b.len() < b2.len(), "quantized is smaller");

        let mut buf = SnapBuffer::default();
        for k in 0..5 {
            let t = k as f64 * 0.1;
            buf.snaps.push_back(Snap { time: t, pos: Vec3::X * k as f32, rot: Quat::IDENTITY, scale: Vec3::ONE, vel: Vec3::X * 10.0 });
        }
        let mid = sample_snapshots(&buf, 0.25, InterpolationMode::Interpolate).unwrap();
        assert!((mid.pos.x - 2.5).abs() < 1e-4);
        let ahead = sample_snapshots(&buf, 0.45, InterpolationMode::Extrapolate).unwrap();
        assert!((ahead.pos.x - 4.5).abs() < 1e-3);
        // Starved Interpolate extrapolates briefly, then holds.
        let brief = sample_snapshots(&buf, 0.45, InterpolationMode::Interpolate).unwrap();
        assert!((brief.pos.x - 4.5).abs() < 1e-3);
        let held = sample_snapshots(&buf, 0.9, InterpolationMode::Interpolate).unwrap();
        assert!((held.pos.x - 5.0).abs() < 1e-3, "capped at LOSS_EXTRAPOLATION");
        // Without sent velocities the last two snapshots give it; a rest
        // snapshot (same position) stops it.
        for s in buf.snaps.iter_mut() {
            s.vel = Vec3::ZERO;
        }
        let est = sample_snapshots(&buf, 0.45, InterpolationMode::Interpolate).unwrap();
        assert!((est.pos.x - 4.5).abs() < 1e-3);
        buf.snaps.push_back(Snap { time: 0.5, pos: Vec3::X * 4.0, rot: Quat::IDENTITY, scale: Vec3::ONE, vel: Vec3::ZERO });
        assert_eq!(sample_snapshots(&buf, 0.6, InterpolationMode::Interpolate).unwrap().pos.x, 4.0);

        let many: Vec<SnapEntry> = (0..200).map(|i| SnapEntry { id: i, ..e }).collect();
        let packets = build_snap_packets(1, 0.0, &many);
        assert!(packets.len() > 1 && packets.iter().all(|p| p.len() <= MTU), "batched under the MTU");
    }

    // End-to-end over real UDP loopback: handshake, ownership, transform, chat.
    #[test]
    fn udp_loopback_session() {
        let mut host = NetSession::host_with(0, cfg("Host")).expect("host");
        let port = host.local_addr().expect("addr").port();
        let mut client = NetSession::join_with(&format!("127.0.0.1:{port}"), cfg("Client")).expect("join");
        let settle = |a: &mut NetSession, b: &mut NetSession| {
            for _ in 0..40 {
                a.pump();
                b.pump();
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        settle(&mut host, &mut client);
        assert_eq!(client.local_peer(), 2, "host assigned the client a peer id");
        let id = ObjectId::new();
        client.request_ownership(id);
        settle(&mut host, &mut client);
        assert!(client.owns(id));
        assert_eq!(host.owner_of(id), 2);
        client.send_transform(id, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);
        settle(&mut host, &mut client);
        let rt = host.remote_transform(id).expect("host got transform");
        assert!((rt.translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-3);
        client.send_message(None, "hello");
        settle(&mut host, &mut client);
        assert!(host.view().messages.iter().any(|(_, _, t)| t == "hello"));
    }

    #[test]
    fn join_secret_gates_the_session_and_forged_packets_are_dropped() {
        let hub = MemHub::new();
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), NetConfig { secret: "open sesame".into(), ..cfg("Host") });
        h.use_manual_clock();
        let join = |secret: &str, name: &str| {
            let mut c = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), NetConfig { secret: secret.into(), ..cfg(name) });
            c.use_manual_clock();
            c
        };
        let mut bad = join("open sesame?", "Mallory");
        let mut good = join("open sesame", "Alice");
        step(&mut [&mut h, &mut bad, &mut good], 20);
        assert!(matches!(bad.state(), ConnState::Disconnected(r) if r.contains("wrong session secret")), "{:?}", bad.state());
        assert!(good.is_connected());
        assert_eq!(h.peer_count(), 2, "only the peer with the secret joined");

        // Stray, forged and replayed datagrams change nothing.
        let mut evil = hub.endpoint();
        let host_addr = h.local_addr().unwrap();
        let rejected = h.stats.rejected;
        evil.send(&[P_SECURE, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], host_addr);
        evil.send(&[P_REL, 0, 0, 0, 0, 1, M_TEXT], host_addr);
        evil.send(&[P_BYE], host_addr);
        let mut buf = [0u8; 64];
        let _ = evil.recv(&mut buf);
        step(&mut [&mut h, &mut good], 5);
        assert!(h.stats.rejected >= rejected + 3);
        assert!(good.is_connected() && h.peer_count() == 2);
        // A captured sealed packet replayed from the same address is a duplicate.
        good.send_message(None, "once");
        let sealed = {
            let mut hb = hub.inner.lock().unwrap();
            let q = hb.queues.get_mut(&host_addr).unwrap();
            let (d, from, at) = q.back().cloned().unwrap();
            q.push_back((d, from, at));
            from
        };
        assert_eq!(sealed, good.local_addr().unwrap());
        step(&mut [&mut h, &mut good], 5);
        let texts: Vec<String> = h.view().messages.into_iter().map(|m| m.2).collect();
        assert_eq!(texts, vec!["once".to_string()], "the replayed copy was dropped");
    }

    #[test]
    fn rtt_jitter_and_loss_are_measured_both_ways() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "A");
        hub.set_latency(0.04, 0.0);
        for _ in 0..60 {
            hub.advance(DT);
            step(&mut [&mut h, &mut a], 1);
        }
        assert!(a.is_connected());
        hub.set_loss(0.1);
        for k in 0..900 {
            hub.advance(DT);
            step(&mut [&mut h, &mut a], 1);
            // Steady unreliable traffic both ways.
            let id = ObjectId::from_raw(k as u128 + 1);
            a.send_snapshot(&[SnapEntry { id: id.raw(), pos: Some(Vec3::X), ..Default::default() }]);
            h.send_snapshot(&[SnapEntry { id: id.raw(), pos: Some(Vec3::Y), ..Default::default() }]);
        }
        let q = a.quality();
        assert!((q.rtt_ms - 80.0).abs() < 25.0, "rtt {q:?}");
        assert!(q.loss_in > 0.04 && q.loss_in < 0.2, "incoming loss {q:?}");
        assert!(q.loss_out > 0.04 && q.loss_out < 0.2, "outgoing loss reported back {q:?}");
        let v = a.view();
        assert!((v.rtt_ms - q.rtt_ms).abs() < 1e-3 && v.quality == q, "exposed to gameplay");
        let host_view = h.peer_infos().into_iter().find(|p| p.id == 2).unwrap();
        assert!(host_view.rtt_ms > 40.0 && host_view.loss > 0.05, "{host_view:?}");
    }

    #[test]
    fn join_targets_parse() {
        let r = parse_relay("127.0.0.1").unwrap();
        assert_eq!(r.port(), relay_proto::DEFAULT_PORT);
        assert_eq!(JoinTarget::parse("10.0.0.2:9000", "").unwrap(), JoinTarget::Direct("10.0.0.2:9000".parse().unwrap()));
        assert_eq!(
            JoinTarget::parse("abc-234", "127.0.0.1:5000").unwrap(),
            JoinTarget::Room { code: "ABC234".into(), relay: "127.0.0.1:5000".parse().unwrap() }
        );
        assert_eq!(
            JoinTarget::parse("XYZ789@127.0.0.1", "").unwrap(),
            JoinTarget::Room { code: "XYZ789".into(), relay: "127.0.0.1:47800".parse().unwrap() }
        );
        assert!(JoinTarget::parse("ABC234", "").is_err(), "a room code needs a relay");
    }

    fn pump_with_relay(relay: &mut zestengine_relay::Relay, all: &mut [&mut NetSession], until: impl Fn(&[&mut NetSession]) -> bool) -> bool {
        for _ in 0..1500 {
            relay.poll();
            for s in all.iter_mut() {
                s.pump();
            }
            if until(all) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    }

    fn relay_session(punch: bool) -> (zestengine_relay::Relay, NetSession, NetSession) {
        let mut relay = zestengine_relay::Relay::bind("127.0.0.1:0", zestengine_relay::RelayConfig::default()).expect("relay");
        let ra = relay.local_addr().unwrap().to_string();
        let net_cfg = |name: &str| NetConfig { relay: ra.clone(), punch, secret: "s3cret".into(), ..cfg(name) };
        let mut host = NetSession::host_with(0, net_cfg("Host")).expect("host");
        assert!(pump_with_relay(&mut relay, &mut [&mut host], |s| s[0].room_code().is_some()), "the relay assigned a room");
        let code = host.room_code().unwrap();
        assert_eq!(code.len(), 6);
        let mut client = NetSession::join_with(&code.to_lowercase(), net_cfg("Client")).expect("join by code");
        assert!(pump_with_relay(&mut relay, &mut [&mut host, &mut client], |s| s[1].is_connected()), "joined through the relay");
        (relay, host, client)
    }

    // In-process relay over UDP loopback, punching off: everything is relayed.
    #[test]
    fn relay_forwards_a_session_by_room_code() {
        let (mut relay, mut host, mut client) = relay_session(false);
        assert_eq!(client.local_peer(), 2);
        assert_eq!(host.peer_route(2), Some("relayed"));
        assert_eq!(client.peer_route(HOST_PEER), Some("relayed"));
        client.send_message(None, "via relay");
        host.send_message(None, "back");
        let mut got_host = false;
        let mut got_client = false;
        for _ in 0..500 {
            relay.poll();
            host.pump();
            client.pump();
            got_host |= host.view().messages.iter().any(|m| m.2 == "via relay");
            got_client |= client.view().messages.iter().any(|m| m.2 == "back");
            if got_host && got_client {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(got_host && got_client, "chat crossed the relay both ways");
        assert!(relay.stats().packets_relayed > 4 && client.quality().relayed);
        // A wrong secret is refused end to end, through the relay.
        let ra = relay.local_addr().unwrap().to_string();
        let code = host.room_code().unwrap();
        let mut intruder = NetSession::join_with(&format!("{code}@{ra}"), NetConfig { relay: String::new(), secret: "guess".into(), ..cfg("X") }).unwrap();
        assert!(pump_with_relay(&mut relay, &mut [&mut host, &mut client, &mut intruder], |s| matches!(s[2].state(), ConnState::Disconnected(_))));
        assert_eq!(host.peer_count(), 2);
    }

    // Loopback always punches through: the route upgrades to direct and the
    // session keeps working.
    #[test]
    fn hole_punch_upgrades_to_a_direct_path() {
        let (mut relay, mut host, mut client) = relay_session(true);
        assert!(
            pump_with_relay(&mut relay, &mut [&mut host, &mut client], |s| s[0].peer_route(2) == Some("direct") && s[1].peer_route(HOST_PEER) == Some("direct")),
            "both sides went direct"
        );
        let relayed_before = relay.stats().packets_relayed;
        client.send_message(None, "direct");
        let mut got = false;
        for _ in 0..300 {
            relay.poll();
            host.pump();
            client.pump();
            if host.view().messages.iter().any(|m| m.2 == "direct") {
                got = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(got);
        assert!(relay.stats().packets_relayed - relayed_before < 4, "traffic left the relay");
        assert!(!client.quality().relayed);
    }

    #[test]
    fn prefix_lengths_become_netmasks_and_broadcasts() {
        use std::net::Ipv4Addr as A;
        assert_eq!(mask_from_prefix(24), A::new(255, 255, 255, 0));
        assert_eq!(mask_from_prefix(20), A::new(255, 255, 240, 0));
        assert_eq!(mask_from_prefix(0), A::UNSPECIFIED);
        assert_eq!(mask_from_prefix(32), A::BROADCAST);
        assert_eq!(mask_from_prefix(200), A::BROADCAST, "clamped");
        assert_eq!(directed_broadcast(A::new(10, 1, 1, 1), mask_from_prefix(20)), Some(A::new(10, 1, 15, 255)));
    }

    #[test]
    fn directed_broadcast_uses_the_real_netmask() {
        use std::net::Ipv4Addr as A;
        assert_eq!(directed_broadcast(A::new(10, 1, 1, 1), A::new(255, 255, 240, 0)), Some(A::new(10, 1, 15, 255)));
        assert_eq!(directed_broadcast(A::new(192, 168, 1, 20), A::new(255, 255, 255, 0)), Some(A::new(192, 168, 1, 255)));
        assert_eq!(directed_broadcast(A::new(10, 0, 0, 1), A::new(255, 255, 255, 254)), None, "/31");
        assert_eq!(directed_broadcast(A::new(10, 0, 0, 1), A::new(255, 255, 255, 255)), None, "/32");
    }

    #[cfg(unix)]
    #[test]
    fn two_hosts_can_share_the_discovery_port() {
        let port = 48000 + (std::process::id() % 900) as u16;
        let a = bind_shared_udp(port).expect("first bind");
        let b = bind_shared_udp(port).expect("second bind on the same port");
        drop((a, b));
    }

    #[test]
    fn lan_discovery_finds_host() {
        let port = 47000 + (std::process::id() % 900) as u16;
        let mut host = NetSession::host_with(0, NetConfig { session_name: "Lan Test".into(), discovery_port: port, ..cfg("Host") })
            .expect("host");
        let Ok(mut browser) = LanBrowser::with_port(port) else {
            return;
        };
        let mut found = false;
        for _ in 0..50 {
            browser.refresh();
            host.pump();
            std::thread::sleep(std::time::Duration::from_millis(5));
            if browser.poll().iter().any(|s| s.name == "Lan Test") {
                found = true;
                break;
            }
        }
        assert!(found || host.discovery.is_none(), "discovery answered");
    }

    // ------------------------------------------------------------ host migration

    /// Short timeouts so migrations happen in a few simulated seconds.
    fn mcfg(name: &str) -> NetConfig {
        NetConfig { timeout: 2.0, migrate_attempt: 1.0, identity: Identity::generate(), ..cfg(name) }
    }

    fn mhost(hub: &MemHub) -> NetSession {
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), mcfg("Host"));
        h.use_manual_clock();
        h
    }

    fn mjoin(hub: &MemHub, addr: SocketAddr, cfg: NetConfig) -> NetSession {
        let mut c = NetSession::join_on(Box::new(hub.endpoint()), addr, cfg);
        c.use_manual_clock();
        c
    }

    /// Step everyone (and the hub's clock); whoever hosts keeps sending a
    /// snapshot of `obj`, so clients can estimate its clock.
    fn mstep(hub: &MemHub, all: &mut [&mut NetSession], obj: u128, frames: usize) {
        for _ in 0..frames {
            hub.advance(DT);
            for s in all.iter_mut() {
                s.advance(DT);
                s.pump();
                if s.is_host() && s.is_connected() {
                    s.send_snapshot(&[SnapEntry { id: obj, pos: Some(Vec3::X), ..Default::default() }]);
                }
            }
        }
    }

    fn names(s: &NetSession) -> Vec<String> {
        s.peer_infos().into_iter().map(|p| p.name).collect()
    }

    /// The host crashes with three clients: the first in line takes over,
    /// the others reconnect to it, ownership, spawns and variables carry on,
    /// unowned objects move to the new host, the shared (water) clock never
    /// jumps, gameplay gets the events, and the new host serves normally.
    #[test]
    fn host_crash_hands_the_session_to_the_next_in_line() {
        let hub = MemHub::new();
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let mut b = mjoin(&hub, ha, mcfg("Bob"));
        let mut c = mjoin(&hub, ha, mcfg("Cy"));
        let clock = ObjectId::new().raw();
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c], clock, 20);
        assert_eq!((a.local_peer(), b.local_peer(), c.local_peer()), (2, 3, 4));
        let bobs = ObjectId::new();
        b.request_ownership(bobs);
        let crate_ = ObjectId::new();
        h.spawn(NetSpawnSource::Prefab("prefabs/crate.prefab".into()), crate_, Vec3::ZERO, Quat::IDENTITY, 0);
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c], clock, 10);
        b.set_var(bobs, "hp", NetValue::Int(80));
        h.set_var(crate_, "loot", NetValue::Int(3));
        b.send_message(None, "before");
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c], clock, 10);
        assert_eq!(b.next_host(), Some(2), "everyone knows who is next");
        assert_eq!(h.next_host(), Some(2));
        for s in [&mut a, &mut b, &mut c] {
            events(s);
        }
        let mut prev: Vec<f64> = [&a, &b, &c].iter().map(|s| s.shared_clock().expect("host clock known")).collect();
        h.crash();
        hub.cut(ha);
        // In flight when the host died: these reach everyone exactly once,
        // through the new host.
        b.send_message(None, "during");
        b.set_var(bobs, "hp", NetValue::Int(70));
        a.send_message(None, "from ann");
        for frame in 0..(6.0 / DT) as usize {
            mstep(&hub, &mut [&mut a, &mut b, &mut c], clock, 1);
            for (k, s) in [&a, &b, &c].iter().enumerate() {
                let t = s.shared_clock().expect("the clock runs on");
                let d = t - prev[k];
                assert!((DT * 0.49..=DT * 1.51).contains(&d), "peer {k}: clock step {d} at frame {frame}");
                prev[k] = t;
            }
        }
        assert!(a.is_host() && a.host_peer() == 2 && a.epoch() == 1, "Ann took over");
        assert!(a.reconnecting().is_empty(), "everyone came back");
        for s in [&b, &c] {
            assert!(s.is_connected() && !s.is_host() && s.host_peer() == 2 && s.epoch() == 1);
            assert!(s.quality().rtt_ms > 0.0);
        }
        for s in [&a, &b, &c] {
            assert_eq!(names(s), vec!["Ann", "Bob", "Cy"], "same player list, same peer ids");
            assert_eq!(s.owner_of(bobs), 3, "ownership kept");
            assert_eq!(s.owner_of(crate_), 0);
            assert_eq!(s.world.var(crate_.raw(), "loot"), Some(&NetValue::Int(3)));
            assert!(s.world.spawns.iter().any(|sp| sp.id == crate_.raw()));
        }
        assert!(a.has_authority_raw(crate_.raw()) && !b.has_authority_raw(crate_.raw()), "unowned objects follow the host");
        let ev = events(&mut b);
        assert!(ev.iter().any(|e| matches!(e, NetEvent::HostMigrating { previous: 1 })));
        assert!(ev.iter().any(|e| matches!(e, NetEvent::HostChanged { host: 2, previous: 1 })));
        assert!(ev.iter().any(|e| matches!(e, NetEvent::PeerLeft { id: 1, .. })));
        let v = a.view();
        assert!(v.events.iter().any(|e| matches!(e, NetEvent::HostChanged { host: 2, .. })) && v.is_server && v.host_peer == 2);
        let (ta, tb) = (a.shared_clock().unwrap(), b.shared_clock().unwrap());
        assert!((ta - tb).abs() < 0.05, "clocks agree: {ta} vs {tb}");
        let heard: Vec<String> = c.view().messages.into_iter().map(|m| m.2).collect();
        assert_eq!(heard.iter().filter(|m| *m == "during").count(), 1, "resent once: {heard:?}");
        assert_eq!(heard.iter().filter(|m| *m == "from ann").count(), 1, "the new host's own unsent message: {heard:?}");
        assert!(!heard.iter().any(|m| m == "before"), "nothing delivered twice: {heard:?}");
        for s in [&a, &c] {
            assert_eq!(s.world.var(bobs.raw(), "hp"), Some(&NetValue::Int(70)), "the in-flight write landed");
        }
        // The new host serves: ownership requests, variables, chat.
        c.request_ownership(crate_);
        mstep(&hub, &mut [&mut a, &mut b, &mut c], clock, 10);
        assert_eq!(b.owner_of(crate_), 4);
        c.set_var(crate_, "loot", NetValue::Int(1));
        b.send_message(None, "still here");
        mstep(&hub, &mut [&mut a, &mut b, &mut c], clock, 10);
        assert!(c.view().messages.iter().any(|m| m.2 == "still here"));
        assert_eq!(b.world.var(crate_.raw(), "loot"), Some(&NetValue::Int(1)));
    }

    /// A host that quits hands off at once (no timeout) with its latest
    /// state; its own player object goes, the others' stay theirs.
    #[test]
    fn graceful_handoff_moves_the_session_at_once() {
        let hub = MemHub::new();
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let mut b = mjoin(&hub, ha, mcfg("Bob"));
        let clock = ObjectId::new().raw();
        mstep(&hub, &mut [&mut h, &mut a, &mut b], clock, 20);
        let mine = h.spawn_everywhere(NetSpawnSource::Prefab("prefabs/player.prefab".into()), Vec3::ZERO, Quat::IDENTITY, 1);
        let anns = h.spawn_everywhere(NetSpawnSource::Prefab("prefabs/player.prefab".into()), Vec3::X, Quat::IDENTITY, 2);
        mstep(&hub, &mut [&mut h, &mut a, &mut b], clock, 10);
        // Written in the host's last frame: the handoff carries it.
        h.set_var(anns, "score", NetValue::Int(5));
        h.disconnect();
        drop(h);
        mstep(&hub, &mut [&mut a, &mut b], clock, 30);
        assert!(a.is_host() && b.is_connected() && b.host_peer() == 2, "half a second, no timeout: {:?}", b.state());
        for s in [&a, &b] {
            assert!(s.world.spawns.iter().all(|sp| sp.id != mine.raw()), "the old host's player is gone");
            assert!(s.world.spawns.iter().any(|sp| sp.id == anns.raw() && sp.player && sp.owner == 2));
            assert_eq!(s.world.var(anns.raw(), "score"), Some(&NetValue::Int(5)));
            assert_eq!(names(s), vec!["Ann", "Bob"]);
        }
        // The new host still despawns a leaving player's object.
        assert_eq!(a.world.player_obj.get(&2), Some(&anns.raw()));
    }

    /// The host and its first successor die together: the next in line
    /// takes over; then that one dies too and the next after it hosts.
    #[test]
    fn when_the_successor_fails_too_the_next_in_line_hosts() {
        let hub = MemHub::new();
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let mut b = mjoin(&hub, ha, mcfg("Bob"));
        let mut c = mjoin(&hub, ha, mcfg("Cy"));
        let mut d = mjoin(&hub, ha, mcfg("Di"));
        let clock = ObjectId::new().raw();
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c, &mut d], clock, 20);
        let item = ObjectId::new();
        d.request_ownership(item);
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c, &mut d], clock, 10);
        h.crash();
        a.crash();
        hub.cut(ha);
        hub.cut(a.local_addr().unwrap());
        mstep(&hub, &mut [&mut b, &mut c, &mut d], clock, (12.0 / DT) as usize);
        assert!(b.is_host() && b.host_peer() == 3);
        for s in [&c, &d] {
            assert!(s.is_connected() && s.host_peer() == 3, "{:?}", s.state());
        }
        assert_eq!(names(&c), vec!["Bob", "Cy", "Di"], "Ann never came back");
        assert_eq!(c.owner_of(item), 5);
        // Bob dies after taking over: Cy is next.
        b.crash();
        hub.cut(b.local_addr().unwrap());
        mstep(&hub, &mut [&mut c, &mut d], clock, (6.0 / DT) as usize);
        assert!(c.is_host() && c.epoch() == 2);
        assert!(d.is_connected() && d.host_peer() == 4 && d.owner_of(item) == 5);
    }

    /// A partition makes two members take over at once; once they can see
    /// each other, the lower-ranked one hands its members to the
    /// higher-ranked one and joins it (deterministic tie-break).
    #[test]
    fn two_promoted_hosts_settle_on_the_better_ranked_one() {
        let hub = MemHub::new();
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let mut b = mjoin(&hub, ha, mcfg("Bob"));
        let mut c = mjoin(&hub, ha, mcfg("Cy"));
        let mut d = mjoin(&hub, ha, mcfg("Di"));
        let clock = ObjectId::new().raw();
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c, &mut d], clock, 20);
        let (aa, ba, da) = (a.local_addr().unwrap(), b.local_addr().unwrap(), d.local_addr().unwrap());
        h.crash();
        hub.cut(ha);
        hub.cut_link(ba, aa);
        hub.cut_link(da, aa);
        mstep(&hub, &mut [&mut a, &mut b, &mut c, &mut d], clock, (5.0 / DT) as usize);
        assert!(a.is_host() && b.is_host(), "split: both took over");
        assert_eq!((c.host_peer(), d.host_peer()), (2, 3));
        hub.heal();
        hub.cut(ha);
        mstep(&hub, &mut [&mut a, &mut b, &mut c, &mut d], clock, (5.0 / DT) as usize);
        assert!(a.is_host() && !b.is_host(), "Bob stepped down for Ann");
        for s in [&b, &c, &d] {
            assert!(s.is_connected() && s.host_peer() == 2, "{:?}", s.state());
        }
        assert!(a.reconnecting().is_empty());
        assert_eq!(names(&d), vec!["Ann", "Bob", "Cy", "Di"]);
    }

    /// Someone joining while the host is being replaced gets in once the
    /// successor took over, with a fresh peer id and the whole world.
    #[test]
    fn a_late_joiner_during_a_migration_joins_the_new_host() {
        let hub = MemHub::new();
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let mut b = mjoin(&hub, ha, mcfg("Bob"));
        let clock = ObjectId::new().raw();
        mstep(&hub, &mut [&mut h, &mut a, &mut b], clock, 20);
        let item = ObjectId::new();
        h.spawn(NetSpawnSource::Prefab("prefabs/crate.prefab".into()), item, Vec3::ZERO, Quat::IDENTITY, 0);
        mstep(&hub, &mut [&mut h, &mut a, &mut b], clock, 10);
        h.crash();
        hub.cut(ha);
        let mut late = mjoin(&hub, a.local_addr().unwrap(), NetConfig { timeout: 10.0, ..mcfg("Late") });
        mstep(&hub, &mut [&mut a, &mut b, &mut late], clock, (5.0 / DT) as usize);
        assert!(a.is_host());
        assert!(late.is_connected() && late.local_peer() == 4, "fresh id after everyone known: {}", late.local_peer());
        assert_eq!(names(&late), vec!["Ann", "Bob", "Late"]);
        assert!(late.world.spawns.iter().any(|s| s.id == item.raw()), "full world");
        assert_eq!(names(&b), vec!["Ann", "Bob", "Late"]);
    }

    // ------------------------------------------------------------ security

    /// Forged rejections during the join, and forged BYEs / kicks / rejects
    /// afterwards (spoofed as the host), change nothing; a real kick works
    /// and doesn't start a migration.
    #[test]
    fn forged_rejects_and_kicks_are_ignored() {
        let hub = MemHub::new();
        hub.set_latency(0.05, 0.0);
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let mut b = mjoin(&hub, ha, mcfg("Bob"));
        let aa = a.local_addr().unwrap();
        let mut hint = vec![P_REJECT, REJECT_HINT];
        put_u16(&mut hint, PROTOCOL);
        put_str(&mut hint, "go away");
        let mut signed = vec![P_REJECT, REJECT_SIGNED, 1];
        put_str(&mut signed, "wrong session secret");
        signed.extend_from_slice(&[7u8; 64]);
        hub.inject(ha, aa, &hint);
        hub.inject(ha, aa, &signed);
        hub.inject(ha, aa, &[P_REJECT, 4, 0, b'n', b'o', b'p', b'e']);
        mstep(&hub, &mut [&mut h, &mut a, &mut b], 0, 30);
        assert!(a.is_connected(), "forged rejections didn't stop the join: {:?}", a.state());
        let rejected = a.stats.rejected;
        let mut kick = vec![P_BYE, BYE_KICKED];
        put_str(&mut kick, "forged");
        hub.inject(ha, aa, &kick);
        hub.inject(ha, aa, &[P_BYE]);
        hub.inject(ha, aa, &signed);
        let mut fake = vec![P_SECURE];
        fake.extend_from_slice(&[0u8; 40]);
        hub.inject(ha, aa, &fake);
        mstep(&hub, &mut [&mut h, &mut a, &mut b], 0, 30);
        assert!(a.is_connected() && h.peer_count() == 3);
        assert!(a.stats.rejected >= rejected + 4, "{} -> {}", rejected, a.stats.rejected);
        h.kick(2, "bye");
        mstep(&hub, &mut [&mut h, &mut a, &mut b], 0, 30);
        assert!(matches!(a.state(), ConnState::Disconnected(r) if r.contains("kicked by the host: bye")), "{:?}", a.state());
        assert!(a.migration_status().is_none(), "a kicked client doesn't look for a new host");
        assert!(b.is_connected() && h.peer_count() == 2);
    }

    /// An on-path attacker that races a forged CHALLENGE (a cookie the host
    /// never issued) ahead of the host's only delays the join: the client
    /// keeps the host's own CHALLENGE and joins with it.
    #[test]
    fn a_raced_forged_challenge_does_not_stall_the_join() {
        let hub = MemHub::new();
        hub.set_latency(0.05, 0.0);
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, mcfg("Ann"));
        let forged = h.challenge_bytes(&[9u8; 16]);
        hub.inject(ha, a.local_addr().unwrap(), &forged);
        mstep(&hub, &mut [&mut h, &mut a], 0, 150);
        assert!(a.is_connected(), "joined despite the forgery: {:?}", a.state());
        assert_eq!(h.peer_count(), 2);
    }

    /// Passphrases go through Argon2id with the host's parameters (carried
    /// in the CHALLENGE); a wrong one is refused with a signed REJECT; a
    /// host offering weak stretching for a passphrase is refused by the
    /// client; generated secrets skip stretching.
    #[test]
    fn passphrases_are_stretched_with_the_hosts_argon2_parameters() {
        use crate::net_secure::MIN_M_KIB;
        let hub = MemHub::new();
        let params = KdfParams { mode: KdfMode::Stretched, m_kib: MIN_M_KIB, passes: 2, ..KdfParams::for_secret("correct horse") };
        let host_with = |secret: &str, kdf: Option<KdfParams>| {
            let mut h = NetSession::host_on(Box::new(hub.endpoint()), NetConfig { secret: secret.into(), kdf, ..mcfg("Host") });
            h.use_manual_clock();
            h
        };
        let mut h = host_with("correct horse", Some(params));
        let ha = h.local_addr().unwrap();
        let mut good = mjoin(&hub, ha, NetConfig { secret: "correct horse".into(), ..mcfg("Good") });
        let mut bad = mjoin(&hub, ha, NetConfig { secret: "correct horse!".into(), ..mcfg("Bad") });
        mstep(&hub, &mut [&mut h, &mut good, &mut bad], 0, 30);
        assert!(good.is_connected());
        assert_eq!(good.kdf(), params, "the client used exactly the host's Argon2id parameters and salt");
        assert!(matches!(bad.state(), ConnState::Disconnected(r) if r.contains("wrong session secret")), "{:?}", bad.state());
        assert_eq!(h.peer_count(), 2);

        let weak = KdfParams { m_kib: 1024, passes: 1, ..params };
        let mut h2 = host_with("pass phrase", Some(weak));
        let mut c2 = mjoin(&hub, h2.local_addr().unwrap(), NetConfig { secret: "pass phrase".into(), ..mcfg("C2") });
        mstep(&hub, &mut [&mut h2, &mut c2], 0, (3.0 / DT) as usize);
        assert!(matches!(c2.state(), ConnState::Disconnected(r) if r.contains("weak key stretching")), "{:?}", c2.state());

        let g = generate_secret();
        let mut h3 = host_with(&g, None);
        let mut c3 = mjoin(&hub, h3.local_addr().unwrap(), NetConfig { secret: g.clone(), ..mcfg("C3") });
        mstep(&hub, &mut [&mut h3, &mut c3], 0, 20);
        assert!(c3.is_connected() && c3.kdf().mode == KdfMode::Random);
    }

    /// The host learns every player's verified id; clients see them too;
    /// bans kick and keep a player out, allow lists keep everyone else
    /// out, and a new host after a migration keeps enforcing the bans.
    #[test]
    fn player_ids_are_verified_and_ban_and_allow_lists_enforced() {
        let hub = MemHub::new();
        let (ann, bob) = (Identity::generate(), Identity::generate());
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut a = mjoin(&hub, ha, NetConfig { identity: ann.clone(), ..mcfg("Ann") });
        let mut b = mjoin(&hub, ha, NetConfig { identity: bob.clone(), ..mcfg("Bob") });
        let mut c = mjoin(&hub, ha, mcfg("Cy"));
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c], 0, 20);
        assert_eq!(h.peer_identity(2), Some(ann.public()));
        assert_eq!(h.peer_fingerprint(3), Some(bob.fingerprint()));
        let seen = c.peer_infos().into_iter().find(|p| p.id == 3).unwrap();
        assert_eq!(seen.fingerprint, bob.fingerprint(), "clients learn the verified ids");
        assert_eq!(seen.identity, Some(bob.public()));
        h.ban(3);
        mstep(&hub, &mut [&mut h, &mut a, &mut b, &mut c], 0, 20);
        assert!(matches!(b.state(), ConnState::Disconnected(r) if r.contains("banned")), "{:?}", b.state());
        let mut again = mjoin(&hub, ha, NetConfig { identity: bob.clone(), ..mcfg("Bob again") });
        mstep(&hub, &mut [&mut h, &mut a, &mut c, &mut again], 0, 20);
        assert!(matches!(again.state(), ConnState::Disconnected(r) if r.contains("banned from this session")), "{:?}", again.state());
        // The ban travels with the succession plan.
        h.crash();
        hub.cut(ha);
        mstep(&hub, &mut [&mut a, &mut c], 0, (5.0 / DT) as usize);
        assert!(a.is_host() && c.host_peer() == 2);
        let mut third = mjoin(&hub, a.local_addr().unwrap(), NetConfig { identity: bob.clone(), ..mcfg("Bob 3") });
        mstep(&hub, &mut [&mut a, &mut c, &mut third], 0, 20);
        assert!(matches!(third.state(), ConnState::Disconnected(r) if r.contains("banned")), "{:?}", third.state());
        // Allow list: only listed ids get in.
        let mut h2 = NetSession::host_on(Box::new(hub.endpoint()), NetConfig { allow: vec![ann.fingerprint()], ..mcfg("Host 2") });
        h2.use_manual_clock();
        let h2a = h2.local_addr().unwrap();
        let mut a2 = mjoin(&hub, h2a, NetConfig { identity: ann.clone(), ..mcfg("Ann") });
        let mut b2 = mjoin(&hub, h2a, NetConfig { identity: bob.clone(), ..mcfg("Bob") });
        mstep(&hub, &mut [&mut h2, &mut a2, &mut b2], 0, 20);
        assert!(a2.is_connected());
        assert!(matches!(b2.state(), ConnState::Disconnected(r) if r.contains("allow list")), "{:?}", b2.state());
    }

    /// HELLOs leave no state and get replies no larger than themselves;
    /// JOINs with a made-up cookie are dropped silently; joins that never
    /// confirm are capped and expire.
    #[test]
    fn handshakes_keep_no_state_until_a_valid_join_and_half_open_joins_are_capped() {
        let hub = MemHub::new();
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), NetConfig { max_players: 200, ..mcfg("Host") });
        h.use_manual_clock();
        let ha = h.local_addr().unwrap();
        let mut probe = hub.endpoint();
        let hello = hello_packet(&[1; 16], JOIN_MODE_JOIN);
        for _ in 0..50 {
            probe.send(&hello, ha);
        }
        let mut short = vec![P_HELLO];
        put_u16(&mut short, PROTOCOL);
        probe.send(&short, ha);
        h.advance(DT);
        h.pump();
        let mut buf = [0u8; 2048];
        let mut replies = 0;
        while let Some((len, _)) = probe.recv(&mut buf) {
            assert!(len <= hello.len(), "reply {len} > request {}", hello.len());
            assert_eq!(buf[0], P_CHALLENGE);
            replies += 1;
        }
        assert_eq!(replies, 50, "the unpadded HELLO got nothing");
        assert_eq!((h.peer_count(), h.half_open()), (1, 0), "no state for HELLOs");
        let mut join = vec![P_JOIN];
        join.extend_from_slice(&[1; 16]);
        join.extend_from_slice(&[9; 16]);
        join.push(JOIN_MODE_JOIN);
        join.extend_from_slice(&[0; 80]);
        let rejected = h.stats.rejected;
        probe.send(&join, ha);
        h.advance(DT);
        h.pump();
        assert!(probe.recv(&mut buf).is_none() && h.stats.rejected == rejected + 1, "bad cookie: dropped, no reply");
        // Joiners that never confirm their key: only MAX_HALF_OPEN are held.
        let mut ghosts: Vec<NetSession> = (0..40).map(|k| mjoin(&hub, ha, mcfg(&format!("g{k}")))).collect();
        h.advance(DT);
        h.pump();
        for g in ghosts.iter_mut() {
            g.advance(DT);
            g.pump();
        }
        h.advance(DT);
        h.pump();
        assert_eq!(h.half_open(), MAX_HALF_OPEN);
        assert!(h.stats.half_open_refused >= 8);
        for _ in 0..(6.0 / DT) as usize {
            h.advance(DT);
            h.pump();
        }
        assert_eq!((h.half_open(), h.peer_count()), (0, 1), "they expired");
        drop(ghosts);
        let mut real = mjoin(&hub, ha, mcfg("Real"));
        mstep(&hub, &mut [&mut h, &mut real], 0, 20);
        assert!(real.is_connected());
    }

    /// An older client gets a clear version error; a mismatched padded
    /// HELLO gets one that fits inside it.
    #[test]
    fn protocol_mismatch_is_explained() {
        let hub = MemHub::new();
        let mut h = mhost(&hub);
        let ha = h.local_addr().unwrap();
        let mut old = hub.endpoint();
        old.send(&[P_HELLO, 5, 0], ha);
        let mut newer = hello_packet(&[2; 16], JOIN_MODE_JOIN);
        newer[1] = 99;
        old.send(&newer, ha);
        h.advance(DT);
        h.pump();
        let mut buf = [0u8; 2048];
        let (n, _) = old.recv(&mut buf).expect("legacy reject");
        assert!(String::from_utf8_lossy(&buf[3..n]).contains("protocol version mismatch (host 7, you 5)"));
        let (n, _) = old.recv(&mut buf).expect("hint reject");
        assert!(n <= newer.len() && buf[1] == REJECT_HINT);
    }

    // ------------------------------------------------------------ relayed migration

    /// Over UDP with an in-process relay: one client came in by room code
    /// (relayed), one directly. The host crashes; the relayed client is next
    /// in line and claims the room's host slot with the room owner key; the
    /// direct client has no direct path to it, so it falls back to the
    /// relay, joins the room and reaches the new host there.
    #[test]
    fn relayed_migration_moves_the_room_and_direct_clients_fall_back_to_the_relay() {
        let mut relay = zestengine_relay::Relay::bind("127.0.0.1:0", zestengine_relay::RelayConfig::default()).expect("relay");
        let ra = relay.local_addr().unwrap().to_string();
        let secret = generate_secret();
        let c = |name: &str| NetConfig {
            relay: ra.clone(),
            punch: false,
            secret: secret.clone(),
            timeout: 1.5,
            migrate_attempt: 1.0,
            identity: Identity::generate(),
            ..cfg(name)
        };
        let mut host = NetSession::host_with(0, c("Host")).expect("host");
        assert!(pump_with_relay(&mut relay, &mut [&mut host], |s| s[0].room_code().is_some()), "room opened");
        let code = host.room_code().unwrap();
        let mut bea = NetSession::join_with(&code, c("Bea")).expect("join by code");
        assert!(pump_with_relay(&mut relay, &mut [&mut host, &mut bea], |s| s[1].is_connected()));
        let port = host.local_addr().unwrap().port();
        let mut lan = NetSession::join_with(&format!("127.0.0.1:{port}"), c("Lan")).expect("direct join");
        assert!(pump_with_relay(&mut relay, &mut [&mut host, &mut bea, &mut lan], |s| {
            s[2].is_connected() && s[1].succession().is_some_and(|p| p.order.len() == 2 && !p.room.is_empty())
                && s[2].succession().is_some_and(|p| p.room_key.is_some())
        }));
        assert_eq!((bea.local_peer(), lan.local_peer()), (2, 3));
        assert!(lan.relay_status().is_none(), "Lan came in directly");
        host.crash();
        let done = |s: &[&mut NetSession]| s[0].is_host() && s[1].is_connected() && s[1].host_peer() == 2 && s[0].reconnecting().is_empty();
        let mut ok = false;
        for _ in 0..4 {
            if pump_with_relay(&mut relay, &mut [&mut bea, &mut lan], done) {
                ok = true;
                break;
            }
        }
        assert!(ok, "bea host {} / lan {:?} host {}", bea.is_host(), lan.state(), lan.host_peer());
        assert_eq!(relay.stats().claims, 1, "the room moved to the new host");
        assert_eq!(relay.room_info(&code), Some((true, 2)));
        assert_eq!(lan.peer_route(2), Some("relayed"), "fell back to the relay");
        assert!(bea.relay_status().is_some_and(|r| r.host));
        lan.send_message(None, "via the new host");
        let mut got = false;
        for _ in 0..500 {
            relay.poll();
            bea.pump();
            lan.pump();
            if bea.view().messages.iter().any(|m| m.2 == "via the new host") {
                got = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(got, "chat reaches the new host through the relay");
    }

    // ------------------------------------------------------------ anti-cheat (SEC-2)

    use zestengine_core::TestProvider;

    fn ac_cfg(provider: &str, require: bool) -> NetConfig {
        NetConfig { anticheat: AntiCheatConfig { provider: provider.into(), require, pending_timeout: 5.0, ..Default::default() }, ..cfg("Host") }
    }

    fn ac_host(hub: &MemHub, c: NetConfig, p: &TestProvider) -> NetSession {
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), c);
        h.use_manual_clock();
        h.set_anticheat(Box::new(p.clone()));
        h
    }

    #[test]
    fn anticheat_holds_client_until_trusted() {
        let hub = MemHub::new();
        let p = TestProvider::trust_after(1.0);
        let mut h = ac_host(&hub, ac_cfg("test", true), &p);
        let mut a = client(&hub, &h, "Ann");
        step(&mut [&mut h, &mut a], 20);
        assert!(a.is_connected());
        assert_eq!(h.anticheat_verdict(2), Some(Verdict::Pending));
        assert!(h.ac_held(2));
        // A held client's reliable chat is ignored.
        let before = h.stats.muted;
        a.send_message(None, "hello");
        step(&mut [&mut h, &mut a], 10);
        assert!(h.stats.muted > before, "held client is muted");
        step(&mut [&mut h, &mut a], 90);
        assert_eq!(h.anticheat_verdict(2), Some(Verdict::Trusted));
        assert!(!h.ac_held(2));
        assert!(a.is_connected());
        assert_eq!(h.anticheat_pending(), 0);
    }

    /// SEC: with `require`, a client that is not Trusted yet is in the session but out of the
    /// world: no catch-up (scene, variables), no player object, no broadcasts.
    #[test]
    fn anticheat_held_client_gets_no_world_state_or_player_object() {
        let hub = MemHub::new();
        let p = TestProvider::trust_after(1.0);
        let mut h = ac_host(&hub, ac_cfg("test", true), &p);
        h.load_scene("scenes/arena.scene");
        let mut a = client(&hub, &h, "Ann");
        step(&mut [&mut h, &mut a], 30);
        assert!(a.is_connected());
        assert_eq!(h.anticheat_held(), vec![2]);
        assert!(!h.players_to_spawn.contains(&2), "no player object for a held client");
        h.send_message(None, "hello, world");
        h.set_var(ObjectId::from_raw(0xABCD), "secret", NetValue::Int(7));
        h.send_rpc(None, "boss.spawned", &[], RpcTarget::All, true);
        step(&mut [&mut h, &mut a], 20);
        let v = a.view();
        assert!(v.world_ops.is_empty() && v.messages.is_empty() && v.rpcs.is_empty(), "nothing leaked while held: {:?} {:?}", v.world_ops.len(), v.messages);
        assert!(a.world.vars.is_empty(), "no variables either");
        // Trusted: the world arrives and the player object is queued.
        step(&mut [&mut h, &mut a], 90);
        assert_eq!(h.anticheat_verdict(2), Some(Verdict::Trusted));
        assert!(h.anticheat_held().is_empty() && h.players_to_spawn.contains(&2));
        assert!(a.view().world_ops.iter().any(|op| matches!(op, NetWorldOp::LoadScene(s) if s == "scenes/arena.scene")), "the scene reaches it now");
        assert_eq!(a.world.var(0xABCD, "secret"), Some(&NetValue::Int(7)));
    }

    /// SEC: the provider's account id is linked to the player id; a ban covers both, so a fresh
    /// key on a banned account is refused and a fresh key on another account is welcome.
    #[test]
    fn a_banned_provider_account_stays_banned_under_a_new_player_key() {
        let hub = MemHub::new();
        let p = TestProvider::trust_after(0.3);
        let mut h = ac_host(&hub, ac_cfg("test", true), &p);
        let mut cheat = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), NetConfig { identity: Identity::generate(), ..cfg("Cheat") });
        cheat.use_manual_clock();
        step(&mut [&mut h, &mut cheat], 10);
        let peer = cheat.local_peer();
        p.set_provider_id(peer, "acct-77");
        step(&mut [&mut h, &mut cheat], 60);
        let key = h.provider_ban_key(peer).expect("linked once trusted");
        assert_eq!(key, crate::net_identity::provider_fingerprint("test", "acct-77"));
        // The operator bans the player: the account goes with it.
        let fp = h.peer_fingerprint(peer).unwrap();
        h.val.bans.add(&fp, 0, "aimbot", crate::net_validate::unix_now());
        assert!(h.ban_provider_link(peer, 0, "aimbot"));
        h.kick(peer, "banned: aimbot");
        step(&mut [&mut h, &mut cheat], 20);
        assert!(!cheat.is_connected());
        assert!(h.val.bans.is_banned(&key, crate::net_validate::unix_now()).is_some());
        // The same account under a new player key: let in by the handshake, removed when the
        // provider names the account, and the new key is banned too.
        let new_key = Identity::generate();
        let mut evader = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), NetConfig { identity: new_key.clone(), ..cfg("Evader") });
        evader.use_manual_clock();
        step(&mut [&mut h, &mut evader], 10);
        p.set_provider_id(evader.local_peer(), "acct-77");
        step(&mut [&mut h, &mut evader], 60);
        assert!(matches!(evader.state(), ConnState::Disconnected(r) if r.contains("banned account")), "{:?}", evader.state());
        assert!(h.val.bans.is_banned(&new_key.fingerprint(), crate::net_validate::unix_now()).is_some(), "the new key is banned as well");
        // Someone else on another account is fine.
        let mut honest = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), NetConfig { identity: Identity::generate(), ..cfg("Honest") });
        honest.use_manual_clock();
        step(&mut [&mut h, &mut honest], 10);
        p.set_provider_id(honest.local_peer(), "acct-1");
        step(&mut [&mut h, &mut honest], 60);
        assert!(honest.is_connected() && h.provider_ban_key(honest.local_peer()).is_some());
    }

    #[test]
    fn anticheat_pending_timeout_removes_client() {
        let hub = MemHub::new();
        let p = TestProvider::hang();
        let mut h = ac_host(&hub, ac_cfg("test", true), &p);
        let mut a = client(&hub, &h, "Ann");
        step(&mut [&mut h, &mut a], (7.0 / DT) as usize);
        assert!(matches!(a.state(), ConnState::Disconnected(r) if r.contains("anti-cheat")), "{:?}", a.state());
        assert_eq!(h.peer_count(), 1);
        assert!(p.state.lock().unwrap().unregistered.contains(&2));
    }

    #[test]
    fn anticheat_rejection_kicks_and_repeat_bans() {
        let hub = MemHub::new();
        let p = TestProvider::reject("modified client");
        let mut h = ac_host(&hub, ac_cfg("test", true), &p);
        let id = Identity::generate();
        let join = |hub: &MemHub, h: &NetSession| {
            let mut c = NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), NetConfig { identity: id.clone(), ..cfg("Cheat") });
            c.use_manual_clock();
            c
        };
        let mut a = join(&hub, &h);
        step(&mut [&mut h, &mut a], 30);
        assert!(matches!(a.state(), ConnState::Disconnected(r) if r.contains("modified client")), "{:?}", a.state());
        assert_eq!(h.stats.ac_rejected, 1);
        assert!(h.val.total >= 1, "counted as a violation");
        // The same player id comes back and fails again: banned.
        let mut b = join(&hub, &h);
        step(&mut [&mut h, &mut a, &mut b], 60);
        assert_eq!(h.val.bans.entries().len(), 1, "second rejection of the same player id bans it");
    }

    #[test]
    fn anticheat_payloads_round_trip() {
        let hub = MemHub::new();
        let sp = TestProvider::handshake();
        let mut h = ac_host(&hub, ac_cfg("test", true), &sp);
        let cp = TestProvider::hang();
        let mut a = client(&hub, &h, "Ann");
        a.cfg.anticheat.provider = "test".into();
        a.set_anticheat(Box::new(cp.clone()));
        step(&mut [&mut h, &mut a], 60);
        {
            let c = cp.state.lock().unwrap();
            assert!(c.client_started);
            assert_eq!(c.client_received, vec![b"challenge".to_vec()]);
        }
        assert_eq!(sp.state.lock().unwrap().server_received, vec![(2, b"response".to_vec())]);
        assert_eq!(h.anticheat_verdict(2), Some(Verdict::Trusted));
        assert!(a.is_connected());
    }

    #[test]
    fn anticheat_off_or_advisory_changes_nothing() {
        // require = false, provider none: no state, nobody held.
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "Ann");
        step(&mut [&mut h, &mut a], 30);
        assert!(h.anticheat_verdict(2).is_none() && !h.ac_held(2));
        assert_eq!(h.anticheat_pending(), 0);
        // Advisory with a rejecting provider: logged and counted, not held or removed.
        let p = TestProvider::reject("x");
        let hub2 = MemHub::new();
        let mut h2 = ac_host(&hub2, ac_cfg("test", false), &p);
        let mut b = client(&hub2, &h2, "Bob");
        step(&mut [&mut h2, &mut b], 30);
        assert!(b.is_connected());
        assert!(!h2.ac_held(2));
        assert_eq!(h2.stats.ac_rejected, 1);
    }

    #[test]
    fn anticheat_unavailable_provider_fails_closed_when_required() {
        let hub = MemHub::new();
        let mut h = NetSession::host_on(Box::new(hub.endpoint()), ac_cfg("eos", true));
        h.use_manual_clock();
        let mut a = client(&hub, &h, "Ann");
        step(&mut [&mut h, &mut a], 30);
        assert!(matches!(a.state(), ConnState::Disconnected(_)));
    }
}
