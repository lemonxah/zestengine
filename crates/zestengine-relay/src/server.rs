//! The relay server: rooms, introductions and packet forwarding, with
//! authenticated room control and abuse limits.
//!
//! - Every REGISTER first needs a stateless cookie (bound to the sender's
//!   endpoint and the room code), so spoofed sources get nothing but one
//!   COOKIE no larger than what they sent.
//! - Creating a room registers its owner key, join key and key-stretching
//!   parameters (and proves the operator token, when one is configured);
//!   each IP may hold only so many rooms.
//! - Joining proves the session secret (HMAC under the join key); failures
//!   from one IP back off exponentially.
//! - Admitted members that have not exchanged data yet count as pending;
//!   rooms and the server cap them.
//! - When the host leaves or goes idle the room stays open for a grace
//!   period, and a member holding the room owner key can CLAIM the host slot
//!   (host migration).

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::proto::{
    self, COOKIE_LEN, HOST_SLOT, MAX_PAYLOAD, MIN_REGISTER, Packet, RELAY_PROTOCOL, ROLE_CLIENT, ROLE_HOST, RegisterAuth,
};

/// Seconds a cookie stays valid (it rotates; the previous one is accepted too).
const COOKIE_PERIOD: f64 = 30.0;
/// Failed join attempts from one IP before it is throttled.
const FREE_FAILURES: u32 = 4;
/// Longest throttle (seconds).
const MAX_BACKOFF: f64 = 60.0;
/// Throttle entries kept at most (oldest go first).
const MAX_THROTTLED: usize = 65_536;

/// Server limits.
#[derive(Clone, Debug)]
pub struct RelayConfig {
    pub max_rooms: usize,
    /// Members per room, the host included.
    pub max_members: usize,
    /// Seconds without a packet before a member is dropped.
    pub idle_timeout: f64,
    /// Relayed bytes per second a room host may send (bursts up to one second's worth).
    pub host_rate: f64,
    /// Relayed bytes per second a client may send.
    pub client_rate: f64,
    /// Operator token hosts must prove to create rooms (None = anyone may).
    pub token: Option<String>,
    /// Rooms one IP address may hold at once.
    pub rooms_per_ip: usize,
    /// Admitted members that haven't exchanged data yet, per room.
    pub max_pending: usize,
    /// The same over the whole server.
    pub max_pending_total: usize,
    /// Seconds a pending member may stay pending.
    pub pending_timeout: f64,
    /// Seconds a room stays open without a host, waiting for a successor's claim.
    pub orphan_grace: f64,
    /// Also run as a server directory (NET-6); `None` = rooms only.
    pub directory: Option<crate::directory::DirConfig>,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            max_rooms: 4096,
            max_members: 64,
            idle_timeout: 30.0,
            host_rate: 4.0e6,
            client_rate: 1.0e6,
            token: None,
            rooms_per_ip: 8,
            max_pending: 16,
            max_pending_total: 1024,
            pending_timeout: 15.0,
            orphan_grace: 30.0,
            directory: None,
        }
    }
}

/// Counters for logs and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct RelayStats {
    pub rooms: usize,
    pub members: usize,
    pub packets_in: u64,
    pub packets_relayed: u64,
    pub bytes_relayed: u64,
    pub dropped: u64,
    /// Registrations refused (bad proof, unknown room, limits).
    pub refused: u64,
    /// Datagrams ignored because their IP is backing off.
    pub throttled: u64,
    /// Host slots taken over by a successor.
    pub claims: u64,
    /// Bytes sent to senders that were not members yet (cookies, errors).
    pub unauthenticated_out: u64,
    /// Directory mode: servers listed now.
    pub dir_servers: usize,
}

/// A member heard from within this many seconds (clients keepalive every 5 s)
/// still outranks a worse-ranked claimant.
const CLAIM_LIVE_WINDOW: f64 = 12.0;
/// The host slot can be claimed after this much silence (hosts keepalive
/// every 0.25 s). Sessions time a host out after at least 1 s, so an honest
/// successor never waits on this; it only stops a member seizing a live room.
const HOST_LIVE_WINDOW: f64 = 1.0;

struct Member {
    addr: SocketAddr,
    nonce: u64,
    token: u64,
    locals: Vec<SocketAddr>,
    last_seen: f64,
    joined: f64,
    tokens: f64,
    last_refill: f64,
    /// Exchanged data at least once (no longer pending).
    active: bool,
    claim_nonce: u64,
}

struct Room {
    members: BTreeMap<u32, Member>,
    next_slot: u32,
    owner: [u8; 32],
    join_key: [u8; 32],
    kdf: Vec<u8>,
    creator: IpAddr,
    /// Since when the room has no host.
    orphaned: Option<f64>,
    /// The host's ranked successors (SUCCESSORS); empty until it sends them.
    successors: Vec<u32>,
}

impl Room {
    fn pending(&self) -> usize {
        self.members.iter().filter(|(s, m)| **s != HOST_SLOT && !m.active).count()
    }
}

#[derive(Clone, Copy, Default)]
struct Throttle {
    failures: u32,
    blocked_until: f64,
    last: f64,
}

/// A relay bound to one UDP socket. Drive it with [`Relay::poll`] (tests,
/// embedding) or [`Relay::run`] (the server binary).
pub struct Relay {
    socket: UdpSocket,
    cfg: RelayConfig,
    rooms: HashMap<String, Room>,
    by_addr: HashMap<SocketAddr, (String, u32)>,
    throttle: HashMap<IpAddr, Throttle>,
    secret: [u8; 32],
    /// X25519 secret hosts seal their room's join key to (public half in
    /// every COOKIE).
    seal_secret: [u8; 32],
    stats: RelayStats,
    start: Instant,
    last_gc: f64,
    dir: Option<crate::directory::Directory>,
}

impl Relay {
    pub fn bind(addr: impl ToSocketAddrs, cfg: RelayConfig) -> io::Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).map_err(|e| io::Error::other(format!("no OS randomness: {e}")))?;
        let mut seal_secret = [0u8; 32];
        getrandom::fill(&mut seal_secret).map_err(|e| io::Error::other(format!("no OS randomness: {e}")))?;
        let dir = cfg.directory.clone().map(crate::directory::Directory::new);
        Ok(Self {
            dir,
            seal_secret,
            socket,
            cfg,
            rooms: HashMap::new(),
            by_addr: HashMap::new(),
            throttle: HashMap::new(),
            secret,
            stats: RelayStats::default(),
            start: Instant::now(),
            last_gc: 0.0,
        })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    pub fn stats(&self) -> RelayStats {
        RelayStats { rooms: self.rooms.len(), members: self.by_addr.len(), dir_servers: self.dir.as_ref().map_or(0, |d| d.len()), ..self.stats }
    }

    /// The server directory, when the relay runs one.
    pub fn directory(&self) -> Option<&crate::directory::Directory> {
        self.dir.as_ref()
    }

    /// Slot and member count of a room (tests, logs).
    pub fn room_info(&self, code: &str) -> Option<(bool, usize)> {
        let r = self.rooms.get(&proto::normalize_code(code))?;
        Some((r.members.contains_key(&HOST_SLOT), r.members.len()))
    }

    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Handle every datagram waiting on the socket. Returns how many.
    pub fn poll(&mut self) -> usize {
        let mut buf = [0u8; 2048];
        let mut n = 0;
        loop {
            match self.socket.recv_from(&mut buf) {
                Ok((len, src)) => {
                    let now = self.now();
                    self.handle(&buf[..len], src, now);
                    n += 1;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::ConnectionReset => continue,
                Err(_) => break,
            }
        }
        self.gc(self.now());
        n
    }

    /// Serve until `stop` is set (checked a few times a second).
    pub fn run(&mut self, stop: &AtomicBool) -> io::Result<()> {
        self.socket.set_nonblocking(false)?;
        self.socket.set_read_timeout(Some(Duration::from_millis(250)))?;
        let mut buf = [0u8; 2048];
        while !stop.load(Ordering::Relaxed) {
            match self.socket.recv_from(&mut buf) {
                Ok((len, src)) => {
                    let now = self.now();
                    self.handle(&buf[..len], src, now);
                }
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::ConnectionReset) => {}
                Err(e) => return Err(e),
            }
            self.gc(self.now());
        }
        Ok(())
    }

    /// To a member (already validated): any size.
    fn send(&mut self, p: &Packet, to: SocketAddr) {
        let _ = self.socket.send_to(&p.encode(), to);
    }

    /// To a sender that isn't validated yet: never larger than its request.
    fn send_bounded(&mut self, bytes: &[u8], to: SocketAddr, request_len: usize) {
        if bytes.len() > request_len {
            self.stats.dropped += 1;
            return;
        }
        self.stats.unauthenticated_out += bytes.len() as u64;
        let _ = self.socket.send_to(bytes, to);
    }

    fn refuse(&mut self, to: SocketAddr, reason: &str, request_len: usize, failure: bool, now: f64) {
        self.stats.refused += 1;
        if failure {
            self.fail(to.ip(), now);
        }
        if let Some(b) = proto::error_within(reason, request_len) {
            self.send_bounded(&b, to, request_len);
        }
    }

    fn fail(&mut self, ip: IpAddr, now: f64) {
        if self.throttle.len() >= MAX_THROTTLED
            && !self.throttle.contains_key(&ip)
            && let Some(oldest) = self.throttle.iter().min_by(|a, b| a.1.last.total_cmp(&b.1.last)).map(|(ip, _)| *ip)
        {
            self.throttle.remove(&oldest);
        }
        let t = self.throttle.entry(ip).or_default();
        t.failures += 1;
        t.last = now;
        if t.failures > FREE_FAILURES {
            let backoff = (2f64).powi((t.failures - FREE_FAILURES - 1).min(16) as i32).min(MAX_BACKOFF);
            t.blocked_until = now + backoff;
        }
    }

    fn cookie(&self, src: SocketAddr, code: &str, now: f64) -> [u8; COOKIE_LEN] {
        proto::cookie_for(&self.secret, src, code, (now / COOKIE_PERIOD) as u64)
    }

    fn cookie_ok(&self, src: SocketAddr, code: &str, cookie: &[u8; COOKIE_LEN], now: f64) -> bool {
        let bucket = (now / COOKIE_PERIOD) as u64;
        [bucket, bucket.saturating_sub(1)]
            .iter()
            .any(|b| proto::tags_equal(&proto::cookie_for(&self.secret, src, code, *b), cookie))
    }

    fn handle(&mut self, data: &[u8], src: SocketAddr, now: f64) {
        self.stats.packets_in += 1;
        let member = self.by_addr.contains_key(&src);
        if !member && self.throttle.get(&src.ip()).is_some_and(|t| now < t.blocked_until) {
            self.stats.throttled += 1;
            return;
        }
        if data.first() == Some(&crate::directory::R_DIRECTORY) {
            self.handle_directory(data, src, now);
            return;
        }
        let Some(p) = Packet::decode(data) else {
            self.stats.dropped += 1;
            return;
        };
        match p {
            Packet::Register { protocol, role, code, nonce, locals, cookie, auth } => {
                if protocol != RELAY_PROTOCOL {
                    let full = format!("relay protocol mismatch (relay {RELAY_PROTOCOL}, you {protocol})");
                    let why = if full.len() + 2 <= data.len() { full.as_str() } else { "relay protocol mismatch" };
                    self.refuse(src, why, data.len(), false, now);
                    return;
                }
                if data.len() < MIN_REGISTER {
                    self.stats.dropped += 1;
                    return;
                }
                let code_n = proto::normalize_code(&code);
                if role == ROLE_CLIENT && !self.rooms.contains_key(&code_n) {
                    self.refuse(src, "no such room", data.len(), true, now);
                    return;
                }
                if !self.cookie_ok(src, &code_n, &cookie, now) {
                    let kdf = if role == ROLE_CLIENT { self.rooms.get(&code_n).map(|r| r.kdf.clone()).unwrap_or_default() } else { Vec::new() };
                    let reply = Packet::Cookie { cookie: self.cookie(src, &code_n, now), kdf, relay_key: proto::relay_public(&self.seal_secret) }.encode();
                    self.send_bounded(&reply, src, data.len());
                    return;
                }
                match (role, auth) {
                    (ROLE_HOST, RegisterAuth::Host { owner, sealed_join_key, kdf, token_proof }) => {
                        if let Some(token) = self.cfg.token.clone()
                            && !proto::tags_equal(&proto::token_proof(&token, &cookie, &code_n, nonce), &token_proof)
                        {
                            self.refuse(src, "relay token required", data.len(), true, now);
                            return;
                        }
                        let Some(join_key) = proto::open_join_key(&self.seal_secret, &cookie, &sealed_join_key) else {
                            self.refuse(src, "join key not sealed to this relay", data.len(), true, now);
                            return;
                        };
                        self.register_host(src, &code_n, nonce, locals, owner, join_key, kdf, now, data.len());
                    }
                    (ROLE_CLIENT, RegisterAuth::Client { proof }) => self.register_client(src, &code_n, nonce, locals, &cookie, &proof, now, data.len()),
                    (ROLE_CLIENT, RegisterAuth::None) => {
                        let kdf = self.rooms.get(&code_n).map(|r| r.kdf.clone()).unwrap_or_default();
                        let reply = Packet::Cookie { cookie: self.cookie(src, &code_n, now), kdf, relay_key: proto::relay_public(&self.seal_secret) }.encode();
                        self.send_bounded(&reply, src, data.len());
                    }
                    _ => {
                        self.stats.dropped += 1;
                    }
                }
            }
            Packet::Data { slot, payload } => self.forward(src, slot, payload, now),
            Packet::Claim { nonce, claim_nonce, sig } => self.claim(src, nonce, claim_nonce, &sig, now),
            Packet::Successors { nonce, slots } => self.set_successors(src, nonce, slots, now),
            Packet::Keepalive { nonce } => {
                if let Some(m) = self.member_mut(src)
                    && m.nonce == nonce
                {
                    m.last_seen = now;
                }
            }
            Packet::Leave { nonce } => {
                if self.member_mut(src).is_some_and(|m| m.nonce == nonce) {
                    self.remove_member(src, now);
                }
            }
            _ => self.stats.dropped += 1,
        }
    }

    /// A directory packet: ANNOUNCE (signed, from a server) or QUERY (a browser).
    fn handle_directory(&mut self, data: &[u8], src: SocketAddr, now: f64) {
        let unix_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
        let Some(mut dir) = self.dir.take() else {
            self.stats.dropped += 1;
            return;
        };
        let reply = match data.get(1) {
            Some(1) => {
                let refused = dir.stats().refused;
                let r = dir.announce(src, data, now, unix_ms);
                // A source that keeps sending bad announcements backs off like a bad joiner.
                if dir.stats().refused > refused {
                    self.fail(src.ip(), now);
                }
                r
            }
            Some(2) => {
                let bucket = (now / COOKIE_PERIOD) as u64;
                let secret = self.secret;
                let cookie_ok = |c: &[u8; COOKIE_LEN]| [bucket, bucket.saturating_sub(1)].iter().any(|b| proto::tags_equal(&proto::cookie_for(&secret, src, "directory", *b), c));
                let fresh = || proto::cookie_for(&secret, src, "directory", bucket);
                dir.query(src, data, now, cookie_ok, fresh)
            }
            _ => None,
        };
        self.dir = Some(dir);
        match reply {
            // The replies are all smaller than the request, except a LIST to a source that proved
            // its address with the cookie.
            Some(b) => {
                let _ = self.socket.send_to(&b, src);
            }
            None => self.stats.dropped += 1,
        }
    }

    fn member_mut(&mut self, addr: SocketAddr) -> Option<&mut Member> {
        let (code, slot) = self.by_addr.get(&addr)?;
        self.rooms.get_mut(code)?.members.get_mut(slot)
    }

    /// Drop whatever membership `addr` had. A host leaving a room with
    /// members orphans it (a successor may claim it); an empty room closes.
    fn remove_member(&mut self, addr: SocketAddr, now: f64) {
        let Some((code, slot)) = self.by_addr.remove(&addr) else {
            return;
        };
        let Some(room) = self.rooms.get_mut(&code) else {
            return;
        };
        room.members.remove(&slot);
        if slot == HOST_SLOT {
            room.orphaned = Some(now);
        }
        if room.members.is_empty() {
            self.rooms.remove(&code);
        }
    }

    fn close_room(&mut self, code: &str, reason: &str) {
        if let Some(room) = self.rooms.remove(code) {
            for m in room.members.values() {
                self.by_addr.remove(&m.addr);
                let _ = self.socket.send_to(&Packet::Error { reason: reason.into() }.encode(), m.addr);
            }
        }
    }

    fn new_member(&self, addr: SocketAddr, nonce: u64, locals: Vec<SocketAddr>, now: f64, burst: f64) -> Member {
        Member {
            addr,
            nonce,
            token: proto::random_u64(),
            locals,
            last_seen: now,
            joined: now,
            tokens: burst,
            last_refill: now,
            active: false,
            claim_nonce: proto::random_u64(),
        }
    }

    fn registered(&mut self, src: SocketAddr, code: &str, slot: u32) {
        let Some(m) = self.rooms.get(code).and_then(|r| r.members.get(&slot)) else {
            return;
        };
        let p = Packet::Registered { nonce: m.nonce, slot, token: m.token, code: code.to_owned(), public: m.addr, claim_nonce: m.claim_nonce };
        self.send(&p, src);
    }

    /// Introduce `slot` and the room's host to each other.
    fn introduce(&mut self, code: &str, slot: u32) {
        let Some(room) = self.rooms.get(code) else {
            return;
        };
        let (Some(me), Some(host)) = (room.members.get(&slot), room.members.get(&HOST_SLOT)) else {
            return;
        };
        let to_host = Packet::Peer { slot, token: me.token, public: me.addr, locals: me.locals.clone() };
        let to_client = Packet::Peer { slot: HOST_SLOT, token: host.token, public: host.addr, locals: host.locals.clone() };
        let (me_addr, host_addr) = (me.addr, host.addr);
        self.send(&to_client, me_addr);
        self.send(&to_host, host_addr);
    }

    #[allow(clippy::too_many_arguments)]
    fn register_host(
        &mut self,
        src: SocketAddr,
        code: &str,
        nonce: u64,
        locals: Vec<SocketAddr>,
        owner: [u8; 32],
        join_key: [u8; 32],
        kdf: Vec<u8>,
        now: f64,
        request_len: usize,
    ) {
        // Re-sent REGISTER from a host we already know: answer again.
        if let Some((known, slot)) = self.by_addr.get(&src).cloned()
            && slot == HOST_SLOT
            && self.rooms.get(&known).and_then(|r| r.members.get(&HOST_SLOT)).is_some_and(|m| m.nonce == nonce)
        {
            if let Some(m) = self.member_mut(src) {
                m.last_seen = now;
            }
            self.registered(src, &known, HOST_SLOT);
            return;
        }
        self.remove_member(src, now);
        let mut code = code.to_owned();
        if code.is_empty() {
            code = loop {
                let c = proto::new_room_code();
                if !self.rooms.contains_key(&c) {
                    break c;
                }
            };
        } else if let Some(room) = self.rooms.get(&code) {
            // The owner (same owner key) may take its room back; nobody else.
            if room.owner != owner {
                self.refuse(src, "room code in use", request_len, true, now);
                return;
            }
            self.close_room(&code.clone(), "the room was reopened by its owner");
        }
        if self.rooms.len() >= self.cfg.max_rooms {
            self.refuse(src, "relay full", request_len, false, now);
            return;
        }
        let held = self.rooms.values().filter(|r| r.creator == src.ip()).count();
        if held >= self.cfg.rooms_per_ip {
            self.refuse(src, "too many rooms from this address", request_len, true, now);
            return;
        }
        let host = self.new_member(src, nonce, locals, now, self.cfg.host_rate);
        let mut room = Room { members: BTreeMap::new(), next_slot: 1, owner, join_key, kdf, creator: src.ip(), orphaned: None, successors: Vec::new() };
        room.members.insert(HOST_SLOT, host);
        self.rooms.insert(code.clone(), room);
        self.by_addr.insert(src, (code.clone(), HOST_SLOT));
        self.throttle.remove(&src.ip());
        self.registered(src, &code, HOST_SLOT);
    }

    #[allow(clippy::too_many_arguments)]
    fn register_client(
        &mut self,
        src: SocketAddr,
        code: &str,
        nonce: u64,
        locals: Vec<SocketAddr>,
        cookie: &[u8; COOKIE_LEN],
        proof: &[u8; proto::PROOF_LEN],
        now: f64,
        request_len: usize,
    ) {
        let Some(room) = self.rooms.get(code) else {
            self.refuse(src, "no such room", request_len, true, now);
            return;
        };
        if !proto::tags_equal(&proto::join_proof(&room.join_key, cookie, code, nonce), proof) {
            self.refuse(src, "wrong session secret", request_len, true, now);
            return;
        }
        let known = self.by_addr.get(&src).cloned();
        let slot = match known {
            Some((c, s)) if c == code && s != HOST_SLOT && self.rooms[code].members.get(&s).is_some_and(|m| m.nonce == nonce) => s,
            _ => {
                self.remove_member(src, now);
                let pending_total: usize = self.rooms.values().map(Room::pending).sum();
                let (max_members, max_pending, max_total) = (self.cfg.max_members, self.cfg.max_pending, self.cfg.max_pending_total);
                let fresh = self.new_member(src, nonce, locals.clone(), now, self.cfg.client_rate);
                let Some(room) = self.rooms.get_mut(code) else {
                    self.refuse(src, "no such room", request_len, false, now);
                    return;
                };
                // The same client behind a rebound NAT port: move its slot.
                if let Some((&s, m)) = room.members.iter_mut().find(|(s, m)| **s != HOST_SLOT && m.nonce == nonce) {
                    self.by_addr.remove(&m.addr);
                    m.addr = src;
                    self.by_addr.insert(src, (code.to_owned(), s));
                    s
                } else {
                    if room.members.len() >= max_members {
                        self.refuse(src, "room full", request_len, false, now);
                        return;
                    }
                    if room.pending() >= max_pending || pending_total >= max_total {
                        self.refuse(src, "room busy, try again", request_len, false, now);
                        return;
                    }
                    let s = room.next_slot;
                    room.next_slot += 1;
                    room.members.insert(s, fresh);
                    self.by_addr.insert(src, (code.to_owned(), s));
                    s
                }
            }
        };
        self.throttle.remove(&src.ip());
        if let Some(m) = self.member_mut(src) {
            m.last_seen = now;
        }
        self.registered(src, code, slot);
        self.introduce(code, slot);
    }

    /// A member takes the host slot with a room-owner signature.
    fn claim(&mut self, src: SocketAddr, nonce: u64, claim_nonce: u64, sig: &[u8; 64], now: f64) {
        let Some((code, slot)) = self.by_addr.get(&src).cloned() else {
            self.stats.dropped += 1;
            return;
        };
        let Some(room) = self.rooms.get(&code) else {
            return;
        };
        if slot == HOST_SLOT {
            // Already ours (a resent claim): confirm.
            if room.members.get(&HOST_SLOT).is_some_and(|m| m.nonce == nonce) {
                self.registered(src, &code, HOST_SLOT);
            }
            return;
        }
        let Some(me) = room.members.get(&slot) else {
            return;
        };
        if me.nonce != nonce || me.claim_nonce != claim_nonce || !proto::verify_claim(&room.owner, &code, slot, nonce, claim_nonce, sig) {
            self.stats.refused += 1;
            self.fail(src.ip(), now);
            return;
        }
        // Every member holds the owner key, so the signature alone can't
        // stop one from seizing the room: the host slot must be silent and
        // no better-ranked successor still live. Not a failure: the claimant
        // retries and gets in once its turn comes.
        let live = |m: &Member| now - m.last_seen < CLAIM_LIVE_WINDOW;
        let rank = |s: u32| room.successors.iter().position(|x| *x == s).unwrap_or(usize::MAX);
        let mine = rank(slot);
        let host_live = room.members.get(&HOST_SLOT).is_some_and(|m| now - m.last_seen < HOST_LIVE_WINDOW);
        let better_live = room.members.iter().any(|(s, m)| *s != HOST_SLOT && *s != slot && rank(*s) < mine && live(m));
        if host_live || better_live {
            self.stats.dropped += 1;
            return;
        }
        let host_rate = self.cfg.host_rate;
        let room = self.rooms.get_mut(&code).expect("checked above");
        let previous = room.members.remove(&HOST_SLOT);
        let mut me = room.members.remove(&slot).expect("checked above");
        me.claim_nonce = proto::random_u64();
        me.token = proto::random_u64();
        me.tokens = host_rate;
        me.last_seen = now;
        me.active = true;
        room.members.insert(HOST_SLOT, me);
        room.orphaned = None;
        room.successors.clear();
        self.by_addr.insert(src, (code.clone(), HOST_SLOT));
        self.stats.claims += 1;
        if let Some(old) = previous {
            self.by_addr.remove(&old.addr);
            let _ = self.socket.send_to(&Packet::Error { reason: "room handed to a successor".into() }.encode(), old.addr);
        }
        self.registered(src, &code, HOST_SLOT);
        let slots: Vec<u32> = self.rooms[&code].members.keys().copied().filter(|s| *s != HOST_SLOT).collect();
        for s in slots {
            self.introduce(&code, s);
        }
    }

    /// Host: record the room's ranked successors.
    fn set_successors(&mut self, src: SocketAddr, nonce: u64, slots: Vec<u32>, now: f64) {
        let Some((code, slot)) = self.by_addr.get(&src).cloned() else {
            self.stats.dropped += 1;
            return;
        };
        let Some(room) = self.rooms.get_mut(&code) else { return };
        match room.members.get_mut(&HOST_SLOT) {
            Some(host) if slot == HOST_SLOT && host.nonce == nonce => {
                host.last_seen = now;
                room.successors = slots;
            }
            _ => self.stats.dropped += 1,
        }
    }

    fn forward(&mut self, src: SocketAddr, dest: u32, payload: &[u8], now: f64) {
        let Some((code, from)) = self.by_addr.get(&src).cloned() else {
            self.stats.dropped += 1;
            return;
        };
        // Star: the host reaches everyone, a client only the host.
        if payload.len() > MAX_PAYLOAD || (from != HOST_SLOT && dest != HOST_SLOT) || from == dest {
            self.stats.dropped += 1;
            return;
        }
        let (host_rate, client_rate) = (self.cfg.host_rate, self.cfg.client_rate);
        let Some(room) = self.rooms.get_mut(&code) else {
            return;
        };
        let Some(to) = room.members.get_mut(&dest).map(|m| {
            m.active = true;
            m.addr
        }) else {
            self.stats.dropped += 1;
            return;
        };
        let Some(me) = room.members.get_mut(&from) else {
            return;
        };
        me.last_seen = now;
        me.active = true;
        let rate = if from == HOST_SLOT { host_rate } else { client_rate };
        me.tokens = (me.tokens + (now - me.last_refill) * rate).min(rate);
        me.last_refill = now;
        if me.tokens < payload.len() as f64 {
            self.stats.dropped += 1;
            return;
        }
        me.tokens -= payload.len() as f64;
        self.stats.packets_relayed += 1;
        self.stats.bytes_relayed += payload.len() as u64;
        let _ = self.socket.send_to(&proto::data_packet(from, payload), to);
    }

    fn gc(&mut self, now: f64) {
        if now - self.last_gc < 1.0 {
            return;
        }
        self.last_gc = now;
        if let Some(d) = self.dir.as_mut() {
            d.gc(now);
        }
        let (timeout, pending_timeout, grace) = (self.cfg.idle_timeout, self.cfg.pending_timeout, self.cfg.orphan_grace);
        let stale: Vec<SocketAddr> = self
            .rooms
            .values()
            .flat_map(|r| r.members.values())
            .filter(|m| now - m.last_seen > timeout || (!m.active && now - m.joined > pending_timeout))
            .map(|m| m.addr)
            .collect();
        for a in stale {
            self.remove_member(a, now);
        }
        let expired: Vec<String> =
            self.rooms.iter().filter(|(_, r)| r.orphaned.is_some_and(|t| now - t > grace)).map(|(c, _)| c.clone()).collect();
        for code in expired {
            self.close_room(&code, "room closed (no host)");
        }
        self.throttle.retain(|_, t| now < t.blocked_until || now - t.last < MAX_BACKOFF * 2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer() -> UdpSocket {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        s
    }

    fn recv(relay: &mut Relay, s: &UdpSocket) -> Vec<u8> {
        relay.poll();
        let mut buf = [0u8; 2048];
        let (n, _) = s.recv_from(&mut buf).expect("reply");
        buf[..n].to_vec()
    }

    fn try_recv(relay: &mut Relay, s: &UdpSocket) -> Option<Vec<u8>> {
        relay.poll();
        let mut buf = [0u8; 2048];
        s.set_read_timeout(Some(Duration::from_millis(60))).unwrap();
        let r = s.recv_from(&mut buf).ok().map(|(n, _)| buf[..n].to_vec());
        s.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        r
    }

    const JOIN_KEY: [u8; 32] = [9; 32];
    const OWNER: [u8; 32] = [5; 32];

    /// The relay's sealing key, as its COOKIEs carry it.
    fn relay_key(relay: &Relay) -> [u8; 32] {
        proto::relay_public(&relay.seal_secret)
    }

    fn register_packet(relay_key: &[u8; 32], role: u8, code: &str, nonce: u64, cookie: [u8; 16], key: &[u8; 32], token: Option<&str>) -> Vec<u8> {
        let auth = if role == ROLE_HOST {
            RegisterAuth::Host {
                owner: proto::owner_public(&OWNER),
                sealed_join_key: proto::seal_join_key(relay_key, &cookie, &JOIN_KEY),
                kdf: vec![1, 2, 3],
                token_proof: token.map(|t| proto::token_proof(t, &cookie, code, nonce)).unwrap_or_default(),
            }
        } else {
            RegisterAuth::Client { proof: proto::join_proof(key, &cookie, code, nonce) }
        };
        Packet::Register { protocol: RELAY_PROTOCOL, role, code: code.into(), nonce, locals: vec![], cookie, auth }.encode()
    }

    /// Cookie round trip, then the real REGISTER. Returns the reply.
    fn register_with(relay: &mut Relay, s: &UdpSocket, role: u8, code: &str, nonce: u64, key: &[u8; 32], token: Option<&str>) -> Vec<u8> {
        let ra = relay.local_addr().unwrap();
        let first = register_packet(&relay_key(relay), role, code, nonce, [0; 16], key, token);
        s.send_to(&first, ra).unwrap();
        let reply = recv(relay, s);
        assert!(reply.len() <= first.len(), "amplification-safe cookie");
        let Some(Packet::Cookie { cookie, kdf, relay_key: offered }) = Packet::decode(&reply) else {
            panic!("expected COOKIE, got {:?}", Packet::decode(&reply));
        };
        assert_eq!(offered, relay_key(relay), "every COOKIE carries the relay's sealing key");
        if role == ROLE_CLIENT && relay.room_info(code).is_some() {
            assert_eq!(kdf, vec![1, 2, 3], "joiners learn the room's stretching parameters");
        }
        s.send_to(&register_packet(&relay_key(relay), role, code, nonce, cookie, key, token), ra).unwrap();
        recv(relay, s)
    }

    fn register(relay: &mut Relay, s: &UdpSocket, role: u8, code: &str, nonce: u64) -> (u32, String, u64) {
        match Packet::decode(&register_with(relay, s, role, code, nonce, &JOIN_KEY, None)) {
            Some(Packet::Registered { slot, code, public, claim_nonce, .. }) => {
                assert_eq!(public, s.local_addr().unwrap(), "the relay reports the public endpoint");
                (slot, code, claim_nonce)
            }
            other => panic!("expected REGISTERED, got {other:?}"),
        }
    }

    #[test]
    fn rooms_introductions_and_star_forwarding() {
        let mut relay = Relay::bind("127.0.0.1:0", RelayConfig::default()).unwrap();
        let ra = relay.local_addr().unwrap();
        let (host, a, b) = (peer(), peer(), peer());
        let (slot, code, _) = register(&mut relay, &host, ROLE_HOST, "", 1);
        assert_eq!((slot, code.len()), (HOST_SLOT, 6));

        // A client joins by code (any case): both sides are introduced.
        let (sa, _, _) = register(&mut relay, &a, ROLE_CLIENT, &code.to_lowercase(), 2);
        assert_eq!(sa, 1);
        assert!(matches!(Packet::decode(&recv(&mut relay, &a)), Some(Packet::Peer { slot: HOST_SLOT, .. })));
        match Packet::decode(&recv(&mut relay, &host)) {
            Some(Packet::Peer { slot: 1, public, .. }) => assert_eq!(public, a.local_addr().unwrap()),
            other => panic!("host introduced to the client, got {other:?}"),
        }
        let (sb, _, _) = register(&mut relay, &b, ROLE_CLIENT, &code, 3);
        assert_eq!(sb, 2);
        let _ = recv(&mut relay, &b);
        let _ = recv(&mut relay, &host);

        // Client -> host and host -> client are forwarded with the source slot.
        a.send_to(&proto::data_packet(HOST_SLOT, b"hi host"), ra).unwrap();
        assert_eq!(Packet::decode(&recv(&mut relay, &host)), Some(Packet::Data { slot: 1, payload: b"hi host" }));
        host.send_to(&proto::data_packet(2, b"hi b"), ra).unwrap();
        assert_eq!(Packet::decode(&recv(&mut relay, &b)), Some(Packet::Data { slot: HOST_SLOT, payload: b"hi b" }));

        // Client -> client is refused (star), as are strangers and unknown rooms.
        a.send_to(&proto::data_packet(2, b"sneaky"), ra).unwrap();
        let stranger = peer();
        stranger.send_to(&proto::data_packet(HOST_SLOT, b"spam"), ra).unwrap();
        relay.poll();
        assert_eq!(relay.stats().dropped, 2);
        let first = register_packet(&relay_key(&relay), ROLE_CLIENT, "ZZZZZZ", 9, [0; 16], &JOIN_KEY, None);
        stranger.send_to(&first, ra).unwrap();
        let reply = recv(&mut relay, &stranger);
        assert!(reply.len() <= first.len());
        assert!(matches!(Packet::decode(&reply), Some(Packet::Error { reason }) if reason == "no such room"));

        // A second host (another owner key) can't take a live room's code.
        let other = peer();
        let rk = relay_key(&relay);
        let auth = |cookie: [u8; 16]| Packet::Register {
            protocol: RELAY_PROTOCOL,
            role: ROLE_HOST,
            code: code.clone(),
            nonce: 10,
            locals: vec![],
            cookie,
            auth: RegisterAuth::Host { owner: proto::owner_public(&[6; 32]), sealed_join_key: proto::seal_join_key(&rk, &cookie, &JOIN_KEY), kdf: vec![], token_proof: [0; 16] },
        };
        other.send_to(&auth([0; 16]).encode(), ra).unwrap();
        let Some(Packet::Cookie { cookie, .. }) = Packet::decode(&recv(&mut relay, &other)) else { panic!("cookie") };
        other.send_to(&auth(cookie).encode(), ra).unwrap();
        assert!(matches!(Packet::decode(&recv(&mut relay, &other)), Some(Packet::Error { reason }) if reason.contains("in use")));

        // The host leaving orphans the room (clients stay, waiting for a successor).
        let (host2, c) = (peer(), peer());
        let (_, code2, _) = register(&mut relay, &host2, ROLE_HOST, "", 11);
        register(&mut relay, &c, ROLE_CLIENT, &code2, 12);
        host2.send_to(&Packet::Leave { nonce: 11 }.encode(), ra).unwrap();
        relay.poll();
        assert_eq!(relay.room_info(&code2), Some((false, 1)), "orphaned, the client kept");
    }

    #[test]
    fn strangers_without_the_secret_are_refused_and_throttled() {
        let mut relay = Relay::bind("127.0.0.1:0", RelayConfig::default()).unwrap();
        let ra = relay.local_addr().unwrap();
        let host = peer();
        let (_, code, _) = register(&mut relay, &host, ROLE_HOST, "", 1);
        let stranger = peer();
        let reply = register_with(&mut relay, &stranger, ROLE_CLIENT, &code, 2, &[1; 32], None);
        assert!(matches!(Packet::decode(&reply), Some(Packet::Error { reason }) if reason.contains("secret")));
        assert_eq!(relay.room_info(&code), Some((true, 1)), "not admitted");
        // Keep failing: after a few attempts the address is ignored entirely.
        for k in 0..FREE_FAILURES + 2 {
            let first = register_packet(&relay_key(&relay), ROLE_CLIENT, &code, 10 + k as u64, [0; 16], &[1; 32], None);
            stranger.send_to(&first, ra).unwrap();
            if let Some(Packet::Cookie { cookie, .. }) = try_recv(&mut relay, &stranger).as_deref().and_then(Packet::decode) {
                stranger.send_to(&register_packet(&relay_key(&relay), ROLE_CLIENT, &code, 10 + k as u64, cookie, &[1; 32], None), ra).unwrap();
                let _ = try_recv(&mut relay, &stranger);
            }
        }
        let before = relay.stats().throttled;
        stranger.send_to(&register_packet(&relay_key(&relay), ROLE_CLIENT, &code, 99, [0; 16], &JOIN_KEY, None), ra).unwrap();
        assert!(try_recv(&mut relay, &stranger).is_none(), "backing off: not even a cookie");
        assert!(relay.stats().throttled > before);
        // Short (unpadded) registers get nothing: no amplification.
        let tiny = [proto::R_REGISTER, 2, 0, ROLE_CLIENT];
        let fresh = peer();
        fresh.send_to(&tiny, ra).unwrap();
        assert!(try_recv(&mut relay, &fresh).is_none());
        // Every unauthenticated reply was at most as large as its request.
        assert!(relay.stats().unauthenticated_out > 0);
    }

    #[test]
    fn per_ip_room_limit_operator_token_and_pending_cap() {
        let cfg = RelayConfig { rooms_per_ip: 2, token: Some("op-token".into()), max_pending: 1, ..Default::default() };
        let mut relay = Relay::bind("127.0.0.1:0", cfg).unwrap();
        let h1 = peer();
        let r = register_with(&mut relay, &h1, ROLE_HOST, "", 1, &JOIN_KEY, None);
        assert!(matches!(Packet::decode(&r), Some(Packet::Error { reason }) if reason.contains("token")), "token required");
        let mut codes = Vec::new();
        for (k, s) in [peer(), peer(), peer()].iter().enumerate() {
            let r = register_with(&mut relay, s, ROLE_HOST, "", 10 + k as u64, &JOIN_KEY, Some("op-token"));
            match Packet::decode(&r) {
                Some(Packet::Registered { code, .. }) => codes.push(code),
                Some(Packet::Error { reason }) => assert!(reason.contains("too many rooms"), "{reason}"),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(codes.len(), 2, "two rooms per IP");
        // One pending joiner per room: the second must wait.
        let (a, b) = (peer(), peer());
        assert!(matches!(Packet::decode(&register_with(&mut relay, &a, ROLE_CLIENT, &codes[0], 20, &JOIN_KEY, None)), Some(Packet::Registered { .. })));
        let r = register_with(&mut relay, &b, ROLE_CLIENT, &codes[0], 21, &JOIN_KEY, None);
        assert!(matches!(Packet::decode(&r), Some(Packet::Error { reason }) if reason.contains("busy")));
    }

    #[test]
    fn a_member_with_the_owner_key_claims_the_host_slot() {
        let mut relay = Relay::bind("127.0.0.1:0", RelayConfig::default()).unwrap();
        let ra = relay.local_addr().unwrap();
        let (host, a, b) = (peer(), peer(), peer());
        let (_, code, _) = register(&mut relay, &host, ROLE_HOST, "", 1);
        let (sa, _, claim_nonce) = register(&mut relay, &a, ROLE_CLIENT, &code, 2);
        let _ = recv(&mut relay, &a);
        let _ = recv(&mut relay, &host);
        let (sb, _, _) = register(&mut relay, &b, ROLE_CLIENT, &code, 3);
        let _ = recv(&mut relay, &b);
        let _ = recv(&mut relay, &host);
        // A forged claim (wrong key) changes nothing.
        let bad = proto::sign_claim(&[1; 32], &code, sa, 2, claim_nonce);
        a.send_to(&Packet::Claim { nonce: 2, claim_nonce, sig: bad }.encode(), ra).unwrap();
        relay.poll();
        assert_eq!(relay.stats().claims, 0);
        // Every member holds the owner key, yet none may seize a live host's room.
        let sig = proto::sign_claim(&OWNER, &code, sa, 2, claim_nonce);
        a.send_to(&Packet::Claim { nonce: 2, claim_nonce, sig }.encode(), ra).unwrap();
        assert!(try_recv(&mut relay, &a).is_none());
        assert_eq!(relay.stats().claims, 0);
        // The host ranks b first; a member can't rewrite that order.
        host.send_to(&Packet::Successors { nonce: 1, slots: vec![sb, sa] }.encode(), ra).unwrap();
        a.send_to(&Packet::Successors { nonce: 2, slots: vec![sa] }.encode(), ra).unwrap();
        relay.poll();
        assert_eq!(relay.rooms[&code].successors, vec![sb, sa]);
        let age = |relay: &mut Relay, slot: u32| relay.rooms.get_mut(&code).unwrap().members.get_mut(&slot).unwrap().last_seen -= CLAIM_LIVE_WINDOW + 1.0;
        // The host crashes (no LEAVE): a waits while the better-ranked b is live...
        relay.rooms.get_mut(&code).unwrap().members.get_mut(&HOST_SLOT).unwrap().last_seen -= HOST_LIVE_WINDOW + 1.0;
        a.send_to(&Packet::Claim { nonce: 2, claim_nonce, sig }.encode(), ra).unwrap();
        assert!(try_recv(&mut relay, &a).is_none());
        assert_eq!(relay.stats().claims, 0);
        // ...and takes over once b has gone silent too.
        age(&mut relay, sb);
        a.send_to(&Packet::Claim { nonce: 2, claim_nonce, sig }.encode(), ra).unwrap();
        assert!(matches!(Packet::decode(&recv(&mut relay, &a)), Some(Packet::Registered { slot: HOST_SLOT, .. })));
        assert_eq!(relay.stats().claims, 1);
        assert!(matches!(Packet::decode(&recv(&mut relay, &host)), Some(Packet::Error { reason }) if reason.contains("successor")));
        // The other member is introduced to its new host, and data flows.
        assert!(matches!(Packet::decode(&recv(&mut relay, &b)), Some(Packet::Peer { slot: HOST_SLOT, .. })));
        b.send_to(&proto::data_packet(HOST_SLOT, b"to the new host"), ra).unwrap();
        let mut got = false;
        for _ in 0..4 {
            if let Some(Packet::Data { payload, .. }) = Packet::decode(&recv(&mut relay, &a)) {
                assert_eq!(payload, b"to the new host");
                got = true;
                break;
            }
        }
        assert!(got);
        // A replayed claim (old claim nonce) is refused.
        a.send_to(&Packet::Claim { nonce: 2, claim_nonce, sig }.encode(), ra).unwrap();
        relay.poll();
        assert_eq!(relay.stats().claims, 1);
    }

    #[test]
    fn rate_limit_drops_floods() {
        let cfg = RelayConfig { client_rate: 4000.0, ..Default::default() };
        let mut relay = Relay::bind("127.0.0.1:0", cfg).unwrap();
        let ra = relay.local_addr().unwrap();
        let (host, a) = (peer(), peer());
        let (_, code, _) = register(&mut relay, &host, ROLE_HOST, "", 1);
        register(&mut relay, &a, ROLE_CLIENT, &code, 2);
        for _ in 0..20 {
            a.send_to(&proto::data_packet(HOST_SLOT, &[0u8; 1000]), ra).unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
        relay.poll();
        let s = relay.stats();
        assert!(s.packets_relayed <= 5 && s.dropped >= 15, "{s:?}");
    }
}
