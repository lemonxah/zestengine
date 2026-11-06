//! Server directory (NET-6): dedicated servers announce themselves, players
//! browse the list. It is a mode of the relay (`RelayConfig::directory`), not a
//! new service: the same UDP socket, the same stateless cookies.
//!
//! Every packet is `R_DIRECTORY`, a sub-type byte and a body.
//!
//! | Sub-type | Direction | Body |
//! |---|---|---|
//! | ANNOUNCE | server -> relay | version, server public key, timestamp (ms), flags, [`ServerInfo`], Ed25519 signature |
//! | ACK / ERROR | relay -> server | echoed timestamp and TTL / a short reason |
//! | QUERY | player -> relay | cookie, page, [`Filter`] (padded to [`MIN_QUERY`]) |
//! | COOKIE | relay -> player | the cookie for this source address (smaller than the query) |
//! | LIST | relay -> player | page, page count, total, [`Listing`]s (at most [`MAX_LIST_BYTES`]) |
//!
//! **No spoofing**: an announcement is signed with the server's Ed25519 key,
//! which is also its listing key, so nobody can list or delist a server they
//! hold no key for. The listed address is the one the announcement came from
//! (never a field the sender wrote), its port the declared game port. **No
//! replay**: the signed timestamp must be within the relay's clock skew and
//! strictly newer than the last one accepted for that key, and the newest
//! timestamp per key outlives the listing, so a captured packet cannot
//! re-list a server that expired or was delisted. **No amplification**:
//! queries need a cookie bound to the sender's address, and everything the
//! relay sends an unauthenticated source is smaller than what it received.
//! **Limits**: token buckets per IP for announcements and queries (checked
//! before any signature is verified), a cap on servers per IP and in total, and
//! a TTL: a server that stops announcing drops off the list.

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::proto::{COOKIE_LEN, put_addr, read_addr};

/// First byte of every directory packet (relay tags are `0xF1..=0xFC`).
pub const R_DIRECTORY: u8 = 0xFD;
/// Directory wire version, inside ANNOUNCE.
pub const DIR_VERSION: u8 = 1;
/// Queries are padded to this size so no reply is larger than the request.
pub const MIN_QUERY: usize = 64;
/// Largest LIST packet.
pub const MAX_LIST_BYTES: usize = 1200;

const D_ANNOUNCE: u8 = 1;
const D_QUERY: u8 = 2;
const D_COOKIE: u8 = 3;
const D_LIST: u8 = 4;
const D_ACK: u8 = 5;
const D_ERROR: u8 = 6;

const DOMAIN: &[u8] = b"zestengine directory announce v1";
const FLAG_DELIST: u8 = 1;

/// What a server says about itself. Text fields are cut to these lengths.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServerInfo {
    /// Display name (48).
    pub name: String,
    /// Game / project identifier, so a browser lists only its own game (32).
    pub game: String,
    /// Game version (24).
    pub version: String,
    /// Current map or scene (48).
    pub map: String,
    /// Region tag such as `eu`, `na-east` (16).
    pub region: String,
    pub players: u16,
    pub max_players: u16,
    /// The server's game port; the host is the address the announcement came from.
    pub port: u16,
    /// Relay room code for servers that are only reachable through a relay (16).
    pub room: String,
    /// A join password is needed.
    pub password: bool,
    /// The game network protocol version.
    pub protocol: u16,
}

impl ServerInfo {
    fn clamp(&mut self) {
        clamp_str(&mut self.name, 48);
        clamp_str(&mut self.game, 32);
        clamp_str(&mut self.version, 24);
        clamp_str(&mut self.map, 48);
        clamp_str(&mut self.region, 16);
        clamp_str(&mut self.room, 16);
    }
}

fn clamp_str(s: &mut String, max: usize) {
    s.retain(|c| !c.is_control());
    if s.len() > max {
        let mut n = max;
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        s.truncate(n);
    }
}

/// A listing a browser gets back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    /// The server's public key (its identity).
    pub key: [u8; 32],
    /// Where it announced from, with its declared game port.
    pub addr: SocketAddr,
    /// Seconds since its last announcement.
    pub age_secs: u16,
    pub info: ServerInfo,
}

/// What a browser asks for. Empty text and `false` mean "any".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Filter {
    /// Region, case-insensitive exact match.
    pub region: String,
    /// Game identifier, case-insensitive exact match.
    pub game: String,
    pub version: String,
    /// Case-insensitive substring of the name.
    pub name_contains: String,
    pub hide_full: bool,
    pub hide_empty: bool,
    pub hide_password: bool,
}

impl Filter {
    pub fn matches(&self, i: &ServerInfo) -> bool {
        let eq = |want: &str, have: &str| want.is_empty() || want.eq_ignore_ascii_case(have);
        eq(&self.region, &i.region)
            && eq(&self.game, &i.game)
            && eq(&self.version, &i.version)
            && (self.name_contains.is_empty() || i.name.to_lowercase().contains(&self.name_contains.to_lowercase()))
            && !(self.hide_full && i.max_players > 0 && i.players >= i.max_players)
            && !(self.hide_empty && i.players == 0)
            && !(self.hide_password && i.password)
    }
}

// ------------------------------------------------------------------ wire

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|b| u16::from_le_bytes([b[0], b[1]]))
    }
    fn u64(&mut self) -> Option<u64> {
        self.take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
    }
    fn arr<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N).and_then(|b| b.try_into().ok())
    }
    fn str(&mut self, max: usize) -> Option<String> {
        let n = self.u8()? as usize;
        let mut s = String::from_utf8_lossy(self.take(n)?).into_owned();
        clamp_str(&mut s, max);
        Some(s)
    }
}

fn put_s(b: &mut Vec<u8>, s: &str, max: usize) {
    let mut s = s.to_owned();
    clamp_str(&mut s, max);
    b.push(s.len() as u8);
    b.extend_from_slice(s.as_bytes());
}

fn put_info(b: &mut Vec<u8>, i: &ServerInfo) {
    put_s(b, &i.name, 48);
    put_s(b, &i.game, 32);
    put_s(b, &i.version, 24);
    put_s(b, &i.map, 48);
    put_s(b, &i.region, 16);
    put_s(b, &i.room, 16);
    b.extend_from_slice(&i.players.to_le_bytes());
    b.extend_from_slice(&i.max_players.to_le_bytes());
    b.extend_from_slice(&i.port.to_le_bytes());
    b.extend_from_slice(&i.protocol.to_le_bytes());
    b.push(u8::from(i.password));
}

fn read_info(r: &mut Reader) -> Option<ServerInfo> {
    Some(ServerInfo {
        name: r.str(48)?,
        game: r.str(32)?,
        version: r.str(24)?,
        map: r.str(48)?,
        region: r.str(16)?,
        room: r.str(16)?,
        players: r.u16()?,
        max_players: r.u16()?,
        port: r.u16()?,
        protocol: r.u16()?,
        password: r.u8()? != 0,
    })
}

fn put_filter(b: &mut Vec<u8>, f: &Filter) {
    put_s(b, &f.region, 16);
    put_s(b, &f.game, 32);
    put_s(b, &f.version, 24);
    put_s(b, &f.name_contains, 48);
    b.push(u8::from(f.hide_full) | u8::from(f.hide_empty) << 1 | u8::from(f.hide_password) << 2);
}

fn read_filter(r: &mut Reader) -> Option<Filter> {
    let (region, game, version, name_contains) = (r.str(16)?, r.str(32)?, r.str(24)?, r.str(48)?);
    let flags = r.u8()?;
    Some(Filter { region, game, version, name_contains, hide_full: flags & 1 != 0, hide_empty: flags & 2 != 0, hide_password: flags & 4 != 0 })
}

/// The announcement packet. `sign` is given the bytes to sign (a domain tag,
/// then everything in the packet before the signature) and returns the
/// Ed25519 signature of `public`'s key. `ts_ms` is unix milliseconds and must
/// grow with every announcement of one key; `delist` removes the listing.
pub fn encode_announce(public: [u8; 32], ts_ms: u64, delist: bool, info: &ServerInfo, sign: impl FnOnce(&[u8]) -> [u8; 64]) -> Vec<u8> {
    let mut b = vec![R_DIRECTORY, D_ANNOUNCE, DIR_VERSION];
    b.extend_from_slice(&public);
    b.extend_from_slice(&ts_ms.to_le_bytes());
    b.push(if delist { FLAG_DELIST } else { 0 });
    put_info(&mut b, info);
    let mut msg = DOMAIN.to_vec();
    msg.extend_from_slice(&b[2..]);
    b.extend_from_slice(&sign(&msg));
    b
}

/// A browser's query for page `page` of the list. Without a valid `cookie`
/// the relay answers with one.
pub fn encode_query(cookie: [u8; COOKIE_LEN], page: u16, filter: &Filter) -> Vec<u8> {
    let mut b = vec![R_DIRECTORY, D_QUERY];
    b.extend_from_slice(&cookie);
    b.extend_from_slice(&page.to_le_bytes());
    put_filter(&mut b, filter);
    if b.len() < MIN_QUERY {
        b.resize(MIN_QUERY, 0);
    }
    b
}

/// What the relay sends back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// Announcement accepted: its timestamp echoed and the listing TTL in seconds.
    Ack { ts_ms: u64, ttl: u16 },
    Error(String),
    Cookie([u8; COOKIE_LEN]),
    List { page: u16, pages: u16, total: u16, entries: Vec<Listing> },
}

pub fn decode_reply(d: &[u8]) -> Option<Reply> {
    if d.first() != Some(&R_DIRECTORY) {
        return None;
    }
    let mut r = Reader { d, p: 2 };
    match *d.get(1)? {
        D_ACK => Some(Reply::Ack { ts_ms: r.u64()?, ttl: r.u16()? }),
        D_ERROR => Some(Reply::Error(r.str(64)?)),
        D_COOKIE => Some(Reply::Cookie(r.arr()?)),
        D_LIST => {
            let (page, pages, total, n) = (r.u16()?, r.u16()?, r.u16()?, r.u8()?);
            let mut entries = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let key = r.arr()?;
                let mut at = r.p;
                let addr = read_addr(d, &mut at)?;
                r.p = at;
                let age_secs = r.u16()?;
                entries.push(Listing { key, addr, age_secs, info: read_info(&mut r)? });
            }
            Some(Reply::List { page, pages, total, entries })
        }
        _ => None,
    }
}

fn encode_listing(b: &mut Vec<u8>, l: &Listing) {
    b.extend_from_slice(&l.key);
    put_addr(b, l.addr);
    b.extend_from_slice(&l.age_secs.to_le_bytes());
    put_info(b, &l.info);
}

fn ack(ts_ms: u64, ttl: u16) -> Vec<u8> {
    let mut b = vec![R_DIRECTORY, D_ACK];
    b.extend_from_slice(&ts_ms.to_le_bytes());
    b.extend_from_slice(&ttl.to_le_bytes());
    b
}

fn error(reason: &str) -> Vec<u8> {
    let mut b = vec![R_DIRECTORY, D_ERROR];
    put_s(&mut b, reason, 40);
    b
}

// ------------------------------------------------------------------ state

/// Directory limits.
#[derive(Clone, Debug)]
pub struct DirConfig {
    /// Seconds a listing lives without a fresh announcement (servers should announce about every third of it).
    pub ttl: f64,
    /// Listings at most.
    pub max_servers: usize,
    /// Listings from one IP address at most.
    pub per_ip: usize,
    /// How far an announcement's timestamp may be from the relay's clock (ms).
    pub max_skew_ms: u64,
    /// Only these server keys may list. Empty = any key.
    pub allow: Vec<[u8; 32]>,
    /// Announcements per IP: burst, then per second.
    pub announce_rate: (f64, f64),
    /// Queries per IP: burst, then per second.
    pub query_rate: (f64, f64),
}

impl Default for DirConfig {
    fn default() -> Self {
        Self { ttl: 90.0, max_servers: 4096, per_ip: 4, max_skew_ms: 60_000, allow: Vec::new(), announce_rate: (6.0, 0.5), query_rate: (8.0, 2.0) }
    }
}

/// Counters for logs and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct DirStats {
    pub servers: usize,
    pub announces: u64,
    pub delisted: u64,
    /// Bad signature, stale or replayed timestamp, unknown key, limits.
    pub refused: u64,
    pub queries: u64,
    /// Dropped silently by a per-IP rate limit.
    pub rate_limited: u64,
}

struct Entry {
    info: ServerInfo,
    addr: SocketAddr,
    last: f64,
}

#[derive(Clone, Copy)]
struct Bucket {
    tokens: f64,
    last: f64,
}

/// The server registry. Pure state (no socket): the relay feeds it datagrams.
pub struct Directory {
    cfg: DirConfig,
    servers: HashMap<[u8; 32], Entry>,
    /// Newest accepted timestamp per key (ms) and when, kept after a listing ends.
    newest: HashMap<[u8; 32], (u64, f64)>,
    announce_bucket: HashMap<IpAddr, Bucket>,
    query_bucket: HashMap<IpAddr, Bucket>,
    stats: DirStats,
    last_gc: f64,
}

const MAX_TRACKED: usize = 65_536;

fn take_token(map: &mut HashMap<IpAddr, Bucket>, ip: IpAddr, (burst, rate): (f64, f64), now: f64) -> bool {
    if map.len() >= MAX_TRACKED && !map.contains_key(&ip) {
        return false;
    }
    let b = map.entry(ip).or_insert(Bucket { tokens: burst, last: now });
    b.tokens = (b.tokens + (now - b.last) * rate).min(burst);
    b.last = now;
    if b.tokens < 1.0 {
        return false;
    }
    b.tokens -= 1.0;
    true
}

impl Directory {
    pub fn new(cfg: DirConfig) -> Self {
        Self {
            cfg,
            servers: HashMap::new(),
            newest: HashMap::new(),
            announce_bucket: HashMap::new(),
            query_bucket: HashMap::new(),
            stats: DirStats::default(),
            last_gc: 0.0,
        }
    }

    pub fn stats(&self) -> DirStats {
        DirStats { servers: self.servers.len(), ..self.stats }
    }

    /// The live listings (tests, logs).
    pub fn len(&self) -> usize {
        self.servers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Handle an ANNOUNCE from `src`. `now`: relay clock (seconds); `unix_ms`: wall clock.
    /// Returns the reply to send, or `None` when the source is rate limited (silence).
    pub fn announce(&mut self, src: SocketAddr, data: &[u8], now: f64, unix_ms: u64) -> Option<Vec<u8>> {
        if !take_token(&mut self.announce_bucket, src.ip(), self.cfg.announce_rate, now) {
            self.stats.rate_limited += 1;
            return None;
        }
        Some(match self.accept(src, data, now, unix_ms) {
            Ok(r) => r,
            Err(why) => {
                self.stats.refused += 1;
                error(why)
            }
        })
    }

    fn accept(&mut self, src: SocketAddr, data: &[u8], now: f64, unix_ms: u64) -> Result<Vec<u8>, &'static str> {
        let sig_at = data.len().checked_sub(64).ok_or("malformed")?;
        let mut r = Reader { d: &data[..sig_at], p: 2 };
        if r.u8().ok_or("malformed")? != DIR_VERSION {
            return Err("unsupported directory version");
        }
        let key: [u8; 32] = r.arr().ok_or("malformed")?;
        let ts_ms = r.u64().ok_or("malformed")?;
        let flags = r.u8().ok_or("malformed")?;
        let mut info = read_info(&mut r).ok_or("malformed")?;
        if r.p != sig_at {
            return Err("malformed");
        }
        if !self.cfg.allow.is_empty() && !self.cfg.allow.contains(&key) {
            return Err("key not allowed");
        }
        let sig: [u8; 64] = data[sig_at..].try_into().map_err(|_| "malformed")?;
        let mut msg = DOMAIN.to_vec();
        msg.extend_from_slice(&data[2..sig_at]);
        let ok = VerifyingKey::from_bytes(&key).is_ok_and(|k| k.verify(&msg, &Signature::from_bytes(&sig)).is_ok());
        if !ok {
            return Err("bad signature");
        }
        if ts_ms.abs_diff(unix_ms) > self.cfg.max_skew_ms {
            return Err("clock out of range");
        }
        if self.newest.get(&key).is_some_and(|(newest, _)| ts_ms <= *newest) {
            return Err("replayed");
        }
        if self.newest.len() >= MAX_TRACKED && !self.newest.contains_key(&key) {
            return Err("directory busy");
        }
        if flags & FLAG_DELIST != 0 {
            self.newest.insert(key, (ts_ms, now));
            if self.servers.remove(&key).is_some() {
                self.stats.delisted += 1;
            }
            return Ok(ack(ts_ms, 0));
        }
        info.clamp();
        if info.port == 0 && info.room.is_empty() {
            return Err("no port or room");
        }
        let known = self.servers.contains_key(&key);
        if !known {
            if self.servers.len() >= self.cfg.max_servers {
                return Err("directory full");
            }
            if self.servers.values().filter(|e| e.addr.ip() == src.ip()).count() >= self.cfg.per_ip {
                return Err("too many servers from this address");
            }
        }
        // Only an accepted announcement uses up its timestamp (a refusal for a limit may be retried).
        self.newest.insert(key, (ts_ms, now));
        let addr = SocketAddr::new(src.ip(), info.port);
        self.servers.insert(key, Entry { info, addr, last: now });
        self.stats.announces += 1;
        Ok(ack(ts_ms, self.cfg.ttl as u16))
    }

    /// Handle a QUERY from `src`. `cookie_ok` / `new_cookie` are the relay's
    /// stateless cookie for that source. `None` = silence (rate limited or malformed).
    pub fn query(&mut self, src: SocketAddr, data: &[u8], now: f64, cookie_ok: impl Fn(&[u8; COOKIE_LEN]) -> bool, new_cookie: impl Fn() -> [u8; COOKIE_LEN]) -> Option<Vec<u8>> {
        if data.len() < MIN_QUERY || !take_token(&mut self.query_bucket, src.ip(), self.cfg.query_rate, now) {
            self.stats.rate_limited += u64::from(data.len() >= MIN_QUERY);
            return None;
        }
        let mut r = Reader { d: data, p: 2 };
        let cookie: [u8; COOKIE_LEN] = r.arr()?;
        let page = r.u16()?;
        let filter = read_filter(&mut r)?;
        if !cookie_ok(&cookie) {
            let mut b = vec![R_DIRECTORY, D_COOKIE];
            b.extend_from_slice(&new_cookie());
            return Some(b);
        }
        self.stats.queries += 1;
        Some(self.list(&filter, page, now))
    }

    fn list(&self, filter: &Filter, page: u16, now: f64) -> Vec<u8> {
        let mut hits: Vec<Listing> = self
            .servers
            .iter()
            .filter(|(_, e)| now - e.last <= self.cfg.ttl && filter.matches(&e.info))
            .map(|(k, e)| Listing { key: *k, addr: e.addr, age_secs: (now - e.last).clamp(0.0, 65535.0) as u16, info: e.info.clone() })
            .collect();
        // Fullest first, then by name; the key breaks ties so pages are stable.
        hits.sort_by(|a, b| b.info.players.cmp(&a.info.players).then_with(|| a.info.name.cmp(&b.info.name)).then_with(|| a.key.cmp(&b.key)));
        let mut pages: Vec<Vec<u8>> = vec![Vec::new()];
        let mut counts: Vec<u8> = vec![0];
        for l in &hits {
            let mut enc = Vec::new();
            encode_listing(&mut enc, l);
            let last = pages.len() - 1;
            if pages[last].len() + enc.len() > MAX_LIST_BYTES - 9 || counts[last] == u8::MAX {
                pages.push(Vec::new());
                counts.push(0);
            }
            let last = pages.len() - 1;
            pages[last].extend_from_slice(&enc);
            counts[last] += 1;
        }
        let i = (page as usize).min(pages.len() - 1);
        let mut b = vec![R_DIRECTORY, D_LIST];
        b.extend_from_slice(&(i as u16).to_le_bytes());
        b.extend_from_slice(&(pages.len() as u16).to_le_bytes());
        b.extend_from_slice(&(hits.len().min(u16::MAX as usize) as u16).to_le_bytes());
        b.push(counts[i]);
        b.extend_from_slice(&pages[i]);
        b
    }

    /// Drop expired listings and stale bookkeeping. Cheap to call often.
    pub fn gc(&mut self, now: f64) {
        if now - self.last_gc < 1.0 {
            return;
        }
        self.last_gc = now;
        let ttl = self.cfg.ttl;
        self.servers.retain(|_, e| now - e.last <= ttl);
        // A replayed packet is only acceptable inside the skew window, so timestamps
        // are remembered for twice that long after their last use.
        let keep = (self.cfg.max_skew_ms as f64 / 1000.0) * 2.0 + ttl;
        self.newest.retain(|k, (_, at)| now - *at <= keep || self.servers.contains_key(k));
        self.announce_bucket.retain(|_, b| now - b.last < 120.0);
        self.query_bucket.retain(|_, b| now - b.last < 120.0);
    }
}

// ------------------------------------------------------------------ client side

/// Ask a directory relay for its list, following pages, with a blocking socket
/// (a game's server browser calls this off the main thread). Returns every page's listings.
pub fn query_blocking(relay: SocketAddr, filter: &Filter, timeout: Duration) -> io::Result<Vec<Listing>> {
    let socket = UdpSocket::bind(if relay.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" })?;
    socket.connect(relay)?;
    socket.set_read_timeout(Some(Duration::from_millis(250)))?;
    let deadline = Instant::now() + timeout;
    let mut cookie = [0u8; COOKIE_LEN];
    let mut page = 0u16;
    let mut out = Vec::new();
    let mut buf = [0u8; 2048];
    let mut last_send: Option<Instant> = None;
    loop {
        if Instant::now() > deadline {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "the directory did not answer"));
        }
        if last_send.is_none_or(|t| t.elapsed() > Duration::from_millis(500)) {
            socket.send(&encode_query(cookie, page, filter))?;
            last_send = Some(Instant::now());
        }
        let n = match socket.recv(&mut buf) {
            Ok(n) => n,
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::ConnectionRefused) => continue,
            Err(e) => return Err(e),
        };
        match decode_reply(&buf[..n]) {
            Some(Reply::Cookie(c)) => {
                cookie = c;
                last_send = None;
            }
            Some(Reply::List { page: p, pages, entries, .. }) if p == page => {
                out.extend(entries);
                if page + 1 >= pages {
                    return Ok(out);
                }
                page += 1;
                last_send = None;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn key(n: u8) -> SigningKey {
        SigningKey::from_bytes(&[n; 32])
    }

    fn info(name: &str, players: u16) -> ServerInfo {
        ServerInfo { name: name.into(), game: "arena".into(), version: "1.2".into(), map: "scenes/main.scene".into(), region: "eu".into(), players, max_players: 16, port: 9000, protocol: 7, ..Default::default() }
    }

    fn announce(k: &SigningKey, ts: u64, delist: bool, i: &ServerInfo) -> Vec<u8> {
        encode_announce(k.verifying_key().to_bytes(), ts, delist, i, |m| k.sign(m).to_bytes())
    }

    fn src(n: u8) -> SocketAddr {
        SocketAddr::from(([10, 0, 0, n], 40000))
    }

    const NOW_MS: u64 = 1_700_000_000_000;

    fn reply_of(d: &mut Directory, from: SocketAddr, pkt: &[u8], now: f64) -> Reply {
        decode_reply(&d.announce(from, pkt, now, NOW_MS).expect("a reply")).expect("decodes")
    }

    fn list_all(d: &mut Directory, f: &Filter, now: f64) -> Vec<Listing> {
        let mut out = Vec::new();
        let mut page = 0;
        loop {
            let from = src(200);
            let first = d.query(from, &encode_query([0; 16], page, f), now, |_| false, || [7; 16]).unwrap();
            assert!(first.len() <= MIN_QUERY, "the cookie reply is no larger than the query");
            let q = encode_query([7; 16], page, f);
            let Some(Reply::List { pages, entries, .. }) = d.query(from, &q, now, |c| *c == [7; 16], || [7; 16]).map(|b| decode_reply(&b).unwrap()) else { panic!("list") };
            out.extend(entries);
            page += 1;
            if page >= pages {
                return out;
            }
            // Space queries out so the per-IP bucket is never the limit here.
        }
    }

    fn open() -> Directory {
        Directory::new(DirConfig { query_rate: (1000.0, 1000.0), announce_rate: (1000.0, 1000.0), per_ip: 100, ..Default::default() })
    }

    #[test]
    fn announce_then_list_with_filters() {
        let mut d = open();
        for (n, name, players, region) in [(1u8, "Alpha", 3u16, "eu"), (2, "Bravo", 9, "na"), (3, "Charlie", 0, "eu")] {
            let mut i = info(name, players);
            i.region = region.into();
            i.password = n == 3;
            assert!(matches!(reply_of(&mut d, src(n), &announce(&key(n), NOW_MS, false, &i), 0.0), Reply::Ack { ts_ms: NOW_MS, ttl: 90 }));
        }
        let all = list_all(&mut d, &Filter::default(), 1.0);
        assert_eq!(all.iter().map(|l| l.info.name.as_str()).collect::<Vec<_>>(), ["Bravo", "Alpha", "Charlie"], "fullest first");
        assert_eq!(all[0].addr, SocketAddr::from(([10, 0, 0, 2], 9000)), "host from the source, port declared");
        assert_eq!(all[0].key, key(2).verifying_key().to_bytes());
        assert_eq!(all[0].age_secs, 1);
        fn f(d: &mut Directory, f: Filter) -> Vec<String> {
            list_all(d, &f, 1.0).into_iter().map(|l| l.info.name).collect()
        }
        assert_eq!(f(&mut d, Filter { region: "EU".into(), ..Default::default() }), ["Alpha", "Charlie"]);
        assert_eq!(f(&mut d, Filter { hide_empty: true, ..Default::default() }), ["Bravo", "Alpha"]);
        assert_eq!(f(&mut d, Filter { hide_password: true, ..Default::default() }), ["Bravo", "Alpha"]);
        assert_eq!(f(&mut d, Filter { name_contains: "ARL".into(), ..Default::default() }), ["Charlie"]);
        assert_eq!(f(&mut d, Filter { game: "other".into(), ..Default::default() }), Vec::<String>::new());
        assert_eq!(f(&mut d, Filter { version: "1.2".into(), ..Default::default() }).len(), 3);
        let mut full = info("Full", 16);
        full.region = "eu".into();
        reply_of(&mut d, src(4), &announce(&key(4), NOW_MS, false, &full), 0.0);
        assert!(!f(&mut d, Filter { hide_full: true, ..Default::default() }).contains(&"Full".to_string()));
    }

    #[test]
    fn forged_tampered_and_foreign_announcements_are_refused() {
        let mut d = open();
        let good = announce(&key(1), NOW_MS, false, &info("Real", 1));
        // Flip one bit anywhere in the signed region (or the signature): all refused.
        for at in [3, 20, 40, 45, 60, good.len() - 70, good.len() - 1] {
            let mut bad = good.clone();
            bad[at] ^= 1;
            assert!(matches!(reply_of(&mut d, src(1), &bad, 0.0), Reply::Error(_)), "byte {at}");
        }
        assert!(d.is_empty());
        // Someone else's key signing for the listed server's key.
        let thief = key(9);
        let forged = encode_announce(key(1).verifying_key().to_bytes(), NOW_MS, false, &info("Fake", 1), |m| thief.sign(m).to_bytes());
        assert_eq!(reply_of(&mut d, src(9), &forged, 0.0), Reply::Error("bad signature".into()));
        // Truncated and garbage packets.
        for cut in [0, 3, 40, good.len() - 65, good.len() - 1] {
            assert!(matches!(reply_of(&mut d, src(1), &good[..cut.max(2)], 0.0), Reply::Error(_)), "cut {cut}");
        }
        assert!(matches!(reply_of(&mut d, src(1), &good, 0.0), Reply::Ack { .. }));
        // The legitimate owner can update; the thief cannot delist.
        let del = encode_announce(key(1).verifying_key().to_bytes(), NOW_MS + 5, true, &info("Real", 1), |m| thief.sign(m).to_bytes());
        assert!(matches!(reply_of(&mut d, src(9), &del, 1.0), Reply::Error(_)));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn replays_and_stale_clocks_are_refused_even_after_the_listing_ended() {
        let mut d = open();
        let first = announce(&key(1), NOW_MS, false, &info("Real", 1));
        assert!(matches!(reply_of(&mut d, src(1), &first, 0.0), Reply::Ack { .. }));
        assert_eq!(reply_of(&mut d, src(1), &first, 1.0), Reply::Error("replayed".into()), "same packet twice");
        let older = announce(&key(1), NOW_MS - 1, false, &info("Real", 1));
        assert_eq!(reply_of(&mut d, src(1), &older, 1.0), Reply::Error("replayed".into()), "older than the newest accepted");
        let newer = announce(&key(1), NOW_MS + 1000, false, &info("Real", 2));
        assert!(matches!(reply_of(&mut d, src(1), &newer, 2.0), Reply::Ack { .. }));
        // Delist, then a captured copy of the previous announcement cannot bring it back.
        let del = announce(&key(1), NOW_MS + 2000, true, &info("Real", 2));
        assert!(matches!(reply_of(&mut d, src(1), &del, 3.0), Reply::Ack { ttl: 0, .. }));
        assert!(d.is_empty());
        assert_eq!(reply_of(&mut d, src(2), &newer, 4.0), Reply::Error("replayed".into()));
        // Expiry then a replay inside the skew window, from a spoofed source.
        let live = announce(&key(2), NOW_MS + 3000, false, &info("Other", 1));
        assert!(matches!(reply_of(&mut d, src(2), &live, 10.0), Reply::Ack { .. }));
        d.gc(10.0 + 91.0 + 1.0);
        assert!(d.is_empty(), "expired");
        assert_eq!(reply_of(&mut d, src(77), &live, 104.0), Reply::Error("replayed".into()));
        // Clock far from the relay's.
        for ts in [NOW_MS - 61_000, NOW_MS + 61_000] {
            assert_eq!(reply_of(&mut d, src(3), &announce(&key(3), ts, false, &info("Skew", 0)), 0.0), Reply::Error("clock out of range".into()));
        }
        assert!(matches!(reply_of(&mut d, src(3), &announce(&key(3), NOW_MS - 59_000, false, &info("Skew", 0)), 0.0), Reply::Ack { .. }));
    }

    #[test]
    fn listings_expire_and_refresh_with_the_ttl() {
        let mut d = Directory::new(DirConfig { ttl: 30.0, ..Default::default() });
        let mut ts = NOW_MS;
        let mut ann = |d: &mut Directory, now: f64| {
            ts += 1000;
            reply_of(d, src(1), &announce(&key(1), ts, false, &info("S", 1)), now)
        };
        ann(&mut d, 0.0);
        ann(&mut d, 20.0);
        d.gc(40.0);
        assert_eq!(d.len(), 1, "refreshed at 20 s, alive at 40 s");
        assert_eq!(list_all(&mut d, &Filter::default(), 40.0).len(), 1);
        assert!(list_all(&mut d, &Filter::default(), 51.0).is_empty(), "a query never returns an expired listing, even before gc");
        d.gc(52.0);
        assert!(d.is_empty());
    }

    #[test]
    fn caps_allow_list_and_rate_limits() {
        // Servers per IP and in total.
        let mut d = Directory::new(DirConfig { per_ip: 2, max_servers: 3, announce_rate: (1000.0, 1000.0), ..Default::default() });
        for n in 1..=2 {
            assert!(matches!(reply_of(&mut d, src(1), &announce(&key(n), NOW_MS, false, &info("S", 0)), 0.0), Reply::Ack { .. }));
        }
        assert_eq!(reply_of(&mut d, src(1), &announce(&key(3), NOW_MS, false, &info("S", 0)), 0.0), Reply::Error("too many servers from this address".into()));
        assert!(matches!(reply_of(&mut d, src(2), &announce(&key(3), NOW_MS, false, &info("S", 0)), 0.0), Reply::Ack { .. }));
        assert_eq!(reply_of(&mut d, src(3), &announce(&key(4), NOW_MS, false, &info("S", 0)), 0.0), Reply::Error("directory full".into()));
        // A listed server may keep refreshing when the directory is full.
        assert!(matches!(reply_of(&mut d, src(1), &announce(&key(1), NOW_MS + 1, false, &info("S", 5)), 1.0), Reply::Ack { .. }));
        // Allow list.
        let mut d = Directory::new(DirConfig { allow: vec![key(1).verifying_key().to_bytes()], ..Default::default() });
        assert!(matches!(reply_of(&mut d, src(1), &announce(&key(1), NOW_MS, false, &info("S", 0)), 0.0), Reply::Ack { .. }));
        assert_eq!(reply_of(&mut d, src(2), &announce(&key(2), NOW_MS, false, &info("S", 0)), 0.0), Reply::Error("key not allowed".into()));
        // Announce rate limit: silence once the burst is spent, then it refills.
        let mut d = Directory::new(DirConfig { announce_rate: (3.0, 1.0), per_ip: 100, ..Default::default() });
        let mut answered = 0;
        for i in 0..10u64 {
            if d.announce(src(1), &announce(&key(1), NOW_MS + i, false, &info("S", 0)), 0.0, NOW_MS).is_some() {
                answered += 1;
            }
        }
        assert_eq!(answered, 3);
        assert_eq!(d.stats().rate_limited, 7);
        assert!(d.announce(src(1), &announce(&key(1), NOW_MS + 99, false, &info("S", 0)), 2.0, NOW_MS).is_some(), "refilled");
        // Another IP is unaffected.
        assert!(d.announce(src(2), &announce(&key(2), NOW_MS, false, &info("S", 0)), 0.0, NOW_MS).is_some());
        // Query rate limit.
        let mut d = Directory::new(DirConfig { query_rate: (2.0, 0.0), ..Default::default() });
        let q = encode_query([7; 16], 0, &Filter::default());
        let got = (0..5).filter(|_| d.query(src(1), &q, 0.0, |_| true, || [7; 16]).is_some()).count();
        assert_eq!(got, 2);
    }

    #[test]
    fn a_long_list_is_paged_within_the_packet_budget() {
        let mut d = open();
        for n in 0..200u8 {
            let mut i = info(&format!("Server number {n:03} with a fairly long display name"), u16::from(n % 17));
            i.room = "ABCDEF".into();
            reply_of(&mut d, SocketAddr::from(([10, 1, n, 1], 40000)), &announce(&key(n), NOW_MS, false, &i), 0.0);
        }
        let q = |p| encode_query([7; 16], p, &Filter::default());
        let mut seen = std::collections::HashSet::new();
        let mut pages = 0;
        for p in 0.. {
            let b = d.query(src(1), &q(p), 1.0, |_| true, || [7; 16]).unwrap();
            assert!(b.len() <= MAX_LIST_BYTES, "page {p}: {} bytes", b.len());
            let Some(Reply::List { page, pages: n, total, entries }) = decode_reply(&b) else { panic!() };
            assert_eq!((page, total), (p, 200));
            pages = n;
            seen.extend(entries.into_iter().map(|l| l.key));
            if p + 1 >= n {
                break;
            }
        }
        assert!(pages > 3, "{pages} pages");
        assert_eq!(seen.len(), 200, "every listing on exactly one page");
    }

    #[test]
    fn text_is_clamped_and_control_characters_removed() {
        let mut d = open();
        let mut i = info("Evil\u{7}\nName", 1);
        i.name.push_str(&"x".repeat(200));
        i.map = "é".repeat(100);
        reply_of(&mut d, src(1), &announce(&key(1), NOW_MS, false, &i), 0.0);
        let l = &list_all(&mut d, &Filter::default(), 1.0)[0];
        assert!(l.info.name.len() <= 48 && !l.info.name.contains(['\n', '\u{7}']), "{:?}", l.info.name);
        assert!(l.info.map.len() <= 48 && l.info.map.chars().all(|c| c == 'é'));
        // Neither a port nor a room: nothing to connect to.
        let mut none = info("Nowhere", 0);
        none.port = 0;
        assert_eq!(reply_of(&mut d, src(2), &announce(&key(2), NOW_MS, false, &none), 0.0), Reply::Error("no port or room".into()));
    }
}
