//! Signed remote admin (NET-6): the console's commands over UDP, authenticated
//! by Ed25519 admin keys listed in `server.ron`.
//!
//! ```text
//! request: "ZADM" version key[32] ts_ms:u64 nonce:u64 len:u16 command signature[64]
//! reply:   "ZADR" version nonce:u64 status:u8 len:u16 text signature[64]
//! ```
//!
//! The request signature covers a domain tag, the **server's public key**
//! (a request for one server cannot be replayed to another) and the packet up
//! to the signature. A request is acted on only when all of these hold:
//! the key is listed, the signature verifies, the timestamp is within the
//! skew window and not older than the server's start (so a restart does not
//! reopen the window for a captured packet), the `(key, nonce)` pair is new,
//! and the key's role allows the command. Anything else gets **no reply** (so
//! the port reveals nothing to a scanner and cannot be used to amplify),
//! except a listed key with a valid signature, which is told why (`Stale`,
//! `Denied`). Sources are rate limited before any signature is checked. Replies
//! are signed with the server identity so the admin tool knows who answered;
//! they are not encrypted (do not put secrets in command output; use the
//! stdin console on the host for those).

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, Signer as _, SigningKey, Verifier as _, VerifyingKey};

use crate::admin::{AdminCommand, AdminKey};

pub const VERSION: u8 = 1;
/// Largest command text.
pub const MAX_COMMAND: usize = 512;
/// Largest reply text (one datagram).
pub const MAX_REPLY: usize = 1100;
/// How far a request's timestamp may be from the server's clock.
pub const SKEW_MS: u64 = 30_000;

const REQ: &[u8; 4] = b"ZADM";
const REP: &[u8; 4] = b"ZADR";
const REQ_DOMAIN: &[u8] = b"zestengine admin request v1";
const REP_DOMAIN: &[u8] = b"zestengine admin reply v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    /// The key's role does not allow the command.
    Denied = 1,
    /// The command failed or did not parse.
    Error = 2,
    /// Replayed nonce or a timestamp outside the window.
    Stale = 3,
}

impl Status {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::Denied,
            3 => Self::Stale,
            _ => Self::Error,
        }
    }
}

/// A verified reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    pub nonce: u64,
    pub status: Status,
    pub text: String,
}

pub fn unix_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

pub fn random_u64() -> u64 {
    let mut b = [0u8; 8];
    let _ = getrandom::fill(&mut b);
    u64::from_le_bytes(b)
}

fn to_hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}

/// 32 bytes from 64 hex digits.
pub fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

pub fn hex(b: &[u8; 32]) -> String {
    to_hex(b)
}

/// A request packet, signed with `key` for the server with public key `server`.
pub fn encode_request(key: &SigningKey, server: &[u8; 32], ts_ms: u64, nonce: u64, command: &str) -> Vec<u8> {
    let cmd = command.as_bytes();
    let cmd = &cmd[..cmd.len().min(MAX_COMMAND)];
    let mut b = REQ.to_vec();
    b.push(VERSION);
    b.extend_from_slice(&key.verifying_key().to_bytes());
    b.extend_from_slice(&ts_ms.to_le_bytes());
    b.extend_from_slice(&nonce.to_le_bytes());
    b.extend_from_slice(&(cmd.len() as u16).to_le_bytes());
    b.extend_from_slice(cmd);
    let mut msg = REQ_DOMAIN.to_vec();
    msg.extend_from_slice(server);
    msg.extend_from_slice(&b[4..]);
    b.extend_from_slice(&key.sign(&msg).to_bytes());
    b
}

struct Request {
    key: [u8; 32],
    ts_ms: u64,
    nonce: u64,
    command: String,
}

/// Parse and verify the signature of a request for the server `server`.
fn decode_request(d: &[u8], server: &[u8; 32]) -> Option<Request> {
    let sig_at = d.len().checked_sub(64)?;
    if d.len() < 4 + 1 + 32 + 16 + 2 + 64 || &d[..4] != REQ || d[4] != VERSION {
        return None;
    }
    let key: [u8; 32] = d[5..37].try_into().ok()?;
    let ts_ms = u64::from_le_bytes(d[37..45].try_into().ok()?);
    let nonce = u64::from_le_bytes(d[45..53].try_into().ok()?);
    let len = u16::from_le_bytes(d[53..55].try_into().ok()?) as usize;
    if len > MAX_COMMAND || 55 + len != sig_at {
        return None;
    }
    let command = String::from_utf8(d[55..sig_at].to_vec()).ok()?;
    let mut msg = REQ_DOMAIN.to_vec();
    msg.extend_from_slice(server);
    msg.extend_from_slice(&d[4..sig_at]);
    let sig = Signature::from_bytes(d[sig_at..].try_into().ok()?);
    VerifyingKey::from_bytes(&key).ok()?.verify(&msg, &sig).ok()?;
    Some(Request { key, ts_ms, nonce, command })
}

fn encode_reply(sign: &dyn Fn(&[u8]) -> [u8; 64], key: &[u8; 32], nonce: u64, status: Status, text: &str) -> Vec<u8> {
    let mut text = text.to_owned();
    if text.len() > MAX_REPLY {
        let mut n = MAX_REPLY - 3;
        while !text.is_char_boundary(n) {
            n -= 1;
        }
        text.truncate(n);
        text.push_str("...");
    }
    let mut b = REP.to_vec();
    b.push(VERSION);
    b.extend_from_slice(&nonce.to_le_bytes());
    b.push(status as u8);
    b.extend_from_slice(&(text.len() as u16).to_le_bytes());
    b.extend_from_slice(text.as_bytes());
    let mut msg = REP_DOMAIN.to_vec();
    msg.extend_from_slice(key);
    msg.extend_from_slice(&b[4..]);
    b.extend_from_slice(&sign(&msg));
    b
}

/// Check a reply to a request sent by `key` and signed by the server `server`.
pub fn verify_reply(d: &[u8], server: &[u8; 32], key: &[u8; 32]) -> Option<Reply> {
    let sig_at = d.len().checked_sub(64)?;
    if d.len() < 4 + 1 + 8 + 1 + 2 + 64 || &d[..4] != REP || d[4] != VERSION {
        return None;
    }
    let nonce = u64::from_le_bytes(d[5..13].try_into().ok()?);
    let status = Status::from_u8(d[13]);
    let len = u16::from_le_bytes(d[14..16].try_into().ok()?) as usize;
    if 16 + len != sig_at {
        return None;
    }
    let mut msg = REP_DOMAIN.to_vec();
    msg.extend_from_slice(key);
    msg.extend_from_slice(&d[4..sig_at]);
    let sig = Signature::from_bytes(d[sig_at..].try_into().ok()?);
    VerifyingKey::from_bytes(server).ok()?.verify(&msg, &sig).ok()?;
    Some(Reply { nonce, status, text: String::from_utf8_lossy(&d[16..sig_at]).into_owned() })
}

/// Send one command and wait for the server's signed answer (the admin tool, tests).
pub fn send_command(server_addr: SocketAddr, server_key: &[u8; 32], key: &SigningKey, command: &str, timeout: Duration) -> io::Result<Reply> {
    let socket = UdpSocket::bind(if server_addr.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" })?;
    socket.connect(server_addr)?;
    socket.set_read_timeout(Some(Duration::from_millis(250)))?;
    let nonce = random_u64();
    let packet = encode_request(key, server_key, unix_ms(), nonce, command);
    let start = Instant::now();
    let mut last_send: Option<Instant> = None;
    let mut buf = [0u8; 2048];
    while start.elapsed() < timeout {
        // Resending the same packet is safe: the nonce makes the server act once, and answer
        // the duplicate with Stale, which we ignore if the real answer is on its way.
        if last_send.is_none_or(|t| t.elapsed() > Duration::from_millis(700)) {
            socket.send(&packet)?;
            last_send = Some(Instant::now());
        }
        match socket.recv(&mut buf) {
            Ok(n) => {
                if let Some(r) = verify_reply(&buf[..n], server_key, &key.verifying_key().to_bytes())
                    && r.nonce == nonce
                    && !(r.status == Status::Stale && last_send.is_some_and(|t| t.elapsed() < Duration::from_secs(5)) && start.elapsed() > Duration::from_millis(700))
                {
                    return Ok(r);
                }
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::ConnectionRefused) => {}
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(io::ErrorKind::TimedOut, "no signed answer from the server (wrong key, wrong server key, blocked port, or the server is down)"))
}

/// Counters for the status line and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct RemoteStats {
    pub executed: u64,
    /// Silently dropped: unknown key, bad signature, malformed.
    pub rejected: u64,
    /// Valid signature but replayed or out of the time window.
    pub stale: u64,
    pub denied: u64,
    pub rate_limited: u64,
}

/// The UDP listener and its replay / rate state.
pub struct RemoteAdmin {
    socket: UdpSocket,
    server_key: [u8; 32],
    keys: HashMap<[u8; 32], AdminKey>,
    started_ms: u64,
    seen: HashMap<([u8; 32], u64), u64>,
    buckets: HashMap<IpAddr, (f64, Instant)>,
    pub stats: RemoteStats,
}

/// Requests per second per source IP (burst 10).
const RATE: f64 = 5.0;
const BURST: f64 = 10.0;
const MAX_SEEN: usize = 20_000;

impl RemoteAdmin {
    pub fn bind(bind: &str, port: u16, server_key: [u8; 32], keys: &[AdminKey]) -> io::Result<Self> {
        let socket = UdpSocket::bind((bind, port))?;
        socket.set_nonblocking(true)?;
        let mut map = HashMap::new();
        for k in keys {
            if let Some(p) = parse_hex32(&k.key) {
                map.insert(p, k.clone());
            }
        }
        Ok(Self { socket, server_key, keys: map, started_ms: unix_ms(), seen: HashMap::new(), buckets: HashMap::new(), stats: RemoteStats::default() })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    pub fn server_key(&self) -> &[u8; 32] {
        &self.server_key
    }

    /// Replace the listed keys (config reload).
    pub fn set_keys(&mut self, keys: &[AdminKey]) {
        self.keys = keys.iter().filter_map(|k| parse_hex32(&k.key).map(|p| (p, k.clone()))).collect();
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    fn take_token(&mut self, ip: IpAddr) -> bool {
        if self.buckets.len() > 4096 {
            self.buckets.retain(|_, (_, t)| t.elapsed() < Duration::from_secs(60));
            if self.buckets.len() > 4096 {
                return false;
            }
        }
        let now = Instant::now();
        let (tokens, last) = self.buckets.entry(ip).or_insert((BURST, now));
        *tokens = (*tokens + last.elapsed().as_secs_f64() * RATE).min(BURST);
        *last = now;
        if *tokens < 1.0 {
            return false;
        }
        *tokens -= 1.0;
        true
    }

    /// Handle every waiting datagram, running accepted commands through `exec`. `sign` signs
    /// replies with the server identity; `exec` runs a command and returns `(status, text)`.
    pub fn poll(&mut self, sign: &dyn Fn(&[u8]) -> [u8; 64], exec: &mut dyn FnMut(AdminCommand, &AdminKey) -> (Status, String)) {
        for p in self.receive(sign) {
            let (status, text) = exec(p.command.clone(), &p.key);
            self.answer(sign, &p, status, &text);
        }
    }

    /// Read and check what is waiting: stale, denied and malformed requests are answered (or
    /// dropped) here, the accepted commands are returned to be run, then [`answer`](Self::answer)ed.
    pub fn receive(&mut self, sign: &dyn Fn(&[u8]) -> [u8; 64]) -> Vec<Pending> {
        let mut accepted = Vec::new();
        let mut buf = [0u8; 2048];
        for _ in 0..64 {
            let (n, src) = match self.socket.recv_from(&mut buf) {
                Ok(v) => v,
                Err(e) if e.kind() == io::ErrorKind::ConnectionReset => continue,
                Err(_) => break,
            };
            if !self.take_token(src.ip()) {
                self.stats.rate_limited += 1;
                continue;
            }
            let now = unix_ms();
            let Some(req) = decode_request(&buf[..n], &self.server_key) else {
                self.stats.rejected += 1;
                continue;
            };
            let Some(entry) = self.keys.get(&req.key).cloned() else {
                self.stats.rejected += 1;
                continue;
            };
            let reply = |status, text: &str| encode_reply(sign, &req.key, req.nonce, status, text);
            self.seen.retain(|_, at| now.saturating_sub(*at) <= SKEW_MS * 2);
            let in_window = req.ts_ms.abs_diff(now) <= SKEW_MS && req.ts_ms >= self.started_ms;
            if !in_window || self.seen.contains_key(&(req.key, req.nonce)) || self.seen.len() >= MAX_SEEN {
                self.stats.stale += 1;
                let _ = self.socket.send_to(&reply(Status::Stale, "stale or replayed request (check the clocks)"), src);
                continue;
            }
            self.seen.insert((req.key, req.nonce), now);
            let command = match AdminCommand::parse(&req.command) {
                Ok(Some(c)) => c,
                Ok(None) => {
                    let _ = self.socket.send_to(&reply(Status::Error, "empty command"), src);
                    continue;
                }
                Err(e) => {
                    let _ = self.socket.send_to(&reply(Status::Error, &e), src);
                    continue;
                }
            };
            if command.required_role() > entry.role {
                self.stats.denied += 1;
                let text = format!("key {:?} ({:?}) may not run that command", entry.name, entry.role);
                let _ = self.socket.send_to(&reply(Status::Denied, &text), src);
                continue;
            }
            self.stats.executed += 1;
            accepted.push(Pending { command, key: entry, src, nonce: req.nonce, key_bytes: req.key });
        }
        accepted
    }

    /// Send the signed result of an accepted command.
    pub fn answer(&self, sign: &dyn Fn(&[u8]) -> [u8; 64], p: &Pending, status: Status, text: &str) {
        let _ = self.socket.send_to(&encode_reply(sign, &p.key_bytes, p.nonce, status, text), p.src);
    }
}

/// An accepted request waiting to be run.
pub struct Pending {
    pub command: AdminCommand,
    pub key: AdminKey,
    src: SocketAddr,
    nonce: u64,
    key_bytes: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admin::AdminRole;

    fn keys() -> (SigningKey, SigningKey, [u8; 32], SigningKey, Vec<AdminKey>) {
        let admin = SigningKey::from_bytes(&[1; 32]);
        let moderator = SigningKey::from_bytes(&[2; 32]);
        let server = SigningKey::from_bytes(&[3; 32]);
        let list = vec![
            AdminKey { name: "ops".into(), key: hex(&admin.verifying_key().to_bytes()), role: AdminRole::Admin },
            AdminKey { name: "mod".into(), key: hex(&moderator.verifying_key().to_bytes()), role: AdminRole::Moderator },
        ];
        (admin, moderator, server.verifying_key().to_bytes(), server, list)
    }

    /// A tiny executor: echoes the command, so tests see what ran.
    fn run(r: &mut RemoteAdmin, server: &SigningKey, ran: &mut Vec<String>) {
        r.poll(&|m| server.sign(m).to_bytes(), &mut |c, k| {
            ran.push(format!("{}:{c:?}", k.name));
            (Status::Ok, "done".into())
        });
    }

    #[test]
    fn signed_requests_run_and_replies_verify() {
        let (admin, _m, spub, server, list) = keys();
        let mut r = RemoteAdmin::bind("127.0.0.1", 0, spub, &list).unwrap();
        let addr = r.local_addr().unwrap();
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        sock.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        let nonce = 77;
        sock.send_to(&encode_request(&admin, &spub, unix_ms(), nonce, "kick 3 rude"), addr).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let mut ran = Vec::new();
        run(&mut r, &server, &mut ran);
        assert_eq!(ran, ["ops:Kick { id: 3, reason: \"rude\" }"]);
        let mut buf = [0u8; 2048];
        let (n, _) = sock.recv_from(&mut buf).unwrap();
        let rep = verify_reply(&buf[..n], &spub, &admin.verifying_key().to_bytes()).expect("signed by the server");
        assert_eq!((rep.nonce, rep.status, rep.text.as_str()), (nonce, Status::Ok, "done"));
        // A reply for another key, or one signed by someone else, does not verify.
        assert!(verify_reply(&buf[..n], &spub, &[9; 32]).is_none());
        let other = SigningKey::from_bytes(&[8; 32]);
        assert!(verify_reply(&buf[..n], &other.verifying_key().to_bytes(), &admin.verifying_key().to_bytes()).is_none());
    }

    #[test]
    fn tampering_unknown_keys_and_wrong_servers_get_silence() {
        let (admin, _m, spub, server, list) = keys();
        let mut r = RemoteAdmin::bind("127.0.0.1", 0, spub, &list).unwrap();
        let addr = r.local_addr().unwrap();
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        sock.set_read_timeout(Some(Duration::from_millis(150))).unwrap();
        let good = encode_request(&admin, &spub, unix_ms(), 1, "quit");
        let mut packets = Vec::new();
        for at in [5, 20, 40, 50, 56, good.len() - 70, good.len() - 1] {
            let mut bad = good.clone();
            bad[at] ^= 1;
            packets.push(bad);
        }
        // Unlisted key with a valid signature; a request signed for a different server; garbage.
        packets.push(encode_request(&SigningKey::from_bytes(&[50; 32]), &spub, unix_ms(), 2, "quit"));
        packets.push(encode_request(&admin, &[7; 32], unix_ms(), 3, "quit"));
        packets.push(b"ZADM\x01 short".to_vec());
        packets.push(vec![0u8; 300]);
        for p in &packets {
            sock.send_to(p, addr).unwrap();
        }
        std::thread::sleep(Duration::from_millis(40));
        let mut ran = Vec::new();
        run(&mut r, &server, &mut ran);
        assert!(ran.is_empty(), "{ran:?}");
        let mut buf = [0u8; 2048];
        assert!(sock.recv_from(&mut buf).is_err(), "no reply to anything unauthenticated");
        assert_eq!(r.stats.rejected as usize + r.stats.rate_limited as usize, packets.len());
        assert_eq!(r.stats.executed, 0);
    }

    #[test]
    fn replays_old_timestamps_and_restart_windows_are_stale() {
        let (admin, _m, spub, server, list) = keys();
        let mut r = RemoteAdmin::bind("127.0.0.1", 0, spub, &list).unwrap();
        let addr = r.local_addr().unwrap();
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        sock.set_read_timeout(Some(Duration::from_millis(500))).unwrap();
        let mut ran = Vec::new();
        let mut buf = [0u8; 2048];
        let mut ask = |r: &mut RemoteAdmin, ran: &mut Vec<String>, pkt: &[u8]| -> Reply {
            sock.send_to(pkt, addr).unwrap();
            std::thread::sleep(Duration::from_millis(30));
            run(r, &server, ran);
            let (n, _) = sock.recv_from(&mut buf).unwrap();
            verify_reply(&buf[..n], &spub, &admin.verifying_key().to_bytes()).unwrap()
        };
        let pkt = encode_request(&admin, &spub, unix_ms(), 10, "say hi");
        assert_eq!(ask(&mut r, &mut ran, &pkt).status, Status::Ok);
        assert_eq!(ask(&mut r, &mut ran, &pkt).status, Status::Stale, "the same packet again");
        assert_eq!(ran.len(), 1, "executed once");
        // Same nonce with a new timestamp is a different signed message but the nonce was used.
        assert_eq!(ask(&mut r, &mut ran, &encode_request(&admin, &spub, unix_ms() + 1, 10, "say hi")).status, Status::Stale);
        // Too old, too new, and from before the server started.
        for ts in [unix_ms() - SKEW_MS - 1000, unix_ms() + SKEW_MS + 1000, r.started_ms - 1] {
            assert_eq!(ask(&mut r, &mut ran, &encode_request(&admin, &spub, ts, random_u64(), "say hi")).status, Status::Stale, "ts {ts}");
        }
        assert_eq!(ran.len(), 1);
        assert_eq!(r.stats.stale, 5);
        // A fresh nonce works.
        assert_eq!(ask(&mut r, &mut ran, &encode_request(&admin, &spub, unix_ms(), 11, "say again")).status, Status::Ok);
    }

    #[test]
    fn roles_limit_what_a_key_may_run() {
        let (_a, moderator, spub, server, list) = keys();
        let mut r = RemoteAdmin::bind("127.0.0.1", 0, spub, &list).unwrap();
        let addr = r.local_addr().unwrap();
        let mut ran = Vec::new();
        for (n, cmd, ok) in [(1, "players", true), (2, "kick 2", true), (3, "ban 2 1h x", true), (4, "shutdown", false), (5, "restart", false), (6, "reload", false), (7, "scene a.scene", false)] {
            let reply = send_in_thread(addr, spub, &moderator, cmd, n);
            std::thread::sleep(Duration::from_millis(30));
            run(&mut r, &server, &mut ran);
            let rep = reply.join().unwrap();
            assert_eq!(rep.status == Status::Ok, ok, "{cmd}: {rep:?}");
        }
        assert_eq!(ran.len(), 3);
        assert_eq!(r.stats.denied, 4);
    }

    fn send_in_thread(addr: SocketAddr, spub: [u8; 32], key: &SigningKey, cmd: &str, nonce: u64) -> std::thread::JoinHandle<Reply> {
        let (key, cmd) = (key.clone(), cmd.to_owned());
        std::thread::spawn(move || {
            let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            sock.send_to(&encode_request(&key, &spub, unix_ms(), nonce, &cmd), addr).unwrap();
            let mut buf = [0u8; 2048];
            let (n, _) = sock.recv_from(&mut buf).unwrap();
            verify_reply(&buf[..n], &spub, &key.verifying_key().to_bytes()).unwrap()
        })
    }

    #[test]
    fn a_flood_from_one_address_is_rate_limited_before_any_signature_check() {
        let (admin, _m, spub, server, list) = keys();
        let mut r = RemoteAdmin::bind("127.0.0.1", 0, spub, &list).unwrap();
        let addr = r.local_addr().unwrap();
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        for i in 0..40 {
            sock.send_to(&encode_request(&admin, &spub, unix_ms(), 1000 + i, "status"), addr).unwrap();
        }
        std::thread::sleep(Duration::from_millis(40));
        let mut ran = Vec::new();
        run(&mut r, &server, &mut ran);
        assert!(ran.len() <= 11, "burst of 10 plus a refill: {}", ran.len());
        assert!(r.stats.rate_limited >= 28, "{:?}", r.stats);
    }
}
