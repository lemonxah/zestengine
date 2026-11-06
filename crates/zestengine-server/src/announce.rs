//! Announces this server to a directory relay (NET-6): a signed packet about
//! every `directory_interval` seconds, and a signed delisting on shutdown.
//! A server that crashes simply stops announcing and drops off the list when
//! the directory's TTL runs out.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant};

use zestengine_engine::net::Identity;
use zestengine_relay::directory::{Reply, ServerInfo, decode_reply, encode_announce};

/// Resolve `host[:port]` (default relay port).
pub fn resolve_relay(text: &str) -> io::Result<SocketAddr> {
    let with_port = if text.rsplit_once(':').is_some_and(|(_, p)| p.parse::<u16>().is_ok()) { text.to_owned() } else { format!("{text}:{}", zestengine_relay::proto::DEFAULT_PORT) };
    with_port.to_socket_addrs()?.next().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("cannot resolve {text:?}")))
}

pub struct Announcer {
    socket: UdpSocket,
    relay: SocketAddr,
    interval: Duration,
    last_send: Option<Instant>,
    last_ts: u64,
    /// The last result worth logging again only when it changes.
    last_state: String,
    pub acks: u64,
    pub errors: u64,
}

impl Announcer {
    pub fn new(relay: &str, interval_secs: f32) -> io::Result<Self> {
        let relay = resolve_relay(relay)?;
        let socket = UdpSocket::bind(if relay.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" })?;
        socket.set_nonblocking(true)?;
        socket.connect(relay)?;
        Ok(Self { socket, relay, interval: Duration::from_secs_f32(interval_secs.max(1.0)), last_send: None, last_ts: 0, last_state: String::new(), acks: 0, errors: 0 })
    }

    pub fn relay(&self) -> SocketAddr {
        self.relay
    }

    /// Timestamps must grow with every announcement of one key, even if the clock steps back.
    fn next_ts(&mut self) -> u64 {
        self.last_ts = crate::remote::unix_ms().max(self.last_ts + 1);
        self.last_ts
    }

    fn send(&mut self, identity: &Identity, info: &ServerInfo, delist: bool) {
        let ts = self.next_ts();
        let packet = encode_announce(identity.public(), ts, delist, info, |m| identity.sign(m));
        let _ = self.socket.send(&packet);
        self.last_send = Some(Instant::now());
    }

    /// Announce if it is time, and read the relay's answers. Call every tick.
    pub fn tick(&mut self, identity: &Identity, info: impl FnOnce() -> ServerInfo) {
        if self.last_send.is_none_or(|t| t.elapsed() >= self.interval) {
            self.send(identity, &info(), false);
        }
        let mut buf = [0u8; 512];
        while let Ok(n) = self.socket.recv(&mut buf) {
            let state = match decode_reply(&buf[..n]) {
                Some(Reply::Ack { ttl, .. }) => {
                    self.acks += 1;
                    format!("listed on {} (expires after {ttl} s without an announcement)", self.relay)
                }
                Some(Reply::Error(why)) => {
                    self.errors += 1;
                    format!("directory {} refused the announcement: {why}", self.relay)
                }
                _ => continue,
            };
            if state != self.last_state {
                if self.errors > 0 && state.contains("refused") {
                    tracing::warn!("{state}");
                } else {
                    tracing::info!("{state}");
                }
                self.last_state = state;
            }
        }
    }

    /// Take the server off the list (best effort; the TTL is the fallback).
    pub fn delist(&mut self, identity: &Identity, info: &ServerInfo) {
        self.send(identity, info, true);
    }
}
