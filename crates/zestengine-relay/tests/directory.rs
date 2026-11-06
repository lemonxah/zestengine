//! The directory over real loopback UDP: a relay on its own thread, servers
//! announcing with signed packets, a browser listing with `query_blocking`.

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signer, SigningKey};
use zestengine_relay::directory::{self, DirConfig, Filter, Reply, ServerInfo, decode_reply, encode_announce, encode_query, query_blocking};
use zestengine_relay::{Relay, RelayConfig};

struct Running {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn start(dir: DirConfig) -> Running {
    let mut relay = Relay::bind("127.0.0.1:0", RelayConfig { directory: Some(dir), ..Default::default() }).unwrap();
    let addr = relay.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let thread = std::thread::spawn(move || relay.run(&flag).unwrap());
    Running { addr, stop, thread: Some(thread) }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

fn info(name: &str, players: u16, port: u16) -> ServerInfo {
    ServerInfo { name: name.into(), game: "arena".into(), version: "1".into(), map: "main".into(), region: "eu".into(), players, max_players: 16, port, protocol: 1, ..Default::default() }
}

fn send_announce(relay: SocketAddr, sock: &UdpSocket, k: &SigningKey, ts: u64, delist: bool, i: &ServerInfo) -> Reply {
    sock.send_to(&encode_announce(k.verifying_key().to_bytes(), ts, delist, i, |m| k.sign(m).to_bytes()), relay).unwrap();
    let mut buf = [0u8; 2048];
    let (n, _) = sock.recv_from(&mut buf).expect("reply");
    decode_reply(&buf[..n]).expect("directory reply")
}

fn sock() -> UdpSocket {
    let s = UdpSocket::bind("127.0.0.1:0").unwrap();
    s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    s
}

#[test]
fn servers_register_and_a_browser_lists_them() {
    let relay = start(DirConfig { per_ip: 10, ..Default::default() });
    let t0 = Instant::now();
    let (s1, s2) = (sock(), sock());
    let (k1, k2) = (SigningKey::from_bytes(&[1; 32]), SigningKey::from_bytes(&[2; 32]));
    assert!(matches!(send_announce(relay.addr, &s1, &k1, unix_ms(), false, &info("Alpha", 3, 9001)), Reply::Ack { .. }));
    let mut other = info("Bravo", 8, 9002);
    other.region = "na".into();
    assert!(matches!(send_announce(relay.addr, &s2, &k2, unix_ms(), false, &other), Reply::Ack { .. }));

    let all = query_blocking(relay.addr, &Filter::default(), Duration::from_secs(3)).unwrap();
    assert_eq!(all.iter().map(|l| l.info.name.as_str()).collect::<Vec<_>>(), ["Bravo", "Alpha"]);
    assert_eq!(all[1].addr, SocketAddr::from(([127, 0, 0, 1], 9001)));
    let eu = query_blocking(relay.addr, &Filter { region: "eu".into(), ..Default::default() }, Duration::from_secs(3)).unwrap();
    assert_eq!(eu.len(), 1);
    assert_eq!(eu[0].key, k1.verifying_key().to_bytes());

    // Delist (signed) removes it at once.
    assert!(matches!(send_announce(relay.addr, &s1, &k1, unix_ms() + 1, true, &info("Alpha", 3, 9001)), Reply::Ack { ttl: 0, .. }));
    assert_eq!(query_blocking(relay.addr, &Filter::default(), Duration::from_secs(3)).unwrap().len(), 1);
    eprintln!("directory register + 4 queries over loopback UDP: {:?}", t0.elapsed());
}

#[test]
fn spoofing_replay_and_amplification_over_the_wire() {
    let relay = start(DirConfig::default());
    let s = sock();
    let k = SigningKey::from_bytes(&[5; 32]);
    let ts = unix_ms();
    let i = info("Real", 1, 9100);
    let packet = encode_announce(k.verifying_key().to_bytes(), ts, false, &i, |m| k.sign(m).to_bytes());
    s.send_to(&packet, relay.addr).unwrap();
    let mut buf = [0u8; 2048];
    let (n, _) = s.recv_from(&mut buf).unwrap();
    assert!(matches!(decode_reply(&buf[..n]), Some(Reply::Ack { .. })));
    // The same bytes again (a replay) and a doctored copy.
    s.send_to(&packet, relay.addr).unwrap();
    let (n, _) = s.recv_from(&mut buf).unwrap();
    assert_eq!(decode_reply(&buf[..n]), Some(Reply::Error("replayed".into())));
    let mut doctored = packet.clone();
    doctored[40] ^= 0x80;
    s.send_to(&doctored, relay.addr).unwrap();
    let (n, _) = s.recv_from(&mut buf).unwrap();
    assert!(matches!(decode_reply(&buf[..n]), Some(Reply::Error(_))));
    assert!(n < packet.len(), "an error reply is smaller than the request");
    // A forger with its own key cannot take over the listing, a different key lists separately.
    let forger = SigningKey::from_bytes(&[6; 32]);
    let stolen = encode_announce(k.verifying_key().to_bytes(), ts + 10, false, &info("Hijack", 0, 6666), |m| forger.sign(m).to_bytes());
    s.send_to(&stolen, relay.addr).unwrap();
    let (n, _) = s.recv_from(&mut buf).unwrap();
    assert_eq!(decode_reply(&buf[..n]), Some(Reply::Error("bad signature".into())));
    let list = query_blocking(relay.addr, &Filter::default(), Duration::from_secs(3)).unwrap();
    assert_eq!((list.len(), list[0].info.name.as_str()), (1, "Real"));

    // A query without a cookie only gets the (smaller) cookie, never the list.
    let q = encode_query([0; 16], 0, &Filter::default());
    s.send_to(&q, relay.addr).unwrap();
    let (n, _) = s.recv_from(&mut buf).unwrap();
    assert!(n < q.len(), "{n} < {}", q.len());
    assert!(matches!(decode_reply(&buf[..n]), Some(Reply::Cookie(_))));
    // A short query (no padding) is ignored: no reply at all.
    s.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
    s.send_to(&q[..20], relay.addr).unwrap();
    assert!(s.recv_from(&mut buf).is_err());
    let _ = directory::R_DIRECTORY;
}

#[test]
fn listings_expire_without_heartbeats_and_live_with_them() {
    let relay = start(DirConfig { ttl: 2.0, ..Default::default() });
    let s = sock();
    let k = SigningKey::from_bytes(&[8; 32]);
    let t0 = Instant::now();
    let mut ts = unix_ms();
    for _ in 0..3 {
        ts += 1;
        assert!(matches!(send_announce(relay.addr, &s, &k, ts, false, &info("Beat", 1, 9200)), Reply::Ack { .. }));
        std::thread::sleep(Duration::from_millis(900));
    }
    assert_eq!(query_blocking(relay.addr, &Filter::default(), Duration::from_secs(3)).unwrap().len(), 1, "heartbeats keep it listed past the TTL");
    std::thread::sleep(Duration::from_millis(2300));
    assert!(query_blocking(relay.addr, &Filter::default(), Duration::from_secs(3)).unwrap().is_empty(), "silent server expired");
    eprintln!("ttl test: {:?}", t0.elapsed());
}

#[test]
fn a_relay_without_directory_mode_ignores_directory_packets() {
    let mut relay = Relay::bind("127.0.0.1:0", RelayConfig::default()).unwrap();
    let addr = relay.local_addr().unwrap();
    let s = sock();
    s.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
    s.send_to(&encode_query([0; 16], 0, &Filter::default()), addr).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    relay.poll();
    let mut buf = [0u8; 64];
    assert!(s.recv_from(&mut buf).is_err());
    assert!(relay.directory().is_none());
}
