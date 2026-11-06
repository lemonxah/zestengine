//! NET-4 over real UDP on the loopback interface: the dedicated topology,
//! the pinned server key, abuse leading to a kick and a ban that survives a
//! server restart. CPU only.

use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use glam::{Quat, Vec3};
use zestengine_core::{NetSpawnSource, NetWorldOp, ObjectId, RpcTarget, Topology};
use zestengine_engine::net::{ConnState, Identity, NetConfig, NetSession, wire};
use zestengine_server::server::ServerHandle;
use zestengine_server::ServerConfig;

/// A scaffolded `vr_social` project in a temp folder.
struct Project(PathBuf);

impl Project {
    fn new() -> Self {
        let parent = std::env::temp_dir().join(format!("zestengine-server-val-{}-{}", std::process::id(), ObjectId::new()));
        std::fs::create_dir_all(&parent).unwrap();
        let root = zestengine_engine::scaffold_project(&parent, "vr_social", "vr_social").expect("scaffold template");
        Project(root)
    }

    fn config(&self, port: u16) -> ServerConfig {
        ServerConfig {
            project: self.0.display().to_string(),
            port: Some(port),
            tick_rate: 60.0,
            lan_discovery: false,
            console: false,
            stats_interval: 0.0,
            ..Default::default()
        }
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
    }
}

fn free_port() -> u16 {
    UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

struct Client {
    net: NetSession,
    ops: Vec<NetWorldOp>,
}

impl Client {
    fn join(addr: SocketAddr, name: &str, identity: Identity) -> Self {
        let cfg = NetConfig { player_name: name.into(), discovery: false, identity, ..Default::default() };
        Self { net: NetSession::join_with(&addr.to_string(), cfg).unwrap(), ops: vec![] }
    }

    fn poll(&mut self) {
        self.net.pump();
        self.ops.extend(self.net.view().world_ops);
    }

    fn disconnected(&self) -> Option<&str> {
        match self.net.state() {
            ConnState::Disconnected(r) => Some(r),
            _ => None,
        }
    }

    fn joined(&self) -> bool {
        self.net.is_connected() && self.ops.iter().any(|op| matches!(op, NetWorldOp::Spawn { owner, .. } if *owner == self.net.local_peer()))
    }
}

fn wait_for(clients: &mut [&mut Client], what: &str, mut done: impl FnMut(&mut [&mut Client]) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        for c in clients.iter_mut() {
            c.poll();
        }
        if done(clients) {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_dedicated_server_announces_its_topology_has_no_migration_plan_and_clients_pin_its_key() {
    let project = Project::new();
    let port = free_port();
    let target = format!("127.0.0.1:{port}");
    let server = ServerHandle::spawn(project.config(port)).unwrap();
    let mut a = Client::join(server.addr, "Ann", Identity::generate());
    let mut b = Client::join(server.addr, "Bob", Identity::generate());
    wait_for(&mut [&mut a, &mut b], "joined", |c| c[0].joined() && c[1].joined());
    assert_eq!(a.net.topology(), Topology::Dedicated, "the server tells clients what it is");
    for c in [&mut a, &mut b] {
        c.poll();
        assert!(c.net.succession().is_none(), "no succession plan, so no player address ever leaves the server");
        assert!(c.net.migration_status().is_none());
    }
    assert!(NetSession::host_is_pinned(&target), "the first join pinned the server's key");
    let server_view = server.remote.call(|s| (s.host().net().unwrap().topology(), s.host().net().unwrap().succession().is_none())).unwrap();
    assert_eq!(server_view, (Topology::Dedicated, true));
    server.shutdown(Duration::from_secs(5)).unwrap();

    // Restart with the same persistent identity: the pin matches.
    let server = ServerHandle::spawn(project.config(port)).unwrap();
    let mut c = Client::join(server.addr, "Cy", Identity::generate());
    wait_for(&mut [&mut c], "joined the restarted server", |c| c[0].joined());
    server.shutdown(Duration::from_secs(5)).unwrap();

    // A different key on the same address (a reinstalled server, or an impostor): refused with a clear error.
    let impostor_key = project.0.join("other-identity.key");
    let cfg = ServerConfig { identity_file: impostor_key.display().to_string(), ..project.config(port) };
    let server = ServerHandle::spawn(cfg).unwrap();
    let mut d = Client::join(server.addr, "Di", Identity::generate());
    wait_for(&mut [&mut d], "refusal", |c| c[0].disconnected().is_some());
    let why = d.disconnected().unwrap().to_owned();
    assert!(why.contains("identity key changed") && why.contains("impostor") && why.contains("forget"), "{why}");
    // It tells the server it is leaving, so the server does not wait for a timeout.
    let deadline = Instant::now() + Duration::from_secs(5);
    while server.remote.call(|s| s.player_count()) != Some(0) {
        assert!(Instant::now() < deadline, "the refused client still counts as a player");
        std::thread::sleep(Duration::from_millis(10));
    }

    // The user clears the pin on purpose: the new key is accepted and pinned.
    assert!(NetSession::forget_host_pin(&target));
    let mut e = Client::join(server.addr, "Ed", Identity::generate());
    wait_for(&mut [&mut e], "joined after clearing the pin", |c| c[0].joined());
    assert!(NetSession::host_is_pinned(&target));
    server.shutdown(Duration::from_secs(5)).unwrap();
    NetSession::forget_host_pin(&target);
}

fn flood(c: &mut Client, mal: u64) {
    for _ in 0..150 {
        c.net.inject_plain(&wire::urpc(mal, RpcTarget::All, 0, "spam", &[]));
    }
}

#[test]
fn abuse_over_udp_is_refused_then_kicks_then_bans_by_key_and_the_ban_survives_a_restart() {
    let project = Project::new();
    let port = free_port();
    let server = ServerHandle::spawn(project.config(port)).unwrap();
    let mut ann = Client::join(server.addr, "Ann", Identity::generate());
    let mallory = Identity::generate();
    let mut mal = Client::join(server.addr, "Mallory", mallory.clone());
    wait_for(&mut [&mut ann, &mut mal], "both joined", |c| c[0].joined() && c[1].joined());
    let spawns_before = server.remote.call(|s| s.host().net().unwrap().world.spawns.len()).unwrap();

    // Spawns are not allowed on a dedicated server by default.
    let me = mal.net.local_peer();
    mal.net.spawn(NetSpawnSource::Prefab("prefabs/ball.prefab".into()), ObjectId::new(), Vec3::ZERO, Quat::IDENTITY, me);
    // Forged sender ids, oversized packets, NaN.
    mal.net.inject_plain(&wire::voice(ann.net.local_peer(), 1, b"x"));
    mal.net.inject_plain(&wire::garbage(9, 1500));
    wait_for(&mut [&mut ann, &mut mal], "refusals counted", |_| {
        server.remote.call(move |s| s.host().net().unwrap().val.counts(me).iter().sum::<u32>()).unwrap() >= 3
    });
    let counts = server.remote.call(move |s| s.host().net().unwrap().val.counts(me)).unwrap();
    assert!(counts.iter().sum::<u32>() >= 3, "{counts:?}");
    assert_eq!(server.remote.call(|s| s.host().net().unwrap().world.spawns.len()).unwrap(), spawns_before, "nothing was spawned");

    // Flooding: kicked, with a reason.
    let deadline = Instant::now() + Duration::from_secs(20);
    while mal.disconnected().is_none() {
        assert!(Instant::now() < deadline, "never kicked");
        flood(&mut mal, me);
        wait_for(&mut [&mut ann, &mut mal], "tick", |_| true);
        std::thread::sleep(Duration::from_millis(10));
    }
    let why = mal.disconnected().unwrap().to_owned();
    assert!(why.contains("kicked") && why.contains("violations"), "{why}");
    assert!(ann.net.is_connected(), "Ann was not affected");

    // The same key again: let in once more, then banned for the next flood.
    let mut mal2 = Client::join(server.addr, "Mallory2", mallory.clone());
    wait_for(&mut [&mut ann, &mut mal2], "rejoined after a kick", |c| c[1].joined());
    let me2 = mal2.net.local_peer();
    let deadline = Instant::now() + Duration::from_secs(20);
    while mal2.disconnected().is_none() {
        assert!(Instant::now() < deadline, "never banned");
        flood(&mut mal2, me2);
        wait_for(&mut [&mut ann, &mut mal2], "tick", |_| true);
        std::thread::sleep(Duration::from_millis(10));
    }
    let why = mal2.disconnected().unwrap().to_owned();
    assert!(why.contains("banned"), "{why}");
    let mut mal3 = Client::join(server.addr, "Mallory3", mallory.clone());
    wait_for(&mut [&mut ann, &mut mal3], "refused at the door", |c| c[1].disconnected().is_some());
    assert!(mal3.disconnected().unwrap().contains("banned"), "{:?}", mal3.disconnected());
    let bans_file = project.0.join(".server/bans.ron");
    assert!(bans_file.is_file(), "the ban list is on disk: {}", bans_file.display());
    let fp = mallory.fingerprint();
    assert!(std::fs::read_to_string(&bans_file).unwrap().contains(&fp));
    server.shutdown(Duration::from_secs(5)).unwrap();

    // A restarted server reads bans.ron: the key is still refused (a different key is not).
    let server = ServerHandle::spawn(project.config(port)).unwrap();
    let mut mal4 = Client::join(server.addr, "Mallory4", mallory);
    let mut fine = Client::join(server.addr, "Fine", Identity::generate());
    wait_for(&mut [&mut mal4, &mut fine], "verdicts", |c| c[0].disconnected().is_some() && c[1].joined());
    assert!(mal4.disconnected().unwrap().contains("banned"), "{:?}", mal4.disconnected());
    server.shutdown(Duration::from_secs(5)).unwrap();
    NetSession::forget_host_pin(&format!("127.0.0.1:{port}"));
}

#[test]
fn server_ron_overrides_the_validation_settings() {
    let cfg = ServerConfig::from_ron_str("(project: \"p\", validation: (max_rewind: 0.2, strict_rpc_permissions: true, rates: (rpc: (rate: 1.0, burst: 2.0))))").unwrap();
    let v = cfg.validation.clone().expect("parsed");
    assert_eq!((v.max_rewind, v.strict_rpc_permissions, v.rates.rpc.burst), (0.2, true, 2.0));
    assert_eq!(v.prop_speed, zestengine_core::NetValidation::default().prop_speed, "unnamed fields keep their defaults");
    assert!(ServerConfig::from_ron_str("(project: \"p\")").unwrap().validation.is_none(), "no override: the scene's settings apply");

    let project = Project::new();
    let server = ServerHandle::spawn(ServerConfig { validation: Some(v), ..project.config(0) }).unwrap();
    let server_addr = server.addr.to_string();
    let seen = server
        .remote
        .call(|s| {
            let net = s.host().net().unwrap();
            let window = net.history.lock().unwrap().max_rewind();
            (net.val.settings().max_rewind, window, net.val.strict(), net.val.settings().rates.rpc.burst)
        })
        .unwrap();
    assert_eq!(seen.0, 0.2);
    assert!((seen.1 - 0.2).abs() < 1e-6, "the lag-compensation window follows the setting: {}", seen.1);
    assert!(seen.2, "dedicated sessions apply the strict rules");
    assert_eq!(seen.3, 2.0);
    // A client sees the rate limit: the third rapid RPC is dropped.
    let mut a = Client::join(server.addr, "Ann", Identity::generate());
    wait_for(&mut [&mut a], "joined", |c| c[0].joined());
    let me = a.net.local_peer();
    for _ in 0..10 {
        a.net.inject_plain(&wire::urpc(me, RpcTarget::Others, 0, "ping", &[]));
    }
    wait_for(&mut [&mut a], "counted", |_| {
        server.remote.call(move |s| s.host().net().unwrap().val.count(me, zestengine_engine::net_validate::Reason::RateLimited)).unwrap() >= 1
    });
    server.shutdown(Duration::from_secs(5)).unwrap();
    NetSession::forget_host_pin(&server_addr);
}
