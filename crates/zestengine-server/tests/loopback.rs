//! In-process server tests over real UDP on the loopback interface: a server
//! thread hosts a template scene, real `NetSession` clients join it. CPU only.

use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use zestengine_core::{NetEvent, NetValue, NetWorldOp};
use zestengine_engine::net::{ConnState, Identity, NetConfig, NetSession, PROTOCOL};
use zestengine_server::server::{ExitReason, ServerHandle, selftest};
use zestengine_server::{AdminCommand, ServerConfig};

/// A scaffolded `vr_social` project (Network Manager, spawn points, avatar prefab) in a temp folder.
struct Project(PathBuf);

impl Project {
    fn new() -> Self {
        let parent = std::env::temp_dir().join(format!("zestengine-server-it-{}-{}", std::process::id(), zestengine_core::ObjectId::new()));
        std::fs::create_dir_all(&parent).unwrap();
        let root = zestengine_engine::scaffold_project(&parent, "vr_social", "vr_social").expect("scaffold template");
        Project(root)
    }

    fn config(&self) -> ServerConfig {
        ServerConfig {
            project: self.0.display().to_string(),
            port: Some(0),
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

/// A real client session plus everything it has been told so far.
struct Client {
    net: NetSession,
    ops: Vec<NetWorldOp>,
    events: Vec<NetEvent>,
    messages: Vec<String>,
}

impl Client {
    fn join(addr: SocketAddr, name: &str, secret: &str) -> Self {
        let cfg = NetConfig { player_name: name.into(), discovery: false, secret: secret.into(), identity: Identity::generate(), ..Default::default() };
        Self { net: NetSession::join_with(&addr.to_string(), cfg).unwrap(), ops: vec![], events: vec![], messages: vec![] }
    }

    fn poll(&mut self) {
        self.net.pump();
        let v = self.net.view();
        self.ops.extend(v.world_ops);
        self.events.extend(v.events);
        self.messages.extend(v.messages.into_iter().map(|m| m.2));
    }

    fn disconnected(&self) -> Option<&str> {
        match self.net.state() {
            ConnState::Disconnected(r) => Some(r),
            _ => None,
        }
    }

    fn player_spawn_owner(&self) -> Option<u64> {
        self.ops.iter().find_map(|op| match op {
            NetWorldOp::Spawn { owner, .. } if *owner == self.net.local_peer() => Some(*owner),
            _ => None,
        })
    }
}

fn wait_for(clients: &mut [&mut Client], what: &str, mut done: impl FnMut(&mut [&mut Client]) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
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

fn joined(c: &mut Client) -> bool {
    c.net.is_connected() && c.player_spawn_owner().is_some()
}

#[test]
fn join_spawn_replicate_motd_and_admin_console() {
    let project = Project::new();
    let server = ServerHandle::spawn(ServerConfig { motd: "welcome to the arena".into(), ..project.config() }).unwrap();
    let mut a = Client::join(server.addr, "Ann", "");
    let mut b = Client::join(server.addr, "Bob", "");
    wait_for(&mut [&mut a, &mut b], "both joined with a player spawned", |c| joined(c[0]) && joined(c[1]));
    assert_ne!(a.net.local_peer(), b.net.local_peer());
    wait_for(&mut [&mut a, &mut b], "motd", |c| c.iter().all(|c| c.messages.iter().any(|m| m == "welcome to the arena")));

    // Dedicated: only the two joiners have players (the server's own peer 1 has none).
    let owners = server.remote.call(|s| (s.player_count(), s.host().net().unwrap().owners_snapshot())).unwrap();
    assert_eq!(owners.0, 2);
    let mut player_owners: Vec<u64> = owners.1.iter().map(|(_, p)| *p).collect();
    player_owners.sort();
    assert_eq!(player_owners, vec![a.net.local_peer().min(b.net.local_peer()), a.net.local_peer().max(b.net.local_peer())], "one player object per peer, none for the server: {owners:?}");

    // A replicated variable written on the server reaches both clients.
    let target = owners.1[0].0;
    server.remote.call(move |s| s.host_mut().net_mut().unwrap().set_var(target, "score", NetValue::Int(7))).unwrap();
    wait_for(&mut [&mut a, &mut b], "variable on both clients", |c| {
        c.iter().all(|c| c.events.iter().any(|e| matches!(e, NetEvent::VarChanged { key, value: NetValue::Int(7), from: 1, .. } if key == "score")))
    });

    // Console commands.
    let status = server.remote.command(AdminCommand::Status).unwrap();
    assert!(status.contains("players 2/16"), "{status}");
    let players = server.remote.command(AdminCommand::Players).unwrap();
    assert!(players.contains("Ann") && players.contains("Bob"), "{players}");
    server.remote.command(AdminCommand::Say("restart in 5".into())).unwrap();
    wait_for(&mut [&mut a, &mut b], "say", |c| c.iter().all(|c| c.messages.iter().any(|m| m == "restart in 5")));
    assert!(server.remote.command(AdminCommand::Kick { id: 99, reason: String::new() }).unwrap().contains("no player"));
    let kick = server.remote.command(AdminCommand::Kick { id: b.net.local_peer(), reason: "afk".into() }).unwrap();
    assert!(kick.contains("kicked Bob"), "{kick}");
    wait_for(&mut [&mut a, &mut b], "Bob kicked", |c| c[1].disconnected().is_some());
    assert!(b.disconnected().unwrap().contains("kicked by the host: afk"), "{:?}", b.disconnected());
    assert!(a.net.is_connected());
    wait_for(&mut [&mut a], "server drops Bob", |_| server.remote.call(|s| s.player_count()) == Some(1));
    assert!(server.remote.command(AdminCommand::Scene("scenes/nope.scene".into())).unwrap().contains("no scene"));
    assert!(server.remote.command(AdminCommand::Quit).unwrap().contains("shutting down"));
    assert_eq!(server.shutdown(Duration::from_secs(5)).unwrap(), ExitReason::AdminQuit);
}

#[test]
fn max_players_is_enforced_server_not_counted() {
    let project = Project::new();
    let server = ServerHandle::spawn(ServerConfig { max_players: 2, ..project.config() }).unwrap();
    let mut a = Client::join(server.addr, "A", "");
    let mut b = Client::join(server.addr, "B", "");
    wait_for(&mut [&mut a, &mut b], "two players in", |c| joined(c[0]) && joined(c[1]));
    let mut c = Client::join(server.addr, "C", "");
    wait_for(&mut [&mut a, &mut b, &mut c], "third refused", |c| c[2].disconnected().is_some());
    assert!(c.disconnected().unwrap().contains("full"), "{:?}", c.disconnected());
    assert_eq!(server.remote.call(|s| s.player_count()), Some(2));
    // A free slot is usable again.
    b.net.disconnect();
    wait_for(&mut [&mut a], "slot freed", |_| server.remote.call(|s| s.player_count()) == Some(1));
    let mut d = Client::join(server.addr, "D", "");
    wait_for(&mut [&mut a, &mut d], "D joins", |c| joined(c[1]));
}

#[test]
fn password_gates_the_session() {
    let project = Project::new();
    let server = ServerHandle::spawn(ServerConfig { password: "open sesame".into(), ..project.config() }).unwrap();
    let mut bad = Client::join(server.addr, "Mallory", "open sesame?");
    let mut none = Client::join(server.addr, "Nobody", "");
    let mut good = Client::join(server.addr, "Alice", "open sesame");
    wait_for(&mut [&mut bad, &mut none, &mut good], "verdicts", |c| c[0].disconnected().is_some() && c[1].disconnected().is_some() && joined(c[2]));
    assert!(bad.disconnected().unwrap().contains("wrong session secret"), "{:?}", bad.disconnected());
    assert!(none.disconnected().unwrap().contains("wrong session secret"), "{:?}", none.disconnected());
    assert_eq!(server.remote.call(|s| s.player_count()), Some(1));
}

#[test]
fn protocol_mismatch_is_rejected_with_an_explanation() {
    let project = Project::new();
    let server = ServerHandle::spawn(project.config()).unwrap();
    let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
    sock.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    let mut buf = [0u8; 2048];

    // An old client (short legacy HELLO, protocol 5).
    sock.send_to(&[1, 5, 0], server.addr).unwrap();
    let (n, _) = sock.recv_from(&mut buf).expect("legacy reject");
    let text = String::from_utf8_lossy(&buf[..n]).into_owned();
    assert!(text.contains(&format!("protocol version mismatch (host {PROTOCOL}, you 5)")), "{text}");

    // A newer client (padded HELLO, protocol 99).
    let mut hello = vec![1u8, 99, 0];
    hello.extend_from_slice(&[7; 16]);
    hello.push(0);
    hello.resize(128, 0);
    sock.send_to(&hello, server.addr).unwrap();
    let (n, _) = sock.recv_from(&mut buf).expect("hint reject");
    let text = String::from_utf8_lossy(&buf[..n]).into_owned();
    assert!(text.contains("protocol version mismatch") && n <= hello.len(), "{text}");

    // Nobody got in, and a matching client still can.
    assert_eq!(server.remote.call(|s| s.player_count()), Some(0));
    let mut ok = Client::join(server.addr, "Ok", "");
    wait_for(&mut [&mut ok], "matching client joins", |c| joined(c[0]));
}

#[test]
fn graceful_shutdown_notifies_players_and_exits_in_time() {
    let project = Project::new();
    let server = ServerHandle::spawn(project.config()).unwrap();
    let mut a = Client::join(server.addr, "Ann", "");
    let mut b = Client::join(server.addr, "Bob", "");
    wait_for(&mut [&mut a, &mut b], "joined", |c| joined(c[0]) && joined(c[1]));
    let t0 = Instant::now();
    let reason = server.shutdown(Duration::from_secs(3)).expect("exits within the budget");
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    assert_eq!(reason, ExitReason::Stopped);
    wait_for(&mut [&mut a, &mut b], "players told", |c| c.iter().all(|c| c.disconnected().is_some()));
    for c in [&a, &b] {
        assert!(c.disconnected().unwrap().contains("server shutting down"), "{:?}", c.disconnected());
    }
}

#[test]
fn startup_errors_are_clear_and_the_port_conflict_is_reported() {
    let project = Project::new();
    let server = ServerHandle::spawn(project.config()).unwrap();
    let busy = server.addr.port();
    let e = ServerHandle::spawn(ServerConfig { port: Some(busy), ..project.config() }).err().expect("second bind fails").to_string();
    assert!(e.contains("binding UDP port"), "{e}");
    let e = ServerHandle::spawn(ServerConfig { scene: Some("scenes/none.scene".into()), ..project.config() }).err().expect("missing scene").to_string();
    assert!(e.contains("not found"), "{e}");
}

#[test]
fn selftest_runs_on_a_template_project() {
    let project = Project::new();
    let summary = selftest(project.config()).expect("selftest");
    assert!(summary.starts_with("selftest ok"), "{summary}");
}
