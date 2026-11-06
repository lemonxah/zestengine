//! NET-6 over real loopback UDP: persistent state (round trip, crash-safety),
//! signed remote admin with bans, the server directory on a relay, dedicated
//! sessions through the relay, and the shutdown redirect. CPU only.

use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use zestengine_core::{NetEvent, NetValue, NetWorldOp, NetworkVariables, ObjectId};
use zestengine_engine::net::{ConnState, Identity, NetConfig, NetSession, fingerprint};
use zestengine_engine::server::{follow_redirect, redirect_target};
use zestengine_relay::directory::{DirConfig, Filter, query_blocking};
use zestengine_relay::{Relay, RelayConfig};
use zestengine_server::remote::{self, Status};
use zestengine_server::server::{ExitReason, ServerHandle};
use zestengine_server::{AdminCommand, AdminKey, AdminRole, ServerConfig};

const LEVER: &str = "11111111-2222-3333-4444-555555555555";

/// A scaffolded `vr_social` project (Network Manager, spawn points, avatar prefab) in a temp folder.
struct Project(PathBuf);

impl Project {
    fn new() -> Self {
        let parent = std::env::temp_dir().join(format!("zestengine-services-{}-{}", std::process::id(), ObjectId::new()));
        std::fs::create_dir_all(&parent).unwrap();
        Project(zestengine_engine::scaffold_project(&parent, "vr_social", "vr_social").expect("scaffold template"))
    }

    fn config(&self) -> ServerConfig {
        ServerConfig {
            project: self.0.display().to_string(),
            port: Some(0),
            tick_rate: 60.0,
            lan_discovery: false,
            console: false,
            stats_interval: 0.0,
            persist_interval: 0.0,
            ..Default::default()
        }
    }

    fn state(&self) -> PathBuf {
        self.0.join(".server")
    }

    /// A "Lever" scene object that persists its transform and variables.
    fn add_lever(&self) {
        let path = self.0.join("scenes/main.scene");
        let mut text = std::fs::read_to_string(&path).unwrap();
        let start = text.find("entries: [").unwrap();
        let end = start + text[start..].find("\n    ],\n").unwrap() + 1;
        let entry = format!(
            "        (\n            id: \"{LEVER}\",\n            name: \"Lever\",\n            source: Empty,\n            material: File(\"materials/default.material\"),\n            components: [\n                (kind: \"Network Variables\", data: \"(vars:[],anyone_can_write:false)\"),\n                (kind: \"Persistent State\", data: \"(transform:true,variables:true,per_player:false)\"),\n            ],\n            translation: (0.0, 0.0, 0.0),\n            rotation: (0.0, 0.0, 0.0, 1.0),\n            scale: (1.0, 1.0, 1.0),\n        ),\n"
        );
        text.insert_str(end, &entry);
        std::fs::write(path, text).unwrap();
    }

    /// The player prefab saves its Network Variables per player.
    fn flag_player(&self) {
        let path = self.0.join("prefabs/net_avatar.prefab");
        let text = std::fs::read_to_string(&path).unwrap();
        let anchor = "kind: \"Network Variables\",";
        let at = text.find(anchor).unwrap();
        let comp = "kind: \"Persistent State\",\n                    data: \"(transform:false,variables:true,per_player:true)\",\n                ),\n                (\n                    ";
        let mut out = text.clone();
        out.insert_str(at, comp);
        std::fs::write(path, out).unwrap();
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
    }
}

struct Client {
    net: NetSession,
    ops: Vec<NetWorldOp>,
    events: Vec<NetEvent>,
    messages: Vec<String>,
}

impl Client {
    fn join_as(target: &str, name: &str, identity: Identity, relay: &str) -> Self {
        let cfg = NetConfig { player_name: name.into(), discovery: false, identity, relay: relay.into(), ..Default::default() };
        Self { net: NetSession::join_with(target, cfg).unwrap(), ops: vec![], events: vec![], messages: vec![] }
    }

    fn join(addr: SocketAddr, name: &str, seed: u8) -> Self {
        Self::join_as(&addr.to_string(), name, Identity::from_seed([seed; 32]), "")
    }

    fn poll(&mut self) {
        self.net.pump();
        let v = self.net.view();
        self.ops.extend(v.world_ops);
        self.events.extend(v.events);
        self.messages.extend(v.messages.into_iter().map(|m| m.2));
    }

    fn disconnected(&self) -> Option<String> {
        match self.net.state() {
            ConnState::Disconnected(r) => Some(r.clone()),
            _ => None,
        }
    }

    fn joined(&self) -> bool {
        self.net.is_connected() && self.ops.iter().any(|op| matches!(op, NetWorldOp::Spawn { owner, .. } if *owner == self.net.local_peer()))
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

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The lever's (translation, "pulls" variable) on the server.
fn lever(server: &ServerHandle) -> (glam::Vec3, Option<NetValue>) {
    server
        .remote
        .call(|s| {
            let scene = s.host().scene();
            let o = scene.objects.iter().find(|o| o.name == "Lever").expect("lever object");
            let v = o.components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkVariables>()).and_then(|v| v.get("pulls").cloned());
            (o.translation, v)
        })
        .unwrap()
}

/// The "gold" variable of a connected player's object, by peer.
fn gold(server: &ServerHandle, peer: u64) -> Option<NetValue> {
    server
        .remote
        .call(move |s| {
            let obj = *s.host().net()?.world.player_obj.get(&peer)?;
            let scene = s.host().scene();
            let i = scene.index_of(ObjectId::from_raw(obj))?;
            scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkVariables>()).and_then(|v| v.get("gold").cloned())
        })
        .flatten()
}

fn set_gold(server: &ServerHandle, peer: u64, value: i64) {
    server
        .remote
        .call(move |s| {
            let obj = *s.host().net().unwrap().world.player_obj.get(&peer).unwrap();
            let i = s.host().scene().index_of(ObjectId::from_raw(obj)).unwrap();
            s.host_mut().scene_mut().objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>()).unwrap().set("gold", NetValue::Int(value));
        })
        .unwrap();
}

fn set_lever(server: &ServerHandle, at: glam::Vec3, pulls: i64) {
    server
        .remote
        .call(move |s| {
            let o = s.host_mut().scene_mut().objects.iter_mut().find(|o| o.name == "Lever").unwrap();
            o.translation = at;
            o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>()).unwrap().set("pulls", NetValue::Int(pulls));
        })
        .unwrap();
}

#[test]
fn world_objects_and_players_survive_a_restart() {
    let t0 = Instant::now();
    let project = Project::new();
    project.add_lever();
    project.flag_player();
    let at = glam::Vec3::new(3.0, 4.0, 5.0);

    // Run 1: change the lever, give a player some gold, stop (saves).
    let server = ServerHandle::spawn(project.config()).unwrap();
    assert_eq!(lever(&server), (glam::Vec3::ZERO, None), "a fresh start keeps the authored scene");
    let mut ann = Client::join(server.addr, "Ann", 7);
    wait_for(&mut [&mut ann], "Ann joined", |c| c[0].joined());
    set_lever(&server, at, 7);
    set_gold(&server, ann.net.local_peer(), 42);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(server.shutdown(Duration::from_secs(5)).unwrap(), ExitReason::Stopped);
    let world = std::fs::read_to_string(project.state().join("world.state")).unwrap();
    assert!(world.starts_with("ZESTSTATE 1 1 ") && world.contains("Lever") && world.contains("pulls"), "{world}");
    let players = std::fs::read_to_string(project.state().join("players.state")).unwrap();
    assert!(players.contains("Ann") && players.contains("gold") && players.contains("sessions: 1"), "{players}");

    // Run 2: everything is back, and Ann (same key) gets her gold on joining; Bob (another key) gets none.
    let server = ServerHandle::spawn(project.config()).unwrap();
    assert_eq!(lever(&server), (at, Some(NetValue::Int(7))), "scene object restored");
    let mut ann = Client::join(server.addr, "Ann", 7);
    let mut bob = Client::join(server.addr, "Bob", 8);
    wait_for(&mut [&mut ann, &mut bob], "both joined", |c| c[0].joined() && c[1].joined());
    wait_until("Ann's gold restored", || gold(&server, ann.net.local_peer()) == Some(NetValue::Int(42)));
    assert_eq!(gold(&server, bob.net.local_peer()), None);
    let late = Client::join(server.addr, "Late", 9);
    drop(late);
    // A newer lever state makes run 2's file a new generation (unchanged state is not rewritten).
    set_lever(&server, at, 8);
    assert_eq!(server.shutdown(Duration::from_secs(5)).unwrap(), ExitReason::Stopped);

    // Crash safety: the current file is damaged (a half-written disk), the backup generation carries on.
    let main = project.state().join("world.state");
    let bytes = std::fs::read(&main).unwrap();
    std::fs::write(&main, &bytes[..bytes.len() / 2]).unwrap();
    assert!(project.state().join("world.state.bak1").is_file(), "rotation kept the previous generation");
    let server = ServerHandle::spawn(project.config()).unwrap();
    assert_eq!(lever(&server), (at, Some(NetValue::Int(7))), "restored from the backup generation (run 1's file; run 2's was damaged)");
    assert!(std::fs::read_dir(project.state()).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().contains("world.state.corrupt-")), "the damaged file is kept for inspection");
    drop(server);

    // Everything damaged: a clean start with the authored scene, not a crash.
    for g in ["world.state", "world.state.bak1", "world.state.bak2", "world.state.bak3"] {
        let p = project.state().join(g);
        if p.is_file() {
            std::fs::write(p, b"ZESTSTATE 1 1 99 deadbeef\nnot the state").unwrap();
        }
    }
    let server = ServerHandle::spawn(project.config()).unwrap();
    assert_eq!(lever(&server), (glam::Vec3::ZERO, None));
    drop(server);
    eprintln!("persistence round trip + crash recovery over loopback: {:?}", t0.elapsed());
}

#[test]
fn state_is_saved_on_an_interval_and_by_the_save_command() {
    let project = Project::new();
    project.add_lever();
    let server = ServerHandle::spawn(ServerConfig { persist_interval: 0.4, ..project.config() }).unwrap();
    set_lever(&server, glam::Vec3::new(1.0, 2.0, 3.0), 1);
    let world = project.state().join("world.state");
    wait_until("the interval save", || std::fs::read_to_string(&world).is_ok_and(|t| t.contains("pulls")));
    // Unchanged state is not rewritten.
    let before = std::fs::metadata(&world).unwrap().modified().unwrap();
    std::thread::sleep(Duration::from_millis(1300));
    assert_eq!(std::fs::metadata(&world).unwrap().modified().unwrap(), before, "no change, no write");
    // The save command writes a change at once.
    set_lever(&server, glam::Vec3::new(9.0, 9.0, 9.0), 2);
    let out = server.remote.command(AdminCommand::Save).unwrap();
    assert!(out.starts_with("saved"), "{out}");
    assert!(std::fs::read_to_string(&world).unwrap().contains("9.0"));
    let stats = server.remote.command(AdminCommand::Stats).unwrap();
    assert!(stats.contains("persistence: 1 scene(s)"), "{stats}");
}

#[test]
fn a_project_without_persistent_content_writes_no_state() {
    let project = Project::new();
    let server = ServerHandle::spawn(project.config()).unwrap();
    let mut a = Client::join(server.addr, "A", 3);
    wait_for(&mut [&mut a], "joined", |c| c[0].joined());
    drop(server);
    assert!(!project.state().join("world.state").exists(), "nothing flagged, no world file");
    assert!(project.state().join("players.state").exists(), "a player record is kept");
    let off = ServerHandle::spawn(ServerConfig { persist: false, ..project.config() }).unwrap();
    let out = off.remote.command(AdminCommand::Save).unwrap();
    assert!(out.contains("persistence is off"), "{out}");
}

struct Keys {
    admin: SigningKey,
    moderator: SigningKey,
    observer: SigningKey,
}

fn keys() -> Keys {
    Keys { admin: SigningKey::from_bytes(&[11; 32]), moderator: SigningKey::from_bytes(&[12; 32]), observer: SigningKey::from_bytes(&[13; 32]) }
}

fn key_entry(name: &str, k: &SigningKey, role: AdminRole) -> AdminKey {
    AdminKey { name: name.into(), key: remote::hex(&k.verifying_key().to_bytes()), role }
}

fn admin_config(project: &Project, k: &Keys) -> ServerConfig {
    ServerConfig {
        admin_port: Some(0),
        admin_bind: "127.0.0.1".into(),
        admin_keys: vec![key_entry("ops", &k.admin, AdminRole::Admin), key_entry("mod", &k.moderator, AdminRole::Moderator), key_entry("eye", &k.observer, AdminRole::Observer)],
        ..project.config()
    }
}

fn admin_addr(server: &ServerHandle) -> (SocketAddr, [u8; 32]) {
    server.remote.call(|s| (s.admin_addr().unwrap(), s.server_key().unwrap())).unwrap()
}

fn run(server: &ServerHandle, key: &SigningKey, cmd: &str) -> remote::Reply {
    let (addr, spub) = admin_addr(server);
    remote::send_command(addr, &spub, key, cmd, Duration::from_secs(5)).unwrap_or_else(|e| panic!("{cmd}: {e}"))
}

#[test]
fn signed_admin_moderates_over_udp_and_bans_persist() {
    let t0 = Instant::now();
    let project = Project::new();
    let k = keys();
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    let mut ann = Client::join(server.addr, "Ann", 21);
    let mut bob = Client::join(server.addr, "Bob", 22);
    wait_for(&mut [&mut ann, &mut bob], "joined", |c| c[0].joined() && c[1].joined());

    // Read-only commands for every role; the observer cannot touch players.
    let status = run(&server, &k.observer, "status");
    assert_eq!(status.status, Status::Ok);
    assert!(status.text.contains("players 2/16"), "{}", status.text);
    let players = run(&server, &k.observer, "players");
    assert!(players.text.contains("Ann") && players.text.contains("Bob"), "{}", players.text);
    let denied = run(&server, &k.observer, &format!("kick {}", bob.net.local_peer()));
    assert_eq!(denied.status, Status::Denied, "{}", denied.text);
    assert!(run(&server, &k.moderator, "shutdown").status == Status::Denied);
    assert!(run(&server, &k.moderator, "reload").status == Status::Denied);

    // say reaches the players.
    assert_eq!(run(&server, &k.moderator, "say hello from ops").status, Status::Ok);
    wait_for(&mut [&mut ann, &mut bob], "say", |c| c.iter().all(|c| c.messages.iter().any(|m| m == "hello from ops")));

    // Ban Bob for an hour: he is kicked with the reason and cannot come back; Ann is untouched.
    let id = bob.net.local_peer();
    let ban = run(&server, &k.moderator, &format!("ban {id} 1h griefing the lobby"));
    assert_eq!(ban.status, Status::Ok, "{}", ban.text);
    wait_for(&mut [&mut ann, &mut bob], "Bob banned", |c| c[1].disconnected().is_some());
    assert!(bob.disconnected().unwrap().contains("banned: griefing the lobby"), "{:?}", bob.disconnected());
    assert!(ann.net.is_connected());
    let bans = run(&server, &k.observer, "bans");
    let bob_fp = fingerprint(&Identity::from_seed([22; 32]).public());
    assert!(bans.text.contains(&bob_fp) && bans.text.contains("griefing the lobby") && bans.text.contains("m left"), "{}", bans.text);
    let file = std::fs::read_to_string(project.state().join("bans.ron")).unwrap();
    assert!(file.contains(&bob_fp) && file.contains("griefing the lobby"), "persisted for the next start: {file}");
    let mut again = Client::join(server.addr, "Bob", 22);
    wait_for(&mut [&mut again], "refused", |c| c[0].disconnected().is_some());
    assert!(again.disconnected().unwrap().contains("banned from this server: griefing the lobby"), "{:?}", again.disconnected());

    // An offline permanent ban by fingerprint, then lift Bob's by a prefix of his fingerprint.
    let cy = fingerprint(&Identity::from_seed([23; 32]).public());
    assert_eq!(run(&server, &k.moderator, &format!("ban {cy} perm cheating")).status, Status::Ok);
    assert_eq!(run(&server, &k.moderator, "ban 99").status, Status::Error, "unknown player id");
    assert_eq!(run(&server, &k.moderator, "ban nonsense").status, Status::Error);
    let prefix: String = bob_fp.chars().filter(|c| *c != '-').take(10).collect();
    let un = run(&server, &k.moderator, &format!("unban {prefix}"));
    assert_eq!(un.status, Status::Ok, "{}", un.text);
    assert_eq!(run(&server, &k.moderator, "unban 0000000000").status, Status::Error, "no such ban");
    let mut back = Client::join(server.addr, "Bob", 22);
    wait_for(&mut [&mut back], "Bob back", |c| c[0].joined());
    let stats = run(&server, &k.observer, "stats");
    assert!(stats.text.contains("remote admin: 3 key(s)") && stats.text.contains("denied"), "{}", stats.text);

    // The console path (stdin) is the same command; the permanent ban survives a restart.
    let out = server.remote.command(AdminCommand::Bans).unwrap();
    assert!(out.contains(&cy) && out.contains("permanent"), "{out}");
    drop(server);
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    let out = server.remote.command(AdminCommand::Bans).unwrap();
    assert!(out.contains(&cy) && !out.contains(&bob_fp), "{out}");
    let mut c = Client::join(server.addr, "Cy", 23);
    wait_for(&mut [&mut c], "Cy refused after the restart", |c| c[0].disconnected().is_some());
    assert!(c.disconnected().unwrap().contains("cheating"));
    eprintln!("signed admin over loopback UDP (ban, kick, say, restart): {:?}", t0.elapsed());
}

#[test]
fn tampered_replayed_and_unknown_requests_do_nothing() {
    let project = Project::new();
    let k = keys();
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    let mut ann = Client::join(server.addr, "Ann", 31);
    wait_for(&mut [&mut ann], "joined", |c| c[0].joined());
    let (addr, spub) = admin_addr(&server);
    let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
    sock.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
    let mut buf = [0u8; 2048];

    // A genuine "say", sent twice (a replay): delivered once, the second is answered Stale.
    let packet = remote::encode_request(&k.moderator, &spub, remote::unix_ms(), 4242, "say once only");
    sock.send_to(&packet, addr).unwrap();
    let (n, _) = sock.recv_from(&mut buf).unwrap();
    assert_eq!(remote::verify_reply(&buf[..n], &spub, &k.moderator.verifying_key().to_bytes()).unwrap().status, Status::Ok);
    sock.send_to(&packet, addr).unwrap();
    let (n, _) = sock.recv_from(&mut buf).unwrap();
    assert_eq!(remote::verify_reply(&buf[..n], &spub, &k.moderator.verifying_key().to_bytes()).unwrap().status, Status::Stale);

    // Bit flips anywhere, an unlisted key, a request meant for another server: silence, and no effect.
    let mut bad = Vec::new();
    for at in [6, 40, 50, 56, packet.len() - 1] {
        let mut p = remote::encode_request(&k.admin, &spub, remote::unix_ms(), 100 + at as u64, "shutdown");
        let at = at.min(p.len() - 1);
        p[at] ^= 1;
        bad.push(p);
    }
    bad.push(remote::encode_request(&SigningKey::from_bytes(&[99; 32]), &spub, remote::unix_ms(), 7, "shutdown"));
    bad.push(remote::encode_request(&k.admin, &[1; 32], remote::unix_ms(), 8, "shutdown"));
    for p in &bad {
        sock.send_to(p, addr).unwrap();
        assert!(sock.recv_from(&mut buf).is_err(), "no answer to a forged request");
    }
    wait_for(&mut [&mut ann], "the one say", |c| c[0].messages.iter().any(|m| m == "once only"));
    std::thread::sleep(Duration::from_millis(150));
    ann.poll();
    assert_eq!(ann.messages.iter().filter(|m| *m == "once only").count(), 1, "replay did not run twice");
    assert!(server.remote.call(|s| s.exit_reason()).unwrap().is_none(), "no forged shutdown got through");
    let stats = server.remote.command(AdminCommand::Stats).unwrap();
    assert!(stats.contains("1 stale/replayed") && stats.contains("command(s) run"), "{stats}");
}

#[test]
fn shutdown_and_restart_commands_end_the_server_with_the_right_reason() {
    let project = Project::new();
    let k = keys();
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    let mut ann = Client::join(server.addr, "Ann", 41);
    wait_for(&mut [&mut ann], "joined", |c| c[0].joined());
    assert_eq!(run(&server, &k.admin, "restart").status, Status::Ok);
    wait_for(&mut [&mut ann], "notice", |c| c[0].disconnected().is_some());
    assert!(ann.disconnected().unwrap().contains("server restarting"), "{:?}", ann.disconnected());
    assert!(redirect_target(&ann.disconnected().unwrap()).is_none(), "no redirect unless one was set");
    assert_eq!(server.shutdown(Duration::from_secs(5)).unwrap(), ExitReason::Restart);
}

#[test]
fn reload_applies_live_settings_and_reports_what_needs_a_restart() {
    let project = Project::new();
    let k = keys();
    let file = project.0.join("server.ron");
    let write = |motd: &str, port: u16| {
        std::fs::write(&file, format!("(project: {:?}, port: {port}, lan_discovery: false, console: false, stats_interval: 0.0, persist_interval: 0.0, motd: {motd:?}, admin_port: 0, admin_bind: \"127.0.0.1\", admin_keys: [(name: \"ops\", key: {:?}, role: Admin)])", project.0.display().to_string(), remote::hex(&k.admin.verifying_key().to_bytes()))).unwrap();
    };
    write("first", 0);
    let cli = zestengine_server::parse_args(vec!["--config".into(), file.display().to_string()]).unwrap();
    let cfg = cli.resolve().unwrap();
    let server = ServerHandle::spawn(cfg).unwrap();
    server.remote.call(move |s| s.set_reload_source(cli)).unwrap();
    let (admin, spub) = admin_addr(&server);
    let game = server.addr;
    let ask = |cmd: &str| remote::send_command(admin, &spub, &k.admin, cmd, Duration::from_secs(5)).unwrap();
    write("second", 9999);
    let out = ask("reload");
    assert_eq!(out.status, Status::Ok, "{}", out.text);
    assert!(out.text.contains("motd") && out.text.contains("needs a restart: port"), "{}", out.text);
    let mut ann = Client::join_as(&game.to_string(), "Ann", Identity::from_seed([51; 32]), "");
    wait_for(&mut [&mut ann], "joined", |c| c[0].joined());
    wait_for(&mut [&mut ann], "new motd", |c| c[0].messages.iter().any(|m| m == "second"));
    // A broken file is refused and the running configuration stays.
    std::fs::write(&file, "(tick_rat: 5.0)").unwrap();
    let out = ask("reload");
    assert_eq!(out.status, Status::Error);
    assert!(out.text.contains("unchanged"), "{}", out.text);
}

fn start_relay(dir: DirConfig) -> (SocketAddr, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    let mut relay = Relay::bind("127.0.0.1:0", RelayConfig { directory: Some(dir), ..Default::default() }).unwrap();
    let addr = relay.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    (addr, stop, std::thread::spawn(move || relay.run(&flag).unwrap()))
}

#[test]
fn a_server_lists_itself_updates_its_player_count_and_delists_on_shutdown() {
    let t0 = Instant::now();
    let (relay, stop, thread) = start_relay(DirConfig::default());
    let project = Project::new();
    let cfg = ServerConfig {
        directory: relay.to_string(),
        directory_interval: 1.0,
        region: "eu".into(),
        game: "arena".into(),
        game_version: "0.3".into(),
        server_name: "Test Arena".into(),
        password: "pw".into(),
        ..project.config()
    };
    let server = ServerHandle::spawn(cfg).unwrap();
    let port = server.addr.port();
    let list = |f: &Filter| query_blocking(relay, f, Duration::from_secs(5)).unwrap();
    wait_until("listed", || !list(&Filter::default()).is_empty());
    let l = list(&Filter::default()).remove(0);
    assert_eq!((l.info.name.as_str(), l.info.game.as_str(), l.info.region.as_str(), l.info.version.as_str()), ("Test Arena", "arena", "eu", "0.3"));
    assert_eq!((l.info.players, l.info.max_players, l.info.port, l.info.password), (0, 16, port, true));
    assert_eq!(l.addr, SocketAddr::from(([127, 0, 0, 1], port)), "address from the announcement's source");
    assert!(l.info.map.ends_with("main.scene"), "{}", l.info.map);
    assert_eq!(l.key, server.remote.call(|s| s.server_key().unwrap()).unwrap(), "listed under the server's identity key");
    assert!(list(&Filter { region: "na".into(), ..Default::default() }).is_empty());
    assert_eq!(list(&Filter { game: "arena".into(), hide_full: true, ..Default::default() }).len(), 1);

    // The player count follows (the next heartbeat), then the listing disappears the moment the server stops.
    let mut ann = Client::join_as(&format!("127.0.0.1:{port}"), "Ann", Identity::from_seed([61; 32]), "");
    ann.net = NetSession::join_with(&format!("127.0.0.1:{port}"), NetConfig { player_name: "Ann".into(), discovery: false, secret: "pw".into(), identity: Identity::from_seed([61; 32]), ..Default::default() }).unwrap();
    wait_for(&mut [&mut ann], "joined", |c| c[0].joined());
    wait_until("player count listed", || list(&Filter::default()).first().is_some_and(|l| l.info.players == 1));
    let stats = server.remote.command(AdminCommand::Stats).unwrap();
    assert!(stats.contains("directory:") && stats.contains(&relay.to_string()), "{stats}");
    drop(server);
    assert!(list(&Filter::default()).is_empty(), "delisted by a signed goodbye, not by waiting for the TTL");
    stop.store(true, Ordering::Relaxed);
    let _ = thread.join();
    eprintln!("directory list/heartbeat/delist over loopback UDP: {:?}", t0.elapsed());
}

#[test]
fn a_dedicated_server_is_joinable_by_room_code_through_the_relay() {
    let (relay, stop, thread) = start_relay(DirConfig::default());
    let project = Project::new();
    let cfg = ServerConfig { relay: relay.to_string(), relay_room: "ARENA1".into(), directory: relay.to_string(), directory_interval: 1.0, ..project.config() };
    let server = ServerHandle::spawn(cfg).unwrap();
    wait_until("room open", || server.remote.call(|s| s.host().net().and_then(|n| n.room_code())).flatten().is_some());
    let code = server.remote.call(|s| s.host().net().and_then(|n| n.room_code())).flatten().unwrap();
    assert_eq!(code, "ARENA1");
    // The directory entry carries the room, so a browser can offer "join by code".
    wait_until("listed with the room", || query_blocking(relay, &Filter::default(), Duration::from_secs(5)).unwrap().first().is_some_and(|l| l.info.room == "ARENA1"));
    let mut ann = Client::join_as(&code, "Ann", Identity::from_seed([71; 32]), &relay.to_string());
    wait_for(&mut [&mut ann], "joined through the relay", |c| c[0].joined());
    assert_eq!(server.remote.call(|s| s.player_count()), Some(1));
    stop.store(true, Ordering::Relaxed);
    let _ = thread.join();
}

#[test]
fn shutdown_redirects_players_who_follow_it() {
    let project_a = Project::new();
    let project_b = Project::new();
    let k = keys();
    let b = ServerHandle::spawn(project_b.config()).unwrap();
    let a = ServerHandle::spawn(admin_config(&project_a, &k)).unwrap();
    let mut ann = Client::join(a.addr, "Ann", 81);
    wait_for(&mut [&mut ann], "joined A", |c| c[0].joined());
    let target = b.addr.to_string();
    assert_eq!(run(&a, &k.admin, &format!("shutdown {target}")).status, Status::Ok);
    wait_for(&mut [&mut ann], "notice", |c| c[0].disconnected().is_some());
    let reason = ann.disconnected().unwrap();
    assert!(reason.contains("server shutting down (admin)"), "{reason}");
    assert_eq!(redirect_target(&reason), Some(target.clone()));
    // The player's game follows it: a new session to B with the same settings.
    let mut net = Some(std::mem::replace(&mut ann.net, NetSession::join_with(&target, NetConfig::default()).unwrap()));
    let scene = zestengine_engine::server::LoadedScene::empty();
    // `ann.net` above was a placeholder; the real old session is `net`.
    let followed = follow_redirect(&mut net, &scene);
    // A client of a dedicated server follows; the old session was dedicated.
    assert_eq!(followed.as_deref(), Some(target.as_str()));
    let mut joined = Client { net: net.take().unwrap(), ops: vec![], events: vec![], messages: vec![] };
    wait_for(&mut [&mut joined], "joined B", |c| c[0].net.is_connected());
    assert_eq!(b.remote.call(|s| s.player_count()), Some(1));
}

// ------------------------------------------------------------------ NET-6 / SEC follow-ups

const CRATE_PREFAB: &str = "(\n    entries: [\n        (\n            id: \"c0a7e000-0000-4000-8000-000000000001\",\n            name: \"Crate\",\n            source: Empty,\n            enabled: true,\n            static_geometry: false,\n            lightmap_scale: 1.0,\n            layer: 0,\n            material: File(\"materials/default.material\"),\n            parent: None,\n            components: [\n                (kind: \"Network Variables\", data: \"(vars:[],anyone_can_write:false)\"),\n                (kind: \"Persistent State\", data: \"(transform:true,variables:true,per_player:false)\"),\n            ],\n            translation: (0.0, 0.0, 0.0),\n            rotation: (0.0, 0.0, 0.0, 1.0),\n            scale: (1.0, 1.0, 1.0),\n        ),\n    ],\n)\n";

fn crates(server: &ServerHandle) -> Vec<(glam::Vec3, Option<NetValue>)> {
    server
        .remote
        .call(|s| {
            s.host()
                .scene()
                .objects
                .iter()
                .filter(|o| o.name == "Crate")
                .map(|o| (o.translation, o.components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkVariables>()).and_then(|v| v.get("fill").cloned())))
                .collect()
        })
        .unwrap()
}

#[test]
fn runtime_spawned_persistent_objects_survive_a_restart() {
    let project = Project::new();
    std::fs::write(project.0.join("prefabs/crate.prefab"), CRATE_PREFAB).unwrap();
    let server = ServerHandle::spawn(project.config()).unwrap();
    assert!(crates(&server).is_empty());
    let at = glam::Vec3::new(3.0, 0.0, 4.0);
    server
        .remote
        .call(move |s| {
            s.host_mut().net_mut().unwrap().spawn_unowned(zestengine_core::NetSpawnSource::Prefab("prefabs/crate.prefab".into()), at, glam::Quat::IDENTITY);
        })
        .unwrap();
    wait_until("the crate is in the scene", || crates(&server).len() == 1);
    server
        .remote
        .call(|s| {
            let o = s.host_mut().scene_mut().objects.iter_mut().find(|o| o.name == "Crate").unwrap();
            o.translation = glam::Vec3::new(6.0, 0.0, 8.0);
            o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>()).unwrap().set("fill", NetValue::Int(5));
        })
        .unwrap();
    assert_eq!(server.shutdown(Duration::from_secs(5)).unwrap(), ExitReason::Stopped);
    let world = std::fs::read_to_string(project.state().join("world.state")).unwrap();
    assert!(world.contains("spawned") && world.contains("prefabs/crate.prefab") && world.contains("fill"), "{world}");

    // Restart: one crate again, where it was, with its variable; a second restart does not duplicate it.
    for _ in 0..2 {
        let server = ServerHandle::spawn(project.config()).unwrap();
        wait_until("the crate came back", || crates(&server).len() == 1 && crates(&server)[0].1.is_some());
        assert_eq!(crates(&server), vec![(glam::Vec3::new(6.0, 0.0, 8.0), Some(NetValue::Int(5)))]);
        // A late joiner is told about it like any spawn.
        let mut ann = Client::join(server.addr, "Ann", 5);
        wait_for(&mut [&mut ann], "joined", |c| c[0].joined());
        wait_until("Ann sees the crate", || {
            ann.poll();
            ann.net.world.spawns.iter().any(|s| matches!(&s.source, zestengine_core::NetSpawnSource::Prefab(p) if p == "prefabs/crate.prefab"))
        });
        assert_eq!(server.shutdown(Duration::from_secs(5)).unwrap(), ExitReason::Stopped);
    }
}

#[test]
fn admin_keys_rotate_without_a_restart_and_survive_one() {
    let project = Project::new();
    let k = keys();
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    let fresh = SigningKey::from_bytes(&[21; 32]);
    let fresh_hex = remote::hex(&fresh.verifying_key().to_bytes());
    assert_eq!(run(&server, &k.moderator, "keys").status, Status::Denied, "key management is for Admin keys");
    assert_eq!(run(&server, &k.observer, &format!("key add x {fresh_hex} admin")).status, Status::Denied);
    let out = run(&server, &k.admin, &format!("key add ops2 {fresh_hex} admin"));
    assert_eq!(out.status, Status::Ok, "{}", out.text);
    assert_eq!(run(&server, &fresh, "status").status, Status::Ok, "the new key signs at once");
    let list = run(&server, &k.admin, "keys").text;
    assert!(list.contains("4 admin key(s)") && list.contains("ops2") && list.contains("(runtime)") && !list.contains(&fresh_hex), "{list}");
    assert_eq!(run(&server, &k.admin, &format!("key add dup {fresh_hex} admin")).status, Status::Error, "no duplicates");
    // Retire the old admin key: it stops being answered, the others keep working.
    let out = run(&server, &fresh, "key remove ops");
    assert_eq!(out.status, Status::Ok, "{}", out.text);
    let (addr, spub) = admin_addr(&server);
    assert!(remote::send_command(addr, &spub, &k.admin, "status", Duration::from_millis(1500)).is_err(), "the retired key gets no answer");
    assert_eq!(run(&server, &k.moderator, "players").status, Status::Ok);
    let out = run(&server, &fresh, "key remove ops2");
    assert_eq!((out.status, out.text.contains("last Admin")), (Status::Error, true), "{}", out.text);
    assert!(project.state().join("admin_keys.ron").is_file());
    drop(server);

    // After a restart the overlay is applied over the unchanged server.ron keys.
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    assert_eq!(run(&server, &fresh, "status").status, Status::Ok);
    let (addr, spub) = admin_addr(&server);
    assert!(remote::send_command(addr, &spub, &k.admin, "status", Duration::from_millis(1500)).is_err(), "still retired");
}

#[test]
fn player_reports_reach_the_moderators() {
    let project = Project::new();
    let k = keys();
    let server = ServerHandle::spawn(admin_config(&project, &k)).unwrap();
    assert_eq!(run(&server, &k.moderator, "reports").text, "no reports");
    let mut ann = Client::join(server.addr, "Ann", 31);
    let mut mal = Client::join(server.addr, "Mallory", 32);
    wait_for(&mut [&mut ann, &mut mal], "both joined", |c| c[0].joined() && c[1].joined());
    let target = mal.net.local_peer();
    ann.net.report_player(target, "wall hacks");
    wait_until("the report is queued", || {
        ann.poll();
        mal.poll();
        run(&server, &k.moderator, "reports").text.contains("wall hacks")
    });
    let text = run(&server, &k.moderator, "reports").text;
    assert!(text.contains("Ann") && text.contains("Mallory") && text.contains("1 report(s)"), "{text}");
    let log = std::fs::read_to_string(project.state().join("reports.log")).unwrap();
    assert!(log.contains("wall hacks") && log.contains(&mal.net.peer_fingerprint(target).unwrap_or_default()), "{log}");
    assert_eq!(run(&server, &k.observer, "reports").status, Status::Denied, "reports are for moderators");
    assert!(run(&server, &k.moderator, "reports clear").text.contains("cleared 1"));
    assert_eq!(run(&server, &k.moderator, "reports").text, "no reports");
}

#[test]
fn the_server_browser_lists_pings_filters_and_picks_the_join_address() {
    use zestengine_engine::server_browser::{BrowserFilter, BrowserStatus, ServerBrowser, SortKey};
    let (relay, stop, thread) = start_relay(DirConfig::default());
    let project = Project::new();
    let a = ServerHandle::spawn(ServerConfig { directory: relay.to_string(), directory_interval: 1.0, region: "eu".into(), game: "arena".into(), server_name: "Alpha Arena".into(), ..project.config() }).unwrap();
    let project_b = Project::new();
    let b = ServerHandle::spawn(ServerConfig { directory: relay.to_string(), directory_interval: 1.0, region: "na".into(), game: "arena".into(), server_name: "Beta Brawl".into(), password: "pw".into(), ..project_b.config() }).unwrap();
    wait_until("both listed", || query_blocking(relay, &Filter::default(), Duration::from_secs(5)).unwrap().len() == 2);

    let mut sb = ServerBrowser::new(&relay.to_string());
    assert_eq!(*sb.status(), BrowserStatus::Idle);
    sb.refresh();
    let start = Instant::now();
    while sb.busy() && start.elapsed() < Duration::from_secs(15) {
        sb.poll();
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(*sb.status(), BrowserStatus::Ready, "{:?}", sb.status());
    let rows: Vec<_> = sb.rows().cloned().collect();
    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert!(r.ping_ms.is_some_and(|p| p < 1000.0), "pinged: {r:?}");
        assert_eq!(r.compatible, Some(true), "the live protocol answer matches this build");
    }
    // Sorted by ping first; by name when asked.
    sb.sort_by(SortKey::Name);
    sb.poll();
    sb.sort_by(SortKey::Name);
    let names: Vec<String> = sb.rows().map(|r| r.name().to_owned()).collect();
    assert!(names.len() == 2, "{names:?}");
    // The join address is the server's own address and game port.
    let alpha = rows.iter().find(|r| r.name() == "Alpha Arena").unwrap();
    assert_eq!(sb.join_target(alpha).unwrap(), format!("127.0.0.1:{}", a.addr.port()));
    assert!(rows.iter().find(|r| r.name() == "Beta Brawl").unwrap().listing.info.password);
    // Filters go to the directory: region and name.
    sb.filter = BrowserFilter { region: "na".into(), ..Default::default() };
    sb.refresh();
    let start = Instant::now();
    while sb.busy() && start.elapsed() < Duration::from_secs(15) {
        sb.poll();
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(sb.rows().map(|r| r.name().to_owned()).collect::<Vec<_>>(), ["Beta Brawl"]);
    // A real join with the address the browser produced.
    let target = sb.join_target(sb.rows().next().unwrap()).unwrap();
    let mut ann = Client::join_as(&target, "Ann", Identity::from_seed([41; 32]), "");
    ann.net = NetSession::join_with(&target, NetConfig { player_name: "Ann".into(), discovery: false, secret: "pw".into(), identity: Identity::from_seed([41; 32]), ..Default::default() }).unwrap();
    wait_for(&mut [&mut ann], "joined from the browser's address", |c| c[0].joined());
    drop((a, b));
    stop.store(true, Ordering::Relaxed);
    let _ = thread.join();
}
