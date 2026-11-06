//! "Play as Server" (NET-3): run the project's scene as a local dedicated server on a background
//! thread of this process, so the editor can join it like a player would and see exactly what a
//! player sees (Server-role content stripped from the player, Client-role content stripped from the
//! server).
//!
//! It is the lean version of the `zestengine-server` binary: the same [`HeadlessHost`] with
//! `net_side: Server`, a [`Topology::Dedicated`] session on loopback, strict validation, no
//! persistence, admin port or directory. The scene is read from disk, so the editor saves first.
//! Components of the project's own plugins are registered through the `register` closure; the
//! editor passes the registration functions of the plugins it loaded (they are plain `fn`
//! pointers into libraries that are never unloaded, so they are safe to call on this thread).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::channel;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use zestengine_core::{ComponentRegistry, NetRole, NetworkManager, Topology};

use crate::net::{Identity, NetConfig, NetSession};
use crate::server::{HeadlessConfig, HeadlessHost};

/// Ticks per second of the local server.
pub const TICK_HZ: f32 = 30.0;

/// A running local server. Dropping it stops the thread.
pub struct LocalServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    ticks: Arc<AtomicU64>,
    players: Arc<AtomicU32>,
    thread: Option<JoinHandle<()>>,
}

fn manager(host: &HeadlessHost) -> Option<NetworkManager> {
    host.scene().objects.iter().find_map(|o| o.components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkManager>().cloned()))
}

impl LocalServer {
    /// Load `scene` (project-relative; `None` = the project's boot scene) of the project at
    /// `project_root` and start hosting it on a free loopback port. Errors (no scene, a scene
    /// that does not load) come back here, not on the thread.
    pub fn start(project_root: PathBuf, scene: Option<String>, register: impl FnOnce(&mut ComponentRegistry) + Send + 'static) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let (ticks, players) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU32::new(0)));
        let (tx, rx) = channel::<Result<SocketAddr, String>>();
        let (flag, t, p) = (stop.clone(), ticks.clone(), players.clone());
        let thread = std::thread::Builder::new()
            .name("zestengine-local-server".into())
            .spawn(move || {
                let mut host = match HeadlessHost::open(HeadlessConfig { assets_root: project_root, scene, physics: None, net_side: NetRole::Server }, register) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                let mut cfg = manager(&host).as_ref().map(NetConfig::from_manager).unwrap_or_default();
                cfg.player_name = "Local server".into();
                cfg.session_name = "Local server".into();
                cfg.discovery = false;
                cfg.relay = String::new();
                cfg.secret = String::new();
                cfg.migration = false;
                cfg.topology = Topology::Dedicated;
                cfg.identity = Identity::generate();
                let net = match NetSession::host_with(0, cfg) {
                    Ok(n) => n,
                    Err(e) => {
                        let _ = tx.send(Err(format!("binding a loopback port: {e}")));
                        return;
                    }
                };
                let port = net.local_addr().map_or(0, |a| a.port());
                host.set_net(net);
                // The Network Manager of the scene may ask for a hosting player; a dedicated server has none.
                for o in &mut host.scene_mut().objects {
                    for c in &mut o.components {
                        if let Some(m) = c.as_any_mut().downcast_mut::<NetworkManager>() {
                            m.topology = Topology::Dedicated;
                        }
                    }
                }
                let _ = tx.send(Ok(SocketAddr::from(([127, 0, 0, 1], port))));
                let step = Duration::from_secs_f32(1.0 / TICK_HZ);
                let mut next = Instant::now();
                while !flag.load(Ordering::Relaxed) {
                    host.tick(1.0 / TICK_HZ);
                    t.store(host.ticks(), Ordering::Relaxed);
                    p.store(host.net().map_or(0, |n| n.peer_count().saturating_sub(1)) as u32, Ordering::Relaxed);
                    if host.quit_requested() {
                        host.clear_quit();
                    }
                    next += step;
                    let now = Instant::now();
                    match next.checked_duration_since(now) {
                        Some(wait) => std::thread::sleep(wait),
                        None => next = now,
                    }
                }
                if let Some(n) = host.net_mut() {
                    n.close_session();
                }
            })
            .context("starting the local server thread")?;
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(Ok(addr)) => Ok(Self { addr, stop, ticks, players, thread: Some(thread) }),
            Ok(Err(e)) => {
                let _ = thread.join();
                bail!("the local server did not start: {e}")
            }
            Err(_) => {
                stop.store(true, Ordering::Relaxed);
                bail!("the local server did not start in time (scene loading)")
            }
        }
    }

    /// Where to join: `127.0.0.1:<port>`.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Ticks run so far.
    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::Relaxed)
    }

    /// Players connected (the server itself is not counted).
    pub fn players(&self) -> u32 {
        self.players.load(Ordering::Relaxed)
    }

    /// Stop and wait for the thread.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for LocalServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_server_hosts_the_scene_and_a_player_joins_it() {
        let parent = std::env::temp_dir().join(format!("zestengine-local-server-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();
        let root = crate::bundle::scaffold_project(&parent, "vr_social", "vr_social").unwrap();
        let server = LocalServer::start(root.clone(), None, |_| {}).expect("starts on the boot scene");
        assert_ne!(server.addr().port(), 0);
        let cfg = NetConfig { player_name: "Editor".into(), discovery: false, identity: Identity::generate(), ..Default::default() };
        let mut player = NetSession::join_with(&server.addr().to_string(), cfg).unwrap();
        let start = Instant::now();
        while !player.is_connected() && start.elapsed() < Duration::from_secs(10) {
            player.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(player.is_connected(), "{:?}", player.state());
        assert_eq!(player.topology(), Topology::Dedicated, "it is a dedicated session: the player pins the key and gets the strict rules");
        assert_eq!(crate::server::session_role(Some(&player)), NetRole::Client);
        let start = Instant::now();
        while server.players() == 0 && start.elapsed() < Duration::from_secs(5) {
            player.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(server.players(), 1);
        assert!(server.ticks() > 0);
        drop(player);
        server.stop();
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn a_missing_scene_is_an_error_not_a_hang() {
        let parent = std::env::temp_dir().join(format!("zestengine-local-server-bad-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();
        let root = crate::bundle::scaffold_project(&parent, "vr_social", "vr_social").unwrap();
        let err = LocalServer::start(root, Some("scenes/nope.scene".into()), |_| {}).err().expect("fails").to_string();
        assert!(err.contains("did not start") && err.contains("nope.scene"), "{err}");
        let _ = std::fs::remove_dir_all(&parent);
    }
}
