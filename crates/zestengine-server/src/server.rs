//! The server: a headless game host plus a dedicated hosting session, paced by
//! the fixed tick loop, with join tracking, stats, admin commands and a graceful
//! shutdown.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use zestengine_core::{ComponentRegistry, NetworkManager, Topology};
use zestengine_engine::net::{ConnState, Identity, NetConfig, NetSession, short_fingerprint};
use zestengine_engine::server::{HeadlessConfig, HeadlessHost, LoadedScene};

mod services;

use crate::admin::{AdminCommand, HELP};
use crate::config::{DEFAULT_PORT, ServerConfig};
use crate::signals;
use crate::tick::{TickBody, run_fixed};
use services::Services;
pub use services::RESTART_EXIT_CODE;

/// Why the run loop ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    /// SIGTERM / SIGINT / Ctrl+C.
    Signal,
    /// [`Remote::stop`] (tests, embedding).
    Stopped,
    /// The `quit` console command.
    AdminQuit,
    /// A component asked to quit and `restart_on_scene_end` is off.
    SceneEnded,
    /// The `restart` command: exit code [`RESTART_EXIT_CODE`] so a supervisor starts it again.
    Restart,
}

impl ExitReason {
    /// The text players are shown.
    fn notice(self) -> &'static str {
        match self {
            Self::Signal | Self::Stopped => "server shutting down",
            Self::AdminQuit => "server shutting down (admin)",
            Self::Restart => "server restarting",
            Self::SceneEnded => "the game ended",
        }
    }
}

type Call = Box<dyn FnOnce(&mut Server) + Send>;

enum Request {
    Command(AdminCommand, Sender<String>),
    Call(Call),
}

/// The receiving half of a [`Remote`]; owned by the run loop.
pub struct Control {
    stop: Arc<AtomicBool>,
    rx: Receiver<Request>,
    watch_signals: bool,
}

/// Talk to a running server from another thread: stop it, run console commands,
/// or run a closure on the server thread (tests).
#[derive(Clone)]
pub struct Remote {
    stop: Arc<AtomicBool>,
    tx: Sender<Request>,
}

/// A linked pair. `watch_signals`: also stop on SIGTERM / SIGINT (the binary).
pub fn control(watch_signals: bool) -> (Control, Remote) {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    (Control { stop: stop.clone(), rx, watch_signals }, Remote { stop, tx })
}

impl Remote {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Run a console command and return its output (None if the server is gone or stuck).
    pub fn command(&self, cmd: AdminCommand) -> Option<String> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Request::Command(cmd, tx)).ok()?;
        rx.recv_timeout(Duration::from_secs(10)).ok()
    }

    /// Run `f` on the server thread between ticks.
    pub fn call<R: Send + 'static>(&self, f: impl FnOnce(&mut Server) -> R + Send + 'static) -> Option<R> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Request::Call(Box::new(move |s| drop(tx.send(f(s)))))).ok()?;
        rx.recv_timeout(Duration::from_secs(10)).ok()
    }
}

/// Counters for one stats window.
#[derive(Clone, Copy, Debug, Default)]
pub struct StatsWindow {
    pub ticks: u32,
    sum_ms: f64,
    pub max_ms: f64,
}

impl StatsWindow {
    pub fn record(&mut self, ms: f64) {
        self.ticks += 1;
        self.sum_ms += ms;
        self.max_ms = self.max_ms.max(ms);
    }

    pub fn mean_ms(&self) -> f64 {
        if self.ticks == 0 { 0.0 } else { self.sum_ms / f64::from(self.ticks) }
    }
}

fn manager(scene: &LoadedScene) -> Option<&NetworkManager> {
    scene.objects.iter().find_map(|o| o.components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkManager>()))
}

/// A running dedicated server (not yet looping; see [`Server::run`]).
pub struct Server {
    cfg: ServerConfig,
    host: HeadlessHost,
    port: u16,
    members: BTreeMap<u64, String>,
    window: StatsWindow,
    last_window: StatsWindow,
    window_start: Instant,
    window_bytes: (u64, u64),
    started: Instant,
    exit: Option<ExitReason>,
    last_overload_warn: Option<Instant>,
    overloaded_since_warn: u64,
    /// NET-6 services: persistence, remote admin, directory.
    svc: Services,
}

impl Server {
    /// Validate the config, load the scene and start hosting (no components beyond the built-ins).
    pub fn open(cfg: ServerConfig) -> Result<Self> {
        Self::open_with(cfg, |_| {})
    }

    /// [`open`](Self::open) with the game's own component types registered.
    pub fn open_with(mut cfg: ServerConfig, register: impl FnOnce(&mut ComponentRegistry)) -> Result<Self> {
        cfg.validate()?;
        if let Some(pak) = cfg.unpack_pak()? {
            tracing::info!("assets unpacked from {} into {}", pak.display(), cfg.project);
        }
        let root = cfg.project_dir();
        // Plugins first: a bad one refuses the start before any scene work, with its own message.
        let plugins = zestengine_engine::RuntimePlugins::load_all(&cfg.plugin_paths()).context("loading the plugins listed in `plugins`")?;
        if !plugins.names().is_empty() {
            tracing::info!("component plugins: {}", plugins.names().join(", "));
        }
        let mut host = HeadlessHost::open(
            HeadlessConfig { assets_root: root.clone(), scene: cfg.scene.clone(), physics: None, net_side: zestengine_core::NetRole::Server },
            |r: &mut ComponentRegistry| {
                plugins.register_all(r);
                register(r);
            },
        )
            .with_context(|| format!("loading the scene from {}", root.display()))?;
        tracing::info!("scene {} loaded: {} objects", host.scene_path(), host.scene().objects.len());

        let mgr = manager(host.scene()).cloned();
        match &mgr {
            None => tracing::warn!("the scene has no Network Manager: players can join but no player objects are spawned"),
            Some(m) if m.player_prefab.is_empty() => tracing::warn!("the Network Manager has no player prefab: players join without an avatar"),
            Some(_) => {}
        }
        let port = cfg.port.or(mgr.as_ref().map(|m| m.port)).unwrap_or(DEFAULT_PORT);

        // The scene's own settings first (relay, accounts, allow/ban lists ...), then the server's.
        let mut net_cfg = mgr.as_ref().map(NetConfig::from_manager).unwrap_or_default();
        net_cfg.player_name = cfg.server_name.clone();
        net_cfg.session_name = cfg.server_name.clone();
        // The server holds peer 1 itself; `max_players` counts players only.
        net_cfg.max_players = cfg.max_players + 1;
        net_cfg.discovery = cfg.lan_discovery;
        if !cfg.password.is_empty() {
            net_cfg.secret = cfg.password.clone();
        }
        // A dedicated session ends with its server; there is no host to migrate to.
        // Clients pin this identity key on first join and the strict validation
        // rules apply to every client.
        net_cfg.migration = false;
        net_cfg.topology = Topology::Dedicated;
        let identity = server_identity(&cfg);
        net_cfg.identity = identity.clone();
        if let Some(v) = &cfg.validation {
            net_cfg.validation = v.clone();
        }
        if let Some(i) = &cfg.interest {
            net_cfg.interest = i.clone();
        }
        if let Some(a) = &cfg.anticheat {
            net_cfg.anticheat = a.clone();
        }
        net_cfg.ban_file = Some(cfg.ban_path());
        services::apply_relay(&cfg, &mut net_cfg);

        let net = NetSession::host_with(port, net_cfg).with_context(|| format!("binding UDP port {port}"))?;
        let bound = net.local_addr().map_or(port, |a| a.port());
        host.set_net(net);
        let mut s = Self {
            cfg,
            host,
            port: bound,
            members: BTreeMap::new(),
            window: StatsWindow::default(),
            last_window: StatsWindow::default(),
            window_start: Instant::now(),
            window_bytes: (0, 0),
            started: Instant::now(),
            exit: None,
            last_overload_warn: None,
            overloaded_since_warn: 0,
            svc: Services::with_identity(identity),
        };
        s.force_dedicated();
        s.open_services()?;
        tracing::info!(
            "hosting {:?} on UDP {} ({} Hz, up to {} players, {}, LAN discovery {})",
            s.cfg.server_name,
            bound,
            s.cfg.tick_rate,
            s.cfg.max_players,
            if s.cfg.password.is_empty() && s.session_open() { "open" } else { "password protected" },
            if s.cfg.lan_discovery { "on" } else { "off" },
        );
        Ok(s)
    }

    /// Whether anyone can join without a secret (the password, or the scene's own).
    fn session_open(&self) -> bool {
        manager(self.host.scene()).is_none_or(|m| m.secret.is_empty())
    }

    /// The Network Manager says "dedicated": the host gets no player of its own.
    /// Re-applied after scene switches, whose manager may say otherwise.
    fn force_dedicated(&mut self) {
        for o in &mut self.host.scene_mut().objects {
            for c in &mut o.components {
                if let Some(m) = c.as_any_mut().downcast_mut::<NetworkManager>() {
                    m.topology = Topology::Dedicated;
                }
            }
        }
    }

    pub fn config(&self) -> &ServerConfig {
        &self.cfg
    }

    pub fn host(&self) -> &HeadlessHost {
        &self.host
    }

    pub fn host_mut(&mut self) -> &mut HeadlessHost {
        &mut self.host
    }

    /// The UDP port the session is bound to.
    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.host.net().and_then(|n| n.local_addr())
    }

    /// Players connected right now (the server's own peer excluded).
    pub fn player_count(&self) -> usize {
        self.members.len()
    }

    pub fn ticks(&self) -> u64 {
        self.host.ticks()
    }

    pub fn exit_reason(&self) -> Option<ExitReason> {
        self.exit
    }

    /// One fixed step: networking, gameplay, physics, replication; then
    /// join/leave tracking, stats and the quit check.
    pub fn tick(&mut self, dt: f32) {
        let t0 = Instant::now();
        self.host.tick(dt);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.window.record(ms);
        self.svc.health.window.record(ms);
        if self.host.ticks().is_multiple_of(30) {
            self.force_dedicated();
        }
        self.track_players();
        if self.host.quit_requested() {
            self.host.clear_quit();
            if self.cfg.restart_on_scene_end {
                let scene = self.host.scene_path().to_string();
                tracing::info!("the scene asked to end: restarting {scene}");
                if let Some(net) = self.host.net_mut() {
                    net.load_scene(&scene);
                }
            } else {
                tracing::info!("the scene asked to quit");
                self.exit.get_or_insert(ExitReason::SceneEnded);
            }
        }
        self.maybe_log_stats();
        self.tick_services();
    }

    /// Log joins and leaves, and greet joiners with the motd.
    fn track_players(&mut self) {
        let Some(net) = self.host.net() else { return };
        let now: BTreeMap<u64, (String, String)> =
            net.peer_infos().into_iter().filter(|p| !p.is_host).map(|p| (p.id, (p.name, p.fingerprint))).collect();
        let joined: Vec<u64> = now.keys().filter(|id| !self.members.contains_key(id)).copied().collect();
        let left: Vec<u64> = self.members.keys().filter(|id| !now.contains_key(id)).copied().collect();
        for id in left {
            let name = self.members.remove(&id).unwrap_or_default();
            tracing::info!("player left: {name} (id {id}), {}/{} players", now.len(), self.cfg.max_players);
        }
        for id in joined {
            let (name, fp) = &now[&id];
            self.members.insert(id, name.clone());
            tracing::info!("player joined: {name} (id {id}, {}), {}/{} players", short_fingerprint(fp), now.len(), self.cfg.max_players);
            if !self.cfg.motd.is_empty() && let Some(net) = self.host.net_mut() {
                net.send_message(Some(id), &self.cfg.motd);
            }
        }
    }

    fn net_totals(&self) -> (u64, u64) {
        self.host.net().map_or((0, 0), |n| (n.stats.bytes_in, n.stats.bytes_out))
    }

    fn maybe_log_stats(&mut self) {
        if self.cfg.stats_interval <= 0.0 || self.window_start.elapsed().as_secs_f32() < self.cfg.stats_interval {
            return;
        }
        let secs = self.window_start.elapsed().as_secs_f64();
        let (bin, bout) = self.net_totals();
        let (din, dout) = (bin - self.window_bytes.0, bout - self.window_bytes.1);
        let w = self.window;
        tracing::info!(
            "stats: {} ticks ({:.1}/s), tick mean {:.2} ms max {:.2} ms, players {}/{}, net in {} out {} ({}/s in, {}/s out), violations {} (muted {}){}",
            w.ticks,
            f64::from(w.ticks) / secs,
            w.mean_ms(),
            w.max_ms,
            self.members.len(),
            self.cfg.max_players,
            human_bytes(din),
            human_bytes(dout),
            human_bytes((din as f64 / secs) as u64),
            human_bytes((dout as f64 / secs) as u64),
            self.host.net().map_or(0, |n| n.stats.violations),
            self.host.net().map_or(0, |n| n.stats.muted),
            self.host.net().filter(|n| n.interest_config().enabled).map_or(String::new(), |n| format!(
                ", interest {:.0} relevant/player, {} held by budget",
                n.stats.interest_relevant_avg, n.stats.interest_dropped
            )),
        );
        self.last_window = w;
        self.window = StatsWindow::default();
        self.window_start = Instant::now();
        self.window_bytes = (bin, bout);
    }

    /// Run a console command; the returned text is what the operator sees.
    pub fn exec(&mut self, cmd: AdminCommand) -> String {
        let cmd = match self.exec_service(cmd) {
            Ok(out) => return out,
            Err(cmd) => cmd,
        };
        match cmd {
            AdminCommand::Help => HELP.to_string(),
            AdminCommand::Status => {
                let (bin, bout) = self.net_totals();
                let w = if self.last_window.ticks > 0 { self.last_window } else { self.window };
                format!(
                    "{:?} up {}s: scene {} ({} objects), UDP {}, {} Hz (tick mean {:.2} ms max {:.2} ms), players {}/{}, {} ticks, net in {} out {}",
                    self.cfg.server_name,
                    self.started.elapsed().as_secs(),
                    self.host.scene_path(),
                    self.host.scene().objects.len(),
                    self.port,
                    self.cfg.tick_rate,
                    w.mean_ms(),
                    w.max_ms,
                    self.members.len(),
                    self.cfg.max_players,
                    self.host.ticks(),
                    human_bytes(bin),
                    human_bytes(bout),
                )
            }
            AdminCommand::Players => {
                let Some(net) = self.host.net() else { return "no session".into() };
                let rows: Vec<String> = net
                    .peer_infos()
                    .into_iter()
                    .filter(|p| !p.is_host)
                    .map(|p| format!("  {:>3}  {:<24} {:>5.0} ms  {}", p.id, p.name, p.rtt_ms, short_fingerprint(&p.fingerprint)))
                    .collect();
                if rows.is_empty() { "no players".into() } else { format!("{} player(s):\n{}", rows.len(), rows.join("\n")) }
            }
            AdminCommand::Interest => {
                let Some(net) = self.host.net() else { return "no session".into() };
                if !net.interest_config().enabled {
                    return "interest management is off (server.ron: interest: (enabled: true))".into();
                }
                let rows: Vec<String> = net
                    .interest_stats()
                    .into_iter()
                    .map(|c| {
                        let name = self.members.get(&c.peer).cloned().unwrap_or_default();
                        format!(
                            "  {:>3}  {:<20} {:>5} relevant  {:>8}/s  {:>4} waiting  {} held by budget",
                            c.peer,
                            name,
                            c.relevant,
                            human_bytes(c.rate_out as u64),
                            c.pending,
                            c.dropped_budget
                        )
                    })
                    .collect();
                let st = net.stats;
                format!(
                    "{} object(s) tracked, {:.0} relevant per player on average, {} update(s) merged by distance tiers\n{}",
                    st.interest_objects,
                    st.interest_relevant_avg,
                    st.interest_throttled,
                    rows.join("\n")
                )
            }
            AdminCommand::Kick { id, reason } => {
                let known = self.members.get(&id).cloned();
                match (known, self.host.net_mut()) {
                    (Some(name), Some(net)) => {
                        net.kick(id, &reason);
                        tracing::info!("kicked {name} (id {id}){}", if reason.is_empty() { String::new() } else { format!(": {reason}") });
                        format!("kicked {name}")
                    }
                    _ => {
                        self.svc.failed = true;
                        format!("no player with id {id} (see `players`)")
                    }
                }
            }
            AdminCommand::Say(text) => {
                if let Some(net) = self.host.net_mut() {
                    net.send_message(None, &text);
                }
                tracing::info!("say: {text}");
                "sent".into()
            }
            AdminCommand::Scene(path) => {
                if path.starts_with('/') || path.contains("..") || !self.cfg.project_dir().join(&path).is_file() {
                    self.svc.failed = true;
                    return format!("no scene {path:?} in the project");
                }
                // The scene about to be replaced: keep its persistent objects as they are now.
                self.capture_world();
                tracing::info!("switching to scene {path}");
                match self.host.net_mut() {
                    Some(net) => net.load_scene(&path),
                    None => return "no session".into(),
                }
                format!("switching to {path}")
            }
            AdminCommand::Quit => {
                self.exit.get_or_insert(ExitReason::AdminQuit);
                "shutting down".into()
            }
            // Handled by `exec_service` above.
            _ => unreachable!("service commands never get here"),
        }
    }

    /// Tell the players why, close the session, log the totals. Idempotent.
    pub fn shutdown(&mut self, reason: ExitReason) {
        tracing::info!("shutting down ({reason:?}) with {} player(s) connected, after {} ticks", self.members.len(), self.host.ticks());
        // Save the state, leave the directory, and word the notice (with a redirect, if one was set).
        let notice = self.shutdown_services(reason);
        if let Some(net) = self.host.net_mut()
            && !matches!(net.state(), ConnState::Disconnected(_))
        {
            net.close_session_with(&notice);
        }
        self.members.clear();
    }

    /// Run the fixed loop until a stop request, then shut down. Returns why it ended.
    pub fn run(&mut self, ctl: Control) -> ExitReason {
        let hz = self.cfg.tick_rate;
        let cap = self.cfg.max_catch_up_ticks;
        let report = run_fixed(&mut Runner { server: self, ctl }, hz, cap);
        let reason = self.exit.unwrap_or(ExitReason::Stopped);
        self.shutdown(reason);
        tracing::info!("ran {} ticks, {} overloaded frame(s) dropped {} tick(s)", report.ticks, report.overloads, report.dropped_ticks);
        reason
    }
}

struct Runner<'a> {
    server: &'a mut Server,
    ctl: Control,
}

impl TickBody for Runner<'_> {
    fn tick(&mut self, dt: f32) {
        self.server.tick(dt);
    }

    fn should_stop(&mut self) -> bool {
        let s = &mut *self.server;
        if s.exit.is_none() {
            if self.ctl.watch_signals && signals::requested() {
                s.exit = Some(ExitReason::Signal);
            } else if self.ctl.stop.load(Ordering::Relaxed) {
                s.exit = Some(ExitReason::Stopped);
            }
        }
        s.exit.is_some()
    }

    fn pre_frame(&mut self) {
        while let Ok(req) = self.ctl.rx.try_recv() {
            match req {
                Request::Command(cmd, reply) => drop(reply.send(self.server.exec(cmd))),
                Request::Call(f) => f(self.server),
            }
        }
    }

    fn overloaded(&mut self, dropped: u64) {
        let s = &mut *self.server;
        s.overloaded_since_warn += dropped;
        if s.last_overload_warn.is_none_or(|t| t.elapsed() >= Duration::from_secs(5)) {
            tracing::warn!(
                "server overloaded: dropped {} tick(s) of backlog (tick budget {:.1} ms, last tick window mean {:.2} ms max {:.2} ms)",
                s.overloaded_since_warn,
                1000.0 / s.cfg.tick_rate,
                s.window.mean_ms(),
                s.window.max_ms,
            );
            s.last_overload_warn = Some(Instant::now());
            s.overloaded_since_warn = 0;
        }
    }
}

/// The server's signing identity: kept on disk so players can recognise it
/// across restarts; a throwaway one if the file cannot be used.
fn server_identity(cfg: &ServerConfig) -> Identity {
    let path = cfg.identity_path();
    Identity::load_or_create(&path).unwrap_or_else(|e| {
        tracing::warn!("server identity file {} unusable ({e}); using a temporary identity", path.display());
        Identity::generate()
    })
}

fn human_bytes(n: u64) -> String {
    match n {
        0..1024 => format!("{n} B"),
        1024..1_048_576 => format!("{:.1} KiB", n as f64 / 1024.0),
        _ => format!("{:.1} MiB", n as f64 / 1_048_576.0),
    }
}

/// A server running on its own thread (tests, tools). The server is built on
/// that thread, so nothing about it needs to be `Send`.
pub struct ServerHandle {
    /// A loopback address clients can join.
    pub addr: SocketAddr,
    pub remote: Remote,
    thread: Option<JoinHandle<ExitReason>>,
}

impl ServerHandle {
    pub fn spawn(cfg: ServerConfig) -> Result<Self> {
        Self::spawn_with(cfg, |_| {})
    }

    pub fn spawn_with(cfg: ServerConfig, register: impl FnOnce(&mut ComponentRegistry) + Send + 'static) -> Result<Self> {
        let (ctl, remote) = control(false);
        let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<u16, String>>();
        let thread = std::thread::Builder::new().name("zestengine-server".into()).spawn(move || {
            let mut server = match Server::open_with(cfg, register) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return ExitReason::Stopped;
                }
            };
            let _ = ready_tx.send(Ok(server.port()));
            server.run(ctl)
        })?;
        match ready_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(Ok(port)) => Ok(Self { addr: SocketAddr::from(([127, 0, 0, 1], port)), remote, thread: Some(thread) }),
            Ok(Err(e)) => Err(anyhow!(e)),
            Err(_) => bail!("the server did not start within 60 s"),
        }
    }

    /// Ask the loop to stop and wait up to `budget` for it to finish.
    pub fn shutdown(mut self, budget: Duration) -> Result<ExitReason> {
        self.remote.stop();
        let thread = self.thread.take().context("already joined")?;
        let deadline = Instant::now() + budget;
        while !thread.is_finished() {
            if Instant::now() > deadline {
                bail!("the server did not stop within {budget:?}");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        thread.join().map_err(|_| anyhow!("the server thread panicked"))
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(t) = self.thread.take() {
            self.remote.stop();
            let _ = t.join();
        }
    }
}

/// `--headless-selftest`: load the scene, host, join a loopback client, check
/// the player is spawned, shut down and check the client was told. No LAN
/// discovery, no console. Returns a one-line summary.
pub fn selftest(cfg: ServerConfig) -> Result<String> {
    selftest_with(cfg, |_| {})
}

/// [`selftest`] with the game's own component types registered.
pub fn selftest_with(mut cfg: ServerConfig, register: impl FnOnce(&mut ComponentRegistry)) -> Result<String> {
    cfg.port = Some(0);
    cfg.lan_discovery = false;
    cfg.console = false;
    let secret = cfg.password.clone();
    let mut server = Server::open_with(cfg, register)?;
    let expects_avatar = manager(server.host.scene()).is_some_and(|m| !m.player_prefab.is_empty());
    let dt = 1.0 / server.cfg.tick_rate;
    let join_cfg = NetConfig { player_name: "selftest".into(), discovery: false, secret, identity: Identity::generate(), ..Default::default() };
    let mut client = NetSession::join_with(&format!("127.0.0.1:{}", server.port()), join_cfg).context("starting the loopback client")?;

    let mut spawned = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    let step = |server: &mut Server, client: &mut NetSession| -> bool {
        server.tick(dt);
        client.pump();
        let got = client.view().world_ops.iter().any(|op| matches!(op, zestengine_core::NetWorldOp::Spawn { .. }));
        std::thread::sleep(Duration::from_secs_f32(dt));
        got
    };
    while !(client.is_connected() && (spawned || !expects_avatar)) {
        if Instant::now() > deadline {
            bail!("loopback client did not join{} within 10 s (state {:?})", if client.is_connected() { " and get a player" } else { "" }, client.state());
        }
        spawned |= step(&mut server, &mut client);
    }
    let players = server.player_count();
    if players != 1 {
        bail!("expected 1 player, the server counts {players}");
    }
    for _ in 0..30 {
        step(&mut server, &mut client);
    }
    server.shutdown(ExitReason::Stopped);
    let deadline = Instant::now() + Duration::from_secs(3);
    while matches!(client.state(), ConnState::Connected) {
        if Instant::now() > deadline {
            bail!("the client was not told about the shutdown");
        }
        client.pump();
        std::thread::sleep(Duration::from_millis(10));
    }
    Ok(format!(
        "selftest ok: {} loaded, {} ticks, loopback client joined{}, told on shutdown ({:?})",
        server.host.scene_path(),
        server.ticks(),
        if expects_avatar { " and got a player" } else { "" },
        client.state()
    ))
}
