//! NET-6 services of the running server: persistent state, signed remote admin,
//! the directory announcement, moderation commands and config reload. A child of
//! `server` so it can use the server's private state.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use zestengine_core::{NetValue, NetworkVariables, ObjectId};
use zestengine_engine::net::{Identity, NetConfig, fingerprint, short_fingerprint};
use zestengine_engine::net_validate::unix_now;

fn same_fp(a: &str, b: &str) -> bool {
    let n = |s: &str| s.chars().filter(|c| c.is_ascii_hexdigit()).map(|c| c.to_ascii_lowercase()).collect::<String>();
    !n(a).is_empty() && n(a) == n(b)
}
use zestengine_relay::directory::ServerInfo;

use super::{ExitReason, Server, human_bytes};
use crate::admin::AdminCommand;
use crate::announce::Announcer;
use crate::config::{Cli, ServerConfig};
use crate::persist::{Persistence, VarWrite};
use crate::remote::{RemoteAdmin, Status};

/// Exit code of `restart`: a supervisor (systemd `RestartForceExitStatus=75`, a container
/// restart policy) starts the server again. 75 is `EX_TEMPFAIL`.
pub const RESTART_EXIT_CODE: u8 = 75;

/// How often flagged scene objects are copied into the state (cheap; the file is written less often).
const CAPTURE_EVERY: Duration = Duration::from_secs(2);
/// A joined player whose player object never shows up stops being waited for after this.
const RESTORE_WAIT: Duration = Duration::from_secs(10);

/// A respawned persistent object waiting to re-enter the scene with its
/// saved variables: (object id, variables, respawn deadline).
type PendingSpawn = (ObjectId, Vec<(String, NetValue)>, Instant);

#[derive(Default)]
pub(super) struct Services {
    pub identity: Option<Identity>,
    pub persist: Option<Persistence>,
    pub remote: Option<RemoteAdmin>,
    pub announcer: Option<Announcer>,
    /// Where `reload` re-reads the configuration from (the command line that started the server).
    pub reload: Option<Cli>,
    /// Offered to players by the next `shutdown` / `restart`.
    pub redirect: String,
    /// Persisted scene path and the last capture.
    scene_path: String,
    last_capture: Option<Instant>,
    /// Peer id to player id fingerprint, for connected players.
    fps: BTreeMap<u64, String>,
    /// Admin keys added / removed while running (kept in the state folder).
    pub key_overlay: crate::admin_keys::KeyOverlay,
    /// Player reports waiting for a moderator.
    pub reports: std::collections::VecDeque<String>,
    /// Respawned persistent objects waiting to appear in the scene, to hand their saved variables to.
    pending_spawned: Vec<PendingSpawn>,
    /// Joined players still waiting for their player object, to hand saved variables to.
    pending_restore: Vec<(u64, Instant)>,
    /// Set by a command that did not do what was asked (the remote reply says so).
    pub failed: bool,
    pub saves: u64,
    pub restored_objects: usize,
    pub restored_players: u64,
    /// NET-7 heartbeat, metrics file and optional loopback HTTP.
    pub health: HealthRt,
}

/// Runtime state of the heartbeat writer.
#[derive(Default)]
pub(super) struct HealthRt {
    pub last: Option<Instant>,
    /// Tick costs since the last reading.
    pub window: super::StatsWindow,
    /// Totals and time of the last reading, for rates.
    pub prev: Option<(Instant, u64, u64)>,
    pub http: Option<crate::health::MetricsServer>,
}

/// Relay settings from `server.ron` over the scene's Network Manager.
pub(super) fn apply_relay(cfg: &ServerConfig, net: &mut NetConfig) {
    if !cfg.relay.is_empty() {
        net.relay = cfg.relay.clone();
    }
    if !cfg.relay_room.is_empty() {
        net.room = cfg.relay_room.clone();
    }
    let token = if cfg.relay_token.is_empty() { std::env::var("ZESTENGINE_RELAY_TOKEN").unwrap_or_default() } else { cfg.relay_token.clone() };
    if !token.is_empty() {
        net.relay_token = token;
    }
}

/// `target` as a fingerprint in the engine's display form, if it is 32 hex digits.
fn full_fingerprint(target: &str) -> Option<String> {
    let hex: String = target.chars().filter(|c| c.is_ascii_hexdigit()).map(|c| c.to_ascii_lowercase()).collect();
    if hex.len() != 32 || target.chars().any(|c| !(c.is_ascii_hexdigit() || c == '-' || c == ' ')) {
        return None;
    }
    Some(hex.as_bytes().chunks(4).map(|c| std::str::from_utf8(c).unwrap_or("")).collect::<Vec<_>>().join("-"))
}

fn remaining(until: u64, now: u64) -> String {
    if until == 0 {
        return "permanent".into();
    }
    let s = until.saturating_sub(now);
    match s {
        0..120 => format!("{s}s left"),
        120..7200 => format!("{}m left", s / 60),
        7200..172_800 => format!("{}h left", s / 3600),
        _ => format!("{}d left", s / 86_400),
    }
}

impl Services {
    pub(super) fn with_identity(identity: Identity) -> Self {
        Self { identity: Some(identity), ..Default::default() }
    }
}

impl Server {
    // ------------------------------------------------------------------ startup

    /// Open persistence, the admin port and the directory announcement. Called once the
    /// session is hosting. Problems are logged, never fatal, except a bad admin port.
    pub(super) fn open_services(&mut self) -> anyhow::Result<()> {
        let identity = self.svc.identity.clone();
        if self.cfg.persist {
            let p = Persistence::open(&self.cfg.state_path(), self.cfg.persist_keep, self.cfg.persist_interval);
            let path = self.host.scene_path().to_string();
            let (n, writes) = p.restore_scene(self.host.scene_mut(), &path);
            if n > 0 {
                self.host.scene_edited();
                tracing::info!("restored {n} persistent object(s) of {path}");
            }
            self.svc.restored_objects += n;
            self.svc.scene_path = path.clone();
            self.svc.persist = Some(p);
            self.apply_writes(writes);
            self.respawn_persisted(&path);
        }
        if !self.cfg.admin_tokens.is_empty() {
            tracing::warn!("admin_tokens is retired and ignored: list Ed25519 public keys in admin_keys (zestengine-admin keygen)");
        }
        self.svc.key_overlay = crate::admin_keys::KeyOverlay::load(&self.cfg.state_path());
        if let (Some(port), Some(id)) = (self.cfg.admin_port, &identity) {
            let keys = self.svc.key_overlay.effective(&self.cfg.admin_keys);
            let r = RemoteAdmin::bind(&self.cfg.admin_bind, port, id.public(), &keys)
                .map_err(|e| anyhow::anyhow!("binding the admin port {}:{port}: {e}", self.cfg.admin_bind))?;
            tracing::info!(
                "remote admin on UDP {} with {} key(s); server key {} (give it to zestengine-admin --server-key)",
                r.local_addr().map_or_else(|_| port.to_string(), |a| a.to_string()),
                r.key_count(),
                crate::remote::hex(&id.public())
            );
            self.svc.remote = Some(r);
        }
        if !self.cfg.directory.is_empty() {
            match Announcer::new(&self.cfg.directory, self.cfg.directory_interval) {
                Ok(a) => {
                    tracing::info!("announcing to the directory {} every {} s", a.relay(), self.cfg.directory_interval);
                    self.svc.announcer = Some(a);
                }
                Err(e) => tracing::error!("directory {:?} unusable ({e}); this server will not be listed", self.cfg.directory),
            }
        }
        if self.cfg.health_interval > 0.0 && !self.cfg.metrics_addr.is_empty() {
            let max_age = ((self.cfg.health_interval * 3.0).ceil() as u64).max(crate::health::DEFAULT_MAX_AGE);
            let addr = crate::health::parse_metrics_addr(&self.cfg.metrics_addr).map_err(|e| anyhow::anyhow!(e))?;
            let http = crate::health::MetricsServer::start(addr, max_age).map_err(|e| anyhow::anyhow!("binding metrics_addr {addr}: {e}"))?;
            tracing::info!("metrics on http://{} (/health, /metrics; loopback only)", http.addr);
            self.svc.health.http = Some(http);
        }
        self.write_health("running");
        Ok(())
    }

    /// The server's vital signs now. Rates and tick costs cover the time since the last reading.
    fn health_snapshot(&mut self, state: &str) -> crate::health::HealthSnapshot {
        let (bin, bout) = self.net_totals();
        let now = Instant::now();
        let (rin, rout) = match self.svc.health.prev {
            Some((t, i, o)) if now > t => {
                let secs = now.duration_since(t).as_secs_f64().max(1e-3);
                (bin.saturating_sub(i) as f64 / secs, bout.saturating_sub(o) as f64 / secs)
            }
            _ => (0.0, 0.0),
        };
        self.svc.health.prev = Some((now, bin, bout));
        let w = std::mem::take(&mut self.svc.health.window);
        let net = self.host.net();
        let st = net.map(|n| n.stats);
        crate::health::HealthSnapshot {
            state: state.into(),
            unix: unix_now(),
            pid: std::process::id(),
            name: self.cfg.server_name.clone(),
            game_port: self.port,
            uptime_s: self.started.elapsed().as_secs(),
            ticks: self.host.ticks(),
            tick_rate: self.cfg.tick_rate,
            tick_mean_ms: w.mean_ms(),
            tick_max_ms: w.max_ms,
            players: self.members.len(),
            max_players: self.cfg.max_players,
            bytes_in: bin,
            bytes_out: bout,
            rate_in: rin,
            rate_out: rout,
            violations: st.map_or(0, |s| s.violations),
            muted: st.map_or(0, |s| s.muted),
            interest_enabled: net.is_some_and(|n| n.interest_config().enabled),
            interest_objects: st.map_or(0, |s| s.interest_objects as u64),
            interest_relevant_avg: st.map_or(0.0, |s| f64::from(s.interest_relevant_avg)),
            interest_dropped: st.map_or(0, |s| s.interest_dropped),
            interest_throttled: st.map_or(0, |s| s.interest_throttled),
            bans: net.map_or(0, |n| n.val.bans.entries().len()),
        }
    }

    /// Write the heartbeat and metrics files and refresh the HTTP texts. Failures are logged once per
    /// minute at most by the caller's cadence; they never stop the server.
    fn write_health(&mut self, state: &str) {
        if self.cfg.health_interval <= 0.0 {
            return;
        }
        self.svc.health.last = Some(Instant::now());
        let snap = self.health_snapshot(state);
        if let Some(h) = &self.svc.health.http {
            h.publish(&snap);
        }
        for (path, text) in [(crate::health::status_path(&self.cfg), snap.status_text()), (crate::health::metrics_path(&self.cfg), snap.prometheus_text())] {
            if let Err(e) = crate::health::write_atomic(&path, &text) {
                tracing::warn!("cannot write {}: {e}", path.display());
            }
        }
    }

    fn tick_health(&mut self) {
        if self.cfg.health_interval > 0.0 && self.svc.health.last.is_none_or(|t| t.elapsed().as_secs_f32() >= self.cfg.health_interval) {
            self.write_health("running");
        }
    }

    /// Where signed remote admin listens, when it is on.
    pub fn admin_addr(&self) -> Option<std::net::SocketAddr> {
        self.svc.remote.as_ref().and_then(|r| r.local_addr().ok())
    }

    /// The public key admin tools bind their requests to (the server's identity).
    pub fn server_key(&self) -> Option<[u8; 32]> {
        self.svc.identity.as_ref().map(Identity::public)
    }

    /// Tell the session about variable writes made on the server's own copy of the scene.
    fn apply_writes(&mut self, writes: Vec<VarWrite>) {
        if let Some(net) = self.host.net_mut() {
            for (id, k, v) in writes {
                net.set_var(id, &k, v);
            }
        }
    }

    /// What the directory lists about this server.
    pub fn directory_info(&self) -> ServerInfo {
        let game = if self.cfg.game.is_empty() {
            self.cfg.project_dir().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        } else {
            self.cfg.game.clone()
        };
        ServerInfo {
            name: self.cfg.server_name.clone(),
            game,
            version: self.cfg.game_version.clone(),
            map: self.host.scene_path().to_string(),
            region: self.cfg.region.clone(),
            players: self.members.len().min(u16::MAX as usize) as u16,
            max_players: self.cfg.max_players.min(u32::from(u16::MAX)) as u16,
            port: self.port,
            room: self.host.net().and_then(|n| n.room_code()).unwrap_or_default(),
            password: !self.cfg.password.is_empty() || !self.session_open(),
            protocol: zestengine_engine::net::PROTOCOL,
        }
    }

    // ------------------------------------------------------------------ per tick

    /// Directory heartbeat, remote admin, persistence.
    pub(super) fn tick_services(&mut self) {
        let info = self.svc.announcer.is_some().then(|| self.directory_info());
        if let (Some(a), Some(id), Some(info)) = (self.svc.announcer.as_mut(), self.svc.identity.as_ref(), info) {
            a.tick(id, || info);
        }
        self.poll_remote();
        self.collect_reports();
        self.tick_persistence();
        self.tick_health();
    }

    fn poll_remote(&mut self) {
        let Some(id) = self.svc.identity.clone() else { return };
        let sign = |m: &[u8]| id.sign(m);
        let pending = match self.svc.remote.as_mut() {
            Some(r) => r.receive(&sign),
            None => return,
        };
        for p in pending {
            tracing::info!("remote admin {:?} ({:?}): {:?}", p.key.name, p.key.role, p.command);
            self.svc.failed = false;
            let out = self.exec(p.command.clone());
            let status = if std::mem::take(&mut self.svc.failed) { Status::Error } else { Status::Ok };
            if let Some(r) = self.svc.remote.as_ref() {
                r.answer(&sign, &p, status, &out);
            }
        }
    }

    fn tick_persistence(&mut self) {
        if self.svc.persist.is_none() {
            return;
        }
        // A scene switch (the host replaced the scene): put the new scene's state back.
        let path = self.host.scene_path().to_string();
        if path != self.svc.scene_path {
            self.svc.scene_path = path.clone();
            if let Some(p) = &self.svc.persist {
                let (n, writes) = p.restore_scene(self.host.scene_mut(), &path);
                if n > 0 {
                    self.host.scene_edited();
                    tracing::info!("restored {n} persistent object(s) of {path}");
                }
                self.svc.restored_objects += n;
                self.apply_writes(writes);
            }
            self.respawn_persisted(&path);
        }
        self.settle_respawned();
        self.track_player_data();
        if self.svc.last_capture.is_none_or(|t| t.elapsed() >= CAPTURE_EVERY) {
            self.capture_world();
        }
        if self.svc.persist.as_ref().is_some_and(Persistence::due) {
            self.save_state();
        }
    }

    /// Spawn the runtime-spawned persistent objects recorded for `path` again (server-owned).
    fn respawn_persisted(&mut self, path: &str) {
        let Some(list) = self.svc.persist.as_ref().map(|p| p.spawned_for(path)).filter(|l| !l.is_empty()) else { return };
        let mut n = 0;
        for s in list {
            let (t, r) = (glam::Vec3::from(s.t), glam::Quat::from_array(s.r));
            if !self.cfg.project_dir().join(&s.prefab).is_file() || !t.is_finite() || !r.is_finite() {
                tracing::warn!("persistent spawn of {:?} skipped (prefab missing or position invalid)", s.prefab);
                continue;
            }
            let Some(net) = self.host.net_mut() else { return };
            let id = net.spawn_unowned(zestengine_core::NetSpawnSource::Prefab(s.prefab.clone()), t, r.normalize());
            self.svc.pending_spawned.push((id, s.vars, Instant::now()));
            n += 1;
        }
        self.svc.restored_objects += n;
        if n > 0 {
            tracing::info!("spawned {n} persistent runtime object(s) of {path} again");
        }
    }

    /// Hand the saved variables to respawned objects as they appear in the scene.
    fn settle_respawned(&mut self) {
        if self.svc.pending_spawned.is_empty() {
            return;
        }
        let waiting = std::mem::take(&mut self.svc.pending_spawned);
        let mut writes = Vec::new();
        for (id, vars, since) in waiting {
            match self.host.scene_mut().index_of(id) {
                Some(i) => {
                    if let Some(v) = self.host.scene_mut().objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>()) {
                        for (k, val) in vars {
                            v.set(&k, val.clone());
                            writes.push((id, k, val));
                        }
                    }
                }
                None if since.elapsed() < RESTORE_WAIT => self.svc.pending_spawned.push((id, vars, since)),
                None => tracing::warn!("a respawned persistent object never appeared in the scene"),
            }
        }
        self.apply_writes(writes);
    }

    pub(super) fn capture_world(&mut self) {
        let path = self.svc.scene_path.clone();
        let spawned: Vec<(ObjectId, String)> = self
            .host
            .net()
            .map(|n| {
                n.world
                    .spawns
                    .iter()
                    .filter(|r| !r.player)
                    .filter_map(|r| match &r.source {
                        zestengine_core::NetSpawnSource::Prefab(p) => Some((ObjectId::from_raw(r.id), p.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        // While respawned objects are still arriving the recorded list is the truth.
        let waiting = !self.svc.pending_spawned.is_empty();
        if let Some(p) = self.svc.persist.as_mut() {
            p.capture_scene(self.host.scene(), &path, (!waiting).then_some(spawned.as_slice()));
        }
        self.svc.last_capture = Some(Instant::now());
    }

    /// Joins and leaves for the per-player store, and the copy of each player's flagged variables.
    fn track_player_data(&mut self) {
        let Some(net) = self.host.net() else { return };
        let joined: Vec<(u64, String)> = self.members.keys().filter(|id| !self.svc.fps.contains_key(id)).filter_map(|id| Some((*id, net.peer_fingerprint(*id)?))).collect();
        self.svc.fps.retain(|id, _| self.members.contains_key(id));
        let objects: BTreeMap<u64, u128> = self.svc.fps.keys().filter_map(|id| Some((*id, *net.world.player_obj.get(id)?))).collect();
        let Some(p) = self.svc.persist.as_mut() else { return };
        for (id, fp) in joined {
            let name = self.members.get(&id).cloned().unwrap_or_default();
            if p.player_joined(&fp, &name) {
                self.svc.pending_restore.push((id, Instant::now()));
            }
            self.svc.fps.insert(id, fp);
        }
        let waiting: Vec<(u64, Instant)> = std::mem::take(&mut self.svc.pending_restore);
        let mut still = Vec::new();
        let mut writes = Vec::new();
        for (id, since) in waiting {
            let Some(fp) = self.svc.fps.get(&id) else { continue };
            // The player object spawns a few ticks after the join.
            match self.host.net().and_then(|n| n.world.player_obj.get(&id).copied()) {
                Some(obj) => match p.restore_player(self.host.scene_mut(), fp, ObjectId::from_raw(obj)) {
                    Some(w) => {
                        if !w.is_empty() {
                            tracing::info!("restored {} saved value(s) for player {id} ({})", w.len(), short_fingerprint(fp));
                            self.svc.restored_players += 1;
                            writes.extend(w);
                        }
                    }
                    // The session has the id a tick before the scene has the object.
                    None if since.elapsed() < RESTORE_WAIT => still.push((id, since)),
                    None => {}
                },
                None if since.elapsed() < RESTORE_WAIT => still.push((id, since)),
                None => {}
            }
        }
        self.svc.pending_restore = still;
        // Keep each player's record current (not while a restore is pending: that would overwrite it with defaults).
        for (id, obj) in objects {
            if self.svc.pending_restore.iter().any(|(p, _)| *p == id) {
                continue;
            }
            if let Some(fp) = self.svc.fps.get(&id) {
                p.capture_player(self.host.scene(), fp, ObjectId::from_raw(obj));
            }
        }
        self.apply_writes(writes);
    }

    /// Capture and write the state now. Returns the files written.
    pub(super) fn save_state(&mut self) -> usize {
        self.capture_world();
        let Some(p) = self.svc.persist.as_mut() else { return 0 };
        let n = p.save();
        if n > 0 {
            self.svc.saves += 1;
            tracing::info!("saved {n} state file(s) to {}", self.cfg.state_path().display());
        }
        n
    }

    /// Final save, delisting and the player notice, before the session closes.
    pub(super) fn shutdown_services(&mut self, reason: ExitReason) -> String {
        self.save_state();
        self.write_health("stopped");
        let info = self.directory_info();
        if let (Some(a), Some(id)) = (self.svc.announcer.as_mut(), self.svc.identity.as_ref()) {
            a.delist(id, &info);
        }
        let redirect = if self.svc.redirect.is_empty() { self.cfg.shutdown_redirect.clone() } else { self.svc.redirect.clone() };
        if redirect.is_empty() { reason.notice().to_string() } else { zestengine_engine::server::redirect_notice(reason.notice(), &redirect) }
    }

    // ------------------------------------------------------------------ commands

    fn fail(&mut self, msg: impl Into<String>) -> String {
        self.svc.failed = true;
        msg.into()
    }

    /// The NET-6 commands. `Err(cmd)` hands any other command back.
    pub(super) fn exec_service(&mut self, cmd: AdminCommand) -> Result<String, AdminCommand> {
        Ok(match cmd {
            AdminCommand::Stats => self.stats_text(),
            AdminCommand::Bans => self.bans_text(),
            AdminCommand::Ban { target, secs, reason } => self.ban(&target, secs, &reason),
            AdminCommand::Unban(target) => self.unban(&target),
            AdminCommand::Save => match self.svc.persist {
                None => self.fail("persistence is off (server.ron: persist: false)"),
                Some(_) => {
                    let n = self.save_state();
                    format!("saved ({n} file(s) changed)")
                }
            },
            AdminCommand::Reload => self.reload_config(),
            AdminCommand::Reports => self.reports_text(),
            AdminCommand::ReportsClear => {
                let n = self.svc.reports.len();
                self.svc.reports.clear();
                format!("cleared {n} report(s)")
            }
            AdminCommand::Keys => self.svc.key_overlay.describe(&self.cfg.admin_keys),
            AdminCommand::KeyAdd { name, key, role } => match self.svc.key_overlay.add(&self.cfg.admin_keys, &name, &key, role) {
                Ok(()) => self.keys_changed(format!("admin key {name:?} ({role:?}) added")),
                Err(e) => self.fail(e),
            },
            AdminCommand::KeyRemove(who) => match self.svc.key_overlay.remove(&self.cfg.admin_keys, &who) {
                Ok(k) => self.keys_changed(format!("admin key {:?} ({:?}) removed", k.name, k.role)),
                Err(e) => self.fail(e),
            },
            AdminCommand::Restart { redirect } => {
                self.svc.redirect = redirect;
                self.exit.get_or_insert(ExitReason::Restart);
                "restarting (exit code 75)".into()
            }
            AdminCommand::Shutdown { redirect } => {
                self.svc.redirect = redirect;
                self.exit.get_or_insert(ExitReason::AdminQuit);
                "shutting down".into()
            }
            other => return Err(other),
        })
    }

    /// Apply the changed key overlay to the listener and write it down.
    fn keys_changed(&mut self, what: String) -> String {
        if let Some(r) = self.svc.remote.as_mut() {
            r.set_keys(&self.svc.key_overlay.effective(&self.cfg.admin_keys));
        }
        let saved = match self.svc.key_overlay.save(&self.cfg.state_path()) {
            Ok(()) => String::new(),
            Err(e) => format!(" (in force now, but NOT saved: {e})"),
        };
        tracing::warn!("{what}{saved}");
        format!("{what}{saved}")
    }

    /// File the reports the session collected: log, queue, and a line in `reports.log`.
    pub(super) fn collect_reports(&mut self) {
        let Some(net) = self.host.net_mut() else { return };
        let taken = net.take_reports();
        if taken.is_empty() {
            return;
        }
        let unix = unix_now();
        for r in taken {
            let line = format!(
                "{unix} report by {} ({}) against {} ({}): {}",
                r.reporter_name,
                short_fingerprint(&r.reporter_fingerprint),
                r.target_name,
                r.target_fingerprint,
                r.reason
            );
            if self.svc.reports.len() >= 200 {
                self.svc.reports.pop_front();
            }
            self.svc.reports.push_back(line.clone());
            if self.cfg.persist {
                use std::io::Write as _;
                let dir = self.cfg.state_path();
                let _ = std::fs::create_dir_all(&dir);
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("reports.log")) {
                    let _ = writeln!(f, "{line}");
                }
            }
        }
    }

    fn reports_text(&self) -> String {
        if self.svc.reports.is_empty() {
            return "no reports".into();
        }
        let shown: Vec<&String> = self.svc.reports.iter().rev().take(20).collect::<Vec<_>>().into_iter().rev().collect();
        format!("{} report(s), newest last (ban with `ban <fingerprint>`):\n  {}", self.svc.reports.len(), shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n  "))
    }

    fn ban(&mut self, target: &str, secs: u64, reason: &str) -> String {
        let reason = if reason.is_empty() { "banned by an admin" } else { reason };
        let Some(net) = self.host.net_mut() else { return self.fail("no session") };
        let (fp, connected) = if let Ok(id) = target.parse::<u64>() {
            match net.peer_fingerprint(id) {
                Some(fp) if self.members.contains_key(&id) => (fp, Some(id)),
                _ => return self.fail(format!("no player with id {id} (see `players`; to ban an offline player give its fingerprint)")),
            }
        } else if let Some(fp) = full_fingerprint(target) {
            let id = self.members.keys().copied().find(|id| net.peer_fingerprint(*id).is_some_and(|f| f == fp));
            (fp, id)
        } else {
            return self.fail(format!("{target:?} is neither a player id nor a 32-digit fingerprint"));
        };
        if self.svc.identity.as_ref().is_some_and(|i| fingerprint(&i.public()) == fp) {
            return self.fail("that is this server's own identity");
        }
        net.val.bans.add(&fp, secs, reason, unix_now());
        // The provider account the anti-cheat layer linked to this player is banned with it.
        let linked = connected.is_some_and(|id| net.ban_provider_link(id, secs, reason));
        let name = connected.and_then(|id| self.members.get(&id).cloned());
        if let (Some(id), Some(net)) = (connected, self.host.net_mut()) {
            net.kick(id, &format!("banned: {reason}"));
        }
        let how = if secs == 0 { "permanently".to_string() } else { format!("for {}", remaining(unix_now() + secs, unix_now()).trim_end_matches(" left")) };
        tracing::info!("banned {} {how}: {reason}", name.as_deref().unwrap_or(&fp));
        let account = if linked { " (and its linked anti-cheat account)" } else { "" };
        match name {
            Some(n) => format!("banned {n} ({fp}){account} {how} and kicked"),
            None => format!("banned {fp} {how}"),
        }
    }

    fn unban(&mut self, target: &str) -> String {
        let digits: String = target.chars().filter(|c| c.is_ascii_hexdigit()).map(|c| c.to_ascii_lowercase()).collect();
        let Some(net) = self.host.net_mut() else { return self.fail("no session") };
        let fp = match full_fingerprint(target) {
            Some(fp) => fp,
            None if digits.len() >= 8 && digits.len() < 32 => {
                let hits: Vec<String> = net
                    .val
                    .bans
                    .entries()
                    .iter()
                    .filter(|b| b.fingerprint.chars().filter(|c| c.is_ascii_hexdigit()).collect::<String>().to_ascii_lowercase().starts_with(&digits))
                    .map(|b| b.fingerprint.clone())
                    .collect();
                match hits.as_slice() {
                    [one] => one.clone(),
                    [] => return self.fail(format!("no ban matches {target:?} (see `bans`)")),
                    many => return self.fail(format!("{target:?} matches {} bans; give more digits", many.len())),
                }
            }
            None => return self.fail("give the fingerprint from `bans` (32 hex digits, or a unique prefix of 8 or more)"),
        };
        let had = net.val.bans.entries().iter().any(|b| same_fp(&b.fingerprint, &fp));
        net.unban(&fp);
        if had {
            tracing::info!("unbanned {fp}");
            format!("unbanned {fp}")
        } else {
            self.fail(format!("{fp} was not banned"))
        }
    }

    fn bans_text(&self) -> String {
        let Some(net) = self.host.net() else { return "no session".into() };
        let now = unix_now();
        let rows: Vec<String> = net
            .val
            .bans
            .entries()
            .iter()
            .filter(|b| b.until == 0 || b.until > now)
            .map(|b| format!("  {}  {:<12} {}", b.fingerprint, remaining(b.until, now), if b.reason.is_empty() { "-" } else { &b.reason }))
            .collect();
        if rows.is_empty() { "no bans".into() } else { format!("{} ban(s):\n{}", rows.len(), rows.join("\n")) }
    }

    fn stats_text(&self) -> String {
        let (bin, bout) = self.net_totals();
        let w = if self.last_window.ticks > 0 { self.last_window } else { self.window };
        let net = self.host.net();
        let mut lines = vec![format!(
            "ticks {} ({} Hz), tick mean {:.2} ms max {:.2} ms; players {}/{}; net in {} out {}; violations {} (muted {})",
            self.host.ticks(),
            self.cfg.tick_rate,
            w.mean_ms(),
            w.max_ms,
            self.members.len(),
            self.cfg.max_players,
            human_bytes(bin),
            human_bytes(bout),
            net.map_or(0, |n| n.stats.violations),
            net.map_or(0, |n| n.stats.muted),
        )];
        match &self.svc.persist {
            Some(p) => lines.push(format!(
                "persistence: {} scene(s) and {} player(s) tracked, {} save(s) this run, restored {} object(s) and {} player(s); state in {}",
                p.world().scenes.len(),
                p.players().players.len(),
                self.svc.saves,
                self.svc.restored_objects,
                self.svc.restored_players,
                self.cfg.state_path().display()
            )),
            None => lines.push("persistence: off".into()),
        }
        match &self.svc.remote {
            Some(r) => lines.push(format!(
                "remote admin: {} key(s); {} command(s) run, {} stale/replayed, {} denied, {} unauthenticated dropped, {} rate limited",
                r.key_count(),
                r.stats.executed,
                r.stats.stale,
                r.stats.denied,
                r.stats.rejected,
                r.stats.rate_limited
            )),
            None => lines.push("remote admin: off".into()),
        }
        match &self.svc.announcer {
            Some(a) => lines.push(format!("directory: {} (acks {}, refusals {})", a.relay(), a.acks, a.errors)),
            None => lines.push("directory: not listed".into()),
        }
        if let Some(room) = net.and_then(|n| n.room_code()) {
            lines.push(format!("relay room: {room}"));
        }
        lines.join("\n")
    }

    /// Re-read the configuration the server was started with and apply what can change live.
    fn reload_config(&mut self) -> String {
        let Some(cli) = self.svc.reload.clone() else { return self.fail("no configuration to reload (the server was started without a command line)") };
        let new = match cli.resolve() {
            Ok(c) => c,
            Err(e) => return self.fail(format!("reload refused, the running configuration is unchanged:\n{e:#}")),
        };
        let old = std::mem::replace(&mut self.cfg, new.clone());
        // These need a restart (a socket, the scene, the session): keep the running values.
        let mut needs_restart = Vec::new();
        macro_rules! keep {
            ($($f:ident),*) => {$(
                if old.$f != new.$f {
                    needs_restart.push(stringify!($f));
                    self.cfg.$f = old.$f.clone();
                }
            )*};
        }
        keep!(project, scene, port, tick_rate, max_catch_up_ticks, max_players, server_name, password, lan_discovery, identity_file, ban_file, validation, interest, state_dir, persist, admin_port, admin_bind, directory, relay, relay_room, relay_token, log_path, log_level, log_max_mb, log_keep, console);
        let mut applied = Vec::new();
        macro_rules! note {
            ($($f:ident),*) => {$( if old.$f != self.cfg.$f { applied.push(stringify!($f)); } )*};
        }
        note!(motd, stats_interval, restart_on_scene_end, admin_keys, persist_interval, persist_keep, directory_interval, region, game, game_version, shutdown_redirect);
        if old.admin_keys != self.cfg.admin_keys && let Some(r) = self.svc.remote.as_mut() {
            // Keys added or retired while running stay in force over the re-read file.
            r.set_keys(&self.svc.key_overlay.effective(&self.cfg.admin_keys));
        }
        if old.directory_interval != self.cfg.directory_interval && !self.cfg.directory.is_empty() {
            match Announcer::new(&self.cfg.directory, self.cfg.directory_interval) {
                Ok(a) => self.svc.announcer = Some(a),
                Err(e) => tracing::warn!("directory announcer not restarted: {e}"),
            }
        }
        if (old.persist_interval != self.cfg.persist_interval || old.persist_keep != self.cfg.persist_keep) && self.svc.persist.is_some() {
            // Save what is pending, then reopen with the new interval / generations (the files are the source of truth).
            self.save_state();
            self.svc.persist = Some(Persistence::open(&self.cfg.state_path(), self.cfg.persist_keep, self.cfg.persist_interval));
        }
        let mut out = if applied.is_empty() { "reloaded: nothing changed".to_string() } else { format!("reloaded: {}", applied.join(", ")) };
        if !needs_restart.is_empty() {
            out.push_str(&format!("\nchanged in the file but needs a restart: {}", needs_restart.join(", ")));
        }
        tracing::info!("{}", out.replace('\n', "; "));
        out
    }

    /// Remember the command line, so `reload` can re-read the configuration.
    pub fn set_reload_source(&mut self, cli: Cli) {
        self.svc.reload = Some(cli);
    }
}
