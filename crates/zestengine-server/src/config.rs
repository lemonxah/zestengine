//! Server configuration: `server.ron` plus command-line flags (flags win).

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use zestengine_core::{InterestConfig, NetValidation};

use crate::admin::AdminKey;

/// Port used when neither the config nor the scene's Network Manager names one.
pub const DEFAULT_PORT: u16 = 9000;

/// Everything `server.ron` can set. Every field has a default, so a file only
/// lists what it changes; unknown fields are an error (a typo should not be
/// silently ignored on a server).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Project folder (holds `project.zestengine`, `scenes/`, `prefabs/` ...).
    pub project: String,
    /// Project-relative scene to start with; `None` takes the project's boot scene.
    pub scene: Option<String>,
    /// UDP port; `None` takes the scene's Network Manager port (9000 without one).
    /// `Some(0)` asks the OS for a free port.
    pub port: Option<u16>,
    /// Fixed tick rate in Hz.
    pub tick_rate: f32,
    /// Most ticks run in one wake-up when the server falls behind; the rest of
    /// the backlog is dropped with a warning.
    pub max_catch_up_ticks: u32,
    /// Players that may be connected at once (the server itself does not count).
    pub max_players: u32,
    /// Session name shown in LAN discovery, and the server's player name.
    pub server_name: String,
    /// Join password (the session secret: joiners must know it and it keys the
    /// session encryption). Empty = open.
    pub password: String,
    /// Log file path; `None` logs to the console only.
    pub log_path: Option<String>,
    /// A level (`error` .. `trace`) or a full `tracing` filter (`info,zestengine_engine::net=debug`).
    pub log_level: String,
    /// Rotate the log file when it passes this many megabytes.
    pub log_max_mb: u32,
    /// Rotated log files kept (`server.log.1` .. `.N`).
    pub log_keep: u32,
    /// Answer LAN discovery queries.
    pub lan_discovery: bool,
    /// Sent privately to every player that joins. Empty = none.
    pub motd: String,
    /// Retired (shared-secret tokens): ignored with a warning. Use `admin_keys`.
    pub admin_tokens: Vec<String>,
    /// When a component asks to quit (the scene ended), reload the scene and keep
    /// the session instead of exiting.
    pub restart_on_scene_end: bool,
    /// Seconds between stats lines; 0 turns them off.
    pub stats_interval: f32,
    /// Read admin commands from stdin.
    pub console: bool,
    /// Where the server's signing identity is kept; empty = `<project>/.server/identity.key`.
    pub identity_file: String,
    /// Where bans (made by the strike system or an admin) are kept; empty = `<project>/.server/bans.ron`.
    pub ban_file: String,
    /// Validation limits for everything clients send (movement, VR poses,
    /// ownership, RPC permissions, spawn allow-list, rate limits, strikes).
    /// `None` takes the scene's Network Manager settings; `Some` replaces them,
    /// and a file only names the fields it changes: `validation: (max_rewind: 0.2)`.
    pub validation: Option<NetValidation>,
    /// Interest management for big worlds (NET-5): only tell each player about
    /// objects near it, at a rate that falls with distance, within a byte
    /// budget. `None` takes the scene's Network Manager settings; `Some`
    /// replaces them, and a file only names the fields it changes:
    /// `interest: (enabled: true, radius: 120.0, budget: 60000)`.
    pub interest: Option<InterestConfig>,
    /// Anti-cheat provider (SEC-2): `anticheat: (provider: "none", require: false)`.
    /// `None` takes the scene's Network Manager settings; `Some` replaces them.
    pub anticheat: Option<zestengine_core::AntiCheatConfig>,
    /// Project component plugins (NET-3): built `cdylib` files (`libzestengine_project_components.so`)
    /// whose components the scene uses. Relative paths resolve against the server's working folder.
    /// Each must carry the ABI stamp (`zestengine_core::export_plugin_abi!()`) and match this
    /// server's rustc, engine version and profile; any failure refuses to start.
    pub plugins: Vec<String>,

    // ---- NET-6: server services -------------------------------------------------
    /// Folder for persistent state, bans and the identity; empty = `<project>/.server`.
    pub state_dir: String,
    /// Save persistent objects and player data this often (seconds); 0 = only at shutdown.
    pub persist_interval: f32,
    /// Backup generations of each state file kept besides the current one (0 to 20).
    pub persist_keep: u32,
    /// Keep scene objects with a Persistent State component and per-player data across restarts.
    pub persist: bool,
    /// Remote admin: public keys that may send signed commands, with a role each.
    pub admin_keys: Vec<AdminKey>,
    /// UDP port for signed remote admin; `None` = no remote admin.
    pub admin_port: Option<u16>,
    /// Address the admin port binds to.
    pub admin_bind: String,
    /// Server directory: the relay (`host[:port]`) to announce this server to; empty = not listed.
    pub directory: String,
    /// Seconds between announcements (the directory's TTL should be about three times this).
    pub directory_interval: f32,
    /// Region tag listed in the directory (`eu`, `na-east`).
    pub region: String,
    /// Game identifier listed in the directory; empty = the project folder's name.
    pub game: String,
    /// Game version listed in the directory.
    pub game_version: String,
    /// Relay (`host[:port]`) to open a room on, so players can join by code when the
    /// server is behind a NAT; empty takes the scene's Network Manager setting.
    pub relay: String,
    /// Room code to ask the relay for; empty = the relay picks.
    pub relay_room: String,
    /// Operator token for relays that require one to open rooms (prefer ZESTENGINE_RELAY_TOKEN).
    pub relay_token: String,
    /// Offered to players when the server stops (`host:port` or a room code): a player's
    /// game may follow it. The `shutdown` / `restart` commands can name one per stop.
    pub shutdown_redirect: String,

    // ---- NET-7: deployment ------------------------------------------------------
    /// Seconds between heartbeat writes to `<state_dir>/health.status` (and `metrics.prom`),
    /// which `--healthcheck` verifies; 0 = off.
    pub health_interval: f32,
    /// `ip:port` (loopback only) of a tiny HTTP listener serving `/health` and `/metrics`;
    /// empty = off.
    pub metrics_addr: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            project: String::new(),
            scene: None,
            port: None,
            tick_rate: 30.0,
            max_catch_up_ticks: 5,
            max_players: 16,
            server_name: "ZestEngine Server".to_string(),
            password: String::new(),
            log_path: None,
            log_level: "info".to_string(),
            log_max_mb: 10,
            log_keep: 5,
            lan_discovery: true,
            motd: String::new(),
            admin_tokens: Vec::new(),
            restart_on_scene_end: false,
            stats_interval: 10.0,
            console: true,
            identity_file: String::new(),
            ban_file: String::new(),
            validation: None,
            interest: None,
            anticheat: None,
            state_dir: String::new(),
            persist_interval: 60.0,
            persist_keep: 3,
            persist: true,
            admin_keys: Vec::new(),
            admin_port: None,
            admin_bind: "0.0.0.0".to_string(),
            directory: String::new(),
            directory_interval: 30.0,
            region: String::new(),
            game: String::new(),
            game_version: String::new(),
            relay: String::new(),
            relay_room: String::new(),
            relay_token: String::new(),
            shutdown_redirect: String::new(),
            health_interval: 5.0,
            metrics_addr: String::new(),
            plugins: Vec::new(),
        }
    }
}

/// RON options for `server.ron`: `port: 9000` may be written without `Some(..)`.
fn ron_options() -> ron::Options {
    ron::Options::default().with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
}

impl ServerConfig {
    pub fn from_ron_str(text: &str) -> Result<Self> {
        ron_options().from_str(text).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_ron_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// The effective settings as RON, secrets (join password, relay token) replaced by
    /// `"<set>"`; what `--print-config` shows. Parses back with [`from_ron_str`](Self::from_ron_str).
    pub fn to_ron_redacted(&self) -> Result<String> {
        let mut c = self.clone();
        for s in [&mut c.password, &mut c.relay_token] {
            if !s.is_empty() {
                *s = "<set>".into();
            }
        }
        let text = ron::ser::to_string_pretty(&c, ron::ser::PrettyConfig::default()).context("serialising the configuration")?;
        Ok(format!("// Effective configuration (secrets hidden). Paths are as given; state: {}\n{text}\n", self.state_path().display()))
    }

    /// `plugins` as paths.
    pub fn plugin_paths(&self) -> Vec<PathBuf> {
        self.plugins.iter().map(PathBuf::from).collect()
    }

    pub fn project_dir(&self) -> PathBuf {
        PathBuf::from(&self.project)
    }

    /// The `assets.pak` the server should unpack: `project` itself when it is a file, or a
    /// folder that has an `assets.pak` but no `project.zestengine` (NET-3). `None` = loose assets.
    pub fn pak_path(&self) -> Option<PathBuf> {
        let p = self.project_dir();
        if p.is_file() {
            return Some(p);
        }
        let pak = p.join(zestengine_assets::game_pak::GAME_PAK_NAME);
        (p.is_dir() && !p.join("project.zestengine").is_file() && pak.is_file()).then_some(pak)
    }

    /// If the project is an `assets.pak`, extract it (once per pak id) into `<state>/assets` and
    /// point `project` at the extraction. The state folder keeps its old default (next to the
    /// pak), not inside the cache. Returns the pak used.
    pub fn unpack_pak(&mut self) -> Result<Option<PathBuf>> {
        let Some(pak) = self.pak_path() else { return Ok(None) };
        if self.state_dir.is_empty() {
            self.state_dir = pak.parent().map(|d| d.join(".server")).unwrap_or_else(|| PathBuf::from(".server")).to_string_lossy().into_owned();
        }
        let cache = self.state_path().join("assets");
        let root = zestengine_assets::game_pak::extract_game_pak(&pak, &cache).with_context(|| format!("unpacking {}", pak.display()))?;
        self.project = root.to_string_lossy().into_owned();
        // The scene may be checked against the extraction now.
        if let Some(scene) = &self.scene {
            anyhow::ensure!(root.join(scene).is_file(), "scene {scene:?} not found in {}", pak.display());
        }
        Ok(Some(pak))
    }

    /// The ban list file.
    pub fn ban_path(&self) -> PathBuf {
        if self.ban_file.is_empty() { self.state_path().join("bans.ron") } else { PathBuf::from(&self.ban_file) }
    }

    /// The state folder: persistent objects and players, bans, identity.
    pub fn state_path(&self) -> PathBuf {
        if !self.state_dir.is_empty() {
            return PathBuf::from(&self.state_dir);
        }
        match self.pak_path() {
            // Next to the pak, never inside the extraction (which is replaced on an upgrade).
            Some(pak) if self.project_dir().is_file() || pak.starts_with(self.project_dir()) => pak.parent().map(|d| d.join(".server")).unwrap_or_else(|| PathBuf::from(".server")),
            _ => self.project_dir().join(".server"),
        }
    }

    /// The server's signing identity file.
    pub fn identity_path(&self) -> PathBuf {
        if self.identity_file.is_empty() { self.state_path().join("identity.key") } else { PathBuf::from(&self.identity_file) }
    }

    /// Check the settings and the project on disk. Reports every problem at once.
    pub fn validate(&self) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();
        if self.project.trim().is_empty() {
            errors.push("no project: pass --project <dir> or set `project` in server.ron".into());
        } else {
            let dir = self.project_dir();
            if self.pak_path().is_some() {
                // An assets.pak: checked after unpacking (`unpack_pak`).
            } else if !dir.is_dir() {
                errors.push(format!("project folder {} does not exist", dir.display()));
            } else if !dir.join("project.zestengine").is_file() {
                errors.push(format!("{} has no project.zestengine (not a project folder)", dir.display()));
            } else if let Some(scene) = &self.scene {
                if Path::new(scene).is_absolute() || scene.split(['/', '\\']).any(|p| p == "..") {
                    errors.push(format!("scene {scene:?} must be a path inside the project"));
                } else if !dir.join(scene).is_file() {
                    errors.push(format!("scene {scene:?} not found in {}", dir.display()));
                }
            }
        }
        if !(1.0..=240.0).contains(&self.tick_rate) || !self.tick_rate.is_finite() {
            errors.push(format!("tick_rate {} out of range (1 to 240 Hz)", self.tick_rate));
        }
        if !(1..=100).contains(&self.max_catch_up_ticks) {
            errors.push(format!("max_catch_up_ticks {} out of range (1 to 100)", self.max_catch_up_ticks));
        }
        if !(1..=4096).contains(&self.max_players) {
            errors.push(format!("max_players {} out of range (1 to 4096)", self.max_players));
        }
        if self.server_name.trim().is_empty() {
            errors.push("server_name must not be empty".into());
        }
        if self.server_name.len() > 64 {
            errors.push("server_name is longer than 64 bytes".into());
        }
        if tracing_subscriber::EnvFilter::try_new(&self.log_level).is_err() {
            errors.push(format!("log_level {:?} is not a level or a tracing filter", self.log_level));
        }
        if self.log_max_mb == 0 {
            errors.push("log_max_mb must be at least 1".into());
        }
        if !self.stats_interval.is_finite() || self.stats_interval < 0.0 {
            errors.push(format!("stats_interval {} must be 0 or more seconds", self.stats_interval));
        }
        self.validate_services(&mut errors);
        if errors.is_empty() {
            Ok(())
        } else {
            bail!("invalid configuration:\n  - {}", errors.join("\n  - "))
        }
    }
}

impl ServerConfig {
    /// NET-6 settings: persistence, remote admin, directory.
    fn validate_services(&self, errors: &mut Vec<String>) {
        if !self.persist_interval.is_finite() || self.persist_interval < 0.0 {
            errors.push(format!("persist_interval {} must be 0 or more seconds", self.persist_interval));
        }
        if self.persist_keep > 20 {
            errors.push(format!("persist_keep {} out of range (0 to 20)", self.persist_keep));
        }
        let mut seen = std::collections::HashSet::new();
        for (i, k) in self.admin_keys.iter().enumerate() {
            let label = if k.name.is_empty() { format!("admin_keys[{i}]") } else { format!("admin_keys[{i}] ({})", k.name) };
            if k.name.trim().is_empty() {
                errors.push(format!("{label}: name must not be empty"));
            }
            match crate::remote::parse_hex32(&k.key) {
                None => errors.push(format!("{label}: key must be 64 hex digits (zestengine-admin keygen)")),
                Some(key) if !seen.insert(key) => errors.push(format!("{label}: the same key is listed twice")),
                Some(_) => {}
            }
        }
        if self.admin_port.is_some() && self.admin_keys.is_empty() {
            errors.push("admin_port is set but admin_keys is empty: nobody could sign a command".into());
        }
        if self.admin_port.is_some() && self.admin_bind.parse::<std::net::IpAddr>().is_err() {
            errors.push(format!("admin_bind {:?} is not an IP address", self.admin_bind));
        }
        if !self.directory.is_empty() && (!self.directory_interval.is_finite() || !(1.0..=600.0).contains(&self.directory_interval)) {
            errors.push(format!("directory_interval {} out of range (1 to 600 s)", self.directory_interval));
        }
        let mut seen_plugins = std::collections::HashSet::new();
        for (i, p) in self.plugins.iter().enumerate() {
            if p.trim().is_empty() {
                errors.push(format!("plugins[{i}] is empty"));
            } else if !seen_plugins.insert(p.as_str()) {
                errors.push(format!("plugins[{i}]: {p:?} is listed twice"));
            }
        }
        if !self.health_interval.is_finite() || !(0.0..=300.0).contains(&self.health_interval) {
            errors.push(format!("health_interval {} out of range (0 to 300 s)", self.health_interval));
        }
        if !self.metrics_addr.is_empty() {
            if let Err(e) = crate::health::parse_metrics_addr(&self.metrics_addr) {
                errors.push(e);
            }
            if self.health_interval <= 0.0 {
                errors.push("metrics_addr needs health_interval above 0 (it serves the heartbeat's readings)".into());
            }
        }
        if !self.shutdown_redirect.is_empty() && self.shutdown_redirect.contains(char::is_whitespace) {
            errors.push("shutdown_redirect must be a host:port or a room code".into());
        }
    }
}

/// What the command line asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cli {
    pub config: Option<PathBuf>,
    pub project: Option<String>,
    pub scene: Option<String>,
    pub port: Option<u16>,
    pub max_players: Option<u32>,
    pub tick_rate: Option<f32>,
    pub name: Option<String>,
    pub password: Option<String>,
    pub log: Option<String>,
    pub log_level: Option<String>,
    pub no_lan: bool,
    pub no_console: bool,
    pub selftest: bool,
    /// Probe the running server's heartbeat file and exit 0 (alive) or 1.
    pub healthcheck: bool,
    /// Heartbeat age the probe accepts, seconds.
    pub health_max_age: Option<u64>,
    /// Print the effective configuration (RON) and exit.
    pub print_config: bool,
    pub help: bool,
    pub version: bool,
}

pub const USAGE: &str = "\
zestengine-server: ZestEngine dedicated server (no window, GPU or audio device)

USAGE:
    zestengine-server --project <dir> [options]

OPTIONS:
    --project <dir>        project folder (project.zestengine, scenes/ ...)
    --scene <path>         project-relative scene (default: the project's boot scene)
    --port <n>             UDP port (default: the scene's Network Manager port, else 9000)
    --max-players <n>      players allowed at once (default 16)
    --tick-rate <hz>       fixed tick rate (default 30)
    --name <text>          server / session name
    --password <text>      join password (also ZESTENGINE_SERVER_PASSWORD; keeps it out of `ps`)
    --config <server.ron>  settings file; flags override it
    --log <file>           also write the log to this file (size-rotated)
    --log-level <level>    error | warn | info | debug | trace, or a tracing filter
    --no-lan               do not answer LAN discovery queries
    --no-console           do not read admin commands from stdin
    --healthcheck          probe a running server (its heartbeat file under state_dir); exit 0 alive, 1 not
    --health-max-age <s>   heartbeat age --healthcheck accepts (default 30)
    --print-config         print the effective configuration (file + env + flags, secrets hidden) and exit
    --headless-selftest    load the scene, host, join a loopback client, shut down; exit 0 on success
    -h, --help             this text
    -V, --version          version

Admin console (stdin): status, players, kick <id> [reason], say <text>, scene <path>, quit, help.
SIGTERM / SIGINT / Ctrl+C shut the server down gracefully (players are told, exit code 0).
";

/// Parse `args` (without the program name). `--flag value` and `--flag=value` both work.
pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (flag, mut inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            inline.take().or_else(|| it.next()).ok_or_else(|| format!("{name} needs a value"))
        };
        fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String> {
            v.parse().map_err(|_| format!("{name}: {v:?} is not a valid number"))
        }
        match flag.as_str() {
            "--project" => cli.project = Some(value("--project")?),
            "--scene" => cli.scene = Some(value("--scene")?),
            "--port" => cli.port = Some(num("--port", &value("--port")?)?),
            "--max-players" => cli.max_players = Some(num("--max-players", &value("--max-players")?)?),
            "--tick-rate" => cli.tick_rate = Some(num("--tick-rate", &value("--tick-rate")?)?),
            "--name" => cli.name = Some(value("--name")?),
            "--password" => cli.password = Some(value("--password")?),
            "--config" => cli.config = Some(PathBuf::from(value("--config")?)),
            "--log" => cli.log = Some(value("--log")?),
            "--log-level" => cli.log_level = Some(value("--log-level")?),
            "--no-lan" => cli.no_lan = true,
            "--no-console" => cli.no_console = true,
            "--headless-selftest" => cli.selftest = true,
            "--healthcheck" => cli.healthcheck = true,
            "--health-max-age" => cli.health_max_age = Some(num("--health-max-age", &value("--health-max-age")?)?),
            "--print-config" => cli.print_config = true,
            "-h" | "--help" => cli.help = true,
            "-V" | "--version" => cli.version = true,
            other => return Err(format!("unknown argument {other:?} (see --help)")),
        }
        if let Some(extra) = inline {
            return Err(format!("{flag} does not take a value (got {extra:?})"));
        }
    }
    Ok(cli)
}

impl Cli {
    /// The settings file (if `--config`), then environment, then flags on top; validated.
    pub fn resolve(&self) -> Result<ServerConfig> {
        let mut cfg = match &self.config {
            Some(path) => ServerConfig::load(path)?,
            None => ServerConfig::default(),
        };
        if let Ok(p) = std::env::var("ZESTENGINE_SERVER_PASSWORD") {
            cfg.password = p;
        }
        self.apply(&mut cfg);
        cfg.validate()?;
        Ok(cfg)
    }

    /// The configuration as `resolve` builds it, without validating (for `--healthcheck`, which
    /// must work in a container that does not carry the project folder).
    pub fn resolve_lenient(&self) -> Result<ServerConfig> {
        let mut cfg = match &self.config {
            Some(path) => ServerConfig::load(path)?,
            None => ServerConfig::default(),
        };
        self.apply(&mut cfg);
        Ok(cfg)
    }

    /// Flags over the file's values (no validation).
    pub fn apply(&self, cfg: &mut ServerConfig) {
        if let Some(v) = &self.project {
            cfg.project = v.clone();
        }
        if let Some(v) = &self.scene {
            cfg.scene = Some(v.clone());
        }
        if let Some(v) = self.port {
            cfg.port = Some(v);
        }
        if let Some(v) = self.max_players {
            cfg.max_players = v;
        }
        if let Some(v) = self.tick_rate {
            cfg.tick_rate = v;
        }
        if let Some(v) = &self.name {
            cfg.server_name = v.clone();
        }
        if let Some(v) = &self.password {
            cfg.password = v.clone();
        }
        if let Some(v) = &self.log {
            cfg.log_path = Some(v.clone());
        }
        if let Some(v) = &self.log_level {
            cfg.log_level = v.clone();
        }
        if self.no_lan {
            cfg.lan_discovery = false;
        }
        if self.no_console {
            cfg.console = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn project_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine-server-cfg-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(dir.join("scenes")).unwrap();
        std::fs::write(dir.join("project.zestengine"), "()").unwrap();
        std::fs::write(dir.join("scenes/a.scene"), "").unwrap();
        dir
    }

    #[test]
    fn empty_file_gives_defaults_and_partial_files_fill_in() {
        assert_eq!(ServerConfig::from_ron_str("()").unwrap(), ServerConfig::default());
        let c = ServerConfig::from_ron_str("(project: \"p\", port: 9100, tick_rate: 60.0, scene: \"scenes/a.scene\", lan_discovery: false)").unwrap();
        assert_eq!((c.port, c.tick_rate, c.lan_discovery, c.max_players), (Some(9100), 60.0, false, 16));
        assert_eq!(c.scene.as_deref(), Some("scenes/a.scene"));
        // The shipped example parses (and stays in sync with the defaults).
        let ex = ServerConfig::from_ron_str(include_str!("../server.example.ron")).unwrap();
        assert_eq!(ex.tick_rate, 30.0);
    }

    #[test]
    fn unknown_fields_and_bad_syntax_are_errors() {
        assert!(ServerConfig::from_ron_str("(tick_rat: 30.0)").is_err());
        assert!(ServerConfig::from_ron_str("(tick_rate: \"fast\")").is_err());
        assert!(ServerConfig::from_ron_str("(").is_err());
    }

    #[test]
    fn flags_parse_in_both_forms() {
        let cli = parse_args(args("--project p --port=9001 --max-players 4 --tick-rate 20 --name=Arena --password hunter2 --no-lan --log s.log --log-level debug --scene scenes/a.scene --headless-selftest")).unwrap();
        assert_eq!(cli.project.as_deref(), Some("p"));
        assert_eq!((cli.port, cli.max_players, cli.tick_rate), (Some(9001), Some(4), Some(20.0)));
        assert!(cli.no_lan && cli.selftest && !cli.no_console);
        assert_eq!(cli.name.as_deref(), Some("Arena"));
        assert!(parse_args(args("--port")).unwrap_err().contains("needs a value"));
        assert!(parse_args(args("--port abc")).unwrap_err().contains("not a valid number"));
        assert!(parse_args(args("--port 70000")).is_err(), "u16 range");
        assert!(parse_args(args("--bogus")).unwrap_err().contains("unknown argument"));
        assert!(parse_args(args("--no-lan=1")).unwrap_err().contains("does not take a value"));
        assert!(parse_args(args("--help")).unwrap().help);
    }

    #[test]
    fn flags_override_the_file() {
        let dir = project_dir();
        let file = dir.join("server.ron");
        std::fs::write(&file, "(project: \"ignored\", port: 9100, max_players: 8, server_name: \"FromFile\", lan_discovery: true, tick_rate: 45.0)").unwrap();
        let cli = parse_args(vec![
            "--config".into(), file.display().to_string(),
            "--project".into(), dir.display().to_string(),
            "--port".into(), "9200".into(),
            "--no-lan".into(),
        ])
        .unwrap();
        let cfg = cli.resolve().unwrap();
        assert_eq!(cfg.project, dir.display().to_string(), "flag beats file");
        assert_eq!(cfg.port, Some(9200));
        assert!(!cfg.lan_discovery);
        assert_eq!((cfg.max_players, cfg.server_name.as_str(), cfg.tick_rate), (8, "FromFile", 45.0), "file values kept where no flag");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validation_reports_every_problem() {
        let e = ServerConfig::default().validate().unwrap_err().to_string();
        assert!(e.contains("no project"), "{e}");
        let dir = project_dir();
        let ok = ServerConfig { project: dir.display().to_string(), ..Default::default() };
        ok.validate().unwrap();
        ServerConfig { scene: Some("scenes/a.scene".into()), ..ok.clone() }.validate().unwrap();
        let bad = ServerConfig {
            tick_rate: 0.0,
            max_players: 0,
            max_catch_up_ticks: 0,
            log_level: "info,=[".into(),
            scene: Some("scenes/missing.scene".into()),
            server_name: " ".into(),
            ..ok.clone()
        };
        let e = bad.validate().unwrap_err().to_string();
        for needle in ["tick_rate", "max_players", "max_catch_up_ticks", "log_level", "not found", "server_name"] {
            assert!(e.contains(needle), "missing {needle:?} in {e}");
        }
        assert!(ServerConfig { scene: Some("../x.scene".into()), ..ok.clone() }.validate().unwrap_err().to_string().contains("inside the project"));
        let e = ServerConfig { project: dir.join("nope").display().to_string(), ..Default::default() }.validate().unwrap_err().to_string();
        assert!(e.contains("does not exist"), "{e}");
        let e = ServerConfig { project: std::env::temp_dir().display().to_string(), ..Default::default() }.validate().unwrap_err().to_string();
        assert!(e.contains("no project.zestengine"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn interest_section_names_only_what_it_changes() {
        let c = ServerConfig::from_ron_str("(project: \"p\", interest: (enabled: true, radius: 80.0, budget: 50000, tiers: [(distance: 0.0, rate: 0.0), (distance: 40.0, rate: 5.0)]))").unwrap();
        let i = c.interest.unwrap();
        assert!(i.enabled && i.radius == 80.0 && i.budget == 50000 && i.tiers.len() == 2);
        assert_eq!(i.hysteresis, InterestConfig::default().hysteresis, "the rest keeps its default");
        assert!(ServerConfig::from_ron_str("(project: \"p\")").unwrap().interest.is_none());
    }

    #[test]
    fn deployment_settings_validate() {
        let dir = project_dir();
        let ok = ServerConfig { project: dir.display().to_string(), ..Default::default() };
        assert_eq!(ok.health_interval, 5.0);
        assert!(ok.metrics_addr.is_empty(), "the HTTP listener is off by default");
        ServerConfig { metrics_addr: "127.0.0.1:9100".into(), ..ok.clone() }.validate().unwrap();
        let e = ServerConfig { metrics_addr: "0.0.0.0:9100".into(), ..ok.clone() }.validate().unwrap_err().to_string();
        assert!(e.contains("loopback"), "{e}");
        let e = ServerConfig { metrics_addr: "127.0.0.1:9100".into(), health_interval: 0.0, ..ok.clone() }.validate().unwrap_err().to_string();
        assert!(e.contains("health_interval"), "{e}");
        assert!(ServerConfig { health_interval: 1000.0, ..ok }.validate().is_err());
        let cli = parse_args(args("--healthcheck --health-max-age 12 --print-config")).unwrap();
        assert!(cli.healthcheck && cli.print_config && cli.health_max_age == Some(12));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn print_config_hides_secrets_and_parses_back() {
        let c = ServerConfig::from_ron_str("(project: \"p\", port: 9100, password: \"hunter2\", relay_token: \"tok\", state_dir: \"/var/lib/g\", interest: (enabled: true), admin_port: 9001, admin_keys: [(name: \"me\", key: \"00\", role: Admin)])").unwrap();
        let text = c.to_ron_redacted().unwrap();
        assert!(!text.contains("hunter2") && !text.contains("\"tok\""), "{text}");
        assert!(text.contains("state_dir: \"/var/lib/g\"") && text.contains("<set>"), "{text}");
        let back = ServerConfig::from_ron_str(&text).unwrap();
        assert_eq!((back.port, back.admin_port, back.state_dir.as_str()), (Some(9100), Some(9001), "/var/lib/g"));
        assert!(back.interest.unwrap().enabled);
    }

    #[test]
    fn anticheat_section_has_defaults() {
        let c = ServerConfig::from_ron_str("(project: \"p\", anticheat: (provider: \"eos\", require: true))").unwrap();
        let a = c.anticheat.unwrap();
        assert!(a.enabled() && a.require && a.pending_timeout == 30.0);
        assert!(ServerConfig::from_ron_str("(project: \"p\")").unwrap().anticheat.is_none());
    }
}
