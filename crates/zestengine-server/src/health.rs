//! Liveness and metrics for a supervisor (NET-7).
//!
//! The server rewrites a small status file (`<state_dir>/health.status`, `key=value` lines)
//! every `health_interval` seconds and, next to it, a Prometheus text file (`metrics.prom`).
//! `zestengine-server --healthcheck` reads the status file, no network and no keys needed, and
//! exits 0 when it is fresh and says `state=running`, 1 otherwise (Docker `HEALTHCHECK`,
//! systemd `ExecStartPost`, a cron probe). An optional loopback-only HTTP listener
//! (`metrics_addr`, off by default) serves `/health` and `/metrics` for a local scraper.

use std::collections::BTreeMap;
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::config::ServerConfig;

/// File name of the heartbeat inside `state_dir`.
pub const STATUS_FILE: &str = "health.status";
/// File name of the Prometheus text file inside `state_dir`.
pub const METRICS_FILE: &str = "metrics.prom";
/// Default `--health-max-age` for the probe, in seconds.
pub const DEFAULT_MAX_AGE: u64 = 30;

/// One reading of the server's vital signs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HealthSnapshot {
    /// `running` or `stopped`.
    pub state: String,
    pub unix: u64,
    pub pid: u32,
    pub name: String,
    pub game_port: u16,
    pub uptime_s: u64,
    pub ticks: u64,
    pub tick_rate: f32,
    pub tick_mean_ms: f64,
    pub tick_max_ms: f64,
    pub players: usize,
    pub max_players: u32,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub rate_in: f64,
    pub rate_out: f64,
    pub violations: u64,
    pub muted: u64,
    pub interest_enabled: bool,
    pub interest_objects: u64,
    pub interest_relevant_avg: f64,
    pub interest_dropped: u64,
    pub interest_throttled: u64,
    pub bans: usize,
}

impl HealthSnapshot {
    /// The heartbeat file text.
    pub fn status_text(&self) -> String {
        let mut s = String::new();
        let mut kv = |k: &str, v: String| {
            s.push_str(k);
            s.push('=');
            s.push_str(&v);
            s.push('\n');
        };
        kv("state", self.state.clone());
        kv("updated_unix", self.unix.to_string());
        kv("pid", self.pid.to_string());
        kv("name", self.name.replace(['\n', '\r'], " "));
        kv("game_port", self.game_port.to_string());
        kv("uptime_s", self.uptime_s.to_string());
        kv("ticks", self.ticks.to_string());
        kv("tick_rate", self.tick_rate.to_string());
        kv("tick_mean_ms", format!("{:.3}", self.tick_mean_ms));
        kv("tick_max_ms", format!("{:.3}", self.tick_max_ms));
        kv("players", self.players.to_string());
        kv("max_players", self.max_players.to_string());
        kv("bytes_in", self.bytes_in.to_string());
        kv("bytes_out", self.bytes_out.to_string());
        kv("bytes_in_per_s", format!("{:.0}", self.rate_in));
        kv("bytes_out_per_s", format!("{:.0}", self.rate_out));
        kv("violations", self.violations.to_string());
        kv("muted", self.muted.to_string());
        kv("interest_enabled", u8::from(self.interest_enabled).to_string());
        kv("interest_objects", self.interest_objects.to_string());
        kv("interest_relevant_avg", format!("{:.1}", self.interest_relevant_avg));
        kv("interest_held_by_budget", self.interest_dropped.to_string());
        kv("interest_merged", self.interest_throttled.to_string());
        kv("bans", self.bans.to_string());
        s
    }

    /// Prometheus text exposition format (version 0.0.4).
    pub fn prometheus_text(&self) -> String {
        let mut s = String::new();
        let mut metric = |name: &str, kind: &str, help: &str, v: String| {
            s.push_str(&format!("# HELP zestengine_{name} {help}\n# TYPE zestengine_{name} {kind}\nzestengine_{name} {v}\n"));
        };
        metric("up", "gauge", "1 while the server is running.", u8::from(self.state == "running").to_string());
        metric("uptime_seconds", "gauge", "Seconds since the server started.", self.uptime_s.to_string());
        metric("ticks_total", "counter", "Fixed ticks run.", self.ticks.to_string());
        metric("tick_rate_hz", "gauge", "Configured tick rate.", self.tick_rate.to_string());
        metric("tick_ms_mean", "gauge", "Mean tick cost in milliseconds since the last reading.", format!("{:.3}", self.tick_mean_ms));
        metric("tick_ms_max", "gauge", "Slowest tick in milliseconds since the last reading.", format!("{:.3}", self.tick_max_ms));
        metric("players", "gauge", "Players connected.", self.players.to_string());
        metric("max_players", "gauge", "Players allowed.", self.max_players.to_string());
        metric("net_bytes_in_total", "counter", "Bytes received.", self.bytes_in.to_string());
        metric("net_bytes_out_total", "counter", "Bytes sent.", self.bytes_out.to_string());
        metric("net_bytes_in_per_second", "gauge", "Receive rate since the last reading.", format!("{:.0}", self.rate_in));
        metric("net_bytes_out_per_second", "gauge", "Send rate since the last reading.", format!("{:.0}", self.rate_out));
        metric("violations_total", "counter", "Validation violations recorded.", self.violations.to_string());
        metric("muted_total", "counter", "Messages muted by rate limits.", self.muted.to_string());
        metric("interest_enabled", "gauge", "1 when interest management is on.", u8::from(self.interest_enabled).to_string());
        metric("interest_objects", "gauge", "Objects tracked by interest management.", self.interest_objects.to_string());
        metric("interest_relevant_avg", "gauge", "Relevant objects per player on average.", format!("{:.1}", self.interest_relevant_avg));
        metric("interest_held_by_budget", "gauge", "Updates held back by the byte budget.", self.interest_dropped.to_string());
        metric("interest_merged_total", "counter", "Updates merged by distance tiers.", self.interest_throttled.to_string());
        metric("bans", "gauge", "Active bans.", self.bans.to_string());
        s
    }
}

/// Parse a heartbeat file into its pairs.
pub fn parse_status(text: &str) -> BTreeMap<String, String> {
    text.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned())).collect()
}

/// Whether the heartbeat in `text` is alive at `now_unix`: `state=running` and updated within
/// `max_age` seconds. Ok carries a one-line summary, Err the reason.
pub fn check_status(text: &str, now_unix: u64, max_age: u64) -> Result<String, String> {
    let kv = parse_status(text);
    let state = kv.get("state").map(String::as_str).unwrap_or("");
    if state != "running" {
        return Err(format!("state is {state:?}, not running"));
    }
    let updated: u64 = kv.get("updated_unix").and_then(|v| v.parse().ok()).ok_or("no updated_unix in the status file")?;
    let age = now_unix.saturating_sub(updated);
    if age > max_age {
        return Err(format!("stale: last heartbeat {age} s ago (limit {max_age} s)"));
    }
    let get = |k: &str| kv.get(k).map(String::as_str).unwrap_or("?");
    Ok(format!("ok: {} players, tick {} ms, {} ticks, heartbeat {age} s ago", get("players"), get("tick_mean_ms"), get("ticks")))
}

/// Where the heartbeat file lives for this configuration.
pub fn status_path(cfg: &ServerConfig) -> PathBuf {
    cfg.state_path().join(STATUS_FILE)
}

pub fn metrics_path(cfg: &ServerConfig) -> PathBuf {
    cfg.state_path().join(METRICS_FILE)
}

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The probe: read the heartbeat file at `path`. Ok = healthy.
pub fn probe(path: &Path, max_age: u64) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    check_status(&text, now_unix(), max_age)
}

/// Write `text` to `path` through a temp file and a rename, so a reader never sees half of it.
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

/// The two texts the HTTP listener serves, replaced on every heartbeat.
#[derive(Default)]
pub struct Shared {
    pub status: String,
    pub metrics: String,
}

/// The loopback HTTP listener. Dropping it does not stop the thread (it ends with the process).
pub struct MetricsServer {
    pub addr: SocketAddr,
    pub shared: Arc<Mutex<Shared>>,
}

/// Check `metrics_addr`: an IP:port on the loopback interface (the listener has no authentication).
pub fn parse_metrics_addr(s: &str) -> Result<SocketAddr, String> {
    let a: SocketAddr = s.parse().map_err(|_| format!("metrics_addr {s:?} is not an ip:port (e.g. 127.0.0.1:9100)"))?;
    if !a.ip().is_loopback() {
        return Err(format!("metrics_addr {s:?} must be a loopback address (127.0.0.1 or ::1): the listener has no authentication"));
    }
    Ok(a)
}

impl MetricsServer {
    pub fn start(addr: SocketAddr, max_age: u64) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let addr = listener.local_addr()?;
        let shared = Arc::new(Mutex::new(Shared::default()));
        let state = shared.clone();
        std::thread::Builder::new().name("metrics-http".into()).spawn(move || {
            for stream in listener.incoming().flatten() {
                serve_one(stream, &state, max_age);
            }
        })?;
        Ok(Self { addr, shared })
    }

    pub fn publish(&self, snap: &HealthSnapshot) {
        if let Ok(mut s) = self.shared.lock() {
            s.status = snap.status_text();
            s.metrics = snap.prometheus_text();
        }
    }
}

fn serve_one(mut stream: TcpStream, shared: &Mutex<Shared>, max_age: u64) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).unwrap_or(0);
    let head = String::from_utf8_lossy(&buf[..n]);
    let mut parts = head.lines().next().unwrap_or("").split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let (code, body) = response(method, path, shared, max_age);
    let reply = format!("HTTP/1.1 {code}\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    let _ = stream.write_all(reply.as_bytes());
}

/// Status line and body for a request. Pure, so the tests need no socket.
pub fn response(method: &str, path: &str, shared: &Mutex<Shared>, max_age: u64) -> (&'static str, String) {
    if method != "GET" {
        return ("405 Method Not Allowed", "GET only\n".into());
    }
    let (status, metrics) = shared.lock().map(|s| (s.status.clone(), s.metrics.clone())).unwrap_or_default();
    match path.split('?').next().unwrap_or("") {
        "/health" | "/healthz" => match check_status(&status, now_unix(), max_age) {
            Ok(line) => ("200 OK", format!("{line}\n")),
            Err(e) => ("503 Service Unavailable", format!("{e}\n")),
        },
        "/metrics" if !metrics.is_empty() => ("200 OK", metrics),
        "/metrics" => ("503 Service Unavailable", "no reading yet\n".into()),
        _ => ("404 Not Found", "/health or /metrics\n".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> HealthSnapshot {
        HealthSnapshot {
            state: "running".into(),
            unix: now_unix(),
            pid: 42,
            name: "Arena".into(),
            game_port: 9000,
            ticks: 900,
            tick_rate: 30.0,
            tick_mean_ms: 1.25,
            tick_max_ms: 4.5,
            players: 3,
            max_players: 16,
            bytes_in: 1000,
            bytes_out: 5000,
            rate_in: 100.0,
            rate_out: 500.0,
            interest_enabled: true,
            interest_relevant_avg: 8.0,
            bans: 2,
            ..Default::default()
        }
    }

    #[test]
    fn status_round_trips_and_fresh_is_healthy() {
        let s = snap();
        let kv = parse_status(&s.status_text());
        assert_eq!(kv["players"], "3");
        assert_eq!(kv["bans"], "2");
        assert_eq!(kv["tick_mean_ms"], "1.250");
        let ok = check_status(&s.status_text(), s.unix + 5, 30).unwrap();
        assert!(ok.contains("3 players"), "{ok}");
    }

    #[test]
    fn stale_stopped_and_missing_fail() {
        let s = snap();
        assert!(check_status(&s.status_text(), s.unix + 31, 30).unwrap_err().contains("stale"));
        let stopped = HealthSnapshot { state: "stopped".into(), ..s.clone() };
        assert!(check_status(&stopped.status_text(), s.unix, 30).unwrap_err().contains("not running"));
        assert!(check_status("", 0, 30).is_err());
        assert!(check_status("state=running\n", 0, 30).unwrap_err().contains("updated_unix"));
        assert!(probe(Path::new("/nonexistent/zest/health.status"), 30).unwrap_err().contains("cannot read"));
    }

    #[test]
    fn probe_reads_the_file_the_server_writes() {
        let dir = std::env::temp_dir().join(format!("zestengine-health-{}", std::process::id()));
        let path = dir.join(STATUS_FILE);
        write_atomic(&path, &snap().status_text()).unwrap();
        assert!(probe(&path, 30).is_ok());
        let old = HealthSnapshot { unix: now_unix() - 120, ..snap() };
        write_atomic(&path, &old.status_text()).unwrap();
        assert!(probe(&path, 30).is_err());
        assert!(probe(&path, 600).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prometheus_text_has_the_series() {
        let t = snap().prometheus_text();
        for needle in ["zestengine_up 1", "zestengine_players 3", "zestengine_tick_ms_mean 1.250", "zestengine_net_bytes_out_per_second 500", "zestengine_bans 2", "zestengine_interest_relevant_avg 8.0", "# TYPE zestengine_ticks_total counter"] {
            assert!(t.contains(needle), "missing {needle:?}:\n{t}");
        }
    }

    #[test]
    fn metrics_addr_must_be_loopback() {
        assert!(parse_metrics_addr("127.0.0.1:9100").is_ok());
        assert!(parse_metrics_addr("[::1]:9100").is_ok());
        assert!(parse_metrics_addr("0.0.0.0:9100").unwrap_err().contains("loopback"));
        assert!(parse_metrics_addr("10.0.0.5:9100").is_err());
        assert!(parse_metrics_addr("nonsense").is_err());
    }

    #[test]
    fn http_routes_answer() {
        let shared = Mutex::new(Shared::default());
        assert_eq!(response("GET", "/health", &shared, 30).0, "503 Service Unavailable", "no reading yet");
        assert_eq!(response("GET", "/metrics", &shared, 30).0, "503 Service Unavailable");
        let s = snap();
        *shared.lock().unwrap() = Shared { status: s.status_text(), metrics: s.prometheus_text() };
        assert_eq!(response("GET", "/health", &shared, 30).0, "200 OK");
        assert!(response("GET", "/metrics?x=1", &shared, 30).1.contains("zestengine_players 3"));
        assert_eq!(response("GET", "/nope", &shared, 30).0, "404 Not Found");
        assert_eq!(response("POST", "/health", &shared, 30).0, "405 Method Not Allowed");
    }

    #[test]
    fn listener_serves_over_tcp() {
        let m = MetricsServer::start("127.0.0.1:0".parse().unwrap(), 30).unwrap();
        m.publish(&snap());
        let mut c = TcpStream::connect(m.addr).unwrap();
        c.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 200 OK") && out.contains("zestengine_players 3"), "{out}");
    }
}
