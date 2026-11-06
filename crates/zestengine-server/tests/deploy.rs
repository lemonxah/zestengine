//! NET-7: heartbeat, `--healthcheck`, metrics, `--print-config` and the generated production
//! `server.ron`. CPU only, loopback only.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use zestengine_core::ObjectId;
use zestengine_engine::server_deploy::{DeployParams, deploy_files};
use zestengine_server::health::{self, METRICS_FILE, STATUS_FILE};
use zestengine_server::server::ServerHandle;
use zestengine_server::{ServerConfig, remote};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("zestengine-deploy-{}-{}", std::process::id(), ObjectId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        Temp(dir)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn project(parent: &Path) -> PathBuf {
    zestengine_engine::scaffold_project(parent, "vr_social", "vr_social").expect("scaffold template")
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

#[test]
fn heartbeat_probe_metrics_and_stop_state() {
    let tmp = Temp::new();
    let root = project(&tmp.0);
    let state = tmp.0.join("state");
    let http_port = free_port();
    let cfg = ServerConfig {
        project: root.display().to_string(),
        port: Some(0),
        tick_rate: 60.0,
        lan_discovery: false,
        console: false,
        stats_interval: 0.0,
        state_dir: state.display().to_string(),
        health_interval: 0.2,
        metrics_addr: format!("127.0.0.1:{http_port}"),
        ..Default::default()
    };
    let status = state.join(STATUS_FILE);
    let server = ServerHandle::spawn(cfg).unwrap();
    wait_until("a heartbeat", || health::probe(&status, 30).is_ok());
    let kv = health::parse_status(&std::fs::read_to_string(&status).unwrap());
    assert_eq!(kv["state"], "running");
    assert_eq!(kv["players"], "0");
    assert_eq!(kv["max_players"], "16");
    assert!(kv["pid"].parse::<u32>().unwrap() > 0);
    wait_until("ticks counted", || health::parse_status(&std::fs::read_to_string(&status).unwrap())["ticks"].parse::<u64>().unwrap() > 10);
    let prom = std::fs::read_to_string(state.join(METRICS_FILE)).unwrap();
    assert!(prom.contains("zestengine_up 1") && prom.contains("zestengine_bans 0"), "{prom}");

    // The loopback HTTP listener.
    let get = |path: &str| {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", http_port)).unwrap();
        c.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes()).unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        out
    };
    let h = get("/health");
    assert!(h.starts_with("HTTP/1.1 200"), "{h}");
    assert!(get("/metrics").contains("zestengine_tick_ms_mean"), "metrics served");

    // The command-line probe agrees (library + binary).
    let cfg_file = tmp.0.join("probe.ron");
    std::fs::write(&cfg_file, format!("(state_dir: \"{}\")", state.display())).unwrap();
    let probe = |extra: &[&str]| Command::new(env!("CARGO_BIN_EXE_zestengine-server")).args(["--healthcheck", "--config"]).arg(&cfg_file).args(extra).output().unwrap();
    let ok = probe(&[]);
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stderr));
    assert!(String::from_utf8_lossy(&ok.stdout).starts_with("ok:"));

    server.shutdown(Duration::from_secs(5)).unwrap();
    let kv = health::parse_status(&std::fs::read_to_string(&status).unwrap());
    assert_eq!(kv["state"], "stopped", "a clean stop says so");
    let bad = probe(&[]);
    assert_eq!(bad.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("not running"));
}

#[test]
fn healthcheck_fails_without_a_server_and_with_bad_arguments() {
    let tmp = Temp::new();
    let cfg_file = tmp.0.join("none.ron");
    std::fs::write(&cfg_file, format!("(state_dir: \"{}\")", tmp.0.join("empty").display())).unwrap();
    let bin = env!("CARGO_BIN_EXE_zestengine-server");
    let out = Command::new(bin).args(["--healthcheck", "--config"]).arg(&cfg_file).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
    let out = Command::new(bin).args(["--healthcheck", "--config", "/no/such/server.ron"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let off = tmp.0.join("off.ron");
    std::fs::write(&off, "(health_interval: 0.0)").unwrap();
    let out = Command::new(bin).args(["--healthcheck", "--config"]).arg(&off).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn print_config_dumps_the_effective_settings_without_secrets() {
    let tmp = Temp::new();
    let root = project(&tmp.0);
    let file = tmp.0.join("server.ron");
    std::fs::write(&file, format!("(project: \"{}\", port: 9100, password: \"hunter2\", state_dir: \"/var/lib/g\")", root.display())).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_zestengine-server"))
        .args(["--print-config", "--no-console", "--max-players", "7", "--config"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(!text.contains("hunter2") && text.contains("<set>"), "{text}");
    let back = ServerConfig::from_ron_str(&text).unwrap();
    assert_eq!((back.port, back.max_players, back.console, back.state_dir.as_str()), (Some(9100), 7, false, "/var/lib/g"), "flags are applied");
}

#[test]
fn generated_production_config_parses_validates_and_has_unusable_placeholders() {
    let tmp = Temp::new();
    let root = project(&tmp.0);
    let files = deploy_files(&DeployParams::new("Vr Social", Some("scenes/main.scene"), 9000));
    let text = files.iter().find(|f| f.name == "server.ron").unwrap().text.clone();
    let cfg = ServerConfig::from_ron_str(&text).unwrap();
    assert_eq!(cfg.state_dir, "/var/lib/vr-social");
    assert_eq!(cfg.port, Some(9000));
    assert_eq!(cfg.project, "/opt/vr-social/assets");
    assert!(!cfg.lan_discovery && cfg.health_interval == 5.0 && cfg.metrics_addr.is_empty() && cfg.admin_port.is_none());
    // Valid once the project exists where the unit expects it (here: pointed at the scaffold).
    ServerConfig { project: root.display().to_string(), ..cfg }.validate().unwrap();
    // Uncommented as shipped, the admin block is refused: a placeholder is not a key.
    assert!(remote::parse_hex32("PUT_THE_64_HEX_DIGIT_PUBLIC_KEY_HERE").is_none());
    let uncommented: String = text
        .lines()
        .map(|l| l.trim_start().strip_prefix("// ").map(str::trim).filter(|r| r.starts_with("admin_") || r.starts_with("(name:") || r.starts_with("],")).map_or(l.to_string(), |r| format!("    {r}")))
        .collect::<Vec<_>>()
        .join("\n");
    let cfg = ServerConfig::from_ron_str(&uncommented).unwrap();
    assert_eq!(cfg.admin_keys.len(), 2);
    let err = ServerConfig { project: root.display().to_string(), ..cfg }.validate().unwrap_err().to_string();
    assert!(err.contains("64 hex digits"), "{err}");
}
