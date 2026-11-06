//! `zestengine-relay`: the deployable relay server.
//!
//! ```text
//! zestengine-relay [--bind 0.0.0.0:47800] [--max-rooms 4096] [--max-members 64] [--idle 30]
//!              [--token SECRET] [--rooms-per-ip 8] [--max-pending 16] [--max-pending-total 1024]
//!              [--orphan-grace 30] [--host-rate 4000000] [--client-rate 1000000]
//!              [--directory [--dir-ttl 90] [--dir-max 4096] [--dir-per-ip 4] [--dir-allow KEYHEX]...]
//! ```

use std::sync::atomic::AtomicBool;

use zestengine_relay::{Relay, RelayConfig, proto};

const USAGE: &str = "zestengine-relay [--bind ADDR] [--max-rooms N] [--max-members N] [--idle SECONDS]
             [--token SECRET] [--rooms-per-ip N] [--max-pending N] [--max-pending-total N]
             [--orphan-grace SECONDS] [--host-rate BYTES] [--client-rate BYTES]
             [--directory] [--dir-ttl SECONDS] [--dir-max N] [--dir-per-ip N] [--dir-allow KEYHEX]

  --token SECRET        hosts must know this operator token to open rooms
                        (also ZESTENGINE_RELAY_TOKEN); joining needs no token
  --rooms-per-ip N      rooms one IP address may hold at once (default 8)
  --max-pending N       joiners per room that haven't exchanged data yet (default 16)
  --max-pending-total N the same over the whole relay (default 1024)
  --orphan-grace S      seconds a room stays open after its host left, for a
                        successor to take it over (host migration, default 30)
  --host-rate BYTES     relayed bytes per second a room host may send (default 4000000);
                        raise it for rooms with many players
  --client-rate BYTES   relayed bytes per second one client may send (default 1000000)
  --directory           also run a server directory: dedicated servers announce with a
                        signed packet, players list them with filters
  --dir-ttl SECONDS     a listing expires this long after the last announcement (default 90)
  --dir-max N           listings at most (default 4096)
  --dir-per-ip N        listings from one IP address (default 4)
  --dir-allow KEYHEX    only this server public key (64 hex digits) may list; repeat for
                        several (default: any key)";

fn main() {
    let mut bind = format!("0.0.0.0:{}", proto::DEFAULT_PORT);
    let mut cfg = RelayConfig { token: std::env::var("ZESTENGINE_RELAY_TOKEN").ok().filter(|t| !t.is_empty()), ..Default::default() };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().unwrap_or_else(|| fail(&format!("{name} needs a value")));
        match a.as_str() {
            "--bind" => bind = value("--bind"),
            "--max-rooms" => cfg.max_rooms = number(&value("--max-rooms")),
            "--max-members" => cfg.max_members = number(&value("--max-members")),
            "--idle" => cfg.idle_timeout = number::<u32>(&value("--idle")) as f64,
            "--token" => cfg.token = Some(value("--token")).filter(|t| !t.is_empty()),
            "--rooms-per-ip" => cfg.rooms_per_ip = number(&value("--rooms-per-ip")),
            "--max-pending" => cfg.max_pending = number(&value("--max-pending")),
            "--max-pending-total" => cfg.max_pending_total = number(&value("--max-pending-total")),
            "--orphan-grace" => cfg.orphan_grace = number::<u32>(&value("--orphan-grace")) as f64,
            "--host-rate" => cfg.host_rate = number::<f64>(&value("--host-rate")),
            "--client-rate" => cfg.client_rate = number::<f64>(&value("--client-rate")),
            "--directory" => {
                cfg.directory.get_or_insert_with(Default::default);
            }
            "--dir-ttl" => cfg.directory.get_or_insert_with(Default::default).ttl = number::<u32>(&value("--dir-ttl")) as f64,
            "--dir-max" => cfg.directory.get_or_insert_with(Default::default).max_servers = number(&value("--dir-max")),
            "--dir-per-ip" => cfg.directory.get_or_insert_with(Default::default).per_ip = number(&value("--dir-per-ip")),
            "--dir-allow" => {
                let v = value("--dir-allow");
                let key = parse_key(&v).unwrap_or_else(|| fail(&format!("not a 64-digit hex key: {v:?}")));
                cfg.directory.get_or_insert_with(Default::default).allow.push(key);
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                println!("Rendezvous + relay server for zestengine internet sessions (UDP, default port {}).", proto::DEFAULT_PORT);
                return;
            }
            other => fail(&format!("unknown argument {other:?} (try --help)")),
        }
    }
    let mut relay = match Relay::bind(&bind, cfg.clone()) {
        Ok(r) => r,
        Err(e) => fail(&format!("bind {bind}: {e}")),
    };
    println!(
        "zestengine-relay listening on {} (protocol {}{})",
        relay.local_addr().map(|a| a.to_string()).unwrap_or(bind),
        proto::RELAY_PROTOCOL,
        if cfg.token.is_some() { ", operator token required to open rooms" } else { "" }
    );
    if let Some(d) = &cfg.directory {
        println!("server directory on: listings expire after {} s, at most {} ({} per IP){}", d.ttl, d.max_servers, d.per_ip, if d.allow.is_empty() { String::new() } else { format!(", {} allowed key(s)", d.allow.len()) });
    }
    let stop = AtomicBool::new(false);
    if let Err(e) = relay.run(&stop) {
        fail(&format!("relay stopped: {e}"));
    }
}

fn parse_key(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn number<T: std::str::FromStr>(s: &str) -> T {
    s.parse().unwrap_or_else(|_| fail(&format!("not a number: {s:?}")))
}

fn fail(msg: &str) -> ! {
    eprintln!("zestengine-relay: {msg}");
    std::process::exit(2)
}
