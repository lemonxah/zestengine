//! `zestengine-admin`: make admin keys and send signed commands to a server's admin port.
//!
//! ```text
//! zestengine-admin keygen [--out admin.key]
//! zestengine-admin --server host:port --server-key HEX --key admin.key <command...>
//! ```
//!
//! `keygen` writes the secret key (owner-only) and prints the public key for `admin_keys` in
//! `server.ron`. The server logs its own key at start ("server key ..."). Commands are the
//! console's: `status`, `players`, `kick`, `ban`, `unban`, `bans`, `say`, `scene`, `save`,
//! `reload`, `restart`, `shutdown` ...; `help` lists them. Exit code 0 when the server ran the
//! command, 1 when it refused or it failed, 2 for a bad command line, 3 when no signed answer came.

use std::net::ToSocketAddrs;
use std::process::ExitCode;
use std::time::Duration;

use ed25519_dalek::SigningKey;
use zestengine_server::remote::{self, Status};

const USAGE: &str = "\
zestengine-admin keygen [--out FILE]
zestengine-admin --server HOST:PORT --server-key HEX --key FILE [--timeout SECONDS] <command...>

  keygen         make an admin key pair: the secret goes to FILE (default admin.key, owner-only),
                 the public key is printed for `admin_keys` in server.ron
  --server       the server's admin_port address
  --server-key   the server's public key, logged at its start (\"server key ...\"); requests are
                 bound to it, and answers are checked against it
  --key          your secret key file (from keygen)
  --timeout      seconds to wait for the signed answer (default 5)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            ExitCode::from(if args.is_empty() { 2 } else { 0 })
        }
        Some("keygen") => keygen(&args[1..]),
        Some(_) => send(&args),
    }
}

fn keygen(args: &[String]) -> ExitCode {
    let mut out = "admin.key".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => match it.next() {
                Some(v) => out = v.clone(),
                None => return bad("--out needs a file"),
            },
            other => return bad(&format!("unknown argument {other:?}")),
        }
    }
    if std::path::Path::new(&out).exists() {
        return bad(&format!("{out} already exists; not overwriting a key"));
    }
    let mut seed = [0u8; 32];
    if getrandom::fill(&mut seed).is_err() {
        return bad("no OS randomness");
    }
    let key = SigningKey::from_bytes(&seed);
    let text = format!("{}\n", seed.iter().map(|b| format!("{b:02x}")).collect::<String>());
    if let Err(e) = std::fs::write(&out, text) {
        return bad(&format!("cannot write {out}: {e}"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o600));
    }
    let public = remote::hex(&key.verifying_key().to_bytes());
    println!("secret key written to {out} (keep it private)\npublic key (put in server.ron):\n  admin_keys: [(name: \"me\", key: \"{public}\", role: Admin)]");
    ExitCode::SUCCESS
}

fn bad(msg: &str) -> ExitCode {
    eprintln!("zestengine-admin: {msg}");
    ExitCode::from(2)
}

fn send(args: &[String]) -> ExitCode {
    let (mut server, mut server_key, mut key_file, mut timeout) = (None, None, None, 5.0f64);
    let mut command = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        let r = match a.as_str() {
            "--server" => value("--server").map(|v| server = Some(v)),
            "--server-key" => value("--server-key").map(|v| server_key = Some(v)),
            "--key" => value("--key").map(|v| key_file = Some(v)),
            "--timeout" => value("--timeout").and_then(|v| v.parse().map(|t| timeout = t).map_err(|_| format!("--timeout: {v:?} is not a number"))),
            _ => {
                command.push(a.clone());
                Ok(())
            }
        };
        if let Err(e) = r {
            return bad(&e);
        }
        // Everything after the first non-flag word belongs to the command.
        if command.len() == 1 {
            command.extend(it.by_ref().cloned());
            break;
        }
    }
    let (Some(server), Some(server_key), Some(key_file)) = (server, server_key, key_file) else { return bad("--server, --server-key and --key are required (see --help)") };
    if command.is_empty() {
        return bad("no command given (try `help`)");
    }
    let Some(server_key) = remote::parse_hex32(&server_key) else { return bad("--server-key must be 64 hex digits") };
    let secret = match std::fs::read_to_string(&key_file).map(|t| remote::parse_hex32(t.trim())) {
        Ok(Some(seed)) => SigningKey::from_bytes(&seed),
        Ok(None) => return bad(&format!("{key_file} is not a key file (64 hex digits)")),
        Err(e) => return bad(&format!("cannot read {key_file}: {e}")),
    };
    let Some(addr) = server.to_socket_addrs().ok().and_then(|mut a| a.next()) else { return bad(&format!("cannot resolve {server:?}")) };
    match remote::send_command(addr, &server_key, &secret, &command.join(" "), Duration::from_secs_f64(timeout.clamp(0.5, 120.0))) {
        Ok(r) => {
            println!("{}", r.text);
            if r.status == Status::Ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Err(e) => {
            eprintln!("zestengine-admin: {e}");
            ExitCode::from(3)
        }
    }
}
