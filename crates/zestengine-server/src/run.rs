//! The server's `main`, callable from a project's own server crate (NET-3): the
//! stock `zestengine-server` binary calls [`run_main`] with no extra components,
//! a generated project crate passes its `zestengine_register`.

use std::io::BufRead as _;
use std::process::ExitCode;

use zestengine_core::ComponentRegistry;

use crate::admin::AdminCommand;
use crate::config::{USAGE, parse_args};
use crate::server::{RESTART_EXIT_CODE, Remote, Server, control, selftest_with};
use crate::{logging, signals};

/// Parse the command line, start logging and serve until stopped. `register`
/// adds the game's own component types to the built-ins.
pub fn run_main(register: impl FnOnce(&mut ComponentRegistry) + Send + 'static) -> ExitCode {
    // The whole process is a dedicated server: content packages it extracts are server-side too.
    zestengine_core::set_process_role(zestengine_core::NetRole::Server);
    let cli = match parse_args(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if cli.help {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if cli.version {
        println!("zestengine-server {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if cli.healthcheck {
        return healthcheck(&cli);
    }
    let cfg = match cli.resolve() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::from(2);
        }
    };
    if cli.print_config {
        return match cfg.to_ron_redacted() {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::from(2)
            }
        };
    }
    let log_file = match logging::init(&cfg) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::from(2);
        }
    };
    // A panic anywhere: log it (with where), flush the file and leave non-zero.
    let hook_file = log_file.clone();
    std::panic::set_hook(Box::new(move |info| {
        let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let msg = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
        tracing::error!("panic at {at}: {msg}");
        if let Some(f) = &hook_file {
            f.sync();
        }
        std::process::exit(101);
    }));
    if let Err(e) = signals::install() {
        tracing::warn!("could not install signal handlers: {e}");
    }

    let code = if cli.selftest {
        match selftest_with(cfg, register) {
            Ok(summary) => {
                println!("{summary}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                tracing::error!("selftest failed: {e:#}");
                ExitCode::FAILURE
            }
        }
    } else {
        serve(cfg, register, &cli)
    };
    if let Some(f) = &log_file {
        f.sync();
    }
    code
}

/// `--healthcheck`: no logging, no project, just the heartbeat file of the configured state_dir.
/// Exit 0 = alive, 1 = not (stale, stopped, missing), 2 = bad arguments.
fn healthcheck(cli: &crate::Cli) -> ExitCode {
    let cfg = match cli.resolve_lenient() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::from(2);
        }
    };
    let path = crate::health::status_path(&cfg);
    // Three missed heartbeats, but never less than the default allowance.
    let max_age = cli.health_max_age.unwrap_or_else(|| ((cfg.health_interval * 3.0).ceil() as u64).max(crate::health::DEFAULT_MAX_AGE));
    if cfg.health_interval <= 0.0 {
        eprintln!("unhealthy: heartbeats are off (health_interval: 0), there is nothing to check");
        return ExitCode::FAILURE;
    }
    match crate::health::probe(&path, max_age) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("unhealthy: {e}");
            ExitCode::FAILURE
        }
    }
}

fn serve(cfg: crate::ServerConfig, register: impl FnOnce(&mut ComponentRegistry), cli: &crate::Cli) -> ExitCode {
    let console = cfg.console;
    let mut server = match Server::open_with(cfg, register) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cannot start: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    server.set_reload_source(cli.clone());
    let (ctl, remote) = control(true);
    if console {
        spawn_console(remote);
    }
    let reason = server.run(ctl);
    tracing::info!("server stopped ({reason:?})");
    if reason == crate::server::ExitReason::Restart { ExitCode::from(RESTART_EXIT_CODE) } else { ExitCode::SUCCESS }
}

/// Read admin commands from stdin until it closes (a service manager gives no stdin: this ends quietly).
fn spawn_console(remote: Remote) {
    let _ = std::thread::Builder::new().name("admin-console".into()).spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            match AdminCommand::parse(&line) {
                Ok(None) => {}
                Ok(Some(cmd)) => match remote.command(cmd) {
                    Some(out) => println!("{out}"),
                    None => break,
                },
                Err(e) => println!("{e}"),
            }
        }
    });
}
