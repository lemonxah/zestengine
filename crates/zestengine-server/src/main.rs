//! `zestengine-server`: the dedicated server binary. See docs/dedicated-server.md.

fn main() -> std::process::ExitCode {
    zestengine_server::run_main(|_| {})
}
