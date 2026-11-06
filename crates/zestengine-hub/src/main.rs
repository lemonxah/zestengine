//! zestengine-hub: the Project Hub. Lists recent projects, creates new ones from
//! the editor's templates and opens them in the editor (`zestengine <dir>`), which
//! runs as its own process.

mod app;
mod folder_picker;
mod hub;
mod settings;
mod thumbnails;
#[cfg(target_os = "linux")]
mod tray;

fn main() -> anyhow::Result<()> {
    if std::env::args().skip(1).any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: zestengine-hub\n\nThe ZestEngine Project Hub. The editor is found next to this executable, or set {} / the path in Settings.",
            zestengine_assets::launcher::EDITOR_ENV
        );
        return Ok(());
    }
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    app::run()
}
