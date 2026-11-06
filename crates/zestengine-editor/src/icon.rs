//! App icon (the official mark from `crate::icons`) plus desktop integration.
//!
//! X11/Windows take the icon directly from winit. Wayland compositors look
//! it up via the window's `app_id`, which points to a `.desktop` file, so we
//! install `zestengine.desktop` and the icon into the user's XDG data dirs on
//! first run, refreshing older icons when the logo changes.

use std::path::PathBuf;

/// The app icon as (width, height, RGBA8); shared with the Project Hub.
pub fn rgba() -> (u32, u32, Vec<u8>) {
    crate::icons::app_icon_rgba()
}

/// Marks a desktop entry written by [`install_desktop_entry`] (and not by a package or the user).
const GENERATED_COMMENT: &str = "zestengine engine editor";

/// The system data dirs (`XDG_DATA_DIRS`, default `/usr/local/share:/usr/share`).
fn system_data_dirs() -> Vec<PathBuf> {
    let dirs = std::env::var("XDG_DATA_DIRS").ok().filter(|d| !d.is_empty()).unwrap_or_else(|| "/usr/local/share:/usr/share".into());
    dirs.split(':').filter(|d| !d.is_empty()).map(PathBuf::from).collect()
}

/// Whether a packaged / `just install` copy of `rel` (relative to a data dir) exists.
fn installed_system_wide(rel: &str) -> bool {
    system_data_dirs().iter().any(|d| d.join(rel).is_file())
}

/// A user-dir entry this function wrote earlier that should go: ZestEngine is installed system-wide
/// now (the user copy would shadow the packaged one, with a dev-build path and the old name), or the
/// binary it launches no longer exists.
fn stale_generated_entry(text: &str, system_entry: bool) -> bool {
    if !text.contains(&format!("Comment={GENERATED_COMMENT}")) {
        return false;
    }
    let exec = text.lines().find_map(|l| l.strip_prefix("Exec=")).unwrap_or("").trim();
    system_entry || (exec.starts_with('/') && !std::path::Path::new(exec).exists())
}

/// Install `zestengine.desktop` into the user's XDG data dir and refresh the icon, so Wayland
/// compositors (which resolve icons by `app_id`) show ours when ZestEngine runs from a build tree.
/// When it is installed system-wide (package or `just install`) the packaged entry and icons are used:
/// nothing is written, and an entry this function wrote earlier is removed so it stops shadowing them.
/// Best-effort: failures only log.
pub fn install_desktop_entry() {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return;
    };
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".local/share"));
    let system_entry = installed_system_wide("applications/zestengine.desktop");
    let desktop_path = data.join("applications/zestengine.desktop");

    if let Ok(text) = std::fs::read_to_string(&desktop_path)
        && stale_generated_entry(&text, system_entry)
    {
        match std::fs::remove_file(&desktop_path) {
            Ok(()) => tracing::info!("removed stale {}", desktop_path.display()),
            Err(e) => tracing::debug!("removing stale desktop entry: {e}"),
        }
    }
    if system_entry {
        return;
    }

    if let Err(e) = crate::icons::install_app_icon() {
        tracing::debug!("installing app icon: {e}");
    }

    if !desktop_path.exists() {
        let exe = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "zestengine".into());
        let entry = format!(
            "[Desktop Entry]\n\
             Type=Application\n\
             Name=ZestEngine Editor\n\
             Comment={GENERATED_COMMENT}\n\
             Exec={exe}\n\
             Icon=zestengine\n\
             Terminal=false\n\
             Categories=Development;Graphics;3DGraphics;\n\
             StartupWMClass=zestengine\n"
        );
        let write = || -> anyhow::Result<()> {
            if let Some(parent) = desktop_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&desktop_path, entry)?;
            Ok(())
        };
        match write() {
            Ok(()) => tracing::info!("installed {}", desktop_path.display()),
            Err(e) => tracing::debug!("installing desktop entry: {e:#}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_our_own_stale_entries_are_removed() {
        let ours = format!("[Desktop Entry]\nComment={GENERATED_COMMENT}\nExec=/no/such/zestengine-binary\n");
        assert!(stale_generated_entry(&ours, false), "binary gone");
        let live = format!("[Desktop Entry]\nComment={GENERATED_COMMENT}\nExec=/bin/sh\n");
        assert!(!stale_generated_entry(&live, false), "binary present, nothing installed");
        assert!(stale_generated_entry(&live, true), "installed system-wide: the user copy shadows it");
        let users = "[Desktop Entry]\nComment=my own launcher\nExec=/no/such/zestengine-binary\n";
        assert!(!stale_generated_entry(users, true), "never touch entries we did not write");
    }
}
