//! Hub preferences (`ZestEngine/hub.ron` in the user's config folder): window
//! size, the editor executable override, what happens after a project opens,
//! and the New Project defaults.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use zestengine_assets::recent_projects::{RecentSort, zestengine_config_dir};
use serde::{Deserialize, Serialize};

pub const SETTINGS_FILE: &str = "hub.ron";

/// What the hub does once it has started the editor on a project.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AfterOpen {
    /// Keep the hub window open and interactive.
    #[default]
    KeepOpen,
    /// Stay running, minimised.
    Minimize,
    Close,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HubSettings {
    /// Logical window size, restored on the next start.
    pub window_size: [f32; 2],
    /// Editor executable; empty finds it next to the hub (see
    /// `zestengine_assets::launcher::candidates`).
    pub editor_path: String,
    pub after_open: AfterOpen,
    /// Parent folder offered for new projects.
    pub default_location: String,
    pub sort: RecentSort,
    /// Template picked last time.
    pub last_template: String,
}

impl Default for HubSettings {
    fn default() -> Self {
        Self {
            window_size: [1120.0, 720.0],
            editor_path: String::new(),
            after_open: AfterOpen::default(),
            default_location: default_location().display().to_string(),
            sort: RecentSort::default(),
            last_template: zestengine_assets::project_templates::DEFAULT_TEMPLATE.to_string(),
        }
    }
}

/// `~/ZestEngineProjects`.
fn default_location() -> PathBuf {
    crate::folder_picker::home_dir().join("ZestEngineProjects")
}

impl HubSettings {
    pub fn default_path() -> Option<PathBuf> {
        zestengine_config_dir().map(|d| d.join(SETTINGS_FILE))
    }

    /// Read the settings; a missing or unreadable file gives the defaults.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        ron::from_str(&text).unwrap_or_else(|e| {
            tracing::warn!("hub settings: ignoring unreadable {}: {e}", path.display());
            Self::default()
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
    }

    /// Window size clamped to something usable (a stale file from a bigger
    /// monitor, or a zero size from a minimised close).
    pub fn window_size_clamped(&self) -> [f32; 2] {
        [self.window_size[0].clamp(MIN_SIZE[0], 4096.0), self.window_size[1].clamp(MIN_SIZE[1], 4096.0)]
    }
}

/// Smallest window the layout fits in.
pub const MIN_SIZE: [f32; 2] = [860.0, 560.0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_tolerate_bad_files() {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zestengine-hub-settings-{}-{n}", std::process::id()));
        let path = dir.join(SETTINGS_FILE);
        assert_eq!(HubSettings::load(&path), HubSettings::default());
        let s = HubSettings {
            window_size: [1300.0, 800.0],
            editor_path: "/opt/zestengine/zestengine".into(),
            after_open: AfterOpen::Close,
            sort: RecentSort::Name,
            ..Default::default()
        };
        s.save(&path).unwrap();
        assert_eq!(HubSettings::load(&path), s);
        // Fields added later default; garbage reads as defaults.
        std::fs::write(&path, "(window_size: (900.0, 600.0))").unwrap();
        let partial = HubSettings::load(&path);
        assert_eq!(partial.window_size, [900.0, 600.0]);
        assert_eq!(partial.after_open, AfterOpen::KeepOpen);
        std::fs::write(&path, "{").unwrap();
        assert_eq!(HubSettings::load(&path), HubSettings::default());
        let tiny = HubSettings { window_size: [0.0, 0.0], ..Default::default() };
        assert_eq!(tiny.window_size_clamped(), MIN_SIZE);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
