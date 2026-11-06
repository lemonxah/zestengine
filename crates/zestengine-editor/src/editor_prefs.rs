//! User-level editor preferences (`ZestEngine/editor.ron` in the user's config
//! folder, next to `body.ron` and the hub's `hub.ron`). Per user and machine,
//! never in the project file: today only whether the editor itself starts
//! OpenXR (View > VR), which the project's "Start in VR" otherwise decides.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

pub const PREFS_FILE: &str = "editor.ron";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EditorPrefs {
    /// The editor's own VR: `Some(true)` starts OpenXR at launch, `Some(false)`
    /// never does, `None` follows the project's "Start in VR". `ZESTENGINE_VR`
    /// and the machine-wide off switch (`zestengine_xr::vr_disabled_globally`)
    /// come first either way.
    pub vr: Option<bool>,
    /// Code editor text zoom (Ctrl+scroll), shared by every code editor and
    /// saved globally (not per project). Defaults to 1.0 via the manual impl
    /// (the derive would give 0.0, a zero-size font).
    pub code_zoom: f32,
}

impl Default for EditorPrefs {
    fn default() -> Self {
        Self { vr: None, code_zoom: 1.0 }
    }
}

impl EditorPrefs {
    /// `<config>/ZestEngine/editor.ron` (see `input_engine::user_config_dir`).
    pub fn default_path() -> Option<PathBuf> {
        zestengine_engine::input_engine::user_config_dir().map(|d| d.join("ZestEngine").join(PREFS_FILE))
    }

    /// Read the preferences; a missing or unreadable file gives the defaults.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        ron::from_str(&text).unwrap_or_else(|e| {
            tracing::warn!("editor preferences: ignoring unreadable {}: {e}", path.display());
            Self::default()
        })
    }

    /// Write the preferences (via a temp file, so a crash never leaves them half written).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        let tmp = path.with_extension("ron.tmp");
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
    }
}

/// Whether the editor should try to start OpenXR at launch, first match wins:
/// an explicit `ZESTENGINE_VR` (`1` / `true` / `on` / `yes` forces it, `0` /
/// `false` / `off` / `no` forbids it), else the machine-wide off switch
/// (`globally_off`, the marker file or the same variable; `XrContext::start`
/// enforces it too), else the editor preference, else the project's "Start in
/// VR". Pure, so the precedence is testable.
pub fn vr_wanted_at_start(env: Option<&str>, globally_off: bool, pref: Option<bool>, project_vr: bool) -> bool {
    match env.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ if globally_off => false,
        _ => pref.unwrap_or(project_vr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vr_start_precedence_is_env_then_global_switch_then_preference_then_project() {
        // Nothing set: the project decides.
        assert!(!vr_wanted_at_start(None, false, None, false), "a flat project never touches the OpenXR loader");
        assert!(vr_wanted_at_start(None, false, None, true));
        // The editor preference overrides the project both ways.
        assert!(vr_wanted_at_start(None, false, Some(true), false));
        assert!(!vr_wanted_at_start(None, false, Some(false), true));
        // The machine-wide switch beats the preference and the project.
        assert!(!vr_wanted_at_start(None, true, Some(true), true));
        // An explicit ZESTENGINE_VR beats everything (the marker still keeps
        // `XrContext::start` flat, but the request itself is honoured).
        assert!(vr_wanted_at_start(Some("1"), false, Some(false), false) && vr_wanted_at_start(Some(" On "), true, None, false), "forced on");
        assert!(!vr_wanted_at_start(Some("0"), false, Some(true), true) && !vr_wanted_at_start(Some("false"), false, None, true), "forced off");
        // Empty / unknown values do not count as explicit.
        assert!(vr_wanted_at_start(Some(""), false, None, true) && !vr_wanted_at_start(Some("maybe"), false, Some(false), true));
    }

    #[test]
    fn prefs_round_trip_and_tolerate_bad_files() {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zestengine-editor-prefs-{}-{n}", std::process::id()));
        let path = dir.join(PREFS_FILE);
        assert_eq!(EditorPrefs::load(&path), EditorPrefs::default(), "missing file: defaults (VR follows the project)");
        for vr in [Some(true), Some(false), None] {
            let p = EditorPrefs { vr, ..EditorPrefs::default() };
            p.save(&path).unwrap();
            assert_eq!(EditorPrefs::load(&path), p);
        }
        // An older file without the field, and garbage, both read as defaults.
        std::fs::write(&path, "()").unwrap();
        assert_eq!(EditorPrefs::load(&path), EditorPrefs::default());
        std::fs::write(&path, "{").unwrap();
        assert_eq!(EditorPrefs::load(&path), EditorPrefs::default());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
