//! The recent-projects list (`ZestEngine/projects.ron` in the user's config
//! folder). The Project Hub shows it; the hub and the editor both record every
//! project they open, so one opened from the command line shows up too.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::project_file::PROJECT_FILE_NAME;

pub const RECENTS_FILE: &str = "projects.ron";
/// Unpinned entries kept; the oldest beyond this drop off. Pinned ones stay.
pub const MAX_UNPINNED: usize = 40;

/// The user's config folder (`%APPDATA%`, `~/Library/Application Support`,
/// `$XDG_CONFIG_HOME` or `~/.config`). `ZESTENGINE_CONFIG_DIR` replaces it.
pub fn user_config_dir() -> Option<PathBuf> {
    std::env::var_os("ZESTENGINE_CONFIG_DIR").map(PathBuf::from).or_else(platform_config_dir)
}

/// The platform's config folder, ignoring `ZESTENGINE_CONFIG_DIR`.
pub fn platform_config_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    }
}

/// `<config>/ZestEngine`: editor-wide settings shared by the hub and the editor.
pub fn zestengine_config_dir() -> Option<PathBuf> {
    user_config_dir().map(|d| d.join("ZestEngine"))
}

/// Seconds since the Unix epoch (0 if the clock is before it).
pub fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Absolute, symlink-free form of `path` for comparing entries. Falls back to
/// the path as given when it no longer exists (a missing project).
pub fn normalize_path(path: &Path) -> PathBuf {
    let abs = match std::fs::canonicalize(path) {
        Ok(p) => p,
        Err(_) if path.is_absolute() => path.to_path_buf(),
        Err(_) => std::env::current_dir().map(|d| d.join(path)).unwrap_or_else(|_| path.to_path_buf()),
    };
    strip_verbatim(abs)
}

/// Drop the `\\?\` prefix Windows' canonicalize adds to local paths, so the
/// list shows (and other tools accept) the ordinary form.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => p,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecentProject {
    pub path: PathBuf,
    pub name: String,
    /// Unix seconds of the last open.
    pub last_opened: u64,
    /// Version of the editor that last opened it.
    pub engine_version: String,
    pub pinned: bool,
}

impl Default for RecentProject {
    fn default() -> Self {
        Self { path: PathBuf::new(), name: String::new(), last_opened: 0, engine_version: String::new(), pinned: false }
    }
}

impl RecentProject {
    /// The folder or its project.zestengine is gone (moved, deleted, unmounted).
    pub fn is_missing(&self) -> bool {
        !self.path.join(PROJECT_FILE_NAME).is_file()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecentSort {
    /// Most recently opened first.
    #[default]
    LastOpened,
    Name,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecentProjects {
    pub projects: Vec<RecentProject>,
    /// The project opened most recently (by the hub or the editor).
    pub last_opened: Option<PathBuf>,
}

impl RecentProjects {
    /// `<config>/ZestEngine/projects.ron`.
    pub fn default_path() -> Option<PathBuf> {
        zestengine_config_dir().map(|d| d.join(RECENTS_FILE))
    }

    /// Read the list; a missing or unreadable file gives an empty list.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        ron::from_str(&text).unwrap_or_else(|e| {
            tracing::warn!("recent projects: ignoring unreadable {}: {e}", path.display());
            Self::default()
        })
    }

    /// Write the list (via a temp file, so a crash never leaves it half written).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        let tmp = path.with_extension("ron.tmp");
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn index_of(&self, path: &Path) -> Option<usize> {
        let key = normalize_path(path);
        self.projects.iter().position(|p| p.path == key || p.path == path)
    }

    pub fn find(&self, path: &Path) -> Option<&RecentProject> {
        self.index_of(path).map(|i| &self.projects[i])
    }

    /// Record an open: updates the existing entry (keeping its pin, and its
    /// version when `engine_version` is empty) or adds one, marks it as the last
    /// opened project and trims old unpinned entries.
    pub fn record(&mut self, path: &Path, name: &str, engine_version: &str, now: u64) {
        let key = normalize_path(path);
        match self.index_of(&key) {
            Some(i) => {
                let p = &mut self.projects[i];
                p.path = key.clone();
                p.name = name.to_string();
                if !engine_version.is_empty() {
                    p.engine_version = engine_version.to_string();
                }
                p.last_opened = now;
            }
            None => self.projects.push(RecentProject {
                path: key.clone(),
                name: name.to_string(),
                last_opened: now,
                engine_version: engine_version.to_string(),
                pinned: false,
            }),
        }
        self.last_opened = Some(key);
        self.trim();
    }

    fn trim(&mut self) {
        let mut unpinned: Vec<u64> = self.projects.iter().filter(|p| !p.pinned).map(|p| p.last_opened).collect();
        if unpinned.len() <= MAX_UNPINNED {
            return;
        }
        unpinned.sort_unstable_by(|a, b| b.cmp(a));
        let cutoff = unpinned[MAX_UNPINNED - 1];
        let mut kept = 0;
        self.projects.retain(|p| {
            if p.pinned {
                return true;
            }
            let keep = p.last_opened >= cutoff && kept < MAX_UNPINNED;
            kept += keep as usize;
            keep
        });
    }

    /// Pin or unpin; false when the project isn't in the list.
    pub fn set_pinned(&mut self, path: &Path, pinned: bool) -> bool {
        match self.index_of(path) {
            Some(i) => {
                self.projects[i].pinned = pinned;
                true
            }
            None => false,
        }
    }

    /// Drop an entry (the project itself is untouched).
    pub fn remove(&mut self, path: &Path) -> bool {
        let Some(i) = self.index_of(path) else { return false };
        let removed = self.projects.remove(i);
        if self.last_opened.as_ref() == Some(&removed.path) {
            self.last_opened = None;
        }
        true
    }

    /// Point a (missing) entry at the folder it moved to.
    pub fn relocate(&mut self, old: &Path, new: &Path) -> bool {
        let Some(i) = self.index_of(old) else { return false };
        let key = normalize_path(new);
        if let Some(j) = self.index_of(&key).filter(|&j| j != i) {
            let pinned = self.projects[i].pinned || self.projects[j].pinned;
            self.projects[j].pinned = pinned;
            self.projects.remove(i);
            return true;
        }
        if self.last_opened.as_ref() == Some(&self.projects[i].path) {
            self.last_opened = Some(key.clone());
        }
        self.projects[i].path = key;
        true
    }

    /// Remove every entry whose folder is gone; returns how many went.
    pub fn prune_missing(&mut self) -> usize {
        let before = self.projects.len();
        self.projects.retain(|p| !p.is_missing());
        if self.last_opened.as_ref().is_some_and(|l| !self.projects.iter().any(|p| &p.path == l)) {
            self.last_opened = None;
        }
        before - self.projects.len()
    }

    /// Entries for display: pinned first, then by `sort`; `filter` matches the
    /// name or path, case-insensitively.
    pub fn sorted(&self, sort: RecentSort, filter: &str) -> Vec<&RecentProject> {
        let needle = filter.trim().to_lowercase();
        let mut out: Vec<&RecentProject> = self
            .projects
            .iter()
            .filter(|p| {
                needle.is_empty()
                    || p.name.to_lowercase().contains(&needle)
                    || p.path.to_string_lossy().to_lowercase().contains(&needle)
            })
            .collect();
        out.sort_by(|a, b| {
            b.pinned.cmp(&a.pinned).then_with(|| match sort {
                RecentSort::LastOpened => b.last_opened.cmp(&a.last_opened),
                RecentSort::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            })
        });
        out
    }
}

/// Record that `root` was opened, in the shared list at the default path.
pub fn record_opened(root: &Path, name: &str, engine_version: &str) -> Result<()> {
    let path = RecentProjects::default_path().context("no user config folder")?;
    let mut recents = RecentProjects::load(&path);
    recents.record(root, name, engine_version, now_unix());
    recents.save(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zestengine-recents-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_project(parent: &Path, name: &str) -> PathBuf {
        let root = parent.join(name);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(PROJECT_FILE_NAME), "(name: \"x\")").unwrap();
        root
    }

    #[test]
    fn record_dedupes_and_tracks_last_opened() {
        let dir = temp_dir("dedupe");
        let a = make_project(&dir, "a");
        let b = make_project(&dir, "b");
        let mut r = RecentProjects::default();
        r.record(&a, "A", "0.1.0", 10);
        r.record(&b, "B", "0.1.0", 20);
        // Same folder through a non-normalized path: still one entry.
        r.record(&dir.join("a").join("..").join("a"), "A2", "0.2.0", 30);
        assert_eq!(r.projects.len(), 2);
        // The hub records without a version; the editor's version is kept.
        r.record(&a, "A2", "", 30);
        let e = r.find(&a).unwrap();
        assert_eq!((e.name.as_str(), e.engine_version.as_str(), e.last_opened), ("A2", "0.2.0", 30));
        assert_eq!(r.last_opened.as_deref(), Some(normalize_path(&a).as_path()));
        let order: Vec<&str> = r.sorted(RecentSort::LastOpened, "").iter().map(|p| p.name.as_str()).collect();
        assert_eq!(order, ["A2", "B"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn persists_round_trip() {
        let dir = temp_dir("persist");
        let a = make_project(&dir, "a");
        let file = dir.join("cfg").join(RECENTS_FILE);
        assert_eq!(RecentProjects::load(&file), RecentProjects::default());
        let mut r = RecentProjects::default();
        r.record(&a, "A", "0.1.0", 5);
        r.set_pinned(&a, true);
        r.save(&file).unwrap();
        let back = RecentProjects::load(&file);
        assert_eq!(back, r);
        assert!(back.projects[0].pinned);
        // A corrupt file reads as empty rather than failing.
        std::fs::write(&file, "not ron").unwrap();
        assert!(RecentProjects::load(&file).projects.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pinned_sort_first_and_survive_trim() {
        let dir = temp_dir("pin");
        let mut r = RecentProjects::default();
        let pinned = make_project(&dir, "old");
        r.record(&pinned, "zz old", "0.1.0", 1);
        assert!(r.set_pinned(&pinned, true));
        for i in 0..MAX_UNPINNED + 5 {
            r.record(&dir.join(format!("p{i}")), &format!("p{i}"), "0.1.0", 100 + i as u64);
        }
        assert_eq!(r.projects.iter().filter(|p| !p.pinned).count(), MAX_UNPINNED);
        assert!(r.find(&pinned).is_some(), "pinned entries are never trimmed");
        assert!(r.find(&dir.join("p0")).is_none(), "the oldest unpinned entries go first");
        assert_eq!(r.sorted(RecentSort::LastOpened, "")[0].name, "zz old");
        assert_eq!(r.sorted(RecentSort::Name, "")[0].name, "zz old");
        assert_eq!(r.sorted(RecentSort::Name, "P4").len(), 5, "filter is case-insensitive: p40..p44 (p4 was trimmed)");
        assert!(!r.set_pinned(&dir.join("nope"), true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_projects_are_flagged_relocated_and_pruned() {
        let dir = temp_dir("missing");
        let a = make_project(&dir, "a");
        let mut r = RecentProjects::default();
        r.record(&a, "A", "0.1.0", 1);
        let gone = dir.join("gone");
        r.record(&gone, "Gone", "0.1.0", 2);
        assert!(!r.find(&a).unwrap().is_missing());
        assert!(r.find(&gone).unwrap().is_missing());

        let moved = make_project(&dir, "moved");
        assert!(r.relocate(&gone, &moved));
        assert!(!r.find(&moved).unwrap().is_missing());
        assert_eq!(r.last_opened.as_deref(), Some(normalize_path(&moved).as_path()));

        std::fs::remove_dir_all(&moved).unwrap();
        assert_eq!(r.prune_missing(), 1);
        assert_eq!(r.projects.len(), 1);
        assert_eq!(r.last_opened, None);
        assert!(r.remove(&a));
        assert!(r.projects.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
