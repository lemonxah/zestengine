//! Admin key rotation without a restart or a config edit (NET-6): `key add` / `key remove`
//! change who may sign admin commands while the server runs.
//!
//! `server.ron` stays the source of the standing keys. Runtime changes are kept as an overlay in
//! `<state_dir>/admin_keys.ron` (keys added, key ids removed), so they survive a restart and a
//! `reload`, and the effective list is always `(file keys - removed) + added`. The rules that keep
//! an operator from locking themselves out live here and are tested on their own: a key cannot be
//! added twice, and the last Admin key cannot be removed.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::admin::{AdminKey, AdminRole};
use crate::remote::parse_hex32;

/// Runtime changes on top of `server.ron`'s `admin_keys`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyOverlay {
    /// Keys added while running.
    pub added: Vec<AdminKey>,
    /// Public keys (64 lowercase hex digits) removed while running, from the file or the overlay.
    pub removed: Vec<String>,
}

fn norm(hex: &str) -> String {
    hex.trim().to_ascii_lowercase()
}

pub fn parse_role(s: &str) -> Option<AdminRole> {
    match s.to_ascii_lowercase().as_str() {
        "observer" | "obs" => Some(AdminRole::Observer),
        "moderator" | "mod" => Some(AdminRole::Moderator),
        "admin" => Some(AdminRole::Admin),
        _ => None,
    }
}

impl KeyOverlay {
    pub const FILE: &'static str = "admin_keys.ron";

    /// Read the overlay in `dir`; a missing or unreadable file is an empty overlay (logged).
    pub fn load(dir: &Path) -> Self {
        let path = dir.join(Self::FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => ron::from_str(&text).unwrap_or_else(|e| {
                tracing::error!("{}: unreadable ({e}); ignoring the runtime admin key changes", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Write it next to the other state (atomically: a half-written file never replaces a good one).
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).map_err(std::io::Error::other)?;
        let tmp = dir.join(format!("{}.tmp", Self::FILE));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, dir.join(Self::FILE))
    }

    /// `(file keys - removed) + added`, each public key once (a later entry replaces an earlier one).
    pub fn effective(&self, file: &[AdminKey]) -> Vec<AdminKey> {
        let mut out: Vec<AdminKey> = Vec::new();
        for k in file.iter().chain(&self.added) {
            let id = norm(&k.key);
            if self.removed.contains(&id) {
                continue;
            }
            match out.iter_mut().find(|o| norm(&o.key) == id) {
                Some(slot) => *slot = k.clone(),
                None => out.push(k.clone()),
            }
        }
        out
    }

    /// Add a key. Refuses a bad key or role, an empty or odd name and a key already listed.
    pub fn add(&mut self, file: &[AdminKey], name: &str, key: &str, role: AdminRole) -> Result<(), String> {
        let id = norm(key);
        if parse_hex32(&id).is_none() {
            return Err("the key must be 64 hex digits (`zestengine-admin keygen` prints it)".into());
        }
        if name.is_empty() || name.len() > 32 || name.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err("the name is 1 to 32 characters without spaces".into());
        }
        if self.effective(file).iter().any(|k| norm(&k.key) == id) {
            return Err("that key is already listed".into());
        }
        self.removed.retain(|r| *r != id);
        self.added.push(AdminKey { name: name.to_owned(), key: id, role });
        Ok(())
    }

    /// Remove the key named `who` (a name, or 8 or more digits of the key). Never the last Admin.
    /// Returns the removed key.
    pub fn remove(&mut self, file: &[AdminKey], who: &str) -> Result<AdminKey, String> {
        let who = who.trim();
        let digits = norm(who);
        let list = self.effective(file);
        let hits: Vec<&AdminKey> = list.iter().filter(|k| k.name == who || (digits.len() >= 8 && norm(&k.key).starts_with(&digits))).collect();
        let target = match hits.as_slice() {
            [] => return Err(format!("no admin key matches {who:?} (see `keys`)")),
            [one] => (*one).clone(),
            many => return Err(format!("{who:?} matches {} keys; give more digits of the key", many.len())),
        };
        let admins_left = list.iter().filter(|k| k.role == AdminRole::Admin && norm(&k.key) != norm(&target.key)).count();
        if target.role == AdminRole::Admin && admins_left == 0 {
            return Err("that is the last Admin key: add another Admin key first, or nobody could manage the server".into());
        }
        let id = norm(&target.key);
        self.added.retain(|k| norm(&k.key) != id);
        if file.iter().any(|k| norm(&k.key) == id) && !self.removed.contains(&id) {
            self.removed.push(id);
        }
        Ok(target)
    }

    /// Listing for `keys`: name, role, first 16 digits, and where it comes from.
    pub fn describe(&self, file: &[AdminKey]) -> String {
        let list = self.effective(file);
        let rows: Vec<String> = list
            .iter()
            .map(|k| {
                let id = norm(&k.key);
                let from = if self.added.iter().any(|a| norm(&a.key) == id) { "runtime" } else { "server.ron" };
                format!("  {:<20} {:<9} {}...  ({from})", k.name, format!("{:?}", k.role), id.get(..16).unwrap_or(&id))
            })
            .collect();
        if rows.is_empty() { "no admin keys".into() } else { format!("{} admin key(s):\n{}", rows.len(), rows.join("\n")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hexn(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    fn key(name: &str, n: u8, role: AdminRole) -> AdminKey {
        AdminKey { name: name.into(), key: hexn(n), role }
    }

    #[test]
    fn rotation_adds_then_removes_and_the_effective_list_follows() {
        let file = vec![key("ops", 1, AdminRole::Admin), key("mod", 2, AdminRole::Moderator)];
        let mut o = KeyOverlay::default();
        assert_eq!(o.effective(&file), file);
        // Rotate: add the new admin key, then retire the old one.
        o.add(&file, "ops2", &hexn(3).to_uppercase(), AdminRole::Admin).unwrap();
        assert_eq!(o.effective(&file).len(), 3);
        let gone = o.remove(&file, "ops").unwrap();
        assert_eq!(gone.name, "ops");
        let now: Vec<String> = o.effective(&file).iter().map(|k| k.name.clone()).collect();
        assert_eq!(now, ["mod", "ops2"]);
        assert_eq!(o.removed, [hexn(1)], "a file key is removed through the overlay");
        // The same key can come back.
        o.add(&file, "ops-again", &hexn(1), AdminRole::Admin).unwrap();
        assert!(o.removed.is_empty() && o.effective(&file).iter().any(|k| k.name == "ops-again"));
        // Removing a runtime key drops it from `added` and leaves no tombstone.
        o.remove(&file, &hexn(3)[..12]).unwrap();
        assert!(o.added.iter().all(|k| k.name != "ops2") && !o.removed.contains(&hexn(3)));
    }

    #[test]
    fn mistakes_are_refused_and_the_last_admin_stays() {
        let file = vec![key("ops", 1, AdminRole::Admin), key("mod", 2, AdminRole::Moderator)];
        let mut o = KeyOverlay::default();
        assert!(o.add(&file, "x", "abc", AdminRole::Admin).unwrap_err().contains("64 hex"));
        assert!(o.add(&file, "", &hexn(9), AdminRole::Admin).unwrap_err().contains("name"));
        assert!(o.add(&file, "two words", &hexn(9), AdminRole::Admin).unwrap_err().contains("name"));
        assert!(o.add(&file, "dup", &hexn(1), AdminRole::Admin).unwrap_err().contains("already"));
        assert!(o.remove(&file, "nobody").unwrap_err().contains("no admin key"));
        assert!(o.remove(&file, "ops").unwrap_err().contains("last Admin"), "locking everyone out is refused");
        assert!(o.remove(&file, "0101").unwrap_err().contains("no admin key"), "fewer than 8 digits never match");
        o.remove(&file, "mod").unwrap();
        assert_eq!(o.effective(&file).len(), 1);
        // Two keys sharing a prefix: ask for more digits.
        let mut a = hexn(5);
        a.replace_range(8..10, "ff");
        let list = vec![key("ops", 1, AdminRole::Admin), AdminKey { name: "a".into(), key: hexn(5), role: AdminRole::Observer }, AdminKey { name: "b".into(), key: a, role: AdminRole::Observer }];
        assert!(KeyOverlay::default().remove(&list, "05050505").unwrap_err().contains("matches 2"));
    }

    #[test]
    fn the_overlay_survives_a_restart() {
        let dir = std::env::temp_dir().join(format!("zestengine-admin-keys-{}", std::process::id()));
        let file = vec![key("ops", 1, AdminRole::Admin)];
        let mut o = KeyOverlay::default();
        o.add(&file, "ops2", &hexn(2), AdminRole::Admin).unwrap();
        o.remove(&file, "ops").unwrap();
        o.save(&dir).unwrap();
        let back = KeyOverlay::load(&dir);
        assert_eq!(back, o);
        assert_eq!(back.effective(&file).iter().map(|k| k.name.as_str()).collect::<Vec<_>>(), ["ops2"]);
        std::fs::write(dir.join(KeyOverlay::FILE), "not ron {").unwrap();
        assert_eq!(KeyOverlay::load(&dir), KeyOverlay::default(), "a damaged file is ignored, not fatal");
        assert_eq!(KeyOverlay::load(&dir.join("missing")), KeyOverlay::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roles_parse_and_the_listing_names_the_source() {
        assert_eq!(parse_role("Admin"), Some(AdminRole::Admin));
        assert_eq!(parse_role("mod"), Some(AdminRole::Moderator));
        assert_eq!(parse_role("root"), None);
        let file = vec![key("ops", 1, AdminRole::Admin)];
        let mut o = KeyOverlay::default();
        o.add(&file, "temp", &hexn(2), AdminRole::Observer).unwrap();
        let text = o.describe(&file);
        assert!(text.contains("2 admin key(s)") && text.contains("(server.ron)") && text.contains("(runtime)") && !text.contains(&hexn(2)), "{text}");
    }
}
