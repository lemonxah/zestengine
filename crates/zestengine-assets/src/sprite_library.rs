//! Sprite libraries and sprite swap (2D-7): `category` / `label` -> sprite
//! tables used for skins, outfits and frame swaps, and the resolver logic that
//! turns an animatable integer into a sprite.
//!
//! * A `.spritelib` file ([`SpriteLibraryAsset`]) lists categories, each with
//!   labelled sprite references (the usual `"path#16-hex"` text). An optional
//!   `primary` path makes it a *variant*: its entries override the primary's
//!   and everything it does not list falls through (Unity: `primaryLibraryGUID`).
//! * A swap is addressed by a **24-bit hash** of `category` + `label`
//!   ([`entry_hash`]). Clips store it as a float key (`TrackProperty::SpriteSwap`),
//!   and 24 bits are exactly representable in an `f32` (Unity masks to 30 bits
//!   and so loses low bits when keyed). Hash 0 means "no entry"; a hash with no
//!   entry leaves the current sprite alone.
//! * [`ResolvedLibrary`] flattens a library (and its primary chain) into a hash
//!   map once; lookups are O(1) and the engine only re-resolves when a
//!   resolver's hash changes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

pub const SPRITE_LIBRARY_EXTENSION: &str = "spritelib";
/// Mask of the swap hash.
pub const HASH_MASK: u32 = zestengine_core::SPRITE_SWAP_HASH_MASK;
/// Longest primary chain followed (guards cycles).
const MAX_CHAIN: usize = 8;

/// FNV-1a of `category`, a separator and `label`, masked to 24 bits, never 0
/// (`zestengine_core::sprite_swap_hash`, so the editor's inspectors agree).
pub fn entry_hash(category: &str, label: &str) -> u32 {
    zestengine_core::sprite_swap_hash(category, label)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LibraryEntry {
    pub label: String,
    /// `"<path>#<16 hex>"`; empty = an explicit "show nothing".
    pub sprite: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LibraryCategory {
    pub name: String,
    pub entries: Vec<LibraryEntry>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteLibraryAsset {
    /// Project-relative path of the library this one varies; empty = none.
    pub primary: String,
    pub categories: Vec<LibraryCategory>,
}

impl SpriteLibraryAsset {
    pub fn set(&mut self, category: &str, label: &str, sprite: &str) {
        let ci = match self.categories.iter().position(|c| c.name == category) {
            Some(i) => i,
            None => {
                self.categories.push(LibraryCategory { name: category.to_owned(), entries: Vec::new() });
                self.categories.len() - 1
            }
        };
        let c = &mut self.categories[ci];
        match c.entries.iter_mut().find(|e| e.label == label) {
            Some(e) => e.sprite = sprite.to_owned(),
            None => c.entries.push(LibraryEntry { label: label.to_owned(), sprite: sprite.to_owned() }),
        }
    }

    pub fn remove(&mut self, category: &str, label: &str) -> bool {
        let Some(c) = self.categories.iter_mut().find(|c| c.name == category) else { return false };
        let before = c.entries.len();
        c.entries.retain(|e| e.label != label);
        c.entries.len() != before
    }

    /// Two different `category` / `label` pairs with the same hash.
    pub fn collisions(&self) -> Vec<(String, String)> {
        let mut seen: HashMap<u32, (&str, &str)> = HashMap::new();
        let mut out = Vec::new();
        for c in &self.categories {
            for e in &c.entries {
                let h = entry_hash(&c.name, &e.label);
                match seen.get(&h) {
                    Some(&(pc, pl)) if (pc, pl) != (c.name.as_str(), e.label.as_str()) => out.push((format!("{pc}/{pl}"), format!("{}/{}", c.name, e.label))),
                    _ => {
                        seen.insert(h, (&c.name, &e.label));
                    }
                }
            }
        }
        out
    }
}

/// A flattened library: hash -> sprite text, with the names kept for the editor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedLibrary {
    map: HashMap<u32, (String, String, String)>,
}

impl ResolvedLibrary {
    /// Flatten `assets` ordered from the most basic (primary) to the variant
    /// itself: later assets override earlier entries of the same
    /// category / label.
    pub fn from_chain(assets: &[&SpriteLibraryAsset]) -> Self {
        let mut map = HashMap::new();
        for a in assets {
            for c in &a.categories {
                for e in &c.entries {
                    map.insert(entry_hash(&c.name, &e.label), (c.name.clone(), e.label.clone(), e.sprite.clone()));
                }
            }
        }
        Self { map }
    }

    /// Sprite text for a hash; `None` when the library has no such entry.
    pub fn sprite(&self, hash: u32) -> Option<&str> {
        self.map.get(&hash).map(|(_, _, s)| s.as_str())
    }

    pub fn names(&self, hash: u32) -> Option<(&str, &str)> {
        self.map.get(&hash).map(|(c, l, _)| (c.as_str(), l.as_str()))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// All `(category, label, hash)` sorted by category then label.
    pub fn entries(&self) -> Vec<(String, String, u32)> {
        let mut v: Vec<_> = self.map.iter().map(|(h, (c, l, _))| (c.clone(), l.clone(), *h)).collect();
        v.sort();
        v
    }
}

/// Load a library and its primary chain (project-relative `primary` paths under
/// `root`) and flatten it. A missing primary is skipped with a warning.
pub fn load_resolved_library(root: &Path, path: &Path) -> Result<ResolvedLibrary> {
    let mut chain: Vec<SpriteLibraryAsset> = Vec::new();
    let mut next = path.to_path_buf();
    for _ in 0..MAX_CHAIN {
        let lib = match load_sprite_library(&next) {
            Ok(l) => l,
            Err(e) if chain.is_empty() => return Err(e),
            Err(e) => {
                tracing::warn!("sprite library primary: {e:#}");
                break;
            }
        };
        let primary = lib.primary.clone();
        chain.push(lib);
        if primary.is_empty() {
            break;
        }
        next = root.join(&primary);
    }
    chain.reverse();
    let refs: Vec<&SpriteLibraryAsset> = chain.iter().collect();
    Ok(ResolvedLibrary::from_chain(&refs))
}

pub fn load_sprite_library(path: impl AsRef<Path>) -> Result<SpriteLibraryAsset> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn save_sprite_library(path: impl AsRef<Path>, lib: &SpriteLibraryAsset) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(lib, ron::ser::PrettyConfig::default())?;
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// What a resolver does when its hash changes: the sprite text to show, or
/// `None` to keep the current one (unknown hash, 0, or the same as before).
pub fn resolve_swap(lib: &ResolvedLibrary, hash: u32, current: &str) -> Option<String> {
    if hash == 0 {
        return None;
    }
    let s = lib.sprite(hash)?;
    (s != current).then(|| s.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_24_bit_nonzero_and_f32_exact() {
        let h = entry_hash("Head", "Happy");
        assert_eq!(h, entry_hash("Head", "Happy"));
        assert!(h != 0 && h <= HASH_MASK);
        assert_eq!(h as f32 as u32, h, "survives a float key");
        assert_ne!(entry_hash("Head", "Sad"), h);
        assert_ne!(entry_hash("HeadH", "appy"), h, "the separator matters");
        assert_ne!(entry_hash("", ""), 0);
    }

    fn lib(primary: &str, entries: &[(&str, &str, &str)]) -> SpriteLibraryAsset {
        let mut l = SpriteLibraryAsset { primary: primary.into(), ..Default::default() };
        for (c, e, s) in entries {
            l.set(c, e, s);
        }
        l
    }

    #[test]
    fn variant_overrides_primary_and_falls_through() {
        let base = lib("", &[("Head", "A", "a.png#0000000000000001"), ("Head", "B", "b.png#0000000000000002")]);
        let var = lib("base.spritelib", &[("Head", "B", "red_b.png#0000000000000003"), ("Hat", "C", "c.png#0000000000000004")]);
        let r = ResolvedLibrary::from_chain(&[&base, &var]);
        assert_eq!(r.len(), 3);
        assert_eq!(r.sprite(entry_hash("Head", "A")), Some("a.png#0000000000000001"));
        assert_eq!(r.sprite(entry_hash("Head", "B")), Some("red_b.png#0000000000000003"));
        assert_eq!(r.names(entry_hash("Hat", "C")), Some(("Hat", "C")));
        assert!(r.sprite(12345).is_none());
        assert_eq!(r.entries()[0].0, "Hat");
    }

    #[test]
    fn swap_only_on_a_known_changed_hash() {
        let r = ResolvedLibrary::from_chain(&[&lib("", &[("Head", "A", "a#1"), ("Head", "B", "b#2")])]);
        let a = entry_hash("Head", "A");
        assert_eq!(resolve_swap(&r, a, "b#2").as_deref(), Some("a#1"));
        assert_eq!(resolve_swap(&r, a, "a#1"), None, "already showing it");
        assert_eq!(resolve_swap(&r, 99, "x"), None, "unknown hash keeps the sprite");
        assert_eq!(resolve_swap(&r, 0, "x"), None);
    }

    #[test]
    fn set_and_remove_and_collisions_and_round_trip() {
        let mut l = SpriteLibraryAsset::default();
        l.set("Head", "A", "x");
        l.set("Head", "A", "y");
        assert_eq!(l.categories[0].entries.len(), 1);
        assert_eq!(l.categories[0].entries[0].sprite, "y");
        assert!(l.remove("Head", "A") && !l.remove("Head", "A"));
        assert!(l.collisions().is_empty());
        let l = lib("p.spritelib", &[("Head", "A", "x#1")]);
        let t = ron::ser::to_string_pretty(&l, Default::default()).unwrap();
        assert_eq!(ron::from_str::<SpriteLibraryAsset>(&t).unwrap(), l);
        // An old/bare file loads.
        let bare: SpriteLibraryAsset = ron::from_str("()").unwrap();
        assert!(bare.categories.is_empty() && bare.primary.is_empty());
    }

    #[test]
    fn hash_collisions_are_rare_over_a_realistic_library() {
        let mut l = SpriteLibraryAsset::default();
        for c in 0..20 {
            for e in 0..100 {
                l.set(&format!("Cat{c}"), &format!("Label{e}"), "s");
            }
        }
        assert!(l.collisions().len() <= 1, "{:?}", l.collisions());
    }

    #[test]
    fn files_chain_through_primary() {
        let dir = std::env::temp_dir().join(format!("zest_splib_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        save_sprite_library(dir.join("base.spritelib"), &lib("", &[("Head", "A", "a#1")])).unwrap();
        save_sprite_library(dir.join("var.spritelib"), &lib("base.spritelib", &[("Head", "A", "v#2")])).unwrap();
        let r = load_resolved_library(&dir, &dir.join("var.spritelib")).unwrap();
        assert_eq!(r.sprite(entry_hash("Head", "A")), Some("v#2"));
        // A cycle terminates.
        save_sprite_library(dir.join("c1.spritelib"), &lib("c2.spritelib", &[("X", "1", "p")])).unwrap();
        save_sprite_library(dir.join("c2.spritelib"), &lib("c1.spritelib", &[("X", "2", "q")])).unwrap();
        assert_eq!(load_resolved_library(&dir, &dir.join("c1.spritelib")).unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
