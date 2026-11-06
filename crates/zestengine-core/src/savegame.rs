//! Save/load game state (ENGINE_FEATURE_CHECKLIST T0/T1 #31).
//!
//! A typed key-value store games write progress into (player position, inventory,
//! flags, current level) and persist to / restore from a `.save` RON file. This is
//! the general "save system" primitive — distinct from the scene `.scene` file
//! (authored content); a save captures *runtime* state.
//!
//! ```no_run
//! # use zestengine_core::{SaveGame, SaveValue};
//! let mut save = SaveGame::new();
//! save.set("level", 3i64);
//! save.set("player.hp", 80.0f32);
//! save.set("has_key", true);
//! save.save("player1.save").unwrap();
//!
//! let loaded = SaveGame::load("player1.save").unwrap();
//! assert_eq!(loaded.int("level"), Some(3));
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A value that can live in a save slot. `BTreeMap`-ordered keys + these variants
/// keep saves human-readable and diff-friendly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SaveValue {
    /// A boolean.
    Bool(bool),
    /// A signed 64-bit integer.
    Int(i64),
    /// A 32-bit float.
    Float(f32),
    /// A string.
    Str(String),
    /// A 3D vector (xyz).
    Vec3([f32; 3]),
    /// A list of values.
    List(Vec<SaveValue>),
}

impl From<bool> for SaveValue {
    fn from(v: bool) -> Self {
        SaveValue::Bool(v)
    }
}
impl From<i64> for SaveValue {
    fn from(v: i64) -> Self {
        SaveValue::Int(v)
    }
}
impl From<f32> for SaveValue {
    fn from(v: f32) -> Self {
        SaveValue::Float(v)
    }
}
impl From<&str> for SaveValue {
    fn from(v: &str) -> Self {
        SaveValue::Str(v.to_string())
    }
}
impl From<String> for SaveValue {
    fn from(v: String) -> Self {
        SaveValue::Str(v)
    }
}
impl From<Vec3> for SaveValue {
    fn from(v: Vec3) -> Self {
        SaveValue::Vec3(v.to_array())
    }
}

/// A persistable game-state slot.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SaveGame {
    /// Format version, so future loaders can migrate old saves.
    #[serde(default = "one")]
    pub version: u32,
    data: BTreeMap<String, SaveValue>,
}

fn one() -> u32 {
    1
}

impl SaveGame {
    /// An empty save (format version 1, no keys).
    pub fn new() -> Self {
        Self { version: 1, data: BTreeMap::new() }
    }

    /// Store a value (overwrites). Accepts anything `Into<SaveValue>`.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<SaveValue>) {
        self.data.insert(key.into(), value.into());
    }

    /// True if the key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Remove a key; returns the old value if present.
    pub fn remove(&mut self, key: &str) -> Option<SaveValue> {
        self.data.remove(key)
    }

    /// Raw value access.
    pub fn get(&self, key: &str) -> Option<&SaveValue> {
        self.data.get(key)
    }

    /// The boolean under `key`, when it is one.
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.data.get(key)? {
            SaveValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// The integer under `key`, when it is one.
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.data.get(key)? {
            SaveValue::Int(i) => Some(*i),
            _ => None,
        }
    }
    /// The float under `key`; an int reads as a float too (whole-number
    /// convenience).
    pub fn float(&self, key: &str) -> Option<f32> {
        match self.data.get(key)? {
            SaveValue::Float(f) => Some(*f),
            // An int is a valid float read (common when a value started whole).
            SaveValue::Int(i) => Some(*i as f32),
            _ => None,
        }
    }
    /// The string under `key`, when it is one.
    pub fn str(&self, key: &str) -> Option<&str> {
        match self.data.get(key)? {
            SaveValue::Str(s) => Some(s),
            _ => None,
        }
    }
    /// The Vec3 under `key`, when it is one.
    pub fn vec3(&self, key: &str) -> Option<Vec3> {
        match self.data.get(key)? {
            SaveValue::Vec3(v) => Some(Vec3::from_array(*v)),
            _ => None,
        }
    }

    /// Number of stored keys.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// True when the save holds no keys.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate the stored entries (sorted by key).
    pub fn iter(&self) -> impl Iterator<Item = (&String, &SaveValue)> {
        self.data.iter()
    }

    /// Serialize to RON.
    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .unwrap_or_default()
    }

    /// Parse from RON.
    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    /// Write to disk (atomically via a temp file + rename so a crash mid-write
    /// can't corrupt an existing save).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let tmp = path.with_extension("save.tmp");
        std::fs::write(&tmp, self.to_ron())?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let s = std::fs::read_to_string(path)?;
        Ok(Self::from_ron(&s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_round_trip_through_ron() {
        let mut save = SaveGame::new();
        save.set("level", 3i64);
        save.set("player.hp", 80.0f32);
        save.set("has_key", true);
        save.set("name", "Hero");
        save.set("spawn", Vec3::new(1.0, 2.0, 3.0));
        save.set("inventory", SaveValue::List(vec![1i64.into(), 2i64.into()]));

        let ron = save.to_ron();
        let back = SaveGame::from_ron(&ron).unwrap();
        assert_eq!(back.int("level"), Some(3));
        assert_eq!(back.float("player.hp"), Some(80.0));
        assert_eq!(back.bool("has_key"), Some(true));
        assert_eq!(back.str("name"), Some("Hero"));
        assert_eq!(back.vec3("spawn"), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(back.len(), 6);
    }

    #[test]
    fn wrong_type_and_missing_return_none() {
        let mut save = SaveGame::new();
        save.set("flag", true);
        assert_eq!(save.int("flag"), None); // wrong type
        assert_eq!(save.float("missing"), None); // absent
        // Int reads as float (whole-number convenience).
        save.set("count", 5i64);
        assert_eq!(save.float("count"), Some(5.0));
    }

    #[test]
    fn overwrite_and_remove() {
        let mut save = SaveGame::new();
        save.set("k", 1i64);
        save.set("k", 2i64);
        assert_eq!(save.int("k"), Some(2));
        assert_eq!(save.remove("k"), Some(SaveValue::Int(2)));
        assert!(!save.contains("k"));
    }

    #[test]
    fn atomic_save_load_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("zestengine_test_savegame.save");
        let mut save = SaveGame::new();
        save.set("level", 7i64);
        save.save(&path).unwrap();
        let loaded = SaveGame::load(&path).unwrap();
        assert_eq!(loaded.int("level"), Some(7));
        let _ = std::fs::remove_file(&path);
    }
}
