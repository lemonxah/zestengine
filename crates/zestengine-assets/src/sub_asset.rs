//! Stable identities for the parts of an imported model (meshes, materials,
//! clips). A scene that places "the body mesh of `hero.fbx`" must keep finding
//! it when the file is re-exported, reordered, or its settings change how many
//! meshes come out, so the reference can't be a flattened index.
//!
//! Scheme (version [`ID_SCHEME`]): every sub-asset has a *name* (for a mesh, the
//! unique node path plus a part suffix, e.g. `Root/Body/Mesh#part2`); its id is
//! a hash of `(kind, name)`. Siblings that share a name are made unique
//! *before* hashing ([`SiblingNamer`]). The first time a name is seen it is
//! recorded in the model's `.meta` [`IdTable`]; later imports look the name up
//! there before hashing, so a rename done through the importer
//! ([`IdTable::rename`]) keeps the id. The scheme version is stored beside the
//! table so the hash can change later without breaking old references.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Current ID-scheme version, written to new `.meta` files.
pub const ID_SCHEME: u32 = 1;

/// Which kind of sub-asset an id names. Part of the hash, so a mesh and a
/// material with the same name get different ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SubAssetKind {
    Mesh,
    Material,
    Skeleton,
    Animation,
    /// A sprite of a texture (`sprite.rs`).
    Sprite,
    /// One placed part (instance) of a model: its primary mesh and which
    /// occurrence of that mesh it is. Scenes name deleted and edited parts by it.
    Part,
}

impl SubAssetKind {
    fn tag(self) -> u8 {
        match self {
            Self::Mesh => 1,
            Self::Material => 2,
            Self::Skeleton => 3,
            Self::Animation => 4,
            Self::Sprite => 5,
            Self::Part => 6,
        }
    }
}

/// Identity of one sub-asset inside a model file. Unique within the model;
/// scenes pair it with the model's `.meta` GUID. Never 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubAssetId(pub u64);

impl SubAssetId {
    /// "No id" placeholder (never produced by hashing).
    pub const NONE: SubAssetId = SubAssetId(0);

    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// FNV-1a 64 over `(scheme tag, kind tag, name, salt)`. Stable across
/// platforms and Rust versions (std's hashers are not), which on-disk ids need.
fn hash_name(kind: SubAssetKind, name: &str, salt: u32) -> SubAssetId {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    feed(kind.tag());
    feed(0);
    for b in name.bytes() {
        feed(b);
    }
    if salt != 0 {
        feed(0);
        for b in salt.to_le_bytes() {
            feed(b);
        }
    }
    SubAssetId(if h == 0 { 1 } else { h })
}

/// The id a name hashes to, ignoring any table. Used for names not recorded yet.
pub fn hashed_id(kind: SubAssetKind, name: &str) -> SubAssetId {
    hash_name(kind, name, 0)
}

/// Makes names unique among siblings (children of one parent), deterministically:
/// the first keeps its name, later ones get `~1`, `~2`, ... in encounter order.
/// Separators in a name are replaced so a name can't forge a path.
#[derive(Default)]
pub struct SiblingNamer {
    taken: HashSet<(String, String)>,
}

impl SiblingNamer {
    /// A unique name for a child of `parent` (an already-unique parent path).
    pub fn unique(&mut self, parent: &str, raw: &str) -> String {
        let base: String = raw.chars().map(|c| if c == '/' || c == '#' { '_' } else { c }).collect();
        let base = if base.is_empty() { "unnamed".to_string() } else { base };
        let mut name = base.clone();
        let mut n = 0u32;
        while !self.taken.insert((parent.to_owned(), name.clone())) {
            n += 1;
            name = format!("{base}~{n}");
        }
        name
    }

    /// Unique child path `parent/name` (or just `name` under the root).
    pub fn unique_path(&mut self, parent: &str, raw: &str) -> String {
        let name = self.unique(parent, raw);
        if parent.is_empty() { name } else { format!("{parent}/{name}") }
    }
}

/// One recorded sub-asset: its (kind, name) and the id handed out for it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdEntry {
    pub kind: SubAssetKind,
    pub name: String,
    pub id: SubAssetId,
}

/// A model's `(kind, name) -> id` table, stored in its `.meta`. Entries are
/// never dropped when a name disappears from a re-export (so it gets the same
/// id if it comes back).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IdTable {
    pub entries: Vec<IdEntry>,
}

impl IdTable {
    /// The recorded id for a name, if any.
    pub fn find(&self, kind: SubAssetKind, name: &str) -> Option<SubAssetId> {
        self.entries.iter().find(|e| e.kind == kind && e.name == name).map(|e| e.id)
    }

    /// The name recorded for an id.
    pub fn name_of(&self, id: SubAssetId) -> Option<&str> {
        self.entries.iter().find(|e| e.id == id).map(|e| e.name.as_str())
    }

    fn id_in_use(&self, id: SubAssetId) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// The id for a name: the recorded one, else a fresh hash. Does not modify
    /// the table (safe for read-only loads). `taken` holds ids already handed
    /// out in this import so a hash collision still yields distinct ids.
    pub fn lookup(&self, kind: SubAssetKind, name: &str, taken: &HashSet<SubAssetId>) -> SubAssetId {
        if let Some(id) = self.find(kind, name) {
            return id;
        }
        let mut salt = 0;
        loop {
            let id = hash_name(kind, name, salt);
            if !taken.contains(&id) && !self.id_in_use(id) {
                return id;
            }
            salt += 1;
        }
    }

    /// Like [`Self::lookup`] but records a new name in the table. Returns the id
    /// and whether the table changed.
    pub fn register(&mut self, kind: SubAssetKind, name: &str, taken: &HashSet<SubAssetId>) -> (SubAssetId, bool) {
        if let Some(id) = self.find(kind, name) {
            return (id, false);
        }
        let id = self.lookup(kind, name, taken);
        self.entries.push(IdEntry { kind, name: name.to_owned(), id });
        (id, true)
    }

    /// Rename a sub-asset through the importer, keeping its id. If `old` isn't
    /// recorded yet its hash id is recorded first. Returns the id, or `None` if
    /// `new` is already a different sub-asset's name.
    pub fn rename(&mut self, kind: SubAssetKind, old: &str, new: &str) -> Option<SubAssetId> {
        if old == new {
            return Some(self.register(kind, old, &HashSet::new()).0);
        }
        if self.find(kind, new).is_some() {
            return None;
        }
        let id = self.register(kind, old, &HashSet::new()).0;
        if let Some(e) = self.entries.iter_mut().find(|e| e.kind == kind && e.name == old) {
            e.name = new.to_owned();
        }
        Some(id)
    }
}

/// Assign ids to a model's names, in order. With `table` mutable the new names
/// are recorded (editor import); with `None` nothing is written (shipped game).
/// Returns the ids and whether the table changed.
pub fn assign_ids(kind: SubAssetKind, names: &[String], table: &mut IdTable, record: bool) -> (Vec<SubAssetId>, bool) {
    let mut taken: HashSet<SubAssetId> = HashSet::new();
    let mut out = Vec::with_capacity(names.len());
    let mut changed = false;
    for name in names {
        let id = if record {
            let (id, c) = table.register(kind, name, &taken);
            changed |= c;
            id
        } else {
            table.lookup(kind, name, &taken)
        };
        taken.insert(id);
        out.push(id);
    }
    (out, changed)
}

/// Where a scene should look a mesh up: `ids[i]` is the id of the model's mesh `i`.
pub fn index_of(ids: &[SubAssetId], id: SubAssetId) -> Option<usize> {
    if id.is_none() {
        return None;
    }
    ids.iter().position(|x| *x == id)
}

/// Map from id to flattened index, for repeated lookups.
pub fn index_map(ids: &[SubAssetId]) -> HashMap<SubAssetId, usize> {
    ids.iter().enumerate().map(|(i, id)| (*id, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_depends_on_kind_and_name_and_is_stable() {
        let a = hashed_id(SubAssetKind::Mesh, "Root/Body#part0");
        assert_eq!(a, hashed_id(SubAssetKind::Mesh, "Root/Body#part0"), "deterministic");
        assert_ne!(a, hashed_id(SubAssetKind::Material, "Root/Body#part0"), "kind is part of the hash");
        assert_ne!(a, hashed_id(SubAssetKind::Mesh, "Root/Body#part1"));
        assert!(!a.is_none());
        // Golden value: changing it would break every saved reference, so a
        // scheme change must bump ID_SCHEME instead.
        assert_eq!(a, SubAssetId(0x9323_c619_111c_dccd));
        assert_eq!(ID_SCHEME, 1);
    }

    #[test]
    fn siblings_are_made_unique_deterministically() {
        let mut n = SiblingNamer::default();
        assert_eq!(n.unique_path("", "Root"), "Root");
        assert_eq!(n.unique_path("Root", "Arm"), "Root/Arm");
        assert_eq!(n.unique_path("Root", "Arm"), "Root/Arm~1");
        assert_eq!(n.unique_path("Root", "Arm"), "Root/Arm~2");
        // Same name under a different parent is not a sibling.
        assert_eq!(n.unique_path("Root/Arm", "Arm"), "Root/Arm/Arm");
        // A literal "Arm~1" can't collide with the generated one.
        assert_eq!(n.unique_path("Root", "Arm~1"), "Root/Arm~1~1");
        // Separators can't forge paths.
        assert_eq!(n.unique_path("Root", "a/b"), "Root/a_b");
        assert_eq!(n.unique_path("Root", ""), "Root/unnamed");
    }

    #[test]
    fn duplicate_sibling_names_get_distinct_ids() {
        let mut n = SiblingNamer::default();
        let names: Vec<String> = (0..3).map(|_| n.unique_path("", "Mesh")).collect();
        let mut table = IdTable::default();
        let (ids, changed) = assign_ids(SubAssetKind::Mesh, &names, &mut table, true);
        assert!(changed);
        let set: HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), 3);
        // Re-import in a different order: names, not positions, decide ids.
        let rev: Vec<String> = names.iter().rev().cloned().collect();
        let (ids2, changed2) = assign_ids(SubAssetKind::Mesh, &rev, &mut table, true);
        assert!(!changed2);
        assert_eq!(ids2, ids.iter().rev().copied().collect::<Vec<_>>());
    }

    #[test]
    fn rename_through_the_importer_keeps_the_id() {
        let mut table = IdTable::default();
        let names = vec!["Root/Body#part0".to_string()];
        let (ids, _) = assign_ids(SubAssetKind::Mesh, &names, &mut table, true);
        let id = table.rename(SubAssetKind::Mesh, "Root/Body#part0", "Root/Torso#part0").unwrap();
        assert_eq!(id, ids[0]);
        // The renamed export now resolves to the old id, not a new hash.
        let renamed = vec!["Root/Torso#part0".to_string()];
        let (ids2, _) = assign_ids(SubAssetKind::Mesh, &renamed, &mut table, true);
        assert_eq!(ids2[0], ids[0]);
        assert_ne!(ids2[0], hashed_id(SubAssetKind::Mesh, "Root/Torso#part0"));
        assert_eq!(table.name_of(id), Some("Root/Torso#part0"));
        // Renaming onto an existing different name is refused.
        let (_, _) = assign_ids(SubAssetKind::Mesh, &["Other#part0".to_string()], &mut table, true);
        assert_eq!(table.rename(SubAssetKind::Mesh, "Root/Torso#part0", "Other#part0"), None);
    }

    #[test]
    fn read_only_assignment_does_not_touch_the_table() {
        let mut table = IdTable::default();
        let (ids, changed) = assign_ids(SubAssetKind::Mesh, &["A#part0".to_string()], &mut table, false);
        assert!(!changed);
        assert!(table.entries.is_empty());
        assert_eq!(ids[0], hashed_id(SubAssetKind::Mesh, "A#part0"));
    }

    #[test]
    fn hash_collisions_still_give_distinct_ids() {
        let mut table = IdTable::default();
        let first = hashed_id(SubAssetKind::Mesh, "B#part0");
        // Pretend another name already owns that hash.
        table.entries.push(IdEntry { kind: SubAssetKind::Mesh, name: "squatter".into(), id: first });
        let (ids, _) = assign_ids(SubAssetKind::Mesh, &["B#part0".to_string()], &mut table, true);
        assert_ne!(ids[0], first);
        assert!(!ids[0].is_none());
    }

    #[test]
    fn index_lookup_by_id() {
        let ids = vec![SubAssetId(5), SubAssetId(9)];
        assert_eq!(index_of(&ids, SubAssetId(9)), Some(1));
        assert_eq!(index_of(&ids, SubAssetId(7)), None);
        assert_eq!(index_of(&ids, SubAssetId::NONE), None);
    }
}
