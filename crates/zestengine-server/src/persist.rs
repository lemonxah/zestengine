//! Persistent server state (NET-6): flagged scene objects and per-player data,
//! kept under the state folder (`<project>/.server/` by default) in two
//! crash-safe files ([`crate::state_store`]).
//!
//! - `world.state`: for each scene, the objects that carry a
//!   [`PersistentState`](zestengine_core::PersistentState) component: their
//!   transform and Network Variables, keyed by the object's scene id. Put back when
//!   the scene loads. Objects spawned at runtime from a prefab have no stable id: those that
//!   carry the component are saved as `spawned` entries (prefab, transform, variables) and
//!   spawned again, server-owned, when the scene loads.
//! - `players.state`: per player id (the fingerprint of the key that joined),
//!   the Network Variables of the player object flagged `per_player`, plus name
//!   and last-seen time. Given back when that player joins again.
//!
//! Files carry a schema version; a file from a newer server is left alone
//! (nothing is saved over it) and the server starts without it.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use zestengine_core::{NetValue, NetworkVariables, ObjectId, PersistentState};
use zestengine_engine::server::LoadedScene;

use crate::state_store::{Loaded, Store};

/// A Network Variable write to replicate: object, name, value.
pub type VarWrite = (ObjectId, String, NetValue);

/// Schema version of both files.
pub const SCHEMA: u32 = 1;
/// Players remembered at most (the least recently seen go first).
pub const MAX_PLAYERS: usize = 50_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Xform {
    pub t: [f32; 3],
    pub r: [f32; 4],
    pub s: [f32; 3],
}

impl Default for Xform {
    fn default() -> Self {
        Self { t: [0.0; 3], r: [0.0, 0.0, 0.0, 1.0], s: [1.0; 3] }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectState {
    /// For humans reading the file.
    pub name: String,
    pub transform: Option<Xform>,
    pub vars: Vec<(String, NetValue)>,
}

/// A runtime-spawned prefab instance that was flagged persistent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpawnedState {
    /// Project-relative prefab path.
    pub prefab: String,
    pub t: [f32; 3],
    pub r: [f32; 4],
    pub vars: Vec<(String, NetValue)>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneState {
    /// Object id (32 hex digits) to its state.
    pub objects: BTreeMap<String, ObjectState>,
    /// Flagged objects that were spawned at runtime from a prefab.
    pub spawned: Vec<SpawnedState>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorldState {
    pub saved_unix: u64,
    /// Project-relative scene path to its state.
    pub scenes: BTreeMap<String, SceneState>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlayerRecord {
    pub name: String,
    pub first_seen: u64,
    pub last_seen: u64,
    pub sessions: u32,
    pub vars: Vec<(String, NetValue)>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlayersState {
    pub saved_unix: u64,
    /// Player id fingerprint to its record.
    pub players: BTreeMap<String, PlayerRecord>,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn object_key(id: ObjectId) -> String {
    format!("{:032x}", id.raw())
}

fn net_vars(c: &[Box<dyn zestengine_core::Component>]) -> Option<&NetworkVariables> {
    c.iter().find_map(|c| c.as_any().downcast_ref::<NetworkVariables>())
}

/// One file's worth of state with its store and change detection.
struct Slot<T> {
    store: Store,
    value: T,
    /// The bytes last written (or loaded): an unchanged state is not rewritten.
    written: Vec<u8>,
    /// A file from a newer server is on disk: never write over it.
    locked: bool,
}

fn load_slot<T: Default + for<'de> Deserialize<'de>>(store: Store, what: &str) -> Slot<T> {
    let mut slot = Slot { store, value: T::default(), written: Vec::new(), locked: false };
    match slot.store.load() {
        Loaded::Missing => tracing::info!("{what}: no saved state, starting fresh"),
        Loaded::Found { version, payload, generation, damaged } => {
            for d in &damaged {
                tracing::error!("{what}: unusable file {d}");
            }
            if version > SCHEMA {
                tracing::error!("{what}: saved by a newer server (schema {version}, this one reads up to {SCHEMA}); starting fresh and NOT saving over it");
                slot.locked = true;
            } else {
                match ron::from_str::<T>(&String::from_utf8_lossy(&payload)) {
                    Ok(v) => {
                        slot.value = v;
                        slot.written = payload;
                        if generation > 0 {
                            tracing::warn!("{what}: restored from backup generation {generation} (the newer file was damaged)");
                        } else {
                            tracing::info!("{what}: loaded");
                        }
                    }
                    Err(e) => tracing::error!("{what}: schema {version} file does not parse ({e}); starting fresh (the file is overwritten at the next save, the backups stay)"),
                }
            }
        }
        Loaded::Unusable { damaged } => {
            for d in &damaged {
                tracing::error!("{what}: unusable file {d}");
            }
            tracing::error!("{what}: no usable generation, starting fresh");
        }
    }
    slot
}

fn save_slot<T: Serialize>(slot: &mut Slot<T>, what: &str, empty: bool) -> bool {
    // Nothing to keep and never kept anything: no file.
    if slot.locked || (empty && slot.written.is_empty()) {
        return false;
    }
    let text = match ron::ser::to_string_pretty(&slot.value, ron::ser::PrettyConfig::default()) {
        Ok(t) => t.into_bytes(),
        Err(e) => {
            tracing::error!("{what}: cannot encode: {e}");
            return false;
        }
    };
    // `saved_unix` changes every time; compare without it.
    let strip = |b: &[u8]| String::from_utf8_lossy(b).lines().filter(|l| !l.trim_start().starts_with("saved_unix")).collect::<Vec<_>>().join("\n");
    if !slot.written.is_empty() && strip(&slot.written) == strip(&text) {
        return false;
    }
    match slot.store.save(SCHEMA, &text) {
        Ok(()) => {
            slot.written = text;
            true
        }
        Err(e) => {
            tracing::error!("{what}: save failed: {e}");
            false
        }
    }
}

/// The server's persistence: load at start, capture and apply, save on an interval and at shutdown.
pub struct Persistence {
    world: Slot<WorldState>,
    players: Slot<PlayersState>,
    interval: Duration,
    last_save: Instant,
}

impl Persistence {
    /// Open the state in `dir` (created at the first save). `interval` 0 = save only at shutdown.
    pub fn open(dir: &Path, keep: u32, interval: f32) -> Self {
        let world = load_slot(Store::new(dir, "world.state", keep), "world state");
        let players = load_slot(Store::new(dir, "players.state", keep), "player state");
        Self { world, players, interval: Duration::from_secs_f32(interval.max(0.0)), last_save: Instant::now() }
    }

    pub fn world(&self) -> &WorldState {
        &self.world.value
    }

    pub fn players(&self) -> &PlayersState {
        &self.players.value
    }

    /// Record the flagged objects of `scene` (replacing what was recorded for `path`).
    ///
    /// `spawned` lists the runtime prefab spawns `(root id, prefab path)` of the session, or `None`
    /// to keep what was recorded for `path` (right after a restore, while the respawned objects are
    /// not in the scene yet).
    pub fn capture_scene(&mut self, scene: &LoadedScene, path: &str, spawned: Option<&[(ObjectId, String)]>) {
        let mut state = SceneState::default();
        let is_spawn = |id: ObjectId| spawned.is_some_and(|l| l.iter().any(|(s, _)| *s == id));
        for o in &scene.objects {
            let Some(flags) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<PersistentState>()) else { continue };
            if flags.per_player || !(flags.transform || flags.variables) {
                continue;
            }
            if let Some((_, prefab)) = spawned.and_then(|l| l.iter().find(|(s, _)| *s == o.id)) {
                let t = if flags.transform && o.translation.is_finite() { o.translation.to_array() } else { [0.0; 3] };
                let r = if flags.transform && o.rotation.is_finite() { o.rotation.to_array() } else { [0.0, 0.0, 0.0, 1.0] };
                let vars = match (flags.variables, net_vars(&o.components)) {
                    (true, Some(v)) => v.vars.iter().map(|v| (v.name.clone(), v.value.clone())).collect(),
                    _ => Vec::new(),
                };
                state.spawned.push(SpawnedState { prefab: prefab.clone(), t, r, vars });
                continue;
            }
            if is_spawn(o.id) {
                continue;
            }
            let mut s = ObjectState { name: o.name.clone(), ..Default::default() };
            if flags.transform && o.translation.is_finite() && o.rotation.is_finite() && o.scale.is_finite() {
                s.transform = Some(Xform { t: o.translation.to_array(), r: o.rotation.to_array(), s: o.scale.to_array() });
            }
            if flags.variables && let Some(v) = net_vars(&o.components) {
                s.vars = v.vars.iter().map(|v| (v.name.clone(), v.value.clone())).collect();
            }
            state.objects.insert(object_key(o.id), s);
        }
        if spawned.is_none() {
            state.spawned = self.world.value.scenes.get(path).map(|s| s.spawned.clone()).unwrap_or_default();
        }
        if state.objects.is_empty() && state.spawned.is_empty() {
            self.world.value.scenes.remove(path);
        } else {
            self.world.value.scenes.insert(path.to_owned(), state);
        }
    }

    /// Put the recorded state of `path` back onto `scene`. Returns how many objects were
    /// restored and the variable writes the session must be told about
    /// (`NetSession::set_var`), so clients that join later see them.
    pub fn restore_scene(&self, scene: &mut LoadedScene, path: &str) -> (usize, Vec<VarWrite>) {
        let Some(state) = self.world.value.scenes.get(path) else { return (0, Vec::new()) };
        let (mut restored, mut writes) = (0, Vec::new());
        for o in &mut scene.objects {
            let Some(saved) = state.objects.get(&object_key(o.id)) else { continue };
            let Some(flags) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<PersistentState>()).cloned() else { continue };
            if flags.transform && let Some(x) = &saved.transform {
                let (t, r, s) = (glam::Vec3::from(x.t), glam::Quat::from_array(x.r), glam::Vec3::from(x.s));
                if t.is_finite() && r.is_finite() && s.is_finite() {
                    o.translation = t;
                    o.rotation = r.normalize();
                    o.scale = s;
                }
            }
            if flags.variables {
                let id = o.id;
                if let Some(v) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>()) {
                    for (k, val) in &saved.vars {
                        v.set(k, val.clone());
                        writes.push((id, k.clone(), val.clone()));
                    }
                }
            }
            restored += 1;
        }
        (restored, writes)
    }

    /// The runtime-spawned objects recorded for `path`, to spawn again.
    pub fn spawned_for(&self, path: &str) -> Vec<SpawnedState> {
        self.world.value.scenes.get(path).map(|s| s.spawned.clone()).unwrap_or_default()
    }

    /// A player (by id fingerprint) joined: note it. Returns whether it has saved data.
    pub fn player_joined(&mut self, fp: &str, name: &str) -> bool {
        let now = unix_now();
        let known = self.players.value.players.contains_key(fp);
        let r = self.players.value.players.entry(fp.to_owned()).or_insert_with(|| PlayerRecord { first_seen: now, ..Default::default() });
        r.name = name.to_owned();
        r.last_seen = now;
        r.sessions += 1;
        known && !r.vars.is_empty()
    }

    /// Copy a flagged player object's variables into the player's record.
    pub fn capture_player(&mut self, scene: &LoadedScene, fp: &str, object: ObjectId) {
        let Some(i) = scene.index_of(object) else { return };
        let o = &scene.objects[i];
        if !o.components.iter().any(|c| c.as_any().downcast_ref::<PersistentState>().is_some_and(|p| p.per_player && p.variables)) {
            return;
        }
        if let (Some(v), Some(r)) = (net_vars(&o.components), self.players.value.players.get_mut(fp)) {
            r.vars = v.vars.iter().map(|v| (v.name.clone(), v.value.clone())).collect();
            r.last_seen = unix_now();
        }
    }

    /// Give a returning player's saved variables to its new player object. The writes are
    /// for `NetSession::set_var`; empty when the player has nothing saved or the object is not flagged.
    /// `None`: the object is not in the scene yet (the session knows the id a tick before the scene has the object).
    pub fn restore_player(&self, scene: &mut LoadedScene, fp: &str, object: ObjectId) -> Option<Vec<VarWrite>> {
        let i = scene.index_of(object)?;
        let Some(rec) = self.players.value.players.get(fp) else { return Some(Vec::new()) };
        let o = &mut scene.objects[i];
        if !o.components.iter().any(|c| c.as_any().downcast_ref::<PersistentState>().is_some_and(|p| p.per_player && p.variables)) {
            return Some(Vec::new());
        }
        let Some(v) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>()) else { return Some(Vec::new()) };
        let mut writes = Vec::new();
        for (k, val) in &rec.vars {
            v.set(k, val.clone());
            writes.push((object, k.clone(), val.clone()));
        }
        Some(writes)
    }

    /// Whether the save interval has passed.
    pub fn due(&self) -> bool {
        !self.interval.is_zero() && self.last_save.elapsed() >= self.interval
    }

    /// Write whatever changed. Returns how many files were written.
    pub fn save(&mut self) -> usize {
        self.last_save = Instant::now();
        let now = unix_now();
        self.world.value.saved_unix = now;
        self.players.value.saved_unix = now;
        if self.players.value.players.len() > MAX_PLAYERS {
            let mut by_age: Vec<(u64, String)> = self.players.value.players.iter().map(|(k, r)| (r.last_seen, k.clone())).collect();
            by_age.sort();
            for (_, k) in by_age.into_iter().take(self.players.value.players.len() - MAX_PLAYERS) {
                self.players.value.players.remove(&k);
            }
        }
        let (we, pe) = (self.world.value.scenes.is_empty(), self.players.value.players.is_empty());
        usize::from(save_slot(&mut self.world, "world state", we)) + usize::from(save_slot(&mut self.players, "player state", pe))
    }

    /// Whether there is anything worth a file (a scene object or a player record).
    pub fn is_empty(&self) -> bool {
        self.world.value.scenes.is_empty() && self.players.value.players.is_empty()
    }
}
