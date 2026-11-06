//! Marks scene content a dedicated server keeps across restarts (NET-6).
//!
//! The server saves the flagged objects of the running scene under its state
//! folder (`<project>/.server/`) on an interval and at shutdown, and puts the
//! values back when that scene loads again. Only scene objects have a stable id;
//! objects spawned at runtime are not saved.

use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// What of this object a dedicated server persists.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistentState {
    /// World: save and restore the object's position, rotation and scale (a pushed crate, an opened door).
    pub transform: bool,
    /// World: save and restore the values of the object's Network Variables (a score, a lever state).
    pub variables: bool,
    /// Per player: on a player object, save its Network Variables (only them) under the owner's
    /// player id and give them back when that player joins again (inventory, progress, currency).
    pub per_player: bool,
}

impl Default for PersistentState {
    fn default() -> Self {
        Self { transform: true, variables: true, per_player: false }
    }
}

impl TypedComponent for PersistentState {
    const NAME: &'static str = "Persistent State";
}
