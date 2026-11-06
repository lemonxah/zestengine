//! GPU-resource liveness tracking per content origin: decide which meshes,
//! textures and materials become unreferenced when an origin is unloaded.

use std::collections::{HashMap, HashSet};

/// The GPU resources (by renderer index) one origin references.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResourceSet {
    pub meshes: HashSet<usize>,
    pub textures: HashSet<usize>,
    pub materials: HashSet<usize>,
}

/// Per-origin resource usage, used to free only what is no longer referenced.
#[derive(Clone, Debug, Default)]
pub struct ResourceTracker {
    origins: HashMap<String, ResourceSet>,
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record (or replace) the resources `origin` uses.
    pub fn record(&mut self, origin: &str, set: ResourceSet) {
        self.origins.insert(origin.to_owned(), set);
    }

    /// The resources `origin` uses and no other origin references — safe to
    /// free when `origin` unloads.
    pub fn free_set_for(&self, origin: &str) -> ResourceSet {
        let Some(target) = self.origins.get(origin) else {
            return ResourceSet::default();
        };
        let mut out = target.clone();
        for (other, set) in &self.origins {
            if other == origin {
                continue;
            }
            out.meshes.retain(|m| !set.meshes.contains(m));
            out.textures.retain(|t| !set.textures.contains(t));
            out.materials.retain(|m| !set.materials.contains(m));
        }
        out
    }

    /// Drop an origin's bookkeeping.
    pub fn clear(&mut self, origin: &str) {
        self.origins.remove(origin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(meshes: &[usize], textures: &[usize], materials: &[usize]) -> ResourceSet {
        ResourceSet {
            meshes: meshes.iter().copied().collect(),
            textures: textures.iter().copied().collect(),
            materials: materials.iter().copied().collect(),
        }
    }

    #[test]
    fn shared_resources_survive_unload() {
        let mut t = ResourceTracker::new();
        t.record("a", set(&[0], &[0], &[]));
        t.record("b", set(&[1], &[0], &[]));
        let free = t.free_set_for("a");
        assert_eq!(free.meshes, HashSet::from([0]));
        assert!(free.textures.is_empty(), "texture 0 is shared with b");
    }

    #[test]
    fn unique_resources_are_freed() {
        let mut t = ResourceTracker::new();
        t.record("a", set(&[0, 1], &[2], &[3]));
        let free = t.free_set_for("a");
        assert_eq!(free.meshes, HashSet::from([0, 1]));
        assert_eq!(free.textures, HashSet::from([2]));
        assert_eq!(free.materials, HashSet::from([3]));
    }

    #[test]
    fn cleared_origin_frees_nothing() {
        let mut t = ResourceTracker::new();
        t.record("a", set(&[0], &[0], &[]));
        t.clear("a");
        assert_eq!(t.free_set_for("a"), ResourceSet::default());
    }
}
