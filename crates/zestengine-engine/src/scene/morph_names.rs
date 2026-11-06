//! Shared, cached blendshape name lists: the inspector asks for an object's
//! (hundreds of) names every frame, so they are built once per mesh set and
//! handed out as `Arc<[String]>` instead of cloned.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::LoadedScene;

/// Cache of `(mesh list -> (signature, names))`.
#[derive(Default)]
pub(crate) struct MorphNameCache(Mutex<HashMap<Vec<usize>, MorphNameEntry>>);

/// One cached morph-name table: signature plus the shared name list.
type MorphNameEntry = (u64, Arc<[String]>);

impl LoadedScene {
    /// `morph_names` as a shared slice, rebuilt only when the object's mesh set or
    /// those meshes' blendshapes change.
    pub fn morph_names_shared(&self, object: usize) -> Arc<[String]> {
        let Some(o) = self.objects.get(object) else { return Arc::from([]) };
        let meshes: Vec<usize> = o.render_slots().map(|r| r.mesh).filter(|m| self.skinned_meshes.iter().any(|sm| sm.mesh_index == *m)).collect();
        if meshes.is_empty() {
            return Arc::from([]);
        }
        // Cheap change signature: shape counts and the address of each first name.
        let sig = meshes.iter().fold(0u64, |h, m| {
            let sm = self.skinned_meshes.iter().find(|sm| sm.mesh_index == *m);
            let (n, p) = sm.map_or((0, 0), |sm| (sm.morphs.len() as u64, sm.morphs.first().map_or(0, |t| t.name.as_ptr() as u64)));
            (h ^ n ^ p.rotate_left(17)).wrapping_mul(0x100_0000_01b3)
        });
        let mut cache = self.morph_name_cache.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((s, names)) = cache.get(&meshes)
            && *s == sig
        {
            return names.clone();
        }
        let names: Arc<[String]> = self.morph_names(object).into();
        if cache.len() > 256 {
            cache.clear();
        }
        cache.insert(meshes, (sig, names.clone()));
        names
    }
}
