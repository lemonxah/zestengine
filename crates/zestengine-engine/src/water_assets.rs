//! Resolve wave assets once for both simulation and rendering, including
//! headless games. Hot reload is throttled and a bad save keeps the last good data.
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::{Duration, Instant, SystemTime}};
use zestengine_core::{GerstnerWave, Water};
use crate::scene::SceneObject;

#[derive(Default)]
pub(crate) struct WaterAssets {
    pub root: PathBuf,
    cache: HashMap<PathBuf, (Option<SystemTime>, Option<Arc<Vec<GerstnerWave>>>)>,
    next_poll: Option<Instant>,
}
impl WaterAssets {
    pub fn sync(&mut self, objects: &mut [SceneObject]) {
        let now = Instant::now();
        let poll = self.next_poll.is_none_or(|t| now >= t);
        if poll { self.next_poll = Some(now + Duration::from_millis(250)); }
        for object in objects {
            for water in object.components.iter_mut().filter_map(|c| c.as_any_mut().downcast_mut::<Water>()) {
                if water.wave_profile.trim().is_empty() { water.profile_waves = None; continue; }
                let path = zestengine_core::asset_ref::resolve_path(&self.root, &water.wave_profile);
                if poll || !self.cache.contains_key(&path) {
                    let stamp = path.metadata().and_then(|m| m.modified()).ok();
                    if self.cache.get(&path).is_none_or(|(old, _)| *old != stamp) {
                        let old = self.cache.get(&path).and_then(|(_, data)| data.clone());
                        let value = match zestengine_assets::water_file::load_water_waves(&path) {
                            Ok(profile) => Some(Arc::new(profile.resolved())),
                            Err(e) => { tracing::warn!("water waves {}: {e:#}", path.display()); old }
                        };
                        self.cache.insert(path.clone(), (stamp, value));
                    }
                }
                water.profile_waves = self.cache.get(&path).and_then(|(_, data)| data.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_waves_reload_and_invalid_edits_preserve_the_last_good_profile() {
        let root = std::env::temp_dir().join(format!("zest-waves-{}", zestengine_core::ObjectId::new()));
        let path = root.join("sea.waterwaves");
        let profile = zestengine_core::WaterWaveProfile::default();
        zestengine_assets::water_file::save_water_waves(&path, &profile).unwrap();
        let mut objects = vec![SceneObject::new_empty(zestengine_core::ObjectId::new(), "Water")];
        objects[0].components.push(Box::new(Water { wave_profile: "sea.waterwaves".into(), ..Default::default() }));
        let mut assets = WaterAssets { root: root.clone(), ..Default::default() };
        assets.sync(&mut objects);
        let waves = objects[0].components[0].as_any().downcast_ref::<Water>().unwrap().profile_waves.clone().unwrap();
        assert_eq!(*waves, profile.resolved());
        // Force a changed stamp even on filesystems with coarse timestamps.
        assets.cache.get_mut(&path).unwrap().0 = None;
        assets.next_poll = None;
        std::fs::write(&path, "broken profile").unwrap();
        assets.sync(&mut objects);
        assert_eq!(objects[0].components[0].as_any().downcast_ref::<Water>().unwrap().profile_waves.as_deref(), Some(waves.as_ref()));
        objects[0].components[0].as_any_mut().downcast_mut::<Water>().unwrap().wave_profile.clear();
        assets.sync(&mut objects);
        assert!(objects[0].components[0].as_any().downcast_ref::<Water>().unwrap().profile_waves.is_none());
        std::fs::remove_dir_all(root).unwrap();
    }
}
