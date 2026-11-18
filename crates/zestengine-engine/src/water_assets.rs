//! Resolve wave assets once for both simulation and rendering, including
//! headless games. Hot reload is throttled and a bad save keeps the last good data.
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::{Duration, Instant, SystemTime}};
use zestengine_core::{GerstnerWave, Water, WaterWaveGenerator};
use crate::scene::SceneObject;

struct ResolvedWaves {waves:Arc<Vec<GerstnerWave>>,ocean:Option<Arc<zestengine_core::OceanSpectrum>>}

#[derive(Default)]
pub(crate) struct WaterAssets {
    pub root: PathBuf,
    pub(crate) cache_jobs:Vec<crate::water_bake::Job>,
    recordings:HashMap<PathBuf,(Option<SystemTime>,Option<Arc<zestengine_core::WaterSimulationCache>>)>,
    cache: HashMap<PathBuf, (Option<SystemTime>, Option<Arc<ResolvedWaves>>)>,
    next_poll: Option<Instant>,
    generators: HashMap<String, Arc<dyn WaterWaveGenerator>>,
}
impl WaterAssets {
    pub fn register_generator(&mut self, id: String, generator: Arc<dyn WaterWaveGenerator>) {
        self.generators.insert(id, generator);
        // Preserve last-good data, but resolve every referenced asset again.
        for (stamp, _) in self.cache.values_mut() { *stamp = None; }
        self.next_poll = None;
    }
    pub fn sync(&mut self, objects: &[SceneObject]) {
        let now = Instant::now();
        let poll = self.next_poll.is_none_or(|t| now >= t);
        if poll { self.next_poll = Some(now + Duration::from_millis(250)); }
        for object in objects {
            for mut water in object.components.iter_typed_mut::<Water>() {
                let asset=&water.simulation_cache.asset;
                if asset.trim().is_empty(){water.cache_data=None;}else{
                    let path=zestengine_core::asset_ref::resolve_path(&self.root,asset);
                    if poll || !self.recordings.contains_key(&path){
                        let stamp=path.metadata().and_then(|m|m.modified()).ok();
                        if self.recordings.get(&path).is_none_or(|(old,_)|*old!=stamp){
                            let old=self.recordings.get(&path).and_then(|(_,data)|data.clone());
                            let data=match zestengine_assets::water_cache_file::load_water_cache(&path){Ok(data)=>Some(Arc::new(data)),Err(e)=>{tracing::warn!("water recording {}: {e:#}",path.display());old}};
                            self.recordings.insert(path.clone(),(stamp,data));
                        }
                    }
                    water.cache_data=self.recordings.get(&path).and_then(|(_,data)|data.clone());
                }
                if water.wave_profile.trim().is_empty() { water.profile_waves = None; water.profile_ocean=None; continue; }
                let path = zestengine_core::asset_ref::resolve_path(&self.root, &water.wave_profile);
                if poll || !self.cache.contains_key(&path) {
                    let stamp = path.metadata().and_then(|m| m.modified()).ok();
                    if self.cache.get(&path).is_none_or(|(old, _)| *old != stamp) {
                        let old = self.cache.get(&path).and_then(|(_, data)| data.clone());
                        let resolved = zestengine_assets::water_file::load_water_waves(&path).and_then(|profile| {
                            let waves=if let Some(config) = &profile.generator {
                                let generator = self.generators.get(&config.id)
                                    .ok_or_else(|| anyhow::anyhow!("unregistered water wave generator '{}'", config.id))?;
                                profile.generated(generator.as_ref()).map_err(anyhow::Error::msg)
                            } else { Ok(profile.resolved()) }?;
                            Ok(ResolvedWaves{waves:Arc::new(waves),ocean:profile.ocean.map(|o|Arc::new(o.sanitized()))})
                        });
                        let value = match resolved {
                            Ok(waves) => Some(Arc::new(waves)),
                            Err(e) => { tracing::warn!("water waves {}: {e:#}", path.display()); old }
                        };
                        self.cache.insert(path.clone(), (stamp, value));
                    }
                }
                water.profile_waves = self.cache.get(&path).and_then(|(_, data)| data.as_ref().map(|d|d.waves.clone()));
                water.profile_ocean = self.cache.get(&path).and_then(|(_,data)|data.as_ref().and_then(|d|d.ocean.clone()));
            }
        }
    }
}

impl crate::scene::LoadedScene {
    /// Register a deterministic wave generator shared by visual and headless
    /// scene updates. Assets select it by id and provide numeric settings.
    pub fn register_water_wave_generator(&mut self, id: impl Into<String>, generator: Arc<dyn WaterWaveGenerator>) {
        self.water_assets.register_generator(id.into(), generator);
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
        assets.sync(&objects);
        let waves = objects[0].components.get_typed_at::<Water>(0).unwrap().profile_waves.clone().unwrap();
        assert_eq!(*waves, profile.resolved());
        // Force a changed stamp even on filesystems with coarse timestamps.
        assets.cache.get_mut(&path).unwrap().0 = None;
        assets.next_poll = None;
        std::fs::write(&path, "broken profile").unwrap();
        assets.sync(&objects);
        assert_eq!(objects[0].components.get_typed_at::<Water>(0).unwrap().profile_waves.as_deref(), Some(waves.as_ref()));
        objects[0].components.get_typed_at_mut::<Water>(0).unwrap().wave_profile.clear();
        assets.sync(&objects);
        assert!(objects[0].components.get_typed_at::<Water>(0).unwrap().profile_waves.is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registered_generators_reload_headlessly_and_preserve_last_good_on_failure() {
        struct Generator(f32);
        impl WaterWaveGenerator for Generator {
            fn generate(&self, settings: &std::collections::BTreeMap<String, f32>) -> Result<Vec<GerstnerWave>, String> {
                Ok(vec![GerstnerWave { amplitude: settings["height"] * self.0, ..Default::default() }])
            }
        }
        let root = std::env::temp_dir().join(format!("zest-wave-generator-{}", zestengine_core::ObjectId::new()));
        let path = root.join("sea.waterwaves");
        let profile = zestengine_core::WaterWaveProfile {
            generator: Some(zestengine_core::WaterWaveGeneratorConfig { id: "test.sea".into(), settings: [("height".into(), 0.3)].into() }),
            ..Default::default()
        };
        zestengine_assets::water_file::save_water_waves(&path, &profile).unwrap();
        let mut objects = vec![SceneObject::new_empty(zestengine_core::ObjectId::new(), "Water")];
        objects[0].components.push(Box::new(Water { wave_profile: "sea.waterwaves".into(), ..Default::default() }));
        let mut assets = WaterAssets { root: root.clone(), ..Default::default() };
        let waves = |objects: &[SceneObject]| objects[0].components.get_typed_at::<Water>(0).unwrap().profile_waves.clone();
        assets.sync(&objects);
        assert!(waves(&objects).is_none());
        assets.register_generator("test.sea".into(), Arc::new(Generator(1.0)));
        assets.sync(&objects);
        assert_eq!(waves(&objects).unwrap()[0].amplitude, 0.3);
        assets.register_generator("test.sea".into(), Arc::new(Generator(2.0)));
        assets.sync(&objects);
        assert_eq!(waves(&objects).unwrap()[0].amplitude, 0.6);
        assets.register_generator("test.sea".into(), Arc::new(Generator(f32::NAN)));
        assets.sync(&objects);
        assert_eq!(waves(&objects).unwrap()[0].amplitude, 0.6);
        std::fs::remove_dir_all(root).unwrap();
    }
}
