//! Audio-specific policy over the shared region selector and transition math.
use std::{collections::HashMap, path::{Path, PathBuf}, time::SystemTime};
use glam::Vec3;
use zestengine_core::{AudioEnvironmentProfile, AudioVolume, ObjectId, transition::Transition, volume::{VolumeRegion, select_volume}};
use crate::scene::LoadedScene;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mix {
    pub gain: f32,
    pub lowpass_hz: f32,
    pub reverb_gain: f32,
}
impl Default for Mix {
    fn default() -> Self { Self { gain: 1.0, lowpass_hz: 22_000.0, reverb_gain: 1.0 } }
}

pub struct MixFade {
    gain: Transition,
    cutoff: Transition,
    reverb: Transition,
    seconds: f32,
}
impl Default for MixFade {
    fn default() -> Self {
        Self { gain: Transition::new(1.0), cutoff: Transition::new(22_000.0), reverb: Transition::new(1.0), seconds: 0.5 }
    }
}
impl MixFade {
    pub fn step(&mut self, target: Mix, seconds: f32, dt: f32) -> Mix {
        // Leaving the last region uses its fade duration, not a hard reset.
        let seconds = if seconds < 0.0 { self.seconds } else { seconds };
        self.seconds = seconds;
        Mix { gain: self.gain.step(target.gain, seconds, dt),
            lowpass_hz: self.cutoff.step(target.lowpass_hz, seconds, dt),
            reverb_gain: self.reverb.step(target.reverb_gain, seconds, dt) }
    }
}

struct Region { id: ObjectId, bounds: VolumeRegion, profile: AudioEnvironmentProfile }
#[derive(Default)]
pub struct AudioEnvironment {
    regions: Vec<Region>,
    cache: HashMap<PathBuf, (Option<SystemTime>, Option<AudioEnvironmentProfile>)>,
    refresh: f32,
}
impl AudioEnvironment {
    pub fn sync(&mut self, scene: &LoadedScene, root: &Path, dt: f32) {
        self.refresh -= dt;
        let refresh = self.refresh <= 0.0;
        if refresh { self.refresh = 0.25; }
        self.regions.clear();
        scene.objects.query(|ecs, order| {
            let mut objects: Vec<_> = ecs.query::<(&ObjectId, &AudioVolume)>().iter().filter_map(|(entity, (id, _))| {
                Some((*order.get(id)?, entity, *id))
            }).collect();
            objects.sort_unstable_by_key(|(index, _, _)| *index);
            for (index, entity, id) in objects {
                if !scene.is_active_in(ecs, index) { continue; }
                let Some(matrix) = ecs.world_transform(entity) else { continue };
                // Regions with equal priority preserve both scene-tree and
                // duplicate component order. Handles carry no payload copies.
                for (_, instance) in scene.objects[index].components.instance_entities::<AudioVolume>() {
                    let Some(volume) = ecs.get::<AudioVolume>(instance) else { continue };
                    if volume.profile.trim().is_empty() { continue; }
                    let path = zestengine_core::asset_ref::resolve_path(root, &volume.profile);
                    if refresh || !self.cache.contains_key(&path) {
                        let stamp = path.metadata().and_then(|m| m.modified()).ok();
                        if self.cache.get(&path).is_none_or(|(old, _)| *old != stamp) {
                            let profile = match zestengine_assets::load_audiofx(&path) {
                                Ok(p) => Some(p),
                                Err(e) => { tracing::warn!("audio volume {}: {e:#}", path.display()); None }
                            };
                            self.cache.insert(path.clone(), (stamp, profile));
                        }
                    }
                    if let Some((_, Some(profile))) = self.cache.get(&path) {
                        self.regions.push(Region { id, profile: *profile,
                            bounds: VolumeRegion { world: matrix, half_extents: Vec3::from(volume.half_extents),
                                global: volume.global, priority: volume.priority } });
                    }
                }
            }
        });
    }
    fn region(&self, at: Vec3) -> Option<&Region> {
        select_volume(self.regions.iter().map(|r| r.bounds), at).map(|i| &self.regions[i])
    }
    /// 2D/UI/music beds bypass spatial regions. Spatial music remains spatial.
    pub fn target(&self, listener: Vec3, source: Option<Vec3>) -> (Mix, f32) {
        let Some(source) = source else { return (Mix::default(), -1.0) };
        let ear = self.region(listener);
        let src = self.region(source);
        let mut mix = Mix { reverb_gain: ear.map_or(1.0, |r| r.profile.reverb_gain), ..Default::default() };
        let mut seconds = ear.map_or(-1.0, |r| r.profile.fade_time);
        if ear.map(|r| r.id) != src.map(|r| r.id) {
            if let Some(r) = ear {
                mix.gain *= r.profile.exterior_gain;
                mix.lowpass_hz = mix.lowpass_hz.min(r.profile.exterior_lowpass_hz);
            }
            if let Some(r) = src {
                mix.gain *= r.profile.interior_gain;
                mix.lowpass_hz = mix.lowpass_hz.min(r.profile.interior_lowpass_hz);
                seconds = seconds.max(r.profile.fade_time);
            }
        }
        (mix, seconds)
    }
}

#[cfg(feature = "client")]
pub type MixSlot = std::sync::Arc<std::sync::Mutex<Mix>>;

/// Output processing shared by Steam Audio and the distance-only backend.
#[cfg(feature = "client")]
pub struct EnvironmentalSource<S> {
    inner: S, slot: MixSlot, current: Mix, target: Mix,
    filtered: Vec<f32>, channel: usize, frame: usize,
}
#[cfg(feature = "client")]
impl<S: rodio::Source<Item = f32>> EnvironmentalSource<S> {
    pub fn new(inner: S, slot: MixSlot) -> Self {
        let filtered = vec![0.0; inner.channels().max(1) as usize];
        Self { inner, slot, current: Mix::default(), target: Mix::default(), filtered, channel: 0, frame: 0 }
    }
}
#[cfg(feature = "client")]
impl<S: rodio::Source<Item = f32>> Iterator for EnvironmentalSource<S> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        let x = self.inner.next()?;
        if self.channel == 0 {
            if self.frame == 0 && let Ok(mix) = self.slot.try_lock() { self.target = *mix; }
            let remaining = (256 - self.frame) as f32;
            self.current.gain += (self.target.gain - self.current.gain) / remaining;
            self.current.lowpass_hz += (self.target.lowpass_hz - self.current.lowpass_hz) / remaining;
            self.frame = (self.frame + 1) % 256;
        }
        let y = &mut self.filtered[self.channel];
        let out = if self.current.lowpass_hz >= 21_999.0 {
            *y = x;
            x
        } else {
            let cutoff = self.current.lowpass_hz.min(self.inner.sample_rate() as f32 * 0.45);
            let a = 1.0 - (-std::f32::consts::TAU * cutoff / self.inner.sample_rate() as f32).exp();
            *y += a * (x - *y);
            *y
        };
        self.channel = (self.channel + 1) % self.filtered.len();
        Some(out * self.current.gain)
    }
}
#[cfg(feature = "client")]
impl<S: rodio::Source<Item = f32>> rodio::Source for EnvironmentalSource<S> {
    fn current_frame_len(&self) -> Option<usize> { self.inner.current_frame_len() }
    fn channels(&self) -> u16 { self.inner.channels() }
    fn sample_rate(&self) -> u32 { self.inner.sample_rate() }
    fn total_duration(&self) -> Option<std::time::Duration> { self.inner.total_duration() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_regions_keep_duplicate_order_and_follow_live_parent_and_component_edits() {
        use crate::scene::SceneObject;
        use zestengine_core::{Disabled, Transform, ecs::{ComponentSet, Enabled}};
        let root = std::env::temp_dir().join(format!("zest-audio-regions-{}", ObjectId::new()));
        std::fs::create_dir_all(&root).unwrap();
        zestengine_assets::save_audiofx(root.join("first.audiofx"), &AudioEnvironmentProfile { reverb_gain: 0.2, ..Default::default() }).unwrap();
        zestengine_assets::save_audiofx(root.join("second.audiofx"), &AudioEnvironmentProfile { reverb_gain: 0.8, ..Default::default() }).unwrap();
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::from_components(ComponentSet::new()).with_transform(Transform { translation: Vec3::X * 10.0,
            rotation: glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), scale: Vec3::splat(2.0) }));
        let first = AudioVolume { profile: "first.audiofx".into(), half_extents: [1.0; 3], ..Default::default() };
        let second = AudioVolume { profile: "second.audiofx".into(), ..first.clone() };
        scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![
            Box::new(Disabled(Box::new(second.clone()))), Box::new(first.clone()), Box::new(second.clone()),
        ])).with_parent(Some(0)).with_transform(Transform { translation: Vec3::X, ..Default::default() }));
        scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![Box::new(second)]))
            .with_transform(Transform { translation: Vec3::new(10.0, 2.0, 0.0), ..Default::default() }));
        let id = scene.objects[1].id();
        let parent = scene.objects[0].id();
        let mut env = AudioEnvironment::default();
        let at = Vec3::new(10.0, 2.0, 0.0);
        env.sync(&scene, &root, 0.01);
        assert_eq!(env.regions.len(), 3);
        assert_eq!(env.target(at, Some(at)).0.reverb_gain, 0.2, "ties keep scene and duplicate order");
        assert_eq!(env.target(at + Vec3::Y * 1.5, Some(at)).0.reverb_gain, 0.2, "parent scale and rotation affect local bounds");
        scene.objects[1].components.set_enabled(1, false);
        env.sync(&scene, &root, 0.01);
        assert_eq!(env.target(at, Some(at)).0.reverb_gain, 0.8);
        scene.objects[1].components.set_enabled(1, true);
        scene.objects.swap(1, 2);
        env.sync(&scene, &root, 0.01);
        assert_eq!(env.target(at, Some(at)).0.reverb_gain, 0.8, "new authoring order breaks equal-priority ties");
        scene.objects.query(|ecs, _| {
            ecs.get_mut::<AudioVolume>(ecs.entity(id).unwrap()).unwrap().priority = 2.0;
            ecs.get_mut::<Transform>(ecs.entity(parent).unwrap()).unwrap().translation += Vec3::Y * 10.0;
        });
        env.sync(&scene, &root, 0.01);
        let moved = at + Vec3::Y * 10.0;
        assert_eq!(env.target(moved, Some(moved)).0.reverb_gain, 0.2);
        scene.objects.query(|ecs, _| ecs.get_mut::<Enabled>(ecs.entity(parent).unwrap()).unwrap().0 = false);
        env.sync(&scene, &root, 0.01);
        assert_eq!(env.target(moved, Some(moved)).0, Mix::default());
        assert_eq!(env.regions.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interior_exterior_and_two_dimensional_bypass() {
        let env = AudioEnvironment { regions: vec![Region { id: ObjectId::new(),
            bounds: VolumeRegion { world: glam::Mat4::IDENTITY, half_extents: Vec3::ONE, global: false, priority: 0.0 },
            profile: AudioEnvironmentProfile { reverb_gain: 0.4, interior_gain: 0.2, exterior_gain: 0.3,
                interior_lowpass_hz: 800.0, exterior_lowpass_hz: 1200.0, ..Default::default() } }], ..Default::default() };
        let outside = Vec3::splat(10.0);
        assert_eq!(env.target(Vec3::ZERO, Some(outside)).0, Mix { gain: 0.3, lowpass_hz: 1200.0, reverb_gain: 0.4 });
        assert_eq!(env.target(outside, Some(Vec3::ZERO)).0.gain, 0.2);
        assert_eq!(env.target(Vec3::ZERO, Some(Vec3::ZERO)).0.gain, 1.0);
        assert_eq!(env.target(Vec3::ZERO, None).0, Mix::default());
    }

    #[test]
    fn leaving_a_region_uses_its_fade_time() {
        let mut fade = MixFade::default();
        let quiet = Mix { gain: 0.0, ..Default::default() };
        assert_eq!(fade.step(quiet, 2.0, 2.0).gain, 0.0);
        assert_eq!(fade.step(Mix::default(), -1.0, 1.0).gain, 0.5);
        assert_eq!(fade.step(Mix::default(), -1.0, 1.0).gain, 1.0);
    }

    #[cfg(feature = "client")]
    #[test]
    fn output_filter_and_gain_work_without_an_audio_device() {
        let samples: Vec<_> = (0..48_000).map(|i| (i as f32 * 8000.0 * std::f32::consts::TAU / 48_000.0).sin()).collect();
        let render = |mix: Mix| {
            let slot = std::sync::Arc::new(std::sync::Mutex::new(mix));
            let source = rodio::buffer::SamplesBuffer::new(1, 48_000, samples.clone());
            EnvironmentalSource::new(source, slot).skip(1024).map(|s| s*s).sum::<f32>()
        };
        let full = render(Mix::default());
        let quiet = render(Mix { gain: 0.25, ..Default::default() });
        assert!((quiet / full - 0.0625).abs() < 1e-5);
        assert!(render(Mix { lowpass_hz: 800.0, ..Default::default() }) < full * 0.05);
        assert_eq!(render(Mix { gain: 0.0, ..Default::default() }), 0.0);
    }
}
