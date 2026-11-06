//! Audio-specific policy over the shared region selector and transition math.
use std::{collections::HashMap, path::{Path, PathBuf}, time::SystemTime};
use glam::Vec3;
use zestengine_core::{AudioEnvironmentProfile, AudioVolume, ObjectId, component_enabled, transition::Transition, volume::{VolumeRegion, select_volume}};
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
        let world = scene.world_transforms();
        for (i, object) in scene.objects.iter().enumerate().filter(|(i, _)| scene.is_active(*i)) {
            for volume in object.components.iter().filter(|c| component_enabled(c.as_ref()))
                .filter_map(|c| c.as_any().downcast_ref::<AudioVolume>())
            {
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
                    self.regions.push(Region { id: object.id, profile: *profile,
                        bounds: VolumeRegion { world: world[i], half_extents: Vec3::from(volume.half_extents),
                            global: volume.global, priority: volume.priority } });
                }
            }
        }
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
