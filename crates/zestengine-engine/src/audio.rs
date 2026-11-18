//! Engine audio: plays `AudioSource` components in Play mode and spatializes
//! them against the active `AudioListener` (falling back to the editor
//! camera). Decoding (.wav/.flac/.mp3/.ogg) and mixing are handled by `rodio`.
//!
//! The scene is the source of truth; each frame the engine builds a list of
//! [`AudioCue`]s (one per active source, with its world transform). With the
//! `steam-audio` feature, spatial sources render through Steam Audio (HRTF,
//! air absorption, directivity, occlusion, reflections / reverb, pathing,
//! ambisonics, virtual surround; see `steam_audio.rs`); otherwise,
//! or if Steam Audio fails to start, their volume follows the listener
//! distance. Play-on-start sources begin the first frame their object is
//! active; a source whose object is destroyed or disabled stops. Gameplay can
//! also fire one-shot clips (`ctx.play_sound`), which play to completion.

#[cfg(feature = "client")]
use std::collections::HashMap;
#[cfg(feature = "client")]
use std::io::BufReader;
use std::path::Path;
#[cfg(feature = "client")]
use std::sync::Arc;
#[cfg(feature = "client")]
use std::sync::atomic::{AtomicU32, Ordering};

use zestengine_core::{AudioListener, AudioMixer, AudioRolloff, AudioSource, ObjectId, SpatialAudioSettings};
use glam::{Mat4, Quat, Vec3};

use crate::scene::LoadedScene;
#[cfg(feature = "steam-audio")]
use zestengine_core::ecs::{Entity, SystemWorld};
#[cfg(feature = "steam-audio")]
use crate::scene::metadata::{Render, ExtraRender, StaticGeometry};

#[cfg(feature = "steam-audio")]
#[path = "steam_audio.rs"]
pub mod steam;

/// A flattened audio source for one frame (decoupled from the component so the
/// scene isn't borrowed while the audio engine runs).
#[derive(Clone)]
pub struct AudioCue {
    pub object: ObjectId,
    pub clip: String,
    pub play_on_start: bool,
    pub looping: bool,
    pub volume: f32,
    pub pitch: f32,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: AudioRolloff,
    pub position: Vec3,
    pub rotation: Quat,
    pub bus: String,
    pub hrtf: bool,
    pub air_absorption: bool,
    pub occlusion: bool,
    pub directivity: f32,
    pub directivity_sharpness: f32,
    pub reflections: bool,
    pub pathing: bool,
    pub ambisonic_order: u8,
    pub via_ambisonics: bool,
}

impl AudioCue {
    pub fn from_source(object: ObjectId, src: &AudioSource, position: Vec3) -> Self {
        Self {
            object,
            clip: src.clip.clone(),
            play_on_start: src.play_on_start,
            looping: src.looping,
            volume: src.volume,
            pitch: src.pitch,
            spatial: src.spatial,
            min_distance: src.min_distance,
            max_distance: src.max_distance,
            rolloff: src.rolloff,
            position,
            rotation: Quat::IDENTITY,
            bus: src.bus.clone(),
            hrtf: src.hrtf,
            air_absorption: src.air_absorption,
            occlusion: src.occlusion,
            directivity: src.directivity,
            directivity_sharpness: src.directivity_sharpness,
            reflections: src.reflections,
            pathing: src.pathing,
            ambisonic_order: src.ambisonic_order,
            via_ambisonics: src.spatialize_via_ambisonics,
        }
    }

    /// Distance falloff (0..1) from the source's rolloff settings; 1 for
    /// non-spatial sources.
    fn attenuation(&self, listener: Vec3) -> f32 {
        if !self.spatial {
            return 1.0;
        }
        let dist = self.position.distance(listener);
        let (lo, hi) = (self.min_distance, self.max_distance.max(self.min_distance + 0.01));
        if dist <= lo {
            1.0
        } else if dist >= hi {
            0.0
        } else {
            let t = (dist - lo) / (hi - lo);
            match self.rolloff {
                AudioRolloff::Linear => 1.0 - t,
                // Normalized inverse-distance: 1 at min, 0 at max.
                AudioRolloff::Logarithmic => {
                    let inv = lo / dist;
                    let inv_max = lo / hi;
                    ((inv - inv_max) / (1.0 - inv_max)).clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Volume multiplier for this cue given the listener distance (the
    /// distance-only fallback).
    fn gain(&self, listener: Vec3) -> f32 {
        (self.volume * self.attenuation(listener)).max(0.0)
    }

    #[cfg(feature = "steam-audio")]
    fn emitter(&self, listener: Vec3) -> steam::Emitter {
        steam::Emitter {
            transform: Mat4::from_rotation_translation(self.rotation, self.position),
            attenuation: self.attenuation(listener),
            air_absorption: self.air_absorption,
            directivity: self.directivity,
            directivity_sharpness: self.directivity_sharpness,
            hrtf: self.hrtf,
        }
    }
}

/// Every active `AudioSource` with its world position and orientation.
pub fn gather_cues(scene: &LoadedScene) -> Vec<AudioCue> {
    scene.objects.map_components::<AudioSource, _>(|i, src, _ecs| {
        if !scene.is_active_in(_ecs, i) { return None; }
        let (_, rotation, position) = scene.world_transform_in(_ecs, i).to_scale_rotation_translation();
        let mut cue = AudioCue::from_source(scene.objects[i].id_in(_ecs), src, position);
        cue.rotation = rotation;
        Some(cue)
    })
}

/// World pose (rotation + position, no scale) of the first active listener.
pub fn listener_pose(scene: &LoadedScene) -> Option<Mat4> {
    scene.objects.find_component::<AudioListener, _>(|i, _, _ecs| {
        if !scene.is_active_in(_ecs, i) { return None; }
        let (_, rotation, position) = scene.world_transform_in(_ecs, i).to_scale_rotation_translation();
        Some(Mat4::from_rotation_translation(rotation, position))
    })
}

/// Native acoustic geometry membership, in authoring order. These are entity
/// handles only; materials and render slots stay borrowed from the world.
#[cfg(feature = "steam-audio")]
fn acoustic_objects_in(scene: &LoadedScene, ecs: &SystemWorld<'_>, order: &std::collections::HashMap<ObjectId, usize>) -> (Vec<Entity>, Vec<Entity>) {
    let mut statics = Vec::new();
    let mut dynamics = Vec::new();
    for (entity, (id, render, fixed)) in ecs.query::<(&ObjectId, &Render, &StaticGeometry)>().iter() {
        let Some(&index) = order.get(id) else { continue };
        if render.0.is_none() || !scene.is_active_in(ecs, index) { continue; }
        if fixed.0 { statics.push((index, entity)); }
        else if ecs.get::<zestengine_core::AcousticMaterial>(entity).is_some_and(|material| material.dynamic) { dynamics.push((index, entity)); }
    }
    statics.sort_unstable_by_key(|(index, _)| *index);
    dynamics.sort_unstable_by_key(|(index, _)| *index);
    (statics.into_iter().map(|(_, entity)| entity).collect(), dynamics.into_iter().map(|(_, entity)| entity).collect())
}

#[cfg(feature = "steam-audio")]
fn acoustic_values_in(ecs: &SystemWorld<'_>, entity: Entity) -> steam::MaterialValues {
    ecs.get::<zestengine_core::AcousticMaterial>(entity).map_or_else(|| zestengine_core::AcousticMaterial::default().values(), |material| material.values())
}

/// Static meshes are baked in world space, while dynamic occluders retain
/// local geometry and receive their current world transforms each frame.
#[cfg(feature = "steam-audio")]
fn gather_geometry_in(scene: &LoadedScene, ecs: &SystemWorld<'_>, statics: &[Entity], dynamics: &[Entity]) -> steam::Geometry {
    let mut geo = steam::Geometry::default();
    geo.materials.push(zestengine_core::AcousticMaterial::default().values());
    let tris = |indices: &[u32], base: i32| -> Vec<[i32; 3]> {
        indices.as_chunks::<3>().0.iter().map(|t| [base + t[0] as i32, base + t[1] as i32, base + t[2] as i32]).collect()
    };
    for &entity in statics {
        let Some(matrix) = ecs.world_transform(entity) else { continue };
        let values = acoustic_values_in(ecs, entity);
        let mat = match geo.materials.iter().position(|m| *m == values) {
            Some(k) => k,
            None => { geo.materials.push(values); geo.materials.len() - 1 }
        };
        let primary = ecs.get::<Render>(entity).and_then(|render| render.0);
        let extra = ecs.get::<ExtraRender>(entity).expect("object render slots");
        for render in primary.into_iter().chain(extra.0.iter().copied()) {
            let Some((positions, indices)) = scene.mesh_geometry(render.mesh) else { continue };
            let base = geo.vertices.len() as i32;
            geo.vertices.extend(positions.iter().map(|p| matrix.transform_point3(*p).to_array()));
            let t = tris(indices, base);
            geo.material_indices.extend(std::iter::repeat_n(mat, t.len()));
            geo.triangles.extend(t);
        }
    }
    for &entity in dynamics {
        let Some(id) = ecs.object_id(entity) else { continue };
        let mut mesh = steam::DynamicMesh { id, vertices: Vec::new(), triangles: Vec::new(), material: acoustic_values_in(ecs, entity) };
        let primary = ecs.get::<Render>(entity).and_then(|render| render.0);
        let extra = ecs.get::<ExtraRender>(entity).expect("object render slots");
        for render in primary.into_iter().chain(extra.0.iter().copied()) {
            let Some((positions, indices)) = scene.mesh_geometry(render.mesh) else { continue };
            let base = mesh.vertices.len() as i32;
            mesh.vertices.extend(positions.iter().map(|p| p.to_array()));
            mesh.triangles.extend(tris(indices, base));
        }
        geo.dynamic.push(mesh);
    }
    geo
}

#[cfg(feature = "steam-audio")]
fn acoustic_key_in(ecs: &SystemWorld<'_>, statics: &[Entity], dynamics: &[Entity]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for (fixed, entities) in [(true, statics), (false, dynamics)] {
        fixed.hash(&mut hash);
        entities.len().hash(&mut hash);
        for &entity in entities {
            ecs.object_id(entity).hash(&mut hash);
            let primary = ecs.get::<Render>(entity).and_then(|render| render.0);
            let extra = ecs.get::<ExtraRender>(entity).expect("object render slots");
            extra.0.len().hash(&mut hash);
            for render in primary.into_iter().chain(extra.0.iter().copied()) { render.mesh.hash(&mut hash); }
            let (absorption, scattering, transmission) = acoustic_values_in(ecs, entity);
            for value in absorption.into_iter().chain([scattering]).chain(transmission) { value.to_bits().hash(&mut hash); }
            // A moved static surface needs a new baked mesh. Dynamic poses are
            // supplied separately, so their movement must not rebuild geometry.
            if fixed && let Some(matrix) = ecs.world_transform(entity) {
                for value in matrix.to_cols_array() { value.to_bits().hash(&mut hash); }
            }
        }
    }
    hash.finish()
}

#[cfg(feature = "steam-audio")]
fn probe_volumes_in(scene: &LoadedScene, ecs: &SystemWorld<'_>, order: &std::collections::HashMap<ObjectId, usize>) -> Vec<steam::BakeVolume> {
    let mut volumes: Vec<_> = ecs.query::<(&ObjectId, &zestengine_core::AudioProbeVolume)>().iter().filter_map(|(entity, (id, volume))| {
        let &index = order.get(id)?;
        if !scene.is_active_in(ecs, index) { return None; }
        Some((index, steam::BakeVolume {
            transform: ecs.world_transform(entity)? * Mat4::from_scale(Vec3::from(volume.size)),
            spacing: volume.spacing, height: volume.height, reverb: volume.reverb, pathing: volume.pathing,
        }))
    }).collect();
    volumes.sort_unstable_by_key(|(index, _)| *index);
    volumes.into_iter().map(|(_, volume)| volume).collect()
}

/// Static geometry and probe volumes of a scene, for the audio bake.
#[cfg(feature = "steam-audio")]
pub fn bake_input(scene: &LoadedScene) -> (steam::Geometry, Vec<steam::BakeVolume>) {
    scene.objects.query(|ecs, order| {
        let (statics, _) = acoustic_objects_in(scene, ecs, order);
        (gather_geometry_in(scene, ecs, &statics, &[]), probe_volumes_in(scene, ecs, order))
    })
}

#[cfg_attr(not(feature = "client"), allow(dead_code))]
/// One-shot falloff: linear from 1 m to 30 m.
fn oneshot_attenuation(position: Vec3, listener: Vec3) -> f32 {
    1.0 - ((position.distance(listener) - 1.0) / 29.0).clamp(0.0, 1.0)
}

// The playback half (rodio output, Steam Audio voices). A dedicated server
// (no `client`) has no audio device: `AudioEngine::new()` is `None` there.
#[cfg(feature = "client")]
pub use engine::AudioEngine;
#[cfg(feature = "client")]
mod engine {
use super::*;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use crate::audio_environment::{AudioEnvironment, MixFade, MixSlot, EnvironmentalSource};

#[cfg(feature = "steam-audio")]
type Slot = steam::VoiceSlot;
#[cfg(not(feature = "steam-audio"))]
type Slot = ();

struct Voice {
    sink: Sink,
    environment: MixSlot,
    fade: MixFade,
    /// Mix tap for the analysis (see `audio_tap`).
    tap: Arc<crate::audio_tap::VoiceTap>,
    /// Steam Audio parameter slot (None: distance-volume fallback).
    slot: Option<Slot>,
    #[cfg(feature = "steam-audio")]
    indirect: Indirect,
}

/// Reflection / pathing slots a Steam Audio voice reads.
#[cfg(feature = "steam-audio")]
#[derive(Clone, Default)]
struct Indirect {
    reflections: Option<steam::ReflSlot>,
    pathing: Option<steam::PathSlot>,
}

struct OneShot {
    position: Option<Vec3>,
    environment: MixSlot,
    fade: MixFade,
    #[cfg_attr(not(feature = "steam-audio"), allow(dead_code))]
    id: u64,
    sink: Sink,
    tap: Arc<crate::audio_tap::VoiceTap>,
    /// World position + slot of a spatial one-shot rendered by Steam Audio.
    #[cfg_attr(not(feature = "steam-audio"), allow(dead_code))]
    spatial: Option<(Vec3, Slot)>,
}

pub struct AudioEngine {
    environment: AudioEnvironment,
    // Kept alive for the engine's lifetime; dropping it kills all sound.
    _stream: OutputStream,
    handle: OutputStreamHandle,
    /// Object -> its active `AudioSource` voice (only populated while playing).
    sinks: HashMap<ObjectId, Voice>,
    /// Fire-and-forget clips; dropped once finished.
    oneshots: Vec<OneShot>,
    next_shot: u64,
    #[cfg(feature = "steam-audio")]
    steam: Option<steam::SteamAudio>,
    #[cfg(feature = "steam-audio")]
    simulation: Option<steam::Simulation>,
    /// Identity of the geometry last sent to the simulation.
    #[cfg(feature = "steam-audio")]
    geometry_key: Option<u64>,
    /// Identity (pointer) of the audio bake last sent to the simulation.
    #[cfg(feature = "steam-audio")]
    bake_key: Option<usize>,
    /// Moving occluders (object ids) sent with the geometry.
    #[cfg(feature = "steam-audio")]
    dynamic: Vec<ObjectId>,
    /// World boxes (unit cube) of baked probe volumes.
    #[cfg(feature = "steam-audio")]
    probe_boxes: Vec<Mat4>,
    /// Latest moving-occluder transforms (sent each update).
    #[cfg(feature = "steam-audio")]
    dynamic_transforms: Vec<(ObjectId, Mat4)>,
    /// Underwater muffle every voice follows (0..1, f32 bits).
    muffle: Arc<AtomicU32>,
    /// The underwater ambience loop playing (its clip and sink).
    ambience: Option<(String, Sink)>,
    underwater: crate::underwater::UnderwaterMix,
    /// Taps on every voice, read by the audio analysis.
    tap: Arc<crate::audio_tap::AudioTap>,
}


impl AudioEngine {
    /// Open the default output device. Returns None (with a warning) if there
    /// is no audio device, so the editor still runs silently.
    pub fn new() -> Option<Self> {
        #[cfg(feature = "steam-audio")]
        let steam = match steam::SteamAudio::new() {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!("Steam Audio unavailable, using distance volume: {e}");
                None
            }
        };
        match OutputStream::try_default() {
            Ok((stream, handle)) => Some(Self {
                environment: AudioEnvironment::default(),
                _stream: stream,
                handle,
                sinks: HashMap::new(),
                oneshots: Vec::new(),
                next_shot: 0,
                #[cfg(feature = "steam-audio")]
                simulation: steam.as_ref().map(steam::Simulation::new),
                #[cfg(feature = "steam-audio")]
                steam,
                #[cfg(feature = "steam-audio")]
                geometry_key: None,
                #[cfg(feature = "steam-audio")]
                bake_key: None,
                #[cfg(feature = "steam-audio")]
                dynamic: Vec::new(),
                #[cfg(feature = "steam-audio")]
                probe_boxes: Vec::new(),
                #[cfg(feature = "steam-audio")]
                dynamic_transforms: Vec::new(),
                muffle: Arc::new(AtomicU32::new(0.0f32.to_bits())),
                ambience: None,
                underwater: Default::default(),
                tap: Default::default(),
            })
            .inspect(|engine| tracing::info!("spatial audio: {}", engine.spatializer())),
            Err(e) => {
                tracing::warn!("no audio output device: {e}");
                None
            }
        }
    }

    /// Apply the project's Steam Audio settings (HRTF, output layout,
    /// reflection type, ray tracer, quality). Rebuilds the Steam Audio context
    /// and simulation; playing voices stop.
    #[allow(unused_variables)]
    pub fn configure(&mut self, settings: &SpatialAudioSettings, project_root: &Path) {
        #[cfg(feature = "steam-audio")]
        {
            if self.steam.as_ref().is_some_and(|s| s.config() == &settings.sanitized()) {
                return;
            }
            self.stop_all();
            self.simulation = None;
            self.steam = match steam::SteamAudio::with_settings(settings, project_root) {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::warn!("Steam Audio unavailable, using distance volume: {e}");
                    None
                }
            };
            self.simulation = self.steam.as_ref().map(steam::Simulation::new);
            self.geometry_key = None;
            self.bake_key = None;
        }
    }

    /// Whether spatial sources render through Steam Audio.
    pub fn spatializer(&self) -> &'static str {
        #[cfg(feature = "steam-audio")]
        if self.steam.is_some() {
            return "Steam Audio";
        }
        "distance volume"
    }

    pub fn stop_all(&mut self) {
        for (_, voice) in self.sinks.drain() {
            voice.sink.stop();
        }
        for shot in self.oneshots.drain(..) {
            shot.sink.stop();
        }
        if let Some((_, sink)) = self.ambience.take() {
            sink.stop();
        }
    }

    /// Follow the listener's underwater state (`under`: the water its ear is
    /// in, if any), fading over `dt` (unscaled seconds): muffle every voice and
    /// play the water's ambience loop (unmuffled) at the faded volume.
    pub fn set_underwater(
        &mut self,
        under: Option<&zestengine_core::UnderwaterSettings>,
        dt: f32,
        project_root: &Path,
        mixer: &AudioMixer,
    ) {
        self.underwater.step(under, dt);
        self.muffle.store(self.underwater.muffle().to_bits(), Ordering::Relaxed);
        let want = self.underwater.ambience().map(|(clip, volume)| (clip.to_string(), volume));
        let want = want.as_ref().map(|(clip, volume)| (clip.as_str(), *volume));
        if self.ambience.as_ref().map(|(clip, _)| clip.as_str()) != want.map(|(clip, _)| clip) {
            if let Some((_, sink)) = self.ambience.take() {
                sink.stop();
            }
            if let Some((clip, _)) = want
                && let Some((sink, decoder)) = self.open(project_root, clip)
            {
                sink.append(decoder.repeat_infinite());
                self.ambience = Some((clip.to_string(), sink));
            }
        }
        if let (Some((_, sink)), Some((_, volume))) = (self.ambience.as_ref(), want) {
            sink.set_volume(volume * mixer.effective_gain("sfx"));
        }
    }

    /// Append `source` to `sink` through Steam Audio when it's available and
    /// the clip needs it (spatial, ambisonic, surround bed), returning the
    /// voice's parameter slot; otherwise append it as is.
    fn append<S>(&self, sink: &Sink, source: S, init: Option<SlotInit>) -> (Option<Slot>, Arc<crate::audio_tap::VoiceTap>, MixSlot)
    where
        S: Source + Send + 'static,
        S::Item: rodio::Sample + Send,
        f32: rodio::cpal::FromSample<S::Item>,
    {
        self.append_muffled(sink, Muffled::new(source.convert_samples::<f32>(), self.muffle.clone()), init)
    }

    #[allow(unused_variables)]
    fn append_muffled<S: Source<Item = f32> + Send + 'static>(&self, sink: &Sink, source: S, init: Option<SlotInit>) -> (Option<Slot>, Arc<crate::audio_tap::VoiceTap>, MixSlot) {
        let tap = self.tap.register(source.sample_rate());
        let environment = MixSlot::default();
        #[cfg(feature = "steam-audio")]
        if let (Some(sa), Some(SlotInit(emitter, listener, opts))) = (self.steam.as_ref(), init)
            && sa.wants(&opts, source.channels())
        {
            let slot: Slot = std::sync::Arc::new(std::sync::Mutex::new(sa.state_for(&emitter, listener)));
            match sa.voice(source, slot.clone(), opts) {
                Ok(voice) => {
                    append_voice(sink, crate::audio_tap::Tapped::new(EnvironmentalSource::new(voice, environment.clone()), tap.clone()));
                    return (Some(slot), tap, environment);
                }
                Err(e) => {
                    tracing::error!("Steam Audio voice: {e}");
                    return (None, tap, environment);
                }
            }
        }
        sink.append(crate::audio_tap::Tapped::new(EnvironmentalSource::new(source, environment.clone()), tap.clone()));
        (None, tap, environment)
    }

    /// Whether voices record samples for the audio analysis (off: no cost on the audio thread).
    pub fn set_tap_enabled(&self, on: bool) {
        self.tap.set_enabled(on);
    }

    /// Sum of every voice's newest samples (mono, 48 kHz, newest last) into `out`,
    /// scaled by each sink's current volume and speed. See `audio_tap`.
    pub fn read_tap(&self, out: &mut [f32]) {
        for v in self.sinks.values() {
            v.tap.set_gain(v.sink.volume());
            v.tap.set_speed(v.sink.speed());
        }
        for s in &self.oneshots {
            s.tap.set_gain(s.sink.volume());
            s.tap.set_speed(s.sink.speed());
        }
        self.tap.mix(out);
    }

    pub fn tap(&self) -> &Arc<crate::audio_tap::AudioTap> {
        &self.tap
    }

    /// (Re)start an object's `AudioSource`.
    pub fn start_cue(&mut self, cue: &AudioCue, listener: Mat4, project_root: &Path, mixer: &AudioMixer) {
        if cue.clip.is_empty() {
            return;
        }
        let Some((sink, decoder)) = self.open(project_root, &cue.clip) else {
            return;
        };
        let ear = listener.w_axis.truncate();
        sink.set_speed(cue.pitch.max(0.01));
        #[cfg(feature = "steam-audio")]
        let indirect = Indirect {
            reflections: (cue.spatial && cue.ambisonic_order == 0).then(|| std::sync::Arc::new(std::sync::Mutex::new(None))),
            pathing: (cue.spatial && cue.pathing && cue.ambisonic_order == 0).then(|| std::sync::Arc::new(std::sync::Mutex::new(None))),
        };
        #[cfg(feature = "steam-audio")]
        let init = Some(slot_init(cue, listener, &indirect));
        #[cfg(not(feature = "steam-audio"))]
        let init = cue.spatial.then(|| slot_init(cue, listener));
        let (slot, tap, environment) = if cue.looping {
            self.append(&sink, decoder.repeat_infinite(), init)
        } else {
            self.append(&sink, decoder, init)
        };
        // Steam Audio applies the distance falloff itself.
        let gain = if slot.is_some() || !cue.spatial { cue.volume.max(0.0) } else { cue.gain(ear) };
        sink.set_volume(gain * mixer.effective_gain(&cue.bus));
        let voice = Voice {
            environment,
            fade: MixFade::default(),
            sink,
            tap,
            #[cfg(feature = "steam-audio")]
            indirect: if slot.is_some() { indirect } else { Indirect::default() },
            slot,
        };
        if let Some(old) = self.sinks.insert(cue.object, voice) {
            old.sink.stop();
        }
    }

    pub fn stop_object(&mut self, object: ObjectId) {
        if let Some(voice) = self.sinks.remove(&object) {
            voice.sink.stop();
        }
    }

    /// Play a clip once. Spatial clips (with a position) follow the listener
    /// while they play; falloff is linear from 1 m to 30 m.
    #[allow(clippy::too_many_arguments)]
    pub fn play_oneshot(
        &mut self,
        project_root: &Path,
        clip: &str,
        position: Option<Vec3>,
        volume: f32,
        listener: Mat4,
        mixer: &AudioMixer,
    ) {
        let Some((sink, decoder)) = self.open(project_root, clip) else {
            return;
        };
        let ear = listener.w_axis.truncate();
        let init = position.map(|p| oneshot_init(p, listener));
        let (slot, tap, environment) = self.append(&sink, decoder, init);
        let gain = match (position, &slot) {
            (Some(p), None) => volume * oneshot_attenuation(p, ear),
            _ => volume,
        };
        sink.set_volume(gain.max(0.0) * mixer.effective_gain("sfx"));
        let spatial = position.zip(slot);
        self.next_shot += 1;
        self.oneshots.push(OneShot { id: self.next_shot, sink, tap, spatial, position, environment, fade: MixFade::default() });
    }

    fn open(&self, project_root: &Path, clip: &str) -> Option<(Sink, Decoder<BufReader<std::fs::File>>)> {
        let path = zestengine_core::asset_ref::resolve_path(project_root, clip);
        let file = std::fs::File::open(&path)
            .map_err(|e| tracing::error!("opening audio {}: {e}", path.display()))
            .ok()?;
        let decoder = Decoder::new(BufReader::new(file))
            .map_err(|e| tracing::error!("decoding audio {}: {e}", path.display()))
            .ok()?;
        let sink = Sink::try_new(&self.handle)
            .map_err(|e| tracing::error!("creating audio sink: {e}"))
            .ok()?;
        Some((sink, decoder))
    }

    /// Send the scene's acoustic geometry (static meshes with materials,
    /// moving occluders) and audio bake to the simulation when they change,
    /// and track probe volumes and occluder transforms.
    #[allow(unused_variables)]
    pub fn sync_geometry(&mut self, scene: &LoadedScene) {
        #[cfg(feature = "steam-audio")]
        if let Some(sim) = self.simulation.as_ref() {
            scene.objects.query(|ecs, order| {
                let (statics, dynamics) = acoustic_objects_in(scene, ecs, order);
                let key = acoustic_key_in(ecs, &statics, &dynamics);
                if self.geometry_key != Some(key) {
                    self.geometry_key = Some(key);
                    sim.set_geometry(gather_geometry_in(scene, ecs, &statics, &dynamics));
                    self.dynamic = dynamics.iter().filter_map(|&entity| ecs.object_id(entity)).collect();
                }
                let bake_key = scene.audio_bake.as_ref().map(|b| std::sync::Arc::as_ptr(b) as usize);
                if self.bake_key != bake_key {
                    self.bake_key = bake_key;
                    sim.set_bake(scene.audio_bake.clone());
                }
                self.probe_boxes = if scene.audio_bake.is_some() {
                    probe_volumes_in(scene, ecs, order).into_iter().map(|volume| volume.transform).collect()
                } else { Vec::new() };
                self.dynamic_transforms = self.dynamic.iter().filter_map(|&id| Some((id, ecs.world_transform(ecs.entity(id)?)?))).collect();
            });
        }
    }

    /// Resolve authored regions without rebuilding Steam Audio or stopping voices.
    pub fn update_environment(&mut self, scene: &LoadedScene, cues: &[AudioCue], listener: Mat4, root: &Path, dt: f32) {
        self.environment.sync(scene, root, dt);
        let ear = listener.w_axis.truncate();
        for (id, voice) in &mut self.sinks {
            let source = cues.iter().find(|c| c.object == *id)
                .filter(|c| c.spatial && c.ambisonic_order == 0).map(|c| c.position);
            let (target, seconds) = self.environment.target(ear, source);
            *voice.environment.lock().unwrap() = voice.fade.step(target, seconds, dt);
        }
        for shot in &mut self.oneshots {
            let (target, seconds) = self.environment.target(ear, shot.position);
            *shot.environment.lock().unwrap() = shot.fade.step(target, seconds, dt);
        }
    }

    /// Per-frame update while playing: drop finished sinks, stop sources whose
    /// object is gone or inactive (absent from `cues`), and refresh each live
    /// voice from the listener pose.
    #[allow(unused_variables)]
    pub fn update(&mut self, cues: &[AudioCue], listener: Mat4, mixer: &AudioMixer) {
        let ear = listener.w_axis.truncate();
        self.oneshots.retain(|s| !s.sink.empty());
        self.sinks.retain(|id, v| !v.sink.empty() && cues.iter().any(|c| c.object == *id));
        for cue in cues {
            let Some(voice) = self.sinks.get(&cue.object) else {
                continue;
            };
            match &voice.slot {
                #[cfg(feature = "steam-audio")]
                Some(slot) => {
                    if let Some(sa) = self.steam.as_ref() {
                        let mut state = sa.state_for(&cue.emitter(ear), listener);
                        state.reverb_gain = voice.environment.lock().unwrap().reverb_gain;
                        let mut s = slot.lock().unwrap();
                        *s = steam::VoiceState { occlusion: s.occlusion, transmission: s.transmission, ..state };
                    }
                    voice.sink.set_volume(cue.volume.max(0.0) * mixer.effective_gain(&cue.bus));
                }
                _ => voice.sink.set_volume(cue.gain(ear) * mixer.effective_gain(&cue.bus)),
            }
        }
        #[cfg(feature = "steam-audio")]
        if let Some(sa) = self.steam.as_ref() {
            for shot in &self.oneshots {
                if let Some((position, slot)) = &shot.spatial {
                    let SlotInit(emitter, _, _) = oneshot_init(*position, listener);
                    let mut state = sa.state_for(&emitter, listener);
                    state.reverb_gain = shot.environment.lock().unwrap().reverb_gain;
                    let mut s = slot.lock().unwrap();
                    *s = steam::VoiceState { occlusion: s.occlusion, transmission: s.transmission, ..state };
                }
            }
        }
        #[cfg(feature = "steam-audio")]
        self.update_simulation(cues, listener);
    }

    /// Submit voices to the simulation and apply its latest direct results.
    #[cfg(feature = "steam-audio")]
    fn update_simulation(&self, cues: &[AudioCue], listener: Mat4) {
        use steam::{SimVoice, VoiceKey};
        let Some(sim) = self.simulation.as_ref() else {
            return;
        };
        let mut voices: Vec<(SimVoice, &Slot)> = Vec::new();
        for cue in cues.iter().filter(|c| c.spatial && c.ambisonic_order == 0) {
            if let Some(voice) = self.sinks.get(&cue.object)
                && let Some(slot) = voice.slot.as_ref()
            {
                let sim_voice = SimVoice {
                    key: VoiceKey::Object(cue.object),
                    position: cue.position,
                    occlusion: cue.occlusion,
                    reflections: voice.indirect.reflections.clone().map(|r| (r, cue.reflections)),
                    pathing: voice.indirect.pathing.clone(),
                };
                voices.push((sim_voice, slot));
            }
        }
        for shot in &self.oneshots {
            if let Some((position, slot)) = &shot.spatial {
                voices.push((
                    SimVoice { key: VoiceKey::Shot(shot.id), position: *position, occlusion: true, reflections: None, pathing: None },
                    slot,
                ));
            }
        }
        let ear = listener.w_axis.truncate();
        let in_bake = self.probe_boxes.iter().any(|b| {
            let p = b.inverse().transform_point3(ear);
            p.abs().max_element() <= 0.5
        });
        sim.submit(listener, voices.iter().map(|(v, _)| v.clone()).collect(), self.dynamic_transforms.clone(), in_bake);
        let results = sim.results();
        for (voice, slot) in voices {
            let (occ, transmission) = if voice.occlusion { results.get(&voice.key).copied().unwrap_or((1.0, [1.0; 3])) } else { (1.0, [1.0; 3]) };
            let mut s = slot.lock().unwrap();
            s.occlusion = occ;
            s.transmission = transmission;
        }
    }
}

/// A voice through the underwater muffle (see `underwater::MuffleFilter`).
struct Muffled<S> {
    inner: S,
    filter: crate::underwater::MuffleFilter,
    target: Arc<AtomicU32>,
    muffle: f32,
    channel: usize,
}

impl<S: Source<Item = f32>> Muffled<S> {
    fn new(inner: S, target: Arc<AtomicU32>) -> Self {
        let filter = crate::underwater::MuffleFilter::new(inner.channels(), inner.sample_rate());
        Self { inner, filter, target, muffle: 0.0, channel: 0 }
    }
}

impl<S: Source<Item = f32>> Iterator for Muffled<S> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let x = self.inner.next()?;
        if self.channel == 0 {
            self.muffle = f32::from_bits(self.target.load(Ordering::Relaxed));
            if self.filter.channels() != self.inner.channels().max(1) as usize {
                self.filter = crate::underwater::MuffleFilter::new(self.inner.channels(), self.inner.sample_rate());
            }
        }
        let y = self.filter.process(x, self.channel, self.muffle);
        self.channel = (self.channel + 1) % self.filter.channels();
        Some(y)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S: Source<Item = f32>> Source for Muffled<S> {
    fn current_frame_len(&self) -> Option<usize> {
        self.inner.current_frame_len()
    }
    fn channels(&self) -> u16 {
        self.inner.channels()
    }
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        self.inner.total_duration()
    }
}

/// Kept out of `AudioEngine::append`, whose `FromSample<S::Item>` bound
/// would otherwise be picked for the voice's `f32` samples.
#[cfg(feature = "steam-audio")]
fn append_voice<V: Source<Item = f32> + Send + 'static>(sink: &Sink, voice: V) {
    sink.append(voice);
}

/// Emitter, listener and rendering options a new voice starts from.
#[cfg(feature = "steam-audio")]
struct SlotInit(steam::Emitter, Mat4, steam::VoiceOptions);
#[cfg(not(feature = "steam-audio"))]
struct SlotInit;

#[cfg(feature = "steam-audio")]
fn slot_init(cue: &AudioCue, listener: Mat4, indirect: &Indirect) -> SlotInit {
    SlotInit(
        cue.emitter(listener.w_axis.truncate()),
        listener,
        steam::VoiceOptions {
            spatial: cue.spatial,
            ambisonic_order: cue.ambisonic_order as u32,
            via_ambisonics: cue.via_ambisonics,
            reflections: indirect.reflections.clone(),
            pathing: indirect.pathing.clone(),
        },
    )
}

#[cfg(not(feature = "steam-audio"))]
fn slot_init(_cue: &AudioCue, _listener: Mat4) -> SlotInit {
    SlotInit
}

#[allow(unused_variables)]
fn oneshot_init(position: Vec3, listener: Mat4) -> SlotInit {
    #[cfg(feature = "steam-audio")]
    return SlotInit(
        steam::Emitter {
            transform: Mat4::from_translation(position),
            attenuation: oneshot_attenuation(position, listener.w_axis.truncate()),
            air_absorption: true,
            directivity: 0.0,
            directivity_sharpness: 1.0,
            hrtf: true,
        },
        listener,
        steam::VoiceOptions { spatial: true, ..Default::default() },
    );
    #[cfg(not(feature = "steam-audio"))]
    SlotInit
}
}

/// Headless build: no audio output exists, so no engine is ever made and the
/// session's `Option<AudioEngine>` stays `None`. The methods keep `PlaySession`
/// compiling unchanged.
#[cfg(not(feature = "client"))]
pub struct AudioEngine {
    never: std::convert::Infallible,
}

#[cfg(not(feature = "client"))]
impl AudioEngine {
    pub fn new() -> Option<Self> {
        None
    }
    pub fn update_environment(&mut self, _scene: &LoadedScene, _cues: &[AudioCue], _listener: Mat4, _root: &Path, _dt: f32) {}
    pub fn configure(&mut self, _settings: &SpatialAudioSettings, _project_root: &Path) {
        match self.never {}
    }
    pub fn stop_all(&mut self) {
        match self.never {}
    }
    pub fn set_tap_enabled(&self, _on: bool) {
        match self.never {}
    }
    pub fn read_tap(&self, _out: &mut [f32]) {
        match self.never {}
    }
    pub fn set_underwater(&mut self, _under: Option<&zestengine_core::UnderwaterSettings>, _dt: f32, _project_root: &Path, _mixer: &AudioMixer) {
        match self.never {}
    }
    pub fn start_cue(&mut self, _cue: &AudioCue, _listener: Mat4, _project_root: &Path, _mixer: &AudioMixer) {
        match self.never {}
    }
    pub fn stop_object(&mut self, _object: ObjectId) {
        match self.never {}
    }
    pub fn play_oneshot(&mut self, _project_root: &Path, _clip: &str, _position: Option<Vec3>, _volume: f32, _listener: Mat4, _mixer: &AudioMixer) {
        match self.never {}
    }
    pub fn sync_geometry(&mut self, _scene: &LoadedScene) {
        match self.never {}
    }
    pub fn update(&mut self, _cues: &[AudioCue], _listener: Mat4, _mixer: &AudioMixer) {
        match self.never {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(spatial: bool, rolloff: AudioRolloff) -> AudioCue {
        let mut c = AudioCue::from_source(ObjectId::new(), &AudioSource::default(), Vec3::ZERO);
        c.spatial = spatial;
        c.rolloff = rolloff;
        c.min_distance = 1.0;
        c.max_distance = 11.0;
        c.volume = 0.5;
        c
    }

    #[cfg(feature = "steam-audio")]
    #[test]
    fn native_acoustic_geometry_and_probes_follow_live_values_and_order() {
        use crate::scene::{SceneObject, RenderInfo};
        use zestengine_core::{AcousticMaterial, AcousticPreset, AudioProbeVolume, Disabled, Transform, ecs::{ComponentSet, Enabled, Parent}};
        let mut scene = LoadedScene::empty();
        let mut asset = zestengine_assets::test_scene();
        asset.meshes = vec![zestengine_assets::primitive_mesh(zestengine_assets::PrimitiveShape::Cube)];
        asset.instances = vec![zestengine_assets::Instance::single("Wall", 0, 0, Mat4::IDENTITY)];
        scene.add_asset_scene_with(None, &asset, None).unwrap();
        *scene.objects[0].transform_mut() = Transform { translation: Vec3::X * 10.0, scale: Vec3::splat(2.0), ..Default::default() };
        let wall = scene.objects[1].id();
        let parent = scene.objects[0].id();
        scene.objects[1].set_parent(Some(0));
        *scene.objects[1].static_geometry_mut() = true;
        scene.objects[1].components.push(Box::new(AcousticMaterial { preset: AcousticPreset::Custom, absorption: [0.2; 3], ..Default::default() }));
        let render = scene.objects[1].render().unwrap();
        scene.objects[1].extra_render_mut().push(render);
        scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![Box::new(AcousticMaterial { dynamic: true, ..Default::default() })]))
            .with_render(Some(RenderInfo { mesh: render.mesh, material: render.material })).with_transform(Transform { translation: Vec3::Z * 3.0, ..Default::default() }));
        let moving = scene.objects[2].id();
        scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![
            Box::new(Disabled(Box::new(AudioProbeVolume { spacing: 99.0, ..Default::default() }))),
            Box::new(AudioProbeVolume { size: [4.0, 2.0, 6.0], spacing: 1.0, ..Default::default() }),
            Box::new(AudioProbeVolume { spacing: 3.0, ..Default::default() }),
        ])).with_parent(Some(0)).with_transform(Transform { translation: Vec3::Y, ..Default::default() }));
        let probe = scene.objects[3].id();
        let (geometry, volumes) = bake_input(&scene);
        let (positions, indices) = { let (positions, indices) = scene.mesh_geometry(render.mesh).unwrap(); (positions.to_vec(), indices.to_vec()) };
        assert_eq!(geometry.vertices.len(), positions.len() * 2, "all static material slots, no dynamic meshes in the bake");
        assert_eq!(geometry.triangles.len(), indices.len() / 3 * 2);
        assert!(geometry.dynamic.is_empty());
        assert_eq!(geometry.vertices[0], (Vec3::X * 10.0 + positions[0] * 2.0).to_array());
        assert_eq!(geometry.materials[geometry.material_indices[0]].0, [0.2; 3]);
        assert_eq!(volumes.len(), 1);
        assert_eq!(volumes[0].spacing, 1.0);
        assert_eq!(volumes[0].transform.w_axis.truncate(), Vec3::new(10.0, 2.0, 0.0));
        assert_eq!(volumes[0].transform.x_axis.truncate().length(), 8.0);
        let live = scene.objects.query(|ecs, order| {
            let (fixed, moving) = acoustic_objects_in(&scene, ecs, order);
            gather_geometry_in(&scene, ecs, &fixed, &moving)
        });
        assert_eq!(live.dynamic.len(), 1);
        assert_eq!(live.dynamic[0].id, moving);
        assert_eq!(live.dynamic[0].vertices[0], positions[0].to_array(), "moving occluder vertices stay in local space");
        scene.objects.swap(1, 3);
        scene.objects[1].components.set_enabled(1, false);
        scene.objects.query(|ecs, _| {
            ecs.get_mut::<Transform>(ecs.entity(parent).unwrap()).unwrap().translation = Vec3::Y * 5.0;
            ecs.get_mut::<AcousticMaterial>(ecs.entity(wall).unwrap()).unwrap().absorption = [0.8; 3];
            ecs.get_mut::<AudioProbeVolume>(ecs.entity(probe).unwrap()).unwrap().height = 2.5;
        });
        let (geometry, volumes) = bake_input(&scene);
        assert_eq!(geometry.vertices[0], (Vec3::Y * 5.0 + positions[0] * 2.0).to_array());
        assert_eq!(geometry.materials[geometry.material_indices[0]].0, [0.8; 3]);
        assert_eq!(volumes[0].spacing, 3.0);
        assert_eq!(volumes[0].height, 2.5);
        scene.objects.query(|ecs, _| ecs.get_mut::<Enabled>(ecs.entity(parent).unwrap()).unwrap().0 = false);
        let (geometry, volumes) = bake_input(&scene);
        assert!(geometry.vertices.is_empty() && volumes.is_empty());
        scene.objects.query(|ecs, _| ecs.get_mut::<Parent>(ecs.entity(probe).unwrap()).unwrap().0 = None);
        assert_eq!(bake_input(&scene).1.len(), 1);
    }

    #[cfg(feature = "steam-audio")]
    #[test]
    fn native_acoustic_cache_changes_only_for_geometry_inputs_and_static_poses() {
        use crate::scene::{SceneObject, RenderInfo};
        use zestengine_core::{AcousticMaterial, Transform, ecs::ComponentSet};
        let mut scene = LoadedScene::empty();
        for fixed in [true, false] {
            scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![Box::new(AcousticMaterial { dynamic: true, ..Default::default() })]))
                .with_render(Some(RenderInfo { mesh: 0, material: 0 })).with_static_geometry(fixed));
        }
        let fixed = scene.objects[0].id();
        let moving = scene.objects[1].id();
        let key = |scene: &LoadedScene| scene.objects.query(|ecs, order| {
            let (fixed, moving) = acoustic_objects_in(scene, ecs, order);
            acoustic_key_in(ecs, &fixed, &moving)
        });
        let first = key(&scene);
        scene.objects.query(|ecs, _| ecs.get_mut::<Transform>(ecs.entity(moving).unwrap()).unwrap().translation = Vec3::X);
        assert_eq!(key(&scene), first, "moving occluders receive poses separately");
        scene.objects.query(|ecs, _| ecs.get_mut::<Transform>(ecs.entity(fixed).unwrap()).unwrap().translation = Vec3::Y);
        let moved = key(&scene);
        assert_ne!(moved, first, "static meshes bake their world pose into geometry");
        *scene.objects[0].static_geometry_mut() = false;
        let reclassified = key(&scene);
        assert_ne!(reclassified, moved);
        scene.objects[0].components.set_enabled(0, false);
        assert_ne!(key(&scene), reclassified, "a disabled material no longer selects a moving occluder");
        scene.objects.remove(0);
        let remaining = key(&scene);
        scene.objects.query(|ecs, _| ecs.get_mut::<AcousticMaterial>(ecs.entity(moving).unwrap()).unwrap().dynamic = false);
        assert_ne!(key(&scene), remaining);
    }

    #[test]
    fn distance_fallback_gain() {
        let c = cue(true, AudioRolloff::Linear);
        assert_eq!(c.gain(Vec3::new(0.5, 0.0, 0.0)), 0.5);
        assert!((c.gain(Vec3::new(6.0, 0.0, 0.0)) - 0.25).abs() < 1e-5);
        assert_eq!(c.gain(Vec3::new(20.0, 0.0, 0.0)), 0.0);
        let log = cue(true, AudioRolloff::Logarithmic);
        let mid = log.gain(Vec3::new(6.0, 0.0, 0.0));
        assert!(mid > 0.0 && mid < 0.25, "inverse distance falls faster early: {mid}");
        assert_eq!(cue(false, AudioRolloff::Linear).gain(Vec3::new(50.0, 0.0, 0.0)), 0.5);
    }

    #[test]
    fn listener_pose_and_cue_orientation_come_from_the_scene() {
        use crate::scene::SceneObject;
        let mut scene = LoadedScene::empty();
        let turn = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        for (pos, components) in [
            (Vec3::new(1.0, 2.0, 3.0), vec![Box::new(AudioListener::default()) as Box<dyn zestengine_core::Component>]),
            (Vec3::new(-4.0, 0.0, 0.0), vec![Box::new(AudioSource::default()) as Box<dyn zestengine_core::Component>]),
        ] {
            scene.objects.push(SceneObject::from_components(zestengine_core::ecs::ComponentSet::from_vec(components)).with_parent(None).with_component_roles(Default::default()).with_extra_render(Vec::new()).with_source(zestengine_assets::ObjectSource::Empty).with_net_role(Default::default()).with_render(None).with_static_geometry(false).with_layer(0).with_id(ObjectId::new()).with_name(String::new()).with_enabled(true).with_transform(zestengine_core::Transform { translation: pos, rotation: turn, scale: Vec3::splat(2.0) }));
        }
        let pose = listener_pose(&scene).expect("listener");
        assert!(pose.w_axis.truncate().distance(Vec3::new(1.0, 2.0, 3.0)) < 1e-5);
        assert!((pose.x_axis.truncate().length() - 1.0).abs() < 1e-5, "scale removed");
        let cues = gather_cues(&scene);
        assert_eq!(cues.len(), 1);
        assert!(cues[0].rotation.dot(turn).abs() > 0.9999, "{:?} vs {turn:?}", cues[0].rotation);
        assert!(cues[0].position.distance(Vec3::new(-4.0, 0.0, 0.0)) < 1e-5);
    }

    #[test]
    fn oneshot_falloff() {
        assert_eq!(oneshot_attenuation(Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)), 1.0);
        assert_eq!(oneshot_attenuation(Vec3::ZERO, Vec3::new(40.0, 0.0, 0.0)), 0.0);
    }
}
