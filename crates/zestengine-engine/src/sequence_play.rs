use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use glam::{Quat, Vec3};
use zestengine_assets::Sequencer;
use zestengine_core::{ActionEvent, CameraComponent, ObjectId, SequenceFinished, SequencePlayer};

use crate::scene::LoadedScene;

const RELOAD_POLL: Duration = Duration::from_millis(500);

struct Cached {
    sequence: Option<Arc<Sequencer>>,
    path: PathBuf,
    stamp: Option<(SystemTime, u64)>,
    checked: Instant,
}

#[derive(Default)]
pub struct SequenceAssets {
    cache: HashMap<String, Cached>,
}

fn stamp_of(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

impl SequenceAssets {
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn insert(&mut self, reference: &str, sequence: Sequencer) {
        self.cache.insert(
            reference.to_owned(),
            Cached { sequence: Some(Arc::new(sequence)), path: PathBuf::new(), stamp: None, checked: Instant::now() + Duration::from_secs(u32::MAX as u64) },
        );
    }

    pub fn get(&mut self, root: &Path, reference: &str) -> Option<Arc<Sequencer>> {
        if reference.trim().is_empty() {
            return None;
        }
        let now = Instant::now();
        if let Some(cached) = self.cache.get_mut(reference) {
            if now < cached.checked || now.duration_since(cached.checked) < RELOAD_POLL {
                return cached.sequence.clone();
            }
            cached.checked = now;
            if stamp_of(&cached.path) == cached.stamp {
                return cached.sequence.clone();
            }
        }
        let path = zestengine_core::asset_ref::resolve_path(root, reference);
        let stamp = stamp_of(&path);
        let sequence = match zestengine_assets::load_sequencer(&path) {
            Ok(sequence) => Some(Arc::new(sequence)),
            Err(e) => {
                tracing::warn!("Sequence Player: {reference}: {e:#}");
                None
            }
        };
        self.cache.insert(reference.to_owned(), Cached { sequence: sequence.clone(), path, stamp, checked: now });
        sequence
    }
}

pub fn apply_sequence(scene: &mut LoadedScene, sequence: &Sequencer, time: f32) {
    for pose in sequence.poses(time) {
        let Some(i) = scene.index_of(pose.target) else {
            continue;
        };
        let object = &mut scene.objects[i];
        if let Some(v) = pose.translation {
            object.transform_mut().translation = Vec3::from_array(v);
        }
        if let Some(v) = pose.rotation {
            object.transform_mut().rotation = Quat::from_array(v);
        }
        if let Some(v) = pose.scale {
            object.transform_mut().scale = Vec3::from_array(v);
        }
        if let Some(fov) = pose.fov
            && let Some(mut camera) = object.components.get_typed_mut::<CameraComponent>()
        {
            camera.fov_deg = fov;
        }
    }
}

#[derive(Default)]
pub struct SequenceOutput {
    pub events: Vec<ActionEvent>,
    pub finished: Vec<SequenceFinished>,
    pub commands: Vec<zestengine_core::ComponentCommand>,
}

pub fn tick(scene: &mut LoadedScene, assets: &mut SequenceAssets, root: &Path, dt: f32) -> SequenceOutput {
    let mut out = SequenceOutput::default();
    let players: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| scene.objects[i].components.contains_type::<SequencePlayer>() && scene.is_active(i))
        .collect();
    let mut held_blend = None;
    for i in players {
        let id: ObjectId = scene.objects[i].id();
        let reference = match scene.objects[i].components.get_typed::<SequencePlayer>() {
            Some(player) => player.sequence.clone(),
            None => continue,
        };
        let Some(sequence) = assets.get(root, &reference) else {
            continue;
        };
        let (step, time, apply, take_camera, release_camera, blend) = {
            let Some(mut player) = scene.objects[i].components.get_typed_mut::<SequencePlayer>() else {
                continue;
            };
            if !player.started {
                player.started = true;
                if player.play_on_start {
                    player.time = 0.0;
                    player.playing = true;
                }
            }
            let step = player.advance(dt, sequence.length, sequence.looping);
            let apply = player.applied != Some(player.time);
            player.applied = Some(player.time);
            let wanted = (player.playing && player.use_camera).then(|| sequence.camera_at(player.time)).flatten();
            let take_camera = wanted.filter(|camera| player.camera_now != Some(*camera));
            let release_camera = std::mem::take(&mut player.release_camera) && player.camera_before.is_some();
            let holds_camera = player.use_camera && !release_camera && (wanted.is_some() || player.camera_now.is_some());
            let blend = if holds_camera { sequence.camera_blend_at(player.time) } else { None };
            (step, player.time, apply, take_camera, release_camera, blend)
        };
        if let Some(camera) = take_camera.filter(|camera| scene.index_of(*camera).is_some()) {
            let before = scene.active_camera_override();
            scene.set_active_camera(camera);
            if let Some(mut player) = scene.objects[i].components.get_typed_mut::<SequencePlayer>() {
                if player.camera_before.is_none() {
                    player.camera_before = Some(before);
                }
                player.camera_now = Some(camera);
            }
        }
        if let Some(blend) = blend {
            held_blend = Some(crate::scene::CameraBlend { from: blend.from, weight: blend.weight });
        }
        if apply {
            apply_sequence(scene, &sequence, time);
        }
        if step.moved_forward() {
            let names: Vec<String> = if step.wrapped {
                sequence.events_between(step.from, sequence.length).into_iter().chain(sequence.events_between(-1.0, step.to)).map(str::to_owned).collect()
            } else {
                sequence.events_between(step.from, step.to).into_iter().map(str::to_owned).collect()
            };
            out.events.extend(names.into_iter().map(|name| ActionEvent { name, source: id, instigator: None }));
            let cues = if step.wrapped {
                let mut cues = sequence.audio_between(step.from, sequence.length);
                cues.extend(sequence.audio_between(-1.0, step.to));
                cues
            } else {
                sequence.audio_between(step.from, step.to)
            };
            for cue in cues {
                if !cue.source.is_nil() {
                    out.commands.push(zestengine_core::ComponentCommand::PlayAudio(cue.source));
                }
                if !cue.clip.trim().is_empty() {
                    out.commands.push(zestengine_core::ComponentCommand::PlaySound { clip: cue.clip.clone(), position: None, volume: cue.volume.clamp(0.0, 4.0) });
                }
            }
        }
        if release_camera {
            let before = scene.objects[i].components.get_typed_mut::<SequencePlayer>().and_then(|mut player| {
                player.camera_now = None;
                player.camera_before.take()
            });
            if let Some(before) = before {
                scene.restore_active_camera(before);
            }
        }
        if step.finished {
            out.finished.push(SequenceFinished { object: id, sequence: reference });
        }
    }
    scene.set_camera_blend(held_blend);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{ObjectSource, PrimitiveShape, SceneEntry, SceneFile, SeqEvent, SeqTrack, SeqTrackKind};
    use zestengine_core::{ComponentRegistry, SequenceLoop};

    fn entry(name: &str, source: ObjectSource) -> SceneEntry {
        SceneEntry {
            id: String::new(),
            name: name.into(),
            source,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: zestengine_assets::MaterialRef::Model,
            extra_materials: Vec::new(),
            renderer_split: false,
            parent: None,
            components: Vec::new(),
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    fn scene() -> (LoadedScene, ObjectId, ObjectId, ObjectId) {
        let file = SceneFile {
            entries: vec![
                entry("Door", ObjectSource::Primitive { shape: PrimitiveShape::Cube }),
                entry("Shot", ObjectSource::Camera),
                entry("Director", ObjectSource::Empty),
            ],
            ..Default::default()
        };
        let mut scene = LoadedScene::load_scene_file(None, &file, Path::new("."), &ComponentRegistry::with_builtins(), &mut crate::shaders::ShaderLibrary::default()).expect("loads without a renderer");
        scene.ensure_camera_components(&ComponentRegistry::with_builtins());
        let ids = (scene.objects[0].id(), scene.objects[1].id(), scene.objects[2].id());
        (scene, ids.0, ids.1, ids.2)
    }

    fn sequence(door: ObjectId, camera: ObjectId, looping: bool) -> Sequencer {
        let mut slide = SeqTrack::new(SeqTrackKind::Transform, door);
        slide.translation.set_key(0.0, [0.0, 0.0, 0.0, 0.0]);
        slide.translation.set_key(2.0, [4.0, 0.0, 0.0, 0.0]);
        let mut shot = SeqTrack::new(SeqTrackKind::Camera, camera);
        shot.translation.set_key(0.0, [0.0, 1.0, 5.0, 0.0]);
        if let Some(fov) = &mut shot.fov {
            fov.set_key(0.0, [60.0, 0.0, 0.0, 0.0]);
            fov.set_key(2.0, [30.0, 0.0, 0.0, 0.0]);
        }
        Sequencer {
            length: 2.0,
            looping,
            tracks: vec![slide, shot],
            events: vec![SeqEvent { time: 1.5, name: "Slam".into() }, SeqEvent { time: 0.5, name: "Creak".into() }],
            ..Default::default()
        }
    }

    fn add_player(scene: &mut LoadedScene, director: ObjectId, player: SequencePlayer) {
        let i = scene.index_of(director).unwrap();
        scene.objects[i].components.push(Box::new(player));
    }

    fn player(scene: &LoadedScene, director: ObjectId) -> SequencePlayer {
        let i = scene.index_of(director).unwrap();
        scene.objects[i].components.get_typed::<SequencePlayer>().unwrap().clone()
    }

    #[test]
    fn a_player_drives_objects_and_the_camera_and_reports_events_once() {
        let (mut scene, door, camera, director) = scene();
        let mut assets = SequenceAssets::default();
        assets.insert("cut.seq", sequence(door, camera, false));
        add_player(&mut scene, director, SequencePlayer::new("cut.seq"));
        let root = Path::new(".");
        let out = tick(&mut scene, &mut assets, root, 1.0);
        let door_i = scene.index_of(door).unwrap();
        let cam_i = scene.index_of(camera).unwrap();
        assert!((scene.objects[door_i].translation().x - 2.0).abs() < 1e-4);
        assert!((scene.objects[cam_i].components.get_typed::<CameraComponent>().unwrap().fov_deg - 45.0).abs() < 1e-3);
        assert_eq!(scene.active_camera_override(), Some(camera), "the sequence's camera takes the view");
        assert_eq!(out.events.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["Creak"]);
        assert!(out.events.iter().all(|e| e.source == director));
        let out = tick(&mut scene, &mut assets, root, 5.0);
        assert_eq!(out.events.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["Slam"]);
        assert_eq!(out.finished, vec![SequenceFinished { object: director, sequence: "cut.seq".into() }]);
        assert!((scene.objects[door_i].translation().x - 4.0).abs() < 1e-4);
        assert_eq!(scene.active_camera_override(), None, "the view goes back when the sequence ends");
        let p = player(&scene, director);
        assert!(!p.playing && p.time == 2.0);
        scene.objects[door_i].transform_mut().translation.x = 9.0;
        let out = tick(&mut scene, &mut assets, root, 1.0);
        assert!(out.events.is_empty() && out.finished.is_empty());
        assert_eq!(scene.objects[door_i].translation().x, 9.0, "a finished sequence lets go of its objects");
    }

    #[test]
    fn a_blended_cut_eases_the_view_from_one_camera_to_the_next() {
        use zestengine_assets::SeqCut;
        let file = SceneFile {
            entries: vec![
                entry("Wide", ObjectSource::Camera),
                SceneEntry { translation: [10.0, 0.0, 0.0], ..entry("Close", ObjectSource::Camera) },
                entry("Director", ObjectSource::Empty),
            ],
            ..Default::default()
        };
        let mut scene = LoadedScene::load_scene_file(None, &file, Path::new("."), &ComponentRegistry::with_builtins(), &mut crate::shaders::ShaderLibrary::default()).expect("loads without a renderer");
        scene.ensure_camera_components(&ComponentRegistry::with_builtins());
        let (wide, close, director) = (scene.objects[0].id(), scene.objects[1].id(), scene.objects[2].id());
        let seq = Sequencer { length: 4.0, tracks: vec![SeqTrack::new(SeqTrackKind::Camera, wide)], cuts: vec![SeqCut { time: 1.0, camera: close, blend: 2.0 }], ..Default::default() };
        let mut assets = SequenceAssets::default();
        assets.insert("blend.seq", seq);
        add_player(&mut scene, director, SequencePlayer::new("blend.seq"));
        let root = Path::new(".");
        let eye = |scene: &LoadedScene| scene.main_camera_view_proj(1.0).unwrap().2.x;
        tick(&mut scene, &mut assets, root, 0.5);
        assert_eq!((scene.active_camera_override(), scene.camera_blend()), (Some(wide), None));
        assert_eq!(eye(&scene), 0.0);
        tick(&mut scene, &mut assets, root, 1.5);
        assert_eq!(scene.active_camera_override(), Some(close), "the cut's camera is the active one during the blend");
        let blend = scene.camera_blend().expect("blending");
        assert_eq!(blend.from, wide);
        assert!((blend.weight - 0.5).abs() < 1e-4, "{}", blend.weight);
        assert!((eye(&scene) - 5.0).abs() < 1e-3, "half way between the two cameras: {}", eye(&scene));
        if let Some(mut player) = scene.objects[2].components.get_typed_mut::<SequencePlayer>() {
            player.pause();
        }
        tick(&mut scene, &mut assets, root, 1.0);
        assert!((scene.camera_blend().expect("a paused player holds its blend").weight - 0.5).abs() < 1e-4);
        if let Some(mut player) = scene.objects[2].components.get_typed_mut::<SequencePlayer>() {
            player.play();
        }
        tick(&mut scene, &mut assets, root, 1.5);
        assert_eq!(scene.camera_blend(), None, "the blend is over");
        assert_eq!(eye(&scene), 10.0);
        scene.set_camera_blend(Some(crate::scene::CameraBlend { from: wide, weight: 0.25 }));
        tick(&mut scene, &mut assets, root, 5.0);
        assert_eq!((scene.active_camera_override(), scene.camera_blend()), (None, None), "ending the sequence clears any blend");
    }

    #[test]
    fn cuts_switch_the_active_camera_and_audio_cues_become_commands() {
        use zestengine_assets::{SeqAudio, SeqCut};
        use zestengine_core::ComponentCommand;
        let (mut scene, door, camera, director) = scene();
        let mut seq = sequence(door, camera, false);
        seq.cuts = vec![SeqCut { time: 1.0, camera: door, blend: 0.0 }];
        seq.audio = vec![SeqAudio { time: 0.25, source: door, ..Default::default() }, SeqAudio { time: 1.5, clip: "audio/slam.wav".into(), volume: 0.5, ..Default::default() }];
        let mut assets = SequenceAssets::default();
        assets.insert("cut.seq", seq);
        add_player(&mut scene, director, SequencePlayer::new("cut.seq"));
        let root = Path::new(".");
        let out = tick(&mut scene, &mut assets, root, 0.5);
        assert_eq!(scene.active_camera_override(), Some(camera), "before the first cut the camera track's camera is used");
        assert!(matches!(out.commands.as_slice(), [ComponentCommand::PlayAudio(id)] if *id == door));
        let out = tick(&mut scene, &mut assets, root, 0.6);
        assert_eq!(scene.active_camera_override(), Some(door), "the cut at 1.0 s switched the view");
        assert!(out.commands.is_empty());
        let out = tick(&mut scene, &mut assets, root, 5.0);
        assert!(matches!(out.commands.as_slice(), [ComponentCommand::PlaySound { clip, volume, position: None }] if clip == "audio/slam.wav" && *volume == 0.5));
        assert_eq!(scene.active_camera_override(), None, "the view from before the sequence is back");
        assert!(player(&scene, director).camera_now.is_none());
    }

    #[test]
    fn a_looping_player_wraps_events_and_keeps_the_camera() {
        let (mut scene, door, camera, director) = scene();
        let mut assets = SequenceAssets::default();
        assets.insert("loop.seq", sequence(door, camera, true));
        add_player(&mut scene, director, SequencePlayer::new("loop.seq"));
        let root = Path::new(".");
        tick(&mut scene, &mut assets, root, 1.75);
        let out = tick(&mut scene, &mut assets, root, 1.0);
        assert_eq!(out.events.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["Creak"], "0.75 after the wrap passes 0.5");
        assert!(out.finished.is_empty());
        assert_eq!(scene.active_camera_override(), Some(camera));
        let i = scene.index_of(director).unwrap();
        scene.objects[i].components.get_typed_mut::<SequencePlayer>().unwrap().stop();
        tick(&mut scene, &mut assets, root, 0.1);
        assert_eq!(scene.active_camera_override(), None);
        let door_i = scene.index_of(door).unwrap();
        assert_eq!(scene.objects[door_i].translation().x, 0.0, "Stop rewinds to the first frame");
    }

    #[test]
    fn a_player_that_does_not_start_waits_for_play_and_respects_its_settings() {
        let (mut scene, door, camera, director) = scene();
        let mut assets = SequenceAssets::default();
        assets.insert("wait.seq", sequence(door, camera, true));
        add_player(&mut scene, director, SequencePlayer { play_on_start: false, use_camera: false, looping: SequenceLoop::Once, speed: 2.0, ..SequencePlayer::new("wait.seq") });
        let root = Path::new(".");
        let out = tick(&mut scene, &mut assets, root, 1.0);
        assert!(out.events.is_empty());
        let door_i = scene.index_of(door).unwrap();
        assert_eq!(scene.objects[door_i].translation().x, 0.0);
        let i = scene.index_of(director).unwrap();
        scene.objects[i].components.get_typed_mut::<SequencePlayer>().unwrap().play();
        let out = tick(&mut scene, &mut assets, root, 0.5);
        assert!((scene.objects[door_i].translation().x - 2.0).abs() < 1e-4, "speed 2 covers a second in half");
        assert_eq!(scene.active_camera_override(), None, "use_camera off leaves the view alone");
        assert_eq!(out.events.len(), 1);
        let out = tick(&mut scene, &mut assets, root, 5.0);
        assert_eq!(out.finished.len(), 1, "Once overrides a looping sequence");
        let missing = SequencePlayer::new("nowhere/none.seq");
        add_player(&mut scene, door, missing);
        let out = tick(&mut scene, &mut assets, root, 0.5);
        assert!(out.finished.is_empty());
    }
}
