//! Tools > Bake Audio: bakes Steam Audio listener-centric reverb and pathing
//! for the scene's Audio Probe Volumes on a worker thread and writes the
//! `<scene>.audiobake` sidecar the scene loader picks up.

use std::path::PathBuf;

use super::EngineApp;
use crate::tasks::NotifyLevel;
use crate::tasks::{TaskKind, TaskPayload};

impl EngineApp {
    pub(super) fn start_audio_bake(&mut self) {
        if self.playing {
            self.tasks.notify("Stop play mode to bake audio", NotifyLevel::Info);
            return;
        }
        let Some(scene_path) = self.current_scene_path.clone() else {
            self.tasks.notify("Save the scene before baking audio", NotifyLevel::Warn);
            return;
        };
                {
            let (geometry, volumes) = zestengine_engine::audio::bake_input(&self.scene);
            let config = match zestengine_assets::project_audio_settings(&self.project, &self.project_root) {
                Ok(config) => config,
                Err(e) => { self.tasks.notify(format!("Audio settings: {e:#}"), NotifyLevel::Warn); return; }
            };
            let sidecar = scene_path.with_extension("audiobake");
            self.tasks.spawn("Bake audio", TaskKind::AudioBake, false, move |cancel, progress| {
                let report = progress.clone();
                let bytes = zestengine_engine::audio::steam::bake(
                    &geometry,
                    &volumes,
                    &config,
                    move |done, label| {
                        *report.lock().unwrap() = crate::tasks::TaskProgress::Fraction { done, label: label.to_owned() };
                    },
                    &cancel,
                )?;
                std::fs::write(&sidecar, &bytes).map_err(|e| format!("writing {}: {e}", sidecar.display()))?;
                Ok(TaskPayload::AudioBake { scene_path, bytes })
            });
        }
    }

    #[allow(dead_code)] // fallback for a future no-steam-audio editor build
    pub(super) fn apply_audio_bake(&mut self, scene_path: PathBuf, bytes: Vec<u8>) {
        let kb = bytes.len().div_ceil(1024);
        if self.current_scene_path.as_ref() == Some(&scene_path) {
            self.scene.audio_bake = Some(std::sync::Arc::new(bytes));
        }
        self.tasks.notify(format!("Audio bake complete ({kb} KB)"), NotifyLevel::Info);
    }
}
