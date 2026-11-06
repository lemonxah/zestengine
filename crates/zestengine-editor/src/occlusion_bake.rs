//! Occlusion tab actions: bake the scene's visibility on a worker thread and
//! write the `<scene>.occlusion` sidecar the scene loader picks up.

use std::path::PathBuf;

use super::EngineApp;
use zestengine_engine::occlusion::OcclusionData;
use crate::tasks::{NotifyLevel, TaskKind, TaskPayload, TaskProgress};

impl EngineApp {
    pub(super) fn start_occlusion_bake(&mut self) {
        if self.playing {
            self.tasks.notify("Stop play mode to bake occlusion", NotifyLevel::Info);
            return;
        }
        let Some(scene_path) = self.current_scene_path.clone() else {
            self.tasks.notify("Save the scene before baking occlusion", NotifyLevel::Warn);
            return;
        };
        let input = zestengine_engine::occlusion::bake_input(&self.scene, self.occlusion_ui.settings);
        if input.objects.is_empty() {
            self.tasks.notify("Nothing to bake: mark occluding objects Static", NotifyLevel::Warn);
            return;
        }
        let sidecar = scene_path.with_extension("occlusion");
        self.tasks.spawn("Bake occlusion", TaskKind::OcclusionBake, false, move |cancel, progress| {
            let start = std::time::Instant::now();
            let data = zestengine_engine::occlusion::bake(&input, &cancel, |done, total| {
                *progress.lock().unwrap() = TaskProgress::Fraction {
                    done: done as f32 / total.max(1) as f32,
                    label: format!("{done}/{total} cells"),
                };
            })
            .ok_or_else(|| "cancelled".to_owned())?;
            std::fs::write(&sidecar, data.to_bytes()).map_err(|e| format!("writing {}: {e}", sidecar.display()))?;
            Ok(TaskPayload::Occlusion { scene_path, data: Box::new(data), secs: start.elapsed().as_secs_f32() })
        });
    }

    pub(super) fn apply_occlusion_bake(&mut self, scene_path: PathBuf, data: OcclusionData, secs: f32) {
        let msg = format!(
            "Occlusion baked: {} cells, {} objects, {} KB in {secs:.1} s",
            data.cell_count(),
            data.objects.len(),
            data.byte_size().div_ceil(1024)
        );
        if self.current_scene_path.as_ref() == Some(&scene_path) {
            self.scene.occlusion = Some(data);
        }
        self.tasks.notify(msg, NotifyLevel::Info);
    }

    pub(super) fn clear_occlusion(&mut self) {
        self.scene.occlusion = None;
        if let Some(p) = &self.current_scene_path {
            let sidecar = p.with_extension("occlusion");
            if sidecar.exists()
                && let Err(e) = std::fs::remove_file(&sidecar)
            {
                tracing::error!("removing {}: {e}", sidecar.display());
            }
        }
        self.tasks.notify("Cleared occlusion data", NotifyLevel::Info);
    }
}
