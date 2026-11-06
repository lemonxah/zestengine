//! Background-task system for the editor.
//!
//! Heavy CPU work (file parsing, FS scans) runs on worker threads and reports
//! results back over per-task channels; the main thread polls each frame and
//! applies the GPU side (see `EngineApp::apply_task_result`). GPU work that
//! can't move off the main thread (light baking) is driven separately as a
//! per-frame stepped job but still registers a task here for the shared UI.
//!
//! winit/egui/the renderer are pinned to the main thread, and the scene holds
//! non-`Send` `dyn Component`s — so worker closures may only capture `Send`
//! data (paths, asset-side `Scene`/`SceneFile`/`TextureData`), never `self`,
//! the renderer, or the `LoadedScene`. The compiler enforces this at `spawn`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

pub type TaskId = u64;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    ImportModel,
    LoadScene,
    Bake,
    FocusReimport,
    /// Steam Audio reverb / pathing bake.
    AudioBake,
    /// Occlusion-culling visibility bake.
    OcclusionBake,
    /// `cargo build` + asset bundling of the game executable.
    BuildGame,
    /// Unity material / folder / package conversion.
    UnityConvert,
    /// Megascans / Fab asset pack import.
    PackImport,
    /// `cargo build` of the project's component plugins.
    BuildComponents,
    /// Cluster DAG build of a model (GEO-3), driven by the scene's worker.
    BuildClusters,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BakePhase {
    Lightmaps,
    Probes,
    FluxVoxel,
}

/// Live progress a worker (or the bake stepper) updates and the UI reads.
#[derive(Clone)]
pub enum TaskProgress {
    /// No countable units (parse/scan) — UI shows an indeterminate spinner.
    Indeterminate,
    /// Light-bake detail (counts known up front from the gather).
    Bake {
        lights: u32,
        bounces: u32,
        lightmap_done: u32,
        lightmap_total: u32,
        probe_volumes: u32,
        flux_voxel_volumes: u32,
        phase: BakePhase,
    },
    /// A plain 0..1 fraction with a phase label.
    Fraction { done: f32, label: String },
}

impl TaskProgress {
    /// 0..1 completion, or None when indeterminate.
    pub fn fraction(&self) -> Option<f32> {
        match self {
            TaskProgress::Indeterminate => None,
            TaskProgress::Bake {
                lightmap_done,
                lightmap_total,
                ..
            } => Some(*lightmap_done as f32 / (*lightmap_total).max(1) as f32),
            TaskProgress::Fraction { done, .. } => Some(done.clamp(0.0, 1.0)),
        }
    }

    /// Short human-readable detail line for the task popup.
    pub fn detail(&self) -> String {
        match self {
            TaskProgress::Indeterminate => "working…".into(),
            TaskProgress::Fraction { done, label } => format!("{label} {:.0}%", done.clamp(0.0, 1.0) * 100.0),
            TaskProgress::Bake {
                lights,
                bounces,
                lightmap_done,
                lightmap_total,
                probe_volumes,
                flux_voxel_volumes,
                phase,
            } => {
                let p = match phase {
                    BakePhase::Lightmaps => "lightmaps",
                    BakePhase::Probes => "probes",
                    BakePhase::FluxVoxel => "FluxVoxel",
                };
                format!(
                    "{p}: lightmap {lightmap_done}/{lightmap_total} · {lights} lights · \
                     {bounces} bounces · {probe_volumes} probe vols · {flux_voxel_volumes} FluxVoxel"
                )
            }
        }
    }
}

/// A finished worker's payload for the main thread to apply (GPU upload etc.).
pub enum TaskPayload {
    /// Nothing to apply (no change found, or a smoke-test task).
    None,
    /// Finished with nothing to apply but a line to tell the user.
    Notice(String),
    /// Parsed model ready for `add_asset_scene` on the main thread. `source` is
    /// the project-relative path. Textures referenced by the model ride along in
    /// the `Scene` and are uploaded by `add_asset_scene`.
    Model {
        source: PathBuf,
        scene: Box<zestengine_assets::Scene>,
    },
    /// A model rebuilt by a per-asset reimport, to be swapped into the live
    /// scene in place (`LoadedScene::reimport_model`). `source` is the
    /// project-relative path.
    ModelReimport {
        source: PathBuf,
        scene: Box<zestengine_assets::Scene>,
    },
    /// A scene file + its models, parsed off-thread. `prepared` holds GPU
    /// resources already uploaded on the loader thread (when a transfer queue is
    /// available), to be installed on the main thread; None means upload on main.
    Scene(Box<zestengine_engine::scene_loader::LoadedSceneData>),
    /// A focus-scan result: the changed assets to re-import. Models that are in
    /// the open scene reimport in place; anything else needs the scene reloaded.
    FocusChanges {
        scene_path: Option<PathBuf>,
        label: String,
        /// Absolute paths of the changed model files.
        models: Vec<PathBuf>,
        /// Absolute paths of the changed `.prefab` files (re-merged into the
        /// scene's instances in place).
        prefabs: Vec<PathBuf>,
        /// Something other than a model or prefab changed (a texture, shader, scene, ...).
        other: bool,
    },
    /// An extract or Search and Remap finished for a model's import session: the
    /// tables it filled, to merge into the inspector's edited settings.
    ModelRemap {
        path: PathBuf,
        material_remap: std::collections::BTreeMap<String, String>,
        texture_remap: std::collections::BTreeMap<String, String>,
        note: String,
        /// Search and Remap only: `(source material, picked .material, why)`.
        reasons: Vec<(String, String, String)>,
    },
    /// A Steam Audio bake for `scene_path` (already written as its sidecar).
    AudioBake { scene_path: PathBuf, bytes: Vec<u8> },
    /// An occlusion bake for `scene_path` (already written as its sidecar).
    Occlusion { scene_path: PathBuf, data: Box<zestengine_engine::occlusion::OcclusionData>, secs: f32 },
    /// A finished game build: the executable, or the failure text.
    GameBuilt(Result<PathBuf, String>),
    /// Component plugins built (names) or the compiler error; the main thread
    /// loads + registers them.
    PluginsBuilt(Result<Vec<String>, String>, Vec<zestengine_plugins::CodeDiagnostic>),
    /// Bake sidecars encoded + written (already on disk); GPU-ready data to swap in.
    BakeSaved(Box<crate::bake_store::BakeSaved>),
}

pub struct TaskResult {
    pub outcome: Result<TaskPayload, String>,
    pub cancelled: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NotifyLevel {
    Info,
    Warn,
}

pub struct Notification {
    pub text: String,
    pub at: Instant,
    pub level: NotifyLevel,
}

/// A live task: its UI-facing handle plus the channel its worker reports on.
/// `rx` is None for the stepped bake (driven on the main thread, completed via
/// `complete_stepped`).
pub struct TaskHandle {
    pub id: TaskId,
    pub label: String,
    /// Task category (reserved for per-type UI such as icons).
    #[allow(dead_code)]
    pub kind: TaskKind,
    pub blocking: bool,
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<Mutex<TaskProgress>>,
}

struct RunningTask {
    handle: TaskHandle,
    rx: Option<mpsc::Receiver<TaskResult>>,
}

pub struct TaskManager {
    next_id: TaskId,
    running: Vec<RunningTask>,
    pub notifications: Vec<Notification>,
}

/// How long a notification stays in the status bar.
const NOTIFY_TTL_SECS: f32 = 8.0;

impl Default for TaskManager {
    fn default() -> Self {
        Self {
            next_id: 1,
            running: Vec::new(),
            notifications: Vec::new(),
        }
    }
}

impl TaskKind {
    /// Short static name for hitch-capture event markers.
    pub fn name(self) -> &'static str {
        match self {
            TaskKind::ImportModel => "import_model",
            TaskKind::LoadScene => "load_scene",
            TaskKind::Bake => "bake",
            TaskKind::FocusReimport => "focus_reimport",
            TaskKind::AudioBake => "audio_bake",
            TaskKind::OcclusionBake => "occlusion_bake",
            TaskKind::BuildGame => "build_game",
            TaskKind::UnityConvert => "unity_convert",
            TaskKind::PackImport => "pack_import",
            TaskKind::BuildComponents => "build_components",
            TaskKind::BuildClusters => "build_clusters",
        }
    }
}

impl TaskManager {
    /// Spawn a CPU worker task. `f` runs on a std thread with the cancel flag and
    /// a progress writer, returning a payload (or an error string).
    pub fn spawn<F>(
        &mut self,
        label: impl Into<String>,
        kind: TaskKind,
        blocking: bool,
        f: F,
    ) -> TaskId
    where
        F: FnOnce(Arc<AtomicBool>, Arc<Mutex<TaskProgress>>) -> Result<TaskPayload, String>
            + Send
            + 'static,
    {
        let id = self.next_id;
        self.next_id += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(Mutex::new(TaskProgress::Indeterminate));
        let (tx, rx) = mpsc::channel();
        let cancel_run = cancel.clone();
        let cancel_check = cancel.clone();
        let progress_run = progress.clone();
        let kind_name = kind.name();
        zestengine_render::hitch::event("task_start", kind_name, id);
        std::thread::spawn(move || {
            let outcome = f(cancel_run, progress_run);
            zestengine_render::hitch::event("task_done", kind_name, id);
            let _ = tx.send(TaskResult {
                outcome,
                cancelled: cancel_check.load(Ordering::Relaxed),
            });
        });
        self.running.push(RunningTask {
            handle: TaskHandle {
                id,
                label: label.into(),
                kind,
                blocking,
                cancel,
                progress,
            },
            rx: Some(rx),
        });
        id
    }

    /// Register a main-thread stepped task (the bake). Returns its id; progress
    /// is updated via `progress_of`, completion via `complete_stepped`.
    pub fn register_stepped(
        &mut self,
        label: impl Into<String>,
        kind: TaskKind,
        progress: TaskProgress,
    ) -> (TaskId, Arc<AtomicBool>, Arc<Mutex<TaskProgress>>) {
        let id = self.next_id;
        self.next_id += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(Mutex::new(progress));
        self.running.push(RunningTask {
            handle: TaskHandle {
                id,
                label: label.into(),
                kind,
                blocking: false,
                cancel: cancel.clone(),
                progress: progress.clone(),
            },
            rx: None,
        });
        (id, cancel, progress)
    }

    /// Remove a stepped task by id (on completion/cancel of the bake).
    pub fn complete_stepped(&mut self, id: TaskId) {
        self.running.retain(|t| t.handle.id != id);
    }

    /// Drain finished worker tasks (removing them from `running`), returning
    /// owned results so the caller can apply them against `&mut self` without
    /// holding a borrow on the manager.
    pub fn poll_workers(&mut self) -> Vec<TaskResult> {
        let mut done = Vec::new();
        self.running.retain_mut(|t| {
            let Some(rx) = t.rx.as_ref() else {
                return true; // stepped task; not channel-driven
            };
            match rx.try_recv() {
                Ok(r) => {
                    done.push(r);
                    false
                }
                Err(mpsc::TryRecvError::Empty) => true,
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Worker died without sending — drop the task.
                    done.push(TaskResult {
                        outcome: Err("worker thread died".into()),
                        cancelled: t.handle.cancel.load(Ordering::Relaxed),
                    });
                    false
                }
            }
        });
        done
    }

    pub fn cancel(&mut self, id: TaskId) {
        if let Some(t) = self.running.iter().find(|t| t.handle.id == id) {
            t.handle.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Running tasks the UI lists (excludes blocking tasks, which show a modal).
    pub fn visible(&self) -> impl Iterator<Item = &TaskHandle> {
        self.running
            .iter()
            .map(|t| &t.handle)
            .filter(|h| !h.blocking)
    }

    /// First blocking task (drives the modal overlay), if any.
    pub fn blocking(&self) -> Option<&TaskHandle> {
        self.running.iter().map(|t| &t.handle).find(|h| h.blocking)
    }

    /// (count, overall 0..1) over visible tasks for the status-bar button.
    pub fn aggregate(&self) -> Option<(usize, f32)> {
        let visible: Vec<&TaskHandle> = self.visible().collect();
        if visible.is_empty() {
            return None;
        }
        let mut sum = 0.0;
        let mut n = 0;
        for h in &visible {
            if let Some(f) = h.progress.lock().unwrap().fraction() {
                sum += f;
                n += 1;
            }
        }
        let frac = if n > 0 { sum / n as f32 } else { 0.0 };
        Some((visible.len(), frac))
    }

    pub fn notify(&mut self, text: impl Into<String>, level: NotifyLevel) {
        self.notifications.push(Notification {
            text: text.into(),
            at: Instant::now(),
            level,
        });
    }

    pub fn prune_notifications(&mut self) {
        self.notifications
            .retain(|n| n.at.elapsed().as_secs_f32() < NOTIFY_TTL_SECS);
    }
}

/// The task-list entry of the cluster DAG build in flight. The scene's worker
/// owns the build; each frame this mirrors its [`BuildProgress`] into a stepped
/// task (cheap atomic reads) and forwards the task's Cancel to the build.
#[derive(Default)]
pub struct ClusterBuildTask {
    shown: Option<Shown>,
}

struct Shown {
    id: TaskId,
    build: Arc<zestengine_assets::geo_cache::BuildProgress>,
    cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<TaskProgress>>,
}

impl ClusterBuildTask {
    /// Reconcile with the build in flight (`None` = none): open the task when one
    /// starts, update its numbers, pass a Cancel on, close it when it ends.
    pub fn sync(&mut self, tasks: &mut TaskManager, active: Option<(&str, &Arc<zestengine_assets::geo_cache::BuildProgress>)>) {
        if let Some(shown) = &self.shown
            && active.is_none_or(|(_, build)| !Arc::ptr_eq(build, &shown.build))
        {
            tasks.complete_stepped(shown.id);
            self.shown = None;
        }
        let Some((name, build)) = active else { return };
        let shown = self.shown.get_or_insert_with(|| {
            let (id, cancel, progress) = tasks.register_stepped(format!("Building zestlets for {name}"), TaskKind::BuildClusters, TaskProgress::Indeterminate);
            Shown { id, build: build.clone(), cancel, progress }
        });
        if shown.cancel.load(Ordering::Relaxed) {
            shown.build.cancel();
        }
        let snap = shown.build.snapshot();
        // A cache hit (or the start) has no totals yet: stay indeterminate.
        *shown.progress.lock().unwrap_or_else(|e| e.into_inner()) =
            if snap.tris_total == 0 { TaskProgress::Indeterminate } else { TaskProgress::Fraction { done: snap.fraction(), label: snap.label() } };
    }

    #[cfg(test)]
    fn is_shown(&self) -> bool {
        self.shown.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::geo_cache::BuildProgress;

    #[test]
    fn the_cluster_build_shows_as_a_task_with_progress_and_cancel() {
        let mut tasks = TaskManager::default();
        let mut mirror = ClusterBuildTask::default();
        mirror.sync(&mut tasks, None);
        assert!(tasks.visible().next().is_none() && !mirror.is_shown());

        let build = BuildProgress::new();
        mirror.sync(&mut tasks, Some(("models/a.fbx", &build)));
        let h = tasks.visible().next().expect("the build is listed");
        assert!(h.label.contains("models/a.fbx") && h.kind == TaskKind::BuildClusters && !h.blocking);
        assert!(h.progress.lock().unwrap().fraction().is_none(), "indeterminate before the totals are known");

        build.begin(180, 20_000);
        build.add_tris(5_000);
        build.mesh_done();
        mirror.sync(&mut tasks, Some(("models/a.fbx", &build)));
        let (n, frac) = tasks.aggregate().unwrap();
        assert_eq!(n, 1);
        assert_eq!(frac, 0.25);
        let detail = tasks.visible().next().unwrap().progress.lock().unwrap().detail();
        assert!(detail.contains("1/180 meshes") && detail.contains("25%"), "{detail}");

        // Cancel in the task list reaches the build.
        let id = tasks.visible().next().unwrap().id;
        assert!(!build.is_cancelled());
        tasks.cancel(id);
        mirror.sync(&mut tasks, Some(("models/a.fbx", &build)));
        assert!(build.is_cancelled());

        // The build ends: the task goes away.
        mirror.sync(&mut tasks, None);
        assert!(tasks.visible().next().is_none() && !mirror.is_shown());
    }

    #[test]
    fn a_new_build_replaces_the_finished_ones_entry() {
        let mut tasks = TaskManager::default();
        let mut mirror = ClusterBuildTask::default();
        let (a, b) = (BuildProgress::new(), BuildProgress::new());
        mirror.sync(&mut tasks, Some(("a", &a)));
        mirror.sync(&mut tasks, Some(("b", &b)));
        let labels: Vec<String> = tasks.visible().map(|h| h.label.clone()).collect();
        assert_eq!(labels.len(), 1);
        assert!(labels[0].contains('b'), "{labels:?}");
    }
}
