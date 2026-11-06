//! The model inspector's "Geometry (Zestlets)" section: whether the stored
//! cluster DAGs (`.zestengine_geocache/<file>.zgeo`) exist and are usable, what
//! each mesh got, and a Rebuild button that runs the builder on a task worker
//! with a progress bar and a Cancel.
//!
//! The section only reads the cache when it is open. The rebuild is a
//! [`GeoJob`] shared between the section (progress, cancel, result) and the
//! worker the editor spawns on the task system ([`GeoJob::run`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

use zestengine_assets::cluster_dag::{DagOptions, GEO_MIN_TRIANGLES};
use zestengine_assets::geo_cache::{self, CacheState, GeoStatus, MeshGeoInfo};
use crate::{theme, widgets};

use crate::tasks::{TaskPayload, TaskProgress};

/// One rebuild: the section reads it, the worker writes it.
#[derive(Default)]
pub struct GeoJob {
    /// Progress in thousandths.
    progress: AtomicU32,
    cancel: AtomicBool,
    done: Mutex<Option<Result<Vec<f32>, String>>>,
}

impl GeoJob {
    pub fn fraction(&self) -> f32 {
        self.progress.load(Ordering::Relaxed) as f32 / 1000.0
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Worker body: rebuild the geometry of `path` and store it. `cancel` and
    /// `progress` are the task system's, so its popup shows the same bar.
    pub fn run(&self, path: &Path, cancel: &AtomicBool, progress: &Mutex<TaskProgress>) -> Result<TaskPayload, String> {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let res = (|| {
            let (scene, _) = zestengine_assets::load_model_cached(path, false).map_err(|e| format!("{e:#}"))?;
            let report = |f: f32| {
                self.progress.store((f.clamp(0.0, 1.0) * 1000.0) as u32, Ordering::Relaxed);
                if let Ok(mut p) = progress.lock() {
                    *p = TaskProgress::Fraction { done: f, label: "zestlets".into() };
                }
            };
            let stop = || self.cancel.load(Ordering::Relaxed) || cancel.load(Ordering::Relaxed);
            let r = geo_cache::rebuild(path, &scene, &DagOptions::default(), &report, &stop).ok_or("cancelled")?;
            Ok::<_, String>((r.geo.clusters(), r.mesh_secs))
        })();
        let out = match &res {
            Ok((clusters, _)) => Ok(TaskPayload::Notice(format!("Rebuilt Zestlet data for {name}: {clusters} zestlets"))),
            Err(e) if e == "cancelled" => Ok(TaskPayload::None),
            Err(e) => Err(e.clone()),
        };
        *self.done.lock().unwrap() = Some(res.map(|(_, secs)| secs));
        out
    }
}

/// Editor state of the section.
#[derive(Default)]
pub struct GeoCacheUi {
    status: Option<(PathBuf, GeoStatus)>,
    rx: Option<(PathBuf, Receiver<Result<GeoStatus, String>>)>,
    job: Option<(PathBuf, Arc<GeoJob>)>,
    /// Seconds per mesh slot of the last rebuild this session, by model.
    times: HashMap<PathBuf, Vec<f32>>,
    last_error: Option<String>,
    /// Cluster rendering is switched On (project setting; set by the editor).
    pub cluster_on: bool,
}

impl GeoCacheUi {
    /// Read the cache report again next frame (after a reimport or rebuild).
    pub fn forget(&mut self) {
        self.status = None;
    }

    fn poll(&mut self) {
        if let Some((p, rx)) = &self.rx {
            match rx.try_recv() {
                Ok(r) => {
                    match r {
                        Ok(s) => self.status = Some((p.clone(), s)),
                        Err(e) => self.last_error = Some(e),
                    }
                    self.rx = None;
                }
                Err(TryRecvError::Disconnected) => self.rx = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some((p, job)) = &self.job {
            let finished = job.done.lock().unwrap().take();
            if let Some(res) = finished {
                match res {
                    Ok(secs) => {
                        self.times.insert(p.clone(), secs);
                        self.last_error = None;
                    }
                    Err(e) if e == "cancelled" => {}
                    Err(e) => self.last_error = Some(e),
                }
                self.job = None;
                self.status = None;
            }
        }
    }

    fn ensure_status(&mut self, path: &Path) {
        let have = self.status.as_ref().is_some_and(|(p, _)| p == path);
        let pending = self.rx.as_ref().is_some_and(|(p, _)| p == path);
        if have || pending || self.job.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let owned = path.to_path_buf();
        std::thread::spawn(move || {
            let res = zestengine_assets::load_model_cached(&owned, false).map(|(s, _)| geo_cache::inspect(&owned, &s)).map_err(|e| format!("{e:#}"));
            let _ = tx.send(res);
        });
        self.rx = Some((path.to_path_buf(), rx));
    }

    /// The collapsed section. Returns a new job when Rebuild was pressed; the
    /// editor spawns it ([`GeoJob::run`]).
    pub fn section(&mut self, ui: &mut egui::Ui, path: &Path) -> Option<Arc<GeoJob>> {
        let (open, _) = widgets::section_header_with(ui, ui.make_persistent_id("model-geo-clusters"), None, "Geometry (Zestlets)", None, false, |_| {});
        if !open {
            return None;
        }
        self.poll();
        self.ensure_status(path);
        let mut started = None;
        let running = self.job.as_ref().filter(|(p, _)| p == path).map(|(_, j)| j.clone());
        if self.rx.is_some() || running.is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        let status = self.status.as_ref().filter(|(p, _)| p == path).map(|(_, s)| s);
        match status {
            Some(s) => {
                widgets::property_row(ui, "Cache File", |ui| {
                    ui.label(egui::RichText::new(state_label(&s.state, s.file_bytes)).size(theme::SIZE_META).color(state_color(&s.state)));
                });
                if let CacheState::Invalid(why) = &s.state {
                    widgets::info_note(ui, &format!("The file is not used and is rebuilt when needed: {why}"));
                }
                if s.state == CacheState::Stale {
                    widgets::info_note(ui, "The file was built for other geometry or settings (the model changed). It is rebuilt when the model is next used, or press Rebuild.");
                }
                let secs = self.times.get(path);
                for (i, m) in s.meshes.iter().enumerate() {
                    ui.label(egui::RichText::new(mesh_line(i, m, secs.and_then(|t| t.get(i)).copied(), s.state == CacheState::Valid)).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                    if let Some(note) = mesh_note(m, s.state == CacheState::Valid) {
                        ui.label(egui::RichText::new(format!("    {note}")).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                    }
                }
                if s.meshes.is_empty() {
                    ui.label(egui::RichText::new("This model has no meshes.").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                }
            }
            None if self.last_error.is_some() => {}
            None => {
                ui.label(egui::RichText::new("Reading the cache\u{2026}").size(theme::SIZE_META).color(theme::TEXT_MUTED));
            }
        }
        if let Some(e) = &self.last_error {
            ui.label(egui::RichText::new(e).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        ui.label(egui::RichText::new(cluster_note(self.cluster_on)).size(theme::SIZE_META).color(theme::TEXT_MUTED));

        ui.add_space(4.0);
        if let Some(job) = &running {
            ui.horizontal(|ui| {
                ui.add(egui::ProgressBar::new(job.fraction()).desired_width(160.0).show_percentage());
                if widgets::secondary_button(ui, None, "Cancel").on_hover_text("Stop after the mesh being built; the file is left as it was").clicked() {
                    job.cancel();
                }
            });
        } else {
            let any = status.is_some_and(|s| s.meshes.iter().any(|m| m.wants_dag));
            ui.add_enabled_ui(status.is_some() && any, |ui| {
                if widgets::secondary_button(ui, None, "Rebuild")
                    .on_hover_text("Build the Zestlet data again from the model and replace the stored file")
                    .on_disabled_hover_text("No mesh is big enough for Zestlet data, or the cache is still being read")
                    .clicked()
                {
                    let job = Arc::new(GeoJob::default());
                    self.job = Some((path.to_path_buf(), job.clone()));
                    self.last_error = None;
                    started = Some(job);
                }
            });
        }
        started
    }
}

fn state_label(state: &CacheState, bytes: u64) -> String {
    match state {
        CacheState::Missing => "not built".into(),
        CacheState::Valid => format!("valid \u{b7} {}", format_bytes(bytes)),
        CacheState::Stale => format!("out of date \u{b7} {}", format_bytes(bytes)),
        CacheState::Invalid(_) => format!("damaged \u{b7} {}", format_bytes(bytes)),
    }
}

fn state_color(state: &CacheState) -> egui::Color32 {
    match state {
        CacheState::Valid => theme::TEXT_MUTED,
        CacheState::Missing => theme::TEXT_MUTED,
        _ => theme::STATUS_WARN_FG,
    }
}

fn format_bytes(b: u64) -> String {
    match b {
        0..=1023 => format!("{b} B"),
        1024..=1_048_575 => format!("{:.1} KB", b as f64 / 1024.0),
        _ => format!("{:.1} MB", b as f64 / 1_048_576.0),
    }
}

/// `Mesh 2 \u{b7} 12,000 tris \u{b7} 96 zestlets \u{b7} 1.2 MB \u{b7} 5 levels \u{b7} built in 0.8 s`.
fn mesh_line(i: usize, m: &MeshGeoInfo, secs: Option<f32>, valid: bool) -> String {
    let mut s = format!("Mesh {i} \u{b7} {} tris", m.triangles);
    if valid && m.clusters > 0 {
        s += &format!(" \u{b7} {} zestlets \u{b7} {} \u{b7} {} levels", m.clusters, format_bytes(m.dag_bytes as u64), m.levels);
        if let Some(t) = secs.filter(|t| *t > 0.0) {
            s += &format!(" \u{b7} built in {t:.1} s");
        }
    }
    s
}

/// Why a mesh has (or may not draw from) no cluster data.
fn mesh_note(m: &MeshGeoInfo, valid: bool) -> Option<String> {
    if !m.wants_dag {
        return Some(format!("Below the {GEO_MIN_TRIANGLES} triangle threshold: no Zestlet data, drawn normally."));
    }
    if valid && m.clusters == 0 {
        return Some("Refused: the builder could not make Zestlet data for this mesh (see the Log). Drawn normally.".into());
    }
    if m.lightmapped {
        return Some("Has a lightmap UV set: draws as Zestlets, except objects with a baked lightmap, which draw normally.".into());
    }
    if valid && m.skinned {
        return Some("Skinned: drawn as Zestlets only on animated instances; drawn normally if the skin pool is full.".into());
    }
    if valid && m.blendshapes > 0 {
        return Some(format!("{} blendshape(s) carried; face detail is kept longer when simplifying.", m.blendshapes));
    }
    None
}

fn cluster_note(on: bool) -> String {
    if on {
        "Zestlets are On (Project Settings): meshes with Zestlet data draw as GPU-culled zestlets.".into()
    } else {
        "Zestlets are Off (Project Settings): the data is kept but not drawn from.".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(tris: usize) -> MeshGeoInfo {
        MeshGeoInfo { triangles: tris, wants_dag: tris >= GEO_MIN_TRIANGLES, lightmapped: false, clusters: 0, dag_bytes: 0, levels: 0, skinned: false, blendshapes: 0 }
    }

    #[test]
    fn notes_explain_missing_cluster_data() {
        assert!(mesh_note(&info(100), true).unwrap().contains("threshold"));
        assert!(mesh_note(&info(5000), true).unwrap().contains("Refused"));
        assert!(mesh_note(&info(5000), false).is_none(), "not built yet is not a refusal");
        let mut m = info(5000);
        m.clusters = 40;
        assert!(mesh_note(&m, true).is_none());
        m.skinned = true;
        assert!(mesh_note(&m, true).unwrap().contains("Skinned"));
        m.skinned = false;
        m.blendshapes = 3;
        assert!(mesh_note(&m, true).unwrap().contains("3 blendshape"));
        m.lightmapped = true;
        assert!(mesh_note(&m, true).unwrap().contains("lightmap UV set"));
    }

    #[test]
    fn lines_and_labels() {
        let mut m = info(12_000);
        m.clusters = 96;
        m.dag_bytes = 1_258_291;
        m.levels = 5;
        assert_eq!(mesh_line(2, &m, Some(0.8), true), "Mesh 2 \u{b7} 12000 tris \u{b7} 96 zestlets \u{b7} 1.2 MB \u{b7} 5 levels \u{b7} built in 0.8 s");
        assert_eq!(mesh_line(0, &info(10), None, true), "Mesh 0 \u{b7} 10 tris");
        assert_eq!(mesh_line(2, &m, None, false), "Mesh 2 \u{b7} 12000 tris");
        assert_eq!(state_label(&CacheState::Missing, 0), "not built");
        assert_eq!(state_label(&CacheState::Valid, 2048), "valid \u{b7} 2.0 KB");
        assert!(cluster_note(true).contains("On") && cluster_note(false).contains("Off"));
    }

    #[test]
    fn job_runs_reports_and_cancels() {
        let dir = std::env::temp_dir().join(format!("zest_geoui_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // A missing model fails with a message and leaves the job's result readable.
        let job = GeoJob::default();
        let progress = Mutex::new(TaskProgress::Indeterminate);
        let r = job.run(&dir.join("none.glb"), &AtomicBool::new(false), &progress);
        assert!(r.is_err());
        assert!(matches!(job.done.lock().unwrap().take(), Some(Err(_))));
        job.cancel();
        assert!(job.cancel.load(Ordering::Relaxed));
        assert_eq!(job.fraction(), 0.0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
