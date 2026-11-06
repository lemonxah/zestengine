//! Editor entry points for the Unity material converter (Files panel context
//! menu: Convert Unity Material, Convert Unity folder, Import Unity package).
//! The work runs on a task thread (`TaskKind::UnityConvert`) with a progress bar
//! and ends in one summary notification; per-material details are written next
//! to each result as `<name>.convert.txt`. Logic lives in
//! `zestengine_assets::unity_convert`.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use zestengine_assets::unity_convert::{ConvertOptions, ConvertSummary, convert_unity_folder, convert_unity_material};

use crate::tasks::{TaskKind, TaskManager, TaskPayload, TaskProgress};

/// What the user asked to convert.
#[derive(Clone, Debug)]
pub enum UnityJob {
    /// One `.mat` file.
    Material(PathBuf),
    /// A folder of Unity assets (`.mat` + `.meta` + textures).
    Folder(PathBuf),
    /// A `.unitypackage`: extracted to a temp folder, then converted as a folder.
    Package(PathBuf),
}

fn name_of(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Start a conversion task. Results land in `<project>/materials/unity`, textures in
/// `<project>/textures/unity`.
pub fn spawn(tasks: &mut TaskManager, job: UnityJob, project_root: &Path) {
    let root = project_root.to_owned();
    let label = match &job {
        UnityJob::Material(p) => format!("Convert Unity material {}", name_of(p)),
        UnityJob::Folder(p) => format!("Convert Unity folder {}", name_of(p)),
        UnityJob::Package(p) => format!("Import Unity package {}", name_of(p)),
    };
    tasks.spawn(label, TaskKind::UnityConvert, false, move |cancel, progress| {
        let opts = ConvertOptions::default();
        let out_dir = root.join("materials").join("unity");
        let mut report = |done: usize, total: usize, current: &str| -> bool {
            if let Ok(mut p) = progress.lock() {
                *p = TaskProgress::Fraction { done: done as f32 / total.max(1) as f32, label: current.to_owned() };
            }
            !cancel.load(Ordering::Relaxed)
        };
        let summary: ConvertSummary = match &job {
            UnityJob::Material(mat) => convert_unity_material(mat, None, &root, &out_dir, &opts),
            UnityJob::Folder(dir) => convert_unity_folder(dir, &root, &out_dir, &opts, &mut report),
            UnityJob::Package(pkg) => {
                let tmp = std::env::temp_dir().join(format!("zest_unitypkg_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
                let extracted = zestengine_assets::unity_package::extract_unitypackage(pkg, &tmp).map_err(|e| format!("{e:#}"));
                let result = extracted.map(|x| {
                    let s = convert_unity_folder(&tmp, &root, &out_dir, &opts, &mut report);
                    (x, s)
                });
                let _ = std::fs::remove_dir_all(&tmp);
                let (x, s) = result?;
                return Ok(TaskPayload::Notice(format!("{} ({} assets in the package)", s.message(), x.assets)));
            }
        };
        Ok(TaskPayload::Notice(summary.message()))
    });
}
