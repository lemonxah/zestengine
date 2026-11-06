//! Editor side of the Megascans / Fab importer (IMP-8): the preview and report
//! dialog, the background import task, and the optional Downloads watcher.
//!
//! The dialog opens from the Files panel ("Import Megascans / Fab Asset..." in
//! the Create menu, on a folder / `.zip` / `.mhpkg`, or by dropping one on the
//! panel). Scanning (and extracting a zip into a temp folder) runs on a worker
//! thread; the import runs as a task with progress in the task list. All
//! logic lives in `zestengine_assets::{pack_scan, pack_convert, mhpkg}`.
//! Nothing here touches the network: "Browse on Fab" only asks the OS to open
//! https://www.fab.com in the browser, and the Downloads watcher only lists a
//! local folder every few seconds, and only while its checkbox is on.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use egui::{Align, Color32, ComboBox, Layout, RichText, ScrollArea, Ui};
use zestengine_assets::megascans::AssetKind;
use zestengine_assets::mhpkg::{EXPORT_HELP, MhPackage, read_mhpkg};
use zestengine_assets::pack_convert::{DuplicatePolicy, ImportOptions, import_assets};
use zestengine_assets::pack_scan::{ArchiveKind, PackAsset, PackScan, peek_archive, scan_pack};
use zestengine_assets::pack_zip::{ZipLimits, extract_zip};
use zestengine_assets::tex_roles::{Confidence, TexRole};
use zestengine_assets::unity_convert::Level;
use crate::{theme, widgets};

use crate::tasks::{TaskKind, TaskManager, TaskPayload, TaskProgress};

/// Where "Browse on Fab" goes.
pub const FAB_URL: &str = "https://www.fab.com";

/// How often the Downloads watcher lists the folder.
const WATCH_INTERVAL: Duration = Duration::from_secs(3);

/// A temp folder removed when dropped (the extracted zip).
pub struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let n = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        Self(std::env::temp_dir().join(format!("zest_packimport_{}_{n}", std::process::id())))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

enum Scanned {
    Pack(PackScan, Option<TempDir>),
    MetaHuman(Box<MhPackage>),
}

enum ScanMsg {
    Progress(String),
    Done(Result<Scanned, String>),
}

enum Phase {
    /// Nothing picked yet: a path field and the recent downloads.
    Choose,
    Scanning { rx: Receiver<ScanMsg>, label: String, cancel: Arc<AtomicBool> },
    Ready { scan: PackScan, temp: Option<TempDir>, selected: usize },
    MetaHuman(Box<MhPackage>),
    Failed(String),
}

struct Dialog {
    path_text: String,
    phase: Phase,
}

/// Everything the editor keeps for the importer.
#[derive(Default)]
pub struct PackImportUi {
    dialog: Option<Dialog>,
    opts: ImportOptions,
    /// Offer new archives that land in Downloads (off by default).
    pub watch: bool,
    watcher: Watcher,
    offers: Vec<PathBuf>,
}


/// Where downloads usually land.
pub fn downloads_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let d = PathBuf::from(home).join("Downloads");
    d.is_dir().then_some(d)
}

/// Ask the OS to open `target` (a URL or a folder) with its default handler.
pub fn open_with_os(target: &str) {
    #[cfg(target_os = "windows")]
    let r = std::process::Command::new("cmd").args(["/C", "start", "", target]).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(target).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(target).spawn();
    if let Err(e) = r {
        tracing::warn!("could not open {target}: {e}");
    }
}

fn scan_source(source: PathBuf, tx: std::sync::mpsc::Sender<ScanMsg>, cancel: Arc<AtomicBool>) {
    let result = (|| -> Result<Scanned, String> {
        let e = |x: anyhow::Error| format!("{x:#}");
        if source.is_dir() {
            let _ = tx.send(ScanMsg::Progress("Scanning the folder".into()));
            return scan_pack(&source).map(|s| Scanned::Pack(s, None)).map_err(e);
        }
        let ext = source.extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        if ext == "mhpkg" {
            return read_mhpkg(&source).map(|m| Scanned::MetaHuman(Box::new(m))).map_err(e);
        }
        if ext != "zip" {
            return Err(format!("{} is not a folder, .zip or .mhpkg", source.display()));
        }
        match peek_archive(&source) {
            ArchiveKind::MetaHuman => read_mhpkg(&source).map(|m| Scanned::MetaHuman(Box::new(m))).map_err(e),
            _ => {
                let temp = TempDir::new();
                let tx2 = tx.clone();
                let report = extract_zip(&source, &temp.0, &ZipLimits::default(), &mut |done, total, name| {
                    if done % 20 == 0 {
                        let _ = tx2.send(ScanMsg::Progress(format!("Extracting {done}/{total}: {name}")));
                    }
                    !cancel.load(Ordering::Relaxed)
                })
                .map_err(e)?;
                let mut scan = scan_pack(&temp.0).map_err(e)?;
                if !report.unsafe_names.is_empty() {
                    scan.warnings.push(format!("{} archive entries with unsafe paths (../ or absolute) were not extracted", report.unsafe_names.len()));
                }
                for (name, why) in report.skipped.iter().take(5) {
                    scan.warnings.push(format!("{name} skipped: {why}"));
                }
                Ok(Scanned::Pack(scan, Some(temp)))
            }
        }
    })();
    let _ = tx.send(ScanMsg::Done(result));
}

fn human_bytes(b: u64) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < 3 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 { format!("{b} B") } else { format!("{v:.1} {}", U[i]) }
}

fn level_color(l: Level) -> Color32 {
    match l {
        Level::Mapped => theme::STATUS_OK,
        Level::Approx => theme::STATUS_WARN_FG,
        Level::Unsupported | Level::Unmapped => theme::STATUS_ERROR,
        Level::Info => theme::TEXT_MUTED,
    }
}

impl PackImportUi {
    /// Open the dialog; `source` starts scanning at once (a dropped / right-clicked path).
    pub fn open(&mut self, source: Option<PathBuf>) {
        self.dialog = Some(Dialog { path_text: source.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(), phase: Phase::Choose });
        if let Some(p) = source {
            self.offers.retain(|o| *o != p);
            self.start_scan(p);
        }
    }

    fn start_scan(&mut self, source: PathBuf) {
        let Some(d) = self.dialog.as_mut() else { return };
        // A previous scan still running is cancelled by dropping its flag's owner.
        if let Phase::Scanning { cancel, .. } = &d.phase {
            cancel.store(true, Ordering::Relaxed);
        }
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        let label = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        std::thread::spawn(move || scan_source(source, tx, c2));
        d.phase = Phase::Scanning { rx, label: format!("Reading {label}"), cancel };
    }

    /// Draw the dialog, the download offers and run the watcher. Call every frame.
    pub fn show(&mut self, ctx: &egui::Context, tasks: &mut TaskManager, project_root: &Path) {
        self.poll_scan(ctx);
        self.run_watcher(ctx);
        self.offers_window(ctx);
        let mut close = false;
        let mut start: Option<PathBuf> = None;
        let mut import: Option<(Vec<PackAsset>, Option<TempDir>)> = None;
        let Some(dialog) = self.dialog.as_mut() else { return };
        let opts = &mut self.opts;
        let watch = &mut self.watch;
        let mut open = true;
        egui::Window::new("Import Megascans / Fab Asset")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([860.0, 620.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                header(ui, dialog, watch, &mut start);
                ui.separator();
                match &mut dialog.phase {
                    Phase::Choose => choose_ui(ui, &mut start),
                    Phase::Scanning { label, .. } => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(label.as_str());
                        });
                    }
                    Phase::Failed(msg) => {
                        ui.colored_label(theme::STATUS_ERROR, msg.as_str());
                    }
                    Phase::MetaHuman(pkg) => {
                        if metahuman_ui(ui, pkg) {
                            close = true;
                        }
                    }
                    Phase::Ready { scan, temp, selected } => {
                        if ready_ui(ui, scan, selected, opts) {
                            import = Some((scan.assets.clone(), temp.take()));
                        }
                    }
                }
            });
        if !open || close {
            self.close();
        } else if let Some(p) = start {
            self.start_scan(p);
        }
        if let Some((assets, temp)) = import {
            let opts = self.opts.clone();
            spawn_import(tasks, assets, opts, project_root.to_owned(), temp);
            self.close();
        }
    }

    fn close(&mut self) {
        if let Some(Dialog { phase: Phase::Scanning { cancel, .. }, .. }) = &self.dialog {
            cancel.store(true, Ordering::Relaxed);
        }
        self.dialog = None;
    }

    fn poll_scan(&mut self, ctx: &egui::Context) {
        let Some(d) = self.dialog.as_mut() else { return };
        let Phase::Scanning { rx, label, .. } = &mut d.phase else { return };
        ctx.request_repaint_after(Duration::from_millis(100));
        let mut done: Option<Result<Scanned, String>> = None;
        loop {
            match rx.try_recv() {
                Ok(ScanMsg::Progress(p)) => *label = p,
                Ok(ScanMsg::Done(r)) => done = Some(r),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if done.is_none() {
                        done = Some(Err("the scan stopped unexpectedly".into()));
                    }
                    break;
                }
            }
        }
        if let Some(r) = done {
            d.phase = match r {
                Ok(Scanned::Pack(scan, temp)) => Phase::Ready { scan, temp, selected: 0 },
                Ok(Scanned::MetaHuman(m)) => Phase::MetaHuman(m),
                Err(e) => Phase::Failed(e),
            };
        }
    }

    fn run_watcher(&mut self, ctx: &egui::Context) {
        if !self.watch {
            self.watcher.reset();
            return;
        }
        ctx.request_repaint_after(WATCH_INTERVAL);
        for p in self.watcher.poll(Instant::now()) {
            if !self.offers.contains(&p) {
                self.offers.push(p);
            }
        }
    }

    fn offers_window(&mut self, ctx: &egui::Context) {
        if self.offers.is_empty() {
            return;
        }
        let mut open_it: Option<PathBuf> = None;
        let mut dismiss: Option<PathBuf> = None;
        egui::Window::new("New download")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::RIGHT_BOTTOM, [-16.0, -40.0])
            .show(ctx, |ui| {
                for p in &self.offers {
                    ui.horizontal(|ui| {
                        ui.label(p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if widgets::primary_button(ui, None, "Import...").clicked() {
                                open_it = Some(p.clone());
                            }
                            if widgets::secondary_button(ui, None, "Dismiss").clicked() {
                                dismiss = Some(p.clone());
                            }
                        });
                    });
                }
            });
        if let Some(p) = dismiss {
            self.offers.retain(|o| *o != p);
        }
        if let Some(p) = open_it {
            self.open(Some(p));
        }
    }
}

fn header(ui: &mut Ui, dialog: &mut Dialog, watch: &mut bool, start: &mut Option<PathBuf>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Source").color(theme::TEXT_MUTED));
        let edit = ui.add(egui::TextEdit::singleline(&mut dialog.path_text).desired_width(ui.available_width() - 260.0).hint_text("folder, .zip or .mhpkg"));
        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if (widgets::secondary_button(ui, None, "Scan").clicked() || enter) && !dialog.path_text.trim().is_empty() {
            *start = Some(PathBuf::from(dialog.path_text.trim()));
        }
        if widgets::secondary_button(ui, None, "Browse on Fab").on_hover_text("Open https://www.fab.com in your browser. Download a pack there, then drop it here.").clicked() {
            open_with_os(FAB_URL);
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(watch, "Watch the Downloads folder")
            .on_hover_text("Every few seconds, look in ~/Downloads for a new .zip / .mhpkg / folder and offer to import it. Local only, no network. Off by default.");
        if *watch && downloads_dir().is_none() {
            ui.colored_label(theme::STATUS_WARN_FG, "no Downloads folder found");
        }
    });
}

fn choose_ui(ui: &mut Ui, start: &mut Option<PathBuf>) {
    widgets::info_note(ui, "Drop a folder or .zip from Fab, Quixel Megascans, Sketchfab or any other source onto the Files panel, or paste its path above. Megascans export folders, Fab downloads, and MetaHuman .mhpkg packages are recognised. Your files are only copied into the project, nothing is uploaded.");
    ui.add_space(6.0);
    widgets::eyebrow(ui, "Recent downloads");
    let Some(dir) = downloads_dir() else {
        ui.label(RichText::new("No Downloads folder found.").color(theme::TEXT_DIM));
        return;
    };
    let mut items: Vec<(SystemTime, PathBuf)> = Vec::new();
    for base in [dir.clone(), dir.join("fab")] {
        let Ok(rd) = std::fs::read_dir(&base) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let ext = p.extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let hidden = e.file_name().to_string_lossy().starts_with('.');
            if hidden || !(p.is_dir() || ext == "zip" || ext == "mhpkg") {
                continue;
            }
            let m = e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            items.push((m, p));
        }
    }
    items.sort_by_key(|x| std::cmp::Reverse(x.0));
    ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
        for (_, p) in items.iter().take(30) {
            let name = p.strip_prefix(&dir).unwrap_or(p).to_string_lossy().into_owned();
            let icon = if p.is_dir() { "folder" } else { "archive" };
            if ui.add(egui::Button::new(format!("{name}   ({icon})")).frame(false)).clicked() {
                *start = Some(p.clone());
            }
        }
        if items.is_empty() {
            ui.label(RichText::new("Nothing found in Downloads.").color(theme::TEXT_DIM));
        }
    });
}

/// Returns true when the user asked to close the dialog.
fn metahuman_ui(ui: &mut Ui, pkg: &MhPackage) -> bool {
    let mut close = false;
    ui.horizontal(|ui| {
        widgets::status_pill(ui, "MetaHuman package", theme::STATUS_INFO);
        ui.label(RichText::new(&pkg.name).strong());
        widgets::badge(ui, &pkg.asset_type);
        if !pkg.engine_version.is_empty() {
            widgets::badge(ui, &format!("Unreal {}", pkg.engine_version));
        }
    });
    ui.add_space(4.0);
    widgets::info_note(ui, "This package is recognised but cannot be imported directly: it holds Unreal Engine .uasset files, which ZestEngine does not read.");
    egui::Grid::new("mhpkg-grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        let row = |ui: &mut Ui, k: &str, v: String| {
            ui.label(RichText::new(k).color(theme::TEXT_MUTED));
            ui.label(v);
            ui.end_row();
        };
        row(ui, "Object path", pkg.object_path.clone());
        row(ui, "Size", human_bytes(pkg.total_size));
        row(ui, "Dependent packages", pkg.dependent_packages.len().to_string());
        if !pkg.platforms.is_empty() {
            row(ui, "Platforms", pkg.platforms.join(", "));
        }
        row(ui, "Editable character", if pkg.editable_character { "yes".into() } else { "no".into() });
        if pkg.verification_warnings + pkg.verification_errors > 0 {
            row(ui, "Verification", format!("{} warning(s), {} error(s)", pkg.verification_warnings, pkg.verification_errors));
        }
    });
    widgets::eyebrow(ui, "Contains (from the manifest counters)");
    for l in pkg.contents() {
        ui.label(format!("  {l}"));
    }
    ui.collapsing(format!("Files ({})", pkg.files.len()), |ui| {
        for f in pkg.files.iter().take(200) {
            ui.label(RichText::new(format!("{}   {}", f.path, human_bytes(f.bytes))).monospace().size(11.0));
        }
    });
    ui.collapsing(format!("Dependent packages ({})", pkg.dependent_packages.len()), |ui| {
        for d in pkg.dependent_packages.iter().take(200) {
            ui.label(RichText::new(d).monospace().size(11.0));
        }
    });
    widgets::eyebrow(ui, "What to do");
    ui.add(egui::Label::new(RichText::new(EXPORT_HELP).color(theme::TEXT_SECONDARY)).wrap());
    widgets::dialog_footer(ui, |ui| {
        if widgets::secondary_button(ui, None, "Close").clicked() {
            close = true;
        }
        if widgets::secondary_button(ui, None, "Open folder").on_hover_text("Open the folder holding the package").clicked()
            && let Some(dir) = pkg.archive.parent()
        {
            open_with_os(&dir.to_string_lossy());
        }
    });
    close
}

/// Returns true when Import was pressed.
fn ready_ui(ui: &mut Ui, scan: &mut PackScan, selected: &mut usize, opts: &mut ImportOptions) -> bool {
    let mut import = false;
    for w in &scan.warnings {
        ui.colored_label(theme::STATUS_WARN_FG, w.as_str());
    }
    if scan.assets.is_empty() {
        ui.label("No model or texture the engine can read was found in this source.");
        return false;
    }
    *selected = (*selected).min(scan.assets.len() - 1);
    // Room for everything under the list (options + footer) is measured on the previous
    // frame, so the Import footer stays on screen whatever the screen height.
    let below_id = egui::Id::new("pack-import-below-list");
    let below = ui.ctx().data_mut(|d| d.get_temp::<f32>(below_id)).unwrap_or(330.0);
    let height = (ui.available_height() - below).max(160.0);
    ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), height), Layout::left_to_right(Align::Min), |ui| {
        ui.allocate_ui_with_layout(egui::vec2(260.0, height), Layout::top_down(Align::Min), |ui| {
            ui.set_width(260.0);
            ScrollArea::vertical().id_salt("pack-assets").max_height(height).show(ui, |ui| {
                for (i, a) in scan.assets.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut a.include, "");
                        let label = format!("{}  ({})", a.name, a.kind.label());
                        if ui.selectable_label(*selected == i, label).clicked() {
                            *selected = i;
                        }
                    });
                }
            });
        });
        ui.separator();
        ui.vertical(|ui| {
            ScrollArea::vertical().id_salt("pack-detail").max_height(height).show(ui, |ui| {
                asset_detail(ui, &mut scan.assets[*selected]);
            });
        });
    });
    ui.separator();
    let below_top = ui.cursor().top();
    options_ui(ui, opts);
    let n = scan.assets.iter().filter(|a| a.include).count();
    ui.label(RichText::new("Import runs in the background; progress is in the task list").color(theme::TEXT_DIM));
    widgets::dialog_footer(ui, |ui| {
        let label = format!("Import {n} asset(s)");
        ui.add_enabled_ui(n > 0, |ui| {
            if widgets::primary_button(ui, None, &label).clicked() {
                import = true;
            }
        });
    });
    let used = ui.min_rect().bottom() - below_top + 24.0;
    ui.ctx().data_mut(|d| d.insert_temp(below_id, used));
    import
}

fn options_ui(ui: &mut Ui, opts: &mut ImportOptions) {
    egui::Grid::new("pack-opts").num_columns(4).spacing([10.0, 4.0]).show(ui, |ui| {
        ui.label("Target folder");
        ui.add(egui::TextEdit::singleline(&mut opts.target_dir).desired_width(180.0)).on_hover_text("Project-relative. Each asset gets its own sub-folder with models, textures, materials and prefabs.");
        ui.label("Max texture size");
        ComboBox::from_id_salt("pack-maxtex").selected_text(if opts.max_texture_size == 0 { "No limit".to_owned() } else { opts.max_texture_size.to_string() }).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for v in [1024u32, 2048, 4096, 8192, 0] {
                ui.selectable_value(&mut opts.max_texture_size, v, if v == 0 { "No limit".to_owned() } else { v.to_string() });
            }
        });
        ui.end_row();
        ui.label("If it already exists");
        ComboBox::from_id_salt("pack-dups").selected_text(opts.duplicates.label()).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for p in DuplicatePolicy::ALL {
                ui.selectable_value(&mut opts.duplicates, p, p.label());
            }
        });
        ui.label("Parallax scale");
        ui.add(egui::DragValue::new(&mut opts.displacement_scale).speed(0.002).range(0.0..=0.2));
        ui.end_row();
        ui.label("Prefabs");
        ui.checkbox(&mut opts.write_prefabs, "write a prefab per mesh (model + materials + LOD group)");
        ui.end_row();
    });
}

fn asset_detail(ui: &mut Ui, a: &mut PackAsset) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(&a.name).strong().size(15.0));
        if a.megascans {
            widgets::badge(ui, "Megascans");
        }
        ComboBox::from_id_salt(("pack-kind", &a.id)).selected_text(a.kind.label()).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for k in AssetKind::ALL {
                ui.selectable_value(&mut a.kind, k, k.label());
            }
        });
    });
    ui.label(RichText::new(format!("{}  models {}  textures {}  ({} texture sets)", a.dir.display(), human_bytes(a.model_bytes()), human_bytes(a.texture_bytes()), a.sets.len())).color(theme::TEXT_DIM).size(11.0));
    for w in &a.warnings {
        ui.colored_label(theme::STATUS_WARN_FG, w.as_str());
    }
    if matches!(a.kind, AssetKind::Surface | AssetKind::Decal | AssetKind::Imperfection | AssetKind::Atlas) {
        ui.horizontal(|ui| {
            let mut known = a.size_m.is_some();
            if ui.checkbox(&mut known, "Real-world size (m)").on_hover_text("One texture repeat covers this area; the material is tiled so 1 UV unit = 1 metre.").changed() {
                a.size_m = known.then_some([1.0, 1.0]);
            }
            if let Some(s) = a.size_m.as_mut() {
                ui.add(egui::DragValue::new(&mut s[0]).speed(0.05).range(0.01..=500.0));
                ui.label("x");
                ui.add(egui::DragValue::new(&mut s[1]).speed(0.05).range(0.01..=500.0));
            }
        });
    }
    if !a.variants.is_empty() {
        widgets::eyebrow(ui, "Models");
        for v in &a.variants {
            let lods: Vec<String> = v.lods.iter().map(|l| format!("LOD{}", l.lod)).collect();
            let name = if v.name.is_empty() { a.name.clone() } else { v.name.clone() };
            ui.label(format!("  {name}: {}", lods.join(" > ")));
        }
    }
    widgets::eyebrow(ui, "Textures (roles are editable)");
    let id = a.id.clone();
    for (si, set) in a.sets.iter_mut().enumerate() {
        let title = format!("{}{}   ({} files)", set.name, if set.billboard { "  [billboard, skipped]" } else { "" }, set.textures.len());
        egui::CollapsingHeader::new(title).id_salt((&id, si)).default_open(!set.billboard).show(ui, |ui| {
            egui::Grid::new(("pack-tex", &id, si)).num_columns(4).spacing([10.0, 3.0]).show(ui, |ui| {
                for (ti, t) in set.textures.iter_mut().enumerate() {
                    let name = t.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    ui.label(RichText::new(name).monospace().size(11.0));
                    let before = t.role;
                    ComboBox::from_id_salt(("pack-role", &id, si, ti)).width(230.0).selected_text(t.role.label()).show_ui(ui, |ui| {
                        widgets::menu_frame(ui);
                        for r in TexRole::ALL {
                            ui.selectable_value(&mut t.role, r, r.label());
                        }
                    });
                    if t.role != before {
                        t.confidence = Confidence::Known;
                    }
                    ui.label(RichText::new(t.resolution.map(|r| format!("{r}px")).unwrap_or_default()).color(theme::TEXT_DIM).size(11.0));
                    if t.confidence == Confidence::Weak && t.role != TexRole::Ignore {
                        widgets::warn_badge(ui, "guessed");
                    } else if t.role == TexRole::Ignore {
                        ui.label(RichText::new("not used").color(theme::TEXT_DIM).size(11.0));
                    } else {
                        ui.label("");
                    }
                    ui.end_row();
                }
            });
        });
    }
    // Conversion preview: what the current roles will map to.
    widgets::eyebrow(ui, "Conversion preview");
    let ctx = zestengine_assets::pack_convert::PlanCtx { kind: a.kind, size_m: a.size_m, max_texture_size: 4096, displacement_scale: 0.03 };
    for set in a.sets.iter().filter(|s| !s.billboard) {
        let plan = zestengine_assets::pack_convert::plan_material(set, &ctx);
        if plan.is_empty() {
            continue;
        }
        ui.label(RichText::new(&set.name).strong());
        for l in &plan.notes {
            if l.level == Level::Info && l.property != "tiling" {
                continue;
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("  {}", l.property)).color(level_color(l.level)));
                ui.label(RichText::new(&l.note).color(theme::TEXT_SECONDARY).size(11.0));
            });
        }
    }
    if a.kind == AssetKind::Plant {
        ui.label(RichText::new("  billboard: skipped, the engine builds its own impostor").color(theme::TEXT_MUTED).size(11.0));
    }
}

fn spawn_import(tasks: &mut TaskManager, assets: Vec<PackAsset>, opts: ImportOptions, project_root: PathBuf, temp: Option<TempDir>) {
    let n = assets.iter().filter(|a| a.include).count();
    let label = format!("Import {n} Megascans / Fab asset(s)");
    // The extracted folder must outlive the import: it is dropped when the task ends.
    let temp = Mutex::new(temp);
    tasks.spawn(label, TaskKind::PackImport, false, move |cancel, progress| {
        let _keep = temp.lock().ok().and_then(|mut t| t.take());
        let outcome = import_assets(&assets, &project_root, &opts, &mut |done, total, current| {
            if let Ok(mut p) = progress.lock() {
                *p = TaskProgress::Fraction { done: done as f32 / total.max(1) as f32, label: current.to_owned() };
            }
            !cancel.load(Ordering::Relaxed)
        });
        Ok(TaskPayload::Notice(outcome.message()))
    });
}

// ---------------------------------------------------------------------------
// Downloads watcher
// ---------------------------------------------------------------------------

/// Polls the Downloads folder for new archives. A candidate is offered once,
/// after its size has stayed the same across two polls (so a download still
/// being written is not opened) and only if it was modified after watching began.
#[derive(Default)]
struct Watcher {
    started: Option<SystemTime>,
    last_poll: Option<Instant>,
    sizes: HashMap<PathBuf, u64>,
    offered: HashSet<PathBuf>,
}

/// Names a browser uses for a download in progress.
fn is_partial(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.ends_with(".part") || l.ends_with(".crdownload") || l.ends_with(".tmp") || l.ends_with(".download") || l.starts_with('.')
}

impl Watcher {
    fn reset(&mut self) {
        if self.started.is_some() {
            *self = Watcher::default();
        }
    }

    fn poll(&mut self, now: Instant) -> Vec<PathBuf> {
        let started = *self.started.get_or_insert_with(SystemTime::now);
        if self.last_poll.is_some_and(|t| now.duration_since(t) < WATCH_INTERVAL) {
            return Vec::new();
        }
        self.last_poll = Some(now);
        let Some(dir) = downloads_dir() else { return Vec::new() };
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&dir) else { return out };
        for e in rd.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if is_partial(&name) || self.offered.contains(&path) {
                continue;
            }
            let Ok(meta) = e.metadata() else { continue };
            if meta.modified().map(|m| m < started).unwrap_or(true) {
                continue;
            }
            let ext = path.extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let is_candidate = meta.is_file() && matches!(ext.as_str(), "zip" | "mhpkg") || meta.is_dir();
            if !is_candidate {
                continue;
            }
            let size = if meta.is_dir() { dir_size_hint(&path) } else { meta.len() };
            // Stable across two polls, and non-empty.
            if size > 0 && self.sizes.insert(path.clone(), size) == Some(size) {
                let looks_right = if meta.is_dir() { scan_pack(&path).is_ok_and(|s| !s.assets.is_empty()) } else { peek_archive(&path) != ArchiveKind::Unknown };
                self.offered.insert(path.clone());
                if looks_right {
                    out.push(path);
                }
            }
        }
        out
    }
}

/// Sum of the sizes of the files directly in `dir` and one level below (cheap change detector).
fn dir_size_hint(dir: &Path) -> u64 {
    let mut total = 0;
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    for e in rd.flatten() {
        let Ok(m) = e.metadata() else { continue };
        if m.is_file() {
            total += m.len();
        } else if m.is_dir() && let Ok(inner) = std::fs::read_dir(e.path()) {
            total += inner.flatten().filter_map(|i| i.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum::<u64>();
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_downloads_are_never_offered() {
        for n in ["a.zip.part", "b.crdownload", ".hidden.zip", "c.tmp", "d.zip.download"] {
            assert!(is_partial(n), "{n}");
        }
        assert!(!is_partial("pack.zip"));
        assert!(!is_partial("asi20.mhpkg"));
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human_bytes(10), "10 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(46_282_420), "44.1 MB");
    }

    #[test]
    fn watcher_is_idle_until_polled_and_resets() {
        let mut w = Watcher::default();
        assert!(w.started.is_none());
        let _ = w.poll(Instant::now());
        assert!(w.started.is_some());
        // A second poll inside the interval does nothing.
        assert!(w.poll(Instant::now()).is_empty());
        w.reset();
        assert!(w.started.is_none() && w.sizes.is_empty());
    }

    #[test]
    fn temp_dir_is_removed_on_drop() {
        let t = TempDir::new();
        std::fs::create_dir_all(t.0.join("a")).unwrap();
        let p = t.0.clone();
        drop(t);
        assert!(!p.exists());
    }
}

#[cfg(test)]
#[path = "shots/pack_import.rs"]
mod shots;
