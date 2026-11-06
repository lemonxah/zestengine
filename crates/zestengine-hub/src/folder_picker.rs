//! In-window folder (or file) browser for Open, New Project location and the
//! editor path: a path field, an Up button and the folder's entries, with
//! zestengine projects marked. Drawn in the hub's own theme, so no native dialog
//! dependency is needed.

use std::path::{Path, PathBuf};

use zestengine_assets::PROJECT_FILE_NAME;
use zestengine_editor::icons::{self, Icon};
use zestengine_editor::{theme, widgets};
use egui::{RichText, Sense, Vec2};

struct Entry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    is_project: bool,
}

pub enum PickerResult {
    Pending,
    Cancelled,
    Picked(PathBuf),
}

pub struct FolderPicker {
    title: String,
    /// Pick a file inside the folder instead of a folder.
    pick_file: bool,
    current: PathBuf,
    path_text: String,
    entries: Vec<Entry>,
    selected: Option<PathBuf>,
    show_hidden: bool,
    error: Option<String>,
}

impl FolderPicker {
    pub fn new(title: &str, start: &Path, pick_file: bool) -> Self {
        let mut p = Self {
            title: title.to_string(),
            pick_file,
            current: PathBuf::new(),
            path_text: String::new(),
            entries: Vec::new(),
            selected: None,
            show_hidden: false,
            error: None,
        };
        // Start at the nearest folder that exists (a typed path may not yet).
        let start = start.ancestors().find(|d| d.is_dir()).map(Path::to_path_buf).unwrap_or_else(home_dir);
        p.navigate(&start);
        p
    }

    fn navigate(&mut self, dir: &Path) {
        match read_entries(dir, self.pick_file, self.show_hidden) {
            Ok(entries) => {
                self.current = dir.to_path_buf();
                self.path_text = dir.display().to_string();
                self.entries = entries;
                self.selected = None;
                self.error = None;
            }
            Err(e) => self.error = Some(format!("{}: {e}", dir.display())),
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> PickerResult {
        let mut result = PickerResult::Pending;
        let mut go: Option<PathBuf> = None;
        let modal = egui::Modal::new(egui::Id::new("zestengine-hub-folder-picker")).show(ctx, |ui| {
            ui.set_width(560.0);
            ui.label(RichText::new(&self.title).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let up = self.current.parent().map(Path::to_path_buf);
                ui.add_enabled_ui(up.is_some(), |ui| {
                    if widgets::icon_button_sized(ui, Icon::ArrowUp, "Up one folder", false, Vec2::splat(theme::H_FIELD)).clicked() {
                        go = up.clone();
                    }
                });
                let field = ui.add(
                    egui::TextEdit::singleline(&mut self.path_text)
                        .font(theme::mono(theme::SIZE_META))
                        .desired_width(ui.available_width() - 70.0),
                );
                if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    go = Some(PathBuf::from(self.path_text.trim()));
                }
                if widgets::secondary_button(ui, None, "Home").clicked() {
                    go = Some(home_dir());
                }
            });
            ui.add_space(6.0);

            let list_h = 320.0;
            egui::Frame::new()
                .fill(theme::BG_FIELD)
                .stroke(egui::Stroke::new(1.0_f32, theme::LINE_FIELD))
                .corner_radius(egui::CornerRadius::same(theme::R_ROW))
                .inner_margin(egui::Margin::same(4))
                .show(ui, |ui| {
                    ui.set_min_height(list_h);
                    egui::ScrollArea::vertical().max_height(list_h).auto_shrink([false, false]).show(ui, |ui| {
                        if self.entries.is_empty() {
                            ui.add_space(8.0);
                            ui.label(RichText::new("This folder is empty.").size(theme::SIZE_META).color(theme::TEXT_DIM));
                        }
                        for e in &self.entries {
                            let selected = self.selected.as_deref() == Some(e.path.as_path());
                            let r = entry_row(ui, e, selected);
                            if r.double_clicked() {
                                // A project folder is picked straight away when picking folders.
                                if e.is_dir && (self.pick_file || !e.is_project) {
                                    go = Some(e.path.clone());
                                } else {
                                    result = PickerResult::Picked(e.path.clone());
                                }
                            } else if r.clicked() {
                                self.selected = Some(e.path.clone());
                            }
                        }
                    });
                });

            if let Some(err) = &self.error {
                ui.add_space(4.0);
                ui.label(RichText::new(err).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.checkbox(&mut self.show_hidden, "Show hidden").changed() {
                    let dir = self.current.clone();
                    self.navigate(&dir);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let pick = if self.pick_file {
                        self.selected.clone().filter(|p| p.is_file())
                    } else {
                        Some(self.selected.clone().filter(|p| p.is_dir()).unwrap_or_else(|| self.current.clone()))
                    };
                    let label = if self.pick_file { "Select File" } else { "Select Folder" };
                    let picked = ui
                        .add_enabled_ui(pick.is_some(), |ui| widgets::primary_button(ui, Some(Icon::Check), label).clicked())
                        .inner;
                    if picked && let Some(p) = pick {
                        result = PickerResult::Picked(p);
                    }
                    if widgets::secondary_button_sized(ui, None, "Cancel", theme::H_PRIMARY).clicked() {
                        result = PickerResult::Cancelled;
                    }
                });
            });
        });
        if let Some(dir) = go {
            self.navigate(&dir);
        }
        if matches!(result, PickerResult::Pending) && modal.should_close() {
            result = PickerResult::Cancelled;
        }
        result
    }
}

fn entry_row(ui: &mut egui::Ui, e: &Entry, selected: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), theme::H_ROW), Sense::click());
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        if selected {
            p.rect_filled(rect, theme::R_FIELD, theme::accent_faint());
        } else if response.hovered() {
            p.rect_filled(rect, theme::R_FIELD, theme::BG_BUTTON);
        }
        let icon = if e.is_dir { Icon::Folder } else { Icon::File };
        let color = if e.is_project { theme::accent() } else { theme::TEXT_MUTED };
        icons::paint(p, egui::Rect::from_center_size(egui::pos2(rect.left() + 16.0, rect.center().y), Vec2::splat(14.0)), icon, color);
        p.text(
            egui::pos2(rect.left() + 32.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            &e.name,
            egui::FontId::proportional(theme::SIZE_BODY),
            if selected { theme::TEXT_SELECTED } else { theme::TEXT_BODY },
        );
        if e.is_project {
            p.text(
                egui::pos2(rect.right() - 10.0, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                "ZestEngine project",
                egui::FontId::proportional(theme::SIZE_META),
                theme::accent(),
            );
        }
    }
    response
}

/// Sub-folders (and files when `files`), folders first, by name.
fn read_entries(dir: &Path, files: bool, hidden: bool) -> std::io::Result<Vec<Entry>> {
    let mut out = Vec::new();
    for item in std::fs::read_dir(dir)?.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        if !hidden && name.starts_with('.') {
            continue;
        }
        let path = item.path();
        let is_dir = path.is_dir();
        if !is_dir && !files {
            continue;
        }
        let is_project = is_dir && path.join(PROJECT_FILE_NAME).is_file();
        out.push(Entry { name, path, is_dir, is_project });
    }
    out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    Ok(out)
}

pub fn home_dir() -> PathBuf {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("/"))
}
