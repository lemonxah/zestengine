//! The Project Hub UI: sidebar navigation, the Projects, New Project and
//! Settings pages, and the open / create / initialise flows. Opening a project
//! starts the editor as its own process (`zestengine <project-dir>`); creating one
//! runs `zestengine --new-project`, so the hub and the editor share one template
//! implementation.

use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use zestengine_assets::launcher::{self, LaunchCommand};
use zestengine_assets::project_templates::{self, ProjectTemplate, TEMPLATES, TemplateCategory, TemplateIcon};
use zestengine_assets::recent_projects::{RecentProject, RecentProjects, RecentSort, now_unix};
use zestengine_assets::{PROJECT_FILE_NAME, ProjectFile};
use zestengine_editor::egui_phosphor::regular as ph;
use zestengine_editor::icons::{self, Icon};
use zestengine_editor::{theme, widgets};
use egui::{Align, Align2, CornerRadius, FontId, Frame, Layout, Margin, Rect, RichText, Sense, Stroke, Ui, Vec2};

use crate::folder_picker::{FolderPicker, PickerResult, home_dir};
use crate::settings::{AfterOpen, HubSettings};
use crate::thumbnails::Thumbnails;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Projects,
    NewProject,
    Settings,
}

/// What a folder picker's result is for.
enum PickFor {
    OpenProject,
    Locate(PathBuf),
    NewLocation,
    DefaultLocation,
    EditorPath,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Info,
    Warn,
    Error,
}

struct Message {
    text: String,
    level: Level,
    at: Instant,
}

/// A running `zestengine --new-project`.
struct Creating {
    name: String,
    template: &'static str,
    rx: mpsc::Receiver<Result<PathBuf, String>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowAction {
    None,
    Open,
    Pin(bool),
    Reveal,
    Locate,
    Remove,
}

/// What the window should do after this frame.
#[derive(Default)]
pub struct HubRequests {
    pub minimize: bool,
    pub close: bool,
}

const ROW_H: f32 = 64.0;
const THUMB: Vec2 = Vec2::new(80.0, 45.0);
const CARD: Vec2 = Vec2::new(212.0, 104.0);
const GAP: f32 = 8.0;
/// How long a status message stays up.
const MESSAGE_SECS: u64 = 10;

pub struct Hub {
    pub settings: HubSettings,
    settings_path: Option<PathBuf>,
    recents: RecentProjects,
    recents_path: Option<PathBuf>,
    page: Page,
    search: String,
    new_name: String,
    new_location: String,
    new_template: &'static str,
    template_filter: Option<TemplateCategory>,
    picker: Option<(FolderPicker, PickFor)>,
    init_prompt: Option<PathBuf>,
    creating: Option<Creating>,
    running: Vec<(PathBuf, Child)>,
    editor: Result<PathBuf, String>,
    message: Option<Message>,
    thumbs: Thumbnails,
    requests: HubRequests,
}

impl Hub {
    pub fn new(ctx: &egui::Context) -> Self {
        let settings_path = HubSettings::default_path();
        let settings = settings_path.as_deref().map(HubSettings::load).unwrap_or_default();
        let recents_path = RecentProjects::default_path();
        let recents = recents_path.as_deref().map(RecentProjects::load).unwrap_or_default();
        let new_template = project_templates::find(&settings.last_template)
            .map_or(project_templates::DEFAULT_TEMPLATE, |t| t.id);
        let new_location = settings.default_location.clone();
        let mut hub = Self {
            page: if recents.projects.is_empty() { Page::NewProject } else { Page::Projects },
            new_name: unique_name(Path::new(&new_location), "my-game"),
            new_location,
            new_template,
            settings,
            settings_path,
            recents,
            recents_path,
            search: String::new(),
            template_filter: None,
            picker: None,
            init_prompt: None,
            creating: None,
            running: Vec::new(),
            editor: Err(String::new()),
            message: None,
            thumbs: Thumbnails::new(ctx),
            requests: HubRequests::default(),
        };
        hub.refresh_editor();
        hub
    }

    fn refresh_editor(&mut self) {
        let explicit = PathBuf::from(self.settings.editor_path.trim());
        self.editor = launcher::find_executable(launcher::EDITOR_BIN, Some(&explicit), launcher::EDITOR_ENV)
            .map_err(|e| format!("{e:#}"));
    }

    /// Re-read state other processes change: the recent list (the editor
    /// records projects it opens) and the editor location. Called on focus.
    pub fn refresh(&mut self) {
        if let Some(path) = &self.recents_path {
            self.recents = RecentProjects::load(path);
        }
        self.thumbs.clear();
        self.refresh_editor();
    }

    pub fn save_settings(&self) {
        if let Some(path) = &self.settings_path
            && let Err(e) = self.settings.save(path)
        {
            tracing::warn!("saving hub settings: {e:#}");
        }
    }

    /// Change the recent list: re-read it first (the editor may have written
    /// since), apply `f`, write it back.
    fn update_recents(&mut self, f: impl FnOnce(&mut RecentProjects)) {
        let Some(path) = self.recents_path.clone() else {
            f(&mut self.recents);
            return;
        };
        let mut recents = RecentProjects::load(&path);
        f(&mut recents);
        if let Err(e) = recents.save(&path) {
            self.notify(format!("Could not save the recent list: {e:#}"), Level::Error);
        }
        self.recents = recents;
    }

    fn notify(&mut self, text: impl Into<String>, level: Level) {
        let text = text.into();
        match level {
            Level::Error => tracing::error!("{text}"),
            Level::Warn => tracing::warn!("{text}"),
            Level::Info => tracing::info!("{text}"),
        }
        self.message = Some(Message { text, level, at: Instant::now() });
    }

    /// Start the editor on `root`, or offer to initialise a folder with no project.
    fn open_project(&mut self, root: &Path) {
        if !root.join(PROJECT_FILE_NAME).is_file() {
            if root.is_dir() {
                self.init_prompt = Some(root.to_path_buf());
            } else {
                self.notify(format!("{} no longer exists", root.display()), Level::Warn);
            }
            return;
        }
        if self.is_running(root) {
            self.notify(format!("{} is already open in the editor", folder_name(root)), Level::Info);
            return;
        }
        let editor = match &self.editor {
            Ok(e) => e.clone(),
            Err(e) => {
                self.notify(format!("Editor not found. Set its path in Settings. {e}"), Level::Error);
                return;
            }
        };
        let cmd = LaunchCommand::open_project(&editor, root);
        match cmd.command().spawn() {
            Ok(child) => {
                tracing::info!("started {}", cmd.display());
                self.running.push((zestengine_assets::recent_projects::normalize_path(root), child));
            }
            Err(e) => {
                self.notify(format!("Could not start the editor ({}): {e}", cmd.display()), Level::Error);
                return;
            }
        }
        let name = zestengine_assets::load_project_file(root).map(|p| p.name).unwrap_or_else(|_| folder_name(root));
        let key = root.to_path_buf();
        // The editor records it too; doing it here keeps the list right even
        // if the editor fails to start up.
        self.update_recents(|r| r.record(&key, &name, "", now_unix()));
        self.notify(format!("Opening {name}"), Level::Info);
        match self.settings.after_open {
            AfterOpen::KeepOpen => {}
            AfterOpen::Minimize => self.requests.minimize = true,
            AfterOpen::Close => self.requests.close = true,
        }
    }

    /// Write a default project.zestengine into `root` and open it.
    fn initialise_and_open(&mut self, root: &Path) {
        let project = ProjectFile { name: folder_name(root), ..Default::default() };
        match zestengine_assets::save_project_file(root, &project) {
            Ok(()) => self.open_project(root),
            Err(e) => self.notify(format!("Could not initialise {}: {e:#}", root.display()), Level::Error),
        }
    }

    fn start_create(&mut self, ctx: &egui::Context) {
        let parent = PathBuf::from(self.new_location.trim());
        let name = self.new_name.trim().to_string();
        if let Err(e) = validate_new_project(&parent, &name) {
            self.notify(e, Level::Warn);
            return;
        }
        let editor = match &self.editor {
            Ok(e) => e.clone(),
            Err(e) => {
                self.notify(format!("Editor not found. Set its path in Settings. {e}"), Level::Error);
                return;
            }
        };
        if let Err(e) = std::fs::create_dir_all(&parent) {
            self.notify(format!("Could not create {}: {e}", parent.display()), Level::Error);
            return;
        }
        self.settings.last_template = self.new_template.to_string();
        self.save_settings();
        let cmd = LaunchCommand::new_project(&editor, &parent, &name, self.new_template);
        tracing::info!("running {}", cmd.display());
        let (tx, rx) = mpsc::channel();
        let root = parent.join(&name);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = match cmd.command().output() {
                Ok(out) if out.status.success() => Ok(root),
                Ok(out) => Err(failure_text(&out)),
                Err(e) => Err(format!("could not run {}: {e}", cmd.display())),
            };
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.creating = Some(Creating { name, template: self.new_template, rx });
    }

    /// Collect finished background work: project creation and exited editors.
    fn poll(&mut self) {
        if let Some(c) = &self.creating
            && let Ok(result) = c.rx.try_recv()
        {
            let name = c.name.clone();
            self.creating = None;
            match result {
                Ok(root) => {
                    self.new_name = unique_name(Path::new(&self.new_location), "my-game");
                    self.page = Page::Projects;
                    self.notify(format!("Created {name}"), Level::Info);
                    self.open_project(&root);
                }
                Err(e) => self.notify(format!("Creating {name} failed: {e}"), Level::Error),
            }
        }
        let mut exited = Vec::new();
        self.running.retain_mut(|(root, child)| match child.try_wait() {
            Ok(None) => true,
            Ok(Some(status)) => {
                if !status.success() {
                    exited.push(format!("The editor for {} exited with {status}", folder_name(root)));
                }
                false
            }
            Err(_) => false,
        });
        for text in exited {
            self.notify(text, Level::Warn);
        }
    }

    fn is_running(&self, root: &Path) -> bool {
        let key = zestengine_assets::recent_projects::normalize_path(root);
        self.running.iter().any(|(p, _)| *p == key)
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> HubRequests {
        self.poll();
        // Creating shows a spinner; a running editor only needs an occasional exit check, so
        // a launcher left open beside it does not redraw twice a second.
        if self.creating.is_some() {
            ctx.request_repaint_after(Duration::from_millis(500));
        } else if !self.running.is_empty() {
            ctx.request_repaint_after(Duration::from_secs(2));
        }
        if self.message.as_ref().is_some_and(|m| m.at.elapsed() > Duration::from_secs(MESSAGE_SECS)) {
            self.message = None;
        }
        let modal_open = self.picker.is_some() || self.init_prompt.is_some() || self.creating.is_some();
        if !modal_open {
            let (new, open) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::COMMAND, egui::Key::N),
                    i.consume_key(egui::Modifiers::COMMAND, egui::Key::O),
                )
            });
            if new {
                self.page = Page::NewProject;
            }
            if open {
                self.pick(PickFor::OpenProject);
            }
        }

        egui::SidePanel::left("hub-nav")
            .exact_width(216.0)
            .resizable(false)
            .frame(Frame::new().fill(theme::BG_CHROME).inner_margin(Margin::symmetric(12, 16)))
            .show(ctx, |ui| self.sidebar(ui));
        egui::TopBottomPanel::bottom("hub-status")
            .exact_height(theme::H_STATUS_BAR)
            .frame(Frame::new().fill(theme::BG_CHROME).inner_margin(Margin::symmetric(16, 0)))
            .show(ctx, |ui| self.status_bar(ui));
        egui::CentralPanel::default()
            .frame(Frame::new().fill(theme::BG_APP).inner_margin(Margin::symmetric(28, 22)))
            .show(ctx, |ui| match self.page {
                Page::Projects => self.projects_page(ui),
                Page::NewProject => self.new_project_page(ui),
                Page::Settings => self.settings_page(ui),
            });

        self.modals(ctx);
        std::mem::take(&mut self.requests)
    }

    fn pick(&mut self, purpose: PickFor) {
        let (title, start, file) = match &purpose {
            PickFor::OpenProject => ("Open a project folder", self.last_parent(), false),
            PickFor::Locate(old) => ("Locate the moved project", old.parent().map_or_else(home_dir, Path::to_path_buf), false),
            PickFor::NewLocation => ("Location for the new project", PathBuf::from(&self.new_location), false),
            PickFor::DefaultLocation => ("Default location for new projects", PathBuf::from(&self.settings.default_location), false),
            PickFor::EditorPath => (
                "Editor executable",
                self.editor.as_ref().ok().and_then(|p| p.parent()).map_or_else(home_dir, Path::to_path_buf),
                true,
            ),
        };
        self.picker = Some((FolderPicker::new(title, &start, file), purpose));
    }

    /// Folder to start the Open picker in: beside the last project.
    fn last_parent(&self) -> PathBuf {
        self.recents
            .last_opened
            .as_deref()
            .and_then(Path::parent)
            .filter(|p| p.is_dir())
            .map_or_else(home_dir, Path::to_path_buf)
    }

    // ---------------------------------------------------------------- chrome

    fn sidebar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(26.0), Sense::hover());
            icons::paint_logo(ui.painter(), rect, theme::accent(), theme::BG_CHROME);
            ui.add_space(4.0);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                ui.label(RichText::new("ZestEngine").font(theme::semibold(15.0)).color(theme::TEXT_STRONG));
                ui.label(RichText::new("Project Hub").size(theme::SIZE_META).color(theme::TEXT_DIM));
            });
        });
        ui.add_space(20.0);
        for (page, icon, label) in [
            (Page::Projects, Icon::Folder, "Projects"),
            (Page::NewProject, Icon::Plus, "New Project"),
            (Page::Settings, Icon::Gear, "Settings"),
        ] {
            if nav_item(ui, icon, label, self.page == page).clicked() {
                self.page = page;
            }
            ui.add_space(2.0);
        }

        ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
            ui.label(
                RichText::new(format!("Hub {}", env!("CARGO_PKG_VERSION"))).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM),
            );
            ui.add_space(2.0);
            let (color, text, tip) = match &self.editor {
                Ok(p) => (theme::STATUS_OK, "Editor ready", p.display().to_string()),
                Err(e) => (theme::STATUS_WARN_FG, "Editor not found", e.clone()),
            };
            let r = ui
                .horizontal(|ui| {
                    let (dot, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                    ui.painter().circle_filled(dot.center(), 3.5, color);
                    ui.label(RichText::new(text).size(theme::SIZE_META).color(theme::TEXT_MUTED));
                })
                .response;
            r.on_hover_text(tip);
        });
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            match &self.message {
                Some(m) => {
                    let (icon, color) = match m.level {
                        Level::Info => (Icon::Info, theme::STATUS_INFO),
                        Level::Warn => (Icon::Warning, theme::STATUS_WARN_FG),
                        Level::Error => (Icon::Warning, theme::STATUS_ERROR),
                    };
                    icons::icon(ui, icon, 13.0, color);
                    ui.add(egui::Label::new(RichText::new(&m.text).size(theme::SIZE_META).color(theme::TEXT_SECONDARY)).truncate())
                        .on_hover_text(&m.text);
                }
                None => {
                    let n = self.recents.projects.len();
                    let running = self.running.len();
                    let mut text = format!("{n} project{}", if n == 1 { "" } else { "s" });
                    if running > 0 {
                        text.push_str(&format!("   {running} open in the editor"));
                    }
                    ui.label(RichText::new(text).size(theme::SIZE_META).color(theme::TEXT_DIM));
                }
            }
        });
    }

    fn page_header(ui: &mut Ui, title: &str, subtitle: &str, right: impl FnOnce(&mut Ui)) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.label(RichText::new(title).font(theme::semibold(18.0)).color(theme::TEXT_STRONG));
                ui.label(RichText::new(subtitle).size(theme::SIZE_BODY).color(theme::TEXT_MUTED));
            });
            ui.with_layout(Layout::right_to_left(Align::Center), right);
        });
        ui.add_space(16.0);
    }

    // ---------------------------------------------------------------- projects

    fn projects_page(&mut self, ui: &mut Ui) {
        let mut go_new = false;
        let mut open = false;
        Self::page_header(ui, "Projects", "Open a recent project, or create one from a template.", |ui| {
            go_new = widgets::primary_button(ui, Some(Icon::Plus), "New Project").clicked();
            open = widgets::secondary_button_sized(ui, Some(Icon::Folder), "Open…", theme::H_PRIMARY)
                .on_hover_text("Open a project folder (Ctrl+O)")
                .clicked();
        });
        if go_new {
            self.page = Page::NewProject;
        }
        if open {
            self.pick(PickFor::OpenProject);
        }

        if self.recents.projects.is_empty() {
            self.empty_state(ui);
            return;
        }

        ui.horizontal(|ui| {
            widgets::search_field(ui, &mut self.search, "Search projects", 300.0);
            ui.add_space(8.0);
            let before = self.settings.sort;
            widgets::segmented(ui, &mut self.settings.sort, &[(RecentSort::LastOpened, "Recent"), (RecentSort::Name, "Name")]);
            if self.settings.sort != before {
                self.save_settings();
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let missing = self.recents.projects.iter().filter(|p| p.is_missing()).count();
                if missing > 0
                    && widgets::secondary_button(ui, None, &format!("Remove {missing} missing"))
                        .on_hover_text("Drop entries whose folder no longer exists (nothing is deleted)")
                        .clicked()
                {
                    self.update_recents(|r| {
                        r.prune_missing();
                    });
                }
            });
        });
        ui.add_space(12.0);

        let list: Vec<RecentProject> = self.recents.sorted(self.settings.sort, &self.search).into_iter().cloned().collect();
        if list.is_empty() {
            ui.label(RichText::new("No projects match the search.").color(theme::TEXT_DIM));
            return;
        }
        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter)) && !ui.ctx().wants_keyboard_input();
        let mut action: Option<(PathBuf, RowAction)> = None;
        let pinned_count = list.iter().filter(|p| p.pinned).count();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for (i, p) in list.iter().enumerate() {
                if i == 0 && pinned_count > 0 {
                    widgets::eyebrow(ui, "Pinned");
                } else if i == pinned_count && pinned_count > 0 {
                    ui.add_space(8.0);
                    widgets::eyebrow(ui, "Recent");
                }
                let a = self.project_row(ui, p);
                if a != RowAction::None {
                    action = Some((p.path.clone(), a));
                }
                ui.add_space(6.0);
            }
        });
        // Enter opens the top project: the most recent one unless one is pinned.
        if enter
            && action.is_none()
            && let Some(top) = list.first()
        {
            action = Some((top.path.clone(), if top.is_missing() { RowAction::Locate } else { RowAction::Open }));
        }
        if let Some((path, a)) = action {
            self.row_action(&path, a);
        }
    }

    fn row_action(&mut self, path: &Path, action: RowAction) {
        match action {
            RowAction::None => {}
            RowAction::Open => self.open_project(path),
            RowAction::Pin(on) => self.update_recents(|r| {
                r.set_pinned(path, on);
            }),
            RowAction::Reveal => {
                if let Err(e) = launcher::reveal_in_file_manager(path) {
                    self.notify(format!("Could not open the file manager: {e}"), Level::Warn);
                }
            }
            RowAction::Locate => self.pick(PickFor::Locate(path.to_path_buf())),
            RowAction::Remove => {
                self.update_recents(|r| {
                    r.remove(path);
                });
            }
        }
    }

    fn project_row(&mut self, ui: &mut Ui, p: &RecentProject) -> RowAction {
        let missing = p.is_missing();
        let running = self.is_running(&p.path);
        let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), ROW_H), Sense::click());
        let mut action = RowAction::None;
        if response.clicked() {
            action = if missing { RowAction::Locate } else { RowAction::Open };
        }
        response.context_menu(|ui| {
            zestengine_editor::widgets::menu_frame(ui);
            if let Some(a) = row_menu(ui, p, missing) {
                action = a;
            }
        });
        if !ui.is_rect_visible(rect) {
            return action;
        }

        // The pointer anywhere in the row, the Open button and kebab included:
        // `hovered()` turns false over the button, which then hid itself.
        let hovered = response.contains_pointer();
        let (fill, stroke) = if hovered {
            (theme::BG_PANEL, Stroke::new(1.0_f32, theme::LINE_BUTTON))
        } else {
            (theme::BG_PANEL, Stroke::new(1.0_f32, theme::LINE_PANEL))
        };
        ui.painter().rect(rect, theme::R_TOOLBAR, fill, stroke, egui::StrokeKind::Inside);

        // Thumbnail, or the project's initial on a field tile.
        let thumb = Rect::from_min_size(egui::pos2(rect.left() + 10.0, rect.center().y - THUMB.y * 0.5), THUMB);
        let tex = if missing { None } else { self.thumbs.get(ui.ctx(), &p.path).cloned() };
        match tex {
            Some(tex) => egui::Image::new(&tex).corner_radius(theme::R_ROW).paint_at(ui, thumb),
            None => {
                let painter = ui.painter();
                painter.rect_filled(thumb, theme::R_ROW, theme::BG_FIELD);
                if missing {
                    icons::paint(painter, Rect::from_center_size(thumb.center(), Vec2::splat(18.0)), Icon::Warning, theme::STATUS_WARN_FG);
                } else {
                    let initial = p.name.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_default();
                    painter.text(thumb.center(), Align2::CENTER_CENTER, initial, theme::semibold(20.0), theme::TEXT_DIM);
                }
            }
        }

        // Right column: open button (on hover), kebab, time + status.
        let kebab_rect = Rect::from_center_size(egui::pos2(rect.right() - 22.0, rect.center().y), Vec2::splat(theme::H_BUTTON_SMALL));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(kebab_rect));
        let kebab = widgets::icon_button_sized(&mut child, Icon::Kebab, "More actions", false, Vec2::splat(theme::H_BUTTON_SMALL));
        egui::Popup::menu(&kebab).show(|ui| {
            if let Some(a) = row_menu(ui, p, missing) {
                action = a;
            }
        });
        let info_right = kebab_rect.left() - 10.0;
        let mut text_right = info_right - 150.0;
        if hovered {
            let label = if missing { "Locate…" } else { "Open" };
            let w = 72.0;
            let btn_rect = Rect::from_min_size(egui::pos2(info_right - w, rect.center().y - theme::H_FIELD * 0.5), Vec2::new(w, theme::H_FIELD));
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(btn_rect).layout(Layout::right_to_left(Align::Center)));
            let clicked = if missing {
                widgets::secondary_button(&mut child, None, label).clicked()
            } else {
                widgets::primary_button_sized(&mut child, None, label, theme::H_FIELD).clicked()
            };
            if clicked {
                action = if missing { RowAction::Locate } else { RowAction::Open };
            }
            text_right = btn_rect.left() - 150.0;
        } else {
            let painter = ui.painter();
            let when = if p.last_opened == 0 { "Never opened".to_string() } else { format!("Opened {}", ago(now_unix(), p.last_opened)) };
            painter.text(egui::pos2(info_right, rect.center().y - 9.0), Align2::RIGHT_CENTER, when, FontId::proportional(theme::SIZE_META), theme::TEXT_MUTED);
            let (status, color) = if missing {
                ("Folder missing".to_string(), theme::STATUS_WARN_FG)
            } else if running {
                ("Open in the editor".to_string(), theme::accent())
            } else if p.engine_version.is_empty() {
                (String::new(), theme::TEXT_DIM)
            } else {
                (format!("ZestEngine {}", p.engine_version), theme::TEXT_DIM)
            };
            painter.text(egui::pos2(info_right, rect.center().y + 9.0), Align2::RIGHT_CENTER, status, theme::mono(theme::SIZE_META), color);
        }

        // Name (+ pin) and path, clipped before the right column.
        let text_left = thumb.right() + 14.0;
        let clip = Rect::from_min_max(egui::pos2(text_left, rect.top()), egui::pos2(text_right.max(text_left + 40.0), rect.bottom()));
        let painter = ui.painter().with_clip_rect(clip);
        let name_color = if missing { theme::TEXT_DIM } else { theme::TEXT_STRONG };
        let name = painter.text(egui::pos2(text_left, rect.center().y - 9.0), Align2::LEFT_CENTER, &p.name, theme::semibold(theme::SIZE_TITLE), name_color);
        if p.pinned {
            painter.text(
                egui::pos2(name.right() + 6.0, rect.center().y - 9.0),
                Align2::LEFT_CENTER,
                ph::PUSH_PIN,
                FontId::proportional(theme::SIZE_BODY),
                theme::accent(),
            );
        }
        painter.text(
            egui::pos2(text_left, rect.center().y + 9.0),
            Align2::LEFT_CENTER,
            p.path.display().to_string(),
            theme::mono(theme::SIZE_META),
            theme::TEXT_DIM,
        );
        action
    }

    fn empty_state(&mut self, ui: &mut Ui) {
        ui.add_space(60.0);
        ui.vertical_centered(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(48.0), Sense::hover());
            icons::paint_logo(ui.painter(), rect, theme::with_alpha(theme::accent(), 0.5), theme::BG_APP);
            ui.add_space(12.0);
            ui.label(RichText::new("No projects yet").font(theme::semibold(15.0)).color(theme::TEXT_STRONG));
            ui.label(
                RichText::new("Create a project from a template, or open a folder that already holds one.").color(theme::TEXT_MUTED),
            );
            ui.add_space(14.0);
            let w = widgets::action_button_width(ui, Some(Icon::Plus), "New Project", true, theme::H_PRIMARY)
                + ui.spacing().item_spacing.x
                + widgets::action_button_width(ui, Some(Icon::Folder), "Open…", false, theme::H_PRIMARY);
            ui.allocate_ui_with_layout(Vec2::new(w, theme::H_PRIMARY), Layout::left_to_right(Align::Center), |ui| {
                if widgets::primary_button(ui, Some(Icon::Plus), "New Project").clicked() {
                    self.page = Page::NewProject;
                }
                if widgets::secondary_button_sized(ui, Some(Icon::Folder), "Open…", theme::H_PRIMARY).clicked() {
                    self.pick(PickFor::OpenProject);
                }
            });
        });
    }

    // ---------------------------------------------------------------- new project

    fn new_project_page(&mut self, ui: &mut Ui) {
        Self::page_header(ui, "New Project", "Pick a template, name the project and choose where it goes.", |_| {});
        ui.horizontal(|ui| {
            widgets::eyebrow(ui, "Templates");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                widgets::segmented(
                    ui,
                    &mut self.template_filter,
                    &[(None, "All"), (Some(TemplateCategory::Desktop), "Desktop"), (Some(TemplateCategory::Vr), "VR")],
                );
            });
        });
        ui.add_space(6.0);
        let columns = (((ui.available_width() + GAP) / (CARD.x + GAP)).floor() as usize).max(1);
        let shown: Vec<&ProjectTemplate> =
            TEMPLATES.iter().filter(|t| self.template_filter.is_none_or(|f| t.category == f)).collect();
        let rows = shown.len().div_ceil(columns).max(1);
        let grid_h = (rows as f32 * (CARD.y + GAP) + GAP).min(260.0);
        egui::ScrollArea::vertical().id_salt("templates").max_height(grid_h).auto_shrink([false, false]).show(ui, |ui| {
            for row in shown.chunks(columns) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = GAP;
                    for t in row {
                        let r = widgets::select_card(ui, CARD, template_icon(t.icon), t.name, t.category.label(), t.description, self.new_template == t.id)
                            .on_hover_text(t.description);
                        if r.clicked() {
                            self.new_template = t.id;
                        }
                        if r.double_clicked() {
                            self.new_template = t.id;
                            self.start_create(ui.ctx());
                        }
                    }
                });
                ui.add_space(GAP);
            }
        });
        ui.add_space(14.0);
        self.new_project_preview(ui);
    }

    fn new_project_preview(&mut self, ui: &mut Ui) {
        let Some(t) = project_templates::find(self.new_template) else { return };
        Frame::new()
            .fill(theme::BG_PANEL)
            .stroke(Stroke::new(1.0_f32, theme::LINE_PANEL))
            .corner_radius(CornerRadius::same(theme::R_TOOLBAR))
            .inner_margin(Margin::same(16))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.allocate_ui_with_layout(Vec2::new(ui.available_width(), 200.0), Layout::top_down(Align::Center), |ui| {
                    ui.add_space(24.0);
                    let (tile, _) = ui.allocate_exact_size(Vec2::splat(96.0), Sense::hover());
                    ui.painter().rect_filled(tile, theme::R_TOOLBAR, theme::with_alpha(theme::accent(), 0.12));
                    icons::paint(ui.painter(), Rect::from_center_size(tile.center(), Vec2::splat(52.0)), template_icon(t.icon), theme::accent());
                    ui.add_space(10.0);
                    ui.label(RichText::new("Template preview image").size(theme::SIZE_META).color(theme::TEXT_DIM));
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(t.name).font(theme::semibold(18.0)).color(theme::TEXT_STRONG));
                    ui.add_space(8.0);
                    widgets::chip(ui, t.category.label());
                });
                ui.add_space(6.0);
                ui.label(RichText::new(t.description).color(theme::TEXT_SECONDARY));
                ui.add_space(8.0);
                for f in t.features {
                    ui.horizontal_top(|ui| {
                        icons::icon(ui, Icon::Check, 12.0, theme::STATUS_OK);
                        ui.add(egui::Label::new(RichText::new(*f).size(theme::SIZE_META).color(theme::TEXT_MUTED)).wrap());
                    });
                }

                ui.add_space(14.0);
                widgets::divider(ui);
                ui.add_space(8.0);
                widgets::eyebrow(ui, "Project");
                let label_w = 72.0;
                widgets::property_row_w(ui, "Name", label_w, |ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.new_name).desired_width(ui.available_width()))
                });
                let mut browse = false;
                widgets::property_row_w(ui, "Location", label_w, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.new_location).desired_width(ui.available_width() - 70.0));
                        browse = widgets::secondary_button(ui, None, "Browse").clicked();
                    });
                });
                if browse {
                    self.pick(PickFor::NewLocation);
                }
                let parent = PathBuf::from(self.new_location.trim());
                let name = self.new_name.trim().to_string();
                ui.add_space(4.0);
                ui.add(
                    egui::Label::new(RichText::new(parent.join(&name).display().to_string()).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM))
                        .wrap(),
                );
                let valid = validate_new_project(&parent, &name);
                if let Err(e) = &valid {
                    ui.add_space(2.0);
                    ui.label(RichText::new(e).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
                }
                if let Err(e) = &self.editor {
                    ui.add_space(2.0);
                    ui.label(RichText::new("Editor not found: set its path in Settings.").size(theme::SIZE_META).color(theme::STATUS_WARN_FG))
                        .on_hover_text(e);
                }
                ui.add_space(12.0);
                let ready = valid.is_ok() && self.editor.is_ok() && self.creating.is_none();
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let create = ui
                        .add_enabled_ui(ready, |ui| widgets::primary_button(ui, Some(Icon::Plus), "Create & Open").clicked())
                        .inner;
                    if create {
                        self.start_create(ui.ctx());
                    }
                });
            });
    }

    // ---------------------------------------------------------------- settings

    fn settings_page(&mut self, ui: &mut Ui) {
        Self::page_header(ui, "Settings", "Where the editor lives and how the hub behaves.", |_| {});
        let mut changed = false;
        let width = ui.available_width().min(640.0);
        let label_w = 150.0;
        ui.allocate_ui_with_layout(Vec2::new(width, ui.available_height()), Layout::top_down(Align::Min), |ui| {
            widgets::eyebrow(ui, "Editor");
            let mut browse = false;
            widgets::property_row_w(ui, "Editor executable", label_w, |ui| {
                ui.horizontal(|ui| {
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut self.settings.editor_path)
                            .hint_text("Automatic (next to the hub)")
                            .desired_width(ui.available_width() - 70.0),
                    );
                    changed |= field.lost_focus() || (field.changed() && self.settings.editor_path.is_empty());
                    browse = widgets::secondary_button(ui, None, "Browse").clicked();
                });
            });
            if browse {
                self.pick(PickFor::EditorPath);
            }
            let (text, color) = match &self.editor {
                Ok(p) => (format!("Using {}", p.display()), theme::TEXT_DIM),
                Err(e) => (e.clone(), theme::STATUS_WARN_FG),
            };
            ui.add(egui::Label::new(RichText::new(text).font(theme::mono(theme::SIZE_META)).color(color)).wrap());
            widgets::info_note(
                ui,
                &format!(
                    "Looked up in order: this setting, the {} environment variable, then next to the hub executable.",
                    launcher::EDITOR_ENV
                ),
            );

            widgets::section_gap(ui);
            widgets::eyebrow(ui, "Opening projects");
            widgets::property_row_w(ui, "After opening", label_w, |ui| {
                let before = self.settings.after_open;
                widgets::segmented(
                    ui,
                    &mut self.settings.after_open,
                    &[(AfterOpen::KeepOpen, "Keep the hub open"), (AfterOpen::Minimize, "Minimise the hub"), (AfterOpen::Close, "Close the hub")],
                );
                changed |= before != self.settings.after_open;
            });

            widgets::section_gap(ui);
            widgets::eyebrow(ui, "New projects");
            let mut browse = false;
            widgets::property_row_w(ui, "Default location", label_w, |ui| {
                ui.horizontal(|ui| {
                    let field = ui.add(egui::TextEdit::singleline(&mut self.settings.default_location).desired_width(ui.available_width() - 70.0));
                    changed |= field.lost_focus();
                    browse = widgets::secondary_button(ui, None, "Browse").clicked();
                });
            });
            if browse {
                self.pick(PickFor::DefaultLocation);
            }

            widgets::section_gap(ui);
            widgets::eyebrow(ui, "Recent projects");
            ui.horizontal(|ui| {
                if widgets::secondary_button(ui, None, "Remove missing").clicked() {
                    self.update_recents(|r| {
                        r.prune_missing();
                    });
                }
                if widgets::secondary_button(ui, None, "Clear unpinned")
                    .on_hover_text("Empty the list except pinned projects (nothing is deleted)")
                    .clicked()
                {
                    self.update_recents(|r| r.projects.retain(|p| p.pinned));
                }
            });
            if let Some(path) = &self.recents_path {
                ui.add_space(4.0);
                ui.add(
                    egui::Label::new(RichText::new(format!("Stored in {}", path.display())).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM))
                        .wrap(),
                );
            }
        });
        if changed {
            self.refresh_editor();
            self.save_settings();
        }
    }

    // ---------------------------------------------------------------- modals

    fn modals(&mut self, ctx: &egui::Context) {
        if let Some((picker, _)) = self.picker.as_mut() {
            match picker.ui(ctx) {
                PickerResult::Pending => {}
                PickerResult::Cancelled => self.picker = None,
                PickerResult::Picked(path) => {
                    let (_, purpose) = self.picker.take().expect("picker open");
                    self.picked(purpose, path);
                }
            }
        }

        if let Some(root) = self.init_prompt.clone() {
            let mut close = false;
            let modal = egui::Modal::new(egui::Id::new("zestengine-hub-init")).show(ctx, |ui| {
                ui.set_width(440.0);
                ui.label(RichText::new("No project in this folder").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                ui.add_space(6.0);
                ui.label(RichText::new(root.display().to_string()).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
                ui.add_space(6.0);
                ui.add(
                    egui::Label::new(
                        RichText::new("It has no project.zestengine. Initialise it as an empty zestengine project? Existing files are left as they are.")
                            .color(theme::TEXT_SECONDARY),
                    )
                    .wrap(),
                );
                widgets::dialog_footer(ui, |ui| {
                    if widgets::primary_button(ui, None, "Initialise & Open").clicked() {
                        self.init_prompt = None;
                        self.initialise_and_open(&root);
                    }
                    if widgets::secondary_button(ui, None, "Cancel").clicked() {
                        close = true;
                    }
                });
            });
            if close || modal.should_close() {
                self.init_prompt = None;
            }
        }

        if let Some(c) = &self.creating {
            let template = project_templates::find(c.template).map_or(c.template, |t| t.name);
            let text = format!("Creating {} from {template}…", c.name);
            egui::Modal::new(egui::Id::new("zestengine-hub-creating")).show(ctx, |ui| {
                ui.set_width(360.0);
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(18.0).color(theme::accent()));
                    ui.add_space(6.0);
                    ui.label(RichText::new(text).color(theme::TEXT_BODY));
                });
                ui.add_space(4.0);
                ui.label(RichText::new("Writing scenes, materials and the game crate.").size(theme::SIZE_META).color(theme::TEXT_DIM));
            });
        }
    }

    fn picked(&mut self, purpose: PickFor, path: PathBuf) {
        match purpose {
            PickFor::OpenProject => self.open_project(&path),
            PickFor::Locate(old) => {
                if path.join(PROJECT_FILE_NAME).is_file() {
                    self.update_recents(|r| {
                        r.relocate(&old, &path);
                    });
                } else {
                    self.notify(format!("No project.zestengine in {}", path.display()), Level::Warn);
                }
            }
            PickFor::NewLocation => {
                self.new_location = path.display().to_string();
                self.new_name = unique_name(&path, self.new_name.trim());
            }
            PickFor::DefaultLocation => {
                self.settings.default_location = path.display().to_string();
                self.save_settings();
            }
            PickFor::EditorPath => {
                self.settings.editor_path = path.display().to_string();
                self.refresh_editor();
                self.save_settings();
            }
        }
    }
}

/// Context / kebab menu entries for a project row.
fn row_menu(ui: &mut Ui, p: &RecentProject, missing: bool) -> Option<RowAction> {
    let mut action = None;
    if missing {
        if ui.button("Locate…").clicked() {
            action = Some(RowAction::Locate);
        }
    } else if ui.button("Open").clicked() {
        action = Some(RowAction::Open);
    }
    if ui.button(if p.pinned { "Unpin" } else { "Pin to top" }).clicked() {
        action = Some(RowAction::Pin(!p.pinned));
    }
    if !missing && ui.button("Show in file manager").clicked() {
        action = Some(RowAction::Reveal);
    }
    ui.separator();
    if ui.button("Remove from list").on_hover_text("The project folder is not touched").clicked() {
        action = Some(RowAction::Remove);
    }
    if action.is_some() {
        ui.close();
    }
    action
}

fn nav_item(ui: &mut Ui, icon: Icon, label: &str, selected: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), theme::H_PRIMARY), Sense::click());
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        if selected {
            p.rect_filled(rect, theme::R_ROW, theme::accent_faint());
        } else if response.hovered() {
            p.rect_filled(rect, theme::R_ROW, theme::BG_BUTTON);
        }
        let icon_color = if selected { theme::accent() } else { theme::TEXT_MUTED };
        icons::paint(p, Rect::from_center_size(egui::pos2(rect.left() + 18.0, rect.center().y), Vec2::splat(15.0)), icon, icon_color);
        p.text(
            egui::pos2(rect.left() + 36.0, rect.center().y),
            Align2::LEFT_CENTER,
            label,
            theme::medium(theme::SIZE_BODY),
            if selected { theme::TEXT_STRONG } else { theme::TEXT_SECONDARY },
        );
        widgets::focus_ring(ui, &response, rect, theme::R_ROW);
    }
    response
}

/// The editor icon drawn for a template.
fn template_icon(icon: TemplateIcon) -> Icon {
    match icon {
        TemplateIcon::Empty => Icon::Plus,
        TemplateIcon::Crosshair => Icon::Target,
        TemplateIcon::Camera => Icon::Camera,
        TemplateIcon::Pointer => Icon::ToolSelect,
        TemplateIcon::Grid => Icon::Grid,
        TemplateIcon::Headset => Icon::Gizmo,
        TemplateIcon::Social => Icon::Audio,
    }
}

fn folder_name(root: &Path) -> String {
    root.file_name().map_or_else(|| root.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// Why `parent/name` can't hold a new project, if it can't.
fn validate_new_project(parent: &Path, name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Name the project.".into());
    }
    if name == "." || name == ".." || name.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()) {
        return Err("The name can't contain / \\ : * ? \" < > |".into());
    }
    if parent.as_os_str().is_empty() || !parent.is_absolute() {
        return Err("Choose a location (a full folder path).".into());
    }
    if parent.exists() && !parent.is_dir() {
        return Err("The location is a file, not a folder.".into());
    }
    let root = parent.join(name);
    if root.join(PROJECT_FILE_NAME).exists() {
        return Err("A project already exists there.".into());
    }
    if root.exists() && std::fs::read_dir(&root).map(|mut d| d.next().is_some()).unwrap_or(true) {
        return Err("That folder exists and is not empty.".into());
    }
    Ok(())
}

/// `base`, or `base-2`, `base-3`... whichever is free under `parent`.
fn unique_name(parent: &Path, base: &str) -> String {
    let base = if base.is_empty() { "my-game" } else { base };
    if !parent.join(base).exists() {
        return base.to_string();
    }
    (2..1000).map(|n| format!("{base}-{n}")).find(|n| !parent.join(n).exists()).unwrap_or_else(|| base.to_string())
}

/// The error line(s) of a failed `zestengine --new-project`.
fn failure_text(out: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let lines: Vec<&str> = stderr.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    match lines.len() {
        0 => format!("the editor exited with {}", out.status),
        n => lines[n.saturating_sub(3)..].join(" "),
    }
}

/// "just now", "5 minutes ago", "3 days ago"...
pub fn ago(now: u64, then: u64) -> String {
    let secs = now.saturating_sub(then);
    let unit = |n: u64, what: &str| format!("{n} {what}{} ago", if n == 1 { "" } else { "s" });
    match secs {
        0..60 => "just now".into(),
        60..3600 => unit(secs / 60, "minute"),
        3600..86_400 => unit(secs / 3600, "hour"),
        86_400..2_592_000 => unit(secs / 86_400, "day"),
        2_592_000..31_536_000 => unit(secs / 2_592_000, "month"),
        _ => unit(secs / 31_536_000, "year"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_times() {
        assert_eq!(ago(100, 100), "just now");
        assert_eq!(ago(100, 200), "just now");
        assert_eq!(ago(3600 * 5 + 10, 10), "5 hours ago");
        assert_eq!(ago(60, 0), "1 minute ago");
        assert_eq!(ago(86_400 * 3, 0), "3 days ago");
        assert_eq!(ago(31_536_000 * 2, 0), "2 years ago");
    }

    #[test]
    fn new_project_validation_and_unique_names() {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zestengine-hub-new-{}-{n}", std::process::id()));
        std::fs::create_dir_all(dir.join("taken")).unwrap();
        std::fs::write(dir.join("taken/file.txt"), "x").unwrap();
        std::fs::create_dir_all(dir.join("empty")).unwrap();
        assert!(validate_new_project(&dir, "My Game").is_ok(), "spaces are fine");
        assert!(validate_new_project(&dir, "empty").is_ok(), "an empty folder can be used");
        assert!(validate_new_project(&dir, "").is_err());
        assert!(validate_new_project(&dir, "a/b").is_err());
        assert!(validate_new_project(&dir, "..").is_err());
        assert!(validate_new_project(&dir, "taken").is_err());
        assert!(validate_new_project(Path::new("relative"), "x").is_err());
        assert!(validate_new_project(&dir.join("not-yet"), "x").is_ok(), "a missing location is created");
        assert_eq!(unique_name(&dir, "fresh"), "fresh");
        assert_eq!(unique_name(&dir, "taken"), "taken-2");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
