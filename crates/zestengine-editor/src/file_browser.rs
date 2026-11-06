//! Project file browser, Unity-style: a folder tree on the left and an
//! icon grid of the selected folder's contents on the right. Click to
//! inspect, double-click models to import / folders to enter, drag tiles
//! (e.g. `.material` onto a material slot, a viewport mesh, or a folder to
//! move), right-click for create/rename/cut/copy/paste.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use egui::{Align2, FontId, RichText, ScrollArea, Sense, Ui};
use egui_phosphor::regular as ph;

const TILE: egui::Vec2 = egui::vec2(78.0, 88.0);

/// Max image-thumbnail decodes per frame (the rest stream in over later frames).
const THUMB_PER_FRAME: u32 = 8;
/// Max cached thumbnails (each is a live GPU texture); oldest evicted past this.
const THUMB_CACHE_CAP: usize = 256;
/// Longest side of a decoded thumbnail, in texels (downscaled for cheap upload).
/// Sized to stay crisp at the largest tile the size slider allows.
const THUMB_MAX: u32 = 160;

struct Clipboard {
    path: PathBuf,
    /// True = cut (paste moves), false = copy.
    cut: bool,
}

struct Renaming {
    path: PathBuf,
    /// The edited stem (extension hidden while renaming).
    text: String,
    /// The original extension (without the dot); re-appended on commit.
    extension: String,
    /// Focus the text field on its first frame.
    focus: bool,
}

pub struct FileBrowser {
    pub root: PathBuf,
    /// Toolbar search: filters the current folder's tiles by name.
    search: String,
    /// Folder whose contents the grid shows.
    current_dir: PathBuf,
    /// Expanded folders in the tree (the root is always open).
    open_dirs: HashSet<PathBuf>,
    clipboard: Option<Clipboard>,
    renaming: Option<Renaming>,
    /// Ferris (the Rust mascot) tile icon for `.rs` files, decoded + uploaded
    /// once on first use.
    ferris: Option<egui::TextureHandle>,
    /// Decoded image thumbnails keyed by path. `Some(None)` records a decode
    /// failure (e.g. unsupported format) so it isn't retried every frame; a
    /// missing key means "not decoded yet". Built lazily for visible image
    /// tiles, bounded per frame by `thumb_budget`.
    thumbs: std::collections::HashMap<PathBuf, Option<egui::TextureHandle>>,
    /// Insertion order for `thumbs`, so the cache can evict oldest-first and
    /// stay bounded (dropping a handle frees its GPU texture via egui).
    thumb_order: std::collections::VecDeque<PathBuf>,
    /// Remaining new-thumbnail decodes allowed this frame, so scrolling into a
    /// folder full of large images doesn't stall the UI on one frame.
    thumb_budget: u32,
    /// Source-file mtime each cached `.material` sphere preview was rendered from,
    /// so editing a material refreshes its tile instead of showing a stale sphere.
    mat_mtimes: std::collections::HashMap<PathBuf, std::time::SystemTime>,
    /// Grid tile width in px (height tracks it); driven by the size slider so
    /// larger tiles show bigger image previews.
    tile_px: f32,
    /// Per-file LSP problem tally (errors, warnings), refreshed each frame from
    /// the engine. Used to badge files (and aggregate onto folders).
    diags: std::collections::HashMap<PathBuf, (u32, u32)>,
    /// Multi-selection of grid entries (ctrl = add/toggle, shift = range from the
    /// anchor). Synced from the engine each frame via [`set_multi`]; the anchor
    /// (last single/ctrl click) is `response.clicked`.
    multi: HashSet<PathBuf>,
    /// Anchor for shift-range selection (last non-shift click).
    last_click: Option<PathBuf>,
    /// Current grid order (dirs then files) of the visible folder, so shift can
    /// resolve the range between two entries. Rebuilt each frame in `grid_ui`.
    order: Vec<PathBuf>,
    /// A file an asset slot asked to show: highlighted (and scrolled to) for
    /// a moment. The flag says the scroll is still due.
    ping: Option<(PathBuf, f64, bool)>,
    /// Sprite count per image (`None` = not a Sprite texture) and the UI time it
    /// was read at; refreshed every few seconds, never decoding the image.
    sprite_counts: std::collections::HashMap<PathBuf, (f64, Option<u32>)>,
    /// Sprite sheets expanded in the grid (their sprites follow the tile).
    sprite_open: HashSet<PathBuf>,
    /// Sprites per sheet (empty = not a Multiple sheet), re-read every few seconds.
    sprite_kids: Probed<std::sync::Arc<Vec<SpriteChild>>>,
    /// The sprite child last clicked: (texture, sprite id).
    selected_sprite: Option<(PathBuf, u64)>,
    /// Name of the sprite atlas packing each texture (`None` = in no atlas).
    atlas_names: Probed<Option<String>>,
}

/// Reads how many sprites a texture file carries from its `.meta` (set once by
/// the engine; this crate does not know about asset metadata).
static SPRITE_PROBE: std::sync::OnceLock<fn(&Path) -> Option<u32>> = std::sync::OnceLock::new();

/// Install the sprite-count probe behind the sprite badge on image tiles.
pub fn set_sprite_probe(f: fn(&Path) -> Option<u32>) {
    let _ = SPRITE_PROBE.set(f);
}

/// Names the sprite atlas packing a texture (set once by the engine).
static ATLAS_PROBE: std::sync::OnceLock<fn(&Path) -> Option<String>> = std::sync::OnceLock::new();

/// Install the probe behind the "in atlas" badge on sprite textures.
pub fn set_atlas_probe(f: fn(&Path) -> Option<String>) {
    let _ = ATLAS_PROBE.set(f);
}

/// Seconds between re-reading a texture's sprite count.
const SPRITE_COUNT_TTL: f64 = 3.0;

/// One sprite of a Multiple-mode sheet, as the Files grid lists it under the
/// expanded texture. The engine's probe fills `reference` (the text form a
/// drag carries, see `SpriteRef` in the assets crate); this crate never
/// parses it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpriteChild {
    pub id: u64,
    pub name: String,
    pub reference: String,
}

/// Lists a texture's sprites: `(absolute path, project-relative path)`. Set
/// once by the engine, like [`set_sprite_probe`].
static SPRITE_CHILDREN_PROBE: std::sync::OnceLock<fn(&Path, &str) -> Vec<SpriteChild>> = std::sync::OnceLock::new();

/// Install the probe behind the expander on sprite sheets.
pub fn set_sprite_children_probe(f: fn(&Path, &str) -> Vec<SpriteChild>) {
    let _ = SPRITE_CHILDREN_PROBE.set(f);
}

/// egui temp data `(String, u64)`: absolute path and sprite id of the sprite
/// child last clicked in the Files grid. The texture inspector's Sprite
/// section takes it (removing it) when it shows that texture and selects the
/// sprite. Same plugin-safe memory pattern as [`crate::DRAG_FILE_KEY`].
pub const SELECTED_SPRITE_KEY: &str = "zestengine-selected-sprite";
/// egui temp data (`String`): the text reference (`path#id`) of the sprite
/// child being dragged. Drop targets can also take the [`SpriteDrag`] payload.
pub const DRAG_SPRITE_KEY: &str = "zestengine-drag-sprite";

/// Drag-and-drop payload of a sprite child: its text reference
/// (`<project-relative path>#<16 hex id>`, see `SpriteRef` in the assets crate).
#[derive(Clone, Debug, PartialEq)]
pub struct SpriteDrag(pub String);

/// A cache of per-file probe results re-read at most every `ttl` seconds, so
/// a probe that reads a file never runs per frame.
struct Probed<T> {
    ttl: f64,
    map: std::collections::HashMap<PathBuf, (f64, T)>,
}

impl<T: Clone> Probed<T> {
    fn new(ttl: f64) -> Self {
        Self { ttl, map: std::collections::HashMap::new() }
    }

    fn get(&mut self, now: f64, path: &Path, probe: impl FnOnce() -> T) -> T {
        if let Some((at, v)) = self.map.get(path)
            && now - at < self.ttl
        {
            return v.clone();
        }
        let v = probe();
        self.map.insert(path.to_path_buf(), (now, v.clone()));
        v
    }

    /// The last value read, however old.
    fn peek(&self, path: &Path) -> Option<&T> {
        self.map.get(path).map(|(_, v)| v)
    }
}

/// Add `path` to `set`, or remove it when present. Returns the new state.
fn toggle_in(set: &mut HashSet<PathBuf>, path: &Path) -> bool {
    if set.remove(path) {
        false
    } else {
        set.insert(path.to_path_buf());
        true
    }
}

#[derive(Default)]
pub struct FileBrowserResponse {
    /// File single-clicked: select / show in Inspector.
    pub clicked: Option<PathBuf>,
    /// File double-clicked: open it (the engine picks the action by type:
    /// import model, open code editor, etc.).
    pub activated: Option<PathBuf>,
    /// Create a new asset inside this directory.
    pub create_material_in: Option<PathBuf>,
    pub create_postfx_in: Option<PathBuf>,
    pub create_audio_in: Option<(PathBuf, bool)>,
    pub create_waterwaves_in: Option<PathBuf>,
    pub create_physmat_in: Option<PathBuf>,
    pub create_spriteshape_in: Option<PathBuf>,
    pub create_spriteatlas_in: Option<PathBuf>,
    pub create_anim_controller_in: Option<PathBuf>,
    pub create_anim_clip_in: Option<PathBuf>,
    pub create_behaviour_tree_in: Option<PathBuf>,
    pub create_shader_graph_in: Option<PathBuf>,
    pub create_scene_in: Option<PathBuf>,
    pub create_shader_in: Option<PathBuf>,
    pub create_folder_in: Option<PathBuf>,
    /// New Rust component in the project's plugin crate (location-independent).
    pub create_component: bool,
    /// (old, new) paths for renames/moves, so selection can follow.
    pub moved: Vec<(PathBuf, PathBuf)>,
    /// File or folder requested for deletion.
    pub delete: Option<PathBuf>,
    /// Image file requested to become the scene skybox.
    pub set_skybox: Option<PathBuf>,
    /// Unity `.mat` file to convert to a `.material` (Convert Unity Material...).
    pub convert_unity_material: Option<PathBuf>,
    /// Folder of Unity assets to batch-convert (Convert Unity folder...).
    pub convert_unity_folder: Option<PathBuf>,
    /// `.unitypackage` to extract and convert (Import Unity package...).
    pub import_unity_package: Option<PathBuf>,
    /// `.prefab` to make a variant of (Create Prefab Variant).
    pub create_prefab_variant: Option<PathBuf>,
    /// `.ase` / `.aseprite` / `.psd` / `.psb` to import as a sprite sheet (Import as Sprites).
    pub import_layered: Option<PathBuf>,
    /// A PSD's `.layers` file to turn into a prefab (`true`: and place an instance in the scene).
    pub layers_prefab: Option<(PathBuf, bool)>,
    /// Open the Megascans / Fab import dialog (Create menu); `Some(path)` scans that
    /// folder / `.zip` / `.mhpkg` at once (context menu, or dropped on the panel).
    pub import_pack: Option<Option<PathBuf>>,
    /// File or folder to show (selected) in the OS file manager.
    pub show_in_file_manager: Option<PathBuf>,
    /// Empty space in the grid was clicked — clear the current file selection.
    pub clear_selection: bool,
}

/// A folder, `.zip` or `.mhpkg`: something the Megascans / Fab importer can open.
fn is_pack_source(p: &Path) -> bool {
    p.is_dir() || p.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip") || e.eq_ignore_ascii_case("mhpkg"))
}

impl FileBrowser {
    pub fn new(root: PathBuf) -> Self {
        Self {
            current_dir: root.clone(),
            root,
            search: String::new(),
            open_dirs: HashSet::new(),
            clipboard: None,
            renaming: None,
            ferris: None,
            thumbs: std::collections::HashMap::new(),
            thumb_order: std::collections::VecDeque::new(),
            thumb_budget: 0,
            mat_mtimes: std::collections::HashMap::new(),
            tile_px: TILE.x,
            diags: std::collections::HashMap::new(),
            multi: HashSet::new(),
            last_click: None,
            order: Vec::new(),
            ping: None,
            sprite_counts: std::collections::HashMap::new(),
            sprite_open: HashSet::new(),
            sprite_kids: Probed::new(SPRITE_COUNT_TTL),
            selected_sprite: None,
            atlas_names: Probed::new(SPRITE_COUNT_TTL),
        }
    }

    /// The sprites of the sheet at `path` (cached; empty when it has none).
    fn sprite_kids_of(&mut self, now: f64, path: &Path) -> std::sync::Arc<Vec<SpriteChild>> {
        let root = &self.root;
        self.sprite_kids.get(now, path, || match SPRITE_CHILDREN_PROBE.get() {
            Some(probe) => {
                let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/");
                std::sync::Arc::new(probe(path, &rel))
            }
            None => std::sync::Arc::new(Vec::new()),
        })
    }

    /// Sprite count of an image tile, cached for [`SPRITE_COUNT_TTL`].
    fn sprite_count(&mut self, now: f64, path: &Path) -> Option<u32> {
        let probe = SPRITE_PROBE.get()?;
        if let Some((at, n)) = self.sprite_counts.get(path)
            && now - at < SPRITE_COUNT_TTL
        {
            return *n;
        }
        let n = probe(path);
        self.sprite_counts.insert(path.to_path_buf(), (now, n));
        n
    }

    /// Sync the multi-selection from the engine's canonical set before drawing.
    pub fn set_multi(&mut self, set: impl IntoIterator<Item = PathBuf>) {
        self.multi = set.into_iter().collect();
    }

    /// The current multi-selection of files (read after `ui`).
    pub fn multi(&self) -> Vec<PathBuf> {
        self.multi.iter().cloned().collect()
    }

    /// Thumbnail texture for an image file, decoded + downscaled on first use
    /// and cached. Returns `None` while still un-decoded this frame (budget
    /// spent) or if the image can't be decoded, so the caller falls back to the
    /// generic image glyph.
    fn image_thumb(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
        if let Some(slot) = self.thumbs.get(path) {
            return slot.clone();
        }
        if self.thumb_budget == 0 {
            return None; // not cached: retried next frame
        }
        self.thumb_budget -= 1;
        let tex = decode_thumb(path).map(|img| {
            ctx.load_texture(
                format!("zestengine-thumb:{}", path.display()),
                img,
                egui::TextureOptions::LINEAR,
            )
        });
        self.thumbs.insert(path.to_owned(), tex.clone());
        self.thumb_order.push_back(path.to_owned());
        // Bound the cache: dropping the evicted handle frees its GPU texture
        // (egui emits the free in its next texture delta), so browsing many
        // image folders can't grow the egui texture set without limit.
        while self.thumb_order.len() > THUMB_CACHE_CAP {
            if let Some(old) = self.thumb_order.pop_front() {
                self.thumbs.remove(&old);
            }
        }
        tex
    }

    /// Sphere preview for a `.material` file: parse its PBR params and CPU-render
    /// a lit sphere (base colour + metallic + roughness + emission), cached and
    /// budgeted exactly like image thumbnails. Re-renders when the file's mtime
    /// changes so an edit refreshes the tile. Returns `None` until decoded /
    /// when the file can't be parsed (caller falls back to the sphere glyph).
    fn material_thumb(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
        let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        // Invalidate a stale preview if the material file changed on disk.
        if let Some(mt) = mtime
            && self.mat_mtimes.get(path) != Some(&mt)
            && self.thumbs.contains_key(path)
        {
            self.thumbs.remove(path);
        }
        if let Some(slot) = self.thumbs.get(path) {
            return slot.clone();
        }
        if self.thumb_budget == 0 {
            return None;
        }
        self.thumb_budget -= 1;
        let tex = render_material_sphere(path).map(|img| {
            ctx.load_texture(
                format!("zestengine-mat:{}", path.display()),
                img,
                egui::TextureOptions::LINEAR,
            )
        });
        self.thumbs.insert(path.to_owned(), tex.clone());
        self.thumb_order.push_back(path.to_owned());
        if let Some(mt) = mtime {
            self.mat_mtimes.insert(path.to_owned(), mt);
        }
        while self.thumb_order.len() > THUMB_CACHE_CAP {
            if let Some(old) = self.thumb_order.pop_front() {
                self.thumbs.remove(&old);
                self.mat_mtimes.remove(&old);
            }
        }
        tex
    }

    /// (errors, warns) for a file, or aggregated over a folder's descendants.
    fn diag_counts(&self, path: &Path, is_dir: bool) -> (u32, u32) {
        if is_dir {
            let (mut e, mut w) = (0, 0);
            for (p, (pe, pw)) in &self.diags {
                if p.starts_with(path) {
                    e += pe;
                    w += pw;
                }
            }
            (e, w)
        } else {
            self.diags.get(path).copied().unwrap_or((0, 0))
        }
    }

    /// Small problem dot (red = has errors, yellow = warnings only).
    fn diag_dot(painter: &egui::Painter, center: egui::Pos2, errors: u32, warns: u32) {
        if errors == 0 && warns == 0 {
            return;
        }
        let color = if errors > 0 {
            egui::Color32::from_rgb(235, 90, 80)
        } else {
            egui::Color32::from_rgb(220, 180, 70)
        };
        painter.circle_filled(center, 4.0, color);
    }

    /// Lazily decode + upload the embedded Ferris image; cheap to clone after.
    fn ferris_texture(&mut self, ctx: &egui::Context) -> egui::TextureHandle {
        self.ferris
            .get_or_insert_with(|| {
                let bytes = include_bytes!("../assets/ferris.png");
                let image = image::load_from_memory(bytes)
                    .expect("embedded ferris.png decodes")
                    .to_rgba8();
                let size = [image.width() as usize, image.height() as usize];
                // Monochrome silhouette: white RGB + original alpha, so drawing
                // it with a tint colours the whole crab one flat colour, matching
                // the rest of the file icons.
                let mono: Vec<u8> = image.pixels().flat_map(|p| [255, 255, 255, p[3]]).collect();
                let color = egui::ColorImage::from_rgba_unmultiplied(size, &mono);
                ctx.load_texture("zestengine-ferris", color, egui::TextureOptions::LINEAR)
            })
            .clone()
    }

    pub fn ui(
        &mut self,
        ui: &mut Ui,
        selected: Option<&Path>,
        diags: &std::collections::HashMap<PathBuf, (u32, u32)>,
    ) -> FileBrowserResponse {
        let mut response = FileBrowserResponse::default();
        // Refresh the per-frame diagnostics snapshot for the badge helpers.
        self.diags = diags.clone();
        // Cap new image-thumbnail decodes per frame (the rest decode over the
        // next frames) so opening an image-heavy folder doesn't hitch.
        self.thumb_budget = THUMB_PER_FRAME;

        // A folder, .zip or .mhpkg dropped from the OS file manager onto this panel
        // opens the Megascans / Fab import dialog on it. Other dropped files are ignored.
        {
            let panel = ui.max_rect();
            let over = ui.input(|i| i.pointer.hover_pos()).is_none_or(|p| panel.contains(p));
            let hovering = over && ui.input(|i| i.raw.hovered_files.iter().any(|f| f.path.as_ref().is_some_and(|p| is_pack_source(p))));
            if hovering {
                let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("files-drop-hint")));
                painter.rect_filled(panel, 0.0, crate::theme::ACCENT_DEFAULT.gamma_multiply(0.10));
                painter.rect_stroke(panel.shrink(2.0), 4.0, egui::Stroke::new(2.0_f32, crate::theme::ACCENT_DEFAULT), egui::StrokeKind::Inside);
                painter.text(panel.center(), egui::Align2::CENTER_CENTER, "Drop to import a Megascans / Fab asset", FontId::proportional(14.0), crate::theme::TEXT_STRONG);
            }
            if over && let Some(p) = ui.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).find(|p| is_pack_source(p))) {
                response.import_pack = Some(Some(p));
            }
        }

        // F2 starts an inline rename of the selected file/folder (matches the
        // right-click Rename). Gated on no text field already focused so it
        // doesn't fire mid-edit elsewhere.
        if self.renaming.is_none()
            && let Some(path) = selected
            && ui.input(|i| i.key_pressed(egui::Key::F2))
            && !ui.memory(|m| m.focused().is_some())
        {
            self.begin_rename(path);
        }
        // An asset slot was clicked: open its folder and point at the file.
        if let Some(abs) = ui.data_mut(|d| d.remove_temp::<String>(egui::Id::new(crate::components::PING_FILE_KEY))) {
            let path = PathBuf::from(abs);
            if let Some(dir) = path.parent() {
                self.current_dir = dir.to_owned();
                let mut d = dir.to_owned();
                while d.starts_with(&self.root) {
                    self.open_dirs.insert(d.clone());
                    if !d.pop() {
                        break;
                    }
                }
            }
            self.search.clear();
            self.ping = Some((path, ui.input(|i| i.time), true));
        }
        if !self.current_dir.starts_with(&self.root) || !self.current_dir.is_dir() {
            self.current_dir = self.root.clone();
        }

        egui::SidePanel::left(ui.id().with("files-tree"))
            .resizable(true)
            .default_width(188.0)
            .width_range(90.0..=320.0)
            .frame(egui::Frame::new().fill(crate::theme::BG_PANEL).inner_margin(egui::Margin::symmetric(6, 6)))
            .show_inside(ui, |ui| {
                ScrollArea::vertical()
                    .id_salt("files-tree-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let root = self.root.clone();
                        self.tree_row(ui, &root, 0, &mut response);
                        // Space below the tree: drop = move to root,
                        // right-click = ops in the root.
                        let mut rest = ui.available_rect_before_wrap();
                        rest.set_height(rest.height().max(24.0));
                        let bg = ui.allocate_rect(rest, Sense::click());
                        if let Some(payload) = bg.dnd_release_payload::<PathBuf>() {
                            self.move_into(&payload, &root, &mut response);
                        }
                        bg.context_menu(|ui| {
                            crate::widgets::menu_frame(ui);
                            self.paste_button(ui, &root, &mut response);
                            self.create_menu(ui, &root, &mut response);
                            ui.separator();
                            Self::show_in_fm_button(ui, &root, &mut response);
                        });
                    });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(crate::theme::BG_PANEL).inner_margin(egui::Margin::symmetric(10, 0)))
            .show_inside(ui, |ui| {
            use crate::{icons::Icon, theme, widgets};
            ui.horizontal(|ui| {
                ui.set_height(40.0);
                let at_root = self.current_dir == self.root;
                if ui
                    .add_enabled_ui(!at_root, |ui| widgets::icon_button_sized(ui, Icon::ArrowUp, "Parent folder", false, egui::vec2(26.0, 26.0)))
                    .inner
                    .clicked()
                    && let Some(parent) = self.current_dir.parent()
                {
                    self.current_dir = parent.to_owned();
                }
                // Breadcrumbs: Project › a › b. Each crumb opens its folder and takes dropped files.
                let crumbs = breadcrumbs(&self.root, &self.current_dir);
                let last = crumbs.len() - 1;
                ui.spacing_mut().item_spacing.x = 2.0;
                for (i, (label, dir)) in crumbs.into_iter().enumerate() {
                    if i > 0 {
                        ui.label(RichText::new("›").size(theme::SIZE_BODY).color(theme::TEXT_DIM));
                    }
                    let current = i == last;
                    let text = RichText::new(label).font(theme::medium(theme::SIZE_BODY));
                    let text = if current { text.color(theme::TEXT_STRONG) } else { text.color(theme::TEXT_MUTED) };
                    let crumb = ui.add(egui::Button::new(text).frame_when_inactive(false));
                    if crumb.dnd_hover_payload::<PathBuf>().is_some() {
                        ui.painter().rect_stroke(crumb.rect, theme::R_FIELD, egui::Stroke::new(1.0_f32, theme::accent()), egui::StrokeKind::Inside);
                    }
                    let crumb = crumb.on_hover_text(dir.strip_prefix(&self.root).map(|p| p.display().to_string()).unwrap_or_default());
                    if crumb.clicked() && !current {
                        self.current_dir = dir.clone();
                    }
                    if let Some(payload) = crumb.dnd_release_payload::<PathBuf>() {
                        self.move_into(&payload, &dir, &mut response);
                    }
                }
                ui.spacing_mut().item_spacing.x = 8.0;
                let (dirs, files) = list_dir(&self.current_dir);
                ui.label(RichText::new(format!("{} items", dirs.len() + files.len())).size(theme::SIZE_META).color(theme::TEXT_DIM));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dir = self.current_dir.clone();
                    let create = widgets::primary_button_sized(ui, Some(Icon::Plus), "Create", theme::H_FIELD);
                    egui::Popup::menu(&create).show(|ui| self.create_menu(ui, &dir, &mut response));
                    ui.spacing_mut().slider_width = 70.0;
                    crate::widgets::slider(ui, egui::Slider::new(&mut self.tile_px, TILE.x..=176.0).show_value(false)).on_hover_text("Tile size");
                    ui.label(RichText::new("Size").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                    widgets::search_field(ui, &mut self.search, "Search files", 180.0);
                });
            });
            ui.separator();
            ScrollArea::vertical()
                .id_salt("files-grid-scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // The whole grid container senses clicks (tiles keep
                    // priority), so right-click works in any empty spot,
                    // including beside the last tile of a row.
                    let dir = self.current_dir.clone();
                    let scope =
                        ui.scope_builder(egui::UiBuilder::new().sense(Sense::click()), |ui| {
                            self.grid_ui(ui, selected, &mut response);
                            // Stretch the container over the leftover panel
                            // space so it's interactive too.
                            let mut rest = ui.available_rect_before_wrap();
                            rest.set_height(rest.height().max(24.0));
                            ui.allocate_rect(rest, Sense::hover());
                        });
                    // A click that landed on empty grid space (no tile consumed
                    // it) clears the selection.
                    if scope.response.clicked() && response.clicked.is_none() {
                        response.clear_selection = true;
                    }
                    scope.response.context_menu(|ui| {
                        crate::widgets::menu_frame(ui);
                        self.paste_button(ui, &dir, &mut response);
                        self.create_menu(ui, &dir, &mut response);
                        ui.separator();
                        Self::show_in_fm_button(ui, &dir, &mut response);
                    });
                });
        });
        response
    }

    // ------------------------------------------------------------- tree

    fn tree_row(
        &mut self,
        ui: &mut Ui,
        dir: &Path,
        depth: usize,
        response: &mut FileBrowserResponse,
    ) {
        if depth > 16 {
            return;
        }
        let is_root = dir == self.root;
        let subdirs = subdirs_of(dir);
        let open = is_root || self.open_dirs.contains(dir);

        if self.renaming.as_ref().is_some_and(|r| r.path == dir) {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(ui.available_width(), 20.0), Sense::hover());
            self.rename_editor(ui, rect, response);
        } else {
            // Full-width row: clicks, right-clicks, hover and drops work
            // across the whole line, not just over the text.
            let (rect, row) =
                ui.allocate_exact_size(egui::vec2(ui.available_width(), 22.0), Sense::click());
            let indent = 4.0 + depth as f32 * 22.0;
            let current = dir == self.current_dir;
            if current {
                ui.painter().rect_filled(rect, crate::theme::R_ROW, crate::theme::accent_faint());
            } else if row.hovered() {
                ui.painter().rect_filled(rect, crate::theme::R_ROW, crate::theme::with_alpha(crate::theme::BG_BUTTON, 0.6));
            }
            let text_color = if current { crate::theme::TEXT_STRONG } else { crate::theme::TEXT_SECONDARY };
            if !subdirs.is_empty() {
                crate::widgets::paint_caret(
                    ui.painter(),
                    egui::Rect::from_center_size(egui::pos2(rect.left() + indent + 6.0, rect.center().y), egui::vec2(11.0, 11.0)),
                    open,
                    crate::theme::TEXT_DIM,
                );
            }
            crate::icons::paint(
                ui.painter(),
                egui::Rect::from_center_size(egui::pos2(rect.left() + indent + 21.0, rect.center().y), egui::vec2(14.0, 14.0)),
                crate::icons::Icon::Folder,
                if current { crate::theme::accent() } else { crate::theme::TEXT_MUTED },
            );
            let name = if is_root {
                "Project".to_owned()
            } else {
                dir.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            };
            ui.painter().text(
                egui::pos2(rect.left() + indent + 33.0, rect.center().y),
                Align2::LEFT_CENTER,
                name,
                FontId::proportional(crate::theme::SIZE_BODY),
                text_color,
            );
            // Aggregate problem badge for the folder's contents.
            let (errs, warns) = self.diag_counts(dir, true);
            Self::diag_dot(
                ui.painter(),
                egui::pos2(rect.right() - 9.0, rect.center().y),
                errs,
                warns,
            );

            if row.clicked() {
                let on_arrow = row
                    .interact_pointer_pos()
                    .is_some_and(|p| p.x < rect.left() + indent + 12.0);
                if on_arrow && !subdirs.is_empty() {
                    self.toggle_open(dir);
                } else {
                    // Single click only navigates the grid. Folders aren't
                    // inspectable assets, so it doesn't change the selection.
                    self.current_dir = dir.to_owned();
                }
            }
            if row.double_clicked() {
                self.toggle_open(dir);
            }
            if let Some(payload) = row.dnd_release_payload::<PathBuf>() {
                self.move_into(&payload, dir, response);
            }
            if row
                .dnd_hover_payload::<PathBuf>()
                .is_some_and(|p| *p != dir)
            {
                ui.painter().rect_stroke(
                    rect,
                    3.0,
                    egui::Stroke::new(1.5_f32, ui.visuals().selection.stroke.color),
                    egui::StrokeKind::Outside,
                );
            }
            let dir = dir.to_owned();
            row.context_menu(|ui| {
                crate::widgets::menu_frame(ui);
                if !is_root {
                    self.item_ops_menu(ui, &dir, response);
                }
                self.paste_button(ui, &dir, response);
                ui.separator();
                self.create_menu(ui, &dir, response);
                if is_root {
                    ui.separator();
                    Self::show_in_fm_button(ui, &dir, response);
                }
            });
        }

        if open {
            for sub in subdirs {
                self.tree_row(ui, &sub, depth + 1, response);
            }
        }
    }

    fn toggle_open(&mut self, dir: &Path) {
        if dir == self.root {
            return;
        }
        if !self.open_dirs.remove(dir) {
            self.open_dirs.insert(dir.to_owned());
        }
    }

    // ------------------------------------------------------------- grid

    fn grid_ui(
        &mut self,
        ui: &mut Ui,
        selected: Option<&Path>,
        response: &mut FileBrowserResponse,
    ) {
        let (mut dirs, mut files) = list_dir(&self.current_dir);
        let needle = self.search.to_lowercase();
        if !needle.is_empty() {
            let keep = |p: &PathBuf| p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().contains(&needle));
            dirs.retain(keep);
            files.retain(keep);
        }
        if dirs.is_empty() && files.is_empty() {
            let msg = if needle.is_empty() { "Empty folder" } else { "No matching files" };
            ui.label(RichText::new(msg).color(crate::theme::TEXT_DIM));
            return;
        }
        // Record the visible order (dirs then files) so shift-range can resolve
        // the span between the anchor and the shift-clicked entry.
        self.order = dirs.iter().chain(files.iter()).cloned().collect();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
            for path in dirs {
                self.tile(ui, &path, true, selected, response);
            }
            for path in files {
                self.tile(ui, &path, false, selected, response);
                if self.sprite_open.contains(&path)
                    && let Some(kids) = self.sprite_kids.peek(&path).cloned()
                {
                    for kid in kids.iter() {
                        self.sprite_tile(ui, &path, kid, response);
                    }
                }
            }
        });
    }

    fn tile(
        &mut self,
        ui: &mut Ui,
        path: &Path,
        is_dir: bool,
        selected: Option<&Path>,
        response: &mut FileBrowserResponse,
    ) {
        // Height tracks the slider-driven width; the name row stays a fixed
        // band at the bottom while the icon/preview area grows.
        let tile_dim = egui::vec2(self.tile_px, self.tile_px + 10.0);
        let s = self.tile_px / TILE.x; // icon scale vs. the default tile
        let (rect, tile) = ui.allocate_exact_size(tile_dim, Sense::click_and_drag());
        if !ui.is_rect_visible(rect) {
            return;
        }
        let is_selected = selected == Some(path) || self.multi.contains(path);
        let mut caret_clicked = false;
        if is_selected {
            ui.painter().rect(
                rect,
                crate::theme::R_TOOLBAR,
                crate::theme::accent_faint(),
                egui::Stroke::new(1.0_f32, crate::theme::with_alpha(crate::theme::accent(), 0.45)),
                egui::StrokeKind::Inside,
            );
        } else if tile.hovered() {
            ui.painter().rect_filled(rect, crate::theme::R_TOOLBAR, crate::theme::with_alpha(crate::theme::BG_BUTTON, 0.6));
        }
        // A pinged file pulses in the accent for a moment.
        if let Some((p, at, scroll)) = &mut self.ping
            && p.as_path() == path
        {
            let age = ui.input(|i| i.time) - *at;
            if *scroll {
                tile.scroll_to_me(Some(egui::Align::Center));
                *scroll = false;
            }
            if age < 1.6 {
                let a = (1.0 - age / 1.6) as f32;
                ui.painter().rect_stroke(rect, crate::theme::R_TOOLBAR, egui::Stroke::new(2.0_f32, crate::theme::with_alpha(crate::theme::accent(), a)), egui::StrokeKind::Inside);
                ui.ctx().request_repaint();
            }
        }

        let is_rust = !is_dir
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("rs"));
        // Centre the icon/preview in the area above the fixed name band.
        let name_top = rect.bottom() - 32.0;
        let icon_center = egui::pos2(rect.center().x, (rect.top() + name_top) * 0.5);
        let icon_color = crate::theme::TEXT_MUTED;
        if is_rust {
            // Monochrome Ferris silhouette for .rs files. ~30px tall, keep aspect.
            let tex = self.ferris_texture(ui.ctx());
            let [tw, th] = tex.size();
            let h = 30.0 * s;
            let w = h * tw as f32 / th as f32;
            let img_rect = egui::Rect::from_center_size(icon_center, egui::vec2(w, h));
            ui.painter().image(
                tex.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                icon_color,
            );
        } else if !is_dir
            && matches!(icon_kind(path), IconKind::Material)
            && let Some(tex) = self.material_thumb(ui.ctx(), path)
        {
            // Sphere preview with the material applied (CPU-shaded from its PBR
            // params). Fit it into the icon area like an image thumbnail.
            let [tw, th] = tex.size();
            let max = 48.0 * s;
            let scale = (max / tw as f32).min(max / th as f32).min(1.0);
            let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
            let img_rect = egui::Rect::from_center_size(icon_center, size);
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            ui.painter().image(tex.id(), img_rect, uv, egui::Color32::WHITE);
        } else if !is_dir
            && matches!(icon_kind(path), IconKind::Image)
            && let Some(tex) = self.image_thumb(ui.ctx(), path)
        {
            // Real image preview: fit the decoded thumbnail (true colour, no
            // selection tint) inside the tile's icon area, keeping aspect.
            let [tw, th] = tex.size();
            let max = 48.0 * s;
            let scale = (max / tw as f32).min(max / th as f32).min(1.0);
            let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
            let img_rect = egui::Rect::from_center_size(icon_center, size);
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            ui.painter().image(tex.id(), img_rect, uv, egui::Color32::WHITE);
            ui.painter().rect_stroke(
                img_rect,
                2.0,
                egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
                egui::StrokeKind::Outside,
            );
        } else {
            // Phosphor icon glyph for the file/folder type (also the fallback
            // for an image whose thumbnail isn't decoded yet / failed).
            match spec_icon(path, is_dir) {
                Some((icon, color)) => crate::icons::paint_with(
                    ui.painter(),
                    egui::Rect::from_center_size(icon_center, egui::vec2(34.0 * s, 34.0 * s)),
                    icon,
                    color,
                    0.9,
                ),
                None => {
                    ui.painter().text(
                        icon_center,
                        Align2::CENTER_CENTER,
                        phosphor_icon(icon_kind(path)),
                        FontId::proportional(30.0 * s),
                        icon_color,
                    );
                }
            }
        }

        // Sprite textures show how many sprites they hold, bottom-right of the icon.
        if !is_dir && matches!(icon_kind(path), IconKind::Image) {
            let now = ui.input(|i| i.time);
            if let Some(n) = self.sprite_count(now, path) {
                let galley = ui.painter().layout_no_wrap(n.to_string(), FontId::proportional(10.0), crate::theme::TEXT_STRONG);
                let size = galley.size() + egui::vec2(8.0, 2.0);
                let badge = egui::Rect::from_min_size(egui::pos2(icon_center.x + 26.0 * s - size.x, icon_center.y + 24.0 * s - size.y), size);
                ui.painter().rect_filled(badge, crate::theme::R_FIELD, crate::theme::accent());
                ui.painter().galley(badge.center() - galley.size() * 0.5, galley, crate::theme::TEXT_STRONG);
                ui.ctx().request_repaint_after(std::time::Duration::from_secs(3));
                // A texture packed into a sprite atlas says so, bottom-left of the icon.
                if let Some(probe) = ATLAS_PROBE.get()
                    && let Some(name) = self.atlas_names.get(now, path, || probe(path))
                {
                    let galley = ui.painter().layout_no_wrap(format!("in atlas {name}"), FontId::proportional(9.0), crate::theme::TEXT_SECONDARY);
                    let size = galley.size() + egui::vec2(8.0, 2.0);
                    let badge = egui::Rect::from_min_size(egui::pos2(icon_center.x - 26.0 * s, icon_center.y + 24.0 * s - size.y), size);
                    ui.painter().rect_filled(badge, crate::theme::R_FIELD, crate::theme::with_alpha(crate::theme::BG_BUTTON, 0.9));
                    ui.painter().galley(badge.center() - galley.size() * 0.5, galley, crate::theme::TEXT_SECONDARY);
                }
                // A Multiple sheet gets an expander (top-left) that lists its sprites after the tile.
                if !self.sprite_kids_of(now, path).is_empty() {
                    let open = self.sprite_open.contains(path);
                    let caret_rect = egui::Rect::from_min_size(rect.min + egui::vec2(2.0, 2.0), egui::vec2(18.0, 18.0));
                    let caret = ui.interact(caret_rect, tile.id.with("sprite-expander"), Sense::click());
                    if caret.hovered() {
                        ui.painter().rect_filled(caret_rect, crate::theme::R_FIELD, crate::theme::with_alpha(crate::theme::BG_BUTTON, 0.8));
                    }
                    crate::widgets::paint_caret(ui.painter(), caret_rect, open, crate::theme::TEXT_MUTED);
                    caret.clone().on_hover_text(if open { "Hide sprites" } else { "Show sprites" });
                    if caret.clicked() {
                        toggle_in(&mut self.sprite_open, path);
                        caret_clicked = true;
                    }
                }
            }
        }

        // LSP problem badge at the icon's top-right.
        let (errs, warns) = self.diag_counts(path, is_dir);
        Self::diag_dot(
            ui.painter(),
            egui::pos2(icon_center.x + 15.0 * s, icon_center.y - 13.0 * s),
            errs,
            warns,
        );

        let renaming_this = self.renaming.as_ref().is_some_and(|r| r.path == path);
        if renaming_this {
            let name_rect = egui::Rect::from_min_max(
                egui::pos2(rect.left() + 2.0, rect.bottom() - 26.0),
                egui::pos2(rect.right() - 2.0, rect.bottom() - 4.0),
            );
            self.rename_editor(ui, name_rect, response);
        } else {
            // Name under the icon, single line, cropped with `…`. Known asset
            // extensions are hidden (the icon conveys the type).
            let name = display_name(path, is_dir);
            let color = if is_selected { crate::theme::TEXT_STRONG } else { crate::theme::TEXT_SECONDARY };
            let mut job = egui::text::LayoutJob::default();
            job.append(
                &name,
                0.0,
                egui::TextFormat {
                    font_id: FontId::proportional(13.0),
                    color,
                    ..Default::default()
                },
            );
            job.wrap = egui::text::TextWrapping {
                max_width: self.tile_px - 8.0,
                max_rows: 1,
                break_anywhere: true,
                overflow_character: Some('…'),
            };
            job.halign = egui::Align::Center;
            let galley = ui.fonts_mut(|f| f.layout_job(job));
            ui.painter().galley(
                egui::pos2(rect.center().x, rect.bottom() - 32.0),
                galley,
                color,
            );
        }

        tile.dnd_set_drag_payload(path.to_owned());
        if tile.dragged() {
            // Also publish the dragged file's project-relative path into egui
            // memory (a String) so inspector drop-fields can read it across the
            // plugin egui boundary. Same pattern as the scene tree's
            // dragged-object key.
            if !is_dir {
                let rel = path
                    .strip_prefix(&self.root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .into_owned();
                ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::DRAG_FILE_KEY), rel));
            }
            // Floating "ghost" of the dragged tile (icon + name) following the
            // cursor, so the drag has visual feedback. Painted on a top layer,
            // mirroring the scene tree's drag ghost.
            if let Some(pos) = ui.ctx().pointer_interact_pos() {
                let painter = ui.ctx().layer_painter(egui::LayerId::new(
                    egui::Order::Tooltip,
                    egui::Id::new("zestengine-drag-ghost"),
                ));
                let glyph = if is_dir {
                    egui_phosphor::regular::FOLDER
                } else {
                    phosphor_icon(icon_kind(path))
                };
                let label = format!("{glyph} {}", display_name(path, is_dir));
                let font = egui::TextStyle::Body.resolve(ui.style());
                let galley = painter.layout_no_wrap(label, font, egui::Color32::WHITE);
                let pad = egui::vec2(8.0, 4.0);
                let origin = pos + egui::vec2(14.0, 6.0);
                let rect = egui::Rect::from_min_size(origin, galley.size() + pad * 2.0);
                painter.rect_filled(rect, crate::theme::R_ROW, crate::theme::BG_SEGMENT_ACTIVE);
                painter.rect_stroke(rect, crate::theme::R_ROW, egui::Stroke::new(1.0_f32, crate::theme::accent()), egui::StrokeKind::Inside);
                painter.galley(origin + pad, galley, crate::theme::TEXT_STRONG);
            }
        }
        if tile.double_clicked() {
            // Double-click opens: folders enter; files are activated.
            if is_dir {
                self.current_dir = path.to_owned();
                self.open_dirs.insert(path.to_owned());
            } else {
                response.activated = Some(path.to_owned());
            }
        } else if tile.clicked() && !renaming_this && !caret_clicked {
            // Selecting the texture itself drops a sprite pick.
            self.selected_sprite = None;
            ui.data_mut(|d| d.remove_temp::<(String, u64)>(egui::Id::new(SELECTED_SPRITE_KEY)));
            // Single-click selects a file *or folder* (so folders can be
            // deleted / copied / duplicated / renamed); double-click still enters
            // a folder. The selection shows in the Inspector.
            //
            // Ctrl/Cmd = toggle into the multi-selection; Shift = select the range
            // between the anchor and here (in visible grid order); plain = single.
            let mods = ui.input(|i| i.modifiers);
            let owned = path.to_owned();
            if mods.command {
                if !self.multi.remove(&owned) {
                    self.multi.insert(owned.clone());
                }
                self.last_click = Some(owned.clone());
            } else if mods.shift {
                if let Some(anchor) = &self.last_click {
                    let ia = self.order.iter().position(|p| p == anchor);
                    let ib = self.order.iter().position(|p| p == &owned);
                    if let (Some(ia), Some(ib)) = (ia, ib) {
                        let (lo, hi) = (ia.min(ib), ia.max(ib));
                        let span: Vec<PathBuf> = self.order[lo..=hi].to_vec();
                        self.multi.extend(span);
                    }
                }
                self.multi.insert(owned.clone());
            } else {
                self.multi.clear();
                self.multi.insert(owned.clone());
                self.last_click = Some(owned.clone());
            }
            response.clicked = Some(owned);
        }
        if is_dir {
            if let Some(payload) = tile.dnd_release_payload::<PathBuf>() {
                self.move_into(&payload, path, response);
            }
            if tile
                .dnd_hover_payload::<PathBuf>()
                .is_some_and(|p| *p != path)
            {
                ui.painter().rect_stroke(
                    rect,
                    6.0,
                    egui::Stroke::new(1.5_f32, ui.visuals().selection.stroke.color),
                    egui::StrokeKind::Outside,
                );
            }
        }
        let path = path.to_owned();
        let parent = self.current_dir.clone();
        tile.context_menu(|ui| {
            crate::widgets::menu_frame(ui);
            self.item_ops_menu(ui, &path, response);
            self.paste_button(ui, &parent, response);
            ui.separator();
            self.create_menu(ui, &parent, response);
        });
    }

    /// One sprite of an expanded sheet: click selects the texture and the
    /// sprite in its inspector; a drag carries the sprite's text reference.
    fn sprite_tile(&mut self, ui: &mut Ui, sheet: &Path, kid: &SpriteChild, response: &mut FileBrowserResponse) {
        let tile_dim = egui::vec2(self.tile_px, self.tile_px + 10.0);
        let s = self.tile_px / TILE.x;
        let (rect, tile) = ui.allocate_exact_size(tile_dim, Sense::click_and_drag());
        if !ui.is_rect_visible(rect) {
            return;
        }
        let picked = self.selected_sprite.as_ref().is_some_and(|(p, id)| p == sheet && *id == kid.id);
        if picked {
            ui.painter().rect(
                rect,
                crate::theme::R_TOOLBAR,
                crate::theme::accent_faint(),
                egui::Stroke::new(1.0_f32, crate::theme::with_alpha(crate::theme::accent(), 0.45)),
                egui::StrokeKind::Inside,
            );
        } else if tile.hovered() {
            ui.painter().rect_filled(rect, crate::theme::R_TOOLBAR, crate::theme::with_alpha(crate::theme::BG_BUTTON, 0.6));
        }
        let icon_center = egui::pos2(rect.center().x, (rect.top() + rect.bottom() - 32.0) * 0.5);
        crate::icons::paint_with(
            ui.painter(),
            egui::Rect::from_center_size(icon_center, egui::vec2(28.0 * s, 28.0 * s)),
            crate::icons::Icon::Grid,
            if picked { crate::theme::accent() } else { crate::theme::TEXT_MUTED },
            0.9,
        );
        let color = if picked { crate::theme::TEXT_STRONG } else { crate::theme::TEXT_SECONDARY };
        let mut job = egui::text::LayoutJob::default();
        job.append(&kid.name, 0.0, egui::TextFormat { font_id: FontId::proportional(13.0), color, ..Default::default() });
        job.wrap = egui::text::TextWrapping { max_width: self.tile_px - 8.0, max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
        job.halign = egui::Align::Center;
        let galley = ui.fonts_mut(|f| f.layout_job(job));
        ui.painter().galley(egui::pos2(rect.center().x, rect.bottom() - 32.0), galley, color);
        tile.clone().on_hover_text(format!("{} (sprite of {})", kid.name, sheet.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()));

        tile.dnd_set_drag_payload(SpriteDrag(kid.reference.clone()));
        if tile.dragged() {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(DRAG_SPRITE_KEY), kid.reference.clone()));
            if let Some(pos) = ui.ctx().pointer_interact_pos() {
                let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("zestengine-drag-ghost")));
                let font = egui::TextStyle::Body.resolve(ui.style());
                let galley = painter.layout_no_wrap(format!("{} {}", ph::IMAGE_SQUARE, kid.name), font, egui::Color32::WHITE);
                let pad = egui::vec2(8.0, 4.0);
                let origin = pos + egui::vec2(14.0, 6.0);
                let ghost = egui::Rect::from_min_size(origin, galley.size() + pad * 2.0);
                painter.rect_filled(ghost, crate::theme::R_ROW, crate::theme::BG_SEGMENT_ACTIVE);
                painter.rect_stroke(ghost, crate::theme::R_ROW, egui::Stroke::new(1.0_f32, crate::theme::accent()), egui::StrokeKind::Inside);
                painter.galley(origin + pad, galley, crate::theme::TEXT_STRONG);
            }
        }
        if tile.clicked() {
            self.multi.clear();
            self.multi.insert(sheet.to_owned());
            self.last_click = Some(sheet.to_owned());
            self.selected_sprite = Some((sheet.to_owned(), kid.id));
            ui.data_mut(|d| d.insert_temp(egui::Id::new(SELECTED_SPRITE_KEY), (sheet.to_string_lossy().into_owned(), kid.id)));
            response.clicked = Some(sheet.to_owned());
        }
    }

    // ----------------------------------------------------------- actions

    fn create_menu(&mut self, ui: &mut Ui, dir: &Path, response: &mut FileBrowserResponse) {
        if ui.button("New Material").clicked() {
            response.create_material_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Scene").clicked() {
            response.create_scene_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Shader").clicked() {
            response.create_shader_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Shader Graph").clicked() {
            response.create_shader_graph_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Water Waves").clicked() {
            response.create_waterwaves_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Audio Environment").clicked() {
            response.create_audio_in = Some((dir.to_owned(), false));
            ui.close();
        }
        if ui.button("New Audio Settings").clicked() {
            response.create_audio_in = Some((dir.to_owned(), true));
            ui.close();
        }
        if ui.button("New Post FX Profile").clicked() {
            response.create_postfx_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Physics Material").clicked() {
            response.create_physmat_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Sprite Atlas").clicked() {
            response.create_spriteatlas_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Sprite Shape Profile").clicked() {
            response.create_spriteshape_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Animator Controller").clicked() {
            response.create_anim_controller_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Behaviour Tree").clicked() {
            response.create_behaviour_tree_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Animation Clip").clicked() {
            response.create_anim_clip_in = Some(dir.to_owned());
            ui.close();
        }
        if ui.button("New Component (Rust)").clicked() {
            response.create_component = true;
            ui.close();
        }
        if ui.button("New Folder").clicked() {
            response.create_folder_in = Some(dir.to_owned());
            ui.close();
        }
        ui.separator();
        if ui
            .button("Import Megascans / Fab Asset...")
            .on_hover_text("Import a downloaded Megascans export, Fab download (folder or .zip) or inspect a MetaHuman .mhpkg")
            .clicked()
        {
            response.import_pack = Some(None);
            ui.close();
        }
    }

    /// Start an inline rename of `path`. The extension is hidden while
    /// editing ("hello.png" shows "hello", all selected) and re-appended on
    /// commit, so typing a new stem renames to "<typed>.png".
    fn begin_rename(&mut self, path: &Path) {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = if extension.is_empty() {
            name
        } else {
            // Strip exactly the final ".<ext>" (keeps multi-dot names intact).
            name.strip_suffix(&format!(".{extension}")).map(str::to_owned).unwrap_or(name)
        };
        self.renaming = Some(Renaming { path: path.to_owned(), text, extension, focus: true });
    }

    /// "Show in File Browser" (Explorer / Finder by platform) for `path`.
    fn show_in_fm_button(ui: &mut Ui, path: &Path, response: &mut FileBrowserResponse) {
        let label = if cfg!(windows) {
            "Show in Explorer"
        } else if cfg!(target_os = "macos") {
            "Reveal in Finder"
        } else {
            "Show in File Browser"
        };
        if ui.button(label).on_hover_text("Select this item in the system file manager").clicked() {
            response.show_in_file_manager = Some(path.to_owned());
            ui.close();
        }
    }

    /// Rename / Cut / Copy / Delete items for one file or folder.
    fn item_ops_menu(&mut self, ui: &mut Ui, path: &Path, response: &mut FileBrowserResponse) {
        if ui.button("Rename").clicked() {
            self.begin_rename(path);
            ui.close();
        }
        if ui.button("Cut").clicked() {
            self.clipboard = Some(Clipboard {
                path: path.to_owned(),
                cut: true,
            });
            ui.close();
        }
        if ui.button("Copy").clicked() {
            self.clipboard = Some(Clipboard {
                path: path.to_owned(),
                cut: false,
            });
            ui.close();
        }
        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .is_some_and(|e| matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tga" | "exr"));
        if is_image
            && ui
                .button("Set as Skybox")
                .on_hover_text("Use this image as the scene's equirectangular skybox")
                .clicked()
        {
            response.set_skybox = Some(path.to_owned());
            ui.close();
        }
        let is_prefab = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("prefab"));
        if is_prefab
            && ui
                .button(format!("{} Create Prefab Variant", ph::CUBE))
                .on_hover_text("A new prefab that is this one plus its own changes; edits to this prefab still reach it")
                .clicked()
        {
            response.create_prefab_variant = Some(path.to_owned());
            ui.close();
        }
        let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        if matches!(ext.as_deref(), Some("ase" | "aseprite" | "psd" | "psb"))
            && ui
                .button("Import as Sprites")
                .on_hover_text("Aseprite: frames to a sprite sheet, tags to animation clips, a sprite library. PSD: layers to sprites at their positions, a layer tree and a rig seed. Written next to the file.")
                .clicked()
        {
            response.import_layered = Some(path.to_owned());
            ui.close();
        }
        // A PSD's layer tree (or the PSD it came from, once imported): spawn it as objects.
        let layers_file = match ext.as_deref() {
            Some("layers") => Some(path.to_owned()),
            Some("psd" | "psb") if path.with_extension("layers").is_file() => Some(path.with_extension("layers")),
            _ => None,
        };
        if let Some(layers) = layers_file {
            if ui
                .button("Create Prefab from Layers")
                .on_hover_text("A prefab next to the file: sprite layers at their positions in PSD stack order, bones as Sprite Bones and Sprite Skins on the skinned layers (needs the import's .layers and .rig2d)")
                .clicked()
            {
                response.layers_prefab = Some((layers.clone(), false));
                ui.close();
            }
            if ui.button("Create Instance from Layers").on_hover_text("Create the prefab and place an instance of it in the scene").clicked() {
                response.layers_prefab = Some((layers, true));
                ui.close();
            }
        }
        if ext.as_deref() == Some("unitypackage")
            && ui
                .button("Import Unity package…")
                .on_hover_text("Extract the package's materials and textures and convert every material")
                .clicked()
        {
            response.import_unity_package = Some(path.to_owned());
            ui.close();
        }
        if (path.is_dir() || matches!(ext.as_deref(), Some("zip" | "mhpkg")))
            && ui
                .button("Import Megascans / Fab Asset...")
                .on_hover_text("Preview and import the models and textures in this folder or archive (MetaHuman .mhpkg packages are only inspected)")
                .clicked()
        {
            response.import_pack = Some(Some(path.to_owned()));
            ui.close();
        }
        ui.separator();
        Self::show_in_fm_button(ui, path, response);
        ui.separator();
        if ui
            .button(format!("{} Delete", ph::TRASH))
            .on_hover_text("Delete this file or folder")
            .clicked()
        {
            response.delete = Some(path.to_owned());
            ui.close();
        }
    }

    fn paste_button(&mut self, ui: &mut Ui, dir: &Path, response: &mut FileBrowserResponse) {
        let Some(clip) = &self.clipboard else { return };
        let label = format!(
            "Paste {:?}{}",
            clip.path.file_name().unwrap_or_default(),
            if clip.cut { " (move)" } else { "" }
        );
        if ui.button(label).clicked() {
            let (source, cut) = (clip.path.clone(), clip.cut);
            let dest = unique_in_dir(dir, &source);
            let result = if cut {
                std::fs::rename(&source, &dest)
            } else {
                copy_recursively(&source, &dest)
            };
            match result {
                Ok(()) => {
                    if cut {
                        response.moved.push((source, dest));
                        self.clipboard = None;
                    }
                }
                Err(e) => tracing::error!("pasting: {e}"),
            }
            ui.close();
        }
    }

    /// Drag-and-drop / cut-paste move into a folder.
    fn move_into(&mut self, source: &Path, dir: &Path, response: &mut FileBrowserResponse) {
        if source.parent() == Some(dir) || dir.starts_with(source) {
            return; // no-op or folder-into-itself
        }
        let dest = unique_in_dir(dir, source);
        match std::fs::rename(source, &dest) {
            Ok(()) => response.moved.push((source.to_owned(), dest)),
            Err(e) => tracing::error!("moving {}: {e}", source.display()),
        }
    }

    /// The inline rename editor, placed at `rect`.
    fn rename_editor(&mut self, ui: &mut Ui, rect: egui::Rect, response: &mut FileBrowserResponse) {
        let Some(renaming) = &mut self.renaming else {
            return;
        };
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
        let output = egui::TextEdit::singleline(&mut renaming.text)
            .font(FontId::proportional(11.0))
            .show(&mut child);
        let edit = output.response;
        if renaming.focus {
            edit.request_focus();
            // Pre-select the whole stem (the extension is hidden while
            // renaming), so typing replaces the name in one go.
            let mut state = output.state;
            state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                egui::text::CCursor::new(0),
                egui::text::CCursor::new(renaming.text.chars().count()),
            )));
            state.store(ui.ctx(), edit.id);
            renaming.focus = false;
        }
        let commit = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let cancel =
            ui.input(|i| i.key_pressed(egui::Key::Escape)) || (edit.lost_focus() && !commit);
        if commit {
            let path = renaming.path.clone();
            let stem = renaming.text.trim();
            // Re-append the hidden extension ("hello2" -> "hello2.png").
            let new_name = if renaming.extension.is_empty()
                || stem.is_empty()
                || stem.ends_with(&format!(".{}", renaming.extension))
            {
                stem.to_owned()
            } else {
                format!("{stem}.{}", renaming.extension)
            };
            if !new_name.is_empty() && !new_name.contains('/') {
                let dest = path.with_file_name(&new_name);
                if dest != path {
                    if dest.exists() {
                        tracing::error!("rename: {} already exists", dest.display());
                    } else {
                        match std::fs::rename(&path, &dest) {
                            Ok(()) => {
                                if self.current_dir == path {
                                    self.current_dir = dest.clone();
                                }
                                response.moved.push((path, dest));
                            }
                            Err(e) => tracing::error!("renaming: {e}"),
                        }
                    }
                }
            }
            self.renaming = None;
        } else if cancel {
            self.renaming = None;
        }
    }
}

/// Decode an image file and downscale it to a thumbnail-sized `ColorImage`.
/// Returns `None` for an unreadable / unsupported file (caller shows a glyph).
fn decode_thumb(path: &Path) -> Option<egui::ColorImage> {
    use image::ColorType;
    let img = image::open(path).ok().or_else(|| decode_exr_fallback(path))?;
    let small = img.thumbnail(THUMB_MAX, THUMB_MAX);
    // HDR sources (EXR / Radiance .hdr) decode to linear float with values that
    // routinely exceed 1.0. A raw float→u8 clamp (the old path) reads almost
    // black for a dim environment and blows the sky out to flat white, so the
    // preview looked broken. Tonemap (ACES) + encode to sRGB the same way the
    // skybox/standard shaders do, so the thumbnail matches the in-engine look.
    let hdr = matches!(
        small.color(),
        ColorType::Rgb32F | ColorType::Rgba32F
    ) || path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("exr") || e.eq_ignore_ascii_case("hdr"));
    let size = [small.width() as usize, small.height() as usize];
    if hdr {
        let f = small.to_rgba32f();
        let mut out = Vec::with_capacity((size[0] * size[1] * 4) as usize);
        for px in f.pixels() {
            for c in 0..3 {
                out.push(linear_to_srgb_u8(aces_tonemap(px[c])));
            }
            out.push(255);
        }
        Some(egui::ColorImage::from_rgba_unmultiplied(size, &out))
    } else {
        let mut rgba = small.to_rgba8();
        // Some LDR sources (e.g. icon TGAs) decode fully transparent; force opaque.
        for px in rgba.pixels_mut() {
            px[3] = 255;
        }
        Some(egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw()))
    }
}

/// CPU-render a sphere shaded with a `.material` file's PBR parameters, for the
/// file-browser tile preview. Parses the RON directly (no asset-crate dep) and
/// shades a unit sphere with a key light + a faux sky environment so base colour,
/// metalness, roughness and emission are all visible. Background is transparent.
fn render_material_sphere(path: &Path) -> Option<egui::ColorImage> {
    use glam::Vec3;
    let text = std::fs::read_to_string(path).ok()?;
    let base = parse_floats(&text, "base_color").unwrap_or_default();
    let albedo = Vec3::new(
        *base.first().unwrap_or(&0.8),
        *base.get(1).unwrap_or(&0.8),
        *base.get(2).unwrap_or(&0.8),
    );
    let metallic = parse_floats(&text, "metallic")
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let roughness = parse_floats(&text, "roughness")
        .and_then(|v| v.first().copied())
        .unwrap_or(0.5)
        .clamp(0.05, 1.0);
    let emiss = parse_floats(&text, "emission_color").unwrap_or_default();
    let emiss_col = Vec3::new(
        *emiss.first().unwrap_or(&0.0),
        *emiss.get(1).unwrap_or(&0.0),
        *emiss.get(2).unwrap_or(&0.0),
    );
    let emiss_int = parse_floats(&text, "emission_intensity")
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    const N: usize = 128;
    let f0 = Vec3::splat(0.04).lerp(albedo, metallic);
    let diffuse = albedo * (1.0 - metallic);
    let light_dir = Vec3::new(-0.4, 0.6, 0.7).normalize();
    let view = Vec3::Z;
    // Specular lobe sharpness from roughness (Blinn-Phong fit).
    let shininess = (2.0 / (roughness * roughness + 1e-3) - 2.0).max(1.0);
    // Faux sky: a vertical gradient sampled by the reflection ray, so metals show
    // a recognisable environment and roughness visibly blurs it (via the lobe).
    let sky = |d: Vec3| -> Vec3 {
        let t = (d.y * 0.5 + 0.5).clamp(0.0, 1.0);
        Vec3::new(0.10, 0.12, 0.16).lerp(Vec3::new(0.65, 0.72, 0.85), t)
    };
    let mut out = vec![0u8; N * N * 4];
    for y in 0..N {
        for x in 0..N {
            // Pixel → [-1,1], y up. Slight inset so the sphere doesn't touch edges.
            let px = (x as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / N as f32 * 2.0;
            let r = 1.06; // sphere radius in NDC (slightly < frame)
            let (sx, sy) = (px * r, py * r);
            let r2 = sx * sx + sy * sy;
            let o = (y * N + x) * 4;
            if r2 > 1.0 {
                continue; // transparent background
            }
            let nz = (1.0 - r2).max(0.0).sqrt();
            let n = Vec3::new(sx, sy, nz);
            let n_dot_l = n.dot(light_dir).max(0.0);
            let n_dot_v = n.dot(view).max(0.0);
            let half = (light_dir + view).normalize();
            let n_dot_h = n.dot(half).max(0.0);
            // Key light: Lambert diffuse + Blinn specular tinted by F0.
            let spec = n_dot_h.powf(shininess) * (shininess + 8.0) / 8.0;
            let mut col = diffuse * n_dot_l + f0 * spec * n_dot_l;
            // Hemisphere ambient fill so the dark side isn't pure black.
            col += diffuse * (0.18 + 0.12 * (n.y * 0.5 + 0.5));
            // Environment reflection (stronger for metals / smoother surfaces).
            let refl = (view * (2.0 * n_dot_v) - n).normalize();
            let fres = f0 + (Vec3::ONE - f0) * (1.0 - n_dot_v).powf(5.0);
            col += fres * sky(refl) * (1.0 - roughness * 0.85);
            // Emission.
            col += emiss_col * emiss_int;
            // Soft 1px silhouette coverage for antialiasing.
            let edge = ((1.0 - r2.sqrt()) * (N as f32 * 0.5)).clamp(0.0, 1.0);
            out[o] = linear_to_srgb_u8(aces_tonemap(col.x));
            out[o + 1] = linear_to_srgb_u8(aces_tonemap(col.y));
            out[o + 2] = linear_to_srgb_u8(aces_tonemap(col.z));
            out[o + 3] = (edge * 255.0) as u8;
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([N, N], &out))
}

/// Pull the float list out of a RON `field: (a, b, c)` or scalar `field: x` in a
/// `.material` file. Matches `field:` exactly (so `metallic` won't catch
/// `metallic_invert`). Returns `None` if the field is absent / unparseable.
fn parse_floats(text: &str, field: &str) -> Option<Vec<f32>> {
    let key = format!("{field}:");
    let start = text.find(&key)? + key.len();
    let rest = text[start..].trim_start();
    let span = if let Some(stripped) = rest.strip_prefix('(') {
        // Tuple: read up to the matching ')'.
        let end = stripped.find(')')?;
        &stripped[..end]
    } else {
        // Scalar: up to the next comma / newline / close paren.
        let end = rest
            .find([',', '\n', ')'])
            .unwrap_or(rest.len());
        &rest[..end]
    };
    let vals: Vec<f32> = span
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    (!vals.is_empty()).then_some(vals)
}

/// Narkowicz ACES filmic tonemap (scalar), matching the engine's HDR rolloff so
/// EXR/HDR thumbnails preview at the same exposure they render at.
fn aces_tonemap(x: f32) -> f32 {
    let x = x.max(0.0);
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// Linear [0,1] → sRGB-encoded u8.
fn linear_to_srgb_u8(l: f32) -> u8 {
    let s = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// EXR fallback for layouts the Rust decoder can't read (DWAA/DWAB compression
/// or single-channel/non-RGB): transcode to a temp lossless ZIP EXR via
/// `oiiotool` if present, then decode (stays float — no PNG). Mirrors the asset
/// loader so previews match what the material pipeline can load.
fn decode_exr_fallback(path: &Path) -> Option<image::DynamicImage> {
    if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")) {
        return None;
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("zestengine_thumb_exr_{:016x}.exr", hasher.finish()));
    let ok = std::process::Command::new("oiiotool")
        .arg(path)
        .args(["--ch", "0,0,0", "--compression", "zip", "-o"])
        .arg(&tmp)
        .status()
        .ok()
        .is_some_and(|s| s.success());
    let img = if ok { image::open(&tmp).ok() } else { None };
    let _ = std::fs::remove_file(&tmp);
    img
}

/// Visible (non-hidden, non-target) sorted subdirectories.
fn subdirs_of(dir: &Path) -> Vec<PathBuf> {
    let (dirs, _) = list_dir(dir);
    dirs
}

/// The breadcrumb trail from the project root to `dir`: ("Project", root), then one (name, path) per folder.
/// A `dir` outside the root gives just the root crumb.
fn breadcrumbs(root: &Path, dir: &Path) -> Vec<(String, PathBuf)> {
    let mut crumbs = vec![("Project".to_owned(), root.to_owned())];
    if let Ok(rel) = dir.strip_prefix(root) {
        let mut path = root.to_owned();
        for part in rel.components() {
            path.push(part);
            crumbs.push((part.as_os_str().to_string_lossy().into_owned(), path.clone()));
        }
    }
    crumbs
}

/// (folders, files) of a directory, hidden/target filtered, sorted by name.
fn list_dir(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (dirs, files);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if !zestengine_core::asset_ref::is_sidecar(&path) {
            // `.meta` sidecars travel with their asset, out of sight.
            files.push(path);
        }
    }
    let by_name = |a: &PathBuf, b: &PathBuf| {
        a.file_name()
            .map(|n| n.to_ascii_lowercase())
            .cmp(&b.file_name().map(|n| n.to_ascii_lowercase()))
    };
    dirs.sort_by(by_name);
    files.sort_by(by_name);
    (dirs, files)
}

/// Destination for `source` inside `dir`, de-duplicated (`name(1).ext`, …).
fn unique_in_dir(dir: &Path, source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    let ext = source.extension().map(|e| e.to_string_lossy().into_owned());
    let make = |n: u32| {
        // `name(N)` to match object duplication. The stem is kept whole (not
        // stripped) so importing a file that legitimately ends in `(1)` keeps it.
        let name = if n == 0 {
            stem.clone()
        } else {
            format!("{stem}({n})")
        };
        match &ext {
            Some(ext) => dir.join(format!("{name}.{ext}")),
            None => dir.join(name),
        }
    };
    (0..1000)
        .map(make)
        .find(|p| !p.exists())
        .unwrap_or_else(|| make(0))
}

/// Copy a file or folder. Copies are new assets: their `.meta` sidecars get
/// fresh ids (importer settings kept), so slots keep pointing at the original.
fn copy_recursively(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)?.flatten() {
            copy_recursively(&entry.path(), &dst.join(entry.file_name()))?;
        }
    } else if zestengine_core::asset_ref::is_sidecar(src) {
        let text = std::fs::read_to_string(src)?;
        std::fs::write(dst, zestengine_core::asset_ref::with_fresh_id(&text))?;
    } else {
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

/// Extensions whose type is conveyed by the icon, so the name hides them.
const KNOWN_EXTS: &[&str] = &[
    "rs", "gltf", "glb", "vrm", "fbx", "material", "scene", "zestengine", "lightmap", "lightdata", "png",
    "jpg", "jpeg", "tga", "bmp", "exr", "vert", "frag", "hlsl", "glsl", "slang", "spv", "postfx", "physmat", "toml", "ron",
    "json", "md", "anim", "animctrl", "seq", "audiofx", "audiosettings", "waterwaves",
];

fn lower_ext(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(str::to_lowercase)
}

/// The kind of icon drawn for a file, by extension.
#[derive(Clone, Copy)]
enum IconKind {
    Scene,
    Material,
    Model,
    Image,
    Shader,
    PostFx,
    PhysMat,
    AnimController,
    AnimClip,
    Sequencer,
    Lightmap,
    Lightdata,
    Config,
    Markdown,
    ZestEngine,
    Doc,
}

fn icon_kind(path: &Path) -> IconKind {
    match lower_ext(path).as_deref() {
        Some("scene") => IconKind::Scene,
        Some("material") => IconKind::Material,
        Some("gltf" | "glb" | "vrm" | "fbx") => IconKind::Model,
        Some("png" | "jpg" | "jpeg" | "tga" | "bmp" | "exr") => IconKind::Image,
        Some("vert" | "frag" | "hlsl" | "glsl" | "slang" | "spv") => IconKind::Shader,
        Some("postfx") => IconKind::PostFx,
        Some("physmat") => IconKind::PhysMat,
        Some("animctrl") => IconKind::AnimController,
        Some("anim") => IconKind::AnimClip,
        Some("seq") => IconKind::Sequencer,
        Some("lightmap") => IconKind::Lightmap,
        Some("lightdata") => IconKind::Lightdata,
        Some("toml" | "ron" | "json" | "wasm" | "audiofx" | "audiosettings" | "waterwaves") => IconKind::Config,
        Some("md") => IconKind::Markdown,
        Some("zestengine") => IconKind::ZestEngine,
        _ => IconKind::Doc,
    }
}

/// Display label: directory name, or the file stem with a known extension
/// hidden (the icon conveys the type), else the full name.
fn display_name(path: &Path, is_dir: bool) -> String {
    let hide = !is_dir
        && lower_ext(path).is_some_and(|e| KNOWN_EXTS.contains(&e.as_str()));
    let part = if hide { path.file_stem() } else { path.file_name() };
    part.map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Spec icon + colour for folders and the file types the design covers
/// (spec §7.1); other types fall back to a Phosphor glyph.
fn spec_icon(path: &Path, is_dir: bool) -> Option<(crate::icons::Icon, egui::Color32)> {
    use crate::icons::Icon;
    use crate::theme;
    if is_dir {
        return Some((Icon::Folder, theme::TYPE_GROUP));
    }
    match icon_kind(path) {
        IconKind::Scene | IconKind::Model => Some((Icon::Model, theme::TYPE_MESH)),
        IconKind::ZestEngine => Some((Icon::Gear, theme::accent())),
        IconKind::Config => Some((Icon::File, theme::TYPE_LIGHT)),
        IconKind::Markdown => Some((Icon::FileDoc, theme::TEXT_SECONDARY)),
        IconKind::Doc => Some((Icon::File, theme::TEXT_MUTED)),
        _ => None,
    }
}

/// Phosphor icon glyph for a file (by type), for asset slots.
pub(crate) fn file_glyph(path: &Path) -> &'static str {
    phosphor_icon(icon_kind(path))
}

/// Phosphor icon glyph for a file kind.
fn phosphor_icon(kind: IconKind) -> &'static str {
    match kind {
        IconKind::Scene => ph::GLOBE_HEMISPHERE_WEST,
        IconKind::Material => ph::SPHERE,
        IconKind::Model => ph::CUBE,
        IconKind::Image => ph::IMAGE,
        IconKind::Shader => ph::SPARKLE,
        IconKind::PostFx => ph::APERTURE,
        IconKind::PhysMat => ph::BASKETBALL,
        IconKind::AnimController => ph::FLOW_ARROW,
        IconKind::AnimClip => ph::FILM_STRIP,
        IconKind::Sequencer => ph::FILM_STRIP,
        IconKind::Lightmap => ph::MAP_TRIFOLD,
        IconKind::Lightdata => ph::DATABASE,
        IconKind::Config => ph::GEAR_SIX,
        IconKind::Markdown => ph::FILE_TEXT,
        IconKind::ZestEngine => ph::ORANGE_SLICE,
        IconKind::Doc => ph::FILE,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breadcrumbs_list_every_folder_from_the_root() {
        let root = Path::new("/p");
        let names = |d: &str| breadcrumbs(root, Path::new(d)).into_iter().map(|(n, p)| (n, p.display().to_string())).collect::<Vec<_>>();
        assert_eq!(names("/p"), vec![("Project".to_owned(), "/p".to_owned())]);
        assert_eq!(
            names("/p/models/avatars"),
            vec![
                ("Project".to_owned(), "/p".to_owned()),
                ("models".to_owned(), "/p/models".to_owned()),
                ("avatars".to_owned(), "/p/models/avatars".to_owned()),
            ]
        );
        assert_eq!(names("/elsewhere").len(), 1, "outside the root: only the root crumb");
    }

    #[test]
    fn probed_values_are_cached_until_the_ttl_passes() {
        let mut cache: Probed<u32> = Probed::new(3.0);
        let calls = std::cell::Cell::new(0);
        let p = Path::new("/a.png");
        let mut read = |now: f64| {
            cache.get(now, p, || {
                calls.set(calls.get() + 1);
                calls.get()
            })
        };
        assert_eq!(read(0.0), 1);
        assert_eq!(read(2.9), 1, "inside the TTL: no second probe");
        assert_eq!(read(3.1), 2, "after the TTL: read again");
        assert_eq!(calls.get(), 2);
        assert_eq!(cache.peek(p), Some(&2));
        assert_eq!(cache.peek(Path::new("/b.png")), None);
    }

    #[test]
    fn expander_toggles() {
        let mut open = HashSet::new();
        let p = Path::new("/sheet.png");
        assert!(toggle_in(&mut open, p));
        assert!(open.contains(p));
        assert!(!toggle_in(&mut open, p));
        assert!(open.is_empty());
    }

    #[test]
    fn kids_come_from_the_installed_probe_with_a_project_relative_path() {
        // No probe installed in this test binary's other tests: a sheet has no children.
        let mut fb = FileBrowser::new(PathBuf::from("/proj"));
        assert!(fb.sprite_kids_of(0.0, Path::new("/proj/a.png")).is_empty());
        fn probe(_abs: &Path, rel: &str) -> Vec<SpriteChild> {
            vec![SpriteChild { id: 7, name: "idle_0".into(), reference: format!("{rel}#0000000000000007") }]
        }
        set_sprite_children_probe(probe);
        let kids = fb.sprite_kids_of(10.0, Path::new("/proj/Sprites/hero.png"));
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].reference, "Sprites/hero.png#0000000000000007");
    }
}
