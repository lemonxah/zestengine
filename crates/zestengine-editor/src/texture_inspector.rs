//! Texture importer inspector and material texture-slot validation (TEX-3).
//!
//! Everything that can be decided without a GUI lives in plain functions at
//! the top (which sections a kind shows, preview channel views, the info
//! line, slot warnings and fixes) so it is unit tested. The egui code below
//! them only lays those results out. Apply / Revert reuse the Model Import
//! session (`FileMeta` in `editor_app.rs`): edits go to a draft, "dirty" is
//! draft != saved, Apply writes the `.meta` and reimports.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use zestengine_assets::{
    AlphaSource, Channel, Compression, FilterMode, FormatOverride, MipFilter, NormalFilter, NpotScale, Platform, PlatformOverride,
    ResizeFilter, SlotUse, TargetFormat, TextureImport, TextureKind, TexturePreview, WarnLevel, WrapMode,
};
use zestengine_core::MaterialTexturePaths;
use crate::{SlotFix, SlotWarning, icons::Icon, theme, widgets};

// ---------------------------------------------------------------------------
// Pure logic
// ---------------------------------------------------------------------------

/// Which setting groups the inspector shows for a texture type (research
/// phase1b §1a / §6d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindSections {
    pub srgb: bool,
    pub alpha: bool,
    pub single_channel: bool,
    pub normal: bool,
    pub npot: bool,
    pub mips: bool,
}

pub fn kind_sections(kind: TextureKind) -> KindSections {
    let none = KindSections { srgb: false, alpha: false, single_channel: false, normal: false, npot: false, mips: false };
    match kind {
        TextureKind::Default => KindSections { srgb: true, alpha: true, npot: true, mips: true, ..none },
        TextureKind::NormalMap => KindSections { normal: true, npot: true, mips: true, ..none },
        TextureKind::Sprite => KindSections { srgb: true, alpha: true, mips: true, ..none },
        TextureKind::Cursor => KindSections { alpha: true, ..none },
        TextureKind::Lightmap => KindSections { npot: true, mips: true, ..none },
        TextureKind::SingleChannel => KindSections { single_channel: true, npot: true, mips: true, ..none },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewChannel {
    Rgb,
    R,
    G,
    B,
    A,
}

impl PreviewChannel {
    fn label(self) -> &'static str {
        match self {
            Self::Rgb => "RGB",
            Self::R => "R",
            Self::G => "G",
            Self::B => "B",
            Self::A => "A",
        }
    }
}

/// Channel buttons the preview offers: none for normal maps; A only when the
/// processed image has alpha.
pub fn available_channels(kind: TextureKind, has_alpha: bool) -> Vec<PreviewChannel> {
    if kind == TextureKind::NormalMap {
        return vec![PreviewChannel::Rgb];
    }
    let mut v = vec![PreviewChannel::Rgb, PreviewChannel::R, PreviewChannel::G, PreviewChannel::B];
    if has_alpha {
        v.push(PreviewChannel::A);
    }
    v
}

/// RGBA8 to display for `channel`: RGB shows colour (opaque), R/G/B/A show
/// that channel as grey.
pub fn channel_view(px: &[u8], channel: PreviewChannel) -> Vec<u8> {
    let idx = match channel {
        PreviewChannel::Rgb => None,
        PreviewChannel::R => Some(0),
        PreviewChannel::G => Some(1),
        PreviewChannel::B => Some(2),
        PreviewChannel::A => Some(3),
    };
    px.as_chunks::<4>().0.iter()
        .flat_map(|p| match idx {
            None => [p[0], p[1], p[2], 255],
            Some(i) => [p[i], p[i], p[i], 255],
        })
        .collect()
}

pub fn human_bytes(b: u64) -> String {
    const KB: f64 = 1024.0;
    let b = b as f64;
    if b >= KB * KB {
        format!("{:.1} MB", b / (KB * KB))
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{b:.0} B")
    }
}

/// `2048×2048 (NPOT) · BC7 sRGB · 5.3 MB · 12 mips`; a resize shows as
/// `4096×4096 → 2048×2048`.
pub fn info_line(p: &TexturePreview) -> String {
    let (sw, sh) = (p.source.width, p.source.height);
    let npot = if sw.is_power_of_two() && sh.is_power_of_two() { "" } else { " (NPOT)" };
    let size = if (p.width, p.height) == (sw, sh) || p.levels.is_empty() {
        format!("{sw}×{sh}{npot}")
    } else {
        format!("{sw}×{sh} → {}×{}{npot}", p.width, p.height)
    };
    let space = if p.source.hdr { "HDR" } else if p.srgb { "sRGB" } else { "linear" };
    format!("{size} · {} {space} · {} · {} mip{}", p.target.label(), human_bytes(p.vram_bytes), p.mip_count, if p.mip_count == 1 { "" } else { "s" })
}

/// `VRAM 21.3 MB → 5.3 MB (-75%)` against the legacy import.
pub fn vram_delta_line(before: u64, after: u64) -> String {
    if before == 0 {
        return format!("VRAM {}", human_bytes(after));
    }
    let pct = (after as f64 / before as f64 - 1.0) * 100.0;
    format!("VRAM {} → {} ({pct:+.0}%)", human_bytes(before), human_bytes(after))
}

pub use zestengine_engine::scene::perf::slot_paths;

/// (owner, textures) of every `.material` file under `root` (hidden folders,
/// `target` and `node_modules` skipped; bounded so a huge tree can't stall).
pub fn scan_material_files(root: &Path) -> Vec<(String, MaterialTexturePaths)> {
    fn walk(dir: &Path, root: &Path, depth: usize, budget: &mut usize, out: &mut Vec<(String, MaterialTexturePaths)>) {
        if depth > 8 || *budget == 0 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if !name.starts_with('.') && name != "target" && name != "node_modules" {
                    walk(&path, root, depth + 1, budget, out);
                }
            } else if path.extension().is_some_and(|e| e == "material") {
                *budget = budget.saturating_sub(1);
                if let Ok(m) = zestengine_assets::load_material_file(&path) {
                    let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                    out.push((rel, zestengine_engine::scene::tex_paths_from_file(&m.textures)));
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut budget = 5000;
    walk(root, root, 0, &mut budget, &mut out);
    out
}

/// absolute texture path -> every (owner, slot) that uses it.
pub fn build_usage(root: &Path, materials: &[(String, MaterialTexturePaths)]) -> HashMap<PathBuf, Vec<SlotUse>> {
    let mut map: HashMap<PathBuf, Vec<SlotUse>> = HashMap::new();
    for (owner, textures) in materials {
        for (slot, rel) in slot_paths(textures) {
            let abs = zestengine_core::asset_ref::resolve_rel_path(root, &rel);
            let uses = map.entry(abs).or_default();
            let u = SlotUse { owner: owner.clone(), slot };
            if !uses.contains(&u) {
                uses.push(u);
            }
        }
    }
    map
}

fn slot_name(slot: usize) -> &'static str {
    zestengine_assets::slot_expectation(slot).map_or("slot", |(_, n)| n)
}

/// Warnings for every slot of `textures` whose (configured) texture settings
/// don't suit it. `meta_of` yields a texture's configured settings,
/// `usage` is [`build_usage`], `owner` names the material being shown so its
/// own entry isn't a "conflict" with itself. `exists` is the file-exists test
/// for naming a duplicate.
pub fn slot_warnings(
    root: &Path,
    textures: &MaterialTexturePaths,
    owner: &str,
    usage: &HashMap<PathBuf, Vec<SlotUse>>,
    ignored: &HashSet<(PathBuf, usize)>,
    meta_of: &mut dyn FnMut(&Path) -> Option<TextureImport>,
    exists: &dyn Fn(&Path) -> bool,
) -> Vec<SlotWarning> {
    let mut out = Vec::new();
    for (slot, rel) in slot_paths(textures) {
        let Some((expect, name)) = zestengine_assets::slot_expectation(slot) else { continue };
        let abs = zestengine_core::asset_ref::resolve_rel_path(root, &rel);
        if ignored.contains(&(abs.clone(), slot)) {
            continue;
        }
        let import = meta_of(&abs);
        let issues = zestengine_assets::slot_issues(expect, import.as_ref());
        // A rendering mismatch outranks the VRAM hint.
        let Some(issue) = issues.iter().find(|i| !i.info).or(issues.first()) else { continue };
        let others: Vec<SlotUse> = usage
            .get(&abs)
            .map(|v| v.iter().filter(|u| !(u.owner == owner && u.slot == slot)).cloned().collect())
            .unwrap_or_default();
        let conflicts = zestengine_assets::conflicting_uses(expect, &others)
            .into_iter()
            .map(|u| format!("{} {}", u.owner, slot_name(u.slot)))
            .collect();
        let dup = zestengine_assets::duplicate_path(&abs, expect, exists);
        out.push(SlotWarning {
            slot,
            text: format!("{} ({name} slot)", issue.text),
            info: issue.info,
            fix_label: if issue.info { "Use single channel".into() } else { "Auto-fix".into() },
            conflicts,
            duplicate_name: dup.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        });
    }
    out
}

/// What the engine must do after a fix.
#[derive(Debug, PartialEq, Eq)]
pub enum FixOutcome {
    /// Settings written: reimport this texture.
    Reimport(PathBuf),
    /// A configured copy was made: assign `new_abs` to the slot.
    Duplicated { slot: usize, new_abs: PathBuf },
    /// Hide this slot's warning for the session.
    Ignored(PathBuf, usize),
}

/// Carry out the user's choice for `slot` of a material using `textures`.
pub fn apply_slot_fix(root: &Path, textures: &MaterialTexturePaths, slot: usize, fix: &SlotFix) -> Result<FixOutcome> {
    let (expect, _) = zestengine_assets::slot_expectation(slot).context("this slot has no expectation")?;
    let rel = slot_paths(textures).into_iter().find(|(s, _)| *s == slot).map(|(_, p)| p).context("slot has no texture")?;
    let abs = zestengine_core::asset_ref::resolve_rel_path(root, &rel);
    match fix {
        SlotFix::Ignore => Ok(FixOutcome::Ignored(abs, slot)),
        SlotFix::AutoFix | SlotFix::FixAnyway => {
            let mut meta = zestengine_assets::load_or_create_asset_meta(&abs)?;
            let mut t = meta.importer.as_texture().cloned().unwrap_or_default();
            t.configured = true;
            zestengine_assets::apply_expectation(&mut t, expect);
            meta.importer = zestengine_assets::ImporterSettings::Texture(t);
            zestengine_assets::save_asset_meta(&abs, &meta)?;
            Ok(FixOutcome::Reimport(abs))
        }
        SlotFix::Duplicate => {
            let dup = zestengine_assets::duplicate_path(&abs, expect, |p| p.exists());
            std::fs::copy(&abs, &dup).with_context(|| format!("copying {}", abs.display()))?;
            // A new asset: fresh id, settings for the slot (never the source's).
            let mut meta = zestengine_assets::AssetMeta::new_for(&dup);
            meta.importer = zestengine_assets::ImporterSettings::Texture(zestengine_assets::import_for_expectation(expect));
            zestengine_assets::save_asset_meta(&dup, &meta)?;
            Ok(FixOutcome::Duplicated { slot, new_abs: dup })
        }
    }
}

/// Write an Ignore choice into the `.material` file at `material` so it
/// survives restarts: (slot, project-relative texture). No-op if present.
pub fn persist_slot_ignore(material: &Path, slot: usize, texture_rel: &Path) -> Result<()> {
    let mut file = zestengine_assets::load_material_file(material)?;
    let entry = (slot, texture_rel.to_path_buf());
    if file.textures.ignored_warnings.contains(&entry) {
        return Ok(());
    }
    file.textures.ignored_warnings.push(entry);
    zestengine_assets::save_material_file(material, &file)
}

/// The project-wide slot usage, built off the UI thread.
type UsageMap = HashMap<PathBuf, Vec<SlotUse>>;

/// Caches behind the material inspector's slot warnings: texture settings
/// (re-read at most once a second), the persisted Ignore choices of the shown
/// material file, and the project's slot usage. The usage scan walks every
/// `.material` file, so it runs on a worker thread and the result comes back
/// over a channel (like the shader rescan); until the first result arrives,
/// or while a refresh runs, the previous usage (or none) is used, so a
/// conflict menu can appear a moment after the warning does.
#[derive(Default)]
pub struct SlotValidator {
    metas: HashMap<PathBuf, (Instant, Option<TextureImport>)>,
    usage: Option<(Instant, UsageMap)>,
    scan: Option<std::sync::mpsc::Receiver<UsageMap>>,
    persisted: HashMap<PathBuf, (Instant, Vec<(usize, PathBuf)>)>,
    ignored: HashSet<(PathBuf, usize)>,
}

impl SlotValidator {
    const META_TTL: Duration = Duration::from_secs(1);
    const USAGE_TTL: Duration = Duration::from_secs(5);

    pub fn invalidate(&mut self) {
        self.metas.clear();
        self.persisted.clear();
        // Keep the old usage until the rescan lands; just make it stale.
        if let Some((t, _)) = &mut self.usage {
            *t = Instant::now() - Self::USAGE_TTL * 2;
        }
    }

    pub fn ignore(&mut self, abs: PathBuf, slot: usize) {
        self.ignored.insert((abs, slot));
    }

    /// True while a usage scan is running: the caller should repaint soon.
    pub fn scanning(&self) -> bool {
        self.scan.is_some()
    }

    fn poll_scan(&mut self) {
        if let Some(rx) = &self.scan {
            match rx.try_recv() {
                Ok(map) => {
                    self.usage = Some((Instant::now(), map));
                    self.scan = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.scan = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Start a rescan if the usage is stale and none is running. The open
    /// scene's materials are in memory only, so they are captured here and
    /// merged with the `.material` files on the worker.
    fn start_scan(&mut self, root: &Path, scene_materials: Vec<(String, MaterialTexturePaths)>) {
        if self.scan.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let root = root.to_path_buf();
        let spawned = std::thread::Builder::new().name("texture-usage-scan".into()).spawn(move || {
            let mut all = scan_material_files(&root);
            all.extend(scene_materials);
            let _ = tx.send(build_usage(&root, &all));
        });
        if spawned.is_ok() {
            self.scan = Some(rx);
        }
    }

    /// Warnings for the material on screen. `scene_materials` are the open
    /// scene's materials (owner, textures) merged with the `.material` files.
    /// `file` is the material's `.material` file, whose saved Ignore choices
    /// hide their warnings.
    pub fn warnings(
        &mut self,
        root: &Path,
        textures: &MaterialTexturePaths,
        owner: &str,
        file: Option<&Path>,
        scene_materials: impl FnOnce() -> Vec<(String, MaterialTexturePaths)>,
    ) -> Vec<SlotWarning> {
        if slot_paths(textures).is_empty() {
            return Vec::new();
        }
        self.poll_scan();
        if self.scan.is_none() && self.usage.as_ref().is_none_or(|(t, _)| t.elapsed() > Self::USAGE_TTL) {
            self.start_scan(root, scene_materials());
        }
        let empty = UsageMap::new();
        let usage = self.usage.as_ref().map_or(&empty, |(_, u)| u);
        // Session ignores plus those saved in the material file.
        let mut ignored = self.ignored.clone();
        if let Some(file) = file {
            let stale = self.persisted.get(file).is_none_or(|(t, _)| t.elapsed() > Self::META_TTL);
            if stale {
                let list = zestengine_assets::load_material_file(file).map(|m| m.textures.ignored_warnings).unwrap_or_default();
                self.persisted.insert(file.to_path_buf(), (Instant::now(), list));
            }
            for (slot, rel) in &self.persisted[file].1 {
                ignored.insert((zestengine_core::asset_ref::resolve_rel_path(root, rel), *slot));
            }
        }
        let metas = &mut self.metas;
        let mut meta_of = |abs: &Path| -> Option<TextureImport> {
            if let Some((t, m)) = metas.get(abs)
                && t.elapsed() < Self::META_TTL
            {
                return m.clone();
            }
            let m = match zestengine_assets::load_asset_meta(abs) {
                Ok(Some(meta)) => meta.importer.as_texture().cloned(),
                _ => None,
            };
            metas.insert(abs.to_path_buf(), (Instant::now(), m.clone()));
            m
        };
        slot_warnings(root, textures, owner, usage, &ignored, &mut meta_of, &|p| p.exists())
    }
}

// ---------------------------------------------------------------------------
// egui
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Default,
    Desktop,
    Quest,
}

impl Tab {
    fn platform(self) -> Platform {
        match self {
            Self::Quest => Platform::Quest,
            _ => Platform::Desktop,
        }
    }
}

/// Result of one frame of the texture inspector.
#[derive(Default)]
pub struct TextureUiOut {
    pub changed: bool,
    pub apply: bool,
    pub revert: bool,
}

type PreviewKey = (PathBuf, TextureImport, Platform);
type PreviewResult = (PreviewKey, Result<(TexturePreview, u64), String>);

struct Loaded {
    key: PreviewKey,
    preview: TexturePreview,
    /// VRAM the legacy import of the same file takes on this platform.
    legacy_vram: u64,
}

/// UI state of the texture inspector (tab, preview channel / mip, the
/// background preview job).
pub struct TextureUi {
    tab: Tab,
    channel: PreviewChannel,
    mip: usize,
    wanted: Option<(PreviewKey, Instant)>,
    job: Option<(PreviewKey, Receiver<PreviewResult>)>,
    loaded: Option<Loaded>,
    error: Option<String>,
    view: Option<(u64, usize, PreviewChannel, egui::TextureHandle)>,
    serial: u64,
    /// The Sprite section (kind Sprite only).
    sprite: crate::sprite_inspector::SpriteUi,
}

impl Default for TextureUi {
    fn default() -> Self {
        Self { tab: Tab::Default, channel: PreviewChannel::Rgb, mip: 0, wanted: None, job: None, loaded: None, error: None, view: None, serial: 0, sprite: Default::default() }
    }
}

const MAX_SIZES: [u32; 10] = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

fn size_label(n: u32) -> String {
    if n == 0 { "No limit".into() } else { n.to_string() }
}

/// Labelled combo box over `options`; returns true when the value changed.
fn combo<T: PartialEq + Copy>(ui: &mut egui::Ui, label: &str, salt: &str, value: &mut T, options: &[(T, String)], enabled: bool) -> bool {
    let mut changed = false;
    widgets::property_row(ui, label, |ui| {
        ui.add_enabled_ui(enabled, |ui| {
            let current = options.iter().find(|(v, _)| v == value).map(|(_, l)| l.as_str()).unwrap_or("-");
            egui::ComboBox::from_id_salt(salt).width(ui.available_width()).selected_text(current).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for (v, l) in options {
                    if ui.selectable_value(value, *v, l).clicked() {
                        changed = true;
                    }
                }
            });
        });
    });
    changed
}

fn check(ui: &mut egui::Ui, label: &str, value: &mut bool, enabled: bool) -> bool {
    widgets::property_row(ui, label, |ui| ui.add_enabled(enabled, egui::Checkbox::without_text(value)).changed())
}

fn opts<T: Copy>(items: &[(T, &str)]) -> Vec<(T, String)> {
    items.iter().map(|(v, l)| (*v, (*l).to_owned())).collect()
}

impl TextureUi {
    /// The whole texture inspector for `path`, editing `t` in place.
    pub fn show(&mut self, ui: &mut egui::Ui, path: &Path, t: &mut TextureImport, dirty: bool) -> TextureUiOut {
        let mut out = TextureUiOut::default();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        ui.heading(if dirty { "Texture Import *" } else { "Texture Import" });
        ui.label(egui::RichText::new(path.display().to_string()).small().weak());
        ui.add_space(6.0);
        if !t.configured {
            widgets::info_note(
                ui,
                "Legacy import: the colour space follows the material slot that uses this texture. Change any setting to take control here.",
            );
            ui.add_space(6.0);
        }

        let platform = self.tab.platform();
        self.drive_preview(ui.ctx(), path, t, platform);
        self.preview_ui(ui, t);
        ui.add_space(8.0);

        // ---- Type and per-type sections ----
        widgets::eyebrow(ui, "Type");
        let kinds: Vec<(TextureKind, String)> = TextureKind::ALL.iter().map(|k| (*k, k.label().to_owned())).collect();
        let mut kind = t.kind;
        if combo(ui, "Texture type", "tex-kind", &mut kind, &kinds, true) {
            t.apply_kind(kind);
            out.changed = true;
        }
        let s = kind_sections(t.kind);
        if s.srgb {
            out.changed |= check(ui, "sRGB (colour)", &mut t.srgb, true);
        }
        if s.single_channel {
            widgets::property_row(ui, "Channel", |ui| {
                out.changed |= widgets::segmented(ui, &mut t.single_channel, &[(Channel::R, "R"), (Channel::G, "G"), (Channel::B, "B"), (Channel::A, "A")]).changed();
            });
        }
        if t.kind == TextureKind::Sprite {
            let size = self.loaded.as_ref().map(|l| (l.preview.source.width, l.preview.source.height));
            self.sprite.session_dirty = dirty;
            out.changed |= self.sprite.show(ui, path, t, size);
            // The Sprite Editor's Apply / Revert All go through the same session.
            let asked = self.sprite.take_editor_requests();
            out.apply |= asked.apply;
            out.revert |= asked.revert_all;
        }
        if s.alpha {
            widgets::property_row(ui, "Alpha source", |ui| {
                out.changed |= widgets::segmented(
                    ui,
                    &mut t.alpha_source,
                    &[(AlphaSource::None, "None"), (AlphaSource::FromInput, "Input"), (AlphaSource::FromGray, "Grayscale")],
                )
                .changed();
            });
            let can = t.alpha_source != AlphaSource::None;
            out.changed |= check(ui, "Alpha is transparency", &mut t.alpha_is_transparency, can);
        }
        if s.normal {
            widgets::eyebrow(ui, "Normal map");
            out.changed |= check(ui, "Flip green", &mut t.normal.flip_green, true);
            out.changed |= check(ui, "From grayscale", &mut t.normal.from_height, true);
            if t.normal.from_height {
                widgets::property_row(ui, "Bumpiness", |ui| {
                    out.changed |= widgets::slider(ui, egui::Slider::new(&mut t.normal.bumpiness, 0.0..=0.3)).changed();
                });
                widgets::property_row(ui, "Filtering", |ui| {
                    out.changed |= widgets::segmented(ui, &mut t.normal.filter, &[(NormalFilter::Sharp, "Sharp"), (NormalFilter::Smooth, "Smooth")]).changed();
                });
            }
        }
        if s.npot || s.mips {
            let (open, _) = widgets::section_header_with(ui, egui::Id::new("tex-advanced"), None, "Advanced", None, false, |_| {});
            if open {
                if s.npot {
                    let o = opts(&[
                        (NpotScale::None, "None"),
                        (NpotScale::ToNearest, "To nearest"),
                        (NpotScale::ToLarger, "To larger"),
                        (NpotScale::ToSmaller, "To smaller"),
                    ]);
                    out.changed |= combo(ui, "Non-power of 2", "tex-npot", &mut t.npot, &o, true);
                }
                if s.mips {
                    out.changed |= check(ui, "Generate mipmaps", &mut t.mips.enabled, true);
                    if t.mips.enabled {
                        widgets::property_row(ui, "Mip filter", |ui| {
                            out.changed |= widgets::segmented(ui, &mut t.mips.filter, &[(MipFilter::Box, "Box"), (MipFilter::Kaiser, "Kaiser")]).changed();
                        });
                        out.changed |= check(ui, "Preserve coverage", &mut t.mips.preserve_coverage, true);
                        if t.mips.preserve_coverage {
                            widgets::property_row(ui, "Alpha cutoff", |ui| {
                                out.changed |= widgets::slider(ui, egui::Slider::new(&mut t.mips.cutoff, 0.0..=1.0)).changed();
                            });
                        }
                    }
                }
            }
        }

        // ---- Sampler ----
        widgets::eyebrow(ui, "Sampler");
        let wraps = opts(&[(WrapMode::Repeat, "Repeat"), (WrapMode::Clamp, "Clamp"), (WrapMode::Mirror, "Mirror")]);
        out.changed |= combo(ui, "Wrap U", "tex-wrap-u", &mut t.sampler.wrap_u, &wraps, true);
        out.changed |= combo(ui, "Wrap V", "tex-wrap-v", &mut t.sampler.wrap_v, &wraps, true);
        let filters = opts(&[(FilterMode::Point, "Point"), (FilterMode::Bilinear, "Bilinear"), (FilterMode::Trilinear, "Trilinear")]);
        out.changed |= combo(ui, "Filter", "tex-filter", &mut t.sampler.filter, &filters, true);
        widgets::property_row(ui, "Aniso level", |ui| {
            let mut a = t.sampler.aniso as i32;
            if widgets::slider(ui, egui::Slider::new(&mut a, 0..=16)).changed() {
                t.sampler.aniso = a as u8;
                out.changed = true;
            }
        });
        widgets::property_row(ui, "Mip bias", |ui| {
            if widgets::slider(ui, egui::Slider::new(&mut t.sampler.mip_bias, -2.0..=2.0).step_by(0.125)).changed() {
                out.changed = true;
            }
        });
        widgets::info_note(ui, "Sampler settings apply to materials using this texture as you edit them, and are saved with the texture.");

        // ---- Platforms ----
        widgets::eyebrow(ui, "Platform");
        widgets::segmented(ui, &mut self.tab, &[(Tab::Default, "Default"), (Tab::Desktop, "Desktop"), (Tab::Quest, "Quest")]);
        ui.add_space(4.0);
        out.changed |= self.platform_ui(ui, t);

        // ---- Warnings ----
        if let Some(l) = &self.loaded {
            let warnings = t.warnings(&l.preview.source, platform, &name);
            if !warnings.is_empty() {
                ui.add_space(6.0);
            }
            for w in warnings {
                let (fill, fg) = match w.level {
                    WarnLevel::Info => (theme::BG_APP, theme::TEXT_MUTED),
                    WarnLevel::Warning => (theme::STATUS_WARN_BG, theme::STATUS_WARN_FG),
                    WarnLevel::Error => (theme::STATUS_WARN_BG, theme::STATUS_ERROR),
                };
                egui::Frame::new().fill(fill).corner_radius(egui::CornerRadius::same(theme::R_ROW)).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(egui::Label::new(egui::RichText::new(&w.text).size(theme::SIZE_META).color(fg)).wrap());
                });
                ui.add_space(4.0);
            }
        }

        // ---- Apply / Revert ----
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.add_enabled_ui(dirty, |ui| {
                if widgets::secondary_button(ui, Some(Icon::Reset), "Revert").on_hover_text("Discard the edits and go back to the saved settings").clicked() {
                    out.revert = true;
                }
                if widgets::primary_button_sized(ui, None, "Apply", theme::H_FIELD).on_hover_text("Save the settings and reimport the texture").clicked() {
                    out.apply = true;
                }
            });
            if dirty {
                ui.label(egui::RichText::new("unapplied").size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        });

        if out.changed {
            t.configured = true;
        }
        out
    }

    /// One platform tab: override checkbox (not on Default), max size,
    /// resize filter, compression and format.
    fn platform_ui(&mut self, ui: &mut egui::Ui, t: &mut TextureImport) -> bool {
        let mut changed = false;
        let platform = self.tab.platform();
        let overridden = match self.tab {
            Tab::Default => true,
            Tab::Desktop => t.desktop.is_some(),
            Tab::Quest => t.quest.is_some(),
        };
        if self.tab != Tab::Default {
            let mut on = overridden;
            let label = format!("Override for {}", platform.label());
            if check(ui, &label, &mut on, true) {
                let seed = t.override_seed(platform);
                let slot = if self.tab == Tab::Desktop { &mut t.desktop } else { &mut t.quest };
                *slot = on.then_some(seed);
                changed = true;
            }
        }
        // The values this tab edits (the override, or the base on Default).
        let mut v = match self.tab {
            Tab::Default => PlatformOverride { max_size: t.max_size, resize: t.resize, compression: t.compression, format: None },
            Tab::Desktop => t.desktop.unwrap_or_else(|| t.platform_settings(Platform::Desktop)),
            Tab::Quest => t.quest.unwrap_or_else(|| t.platform_settings(Platform::Quest)),
        };
        let before = v;
        let enabled = overridden;
        let sizes: Vec<(u32, String)> = MAX_SIZES.iter().map(|n| (*n, size_label(*n))).collect();
        combo(ui, "Max size", "tex-max", &mut v.max_size, &sizes, enabled);
        ui.add_enabled_ui(enabled, |ui| {
            widgets::property_row(ui, "Resize", |ui| {
                widgets::segmented(ui, &mut v.resize, &[(ResizeFilter::Mitchell, "Mitchell"), (ResizeFilter::Bilinear, "Bilinear")]);
            });
        });
        let comps: Vec<(Compression, String)> = Compression::ALL.iter().map(|c| (*c, c.label().to_owned())).collect();
        combo(ui, "Compression", "tex-comp", &mut v.compression, &comps, enabled && v.format.is_none());
        if self.tab != Tab::Default {
            let mut formats: Vec<(Option<FormatOverride>, String)> = vec![(None, "Automatic".into()), (Some(FormatOverride::Rgba8), "RGBA8".into())];
            if platform == Platform::Desktop {
                for f in [FormatOverride::Bc1, FormatOverride::Bc4, FormatOverride::Bc5, FormatOverride::Bc7, FormatOverride::Bc6h, FormatOverride::RgbaF16] {
                    formats.push((Some(f), f.label()));
                }
            } else {
                for b in [4u8, 5, 6, 8] {
                    formats.push((Some(FormatOverride::Astc(b)), FormatOverride::Astc(b).label()));
                }
            }
            combo(ui, "Format", "tex-format", &mut v.format, &formats, enabled);
        }
        if let Some(l) = &self.loaded {
            // Automatic shows what it resolved to, locked.
            if (self.tab == Tab::Default || v.format.is_none()) && overridden {
                widgets::property_row(ui, "Resolved", |ui| {
                    ui.label(egui::RichText::new(l.preview.target.label()).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
                });
            }
        }
        if v != before {
            match self.tab {
                Tab::Default => {
                    t.max_size = v.max_size;
                    t.resize = v.resize;
                    t.compression = v.compression;
                }
                Tab::Desktop => t.desktop = Some(v),
                Tab::Quest => t.quest = Some(v),
            }
            changed = true;
        }
        changed
    }

    /// Start / collect the background preview for the current draft.
    fn drive_preview(&mut self, ctx: &egui::Context, path: &Path, t: &TextureImport, platform: Platform) {
        let key: PreviewKey = (path.to_path_buf(), t.clone(), platform);
        if self.wanted.as_ref().is_none_or(|(k, _)| *k != key) {
            self.wanted = Some((key.clone(), Instant::now()));
        }
        if let Some((jkey, rx)) = &self.job {
            match rx.try_recv() {
                Ok((rkey, result)) => {
                    let _ = jkey;
                    self.job = None;
                    match result {
                        Ok((preview, legacy_vram)) => {
                            tracing::info!(
                                "texture {}: {} ({}), {}",
                                rkey.0.display(),
                                info_line(&preview),
                                rkey.2.label(),
                                vram_delta_line(legacy_vram, preview.vram_bytes)
                            );
                            self.serial += 1;
                            self.mip = self.mip.min(preview.mip_count.saturating_sub(1));
                            self.loaded = Some(Loaded { key: rkey, preview, legacy_vram });
                            self.error = None;
                        }
                        Err(e) => {
                            self.error = Some(e);
                            self.loaded = None;
                        }
                    }
                }
                Err(TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
                Err(TryRecvError::Disconnected) => self.job = None,
            }
        }
        let current = self.loaded.as_ref().map(|l| &l.key) == Some(&key);
        let settled = self.wanted.as_ref().is_some_and(|(_, since)| since.elapsed() > Duration::from_millis(250));
        if self.job.is_none() && !current && settled && self.error_key_differs(&key) {
            let (tx, rx) = std::sync::mpsc::channel();
            let job_key = key.clone();
            let ctx2 = ctx.clone();
            std::thread::spawn(move || {
                let r = (|| {
                    let preview = zestengine_assets::preview_texture(&job_key.0, &job_key.1, job_key.2).map_err(|e| format!("{e:#}"))?;
                    let legacy = TextureImport::legacy(job_key.1.effective_srgb());
                    let legacy_vram = zestengine_assets::preview_texture(&job_key.0, &legacy, job_key.2).map(|p| p.vram_bytes).unwrap_or(0);
                    Ok((preview, legacy_vram))
                })();
                let _ = tx.send((job_key, r));
                ctx2.request_repaint();
            });
            self.job = Some((key, rx));
        } else if !current && !settled {
            ctx.request_repaint_after(Duration::from_millis(120));
        }
    }

    /// Don't retry a failing key every frame.
    fn error_key_differs(&self, key: &PreviewKey) -> bool {
        self.error.is_none() || self.wanted.as_ref().is_some_and(|(k, _)| k != key)
    }

    /// Channel buttons, the image, the mip slider and the info lines.
    fn preview_ui(&mut self, ui: &mut egui::Ui, t: &TextureImport) {
        let busy = self.job.is_some() || self.loaded.as_ref().map(|l| &l.key.1) != Some(t);
        let Some(l) = &self.loaded else {
            if let Some(e) = &self.error {
                ui.label(egui::RichText::new(format!("Preview failed: {e}")).color(theme::STATUS_ERROR).size(theme::SIZE_META));
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new("Processing preview…").size(theme::SIZE_META).color(theme::TEXT_DIM));
                });
            }
            return;
        };
        let p = &l.preview;
        if !p.levels.is_empty() {
            let channels = available_channels(t.kind, p.has_alpha);
            if !channels.contains(&self.channel) {
                self.channel = PreviewChannel::Rgb;
            }
            if channels.len() > 1 {
                ui.horizontal(|ui| {
                    widgets::trough(ui, |ui| {
                        for c in &channels {
                            if widgets::segment(ui, self.channel == *c, c.label()).clicked() {
                                self.channel = *c;
                            }
                        }
                    });
                });
                ui.add_space(4.0);
            }
            let mip = self.mip.min(p.levels.len() - 1);
            let (px, w, h) = &p.levels[mip];
            let stale = !matches!(&self.view, Some((s, m, c, _)) if *s == self.serial && *m == mip && *c == self.channel);
            if stale {
                let img = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], &channel_view(px, self.channel));
                let handle = ui.ctx().load_texture("texture-import-preview", img, egui::TextureOptions::NEAREST);
                self.view = Some((self.serial, mip, self.channel, handle));
            }
            if let Some((_, _, _, handle)) = &self.view {
                // The base size sets the frame, so smaller mips show scaled up.
                let (bw, bh) = (p.levels[0].1 as f32, p.levels[0].2 as f32);
                let avail = ui.available_width().min(300.0);
                let scale = (avail / bw).min(220.0 / bh);
                let size = egui::vec2(bw * scale, bh * scale);
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                ui.painter().rect_filled(rect, egui::CornerRadius::same(theme::R_ROW), theme::BG_FIELD);
                ui.painter().image(handle.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            if p.levels.len() > 1 {
                let mut m = mip;
                widgets::property_row(ui, "Mip", |ui| {
                    widgets::slider(ui, egui::Slider::new(&mut m, 0..=p.levels.len() - 1).custom_formatter(|v, _| {
                        let i = v as usize;
                        format!("{i} ({}×{})", p.levels[i].1, p.levels[i].2)
                    }));
                });
                self.mip = m;
            }
        }
        let mono = theme::mono(theme::SIZE_META);
        ui.label(egui::RichText::new(info_line(p)).font(mono.clone()).color(if busy { theme::TEXT_DIM } else { theme::TEXT_SECONDARY }));
        ui.label(egui::RichText::new(vram_delta_line(l.legacy_vram, p.vram_bytes) + " vs the legacy import").font(mono).color(theme::TEXT_DIM));
        if matches!(p.target, TargetFormat::Rgba8 | TargetFormat::RgbaF16) && t.configured {
            ui.label(egui::RichText::new("Uncompressed").size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{ImporterSettings, SourceInfo};

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine_texinsp_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn configured(kind: TextureKind) -> TextureImport {
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        t.apply_kind(kind);
        t
    }

    fn textures(albedo: Option<&str>, normal: Option<&str>, rough: Option<&str>) -> MaterialTexturePaths {
        MaterialTexturePaths {
            albedo: albedo.map(PathBuf::from),
            normal: normal.map(PathBuf::from),
            roughness: rough.map(PathBuf::from),
            ..MaterialTexturePaths::default()
        }
    }

    #[test]
    fn ignore_choice_is_saved_in_the_material_and_hides_the_warning() {
        let root = temp_dir("ignore");
        std::fs::create_dir_all(root.join("tex")).unwrap();
        std::fs::write(root.join("tex/n.png"), b"x").unwrap();
        let mat = root.join("a.material");
        let mut file = zestengine_assets::MaterialFile {
            name: "a".into(),
            shader: "standard".into(),
            params: Default::default(),
            features: Default::default(),
            render_queue: None,
            textures: Default::default(),
            custom: Default::default(),
        };
        file.textures.normal = Some("tex/n.png".into());
        zestengine_assets::save_material_file(&mat, &file).unwrap();
        persist_slot_ignore(&mat, 1, Path::new("tex/n.png")).unwrap();
        persist_slot_ignore(&mat, 1, Path::new("tex/n.png")).unwrap();
        let back = zestengine_assets::load_material_file(&mat).unwrap();
        assert_eq!(back.textures.ignored_warnings, vec![(1, PathBuf::from("tex/n.png"))], "saved once");
        assert_eq!(back.textures.normal.as_deref(), Some(Path::new("tex/n.png")), "textures untouched");
        // An old file without the field loads with none.
        let old = zestengine_assets::MaterialTextures::default();
        assert!(old.ignored_warnings.is_empty());

        // The validator reads the saved choice: no warning for the normal slot
        // (the texture has no configured meta, so it would otherwise be fine; use
        // an unconfigured sRGB-default meta to force the NotNormalMap issue).
        let mut meta = zestengine_assets::AssetMeta::new_for(&root.join("tex/n.png"));
        meta.importer = ImporterSettings::Texture(TextureImport { configured: true, ..TextureImport::default() });
        zestengine_assets::save_asset_meta(root.join("tex/n.png"), &meta).unwrap();
        let tex = textures(None, Some("tex/n.png"), None);
        let mut v = SlotValidator::default();
        let with = v.warnings(&root, &tex, "a.material", Some(&mat), Vec::new);
        assert!(with.is_empty(), "saved Ignore hides it: {with:?}");
        let mut fresh = SlotValidator::default();
        let without = fresh.warnings(&root, &tex, "a.material", None, Vec::new);
        assert_eq!(without.len(), 1, "the same texture warns without the saved choice");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn usage_scan_runs_off_thread_and_lands_later() {
        let root = temp_dir("scan");
        let mat = root.join("m.material");
        let mut file = zestengine_assets::MaterialFile {
            name: "m".into(),
            shader: "standard".into(),
            params: Default::default(),
            features: Default::default(),
            render_queue: None,
            textures: Default::default(),
            custom: Default::default(),
        };
        file.textures.albedo = Some("t.png".into());
        zestengine_assets::save_material_file(&mat, &file).unwrap();
        let tex = textures(Some("t.png"), None, None);
        let mut v = SlotValidator::default();
        let _ = v.warnings(&root, &tex, "x", None, Vec::new);
        assert!(v.scanning() || v.usage.is_some(), "a scan was started");
        let t0 = Instant::now();
        while v.scanning() && t0.elapsed() < Duration::from_secs(5) {
            let _ = v.warnings(&root, &tex, "x", None, Vec::new);
            std::thread::sleep(Duration::from_millis(5));
        }
        let usage = &v.usage.as_ref().expect("scan finished").1;
        assert!(usage.contains_key(&zestengine_core::asset_ref::resolve_rel_path(&root, Path::new("t.png"))));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sections_follow_the_texture_type() {
        let d = kind_sections(TextureKind::Default);
        assert!(d.srgb && d.alpha && d.mips && d.npot && !d.normal && !d.single_channel);
        let n = kind_sections(TextureKind::NormalMap);
        assert!(n.normal && !n.srgb && !n.alpha);
        let s = kind_sections(TextureKind::SingleChannel);
        assert!(s.single_channel && !s.srgb && !s.alpha);
        assert!(kind_sections(TextureKind::Cursor).alpha && !kind_sections(TextureKind::Cursor).mips);
    }

    #[test]
    fn channel_buttons_and_views() {
        assert_eq!(available_channels(TextureKind::NormalMap, true), vec![PreviewChannel::Rgb]);
        assert!(!available_channels(TextureKind::Default, false).contains(&PreviewChannel::A));
        assert!(available_channels(TextureKind::Default, true).contains(&PreviewChannel::A));
        let px = [10, 20, 30, 40];
        assert_eq!(channel_view(&px, PreviewChannel::Rgb), [10, 20, 30, 255]);
        assert_eq!(channel_view(&px, PreviewChannel::G), [20, 20, 20, 255]);
        assert_eq!(channel_view(&px, PreviewChannel::A), [40, 40, 40, 255]);
    }

    #[test]
    fn info_line_reports_format_size_and_vram() {
        let source = SourceInfo { width: 300, height: 200, has_alpha: false, hdr: false };
        let p = TexturePreview {
            source,
            target: TargetFormat::Bc1,
            format: zestengine_render::TexFormat::Bc1Srgb,
            width: 256,
            height: 128,
            levels: vec![(vec![0; 4], 1, 1)],
            mip_count: 9,
            vram_bytes: 5_592_405,
            srgb: true,
            has_alpha: false,
        };
        assert_eq!(info_line(&p), "300×200 → 256×128 (NPOT) · BC1 sRGB · 5.3 MB · 9 mips");
        assert_eq!(vram_delta_line(21_000_000, 5_250_000), "VRAM 20.0 MB → 5.0 MB (-75%)");
        assert_eq!(human_bytes(512), "512 B");
    }

    #[test]
    fn preview_reports_the_vram_win_over_the_legacy_import() {
        let dir = temp_dir("vram");
        let png = dir.join("floor.png");
        image::RgbaImage::from_fn(300, 200, |x, y| image::Rgba([(x % 256) as u8, (y % 256) as u8, 90, 255])).save(&png).unwrap();
        let legacy = zestengine_assets::preview_texture(&png, &TextureImport::legacy(true), Platform::Desktop).unwrap();
        assert_eq!(legacy.target, TargetFormat::Bc7);
        let mut t = configured(TextureKind::Default);
        t.max_size = 128;
        let p = zestengine_assets::preview_texture(&png, &t, Platform::Desktop).unwrap();
        assert_eq!(p.target, TargetFormat::Bc1);
        assert!(p.width <= 128 && p.height <= 128);
        assert!(p.vram_bytes * 4 < legacy.vram_bytes, "{} vs {}", p.vram_bytes, legacy.vram_bytes);
        assert!(p.levels.len() > 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn slot_warnings_flag_mismatches_and_conflicts() {
        let root = Path::new("/proj");
        let mat = textures(Some("tex/wood.png"), Some("tex/wood.png"), None);
        let abs = root.join("tex/wood.png");
        let mut metas = HashMap::new();
        metas.insert(abs.clone(), configured(TextureKind::Default));
        let mut meta_of = |p: &Path| metas.get(p).cloned();
        // The same colour texture is also this material's normal map, and another material's albedo.
        let usage = build_usage(root, &[("a.material".into(), mat.clone()), ("b.material".into(), textures(Some("tex/wood.png"), None, None))]);
        let w = slot_warnings(root, &mat, "a.material", &usage, &HashSet::new(), &mut meta_of, &|_| false);
        assert_eq!(w.len(), 1, "albedo is fine, the normal slot is not: {w:?}");
        assert_eq!(w[0].slot, 1);
        assert!(w[0].text.contains("normal map") && w[0].text.contains("Normal slot"));
        // Conflicts: albedo uses (this material's slot 0 and b's slot 0) expect colour, the normal slot a different class.
        assert_eq!(w[0].conflicts.len(), 2, "{:?}", w[0].conflicts);
        assert_eq!(w[0].duplicate_name, "wood_normal.png");
        // Ignoring hides it.
        let mut ignored = HashSet::new();
        ignored.insert((abs, 1));
        assert!(slot_warnings(root, &mat, "a.material", &usage, &ignored, &mut meta_of, &|_| false).is_empty());
        // A texture nobody configured never warns.
        let mut none = |_: &Path| None;
        assert!(slot_warnings(root, &mat, "a.material", &usage, &HashSet::new(), &mut none, &|_| false).is_empty());
    }

    #[test]
    fn single_use_mismatch_is_a_plain_auto_fix() {
        let root = Path::new("/proj");
        let mat = textures(None, Some("n.png"), None);
        let mut meta_of = |_: &Path| Some(configured(TextureKind::Default));
        let usage = build_usage(root, &[("m".into(), mat.clone())]);
        let w = slot_warnings(root, &mat, "m", &usage, &HashSet::new(), &mut meta_of, &|_| false);
        assert_eq!(w.len(), 1);
        assert!(w[0].conflicts.is_empty() && w[0].fix_label == "Auto-fix" && !w[0].info);
    }

    #[test]
    fn auto_fix_rewrites_the_meta_and_duplicate_makes_a_configured_copy() {
        let dir = temp_dir("fix");
        let png = dir.join("n.png");
        image::RgbaImage::from_pixel(8, 8, image::Rgba([128, 128, 255, 255])).save(&png).unwrap();
        let mut meta = zestengine_assets::AssetMeta::new_for(&png);
        meta.importer = ImporterSettings::Texture(configured(TextureKind::Default));
        zestengine_assets::save_asset_meta(&png, &meta).unwrap();
        let mat = textures(None, Some("n.png"), None);

        let out = apply_slot_fix(&dir, &mat, 1, &SlotFix::AutoFix).unwrap();
        assert_eq!(out, FixOutcome::Reimport(png.clone()));
        let fixed = zestengine_assets::load_asset_meta(&png).unwrap().unwrap();
        let t = fixed.importer.as_texture().unwrap();
        assert!(t.configured && t.kind == TextureKind::NormalMap && !t.srgb);
        assert_eq!(fixed.id, meta.id, "the asset id survives a fix");

        // Roughness slot on the same file: duplicate, don't touch the original.
        let rough = textures(None, None, Some("n.png"));
        let out = apply_slot_fix(&dir, &rough, 13, &SlotFix::Duplicate).unwrap();
        let FixOutcome::Duplicated { slot, new_abs } = out else { panic!("duplicated") };
        assert_eq!((slot, new_abs.file_name().unwrap().to_str().unwrap()), (13, "n_linear.png"));
        assert!(new_abs.is_file());
        let dm = zestengine_assets::load_asset_meta(&new_abs).unwrap().unwrap();
        assert_ne!(dm.id, meta.id);
        assert_eq!(dm.importer.as_texture().unwrap().kind, TextureKind::SingleChannel);
        let original = zestengine_assets::load_asset_meta(&png).unwrap().unwrap();
        assert_eq!(original.importer.as_texture().unwrap().kind, TextureKind::NormalMap);

        assert_eq!(apply_slot_fix(&dir, &rough, 13, &SlotFix::Ignore).unwrap(), FixOutcome::Ignored(png.clone(), 13));
        assert!(apply_slot_fix(&dir, &rough, 12, &SlotFix::AutoFix).is_err(), "empty slot");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
