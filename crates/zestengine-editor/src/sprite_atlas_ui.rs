//! The `.spriteatlas` inspector (2D-4): what to pack, the packing rules, the
//! page texture settings, platform overrides and variants, a **Pack now**
//! button, and a preview of the packed pages.
//!
//! The decisions are in `zestengine_assets::sprite_atlas` (unit tested); the
//! pure helpers here ([`normalize_packable`], [`add_packable`], [`summary`],
//! [`preview_pack`]) are tested too, and the egui code only lays them out. The
//! pack runs on a worker thread (packing a big atlas takes seconds); its pages
//! come back as RGBA, never the encoded (BC7 / ASTC) pages, so the preview shows
//! what the packer placed, with padding, rotation and trimming.
//!
//! Edits save straight to the `.spriteatlas` file (like the other asset
//! inspectors); the cooked-atlas cache is keyed by the settings, so the next
//! Play or build repacks on its own. This GUI has not been seen on screen.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};

use zestengine_assets::sprite_atlas::{AtlasDb, AtlasPlatform, AtlasVariant, PackedAtlas, PackingMode, PageLayout, SpriteAtlasFile, pack_members};
use zestengine_assets::{Compression, Platform};
use crate::{icons::Icon, theme, widgets};

/// A project path as a packable string: forward slashes, relative to `root`
/// when it lies inside it. `None` for an empty string or a path that climbs out
/// of the project.
pub fn normalize_packable(root: &Path, input: &str) -> Option<String> {
    let s = input.trim().replace('\\', "/");
    let s = s.strip_prefix(&format!("{}/", root.to_string_lossy().replace('\\', "/"))).unwrap_or(&s).trim_start_matches("./").to_owned();
    let (path, id) = match s.split_once('#') {
        Some((p, id)) => (p.trim_end_matches('/'), Some(id)),
        None => (s.trim_end_matches('/'), None),
    };
    if path.is_empty() || path.starts_with('/') || path.split('/').any(|p| p == "..") {
        return None;
    }
    Some(match id {
        Some(id) => format!("{path}#{id}"),
        None => path.to_owned(),
    })
}

/// Add a packable (normalized, no duplicates). Returns whether the list changed.
pub fn add_packable(file: &mut SpriteAtlasFile, root: &Path, input: &str) -> bool {
    match normalize_packable(root, input) {
        Some(p) if !file.packables.contains(&p) => {
            file.packables.push(p);
            true
        }
        _ => false,
    }
}

/// What an atlas holds, from its packed table: `(pages, sprites, used fraction of the page area)`.
pub fn summary(packed: &PackedAtlas) -> (usize, usize, f32) {
    let total: u64 = packed.table.pages.iter().map(|p| p[0] as u64 * p[1] as u64).sum();
    let used: u64 = packed.table.sprites.iter().map(|s| s.rect.w as u64 * s.rect.h as u64).sum();
    (packed.table.pages.len(), packed.table.sprites.len(), if total == 0 { 0.0 } else { used as f32 / total as f32 })
}

/// Pack the atlas at `rel` (project-relative) for `platform` and return the
/// packed pages. This is `Pack now`'s worker.
pub fn preview_pack(root: &Path, rel: &str, platform: Platform) -> Result<PackedAtlas, String> {
    let db = AtlasDb::scan(root);
    let e = db.atlases.iter().find(|e| e.rel == rel).ok_or_else(|| format!("{rel} is not a readable atlas"))?;
    if e.members.is_empty() {
        return Err("nothing to pack: the list matches no sprite texture (a texture needs the Sprite type, and a sprite table), or an earlier atlas already owns it".into());
    }
    let load = |m: &zestengine_assets::sprite_atlas::Member| {
        let img = image::open(&m.abs).ok()?.to_rgba8();
        Some(zestengine_assets::sprite_atlas::Rgba { w: img.width(), h: img.height(), px: img.into_raw() })
    };
    pack_members(&e.name, &e.members, &e.file, platform, &load)
}

struct Preview {
    packed: PackedAtlas,
    textures: Vec<egui::TextureHandle>,
    page: usize,
}

/// Per-inspector state: the file being edited, the worker, the preview.
#[derive(Default)]
pub struct SpriteAtlasInspector {
    path: Option<PathBuf>,
    file: SpriteAtlasFile,
    error: Option<String>,
    add_text: String,
    variant_text: String,
    job: Option<Receiver<Result<PackedAtlas, String>>>,
    preview: Option<Preview>,
    status: Option<String>,
    tab: Tab,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Tab {
    #[default]
    Desktop,
    Quest,
}

impl SpriteAtlasInspector {
    /// Draw the inspector for the atlas at `path`; edits are saved at once.
    pub fn ui(&mut self, ui: &mut egui::Ui, root: &Path, path: &Path) {
        if self.path.as_deref() != Some(path) {
            *self = Self { path: Some(path.to_path_buf()), ..Default::default() };
            match SpriteAtlasFile::load(path) {
                Ok(f) => self.file = f,
                Err(e) => self.error = Some(format!("{e:#}")),
            }
        }
        if let Some(e) = &self.error {
            ui.label(egui::RichText::new(format!("Failed to open the atlas: {e}")).color(theme::TEXT_MUTED));
            if widgets::secondary_button(ui, None, "Retry").clicked() {
                self.path = None;
            }
            return;
        }
        self.poll(ui.ctx());
        let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/");
        let mut changed = false;
        ui.heading("Sprite Atlas");
        ui.label(egui::RichText::new(&rel).small().weak());
        ui.separator();
        changed |= self.packables(ui, root);
        changed |= self.rules(ui);
        changed |= self.pages(ui);
        changed |= self.platforms(ui);
        changed |= self.variants(ui);
        if changed && let Err(e) = self.file.save(path) {
            tracing::error!("saving sprite atlas: {e:#}");
        }
        self.pack_section(ui, root, &rel);
    }

    fn packables(&mut self, ui: &mut egui::Ui, root: &Path) -> bool {
        widgets::eyebrow(ui, "Objects for packing");
        let zone = egui::Frame::new().show(ui, |ui| {
            ui.set_width(ui.available_width());
            let mut changed = false;
            let mut remove = None;
            for (i, p) in self.file.packables.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(egui::RichText::new(p).size(theme::SIZE_BODY).color(theme::TEXT_BODY)).truncate());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, Icon::Trash, "Remove from the atlas", false).clicked() {
                            remove = Some(i);
                        }
                    });
                });
            }
            if let Some(i) = remove {
                self.file.packables.remove(i);
                changed = true;
            }
            if self.file.packables.is_empty() {
                widgets::info_note(ui, "Drop sprite textures or folders from the Files panel here, or type a path. A folder packs its sprite textures, recursively. A texture is packed by one atlas only: the first atlas, by path, that lists it.");
            }
            ui.horizontal(|ui| {
                let add = ui.add(egui::TextEdit::singleline(&mut self.add_text).hint_text("Sprites/ui or Sprites/hero.png").desired_width((ui.available_width() - 70.0).max(60.0)));
                let go = widgets::secondary_button(ui, Some(Icon::Plus), "Add").clicked() || (add.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                if go && add_packable(&mut self.file, root, &self.add_text) {
                    self.add_text.clear();
                    changed = true;
                }
            });
            changed
        });
        let mut changed = zone.inner;
        if let Some(dropped) = zone.response.dnd_release_payload::<PathBuf>() {
            changed |= add_packable(&mut self.file, root, &dropped.to_string_lossy());
        }
        changed
    }

    fn rules(&mut self, ui: &mut egui::Ui) -> bool {
        let f = &mut self.file;
        let mut changed = false;
        widgets::eyebrow(ui, "Packing");
        widgets::property_row(ui, "Mode", |ui| {
            let opts: Vec<(PackingMode, &str)> = PackingMode::ALL.iter().map(|m| (*m, m.label())).collect();
            changed |= widgets::segmented(ui, &mut f.packing, &opts).changed();
        });
        if f.packing == PackingMode::Tight {
            widgets::info_note(ui, "Tight mode also records each sprite's alpha outline, but packs the trimmed bounds (concave shapes do not interlock).");
        }
        widgets::property_row(ui, "Allow rotation", |ui| changed |= ui.checkbox(&mut f.allow_rotation, "").on_hover_text("Turn sprites 90 degrees to pack tighter").changed());
        widgets::property_row(ui, "Trim", |ui| changed |= ui.checkbox(&mut f.trim, "").on_hover_text("Cut fully transparent borders (not off 9-sliced sprites)").changed());
        widgets::property_row(ui, "Padding", |ui| changed |= ui.add(egui::DragValue::new(&mut f.padding).range(0..=64).suffix(" px")).changed());
        widgets::property_row(ui, "Max size", |ui| {
            let sizes = [64u32, 128, 256, 512, 1024, 2048, 4096, 8192];
            egui::ComboBox::from_id_salt("atlas-max-size").width(ui.available_width()).selected_text(f.max_size.to_string()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for s in sizes {
                    if ui.selectable_label(f.max_size == s, s.to_string()).clicked() {
                        f.max_size = s;
                        changed = true;
                    }
                }
            });
        });
        widgets::property_row(ui, "Power of two", |ui| changed |= ui.checkbox(&mut f.power_of_two, "").changed());
        widgets::property_row(ui, "Pixel art", |ui| changed |= ui.checkbox(&mut f.pixel_art, "").on_hover_text("No mips, no filtering margin beyond the extruded edge").changed());
        widgets::property_row(ui, "Include in build", |ui| changed |= ui.checkbox(&mut f.include_in_build, "").on_hover_text("Off: the editor still packs it for Play, game builds leave it out").changed());
        changed
    }

    fn pages(&mut self, ui: &mut egui::Ui) -> bool {
        let f = &mut self.file;
        let mut changed = false;
        widgets::eyebrow(ui, "Pages");
        widgets::property_row(ui, "Layout", |ui| {
            let opts: Vec<(PageLayout, &str)> = PageLayout::ALL.iter().map(|l| (*l, l.label())).collect();
            changed |= widgets::segmented(ui, &mut f.layout, &opts).changed();
        });
        widgets::property_row(ui, "Mip maps", |ui| changed |= ui.checkbox(&mut f.texture.mips.enabled, "").changed());
        if f.texture.mips.enabled && !f.pixel_art {
            widgets::property_row(ui, "Mip levels", |ui| changed |= ui.add(egui::DragValue::new(&mut f.mip_levels).range(1..=zestengine_assets::sprite_atlas::MAX_MIP_LEVELS)).on_hover_text("Padding grows to 2^levels so mips never read a neighbour").changed());
        }
        widgets::property_row(ui, "sRGB", |ui| changed |= ui.checkbox(&mut f.texture.srgb, "").changed());
        widgets::property_row(ui, "Compression", |ui| {
            egui::ComboBox::from_id_salt("atlas-compression").width(ui.available_width()).selected_text(f.texture.compression.label()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for c in Compression::ALL {
                    if ui.selectable_label(f.texture.compression == c, c.label()).clicked() {
                        f.texture.compression = c;
                        changed = true;
                    }
                }
            });
        });
        changed
    }

    fn platforms(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        widgets::eyebrow(ui, "Platform overrides");
        widgets::segmented(ui, &mut self.tab, &[(Tab::Desktop, "Desktop"), (Tab::Quest, "Quest")]);
        let variants: Vec<String> = self.file.variants.iter().map(|v| v.name.clone()).collect();
        let (default_max, default_pad, default_rot) = (self.file.max_size, self.file.padding, self.file.allow_rotation);
        let slot = match self.tab {
            Tab::Desktop => &mut self.file.desktop,
            Tab::Quest => &mut self.file.quest,
        };
        let mut on = slot.is_some();
        widgets::property_row(ui, "Override", |ui| changed |= ui.checkbox(&mut on, "").changed());
        if on && slot.is_none() {
            *slot = Some(AtlasPlatform::default());
        } else if !on {
            *slot = None;
        }
        if let Some(o) = slot {
            let mut max = o.max_size.unwrap_or(default_max);
            let mut has_max = o.max_size.is_some();
            widgets::property_row(ui, "Max size", |ui| {
                changed |= ui.checkbox(&mut has_max, "").changed();
                ui.add_enabled_ui(has_max, |ui| changed |= ui.add(egui::DragValue::new(&mut max).range(16..=16384)).changed());
            });
            o.max_size = has_max.then_some(max);
            let (mut pad, mut has_pad) = (o.padding.unwrap_or(default_pad), o.padding.is_some());
            widgets::property_row(ui, "Padding", |ui| {
                changed |= ui.checkbox(&mut has_pad, "").changed();
                ui.add_enabled_ui(has_pad, |ui| changed |= ui.add(egui::DragValue::new(&mut pad).range(0..=64)).changed());
            });
            o.padding = has_pad.then_some(pad);
            let (mut rot, mut has_rot) = (o.allow_rotation.unwrap_or(default_rot), o.allow_rotation.is_some());
            widgets::property_row(ui, "Rotation", |ui| {
                changed |= ui.checkbox(&mut has_rot, "").changed();
                ui.add_enabled_ui(has_rot, |ui| changed |= ui.checkbox(&mut rot, "").changed());
            });
            o.allow_rotation = has_rot.then_some(rot);
            widgets::property_row(ui, "Variant", |ui| {
                let shown = o.variant.clone().unwrap_or_else(|| "Full size".into());
                egui::ComboBox::from_id_salt(("atlas-variant", self.tab as u8)).width(ui.available_width()).selected_text(shown).show_ui(ui, |ui| {
                    widgets::menu_frame(ui);
                    if ui.selectable_label(o.variant.is_none(), "Full size").clicked() {
                        o.variant = None;
                        changed = true;
                    }
                    for v in &variants {
                        if ui.selectable_label(o.variant.as_deref() == Some(v), v).clicked() {
                            o.variant = Some(v.clone());
                            changed = true;
                        }
                    }
                });
            });
        }
        changed
    }

    fn variants(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        widgets::eyebrow(ui, "Variants");
        let mut remove = None;
        for (i, v) in self.file.variants.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&v.name).color(theme::TEXT_BODY));
                changed |= ui.add(egui::Slider::new(&mut v.scale, 0.05..=1.0).fixed_decimals(2)).on_hover_text("Page scale: 0.5 is a quarter of the memory; UVs do not change").changed();
                if widgets::icon_button(ui, Icon::Trash, "Remove the variant", false).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            let gone = self.file.variants.remove(i).name;
            for o in [&mut self.file.desktop, &mut self.file.quest].into_iter().flatten() {
                if o.variant.as_deref() == Some(gone.as_str()) {
                    o.variant = None;
                }
            }
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.variant_text).hint_text("Variant name").desired_width((ui.available_width() - 70.0).max(60.0)));
            let name = self.variant_text.trim().to_owned();
            let dup = self.file.variants.iter().any(|v| v.name == name);
            if ui.add_enabled(!name.is_empty() && !dup, egui::Button::new("Add")).clicked() {
                self.file.variants.push(AtlasVariant { name, scale: 0.5 });
                self.variant_text.clear();
                changed = true;
            }
        });
        changed
    }

    fn pack_section(&mut self, ui: &mut egui::Ui, root: &Path, rel: &str) {
        widgets::eyebrow(ui, "Preview");
        let busy = self.job.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(!busy, egui::Button::new(format!("{} Pack now", crate::egui_phosphor::regular::PACKAGE))).on_hover_text("Pack the atlas for the desktop and show the pages").clicked() {
                let (tx, rx) = std::sync::mpsc::channel();
                let (root, rel) = (root.to_path_buf(), rel.to_owned());
                std::thread::spawn(move || {
                    let _ = tx.send(preview_pack(&root, &rel, Platform::Desktop));
                });
                self.job = Some(rx);
                self.status = None;
            }
            if busy {
                ui.spinner();
            }
        });
        if let Some(s) = &self.status {
            ui.label(egui::RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_MUTED));
        }
        let Some(p) = self.preview.as_mut() else { return };
        let (pages, sprites, fill) = summary(&p.packed);
        ui.label(egui::RichText::new(format!("{pages} page(s), {sprites} sprite(s), {:.0}% of the page area used", fill * 100.0)).size(theme::SIZE_META).color(theme::TEXT_MUTED));
        if pages > 1 {
            ui.horizontal(|ui| {
                for i in 0..pages {
                    if widgets::segment(ui, p.page == i, &format!("{}", i + 1)).clicked() {
                        p.page = i;
                    }
                }
            });
        }
        if let Some(tex) = p.textures.get(p.page) {
            let size = tex.size_vec2();
            let w = ui.available_width().min(size.x.max(1.0) * 2.0);
            let scale = w / size.x.max(1.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(w, size.y * scale), egui::Sense::hover());
            // A checkerboard shows transparent padding.
            ui.painter().rect_filled(rect, 2.0, theme::BG_FIELD);
            let step = 8.0;
            let (mut y, mut row) = (rect.top(), 0);
            while y < rect.bottom() {
                let (mut x, mut col) = (rect.left(), 0);
                while x < rect.right() {
                    if (row + col) % 2 == 0 {
                        ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(step, step)).intersect(rect), 0.0, theme::BG_PANEL);
                    }
                    x += step;
                    col += 1;
                }
                y += step;
                row += 1;
            }
            ui.painter().image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, theme::LINE_FIELD), egui::StrokeKind::Outside);
            // Placed rects, outlined; rotated ones are marked.
            let page_px = p.packed.table.pages.get(p.page).copied().unwrap_or([1, 1]);
            for s in p.packed.table.sprites.iter().filter(|s| s.page as usize == p.page) {
                let r = egui::Rect::from_min_size(
                    rect.min + egui::vec2(s.rect.x as f32 / page_px[0].max(1) as f32 * rect.width(), s.rect.y as f32 / page_px[1].max(1) as f32 * rect.height()),
                    egui::vec2(s.rect.w as f32 / page_px[0].max(1) as f32 * rect.width(), s.rect.h as f32 / page_px[1].max(1) as f32 * rect.height()),
                );
                ui.painter().rect_stroke(r, 0.0, egui::Stroke::new(1.0_f32, theme::accent().gamma_multiply(if s.rotated { 0.9 } else { 0.4 })), egui::StrokeKind::Inside);
            }
        }
        let _ = root;
    }

    /// Collect a finished pack job.
    fn poll(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.job else { return };
        match rx.try_recv() {
            Ok(Ok(packed)) => {
                let textures = packed
                    .pages
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let img = egui::ColorImage::from_rgba_unmultiplied([p.w as usize, p.h as usize], &p.px);
                        ctx.load_texture(format!("atlas-page-{i}"), img, egui::TextureOptions::NEAREST)
                    })
                    .collect();
                let (pages, sprites, _) = summary(&packed);
                self.status = Some(format!("Packed {sprites} sprite(s) into {pages} page(s)"));
                self.preview = Some(Preview { packed, textures, page: 0 });
                self.job = None;
            }
            Ok(Err(e)) => {
                self.status = Some(format!("Pack failed: {e}"));
                self.preview = None;
                self.job = None;
            }
            Err(TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            Err(TryRecvError::Disconnected) => self.job = None,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use zestengine_assets::sprite::{PixelRect, SpriteDef, SpriteMode};
    use zestengine_assets::{AssetMeta, ImporterSettings, TextureImport, TextureKind};

    #[test]
    fn packables_are_normalized_deduplicated_and_kept_inside_the_project() {
        let root = Path::new("/proj");
        assert_eq!(normalize_packable(root, "Sprites\\ui\\"), Some("Sprites/ui".into()));
        assert_eq!(normalize_packable(root, "/proj/Sprites/a.png"), Some("Sprites/a.png".into()));
        assert_eq!(normalize_packable(root, "./Sprites/a.png#1f"), Some("Sprites/a.png#1f".into()));
        for bad in ["", "  ", "../x.png", "a/../../x.png", "/etc/passwd"] {
            assert_eq!(normalize_packable(root, bad), None, "{bad}");
        }
        let mut f = SpriteAtlasFile::default();
        assert!(add_packable(&mut f, root, "Sprites"));
        assert!(!add_packable(&mut f, root, "Sprites/"), "a duplicate is ignored");
        assert_eq!(f.packables, vec!["Sprites"]);
    }

    pub(crate) fn project(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("zestengine-atlas-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Sprites")).unwrap();
        let tex = root.join("Sprites/hero.png");
        image::RgbaImage::from_fn(64, 32, |x, y| image::Rgba([x as u8 * 3, y as u8 * 5, 40, if x < 60 && y < 30 { 255 } else { 0 }])).save(&tex).unwrap();
        let mut meta = AssetMeta::new_for(&tex);
        let mut t = TextureImport { configured: true, kind: TextureKind::Sprite, ..TextureImport::default() };
        t.sprite.mode = SpriteMode::Multiple;
        t.sprite.sprites = vec![SpriteDef::named("left", PixelRect::new(0, 0, 32, 32)), SpriteDef::named("right", PixelRect::new(32, 0, 32, 32))];
        t.sprite.ensure_ids("hero");
        meta.importer = ImporterSettings::Texture(t);
        zestengine_assets::save_asset_meta(&tex, &meta).unwrap();
        root
    }

    #[test]
    fn pack_now_packs_the_listed_sprites_and_reports_an_empty_list() {
        let root = project("pack");
        let mut f = SpriteAtlasFile { padding: 2, max_size: 128, ..SpriteAtlasFile::default() };
        f.save(&root.join("ui.spriteatlas")).unwrap();
        let err = preview_pack(&root, "ui.spriteatlas", Platform::Desktop).err().expect("an empty atlas cannot pack");
        assert!(err.contains("nothing to pack"), "{err}");
        add_packable(&mut f, &root, "Sprites");
        f.allow_rotation = true;
        f.save(&root.join("ui.spriteatlas")).unwrap();
        let packed = preview_pack(&root, "ui.spriteatlas", Platform::Desktop).unwrap();
        let (pages, sprites, fill) = summary(&packed);
        assert_eq!((pages, sprites), (1, 2));
        assert!(fill > 0.05 && fill <= 1.0, "{fill}");
        assert_eq!(packed.pages.len(), pages);
        assert!(preview_pack(&root, "missing.spriteatlas", Platform::Desktop).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_inspector_state_round_trips_through_the_file() {
        // Everything the UI edits is plain data in the file: a quest override and a variant survive save + load.
        let root = project("rt");
        let mut f = SpriteAtlasFile::default();
        f.variants.push(AtlasVariant { name: "Half".into(), scale: 0.5 });
        f.quest = Some(AtlasPlatform { max_size: Some(1024), variant: Some("Half".into()), ..Default::default() });
        f.packing = PackingMode::Tight;
        f.layout = PageLayout::Separate;
        f.save(&root.join("a.spriteatlas")).unwrap();
        assert_eq!(SpriteAtlasFile::load(&root.join("a.spriteatlas")).unwrap(), f);
        let eff = f.effective(Platform::Quest);
        assert_eq!((eff.max_size, eff.scale), (1024, 0.5));
        let _ = std::fs::remove_dir_all(&root);
    }
}
