//! The text system: font loading and stacks, shaped-run and layout caches, the
//! glyph atlases, and conversion of layouts into GPU instance lists.
//!
//! One `TextSystem` serves every UI canvas and 3D text in the process. Nothing
//! here touches Vulkan; the renderer drains [`TextSystem::take_uploads`] and
//! draws the [`TextBlock`] instance lists it is handed.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::atlas::{Atlas, AtlasConfig, AtlasFormat, AtlasUpload, GlyphKey, SlotId, SlotRect};
use crate::font::{FontFace, FontId, FontRegistry};
use crate::gpu::{FLAG_ITALIC, GlyphInstance, pack_fx, pack_meta, pack_rgba};
use crate::layout::{self, LayoutParams, PlacedGlyph, ShapeEnv, ShapedText, TextLayout};
use crate::markup::{self, Rgba};
use crate::sdf::{self, SdfMode};
use crate::shape::Shaper;
use crate::stack::{FontAsset, FontMode, FontStack, GlyphKind, MTSDF_MAX_GLYPHS, default_raster};

/// Reads font and stack files (project-relative paths).
pub trait FontLoader: Send {
    fn load(&mut self, path: &str) -> Option<Vec<u8>>;
}

/// Loads from a directory (the project's asset root).
pub struct FsLoader {
    pub root: PathBuf,
}

impl FontLoader for FsLoader {
    fn load(&mut self, path: &str) -> Option<Vec<u8>> {
        let p = std::path::Path::new(path);
        let full = if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) };
        std::fs::read(full).ok()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TextConfig {
    /// MTSDF and colour glyphs.
    pub atlas_rgba: AtlasConfig,
    /// Single-channel SDF glyphs.
    pub atlas_r8: AtlasConfig,
    /// Rasterization time allowed per frame; text with glyphs still pending is
    /// drawn without them and completed on following frames.
    pub glyph_budget: Duration,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            atlas_rgba: AtlasConfig { page_size: 1024, max_pages: 4, guard_frames: 4 },
            atlas_r8: AtlasConfig { page_size: 1024, max_pages: 4, guard_frames: 4 },
            glyph_budget: Duration::from_millis(4),
        }
    }
}

/// Per-face raster parameters.
#[derive(Clone, Copy, Debug)]
struct FontMeta {
    kind: GlyphKind,
    em_px: f32,
    spread: f32,
    bold_dilate: f32,
}

#[derive(Clone, Copy, Debug)]
enum Geom {
    Empty,
    Rect { left: f32, top: f32 },
}

/// Effects baked into instances (outline, softness).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Effects {
    /// Outline width in em (0 = none).
    pub outline_em: f32,
    pub outline_color: Rgba,
    /// Edge softness in em.
    pub softness_em: f32,
    /// Extra face dilation in em (negative thins).
    pub dilate_em: f32,
}

impl Default for Effects {
    fn default() -> Self {
        Self { outline_em: 0.0, outline_color: [0, 0, 0, 255], softness_em: 0.0, dilate_em: 0.0 }
    }
}

/// A text layout request. Everything but the draw tint (the text colour)
/// participates in the cache key, so recolouring never relays out.
#[derive(Clone, Copy)]
pub struct TextRequest<'a> {
    pub text: &'a str,
    /// Font file / `.zfont` / `.zfontstack`; empty = the built-in stack.
    pub font: &'a str,
    pub rich: bool,
    pub params: LayoutParams,
    pub effects: Effects,
    /// Force the base direction.
    pub rtl: Option<bool>,
}

/// A laid-out, atlas-resolved text ready to draw.
pub struct TextBlock {
    pub instances: Arc<[GlyphInstance]>,
    pub size: f32,
    pub width: f32,
    pub height: f32,
    pub lines: u32,
    pub truncated: bool,
    pub page_count: u32,
    pub clip: Option<[f32; 4]>,
    /// False while some glyphs are still being rasterized (budget / atlas busy).
    pub complete: bool,
    pub links: Vec<String>,
    slots: Vec<(bool, SlotId)>,
    epoch: (u64, u64),
}

struct CachedBlock {
    block: Arc<TextBlock>,
    used: u64,
}

pub struct TextSystem {
    cfg: TextConfig,
    fonts: FontRegistry,
    metas: Vec<FontMeta>,
    shaper: Shaper,
    atlas_rgba: Atlas,
    atlas_r8: Atlas,
    loader: Option<Box<dyn FontLoader>>,
    fs_root: Option<PathBuf>,
    /// Built-in primary + fallbacks, appended to every chain.
    default_chain: Vec<FontId>,
    chains: HashMap<u64, Arc<Vec<FontId>>>,
    failed: HashSet<String>,
    geoms: HashMap<GlyphKey, Geom>,
    shaped: HashMap<u64, (Arc<ShapedText>, u64)>,
    blocks: HashMap<u64, CachedBlock>,
    lang: String,
    /// `<style=name>` definitions, and a counter that keys cached text on them.
    styles: markup::StyleSheet,
    style_gen: u64,
    /// Glyphs queued by `warm_language`, rasterized inside the per-frame budget.
    warm: std::collections::VecDeque<(FontId, u32)>,
    warmed: HashSet<(String, u64)>,
    frame: u64,
    /// Rasterization time spent since the last `next_frame`.
    spent: Duration,
    generated_this_frame: u32,
    pub glyphs_generated: u64,
    pub layouts_built: u64,
}

fn hash_of(f: impl FnOnce(&mut std::collections::hash_map::DefaultHasher)) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    f(&mut h);
    h.finish()
}

impl TextSystem {
    pub fn new(cfg: TextConfig) -> Self {
        Self {
            cfg,
            fonts: FontRegistry::default(),
            metas: Vec::new(),
            shaper: Shaper::default(),
            atlas_rgba: Atlas::new(AtlasFormat::Rgba8, cfg.atlas_rgba),
            atlas_r8: Atlas::new(AtlasFormat::R8, cfg.atlas_r8),
            loader: None,
            fs_root: None,
            default_chain: Vec::new(),
            chains: HashMap::new(),
            failed: HashSet::new(),
            geoms: HashMap::new(),
            shaped: HashMap::new(),
            blocks: HashMap::new(),
            lang: String::new(),
            styles: markup::StyleSheet::default(),
            style_gen: 0,
            warm: Default::default(),
            warmed: HashSet::new(),
            frame: 0,
            spent: Duration::ZERO,
            generated_this_frame: 0,
            glyphs_generated: 0,
            layouts_built: 0,
        }
    }

    pub fn config(&self) -> TextConfig {
        self.cfg
    }

    pub fn set_loader(&mut self, loader: Box<dyn FontLoader>) {
        self.loader = Some(loader);
        self.chains.clear();
        self.failed.clear();
    }

    /// Load project fonts from `root`; the loader is only rebuilt when the
    /// root changes (callers may repeat this every frame).
    pub fn set_fs_root(&mut self, root: &std::path::Path, make: impl FnOnce(&std::path::Path) -> Box<dyn FontLoader>) {
        if self.fs_root.as_deref() != Some(root) {
            self.fs_root = Some(root.to_path_buf());
            self.set_loader(make(root));
        }
    }

    /// Register a built-in font. The first is the primary; the rest are
    /// fallbacks appended to every chain.
    pub fn add_builtin_font(&mut self, name: &str, data: Vec<u8>) -> Option<FontId> {
        let face = FontFace::new(name, data, 0)?;
        let id = self.register(face, None);
        if !self.default_chain.contains(&id) {
            self.default_chain.push(id);
        }
        self.chains.clear();
        Some(id)
    }

    pub fn has_fonts(&self) -> bool {
        !self.default_chain.is_empty()
    }

    /// Active language (BCP-47) from Localization: picks locale overrides in
    /// font stacks and tags shaping.
    pub fn set_language(&mut self, lang: &str) {
        if self.lang != lang {
            self.lang = lang.to_owned();
        }
    }

    pub fn language(&self) -> &str {
        &self.lang
    }

    /// Replace the `<style=name>` sheet; cached text built with the old one is rebuilt.
    pub fn set_styles(&mut self, styles: markup::StyleSheet) {
        if self.styles != styles {
            self.styles = styles;
            self.style_gen += 1;
        }
    }

    pub fn styles(&self) -> &markup::StyleSheet {
        &self.styles
    }

    /// Queue the glyphs of `chars` (a language's character set, see
    /// `Localization::used_chars`) on the first face of `font_key`'s chain, so switching
    /// language does not hitch on first use. They are rasterized inside the per-frame glyph
    /// budget by [`Self::next_frame`] and are not pinned (the LRU may still evict them).
    /// `stamp` identifies the character set: asking again with the same `(lang, stamp)` does
    /// nothing, and `chars` is only evaluated when it is new. Returns the glyphs queued.
    pub fn warm_language(&mut self, font_key: &str, lang: &str, stamp: u64, chars: impl FnOnce() -> String) -> usize {
        if !self.warmed.insert((format!("{font_key}|{lang}"), stamp)) {
            return 0;
        }
        let chain = self.chain_for(font_key);
        let Some(&font) = chain.first() else { return 0 };
        let Some(face) = self.fonts.get(font) else { return 0 };
        let mut gids: Vec<u32> = chars().chars().filter(|&c| face.has_char(c)).filter_map(|c| face.glyph_for(c)).collect();
        gids.sort_unstable();
        gids.dedup();
        let n = gids.len();
        self.warm.extend(gids.into_iter().map(|g| (font, g)));
        n
    }

    /// Glyphs still waiting from `warm_language`.
    pub fn warm_pending(&self) -> usize {
        self.warm.len()
    }

    fn drain_warm(&mut self) {
        while let Some(&(font, gid)) = self.warm.front() {
            match self.glyph(font, gid, false) {
                Glyph::Pending if self.spent >= self.cfg.glyph_budget => break,
                // An atlas with no room left also reports Pending: stop warming rather than churn it.
                Glyph::Pending => {
                    self.warm.clear();
                    break;
                }
                _ => {
                    self.warm.pop_front();
                }
            }
        }
    }

    /// Call once per rendered frame (the renderer does, after recording): ages
    /// the LRU and resets the rasterization budget.
    pub fn next_frame(&mut self) {
        self.frame += 1;
        self.atlas_rgba.begin_frame();
        self.atlas_r8.begin_frame();
        self.spent = Duration::ZERO;
        self.generated_this_frame = 0;
        self.drain_warm();
        if self.frame.is_multiple_of(64) {
            let cutoff = self.frame.saturating_sub(600);
            if self.blocks.len() > 256 {
                self.blocks.retain(|_, b| b.used >= cutoff);
            }
            if self.shaped.len() > 256 {
                self.shaped.retain(|_, s| s.1 >= cutoff);
            }
        }
    }

    /// Rects the GPU atlas images still need (both formats).
    pub fn take_uploads(&mut self) -> Vec<AtlasUpload> {
        let mut v = self.atlas_rgba.take_uploads();
        v.extend(self.atlas_r8.take_uploads());
        v
    }

    pub fn pending_uploads(&self) -> usize {
        self.atlas_rgba.pending_uploads() + self.atlas_r8.pending_uploads()
    }

    /// Drop all cached layouts and glyphs (device lost, fonts changed on disk).
    pub fn clear_caches(&mut self) {
        self.atlas_rgba.clear();
        self.atlas_r8.clear();
        self.geoms.clear();
        self.blocks.clear();
        self.shaped.clear();
        self.shaper.clear();
        self.chains.clear();
        self.failed.clear();
    }

    pub fn atlas_stats(&self) -> AtlasStats {
        AtlasStats {
            rgba_pages: self.atlas_rgba.page_count(),
            r8_pages: self.atlas_r8.page_count(),
            glyphs: self.atlas_rgba.live_glyphs() + self.atlas_r8.live_glyphs(),
            evictions: self.atlas_rgba.eviction_count() + self.atlas_r8.eviction_count(),
        }
    }

    fn epoch(&self) -> (u64, u64) {
        (self.atlas_rgba.epoch(), self.atlas_r8.epoch())
    }

    fn register(&mut self, face: FontFace, asset: Option<&FontAsset>) -> FontId {
        let kind = match asset.map(|a| a.mode).unwrap_or(FontMode::Auto) {
            FontMode::Mtsdf => GlyphKind::Mtsdf,
            FontMode::Sdf => GlyphKind::Sdf,
            FontMode::Auto => {
                if face.glyph_count <= MTSDF_MAX_GLYPHS { GlyphKind::Mtsdf } else { GlyphKind::Sdf }
            }
        };
        let (dem, dspread) = default_raster(kind);
        let pick = |v: Option<f32>, d: f32| v.filter(|v| *v > 0.0).unwrap_or(d);
        let meta = FontMeta {
            kind,
            em_px: pick(asset.map(|a| a.em_px), dem),
            spread: pick(asset.map(|a| a.spread_px), dspread),
            bold_dilate: asset.map(|a| a.bold_dilate).unwrap_or(0.02),
        };
        let before = self.fonts.len();
        let id = self.fonts.add(face);
        if self.fonts.len() > before {
            self.metas.push(meta);
        }
        id
    }

    /// Load (or fetch) one stack entry: a font file, a `.zfont`, or "" for the
    /// built-in primary.
    fn load_entry(&mut self, path: &str) -> Option<FontId> {
        if path.is_empty() {
            return self.default_chain.first().copied();
        }
        if let Some(id) = self.fonts.find(path) {
            return Some(id);
        }
        if self.failed.contains(path) {
            return None;
        }
        let result = self.load_entry_uncached(path);
        if result.is_none() {
            tracing::warn!("text: could not load font {path:?}");
            self.failed.insert(path.to_owned());
        }
        result
    }

    fn load_entry_uncached(&mut self, path: &str) -> Option<FontId> {
        let loader = self.loader.as_mut()?;
        if path.ends_with(".zfont") {
            let asset = FontAsset::from_ron(&String::from_utf8(loader.load(path)?).ok()?).ok()?;
            // `source` is relative to the .zfont's folder.
            let dir = path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            let src = if dir.is_empty() { asset.source.clone() } else { format!("{dir}/{}", asset.source) };
            let data = loader.load(&src)?;
            let mut face = FontFace::new(path, data, asset.face_index)?;
            face.name = path.to_owned();
            let id = self.register(face, Some(&asset));
            let chars: Vec<char> = asset.prewarm.iter().flat_map(|c| c.chars()).collect();
            self.prewarm_chars(id, &chars);
            Some(id)
        } else {
            let data = loader.load(path)?;
            let face = FontFace::new(path, data, 0)?;
            Some(self.register(face, None))
        }
    }

    /// Font chain for a font key under the active language.
    fn chain_for(&mut self, key: &str) -> Arc<Vec<FontId>> {
        let ck = hash_of(|h| {
            key.hash(h);
            self.lang.hash(h);
        });
        if let Some(c) = self.chains.get(&ck) {
            return c.clone();
        }
        let mut chain: Vec<FontId> = Vec::new();
        let push = |id: Option<FontId>, chain: &mut Vec<FontId>| {
            if let Some(id) = id
                && !chain.contains(&id)
            {
                chain.push(id);
            }
        };
        if key.ends_with(".zfontstack") {
            let stack = self
                .loader
                .as_mut()
                .and_then(|l| l.load(key))
                .and_then(|b| String::from_utf8(b).ok())
                .and_then(|s| FontStack::from_ron(&s).ok());
            match stack {
                Some(stack) => {
                    for entry in stack.chain(&self.lang) {
                        let id = self.load_entry(&entry);
                        push(id, &mut chain);
                    }
                }
                None => tracing::warn!("text: could not read font stack {key:?}"),
            }
        } else if !key.is_empty() {
            let id = self.load_entry(key);
            push(id, &mut chain);
        }
        for id in self.default_chain.clone() {
            push(Some(id), &mut chain);
        }
        let chain = Arc::new(chain);
        self.chains.insert(ck, chain.clone());
        chain
    }

    /// Rasterize `chars` of a face ahead of time and pin them.
    pub fn prewarm_chars(&mut self, font: FontId, chars: &[char]) {
        for &c in chars {
            let Some(gid) = self.fonts.get(font).and_then(|f| f.glyph_for(c)) else { continue };
            if let Glyph::Ready { slot, is_r8, .. } = self.glyph(font, gid, true) {
                if is_r8 { self.atlas_r8.pin(slot) } else { self.atlas_rgba.pin(slot) }
            }
        }
    }

    /// Prewarm by font key (path or empty for built-in).
    pub fn prewarm(&mut self, font_key: &str, chars: &[char]) {
        let chain = self.chain_for(font_key);
        if let Some(&id) = chain.first() {
            self.prewarm_chars(id, chars);
        }
    }

    fn request_key(&self, r: &TextRequest<'_>) -> u64 {
        hash_of(|h| {
            r.text.hash(h);
            r.font.hash(h);
            r.rich.hash(h);
            r.rtl.hash(h);
            self.lang.hash(h);
            self.style_gen.hash(h);
            let p = &r.params;
            for v in [
                p.size, p.box_size[0], p.box_size[1], p.char_spacing, p.word_spacing, p.line_spacing,
                p.paragraph_spacing, p.margins[0], p.margins[1], p.margins[2], p.margins[3],
                r.effects.outline_em, r.effects.softness_em, r.effects.dilate_em,
            ] {
                v.to_bits().hash(h);
            }
            (p.wrap, p.halign, p.valign, p.overflow, p.bold, p.italic, p.page).hash(h);
            p.auto_size.map(|(a, b)| (a.to_bits(), b.to_bits())).hash(h);
            r.effects.outline_color.hash(h);
        })
    }

    fn shaped_for(&mut self, r: &TextRequest<'_>) -> Option<Arc<ShapedText>> {
        let key = hash_of(|h| {
            r.text.hash(h);
            r.font.hash(h);
            r.rich.hash(h);
            r.rtl.hash(h);
            self.lang.hash(h);
            self.style_gen.hash(h);
        });
        if let Some((s, used)) = self.shaped.get_mut(&key) {
            *used = self.frame;
            return Some(s.clone());
        }
        let chain = self.chain_for(r.font);
        if chain.is_empty() {
            return None;
        }
        let parsed = markup::parse_with(r.text, r.rich, &self.styles);
        let lang = self.lang.clone();
        let mut env = ShapeEnv { fonts: &self.fonts, chain: &chain, shaper: &mut self.shaper, lang: &lang, rtl: r.rtl };
        let st = Arc::new(layout::shape_text(&mut env, parsed));
        self.shaped.insert(key, (st.clone(), self.frame));
        Some(st)
    }

    /// The laid-out, atlas-resolved text for `req`. Cached: asking again with
    /// the same request returns the same block without allocating. `None`
    /// when no usable font exists (callers fall back to the egui path).
    pub fn block(&mut self, req: &TextRequest<'_>) -> Option<Arc<TextBlock>> {
        let key = self.request_key(req);
        let epoch = self.epoch();
        if let Some(c) = self.blocks.get_mut(&key)
            && c.block.epoch == epoch
            && c.block.complete
        {
            c.used = self.frame;
            let b = c.block.clone();
            for &(r8, id) in &b.slots {
                if r8 { self.atlas_r8.touch(id) } else { self.atlas_rgba.touch(id) }
            }
            return Some(b);
        }
        let shaped = self.shaped_for(req)?;
        let lay = layout::layout(&shaped, &self.fonts, &req.params);
        self.layouts_built += 1;
        let block = Arc::new(self.build_block(&lay, &shaped, req));
        self.blocks.insert(key, CachedBlock { block: block.clone(), used: self.frame });
        Some(block)
    }

    fn build_block(&mut self, lay: &TextLayout, shaped: &ShapedText, req: &TextRequest<'_>) -> TextBlock {
        let mut inst: Vec<GlyphInstance> = Vec::with_capacity(lay.glyphs.len() + lay.under.len() + lay.over.len());
        let mut slots: Vec<(bool, SlotId)> = Vec::new();
        let mut complete = true;
        let solid = |r: &layout::DecorRect| GlyphInstance {
            pos: [r.x, r.y],
            size: [r.w, r.h],
            uv: [0; 4],
            color: pack_rgba(r.color),
            meta: pack_meta(0, GlyphKind::Solid, 0, 0.0),
            fx: 0,
            outline: 0,
        };
        inst.extend(lay.under.iter().map(solid));
        for g in &lay.glyphs {
            match self.glyph_instance(g, &req.effects) {
                GlyphOut::Instance(i, r8, slot) => {
                    inst.push(i);
                    if !slots.contains(&(r8, slot)) {
                        slots.push((r8, slot));
                    }
                }
                GlyphOut::Skip => {}
                GlyphOut::Pending => complete = false,
            }
        }
        inst.extend(lay.over.iter().map(solid));
        TextBlock {
            instances: inst.into(),
            size: lay.size,
            width: lay.width,
            height: lay.height,
            lines: lay.lines.len() as u32,
            truncated: lay.truncated,
            page_count: lay.page_count,
            clip: lay.clip,
            complete,
            links: shaped.parsed.links.clone(),
            slots,
            epoch: self.epoch(),
        }
    }

    fn glyph_instance(&mut self, g: &PlacedGlyph, fx: &Effects) -> GlyphOut {
        let (slot, rect, left, top, kind, spread, em_px, is_r8, bold_dilate) = match self.glyph(g.font, g.glyph, false) {
            Glyph::Empty => return GlyphOut::Skip,
            Glyph::Pending => return GlyphOut::Pending,
            Glyph::Ready { slot, rect, left, top, kind, spread, em_px, is_r8, bold_dilate } => {
                (slot, rect, left, top, kind, spread, em_px, is_r8, bold_dilate)
            }
        };
        let page = if is_r8 { self.cfg.atlas_r8.page_size } else { self.cfg.atlas_rgba.page_size } as f32;
        let s = g.size / em_px;
        let uvn = |v: u32| ((v as f32 / page) * 65535.0).round().clamp(0.0, 65535.0) as u16;
        // Normalized distance units: texels / (2 * spread).
        let norm = |em: f32| em * em_px / (2.0 * spread);
        let dilate = norm(fx.dilate_em) + if g.bold { norm(bold_dilate) } else { 0.0 };
        let flags = if g.italic { FLAG_ITALIC } else { 0 };
        let i = GlyphInstance {
            pos: [g.x + left * s, g.y - top * s],
            size: [rect.w as f32 * s, rect.h as f32 * s],
            uv: [uvn(rect.x), uvn(rect.y), uvn(rect.x + rect.w), uvn(rect.y + rect.h)],
            color: pack_rgba(g.color),
            meta: pack_meta(rect.layer, kind, flags, spread),
            fx: pack_fx(dilate, norm(fx.outline_em), norm(fx.softness_em)),
            outline: pack_rgba(fx.outline_color),
        };
        GlyphOut::Instance(i, is_r8, slot)
    }

    /// Resolve a glyph to atlas texels, rasterizing it on first use unless the
    /// frame's time budget is spent (`force` ignores the budget).
    fn glyph(&mut self, font: FontId, gid: u32, force: bool) -> Glyph {
        let Some(meta) = self.metas.get(font.0 as usize).copied() else { return Glyph::Empty };
        let key = GlyphKey { font: font.0, glyph: gid, kind: meta.kind as u8 };
        let is_r8 = meta.kind == GlyphKind::Sdf;
        let atlas = if is_r8 { &mut self.atlas_r8 } else { &mut self.atlas_rgba };
        if let Some(Geom::Empty) = self.geoms.get(&key) {
            return Glyph::Empty;
        }
        if let (Some((slot, rect)), Some(Geom::Rect { left, top })) = (atlas.get(&key), self.geoms.get(&key)) {
            return Glyph::Ready { slot, rect, left: *left, top: *top, kind: meta.kind, spread: meta.spread, em_px: meta.em_px, is_r8, bold_dilate: meta.bold_dilate };
        }
        if !force && self.generated_this_frame > 0 && self.spent >= self.cfg.glyph_budget {
            return Glyph::Pending;
        }
        let started = Instant::now();
        let Some(face) = self.fonts.get(font) else { return Glyph::Empty };
        let Some(mut shape) = face.outline(gid, meta.em_px) else {
            self.geoms.insert(key, Geom::Empty);
            return Glyph::Empty;
        };
        let mode = if meta.kind == GlyphKind::Sdf { SdfMode::Sdf } else { SdfMode::Mtsdf };
        if mode == SdfMode::Mtsdf {
            shape.colour_edges();
        }
        let Some(bm) = sdf::generate(&shape, meta.spread, mode) else {
            self.geoms.insert(key, Geom::Empty);
            return Glyph::Empty;
        };
        self.generated_this_frame += 1;
        self.glyphs_generated += 1;
        self.spent += started.elapsed();
        let atlas = if is_r8 { &mut self.atlas_r8 } else { &mut self.atlas_rgba };
        match atlas.insert(key, bm.width, bm.height, bm.data) {
            Some((slot, rect)) => {
                self.geoms.insert(key, Geom::Rect { left: bm.left, top: bm.top });
                Glyph::Ready { slot, rect, left: bm.left, top: bm.top, kind: meta.kind, spread: meta.spread, em_px: meta.em_px, is_r8, bold_dilate: meta.bold_dilate }
            }
            None => Glyph::Pending,
        }
    }

    /// Measure without keeping a block around (editor inspectors, layout code).
    pub fn measure(&mut self, req: &TextRequest<'_>) -> Option<(f32, f32)> {
        let shaped = self.shaped_for(req)?;
        let lay = layout::layout(&shaped, &self.fonts, &req.params);
        Some((lay.width, lay.height))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtlasStats {
    pub rgba_pages: usize,
    pub r8_pages: usize,
    pub glyphs: usize,
    pub evictions: u64,
}

enum Glyph {
    Empty,
    Pending,
    Ready {
        slot: SlotId,
        rect: SlotRect,
        left: f32,
        top: f32,
        kind: GlyphKind,
        spread: f32,
        em_px: f32,
        is_r8: bool,
        bold_dilate: f32,
    },
}

enum GlyphOut {
    Instance(GlyphInstance, bool, SlotId),
    Skip,
    Pending,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::HAlign;

    const PLEX: &[u8] = include_bytes!("../../zestengine-editor/assets/fonts/IBMPlexSans-Regular.ttf");
    const PLEX_MEDIUM: &[u8] = include_bytes!("../../zestengine-editor/assets/fonts/IBMPlexSans-Medium.ttf");

    struct MemLoader(HashMap<String, Vec<u8>>);
    impl FontLoader for MemLoader {
        fn load(&mut self, path: &str) -> Option<Vec<u8>> {
            self.0.get(path).cloned()
        }
    }

    fn system() -> TextSystem {
        let mut s = TextSystem::new(TextConfig { glyph_budget: Duration::from_secs(10), ..Default::default() });
        s.add_builtin_font("builtin", PLEX.to_vec()).unwrap();
        s.next_frame();
        s
    }

    fn req<'a>(text: &'a str) -> TextRequest<'a> {
        TextRequest {
            text,
            font: "",
            rich: false,
            params: LayoutParams { size: 24.0, box_size: [300.0, 100.0], ..Default::default() },
            effects: Effects::default(),
            rtl: None,
        }
    }

    #[test]
    fn block_has_one_instance_per_visible_glyph_and_queues_uploads() {
        let mut s = system();
        let b = s.block(&req("Hi there")).unwrap();
        assert!(b.complete);
        // Seven letters, one space (no instance).
        assert_eq!(b.instances.len(), 7);
        let ups = s.take_uploads();
        assert!(!ups.is_empty() && ups.iter().all(|u| u.format == AtlasFormat::Rgba8));
        assert!(b.instances.iter().all(|i| i.size[0] > 0.0 && i.size[1] > 0.0));
        // Bitmap includes the SDF padding: the quad is bigger than the ink.
        let h = &b.instances[0];
        assert!(h.size[1] > 24.0 * 0.6);
    }

    #[test]
    fn unchanged_request_hits_the_cache_without_relayout() {
        let mut s = system();
        let a = s.block(&req("cached")).unwrap();
        let built = s.layouts_built;
        s.next_frame();
        let b = s.block(&req("cached")).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(s.layouts_built, built);
        // Changing the box relays out but reuses shaping and glyphs.
        let mut r = req("cached");
        r.params.box_size[0] = 50.0;
        let gens = s.glyphs_generated;
        let misses = s.shaper.misses;
        let c = s.block(&r).unwrap();
        assert_eq!(s.layouts_built, built + 1);
        assert_eq!(s.glyphs_generated, gens);
        assert_eq!(s.shaper.misses, misses);
        assert!(c.lines > a.lines);
    }

    #[test]
    fn glyphs_are_shared_across_texts_and_sizes() {
        let mut s = system();
        s.block(&req("abc")).unwrap();
        let gens = s.glyphs_generated;
        let mut r = req("cab");
        r.params.size = 90.0;
        s.block(&r).unwrap();
        assert_eq!(s.glyphs_generated, gens, "one MTSDF serves every size");
    }

    #[test]
    fn budget_defers_glyphs_and_completes_later() {
        let mut s = TextSystem::new(TextConfig { glyph_budget: Duration::ZERO, ..Default::default() });
        s.add_builtin_font("builtin", PLEX.to_vec()).unwrap();
        s.next_frame();
        let b = s.block(&req("abcdef")).unwrap();
        assert!(!b.complete);
        assert_eq!(b.instances.len(), 1, "at least one glyph per frame still makes progress");
        for _ in 0..8 {
            s.next_frame();
            s.block(&req("abcdef")).unwrap();
        }
        s.next_frame();
        let done = s.block(&req("abcdef")).unwrap();
        assert!(done.complete);
        assert_eq!(done.instances.len(), 6);
    }

    #[test]
    fn locale_stack_selects_fonts_and_falls_back() {
        let stack = FontStack {
            primary: "a.ttf".into(),
            fallbacks: vec![],
            locale_overrides: vec![crate::stack::LocaleFonts { lang: "fr".into(), fonts: vec!["b.ttf".into()] }],
        };
        let mut files = HashMap::new();
        files.insert("a.ttf".to_owned(), PLEX.to_vec());
        files.insert("b.ttf".to_owned(), PLEX_MEDIUM.to_vec());
        files.insert("s.zfontstack".to_owned(), stack.to_ron().into_bytes());
        let mut s = system();
        s.set_loader(Box::new(MemLoader(files)));
        let en = s.chain_for("s.zfontstack");
        s.set_language("fr-CA");
        let fr = s.chain_for("s.zfontstack");
        let name = |s: &TextSystem, c: &Arc<Vec<FontId>>| s.fonts.get(c[0]).unwrap().name.clone();
        assert_eq!(name(&s, &en), "a.ttf");
        assert_eq!(name(&s, &fr), "b.ttf");
        // The built-in font is always the last fallback.
        assert_eq!(s.fonts.get(*fr.last().unwrap()).unwrap().name, "builtin");
        // A missing file in a stack is skipped, not fatal.
        let mut bad = HashMap::new();
        bad.insert("m.zfontstack".to_owned(), FontStack { primary: "missing.ttf".into(), ..Default::default() }.to_ron().into_bytes());
        s.set_loader(Box::new(MemLoader(bad)));
        assert_eq!(s.chain_for("m.zfontstack").len(), 1);
    }

    #[test]
    fn zfont_asset_sets_raster_params_and_prewarms() {
        let asset = FontAsset { source: "p.ttf".into(), em_px: 32.0, spread_px: 4.0, mode: FontMode::Sdf, prewarm: vec![crate::stack::CharSet::Chars("ab".into())], ..Default::default() };
        let mut files = HashMap::new();
        files.insert("fonts/p.zfont".to_owned(), asset.to_ron().into_bytes());
        files.insert("fonts/p.ttf".to_owned(), PLEX.to_vec());
        let mut s = system();
        s.set_loader(Box::new(MemLoader(files)));
        let mut r = req("ab");
        r.font = "fonts/p.zfont";
        let b = s.block(&r).unwrap();
        assert_eq!(b.instances.len(), 2);
        let ups = s.take_uploads();
        assert!(ups.iter().all(|u| u.format == AtlasFormat::R8), "SDF mode uses the R8 atlas");
        let m = (b.instances[0].meta >> 16) & 0xff;
        assert_eq!(m, 32, "spread 4 px in 1/8 texel units");
        assert!(s.atlas_stats().glyphs >= 2);
    }

    #[test]
    fn missing_font_everywhere_returns_none() {
        let mut s = TextSystem::new(TextConfig::default());
        s.next_frame();
        assert!(s.block(&req("x")).is_none());
    }

    #[test]
    fn effects_and_bold_italic_land_in_instances() {
        let mut s = system();
        let mut r = req("<b><i>x</i></b>");
        r.rich = true;
        r.effects.outline_em = 0.05;
        r.effects.outline_color = [1, 2, 3, 255];
        let b = s.block(&r).unwrap();
        let i = &b.instances[0];
        assert_eq!((i.meta >> 10) & 0x3f, FLAG_ITALIC);
        let fx = i.fx;
        assert_ne!(fx & 0xff, 0, "bold dilation");
        assert_ne!((fx >> 8) & 0xff, 0, "outline width");
        assert_eq!(i.outline, pack_rgba([1, 2, 3, 255]));
        let plain = s.block(&req("x")).unwrap();
        assert_eq!(plain.instances[0].fx, 0);
    }

    #[test]
    fn decorations_become_solid_instances() {
        let mut s = system();
        let mut r = req("<u>ab</u>");
        r.rich = true;
        let b = s.block(&r).unwrap();
        let solids = b.instances.iter().filter(|i| (i.meta >> 8) & 3 == GlyphKind::Solid as u32).count();
        assert_eq!(solids, 1);
        assert_eq!(b.instances.len(), 3);
        // Solid rects come last (over the glyphs).
        assert_eq!((b.instances[2].meta >> 8) & 3, GlyphKind::Solid as u32);
    }

    #[test]
    fn eviction_invalidates_blocks_and_rebuilds() {
        let tiny = AtlasConfig { page_size: 128, max_pages: 1, guard_frames: 1 };
        let mut s = TextSystem::new(TextConfig { atlas_rgba: tiny, glyph_budget: Duration::from_secs(10), ..Default::default() });
        s.add_builtin_font("builtin", PLEX.to_vec()).unwrap();
        s.next_frame();
        let first = s.block(&req("abcd")).unwrap();
        assert!(first.complete);
        // Fill the 128 px page with other glyphs over several frames.
        for text in ["efgh", "ijkl", "mnop", "qrst", "uvwx"] {
            for _ in 0..4 {
                s.next_frame();
            }
            s.block(&req(text)).unwrap();
        }
        assert!(s.atlas_stats().evictions > 0);
        for _ in 0..4 {
            s.next_frame();
        }
        let again = s.block(&req("abcd")).unwrap();
        assert!(!Arc::ptr_eq(&first, &again), "block was rebuilt after eviction");
    }

    #[test]
    fn halign_changes_key() {
        let mut s = system();
        let a = s.block(&req("k")).unwrap();
        let mut r = req("k");
        r.params.halign = HAlign::Right;
        let b = s.block(&r).unwrap();
        assert!(b.instances[0].pos[0] > a.instances[0].pos[0] + 100.0);
    }

    #[test]
    fn changing_the_style_sheet_rebuilds_cached_text() {
        let mut sys = system();
        let rich = |text| TextRequest { rich: true, ..req(text) };
        let plain = sys.block(&rich("<style=hot>Hi</style>")).unwrap();
        let mut sheet = markup::StyleSheet::default();
        sheet.insert("hot", "<color=#ff0000>");
        sys.set_styles(sheet.clone());
        let styled = sys.block(&rich("<style=hot>Hi</style>")).unwrap();
        assert!(!Arc::ptr_eq(&plain, &styled), "a new sheet must not reuse the old block");
        assert_ne!(plain.instances[0].color, styled.instances[0].color, "the style colours the glyphs");
        let before = sys.layouts_built;
        sys.set_styles(sheet);
        sys.block(&rich("<style=hot>Hi</style>")).unwrap();
        assert_eq!(sys.layouts_built, before, "the same sheet again is a cache hit");
    }

    #[test]
    fn language_warming_spreads_glyph_generation_over_frames_and_runs_once() {
        let mut s = TextSystem::new(TextConfig { glyph_budget: Duration::ZERO, ..Default::default() });
        s.add_builtin_font("builtin", PLEX.to_vec()).unwrap();
        s.next_frame();
        let queued = s.warm_language("", "en", 1, || "abc abc 漢".to_owned());
        assert_eq!(queued, 4, "distinct covered glyphs only: a b c and the space (the uncovered CJK one is skipped)");
        assert_eq!(s.warm_pending(), 4);
        let mut frames = 0;
        while s.warm_pending() > 0 && frames < 20 {
            s.next_frame();
            frames += 1;
        }
        assert_eq!(s.warm_pending(), 0);
        assert!(frames >= 3, "a zero budget generates at most one glyph per frame, took {frames}");
        assert_eq!(s.glyphs_generated, 3, "a b c have outlines; the space does not");
        // The same stamp is a no-op and never evaluates the set; a new stamp queues again.
        assert_eq!(s.warm_language("", "en", 1, || panic!("must not run")), 0);
        assert_eq!(s.warm_language("", "en", 2, || "ab".to_owned()), 2);
        // Warmed glyphs serve the text without generating more.
        for _ in 0..4 {
            s.next_frame();
        }
        let before = s.glyphs_generated;
        s.cfg.glyph_budget = Duration::from_secs(10);
        s.block(&req("abc")).unwrap();
        assert_eq!(s.glyphs_generated, before, "abc was warmed");
    }
}
