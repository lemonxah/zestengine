//! Text layout: itemization (bidi, font fallback, script, rich-text runs),
//! shaping, UAX #14 line breaking, alignment, overflow and auto-size.
//!
//! Layout is split in two so auto-size is cheap: `shape_text` runs once and
//! produces size-independent atoms (one per shaped cluster, advances in em);
//! `layout` then only re-runs line breaking and placement over those cached
//! advances for each candidate size.

use unicode_bidi::{BidiInfo, Level};
use unicode_script::{Script, UnicodeScript};

use crate::font::{FontId, FontRegistry};
use crate::markup::{Parsed, Rgba};
use crate::shape::Shaper;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum HAlign {
    /// Start of the paragraph's direction.
    #[default]
    Start,
    End,
    Left,
    Center,
    Right,
    /// Fills the line (the last line of a paragraph stays ragged).
    Justified,
    /// Fills every line including the last.
    Flush,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum VAlign {
    #[default]
    Top,
    Middle,
    Bottom,
    /// The first line's baseline sits on the box's vertical centre line.
    Baseline,
    /// The middle of the first line's x-height sits on the centre line.
    Midline,
    /// The first line's cap height (top of capitals) sits on the centre line.
    Capline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Overflow {
    /// Draw everything, outside the box if needed.
    #[default]
    Overflow,
    /// Cut off what does not fit and end with "...".
    Ellipsis,
    /// Cut off what does not fit.
    Truncate,
    /// Draw everything but clip to the box.
    Masking,
    /// Show page `page` of the lines that fit the box height.
    Page,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutParams {
    /// Base font size in layout px.
    pub size: f32,
    /// Box size in layout px; a non-positive height means unbounded.
    pub box_size: [f32; 2],
    pub wrap: bool,
    pub halign: HAlign,
    pub valign: VAlign,
    pub overflow: Overflow,
    /// `(min, max)` size range; the largest size that fits wins.
    pub auto_size: Option<(f32, f32)>,
    /// All spacings in em/100 (like TextMeshPro).
    pub char_spacing: f32,
    pub word_spacing: f32,
    pub line_spacing: f32,
    pub paragraph_spacing: f32,
    /// `[left, top, right, bottom]` in px.
    pub margins: [f32; 4],
    pub bold: bool,
    pub italic: bool,
    pub page: u32,
}

impl Default for LayoutParams {
    fn default() -> Self {
        Self {
            size: 32.0,
            box_size: [0.0, 0.0],
            wrap: true,
            halign: HAlign::Start,
            valign: VAlign::Top,
            overflow: Overflow::Overflow,
            auto_size: None,
            char_spacing: 0.0,
            word_spacing: 0.0,
            line_spacing: 0.0,
            paragraph_spacing: 0.0,
            margins: [0.0; 4],
            bold: false,
            italic: false,
            page: 0,
        }
    }
}

/// A shaped glyph in em units (relative to its atom's pen position).
#[derive(Clone, Copy, Debug)]
pub struct PGlyph {
    pub glyph: u32,
    pub x_off: f32,
    pub y_off: f32,
    pub x_adv: f32,
}

/// One shaped cluster: the unit of line breaking and placement.
#[derive(Clone, Debug)]
pub struct Atom {
    pub font: FontId,
    /// Index into `Parsed::runs`.
    pub run: u32,
    /// Byte offset of the cluster in the whole plain text.
    pub byte: u32,
    pub level: u8,
    /// Advance in em.
    pub adv: f32,
    pub glyphs: std::ops::Range<u32>,
    pub break_before: bool,
    pub space: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Paragraph {
    pub atoms: Vec<Atom>,
    pub glyphs: Vec<PGlyph>,
    pub rtl: bool,
    /// Font for the line height of an empty paragraph.
    pub font: Option<FontId>,
    pub run: u32,
}

/// The "..." appended by Ellipsis overflow.
#[derive(Clone, Debug)]
pub struct Ellipsis {
    pub font: FontId,
    pub glyphs: Vec<PGlyph>,
    pub adv: f32,
}

/// Size-independent shaping result; cache this per (text, fonts, language).
#[derive(Clone, Debug, Default)]
pub struct ShapedText {
    pub parsed: Parsed,
    pub paragraphs: Vec<Paragraph>,
    pub ellipsis: Option<Ellipsis>,
}

/// Everything shaping needs.
pub struct ShapeEnv<'a> {
    pub fonts: &'a FontRegistry,
    /// Fallback chain for the active locale, best first (never empty).
    pub chain: &'a [FontId],
    pub shaper: &'a mut Shaper,
    pub lang: &'a str,
    /// Force the base direction (None = per paragraph, from the text).
    pub rtl: Option<bool>,
}

fn is_extend(c: char) -> bool {
    matches!(c as u32, 0x200c | 0x200d | 0xfe00..=0xfe0f | 0xe0100..=0xe01ef | 0x1f3fb..=0x1f3ff) || c.script() == Script::Inherited
}

fn is_hang_space(c: char) -> bool {
    c.is_whitespace() && !matches!(c, '\u{a0}' | '\u{202f}' | '\u{2007}')
}

/// Shape `parsed` into paragraphs of atoms.
pub fn shape_text(env: &mut ShapeEnv<'_>, parsed: Parsed) -> ShapedText {
    let mut paragraphs = Vec::new();
    let text = parsed.text.clone();
    let mut start = 0usize;
    loop {
        let end = text[start..].find('\n').map(|i| start + i).unwrap_or(text.len());
        paragraphs.push(shape_paragraph(env, &parsed, start, end));
        if end >= text.len() {
            break;
        }
        start = end + 1;
    }
    let ellipsis = shape_ellipsis(env);
    ShapedText { parsed, paragraphs, ellipsis }
}

fn shape_ellipsis(env: &mut ShapeEnv<'_>) -> Option<Ellipsis> {
    ["\u{2026}", "..."].into_iter().find_map(|text| {
        let font = env.chain.iter().copied().find(|&f| {
            env.fonts.get(f).is_some_and(|face| text.chars().all(|c| face.has_char(c)))
        })?;
        let face = env.fonts.get(font)?;
        let g = env.shaper.shape(face, text, false, Script::Common, env.lang);
        let glyphs: Vec<PGlyph> = g
            .iter()
            .map(|g| PGlyph { glyph: g.glyph, x_off: g.x_offset, y_off: g.y_offset, x_adv: g.x_advance })
            .collect();
        let adv = glyphs.iter().map(|g| g.x_adv).sum();
        Some(Ellipsis { font, glyphs, adv })
    })
}

fn shape_paragraph(env: &mut ShapeEnv<'_>, parsed: &Parsed, s: usize, e: usize) -> Paragraph {
    let ptxt = &parsed.text[s..e];
    let mut para = Paragraph {
        font: env.chain.first().copied(),
        run: parsed.run_at(s as u32) as u32,
        ..Default::default()
    };
    if ptxt.is_empty() {
        return para;
    }
    // Bidi levels per byte. Pure left-to-right text skips the algorithm.
    let needs_bidi = env.rtl == Some(true) || ptxt.chars().any(|c| c as u32 >= 0x590);
    let (levels, base_rtl): (Vec<Level>, bool) = if needs_bidi {
        let base = env.rtl.map(|r| if r { Level::rtl() } else { Level::ltr() });
        let info = BidiInfo::new(ptxt, base);
        let rtl = info.paragraphs.first().is_some_and(|p| p.level.is_rtl());
        (info.levels, rtl)
    } else {
        (Vec::new(), false)
    };
    para.rtl = base_rtl;
    let level_at = |i: usize| -> u8 { levels.get(i).map(|l| l.number()).unwrap_or(0) };

    // Per-char script, resolving Common/Inherited from context.
    let chars: Vec<(usize, char)> = ptxt.char_indices().collect();
    let mut scripts: Vec<Script> = chars.iter().map(|(_, c)| c.script()).collect();
    let weak = |s: Script| matches!(s, Script::Common | Script::Inherited | Script::Unknown);
    let mut prev = None;
    for sc in scripts.iter_mut() {
        if weak(*sc) {
            if let Some(p) = prev {
                *sc = p;
            }
        } else {
            prev = Some(*sc);
        }
    }
    let mut next = None;
    for sc in scripts.iter_mut().rev() {
        if weak(*sc) {
            *sc = next.unwrap_or(Script::Latin);
        } else {
            next = Some(*sc);
        }
    }

    // Font per char: previous font when it covers weak characters (spaces,
    // punctuation, combining marks), else the first font in the chain with a glyph.
    let mut fonts_of: Vec<FontId> = Vec::with_capacity(chars.len());
    let mut prev_font: Option<FontId> = None;
    for &(bi, c) in &chars {
        let covers = |f: FontId| env.fonts.get(f).is_some_and(|face| face.has_char(c));
        // `<font="name">` wins when that face has the character.
        let named = parsed.runs[parsed.run_at((s + bi) as u32)].attrs.font.as_deref().and_then(|n| env.fonts.find_loose(n)).filter(|&f| covers(f));
        let pick = match prev_font {
            _ if named.is_some() => named.unwrap(),
            Some(p) if is_extend(c) || (c.script() == Script::Common && covers(p)) => p,
            _ => env.chain.iter().copied().find(|&f| covers(f)).or(prev_font).unwrap_or(env.chain[0]),
        };
        fonts_of.push(pick);
        prev_font = Some(pick);
    }

    // Break opportunities (UAX #14), by byte offset within the paragraph.
    let mut allowed = vec![false; ptxt.len() + 1];
    for (off, op) in unicode_linebreak::linebreaks(ptxt) {
        if op == unicode_linebreak::BreakOpportunity::Allowed {
            allowed[off] = true;
        }
    }

    // Pieces: maximal spans with one (level, run, font, script).
    let mut i = 0;
    while i < chars.len() {
        let (b0, _) = chars[i];
        let level = level_at(b0);
        let run = parsed.run_at((s + b0) as u32);
        let (font, script) = (fonts_of[i], scripts[i]);
        let mut j = i + 1;
        while j < chars.len() {
            let (bj, _) = chars[j];
            if level_at(bj) != level || fonts_of[j] != font || scripts[j] != script || parsed.run_at((s + bj) as u32) != run {
                break;
            }
            j += 1;
        }
        let b1 = if j < chars.len() { chars[j].0 } else { ptxt.len() };
        let face = env.fonts.get(font);
        if let Some(face) = face {
            let rtl = level % 2 == 1;
            let shaped = env.shaper.shape(face, &ptxt[b0..b1], rtl, script, env.lang);
            // Logical order for breaking: harfbuzz hands RTL runs back visual.
            let logical: Vec<_> = if rtl { shaped.iter().rev().copied().collect() } else { shaped.to_vec() };
            let mut k = 0;
            while k < logical.len() {
                let cl = logical[k].cluster;
                let g0 = para.glyphs.len() as u32;
                let mut adv = 0.0;
                while k < logical.len() && logical[k].cluster == cl {
                    let g = logical[k];
                    para.glyphs.push(PGlyph { glyph: g.glyph, x_off: g.x_offset, y_off: g.y_offset, x_adv: g.x_advance });
                    adv += g.x_advance;
                    k += 1;
                }
                let byte_in_para = b0 + cl as usize;
                let abs = (s + byte_in_para) as u32;
                let ch = ptxt[byte_in_para..].chars().next().unwrap_or(' ');
                let r = &parsed.runs[run];
                let interior_nobr = r.attrs.nobr && abs > r.start;
                para.atoms.push(Atom {
                    font,
                    run: run as u32,
                    byte: abs,
                    level,
                    adv,
                    glyphs: g0..para.glyphs.len() as u32,
                    break_before: allowed[byte_in_para] && !interior_nobr,
                    space: is_hang_space(ch),
                });
            }
        }
        i = j;
    }
    para
}

/// A glyph with its final position (layout px, y down from the box top,
/// `y` is the baseline).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub font: FontId,
    pub glyph: u32,
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub color: Rgba,
    pub bold: bool,
    pub italic: bool,
    /// `Parsed::links` index or `u32::MAX`.
    pub link: u32,
    pub cluster: u32,
    pub line: u32,
}

/// A flat rectangle (underline, strikethrough, highlight).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecorRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub color: Rgba,
}

/// The box of a `<link>` on one line, in layout px (a link that wraps has one per line).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkRect {
    /// `Parsed::links` index.
    pub link: u32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub line: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineInfo {
    pub top: f32,
    pub baseline: f32,
    pub height: f32,
    pub width: f32,
    pub glyph_start: u32,
    pub glyph_end: u32,
}

#[derive(Clone, Debug, Default)]
pub struct TextLayout {
    pub glyphs: Vec<PlacedGlyph>,
    /// Drawn before the glyphs (highlights).
    pub under: Vec<DecorRect>,
    /// Drawn after the glyphs (underline, strikethrough).
    pub over: Vec<DecorRect>,
    pub lines: Vec<LineInfo>,
    /// Extents of the laid-out content in px.
    pub width: f32,
    pub height: f32,
    /// Size actually used (differs from the request under auto-size).
    pub size: f32,
    /// Lines or glyphs were cut off by Truncate / Ellipsis / Page.
    pub truncated: bool,
    pub page_count: u32,
    /// Clip rect `[x, y, w, h]` for Masking.
    pub clip: Option<[f32; 4]>,
    /// Where each `<link>` was placed (for hit-testing).
    pub links: Vec<LinkRect>,
}

impl TextLayout {
    /// The `Parsed::links` index under the point `(x, y)` (layout px, y down from the
    /// box top), if any. The topmost rect wins when links overlap (a later one).
    pub fn link_at(&self, x: f32, y: f32) -> Option<u32> {
        self.links.iter().rev().find(|r| x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h).map(|r| r.link)
    }

    /// Every rect of link `link` (to highlight a hovered link).
    pub fn link_rects(&self, link: u32) -> impl Iterator<Item = &LinkRect> {
        self.links.iter().filter(move |r| r.link == link)
    }
}

#[derive(Clone, Copy, Debug)]
struct Span {
    start: usize,
    end: usize,
    /// Width including trailing spaces, kept for debugging/measure tools.
    #[allow(dead_code)]
    width: f32,
    content: f32,
}

struct ParaPlan {
    adv: Vec<f32>,
    spans: Vec<Span>,
    /// Per span: (ascent, descent (negative), line height) in px.
    metrics: Vec<(f32, f32, f32)>,
}

struct Plan {
    paras: Vec<ParaPlan>,
    total_h: f32,
    max_content_w: f32,
    emergency: bool,
}

/// Left indent of the line `a` starts, in px.
fn atom_indent(st: &ShapedText, a: &Atom, size: f32, avail_w: f32) -> f32 {
    let attrs = &st.parsed.runs[a.run as usize].attrs;
    attrs.indent.map(|l| l.resolve(attrs.size(size), avail_w)).unwrap_or(0.0).max(0.0)
}

fn atom_size(st: &ShapedText, a: &Atom, size: f32) -> f32 {
    st.parsed.runs[a.run as usize].attrs.size(size)
}

/// Greedy line fill over cached advances. Spaces at a line end hang past the
/// edge; a word wider than the line breaks between clusters (`emergency`).
fn break_lines(atoms: &[Atom], adv: &[f32], max_w: f32, indent: &[f32]) -> (Vec<Span>, bool) {
    let mut spans = Vec::new();
    let mut emergency = false;
    let (mut start, mut width, mut content) = (0usize, 0.0f32, 0.0f32);
    let mut last_break: Option<(usize, f32, f32)> = None;
    let mut i = 0;
    while i < atoms.len() {
        let a = &atoms[i];
        if i > start && a.break_before {
            last_break = Some((i, width, content));
        }
        let new_w = width + adv[i];
        let eff = if a.space { content } else { new_w };
        // A line is as wide as the box minus the indent of the run it starts with.
        if eff > max_w - indent[start] && i > start {
            match last_break.take() {
                Some((bi, w, c)) => {
                    spans.push(Span { start, end: bi, width: w, content: c });
                    start = bi;
                    i = bi;
                }
                None => {
                    emergency = true;
                    spans.push(Span { start, end: i, width, content });
                    start = i;
                }
            }
            width = 0.0;
            content = 0.0;
            continue;
        }
        width = new_w;
        if !a.space {
            content = new_w;
        }
        i += 1;
    }
    spans.push(Span { start, end: atoms.len(), width, content });
    (spans, emergency)
}

fn plan(st: &ShapedText, fonts: &FontRegistry, p: &LayoutParams, size: f32) -> Plan {
    let avail_w = (p.box_size[0] - p.margins[0] - p.margins[2]).max(0.0);
    let max_w = if p.wrap && p.box_size[0] > 0.0 { avail_w } else { f32::INFINITY };
    let mut out = Plan { paras: Vec::new(), total_h: 0.0, max_content_w: 0.0, emergency: false };
    for para in &st.paragraphs {
        let adv: Vec<f32> = para
            .atoms
            .iter()
            .map(|a| {
                let r = &st.parsed.runs[a.run as usize].attrs;
                let sz = r.size(size);
                let mut w = a.adv * sz + (p.char_spacing / 100.0 + r.cspace_em) * sz;
                if a.space {
                    w += p.word_spacing / 100.0 * sz;
                }
                w
            })
            .collect();
        let indent: Vec<f32> = para.atoms.iter().map(|a| atom_indent(st, a, size, avail_w)).collect();
        let (spans, emergency) = break_lines(&para.atoms, &adv, max_w, &indent);
        out.emergency |= emergency;
        let mut metrics = Vec::with_capacity(spans.len());
        for sp in &spans {
            let (mut asc, mut desc, mut gap) = (0.0f32, 0.0f32, 0.0f32);
            if sp.start == sp.end {
                let sz = st.parsed.runs.get(para.run as usize).map(|r| r.attrs.size(size)).unwrap_or(size);
                if let Some(f) = para.font.and_then(|f| fonts.get(f)) {
                    asc = f.metrics.ascent * sz;
                    desc = f.metrics.descent * sz;
                    gap = f.metrics.line_gap * sz;
                }
            }
            for a in &para.atoms[sp.start..sp.end] {
                if let Some(f) = fonts.get(a.font) {
                    let sz = atom_size(st, a, size);
                    asc = asc.max(f.metrics.ascent * sz);
                    desc = desc.min(f.metrics.descent * sz);
                    gap = gap.max(f.metrics.line_gap * sz);
                }
            }
            let natural = asc - desc;
            // `<line-height>` replaces the font's natural line height (the tallest on the line wins).
            let tag_h = para.atoms[sp.start..sp.end]
                .iter()
                .filter_map(|a| {
                    let attrs = &st.parsed.runs[a.run as usize].attrs;
                    let sz = attrs.size(size);
                    attrs.line_height.map(|l| l.resolve(sz, sz))
                })
                .fold(None, |m: Option<f32>, h| Some(m.map_or(h, |m| m.max(h))));
            let height = tag_h.map(|h| h.max(0.0)).unwrap_or(natural + gap) * (1.0 + p.line_spacing / 100.0);
            out.total_h += height;
            out.max_content_w = out.max_content_w.max(sp.content);
            metrics.push((asc, desc, height));
        }
        out.total_h += p.paragraph_spacing / 100.0 * size;
        out.paras.push(ParaPlan { adv, spans, metrics });
    }
    out
}

fn fits(plan: &Plan, p: &LayoutParams) -> bool {
    let avail_h = p.box_size[1] - p.margins[1] - p.margins[3];
    let avail_w = p.box_size[0] - p.margins[0] - p.margins[2];
    if plan.emergency {
        return false;
    }
    if p.box_size[1] > 0.0 && plan.total_h > avail_h + 0.01 {
        return false;
    }
    p.wrap || p.box_size[0] <= 0.0 || plan.max_content_w <= avail_w + 0.01
}

/// Largest size in `[min, max]` that fits the box, found by bisection over
/// cached advances (no reshaping).
pub fn fit_size(st: &ShapedText, fonts: &FontRegistry, p: &LayoutParams, min: f32, max: f32) -> f32 {
    let (min, max) = (min.min(max).max(0.5), max.max(min).max(0.5));
    if fits(&plan(st, fonts, p, max), p) {
        return max;
    }
    let (mut lo, mut hi) = (min, max);
    for _ in 0..16 {
        if hi - lo <= 0.05 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if fits(&plan(st, fonts, p, mid), p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Visual order of `levels` (UAX #9 rule L2) as indices into the logical line.
pub fn visual_order(levels: &[u8]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..levels.len()).collect();
    let (Some(&max), Some(min_odd)) = (levels.iter().max(), levels.iter().copied().filter(|l| l % 2 == 1).min()) else {
        return order;
    };
    let mut lvl = max;
    while lvl >= min_odd {
        let mut i = 0;
        while i < order.len() {
            if levels[order[i]] >= lvl {
                let mut j = i;
                while j < order.len() && levels[order[j]] >= lvl {
                    j += 1;
                }
                order[i..j].reverse();
                i = j;
            } else {
                i += 1;
            }
        }
        if lvl == 0 {
            break;
        }
        lvl -= 1;
    }
    order
}

/// One placeable element of a line (an atom, or the ellipsis).
struct Item {
    font: FontId,
    run: usize,
    level: u8,
    byte: u32,
    adv: f32,
    size: f32,
    space: bool,
    glyphs: std::ops::Range<u32>,
    ellipsis: bool,
}

/// Lay `st` out in the box of `p`.
pub fn layout(st: &ShapedText, fonts: &FontRegistry, p: &LayoutParams) -> TextLayout {
    let size = match p.auto_size {
        Some((lo, hi)) => fit_size(st, fonts, p, lo, hi),
        None => p.size,
    };
    layout_at(st, fonts, p, size)
}

fn layout_at(st: &ShapedText, fonts: &FontRegistry, p: &LayoutParams, size: f32) -> TextLayout {
    let pl = plan(st, fonts, p, size);
    let avail_w = (p.box_size[0] - p.margins[0] - p.margins[2]).max(0.0);
    let bounded_w = p.box_size[0] > 0.0;
    let bounded_h = p.box_size[1] > 0.0;
    let avail_h = (p.box_size[1] - p.margins[1] - p.margins[3]).max(0.0);
    let mut out = TextLayout { size, page_count: 1, ..Default::default() };

    // Flatten (paragraph, span) pairs to decide which lines survive.
    struct LineRef {
        para: usize,
        span: usize,
        height: f32,
        /// Last line of its paragraph.
        last: bool,
    }
    let mut lines: Vec<LineRef> = Vec::new();
    for (pi, pp) in pl.paras.iter().enumerate() {
        for si in 0..pp.spans.len() {
            lines.push(LineRef { para: pi, span: si, height: pp.metrics[si].2, last: si + 1 == pp.spans.len() });
        }
    }
    let para_gap = p.paragraph_spacing / 100.0 * size;

    // Lines that survive vertical overflow.
    let (first, count, cut_after) = if bounded_h && p.wrap {
        match p.overflow {
            Overflow::Truncate | Overflow::Ellipsis => {
                let mut used = 0.0;
                let mut n = 0;
                for l in &lines {
                    let gap = if l.last { para_gap } else { 0.0 };
                    if used + l.height > avail_h + 0.5 && n > 0 {
                        break;
                    }
                    used += l.height + gap;
                    n += 1;
                }
                let cut = n < lines.len();
                out.truncated = cut;
                (0, n, cut)
            }
            Overflow::Page => {
                // Split into pages by height.
                let mut pages: Vec<(usize, usize)> = Vec::new();
                let (mut used, mut start) = (0.0, 0usize);
                for (i, l) in lines.iter().enumerate() {
                    if used + l.height > avail_h + 0.5 && i > start {
                        pages.push((start, i));
                        start = i;
                        used = 0.0;
                    }
                    used += l.height + if l.last { para_gap } else { 0.0 };
                }
                pages.push((start, lines.len()));
                out.page_count = pages.len() as u32;
                let (s, e) = pages[(p.page as usize).min(pages.len() - 1)];
                out.truncated = pages.len() > 1;
                (s, e - s, false)
            }
            _ => (0, lines.len(), false),
        }
    } else {
        (0, lines.len(), false)
    };
    if p.overflow == Overflow::Masking {
        out.clip = Some([p.margins[0], p.margins[1], avail_w, avail_h]);
        if !bounded_w || !bounded_h {
            out.clip = None;
        }
    }

    let visible = &lines[first..first + count];
    let total_h: f32 = visible.iter().map(|l| l.height + if l.last { para_gap } else { 0.0 }).sum();
    // Baseline / Midline / Capline: put a line of the first visible line on the box's centre.
    let first_line_anchor = |kind: VAlign| -> f32 {
        let Some(l) = visible.first() else { return 0.0 };
        let para = &st.paragraphs[l.para];
        let sp = pl.paras[l.para].spans[l.span];
        let (asc, desc, height) = pl.paras[l.para].metrics[l.span];
        let baseline = asc + (height - (asc - desc)) * 0.5;
        let (font, sz) = para
            .atoms
            .get(sp.start)
            .map(|a| (a.font, atom_size(st, a, size)))
            .or_else(|| para.font.map(|f| (f, size)))
            .unzip();
        let m = font.and_then(|f| fonts.get(f)).map(|f| f.metrics);
        let sz = sz.unwrap_or(size);
        match (kind, m) {
            (VAlign::Midline, Some(m)) => baseline - m.x_height * sz * 0.5,
            (VAlign::Capline, Some(m)) => baseline - m.cap_height * sz,
            _ => baseline,
        }
    };
    let y0 = p.margins[1]
        + if bounded_h {
            match p.valign {
                VAlign::Top => 0.0,
                VAlign::Middle => (avail_h - total_h) * 0.5,
                VAlign::Bottom => avail_h - total_h,
                k @ (VAlign::Baseline | VAlign::Midline | VAlign::Capline) => avail_h * 0.5 - first_line_anchor(k),
            }
        } else {
            0.0
        };

    let mut y = y0;
    let mut under_raw: Vec<(u32, DecorRect)> = Vec::new();
    let mut over_raw: Vec<(u32, DecorRect)> = Vec::new();
    let mut link_raw: Vec<LinkRect> = Vec::new();
    for (vi, l) in visible.iter().enumerate() {
        let para = &st.paragraphs[l.para];
        let pp = &pl.paras[l.para];
        let sp = pp.spans[l.span];
        let (asc, desc, height) = pp.metrics[l.span];
        let natural = asc - desc;
        let baseline = y + asc + (height - natural) * 0.5;
        let is_cut_line = cut_after && vi + 1 == visible.len();

        // Items in logical order, trimmed for Truncate / Ellipsis.
        let mut items: Vec<Item> = para.atoms[sp.start..sp.end]
            .iter()
            .enumerate()
            .map(|(k, a)| Item {
                font: a.font,
                run: a.run as usize,
                level: a.level,
                byte: a.byte,
                adv: pp.adv[sp.start + k],
                size: atom_size(st, a, size),
                space: a.space,
                glyphs: a.glyphs.clone(),
                ellipsis: false,
            })
            .collect();
        let content = |items: &[Item]| -> f32 {
            let end = items.iter().rposition(|i| !i.space).map(|i| i + 1).unwrap_or(0);
            items[..end].iter().map(|i| i.adv).sum()
        };
        let too_wide = bounded_w && content(&items) > avail_w + 0.01 && !p.wrap;
        let width_trim = too_wide && matches!(p.overflow, Overflow::Ellipsis | Overflow::Truncate);
        let want_ellipsis = (is_cut_line && p.overflow == Overflow::Ellipsis) || (width_trim && p.overflow == Overflow::Ellipsis);
        if width_trim || (want_ellipsis && bounded_w) {
            let ell_w = if want_ellipsis { st.ellipsis.as_ref().map(|e| e.adv * size).unwrap_or(0.0) } else { 0.0 };
            while items.len() > 1 && content(&items) + ell_w > avail_w + 0.01 {
                items.pop();
            }
            out.truncated = true;
        }
        if want_ellipsis && let Some(e) = &st.ellipsis {
            // Trailing spaces would sit before the dots: drop them.
            while items.last().is_some_and(|i| i.space) {
                items.pop();
            }
            let run = items.last().map(|i| i.run).unwrap_or(para.run as usize);
            items.push(Item {
                font: e.font,
                run,
                level: if para.rtl { 1 } else { 0 },
                byte: items.last().map(|i| i.byte).unwrap_or(0),
                adv: e.adv * size,
                size,
                space: false,
                glyphs: 0..0,
                ellipsis: true,
            });
        }

        // Alignment.
        let levels: Vec<u8> = items.iter().map(|i| i.level).collect();
        let order = visual_order(&levels);
        let line_content = content(&items);
        let line_total: f32 = items.iter().map(|i| i.adv).sum();
        // `<align>` and `<indent>` of the run each line starts with.
        let lead = items.iter().find(|i| !i.space && !i.ellipsis).or(items.first()).filter(|i| !i.ellipsis);
        let line_align = lead.and_then(|i| st.parsed.runs[i.run].attrs.align).unwrap_or(p.halign);
        let indent = lead
            .map(|i| st.parsed.runs[i.run].attrs.indent.map(|l| l.resolve(i.size, avail_w)).unwrap_or(0.0).max(0.0))
            .unwrap_or(0.0);
        let indent = if bounded_w { indent.min(avail_w) } else { indent };
        let room = if bounded_w { avail_w - indent } else { line_content };
        let lead_left = if para.rtl { 0.0 } else { indent };
        let h = match line_align {
            HAlign::Start => if para.rtl { HAlign::Right } else { HAlign::Left },
            HAlign::End => if para.rtl { HAlign::Left } else { HAlign::Right },
            h => h,
        };
        let mut justify_extra = 0.0;
        let x_start = p.margins[0]
            + lead_left
            + match h {
                HAlign::Left => 0.0,
                HAlign::Center => (room - line_content) * 0.5,
                HAlign::Right => room - line_content,
                HAlign::Justified | HAlign::Flush => {
                    let inner_spaces = {
                        let end = items.iter().rposition(|i| !i.space).map(|i| i + 1).unwrap_or(0);
                        items[..end].iter().filter(|i| i.space).count()
                    };
                    if (h == HAlign::Flush || !l.last) && inner_spaces > 0 && bounded_w && line_content < room {
                        justify_extra = (room - line_content) / inner_spaces as f32;
                    }
                    if para.rtl { room - line_content - justify_extra * inner_spaces as f32 } else { 0.0 }
                }
                _ => 0.0,
            };
        let trailing_start = items.iter().rposition(|i| !i.space).map(|i| i + 1).unwrap_or(0);

        let glyph_start = out.glyphs.len() as u32;
        let mut x = x_start;
        for &idx in &order {
            let it = &items[idx];
            let attrs = &st.parsed.runs[it.run].attrs;
            let face = fonts.get(it.font);
            let color = attrs.color.unwrap_or([255; 4]);
            let (bold, italic) = (attrs.bold || p.bold, attrs.italic || p.italic);
            let link = attrs.link.unwrap_or(u32::MAX);
            let voff = attrs.voffset_em * it.size;
            let extra = if it.space && idx < trailing_start { justify_extra } else { 0.0 };
            let item_w = it.adv + extra;
            let glyph_src: &[PGlyph] = if it.ellipsis {
                st.ellipsis.as_ref().map(|e| e.glyphs.as_slice()).unwrap_or(&[])
            } else {
                &para.glyphs[it.glyphs.start as usize..it.glyphs.end as usize]
            };
            let mut pen = x;
            let mut place = |g: &PGlyph, pen: &mut f32| {
                out.glyphs.push(PlacedGlyph {
                    font: it.font,
                    glyph: g.glyph,
                    x: *pen + g.x_off * it.size,
                    y: baseline - g.y_off * it.size - voff,
                    size: it.size,
                    color,
                    bold,
                    italic,
                    link,
                    cluster: it.byte,
                    line: vi as u32,
                });
                *pen += g.x_adv * it.size;
            };
            // Glyphs inside an RTL atom were reversed for breaking: restore
            // the shaper's visual order.
            if it.level % 2 == 1 && !it.ellipsis {
                glyph_src.iter().rev().for_each(|g| place(g, &mut pen));
            } else {
                glyph_src.iter().for_each(|g| place(g, &mut pen));
            }
            if let Some(l) = attrs.link {
                match link_raw.last_mut() {
                    Some(r) if r.link == l && r.line == vi as u32 && (r.x + r.w - x).abs() < 0.05 => r.w += item_w,
                    _ => link_raw.push(LinkRect { link: l, x, y, w: item_w, h: height, line: vi as u32 }),
                }
            }
            // Decorations spanning this item.
            if let Some(f) = face {
                if let Some(m) = attrs.mark {
                    under_raw.push((vi as u32, DecorRect { x, y: baseline - asc, w: item_w, h: natural, color: m }));
                }
                if attrs.underline {
                    let th = (f.metrics.underline_thickness * it.size).max(1.0);
                    let top = baseline - f.metrics.underline_pos * it.size;
                    over_raw.push((vi as u32, DecorRect { x, y: top, w: item_w, h: th, color }));
                }
                if attrs.strike {
                    let th = (f.metrics.strike_thickness * it.size).max(1.0);
                    let top = baseline - f.metrics.strike_pos * it.size - th * 0.5;
                    over_raw.push((vi as u32, DecorRect { x, y: top, w: item_w, h: th, color }));
                }
            }
            x += item_w;
        }
        out.lines.push(LineInfo {
            top: y,
            baseline,
            height,
            width: line_total,
            glyph_start,
            glyph_end: out.glyphs.len() as u32,
        });
        out.width = out.width.max(line_content);
        y += height + if l.last { para_gap } else { 0.0 };
    }
    out.height = y - y0;
    out.links = link_raw;
    out.under = merge_decor(under_raw);
    out.over = merge_decor(over_raw);
    out
}

/// Join horizontally adjacent rects of one line with identical style.
fn merge_decor(raw: Vec<(u32, DecorRect)>) -> Vec<DecorRect> {
    let mut out: Vec<(u32, DecorRect)> = Vec::new();
    for (line, r) in raw {
        if let Some((l, last)) = out.last_mut()
            && *l == line
            && last.color == r.color
            && (last.y - r.y).abs() < 0.01
            && (last.h - r.h).abs() < 0.01
            && (last.x + last.w - r.x).abs() < 0.05
        {
            last.w += r.w;
            continue;
        }
        out.push((line, r));
    }
    out.into_iter().map(|(_, r)| r).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::test_fonts::plex_sans;
    use crate::markup::parse;

    struct Fixture {
        fonts: FontRegistry,
        shaper: Shaper,
        chain: Vec<FontId>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut fonts = FontRegistry::default();
            let id = fonts.add(plex_sans());
            Self { fonts, shaper: Shaper::default(), chain: vec![id] }
        }

        fn shape(&mut self, src: &str, rich: bool) -> ShapedText {
            let mut env = ShapeEnv { fonts: &self.fonts, chain: &self.chain, shaper: &mut self.shaper, lang: "en", rtl: None };
            shape_text(&mut env, parse(src, rich))
        }
    }

    fn params(w: f32, h: f32) -> LayoutParams {
        LayoutParams { size: 20.0, box_size: [w, h], ..Default::default() }
    }

    #[test]
    fn single_line_width_is_sum_of_advances() {
        let mut f = Fixture::new();
        let st = f.shape("Hello", false);
        let l = layout(&st, &f.fonts, &params(0.0, 0.0));
        assert_eq!(l.lines.len(), 1);
        assert_eq!(l.glyphs.len(), 5);
        let sum: f32 = st.paragraphs[0].glyphs.iter().map(|g| g.x_adv).sum::<f32>() * 20.0;
        assert!((l.width - sum).abs() < 0.01, "{} vs {sum}", l.width);
        // Glyphs advance left to right.
        assert!(l.glyphs.windows(2).all(|w| w[1].x > w[0].x));
    }

    #[test]
    fn wraps_at_word_boundaries_within_the_box() {
        let mut f = Fixture::new();
        let st = f.shape("the quick brown fox jumps over the lazy dog", false);
        let one = layout(&st, &f.fonts, &params(0.0, 0.0)).width;
        let l = layout(&st, &f.fonts, &params(one / 3.0, 0.0));
        assert!(l.lines.len() >= 3, "{} lines", l.lines.len());
        for line in &l.lines {
            assert!(line.width <= one / 3.0 + 20.0 * 0.6, "line too wide: {}", line.width);
        }
        // Breaks only happen after spaces: no line starts mid-word.
        for line in l.lines.iter().skip(1) {
            let g = &l.glyphs[line.glyph_start as usize];
            let ch = st.parsed.text[g.cluster as usize..].chars().next().unwrap();
            assert_ne!(ch, ' ');
        }
    }

    #[test]
    fn long_word_breaks_between_clusters() {
        let mut f = Fixture::new();
        let st = f.shape("Supercalifragilisticexpialidocious", false);
        let l = layout(&st, &f.fonts, &params(80.0, 0.0));
        assert!(l.lines.len() > 1);
        assert!(l.lines.iter().all(|l| l.width <= 80.0 + 20.0));
    }

    #[test]
    fn newlines_and_br_make_paragraphs() {
        let mut f = Fixture::new();
        let st = f.shape("a\n\nb<br>c", true);
        let l = layout(&st, &f.fonts, &params(0.0, 0.0));
        assert_eq!(l.lines.len(), 4);
        assert_eq!(l.glyphs.len(), 3);
        // Line tops strictly increase, including the empty one.
        assert!(l.lines.windows(2).all(|w| w[1].top > w[0].top));
    }

    #[test]
    fn alignment_offsets() {
        let mut f = Fixture::new();
        let st = f.shape("ab", false);
        let w = 200.0;
        let at = |h| {
            let mut p = params(w, 0.0);
            p.halign = h;
            layout(&st, &f.fonts, &p)
        };
        let left = at(HAlign::Left);
        let right = at(HAlign::Right);
        let center = at(HAlign::Center);
        let cw = left.width;
        assert!((left.glyphs[0].x - 0.0).abs() < 0.01);
        assert!((right.glyphs[0].x - (w - cw)).abs() < 0.01);
        assert!((center.glyphs[0].x - (w - cw) * 0.5).abs() < 0.01);
    }

    #[test]
    fn justify_fills_non_last_lines() {
        let mut f = Fixture::new();
        let st = f.shape("aaa bbb ccc ddd eee fff", false);
        let one = layout(&st, &f.fonts, &params(0.0, 0.0)).width;
        let mut p = params(one * 0.6, 0.0);
        p.halign = HAlign::Justified;
        let l = layout(&st, &f.fonts, &p);
        assert!(l.lines.len() >= 2);
        // The right edge of the last glyph of line 0 reaches the box edge.
        let end = l.lines[0].glyph_end as usize;
        let g = l.glyphs[..end]
            .iter()
            .rev()
            .find(|g| !st.parsed.text[g.cluster as usize..].starts_with(' '))
            .unwrap();
        let adv = st.paragraphs[0].glyphs.iter().find(|x| x.glyph == g.glyph).unwrap().x_adv * g.size;
        assert!((g.x + adv - one * 0.6).abs() < 1.5, "{} vs {}", g.x + adv, one * 0.6);
    }

    #[test]
    fn vertical_alignment_uses_free_height() {
        let mut f = Fixture::new();
        let st = f.shape("x", false);
        let mut p = params(100.0, 200.0);
        let top = layout(&st, &f.fonts, &p).lines[0].top;
        p.valign = VAlign::Middle;
        let mid = layout(&st, &f.fonts, &p);
        p.valign = VAlign::Bottom;
        let bot = layout(&st, &f.fonts, &p);
        assert_eq!(top, 0.0);
        assert!((mid.lines[0].top - (200.0 - mid.height) * 0.5).abs() < 0.01);
        assert!((bot.lines[0].top + bot.height - 200.0).abs() < 0.01);
    }

    #[test]
    fn ellipsis_truncates_height_and_width() {
        let mut f = Fixture::new();
        let st = f.shape("one two three four five six seven eight nine ten eleven twelve", false);
        let mut p = params(120.0, 50.0);
        p.overflow = Overflow::Ellipsis;
        let l = layout(&st, &f.fonts, &p);
        assert!(l.truncated);
        assert!(l.height <= 50.0 + 1.0, "{}", l.height);
        assert!(l.lines.len() < 6);
        let last = l.lines.last().unwrap();
        let tail = &l.glyphs[last.glyph_end as usize - 1];
        let ell = st.ellipsis.as_ref().unwrap();
        assert_eq!(tail.glyph, ell.glyphs.last().unwrap().glyph, "line ends with the ellipsis glyph");
        assert!(last.width <= 120.0 + 0.5);

        // Single line wider than the box.
        let mut p = params(60.0, 0.0);
        p.wrap = false;
        p.overflow = Overflow::Ellipsis;
        let l = layout(&st, &f.fonts, &p);
        assert_eq!(l.lines.len(), 1);
        assert!(l.truncated && l.lines[0].width <= 60.5);
    }

    #[test]
    fn truncate_and_masking_and_pages() {
        let mut f = Fixture::new();
        let st = f.shape("aaa bbb ccc ddd eee fff ggg hhh iii jjj", false);
        let mut p = params(50.0, 50.0);
        p.overflow = Overflow::Truncate;
        let t = layout(&st, &f.fonts, &p);
        assert!(t.truncated && t.height <= 50.5);
        p.overflow = Overflow::Masking;
        let m = layout(&st, &f.fonts, &p);
        assert_eq!(m.clip, Some([0.0, 0.0, 50.0, 50.0]));
        assert!(m.height > 50.0 && !m.truncated);
        p.overflow = Overflow::Page;
        let p0 = layout(&st, &f.fonts, &p);
        assert!(p0.page_count >= 2);
        p.page = 1;
        let p1 = layout(&st, &f.fonts, &p);
        assert_eq!(p1.page_count, p0.page_count);
        assert!(p1.glyphs[0].cluster > p0.glyphs[0].cluster, "page 1 starts later in the text");
        assert!((p1.lines[0].top - 0.0).abs() < 0.01);
    }

    #[test]
    fn auto_size_picks_largest_fit_without_reshaping() {
        let mut f = Fixture::new();
        let st = f.shape("auto sizing text that has to fit", false);
        let misses = f.shaper.misses;
        let mut p = params(200.0, 60.0);
        p.auto_size = Some((8.0, 64.0));
        let l = layout(&st, &f.fonts, &p);
        assert!(l.size > 8.0 && l.size < 64.0, "{}", l.size);
        assert!(l.height <= 60.0 + 0.1);
        // 1px bigger would not fit (bisection resolution is 0.05).
        let mut q = p;
        q.auto_size = None;
        q.size = l.size + 1.0;
        assert!(layout(&st, &f.fonts, &q).height > 60.0 || {
            let pl = plan(&st, &f.fonts, &q, q.size);
            !fits(&pl, &q)
        });
        assert_eq!(f.shaper.misses, misses, "auto-size never reshapes");
        // Generous box: clamps to max. Tiny box: clamps to min.
        let mut big = p;
        big.box_size = [4000.0, 4000.0];
        assert_eq!(layout(&st, &f.fonts, &big).size, 64.0);
        let mut tiny = p;
        tiny.box_size = [10.0, 10.0];
        assert!((layout(&st, &f.fonts, &tiny).size - 8.0).abs() < 0.06);
    }

    #[test]
    fn rich_text_sizes_colours_and_decorations() {
        let mut f = Fixture::new();
        let st = f.shape("a<size=40><color=#00ff00>b</color></size><u>c</u><s>d</s><mark=#ff000080>e</mark>", true);
        let l = layout(&st, &f.fonts, &params(0.0, 0.0));
        assert_eq!(l.glyphs.len(), 5);
        assert_eq!(l.glyphs[0].size, 20.0);
        assert_eq!(l.glyphs[1].size, 40.0);
        assert_eq!(l.glyphs[1].color, [0, 255, 0, 255]);
        assert_eq!(l.over.len(), 2, "underline + strikethrough");
        assert_eq!(l.under.len(), 1);
        assert_eq!(l.under[0].color, [255, 0, 0, 128]);
        // Underline sits below the baseline.
        let base = l.lines[0].baseline;
        assert!(l.over[0].y > base - 2.0);
    }

    #[test]
    fn spacing_adds_to_advances() {
        let mut f = Fixture::new();
        let st = f.shape("ab", false);
        let base = layout(&st, &f.fonts, &params(0.0, 0.0));
        let mut p = params(0.0, 0.0);
        p.char_spacing = 50.0;
        let spaced = layout(&st, &f.fonts, &p);
        assert!((spaced.glyphs[1].x - base.glyphs[1].x - 0.5 * 20.0).abs() < 0.01);
        let mut q = params(100.0, 0.0);
        q.line_spacing = 100.0;
        let st2 = f.shape("a\nb", false);
        let tight = layout(&st2, &f.fonts, &params(100.0, 0.0));
        let loose = layout(&st2, &f.fonts, &q);
        assert!(loose.lines[1].top - loose.lines[0].top > 1.9 * (tight.lines[1].top - tight.lines[0].top));
    }

    #[test]
    fn nobr_keeps_words_together() {
        let mut f = Fixture::new();
        let plain = f.shape("aaa bbb ccc", false);
        let nobr = f.shape("aaa <nobr>bbb ccc</nobr>", true);
        let w = layout(&plain, &f.fonts, &params(0.0, 0.0)).width * 0.7;
        let a = layout(&plain, &f.fonts, &params(w, 0.0));
        let b = layout(&nobr, &f.fonts, &params(w, 0.0));
        assert_eq!(a.lines.len(), 2);
        // The nobr group wraps as a unit: "bbb ccc" lands on one line.
        assert_eq!(b.lines.len(), 2);
        assert_eq!(b.lines[1].glyph_end - b.lines[1].glyph_start, 7);
    }

    #[test]
    fn visual_order_reverses_rtl_runs() {
        assert_eq!(visual_order(&[0, 0, 0]), vec![0, 1, 2]);
        assert_eq!(visual_order(&[1, 1, 1]), vec![2, 1, 0]);
        // LTR, RTL pair, LTR.
        assert_eq!(visual_order(&[0, 1, 1, 0]), vec![0, 2, 1, 3]);
        // Nested: RTL paragraph with an LTR number inside.
        assert_eq!(visual_order(&[1, 2, 2, 1]), vec![3, 1, 2, 0]);
    }

    #[test]
    fn rtl_text_lays_out_right_aligned_by_default() {
        let mut f = Fixture::new();
        // Hebrew letters: the Latin test font has no glyphs, so every char
        // falls to .notdef, but direction and ordering still apply.
        let st = f.shape("\u{5d0}\u{5d1}\u{5d2}", false);
        assert!(st.paragraphs[0].rtl);
        let l = layout(&st, &f.fonts, &params(300.0, 0.0));
        let g = &l.glyphs;
        assert_eq!(g.len(), 3);
        // Visual order: the first logical char (cluster 0) is rightmost.
        assert!(g[0].cluster > g[1].cluster && g[1].cluster > g[2].cluster || g.windows(2).all(|w| w[0].x < w[1].x));
        let right_edge = g.iter().map(|g| g.x).fold(0.0, f32::max);
        assert!(right_edge > 150.0, "Start alignment in RTL hugs the right edge");
    }

    #[test]
    fn mixed_direction_keeps_numbers_ltr() {
        let mut f = Fixture::new();
        let st = f.shape("\u{5d0} 12 \u{5d1}", false);
        let l = layout(&st, &f.fonts, &params(0.0, 0.0));
        let by_cluster = |c: u32| l.glyphs.iter().find(|g| g.cluster == c).unwrap().x;
        // "1" (byte 3) sits left of "2" (byte 4) even in an RTL paragraph.
        assert!(by_cluster(4) > by_cluster(3));
        // First Hebrew letter (byte 0) is right of the digits.
        assert!(by_cluster(0) > by_cluster(4));
    }

    // ---- layout tags, links, vertical anchors

    fn first_x(l: &TextLayout, line: u32) -> f32 {
        l.glyphs.iter().filter(|g| g.line == line).map(|g| g.x).fold(f32::INFINITY, f32::min)
    }

    #[test]
    fn align_tag_overrides_the_box_alignment_per_line() {
        let mut f = Fixture::new();
        let st = f.shape("left<br><align=right>right<br><align=center>mid", true);
        let l = layout(&st, &f.fonts, &params(400.0, 0.0));
        assert_eq!(l.lines.len(), 3);
        let (a, b, c) = (first_x(&l, 0), first_x(&l, 1), first_x(&l, 2));
        assert!(a < 1.0, "first line stays left: {a}");
        assert!(b > 300.0, "right-aligned line starts far right: {b}");
        assert!(c > 150.0 && c < 300.0, "centred line sits in the middle: {c}");
    }

    #[test]
    fn indent_shifts_lines_and_narrows_wrapping() {
        let mut f = Fixture::new();
        let plain = f.shape("aaa bbb ccc ddd eee fff", true);
        let l0 = layout(&plain, &f.fonts, &params(200.0, 0.0));
        let ind = f.shape("<indent=50>aaa bbb ccc ddd eee fff", true);
        let l1 = layout(&ind, &f.fonts, &params(200.0, 0.0));
        assert!((first_x(&l1, 0) - first_x(&l0, 0) - 50.0).abs() < 0.01, "indent moves the first glyph by 50 px");
        assert!(l1.lines.len() >= l0.lines.len(), "a narrower line never wraps less");
        assert!(l1.lines.iter().all(|ln| ln.width <= 150.5), "every line fits box - indent: {:?}", l1.lines.iter().map(|l| l.width).collect::<Vec<_>>());
        // Percent is a share of the box width; em of the font size.
        let pct = layout(&f.shape("<indent=25%>x", true), &f.fonts, &params(200.0, 0.0));
        assert!((first_x(&pct, 0) - 50.0).abs() < 1.0);
        let em = layout(&f.shape("<indent=2em>x", true), &f.fonts, &params(200.0, 0.0));
        assert!((first_x(&em, 0) - 40.0).abs() < 1.0);
    }

    #[test]
    fn line_height_tag_replaces_the_natural_height() {
        let mut f = Fixture::new();
        let natural = layout(&f.shape("a<br>b", true), &f.fonts, &params(0.0, 0.0)).lines[0].height;
        let tagged = layout(&f.shape("<line-height=60>a<br>b", true), &f.fonts, &params(0.0, 0.0));
        assert!((tagged.lines[0].height - 60.0).abs() < 1e-3 && (tagged.lines[1].height - 60.0).abs() < 1e-3);
        assert!(natural < 60.0);
        let pct = layout(&f.shape("<line-height=200%>a", true), &f.fonts, &params(0.0, 0.0));
        assert!((pct.lines[0].height - 40.0).abs() < 1e-3, "200 % of the 20 px font: {}", pct.lines[0].height);
        // The tallest run on a line wins.
        let mixed = layout(&f.shape("<line-height=30>a</line-height><line-height=50>b", true), &f.fonts, &params(0.0, 0.0));
        assert!((mixed.lines[0].height - 50.0).abs() < 1e-3);
    }

    #[test]
    fn link_rects_cover_the_linked_words_per_line_and_hit_test() {
        let mut f = Fixture::new();
        let st = f.shape("go <link=door>open the door now</link> ok", true);
        let l = layout(&st, &f.fonts, &params(120.0, 0.0));
        assert!(l.lines.len() >= 2, "the link wraps");
        assert!(l.links.len() >= 2 && l.links.iter().all(|r| r.link == 0));
        let first = l.links[0];
        assert_eq!(l.link_at(first.x + first.w * 0.5, first.y + first.h * 0.5), Some(0));
        assert_eq!(l.link_at(first.x - 5.0, first.y + first.h * 0.5), None, "left of the link is outside it");
        assert_eq!(l.link_at(0.5, first.y + first.h * 0.5), None);
        // Unlinked text has none.
        assert!(layout(&f.shape("plain", true), &f.fonts, &params(0.0, 0.0)).links.is_empty());
        // One rect per line, not per word.
        let lines: std::collections::HashSet<u32> = l.links.iter().map(|r| r.line).collect();
        assert_eq!(lines.len(), l.links.len());
    }

    #[test]
    fn baseline_midline_capline_anchor_the_first_line_on_the_centre() {
        let mut f = Fixture::new();
        let st = f.shape("Hxg", false);
        let metrics = f.fonts.get(f.chain[0]).unwrap().metrics;
        let at = |v: VAlign| {
            let l = layout(&st, &f.fonts, &LayoutParams { valign: v, ..params(200.0, 100.0) });
            l.lines[0].baseline
        };
        // Baseline: the baseline is on y = 50.
        assert!((at(VAlign::Baseline) - 50.0).abs() < 0.01);
        // Midline: half an x-height above the baseline is the centre, so the baseline is lower.
        assert!((at(VAlign::Midline) - (50.0 + metrics.x_height * 20.0 * 0.5)).abs() < 0.01);
        assert!((at(VAlign::Capline) - (50.0 + metrics.cap_height * 20.0)).abs() < 0.01);
        assert!(at(VAlign::Capline) > at(VAlign::Midline) && at(VAlign::Midline) > at(VAlign::Baseline));
        // Unbounded height: no effect (same as Top).
        let top = layout(&st, &f.fonts, &params(200.0, 0.0)).lines[0].baseline;
        let base = layout(&st, &f.fonts, &LayoutParams { valign: VAlign::Capline, ..params(200.0, 0.0) }).lines[0].baseline;
        assert_eq!(top, base);
    }

    #[test]
    fn font_tag_picks_a_registered_face_and_falls_back_when_unknown() {
        let mut f = Fixture::new();
        let mut other = plex_sans();
        other.name = "fonts/Display-Bold.ttf".into();
        let other_id = f.fonts.add(other);
        let st = f.shape("a<font=\"display-bold\">b</font>c<font=\"nope\">d", true);
        let fonts_of: Vec<FontId> = st.paragraphs[0].atoms.iter().map(|a| a.font).collect();
        assert_eq!(fonts_of, vec![f.chain[0], other_id, f.chain[0], f.chain[0]]);
    }
}
