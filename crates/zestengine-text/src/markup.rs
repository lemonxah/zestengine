//! Rich text tags (a small, TMP-style subset) parsed once into plain text plus
//! attribute runs. Unknown tags stay literal, like TextMeshPro.
//!
//! Supported: `b i u s mark sub sup color alpha size voffset cspace nobr br
//! noparse lowercase uppercase allcaps smallcaps link align indent line-height
//! font style`. Values take `px`, `em` and `%` units where a length applies.
//! `<style=name>` expands a named tag sequence from a [`StyleSheet`].

use std::collections::HashMap;

use crate::layout::HAlign;

/// Packed RGBA8, straight alpha.
pub type Rgba = [u8; 4];

/// A length from a tag: `px`, `em` (of the run's font size) or `%` (of the run's
/// font size for `line-height`, of the box width for `indent`; stored as a fraction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Len {
    Px(f32),
    Em(f32),
    Percent(f32),
}

impl Len {
    /// Resolve against a font size and a reference width (the box's inner width).
    pub fn resolve(self, font_px: f32, reference: f32) -> f32 {
        match self {
            Len::Px(v) => v,
            Len::Em(v) => v * font_px,
            Len::Percent(v) => v * reference,
        }
    }
}

/// Named tag sequences for `<style=name>`: `"title" -> "<b><size=150%><color=#ffd060>"`.
/// Only tags count (text inside a definition is ignored); styles may use other styles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StyleSheet {
    pub styles: HashMap<String, String>,
}

impl StyleSheet {
    pub fn insert(&mut self, name: impl Into<String>, tags: impl Into<String>) {
        self.styles.insert(name.into().to_ascii_lowercase(), tags.into());
    }

    pub fn is_empty(&self) -> bool {
        self.styles.is_empty()
    }
}

/// Per-run text attributes.
#[derive(Clone, Debug, PartialEq)]
pub struct Attrs {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    pub mark: Option<Rgba>,
    pub color: Option<Rgba>,
    /// Absolute size in layout px (`<size=24>`); `None` inherits the base size.
    pub size_px: Option<f32>,
    /// Added to the (base or absolute) size in px (`<size=+4>`).
    pub size_delta: f32,
    /// Multiplier on the size (`<size=150%>`, `<sub>`, small caps).
    pub size_em: f32,
    /// Baseline shift in em, up positive (`<voffset>`, `<sub>`, `<sup>`).
    pub voffset_em: f32,
    /// Extra advance per character, in em (`<cspace>`).
    pub cspace_em: f32,
    pub nobr: bool,
    pub link: Option<u32>,
    /// `<align=...>`: horizontal alignment of the lines this run starts.
    pub align: Option<HAlign>,
    /// `<indent=...>`: left indent of the lines this run starts.
    pub indent: Option<Len>,
    /// `<line-height=...>`: height of the lines this run is on (the tallest wins).
    pub line_height: Option<Len>,
    /// `<font="name">`: a registered face (path or built-in name, case-insensitive,
    /// extension optional), tried before the font chain.
    pub font: Option<String>,
}

impl Default for Attrs {
    fn default() -> Self {
        Self {
            bold: false,
            italic: false,
            underline: false,
            strike: false,
            mark: None,
            color: None,
            size_px: None,
            size_delta: 0.0,
            size_em: 1.0,
            voffset_em: 0.0,
            cspace_em: 0.0,
            nobr: false,
            link: None,
            align: None,
            indent: None,
            line_height: None,
            font: None,
        }
    }
}

impl Attrs {
    /// Effective size for a base size.
    pub fn size(&self, base: f32) -> f32 {
        (self.size_px.unwrap_or(base + self.size_delta) * self.size_em).max(0.01)
    }
}

/// A byte range of the plain text sharing one set of attributes.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub start: u32,
    pub end: u32,
    pub attrs: Attrs,
}

/// Parse result: tags removed, runs cover the whole text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    pub text: String,
    pub runs: Vec<Run>,
    /// `<link=id>` targets, indexed by `Attrs::link`.
    pub links: Vec<String>,
}

impl Parsed {
    /// Index of the run containing byte `at` (the last run for `at == len`).
    pub fn run_at(&self, at: u32) -> usize {
        match self.runs.binary_search_by(|r| {
            if at < r.start {
                std::cmp::Ordering::Greater
            } else if at >= r.end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(i) => i,
            Err(_) => self.runs.len().saturating_sub(1),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Caps {
    None,
    Upper,
    Lower,
    Small,
}

struct Builder<'a> {
    out: Parsed,
    cur: Attrs,
    caps: Caps,
    styles: &'a StyleSheet,
    /// Nesting of `<style>` expansion (a style that names itself stops).
    depth: u32,
}

impl Builder<'_> {
    fn emit(&mut self, s: &str) {
        for c in s.chars() {
            self.emit_char(c);
        }
    }

    fn emit_char(&mut self, c: char) {
        let mut attrs = &self.cur;
        let small;
        let mut buf = [0u8; 4];
        let s: String = match self.caps {
            Caps::None => c.encode_utf8(&mut buf).to_owned(),
            Caps::Upper => c.to_uppercase().collect(),
            Caps::Lower => c.to_lowercase().collect(),
            Caps::Small => {
                if c.is_lowercase() {
                    let mut a = self.cur.clone();
                    a.size_em *= 0.8;
                    small = a;
                    attrs = &small;
                    c.to_uppercase().collect()
                } else {
                    c.encode_utf8(&mut buf).to_owned()
                }
            }
        };
        let start = self.out.text.len() as u32;
        self.out.text.push_str(&s);
        let end = self.out.text.len() as u32;
        match self.out.runs.last_mut() {
            Some(last) if last.attrs == *attrs && last.end == start => last.end = end,
            _ => self.out.runs.push(Run { start, end, attrs: attrs.clone() }),
        }
    }
}

/// Parse `src`. With `rich == false` the text passes through untouched.
pub fn parse(src: &str, rich: bool) -> Parsed {
    parse_with(src, rich, &StyleSheet::default())
}

/// [`parse`] with a style sheet for `<style=name>`.
pub fn parse_with(src: &str, rich: bool, styles: &StyleSheet) -> Parsed {
    if !rich || !src.contains('<') {
        let mut p = Parsed { text: src.to_owned(), ..Default::default() };
        p.runs.push(Run { start: 0, end: src.len() as u32, attrs: Attrs::default() });
        return p;
    }
    let mut b = Builder { out: Parsed::default(), cur: Attrs::default(), caps: Caps::None, styles, depth: 0 };
    // Attribute stacks: each open tag remembers what to restore on close.
    let mut stack: Vec<(String, Attrs, bool /*caps changed*/, Caps)> = Vec::new();
    let mut rest = src;
    while let Some(lt) = rest.find('<') {
        b.emit(&rest[..lt]);
        let after = &rest[lt + 1..];
        let Some(gt) = after.find('>') else {
            b.emit(&rest[lt..]);
            rest = "";
            break;
        };
        let tag = &after[..gt];
        let consumed = lt + 1 + gt + 1;
        if apply_tag(&mut b, &mut stack, tag, &mut rest, consumed, src) {
            continue;
        }
        // Unknown tag: keep it literally.
        b.emit(&rest[lt..consumed]);
        rest = &rest[consumed..];
    }
    b.emit(rest);
    if b.out.runs.is_empty() {
        b.out.runs.push(Run { start: 0, end: 0, attrs: Attrs::default() });
    }
    b.out
}

/// Handle one tag. Returns false for an unrecognised tag (caller emits it).
fn apply_tag(
    b: &mut Builder<'_>,
    stack: &mut Vec<(String, Attrs, bool, Caps)>,
    tag: &str,
    rest: &mut &str,
    consumed: usize,
    _src: &str,
) -> bool {
    let tag = tag.trim();
    if let Some(name) = tag.strip_prefix('/') {
        let name = name.trim().to_ascii_lowercase();
        if let Some(pos) = stack.iter().rposition(|(n, ..)| *n == name) {
            let (_, attrs, caps_changed, caps) = stack.remove(pos);
            b.cur = attrs;
            if caps_changed {
                b.caps = caps;
            }
            *rest = &rest[consumed..];
            return true;
        }
        return false;
    }
    let (name, value) = match tag.split_once('=') {
        Some((n, v)) => (n.trim().to_ascii_lowercase(), Some(v.trim().trim_matches('"'))),
        None => (tag.to_ascii_lowercase(), None),
    };
    let saved = b.cur.clone();
    let saved_caps = b.caps;
    let mut caps_changed = false;
    let mut pushes = true;
    match name.as_str() {
        "noparse" => {
            // Everything up to </noparse> is literal.
            let body = &rest[consumed..];
            let end = body.to_ascii_lowercase().find("</noparse>");
            let (lit, skip) = match end {
                Some(e) => (&body[..e], e + "</noparse>".len()),
                None => (body, body.len()),
            };
            b.emit(lit);
            *rest = &body[skip..];
            return true;
        }
        "br" => {
            b.emit("\n");
            pushes = false;
        }
        "b" => b.cur.bold = true,
        "i" => b.cur.italic = true,
        "u" => b.cur.underline = true,
        "s" => b.cur.strike = true,
        "nobr" => b.cur.nobr = true,
        "sub" => {
            b.cur.size_em *= 0.6;
            b.cur.voffset_em -= 0.14;
        }
        "sup" => {
            b.cur.size_em *= 0.6;
            b.cur.voffset_em += 0.35;
        }
        "mark" => b.cur.mark = Some(value.and_then(parse_color).unwrap_or([255, 255, 0, 128])),
        "color" => match value.and_then(parse_color) {
            Some(c) => {
                // Keep a previously set alpha.
                let a = b.cur.color.map(|c| c[3]).unwrap_or(255);
                b.cur.color = Some([c[0], c[1], c[2], if c[3] == 255 { a } else { c[3] }]);
            }
            None => return false,
        },
        "alpha" => match value.and_then(parse_color) {
            Some(c) => {
                let base = b.cur.color.unwrap_or([255; 4]);
                b.cur.color = Some([base[0], base[1], base[2], c[3]]);
            }
            None => return false,
        },
        "size" => match value.and_then(parse_size) {
            Some(Size::Px(v)) => {
                b.cur.size_px = Some(v);
                b.cur.size_delta = 0.0;
                b.cur.size_em = 1.0;
            }
            Some(Size::Delta(v)) => b.cur.size_delta += v,
            Some(Size::Em(v)) => b.cur.size_em *= v,
            None => return false,
        },
        "voffset" => match value.and_then(parse_size) {
            // px offsets are approximated at 1 em = 32 px; em/% are exact.
            Some(Size::Px(v)) | Some(Size::Delta(v)) => b.cur.voffset_em += v / 32.0,
            Some(Size::Em(v)) => b.cur.voffset_em += v,
            None => return false,
        },
        "cspace" => match value.and_then(parse_size) {
            Some(Size::Px(v)) | Some(Size::Delta(v)) => b.cur.cspace_em += v / 32.0,
            Some(Size::Em(v)) => b.cur.cspace_em += v,
            None => return false,
        },
        "link" => {
            let id = value.unwrap_or("").to_owned();
            b.out.links.push(id);
            b.cur.link = Some(b.out.links.len() as u32 - 1);
            b.cur.underline = true;
        }
        "align" => match value.map(|v| v.to_ascii_lowercase()).as_deref() {
            Some("left") => b.cur.align = Some(HAlign::Left),
            Some("center") => b.cur.align = Some(HAlign::Center),
            Some("right") => b.cur.align = Some(HAlign::Right),
            Some("justified") => b.cur.align = Some(HAlign::Justified),
            Some("flush") => b.cur.align = Some(HAlign::Flush),
            _ => return false,
        },
        "indent" => match value.and_then(parse_len) {
            Some(l) => b.cur.indent = Some(l),
            None => return false,
        },
        "line-height" => match value.and_then(parse_len) {
            Some(l) => b.cur.line_height = Some(l),
            None => return false,
        },
        "font" => match value.filter(|v| !v.is_empty()) {
            Some(v) => b.cur.font = Some(v.to_owned()),
            None => return false,
        },
        "style" => {
            let Some(def) = value.and_then(|v| b.styles.styles.get(&v.to_ascii_lowercase())).cloned() else { return false };
            if b.depth < 8 {
                b.depth += 1;
                apply_style(b, &def);
                b.depth -= 1;
                caps_changed = b.caps != saved_caps;
            }
        }
        "lowercase" => {
            b.caps = Caps::Lower;
            caps_changed = true;
        }
        "uppercase" | "allcaps" => {
            b.caps = Caps::Upper;
            caps_changed = true;
        }
        "smallcaps" => {
            b.caps = Caps::Small;
            caps_changed = true;
        }
        _ => return false,
    }
    if pushes {
        stack.push((name, saved, caps_changed, saved_caps));
    }
    *rest = &rest[consumed..];
    true
}

/// Apply the tags of a style definition to the builder's current attributes
/// (the surrounding `<style>` tag restores them when it closes).
fn apply_style(b: &mut Builder<'_>, def: &str) {
    let mut sub: Vec<(String, Attrs, bool, Caps)> = Vec::new();
    let mut rest = def;
    while let Some(lt) = rest.find('<') {
        let after = &rest[lt + 1..];
        let Some(gt) = after.find('>') else { break };
        let tag = &after[..gt];
        let consumed = lt + 1 + gt + 1;
        let name = tag.trim().to_ascii_lowercase();
        // Tags that produce text have no place in a definition.
        let mut tail = &rest[consumed..];
        if !(name == "br" || name.starts_with("noparse")) {
            let mut at = rest;
            if apply_tag(b, &mut sub, tag, &mut at, consumed, def) {
                tail = at;
            }
        }
        rest = tail;
    }
}

/// `px`, `em` or `%` length for the layout tags.
fn parse_len(v: &str) -> Option<Len> {
    let v = v.trim();
    if let Some(p) = v.strip_suffix('%') {
        return Some(Len::Percent(p.trim().parse::<f32>().ok()? / 100.0));
    }
    if let Some(e) = v.strip_suffix("em") {
        return Some(Len::Em(e.trim().parse().ok()?));
    }
    Some(Len::Px(v.strip_suffix("px").unwrap_or(v).trim().parse().ok()?))
}

enum Size {
    Px(f32),
    Delta(f32),
    Em(f32),
}

fn parse_size(v: &str) -> Option<Size> {
    let v = v.trim();
    if let Some(p) = v.strip_suffix('%') {
        return Some(Size::Em(p.trim().parse::<f32>().ok()? / 100.0));
    }
    if let Some(e) = v.strip_suffix("em") {
        return Some(Size::Em(e.trim().parse().ok()?));
    }
    let n = v.strip_suffix("px").unwrap_or(v).trim();
    let value: f32 = n.parse().ok()?;
    if n.starts_with('+') || n.starts_with('-') { Some(Size::Delta(value)) } else { Some(Size::Px(value)) }
}

/// `#rgb`, `#rrggbb`, `#rrggbbaa` or a colour name.
pub fn parse_color(v: &str) -> Option<Rgba> {
    let v = v.trim();
    if let Some(h) = v.strip_prefix('#') {
        let d = |i: usize| u8::from_str_radix(h.get(i..i + 1)?, 16).ok().map(|x| x * 17);
        let p = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
        return match h.len() {
            // `<alpha=#80>`: alpha only.
            2 => Some([255, 255, 255, p(0)?]),
            3 => Some([d(0)?, d(1)?, d(2)?, 255]),
            6 => Some([p(0)?, p(2)?, p(4)?, 255]),
            8 => Some([p(0)?, p(2)?, p(4)?, p(6)?]),
            _ => None,
        };
    }
    Some(match v.to_ascii_lowercase().as_str() {
        "white" => [255, 255, 255, 255],
        "black" => [0, 0, 0, 255],
        "red" => [255, 0, 0, 255],
        "green" => [0, 128, 0, 255],
        "blue" => [0, 0, 255, 255],
        "yellow" => [255, 255, 0, 255],
        "orange" => [255, 128, 0, 255],
        "purple" => [160, 32, 240, 255],
        "grey" | "gray" => [128, 128, 128, 255],
        "cyan" => [0, 255, 255, 255],
        "magenta" => [255, 0, 255, 255],
        "lightblue" => [173, 216, 230, 255],
        "teal" => [0, 128, 128, 255],
        "darkblue" => [0, 0, 139, 255],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through() {
        let p = parse("a <b> c", false);
        assert_eq!(p.text, "a <b> c");
        assert_eq!(p.runs.len(), 1);
        let p = parse("no tags", true);
        assert_eq!(p.text, "no tags");
    }

    #[test]
    fn bold_color_and_nesting() {
        let p = parse("a<b>b<color=#ff0000>c</color>d</b>e", true);
        assert_eq!(p.text, "abcde");
        let at = |i: usize| &p.runs[p.run_at(i as u32)].attrs;
        assert!(!at(0).bold);
        assert!(at(1).bold && at(1).color.is_none());
        assert!(at(2).bold && at(2).color == Some([255, 0, 0, 255]));
        assert!(at(3).bold && at(3).color.is_none());
        assert!(!at(4).bold);
    }

    #[test]
    fn size_units() {
        let p = parse("<size=24>a</size><size=150%>b</size><size=+4>c</size><size=2em>d</size>", true);
        let a = |i: u32| p.runs[p.run_at(i)].attrs.clone();
        assert_eq!(a(0).size(10.0), 24.0);
        assert_eq!(a(1).size(10.0), 15.0);
        assert_eq!(a(2).size(10.0), 14.0);
        assert_eq!(a(3).size(10.0), 20.0);
    }

    #[test]
    fn br_noparse_and_unknown_tags() {
        let p = parse("a<br>b<noparse><b>x</b></noparse><wat>c", true);
        assert_eq!(p.text, "a\nb<b>x</b><wat>c");
        assert!(p.runs.iter().all(|r| !r.attrs.bold));
    }

    #[test]
    fn caps_and_alpha_and_links() {
        let p = parse("<uppercase>ab</uppercase>c<smallcaps>dE</smallcaps><alpha=#80>f", true);
        assert_eq!(p.text, "ABcDEf");
        let small = &p.runs[p.run_at(3)].attrs;
        assert_eq!(small.size_em, 0.8);
        assert_eq!(p.runs[p.run_at(4)].attrs.size_em, 1.0);
        assert_eq!(p.runs[p.run_at(5)].attrs.color, Some([255, 255, 255, 0x80]));
        let l = parse("<link=door>go</link> now", true);
        assert_eq!(l.links, vec!["door"]);
        assert!(l.runs[0].attrs.link == Some(0) && l.runs[0].attrs.underline);
        assert!(l.runs.last().unwrap().attrs.link.is_none());
    }

    #[test]
    fn unclosed_and_stray_tags_do_not_panic() {
        assert_eq!(parse("a<b", true).text, "a<b");
        assert_eq!(parse("a</b>b", true).text, "a</b>b");
        let p = parse("<b>open", true);
        assert_eq!(p.text, "open");
        assert!(p.runs[0].attrs.bold);
    }

    #[test]
    fn colour_parsing() {
        assert_eq!(parse_color("#fff"), Some([255, 255, 255, 255]));
        assert_eq!(parse_color("#11223344"), Some([0x11, 0x22, 0x33, 0x44]));
        assert_eq!(parse_color("Red"), Some([255, 0, 0, 255]));
        assert_eq!(parse_color("nope"), None);
    }

    #[test]
    fn layout_tags_set_run_attributes() {
        let p = parse("<align=center>a</align><indent=2em>b</indent><line-height=150%>c</line-height><font=\"Mono\">d", true);
        let at = |i: u32| p.runs[p.run_at(i)].attrs.clone();
        assert_eq!(at(0).align, Some(HAlign::Center));
        assert_eq!(at(1).indent, Some(Len::Em(2.0)));
        assert_eq!(at(2).line_height, Some(Len::Percent(1.5)));
        assert_eq!(at(3).font.as_deref(), Some("Mono"));
        assert!(at(1).align.is_none() && at(3).indent.is_none());
        // Bad values keep the tag literal.
        assert_eq!(parse("<align=sideways>x", true).text, "<align=sideways>x");
        assert_eq!(parse("<indent=wide>x", true).text, "<indent=wide>x");
    }

    #[test]
    fn style_sheets_expand_tag_sequences_and_close_cleanly() {
        let mut sheet = StyleSheet::default();
        sheet.insert("Title", "<b><size=150%><color=#ff0000>");
        sheet.insert("loud", "<style=title><u><uppercase>");
        let p = parse_with("a<style=title>b</style>c", true, &sheet);
        assert_eq!(p.text, "abc");
        let at = |i: u32| p.runs[p.run_at(i)].attrs.clone();
        assert!(!at(0).bold && at(0).color.is_none());
        assert!(at(1).bold && at(1).size_em == 1.5 && at(1).color == Some([255, 0, 0, 255]));
        assert!(!at(2).bold && at(2).color.is_none() && at(2).size_em == 1.0, "closing restores everything the style set");
        // Styles may use styles, and caps survive nesting and are restored.
        let q = parse_with("<style=loud>ab</style>cd", true, &sheet);
        assert_eq!(q.text, "ABcd");
        assert!(q.runs[q.run_at(0)].attrs.underline && q.runs[q.run_at(0)].attrs.bold);
        assert!(!q.runs[q.run_at(2)].attrs.underline);
        // Unknown styles stay literal; a self-referencing style stops instead of looping.
        assert_eq!(parse_with("<style=nope>x", true, &sheet).text, "<style=nope>x");
        sheet.insert("loop", "<style=loop><i>");
        let r = parse_with("<style=loop>x</style>", true, &sheet);
        assert_eq!(r.text, "x");
    }
}
