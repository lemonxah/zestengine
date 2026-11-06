//! Shaping runs with harfrust: kerning, ligatures, marks and complex scripts.
//! Output is in em units, independent of the size the text is drawn at, so a
//! shaped run is cached once and reused by every relayout and auto-size probe.

use std::collections::HashMap;
use std::sync::Arc;

use harfrust::{Buffer, Direction, Language, Script, ShapeOptions, ShaperFont, Tag};

use crate::font::FontFace;

/// One positioned glyph of a shaped run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    pub glyph: u32,
    /// Byte offset of the cluster in the shaped text (run-relative).
    pub cluster: u32,
    pub x_advance: f32,
    pub x_offset: f32,
    pub y_offset: f32,
}

/// Cache key for a shaped run.
fn run_key(face: &FontFace, text: &str, rtl: bool, script: unicode_script::Script, lang: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    face.name.hash(&mut h);
    text.hash(&mut h);
    rtl.hash(&mut h);
    (script as u32).hash(&mut h);
    lang.hash(&mut h);
    h.finish()
}

/// Reusable shaping state: one harfrust buffer plus the shaped-run cache.
#[derive(Default)]
pub struct Shaper {
    buffer: Option<Buffer>,
    cache: HashMap<u64, Arc<[ShapedGlyph]>>,
    pub hits: u64,
    pub misses: u64,
}

/// Cap before the cache is dropped wholesale (cheap, and rebuilt on demand).
const CACHE_LIMIT: usize = 8192;

fn hb_script(s: unicode_script::Script) -> Option<Script> {
    let name = s.short_name();
    let b = name.as_bytes();
    if b.len() != 4 {
        return None;
    }
    Script::from_iso15924_tag(Tag::new(&[b[0], b[1], b[2], b[3]]))
}

impl Shaper {
    /// Shape `text` (one single-font, single-direction, single-script piece).
    /// Glyphs come back in visual order (right-to-left runs reversed), with
    /// cluster values as byte offsets into `text`.
    pub fn shape(
        &mut self,
        face: &FontFace,
        text: &str,
        rtl: bool,
        script: unicode_script::Script,
        lang: &str,
    ) -> Arc<[ShapedGlyph]> {
        let key = run_key(face, text, rtl, script, lang);
        if let Some(hit) = self.cache.get(&key) {
            self.hits += 1;
            return hit.clone();
        }
        self.misses += 1;
        if self.cache.len() >= CACHE_LIMIT {
            self.cache.clear();
        }
        let out = self.shape_uncached(face, text, rtl, script, lang);
        let out: Arc<[ShapedGlyph]> = out.into();
        self.cache.insert(key, out.clone());
        out
    }

    fn shape_uncached(
        &mut self,
        face: &FontFace,
        text: &str,
        rtl: bool,
        script: unicode_script::Script,
        lang: &str,
    ) -> Vec<ShapedGlyph> {
        let mut buf = self.buffer.take().unwrap_or_default();
        buf.clear();
        buf.push_str(text);
        buf.set_direction(if rtl { Direction::RightToLeft } else { Direction::LeftToRight });
        buf.set_script(hb_script(script));
        if !lang.is_empty() {
            buf.set_language(lang.parse::<Language>().ok());
        }
        let sf = ShaperFont::new(&face.font);
        let upem = face.units_per_em;
        let mut out = Vec::new();
        match harfrust::shape(&sf, &mut buf, ShapeOptions::new()) {
            Ok(()) => {
                out.reserve(buf.len());
                for (i, p) in buf.glyph_infos().iter().zip(buf.glyph_positions()) {
                    out.push(ShapedGlyph {
                        glyph: i.glyph_id,
                        cluster: i.cluster,
                        x_advance: p.x_advance as f32 / upem,
                        x_offset: p.x_offset as f32 / upem,
                        y_offset: p.y_offset as f32 / upem,
                    });
                }
            }
            Err(e) => {
                tracing::warn!("shaping failed ({e}); falling back to nominal advances");
                for (i, c) in text.char_indices() {
                    let g = face.glyph_for(c).unwrap_or(0);
                    out.push(ShapedGlyph {
                        glyph: g,
                        cluster: i as u32,
                        x_advance: face.advance(g),
                        x_offset: 0.0,
                        y_offset: 0.0,
                    });
                }
            }
        }
        buf.clear();
        self.buffer = Some(buf);
        out
    }

    pub fn cached_runs(&self) -> usize {
        self.cache.len()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::test_fonts::plex_sans;

    #[test]
    fn shaped_advances_match_the_font() {
        let f = plex_sans();
        let mut s = Shaper::default();
        let g = s.shape(&f, "Hi", false, unicode_script::Script::Latin, "en");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].cluster, 0);
        assert_eq!(g[1].cluster, 1);
        let h = f.glyph_for('H').unwrap();
        assert_eq!(g[0].glyph, h);
        // Kerning may nudge the pair, but advances stay close to nominal.
        assert!((g[0].x_advance - f.advance(h)).abs() < 0.08);
    }

    #[test]
    fn kerning_changes_pair_advance() {
        let f = plex_sans();
        let mut s = Shaper::default();
        let av = s.shape(&f, "AV", false, unicode_script::Script::Latin, "en");
        let a = f.advance(f.glyph_for('A').unwrap());
        // "AV" is a classic kerned pair: A's advance shrinks.
        assert!(av[0].x_advance < a - 0.005, "{} vs {a}", av[0].x_advance);
    }

    #[test]
    fn runs_are_cached() {
        let f = plex_sans();
        let mut s = Shaper::default();
        let a = s.shape(&f, "cache me", false, unicode_script::Script::Latin, "en");
        let b = s.shape(&f, "cache me", false, unicode_script::Script::Latin, "en");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!((s.hits, s.misses), (1, 1));
        // Direction is part of the key.
        let c = s.shape(&f, "cache me", true, unicode_script::Script::Latin, "en");
        assert!(!Arc::ptr_eq(&a, &c));
    }

    #[test]
    fn rtl_run_comes_back_visual_order() {
        let f = plex_sans();
        let mut s = Shaper::default();
        let g = s.shape(&f, "abc", true, unicode_script::Script::Latin, "");
        // Visual order of an RTL run reverses the logical clusters.
        let clusters: Vec<u32> = g.iter().map(|g| g.cluster).collect();
        assert_eq!(clusters, vec![2, 1, 0]);
    }
}
