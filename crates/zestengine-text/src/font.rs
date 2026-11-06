//! Loaded fonts: metrics, character map, advances and outlines (skrifa).
//!
//! Everything a face reports is in em units (1.0 = one em), so layout is
//! independent of the size a glyph is eventually drawn at.

use std::collections::HashMap;

use skrifa::{DrawMemory, Font, GlyphId, ScalerFont};

use crate::sdf::{Shape, ShapeBuilder};

/// `Fonts/Roboto-Bold.ttf` and `roboto-bold` both become `roboto-bold`.
fn loose_key(name: &str) -> String {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = match file.rsplit_once('.') {
        Some((s, ext)) if matches!(ext.to_ascii_lowercase().as_str(), "ttf" | "otf" | "ttc" | "zfont") => s,
        _ => file,
    };
    stem.trim().to_ascii_lowercase()
}

/// Index of a face in a [`FontRegistry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FontId(pub u32);

/// Vertical metrics in em units (ascent above the baseline is positive,
/// descent below it is negative).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LineMetrics {
    pub ascent: f32,
    pub descent: f32,
    pub line_gap: f32,
    pub cap_height: f32,
    pub x_height: f32,
    pub underline_pos: f32,
    pub underline_thickness: f32,
    pub strike_pos: f32,
    pub strike_thickness: f32,
    pub sub_offset: f32,
    pub sup_offset: f32,
}

/// One face of a font file.
pub struct FontFace {
    /// Where it came from (path, or a built-in name); the cache key.
    pub name: String,
    pub font: Font,
    scaler: ScalerFont,
    pub units_per_em: f32,
    pub metrics: LineMetrics,
    pub glyph_count: u32,
}

impl FontFace {
    /// Parse `data` (a TTF/OTF, or a TTC with `index`).
    pub fn new(name: impl Into<String>, data: Vec<u8>, index: u32) -> Option<Self> {
        let font = Font::new(data, index)?;
        let scaler = ScalerFont::unhinted(font.clone(), None);
        let units_per_em = font.units_per_em().max(1) as f32;
        let m = font.metrics().scaled_f32(units_per_em);
        let line = m.h_line();
        let (mut ascent, mut descent, mut gap) = line
            .map(|l| (l.ascender, l.descender, l.line_gap))
            .unwrap_or((0.8 * units_per_em, -0.2 * units_per_em, 0.0));
        if ascent - descent <= 0.0 {
            ascent = 0.8 * units_per_em;
            descent = -0.2 * units_per_em;
            gap = 0.0;
        }
        let s = font.style_metrics().scaled_f32(units_per_em);
        let um = units_per_em;
        let deco = |d: Option<skrifa::raw::model::metrics::Decoration<f32>>, pos: f32, th: f32| {
            d.map(|d| (d.position / um, d.thickness / um)).unwrap_or((pos, th))
        };
        let (up, ut) = deco(s.underline(), -0.1, 0.06);
        let (sp, st) = deco(s.strike_through(), 0.3, 0.06);
        let metrics = LineMetrics {
            ascent: ascent / um,
            descent: descent / um,
            line_gap: gap / um,
            cap_height: m.cap_height().map(|v| v / um).unwrap_or(0.7),
            x_height: m.x_height().map(|v| v / um).unwrap_or(0.5),
            underline_pos: up,
            underline_thickness: ut.max(0.02),
            strike_pos: sp,
            strike_thickness: st.max(0.02),
            sub_offset: s.subscript().map(|v| v.y_offset / um).unwrap_or(0.14),
            sup_offset: s.superscript().map(|v| v.y_offset / um).unwrap_or(0.35),
        };
        let glyph_count = font.num_glyphs();
        Some(Self { name: name.into(), font, scaler, units_per_em, metrics, glyph_count })
    }

    /// Glyph id for a code point (None when the face has no glyph for it).
    pub fn glyph_for(&self, c: char) -> Option<u32> {
        self.font
            .charmap()
            .map_unicode(c as u32)
            .map(|g| g.to_u32())
            .filter(|&g| g != 0)
    }

    /// Whether the face covers `c`.
    pub fn has_char(&self, c: char) -> bool {
        self.glyph_for(c).is_some()
    }

    /// Horizontal advance in em units.
    pub fn advance(&self, glyph: u32) -> f32 {
        self.font.glyph_metrics().h_advance(GlyphId::new(glyph)) / self.units_per_em
    }

    /// The glyph's outline scaled to `em_px` pixels per em, y up.
    /// `None` for glyphs without an outline (spaces) or on failure.
    pub fn outline(&self, glyph: u32, em_px: f32) -> Option<Shape> {
        let outline = self.scaler.outline(GlyphId::new(glyph))?;
        let mut b = ShapeBuilder::new(em_px / self.units_per_em);
        outline.draw(DrawMemory::Scratch, &mut b).ok()?;
        let shape = b.finish();
        (!shape.is_empty()).then_some(shape)
    }

    /// Whether the face carries colour or bitmap glyphs (emoji fonts).
    pub fn has_colour_glyphs(&self) -> bool {
        !self.font.color_glyphs().is_empty() || !self.font.bitmap_strikes().is_empty()
    }
}

/// Owns every loaded face; ids are stable for the registry's lifetime.
#[derive(Default)]
pub struct FontRegistry {
    faces: Vec<FontFace>,
    by_name: HashMap<String, FontId>,
}

impl FontRegistry {
    pub fn get(&self, id: FontId) -> Option<&FontFace> {
        self.faces.get(id.0 as usize)
    }

    pub fn find(&self, name: &str) -> Option<FontId> {
        self.by_name.get(name).copied()
    }

    /// A face by a loose name (`<font="name">`): the registered name, or its file stem,
    /// compared case-insensitively with the extension optional. The first registered wins.
    pub fn find_loose(&self, name: &str) -> Option<FontId> {
        let want = loose_key(name);
        if want.is_empty() {
            return None;
        }
        self.faces.iter().position(|f| loose_key(&f.name) == want).map(|i| FontId(i as u32))
    }

    /// Register a face under its name (an already-registered name returns the
    /// existing id).
    pub fn add(&mut self, face: FontFace) -> FontId {
        if let Some(id) = self.by_name.get(&face.name) {
            return *id;
        }
        let id = FontId(self.faces.len() as u32);
        self.by_name.insert(face.name.clone(), id);
        self.faces.push(face);
        id
    }

    pub fn len(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

#[cfg(test)]
pub(crate) mod test_fonts {
    use super::*;

    /// The sans font the editor already embeds; Latin only.
    pub fn plex_sans() -> FontFace {
        let data = include_bytes!("../../zestengine-editor/assets/fonts/IBMPlexSans-Regular.ttf").to_vec();
        FontFace::new("plex-sans", data, 0).expect("embedded font parses")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_sane_em_units() {
        let f = test_fonts::plex_sans();
        let m = f.metrics;
        assert!(m.ascent > 0.5 && m.ascent < 1.3, "{m:?}");
        assert!(m.descent < 0.0 && m.descent > -0.6, "{m:?}");
        assert!(m.cap_height > 0.5 && m.cap_height < 0.9, "{m:?}");
        assert!(m.underline_thickness > 0.0);
    }

    #[test]
    fn cmap_advances_and_outlines() {
        let f = test_fonts::plex_sans();
        let a = f.glyph_for('A').expect("A");
        assert!(f.has_char('g'));
        assert!(!f.has_char('\u{4e2d}'), "Latin font has no Han");
        let adv = f.advance(a);
        assert!(adv > 0.4 && adv < 1.0, "{adv}");
        let shape = f.outline(a, 48.0).expect("A has an outline");
        let b = shape.bounds().unwrap();
        // Capital A is about cap-height tall at 48 px.
        let h = b[3] - b[1];
        assert!(h > 0.55 * 48.0 && h < 0.85 * 48.0, "{h}");
        let space = f.glyph_for(' ').unwrap();
        assert!(f.outline(space, 48.0).is_none());
    }

    #[test]
    fn registry_dedupes_by_name() {
        let mut r = FontRegistry::default();
        let a = r.add(test_fonts::plex_sans());
        let b = r.add(test_fonts::plex_sans());
        assert_eq!(a, b);
        assert_eq!(r.len(), 1);
        assert_eq!(r.find("plex-sans"), Some(a));
    }
}
