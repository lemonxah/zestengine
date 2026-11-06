//! Font assets (`.zfont`) and font stacks (`.zfontstack`): which faces to use,
//! in what order, per locale.
//!
//! Han unification means one code point needs a different face per language,
//! so a stack carries per-locale overrides (`ja`, `zh-Hans`, `ar`, ...) that go
//! in front of the default chain. The locale comes from the Localization
//! system; a language change just picks a different chain.

use serde::{Deserialize, Serialize};

/// How glyphs of a face are rasterized into the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FontMode {
    /// MTSDF for faces under ~2k glyphs, single-channel SDF above.
    #[default]
    Auto,
    Mtsdf,
    Sdf,
}

/// Raster kind of an atlas glyph (the `GlyphKey::kind` byte and shader kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GlyphKind {
    Mtsdf = 0,
    Sdf = 1,
    /// RGBA coverage bitmap (emoji / colour fonts).
    Color = 2,
    /// Flat quad (underline, strikethrough, highlight): no atlas texels.
    Solid = 3,
}

/// Characters to rasterize ahead of time so first use never hitches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CharSet {
    Ascii,
    Latin1,
    Range(u32, u32),
    Chars(String),
}

impl CharSet {
    pub fn chars(&self) -> Vec<char> {
        match self {
            Self::Ascii => (0x20u8..0x7f).map(|b| b as char).collect(),
            Self::Latin1 => (0x20u32..0x100).filter(|c| !(0x7f..0xa0).contains(c)).filter_map(char::from_u32).collect(),
            Self::Range(a, b) => (*a..=*b).filter_map(char::from_u32).collect(),
            Self::Chars(s) => s.chars().collect(),
        }
    }
}

/// `.zfont`: one source face plus how to bake it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FontAsset {
    /// Project-relative .ttf / .otf / .ttc.
    pub source: String,
    pub face_index: u32,
    pub mode: FontMode,
    /// Atlas em size in px: the raster resolution of the distance field.
    pub em_px: f32,
    /// Distance range each side of the edge, in atlas texels.
    pub spread_px: f32,
    /// Synthesized bold: SDF dilation in em (0 = none).
    pub bold_dilate: f32,
    /// Characters to rasterize at load (pinned against eviction).
    pub prewarm: Vec<CharSet>,
}

impl Default for FontAsset {
    fn default() -> Self {
        Self {
            source: String::new(),
            face_index: 0,
            mode: FontMode::Auto,
            em_px: 0.0,
            spread_px: 0.0,
            bold_dilate: 0.02,
            prewarm: vec![CharSet::Ascii],
        }
    }
}

impl FontAsset {
    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap_or_default()
    }
}

/// Glyph-count threshold between MTSDF and single-channel SDF in `Auto`.
pub const MTSDF_MAX_GLYPHS: u32 = 2000;

/// Raster sizing for a kind: (em px, spread px).
pub fn default_raster(kind: GlyphKind) -> (f32, f32) {
    match kind {
        GlyphKind::Sdf => (40.0, 5.0),
        _ => (48.0, 6.0),
    }
}

/// `.zfontstack`: primary face, fallbacks, and per-locale overrides.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FontStack {
    /// Project-relative font (.ttf/.otf/.zfont); empty = the built-in font.
    pub primary: String,
    pub fallbacks: Vec<String>,
    pub locale_overrides: Vec<LocaleFonts>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocaleFonts {
    /// BCP-47 tag or prefix: `ja`, `zh-Hans`, `ar`.
    pub lang: String,
    pub fonts: Vec<String>,
}

/// Whether override tag `pattern` applies to language `lang`: equal, or a
/// prefix on a subtag boundary (`zh` covers `zh-Hans-CN`; `zh-Hans` does not
/// cover `zh-Hant`). Case-insensitive.
pub fn lang_matches(pattern: &str, lang: &str) -> bool {
    let (p, l) = (pattern.to_ascii_lowercase().replace('_', "-"), lang.to_ascii_lowercase().replace('_', "-"));
    !p.is_empty() && (l == p || l.starts_with(&format!("{p}-")))
}

impl FontStack {
    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap_or_default()
    }

    /// Faces to try for `lang`, best first: the most specific matching
    /// overrides, then the primary, then the fallbacks. Duplicates dropped.
    pub fn chain(&self, lang: &str) -> Vec<String> {
        let mut matching: Vec<&LocaleFonts> = self.locale_overrides.iter().filter(|o| lang_matches(&o.lang, lang)).collect();
        // Longer (more specific) tags first.
        matching.sort_by_key(|o| std::cmp::Reverse(o.lang.len()));
        let mut out: Vec<String> = Vec::new();
        let mut push = |s: &str| {
            if !out.iter().any(|o| o == s) {
                out.push(s.to_owned());
            }
        };
        for o in matching {
            o.fonts.iter().for_each(|f| push(f));
        }
        push(&self.primary);
        self.fallbacks.iter().for_each(|f| push(f));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack() -> FontStack {
        FontStack {
            primary: "fonts/Latin.ttf".into(),
            fallbacks: vec!["fonts/Symbols.ttf".into()],
            locale_overrides: vec![
                LocaleFonts { lang: "ja".into(), fonts: vec!["fonts/JP.otf".into()] },
                LocaleFonts { lang: "zh".into(), fonts: vec!["fonts/SC.otf".into()] },
                LocaleFonts { lang: "zh-Hant".into(), fonts: vec!["fonts/TC.otf".into()] },
            ],
        }
    }

    #[test]
    fn default_chain_without_overrides() {
        let c = stack().chain("en");
        assert_eq!(c, vec!["fonts/Latin.ttf", "fonts/Symbols.ttf"]);
    }

    #[test]
    fn locale_override_goes_first() {
        let c = stack().chain("ja");
        assert_eq!(c, vec!["fonts/JP.otf", "fonts/Latin.ttf", "fonts/Symbols.ttf"]);
        let c = stack().chain("ja-JP");
        assert_eq!(c[0], "fonts/JP.otf");
    }

    #[test]
    fn most_specific_override_wins_and_both_apply() {
        let c = stack().chain("zh-Hant-TW");
        assert_eq!(c[0], "fonts/TC.otf");
        assert_eq!(c[1], "fonts/SC.otf", "the broader zh override stays as a fallback");
        let c = stack().chain("zh-CN");
        assert_eq!(c[0], "fonts/SC.otf");
        assert!(!c.contains(&"fonts/TC.otf".to_string()));
    }

    #[test]
    fn lang_matching_respects_subtag_boundaries() {
        assert!(lang_matches("zh", "zh-Hans"));
        assert!(lang_matches("zh_Hans", "zh-hans-CN"));
        assert!(!lang_matches("zh-Hans", "zh-Hant"));
        assert!(!lang_matches("e", "en"));
        assert!(!lang_matches("", "en"));
    }

    #[test]
    fn ron_roundtrip_and_old_files_load() {
        let s = stack();
        assert_eq!(FontStack::from_ron(&s.to_ron()).unwrap(), s);
        // Missing fields default.
        let f = FontAsset::from_ron("(source: \"a.ttf\")").unwrap();
        assert_eq!(f.source, "a.ttf");
        assert_eq!(f.mode, FontMode::Auto);
        assert!(FontStack::from_ron("()").unwrap().primary.is_empty());
    }

    #[test]
    fn charsets_expand() {
        assert_eq!(CharSet::Ascii.chars().len(), 95);
        assert!(CharSet::Latin1.chars().contains(&'é'));
        assert_eq!(CharSet::Range(0x41, 0x43).chars(), vec!['A', 'B', 'C']);
        assert_eq!(CharSet::Chars("ab".into()).chars(), vec!['a', 'b']);
    }
}
