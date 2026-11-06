//! 3D text: SDF text in the world, drawn straight into the scene pass.

use serde::{Deserialize, Serialize};

use crate::{TypedComponent, UiAlign, UiTextOverflow};

/// How 3D text turns toward the camera.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Text3DBillboard {
    /// The text stays in the object's plane.
    #[default]
    None,
    /// Rotate around the world Y axis only.
    Y,
    /// Always face the camera.
    Full,
}

impl Text3DBillboard {
    /// Every mode, in declaration order.
    pub const ALL: [Text3DBillboard; 3] = [Self::None, Self::Y, Self::Full];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Y => "Y axis",
            Self::Full => "Full",
        }
    }
}

/// Text placed in the world at the object's transform (centred on its origin,
/// in the object's XY plane facing +Z). Sharp at any distance and scale; rich
/// text, shaping, bidi, overflow and localization work as for `UiText`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Text3D {
    /// The text to draw.
    pub text: String,
    /// Localization key (see `UiText::loc_key`).
    pub loc_key: String,
    /// Project-relative .ttf / .otf / .zfont / .zfontstack (empty = built-in).
    pub font: String,
    /// Font size in metres (height of one em).
    pub size: f32,
    /// Text colour, linear RGBA.
    pub color: [f32; 4],
    /// Parse rich-text tags (`<b>`, `<color=...>`, ...).
    pub rich_text: bool,
    /// Text box in metres; 0 = unbounded in that direction.
    pub width: f32,
    /// Text box height in metres; 0 = unbounded.
    pub height: f32,
    /// Horizontal alignment in the box.
    pub align_x: UiAlign,
    /// Vertical alignment in the box.
    pub align_y: UiAlign,
    /// Wrap at the box width.
    pub wrap: bool,
    /// What happens when the text does not fit the box.
    pub overflow: UiTextOverflow,
    /// Fit the size within `auto_size_min..=auto_size_max` metres.
    pub auto_size: bool,
    /// Lower bound of the auto-size range (m).
    pub auto_size_min: f32,
    /// Upper bound of the auto-size range (m).
    pub auto_size_max: f32,
    /// Draw the text bold.
    pub bold: bool,
    /// Draw the text italic.
    pub italic: bool,
    /// Spacing in em/100.
    pub char_spacing: f32,
    /// Extra space between lines, in em/100.
    pub line_spacing: f32,
    /// Outline width in em/100 (0 = none).
    pub outline_width: f32,
    /// Outline colour, linear RGBA (only when `outline_width` is above 0).
    pub outline_color: [f32; 4],
    /// How the text turns toward the camera.
    pub billboard: Text3DBillboard,
}

impl Default for Text3D {
    fn default() -> Self {
        Self {
            text: "Text".into(),
            loc_key: String::new(),
            font: String::new(),
            size: 0.1,
            color: [1.0; 4],
            rich_text: false,
            width: 0.0,
            height: 0.0,
            align_x: UiAlign::Center,
            align_y: UiAlign::Center,
            wrap: true,
            overflow: UiTextOverflow::Overflow,
            auto_size: false,
            auto_size_min: 0.02,
            auto_size_max: 0.5,
            bold: false,
            italic: false,
            char_spacing: 0.0,
            line_spacing: 0.0,
            outline_width: 0.0,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            billboard: Text3DBillboard::None,
        }
    }
}

impl TypedComponent for Text3D {
    const NAME: &'static str = "Text 3D";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_ron_loads_with_defaults() {
        let t: Text3D = ron::from_str(r#"(text: "Door", size: 0.25)"#).unwrap();
        assert_eq!((t.text.as_str(), t.size), ("Door", 0.25));
        assert_eq!(t.billboard, Text3DBillboard::None);
        assert!(t.wrap && t.width == 0.0);
    }
}
