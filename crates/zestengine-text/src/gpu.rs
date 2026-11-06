//! GPU-facing data: the instance record the text pipeline pulls from a storage
//! buffer, and the per-draw description the renderer consumes.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use crate::stack::GlyphKind;
use crate::system::TextSystem;

/// Italic synthesis: horizontal shear (tan of the slant angle).
pub const ITALIC_SHEAR: f32 = 0.21;

pub const FLAG_ITALIC: u32 = 1;

/// One glyph (or flat rect) quad, 40 bytes. Positions are layout px relative
/// to the text box; the per-draw transform places the box.
///
/// `meta`: layer (bits 0-7) | kind (8-9, [`GlyphKind`]) | flags (10-15) |
/// spread in 1/8 texel (16-23). `fx`: bold dilation (i8, ±0.25 of the
/// distance range) | outline width (u8, 0..0.5) | softness (u8, 0..0.5).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GlyphInstance {
    pub pos: [f32; 2],
    pub size: [f32; 2],
    /// `u0, v0, u1, v1` normalized to the page.
    pub uv: [u16; 4],
    /// RGBA8, sRGB-encoded colour (decoded to linear in the shader).
    pub color: u32,
    pub meta: u32,
    pub fx: u32,
    /// RGBA8 outline colour.
    pub outline: u32,
}

pub fn pack_rgba(c: [u8; 4]) -> u32 {
    u32::from_le_bytes(c)
}

pub fn pack_meta(layer: u32, kind: GlyphKind, flags: u32, spread: f32) -> u32 {
    let spread_q = (spread * 8.0).round().clamp(0.0, 255.0) as u32;
    (layer & 0xff) | ((kind as u32) << 8) | ((flags & 0x3f) << 10) | (spread_q << 16)
}

/// `dilate_norm` in -0.25..0.25, `outline_norm`/`softness_norm` in 0..0.5
/// (normalized distance units, 1.0 = the full 2 * spread range).
pub fn pack_fx(dilate_norm: f32, outline_norm: f32, softness_norm: f32) -> u32 {
    let d = ((dilate_norm / 0.25).clamp(-1.0, 1.0) * 127.0).round() as i8 as u8 as u32;
    let o = ((outline_norm / 0.5).clamp(0.0, 1.0) * 255.0).round() as u32;
    let s = ((softness_norm / 0.5).clamp(0.0, 1.0) * 255.0).round() as u32;
    d | (o << 8) | (s << 16)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextSpace {
    /// Layout px to physical screen px, drawn over the finished frame.
    Screen,
    /// Layout px to world metres, drawn in the scene pass with depth testing.
    World,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Billboard {
    #[default]
    None,
    /// Rotate around the world Y axis to face the camera.
    Y,
    /// Always face the camera.
    Full,
}

/// One draw: a cached instance list plus where and how to place it.
#[derive(Clone)]
pub struct TextDraw {
    pub instances: Arc<[GlyphInstance]>,
    /// Column-major model matrix: layout px -> screen px (`Screen`) or world.
    pub transform: [[f32; 4]; 4],
    pub space: TextSpace,
    pub billboard: Billboard,
    /// Linear RGBA multiplied into every glyph.
    pub tint: [f32; 4],
    /// Layout-px offset added before `transform` (the box pivot, so rotation
    /// and billboarding turn around it).
    pub local_offset: [f32; 2],
    /// Layout-px clip rect `[x, y, w, h]` (Masking overflow).
    pub clip: Option<[f32; 4]>,
    /// Four rotated-grid samples where text is minified (far VR text).
    pub supersample: bool,
    /// Screen draws sort ascending by this.
    pub order: i32,
}

impl TextDraw {
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// Shared between the renderer (which owns the images and records draws) and
/// the UI systems (which lay text out and queue draws).
pub struct TextShared {
    pub system: Mutex<TextSystem>,
    pub draws: Mutex<Vec<TextDraw>>,
}

pub type TextHandle = Arc<TextShared>;

impl TextShared {
    pub fn new(system: TextSystem) -> TextHandle {
        Arc::new(Self { system: Mutex::new(system), draws: Mutex::new(Vec::new()) })
    }

    pub fn push(&self, draw: TextDraw) {
        if !draw.is_empty()
            && let Ok(mut q) = self.draws.lock()
        {
            // Nothing is rendering (minimized window): don't pile up frames.
            if q.len() >= 8192 {
                q.clear();
            }
            q.push(draw);
        }
    }
}

/// sRGB-encoded 0..1 colour to a linear tint.
pub fn srgb_to_linear(c: [f32; 4]) -> [f32; 4] {
    let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    [f(c[0]), f(c[1]), f(c[2]), c[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_is_forty_bytes() {
        assert_eq!(std::mem::size_of::<GlyphInstance>(), 40);
    }

    #[test]
    fn meta_and_fx_roundtrip() {
        let m = pack_meta(5, GlyphKind::Sdf, FLAG_ITALIC, 5.0);
        assert_eq!(m & 0xff, 5);
        assert_eq!((m >> 8) & 3, GlyphKind::Sdf as u32);
        assert_eq!((m >> 10) & 0x3f, FLAG_ITALIC);
        assert_eq!((m >> 16) & 0xff, 40);
        let fx = pack_fx(-0.25, 0.25, 0.5);
        assert_eq!((fx & 0xff) as u8 as i8, -127);
        assert_eq!((fx >> 8) & 0xff, 128);
        assert_eq!((fx >> 16) & 0xff, 255);
        assert_eq!(pack_fx(0.0, 0.0, 0.0), 0);
    }

    #[test]
    fn srgb_decode() {
        let l = srgb_to_linear([1.0, 0.0, 0.5, 0.3]);
        assert_eq!((l[0], l[1], l[3]), (1.0, 0.0, 0.3));
        assert!((l[2] - 0.214).abs() < 0.01);
    }
}
