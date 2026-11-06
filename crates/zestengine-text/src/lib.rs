//! Text for ZestEngine: font assets and stacks, shaping (harfrust), bidi and
//! UAX #14 line breaking, rich text, MTSDF/SDF glyph generation and a paged,
//! LRU-evicting atlas. Pure Rust with no renderer or egui dependency; the
//! render crate consumes [`gpu::GlyphInstance`] lists and [`atlas::AtlasUpload`]s.

pub mod atlas;
pub mod font;
pub mod gpu;
pub mod layout;
pub mod markup;
pub mod sdf;
pub mod shape;
pub mod stack;
pub mod system;

pub use font::FontId;
pub use gpu::{Billboard, GlyphInstance, TextDraw, TextHandle, TextShared, TextSpace};
pub use layout::{HAlign, LayoutParams, LinkRect, Overflow, VAlign};
pub use markup::{Len, StyleSheet};
pub use stack::{FontAsset, FontStack};
pub use system::{Effects, TextBlock, TextConfig, TextRequest, TextSystem};
