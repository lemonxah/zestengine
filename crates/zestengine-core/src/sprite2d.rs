//! 2D sprite components (2D-3): the authoring data for the sprite render pass
//! (`zestengine-render/src/sprite2d.rs`). Plain serde data, egui-free; the
//! inspectors and the scene gather come with the engine integration.
//!
//! A sprite is referenced by its text form `"<project-relative path>#<16 hex
//! digits of the sprite's SubAssetId>"` (the same string the assets crate's
//! `sprite_ref_string` / `SpriteRef::from_ref_string` read and write; core cannot
//! name that type, assets depends on core). An empty string draws nothing.
//!
//! # Sorting
//!
//! Draw order is decided by, in this order: the sorting layer's **value**, the
//! **order in layer**, the **sorting group** (a group's renderers sort as one
//! unit against everything outside it), the material / texture, then depth.
//! Renderers store a stable [`SortingLayerId`], never the layer's position, so
//! reordering the project's [`SortingLayers`] never rewrites scene data. The
//! "Default" layer has id 0 and value 0; layers listed before it have negative
//! values and layers after it positive ones (`value = index - index_of(Default)`).

use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Stable id of a sorting layer (random-looking for new layers, 0 = Default).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SortingLayerId(pub u32);

/// The id of the built-in "Default" sorting layer.
pub const DEFAULT_SORTING_LAYER: SortingLayerId = SortingLayerId(0);

/// One project sorting layer.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SortingLayer {
    /// Stable id (0 = Default).
    pub id: SortingLayerId,
    /// Layer name.
    pub name: String,
    /// Editor picking aid only: a locked layer's sprites cannot be picked.
    pub locked: bool,
    /// Draw this layer's sprites (and every layer below it) before the 3D
    /// transparent queue, so transparent 3D objects composite over them.
    pub behind_3d: bool,
}

impl Default for SortingLayer {
    fn default() -> Self {
        Self { id: DEFAULT_SORTING_LAYER, name: "Default".to_string(), locked: false, behind_3d: false }
    }
}

/// The project's ordered sorting layers (back to front). Stored next to the
/// object-layer names in [`crate::LayerSettings`] (`sorting`), serde default =
/// one "Default" layer.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SortingLayers {
    /// Layers back to front.
    pub layers: Vec<SortingLayer>,
}

impl Default for SortingLayers {
    fn default() -> Self {
        Self { layers: vec![SortingLayer::default()] }
    }
}

impl SortingLayers {
    /// Index of the Default layer (a repaired list always has one).
    pub fn default_index(&self) -> usize {
        self.layers.iter().position(|l| l.id == DEFAULT_SORTING_LAYER).unwrap_or(0)
    }

    /// The layer's position in the list.
    pub fn index_of(&self, id: SortingLayerId) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// The layer with this id.
    pub fn get(&self, id: SortingLayerId) -> Option<&SortingLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// The layer's draw-order value: `index - index_of(Default)`, clamped to
    /// `i16`. An unknown id (a deleted layer) sorts as Default (0).
    pub fn value_of(&self, id: SortingLayerId) -> i16 {
        match self.index_of(id) {
            Some(i) => (i as i64 - self.default_index() as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16,
            None => 0,
        }
    }

    /// The layer's name ("Default" for an unknown id).
    pub fn name_of(&self, id: SortingLayerId) -> &str {
        self.get(id).map(|l| l.name.as_str()).unwrap_or("Default")
    }

    /// The highest layer value flagged `behind_3d` (layers at or below it draw
    /// behind the 3D transparents), if any is.
    pub fn behind_max(&self) -> Option<i16> {
        self.layers.iter().filter(|l| l.behind_3d).map(|l| self.value_of(l.id)).max()
    }

    /// True when the layer exists and is locked (unpickable).
    pub fn is_locked(&self, id: SortingLayerId) -> bool {
        self.get(id).is_some_and(|l| l.locked)
    }

    /// Append a layer named `name` (front-most) with a fresh stable id.
    pub fn add(&mut self, name: &str) -> SortingLayerId {
        // FNV-1a over the name, then probe until unused and non-zero: stable
        // enough to look random, deterministic for tests.
        let mut h: u32 = 0x811c_9dc5;
        for b in name.bytes().chain((self.layers.len() as u32).to_le_bytes()) {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
        while h == 0 || self.index_of(SortingLayerId(h)).is_some() {
            h = h.wrapping_add(1);
        }
        let id = SortingLayerId(h);
        self.layers.push(SortingLayer { id, name: name.to_string(), locked: false, behind_3d: false });
        id
    }

    /// Remove a layer; the Default layer cannot be removed. Returns whether
    /// anything was removed.
    pub fn remove(&mut self, id: SortingLayerId) -> bool {
        if id == DEFAULT_SORTING_LAYER {
            return false;
        }
        let before = self.layers.len();
        self.layers.retain(|l| l.id != id);
        self.layers.len() != before
    }

    /// Move the layer at `from` to position `to` (both clamped).
    pub fn move_layer(&mut self, from: usize, to: usize) {
        if self.layers.is_empty() {
            return;
        }
        let from = from.min(self.layers.len() - 1);
        let to = to.min(self.layers.len() - 1);
        let l = self.layers.remove(from);
        self.layers.insert(to, l);
    }

    /// Repair a deserialized value: a Default layer exists, ids are unique.
    pub fn normalized(mut self) -> Self {
        let mut seen = std::collections::HashSet::new();
        self.layers.retain(|l| seen.insert(l.id));
        if !self.layers.iter().any(|l| l.id == DEFAULT_SORTING_LAYER) {
            self.layers.insert(0, SortingLayer::default());
        }
        self
    }
}

/// How the sprite fills its rect.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum SpriteDrawMode {
    /// The sprite mesh at the object's scale.
    #[default]
    Simple,
    /// 9-slice: corners keep their size, edges and centre stretch to `size`.
    Sliced,
    /// 9-slice with the edges and centre repeated instead of stretched.
    Tiled,
}

impl SpriteDrawMode {
    /// Every mode, in declaration order.
    pub const ALL: [SpriteDrawMode; 3] = [Self::Simple, Self::Sliced, Self::Tiled];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Sliced => "Sliced",
            Self::Tiled => "Tiled",
        }
    }
}

/// Tiled mode's filling rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum SpriteTileMode {
    /// Whole tiles plus a clipped last one.
    #[default]
    Continuous,
    /// Stretch like Sliced until the stretch ratio passes `adaptive_threshold`,
    /// then tile.
    Adaptive,
}

impl SpriteTileMode {
    /// Every mode, in declaration order.
    pub const ALL: [SpriteTileMode; 2] = [Self::Continuous, Self::Adaptive];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Continuous => "Continuous",
            Self::Adaptive => "Adaptive",
        }
    }
}

/// How a renderer reacts to [`SpriteMask`]s.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum MaskInteraction2D {
    /// Masks do not affect this renderer.
    #[default]
    None,
    /// Drawn only inside mask shapes.
    VisibleInsideMask,
    /// Drawn only outside mask shapes.
    VisibleOutsideMask,
}

impl MaskInteraction2D {
    /// Every interaction, in declaration order.
    pub const ALL: [MaskInteraction2D; 3] = [Self::None, Self::VisibleInsideMask, Self::VisibleOutsideMask];
    /// Inspector label of this interaction.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::VisibleInsideMask => "Visible Inside Mask",
            Self::VisibleOutsideMask => "Visible Outside Mask",
        }
    }
}

/// Which point of a sprite is measured for depth tie-breaks.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum SpriteSortPoint {
    /// Sort by the sprite's centre.
    #[default]
    Center,
    /// Sort by the sprite's pivot.
    Pivot,
}

impl SpriteSortPoint {
    /// Every point, in declaration order.
    pub const ALL: [SpriteSortPoint; 2] = [Self::Center, Self::Pivot];
    /// Inspector label of this point.
    pub fn label(self) -> &'static str {
        match self {
            Self::Center => "Center",
            Self::Pivot => "Pivot",
        }
    }
}

fn white() -> [f32; 4] {
    [1.0; 4]
}

fn one_by_one() -> [f32; 2] {
    [1.0, 1.0]
}

fn adaptive_default() -> f32 {
    0.5
}

/// Draws one sprite in the XY plane of the scene (rotation about Z, scale from
/// the transform). In `Sliced` / `Tiled` mode `size` is the drawn rect in world
/// units (the transform's scale applies on top). Needs a Full Rect sprite for
/// those two modes (a tight mesh has no rect to slice).
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteRenderer {
    /// `"<path>#<sprite id hex>"`; empty draws nothing.
    pub sprite: String,
    /// Tint, linear RGBA (alpha multiplies the sprite's).
    pub color: [f32; 4],
    /// Mirrored in the vertex shader (never a negative scale: flipping does not
    /// break a batch).
    pub flip_x: bool,
    /// Mirrored vertically.
    pub flip_y: bool,
    /// How the sprite fills its rect.
    pub draw_mode: SpriteDrawMode,
    /// Drawn rect in world units (Sliced / Tiled).
    pub size: [f32; 2],
    /// Tiled mode's filling rule.
    pub tile_mode: SpriteTileMode,
    /// Adaptive tiling: the stretch ratio past which the middle tiles instead
    /// of stretching (Unity's default 0.5).
    pub adaptive_threshold: f32,
    /// Material override (`.mat` path); empty = the default sprite material.
    pub material: String,
    /// Sorting layer.
    pub sorting_layer: SortingLayerId,
    /// Sort key inside the layer.
    pub order_in_layer: i16,
    /// How sprite masks affect this renderer.
    pub mask_interaction: MaskInteraction2D,
    /// Which point depth tie-breaks use.
    pub sort_point: SpriteSortPoint,
}

impl Default for SpriteRenderer {
    fn default() -> Self {
        Self {
            sprite: String::new(),
            color: white(),
            flip_x: false,
            flip_y: false,
            draw_mode: SpriteDrawMode::Simple,
            size: one_by_one(),
            tile_mode: SpriteTileMode::Continuous,
            adaptive_threshold: adaptive_default(),
            material: String::new(),
            sorting_layer: DEFAULT_SORTING_LAYER,
            order_in_layer: 0,
            mask_interaction: MaskInteraction2D::None,
            sort_point: SpriteSortPoint::Center,
        }
    }
}

impl SpriteRenderer {
    /// Split `sprite` into its path and 64-bit sprite id; `None` when it is
    /// empty or malformed.
    pub fn sprite_parts(&self) -> Option<(&str, u64)> {
        split_sprite_ref(&self.sprite)
    }
}

/// Parse the `"<path>#<16 hex>"` sprite reference text (id must be non-zero).
pub fn split_sprite_ref(text: &str) -> Option<(&str, u64)> {
    let (path, id) = text.rsplit_once('#')?;
    if path.is_empty() || id.len() != 16 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let id = u64::from_str_radix(id, 16).ok()?;
    (id != 0).then_some((path, id))
}

impl TypedComponent for SpriteRenderer {
    const NAME: &'static str = "Sprite Renderer";
}

/// Renders its descendants' sprites (and nested groups) as one unit: they sort
/// against each other inside the group, and the group sorts against everything
/// else by its own layer and order. Nested groups flatten into the outermost
/// enabled ancestor unless a group has `sort_at_root`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SortingGroup {
    /// The group's sorting layer.
    pub sorting_layer: SortingLayerId,
    /// The group's sort key inside the layer.
    pub order_in_layer: i16,
    /// This group is a root even when nested in another group.
    pub sort_at_root: bool,
}

impl Default for SortingGroup {
    fn default() -> Self {
        Self { sorting_layer: DEFAULT_SORTING_LAYER, order_in_layer: 0, sort_at_root: false }
    }
}

impl TypedComponent for SortingGroup {
    const NAME: &'static str = "Sorting Group";
}

/// Which sorting range a [`SpriteMask`] affects.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum SpriteMaskRange {
    /// From the mask's own sort position to the next mask's.
    #[default]
    Auto,
    /// From (`back_layer`, `back_order`) to (`front_layer`, `front_order`),
    /// inclusive.
    Custom {
        /// First layer the mask affects.
        back_layer: SortingLayerId,
        /// First order in the first layer.
        back_order: i16,
        /// Last layer the mask affects.
        front_layer: SortingLayerId,
        /// Last order in the last layer.
        front_order: i16,
    },
}

fn mask_cutoff() -> f32 {
    0.2
}

/// Writes its sprite's shape into the stencil buffer; renderers with a
/// [`MaskInteraction2D`] other than `None` inside its range are then shown only
/// inside / outside that shape.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteMask {
    /// `"<path>#<sprite id hex>"` like [`SpriteRenderer::sprite`].
    pub sprite: String,
    /// Texels with alpha below this do not write the mask.
    pub alpha_cutoff: f32,
    /// Which sorting range the mask affects.
    pub range: SpriteMaskRange,
    /// Sorting layer.
    pub sorting_layer: SortingLayerId,
    /// Sort key inside the layer.
    pub order_in_layer: i16,
}

impl Default for SpriteMask {
    fn default() -> Self {
        Self {
            sprite: String::new(),
            alpha_cutoff: mask_cutoff(),
            range: SpriteMaskRange::Auto,
            sorting_layer: DEFAULT_SORTING_LAYER,
            order_in_layer: 0,
        }
    }
}

impl TypedComponent for SpriteMask {
    const NAME: &'static str = "Sprite Mask";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behind_max_is_the_highest_flagged_layer_value() {
        let mut s = SortingLayers::default();
        assert_eq!(s.behind_max(), None);
        let bg = s.add("Background");
        let fg = s.add("Foreground");
        s.move_layer(1, 0); // Background below Default
        s.move_layer(2, 2);
        assert_eq!(s.value_of(bg), -1);
        s.layers.iter_mut().find(|l| l.id == bg).unwrap().behind_3d = true;
        assert_eq!(s.behind_max(), Some(-1));
        s.layers.iter_mut().find(|l| l.id == DEFAULT_SORTING_LAYER).unwrap().behind_3d = true;
        assert_eq!(s.behind_max(), Some(0));
        assert_eq!(s.value_of(fg), 1);
        let back: SortingLayers = ron::from_str(&ron::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn default_has_one_default_layer() {
        let s = SortingLayers::default();
        assert_eq!(s.layers.len(), 1);
        assert_eq!(s.value_of(DEFAULT_SORTING_LAYER), 0);
        assert_eq!(s.name_of(DEFAULT_SORTING_LAYER), "Default");
    }

    #[test]
    fn values_are_relative_to_default() {
        let mut s = SortingLayers::default();
        let bg = s.add("Background");
        let fg = s.add("Foreground");
        s.move_layer(1, 0); // Background before Default
        assert_eq!(s.value_of(bg), -1);
        assert_eq!(s.value_of(DEFAULT_SORTING_LAYER), 0);
        assert_eq!(s.value_of(fg), 1);
        // Reordering never changes ids, only values.
        s.move_layer(0, 2);
        assert_eq!(s.value_of(bg), 2);
        assert_eq!(s.value_of(fg), 1);
    }

    #[test]
    fn unknown_id_sorts_as_default_and_default_is_permanent() {
        let mut s = SortingLayers::default();
        let a = s.add("A");
        assert!(s.remove(a));
        assert_eq!(s.value_of(a), 0);
        assert!(!s.remove(DEFAULT_SORTING_LAYER));
    }

    #[test]
    fn add_gives_unique_nonzero_ids() {
        let mut s = SortingLayers::default();
        let ids: Vec<_> = (0..50).map(|i| s.add(&format!("L{}", i % 3))).collect();
        let set: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), ids.len());
        assert!(ids.iter().all(|i| i.0 != 0));
    }

    #[test]
    fn normalized_repairs() {
        let s = SortingLayers { layers: vec![] }.normalized();
        assert_eq!(s, SortingLayers::default());
        let dup = SortingLayers { layers: vec![SortingLayer::default(), SortingLayer::default()] }.normalized();
        assert_eq!(dup.layers.len(), 1);
    }

    #[test]
    fn sprite_ref_parses() {
        assert_eq!(split_sprite_ref("art/hero.png#00000000000000ff"), Some(("art/hero.png", 0xff)));
        assert_eq!(split_sprite_ref("art/hero.png#0000000000000000"), None);
        assert_eq!(split_sprite_ref("art/hero.png"), None);
        assert_eq!(split_sprite_ref("#00000000000000ff"), None);
    }

    #[test]
    fn components_round_trip_and_default_from_empty() {
        let r = SpriteRenderer { flip_x: true, order_in_layer: -7, draw_mode: SpriteDrawMode::Tiled, ..Default::default() };
        let text = ron::to_string(&r).unwrap();
        assert_eq!(ron::from_str::<SpriteRenderer>(&text).unwrap(), r);
        // Old / partial files load with defaults.
        let d: SpriteRenderer = ron::from_str("(order_in_layer: 3)").unwrap();
        assert_eq!(d.order_in_layer, 3);
        assert_eq!(d.color, [1.0; 4]);
        assert_eq!(d.size, [1.0, 1.0]);
        let m = SpriteMask { range: SpriteMaskRange::Custom { back_layer: SortingLayerId(5), back_order: -2, front_layer: SortingLayerId(5), front_order: 9 }, ..Default::default() };
        let text = ron::to_string(&m).unwrap();
        assert_eq!(ron::from_str::<SpriteMask>(&text).unwrap(), m);
        let g: SortingGroup = ron::from_str("()").unwrap();
        assert_eq!(g, SortingGroup::default());
    }
}
