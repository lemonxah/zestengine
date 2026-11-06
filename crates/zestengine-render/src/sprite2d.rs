//! The 2D sprite pass, CPU side (2D-3): everything between "a list of visible
//! sprites" and "a list of instanced draw commands", with no GPU and no
//! dependency on the engine or core (the engine fills [`SpriteSource`]s; the GPU
//! half is `sprite_gpu.rs`).
//!
//! # Pipeline
//!
//! 1. **Gather** ([`SpritePassBuilder::build`]): every [`SpriteSource`] becomes
//!    one [`SpriteInstance`] (Simple, Sliced) or several (Tiled: one per tile,
//!    expanded on the CPU; a clipped last tile shortens geometry *and* UVs, so
//!    atlased sprites never sample a neighbour). Instance counts are prefix-summed
//!    first, then the instances, packed [`SortKey`]s and batching metadata are
//!    written in parallel (std scoped threads over disjoint chunks; the result
//!    does not depend on the thread count).
//! 2. **Sort**: an LSD radix sort on the 64-bit key (stable, so equal keys keep
//!    gather order; byte passes where every key agrees are skipped).
//! 3. **Mask planning** and **batching** in one walk over the sorted stream.
//!    A batch breaks only when the *material*, the *texture* (atlas / array) or the
//!    *stencil state* changes (plus, optionally, once at the "behind 3D
//!    transparents" boundary, see [`BuildConfig::behind_layer_max`]). The layer
//!    inside an array texture is per-instance data and never breaks a batch.
//!    Consecutive instances with the same mesh range merge into one
//!    `DrawIndexedIndirect` command with `instance_count > 1`.
//!
//! # Sort key (64 bits, ascending = drawn first = furthest back)
//!
//! ```text
//! 63..56  sorting-layer value, biased +128 (clamped to -128..=127)
//! 55..40  order in layer, biased +0x8000
//! 39..24  sorting-group ordinal (0 = not in a group)
//! 23..14  batch bits: mask interaction (2) + material/texture hash (8)
//! 13..0   depth, 14 bits, farther = smaller (back to front)
//! ```
//!
//! Layer and order always come from the *root* sorting group for grouped
//! renderers, so a group sorts as one unit; the group ordinal then keeps each
//! group's renderers contiguous and in their internal order
//! ([`flatten_groups`]). The batch bits only order ties (they never affect
//! correctness: batching compares real ids).
//!
//! # Instance layout ([`SpriteInstance`], 112 bytes, vertex attributes of binding 1)
//!
//! See the struct; the vertex shader `shaders/sprite.vert` mirrors the 9-slice
//! maths of [`nine_slice_stops`] / [`slice_vertex`] with the same formulas.

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Sort key
// ---------------------------------------------------------------------------

/// Packed 64-bit sort key; see the module docs for the layout.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SortKey(pub u64);

pub const KEY_LAYER_MIN: i16 = -128;
pub const KEY_LAYER_MAX: i16 = 127;
pub const KEY_BATCH_MASK: u16 = 0x3FF;
pub const KEY_DEPTH_MASK: u16 = 0x3FFF;

impl SortKey {
    pub fn pack(layer: i16, order: i16, group: u16, batch: u16, depth: u16) -> Self {
        let l = (layer.clamp(KEY_LAYER_MIN, KEY_LAYER_MAX) as i32 + 128) as u64;
        let o = (order as i32 + 0x8000) as u64;
        Self(l << 56 | o << 40 | (group as u64) << 24 | ((batch & KEY_BATCH_MASK) as u64) << 14 | (depth & KEY_DEPTH_MASK) as u64)
    }

    pub fn layer(self) -> i16 {
        ((self.0 >> 56) as i32 - 128) as i16
    }
    pub fn order(self) -> i16 {
        (((self.0 >> 40) & 0xFFFF) as i32 - 0x8000) as i16
    }
    pub fn group(self) -> u16 {
        ((self.0 >> 24) & 0xFFFF) as u16
    }
    pub fn batch(self) -> u16 {
        ((self.0 >> 14) & KEY_BATCH_MASK as u64) as u16
    }
    pub fn depth(self) -> u16 {
        (self.0 & KEY_DEPTH_MASK as u64) as u16
    }
    /// Layer and order as one comparable 24-bit value (masks use it for ranges).
    pub fn layer_order(self) -> u32 {
        (self.0 >> 40) as u32
    }
}

/// [`SortKey::layer_order`] of a (layer value, order) pair.
pub fn layer_order(layer: i16, order: i16) -> u32 {
    SortKey::pack(layer, order, 0, 0, 0).layer_order()
}

/// Tie-break bits from the batch state: mask interaction in the top 2 bits, an
/// 8-bit hash of material and texture below. Collisions only reorder ties.
pub fn batch_bits(material: u32, texture: u32, mask: MaskMode) -> u16 {
    let h = (material.wrapping_mul(0x9E37_79B1) ^ texture.wrapping_mul(0x85EB_CA6B)).rotate_left(13) >> 24;
    (mask as u16) << 8 | h as u16
}

/// Depth along the sorting axis, quantised to 14 bits, monotone and flipped so
/// farther sprites get smaller values (drawn first). Non-finite input counts as
/// the farthest.
pub fn quantize_depth(distance: f32, near: f32, far: f32) -> u16 {
    if !distance.is_finite() {
        return 0;
    }
    let span = (far - near).max(1e-6);
    let t = ((distance - near) / span).clamp(0.0, 1.0);
    ((1.0 - t) * KEY_DEPTH_MASK as f32).round() as u16
}

/// Where depth is measured: `dot(point - origin, axis)`. A perspective camera
/// uses its own position and forward; an orthographic one its forward (view
/// depth) or a custom axis such as `(0, 1, 0)` for top-down "lower is in front"
/// (pass the *negated* axis so that larger values mean farther).
#[derive(Clone, Copy, Debug)]
pub struct SortAxis {
    pub origin: [f32; 3],
    pub axis: [f32; 3],
}

impl Default for SortAxis {
    fn default() -> Self {
        Self { origin: [0.0; 3], axis: [0.0, 0.0, -1.0] }
    }
}

impl SortAxis {
    pub fn distance(&self, p: [f32; 3]) -> f32 {
        (p[0] - self.origin[0]) * self.axis[0] + (p[1] - self.origin[1]) * self.axis[1] + (p[2] - self.origin[2]) * self.axis[2]
    }
}

// ---------------------------------------------------------------------------
// Small enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DrawMode {
    #[default]
    Simple,
    Sliced,
    Tiled,
    /// One quad over `SpriteSource::rect` with `uv_rect` (tilemap tiles).
    Quad,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TileMode {
    #[default]
    Continuous,
    Adaptive,
}

/// A renderer's reaction to sprite masks. The discriminant is the 2-bit field
/// in the sort key and in [`SpriteInstance::flags`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum MaskMode {
    #[default]
    None = 0,
    VisibleInside = 1,
    VisibleOutside = 2,
}

impl MaskMode {
    fn from_bits(b: u32) -> Self {
        match b & 3 {
            1 => Self::VisibleInside,
            2 => Self::VisibleOutside,
            _ => Self::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Stencil
// ---------------------------------------------------------------------------
//
// The sprite pass owns ONE stencil bit, bit 7 (`MASK_BIT`, 0x80): "inside some
// active mask". Mask draws write it where the mask sprite's alpha passes its
// cutoff (compare mask / write mask 0x80, so the 3D materials' stencil bits 0..6
// are neither read nor changed); clearing erases by redrawing the same shapes
// with a ZERO op (no clear rect or render area needed), and the pass erases what
// it still has written when it ends, so bit 7 is 0 before and after the pass.
// Masked sprites only read:
//   VisibleInside   passes where bit 7 is set
//   VisibleOutside  passes where bit 7 is clear
// A sprite with no mask covering it sees bit 7 clear, so VisibleInside hides it
// and VisibleOutside shows it, as in Unity. Ops are always KEEP on fail /
// depth-fail, so masked sprites never modify the buffer. A 3D material must not
// write bit 7 (reference or write mask >= 0x80) in a scene that uses sprite masks.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StencilCompare {
    Always,
    Equal,
    NotEqual,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StencilOp {
    Keep,
    Replace,
    Zero,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilState {
    pub compare: StencilCompare,
    /// Applied when the stencil test passes (the depth test is off in this pass).
    pub pass_op: StencilOp,
    pub reference: u8,
    pub compare_mask: u8,
    pub write_mask: u8,
}

/// The stencil bit the sprite pass owns (see above).
pub const MASK_BIT: u8 = 0x80;
/// The value mask draws write and masked sprites compare against.
pub const MASK_REF: u8 = MASK_BIT;

/// State of the mask-shape draws (colour writes are off in that pipeline).
pub const MASK_WRITE_STATE: StencilState =
    StencilState { compare: StencilCompare::Always, pass_op: StencilOp::Replace, reference: MASK_REF, compare_mask: MASK_BIT, write_mask: MASK_BIT };

/// State of the erase draws: the written mask shapes replayed to reset to 0.
pub const MASK_ERASE_STATE: StencilState =
    StencilState { compare: StencilCompare::Always, pass_op: StencilOp::Zero, reference: 0, compare_mask: MASK_BIT, write_mask: MASK_BIT };

impl MaskMode {
    /// The read-only test state for a masked sprite; `None` = stencil off.
    pub fn test_state(self) -> Option<StencilState> {
        let compare = match self {
            Self::None => return None,
            Self::VisibleInside => StencilCompare::Equal,
            Self::VisibleOutside => StencilCompare::NotEqual,
        };
        Some(StencilState { compare, pass_op: StencilOp::Keep, reference: MASK_REF, compare_mask: MASK_BIT, write_mask: 0 })
    }
}

// ---------------------------------------------------------------------------
// Pooled sprite meshes
// ---------------------------------------------------------------------------

/// One vertex of the pooled sprite mesh buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub struct SpriteVertex {
    /// Sprite meshes: pivot-relative local position, world units, y up.
    /// Templates: a grid coordinate (quad 0..1, 9-slice 0..3).
    pub pos: [f32; 2],
    /// Sprite meshes: UV into the sprite's texture or atlas page (v = 0 at the
    /// top). Templates: unused.
    pub uv: [f32; 2],
}

/// A mesh inside the pool: `u16` indices are relative to `vertex_offset`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MeshRange {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

/// The unit quad template (4 vertices, 6 indices) tiles are drawn with.
pub const QUAD_TEMPLATE: MeshRange = MeshRange { first_index: 0, index_count: 6, vertex_offset: 0 };
/// The 4 x 4 vertex 9-slice template (16 vertices, 54 indices).
pub const SLICE_TEMPLATE: MeshRange = MeshRange { first_index: 6, index_count: 54, vertex_offset: 4 };
const TEMPLATE_VERTICES: usize = 20;
const TEMPLATE_INDICES: usize = 60;

/// CPU side of the one global static sprite vertex / index buffer. Append-only
/// (meshes are cooked, immutable assets), so a region the GPU has seen never
/// changes. The first [`TEMPLATE_VERTICES`] / [`TEMPLATE_INDICES`] entries are
/// the quad and 9-slice templates.
pub struct SpriteMeshPool {
    vertices: Vec<SpriteVertex>,
    indices: Vec<u16>,
    max_vertices: usize,
    max_indices: usize,
    uploaded_vertices: usize,
    uploaded_indices: usize,
    /// Skinned meshes rewritten each frame (see [`DynamicMesh`]).
    dynamic: Vec<DynMeshData>,
    /// Vertex ranges `(first, count)` of already-uploaded vertices rewritten since the last upload.
    dirty: Vec<(usize, usize)>,
    /// Rewritable meshes with a changing topology (sprite shapes, 2D-8).
    shapes: Vec<ShapeMeshData>,
    /// Index ranges `(first, count)` of already-uploaded indices rewritten since the last upload.
    dirty_indices: Vec<(usize, usize)>,
}

/// Slots (copies of a dynamic mesh's vertices) per skinned mesh. A write goes to
/// the slot after the last written one, so the GPU can still read the previous
/// slots of frames in flight; this must exceed `FRAMES_IN_FLIGHT` (2).
pub const DYNAMIC_SLOTS: usize = 3;

/// Handle of a CPU-skinned sprite mesh (2D-7): fixed UVs and indices, positions
/// rewritten by [`SpriteMeshPool::write_dynamic`] when the pose changes. A
/// skinned sprite whose pose is unchanged costs nothing per frame: its last
/// slot stays valid and keeps being drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicMesh(u32);

/// Handle of a rewritable mesh whose vertex *and* index lists change size
/// (2D-8 sprite shapes): [`DYNAMIC_SLOTS`] blocks of `cap_vertices` /
/// `cap_indices`, written round-robin like a [`DynamicMesh`]. A shape that grows
/// past its capacity asks for a bigger mesh and releases the old one
/// ([`SpriteMeshPool::release_shape_mesh`]); released meshes are reused by any
/// request they are big enough for, so a spline being dragged does not eat the
/// pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeMesh(u32);

#[derive(Clone, Copy)]
struct ShapeSlot {
    vertex_offset: i32,
    first_index: u32,
}

struct ShapeMeshData {
    cap_vertices: usize,
    cap_indices: usize,
    slots: [ShapeSlot; DYNAMIC_SLOTS],
    current: Option<usize>,
    free: bool,
}

struct DynMeshData {
    slots: [MeshRange; DYNAMIC_SLOTS],
    vertex_count: usize,
    /// Slot written last (`None` until the first write).
    current: Option<usize>,
}

impl SpriteMeshPool {
    pub fn new(max_vertices: usize, max_indices: usize) -> Self {
        let mut vertices = Vec::with_capacity(TEMPLATE_VERTICES);
        let mut indices = Vec::with_capacity(TEMPLATE_INDICES);
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            vertices.push(SpriteVertex { pos: [x, y], uv: [0.0; 2] });
        }
        indices.extend_from_slice(&[0, 1, 2, 2, 1, 3]);
        for j in 0..4u16 {
            for i in 0..4u16 {
                vertices.push(SpriteVertex { pos: [i as f32, j as f32], uv: [0.0; 2] });
            }
        }
        for j in 0..3u16 {
            for i in 0..3u16 {
                let v00 = j * 4 + i;
                indices.extend_from_slice(&[v00, v00 + 1, v00 + 4, v00 + 4, v00 + 1, v00 + 5]);
            }
        }
        debug_assert_eq!(vertices.len(), TEMPLATE_VERTICES);
        debug_assert_eq!(indices.len(), TEMPLATE_INDICES);
        Self { vertices, indices, max_vertices: max_vertices.max(TEMPLATE_VERTICES), max_indices: max_indices.max(TEMPLATE_INDICES), uploaded_vertices: 0, uploaded_indices: 0, dynamic: Vec::new(), dirty: Vec::new(), shapes: Vec::new(), dirty_indices: Vec::new() }
    }

    /// Append a sprite mesh (pivot-relative positions, UVs, CCW `u16` indices).
    /// `None` when the lengths disagree, an index is out of range, or the pool is full.
    pub fn add(&mut self, positions: &[[f32; 2]], uvs: &[[f32; 2]], indices: &[u16]) -> Option<MeshRange> {
        let n = positions.len();
        if n == 0 || n != uvs.len() || indices.is_empty() || indices.iter().any(|&i| i as usize >= n) {
            return None;
        }
        if self.vertices.len() + n > self.max_vertices || self.indices.len() + indices.len() > self.max_indices {
            return None;
        }
        let range = MeshRange { first_index: self.indices.len() as u32, index_count: indices.len() as u32, vertex_offset: self.vertices.len() as i32 };
        self.vertices.extend(positions.iter().zip(uvs).map(|(&pos, &uv)| SpriteVertex { pos, uv }));
        self.indices.extend_from_slice(indices);
        Some(range)
    }

    /// Allocate a skinned mesh: [`DYNAMIC_SLOTS`] vertex blocks sharing one index
    /// list. Positions start at the origin; call [`Self::write_dynamic`] before
    /// drawing it. `None` for bad input or a full pool.
    pub fn add_dynamic(&mut self, uvs: &[[f32; 2]], indices: &[u16]) -> Option<DynamicMesh> {
        let n = uvs.len();
        if n == 0 || indices.is_empty() || indices.iter().any(|&i| i as usize >= n) {
            return None;
        }
        if self.vertices.len() + n * DYNAMIC_SLOTS > self.max_vertices || self.indices.len() + indices.len() > self.max_indices {
            return None;
        }
        let first_index = self.indices.len() as u32;
        self.indices.extend_from_slice(indices);
        let mut slots = [MeshRange::default(); DYNAMIC_SLOTS];
        for slot in slots.iter_mut() {
            *slot = MeshRange { first_index, index_count: indices.len() as u32, vertex_offset: self.vertices.len() as i32 };
            self.vertices.extend(uvs.iter().map(|&uv| SpriteVertex { pos: [0.0; 2], uv }));
        }
        self.dynamic.push(DynMeshData { slots, vertex_count: n, current: None });
        Some(DynamicMesh(self.dynamic.len() as u32 - 1))
    }

    /// Store new positions in the next slot and return the range to draw.
    /// `None` when the handle is unknown or the length differs.
    pub fn write_dynamic(&mut self, d: DynamicMesh, positions: &[[f32; 2]]) -> Option<MeshRange> {
        let data = self.dynamic.get_mut(d.0 as usize)?;
        if positions.len() != data.vertex_count {
            return None;
        }
        let slot = data.current.map_or(0, |c| (c + 1) % DYNAMIC_SLOTS);
        data.current = Some(slot);
        let range = data.slots[slot];
        let first = range.vertex_offset as usize;
        for (v, &p) in self.vertices[first..first + positions.len()].iter_mut().zip(positions) {
            v.pos = p;
        }
        // Vertices not uploaded yet go out with the pending tail.
        if first < self.uploaded_vertices {
            self.dirty.push((first, positions.len()));
        }
        Some(range)
    }

    /// Allocate a rewritable shape mesh with room for `cap_vertices` /
    /// `cap_indices` (at most 65535 vertices). A released mesh at least that big
    /// and at most four times as big is reused. `None` for a bad size or a full pool.
    pub fn add_shape_mesh(&mut self, cap_vertices: usize, cap_indices: usize) -> Option<ShapeMesh> {
        if cap_vertices == 0 || cap_indices == 0 || cap_vertices > u16::MAX as usize {
            return None;
        }
        if let Some(i) = self.shapes.iter().position(|m| m.free && m.cap_vertices >= cap_vertices && m.cap_indices >= cap_indices && m.cap_vertices <= cap_vertices * 4 + 64) {
            self.shapes[i].free = false;
            return Some(ShapeMesh(i as u32));
        }
        if self.vertices.len() + cap_vertices * DYNAMIC_SLOTS > self.max_vertices || self.indices.len() + cap_indices * DYNAMIC_SLOTS > self.max_indices {
            return None;
        }
        let mut slots = [ShapeSlot { vertex_offset: 0, first_index: 0 }; DYNAMIC_SLOTS];
        for slot in slots.iter_mut() {
            *slot = ShapeSlot { vertex_offset: self.vertices.len() as i32, first_index: self.indices.len() as u32 };
            self.vertices.extend(std::iter::repeat_n(SpriteVertex { pos: [0.0; 2], uv: [0.0; 2] }, cap_vertices));
            self.indices.extend(std::iter::repeat_n(0u16, cap_indices));
        }
        self.shapes.push(ShapeMeshData { cap_vertices, cap_indices, slots, current: None, free: false });
        Some(ShapeMesh(self.shapes.len() as u32 - 1))
    }

    /// Store a new mesh in the next slot and return the range to draw. `None`
    /// when the handle is unknown or released, the lists disagree, an index is
    /// out of range, or the mesh does not fit the capacity.
    pub fn write_shape_mesh(&mut self, h: ShapeMesh, positions: &[[f32; 2]], uvs: &[[f32; 2]], indices: &[u16]) -> Option<MeshRange> {
        let data = self.shapes.get_mut(h.0 as usize).filter(|d| !d.free)?;
        let n = positions.len();
        if n != uvs.len() || n > data.cap_vertices || indices.len() > data.cap_indices || indices.iter().any(|&i| i as usize >= n) {
            return None;
        }
        let slot = data.current.map_or(0, |c| (c + 1) % DYNAMIC_SLOTS);
        data.current = Some(slot);
        let ShapeSlot { vertex_offset, first_index } = data.slots[slot];
        let first = vertex_offset as usize;
        for ((v, &p), &uv) in self.vertices[first..first + n].iter_mut().zip(positions).zip(uvs) {
            *v = SpriteVertex { pos: p, uv };
        }
        let fi = first_index as usize;
        self.indices[fi..fi + indices.len()].copy_from_slice(indices);
        if n > 0 && first < self.uploaded_vertices {
            self.dirty.push((first, n));
        }
        if !indices.is_empty() && fi < self.uploaded_indices {
            self.dirty_indices.push((fi, indices.len()));
        }
        Some(MeshRange { first_index, index_count: indices.len() as u32, vertex_offset })
    }

    /// Hand a shape mesh back for reuse. Its last range stays valid for frames already in flight.
    pub fn release_shape_mesh(&mut self, h: ShapeMesh) {
        if let Some(d) = self.shapes.get_mut(h.0 as usize) {
            d.free = true;
        }
    }

    /// `(vertices, indices)` the mesh can hold.
    pub fn shape_mesh_capacity(&self, h: ShapeMesh) -> Option<(usize, usize)> {
        self.shapes.get(h.0 as usize).map(|d| (d.cap_vertices, d.cap_indices))
    }

    /// Shape meshes allocated (released ones included).
    pub fn shape_mesh_count(&self) -> usize {
        self.shapes.len()
    }

    /// Rewritten index ranges to upload, `(first index, count)`; hand the list
    /// back with [`Self::recycle_dirty_indices`].
    pub fn take_dirty_indices(&mut self) -> Vec<(usize, usize)> {
        std::mem::take(&mut self.dirty_indices)
    }

    pub fn recycle_dirty_indices(&mut self, mut v: Vec<(usize, usize)>) {
        v.clear();
        if self.dirty_indices.is_empty() {
            self.dirty_indices = v;
        }
    }

    /// The range last written (`None` before the first write).
    pub fn dynamic_range(&self, d: DynamicMesh) -> Option<MeshRange> {
        let data = self.dynamic.get(d.0 as usize)?;
        data.current.map(|c| data.slots[c])
    }

    /// Rewritten vertex ranges to upload, `(first vertex, count)`; hand the
    /// list back with [`Self::recycle_dirty`].
    pub fn take_dirty(&mut self) -> Vec<(usize, usize)> {
        std::mem::take(&mut self.dirty)
    }

    pub fn recycle_dirty(&mut self, mut v: Vec<(usize, usize)>) {
        v.clear();
        if self.dirty.is_empty() {
            self.dirty = v;
        }
    }

    /// Skinned meshes allocated.
    pub fn dynamic_count(&self) -> usize {
        self.dynamic.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }
    pub fn vertices(&self) -> &[SpriteVertex] {
        &self.vertices
    }
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// The not-yet-uploaded tails: `(first vertex, vertices, first index, indices)`.
    /// Marks them uploaded.
    pub fn take_pending(&mut self) -> (usize, &[SpriteVertex], usize, &[u16]) {
        let (v0, i0) = (self.uploaded_vertices, self.uploaded_indices);
        self.uploaded_vertices = self.vertices.len();
        self.uploaded_indices = self.indices.len();
        (v0, &self.vertices[v0..], i0, &self.indices[i0..])
    }
}

// ---------------------------------------------------------------------------
// Per-instance and per-draw GPU data
// ---------------------------------------------------------------------------

/// `flags` bits.
pub const FLAG_FLIP_X: u32 = 1;
pub const FLAG_FLIP_Y: u32 = 2;
/// Bits 2..3: geometry mode (see `MODE_*`).
pub const FLAG_MODE_SHIFT: u32 = 2;
/// Bits 5..6: [`MaskMode`] (for the CPU walk; the shaders ignore them).
pub const FLAG_MASK_SHIFT: u32 = 5;
/// The sprite's content is stored turned 90 degrees clockwise in its atlas
/// page; MODE_QUAD / MODE_SLICE draws turn the UVs back (see [`quad_uv`]).
pub const FLAG_ROTATED: u32 = 1 << 7;
/// Tile animation: `anim` indexes the frame table (see [`AnimHeader`]); the
/// vertex shader picks the frame from the frame's time push constant.
pub const FLAG_ANIM: u32 = 1 << 8;
/// The quad lies in the world XZ plane (a ground tilemap): `pos` is world (x, z) and `z` the height.
pub const FLAG_XZ: u32 = 1 << 9;
/// OR'd into a sprite's material id to draw it in the XZ plane with the depth test on (batches by
/// material, so XZ sprites never share a draw with overlay sprites). The GPU side masks it off to find
/// the material proper.
pub const SPRITE_MATERIAL_XZ: u32 = 1 << 30;
/// Pooled sprite mesh: positions and UVs from the vertex buffer.
pub const MODE_MESH: u32 = 0;
/// 9-slice template: positions and UVs computed from `rect`, `border`, `uv_*`.
pub const MODE_SLICE: u32 = 1;
/// Unit-quad template: one rect, one UV rect (tiles).
pub const MODE_QUAD: u32 = 2;

/// Per-instance draw data (112 bytes, vertex attributes on binding 1).
///
/// World position = `affine` (columns `(a, b)`, `(c, d)`: rotation about Z times
/// scale) applied to the local position, plus `pos` / `z`. Local geometry is
/// pivot-relative; flips mirror it about the pivot (never a negative scale).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub struct SpriteInstance {
    /// `a, b, c, d` of the 2x2 part.
    pub affine: [f32; 4],
    pub pos: [f32; 2],
    pub z: f32,
    /// See the `FLAG_*` constants.
    pub flags: u32,
    /// `u0, v0, u1, v1` of the sprite (or tile) in its texture; `v0` is the top.
    pub uv_rect: [f32; 4],
    /// Local drawn rect: `min x, min y, size x, size y` (Slice / Quad modes).
    pub rect: [f32; 4],
    /// 9-slice border, world units: left, bottom, right, top.
    pub border: [f32; 4],
    /// The same border as UV distances: left du, bottom dv, right du, top dv.
    pub uv_border: [f32; 4],
    /// Linear RGBA8 tint (`unpackUnorm4x8`).
    pub color: u32,
    /// Array layer inside the batch's atlas texture (0 for plain textures).
    pub layer: u32,
    /// Texels with alpha below this are discarded (mask draws; 0 = off).
    pub alpha_cutoff: f32,
    /// Index of the animation header in the frame table (FLAG_ANIM only).
    pub anim: u32,
}

impl SpriteInstance {
    pub fn mode(&self) -> u32 {
        (self.flags >> FLAG_MODE_SHIFT) & 3
    }
    pub fn mask_mode(&self) -> MaskMode {
        MaskMode::from_bits(self.flags >> FLAG_MASK_SHIFT)
    }
}

/// `VkDrawIndexedIndirectCommand`, byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq, Eq)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

// ---------------------------------------------------------------------------
// 9-slice maths (mirrored by shaders/sprite.vert)
// ---------------------------------------------------------------------------

/// Positions of the four grid lines along one axis of a 9-slice drawn at
/// `size`, with `lo` / `hi` the border widths at the low / high end: `[0, lo',
/// size - hi', size]`. When the target is smaller than the border sum both
/// borders shrink by the same factor so they meet in the middle (corner sizes
/// stay in proportion; the centre collapses to zero width).
pub fn nine_slice_stops(size: f32, lo: f32, hi: f32) -> [f32; 4] {
    let size = size.max(0.0);
    let (lo, hi) = (lo.max(0.0), hi.max(0.0));
    let sum = lo + hi;
    let s = if sum > size && sum > 0.0 { size / sum } else { 1.0 };
    [0.0, lo * s, size - hi * s, size]
}

/// UV lines along one axis, matching [`nine_slice_stops`]'s four lines. `uv_lo`
/// is the UV at grid line 0, `uv_hi` at grid line 3, `d_lo` / `d_hi` the border
/// UV distances measured inwards from each end.
pub fn nine_slice_uv_stops(uv_lo: f32, uv_hi: f32, d_lo: f32, d_hi: f32) -> [f32; 4] {
    [uv_lo, uv_lo + d_lo, uv_hi - d_hi, uv_hi]
}

/// Local position and UV of vertex `(i, j)` (0..=3 each; j = 0 is the bottom
/// row) of the 9-slice template. `rect` = local `min x, min y, w, h`,
/// `border` = world left/bottom/right/top, `uv_rect` = `u0, v0, u1, v1` (v0 top),
/// `uv_border` = UV distances left/bottom/right/top. Flips mirror the local
/// position about the pivot after the layout, exactly as the shader does.
pub fn slice_vertex(i: usize, j: usize, rect: [f32; 4], border: [f32; 4], uv_rect: [f32; 4], uv_border: [f32; 4], flip: [bool; 2]) -> ([f32; 2], [f32; 2]) {
    let xs = nine_slice_stops(rect[2], border[0], border[2]);
    let ys = nine_slice_stops(rect[3], border[1], border[3]);
    let us = nine_slice_uv_stops(uv_rect[0], uv_rect[2], uv_border[0], uv_border[2]);
    // v runs top-down in the image, y runs bottom-up in the sprite: the bottom
    // grid line (j = 0) is v1, the top (j = 3) is v0.
    let vs = [uv_rect[3], uv_rect[3] - uv_border[1], uv_rect[1] + uv_border[3], uv_rect[1]];
    let mut p = [rect[0] + xs[i], rect[1] + ys[j]];
    if flip[0] {
        p[0] = -p[0];
    }
    if flip[1] {
        p[1] = -p[1];
    }
    (p, [us[i], vs[j]])
}

// ---------------------------------------------------------------------------
// Tiling
// ---------------------------------------------------------------------------

/// Most tiles one Tiled renderer may expand into; beyond it the renderer draws
/// as Sliced (stretched) instead of flooding the instance buffer.
pub const MAX_TILES_PER_SPRITE: usize = 4096;

/// What tiling a Tiled sprite needs.
#[derive(Clone, Copy, Debug)]
pub struct TileParams {
    /// The sprite's rect in world units.
    pub sprite_size: [f32; 2],
    /// Left, bottom, right, top, world units.
    pub border: [f32; 4],
    /// Pivot from the rect's bottom-left, world units.
    pub pivot: [f32; 2],
    /// Drawn size, world units.
    pub size: [f32; 2],
    pub mode: TileMode,
    /// Adaptive: stretch until `span / tile >= 1 + threshold`, then tile.
    pub threshold: f32,
    pub uv_rect: [f32; 4],
}

/// One tile: local rect and its UV rect (`u0, v0, u1, v1`, v0 the top edge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileQuad {
    pub min: [f32; 2],
    pub size: [f32; 2],
    pub uv: [f32; 4],
}

const SPAN_EPS: f32 = 1e-6;

/// Tiling of one axis: three regions (low corner, middle, high corner).
#[derive(Clone, Copy, Debug)]
struct AxisPlan {
    /// Region start along the axis (local, before the pivot offset) and length.
    start: [f32; 3],
    span: [f32; 3],
    /// Tiles per region and each tile's length.
    n: [u32; 3],
    tile: [f32; 3],
}

fn axis_plan(size: f32, lo: f32, hi: f32, inner: f32, mode: TileMode, threshold: f32) -> AxisPlan {
    let stops = nine_slice_stops(size, lo, hi);
    let start = [stops[0], stops[1], stops[2]];
    let span = [stops[1] - stops[0], stops[2] - stops[1], stops[3] - stops[2]];
    let mut n = [0u32; 3];
    let mut tile = span;
    for k in 0..3 {
        if span[k] <= SPAN_EPS {
            continue;
        }
        n[k] = 1;
        if k == 1 && inner > SPAN_EPS {
            let stretched = mode == TileMode::Adaptive && span[1] / inner < 1.0 + threshold.max(0.0);
            if !stretched {
                n[1] = ((span[1] / inner) - 1e-4).ceil().max(1.0) as u32;
                tile[1] = inner;
            }
        }
    }
    AxisPlan { start, span, n, tile }
}

fn axis_plans(p: &TileParams) -> (AxisPlan, AxisPlan) {
    let inner_w = p.sprite_size[0] - p.border[0] - p.border[2];
    let inner_h = p.sprite_size[1] - p.border[1] - p.border[3];
    (
        axis_plan(p.size[0], p.border[0], p.border[2], inner_w, p.mode, p.threshold),
        axis_plan(p.size[1], p.border[1], p.border[3], inner_h, p.mode, p.threshold),
    )
}

/// Instances a Tiled renderer expands into (the same plan [`for_each_tile`]
/// walks). `None` when it would exceed [`MAX_TILES_PER_SPRITE`].
pub fn tile_count(p: &TileParams) -> Option<usize> {
    let (x, y) = axis_plans(p);
    let sum = |a: &AxisPlan| a.n.iter().map(|&n| n as usize).sum::<usize>();
    // Regions multiply per axis pair; the sum of products factorises.
    let total = sum(&x).checked_mul(sum(&y))?;
    (total <= MAX_TILES_PER_SPRITE).then_some(total)
}

/// Visit every tile of a Tiled renderer (corners, tiled edges, tiled centre).
/// Call only when [`tile_count`] is `Some`.
pub fn for_each_tile(p: &TileParams, f: &mut impl FnMut(TileQuad)) {
    let (xp, yp) = axis_plans(p);
    let w = p.sprite_size[0].max(SPAN_EPS);
    let h = p.sprite_size[1].max(SPAN_EPS);
    let du = [p.border[0] / w * (p.uv_rect[2] - p.uv_rect[0]), p.border[2] / w * (p.uv_rect[2] - p.uv_rect[0])];
    let dv = [p.border[1] / h * (p.uv_rect[3] - p.uv_rect[1]), p.border[3] / h * (p.uv_rect[3] - p.uv_rect[1])];
    let us = nine_slice_uv_stops(p.uv_rect[0], p.uv_rect[2], du[0], du[1]);
    let vs = [p.uv_rect[3], p.uv_rect[3] - dv[0], p.uv_rect[1] + dv[1], p.uv_rect[1]];
    let origin = [-p.pivot[0] / w * p.size[0], -p.pivot[1] / h * p.size[1]];
    for ry in 0..3 {
        for rx in 0..3 {
            let (nx, ny) = (xp.n[rx], yp.n[ry]);
            // Source UV rect of the region: u left to right, v from the low-y
            // edge (`va`) to the high-y edge (`vb`).
            let (ua, ub) = (us[rx], us[rx + 1]);
            let (va, vb) = (vs[ry], vs[ry + 1]);
            for ty in 0..ny {
                let y0 = ty as f32 * yp.tile[ry];
                let y1 = (y0 + yp.tile[ry]).min(yp.span[ry]);
                // Fraction of the tile shown (a clipped last tile shows less of it).
                let sy = (y1 - y0) / yp.tile[ry];
                for tx in 0..nx {
                    let x0 = tx as f32 * xp.tile[rx];
                    let x1 = (x0 + xp.tile[rx]).min(xp.span[rx]);
                    let sx = (x1 - x0) / xp.tile[rx];
                    f(TileQuad {
                        min: [origin[0] + xp.start[rx] + x0, origin[1] + yp.start[ry] + y0],
                        size: [x1 - x0, y1 - y0],
                        uv: [ua, va + (vb - va) * sy, ua + (ub - ua) * sx, va],
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sorting groups
// ---------------------------------------------------------------------------

/// A sorting group node: parent index, its own layer value and order.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroupNode {
    pub parent: Option<u32>,
    pub layer: i16,
    pub order: i16,
    /// Stays a root even when nested in another group.
    pub sort_at_root: bool,
    /// Distance along the sort axis (larger = farther), orders roots that tie.
    pub depth: f32,
}

/// A renderer's innermost group and own layer / order.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroupMember {
    pub group: Option<u32>,
    pub layer: i16,
    pub order: i16,
}

/// What the key gets for one renderer: effective layer / order (the root
/// group's for grouped renderers) and the group ordinal (0 = ungrouped).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GroupAssign {
    pub layer: i16,
    pub order: i16,
    pub ordinal: u16,
}

/// Renderers per frame that fit the 16-bit ordinal.
pub const MAX_GROUPED: usize = u16::MAX as usize;

/// Reusable scratch for [`flatten_groups`].
#[derive(Default)]
pub struct GroupScratch {
    root_of: Vec<u32>,
    roots: Vec<u32>,
    /// `(group, layer, order, element)`; element = renderer index, or
    /// `GROUP_BIT | group` for a nested group.
    items: Vec<(u32, i16, i16, u32)>,
    first: Vec<u32>,
    stack: Vec<u32>,
}

const GROUP_BIT: u32 = 0x8000_0000;

/// Flatten sorting groups the way Unity does (own algorithm, same semantics):
/// each root group gathers its renderers and nested groups, stable-sorts them by
/// (layer, order), and numbers its renderers depth-first, a nested group taking
/// its slot among its siblings and then recursing. Roots are numbered in
/// (layer, order, farther-first, index) order, so the ordinal of a renderer is a
/// single number that keeps every group contiguous in the final sort. Returns
/// how many grouped renderers did not fit the 16-bit ordinal (they share the
/// last ordinal).
pub fn flatten_groups(groups: &[GroupNode], members: &[GroupMember], scratch: &mut GroupScratch, out: &mut Vec<GroupAssign>) -> usize {
    out.clear();
    let ng = groups.len();
    // Root of every group (walk up, stop at sort_at_root; the step bound breaks cycles).
    scratch.root_of.clear();
    for g in 0..ng {
        let mut r = g;
        let mut steps = 0;
        while !groups[r].sort_at_root {
            match groups[r].parent {
                Some(p) if (p as usize) < ng && steps <= ng => {
                    r = p as usize;
                    steps += 1;
                }
                _ => break,
            }
        }
        scratch.root_of.push(r as u32);
    }
    let valid_group = |g: Option<u32>| g.filter(|&g| (g as usize) < ng);
    // Children lists.
    scratch.items.clear();
    for (i, m) in members.iter().enumerate() {
        if let Some(g) = valid_group(m.group) {
            scratch.items.push((g, m.layer, m.order, i as u32));
        }
    }
    for (g, n) in groups.iter().enumerate() {
        // A nested (non-root) group is an element of its parent.
        if scratch.root_of[g] as usize != g
            && let Some(p) = valid_group(n.parent)
        {
            scratch.items.push((p, n.layer, n.order, GROUP_BIT | g as u32));
        }
    }
    // Stable: equal (group, layer, order) keep renderer-then-group, index order.
    scratch.items.sort_by_key(|&(g, l, o, _)| (g, l, o));
    scratch.first.clear();
    scratch.first.resize(ng + 1, 0);
    for &(g, ..) in &scratch.items {
        scratch.first[g as usize + 1] += 1;
    }
    for g in 0..ng {
        scratch.first[g + 1] += scratch.first[g];
    }
    // Roots in key order.
    scratch.roots.clear();
    scratch.roots.extend((0..ng as u32).filter(|&g| scratch.root_of[g as usize] == g));
    scratch.roots.sort_by(|&a, &b| {
        let (na, nb) = (&groups[a as usize], &groups[b as usize]);
        (na.layer, na.order)
            .cmp(&(nb.layer, nb.order))
            .then(nb.depth.total_cmp(&na.depth))
            .then(a.cmp(&b))
    });
    out.resize(members.len(), GroupAssign::default());
    for (i, m) in members.iter().enumerate() {
        out[i] = GroupAssign { layer: m.layer, order: m.order, ordinal: 0 };
    }
    let mut counter = 0usize;
    let mut overflow = 0usize;
    for ri in 0..scratch.roots.len() {
        let root = scratch.roots[ri];
        let (rl, ro) = (groups[root as usize].layer, groups[root as usize].order);
        // Iterative depth-first walk: a nested group is expanded in place by
        // pushing its element range in reverse, so it takes its slot among its
        // siblings and its renderers number consecutively.
        scratch.stack.clear();
        let (a, b) = (scratch.first[root as usize] as usize, scratch.first[root as usize + 1] as usize);
        scratch.stack.extend(scratch.items[a..b].iter().rev().map(|&(.., e)| e));
        while let Some(e) = scratch.stack.pop() {
            if e & GROUP_BIT != 0 {
                let g = (e & !GROUP_BIT) as usize;
                let (a, b) = (scratch.first[g] as usize, scratch.first[g + 1] as usize);
                scratch.stack.extend(scratch.items[a..b].iter().rev().map(|&(.., e)| e));
            } else {
                counter += 1;
                if counter > MAX_GROUPED {
                    overflow += 1;
                }
                out[e as usize] = GroupAssign { layer: rl, order: ro, ordinal: counter.min(MAX_GROUPED) as u16 };
            }
        }
    }
    overflow
}

// ---------------------------------------------------------------------------
// Radix sort
// ---------------------------------------------------------------------------

/// Reusable LSD radix sorter for `(u64 key, u32 payload)` pairs.
#[derive(Default)]
pub struct RadixSorter {
    keys2: Vec<u64>,
    vals2: Vec<u32>,
}

impl RadixSorter {
    /// Stable ascending sort of `keys` with `vals` carried along. Eight byte
    /// passes; a pass is skipped when every key has the same byte there.
    pub fn sort(&mut self, keys: &mut Vec<u64>, vals: &mut Vec<u32>) {
        let n = keys.len();
        debug_assert_eq!(n, vals.len());
        if n < 2 {
            return;
        }
        let mut hist = [[0u32; 256]; 8];
        for &k in keys.iter() {
            for (b, h) in hist.iter_mut().enumerate() {
                h[((k >> (b * 8)) & 0xFF) as usize] += 1;
            }
        }
        self.keys2.clear();
        self.keys2.resize(n, 0);
        self.vals2.clear();
        self.vals2.resize(n, 0);
        for (b, h) in hist.iter().enumerate() {
            if h.iter().any(|&c| c as usize == n) {
                continue;
            }
            let mut offs = [0u32; 256];
            let mut sum = 0u32;
            for (o, &c) in offs.iter_mut().zip(h.iter()) {
                *o = sum;
                sum += c;
            }
            let shift = b * 8;
            for idx in 0..n {
                let k = keys[idx];
                let d = ((k >> shift) & 0xFF) as usize;
                let at = offs[d] as usize;
                offs[d] += 1;
                self.keys2[at] = k;
                self.vals2[at] = vals[idx];
            }
            std::mem::swap(keys, &mut self.keys2);
            std::mem::swap(vals, &mut self.vals2);
        }
    }
}

// ---------------------------------------------------------------------------
// Input records
// ---------------------------------------------------------------------------

/// One visible sprite renderer, as the engine's gather hands it over.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpriteSource {
    /// Object origin = the sprite's pivot, world space.
    pub position: [f32; 3],
    /// Rotation about Z, radians.
    pub rotation: f32,
    pub scale: [f32; 2],
    /// World point measured for the depth tie-break (centre or pivot).
    pub sort_point: [f32; 3],
    /// The sprite's pooled mesh (Simple mode).
    pub mesh: MeshRange,
    /// `u0, v0, u1, v1` of the sprite's rect in its texture or atlas page.
    pub uv_rect: [f32; 4],
    /// Sprite rect size, world units.
    pub sprite_size: [f32; 2],
    /// 9-slice border, world units: left, bottom, right, top.
    pub border: [f32; 4],
    /// Pivot from the rect's bottom-left, world units.
    pub pivot: [f32; 2],
    /// Linear RGBA.
    pub color: [f32; 4],
    pub draw: DrawMode,
    /// Drawn size (Sliced / Tiled), world units.
    pub size: [f32; 2],
    pub tile: TileMode,
    pub adaptive_threshold: f32,
    pub flip: [bool; 2],
    /// Sprite material id (blend mode, cutoff): the GPU side resolves it.
    pub material: u32,
    /// Texture / atlas id: the batch key; resolved to a descriptor by the GPU side.
    pub texture: u32,
    /// Array layer inside `texture` (per-instance; never breaks a batch).
    pub atlas_layer: u32,
    /// Sorting-layer value (from the project's list) and order in layer.
    pub layer: i16,
    pub order: i16,
    /// Group ordinal from [`flatten_groups`] (0 = none); `layer` / `order` must
    /// already be the root group's for grouped sprites.
    pub group: u16,
    pub mask: MaskMode,
    /// Quad mode (tiles): local rect `min x, min y, size x, size y`.
    pub rect: [f32; 4],
    /// The placed content is turned 90 degrees in its atlas page (Quad mode).
    pub rotated: bool,
    /// Animation header index in the frame table; 0 = not animated.
    pub anim: u32,
    /// Seconds added to the frame time of an animated source.
    pub anim_phase: f32,
}

/// Which sorting range a mask affects.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MaskRange {
    /// From the mask's own position up to (excluding) the next mask's.
    #[default]
    Auto,
    /// Inclusive `(layer value, order)` bounds, back and front.
    Custom { back: (i16, i16), front: (i16, i16) },
}

/// One sprite mask. Masks are Simple-mode sprites drawn into the stencil buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaskSource {
    pub position: [f32; 3],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub mesh: MeshRange,
    pub uv_rect: [f32; 4],
    pub material: u32,
    pub texture: u32,
    pub atlas_layer: u32,
    pub alpha_cutoff: f32,
    pub layer: i16,
    pub order: i16,
    pub range: MaskRange,
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Draws that share material, texture and stencil state: one pipeline and
/// descriptor bind, then `draw_count` indirect commands from `first_draw`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpriteBatch {
    pub material: u32,
    pub texture: u32,
    pub stencil: MaskMode,
    pub first_draw: u32,
    pub draw_count: u32,
    pub instance_count: u32,
}

/// One step of the pass, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassItem {
    Batch(SpriteBatch),
    /// Draw a mask shape into the stencil buffer (`draws[draw]`, colour off).
    MaskWrite { material: u32, texture: u32, draw: u32 },
    /// Replay a written mask shape (`draws[draw]`) with the ZERO op, resetting
    /// its texels; emitted for every written mask when the active set changes.
    MaskErase { material: u32, texture: u32, draw: u32 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpriteStats {
    pub sprites: u32,
    pub instances: u32,
    pub batches: u32,
    pub draws: u32,
    pub mask_writes: u32,
    pub mask_erases: u32,
    /// Tiled renderers drawn stretched because they exceeded the tile cap.
    pub tiled_fallbacks: u32,
    /// Sprites dropped by the instance cap.
    pub dropped: u32,
    /// Grouped renderers beyond the ordinal range (see [`flatten_groups`]).
    pub group_overflow: u32,
}

/// Everything the GPU half needs for one frame. Owned by the builder, reused.
#[derive(Default)]
pub struct SpriteFrame {
    /// Instances in draw order (upload as-is).
    pub instances: Vec<SpriteInstance>,
    pub draws: Vec<DrawIndexedIndirect>,
    pub items: Vec<PassItem>,
    /// Items `[..behind_items]` belong to layers at or below
    /// [`BuildConfig::behind_layer_max`] (drawn before 3D transparents).
    pub behind_items: usize,
    pub stats: SpriteStats,
}

impl SpriteFrame {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn clear(&mut self) {
        self.instances.clear();
        self.draws.clear();
        self.items.clear();
        self.behind_items = 0;
        self.stats = SpriteStats::default();
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct BuildConfig {
    pub axis: SortAxis,
    /// Depth range quantised into the key's 14 bits.
    pub depth_near: f32,
    pub depth_far: f32,
    /// Hard cap on instances (the GPU buffer size); sources past it are dropped.
    pub max_instances: usize,
    /// Worker threads for the gather (1 = serial). The output never depends on it.
    pub threads: usize,
    /// Sprites on sorting layers `<=` this value draw before the 3D transparent
    /// queue (a batch break is forced at the boundary); `None` = all after it.
    pub behind_layer_max: Option<i16>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            axis: SortAxis::default(),
            depth_near: -1000.0,
            depth_far: 1000.0,
            max_instances: 1 << 20,
            threads: 3,
            behind_layer_max: None,
        }
    }
}

/// Batching data kept beside each instance (not uploaded).
#[derive(Clone, Copy, Default)]
struct InstanceMeta {
    material: u32,
    texture: u32,
    mesh: MeshRange,
}

/// Minimum instances per worker before threads are worth spawning.
const PAR_MIN_PER_THREAD: usize = 4096;

fn pack_color(c: [f32; 4]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    q(c[0]) | q(c[1]) << 8 | q(c[2]) << 16 | q(c[3]) << 24
}

fn base_instance(position: [f32; 3], rotation: f32, scale: [f32; 2]) -> SpriteInstance {
    let (s, c) = rotation.sin_cos();
    SpriteInstance {
        affine: [c * scale[0], s * scale[0], -s * scale[1], c * scale[1]],
        pos: [position[0], position[1]],
        z: position[2],
        ..Default::default()
    }
}

fn tile_params(s: &SpriteSource) -> TileParams {
    TileParams {
        sprite_size: s.sprite_size,
        border: s.border,
        pivot: s.pivot,
        size: s.size,
        mode: s.tile,
        threshold: s.adaptive_threshold,
        uv_rect: s.uv_rect,
    }
}

/// How many instances a source expands into, and its effective draw mode (a
/// Tiled source over the tile cap draws as Sliced).
fn plan_source(s: &SpriteSource) -> (usize, DrawMode) {
    match s.draw {
        DrawMode::Tiled => match tile_count(&tile_params(s)) {
            Some(n) if n > 0 => (n, DrawMode::Tiled),
            Some(_) => (0, DrawMode::Tiled),
            None => (1, DrawMode::Sliced),
        },
        d => (1, d),
    }
}

/// The UV of the unit-quad corner `corner` (`x, y` in 0..1, y up) of a quad
/// drawn from `uv` = `u0, v0, u1, v1` (`v0` is the image top). A `rotated`
/// sprite is stored turned 90 degrees clockwise in its page, so the quad's x
/// axis runs down the page and its y axis runs along it (right edge to left).
/// `sprite.vert` evaluates the same expressions.
pub fn quad_uv(uv: [f32; 4], corner: [f32; 2], rotated: bool) -> [f32; 2] {
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    if rotated {
        [mix(uv[0], uv[2], corner[1]), mix(uv[1], uv[3], corner[0])]
    } else {
        [mix(uv[0], uv[2], corner[0]), mix(uv[3], uv[1], corner[1])]
    }
}

// ---------------------------------------------------------------------------
// Tile animation (frame table)
// ---------------------------------------------------------------------------
//
// The engine appends every animation it needs to one `Vec<[f32; 4]>` per frame
// (`SpriteFrameInput::anim`); entry 0 is a dummy so 0 can mean "none". An
// animation is a header followed by its frames:
//
//   [index]        header: count (u32 bits), fps (f32), flags (u32 bits), rotated mask (u32 bits)
//   [index+1+2k]   frame k UV rect: `u0, v0, u1, v1`
//   [index+2+2k]   frame k local rect: `min x, min y, size x, size y`
//
// Each frame carries its own local rect so frames of different trim (atlas
// packing) stay registered on the pivot. Up to [`ANIM_MAX_FRAMES`] frames (one
// bit each in the rotated mask). A frame
// time `t` (seconds since the animation's start, the global time plus the
// instance's phase) selects frame [`anim_frame`]. All frames of one animation
// share the instance's texture and array layer (the gather checks that).

/// Frames one animation can have (bits of the rotated mask).
pub const ANIM_MAX_FRAMES: usize = 32;
/// Header flag: play once and hold the last frame.
pub const ANIM_LOOP_ONCE: u32 = 1;

/// The decoded frame-table header.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AnimHeader {
    pub count: u32,
    pub fps: f32,
    pub flags: u32,
    pub rotated: u32,
}

impl AnimHeader {
    pub fn encode(&self) -> [f32; 4] {
        [f32::from_bits(self.count), self.fps, f32::from_bits(self.flags), f32::from_bits(self.rotated)]
    }
    pub fn decode(v: [f32; 4]) -> Self {
        Self { count: v[0].to_bits(), fps: v[1], flags: v[2].to_bits(), rotated: v[3].to_bits() }
    }
}

/// Append one animation to a frame table (creating the dummy entry 0 first).
/// `frames` = `(uv rect, local rect, rotated)`, at most [`ANIM_MAX_FRAMES`] are
/// kept. Returns the header index to store in `SpriteSource::anim`.
pub fn push_animation(table: &mut Vec<[f32; 4]>, frames: &[([f32; 4], [f32; 4], bool)], fps: f32, loop_once: bool) -> u32 {
    if table.is_empty() {
        table.push([0.0; 4]);
    }
    let n = frames.len().min(ANIM_MAX_FRAMES);
    let mut rotated = 0u32;
    for (k, f) in frames.iter().take(n).enumerate() {
        rotated |= (f.2 as u32) << k;
    }
    let at = table.len() as u32;
    table.push(AnimHeader { count: n as u32, fps, flags: if loop_once { ANIM_LOOP_ONCE } else { 0 }, rotated }.encode());
    for f in frames.iter().take(n) {
        table.push(f.0);
        table.push(f.1);
    }
    at
}

/// The frame shown `t` seconds into an animation of `count` frames at `fps`:
/// looping, or holding the last frame with `ANIM_LOOP_ONCE`. Negative times
/// and non-positive rates show frame 0. `sprite.vert` evaluates the same.
pub fn anim_frame(count: u32, fps: f32, flags: u32, t: f32) -> u32 {
    if count <= 1
        || fps.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        || t.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        return 0;
    }
    let n = (t * fps).floor();
    if flags & ANIM_LOOP_ONCE != 0 {
        n.min((count - 1) as f32) as u32
    } else {
        (n % count as f32) as u32
    }
}

/// What the vertex shader reads for an animated instance: the frame's UV rect,
/// its local rect and whether that frame is stored rotated. Mirrors `sprite.vert`.
pub fn anim_lookup(table: &[[f32; 4]], index: u32, t: f32) -> Option<([f32; 4], [f32; 4], bool)> {
    let h = AnimHeader::decode(*table.get(index as usize)?);
    let k = anim_frame(h.count, h.fps, h.flags, t) as usize;
    let uv = *table.get(index as usize + 1 + 2 * k)?;
    let rect = *table.get(index as usize + 2 + 2 * k)?;
    Some((uv, rect, (h.rotated >> k) & 1 != 0))
}

fn key_for(s: &SpriteSource, cfg: &BuildConfig) -> u64 {
    let depth = quantize_depth(cfg.axis.distance(s.sort_point), cfg.depth_near, cfg.depth_far);
    SortKey::pack(s.layer, s.order, s.group, batch_bits(s.material, s.texture, s.mask), depth).0
}

/// Write the instances of one source into `out` (`out.len()` = its plan count).
fn write_source(s: &SpriteSource, mode: DrawMode, out: &mut [SpriteInstance]) {
    let mut base = base_instance(s.position, s.rotation, s.scale);
    base.color = pack_color(s.color);
    base.layer = s.atlas_layer;
    base.uv_rect = s.uv_rect;
    let mut flags = (s.flip[0] as u32 * FLAG_FLIP_X) | (s.flip[1] as u32 * FLAG_FLIP_Y) | (s.mask as u32) << FLAG_MASK_SHIFT;
    if s.material & SPRITE_MATERIAL_XZ != 0 {
        flags |= FLAG_XZ;
    }
    match mode {
        DrawMode::Simple => {
            base.flags = flags | MODE_MESH << FLAG_MODE_SHIFT;
            out[0] = base;
        }
        DrawMode::Sliced => {
            let (w, h) = (s.sprite_size[0].max(SPAN_EPS), s.sprite_size[1].max(SPAN_EPS));
            let du = (s.uv_rect[2] - s.uv_rect[0]) / w;
            let dv = (s.uv_rect[3] - s.uv_rect[1]) / h;
            base.flags = flags | MODE_SLICE << FLAG_MODE_SHIFT;
            base.rect = [-s.pivot[0] / w * s.size[0], -s.pivot[1] / h * s.size[1], s.size[0], s.size[1]];
            base.border = s.border;
            base.uv_border = [s.border[0] * du, s.border[1] * dv, s.border[2] * du, s.border[3] * dv];
            out[0] = base;
        }
        DrawMode::Quad => {
            flags |= MODE_QUAD << FLAG_MODE_SHIFT;
            if s.rotated {
                flags |= FLAG_ROTATED;
            }
            if s.anim != 0 {
                flags |= FLAG_ANIM;
                base.anim = s.anim;
                base.alpha_cutoff = s.anim_phase;
            }
            base.flags = flags;
            base.rect = s.rect;
            out[0] = base;
        }
        DrawMode::Tiled => {
            flags |= MODE_QUAD << FLAG_MODE_SHIFT;
            base.flags = flags;
            let mut k = 0;
            for_each_tile(&tile_params(s), &mut |t| {
                let mut inst = base;
                inst.rect = [t.min[0], t.min[1], t.size[0], t.size[1]];
                inst.uv_rect = t.uv;
                out[k] = inst;
                k += 1;
            });
            debug_assert_eq!(k, out.len());
        }
    }
}

/// Reusable frame builder. Keep one alive: every buffer inside is pooled, so a
/// steady-state frame allocates nothing.
pub struct SpritePassBuilder {
    instances: Vec<SpriteInstance>,
    metas: Vec<InstanceMeta>,
    keys: Vec<u64>,
    vals: Vec<u32>,
    offsets: Vec<u32>,
    modes: Vec<DrawMode>,
    sorter: RadixSorter,
    mask_extent: Vec<(u32, u32)>,
    mask_order: Vec<u32>,
    written: Vec<u32>,
    written_draws: Vec<(u32, u32, u32)>,
    needed: Vec<u32>,
    frame: SpriteFrame,
}

impl Default for SpritePassBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpritePassBuilder {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            metas: Vec::new(),
            keys: Vec::new(),
            vals: Vec::new(),
            offsets: Vec::new(),
            modes: Vec::new(),
            sorter: RadixSorter::default(),
            mask_extent: Vec::new(),
            mask_order: Vec::new(),
            written: Vec::new(),
            written_draws: Vec::new(),
            needed: Vec::new(),
            frame: SpriteFrame::default(),
        }
    }

    /// The last built frame.
    pub fn frame(&self) -> &SpriteFrame {
        &self.frame
    }

    /// Gather, sort and batch. Deterministic: the same inputs give the same
    /// frame for any `cfg.threads`.
    pub fn build(&mut self, sprites: &[SpriteSource], masks: &[MaskSource], cfg: &BuildConfig) -> &SpriteFrame {
        self.frame.clear();
        let max = cfg.max_instances.max(1);

        // 1. Plan: instance counts and the prefix sum, serial and cheap.
        self.offsets.clear();
        self.modes.clear();
        let mut total = 0usize;
        let mut taken = 0usize;
        let mut fallbacks = 0u32;
        for s in sprites {
            let (n, mode) = plan_source(s);
            if total + n > max {
                break;
            }
            if mode != s.draw {
                fallbacks += 1;
            }
            self.offsets.push(total as u32);
            self.modes.push(mode);
            total += n;
            taken += 1;
        }
        self.offsets.push(total as u32);
        let sprites_in = &sprites[..taken];
        self.frame.stats.dropped = (sprites.len() - taken) as u32;
        self.frame.stats.tiled_fallbacks = fallbacks;
        self.frame.stats.sprites = taken as u32;

        // 2. Gather in parallel into the pooled arrays.
        self.instances.clear();
        self.instances.resize(total, SpriteInstance::default());
        self.metas.clear();
        self.metas.resize(total, InstanceMeta::default());
        self.keys.clear();
        self.keys.resize(total, 0);
        let threads = cfg.threads.max(1).min(total / PAR_MIN_PER_THREAD).max(1);
        if threads <= 1 {
            gather_range(sprites_in, &self.modes, &self.offsets, 0, &mut self.instances, &mut self.metas, &mut self.keys, cfg);
        } else {
            // Split the sources into `threads` contiguous runs of roughly equal
            // instance count; each run owns a disjoint slice of every output.
            let per = total.div_ceil(threads);
            let mut bounds = vec![0usize];
            let mut next = per;
            for (i, &o) in self.offsets.iter().enumerate().take(taken) {
                if o as usize >= next && i > *bounds.last().unwrap() {
                    bounds.push(i);
                    next = o as usize + per;
                }
            }
            bounds.push(taken);
            let (mut inst_rest, mut meta_rest, mut key_rest) = (&mut self.instances[..], &mut self.metas[..], &mut self.keys[..]);
            let (modes, offsets) = (&self.modes[..], &self.offsets[..]);
            std::thread::scope(|scope| {
                for w in bounds.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    let (base, end) = (offsets[a] as usize, offsets[b] as usize);
                    let (inst, ir) = std::mem::take(&mut inst_rest).split_at_mut(end - base);
                    let (meta, mr) = std::mem::take(&mut meta_rest).split_at_mut(end - base);
                    let (key, kr) = std::mem::take(&mut key_rest).split_at_mut(end - base);
                    (inst_rest, meta_rest, key_rest) = (ir, mr, kr);
                    let src = &sprites_in[a..b];
                    scope.spawn(move || gather_range(src, &modes[a..b], &offsets[a..=b], base as u32, inst, meta, key, cfg));
                }
            });
        }

        // 3. Sort. The payload is the instance index.
        self.vals.clear();
        self.vals.extend(0..total as u32);
        self.sorter.sort(&mut self.keys, &mut self.vals);

        // 4. Masks, then one walk: plan stencil, batch, merge draws.
        self.plan_masks(masks);
        self.walk(masks, cfg);
        self.frame.stats.instances = self.frame.instances.len() as u32;
        self.frame.stats.draws = self.frame.draws.len() as u32;
        self.frame.stats.batches = self.frame.items.iter().filter(|i| matches!(i, PassItem::Batch(_))).count() as u32;
        &self.frame
    }

    /// Resolve every mask's inclusive `layer_order` extent.
    fn plan_masks(&mut self, masks: &[MaskSource]) {
        self.mask_extent.clear();
        // Auto ends before the next strictly greater mask position.
        self.mask_order.clear();
        self.mask_order.extend(masks.iter().map(|m| layer_order(m.layer, m.order)));
        self.mask_order.sort_unstable();
        for m in masks {
            let own = layer_order(m.layer, m.order);
            let ext = match m.range {
                MaskRange::Auto => {
                    let next = self.mask_order.iter().copied().find(|&o| o > own);
                    (own, next.map_or(u32::MAX, |n| n - 1))
                }
                MaskRange::Custom { back, front } => {
                    let (a, b) = (layer_order(back.0, back.1), layer_order(front.0, front.1));
                    (a.min(b), a.max(b))
                }
            };
            self.mask_extent.push(ext);
        }
    }

    fn walk(&mut self, masks: &[MaskSource], cfg: &BuildConfig) {
        let f = &mut self.frame;
        f.instances.reserve(self.vals.len() + masks.len());
        self.written.clear();
        self.written_draws.clear();
        let mut cur: Option<SpriteBatch> = None;
        let mut last_lo = u32::MAX;
        let mut crossed = cfg.behind_layer_max.is_none();
        let mut state_known = false;
        for (k, &v) in self.vals.iter().enumerate() {
            let key = SortKey(self.keys[k]);
            let inst = self.instances[v as usize];
            let meta = self.metas[v as usize];
            let mode = inst.mask_mode();

            // The "behind 3D transparents" boundary: one forced break.
            if !crossed && key.layer() > cfg.behind_layer_max.unwrap_or(i16::MAX) {
                crossed = true;
                if let Some(b) = cur.take() {
                    f.items.push(PassItem::Batch(b));
                }
                f.behind_items = f.items.len();
            }

            // Stencil contents must equal the union of the masks covering this
            // sprite. Recomputed only when the (layer, order) changes.
            if mode != MaskMode::None {
                let lo = key.layer_order();
                if !state_known || lo != last_lo {
                    last_lo = lo;
                    state_known = true;
                    self.needed.clear();
                    for (mi, &(a, b)) in self.mask_extent.iter().enumerate() {
                        if lo >= a && lo <= b {
                            self.needed.push(mi as u32);
                        }
                    }
                }
                if self.needed != self.written {
                    if let Some(b) = cur.take() {
                        f.items.push(PassItem::Batch(b));
                    }
                    for &(material, texture, draw) in &self.written_draws {
                        f.items.push(PassItem::MaskErase { material, texture, draw });
                        f.stats.mask_erases += 1;
                    }
                    self.written_draws.clear();
                    for &mi in &self.needed {
                        let m = &masks[mi as usize];
                        let mut mi_inst = base_instance(m.position, m.rotation, m.scale);
                        mi_inst.uv_rect = m.uv_rect;
                        mi_inst.layer = m.atlas_layer;
                        mi_inst.color = u32::MAX;
                        mi_inst.alpha_cutoff = m.alpha_cutoff.clamp(0.0, 1.0);
                        let draw = f.draws.len() as u32;
                        f.draws.push(DrawIndexedIndirect {
                            index_count: m.mesh.index_count,
                            instance_count: 1,
                            first_index: m.mesh.first_index,
                            vertex_offset: m.mesh.vertex_offset,
                            first_instance: f.instances.len() as u32,
                        });
                        f.instances.push(mi_inst);
                        f.items.push(PassItem::MaskWrite { material: m.material, texture: m.texture, draw });
                        self.written_draws.push((m.material, m.texture, draw));
                        f.stats.mask_writes += 1;
                    }
                    self.written.clear();
                    self.written.extend_from_slice(&self.needed);
                }
            }

            // Batch: breaks only on material, texture, stencil mode.
            let same = cur.as_ref().is_some_and(|b| b.material == meta.material && b.texture == meta.texture && b.stencil == mode);
            if !same {
                if let Some(b) = cur.take() {
                    f.items.push(PassItem::Batch(b));
                }
                cur = Some(SpriteBatch {
                    material: meta.material,
                    texture: meta.texture,
                    stencil: mode,
                    first_draw: f.draws.len() as u32,
                    draw_count: 0,
                    instance_count: 0,
                });
            }
            let b = cur.as_mut().unwrap();
            let at = f.instances.len() as u32;
            f.instances.push(inst);
            // Merge into the previous command when it is the same mesh and
            // directly precedes this instance.
            let merge = b.draw_count > 0
                && f.draws.last().is_some_and(|d| {
                    d.first_index == meta.mesh.first_index
                        && d.index_count == meta.mesh.index_count
                        && d.vertex_offset == meta.mesh.vertex_offset
                        && d.first_instance + d.instance_count == at
                });
            if merge {
                f.draws.last_mut().unwrap().instance_count += 1;
            } else {
                f.draws.push(DrawIndexedIndirect {
                    index_count: meta.mesh.index_count,
                    instance_count: 1,
                    first_index: meta.mesh.first_index,
                    vertex_offset: meta.mesh.vertex_offset,
                    first_instance: at,
                });
                b.draw_count += 1;
            }
            b.instance_count += 1;
        }
        if let Some(b) = cur.take() {
            f.items.push(PassItem::Batch(b));
        }
        if !crossed {
            // No sprite lay past the boundary: every item is "behind".
            f.behind_items = f.items.len();
        }
    }
}

/// Fill the slices for sources `src` (whose prefix offsets are `offsets`, global
/// indices; `base` is the first one this call owns).
#[allow(clippy::too_many_arguments)]
fn gather_range(
    src: &[SpriteSource],
    modes: &[DrawMode],
    offsets: &[u32],
    base: u32,
    inst: &mut [SpriteInstance],
    meta: &mut [InstanceMeta],
    key: &mut [u64],
    cfg: &BuildConfig,
) {
    for (i, s) in src.iter().enumerate() {
        let (a, b) = ((offsets[i] - base) as usize, (offsets[i + 1] - base) as usize);
        write_source(s, modes[i], &mut inst[a..b]);
        let mesh = match modes[i] {
            DrawMode::Simple => s.mesh,
            DrawMode::Sliced => SLICE_TEMPLATE,
            DrawMode::Tiled | DrawMode::Quad => QUAD_TEMPLATE,
        };
        let k = key_for(s, cfg);
        for j in a..b {
            meta[j] = InstanceMeta { material: s.material, texture: s.texture, mesh };
            key[j] = k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(layer: i16, order: i16) -> SpriteSource {
        SpriteSource {
            scale: [1.0, 1.0],
            sprite_size: [1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            mesh: MeshRange { first_index: 60, index_count: 6, vertex_offset: 20 },
            layer,
            order,
            ..Default::default()
        }
    }

    fn cfg() -> BuildConfig {
        BuildConfig { threads: 1, ..Default::default() }
    }

    // --- key ---------------------------------------------------------------

    #[test]
    fn key_packs_and_unpacks() {
        let k = SortKey::pack(-3, -7, 513, 0x2AB, 0x1555);
        assert_eq!((k.layer(), k.order(), k.group(), k.batch(), k.depth()), (-3, -7, 513, 0x2AB, 0x1555));
        // Out-of-range fields are clamped / masked, never bleed into neighbours.
        let k = SortKey::pack(1000, 5, 0, 0xFFFF, 0xFFFF);
        assert_eq!((k.layer(), k.order(), k.group()), (KEY_LAYER_MAX, 5, 0));
        assert_eq!(SortKey::pack(-1000, 0, 0, 0, 0).layer(), KEY_LAYER_MIN);
    }

    #[test]
    fn key_orders_layer_then_order_then_group_then_batch_then_depth() {
        let k = SortKey::pack;
        assert!(k(0, 100, 0, 0, 0) < k(1, -100, 0, 0, 0)); // layer beats order
        assert!(k(0, -1, 9, 9, 9) < k(0, 0, 0, 0, 0)); // order beats group
        assert!(k(0, 0, 1, 1023, 9) < k(0, 0, 2, 0, 0)); // group beats batch
        assert!(k(0, 0, 0, 1, 16383) < k(0, 0, 0, 2, 0)); // batch beats depth
        assert!(k(-128, -32768, 0, 0, 0) < k(127, 32767, 65535, 1023, 16383));
        assert!(k(-5, 0, 0, 0, 0) < k(-4, 0, 0, 0, 0)); // negative layers sort first
    }

    #[test]
    fn depth_is_back_to_front_and_monotone() {
        let (n, f) = (0.0, 100.0);
        assert!(quantize_depth(90.0, n, f) < quantize_depth(10.0, n, f)); // farther = smaller
        assert_eq!(quantize_depth(-5.0, n, f), KEY_DEPTH_MASK);
        assert_eq!(quantize_depth(500.0, n, f), 0);
        assert_eq!(quantize_depth(f32::NAN, n, f), 0);
        let mut prev = u16::MAX;
        for i in 0..=100 {
            let q = quantize_depth(i as f32, n, f);
            assert!(q <= prev);
            prev = q;
        }
    }

    #[test]
    fn sort_axis_custom_top_down() {
        // Lower on screen is in front: farther = higher y, so axis +y.
        let a = SortAxis { origin: [0.0; 3], axis: [0.0, 1.0, 0.0] };
        assert!(a.distance([0.0, 5.0, 0.0]) > a.distance([0.0, -5.0, 0.0]));
    }

    // --- radix -------------------------------------------------------------

    #[test]
    fn radix_matches_std_sort_and_is_stable() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for n in [0usize, 1, 2, 17, 1000, 5000] {
            // Few distinct keys force many ties; some keys use the high bytes.
            let mut keys: Vec<u64> = (0..n).map(|_| if next() % 3 == 0 { next() } else { next() % 7 }).collect();
            let mut vals: Vec<u32> = (0..n as u32).collect();
            let mut expect: Vec<(u64, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
            expect.sort_by_key(|&(k, _)| k); // std stable sort
            RadixSorter::default().sort(&mut keys, &mut vals);
            let got: Vec<(u64, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
            assert_eq!(got, expect, "n = {n}");
        }
    }

    // --- 9-slice -----------------------------------------------------------

    #[test]
    fn nine_slice_stops_normal_and_degenerate() {
        assert_eq!(nine_slice_stops(10.0, 2.0, 3.0), [0.0, 2.0, 7.0, 10.0]);
        assert_eq!(nine_slice_stops(10.0, 0.0, 0.0), [0.0, 0.0, 10.0, 10.0]);
        // Target smaller than the border sum: both shrink by the same factor
        // and meet; corner widths keep their ratio.
        let s = nine_slice_stops(2.0, 1.0, 3.0);
        assert_eq!(s, [0.0, 0.5, 0.5, 2.0]);
        assert_eq!(s[1], s[2], "centre collapses");
        assert!((s[1] / (s[3] - s[2]) - 1.0 / 3.0).abs() < 1e-6);
        // Exactly the border sum: no scaling, zero-width centre.
        assert_eq!(nine_slice_stops(4.0, 1.0, 3.0), [0.0, 1.0, 1.0, 4.0]);
        // Zero / negative size never produces negative widths.
        assert_eq!(nine_slice_stops(0.0, 1.0, 1.0), [0.0; 4]);
        assert_eq!(nine_slice_stops(-3.0, 1.0, 1.0), [0.0; 4]);
    }

    #[test]
    fn slice_vertex_corners_uvs_and_flip() {
        let rect = [-1.0, -1.0, 4.0, 4.0]; // pivot at (1,1) of a 4x4 target
        let border = [1.0, 1.0, 1.0, 1.0];
        let uv_rect = [0.25, 0.5, 0.75, 1.0];
        let uv_border = [0.125, 0.125, 0.125, 0.125];
        let v = |i, j, fl| slice_vertex(i, j, rect, border, uv_rect, uv_border, fl);
        // Bottom-left corner vertex = rect min, UV = (u0, v1).
        assert_eq!(v(0, 0, [false; 2]), ([-1.0, -1.0], [0.25, 1.0]));
        // Top-right = rect max, UV = (u1, v0).
        assert_eq!(v(3, 3, [false; 2]), ([3.0, 3.0], [0.75, 0.5]));
        // Inner lines sit one border in, with UVs one border distance in.
        assert_eq!(v(1, 1, [false; 2]), ([0.0, 0.0], [0.375, 0.875]));
        assert_eq!(v(2, 2, [false; 2]), ([2.0, 2.0], [0.625, 0.625]));
        // Flip mirrors the position about the pivot (local origin), UV unchanged.
        let (p, uv) = v(3, 0, [true, false]);
        assert_eq!((p, uv), ([-3.0, -1.0], [0.75, 1.0]));
        let (p, _) = v(3, 3, [true, true]);
        assert_eq!(p, [-3.0, -3.0]);
    }

    #[test]
    fn slice_template_covers_the_grid() {
        let pool = SpriteMeshPool::new(100, 100);
        let idx = &pool.indices()[SLICE_TEMPLATE.first_index as usize..][..SLICE_TEMPLATE.index_count as usize];
        assert_eq!(idx.len(), 54);
        assert!(idx.iter().all(|&i| i < 16));
        // Every one of the nine cells is covered by exactly two triangles.
        let verts = &pool.vertices()[SLICE_TEMPLATE.vertex_offset as usize..][..16];
        let mut area = 0.0;
        for t in idx.chunks(3) {
            let (a, b, c) = (verts[t[0] as usize].pos, verts[t[1] as usize].pos, verts[t[2] as usize].pos);
            area += 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
        }
        assert!((area - 9.0).abs() < 1e-6, "area {area}");
        let q = &pool.indices()[..6];
        assert_eq!(q, &[0, 1, 2, 2, 1, 3]);
    }

    // --- tiling ------------------------------------------------------------

    fn tile_params(size: [f32; 2], mode: TileMode) -> TileParams {
        TileParams {
            sprite_size: [4.0, 4.0],
            border: [1.0, 1.0, 1.0, 1.0],
            pivot: [2.0, 2.0],
            size,
            mode,
            threshold: 0.5,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
        }
    }

    fn collect(p: &TileParams) -> Vec<TileQuad> {
        let mut v = Vec::new();
        for_each_tile(p, &mut |t| v.push(t));
        v
    }

    #[test]
    fn tiled_counts_match_the_plan() {
        // Inner tile = 2x2. Target 10x6: middle span 8 x 4 -> 4 x 2 tiles, edges 4+4 and 2+2, 4 corners.
        let p = tile_params([10.0, 6.0], TileMode::Continuous);
        assert_eq!(tile_count(&p), Some(4 + 2 * 4 + 2 * 2 + 4 * 2));
        assert_eq!(collect(&p).len(), tile_count(&p).unwrap());
        // Non-multiple: ceil, the last tile clipped. Middle 7 wide -> 4 tiles.
        let p = tile_params([9.0, 4.0], TileMode::Continuous);
        let n = tile_count(&p).unwrap();
        assert_eq!(collect(&p).len(), n);
        // Target equal to the sprite: 9 cells, one each.
        assert_eq!(tile_count(&tile_params([4.0, 4.0], TileMode::Continuous)), Some(9));
        // No borders: pure tiling with the whole 4 x 4 sprite as the tile: 6 x 4
        // is a whole tile plus a clipped half, 2 x 1.
        let mut p = tile_params([6.0, 4.0], TileMode::Continuous);
        p.border = [0.0; 4];
        assert_eq!(tile_count(&p), Some(2));
        assert_eq!(collect(&p).len(), 2);
        // Cap: absurd sizes fall back (None).
        let p = tile_params([1e6, 1e6], TileMode::Continuous);
        assert_eq!(tile_count(&p), None);
    }

    #[test]
    fn tiles_cover_the_rect_exactly_and_clip_uvs() {
        let p = tile_params([9.0, 4.0], TileMode::Continuous);
        let tiles = collect(&p);
        // Origin: pivot (2,2) of a 4x4 sprite -> (-0.5, -0.5) normalised -> min (-4.5, -2.0).
        let area: f32 = tiles.iter().map(|t| t.size[0] * t.size[1]).sum();
        assert!((area - 36.0).abs() < 1e-4, "area {area}");
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for t in &tiles {
            for k in 0..2 {
                lo[k] = lo[k].min(t.min[k]);
                hi[k] = hi[k].max(t.min[k] + t.size[k]);
            }
            // UVs stay inside the sprite and never wrap.
            assert!(t.uv[0] >= 0.0 && t.uv[2] <= 1.0 && t.uv[3] <= 1.0 && t.uv[1] >= 0.0);
            assert!(t.uv[0] <= t.uv[2] && t.uv[1] <= t.uv[3]);
        }
        assert_eq!(lo, [-4.5, -2.0]);
        assert_eq!(hi, [4.5, 2.0]);
        // The clipped last middle tile (width 1 of a 2-wide tile) shows half of
        // the centre's u range: 0.25 .. 0.75 -> 0.25 .. 0.5.
        let clipped = tiles.iter().find(|t| (t.size[0] - 1.0).abs() < 1e-5 && t.size[1] > 1.5 && t.uv[0] > 0.2 && t.uv[0] < 0.3).unwrap();
        assert!((clipped.uv[2] - 0.5).abs() < 1e-5, "{:?}", clipped);
    }

    #[test]
    fn adaptive_stretches_until_the_threshold() {
        // Inner tile 2: span 2.5 < 2 * 1.5 stretches (1 tile); span 4 tiles.
        let mut p = tile_params([4.5, 4.0], TileMode::Adaptive); // middle span 2.5
        p.border = [1.0, 1.0, 1.0, 1.0];
        let stretched = tile_count(&p).unwrap();
        let mut q = p;
        q.size = [8.0, 4.0]; // middle span 6 -> 3 tiles
        let tiled = tile_count(&q).unwrap();
        assert!(tiled > stretched);
        let tiles = collect(&p);
        assert!(tiles.iter().any(|t| (t.size[0] - 2.5).abs() < 1e-5), "stretched middle");
    }

    #[test]
    fn tiled_over_cap_falls_back_to_sliced() {
        let mut s = src(0, 0);
        s.draw = DrawMode::Tiled;
        s.size = [1e6, 1e6];
        s.border = [0.1; 4];
        s.sprite_size = [1.0, 1.0];
        let mut b = SpritePassBuilder::new();
        let f = b.build(&[s], &[], &cfg());
        assert_eq!(f.stats.tiled_fallbacks, 1);
        assert_eq!(f.instances.len(), 1);
        assert_eq!(f.instances[0].mode(), MODE_SLICE);
    }

    #[test]
    fn tiled_instances_merge_into_one_draw() {
        let mut s = src(0, 0);
        s.draw = DrawMode::Tiled;
        s.size = [10.0, 6.0];
        s.sprite_size = [4.0, 4.0];
        s.border = [1.0; 4];
        s.pivot = [2.0, 2.0];
        let mut b = SpritePassBuilder::new();
        let f = b.build(&[s], &[], &cfg());
        assert_eq!(f.instances.len(), 4 + 8 + 4 + 8);
        assert_eq!(f.draws.len(), 1);
        assert_eq!(f.draws[0].instance_count as usize, f.instances.len());
        assert_eq!(f.draws[0].first_index, QUAD_TEMPLATE.first_index);
        assert!(f.instances.iter().all(|i| i.mode() == MODE_QUAD));
    }

    // --- batching ----------------------------------------------------------

    fn batches(f: &SpriteFrame) -> Vec<SpriteBatch> {
        f.items.iter().filter_map(|i| if let PassItem::Batch(b) = i { Some(*b) } else { None }).collect()
    }

    #[test]
    fn batches_break_only_on_material_texture_stencil() {
        let mut v = Vec::new();
        for i in 0..6 {
            let mut s = src(0, i);
            s.material = 1;
            s.texture = 1;
            v.push(s);
        }
        let mut b = SpritePassBuilder::new();
        // Everything equal except: layer-in-atlas, colour, flips, order, depth, mesh.
        for (i, s) in v.iter_mut().enumerate() {
            s.atlas_layer = i as u32 % 3;
            s.color = [i as f32 / 6.0, 1.0, 1.0, 1.0];
            s.flip = [i % 2 == 0, i % 3 == 0];
            s.sort_point = [0.0, 0.0, i as f32];
            s.mesh.first_index = 60 + i as u32 * 6;
        }
        assert_eq!(batches(b.build(&v, &[], &cfg())).len(), 1, "layer, colour, flip, depth, mesh never break");
        // Material change breaks.
        v[3].material = 2;
        let f = b.build(&v, &[], &cfg());
        let bs = batches(f);
        assert_eq!(bs.len(), 3);
        assert_eq!(bs.iter().map(|b| b.instance_count).sum::<u32>(), 6);
        // Texture change breaks.
        v[3].material = 1;
        v[2].texture = 9;
        assert_eq!(batches(b.build(&v, &[], &cfg())).len(), 3);
        // Stencil (mask interaction) change breaks.
        v[2].texture = 1;
        v[4].mask = MaskMode::VisibleInside;
        let bs = batches(b.build(&v, &[], &cfg()));
        assert!(bs.len() >= 3);
        assert!(bs.iter().any(|b| b.stencil == MaskMode::VisibleInside));
    }

    #[test]
    fn same_mesh_runs_merge_but_different_meshes_do_not() {
        let a = src(0, 0);
        let mut v = vec![a; 5];
        let mut b = SpritePassBuilder::new();
        let f = b.build(&v, &[], &cfg());
        assert_eq!(f.draws.len(), 1);
        assert_eq!(f.draws[0].instance_count, 5);
        v[2].mesh.first_index += 6;
        let f = b.build(&v, &[], &cfg());
        assert_eq!(f.draws.len(), 3);
        assert_eq!(batches(f).len(), 1);
        assert_eq!(batches(f)[0].draw_count, 3);
        // Draw commands index the instances in draw order.
        let mut next = 0;
        for d in &f.draws {
            assert_eq!(d.first_instance, next);
            next += d.instance_count;
        }
        assert_eq!(next as usize, f.instances.len());
    }

    #[test]
    fn layers_and_orders_decide_draw_order() {
        let mut a = src(1, 0);
        a.texture = 1;
        let mut b = src(0, 5);
        b.texture = 2;
        let mut c = src(0, -5);
        c.texture = 3;
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[a, b, c], &[], &cfg());
        let order: Vec<u32> = batches(f).iter().map(|b| b.texture).collect();
        assert_eq!(order, vec![3, 2, 1], "layer 0 order -5, layer 0 order 5, layer 1");
    }

    #[test]
    fn depth_breaks_ties_back_to_front() {
        let mut near = src(0, 0);
        near.sort_point = [0.0, 0.0, -1.0]; // axis -z: larger = farther
        near.position = [1.0, 0.0, 0.0];
        let mut far = src(0, 0);
        far.sort_point = [0.0, 0.0, -50.0];
        far.position = [2.0, 0.0, 0.0];
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[near, far], &[], &cfg());
        // distance(far) = 50 > distance(near) = 1 -> far is drawn first.
        assert_eq!(f.instances[0].pos[0], 2.0);
        assert_eq!(f.instances[1].pos[0], 1.0);
    }

    #[test]
    fn equal_keys_keep_input_order() {
        let mut v = Vec::new();
        for i in 0..50 {
            let mut s = src(0, 0);
            s.position = [i as f32, 0.0, 0.0];
            s.mesh.first_index = 60 + i * 6; // distinct meshes so none merge
            v.push(s);
        }
        let mut p = SpritePassBuilder::new();
        let f = p.build(&v, &[], &cfg());
        let xs: Vec<f32> = f.instances.iter().map(|i| i.pos[0]).collect();
        assert_eq!(xs, (0..50).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn instance_cap_drops_trailing_sources() {
        let v = vec![src(0, 0); 10];
        let mut p = SpritePassBuilder::new();
        let f = p.build(&v, &[], &BuildConfig { max_instances: 4, ..cfg() });
        assert_eq!(f.instances.len(), 4);
        assert_eq!(f.stats.dropped, 6);
    }

    // --- determinism / parallel gather ------------------------------------

    fn big_scene(n: usize) -> Vec<SpriteSource> {
        let mut state = 99u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        (0..n)
            .map(|i| {
                let mut s = src((next() % 5) as i16 - 2, (next() % 20) as i16 - 10);
                s.position = [(next() % 1000) as f32 * 0.1, (next() % 1000) as f32 * 0.1, (next() % 100) as f32 * 0.01];
                s.sort_point = s.position;
                s.material = next() % 3;
                s.texture = next() % 4;
                s.atlas_layer = next() % 8;
                s.flip = [next() % 2 == 0, next() % 2 == 0];
                s.mesh.first_index = 60 + (next() % 5) * 6;
                if i % 97 == 0 {
                    s.draw = DrawMode::Tiled;
                    s.size = [3.0 + (next() % 5) as f32, 2.0];
                    s.border = [0.25; 4];
                } else if i % 31 == 0 {
                    s.draw = DrawMode::Sliced;
                    s.size = [2.0, 2.0];
                    s.border = [0.25; 4];
                }
                s
            })
            .collect()
    }

    #[test]
    fn output_is_independent_of_thread_count() {
        let scene = big_scene(30_000);
        let mut one = SpritePassBuilder::new();
        let mut many = SpritePassBuilder::new();
        let a = one.build(&scene, &[], &BuildConfig { threads: 1, ..Default::default() });
        let (ai, ad, ait) = (a.instances.clone(), a.draws.clone(), a.items.clone());
        let b = many.build(&scene, &[], &BuildConfig { threads: 4, ..Default::default() });
        assert_eq!(ai, b.instances);
        assert_eq!(ad, b.draws);
        assert_eq!(ait, b.items);
        // And rebuilding with warm pools gives the same frame again.
        let c = many.build(&scene, &[], &BuildConfig { threads: 4, ..Default::default() });
        assert_eq!(ai, c.instances);
    }

    #[test]
    fn large_scene_is_sorted_and_counts_add_up() {
        let scene = big_scene(20_000);
        let mut p = SpritePassBuilder::new();
        let f = p.build(&scene, &[], &cfg());
        let total_draw: u32 = f.draws.iter().map(|d| d.instance_count).sum();
        assert_eq!(total_draw as usize, f.instances.len());
        let total_batch: u32 = batches(f).iter().map(|b| b.instance_count).sum();
        assert_eq!(total_batch as usize, f.instances.len());
        assert!(f.stats.batches as usize <= f.instances.len());
    }

    // --- sorting groups ----------------------------------------------------

    #[test]
    fn group_is_one_unit_with_internal_order() {
        // Group A (layer 0, order 10) holds renderers with orders 5 and -5 (inside
        // it, -5 first); an outside renderer at order 8 must not interleave.
        let groups = [GroupNode { layer: 0, order: 10, ..Default::default() }];
        let members = [
            GroupMember { group: Some(0), layer: 0, order: 5 },
            GroupMember { group: Some(0), layer: 0, order: -5 },
            GroupMember { group: None, layer: 0, order: 8 },
            GroupMember { group: None, layer: 0, order: 12 },
        ];
        let mut out = Vec::new();
        assert_eq!(flatten_groups(&groups, &members, &mut GroupScratch::default(), &mut out), 0);
        assert_eq!((out[0].layer, out[0].order), (0, 10), "grouped renderers take the root's layer/order");
        assert_eq!((out[2].order, out[2].ordinal), (8, 0));
        assert!(out[1].ordinal < out[0].ordinal, "internal order");
        // Full sort: 8 (outside) < group{-5, 5} < 12 (outside).
        let mut keys: Vec<(SortKey, usize)> = out.iter().enumerate().map(|(i, a)| (SortKey::pack(a.layer, a.order, a.ordinal, 0, 0), i)).collect();
        keys.sort();
        assert_eq!(keys.iter().map(|k| k.1).collect::<Vec<_>>(), vec![2, 1, 0, 3]);
    }

    #[test]
    fn nested_groups_flatten_depth_first_and_sort_at_root_splits() {
        // root(0) contains renderer r0 (order 0), nested group 1 (order 5) holding
        // r1 (order 1), r2 (order 0), and renderer r3 (order 9).
        let groups = [
            GroupNode { layer: 0, order: 0, ..Default::default() },
            GroupNode { parent: Some(0), layer: 0, order: 5, ..Default::default() },
        ];
        let members = [
            GroupMember { group: Some(0), layer: 0, order: 0 },
            GroupMember { group: Some(1), layer: 0, order: 1 },
            GroupMember { group: Some(1), layer: 0, order: 0 },
            GroupMember { group: Some(0), layer: 0, order: 9 },
        ];
        let mut out = Vec::new();
        let mut sc = GroupScratch::default();
        flatten_groups(&groups, &members, &mut sc, &mut out);
        // Depth-first: r0, [nested: r2, r1], r3.
        let seq = {
            let mut ix: Vec<usize> = (0..4).collect();
            ix.sort_by_key(|&i| out[i].ordinal);
            ix
        };
        assert_eq!(seq, vec![0, 2, 1, 3]);
        assert!(out.iter().all(|a| (a.layer, a.order) == (0, 0)), "all take the root's key");
        // sort_at_root: the nested group becomes its own root with its own key.
        let mut g2 = groups;
        g2[1].sort_at_root = true;
        flatten_groups(&g2, &members, &mut sc, &mut out);
        assert_eq!((out[1].layer, out[1].order), (0, 5));
        assert_eq!((out[0].layer, out[0].order), (0, 0));
    }

    #[test]
    fn tied_roots_order_far_first_and_stay_contiguous() {
        let groups = [
            GroupNode { layer: 0, order: 0, depth: 1.0, ..Default::default() }, // near
            GroupNode { layer: 0, order: 0, depth: 9.0, ..Default::default() }, // far: first
        ];
        let members = [
            GroupMember { group: Some(0), layer: 0, order: 0 },
            GroupMember { group: Some(1), layer: 0, order: 0 },
            GroupMember { group: Some(0), layer: 0, order: 1 },
            GroupMember { group: Some(1), layer: 0, order: 1 },
        ];
        let mut out = Vec::new();
        flatten_groups(&groups, &members, &mut GroupScratch::default(), &mut out);
        let mut ix: Vec<usize> = (0..4).collect();
        ix.sort_by_key(|&i| out[i].ordinal);
        assert_eq!(ix, vec![1, 3, 0, 2], "far group's renderers first, each group contiguous");
    }

    #[test]
    fn group_cycles_and_bad_indices_are_survivable() {
        let groups = [
            GroupNode { parent: Some(1), ..Default::default() },
            GroupNode { parent: Some(0), ..Default::default() },
            GroupNode { parent: Some(77), ..Default::default() },
        ];
        let members = [GroupMember { group: Some(0), ..Default::default() }, GroupMember { group: Some(99), ..Default::default() }, GroupMember { group: Some(2), ..Default::default() }];
        let mut out = Vec::new();
        flatten_groups(&groups, &members, &mut GroupScratch::default(), &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].ordinal, 0, "invalid group = ungrouped");
    }

    // --- masks -------------------------------------------------------------

    fn mask(layer: i16, order: i16, range: MaskRange) -> MaskSource {
        MaskSource { scale: [1.0; 2], mesh: MeshRange { first_index: 66, index_count: 6, vertex_offset: 20 }, texture: 7, layer, order, range, alpha_cutoff: 0.3, ..Default::default() }
    }

    fn kinds(f: &SpriteFrame) -> Vec<&'static str> {
        f.items
            .iter()
            .map(|i| match i {
                PassItem::Batch(b) => match b.stencil {
                    MaskMode::None => "S",
                    MaskMode::VisibleInside => "I",
                    MaskMode::VisibleOutside => "O",
                },
                PassItem::MaskWrite { .. } => "W",
                PassItem::MaskErase { .. } => "C",
            })
            .collect()
    }

    #[test]
    fn stencil_ops_are_defined() {
        assert_eq!(MASK_WRITE_STATE.compare, StencilCompare::Always);
        assert_eq!(MASK_WRITE_STATE.pass_op, StencilOp::Replace);
        assert_eq!(MASK_WRITE_STATE.reference, MASK_BIT, "masks write the one bit the sprite pass owns");
        assert_eq!((MASK_WRITE_STATE.compare_mask, MASK_WRITE_STATE.write_mask), (MASK_BIT, MASK_BIT));
        assert_eq!(MASK_BIT, 0x80);
        assert_eq!(MASK_ERASE_STATE.pass_op, StencilOp::Zero);
        assert_eq!(MaskMode::None.test_state(), None);
        let i = MaskMode::VisibleInside.test_state().unwrap();
        let o = MaskMode::VisibleOutside.test_state().unwrap();
        assert_eq!((i.compare, i.pass_op, i.write_mask), (StencilCompare::Equal, StencilOp::Keep, 0));
        assert_eq!((o.compare, o.pass_op, o.write_mask), (StencilCompare::NotEqual, StencilOp::Keep, 0));
        assert_eq!(i.reference, o.reference);
    }

    #[test]
    fn mask_is_written_before_the_sprites_it_affects() {
        let mut spr = src(0, 5);
        spr.mask = MaskMode::VisibleInside;
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[spr], &[mask(0, 0, MaskRange::Auto)], &cfg());
        assert_eq!(kinds(f), vec!["W", "I"]);
        let PassItem::MaskWrite { texture, draw, .. } = f.items[0] else { panic!() };
        assert_eq!(texture, 7);
        assert_eq!(f.draws[draw as usize].index_count, 6);
        assert!((f.instances[f.draws[draw as usize].first_instance as usize].alpha_cutoff - 0.3).abs() < 1e-6);
    }

    #[test]
    fn unmasked_sprites_never_touch_the_stencil() {
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[src(0, 1), src(0, 2)], &[mask(0, 0, MaskRange::Auto)], &cfg());
        assert_eq!(kinds(f), vec!["S"], "no masked sprite, no mask draw");
        assert_eq!(f.stats.mask_writes, 0);
    }

    #[test]
    fn mask_state_machine_clears_and_rewrites_on_range_change() {
        let mut inside = src(0, 1);
        inside.mask = MaskMode::VisibleInside;
        let mut outside = src(0, 20);
        outside.mask = MaskMode::VisibleOutside;
        // Auto ranges: mask A [order 0, order 9], mask B [order 10, ...).
        let masks = [mask(0, 0, MaskRange::Auto), mask(0, 10, MaskRange::Auto)];
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[inside, outside], &masks, &cfg());
        // A written, inside sprite, clear, B written, outside sprite.
        assert_eq!(kinds(f), vec!["W", "I", "C", "W", "O"]);
        let writes: Vec<u32> = f.items.iter().filter_map(|i| if let PassItem::MaskWrite { draw, .. } = i { Some(*draw) } else { None }).collect();
        assert_eq!(writes.len(), 2);
        assert_eq!(f.stats.mask_erases, 1);
        // The erase replays exactly the draw that wrote mask A.
        let erase = f.items.iter().find_map(|i| if let PassItem::MaskErase { draw, .. } = i { Some(*draw) } else { None }).unwrap();
        assert_eq!(erase, writes[0]);
    }

    #[test]
    fn overlapping_masks_union_and_same_state_is_not_rewritten() {
        let a = mask(0, 0, MaskRange::Custom { back: (0, 0), front: (0, 100) });
        let b = mask(0, 5, MaskRange::Custom { back: (0, 0), front: (0, 100) });
        let mut s1 = src(0, 1);
        let mut s2 = src(0, 2);
        s1.mask = MaskMode::VisibleInside;
        s2.mask = MaskMode::VisibleInside;
        s2.material = 1; // forces a second batch but not a second mask write
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[s1, s2], &[a, b], &cfg());
        assert_eq!(kinds(f), vec!["W", "W", "I", "I"], "both masks, written once");
    }

    #[test]
    fn sprite_outside_every_range_sees_a_clear_buffer() {
        let mut inside = src(0, 1);
        inside.mask = MaskMode::VisibleInside;
        let mut after = src(0, 50);
        after.mask = MaskMode::VisibleInside;
        let m = mask(0, 0, MaskRange::Custom { back: (0, 0), front: (0, 10) });
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[inside, after], &[m], &cfg());
        // Second sprite is out of range: clear, then it draws against empty stencil
        // (VisibleInside shows nothing there), as a separate batch? same state -> a
        // new batch only because the clear closed the first.
        assert_eq!(kinds(f), vec!["W", "I", "C", "I"]);
    }

    #[test]
    fn custom_range_swapped_bounds_and_layer_precedence() {
        let m = mask(0, 0, MaskRange::Custom { back: (1, 0), front: (0, 0) });
        let mut s = src(1, -5); // layer 1 beats order: inside [ (0,0) .. (1,0) ]
        s.mask = MaskMode::VisibleOutside;
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[s], &[m], &cfg());
        assert_eq!(kinds(f), vec!["W", "O"]);
    }

    // --- behind / in front of 3D transparents -----------------------------

    #[test]
    fn behind_boundary_forces_one_break_and_counts_items() {
        let mut v = vec![src(-1, 0), src(0, 0), src(1, 0)];
        for s in &mut v {
            s.material = 1;
            s.texture = 1;
        }
        let mut p = SpritePassBuilder::new();
        let f = p.build(&v, &[], &BuildConfig { behind_layer_max: Some(0), ..cfg() });
        assert_eq!(f.items.len(), 2, "one batch behind, one in front, despite equal state");
        assert_eq!(f.behind_items, 1, "layers <= 0 are the first batch");
        // Layers -1 and 0 share a batch; the boundary item index is after it.
        let f = p.build(&v, &[], &BuildConfig { behind_layer_max: Some(5), ..cfg() });
        assert_eq!(f.behind_items, f.items.len(), "everything behind");
        let f = p.build(&v, &[], &cfg());
        assert_eq!(f.behind_items, 0, "default: everything after 3D transparents");
    }

    // --- mesh pool / instance layout --------------------------------------

    #[test]
    fn instance_layout_is_stable() {
        assert_eq!(std::mem::size_of::<SpriteInstance>(), 112);
        assert_eq!(std::mem::size_of::<SpriteVertex>(), 16);
        assert_eq!(std::mem::size_of::<DrawIndexedIndirect>(), 20);
        assert_eq!(std::mem::offset_of!(SpriteInstance, uv_rect), 32);
        assert_eq!(std::mem::offset_of!(SpriteInstance, color), 96);
    }

    #[test]
    fn mesh_pool_appends_and_tracks_upload() {
        let mut pool = SpriteMeshPool::new(64, 64);
        let (v0, v, i0, i) = pool.take_pending();
        assert_eq!((v0, v.len(), i0, i.len()), (0, 20, 0, 60), "templates first");
        let r = pool.add(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &[[0.0; 2]; 3], &[0, 1, 2]).unwrap();
        assert_eq!(r, MeshRange { first_index: 60, index_count: 3, vertex_offset: 20 });
        let (v0, v, i0, i) = pool.take_pending();
        assert_eq!((v0, v.len(), i0, i.len()), (20, 3, 60, 3));
        assert!(pool.add(&[[0.0; 2]], &[[0.0; 2]], &[5]).is_none(), "index out of range");
        assert!(pool.add(&[[0.0; 2]; 2], &[[0.0; 2]; 1], &[0]).is_none(), "length mismatch");
        assert!(pool.add(&[[0.0; 2]; 50], &[[0.0; 2]; 50], &[0]).is_none(), "pool full");
    }

    #[test]
    fn shape_meshes_change_size_rotate_slots_and_are_reused() {
        let mut pool = SpriteMeshPool::new(512, 512);
        let _ = pool.take_pending();
        let h = pool.add_shape_mesh(8, 12).unwrap();
        assert_eq!(pool.shape_mesh_capacity(h), Some((8, 12)));
        let tri = |x: f32| [[x, 0.0], [x + 1.0, 0.0], [x, 1.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        // First write rides with the pending tail: nothing dirty.
        let r0 = pool.write_shape_mesh(h, &tri(0.0), &uv, &[0, 1, 2]).unwrap();
        assert!(pool.take_dirty().is_empty() && pool.take_dirty_indices().is_empty());
        let _ = pool.take_pending();
        // A different size and topology goes to the next slot and marks both lists dirty.
        let quad_pos = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let r1 = pool.write_shape_mesh(h, &quad_pos, &[[0.0; 2]; 4], &[0, 1, 2, 0, 2, 3]).unwrap();
        assert_ne!(r0.vertex_offset, r1.vertex_offset);
        assert_ne!(r0.first_index, r1.first_index, "indices rotate too");
        assert_eq!((r1.index_count, pool.dynamic_count()), (6, 0));
        assert_eq!(pool.take_dirty(), vec![(r1.vertex_offset as usize, 4)]);
        assert_eq!(pool.take_dirty_indices(), vec![(r1.first_index as usize, 6)]);
        assert_eq!(pool.indices()[r1.first_index as usize + 5], 3);
        assert_eq!(pool.indices()[r0.first_index as usize + 2], 2, "the previous slot is untouched");
        let r2 = pool.write_shape_mesh(h, &tri(5.0), &uv, &[0, 1, 2]).unwrap();
        let r3 = pool.write_shape_mesh(h, &tri(6.0), &uv, &[0, 1, 2]).unwrap();
        assert_eq!(r3.vertex_offset, r0.vertex_offset, "wraps after DYNAMIC_SLOTS writes");
        assert!(r2.vertex_offset != r1.vertex_offset);
        // Bad input.
        assert!(pool.write_shape_mesh(h, &[[0.0; 2]; 9], &[[0.0; 2]; 9], &[0, 1, 2]).is_none(), "over capacity");
        assert!(pool.write_shape_mesh(h, &tri(0.0), &uv, &[0, 1, 3]).is_none(), "index out of range");
        assert!(pool.write_shape_mesh(h, &tri(0.0), &uv[..2], &[0, 1, 2]).is_none(), "uv length");
        assert!(pool.write_shape_mesh(ShapeMesh(77), &tri(0.0), &uv, &[0, 1, 2]).is_none());
        assert!(pool.add_shape_mesh(0, 3).is_none() && pool.add_shape_mesh(70_000, 3).is_none());
        // Release + reuse: same capacity class, no growth.
        let used = (pool.vertex_count(), pool.index_count());
        pool.release_shape_mesh(h);
        assert!(pool.write_shape_mesh(h, &tri(0.0), &uv, &[0, 1, 2]).is_none(), "released handles do not write");
        let again = pool.add_shape_mesh(6, 9).unwrap();
        assert_eq!(again, h);
        assert_eq!((pool.vertex_count(), pool.index_count()), used);
        assert_eq!(pool.shape_mesh_count(), 1);
        // Too big for the freed one: a fresh mesh. A pool that cannot fit three slots refuses.
        let big = pool.add_shape_mesh(100, 100).unwrap();
        assert_ne!(big, h);
        assert!(pool.add_shape_mesh(400, 10).is_none());
        pool.recycle_dirty_indices(Vec::new());
    }

    #[test]
    fn dynamic_meshes_rotate_slots_and_mark_only_uploaded_ranges_dirty() {
        let mut pool = SpriteMeshPool::new(256, 256);
        let _ = pool.take_pending();
        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let d = pool.add_dynamic(&uvs, &[0, 1, 2]).unwrap();
        assert!(pool.dynamic_range(d).is_none());
        // Not uploaded yet: the first write rides with the pending tail.
        let r0 = pool.write_dynamic(d, &[[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]).unwrap();
        assert!(pool.take_dirty().is_empty());
        let (v0, v, _, i) = pool.take_pending();
        assert_eq!((v.len(), i.len()), (3 * DYNAMIC_SLOTS, 3));
        assert_eq!(v0, 20);
        // Later writes go to the next slot, never the one the GPU may be reading.
        let r1 = pool.write_dynamic(d, &[[3.0, 3.0], [4.0, 3.0], [3.0, 4.0]]).unwrap();
        let r2 = pool.write_dynamic(d, &[[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]]).unwrap();
        let r3 = pool.write_dynamic(d, &[[7.0, 7.0], [8.0, 7.0], [7.0, 8.0]]).unwrap();
        assert!(r0.vertex_offset != r1.vertex_offset && r1.vertex_offset != r2.vertex_offset && r2.vertex_offset != r0.vertex_offset);
        assert_eq!(r3.vertex_offset, r0.vertex_offset, "wraps after DYNAMIC_SLOTS writes");
        assert_eq!((r0.first_index, r1.first_index), (r1.first_index, r2.first_index), "one shared index list");
        assert_eq!(pool.dynamic_range(d), Some(r3));
        let dirty = pool.take_dirty();
        assert_eq!(dirty.len(), 3);
        assert_eq!(dirty[0], (r1.vertex_offset as usize, 3));
        // The slots hold their own positions and the shared UVs.
        assert_eq!(pool.vertices()[r1.vertex_offset as usize].pos, [3.0, 3.0]);
        assert_eq!(pool.vertices()[r2.vertex_offset as usize + 2].pos, [5.0, 6.0]);
        assert_eq!(pool.vertices()[r2.vertex_offset as usize + 2].uv, [0.0, 1.0]);
        pool.recycle_dirty(dirty);
        assert!(pool.take_dirty().is_empty());
        // Bad input.
        assert!(pool.write_dynamic(d, &[[0.0; 2]]).is_none(), "length mismatch");
        assert!(pool.write_dynamic(DynamicMesh(99), &[]).is_none());
        assert!(pool.add_dynamic(&uvs, &[0, 1, 5]).is_none());
        let mut small = SpriteMeshPool::new(TEMPLATE_VERTICES + 4, 100);
        assert!(small.add_dynamic(&uvs, &[0, 1, 2]).is_none(), "3 slots do not fit");
        assert_eq!(pool.dynamic_count(), 1);
    }

    #[test]
    fn simple_instance_has_rotation_scale_and_flags() {
        let mut s = src(0, 0);
        s.rotation = std::f32::consts::FRAC_PI_2;
        s.scale = [2.0, 3.0];
        s.position = [1.0, 2.0, 0.5];
        s.flip = [true, false];
        s.mask = MaskMode::VisibleOutside;
        s.color = [1.0, 0.5, 0.0, 1.0];
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[s], &[], &cfg());
        let i = f.instances[0];
        assert!((i.affine[0]).abs() < 1e-6 && (i.affine[1] - 2.0).abs() < 1e-6);
        assert!((i.affine[2] + 3.0).abs() < 1e-6 && (i.affine[3]).abs() < 1e-5);
        assert_eq!((i.pos, i.z), ([1.0, 2.0], 0.5));
        assert_eq!(i.flags & FLAG_FLIP_X, FLAG_FLIP_X);
        assert_eq!(i.flags & FLAG_FLIP_Y, 0);
        assert_eq!(i.mode(), MODE_MESH);
        assert_eq!(i.mask_mode(), MaskMode::VisibleOutside);
        assert_eq!(i.color, 0xFF00_80FF);
    }

    #[test]
    fn sliced_instance_carries_rect_and_uv_borders() {
        let mut s = src(0, 0);
        s.draw = DrawMode::Sliced;
        s.size = [8.0, 4.0];
        s.sprite_size = [4.0, 4.0];
        s.pivot = [2.0, 0.0]; // bottom-centre
        s.border = [1.0, 1.0, 1.0, 1.0];
        s.uv_rect = [0.0, 0.0, 0.5, 0.5];
        let mut p = SpritePassBuilder::new();
        let f = p.build(&[s], &[], &cfg());
        let i = f.instances[0];
        assert_eq!(i.mode(), MODE_SLICE);
        assert_eq!(i.rect, [-4.0, 0.0, 8.0, 4.0]);
        assert_eq!(i.uv_border, [0.125, 0.125, 0.125, 0.125]);
        assert_eq!(f.draws[0].first_index, SLICE_TEMPLATE.first_index);
        assert_eq!(f.draws[0].vertex_offset, SLICE_TEMPLATE.vertex_offset);
    }

    #[test]
    fn quad_uv_plain_and_rotated_corners() {
        let uv = [0.1, 0.2, 0.3, 0.4]; // u0 v0(top) u1 v1(bottom)
        // Plain: bottom-left corner is (u0, v1), top-right is (u1, v0).
        assert_eq!(quad_uv(uv, [0.0, 0.0], false), [0.1, 0.4]);
        assert_eq!(quad_uv(uv, [1.0, 1.0], false), [0.3, 0.2]);
        // Rotated 90 degrees clockwise in the page: the sprite's top-left lands
        // at the page rect's top-right, its bottom-left at the top-left.
        let r = |c| quad_uv(uv, c, true);
        let close = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
        assert!(close(r([0.0, 0.0]), [0.1, 0.2]));
        assert!(close(r([0.0, 1.0]), [0.3, 0.2]));
        assert!(close(r([1.0, 1.0]), [0.3, 0.4]));
        assert!(close(r([1.0, 0.0]), [0.1, 0.4]));
    }

    #[test]
    fn animation_frames_loop_hold_and_encode() {
        assert_eq!(anim_frame(4, 2.0, 0, 0.0), 0);
        assert_eq!(anim_frame(4, 2.0, 0, 0.49), 0);
        assert_eq!(anim_frame(4, 2.0, 0, 0.5), 1);
        assert_eq!(anim_frame(4, 2.0, 0, 2.0), 0, "wraps");
        assert_eq!(anim_frame(4, 2.0, 0, 7.6), 3);
        assert_eq!(anim_frame(4, 2.0, ANIM_LOOP_ONCE, 100.0), 3, "holds the last frame");
        assert_eq!(anim_frame(4, 2.0, 0, -3.0), 0);
        assert_eq!(anim_frame(4, 0.0, 0, 5.0), 0);
        assert_eq!(anim_frame(1, 5.0, 0, 5.0), 0);
        let mut t = Vec::new();
        let f = |k: f32| ([k, 0.0, k + 1.0, 1.0], [0.0, 0.0, 1.0, 1.0 + k], k == 1.0);
        let a = push_animation(&mut t, &[f(0.0), f(1.0), f(2.0)], 3.0, false);
        let b = push_animation(&mut t, &[f(5.0), f(6.0)], 1.0, true);
        assert_eq!(a, 1, "entry 0 is the dummy");
        assert_eq!(b, 1 + 1 + 6);
        let (uv, rect, rot) = anim_lookup(&t, a, 1.0 / 3.0 + 1e-3).unwrap();
        assert_eq!((uv[0], rect[3], rot), (1.0, 2.0, true));
        let (uv, _, rot) = anim_lookup(&t, b, 99.0).unwrap();
        assert_eq!((uv[0], rot), (6.0, false));
        assert!(anim_lookup(&t, 500, 0.0).is_none());
        // More frames than the rotated mask has bits are cut.
        let many: Vec<_> = (0..40).map(|k| f(k as f32)).collect();
        let c = push_animation(&mut t, &many, 1.0, false);
        assert_eq!(AnimHeader::decode(t[c as usize]).count as usize, ANIM_MAX_FRAMES);
    }

    #[test]
    fn quad_sources_carry_rotation_animation_and_rect_into_instances() {
        let s = SpriteSource {
            scale: [1.0, 1.0],
            draw: DrawMode::Quad,
            rect: [-0.5, -0.5, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 0.5, 0.5],
            rotated: true,
            anim: 7,
            anim_phase: 0.25,
            atlas_layer: 3,
            ..Default::default()
        };
        let mut b = SpritePassBuilder::new();
        let f = b.build(&[s], &[], &BuildConfig { threads: 1, ..Default::default() });
        let i = f.instances[0];
        assert_eq!(i.mode(), MODE_QUAD);
        assert!(i.flags & FLAG_ROTATED != 0 && i.flags & FLAG_ANIM != 0);
        assert_eq!((i.anim, i.alpha_cutoff, i.layer, i.rect), (7, 0.25, 3, [-0.5, -0.5, 1.0, 1.0]));
        assert_eq!(f.draws[0].index_count, QUAD_TEMPLATE.index_count);
        assert_eq!(std::mem::size_of::<SpriteInstance>(), 112, "the instance layout is unchanged");
        // A plain quad has neither flag.
        let p = SpriteSource { scale: [1.0, 1.0], draw: DrawMode::Quad, ..Default::default() };
        let f = b.build(&[p], &[], &BuildConfig { threads: 1, ..Default::default() });
        assert_eq!(f.instances[0].flags & (FLAG_ROTATED | FLAG_ANIM), 0);
    }
}
