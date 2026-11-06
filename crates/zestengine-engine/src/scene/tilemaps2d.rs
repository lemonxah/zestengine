//! Tilemaps in the 2D sprite pass (2D-5): turns `Tilemap` + `TilemapRenderer`
//! components into `SpriteSource` records appended to the renderer's sprite
//! input each frame.
//!
//! # Design (documented choice: reuse the sprite instance, no new draw variant)
//!
//! A tile is a `DrawMode::Quad` sprite: the sprite pass draws it from the
//! shared unit-quad template with a per-instance rect and UV rect, so tiles
//! batch, sort and cull with everything else and need no second pipeline. The
//! 112-byte `SpriteInstance` stays as it was.
//!
//! - **Chunks**: the 16 x 16 storage chunks are the render chunks. Each keeps a
//!   cached `Vec<SpriteSource>` (already ordered by the renderer's tile sort
//!   order) tagged with the chunk's `version`; an edit re-gathers only the
//!   chunks whose version changed (an edit on a chunk border also bumps the
//!   neighbour, see `Tilemap`). A cached chunk is rebuilt when the object
//!   moves, a tile file changes or new sprites resolve (the map's cache key).
//! - **Culling**: each frame a chunk's world AABB (exact once built, the cell
//!   rectangle plus a margin before) is tested against the camera's
//!   view-projection; only visible chunks are copied into the frame.
//! - **Batching**: tiles of one atlas page are one batch (texture and material
//!   break batches, the array layer does not). Chunk mode gives all tiles of
//!   the map one sort key, so the draw order inside is the emission order.
//!   Individual mode sorts every tile by its own depth against other sprites.
//! - **Animation**: an animated tile carries a frame-table index and a phase;
//!   the vertex shader picks the frame from the frame clock (no CPU work per
//!   frame). Identical animations share one table entry.
//! - **Eviction**: chunks not drawn for `max_frame_age` frames are dropped once
//!   the cache holds more than the visible count plus `max_chunk_count`.
//!
//! Cost: a scene without tilemaps does one component type check per object per
//! frame (the sprite scan already does it); a quiet map costs the visible
//! chunks' memcpy.

use std::collections::HashMap;
use std::path::Path;

use glam::{IVec2, Mat4, Vec2, Vec3};
use zestengine_core::{
    Component, Grid, Tilemap, TilemapMode, TilemapOrientation, TilemapRenderer, TileSortOrder, TileXform, component_enabled, component_inner,
};
use zestengine_render::Renderer;
use zestengine_render::sprite2d::{ANIM_MAX_FRAMES, DrawMode, MaskMode, SpriteSource, push_animation};

use super::LoadedScene;
use super::sprites2d::SpriteCache;
use crate::tile_db::{LoadedTile, TileDb, grid_for, resolve_cell};

/// Frames between tile file timestamp checks.
const DB_CHECK: u32 = 60;
/// Animation table entry cap (matches the GPU buffer).
const ANIM_CAP: usize = zestengine_render::SPRITE_ANIM_CAP;

pub fn is_tile_component(c: &dyn Component) -> bool {
    let a = c.as_any();
    a.is::<Tilemap>() || a.is::<TilemapRenderer>()
}

/// What a tile needs of a resolved sprite.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TileSprite {
    pub texture: u32,
    pub layer: u32,
    /// Local rect relative to the sprite pivot (min, size).
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub rotated: bool,
}

/// The shared animation frame table plus a de-duplicating index.
#[derive(Default)]
pub struct AnimTable {
    pub table: Vec<[f32; 4]>,
    index: HashMap<AnimKey, u32>,
}

/// An animation's identity: frame row set, fps, loop flag, and fps-in-row encoding.
type AnimKey = (Vec<[u32; 8]>, u32, u32, bool);

impl AnimTable {
    /// The header index of the animation of `frames` (all on the first frame's
    /// texture and layer; later frames elsewhere are dropped). `None` when the
    /// table is full or there is only one frame.
    pub fn intern(&mut self, frames: &[TileSprite], fps: f32, loop_once: bool) -> Option<u32> {
        let first = *frames.first()?;
        let same: Vec<&TileSprite> = frames.iter().take_while(|f| f.texture == first.texture && f.layer == first.layer).take(ANIM_MAX_FRAMES).collect();
        if same.len() < 2 {
            return None;
        }
        let mut rotated = 0u32;
        let key_frames: Vec<[u32; 8]> = same
            .iter()
            .enumerate()
            .map(|(k, f)| {
                rotated |= (f.rotated as u32) << k;
                let b = |a: [f32; 4]| a.map(f32::to_bits);
                let (u, r) = (b(f.uv), b(f.rect));
                [u[0], u[1], u[2], u[3], r[0], r[1], r[2], r[3]]
            })
            .collect();
        let key = (key_frames, rotated, fps.to_bits(), loop_once);
        if let Some(&i) = self.index.get(&key) {
            return Some(i);
        }
        if self.table.len() + 2 + 2 * same.len() > ANIM_CAP {
            return None;
        }
        let list: Vec<([f32; 4], [f32; 4], bool)> = same.iter().map(|f| (f.uv, f.rect, f.rotated)).collect();
        let at = push_animation(&mut self.table, &list, fps, loop_once);
        self.index.insert(key, at);
        Some(at)
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.index.clear();
    }
}

/// Per-map values the chunk builder reads.
pub struct MapEnv {
    pub world: Mat4,
    pub grid: Grid,
    pub anchor: [f32; 2],
    pub map_color: [f32; 4],
    pub frame_rate: f32,
    pub time_offset: f32,
    pub mode: TilemapMode,
    pub sort_order: TileSortOrder,
    pub layer: i16,
    pub order: i16,
    pub mask: MaskMode,
    /// A ground map (`TilemapOrientation::XZ`): cell (x, y) at local (x, 0, y), drawn flat in the
    /// world XZ plane through the sprite pass (depth tested, sprite "up" toward -Z).
    pub xz: bool,
    /// The sprite material id of `TilemapRenderer.material` (the default material when none) and the
    /// tint its `base_color` multiplies into every tile.
    pub material: u32,
    pub tint: [f32; 4],
}

/// One built chunk: its sources and the world AABB of their geometry.
#[derive(Default)]
pub struct CachedChunk {
    pub version: u32,
    pub sources: Vec<SpriteSource>,
    pub lo: Vec2,
    pub hi: Vec2,
    used: u32,
}

struct MapCache {
    chunks: HashMap<(i32, i32), CachedChunk>,
    key: u64,
    /// `(serial, db revision)` the sprites were last resolved for.
    resolved_for: (u64, u64),
}

/// Counters of the last gather (tests, the stats overlay).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct TileStats {
    pub maps: u32,
    pub chunks_total: u32,
    pub chunks_visible: u32,
    pub chunks_rebuilt: u32,
    pub tiles: u32,
}

#[derive(Default)]
pub struct TileGatherCache {
    db: Option<TileDb>,
    maps: HashMap<zestengine_core::ObjectId, MapCache>,
    pub anim: AnimTable,
    order: Vec<IVec2>,
    cells: Vec<(IVec2, u16)>,
    need: Vec<String>,
    targets: Vec<usize>,
    frame: u32,
    pub stats: TileStats,
}

impl TileGatherCache {
    pub fn invalidate(&mut self) {
        self.db = None;
        self.maps.clear();
        self.anim.clear();
    }
}

fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x100_0000_01B3).rotate_left(23)
}

/// Whether the AABB `lo`..`hi` (world, at depth `z`) lies fully outside the
/// clip volume of `view_proj` on one side (x or y); conservative for boxes
/// that reach behind the camera.
#[allow(dead_code)] // bounds check used by the tilemap tests
pub fn aabb_outside(view_proj: &Mat4, lo: Vec2, hi: Vec2, z: f32) -> bool {
    aabb_outside_in(view_proj, lo, hi, z, false)
}

/// [`aabb_outside`] for a box in the XY plane at height `z`, or (`xz`) in the XZ plane at height
/// `y = z`, where `lo` / `hi` are (x, z).
pub fn aabb_outside_in(view_proj: &Mat4, lo: Vec2, hi: Vec2, z: f32, xz: bool) -> bool {
    let at = |x: f32, v: f32| if xz { Vec3::new(x, z, v) } else { Vec3::new(x, v, z) };
    let corners = [at(lo.x, lo.y), at(hi.x, lo.y), at(hi.x, hi.y), at(lo.x, hi.y)];
    let mut out = [true; 4]; // left, right, bottom, top
    for c in corners {
        let p = *view_proj * c.extend(1.0);
        if p.w <= 1e-6 {
            return false;
        }
        out[0] &= p.x < -p.w;
        out[1] &= p.x > p.w;
        out[2] &= p.y < -p.w;
        out[3] &= p.y > p.w;
    }
    out.iter().any(|&o| o)
}

/// World AABB of a chunk's cell block (cell corners, plus `margin` world units).
fn chunk_cell_aabb(env: &MapEnv, ck: IVec2, margin: f32) -> (Vec2, Vec2) {
    let c0 = ck * zestengine_core::CHUNK_SIZE;
    let n = zestengine_core::CHUNK_SIZE - 1;
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for d in [IVec2::ZERO, IVec2::new(n, 0), IVec2::new(0, n), IVec2::new(n, n)] {
        let cell = (c0 + d).extend(0);
        for p in env.grid.cell_corners(cell) {
            let w = if env.xz { plane_point(&env.world, p) } else { env.world.transform_point3(p).truncate() };
            lo = lo.min(w);
            hi = hi.max(w);
        }
    }
    (lo - Vec2::splat(margin), hi + Vec2::splat(margin))
}

/// A cell-local point of an XZ map (x, y -> local x, z) in world space, as plane coordinates (x, z).
fn plane_point(w: &Mat4, p: Vec3) -> Vec2 {
    let q = w.transform_point3(Vec3::new(p.x, 0.0, p.y));
    Vec2::new(q.x, q.z)
}

/// Rotation about Z, scale, and whether the 2x2 mirrors (as in sprites2d).
fn decompose_2d(w: &Mat4) -> (f32, [f32; 2], bool) {
    decompose_axes(Vec2::new(w.x_axis.x, w.x_axis.y), Vec2::new(w.y_axis.x, w.y_axis.y))
}

/// The same for an XZ map: a tile's x axis is the object's x axis and its "up" is the object's -z
/// axis (so the picture is upright seen from above with -Z at the top), both in (x, z) plane coordinates.
fn decompose_xz(w: &Mat4) -> (f32, [f32; 2], bool) {
    decompose_axes(Vec2::new(w.x_axis.x, w.x_axis.z), Vec2::new(-w.z_axis.x, -w.z_axis.z))
}

fn decompose_axes(ax: Vec2, ay: Vec2) -> (f32, [f32; 2], bool) {
    let det = ax.x * ay.y - ax.y * ay.x;
    let scale = [ax.length(), ay.length()];
    if det < 0.0 { ((-ay.x).atan2(ay.y), scale, true) } else { (ax.y.atan2(ax.x), scale, false) }
}

/// Build the sources of chunk `ck` of `map`. `sprite` resolves a sprite
/// reference; `tile` a tile path (loading it if need be).
#[allow(clippy::too_many_arguments)]
pub fn build_chunk(
    map: &Tilemap,
    ck: IVec2,
    env: &MapEnv,
    sprite: &dyn Fn(&str) -> Option<TileSprite>,
    tile: &mut dyn FnMut(&str) -> Option<std::sync::Arc<LoadedTile>>,
    anim: &mut AnimTable,
    cells: &mut Vec<(IVec2, u16)>,
    out: &mut CachedChunk,
) {
    out.sources.clear();
    let Some(chunk) = map.chunk(ck) else { return };
    out.version = chunk.version;
    cells.clear();
    for (i, &t) in chunk.tiles.iter().enumerate() {
        if t != 0 {
            cells.push((zestengine_core::cell_at(ck, i), t));
        }
    }
    cells.sort_by_key(|(c, _)| env.sort_order.key(*c));
    let (obj_rot, scale, mirrored) = if env.xz { decompose_xz(&env.world) } else { decompose_2d(&env.world) };
    let origin = env.world.w_axis.truncate();
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for &(cell, t) in cells.iter() {
        let Some(path) = map.tile_path(t) else { continue };
        let Some(loaded) = tile(path) else { continue };
        let r = resolve_cell(map, &loaded, path, cell);
        let Some(first) = r.sel.sprites.first().and_then(|s| sprite(s)) else { continue };
        let pivot_local = env.grid.tile_pivot(cell.extend(0), env.anchor);
        // The instance position: world (x, y) in the 2D pass; (x, z) with the height in `z` for a ground map.
        let world = if env.xz {
            let w = env.world.transform_point3(Vec3::new(pivot_local.x, 0.0, pivot_local.y));
            Vec3::new(w.x, w.z, w.y)
        } else {
            env.world.transform_point3(pivot_local)
        };
        let x: TileXform = r.xform;
        let (rotation, flip) = if mirrored { (obj_rot - x.angle(), !x.mirror) } else { (obj_rot + x.angle(), x.mirror) };
        let mut color = r.color;
        for (k, c) in color.iter_mut().enumerate() {
            *c *= env.map_color[k] * env.tint[k];
        }
        let mut src = SpriteSource {
            position: world.to_array(),
            rotation,
            scale,
            // The real world point depth is measured at (for an XZ map `world` holds (x, z, height)).
            sort_point: if env.mode == TilemapMode::Individual {
                if env.xz { [world.x, world.z, world.y] } else { world.to_array() }
            } else {
                origin.to_array()
            },
            draw: DrawMode::Quad,
            rect: first.rect,
            uv_rect: first.uv,
            rotated: first.rotated,
            color,
            flip: [flip, false],
            material: env.material | if env.xz { zestengine_render::SPRITE_MATERIAL_XZ } else { 0 },
            texture: first.texture,
            atlas_layer: first.layer,
            layer: env.layer,
            order: env.order,
            mask: env.mask,
            ..Default::default()
        };
        if let Some(a) = r.sel.anim {
            let frames: Vec<TileSprite> = r.sel.sprites.iter().filter_map(|s| sprite(s)).collect();
            if let Some(idx) = anim.intern(&frames, a.speed * env.frame_rate, a.loop_once) {
                src.anim = idx;
                src.anim_phase = a.start_time + env.time_offset;
            }
        }
        // The geometry's bounds: the quad corners (the largest frame for an animation).
        let rects: Vec<[f32; 4]> = if src.anim != 0 { r.sel.sprites.iter().filter_map(|s| sprite(s)).map(|f| f.rect).collect() } else { vec![first.rect] };
        let (s, c) = rotation.sin_cos();
        for q in rects {
            for (px, py) in [(q[0], q[1]), (q[0] + q[2], q[1]), (q[0] + q[2], q[1] + q[3]), (q[0], q[1] + q[3])] {
                let (lx, ly) = (px * scale[0], py * scale[1]);
                let p = Vec2::new(world.x + c * lx - s * ly, world.y + s * lx + c * ly);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        out.sources.push(src);
    }
    out.lo = lo;
    out.hi = hi;
}

impl LoadedScene {
    /// World-space corners of the bounding rectangle of the cells a tilemap object has painted
    /// (the selection outline of a map); None for other objects or an empty map.
    pub fn tilemap_outline(&self, index: usize) -> Option<[Vec3; 4]> {
        let map = self.objects.get(index)?.components.iter().find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<Tilemap>())?;
        let (lo, hi) = map.bounds()?;
        let grid = grid_for(self, index);
        let (mut mn, mut mx) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for c in [lo, hi, IVec2::new(lo.x, hi.y), IVec2::new(hi.x, lo.y)] {
            for p in grid.cell_corners(c.extend(0)) {
                mn = mn.min(p.truncate());
                mx = mx.max(p.truncate());
            }
        }
        let w = self.world_transform(index);
        let xz = map.orientation == TilemapOrientation::XZ;
        let at = |x: f32, y: f32| if xz { w.transform_point3(Vec3::new(x, 0.0, y)) } else { w.transform_point3(Vec3::new(x, y, 0.0)) };
        Some([at(mn.x, mn.y), at(mx.x, mn.y), at(mx.x, mx.y), at(mn.x, mx.y)])
    }

    pub(super) fn gather_tilemaps(&mut self, c: &mut SpriteCache, renderer: &mut Renderer, project_root: &Path, view: Mat4, proj: Mat4, time: f32) {
        let mut tc = std::mem::take(&mut c.tiles);
        tc.stats = TileStats::default();
        tc.frame = tc.frame.wrapping_add(1);
        let mut targets = std::mem::take(&mut tc.targets);
        targets.clear();
        targets.extend((0..self.objects.len()).filter(|&i| {
                let o = &self.objects[i];
                (self.visible_layers >> (o.layer as u32 & 31)) & 1 != 0
                    && o.components.iter().any(|k| component_inner(k.as_ref()).as_any().is::<Tilemap>())
                    && o.components.iter().any(|k| component_enabled(k.as_ref()) && component_inner(k.as_ref()).as_any().is::<TilemapRenderer>())
                    && self.is_active(i)
                    && !self.view_hidden(i)
            }));
        if targets.is_empty() {
            tc.maps.clear();
            tc.targets = targets;
            c.tiles = tc;
            return;
        }
        let db = tc.db.get_or_insert_with(|| TileDb::new(project_root));
        if tc.frame.is_multiple_of(DB_CHECK) && db.refresh() {
            tc.anim.clear();
        }
        let view_proj = proj * view;
        let mut sources_out: Vec<SpriteSource> = std::mem::take(&mut renderer.sprite_input_mut().sprites);
        for &i in &targets {
            self.gather_one_tilemap(c, &mut tc, renderer, project_root, i, &view_proj, &mut sources_out);
        }
        let input = renderer.sprite_input_mut();
        input.sprites = sources_out;
        input.time = time;
        input.anim.clone_from(&tc.anim.table);
        // Drop caches of maps that went away.
        tc.maps.retain(|id, _| targets.iter().any(|&i| self.objects[i].id == *id));
        tc.stats.maps = targets.len() as u32;
        tc.targets = targets;
        c.tiles = tc;
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_one_tilemap(
        &mut self,
        c: &mut SpriteCache,
        tc: &mut TileGatherCache,
        renderer: &mut Renderer,
        project_root: &Path,
        i: usize,
        view_proj: &Mat4,
        out: &mut Vec<SpriteSource>,
    ) {
        // 1. Which sprites does the map need? Resolve the new ones.
        let id = self.objects[i].id;
        let Some(map) = self.objects[i].components.iter().find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<Tilemap>()) else { return };
        let Some(rend) = self.objects[i].components.iter().find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<TilemapRenderer>()).cloned() else { return };
        let xz = map.orientation == TilemapOrientation::XZ;
        c.need_material(&rend.material);
        c.register_materials(renderer, project_root);
        let (material, tint) = c.material_of(&rend.material);
        let serial = map.serial();
        let db_rev = tc.db.as_ref().map_or(0, |d| d.revision());
        let entry = tc.maps.entry(id).or_insert_with(|| MapCache { chunks: HashMap::new(), key: 0, resolved_for: (u64::MAX, 0) });
        if entry.resolved_for != (serial, db_rev) {
            tc.need.clear();
            let db = tc.db.as_mut().unwrap();
            for t in map.tile_table() {
                if let Some(tile) = db.get(&t.tile) {
                    for s in tile.sprites() {
                        if !s.is_empty() && !c.resolved_has(s) && !tc.need.iter().any(|n| n == s) {
                            tc.need.push(s.to_string());
                        }
                    }
                }
            }
            entry.resolved_for = (serial, db.revision());
            let mut need = std::mem::take(&mut tc.need);
            for text in need.drain(..) {
                let r = self.resolve_sprite(renderer, project_root, &text, false, c);
                if r.is_none() {
                    tracing::warn!("tile sprite '{text}' could not be resolved");
                }
                c.insert_resolved(text, r);
                entry.key = 0; // rebuild with the new sprites
            }
            tc.need = need;
        }
        // 2. Build / reuse chunks and copy the visible ones.
        let Some(map) = self.objects[i].components.iter().find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<Tilemap>()) else { return };
        let world = self.world_transform(i);
        let grid = grid_for(self, i);
        let layers = &self.layers.sorting;
        let env = MapEnv {
            world,
            grid,
            anchor: map.anchor,
            map_color: map.color,
            frame_rate: map.animation_frame_rate,
            time_offset: map.animation_time_offset,
            mode: rend.mode,
            sort_order: rend.sort_order,
            layer: layers.value_of(rend.sorting_layer),
            order: rend.order_in_layer,
            mask: match rend.mask_interaction {
                zestengine_core::MaskInteraction2D::None => MaskMode::None,
                zestengine_core::MaskInteraction2D::VisibleInsideMask => MaskMode::VisibleInside,
                zestengine_core::MaskInteraction2D::VisibleOutsideMask => MaskMode::VisibleOutside,
            },
            xz,
            material,
            tint,
        };
        // The cache key: everything but the cell contents that shapes a built chunk.
        let mut key = mix(0, db_rev);
        for v in world.to_cols_array() {
            key = mix(key, v.to_bits() as u64);
        }
        key = mix(key, (rend.mode as u64) << 8 | rend.sort_order as u64);
        key = mix(key, env.layer as u16 as u64 | (env.order as u16 as u64) << 16 | (env.mask as u64) << 32);
        for v in env.map_color.iter().chain(&map.anchor).chain(&[map.animation_frame_rate, map.animation_time_offset]) {
            key = mix(key, v.to_bits() as u64);
        }
        for v in env.grid.cell_size.iter().chain(&env.grid.cell_gap) {
            key = mix(key, v.to_bits() as u64);
        }
        key = mix(key, env.grid.layout as u64 | (env.grid.swizzle as u64) << 8);
        key = mix(key, c.resolved_len() as u64);
        key = mix(key, xz as u64 | (material as u64) << 8);
        for v in tint {
            key = mix(key, v.to_bits() as u64);
        }
        let entry = tc.maps.get_mut(&id).unwrap();
        if entry.key != key {
            entry.key = key;
            entry.chunks.clear();
        }
        let margin = rend.chunk_culling_margin.max(0.0) + env.grid.cell_size[0].max(env.grid.cell_size[1]);
        let db = tc.db.as_mut().unwrap();
        let (frame, mut visible) = (tc.frame, 0usize);
        tc.order.clear();
        let sort = rend.sort_order;
        for ck in map.chunk_keys() {
            let (lo, hi) = match entry.chunks.get(&(ck.x, ck.y)) {
                Some(cc) if cc.version == map.chunk(ck).map_or(0, |ch| ch.version) => (cc.lo, cc.hi),
                _ => chunk_cell_aabb(&env, ck, margin * 2.0 + 1.0),
            };
            tc.stats.chunks_total += 1;
            if aabb_outside_in(view_proj, lo, hi, if xz { world.w_axis.y } else { world.w_axis.z }, xz) {
                continue;
            }
            tc.order.push(ck);
        }
        tc.order.sort_by_key(|ck| sort.key(*ck));
        for &ck in &tc.order {
            let version = map.chunk(ck).map_or(0, |ch| ch.version);
            let stale = entry.chunks.get(&(ck.x, ck.y)).is_none_or(|cc| cc.version != version);
            if stale {
                let cc = entry.chunks.entry((ck.x, ck.y)).or_default();
                let resolved = |s: &str| c.tile_sprite(s);
                build_chunk(map, ck, &env, &resolved, &mut |p| db.get(p), &mut tc.anim, &mut tc.cells, cc);
                tc.stats.chunks_rebuilt += 1;
            }
            let cc = entry.chunks.get_mut(&(ck.x, ck.y)).unwrap();
            // The exact bounds may now say it is off screen.
            cc.used = frame;
            if aabb_outside_in(view_proj, cc.lo, cc.hi, if xz { world.w_axis.y } else { world.w_axis.z }, xz) {
                continue;
            }
            visible += 1;
            tc.stats.tiles += cc.sources.len() as u32;
            out.extend_from_slice(&cc.sources);
        }
        tc.stats.chunks_visible += visible as u32;
        // 3. Trim the cache: gone chunks always, old ones past the budget.
        entry.chunks.retain(|k, _| map.chunk(IVec2::new(k.0, k.1)).is_some());
        let budget = visible + rend.max_chunk_count as usize;
        if entry.chunks.len() > budget {
            let age = rend.max_frame_age;
            entry.chunks.retain(|_, cc| cc.used.wrapping_add(age) >= frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{TileAsset, TileFile};

    fn sprite_for(_: &str) -> Option<TileSprite> {
        Some(TileSprite { texture: 3, layer: 2, rect: [-0.5, -0.5, 1.0, 1.0], uv: [0.0, 0.0, 0.25, 0.25], rotated: false })
    }

    fn env() -> MapEnv {
        MapEnv {
            world: Mat4::IDENTITY,
            grid: Grid::default(),
            anchor: [0.5, 0.5],
            map_color: [1.0; 4],
            frame_rate: 1.0,
            time_offset: 0.0,
            mode: TilemapMode::Chunk,
            sort_order: TileSortOrder::BottomLeft,
            layer: 4,
            order: 7,
            mask: MaskMode::None,
            xz: false,
            material: 0,
            tint: [1.0; 4],
        }
    }

    fn db_with(tiles: &[&str]) -> TileDb {
        let mut db = TileDb::new(Path::new("."));
        for t in tiles {
            db.insert(t, TileFile::Tile(TileAsset::with_sprite("a.png#0000000000000001")));
        }
        db
    }

    fn build(map: &Tilemap, ck: IVec2, e: &MapEnv, anim: &mut AnimTable) -> CachedChunk {
        let mut db = db_with(&["g.tile", "w.tile"]);
        let mut cc = CachedChunk::default();
        build_chunk(map, ck, e, &sprite_for, &mut |p| db.get(p), anim, &mut Vec::new(), &mut cc);
        cc
    }

    /// Where the vertex shader puts a sprite-local point of `src` in plane coordinates (flip, scale,
    /// rotation, position), mirroring `base_instance` and `sprite.vert`.
    fn shade(src: &SpriteSource, local: [f32; 2]) -> [f32; 2] {
        let (mut lx, mut ly) = (local[0], local[1]);
        if src.flip[0] {
            lx = -lx;
        }
        if src.flip[1] {
            ly = -ly;
        }
        let (s, c) = src.rotation.sin_cos();
        let (x, y) = (lx * src.scale[0], ly * src.scale[1]);
        [c * x - s * y + src.position[0], s * x + c * y + src.position[1]]
    }

    fn xz_env(world: Mat4) -> MapEnv {
        MapEnv { world, xz: true, material: 5, tint: [1.0, 0.5, 0.25, 1.0], ..env() }
    }

    #[test]
    fn an_xz_map_builds_flat_ground_sources_with_the_depth_tested_material() {
        let mut map = Tilemap::default();
        map.orientation = TilemapOrientation::XZ;
        map.set_tile(IVec2::new(0, 0), Some("g.tile"));
        map.set_tile(IVec2::new(2, 1), Some("g.tile"));
        let mut anim = AnimTable::default();
        let c = build(&map, IVec2::ZERO, &xz_env(Mat4::from_translation(Vec3::new(10.0, 3.0, 20.0))), &mut anim);
        assert_eq!(c.sources.len(), 2);
        // Cell (x, y) at world (x, height, y) + the object's offset; the instance holds (x, z) and the height.
        assert_eq!(c.sources[0].position, [10.5, 20.5, 3.0]);
        assert_eq!(c.sources[1].position, [12.5, 21.5, 3.0]);
        for s in &c.sources {
            assert_eq!(s.material, 5 | zestengine_render::SPRITE_MATERIAL_XZ, "the XZ bit rides on the material id");
            assert_eq!(s.color, [1.0, 0.5, 0.25, 1.0], "the material's tint is multiplied into the tile");
        }
        // A tile's "up" points toward -Z and its x toward +X: the picture is upright seen from above.
        let up = shade(&c.sources[0], [0.0, 0.5]);
        let right = shade(&c.sources[0], [0.5, 0.0]);
        assert!((up[0] - 10.5).abs() < 1e-5 && (up[1] - 20.0).abs() < 1e-5, "up -> -z: {up:?}");
        assert!((right[0] - 11.0).abs() < 1e-5 && (right[1] - 20.5).abs() < 1e-5, "right -> +x: {right:?}");
        // The chunk bounds are plane coordinates around the cells.
        assert!(c.lo.x <= 10.0 && c.hi.x >= 13.0 && c.lo.y <= 20.0 && c.hi.y >= 22.0, "{:?} {:?}", c.lo, c.hi);
        // Turn the object 90 degrees about Y: +x becomes -z, so the whole picture turns with it.
        let yawed = build(&map, IVec2::ZERO, &xz_env(Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)), &mut anim);
        let s0 = &yawed.sources[0];
        assert!((s0.position[0] - 0.5).abs() < 1e-5 && (s0.position[1] + 0.5).abs() < 1e-5, "{:?}", s0.position);
        let right = shade(s0, [0.5, 0.0]);
        assert!((right[0] - 0.5).abs() < 1e-5 && (right[1] + 1.0).abs() < 1e-5, "x now runs along -z: {right:?}");
        let up = shade(s0, [0.0, 0.5]);
        assert!((up[0] - 0.0).abs() < 1e-5 && (up[1] + 0.5).abs() < 1e-5, "up turned to -x: {up:?}");
        // An XY map is untouched by all of this.
        let flat = build(&map, IVec2::ZERO, &env(), &mut anim);
        assert_eq!(flat.sources[0].material & zestengine_render::SPRITE_MATERIAL_XZ, 0);
        assert_eq!(flat.sources[0].position[..2], [0.5, 0.5]);
    }

    #[test]
    fn xz_frustum_culling_tests_the_ground_plane() {
        // Top-down camera 10 above the origin looking down; the ground box at height 0 near the middle is in view,
        // one far to the side is not.
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO, Vec3::NEG_Z);
        let vp = proj * view;
        assert!(!aabb_outside_in(&vp, Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), 0.0, true));
        assert!(aabb_outside_in(&vp, Vec2::new(40.0, -1.0), Vec2::new(42.0, 1.0), 0.0, true));
        assert!(aabb_outside_in(&vp, Vec2::new(-1.0, 40.0), Vec2::new(1.0, 42.0), 0.0, true));
    }

    #[test]
    fn chunk_gather_counts_and_order() {
        let mut map = Tilemap::default();
        for x in 0..16 {
            for y in 0..16 {
                map.set_tile(IVec2::new(x, y), Some("g.tile"));
            }
        }
        map.set_tile(IVec2::new(16, 0), Some("w.tile"));
        let mut anim = AnimTable::default();
        let a = build(&map, IVec2::new(0, 0), &env(), &mut anim);
        assert_eq!(a.sources.len(), 256);
        let b = build(&map, IVec2::new(1, 0), &env(), &mut anim);
        assert_eq!(b.sources.len(), 1);
        assert_eq!(build(&map, IVec2::new(5, 5), &env(), &mut anim).sources.len(), 0, "no chunk, nothing");
        // Bottom-left first: row-major from the low corner; the tile centred in its cell.
        assert_eq!(a.sources[0].position[..2], [0.5, 0.5]);
        assert_eq!(a.sources[1].position[..2], [1.5, 0.5]);
        assert_eq!(a.sources[16].position[..2], [0.5, 1.5]);
        assert!(a.sources.iter().all(|s| s.layer == 4 && s.order == 7 && s.draw == DrawMode::Quad && s.texture == 3 && s.atlas_layer == 2));
        // Top-right first reverses it.
        let mut e = env();
        e.sort_order = TileSortOrder::TopRight;
        let t = build(&map, IVec2::new(0, 0), &e, &mut anim);
        assert_eq!(t.sources[0].position[..2], [15.5, 15.5]);
        // Chunk mode: one sort point for the whole map; individual mode: per tile.
        assert!(a.sources.windows(2).all(|w| w[0].sort_point == w[1].sort_point));
        e.mode = TilemapMode::Individual;
        let i = build(&map, IVec2::new(0, 0), &e, &mut anim);
        assert_ne!(i.sources[0].sort_point, i.sources[1].sort_point);
        // Bounds cover the geometry.
        assert_eq!((a.lo, a.hi), (Vec2::ZERO, Vec2::splat(16.0)));
    }

    #[test]
    fn rotated_and_flipped_tiles_and_object_transforms() {
        let mut map = Tilemap::default();
        map.set_tile(IVec2::new(0, 0), Some("g.tile"));
        map.set_xform(IVec2::new(0, 0), TileXform::new(6, false));
        map.set_tile(IVec2::new(1, 0), Some("g.tile"));
        map.set_xform(IVec2::new(1, 0), TileXform::new(0, true));
        let mut anim = AnimTable::default();
        let c = build(&map, IVec2::ZERO, &env(), &mut anim);
        assert!((c.sources[0].rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-5 && !c.sources[0].flip[0]);
        assert!(c.sources[1].flip[0] && c.sources[1].rotation == 0.0);
        // The object's own rotation adds; a mirrored object flips the tile's sense.
        let mut e = env();
        e.world = Mat4::from_rotation_z(0.5) * Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let c = build(&map, IVec2::ZERO, &e, &mut anim);
        assert!((c.sources[0].rotation - (0.5 + std::f32::consts::FRAC_PI_2)).abs() < 1e-5);
        e.world = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let c = build(&map, IVec2::ZERO, &e, &mut anim);
        assert!(!c.sources[1].flip[0], "mirror on a mirrored object cancels");
        assert!(c.sources[0].flip[0], "a rotation-only tile under a mirrored object mirrors");
        // Rotated atlas sprites carry the flag into Quad mode.
        let mut e = env();
        e.world = Mat4::IDENTITY;
        let mut db = db_with(&["g.tile"]);
        let mut cc = CachedChunk::default();
        let rot = |_: &str| Some(TileSprite { texture: 1, layer: 0, rect: [0.0, 0.0, 1.0, 1.0], uv: [0.1, 0.1, 0.2, 0.3], rotated: true });
        build_chunk(&map, IVec2::ZERO, &e, &rot, &mut |p| db.get(p), &mut anim, &mut Vec::new(), &mut cc);
        assert!(cc.sources.iter().all(|s| s.rotated));
    }

    #[test]
    fn animated_tiles_share_table_entries_and_cost_no_cpu_per_frame() {
        let mut db = TileDb::new(Path::new("."));
        let anim_tile = TileAsset {
            animation: Some(zestengine_core::TileAnimation { sprites: (1..=4).map(|i| format!("a.png#{i:016x}")).collect(), min_speed: 2.0, max_speed: 2.0, start_time: 0.5, loop_once: false }),
            ..Default::default()
        };
        db.insert("w.tile", TileFile::Tile(anim_tile));
        let mut map = Tilemap::default();
        map.animation_frame_rate = 3.0;
        map.animation_time_offset = 1.0;
        for x in 0..10 {
            map.set_tile(IVec2::new(x, 0), Some("w.tile"));
        }
        let frames = |s: &str| {
            let n = u64::from_str_radix(s.rsplit('#').next().unwrap(), 16).unwrap() as f32;
            Some(TileSprite { texture: 1, layer: 0, rect: [0.0, 0.0, 1.0, 1.0], uv: [n * 0.1, 0.0, n * 0.1 + 0.1, 0.1], rotated: n as u32 == 3 })
        };
        let mut anim = AnimTable::default();
        let mut cc = CachedChunk::default();
        let mut e = env();
        e.frame_rate = map.animation_frame_rate;
        e.time_offset = map.animation_time_offset;
        build_chunk(&map, IVec2::ZERO, &e, &frames, &mut |p| db.get(p), &mut anim, &mut Vec::new(), &mut cc);
        assert_eq!(cc.sources.len(), 10);
        assert!(cc.sources.iter().all(|s| s.anim == cc.sources[0].anim && s.anim != 0));
        assert_eq!(anim.table.len(), 1 + 1 + 2 * 4, "dummy + header + 4 frames of (uv, rect)");
        assert!((cc.sources[0].anim_phase - 1.5).abs() < 1e-6);
        // fps = speed 2 x frame rate 3.
        let h = zestengine_render::sprite2d::AnimHeader::decode(anim.table[cc.sources[0].anim as usize]);
        assert_eq!((h.count, h.fps, h.rotated), (4, 6.0, 1 << 2));
        // The frame at t seconds is a pure function of the table (what the shader does).
        let pick = |t: f32| zestengine_render::sprite2d::anim_lookup(&anim.table, cc.sources[0].anim, t).unwrap();
        assert!((pick(0.0).0[0] - 0.1).abs() < 1e-6);
        assert!((pick(1.0 / 6.0 + 0.001).0[0] - 0.2).abs() < 1e-6);
        assert!(!pick(3.0 / 6.0 + 0.001).2 && pick(2.0 / 6.0 + 0.001).2, "frame 3 is the rotated one");
        assert!((pick(4.0 / 6.0 + 0.001).0[0] - 0.1).abs() < 1e-6, "loops");
        // A second map with the same animation adds nothing to the table.
        let before = anim.table.len();
        build_chunk(&map, IVec2::ZERO, &e, &frames, &mut |p| db.get(p), &mut anim, &mut Vec::new(), &mut cc);
        assert_eq!(anim.table.len(), before);
    }

    #[test]
    fn frustum_culling_is_conservative() {
        let proj = Mat4::orthographic_rh(-8.0, 8.0, -4.5, 4.5, -100.0, 100.0);
        let view = Mat4::from_translation(Vec3::new(-100.0, 0.0, 0.0)); // camera at x = +100... view moves the world by -100
        let vp = proj * Mat4::IDENTITY;
        assert!(!aabb_outside(&vp, Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), 0.0));
        assert!(!aabb_outside(&vp, Vec2::new(7.0, 0.0), Vec2::new(12.0, 2.0), 0.0), "straddling the edge is visible");
        assert!(aabb_outside(&vp, Vec2::new(9.0, 0.0), Vec2::new(12.0, 2.0), 0.0));
        assert!(aabb_outside(&vp, Vec2::new(-3.0, 5.0), Vec2::new(3.0, 9.0), 0.0));
        let moved = proj * view;
        assert!(aabb_outside(&moved, Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), 0.0));
        assert!(!aabb_outside(&moved, Vec2::new(99.0, -1.0), Vec2::new(101.0, 1.0), 0.0));
    }

    #[test]
    fn untouched_chunks_are_not_rebuilt_by_a_distant_edit() {
        let mut map = Tilemap::default();
        map.set_tile(IVec2::new(0, 0), Some("g.tile"));
        map.set_tile(IVec2::new(40, 40), Some("g.tile"));
        let v0 = map.chunk(IVec2::new(0, 0)).unwrap().version;
        let v1 = map.chunk(IVec2::new(2, 2)).unwrap().version;
        map.set_tile(IVec2::new(41, 40), Some("g.tile"));
        assert_eq!(map.chunk(IVec2::new(0, 0)).unwrap().version, v0, "far chunk keeps its cache");
        assert_ne!(map.chunk(IVec2::new(2, 2)).unwrap().version, v1);
    }

    #[test]
    fn rotated_atlas_uv_maths_matches_the_cooked_table() {
        use zestengine_assets::sprite::{PixelRect, SpriteDef, SpriteSheet};
        use zestengine_assets::sprite_atlas::{CookedSprite, atlas_quad};
        use zestengine_render::sprite2d::quad_uv;
        let page = [256u32, 128];
        for rotated in [false, true] {
            // A 20 x 10 sprite with a 2 px transparent border, placed at (40, 30).
            let trim = PixelRect::new(2, 2, 16, 6);
            let placed = if rotated { PixelRect::new(40, 30, 6, 16) } else { PixelRect::new(40, 30, 16, 6) };
            let cs = CookedSprite { page: 1, rect: placed, rotated, trim, size: [20, 10], ..Default::default() };
            let def = SpriteDef::named("s", PixelRect::new(0, 0, 20, 10));
            let sheet = SpriteSheet::default();
            let q = atlas_quad(&def, &sheet, &cs, page);
            assert_eq!((q.page, q.rotated), (1, rotated));
            // The trimmed quad's size in world units: 16 x 6 px at the default PPU.
            let ppu = def.effective_ppu(&sheet);
            assert!((q.rect[2] - 16.0 / ppu).abs() < 1e-5 && (q.rect[3] - 6.0 / ppu).abs() < 1e-5);
            // Every corner: the shader formula agrees with the packer's own mapping.
            for (a, b) in [(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.25)] {
                let px = [trim.x as f32 + a * trim.w as f32, trim.y as f32 + (1.0 - b) * trim.h as f32];
                let want = cs.atlas_uv(px, page);
                let got = quad_uv(q.uv_rect, [a, b], q.rotated);
                assert!((want[0] - got[0]).abs() < 1e-5 && (want[1] - got[1]).abs() < 1e-5, "rotated={rotated} corner ({a},{b}): {want:?} vs {got:?}");
            }
        }
    }
}
