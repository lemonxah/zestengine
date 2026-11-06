//! Tilemap colliders in the 2D world (2D-6 leftover). See `TilemapCollider2D`.
//!
//! # Strategy
//!
//! One compound collider per 16 x 16 chunk, on the static body that owns the
//! tilemap object, installed with [`PhysicsWorld2D::set_compound`]:
//!
//! - `Grid` tiles on a rectangle grid are merged into maximal rectangles by
//!   greedy meshing (`tile_collider::greedy_rects`), each a cuboid. With
//!   `outlines` (or a sibling Composite Collider 2D in Outlines mode) the solid
//!   cells become hollow boundary chains instead (`trace_outlines`), one
//!   polyline shape per chunk.
//! - `Grid` tiles on hexagon / isometric grids are one convex cell polygon each.
//! - `Sprite` tiles use the sprite's physics outline (convex pieces cached per
//!   sprite), placed at the cell's tile pivot with the cell's transform.
//!
//! Editing a cell bumps its chunk's version (and a border neighbour's, since
//! it can matter for rule tiles); [`PhysicsWorld2D::sync_tilemaps`] compares
//! versions and regenerates only those chunks, replacing only their compounds.
//! A chunk that became empty removes its compound. With `used_by_composite`
//! (a sibling Composite Collider 2D) all chunk shapes are merged into the one
//! compound the composite owns (key `u64::MAX` of the tilemap); only the
//! changed chunks are regenerated, the concatenation is redone.

use std::collections::HashMap;

use glam::{IVec2, Vec2};
use zestengine_core::{CompositeCollider2D, CompositeGeometry2D, Grid, ObjectId, TileColliderType, Tilemap, TilemapCollider2D, TileXform};

use super::*;
use crate::tile_collider::{self, CellRect, N};
use crate::tile_db::{TileDb, grid_for, resolve_cell};

/// Per tilemap: what was built and for which versions.
#[derive(Default)]
pub struct TileColliderState {
    pub owner: Option<ObjectId>,
    serial: u64,
    settings: u64,
    db_rev: u64,
    built: HashMap<(i32, i32), u32>,
    parts: HashMap<(i32, i32), Vec<(Pose, SharedShape)>>,
}

/// Counters of the last tilemap sync (tests, perf report).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct TileColliderStats {
    pub chunks_rebuilt: u32,
    pub chunks_removed: u32,
}

fn chunk_key(tile_obj: ObjectId, ck: (i32, i32)) -> u64 {
    let mut h = tile_obj.raw() as u64 ^ 0x9E37_79B9_7F4A_7C15;
    h = (h ^ ((ck.0 as u32 as u64) << 32 | ck.1 as u32 as u64)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^ (h >> 31)
}

fn settings_hash(tc: &TilemapCollider2D, comp: Option<CompositeGeometry2D>, scale: Vec2, grid: &Grid) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x100_0000_01B3);
    mix(tc.extrusion_factor.to_bits() as u64);
    mix(tc.outlines as u64 | (tc.merge_cells as u64) << 1 | (tc.is_trigger as u64) << 2 | (tc.used_by_composite as u64) << 3);
    for b in tc.material.bytes() {
        mix(b as u64);
    }
    mix(scale.x.to_bits() as u64 ^ (scale.y.to_bits() as u64) << 32);
    mix(comp.map_or(9, |g| g as u64));
    for v in grid.cell_size.iter().chain(&grid.cell_gap) {
        mix(v.to_bits() as u64);
    }
    mix(grid.layout as u64 | (grid.swizzle as u64) << 8);
    h
}

impl PhysicsWorld2D {
    /// The tile database the colliders read (created on first use under the
    /// world's project root).
    pub fn tile_db_mut(&mut self) -> &mut TileDb {
        let root = self.root.clone().unwrap_or_default();
        self.tile_db.get_or_insert_with(|| TileDb::new(&root))
    }

    /// Provide a sprite's physics outline (world units, pivot relative)
    /// without reading its texture (tests, generated sprites).
    pub fn set_sprite_physics(&mut self, sprite: &str, paths: &[Vec<[f32; 2]>]) {
        self.sprite_pieces.insert(sprite.to_string(), polygon_pieces(paths));
    }

    /// Chunk compounds rebuilt / removed by the last [`Self::sync_tilemaps`].
    pub fn tile_stats(&self) -> TileColliderStats {
        self.tile_stats
    }

    fn sprite_pieces_of(&mut self, sprite: &str) -> Vec<Vec<Vec2>> {
        if let Some(p) = self.sprite_pieces.get(sprite) {
            return p.clone();
        }
        let mut pieces = Vec::new();
        if let (Some(root), Some(r)) = (self.root.clone(), zestengine_assets::sprite::SpriteRef::from_ref_string(sprite))
            && let Some((table, idx)) = zestengine_assets::sprite::resolve_sprite_ref(&root, &r)
            && let Some(geom) = table.geometry(idx)
        {
            pieces = polygon_pieces(&polygon_from_sprite(&geom).paths);
        }
        self.sprite_pieces.insert(sprite.to_string(), pieces.clone());
        pieces
    }

    /// Regenerate the chunk compounds of every `Tilemap Collider 2D` whose map
    /// changed since the last call. Cheap when nothing changed (one serial
    /// compare per tilemap).
    pub fn sync_tilemaps(&mut self, scene: &LoadedScene) {
        self.tile_stats = TileColliderStats::default();
        for k in 0..self.tile_objs.len() {
            let (tid, owner) = self.tile_objs[k];
            self.sync_one_tilemap(scene, tid, owner);
        }
    }

    fn sync_one_tilemap(&mut self, scene: &LoadedScene, tid: ObjectId, owner: ObjectId) {
        let Some(ti) = scene.index_of(tid) else { return };
        let Some(oi) = scene.index_of(owner) else { return };
        let o = &scene.objects[ti];
        let Some(map) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<Tilemap>()) else { return };
        let Some(tc) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<TilemapCollider2D>()) else { return };
        let comp = o.components.iter().find_map(|c| c.as_any().downcast_ref::<CompositeCollider2D>());
        let tc = tc.clone();
        let comp = comp.map(|c| c.geometry);
        let grid = grid_for(scene, ti);
        let world = (scene.world_transform(ti), scene.world_transform(oi));
        let (pos, angle, scale) = decompose(world.0);
        let (bpos, bangle, _) = decompose(world.1);
        let rel = pose2(bpos, bangle).inverse() * pose2(pos, angle);
        let composite = tc.used_by_composite && comp.is_some();
        let outlines = tc.outlines || (composite && comp == Some(CompositeGeometry2D::Outlines));
        let settings = settings_hash(&tc, comp, scale, &grid);
        let db_rev = self.tile_db_mut().revision();
        let mut st = self.tile_state.remove(&tid).unwrap_or_default();
        if st.owner != Some(owner) || st.settings != settings || st.db_rev != db_rev {
            // Everything is stale (settings, owner or a tile file changed).
            for ck in st.built.keys().copied().collect::<Vec<_>>() {
                self.set_compound(owner, chunk_key(tid, ck), None, &CompoundProps::default(), scene);
            }
            if composite || st.owner == Some(owner) {
                self.set_compound(owner, chunk_key(tid, (i32::MAX, i32::MAX)), None, &CompoundProps::default(), scene);
            }
            st = TileColliderState { owner: Some(owner), settings, db_rev, ..Default::default() };
            st.serial = u64::MAX;
        }
        if st.serial == map.serial() {
            self.tile_state.insert(tid, st);
            return;
        }
        let props = CompoundProps { is_trigger: tc.is_trigger, material: tc.material.clone() };
        // Chunks that disappeared.
        let gone: Vec<(i32, i32)> = st.built.keys().copied().filter(|&(x, y)| map.chunk(IVec2::new(x, y)).is_none()).collect();
        let mut any_change = false;
        for ck in gone {
            st.built.remove(&ck);
            st.parts.remove(&ck);
            if !composite {
                self.set_compound(owner, chunk_key(tid, ck), None, &props, scene);
            }
            self.tile_stats.chunks_removed += 1;
            any_change = true;
        }
        // Chunks whose version moved.
        for ck in map.chunk_keys() {
            let version = map.chunk(ck).map_or(0, |c| c.version);
            if st.built.get(&(ck.x, ck.y)) == Some(&version) {
                continue;
            }
            let parts = self.chunk_parts(map, ck, &grid, &tc, outlines, scale, rel);
            st.built.insert((ck.x, ck.y), version);
            any_change = true;
            self.tile_stats.chunks_rebuilt += 1;
            if composite {
                st.parts.insert((ck.x, ck.y), parts);
            } else {
                self.set_compound(owner, chunk_key(tid, (ck.x, ck.y)), compound_shape(parts), &props, scene);
            }
        }
        if composite && any_change {
            let mut keys: Vec<_> = st.parts.keys().copied().collect();
            keys.sort();
            let all: Vec<(Pose, SharedShape)> = keys.iter().flat_map(|k| st.parts[k].iter().cloned()).collect();
            self.set_compound(owner, chunk_key(tid, (i32::MAX, i32::MAX)), compound_shape(all), &props, scene);
        }
        st.serial = map.serial();
        self.tile_state.insert(tid, st);
    }

    #[allow(clippy::too_many_arguments)]
    fn chunk_parts(&mut self, map: &Tilemap, ck: IVec2, grid: &Grid, tc: &TilemapCollider2D, outlines: bool, scale: Vec2, rel: Pose) -> Vec<(Pose, SharedShape)> {
        let chunk = match map.chunk(ck) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let base = ck * zestengine_core::CHUNK_SIZE;
        let rect_grid = matches!(grid.layout, zestengine_core::CellLayout::Rectangle) && grid.swizzle == zestengine_core::CellSwizzle::XYZ;
        let mut mask = [false; N * N];
        let mut parts: Vec<(Pose, SharedShape)> = Vec::new();
        let ext = tc.extrusion_factor.max(0.0);
        // Resolve the tiles first (needs the db), then build.
        struct Sprited {
            cell: IVec2,
            sprite: String,
            xform: TileXform,
        }
        let mut sprited: Vec<Sprited> = Vec::new();
        let mut grid_cells: Vec<IVec2> = Vec::new();
        {
            let db = self.tile_db_mut();
            for (i, &t) in chunk.tiles.iter().enumerate() {
                if t == 0 {
                    continue;
                }
                let cell = zestengine_core::cell_at(ck, i);
                let Some(path) = map.tile_path(t) else { continue };
                let Some(tile) = db.get(path) else { continue };
                let r = resolve_cell(map, &tile, path, cell);
                match r.collider {
                    TileColliderType::None => {}
                    TileColliderType::Grid => {
                        if rect_grid {
                            mask[i] = true;
                        } else {
                            grid_cells.push(cell);
                        }
                    }
                    TileColliderType::Sprite => {
                        if let Some(s) = r.sel.sprites.first().filter(|s| !s.is_empty()) {
                            sprited.push(Sprited { cell, sprite: s.clone(), xform: r.xform });
                        }
                    }
                }
            }
        }
        let stride = Vec2::new(grid.cell_size[0] + grid.cell_gap[0], grid.cell_size[1] + grid.cell_gap[1]);
        let size = Vec2::new(grid.cell_size[0], grid.cell_size[1]);
        let origin = |c: IVec2| -> Vec2 {
            let p = grid.cell_to_local(c.extend(0));
            Vec2::new(p.x, p.y)
        };
        // Grid cells on a rectangle grid.
        if outlines {
            let rings = tile_collider::trace_outlines(&mask);
            if !rings.is_empty() {
                let mut verts: Vec<Vector> = Vec::new();
                let mut idx: Vec<[u32; 2]> = Vec::new();
                for ring in rings {
                    let b = verts.len() as u32;
                    let n = ring.len() as u32;
                    for p in &ring {
                        let w = origin(base + *p) * scale;
                        verts.push(rel.transform_point(rv(w)));
                    }
                    idx.extend((0..n).map(|i| [b + i, b + (i + 1) % n]));
                }
                parts.push((Pose::IDENTITY, SharedShape::polyline(verts, Some(idx))));
            }
        } else {
            let rects = if tc.merge_cells { tile_collider::greedy_rects(&mask) } else { tile_collider::single_rects(&mask) };
            for CellRect { x, y, w, h } in rects {
                let min = origin(base + IVec2::new(x, y));
                let sz = Vec2::new(w as f32 * stride.x - grid.cell_gap[0], h as f32 * stride.y - grid.cell_gap[1]);
                let centre = (min + sz * 0.5) * scale;
                let half = (sz * 0.5 * scale).abs() + Vec2::splat(ext);
                parts.push((rel * Pose::from_translation(rv(centre)), SharedShape::cuboid(half.x.max(1e-3), half.y.max(1e-3))));
            }
        }
        // Grid cells on other layouts: the cell polygon.
        for cell in grid_cells {
            let pts: Vec<Vector> = grid.cell_corners(cell.extend(0)).iter().map(|p| rv(Vec2::new(p.x, p.y) * scale)).collect();
            if let Some(s) = SharedShape::convex_hull(&pts) {
                parts.push((rel, s));
            }
        }
        // Sprite outlines.
        for s in sprited {
            let pieces = self.sprite_pieces_of(&s.sprite);
            let pivot = grid.tile_pivot(s.cell.extend(0), map.anchor);
            let pivot = Vec2::new(pivot.x, pivot.y);
            for piece in pieces {
                let mut pts: Vec<Vec2> = piece.iter().map(|p| (pivot + s.xform.apply(*p)) * scale).collect();
                if s.xform.mirror {
                    pts.reverse();
                }
                if ext > 0.0 {
                    let c = pts.iter().copied().sum::<Vec2>() / pts.len() as f32;
                    for p in &mut pts {
                        *p += (*p - c).normalize_or_zero() * ext;
                    }
                }
                let v: Vec<Vector> = pts.iter().map(|p| rv(*p)).collect();
                if let Some(sh) = SharedShape::convex_hull(&v) {
                    parts.push((rel, sh));
                }
            }
        }
        let _ = size;
        parts
    }
}
