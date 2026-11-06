//! Walls the GPU-simulated water reflects off: every static object (its
//! colliders, or its render mesh when it has none) and every still collider
//! through the surface, in the editor and in Play alike. Moving bodies are
//! not walls: they push the water as displacers instead, and a collider
//! being moved counts as moving until it has stood still a moment. Only
//! objects whose layer collides with the water's count (the Layers matrix).
//!
//! Closed shapes mark the cells whose centres at the surface lie inside them;
//! triangle meshes (which need not be closed: a pool's single-sided walls)
//! mark the cells along the line where their steep faces cross the surface.
//! The same static geometry is the water's floor: under each cell, the top
//! of what lies below the surface (a pool's basin, terrain, a sloped beach)
//! sets its depth, and ground at or above the surface is dry shore, so waves
//! slow and shorten over shallows as they do in real water. Shapes are cached
//! per object until it moves or its colliders change, and a water's cells
//! are only rebuilt when its window or the geometry in it change.
//!
//! Every water with a `shore_depth` (lakes, pools, oceans; rivers calm at
//! their banks already) gets the same floor baked over its whole area into
//! a `ShoreDepth` grid, rebaked when the water or the geometry under it
//! changes: its waves calm over the shallows near a shore.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use zestengine_core::{
    BodyKind, BoxCollider, CapsuleCollider, CharacterController, MeshCollider, ObjectId, RigidBody, ShoreDepth, SimWindow, SphereCollider, Water,
    WaterBody,
};
use glam::{IVec2, Mat4, Vec2, Vec3};

use crate::physics::{Solid, solid_below, solid_contains, world_solids};
use crate::scene::LoadedScene;

/// Updates a collider counts as moving for after it last moved.
const SETTLE_UPDATES: u32 = 10;
/// Faces steeper than this (|normal.y| below it) are walls; flatter ones
/// are floor.
const STEEP: f32 = 0.6;
/// Ground up to this far above the surface (m) still counts as the floor
/// there (a beach up to `RUNUP_HEIGHT`, a bank above); higher is an
/// overhang the water ignores.
const RISE: f32 = zestengine_core::RUNUP_HEIGHT + 0.1;
/// Cells per side of a shore-depth bake.
const SHORE_CELLS: u32 = 128;
/// Farthest (m) an endless water's shore bake reaches from its origin.
const SHORE_REACH: f32 = 1024.0;
/// Ground up to this far above the surface (m) is dry shore to a shore
/// bake (so a beach reads as shore all the way up); higher is an overhang.
const SHORE_RISE: f32 = 10.0;

struct Cached {
    key: u64,
    solids: Arc<Vec<Solid>>,
    /// World bounds of the solids.
    lo: Vec3,
    hi: Vec3,
}

#[derive(Default)]
pub struct WaterWalls {
    shapes: HashMap<ObjectId, Cached>,
    /// Each object's last world transform and updates left until it counts
    /// as still.
    moved: HashMap<ObjectId, ([f32; 16], u32)>,
    /// Each water's last wall set.
    masks: HashMap<ObjectId, u64>,
    /// Each water's last shore bake (what it was baked from).
    shores: HashMap<ObjectId, u64>,
}

impl WaterWalls {
        pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Refresh the wall cells of every GPU-simulated water that has a
    /// window from what stands in it now, and rebake the shore depth of
    /// every water whose shore calms its waves when it or its ground changed.
    pub fn update(&mut self, scene: &mut LoadedScene, world: &[Mat4]) {
        let active = scene.active_flags();
        let waters: Vec<usize> = (0..scene.objects.len())
            .filter(|&i| active[i])
            .filter(|&i| water_of(scene, i).is_some_and(|w| w.interactive && w.sim_feed.window.is_some() && w.ripples.as_ref().is_some_and(|r| r.gpu)))
            .collect();
        let shores: Vec<usize> = (0..scene.objects.len())
            .filter(|&i| active[i])
            .filter(|&i| water_of(scene, i).is_some_and(|w| w.shore_depth > 0.0 && w.body != WaterBody::River))
            .collect();
        if waters.is_empty() && shores.is_empty() {
            return;
        }
        let mut walls = Vec::new();
        let mut seen = HashMap::new();
        for (i, o) in scene.objects.iter().enumerate() {
            if !active[i] || moves(o) || !(has_collider(o) || (o.static_geometry && o.render.is_some())) {
                continue;
            }
            let m = world[i].to_cols_array();
            let settle = match self.moved.get(&o.id) {
                Some((last, left)) if *last == m => left.saturating_sub(1),
                Some(_) => SETTLE_UPDATES,
                None => 0,
            };
            seen.insert(o.id, (m, settle));
            if settle > 0 {
                continue;
            }
            let key = shape_key(o, &m);
            if self.shapes.get(&o.id).is_none_or(|c| c.key != key) {
                let solids = world_solids(scene, i, world[i]);
                let (lo, hi) = bounds(&solids);
                self.shapes.insert(o.id, Cached { key, solids: Arc::new(solids), lo, hi });
            }
            walls.push(i);
        }
        self.moved = seen;
        self.shapes.retain(|id, _| self.moved.contains_key(id));
        for wi in waters {
            let Some(water) = water_of(scene, wi) else { continue };
            let Some(window) = water.sim_feed.window else { continue };
            let water_world = world[wi];
            let to_local = water_world.inverse();
            let (wlo, whi) = window_bounds(window, water_world, (water.depth * 3.0).max(3.0));
            let mut here: Vec<(ObjectId, u64)> = walls
                .iter()
                .filter(|&&i| i != wi && crate::physics::layers_meet(scene, wi, i))
                .filter_map(|&i| {
                    let id = scene.objects[i].id;
                    let c = self.shapes.get(&id)?;
                    let overlaps = c.hi.x >= wlo.x && c.lo.x <= whi.x && c.hi.z >= wlo.z && c.lo.z <= whi.z && c.hi.y >= wlo.y && c.lo.y <= whi.y;
                    overlaps.then_some((id, c.key))
                })
                .collect();
            here.sort_by_key(|(id, _)| id.raw());
            let mut h = DefaultHasher::new();
            (window.n, window.cell.to_bits(), window.origin.x, window.origin.y).hash(&mut h);
            here.hash(&mut h);
            let key = h.finish();
            if self.masks.get(&scene.objects[wi].id) == Some(&key) && water.sim_feed.blocked.as_ref().is_some_and(|(w, _)| *w == window) {
                continue;
            }
            let cells = (window.n * window.n) as usize;
            let mut mask = vec![false; cells];
            let mut ground = vec![f32::NEG_INFINITY; cells];
            for (id, _) in &here {
                if let Some(c) = self.shapes.get(id) {
                    mark(&mut mask, &mut ground, window, water, &to_local, water_world, &c.solids, RISE);
                }
            }
            let bed: Vec<f32> = ground
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let xz = cell_centre(window, i);
                    if y.is_finite() { water.column(xz).map_or(0.0, |c| c.surface) - y } else { f32::NAN }
                })
                .collect();
            let id = scene.objects[wi].id;
            self.masks.insert(id, key);
            if let Some(w) = water_mut(scene, wi) {
                w.sim_feed.blocked = Some((window, Arc::new(mask)));
                w.sim_feed.bed = Some((window, Arc::new(bed)));
            }
        }
        for wi in shores {
            self.update_shore(scene, world, &walls, wi);
        }
    }

    /// Rebake water `wi`'s shore depth when it or the static geometry
    /// under it (`walls`, object indices) changed.
    fn update_shore(&mut self, scene: &mut LoadedScene, world: &[Mat4], walls: &[usize], wi: usize) {
        let Some(water) = water_of(scene, wi) else { return };
        let water_world = world[wi];
        let to_local = water_world.inverse();
        let below = (water.depth * 3.0).max(3.0);
        let mine: Vec<(ObjectId, &Cached)> = walls
            .iter()
            .filter(|&&i| i != wi && crate::physics::layers_meet(scene, wi, i))
            .filter_map(|&i| self.shapes.get(&scene.objects[i].id).map(|c| (scene.objects[i].id, c)))
            .collect();
        let overlapping = |window: SimWindow| {
            let (lo, hi) = window_bounds(window, water_world, below);
            mine.iter()
                .filter(|(_, c)| c.hi.x >= lo.x && c.lo.x <= hi.x && c.hi.z >= lo.z && c.lo.z <= hi.z && c.hi.y >= lo.y && c.lo.y <= hi.y)
                .collect::<Vec<_>>()
        };
        let half = match water.extent() {
            Some((lo, hi)) => lo.abs().max(hi.abs()).max_element(),
            // Endless: as far as the geometry near its surface reaches.
            None => overlapping(shore_window(SHORE_REACH))
                .iter()
                .flat_map(|(_, c)| {
                    (0..8).map(|k| {
                        let p = to_local.transform_point3(Vec3::new(
                            if k & 1 == 0 { c.lo.x } else { c.hi.x },
                            if k & 2 == 0 { c.lo.y } else { c.hi.y },
                            if k & 4 == 0 { c.lo.z } else { c.hi.z },
                        ));
                        p.x.abs().max(p.z.abs())
                    })
                })
                .fold(0.0f32, f32::max)
                .min(SHORE_REACH),
        };
        let id = scene.objects[wi].id;
        if half <= 1e-3 {
            self.shores.remove(&id);
            if let Some(w) = water_mut(scene, wi) {
                w.shore = None;
            }
            return;
        }
        let window = shore_window(half);
        let mut here: Vec<(ObjectId, &Cached)> = overlapping(window).into_iter().copied().collect();
        here.sort_by_key(|(id, _)| id.raw());
        let mut h = DefaultHasher::new();
        half.to_bits().hash(&mut h);
        water_world.to_cols_array().map(f32::to_bits).hash(&mut h);
        (format!("{:?}", water.body), water.depth.to_bits(), water.size.map(f32::to_bits)).hash(&mut h);
        water.outline().iter().flatten().for_each(|v| v.to_bits().hash(&mut h));
        here.iter().for_each(|(id, c)| (id.raw(), c.key).hash(&mut h));
        let key = h.finish();
        if self.shores.get(&id) == Some(&key) {
            return;
        }
        let solids: Vec<&[Solid]> = here.iter().map(|(_, c)| c.solids.as_slice()).collect();
        let bake = bake_shore(water, water_world, half, &solids);
        self.shores.insert(id, key);
        if let Some(w) = water_mut(scene, wi) {
            w.shore = bake.map(Arc::new);
        }
    }
}

/// The square window a shore bake samples: `SHORE_CELLS` across local XZ
/// -`half`..`half`.
fn shore_window(half: f32) -> SimWindow {
    let n = SHORE_CELLS;
    SimWindow { origin: IVec2::splat(-(n as i32) / 2), n, cell: 2.0 * half / n as f32, closed: true }
}

/// The depth of `water` (posed at `water_world`) over the top of `solids`
/// across local XZ -`half`..`half`; None when nothing lies under it.
fn bake_shore(water: &Water, water_world: Mat4, half: f32, solids: &[&[Solid]]) -> Option<ShoreDepth> {
    let window = shore_window(half);
    let to_local = water_world.inverse();
    let cells = (window.n * window.n) as usize;
    let mut mask = vec![false; cells];
    let mut ground = vec![f32::NEG_INFINITY; cells];
    for s in solids {
        mark(&mut mask, &mut ground, window, water, &to_local, water_world, s, SHORE_RISE);
    }
    if ground.iter().all(|y| !y.is_finite()) {
        return None;
    }
    let depth = ground
        .iter()
        .enumerate()
        .map(|(i, &y)| if y.is_finite() { water.column(cell_centre(window, i)).map_or(0.0, |c| c.surface) - y } else { f32::NAN })
        .collect();
    Some(ShoreDepth::new(window.n as usize, half, depth))
}

/// Directions the outline is sampled in when fitting a water to its basin.
const FIT_RAYS: usize = 96;
/// Farthest a fitted outline reaches from the water's origin (m).
const FIT_REACH: f32 = 200.0;
/// Outline points closer than this (m) to the line through their
/// neighbours are dropped.
const FIT_TOLERANCE: f32 = 0.04;

/// The outline and depth of the basin around water `wi`'s origin: the static
/// geometry (on layers the water meets) sliced at its surface, seen from the
/// origin in every direction, and the floor under the origin. None when
/// nothing surrounds it.
pub fn fit_basin(scene: &LoadedScene, wi: usize, world: &[Mat4]) -> Option<(Vec<[f32; 2]>, Option<f32>)> {
    let water = water_of(scene, wi)?;
    let to_local = world[wi].inverse();
    let level = water.column(Vec2::ZERO).map_or(0.0, |c| c.surface);
    let mut segments: Vec<(Vec2, Vec2)> = Vec::new();
    let mut shapes = Vec::new();
    let mut floor = f32::NEG_INFINITY;
    for (i, o) in scene.objects.iter().enumerate() {
        if i == wi || !scene.is_active(i) || moves(o) || !(has_collider(o) || (o.static_geometry && o.render.is_some())) {
            continue;
        }
        if !crate::physics::layers_meet(scene, wi, i) {
            continue;
        }
        for solid in world_solids(scene, i, world[i]) {
            match solid {
                Solid::Mesh(verts, tris) => {
                    let local: Vec<Vec3> = verts.iter().map(|v| to_local.transform_point3(*v)).collect();
                    for t in &tris {
                        let p = [local[t[0] as usize], local[t[1] as usize], local[t[2] as usize]];
                        let d = p.map(|v| v.y - level);
                        let mut ends = Vec::with_capacity(2);
                        for k in 0..3 {
                            let (a, b) = (k, (k + 1) % 3);
                            if (d[a] <= 0.0) != (d[b] <= 0.0) {
                                let q = p[a].lerp(p[b], d[a] / (d[a] - d[b]));
                                ends.push(Vec2::new(q.x, q.z));
                            }
                        }
                        if let [a, b] = ends[..] {
                            segments.push((a, b));
                        }
                        // The floor right under the origin.
                        let q = p.map(|v| Vec2::new(v.x, v.z));
                        let area = (q[1] - q[0]).perp_dot(q[2] - q[0]);
                        if area.abs() > 1e-9 {
                            let w0 = (q[1]).perp_dot(q[2]) / area;
                            let w1 = (q[2]).perp_dot(q[0]) / area;
                            let w2 = 1.0 - w0 - w1;
                            let y = p[0].y * w0 + p[1].y * w1 + p[2].y * w2;
                            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 && y < level {
                                floor = floor.max(y);
                            }
                        }
                    }
                }
                Solid::Shape(col) => shapes.push(col),
            }
        }
    }
    let origin = world[wi].transform_point3(Vec3::new(0.0, level, 0.0));
    for col in &shapes {
        if let Some(down) = solid_below(col, origin, FIT_REACH)
            && down > 0.0
        {
            floor = floor.max(to_local.transform_point3(origin - Vec3::Y * down).y);
        }
    }
    let mut outline = Vec::new();
    for k in 0..FIT_RAYS {
        let a = k as f32 / FIT_RAYS as f32 * std::f32::consts::TAU;
        let dir = Vec2::new(a.cos(), a.sin());
        let mut best = FIT_REACH;
        for (p, q) in &segments {
            // Ray from the origin against the segment p..q.
            let e = *q - *p;
            let den = dir.perp_dot(e);
            if den.abs() < 1e-9 {
                continue;
            }
            let t = p.perp_dot(e) / den;
            let u = p.perp_dot(dir) / den;
            if t > 1e-3 && (0.0..=1.0).contains(&u) {
                best = best.min(t);
            }
        }
        for col in &shapes {
            let from = world[wi].transform_point3(Vec3::new(0.0, level, 0.0));
            let along = world[wi].transform_vector3(Vec3::new(dir.x, 0.0, dir.y)).normalize_or_zero();
            if let Some(t) = crate::physics::solid_along(col, from, along, FIT_REACH)
                && t > 1e-3
            {
                best = best.min(t);
            }
        }
        if best < FIT_REACH {
            outline.push(dir * best);
        }
    }
    if outline.len() < 3 {
        return None;
    }
    // Drop points on (nearly) straight runs.
    loop {
        let n = outline.len();
        let mut drop = None;
        for k in 0..n {
            let (a, b, c) = (outline[(k + n - 1) % n], outline[k], outline[(k + 1) % n]);
            let e = c - a;
            let off = if e.length() > 1e-6 { (b - a).perp_dot(e).abs() / e.length() } else { (b - a).length() };
            if off < FIT_TOLERANCE {
                drop = Some(k);
                break;
            }
        }
        match drop {
            Some(k) if n > 3 => {
                outline.remove(k);
            }
            _ => break,
        }
    }
    let depth = floor.is_finite().then_some(level - floor).filter(|d| *d > 0.0);
    Some((outline.into_iter().map(|p| p.to_array()).collect(), depth))
}

/// Centre (water-local XZ) of window cell `i` (row-major).
fn cell_centre(window: SimWindow, i: usize) -> Vec2 {
    let n = window.n as usize;
    window.lo() + (Vec2::new((i % n) as f32, (i / n) as f32) + 0.5) * window.cell
}

/// Mark the window's cells `solids` fill at the water's surface in `mask`,
/// and raise `ground` (water-local height per cell) to the top of what lies
/// under each cell (ground up to `rise` above the surface counts).
#[allow(clippy::too_many_arguments)]
fn mark(mask: &mut [bool], ground: &mut [f32], window: SimWindow, water: &Water, to_local: &Mat4, water_world: Mat4, solids: &[Solid], rise: f32) {
    let n = window.n as i32;
    let lo = window.lo();
    let cell = window.cell;
    let surface = |xz: Vec2| water.column(xz).map_or(0.0, |c| c.surface);
    let reach = rise + (water.depth * 3.0).max(3.0);
    // Cells (x0..x1, z0..z1) whose centres the local XZ box a..b covers.
    let span = |a: Vec2, b: Vec2| {
        let x0 = (((a.x - lo.x) / cell - 0.5).ceil() as i32).max(0);
        let x1 = (((b.x - lo.x) / cell - 0.5).floor() as i32 + 1).min(n);
        let z0 = (((a.y - lo.y) / cell - 0.5).ceil() as i32).max(0);
        let z1 = (((b.y - lo.y) / cell - 0.5).floor() as i32 + 1).min(n);
        (x0, x1, z0, z1)
    };
    for s in solids {
        match s {
            Solid::Shape(col) => {
                let aabb = col.compute_aabb();
                let (a, b) = (Vec3::new(aabb.mins.x, aabb.mins.y, aabb.mins.z), Vec3::new(aabb.maxs.x, aabb.maxs.y, aabb.maxs.z));
                let corners: [Vec3; 8] = std::array::from_fn(|k| {
                    to_local.transform_point3(Vec3::new(
                        if k & 1 == 0 { a.x } else { b.x },
                        if k & 2 == 0 { a.y } else { b.y },
                        if k & 4 == 0 { a.z } else { b.z },
                    ))
                });
                let (fl, fh) = corners.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(p, q), c| (p.min(*c), q.max(*c)));
                let (x0, x1, z0, z1) = span(Vec2::new(fl.x, fl.z) - cell * 0.5, Vec2::new(fh.x, fh.z) + cell * 0.5);
                for z in z0..z1 {
                    for x in x0..x1 {
                        let i = (z * n + x) as usize;
                        let xz = lo + (Vec2::new(x as f32, z as f32) + 0.5) * cell;
                        let level = surface(xz);
                        let in_water = solid_contains(col, water_world.transform_point3(Vec3::new(xz.x, level, xz.y)));
                        if in_water {
                            mask[i] = true;
                        }
                        let from = water_world.transform_point3(Vec3::new(xz.x, level + rise, xz.y));
                        if let Some(down) = solid_below(col, from, reach) {
                            let y = to_local.transform_point3(from - Vec3::Y * down).y;
                            // Its top above the level without standing in
                            // the water there is an overhang (a tilted box's
                            // upper corner), not a beach: the water under it
                            // is free. Ground only where it is in the water.
                            if y <= level || in_water {
                                ground[i] = ground[i].max(y);
                            }
                        }
                    }
                }
            }
            Solid::Mesh(verts, tris) => {
                let local: Vec<Vec3> = verts.iter().map(|v| to_local.transform_point3(*v)).collect();
                for t in tris {
                    let p = [local[t[0] as usize], local[t[1] as usize], local[t[2] as usize]];
                    let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                    let mid = (p[0] + p[1] + p[2]) / 3.0;
                    let level = surface(Vec2::new(mid.x, mid.z));
                    if normal.y.abs() < STEEP {
                        // A wall: the line where it crosses the level.
                        let d = p.map(|v| v.y - level);
                        let mut ends = Vec::with_capacity(2);
                        for k in 0..3 {
                            let (a, b) = (k, (k + 1) % 3);
                            if (d[a] <= 0.0) != (d[b] <= 0.0) {
                                let f = d[a] / (d[a] - d[b]);
                                let q = p[a].lerp(p[b], f);
                                ends.push(Vec2::new(q.x, q.z));
                            }
                        }
                        if let [a, b] = ends[..] {
                            let steps = ((b - a).length() / (cell * 0.5)).ceil().max(1.0) as usize;
                            for k in 0..=steps {
                                let c = ((a.lerp(b, k as f32 / steps as f32) - lo) / cell).floor().as_ivec2();
                                if c.x >= 0 && c.y >= 0 && c.x < n && c.y < n {
                                    mask[(c.y * n + c.x) as usize] = true;
                                }
                            }
                        }
                        continue;
                    }
                    // Floor: its height under each cell centre it covers.
                    if p.iter().all(|v| v.y > level + rise) {
                        continue;
                    }
                    let q = p.map(|v| Vec2::new(v.x, v.z));
                    let (a, b) = (q[0].min(q[1]).min(q[2]), q[0].max(q[1]).max(q[2]));
                    let (x0, x1, z0, z1) = span(a, b);
                    let area = (q[1] - q[0]).perp_dot(q[2] - q[0]);
                    if area.abs() < 1e-9 {
                        continue;
                    }
                    for z in z0..z1 {
                        for x in x0..x1 {
                            let c = lo + (Vec2::new(x as f32, z as f32) + 0.5) * cell;
                            let w0 = (q[1] - c).perp_dot(q[2] - c) / area;
                            let w1 = (q[2] - c).perp_dot(q[0] - c) / area;
                            let w2 = 1.0 - w0 - w1;
                            if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                                continue;
                            }
                            let y = p[0].y * w0 + p[1].y * w1 + p[2].y * w2;
                            if y <= surface(c) + rise {
                                let i = (z * n + x) as usize;
                                ground[i] = ground[i].max(y);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn bounds(solids: &[Solid]) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for s in solids {
        match s {
            Solid::Shape(c) => {
                let a = c.compute_aabb();
                lo = lo.min(Vec3::new(a.mins.x, a.mins.y, a.mins.z));
                hi = hi.max(Vec3::new(a.maxs.x, a.maxs.y, a.maxs.z));
            }
            Solid::Mesh(v, _) => {
                for p in v {
                    lo = lo.min(*p);
                    hi = hi.max(*p);
                }
            }
        }
    }
    (lo, hi)
}

/// World bounds of a water's window, from `below` under its surface to a
/// little above it.
fn window_bounds(window: SimWindow, water_world: Mat4, below: f32) -> (Vec3, Vec3) {
    let (a, b) = (window.lo(), window.lo() + Vec2::splat(window.size()));
    let corners: [Vec3; 8] = std::array::from_fn(|k| {
        water_world.transform_point3(Vec3::new(if k & 1 == 0 { a.x } else { b.x }, if k & 2 == 0 { -below } else { 2.0 }, if k & 4 == 0 { a.y } else { b.y }))
    });
    corners.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(p, q), c| (p.min(*c), q.max(*c)))
}

/// Changes when the object moves or its colliders or mesh change.
fn shape_key(o: &crate::scene::SceneObject, world: &[f32; 16]) -> u64 {
    let mut h = DefaultHasher::new();
    for v in world {
        v.to_bits().hash(&mut h);
    }
    o.static_geometry.hash(&mut h);
    o.render.map(|r| r.mesh).hash(&mut h);
    for c in &o.components {
        let a = c.as_any();
        if let Some(b) = a.downcast_ref::<BoxCollider>() {
            (0u8, b.is_trigger).hash(&mut h);
            b.center.iter().chain(&b.size).for_each(|v| v.to_bits().hash(&mut h));
        } else if let Some(s) = a.downcast_ref::<SphereCollider>() {
            (1u8, s.is_trigger, s.radius.to_bits()).hash(&mut h);
            s.center.iter().for_each(|v| v.to_bits().hash(&mut h));
        } else if let Some(cap) = a.downcast_ref::<CapsuleCollider>() {
            (2u8, cap.is_trigger, cap.radius.to_bits(), cap.height.to_bits(), cap.direction as u8).hash(&mut h);
            cap.center.iter().for_each(|v| v.to_bits().hash(&mut h));
        } else if let Some(m) = a.downcast_ref::<MeshCollider>() {
            (3u8, m.is_trigger, m.convex, m.follow_shape, m.detail as u8).hash(&mut h);
        }
    }
    h.finish()
}

fn has_collider(o: &crate::scene::SceneObject) -> bool {
    o.components.iter().any(|c| {
        let a = c.as_any();
        a.is::<BoxCollider>() || a.is::<SphereCollider>() || a.is::<CapsuleCollider>() || a.is::<MeshCollider>()
    })
}

/// Bodies that move the water instead: non-fixed Rigid Bodies, characters.
fn moves(o: &crate::scene::SceneObject) -> bool {
    o.components.iter().any(|c| {
        let a = c.as_any();
        a.is::<CharacterController>() || a.downcast_ref::<RigidBody>().is_some_and(|r| r.kind != BodyKind::Fixed)
    })
}

fn water_of(scene: &LoadedScene, i: usize) -> Option<&Water> {
    scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Water>())
}

fn water_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut Water> {
    scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Water>())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 20 m lake over a floor 3 m down for x < 0 that slopes up 0.4 m/m
    /// toward a shore (dry from x = 7.5).
    fn beach() -> (Water, Vec<Solid>) {
        let water = Water { size: [20.0, 20.0], ..Default::default() };
        let v = |x: f32, z: f32| Vec3::new(x, -3.0 + 0.4 * x.max(0.0), z);
        let verts = vec![v(-12.0, -12.0), v(0.0, -12.0), v(0.0, 12.0), v(-12.0, 12.0), v(12.0, -12.0), v(12.0, 12.0)];
        let tris = vec![[0, 2, 1], [0, 3, 2], [1, 5, 4], [1, 2, 5]];
        (water, vec![Solid::Mesh(verts, tris)])
    }

    #[test]
    fn shore_bake_shallows_toward_the_shore_and_calms_the_waves() {
        let (mut water, solids) = beach();
        let bake = bake_shore(&water, Mat4::IDENTITY, 10.0, &[&solids]).expect("ground under the lake");
        assert_eq!((bake.n, bake.half), (SHORE_CELLS as usize, 10.0));
        let at = |x: f32| bake.depth_at(Vec2::new(x, 1.3)).unwrap();
        assert!((at(-5.0) - 3.0).abs() < 0.05, "flat floor 3 m down: {}", at(-5.0));
        for x in [1.0, 3.0, 5.0, 7.0] {
            assert!((at(x) - (3.0 - 0.4 * x)).abs() < 0.05, "slope at {x}: {}", at(x));
        }
        let mut last = f32::MAX;
        for k in 0..=90 {
            let d = at(-5.0 + k as f32 * 0.15);
            assert!(d <= last + 1e-4, "depth only falls toward the shore");
            last = d;
        }
        assert_eq!(at(9.0), 0.0, "the dry beach is 0 deep");

        // Damping: smoothstep(0, shore_depth, depth).
        water.shore = Some(Arc::new(bake));
        water.shore_depth = 2.0;
        let damp = |x: f32| water.shore_damping(Vec2::new(x, 1.3));
        assert_eq!(damp(-5.0), 1.0, "deeper than shore_depth: full waves");
        let d = 3.0 - 0.4 * 4.0;
        let t: f32 = d / 2.0;
        assert!((damp(4.0) - t * t * (3.0 - 2.0 * t)).abs() < 0.03, "{}", damp(4.0));
        assert!(damp(7.4) < 0.01 && damp(9.0) == 0.0, "calm at the waterline");
        assert!(damp(2.0) > damp(4.0) && damp(4.0) > damp(6.0));
        let open = Water { size: [20.0, 20.0], ..Default::default() };
        let xz = Vec2::new(6.0, 1.3);
        let ((d_shore, _), (d_open, _)) = (water.displacement(xz, 0.7), open.displacement(xz, 0.7));
        assert!((d_shore - d_open * damp(6.0)).length() < 1e-5, "the waves scale by it");
        water.shore_depth = 0.0;
        assert_eq!(water.shore_damping(Vec2::new(9.0, 0.0)), 1.0, "0 turns it off");
    }

    #[test]
    fn water_with_nothing_under_it_keeps_full_waves() {
        let (water, _) = beach();
        assert!(bake_shore(&water, Mat4::IDENTITY, 10.0, &[]).is_none());
        // A floor only under half of it: open water elsewhere.
        let half_floor = vec![Solid::Mesh(
            vec![Vec3::new(-12.0, -0.5, -12.0), Vec3::new(-2.0, -0.5, -12.0), Vec3::new(-2.0, -0.5, 12.0), Vec3::new(-12.0, -0.5, 12.0)],
            vec![[0, 2, 1], [0, 3, 2]],
        )];
        let bake = bake_shore(&water, Mat4::IDENTITY, 10.0, &[&half_floor]).unwrap();
        assert!((bake.depth_at(Vec2::new(-6.0, 0.0)).unwrap() - 0.5).abs() < 1e-3);
        assert_eq!(bake.depth_at(Vec2::new(6.0, 0.0)), Some(zestengine_core::SHORE_NO_GROUND));
        assert_eq!(zestengine_core::shore_damping(zestengine_core::SHORE_NO_GROUND, 2.0), 1.0);
    }
}
