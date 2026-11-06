//! Baked occlusion culling (potentially visible sets).
//!
//! The bake splits the static geometry's bounds into view cells and, for every
//! cell, finds the static objects visible from anywhere inside it: rays are shot
//! in all directions from sample points in the cell (first hit marks an object
//! visible), then every object not yet found is tested with rays aimed at
//! points on its surface. The result is one bitset per cell, saved to the
//! `<scene>.occlusion` sidecar. At runtime the culling camera's cell selects
//! the bitset; static objects outside it skip the main camera pass. Dynamic
//! objects are never occlusion culled.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use zestengine_core::ObjectId;
use glam::Vec3;

use crate::scene::LoadedScene;

const MAGIC: &[u8; 8] = b"CITROCC1";
/// Upper bound on view cells; the cell size grows to stay under it.
pub const MAX_CELLS: usize = 32_768;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionSettings {
    /// View-cell edge length (m).
    pub cell_size: f32,
    /// Rays shot per sample point in the omnidirectional pass.
    pub rays: u32,
}

impl Default for OcclusionSettings {
    fn default() -> Self {
        Self { cell_size: 4.0, rays: 1024 }
    }
}

/// Baked visibility: per-cell bitsets over the static objects.
#[derive(Clone, Debug)]
pub struct OcclusionData {
    pub origin: Vec3,
    pub cell_size: f32,
    pub dims: [u32; 3],
    pub objects: Vec<ObjectId>,
    words: usize,
    bits: Vec<u64>,
    slot_of: HashMap<ObjectId, u32>,
}

impl OcclusionData {
    fn new(origin: Vec3, cell_size: f32, dims: [u32; 3], objects: Vec<ObjectId>, bits: Vec<u64>) -> Self {
        let words = objects.len().div_ceil(64).max(1);
        let slot_of = objects.iter().enumerate().map(|(i, id)| (*id, i as u32)).collect();
        Self { origin, cell_size, dims, objects, words, bits, slot_of }
    }

    pub fn cell_count(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }

    pub fn byte_size(&self) -> usize {
        self.bits.len() * 8 + self.objects.len() * 16
    }

    /// The cell containing `p`, or None outside the baked bounds.
    pub fn cell_of(&self, p: Vec3) -> Option<usize> {
        let c = ((p - self.origin) / self.cell_size).floor();
        if c.min_element() < 0.0 {
            return None;
        }
        let (x, y, z) = (c.x as u32, c.y as u32, c.z as u32);
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return None;
        }
        Some(((z * self.dims[1] + y) * self.dims[0] + x) as usize)
    }

    pub fn cell_coords(&self, cell: usize) -> [u32; 3] {
        let x = cell as u32 % self.dims[0];
        let y = (cell as u32 / self.dims[0]) % self.dims[1];
        let z = cell as u32 / (self.dims[0] * self.dims[1]);
        [x, y, z]
    }

    /// World AABB of a cell.
    pub fn cell_bounds(&self, cell: usize) -> (Vec3, Vec3) {
        let [x, y, z] = self.cell_coords(cell);
        let mn = self.origin + Vec3::new(x as f32, y as f32, z as f32) * self.cell_size;
        (mn, mn + Vec3::splat(self.cell_size))
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let ext = Vec3::new(self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32) * self.cell_size;
        (self.origin, self.origin + ext)
    }

    /// Whether `id` is visible from `cell`. Objects the bake didn't know about
    /// (added since) are always visible.
    pub fn visible(&self, cell: usize, id: ObjectId) -> bool {
        match self.slot_of.get(&id) {
            Some(&s) => self.bits[cell * self.words + (s as usize >> 6)] & (1u64 << (s & 63)) != 0,
            None => true,
        }
    }

    /// Number of baked objects visible from `cell`.
    pub fn visible_count(&self, cell: usize) -> usize {
        self.bits[cell * self.words..(cell + 1) * self.words].iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.byte_size());
        out.extend_from_slice(MAGIC);
        for v in self.origin.to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.cell_size.to_le_bytes());
        for d in self.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(self.objects.len() as u32).to_le_bytes());
        for id in &self.objects {
            out.extend_from_slice(&id.raw().to_le_bytes());
        }
        for w in &self.bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader { b, at: 0 };
        if r.take(8)? != MAGIC {
            return None;
        }
        let origin = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
        let cell_size = r.f32()?;
        let dims = [r.u32()?, r.u32()?, r.u32()?];
        let n = r.u32()? as usize;
        let mut objects = Vec::with_capacity(n);
        for _ in 0..n {
            objects.push(ObjectId::from_raw(u128::from_le_bytes(r.take(16)?.try_into().ok()?)));
        }
        let cells: usize = dims.iter().map(|d| *d as usize).product();
        let words = n.div_ceil(64).max(1);
        let mut bits = Vec::with_capacity(cells * words);
        for _ in 0..cells * words {
            bits.push(u64::from_le_bytes(r.take(8)?.try_into().ok()?));
        }
        if cell_size <= 0.0 || cells == 0 {
            return None;
        }
        Some(Self::new(origin, cell_size, dims, objects, bits))
    }

    /// Load `<base>.occlusion` if present and valid.
    pub fn read_sidecar(base: &Path) -> Option<Self> {
        let bytes = std::fs::read(base.with_extension("occlusion")).ok()?;
        let data = Self::from_bytes(&bytes);
        if data.is_none() {
            tracing::warn!("ignoring malformed occlusion data next to {}", base.display());
        }
        data
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n)?;
        self.at += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

/// Everything the bake needs, gathered from the scene on the main thread.
pub struct BakeInput {
    pub tris: Vec<[Vec3; 3]>,
    /// Object slot of each triangle.
    pub tri_obj: Vec<u32>,
    pub objects: Vec<ObjectId>,
    pub obj_bounds: Vec<(Vec3, Vec3)>,
    pub settings: OcclusionSettings,
}

/// Static, active, rendered objects and their world triangles. Shadows Only
/// objects are never seen, so they neither occlude nor get culled.
pub fn bake_input(scene: &LoadedScene, settings: OcclusionSettings) -> BakeInput {
    let world = scene.world_transforms();
    let mut input = BakeInput { tris: Vec::new(), tri_obj: Vec::new(), objects: Vec::new(), obj_bounds: Vec::new(), settings };
    for (i, o) in scene.objects.iter().enumerate() {
        if !o.static_geometry || !o.has_active_renderer() || !scene.is_active(i) || !o.cast_shadows().visible() {
            continue;
        }
        let slot = input.objects.len() as u32;
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for render in o.render_slots() {
            let Some((pos, idx)) = scene.mesh_geometry(render.mesh) else {
                continue;
            };
            for t in idx.as_chunks::<3>().0 {
                let tri = [0, 1, 2].map(|k| world[i].transform_point3(pos[t[k] as usize]));
                for p in tri {
                    mn = mn.min(p);
                    mx = mx.max(p);
                }
                input.tris.push(tri);
                input.tri_obj.push(slot);
            }
        }
        if mn.x <= mx.x {
            input.objects.push(o.id);
            input.obj_bounds.push((mn, mx));
        }
    }
    input
}

// ---------------------------------------------------------------------------
// BVH over the bake triangles.

struct Node {
    mn: Vec3,
    mx: Vec3,
    /// Leaf: first triangle in `order`; inner: index of the left child (right = left + 1).
    start: u32,
    /// Leaf triangle count; 0 for inner nodes.
    count: u32,
}

struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
}

impl Bvh {
    fn build(tris: &[[Vec3; 3]]) -> Self {
        let centroids: Vec<Vec3> = tris.iter().map(|t| (t[0] + t[1] + t[2]) / 3.0).collect();
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = vec![Node { mn: Vec3::ZERO, mx: Vec3::ZERO, start: 0, count: tris.len() as u32 }];
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let (start, count) = (nodes[n].start as usize, nodes[n].count as usize);
            let mut mn = Vec3::splat(f32::MAX);
            let mut mx = Vec3::splat(f32::MIN);
            let mut cmn = Vec3::splat(f32::MAX);
            let mut cmx = Vec3::splat(f32::MIN);
            for &t in &order[start..start + count] {
                for p in tris[t as usize] {
                    mn = mn.min(p);
                    mx = mx.max(p);
                }
                cmn = cmn.min(centroids[t as usize]);
                cmx = cmx.max(centroids[t as usize]);
            }
            nodes[n].mn = mn;
            nodes[n].mx = mx;
            if count <= 4 {
                continue;
            }
            let ext = cmx - cmn;
            let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
            let slice = &mut order[start..start + count];
            let mid = count / 2;
            slice.select_nth_unstable_by(mid, |a, b| {
                centroids[*a as usize][axis].total_cmp(&centroids[*b as usize][axis])
            });
            let left = nodes.len();
            nodes.push(Node { mn: Vec3::ZERO, mx: Vec3::ZERO, start: start as u32, count: mid as u32 });
            nodes.push(Node { mn: Vec3::ZERO, mx: Vec3::ZERO, start: (start + mid) as u32, count: (count - mid) as u32 });
            nodes[n].start = left as u32;
            nodes[n].count = 0;
            stack.push(left);
            stack.push(left + 1);
        }
        Self { nodes, order }
    }

    /// Closest hit along `o + d*t`, t in (0, t_max): (t, triangle).
    fn closest(&self, tris: &[[Vec3; 3]], o: Vec3, d: Vec3, t_max: f32) -> Option<(f32, u32)> {
        if self.order.is_empty() {
            return None;
        }
        let inv = d.recip();
        let mut best = t_max;
        let mut hit = None;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !slab(node.mn, node.mx, o, inv, best) {
                continue;
            }
            if node.count > 0 {
                for &t in &self.order[node.start as usize..(node.start + node.count) as usize] {
                    if let Some(tt) = ray_tri(o, d, &tris[t as usize])
                        && tt < best
                    {
                        best = tt;
                        hit = Some((tt, t));
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.start;
                stack[sp + 1] = node.start + 1;
                sp += 2;
            }
        }
        hit
    }
}

fn slab(mn: Vec3, mx: Vec3, o: Vec3, inv: Vec3, t_max: f32) -> bool {
    let t1 = (mn - o) * inv;
    let t2 = (mx - o) * inv;
    let tmin = t1.min(t2).max_element().max(0.0);
    let tmax = t1.max(t2).min_element().min(t_max);
    tmin <= tmax
}

/// Two-sided Moller-Trumbore.
fn ray_tri(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> Option<f32> {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - t[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let tt = e2.dot(q) * inv;
    (tt > 1e-4).then_some(tt)
}

// ---------------------------------------------------------------------------
// Bake.

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}

fn rand01(seed: &mut u32) -> f32 {
    *seed = hash(*seed);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

/// `n` near-uniform directions (Fibonacci sphere) rotated by a random offset.
fn sphere_dirs(n: u32, seed: &mut u32) -> Vec<Vec3> {
    let rot = glam::Quat::from_euler(glam::EulerRot::XYZ, rand01(seed) * std::f32::consts::TAU, rand01(seed) * std::f32::consts::TAU, rand01(seed) * std::f32::consts::TAU);
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let a = golden * i as f32;
            rot * Vec3::new(a.cos() * r, y, a.sin() * r)
        })
        .collect()
}

/// Area-weighted points on each object's surface, nudged off it along the
/// face normal, plus its bounds centre: the targets of the aimed pass.
fn surface_targets(input: &BakeInput, per_object: usize) -> Vec<Vec<Vec3>> {
    let mut tris_of: Vec<Vec<u32>> = vec![Vec::new(); input.objects.len()];
    for (t, &o) in input.tri_obj.iter().enumerate() {
        tris_of[o as usize].push(t as u32);
    }
    let mut seed = 0x51ed_270bu32;
    tris_of
        .iter()
        .enumerate()
        .map(|(o, ts)| {
            let (mn, mx) = input.obj_bounds[o];
            let mut pts = vec![(mn + mx) * 0.5];
            if ts.is_empty() {
                return pts;
            }
            let areas: Vec<f32> = ts
                .iter()
                .map(|&t| {
                    let tri = &input.tris[t as usize];
                    (tri[1] - tri[0]).cross(tri[2] - tri[0]).length() * 0.5
                })
                .collect();
            let total: f32 = areas.iter().sum();
            if total <= 0.0 {
                return pts;
            }
            let nudge = ((mx - mn).length() * 0.002).clamp(0.002, 0.05);
            for _ in 0..per_object {
                let mut pick = rand01(&mut seed) * total;
                let mut k = 0;
                while k + 1 < areas.len() && pick > areas[k] {
                    pick -= areas[k];
                    k += 1;
                }
                let tri = &input.tris[ts[k] as usize];
                let (mut u, mut v) = (rand01(&mut seed), rand01(&mut seed));
                if u + v > 1.0 {
                    u = 1.0 - u;
                    v = 1.0 - v;
                }
                let p = tri[0] + (tri[1] - tri[0]) * u + (tri[2] - tri[0]) * v;
                let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
                // Both sides: a target behind the face from one viewpoint is in
                // front of it from another.
                pts.push(p + n * nudge);
                pts.push(p - n * nudge);
            }
            pts
        })
        .collect()
}

/// Grid covering the static bounds (padded by half a cell), with the cell size
/// raised until the cell count fits [`MAX_CELLS`].
pub fn grid_for(input: &BakeInput) -> Option<(Vec3, f32, [u32; 3])> {
    let mut mn = Vec3::splat(f32::MAX);
    let mut mx = Vec3::splat(f32::MIN);
    for (a, b) in &input.obj_bounds {
        mn = mn.min(*a);
        mx = mx.max(*b);
    }
    if mn.x > mx.x {
        return None;
    }
    let mut cell = input.settings.cell_size.max(0.25);
    loop {
        let lo = mn - Vec3::splat(cell * 0.5);
        let ext = (mx + Vec3::splat(cell * 0.5)) - lo;
        let dims = (ext / cell).ceil().max(Vec3::ONE);
        let dims = [dims.x as u32, dims.y as u32, dims.z as u32];
        if dims.iter().map(|d| *d as usize).product::<usize>() <= MAX_CELLS {
            return Some((lo, cell, dims));
        }
        cell *= 1.25;
    }
}

/// Run the bake. `progress(done, total)` is called as cells finish; returns
/// None when cancelled.
pub fn bake(
    input: &BakeInput,
    cancel: &AtomicBool,
    progress: impl Fn(usize, usize) + Sync,
) -> Option<OcclusionData> {
    let (origin, cell_size, dims) = grid_for(input)?;
    let cells: usize = dims.iter().map(|d| *d as usize).product();
    let n_obj = input.objects.len();
    let words = n_obj.div_ceil(64).max(1);
    let bvh = Bvh::build(&input.tris);
    let targets = surface_targets(input, 12);
    let grid = OcclusionData::new(origin, cell_size, dims, input.objects.clone(), Vec::new());
    let rows: Vec<std::sync::Mutex<Vec<u64>>> = (0..cells).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let threads = crate::sw_gi::bake_worker_count().min(cells).max(1);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                crate::sw_gi::lower_compute_priority();
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= cells {
                        break;
                    }
                    let row = bake_cell(input, &bvh, &targets, &grid, c, words);
                    *rows[c].lock().unwrap() = row;
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1, cells);
                }
            });
        }
    });
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let mut bits = Vec::with_capacity(cells * words);
    for r in rows {
        bits.extend(r.into_inner().unwrap());
    }
    Some(OcclusionData::new(origin, cell_size, dims, input.objects.clone(), bits))
}

fn bake_cell(
    input: &BakeInput,
    bvh: &Bvh,
    targets: &[Vec<Vec3>],
    grid: &OcclusionData,
    cell: usize,
    words: usize,
) -> Vec<u64> {
    let n_obj = input.objects.len();
    let mut row = vec![0u64; words];
    let set = |row: &mut Vec<u64>, s: usize| row[s >> 6] |= 1u64 << (s & 63);
    let has = |row: &Vec<u64>, s: usize| row[s >> 6] & (1u64 << (s & 63)) != 0;
    let (mn, mx) = grid.cell_bounds(cell);
    let size = mx - mn;

    // Objects overlapping the cell are always visible from it.
    let pad = size * 0.05;
    for (s, (a, b)) in input.obj_bounds.iter().enumerate() {
        if a.cmple(mx + pad).all() && b.cmpge(mn - pad).all() {
            set(&mut row, s);
        }
    }

    // Sample points: centre, inset corners and face centres. Points inside
    // solid geometry (most axis rays hit back faces) are dropped; a cell with
    // no usable point keeps everything visible.
    let mut seed = hash(cell as u32 ^ 0x9e37_79b9);
    let mut points = vec![(mn + mx) * 0.5];
    for k in 0..8 {
        let f = Vec3::new((k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32);
        points.push(mn + size * (Vec3::splat(0.08) + f * 0.84));
    }
    for axis in 0..3 {
        for side in [0.08f32, 0.92] {
            let mut f = Vec3::splat(0.5);
            f[axis] = side;
            points.push(mn + size * f);
        }
    }
    let axes = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
    points.retain(|&p| {
        let back = axes
            .iter()
            .filter(|&&d| {
                bvh.closest(&input.tris, p, d, f32::MAX).is_some_and(|(_, t)| {
                    let tri = &input.tris[t as usize];
                    (tri[1] - tri[0]).cross(tri[2] - tri[0]).dot(d) > 0.0
                })
            })
            .count();
        back < 4
    });
    if points.is_empty() {
        return vec![u64::MAX; words];
    }

    // Omnidirectional pass: the first hit of every ray is visible.
    for &p in &points {
        for d in sphere_dirs(input.settings.rays, &mut seed) {
            if let Some((_, t)) = bvh.closest(&input.tris, p, d, f32::MAX) {
                set(&mut row, input.tri_obj[t as usize] as usize);
            }
        }
    }

    // Aimed pass: rays at surface points of every object still hidden. A ray
    // that reaches the target unblocked, or whose first hit is the target
    // object itself, makes it visible.
    for (s, target) in targets.iter().enumerate().take(n_obj) {
        if has(&row, s) {
            continue;
        }
        'search: for &p in &points {
            for &q in target {
                let v = q - p;
                let dist = v.length();
                if dist < 1e-4 {
                    set(&mut row, s);
                    break 'search;
                }
                let d = v / dist;
                match bvh.closest(&input.tris, p, d, dist) {
                    None => {
                        set(&mut row, s);
                        break 'search;
                    }
                    Some((_, t)) if input.tri_obj[t as usize] as usize == s => {
                        set(&mut row, s);
                        break 'search;
                    }
                    _ => {}
                }
            }
        }
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(tris: &mut Vec<[Vec3; 3]>, obj: &mut Vec<u32>, slot: u32, c: Vec3, u: Vec3, v: Vec3) {
        let (a, b, cc, d) = (c - u - v, c + u - v, c + u + v, c - u + v);
        tris.push([a, b, cc]);
        tris.push([a, cc, d]);
        obj.extend([slot, slot]);
    }

    fn bounds(tris: &[[Vec3; 3]], obj: &[u32], n: usize) -> Vec<(Vec3, Vec3)> {
        let mut out = vec![(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)); n];
        for (t, &o) in tris.iter().zip(obj) {
            for p in t {
                out[o as usize].0 = out[o as usize].0.min(*p);
                out[o as usize].1 = out[o as usize].1.max(*p);
            }
        }
        out
    }

    /// A wall between two small boxes: each box is hidden from the cells on
    /// the far side of the wall, and visible from its own side.
    #[test]
    fn wall_hides_far_side() {
        let mut tris = Vec::new();
        let mut obj = Vec::new();
        // Slot 0: a large wall at x = 0 spanning y,z in [-20, 20].
        quad(&mut tris, &mut obj, 0, Vec3::ZERO, Vec3::new(0.0, 20.0, 0.0), Vec3::new(0.0, 0.0, 20.0));
        // Slot 1: a small plate at x = -8; slot 2: a small plate at x = +8.
        quad(&mut tris, &mut obj, 1, Vec3::new(-8.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.0, 0.0, 0.5));
        quad(&mut tris, &mut obj, 2, Vec3::new(8.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.0, 0.0, 0.5));
        let obj_bounds = bounds(&tris, &obj, 3);
        let ids: Vec<ObjectId> = (1..=3u128).map(ObjectId::from_raw).collect();
        let input = BakeInput {
            tris,
            tri_obj: obj,
            objects: ids.clone(),
            obj_bounds,
            settings: OcclusionSettings { cell_size: 4.0, rays: 256 },
        };
        let data = bake(&input, &AtomicBool::new(false), |_, _| {}).unwrap();
        let left = data.cell_of(Vec3::new(-6.0, 0.5, 0.5)).unwrap();
        let right = data.cell_of(Vec3::new(6.0, 0.5, 0.5)).unwrap();
        assert!(data.visible(left, ids[1]));
        assert!(!data.visible(left, ids[2]));
        assert!(data.visible(right, ids[2]));
        assert!(!data.visible(right, ids[1]));
        assert!(data.visible(left, ids[0]) && data.visible(right, ids[0]));
        // Unknown objects are never culled.
        assert!(data.visible(left, ObjectId::from_raw(99)));
        // Round trip.
        let back = OcclusionData::from_bytes(&data.to_bytes()).unwrap();
        assert_eq!(back.dims, data.dims);
        assert_eq!(back.visible(left, ids[2]), data.visible(left, ids[2]));
        assert_eq!(back.visible(right, ids[2]), data.visible(right, ids[2]));
    }
}
