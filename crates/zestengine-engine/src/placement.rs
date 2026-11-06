//! Asset placement: dragging a model / prefab / audio clip out of the Files panel
//! into the scene. The logic here is pure (no egui, no GPU) so it can be unit
//! tested; `placement_glue.rs` (a child of `editor_app`) drives it from the UI.
//!
//! * Snap rays: ray vs triangle / AABB / oriented box / sphere / capsule, a
//!   per-mesh BVH (built off-thread for big meshes), nearest-hit selection.
//! * Placement math: bounds offset along a surface normal, align-to-normal that
//!   keeps the yaw, yaw stepping, grid snapping.
//! * The hierarchy drop rule (world origin, parent compensation).
//! * The viewport placement state machine (enter / follow / leave / commit /
//!   cancel, async model loads arriving in any order).
//!
//! See `docs/design/asset-placement.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use glam::{Mat4, Quat, Vec3};
use zestengine_core::ObjectId;

use crate::scene::LoadedScene;

// ---------------------------------------------------------------------------
// Asset kinds
// ---------------------------------------------------------------------------

/// What a dragged file spawns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    /// `.fbx .glb .gltf .vrm .obj`: parsed on a worker, uploaded on arrival.
    Model,
    /// `.prefab`: a linked instance (synchronous).
    Prefab,
    /// `.wav .ogg .mp3 .flac`: an Audio Source object playing the clip.
    Audio,
}

/// The spawnable kind of a file, from its extension.
pub fn asset_kind(path: &Path) -> Option<AssetKind> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "fbx" | "glb" | "gltf" | "vrm" | "obj" => Some(AssetKind::Model),
        "prefab" => Some(AssetKind::Prefab),
        "wav" | "ogg" | "mp3" | "flac" => Some(AssetKind::Audio),
        _ => None,
    }
}

/// What a file dropped on the Scene hierarchy does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HierarchyDrop {
    /// Spawn at world (0,0,0); `parent` is the row it was dropped on.
    Spawn(AssetKind),
    /// A `.material` on a row: assign it to that object (the existing meaning).
    AssignMaterial,
    /// Anything else keeps its existing behaviour (nothing happens here).
    Ignore,
}

/// Decide a hierarchy drop. `on_row` is true when it landed on an object row.
pub fn classify_hierarchy_drop(path: &Path, on_row: bool) -> HierarchyDrop {
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    if ext.as_deref() == Some("material") {
        return if on_row { HierarchyDrop::AssignMaterial } else { HierarchyDrop::Ignore };
    }
    match asset_kind(path) {
        Some(kind) => HierarchyDrop::Spawn(kind),
        None => HierarchyDrop::Ignore,
    }
}

/// Local transform that puts an object at WORLD (0,0,0), identity rotation and scale 1
/// under a parent with world matrix `parent_world` (None = root level). Shear (a rotated
/// parent with non-uniform scale) cannot be undone exactly; the closest TRS is returned.
pub fn local_for_world_origin(parent_world: Option<Mat4>) -> (Vec3, Quat, Vec3) {
    let Some(pw) = parent_world else {
        return (Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
    };
    if pw.determinant().abs() < 1e-12 {
        return (Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
    }
    let (s, r, t) = pw.inverse().to_scale_rotation_translation();
    (t, r, s)
}

// ---------------------------------------------------------------------------
// Rays
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    /// Unit length.
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir: dir.normalize_or(Vec3::NEG_Z) }
    }
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// A ray hit. `normal` is unit length and faces the ray origin.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    /// Index of the scene object that was hit (0 for bare primitive tests).
    pub object: usize,
}

fn face(n: Vec3, dir: Vec3) -> Vec3 {
    if n.dot(dir) > 0.0 { -n } else { n }
}

/// Moller-Trumbore, two sided. Returns (t, normal facing the ray).
pub fn ray_triangle(ray: &Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, Vec3)> {
    let e1 = b - a;
    let e2 = c - a;
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = ray.origin - a;
    let u = tv.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tv.cross(e1);
    let v = ray.dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t <= 1e-5 {
        return None;
    }
    let n = e1.cross(e2).normalize_or(Vec3::Y);
    Some((t, face(n, ray.dir)))
}

/// Slab test returning (entry t, entry-face normal). A ray that starts inside misses
/// (solid colliders are not snap targets from within).
pub fn ray_aabb(ray: &Ray, min: Vec3, max: Vec3) -> Option<(f32, Vec3)> {
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    let mut axis = 0usize;
    let mut sign = 1.0f32;
    for i in 0..3 {
        let (o, d, lo, hi) = (ray.origin[i], ray.dir[i], min[i], max[i]);
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (mut t1, mut t2) = ((lo - o) / d, (hi - o) / d);
        let mut s = -1.0; // entering through the low face: outward normal is -axis
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
            s = 1.0;
        }
        if t1 > t_enter {
            t_enter = t1;
            axis = i;
            sign = s;
        }
        t_exit = t_exit.min(t2);
        if t_enter > t_exit {
            return None;
        }
    }
    if t_exit < 0.0 || t_enter < 0.0 || !t_enter.is_finite() {
        return None;
    }
    let mut n = Vec3::ZERO;
    n[axis] = sign;
    Some((t_enter, n))
}

/// Normal matrix of an affine transform (inverse transpose of its linear part).
fn normal_matrix(world: &Mat4) -> glam::Mat3 {
    glam::Mat3::from_mat4(*world).inverse().transpose()
}

/// Box collider: object-space `center` / `half` extents under `world`.
pub fn ray_box(ray: &Ray, world: &Mat4, center: Vec3, half: Vec3) -> Option<(f32, Vec3)> {
    if world.determinant().abs() < 1e-12 {
        return None;
    }
    let inv = world.inverse();
    let local = Ray { origin: inv.transform_point3(ray.origin), dir: inv.transform_vector3(ray.dir) };
    // `local.dir` is not unit: the parameter t stays the same as the world ray's.
    let (t, n) = ray_aabb(&local, center - half, center + half)?;
    let nw = (normal_matrix(world) * n).normalize_or(Vec3::Y);
    Some((t, face(nw, ray.dir)))
}

pub fn ray_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let oc = ray.origin - center;
    let b = oc.dot(ray.dir);
    let c = oc.length_squared() - radius * radius;
    if c < 0.0 {
        return None; // inside
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t < 0.0 {
        return None;
    }
    let p = ray.at(t);
    Some((t, ((p - center) / radius.max(1e-9)).normalize_or(Vec3::Y)))
}

/// Capsule: segment `a`..`b` swept by `radius`.
pub fn ray_capsule(ray: &Ray, a: Vec3, b: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let d = b - a;
    let len = d.length();
    if len < 1e-6 {
        return ray_sphere(ray, a, radius);
    }
    let u = d / len;
    // Starting inside the capsule: no hit.
    let s0 = (ray.origin - a).dot(u).clamp(0.0, len);
    if (ray.origin - (a + u * s0)).length_squared() < radius * radius {
        return None;
    }
    let mut best: Option<(f32, Vec3)> = None;
    let mut consider = |h: Option<(f32, Vec3)>| {
        if let Some(h) = h
            && best.is_none_or(|b| h.0 < b.0)
        {
            best = Some(h);
        }
    };
    consider(ray_sphere(ray, a, radius));
    consider(ray_sphere(ray, b, radius));
    // Side wall: infinite cylinder, clipped to the segment.
    let oc = ray.origin - a;
    let dp = ray.dir - u * ray.dir.dot(u);
    let op = oc - u * oc.dot(u);
    let qa = dp.length_squared();
    if qa > 1e-12 {
        let qb = 2.0 * dp.dot(op);
        let qc = op.length_squared() - radius * radius;
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let t = (-qb - disc.sqrt()) / (2.0 * qa);
            if t >= 0.0 {
                let p = ray.at(t);
                let s = (p - a).dot(u);
                if (0.0..=len).contains(&s) {
                    consider(Some((t, (p - (a + u * s)).normalize_or(Vec3::Y))));
                }
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Mesh BVH
// ---------------------------------------------------------------------------

const LEAF_TRIS: usize = 4;

#[derive(Clone, Copy)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// Leaf (`count > 0`): first slot in `order`. Inner: left child (right is +1).
    first: u32,
    count: u32,
}

/// Bounding volume hierarchy over a triangle list (median split on the longest axis).
pub struct Bvh {
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
}

fn tri_verts(pos: &[Vec3], idx: &[u32], tri: usize) -> Option<[Vec3; 3]> {
    let i = tri * 3;
    Some([*pos.get(*idx.get(i)? as usize)?, *pos.get(*idx.get(i + 1)? as usize)?, *pos.get(*idx.get(i + 2)? as usize)?])
}

impl Bvh {
    pub fn build(pos: &[Vec3], idx: &[u32]) -> Self {
        let ntri = idx.len() / 3;
        let verts: Vec<Option<[Vec3; 3]>> = (0..ntri).map(|t| tri_verts(pos, idx, t)).collect();
        let cent: Vec<Vec3> = verts.iter().map(|v| v.map_or(Vec3::ZERO, |v| (v[0] + v[1] + v[2]) / 3.0)).collect();
        let mut order: Vec<u32> = (0..ntri as u32).filter(|&t| verts[t as usize].is_some()).collect();
        let mut nodes = vec![BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: 0 }];
        let mut stack = vec![(0usize, 0usize, order.len())];
        while let Some((node, start, end)) = stack.pop() {
            let (mut mn, mut mx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            let (mut cmn, mut cmx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for &t in &order[start..end] {
                if let Some(v) = verts[t as usize] {
                    for p in v {
                        mn = mn.min(p);
                        mx = mx.max(p);
                    }
                }
                cmn = cmn.min(cent[t as usize]);
                cmx = cmx.max(cent[t as usize]);
            }
            if end - start <= LEAF_TRIS || (cmx - cmn).max_element() <= 1e-9 {
                nodes[node] = BvhNode { min: mn, max: mx, first: start as u32, count: (end - start) as u32 };
                continue;
            }
            let ext = cmx - cmn;
            let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
            let mid = (start + end) / 2;
            order[start..end].select_nth_unstable_by(mid - start, |&a, &b| cent[a as usize][axis].total_cmp(&cent[b as usize][axis]));
            let left = nodes.len();
            nodes.push(BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: 0 });
            nodes.push(BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: 0 });
            nodes[node] = BvhNode { min: mn, max: mx, first: left as u32, count: 0 };
            stack.push((left, start, mid));
            stack.push((left + 1, mid, end));
        }
        Self { nodes, order }
    }

    /// Nearest triangle hit (t, unit normal facing the ray). `ray.dir` may be unnormalized
    /// (object space); `t` is then in units of that direction.
    pub fn raycast(&self, pos: &[Vec3], idx: &[u32], ray: &Ray, t_max: f32) -> Option<(f32, Vec3)> {
        if self.order.is_empty() {
            return None;
        }
        let mut best: Option<(f32, Vec3)> = None;
        let mut limit = t_max;
        let mut stack = vec![0usize];
        let inv = Vec3::new(safe_recip(ray.dir.x), safe_recip(ray.dir.y), safe_recip(ray.dir.z));
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if !slab_hit(ray.origin, inv, node.min, node.max, limit) {
                continue;
            }
            if node.count > 0 {
                for k in 0..node.count as usize {
                    let tri = self.order[node.first as usize + k] as usize;
                    if let Some([a, b, c]) = tri_verts(pos, idx, tri)
                        && let Some((t, nrm)) = ray_triangle(ray, a, b, c)
                        && t < limit
                    {
                        limit = t;
                        best = Some((t, nrm));
                    }
                }
            } else {
                stack.push(node.first as usize);
                stack.push(node.first as usize + 1);
            }
        }
        best
    }
}

fn safe_recip(v: f32) -> f32 {
    if v.abs() < 1e-20 { 1e20f32.copysign(if v == 0.0 { 1.0 } else { v }) } else { 1.0 / v }
}

fn slab_hit(o: Vec3, inv: Vec3, min: Vec3, max: Vec3, t_max: f32) -> bool {
    let t1 = (min - o) * inv;
    let t2 = (max - o) * inv;
    let lo = t1.min(t2).max_element();
    let hi = t1.max(t2).min_element();
    hi >= lo.max(0.0) && lo <= t_max
}

/// Brute-force nearest triangle hit (small meshes, tests).
pub fn raycast_triangles(pos: &[Vec3], idx: &[u32], ray: &Ray, t_max: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    for tri in 0..idx.len() / 3 {
        if let Some([a, b, c]) = tri_verts(pos, idx, tri)
            && let Some((t, n)) = ray_triangle(ray, a, b, c)
            && t < best.map_or(t_max, |b| b.0)
        {
            best = Some((t, n));
        }
    }
    best
}

/// A mesh's CPU geometry.
pub struct MeshRef<'a> {
    pub pos: &'a [Vec3],
    pub idx: &'a [u32],
    pub bounds: (Vec3, Vec3),
}

/// Where the CPU geometry of scene meshes comes from.
pub trait MeshProvider {
    fn mesh(&self, mesh: usize) -> Option<MeshRef<'_>>;
}

impl MeshProvider for LoadedScene {
    fn mesh(&self, mesh: usize) -> Option<MeshRef<'_>> {
        let (pos, idx, bounds) = self.mesh_cpu(mesh)?;
        Some(MeshRef { pos, idx, bounds })
    }
}

/// Meshes up to this many triangles are tested without a BVH.
const BRUTE_TRIS: usize = 512;
/// Meshes up to this many triangles build their BVH inline; bigger ones build on a thread.
const INLINE_BUILD_TRIS: usize = 40_000;

enum Slot {
    Ready { key: (usize, usize), bvh: Arc<Bvh> },
    Building { key: (usize, usize), rx: Receiver<Bvh> },
}

/// Per-mesh BVHs, built lazily. A big mesh builds off the UI thread; until it is ready
/// the mesh's bounding box answers (so the UI never waits on a build).
#[derive(Default)]
pub struct SnapCache {
    slots: HashMap<usize, Slot>,
}

impl SnapCache {
    /// The BVH of `mesh` if it is ready (starting the build when needed).
    pub fn bvh(&mut self, mesh: usize, m: &MeshRef<'_>) -> Option<Arc<Bvh>> {
        let key = (m.pos.len(), m.idx.len());
        if let Some(slot) = self.slots.get_mut(&mesh) {
            match slot {
                Slot::Ready { key: k, bvh } if *k == key => return Some(bvh.clone()),
                Slot::Building { key: k, rx } if *k == key => {
                    if let Ok(b) = rx.try_recv() {
                        let bvh = Arc::new(b);
                        *slot = Slot::Ready { key, bvh: bvh.clone() };
                        return Some(bvh);
                    }
                    return None;
                }
                _ => {} // stale (the mesh changed in place): rebuild
            }
        }
        if m.idx.len() / 3 <= INLINE_BUILD_TRIS {
            let bvh = Arc::new(Bvh::build(m.pos, m.idx));
            self.slots.insert(mesh, Slot::Ready { key, bvh: bvh.clone() });
            Some(bvh)
        } else {
            let (pos, idx) = (m.pos.to_vec(), m.idx.to_vec());
            let (tx, rx) = channel();
            std::thread::spawn(move || {
                let _ = tx.send(Bvh::build(&pos, &idx));
            });
            self.slots.insert(mesh, Slot::Building { key, rx });
            None
        }
    }

    /// Whether any BVH is still building (the caller may want another frame).
    pub fn building(&self) -> bool {
        self.slots.values().any(|s| matches!(s, Slot::Building { .. }))
    }
}

// ---------------------------------------------------------------------------
// Snap targets
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// A scene mesh (its render geometry; also what a Mesh Collider uses).
    Mesh(usize),
    Box { center: Vec3, half: Vec3 },
    Sphere { center: Vec3, radius: f32 },
    /// `axis` is the object-space unit direction the capsule runs along.
    Capsule { center: Vec3, radius: f32, height: f32, axis: Vec3 },
}

#[derive(Clone, Copy, Debug)]
pub struct Target {
    /// Index of the owning scene object.
    pub object: usize,
    pub world: Mat4,
    pub shape: Shape,
}

fn max_axis_scale(world: &Mat4) -> f32 {
    world.x_axis.truncate().length().max(world.y_axis.truncate().length()).max(world.z_axis.truncate().length())
}

/// Hit one target. `max_t` prunes anything farther.
pub fn hit_target(ray: &Ray, target: &Target, meshes: &dyn MeshProvider, cache: &mut SnapCache, max_t: f32) -> Option<Hit> {
    let (t, normal) = match target.shape {
        Shape::Box { center, half } => ray_box(ray, &target.world, center, half)?,
        Shape::Sphere { center, radius } => {
            let c = target.world.transform_point3(center);
            ray_sphere(ray, c, radius * max_axis_scale(&target.world))?
        }
        Shape::Capsule { center, radius, height, axis } => {
            let s = max_axis_scale(&target.world);
            let half = (height * 0.5 - radius).max(0.0);
            let a = target.world.transform_point3(center - axis * half);
            let b = target.world.transform_point3(center + axis * half);
            ray_capsule(ray, a, b, radius * s)?
        }
        Shape::Mesh(mesh) => {
            if target.world.determinant().abs() < 1e-14 {
                return None;
            }
            let m = meshes.mesh(mesh)?;
            let inv = target.world.inverse();
            let local = Ray { origin: inv.transform_point3(ray.origin), dir: inv.transform_vector3(ray.dir) };
            let (bt, bn) = ray_aabb_or_inside(&local, m.bounds.0, m.bounds.1)?;
            if bt > max_t {
                return None;
            }
            let tris = m.idx.len() / 3;
            let local_hit = if tris == 0 {
                Some((bt, bn))
            } else if tris <= BRUTE_TRIS {
                raycast_triangles(m.pos, m.idx, &local, max_t)
            } else {
                match cache.bvh(mesh, &m) {
                    Some(bvh) => bvh.raycast(m.pos, m.idx, &local, max_t),
                    None => Some((bt, bn)), // still building: the box stands in
                }
            }?;
            let nw = (normal_matrix(&target.world) * local_hit.1).normalize_or(Vec3::Y);
            (local_hit.0, face(nw, ray.dir))
        }
    };
    if t > max_t || !t.is_finite() {
        return None;
    }
    Some(Hit { t, point: ray.at(t), normal, object: target.object })
}

/// Slab test for meshes: a ray starting inside the box still tests the triangles.
fn ray_aabb_or_inside(ray: &Ray, min: Vec3, max: Vec3) -> Option<(f32, Vec3)> {
    let inside = ray.origin.cmpge(min).all() && ray.origin.cmple(max).all();
    if inside {
        return Some((0.0, Vec3::Y));
    }
    ray_aabb(ray, min, max)
}

/// The nearest hit over `targets`, ignoring targets whose object `skip` rejects.
pub fn nearest_hit(
    ray: &Ray,
    targets: &[Target],
    meshes: &dyn MeshProvider,
    cache: &mut SnapCache,
    max_t: f32,
    skip: &dyn Fn(usize) -> bool,
) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for t in targets {
        if skip(t.object) {
            continue;
        }
        let limit = best.map_or(max_t, |b| b.t);
        if let Some(h) = hit_target(ray, t, meshes, cache, limit)
            && best.is_none_or(|b| h.t < b.t)
        {
            best = Some(h);
        }
    }
    best
}

/// Which objects may be snapped to: active, not scene-hidden, and not the object being
/// placed or anything under it. Locked objects stay valid targets.
pub fn snap_candidates(parents: &[Option<usize>], active: &[bool], hidden: &[bool], exclude_root: Option<usize>) -> Vec<bool> {
    let n = parents.len();
    let excluded = exclude_root.map_or_else(|| vec![false; n], |r| subtree_flags(parents, r));
    (0..n).map(|i| active.get(i).copied().unwrap_or(false) && !hidden.get(i).copied().unwrap_or(false) && !excluded[i]).collect()
}

/// `true` for `root` and every descendant.
pub fn subtree_flags(parents: &[Option<usize>], root: usize) -> Vec<bool> {
    let n = parents.len();
    let mut flags = vec![false; n];
    if root >= n {
        return flags;
    }
    flags[root] = true;
    // Parents may come after children in the array: iterate to a fixpoint.
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if !flags[i]
                && let Some(p) = parents[i]
                && p < n
                && flags[p]
            {
                flags[i] = true;
                changed = true;
            }
        }
    }
    flags
}

// ---------------------------------------------------------------------------
// Surface resolution
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    /// A collider or visible mesh (object index).
    Object(usize),
    /// The y = 0 ground plane.
    Ground,
    /// Nothing under the cursor: a point at a fixed distance along the ray.
    Air,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub kind: SurfaceKind,
}

pub const FALLBACK_DISTANCE: f32 = 8.0;
pub const GROUND_MAX_DISTANCE: f32 = 2000.0;

/// Colliders / meshes first, then the ground plane, then a point `FALLBACK_DISTANCE` out.
pub fn resolve_surface(ray: &Ray, targets: &[Target], meshes: &dyn MeshProvider, cache: &mut SnapCache, ground: bool) -> SurfaceHit {
    if let Some(h) = nearest_hit(ray, targets, meshes, cache, f32::INFINITY, &|_| false) {
        return SurfaceHit { point: h.point, normal: h.normal, kind: SurfaceKind::Object(h.object) };
    }
    if ground && ray.dir.y < -1e-4 {
        let t = -ray.origin.y / ray.dir.y;
        if t > 0.0 && t < GROUND_MAX_DISTANCE {
            return SurfaceHit { point: ray.at(t), normal: Vec3::Y, kind: SurfaceKind::Ground };
        }
    }
    SurfaceHit { point: ray.at(FALLBACK_DISTANCE), normal: Vec3::Y, kind: SurfaceKind::Air }
}

// ---------------------------------------------------------------------------
// Placement math
// ---------------------------------------------------------------------------

fn corners((min, max): (Vec3, Vec3)) -> [Vec3; 8] {
    std::array::from_fn(|i| Vec3::new(if i & 1 == 0 { min.x } else { max.x }, if i & 2 == 0 { min.y } else { max.y }, if i & 4 == 0 { min.z } else { max.z }))
}

/// The lowest extent of the object's bounds (given in its own space) along `normal`
/// once rotated by `rot` and scaled: the signed distance from the pivot to the
/// bounds' deepest point towards the surface. Negative for bounds around the pivot.
pub fn min_extent_along(bounds: Option<(Vec3, Vec3)>, rot: Quat, scale: Vec3, normal: Vec3) -> f32 {
    let Some(b) = bounds else { return 0.0 };
    corners(b).iter().map(|c| normal.dot(rot * (*c * scale))).fold(f32::INFINITY, f32::min)
}

/// Pivot position so the object's world bounds just touch the surface at `point`
/// (bounds on the normal's side). Objects without bounds sit with the pivot at `point`.
pub fn placed_position(point: Vec3, normal: Vec3, rot: Quat, scale: Vec3, bounds: Option<(Vec3, Vec3)>) -> Vec3 {
    point - normal * min_extent_along(bounds, rot, scale, normal)
}

/// Rotation whose up axis follows `normal` while the yaw (rotation about the object's own
/// up) stays `yaw_deg`. With `align` off only the yaw applies.
pub fn align_rotation(normal: Vec3, yaw_deg: f32, align: bool) -> Quat {
    let yaw = Quat::from_rotation_y(yaw_deg.to_radians());
    if !align {
        return yaw;
    }
    let n = normal.normalize_or(Vec3::Y);
    let swing = if n.dot(Vec3::Y) > 0.99999 {
        Quat::IDENTITY
    } else if n.dot(Vec3::Y) < -0.99999 {
        Quat::from_rotation_x(std::f32::consts::PI) // ceiling: hang upside down
    } else {
        Quat::from_rotation_arc(Vec3::Y, n)
    };
    swing * yaw
}

/// Step the yaw by `ticks` notches: 15 degrees, or 1 degree when `fine`. Coarse steps land on
/// multiples of 15. The result is in [0, 360).
pub fn step_yaw(yaw_deg: f32, ticks: i32, fine: bool) -> f32 {
    if fine {
        return (yaw_deg + ticks as f32).rem_euclid(360.0);
    }
    let g = yaw_deg / 15.0;
    let on_grid = (g - g.round()).abs() < 1e-3;
    let k = if on_grid {
        g.round() + ticks as f32
    } else if ticks > 0 {
        g.floor() + ticks as f32
    } else {
        g.ceil() + ticks as f32
    };
    (k * 15.0).rem_euclid(360.0)
}

/// Snap a surface point to the grid on the axes lying in the surface (the axis the normal
/// mostly points along keeps the surface's own coordinate).
pub fn snap_point_to_grid(point: Vec3, normal: Vec3, step: f32) -> Vec3 {
    if step <= 1e-6 {
        return point;
    }
    let a = normal.abs();
    let keep = if a.x >= a.y && a.x >= a.z { 0 } else if a.y >= a.z { 1 } else { 2 };
    let mut p = point;
    for i in 0..3 {
        if i != keep {
            p[i] = (p[i] / step).round() * step;
        }
    }
    p
}

/// Bounds in the root's space from the world-space boxes (min, max, world matrix) of its
/// meshes: every box corner brought into `root_world`'s space.
pub fn bounds_in_root_space(root_world: &Mat4, parts: impl IntoIterator<Item = (Vec3, Vec3, Mat4)>) -> Option<(Vec3, Vec3)> {
    if root_world.determinant().abs() < 1e-14 {
        return None;
    }
    let inv = root_world.inverse();
    let (mut mn, mut mx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for (lo, hi, world) in parts {
        if !(lo.is_finite() && hi.is_finite()) {
            continue;
        }
        let m = inv * world;
        for c in corners((lo, hi)) {
            let p = m.transform_point3(c);
            mn = mn.min(p);
            mx = mx.max(p);
        }
    }
    mn.is_finite().then_some((mn, mx))
}

// ---------------------------------------------------------------------------
// Scene glue (CPU only)
// ---------------------------------------------------------------------------

/// Every snap target in the scene: Box / Sphere / Capsule colliders and render meshes of
/// active, non-hidden objects, minus `exclude_root` and its descendants.
pub fn collect_targets(scene: &LoadedScene, exclude_root: Option<usize>) -> Vec<Target> {
    let n = scene.objects.len();
    let parents: Vec<Option<usize>> = scene.objects.iter().map(|o| o.parent).collect();
    let active: Vec<bool> = (0..n).map(|i| scene.is_active(i)).collect();
    let hidden: Vec<bool> = (0..n).map(|i| scene.view_hidden(i)).collect();
    let ok = snap_candidates(&parents, &active, &hidden, exclude_root);
    let worlds = scene.world_transforms();
    let mut out = Vec::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !ok[i] {
            continue;
        }
        for slot in o.render_slots() {
            out.push(Target { object: i, world: worlds[i], shape: Shape::Mesh(slot.mesh) });
        }
        for c in &o.components {
            if let Some(b) = c.as_any().downcast_ref::<zestengine_core::BoxCollider>() {
                out.push(Target { object: i, world: worlds[i], shape: Shape::Box { center: Vec3::from(b.center), half: Vec3::from(b.size) * 0.5 } });
            } else if let Some(s) = c.as_any().downcast_ref::<zestengine_core::SphereCollider>() {
                out.push(Target { object: i, world: worlds[i], shape: Shape::Sphere { center: Vec3::from(s.center), radius: s.radius } });
            } else if let Some(k) = c.as_any().downcast_ref::<zestengine_core::CapsuleCollider>() {
                out.push(Target {
                    object: i,
                    world: worlds[i],
                    shape: Shape::Capsule { center: Vec3::from(k.center), radius: k.radius, height: k.height, axis: k.direction.vector() },
                });
            }
        }
    }
    out
}

/// Bounds of `root` and everything under it, in `root`'s own space (None when it has no mesh).
pub fn subtree_bounds(scene: &LoadedScene, root: usize) -> Option<(Vec3, Vec3)> {
    if root >= scene.objects.len() {
        return None;
    }
    let parents: Vec<Option<usize>> = scene.objects.iter().map(|o| o.parent).collect();
    let in_tree = subtree_flags(&parents, root);
    let worlds = scene.world_transforms();
    let mut parts = Vec::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !in_tree[i] {
            continue;
        }
        for slot in o.render_slots() {
            if let Some(m) = scene.mesh_cpu(slot.mesh) {
                parts.push((m.2.0, m.2.1, worlds[i]));
            }
        }
    }
    bounds_in_root_space(&worlds[root], parts)
}

/// Put `index` at world (0,0,0) with identity rotation and scale 1, under `parent`
/// (None = root level). The parent's inverse is compensated, so the object does not move.
/// Returns false when the parent would make a cycle or does not exist.
pub fn place_at_world_origin(scene: &mut LoadedScene, index: usize, parent: Option<usize>) -> bool {
    if index >= scene.objects.len() {
        return false;
    }
    if let Some(p) = parent {
        if p >= scene.objects.len() || p == index {
            return false;
        }
        let parents: Vec<Option<usize>> = scene.objects.iter().map(|o| o.parent).collect();
        if subtree_flags(&parents, index)[p] {
            return false;
        }
    }
    let pw = parent.map(|p| scene.world_transform(p));
    let (t, r, s) = local_for_world_origin(pw);
    let o = &mut scene.objects[index];
    o.parent = parent;
    o.translation = t;
    o.rotation = r;
    o.scale = s;
    true
}

/// Set a ROOT-level object's pose from a world position and rotation.
pub fn set_root_pose(scene: &mut LoadedScene, index: usize, pos: Vec3, rot: Quat) {
    if let Some(o) = scene.objects.get_mut(index) {
        o.translation = pos;
        o.rotation = rot;
    }
}

// ---------------------------------------------------------------------------
// Session state machine
// ---------------------------------------------------------------------------

/// What the host must do after `Placement::frame` / `spawned`.
#[derive(Clone, Debug, PartialEq)]
pub enum Cmd {
    /// Begin a spawn: snapshot the scene for undo, then create the object (a model starts
    /// an async load and reports back via `model_arrived`).
    Spawn { ticket: u64, path: PathBuf, kind: AssetKind },
    /// The pointer came back: show the parked object again and select it.
    Reveal,
    /// The pointer left the viewport: hide the object (kept for re-entry).
    Park,
    /// Released inside: keep the object where it is, select it, switch to Move, record ONE
    /// undo step.
    Commit,
    /// Esc / released outside / drag ended: remove the object and leave no trace.
    Cancel,
}

/// Per-frame facts about the drag.
#[derive(Clone, Debug, Default)]
pub struct FrameInput {
    /// The file being dragged (None when no file drag is in progress).
    pub drag: Option<PathBuf>,
    /// The pointer is over the scene viewport (not over a floating window).
    pub inside: bool,
    /// The primary button was released this frame.
    pub released: bool,
    pub esc: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Idle,
    /// Spawned, model still loading. `commit_pending`: released before it arrived.
    Loading { commit_pending: bool },
    Live,
    Parked,
}

/// Why a model load was requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// A viewport placement session (its ticket).
    Session(u64),
    /// A hierarchy drop: place at world origin under `parent` (None = root).
    Origin { parent: Option<ObjectId> },
}

/// What to do with a model that finished loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    /// The live placement session wants it.
    Session(u64),
    Origin { parent: Option<ObjectId> },
    /// The session that asked for it is gone: drop the data without touching the scene.
    Discard,
    /// Not requested through placement: the normal import path.
    Unrelated,
}

#[derive(Default)]
pub struct Placement {
    phase: Option<Phase>,
    ticket: u64,
    live: Option<u64>,
    inside: bool,
    /// Esc pressed: ignore the rest of this drag.
    latched: bool,
    loads: Vec<(PathBuf, Purpose)>,
    /// Yaw about the object's up axis, degrees.
    pub yaw_deg: f32,
}

impl Placement {
    fn phase(&self) -> Phase {
        self.phase.unwrap_or(Phase::Idle)
    }

    /// A session is running (any state but idle).
    pub fn is_active(&self) -> bool {
        self.phase() != Phase::Idle
    }

    /// The object exists and follows the cursor.
    pub fn is_following(&self) -> bool {
        self.phase() == Phase::Live
    }

    /// The model is still loading (show the placeholder).
    pub fn is_loading(&self) -> bool {
        matches!(self.phase(), Phase::Loading { .. })
    }

    /// The drag is being handled by a session: viewport camera input (wheel dolly) yields.
    pub fn captures_input(&self) -> bool {
        self.is_active() || self.latched
    }

    fn cancel(&mut self) -> Vec<Cmd> {
        let was = self.is_active();
        self.phase = Some(Phase::Idle);
        self.live = None;
        if was { vec![Cmd::Cancel] } else { vec![] }
    }

    /// Advance with this frame's drag facts.
    pub fn frame(&mut self, input: &FrameInput) -> Vec<Cmd> {
        let spawn_kind = input.drag.as_deref().and_then(asset_kind);
        if self.latched {
            if input.drag.is_none() {
                self.latched = false;
            }
            return vec![];
        }
        match self.phase() {
            Phase::Idle => {
                let Some(kind) = spawn_kind else { return vec![] };
                if !input.inside {
                    return vec![];
                }
                let path = input.drag.clone().unwrap_or_default();
                self.ticket += 1;
                self.live = Some(self.ticket);
                self.inside = true;
                // A release in the very frame the pointer arrives still places the item.
                self.phase = Some(Phase::Loading { commit_pending: input.released });
                vec![Cmd::Spawn { ticket: self.ticket, path, kind }]
            }
            Phase::Loading { commit_pending } => {
                if input.esc {
                    self.latched = input.drag.is_some();
                    return self.cancel();
                }
                self.inside = input.inside;
                if commit_pending {
                    return vec![];
                }
                if input.released {
                    if input.inside {
                        self.phase = Some(Phase::Loading { commit_pending: true });
                        vec![]
                    } else {
                        self.cancel()
                    }
                } else if input.drag.is_none() {
                    self.cancel()
                } else {
                    vec![]
                }
            }
            Phase::Live | Phase::Parked => {
                if input.esc {
                    self.latched = input.drag.is_some();
                    return self.cancel();
                }
                if input.released {
                    return if input.inside && self.phase() == Phase::Live {
                        self.phase = Some(Phase::Idle);
                        self.live = None;
                        vec![Cmd::Commit]
                    } else {
                        self.cancel()
                    };
                }
                if input.drag.is_none() {
                    return self.cancel();
                }
                self.inside = input.inside;
                match (self.phase(), input.inside) {
                    (Phase::Live, false) => {
                        self.phase = Some(Phase::Parked);
                        vec![Cmd::Park]
                    }
                    (Phase::Parked, true) => {
                        self.phase = Some(Phase::Live);
                        vec![Cmd::Reveal]
                    }
                    _ => vec![],
                }
            }
        }
    }

    /// The host finished creating the object for `ticket` (immediately for a prefab or
    /// clip, on arrival for a model). `ok == false`: it could not be created.
    pub fn spawned(&mut self, ticket: u64, ok: bool) -> Vec<Cmd> {
        if self.live != Some(ticket) {
            return vec![Cmd::Cancel];
        }
        let Phase::Loading { commit_pending } = self.phase() else { return vec![] };
        if !ok {
            self.phase = Some(Phase::Idle);
            self.live = None;
            return vec![];
        }
        if commit_pending {
            self.phase = Some(Phase::Idle);
            self.live = None;
            return vec![Cmd::Commit];
        }
        if self.inside {
            self.phase = Some(Phase::Live);
            vec![]
        } else {
            self.phase = Some(Phase::Parked);
            vec![Cmd::Park]
        }
    }

    /// Remember that `source` was requested for `purpose` (call when starting the load).
    pub fn expect_model(&mut self, source: PathBuf, purpose: Purpose) {
        self.loads.push((source, purpose));
    }

    /// A model finished loading. Matches requests in order per source path.
    pub fn model_arrived(&mut self, source: &Path) -> Arrival {
        let Some(at) = self.loads.iter().position(|(s, _)| s == source) else { return Arrival::Unrelated };
        let (_, purpose) = self.loads.remove(at);
        match purpose {
            Purpose::Session(t) if self.live == Some(t) => Arrival::Session(t),
            Purpose::Session(_) => Arrival::Discard,
            Purpose::Origin { parent } => Arrival::Origin { parent },
        }
    }

    /// The hint to show: loading placeholder or live follow.
    pub fn hint_visible(&self) -> bool {
        matches!(self.phase(), Phase::Live | Phase::Loading { .. })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(o: [f32; 3], d: [f32; 3]) -> Ray {
        Ray::new(Vec3::from(o), Vec3::from(d))
    }
    fn near(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-4
    }

    struct Meshes(Vec<(Vec<Vec3>, Vec<u32>)>);
    impl MeshProvider for Meshes {
        fn mesh(&self, i: usize) -> Option<MeshRef<'_>> {
            let (pos, idx) = self.0.get(i)?;
            let mn = pos.iter().fold(Vec3::splat(f32::INFINITY), |a, p| a.min(*p));
            let mx = pos.iter().fold(Vec3::splat(f32::NEG_INFINITY), |a, p| a.max(*p));
            Some(MeshRef { pos, idx, bounds: (mn, mx) })
        }
    }

    /// A flat quad at y = 0 spanning +-5, facing +Y.
    fn floor_mesh() -> (Vec<Vec3>, Vec<u32>) {
        (vec![Vec3::new(-5.0, 0.0, -5.0), Vec3::new(5.0, 0.0, -5.0), Vec3::new(5.0, 0.0, 5.0), Vec3::new(-5.0, 0.0, 5.0)], vec![0, 2, 1, 0, 3, 2])
    }

    #[test]
    fn triangle_hit_and_miss() {
        let (a, b, c) = (Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let (t, n) = ray_triangle(&ray([0.0, 3.0, 0.0], [0.0, -1.0, 0.0]), a, b, c).unwrap();
        assert!((t - 3.0).abs() < 1e-5);
        assert!(near(n, Vec3::Y), "normal faces the ray");
        let (_, n) = ray_triangle(&ray([0.0, -3.0, 0.0], [0.0, 1.0, 0.0]), a, b, c).unwrap();
        assert!(near(n, Vec3::NEG_Y), "two sided");
        assert!(ray_triangle(&ray([5.0, 3.0, 0.0], [0.0, -1.0, 0.0]), a, b, c).is_none());
        assert!(ray_triangle(&ray([0.0, 3.0, 0.0], [0.0, 1.0, 0.0]), a, b, c).is_none(), "behind");
    }

    #[test]
    fn aabb_entry_face_normals() {
        let (mn, mx) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        let (t, n) = ray_aabb(&ray([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), mn, mx).unwrap();
        assert!((t - 4.0).abs() < 1e-5 && near(n, Vec3::X));
        let (t, n) = ray_aabb(&ray([0.0, -5.0, 0.0], [0.0, 1.0, 0.0]), mn, mx).unwrap();
        assert!((t - 4.0).abs() < 1e-5 && near(n, Vec3::NEG_Y));
        assert!(ray_aabb(&ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), mn, mx).is_none(), "inside is no hit");
        assert!(ray_aabb(&ray([3.0, 3.0, 0.0], [0.0, 1.0, 0.0]), mn, mx).is_none());
        // Parallel to a slab, outside it.
        assert!(ray_aabb(&ray([0.0, 5.0, 0.0], [1.0, 0.0, 0.0]), mn, mx).is_none());
    }

    #[test]
    fn rotated_scaled_box() {
        // 2x1x1 box rotated 90 degrees about Y, moved to x=10: its long side now runs along Z.
        let world = Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::from_rotation_y(std::f32::consts::FRAC_PI_2), Vec3::new(10.0, 0.0, 0.0));
        let r = ray([10.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let (t, n) = ray_box(&r, &world, Vec3::ZERO, Vec3::new(1.0, 0.5, 0.5)).unwrap();
        assert!((t - 9.0).abs() < 1e-4, "t = {t}");
        assert!(near(n, Vec3::Z));
        // Non-uniform scale keeps t honest: a unit cube scaled x4.
        let world = Mat4::from_scale(Vec3::new(4.0, 1.0, 1.0));
        let (t, _) = ray_box(&ray([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), &world, Vec3::ZERO, Vec3::splat(0.5)).unwrap();
        assert!((t - 8.0).abs() < 1e-4);
    }

    #[test]
    fn sphere_hit() {
        let (t, n) = ray_sphere(&ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]), Vec3::ZERO, 2.0).unwrap();
        assert!((t - 8.0).abs() < 1e-5 && near(n, Vec3::Z));
        assert!(ray_sphere(&ray([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]), Vec3::ZERO, 2.0).is_none(), "inside");
        assert!(ray_sphere(&ray([5.0, 0.0, 10.0], [0.0, 0.0, -1.0]), Vec3::ZERO, 2.0).is_none());
    }

    #[test]
    fn capsule_side_cap_and_inside() {
        let (a, b) = (Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let (t, n) = ray_capsule(&ray([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), a, b, 0.5).unwrap();
        assert!((t - 4.5).abs() < 1e-4 && near(n, Vec3::X), "side wall");
        let (t, n) = ray_capsule(&ray([0.0, 5.0, 0.0], [0.0, -1.0, 0.0]), a, b, 0.5).unwrap();
        assert!((t - 3.5).abs() < 1e-4 && near(n, Vec3::Y), "top cap");
        assert!(ray_capsule(&ray([5.0, 3.0, 0.0], [-1.0, 0.0, 0.0]), a, b, 0.5).is_none(), "above the cap");
        assert!(ray_capsule(&ray([0.1, 0.0, 0.0], [1.0, 0.0, 0.0]), a, b, 0.5).is_none(), "inside");
        // Aimed at the cap's shoulder (off-axis) still hits the sphere part.
        assert!(ray_capsule(&ray([0.3, 5.0, 0.0], [0.0, -1.0, 0.0]), a, b, 0.5).is_some());
    }

    #[test]
    fn bvh_matches_brute_force() {
        // A bumpy grid of 2 * 20 * 20 triangles.
        let n = 21;
        let mut pos = Vec::new();
        for z in 0..n {
            for x in 0..n {
                pos.push(Vec3::new(x as f32, ((x * 7 + z * 3) % 5) as f32 * 0.2, z as f32));
            }
        }
        let mut idx = Vec::new();
        for z in 0..n - 1 {
            for x in 0..n - 1 {
                let i = (z * n + x) as u32;
                idx.extend([i, i + n as u32, i + 1, i + 1, i + n as u32, i + n as u32 + 1]);
            }
        }
        let bvh = Bvh::build(&pos, &idx);
        for k in 0..60 {
            let o = Vec3::new(1.0 + (k % 8) as f32 * 2.3, 6.0, 1.0 + (k / 8) as f32 * 2.7);
            let r = Ray::new(o, Vec3::new(((k % 5) as f32 - 2.0) * 0.05, -1.0, ((k % 3) as f32 - 1.0) * 0.05));
            let a = bvh.raycast(&pos, &idx, &r, f32::INFINITY);
            let b = raycast_triangles(&pos, &idx, &r, f32::INFINITY);
            match (a, b) {
                (Some(a), Some(b)) => assert!((a.0 - b.0).abs() < 1e-4, "{a:?} vs {b:?}"),
                (None, None) => {}
                other => panic!("mismatch {other:?}"),
            }
        }
    }

    #[test]
    fn mesh_target_through_transform_with_bvh_cache() {
        // A big floor made of many triangles so the BVH path (not brute force) is used.
        let n = 30;
        let mut pos = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                pos.push(Vec3::new(x as f32 - 15.0, 0.0, z as f32 - 15.0));
            }
        }
        let mut idx = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let i = (z * (n + 1) + x) as u32;
                let w = (n + 1) as u32;
                idx.extend([i, i + w, i + 1, i + 1, i + w, i + w + 1]);
            }
        }
        assert!(idx.len() / 3 > BRUTE_TRIS);
        let meshes = Meshes(vec![(pos, idx)]);
        // Tilted 45 degrees about X and lifted by 2.
        let world = Mat4::from_rotation_translation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_4), Vec3::new(0.0, 2.0, 0.0));
        let targets = [Target { object: 3, world, shape: Shape::Mesh(0) }];
        let mut cache = SnapCache::default();
        let h = nearest_hit(&ray([0.0, 10.0, 0.0], [0.0, -1.0, 0.0]), &targets, &meshes, &mut cache, f32::INFINITY, &|_| false).unwrap();
        assert_eq!(h.object, 3);
        assert!(near(h.point, Vec3::new(0.0, 2.0, 0.0)));
        let expect = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4) * Vec3::Y;
        assert!(near(h.normal, expect), "normal {:?} vs {:?}", h.normal, expect);
        assert!(!cache.building());
    }

    #[test]
    fn nearest_hit_picks_closest_and_honours_exclusions() {
        let meshes = Meshes(vec![floor_mesh()]);
        let box_at = |obj: usize, y: f32| Target { object: obj, world: Mat4::from_translation(Vec3::new(0.0, y, 0.0)), shape: Shape::Box { center: Vec3::ZERO, half: Vec3::splat(0.5) } };
        let targets = [Target { object: 0, world: Mat4::IDENTITY, shape: Shape::Mesh(0) }, box_at(1, 1.0), box_at(2, 3.0)];
        let mut cache = SnapCache::default();
        let r = ray([0.0, 10.0, 0.0], [0.0, -1.0, 0.0]);
        let h = nearest_hit(&r, &targets, &meshes, &mut cache, f32::INFINITY, &|_| false).unwrap();
        assert_eq!(h.object, 2, "highest box is nearest");
        assert!((h.point.y - 3.5).abs() < 1e-4 && near(h.normal, Vec3::Y));
        // Skipping "self" (object 2) lands on the next box, skipping that too reaches the floor.
        let h = nearest_hit(&r, &targets, &meshes, &mut cache, f32::INFINITY, &|o| o == 2).unwrap();
        assert_eq!(h.object, 1);
        let h = nearest_hit(&r, &targets, &meshes, &mut cache, f32::INFINITY, &|o| o != 0).unwrap();
        assert_eq!(h.object, 0);
        assert!(h.point.y.abs() < 1e-5);
        assert!(nearest_hit(&r, &targets, &meshes, &mut cache, f32::INFINITY, &|_| true).is_none());
        // max_t prunes.
        assert!(nearest_hit(&r, &targets, &meshes, &mut cache, 1.0, &|_| false).is_none());
    }

    #[test]
    fn candidates_exclude_self_children_hidden_inactive_but_not_locked() {
        // 0 root, 1 child of 0, 2 grandchild, 3 unrelated, 4 hidden, 5 inactive, 6 locked (not hidden).
        let parents = [None, Some(0), Some(1), None, None, None, None];
        let active = [true, true, true, true, true, false, true];
        let hidden = [false, false, false, false, true, false, false];
        let ok = snap_candidates(&parents, &active, &hidden, Some(0));
        assert_eq!(ok, vec![false, false, false, true, false, false, true]);
        let all = snap_candidates(&parents, &active, &hidden, None);
        assert_eq!(all, vec![true, true, true, true, false, false, true]);
        // Children listed before their parent are still caught.
        let parents = [Some(1), None, Some(0)];
        assert_eq!(subtree_flags(&parents, 1), vec![true, true, true]);
    }

    #[test]
    fn resolve_surface_falls_back_to_ground_then_air() {
        let meshes = Meshes(vec![]);
        let mut cache = SnapCache::default();
        let s = resolve_surface(&ray([0.0, 4.0, 0.0], [0.0, -1.0, 1.0]), &[], &meshes, &mut cache, true);
        assert_eq!(s.kind, SurfaceKind::Ground);
        assert!(near(s.point, Vec3::new(0.0, 0.0, 4.0)) && near(s.normal, Vec3::Y));
        let s = resolve_surface(&ray([0.0, 4.0, 0.0], [0.0, 1.0, 0.0]), &[], &meshes, &mut cache, true);
        assert_eq!(s.kind, SurfaceKind::Air);
        assert!((s.point.y - (4.0 + FALLBACK_DISTANCE)).abs() < 1e-4);
        // A collider wins over the ground.
        let t = [Target { object: 9, world: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)), shape: Shape::Sphere { center: Vec3::ZERO, radius: 1.0 } }];
        let s = resolve_surface(&ray([0.0, 5.0, 0.0], [0.0, -1.0, 0.0]), &t, &meshes, &mut cache, true);
        assert_eq!(s.kind, SurfaceKind::Object(9));
        assert!((s.point.y - 2.0).abs() < 1e-4);
        // Ground off.
        let s = resolve_surface(&ray([0.0, 4.0, 0.0], [0.0, -1.0, 0.0]), &[], &meshes, &mut cache, false);
        assert_eq!(s.kind, SurfaceKind::Air);
    }

    #[test]
    fn offset_by_bounds_extent_along_normal() {
        // 2 x 4 x 2 object, pivot at its centre.
        let b = Some((Vec3::new(-1.0, -2.0, -1.0), Vec3::new(1.0, 2.0, 1.0)));
        // Floor: lifted by half the height.
        let p = placed_position(Vec3::ZERO, Vec3::Y, Quat::IDENTITY, Vec3::ONE, b);
        assert!(near(p, Vec3::new(0.0, 2.0, 0.0)));
        // Wall facing +X: pushed out by half the width.
        let p = placed_position(Vec3::ZERO, Vec3::X, Quat::IDENTITY, Vec3::ONE, b);
        assert!(near(p, Vec3::new(1.0, 0.0, 0.0)));
        // Same wall, object rotated 90 degrees about Z: its 4 m axis now points along the normal.
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let p = placed_position(Vec3::ZERO, Vec3::X, rot, Vec3::ONE, b);
        assert!(near(p, Vec3::new(2.0, 0.0, 0.0)), "{p:?}");
        // Arbitrary normal: the extent is support-function exact (box corner).
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let e = min_extent_along(b, Quat::IDENTITY, Vec3::ONE, n);
        assert!((e + 3.0 / 2.0f32.sqrt()).abs() < 1e-5, "e = {e}");
        // Scale grows it.
        let p = placed_position(Vec3::ZERO, Vec3::Y, Quat::IDENTITY, Vec3::splat(2.0), b);
        assert!(near(p, Vec3::new(0.0, 4.0, 0.0)));
        // No bounds: pivot at the hit.
        assert!(near(placed_position(Vec3::new(1.0, 2.0, 3.0), Vec3::Y, Quat::IDENTITY, Vec3::ONE, None), Vec3::new(1.0, 2.0, 3.0)));
        // The placed bounds touch the surface exactly (lowest corner sits on the plane through the hit).
        let rot = Quat::from_euler(glam::EulerRot::YXZ, 0.7, 0.3, -0.2);
        let n = Vec3::new(0.2, 1.0, 0.1).normalize();
        let hit = Vec3::new(3.0, 1.0, -2.0);
        let p = placed_position(hit, n, rot, Vec3::ONE, b);
        let Some(b) = b else { panic!("bounds given") };
        let lowest = corners(b).iter().map(|c| n.dot(p + rot * *c - hit)).fold(f32::INFINITY, f32::min);
        assert!(lowest.abs() < 1e-4, "lowest {lowest}");
    }

    #[test]
    fn align_keeps_yaw_on_floor_wall_ceiling_slope() {
        // Floor: only yaw.
        let q = align_rotation(Vec3::Y, 90.0, true);
        assert!(near(q * Vec3::Y, Vec3::Y));
        assert!(near(q * Vec3::Z, Vec3::X * 1.0), "yaw 90 about Y turns +Z to +X: {:?}", q * Vec3::Z);
        // Wall: up follows the normal.
        let q = align_rotation(Vec3::X, 0.0, true);
        assert!(near(q * Vec3::Y, Vec3::X));
        // Wall with a yaw: still up == normal, and the yaw spun the object about it.
        let q0 = align_rotation(Vec3::X, 0.0, true);
        let q1 = align_rotation(Vec3::X, 90.0, true);
        assert!(near(q1 * Vec3::Y, Vec3::X));
        assert!((q0 * Vec3::Z).dot(q1 * Vec3::Z).abs() < 1e-4, "quarter turn about the normal");
        // Ceiling: hangs.
        let q = align_rotation(Vec3::NEG_Y, 30.0, true);
        assert!(near(q * Vec3::Y, Vec3::NEG_Y));
        // Slope of 30 degrees tilts up by exactly 30.
        let n = Vec3::new(30f32.to_radians().sin(), 30f32.to_radians().cos(), 0.0);
        let q = align_rotation(n, 45.0, true);
        assert!(near(q * Vec3::Y, n));
        // Align off: only yaw regardless of the normal.
        let q = align_rotation(Vec3::X, 90.0, false);
        assert!(near(q * Vec3::Y, Vec3::Y));
    }

    #[test]
    fn yaw_stepping() {
        assert_eq!(step_yaw(0.0, 1, false), 15.0);
        assert_eq!(step_yaw(0.0, -1, false), 345.0);
        assert_eq!(step_yaw(350.0, 1, false), 0.0 + 360.0 - 360.0);
        assert_eq!(step_yaw(20.0, 1, false), 30.0, "off grid lands on the next line");
        assert_eq!(step_yaw(20.0, -1, false), 15.0);
        assert_eq!(step_yaw(90.0, 3, false), 135.0);
        assert_eq!(step_yaw(0.0, 1, true), 1.0);
        assert_eq!(step_yaw(0.0, -1, true), 359.0);
        assert_eq!(step_yaw(359.0, 2, true), 1.0);
    }

    #[test]
    fn grid_snap_keeps_the_surface_axis() {
        // Floor: x/z snap, y stays on the surface.
        let p = snap_point_to_grid(Vec3::new(1.26, 0.37, -0.74), Vec3::Y, 0.5);
        assert!(near(p, Vec3::new(1.5, 0.37, -0.5)), "{p:?}");
        // Wall facing +X: x stays, y/z snap.
        let p = snap_point_to_grid(Vec3::new(2.13, 1.26, 0.74), Vec3::X, 0.5);
        assert!(near(p, Vec3::new(2.13, 1.5, 0.5)), "{p:?}");
        // Degenerate step leaves the point alone.
        assert_eq!(snap_point_to_grid(Vec3::new(1.1, 2.2, 3.3), Vec3::Y, 0.0), Vec3::new(1.1, 2.2, 3.3));
    }

    #[test]
    fn asset_kinds_and_hierarchy_drop_rules() {
        assert_eq!(asset_kind(Path::new("a/b.FBX")), Some(AssetKind::Model));
        assert_eq!(asset_kind(Path::new("x.vrm")), Some(AssetKind::Model));
        assert_eq!(asset_kind(Path::new("x.prefab")), Some(AssetKind::Prefab));
        assert_eq!(asset_kind(Path::new("x.wav")), Some(AssetKind::Audio));
        assert_eq!(asset_kind(Path::new("x.png")), None);
        assert_eq!(asset_kind(Path::new("noext")), None);
        assert_eq!(classify_hierarchy_drop(Path::new("m.glb"), false), HierarchyDrop::Spawn(AssetKind::Model));
        assert_eq!(classify_hierarchy_drop(Path::new("m.glb"), true), HierarchyDrop::Spawn(AssetKind::Model));
        // A material on a row still assigns; on empty space it does nothing here.
        assert_eq!(classify_hierarchy_drop(Path::new("wood.material"), true), HierarchyDrop::AssignMaterial);
        assert_eq!(classify_hierarchy_drop(Path::new("wood.material"), false), HierarchyDrop::Ignore);
        // Scenes and textures keep their existing behaviour.
        assert_eq!(classify_hierarchy_drop(Path::new("level.scene"), true), HierarchyDrop::Ignore);
        assert_eq!(classify_hierarchy_drop(Path::new("t.png"), true), HierarchyDrop::Ignore);
    }

    #[test]
    fn world_origin_compensates_the_parent() {
        assert_eq!(local_for_world_origin(None), (Vec3::ZERO, Quat::IDENTITY, Vec3::ONE));
        // Rotated + uniformly scaled parent away from the origin.
        let pw = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), Quat::from_rotation_y(0.6), Vec3::new(5.0, 1.0, -3.0));
        let (t, r, s) = local_for_world_origin(Some(pw));
        let world = pw * Mat4::from_scale_rotation_translation(s, r, t);
        assert!(world.abs_diff_eq(Mat4::IDENTITY, 1e-4), "{world:?}");
        // A singular parent falls back to identity instead of NaN.
        let (t, _, s) = local_for_world_origin(Some(Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0))));
        assert!(t.is_finite() && s.is_finite());
    }

    fn scene_with(parents: &[(Option<usize>, Vec3, Quat, Vec3)]) -> LoadedScene {
        let mut scene = LoadedScene::empty();
        for (i, (p, t, r, s)) in parents.iter().enumerate() {
            let mut o = crate::scene::SceneObject::new_empty(ObjectId::new(), format!("o{i}"));
            o.parent = *p;
            o.translation = *t;
            o.rotation = *r;
            o.scale = *s;
            scene.objects.push(o);
        }
        scene
    }

    #[test]
    fn hierarchy_drop_lands_at_world_origin_with_and_without_parent() {
        let tilted = Quat::from_euler(glam::EulerRot::YXZ, 0.9, 0.2, -0.4);
        let mut scene = scene_with(&[
            (None, Vec3::new(7.0, 2.0, -4.0), tilted, Vec3::new(3.0, 3.0, 3.0)), // 0: the row
            (None, Vec3::new(9.0, 9.0, 9.0), Quat::from_rotation_x(1.0), Vec3::splat(5.0)), // 1: the new object
            (None, Vec3::new(1.0, 1.0, 1.0), Quat::IDENTITY, Vec3::ONE),         // 2: dropped on empty space
        ]);
        // Under the rotated + scaled row.
        assert!(place_at_world_origin(&mut scene, 1, Some(0)));
        assert_eq!(scene.objects[1].parent, Some(0));
        let w = scene.world_transform(1);
        assert!(w.abs_diff_eq(Mat4::IDENTITY, 1e-3), "world = {w:?}");
        // Root level.
        assert!(place_at_world_origin(&mut scene, 2, None));
        assert_eq!(scene.objects[2].parent, None);
        assert!(scene.world_transform(2).abs_diff_eq(Mat4::IDENTITY, 1e-6));
        // Not into itself or its own subtree.
        assert!(!place_at_world_origin(&mut scene, 0, Some(1)), "1 is a child of 0");
        assert!(!place_at_world_origin(&mut scene, 0, Some(0)));
        assert!(!place_at_world_origin(&mut scene, 0, Some(99)));
    }

    #[test]
    fn scene_targets_skip_hidden_inactive_and_the_placed_subtree() {
        let one = Vec3::ONE;
        let mut scene = scene_with(&[
            (None, Vec3::ZERO, Quat::IDENTITY, one),                  // 0: collider, valid
            (None, Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY, one),    // 1: collider on the placed model root
            (Some(1), Vec3::ZERO, Quat::IDENTITY, one),               // 2: its child collider
            (None, Vec3::new(0.0, 9.0, 0.0), Quat::IDENTITY, one),    // 3: disabled
            (None, Vec3::new(0.0, 7.0, 0.0), Quat::IDENTITY, one),    // 4: scene-hidden
            (None, Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY, one),    // 5: locked: still a target
        ]);
        for i in [0usize, 1, 2, 3, 4, 5] {
            scene.objects[i].components.push(Box::new(zestengine_core::BoxCollider::default()));
        }
        scene.objects[3].enabled = false;
        scene.scene_view.edit = true;
        scene.set_view_hidden(4, true);
        scene.set_view_locked(5, true);
        let meshes = Meshes(vec![]);
        let mut cache = SnapCache::default();
        let r = ray([0.0, 20.0, 0.0], [0.0, -1.0, 0.0]);
        let all = collect_targets(&scene, None);
        let hit = nearest_hit(&r, &all, &meshes, &mut cache, f32::INFINITY, &|_| false).unwrap();
        assert_eq!(hit.object, 1, "the placed model would snap to itself without the exclusion");
        let targets = collect_targets(&scene, Some(1));
        assert!(targets.iter().all(|t| t.object != 1 && t.object != 2 && t.object != 3), "self, child, disabled out");
        let hit = nearest_hit(&r, &targets, &meshes, &mut cache, f32::INFINITY, &|_| false).unwrap();
        assert_eq!(hit.object, 5, "locked objects remain snap targets, hidden ones do not");
        assert!((hit.point.y - 3.5).abs() < 1e-4);
    }

    fn input(drag: Option<&str>, inside: bool, released: bool) -> FrameInput {
        FrameInput { drag: drag.map(PathBuf::from), inside, released, esc: false }
    }

    #[test]
    fn session_enter_follow_leave_reenter_commit() {
        let mut p = Placement::default();
        // Dragging outside: nothing happens.
        assert!(p.frame(&input(Some("a.prefab"), false, false)).is_empty());
        assert!(!p.is_active());
        // Entering spawns exactly once.
        let cmds = p.frame(&input(Some("a.prefab"), true, false));
        let ticket = match &cmds[..] {
            [Cmd::Spawn { ticket, kind: AssetKind::Prefab, .. }] => *ticket,
            other => panic!("{other:?}"),
        };
        assert!(p.is_loading());
        assert!(p.frame(&input(Some("a.prefab"), true, false)).is_empty(), "hover updates do not respawn");
        assert!(p.spawned(ticket, true).is_empty());
        assert!(p.is_following());
        // Leaving parks; re-entering reveals (no second spawn).
        assert_eq!(p.frame(&input(Some("a.prefab"), false, false)), vec![Cmd::Park]);
        assert!(!p.is_following() && p.is_active());
        assert_eq!(p.frame(&input(Some("a.prefab"), true, false)), vec![Cmd::Reveal]);
        // Release inside commits and ends the session.
        assert_eq!(p.frame(&input(Some("a.prefab"), true, true)), vec![Cmd::Commit]);
        assert!(!p.is_active());
        assert!(p.frame(&input(None, true, false)).is_empty());
    }

    #[test]
    fn session_cancel_paths_leave_no_trace() {
        let spawn = |p: &mut Placement| {
            let c = p.frame(&input(Some("a.wav"), true, false));
            let Cmd::Spawn { ticket, .. } = c[0] else { panic!() };
            p.spawned(ticket, true);
        };
        // Release outside the viewport.
        let mut p = Placement::default();
        spawn(&mut p);
        assert_eq!(p.frame(&input(Some("a.wav"), false, false)), vec![Cmd::Park]);
        assert_eq!(p.frame(&input(Some("a.wav"), false, true)), vec![Cmd::Cancel]);
        assert!(!p.is_active());
        // Esc while following, and the rest of that drag is ignored.
        let mut p = Placement::default();
        spawn(&mut p);
        let esc = FrameInput { esc: true, ..input(Some("a.wav"), true, false) };
        assert_eq!(p.frame(&esc), vec![Cmd::Cancel]);
        assert!(p.frame(&input(Some("a.wav"), true, false)).is_empty(), "no respawn after Esc");
        assert!(p.frame(&input(Some("a.wav"), true, true)).is_empty(), "release after Esc commits nothing");
        assert!(p.frame(&input(None, true, false)).is_empty());
        // A fresh drag afterwards works again.
        assert!(matches!(p.frame(&input(Some("a.wav"), true, false))[0], Cmd::Spawn { .. }));
        // The drag vanishes (e.g. dropped somewhere that swallowed it): cancel.
        let mut p = Placement::default();
        spawn(&mut p);
        assert_eq!(p.frame(&input(None, true, false)), vec![Cmd::Cancel]);
        // Non-spawnable files never start a session.
        let mut p = Placement::default();
        assert!(p.frame(&input(Some("a.png"), true, false)).is_empty());
        assert!(p.frame(&input(Some("a.material"), true, false)).is_empty());
    }

    #[test]
    fn async_load_completion_orders() {
        // 1. Arrives while following.
        let mut p = Placement::default();
        let Cmd::Spawn { ticket, path, .. } = p.frame(&input(Some("big.fbx"), true, false)).remove(0) else { panic!() };
        p.expect_model(path.clone(), Purpose::Session(ticket));
        assert!(p.is_loading() && p.hint_visible());
        assert!(p.frame(&input(Some("big.fbx"), true, false)).is_empty());
        assert_eq!(p.model_arrived(&path), Arrival::Session(ticket));
        assert!(p.spawned(ticket, true).is_empty());
        assert!(p.is_following());

        // 2. Released before it arrived: it commits the moment it does.
        let mut p = Placement::default();
        let Cmd::Spawn { ticket, path, .. } = p.frame(&input(Some("big.fbx"), true, false)).remove(0) else { panic!() };
        p.expect_model(path.clone(), Purpose::Session(ticket));
        assert!(p.frame(&input(Some("big.fbx"), true, true)).is_empty());
        assert!(p.frame(&input(None, false, false)).is_empty(), "waiting for the load survives the drag ending");
        assert_eq!(p.model_arrived(&path), Arrival::Session(ticket));
        assert_eq!(p.spawned(ticket, true), vec![Cmd::Commit]);
        assert!(!p.is_active());

        // 3. Pointer left while loading: it arrives parked.
        let mut p = Placement::default();
        let Cmd::Spawn { ticket, path, .. } = p.frame(&input(Some("big.fbx"), true, false)).remove(0) else { panic!() };
        p.expect_model(path.clone(), Purpose::Session(ticket));
        assert!(p.frame(&input(Some("big.fbx"), false, false)).is_empty());
        assert_eq!(p.model_arrived(&path), Arrival::Session(ticket));
        assert_eq!(p.spawned(ticket, true), vec![Cmd::Park]);

        // 4. Cancelled before it arrived: the late data is discarded, not added to the scene.
        let mut p = Placement::default();
        let Cmd::Spawn { ticket, path, .. } = p.frame(&input(Some("big.fbx"), true, false)).remove(0) else { panic!() };
        p.expect_model(path.clone(), Purpose::Session(ticket));
        let esc = FrameInput { esc: true, ..input(Some("big.fbx"), true, false) };
        assert_eq!(p.frame(&esc), vec![Cmd::Cancel]);
        assert_eq!(p.model_arrived(&path), Arrival::Discard);
        assert_eq!(p.model_arrived(&path), Arrival::Unrelated, "consumed once");

        // 5. Stale arrival after a NEW session started for the same file.
        let mut p = Placement::default();
        let Cmd::Spawn { ticket: t1, path, .. } = p.frame(&input(Some("big.fbx"), true, false)).remove(0) else { panic!() };
        p.expect_model(path.clone(), Purpose::Session(t1));
        p.frame(&esc);
        p.frame(&input(None, false, false));
        let Cmd::Spawn { ticket: t2, .. } = p.frame(&input(Some("big.fbx"), true, false)).remove(0) else { panic!() };
        p.expect_model(path.clone(), Purpose::Session(t2));
        assert_ne!(t1, t2);
        assert_eq!(p.model_arrived(&path), Arrival::Discard, "first request belongs to the dead session");
        assert_eq!(p.model_arrived(&path), Arrival::Session(t2));

        // 6. A hierarchy drop load is independent of any session.
        let mut p = Placement::default();
        let parent = Some(ObjectId::new());
        p.expect_model(PathBuf::from("m.glb"), Purpose::Origin { parent });
        assert_eq!(p.model_arrived(Path::new("m.glb")), Arrival::Origin { parent });
        assert_eq!(p.model_arrived(Path::new("other.glb")), Arrival::Unrelated);

        // 7. A failed load ends the session quietly.
        let mut p = Placement::default();
        let Cmd::Spawn { ticket, .. } = p.frame(&input(Some("bad.fbx"), true, false)).remove(0) else { panic!() };
        assert!(p.spawned(ticket, false).is_empty());
        assert!(!p.is_active());
    }

    #[test]
    fn instant_release_on_arrival_still_places() {
        let mut p = Placement::default();
        let Cmd::Spawn { ticket, .. } = p.frame(&input(Some("a.prefab"), true, true)).remove(0) else { panic!() };
        assert_eq!(p.spawned(ticket, true), vec![Cmd::Commit]);
    }

    #[test]
    fn undo_is_one_step_by_construction() {
        // The host snapshots on `Spawn` and records on `Commit`; between them only
        // Park / Reveal happen, and Cancel records nothing. Count the commands.
        let mut p = Placement::default();
        let mut log = Vec::new();
        let seq = [(true, false), (true, false), (false, false), (true, false), (true, true)];
        for (inside, released) in seq {
            log.extend(p.frame(&input(Some("a.prefab"), inside, released)));
            if let Some(Cmd::Spawn { ticket, .. }) = log.last().cloned() {
                log.extend(p.spawned(ticket, true));
            }
        }
        let spawns = log.iter().filter(|c| matches!(c, Cmd::Spawn { .. })).count();
        let commits = log.iter().filter(|c| matches!(c, Cmd::Commit)).count();
        assert_eq!((spawns, commits), (1, 1));
    }
}
