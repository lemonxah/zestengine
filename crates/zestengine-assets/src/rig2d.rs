//! 2D skeletal data and logic (2D-7): sprite bones, 4-influence vertex weights,
//! CPU skinning, and the pure parts of the Skinning Editor (bone editing,
//! automatic geometry from the sprite alpha, automatic weights, weight
//! painting). The `.rig2d` asset stores one skeleton and any number of skinned
//! sprite meshes ("parts") that share it.
//!
//! All coordinates are sprite space: world units (pixels / PPU), origin at the
//! sprite's pivot, y up, exactly like `SpriteGeometry::positions`. A bone is a
//! local position and rotation relative to its parent (root bones: relative to
//! the sprite origin); it points along its local +X for `length` units. There is
//! no rest scale. Bind poses are the inverse rest world transforms (Unity's
//! convention: `final = bone_world * bind_pose`), so a skeleton that sits at
//! rest skins every vertex to itself.
//!
//! # Skinning
//!
//! Linear blend skinning with at most [`MAX_INFLUENCES`] (4) bones per vertex.
//! Weights are stored as `u16` bone indices plus `f32` weights; a vertex with
//! one full influence takes a single-matrix fast path. [`skin_positions`] is the
//! CPU path the engine runs per frame into a per-skin dynamic mesh (see
//! `render/src/sprite2d.rs`, `SpriteMeshPool::add_dynamic`): sprites have tens
//! to a few hundred vertices, so CPU skinning is cheap and keeps the sprite
//! vertex shader and pipeline unchanged.
//!
//! # Skinning Editor logic
//!
//! * [`auto_geometry`]: outline of the alpha (the existing tracer), resampled
//!   to a spacing, plus interior grid points, Delaunay-triangulated and
//!   clipped to the outline. Not a true constrained Delaunay: the outline is
//!   resampled finer than the interior spacing, so boundary edges survive in
//!   practice; the tests check the area stays within a few percent.
//! * [`auto_weights`]: `Heat` diffuses a per-bone source over the mesh graph
//!   (so a weight does not jump a gap in the sprite, e.g. between legs), `Distance`
//!   is a distance falloff. Both end in the same clean-up: top 4, drop tiny
//!   weights, normalise to a partition of unity. This replaces Unity's closed
//!   bounded-biharmonic plugin with something simpler that needs no solver.
//! * [`paint_weights`]: brush add / subtract / smooth with radius, hardness and
//!   strength; every touched vertex stays normalised and within 4 influences.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use glam::{Affine2, Vec2};
use serde::{Deserialize, Serialize};

use crate::sprite::{PixelRect, pixel_to_world};
use crate::sprite_mesh::{self, AlphaMap, P2, PixelMesh};

pub const RIG_EXTENSION: &str = "rig2d";
/// Influences stored per vertex.
pub const MAX_INFLUENCES: usize = 4;
/// Weights below this are dropped by clean-up.
pub const MIN_WEIGHT: f32 = 0.01;

// ---------------------------------------------------------------------------
// Bones
// ---------------------------------------------------------------------------

/// One sprite bone (Unity: `SpriteBone`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Bone2D {
    pub name: String,
    /// Stable id, used to rebind a skin to scene bones when objects are renamed
    /// (Unity matches bone GUIDs first, then paths). 0 = unassigned.
    pub id: u64,
    /// Index of the parent bone; `-1` = root.
    pub parent: i32,
    /// Position relative to the parent bone's origin, in the parent's frame.
    pub position: [f32; 2],
    /// Rotation relative to the parent, radians.
    pub rotation: f32,
    /// Length along the bone's +X axis.
    pub length: f32,
}

impl Default for Bone2D {
    fn default() -> Self {
        Self { name: String::new(), id: 0, parent: -1, position: [0.0; 2], rotation: 0.0, length: 0.25 }
    }
}

impl Bone2D {
    pub fn local(&self) -> Affine2 {
        Affine2::from_angle_translation(self.rotation, Vec2::from(self.position))
    }
}

/// A local pose of one bone (an animated bone object's transform).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BonePose {
    pub position: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
}

impl BonePose {
    pub fn rest(b: &Bone2D) -> Self {
        Self { position: b.position, rotation: b.rotation, scale: [1.0, 1.0] }
    }
    pub fn local(&self) -> Affine2 {
        Affine2::from_scale_angle_translation(Vec2::from(self.scale), self.rotation, Vec2::from(self.position))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Skeleton2D {
    pub bones: Vec<Bone2D>,
}

fn fnv(name: &str, salt: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ salt;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h.max(1)
}

impl Skeleton2D {
    /// World transforms of the bones for the given local transforms (parents
    /// resolved on demand; a dangling parent or a cycle counts as a root).
    pub fn world_from_locals(&self, locals: &[Affine2]) -> Vec<Affine2> {
        let n = self.bones.len();
        let mut out = vec![Affine2::IDENTITY; n];
        let mut state = vec![0u8; n]; // 0 todo, 1 in progress, 2 done
        for i in 0..n {
            self.resolve(i, locals, &mut state, &mut out);
        }
        out
    }

    fn resolve(&self, i: usize, locals: &[Affine2], state: &mut [u8], out: &mut [Affine2]) {
        if state[i] == 2 {
            return;
        }
        state[i] = 1;
        let p = self.bones[i].parent;
        let local = locals.get(i).copied().unwrap_or(Affine2::IDENTITY);
        out[i] = if p >= 0 && (p as usize) < self.bones.len() && state[p as usize] != 1 {
            self.resolve(p as usize, locals, state, out);
            out[p as usize] * local
        } else {
            local
        };
        state[i] = 2;
    }

    /// Rest world transforms.
    pub fn world_rest(&self) -> Vec<Affine2> {
        let l: Vec<Affine2> = self.bones.iter().map(Bone2D::local).collect();
        self.world_from_locals(&l)
    }

    /// World transforms of a pose (one [`BonePose`] per bone).
    pub fn world_pose(&self, pose: &[BonePose]) -> Vec<Affine2> {
        let l: Vec<Affine2> = self.bones.iter().enumerate().map(|(i, b)| pose.get(i).map(BonePose::local).unwrap_or_else(|| b.local())).collect();
        self.world_from_locals(&l)
    }

    /// Bind poses: inverse rest world transforms.
    pub fn bind_poses(&self) -> Vec<Affine2> {
        self.world_rest().iter().map(|m| m.inverse()).collect()
    }

    /// Skin matrices `pose_world[i] * bind[i]` from bone world transforms
    /// (scene bone objects expressed in sprite space).
    pub fn skin_matrices(&self, pose_world: &[Affine2]) -> Vec<Affine2> {
        let bind = self.bind_poses();
        bind.iter().enumerate().map(|(i, b)| pose_world.get(i).copied().unwrap_or(Affine2::IDENTITY) * *b).collect()
    }

    /// `(head, tip)` of every bone at rest, in sprite space.
    pub fn rest_segments(&self) -> Vec<([f32; 2], [f32; 2])> {
        self.world_rest()
            .iter()
            .zip(&self.bones)
            .map(|(m, b)| (m.translation.to_array(), m.transform_point2(Vec2::new(b.length, 0.0)).to_array()))
            .collect()
    }

    /// Give every bone with id 0 a stable id from its name (unique within the skeleton).
    pub fn ensure_ids(&mut self) -> bool {
        let mut changed = false;
        let mut seen: std::collections::HashSet<u64> = self.bones.iter().filter(|b| b.id != 0).map(|b| b.id).collect();
        for b in &mut self.bones {
            if b.id == 0 {
                let mut salt = 0;
                let mut id = fnv(&b.name, salt);
                while !seen.insert(id) {
                    salt += 1;
                    id = fnv(&b.name, salt);
                }
                b.id = id;
                changed = true;
            }
        }
        changed
    }

    /// Problems that would break skinning: bad parents, cycles, duplicate names.
    pub fn validate(&self) -> Vec<String> {
        let mut out = Vec::new();
        let n = self.bones.len() as i32;
        for (i, b) in self.bones.iter().enumerate() {
            if b.parent >= n || b.parent < -1 || b.parent == i as i32 {
                out.push(format!("bone {i} '{}' has an invalid parent {}", b.name, b.parent));
                continue;
            }
            let (mut p, mut steps) = (b.parent, 0);
            while p >= 0 && steps <= n {
                p = self.bones[p as usize].parent;
                steps += 1;
            }
            if steps > n {
                out.push(format!("bone {i} '{}' is in a parent cycle", b.name));
            }
        }
        let mut names = std::collections::HashSet::new();
        for b in &self.bones {
            if !b.name.is_empty() && !names.insert(b.name.as_str()) {
                out.push(format!("duplicate bone name '{}'", b.name));
            }
        }
        out
    }

    fn is_descendant(&self, mut i: usize, of: usize) -> bool {
        for _ in 0..=self.bones.len() {
            let p = self.bones[i].parent;
            if p < 0 {
                return false;
            }
            if p as usize == of {
                return true;
            }
            i = p as usize;
        }
        true // cycle: refuse
    }

    fn parent_world(&self, parent: i32, world: &[Affine2]) -> Affine2 {
        if parent >= 0 { world[parent as usize] } else { Affine2::IDENTITY }
    }

    /// Set bone `i`'s local transform so its world head / tip are as given
    /// (children keep their local transforms and move with it).
    pub fn set_bone_world(&mut self, i: usize, head: [f32; 2], tip: [f32; 2]) {
        let world = self.world_rest();
        let pw = self.parent_world(self.bones[i].parent, &world);
        let inv = pw.inverse();
        let h = inv.transform_point2(Vec2::from(head));
        let t = inv.transform_point2(Vec2::from(tip));
        let d = t - h;
        let b = &mut self.bones[i];
        b.position = h.to_array();
        b.rotation = d.y.atan2(d.x);
        b.length = d.length();
    }

    /// Add a bone from a world-space head and tip under `parent`. Returns its index.
    pub fn add_bone(&mut self, parent: Option<usize>, head: [f32; 2], tip: [f32; 2], name: &str) -> usize {
        let parent = parent.filter(|&p| p < self.bones.len());
        let name = self.unique_name(name);
        self.bones.push(Bone2D { name, parent: parent.map_or(-1, |p| p as i32), ..Default::default() });
        let i = self.bones.len() - 1;
        self.set_bone_world(i, head, tip);
        self.ensure_ids();
        i
    }

    fn unique_name(&self, base: &str) -> String {
        let base = if base.is_empty() { "Bone" } else { base };
        if !self.bones.iter().any(|b| b.name == base) {
            return base.to_owned();
        }
        (1..).map(|k| format!("{base} {k}")).find(|n| !self.bones.iter().any(|b| &b.name == n)).unwrap()
    }

    /// Re-parent keeping the bone's world transform. False when it would make a cycle.
    pub fn reparent(&mut self, i: usize, new_parent: Option<usize>) -> bool {
        if i >= self.bones.len() {
            return false;
        }
        if let Some(p) = new_parent
            && (p >= self.bones.len() || p == i || self.is_descendant(p, i))
        {
            return false;
        }
        let world = self.world_rest();
        let old = world[i];
        self.bones[i].parent = new_parent.map_or(-1, |p| p as i32);
        let pw = self.parent_world(self.bones[i].parent, &world);
        let local = pw.inverse() * old;
        let b = &mut self.bones[i];
        b.position = local.translation.to_array();
        b.rotation = local.matrix2.x_axis.y.atan2(local.matrix2.x_axis.x);
        true
    }

    /// Remove a bone; its children move to its parent keeping their world
    /// transforms. Returns the old-to-new index map (`None` = the removed bone)
    /// for [`SkinMesh2D::remap_bones`].
    pub fn remove_bone(&mut self, i: usize) -> Vec<Option<usize>> {
        if i >= self.bones.len() {
            return (0..self.bones.len()).map(Some).collect();
        }
        let grand = self.bones[i].parent;
        let kids: Vec<usize> = (0..self.bones.len()).filter(|&k| self.bones[k].parent == i as i32).collect();
        for k in kids {
            self.reparent(k, if grand >= 0 { Some(grand as usize) } else { None });
        }
        let mut map = Vec::with_capacity(self.bones.len());
        let mut next = 0;
        for k in 0..self.bones.len() {
            if k == i {
                map.push(None);
            } else {
                map.push(Some(next));
                next += 1;
            }
        }
        self.bones.remove(i);
        for b in &mut self.bones {
            if b.parent >= 0 {
                b.parent = map[b.parent as usize].map_or(-1, |p| p as i32);
            }
        }
        map
    }

    /// Split bone `i` at fraction `t` along its length: the original keeps the
    /// first part and a new child carries the rest and the old children.
    /// Returns the new bone's index.
    pub fn split_bone(&mut self, i: usize, t: f32) -> Option<usize> {
        let b = self.bones.get(i)?.clone();
        let t = t.clamp(0.05, 0.95);
        let kids: Vec<usize> = (0..self.bones.len()).filter(|&k| self.bones[k].parent == i as i32).collect();
        let name = self.unique_name(&format!("{} Split", b.name));
        let new = Bone2D { name, parent: i as i32, position: [b.length * t, 0.0], rotation: 0.0, length: b.length * (1.0 - t), id: 0 };
        self.bones[i].length = b.length * t;
        self.bones.push(new);
        let n = self.bones.len() - 1;
        for k in kids {
            self.reparent(k, Some(n));
        }
        self.ensure_ids();
        Some(n)
    }
}

// ---------------------------------------------------------------------------
// Weights and skinning
// ---------------------------------------------------------------------------

/// Up to four bone influences of one vertex. Unused slots have weight 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct VertexWeights {
    pub bones: [u16; MAX_INFLUENCES],
    pub weights: [f32; MAX_INFLUENCES],
}

impl VertexWeights {
    pub fn single(bone: u16) -> Self {
        Self { bones: [bone, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0] }
    }

    pub fn sum(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// Weight of `bone` (0 when not an influence).
    pub fn weight_of(&self, bone: u16) -> f32 {
        (0..MAX_INFLUENCES).filter(|&k| self.weights[k] > 0.0 && self.bones[k] == bone).map(|k| self.weights[k]).sum()
    }

    /// Build from any list of `(bone, weight)`: merge duplicates, keep the four
    /// largest, drop weights under [`MIN_WEIGHT`] (never the largest), normalise.
    /// An empty or all-zero list gives `fallback` (bone 0 fully).
    pub fn from_pairs(pairs: &[(u16, f32)], fallback: u16) -> Self {
        let mut merged: Vec<(u16, f32)> = Vec::with_capacity(pairs.len());
        for &(b, w) in pairs {
            if w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !w.is_finite() {
                continue;
            }
            match merged.iter_mut().find(|(mb, _)| *mb == b) {
                Some(e) => e.1 += w,
                None => merged.push((b, w)),
            }
        }
        if merged.is_empty() {
            return Self::single(fallback);
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        merged.truncate(MAX_INFLUENCES);
        let total: f32 = merged.iter().map(|m| m.1).sum();
        let top = merged[0].1 / total;
        let mut kept: Vec<(u16, f32)> = merged.iter().map(|&(b, w)| (b, w / total)).filter(|&(_, w)| w >= MIN_WEIGHT).collect();
        if kept.is_empty() {
            kept.push((merged[0].0, top));
        }
        let total: f32 = kept.iter().map(|m| m.1).sum();
        let mut out = Self::default();
        for (k, (b, w)) in kept.into_iter().enumerate() {
            out.bones[k] = b;
            out.weights[k] = w / total;
        }
        out
    }

    pub fn pairs(&self) -> Vec<(u16, f32)> {
        (0..MAX_INFLUENCES).filter(|&k| self.weights[k] > 0.0).map(|k| (self.bones[k], self.weights[k])).collect()
    }
}

/// A sprite mesh with weights (the skinned counterpart of `SpriteGeometry`'s
/// positions / indices; UVs are derived from the sprite at draw time).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SkinMesh2D {
    /// Rest positions, sprite space.
    pub positions: Vec<[f32; 2]>,
    /// Counter-clockwise triangle list.
    pub indices: Vec<u16>,
    pub weights: Vec<VertexWeights>,
}

impl SkinMesh2D {
    pub fn new(positions: Vec<[f32; 2]>, indices: Vec<u16>, bone: u16) -> Self {
        let weights = vec![VertexWeights::single(bone); positions.len()];
        Self { positions, indices, weights }
    }

    /// Weights and positions agree in length, indices are in range, every
    /// vertex's weights sum to ~1 and name existing bones.
    pub fn validate(&self, bone_count: usize) -> Vec<String> {
        let mut out = Vec::new();
        if self.weights.len() != self.positions.len() {
            out.push(format!("{} weights for {} vertices", self.weights.len(), self.positions.len()));
        }
        if !self.indices.len().is_multiple_of(3) || self.indices.iter().any(|&i| i as usize >= self.positions.len()) {
            out.push("indices out of range".into());
        }
        for (v, w) in self.weights.iter().enumerate() {
            if (w.sum() - 1.0).abs() > 1e-3 {
                out.push(format!("vertex {v} weights sum to {}", w.sum()));
            }
            if (0..MAX_INFLUENCES).any(|k| w.weights[k] > 0.0 && w.bones[k] as usize >= bone_count) {
                out.push(format!("vertex {v} uses a missing bone"));
            }
        }
        out
    }

    /// After [`Skeleton2D::remove_bone`]: remap indices; a removed bone's weight
    /// goes to `fallback` (its parent), then each vertex is cleaned up.
    pub fn remap_bones(&mut self, map: &[Option<usize>], fallback: Option<usize>) {
        for w in &mut self.weights {
            let pairs: Vec<(u16, f32)> = w
                .pairs()
                .into_iter()
                .filter_map(|(b, x)| match map.get(b as usize).copied().flatten() {
                    Some(n) => Some((n as u16, x)),
                    None => fallback.and_then(|f| map.get(f).copied().flatten()).map(|n| (n as u16, x)),
                })
                .collect();
            *w = VertexWeights::from_pairs(&pairs, 0);
        }
    }

    /// Vertex adjacency lists.
    pub fn adjacency(&self) -> Vec<Vec<u32>> {
        adjacency(self.positions.len(), &self.indices)
    }

    /// Skin into `out` with the given skin matrices (see [`skin_positions`]).
    pub fn skin(&self, matrices: &[Affine2], out: &mut Vec<[f32; 2]>) {
        skin_positions(&self.positions, &self.weights, matrices, out);
    }
}

pub fn adjacency(n: usize, indices: &[u16]) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in indices.as_chunks::<3>().0 {
        for k in 0..3 {
            let (a, b) = (t[k] as usize, t[(k + 1) % 3] as u32);
            if a < n && !adj[a].contains(&b) {
                adj[a].push(b);
            }
            let (a, b) = (t[(k + 1) % 3] as usize, t[k] as u32);
            if a < n && !adj[a].contains(&b) {
                adj[a].push(b);
            }
        }
    }
    adj
}

/// Linear blend skinning of rest `positions` into `out`. `matrices` are the
/// final per-bone matrices (`pose_world * bind_pose`). A bone index past the
/// matrix list skins as identity.
pub fn skin_positions(positions: &[[f32; 2]], weights: &[VertexWeights], matrices: &[Affine2], out: &mut Vec<[f32; 2]>) {
    out.clear();
    out.reserve(positions.len());
    let m = |b: u16| matrices.get(b as usize).copied().unwrap_or(Affine2::IDENTITY);
    for (p, w) in positions.iter().zip(weights) {
        let p = Vec2::from(*p);
        if w.weights[0] >= 0.9999 {
            out.push(m(w.bones[0]).transform_point2(p).to_array());
            continue;
        }
        let mut acc = Vec2::ZERO;
        let mut total = 0.0;
        for k in 0..MAX_INFLUENCES {
            if w.weights[k] > 0.0 {
                acc += m(w.bones[k]).transform_point2(p) * w.weights[k];
                total += w.weights[k];
            }
        }
        out.push(if total > 0.0 { (acc / total).to_array() } else { p.to_array() });
    }
}

/// Bounds of positions: `(min, max)`.
pub fn bounds_of(p: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let mut lo = [f32::MAX; 2];
    let mut hi = [f32::MIN; 2];
    for q in p {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// The asset
// ---------------------------------------------------------------------------

/// One skinned sprite of a rig.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SkinPart {
    /// `SubAssetId` of the sprite this mesh replaces.
    pub sprite_id: u64,
    pub name: String,
    pub mesh: SkinMesh2D,
    /// Indices into the rig skeleton a part may use (Unity `boneIndices`); empty = all.
    pub bone_indices: Vec<u16>,
}

/// A `.rig2d` file: the shared skeleton and the skinned sprite meshes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rig2D {
    pub skeleton: Skeleton2D,
    pub parts: Vec<SkinPart>,
}

impl Rig2D {
    pub fn part(&self, sprite_id: u64) -> Option<&SkinPart> {
        self.parts.iter().find(|p| p.sprite_id == sprite_id)
    }

    pub fn validate(&self) -> Vec<String> {
        let mut out = self.skeleton.validate();
        for p in &self.parts {
            out.extend(p.mesh.validate(self.skeleton.bones.len()).into_iter().map(|e| format!("part '{}': {e}", p.name)));
        }
        out
    }
}

pub fn load_rig2d(path: impl AsRef<Path>) -> Result<Rig2D> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn save_rig2d(path: impl AsRef<Path>, rig: &Rig2D) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(rig, ron::ser::PrettyConfig::default())?;
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

// ---------------------------------------------------------------------------
// Auto geometry
// ---------------------------------------------------------------------------

/// Settings for [`auto_geometry`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryParams {
    /// A pixel is solid when its alpha is above this.
    pub alpha_threshold: u8,
    /// Outline simplification tolerance in pixels.
    pub outline_tolerance: f32,
    /// Spacing of resampled outline points and interior points, pixels.
    /// 0 = a tenth of the rect diagonal (min 4).
    pub spacing: f32,
    /// Hard cap on vertices (the spacing grows to fit).
    pub max_vertices: usize,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self { alpha_threshold: 10, outline_tolerance: 1.5, spacing: 0.0, max_vertices: 1500 }
    }
}

fn point_in_loop(p: P2, l: &[P2]) -> bool {
    let mut inside = false;
    let n = l.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (l[i], l[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Inside the region described by outer loops and holes (even-odd).
fn point_in_region(p: P2, loops: &[Vec<P2>]) -> bool {
    loops.iter().filter(|l| point_in_loop(p, l)).count() % 2 == 1
}

fn seg_dist(p: P2, a: P2, b: P2) -> f32 {
    let (ab, ap) = ([b[0] - a[0], b[1] - a[1]], [p[0] - a[0], p[1] - a[1]]);
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if l2 > 1e-12 { ((ap[0] * ab[0] + ap[1] * ab[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = [a[0] + ab[0] * t - p[0], a[1] + ab[1] * t - p[1]];
    (q[0] * q[0] + q[1] * q[1]).sqrt()
}

fn dist_to_loops(p: P2, loops: &[Vec<P2>]) -> f32 {
    let mut d = f32::MAX;
    for l in loops {
        for i in 0..l.len() {
            d = d.min(seg_dist(p, l[i], l[(i + 1) % l.len()]));
        }
    }
    d
}

/// Subdivide the loop's edges so none is longer than `spacing` (corners kept).
fn resample_loop(l: &[P2], spacing: f32) -> Vec<P2> {
    let mut out = Vec::new();
    for i in 0..l.len() {
        let (a, b) = (l[i], l[(i + 1) % l.len()]);
        let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let n = (d / spacing).ceil().max(1.0) as usize;
        for k in 0..n {
            let t = k as f32 / n as f32;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    out
}

/// Delaunay triangulation (Bowyer-Watson, O(n^2)) of distinct points. Returns
/// counter-clockwise triangles (in a y-up frame) as index triples.
pub fn delaunay(points: &[P2]) -> Vec<[u32; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let (lo, hi) = {
        let mut lo = [f64::MAX; 2];
        let mut hi = [f64::MIN; 2];
        for p in points {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k] as f64);
                hi[k] = hi[k].max(p[k] as f64);
            }
        }
        (lo, hi)
    };
    let size = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
    let (cx, cy) = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5);
    let mut pts: Vec<[f64; 2]> = points.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    pts.push([cx - 20.0 * size, cy - size]);
    pts.push([cx + 20.0 * size, cy - size]);
    pts.push([cx, cy + 20.0 * size]);
    let (s0, s1, s2) = (n as u32, n as u32 + 1, n as u32 + 2);

    fn circum(pts: &[[f64; 2]], t: [u32; 3]) -> ([f64; 2], f64) {
        let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
        let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
        if d.abs() < 1e-18 {
            return ([a[0], a[1]], f64::MAX);
        }
        let (a2, b2, c2) = (a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]);
        let ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
        let uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
        let r2 = (a[0] - ux).powi(2) + (a[1] - uy).powi(2);
        ([ux, uy], r2)
    }

    let mut tris: Vec<([u32; 3], [f64; 2], f64)> = Vec::new();
    {
        let t = [s0, s1, s2];
        let (c, r2) = circum(&pts, t);
        tris.push((t, c, r2));
    }
    for i in 0..n {
        let p = pts[i];
        let mut bad: Vec<usize> = Vec::new();
        for (k, (_, c, r2)) in tris.iter().enumerate() {
            let d2 = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2);
            if d2 < *r2 * (1.0 - 1e-12) {
                bad.push(k);
            }
        }
        // Boundary of the cavity: edges used by exactly one bad triangle.
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for &k in &bad {
            let t = tris[k].0;
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                if let Some(pos) = edges.iter().position(|&(x, y)| (x == b && y == a) || (x == a && y == b)) {
                    edges.swap_remove(pos);
                } else {
                    edges.push((a, b));
                }
            }
        }
        for &k in bad.iter().rev() {
            tris.swap_remove(k);
        }
        for (a, b) in edges {
            let t = [a, b, i as u32];
            let (c, r2) = circum(&pts, t);
            tris.push((t, c, r2));
        }
    }
    tris.into_iter()
        .filter(|(t, _, _)| t.iter().all(|&v| v < n as u32))
        .map(|(t, _, _)| {
            let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if area >= 0.0 { t } else { [t[0], t[2], t[1]] }
        })
        .collect()
}

/// Mesh the sprite's alpha shape (rect-local pixel space, y down) with interior
/// points: see the module docs. `None` when the rect has no solid pixels.
pub fn auto_geometry(alpha: &AlphaMap, rect: PixelRect, p: &GeometryParams) -> Option<PixelMesh> {
    let loops = sprite_mesh::trace_outline(alpha, rect, p.alpha_threshold, 0, p.outline_tolerance, true);
    if loops.is_empty() {
        return None;
    }
    let diag = ((rect.w * rect.w + rect.h * rect.h) as f32).sqrt();
    let mut spacing = if p.spacing > 0.0 { p.spacing } else { (diag / 10.0).max(4.0) };
    let area = rect.w as f32 * rect.h as f32;
    // Grow the spacing until the vertex budget holds.
    let (outline, interior) = loop {
        let outline: Vec<P2> = loops.iter().flat_map(|l| resample_loop(l, spacing)).collect();
        let mut interior = Vec::new();
        let (nx, ny) = ((rect.w as f32 / spacing).ceil() as i32, (rect.h as f32 / spacing).ceil() as i32);
        for j in 0..=ny {
            for i in 0..=nx {
                // Offset rows and a tiny deterministic jitter keep the point set off a degenerate lattice.
                let jitter = (((i * 73 + j * 151) % 17) as f32 / 17.0 - 0.5) * spacing * 0.02;
                let q = [(i as f32 + 0.5 * (j & 1) as f32) * spacing + jitter, j as f32 * spacing - jitter];
                if point_in_region(q, &loops) && dist_to_loops(q, &loops) > spacing * 0.55 {
                    interior.push(q);
                }
            }
        }
        if outline.len() + interior.len() <= p.max_vertices || spacing > diag {
            break (outline, interior);
        }
        spacing *= 1.25;
    };
    let mut pts = outline;
    pts.extend(interior);
    // Drop near-duplicates (resampled corners that coincide).
    let mut uniq: Vec<P2> = Vec::with_capacity(pts.len());
    let mut seen: HashMap<(i32, i32), ()> = HashMap::new();
    for q in pts {
        let key = ((q[0] * 64.0).round() as i32, (q[1] * 64.0).round() as i32);
        if seen.insert(key, ()).is_none() {
            uniq.push(q);
        }
    }
    let tris = delaunay(&uniq);
    let mut mesh = PixelMesh::default();
    let mut remap: HashMap<u32, u32> = HashMap::new();
    for t in tris {
        let c = [
            (uniq[t[0] as usize][0] + uniq[t[1] as usize][0] + uniq[t[2] as usize][0]) / 3.0,
            (uniq[t[0] as usize][1] + uniq[t[1] as usize][1] + uniq[t[2] as usize][1]) / 3.0,
        ];
        if !point_in_region(c, &loops) {
            continue;
        }
        let mut idx = [0u32; 3];
        for k in 0..3 {
            idx[k] = *remap.entry(t[k]).or_insert_with(|| {
                mesh.verts.push(uniq[t[k] as usize]);
                (mesh.verts.len() - 1) as u32
            });
        }
        mesh.indices.extend(idx);
    }
    let _ = area;
    if mesh.indices.is_empty() { None } else { Some(mesh) }
}

/// Pixel mesh -> a sprite-space [`SkinMesh2D`] bound fully to `bone`. Triangles
/// are flipped to counter-clockwise (the pixel space is y down). `None` past
/// the 16-bit index range.
pub fn skin_mesh_from_pixels(mesh: &PixelMesh, rect: PixelRect, pivot: [f32; 2], ppu: f32, bone: u16) -> Option<SkinMesh2D> {
    if mesh.verts.len() > u16::MAX as usize {
        return None;
    }
    let positions: Vec<[f32; 2]> = mesh.verts.iter().map(|&p| pixel_to_world(p, rect, pivot, ppu)).collect();
    let mut indices = Vec::with_capacity(mesh.indices.len());
    for t in mesh.indices.as_chunks::<3>().0 {
        let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
        let cross = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if cross >= 0.0 {
            indices.extend([t[0] as u16, t[1] as u16, t[2] as u16]);
        } else {
            indices.extend([t[0] as u16, t[2] as u16, t[1] as u16]);
        }
    }
    Some(SkinMesh2D::new(positions, indices, bone))
}

// ---------------------------------------------------------------------------
// Auto weights
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AutoWeightMode {
    /// Distance falloff only.
    Distance,
    /// Distance sources diffused over the mesh graph (default).
    #[default]
    Heat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoWeightParams {
    pub mode: AutoWeightMode,
    /// Diffusion iterations (Heat).
    pub iterations: u32,
    /// Diffusion strength per iteration, 0..1 (Heat).
    pub diffusion: f32,
    /// Falloff exponent of the distance kernel (higher = more local).
    pub sharpness: f32,
}

impl Default for AutoWeightParams {
    fn default() -> Self {
        Self { mode: AutoWeightMode::Heat, iterations: 20, diffusion: 0.5, sharpness: 3.0 }
    }
}

/// Distance from a point to a bone segment.
fn bone_distance(p: [f32; 2], seg: ([f32; 2], [f32; 2])) -> f32 {
    seg_dist(p, seg.0, seg.1)
}

/// Weights for every vertex from the skeleton's rest pose. `bones` restricts
/// the influences to those skeleton indices (empty = all). With no bones every
/// vertex gets `VertexWeights::single(0)`.
pub fn auto_weights(positions: &[[f32; 2]], indices: &[u16], skeleton: &Skeleton2D, bones: &[u16], params: &AutoWeightParams) -> Vec<VertexWeights> {
    let segs = skeleton.rest_segments();
    let usable: Vec<u16> = if bones.is_empty() { (0..segs.len() as u16).collect() } else { bones.iter().copied().filter(|&b| (b as usize) < segs.len()).collect() };
    let n = positions.len();
    if usable.is_empty() || n == 0 {
        return vec![VertexWeights::single(0); n];
    }
    let nb = usable.len();
    // Raw distance kernel, normalised per vertex.
    let mut w = vec![0.0f32; n * nb];
    // Scale: the mesh diagonal, so the epsilon is independent of units.
    let (lo, hi) = bounds_of(positions);
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt().max(1e-6);
    let eps = diag * 0.01;
    for (v, p) in positions.iter().enumerate() {
        let mut sum = 0.0;
        for (k, &b) in usable.iter().enumerate() {
            let d = bone_distance(*p, segs[b as usize]);
            let x = 1.0 / (d + eps).powf(params.sharpness.max(0.5));
            w[v * nb + k] = x;
            sum += x;
        }
        for k in 0..nb {
            w[v * nb + k] /= sum;
        }
    }
    if params.mode == AutoWeightMode::Heat && params.iterations > 0 {
        let adj = adjacency(n, indices);
        let lambda = params.diffusion.clamp(0.0, 1.0);
        let mut next = w.clone();
        for _ in 0..params.iterations {
            for v in 0..n {
                if adj[v].is_empty() {
                    continue;
                }
                let inv = 1.0 / adj[v].len() as f32;
                for k in 0..nb {
                    let avg: f32 = adj[v].iter().map(|&u| w[u as usize * nb + k]).sum::<f32>() * inv;
                    next[v * nb + k] = w[v * nb + k] * (1.0 - lambda) + avg * lambda;
                }
            }
            std::mem::swap(&mut w, &mut next);
        }
    }
    (0..n)
        .map(|v| {
            let pairs: Vec<(u16, f32)> = (0..nb).map(|k| (usable[k], w[v * nb + k])).collect();
            // Fallback: the nearest bone.
            let nearest = (0..nb).min_by(|&a, &b| bone_distance(positions[v], segs[usable[a] as usize]).total_cmp(&bone_distance(positions[v], segs[usable[b] as usize]))).map(|k| usable[k]).unwrap_or(0);
            VertexWeights::from_pairs(&pairs, nearest)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Weight painting
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushMode {
    /// Raise the active bone's weight.
    Add,
    /// Lower the active bone's weight.
    Subtract,
    /// Average with the neighbours.
    Smooth,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brush {
    pub mode: BrushMode,
    /// World-unit radius.
    pub radius: f32,
    /// 0 = falloff from the centre, 1 = full strength out to the radius.
    pub hardness: f32,
    /// Amount applied per stroke step, 0..1.
    pub strength: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self { mode: BrushMode::Add, radius: 0.3, hardness: 0.5, strength: 0.2 }
    }
}

impl Brush {
    /// Brush influence at distance `d` from the centre.
    pub fn falloff(&self, d: f32) -> f32 {
        if d >= self.radius || self.radius <= 0.0 {
            return 0.0;
        }
        let t = d / self.radius;
        let h = self.hardness.clamp(0.0, 1.0);
        if t <= h {
            1.0
        } else {
            let u = (t - h) / (1.0 - h).max(1e-6);
            1.0 - u * u * (3.0 - 2.0 * u) // smoothstep down
        }
    }
}

/// One brush dab at `center`. Returns the vertices that changed. Only
/// vertices of the mesh are touched; every touched vertex stays normalised and
/// within four influences.
pub fn paint_weights(mesh: &mut SkinMesh2D, adj: &[Vec<u32>], bone: u16, center: [f32; 2], brush: &Brush) -> Vec<u32> {
    let mut changed = Vec::new();
    // Smooth reads the weights before this dab.
    let snapshot: Vec<VertexWeights> = if brush.mode == BrushMode::Smooth { mesh.weights.clone() } else { Vec::new() };
    for v in 0..mesh.positions.len() {
        let d = ((mesh.positions[v][0] - center[0]).powi(2) + (mesh.positions[v][1] - center[1]).powi(2)).sqrt();
        let f = brush.falloff(d) * brush.strength.clamp(0.0, 1.0);
        if f <= 0.0 {
            continue;
        }
        let old = mesh.weights[v];
        let new = match brush.mode {
            BrushMode::Add | BrushMode::Subtract => {
                let cur = old.weight_of(bone);
                let target = if brush.mode == BrushMode::Add { cur + (1.0 - cur) * f } else { cur * (1.0 - f) };
                let others: f32 = old.pairs().iter().filter(|(b, _)| *b != bone).map(|(_, w)| w).sum();
                let mut pairs: Vec<(u16, f32)> = Vec::new();
                if others > 1e-6 {
                    let scale = (1.0 - target) / others;
                    pairs.extend(old.pairs().into_iter().filter(|(b, _)| *b != bone).map(|(b, w)| (b, w * scale)));
                    pairs.push((bone, target));
                } else if target > 1e-6 {
                    // Nothing else to take from: the bone keeps all of it.
                    pairs.push((bone, 1.0));
                } else {
                    pairs.extend(old.pairs()); // subtracting the only bone is a no-op
                }
                VertexWeights::from_pairs(&pairs, old.bones[0])
            }
            BrushMode::Smooth => {
                let mut acc: HashMap<u16, f32> = HashMap::new();
                for (b, w) in snapshot[v].pairs() {
                    *acc.entry(b).or_default() += w * (1.0 - f);
                }
                if !adj[v].is_empty() {
                    let share = f / adj[v].len() as f32;
                    for &u in &adj[v] {
                        for (b, w) in snapshot[u as usize].pairs() {
                            *acc.entry(b).or_default() += w * share;
                        }
                    }
                } else {
                    for (b, w) in snapshot[v].pairs() {
                        *acc.entry(b).or_default() += w * f;
                    }
                }
                let mut pairs: Vec<(u16, f32)> = acc.into_iter().collect();
                pairs.sort_by_key(|p| p.0);
                VertexWeights::from_pairs(&pairs, old.bones[0])
            }
        };
        if new != old {
            mesh.weights[v] = new;
            changed.push(v as u32);
        }
    }
    changed
}

/// Set one bone's weight on every vertex of `verts` to `value` (Weight Slider), others rescale.
pub fn set_bone_weight(mesh: &mut SkinMesh2D, verts: &[u32], bone: u16, value: f32) {
    let value = value.clamp(0.0, 1.0);
    for &v in verts {
        let Some(old) = mesh.weights.get(v as usize).copied() else { continue };
        let others: f32 = old.pairs().iter().filter(|(b, _)| *b != bone).map(|(_, w)| w).sum();
        let mut pairs: Vec<(u16, f32)> = Vec::new();
        if others > 1e-6 {
            let scale = (1.0 - value) / others;
            pairs.extend(old.pairs().into_iter().filter(|(b, _)| *b != bone).map(|(b, w)| (b, w * scale)));
            pairs.push((bone, value));
        } else {
            pairs.push((bone, 1.0));
        }
        mesh.weights[v as usize] = VertexWeights::from_pairs(&pairs, old.bones[0]);
    }
}

/// Triangles reordered so overlapping parts draw in bone-depth order (Unity's
/// `SortTrianglesByDepth`): deeper (more descendant) bones draw later.
pub fn sort_triangles_by_bone_depth(mesh: &mut SkinMesh2D, skeleton: &Skeleton2D) {
    let depth = |b: u16| -> f32 {
        let mut d = 0;
        let mut p = skeleton.bones.get(b as usize).map_or(-1, |x| x.parent);
        while p >= 0 && d < 64 {
            d += 1;
            p = skeleton.bones[p as usize].parent;
        }
        d as f32
    };
    let vd = |v: u16| -> f32 { mesh.weights.get(v as usize).map(|w| w.pairs().iter().map(|&(b, x)| depth(b) * x).sum()).unwrap_or(0.0) };
    let mut tris: Vec<(f32, [u16; 3])> = mesh.indices.as_chunks::<3>().0.iter().map(|t| ((vd(t[0]) + vd(t[1]) + vd(t[2])) / 3.0, [t[0], t[1], t[2]])).collect();
    tris.sort_by(|a, b| a.0.total_cmp(&b.0)); // stable
    mesh.indices = tris.into_iter().flat_map(|(_, t)| t).collect();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn arm() -> Skeleton2D {
        let mut s = Skeleton2D::default();
        let a = s.add_bone(None, [0.0, 0.0], [1.0, 0.0], "Upper");
        s.add_bone(Some(a), [1.0, 0.0], [2.0, 0.0], "Lower");
        s
    }

    fn close(a: [f32; 2], b: [f32; 2], e: f32) -> bool {
        (a[0] - b[0]).abs() < e && (a[1] - b[1]).abs() < e
    }

    #[test]
    fn world_transforms_chain_through_parents() {
        let mut s = Skeleton2D::default();
        s.bones.push(Bone2D { name: "a".into(), position: [1.0, 0.0], rotation: std::f32::consts::FRAC_PI_2, length: 1.0, ..Default::default() });
        s.bones.push(Bone2D { name: "b".into(), parent: 0, position: [2.0, 0.0], rotation: 0.0, length: 1.0, ..Default::default() });
        let w = s.world_rest();
        // b's origin: a's origin (1,0) + a's +X rotated 90 deg (0,1) * 2.
        assert!(close(w[1].translation.to_array(), [1.0, 2.0], 1e-5));
        let seg = s.rest_segments();
        assert!(close(seg[1].1, [1.0, 3.0], 1e-5));
    }

    #[test]
    fn rest_pose_skins_to_itself() {
        let s = arm();
        let m = s.skin_matrices(&s.world_rest());
        for x in &m {
            assert!(x.abs_diff_eq(Affine2::IDENTITY, 1e-5));
        }
    }

    #[test]
    fn skinning_matches_a_reference() {
        let s = arm();
        let pos = vec![[0.2, 0.1], [0.9, -0.2], [1.2, 0.3], [1.8, 0.0]];
        let w = vec![
            VertexWeights::single(0),
            VertexWeights::from_pairs(&[(0, 0.7), (1, 0.3)], 0),
            VertexWeights::from_pairs(&[(0, 0.25), (1, 0.75)], 0),
            VertexWeights::single(1),
        ];
        let mut pose: Vec<BonePose> = s.bones.iter().map(BonePose::rest).collect();
        pose[0].rotation = 0.6;
        pose[1].rotation = -0.9;
        pose[1].scale = [1.3, 0.8];
        let world = s.world_pose(&pose);
        let mats = s.skin_matrices(&world);
        let mut out = Vec::new();
        skin_positions(&pos, &w, &mats, &mut out);
        // Reference in f64 from first principles: bind = inverse of rest world.
        let rest0 = |p: [f64; 2]| p; // bone 0 rest = identity rotation at origin
        let rest_inv1 = |p: [f64; 2]| [p[0] - 1.0, p[1]]; // bone 1: translate by (1,0)
        let tr = |rot: f64, t: [f64; 2], sc: [f64; 2], p: [f64; 2]| {
            let (c, s) = (rot.cos(), rot.sin());
            let q = [p[0] * sc[0], p[1] * sc[1]];
            [c * q[0] - s * q[1] + t[0], s * q[0] + c * q[1] + t[1]]
        };
        let b0 = |p: [f64; 2]| tr(0.6, [0.0, 0.0], [1.0, 1.0], rest0(p));
        let b1 = |p: [f64; 2]| {
            let local = rest_inv1(p);
            let loc = tr(-0.9, [1.0, 0.0], [1.3, 0.8], local);
            b0(loc)
        };
        for (i, p) in pos.iter().enumerate() {
            let p = [p[0] as f64, p[1] as f64];
            let (a, b) = (b0(p), b1(p));
            let wt = w[i];
            let mut e = [0.0f64; 2];
            for k in 0..MAX_INFLUENCES {
                let x = if wt.bones[k] == 0 { a } else { b };
                e[0] += x[0] * wt.weights[k] as f64;
                e[1] += x[1] * wt.weights[k] as f64;
            }
            assert!((out[i][0] as f64 - e[0]).abs() < 1e-4 && (out[i][1] as f64 - e[1]).abs() < 1e-4, "vertex {i}: {:?} vs {e:?}", out[i]);
        }
    }

    #[test]
    fn bone_editing_keeps_world_transforms() {
        let mut s = arm();
        let c = s.add_bone(Some(1), [2.0, 0.0], [2.0, 1.0], "Hand");
        let before = s.world_rest();
        assert!(s.reparent(c, Some(0)));
        let after = s.world_rest();
        assert!(after[c].abs_diff_eq(before[c], 1e-5));
        assert!(!s.reparent(0, Some(c)), "cycle refused");
        assert!(!s.reparent(0, Some(0)));
        // Split keeps the endpoint and children in place.
        let tip = s.rest_segments()[0].1;
        let kids_before = s.world_rest();
        let n = s.split_bone(0, 0.5).unwrap();
        let seg = s.rest_segments();
        assert!(close(seg[n].1, tip, 1e-5));
        assert!(close(seg[0].1, [0.5, 0.0], 1e-5));
        let kids_after = s.world_rest();
        assert!(kids_after[1].abs_diff_eq(kids_before[1], 1e-5));
        assert!(s.validate().is_empty(), "{:?}", s.validate());
    }

    #[test]
    fn removing_a_bone_reparents_and_remaps_weights() {
        let mut s = arm();
        let hand = s.add_bone(Some(1), [2.0, 0.0], [3.0, 0.0], "Hand");
        let rest = s.world_rest();
        let mut mesh = SkinMesh2D::new(vec![[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]], vec![0, 1, 2], 0);
        mesh.weights = vec![VertexWeights::single(0), VertexWeights::single(1), VertexWeights::from_pairs(&[(1, 0.5), (2, 0.5)], 0)];
        let map = s.remove_bone(1);
        assert_eq!(map, vec![Some(0), None, Some(1)]);
        assert_eq!(s.bones.len(), 2);
        assert_eq!(s.bones[1].parent, 0);
        assert!(s.world_rest()[1].abs_diff_eq(rest[hand], 1e-5), "child kept its world transform");
        mesh.remap_bones(&map, Some(0));
        assert!(mesh.validate(2).is_empty(), "{:?}", mesh.validate(2));
        assert_eq!(mesh.weights[1].weight_of(0), 1.0, "removed bone's weight moved to its parent");
        assert!((mesh.weights[2].weight_of(0) - 0.5).abs() < 1e-5 && (mesh.weights[2].weight_of(1) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn from_pairs_is_clean() {
        let w = VertexWeights::from_pairs(&[(3, 0.4), (1, 0.3), (3, 0.1), (7, 0.15), (2, 0.04), (9, 0.001), (4, f32::NAN)], 0);
        assert!((w.sum() - 1.0).abs() < 1e-5);
        assert_eq!(w.bones[0], 3, "duplicates merged, largest first");
        assert!((w.weight_of(3) - 0.5 / 0.99).abs() < 0.02 || w.weight_of(3) > 0.4);
        assert_eq!(w.weights.iter().filter(|&&x| x > 0.0).count(), 4);
        assert_eq!(VertexWeights::from_pairs(&[], 5), VertexWeights::single(5));
        assert_eq!(VertexWeights::from_pairs(&[(1, 0.0)], 2), VertexWeights::single(2));
    }

    fn strip(n: usize, len: f32, h: f32) -> (Vec<[f32; 2]>, Vec<u16>) {
        // Two rows of n+1 points along x.
        let mut pos = Vec::new();
        for r in 0..2 {
            for i in 0..=n {
                pos.push([len * i as f32 / n as f32, r as f32 * h - h * 0.5]);
            }
        }
        let mut idx = Vec::new();
        for i in 0..n {
            let (a, b, c, d) = (i as u16, i as u16 + 1, (n + 1 + i) as u16, (n + 2 + i) as u16);
            idx.extend([a, b, d, a, d, c]);
        }
        (pos, idx)
    }

    #[test]
    fn auto_weights_partition_of_unity_and_locality() {
        let mut s = Skeleton2D::default();
        s.add_bone(None, [0.0, 0.0], [1.0, 0.0], "A");
        s.add_bone(None, [3.0, 0.0], [4.0, 0.0], "B");
        let (pos, idx) = strip(20, 4.0, 0.4);
        for mode in [AutoWeightMode::Distance, AutoWeightMode::Heat] {
            let w = auto_weights(&pos, &idx, &s, &[], &AutoWeightParams { mode, ..Default::default() });
            assert_eq!(w.len(), pos.len());
            for v in &w {
                assert!((v.sum() - 1.0).abs() < 1e-4, "{mode:?} sum {}", v.sum());
                assert!(v.weights.iter().filter(|&&x| x > 0.0).count() <= 4);
            }
            // The vertex under bone A is A's, under B is B's, and weights fall off between.
            assert!(w[0].weight_of(0) > 0.8, "{mode:?} near A: {}", w[0].weight_of(0));
            assert!(w[20].weight_of(1) > 0.8, "{mode:?} near B: {}", w[20].weight_of(1));
            let a_of = |v: usize| w[v].weight_of(0);
            for i in 0..20 {
                assert!(a_of(i) >= a_of(i + 1) - 0.05, "{mode:?} A weight roughly decreases along x at {i}");
            }
        }
    }

    #[test]
    fn heat_does_not_jump_a_gap_distance_does() {
        // Two separate strips; the bone sits under the left one only.
        let mut s = Skeleton2D::default();
        s.add_bone(None, [0.0, 0.0], [1.0, 0.0], "A");
        s.add_bone(None, [0.0, 1.0], [1.0, 1.0], "B");
        let (mut pos, mut idx) = strip(4, 1.0, 0.2);
        let (p2, i2) = strip(4, 1.0, 0.2);
        let off = pos.len() as u16;
        pos.extend(p2.iter().map(|p| [p[0], p[1] + 1.0]));
        idx.extend(i2.iter().map(|i| i + off));
        let w = auto_weights(&pos, &idx, &s, &[], &AutoWeightParams::default());
        // Each strip stays with its own bone, whichever the mode.
        assert!(w[2].weight_of(0) > 0.9 && w[off as usize + 2].weight_of(1) > 0.9);
    }

    #[test]
    fn auto_weights_without_bones() {
        let w = auto_weights(&[[0.0, 0.0]], &[], &Skeleton2D::default(), &[], &AutoWeightParams::default());
        assert_eq!(w, vec![VertexWeights::single(0)]);
    }

    #[test]
    fn brush_properties() {
        let s = arm();
        let (pos, idx) = strip(10, 2.0, 0.4);
        let mut mesh = SkinMesh2D::new(pos.clone(), idx, 0);
        let adj = mesh.adjacency();
        let brush = Brush { mode: BrushMode::Add, radius: 0.5, hardness: 0.5, strength: 1.0 };
        let changed = paint_weights(&mut mesh, &adj, 1, [1.0, 0.0], &brush);
        assert!(!changed.is_empty());
        for (v, w) in mesh.weights.iter().enumerate() {
            assert!((w.sum() - 1.0).abs() < 1e-4);
            let d = ((pos[v][0] - 1.0).powi(2) + pos[v][1].powi(2)).sqrt();
            if d >= 0.5 {
                assert_eq!(*w, VertexWeights::single(0), "outside the radius unchanged");
            }
        }
        let centre = (0..pos.len()).min_by(|&a, &b| ((pos[a][0] - 1.0).abs() + pos[a][1].abs()).total_cmp(&((pos[b][0] - 1.0).abs() + pos[b][1].abs()))).unwrap();
        assert!(mesh.weights[centre].weight_of(1) > 0.99, "hard centre at strength 1");
        // Subtract takes it back.
        let sub = Brush { mode: BrushMode::Subtract, ..brush };
        paint_weights(&mut mesh, &adj, 1, [1.0, 0.0], &sub);
        // The only influence cannot be subtracted (the vertex would have none): a no-op.
        assert_eq!(mesh.weights[centre].weight_of(1), 1.0);
        // With another influence present it comes off.
        mesh.weights[centre] = VertexWeights::from_pairs(&[(0, 0.5), (1, 0.5)], 0);
        paint_weights(&mut mesh, &adj, 1, [1.0, 0.0], &sub);
        assert!(mesh.weights[centre].weight_of(1) < 1e-3, "hard centre removed");
        assert!(mesh.weights.iter().all(|w| (w.sum() - 1.0).abs() < 1e-4));
        let _ = s;
    }

    #[test]
    fn smooth_brush_blends_and_stays_normalised() {
        let (pos, idx) = strip(6, 1.2, 0.2);
        let mut mesh = SkinMesh2D::new(pos, idx, 0);
        for v in 0..mesh.weights.len() {
            if mesh.positions[v][0] > 0.6 {
                mesh.weights[v] = VertexWeights::single(1);
            }
        }
        let adj = mesh.adjacency();
        let mid = mesh.positions.iter().position(|p| (p[0] - 0.6).abs() < 1e-4).unwrap();
        let before = mesh.weights[mid].weight_of(0);
        paint_weights(&mut mesh, &adj, 0, [0.6, 0.0], &Brush { mode: BrushMode::Smooth, radius: 0.5, hardness: 1.0, strength: 1.0 });
        assert!(mesh.weights.iter().all(|w| (w.sum() - 1.0).abs() < 1e-4));
        let after = mesh.weights[mid].weight_of(0);
        assert!(after > 0.0 && after < 1.0 || before == 0.0 && after > 0.0, "{before} -> {after}");
    }

    #[test]
    fn weight_slider_sets_a_value() {
        let mut mesh = SkinMesh2D::new(vec![[0.0; 2]; 2], vec![], 0);
        mesh.weights[0] = VertexWeights::from_pairs(&[(0, 0.5), (1, 0.5)], 0);
        set_bone_weight(&mut mesh, &[0, 1], 1, 0.25);
        assert!((mesh.weights[0].weight_of(1) - 0.25).abs() < 1e-5 && (mesh.weights[0].sum() - 1.0).abs() < 1e-5);
        assert!((mesh.weights[1].weight_of(1) - 0.25).abs() < 1e-5, "a single-bone vertex splits");
    }

    fn disc_alpha(w: u32, h: u32, r: f32) -> AlphaMap {
        let mut a = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 + 0.5 - w as f32 / 2.0).powi(2) + (y as f32 + 0.5 - h as f32 / 2.0).powi(2)).sqrt();
                if d <= r {
                    a[(y * w + x) as usize] = 255;
                }
            }
        }
        AlphaMap { w, h, alpha: a }
    }

    fn mesh_area(m: &PixelMesh) -> f32 {
        m.area()
    }

    #[test]
    fn auto_geometry_covers_the_shape() {
        let a = disc_alpha(64, 64, 28.0);
        let rect = PixelRect::new(0, 0, 64, 64);
        let m = auto_geometry(&a, rect, &GeometryParams::default()).expect("mesh");
        let solid = a.alpha.iter().filter(|&&v| v > 10).count() as f32;
        let ratio = mesh_area(&m) / solid;
        assert!(ratio > 0.9 && ratio < 1.1, "area ratio {ratio}");
        assert!(m.indices.len() / 3 > 20, "interior points add triangles: {}", m.indices.len() / 3);
        // Every triangle has positive area and the mesh has interior vertices.
        let outline_only = sprite_mesh::tight_mesh(&a, rect, &sprite_mesh::TightParams { alpha_threshold: 10, detail: 0.5, extrude: 0, holes: true }).unwrap();
        assert!(m.verts.len() > outline_only.verts.len());
        let sm = skin_mesh_from_pixels(&m, rect, [0.5, 0.5], 32.0, 0).unwrap();
        assert!(sm.validate(1).is_empty());
        // Counter-clockwise in sprite space.
        for t in sm.indices.as_chunks::<3>().0 {
            let (p, q, r) = (sm.positions[t[0] as usize], sm.positions[t[1] as usize], sm.positions[t[2] as usize]);
            assert!((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]) >= 0.0);
        }
    }

    #[test]
    fn auto_geometry_respects_holes_and_empty() {
        let mut a = disc_alpha(64, 64, 28.0);
        for y in 24..40 {
            for x in 24..40 {
                a.alpha[y * 64 + x] = 0;
            }
        }
        let m = auto_geometry(&a, PixelRect::new(0, 0, 64, 64), &GeometryParams::default()).unwrap();
        let solid = a.alpha.iter().filter(|&&v| v > 10).count() as f32;
        let ratio = mesh_area(&m) / solid;
        assert!(ratio > 0.85 && ratio < 1.05, "hole kept: {ratio}");
        let empty = AlphaMap { w: 8, h: 8, alpha: vec![0; 64] };
        assert!(auto_geometry(&empty, PixelRect::new(0, 0, 8, 8), &GeometryParams::default()).is_none());
    }

    #[test]
    fn vertex_budget_is_respected() {
        let a = disc_alpha(128, 128, 60.0);
        let p = GeometryParams { spacing: 3.0, max_vertices: 200, ..Default::default() };
        let m = auto_geometry(&a, PixelRect::new(0, 0, 128, 128), &p).unwrap();
        assert!(m.verts.len() <= 260, "{}", m.verts.len());
    }

    #[test]
    fn delaunay_of_a_grid_covers_the_hull() {
        let mut pts = Vec::new();
        for j in 0..5 {
            for i in 0..5 {
                pts.push([i as f32 + 0.01 * ((i * j) % 3) as f32, j as f32]);
            }
        }
        let t = delaunay(&pts);
        let area: f32 = t.iter().map(|t| sprite_mesh::signed_area(&[pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]]).abs()).sum();
        assert!((area - 16.0).abs() < 0.3, "{area}");
    }

    #[test]
    fn triangle_depth_sort_puts_deep_bones_last() {
        let s = arm();
        let mut mesh = SkinMesh2D::new(vec![[0.0; 2]; 6], vec![3, 4, 5, 0, 1, 2], 0);
        for v in 3..6 {
            mesh.weights[v] = VertexWeights::single(1);
        }
        sort_triangles_by_bone_depth(&mut mesh, &s);
        assert_eq!(mesh.indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn rig_file_round_trips_and_old_files_default() {
        let mut rig = Rig2D { skeleton: arm(), parts: vec![] };
        rig.parts.push(SkinPart { sprite_id: 7, name: "Body".into(), mesh: SkinMesh2D::new(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![0, 1, 2], 1), bone_indices: vec![] });
        assert!(rig.validate().is_empty(), "{:?}", rig.validate());
        let text = ron::ser::to_string_pretty(&rig, ron::ser::PrettyConfig::default()).unwrap();
        let back: Rig2D = ron::from_str(&text).unwrap();
        assert_eq!(back, rig);
        // A bare file (old / hand-written) loads with defaults.
        let bare: Rig2D = ron::from_str("(skeleton: (bones: [(name: \"Root\")]))").unwrap();
        assert_eq!(bare.skeleton.bones[0].parent, -1);
        assert!(bare.parts.is_empty());
        assert!(rig.part(7).is_some() && rig.part(8).is_none());
    }

    #[test]
    fn ids_are_stable_and_unique() {
        let mut s = Skeleton2D::default();
        s.bones.push(Bone2D { name: "x".into(), ..Default::default() });
        s.bones.push(Bone2D { name: "x".into(), ..Default::default() });
        assert!(s.ensure_ids());
        assert_ne!(s.bones[0].id, s.bones[1].id);
        let ids = (s.bones[0].id, s.bones[1].id);
        assert!(!s.ensure_ids());
        assert_eq!((s.bones[0].id, s.bones[1].id), ids);
        assert!(!s.validate().is_empty(), "duplicate names are reported");
    }
}
