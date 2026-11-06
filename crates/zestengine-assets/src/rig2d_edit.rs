//! Skinning Editor model (2D-7): the editable state behind the engine's
//! Skinning Editor window, with no UI. A [`SkinningModel`] owns a [`Rig2D`],
//! the active sprite part and bone, and whole-rig undo snapshots (rigs are
//! small, so snapshots are cheap; Unity's skinning cache also snapshots the
//! whole cache). Every edit that changes the rig goes through a method here, so
//! the window is a thin shell and the behaviour is unit-testable.
//!
//! Brush strokes: call [`SkinningModel::begin_stroke`] on press (one undo
//! entry), [`SkinningModel::paint`] while dragging.

use crate::rig2d::{
    AutoWeightParams, Brush, GeometryParams, Rig2D, SkinMesh2D, SkinPart, Skeleton2D, VertexWeights, adjacency, auto_geometry, auto_weights, paint_weights, set_bone_weight, skin_mesh_from_pixels,
    sort_triangles_by_bone_depth,
};
use crate::sprite::PixelRect;
use crate::sprite_mesh::AlphaMap;

const MAX_UNDO: usize = 100;

/// The name a bone's mirror counterpart has: `_L` / `_R`, `.L` / `.R`, `-L` / `-R` suffixes, `L_` / `R_`
/// prefixes and `Left` / `Right` words (any case), swapped. `None` when the name has no side marker.
pub fn mirror_bone_name(name: &str) -> Option<String> {
    fn swap_word(name: &str, a: &str, b: &str) -> Option<String> {
        let lower = name.to_ascii_lowercase();
        for (x, y) in [(a, b), (b, a)] {
            if let Some(at) = lower.find(x) {
                let src = &name[at..at + x.len()];
                let mut rep = y.to_string();
                if src.chars().next().is_some_and(|c| c.is_uppercase()) {
                    rep = rep[..1].to_uppercase() + &rep[1..];
                }
                if src.chars().all(|c| c.is_uppercase()) {
                    rep = rep.to_uppercase();
                }
                return Some(format!("{}{}{}", &name[..at], rep, &name[at + x.len()..]));
            }
        }
        None
    }
    if let Some(s) = swap_word(name, "left", "right") {
        return Some(s);
    }
    for sep in ['_', '.', '-', ' '] {
        for (a, b) in [('L', 'R'), ('l', 'r')] {
            for (x, y) in [(a, b), (b, a)] {
                if let Some(stem) = name.strip_suffix(&format!("{sep}{x}")) {
                    return Some(format!("{stem}{sep}{y}"));
                }
                if let Some(stem) = name.strip_prefix(&format!("{x}{sep}")) {
                    return Some(format!("{y}{sep}{stem}"));
                }
            }
        }
    }
    None
}

/// For each bone, the index of its mirror counterpart by name, or itself when it has none (a spine
/// bone, or a mirror that does not exist).
pub fn mirror_bone_map(skel: &Skeleton2D) -> Vec<usize> {
    skel.bones
        .iter()
        .enumerate()
        .map(|(i, b)| mirror_bone_name(&b.name).and_then(|m| skel.bones.iter().position(|o| o.name == m)).unwrap_or(i))
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct SkinningModel {
    pub rig: Rig2D,
    /// Sprite id of the part being edited.
    pub sprite_id: u64,
    pub active_bone: Option<usize>,
    /// Selected vertices of the active part (the Weights tool); not part of undo.
    pub selection: Vec<u32>,
    /// Copied vertex weights (Copy / Paste Weights).
    clipboard: Option<Vec<(u16, f32)>>,
    undo: Vec<Rig2D>,
    redo: Vec<Rig2D>,
    /// Rig as last saved (the dirty compare).
    saved: Rig2D,
}

impl SkinningModel {
    pub fn new(rig: Rig2D) -> Self {
        Self { saved: rig.clone(), rig, ..Default::default() }
    }

    pub fn is_dirty(&self) -> bool {
        self.rig != self.saved
    }

    pub fn mark_saved(&mut self) {
        self.saved = self.rig.clone();
    }

    /// Push an undo entry for the state before an edit.
    pub fn checkpoint(&mut self) {
        if self.undo.last() != Some(&self.rig) {
            self.undo.push(self.rig.clone());
            if self.undo.len() > MAX_UNDO {
                self.undo.remove(0);
            }
        }
        self.redo.clear();
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.rig, prev));
                self.fix_selection();
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.rig, next));
                self.fix_selection();
                true
            }
            None => false,
        }
    }

    fn fix_selection(&mut self) {
        let n = self.part().map_or(0, |p| p.mesh.positions.len() as u32);
        self.selection.retain(|&v| v < n);
        if self.active_bone.is_some_and(|b| b >= self.rig.skeleton.bones.len()) {
            self.active_bone = None;
        }
    }

    pub fn part_index(&self) -> Option<usize> {
        self.rig.parts.iter().position(|p| p.sprite_id == self.sprite_id)
    }

    pub fn part(&self) -> Option<&SkinPart> {
        self.part_index().map(|i| &self.rig.parts[i])
    }

    // ---- bones -----------------------------------------------------------

    /// Create a bone from a drag (sprite-space head and tip) under the active
    /// bone or `parent`; it becomes the active bone.
    pub fn create_bone(&mut self, parent: Option<usize>, head: [f32; 2], tip: [f32; 2]) -> usize {
        self.checkpoint();
        let i = self.rig.skeleton.add_bone(parent, head, tip, "Bone");
        self.active_bone = Some(i);
        i
    }

    pub fn reparent_bone(&mut self, i: usize, parent: Option<usize>) -> bool {
        let mut trial = self.rig.skeleton.clone();
        if !trial.reparent(i, parent) {
            return false;
        }
        self.checkpoint();
        self.rig.skeleton = trial;
        true
    }

    pub fn split_bone(&mut self, i: usize, t: f32) -> Option<usize> {
        if i >= self.rig.skeleton.bones.len() {
            return None;
        }
        self.checkpoint();
        let n = self.rig.skeleton.split_bone(i, t)?;
        self.active_bone = Some(n);
        Some(n)
    }

    /// Delete a bone; weights move to its parent in every part.
    pub fn delete_bone(&mut self, i: usize) -> bool {
        if i >= self.rig.skeleton.bones.len() {
            return false;
        }
        self.checkpoint();
        let parent = self.rig.skeleton.bones[i].parent;
        let map = self.rig.skeleton.remove_bone(i);
        let fb = (parent >= 0).then_some(parent as usize);
        for p in &mut self.rig.parts {
            p.mesh.remap_bones(&map, fb);
            p.bone_indices = p.bone_indices.iter().filter_map(|&b| map.get(b as usize).copied().flatten().map(|n| n as u16)).collect();
        }
        self.active_bone = None;
        true
    }

    pub fn rename_bone(&mut self, i: usize, name: &str) {
        if i < self.rig.skeleton.bones.len() && !name.is_empty() && self.rig.skeleton.bones[i].name != name && !self.rig.skeleton.bones.iter().any(|b| b.name == name) {
            self.checkpoint();
            self.rig.skeleton.bones[i].name = name.to_owned();
        }
    }

    // ---- geometry and weights -------------------------------------------

    /// Auto Geometry: mesh the sprite's alpha, replace (or add) the part and
    /// bind every vertex with auto-weights. Returns the vertex count, or `None`
    /// when the sprite has no solid pixels or the mesh exceeds 16-bit indices.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_geometry(&mut self, alpha: &AlphaMap, rect: PixelRect, pivot: [f32; 2], ppu: f32, name: &str, geo: &GeometryParams, weights: &AutoWeightParams) -> Option<usize> {
        let pm = auto_geometry(alpha, rect, geo)?;
        let mut mesh = skin_mesh_from_pixels(&pm, rect, pivot, ppu, 0)?;
        mesh.weights = auto_weights(&mesh.positions, &mesh.indices, &self.rig.skeleton, &[], weights);
        sort_triangles_by_bone_depth(&mut mesh, &self.rig.skeleton);
        self.checkpoint();
        let n = mesh.positions.len();
        match self.part_index() {
            Some(i) => {
                self.rig.parts[i].mesh = mesh;
                self.rig.parts[i].name = name.to_owned();
            }
            None => self.rig.parts.push(SkinPart { sprite_id: self.sprite_id, name: name.to_owned(), mesh, bone_indices: Vec::new() }),
        }
        Some(n)
    }

    /// Auto Weights for the active part's current geometry.
    pub fn generate_weights(&mut self, params: &AutoWeightParams) -> bool {
        let Some(pi) = self.part_index() else { return false };
        self.checkpoint();
        let skel = &self.rig.skeleton;
        let part = &mut self.rig.parts[pi];
        part.mesh.weights = auto_weights(&part.mesh.positions, &part.mesh.indices, skel, &part.bone_indices, params);
        sort_triangles_by_bone_depth(&mut part.mesh, skel);
        true
    }

    /// Start of a brush stroke or slider drag: one undo entry for the whole stroke.
    pub fn begin_stroke(&mut self) {
        self.checkpoint();
    }

    /// One brush dab with the active bone; the vertices that changed.
    pub fn paint(&mut self, center: [f32; 2], brush: &Brush) -> Vec<u32> {
        let (Some(bone), Some(pi)) = (self.active_bone, self.part_index()) else { return Vec::new() };
        let mesh: &mut SkinMesh2D = &mut self.rig.parts[pi].mesh;
        let adj = adjacency(mesh.positions.len(), &mesh.indices);
        paint_weights(mesh, &adj, bone as u16, center, brush)
    }

    /// Set the active bone's weight on `verts` (Weight Slider).
    pub fn set_weight(&mut self, verts: &[u32], value: f32) {
        let (Some(bone), Some(pi)) = (self.active_bone, self.part_index()) else { return };
        set_bone_weight(&mut self.rig.parts[pi].mesh, verts, bone as u16, value);
    }

    /// Vertices within `radius` of a point (Weight Slider / selection helper).
    pub fn vertices_near(&self, p: [f32; 2], radius: f32) -> Vec<u32> {
        self.part().map_or(Vec::new(), |part| {
            part.mesh.positions.iter().enumerate().filter(|(_, q)| ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt() <= radius).map(|(i, _)| i as u32).collect()
        })
    }


    // ---- weight sliders, copy / paste, mirror ------------------------------

    /// Replace the vertex selection (`add` keeps the old one). Out-of-range indices are dropped.
    pub fn select_vertices(&mut self, verts: &[u32], add: bool) {
        let n = self.part().map_or(0, |p| p.mesh.positions.len() as u32);
        if !add {
            self.selection.clear();
        }
        for &v in verts {
            if v < n && !self.selection.contains(&v) {
                self.selection.push(v);
            }
        }
        self.selection.sort_unstable();
    }

    /// Select if absent, deselect if present (shift-click).
    pub fn toggle_vertex(&mut self, v: u32) {
        match self.selection.iter().position(|&s| s == v) {
            Some(i) => {
                self.selection.remove(i);
            }
            None => self.select_vertices(&[v], true),
        }
    }

    /// Nearest vertex of the part to `p` within `radius`.
    pub fn nearest_vertex(&self, p: [f32; 2], radius: f32) -> Option<u32> {
        let part = self.part()?;
        part.mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, q)| (i as u32, ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt()))
            .filter(|(_, d)| *d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// The bones a part may use (its `bone_indices`, or all bones).
    pub fn usable_bones(&self) -> Vec<usize> {
        let n = self.rig.skeleton.bones.len();
        match self.part() {
            Some(p) if !p.bone_indices.is_empty() => p.bone_indices.iter().map(|&b| b as usize).filter(|&b| b < n).collect(),
            _ => (0..n).collect(),
        }
    }

    /// Mean weight of `bone` over the selected vertices (what its slider shows).
    pub fn selection_weight(&self, bone: usize) -> f32 {
        let Some(part) = self.part() else { return 0.0 };
        let ws: Vec<f32> = self.selection.iter().filter_map(|&v| part.mesh.weights.get(v as usize)).map(|w| w.weight_of(bone as u16)).collect();
        if ws.is_empty() { 0.0 } else { ws.iter().sum::<f32>() / ws.len() as f32 }
    }

    /// Set `bone`'s weight on every selected vertex to `value`; the other influences rescale so each
    /// vertex still sums to one (four influences at most). Call [`Self::begin_stroke`] when a
    /// slider drag starts so the drag is one undo step.
    pub fn set_selection_weight(&mut self, bone: usize, value: f32) {
        let Some(pi) = self.part_index() else { return };
        if bone >= self.rig.skeleton.bones.len() {
            return;
        }
        let verts = self.selection.clone();
        set_bone_weight(&mut self.rig.parts[pi].mesh, &verts, bone as u16, value);
    }

    /// Remember the first selected vertex's weights.
    pub fn copy_weights(&mut self) -> bool {
        let Some(v) = self.selection.first().copied() else { return false };
        let Some(w) = self.part().and_then(|p| p.mesh.weights.get(v as usize)).copied() else { return false };
        self.clipboard = Some(w.pairs());
        true
    }

    pub fn has_copied_weights(&self) -> bool {
        self.clipboard.is_some()
    }

    /// Give every selected vertex the copied weights (one undo step). Returns how many changed.
    pub fn paste_weights(&mut self) -> usize {
        let (Some(pairs), Some(pi)) = (self.clipboard.clone(), self.part_index()) else { return 0 };
        let n = self.rig.skeleton.bones.len();
        let pairs: Vec<(u16, f32)> = pairs.into_iter().filter(|(b, _)| (*b as usize) < n).collect();
        if pairs.is_empty() || self.selection.is_empty() {
            return 0;
        }
        let w = VertexWeights::from_pairs(&pairs, pairs[0].0);
        let todo: Vec<u32> = self.selection.iter().copied().filter(|&v| self.rig.parts[pi].mesh.weights.get(v as usize).is_some_and(|s| *s != w)).collect();
        if todo.is_empty() {
            return 0;
        }
        self.checkpoint();
        for &v in &todo {
            self.rig.parts[pi].mesh.weights[v as usize] = w;
        }
        todo.len()
    }

    /// Mirror weights across the vertical line `x = axis_x`: each vertex on the destination side
    /// takes the weights of the nearest source-side vertex to its mirror image (within `tol`),
    /// with bones swapped for their mirror counterpart (see [`mirror_bone_map`]). `from_positive`
    /// picks the source side. With a selection, only selected vertices are written. One undo step;
    /// returns how many vertices were written.
    pub fn mirror_weights(&mut self, axis_x: f32, from_positive: bool, tol: f32) -> usize {
        let Some(pi) = self.part_index() else { return 0 };
        let map = mirror_bone_map(&self.rig.skeleton);
        let pos = self.rig.parts[pi].mesh.positions.clone();
        let old = self.rig.parts[pi].mesh.weights.clone();
        if old.len() != pos.len() {
            return 0;
        }
        let on_source = |x: f32| if from_positive { x > axis_x + 1e-5 } else { x < axis_x - 1e-5 };
        let only: Option<std::collections::HashSet<u32>> = (!self.selection.is_empty()).then(|| self.selection.iter().copied().collect());
        let mut writes: Vec<(usize, VertexWeights)> = Vec::new();
        for (i, p) in pos.iter().enumerate() {
            if on_source(p[0]) || (p[0] - axis_x).abs() <= 1e-5 || only.as_ref().is_some_and(|s| !s.contains(&(i as u32))) {
                continue;
            }
            let mirrored = [2.0 * axis_x - p[0], p[1]];
            let best = pos
                .iter()
                .enumerate()
                .filter(|(_, q)| on_source(q[0]))
                .map(|(j, q)| (j, ((q[0] - mirrored[0]).powi(2) + (q[1] - mirrored[1]).powi(2)).sqrt()))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((j, d)) = best
                && d <= tol
            {
                let pairs: Vec<(u16, f32)> = old[j].pairs().into_iter().map(|(b, w)| (map.get(b as usize).copied().unwrap_or(b as usize) as u16, w)).collect();
                writes.push((i, VertexWeights::from_pairs(&pairs, old[i].bones[0])));
            }
        }
        writes.retain(|(i, w)| old[*i] != *w);
        if writes.is_empty() {
            return 0;
        }
        self.checkpoint();
        for (i, w) in &writes {
            self.rig.parts[pi].mesh.weights[*i] = *w;
        }
        writes.len()
    }

    /// Weight of `bone` at every vertex of the active part (the heat-map overlay).
    pub fn weight_map(&self, bone: usize) -> Vec<f32> {
        self.part().map_or(Vec::new(), |p| p.mesh.weights.iter().map(|w| w.weight_of(bone as u16)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig2d::BrushMode;

    fn limb_alpha() -> AlphaMap {
        // A 24 x 64 vertical limb.
        let (w, h) = (24u32, 64u32);
        AlphaMap { w, h, alpha: vec![255; (w * h) as usize] }
    }

    fn model() -> SkinningModel {
        let mut m = SkinningModel::new(Rig2D::default());
        m.sprite_id = 42;
        // Pivot centre, 100 PPU: the limb spans x +-0.12, y +-0.32; two bones stacked in y.
        let upper = m.create_bone(None, [0.0, 0.3], [0.0, 0.0]);
        m.create_bone(Some(upper), [0.0, 0.0], [0.0, -0.3]);
        m
    }

    #[test]
    fn undo_redo_and_dirty_tracking() {
        let mut m = model();
        assert!(m.is_dirty());
        m.mark_saved();
        assert!(!m.is_dirty());
        let n = m.rig.skeleton.bones.len();
        m.create_bone(None, [1.0, 1.0], [2.0, 1.0]);
        assert_eq!(m.rig.skeleton.bones.len(), n + 1);
        assert!(m.is_dirty());
        assert!(m.undo());
        assert_eq!(m.rig.skeleton.bones.len(), n);
        assert!(!m.is_dirty());
        assert!(m.redo());
        assert_eq!(m.rig.skeleton.bones.len(), n + 1);
        m.undo();
        m.create_bone(None, [3.0, 3.0], [4.0, 3.0]);
        assert!(!m.redo(), "a new edit clears redo");
        // Undo until empty never panics and the selection stays valid.
        while m.undo() {}
        assert!(m.rig.skeleton.bones.is_empty());
        assert!(m.active_bone.is_none());
    }

    #[test]
    fn geometry_and_weights_end_to_end() {
        let mut m = model();
        let rect = PixelRect::new(0, 0, 24, 64);
        let n = m.generate_geometry(&limb_alpha(), rect, [0.5, 0.5], 100.0, "Limb", &GeometryParams::default(), &AutoWeightParams::default()).expect("geometry");
        assert!(n >= 8, "{n} vertices");
        let part = m.part().unwrap();
        assert!(part.mesh.validate(2).is_empty(), "{:?}", part.mesh.validate(2));
        // The top of the limb belongs to the upper bone, the bottom to the lower bone.
        let (mut top, mut bottom) = (0usize, 0usize);
        for (i, p) in part.mesh.positions.iter().enumerate() {
            if p[1] > part.mesh.positions.iter().map(|q| q[1]).fold(f32::MIN, f32::max) - 0.01 {
                assert!(part.mesh.weights[i].weight_of(0) > 0.5, "top vertex favours the upper bone");
                top += 1;
            }
            if p[1] < part.mesh.positions.iter().map(|q| q[1]).fold(f32::MAX, f32::min) + 0.01 {
                assert!(part.mesh.weights[i].weight_of(1) > 0.5, "bottom vertex favours the lower bone");
                bottom += 1;
            }
        }
        assert!(top > 0 && bottom > 0);
        // Bending the lower bone moves the bottom, not the top; the rest pose is identity.
        let skel = &m.rig.skeleton;
        let rest = skel.skin_matrices(&skel.world_rest());
        let mut out = Vec::new();
        part.mesh.skin(&rest, &mut out);
        assert!(out.iter().zip(&part.mesh.positions).all(|(a, b)| (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5), "rest pose skins to itself");
        let mut pose: Vec<crate::rig2d::BonePose> = skel.bones.iter().map(crate::rig2d::BonePose::rest).collect();
        pose[1].rotation += 0.8;
        let mats = skel.skin_matrices(&skel.world_pose(&pose));
        part.mesh.skin(&mats, &mut out);
        let moved = |i: usize| ((out[i][0] - part.mesh.positions[i][0]).powi(2) + (out[i][1] - part.mesh.positions[i][1]).powi(2)).sqrt();
        let top_i = (0..out.len()).max_by(|&a, &b| part.mesh.positions[a][1].total_cmp(&part.mesh.positions[b][1])).unwrap();
        let bot_i = (0..out.len()).min_by(|&a, &b| part.mesh.positions[a][1].total_cmp(&part.mesh.positions[b][1])).unwrap();
        assert!(moved(top_i) < 0.05 && moved(bot_i) > 0.1, "top {} bottom {}", moved(top_i), moved(bot_i));
    }

    #[test]
    fn delete_bone_moves_weights_to_its_parent_in_every_part() {
        let mut m = model();
        m.generate_geometry(&limb_alpha(), PixelRect::new(0, 0, 24, 64), [0.5, 0.5], 100.0, "Limb", &GeometryParams::default(), &AutoWeightParams::default());
        assert!(m.delete_bone(1));
        assert_eq!(m.rig.skeleton.bones.len(), 1);
        let part = m.part().unwrap();
        assert!(part.mesh.validate(1).is_empty());
        assert!(part.mesh.weights.iter().all(|w| w.weight_of(0) > 0.999));
        assert!(m.undo());
        assert_eq!(m.rig.skeleton.bones.len(), 2);
        assert!(m.part().unwrap().mesh.validate(2).is_empty());
        assert!(!m.delete_bone(9));
    }

    #[test]
    fn painting_is_one_undo_step_per_stroke() {
        let mut m = model();
        m.generate_geometry(&limb_alpha(), PixelRect::new(0, 0, 24, 64), [0.5, 0.5], 100.0, "Limb", &GeometryParams::default(), &AutoWeightParams::default());
        let before = m.rig.clone();
        m.active_bone = Some(0);
        m.begin_stroke();
        let brush = Brush { mode: BrushMode::Add, radius: 0.2, hardness: 1.0, strength: 1.0 };
        let mut changed = 0;
        for y in [-0.3f32, -0.2, -0.1] {
            changed += m.paint([0.0, y], &brush).len();
        }
        assert!(changed > 0);
        assert!(m.weight_map(0).iter().cloned().fold(0.0, f32::max) > 0.999);
        assert!(m.undo());
        assert_eq!(m.rig, before, "the whole stroke undoes at once");
        // Painting needs an active bone and a part.
        m.active_bone = None;
        assert!(m.paint([0.0, 0.0], &brush).is_empty());
        m.active_bone = Some(0);
        let near = m.vertices_near([0.0, 0.0], 0.1);
        m.set_weight(&near, 1.0);
        assert!(near.iter().all(|&v| m.weight_map(0)[v as usize] > 0.999));
    }

    /// A symmetric part over two arm bones named `Arm_L` / `Arm_R` and a `Spine`.
    fn symmetric() -> SkinningModel {
        let mut m = SkinningModel::new(Rig2D::default());
        m.sprite_id = 7;
        m.create_bone(None, [0.0, 0.0], [0.0, 0.3]);
        m.create_bone(Some(0), [0.0, 0.3], [0.3, 0.3]);
        m.create_bone(Some(0), [0.0, 0.3], [-0.3, 0.3]);
        m.rename_bone(0, "Spine");
        m.rename_bone(1, "Arm_L");
        m.rename_bone(2, "Arm_R");
        let xs = [-0.4f32, -0.2, 0.0, 0.2, 0.4];
        let positions: Vec<[f32; 2]> = xs.iter().flat_map(|&x| [[x, 0.0], [x, 0.3]]).collect();
        let mut mesh = SkinMesh2D::new(positions, vec![0, 2, 1, 1, 2, 3], 0);
        // Right side (x > 0) leans on Arm_L (index 1), left side on Arm_R: weights not yet mirrored.
        for (i, p) in mesh.positions.clone().iter().enumerate() {
            mesh.weights[i] = if p[0] > 0.0 { VertexWeights::from_pairs(&[(1, 0.7), (0, 0.3)], 0) } else { VertexWeights::single(0) };
        }
        m.rig.parts.push(SkinPart { sprite_id: 7, name: "Body".into(), mesh, bone_indices: vec![] });
        m
    }

    #[test]
    fn mirror_names_swap_sides() {
        for (a, b) in [("Arm_L", "Arm_R"), ("hand.l", "hand.r"), ("L_Leg", "R_Leg"), ("LeftFoot", "RightFoot"), ("upper_left_arm", "upper_right_arm"), ("Wing-R", "Wing-L")] {
            assert_eq!(mirror_bone_name(a).as_deref(), Some(b), "{a}");
            assert_eq!(mirror_bone_name(b).as_deref(), Some(a), "{b}");
        }
        assert_eq!(mirror_bone_name("Spine"), None);
        assert_eq!(mirror_bone_name("Shoulder"), None);
        let m = symmetric();
        assert_eq!(mirror_bone_map(&m.rig.skeleton), vec![0, 2, 1]);
    }

    #[test]
    fn weight_sliders_normalise_and_undo_as_one_step() {
        let mut m = symmetric();
        let right: Vec<u32> = (0..m.part().unwrap().mesh.positions.len() as u32).filter(|&v| m.part().unwrap().mesh.positions[v as usize][0] > 0.0).collect();
        m.select_vertices(&right, false);
        assert_eq!(m.selection.len(), right.len());
        assert!((m.selection_weight(1) - 0.7).abs() < 1e-5);
        let before = m.rig.clone();
        m.begin_stroke();
        for v in [0.8f32, 0.9, 0.5] {
            m.set_selection_weight(1, v);
        }
        for &i in &right {
            let w = m.part().unwrap().mesh.weights[i as usize];
            assert!((w.sum() - 1.0).abs() < 1e-4, "normalised: {}", w.sum());
            assert!((w.weight_of(1) - 0.5).abs() < 1e-4);
            assert!((w.weight_of(0) - 0.5).abs() < 1e-4, "the other influence took the rest");
        }
        // A third bone joins: still normalised, at most four influences.
        m.set_selection_weight(2, 0.25);
        for &i in &right {
            let w = m.part().unwrap().mesh.weights[i as usize];
            assert!((w.sum() - 1.0).abs() < 1e-4 && (w.weight_of(2) - 0.25).abs() < 1e-4);
        }
        assert!(m.undo());
        assert_eq!(m.rig, before, "the slider drag was one undo step");
        // Out-of-range bone and no selection are no-ops.
        m.set_selection_weight(9, 1.0);
        assert_eq!(m.rig, before);
        m.select_vertices(&[], false);
        m.set_selection_weight(1, 1.0);
        assert_eq!(m.rig, before);
    }

    #[test]
    fn selection_helpers() {
        let mut m = symmetric();
        let n = m.part().unwrap().mesh.positions.len() as u32;
        m.select_vertices(&[1, 3, 99], false);
        assert_eq!(m.selection, vec![1, 3], "out of range dropped");
        m.toggle_vertex(3);
        m.toggle_vertex(5);
        assert_eq!(m.selection, vec![1, 5]);
        assert_eq!(m.nearest_vertex([0.41, 0.01], 0.05), Some(n - 2));
        assert_eq!(m.nearest_vertex([5.0, 5.0], 0.05), None);
        assert_eq!(m.usable_bones(), vec![0, 1, 2]);
        m.rig.parts[0].bone_indices = vec![2, 9];
        assert_eq!(m.usable_bones(), vec![2], "restricted to the part's bones, bad index dropped");
    }

    #[test]
    fn copy_and_paste_weights() {
        let mut m = symmetric();
        assert!(!m.copy_weights(), "nothing selected");
        assert_eq!(m.paste_weights(), 0);
        // Copy a right-side vertex (index 6: x = 0.2 .. positions are pairs per x).
        let src = m.nearest_vertex([0.2, 0.0], 0.01).unwrap();
        m.select_vertices(&[src], false);
        assert!(m.copy_weights() && m.has_copied_weights());
        let dst: Vec<u32> = [m.nearest_vertex([-0.2, 0.0], 0.01).unwrap(), m.nearest_vertex([-0.4, 0.3], 0.01).unwrap()].to_vec();
        m.select_vertices(&dst, false);
        let before = m.rig.clone();
        assert_eq!(m.paste_weights(), 2);
        for &d in &dst {
            let w = m.part().unwrap().mesh.weights[d as usize];
            assert!((w.weight_of(1) - 0.7).abs() < 1e-4 && (w.weight_of(0) - 0.3).abs() < 1e-4);
        }
        assert_eq!(m.paste_weights(), 0, "pasting the same weights again changes nothing");
        assert!(m.undo());
        assert_eq!(m.rig, before);
    }

    #[test]
    fn mirror_weights_copies_across_the_axis_with_bones_swapped() {
        let mut m = symmetric();
        let before = m.rig.clone();
        let n = m.mirror_weights(0.0, true, 0.01);
        assert_eq!(n, 4, "the four left-side vertices were written");
        let part = m.part().unwrap();
        for (i, p) in part.mesh.positions.iter().enumerate() {
            if p[0] < -1e-5 {
                let w = part.mesh.weights[i];
                assert!((w.weight_of(2) - 0.7).abs() < 1e-4, "Arm_L weight became Arm_R's on the mirrored vertex: {:?}", w.pairs());
                assert!((w.weight_of(0) - 0.3).abs() < 1e-4 && w.weight_of(1) < 1e-6);
            } else {
                assert!(w_eq(&part.mesh.weights[i], &before.parts[0].mesh.weights[i]), "source side untouched");
            }
        }
        // The centre line is untouched; a selection restricts the destination; no partner in range writes nothing.
        let mut m2 = symmetric();
        let one = m2.nearest_vertex([-0.2, 0.3], 0.01).unwrap();
        m2.select_vertices(&[one], false);
        assert_eq!(m2.mirror_weights(0.0, true, 0.01), 1);
        let mut m3 = symmetric();
        assert_eq!(m3.mirror_weights(0.07, true, 0.01), 0, "an axis off the symmetry finds no partner within the tolerance");
        // Undo is one step; the other direction mirrors right <- left.
        assert!(m.undo());
        assert_eq!(m.rig, before);
        assert_eq!(m.mirror_weights(0.0, false, 0.01), 4);
    }

    fn w_eq(a: &VertexWeights, b: &VertexWeights) -> bool {
        a == b
    }

    #[test]
    fn bone_edits_validate() {
        let mut m = model();
        assert!(!m.reparent_bone(0, Some(1)), "cycle refused without an undo entry");
        assert!(m.reparent_bone(1, None));
        assert_eq!(m.rig.skeleton.bones[1].parent, -1);
        let n = m.split_bone(0, 0.5).unwrap();
        assert_eq!(m.active_bone, Some(n));
        m.rename_bone(n, "Forearm");
        m.rename_bone(0, "Forearm");
        assert_eq!(m.rig.skeleton.bones[0].name, "Bone", "duplicate names are refused");
        assert!(m.rig.skeleton.validate().is_empty(), "{:?}", m.rig.skeleton.validate());
        assert!(m.split_bone(99, 0.5).is_none());
    }

    #[test]
    fn empty_sprite_gives_no_geometry() {
        let mut m = model();
        let blank = AlphaMap { w: 8, h: 8, alpha: vec![0; 64] };
        assert!(m.generate_geometry(&blank, PixelRect::new(0, 0, 8, 8), [0.5, 0.5], 100.0, "x", &GeometryParams::default(), &AutoWeightParams::default()).is_none());
        assert!(m.part().is_none());
        assert!(!m.generate_weights(&AutoWeightParams::default()));
    }
}
