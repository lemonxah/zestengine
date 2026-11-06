//! 2D skeletal animation at runtime (2D-7): 2D IK, sprite swap resolvers and
//! CPU skinning of sprite meshes, driven from the sprite gather
//! (`sprites2d.rs`). The data is in `zestengine_assets::{rig2d, ik2d,
//! sprite_library}`; the components in `zestengine_core::sprite_rig`.
//!
//! # Per frame, in order (only when a scene has one of the rig components)
//!
//! 1. **IK** (`IkSolver2D`): after animation, before skinning. The chain is the
//!    `tip` object plus `chain_length` ancestors; segment lengths and the
//!    current pose are read from the world transforms, the solver
//!    (`ik2d::IkChain`) runs, and only the joints' Z rotations are written back
//!    (blended by `weight`). IK assumes a planar chain with positive uniform
//!    scale (a mirrored parent flips angle signs).
//! 2. **Sprite swap** (`SpriteResolver`): when the resolver's hash changes
//!    (an animation `SpriteSwap` track, or the inspector) the library entry is
//!    looked up once and kept as a render-time override of the sibling
//!    `SpriteRenderer`'s sprite; the scene component itself is not modified. An
//!    unknown hash keeps the current sprite. Cost: O(resolvers whose hash
//!    changed).
//! 3. **Skinning** (`SpriteSkin`): bone object world transforms are expressed in
//!    the skin object's space, multiplied by the rig's bind poses, and the
//!    rig's weighted mesh is skinned on the CPU into a *dynamic mesh* of the
//!    sprite pool (`SpriteMeshPool::add_dynamic`): same pipeline and
//!    shaders as any sprite mesh, no per-vertex bone data on the GPU. This is
//!    the documented choice over a GPU skinning vertex path: a sprite mesh is
//!    tens to hundreds of vertices, the pass stays one instanced indexed draw
//!    per batch, and a skin whose bones did not move costs one matrix compare
//!    and no upload. Skinned sprites draw from their own texture (the
//!    "plain" resolve, not an atlas page), since the skin UVs are derived from
//!    the sprite rect.
//!
//! A scene without these components pays one component-type compare per
//! sprite-component object (already part of the sprite scan) and nothing else.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use glam::{Affine2, EulerRot, Mat4, Quat, Vec2};
use zestengine_assets::ik2d::{IkBone, IkChain};
use zestengine_assets::rig2d::{Rig2D, load_rig2d};
use zestengine_assets::sprite_library::{ResolvedLibrary, load_resolved_library, resolve_swap};
use zestengine_core::{
    Component, IkSolver2D, IkSolverKind, ObjectId, SpriteBone, SpriteLibraryRef, SpriteRenderer, SpriteResolver, SpriteSkin, component_enabled, component_inner,
    split_sprite_ref,
};
use zestengine_render::Renderer;
use zestengine_render::sprite2d::{DynamicMesh, MeshRange};

use super::LoadedScene;
use super::sprites2d::SpriteCache;

/// Frames between asset timestamp checks.
const STAMP_CHECK: u32 = 60;
/// Frames between rebind attempts for a skin with unresolved bones.
const REBIND_RETRY: u32 = 60;

struct RigAsset {
    rig: Rig2D,
    bind: Vec<Affine2>,
    path: PathBuf,
    stamp: Option<SystemTime>,
}

struct LibAsset {
    lib: ResolvedLibrary,
    path: PathBuf,
    stamp: Option<SystemTime>,
}

struct SkinState {
    rig_path: String,
    sprite_id: u64,
    part: usize,
    dynamic: Option<DynamicMesh>,
    /// Scene object index of each skeleton bone (`None` = skins as rest).
    bones: Vec<Option<usize>>,
    unresolved_since: u32,
    last_mats: Vec<Affine2>,
    range: Option<MeshRange>,
    centre: [f32; 2],
    /// Pool full or a bad part: stop retrying until the rig changes.
    failed: bool,
    seen: u32,
}

#[derive(Default)]
struct ResolverState {
    hash_seen: u32,
    epoch: u32,
    text: Option<Arc<str>>,
}

/// Where a skinned sprite draws this frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct SkinOut {
    pub mesh: MeshRange,
    pub centre: [f32; 2],
}

#[derive(Default)]
pub struct RigCache {
    rigs: HashMap<String, Option<Arc<RigAsset>>>,
    libs: HashMap<String, Option<Arc<LibAsset>>>,
    skins: HashMap<ObjectId, SkinState>,
    resolvers: HashMap<ObjectId, ResolverState>,
    /// Objects with a rig component, collected by the sprite scan.
    pub(super) objs: Vec<usize>,
    skin_out: HashMap<usize, SkinOut>,
    swap_out: HashMap<usize, Arc<str>>,
    lib_epoch: u32,
    frame: u32,
}

impl RigCache {
    pub(super) fn invalidate(&mut self) {
        self.rigs.clear();
        self.libs.clear();
        self.skins.clear();
        self.resolvers.clear();
        self.lib_epoch += 1;
    }

    /// Whether the scan found a rig component this frame.
    pub(super) fn active(&self) -> bool {
        !self.objs.is_empty()
    }

    pub(super) fn skin_for(&self, obj: usize) -> Option<SkinOut> {
        self.skin_out.get(&obj).copied()
    }

    /// Forget last frame's outputs (the scene lost its rig components).
    pub(super) fn clear_outputs(&mut self) {
        self.skin_out.clear();
        self.swap_out.clear();
    }

    pub(super) fn swap_for(&self, obj: usize) -> Option<&str> {
        self.swap_out.get(&obj).map(|s| &**s)
    }
}

pub(super) fn is_rig_component(c: &dyn Component) -> bool {
    let a = component_inner(c).as_any();
    a.is::<SpriteSkin>() || a.is::<SpriteResolver>() || a.is::<IkSolver2D>()
}

fn find<T: 'static>(comps: &[Box<dyn Component>]) -> Option<&T> {
    comps.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<T>())
}

fn enabled_find<T: 'static>(comps: &[Box<dyn Component>]) -> Option<&T> {
    comps.iter().find_map(|c| if component_enabled(c.as_ref()) { component_inner(c.as_ref()).as_any().downcast_ref::<T>() } else { None })
}

/// Bone object transform in the skin object's space as a 2D affine (the XY
/// plane; rotation about Z).
pub fn affine_in_space(skin_world_inv: Mat4, bone_world: Mat4) -> Affine2 {
    let m = skin_world_inv * bone_world;
    Affine2::from_cols(Vec2::new(m.x_axis.x, m.x_axis.y), Vec2::new(m.y_axis.x, m.y_axis.y), Vec2::new(m.w_axis.x, m.w_axis.y))
}

/// The skinning matrix of one bone for a skin object `skin_w` whose rig is expressed in the space of
/// `root_w` (a shared character root: every skinned sprite of a character sits at its own place but
/// one skeleton drives them all). The bone's motion `M = bone_in_root * bind` happens in root space;
/// the skin's vertices live in its own space, so the matrix applied to them is `S^-1 * M * S`
/// where `S` is the skin object in root space. When the root is the skin object itself (`S` is the
/// identity) this is the plain `bone_in_skin * bind`.
pub fn skin_matrix(root_w: Mat4, skin_w: Mat4, bone_w: Mat4, bind: Affine2) -> Affine2 {
    let inv_root = root_w.inverse();
    let m = affine_in_space(inv_root, bone_w) * bind;
    let s = affine_in_space(inv_root, skin_w);
    if s == Affine2::IDENTITY { m } else { s.inverse() * m * s }
}

/// UV of a sprite-space point for a sprite drawn from its own texture:
/// `uv_rect` = (u0, v_top, u1, v_bottom), `size` / `pivot` in world units
/// (pivot from the rect's bottom-left).
pub fn skin_uv(p: [f32; 2], uv_rect: [f32; 4], size: [f32; 2], pivot: [f32; 2]) -> [f32; 2] {
    let fx = if size[0] > 0.0 { (p[0] + pivot[0]) / size[0] } else { 0.0 };
    let fy = if size[1] > 0.0 { (size[1] - (p[1] + pivot[1])) / size[1] } else { 0.0 };
    [uv_rect[0] + (uv_rect[2] - uv_rect[0]) * fx, uv_rect[1] + (uv_rect[3] - uv_rect[1]) * fy]
}

fn same(a: &[Affine2], b: &[Affine2]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.abs_diff_eq(*y, 1e-6))
}

fn angle_of(m: &Mat4) -> f32 {
    m.x_axis.y.atan2(m.x_axis.x)
}

fn z_angle(q: Quat) -> f32 {
    q.to_euler(EulerRot::ZYX).0
}

impl LoadedScene {
    /// Step 1 and 2 (before the frame's sprite references resolve): IK and
    /// sprite swaps. Adds the swapped sprites to `c.need`.
    pub(super) fn update_rigs_pre(&mut self, c: &mut SpriteCache, project_root: &Path) {
        c.rig.frame = c.rig.frame.wrapping_add(1);
        c.rig.skin_out.clear();
        c.rig.swap_out.clear();
        if c.rig.frame.is_multiple_of(STAMP_CHECK) {
            let stale_rig = c.rig.rigs.values().any(|r| r.as_ref().is_some_and(|r| std::fs::metadata(&r.path).and_then(|m| m.modified()).ok() != r.stamp));
            let stale_lib = c.rig.libs.values().any(|l| l.as_ref().is_some_and(|l| std::fs::metadata(&l.path).and_then(|m| m.modified()).ok() != l.stamp));
            if stale_rig {
                c.rig.rigs.clear();
                c.rig.skins.clear();
            }
            if stale_lib {
                c.rig.libs.clear();
                c.rig.lib_epoch += 1;
            }
        }
        let ids: HashMap<ObjectId, usize> = self.id_index();
        self.run_ik(&c.rig.objs, &ids);
        self.run_resolvers(c, project_root);
    }

    fn run_ik(&mut self, objs: &[usize], ids: &HashMap<ObjectId, usize>) {
        for &oi in objs {
            let Some(o) = self.objects.get(oi) else { continue };
            if !o.enabled || !self.is_active(oi) {
                continue;
            }
            let Some(ik) = enabled_find::<IkSolver2D>(&o.components).cloned() else { continue };
            if ik.weight <= 0.0 {
                continue;
            }
            let (Some(tip), Some(target)) = (ik.tip.id().and_then(|i| ids.get(&i).copied()), ik.target.id().and_then(|i| ids.get(&i).copied())) else { continue };
            // Joints: tip first, then its ancestors.
            let mut joints = vec![tip];
            let want = if ik.solver == IkSolverKind::Limb { 2 } else { ik.chain_length.max(1) as usize };
            while joints.len() <= want {
                match self.objects[*joints.last().unwrap()].parent.filter(|&p| p < self.objects.len() && !joints.contains(&p)) {
                    Some(p) => joints.push(p),
                    None => break,
                }
            }
            if joints.len() < 2 {
                continue;
            }
            joints.reverse(); // root first; bone k runs joints[k] -> joints[k + 1]
            let world = self.world_transforms();
            let p: Vec<Vec2> = joints.iter().map(|&j| Vec2::new(world[j].w_axis.x, world[j].w_axis.y)).collect();
            let n = joints.len() - 1;
            let seg_abs: Vec<f32> = (0..n).map(|k| (p[k + 1].y - p[k].y).atan2(p[k + 1].x - p[k].x)).collect();
            let base_angle = self.objects[joints[0]].parent.map_or(0.0, |par| angle_of(&world[par]));
            let bones: Vec<IkBone> = (0..n)
                .map(|k| {
                    let rel = zestengine_assets::ik2d::wrap_angle(seg_abs[k] - if k == 0 { base_angle } else { seg_abs[k - 1] });
                    let limit = find::<SpriteBone>(&self.objects[joints[k]].components).filter(|b| b.limit_rotation).map(|b| (b.min_angle.to_radians(), b.max_angle.to_radians()));
                    IkBone { length: (p[k + 1] - p[k]).length(), angle: rel, limit }
                })
                .collect();
            let tgt = Vec2::new(world[target].w_axis.x, world[target].w_axis.y).to_array();
            let mut chain = IkChain::new(p[0].to_array(), base_angle, bones);
            let iters = ik.iterations.clamp(1, 256);
            let tol = ik.tolerance.max(1e-5);
            if (Vec2::from(chain.tip()) - Vec2::from(tgt)).length() <= tol {
                continue;
            }
            match ik.solver {
                IkSolverKind::Ccd => chain.solve_ccd(tgt, iters, tol),
                IkSolverKind::Fabrik => chain.solve_fabrik(tgt, iters, tol),
                IkSolverKind::Limb => chain.solve_limb(tgt, ik.flip, tol),
            };
            // Write back the Z rotations: bone k turns by the change of its segment
            // direction, minus what its already-rotated parents contribute.
            let weight = ik.weight.clamp(0.0, 1.0);
            let mut acc = 0.0f32;
            for k in 0..n {
                let solved = chain.absolute_angle(k);
                let want_total = zestengine_assets::ik2d::wrap_angle(solved - seg_abs[k]);
                let d = zestengine_assets::ik2d::wrap_angle(want_total - acc);
                acc += d;
                let o = &mut self.objects[joints[k]];
                o.rotation = Quat::from_rotation_z(z_angle(o.rotation) + d * weight);
            }
        }
    }

    fn lib_for(&self, c: &mut SpriteCache, project_root: &Path, obj: usize) -> Option<Arc<LibAsset>> {
        // Nearest Sprite Library on the object or an ancestor.
        let mut i = obj;
        let mut path = None;
        for _ in 0..64 {
            if let Some(l) = enabled_find::<SpriteLibraryRef>(&self.objects[i].components) {
                path = Some(l.library.clone());
                break;
            }
            match self.objects[i].parent.filter(|&p| p < self.objects.len()) {
                Some(p) => i = p,
                None => break,
            }
        }
        let path = path.filter(|p| !p.is_empty())?;
        c.rig
            .libs
            .entry(path.clone())
            .or_insert_with(|| {
                let full = project_root.join(&path);
                match load_resolved_library(project_root, &full) {
                    Ok(lib) => Some(Arc::new(LibAsset { lib, stamp: std::fs::metadata(&full).and_then(|m| m.modified()).ok(), path: full })),
                    Err(e) => {
                        tracing::warn!("sprite library {path}: {e:#}");
                        None
                    }
                }
            })
            .clone()
    }

    fn run_resolvers(&mut self, c: &mut SpriteCache, project_root: &Path) {
        let objs = std::mem::take(&mut c.rig.objs);
        for &oi in &objs {
            let Some(o) = self.objects.get(oi) else { continue };
            let Some(res) = enabled_find::<SpriteResolver>(&o.components) else { continue };
            let Some(rend) = find::<SpriteRenderer>(&o.components) else { continue };
            let (hash, id, base) = (res.hash, o.id, rend.sprite.clone());
            let epoch = c.rig.lib_epoch;
            let st = c.rig.resolvers.entry(id).or_default();
            let changed = st.hash_seen != hash || st.epoch != epoch;
            st.hash_seen = hash;
            st.epoch = epoch;
            if changed {
                let lib = self.lib_for(c, project_root, oi);
                let st = c.rig.resolvers.get_mut(&id).expect("inserted above");
                if let Some(lib) = lib {
                    let cur = st.text.as_deref().unwrap_or(&base);
                    if let Some(new) = resolve_swap(&lib.lib, hash, cur) {
                        st.text = Some(Arc::from(new));
                    }
                }
            }
            if let Some(t) = c.rig.resolvers.get(&id).and_then(|s| s.text.clone()) {
                let plain = find::<SpriteSkin>(&self.objects[oi].components).is_some() || rend.draw_mode != zestengine_core::SpriteDrawMode::Simple;
                c.need_push(&t, plain);
                if plain {
                    c.need_push(&t, false);
                }
                c.rig.swap_out.insert(oi, t);
            }
        }
        c.rig.objs = objs;
    }

    fn rig_asset(&self, c: &mut SpriteCache, project_root: &Path, path: &str) -> Option<Arc<RigAsset>> {
        c.rig
            .rigs
            .entry(path.to_owned())
            .or_insert_with(|| {
                let full = project_root.join(path);
                match load_rig2d(&full) {
                    Ok(mut rig) => {
                        rig.skeleton.ensure_ids();
                        for e in rig.validate() {
                            tracing::warn!("rig {path}: {e}");
                        }
                        let bind = rig.skeleton.bind_poses();
                        Some(Arc::new(RigAsset { rig, bind, stamp: std::fs::metadata(&full).and_then(|m| m.modified()).ok(), path: full }))
                    }
                    Err(e) => {
                        tracing::warn!("rig {path}: {e:#}");
                        None
                    }
                }
            })
            .clone()
    }

    /// Bone objects for a skin: the explicit `bones` first, then (with
    /// `auto_rebind`) the descendants of the root by bone id, then by name.
    fn bind_bones(&self, skin_obj: usize, skin: &SpriteSkin, rig: &Rig2D, ids: &HashMap<ObjectId, usize>) -> Vec<Option<usize>> {
        let n = rig.skeleton.bones.len();
        let mut out: Vec<Option<usize>> = (0..n).map(|i| skin.bones.get(i).and_then(|r| r.id()).and_then(|id| ids.get(&id).copied())).collect();
        if skin.auto_rebind && out.iter().any(Option::is_none) {
            let root = skin.root.id().and_then(|id| ids.get(&id).copied()).unwrap_or(skin_obj);
            let below = |mut i: usize| -> bool {
                for _ in 0..64 {
                    if i == root {
                        return true;
                    }
                    match self.objects[i].parent.filter(|&p| p < self.objects.len()) {
                        Some(p) => i = p,
                        None => return false,
                    }
                }
                false
            };
            for (bi, b) in rig.skeleton.bones.iter().enumerate() {
                if out[bi].is_some() {
                    continue;
                }
                let by_id = if b.id != 0 { (0..self.objects.len()).find(|&i| below(i) && find::<SpriteBone>(&self.objects[i].components).is_some_and(|sb| sb.bone_id == b.id)) } else { None };
                out[bi] = by_id.or_else(|| (0..self.objects.len()).find(|&i| i != root && below(i) && self.objects[i].name == b.name && find::<SpriteBone>(&self.objects[i].components).is_some()));
            }
        }
        out
    }

    /// Step 3 (after the frame's sprite references resolve): skin every
    /// `SpriteSkin` into its dynamic mesh.
    pub(super) fn update_rigs_skin(&mut self, c: &mut SpriteCache, renderer: &mut Renderer, project_root: &Path) {
        let mut world: Option<Vec<Mat4>> = None;
        let mut ids: Option<HashMap<ObjectId, usize>> = None;
        let objs = std::mem::take(&mut c.rig.objs);
        let frame = c.rig.frame;
        for &oi in &objs {
            let Some(o) = self.objects.get(oi) else { continue };
            let Some(skin) = enabled_find::<SpriteSkin>(&o.components).cloned() else { continue };
            let Some(rend) = find::<SpriteRenderer>(&o.components) else { continue };
            if skin.rig.is_empty() || !o.enabled || !self.is_active(oi) {
                continue;
            }
            let text: String = c.rig.swap_for(oi).map(str::to_owned).unwrap_or_else(|| rend.sprite.clone());
            let Some((_, sprite_id)) = split_sprite_ref(&text) else { continue };
            if (self.visible_layers >> (o.layer as u32 & 31)) & 1 == 0 && !skin.always_update {
                continue;
            }
            let Some(rig) = self.rig_asset(c, project_root, &skin.rig) else { continue };
            let Some(part_idx) = rig.rig.parts.iter().position(|p| p.sprite_id == sprite_id) else { continue };
            let Some(sprite) = c.resolved_plain_get(&text) else { continue };
            let (uv_rect, size, pivot) = (sprite.uv_rect, sprite.size, sprite.pivot);
            let ids_ref = ids.get_or_insert_with(|| self.id_index());
            let id = o.id;
            // (Re)build the per-skin state when the rig, sprite or part changed.
            let rebuild = c.rig.skins.get(&id).is_none_or(|s| s.rig_path != skin.rig || s.sprite_id != sprite_id || s.part != part_idx);
            if rebuild {
                let part = &rig.rig.parts[part_idx];
                let bones = self.bind_bones(oi, &skin, &rig.rig, ids_ref);
                let uvs: Vec<[f32; 2]> = part.mesh.positions.iter().map(|&p| skin_uv(p, uv_rect, size, pivot)).collect();
                let dynamic = renderer.sprite_mesh_pool().add_dynamic(&uvs, &part.mesh.indices);
                let failed = dynamic.is_none();
                if failed {
                    tracing::warn!("skinned sprite '{}': the sprite mesh pool is full or the mesh is invalid", self.objects[oi].name);
                }
                c.rig.skins.insert(
                    id,
                    SkinState {
                        rig_path: skin.rig.clone(),
                        sprite_id,
                        part: part_idx,
                        dynamic,
                        bones,
                        unresolved_since: frame,
                        last_mats: Vec::new(),
                        range: None,
                        centre: [0.0; 2],
                        failed,
                        seen: frame,
                    },
                );
            }
            let st = c.rig.skins.get_mut(&id).expect("inserted above");
            st.seen = frame;
            if st.failed {
                continue;
            }
            // Retry unresolved bones now and then (a bone object created after the skin).
            if skin.auto_rebind && st.bones.iter().any(Option::is_none) && frame.wrapping_sub(st.unresolved_since) >= REBIND_RETRY {
                st.unresolved_since = frame;
                let fresh = self.bind_bones(oi, &skin, &rig.rig, ids_ref);
                let st = c.rig.skins.get_mut(&id).expect("exists");
                if fresh != st.bones {
                    st.bones = fresh;
                    st.last_mats.clear();
                }
            }
            let st = c.rig.skins.get_mut(&id).expect("exists");
            let w = world.get_or_insert_with(|| self.world_transforms());
            // The rig lives in the space of the skin's `root` object when one is set (a character
            // root shared by many skinned sprites), else in the skin object's own space.
            let root_i = skin.root.id().and_then(|id| ids_ref.get(&id).copied()).filter(|&r| r < w.len()).unwrap_or(oi);
            let mats: Vec<Affine2> = st
                .bones
                .iter()
                .enumerate()
                .map(|(bi, b)| match b {
                    Some(j) => skin_matrix(w[root_i], w[oi], w[*j], rig.bind[bi]),
                    None => Affine2::IDENTITY,
                })
                .collect();
            let (Some(dynamic), part) = (st.dynamic, &rig.rig.parts[part_idx]) else { continue };
            let range = match (st.range, same(&mats, &st.last_mats)) {
                (Some(r), true) => r, // bones did not move: the last slot is still valid
                _ => {
                    let mut out = Vec::with_capacity(part.mesh.positions.len());
                    part.mesh.skin(&mats, &mut out);
                    let (lo, hi) = zestengine_assets::rig2d::bounds_of(&out);
                    st.centre = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
                    match renderer.sprite_mesh_pool().write_dynamic(dynamic, &out) {
                        Some(r) => {
                            st.last_mats = mats;
                            st.range = Some(r);
                            r
                        }
                        None => continue,
                    }
                }
            };
            c.rig.skin_out.insert(oi, SkinOut { mesh: range, centre: st.centre });
        }
        c.rig.objs = objs;
        // Forget state of objects that stopped being skinned.
        if frame.is_multiple_of(STAMP_CHECK) {
            c.rig.skins.retain(|_, s| frame.wrapping_sub(s.seen) < STAMP_CHECK * 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use zestengine_core::{ObjectRef, TypedComponent};

    use crate::scene::{ObjectSource, SceneObject};

    fn object(name: &str, parent: Option<usize>, t: Vec3, comps: Vec<Box<dyn Component>>) -> SceneObject {
        SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: name.into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent,
            translation: t,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: comps,
        }
    }

    #[test]
    fn affine_in_skin_space() {
        let skin = Mat4::from_rotation_translation(Quat::from_rotation_z(0.5), Vec3::new(3.0, 1.0, 0.0));
        let bone_local = Mat4::from_rotation_translation(Quat::from_rotation_z(0.25), Vec3::new(1.0, 2.0, 0.0));
        let a = affine_in_space(skin.inverse(), skin * bone_local);
        assert!((a.translation - Vec2::new(1.0, 2.0)).length() < 1e-5);
        let ang = a.matrix2.x_axis.y.atan2(a.matrix2.x_axis.x);
        assert!((ang - 0.25).abs() < 1e-5);
    }

    #[test]
    fn skin_matrix_in_root_space_is_identity_at_rest_and_conjugates_motion() {
        // Character root at (10, 0); a skinned sprite object 2 m to its right; one bone at the
        // root's origin (bind = identity there).
        let root = Mat4::from_translation(glam::Vec3::new(10.0, 0.0, 0.0));
        let skin = Mat4::from_translation(glam::Vec3::new(12.0, 0.5, 0.0));
        let bone_rest = root;
        assert!(skin_matrix(root, skin, bone_rest, Affine2::IDENTITY).abs_diff_eq(Affine2::IDENTITY, 1e-5), "rest pose does not move the sprite whatever its place");
        // Rotate the bone 90 degrees about its origin: a vertex at the skin object's origin
        // (root-space (2, 0.5)) swings to (-0.5, 2) in root space, i.e. skin-local (-2.5, 1.5).
        let bone = root * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let m = skin_matrix(root, skin, bone, Affine2::IDENTITY);
        let moved = m.transform_point2(glam::Vec2::ZERO);
        assert!((moved - glam::Vec2::new(-2.5, 1.5)).length() < 1e-4, "{moved:?}");
        // With the skin object as its own root it is the old behaviour (the bone in the skin's space).
        let plain = skin_matrix(skin, skin, skin * Mat4::from_rotation_z(0.3), Affine2::IDENTITY);
        assert!((plain.matrix2.x_axis.y.atan2(plain.matrix2.x_axis.x) - 0.3).abs() < 1e-5 && plain.translation.length() < 1e-5);
    }

    #[test]
    fn uv_matches_the_sprite_rect_corners() {
        // A 2x1 sprite with its pivot at the centre, UVs from (0.25, 0.5) to (0.75, 1.0).
        let (size, pivot, rect) = ([2.0, 1.0], [1.0, 0.5], [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(skin_uv([-1.0, 0.5], rect, size, pivot), [0.25, 0.5], "top-left");
        assert_eq!(skin_uv([1.0, -0.5], rect, size, pivot), [0.75, 1.0], "bottom-right");
        assert_eq!(skin_uv([0.0, 0.0], rect, size, pivot), [0.5, 0.75]);
    }

    fn chain_scene(weight: f32, solver: IkSolverKind, limit: bool) -> (LoadedScene, usize, usize, usize) {
        // Root bone at the origin -> Mid at (1,0) -> Tip at (1,0) more; target above.
        let mut s = LoadedScene::empty();
        let target = object("Target", None, Vec3::new(1.2, 0.9, 0.0), vec![]);
        let tid = target.id;
        let bone = |lim: bool| -> Vec<Box<dyn Component>> {
            vec![Box::new(SpriteBone { bone_id: 0, limit_rotation: lim, min_angle: -170.0, max_angle: 170.0 })]
        };
        s.objects.push(object("Root", None, Vec3::ZERO, bone(limit)));
        s.objects.push(object("Mid", Some(0), Vec3::new(1.0, 0.0, 0.0), bone(limit)));
        let tip = object("Tip", Some(1), Vec3::new(1.0, 0.0, 0.0), vec![]);
        let tip_id = tip.id;
        s.objects.push(tip);
        s.objects.push(target);
        let ik = IkSolver2D { solver, tip: ObjectRef::new(tip_id), target: ObjectRef::new(tid), chain_length: 2, weight, iterations: 64, tolerance: 1e-3, flip: false };
        s.objects.push(object("Solver", None, Vec3::ZERO, vec![Box::new(ik)]));
        (s, 2, 3, 4)
    }

    fn tip_pos(s: &LoadedScene, tip: usize) -> Vec2 {
        let w = s.world_transform(tip);
        Vec2::new(w.w_axis.x, w.w_axis.y)
    }

    #[test]
    fn ik_moves_the_chain_to_the_target_for_every_solver() {
        for solver in [IkSolverKind::Ccd, IkSolverKind::Fabrik, IkSolverKind::Limb] {
            let (mut s, tip, target, solver_obj) = chain_scene(1.0, solver, false);
            let ids = s.id_index();
            s.run_ik(&[solver_obj], &ids);
            let t = tip_pos(&s, target);
            assert!((tip_pos(&s, tip) - t).length() < 5e-3, "{solver:?}: {:?} vs {t:?}", tip_pos(&s, tip));
            // Bone lengths are untouched.
            assert!((s.objects[1].translation.length() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn ik_weight_blends_and_zero_does_nothing() {
        let (mut s, tip, target, solver_obj) = chain_scene(0.0, IkSolverKind::Ccd, false);
        let before = tip_pos(&s, tip);
        let ids = s.id_index();
        s.run_ik(&[solver_obj], &ids);
        assert_eq!(tip_pos(&s, tip), before);
        let (mut s, tip, target2, solver_obj) = chain_scene(0.5, IkSolverKind::Ccd, false);
        let ids = s.id_index();
        s.run_ik(&[solver_obj], &ids);
        let d_half = (tip_pos(&s, tip) - tip_pos(&s, target2)).length();
        let d_start = (before - tip_pos(&s, target)).length();
        assert!(d_half < d_start && d_half > 5e-3, "half weight lands between: {d_half} (start {d_start})");
    }

    #[test]
    fn ik_respects_a_bone_limit_and_missing_objects_are_skipped() {
        let (mut s, _tip, _t, solver_obj) = chain_scene(1.0, IkSolverKind::Ccd, true);
        for o in &mut s.objects {
            if let Some(c) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<SpriteBone>()) {
                c.min_angle = -5.0;
                c.max_angle = 5.0;
            }
        }
        let ids = s.id_index();
        s.run_ik(&[solver_obj], &ids);
        for i in [0usize, 1] {
            let a = z_angle(s.objects[i].rotation).to_degrees();
            assert!(a.abs() <= 5.5, "joint {i} rotated {a} degrees past its limit");
        }
        // A solver with no target does nothing and does not panic.
        let mut s2 = LoadedScene::empty();
        s2.objects.push(object("S", None, Vec3::ZERO, vec![Box::new(IkSolver2D::default())]));
        let ids = s2.id_index();
        s2.run_ik(&[0], &ids);
    }

    #[test]
    fn bone_binding_by_explicit_ref_then_id_then_name() {
        let mut s = LoadedScene::empty();
        let mut skin_obj = object("Body", None, Vec3::ZERO, vec![]);
        let bone_a = object("a", Some(0), Vec3::ZERO, vec![Box::new(SpriteBone { bone_id: 77, ..Default::default() })]);
        let bone_b = object("Hand", Some(0), Vec3::ZERO, vec![Box::new(SpriteBone::default())]);
        let a_id = bone_a.id;
        let _ = a_id;
        let mut rig = Rig2D::default();
        rig.skeleton.add_bone(None, [0.0; 2], [1.0, 0.0], "Arm");
        rig.skeleton.add_bone(None, [0.0; 2], [1.0, 0.0], "Hand");
        rig.skeleton.add_bone(None, [0.0; 2], [1.0, 0.0], "Other");
        rig.skeleton.bones[0].id = 77;
        let explicit = object("pinned", None, Vec3::ZERO, vec![]);
        let explicit_id = explicit.id;
        skin_obj.components.push(Box::new(SpriteSkin { bones: vec![ObjectRef::NONE, ObjectRef::NONE, ObjectRef::new(explicit_id)], ..Default::default() }));
        s.objects.push(skin_obj);
        s.objects.push(bone_a);
        s.objects.push(bone_b);
        s.objects.push(explicit);
        let skin = find::<SpriteSkin>(&s.objects[0].components).unwrap().clone();
        let ids = s.id_index();
        let bound = s.bind_bones(0, &skin, &rig, &ids);
        assert_eq!(bound, vec![Some(1), Some(2), Some(3)], "bone 0 by id, 1 by name, 2 by explicit reference");
        // With auto_rebind off only explicit references bind.
        let off = SpriteSkin { auto_rebind: false, ..skin };
        assert_eq!(s.bind_bones(0, &off, &rig, &ids), vec![None, None, Some(3)]);
        let _ = <SpriteSkin as TypedComponent>::NAME;
    }
}
