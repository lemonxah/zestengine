//! Automatic screen-size LOD (GEO-2): every static model with enough triangles
//! gets a generated chain, built on a worker thread the first time the model
//! is placed, and objects swap to a coarser level as their bounding sphere
//! covers less of the screen. Also the screen-size helpers the explicit
//! `LodComponent` mode and the animation distance LOD share.
//!
//! Skinned models are swapped too, through the object's own skinned copies: each
//! level the object selects is uploaded lazily as a GPU-skinned copy
//! ([`LevelCache`]) and only the selected level is skinned. Levels it stops
//! using are released after a grace period. The draw keeps the previous mesh
//! until the new level's copy exists, and the prepass and main pass read one
//! draw list, so both always see the same skin pool buffer (the depth-EQUAL main
//! pass needs identical vertices). Objects with Mesh Cloth keep their original
//! mesh. The animation update rate is throttled by distance as well (see
//! `LoadedScene::animation_interval`).

use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};

use zestengine_assets::auto_lod;
use zestengine_core::lod::{self, AUTO_LOD_STEPS};

use super::*;

/// Where an automatic chain's levels live in the scene's mesh arrays.
#[derive(Clone, Debug)]
pub(crate) struct AutoLod {
    /// First mesh index of the original model's template.
    pub origin: usize,
    /// Meshes in the model.
    pub count: usize,
    /// First mesh index of each level's template (level 0 = the original).
    pub bases: [Option<usize>; AUTO_LOD_STEPS.len()],
}

/// The mesh range an object draws instead of its own at the selected level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LodPick {
    pub base: usize,
    pub count: usize,
    /// First mesh of the model the object was placed from. With it the object
    /// keeps its own mesh offset (an automatic chain); without, slot `k` takes
    /// mesh `k` of the level (an authored LOD group).
    pub origin: Option<usize>,
}

impl LodPick {
    /// The mesh to draw for the object's render slot `k` whose own mesh is `own`.
    pub fn mesh_for(&self, own: usize, k: usize) -> usize {
        match self.origin {
            Some(o) if own >= o && own - o < self.count => self.base + (own - o),
            Some(_) => own,
            None => self.base + k.min(self.count.max(1) - 1),
        }
    }
}

/// Frames an unselected skinned level keeps its GPU copy before it is released
/// (several seconds at 60 fps; longer than any animation throttle interval, so
/// the selected level, touched on every full pass, never expires).
pub(crate) const SKIN_LEVEL_GRACE_FRAMES: u32 = 600;

/// Skinned render slots per object that can swap levels (a model has a few).
pub(crate) const MAX_SKIN_SLOTS: usize = 8;

/// The mesh base of level `want` of a chain, falling back to the next finer
/// level whose template installed. `None` = draw the original (level 0).
pub(crate) fn level_base(bases: &[Option<usize>], want: usize) -> Option<usize> {
    let level = (0..=want.min(bases.len().saturating_sub(1))).rev().find(|&l| bases[l].is_some())?;
    if level == 0 { None } else { bases[level] }
}

/// Per-object GPU copies of the skinned LOD levels it has used: lazily added,
/// stamped when selected, expired after a grace period.
#[derive(Debug)]
pub(crate) struct LevelCache<H: Copy> {
    entries: Vec<(usize, H, u32)>,
}

impl<H: Copy> Default for LevelCache<H> {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

impl<H: Copy> LevelCache<H> {
    pub fn contains(&self, mesh: usize) -> bool {
        self.entries.iter().any(|e| e.0 == mesh)
    }

    /// The copy for `mesh`, marking it used at `frame`.
    pub fn touch(&mut self, mesh: usize, frame: u32) -> Option<H> {
        let e = self.entries.iter_mut().find(|e| e.0 == mesh)?;
        e.2 = frame;
        Some(e.1)
    }

    pub fn insert(&mut self, mesh: usize, handle: H, frame: u32) {
        self.entries.push((mesh, handle, frame));
    }

    /// Remove and report copies unused for more than `grace` frames.
    pub fn expire(&mut self, frame: u32, grace: u32, mut release: impl FnMut(H)) {
        self.entries.retain(|&(_, h, used)| {
            let keep = frame.wrapping_sub(used) <= grace;
            if !keep {
                release(h);
            }
            keep
        });
    }

    pub fn handles(&self) -> impl Iterator<Item = H> + '_ {
        self.entries.iter().map(|e| e.1)
    }
}

/// The skinned copy to draw for the slot of source mesh `mesh`: what the pose
/// pass skinned for it last (`active`: the selected LOD level's copy, else the
/// original) on the GPU path, else the slot's own copy. The level's shared
/// template mesh is never posed, so drawing it is a bind pose (T-pose).
pub(crate) fn posed_copy<H: Copy>(slots: &[(usize, H)], active: &[H], gpu: bool, mesh: usize, valid: impl Fn(H) -> bool) -> Option<H> {
    let k = slots.iter().position(|(m, _)| *m == mesh)?;
    Some(gpu.then(|| active.get(k).copied().filter(|h| valid(*h))).flatten().unwrap_or(slots[k].1))
}

type Built = Option<Vec<(f32, zestengine_assets::Scene)>>;

/// Runtime state of the automatic LOD and the screen-size distance LOD.
pub(crate) struct LodRuntime {
    /// Automatic chains on (opt-out per object: an empty LOD Group).
    pub enabled: bool,
    /// tan(vertical FOV / 2) of the camera the host last set; `None` for an
    /// orthographic camera, where screen-size selection is off.
    pub tan_half_fov: Option<f32>,
    /// Chain per model path; `None` = the model gets no chain.
    pub(super) models: HashMap<String, Option<AutoLod>>,
    queue: VecDeque<String>,
    in_flight: Option<(String, Receiver<Built>)>,
    /// (object count, template count) at the last scan for new models.
    scan: (usize, usize),
    /// Skinning passes so far (staggers far animators across frames).
    pub frame: u32,
    /// Time not yet applied to a throttled animator.
    pub accum: HashMap<ObjectId, f32>,
    /// Last palette of a throttled animator, re-sent on skipped frames.
    pub palettes: HashMap<ObjectId, Vec<Mat4>>,
    /// View-projection of the host camera, for Animator culling (ANIM-4).
    pub cull_view_proj: Option<Mat4>,
    /// More cameras that see the scene (the XR eyes' combined view).
    pub cull_extra: Vec<Mat4>,
    /// Animated objects not fully posed last pass (hidden): shown again, they
    /// are posed and sent at once, never throttled.
    pub hidden: HashSet<ObjectId>,
    /// Palette hash per animated object last pass: an unchanged pose keeps its
    /// bounds and bone matrices.
    pub palette_hash: HashMap<ObjectId, u64>,
    /// Pose last given to each shared (whole-scene posed) mesh for the cluster path.
    pub shared_sent: HashMap<usize, crate::anim_cull::SentPose>,
}

impl Default for LodRuntime {
    fn default() -> Self {
        Self {
            enabled: true,
            tan_half_fov: None,
            models: HashMap::new(),
            queue: VecDeque::new(),
            in_flight: None,
            scan: (usize::MAX, usize::MAX),
            frame: 0,
            accum: HashMap::new(),
            palettes: HashMap::new(),
            cull_view_proj: None,
            cull_extra: Vec::new(),
            hidden: HashSet::new(),
            palette_hash: HashMap::new(),
            shared_sent: HashMap::new(),
        }
    }
}

impl LodRuntime {
    /// Forget a model's chain (it was reimported or unloaded).
    pub fn forget(&mut self, path: &str) {
        self.models.remove(path);
        self.scan = (usize::MAX, usize::MAX);
    }
}

/// Largest axis scale of a transform.
fn max_scale(m: &Mat4) -> f32 {
    m.x_axis.truncate().length().max(m.y_axis.truncate().length()).max(m.z_axis.truncate().length())
}

impl LoadedScene {
    /// Tell the LOD system which projection the host renders with.
    pub fn set_lod_projection(&mut self, proj: &Mat4) {
        let perspective = proj.w_axis.w == 0.0 && proj.y_axis.y.abs() > 1e-6;
        self.lods.tan_half_fov = perspective.then(|| 1.0 / proj.y_axis.y.abs());
    }

    /// Tell Animator culling which camera sees the scene (`proj * view`). Set it
    /// before `update_skinning` each frame: an object coming into view is posed
    /// in the same frame.
    pub fn set_cull_view(&mut self, view_proj: &Mat4) {
        self.lods.cull_view_proj = Some(*view_proj);
    }

    /// More cameras that see the scene this frame (the XR eyes).
    pub fn set_extra_cull_views(&mut self, view_projs: &[Mat4]) {
        self.lods.cull_extra.clear();
        self.lods.cull_extra.extend_from_slice(view_projs);
    }

    /// Turn Animator culling off until the next `set_cull_view` (a view the
    /// scene cannot describe sees it, e.g. XR eyes not reported yet).
    pub fn clear_cull_view(&mut self) {
        self.lods.cull_view_proj = None;
        self.lods.cull_extra.clear();
    }

    /// The views animators are culled against this frame; `None` (nothing is
    /// culled) without a host camera or while the renderer draws a view the
    /// scene cannot test (`info.secondary`). `info` = the renderer's last frame.
    pub(crate) fn animator_cull_view(&self, info: Option<&zestengine_render::ViewCullInfo>) -> Option<crate::anim_cull::CullView> {
        use crate::anim_cull::{CullView, ShadowFrustum, ShadowSource};
        let vp = self.lods.cull_view_proj?;
        if info.is_some_and(|i| i.secondary) {
            return None;
        }
        let shadows = self
            .gather_lights()
            .into_iter()
            .filter(|l| l.cast_shadows)
            .map(|l| match l.kind {
                zestengine_render::LightKind::Directional => ShadowSource::Directional(l.direction),
                _ => ShadowSource::Local { position: l.position, range: l.range },
            })
            .collect();
        let mut view = CullView::new(&vp, shadows);
        for e in &self.lods.cull_extra {
            view.add_view(e);
        }
        if let Some(info) = info {
            view.shadow_views = info.shadow_view_projs.iter().map(ShadowFrustum::new).collect();
        }
        Some(view)
    }

    /// Whether object `i` (skinned meshes `slots`, at `world`) is outside `view`.
    pub(crate) fn animator_hidden(&self, view: &crate::anim_cull::CullView, i: usize, world: &Mat4, slots: &[usize]) -> bool {
        let id = self.objects[i].id;
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &m in slots {
            let (a, b) = self
                .pose_bounds
                .get(&(id, m))
                .or_else(|| self.shared_pose_bounds.get(&m))
                .copied()
                .unwrap_or_else(|| self.mesh_aabb(m));
            mn = mn.min(a);
            mx = mx.max(b);
        }
        if mn.x > mx.x {
            return false;
        }
        let center = world.transform_point3((mn + mx) * 0.5);
        // The bounds are the last posed frame's: a hidden animation may have moved past them.
        let radius = (mx - mn).length() * 0.5 * max_scale(world) * crate::anim_cull::CULL_MARGIN_SCALE + crate::anim_cull::CULL_MARGIN_M;
        !view.visible(center, radius, self.objects[i].cast_shadows().casts())
    }

    /// Share of the screen height that `mesh`'s bounding sphere covers when
    /// placed at `world`. `None` before the host set a camera or for
    /// orthographic views.
    pub(crate) fn screen_coverage_of(&self, world: &Mat4, mesh: usize) -> Option<f32> {
        let cam = self.lod_camera?;
        let tan = self.lods.tan_half_fov?;
        let (mn, mx) = *self.mesh_bounds.get(mesh)?;
        let radius = (mx - mn).length() * 0.5 * max_scale(world);
        let center = world.transform_point3((mn + mx) * 0.5);
        Some(lod::screen_coverage(radius, center.distance(cam), tan))
    }

    /// The level an automatic chain gives object `i`, or `None` to draw its
    /// own mesh.
    pub(crate) fn auto_lod_pick(&mut self, i: usize, world: &Mat4) -> Option<LodPick> {
        if !self.lods.enabled || self.lods.models.is_empty() {
            return None;
        }
        let ObjectSource::Model { path, .. } = &self.objects[i].source else { return None };
        let auto = self.lods.models.get(path.as_str())?.as_ref()?;
        let render = self.objects[i].render?;
        if render.mesh < auto.origin || render.mesh - auto.origin >= auto.count {
            return None;
        }
        // A skinned object swaps only through its own GPU-skinned copies (see
        // `skin_lod_mesh`), so it needs a GPU Animator instance and no cloth.
        if self.skinned_meshes.iter().any(|sm| sm.mesh_index == render.mesh)
            && (crate::cloth::has_mesh_cloth(&self.objects[i].components) || !self.anim_instances.get(&self.objects[i].id).is_some_and(|inst| inst.gpu))
        {
            return None;
        }
        // Baked lightmaps belong to the original mesh's UVs.
        if self.baked.as_ref().is_some_and(|b| b.object_lightmap.contains_key(&i)) {
            return None;
        }
        // A mesh drawn from clusters has its own per-cluster LOD: no discrete swap.
        if self.cluster_geo.covers_mesh(render.mesh) {
            let id = self.objects[i].id;
            self.lod_state.remove(&id);
            return None;
        }
        let (origin, count, bases) = (auto.origin, auto.count, auto.bases);
        let coverage = self.screen_coverage_of(world, render.mesh)?;
        let id = self.objects[i].id;
        let prev = self.lod_state.get(&id).copied().flatten();
        let want = lod::select_by_coverage(&lod::auto_thresholds(), coverage, prev)?;
        self.lod_state.insert(id, Some(want));
        // A level whose template failed to install falls back to the next finer.
        Some(LodPick { base: level_base(&bases, want)?, count, origin: Some(origin) })
    }

    /// The mesh a skinned object's slot `own` is skinned as: its selected
    /// automatic level, or `None` for the original. Reads the level the draw
    /// pass last chose (`auto_lod_pick`), with the same eligibility.
    pub(crate) fn skin_lod_mesh(&self, i: usize, own: usize) -> Option<usize> {
        if !self.lods.enabled || self.lods.models.is_empty() {
            return None;
        }
        let o = &self.objects[i];
        let ObjectSource::Model { path, .. } = &o.source else { return None };
        if o.components.iter().any(|c| c.as_any().is::<zestengine_core::LodComponent>()) || crate::cloth::has_mesh_cloth(&o.components) {
            return None;
        }
        let auto = self.lods.models.get(path.as_str())?.as_ref()?;
        if own < auto.origin || own - auto.origin >= auto.count {
            return None;
        }
        let want = self.lod_state.get(&o.id).copied().flatten()?;
        level_base(&auto.bases, want).map(|b| b + (own - auto.origin))
    }

    /// The level mesh each skinned slot of object `i` should be skinned as
    /// (`None` = original), for up to [`MAX_SKIN_SLOTS`] slots.
    pub(super) fn skin_wants(&self, i: usize, id: ObjectId) -> [Option<usize>; MAX_SKIN_SLOTS] {
        let mut wants = [None; MAX_SKIN_SLOTS];
        if let Some(inst) = self.anim_instances.get(&id) {
            for (w, &(m, _)) in wants.iter_mut().zip(&inst.slots) {
                *w = self.skin_lod_mesh(i, m);
            }
        }
        wants
    }

    /// Pick, for each skinned slot of an animator, the copy to skin this pass:
    /// the wanted level's (uploaded now if it is new) or the original. Only that
    /// copy is skinned; unselected levels keep their stale pose until
    /// reselected. Takes the scene's fields separately so the caller can keep
    /// its skeleton borrowed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_skin_levels(
        renderer: &mut Renderer,
        anim_instances: &mut HashMap<ObjectId, AnimInstance>,
        skinned_meshes: &[SkinnedMesh],
        mesh_geometry: &[(Vec<Vec3>, Vec<u32>)],
        pose_bounds: &mut HashMap<(ObjectId, usize), (Vec3, Vec3)>,
        id: ObjectId,
        frame: u32,
        joints: usize,
        wants: &[Option<usize>; MAX_SKIN_SLOTS],
        skinned_by_mesh: &HashMap<usize, usize>,
    ) {
        let Some(inst) = anim_instances.get_mut(&id) else { return };
        let slots = inst.slots.len();
        inst.active.resize(slots, MeshHandle::NONE);
        for k in 0..slots {
            let (m, original) = inst.slots[k];
            let want = wants.get(k).copied().flatten().filter(|lm| skinned_by_mesh.contains_key(lm));
            let mut handle = original;
            if let Some(lm) = want {
                handle = match inst.levels.touch(lm, frame) {
                    Some(h) => h,
                    None => {
                        let sm = &skinned_meshes[skinned_by_mesh[&lm]];
                        let data = zestengine_render::MeshData {
                            vertices: sm.base_vertices.clone(),
                            indices: mesh_geometry[lm].1.clone(),
                            has_lightmap_uv: false,
                        };
                        let shapes: Vec<zestengine_render::MorphDeltas<'_>> = sm.morphs.iter().map(|t| t.deltas.as_slice()).collect();
                        match renderer.upload_mesh_gpu_skinned_morphed(&data, joints, &shapes) {
                            Ok(h) => {
                                inst.levels.insert(lm, h, frame);
                                h
                            }
                            Err(e) => {
                                tracing::warn!("skinned LOD upload: {e:#}");
                                original
                            }
                        }
                    }
                };
                // Culling bounds follow the pose for the level's mesh too.
                if let Some(b) = pose_bounds.get(&(id, m)).copied() {
                    pose_bounds.insert((id, lm), b);
                }
            }
            inst.active[k] = handle;
        }
    }

    /// Release the GPU copies of skinned levels no object has selected for the
    /// grace period. One device-idle wait covers the whole batch.
    pub(super) fn release_stale_skin_levels(&mut self, renderer: &mut Renderer) {
        let frame = self.lods.frame;
        let mut free: HashSet<usize> = HashSet::new();
        for inst in self.anim_instances.values_mut() {
            inst.levels.expire(frame, SKIN_LEVEL_GRACE_FRAMES, |h| {
                free.insert(h.index());
            });
        }
        if !free.is_empty()
            && let Err(e) = renderer.free_resources(&free, &HashSet::new(), &HashSet::new())
        {
            tracing::warn!("releasing skinned LOD copies: {e:#}");
        }
    }

    /// Whether object `id` can draw `mesh` as a skinned LOD level now: it has
    /// no original skinned slot of that model (static or unskinned) or the
    /// level's skinned copy already exists.
    pub(super) fn skin_level_ready(&self, id: ObjectId, original: usize, level: usize) -> bool {
        match self.anim_instances.get(&id) {
            Some(inst) if inst.slots.iter().any(|s| s.0 == original) => inst.levels.contains(level),
            _ => true,
        }
    }

    /// How many frames pass between pose updates of the animator on object `i`
    /// (1 = every frame). Only a plain Animator throttles: IK, face tracking,
    /// cloth, and anything a Bone Attach or Weapon Hold follows (`followed`)
    /// keep full rate.
    pub(crate) fn animation_throttle(&self, i: usize, world: &Mat4, mesh: usize, followed: bool) -> u32 {
        use zestengine_core::{Animator, BlendShapes, FaceTracking, HumanoidIk, LipSync};
        let comps = &self.objects[i].components;
        if !comps.iter().any(|c| c.as_any().is::<Animator>())
            || comps.iter().any(|c| {
                let a = c.as_any();
                a.is::<HumanoidIk>() || a.is::<FaceTracking>() || a.is::<BlendShapes>() || a.is::<LipSync>()
            })
            || crate::cloth::has_cloth(comps)
            || crate::cloth::has_mesh_cloth(comps)
        {
            return 1;
        }
        if followed {
            return 1;
        }
        self.animation_interval(world, mesh)
    }

    /// How many frames pass between animation updates of an animated object at
    /// `world` (1 = every frame). Unknown camera or bounds keep full rate.
    pub(crate) fn animation_interval(&self, world: &Mat4, mesh: usize) -> u32 {
        self.screen_coverage_of(world, mesh).map_or(1, lod::animation_interval)
    }

    /// Install finished chains and start building the next model's. Cheap
    /// when nothing is pending; call once per frame with the renderer.
    pub fn update_auto_lods(&mut self, renderer: &mut Renderer, project_root: &Path) {
        if !self.lods.enabled {
            return;
        }
        if let Some((path, rx)) = self.lods.in_flight.take() {
            match rx.try_recv() {
                Ok(built) => self.install_auto_lods(renderer, &path, built),
                Err(TryRecvError::Empty) => self.lods.in_flight = Some((path, rx)),
                Err(TryRecvError::Disconnected) => {
                    self.lods.models.insert(path, None);
                }
            }
        }
        let key = (self.objects.len(), self.model_templates.len());
        if key != self.lods.scan {
            self.lods.scan = key;
            for o in &self.objects {
                let ObjectSource::Model { path, .. } = &o.source else { continue };
                if o.render.is_none()
                    || self.lods.models.contains_key(path)
                    || self.lods.queue.contains(path)
                    || self.lods.in_flight.as_ref().is_some_and(|f| f.0 == *path)
                    || !self.model_templates.contains_key(path)
                    // An authored LOD Group (even an empty one) takes over.
                    || o.components.iter().any(|c| c.as_any().is::<zestengine_core::LodComponent>())
                {
                    continue;
                }
                self.lods.queue.push_back(path.clone());
            }
        }
        if self.lods.in_flight.is_none()
            && let Some(path) = self.lods.queue.pop_front()
        {
            let full = project_root.join(&path);
            let (tx, rx) = std::sync::mpsc::channel();
            let spawned = std::thread::Builder::new().name("auto-lod".into()).spawn(move || {
                let built = auto_lod::load_or_build(&full);
                let _ = tx.send(built);
            });
            match spawned {
                Ok(_) => self.lods.in_flight = Some((path, rx)),
                Err(e) => {
                    tracing::warn!("auto LOD thread: {e}");
                    self.lods.models.insert(path, None);
                }
            }
        }
    }

    fn install_auto_lods(&mut self, renderer: &mut Renderer, path: &str, built: Built) {
        let origin = self.model_templates.get(path).map(|t| t.0);
        let (Some(levels), Some(origin)) = (built, origin) else {
            self.lods.models.insert(path.to_owned(), None);
            return;
        };
        let count = levels.first().map_or(0, |l| l.1.meshes.len());
        let mut bases = [None; AUTO_LOD_STEPS.len()];
        bases[0] = Some(origin);
        for (n, (keep, scene)) in levels.iter().enumerate() {
            let key = auto_lod::level_key(path, *keep);
            match self.install_model_template(renderer, &key, scene, None) {
                Ok(()) => bases[n + 1] = self.model_templates.get(&key).map(|t| t.0),
                Err(e) => tracing::warn!("auto LOD {key}: {e:#}"),
            }
        }
        tracing::info!("auto LOD for {path}: {} levels", bases.iter().flatten().count());
        self.lods.models.insert(path.to_owned(), Some(AutoLod { origin, count, bases }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_keeps_the_objects_mesh_offset() {
        let auto = LodPick { base: 40, count: 3, origin: Some(10) };
        assert_eq!(auto.mesh_for(10, 0), 40);
        assert_eq!(auto.mesh_for(12, 0), 42, "a later mesh of the model keeps its offset");
        assert_eq!(auto.mesh_for(99, 0), 99, "a mesh outside the model is left alone");
        let authored = LodPick { base: 7, count: 2, origin: None };
        assert_eq!(authored.mesh_for(123, 0), 7);
        assert_eq!(authored.mesh_for(123, 5), 8, "slots past the level clamp to its last mesh");
    }

    #[test]
    fn level_base_falls_back_to_the_next_finer_installed_level() {
        let bases = [Some(5), Some(40), None, Some(80), Some(90)];
        assert_eq!(level_base(&bases, 0), None, "level 0 is the original");
        assert_eq!(level_base(&bases, 1), Some(40));
        assert_eq!(level_base(&bases, 2), Some(40), "level 2 failed to install");
        assert_eq!(level_base(&bases, 4), Some(90));
        assert_eq!(level_base(&bases, 9), Some(90), "past the end clamps");
        assert_eq!(level_base(&[Some(5), None], 1), None, "nothing finer than the original left");
    }

    #[test]
    fn a_far_animator_draws_the_copy_that_was_posed_not_the_levels_bind_pose() {
        // Slot of source mesh 10 owns copy 100; its coarse level's copy is 200.
        let slots = [(10usize, 100u32), (11, 101)];
        let ok = |h: u32| h != u32::MAX;
        // Level selected: the pose pass skinned 200, and that is what is drawn
        // (the draw looks the slot up by its own mesh, not the level's index).
        assert_eq!(posed_copy(&slots, &[200, 101], true, 10, ok), Some(200));
        assert_eq!(posed_copy(&slots, &[100, 101], true, 10, ok), Some(100));
        // CPU-skinned instances always draw the slot's own buffer.
        assert_eq!(posed_copy(&slots, &[200, 101], false, 10, ok), Some(100));
        // Not yet filled in: the original. Unknown mesh: the shared mesh.
        assert_eq!(posed_copy(&slots, &[u32::MAX, 101], true, 10, ok), Some(100));
        assert_eq!(posed_copy(&slots, &[200, 101], true, 99, ok), None);
    }

    #[test]
    fn level_copies_are_created_once_used_and_expire_after_the_grace_period() {
        let mut c: LevelCache<u32> = LevelCache::default();
        assert!(c.touch(7, 0).is_none() && !c.contains(7));
        c.insert(7, 100, 0);
        c.insert(8, 101, 0);
        assert_eq!(c.touch(7, 590), Some(100), "selected again: stamped");
        let mut freed = Vec::new();
        c.expire(SKIN_LEVEL_GRACE_FRAMES, SKIN_LEVEL_GRACE_FRAMES, |h| freed.push(h));
        assert!(freed.is_empty(), "still inside the grace period");
        c.expire(SKIN_LEVEL_GRACE_FRAMES + 1, SKIN_LEVEL_GRACE_FRAMES, |h| freed.push(h));
        assert_eq!(freed, vec![101], "only the unused level is released");
        assert!(c.contains(7) && !c.contains(8));
        assert_eq!(c.handles().collect::<Vec<_>>(), vec![100]);
        // The frame counter wrapping must not release a fresh copy.
        c.insert(9, 102, u32::MAX - 2);
        c.expire(3, SKIN_LEVEL_GRACE_FRAMES, |h| freed.push(h));
        assert!(!freed.contains(&102));
    }

    #[test]
    fn projection_sets_the_fov_scale_and_ignores_ortho() {
        let mut s = LoadedScene::empty();
        s.set_lod_projection(&Mat4::perspective_rh(60f32.to_radians(), 1.5, 0.1, 100.0));
        assert!((s.lods.tan_half_fov.unwrap() - 30f32.to_radians().tan()).abs() < 1e-4);
        s.set_lod_projection(&Mat4::orthographic_rh(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0));
        assert!(s.lods.tan_half_fov.is_none());
    }

    #[test]
    fn coverage_needs_a_camera_and_shrinks_with_distance() {
        let mut s = LoadedScene::empty();
        s.mesh_bounds.push((Vec3::splat(-1.0), Vec3::splat(1.0)));
        s.set_lod_projection(&Mat4::perspective_rh(60f32.to_radians(), 1.5, 0.1, 100.0));
        assert!(s.screen_coverage_of(&Mat4::IDENTITY, 0).is_none(), "no camera yet");
        s.lod_camera = Some(Vec3::new(0.0, 0.0, 10.0));
        let near = s.screen_coverage_of(&Mat4::IDENTITY, 0).unwrap();
        s.lod_camera = Some(Vec3::new(0.0, 0.0, 100.0));
        let far = s.screen_coverage_of(&Mat4::IDENTITY, 0).unwrap();
        assert!(near > far * 5.0);
        // Scaling the object up covers more.
        let big = s.screen_coverage_of(&Mat4::from_scale(Vec3::splat(4.0)), 0).unwrap();
        assert!(big > far * 3.0);
    }
}
