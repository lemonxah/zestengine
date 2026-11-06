//! Who poses which skinned mesh, and how much posing an animated object does
//! this frame (see `anim_cull` for the visibility test and the pure rules).
//!
//! A *pose driver* is an object that poses its skeleton itself: it has an
//! Animator, Humanoid IK, face components (Face Tracking, Blend Shapes, Lip
//! Sync) or Mesh Cloth. A model split into one object per mesh (an imported
//! avatar: Body with blendshapes, Hair, Ears, clothes as siblings) has skinned
//! objects with no driver of their own; each follows the nearest driver of its
//! hierarchy that binds the same skeleton (the driver's palette applies) and
//! draws a GPU-skinned copy posed in the driver's pass, so it has a pose for
//! the skinned cluster path like the driver's own meshes. Only a skinned object
//! with no driver anywhere above it (or bone cloth of its own) keeps the shared
//! whole-scene pose.

use crate::anim_cull::SentPose;
use zestengine_core::{Animator, AnimatorCullingMode, BlendShapes, Component, FaceTracking, HumanoidIk, LipSync};

use super::*;

/// Whether an object with these components poses its skeleton itself.
pub(crate) fn is_pose_driver(components: &[Box<dyn Component>]) -> bool {
    components.iter().any(|c| {
        let a = c.as_any();
        a.is::<Animator>() || a.is::<HumanoidIk>() || a.is::<FaceTracking>() || a.is::<BlendShapes>() || a.is::<LipSync>()
    }) || crate::cloth::has_mesh_cloth(components)
}

/// Send one skinned copy its pose unless it already shows it (see [`SentPose`]):
/// blendshape weights (when the object drives them) and the palette. `settle`
/// sends it twice in one frame, so the skinner's previous-frame pose equals
/// the new one (an object that was hidden shows no motion from its stale pose).
#[allow(clippy::too_many_arguments)]
pub(super) fn send_gpu_pose(
    renderer: &mut Renderer,
    sent: &mut Vec<(usize, SentPose)>,
    handle: MeshHandle,
    palette: &[Mat4],
    key: u64,
    weights: Option<&[f32]>,
    settle: bool,
) -> bool {
    let k = match sent.iter().position(|s| s.0 == handle.index()) {
        Some(k) => k,
        None => {
            sent.push((handle.index(), SentPose::default()));
            sent.len() - 1
        }
    };
    if !sent[k].1.need(key) {
        return false;
    }
    if let Some(w) = weights {
        renderer.set_mesh_morph_weights(handle, w);
    }
    renderer.skin_mesh_gpu(handle, palette);
    if settle && sent[k].1.need(key) {
        renderer.skin_mesh_gpu(handle, palette);
    }
    true
}

impl LoadedScene {
    /// The skeleton every skinned render slot of object `i` binds to, when it
    /// has skinned slots and they agree.
    pub(super) fn slot_skeleton(&self, i: usize, skinned_by_mesh: &HashMap<usize, usize>) -> Option<usize> {
        let mut skel = None;
        for r in self.objects[i].render_slots() {
            let Some(&k) = skinned_by_mesh.get(&r.mesh) else { continue };
            let s = self.skinned_meshes[k].skeleton;
            match skel {
                None => skel = Some(s),
                Some(t) if t != s => return None,
                _ => {}
            }
        }
        skel
    }

    /// Active skinned objects without a pose driver of their own, each with the
    /// driver whose pose it follows: the nearest one (searching the subtree of
    /// each ancestor in turn, nearest ancestor first) that binds the same
    /// skeleton. Objects with bone cloth (it poses the shared skeleton) and
    /// re-skinned outfits (`skin_rebound`) are left out.
    pub(super) fn pose_followers(&self, active: &[bool], skinned_by_mesh: &HashMap<usize, usize>) -> HashMap<usize, usize> {
        let mut out = HashMap::new();
        if skinned_by_mesh.is_empty() {
            return out;
        }
        let n = self.objects.len();
        // Ancestor (or the driver itself) -> drivers in its subtree, by skeleton.
        let mut under: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for (i, is_active) in active.iter().enumerate() {
            if !is_active || self.skin_owner.contains_key(&self.objects[i].id) {
                continue;
            }
            let Some(skel) = self.slot_skeleton(i, skinned_by_mesh) else { continue };
            let comps = &self.objects[i].components;
            if is_pose_driver(comps) {
                let mut at = Some(i);
                for _ in 0..=n {
                    let Some(a) = at else { break };
                    under.entry(a).or_default().push((i, skel));
                    at = self.objects[a].parent;
                }
            } else if !crate::cloth::has_cloth(comps) {
                candidates.push((i, skel));
            }
        }
        if under.is_empty() {
            return out;
        }
        for (f, skel) in candidates {
            let mut at = self.objects[f].parent;
            for _ in 0..=n {
                let Some(a) = at else { break };
                if let Some(&(d, _)) = under.get(&a).and_then(|v| v.iter().find(|(_, s)| *s == skel)) {
                    out.insert(f, d);
                    break;
                }
                at = self.objects[a].parent;
            }
        }
        out
    }

    /// Objects some active Bone Attach or Weapon Hold names as its avatar.
    pub(super) fn followed_avatars(&self, attaches: &[(usize, zestengine_core::BoneAttach)], holds: &[(usize, zestengine_core::WeaponHold)]) -> Vec<usize> {
        let mut out: Vec<usize> = attaches
            .iter()
            .map(|(_, a)| a.avatar)
            .chain(holds.iter().map(|(_, h)| h.avatar))
            .filter_map(|r| r.0.and_then(|id| self.index_of(id)))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The culling mode of object `i`'s Animator (an object posed without one
    /// culls like the default).
    pub(super) fn culling_mode(&self, i: usize) -> AnimatorCullingMode {
        self.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Animator>()).map(|a| a.culling).unwrap_or_default()
    }

    /// Meshes of the objects that follow `owner`'s pose (`skin_rebound`), each
    /// with whether an active object draws it: an inactive outfit's copy stays
    /// ready but is not skinned until it is shown.
    pub(crate) fn follower_meshes_drawn(&self, owner: ObjectId, active: &[bool]) -> Vec<(usize, bool)> {
        let Some(list) = self.skin_followers.get(&owner) else { return Vec::new() };
        let mut out: Vec<(usize, bool)> = Vec::new();
        for id in list {
            let Some(o) = self.objects.iter().position(|o| o.id == *id) else { continue };
            let on = active.get(o).copied().unwrap_or(false);
            for r in self.objects[o].render_slots() {
                if !self.skinned_meshes.iter().any(|sm| sm.mesh_index == r.mesh) {
                    continue;
                }
                match out.iter_mut().find(|e| e.0 == r.mesh) {
                    Some(e) => e.1 |= on,
                    None => out.push((r.mesh, on)),
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(s: &mut LoadedScene, name: &str, parent: Option<usize>, mesh: Option<usize>, components: Vec<Box<dyn Component>>) -> usize {
        s.objects.push(SceneObject {
            id: ObjectId::new(),
            name: name.into(),
            render: mesh.map(|m| RenderInfo { mesh: m, material: 0 }),
            extra_render: vec![],
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            net_role: Default::default(),
            component_roles: Default::default(),
            parent,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        });
        s.objects.len() - 1
    }

    /// Skinned meshes 0..n, mesh `k` bound to skeleton `skels[k]`.
    fn meshes(s: &mut LoadedScene, skels: &[usize]) -> HashMap<usize, usize> {
        for (k, &sk) in skels.iter().enumerate() {
            s.skinned_meshes.push(SkinnedMesh::new(k, Vec::new(), sk));
        }
        s.skinned_meshes.iter().enumerate().map(|(k, sm)| (sm.mesh_index, k)).collect()
    }

    #[test]
    fn a_split_avatars_parts_follow_its_driver_with_the_same_skeleton() {
        let mut s = LoadedScene::empty();
        let by_mesh = meshes(&mut s, &[0, 0, 0, 1, 0, 0]);
        let world = obj(&mut s, "World", None, None, vec![]);
        // Two avatars of the same model under one parent.
        let a = obj(&mut s, "A", Some(world), None, vec![]);
        let a_body = obj(&mut s, "Body", Some(a), Some(0), vec![Box::new(BlendShapes::default())]);
        let a_hair = obj(&mut s, "Hair", Some(a), Some(1), vec![]);
        let a_prop = obj(&mut s, "Prop", Some(a), Some(3), vec![]);
        let b = obj(&mut s, "B", Some(world), None, vec![]);
        let b_body = obj(&mut s, "Body", Some(b), Some(4), vec![Box::new(Animator::default())]);
        let b_hair = obj(&mut s, "Hair", Some(b), Some(5), vec![]);
        // A loose skinned model with no driver anywhere above it.
        let loose = obj(&mut s, "Loose", None, Some(2), vec![]);
        let active = s.active_flags();
        let f = s.pose_followers(&active, &by_mesh);
        assert_eq!(f.get(&a_hair), Some(&a_body), "the nearest driver: its own avatar's, not the other one");
        assert_eq!(f.get(&b_hair), Some(&b_body));
        assert!(!f.contains_key(&a_prop), "another skeleton: not this driver's palette");
        assert!(!f.contains_key(&loose), "no driver above it: the shared pose");
        assert!(!f.contains_key(&a_body) && !f.contains_key(&b_body), "drivers pose themselves");
        // A part under its driver follows it too; an inactive one follows nobody.
        let child = obj(&mut s, "Strap", Some(a_body), Some(1), vec![]);
        s.objects[b_hair].enabled = false;
        let f = s.pose_followers(&s.active_flags(), &by_mesh);
        assert_eq!(f.get(&child), Some(&a_body));
        assert!(!f.contains_key(&b_hair));
    }

    #[test]
    fn bone_cloth_and_mixed_skeletons_keep_the_shared_pose() {
        let mut s = LoadedScene::empty();
        let by_mesh = meshes(&mut s, &[0, 0, 1]);
        let root = obj(&mut s, "Root", None, None, vec![]);
        let _body = obj(&mut s, "Body", Some(root), Some(0), vec![Box::new(Animator::default())]);
        let mixed = obj(&mut s, "Mixed", Some(root), Some(1), vec![]);
        s.objects[mixed].extra_render.push(RenderInfo { mesh: 2, material: 0 });
        assert_eq!(s.slot_skeleton(mixed, &by_mesh), None);
        let f = s.pose_followers(&s.active_flags(), &by_mesh);
        assert!(f.is_empty());
    }

    #[test]
    fn an_object_is_hidden_only_when_no_view_sees_it() {
        let mut s = LoadedScene::empty();
        s.mesh_bounds.push((Vec3::splat(-0.5), Vec3::splat(0.5)));
        let o = obj(&mut s, "A", None, Some(0), vec![]);
        let behind = Mat4::from_translation(Vec3::new(0.0, 0.0, 5.0));
        let ahead = Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0));
        assert!(s.animator_cull_view(None).is_none(), "no camera: nothing is culled");
        // Camera at the origin looking down -Z.
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        s.set_cull_view(&proj);
        let v = s.animator_cull_view(None).unwrap();
        assert!(s.animator_hidden(&v, o, &behind, &[0]));
        assert!(!s.animator_hidden(&v, o, &ahead, &[0]));
        // Just past the frustum's side: the margin for a pose that moved while hidden keeps it.
        let edge = Mat4::from_translation(Vec3::new(5.0 + 1.0, 0.0, -5.0));
        assert!(!s.animator_hidden(&v, o, &edge, &[0]));
        // A view the scene cannot describe (mirror, render texture, preview) turns culling off.
        let mirror = zestengine_render::ViewCullInfo { shadow_view_projs: Vec::new(), secondary: true };
        assert!(s.animator_cull_view(Some(&mirror)).is_none());
        // An XR eye looking back sees it.
        let eye = proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y);
        s.set_extra_cull_views(&[eye]);
        let v = s.animator_cull_view(None).unwrap();
        assert!(!s.animator_hidden(&v, o, &behind, &[0]));
        s.clear_cull_view();
        assert!(s.animator_cull_view(None).is_none());
    }

    #[test]
    fn the_default_culling_mode_applies_without_an_animator() {
        let mut s = LoadedScene::empty();
        let a = obj(&mut s, "A", None, None, vec![Box::new(Animator { culling: AnimatorCullingMode::CullCompletely, ..Default::default() })]);
        let b = obj(&mut s, "B", None, None, vec![Box::new(BlendShapes::default())]);
        assert_eq!(s.culling_mode(a), AnimatorCullingMode::CullCompletely);
        assert_eq!(s.culling_mode(b), AnimatorCullingMode::CullUpdateTransforms);
        assert_eq!(Animator::default().culling, AnimatorCullingMode::CullUpdateTransforms);
        // Scenes saved before the field existed load with the default.
        let old: Animator = ron::from_str("(clip: \"Idle\")").unwrap();
        assert_eq!(old.culling, AnimatorCullingMode::CullUpdateTransforms);
        let saved: Animator = ron::from_str("(culling: AlwaysAnimate)").unwrap();
        assert_eq!(saved.culling, AnimatorCullingMode::AlwaysAnimate);
    }
}
