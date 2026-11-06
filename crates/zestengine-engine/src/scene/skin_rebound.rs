//! Runtime half of `Merge Armature` skin rebinding.
//!
//! The avatar build tags every merged outfit mesh with a `Skin Rebound` component. When the
//! scene loads (or content is added), [`LoadedScene::apply_skin_rebounds`] plans the rebind of
//! that mesh's skeleton onto the avatar's ([`zestengine_assets::skin_rebind::plan_rebind`]),
//! rewrites the vertices (bone indices by humanoid bone / name, bind-pose correction, weights
//! untouched), uploads the result as a new skinned mesh bound to the avatar's skeleton and adds
//! any appended joints to that skeleton.
//!
//! Posing: skeletal animation runs per object with an Animator over that object's own skinned
//! slots. A rebound mesh stays on its outfit object (so the outfit's active flag, layer and
//! toggles still apply) and is registered as a *follower* of the avatar object: the avatar's
//! Animator poses the follower's mesh in the same pass, and the outfit object draws the posed
//! copy ([`LoadedScene::skin_owner`]). Joints the avatar's clips do not animate stay at their
//! rest pose under their parent.

use zestengine_assets::skin_rebind::{ExtraJoints, RebindOptions, plan_rebind};
use zestengine_core::SkinRebound;

use super::*;

/// What one application did, for logs and tests.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinReboundReport {
    pub objects: usize,
    pub meshes: usize,
    pub joints_added: usize,
    pub missing: Vec<String>,
    pub warnings: Vec<String>,
}

impl LoadedScene {
    /// Re-skin every not-yet-applied `Skin Rebound` object onto its avatar's skeleton.
    /// Safe to call after any load or content add: objects that were done are skipped.
    pub fn apply_skin_rebounds(&mut self, mut renderer: Option<&mut Renderer>) -> SkinReboundReport {
        let mut report = SkinReboundReport::default();
        let jobs: Vec<(usize, SkinRebound)> = self
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| !self.skin_owner.contains_key(&o.id))
            .filter_map(|(i, o)| o.components.iter().find_map(|c| c.as_any().downcast_ref::<SkinRebound>()).map(|r| (i, r.clone())))
            .collect();
        for (i, rb) in jobs {
            if self.rebind_object(renderer.as_deref_mut(), i, &rb, &mut report) {
                report.objects += 1;
            }
        }
        for w in &report.warnings {
            tracing::warn!("skin rebound: {w}");
        }
        report
    }

    /// The object whose Animator poses `id`'s skinned meshes when they were rebound onto another
    /// object's skeleton, or when `id` has no pose driver and follows its model's
    /// (`pose_groups`), else `id` itself.
    pub(crate) fn skin_pose_owner(&self, id: ObjectId) -> ObjectId {
        self.skin_owner.get(&id).or_else(|| self.pose_follow.get(&id)).copied().unwrap_or(id)
    }

    /// First skinned mesh (index into `skinned_meshes`) drawn by `obj` or, failing that, by one of its descendants.
    fn first_skinned_below(&self, obj: usize) -> Option<(usize, usize)> {
        let skinned = |o: usize| self.objects[o].render_slots().find_map(|r| self.skinned_meshes.iter().position(|sm| sm.mesh_index == r.mesh));
        if let Some(k) = skinned(obj) {
            return Some((obj, k));
        }
        (0..self.objects.len()).filter(|&o| o != obj && std::iter::successors(self.objects[o].parent, |&p| self.objects[p].parent).any(|p| p == obj)).find_map(|o| skinned(o).map(|k| (o, k)))
    }

    fn rebind_object(&mut self, mut renderer: Option<&mut Renderer>, i: usize, rb: &SkinRebound, report: &mut SkinReboundReport) -> bool {
        let name = self.objects[i].name.clone();
        let Some(avatar_obj) = self.objects.iter().position(|o| o.id.to_string() == rb.avatar) else {
            report.warnings.push(format!("{name}: the avatar object {} is not in the scene", rb.avatar));
            return false;
        };
        if avatar_obj == i {
            return false;
        }
        let Some((owner, avatar_sm)) = self.first_skinned_below(avatar_obj) else {
            report.warnings.push(format!("{name}: the avatar object draws no skinned mesh"));
            return false;
        };
        let avatar_skel = self.skinned_meshes[avatar_sm].skeleton;
        let opts = RebindOptions {
            prefix: rb.prefix.clone(),
            suffix: rb.suffix.clone(),
            mangle_names: rb.mangle_names,
            auto_align: rb.auto_align,
            extras: if rb.fold_extras { ExtraJoints::Fold } else { ExtraJoints::Append },
            correct_bind_pose: true,
        };
        // (slot is the object's extra slot index or None for `render`, the replacement mesh index)
        let mut replaced: Vec<(Option<usize>, usize)> = Vec::new();
        let slots: Vec<(Option<usize>, usize)> = std::iter::once(self.objects[i].render.map(|r| (None, r.mesh)))
            .flatten()
            .chain(self.objects[i].extra_render.iter().enumerate().map(|(k, r)| (Some(k), r.mesh)))
            .collect();
        for (slot, mesh) in slots {
            let Some(k) = self.skinned_meshes.iter().position(|sm| sm.mesh_index == mesh) else { continue };
            let outfit_skel = self.skinned_meshes[k].skeleton;
            if outfit_skel == avatar_skel {
                continue;
            }
            let plan = plan_rebind(&self.skeletons[avatar_skel], &self.skeletons[outfit_skel], &opts);
            for w in &plan.warnings {
                report.warnings.push(format!("{name}: {w}"));
            }
            report.missing.extend(plan.missing.iter().cloned());
            let mut vertices = self.skinned_meshes[k].base_vertices.clone();
            plan.apply(&mut vertices);
            let indices = self.mesh_geometry[mesh].1.clone();
            let data = zestengine_render::MeshData { vertices, indices, has_lightmap_uv: self.mesh_has_lightmap_uv.get(mesh).copied().unwrap_or(false) };
            let handle = match renderer.as_deref_mut() {
                Some(r) => match r.upload_mesh_skinned(&data) {
                    Ok(h) => h,
                    Err(e) => {
                        report.warnings.push(format!("{name}: uploading the rebound mesh failed: {e:#}"));
                        continue;
                    }
                },
                None => zestengine_render::MeshHandle::NONE,
            };
            let morphs = self.skinned_meshes[k].morphs.clone();
            let new_index = self.push_rebound_mesh(handle, data, avatar_skel, morphs);
            if plan.skeleton.joints.len() > self.skeletons[avatar_skel].joints.len() {
                report.joints_added += plan.skeleton.joints.len() - self.skeletons[avatar_skel].joints.len();
                self.skeletons[avatar_skel] = plan.skeleton;
            }
            replaced.push((slot, new_index));
            report.meshes += 1;
        }
        if replaced.is_empty() {
            return false;
        }
        for (slot, mesh) in replaced {
            match slot {
                None => {
                    if let Some(r) = self.objects[i].render.as_mut() {
                        r.mesh = mesh;
                    }
                }
                Some(k) => self.objects[i].extra_render[k].mesh = mesh,
            }
        }
        let (id, owner_id) = (self.objects[i].id, self.objects[owner].id);
        self.skin_owner.insert(id, owner_id);
        let followers = self.skin_followers.entry(owner_id).or_default();
        if !followers.contains(&id) {
            followers.push(id);
        }
        true
    }

    /// Register an uploaded skinned mesh with every per-mesh table (the same ones a model install fills).
    fn push_rebound_mesh(&mut self, handle: MeshHandle, data: zestengine_render::MeshData, skeleton: usize, morphs: Vec<zestengine_assets::MorphTarget>) -> usize {
        self.mesh_handles.push(handle);
        let index = self.mesh_handles.len() - 1;
        let mut sm = SkinnedMesh::new(index, data.vertices.clone(), skeleton);
        sm.morphs = morphs;
        self.skinned_meshes.push(sm);
        self.mesh_geometry.push((data.vertices.iter().map(|v| Vec3::from(v.position)).collect(), data.indices.clone()));
        self.mesh_sdf.push(None);
        self.mesh_infos.push(MeshInfo { vertices: data.vertices.len() as u32, triangles: data.indices.len() as u32 / 3 });
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for v in &data.vertices {
            lo = lo.min(Vec3::from(v.position));
            hi = hi.max(Vec3::from(v.position));
        }
        self.mesh_bounds.push((lo, hi));
        self.mesh_has_lightmap_uv.push(data.has_lightmap_uv);
        index
    }

    /// Skinned meshes of the objects that follow `owner`'s pose, for the Animator pass to add to its
    /// own slots. Active or not: the posed copies are made once, so a hidden outfit that is shown
    /// again is already there.
    #[allow(dead_code)] // follower mesh query used by the skin-rebound paths
    pub(crate) fn follower_meshes(&self, owner: ObjectId) -> Vec<usize> {
        let Some(list) = self.skin_followers.get(&owner) else { return Vec::new() };
        let mut out = Vec::new();
        for id in list {
            let Some(o) = self.objects.iter().position(|o| o.id == *id) else { continue };
            for r in self.objects[o].render_slots() {
                if self.skinned_meshes.iter().any(|sm| sm.mesh_index == r.mesh) && !out.contains(&r.mesh) {
                    out.push(r.mesh);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{MaterialRef, SceneFile};
    use zestengine_core::TypedComponent;

    fn entry(name: &str, parent: Option<usize>, path: Option<&str>, components: Vec<zestengine_assets::ComponentData>) -> SceneEntry {
        SceneEntry {
            id: ObjectId::new().to_string(),
            name: name.into(),
            source: match path {
                Some(p) => ObjectSource::Model { path: p.into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] },
                None => ObjectSource::Empty,
            },
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: Default::default(),
            layer: 0,
            material: MaterialRef::Model,
            extra_materials: vec![],
            renderer_split: false,
            parent,
            components,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    const FEMALE: &str = "models/avatars/default/female.glb";
    const MALE: &str = "models/avatars/default/male.glb";

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn load(file: &SceneFile) -> Option<LoadedScene> {
        if !root().join(FEMALE).exists() {
            eprintln!("default avatars not found: skipped");
            return None;
        }
        let registry = ComponentRegistry::with_builtins();
        Some(LoadedScene::load_scene_file(None, file, &root(), &registry, &mut ShaderLibrary::default()).expect("scene loads without a renderer"))
    }

    fn tag(avatar: &SceneEntry) -> zestengine_assets::ComponentData {
        let rb = SkinRebound { avatar: avatar.id.clone(), ..Default::default() };
        zestengine_assets::ComponentData::new(SkinRebound::NAME, ron::to_string(&rb).unwrap())
    }

    #[test]
    fn a_tagged_outfit_is_re_skinned_onto_the_avatar_and_follows_its_pose() {
        let body = entry("Body", None, Some(FEMALE), vec![]);
        let shirt = entry("Shirt", None, Some(MALE), vec![tag(&body)]);
        let file = SceneFile { entries: vec![body, shirt], ..Default::default() };
        let Some(scene) = load(&file) else { return };
        let (body_i, shirt_i) = (0, 1);
        // The load applied it: the shirt's mesh is a new skinned mesh on the avatar's skeleton.
        let body_mesh = scene.objects[body_i].render.unwrap().mesh;
        let avatar_skel = scene.skinned_meshes.iter().find(|sm| sm.mesh_index == body_mesh).unwrap().skeleton;
        let shirt_mesh = scene.objects[shirt_i].render.unwrap().mesh;
        let sm = scene.skinned_meshes.iter().find(|sm| sm.mesh_index == shirt_mesh).expect("the shirt mesh is skinned");
        assert_eq!(sm.skeleton, avatar_skel, "bound to the avatar's skeleton");
        let joints = scene.skeletons[avatar_skel].joints.len();
        assert!(sm.base_vertices.iter().all(|v| v.joints.iter().zip(v.weights).all(|(j, w)| w <= 0.0 || (*j as usize) < joints)), "every bone index is valid on the avatar");
        // Pose ownership: the avatar's object poses it, the outfit keeps drawing.
        let (body_id, shirt_id) = (scene.objects[body_i].id, scene.objects[shirt_i].id);
        assert_eq!(scene.skin_pose_owner(shirt_id), body_id);
        assert_eq!(scene.skin_pose_owner(body_id), body_id);
        assert_eq!(scene.follower_meshes(body_id), vec![shirt_mesh]);
        // Applying again is a no-op.
        let mut scene = scene;
        let again = scene.apply_skin_rebounds(None);
        assert_eq!((again.objects, again.meshes), (0, 0));
        // The outfit's own active flag still hides only the outfit.
        scene.objects[shirt_i].enabled = false;
        assert!(scene.follower_meshes(body_id).contains(&shirt_mesh), "the posed copy stays ready for when it is shown again");
        assert!(!scene.is_active(shirt_i) && scene.is_active(body_i));
    }

    #[test]
    fn a_missing_or_foreign_avatar_leaves_the_outfit_alone() {
        let body = entry("Body", None, Some(FEMALE), vec![]);
        let stray = zestengine_assets::ComponentData::new(SkinRebound::NAME, ron::to_string(&SkinRebound { avatar: "nope".into(), ..Default::default() }).unwrap());
        let shirt = entry("Shirt", None, Some(MALE), vec![stray]);
        let own = entry("Self", None, Some(MALE), vec![]);
        let own = SceneEntry { components: vec![tag(&own)], ..own };
        let file = SceneFile { entries: vec![body, shirt, own], ..Default::default() };
        let Some(mut scene) = load(&file) else { return };
        let before: Vec<usize> = scene.objects.iter().map(|o| o.render.unwrap().mesh).collect();
        let report = scene.apply_skin_rebounds(None);
        assert_eq!(report.objects, 0);
        assert!(report.warnings.iter().any(|w| w.contains("not in the scene")), "{:?}", report.warnings);
        assert_eq!(scene.objects.iter().map(|o| o.render.unwrap().mesh).collect::<Vec<_>>(), before);
        assert!(scene.skin_owner.is_empty());
    }
}
