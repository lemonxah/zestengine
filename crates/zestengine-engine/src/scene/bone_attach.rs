//! Attaching objects to bones from the editor: the hierarchy's bone rows are drop targets.
//! An attached object is parented to the model root and gets a [`BoneAttach`] whose offset
//! keeps its current world transform; dragging it out removes the component again.

use zestengine_core::{BoneAttach, ObjectRef};

use super::*;

/// The Bone Attach offset (position, XYZ Euler degrees) that puts an object at `object_world`
/// when `apply_bone_attaches` composes `bone_world * offset`. `None` for a bone with no
/// usable transform.
pub(crate) fn bone_offset(bone_world: Mat4, object_world: Mat4) -> Option<([f32; 3], [f32; 3])> {
    if !bone_world.is_finite() || !object_world.is_finite() || bone_world.determinant().abs() < 1e-18 {
        return None;
    }
    let (_, rotation, translation) = (bone_world.inverse() * object_world).to_scale_rotation_translation();
    let (rx, ry, rz) = rotation.to_euler(glam::EulerRot::XYZ);
    Some((translation.to_array(), [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()]))
}

impl LoadedScene {
    /// World matrix of joint `bone` on the first skinned model under `avatar`, in the pose
    /// `apply_bone_attaches` last saw (the rest pose before any frame ran), and that model's
    /// object index.
        fn bone_world_under(&self, avatar: usize, bone: &str, world: &[Mat4]) -> Option<(usize, Mat4)> {
        (0..self.objects.len()).filter(|&i| self.is_within(i, avatar)).find_map(|i| {
            let (skeleton, _) = self.skeleton_index_of(i)?;
            let skel = &self.skeletons[skeleton];
            let j = skel.joints.iter().position(|j| j.name == bone)?;
            let live = self.bone_poses.get(&self.objects[i].id).filter(|(s, m)| *s == skeleton && m.len() == skel.joints.len());
            let joint = match live {
                Some((_, m)) => m[j],
                None => skel.world_matrices(&skel.rest_locals())[j],
            };
            Some((i, world[i] * joint))
        })
    }

    /// Attach `child` to joint `bone` of the model under `owner`, keeping its world transform.
    /// The object goes under `owner` (unless it is already inside it).
        pub fn attach_to_bone(&mut self, child: usize, owner: usize, bone: &str) -> Result<(), String> {
        if child >= self.objects.len() || owner >= self.objects.len() {
            return Err("object is gone".into());
        }
        let world = self.world_transforms();
        let Some((model, bone_world)) = self.bone_world_under(owner, bone, &world) else {
            return Err(format!("no bone named {bone}"));
        };
        if self.is_within(model, child) {
            return Err("an object cannot attach to its own bone".into());
        }
        self.commit_bone_attach(child, owner, bone, bone_world, world[child])
    }

    /// The parenting and the component of [`Self::attach_to_bone`], given the bone's world matrix.
    fn commit_bone_attach(&mut self, child: usize, owner: usize, bone: &str, bone_world: Mat4, child_world: Mat4) -> Result<(), String> {
        if child == owner || self.is_within(owner, child) {
            return Err("an object cannot attach to its own child".into());
        }
        let Some((position, rotation)) = bone_offset(bone_world, child_world) else {
            return Err(format!("bone {bone} has no usable transform"));
        };
        if !self.is_within(child, owner) {
            self.set_parent(child, Some(owner));
        }
        let avatar = ObjectRef::new(self.objects[owner].id);
        let components = &mut self.objects[child].components;
        components.retain(|c| !c.as_any().is::<BoneAttach>());
        components.push(Box::new(BoneAttach { avatar, bone: bone.to_owned(), position, rotation }));
        Ok(())
    }

    /// Remove object `i`'s Bone Attach (it keeps the transform it has now). True when it had one.
        pub fn detach_bone_attach(&mut self, i: usize) -> bool {
        let Some(o) = self.objects.get_mut(i) else { return false };
        let before = o.components.len();
        o.components.retain(|c| !c.as_any().is::<BoneAttach>());
        o.components.len() != before
    }

    /// The bone object `i` is attached to, if it has a Bone Attach.
        pub fn attached_bone(&self, i: usize) -> Option<&str> {
        self.objects.get(i)?.components.iter().find_map(|c| c.as_any().downcast_ref::<BoneAttach>()).map(|a| a.bone.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(scene: &mut LoadedScene, name: &str, parent: Option<usize>, t: Vec3) -> usize {
        let mut o = SceneObject::new_empty(ObjectId::new(), name);
        o.parent = parent;
        o.translation = t;
        scene.objects.push(o);
        scene.objects.len() - 1
    }

    /// What `apply_bone_attaches` computes for an object's world matrix.
    fn applied(bone_world: Mat4, a: &BoneAttach) -> Mat4 {
        let [rx, ry, rz] = a.rotation.map(f32::to_radians);
        bone_world * Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz), Vec3::from(a.position))
    }

    fn attach_of(scene: &LoadedScene, i: usize) -> BoneAttach {
        scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<BoneAttach>()).expect("attached").clone()
    }

    fn close(a: Mat4, b: Mat4) {
        for (x, y) in a.to_cols_array().iter().zip(b.to_cols_array()) {
            assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn offset_reproduces_the_object_world() {
        let bone = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 2.0), Vec3::new(1.0, 2.0, -3.0));
        let object = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, -0.3, 0.7, 0.2), Vec3::new(0.5, 2.5, -2.0));
        let (position, rotation) = bone_offset(bone, object).unwrap();
        let a = BoneAttach { position, rotation, ..Default::default() };
        close(applied(bone, &a), object);
        assert!(bone_offset(Mat4::ZERO, object).is_none());
    }

    #[test]
    fn offset_through_a_scaled_bone() {
        // A centimetre-unit model: the bone world carries a 0.01 scale.
        let bone = Mat4::from_scale_rotation_translation(Vec3::splat(0.01), Quat::from_rotation_y(0.9), Vec3::new(0.2, 1.4, 0.1));
        let object = Mat4::from_rotation_translation(Quat::from_rotation_x(0.3), Vec3::new(0.25, 1.5, 0.1));
        let (position, rotation) = bone_offset(bone, object).unwrap();
        let got = applied(bone, &BoneAttach { position, rotation, ..Default::default() });
        assert!(got.w_axis.truncate().distance(object.w_axis.truncate()) < 1e-4);
    }

    #[test]
    fn attach_parents_to_the_owner_and_keeps_the_world_transform() {
        let mut s = LoadedScene::empty();
        let owner = obj(&mut s, "Avatar", None, Vec3::new(10.0, 0.0, 0.0));
        let other = obj(&mut s, "Crate", None, Vec3::new(3.0, 4.0, 5.0));
        let before = s.world_transforms()[other];
        let bone_world = Mat4::from_translation(Vec3::new(10.5, 1.0, 0.0));
        s.commit_bone_attach(other, owner, "hand_r", bone_world, before).unwrap();
        assert_eq!(s.objects[other].parent, Some(owner));
        assert_eq!(s.attached_bone(other), Some("hand_r"));
        let a = attach_of(&s, other);
        assert_eq!(a.avatar, ObjectRef::new(s.objects[owner].id));
        close(applied(bone_world, &a), before);
        // Local transform under the owner still gives the same world.
        close(s.world_transforms()[other], before);
    }

    #[test]
    fn reattach_replaces_and_detach_removes() {
        let mut s = LoadedScene::empty();
        let owner = obj(&mut s, "Avatar", None, Vec3::ZERO);
        let mesh = obj(&mut s, "Body", Some(owner), Vec3::ZERO);
        let sword = obj(&mut s, "Sword", Some(owner), Vec3::new(0.0, 1.0, 0.0));
        let w = s.world_transforms()[sword];
        s.commit_bone_attach(sword, owner, "hand_r", Mat4::from_translation(Vec3::Y), w).unwrap();
        s.commit_bone_attach(sword, owner, "hand_l", Mat4::from_translation(Vec3::Y), w).unwrap();
        assert_eq!(s.objects[sword].components.iter().filter(|c| c.as_any().is::<BoneAttach>()).count(), 1);
        assert_eq!(s.attached_bone(sword), Some("hand_l"));
        assert_eq!(s.objects[sword].parent, Some(owner), "already inside the owner: parent kept");
        assert!(s.detach_bone_attach(sword));
        assert!(!s.detach_bone_attach(sword));
        assert_eq!(s.attached_bone(sword), None);
        assert_eq!(s.attached_bone(mesh), None);
    }

    #[test]
    fn cannot_attach_an_object_to_its_own_descendant() {
        let mut s = LoadedScene::empty();
        let parent = obj(&mut s, "Parent", None, Vec3::ZERO);
        let child = obj(&mut s, "Child", Some(parent), Vec3::ZERO);
        assert!(s.commit_bone_attach(parent, child, "hand_r", Mat4::IDENTITY, Mat4::IDENTITY).is_err());
        assert!(s.commit_bone_attach(parent, parent, "hand_r", Mat4::IDENTITY, Mat4::IDENTITY).is_err());
        assert_eq!(s.attached_bone(parent), None);
    }

    #[test]
    fn attached_bone_survives_a_scene_round_trip_of_the_component() {
        let a = BoneAttach { avatar: ObjectRef::NONE, bone: "hand_r".into(), position: [0.1, 0.2, 0.3], rotation: [10.0, 20.0, 30.0] };
        let text = ron::to_string(&a).unwrap();
        assert!(a == ron::from_str::<BoneAttach>(&text).unwrap());
    }
}
