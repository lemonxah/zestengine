//! Apply portable finger-pose settings to a loaded skeleton after body IK.
use glam::{Mat4, Quat, Vec3};
use zestengine_assets::Skeleton;
use zestengine_core::AvatarHands;

pub(crate) fn apply(skel: &Skeleton, locals: &mut [Mat4], hands: &AvatarHands) {
    let poses = hands.posed();
    let curl_axis = Vec3::from(hands.curl_axis).normalize_or(Vec3::X);
    let spread_axis = Vec3::from(hands.spread_axis).normalize_or(Vec3::Z);
    let weight = hands.weight.clamp(0.0, 1.0);
    if weight == 0.0 { return; }
    let by_name: std::collections::HashMap<&str, usize> = skel.joints.iter().enumerate().map(|(i, j)| (j.name.as_str(), i)).collect();
    let rest_world = hands.automatic_axes.then(|| skel.world_matrices(&skel.rest_locals()));
    for side in 0..2 {
        let sign = if side == 0 { 1.0 } else { -1.0 };
        let palm = rest_world.as_ref().and_then(|world| {
            let at = |finger: usize, joint: usize| by_name.get(hands.bones[side][finger][joint].as_str()).map(|i| world[*i].w_axis.truncate());
            let forward = (at(2, 1)? - at(2, 0)?).normalize_or_zero();
            let across = at(1, 0)? - at(4, 0)?;
            Some((forward.cross(across) * sign).normalize_or_zero())
        }).filter(|n| n.length_squared() > 0.5);
        for finger in 0..5 {
            for joint in 0..3 {
                let Some(&index) = by_name.get(hands.bones[side][finger][joint].as_str()) else { continue };
                let Some(local) = locals.get_mut(index) else { continue };
                let degrees = if finger == 0 { hands.thumb_curl_degrees[joint] } else { hands.curl_degrees[joint] };
                let bend = rest_world.as_ref().zip(palm).and_then(|(world, normal)| {
                    // Use the next joint (or this joint's incoming segment at the tip).
                    let next = by_name.get(hands.bones[side][finger][(joint + 1).min(2)].as_str())?;
                    let prev = by_name.get(hands.bones[side][finger][joint.saturating_sub(1)].as_str())?;
                    let forward = if joint < 2 { world[*next].w_axis.truncate() - world[index].w_axis.truncate() }
                        else { world[index].w_axis.truncate() - world[*prev].w_axis.truncate() };
                    let axis = forward.cross(normal).normalize_or_zero();
                    let local = world[index].to_scale_rotation_translation().1.inverse() * axis;
                    (local.length_squared() > 0.5).then_some(local.normalize())
                }).unwrap_or(curl_axis);
                let curl = Quat::from_axis_angle(bend, (poses[side].curl[finger].clamp(0.0, 1.0) * degrees).to_radians());
                let spread = if joint == 0 { Quat::from_axis_angle(spread_axis, (poses[side].spread[finger] * sign).to_radians()) } else { Quat::IDENTITY };
                let target = skel.joints[index].rest_rotation * spread * curl;
                let (scale, rotation, translation) = local.to_scale_rotation_translation();
                *local = Mat4::from_scale_rotation_translation(scale, rotation.slerp(target, weight).normalize(), translation);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::HandPose;
    #[test]
    fn worn_ugc_avatars_use_imported_finger_mappings_without_replacing_authored_settings() {
        use zestengine_core::{HumanoidIk, ObjectId, VrCharacter};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut skel = zestengine_assets::load_gltf(root.join("models/avatars/default/male.glb")).unwrap().skeletons.remove(0);
        let mut rig = crate::humanoid::auto_rig_import(&skel);
        let names = AvatarHands::default().bones;
        for (i, joint) in skel.joints.iter_mut().enumerate() {
            if names.iter().flatten().flatten().any(|name| *name == joint.name) {
                let new = format!("ugc_finger_{i}");
                for (_, mapped) in &mut rig.human { if *mapped == joint.name { *mapped = new.clone(); } }
                joint.name = new;
            }
        }
        skel.rig = Some(rig);
        let mut scene = crate::scene::LoadedScene::empty();
        let mut player = crate::scene::SceneObject::new_empty(ObjectId::new(), "VR Character");
        player.components.push(Box::new(VrCharacter::default()));
        player.components.push(Box::new(zestengine_core::NetworkAvatar::default()));
        let mut avatar = crate::scene::SceneObject::new_empty(ObjectId::new(), "Worn UGC");
        avatar.parent = Some(0); avatar.components.push(Box::new(HumanoidIk::default()));
        scene.objects.extend([player, avatar]);
        scene.add_test_skeleton(1, skel);
        ensure_defaults(&mut scene, true);
        let hands = scene.objects[1].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarHands>()).unwrap();
        assert!(hands.bones.iter().flatten().flatten().all(|name| name.starts_with("ugc_finger_")));
        hands.left = HandPose::Peace;
        hands.player = zestengine_core::ObjectRef::NONE;
        ensure_defaults(&mut scene, true);
        let hands: Vec<_> = scene.objects[1].components.iter().filter_map(|c| c.as_any().downcast_ref::<AvatarHands>()).collect();
        assert_eq!(hands.len(), 1);
        assert_eq!(hands[0].left, HandPose::Peace);
        assert_eq!(hands[0].player.id(), Some(scene.objects[0].id));
        assert_eq!(scene.objects[0].components.iter().filter(|c| c.as_any().is::<zestengine_core::NetworkVariables>()).count(), 1);
    }

    #[test]
    fn both_default_avatars_have_all_thirty_finger_bones_and_pose_independently() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for name in ["male", "female"] {
            let model = zestengine_assets::load_gltf(root.join(format!("models/avatars/default/{name}.glb"))).unwrap();
            let skel = &model.skeletons[0];
            let hands = AvatarHands { left: HandPose::Fist, right: HandPose::Point, ..Default::default() };
            for bone in hands.bones.iter().flatten().flatten() {
                assert!(skel.joints.iter().any(|j| j.name == *bone), "{name}: missing {bone}");
            }
            let mut locals = skel.rest_locals();
            let rest = locals.clone();
            apply(skel, &mut locals, &hands);
            let index = |name: &str| skel.joints.iter().position(|j| j.name == name).unwrap();
            assert!(Quat::from_mat4(&locals[index("index_01_l")]).angle_between(Quat::from_mat4(&rest[index("index_01_l")])) > 0.5);
            assert!(Quat::from_mat4(&locals[index("index_01_r")]).angle_between(Quat::from_mat4(&rest[index("index_01_r")])) < 0.001);
            assert!(Quat::from_mat4(&locals[index("middle_01_r")]).angle_between(Quat::from_mat4(&rest[index("middle_01_r")])) > 0.5);
            for (i, j) in skel.joints.iter().enumerate() {
                if !hands.bones.iter().flatten().flatten().any(|name| *name == j.name) { assert_eq!(locals[i], rest[i]); }
            }
        }
    }
}

/// Give any worn humanoid avatar the engine hand defaults, including UGC.
/// An authored component (even disabled) takes precedence.
pub(crate) fn ensure_defaults(scene: &mut crate::scene::LoadedScene, apply: bool) {
    if !apply { return; }
    let avatars: Vec<_> = scene.humanoid_avatars().into_iter().map(|avatar| {
        let root = avatar.network_avatar.or(avatar.tracking_root).unwrap_or(avatar.index);
        let map = avatar.skeleton.rig.as_ref().map(|r| r.human.clone()).unwrap_or_else(|| zestengine_assets::auto_map::auto_map(avatar.skeleton).to_human(avatar.skeleton));
        (avatar.index, root, avatar.network_avatar.is_some(), map)
    }).collect();
    for (index, root, networked, map) in avatars {
        let player = zestengine_core::ObjectRef::new(scene.objects[root].id);
        if networked && !scene.objects[root].components.iter().any(|c| zestengine_core::component_inner(c.as_ref()).as_any().is::<zestengine_core::NetworkVariables>()) {
            scene.objects[root].components.push(Box::new(zestengine_core::NetworkVariables::default()));
        }
        // Authored poses and mappings win, but a portable avatar need not know
        // the object ID of the player who will eventually wear it.
        if let Some(hands) = scene.objects[index].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarHands>()) {
            if hands.player.id().is_none() { hands.player = player; }
            continue;
        }
        if scene.objects[index].components.iter().any(|c| zestengine_core::component_inner(c.as_ref()).as_any().is::<AvatarHands>()) { continue; }
        let mut hands = AvatarHands { player, ..Default::default() };
        // The humanoid contract lays out 15 left then 15 right finger bones.
        for side in 0..2 { for finger in 0..5 { for joint in 0..3 {
            let role = zestengine_assets::human_rig::ALL[24 + side * 15 + finger * 3 + joint];
            hands.bones[side][finger][joint] = map.iter().find(|(bone, _)| *bone == role).map(|(_, name)| name.clone()).unwrap_or_default();
        }}}
        scene.objects[index].components.push(Box::new(hands));
    }
}
