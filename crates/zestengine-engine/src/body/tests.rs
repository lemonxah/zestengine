use super::*;
use crate::scene::SceneObject;
use zestengine_assets::{Joint, ObjectSource, Skeleton};
use zestengine_core::{Component, HumanoidIk, NetworkAvatar, VrPlayer};

fn obj(scene: &mut LoadedScene, parent: Option<usize>, components: Vec<Box<dyn Component>>) -> usize {
    scene.objects.push(SceneObject {
        net_role: Default::default(),
        component_roles: Default::default(),
        id: ObjectId::new(),
        name: String::new(),
        render: None,
        extra_render: Vec::new(),
        source: ObjectSource::Empty,
        enabled: true,
        static_geometry: false,
        layer: 0,
        parent,
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        components,
    });
    scene.objects.len() - 1
}

fn joint(name: &str, parent: Option<usize>, t: Vec3) -> Joint {
    Joint { name: name.into(), parent, inverse_bind: Mat4::IDENTITY, rest_translation: t, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE }
}

/// A T-pose humanoid facing +Z: hips 1 m, head bone 1.65 m, hands 1.3 m apart.
fn humanoid() -> Skeleton {
    let mut joints = vec![
        joint("Hips", None, Vec3::new(0.0, 1.0, 0.0)),
        joint("Spine", Some(0), Vec3::new(0.0, 0.15, 0.0)),
        joint("Chest", Some(1), Vec3::new(0.0, 0.2, 0.0)),
        joint("Neck", Some(2), Vec3::new(0.0, 0.2, 0.0)),
        joint("Head", Some(3), Vec3::new(0.0, 0.1, 0.0)),
    ];
    for (side, x) in [("Left", 1.0f32), ("Right", -1.0)] {
        let base = joints.len();
        joints.push(joint(&format!("{side}Shoulder"), Some(2), Vec3::new(0.05 * x, 0.15, 0.0)));
        joints.push(joint(&format!("{side}UpperArm"), Some(base), Vec3::new(0.1 * x, 0.0, 0.0)));
        joints.push(joint(&format!("{side}LowerArm"), Some(base + 1), Vec3::new(0.25 * x, 0.0, 0.0)));
        joints.push(joint(&format!("{side}Hand"), Some(base + 2), Vec3::new(0.25 * x, 0.0, 0.0)));
        joints.push(joint(&format!("{side}UpperLeg"), Some(0), Vec3::new(0.1 * x, -0.05, 0.0)));
        joints.push(joint(&format!("{side}LowerLeg"), Some(base + 4), Vec3::new(0.0, -0.45, 0.0)));
        joints.push(joint(&format!("{side}Foot"), Some(base + 5), Vec3::new(0.0, -0.42, 0.0)));
    }
    Skeleton { joints, rig: None, ..Default::default() }
}

/// The fixture's eye height (no eye bones: the head bone plus a little).
const AVATAR_EYE: f32 = 1.65 + 0.07 * (1.65 / 1.6);

fn frame(tracking: bool) -> BodyFrame {
    BodyFrame { dt: 1.0 / 60.0, tracking, apply: true, triggers: [false; 2] }
}

fn pose(x: f32, y: f32, z: f32) -> Option<(Vec3, Quat)> {
    Some((Vec3::new(x, y, z), Quat::IDENTITY))
}

fn device(id: &str, native: usize, p: Option<(Vec3, Quat)>) -> TrackerDevice {
    TrackerDevice { id: id.into(), label: id.into(), native: Some(native), pose: p }
}

#[test]
fn assemble_applies_roles_and_the_floor_offset() {
    let mut body = BodyTracking::default();
    body.profile.floor_offset = 0.02;
    body.profile.set_role("xr:waist", Some(6));
    body.profile.set_role("slimevr:a", None);
    let devices = vec![
        device("xr:waist", 3, pose(0.0, 1.0, 0.0)),
        device("slimevr:a", 4, pose(-0.1, 0.1, 0.0)),
        device("slimevr:b", 5, pose(0.1, 0.1, 0.0)),
        device("slimevr:c", 5, pose(0.2, 0.1, 0.0)),
    ];
    let t = body.assemble(pose(0.0, 1.7, 0.0), None, None, devices);
    assert!((t.head.unwrap().0.y - 1.68).abs() < 1e-6, "floor corrected");
    assert!((t.chest.unwrap().0.y - 0.98).abs() < 1e-6, "the waist tracker was moved to the chest");
    assert!(t.hips.is_none() && t.left_foot.is_none(), "ignored device drives nothing");
    assert!((t.right_foot.unwrap().0.x - 0.1).abs() < 1e-6, "the first device on a slot wins");
}

/// A 1.75 m player's T-pose: (true slot, position) of each tracker.
fn tpose_devices() -> Vec<(usize, Vec3)> {
    vec![
        (3, Vec3::new(0.0, 0.96, 0.08)),
        (4, Vec3::new(-0.12, 0.09, -0.04)),
        (5, Vec3::new(0.11, 0.1, -0.05)),
        (9, Vec3::new(-0.1, 0.5, -0.08)),
        (10, Vec3::new(0.11, 0.48, -0.07)),
    ]
}

#[test]
fn full_calibration_measures_assigns_roles_and_checks_the_floor() {
    let mut body = BodyTracking::default();
    let mut scene = LoadedScene::empty();
    // The runtime reports the feet swapped and the knees as elbows.
    let natives = [3usize, 5, 4, 7, 8];
    let devices = |k: usize| -> Vec<TrackerDevice> {
        tpose_devices()
            .iter()
            .zip(natives)
            .enumerate()
            .map(|(n, ((_, p), native))| device(&format!("xr:{n}"), native, Some((*p + Vec3::Y * 0.001 * (k % 3) as f32, Quat::IDENTITY))))
            .collect()
    };
    body.command(BodyCommand::Calibrate);
    let mut k = 0;
    while !matches!(body.calibration.as_ref().map(|c| &c.step), Some(Step::Floor) | None) && k < 1200 {
        let head = pose(0.0, 1.64 + 0.002 * ((k % 4) as f32 - 1.5), 0.0);
        let raw = body.assemble(head, pose(-0.81, 1.44, 0.0), pose(0.81, 1.44, 0.0), devices(k));
        body.update(&mut scene, &raw, frame(true));
        k += 1;
    }
    assert!(matches!(body.calibration.as_ref().map(|c| &c.step), Some(Step::Floor)), "reached the floor check: {:?}", body.status);
    assert!((body.profile.measured_eye_height.unwrap() - 1.64).abs() < 0.003);
    assert!((body.profile.arm_span.unwrap() - 1.62).abs() < 1e-3);
    for (n, (slot, _)) in tpose_devices().iter().enumerate() {
        assert_eq!(body.profile.role_of(&format!("xr:{n}")), Some(Some(*slot)), "device {n}");
    }
    let tpose = body.profile.tpose_targets().expect("T-pose stored");
    assert!(tpose.left_foot.unwrap().0.x < 0.0 && tpose.right_knee.unwrap().0.x > 0.0);
    assert!(tpose.left_elbow.is_none());

    // A trigger pulled with the controller at chest height is a click, not the floor.
    let raw = body.assemble(pose(0.0, 1.64, 0.0), pose(-0.2, 1.1, -0.3), pose(0.2, 1.1, -0.3), devices(0));
    body.update(&mut scene, &raw, BodyFrame { triggers: [true, false], ..frame(true) });
    body.update(&mut scene, &raw, frame(true));
    assert!(body.calibration.is_some() && body.profile.floor_offset == 0.0);
    // On the floor: the play area floor was 1.5 cm low.
    let raw = body.assemble(pose(0.0, 1.64, 0.0), pose(0.3, 0.05, -0.3), pose(0.2, 1.1, -0.3), devices(0));
    body.update(&mut scene, &raw, BodyFrame { triggers: [true, false], ..frame(true) });
    assert!(body.calibration.is_none(), "{:?}", body.status);
    assert!((body.profile.floor_offset - 0.015).abs() < 1e-4);
    assert!((body.profile.measured_eye_height.unwrap() - 1.625).abs() < 0.003, "measured against the corrected floor");
    assert!((body.profile.tpose_targets().unwrap().hips.unwrap().0.y - 0.946).abs() < 3e-3);
}

#[test]
fn quick_recalibrate_keeps_roles_and_a_moving_head_fails_the_height() {
    let mut body = BodyTracking::default();
    body.profile.set_role("xr:a", Some(6));
    let mut scene = LoadedScene::empty();
    body.command(BodyCommand::QuickCalibrate);
    for _ in 0..300 {
        let raw = body.assemble(pose(0.0, 1.6, 0.0), pose(-0.8, 1.4, 0.0), pose(0.8, 1.4, 0.0), vec![device("xr:a", 3, pose(0.0, 0.95, 0.0))]);
        body.update(&mut scene, &raw, frame(true));
    }
    assert!(body.calibration.is_none() && body.calibrated());
    assert_eq!(body.profile.role_of("xr:a"), Some(Some(6)), "roles kept");
    assert!(body.profile.tpose_targets().unwrap().chest.is_some());

    body.command(BodyCommand::MeasureHeight);
    for k in 0..400 {
        let raw = body.assemble(pose(0.0, 1.2 + k as f32 * 0.003, 0.0), None, None, Vec::new());
        body.update(&mut scene, &raw, frame(true));
    }
    assert!(body.calibration.is_none());
    assert!(body.status.as_deref().is_some_and(|s| s.contains("not measured")), "{:?}", body.status);
    assert!(body.profile.measured_eye_height.is_none());
}

/// A VR Player rig and a humanoid avatar (child of the rig when `child`).
fn rig_scene(mode: AvatarScaling, child: bool) -> (LoadedScene, usize, usize) {
    let mut scene = LoadedScene::empty();
    let rig = obj(&mut scene, None, vec![Box::new(VrPlayer { avatar_scaling: mode, ..Default::default() })]);
    let avatar = obj(&mut scene, child.then_some(rig), vec![Box::new(HumanoidIk::default())]);
    scene.add_test_skeleton(avatar, humanoid());
    (scene, rig, avatar)
}

fn head_at(y: f32) -> TrackerTargets {
    TrackerTargets { head: Some((Vec3::new(0.0, y, 0.0), Quat::from_rotation_y(std::f32::consts::PI))), ..Default::default() }
}

#[test]
fn scale_avatar_fits_the_avatar_to_the_player_and_stays_put() {
    let (mut scene, rig, avatar) = rig_scene(AvatarScaling::ScaleAvatar, false);
    let mut body = BodyTracking::default();
    body.profile.measured_eye_height = Some(1.8);
    let want = 1.8 / AVATAR_EYE;
    let mut targets = TrackerTargets::default();
    for _ in 0..3 {
        targets = body.update(&mut scene, &head_at(1.8), frame(true));
    }
    assert!((scene.objects[avatar].scale.x - want).abs() < 1e-4, "{}", scene.objects[avatar].scale);
    assert_eq!(scene.objects[rig].scale, Vec3::ONE);
    assert!((body.info().avatar_scale - want).abs() < 1e-4 && (scene.local_body.avatar_scale - want).abs() < 1e-4);
    assert!((body.info().player_height - (1.8 + zestengine_core::EYE_TO_CROWN)).abs() < 1e-5);
    // The headset drives the head bone of the scaled avatar.
    let head = targets.head.unwrap().0;
    assert!((head.y - (1.8 - 0.07 * want)).abs() < 0.01, "head target {head}");
    // Without a headset the avatar goes back to its authored size.
    body.update(&mut scene, &head_at(1.8), frame(false));
    assert!((scene.objects[avatar].scale.x - 1.0).abs() < 1e-5);
}

#[test]
fn scale_player_scales_the_rig_and_keeps_a_child_avatar_at_its_size() {
    let (mut scene, rig, avatar) = rig_scene(AvatarScaling::ScalePlayer, true);
    let mut body = BodyTracking::default();
    body.profile.height_override_cm = Some(150.0 + zestengine_core::EYE_TO_CROWN * 100.0);
    let k = AVATAR_EYE / 1.5;
    for _ in 0..3 {
        body.update(&mut scene, &head_at(1.5), frame(true));
    }
    assert!((scene.objects[rig].scale.x - k).abs() < 1e-4, "rig {}", scene.objects[rig].scale);
    let world = scene.world_transforms();
    assert!((world[avatar].x_axis.length() - 1.0).abs() < 1e-4, "avatar keeps its size");
    let rig_view = scene.vr_player_rig().unwrap();
    assert!((rig_view.scale - k).abs() < 1e-4, "the eyes render from the scaled play space");
    assert!((rig_view.stage_to_world(Vec3::Y * 1.5).y - AVATAR_EYE).abs() < 1e-3, "the headset sits at the avatar's eyes");
    assert!((body.fit.model_to_stage - 1.5 / AVATAR_EYE).abs() < 1e-4);

    // Switching the mode off restores both.
    if let Some(p) = scene.objects[rig].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<VrPlayer>()) {
        p.avatar_scaling = AvatarScaling::Off;
    }
    body.update(&mut scene, &head_at(1.5), frame(true));
    assert!((scene.objects[rig].scale.x - 1.0).abs() < 1e-5 && (scene.objects[avatar].scale.x - 1.0).abs() < 1e-5);
}

#[test]
fn a_scale_set_by_the_game_becomes_the_new_authored_size() {
    let (mut scene, _, avatar) = rig_scene(AvatarScaling::ScaleAvatar, false);
    let mut body = BodyTracking::default();
    body.profile.measured_eye_height = Some(AVATAR_EYE * 1.2);
    body.update(&mut scene, &head_at(1.8), frame(true));
    assert!((scene.objects[avatar].scale.x - 1.2).abs() < 1e-4);
    // Gameplay doubles the avatar: the fit keeps fitting the player's eyes.
    scene.objects[avatar].scale = Vec3::splat(2.0);
    body.update(&mut scene, &head_at(1.8), frame(true));
    assert!((scene.objects[avatar].scale.x - 1.2).abs() < 1e-4, "{}", scene.objects[avatar].scale);
}

#[test]
fn remote_avatars_take_the_scale_their_owner_replicates() {
    let mut scene = LoadedScene::empty();
    let net = obj(&mut scene, None, vec![Box::new(NetworkAvatar { remote: true, avatar_scale: 1.3, ..Default::default() })]);
    let avatar = obj(&mut scene, Some(net), vec![Box::new(HumanoidIk::default())]);
    scene.add_test_skeleton(avatar, humanoid());
    let mut body = BodyTracking::default();
    body.update(&mut scene, &TrackerTargets::default(), frame(false));
    assert!((scene.objects[avatar].scale.x - 1.3).abs() < 1e-5);
    assert_eq!(scene.objects[net].scale, Vec3::ONE);
    assert!(body.avatar.is_none(), "a remote avatar is not the local one");
}

#[test]
fn arm_span_matching_sets_the_hand_reach() {
    let (mut scene, rig, _) = rig_scene(AvatarScaling::ScaleAvatar, false);
    if let Some(p) = scene.objects[rig].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<VrPlayer>()) {
        p.match_arm_span = true;
    }
    let mut body = BodyTracking::default();
    body.profile.measured_eye_height = Some(AVATAR_EYE);
    body.profile.arm_span = Some(1.69);
    body.update(&mut scene, &head_at(1.7), frame(true));
    // The fixture's wrist-to-wrist span is 1.3 m: its arms reach less far.
    assert!((scene.local_body.arm_reach - 1.3 / 1.69).abs() < 1e-3, "{}", scene.local_body.arm_reach);
}

#[test]
fn eleven_point_tracking_preserves_hmd_hands_and_all_body_roles() {
    let mut body = BodyTracking::default();
    let targets = TrackerTargets::from_slots(std::array::from_fn(|slot| Some((Vec3::new(slot as f32 * 0.02, 1.0 + slot as f32 * 0.03, 0.0), Quat::from_rotation_y(slot as f32 * 0.02)))));
    let devices = (3..TRACKER_SLOTS).map(|slot| device(&format!("tracker{slot}"), slot, targets.slots()[slot])).collect();
    let assembled = body.assemble(targets.head, targets.left_hand, targets.right_hand, devices);
    assert_eq!(assembled, targets);
    assert_eq!(assembled.slots().iter().flatten().count(), 11);
    body.capture_tpose(&assembled, false).unwrap();
    assert_eq!(body.profile.tpose_targets().unwrap(), assembled);
    let skel = humanoid();
    let rig = humanoid::HumanoidRig::map(&skel);
    let raw = TrackerPoses::from_targets(&assembled);
    let cal = humanoid::calibrate_tpose(&skel, &rig, &raw, 1.0);
    let solved = humanoid::targets_from_calibrated(&cal, &raw);
    assert_eq!(solved.slots().iter().flatten().count(), 11, "elbows, knees and chest reach the solver too");
    assert!(solved.slots().into_iter().flatten().all(|(p, q)| p.is_finite() && q.is_finite()));
    // Missing trackers are optional, and an external device cannot overwrite the headset.
    let raw = body.assemble(targets.head, targets.left_hand, targets.right_hand,
        vec![device("bad-role", 0, pose(99.0, 99.0, 99.0)), device("lost", 3, None)]);
    assert_eq!(raw.head, targets.head);
    assert!(raw.hips.is_none());
}

#[test]
fn avatar_alignment_requires_release_then_both_triggers_and_cleans_up() {
    let (mut scene, _, avatar) = rig_scene(AvatarScaling::Off, true);
    let mut body = BodyTracking::default();
    body.profile.auto_assign = false;
    let raw = TrackerTargets { head: pose(0.0, 1.7, 0.0), left_hand: pose(0.7, 1.45, 0.0), right_hand: pose(-0.7, 1.45, 0.0), ..Default::default() };
    body.command(BodyCommand::AlignAvatar);
    let mut input = frame(true);
    input.triggers = [true; 2];
    body.update(&mut scene, &raw, input);
    assert!(!body.calibrated(), "opening with held triggers never accepts");
    assert_eq!(scene.calibration_pose.as_ref().unwrap().0, scene.objects[avatar].id);
    input.triggers = [false; 2]; body.update(&mut scene, &raw, input);
    input.triggers = [true, false]; body.update(&mut scene, &raw, input);
    assert!(!body.calibrated(), "one trigger does not accept");
    input.triggers = [true; 2]; body.update(&mut scene, &raw, input);
    assert!(body.calibrated() && !body.info().calibrating);
    assert!(scene.calibration_pose.is_none());
    let saved = body.profile.to_ron();
    body.command(BodyCommand::AlignAvatar);
    body.update(&mut scene, &raw, frame(true));
    body.cancel(); body.update(&mut scene, &raw, frame(true));
    assert_eq!(body.profile.to_ron(), saved, "cancel preserves the accepted profile");
    assert!(scene.calibration_pose.is_none());
}

#[test]
fn calibration_follows_the_selected_avatar_and_rebuilds_its_offsets() {
    let (mut scene, player, first) = rig_scene(AvatarScaling::Off, true);
    let second = obj(&mut scene, Some(player), vec![Box::new(HumanoidIk::default())]);
    let mut skeleton = humanoid();
    for joint in &mut skeleton.joints { joint.rest_translation *= 1.3; }
    scene.add_test_skeleton(second, skeleton);
    scene.objects[second].enabled = false;
    let mut body = BodyTracking::default();
    let raw = head_at(1.7);
    body.command(BodyCommand::AlignAvatar);
    body.update(&mut scene, &raw, frame(true));
    assert_eq!(body.solved.unwrap().avatar.0, scene.objects[first].id);
    assert_eq!(scene.calibration_pose.as_ref().unwrap().0, scene.objects[first].id);
    scene.objects[first].enabled = false;
    scene.objects[second].enabled = true;
    body.update(&mut scene, &raw, frame(true));
    assert_eq!(body.solved.unwrap().avatar.0, scene.objects[second].id);
    assert_eq!(scene.calibration_pose.as_ref().unwrap().0, scene.objects[second].id);
    assert!(scene.objects[second].components.iter().any(|c| c.as_any().is::<zestengine_core::AvatarHands>()));
}

#[test]
fn avatar_alignment_rejects_missing_hands_and_a_missing_worn_avatar() {
    let (mut scene, _, _) = rig_scene(AvatarScaling::Off, true);
    let mut body = BodyTracking::default();
    body.command(BodyCommand::AlignAvatar);
    let raw = head_at(1.7);
    body.update(&mut scene, &raw, frame(true));
    body.update(&mut scene, &raw, BodyFrame { triggers: [true; 2], ..frame(true) });
    assert!(!body.calibrated() && body.info().calibrating);
    assert!(body.status.as_deref().unwrap().contains("both hands"));
    body.cancel();
    let mut empty = LoadedScene::empty();
    body.command(BodyCommand::AlignAvatar);
    body.update(&mut empty, &raw, frame(true));
    body.update(&mut empty, &raw, BodyFrame { triggers: [true; 2], ..frame(true) });
    assert!(!body.calibrated());
    assert!(body.status.as_deref().unwrap().contains("humanoid avatar"));
}
