//! CPU regressions for native UI, menu, avatar and portal behaviour.
use super::{ContactPhase, DisabledComponent, Enabled, NativeTestFixture, SceneEvent};
use crate::*;
use glam::{Quat, Vec2, Vec3};

fn frame(fixture: &mut NativeTestFixture, screen: &ScreenInfo) -> Vec<ComponentCommand> {
    fixture.frame_dt(&InputState::default(), &NetView::default(), screen, 0.0)
}

#[test]
fn native_ui_buttons_resolve_first_named_object_and_skip_disabled_instances() {
    let mut fixture = NativeTestFixture::new();
    let first = fixture.add("Panel", vec![]);
    let second = fixture.add("Panel", vec![]);
    fixture.scene.get_mut::<Enabled>(first).unwrap().0 = false;
    fixture.scene.get_mut::<Enabled>(second).unwrap().0 = false;
    let button = fixture.add("Button", vec![
        Box::new(UiButton { on_click: vec![UiAction::Show("Panel".into()), UiAction::LoadScene(String::new())], ..Default::default() }),
        Box::new(Disabled(Box::new(UiButton { on_click: vec![UiAction::Quit], ..Default::default() }))),
        Box::new(UiButton { on_click: vec![UiAction::LoadScene("next.scene".into()), UiAction::Quit], ..Default::default() }),
    ]);
    let screen = ScreenInfo { ui_clicks: vec![fixture.id(button)], ..Default::default() };
    let commands = frame(&mut fixture, &screen);
    assert!(matches!(commands.as_slice(), [ComponentCommand::SetActive(id, true), ComponentCommand::LoadScene(path), ComponentCommand::Quit] if *id == fixture.id(first) && path == "next.scene"));
    fixture.apply_output(commands).unwrap();
    assert!(fixture.scene.get::<Enabled>(first).unwrap().0);
    assert!(!fixture.scene.get::<Enabled>(second).unwrap().0);
    fixture.scene.get_mut::<Enabled>(button).unwrap().0 = false;
    assert!(frame(&mut fixture, &screen).is_empty(), "inactive owners do not receive clicks");
}

#[test]
fn native_pause_menu_updates_while_time_is_stopped_and_retains_transition_state() {
    let mut fixture = NativeTestFixture::new();
    let panel = fixture.add("Pause Panel", vec![]);
    fixture.scene.get_mut::<Enabled>(panel).unwrap().0 = false;
    let menu = fixture.add("Canvas", vec![Box::new(UiPauseMenu { lock_cursor_in_game: true, ..Default::default() })]);
    let mut screen = ScreenInfo::default();
    assert!(frame(&mut fixture, &screen).is_empty());
    screen.ui_cancel = true;
    let commands = frame(&mut fixture, &screen);
    assert!(matches!(commands.as_slice(), [ComponentCommand::SetActive(id, true)] if *id == fixture.id(panel)));
    fixture.apply_output(commands).unwrap();
    screen.ui_cancel = false;
    assert!(matches!(frame(&mut fixture, &screen).as_slice(), [ComponentCommand::SetTimeScale(0.0), ComponentCommand::SetCursorLocked(false)]));
    assert!(frame(&mut fixture, &screen).is_empty(), "idle paused frames do not reapply commands");
    screen.ui_cancel = true;
    let commands = frame(&mut fixture, &screen);
    fixture.apply_output(commands).unwrap();
    screen.ui_cancel = false;
    assert!(matches!(frame(&mut fixture, &screen).as_slice(), [ComponentCommand::SetTimeScale(1.0), ComponentCommand::SetCursorLocked(true)]));
    let saved = fixture.scene.get::<UiPauseMenu>(menu).unwrap().save();
    let loaded: UiPauseMenu = ron::from_str(&saved).unwrap();
    assert_eq!(loaded.open, None, "runtime transitions do not leak into saved scenes");
}

#[test]
fn native_toggle_restores_player_setting_then_tunes_only_matching_schemes() {
    let mut fixture = NativeTestFixture::new();
    let toggle = fixture.add("Invert Look", vec![Box::new(UiToggle { invert_y_action: "Look".into(), ..Default::default() })]);
    let look = || ActionBinding { invert_y: true, ..ActionBinding::empty("Look", ActionKind::Axis2) };
    let bindings = Bindings::from_schemes(vec![
        ControlScheme { name: "Keyboard".into(), actions: vec![look()] },
        ControlScheme { name: "Gamepad".into(), actions: vec![look()] },
        ControlScheme { name: "Other".into(), actions: vec![] },
    ]);
    let mut runtime = ControlsRuntime::new(bindings);
    let mut screen = ScreenInfo { controls: runtime.info(), ..Default::default() };
    assert!(frame(&mut fixture, &screen).is_empty());
    assert!(fixture.scene.get::<UiToggle>(toggle).unwrap().on);
    fixture.scene.get_mut::<UiToggle>(toggle).unwrap().on = false;
    let commands = frame(&mut fixture, &screen);
    assert_eq!(commands.len(), 2);
    for command in commands {
        let ComponentCommand::Input(InputCommand::Tune { ref action, tuning, .. }) = command else { panic!("unexpected command") };
        assert_eq!(action, "Look");
        assert_eq!(tuning.invert_y, Some(false));
        if let ComponentCommand::Input(command) = command { runtime.apply(command).unwrap(); }
    }
    assert!(runtime.bindings().schemes.iter().filter_map(|scheme| scheme.action("Look")).all(|action| !action.invert_y));
    screen.controls = runtime.info();
    assert!(frame(&mut fixture, &screen).is_empty());
}

#[test]
fn native_sliders_apply_fraction_once_and_honor_inherited_activation() {
    let mut fixture = NativeTestFixture::new();
    let parent = fixture.add("Parent", vec![]);
    let slider = fixture.add("Volume", vec![Box::new(UiSlider { min: 10.0, max: 30.0, value: 15.0, audio_bus: "Music".into(), ..Default::default() })]);
    fixture.scene.set_parent(slider, Some(parent), false).unwrap();
    fixture.scene.get_mut::<Enabled>(parent).unwrap().0 = false;
    let screen = ScreenInfo::default();
    assert!(frame(&mut fixture, &screen).is_empty());
    fixture.scene.get_mut::<Enabled>(parent).unwrap().0 = true;
    assert!(matches!(frame(&mut fixture, &screen).as_slice(), [ComponentCommand::SetBusVolume { bus, volume }] if bus == "Music" && *volume == 0.25));
    assert!(frame(&mut fixture, &screen).is_empty());
    let value = fixture.scene.remove::<UiSlider>(slider).unwrap();
    fixture.scene.insert_disabled(slider, value).unwrap();
    fixture.scene.get_mut::<DisabledComponent<UiSlider>>(slider).unwrap().value_mut().value = 30.0;
    assert!(frame(&mut fixture, &screen).is_empty());
    assert_eq!(fixture.scene.get::<DisabledComponent<UiSlider>>(slider).unwrap().applied, Some(15.0));
}

fn button_text(fixture: &mut NativeTestFixture, name: &str) -> super::Entity {
    fixture.add(name, vec![Box::new(UiButton::default()), Box::new(UiText::default())])
}

#[test]
fn native_controls_menu_updates_replaced_ui_data_without_cached_mirrors() {
    let mut fixture = NativeTestFixture::new();
    fixture.add("Menu", vec![Box::new(ControlsMenu { rows: 1, columns: 1, ..Default::default() })]);
    let page = fixture.add("Controls Page", vec![Box::new(UiText::default())]);
    let hidden = button_text(&mut fixture, "Controls Row 1 Bind 2");
    let screen = ScreenInfo::default();
    let commands = frame(&mut fixture, &screen);
    fixture.apply_output(commands).unwrap();
    let page_text = fixture.scene.get::<UiText>(page).unwrap().text.clone();
    let text_pointer = fixture.scene.get::<UiText>(page).unwrap().text.as_ptr();
    assert!(!page_text.is_empty());
    assert!(!fixture.scene.get::<Enabled>(hidden).unwrap().0);
    assert!(frame(&mut fixture, &screen).is_empty());
    assert_eq!(fixture.scene.get::<UiText>(page).unwrap().text.as_ptr(), text_pointer, "unchanged text keeps its allocation");
    fixture.scene.insert(page, UiText::default()).unwrap();
    fixture.scene.get_mut::<Enabled>(hidden).unwrap().0 = true;
    let commands = frame(&mut fixture, &screen);
    fixture.apply_output(commands).unwrap();
    assert_eq!(fixture.scene.get::<UiText>(page).unwrap().text, page_text);
    assert!(!fixture.scene.get::<Enabled>(hidden).unwrap().0, "replacement UI is reconciled from current ECS data");
}

#[test]
fn native_controls_capture_is_scoped_to_owner_and_drives_rebinding_commands() {
    let mut fixture = NativeTestFixture::new();
    let menu = fixture.add("Menu", vec![Box::new(ControlsMenu { rows: 1, columns: 1, ..Default::default() })]);
    let bind_button = button_text(&mut fixture, "Controls Row 1 Bind 1");
    let reset = button_text(&mut fixture, "Controls Reset");
    let mut runtime = ControlsRuntime::new(Bindings::from_schemes(vec![ControlScheme {
        name: "Keyboard".into(), actions: vec![ActionBinding { buttons: vec![InputSource::Key(Key::Space)], ..ActionBinding::empty("Jump", ActionKind::Button) }],
    }]));
    let mut screen = ScreenInfo { ui_clicks: vec![fixture.id(bind_button)], controls: runtime.info(), ..Default::default() };
    let commands = frame(&mut fixture, &screen);
    assert!(commands.iter().any(|command| matches!(command, ComponentCommand::Input(InputCommand::Capture { owner, .. }) if *owner == fixture.id(menu))));
    for command in commands { if let ComponentCommand::Input(command) = command { runtime.apply(command).unwrap(); } }
    assert!(!fixture.scene.get::<UiButton>(reset).unwrap().interactable);
    screen.ui_clicks.clear();
    screen.controls = runtime.info();
    screen.controls.captured = Some((fixture.id(reset), CaptureResult::Captured(InputSource::Key(Key::G))));
    assert!(frame(&mut fixture, &screen).iter().all(|command| !matches!(command, ComponentCommand::Input(InputCommand::Rebind { .. }))));
    screen.controls.captured = Some((fixture.id(menu), CaptureResult::Captured(InputSource::Key(Key::G))));
    let commands = frame(&mut fixture, &screen);
    assert!(fixture.scene.get::<UiButton>(reset).unwrap().interactable);
    for command in commands { if let ComponentCommand::Input(command) = command { runtime.apply(command).unwrap(); } }
    assert_eq!(runtime.bindings().schemes[0].actions[0].buttons, vec![InputSource::Key(Key::G)]);
    assert_eq!(fixture.scene.get::<UiText>(bind_button).unwrap().text, "Space", "captured binding is presented from the next host input frame");
    screen.controls = runtime.info();
    frame(&mut fixture, &screen);
    assert_eq!(fixture.scene.get::<UiText>(bind_button).unwrap().text, "G");
}

#[test]
fn native_avatar_following_lends_same_type_instances_and_preserves_authoring_order() {
    let mut fixture = NativeTestFixture::new();
    let rig = fixture.add("VR rig", vec![Box::new(VrCharacter::default())]);
    let player = fixture.add("Player", vec![Box::new(NetworkVariables::default())]);
    let before = fixture.add("Earlier follower", vec![Box::new(TemplateAvatar { follow_local_player: true, player: fixture.id(player).into(), ..Default::default() })]);
    let source = fixture.add("Source", vec![Box::new(TemplateAvatar { player: fixture.id(rig).into(), ..Default::default() })]);
    let after = fixture.add("Later follower", vec![Box::new(TemplateAvatar { follow_local_player: true, ..Default::default() })]);
    let input = InputState::from_actions([("Switch Avatar".into(), ActionValue { just_pressed: true, ..Default::default() })], Vec2::ZERO);
    fixture.frame(&input, &NetView::default(), &ScreenInfo::default());
    assert_eq!(fixture.scene.get::<TemplateAvatar>(before).unwrap().variant, AvatarVariant::Male);
    assert_eq!(fixture.scene.get::<TemplateAvatar>(source).unwrap().variant, AvatarVariant::Female);
    assert_eq!(fixture.scene.get::<TemplateAvatar>(after).unwrap().variant, AvatarVariant::Female);
    let commands = frame(&mut fixture, &ScreenInfo::default());
    assert_eq!(fixture.scene.get::<TemplateAvatar>(before).unwrap().variant, AvatarVariant::Female);
    assert_eq!(fixture.scene.get::<NetworkVariables>(player).unwrap().get("zestengine.avatar_female"), Some(&NetValue::Bool(true)));
    assert!(commands.iter().any(|command| matches!(command, ComponentCommand::NetSetVar { id, value: NetValue::Bool(true), .. } if *id == fixture.id(player))));
    fixture.scene.get_mut::<Enabled>(source).unwrap().0 = false;
    fixture.scene.get_mut::<TemplateAvatar>(source).unwrap().variant = AvatarVariant::Male;
    frame(&mut fixture, &ScreenInfo::default());
    assert_eq!(fixture.scene.get::<TemplateAvatar>(after).unwrap().variant, AvatarVariant::Male, "inactive local sources remain readable");
}

#[test]
fn native_remote_avatar_uses_parented_world_pose_and_updates_animators_directly() {
    let mut fixture = NativeTestFixture::new();
    let parent = fixture.add("Parent", vec![]);
    fixture.scene.get_mut::<Transform>(parent).unwrap().translation = Vec3::X * 10.0;
    fixture.scene.get_mut::<Transform>(parent).unwrap().rotation = Quat::from_rotation_y(0.5);
    let mut variables = NetworkVariables::default();
    variables.set("zestengine.avatar_female", NetValue::Bool(true));
    let player = fixture.add("Remote", vec![Box::new(NetworkAvatar { remote: true, ..Default::default() }), Box::new(variables)]);
    fixture.scene.set_parent(player, Some(parent), false).unwrap();
    let male = fixture.add("Male", vec![Box::new(Animator::default())]);
    let female = fixture.add("Female", vec![Box::new(Animator::default())]);
    let avatar = fixture.add("Selector", vec![Box::new(TemplateAvatar {
        player: fixture.id(player).into(), male: fixture.id(male).into(), female: fixture.id(female).into(), ..Default::default()
    })]);
    let commands = fixture.frame(&InputState::default(), &NetView::default(), &ScreenInfo::default());
    fixture.apply_output(commands).unwrap();
    assert!(!fixture.scene.get::<Enabled>(male).unwrap().0 && fixture.scene.get::<Enabled>(female).unwrap().0);
    assert_eq!(fixture.scene.get::<TemplateAvatar>(avatar).unwrap().previous.unwrap().0, Vec3::X * 10.0);
    fixture.scene.get_mut::<Transform>(player).unwrap().translation = Vec3::NEG_Z;
    let commands = fixture.frame(&InputState::default(), &NetView::default(), &ScreenInfo::default());
    assert!(commands.is_empty(), "animation writes are native data edits and idle variants do not enqueue commands");
    for entity in [male, female] {
        let animator = fixture.scene.get::<Animator>(entity).unwrap();
        assert!(animator.get_float("Forward") > 6.0);
        assert!(animator.get_float("Right").abs() < 1e-4);
        assert!(animator.get_bool("Grounded"));
    }
}

fn contact(object: ObjectId, other: ObjectId, two_dimensional: bool) -> SceneEvent {
    SceneEvent::Contact { object, other, trigger: true, phase: ContactPhase::Enter, two_dimensional }
}

#[test]
fn native_portal_preload_and_events_use_world_poses_and_stable_activator_ids() {
    let mut fixture = NativeTestFixture::new();
    let parent = fixture.add("Parent", vec![]);
    fixture.scene.get_mut::<Transform>(parent).unwrap().translation = Vec3::X * 100.0;
    let ignored = fixture.add("Inactive pawn", vec![Box::new(Pawn::default())]);
    fixture.scene.get_mut::<Enabled>(ignored).unwrap().0 = false;
    let player = fixture.add("Player", vec![Box::new(Pawn::default())]);
    fixture.scene.get_mut::<Transform>(player).unwrap().translation = Vec3::X * 103.0;
    let portal = fixture.add("Portal", vec![Box::new(ScenePortal { scene: "next.scene".into(), preload_distance: 4.0, ..Default::default() })]);
    fixture.scene.set_parent(portal, Some(parent), false).unwrap();
    assert!(matches!(frame(&mut fixture, &ScreenInfo::default()).as_slice(), [ComponentCommand::PreloadScene(path)] if path == "next.scene"));
    assert!(frame(&mut fixture, &ScreenInfo::default()).is_empty());
    assert!(fixture.events(&[contact(fixture.id(portal), fixture.id(ignored), false)]).is_empty());
    let commands = fixture.events(&[contact(fixture.id(portal), fixture.id(player), true), contact(fixture.id(portal), fixture.id(player), false)]);
    assert!(matches!(commands.as_slice(), [ComponentCommand::KeepOnSceneChange(id, true), ComponentCommand::LoadScene(path)] if *id == fixture.id(player) && path == "next.scene"));
    assert!(fixture.events(&[contact(fixture.id(portal), fixture.id(player), false)]).is_empty(), "one-shot portals retain native runtime state");
}

#[test]
fn native_portal_streaming_modes_filter_contacts_and_preserve_explicit_inactive_targets() {
    let mut fixture = NativeTestFixture::new();
    let player = fixture.add("Explicit target", vec![]);
    fixture.scene.get_mut::<Enabled>(player).unwrap().0 = false;
    let portal = fixture.add("Portal", vec![Box::new(ScenePortal { scene: "stream.scene".into(), mode: PortalMode::Unload, activator: fixture.id(player).into(), one_shot: false, ..Default::default() })]);
    assert!(frame(&mut fixture, &ScreenInfo::default()).is_empty(), "unload mode never preloads");
    let mut collision = contact(fixture.id(portal), fixture.id(player), false);
    if let SceneEvent::Contact { trigger, .. } = &mut collision { *trigger = false; }
    assert!(fixture.events(&[collision]).is_empty());
    let enter = contact(fixture.id(portal), fixture.id(player), false);
    for _ in 0..2 {
        assert!(matches!(fixture.events(std::slice::from_ref(&enter)).as_slice(), [ComponentCommand::UnloadScene(path)] if path == "stream.scene"));
    }
    fixture.scene.get_mut::<ScenePortal>(portal).unwrap().mode = PortalMode::Additive;
    assert!(matches!(fixture.events(std::slice::from_ref(&enter)).as_slice(), [ComponentCommand::LoadSceneAdditive(path)] if path == "stream.scene"));
    fixture.scene.get_mut::<Enabled>(portal).unwrap().0 = false;
    assert!(fixture.events(&[enter]).is_empty());
}


#[test]
fn native_avatar_source_selection_respects_first_instance_and_excludes_executing_slot() {
    let mut fixture = NativeTestFixture::new();
    let rig = fixture.add("VR rig", vec![Box::new(VrCharacter::default())]);
    let follower = fixture.add("Follower", vec![Box::new(TemplateAvatar { follow_local_player: true, ..Default::default() })]);
    let source = fixture.add("Duplicate selectors", vec![
        Box::new(TemplateAvatar { follow_local_player: true, ..Default::default() }),
        Box::new(TemplateAvatar { player: fixture.id(rig).into(), variant: AvatarVariant::Female, ..Default::default() }),
    ]);
    for _ in 0..2 { frame(&mut fixture, &ScreenInfo::default()); }
    assert_eq!(fixture.scene.get::<TemplateAvatar>(source).unwrap().variant, AvatarVariant::Female, "the executing instance can read its next enabled sibling");
    assert_eq!(fixture.scene.get::<TemplateAvatar>(follower).unwrap().variant, AvatarVariant::Male, "other objects select the first enabled instance, which is a follower");
}

#[test]
fn native_network_avatar_follows_parented_rig_and_rescans_removed_sources() {
    let mut fixture = NativeTestFixture::new();
    let source_head = fixture.add("Source head", vec![]);
    let rig = fixture.add("VR rig", vec![Box::new(VrCharacter { head: fixture.id(source_head).into(), ..Default::default() })]);
    fixture.scene.get_mut::<Transform>(rig).unwrap().translation = Vec3::new(10.0, 1.0, 2.0);
    fixture.scene.get_mut::<Transform>(rig).unwrap().rotation = Quat::from_rotation_y(0.8);
    fixture.scene.set_parent(source_head, Some(rig), false).unwrap();
    fixture.scene.get_mut::<Transform>(source_head).unwrap().translation = Vec3::Y * 1.7;
    let parent = fixture.add("Avatar parent", vec![]);
    fixture.scene.get_mut::<Transform>(parent).unwrap().translation = Vec3::X * 5.0;
    fixture.scene.get_mut::<Transform>(parent).unwrap().rotation = Quat::from_rotation_y(-0.4);
    let target_head = fixture.add("Target head", vec![]);
    let avatar = fixture.add("Avatar", vec![Box::new(NetworkAvatar {
        follow_local_player: true, head: fixture.id(target_head).into(), owner_hidden: vec![fixture.id(target_head).into()], ..Default::default()
    })]);
    fixture.scene.set_parent(avatar, Some(parent), false).unwrap();
    fixture.scene.set_parent(target_head, Some(avatar), false).unwrap();
    let commands = frame(&mut fixture, &ScreenInfo::default());
    assert!(matches!(commands.as_slice(), [ComponentCommand::SetActive(id, false)] if *id == fixture.id(target_head)));
    fixture.scene.with_world(|world| {
        assert!(world.world_transform(avatar).unwrap().abs_diff_eq(world.world_transform(rig).unwrap(), 1e-4));
        assert!(world.world_transform(target_head).unwrap().abs_diff_eq(world.world_transform(source_head).unwrap(), 1e-4));
    });
    let previous_id = fixture.id(rig);
    fixture.scene.despawn_recursive(rig).unwrap();
    let replacement = fixture.add("Replacement rig", vec![Box::new(VrCharacter::default())]);
    fixture.scene.get_mut::<Transform>(replacement).unwrap().translation = Vec3::Z * 9.0;
    assert!(frame(&mut fixture, &ScreenInfo::default()).is_empty(), "owner visibility is applied once");
    let value = fixture.scene.get::<NetworkAvatar>(avatar).unwrap();
    assert_ne!(value.player, Some(previous_id));
    assert_eq!(value.player, Some(fixture.id(replacement)));
    drop(value);
    fixture.scene.with_world(|world| assert!(world.world_transform(avatar).unwrap().w_axis.truncate().abs_diff_eq(Vec3::Z * 9.0, 1e-4)));
}

#[test]
fn native_network_avatar_tracking_requires_authority_and_active_owner() {
    let mut fixture = NativeTestFixture::new();
    let head = fixture.add("Head", vec![]);
    let avatar = fixture.add("Avatar", vec![Box::new(NetworkAvatar { head: fixture.id(head).into(), ..Default::default() })]);
    let id = fixture.id(avatar);
    let mut net = NetView { connected: true, local_peer: 1, ..Default::default() };
    net.owners.insert(id, 2);
    let mut screen = ScreenInfo::default();
    screen.vr.active = true;
    screen.vr.head = Some((Vec3::Y * 1.6, Quat::from_rotation_y(0.3)));
    assert!(fixture.frame(&InputState::default(), &net, &screen).is_empty());
    assert_eq!(fixture.scene.get::<Transform>(head).unwrap().translation, Vec3::ZERO);
    net.owners.insert(id, 1);
    assert!(fixture.frame(&InputState::default(), &net, &screen).is_empty());
    assert_eq!(fixture.scene.get::<Transform>(head).unwrap().translation, Vec3::Y * 1.6);
    fixture.scene.get_mut::<Enabled>(avatar).unwrap().0 = false;
    screen.vr.head = Some((Vec3::Y * 2.0, Quat::IDENTITY));
    fixture.frame(&InputState::default(), &net, &screen);
    assert_eq!(fixture.scene.get::<Transform>(head).unwrap().translation, Vec3::Y * 1.6);
}

#[test]
fn native_hand_poses_use_local_inputs_or_remote_variables_and_keep_typed_state() {
    let mut fixture = NativeTestFixture::new();
    let player = fixture.add("Player", vec![Box::new(NetworkVariables::default())]);
    let hands = fixture.add("Hands", vec![Box::new(AvatarHands { player: fixture.id(player).into(), ..Default::default() })]);
    let input = InputState::from_actions([("Left Hand Peace".into(), ActionValue { down: true, ..Default::default() })], Vec2::ZERO);
    let mut screen = ScreenInfo::default();
    screen.vr.active = true;
    screen.vr.left_trigger = true;
    screen.vr.right_grip = 1.0;
    let commands = fixture.frame(&input, &NetView::default(), &screen);
    assert_eq!(commands.len(), 2, "hand variables forward their changes for replication");
    assert_eq!(fixture.scene.get::<AvatarHands>(hands).unwrap().selected, [HandPose::Peace, HandPose::Point]);
    assert_eq!(fixture.scene.get::<NetworkVariables>(player).unwrap().get("zestengine.hand_left"), Some(&NetValue::Int(HandPose::Peace as i64)));
    assert!(fixture.frame(&input, &NetView::default(), &screen).is_empty());
    fixture.scene.insert(player, NetworkAvatar { remote: true, ..Default::default() }).unwrap();
    {
        let mut variables = fixture.scene.get_mut::<NetworkVariables>(player).unwrap();
        variables.set("zestengine.hand_left", NetValue::Int(999));
        variables.set("zestengine.hand_right", NetValue::Int(HandPose::Fist as i64));
    }
    assert!(fixture.frame(&input, &NetView::default(), &screen).is_empty());
    assert_eq!(fixture.scene.get::<AvatarHands>(hands).unwrap().selected, [HandPose::Idle, HandPose::Fist]);
    assert!(fixture.scene.get::<AvatarHands>(hands).unwrap().current.is_some());
}
