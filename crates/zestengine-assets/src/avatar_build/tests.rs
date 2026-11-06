use std::path::PathBuf;

use super::*;
use crate::{MaterialRef, ObjectSource};
use zestengine_core::{AvatarMenu, MenuInstaller, ObjectToggleEntry, SkinRebound, split_menu_path};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zest-avatar-build-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn entry(name: &str, parent: Option<usize>, components: Vec<ComponentData>) -> SceneEntry {
    SceneEntry {
        id: format!("id-{name}"),
        name: name.into(),
        source: ObjectSource::Empty,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: Default::default(),
        layer: 0,
        material: MaterialRef::Model,
        extra_materials: Vec::new(),
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

fn comp<T: TypedComponent>(v: &T) -> ComponentData {
    ComponentData::new(T::NAME, ron::to_string(v).unwrap())
}

fn env(root: &Path) -> BuildEnv<'_> {
    BuildEnv { project_root: root, write: false, structural: true, budget_cap: 256 }
}

fn hat_toggle(path: &str, target: &str) -> MenuToggle {
    MenuToggle { menu_path: path.into(), actions: vec![ToggleAction::Active { target: target.into(), on: true }], ..Default::default() }
}

/// Avatar root with a descriptor and an Animator, plus the given children.
fn avatar(children: Vec<SceneEntry>) -> SceneFile {
    let mut entries = vec![entry("Avatar", None, vec![comp(&AvatarDescriptor::default()), comp(&Animator::default())])];
    for mut c in children {
        c.parent = c.parent.map(|p| p + 1).or(Some(0));
        entries.push(c);
    }
    SceneFile { entries, ..Default::default() }
}

fn load_ctrl(b: &AvatarBuild) -> AnimatorController {
    let bytes = b.files.get(&format!("{}/controller.animctrl", b.out_dir)).expect("controller written");
    ron::from_str(std::str::from_utf8(bytes).unwrap()).unwrap()
}

fn load_clip(b: &AvatarBuild, rel: &str) -> AnimClip {
    ron::from_str(std::str::from_utf8(b.files.get(rel).unwrap_or_else(|| panic!("{rel} missing in {:?}", b.files.keys()))).unwrap()).unwrap()
}

fn layer<'a>(c: &'a AnimatorController, name: &str) -> &'a AnimLayer {
    c.layers.iter().find(|l| l.name == name).unwrap_or_else(|| panic!("no layer {name} in {:?}", c.layers.iter().map(|l| &l.name).collect::<Vec<_>>()))
}

#[test]
fn a_menu_toggle_becomes_parameter_menu_entry_clip_and_layer() {
    let dir = temp_dir("toggle");
    let scene = avatar(vec![entry("Hat", None, vec![comp(&hat_toggle("Outfit/Hat", "Hat"))])]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);

    // Parameter: a Bool, synced and saved, off by default.
    assert_eq!(b.generated_params.len(), 1);
    let name = b.generated_params[0].clone();
    assert!(name.starts_with("Tgl_Hat_"), "{name}");
    let p = b.params.iter().find(|p| p.name == name).unwrap();
    assert_eq!((p.kind, p.saved, p.is_synced(), p.default), (AvatarParamKind::Bool, true, true, 0.0));
    assert_eq!(b.budget.unwrap().used, 1);

    // Menu entry under "Outfit".
    let outfit = &b.menu.root[0];
    assert_eq!((outfit.item.label.as_str(), outfit.item.kind), ("Outfit", MenuItemKind::SubMenu));
    let hat = &outfit.children[0].item;
    assert_eq!((hat.label.as_str(), hat.kind, hat.parameter.as_str()), ("Hat", MenuItemKind::Toggle, name.as_str()));

    // Controller: Defaults at index 0, then the toggle's layer.
    let c = load_ctrl(&b);
    assert_eq!(c.layers[0].name, "Defaults");
    let l = layer(&c, "[gen] Outfit/Hat");
    assert_eq!(l.machine.states.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["Off", "On"]);
    assert_eq!(l.machine.default_state, "Off");
    let t = &l.machine.transitions;
    assert_eq!((t[0].from.as_deref(), t[0].to.as_str(), t[0].conditions[0].op), (Some("Off"), "On", CondOp::If));
    assert_eq!((t[1].from.as_deref(), t[1].to.as_str(), t[1].conditions[0].op), (Some("On"), "Off", CondOp::IfNot));
    assert!(t.iter().all(|t| t.conditions[0].parameter == name && !t.has_exit_time));
    assert!(l.machine.states.iter().all(|s| s.write_defaults), "states name only what they animate; the Defaults layer owns resting values");

    // The on clip turns the hat on; the defaults clip keeps it off.
    let Motion::Clip(on_clip) = &l.machine.states[1].motion else { panic!("On plays a clip") };
    assert!(on_clip.starts_with(&format!("{}/toggles/", b.out_dir)));
    let clip = load_clip(&b, on_clip);
    assert_eq!(clip.tracks.len(), 1);
    assert_eq!((clip.tracks[0].target.as_str(), clip.tracks[0].property, clip.tracks[0].keys[0].value[0]), ("Hat", TrackProperty::Active, 1.0));
    let Motion::Clip(def) = &c.layers[0].machine.states[0].motion else { panic!() };
    let defaults = load_clip(&b, def);
    assert_eq!((defaults.tracks[0].target.as_str(), defaults.tracks[0].keys[0].value[0]), ("Hat", 0.0));

    // The clone starts with the hat off and carries the finished components.
    assert!(!built.entries[1].enabled);
    assert!(scene.entries[1].enabled, "the authored scene is untouched");
    assert!(built.entries[1].components.iter().all(|c| c.kind != MenuToggle::NAME), "authoring components are stripped");
    let anim: Animator = comps(&built.entries[0]).remove(0);
    assert_eq!(anim.controller, format!("{}/controller.animctrl", b.out_dir));
    let ap: AvatarParameters = comps(&built.entries[0]).remove(0);
    assert_eq!(ap.params.len(), 1);
    let md: AvatarMenuData = comps(&built.entries[0]).remove(0);
    assert_eq!(md.menu, b.menu);
}

#[test]
fn default_on_toggles_start_on_and_slider_toggles_scrub_their_clip() {
    let dir = temp_dir("modes");
    let mut on = hat_toggle("Hat", "Hat");
    on.default_on = true;
    let slider = MenuToggle {
        menu_path: "Smile".into(),
        mode: ToggleMode::Slider { default: 0.0, inactive_at_zero: true },
        actions: vec![ToggleAction::BlendShape { target: "Body".into(), shape: "Smile".into(), value: 1.0 }],
        ..Default::default()
    };
    let scene = avatar(vec![entry("Hat", None, vec![comp(&on)]), entry("Body", None, vec![comp(&slider)])]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    assert!(built.entries[1].enabled, "default on");
    let c = load_ctrl(&b);
    assert_eq!(layer(&c, "[gen] Hat").machine.default_state, "On");
    let hat = b.params.iter().find(|p| p.name.starts_with("Tgl_Hat")).unwrap();
    assert_eq!(hat.default, 1.0);

    // Slider: a Float (8 bits), a radial puppet, motion time = the parameter.
    let sl = b.params.iter().find(|p| p.name.starts_with("Tgl_Smile")).unwrap();
    assert_eq!(sl.kind, AvatarParamKind::Float);
    assert_eq!(b.budget.unwrap().used, 1 + 8);
    let node = b.menu.root.iter().find(|n| n.item.label == "Smile").unwrap();
    assert_eq!((node.item.kind, node.item.sub_parameters.as_slice()), (MenuItemKind::RadialPuppet, &[sl.name.clone()][..]));
    let l = layer(&c, "[gen] Smile");
    assert_eq!(l.machine.states[1].time_parameter, sl.name);
    assert_eq!(l.machine.transitions[0].conditions[0].op, CondOp::Greater, "inactive at zero");
    let Motion::Clip(rel) = &l.machine.states[1].motion else { panic!() };
    let clip = load_clip(&b, rel);
    assert_eq!(clip.length, 1.0);
    let t = &clip.tracks[0];
    assert_eq!((t.property, t.name.as_str(), t.keys.len()), (TrackProperty::BlendShape, "Smile", 2));
    assert_eq!((t.keys[0].time, t.keys[0].value[0], t.keys[1].time, t.keys[1].value[0]), (0.0, 0.0, 1.0, 1.0));
}

#[test]
fn building_twice_gives_identical_bytes_and_scene() {
    let dir = temp_dir("twice");
    let tags = |t: &str| MenuToggle { exclusive_tags: vec![t.into()], ..hat_toggle(&format!("Outfit/{t}"), "Hat") };
    let scene = avatar(vec![
        entry("Hat", None, vec![comp(&tags("A")), comp(&hat_toggle("Extra", "Hat"))]),
        entry("Cape", None, vec![comp(&MenuToggle { exclusive_tags: vec!["A".into()], ..hat_toggle("Cape", "Cape") })]),
    ]);
    let mut e = env(&dir);
    e.write = true;
    let (s1, b1) = build_avatar_cloned(&scene, 0, &e);
    let first = std::fs::read(dir.join(&b1.out_dir).join("controller.animctrl")).unwrap();
    let (s2, b2) = build_avatar_cloned(&scene, 0, &e);
    assert_eq!(b1.files, b2.files);
    assert_eq!(b1.params, b2.params);
    assert_eq!(b1.menu, b2.menu);
    assert_eq!(ron::to_string(&s1).unwrap(), ron::to_string(&s2).unwrap());
    assert_eq!(first, std::fs::read(dir.join(&b2.out_dir).join("controller.animctrl")).unwrap());
    // Rebuilding the scene from an already-built clone is a no-op for files.
    let (_, b3) = build_avatar_cloned(&s1, 0, &e);
    assert!(b3.layers.is_empty() && b3.generated_params.is_empty(), "nothing left to generate from a built avatar");
    // Stale files from an earlier build disappear on the next one.
    let stale = dir.join(&b1.out_dir).join("toggles/old.anim");
    std::fs::write(&stale, "x").unwrap();
    let (_, _b4) = build_avatar_cloned(&scene, 0, &e);
    assert!(!stale.exists());
    // Every listed file is on disk with the same bytes.
    for (rel, bytes) in &b1.files {
        assert_eq!(&std::fs::read(dir.join(rel)).unwrap(), bytes, "{rel}");
    }
    // Names are hashes of paths, not counters: they do not depend on other toggles.
    let solo = avatar(vec![entry("Hat", None, vec![comp(&tags("A"))])]);
    let (_, bs) = build_avatar_cloned(&solo, 0, &env(&dir));
    assert!(b1.generated_params.contains(&bs.generated_params[0]));
}

#[test]
fn exclusive_tags_switch_the_others_off() {
    let dir = temp_dir("exclusive");
    let t = |path: &str, target: &str, tag: &str| MenuToggle { exclusive_tags: vec![tag.into()], ..hat_toggle(path, target) };
    let scene = avatar(vec![
        entry("Hat", None, vec![comp(&t("Hat", "Hat", "headwear"))]),
        entry("Cap", None, vec![comp(&t("Cap", "Cap", "headwear"))]),
        entry("Crown", None, vec![comp(&t("Crown", "Crown", "headwear"))]),
        // A tag used once makes no group.
        entry("Wing", None, vec![comp(&t("Wing", "Wing", "lonely"))]),
        // The same members under two tags make one group.
        entry("S1", None, vec![comp(&MenuToggle { exclusive_tags: vec!["x".into(), "y".into()], ..hat_toggle("S1", "S1") })]),
        entry("S2", None, vec![comp(&MenuToggle { exclusive_tags: vec!["x".into(), "y".into()], ..hat_toggle("S2", "S2") })]),
    ]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let c = load_ctrl(&b);
    let names: Vec<&str> = c.layers.iter().map(|l| l.name.as_str()).filter(|n| n.starts_with("Exclusive")).collect();
    assert_eq!(names, vec!["Exclusive Tag - headwear", "Exclusive Tag - x"], "lonely tags and duplicate groups are skipped: {names:?}");
    let l = layer(&c, "Exclusive Tag - headwear");
    assert_eq!(l.machine.states.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["Idle", "On Hat", "On Cap", "On Crown"]);
    // Entering "On Cap" drives the other two toggles off, and only those.
    let cap = l.machine.states.iter().find(|s| s.name == "On Cap").unwrap();
    assert_eq!(cap.behaviours, vec![PARAMETER_DRIVER.to_string()]);
    let settings = ParamDriverSettings::from_ron(&cap.behaviour_settings[PARAMETER_DRIVER]).unwrap();
    let driven: Vec<&str> = settings.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(driven.len(), 2);
    assert!(driven.iter().all(|n| n.starts_with("Tgl_Hat_") || n.starts_with("Tgl_Crown_")), "{driven:?}");
    assert!(settings.entries.iter().all(|e| matches!(e.op, DriverOp::Set { value } if value == 0.0)));
    // Entered from Any State when its own toggle turns on.
    let enter = l.machine.transitions.iter().find(|t| t.to == "On Cap").unwrap();
    assert!(enter.from.is_none() && !enter.can_transition_to_self);
    assert_eq!(enter.conditions[0].op, CondOp::If);
}

#[test]
fn exclusive_off_state_forces_a_default_member_on() {
    let dir = temp_dir("offstate");
    let mk = |n: &str, off: bool| MenuToggle { exclusive_tags: vec!["outfit".into()], exclusive_off_state: off, ..hat_toggle(n, n) };
    let scene = avatar(vec![entry("Base", None, vec![comp(&mk("Base", true))]), entry("Fancy", None, vec![comp(&mk("Fancy", false))])]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    let c = load_ctrl(&b);
    let l = layer(&c, "Exclusive Tag - Off States");
    let force = l.machine.transitions.iter().find(|t| t.to == "Force Base").unwrap();
    assert_eq!(force.conditions.len(), 2, "this toggle and the one it conflicts with are both off");
    assert!(force.conditions.iter().all(|c| c.op == CondOp::IfNot));
}

fn ctrl_with_condition(param: &str, clip: &str) -> AnimatorController {
    let mut c = AnimatorController::default();
    c.parameters.push(AnimParameter { name: param.into(), kind: AnimParamKind::Float, default: 0.0 });
    let m = &mut c.layers[0].machine;
    m.states.push(AnimState { name: "Idle".into(), ..Default::default() });
    m.states.push(AnimState { name: "Glow".into(), motion: Motion::Clip(clip.into()), ..Default::default() });
    m.transitions.push(AnimTransition {
        from: Some("Idle".into()),
        to: "Glow".into(),
        conditions: vec![AnimCondition { parameter: param.into(), op: CondOp::Greater, threshold: 0.5 }],
        ..Default::default()
    });
    c
}

#[test]
fn merged_animators_are_scoped_and_rebased() {
    let dir = temp_dir("merge");
    let mut clip = AnimClip::default();
    let mut t = AnimTrack::new("Lamp", TrackProperty::Active);
    t.keys.push(Keyframe { time: 0.0, value: [1.0, 0.0, 0.0, 0.0], ..Default::default() });
    clip.tracks.push(t);
    crate::save_anim_clip(dir.join("glow.anim"), &clip).unwrap();
    crate::save_anim_controller(dir.join("glow.animctrl"), &ctrl_with_condition("Glow", "glow.anim")).unwrap();

    let prop_params = AvatarParameters {
        params: vec![AvatarParam { scope: AvatarParamScope::Internal, ..AvatarParam::new("Glow", AvatarParamKind::Float) }],
        ..Default::default()
    };
    let merge = MergeAnimator { controller: "glow.animctrl".into(), ..Default::default() };
    let item = AvatarMenuItem { label: "Glow".into(), kind: MenuItemKind::RadialPuppet, sub_parameters: vec!["Glow".into()], ..Default::default() };
    let scene = avatar(vec![
        entry("Prop", None, vec![comp(&prop_params), comp(&merge), comp(&item)]),
        entry("Lamp", Some(0), vec![]),
    ]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);

    // The internal parameter is renamed deterministically: name$ + 12 hex of the path hash.
    let renamed = b.params.iter().find(|p| p.name.starts_with("Glow$")).expect("renamed parameter");
    assert_eq!(renamed.name, format!("Glow${}", &sha256_hex(b"Prop")[..12]));
    assert!(b.params.iter().all(|p| p.name != "Glow"));

    // Everything in the subtree follows: the merged controller and the menu.
    let c = load_ctrl(&b);
    let l = layer(&c, "[merge] Base Layer");
    assert_eq!(l.machine.transitions[0].conditions[0].parameter, renamed.name);
    assert!(c.parameters.iter().any(|p| p.name == renamed.name) && c.parameters.iter().all(|p| p.name != "Glow"));
    assert_eq!(b.menu.root[0].item.sub_parameters, vec![renamed.name.clone()]);

    // Clip paths gain the merge object's path.
    let Motion::Clip(rebased) = &l.machine.states[1].motion else { panic!() };
    assert!(rebased.contains("/merged/"), "{rebased}");
    assert_eq!(load_clip(&b, rebased).tracks[0].target, "Prop/Lamp");
}

#[test]
fn replace_mode_drops_the_avatars_own_layers_and_priority_orders_appends() {
    let dir = temp_dir("replace");
    let mut base = AnimatorController::default();
    base.layers[0].name = "Locomotion".into();
    crate::save_anim_controller(dir.join("base.animctrl"), &base).unwrap();
    let mk = |name: &str| {
        let mut c = AnimatorController::default();
        c.layers[0].name = name.into();
        crate::save_anim_controller(dir.join(format!("{name}.animctrl")), &c).unwrap();
    };
    mk("A");
    mk("B");
    let animator = Animator { controller: "base.animctrl".into(), ..Default::default() };
    let m = |name: &str, priority: i32, mode: zestengine_core::MergeMode| {
        comp(&MergeAnimator { controller: format!("{name}.animctrl"), priority, mode, ..Default::default() })
    };
    let mut scene = avatar(vec![
        entry("MA", None, vec![m("A", 5, zestengine_core::MergeMode::Append)]),
        entry("MB", None, vec![m("B", 1, zestengine_core::MergeMode::Append)]),
    ]);
    scene.entries[0].components = vec![comp(&AvatarDescriptor::default()), comp(&animator)];
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    let names: Vec<String> = load_ctrl(&b).layers.iter().map(|l| l.name.clone()).collect();
    assert_eq!(names, vec!["Locomotion", "[merge] B", "[merge] A"], "avatar first, then merges by priority");

    scene.entries[1].components = vec![m("A", 5, zestengine_core::MergeMode::Replace)];
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    let names: Vec<String> = load_ctrl(&b).layers.iter().map(|l| l.name.clone()).collect();
    assert_eq!(names, vec!["[merge] A", "[merge] B"], "Replace clears the avatar's layers and runs first");
}

#[test]
fn parameter_conflicts_are_reported() {
    let dir = temp_dir("conflicts");
    let decl = |kind, default| AvatarParameters { params: vec![AvatarParam { default, ..AvatarParam::new("Shared", kind) }], ..Default::default() };
    let scene = avatar(vec![
        entry("A", None, vec![comp(&decl(AvatarParamKind::Float, 0.5))]),
        entry("B", None, vec![comp(&decl(AvatarParamKind::Float, 0.25))]),
    ]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok());
    assert!(b.warnings.iter().any(|w| w.contains("Shared") && w.contains("last one wins")), "{:?}", b.warnings);
    assert_eq!(b.params.iter().filter(|p| p.name == "Shared").count(), 1);
    assert_eq!(b.params.iter().find(|p| p.name == "Shared").unwrap().default, 0.25);

    let scene = avatar(vec![
        entry("A", None, vec![comp(&decl(AvatarParamKind::Float, 0.0))]),
        entry("B", None, vec![comp(&decl(AvatarParamKind::Int, 0.0))]),
    ]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.errors.iter().any(|e| e.contains("Shared") && e.contains("Int") && e.contains("Float")), "{:?}", b.errors);
}

#[test]
fn rename_to_scope_and_global_scope() {
    let dir = temp_dir("rename");
    let params = AvatarParameters {
        params: vec![
            AvatarParam { scope: AvatarParamScope::RenameTo("Wetness".into()), ..AvatarParam::new("Wet", AvatarParamKind::Float) },
            AvatarParam::new("Hat", AvatarParamKind::Bool),
        ],
        ..Default::default()
    };
    let item = AvatarMenuItem { label: "Wet".into(), kind: MenuItemKind::Toggle, parameter: "Wet".into(), ..Default::default() };
    let scene = avatar(vec![entry("Prop", None, vec![comp(&params), comp(&item)])]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    let names: Vec<&str> = b.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["Wetness", "Hat"]);
    assert_eq!(b.menu.root[0].item.parameter, "Wetness");
}

#[test]
fn object_toggle_follows_the_nearest_menu_item() {
    let dir = temp_dir("reactive");
    let item = AvatarMenuItem { label: "Wings".into(), kind: MenuItemKind::Toggle, ..Default::default() };
    let reactive = ObjectToggle { objects: vec![ObjectToggleEntry { target: "Wings/Left".into(), active: true }], inverted: false };
    let scene = avatar(vec![
        entry("Wings", None, vec![comp(&item)]),
        entry("Reactor", Some(0), vec![comp(&reactive)]),
        entry("Left", Some(0), vec![]),
        // No menu item above this one: reported, not built.
        entry("Orphan", None, vec![comp(&ObjectToggle { objects: vec![], inverted: false })]),
    ]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    // The item had no parameter, so it was given one.
    let auto = b.generated_params.iter().find(|p| p.starts_with("AutoParam_Wings_")).expect("auto parameter");
    assert_eq!(b.menu.root[0].item.parameter, *auto);
    let c = load_ctrl(&b);
    let l = layer(&c, "[gen] reactive Wings/Reactor");
    assert_eq!(l.machine.transitions[0].conditions[0].parameter, *auto);
    // Resting: the object the item switches on starts off.
    assert!(!built.entries[3].enabled);
    assert!(b.warnings.iter().any(|w| w.contains("Orphan") && w.contains("Avatar Menu Item")), "{:?}", b.warnings);
    assert!(built.entries.iter().all(|e| e.components.iter().all(|c| c.kind != ObjectToggle::NAME)));
}

#[test]
fn conflicting_resting_states_are_an_error() {
    let dir = temp_dir("rest-conflict");
    let on = hat_toggle("Show", "Hat");
    let off = MenuToggle { actions: vec![ToggleAction::Active { target: "Hat".into(), on: false }], ..hat_toggle("Hide", "Hat") };
    let scene = avatar(vec![entry("Hat", None, vec![comp(&on), comp(&off)])]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.errors.iter().any(|e| e.contains("disagree") && e.contains("Hat")), "{:?}", b.errors);
}

#[test]
fn the_synced_bit_budget_is_enforced() {
    let dir = temp_dir("budget");
    let many: Vec<SceneEntry> = (0..9).map(|i| entry(&format!("S{i}"), None, vec![comp(&MenuToggle { mode: ToggleMode::Slider { default: 0.0, inactive_at_zero: true }, ..hat_toggle(&format!("S{i}"), &format!("S{i}")) })])).collect();
    let scene = avatar(many);
    let mut e = env(&dir);
    let (_, b) = build_avatar_cloned(&scene, 0, &e);
    // 9 floats = 72 bits, fine; the project cap can still refuse it.
    assert!(b.is_ok());
    assert_eq!(b.budget.unwrap().used, 72);
    e.budget_cap = 64;
    let (_, b) = build_avatar_cloned(&scene, 0, &e);
    assert!(b.errors.iter().any(|e| e.contains("72") && e.contains("64")), "{:?}", b.errors);
    // Local-only toggles are free.
    let local = avatar(vec![entry("L", None, vec![comp(&MenuToggle { synced: false, ..hat_toggle("L", "L") })])]);
    let (_, b) = build_avatar_cloned(&local, 0, &env(&dir));
    assert_eq!(b.budget.unwrap().used, 0);
}

#[test]
fn menus_merge_hierarchy_items_with_generated_entries_and_page_at_eight() {
    let dir = temp_dir("menu");
    let sub = AvatarMenuItem { label: "Outfit".into(), kind: MenuItemKind::SubMenu, parameter: String::new(), ..Default::default() };
    let child = AvatarMenuItem { label: "Shoes".into(), kind: MenuItemKind::Toggle, parameter: "ShoesOn".into(), ..Default::default() };
    let decl = AvatarParameters { params: vec![AvatarParam::new("ShoesOn", AvatarParamKind::Bool)], ..Default::default() };
    let mut kids = vec![
        entry("Outfit", None, vec![comp(&sub)]),
        entry("Shoes", Some(0), vec![comp(&child)]),
        entry("Decl", None, vec![comp(&decl)]),
        // A generated entry joins the hierarchy's "outfit" sub menu (labels match loosely).
        entry("Hat", None, vec![comp(&hat_toggle("outfit/Hat", "Hat"))]),
    ];
    for i in 0..8 {
        kids.push(entry(&format!("T{i}"), None, vec![comp(&hat_toggle(&format!("T{i}"), &format!("T{i}")))]));
    }
    let (_, b) = build_avatar_cloned(&avatar(kids), 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let outfit = b.menu.root.iter().find(|n| n.item.label == "Outfit").unwrap();
    assert_eq!(outfit.children.iter().map(|n| n.item.label.as_str()).collect::<Vec<_>>(), vec!["Shoes", "Hat"]);
    assert!(b.menu.root.len() <= zestengine_core::avatar_params::MAX_MENU_CONTROLS);
    assert_eq!(b.menu.root.last().unwrap().item.label, "More");
    assert!(!b.warnings.iter().any(|w| w.starts_with("menu:")), "{:?}", b.warnings);
    // A menu item naming an undeclared parameter is reported.
    let bad = AvatarMenuItem { label: "Bad".into(), parameter: "Nope".into(), ..Default::default() };
    let (_, b) = build_avatar_cloned(&avatar(vec![entry("Bad", None, vec![comp(&bad)])]), 0, &env(&dir));
    assert!(b.warnings.iter().any(|w| w.starts_with("menu:") && w.contains("Nope")), "{:?}", b.warnings);
}

#[test]
fn an_avatar_without_authoring_components_is_left_alone() {
    let dir = temp_dir("plain");
    let scene = avatar(vec![entry("Body", None, vec![])]);
    assert!(!needs_build(&scene, 0));
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok() && b.files.is_empty() && b.layers.is_empty());
    assert_eq!(ron::to_string(&built).unwrap(), ron::to_string(&scene).unwrap());
    assert!(needs_build(&avatar(vec![entry("H", None, vec![comp(&hat_toggle("H", "H"))])]), 0));
    assert_eq!(avatar_roots(&scene), vec![0]);
}

#[test]
fn passes_run_in_the_documented_order() {
    let dir = temp_dir("order");
    let (_, b) = build_avatar_cloned(&avatar(vec![]), 0, &env(&dir));
    assert_eq!(
        b.passes,
        vec![
            "resolve",
            "scope_parameters",
            "assign_menu_parameters",
            "generate_toggles",
            "generate_reactive",
            "exclusive_tags",
            "optimise_toggles",
            "merge_animators",
            "resting_state",
            "merge_armature",
            "defaults_layer",
            "flatten_controller",
            "finalize_menu",
            "lints",
            "budget",
            "emit"
        ]
    );
}

#[test]
fn merge_armature_zips_matching_bones_and_keeps_the_rest_in_place() {
    let dir = temp_dir("armature");
    let at = |mut e: SceneEntry, t: [f32; 3]| {
        e.translation = t;
        e
    };
    // Avatar: Armature/Hips/Spine.   Outfit: Outfit/Armature/Hips/Spine/Skirt and Hips/Tail.
    let merge = MergeArmature { target: "Armature".into(), ..Default::default() };
    let scene = avatar(vec![
        at(entry("Armature", None, vec![]), [0.0, 0.0, 0.0]),
        at(entry("Hips", Some(0), vec![]), [0.0, 1.0, 0.0]),
        at(entry("Spine", Some(1), vec![]), [0.0, 0.2, 0.0]),
        entry("Outfit", None, vec![]),
        at(entry("Armature", Some(3), vec![comp(&merge)]), [0.0, 0.0, 0.0]),
        at(entry("Hips", Some(4), vec![]), [0.0, 1.0, 0.0]),
        at(entry("Spine", Some(5), vec![]), [0.0, 0.2, 0.0]),
        at(entry("Skirt", Some(6), vec![]), [0.0, 0.1, 0.5]),
        at(entry("Tail", Some(5), vec![]), [0.0, 0.0, -0.3]),
    ]);
    // A toggle clip aimed at the outfit's skirt must follow it.
    let toggle = MenuToggle { actions: vec![ToggleAction::Active { target: "Outfit/Armature/Hips/Spine/Skirt".into(), on: true }], ..hat_toggle("Skirt", "x") };
    let mut scene = scene;
    scene.entries[4].components.push(comp(&toggle));
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let names: Vec<&str> = built.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["Avatar", "Armature", "Hips", "Spine", "Outfit", "Skirt", "Tail"], "{names:?}");
    let idx = |n: &str| built.entries.iter().position(|e| e.name == n).unwrap();
    // Skirt hangs from the avatar's Spine, Tail from its Hips; positions kept in world space.
    assert_eq!(built.entries[idx("Skirt")].parent, Some(idx("Spine")));
    assert_eq!(built.entries[idx("Tail")].parent, Some(idx("Hips")));
    let skirt = built.entries[idx("Skirt")].translation;
    assert!((skirt[1] - 0.1).abs() < 1e-5 && (skirt[2] - 0.5).abs() < 1e-5, "{skirt:?}");
    let tail = built.entries[idx("Tail")].translation;
    assert!(tail[1].abs() < 1e-5 && (tail[2] + 0.3).abs() < 1e-5, "same world place under the avatar's Hips: {tail:?}");
    // All parents come before their children.
    for (i, e) in built.entries.iter().enumerate() {
        assert!(e.parent.is_none_or(|p| p < i));
    }
    // The clip aimed at the old path follows the bone.
    let ctrl = load_ctrl(&b);
    let l = ctrl.layers.iter().find(|l| l.name.starts_with("[gen] Skirt")).unwrap();
    let Motion::Clip(rel) = &l.machine.states[1].motion else { panic!() };
    assert_eq!(load_clip(&b, rel).tracks[0].target, "Armature/Hips/Spine/Skirt");

    // Inside the editor's Play nothing moves.
    let mut play = env(&dir);
    play.structural = false;
    let (kept, b) = build_avatar_cloned(&scene, 0, &play);
    assert_eq!(kept.entries.len(), scene.entries.len());
    assert!(b.warnings.iter().any(|w| w.contains("Merge Armature")), "{:?}", b.warnings);
}

#[test]
fn menu_paths_are_split_with_escapes() {
    assert_eq!(split_menu_path("A/B"), vec!["A", "B"]);
}

#[test]
fn packaging_an_avatar_runs_the_build_and_ships_the_generated_assets() {
    use crate::package::{BuildOptions, ComponentStatus, ContentLimits, plan_package};
    let dir = temp_dir("package");
    let scene = avatar(vec![entry("Hat", None, vec![comp(&hat_toggle("Hat", "Hat"))])]);
    let classify = |_: &str| ComponentStatus::Builtin;
    let opts = BuildOptions {
        kind: zestengine_core::ContentKind::Avatar,
        name: "Test".into(),
        author: String::new(),
        version: "1".into(),
        description: String::new(),
        project_root: dir.clone(),
        scene,
        entry: "avatars/test.scene".into(),
        thumbnail: None,
        classify: &classify,
        strip_plugin_components: false,
        include_texture_cache: false,
        limits: ContentLimits::for_kind(zestengine_core::ContentKind::Avatar),
    };
    let plan = plan_package(&opts).unwrap();
    assert!(plan.is_ok(), "{:?}", plan.errors);
    let files: Vec<&String> = plan.files.keys().collect();
    assert!(files.iter().any(|f| f.ends_with("controller.animctrl")), "{files:?}");
    assert!(files.iter().any(|f| f.contains("/toggles/") && f.ends_with(".anim")), "{files:?}");
    assert!(plan.manifest.components.iter().any(|c| c == "Avatar Parameters") && plan.manifest.components.iter().any(|c| c == "Avatar Menu"));
    assert!(!plan.manifest.components.iter().any(|c| c == MenuToggle::NAME), "authoring components are not shipped");
    let scene_text = String::from_utf8(plan.files.get("files/avatars/test.scene").unwrap().clone()).unwrap();
    assert!(!scene_text.contains("Menu Toggle"));
    // The over-budget case fails the package.
    let mut over = avatar((0..40).map(|i| entry(&format!("S{i}"), None, vec![comp(&MenuToggle { mode: ToggleMode::Slider { default: 0.0, inactive_at_zero: true }, ..hat_toggle(&format!("S{i}"), &format!("S{i}")) })])).collect());
    over.entries[0].components.push(comp(&AvatarParameters::default()));
    let opts = BuildOptions { scene: over, ..opts };
    let plan = plan_package(&opts).unwrap();
    assert!(plan.errors.iter().any(|e| e.contains("avatar build") && e.contains("synced bits")), "{:?}", plan.errors);
}

// ----------------------------------------------------- AVP-3 build leftovers

fn two_hat_toggles() -> SceneFile {
    // Two toggles on different parameters that both switch the same object.
    avatar(vec![
        entry("Hat", None, vec![]),
        entry("A", None, vec![comp(&hat_toggle("Show A", "Hat"))]),
        entry("B", None, vec![comp(&MenuToggle { actions: vec![ToggleAction::Active { target: "Hat".into(), on: true }], ..hat_toggle("Show B", "Hat") })]),
    ])
}

#[test]
fn toggles_on_the_same_properties_fold_into_one_layer_later_wins() {
    let dir = temp_dir("fold");
    let (_, b) = build_avatar_cloned(&two_hat_toggles(), 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    assert_eq!(b.layers_merged, 1);
    let c = load_ctrl(&b);
    let toggle_layers: Vec<&AnimLayer> = c.layers.iter().filter(|l| l.name.starts_with("[gen]")).collect();
    assert_eq!(toggle_layers.len(), 1, "{:?}", c.layers.iter().map(|l| &l.name).collect::<Vec<_>>());
    let l = toggle_layers[0];
    assert_eq!(l.machine.states.len(), 3, "Off plus one On per toggle");
    // The first toggle only wins while the second is off.
    let first = l.machine.transitions.iter().find(|t| t.to == "On Show A").unwrap();
    assert_eq!(first.from, None);
    assert_eq!(first.conditions.len(), 2);
    let last = l.machine.transitions.iter().find(|t| t.to == "On Show B").unwrap();
    assert_eq!(last.conditions.len(), 1);
    // Both toggles still have their clip and parameter.
    assert_eq!(b.generated_params.len(), 2);
    // Different properties are left alone.
    let (_, b2) = build_avatar_cloned(
        &avatar(vec![entry("Hat", None, vec![comp(&hat_toggle("Hat", "Hat"))]), entry("Coat", None, vec![comp(&hat_toggle("Coat", "Coat"))])]),
        0,
        &env(&dir),
    );
    assert_eq!(b2.layers_merged, 0);
}

#[test]
fn toggles_writing_one_property_from_different_parameters_are_linted() {
    let dir = temp_dir("lint");
    let (_, b) = build_avatar_cloned(&two_hat_toggles(), 0, &env(&dir));
    assert!(b.warnings.iter().any(|w| w.starts_with("lint:") && w.contains("Hat") && w.contains("last one")), "{:?}", b.warnings);
    // An action whose target is missing leaves an empty clip.
    let scene = avatar(vec![entry("A", None, vec![comp(&hat_toggle("Ghost", "Nope"))])]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.warnings.iter().any(|w| w.contains("lint:") && w.contains("empty")), "{:?}", b.warnings);
}

#[test]
fn reactive_toggles_sharing_an_object_get_one_layer_for_it() {
    let dir = temp_dir("property-centric");
    let item = |l: &str| AvatarMenuItem { label: l.into(), kind: MenuItemKind::Toggle, ..Default::default() };
    let rx = |t: &[&str]| ObjectToggle { objects: t.iter().map(|t| ObjectToggleEntry { target: (*t).into(), active: true }).collect(), inverted: false };
    let scene = avatar(vec![
        entry("Wings", None, vec![comp(&item("Wings")), comp(&rx(&["Shared", "WingOnly"]))]),
        entry("Glow", None, vec![comp(&item("Glow")), comp(&rx(&["Shared"]))]),
        entry("Shared", None, vec![]),
        entry("WingOnly", None, vec![]),
    ]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    // Shared: one merged layer; WingOnly keeps a layer of its own.
    assert_eq!(b.layers_merged, 1);
    assert_eq!(b.layers.iter().filter(|l| l.name.contains("merged")).count(), 1);
    assert_eq!(b.layers.iter().filter(|l| l.name.starts_with("[gen] reactive Wings")).count(), 2);
}

#[test]
fn a_selector_is_one_int_parameter_with_numbered_options() {
    let dir = temp_dir("selector");
    let opt = |name: &str, target: &str, on: bool| MenuToggle { selector: "Outfit".into(), default_on: on, ..hat_toggle(&format!("Outfit/{name}"), target) };
    let scene = avatar(vec![
        entry("Red", None, vec![comp(&opt("Red", "Red", false))]),
        entry("Blue", None, vec![comp(&opt("Blue", "Blue", true))]),
        entry("Green", None, vec![comp(&opt("Green", "Green", false))]),
    ]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let ints: Vec<&AvatarParam> = b.params.iter().filter(|p| p.kind == AvatarParamKind::Int).collect();
    assert_eq!(ints.len(), 1, "one Int, not three Bools: {:?}", b.params);
    assert_eq!(ints[0].default, 2.0, "Blue is the default option");
    assert_eq!(b.budget.unwrap().used, 8);
    // Menu items carry the option numbers.
    let page = &b.menu.root[0].children;
    let values: Vec<f32> = page.iter().map(|n| n.item.value).collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
    assert!(page.iter().all(|n| n.item.parameter == ints[0].name));
    // Each option's layer reads Equals(n), and defaults rest on Blue.
    let c = load_ctrl(&b);
    let blue = c.layers.iter().find(|l| l.name.contains("Blue")).unwrap();
    assert_eq!(blue.machine.transitions[0].conditions[0].op, CondOp::Equals);
    assert_eq!(blue.machine.transitions[0].conditions[0].threshold, 2.0);
    assert!(built.entries.iter().find(|e| e.name == "Blue").unwrap().enabled);
    assert!(!built.entries.iter().find(|e| e.name == "Red").unwrap().enabled);
}

#[test]
fn local_actions_get_their_own_is_local_layer_and_fades_apply() {
    let dir = temp_dir("local");
    let t = MenuToggle {
        local_actions: vec![ToggleAction::Active { target: "FirstPersonArm".into(), on: true }],
        transition_in: 0.3,
        transition_out: 0.1,
        ..hat_toggle("Hat", "Hat")
    };
    let scene = avatar(vec![entry("Hat", None, vec![comp(&t)]), entry("FirstPersonArm", None, vec![])]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let c = load_ctrl(&b);
    let main = layer(&c, "[gen] Hat");
    assert!((main.machine.transitions[0].duration - 0.3).abs() < 1e-6);
    assert!((main.machine.transitions[1].duration - 0.1).abs() < 1e-6);
    let local = layer(&c, "[gen] Hat (local)");
    assert!(local.machine.transitions[0].conditions.iter().any(|c| c.parameter == "IsLocal" && c.op == CondOp::If));
    assert!(local.machine.transitions.iter().any(|t| t.conditions.iter().any(|c| c.parameter == "IsLocal" && c.op == CondOp::IfNot)));
    assert!(c.parameters.iter().any(|p| p.name == "IsLocal"));
}

#[test]
fn a_menu_installer_puts_an_authored_page_at_its_install_target() {
    let dir = temp_dir("installer");
    let mut page = AvatarMenu::default();
    page.insert_at(
        &["Cape".to_owned()],
        AvatarMenuItem { label: "Cape".into(), kind: MenuItemKind::Toggle, parameter: "CapeOn".into(), ..Default::default() },
    );
    std::fs::create_dir_all(dir.join("menus")).unwrap();
    page.save(&dir.join("menus/extras.avmenu")).unwrap();
    let inst = MenuInstaller { menu: "menus/extras.avmenu".into(), install_to: "Outfits/Extras".into() };
    let scene = avatar(vec![entry("Pack", None, vec![comp(&inst)])]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let outfits = &b.menu.root[0];
    assert_eq!(outfits.item.label, "Outfits");
    assert_eq!(outfits.children[0].item.label, "Extras");
    assert_eq!(outfits.children[0].children[0].item.label, "Cape");
    assert!(built.entries.iter().all(|e| e.components.iter().all(|c| c.kind != MenuInstaller::NAME)));
    // A missing asset is a warning, not a failure.
    let scene = avatar(vec![entry("Pack", None, vec![comp(&MenuInstaller { menu: "nope.avmenu".into(), install_to: String::new() })])]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.warnings.iter().any(|w| w.contains("Menu Installer")), "{:?}", b.warnings);
}

#[test]
fn big_menu_icons_are_downscaled_into_the_build_folder() {
    let dir = temp_dir("icons");
    let img = image::RgbaImage::from_pixel(1024, 512, image::Rgba([200, 30, 30, 255]));
    img.save(dir.join("big.png")).unwrap();
    image::RgbaImage::from_pixel(64, 64, image::Rgba([1, 2, 3, 255])).save(dir.join("small.png")).unwrap();
    let toggle = |icon: &str, p: &str| MenuToggle { icon: icon.into(), ..hat_toggle(p, "Hat") };
    let scene = avatar(vec![
        entry("Hat", None, vec![comp(&toggle("big.png", "Big"))]),
        entry("Cap", None, vec![comp(&toggle("small.png", "Small"))]),
    ]);
    let (_, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    let big = b.menu.root.iter().find(|n| n.item.label == "Big").unwrap();
    assert!(big.item.icon.contains("/icons/"), "{}", big.item.icon);
    let bytes = b.files.get(&big.item.icon).expect("icon file generated");
    let out = image::load_from_memory(bytes).unwrap();
    assert_eq!((out.width(), out.height()), (256, 128));
    let small = b.menu.root.iter().find(|n| n.item.label == "Small").unwrap();
    assert_eq!(small.item.icon, "small.png", "already small: untouched");
    assert!(downscale_icon(bytes, 256).is_none());
}

#[test]
fn an_unchanged_build_is_recognised_by_its_input_hash() {
    let dir = temp_dir("hash");
    let scene = avatar(vec![entry("Hat", None, vec![comp(&hat_toggle("Hat", "Hat"))])]);
    let e = BuildEnv { write: true, ..env(&dir) };
    let (_, first) = build_avatar_cloned(&scene, 0, &e);
    assert!(!first.input_hash.is_empty() && !first.unchanged);
    let (_, second) = build_avatar_cloned(&scene, 0, &e);
    assert!(second.unchanged);
    assert_eq!(first.input_hash, second.input_hash);
    assert_eq!(first.files, second.files, "skipping must not change the result");
    // Any input change (here the budget cap) rebuilds; a deleted output too.
    let (_, third) = build_avatar_cloned(&scene, 0, &BuildEnv { budget_cap: 100, ..BuildEnv { project_root: e.project_root, write: e.write, structural: e.structural, budget_cap: e.budget_cap } });
    assert!(!third.unchanged && third.input_hash != first.input_hash);
    let any = dir.join(first.files.keys().next().unwrap());
    std::fs::remove_file(&any).unwrap();
    let (_, fourth) = build_avatar_cloned(&scene, 0, &e);
    assert!(!fourth.unchanged);
    assert!(any.exists());
}

#[test]
fn the_budget_breakdown_lists_synced_costs_biggest_first() {
    let p = |n: &str, k: AvatarParamKind, sync| AvatarParam { name: n.into(), kind: k, sync, ..Default::default() };
    let v = budget_breakdown(&[
        p("b", AvatarParamKind::Bool, AvatarParamSync::Synced),
        p("i", AvatarParamKind::Int, AvatarParamSync::Synced),
        p("l", AvatarParamKind::Float, AvatarParamSync::LocalOnly),
    ]);
    assert_eq!(v, vec![("i".to_owned(), 8), ("b".to_owned(), 1)]);
}

fn model_entry(name: &str, parent: Option<usize>, path: &str) -> SceneEntry {
    let mut e = entry(name, parent, vec![]);
    e.source = ObjectSource::Model { path: path.into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] };
    e
}

/// The repo's default avatars copied into a project folder; `None` when they are not there.
fn avatar_project(tag: &str) -> Option<PathBuf> {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/avatars/default");
    let dir = temp_dir(tag);
    for f in ["male.glb", "female.glb"] {
        std::fs::copy(src.join(f), dir.join(f)).ok()?;
    }
    Some(dir)
}

#[test]
fn merge_armature_tags_merged_skinned_meshes_for_rebinding_and_reports_the_plan() {
    let Some(dir) = avatar_project("skinrebind") else {
        eprintln!("default avatars not found: skipped");
        return;
    };
    let merge = MergeArmature { target: "Armature".into(), ..Default::default() };
    let scene = avatar(vec![
        entry("Armature", None, vec![]),
        model_entry("Body", None, "female.glb"),
        entry("Outfit", None, vec![comp(&merge)]),
        model_entry("Shirt", Some(2), "male.glb"),
    ]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.is_ok(), "{:?}", b.errors);
    let shirt = built.entries.iter().find(|e| e.name == "Shirt").unwrap();
    let tag: SkinRebound = comps::<SkinRebound>(shirt).pop().expect("the merged mesh carries Skin Rebound");
    assert_eq!(tag.avatar, "id-Body", "bound to the object that draws the avatar's skinned model");
    assert!(comps::<SkinRebound>(built.entries.iter().find(|e| e.name == "Body").unwrap()).is_empty(), "the avatar's own mesh is not tagged");
    assert_eq!(b.skin_rebinds.len(), 1);
    let r = &b.skin_rebinds[0];
    eprintln!("skin rebind report: {r:?}");
    assert_eq!((r.outfit_model.as_str(), r.avatar_model.as_str()), ("male.glb", "female.glb"));
    assert!(r.matched > 0 && r.matched <= r.joints && r.missing.len() + r.matched == r.joints);
    // Building again gives the same scene (the tag is deterministic and not duplicated).
    let (again, _) = build_avatar_cloned(&built, 0, &env(&dir));
    assert_eq!(again.entries.iter().find(|e| e.name == "Shirt").unwrap().components.iter().filter(|c| c.kind == SkinRebound::NAME).count(), 1);
    // In the editor's Play (no structural passes) nothing is tagged.
    let mut play = env(&dir);
    play.structural = false;
    let (kept, _) = build_avatar_cloned(&scene, 0, &play);
    assert!(kept.entries.iter().all(|e| comps::<SkinRebound>(e).is_empty()));
}

#[test]
fn merge_armature_warns_when_the_avatar_has_no_skinned_model_or_the_outfit_cannot_be_read() {
    let dir = temp_dir("skinwarn");
    let merge = MergeArmature::default();
    let scene = avatar(vec![entry("Armature", None, vec![]), entry("Outfit", None, vec![comp(&merge)]), model_entry("Shirt", Some(1), "missing.glb")]);
    let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
    assert!(b.warnings.iter().any(|w| w.contains("no skinned model")), "{:?}", b.warnings);
    assert!(b.skin_rebinds.is_empty());
    assert!(comps::<SkinRebound>(built.entries.iter().find(|e| e.name == "Shirt").unwrap()).is_empty());
    // With an avatar mesh present but an unreadable outfit model, the outfit is reported and untouched.
    if let Some(dir) = avatar_project("skinwarn2") {
        let scene = avatar(vec![entry("Armature", None, vec![]), model_entry("Body", None, "female.glb"), entry("Outfit", None, vec![comp(&merge)]), model_entry("Shirt", Some(2), "gone.glb")]);
        let (built, b) = build_avatar_cloned(&scene, 0, &env(&dir));
        assert!(b.warnings.iter().any(|w| w.contains("gone.glb")), "{:?}", b.warnings);
        assert!(comps::<SkinRebound>(built.entries.iter().find(|e| e.name == "Shirt").unwrap()).is_empty());
    }
}
