//! Tests of the FBX extras: Bake Animations, Import Constraints, Animated
//! Custom Properties, Keep Tangents and the compression modes, on
//! synthesised ASCII files (checked against ufbx's own evaluation) and on the
//! real files of the author's machine (skipped when absent).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec3};
use zestengine_core::{ConstraintKind, CustomValue};

use crate::anim_clip::TrackProperty;
use crate::fbx_test_util::{Chan, Doc, import, vec_prop};
use crate::skeleton::{AnimationClip, ChannelPath, Skeleton};
use crate::{AnimCompression, ModelImport, Scene};

fn load_ufbx(path: &Path) -> ufbx::SceneRoot {
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::TransformRoot,
        ..Default::default()
    };
    ufbx::load_file(path.to_str().unwrap(), opts).unwrap_or_else(|e| panic!("{}", e.description.as_ref()))
}

/// World matrices by node name from ufbx's own scene evaluation.
fn ufbx_worlds(path: &Path, stack: usize, t: f64) -> HashMap<String, Mat4> {
    let fbx = load_ufbx(path);
    let s = &fbx.anim_stacks[stack];
    let ev = ufbx::evaluate_scene(&fbx, &s.anim, s.time_begin + t, ufbx::EvaluateOpts::default()).unwrap();
    ev.nodes.iter().map(|n| (n.element.name.to_string(), crate::fbx_loader::matrix_to_mat4(&n.node_to_world))).collect()
}

fn joint_worlds(skel: &Skeleton, clip: &AnimationClip, t: f32) -> HashMap<String, Mat4> {
    let world = skel.world_matrices(&clip.sample(skel, t));
    skel.joints.iter().zip(world).map(|(j, w)| (j.name.clone(), w)).collect()
}

fn pos_of(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

/// Settings with every animation extra off and no key reduction.
fn dense() -> ModelImport {
    let mut m = ModelImport::default();
    m.animation.compression = AnimCompression::Off;
    m
}

fn write(doc: &Doc, tag: &str) -> PathBuf {
    doc.write(tag)
}

fn cleanup(p: &Path) {
    let _ = std::fs::remove_dir_all(p.parent().unwrap());
}

fn props(items: &[String]) -> String {
    items.join("\n")
}

/// A file with a pivot / offset / pre-rotation / non-uniform-scale chain, animated.
fn pivot_chain() -> Doc {
    let mut d = Doc::new();
    let root = d.bone("Root", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 0.0, 0.0]), 0);
    let mid = d.bone(
        "Mid",
        &props(&[
            vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 2.0, 0.0]),
            vec_prop("RotationPivot", "Vector3D", "", [3.0, 0.0, 0.0]),
            vec_prop("RotationOffset", "Vector3D", "", [0.0, 1.0, 0.0]),
            vec_prop("PreRotation", "Vector3D", "", [0.0, 20.0, 0.0]),
            vec_prop("Lcl Scaling", "Lcl Scaling", "A", [2.0, 1.0, 1.0]),
        ]),
        root,
    );
    let _tip = d.bone("Tip", &vec_prop("Lcl Translation", "Lcl Translation", "A", [1.0, 0.0, 0.0]), mid);
    d.stack(
        "Take",
        (0.0, 1.0),
        &[
            (mid, "Lcl Rotation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 90.0)]), Chan::linear("d|Y", &[(0.0, 0.0), (1.0, 30.0)])]),
            (root, "Lcl Translation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 5.0)])]),
        ],
    );
    d
}

#[test]
fn bake_animations_follow_pivots_offsets_and_inherited_scale() {
    let d = pivot_chain();
    let path = write(&d, "bake_pivot");
    let mut plain = dense();
    plain.animation.bake_animations = false;
    let mut baked = dense();
    baked.animation.bake_animations = true;
    let a = crate::load_fbx_with(&path, &plain).unwrap();
    let b = crate::load_fbx_with(&path, &baked).unwrap();
    let mut plain_err = 0.0f32;
    let mut baked_err = 0.0f32;
    for t in [0.0f32, 0.25, 0.5, 0.9999] {
        let truth = ufbx_worlds(&path, 0, t as f64);
        let wa = joint_worlds(&a.skeletons[0], &a.animations[0], t);
        let wb = joint_worlds(&b.skeletons[0], &b.animations[0], t);
        for name in ["Root", "Mid", "Tip"] {
            // The origin and a point a metre along each axis, so rotation and scale count too.
            for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
                let want = truth[name].transform_point3(p);
                plain_err = plain_err.max((wa[name].transform_point3(p) - want).length());
                baked_err = baked_err.max((wb[name].transform_point3(p) - want).length());
            }
        }
    }
    cleanup(&path);
    assert!(baked_err < 1e-3, "baked pose matches ufbx's evaluation (max {baked_err})");
    // ufbx folds pivots and offsets into each node's TRS, so plain sampling already follows them.
    assert!(plain_err < 0.01, "plain sampling follows pivots too ({plain_err})");
}

#[test]
fn non_standard_inherit_modes_are_followed_in_both_modes() {
    // "Rrs" (no parent scale inheritance) under a scaled, rotating parent: ufbx adds a scale
    // helper node, which the animation-only import keeps as a joint, so plain sampling is right too.
    let mut d = Doc::new();
    let root = d.bone("Root", "", 0);
    let mid = d.bone("Mid", &vec_prop("Lcl Scaling", "Lcl Scaling", "A", [2.0, 2.0, 2.0]), root);
    let _tip = d.bone("Tip", &props(&["P: \"InheritType\", \"enum\", \"\", \"\",2".to_string(), vec_prop("Lcl Translation", "Lcl Translation", "A", [1.0, 0.0, 0.0])]), mid);
    d.stack("Take", (0.0, 1.0), &[(mid, "Lcl Rotation", vec![Chan::linear("d|Z", &[(0.0, 0.0), (1.0, 90.0)])])]);
    let path = write(&d, "bake_inherit");
    let mut baked = dense();
    baked.animation.bake_animations = true;
    let a = crate::load_fbx_with(&path, &dense()).unwrap();
    let b = crate::load_fbx_with(&path, &baked).unwrap();
    let (mut plain_err, mut baked_err) = (0.0f32, 0.0f32);
    for t in [0.0f32, 0.5, 0.9999] {
        let truth = ufbx_worlds(&path, 0, t as f64)["Tip"].transform_point3(Vec3::ZERO);
        plain_err = plain_err.max((joint_worlds(&a.skeletons[0], &a.animations[0], t)["Tip"].transform_point3(Vec3::ZERO) - truth).length());
        baked_err = baked_err.max((joint_worlds(&b.skeletons[0], &b.animations[0], t)["Tip"].transform_point3(Vec3::ZERO) - truth).length());
    }
    cleanup(&path);
    assert!(baked_err < 1e-3, "baked: {baked_err}");
    assert!(plain_err < 1e-3, "plain: {plain_err}");
}

#[test]
fn baked_constraints_move_the_constrained_bone() {
    let mut d = Doc::new();
    let root = d.bone("Root", "", 0);
    let target = d.bone("Target", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 0.0, 0.0]), root);
    let follower = d.bone("Follower", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 3.0, 0.0]), root);
    d.stack("Take", (0.0, 1.0), &[(target, "Lcl Translation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 10.0)])])]);
    d.constraint("Position From Positions", follower, &[("Target", target)], "");
    let path = write(&d, "bake_pos");
    let mut m = dense();
    m.animation.bake_animations = true;
    let baked = crate::load_fbx_with(&path, &m).unwrap();
    let plain = crate::load_fbx_with(&path, &dense()).unwrap();
    let at = |s: &Scene, t: f32| pos_of(&joint_worlds(&s.skeletons[0], &s.animations[0], t)["Follower"]);
    for (t, x) in [(0.0f32, 0.0f32), (0.5, 5.0), (0.9999, 10.0)] {
        // A position constraint sets the world position outright.
        assert!((at(&baked, t) - Vec3::new(x, 0.0, 0.0)).length() < 1e-3, "t={t}: {:?}", at(&baked, t));
        assert!((at(&plain, t) - Vec3::new(0.0, 3.0, 0.0)).length() < 1e-3, "without Bake the follower ignores the constraint");
    }
    cleanup(&path);

    // Half weight meets halfway between the follower's own place and the target.
    let mut d = Doc::new();
    let root = d.bone("Root", "", 0);
    let target = d.bone("Target", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 0.0, 0.0]), root);
    let follower = d.bone("Follower", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 3.0, 0.0]), root);
    d.stack("Take", (0.0, 1.0), &[(target, "Lcl Translation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 10.0)])])]);
    d.constraint("Position From Positions", follower, &[("Target", target)], "P: \"Weight\", \"Number\", \"\", \"A\",50");
    let path = write(&d, "bake_pos_half");
    let half = crate::load_fbx_with(&path, &m).unwrap();
    assert!((at(&half, 0.9999) - Vec3::new(5.0, 1.5, 0.0)).length() < 1e-2, "{:?}", at(&half, 0.9999));
    cleanup(&path);
}

#[test]
fn baked_aim_constraint_turns_the_bone_towards_its_target() {
    let mut d = Doc::new();
    let root = d.bone("Root", "", 0);
    let target = d.bone("Target", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 0.0, 0.0]), root);
    let follower = d.bone("Follower", &vec_prop("Lcl Translation", "Lcl Translation", "A", [0.0, 3.0, 0.0]), root);
    d.stack("Take", (0.0, 1.0), &[(target, "Lcl Translation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 10.0)])])]);
    d.constraint("Aim", follower, &[("Target", target)], "");
    let path = write(&d, "bake_aim");
    let mut m = dense();
    m.animation.bake_animations = true;
    let s = crate::load_fbx_with(&path, &m).unwrap();
    cleanup(&path);
    for t in [0.5f32, 0.9999] {
        let w = joint_worlds(&s.skeletons[0], &s.animations[0], t)["Follower"];
        let dir = (w.transform_vector3(Vec3::X)).normalize();
        let want = (Vec3::new(10.0 * t, 0.0, 0.0) - Vec3::new(0.0, 3.0, 0.0)).normalize();
        assert!((dir - want).length() < 1e-3, "t={t}: {dir:?} vs {want:?}");
    }
}

#[test]
fn constraints_are_read_into_specs_when_asked() {
    let mut d = Doc::new();
    let root = d.bone("Root", "", 0);
    let a = d.bone("A", "", root);
    let b = d.bone("B", "", root);
    let c = d.bone("C", "", root);
    d.stack("Take", (0.0, 1.0), &[(a, "Lcl Translation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 1.0)])])]);
    d.constraint("Parent-Child", c, &[("A", a), ("B", b)], "P: \"Weight\", \"Number\", \"\", \"A\",80\nP: \"A.Offset T\", \"Vector3D\", \"Vector\", \"\",0,1,0\nP: \"AffectScaleY\", \"bool\", \"\", \"\",1");
    d.constraint("Aim", c, &[("B", b)], "P: \"WorldUpType\", \"int\", \"Integer\", \"\",3\nP: \"UpVector\", \"Vector3D\", \"Vector\", \"\",0,0,1");
    d.constraint("Single Chain IK", c, &[], "");
    let on = ModelImport { import_constraints: true, ..Default::default() };
    let s = import(&d, "constraints", &on).unwrap();
    let specs = &s.identity.constraints;
    assert_eq!(specs.len(), 2, "{specs:?}");
    let parent = specs.iter().find(|c| c.kind == ConstraintKind::Parent).unwrap();
    assert_eq!(parent.node, "C");
    assert!((parent.weight - 0.8).abs() < 1e-5);
    assert_eq!(parent.sources.iter().map(|s| s.node.as_str()).collect::<Vec<_>>(), ["A", "B"]);
    assert!((parent.sources[0].weight - 1.0).abs() < 1e-5);
    assert!((parent.sources[0].offset.t() - Vec3::Y).length() < 1e-5, "the per-source offset comes through");
    assert!(parent.scale_axes[1] && !parent.scale_axes[0]);
    let aim = specs.iter().find(|c| c.kind == ConstraintKind::Aim).unwrap();
    assert_eq!(aim.aim_up, zestengine_core::AimUp::Vector);
    assert_eq!(aim.up_vector, [0.0, 0.0, 1.0]);
    assert!(s.identity.import_warnings.iter().any(|w| w.contains("constraint")), "{:?}", s.identity.import_warnings);
    // Off by default.
    assert!(import(&d, "constraints_off", &ModelImport::default()).unwrap().identity.constraints.is_empty());
}

#[test]
fn animated_user_properties_become_property_tracks() {
    let mut d = Doc::new();
    let root = d.bone(
        "Root",
        &props(&[
            "P: \"Speed\", \"Number\", \"\", \"A+U\",1.5".to_string(),
            "P: \"Count\", \"int\", \"Integer\", \"A+U\",2".to_string(),
            "P: \"On\", \"bool\", \"\", \"A+U\",1".to_string(),
            "P: \"Dir\", \"Vector3D\", \"Vector\", \"A+U\",1,2,3".to_string(),
            "P: \"Still\", \"Number\", \"\", \"A+U\",7".to_string(),
        ]),
        0,
    );
    d.stack(
        "Take",
        (0.0, 2.0),
        &[
            (root, "Speed", vec![Chan::linear("d|Speed", &[(0.0, 1.0), (2.0, 5.0)])]),
            (root, "Count", vec![Chan { name: "d|Count", keys: vec![(0.0, 2.0), (1.0, 4.0), (2.0, 4.0)], flags: 2 }]),
            (root, "Dir", vec![Chan::linear("d|X", &[(0.0, 1.0), (2.0, 3.0)])]),
        ],
    );
    let mut m = dense();
    m.animation.animated_custom_properties = true;
    let s = import(&d, "custom", &m).unwrap();
    let tracks = &s.identity.clip_info[0].morph_tracks;
    let find = |n: &str| tracks.iter().find(|t| t.property == TrackProperty::CustomProperty && t.name == n);
    let speed = find("Speed").expect("float track");
    assert_eq!(speed.keys.len(), 2);
    assert!((speed.sample(1.0)[0] - 3.0).abs() < 1e-4, "linear between the keys");
    let count = find("Count").expect("int track");
    assert!(count.keys.iter().all(|k| k.interp == crate::anim_clip::KeyInterp::Step));
    assert!(find("Dir.x").is_some(), "vector lanes are named Dir.x / .y / .z");
    assert!(find("Still").is_none(), "a property without curves gets no track");
    assert_eq!(s.animations[0].morph_tracks.len(), tracks.len(), "the clip carries them too");
    // Initial values land in the identity for the component.
    let (node, vals) = s.identity.custom_props.iter().find(|(n, _)| n == "Root").unwrap();
    assert_eq!(node, "Root");
    let get = |n: &str| vals.iter().find(|(k, _)| k == n).map(|(_, v)| *v);
    assert_eq!(get("Still"), Some(CustomValue::Float(7.0)));
    assert_eq!(get("Count"), Some(CustomValue::Int(2)));
    assert_eq!(get("On"), Some(CustomValue::Bool(true)));
    assert_eq!(get("Dir"), Some(CustomValue::Vec3([1.0, 2.0, 3.0])));
    // Off: nothing.
    let off = import(&d, "custom_off", &dense()).unwrap();
    assert!(off.identity.clip_info[0].morph_tracks.is_empty() && off.identity.custom_props.is_empty());
}

/// Local TRS of the file's node `name` at `t`, straight from ufbx's per-node evaluation.
fn ufbx_local(path: &Path, name: &str, t: f64) -> (Vec3, Quat) {
    let fbx = load_ufbx(path);
    let node = fbx.nodes.iter().find(|n| n.element.name.as_ref() == name).unwrap();
    let tr = node.evaluate_transform(&fbx.anim_stacks[0].anim, t);
    (Vec3::new(tr.translation.x as f32, tr.translation.y as f32, tr.translation.z as f32), Quat::from_xyzw(tr.rotation.x as f32, tr.rotation.y as f32, tr.rotation.z as f32, tr.rotation.w as f32))
}

fn sparse_cubic() -> Doc {
    let mut d = Doc::new();
    let root = d.bone(
        "Root",
        &props(&[
            vec_prop("PreRotation", "Vector3D", "", [10.0, 20.0, 30.0]),
            "P: \"RotationOrder\", \"enum\", \"\", \"\",5".to_string(),
        ]),
        0,
    );
    let k = |a: f32, b: f32, c: f32, e: f32| vec![(0.0, a), (0.7, b), (1.4, c), (2.0, e)];
    d.stack(
        "Take",
        (0.0, 2.0),
        &[
            (root, "Lcl Rotation", vec![Chan::cubic("d|X", &k(0.0, 80.0, -40.0, 20.0)), Chan::cubic("d|Y", &k(0.0, 30.0, 60.0, -10.0)), Chan::cubic("d|Z", &k(10.0, -50.0, 40.0, 0.0))]),
            (root, "Lcl Translation", vec![Chan::cubic("d|X", &k(0.0, 4.0, -2.0, 1.0)), Chan::cubic("d|Y", &k(1.0, 1.0, 5.0, 2.0))]),
        ],
    );
    d
}

#[test]
fn keep_tangents_follow_the_source_curve_with_few_keys() {
    let d = sparse_cubic();
    let path = write(&d, "tangents");
    let mut keep = dense();
    keep.animation.keep_tangents = true;
    keep.animation.rotation_error = 0.5;
    keep.animation.position_error = 0.5;
    let mut linear = dense();
    linear.animation.resample_curves = false;
    let kept = crate::load_fbx_with(&path, &keep).unwrap();
    let authored = crate::load_fbx_with(&path, &linear).unwrap();
    let dense_scene = crate::load_fbx_with(&path, &dense()).unwrap();
    let clip = &kept.animations[0];
    assert!(clip.channels.iter().any(|c| c.tangents().is_some()), "channels carry tangents");
    let rot = clip.channels.iter().find(|c| c.path == ChannelPath::Rotation).unwrap();
    let dense_rot = dense_scene.animations[0].channels.iter().find(|c| c.path == ChannelPath::Rotation).unwrap();
    assert!(rot.len() * 2 < dense_rot.len(), "{} keys vs {} dense", rot.len(), dense_rot.len());
    let worst = |scene: &Scene| {
        let (mut rot_err, mut pos_err) = (0.0f32, 0.0f32);
        let c = &scene.animations[0];
        for k in 0..=100 {
            let t = 1.9999 * k as f32 / 100.0;
            let (want_t, want_r) = ufbx_local(&path, "Root", t as f64);
            let locals = c.sample(&scene.skeletons[0], t);
            let (_, r, tr) = locals[0].to_scale_rotation_translation();
            rot_err = rot_err.max(2.0 * r.dot(want_r).abs().clamp(0.0, 1.0).acos().to_degrees());
            pos_err = pos_err.max((tr - want_t).length());
        }
        (rot_err, pos_err)
    };
    let (kr, kp) = worst(&kept);
    let (ar, ap) = worst(&authored);
    let (dr, _) = worst(&dense_scene);
    cleanup(&path);
    println!("keep tangents: rot {kr:.3} deg pos {kp:.4} ({} keys); authored linear keys: rot {ar:.3} deg pos {ap:.4}; dense 30 fps: rot {dr:.3} deg", rot.len());
    assert!(kr < 0.75, "kept curves stay within the rotation bound of ufbx's evaluation: {kr} deg");
    assert!(kp < 0.06, "and the position bound: {kp}");
    assert!(ar > kr * 2.0, "straight lines between the authored keys are much worse ({ar} vs {kr})");
}

#[test]
fn keep_tangents_make_translation_exact_on_a_plain_chain() {
    // No pre-rotation, XYZ order: translation and scale are the source curves themselves.
    let mut d = Doc::new();
    let root = d.bone("Root", "", 0);
    d.stack("Take", (0.0, 2.0), &[(root, "Lcl Translation", vec![Chan::cubic("d|X", &[(0.0, 0.0), (1.0, 8.0), (2.0, 3.0)]), Chan::cubic("d|Z", &[(0.0, 1.0), (1.0, -4.0), (2.0, 2.0)])])]);
    let path = write(&d, "tangent_plain");
    let mut keep = dense();
    keep.animation.keep_tangents = true;
    let s = crate::load_fbx_with(&path, &keep).unwrap();
    let tch = s.animations[0].channels.iter().find(|c| c.path == ChannelPath::Translation).unwrap();
    assert!(tch.len() <= 6, "authored keys (plus a few refinements): {}", tch.len());
    let mut worst = 0.0f32;
    for k in 0..=100 {
        let t = 1.9999 * k as f32 / 100.0;
        let (want, _) = ufbx_local(&path, "Root", t as f64);
        worst = worst.max((tch.sample_vec(t).unwrap() - want).length());
    }
    cleanup(&path);
    assert!(worst < 0.05, "{worst}");
}

#[test]
fn compression_modes_round_trip_through_the_cache() {
    let d = sparse_cubic();
    let path = write(&d, "cache_modes");
    for mode in [AnimCompression::Off, AnimCompression::KeyframeReduction, AnimCompression::KeyframeReductionAndCompression, AnimCompression::Optimal] {
        for keep in [false, true] {
            let mut m = ModelImport::default();
            m.animation.compression = mode;
            m.animation.keep_tangents = keep;
            let s = crate::load_fbx_with(&path, &m).unwrap();
            let bytes = crate::import_cache::encode(&s, &path, &[], "");
            let back = crate::import_cache::decode(&bytes, &path).unwrap();
            for (a, b) in s.animations[0].channels.iter().zip(&back.animations[0].channels) {
                assert_eq!((a.joint, a.path, &a.times), (b.joint, b.path, &b.times));
                assert_eq!(a.extra, b.extra, "{mode:?} keep={keep}");
                assert_eq!((a.vec_values.len(), a.quat_values.len()), (b.vec_values.len(), b.quat_values.len()));
                for t in [0.0f32, 0.3, 1.1, 2.0] {
                    match a.path {
                        ChannelPath::Rotation => assert!(a.sample_quat(t).unwrap().dot(b.sample_quat(t).unwrap()) > 0.99999),
                        _ => assert!((a.sample_vec(t).unwrap() - b.sample_vec(t).unwrap()).length() < 1e-6),
                    }
                }
            }
        }
    }
    cleanup(&path);
}

#[test]
fn old_meta_files_load_with_defaults() {
    // A model meta written before any of these settings existed.
    let old = "(scale: 1.0, animation: (resample_curves: true, compression: Optimal, rotation_error: 0.5))";
    let m: ModelImport = ron::from_str(old).unwrap();
    assert!(!m.animation.bake_animations && !m.animation.keep_tangents && !m.animation.animated_custom_properties);
    assert_eq!(m.animation.humanoid_oversampling, 1);
    assert!(!m.import_constraints && !m.optimize_game_objects);
    assert_eq!(m.max_bones_per_vertex, 4);
    assert!(m.extra_expose_transforms.is_empty());
    let mut m = m;
    m.max_bones_per_vertex = 9;
    m.animation.humanoid_oversampling = 5;
    m.extra_expose_transforms = vec!["Hand".into(), " ".into(), "Hand".into()];
    m.validate();
    assert_eq!((m.max_bones_per_vertex, m.animation.humanoid_oversampling), (4, 4));
    assert_eq!(m.extra_expose_transforms, vec!["Hand".to_string()]);
}

// ----------------------------------------------------------------- real files

fn mixamo_files() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else { return Vec::new() };
    let dir = PathBuf::from(home).join("Downloads/maximo");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "fbx")).collect()).unwrap_or_default();
    v.sort();
    v
}

/// The worst deviation of `b`'s channels from `a`'s (dense) at the dense key times:
/// (degrees, share of each translation channel's range in percent).
fn clip_error(a: &AnimationClip, b: &AnimationClip) -> (f32, f32) {
    let (mut deg, mut pct) = (0.0f32, 0.0f32);
    for (x, y) in a.channels.iter().zip(&b.channels) {
        assert_eq!((x.joint, x.path), (y.joint, y.path));
        match x.path {
            ChannelPath::Rotation => {
                for (i, t) in x.times.iter().enumerate() {
                    let q = y.sample_quat(*t).unwrap();
                    deg = deg.max(2.0 * q.dot(x.quat_at(i).unwrap()).abs().clamp(0.0, 1.0).acos().to_degrees());
                }
            }
            ChannelPath::Translation => {
                let vals: Vec<Vec3> = (0..x.len()).map(|i| x.vec_at(i).unwrap()).collect();
                let (lo, hi) = vals.iter().fold((vals[0], vals[0]), |(l, h), v| (l.min(*v), h.max(*v)));
                let range = (hi - lo).max_element().max(1e-3);
                for (i, t) in x.times.iter().enumerate() {
                    pct = pct.max((y.sample_vec(*t).unwrap() - vals[i]).abs().max_element() / range * 100.0);
                }
            }
            ChannelPath::Scale => {}
        }
    }
    (deg, pct)
}

#[test]
fn mixamo_compression_sizes_and_error_bounds() {
    let files = mixamo_files();
    if files.is_empty() {
        return;
    }
    let mut table = Vec::new();
    let (mut total_dense, mut total_kr, mut total_krc, mut total_opt) = (0usize, 0usize, 0usize, 0usize);
    for f in &files {
        let load = |mode: AnimCompression| {
            let mut m = ModelImport::default();
            m.animation.compression = mode;
            crate::load_fbx_with(f, &m).unwrap()
        };
        let off = load(AnimCompression::Off);
        let size = |s: &Scene| {
            let clips = Scene {
                meshes: Vec::new(),
                morphs: Vec::new(),
                springs: Default::default(),
                textures: Vec::new(),
                materials: Vec::new(),
                instances: Vec::new(),
                skeletons: Vec::new(),
                animations: s.animations.clone(),
                collision: Vec::new(),
                identity: Default::default(),
            };
            crate::import_cache::encode(&clips, f, &[], "").len()
        };
        let (kr, krc, opt) = (load(AnimCompression::KeyframeReduction), load(AnimCompression::KeyframeReductionAndCompression), load(AnimCompression::Optimal));
        for (mode, s) in [("Keyframes", &kr), ("+Compress", &krc), ("Optimal", &opt)] {
            for (a, b) in off.animations.iter().zip(&s.animations) {
                let (deg, pct) = clip_error(a, b);
                assert!(deg <= 0.55, "{mode} {}: {deg} deg", f.display());
                assert!(pct <= 0.6, "{mode} {}: {pct} %", f.display());
            }
        }
        let (d, a, b, c) = (size(&off), size(&kr), size(&krc), size(&opt));
        assert!(c <= b, "{}: Optimal {c} must not exceed Keyframes + Compress {b}", f.display());
        assert!(a < d, "{}: reduction shrinks the clip", f.display());
        table.push(format!("{:<28} dense {d:>9}  keyframes {a:>8}  +compress {b:>8}  optimal {c:>8}", f.file_name().unwrap().to_string_lossy()));
        total_dense += d;
        total_kr += a;
        total_krc += b;
        total_opt += c;
    }
    println!("{}", table.join("\n"));
    println!(
        "TOTAL {} files: dense {total_dense} B, keyframes {total_kr} B ({:.1}%), +compress {total_krc} B ({:.1}%), optimal {total_opt} B ({:.1}%)",
        files.len(),
        total_kr as f64 / total_dense as f64 * 100.0,
        total_krc as f64 / total_dense as f64 * 100.0,
        total_opt as f64 / total_dense as f64 * 100.0
    );
    assert!(total_opt <= total_krc);
}

#[test]
fn mixamo_bake_matches_the_plain_evaluation() {
    // Mixamo bones have PreRotation but no pivots or constraints: baking changes nothing visible.
    let files = mixamo_files();
    let Some(f) = files.iter().find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().contains("Zombie Idle"))).or(files.first()) else { return };
    let plain = crate::load_fbx_with(f, &dense()).unwrap();
    let mut b = dense();
    b.animation.bake_animations = true;
    let baked = crate::load_fbx_with(f, &b).unwrap();
    let (cp, cb) = (&plain.animations[0], &baked.animations[0]);
    let mut worst = 0.0f32;
    for k in 0..=20 {
        let t = cp.duration * k as f32 / 20.0;
        let (wp, wb) = (joint_worlds(&plain.skeletons[0], cp, t), joint_worlds(&baked.skeletons[0], cb, t));
        for (name, m) in &wp {
            worst = worst.max((pos_of(m) - pos_of(&wb[name])).length());
        }
    }
    assert!(worst < 0.01, "{worst}");
}

#[test]
fn mixamo_keep_tangents_stay_within_the_bounds_and_shrink_the_clip() {
    let files = mixamo_files();
    let Some(f) = files.iter().find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().contains("Zombie Walking.fbx"))).or(files.first()) else { return };
    let reference = crate::load_fbx_with(f, &dense()).unwrap();
    let mut k = dense();
    k.animation.keep_tangents = true;
    let kept = crate::load_fbx_with(f, &k).unwrap();
    let (cr, ck) = (&reference.animations[0], &kept.animations[0]);
    let keys = |c: &AnimationClip| c.channels.iter().map(|c| c.len()).sum::<usize>();
    let (mut deg, mut pos) = (0.0f32, 0.0f32);
    for (a, b) in cr.channels.iter().zip(&ck.channels) {
        // Channels may be keyed differently; compare joint by joint through the same joint order.
        assert_eq!((a.joint, a.path), (b.joint, b.path));
        for (i, t) in a.times.iter().enumerate() {
            match a.path {
                ChannelPath::Rotation => deg = deg.max(2.0 * b.sample_quat(*t).unwrap().dot(a.quat_at(i).unwrap()).abs().clamp(0.0, 1.0).acos().to_degrees()),
                ChannelPath::Translation => pos = pos.max((b.sample_vec(*t).unwrap() - a.vec_at(i).unwrap()).length()),
                ChannelPath::Scale => {}
            }
        }
    }
    println!("{}: dense {} keys, kept tangents {} keys, worst rot {deg:.3} deg, worst pos {pos:.4}", f.display(), keys(cr), keys(ck));
    assert!(deg < 1.5, "{deg}");
    assert!(keys(ck) < keys(cr));
}

#[test]
#[allow(clippy::field_reassign_with_default)] // import settings written field by field
fn real_files_import_with_bake_and_extras_when_present() {
    // The author's Blender / DCC files: import with every extra on; nothing may fail or produce NaNs.
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    for rel in ["shared/avi/lemonnew.fbx"] {
        let p = home.join(rel);
        if !p.exists() {
            continue;
        }
        let mut m = ModelImport::default();
        m.import_constraints = true;
        m.animation.bake_animations = true;
        m.animation.animated_custom_properties = true;
        m.animation.keep_tangents = true;
        let s = crate::load_fbx_with(&p, &m).unwrap();
        for c in &s.animations {
            assert!(c.channels.iter().all(|ch| (0..ch.len()).all(|i| ch.vec_at(i).is_none_or(|v| v.is_finite()) && ch.quat_at(i).is_none_or(|q| q.is_finite()))));
        }
        println!("{rel}: {} meshes, {} clips, {} constraints, {} warnings", s.meshes.len(), s.animations.len(), s.identity.constraints.len(), s.identity.import_warnings.len());
    }
}


/// The FBX texture UV transform (Blender's mapping-node scale/location) must
/// land as the material's tiling/offset. Verified against ufbx's own reading
/// of the author's real Blender export (skipped when the file is absent), so
/// the expected values are whatever the exporter wrote - a round-trip check.
#[test]
fn texture_uv_transform_imports_as_tiling_and_offset() {
    let path = std::path::PathBuf::from("/home/lemonxah/ZestEngineProjects/LongDark/models/LongDark1.fbx");
    if !path.is_file() {
        eprintln!("[test] LongDark1.fbx not present: skipped");
        return;
    }
    let fbx = load_ufbx(&path);
    let scene = crate::load_fbx(&path).expect("author's FBX loads");
    let mut checked = 0;
    for (m, t) in fbx.materials.iter().zip(scene.materials.iter()) {
        for (map, got_tiling, got_offset) in [
            (&m.pbr.base_color, t.params.albedo_tiling, t.params.albedo_offset),
            (&m.pbr.normal_map, t.params.normal_tiling, t.params.normal_offset),
            (&m.pbr.emission_color, t.params.emission_tiling, t.params.emission_offset),
        ] {
            let Some(tex) = map.texture.as_ref().filter(|x| x.has_uv_transform) else { continue };
            assert_eq!(
                got_tiling,
                [tex.uv_transform.scale.x as f32, tex.uv_transform.scale.y as f32],
                "{}: tiling matches ufbx's uv scale",
                m.element.name,
            );
            assert_eq!(
                got_offset,
                [tex.uv_transform.translation.x as f32, tex.uv_transform.translation.y as f32],
                "{}: offset matches ufbx's uv translation",
                m.element.name,
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "the author's FBX carries at least one texture UV transform to check");
}
