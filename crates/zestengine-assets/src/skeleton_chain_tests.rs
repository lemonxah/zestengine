//! Skeleton ancestor chain (the non-bone nodes above the topmost bone, such as
//! an `Armature` empty): real-file survey and synthesised-file tests.

#[allow(dead_code)]
fn survey(path: &str) -> Option<String> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return None;
    }
    let scene = crate::load_model_with_meta(p).ok()?;
    let skel = scene.skeletons.first()?;
    let mask = crate::rig_import::real_bone_mask(skel, &scene.meshes);
    let map = crate::auto_map::auto_map_with(skel, Some(&mask));
    let human = map.to_human(skel);
    let issues = crate::human_rig::validate_rig(skel, &human);
    let roots: Vec<&str> = skel.joints.iter().filter(|j| j.parent.is_none()).map(|j| j.name.as_str()).collect();
    let hips = human.iter().find(|(b, _)| *b == crate::human_rig::HumanBone::Hips).map(|(_, n)| n.clone());
    Some(format!(
        "{path}: joints {} containers {} roots {roots:?} mapped {} required {} hips {hips:?} issues {:?}",
        skel.joints.len(),
        skel.containers.len(),
        human.len(),
        map.required_count(),
        issues.iter().map(|i| i.message.clone()).collect::<Vec<_>>()
    ))
}

#[test]
fn real_files_survey() {
    for f in [
        "/home/lemonxah/Downloads/maximo/T-Pose.fbx",
        "/home/lemonxah/shared/Nadia/Vivixx Vtube.vrm",
        "/home/lemonxah/shared/avi/lemonnew.fbx",
        "/home/lemonxah/storage/blender/lemonbase.fbx",
    ] {
        match survey(f) {
            Some(s) => {
                eprintln!("SURVEY {s}");
                // The Blender files hang the bones off an `Armature` node: with it the avatar has no issue.
                if std::path::Path::new(f).file_name().is_some_and(|n| n.to_string_lossy().starts_with("lemon")) {
                    assert!(s.contains("containers 1") && s.contains("issues []") && s.contains("mapped 53"), "{s}");
                }
            }
            None => eprintln!("SURVEY {f}: absent or not skinned"),
        }
    }
}


use crate::fbx_test_util::{Chan, Doc, import, vec_prop};
use crate::human_rig::{HumanBone, validate_rig};
use crate::{ModelImport, Scene, Skeleton};
use glam::{EulerRot, Mat4, Quat, Vec3};

/// Mixamo-named humanoid: (name, parent, offset as (side, up, forward) in metres).
const BONES: &[(&str, Option<usize>, [f32; 3])] = &[
    ("mixamorig:Hips", None, [0.0, 1.0, 0.0]),
    ("mixamorig:Spine", Some(0), [0.0, 0.1, 0.0]),
    ("mixamorig:Spine1", Some(1), [0.0, 0.15, 0.0]),
    ("mixamorig:Spine2", Some(2), [0.0, 0.15, 0.0]),
    ("mixamorig:Neck", Some(3), [0.0, 0.15, 0.0]),
    ("mixamorig:Head", Some(4), [0.0, 0.1, 0.0]),
    ("mixamorig:LeftShoulder", Some(3), [0.05, 0.1, 0.0]),
    ("mixamorig:LeftArm", Some(6), [0.1, 0.0, 0.0]),
    ("mixamorig:LeftForeArm", Some(7), [0.28, 0.0, 0.0]),
    ("mixamorig:LeftHand", Some(8), [0.25, 0.0, 0.0]),
    ("mixamorig:RightShoulder", Some(3), [-0.05, 0.1, 0.0]),
    ("mixamorig:RightArm", Some(10), [-0.1, 0.0, 0.0]),
    ("mixamorig:RightForeArm", Some(11), [-0.28, 0.0, 0.0]),
    ("mixamorig:RightHand", Some(12), [-0.25, 0.0, 0.0]),
    ("mixamorig:LeftUpLeg", Some(0), [0.1, -0.05, 0.0]),
    ("mixamorig:LeftLeg", Some(14), [0.0, -0.45, 0.0]),
    ("mixamorig:LeftFoot", Some(15), [0.0, -0.42, 0.0]),
    ("mixamorig:LeftToeBase", Some(16), [0.0, -0.05, 0.12]),
    ("mixamorig:RightUpLeg", Some(0), [-0.1, -0.05, 0.0]),
    ("mixamorig:RightLeg", Some(18), [0.0, -0.45, 0.0]),
    ("mixamorig:RightFoot", Some(19), [0.0, -0.42, 0.0]),
    ("mixamorig:RightToeBase", Some(20), [0.0, -0.05, 0.12]),
];

/// What the synthesised file looks like.
#[derive(Clone, Copy)]
struct Variant {
    /// `UnitScaleFactor`: 100 = metres, 1 = centimetres (the root carries a 0.01 scale).
    unit_scale: f64,
    /// The file is Z-up.
    z_up_file: bool,
    /// The bones are modelled along +Z in the armature's frame (Blender) instead of +Y.
    bones_z_up: bool,
    /// The `Armature` empty's local rotation (degrees, XYZ) and translation, in file units.
    armature: Option<([f32; 3], [f32; 3])>,
    animate_hips: bool,
    animate_armature: bool,
    /// The mesh node carries the Armature's transform and its vertices are in its local space, the
    /// cluster's `Transform` is the mesh's bind matrix (what Blender writes).
    blender_mesh: bool,
}

impl Variant {
    fn k(self) -> f32 {
        if self.unit_scale >= 50.0 { 1.0 } else { 100.0 }
    }
}

fn euler_deg(r: [f32; 3]) -> Quat {
    Quat::from_euler(EulerRot::ZYX, r[2].to_radians(), r[1].to_radians(), r[0].to_radians())
}

fn mat_text(m: Mat4) -> String {
    m.to_cols_array().iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(",")
}

/// A skinned humanoid mesh under (optionally) an `Armature` empty.
fn skinned_doc(v: Variant) -> Doc {
    let k = v.k();
    let mut doc = Doc::new();
    doc.unit_scale = v.unit_scale;
    doc.z_up = v.z_up_file;
    let (parent, base) = match v.armature {
        Some((rot, t)) => {
            let props = format!("{}\n{}", vec_prop("Lcl Translation", "Lcl Translation", "A", t), vec_prop("Lcl Rotation", "Lcl Rotation", "A", rot));
            (doc.model("Armature", "Null", &props, 0), Mat4::from_rotation_translation(euler_deg(rot), Vec3::from(t)))
        }
        None => (0, Mat4::IDENTITY),
    };
    let mut ids = Vec::new();
    let mut worlds: Vec<Mat4> = Vec::new();
    for (name, p, off) in BONES {
        let o = if v.bones_z_up { Vec3::new(off[0], -off[2], off[1]) } else { Vec3::new(off[0], off[1], off[2]) } * k;
        let props = vec_prop("Lcl Translation", "Lcl Translation", "A", o.to_array());
        let id = doc.bone(name, &props, p.map_or(parent, |p| ids[p]));
        worlds.push(p.map_or(base, |p| worlds[p]) * Mat4::from_translation(o));
        ids.push(id);
    }
    // One vertex per bone, a little off the bone; weights 0.7 / 0.3 between a bone and the next.
    let n = BONES.len();
    let pos: Vec<Vec3> = worlds.iter().map(|w| w.transform_point3(Vec3::new(0.01, 0.02, 0.03) * k)).collect();
    let pos: Vec<Vec3> = if v.blender_mesh { pos.iter().map(|p| base.inverse().transform_point3(*p)).collect() } else { pos };
    let verts: Vec<String> = pos.iter().map(|p| format!("{},{},{}", p.x, p.y, p.z)).collect();
    let tris: Vec<String> = (0..n - 2).map(|i| format!("{},{},{}", i, i + 1, -((i + 2) as i32) - 1)).collect();
    let geom = doc.id();
    let mesh_props = match v.armature {
        Some((rot, t)) if v.blender_mesh => format!("{}\n{}", vec_prop("Lcl Translation", "Lcl Translation", "A", t), vec_prop("Lcl Rotation", "Lcl Rotation", "A", rot)),
        _ => String::new(),
    };
    let mesh = doc.model("Body", "Mesh", &mesh_props, 0);
    doc.raw_object(&format!(
        "\tGeometry: {geom}, \"Geometry::Body\", \"Mesh\" {{\n\t\tVertices: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tPolygonVertexIndex: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tGeometryVersion: 124\n\t}}\n",
        n * 3,
        verts.join(","),
        (n - 2) * 3,
        tris.join(",")
    ));
    doc.conn(&format!("C: \"OO\",{geom},{mesh}"));
    let skin = doc.id();
    doc.raw_object(&format!("\tDeformer: {skin}, \"Deformer::Skin\", \"Skin\" {{\n\t\tVersion: 101\n\t\tLink_DeformAcuracy: 50\n\t}}\n"));
    doc.conn(&format!("C: \"OO\",{skin},{geom}"));
    for (b, id) in ids.iter().enumerate() {
        let mut idx = vec![b];
        let mut wts = vec![0.7f32];
        idx.push((b + n - 1) % n);
        wts.push(0.3);
        let cl = doc.id();
        let (idx_s, wts_s): (Vec<String>, Vec<String>) = (idx.iter().map(|i| i.to_string()).collect(), wts.iter().map(|w| w.to_string()).collect());
        doc.raw_object(&format!(
            "\tDeformer: {cl}, \"SubDeformer::\", \"Cluster\" {{\n\t\tVersion: 100\n\t\tUserData: \"\", \"\"\n\t\tIndexes: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tWeights: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tTransform: *16 {{\n\t\t\ta: {}\n\t\t}}\n\t\tTransformLink: *16 {{\n\t\t\ta: {}\n\t\t}}\n\t}}\n",
            idx.len(),
            idx_s.join(","),
            wts.len(),
            wts_s.join(","),
            mat_text(if v.blender_mesh { Mat4::IDENTITY } else { worlds[b].inverse() }),
            mat_text(worlds[b])
        ));
        doc.conn(&format!("C: \"OO\",{cl},{skin}"));
        doc.conn(&format!("C: \"OO\",{id},{cl}"));
    }
    // Vertex b is skinned (0.7) to cluster b and (0.3) to cluster b-1: the second index list
    // above names vertex (b-1), so give each cluster the vertex of its own bone and the next one's.
    let mut curves = Vec::new();
    if v.animate_hips {
        curves.push((ids[0], "Lcl Rotation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 25.0)]), Chan::linear("d|Y", &[(0.0, 0.0), (1.0, 40.0)])]));
        curves.push((ids[3], "Lcl Rotation", vec![Chan::linear("d|Z", &[(0.0, 0.0), (1.0, 30.0)])]));
    }
    if v.animate_armature && parent != 0 {
        curves.push((parent, "Lcl Translation", vec![Chan::linear("d|X", &[(0.0, 0.0), (1.0, 0.5 * k)])]));
    }
    if !curves.is_empty() {
        doc.stack("Take 001", (0.0, 1.0), &curves);
    }
    doc
}

fn joint_of(skel: &Skeleton, name: &str) -> usize {
    skel.joints.iter().position(|j| j.name == name).unwrap_or_else(|| panic!("no joint {name}"))
}

/// The skeleton as it was before the container joints: the bones only, the topmost one parentless.
fn stripped(skel: &Skeleton) -> Skeleton {
    let n = skel.containers.iter().copied().min().unwrap_or(skel.joints.len());
    let mut s = skel.clone();
    s.joints.truncate(n);
    for j in &mut s.joints {
        j.parent = j.parent.filter(|p| *p < n);
    }
    s.containers.clear();
    s
}

/// Every skinned vertex of the scene, posed by `palette`, in world space (`Instance.transform` applied).
fn posed(scene: &Scene, palette: &[Mat4], instance_scale: Mat4) -> Vec<Vec3> {
    let mut out = Vec::new();
    for inst in &scene.instances {
        for slot in &inst.slots {
            for v in &scene.meshes[slot.mesh].vertices {
                let mut m = Mat4::ZERO;
                for (j, w) in v.joints.iter().zip(v.weights) {
                    if w > 0.0 {
                        m += palette[*j as usize] * w;
                    }
                }
                out.push((inst.transform * instance_scale).transform_point3(m.transform_point3(Vec3::from(v.position))));
            }
        }
    }
    out
}

fn assert_same(a: &[Vec3], b: &[Vec3], what: &str) {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let tol = 1e-4 * x.length().max(1.0);
        assert!((*x - *y).length() <= tol, "{what}: vertex {i}: {x:?} vs {y:?}");
    }
}

/// The checks every variant shares: the container, the pose before / after, the clips and the humanoid map.
fn check(v: Variant, tag: &str) {
    let scene = import(&skinned_doc(v), tag, &ModelImport::default()).unwrap();
    let skel = &scene.skeletons[0];
    let armature = v.armature.is_some();
    assert_eq!(skel.containers.len(), armature as usize, "{tag}: containers {:?}", skel.containers);
    let old = stripped(skel);
    let hips = joint_of(skel, "mixamorig:Hips");
    if armature {
        let c = skel.containers[0];
        assert_eq!(skel.joints[c].name, "Armature");
        assert_eq!(c, BONES.len(), "container joints are appended after the bones");
        assert_eq!(skel.joints[hips].parent, Some(c), "Hips has the Armature as its parent");
        assert_eq!(skel.joints[c].parent, None);
        assert_eq!(skel.motion_root(), Some(hips), "the motion root stays the Hips");
        assert_eq!(skel.joints.iter().filter(|j| j.parent.is_none()).count(), 1);
    } else {
        assert!(skel.containers.is_empty());
        assert_eq!(skel.joints[hips].parent, None);
    }

    // The bones sit under the container chain, but `world_matrices` takes its basis back out: the
    // palette, and with it the skin under the file's own instance transform, is what the old rule
    // (bones only) gave.
    let rest_new = skel.palette(&skel.rest_locals());
    let rest_old = old.palette(&old.rest_locals());
    assert_same(&posed(&scene, &rest_new, Mat4::IDENTITY), &posed(&scene, &rest_old, Mat4::IDENTITY), &format!("{tag} rest"));

    // Hips-only animation (an Armature that is not animated gets no tracks).
    if v.animate_hips {
        let clip = scene.animations.first().expect("a clip");
        assert!(
            clip.channels.iter().all(|c| c.joint < BONES.len() || v.animate_armature),
            "{tag}: only an animated Armature gets channels"
        );
        for t in [0.0f32, 0.4, 1.0] {
            let locals = clip.sample(skel, t);
            let old_locals = clip.sample(&old, t);
            let new = posed(&scene, &skel.palette(&locals), Mat4::IDENTITY);
            if !v.animate_armature {
                assert_same(&new, &posed(&scene, &old.palette(&old_locals), Mat4::IDENTITY), &format!("{tag} t={t}"));
            }
        }
    }
    if v.animate_armature && armature {
        let clip = scene.animations.first().expect("a clip");
        let c = skel.containers[0];
        assert!(clip.channels.iter().any(|ch| ch.joint == c), "{tag}: the animated Armature has tracks");
        // It moves the whole character along its local X by half a unit.
        let a = clip.sample(skel, 0.0);
        let b = clip.sample(skel, clip.duration - 1e-4);
        let pa = skel.world_matrices(&a)[hips].w_axis.truncate();
        let pb = skel.world_matrices(&b)[hips].w_axis.truncate();
        assert!(((pb - pa).length() - 0.5 * v.k()).abs() < 2e-2 * v.k().max(1.0), "{tag}: moved {:?}", pb - pa);
    }

    // Humanoid: 15 required bones mapped, no issue, and the same bones as without the container.
    let mask = crate::rig_import::real_bone_mask(skel, &scene.meshes);
    let map = crate::auto_map::auto_map_with(skel, Some(&mask));
    assert_eq!(map.required_count(), 15, "{tag}");
    let human = map.to_human(skel);
    let issues = validate_rig(skel, &human);
    if armature {
        assert!(issues.is_empty(), "{tag}: {:?}", issues.iter().map(|i| i.message.clone()).collect::<Vec<_>>());
    }
    let mask_old = crate::rig_import::real_bone_mask(&old, &scene.meshes);
    let human_old = crate::auto_map::auto_map_with(&old, Some(&mask_old)).to_human(&old);
    assert_eq!(human, human_old, "{tag}: the automatic mapping does not change");
    assert!(human.iter().any(|(b, n)| *b == HumanBone::Hips && n == "mixamorig:Hips"));
}

const BLENDER: Variant = Variant {
    unit_scale: 100.0,
    z_up_file: false,
    bones_z_up: true,
    armature: Some(([-90.0, 0.0, 0.0], [0.0, 0.0, 0.0])),
    animate_hips: true,
    animate_armature: false,
    blender_mesh: false,
};

#[test]
fn armature_parent_becomes_a_container_joint_blender_style() {
    check(BLENDER, "chain_blender");
}

#[test]
fn armature_with_translation_and_a_rotation_about_y() {
    check(Variant { armature: Some(([0.0, 30.0, 0.0], [0.5, 0.0, 0.25])), bones_z_up: false, ..BLENDER }, "chain_offset");
}

#[test]
fn centimetre_file_keeps_the_pose() {
    // The root carries a 0.01 unit scale; the Armature's own translation is in centimetres.
    check(Variant { unit_scale: 1.0, armature: Some(([-90.0, 0.0, 0.0], [0.0, 10.0, 0.0])), ..BLENDER }, "chain_cm");
}

#[test]
fn z_up_file_keeps_the_pose() {
    check(Variant { z_up_file: true, bones_z_up: true, armature: Some(([0.0, 0.0, 15.0], [0.0, 0.0, 0.0])), ..BLENDER }, "chain_zup");
    check(Variant { unit_scale: 1.0, z_up_file: true, armature: Some(([-90.0, 0.0, 0.0], [0.0, 0.0, 0.0])), ..BLENDER }, "chain_zup_cm");
}

#[test]
fn animated_armature_gets_tracks_and_moves_the_character() {
    check(Variant { animate_armature: true, ..BLENDER }, "chain_animated");
}

#[test]
fn without_an_armature_nothing_changes() {
    // Mixamo style: the bones hang off the scene root.
    check(Variant { armature: None, bones_z_up: false, ..BLENDER }, "chain_none");
    check(Variant { unit_scale: 1.0, armature: None, bones_z_up: false, ..BLENDER }, "chain_none_cm");
}

#[test]
fn identity_armature_is_still_a_parent() {
    let v = Variant { armature: Some(([0.0; 3], [0.0; 3])), bones_z_up: false, ..BLENDER };
    check(v, "chain_identity");
}

#[test]
fn container_joints_survive_the_import_cache_and_old_entries_are_invalid() {
    const { assert!(crate::import_cache::IMPORTER_VERSION >= 9, "cooked skeletons changed: the importer version is bumped") };
    let doc = skinned_doc(BLENDER);
    let path = doc.write("chain_cache");
    let (a, ra) = crate::load_model_cached(&path, false).unwrap();
    let (b, rb) = crate::load_model_cached(&path, false).unwrap();
    assert_eq!((ra.outcome, rb.outcome), (crate::import_cache::CacheOutcome::Miss, crate::import_cache::CacheOutcome::Hit));
    assert_eq!(a.skeletons[0].containers, vec![BONES.len()]);
    assert_eq!(b.skeletons[0].containers, a.skeletons[0].containers);
    assert_eq!(b.skeletons[0].joints[BONES.len()].name, "Armature");
    assert_eq!(b.skeletons[0].joints[0].parent, Some(BONES.len()));
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn clips_still_bind_to_the_bones_by_index() {
    // Joint indices of the bones are unchanged, so a clip made for the bones alone still samples.
    let scene = import(&skinned_doc(BLENDER), "chain_clip", &ModelImport::default()).unwrap();
    let skel = &scene.skeletons[0];
    let clip = scene.animations.first().expect("a clip");
    assert!(clip.channels.iter().all(|c| c.joint < skel.containers[0]));
    let locals = clip.sample(skel, 0.5);
    assert_eq!(locals.len(), skel.joints.len());
    assert!(locals[skel.containers[0]].abs_diff_eq(skel.rest_locals()[skel.containers[0]], 1e-6), "the container stays at rest");
}

#[test]
fn add_containers_is_a_no_op_for_an_empty_chain() {
    let mut s = Skeleton::default();
    assert!(s.add_containers(&[], &[]).is_none());
    assert!(s.containers.is_empty());
}


/// `Armature` (rotated -90 degrees about X, offset) -> Hips -> Spine, a skinned triangle at the root.
fn gltf_with_armature(tag: &str) -> std::path::PathBuf {
    let arm = Mat4::from_rotation_translation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2), Vec3::new(0.5, 0.0, 0.25));
    let hips = Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0));
    let spine = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5));
    let ibm = [(arm * hips).inverse(), (arm * hips * spine).inverse()];
    let mut buf: Vec<u8> = Vec::new();
    let f32s = |buf: &mut Vec<u8>, v: &[f32]| v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
    // positions (0..36), indices (36..42 + pad), joints (44..56), weights (56..104), ibm (104..232)
    f32s(&mut buf, &[0.0, 0.0, 0.9, 0.2, 0.1, 1.4, -0.2, 0.0, 1.6]);
    for i in [0u16, 1, 2] {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]);
    f32s(&mut buf, &[1.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0]);
    for m in &ibm {
        f32s(&mut buf, &m.to_cols_array());
    }
    let q = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0,3]}}],
"nodes":[{{"name":"Armature","translation":[0.5,0,0.25],"rotation":[{},{},{},{}],"children":[1]}},
{{"name":"Hips","translation":[0,0,1],"children":[2]}},{{"name":"Spine","translation":[0,0,0.5]}},
{{"name":"Body","mesh":0,"skin":0}}],
"skins":[{{"joints":[1,2],"inverseBindMatrices":3}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"JOINTS_0":1,"WEIGHTS_0":2}},"indices":4}}]}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-0.2,0,0.9],"max":[0.2,0.1,1.6]}},
{{"bufferView":1,"componentType":5121,"count":3,"type":"VEC4"}},{{"bufferView":2,"componentType":5126,"count":3,"type":"VEC4"}},
{{"bufferView":3,"componentType":5126,"count":2,"type":"MAT4"}},{{"bufferView":4,"componentType":5123,"count":3,"type":"SCALAR"}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":44,"byteLength":12}},{{"buffer":0,"byteOffset":56,"byteLength":48}},
{{"buffer":0,"byteOffset":104,"byteLength":128}},{{"buffer":0,"byteOffset":36,"byteLength":6}}],
"buffers":[{{"byteLength":{},"uri":"model.bin"}}]}}"#,
        q.x,
        q.y,
        q.z,
        q.w,
        buf.len()
    );
    let dir = std::env::temp_dir().join(format!("zest_gltf_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.gltf");
    std::fs::write(&path, json).unwrap();
    std::fs::write(dir.join("model.bin"), &buf).unwrap();
    path
}

#[test]
fn gltf_armature_node_becomes_a_container_joint() {
    let path = gltf_with_armature("chain");
    let scene = crate::load_gltf(&path).unwrap();
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    let skel = &scene.skeletons[0];
    assert_eq!(skel.containers, vec![2]);
    assert_eq!(skel.joints[2].name, "Armature");
    assert_eq!((skel.joints[0].name.as_str(), skel.joints[0].parent), ("Hips", Some(2)));
    assert_eq!(skel.joints[2].parent, None);
    assert_eq!(skel.motion_root(), Some(0));
    let old = stripped(skel);
    let mut locals = skel.rest_locals();
    let mut old_locals = old.rest_locals();
    assert_same(&posed(&scene, &skel.palette(&locals), Mat4::IDENTITY), &posed(&scene, &old.palette(&old_locals), Mat4::IDENTITY), "gltf rest");
    // A bent spine.
    let bend = Mat4::from_rotation_x(0.7);
    locals[1] *= bend;
    old_locals[1] *= bend;
    assert_same(&posed(&scene, &skel.palette(&locals), Mat4::IDENTITY), &posed(&scene, &old.palette(&old_locals), Mat4::IDENTITY), "gltf posed");
}

#[test]
fn gltf_without_a_parent_node_is_unchanged() {
    // Joint nodes at the scene root: nothing to add.
    let path = gltf_with_armature("plain");
    let text = std::fs::read_to_string(&path).unwrap().replace(r#""nodes":[0,3]"#, r#""nodes":[1,3]"#).replace(r#","children":[1]}"#, "}");
    std::fs::write(&path, text).unwrap();
    let scene = crate::load_gltf(&path).unwrap();
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    assert!(scene.skeletons[0].containers.is_empty());
    assert_eq!(scene.skeletons[0].joints.len(), 2);
}

#[test]
fn old_saved_scene_and_fresh_import_agree_with_the_pre_container_pose() {
    // A scene saved before the container joints stores the instance transform of the old importer, which is
    // exactly the transform a fresh import gives now. The pose of the skinned vertices, bone worlds, bone
    // positions an attachment is placed at, and a descriptor-style offset must all be the pre-change numbers.
    for (tag, v) in [("probe_blender", BLENDER), ("probe_offset", Variant { armature: Some(([0.0, 30.0, 0.0], [0.5, 0.0, 0.25])), bones_z_up: false, ..BLENDER })] {
        let scene = import(&skinned_doc(v), tag, &ModelImport::default()).unwrap();
        let skel = &scene.skeletons[0];
        let old = stripped(skel);
        assert!(!skel.containers.is_empty());
        let old_instance = scene.instances[0].transform; // what the old importer stored
        let basis = skel.container_basis();
        assert!(!(basis - Mat4::IDENTITY).abs_diff_eq(Mat4::ZERO, 1e-3), "{tag}: the chain is not the identity");

        let pose = |locals_new: &[Mat4], locals_old: &[Mat4]| {
            (posed(&scene, &skel.palette(locals_new), Mat4::IDENTITY), posed(&scene, &old.palette(locals_old), Mat4::IDENTITY))
        };
        let mut ln = skel.rest_locals();
        let mut lo = old.rest_locals();
        let (a, b) = pose(&ln, &lo);
        let dev = a.iter().zip(&b).map(|(x, y)| (*x - *y).length()).fold(0.0f32, f32::max);
        assert!(dev < 1e-4, "{tag}: skinned vertices deviate by {dev} m from the pre-change pose");
        assert_eq!(scene.instances[0].transform, old_instance);
        // Bent spine / hips.
        for j in [1usize, 3, 7] {
            let bend = Mat4::from_rotation_x(0.6) * Mat4::from_rotation_z(0.3);
            ln[j] *= bend;
            lo[j] *= bend;
        }
        let (a, b) = pose(&ln, &lo);
        assert_same(&a, &b, &format!("{tag} bent"));

        // A child object placed at a bone (object world = instance * bone world), an eye offset in the head
        // bone's frame, and the avatar-space bone positions.
        let wn = skel.world_matrices(&ln);
        let wo = old.world_matrices(&lo);
        for (i, (x, y)) in wo.iter().zip(&wn).enumerate() {
            assert!(x.abs_diff_eq(*y, 1e-4), "{tag}: bone {i} world differs: {x:?} vs {y:?}");
        }
        let head = joint_of(skel, "mixamorig:Head");
        let eye = Vec3::new(0.03, 0.08, 0.1);
        let at = |w: &[Mat4]| (old_instance * w[head]).transform_point3(eye);
        assert!((at(&wn) - at(&wo)).length() < 1e-4, "{tag}: head-relative offset");
        // The rest pose also matches the bind (an outfit's `bind_world`, `inverse_bind.inverse()`, is not touched).
        for (i, j) in old.joints.iter().enumerate() {
            assert!(j.inverse_bind.abs_diff_eq(skel.joints[i].inverse_bind, 1e-6), "{tag}: bind {i}");
        }
    }
}

#[test]
fn skeleton_without_containers_has_untouched_world_matrices() {
    let scene = import(&skinned_doc(Variant { armature: None, bones_z_up: false, ..BLENDER }), "probe_none", &ModelImport::default()).unwrap();
    let s = &scene.skeletons[0];
    let w = s.world_matrices(&s.rest_locals());
    assert!(w[0].abs_diff_eq(Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0) * Variant { armature: None, bones_z_up: false, ..BLENDER }.k()), 1e-4));
}

/// A mesh nobody has posed yet (no animation, no Animator) is drawn as its raw vertices under the
/// instance transform; once a Blend Shapes component or an Animator takes it, it is drawn skinned by
/// the rest-pose palette. For the two to be one picture the instance transform must stay the mesh
/// node's own world transform: the container joints must not leak into it (they used to, as
/// `L.inverse()`, which turned the skinned picture 90 degrees away from the raw one).
#[test]
fn instance_transform_is_the_nodes_own_and_the_rest_palette_adds_no_container_turn() {
    let scene = import(&skinned_doc(Variant { blender_mesh: true, ..BLENDER }), "raw_blender", &ModelImport::default()).unwrap();
    let skel = &scene.skeletons[0];
    assert_eq!(skel.containers.len(), 1);
    let want = Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    assert!(scene.instances[0].transform.abs_diff_eq(want, 1e-4), "{:?}", scene.instances[0].transform);
    // The rest palette of a bone is its bind relative to the model, with no extra container rotation
    // beyond what the stripped (pre-container) skeleton gives.
    let old = stripped(skel);
    for (a, b) in skel.palette(&skel.rest_locals()).iter().zip(old.palette(&old.rest_locals())).take(old.joints.len()) {
        assert!(a.abs_diff_eq(b, 1e-4));
    }
    let path = gltf_with_armature("raw");
    let g = crate::load_gltf(&path).unwrap();
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    assert!(g.instances[0].transform.abs_diff_eq(Mat4::IDENTITY, 1e-5), "glTF skinned node keeps its own transform: {:?}", g.instances[0].transform);
}
