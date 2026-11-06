//! Import-time rig evidence: which joints are real bones (for the auto-mapper)
//! and the explicit humanoid bone map a VRM file ships with.
//!
//! [`real_bone_mask`] marks joints that skin vertices, or that branch; helper
//! and dummy joints between them add no hierarchy level in
//! [`crate::auto_map::auto_map_with`]. [`vrm_human_map`] reads `humanoid` from
//! VRM 0.x (`extensions.VRM.humanoid.humanBones`, a list) and VRMC_vrm 1.0
//! (`extensions.VRMC_vrm.humanoid.humanBones`, a map). A file map that names
//! all 15 required bones and validates wins over the auto-mapper
//! ([`attach_vrm_rig`]); the rig it makes has `from_file` set.

use std::path::Path;

use serde_json::Value;
use zestengine_render::MeshData;

use crate::human_rig::{AnimationType, HumanBone, RigImport, validate_rig};
use crate::ref_pose::compute_reference_pose;
use crate::{Scene, Skeleton};

/// Per joint: whether it is a real bone (skinned by some vertex of `meshes`,
/// or branching into two or more children). Meshes are taken to be skinned to
/// `skel`, the engine's single-armature assumption.
pub fn real_bone_mask(skel: &Skeleton, meshes: &[MeshData]) -> Vec<bool> {
    let n = skel.joints.len();
    let mut real = vec![false; n];
    for v in meshes.iter().flat_map(|m| m.vertices.iter()) {
        for (j, w) in v.joints.iter().zip(v.weights) {
            if w > 0.0 && (*j as usize) < n {
                real[*j as usize] = true;
            }
        }
    }
    let mut children = vec![0u32; n];
    for j in &skel.joints {
        if let Some(p) = j.parent.filter(|&p| p < n) {
            children[p] += 1;
        }
    }
    for (r, c) in real.iter_mut().zip(children) {
        *r |= c >= 2;
    }
    real
}

/// The humanoid table of a glTF/VRM document as `(bone, node index)`.
/// Unknown bone names are skipped. VRM 1.0 thumbs have a metacarpal where
/// Unity-style rigs have none, so its Metacarpal/Proximal/Distal become
/// Proximal/Intermediate/Distal.
pub fn vrm_humanoid_nodes(json: &Value) -> Vec<(HumanBone, usize)> {
    let ext = json.get("extensions");
    let mut out = Vec::new();
    if let Some(bones) = ext.and_then(|e| e.get("VRMC_vrm")).and_then(|v| v.get("humanoid")).and_then(|h| h.get("humanBones")).and_then(Value::as_object) {
        for (name, v) in bones {
            let Some(node) = v.get("node").and_then(Value::as_u64) else { continue };
            let name = match name.as_str() {
                "leftThumbMetacarpal" => "leftThumbProximal",
                "leftThumbProximal" => "leftThumbIntermediate",
                "rightThumbMetacarpal" => "rightThumbProximal",
                "rightThumbProximal" => "rightThumbIntermediate",
                other => other,
            };
            if let Some(b) = HumanBone::from_name(name) {
                out.push((b, node as usize));
            }
        }
    } else if let Some(bones) = ext.and_then(|e| e.get("VRM")).and_then(|v| v.get("humanoid")).and_then(|h| h.get("humanBones")).and_then(Value::as_array) {
        for v in bones {
            let (Some(name), Some(node)) = (v.get("bone").and_then(Value::as_str), v.get("node").and_then(Value::as_u64)) else { continue };
            if let Some(b) = HumanBone::from_name(name) {
                out.push((b, node as usize));
            }
        }
    }
    // The first entry for a bone wins; keep a stable order.
    out.sort_by_key(|(b, _)| *b as usize);
    out.dedup_by_key(|(b, _)| *b);
    out
}

/// The file's bone map in terms of `skel`'s joint names, when it names all 15
/// required bones on joints of this skeleton and passes [`validate_rig`].
/// Joints are matched by node name, so a name shared by several joints (or a
/// node missing from the skeleton) makes the map unusable for it.
pub fn vrm_human_map(json: &Value, skel: &Skeleton) -> Option<Vec<(HumanBone, String)>> {
    let nodes = json.get("nodes")?.as_array()?;
    let mut human = Vec::new();
    for (bone, node) in vrm_humanoid_nodes(json) {
        // Same fallback the glTF importer gives an unnamed node.
        let Some(n) = nodes.get(node) else {
            if bone.is_required() {
                return None;
            }
            continue;
        };
        let name = n.get("name").and_then(Value::as_str).unwrap_or("joint");
        let mut hits = skel.joints.iter().filter(|j| j.name == name);
        if let (Some(_), None) = (hits.next(), hits.next()) {
            human.push((bone, name.to_owned()));
        } else if bone.is_required() {
            return None;
        }
    }
    let complete = crate::human_rig::ALL.iter().filter(|b| b.is_required()).all(|b| human.iter().any(|(h, _)| h == b));
    (complete && validate_rig(skel, &human).is_empty()).then_some(human)
}

/// The rig a file map gives `skel`: humanoid, reference pose computed,
/// `auto_generated` + `from_file`.
pub fn vrm_rig(json: &Value, skel: &Skeleton) -> Option<RigImport> {
    let human = vrm_human_map(json, skel)?;
    Some(RigImport {
        animation_type: AnimationType::Humanoid,
        reference_pose: compute_reference_pose(skel, &human),
        human,
        auto_generated: true,
        from_file: true,
        ..Default::default()
    })
}

/// Give every skeleton of a freshly loaded `.vrm`/`.glb` the file's humanoid
/// map when it has a usable one. Skeletons it does not fit are left for the
/// auto-mapper. Cheap for files without a VRM extension (the JSON is only
/// parsed when the bytes mention one).
pub fn attach_vrm_rig(path: &Path, scene: &mut Scene) {
    if scene.skeletons.is_empty() {
        return;
    }
    let Some(json) = crate::vrm::gltf_json(path) else { return };
    let ext = json.get("extensions");
    if ext.and_then(|e| e.get("VRM")).is_none() && ext.and_then(|e| e.get("VRMC_vrm")).is_none() {
        return;
    }
    for skel in &mut scene.skeletons {
        if skel.rig.is_none() {
            skel.rig = vrm_rig(&json, skel);
        }
    }
}

/// The rig of another model applied to `skel` by joint name (Avatar Definition's
/// "Copy From Other Avatar"). Bones whose joint the skeleton lacks, or names
/// more than one joint, are dropped; the copy is used only when what remains
/// still maps all 15 required bones and validates. The reference pose is
/// carried over for the joints that exist.
pub fn copy_rig(source: &RigImport, skel: &Skeleton) -> Option<RigImport> {
    let unique = |name: &str| skel.joints.iter().filter(|j| j.name == name).count() == 1;
    let human: Vec<(HumanBone, String)> = source.human.iter().filter(|(_, n)| unique(n)).cloned().collect();
    let candidate = RigImport {
        animation_type: AnimationType::Humanoid,
        avatar_setup: source.avatar_setup.clone(),
        root_node: if unique(&source.root_node) { source.root_node.clone() } else { String::new() },
        reference_pose: source.reference_pose.iter().filter(|p| unique(&p.name)).cloned().collect(),
        human,
        auto_generated: false,
        from_file: false,
        muscles: source.muscles.clone(),
        force_tpose: source.force_tpose,
        humanoid_oversampling: 0,
    };
    (candidate.has_required_bones() && validate_rig(skel, &candidate.human).is_empty()).then_some(candidate)
}

/// Where the model `source` (as typed in the Rig section: a path relative to
/// the project) lives: `source` joined onto the model's folder or one of its
/// ancestors, the first that exists.
pub fn resolve_rig_source(model: &Path, source: &str) -> Option<std::path::PathBuf> {
    let source = source.trim().replace('\\', "/");
    if source.is_empty() {
        return None;
    }
    let direct = Path::new(&source);
    if direct.is_absolute() {
        return direct.is_file().then(|| direct.to_path_buf());
    }
    model.ancestors().skip(1).map(|dir| dir.join(&source)).find(|p| p.is_file())
}

/// [`copy_rig`] from the model file at `source` (see [`resolve_rig_source`]),
/// using the rig stored in its `.meta`. `None` when the source is missing, has
/// no humanoid rig, or does not fit `skel`. The source's own Copy From is not
/// followed.
pub fn copy_rig_from(model: &Path, source: &str, skel: &Skeleton) -> Option<RigImport> {
    let src_path = resolve_rig_source(model, source)?;
    if src_path == model {
        return None;
    }
    let meta = crate::load_asset_meta(&src_path).ok().flatten()?;
    let rig = &meta.importer.as_model()?.rig;
    if !rig.is_configured() {
        return None;
    }
    let mut copied = copy_rig(rig, skel)?;
    copied.avatar_setup = crate::human_rig::AvatarSetup::CopyFrom(source.to_owned());
    Some(copied)
}

/// Load-time Copy From: every skeleton takes the source model's
/// rig when it fits. Returns whether any did (otherwise the model's own stored
/// rig applies).
pub fn apply_copy_from(model: &Path, source: &str, skeletons: &mut [Skeleton]) -> bool {
    let mut any = false;
    for skel in skeletons {
        match copy_rig_from(model, source, skel) {
            Some(r) => {
                skel.rig = Some(r);
                any = true;
            }
            None => tracing::warn!("{}: rig copied from {source:?} does not fit this skeleton (or the source has no humanoid rig)", model.display()),
        }
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;
    use glam::{Mat4, Quat, Vec3};
    use serde_json::json;
    use zestengine_render::Vertex;

    /// root -> hips -> spine -> head, with a leg and an arm chain per side.
    const NAMES: [(&str, Option<usize>); 16] = [
        ("root", None),
        ("hips", Some(0)),
        ("spine", Some(1)),
        ("head", Some(2)),
        ("lul", Some(1)),
        ("lll", Some(4)),
        ("lf", Some(5)),
        ("rul", Some(1)),
        ("rll", Some(7)),
        ("rf", Some(8)),
        ("lua", Some(2)),
        ("lla", Some(10)),
        ("lh", Some(11)),
        ("rua", Some(2)),
        ("rla", Some(13)),
        ("rh", Some(14)),
    ];

    fn skel() -> Skeleton {
        let joints = NAMES
            .iter()
            .enumerate()
            .map(|(i, (n, p))| {
                let t = Vec3::new(0.0, i as f32, 0.0);
                Joint {
                    name: (*n).into(),
                    parent: *p,
                    inverse_bind: Mat4::from_translation(-t),
                    rest_translation: t,
                    rest_rotation: Quat::IDENTITY,
                    rest_scale: Vec3::ONE,
                }
            })
            .collect();
        Skeleton { joints, rig: None, ..Default::default() }
    }

    fn nodes() -> Value {
        Value::Array(NAMES.iter().map(|(n, _)| json!({ "name": n })).collect())
    }

    /// (VRM bone name, node index) of the 15 required bones.
    const REQ: [(&str, usize); 15] = [
        ("hips", 1),
        ("spine", 2),
        ("head", 3),
        ("leftUpperLeg", 4),
        ("leftLowerLeg", 5),
        ("leftFoot", 6),
        ("rightUpperLeg", 7),
        ("rightLowerLeg", 8),
        ("rightFoot", 9),
        ("leftUpperArm", 10),
        ("leftLowerArm", 11),
        ("leftHand", 12),
        ("rightUpperArm", 13),
        ("rightLowerArm", 14),
        ("rightHand", 15),
    ];

    fn vrm0(extra: &[(&str, usize)], skip: &[&str]) -> Value {
        let bones: Vec<Value> = REQ.iter().chain(extra).filter(|(b, _)| !skip.contains(b)).map(|(b, n)| json!({ "bone": b, "node": n, "useDefaultValues": true })).collect();
        json!({ "nodes": nodes(), "extensions": { "VRM": { "humanoid": { "humanBones": bones } } } })
    }

    fn vrm1(extra: &[(&str, usize)]) -> Value {
        let mut bones = serde_json::Map::new();
        for (b, n) in REQ.iter().chain(extra) {
            bones.insert((*b).into(), json!({ "node": n }));
        }
        json!({ "nodes": nodes(), "extensions": { "VRMC_vrm": { "specVersion": "1.0", "humanoid": { "humanBones": bones } } } })
    }

    #[test]
    fn reads_vrm0_and_vrm1_tables() {
        let s = skel();
        for j in [vrm0(&[], &[]), vrm1(&[])] {
            let map = vrm_human_map(&j, &s).expect("complete map");
            assert_eq!(map.len(), 15);
            assert!(map.contains(&(HumanBone::LeftHand, "lh".into())));
            assert!(map.contains(&(HumanBone::Hips, "hips".into())));
        }
    }

    #[test]
    fn vrm1_thumb_metacarpal_maps_to_proximal() {
        let j = vrm1(&[("leftThumbMetacarpal", 12), ("leftThumbProximal", 12), ("leftThumbDistal", 12), ("hair", 3)]);
        let nodes = vrm_humanoid_nodes(&j);
        let get = |b: HumanBone| nodes.iter().find(|(x, _)| *x == b).map(|(_, n)| *n);
        assert_eq!(get(HumanBone::LeftThumbProximal), Some(12));
        assert_eq!(get(HumanBone::LeftThumbIntermediate), Some(12));
        assert_eq!(get(HumanBone::LeftThumbDistal), Some(12));
        assert_eq!(nodes.len(), 18, "unknown 'hair' is skipped");
    }

    #[test]
    fn incomplete_or_foreign_maps_are_rejected() {
        let s = skel();
        assert!(vrm_human_map(&vrm0(&[], &["leftHand"]), &s).is_none(), "required bone missing");
        // A node the skeleton does not have.
        let mut j = vrm0(&[], &[]);
        j["nodes"][12]["name"] = json!("not_a_joint");
        assert!(vrm_human_map(&j, &s).is_none());
        // Ambiguous: two joints share a name.
        let mut dup = skel();
        dup.joints[12].name = "lla".into();
        assert!(vrm_human_map(&vrm0(&[], &[]), &dup).is_none());
        // No VRM extension at all.
        assert!(vrm_human_map(&json!({ "nodes": nodes() }), &s).is_none());
        // An optional bone that does not resolve is dropped, a required one is fatal.
        assert_eq!(vrm_human_map(&vrm0(&[("leftToes", 99)], &[]), &s).map(|m| m.len()), Some(15));
        let mut gone = vrm0(&[], &[]);
        gone["extensions"]["VRM"]["humanoid"]["humanBones"][0]["node"] = json!(99);
        assert!(vrm_human_map(&gone, &s).is_none());
    }

    #[test]
    fn rig_is_marked_from_file() {
        let rig = vrm_rig(&vrm0(&[], &[]), &skel()).unwrap();
        assert!(rig.from_file && rig.auto_generated);
        assert_eq!(rig.animation_type, AnimationType::Humanoid);
        assert!(rig.has_required_bones());
        let back: RigImport = ron::from_str(&ron::to_string(&rig).unwrap()).unwrap();
        assert!(back.from_file);
        // Old files without the field load.
        let old: RigImport = ron::from_str("(animation_type: Humanoid)").unwrap();
        assert!(!old.from_file);
    }

    fn humanoid_rig(skel: &Skeleton) -> RigImport {
        vrm_rig(&vrm0(&[], &[]), skel).unwrap()
    }

    #[test]
    fn copy_rig_maps_by_joint_name_and_needs_the_required_bones() {
        let source = RigImport { avatar_setup: crate::human_rig::AvatarSetup::CreateFromThisModel, ..humanoid_rig(&skel()) };
        let copy = copy_rig(&source, &skel()).unwrap();
        assert_eq!(copy.human, source.human);
        assert!(!copy.auto_generated && !copy.from_file);
        assert_eq!(copy.reference_pose.len(), source.reference_pose.len());
        // The target lacks a required joint: no copy.
        let mut short = skel();
        short.joints[12].name = "renamed_hand".into();
        assert!(copy_rig(&source, &short).is_none());
        // A missing optional joint just drops that bone.
        let mut with_eye = source.clone();
        with_eye.human.push((HumanBone::LeftEye, "no_such_eye".into()));
        assert_eq!(copy_rig(&with_eye, &skel()).unwrap().human.len(), 15);
    }

    #[test]
    fn copy_from_loads_the_sources_meta_rig() {
        let dir = std::env::temp_dir().join(format!("zest_copy_rig_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::create_dir_all(dir.join("props")).unwrap();
        let src = dir.join("models/hero.fbx");
        std::fs::write(&src, b"x").unwrap();
        let mut meta = crate::load_or_create_asset_meta(&src).unwrap();
        meta.importer.as_model_mut().unwrap().rig = humanoid_rig(&skel());
        crate::save_asset_meta(&src, &meta).unwrap();
        // Another model, resolved against an ancestor folder (project root).
        let mine = dir.join("props/npc.fbx");
        assert_eq!(resolve_rig_source(&mine, "models/hero.fbx"), Some(src.clone()));
        assert!(resolve_rig_source(&mine, "models/missing.fbx").is_none());
        let mut skels = vec![skel()];
        assert!(apply_copy_from(&mine, "models/hero.fbx", &mut skels));
        let rig = skels[0].rig.as_ref().unwrap();
        assert!(rig.has_required_bones());
        assert_eq!(rig.avatar_setup, crate::human_rig::AvatarSetup::CopyFrom("models/hero.fbx".into()));
        // A source that does not exist leaves the skeleton alone.
        let mut skels = vec![skel()];
        assert!(!apply_copy_from(&mine, "models/nope.fbx", &mut skels));
        assert!(skels[0].rig.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn real_bones_are_skinned_or_branching() {
        let s = skel();
        let skinned = |joints: &[u32]| MeshData {
            vertices: joints.iter().map(|&j| Vertex { joints: [j, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() }).collect(),
            indices: Vec::new(),
            has_lightmap_uv: false,
        };
        // Skinned: lf (6), rf (9); weight-0 slots do not count. Branching: hips (1), spine (2).
        let mask = real_bone_mask(&s, &[skinned(&[6, 9])]);
        let real: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        assert_eq!(real, vec![1, 2, 6, 9]);
        assert!(real_bone_mask(&s, &[]).iter().filter(|r| **r).count() == 2);
    }
}
