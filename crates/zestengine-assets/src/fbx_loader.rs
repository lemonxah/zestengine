//! FBX import via ufbx.
//!
//! Geometry is converted to y-up meters by ufbx itself; meshes are split per
//! material part (mirroring the glTF loader's one-mesh-per-primitive model),
//! and PBR factors plus base color / normal / emission textures are mapped
//! onto the standard shader. Roughness/metalness *textures* are not packed
//! into ORM yet (factors only); tracked in TODO.md.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result, anyhow, bail};
use zestengine_render::{AlphaMode, MaterialFeatures, MaterialParams, TextureData, Vertex};
use glam::{Mat4, Quat, Vec3};

use crate::skeleton::{AnimationClip, ChannelPath, Joint, Skeleton};
#[cfg(test)]
use crate::skeleton::AnimChannel;
use crate::{Instance, MeshSlot, ModelImport, Scene, SceneMaterial};

/// Import an FBX with default settings.
pub fn load_fbx(path: impl AsRef<Path>) -> Result<Scene> {
    load_fbx_with(path, &ModelImport::default())
}

/// Import an FBX, applying the asset's [`ModelImport`] settings (scale, flip-UV,
/// import-materials) from its `.meta` sidecar.
pub fn load_fbx_with(path: impl AsRef<Path>, settings: &ModelImport) -> Result<Scene> {
    let path = path.as_ref();
    // Same validation a `.meta` load runs, for settings built in code.
    let mut settings = settings.clone();
    settings.validate();
    let settings = &settings;
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        // Bake off keeps the axis conversion as a rotation on the root; on bakes it
        // into every node's geometry and animation.
        space_conversion: if settings.bake_axis_conversion { ufbx::SpaceConversion::ModifyGeometry } else { ufbx::SpaceConversion::TransformRoot },
        generate_missing_normals: true,
        load_external_files: true,
        ignore_missing_external_files: true,
        ..Default::default()
    };
    let fbx = ufbx::load_file(path.to_str().context("non-UTF8 path")?, opts)
        .map_err(|e| anyhow!("loading FBX {}: {}", path.display(), e.description.as_ref()))?;

    let base_dir = path.parent().unwrap_or(Path::new("."));
    let mut scene = Scene {
        meshes: Vec::new(),
        textures: Vec::new(),
        materials: Vec::new(),
        instances: Vec::new(),
        skeletons: Vec::new(),
        morphs: Vec::new(),
        springs: Default::default(),
        animations: Vec::new(),
        collision: Vec::new(),
        identity: Default::default(),
    };

    // Default material for parts without one.
    scene.materials.push(SceneMaterial {
        name: "default".into(),
        params: MaterialParams::default(),
        features: MaterialFeatures::default(),
        albedo: None,
        normal: None,
        orm: None,
        emission: None,
    });

    // element_id -> scene material index
    let mut material_cache: HashMap<u32, usize> = HashMap::new();
    // (texture element_id, srgb) -> scene texture index
    let mut texture_cache: HashMap<(u32, bool), usize> = HashMap::new();
    // (mesh element_id, part index) -> (scene mesh index, scene material index)
    let mut part_cache: HashMap<(u32, usize), Vec<(usize, usize)>> = HashMap::new();

    // Unique node path of every node (siblings made unique before the ids are
    // hashed from them); a mesh part's name is `<node path>#part<k>`.
    let mut node_paths: HashMap<u32, String> = HashMap::new();
    let mut namer = crate::sub_asset::SiblingNamer::default();
    for node in &fbx.nodes {
        let parent = node.parent.as_ref().and_then(|p| node_paths.get(&p.element.element_id)).cloned().unwrap_or_default();
        let raw = if node.element.name.is_empty() { "node" } else { node.element.name.as_ref() };
        let path = namer.unique_path(&parent, raw);
        node_paths.insert(node.element.element_id, path);
    }

    // The mesh pipeline (triangulate, normals, tangents, weld, optimise) of every
    // part is independent, so the soups are read off the ufbx scene here and the
    // pipelines run side by side; the loop below takes its results in file order.
    let mut results: HashMap<(u32, usize), Vec<crate::mesh_pipeline::Imported>> = {
        let mut keys = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut soups = Vec::new();
        for node in &fbx.nodes {
            let Some(mesh) = &node.mesh else { continue };
            if mesh.num_triangles == 0 {
                continue;
            }
            for part_index in 0..mesh.material_parts.len().max(1) {
                let key = (mesh.element.element_id, part_index);
                if !seen.insert(key) {
                    continue;
                }
                keys.push(key);
                soups.push(part_soup(mesh, part_index, settings));
            }
        }
        let pipeline = crate::mesh_pipeline::Settings::from_import(settings);
        let mut done: Vec<Option<Vec<crate::mesh_pipeline::Imported>>> = Vec::new();
        done.resize_with(soups.len(), || None);
        let mut work: Vec<(Option<crate::mesh_pipeline::ImportMesh>, &mut Option<Vec<crate::mesh_pipeline::Imported>>)> =
            soups.into_iter().zip(done.iter_mut()).collect();
        crate::mesh_pipeline::par_each_dyn(&mut work, |(soup, out)| {
            **out = Some(soup.take().map_or_else(Vec::new, |s| crate::mesh_pipeline::run_split(s, &pipeline)));
        });
        drop(work);
        keys.into_iter().zip(done.into_iter().map(Option::unwrap_or_default)).collect()
    };

    // Names of the file's blendshape channels: hidden nodes named like them are exporter-left
    // target copies, not part of the model.
    let target_names: std::collections::HashSet<String> = if settings.import_blend_shapes && settings.skip_blend_shape_targets {
        let mut names = std::collections::HashSet::new();
        for node in &fbx.nodes {
            let Some(mesh) = &node.mesh else { continue };
            for channel in mesh.blend_deformers.iter().flat_map(|d| d.channels.iter()) {
                let raw: &str = channel.element.name.as_ref();
                names.insert(crate::fbx_anim::shape_name(raw));
                names.insert(raw.to_owned());
            }
        }
        names
    } else {
        Default::default()
    };
    let mut skipped_targets = 0usize;
    for node in &fbx.nodes {
        let Some(mesh) = &node.mesh else { continue };
        if mesh.num_triangles == 0 {
            continue;
        }
        if !target_names.is_empty() && !node_visible(node) {
            let raw: &str = node.element.name.as_ref();
            if target_names.contains(raw) || target_names.contains(&crate::fbx_anim::shape_name(raw)) {
                skipped_targets += 1;
                continue;
            }
        }
        // Build the armature + animation clips from the first skinned mesh's
        // deformer (cluster order matches the skin-local joint indices on the
        // vertices); sample every FBX anim stack onto the bone nodes.
        if scene.skeletons.is_empty()
            && let Some((skel, clips, infos, _)) = build_rig(&fbx, mesh, settings) {
                scene.skeletons.push(skel);
                scene.animations.extend(clips);
                scene.identity.clip_info.extend(infos);
            }
        // The skinned instance keeps the file's own transform (what a scene saved
        // before the container joints stored): `Skeleton::world_matrices` takes the
        // containers' rest basis out of the pose instead.
        let transform = matrix_to_mat4(&node.geometry_to_world);
        let node_name = if node.element.name.is_empty() {
            format!("node {}", node.element.element_id)
        } else {
            node.element.name.to_string()
        };

        // Parts: one per material, or the whole mesh when unpartitioned. All
        // parts of a node become one instance with a material slot per part.
        let part_count = mesh.material_parts.len().max(1);
        let mut slots: Vec<MeshSlot> = Vec::new();
        for part_index in 0..part_count {
            let key = (mesh.element.element_id, part_index);
            let entry: Vec<(usize, usize)> = match part_cache.get(&key) {
                Some(entry) => entry.clone(),
                None => {
                    // A forced 16 bit format can cut one part into several meshes.
                    let pieces = results.remove(&key).unwrap_or_default();
                    if pieces.is_empty() {
                        continue;
                    }
                    let material_index = mesh
                        .materials
                        .get(part_index)
                        .map(|material| {
                            material_for(
                                material,
                                base_dir,
                                &mut scene,
                                &mut material_cache,
                                &mut texture_cache,
                            )
                        })
                        .unwrap_or(0);
                    let node_path = node_paths.get(&node.element.element_id).map(String::as_str).unwrap_or("node");
                    let mut entry = Vec::new();
                    for (k, imported) in pieces.into_iter().enumerate() {
                        let (mesh_data, mut morphs) = (imported.mesh, imported.morphs);
                        crate::anim_import::finish_morphs(&mesh_data, &mut morphs, &settings.blend_shapes);
                        if !morphs.is_empty() {
                            scene.morphs.resize(scene.meshes.len(), Vec::new());
                            scene.morphs.push(morphs);
                        }
                        scene.meshes.push(mesh_data);
                        let mesh_index = scene.meshes.len() - 1;
                        if settings.import_blend_shapes && settings.blend_shapes.import_deform_percent {
                            // Initial weights: the channels' DeformPercent, for the shapes that survived.
                            let kept = scene.morphs.get(mesh_index).map(Vec::as_slice).unwrap_or_default();
                            for channel in mesh.blend_deformers.iter().flat_map(|d| d.channels.iter()) {
                                let (name, w) = (crate::fbx_anim::shape_name(channel.element.name.as_ref()), channel.weight as f32);
                                if w.abs() > 1e-6 && kept.iter().any(|t| t.name == name) {
                                    scene.identity.morph_weights.push((mesh_index, name, w));
                                }
                            }
                        }
                        scene.identity.set_quads(mesh_index, imported.quads);
                        scene.identity.mesh_names.push(if k == 0 { format!("{node_path}#part{part_index}") } else { format!("{node_path}#part{part_index}~s{k}") });
                        entry.push((mesh_index, material_index));
                    }
                    part_cache.insert(key, entry.clone());
                    entry
                }
            };
            slots.extend(entry.into_iter().map(|(mesh, material)| MeshSlot { mesh, material }));
        }
        if !slots.is_empty() {
            if settings.import_visibility && !node_visible(node) {
                scene.identity.hidden_instances.push(scene.instances.len());
            }
            scene.instances.push(Instance {
                name: node_name,
                transform,
                slots,
            });
        }
    }

    if skipped_targets > 0 {
        tracing::info!("{}: skipped {skipped_targets} hidden blendshape target mesh(es); the shapes come from the deformer", path.display());
    }
    if settings.import_cameras || settings.import_lights {
        read_cameras_and_lights(&fbx, settings, &mut scene);
    }
    scene.identity.takes = crate::fbx_anim::list_takes(&fbx);
    read_extras(&fbx, settings, &mut scene);
    if scene.instances.is_empty() {
        // No mesh: an animation-only file (Mixamo "Without Skin") is a clip
        // library whose skeleton comes from the node hierarchy.
        match crate::fbx_anim::build_anim_only(&fbx, settings) {
            Some((skel, clips, infos)) if !skel.joints.is_empty() && !clips.is_empty() => {
                scene.skeletons.push(skel);
                scene.animations.extend(clips);
                scene.identity.clip_info.extend(infos);
            }
            _ => bail!("FBX file {} produced no renderable meshes and no animation", path.display()),
        }
    }

    // Apply per-asset import settings (.meta) to the converted scene.
    if settings.scale != 1.0 {
        apply_scale(&mut scene, settings.scale);
    }
    if !settings.import_materials {
        // Strip imported materials: keep geometry, fall back to plain defaults.
        for m in &mut scene.materials {
            m.params = MaterialParams::default();
            m.features = MaterialFeatures::default();
            m.albedo = None;
            m.normal = None;
            m.orm = None;
            m.emission = None;
        }
    }

    tracing::info!(
        meshes = scene.meshes.len(),
        materials = scene.materials.len(),
        textures = scene.textures.len(),
        instances = scene.instances.len(),
        "FBX scene loaded"
    );
    Ok(scene)
}

/// The import-side extras of a file: bone-influence warnings, constraints,
/// user properties and the Optimize Game Objects record.
fn read_extras(fbx: &ufbx::Scene, settings: &ModelImport, scene: &mut Scene) {
    // Skin Weights: vertices with more influences than the limit lose the weakest ones.
    let limit = settings.max_bones_per_vertex.clamp(1, 4) as usize;
    let (mut total, mut over, mut worst) = (0usize, 0usize, 0usize);
    // The largest share of one vertex's weight that the cut throws away.
    let mut max_lost = 0.0f32;
    let mut seen = std::collections::HashSet::new();
    for node in &fbx.nodes {
        let Some(mesh) = &node.mesh else { continue };
        if !seen.insert(mesh.element.element_id) {
            continue;
        }
        let Some(skin) = (&mesh.skin_deformers).into_iter().next() else { continue };
        for sv in &skin.vertices {
            total += 1;
            let n = sv.num_weights as usize;
            worst = worst.max(n);
            if n > limit {
                over += 1;
                let begin = sv.weight_begin as usize;
                let mut w: Vec<f32> = (begin..begin + n).filter_map(|k| skin.weights.get(k)).map(|s| s.weight as f32).collect();
                w.sort_by(|a, b| b.total_cmp(a));
                let sum: f32 = w.iter().sum();
                if sum > 1e-6 {
                    max_lost = max_lost.max(w[limit.min(w.len())..].iter().sum::<f32>() / sum);
                }
            }
        }
    }
    if over > 0 {
        let share = over as f32 / total.max(1) as f32;
        // The engine skins with at most 4 influences (as Unity's default does); a few
        // vertices with a fifth, tiny weight are normal in humanoid rigs and invisible.
        // Only a notable loss is worth a warning in the inspector.
        let msg = format!(
            "{over} of {total} skinned vertices ({:.1}%) have more than {limit} bone influences (up to {worst}); the weakest are dropped and the rest renormalised (at most {:.1}% of a vertex's weight is lost). Skinning uses at most 4 influences; reduce the extra weights in the source if the deformation looks wrong.",
            share * 100.0,
            max_lost * 100.0
        );
        if max_lost < 0.10 && share < 0.05 {
            tracing::info!("{msg}");
        } else {
            scene.identity.import_warnings.push(msg);
        }
    }
    if settings.import_constraints {
        let (specs, report) = crate::fbx_extras::constraint_specs(fbx);
        scene.identity.constraints = specs;
        if report.unsupported > 0 {
            scene.identity.import_warnings.push(format!(
                "{} constraint(s) were not imported (IK chains and constraints without a node or sources are not supported); Bake Animations does not solve them either.",
                report.unsupported
            ));
        }
    }
    if settings.animation.animated_custom_properties {
        for node in &fbx.nodes {
            let props = crate::fbx_extras::user_props(node);
            if !props.is_empty() {
                let name = if node.element.name.is_empty() { format!("node {}", node.element.element_id) } else { node.element.name.to_string() };
                scene.identity.custom_props.push((name, props.into_iter().map(|p| (p.name, p.value)).collect()));
            }
        }
    }
}

/// Fill [`crate::OptimizeInfo`] once the skeletons exist: the joints Optimize
/// Game Objects keeps inside the skeleton asset, and which Extra Expose
/// Transforms name something in the model.
pub(crate) fn record_optimize(scene: &mut Scene, settings: &ModelImport) {
    let mut info = crate::OptimizeInfo { enabled: settings.optimize_game_objects, ..Default::default() };
    if settings.optimize_game_objects {
        let names: std::collections::HashSet<&str> =
            scene.skeletons.iter().flat_map(|s| s.joints.iter().map(|j| j.name.as_str())).chain(scene.instances.iter().map(|i| i.name.as_str())).collect();
        info.exposed = settings.extra_expose_transforms.iter().filter(|n| names.contains(n.as_str())).cloned().collect();
        info.internal_joints = scene.skeletons.iter().flat_map(|s| &s.joints).filter(|j| !settings.extra_expose_transforms.contains(&j.name)).count();
    }
    scene.identity.optimize = info;
}

/// Apply the import `scale` to everything that holds a length, not only the
/// vertices: node transforms, bone rest poses and inverse-binds, animation
/// translation keys and blendshape position deltas. A uniform scale `s` turns
/// a transform's translation into `s * t` and leaves rotation and scale alone,
/// and an inverse-bind `B` into `S B S^-1` (translation scaled), which keeps
/// skinning consistent with the scaled vertices.
pub(crate) fn apply_scale(scene: &mut Scene, s: f32) {
    for mesh in &mut scene.meshes {
        for v in &mut mesh.vertices {
            for c in &mut v.position {
                *c *= s;
            }
        }
        if s < 0.0 {
            // A negative scale turns the surface inside out: swap each triangle's
            // winding and flip the normal and tangent frame to match, so the
            // mesh still faces outward.
            for tri in mesh.indices.as_chunks_mut::<3>().0 {
                tri.swap(1, 2);
            }
            for v in &mut mesh.vertices {
                v.normal = v.normal.map(|c| -c);
                v.tangent = [-v.tangent[0], -v.tangent[1], -v.tangent[2], -v.tangent[3]];
            }
        }
    }
    for morph in scene.morphs.iter_mut().flatten() {
        for (_, dp, dn) in &mut morph.deltas {
            for c in dp.iter_mut() {
                *c *= s;
            }
            if s < 0.0 {
                for c in dn.iter_mut() {
                    *c = -*c;
                }
            }
        }
    }
    for inst in &mut scene.instances {
        inst.transform.w_axis *= glam::Vec4::new(s, s, s, 1.0);
    }
    for cam in &mut scene.identity.cameras {
        cam.transform[12..15].iter_mut().for_each(|c| *c *= s);
        cam.near *= s.abs();
        cam.far *= s.abs();
    }
    for light in &mut scene.identity.lights {
        light.transform[12..15].iter_mut().for_each(|c| *c *= s);
        light.range *= s.abs();
    }
    scene.springs.scale(s);
    for c in &mut scene.identity.constraints {
        for t in std::iter::once(&mut c.offset).chain(c.sources.iter_mut().map(|src| &mut src.offset)) {
            t.translation = t.translation.map(|v| v * s);
        }
    }
    for skel in &mut scene.skeletons {
        for j in &mut skel.joints {
            j.rest_translation *= s;
            j.inverse_bind.w_axis *= glam::Vec4::new(s, s, s, 1.0);
        }
    }
    for clip in &mut scene.animations {
        for ch in &mut clip.channels {
            if ch.path == ChannelPath::Translation {
                ch.scale_translation(s);
            }
        }
    }
}

/// Skin weights for one triangle corner: up to 4 (joint = skin-cluster index,
/// weight) from the mesh's first skin deformer, keyed by control-point index.
/// Returns zeros for unrigged meshes. Cluster indices line up with the skeleton
/// the importer builds from the same deformer's cluster order.
fn skin_corner(mesh: &ufbx::Mesh, corner: usize, max_bones: usize) -> ([u32; 4], [f32; 4]) {
    let mut joints = [0u32; 4];
    let mut weights = [0.0f32; 4];
    let Some(skin) = (&mesh.skin_deformers).into_iter().next() else {
        return (joints, weights);
    };
    if corner >= mesh.vertex_indices.count {
        return (joints, weights);
    }
    let cp = mesh.vertex_indices[corner] as usize;
    if cp >= skin.vertices.count {
        return (joints, weights);
    }
    let sv = skin.vertices[cp];
    let begin = sv.weight_begin as usize;
    let n = (sv.num_weights as usize).min(max_bones.clamp(1, 4));
    for k in 0..n {
        let w = skin.weights[begin + k];
        joints[k] = w.cluster_index;
        weights[k] = w.weight as f32;
    }
    let sum: f32 = weights.iter().sum();
    if sum > 1e-6 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    (joints, weights)
}

fn ufbx_v3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}
fn ufbx_quat(q: ufbx::Quat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

/// Build a [`Skeleton`] + animation clips from a mesh's first skin deformer.
/// Joints are emitted in cluster order (matching the skin-local `cluster_index`
/// stored on vertices); each cluster's `geometry_to_bone` is the inverse-bind,
/// The rig of one FBX: its skeleton, sampled clips, clip info and bind matrix.
type BuiltRig = (Skeleton, Vec<AnimationClip>, Vec<crate::ClipImportInfo>, Option<Mat4>);

/// parents come from the bone nodes' FBX parent chain, and every FBX anim stack
/// is sampled (at the file's frame rate) onto the bone nodes into per-joint TRS channels.
fn build_rig<'a>(
    fbx: &'a ufbx::Scene,
    mesh: &'a ufbx::Mesh,
    settings: &ModelImport,
) -> Option<BuiltRig> {
    let skin = (&mesh.skin_deformers).into_iter().next()?;
    if skin.clusters.count == 0 {
        return None;
    }
    let mut id_to_joint: HashMap<u32, usize> = HashMap::new();
    for (i, cluster) in (&skin.clusters).into_iter().enumerate() {
        if let Some(node) = &cluster.bone_node {
            id_to_joint.insert(node.element.element_id, i);
        }
    }
    let mut joints = Vec::new();
    let mut bone_nodes: Vec<Option<&ufbx::Node>> = Vec::new();
    for cluster in &skin.clusters {
        let Some(node) = &cluster.bone_node else {
            joints.push(Joint {
                name: "joint".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY,
                rest_translation: Vec3::ZERO,
                rest_rotation: Quat::IDENTITY,
                rest_scale: Vec3::ONE,
            });
            bone_nodes.push(None);
            continue;
        };
        // The nearest ancestor that is a joint: plain nodes and scale helpers between
        // two bones do not break the hierarchy.
        let mut parent = None;
        let mut up = node.parent.as_ref();
        while let Some(p) = up {
            if let Some(&j) = id_to_joint.get(&p.element.element_id) {
                parent = Some(j);
                break;
            }
            up = p.parent.as_ref();
        }
        let t = node.local_transform;
        joints.push(Joint {
            name: if node.element.name.is_empty() {
                format!("joint {}", node.element.element_id)
            } else {
                node.element.name.to_string()
            },
            parent,
            inverse_bind: matrix_to_mat4(&cluster.geometry_to_bone),
            rest_translation: ufbx_v3(t.translation),
            rest_rotation: ufbx_quat(t.rotation),
            rest_scale: ufbx_v3(t.scale),
        });
        bone_nodes.push(Some(node.as_ref()));
    }
    let mut skeleton = Skeleton { joints, rig: None, ..Default::default() };

    // The nodes between the scene root and the topmost bones (a Blender
    // `Armature` empty) stay in the skeleton as container joints, as Unity keeps
    // them above the Hips. The scene root itself (the unit / axis conversion)
    // is not one of them: the instance transform carries it.
    let chain_fix = add_container_joints(&mut skeleton, &mut bone_nodes, fbx);

    // Takes become clips (cut by the model's clip list, sampled, thinned): see `fbx_anim`.
    let morphs = crate::fbx_anim::morph_sources(fbx);
    let (clips, infos) = crate::fbx_anim::build_clips(fbx, &skeleton, &bone_nodes, settings, &morphs, &[]);
    Some((skeleton, clips, infos, chain_fix))
}

/// Add the non-bone ancestors of the skeleton's parentless bones as container
/// joints ([`Skeleton::add_containers`]). Only when every parentless bone has the
/// same ancestor chain; a container joint gets its node (so clips sample it) only
/// when the file animates it. Returns the inverse of the chain's transform (unused:
/// `Skeleton::world_matrices` takes the chain out of the pose instead).
fn add_container_joints<'a>(skeleton: &mut Skeleton, bone_nodes: &mut Vec<Option<&'a ufbx::Node>>, fbx: &'a ufbx::Scene) -> Option<Mat4> {
    let roots: Vec<usize> = (0..skeleton.joints.len()).filter(|&j| skeleton.joints[j].parent.is_none() && bone_nodes[j].is_some()).collect();
    let chain_of = |node: &ufbx::Node| -> Vec<u32> {
        let mut ids = Vec::new();
        let mut up = node.parent.as_ref();
        while let Some(p) = up {
            if p.is_root {
                break;
            }
            ids.push(p.element.element_id);
            up = p.parent.as_ref();
        }
        ids.reverse();
        ids
    };
    let first = chain_of(bone_nodes[*roots.first()?]?);
    if first.is_empty() || roots.iter().any(|&j| bone_nodes[j].map(&chain_of).as_ref() != Some(&first)) {
        return None;
    }
    let nodes: Vec<&ufbx::Node> = first.iter().filter_map(|id| fbx.nodes.iter().find(|n| n.element.element_id == *id).map(|n| &**n)).collect();
    if nodes.len() != first.len() {
        return None;
    }
    let chain: Vec<(String, Vec3, Quat, Vec3)> = nodes
        .iter()
        .map(|n| {
            let t = n.local_transform;
            let name = if n.element.name.is_empty() { format!("joint {}", n.element.element_id) } else { n.element.name.to_string() };
            (name, ufbx_v3(t.translation), ufbx_quat(t.rotation), ufbx_v3(t.scale))
        })
        .collect();
    let combined = skeleton.add_containers(&roots, &chain)?;
    for n in nodes {
        bone_nodes.push(crate::fbx_anim::is_animated(fbx, n).then_some(n));
    }
    Some(combined.inverse())
}

/// The source frame rate to sample animation at: the file's own, or 30 when it
/// is missing or nonsensical (Unity resets a bad rate to 1; we keep clips usable).
pub(crate) fn source_frame_rate(fps: f64) -> f64 {
    if fps.is_finite() && fps > 0.0 {
        fps.clamp(1.0, 960.0)
    } else {
        tracing::warn!("FBX frame rate {fps} is invalid; sampling at 30 fps");
        30.0
    }
}

/// Times (seconds from the clip start) at which a `duration` second clip is
/// sampled: one per source frame, plus the exact end so the last pose is not
/// cut short or overshot.
pub(crate) fn sample_times(duration: f64, fps: f64) -> Vec<f64> {
    let frames = (duration * fps - 1e-6).ceil().max(0.0) as usize + 1;
    (0..frames).map(|f| (f as f64 / fps).min(duration)).collect()
}

/// Convert one material part of a ufbx mesh into the pipeline's polygon soup
/// and run it. Quads stay quads (the Triangulate pass splits them, or Keep
/// Quads keeps them); larger polygons are triangulated by ufbx. `None` when the
/// part has no geometry left (more than one piece when a forced 16 bit index
/// format cuts a big part).
#[allow(dead_code)] // conversion entry used by the model importer
fn convert_part(mesh: &ufbx::Mesh, part_index: usize, settings: &ModelImport) -> Vec<crate::mesh_pipeline::Imported> {
    match part_soup(mesh, part_index, settings) {
        Some(soup) => crate::mesh_pipeline::run_split(soup, &crate::mesh_pipeline::Settings::from_import(settings)),
        None => Vec::new(),
    }
}

/// The polygon soup of one material part (corners, skin weights, blendshape
/// deltas by corner), before the mesh pipeline runs. Reads the ufbx mesh, so it
/// stays on the loading thread; the pipeline passes after it are plain data.
fn part_soup(mesh: &ufbx::Mesh, part_index: usize, settings: &ModelImport) -> Option<crate::mesh_pipeline::ImportMesh> {
    use crate::mesh_pipeline::ImportMesh;
    let mut soup = ImportMesh {
        has_normals: mesh.vertex_normal.exists,
        has_tangents: mesh.vertex_tangent.exists,
        // Second UV set = lightmap UVs, if the mesh has one.
        real_uv1: mesh.uv_sets.get(1).is_some_and(|s| s.vertex_uv.exists),
        ..Default::default()
    };
    let uv_set1 = mesh.uv_sets.get(1).filter(|s| s.vertex_uv.exists);
    let mut tri_indices = vec![0u32; mesh.max_face_triangles.max(1) * 3];
    // Control point of every soup corner, to carry blendshape offsets over.
    let mut control_points: Vec<u32> = Vec::new();
    let face_indices: Vec<u32> = match mesh.material_parts.get(part_index) {
        Some(part) => part.face_indices.as_ref().to_vec(),
        None => (0..mesh.faces.len() as u32).collect(),
    };

    let corner = |i: usize| -> Vertex {
        let p = mesh.vertex_position[i];
        let n = if mesh.vertex_normal.exists {
            let n = mesh.vertex_normal[i];
            [n.x as f32, n.y as f32, n.z as f32]
        } else {
            [0.0, 1.0, 0.0]
        };
        let uv = if mesh.vertex_uv.exists {
            let uv = mesh.vertex_uv[i];
            // FBX uses bottom-left UV origin; textures are top-left. Flip UV (V)
            // undoes that for sources authored top-left.
            [uv.x as f32, if settings.flip_uv { uv.y as f32 } else { 1.0 - uv.y as f32 }]
        } else {
            [0.0, 0.0]
        };
        let color = if mesh.vertex_color.exists {
            let c = mesh.vertex_color[i];
            [c.x as f32, c.y as f32, c.z as f32, c.w as f32]
        } else {
            [1.0; 4]
        };
        let tangent = if mesh.vertex_tangent.exists {
            let t = mesh.vertex_tangent[i];
            [t.x as f32, t.y as f32, t.z as f32, 1.0]
        } else {
            [1.0, 0.0, 0.0, 1.0]
        };
        let uv1 = match uv_set1 {
            Some(set) => {
                let uv = set.vertex_uv[i];
                [uv.x as f32, 1.0 - uv.y as f32]
            }
            None => [0.0, 0.0],
        };
        // Skin weights are per control point in ufbx; resolve them per corner.
        let (joints, weights) = skin_corner(mesh, i, settings.max_bones_per_vertex as usize);
        Vertex { position: [p.x as f32, p.y as f32, p.z as f32], normal: n, uv, color, tangent, uv1, joints, weights }
    };

    for &face_index in &face_indices {
        let face = mesh.faces[face_index as usize];
        if face.num_indices == 4 {
            let begin = face.index_begin as usize;
            let quad: Vec<Vertex> = (begin..begin + 4).map(corner).collect();
            control_points.extend((begin..begin + 4).map(|i| mesh.vertex_indices[i]));
            soup.push_face(&quad);
            continue;
        }
        let num_tris = ufbx::triangulate_face(&mut tri_indices, mesh, face);
        for tri in tri_indices[..num_tris as usize * 3].as_chunks::<3>().0 {
            let verts: Vec<Vertex> = tri.iter().map(|&c| corner(c as usize)).collect();
            control_points.extend(tri.iter().map(|&c| mesh.vertex_indices[c as usize]));
            soup.push_face(&verts);
        }
    }
    if soup.faces.is_empty() {
        return None;
    }
    if settings.import_blend_shapes {
        soup.morphs = part_morphs(mesh, &control_points);
    }
    Some(soup)
}

/// Whether the node and every ancestor are visible (FBX has a per-node flag).
fn node_visible(node: &ufbx::Node) -> bool {
    let mut cur = Some(node);
    while let Some(n) = cur {
        if !n.visible {
            return false;
        }
        cur = n.parent.as_deref();
    }
    true
}

fn material_for(
    material: &ufbx::Material,
    base_dir: &Path,
    scene: &mut Scene,
    material_cache: &mut HashMap<u32, usize>,
    texture_cache: &mut HashMap<(u32, bool), usize>,
) -> usize {
    let id = material.element.element_id;
    if let Some(&index) = material_cache.get(&id) {
        return index;
    }

    let pbr = &material.pbr;
    let vec4 = |map: &ufbx::MaterialMap, fallback: [f32; 4]| {
        if map.has_value {
            [
                map.value_vec4.x as f32,
                map.value_vec4.y as f32,
                map.value_vec4.z as f32,
                map.value_vec4.w as f32,
            ]
        } else {
            fallback
        }
    };
    let real = |map: &ufbx::MaterialMap, fallback: f32| {
        if map.has_value {
            map.value_vec4.x as f32
        } else {
            fallback
        }
    };

    let mut load_tex = |scene: &mut Scene, map: &ufbx::MaterialMap, srgb: bool| -> Option<usize> {
        let texture = map.texture.as_ref()?;
        let key = (texture.element.element_id, srgb);
        if let Some(&index) = texture_cache.get(&key) {
            return Some(index);
        }
        match texture_data(texture, base_dir, srgb) {
            Ok(data) => {
                scene.textures.push(data);
                let file = [texture.relative_filename.as_ref(), texture.filename.as_ref(), texture.absolute_filename.as_ref()].into_iter().find(|n| !n.is_empty()).unwrap_or(texture.element.name.as_ref());
                scene.identity.texture_names.push(crate::material_remap::texture_stem(file));
                let index = scene.textures.len() - 1;
                texture_cache.insert(key, index);
                Some(index)
            }
            Err(e) => {
                tracing::warn!("skipping FBX texture: {e:#}");
                None
            }
        }
    };

    // The FBX texture's UV transform (Blender's mapping-node scale and
    // offset, when the exporter writes them): the scale is the standard
    // shader's tiling and the translation its offset. Rotation is not
    // representable in the shader's tiling/offset pair and is ignored.
    let uv_of = |map: &ufbx::MaterialMap| -> ([f32; 2], [f32; 2]) {
        let Some(tex) = map.texture.as_ref().filter(|t| t.has_uv_transform) else { return ([1.0; 2], [0.0; 2]) };
        (
            [tex.uv_transform.scale.x as f32, tex.uv_transform.scale.y as f32],
            [tex.uv_transform.translation.x as f32, tex.uv_transform.translation.y as f32],
        )
    };
    let (albedo_tiling, albedo_offset) = uv_of(&pbr.base_color);
    let (normal_tiling, normal_offset) = uv_of(&pbr.normal_map);
    let (emission_tiling, emission_offset) = uv_of(&pbr.emission_color);

    let albedo = load_tex(scene, &pbr.base_color, true);
    let normal = load_tex(scene, &pbr.normal_map, false);
    let emission_tex = load_tex(scene, &pbr.emission_color, true);

    let base = vec4(&pbr.base_color, [1.0; 4]);
    let base_factor = real(&pbr.base_factor, 1.0);
    let emission_color = vec4(&pbr.emission_color, [0.0; 4]);
    let emission_factor = real(&pbr.emission_factor, 1.0);
    let opacity = real(&pbr.opacity, 1.0);
    let has_emission = emission_tex.is_some()
        || (emission_factor > 0.0 && emission_color[..3].iter().any(|&c| c > 0.0));

    scene.materials.push(SceneMaterial {
        name: if material.element.name.is_empty() {
            format!("material {id}")
        } else {
            material.element.name.to_string()
        },
        params: MaterialParams {
            base_color: [
                base[0] * base_factor,
                base[1] * base_factor,
                base[2] * base_factor,
                opacity,
            ],
            metallic: real(&pbr.metalness, 0.0),
            roughness: real(&pbr.roughness, 0.7),
            emission_color: [emission_color[0], emission_color[1], emission_color[2]],
            emission_intensity: emission_factor,
            albedo_tiling,
            albedo_offset,
            normal_tiling,
            normal_offset,
            emission_tiling,
            emission_offset,
            pbr_toon_blend: 0.0,
            ..MaterialParams::default()
        },
        features: MaterialFeatures {
            toon: false,
            normal_map: normal.is_some(),
            emission: has_emission,
            alpha_mode: if opacity < 1.0 {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            },
            double_sided: material.features.double_sided.enabled,
            water: false,
            per_instance: false,
            ..Default::default()
        },
        albedo,
        normal,
        orm: None,
        emission: emission_tex,
    });
    let index = scene.materials.len() - 1;
    material_cache.insert(id, index);
    index
}

fn texture_data(texture: &ufbx::Texture, base_dir: &Path, srgb: bool) -> Result<TextureData> {
    let img = if !texture.content.is_empty() {
        // ufbx may have read an external file into `content`: track it for the import cache.
        for name in [texture.relative_filename.as_ref(), texture.filename.as_ref()] {
            let p = base_dir.join(name);
            if !name.is_empty() && p.is_file() {
                crate::import_cache::record_dependency(&p);
            }
        }
        image::load_from_memory(&texture.content).context("decoding embedded FBX texture")?
    } else {
        let mut candidates = Vec::new();
        if !texture.absolute_filename.is_empty() {
            candidates.push(std::path::PathBuf::from(texture.absolute_filename.as_ref()));
        }
        if !texture.relative_filename.is_empty() {
            candidates.push(base_dir.join(texture.relative_filename.as_ref()));
        }
        if !texture.filename.is_empty() {
            candidates.push(base_dir.join(texture.filename.as_ref()));
        }
        let path = candidates
            .into_iter()
            .find(|p| p.exists())
            .context("FBX texture file not found on disk")?;
        crate::import_cache::record_dependency(&path);
        image::open(&path).with_context(|| format!("loading texture {}", path.display()))?
    };
    let img = img.into_rgba8();
    Ok(TextureData {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
        srgb,
        hdr: false,
    })
}

/// ufbx matrices are column-major 4x3 (rotation+translation).
pub(crate) fn matrix_to_mat4(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols_array(&[
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0,
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0,
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0,
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0,
    ])
}

/// Blend shapes of one material part, keyed by soup corner. `control_points`
/// is the ufbx control point behind every corner, in soup order.
fn part_morphs(mesh: &ufbx::Mesh, control_points: &[u32]) -> Vec<crate::MorphTarget> {
    let channels: Vec<&ufbx::BlendChannel> = mesh.blend_deformers.iter().flat_map(|d| d.channels.iter()).map(|c| &**c).collect();
    if channels.is_empty() {
        return Vec::new();
    }
    // Corners of every control point (CSR, corner ascending), so a shape visits
    // only the corners its offsets touch instead of scanning all of them per shape.
    let points = control_points.iter().max().map_or(0, |m| *m as usize + 1);
    let mut start = vec![0u32; points + 1];
    for &cp in control_points {
        start[cp as usize + 1] += 1;
    }
    for i in 0..points {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut corners = vec![0u32; control_points.len()];
    for (v, &cp) in control_points.iter().enumerate() {
        corners[fill[cp as usize] as usize] = v as u32;
        fill[cp as usize] += 1;
    }
    // Last offset index per control point (a repeated vertex: the last one wins).
    let mut last = vec![u32::MAX; points];
    let mut out = Vec::new();
    for channel in channels {
      for (shape, frame) in crate::fbx_anim::channel_frames(channel) {
        let touched = |k: usize| (shape.offset_vertices[k] as usize) < points;
        for k in (0..shape.num_offsets).filter(|&k| touched(k)) {
            last[shape.offset_vertices[k] as usize] = k as u32;
        }
        let mut deltas: Vec<(u32, [f32; 3], [f32; 3])> = Vec::new();
        for k in (0..shape.num_offsets).filter(|&k| touched(k)) {
            let cp = shape.offset_vertices[k] as usize;
            if last[cp] != k as u32 {
                continue;
            }
            let p = shape.position_offsets[k];
            let n = shape.normal_offsets.get(k).map(|n| [n.x as f32, n.y as f32, n.z as f32]).unwrap_or_default();
            let (p, n) = ([p.x as f32, p.y as f32, p.z as f32], n);
            deltas.extend(corners[start[cp] as usize..start[cp + 1] as usize].iter().map(|&v| (v, p, n)));
        }
        for k in (0..shape.num_offsets).filter(|&k| touched(k)) {
            last[shape.offset_vertices[k] as usize] = u32::MAX;
        }
        if deltas.is_empty() {
            continue;
        }
        deltas.sort_unstable_by_key(|d| d.0);
        let name = channel.element.name.to_string();
        let name = name.rsplit(['.', '|']).next().unwrap_or(&name).to_owned();
        let name = match frame {
            Some(w) => crate::anim_import::inbetween_name(&name, w),
            None => name,
        };
        out.push(crate::MorphTarget { name, deltas });
      }
    }
    out
}

/// Cameras and lights of the file, as [`crate::ImportedCamera`] / [`crate::ImportedLight`]
/// in world space (scale is applied afterwards with the rest of the scene).
fn read_cameras_and_lights(fbx: &ufbx::Scene, settings: &ModelImport, scene: &mut Scene) {
    use crate::{ImportedCamera, ImportedLight};
    use zestengine_core::LightKind;
    for node in &fbx.nodes {
        let name = if node.element.name.is_empty() { format!("node {}", node.element.element_id) } else { node.element.name.to_string() };
        let world = matrix_to_mat4(&node.node_to_world);
        if settings.import_cameras
            && let Some(cam) = &node.camera
        {
            // FBX cameras look down their local +X; ours look down -Z.
            let aim = Mat4::from_quat(Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::X));
            let fov = cam.field_of_view_deg.y as f32;
            scene.identity.cameras.push(ImportedCamera {
                name: name.clone(),
                transform: (world * aim).to_cols_array(),
                fov_deg: if fov.is_finite() && fov > 1.0 { fov.min(170.0) } else { 60.0 },
                near: (cam.near_plane as f32).max(1e-4),
                far: (cam.far_plane as f32).max(0.01),
            });
        }
        if settings.import_lights
            && let Some(light) = &node.light
        {
            let kind = match light.type_ {
                ufbx::LightType::Directional => LightKind::Directional,
                ufbx::LightType::Spot => LightKind::Spot,
                // Area and volume lights have no equivalent: approximate with a point light.
                _ => LightKind::Point,
            };
            let dir = ufbx_v3(light.local_direction).normalize_or(Vec3::NEG_Y);
            let aim = Mat4::from_quat(Quat::from_rotation_arc(Vec3::NEG_Z, dir));
            // FBX intensity is a percentage; the engine's default light is 3.
            let intensity = (light.intensity as f32 / 100.0 * 3.0).max(0.0);
            scene.identity.lights.push(ImportedLight {
                name,
                transform: (world * aim).to_cols_array(),
                kind,
                color: [light.color.x as f32, light.color.y as f32, light.color.z as f32],
                intensity,
                range: 10.0,
                spot_angle_deg: if light.outer_angle > 0.0 { light.outer_angle as f32 } else { 45.0 },
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MorphTarget;
    use zestengine_render::MeshData;

    fn vert(p: [f32; 3]) -> Vertex {
        Vertex { position: p, normal: [0.0, 1.0, 0.0], uv: [0.0; 2], color: [1.0; 4], tangent: [1.0, 0.0, 0.0, 1.0], uv1: [0.0; 2], joints: [0; 4], weights: [1.0, 0.0, 0.0, 0.0] }
    }

    /// Two nodes at different translations plus a 2-joint skinned mesh.
    fn sample_scene() -> Scene {
        let skeleton = Skeleton { joints: vec![
            Joint { name: "root".into(), parent: None, inverse_bind: Mat4::IDENTITY, rest_translation: Vec3::new(0.0, 100.0, 0.0), rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE },
            Joint { name: "arm".into(), parent: Some(0), inverse_bind: Mat4::from_translation(Vec3::new(0.0, -150.0, 0.0)), rest_translation: Vec3::new(50.0, 0.0, 0.0), rest_rotation: Quat::from_rotation_z(0.5), rest_scale: Vec3::ONE },
        ], ..Default::default() };
        Scene {
            meshes: vec![MeshData { vertices: vec![vert([100.0, 0.0, 0.0]), vert([0.0, 200.0, 0.0]), vert([0.0, 0.0, 300.0])], indices: vec![0, 1, 2], has_lightmap_uv: false }],
            morphs: vec![vec![MorphTarget { name: "smile".into(), deltas: vec![(0, [10.0, 20.0, 30.0], [0.5, 0.5, 0.0])] }]],
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: vec![
                Instance::single("A", 0, 0, Mat4::from_translation(Vec3::new(1000.0, 0.0, 0.0)) * Mat4::from_rotation_y(0.3)),
                Instance::single("B", 0, 0, Mat4::from_scale(Vec3::splat(2.0)) * Mat4::from_translation(Vec3::new(0.0, 500.0, 0.0))),
            ],
            skeletons: vec![skeleton],
            animations: vec![AnimationClip {
                name: "walk".into(),
                duration: 1.0,
                channels: vec![
                    AnimChannel { joint: 1, path: ChannelPath::Translation, times: vec![0.0, 1.0], vec_values: vec![Vec3::new(50.0, 0.0, 0.0), Vec3::new(60.0, 10.0, 0.0)], quat_values: Vec::new(), extra: None },
                    AnimChannel { joint: 1, path: ChannelPath::Scale, times: vec![0.0], vec_values: vec![Vec3::splat(2.0)], quat_values: Vec::new(), extra: None },
                ],
                morph_tracks: Vec::new(), playback: None,
            }],
            collision: Vec::new(),
            identity: Default::default(),
        }
    }

    #[test]
    fn scale_applies_to_node_transforms_not_only_vertices() {
        let mut s = sample_scene();
        apply_scale(&mut s, 0.01);
        assert!((s.meshes[0].vertices[0].position[0] - 1.0).abs() < 1e-5);
        assert!((s.instances[0].transform.w_axis.truncate() - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4, "node translation scaled");
        assert!((s.instances[1].transform.w_axis.truncate() - Vec3::new(0.0, 10.0, 0.0)).length() < 1e-4, "translation of a scaled node");
        // Rotation and scale parts of a node are untouched.
        let (sc, _, _) = s.instances[1].transform.to_scale_rotation_translation();
        assert!((sc - Vec3::splat(2.0)).length() < 1e-5);
        assert!((s.morphs[0][0].deltas[0].1[2] - 0.3).abs() < 1e-6, "blendshape position delta scaled");
        assert_eq!(s.morphs[0][0].deltas[0].2, [0.5, 0.5, 0.0], "normal delta is unitless");
    }

    #[test]
    fn scale_applies_to_bones_and_animation() {
        let mut s = sample_scene();
        apply_scale(&mut s, 0.01);
        let j = &s.skeletons[0].joints;
        assert!((j[0].rest_translation - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((j[1].rest_translation - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-6);
        assert!((j[1].inverse_bind.w_axis.truncate() - Vec3::new(0.0, -1.5, 0.0)).length() < 1e-6);
        assert_eq!(j[1].rest_rotation, Quat::from_rotation_z(0.5), "rotation untouched");
        let clip = &s.animations[0];
        assert!((clip.channels[0].vec_values[1] - Vec3::new(0.6, 0.1, 0.0)).length() < 1e-6);
        assert_eq!(clip.channels[1].vec_values[0], Vec3::splat(2.0), "scale keys are not lengths");
    }

    #[test]
    fn scaled_skinning_equals_scaled_original() {
        // Skin position = jointWorld * inverseBind * v. Scaling every length by s
        // must scale the skinned position by s.
        let skin = |sc: &Scene| {
            let j = &sc.skeletons[0].joints;
            let w0 = Mat4::from_scale_rotation_translation(j[0].rest_scale, j[0].rest_rotation, j[0].rest_translation);
            let w1 = w0 * Mat4::from_scale_rotation_translation(j[1].rest_scale, j[1].rest_rotation, j[1].rest_translation);
            (w1 * j[1].inverse_bind).transform_point3(Vec3::from(sc.meshes[0].vertices[0].position))
        };
        let a = skin(&sample_scene());
        let mut scaled = sample_scene();
        apply_scale(&mut scaled, 0.01);
        assert!((skin(&scaled) - a * 0.01).length() < 1e-4);
    }

    #[test]
    fn negative_scale_flips_winding_and_frame() {
        let mut s = sample_scene();
        let tri = s.meshes[0].indices[..3].to_vec();
        let (n0, t0) = (s.meshes[0].vertices[tri[0] as usize].normal, s.meshes[0].vertices[tri[0] as usize].tangent);
        let pos = |s: &Scene, k: u32| Vec3::from(s.meshes[0].vertices[k as usize].position);
        let face = |s: &Scene| {
            let i = &s.meshes[0].indices;
            (pos(s, i[1]) - pos(s, i[0])).cross(pos(s, i[2]) - pos(s, i[0]))
        };
        let before = face(&s);
        apply_scale(&mut s, -1.0);
        let after = face(&s);
        let v = &s.meshes[0].vertices[tri[0] as usize];
        assert!(after.dot(Vec3::from(v.normal)) > 0.0 || before.length() < 1e-9, "the winding faces the same way as the flipped normal");
        assert_eq!(v.normal, n0.map(|c| -c));
        assert_eq!(v.tangent[3], -t0[3]);
        assert_eq!(&s.meshes[0].indices[..3], &[tri[0], tri[2], tri[1]]);
    }

    #[test]
    fn unit_scale_is_a_no_op() {
        let mut s = sample_scene();
        let before = s.instances[0].transform;
        apply_scale(&mut s, 1.0);
        assert_eq!(s.instances[0].transform, before);
    }

    /// The same model imported at 1.0 and at 0.01 differs by exactly 0.01 in
    /// every length (real multi-node and, if present, skinned files).
    #[test]
    fn real_fbx_at_one_percent() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut tested = 0;
        for rel in ["models/leisure_area.FBX", "example-game/lemonjuice.fbx"] {
            let path = root.join(rel);
            if !path.exists() {
                continue;
            }
            let base = load_fbx_with(&path, &ModelImport::default()).unwrap();
            let small = load_fbx_with(&path, &ModelImport { scale: 0.01, ..Default::default() }).unwrap();
            assert_eq!(base.instances.len(), small.instances.len());
            for (a, b) in base.instances.iter().zip(&small.instances) {
                let (ta, tb) = (a.transform.w_axis.truncate(), b.transform.w_axis.truncate());
                assert!((ta * 0.01 - tb).length() <= 1e-3 + ta.length() * 1e-5, "{}: node translation scales", a.name);
            }
            for (a, b) in base.skeletons.iter().zip(&small.skeletons) {
                for (ja, jb) in a.joints.iter().zip(&b.joints) {
                    assert!((ja.rest_translation * 0.01 - jb.rest_translation).length() <= 1e-3 + ja.rest_translation.length() * 1e-5);
                    assert!((ja.inverse_bind.w_axis.truncate() * 0.01 - jb.inverse_bind.w_axis.truncate()).length() <= 1e-3 + ja.inverse_bind.w_axis.length() * 1e-5);
                }
            }
            tested += 1;
        }
        if tested == 0 {
            eprintln!("no sample FBX in the repo; skipped");
        }
    }

    #[test]
    fn recalculate_normals_setting_is_applied_and_names_are_unique() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("models/leisure_area.FBX");
        if !path.exists() {
            return;
        }
        let scene = load_fbx_with(&path, &ModelImport { recalculate_normals: true, smoothing_angle: 60.0, ..Default::default() }).unwrap();
        assert_eq!(scene.identity.mesh_names.len(), scene.meshes.len());
        let unique: std::collections::HashSet<_> = scene.identity.mesh_names.iter().collect();
        assert_eq!(unique.len(), scene.meshes.len(), "every mesh gets a unique name");
        for m in scene.meshes.iter().take(5) {
            assert!(m.vertices.iter().all(|v| (Vec3::from(v.normal).length() - 1.0).abs() < 1e-3));
        }
    }

    #[test]
    fn animation_is_sampled_once_per_source_frame() {
        // 1 s at 24 fps: 24 intervals, 25 samples, the last exactly at the end.
        let t = sample_times(1.0, 24.0);
        assert_eq!(t.len(), 25);
        assert_eq!((t[0], t[24]), (0.0, 1.0));
        assert!((t[1] - 1.0 / 24.0).abs() < 1e-12 && (t[12] - 0.5).abs() < 1e-12);
        // 0.5 s at 60 fps.
        assert_eq!(sample_times(0.5, 60.0).len(), 31);
        // Not a whole number of frames: one more sample, clamped to the clip end.
        let t = sample_times(1.01, 30.0);
        assert_eq!(t.len(), 32);
        assert_eq!(*t.last().unwrap(), 1.01);
        assert!(t.windows(2).all(|w| w[1] > w[0]), "strictly increasing");
        // The old fixed 30 fps would have produced 31 samples for a 24 fps file.
        assert_ne!(sample_times(1.0, 24.0).len(), sample_times(1.0, 30.0).len());
    }

    #[test]
    fn a_bad_frame_rate_falls_back_to_30() {
        assert_eq!(source_frame_rate(24.0), 24.0);
        assert_eq!(source_frame_rate(0.0), 30.0);
        assert_eq!(source_frame_rate(-5.0), 30.0);
        assert_eq!(source_frame_rate(f64::NAN), 30.0);
        assert_eq!(source_frame_rate(1e9), 960.0);
    }

    /// Real FBX: the default import is indexed and welded, and the pipeline
    /// settings reach the output.
    #[test]
    fn real_fbx_is_welded_and_the_settings_change_the_output() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("models/leisure_area.FBX");
        if !path.exists() {
            return;
        }
        let welded = load_fbx_with(&path, &ModelImport::default()).unwrap();
        let raw = load_fbx_with(&path, &ModelImport { weld_vertices: false, optimize_mesh: false, ..Default::default() }).unwrap();
        assert_eq!(welded.meshes.len(), raw.meshes.len());
        let (wv, rv): (usize, usize) = (welded.meshes.iter().map(|m| m.vertices.len()).sum(), raw.meshes.iter().map(|m| m.vertices.len()).sum());
        let (wi, ri): (usize, usize) = (welded.meshes.iter().map(|m| m.indices.len()).sum(), raw.meshes.iter().map(|m| m.indices.len()).sum());
        assert!(wv < rv, "welding shares vertices: {wv} < {rv}");
        assert!(wi <= ri && wi > 0, "no triangle appears; degenerates may go: {wi} vs {ri}");
        assert!(raw.meshes.iter().all(|m| m.vertices.len() == m.indices.len()), "unwelded stays one vertex per corner");
        assert!(welded.meshes.iter().all(|m| m.indices.iter().all(|&i| (i as usize) < m.vertices.len())));
    }

    /// Both axis-conversion modes place the model in the same world space.
    #[test]
    fn bake_axis_conversion_keeps_the_world_bounds() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("models/leisure_area.FBX");
        if !path.exists() {
            return;
        }
        let bounds = |s: &Scene| -> (Vec3, Vec3) {
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for inst in &s.instances {
                for slot in &inst.slots {
                    for v in &s.meshes[slot.mesh].vertices {
                        let p = inst.transform.transform_point3(Vec3::from(v.position));
                        lo = lo.min(p);
                        hi = hi.max(p);
                    }
                }
            }
            (lo, hi)
        };
        let off = load_fbx_with(&path, &ModelImport::default()).unwrap();
        let on = load_fbx_with(&path, &ModelImport { bake_axis_conversion: true, ..Default::default() }).unwrap();
        let (a, b) = (bounds(&off), bounds(&on));
        let size = (a.1 - a.0).length().max(1.0);
        assert!((a.0 - b.0).length() < size * 1e-3 && (a.1 - b.1).length() < size * 1e-3, "{a:?} vs {b:?}");
    }
}


