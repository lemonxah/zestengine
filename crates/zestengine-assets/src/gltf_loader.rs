//! glTF 2.0 import into a [`Scene`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use zestengine_render::{AlphaMode, MaterialFeatures, MaterialParams, Vertex};
use glam::{Mat4, Quat, Vec3};

use crate::skeleton::{AnimChannel, AnimationClip, ChannelPath, Joint, Skeleton};
use crate::mesh_pipeline::{ImportMesh, Settings as PipelineSettings};
use crate::{ImportedCamera, ImportedLight, Instance, MeshSlot, ModelImport, Scene, SceneMaterial};

/// Import a glTF/GLB/VRM with default settings.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<Scene> {
    load_gltf_with(path, &ModelImport::default())
}

/// Import a glTF/GLB/VRM, applying the asset's [`ModelImport`] settings: the
/// mesh pipeline (normals, tangents, weld, optimise, ...), scale, flip-UV,
/// import-materials, blendshapes, cameras and lights. The FBX-only settings
/// (axis conversion) do not apply.
pub fn load_gltf_with(path: impl AsRef<Path>, settings: &ModelImport) -> Result<Scene> {
    let path = path.as_ref();
    // Same validation a `.meta` load runs, for settings built in code.
    let mut settings = settings.clone();
    settings.validate();
    let settings = &settings;
    let pipeline = PipelineSettings::from_import(settings);
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("importing glTF file {}", path.display()))?;

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
    // (image index, srgb) -> scene texture index. The same image can be
    // legally referenced as both color and data; cache per usage.
    let mut texture_cache: HashMap<(usize, bool), usize> = HashMap::new();

    // Source file stem of each image, kept for Search and Remap by texture name.
    let image_names: Vec<String> = doc
        .images()
        .map(|img| match img.source() {
            gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => crate::material_remap::texture_stem(uri),
            _ => img.name().map(crate::material_remap::texture_stem).unwrap_or_default(),
        })
        .collect();

    let mut get_texture = |scene: &mut Scene, image_index: usize, srgb: bool| -> Result<usize> {
        if let Some(&index) = texture_cache.get(&(image_index, srgb)) {
            return Ok(index);
        }
        let data = rgba8_from_gltf(&images[image_index], srgb)?;
        scene.textures.push(data);
        scene.identity.texture_names.push(image_names.get(image_index).cloned().unwrap_or_default());
        let index = scene.textures.len() - 1;
        texture_cache.insert((image_index, srgb), index);
        Ok(index)
    };

    // Materials. Index i = glTF material i; one extra default at the end.
    let mut volume_maps: Vec<Option<usize>> = Vec::new();
    for material in doc.materials() {
        let pbr = material.pbr_metallic_roughness();

        let albedo_info = pbr.base_color_texture();
        let albedo = albedo_info.as_ref()
            .map(|t| get_texture(&mut scene, t.texture().source().index(), true))
            .transpose()?;
        let orm_info = pbr.metallic_roughness_texture();
        let orm = orm_info
            .as_ref()
            .map(|t| get_texture(&mut scene, t.texture().source().index(), false))
            .transpose()?;
        let normal_info = material.normal_texture();
        let normal = normal_info
            .as_ref()
            .map(|t| get_texture(&mut scene, t.texture().source().index(), false))
            .transpose()?;
        let emission_info = material.emissive_texture();
        let emission_tex = emission_info
            .as_ref()
            .map(|t| get_texture(&mut scene, t.texture().source().index(), true))
            .transpose()?;
        // KHR_texture_transform (Blender's mapping-node tiling): the scale is
        // the engine's tiling and the offset is the engine's offset, per slot.
        let tex_uv = |info: Option<&gltf::texture::Info<'_>>| -> ([f32; 2], [f32; 2]) {
            let Some(t) = info.and_then(|t| t.texture_transform()) else { return ([1.0; 2], [0.0; 2]) };
            (t.scale(), t.offset())
        };
        // The gltf crate exposes the transform extension on `Info` only; the
        // normal texture's copy lives in the raw extension JSON.
        let normal_uv = |info: Option<&gltf::material::NormalTexture<'_>>| -> ([f32; 2], [f32; 2]) {
            let Some(v) = info.and_then(|n| n.extension_value("KHR_texture_transform")) else { return ([1.0; 2], [0.0; 2]) };
            let f = |k: &str, d: [f32; 2]| {
                v.get(k).and_then(|x| x.as_array()).and_then(|a| Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32])).unwrap_or(d)
            };
            (f("scale", [1.0; 2]), f("offset", [0.0; 2]))
        };
        let (albedo_tiling, albedo_offset) = tex_uv(albedo_info.as_ref());
        let (orm_tiling, orm_offset) = tex_uv(orm_info.as_ref());
        let (normal_tiling, normal_offset) = normal_uv(normal_info.as_ref());
        let (emission_tiling, emission_offset) = tex_uv(emission_info.as_ref());

        // Occlusion is only honored when packed into the same image as
        // metallic-roughness (the common ORM layout); a separate occlusion
        // texture would need its own slot.
        let occlusion_strength = match (material.occlusion_texture(), &orm_info) {
            (Some(occ), Some(mr))
                if occ.texture().source().index() == mr.texture().source().index() =>
            {
                occ.strength()
            }
            _ => 0.0,
        };

        let emissive_factor = material.emissive_factor();
        let emissive_strength = material.emissive_strength().unwrap_or(1.0);
        let has_emission = emission_tex.is_some() || emissive_factor.iter().any(|&c| c > 0.0);

        let (alpha_mode, alpha_cutoff) = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => (AlphaMode::Opaque, 0.5),
            gltf::material::AlphaMode::Mask => {
                (AlphaMode::Cutout, material.alpha_cutoff().unwrap_or(0.5))
            }
            gltf::material::AlphaMode::Blend => (AlphaMode::Blend, 0.5),
        };

        // KHR_materials_transmission / KHR_materials_volume: the see-through share
        // and the volume's attenuation colour over its thickness (Beer-Lambert,
        // tint = colour^(thickness / distance)). The transmission texture (R) and
        // the thickness texture (G) are repacked into one map (R thickness, G
        // transmission) that rides in the translucency / thickness slot.
        let transmission = material.transmission().map_or(0.0, |t| t.transmission_factor());
        let mut tint = [1.0f32; 3];
        let mut thickness = 0.0f32;
        let mut volume_map = None;
        if transmission > 0.0 {
            let tex_image = |t: Option<gltf::texture::Info<'_>>| t.map(|i| i.texture().source().index());
            let thickness_image = material.volume().and_then(|v| v.thickness_texture()).map(|i| i.texture().source().index());
            let transmission_image = material.transmission().and_then(|t| tex_image(t.transmission_texture()));
            if thickness_image.is_some() || transmission_image.is_some() {
                let load = |i: Option<usize>| i.map(|i| rgba8_from_gltf(&images[i], false)).transpose();
                let packed = pack_volume_map(load(thickness_image)?.as_ref(), load(transmission_image)?.as_ref());
                scene.textures.push(packed);
                scene.identity.texture_names.resize(scene.textures.len() - 1, String::new());
                scene.identity.texture_names.push(format!("{}_volume", material.name().unwrap_or("material")));
                volume_map = Some(scene.textures.len() - 1);
            }
        }
        if let Some(v) = material.volume() {
            // Unit-free thickness 0..1 for the blur: 1 - exp(-2 * metres).
            thickness = 1.0 - (-2.0 * v.thickness_factor().max(0.0)).exp();
            let dist = v.attenuation_distance();
            if dist.is_finite() && dist > 0.0 {
                let e = v.thickness_factor() / dist;
                let c = v.attenuation_color();
                tint = [c[0].powf(e), c[1].powf(e), c[2].powf(e)];
            }
        }

        volume_maps.push(volume_map);
        scene.materials.push(SceneMaterial {
            name: material
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("material {}", material.index().unwrap_or(0))),
            params: MaterialParams {
                base_color: pbr.base_color_factor(),
                emission_color: emissive_factor,
                emission_intensity: emissive_strength,
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                alpha_cutoff,
                normal_strength: normal_info.as_ref().map_or(1.0, |n| n.scale()),
                occlusion_strength,
                albedo_tiling,
                albedo_offset,
                normal_tiling,
                normal_offset,
                orm_tiling,
                orm_offset,
                emission_tiling,
                emission_offset,
                pbr_toon_blend: 0.0, // glTF imports as pure PBR; toon is opt-in
                transmission: zestengine_render::TransmissionParams {
                    transmission,
                    tint,
                    thickness,
                    // With a packed map: red is the thickness texture when one was given.
                    map_is_thickness: volume_map.is_some() && material.volume().is_some_and(|v| v.thickness_texture().is_some()),
                    map_has_transmission: volume_map.is_some() && material.transmission().is_some_and(|t| t.transmission_texture().is_some()),
                    ..Default::default()
                },
                ..MaterialParams::default()
            },
            features: MaterialFeatures {
                toon: false,
                normal_map: normal.is_some(),
                emission: has_emission,
                alpha_mode,
                double_sided: material.double_sided(),
                water: false,
                per_instance: false,
                // Transmission samples the opaque scene copy.
                grab: transmission > 0.0,
                ..Default::default()
            },
            albedo,
            normal,
            orm,
            emission: emission_tex,
        });
    }
    // Fallback for primitives without a material.
    volume_maps.push(None);
    scene.materials.push(SceneMaterial {
        name: "default".into(),
        params: MaterialParams::default(),
        features: MaterialFeatures::default(),
        albedo: None,
        normal: None,
        orm: None,
        emission: None,
    });
    let default_material = scene.materials.len() - 1;
    scene.identity.material_volume_maps = volume_maps;

    // Meshes: each glTF primitive becomes one MeshData.
    // (gltf mesh index, primitive index) -> (scene mesh, scene material)
    let mut primitive_map: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    // Mesh names (unique among siblings) feed the stable sub-asset ids.
    let mut namer = crate::sub_asset::SiblingNamer::default();
    for mesh in doc.meshes() {
        let mesh_name = namer.unique("", &mesh.name().map(str::to_owned).unwrap_or_else(|| format!("mesh{}", mesh.index())));
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                tracing::warn!(
                    mesh = mesh.name().unwrap_or("?"),
                    "skipping non-triangle primitive"
                );
                continue;
            }
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let mut uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            if settings.flip_uv {
                for uv in &mut uvs {
                    uv[1] = 1.0 - uv[1];
                }
            }
            // Second UV set = lightmap UVs; fall back to the primary set when
            // the model has no TEXCOORD_1. `real_uv1` distinguishes a genuine
            // non-overlapping lightmap UV from the uv0 fallback (not bakeable).
            let real_uv1 = reader.read_tex_coords(1).is_some();
            let uvs1: Vec<[f32; 2]> = reader
                .read_tex_coords(1)
                .map(|t| t.into_f32().collect())
                .unwrap_or_else(|| uvs.clone());
            let colors: Vec<[f32; 4]> = reader
                .read_colors(0)
                .map(|c| c.into_rgba_f32().collect())
                .unwrap_or_else(|| vec![[1.0; 4]; positions.len()]);
            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|t| t.collect())
                .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]);
            // Skinning: joint indices (skin-local) + weights. Absent on static
            // meshes → zero (the skinned pipeline variant is unused for them).
            let joints: Vec<[u32; 4]> = reader
                .read_joints(0)
                .map(|j| {
                    j.into_u16()
                        .map(|[a, b, c, d]| [a as u32, b as u32, c as u32, d as u32])
                        .collect()
                })
                .unwrap_or_else(|| vec![[0u32; 4]; positions.len()]);
            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|w| w.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0f32; 4]; positions.len()]);

            let has_normals = reader.read_normals().is_some();
            let has_tangents = reader.read_tangents().is_some();
            let vertices: Vec<Vertex> = (0..positions.len())
                .map(|i| Vertex {
                    position: positions[i],
                    normal: normals[i],
                    uv: uvs[i],
                    color: colors[i],
                    tangent: tangents[i],
                    uv1: uvs1[i],
                    joints: joints[i],
                    weights: weights[i],
                })
                .collect();
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..vertices.len() as u32).collect());

            let morphs = if settings.import_blend_shapes { read_morphs(&reader, &mesh, &primitive, vertices.len()) } else { Vec::new() };
            // The pipeline decides whether uv1 is a real lightmap UV (a genuine
            // TEXCOORD_1) or a copy of uv0, which is still bakeable IF uv0 is a
            // non-overlapping atlas (most models); only overlapping/tiled uv0
            // needs a generated set.
            let mut soup = ImportMesh::from_indexed(&vertices, &indices, morphs);
            soup.has_normals = has_normals;
            soup.has_tangents = has_tangents;
            soup.real_uv1 = real_uv1;
            // A forced 16 bit index format can cut a big primitive into several meshes.
            let pieces = crate::mesh_pipeline::run_split(soup, &pipeline);
            let material = primitive.material().index().unwrap_or(default_material);
            let mut slots = Vec::new();
            for (k, imported) in pieces.into_iter().enumerate() {
                if !imported.morphs.is_empty() {
                    scene.morphs.resize(scene.meshes.len(), Vec::new());
                    scene.morphs.push(imported.morphs);
                }
                scene.meshes.push(imported.mesh);
                let index = scene.meshes.len() - 1;
                scene.identity.set_quads(index, imported.quads);
                scene.identity.mesh_names.push(if k == 0 { format!("{mesh_name}#part{}", primitive.index()) } else { format!("{mesh_name}#part{}~s{k}", primitive.index()) });
                slots.push((index, material));
            }
            if !slots.is_empty() {
                primitive_map.insert((mesh.index(), primitive.index()), slots);
            }
        }
    }

    // Skeletons (armatures) — one per glTF skin. Joints stay in skin order so
    // the skin-local vertex joint indices line up. Parent links come from the
    // node hierarchy (node -> parent node, mapped back to joint index).
    let mut node_parent: HashMap<usize, usize> = HashMap::new();
    for node in doc.nodes() {
        for child in node.children() {
            node_parent.insert(child.index(), node.index());
        }
    }
    for skin in doc.skins() {
        let reader = skin.reader(|b| Some(&buffers[b.index()]));
        let ibms: Vec<Mat4> = reader
            .read_inverse_bind_matrices()
            .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_default();
        let joint_nodes: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        let node_to_joint: HashMap<usize, usize> =
            joint_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut joints = Vec::with_capacity(joint_nodes.len());
        for (i, node) in skin.joints().enumerate() {
            let (t, r, s) = node.transform().decomposed();
            let parent = node_parent
                .get(&node.index())
                .and_then(|p| node_to_joint.get(p))
                .copied();
            joints.push(Joint {
                name: node.name().unwrap_or("joint").to_string(),
                parent,
                inverse_bind: ibms.get(i).copied().unwrap_or(Mat4::IDENTITY),
                rest_translation: Vec3::from(t),
                rest_rotation: Quat::from_array(r),
                rest_scale: Vec3::from(s),
            });
        }
        let mut skeleton = Skeleton { joints, rig: None, ..Default::default() };
        // The nodes above the topmost joints (an `Armature` node, the VRM scene
        // root) stay in the skeleton as container joints, as Unity keeps them
        // above the Hips: only when every parentless joint shares one chain.
        let roots: Vec<usize> = (0..skeleton.joints.len()).filter(|&j| skeleton.joints[j].parent.is_none()).collect();
        let chain_of = |mut node: usize| -> Vec<usize> {
            let mut ids = Vec::new();
            while let Some(&p) = node_parent.get(&node) {
                ids.push(p);
                node = p;
            }
            ids.reverse();
            ids
        };
        if let Some(&first) = roots.first() {
            let chain = chain_of(joint_nodes[first]);
            if !chain.is_empty() && roots.iter().all(|&j| chain_of(joint_nodes[j]) == chain) {
                let spec: Vec<(String, Vec3, Quat, Vec3)> = chain
                    .iter()
                    .filter_map(|&n| doc.nodes().nth(n))
                    .map(|n| {
                        let (t, r, s) = n.transform().decomposed();
                        (n.name().unwrap_or("node").to_string(), Vec3::from(t), Quat::from_array(r), Vec3::from(s))
                    })
                    .collect();
                if spec.len() == chain.len() {
                    // The skinned node keeps its own transform: `Skeleton::world_matrices`
                    // takes the containers' rest basis out of the pose.
                    skeleton.add_containers(&roots, &spec);
                }
            }
        }
        scene.skeletons.push(skeleton);
    }

    // Animations: per-joint TRS channels. Joint indices are resolved against the
    // first skin (the common single-armature case); channels targeting nodes not
    // in that skin are skipped.
    if let Some(skin) = doc.skins().next() {
        let node_to_joint: HashMap<usize, usize> = skin
            .joints()
            .enumerate()
            .map(|(i, j)| (j.index(), i))
            .collect();
        for anim in doc.animations() {
            let mut channels = Vec::new();
            let mut duration = 0.0f32;
            for ch in anim.channels() {
                let Some(&joint) = node_to_joint.get(&ch.target().node().index()) else {
                    continue;
                };
                let reader = ch.reader(|b| Some(&buffers[b.index()]));
                let times: Vec<f32> = match reader.read_inputs() {
                    Some(it) => it.collect(),
                    None => continue,
                };
                if let Some(last) = times.last() {
                    duration = duration.max(*last);
                }
                use gltf::animation::util::ReadOutputs;
                let (path, vecs, quats) = match reader.read_outputs() {
                    Some(ReadOutputs::Translations(it)) => {
                        (ChannelPath::Translation, it.map(Vec3::from).collect(), Vec::new())
                    }
                    Some(ReadOutputs::Scales(it)) => {
                        (ChannelPath::Scale, it.map(Vec3::from).collect(), Vec::new())
                    }
                    Some(ReadOutputs::Rotations(it)) => (
                        ChannelPath::Rotation,
                        Vec::new(),
                        it.into_f32().map(Quat::from_array).collect(),
                    ),
                    _ => continue, // morph-target weights unsupported
                };
                channels.push(AnimChannel {
                    joint,
                    path,
                    times,
                    vec_values: vecs,
                    quat_values: quats, extra: None
                });
            }
            if !channels.is_empty() {
                scene.animations.push(AnimationClip {
                    name: anim.name().unwrap_or("clip").to_string(),
                    duration,
                    channels,
                    morph_tracks: Vec::new(), playback: None,
                });
            }
        }
    }

    // Instances from the node hierarchy.
    let gltf_scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .context("glTF file contains no scenes")?;
    for node in gltf_scene.nodes() {
        visit_node(&node, Mat4::IDENTITY, &primitive_map, &mut scene.instances);
        if settings.import_cameras || settings.import_lights {
            visit_extras(&node, Mat4::IDENTITY, settings, &mut scene.identity);
        }
    }

    if scene.instances.is_empty() {
        bail!("glTF scene produced no renderable instances");
    }
    scene.springs = crate::vrm::read_springs(path);
    crate::rig_import::attach_vrm_rig(path, &mut scene);
    if settings.scale != 1.0 {
        crate::fbx_loader::apply_scale(&mut scene, settings.scale);
    }
    if !settings.import_materials {
        // Keep geometry, fall back to plain defaults.
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
        "glTF scene loaded"
    );
    Ok(scene)
}

fn visit_node(
    node: &gltf::Node<'_>,
    parent: Mat4,
    primitive_map: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    instances: &mut Vec<Instance>,
) {
    let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        let base_name = node
            .name()
            .or_else(|| mesh.name())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("node {}", node.index()));
        // All of the mesh's primitives become material slots on one instance.
        let slots: Vec<MeshSlot> = mesh
            .primitives()
            .filter_map(|primitive| primitive_map.get(&(mesh.index(), primitive.index())))
            .flatten()
            .map(|&(mesh, material)| MeshSlot { mesh, material })
            .collect();
        if !slots.is_empty() {
            instances.push(Instance {
                name: base_name,
                transform,
                slots,
            });
        }
    }
    for child in node.children() {
        visit_node(&child, transform, primitive_map, instances);
    }
}

/// Cameras and lights of the node tree, in world space.
fn visit_extras(node: &gltf::Node<'_>, parent: Mat4, settings: &ModelImport, out: &mut crate::ModelIdentity) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    let name = node.name().map(str::to_owned).unwrap_or_else(|| format!("node {}", node.index()));
    if settings.import_cameras
        && let Some(cam) = node.camera()
        && let gltf::camera::Projection::Perspective(p) = cam.projection()
    {
        // glTF cameras already look down -Z.
        out.cameras.push(ImportedCamera {
            name: name.clone(),
            transform: world.to_cols_array(),
            fov_deg: p.yfov().to_degrees().clamp(1.0, 170.0),
            near: p.znear().max(1e-4),
            far: p.zfar().unwrap_or(1000.0).max(0.01),
        });
    }
    if settings.import_lights
        && let Some(light) = node.light()
    {
        use gltf::khr_lights_punctual::Kind;
        let (kind, spot) = match light.kind() {
            Kind::Directional => (zestengine_core::LightKind::Directional, 45.0),
            Kind::Point => (zestengine_core::LightKind::Point, 45.0),
            Kind::Spot { outer_cone_angle, .. } => (zestengine_core::LightKind::Spot, (outer_cone_angle * 2.0).to_degrees().clamp(1.0, 179.0)),
        };
        out.lights.push(ImportedLight {
            name: name.clone(),
            transform: world.to_cols_array(),
            kind,
            color: light.color(),
            // glTF intensities are photometric (lux / candela); the engine's
            // are a radiance multiplier around 1-10, so cap the range.
            intensity: light.intensity().clamp(0.0, 20.0),
            range: light.range().unwrap_or(10.0),
            spot_angle_deg: spot,
        });
    }
    for child in node.children() {
        visit_extras(&child, world, settings, out);
    }
}

/// glTF `thicknessTexture` (green) and `transmissionTexture` (red) as one RGBA8 map:
/// red = thickness, green = transmission (255 where a texture is missing), sized
/// to the larger source; the smaller one is read nearest-neighbour.
pub(crate) fn pack_volume_map(
    thickness: Option<&zestengine_render::TextureData>,
    transmission: Option<&zestengine_render::TextureData>,
) -> zestengine_render::TextureData {
    let (w, h) = [thickness, transmission]
        .iter()
        .flatten()
        .fold((1, 1), |(w, h), t| (w.max(t.width), h.max(t.height)));
    let read = |t: Option<&zestengine_render::TextureData>, x: u32, y: u32, channel: usize| -> u8 {
        let Some(t) = t else { return 255 };
        let (sx, sy) = ((x * t.width / w).min(t.width - 1), (y * t.height / h).min(t.height - 1));
        t.pixels[((sy * t.width + sx) * 4) as usize + channel]
    };
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[read(thickness, x, y, 1), read(transmission, x, y, 0), 0, 255]);
        }
    }
    zestengine_render::TextureData { width: w, height: h, pixels, srgb: false, hdr: false }
}

pub(crate) fn rgba8_from_gltf(data: &gltf::image::Data, srgb: bool) -> Result<zestengine_render::TextureData> {
    use gltf::image::Format;
    let pixel_count = (data.width * data.height) as usize;
    let pixels = match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.as_chunks::<3>().0 {
                out.extend_from_slice(chunk);
                out.push(255);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &v in &data.pixels {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        Format::R8G8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.as_chunks::<2>().0 {
                out.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
            }
            out
        }
        other => bail!("unsupported glTF image format {other:?} (16-bit formats land in M3)"),
    };
    Ok(zestengine_render::TextureData {
        width: data.width,
        height: data.height,
        pixels,
        srgb,
        hdr: false,
    })
}

/// Blendshape names from `extras.targetNames` (on the primitive, else the mesh,
/// as exporters write them).
fn target_names(mesh: &gltf::Mesh, primitive: &gltf::Primitive) -> Vec<String> {
    let parse = |extras: &gltf::json::Extras| -> Option<Vec<String>> {
        let raw = extras.as_ref()?;
        let v: serde_json::Value = serde_json::from_str(raw.get()).ok()?;
        let names = v.get("targetNames")?.as_array()?;
        Some(names.iter().map(|n| n.as_str().unwrap_or_default().to_owned()).collect())
    };
    parse(primitive.extras()).or_else(|| parse(mesh.extras())).unwrap_or_default()
}

/// A primitive's morph targets, keeping only the vertices each one moves.
fn read_morphs<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    mesh: &gltf::Mesh,
    primitive: &gltf::Primitive,
    vertex_count: usize,
) -> Vec<crate::MorphTarget>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let names = target_names(mesh, primitive);
    let mut out = Vec::new();
    for (k, (positions, normals, _)) in reader.read_morph_targets().enumerate() {
        let p: Vec<[f32; 3]> = positions.map(|it| it.collect()).unwrap_or_default();
        let n: Vec<[f32; 3]> = normals.map(|it| it.collect()).unwrap_or_default();
        let mut deltas = Vec::new();
        for v in 0..vertex_count {
            let dp = p.get(v).copied().unwrap_or_default();
            let dn = n.get(v).copied().unwrap_or_default();
            if dp.iter().chain(&dn).any(|x| x.abs() > 1e-7) {
                deltas.push((v as u32, dp, dn));
            }
        }
        let name = names.get(k).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| format!("Shape {k}"));
        out.push(crate::MorphTarget { name, deltas });
    }
    out
}


#[cfg(test)]
mod transmission_tests {
    use super::*;

    /// The glTF thickness texture (green) and transmission texture (red) are repacked into one
    /// map: red = thickness, green = transmission, a missing one reads 255, the larger size wins.
    #[test]
    fn thickness_and_transmission_textures_pack_into_one_map() {
        let tex = |w: u32, h: u32, px: [u8; 4]| zestengine_render::TextureData { width: w, height: h, pixels: px.repeat((w * h) as usize), srgb: false, hdr: false };
        let thickness = tex(2, 2, [9, 200, 9, 255]);
        let transmission = tex(4, 4, [40, 7, 7, 255]);
        let both = super::pack_volume_map(Some(&thickness), Some(&transmission));
        assert_eq!((both.width, both.height), (4, 4));
        assert_eq!(&both.pixels[..4], &[200, 40, 0, 255], "R = thickness.G, G = transmission.R");
        let only_t = super::pack_volume_map(Some(&thickness), None);
        assert_eq!((only_t.width, &only_t.pixels[..4]), (2, &[200u8, 255, 0, 255][..]), "no transmission texture: G = 255");
        let only_x = super::pack_volume_map(None, Some(&transmission));
        assert_eq!(&only_x.pixels[..4], &[255, 40, 0, 255], "no thickness texture: R = 255");
    }

    /// `KHR_materials_transmission` and `KHR_materials_volume` import as the
    /// standard shader's transmission with Scene Colour Read on, and the volume's
    /// attenuation as a tint over the thickness.
    #[test]
    fn khr_transmission_and_volume_map_to_transmission() {
        let json = r#"{
          "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
          "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
          "buffers":[{"uri":"b.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteLength":36}],
          "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
          "extensionsUsed":["KHR_materials_transmission","KHR_materials_volume"],
          "materials":[
            {"name":"Glass","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"roughnessFactor":0.1},
             "extensions":{"KHR_materials_transmission":{"transmissionFactor":0.8},
                           "KHR_materials_volume":{"thicknessFactor":2.0,"attenuationDistance":2.0,"attenuationColor":[0.25,1.0,1.0]}}},
            {"name":"Plain"}]}"#;
        let dir = std::env::temp_dir().join(format!("zest_gltf_trans_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.gltf");
        std::fs::write(&path, json).unwrap();
        let tri: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(dir.join("b.bin"), tri).unwrap();
        let scene = load_gltf(&path).expect("a one-triangle glTF loads");
        let g = &scene.materials[0];
        assert_eq!(g.params.transmission.transmission, 0.8);
        assert!(g.features.grab, "transmission needs the opaque scene copy");
        // thickness / distance = 1 -> the tint is the attenuation colour itself.
        assert!((g.params.transmission.tint[0] - 0.25).abs() < 1e-5 && g.params.transmission.tint[1] == 1.0, "{:?}", g.params.transmission.tint);
        let p = &scene.materials[1];
        assert_eq!(p.params.transmission.transmission, 0.0);
        assert!(!p.features.grab);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `KHR_texture_transform` (Blender's mapping-node scale/offset) imports as
    /// the standard shader's per-slot tiling + offset.
    #[test]
    fn khr_texture_transform_imports_as_tiling_and_offset() {
        let json = r#"{
          "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
          "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
          "buffers":[{"uri":"b.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteLength":36}],
          "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
          "images":[{"uri":"t.png"}],"textures":[{"source":0}],
          "extensionsUsed":["KHR_texture_transform"],
          "materials":[
            {"name":"Tiled","pbrMetallicRoughness":{
              "baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{"scale":[3.0,2.0],"offset":[0.25,0.5]}}},
              "metallicRoughnessTexture":{"index":0,"extensions":{"KHR_texture_transform":{"scale":[5.0,1.0]}}}
            }},
            {"name":"Plain"}]}"#;
        let dir = std::env::temp_dir().join(format!("zest_gltf_uv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.gltf");
        std::fs::write(&path, json).unwrap();
        let tri: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(dir.join("b.bin"), tri).unwrap();
        // A 1x1 PNG written by the image crate the loader decodes with.
        let mut png: Vec<u8> = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut png),
            &[255u8, 255, 255, 255],
            1,
            1,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();
        std::fs::write(dir.join("t.png"), png).unwrap();
        let scene = load_gltf(&path).expect("a one-triangle glTF with a texture loads");
        let g = &scene.materials[0];
        assert_eq!(g.params.albedo_tiling, [3.0, 2.0], "Blender's mapping scale becomes tiling");
        assert_eq!(g.params.albedo_offset, [0.25, 0.5]);
        assert_eq!(g.params.orm_tiling, [5.0, 1.0]);
        assert_eq!(g.params.orm_offset, [0.0, 0.0]);
        let p = &scene.materials[1];
        assert_eq!(p.params.albedo_tiling, [1.0, 1.0], "materials without the extension stay untransformed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
