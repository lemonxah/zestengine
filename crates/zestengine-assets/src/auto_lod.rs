//! Automatic LOD chains for imported models (GEO-2).
//!
//! Every model with enough triangles gets coarser copies of its meshes at the
//! shares in [`zestengine_core::lod::AUTO_LOD_STEPS`]. The copies are geometry
//! only (no textures or animation clips): the placed object keeps its own
//! materials and the engine just swaps the mesh by screen size.
//!
//! Skinned models are included. Their levels are simplified skin-aware (no
//! collapse crosses a hard bone-weight step, see
//! [`crate::simplify::simplify_skinned_with_provenance`]), keep every surviving
//! vertex's bones and weights, carry the skeleton so the engine registers the
//! level meshes as skinned, and carry the blendshape deltas remapped onto the
//! surviving vertices. Pure CPU work, so the engine runs it on a worker thread.

use zestengine_core::lod::{AUTO_LOD_MIN_TRIANGLES, AUTO_LOD_STEPS};

use crate::{Instance, ModelIdentity, Scene, SceneMaterial};

/// Triangles across all meshes.
pub fn triangle_count(scene: &Scene) -> usize {
    scene.meshes.iter().map(|m| m.indices.len() / 3).sum()
}

/// Whether a model gets an automatic chain: big enough to be worth the extra
/// levels (skinned or not).
pub fn wants_auto_lod(scene: &Scene) -> bool {
    !scene.meshes.is_empty() && triangle_count(scene) >= AUTO_LOD_MIN_TRIANGLES
}

/// Whether mesh `m` of a model with a skeleton carries skin weights.
fn mesh_is_skinned(scene: &Scene, m: &zestengine_render::MeshData) -> bool {
    !scene.skeletons.is_empty() && m.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0)
}

/// A cut of the mesh's cluster DAG holding about `keep` of its triangles, with
/// its blendshapes carried over, when the DAG can get that low (it cannot
/// below its roots, or on meshes it could not reduce). `None` falls back to
/// the simplifier.
fn dag_cut_mesh(scene: &Scene, index: usize, keep: f32, dag: &crate::cluster_dag::ClusterDag) -> Option<(zestengine_render::MeshData, Vec<crate::MorphTarget>)> {
    let mesh = &scene.meshes[index];
    let want = mesh.indices.len() as f32 / 3.0 * keep;
    let cut = dag.discrete_lods(&[keep]).pop()?;
    if cut.triangles as f32 > want * 1.25 + 128.0 {
        return None;
    }
    let (lod, provenance) = dag.cut_mesh_with_provenance(mesh, &cut.clusters);
    let morphs = scene.morphs.get(index).map(Vec::as_slice).unwrap_or(&[]);
    let morphs = remap_all(morphs, |d| crate::simplify::remap_morph_deltas(d, &provenance));
    Some((lod, morphs))
}

/// Every target of `morphs` with its deltas run through `remap` (targets are
/// independent, and a face model has hundreds).
fn remap_all(morphs: &[crate::MorphTarget], remap: impl Fn(&[(u32, [f32; 3], [f32; 3])]) -> Vec<(u32, [f32; 3], [f32; 3])> + Sync) -> Vec<crate::MorphTarget> {
    let mut work: Vec<(&crate::MorphTarget, crate::MorphTarget)> = morphs.iter().map(|t| (t, crate::MorphTarget { name: t.name.clone(), deltas: Vec::new() })).collect();
    crate::mesh_pipeline::par_each(&mut work, |(src, out)| out.deltas = remap(&src.deltas));
    work.into_iter().map(|(_, out)| out).collect()
}

/// One mesh at `keep` of its triangles, with its blendshapes carried over.
/// Static meshes take a cut of their cluster DAG when `geo` has one; skinned
/// meshes (and anything the DAG cannot reach) use the simplifier.
fn simplified_mesh(scene: &Scene, index: usize, keep: f32, geo: Option<&crate::geo_cache::ModelGeometry>) -> (zestengine_render::MeshData, Vec<crate::MorphTarget>) {
    let mesh = &scene.meshes[index];
    if !mesh_is_skinned(scene, mesh)
        && let Some(dag) = geo.and_then(|g| g.meshes.get(index)).and_then(Option::as_ref)
        && let Some(cut) = dag_cut_mesh(scene, index, keep, dag)
    {
        return cut;
    }
    let morphs = scene.morphs.get(index).map(Vec::as_slice).unwrap_or(&[]);
    if mesh_is_skinned(scene, mesh) {
        let (lod, provenance) = crate::simplify::simplify_skinned_with_provenance(mesh, keep, crate::simplify::SKIN_REGION_MAX_STEP);
        let morphs = remap_all(morphs, |d| crate::simplify::remap_morph_deltas_multi(d, &provenance));
        (lod, morphs)
    } else {
        let (lod, provenance) = crate::simplify::simplify_with_provenance(mesh, keep);
        let morphs = remap_all(morphs, |d| crate::simplify::remap_morph_deltas(d, &provenance));
        (lod, morphs)
    }
}

/// A copy of `scene` with `meshes` swapped in and everything but the geometry
/// layout (instances, one material per original so slot counts match) dropped.
fn geometry_variant(scene: &Scene, meshes: Vec<zestengine_render::MeshData>, morphs: Vec<Vec<crate::MorphTarget>>) -> Scene {
    Scene {
        meshes,
        morphs,
        springs: Default::default(),
        textures: Vec::new(),
        materials: scene
            .materials
            .iter()
            .map(|m| SceneMaterial { name: m.name.clone(), params: m.params, features: m.features, albedo: None, normal: None, orm: None, emission: None })
            .collect(),
        instances: scene.instances.iter().map(|i| Instance { name: i.name.clone(), transform: i.transform, slots: i.slots.clone() }).collect(),
        // Skinned levels keep the skeleton (joint indices in the vertices refer
        // to it); clips stay with the original model.
        skeletons: scene.skeletons.clone(),
        animations: Vec::new(),
        collision: Vec::new(),
        identity: ModelIdentity {
            guid: scene.identity.guid,
            mesh_names: scene.identity.mesh_names.clone(),
            mesh_ids: scene.identity.mesh_ids.clone(),
            ..Default::default()
        },
    }
}

/// The coarser levels (all of [`AUTO_LOD_STEPS`] after the first) as
/// `(share kept, scene)`, or `None` when the model gets no chain.
pub fn build_auto_lods(scene: &Scene) -> Option<Vec<(f32, Scene)>> {
    build_auto_lods_with(scene, None)
}

/// [`build_auto_lods`] taking cuts of the model's cluster DAGs (see
/// [`crate::geo_cache`]) for the meshes that have one, instead of simplifying
/// to a triangle share. The levels are the coarser, view-independent cuts of
/// the same DAG the cluster renderer draws from.
pub fn build_auto_lods_with(scene: &Scene, geo: Option<&crate::geo_cache::ModelGeometry>) -> Option<Vec<(f32, Scene)>> {
    if !wants_auto_lod(scene) {
        return None;
    }
    // Every (level, mesh) is independent: simplify them side by side, then
    // gather the levels in order.
    let keeps: Vec<f32> = AUTO_LOD_STEPS.iter().skip(1).map(|&(keep, _)| keep).collect();
    let count = scene.meshes.len();
    type Cut = Option<(zestengine_render::MeshData, Vec<crate::MorphTarget>)>;
    let mut jobs: Vec<(f32, usize, Cut)> = keeps.iter().flat_map(|&keep| (0..count).map(move |i| (keep, i, None))).collect();
    crate::mesh_pipeline::par_each_dyn(&mut jobs, |(keep, i, out)| *out = Some(simplified_mesh(scene, *i, *keep, geo)));
    let mut cuts = jobs.into_iter().map(|(_, _, cut)| cut.expect("every job ran"));
    Some(
        keeps
            .iter()
            .map(|&keep| {
                let (meshes, morphs): (Vec<_>, Vec<_>) = cuts.by_ref().take(count).unzip();
                let morphs = if scene.morphs.iter().all(Vec::is_empty) { Vec::new() } else { morphs };
                (keep, geometry_variant(scene, meshes, morphs))
            })
            .collect(),
    )
}

/// The chain for the model at `path`, from the stored copy when its import-cache
/// key is unchanged, else built and stored (see [`crate::lod_cache`]). `None`
/// when the model opts out in its `.meta` (`ModelImport::auto_lod`), is too
/// small, or can't be imported.
pub fn load_or_build(path: &std::path::Path) -> Option<Vec<(f32, Scene)>> {
    let model = crate::load_asset_meta(path).ok().flatten().and_then(|m| m.importer.as_model().cloned()).unwrap_or_default();
    if !model.auto_lod {
        return None;
    }
    let key = crate::import_cache::cache_key(path, &model);
    if let Some(key) = &key
        && let Some(stored) = crate::lod_cache::load(path, key)
    {
        return (!stored.is_empty()).then_some(stored);
    }
    let scene = crate::load_model_with_meta(path).ok()?;
    // The cluster DAGs are built (or read back) here, on the worker; the levels are cuts of them.
    let geo = crate::geo_cache::load_or_build(path, &scene);
    let built = build_auto_lods_with(&scene, Some(&geo));
    if let Some(key) = &key {
        crate::lod_cache::store(path, key, built.as_deref().unwrap_or(&[]));
    }
    built
}

/// Template key the engine installs a level under.
pub fn level_key(path: &str, keep: f32) -> String {
    format!("{path}#lod{keep:.3}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_of(shape: crate::PrimitiveShape, skinned: bool) -> Scene {
        let mesh = crate::procedural::primitive_mesh(shape);
        let mut skeletons = Vec::new();
        if skinned {
            skeletons.push(crate::Skeleton { joints: Vec::new(), rig: None, ..Default::default() });
        }
        Scene {
            meshes: vec![mesh],
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: vec![Instance::single("m", 0, 0, glam::Mat4::IDENTITY)],
            skeletons,
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        }
    }

    #[test]
    fn chain_shrinks_each_level_and_keeps_layout() {
        let scene = scene_of(crate::PrimitiveShape::Sphere, false);
        let tris = triangle_count(&scene);
        if tris < AUTO_LOD_MIN_TRIANGLES {
            assert!(build_auto_lods(&scene).is_none(), "small models get no chain");
            return;
        }
        let levels = build_auto_lods(&scene).unwrap();
        assert_eq!(levels.len(), AUTO_LOD_STEPS.len() - 1);
        let mut last = tris;
        for (keep, lod) in &levels {
            let t = triangle_count(lod);
            assert!(t < last, "keep {keep}: {t} !< {last}");
            last = t;
            assert_eq!(lod.meshes.len(), scene.meshes.len());
            assert_eq!(lod.instances.len(), scene.instances.len());
            assert!(lod.textures.is_empty() && lod.skeletons.is_empty(), "a static model's levels carry no skeleton");
        }
    }

    #[test]
    fn levels_are_cuts_of_the_cluster_dag_when_there_is_one() {
        let scene = scene_of(crate::PrimitiveShape::Sphere, false);
        let geo = crate::geo_cache::build_model(&scene, &crate::cluster_dag::DagOptions::default());
        let dag = geo.meshes[0].as_ref().expect("the sphere is above the DAG threshold");
        let levels = build_auto_lods_with(&scene, Some(&geo)).unwrap();
        let tris = triangle_count(&scene);
        let mut last = tris;
        for (keep, lod) in &levels {
            let t = triangle_count(lod);
            assert!(t < last, "keep {keep}: {t} !< {last}");
            assert!(t as f32 <= tris as f32 * keep * 1.25 + 128.0, "keep {keep}: {t} triangles");
            last = t;
            // Every level vertex is a source vertex: a cut, not a re-fit surface.
            let src: std::collections::HashSet<[u32; 3]> = scene.meshes[0].vertices.iter().map(|v| v.position.map(f32::to_bits)).collect();
            assert!(lod.meshes[0].vertices.iter().all(|v| src.contains(&v.position.map(f32::to_bits))));
            // And it matches the DAG's own cut for that share.
            assert_eq!(t, dag.discrete_lods(&[*keep])[0].triangles);
        }
        // Without a DAG the old path still produces a chain.
        assert_eq!(build_auto_lods(&scene).unwrap().len(), levels.len());
    }

    #[test]
    fn tiny_models_are_skipped() {
        let mut tiny = scene_of(crate::PrimitiveShape::Sphere, false);
        tiny.meshes[0].indices.truncate(30);
        assert!(build_auto_lods(&tiny).is_none());
    }

    /// A `n` x `n` quad grid in the XZ plane skinned to two joints: x < n/2
    /// is bound hard to joint 0, the rest to joint 1 (a rigid step at the
    /// middle column), with a blendshape lifting every vertex.
    fn skinned_grid(n: u32) -> Scene {
        use zestengine_render::{MeshData, Vertex};
        let mut vertices = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                let joint = u32::from(x > n / 2);
                vertices.push(Vertex {
                    position: [x as f32, 0.0, z as f32],
                    normal: [0.0, 1.0, 0.0],
                    joints: [joint, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                });
            }
        }
        let mut indices = Vec::new();
        let w = n + 1;
        for z in 0..n {
            for x in 0..n {
                let i = z * w + x;
                indices.extend([i, i + w, i + 1, i + 1, i + w, i + w + 1]);
            }
        }
        let deltas = (0..vertices.len() as u32).map(|v| (v, [0.0, 0.1, 0.0], [0.0; 3])).collect();
        let joint = |name: &str| crate::skeleton::Joint {
            name: name.into(),
            parent: None,
            inverse_bind: glam::Mat4::IDENTITY,
            rest_translation: glam::Vec3::ZERO,
            rest_rotation: glam::Quat::IDENTITY,
            rest_scale: glam::Vec3::ONE,
        };
        let mut scene = scene_of(crate::PrimitiveShape::Cube, true);
        scene.meshes = vec![MeshData { vertices, indices, has_lightmap_uv: false }];
        scene.morphs = vec![vec![crate::MorphTarget { name: "up".into(), deltas }]];
        scene.skeletons = vec![crate::Skeleton { joints: vec![joint("a"), joint("b")], rig: None, ..Default::default() }];
        scene
    }

    #[test]
    fn skinned_models_get_skin_aware_levels_with_bones_and_blendshapes() {
        let scene = skinned_grid(40);
        assert!(wants_auto_lod(&scene));
        let levels = build_auto_lods(&scene).expect("skinned chain");
        let mut last = triangle_count(&scene);
        for (_, lod) in &levels {
            let t = triangle_count(lod);
            assert!(t < last, "each level is coarser ({t} !< {last})");
            last = t;
            assert_eq!(lod.skeletons.len(), 1, "skeleton kept for the engine's skinned registration");
            let mesh = &lod.meshes[0];
            // Every surviving vertex keeps valid bones and unit weights.
            for v in &mesh.vertices {
                assert!(v.joints[0] < 2 && (v.weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
            // The rigid step at x = n/2 (joint 0 | 1) is kept: both joints stay
            // present and the shared boundary column is not collapsed away.
            let j0 = mesh.vertices.iter().filter(|v| v.joints[0] == 0).count();
            let j1 = mesh.vertices.iter().filter(|v| v.joints[0] == 1).count();
            assert!(j0 > 0 && j1 > 0);
            let edge = mesh.vertices.iter().filter(|v| (v.position[0] - 20.0).abs() < 1e-3).count();
            assert!(edge >= 41, "boundary column stays intact: {edge}");
            // Blendshape deltas follow the vertices (including duplicated border ones).
            let delta_targets = &lod.morphs[0][0].deltas;
            assert_eq!(delta_targets.len(), mesh.vertices.len(), "every surviving vertex keeps its delta");
            assert!(delta_targets.iter().all(|d| (d.0 as usize) < mesh.vertices.len()));
        }
    }

    #[test]
    fn skin_regions_split_at_hard_steps_only() {
        let scene = skinned_grid(8);
        let regions = crate::simplify::skin_regions(&scene.meshes[0], crate::simplify::SKIN_REGION_MAX_STEP);
        let count = regions.iter().copied().max().unwrap() + 1;
        assert!(count >= 2, "the rigid halves (and the straddling strip) are separate regions: {count}");
        // Smooth blend: no split.
        let m = &scene.meshes[0];
        let mut smooth = zestengine_render::MeshData { vertices: m.vertices.clone(), indices: m.indices.clone(), has_lightmap_uv: false };
        for v in &mut smooth.vertices {
            let t = (v.position[0] / 8.0).clamp(0.0, 1.0);
            v.joints = [0, 1, 0, 0];
            v.weights = [1.0 - t, t, 0.0, 0.0];
        }
        let regions = crate::simplify::skin_regions(&smooth, crate::simplify::SKIN_REGION_MAX_STEP);
        assert_eq!(regions.iter().copied().max().unwrap() + 1, 1, "a gradual blend is one region");
    }

    #[test]
    fn keys_match_the_distance_lod_scheme() {
        assert_eq!(level_key("models/a.glb", 0.5), "models/a.glb#lod0.500");
    }
}
