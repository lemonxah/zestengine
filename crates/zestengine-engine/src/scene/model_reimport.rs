//! Per-asset reimport: rebuild one model and swap its GPU meshes, materials,
//! skeletons and clips in the live scene, in place and keyed by `SubAssetId`.
//!
//! Objects keep their indices, so everything that points at them (instance
//! overrides, material overrides, components, parents, animator state) stays
//! valid: only the data behind the model's mesh / material slots changes. That
//! is why no scene reload is needed and play mode can keep running.
//!
//! What is matched and what happens to the rest:
//! - Meshes, template materials, skeletons and clips are matched by their
//!   `SubAssetId` (position when the model has no ids, and for the leftovers
//!   of an id that changed).
//! - A mesh the new file adds gets a slot: the model's meshes move to a fresh
//!   contiguous block (the scene addresses a model as `base + local index`),
//!   existing objects follow their mesh id into it, and the old block's GPU
//!   meshes are released. A mesh the new file no longer has keeps its old
//!   geometry (its objects are counted in the report, not deleted).
//! - A template material the user edited keeps its edits; an untouched one
//!   takes the new import. New materials are created.
//! - The model's remap table (material and texture remaps) is applied again, and
//!   objects follow it.
//! - Generated LOD levels (`path#lod...`) are dropped and rebuilt from the new
//!   geometry. The materials those levels created (and any textures only they
//!   used) are released with the meshes, unless something else still uses them
//!   (see `release_materials`).

use std::collections::{HashMap, HashSet};

use zestengine_assets::sub_asset::{self, SubAssetId};

use super::*;

/// Where a model's data landed in the scene arrays when it was installed, so a
/// reimport can find it again.
#[derive(Clone, Debug, Default)]
pub(crate) struct ModelSlots {
    /// Id of each mesh slot (empty when the model had none).
    pub mesh_ids: Vec<SubAssetId>,
    /// Scene material index and id of each of the model's materials, in file order.
    pub material_idx: Vec<usize>,
    pub material_ids: Vec<SubAssetId>,
    /// Scene skeleton / clip indices and ids, in file order.
    pub skel_idx: Vec<usize>,
    pub skel_ids: Vec<SubAssetId>,
    pub clip_idx: Vec<usize>,
    pub clip_ids: Vec<SubAssetId>,
}

/// A remap to apply once a renderer is at hand: the template material it
/// belongs to, the `.material` it points at, and its remapped texture paths.
#[derive(Clone, Debug)]
pub(crate) struct PendingRemap {
    pub template: usize,
    pub file: Option<String>,
    pub textures: [Option<String>; 4],
}

impl ModelSlots {
    /// The slots of a model just installed at the given scene array positions.
    pub(crate) fn installed(asset: &zestengine_assets::Scene, skel_base: usize, anim_base: usize, material_base: usize) -> Self {
        let id = &asset.identity;
        Self {
            mesh_ids: id.mesh_ids.clone(),
            material_idx: (material_base..material_base + asset.materials.len()).collect(),
            material_ids: id.material_ids.clone(),
            skel_idx: (skel_base..skel_base + asset.skeletons.len()).collect(),
            skel_ids: id.skeleton_ids.clone(),
            clip_idx: (anim_base..anim_base + asset.animations.len()).collect(),
            clip_ids: id.clip_ids.clone(),
        }
    }
}

/// What a reimport changed, for the status line and tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelReimport {
    /// Mesh slots re-uploaded.
    pub meshes: usize,
    /// Meshes in the new file that had no slot; they were given one.
    pub added_meshes: usize,
    /// Mesh slots whose id is gone from the new file (left as they were).
    pub missing_meshes: usize,
    /// The model's meshes moved to a new block because meshes were added.
    pub moved: bool,
    /// Objects still showing a mesh the file no longer has.
    pub orphaned_objects: usize,
    /// Template materials updated.
    pub materials: usize,
    /// Template materials created for materials the file added.
    pub materials_added: usize,
    pub skeletons: usize,
    pub clips: usize,
    pub clips_added: usize,
    /// GPU meshes and textures released.
    pub freed_meshes: usize,
    pub freed_textures: usize,
    /// Materials of the replaced LOD levels that were released.
    pub freed_materials: usize,
}

/// What `release_materials` let go of.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReleasedMaterials {
    /// Scene material indices released.
    pub scene: Vec<usize>,
    /// Renderer material / texture indices to free (headless scenes have none).
    pub materials: HashSet<usize>,
    pub textures: HashSet<usize>,
}

/// Pair each new name with an old slot: same name first, then same position.
/// `(new, old)`; every old slot is used at most once.
fn pair_by_name(old: &[&str], new: &[&str]) -> Vec<(usize, usize)> {
    pair_keys(old, new)
}

/// Pair each new id with an old slot: the same id first, then (for ids that
/// changed, or a model without ids) the same position. `(new, old)`; every old
/// slot is used at most once.
fn pair_by_id(old: &[SubAssetId], new: &[SubAssetId], old_len: usize, new_len: usize) -> Vec<(usize, usize)> {
    if old.len() != old_len || new.len() != new_len {
        // No usable ids on one side: positions.
        return (0..old_len.min(new_len)).map(|k| (k, k)).collect();
    }
    pair_keys(old, new)
}

fn pair_keys<T: PartialEq>(old: &[T], new: &[T]) -> Vec<(usize, usize)> {
    let mut taken = vec![false; old.len()];
    let mut out = Vec::new();
    let mut unpaired = Vec::new();
    for (k, key) in new.iter().enumerate() {
        match old.iter().enumerate().find(|(i, o)| !taken[*i] && **o == *key) {
            Some((i, _)) => {
                taken[i] = true;
                out.push((k, i));
            }
            None => unpaired.push(k),
        }
    }
    for k in unpaired {
        if k < old.len() && !taken[k] {
            taken[k] = true;
            out.push((k, k));
        }
    }
    out.sort_unstable();
    out
}

impl LoadedScene {
    /// Swap the live copy of model `path` (project-relative, as scenes store
    /// it) for `asset`. `None` when the scene has not loaded that model.
    /// With no renderer (headless) only the CPU side is updated.
    pub fn reimport_model(
        &mut self,
        mut renderer: Option<&mut Renderer>,
        shaders: &mut ShaderLibrary,
        project_root: &Path,
        path: &str,
        asset: &zestengine_assets::Scene,
    ) -> Result<Option<ModelReimport>> {
        let Some((mesh_base, template_materials)) = self.model_templates.get(path).cloned() else { return Ok(None) };
        let Some(slots) = self.model_slots.get(path).cloned() else { return Ok(None) };
        let mut report = ModelReimport::default();
        let old_meshes = template_materials.len();
        let abs = project_root.join(path);

        // New mesh j -> old slot i, by id.
        let have_ids = slots.mesh_ids.len() == old_meshes && asset.identity.mesh_ids.len() == asset.meshes.len();
        let mut slot_taken = vec![false; old_meshes];
        let mut mesh_pairs: Vec<(usize, usize)> = Vec::new();
        for j in 0..asset.meshes.len() {
            let slot = if have_ids { sub_asset::index_of(&slots.mesh_ids, asset.identity.mesh_ids[j]) } else { Some(j) };
            match slot.filter(|&i| i < old_meshes && !slot_taken[i]) {
                Some(i) => {
                    slot_taken[i] = true;
                    mesh_pairs.push((j, i));
                }
                None => report.added_meshes += 1,
            }
        }
        report.missing_meshes = slot_taken.iter().filter(|t| !**t).count();
        // New meshes need slots, and a model's slots are one contiguous block.
        let moved = report.added_meshes > 0;
        report.moved = moved;

        // Template materials: which new material replaces which old one. A
        // material the user pointed at a `.material` file is theirs; skip it.
        let old_mat_names: Vec<&str> = slots.material_idx.iter().map(|&i| self.materials.get(i).map_or("", |m| m.default.name.as_str())).collect();
        let new_mat_names: Vec<&str> = asset.materials.iter().map(|m| m.name.as_str()).collect();
        let have_mat_ids = slots.material_ids.len() == slots.material_idx.len() && asset.identity.material_ids.len() == asset.materials.len();
        let all_material_pairs = if have_mat_ids {
            pair_by_id(&slots.material_ids, &asset.identity.material_ids, slots.material_idx.len(), asset.materials.len())
        } else {
            pair_by_name(&old_mat_names, &new_mat_names)
        };
        let material_pairs: Vec<(usize, usize)> = all_material_pairs
            .iter()
            .copied()
            .filter(|&(_, old)| self.materials.get(slots.material_idx[old]).is_some_and(|m| m.file.is_none()))
            .collect();
        let paired_new: HashSet<usize> = all_material_pairs.iter().map(|p| p.0).collect();
        let fresh_materials: Vec<usize> = (0..asset.materials.len()).filter(|k| !paired_new.contains(k)).collect();

        // Upload the textures the updated and new materials use (the rest are not needed).
        let mut textures = vec![TextureHandle::NONE; asset.textures.len()];
        if let Some(r) = renderer.as_deref_mut() {
            let mut wanted: Vec<usize> = Vec::new();
            for k in material_pairs.iter().map(|p| p.0).chain(fresh_materials.iter().copied()) {
                let m = &asset.materials[k];
                wanted.extend([m.albedo, m.normal, m.orm, m.emission].into_iter().flatten());
            }
            wanted.sort_unstable();
            wanted.dedup();
            for t in wanted {
                if let Some(data) = asset.textures.get(t) {
                    textures[t] = r.upload_compressed_texture(&zestengine_assets::compress_embedded_texture(Some(&abs), t, data))?;
                }
            }
        }

        let mut free_meshes: HashSet<usize> = HashSet::new();
        let mut free_textures: HashSet<usize> = HashSet::new();

        // Scene material index of each of the new file's materials.
        let mut material_idx = vec![usize::MAX; asset.materials.len()];
        for &(k, old) in &all_material_pairs {
            material_idx[k] = slots.material_idx[old];
        }
        for &k in &fresh_materials {
            material_idx[k] = self.push_template_material(renderer.as_deref_mut(), asset, &textures, k)?;
            report.materials_added += 1;
        }

        // Skeletons: by id (position for leftovers); new ones are appended. The
        // stored rig is recomputed from the new bones unless the user set one.
        let have_skel_ids = slots.skel_ids.len() == slots.skel_idx.len() && asset.identity.skeleton_ids.len() == asset.skeletons.len();
        let skel_pairs = if have_skel_ids {
            pair_by_id(&slots.skel_ids, &asset.identity.skeleton_ids, slots.skel_idx.len(), asset.skeletons.len())
        } else {
            (0..slots.skel_idx.len().min(asset.skeletons.len())).map(|k| (k, k)).collect()
        };
        let mut skel_idx = vec![usize::MAX; asset.skeletons.len()];
        for &(k, old) in &skel_pairs {
            let at = slots.skel_idx[old];
            let mut skel = asset.skeletons[k].clone();
            if skel.rig.is_none() {
                skel.rig = self.skeletons[at].rig.clone().filter(|r| !r.auto_generated);
            }
            self.skeletons[at] = skel;
            self.anim_assets.forget_skeleton_rig(at);
            crate::humanoid::ensure_rigs(std::slice::from_mut(&mut self.skeletons[at]));
            skel_idx[k] = at;
            report.skeletons += 1;
        }
        for (k, skel) in asset.skeletons.iter().enumerate() {
            if skel_idx[k] == usize::MAX {
                self.skeletons.push(skel.clone());
                let at = self.skeletons.len() - 1;
                crate::humanoid::ensure_rigs(std::slice::from_mut(&mut self.skeletons[at]));
                skel_idx[k] = at;
                report.skeletons += 1;
            }
        }
        // A skinned mesh binds the model's first skeleton.
        let skel0 = skel_idx.first().copied().unwrap_or(0);

        // Meshes: new GPU copy, swapped under the same scene mesh index (or, when
        // the model grew, uploaded into a fresh block that existing objects move to).
        let new_base = if moved { self.mesh_handles.len() } else { mesh_base };
        let mut new_global = vec![usize::MAX; asset.meshes.len()];
        let slot_of: HashMap<usize, usize> = mesh_pairs.iter().copied().collect();
        for (j, ng) in new_global.iter_mut().enumerate() {
            let at = if moved { None } else { slot_of.get(&j).map(|&i| mesh_base + i) };
            if !moved && at.is_none() {
                continue;
            }
            let mi = self.put_mesh(renderer.as_deref_mut(), asset, j, skel0, at, &mut free_meshes)?;
            *ng = mi;
            report.meshes += usize::from(slot_of.contains_key(&j));
        }
        // Mesh index of each old slot after the move (identity when in place).
        let old_to_new: HashMap<usize, usize> = mesh_pairs.iter().map(|&(j, i)| (mesh_base + i, new_global[j])).collect();
        if moved {
            let users: Vec<ObjectId> = self.objects.iter().filter(|o| o.render_slots().any(|r| old_to_new.contains_key(&r.mesh))).map(|o| o.id).collect();
            for o in &mut self.objects {
                for r in o.render.iter_mut().chain(o.extra_render.iter_mut()) {
                    if let Some(&to) = old_to_new.get(&r.mesh) {
                        r.mesh = to;
                    }
                }
            }
            for &old in old_to_new.keys() {
                let h = std::mem::replace(&mut self.mesh_handles[old], MeshHandle::NONE);
                if h != MeshHandle::NONE {
                    free_meshes.insert(h.index());
                }
                self.skinned_meshes.retain(|sm| sm.mesh_index != old);
            }
            for id in users {
                if let Some(inst) = self.anim_instances.remove(&id) {
                    free_meshes.extend(inst.slots.iter().map(|(_, h)| h.index()).chain(inst.levels.handles().map(|h| h.index())).filter(|h| *h != MeshHandle::NONE.index()));
                }
            }
        }

        // Clips: replaced by id (name for a model without ids), new ones appended.
        let have_clip_ids = slots.clip_ids.len() == slots.clip_idx.len() && asset.identity.clip_ids.len() == asset.animations.len();
        let clip_pairs = if have_clip_ids {
            pair_by_id(&slots.clip_ids, &asset.identity.clip_ids, slots.clip_idx.len(), asset.animations.len())
        } else {
            let old_names: Vec<&str> = slots.clip_idx.iter().map(|&i| self.animations.get(i).map_or("", |c| c.name.as_str())).collect();
            let new_names: Vec<&str> = asset.animations.iter().map(|c| c.name.as_str()).collect();
            pair_by_name(&old_names, &new_names)
        };
        let mut clip_idx = vec![usize::MAX; asset.animations.len()];
        for &(k, old) in &clip_pairs {
            let at = slots.clip_idx[old];
            self.animations[at] = asset.animations[k].clone();
            clip_idx[k] = at;
            report.clips += 1;
        }
        for (k, clip) in asset.animations.iter().enumerate() {
            if clip_idx[k] == usize::MAX {
                self.animations.push(clip.clone());
                self.clip_skeleton.push(skel0);
                clip_idx[k] = self.animations.len() - 1;
                report.clips_added += 1;
            }
        }

        // Template materials, in place: same entry, same GPU material, new data.
        let old_remaps: HashMap<usize, usize> = slots.material_idx.iter().filter_map(|t| self.material_remaps.remove(t).map(|target| (*t, target))).collect();
        for &(k, old) in &material_pairs {
            let idx = slots.material_idx[old];
            let m = &asset.materials[k];
            let handle_of = |i: Option<usize>| i.map(|i| textures[i]).filter(|h| *h != TextureHandle::NONE);
            let mut base_textures = TexSlots::default();
            base_textures[0] = handle_of(m.albedo);
            base_textures[1] = handle_of(m.normal);
            base_textures[2] = handle_of(m.orm);
            base_textures[3] = handle_of(m.emission);
            let mut base_tex_data: [Option<zestengine_render::TextureData>; 16] = Default::default();
            base_tex_data[0] = m.albedo.map(|i| asset.textures[i].clone());
            base_tex_data[1] = m.normal.map(|i| asset.textures[i].clone());
            base_tex_data[2] = m.orm.map(|i| asset.textures[i].clone());
            base_tex_data[3] = m.emission.map(|i| asset.textures[i].clone());
            let model = model_from_material(&m.name, &m.params, &m.features, m.normal.is_some());
            let entry = &mut self.materials[idx];
            free_textures.extend(entry.base_textures.iter().flatten().map(|h| h.index()));
            // Edits the user made to the template material survive; an untouched one follows the file.
            let edited = entry.model != entry.default;
            entry.default = model.clone();
            if !edited {
                entry.model = model;
            }
            entry.base_textures = base_textures;
            entry.base_tex_data = base_tex_data;
            entry.embedded_textures = base_textures.iter().any(Option::is_some);
            if let Some(r) = renderer.as_deref_mut() {
                self.apply_material(r, shaders, project_root, idx);
            }
            report.materials += 1;
        }
        // Never free a texture another material still shows.
        let in_use: HashSet<usize> = self
            .materials
            .iter()
            .flat_map(|m| m.base_textures.iter().chain(m.bound_textures.iter()))
            .flatten()
            .map(|h| h.index())
            .collect();
        free_textures.retain(|t| !in_use.contains(t) && *t != TextureHandle::NONE.index());

        // The model's template: where its meshes are now, and which material each shows.
        let mut template_new: Vec<usize> = if moved { vec![0; asset.meshes.len()] } else { template_materials.clone() };
        let first_material = material_idx.first().copied();
        for (j, &mi) in new_global.iter().enumerate() {
            if mi == usize::MAX {
                continue;
            }
            let local = if moved { j } else { mi - mesh_base };
            let from_file = asset.instances.iter().flat_map(|i| &i.slots).find(|s| s.mesh == j).and_then(|s| material_idx.get(s.material).copied());
            template_new[local] = from_file.or_else(|| slot_of.get(&j).map(|&i| template_materials[i])).or(first_material).unwrap_or(0);
        }
        let old_template_of = |mesh_global: usize| -> Option<usize> { mesh_global.checked_sub(mesh_base).and_then(|l| template_materials.get(l).copied()) };
        // Authored collision and the saved-scene baseline follow the new file.
        self.authored_collision.replace(asset, path, mesh_base, old_meshes, &|m| (new_global.get(m).copied()).filter(|g| *g != usize::MAX));
        self.model_parts.insert(path.to_owned(), zestengine_assets::ModelPart::of(asset));
        self.model_templates.insert(path.to_owned(), (new_base, template_new.clone()));
        self.model_mesh_base.insert(PathBuf::from(path), new_base);
        self.model_slots.insert(
            path.to_owned(),
            ModelSlots {
                mesh_ids: if moved { asset.identity.mesh_ids.clone() } else { slots.mesh_ids.clone() },
                material_idx: material_idx.clone(),
                material_ids: asset.identity.material_ids.clone(),
                skel_idx: skel_idx.clone(),
                skel_ids: asset.identity.skeleton_ids.clone(),
                clip_idx: clip_idx.clone(),
                clip_ids: asset.identity.clip_ids.clone(),
            },
        );

        // Remaps: the new file's tables, applied to the (possibly new) template materials.
        let mut pending = Vec::new();
        for (k, &idx) in material_idx.iter().enumerate() {
            let file = asset.identity.material_remap.get(k).cloned().flatten();
            let textures = asset.identity.texture_remap.get(k).cloned().unwrap_or_default();
            if file.is_some() || textures.iter().any(Option::is_some) {
                pending.push(PendingRemap { template: idx, file, textures });
            }
        }
        if let Some(r) = renderer.as_deref_mut() {
            for p in pending {
                self.apply_remap(r, shaders, project_root, p);
            }
        }

        // Objects follow: a slot showing a template material (or what it was remapped to)
        // moves to whatever the template now resolves to.
        let slot_template: HashMap<usize, (usize, Option<usize>)> = (0..template_new.len())
            .map(|l| {
                let global = new_base + l;
                let old_tm = if moved { None } else { old_template_of(global) };
                (global, (template_new[l], old_tm))
            })
            .collect();
        let model_materials: HashSet<usize> = material_idx.iter().copied().collect();
        for o in &mut self.objects {
            let is_model = matches!(&o.source, ObjectSource::Model { path: p, .. } if p == path);
            if !is_model {
                continue;
            }
            for r in o.render.iter_mut().chain(o.extra_render.iter_mut()) {
                let Some(&(tm, old_tm)) = slot_template.get(&r.mesh) else { continue };
                let was_template = Some(r.material) == old_tm || r.material == tm || old_tm.is_some_and(|t| old_remaps.get(&t) == Some(&r.material));
                // After a move the previous template is unknown per mesh; any model
                // material or old remap target of this model counts as "following it".
                let was_model = moved && (model_materials.contains(&r.material) || old_remaps.values().any(|t| *t == r.material));
                if was_template || was_model {
                    r.material = self.material_remaps.get(&tm).copied().unwrap_or(tm);
                }
            }
        }
        report.orphaned_objects = self
            .objects
            .iter()
            .filter(|o| matches!(&o.source, ObjectSource::Model { path: p, .. } if p == path))
            .filter(|o| o.render_slots().any(|r| (mesh_base..mesh_base + old_meshes).contains(&r.mesh) && slot_taken.get(r.mesh - mesh_base) == Some(&false)))
            .count();
        // The dropped LOD levels' materials go with their meshes; one free call covers all of it.
        let lod_materials = self.refresh_lod_templates(path, &mut free_meshes);
        let released = self.release_materials(lod_materials);
        report.freed_materials = released.scene.len();
        free_textures.extend(released.textures);
        self.draws.clear();

        if let Some(r) = renderer
            && (!free_meshes.is_empty() || !free_textures.is_empty() || !released.materials.is_empty())
        {
            report.freed_meshes = free_meshes.len();
            report.freed_textures = free_textures.len();
            if let Err(e) = r.free_resources(&free_meshes, &free_textures, &released.materials) {
                tracing::warn!("reimport {path}: freeing old resources: {e:#}");
            }
        }
        Ok(Some(report))
    }

    /// Mesh ids of a loaded model's slots (tests and tools).
    pub fn model_mesh_ids(&self, path: &str) -> Option<&[SubAssetId]> {
        self.model_slots.get(path).map(|s| s.mesh_ids.as_slice())
    }

    /// Upload mesh `j` of `asset` into the scene's mesh arrays: over slot `at`
    /// when given, else as a new slot. Returns the scene mesh index.
    fn put_mesh(
        &mut self,
        renderer: Option<&mut Renderer>,
        asset: &zestengine_assets::Scene,
        j: usize,
        skel: usize,
        at: Option<usize>,
        free_meshes: &mut HashSet<usize>,
    ) -> Result<usize> {
        let mesh = &asset.meshes[j];
        let skinned = !asset.skeletons.is_empty() && mesh.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0);
        let handle = match renderer {
            Some(r) if skinned => r.upload_mesh_skinned(mesh)?,
            Some(r) => r.upload_mesh(mesh)?,
            None => MeshHandle::NONE,
        };
        let geometry = (mesh.vertices.iter().map(|v| Vec3::from(v.position)).collect::<Vec<_>>(), mesh.indices.clone());
        let info = MeshInfo { vertices: mesh.vertices.len() as u32, triangles: mesh.indices.len() as u32 / 3 };
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &mesh.vertices {
            min = min.min(Vec3::from(v.position));
            max = max.max(Vec3::from(v.position));
        }
        let mi = match at {
            Some(mi) => {
                let old = std::mem::replace(&mut self.mesh_handles[mi], handle);
                if old != MeshHandle::NONE {
                    free_meshes.insert(old.index());
                }
                self.mesh_geometry[mi] = geometry;
                self.mesh_sdf[mi] = None;
                self.mesh_infos[mi] = info;
                self.mesh_bounds[mi] = (min, max);
                self.mesh_has_lightmap_uv[mi] = mesh.has_lightmap_uv;
                mi
            }
            None => {
                self.mesh_handles.push(handle);
                self.mesh_geometry.push(geometry);
                self.mesh_sdf.push(None);
                self.mesh_infos.push(info);
                self.mesh_bounds.push((min, max));
                self.mesh_has_lightmap_uv.push(mesh.has_lightmap_uv);
                self.mesh_handles.len() - 1
            }
        };
        self.skinned_meshes.retain(|sm| sm.mesh_index != mi);
        if skinned {
            let mut sm = SkinnedMesh::new(mi, mesh.vertices.clone(), skel);
            sm.morphs = asset.morphs.get(j).cloned().unwrap_or_default();
            self.skinned_meshes.push(sm);
        }
        if at.is_some() {
            // Per-object copies of this skinned mesh rebuild from the new data next frame.
            let users: Vec<ObjectId> = self.objects.iter().filter(|o| o.render_slots().any(|r| r.mesh == mi)).map(|o| o.id).collect();
            for id in users {
                if let Some(inst) = self.anim_instances.remove(&id) {
                    free_meshes.extend(inst.slots.iter().map(|(_, h)| h.index()).chain(inst.levels.handles().map(|h| h.index())).filter(|h| *h != MeshHandle::NONE.index()));
                }
            }
        }
        Ok(mi)
    }

    /// Create the template material entry for material `k` of `asset` (one the
    /// model gained since it was installed).
    fn push_template_material(&mut self, renderer: Option<&mut Renderer>, asset: &zestengine_assets::Scene, textures: &[TextureHandle], k: usize) -> Result<usize> {
        let material = &asset.materials[k];
        let handle_of = |i: Option<usize>| i.map(|i| textures[i]).filter(|h| *h != TextureHandle::NONE);
        let desc = MaterialDesc {
            name: material.name.clone(),
            params: material.params,
            features: material.features,
            albedo: handle_of(material.albedo),
            normal: handle_of(material.normal),
            orm: handle_of(material.orm),
            emission: handle_of(material.emission),
            error: false,
            ..Default::default()
        };
        let handle = match renderer {
            Some(r) => r.create_material(&desc)?,
            None => MaterialHandle::NONE,
        };
        let model = model_from_material(&material.name, &material.params, &material.features, material.normal.is_some());
        let mut base_textures = TexSlots::default();
        base_textures[0] = desc.albedo;
        base_textures[1] = desc.normal;
        base_textures[2] = desc.orm;
        base_textures[3] = desc.emission;
        let mut base_tex_data: [Option<zestengine_render::TextureData>; 16] = Default::default();
        for (slot, src) in [material.albedo, material.normal, material.orm, material.emission].into_iter().enumerate() {
            base_tex_data[slot] = src.map(|i| asset.textures[i].clone());
        }
        self.materials.push(MaterialEntry {
            default: model.clone(),
            model,
            handle,
            emission_map_mean: [1.0; 3],
            bake_albedo: [1.0; 3],
            bake_metallic: 0.0,
            file: None,
            embedded_textures: base_textures.iter().any(Option::is_some),
            base_textures,
            bound_textures: base_textures,
            base_tex_data,
        });
        Ok(self.materials.len() - 1)
    }

    /// Apply the remap tables of freshly installed models (queued by
    /// `install_model_template_with`, which has no renderer or shader library).
    pub(crate) fn apply_pending_remaps(&mut self, renderer: Option<&mut Renderer>, shaders: &mut ShaderLibrary, project_root: &Path) {
        let pending = std::mem::take(&mut self.pending_remaps);
        let Some(renderer) = renderer else { return };
        for p in pending {
            self.apply_remap(renderer, shaders, project_root, p);
        }
    }

    /// One template material's remap: textures are written into the material's
    /// texture paths; a `.material` becomes the material every object using the
    /// template shows (the template entry itself stays, so clearing the remap
    /// brings it back).
    fn apply_remap(&mut self, renderer: &mut Renderer, shaders: &mut ShaderLibrary, project_root: &Path, p: PendingRemap) {
        let idx = p.template;
        if idx >= self.materials.len() {
            return;
        }
        if let Some(file) = &p.file {
            let abs = zestengine_core::asset_ref::resolve_path(project_root, file);
            if abs.is_file() {
                let target = self.material_from_file(renderer, shaders, &abs, project_root);
                if target != idx {
                    self.material_remaps.insert(idx, target);
                }
                return;
            }
            tracing::warn!("material remap {file} not found; keeping the imported material");
        }
        if p.textures.iter().any(Option::is_some) {
            let entry = &mut self.materials[idx];
            for (slot, path) in p.textures.iter().enumerate() {
                let Some(path) = path else { continue };
                let path = Some(PathBuf::from(path));
                for model in [&mut entry.model, &mut entry.default] {
                    match slot {
                        0 => model.textures.albedo = path.clone(),
                        1 => model.textures.normal = path.clone(),
                        2 => model.textures.orm = path.clone(),
                        _ => model.textures.emission = path.clone(),
                    }
                }
            }
            self.apply_material(renderer, shaders, project_root, idx);
        }
    }

    /// True when material `index` is what some template material of `path` was
    /// remapped to (so the saved scene should say "model material", not "file").
    pub(crate) fn is_remap_target_of(&self, path: &str, index: usize) -> bool {
        !self.material_remaps.is_empty() && self.model_slots.get(path).is_some_and(|s| s.material_idx.iter().any(|t| self.material_remaps.get(t) == Some(&index)))
    }

    /// Drop a model's generated LOD levels (`path#lod...` templates, their
    /// meshlets and the automatic chain), so they are rebuilt from the new
    /// geometry. Returns the scene materials those levels created, to be
    /// released by `release_materials` once nothing else is known to use them.
    fn refresh_lod_templates(&mut self, path: &str, free_meshes: &mut HashSet<usize>) -> HashSet<usize> {
        let prefix = format!("{path}#lod");
        let keys: Vec<String> = self.model_templates.keys().filter(|k| k.starts_with(&prefix)).cloned().collect();
        let mut materials = HashSet::new();
        for key in keys {
            if let Some(slots) = self.model_slots.get(&key) {
                materials.extend(slots.material_idx.iter().copied());
            }
            if let Some((base, mats)) = self.model_templates.remove(&key) {
                materials.extend(mats.iter().copied());
                for m in base..(base + mats.len()).min(self.mesh_handles.len()) {
                    let h = std::mem::replace(&mut self.mesh_handles[m], MeshHandle::NONE);
                    if h != MeshHandle::NONE {
                        free_meshes.insert(h.index());
                    }
                    self.skinned_meshes.retain(|sm| sm.mesh_index != m);
                }
            }
            self.model_slots.remove(&key);
            self.lod_meshlets.remove(&key);
            self.lods.forget(&key);
        }
        self.lod_failed.retain(|k| !k.starts_with(&prefix));
        self.lod_state.clear();
        self.lods.forget(path);
        materials
    }

    /// How many places use each scene material: object render slots, every
    /// model's per-mesh template materials, the remap tables (both ends), each
    /// model's own material slots, queued remaps, the default material and the
    /// `.material` cache.
    fn material_refcounts(&self) -> HashMap<usize, u32> {
        let mut n: HashMap<usize, u32> = HashMap::new();
        let mut bump = |m: usize| *n.entry(m).or_default() += 1;
        for o in &self.objects {
            o.render_slots().for_each(|r| bump(r.material));
        }
        for (_, mats) in self.model_templates.values().chain(self.builtin_template.iter()) {
            mats.iter().for_each(|m| bump(*m));
        }
        for (k, v) in &self.material_remaps {
            bump(*k);
            bump(*v);
        }
        for s in self.model_slots.values() {
            s.material_idx.iter().for_each(|m| bump(*m));
        }
        self.pending_remaps.iter().for_each(|p| bump(p.template));
        self.default_material.into_iter().for_each(&mut bump);
        self.material_file_cache.values().for_each(|m| bump(*m));
        n
    }

    /// Release the materials in `candidates` that nothing uses any more: their
    /// GPU material (the caller frees `materials`/`textures` through the
    /// renderer's one batched free), the entry's textures and data, and then
    /// any dead entries at the end of `materials`. A candidate still counted by
    /// `material_refcounts` is left alone (the replaced LOD levels share nothing
    /// by design, but an object may have been pointed at one). A texture is only
    /// freed when no live material and no cached file texture still shows it.
    pub(crate) fn release_materials(&mut self, candidates: HashSet<usize>) -> ReleasedMaterials {
        let counts = self.material_refcounts();
        let mut dead: Vec<usize> = candidates.into_iter().filter(|m| *m < self.materials.len() && !self.dead_materials.contains(m) && counts.get(m).copied().unwrap_or(0) == 0).collect();
        dead.sort_unstable();
        let mut out = ReleasedMaterials::default();
        for &m in &dead {
            let e = &mut self.materials[m];
            if e.handle != MaterialHandle::NONE {
                out.materials.insert(e.handle.index());
            }
            out.textures.extend(e.base_textures.iter().chain(e.bound_textures.iter()).flatten().map(|h| h.index()));
            e.handle = MaterialHandle::NONE;
            e.base_textures = TexSlots::default();
            e.bound_textures = TexSlots::default();
            e.base_tex_data = Default::default();
            e.file = None;
            self.dead_materials.insert(m);
        }
        let live: HashSet<usize> = self
            .materials
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.dead_materials.contains(i))
            .flat_map(|(_, m)| m.base_textures.iter().chain(m.bound_textures.iter()))
            .flatten()
            .map(|h| h.index())
            .chain(self.texture_file_cache.values().map(|h| h.index()))
            .collect();
        out.textures.retain(|t| !live.contains(t) && *t != TextureHandle::NONE.index());
        // Dead entries at the end hold no index anyone can name: drop them so the list does not grow per reimport.
        while self.materials.len().checked_sub(1).is_some_and(|last| self.dead_materials.remove(&last)) {
            self.materials.pop();
        }
        out.scene = dead;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, mesh: usize, id: SubAssetId, name: &str, at: [f32; 3]) -> zestengine_assets::SceneEntry {
        zestengine_assets::SceneEntry {
            id: String::new(),
            name: name.into(),
            source: ObjectSource::Model { path: path.into(), mesh, extra_meshes: Vec::new(), model: None, mesh_id: Some(id), extra_mesh_ids: Vec::new() },
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: Default::default(),
            layer: 0,
            material: MaterialRef::Model,
            extra_materials: Vec::new(),
            renderer_split: false,
            parent: None,
            components: Vec::new(),
            translation: at,
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

    /// A project dir with a copy of a real model, its loaded scene, and two placed instances.
    fn setup(tag: &str) -> Option<(PathBuf, LoadedScene, zestengine_assets::Scene)> {
        let sample = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../animations/kaykit/Mannequin_Medium.glb");
        if !sample.exists() {
            eprintln!("sample model missing; skipped");
            return None;
        }
        let root = std::env::temp_dir().join(format!("zest_reimport_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::copy(&sample, root.join("m.glb")).unwrap();
        let asset = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        assert!(asset.meshes.len() >= 2, "the sample has several meshes");
        let ids = asset.identity.mesh_ids.clone();
        let mut scene = LoadedScene::empty();
        let entries = [entry("m.glb", 0, ids[0], "A", [1.0, 0.0, 0.0]), entry("m.glb", 1, ids[1], "B", [0.0, 5.0, 0.0])];
        scene
            .append_entries_with(None, &mut ShaderLibrary::default(), &root, &ComponentRegistry::with_builtins(), &entries)
            .unwrap();
        Some((root, scene, asset))
    }

    fn object(scene: &LoadedScene, name: &str) -> usize {
        scene.objects.iter().position(|o| o.name == name).unwrap()
    }

    fn bounds_of(scene: &LoadedScene, mesh: usize) -> (Vec3, Vec3) {
        scene.mesh_bounds[mesh]
    }

    #[test]
    fn reimport_swaps_geometry_in_place_and_keeps_instances() {
        let Some((root, mut scene, asset)) = setup("inplace") else { return };
        let (a, b) = (object(&scene, "A"), object(&scene, "B"));
        let (mesh_a, mesh_b) = (scene.objects[a].render.unwrap().mesh, scene.objects[b].render.unwrap().mesh);
        let before = (bounds_of(&scene, mesh_a), bounds_of(&scene, mesh_b));
        let ids_before = scene.model_mesh_ids("m.glb").unwrap().to_vec();
        let object_count = scene.objects.len();

        // The "new file": everything twice as big.
        let mut bigger = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        for m in &mut bigger.meshes {
            for v in &mut m.vertices {
                v.position = v.position.map(|c| c * 2.0);
            }
        }
        let mut shaders = ShaderLibrary::default();
        let report = scene.reimport_model(None, &mut shaders, &root, "m.glb", &bigger).unwrap().expect("model is loaded");

        assert_eq!(report.meshes, asset.meshes.len());
        assert_eq!((report.added_meshes, report.missing_meshes), (0, 0));
        let after = (bounds_of(&scene, mesh_a), bounds_of(&scene, mesh_b));
        assert!((after.0.1 - before.0.1 * 2.0).length() < 1e-4 && (after.1.1 - before.1.1 * 2.0).length() < 1e-4, "geometry swapped");
        // Instance state is untouched: same objects, same slots, same overrides.
        assert_eq!(scene.objects.len(), object_count);
        assert_eq!(scene.objects[a].render.unwrap().mesh, mesh_a);
        assert_eq!(scene.objects[b].render.unwrap().mesh, mesh_b);
        assert_eq!(scene.objects[a].translation, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(scene.objects[b].translation, Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(scene.model_mesh_ids("m.glb").unwrap(), &ids_before[..], "ids are stable");
        assert_eq!(scene.mesh_infos[mesh_a].vertices as usize, bigger.meshes[0].vertices.len());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn reordered_meshes_land_in_their_own_slots_by_id() {
        let Some((root, mut scene, _)) = setup("reorder") else { return };
        let (a, b) = (object(&scene, "A"), object(&scene, "B"));
        let (mesh_a, mesh_b) = (scene.objects[a].render.unwrap().mesh, scene.objects[b].render.unwrap().mesh);
        let (bounds_a, bounds_b) = (bounds_of(&scene, mesh_a), bounds_of(&scene, mesh_b));

        // The new file lists its meshes in reverse, and mesh 1 (B) is now 3x bigger.
        let mut flipped = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        for v in &mut flipped.meshes[1].vertices {
            v.position = v.position.map(|c| c * 3.0);
        }
        flipped.meshes.reverse();
        flipped.identity.mesh_ids.reverse();
        flipped.identity.mesh_names.reverse();
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &flipped).unwrap().unwrap();
        assert_eq!((report.added_meshes, report.missing_meshes), (0, 0));
        assert_eq!(bounds_of(&scene, mesh_a), bounds_a, "A still shows its own, unchanged mesh");
        let new_b = bounds_of(&scene, mesh_b);
        assert!((new_b.1 - bounds_b.1 * 3.0).length() < 1e-3, "B got the 3x mesh with its id");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_added_mesh_gets_a_slot_and_objects_follow_their_ids() {
        let Some((root, mut scene, asset)) = setup("added") else { return };
        let (a, b) = (object(&scene, "A"), object(&scene, "B"));
        let (mesh_a, mesh_b) = (scene.objects[a].render.unwrap().mesh, scene.objects[b].render.unwrap().mesh);
        let (bounds_a, bounds_b) = (bounds_of(&scene, mesh_a), bounds_of(&scene, mesh_b));
        let handles_before = scene.mesh_handles.len();
        let old_base = scene.model_templates["m.glb"].0;

        // The new file has one more mesh (a copy of mesh 0 under a new name), and B is 3x bigger.
        let mut next = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        for v in &mut next.meshes[1].vertices {
            v.position = v.position.map(|c| c * 3.0);
        }
        let src = &next.meshes[0];
        let copy = zestengine_render::MeshData { vertices: src.vertices.clone(), indices: src.indices.clone(), has_lightmap_uv: src.has_lightmap_uv };
        next.meshes.push(copy);
        next.identity.mesh_ids.push(SubAssetId(0xABCD_0001));
        next.identity.mesh_names.push("extra#part0".into());
        let n = next.meshes.len();
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &next).unwrap().unwrap();

        assert_eq!((report.added_meshes, report.missing_meshes), (1, 0));
        assert!(report.moved);
        assert_eq!(scene.mesh_handles.len(), handles_before + n, "a fresh block for every mesh");
        let (new_base, templates) = scene.model_templates["m.glb"].clone();
        assert_eq!(new_base, handles_before);
        assert_eq!(templates.len(), n, "the template covers the new mesh");
        assert_ne!(new_base, old_base);
        assert_eq!(scene.model_mesh_ids("m.glb").unwrap().len(), n);
        // Both objects follow their mesh ids into the new block and keep their transforms.
        let (new_a, new_b) = (scene.objects[a].render.unwrap().mesh, scene.objects[b].render.unwrap().mesh);
        assert_eq!(new_a, new_base + (mesh_a - old_base));
        assert_eq!(new_b, new_base + (mesh_b - old_base));
        assert_eq!(bounds_of(&scene, new_a), bounds_a);
        assert!((bounds_of(&scene, new_b).1 - bounds_b.1 * 3.0).length() < 1e-3, "B shows the new, bigger mesh");
        assert_eq!(scene.objects[a].translation, Vec3::new(1.0, 0.0, 0.0));
        // A scene entry placed afterwards can use the new mesh.
        let entries = [entry("m.glb", n - 1, SubAssetId(0xABCD_0001), "C", [0.0, 0.0, 9.0])];
        scene.append_entries_with(None, &mut ShaderLibrary::default(), &root, &ComponentRegistry::with_builtins(), &entries).unwrap();
        assert_eq!(scene.objects[object(&scene, "C")].render.unwrap().mesh, new_base + n - 1);
        // The old block is released (its slots are tombstones, not reused).
        assert!(asset.meshes.len() >= 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_removed_mesh_keeps_its_objects_and_is_counted() {
        let Some((root, mut scene, asset)) = setup("removed") else { return };
        let mut changed = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        // Drop the last mesh.
        let last = asset.meshes.len() - 1;
        let last_id = changed.identity.mesh_ids[last];
        changed.meshes.truncate(last);
        changed.identity.mesh_ids.truncate(last);
        changed.identity.mesh_names.truncate(last);
        // Place an object on the mesh that is about to go.
        let entries = [entry("m.glb", last, last_id, "Gone", [3.0, 0.0, 0.0])];
        scene.append_entries_with(None, &mut ShaderLibrary::default(), &root, &ComponentRegistry::with_builtins(), &entries).unwrap();
        let gone = object(&scene, "Gone");
        let gone_mesh = scene.objects[gone].render.unwrap().mesh;
        let verts = scene.mesh_infos[gone_mesh].vertices;
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &changed).unwrap().unwrap();
        assert_eq!((report.added_meshes, report.missing_meshes), (0, 1));
        assert!(!report.moved, "nothing was added, so the block stays put");
        assert_eq!(report.meshes, last);
        assert_eq!(report.orphaned_objects, 1);
        assert_eq!(scene.objects[gone].render.unwrap().mesh, gone_mesh, "the object keeps its mesh");
        assert_eq!(scene.mesh_infos[gone_mesh].vertices, verts, "with the geometry it had");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn materials_are_paired_by_id_not_by_position() {
        let Some((root, mut scene, asset)) = setup("matid") else { return };
        assert!(asset.materials.len() >= 2);
        let slots = scene.model_slots["m.glb"].clone();
        scene.materials[slots.material_idx[0]].model.roughness = 0.123;
        let first_name = scene.materials[slots.material_idx[0]].default.name.clone();
        let mut next = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        for m in &mut next.materials {
            m.params.roughness = 0.9;
        }
        // The file now lists its materials in reverse; ids follow the names.
        next.materials.reverse();
        next.identity.material_ids.reverse();
        next.identity.material_names.reverse();
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &next).unwrap().unwrap();
        assert_eq!(report.materials_added, 0);
        let entry = &scene.materials[slots.material_idx[0]];
        assert_eq!(entry.default.name, first_name, "the first template material is still the same material");
        assert_eq!(entry.model.roughness, 0.123, "and keeps the user's edit");
        // Slots now list the new file's order.
        let now = &scene.model_slots["m.glb"];
        assert_eq!(now.material_idx[now.material_idx.len() - 1], slots.material_idx[0]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_new_material_is_created_and_a_removed_remap_restores_the_template() {
        let Some((root, mut scene, asset)) = setup("newmat") else { return };
        let slots = scene.model_slots["m.glb"].clone();
        let (a, mesh_a) = (object(&scene, "A"), scene.objects[object(&scene, "A")].render.unwrap().mesh);
        let template = scene.model_templates["m.glb"].1[mesh_a - scene.model_templates["m.glb"].0];
        // Pretend the template material was remapped to another material.
        let other = slots.material_idx[slots.material_idx.len() - 1].max(template + 1).min(scene.materials.len() - 1);
        scene.material_remaps.insert(template, other);
        scene.objects[a].render.as_mut().unwrap().material = other;
        assert!(scene.is_remap_target_of("m.glb", other));

        let mut next = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        let src = &next.materials[0];
        let extra = zestengine_assets::SceneMaterial { name: "Brand New".into(), params: src.params, features: src.features, albedo: None, normal: None, orm: None, emission: None };
        next.materials.push(extra);
        next.identity.material_names.push("Brand New".into());
        next.identity.material_ids.push(SubAssetId(0x1234_5678));
        let before = scene.materials.len();
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &next).unwrap().unwrap();
        assert_eq!(report.materials_added, 1);
        assert_eq!(scene.materials.len(), before + 1);
        assert_eq!(scene.materials[before].default.name, "Brand New");
        assert_eq!(scene.model_slots["m.glb"].material_idx.len(), asset.materials.len() + 1);
        // The new file has no remap, so the object shows the template material again.
        assert!(scene.material_remaps.is_empty());
        assert_eq!(scene.objects[a].render.unwrap().material, template);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn generated_lod_templates_are_refreshed() {
        let Some((root, mut scene, _)) = setup("lod") else { return };
        // A generated level installed earlier: its template, meshlets and a failure record.
        let lod_key = "m.glb#lod0.500".to_owned();
        let lod_base = scene.mesh_handles.len();
        scene.mesh_handles.push(MeshHandle::NONE);
        scene.mesh_geometry.push((Vec::new(), Vec::new()));
        scene.mesh_sdf.push(None);
        scene.mesh_infos.push(MeshInfo { vertices: 0, triangles: 0 });
        scene.mesh_bounds.push((Vec3::ZERO, Vec3::ZERO));
        scene.mesh_has_lightmap_uv.push(false);
        scene.model_templates.insert(lod_key.clone(), (lod_base, vec![0]));
        scene.lod_meshlets.insert(lod_key.clone(), vec![Vec::new()]);
        scene.lod_failed.insert("m.glb#lod0.250".to_owned());
        scene.lod_failed.insert("other.glb#lod0.500".to_owned());
        scene.lods.models.insert("m.glb".to_owned(), None);

        let next = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &next).unwrap().unwrap();
        assert!(!scene.model_templates.contains_key(&lod_key), "the stale level is dropped");
        assert!(!scene.lod_meshlets.contains_key(&lod_key));
        assert!(!scene.lod_failed.contains("m.glb#lod0.250"), "failures are retried");
        assert!(scene.lod_failed.contains("other.glb#lod0.500"), "other models are untouched");
        assert!(!scene.lods.models.contains_key("m.glb"), "the automatic chain is rebuilt");
        assert!(scene.model_templates.contains_key("m.glb"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A generated LOD level of `asset`: same geometry layout, texture-less
    /// materials (what `auto_lod::geometry_variant` builds).
    fn lod_level_of(root: &Path) -> zestengine_assets::Scene {
        let asset = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        let mut level = asset;
        level.textures.clear();
        level.animations.clear();
        level.collision.clear();
        for m in &mut level.materials {
            (m.albedo, m.normal, m.orm, m.emission) = (None, None, None, None);
        }
        level.identity = zestengine_assets::ModelIdentity { guid: level.identity.guid, mesh_names: level.identity.mesh_names.clone(), mesh_ids: level.identity.mesh_ids.clone(), ..Default::default() };
        level
    }

    fn live_materials(scene: &LoadedScene) -> usize {
        scene.materials.len() - scene.dead_materials.len()
    }

    #[test]
    fn repeated_reimports_do_not_accumulate_lod_level_materials() {
        let Some((root, mut scene, asset)) = setup("lodmats") else { return };
        let (base_live, base_len) = (live_materials(&scene), scene.materials.len());
        let level = lod_level_of(&root);
        let key = "m.glb#lod0.500";
        for round in 0..5 {
            // update_auto_lods installs the level lazily after each reimport.
            scene.install_model_template_with(None, key, &level, None).unwrap();
            assert_eq!(live_materials(&scene), base_live + level.materials.len(), "round {round}: the level added its materials");
            let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &asset).unwrap().unwrap();
            assert_eq!(report.freed_materials, level.materials.len(), "round {round}");
            assert_eq!(live_materials(&scene), base_live, "round {round}: the level's materials were released");
            assert_eq!(scene.materials.len(), base_len, "round {round}: and their entries reclaimed, not left as tombstones");
            assert!(!scene.model_templates.contains_key(key) && !scene.model_slots.contains_key(key));
        }
        // The model's own materials are untouched and still addressable.
        let slots = &scene.model_slots["m.glb"];
        assert!(slots.material_idx.iter().all(|&i| i < scene.materials.len() && !scene.dead_materials.contains(&i)));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_lod_material_that_something_still_uses_is_not_released() {
        let Some((root, mut scene, _asset)) = setup("lodshared") else { return };
        let base_live = live_materials(&scene);
        let level = lod_level_of(&root);
        let key = "m.glb#lod0.500";
        scene.install_model_template_with(None, key, &level, None).unwrap();
        let lod_mats = scene.model_slots[key].material_idx.clone();
        assert!(lod_mats.len() >= 2);
        // An object pointed at one of the level's materials, and a remap that targets another.
        let a = object(&scene, "A");
        scene.objects[a].render.as_mut().unwrap().material = lod_mats[0];
        let template = scene.model_slots["m.glb"].material_idx[0];
        scene.material_remaps.insert(template, lod_mats[1]);
        let released = scene.release_materials(lod_mats.iter().copied().collect());
        assert_eq!(released.scene.len(), lod_mats.len() - 2, "the two in use stay");
        assert!(!released.scene.contains(&lod_mats[0]) && !released.scene.contains(&lod_mats[1]));
        assert!(!scene.dead_materials.contains(&lod_mats[0]) && !scene.dead_materials.contains(&lod_mats[1]));
        assert_eq!(live_materials(&scene), base_live + 2);
        // Releasing again frees nothing twice.
        assert!(scene.release_materials(lod_mats.iter().copied().collect()).scene.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn edited_template_material_keeps_edits_and_untouched_follows_the_file() {
        let Some((root, mut scene, asset)) = setup("mats") else { return };
        assert!(asset.materials.len() >= 2, "the sample has several materials");
        let slots = scene.model_slots.get("m.glb").unwrap().clone();
        let (edited, plain) = (slots.material_idx[0], slots.material_idx[1]);
        scene.materials[edited].model.roughness = 0.123;
        let mut next = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        for m in &mut next.materials {
            m.params.roughness = 0.9;
        }
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &next).unwrap().unwrap();
        assert!(report.materials >= 2);
        assert_eq!(scene.materials[edited].model.roughness, 0.123, "the user's edit survives");
        assert_eq!(scene.materials[edited].default.roughness, 0.9, "but the baseline follows the file");
        assert_eq!(scene.materials[plain].model.roughness, 0.9, "an untouched material takes the new import");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn skeleton_and_clips_are_replaced_by_name() {
        let Some((root, mut scene, asset)) = setup("skel") else { return };
        if asset.skeletons.is_empty() {
            return;
        }
        let slots = scene.model_slots.get("m.glb").unwrap().clone();
        let mut next = zestengine_assets::load_model_with_meta(root.join("m.glb")).unwrap();
        next.skeletons[0].joints[0].rest_translation = Vec3::new(7.0, 8.0, 9.0);
        next.animations.push(zestengine_assets::AnimationClip { name: "brand new".into(), duration: 1.0, channels: Vec::new(), morph_tracks: Vec::new(), playback: None, });
        let clips_before = scene.animations.len();
        let report = scene.reimport_model(None, &mut ShaderLibrary::default(), &root, "m.glb", &next).unwrap().unwrap();
        assert_eq!(scene.skeletons[slots.skel_idx[0]].joints[0].rest_translation, Vec3::new(7.0, 8.0, 9.0));
        assert_eq!(report.clips_added, 1);
        assert_eq!(scene.animations.len(), clips_before + 1);
        assert_eq!(scene.clip_skeleton.len(), scene.animations.len());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unloaded_model_is_not_reimported() {
        let mut scene = LoadedScene::empty();
        let asset = zestengine_assets::test_scene();
        let out = scene.reimport_model(None, &mut ShaderLibrary::default(), Path::new("."), "nope.glb", &asset).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn pairs_prefer_names_then_positions() {
        // Reordered: names win.
        assert_eq!(pair_by_name(&["a", "b", "c"], &["c", "a", "b"]), vec![(0, 2), (1, 0), (2, 1)]);
        // A renamed one falls back to its position; extras stay unpaired.
        assert_eq!(pair_by_name(&["a", "b"], &["a", "x", "y"]), vec![(0, 0), (1, 1)]);
        // Never reuses an old slot.
        assert_eq!(pair_by_name(&["a"], &["a", "a"]), vec![(0, 0)]);
    }
}
