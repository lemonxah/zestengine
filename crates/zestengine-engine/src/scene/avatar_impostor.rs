//! Impostor stand-ins for avatars (GEO-5), the step between the full model
//! and the rank-tinted capsule of the avatar safety policy.
//!
//! The fallback chain, nearest to farthest / best to worst:
//!   1. the full cluster cut (or the ordinary draw) of the avatar,
//!   2. a coarser discrete LOD (the animation / skin LOD of GEO-2),
//!   3. an octahedral impostor: the avatar's REST pose captured from 8 x 8
//!      directions, drawn as one billboard that turns towards the camera,
//!   4. the rank-tinted capsule (`perf.rs`).
//!
//! A hidden avatar (`FallbackReason::Rank` / `Distance` / `TooMany`) or one past
//! `AvatarPolicy::impostor_distance` (`Far`) gets the impostor once its capture
//! is ready (a worker thread captures the merged bind-pose meshes and the
//! first frames still show the capsule); if the capture or the upload fails it
//! keeps the capsule. Impostors are keyed by the avatar's mesh set, so every
//! copy of one model shares one atlas.
//!
//! An impostor of an animated avatar is a statue turning to face the camera:
//! the pose in motion is not shown. It is a distance / safety stand-in only.
//! It casts no shadow and takes no part in the cluster path.

use std::collections::HashSet;
use std::sync::mpsc::{Receiver, TryRecvError, channel};

use zestengine_assets::impostor::{Impostor, ImpostorSettings};
use zestengine_render::{ImpostorHandle, ImpostorUpload, MeshData, Vertex};

use super::perf::rank_tint;
use super::*;

/// What stands in for a replaced avatar this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandIn {
    /// Not replaced.
    Full,
    Impostor,
    Capsule,
}

/// The fallback chain's last two links: an avatar the policy replaced is an
/// impostor once its atlas is on the GPU, a capsule until then (or when it failed).
pub fn stand_in(replaced: bool, impostor_ready: bool) -> StandIn {
    match (replaced, impostor_ready) {
        (false, _) => StandIn::Full,
        (true, true) => StandIn::Impostor,
        (true, false) => StandIn::Capsule,
    }
}

/// One mesh of an avatar: vertices, indices, the transform into the avatar
/// root's space and the material's base colour.
pub struct Part<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u32],
    pub to_root: Mat4,
    pub base_color: [f32; 4],
}

/// Merge the parts into one mesh in the avatar root's space: positions and
/// normals transformed, vertex colours multiplied by the material colour (the
/// capture does not read textures).
pub fn merge_parts(parts: &[Part]) -> MeshData {
    let mut out = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
    for part in parts {
        let base = out.vertices.len() as u32;
        let normal_matrix = glam::Mat3::from_mat4(part.to_root).inverse().transpose();
        for v in part.vertices {
            let mut v = *v;
            v.position = part.to_root.transform_point3(Vec3::from(v.position)).to_array();
            v.normal = (normal_matrix * Vec3::from(v.normal)).normalize_or_zero().to_array();
            for k in 0..4 {
                v.color[k] *= part.base_color[k];
            }
            out.vertices.push(v);
        }
        out.indices.extend(part.indices.iter().map(|i| i + base));
    }
    out
}

/// Smooth vertex normals of a position-only mesh (static avatar parts keep no normals in the scene).
fn vertices_from_positions(positions: &[Vec3], indices: &[u32]) -> Vec<Vertex> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for t in indices.as_chunks::<3>().0 {
        let (Some(&a), Some(&b), Some(&c)) = (positions.get(t[0] as usize), positions.get(t[1] as usize), positions.get(t[2] as usize)) else { continue };
        let n = (b - a).cross(c - a);
        for &i in t {
            if let Some(slot) = normals.get_mut(i as usize) {
                *slot += n;
            }
        }
    }
    positions
        .iter()
        .zip(&normals)
        .map(|(p, n)| Vertex { position: p.to_array(), normal: n.normalize_or(Vec3::Y).to_array(), color: [1.0; 4], ..Default::default() })
        .collect()
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Key(Vec<usize>);

enum Entry {
    Capturing,
    Ready(ImpostorHandle),
    Failed,
}

#[derive(Default)]
pub(crate) struct ImpostorRuntime {
    entries: HashMap<Key, Entry>,
    jobs: Vec<(Key, Receiver<Option<Impostor>>)>,
    /// Hidden avatars an impostor is drawn for this frame (the capsule skips them).
    drawn: HashSet<ObjectId>,
    /// Avatar count at the last sweep for atlases no avatar uses any more
    /// (`Renderer::reset_scene` frees every atlas on a scene switch).
    swept_avatars: usize,
}

/// Keys of `entries` that no avatar in `live` has (the model was unloaded).
fn stale_keys(entries: &HashMap<Key, Entry>, live: &HashSet<Key>) -> Vec<Key> {
    entries.keys().filter(|k| !live.contains(*k)).cloned().collect()
}

impl ImpostorRuntime {
    pub(crate) fn is_drawn(&self, id: ObjectId) -> bool {
        self.drawn.contains(&id)
    }

    /// Free the atlases whose model no live avatar has any more.
    fn release_stale(&mut self, renderer: &mut Renderer, live: &HashSet<Key>) -> usize {
        let stale = stale_keys(&self.entries, live);
        for k in &stale {
            if let Some(Entry::Ready(h)) = self.entries.remove(k) {
                renderer.free_impostor(h);
            }
        }
        self.jobs.retain(|(k, _)| !stale.contains(k));
        stale.len()
    }

    /// Atlases uploaded / being captured / failed (tests, stats).
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut c = (0, 0, 0);
        for e in self.entries.values() {
            match e {
                Entry::Ready(_) => c.0 += 1,
                Entry::Capturing => c.1 += 1,
                Entry::Failed => c.2 += 1,
            }
        }
        c
    }
}

impl LoadedScene {
    /// Finish captures, start missing ones and queue this frame's impostor
    /// billboards for the hidden avatars (`perf.proxies`). Cheap when no avatar
    /// is hidden.
    pub(crate) fn update_avatar_impostors(&mut self, renderer: &mut Renderer) {
        self.avatar_impostors.drawn.clear();
        // An avatar left (an origin unloaded): free the atlases of models nobody wears any more.
        if self.perf.avatars.len() != self.avatar_impostors.swept_avatars {
            self.avatar_impostors.swept_avatars = self.perf.avatars.len();
            if !self.avatar_impostors.entries.is_empty() {
                let roots: Vec<usize> = self.perf.avatars.iter().filter_map(|s| self.index_of(s.object)).collect();
                let live: HashSet<Key> = roots.into_iter().map(|i| self.avatar_impostor_key(i)).collect();
                let freed = self.avatar_impostors.release_stale(renderer, &live);
                if freed > 0 {
                    let (ready, capturing, failed) = self.avatar_impostors.counts();
                    tracing::debug!("freed {freed} avatar impostor atlases ({ready} ready, {capturing} capturing, {failed} failed left)");
                }
            }
        }
        // Finished captures: upload them.
        let mut finished = Vec::new();
        self.avatar_impostors.jobs.retain(|(key, rx)| match rx.try_recv() {
            Ok(result) => {
                finished.push((key.clone(), result));
                false
            }
            Err(TryRecvError::Empty) => true,
            Err(TryRecvError::Disconnected) => {
                finished.push((key.clone(), None));
                false
            }
        });
        for (key, result) in finished {
            let entry = match result {
                Some(imp) => match renderer.upload_impostor(&ImpostorUpload {
                    grid: imp.grid,
                    tile: imp.tile,
                    centre: imp.centre,
                    radius: imp.radius,
                    albedo: &imp.albedo,
                    aux: &imp.aux,
                }) {
                    Ok(h) => Entry::Ready(h),
                    Err(e) => {
                        tracing::warn!("avatar impostor upload: {e:#}");
                        Entry::Failed
                    }
                },
                None => Entry::Failed,
            };
            self.avatar_impostors.entries.insert(key, entry);
        }
        for k in 0..self.perf.proxies.len() {
            let (id, _, _, rank) = self.perf.proxies[k];
            let Some(i) = self.index_of(id).filter(|&i| self.is_active(i)) else { continue };
            let key = self.avatar_impostor_key(i);
            let ready = matches!(self.avatar_impostors.entries.get(&key), Some(Entry::Ready(_)));
            match self.avatar_impostors.entries.get(&key) {
                // The chain's last two links: an impostor once its atlas is up, the capsule until then.
                Some(Entry::Ready(handle)) if stand_in(true, ready) == StandIn::Impostor => {
                    let t = rank_tint(rank);
                    renderer.draw_impostor(*handle, self.frame_world(i), [t[0], t[1], t[2], 0.15]);
                    self.avatar_impostors.drawn.insert(id);
                }
                Some(_) => {}
                None => {
                    let mesh = self.avatar_bind_mesh(i);
                    if mesh.indices.len() < 3 {
                        self.avatar_impostors.entries.insert(key, Entry::Failed);
                        continue;
                    }
                    let (tx, rx) = channel();
                    let spawned = std::thread::Builder::new().name("avatar-impostor".into()).spawn(move || {
                        let captured = match Impostor::capture(&mesh, [1.0; 4], &ImpostorSettings::default()) {
                            Ok(imp) => Some(imp),
                            Err(e) => {
                                tracing::warn!("avatar impostor capture: {e}");
                                None
                            }
                        };
                        let _ = tx.send(captured);
                    });
                    match spawned {
                        Ok(_) => {
                            self.avatar_impostors.entries.insert(key.clone(), Entry::Capturing);
                            self.avatar_impostors.jobs.push((key, rx));
                        }
                        Err(e) => {
                            tracing::warn!("avatar impostor thread: {e}");
                            self.avatar_impostors.entries.insert(key, Entry::Failed);
                        }
                    }
                }
            }
        }
    }

    /// Identity of an avatar's geometry: its sorted mesh set (copies of one model share it).
    fn avatar_impostor_key(&self, root: usize) -> Key {
        let mut meshes: Vec<usize> = (0..self.objects.len()).filter(|&j| self.is_within(j, root)).flat_map(|j| self.objects[j].render_slots().map(|r| r.mesh)).collect();
        meshes.sort_unstable();
        meshes.dedup();
        Key(meshes)
    }

    /// The avatar's meshes at their bind pose, merged in the root's space.
    fn avatar_bind_mesh(&self, root: usize) -> MeshData {
        type OwnedPart = (Vec<Vertex>, Vec<u32>, Mat4, [f32; 4]);
        let inv_root = self.frame_world(root).inverse();
        let mut owned: Vec<OwnedPart> = Vec::new();
        for j in 0..self.objects.len() {
            if !self.is_within(j, root) {
                continue;
            }
            let to_root = inv_root * self.frame_world(j);
            for r in self.objects[j].render_slots() {
                let Some((positions, indices)) = self.mesh_geometry.get(r.mesh) else { continue };
                let vertices = match self.skinned_meshes.iter().find(|sm| sm.mesh_index == r.mesh) {
                    Some(sm) => sm.base_vertices.clone(),
                    None => vertices_from_positions(positions, indices),
                };
                let c = self.materials.get(r.material).map_or([0.7; 3], |m| m.bake_albedo);
                owned.push((vertices, indices.clone(), to_root, [c[0], c[1], c[2], 1.0]));
            }
        }
        let parts: Vec<Part> = owned.iter().map(|(v, i, m, c)| Part { vertices: v, indices: i, to_root: *m, base_color: *c }).collect();
        merge_parts(&parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(offset: f32) -> (Vec<Vertex>, Vec<u32>) {
        let v = |x: f32, y: f32| Vertex { position: [x + offset, y, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0; 4], ..Default::default() };
        (vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)], vec![0, 1, 2, 0, 2, 3])
    }

    #[test]
    fn the_chain_ends_in_the_capsule_until_the_impostor_exists() {
        assert_eq!(stand_in(false, false), StandIn::Full);
        assert_eq!(stand_in(false, true), StandIn::Full, "an impostor never replaces an avatar the policy shows");
        assert_eq!(stand_in(true, false), StandIn::Capsule);
        assert_eq!(stand_in(true, true), StandIn::Impostor);
    }

    #[test]
    fn atlases_of_unloaded_models_are_stale() {
        let mut entries: HashMap<Key, Entry> = HashMap::new();
        entries.insert(Key(vec![1, 2]), Entry::Failed);
        entries.insert(Key(vec![3]), Entry::Capturing);
        entries.insert(Key(vec![4, 5]), Entry::Failed);
        let live: HashSet<Key> = [Key(vec![3])].into_iter().collect();
        let mut stale = stale_keys(&entries, &live);
        stale.sort_by_key(|k| k.0.clone());
        assert_eq!(stale, vec![Key(vec![1, 2]), Key(vec![4, 5])]);
        assert!(stale_keys(&entries, &entries.keys().cloned().collect()).is_empty());
    }

    #[test]
    fn merged_parts_are_in_root_space_with_material_colour() {
        let (v0, i0) = quad(0.0);
        let (v1, i1) = quad(0.0);
        let rotate = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2) * Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        let merged = merge_parts(&[
            Part { vertices: &v0, indices: &i0, to_root: Mat4::IDENTITY, base_color: [1.0, 0.5, 0.25, 1.0] },
            Part { vertices: &v1, indices: &i1, to_root: rotate, base_color: [0.2, 0.2, 0.2, 1.0] },
        ]);
        assert_eq!(merged.vertices.len(), 8);
        assert_eq!(merged.indices.len(), 12);
        assert_eq!(merged.indices[6], 4, "the second part's indices are offset");
        assert_eq!(merged.vertices[1].color, [1.0, 0.5, 0.25, 1.0]);
        // (1, 0, 0) -> translate (1, 2, 0) -> rotate 90 degrees about Y: (0, 2, -1).
        let p = Vec3::from(merged.vertices[5].position);
        assert!((p - Vec3::new(0.0, 2.0, -1.0)).length() < 1e-5, "{p:?}");
        // The +Z normal turned with it: +X.
        let n = Vec3::from(merged.vertices[5].normal);
        assert!((n - Vec3::X).length() < 1e-5, "{n:?}");
    }

    #[test]
    fn position_only_parts_get_smooth_normals() {
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        let v = vertices_from_positions(&positions, &[0, 1, 2]);
        assert!(v.iter().all(|v| (Vec3::from(v.normal) - Vec3::Z).length() < 1e-5));
        // A degenerate triangle falls back to up instead of NaN.
        let d = vertices_from_positions(&positions, &[0, 0, 0]);
        assert!(d.iter().all(|v| v.normal.iter().all(|c| c.is_finite())));
    }
}
