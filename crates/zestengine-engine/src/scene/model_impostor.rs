//! Octahedral impostors for ordinary static models beyond their last LOD
//! (GEO-3 / GEO-5). The render side (`render/src/impostor.rs`: atlas upload,
//! billboard pipeline, textured capture through the renderer) and the avatar
//! fallback chain (`avatar_impostor.rs`) exist; this is the same machinery for
//! scenery: a mesh that covers less than `lod::IMPOSTOR_COVERAGE` of the
//! screen (the automatic chain's last level serves everything under 0.05)
//! draws as one billboard of its captured atlas.
//!
//! * One atlas per (mesh, material) slot, captured in the mesh's own space
//!   with the engine's own pipelines and the material's textures
//!   (`Renderer::start_impostor_capture`, one pass per frame inside the frame
//!   loop, so a capture never stalls it). Every object that draws the mesh
//!   shares the atlas. Without the `client` feature, or when the capture is
//!   refused, the atlas is the CPU rasteriser's (flat albedo).
//! * `update_model_impostors` (once per frame) starts the captures the last
//!   `sync_draws` asked for, uploads finished ones and frees atlases whose mesh
//!   nothing draws any more; `sync_draws` decides per object (coverage with
//!   hysteresis) and queues the billboards; `flush_model_impostors` hands them
//!   to the renderer right after `sync_draws`, in the same frame.
//! * A swapped object keeps a shadow-only draw of its last mesh (impostors cast
//!   no shadows). Skinned, lightmapped, cloth, transparent, tinted, selected
//!   and planar-reflection objects, and authored LOD Groups, never swap.
//! * Off by default: the shipped game and play mode turn it on
//!   (`set_model_impostors`); `ZESTENGINE_NO_MODEL_IMPOSTORS=1` forces it off.

use std::collections::{HashMap, HashSet};

use zestengine_assets::impostor::{Impostor, ImpostorSettings};
use zestengine_core::lod;
use zestengine_render::{ImpostorHandle, ImpostorUpload};

use super::*;

/// Meshes below this many triangles are not worth an atlas (the billboard saves nothing).
pub const MIN_TRIANGLES: u32 = 400;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Slot {
    pub mesh: usize,
    pub material: usize,
}

enum Entry {
    /// A capture was requested and has not finished.
    Capturing,
    Ready(ImpostorHandle),
    Failed,
}

#[derive(Default)]
pub(crate) struct ModelImpostors {
    pub(crate) enabled: bool,
    entries: HashMap<Slot, Entry>,
    /// Slots the last `sync_draws` wants an atlas for (coverage under the prepare threshold).
    wanted: HashSet<Slot>,
    /// Slots drawn (swapped or wanted) in the last `sync_draws`, for freeing the rest.
    #[cfg(feature = "client")]
    tickets: Vec<(Slot, zestengine_render::CaptureTicket, [f32; 3], f32)>,
    /// Objects drawn as billboards last frame (hysteresis).
    swapped: HashSet<ObjectId>,
    /// This frame's billboards (handle, transform), flushed after `sync_draws`.
    queued: Vec<(ImpostorHandle, Mat4)>,
    /// Objects that swapped in the last `sync_draws` (tests, stats).
    pub(crate) swapped_count: usize,
    /// Object count at the last sweep for atlases nothing draws any more.
    swept: usize,
}

impl ModelImpostors {
    /// (ready, capturing, failed).
    pub(crate) fn counts(&self) -> (usize, usize, usize) {
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

    fn ready(&self, slot: Slot) -> Option<ImpostorHandle> {
        match self.entries.get(&slot) {
            Some(Entry::Ready(h)) => Some(*h),
            _ => None,
        }
    }
}

fn env_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("ZESTENGINE_NO_MODEL_IMPOSTORS").is_ok_and(|v| v != "0"))
}

impl LoadedScene {
    /// Turn the scenery impostors on (play, the shipped game) or off (the editor viewport).
    pub fn set_model_impostors(&mut self, on: bool) {
        self.model_impostors.enabled = on && !env_off();
    }

    /// (atlases ready, capturing, failed).
    pub fn model_impostor_counts(&self) -> (usize, usize, usize) {
        self.model_impostors.counts()
    }

    /// Objects drawn as a billboard in the last `sync_draws`.
    pub fn model_impostors_drawn(&self) -> usize {
        self.model_impostors.swapped_count
    }

    /// Start of `sync_draws`: forget last frame's queue.
    pub(crate) fn model_impostors_begin(&mut self) {
        self.model_impostors.queued.clear();
        self.model_impostors.wanted.clear();
        self.model_impostors.swapped_count = 0;
    }

    /// Whether object `i` (at `transform`) draws as billboards this frame. When
    /// it does, the billboards are queued and the caller draws only shadows.
    /// `lod_mesh` maps a slot's mesh to the one the LOD pick chose.
    pub(crate) fn model_impostor_swap(&mut self, i: usize, transform: &Mat4, highlight: f32, planar: bool, instance_tinted: bool) -> bool {
        if !self.model_impostors.enabled || highlight > 0.0 || planar || instance_tinted {
            return false;
        }
        let o = &self.objects[i];
        let Some(render) = o.render else { return false };
        if !matches!(o.source, ObjectSource::Model { .. }) || o.components.iter().any(|c| c.as_any().is::<zestengine_core::LodComponent>()) {
            return false;
        }
        if self.baked.as_ref().is_some_and(|b| b.object_lightmap.contains_key(&i)) || crate::cloth::has_mesh_cloth(&o.components) {
            return false;
        }
        let id = o.id;
        let slots: Vec<Slot> = o.render_slots().map(|r| Slot { mesh: r.mesh, material: r.material }).collect();
        if slots.is_empty() {
            return false;
        }
        for s in &slots {
            let skinned = self.skinned_meshes.iter().any(|sm| sm.mesh_index == s.mesh);
            let blended = self.materials.get(s.material).is_none_or(|m| m.model.render_queue >= zestengine_render::RENDER_QUEUE_TRANSPARENT);
            if skinned || blended || self.mesh_infos.get(s.mesh).is_none_or(|m| m.triangles < MIN_TRIANGLES) {
                return false;
            }
        }
        let Some(coverage) = self.screen_coverage_of(transform, render.mesh) else { return false };
        if coverage < lod::IMPOSTOR_PREPARE_COVERAGE {
            self.model_impostors.wanted.extend(slots.iter().copied());
        }
        let was = self.model_impostors.swapped.contains(&id);
        if !lod::use_impostor(coverage, was) {
            self.model_impostors.swapped.remove(&id);
            return false;
        }
        let handles: Option<Vec<ImpostorHandle>> = slots.iter().map(|s| self.model_impostors.ready(*s)).collect();
        let Some(handles) = handles else {
            self.model_impostors.swapped.remove(&id);
            return false;
        };
        self.model_impostors.swapped.insert(id);
        self.model_impostors.swapped_count += 1;
        for h in handles {
            self.model_impostors.queued.push((h, *transform));
        }
        true
    }

    /// Queue this frame's billboards (call right after `sync_draws`).
    pub fn flush_model_impostors(&mut self, renderer: &mut Renderer) {
        for (h, m) in self.model_impostors.queued.drain(..) {
            renderer.draw_impostor(h, m, [0.0; 4]);
        }
    }

    /// Start the captures `sync_draws` asked for, upload finished ones, free the
    /// atlases of meshes that left the scene. Cheap when nothing is wanted.
    pub fn update_model_impostors(&mut self, renderer: &mut Renderer) {
        if !self.model_impostors.enabled {
            return;
        }
        if self.objects.len() != self.model_impostors.swept {
            self.model_impostors.swept = self.objects.len();
            self.release_model_impostors(renderer);
        }
        self.finish_model_captures(renderer);
        let wanted: Vec<Slot> = self.model_impostors.wanted.iter().copied().filter(|s| !self.model_impostors.entries.contains_key(s)).collect();
        // One capture at a time: each is a full pass per view row, and the first also compiles pipelines.
        let busy = self.model_impostors.entries.values().any(|e| matches!(e, Entry::Capturing));
        if let Some(slot) = wanted.first().filter(|_| !busy) {
            self.start_model_capture(renderer, *slot);
        }
    }

    fn mesh_for_capture(&self, mesh: usize) -> Option<zestengine_render::MeshData> {
        let (positions, indices) = self.mesh_geometry.get(mesh)?;
        if indices.len() < 3 {
            return None;
        }
        let mut normals = vec![Vec3::ZERO; positions.len()];
        for t in indices.as_chunks::<3>().0 {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            let n = (b - a).cross(c - a);
            for &k in t {
                normals[k as usize] += n;
            }
        }
        Some(zestengine_render::MeshData {
            vertices: positions.iter().zip(&normals).map(|(p, n)| zestengine_render::Vertex { position: p.to_array(), normal: n.normalize_or(Vec3::Y).to_array(), color: [1.0; 4], ..Default::default() }).collect(),
            indices: indices.clone(),
            has_lightmap_uv: false,
        })
    }

    fn start_model_capture(&mut self, renderer: &mut Renderer, slot: Slot) {
        let Some(&(mn, mx)) = self.mesh_bounds.get(slot.mesh) else {
            self.model_impostors.entries.insert(slot, Entry::Failed);
            return;
        };
        let (centre, radius) = ((mn + mx) * 0.5, (mx - mn).length() * 0.5);
        #[cfg(feature = "client")]
        {
            let (Some(&mesh), Some(material)) = (self.mesh_handles.get(slot.mesh), self.materials.get(slot.material)) else {
                self.model_impostors.entries.insert(slot, Entry::Failed);
                return;
            };
            let settings = ImpostorSettings::default();
            let draw = DrawCmd {
                gi_source: None,
                mesh,
                material: material.handle,
                transform: Mat4::IDENTITY,
                highlight: 0.0,
                mesh_center: centre,
                bound_radius: radius,
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index: -1,
                occluded: false,
                shadow_casting: zestengine_render::ShadowCasting::Off,
                instance_color: None,
                instance_param: None,
            };
            let req = zestengine_render::ImpostorCaptureRequest { grid: settings.grid, tile: settings.tile, centre: centre.to_array(), radius, draws: vec![draw] };
            match renderer.start_impostor_capture(req) {
                Ok(ticket) => {
                    self.model_impostors.entries.insert(slot, Entry::Capturing);
                    self.model_impostors.tickets.push((slot, ticket, centre.to_array(), radius));
                    return;
                }
                Err(e) => tracing::debug!("impostor capture refused for mesh {}: {e:#}; using the CPU capture", slot.mesh),
            }
        }
        self.cpu_model_capture(renderer, slot);
    }

    /// Flat-albedo capture on the CPU (no renderer capture available or it failed).
    fn cpu_model_capture(&mut self, renderer: &mut Renderer, slot: Slot) {
        let Some(mesh) = self.mesh_for_capture(slot.mesh) else {
            self.model_impostors.entries.insert(slot, Entry::Failed);
            return;
        };
        let c = self.materials.get(slot.material).map_or([0.7; 3], |m| m.bake_albedo);
        let entry = match Impostor::capture(&mesh, [c[0], c[1], c[2], 1.0], &ImpostorSettings::default()) {
            Ok(imp) => upload_atlas(renderer, &imp),
            Err(e) => {
                tracing::debug!("impostor capture of mesh {}: {e}", slot.mesh);
                Entry::Failed
            }
        };
        self.model_impostors.entries.insert(slot, entry);
    }

    #[cfg(feature = "client")]
    fn finish_model_captures(&mut self, renderer: &mut Renderer) {
        let mut done = Vec::new();
        self.model_impostors.tickets.retain(|&(slot, ticket, centre, radius)| match renderer.take_impostor_capture(ticket) {
            Some(res) => {
                done.push((slot, res, centre, radius));
                false
            }
            None => true,
        });
        for (slot, res, centre, radius) in done {
            let settings = ImpostorSettings::default();
            let built = res.map_err(|e| format!("{e:#}")).and_then(|c| {
                // Identity key: these atlases are not cached on disk.
                Impostor::from_render(((slot.mesh as u128) << 64) | slot.material as u128, &settings, centre, radius, c.albedo, c.aux)
            });
            match built {
                Ok(imp) => {
                    let entry = upload_atlas(renderer, &imp);
                    self.model_impostors.entries.insert(slot, entry);
                }
                Err(e) => {
                    tracing::warn!("impostor capture of mesh {} failed ({e}); using the CPU capture", slot.mesh);
                    self.cpu_model_capture(renderer, slot);
                }
            }
        }
    }

    #[cfg(not(feature = "client"))]
    fn finish_model_captures(&mut self, _renderer: &mut Renderer) {}

    /// Free the atlases of slots no object uses any more (a model unloaded).
    pub(crate) fn release_model_impostors(&mut self, renderer: &mut Renderer) {
        let live: HashSet<Slot> = self.objects.iter().flat_map(|o| o.render_slots().map(|r| Slot { mesh: r.mesh, material: r.material })).collect();
        let stale: Vec<Slot> = self.model_impostors.entries.keys().filter(|s| !live.contains(s)).copied().collect();
        for s in stale {
            if let Some(Entry::Ready(h)) = self.model_impostors.entries.remove(&s) {
                renderer.free_impostor(h);
            }
        }
    }
}

fn upload_atlas(renderer: &mut Renderer, imp: &Impostor) -> Entry {
    match renderer.upload_impostor(&ImpostorUpload { grid: imp.grid, tile: imp.tile, centre: imp.centre, radius: imp.radius, albedo: &imp.albedo, aux: &imp.aux }) {
        Ok(h) => Entry::Ready(h),
        Err(e) => {
            tracing::warn!("scenery impostor upload: {e:#}");
            Entry::Failed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A selected object (any highlight, the secondary 0.5 too) must stay a real mesh: the
    /// selection outline is built from its draws, and a billboard has none to outline.
    #[test]
    fn selected_objects_are_never_swapped_for_billboards() {
        let mut scene = LoadedScene::empty();
        scene.model_impostors.enabled = true;
        // The guard returns before the (here nonexistent) object is looked at.
        for h in [1.0, 0.5] {
            assert!(!scene.model_impostor_swap(0, &Mat4::IDENTITY, h, false, false));
        }
    }
}
