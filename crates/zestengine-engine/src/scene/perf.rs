//! Performance rank of what is in the scene (GEO-2): measures an object
//! hierarchy against the budgets in `zestengine_assets::perf_rank`, and applies
//! the player's avatar safety policy (minimum rank, shown-avatar limit,
//! culling distance) to other players' avatars.
//!
//! An avatar the policy hides is not simply removed: it is drawn as an
//! octahedral impostor once one is captured (`avatar_impostor.rs`, the fallback
//! chain is documented there), else as a flat capsule proxy sized to its bounds
//! and tinted by its rank, so a crowd of too-expensive avatars still reads as people.
//!
//! Measuring walks the hierarchy, so the policy runs a few times a second, not
//! every frame; hiding is a parent-chain lookup that costs nothing while no
//! avatar is hidden.

use std::collections::{HashMap, HashSet};

use zestengine_assets::perf_rank::{self, AvatarDisplay, AvatarPolicy, FallbackReason, PerfRank, PerfReport, PerfStats, PerfTier};
use zestengine_core::ContentKind;

use super::*;

/// Frames between policy evaluations.
const POLICY_INTERVAL: u32 = 30;

/// A remote avatar the policy looked at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvatarStatus {
    pub object: ObjectId,
    pub rank: PerfRank,
    pub distance: f32,
    /// `Some` while it is replaced by the fallback.
    pub hidden: Option<FallbackReason>,
}

/// Rank colour of the fallback proxy (linear-ish RGB, green to red).
pub fn rank_tint(rank: PerfRank) -> [f32; 3] {
    match rank {
        PerfRank::Excellent => [0.25, 0.75, 0.35],
        PerfRank::Good => [0.55, 0.75, 0.25],
        PerfRank::Medium => [0.85, 0.75, 0.2],
        PerfRank::Poor => [0.9, 0.5, 0.15],
        PerfRank::VeryPoor => [0.85, 0.2, 0.2],
    }
}

/// Dimensions of the primitive capsule mesh the proxy scales (radius 0.25,
/// 1.0 of cylinder, so 0.5 wide and 1.5 tall).
const CAPSULE_SIZE: Vec3 = Vec3::new(0.5, 1.5, 0.5);

/// World transform that fits the capsule mesh around a box given in the avatar
/// root's local space (`center`, half-size `half`), so it follows the avatar.
pub fn proxy_transform(root_world: &Mat4, center: Vec3, half: Vec3) -> Mat4 {
    let width = half.x.max(half.z).max(0.05) * 2.0;
    let scale = Vec3::new(width / CAPSULE_SIZE.x, half.y.max(0.05) * 2.0 / CAPSULE_SIZE.y, width / CAPSULE_SIZE.z);
    *root_world * Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, center)
}

/// Box (center, half-size) enclosing the given world-space points, expressed
/// in the frame of `root_world`. `None` for no points.
pub fn local_bounds(root_world: &Mat4, points: impl IntoIterator<Item = Vec3>) -> Option<(Vec3, Vec3)> {
    let inv = root_world.inverse();
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in points {
        let l = inv.transform_point3(p);
        lo = lo.min(l);
        hi = hi.max(l);
    }
    (lo.x <= hi.x).then(|| ((lo + hi) * 0.5, (hi - lo) * 0.5))
}

/// GPU resources of the fallback proxy (made on first use).
struct AvatarProxy {
    mesh: usize,
    /// One flat-tinted material per rank, indexed by `PerfRank as usize`.
    materials: [usize; 5],
}

pub(crate) struct PerfRuntime {
    pub tier: PerfTier,
    pub policy: AvatarPolicy,
    frame: u32,
    /// Remote avatars at the last evaluation.
    pub avatars: Vec<AvatarStatus>,
    hidden: HashSet<ObjectId>,
    /// Bounds (avatar-root space) and rank of each hidden avatar, for the proxy.
    pub(super) proxies: Vec<(ObjectId, Vec3, Vec3, PerfRank)>,
    proxy: Option<AvatarProxy>,
    /// Project root, to find the files `.material` slots point at.
    project_root: Option<PathBuf>,
    /// Estimated GPU bytes per texture file (header read once).
    file_bytes: std::sync::Mutex<HashMap<PathBuf, u64>>,
}

impl Default for PerfRuntime {
    fn default() -> Self {
        Self {
            tier: PerfTier::Desktop,
            policy: AvatarPolicy::default(),
            frame: 0,
            avatars: Vec::new(),
            hidden: HashSet::new(),
            proxies: Vec::new(),
            proxy: None,
            project_root: None,
            file_bytes: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

/// Material texture slots (0..15) that hold a texture, in slot order.
pub fn slot_paths(textures: &zestengine_core::MaterialTexturePaths) -> Vec<(usize, PathBuf)> {
    let mut t = textures.clone();
    (0..16).filter_map(|i| t.slot_mut(i).and_then(|s| s.clone()).map(|p| (i, p))).collect()
}

impl LoadedScene {
    /// Cost of the hierarchy rooted at `root` (the whole scene when `None`).
    /// Texture memory counts the textures embedded in models plus the files
    /// the materials' slots point at (compressed estimate; each distinct
    /// texture once). The slot files need the project root
    /// ([`Self::set_perf_root`]); without it only embedded textures count.
    pub fn perf_stats(&self, root: Option<usize>) -> PerfStats {
        let mut stats = PerfStats::default();
        let mut materials: HashSet<usize> = HashSet::new();
        let mut skeletons: HashSet<usize> = HashSet::new();
        let mut skinned: HashSet<usize> = HashSet::new();
        let mut textures: HashSet<*const zestengine_render::TextureData> = HashSet::new();
        let mut files: HashSet<PathBuf> = HashSet::new();
        let active = self.active_flags();
        for (j, o) in self.objects.iter().enumerate() {
            if !active[j] || root.is_some_and(|r| !self.is_within(j, r)) {
                continue;
            }
            for r in o.render_slots() {
                if let Some(g) = self.mesh_geometry.get(r.mesh) {
                    stats.triangles += g.1.len() as u64 / 3;
                }
                if let Some(sm) = self.skinned_meshes.iter().find(|sm| sm.mesh_index == r.mesh)
                    && skinned.insert(r.mesh)
                {
                    stats.blendshapes += sm.morphs.len() as u32;
                    skeletons.insert(sm.skeleton);
                }
                if materials.insert(r.material)
                    && let Some(m) = self.materials.get(r.material)
                {
                    for t in m.base_tex_data.iter().flatten() {
                        if textures.insert(t as *const _) {
                            let per_texel = if t.hdr { 8 } else { 1 };
                            stats.texture_bytes += t.width as u64 * t.height as u64 * per_texel * 4 / 3;
                        }
                    }
                    stats.texture_bytes += self.material_file_texture_bytes(&m.model.textures, &mut files);
                }
            }
        }
        stats.materials = materials.len() as u32;
        stats.bones = skeletons.iter().filter_map(|&s| self.skeletons.get(s)).map(|s| s.joints.len() as u32).sum();
        stats
    }

    /// Project root used to resolve texture files named by material slots.
    pub fn set_perf_root(&mut self, root: &Path) {
        if self.perf.project_root.as_deref() != Some(root) {
            self.perf.project_root = Some(root.to_path_buf());
            self.perf.file_bytes.lock().unwrap().clear();
        }
    }

    /// Estimated bytes of the not-yet-counted texture files in `textures`.
    /// Embedded slots (already counted from `base_tex_data`) have no path.
    fn material_file_texture_bytes(&self, textures: &zestengine_core::MaterialTexturePaths, seen: &mut HashSet<PathBuf>) -> u64 {
        let Some(root) = &self.perf.project_root else { return 0 };
        let mut cache = self.perf.file_bytes.lock().unwrap();
        let mut total = 0;
        for (_, rel) in slot_paths(textures) {
            // A render-texture camera feed is not a file.
            if rel.to_string_lossy().starts_with("camera:") {
                continue;
            }
            let abs = zestengine_core::asset_ref::resolve_rel_path(root, &rel);
            if !seen.insert(abs.clone()) {
                continue;
            }
            total += *cache
                .entry(abs.clone())
                .or_insert_with(|| zestengine_assets::package::estimate_texture_bytes(&abs).map_or(0, |(_, _, b)| b));
        }
        total
    }

    /// Rank of the hierarchy rooted at `root` for `kind` on `tier`.
    pub fn perf_report(&self, root: Option<usize>, kind: ContentKind, tier: PerfTier) -> PerfReport {
        perf_rank::rank(&self.perf_stats(root), kind, tier)
    }

    /// Device tier the budgets and the default policy use.
    pub fn set_perf_tier(&mut self, tier: PerfTier) {
        if self.perf.tier != tier {
            self.perf.tier = tier;
            self.perf.policy = AvatarPolicy::for_tier(tier);
            self.perf.frame = POLICY_INTERVAL;
        }
    }

    pub fn perf_tier(&self) -> PerfTier {
        self.perf.tier
    }

    pub fn avatar_policy(&self) -> AvatarPolicy {
        self.perf.policy
    }

    pub fn set_avatar_policy(&mut self, policy: AvatarPolicy) {
        if self.perf.policy != policy {
            self.perf.policy = policy;
            self.perf.frame = POLICY_INTERVAL;
        }
    }

    /// Other players' avatars as the policy last saw them.
    pub fn avatar_statuses(&self) -> &[AvatarStatus] {
        &self.perf.avatars
    }

    /// Re-evaluate the avatar policy a couple of times a second.
    pub fn update_avatar_policy(&mut self) {
        self.perf.frame += 1;
        if self.perf.frame < POLICY_INTERVAL {
            return;
        }
        self.perf.frame = 0;
        let remote: Vec<usize> = self
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::NetworkAvatar>().is_some_and(|a| a.remote)))
            .map(|(i, _)| i)
            .collect();
        if remote.is_empty() && self.perf.avatars.is_empty() {
            return;
        }
        let tier = self.perf.tier;
        let inputs: Vec<(PerfRank, f32)> = remote
            .iter()
            .map(|&i| {
                let rank = self.perf_report(Some(i), ContentKind::Avatar, tier).overall;
                let distance = self.lod_camera.map_or(0.0, |c| self.frame_world(i).w_axis.truncate().distance(c));
                (rank, distance)
            })
            .collect();
        let shown = perf_rank::decide_avatars(&inputs, &self.perf.policy);
        let before = std::mem::take(&mut self.perf.hidden);
        self.perf.avatars = remote
            .iter()
            .zip(inputs.iter().zip(&shown))
            .map(|(&i, (&(rank, distance), display))| {
                let object = self.objects[i].id;
                let hidden = match display {
                    AvatarDisplay::Full => None,
                    AvatarDisplay::Fallback(why) => Some(*why),
                };
                if hidden.is_some() {
                    self.perf.hidden.insert(object);
                }
                AvatarStatus { object, rank, distance, hidden }
            })
            .collect();
        // Bounds of each hidden avatar, for its fallback proxy.
        let mut proxies = Vec::new();
        for (&i, (&(rank, _), display)) in remote.iter().zip(inputs.iter().zip(&shown)) {
            if matches!(display, AvatarDisplay::Full) {
                continue;
            }
            let root = self.frame_world(i);
            let mut points = Vec::new();
            for j in 0..self.objects.len() {
                if !self.is_within(j, i) {
                    continue;
                }
                let world = self.frame_world(j);
                for r in self.objects[j].render_slots() {
                    let Some(&(mn, mx)) = self.mesh_bounds.get(r.mesh) else { continue };
                    for c in 0..8 {
                        let corner = Vec3::new(if c & 1 == 0 { mn.x } else { mx.x }, if c & 2 == 0 { mn.y } else { mx.y }, if c & 4 == 0 { mn.z } else { mx.z });
                        points.push(world.transform_point3(corner));
                    }
                }
            }
            if let Some((center, half)) = local_bounds(&root, points) {
                proxies.push((self.objects[i].id, center, half, rank));
            }
        }
        self.perf.proxies = proxies;
        for s in &self.perf.avatars {
            if let Some(why) = s.hidden
                && !before.contains(&s.object)
            {
                tracing::info!("avatar {:?} hidden: {}", s.object, why.label());
            }
        }
    }

    /// Create the fallback proxy's mesh and rank materials once an avatar is
    /// hidden. Cheap otherwise.
    pub fn ensure_avatar_proxy(&mut self, renderer: &mut Renderer) {
        // Impostors first: the capsule below is the stand-in until they are ready.
        self.update_avatar_impostors(renderer);
        if self.perf.proxy.is_some() || self.perf.proxies.is_empty() {
            return;
        }
        let built = (|| -> Result<AvatarProxy> {
            let mesh = self.ensure_primitive_mesh(Some(&mut *renderer), zestengine_assets::PrimitiveShape::Capsule)?;
            let mut materials = [0usize; 5];
            for rank in PerfRank::ALL {
                let t = rank_tint(rank);
                let params = MaterialParams {
                    base_color: [t[0], t[1], t[2], 1.0],
                    emission_color: t,
                    emission_intensity: 0.6,
                    roughness: 1.0,
                    metallic: 0.0,
                    ..Default::default()
                };
                let name = format!("Avatar proxy {}", rank.label());
                let desc = zestengine_render::MaterialDesc { name: name.clone(), params, features: MaterialFeatures::default(), ..Default::default() };
                let handle = renderer.create_material(&desc)?;
                let model = model_from_material(&name, &params, &MaterialFeatures::default(), false);
                self.materials.push(MaterialEntry {
                    default: model.clone(),
                    model,
                    handle,
                    emission_map_mean: [1.0; 3],
                    bake_albedo: t,
                    bake_metallic: 0.0,
                    file: None,
                    embedded_textures: false,
                    base_textures: TexSlots::default(),
                    bound_textures: TexSlots::default(),
                    base_tex_data: Default::default(),
                });
                materials[rank as usize] = self.materials.len() - 1;
            }
            Ok(AvatarProxy { mesh, materials })
        })();
        match built {
            Ok(p) => self.perf.proxy = Some(p),
            Err(e) => tracing::warn!("avatar fallback proxy: {e:#}"),
        }
    }

    /// Append the fallback proxy draws of the hidden avatars (the drawable
    /// stand-in for objects skipped by `under_hidden_avatar`).
    pub(super) fn push_avatar_proxy_draws(&mut self) {
        let Some(proxy) = self.perf.proxy.as_ref() else { return };
        let (mesh, materials) = (proxy.mesh, proxy.materials);
        for k in 0..self.perf.proxies.len() {
            let (id, center, half, rank) = self.perf.proxies[k];
            if self.avatar_impostors.is_drawn(id) {
                continue;
            }
            let Some(i) = self.index_of(id).filter(|&i| self.is_active(i)) else { continue };
            let transform = proxy_transform(&self.frame_world(i), center, half);
            let (mn, mx) = self.mesh_bounds[mesh];
            self.draws.push(DrawCmd {
                gi_source: None,
                mesh: self.mesh_handles[mesh],
                material: self.materials[materials[rank as usize]].handle,
                transform,
                highlight: 0.0,
                mesh_center: (mn + mx) * 0.5,
                bound_radius: ((mx - mn) * 0.5).length(),
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index: -1,
                occluded: false,
                shadow_casting: zestengine_render::ShadowCasting::Off,
                instance_color: None,
                instance_param: None,
            });
        }
    }

    /// True when object `i` is part of an avatar the policy replaced.
    pub(crate) fn under_hidden_avatar(&self, i: usize) -> bool {
        if self.perf.hidden.is_empty() {
            return false;
        }
        let mut at = Some(i);
        for _ in 0..=self.objects.len() {
            let Some(k) = at else { return false };
            if self.perf.hidden.contains(&self.objects[k].id) {
                return true;
            }
            at = self.objects[k].parent;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(name: &str, parent: Option<usize>) -> SceneObject {
        SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: name.into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: Vec::new(),
        }
    }

    #[test]
    fn proxy_fits_the_avatar_bounds_and_follows_the_root() {
        let root = Mat4::from_translation(Vec3::new(10.0, 0.0, 5.0));
        // A 0.6 x 1.8 x 0.4 m body standing on the root, in world space.
        let pts = [Vec3::new(9.7, 0.0, 4.8), Vec3::new(10.3, 1.8, 5.2)];
        let (center, half) = local_bounds(&root, pts).unwrap();
        assert!((center - Vec3::new(0.0, 0.9, 0.0)).length() < 1e-5 && (half - Vec3::new(0.3, 0.9, 0.2)).length() < 1e-5);
        let m = proxy_transform(&root, center, half);
        // The capsule's bottom (y = -0.75) and top (+0.75) land on the body's feet and head.
        assert!((m.transform_point3(Vec3::new(0.0, -0.75, 0.0)) - Vec3::new(10.0, 0.0, 5.0)).length() < 1e-4);
        assert!((m.transform_point3(Vec3::new(0.0, 0.75, 0.0)) - Vec3::new(10.0, 1.8, 5.0)).length() < 1e-4);
        // Width follows the wider of x and z.
        assert!((m.transform_vector3(Vec3::new(0.25, 0.0, 0.0)).x - 0.3).abs() < 1e-5);
        // The avatar walks away: the proxy goes with it.
        let moved = proxy_transform(&Mat4::from_translation(Vec3::new(20.0, 0.0, 5.0)), center, half);
        assert!((moved.w_axis.x - 20.0).abs() < 1e-5);
        assert!(local_bounds(&root, []).is_none());
    }

    #[test]
    fn rank_tints_run_from_green_to_red() {
        assert!(rank_tint(PerfRank::Excellent)[1] > rank_tint(PerfRank::Excellent)[0]);
        assert!(rank_tint(PerfRank::VeryPoor)[0] > rank_tint(PerfRank::VeryPoor)[1]);
    }

    #[test]
    fn hiding_follows_the_parent_chain_only_while_something_is_hidden() {
        let mut s = LoadedScene::empty();
        s.objects.push(obj("player", None));
        s.objects.push(obj("body", Some(0)));
        s.objects.push(obj("other", None));
        assert!(!s.under_hidden_avatar(1));
        s.perf.hidden.insert(s.objects[0].id);
        assert!(s.under_hidden_avatar(0) && s.under_hidden_avatar(1));
        assert!(!s.under_hidden_avatar(2));
    }

    fn test_material() -> MaterialEntry {
        let model = model_from_material("m", &MaterialParams::default(), &MaterialFeatures::default(), false);
        MaterialEntry {
            default: model.clone(),
            model,
            handle: zestengine_render::MaterialHandle::NONE,
            emission_map_mean: [1.0; 3],
            bake_albedo: [1.0; 3],
            bake_metallic: 0.0,
            file: None,
            embedded_textures: false,
            base_textures: TexSlots::default(),
            bound_textures: TexSlots::default(),
            base_tex_data: Default::default(),
        }
    }

        #[test]
    fn material_slot_files_count_toward_texture_memory_once() {
        let dir = std::env::temp_dir().join(format!("zest_perf_tex_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::new(64, 64).save(dir.join("a.png")).unwrap();
        let mut s = LoadedScene::empty();
        s.set_perf_root(&dir);
        s.mesh_geometry.push((Vec::new(), vec![0, 1, 2]));
        for name in ["a", "b"] {
            s.objects.push(obj(name, None));
        }
        let entry = |s: &mut LoadedScene| {
            let mut e = test_material();
            e.model.textures.albedo = Some("a.png".into());
            e.model.textures.emission = Some("a.png".into());
            s.materials.push(e);
        };
        entry(&mut s);
        entry(&mut s);
        s.objects[0].render = Some(RenderInfo { mesh: 0, material: 0 });
        s.objects[1].render = Some(RenderInfo { mesh: 0, material: 1 });
        let expected = zestengine_assets::package::estimate_texture_bytes(&dir.join("a.png")).unwrap().2;
        assert!(expected > 0);
        // The same file in two slots of two materials is one texture.
        assert_eq!(s.perf_stats(None).texture_bytes, expected);
        // A missing root counts nothing from files.
        let mut none = LoadedScene::empty();
        none.mesh_geometry.push((Vec::new(), vec![0, 1, 2]));
        none.objects.push(obj("a", None));
        none.objects[0].render = Some(RenderInfo { mesh: 0, material: 0 });
        none.materials.push(test_material());
        none.materials[0].model.textures.albedo = Some("a.png".into());
        assert_eq!(none.perf_stats(None).texture_bytes, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_count_only_the_requested_hierarchy() {
        let mut s = LoadedScene::empty();
        s.mesh_geometry.push((Vec::new(), vec![0, 1, 2, 0, 2, 3]));
        s.mesh_geometry.push((Vec::new(), vec![0, 1, 2]));
        s.objects.push(obj("a", None));
        s.objects.push(obj("b", None));
        s.objects[0].render = Some(RenderInfo { mesh: 0, material: 0 });
        s.objects[1].render = Some(RenderInfo { mesh: 1, material: 1 });
        assert_eq!(s.perf_stats(Some(0)).triangles, 2);
        assert_eq!(s.perf_stats(Some(1)).triangles, 1);
        let all = s.perf_stats(None);
        assert_eq!((all.triangles, all.materials), (3, 2));
        assert_eq!(s.perf_report(Some(0), ContentKind::Avatar, PerfTier::Desktop).overall, PerfRank::Excellent);
    }
}
