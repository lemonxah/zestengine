//! Sprite Shapes in the scene (2D-8): gathers `SpriteShapeRenderer`s into the
//! sprite pass and bakes their collider outlines.
//!
//! # Per frame (only when a scene has a shape)
//!
//! 1. The sprite scan lists the objects with a shape (`ShapeCache::objs`); a
//!    scene without any pays the same type-id compare it already pays for the
//!    other 2D components and nothing else.
//! 2. Each shape's *key* is `content_hash` (spline + settings) mixed with the
//!    profile's content hash and the sprite epoch (bumped when sprites are
//!    re-resolved after a re-slice). The key is compared against the one the
//!    mesh was built from: **equal key = no work** (the last ranges keep
//!    drawing). A changed key regenerates the geometry on the CPU
//!    ([`zestengine_core::sprite_shape::generate_shape`]) and writes each group
//!    into a rewritable mesh of the sprite pool (`SpriteMeshPool::write_shape_mesh`,
//!    three slots written round-robin, so frames in flight keep their copy).
//!    A group that outgrows its mesh gets a bigger one (the old is released and
//!    reused by the next request that fits).
//! 3. `fill_sprite_input` (in `sprites2d.rs`) turns every group into one
//!    `SpriteSource` in `DrawMode::Simple` with the shape's world transform,
//!    layer, order, tint and mask mode. The fill is group 0 and the edge groups
//!    take the next group ordinals so edges draw over the fill of the same layer
//!    and order (inside a Sorting Group the sources keep their emission order).
//!
//! Sprites are drawn from their own textures (the same "plain" resolve the
//! skinned sprites use), not atlas pages, because the generated UVs are in the
//! sprite's own texture. The fill texture is a plain project texture sampled
//! with its import wrap mode: it needs Repeat for the tiling to work.
//!
//! # Collider
//!
//! [`LoadedScene::bake_shape_colliders`] writes the outline into the Polygon
//! Collider 2D (closed) or Edge Collider 2D (open) on the same object whenever
//! the shape's collider key changed or the collider component appeared. It runs
//! from the client gather every frame for the objects with shapes; the physics
//! world is rebuilt by Play when it reports `take_shape_collider_dirty`. A
//! dedicated server does not run the gather, so it relies on the outline saved
//! in the scene file.

use std::collections::HashMap;

use glam::{Vec2, Vec3};
use zestengine_core::sprite_shape::{ColliderOutline, collider_outline};
use zestengine_core::{Component, EdgeCollider2D, ObjectId, PolygonCollider2D, SpriteShapeRenderer, component_enabled, component_inner};

use super::LoadedScene;

/// Frames between profile file timestamp checks and state sweeps.
pub const STAMP_CHECK: u32 = 60;

pub fn is_shape_component(c: &dyn Component) -> bool {
    component_inner(c).as_any().is::<SpriteShapeRenderer>()
}

/// Remembers the key each shape was last built from.
#[derive(Default)]
pub struct DirtyTracker {
    keys: HashMap<ObjectId, u64>,
}

impl DirtyTracker {
    /// Whether `key` differs from the remembered one (and remember it).
    pub fn changed(&mut self, id: ObjectId, key: u64) -> bool {
        match self.keys.insert(id, key) {
            Some(old) => old != key,
            None => true,
        }
    }
    pub fn forget(&mut self, id: ObjectId) {
        self.keys.remove(&id);
    }
    #[allow(dead_code)] // clear/len round out the key set API
    pub fn clear(&mut self) {
        self.keys.clear();
    }
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

/// The key a shape's mesh is rebuilt on.
pub fn build_key(shape: &SpriteShapeRenderer, profile_hash: u64, epoch: u32) -> u64 {
    shape.content_hash() ^ profile_hash.rotate_left(21) ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn find<T: 'static>(comps: &[Box<dyn Component>]) -> Option<&T> {
    comps.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<T>())
}

fn enabled_shape(comps: &[Box<dyn Component>]) -> Option<&SpriteShapeRenderer> {
    comps.iter().find_map(|c| if component_enabled(c.as_ref()) { component_inner(c.as_ref()).as_any().downcast_ref::<SpriteShapeRenderer>() } else { None })
}

impl LoadedScene {
    /// Write the collider outlines of the shapes at `objects` into their sibling
    /// Polygon / Edge Collider 2D components. Returns how many colliders were
    /// written; the scene remembers that something changed
    /// ([`Self::take_shape_collider_dirty`]).
    pub fn bake_shape_colliders(&mut self, objects: &[usize]) -> usize {
        let mut written = 0;
        for &oi in objects {
            let Some(o) = self.objects.get(oi) else { continue };
            let Some(shape) = enabled_shape(&o.components) else { continue };
            if !shape.auto_update_collider {
                continue;
            }
            let has_poly = find::<PolygonCollider2D>(&o.components).is_some();
            let has_edge = find::<EdgeCollider2D>(&o.components).is_some();
            if !(has_poly && shape.closed || has_edge && !shape.closed) {
                self.shape_baked.remove(&o.id);
                continue;
            }
            let key = shape.collider_hash() ^ ((has_poly as u64) << 1 | has_edge as u64);
            let id = o.id;
            if self.shape_baked.get(&id) == Some(&key) {
                continue;
            }
            let outline = collider_outline(shape);
            self.shape_baked.insert(id, key);
            if write_outline(&mut self.objects[oi].components, &outline) {
                written += 1;
                self.shape_collider_dirty = true;
            }
        }
        written
    }

    /// Whether a collider was written since the last call (Play rebuilds its 2D world).
    pub fn take_shape_collider_dirty(&mut self) -> bool {
        std::mem::take(&mut self.shape_collider_dirty)
    }

    /// Local bounds of the object's shape mesh (selection outline, focus).
    pub fn shape_outline(&self, index: usize) -> Option<[Vec3; 4]> {
        #[cfg(feature = "client")]
        {
            let (lo, hi) = self.sprite_cache.shape.bounds_of(self.objects.get(index)?.id)?;
            let w = self.world_transform(index);
            let c = |x: f32, y: f32| w.transform_point3(Vec3::new(x, y, 0.0));
            Some([c(lo[0], lo[1]), c(hi[0], lo[1]), c(hi[0], hi[1]), c(lo[0], hi[1])])
        }
        #[cfg(not(feature = "client"))]
        {
            let _ = index;
            None
        }
    }
}

/// Copy the outline into the matching collider; whether anything changed.
fn write_outline(comps: &mut [Box<dyn Component>], outline: &ColliderOutline) -> bool {
    let mut changed = false;
    for c in comps.iter_mut() {
        let any = c.as_any_mut();
        if outline.closed
            && let Some(p) = any.downcast_mut::<PolygonCollider2D>()
        {
            let mut want = vec![outline.points.clone()];
            want.extend(outline.holes.iter().cloned());
            if p.paths != want {
                p.paths = want;
                changed = true;
            }
        } else if !outline.closed
            && let Some(e) = any.downcast_mut::<EdgeCollider2D>()
            && (e.points != outline.points || e.closed) {
                e.points = outline.points.clone();
                e.closed = false;
                changed = true;
            }
    }
    changed
}

/// Distance from a local point to the shape's pick polygon: inside a filled
/// closed outline counts as 0.
pub fn pick_distance(poly: &[Vec2], closed_fill: bool, p: Vec2) -> f32 {
    use zestengine_core::sprite_shape::spline::distance_to_polyline;
    if closed_fill && inside(poly, p) {
        return 0.0;
    }
    distance_to_polyline(p, poly, closed_fill)
}

fn inside(poly: &[Vec2], p: Vec2) -> bool {
    let n = poly.len();
    let mut c = false;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            c = !c;
        }
        j = i;
    }
    c
}

// ---------------------------------------------------------------------------
// Client side: sprite pool meshes and draw outputs
// ---------------------------------------------------------------------------

#[cfg(feature = "client")]
pub use client::ShapeCache;

#[cfg(feature = "client")]
mod client {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::SystemTime;

    use glam::Vec2;
    use zestengine_core::sprite_shape::{FillTexture, GenInput, GroupKind, SpriteMetrics, generate_shape, load_spriteshape};
    use zestengine_core::{SpriteShapeProfile, SpriteShapeRenderer, component_enabled, component_inner};
    use zestengine_render::Renderer;
    use zestengine_render::sprite2d::{MeshRange, ShapeMesh};

    use super::{DirtyTracker, STAMP_CHECK, build_key};
    use crate::scene::LoadedScene;
    use crate::scene::sprites2d::SpriteCache;

    struct ProfileAsset {
        profile: Arc<SpriteShapeProfile>,
        hash: u64,
        path: PathBuf,
        stamp: Option<SystemTime>,
    }

    #[derive(Clone, Copy)]
    struct FillTex {
        texture: u32,
        size_px: [u32; 2],
    }

    struct GroupState {
        kind: GroupKind,
        texture: u32,
        mesh: ShapeMesh,
        range: MeshRange,
    }

    #[derive(Default)]
    struct ShapeState {
        groups: Vec<GroupState>,
        bounds: Option<([f32; 2], [f32; 2])>,
        /// Tessellated outline for picking.
        pick: Vec<Vec2>,
        closed_fill: bool,
        seen: u32,
        failed: bool,
    }

    /// One draw group of a shape this frame.
    #[derive(Clone, Copy, Debug)]
    pub struct ShapeOut {
        pub mesh: MeshRange,
        pub texture: u32,
        pub kind: GroupKind,
        /// Local centre of the shape bounds (depth sorting point).
        pub centre: [f32; 2],
    }

    #[derive(Default)]
    pub struct ShapeCache {
        profiles: HashMap<String, Option<ProfileAsset>>,
        fill: HashMap<String, Option<FillTex>>,
        states: HashMap<zestengine_core::ObjectId, ShapeState>,
        tracker: DirtyTracker,
        /// Objects with a shape, collected by the sprite scan.
        pub(in crate::scene) objs: Vec<usize>,
        out: HashMap<usize, Vec<ShapeOut>>,
        epoch: u32,
        frame: u32,
        /// Meshes regenerated so far (tests, stats).
        pub rebuilds: u64,
    }

    use std::collections::HashMap;

    impl ShapeCache {
        /// Forget everything derived from sprites and files (project switch, re-slice).
        pub fn invalidate(&mut self) {
            self.profiles.clear();
            self.fill.clear();
            self.epoch = self.epoch.wrapping_add(1);
        }

        pub(in crate::scene) fn active(&self) -> bool {
            !self.objs.is_empty()
        }

        pub(in crate::scene) fn clear_outputs(&mut self) {
            self.out.clear();
        }

        pub(in crate::scene) fn out_for(&self, obj: usize) -> &[ShapeOut] {
            self.out.get(&obj).map_or(&[], |v| v.as_slice())
        }

        pub fn bounds_of(&self, id: zestengine_core::ObjectId) -> Option<([f32; 2], [f32; 2])> {
            self.states.get(&id)?.bounds
        }

        pub(in crate::scene) fn pick_of(&self, id: zestengine_core::ObjectId) -> Option<(&[Vec2], bool)> {
            self.states.get(&id).filter(|s| !s.pick.is_empty()).map(|s| (s.pick.as_slice(), s.closed_fill))
        }

        fn profile(&mut self, root: &Path, path: &str) -> Option<(Arc<SpriteShapeProfile>, u64)> {
            if path.is_empty() {
                return None;
            }
            let entry = self.profiles.entry(path.to_owned()).or_insert_with(|| {
                // The slot stores `@<id>` (survives renames) or a plain path.
                let rel = zestengine_core::asset_ref::resolve(root, path)?;
                let full = root.join(rel);
                match load_spriteshape(&full) {
                    Ok(p) => Some(ProfileAsset { hash: p.content_hash(), profile: Arc::new(p), stamp: std::fs::metadata(&full).and_then(|m| m.modified()).ok(), path: full }),
                    Err(e) => {
                        tracing::warn!("sprite shape profile {path}: {e}");
                        None
                    }
                }
            });
            entry.as_ref().map(|a| (a.profile.clone(), a.hash))
        }
    }

    impl LoadedScene {
        /// Regenerate the shapes whose key changed and refresh this frame's draw
        /// groups. Called by the sprite gather after the sprite references resolved.
        pub(in crate::scene) fn update_shapes(&mut self, c: &mut SpriteCache, renderer: &mut Renderer, project_root: &Path) {
            let frame = {
                c.shape.frame = c.shape.frame.wrapping_add(1);
                c.shape.frame
            };
            c.shape.out.clear();
            if frame % STAMP_CHECK == 0 {
                let stale = c.shape.profiles.values().any(|p| p.as_ref().is_some_and(|p| std::fs::metadata(&p.path).and_then(|m| m.modified()).ok() != p.stamp));
                if stale {
                    c.shape.profiles.clear();
                }
            }
            let objs = std::mem::take(&mut c.shape.objs);
            for &oi in &objs {
                let Some(o) = self.objects.get(oi) else { continue };
                let Some(shape) = o.components.iter().find_map(|k| if component_enabled(k.as_ref()) { component_inner(k.as_ref()).as_any().downcast_ref::<SpriteShapeRenderer>() } else { None }) else {
                    continue;
                };
                if !o.enabled || !self.is_active(oi) {
                    continue;
                }
                let id = o.id;
                let loaded = c.shape.profile(project_root, &shape.profile);
                let (profile, phash) = loaded.unwrap_or_else(|| (Arc::new(SpriteShapeProfile::default()), 0));
                let key = build_key(shape, phash, c.shape.epoch);
                let known = c.shape.states.contains_key(&id);
                if c.shape.tracker.changed(id, key) || !known {
                    let shape = shape.clone();
                    self.rebuild_shape(c, renderer, project_root, id, &shape, &profile);
                }
                if let Some(st) = c.shape.states.get_mut(&id) {
                    st.seen = frame;
                    if !st.failed {
                        let centre = st.bounds.map_or([0.0; 2], |(lo, hi)| [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]);
                        let outs: Vec<ShapeOut> = st.groups.iter().map(|g| ShapeOut { mesh: g.range, texture: g.texture, kind: g.kind, centre }).collect();
                        c.shape.out.insert(oi, outs);
                    }
                }
            }
            self.bake_shape_colliders(&objs);
            c.shape.objs = objs;
            // Release the meshes of shapes that went away.
            if frame % STAMP_CHECK == 0 {
                let gone: Vec<_> = c.shape.states.iter().filter(|(_, s)| frame.wrapping_sub(s.seen) >= STAMP_CHECK * 2).map(|(id, _)| *id).collect();
                for id in gone {
                    if let Some(st) = c.shape.states.remove(&id) {
                        for g in st.groups {
                            renderer.sprite_mesh_pool().release_shape_mesh(g.mesh);
                        }
                    }
                    c.shape.tracker.forget(id);
                    self.shape_baked.remove(&id);
                }
            }
        }

        fn fill_texture(&mut self, c: &mut SpriteCache, renderer: &mut Renderer, root: &Path, rel: &str) -> Option<FillTex> {
            if rel.is_empty() {
                return None;
            }
            if let Some(f) = c.shape.fill.get(rel) {
                return *f;
            }
            let r = (|| {
                let resolved = zestengine_core::asset_ref::resolve(root, rel)?;
                let rel_path = Path::new(&resolved);
                let handle = match self.load_texture_cached(renderer, root, rel_path, true) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::warn!("sprite shape fill texture {rel}: {e:#}");
                        return None;
                    }
                };
                let texture = c.shape_texture_id(renderer, handle)?;
                let (w, h) = image::image_dimensions(root.join(&resolved)).ok()?;
                Some(FillTex { texture, size_px: [w, h] })
            })();
            c.shape.fill.insert(rel.to_owned(), r);
            r
        }

        fn rebuild_shape(&mut self, c: &mut SpriteCache, renderer: &mut Renderer, root: &Path, id: zestengine_core::ObjectId, shape: &SpriteShapeRenderer, profile: &SpriteShapeProfile) {
            // Sprites the profile can draw resolve from their own textures.
            for name in profile.all_sprites() {
                if !c.resolved_plain_has(name) {
                    let r = self.resolve_sprite(renderer, root, name, true, c);
                    if r.is_none() {
                        tracing::warn!("sprite shape sprite '{name}' could not be resolved");
                    }
                    c.insert_resolved_plain(name.to_owned(), r);
                }
            }
            let fill = self.fill_texture(c, renderer, root, &profile.fill_texture);
            let lookup = |name: &str| c.resolved_plain_get(name).map(|r| SpriteMetrics { texture: r.texture, uv_rect: r.uv_rect, size: r.size });
            let mesh = generate_shape(&GenInput { shape, profile, sprite: &lookup, fill_texture: fill.map(|f| FillTexture { texture: f.texture, size_px: f.size_px }) });
            c.shape.rebuilds += 1;
            if mesh.stats.truncated {
                tracing::warn!("sprite shape: more than 65535 vertices per texture group; part of the shape is dropped");
            }
            let st = c.shape.states.entry(id).or_default();
            st.failed = false;
            st.bounds = mesh.bounds;
            st.closed_fill = shape.closed && shape.fill;
            st.pick = zestengine_core::sprite_shape::tessellate(&shape.points, shape.closed, shape.tess_mode(), shape.max_points).pts;
            // Reuse the group meshes in order; grow (release + new) when one is too small.
            let pool = renderer.sprite_mesh_pool();
            let mut old: Vec<GroupState> = std::mem::take(&mut st.groups);
            let mut groups = Vec::with_capacity(mesh.groups.len());
            for (gi, g) in mesh.groups.iter().enumerate() {
                let (nv, ni) = (g.positions.len(), g.indices.len());
                let prev = (gi < old.len()).then(|| old[gi].mesh);
                let fits = prev.and_then(|m| pool.shape_mesh_capacity(m)).is_some_and(|(cv, ci)| cv >= nv && ci >= ni && cv <= nv * 4 + 64);
                let handle = if fits {
                    prev
                } else {
                    if let Some(m) = prev {
                        pool.release_shape_mesh(m);
                    }
                    // Headroom so a few edit steps do not reallocate.
                    pool.add_shape_mesh((nv + nv / 2 + 16).min(u16::MAX as usize), ni + ni / 2 + 48)
                };
                let Some(handle) = handle else {
                    tracing::warn!("sprite shape: the sprite mesh pool is full");
                    st.failed = true;
                    break;
                };
                match pool.write_shape_mesh(handle, &g.positions, &g.uvs, &g.indices) {
                    Some(range) => groups.push(GroupState { kind: g.kind, texture: g.texture, mesh: handle, range }),
                    None => {
                        st.failed = true;
                        break;
                    }
                }
            }
            // Meshes of groups that no longer exist.
            for (gi, g) in old.drain(..).enumerate() {
                if gi >= mesh.groups.len() {
                    pool.release_shape_mesh(g.mesh);
                }
            }
            st.groups = groups;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{ObjectSource, SceneObject};
    use glam::Quat;
    use zestengine_core::sprite_shape::ShapeTangentMode;
    use zestengine_core::{ShapePoint, TypedComponent};

    fn object(name: &str, comps: Vec<Box<dyn Component>>) -> SceneObject {
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
            parent: None,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: comps,
        }
    }

    fn square_shape(closed: bool) -> SpriteShapeRenderer {
        SpriteShapeRenderer {
            points: vec![ShapePoint::at(-2.0, 2.0), ShapePoint::at(2.0, 2.0), ShapePoint::at(2.0, -2.0), ShapePoint::at(-2.0, -2.0)],
            closed,
            collider_optimize: 0.0,
            ..Default::default()
        }
    }

    fn polygon(s: &LoadedScene, i: usize) -> Vec<Vec<[f32; 2]>> {
        find::<PolygonCollider2D>(&s.objects[i].components).unwrap().paths.clone()
    }

    #[test]
    fn dirty_tracker_reports_only_changes() {
        let mut t = DirtyTracker::default();
        let id = ObjectId::new();
        assert!(t.changed(id, 5), "first sight builds");
        assert!(!t.changed(id, 5));
        assert!(!t.changed(id, 5));
        assert!(t.changed(id, 6));
        t.forget(id);
        assert!(t.changed(id, 6));
        // The key moves with the spline and the profile, not with the tint.
        let mut a = square_shape(true);
        let k0 = build_key(&a, 7, 0);
        a.color = [0.2; 4];
        a.order_in_layer = 3;
        assert_eq!(build_key(&a, 7, 0), k0);
        a.points[0].position[0] += 0.5;
        assert_ne!(build_key(&a, 7, 0), k0);
        assert_ne!(build_key(&square_shape(true), 8, 0), k0, "profile edits rebuild");
        assert_ne!(build_key(&square_shape(true), 7, 1), k0, "re-resolved sprites rebuild");
    }

    #[test]
    fn colliders_follow_the_spline_and_only_write_on_change() {
        let mut s = LoadedScene::empty();
        s.objects.push(object("Ground", vec![Box::new(square_shape(true)), Box::new(PolygonCollider2D::default())]));
        s.objects.push(object("Rope", vec![Box::new(square_shape(false)), Box::new(EdgeCollider2D::default())]));
        s.objects.push(object("NoCollider", vec![Box::new(square_shape(true))]));
        s.objects.push(object("Plain", vec![]));
        assert_eq!(s.bake_shape_colliders(&[0, 1, 2, 3, 99]), 2);
        assert!(s.take_shape_collider_dirty() && !s.take_shape_collider_dirty());
        assert_eq!(polygon(&s, 0), vec![vec![[-2.0, 2.0], [2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]]]);
        let e = find::<EdgeCollider2D>(&s.objects[1].components).unwrap();
        assert_eq!((e.points.len(), e.closed), (4, false));
        // Quiet frames write nothing.
        assert_eq!(s.bake_shape_colliders(&[0, 1, 2]), 0);
        assert!(!s.take_shape_collider_dirty());
        // Moving a point rebakes that object only.
        let sh = s.objects[0].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<SpriteShapeRenderer>()).unwrap();
        sh.points[1].position = [3.0, 2.0];
        assert_eq!(s.bake_shape_colliders(&[0, 1]), 1);
        assert_eq!(polygon(&s, 0)[0][1], [3.0, 2.0]);
        // The detail setting matters for curves; a collider added later gets filled.
        let sh = s.objects[2].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<SpriteShapeRenderer>()).unwrap();
        sh.points[0].set_mode(ShapeTangentMode::Broken, [0.0, -1.0], 1.0);
        sh.collider_detail = 4;
        s.objects[2].components.push(Box::new(PolygonCollider2D::default()));
        assert_eq!(s.bake_shape_colliders(&[2]), 1);
        let n4 = polygon(&s, 2)[0].len();
        let sh = s.objects[2].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<SpriteShapeRenderer>()).unwrap();
        sh.collider_detail = 16;
        assert_eq!(s.bake_shape_colliders(&[2]), 1);
        assert!(polygon(&s, 2)[0].len() > n4);
        // Auto update off: hands off.
        let sh = s.objects[2].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<SpriteShapeRenderer>()).unwrap();
        sh.auto_update_collider = false;
        sh.collider_detail = 2;
        assert_eq!(s.bake_shape_colliders(&[2]), 0);
        let _ = SpriteShapeRenderer::NAME;
    }

    #[test]
    fn open_outline_does_not_touch_a_polygon_collider() {
        let mut s = LoadedScene::empty();
        s.objects.push(object("Strip", vec![Box::new(square_shape(false)), Box::new(PolygonCollider2D::default())]));
        let before = polygon(&s, 0);
        assert_eq!(s.bake_shape_colliders(&[0]), 0, "an open shape feeds an Edge collider only");
        assert_eq!(polygon(&s, 0), before);
    }

    #[test]
    fn picking_uses_fill_and_line_distance() {
        let ring = [Vec2::new(-2.0, -2.0), Vec2::new(2.0, -2.0), Vec2::new(2.0, 2.0), Vec2::new(-2.0, 2.0)];
        assert_eq!(pick_distance(&ring, true, Vec2::new(0.5, 0.5)), 0.0);
        assert!((pick_distance(&ring, true, Vec2::new(3.0, 0.0)) - 1.0).abs() < 1e-5);
        assert!((pick_distance(&ring, false, Vec2::new(0.0, 0.0)) - 2.0).abs() < 1e-5, "an unfilled ring is only its line");
        assert_eq!(pick_distance(&[], true, Vec2::ZERO), f32::INFINITY);
    }
}
