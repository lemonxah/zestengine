//! 2D sprites in the scene (2D-3 phase 2): turns `SpriteRenderer`, `SortingGroup`
//! and `SpriteMask` components into the render crate's `SpriteSource` /
//! `MaskSource` lists each frame, resolves sprite references to pooled meshes
//! and textures (cached), and answers viewport picking and bounds for sprites.
//!
//! Cost model: a scene without sprites pays one component scan every
//! [`IDLE_RESCAN`] frames (a type-id compare per component) and nothing else; a
//! scene with sprites walks its objects once per frame, resolves each distinct
//! sprite reference once (re-checked every [`STAMP_CHECK`] frames against its
//! texture's `.meta` timestamp, so re-slicing in the Sprite Editor shows up),
//! and fills pooled vectors.
//!
//! Client builds only: the server has no renderer (`render-lite` has no sprite
//! API) and never draws.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use glam::{Mat4, Vec2, Vec3};
use zestengine_assets::sprite::{SpriteRef, resolve_sprite_ref};
use zestengine_assets::sprite_atlas::{AtlasDb, AtlasHit, PageLayout};
use zestengine_assets::Platform;
use zestengine_core::{
    Component, MaskInteraction2D, SortingGroup, SpriteDrawMode, SpriteMask, SpriteMaskRange, SpriteRenderer, SpriteShapeRenderer, SpriteSortPoint, SpriteTileMode, component_enabled,
    component_inner,
};
use zestengine_render::Renderer;
use zestengine_render::sprite2d::{
    DrawMode, GroupAssign, GroupMember, GroupNode, GroupScratch, MaskMode, MaskRange, MaskSource, MeshRange, SortAxis, SpriteSource, TileMode, flatten_groups,
};

use super::LoadedScene;

/// Frames between component scans while the scene has no sprites.
const IDLE_RESCAN: u32 = 8;
/// Frames between `.meta` timestamp checks of resolved sprites.
const STAMP_CHECK: u32 = 60;
/// How far (local units) from a shape's spline a click still picks it.
const SHAPE_PICK_RADIUS: f32 = 0.25;

/// One resolved sprite: GPU ids plus the CPU geometry picking needs.
pub struct ResolvedSprite {
    pub mesh: MeshRange,
    pub texture: u32,
    pub uv_rect: [f32; 4],
    pub size: [f32; 2],
    pub border: [f32; 4],
    pub pivot: [f32; 2],
    /// Local centre of the mesh bounds (depth sorting point).
    pub centre: [f32; 2],
    pub bounds: ([f32; 2], [f32; 2]),
    /// Array layer (atlas page) inside `texture`; 0 for a plain texture.
    pub layer: u32,
    /// The sprite as one quad (tilemap tiles): local rect (min, size) relative
    /// to the pivot, its UV rect, and whether the content is stored rotated.
    pub quad_rect: [f32; 4],
    pub quad_uv: [f32; 4],
    pub rotated: bool,
    positions: Arc<Vec<[f32; 2]>>,
    indices: Arc<Vec<u16>>,
    meta: PathBuf,
    stamp: Option<SystemTime>,
}

/// Per-scene sprite state (pooled vectors and resolve caches).
#[derive(Default)]
pub struct SpriteCache {
    /// Sprites through their cooked atlas when they have one.
    resolved: HashMap<String, Option<ResolvedSprite>>,
    /// The same references from their own textures (Sliced / Tiled draws and
    /// masks need the sprite rect, which an atlas trims and may rotate).
    resolved_plain: HashMap<String, Option<ResolvedSprite>>,
    tex_ids: HashMap<usize, u32>,
    /// Atlas page array textures by the cooked atlas's address.
    atlas_tex: HashMap<usize, u32>,
    atlas_db: Option<Arc<AtlasDb>>,
    atlas_scanned: bool,
    /// `.material` overrides registered with the sprite pass: path -> (material id, tint); `None` =
    /// unreadable (the default material draws instead).
    materials: HashMap<String, Option<(u32, [f32; 4])>>,
    /// Material paths seen this frame that are not registered yet.
    mat_need: Vec<String>,
    /// Tilemap renderer state (chunk caches, tile resolve tables).
    pub(super) tiles: super::tilemaps2d::TileGatherCache,
    /// 2D rigs (IK, sprite swap, skinning); empty work when the scene has none.
    pub(super) rig: super::rig2d::RigCache,
    /// Sprite shapes (spline meshes); empty work when the scene has none.
    pub(super) shape: super::shape2d::ShapeCache,
    /// Sprite scene seen present at the last scan.
    present: bool,
    idle: u32,
    frame: u32,
    need: Vec<(String, bool)>,
    sources: Vec<SpriteSource>,
    masks: Vec<MaskSource>,
    group_of: Vec<u32>,
    group_objs: Vec<usize>,
    nodes: Vec<GroupNode>,
    members: Vec<GroupMember>,
    member_src: Vec<usize>,
    assigned: Vec<GroupAssign>,
    scratch: GroupScratch,
}

#[allow(dead_code)] // hooks for the Sprite Editor / project switch
impl SpriteCache {
    /// Forget every resolved sprite (project switch, texture reimport).
    pub fn invalidate(&mut self) {
        self.resolved.clear();
        self.resolved_plain.clear();
        self.materials.clear();
        self.atlas_db = None;
        self.atlas_scanned = false;
        self.tiles.invalidate();
        self.rig.invalidate();
        self.shape.invalidate();
    }

    /// Whether any sprite was resolved (picking has something to test).
    pub fn is_empty(&self) -> bool {
        self.resolved.is_empty()
    }
}

fn is_sprite_component(c: &dyn Component) -> bool {
    let a = component_inner(c).as_any();
    a.is::<SpriteRenderer>() || a.is::<SpriteMask>() || a.is::<SortingGroup>()
}

impl SpriteCache {
    /// Ask for the sprite material of `path` (an empty path is the default material).
    pub(super) fn need_material(&mut self, path: &str) {
        if !path.is_empty() && !self.materials.contains_key(path) && !self.mat_need.iter().any(|p| p == path) {
            self.mat_need.push(path.to_owned());
        }
    }

    /// Register the materials asked for this frame with the sprite pass.
    pub(super) fn register_materials(&mut self, renderer: &mut Renderer, project_root: &Path) {
        let mut need = std::mem::take(&mut self.mat_need);
        for path in need.drain(..) {
            let entry = match zestengine_assets::load_material_file(zestengine_core::asset_ref::resolve_path(project_root, &path)) {
                Ok(file) => {
                    if file.shader != "standard" {
                        tracing::warn!("sprite material '{path}': custom shaders do not run in the 2D pass; its blend, cutoff and tint apply");
                    }
                    let (desc, tint) = super::sprite_material::sprite_material_from(&file);
                    match renderer.register_sprite_material(desc) {
                        Ok(id) => Some((id, tint)),
                        Err(e) => {
                            tracing::warn!("sprite material '{path}': {e:#}");
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("sprite material '{path}': {e:#}");
                    None
                }
            };
            self.materials.insert(path, entry);
        }
        self.mat_need = need;
    }

    /// The registered material id and tint of `path`; the default material and a white tint when the
    /// path is empty or its material could not be loaded.
    pub(super) fn material_of(&self, path: &str) -> (u32, [f32; 4]) {
        if path.is_empty() {
            return (zestengine_render::DEFAULT_SPRITE_MATERIAL, [1.0; 4]);
        }
        self.materials.get(path).copied().flatten().unwrap_or((zestengine_render::DEFAULT_SPRITE_MATERIAL, [1.0; 4]))
    }

    pub(super) fn resolved_has(&self, s: &str) -> bool {
        self.resolved.contains_key(s)
    }
    pub(super) fn resolved_plain_has(&self, s: &str) -> bool {
        self.resolved_plain.contains_key(s)
    }
    /// A resolved own-texture sprite (`None` while unresolved or when it failed).
    pub(super) fn resolved_plain_get(&self, s: &str) -> Option<&ResolvedSprite> {
        self.resolved_plain.get(s).and_then(|r| r.as_ref())
    }
    /// Queue a sprite reference for resolving this frame.
    pub(super) fn need_push(&mut self, text: &str, plain: bool) {
        let known = if plain { self.resolved_plain.contains_key(text) } else { self.resolved.contains_key(text) };
        if !text.is_empty() && !known && !self.need.iter().any(|(t, p)| t == text && *p == plain) {
            self.need.push((text.to_owned(), plain));
        }
    }
    pub(super) fn insert_resolved_plain(&mut self, s: String, r: Option<ResolvedSprite>) {
        self.resolved_plain.insert(s, r);
    }
    /// The sprite-pass texture id of a loaded texture (registered once).
    pub(super) fn shape_texture_id(&mut self, renderer: &mut Renderer, handle: zestengine_render::TextureHandle) -> Option<u32> {
        if let Some(&t) = self.tex_ids.get(&handle.index()) {
            return Some(t);
        }
        match renderer.register_sprite_texture(handle, false) {
            Ok(t) => {
                self.tex_ids.insert(handle.index(), t);
                Some(t)
            }
            Err(e) => {
                tracing::warn!("sprite texture registration: {e:#}");
                None
            }
        }
    }
    pub(super) fn insert_resolved(&mut self, s: String, r: Option<ResolvedSprite>) {
        self.resolved.insert(s, r);
    }
    pub(super) fn resolved_len(&self) -> usize {
        self.resolved.len()
    }
    /// A resolved sprite as a tile quad.
    pub(super) fn tile_sprite(&self, s: &str) -> Option<super::tilemaps2d::TileSprite> {
        let r = self.resolved.get(s)?.as_ref()?;
        Some(super::tilemaps2d::TileSprite { texture: r.texture, layer: r.layer, rect: r.quad_rect, uv: r.quad_uv, rotated: r.rotated })
    }

    /// The project's atlases, scanned once (and again after an invalidate).
    fn atlases(&mut self, project_root: &Path) -> Option<Arc<AtlasDb>> {
        if !self.atlas_scanned {
            self.atlas_scanned = true;
            let db = AtlasDb::scan(project_root);
            self.atlas_db = (!db.atlases.is_empty()).then(|| Arc::new(db));
        }
        self.atlas_db.clone()
    }
}

fn draw_mode(m: SpriteDrawMode) -> DrawMode {
    match m {
        SpriteDrawMode::Simple => DrawMode::Simple,
        SpriteDrawMode::Sliced => DrawMode::Sliced,
        SpriteDrawMode::Tiled => DrawMode::Tiled,
    }
}

fn mask_mode(m: MaskInteraction2D) -> MaskMode {
    match m {
        MaskInteraction2D::None => MaskMode::None,
        MaskInteraction2D::VisibleInsideMask => MaskMode::VisibleInside,
        MaskInteraction2D::VisibleOutsideMask => MaskMode::VisibleOutside,
    }
}

/// Rotation about Z, scale and whether the 2x2 part mirrors, from a world matrix.
fn decompose_2d(w: &Mat4) -> (f32, [f32; 2], bool) {
    let ax = Vec2::new(w.x_axis.x, w.x_axis.y);
    let ay = Vec2::new(w.y_axis.x, w.y_axis.y);
    let det = ax.x * ay.y - ax.y * ay.x;
    let scale = [ax.length(), ay.length()];
    if det < 0.0 { ((-ay.x).atan2(ay.y), scale, true) } else { (ax.y.atan2(ax.x), scale, false) }
}

impl LoadedScene {
    /// Register an atlas's pages as one array texture (cached per atlas); `None`
    /// when the pages cannot form one (separate layout, mixed sizes) or the
    /// upload fails.
    fn atlas_texture(&self, renderer: &mut Renderer, c: &mut SpriteCache, hit: &AtlasHit) -> Option<u32> {
        let key = Arc::as_ptr(&hit.atlas) as usize;
        if let Some(&t) = c.atlas_tex.get(&key) {
            return Some(t);
        }
        let a = &hit.atlas;
        if a.table.layout != PageLayout::Array || a.pages.is_empty() {
            return None;
        }
        let pages: Vec<&zestengine_render::CompressedTexture> = a.pages.iter().collect();
        let same = pages.iter().all(|p| p.format == pages[0].format && p.width == pages[0].width && p.height == pages[0].height && p.mips.len() == pages[0].mips.len());
        if !same {
            tracing::warn!("atlas '{}': pages differ in size or format; its sprites draw from their own textures", a.table.name);
            return None;
        }
        let tex = match renderer.upload_compressed_texture_array(&pages).and_then(|h| renderer.register_sprite_texture(h, true)) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("atlas '{}' upload: {e:#}", a.table.name);
                return None;
            }
        };
        c.atlas_tex.insert(key, tex);
        Some(tex)
    }

    /// Resolve a sprite reference text: sprite table, geometry, texture (loaded
    /// and registered with the renderer) and pooled mesh. With `plain` false a
    /// sprite that belongs to a cooked atlas draws from the atlas page array
    /// (layer, trimmed placed rect, rotation); anything unresolved there keeps
    /// the original texture path.
    pub(super) fn resolve_sprite(&mut self, renderer: &mut Renderer, project_root: &Path, text: &str, plain: bool, c: &mut SpriteCache) -> Option<ResolvedSprite> {
        let r = SpriteRef::from_ref_string(text)?;
        if !plain
            && let Some(db) = c.atlases(project_root)
            && let Some(hit) = db.resolve(Platform::Desktop, &r)
            && let Some(texture) = self.atlas_texture(renderer, c, &hit)
        {
            let ag = hit.geometry();
            let q = hit.quad();
            let g = ag.geometry;
            if let Some(mesh) = renderer.sprite_mesh_pool().add(&g.positions, &g.uvs, &g.indices) {
                let (lo, hi) = g.bounds();
                let meta = PathBuf::from(format!("{}.meta", project_root.join(&r.path).display()));
                let stamp = std::fs::metadata(&meta).and_then(|m| m.modified()).ok();
                return Some(ResolvedSprite {
                    mesh,
                    texture,
                    uv_rect: g.uv_rect,
                    size: g.size_world,
                    border: g.border_world,
                    pivot: g.pivot_world,
                    centre: [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5],
                    bounds: (lo, hi),
                    layer: q.page,
                    quad_rect: q.rect,
                    quad_uv: q.uv_rect,
                    rotated: q.rotated,
                    positions: Arc::new(g.positions),
                    indices: Arc::new(g.indices),
                    meta,
                    stamp,
                });
            }
        }
        let (table, idx) = resolve_sprite_ref(project_root, &r)?;
        let geom = table.geometry(idx)?;
        let rel = table.texture_path.strip_prefix(project_root).unwrap_or(&table.texture_path).to_path_buf();
        let handle = match self.load_texture_cached(renderer, project_root, &rel, true) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("sprite texture {}: {e:#}", rel.display());
                return None;
            }
        };
        let texture = match c.tex_ids.get(&handle.index()) {
            Some(&t) => t,
            None => match renderer.register_sprite_texture(handle, false) {
                Ok(t) => {
                    c.tex_ids.insert(handle.index(), t);
                    t
                }
                Err(e) => {
                    tracing::warn!("sprite texture registration: {e:#}");
                    return None;
                }
            },
        };
        let mesh = renderer.sprite_mesh_pool().add(&geom.positions, &geom.uvs, &geom.indices)?;
        let (lo, hi) = geom.bounds();
        let meta = PathBuf::from(format!("{}.meta", table.texture_path.display()));
        let stamp = std::fs::metadata(&meta).and_then(|m| m.modified()).ok();
        Some(ResolvedSprite {
            mesh,
            texture,
            uv_rect: geom.uv_rect,
            size: geom.size_world,
            border: geom.border_world,
            pivot: geom.pivot_world,
            centre: [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5],
            bounds: (lo, hi),
            layer: 0,
            quad_rect: [-geom.pivot_world[0], -geom.pivot_world[1], geom.size_world[0], geom.size_world[1]],
            quad_uv: geom.uv_rect,
            rotated: false,
            positions: Arc::new(geom.positions),
            indices: Arc::new(geom.indices),
            meta,
            stamp,
        })
    }

    /// Fill the renderer's sprite input (and queue the tilemaps) for this
    /// frame from the scene's components. `view` / `proj` are the camera's
    /// matrices (depth sorting axis, tilemap chunk culling) and `time` the
    /// animation clock in seconds. Cheap and allocation-free when the scene has
    /// no sprites or tilemaps.
    pub fn gather_sprites(&mut self, renderer: &mut Renderer, project_root: &Path, view: Mat4, proj: Mat4, time: f32) {
        let mut cache = std::mem::take(&mut self.sprite_cache);
        self.gather_sprites_inner(&mut cache, renderer, project_root, view, proj, time);
        self.sprite_cache = cache;
    }

    fn gather_sprites_inner(&mut self, c: &mut SpriteCache, renderer: &mut Renderer, project_root: &Path, view: Mat4, proj: Mat4, time: f32) {
        c.frame = c.frame.wrapping_add(1);
        if !c.present {
            c.idle += 1;
            if !c.idle.is_multiple_of(IDLE_RESCAN) {
                return;
            }
        }
        // Collect the references still to resolve, and whether anything is here.
        c.need.clear();
        c.rig.objs.clear();
        c.shape.objs.clear();
        let mut any = false;
        for (oi, o) in self.objects.iter().enumerate() {
            for comp in &o.components {
                let inner = component_inner(comp.as_ref());
                if !is_sprite_component(inner) && !super::tilemaps2d::is_tile_component(inner) {
                    if super::shape2d::is_shape_component(inner) {
                        any = true;
                        if c.shape.objs.last() != Some(&oi) {
                            c.shape.objs.push(oi);
                        }
                    } else if super::rig2d::is_rig_component(inner) {
                        any = true;
                        if c.rig.objs.last() != Some(&oi) {
                            c.rig.objs.push(oi);
                        }
                    }
                    continue;
                }
                any = true;
                let (text, plain) = if let Some(s) = inner.as_any().downcast_ref::<SpriteRenderer>() {
                    c.need_material(&s.material);
                    // A skinned sprite draws from its own texture; the atlas entry stays the fallback.
                    let skinned = o.components.iter().any(|k| component_inner(k.as_ref()).as_any().is::<zestengine_core::SpriteSkin>());
                    if skinned {
                        c.need_push(&s.sprite, false);
                    }
                    (&s.sprite, s.draw_mode != SpriteDrawMode::Simple || skinned)
                } else if let Some(m) = inner.as_any().downcast_ref::<SpriteMask>() {
                    (&m.sprite, true)
                } else {
                    continue;
                };
                c.need_push(text, plain);
            }
        }
        c.present = any;
        if !any {
            c.idle = 0;
            return;
        }
        // Re-check timestamps now and then: a re-sliced sheet resolves again.
        if c.frame.is_multiple_of(STAMP_CHECK) {
            let stale = |r: &Option<ResolvedSprite>| match r {
                Some(r) => std::fs::metadata(&r.meta).and_then(|m| m.modified()).ok() != r.stamp,
                None => false,
            };
            if c.resolved.values().chain(c.resolved_plain.values()).any(stale) {
                c.resolved.clear();
                c.resolved_plain.clear();
                c.atlas_db = None;
                c.atlas_scanned = false;
                c.tiles.invalidate();
                c.shape.invalidate();
            }
        }
        // IK and sprite swaps first: swapped sprites join this frame's resolve list.
        if c.rig.active() {
            self.update_rigs_pre(c, project_root);
        } else {
            c.rig.clear_outputs();
        }
        let mut need = std::mem::take(&mut c.need);
        for (text, plain) in need.drain(..) {
            let r = self.resolve_sprite(renderer, project_root, &text, plain, c);
            if r.is_none() {
                tracing::warn!("sprite '{text}' could not be resolved");
            }
            if plain { c.resolved_plain.insert(text, r) } else { c.resolved.insert(text, r) };
        }
        c.need = need;
        c.register_materials(renderer, project_root);
        if c.rig.active() {
            self.update_rigs_skin(c, renderer, project_root);
        }
        if c.shape.active() {
            self.update_shapes(c, renderer, project_root);
        } else {
            c.shape.clear_outputs();
        }
        self.fill_sprite_input(c, renderer, view);
        self.gather_tilemaps(c, renderer, project_root, view, proj, time);
    }

    fn fill_sprite_input(&self, c: &mut SpriteCache, renderer: &mut Renderer, view: Mat4) {
        let n = self.objects.len();
        let layers = &self.layers.sorting;
        // Sorting groups: object index -> group node.
        c.group_of.clear();
        c.group_of.resize(n, u32::MAX);
        c.group_objs.clear();
        for (i, o) in self.objects.iter().enumerate() {
            if o.components.iter().any(|k| component_enabled(k.as_ref()) && component_inner(k.as_ref()).as_any().is::<SortingGroup>()) && self.is_active(i) {
                c.group_of[i] = c.group_objs.len() as u32;
                c.group_objs.push(i);
            }
        }
        let inv = view.inverse();
        let forward = -inv.z_axis.truncate().normalize_or_zero();
        let origin = inv.w_axis.truncate();
        let nearest_group = |mut i: usize, group_of: &[u32]| -> Option<u32> {
            for _ in 0..64 {
                if group_of[i] != u32::MAX {
                    return Some(group_of[i]);
                }
                i = self.objects[i].parent.filter(|&p| p < n)?;
            }
            None
        };
        c.nodes.clear();
        for &gi in &c.group_objs {
            let o = &self.objects[gi];
            let g = o.components.iter().find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<SortingGroup>()).expect("group object");
            let parent = o.parent.filter(|&p| p < n).and_then(|p| nearest_group(p, &c.group_of));
            let w = self.world_transform(gi);
            c.nodes.push(GroupNode {
                parent,
                layer: layers.value_of(g.sorting_layer),
                order: g.order_in_layer,
                sort_at_root: g.sort_at_root,
                depth: (w.w_axis.truncate() - origin).dot(forward),
            });
        }
        // Renderers.
        c.sources.clear();
        c.members.clear();
        c.member_src.clear();
        c.masks.clear();
        for (i, o) in self.objects.iter().enumerate() {
            if (self.visible_layers >> (o.layer as u32 & 31)) & 1 == 0 {
                continue;
            }
            let mut world: Option<Mat4> = None;
            for comp in &o.components {
                if !component_enabled(comp.as_ref()) {
                    continue;
                }
                let inner = component_inner(comp.as_ref()).as_any();
                if let Some(s) = inner.downcast_ref::<SpriteRenderer>() {
                    // 2D rigs: a swapped sprite and a CPU-skinned mesh override the scene's.
                    let skin = c.rig.skin_for(i);
                    let text: &str = c.rig.swap_for(i).unwrap_or(&s.sprite);
                    let map = if s.draw_mode == SpriteDrawMode::Simple && skin.is_none() { &c.resolved } else { &c.resolved_plain };
                    let Some(Some(r)) = map.get(text) else { continue };
                    let (mesh, centre) = skin.map_or((r.mesh, r.centre), |k| (k.mesh, k.centre));
                    if !self.is_active(i) || self.view_hidden(i) {
                        break;
                    }
                    let w = *world.get_or_insert_with(|| self.world_transform(i));
                    let (rotation, scale, mirrored) = decompose_2d(&w);
                    let pos = w.w_axis.truncate();
                    let sort_point = match s.sort_point {
                        SpriteSortPoint::Pivot => pos,
                        SpriteSortPoint::Center => w.transform_point3(Vec3::new(centre[0], centre[1], 0.0)),
                    };
                    let layer = layers.value_of(s.sorting_layer);
                    let (material, tint) = c.material_of(&s.material);
                    let src = SpriteSource {
                        position: pos.to_array(),
                        rotation,
                        scale,
                        sort_point: sort_point.to_array(),
                        mesh,
                        uv_rect: r.uv_rect,
                        sprite_size: r.size,
                        border: r.border,
                        pivot: r.pivot,
                        color: [s.color[0] * tint[0], s.color[1] * tint[1], s.color[2] * tint[2], s.color[3] * tint[3]],
                        draw: draw_mode(s.draw_mode),
                        size: s.size,
                        tile: match s.tile_mode {
                            SpriteTileMode::Continuous => TileMode::Continuous,
                            SpriteTileMode::Adaptive => TileMode::Adaptive,
                        },
                        adaptive_threshold: s.adaptive_threshold,
                        flip: [s.flip_x ^ mirrored, s.flip_y],
                        material,
                        texture: r.texture,
                        atlas_layer: r.layer,
                        layer,
                        order: s.order_in_layer,
                        group: 0,
                        mask: mask_mode(s.mask_interaction),
                        ..Default::default()
                    };
                    c.members.push(GroupMember { group: nearest_group(i, &c.group_of), layer, order: s.order_in_layer });
                    c.member_src.push(c.sources.len());
                    c.sources.push(src);
                } else if let Some(sh) = inner.downcast_ref::<SpriteShapeRenderer>() {
                    // Sprite shapes: one source per texture group (fill = group 0, edges after it).
                    let count = c.shape.out_for(i).len();
                    if count == 0 || !self.is_active(i) || self.view_hidden(i) {
                        continue;
                    }
                    let w = *world.get_or_insert_with(|| self.world_transform(i));
                    let (rotation, scale, mirrored) = decompose_2d(&w);
                    let pos = w.w_axis.truncate();
                    let layer = layers.value_of(sh.sorting_layer);
                    let mut edge_ordinal = 0u16;
                    for k in 0..count {
                        let out = c.shape.out_for(i)[k];
                        let group = if out.kind == zestengine_core::sprite_shape::GroupKind::Edge {
                            edge_ordinal += 1;
                            edge_ordinal
                        } else {
                            0
                        };
                        let sort_point = w.transform_point3(Vec3::new(out.centre[0], out.centre[1], 0.0));
                        let src = SpriteSource {
                            position: pos.to_array(),
                            rotation,
                            scale,
                            sort_point: sort_point.to_array(),
                            mesh: out.mesh,
                            uv_rect: [0.0, 0.0, 1.0, 1.0],
                            sprite_size: [1.0, 1.0],
                            color: sh.color,
                            draw: DrawMode::Simple,
                            size: [1.0, 1.0],
                            adaptive_threshold: 0.5,
                            flip: [mirrored, false],
                            material: zestengine_render::DEFAULT_SPRITE_MATERIAL,
                            texture: out.texture,
                            atlas_layer: 0,
                            layer,
                            order: sh.order_in_layer,
                            group,
                            mask: mask_mode(sh.mask_interaction),
                            ..Default::default()
                        };
                        c.members.push(GroupMember { group: nearest_group(i, &c.group_of), layer, order: sh.order_in_layer });
                        c.member_src.push(c.sources.len());
                        c.sources.push(src);
                    }
                } else if let Some(m) = inner.downcast_ref::<SpriteMask>() {
                    let Some(Some(r)) = c.resolved_plain.get(&m.sprite) else { continue };
                    if !self.is_active(i) || self.view_hidden(i) {
                        break;
                    }
                    let w = *world.get_or_insert_with(|| self.world_transform(i));
                    let (rotation, scale, _) = decompose_2d(&w);
                    let v = |l| layers.value_of(l);
                    c.masks.push(MaskSource {
                        position: w.w_axis.truncate().to_array(),
                        rotation,
                        scale,
                        mesh: r.mesh,
                        uv_rect: r.uv_rect,
                        material: zestengine_render::DEFAULT_SPRITE_MATERIAL,
                        texture: r.texture,
                        atlas_layer: 0,
                        alpha_cutoff: m.alpha_cutoff,
                        layer: v(m.sorting_layer),
                        order: m.order_in_layer,
                        range: match m.range {
                            SpriteMaskRange::Auto => MaskRange::Auto,
                            SpriteMaskRange::Custom { back_layer, back_order, front_layer, front_order } => {
                                MaskRange::Custom { back: (v(back_layer), back_order), front: (v(front_layer), front_order) }
                            }
                        },
                    });
                }
            }
        }
        // Flatten sorting groups: grouped renderers take their root's layer / order.
        if !c.nodes.is_empty() && !c.members.is_empty() {
            let over = flatten_groups(&c.nodes, &c.members, &mut c.scratch, &mut c.assigned);
            if over > 0 {
                tracing::warn!("{over} grouped sprites exceed the sorting-group ordinal range");
            }
            for (k, a) in c.assigned.iter().enumerate() {
                let s = &mut c.sources[c.member_src[k]];
                s.layer = a.layer;
                s.order = a.order;
                s.group = a.ordinal;
            }
        }
        // Hand the lists to the renderer (swap keeps both sides' capacity).
        let input = renderer.sprite_input_mut();
        std::mem::swap(&mut input.sprites, &mut c.sources);
        std::mem::swap(&mut input.masks, &mut c.masks);
        input.config.axis = SortAxis { origin: origin.to_array(), axis: forward.to_array() };
        input.config.behind_layer_max = layers.behind_max();
    }

    /// Closest sprite under the ray (`origin`, `dir` in world space): topmost by
    /// sorting layer and order, then nearest. Honours locked sorting layers and
    /// disabled objects. Tight sprites hit by their alpha-outline mesh; Sliced
    /// and Tiled by their `size` rectangle.
    pub fn pick_sprite(&self, origin: Vec3, dir: Vec3) -> Option<(usize, f32)> {
        if self.sprite_cache.resolved.is_empty() && self.sprite_cache.resolved_plain.is_empty() && !self.sprite_cache.shape.active() {
            return None;
        }
        let layers = &self.layers.sorting;
        let mut best: Option<((i16, i16), usize, f32)> = None;
        // Sprite shapes: inside a filled outline, or near the spline.
        for &i in &self.sprite_cache.shape.objs {
            let Some(o) = self.objects.get(i) else { continue };
            let Some(sh) = o.components.iter().filter(|k| component_enabled(k.as_ref())).find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<SpriteShapeRenderer>()) else { continue };
            let Some((poly, filled)) = self.sprite_cache.shape.pick_of(o.id) else { continue };
            if layers.is_locked(sh.sorting_layer) || !self.is_active(i) || self.is_stage_locked(i) || !self.view_pickable(i) {
                continue;
            }
            let world = self.world_transform(i);
            let inv = world.inverse();
            let (lo, ld) = (inv.transform_point3(origin), inv.transform_vector3(dir));
            if ld.z.abs() < 1e-8 {
                continue;
            }
            let t = -lo.z / ld.z;
            if t < 0.0 || super::shape2d::pick_distance(poly, filled, Vec2::new(lo.x + ld.x * t, lo.y + ld.y * t)) > SHAPE_PICK_RADIUS {
                continue;
            }
            let t_world = (world.transform_point3(lo + ld * t) - origin).length();
            let key = (layers.value_of(sh.sorting_layer), sh.order_in_layer);
            if best.is_none_or(|(k, _, bt)| key > k || (key == k && t_world < bt)) {
                best = Some((key, i, t_world));
            }
        }
        for (i, o) in self.objects.iter().enumerate() {
            let Some(s) = o.components.iter().filter(|k| component_enabled(k.as_ref())).find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<SpriteRenderer>()) else {
                continue;
            };
            if layers.is_locked(s.sorting_layer) || !self.is_active(i) || self.is_stage_locked(i) || !self.view_pickable(i) {
                continue;
            }
            let Some(Some(r)) = self.sprite_cache.resolved.get(&s.sprite).or_else(|| self.sprite_cache.resolved_plain.get(&s.sprite)) else { continue };
            let world = self.world_transform(i);
            let inv = world.inverse();
            let lo = inv.transform_point3(origin);
            let ld = inv.transform_vector3(dir);
            if ld.z.abs() < 1e-8 {
                continue;
            }
            let t = -lo.z / ld.z;
            if t < 0.0 {
                continue;
            }
            let mut p = Vec2::new(lo.x + ld.x * t, lo.y + ld.y * t);
            if s.flip_x {
                p.x = -p.x;
            }
            if s.flip_y {
                p.y = -p.y;
            }
            let hit = match s.draw_mode {
                SpriteDrawMode::Simple => point_in_mesh(p, &r.positions, &r.indices),
                _ => {
                    let w = r.size[0].max(1e-6);
                    let h = r.size[1].max(1e-6);
                    let min = Vec2::new(-r.pivot[0] / w * s.size[0], -r.pivot[1] / h * s.size[1]);
                    p.x >= min.x && p.x <= min.x + s.size[0] && p.y >= min.y && p.y <= min.y + s.size[1]
                }
            };
            if !hit {
                continue;
            }
            let t_world = (world.transform_point3(lo + ld * t) - origin).length();
            let key = (layers.value_of(s.sorting_layer), s.order_in_layer);
            if best.is_none_or(|(k, _, bt)| key > k || (key == k && t_world < bt)) {
                best = Some((key, i, t_world));
            }
        }
        best.map(|(_, i, t)| (i, t))
    }

    /// World-space corners of the object's sprite rect (the selection outline
    /// and focus bounds): the mesh bounds for Simple, `size` for Sliced / Tiled.
    pub fn sprite_outline(&self, index: usize) -> Option<[Vec3; 4]> {
        let Some(s) = self.objects[index].components.iter().find_map(|k| component_inner(k.as_ref()).as_any().downcast_ref::<SpriteRenderer>()) else {
            return self.shape_outline(index).or_else(|| self.tilemap_outline(index));
        };
        let r = self.sprite_cache.resolved.get(&s.sprite).or_else(|| self.sprite_cache.resolved_plain.get(&s.sprite))?.as_ref()?;
        let (lo, hi) = match s.draw_mode {
            SpriteDrawMode::Simple => r.bounds,
            _ => {
                let min = [-r.pivot[0] / r.size[0].max(1e-6) * s.size[0], -r.pivot[1] / r.size[1].max(1e-6) * s.size[1]];
                (min, [min[0] + s.size[0], min[1] + s.size[1]])
            }
        };
        let fx = if s.flip_x { -1.0 } else { 1.0 };
        let fy = if s.flip_y { -1.0 } else { 1.0 };
        let w = self.world_transform(index);
        let c = |x: f32, y: f32| w.transform_point3(Vec3::new(x * fx, y * fy, 0.0));
        Some([c(lo[0], lo[1]), c(hi[0], lo[1]), c(hi[0], hi[1]), c(lo[0], hi[1])])
    }
}

/// Whether `p` lies inside any triangle of the mesh (either winding).
fn point_in_mesh(p: Vec2, pos: &[[f32; 2]], idx: &[u16]) -> bool {
    idx.as_chunks::<3>().0.iter().any(|t| {
        let (a, b, c) = (Vec2::from(pos[t[0] as usize]), Vec2::from(pos[t[1] as usize]), Vec2::from(pos[t[2] as usize]));
        let s = |u: Vec2, v: Vec2, w: Vec2| (u.x - w.x) * (v.y - w.y) - (v.x - w.x) * (u.y - w.y);
        let (d1, d2, d3) = (s(p, a, b), s(p, b, c), s(p, c, a));
        let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let pos_ = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(neg && pos_)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_rotation_scale_and_mirror() {
        let w = Mat4::from_scale_rotation_translation(Vec3::new(2.0, 3.0, 1.0), glam::Quat::from_rotation_z(0.5), Vec3::ZERO);
        let (rot, sc, mir) = decompose_2d(&w);
        assert!((rot - 0.5).abs() < 1e-5 && (sc[0] - 2.0).abs() < 1e-5 && (sc[1] - 3.0).abs() < 1e-5 && !mir);
        // A negative X scale mirrors: rebuild the matrix from the pieces and compare.
        let m = Mat4::from_scale_rotation_translation(Vec3::new(-2.0, 3.0, 1.0), glam::Quat::from_rotation_z(0.5), Vec3::ZERO);
        let (rot, sc, mir) = decompose_2d(&m);
        assert!(mir);
        let rebuilt = Mat4::from_scale_rotation_translation(Vec3::new(if mir { -sc[0] } else { sc[0] }, sc[1], 1.0), glam::Quat::from_rotation_z(rot), Vec3::ZERO);
        for (a, b) in m.to_cols_array().iter().zip(rebuilt.to_cols_array()) {
            assert!((a - b).abs() < 1e-4, "{m:?} vs {rebuilt:?}");
        }
    }

    #[test]
    fn point_in_mesh_hits_inside_only() {
        let pos = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let idx = [0u16, 1, 2];
        assert!(point_in_mesh(Vec2::new(0.2, 0.2), &pos, &idx));
        assert!(!point_in_mesh(Vec2::new(0.9, 0.9), &pos, &idx));
        // Clockwise triangles hit too.
        assert!(point_in_mesh(Vec2::new(0.2, 0.2), &pos, &[0, 2, 1]));
    }
}
