//! Lazy 3D previews of model files and prefabs for the file inspector (AUI-4 /
//! IMP-7): a small orbitable view of the asset under a studio light, rendered by
//! the renderer's model-only path (`Renderer::set_model_only_preview`), so the
//! asset needs no instance in the scene and nothing in the scene or the asset
//! changes.
//!
//! Cost model, in the order the states are entered:
//!
//! * hidden (the inspector shows something else): nothing runs; after
//!   [`KEEP_HIDDEN`] the uploaded meshes, textures and materials are freed;
//! * visible with a cached image and an unmoved camera: the cached egui texture
//!   is drawn, no model is loaded or rendered;
//! * the first time, or after the camera moved: the model is parsed on a worker
//!   thread, uploaded, and rendered each frame; once the camera has been still
//!   for [`SETTLE_FRAMES`] frames the target is read back into the cache and the
//!   renderer's target is released again.
//!
//! The pure parts (draw plan from a model or prefab, bounds, framing, the settle
//! tracker, the cache key) are tested without a GPU; `tests/asset_preview_gpu.rs`
//! renders a real model on a headless window and checks the pixels.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use glam::{Mat4, Quat, Vec3};
use zestengine_assets::{ObjectSource, Scene, SceneEntry};
use zestengine_render::{DrawCmd, MaterialDesc, MeshData, MaterialHandle, MeshHandle, ModelOnlyPreview, PostFx, Renderer, ShadowCasting, TextureHandle};

use crate::model_preview::{self, ModelPreview, Orbit, PreviewRequest};

/// Frames the camera must stay still before the view is read back into the cache.
pub const SETTLE_FRAMES: u32 = 3;
/// How long the uploaded resources are kept after the preview stops being shown.
pub const KEEP_HIDDEN: Duration = Duration::from_secs(4);
/// Height of the preview area in points.
pub const HEIGHT: f32 = 230.0;
/// Cached images kept (oldest dropped first).
const CACHE_LIMIT: usize = 24;
/// The view the first framing looks from: a three-quarter view slightly from above.
pub const HOME_YAW: f32 = 0.6;
pub const HOME_PITCH: f32 = 0.22;

/// Linear background behind the model (a dark studio grey).
const BACKGROUND: [f32; 4] = [0.012, 0.013, 0.017, 1.0];

/// What to preview.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PreviewSource {
    /// An imported model file (absolute path).
    Model(PathBuf),
    /// A `.prefab` (absolute path) with the project root its model paths are relative to.
    Prefab { path: PathBuf, root: PathBuf },
}

impl PreviewSource {
    pub fn path(&self) -> &Path {
        match self {
            Self::Model(p) | Self::Prefab { path: p, .. } => p,
        }
    }
}

// ------------------------------------------------------------------ pure plan

/// One draw of the plan: a mesh and material of one of the loaded models at a transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanItem {
    /// Index into the plan's model list.
    pub model: usize,
    pub mesh: usize,
    pub material: usize,
    pub transform: Mat4,
}

/// Every instance slot of a model at the instance's own transform.
pub fn plan_model(scene: &Scene, model: usize) -> Vec<PlanItem> {
    let mut out = Vec::new();
    // Meshes the file hides (Maya leaves blendshape target copies in as hidden nodes) are not
    // part of the model: drawing them would stack dozens of body copies on the real one.
    let hidden: std::collections::HashSet<usize> = scene.identity.hidden_instances.iter().copied().collect();
    for (index, inst) in scene.instances.iter().enumerate() {
        if hidden.contains(&index) {
            continue;
        }
        for slot in &inst.slots {
            if slot.mesh < scene.meshes.len() {
                out.push(PlanItem { model, mesh: slot.mesh, material: slot.material, transform: inst.transform });
            }
        }
    }
    out
}

/// The model-sourced entries of a prefab at their world transforms (parent chain
/// applied; a disabled entry hides its subtree). Returns the model paths in first-use
/// order and the items, whose `material` is `usize::MAX` (the model's own material for
/// that mesh, resolved when the models are loaded). Entries that are not models
/// (primitives, lights, empties) draw nothing here.
pub fn plan_prefab(entries: &[SceneEntry]) -> (Vec<String>, Vec<PlanItem>) {
    let mut paths: Vec<String> = Vec::new();
    let mut items = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let ObjectSource::Model { path, mesh, extra_meshes, .. } = &e.source else { continue };
        let Some(world) = entry_world(entries, i) else { continue };
        let model = paths.iter().position(|p| p == path).unwrap_or_else(|| {
            paths.push(path.clone());
            paths.len() - 1
        });
        for m in std::iter::once(mesh).chain(extra_meshes) {
            items.push(PlanItem { model, mesh: *m, material: usize::MAX, transform: world });
        }
    }
    (paths, items)
}

/// World matrix of entry `i` (`None` when it or an ancestor is disabled, or the
/// parent chain loops).
fn entry_world(entries: &[SceneEntry], i: usize) -> Option<Mat4> {
    let mut world = Mat4::IDENTITY;
    let mut at = Some(i);
    let mut hops = 0;
    while let Some(k) = at {
        let e = entries.get(k)?;
        if !e.enabled || hops > entries.len() {
            return None;
        }
        let local = Mat4::from_scale_rotation_translation(Vec3::from(e.scale), Quat::from_array(e.rotation).normalize(), Vec3::from(e.translation));
        world = local * world;
        at = e.parent;
        hops += 1;
    }
    Some(world)
}

/// Axis-aligned bounds of a mesh's positions.
pub fn mesh_bounds(mesh: &zestengine_render::MeshData) -> Option<(Vec3, Vec3)> {
    let mut it = mesh.vertices.iter().map(|v| Vec3::from(v.position));
    let first = it.next()?;
    Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
}

/// The eight corners of `bounds` under `transform` (what the framing fits).
pub fn corners(bounds: (Vec3, Vec3), transform: Mat4) -> [Vec3; 8] {
    let (lo, hi) = bounds;
    std::array::from_fn(|k| {
        let p = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
        transform.transform_point3(p)
    })
}

/// Framing points of the whole plan: the corners of every drawn mesh's bounds.
pub fn plan_points(items: &[PlanItem], bounds: &[Vec<Option<(Vec3, Vec3)>>]) -> Vec<Vec3> {
    items
        .iter()
        .filter_map(|it| bounds.get(it.model)?.get(it.mesh).copied().flatten().map(|b| corners(b, it.transform)))
        .flatten()
        .collect()
}

/// Share of the view the framing lets the points fill (the rest is margin).
const FILL: f32 = 0.9;

/// The home orbit of a plan: the three-quarter view that fits `points`. The distance is the
/// nearest one at which every point is inside the view (not the bounding sphere's, which for a
/// boxy asset leaves it a third of the view).
pub fn home_orbit(points: &[Vec3], aspect: f32) -> Option<Orbit> {
    let mut o = model_preview::frame_points(points, HOME_YAW, HOME_PITCH, aspect)?;
    let (right, up) = o.basis();
    let toward_eye = (o.eye() - o.target).normalize_or_zero();
    let th = (model_preview::FOV_Y * 0.5).tan();
    let tw = th * aspect.max(0.2);
    let mut dist = 0.0f32;
    for p in points {
        let r = *p - o.target;
        let depth_toward_eye = r.dot(toward_eye);
        dist = dist
            .max(depth_toward_eye + r.dot(right).abs() / (tw * FILL))
            .max(depth_toward_eye + r.dot(up).abs() / (th * FILL));
    }
    if dist > 1e-3 {
        o.dist = dist;
    }
    Some(o)
}

/// Counts consecutive frames with the same camera.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settle {
    last: Option<Mat4>,
    still: u32,
}

impl Settle {
    /// Feed this frame's view matrix; `true` once it has been unchanged for
    /// [`SETTLE_FRAMES`] frames in a row.
    pub fn step(&mut self, view: Mat4) -> bool {
        if self.last == Some(view) {
            self.still += 1;
        } else {
            self.last = Some(view);
            self.still = 0;
        }
        self.still >= SETTLE_FRAMES
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Identity of a cached render: the source files' modification stamps.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub source: PreviewSource,
    /// Modification times (ms since the epoch) of the source and its `.meta`.
    pub stamps: [u128; 2],
}

fn stamp(path: &Path) -> u128 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

impl CacheKey {
    pub fn of(source: &PreviewSource) -> Self {
        let path = source.path();
        let mut meta = path.as_os_str().to_owned();
        meta.push(".meta");
        Self { source: source.clone(), stamps: [stamp(path), stamp(Path::new(&meta))] }
    }
}

/// A DrawCmd for the model-only pass (no culling data from the scene, no lightmap, no highlight).
pub fn draw_cmd(mesh: MeshHandle, material: MaterialHandle, transform: Mat4, bounds: Option<(Vec3, Vec3)>) -> DrawCmd {
    let (lo, hi) = bounds.unwrap_or((Vec3::ZERO, Vec3::ZERO));
    DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform,
        highlight: 0.0,
        mesh_center: (lo + hi) * 0.5,
        bound_radius: ((hi - lo) * 0.5).length(),
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index: -1,
        occluded: false,
        shadow_casting: ShadowCasting::On,
        instance_color: None,
        instance_param: None,
    }
}

// ------------------------------------------------------------------ GPU side

/// What the worker thread parsed.
struct Loaded {
    scenes: Vec<(PathBuf, Scene)>,
    items: Vec<PlanItem>,
    /// Models that failed to load (prefab previews skip them).
    missing: Vec<String>,
}

/// Uploaded resources of one model.
struct GpuModel {
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    textures: Vec<TextureHandle>,
    bounds: Vec<Option<(Vec3, Vec3)>>,
    /// Per mesh: the material the model assigns it (the first slot that uses it).
    mesh_material: Vec<usize>,
}

struct Gpu {
    models: Vec<GpuModel>,
    items: Vec<PlanItem>,
    missing: Vec<String>,
}

/// `skinned`: meshes with skin weights go to host-visible buffers, so they can be rewritten
/// each frame with a CPU-skinned pose ([`AvatarFigure`]); otherwise every mesh is static.
fn upload(renderer: &mut Renderer, loaded: Loaded, skinned: bool) -> anyhow::Result<Gpu> {
    let mut models = Vec::new();
    for (path, scene) in &loaded.scenes {
        let textures = scene
            .textures
            .iter()
            .enumerate()
            .map(|(k, t)| renderer.upload_compressed_texture(&zestengine_assets::compress_embedded_texture(Some(path), k, t)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut materials = Vec::new();
        for m in &scene.materials {
            let desc = MaterialDesc {
                name: m.name.clone(),
                params: m.params,
                features: m.features,
                albedo: m.albedo.map(|i| textures[i]),
                normal: m.normal.map(|i| textures[i]),
                orm: m.orm.map(|i| textures[i]),
                emission: m.emission.map(|i| textures[i]),
                error: false,
                ..Default::default()
            };
            materials.push(renderer.create_material(&desc)?);
        }
        if materials.is_empty() {
            materials.push(renderer.create_material(&MaterialDesc { name: "preview".into(), ..Default::default() })?);
        }
        let mut meshes = Vec::new();
        for mesh in &scene.meshes {
            meshes.push(if skinned && is_skinned(mesh) { renderer.upload_mesh_skinned(mesh)? } else { renderer.upload_mesh(mesh)? });
        }
        let mut mesh_material = vec![0usize; scene.meshes.len()];
        let mut seen = vec![false; scene.meshes.len()];
        for slot in scene.instances.iter().flat_map(|i| &i.slots) {
            if slot.mesh < seen.len() && !seen[slot.mesh] {
                seen[slot.mesh] = true;
                mesh_material[slot.mesh] = slot.material.min(materials.len() - 1);
            }
        }
        let bounds = scene.meshes.iter().map(mesh_bounds).collect();
        models.push(GpuModel { meshes, materials, textures, bounds, mesh_material });
    }
    Ok(Gpu { models, items: loaded.items, missing: loaded.missing })
}

impl Gpu {
    fn release(self, renderer: &mut Renderer) {
        let (mut m, mut t, mut mat) = (HashSet::new(), HashSet::new(), HashSet::new());
        for g in &self.models {
            m.extend(g.meshes.iter().map(|h| h.index()));
            t.extend(g.textures.iter().map(|h| h.index()));
            mat.extend(g.materials.iter().map(|h| h.index()));
        }
        if let Err(e) = renderer.free_resources(&m, &t, &mat) {
            tracing::warn!("freeing the asset preview: {e:#}");
        }
    }

    fn points(&self) -> Vec<Vec3> {
        let bounds: Vec<Vec<Option<(Vec3, Vec3)>>> = self.models.iter().map(|m| m.bounds.clone()).collect();
        plan_points(&self.items, &bounds)
    }

    fn draws(&self, out: &mut Vec<DrawCmd>) {
        out.clear();
        for it in &self.items {
            let Some(g) = self.models.get(it.model) else { continue };
            let (Some(&mesh), Some(&b)) = (g.meshes.get(it.mesh), g.bounds.get(it.mesh)) else { continue };
            let mat = if it.material == usize::MAX { g.mesh_material[it.mesh] } else { it.material };
            let Some(&material) = g.materials.get(mat.min(g.materials.len().saturating_sub(1))) else { continue };
            out.push(draw_cmd(mesh, material, it.transform, b));
        }
    }
}

fn load(source: &PreviewSource) -> Result<Loaded, String> {
    match source {
        PreviewSource::Model(path) => {
            let scene = zestengine_assets::load_model_with_meta(path).map_err(|e| format!("{e:#}"))?;
            let items = plan_model(&scene, 0);
            Ok(Loaded { scenes: vec![(path.clone(), scene)], items, missing: Vec::new() })
        }
        PreviewSource::Prefab { path, root } => {
            let prefab = zestengine_engine::prefab::Prefab::load(path).map_err(|e| format!("{e:#}"))?;
            let (paths, mut items) = plan_prefab(&prefab.entries);
            let mut scenes = Vec::new();
            let mut missing = Vec::new();
            let mut remap = HashMap::new();
            for (k, p) in paths.iter().enumerate() {
                let abs = root.join(p);
                match zestengine_assets::load_model_with_meta(&abs) {
                    Ok(scene) => {
                        remap.insert(k, scenes.len());
                        scenes.push((abs, scene));
                    }
                    Err(e) => missing.push(format!("{p}: {e:#}")),
                }
            }
            items.retain_mut(|it| match remap.get(&it.model) {
                Some(&n) => {
                    it.model = n;
                    scenes[n].1.meshes.len() > it.mesh
                }
                None => false,
            });
            Ok(Loaded { scenes, items, missing })
        }
    }
}

fn is_skinned(mesh: &MeshData) -> bool {
    mesh.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0)
}

/// A finished render kept as an egui texture.
struct Cached {
    texture: egui::TextureHandle,
    /// Orbit the image was rendered from, the framing it started from, and its pixel size.
    orbit: Orbit,
    home: Orbit,
    view: Mat4,
    /// Pixel size of the image, and of the request it answered.
    size: [u32; 2],
    req_size: [u32; 2],
}

enum Stage {
    /// Nothing loaded; the cached image (if any) is shown.
    Idle,
    Loading(Receiver<Result<Loaded, String>>),
    Ready(Gpu),
    Failed(String),
}

/// The preview pane. One lives in the editor; [`Self::ui`] draws it in the inspector
/// and [`Self::frame`] feeds the renderer once per frame.
pub struct AssetPreview {
    widget: ModelPreview,
    source: Option<PreviewSource>,
    key: Option<CacheKey>,
    stage: Stage,
    ctx: Option<egui::Context>,
    cache: HashMap<CacheKey, Cached>,
    cache_order: Vec<CacheKey>,
    /// `ui` ran since the last `frame`.
    shown: bool,
    last_shown: Instant,
    request: Option<PreviewRequest>,
    settle: Settle,
    draws: Vec<DrawCmd>,
    /// A model-only request was made last frame (the renderer's target is live).
    live: bool,
    /// The widget has a framing (restored from the cache or computed from the bounds).
    framed: bool,
    /// The loaded data no longer matches the file: drop it before the next use.
    stale: bool,
    /// When the target last rendered (idle resources are freed after [`KEEP_HIDDEN`]).
    last_render: Instant,
    next_stamp_check: Instant,
}

impl Default for AssetPreview {
    fn default() -> Self {
        Self {
            widget: ModelPreview::default(),
            source: None,
            key: None,
            stage: Stage::Idle,
            ctx: None,
            cache: HashMap::new(),
            cache_order: Vec::new(),
            shown: false,
            last_shown: Instant::now(),
            request: None,
            settle: Settle::default(),
            draws: Vec::new(),
            live: false,
            framed: false,
            stale: false,
            last_render: Instant::now(),
            next_stamp_check: Instant::now(),
        }
    }
}

impl AssetPreview {
    /// The renderer's model-preview texture for this frame (`None` while the rig
    /// Configure window owns the target, or nothing is rendering).
    pub fn set_live_texture(&mut self, texture: Option<(egui::TextureId, [f32; 2])>) {
        self.widget.set_texture(if self.live { texture } else { None });
    }

    /// Draw the preview of `source` in the inspector and take the camera input.
    pub fn ui(&mut self, ui: &mut egui::Ui, source: &PreviewSource) {
        self.ctx = Some(ui.ctx().clone());
        self.shown = true;
        self.last_shown = Instant::now();
        if self.source.as_ref() != Some(source) {
            self.switch(source);
        } else if Instant::now() >= self.next_stamp_check {
            // A changed file (or `.meta`) invalidates: checked twice a second at most.
            self.next_stamp_check = Instant::now() + Duration::from_millis(500);
            let key = CacheKey::of(source);
            if self.key.as_ref() != Some(&key) {
                self.key = Some(key);
                self.stale = true;
                self.framed = false;
            }
        }
        crate::widgets::eyebrow(ui, "Preview");
        let size = egui::vec2(ui.available_width().clamp(120.0, 640.0), HEIGHT);
        if !self.framed
            && let Some(c) = self.cached()
        {
            let (orbit, home) = (c.orbit, c.home);
            self.widget.restore(home, orbit);
            self.framed = true;
        }
        let name = source.path().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let mut has_texture_now = self.widget.has_texture();
        if !self.live
            && let Some(c) = self.cached() {
                self.widget.set_texture(Some((c.texture.id(), [c.size[0] as f32, c.size[1] as f32])));
                has_texture_now = true;
            }
        let stage_note = match &self.stage {
            Stage::Loading(_) => Some("Loading…".to_owned()),
            Stage::Failed(e) => Some(format!("Preview failed: {e}")),
            Stage::Idle if !has_texture_now => Some("Drag to preview".to_owned()),
            Stage::Ready(g) if !g.missing.is_empty() => Some(format!("{} model(s) not found", g.missing.len())),
            _ => None,
        };
        self.widget.ui(ui, size, Mat4::IDENTITY, |painter, proj, resp| {
            if let Some(note) = &stage_note {
                painter.text(proj.rect().center(), egui::Align2::CENTER_CENTER, note, egui::FontId::proportional(crate::theme::SIZE_META), crate::theme::TEXT_DIM);
            }
            if resp.hovered() {
                painter.text(
                    proj.rect().left_bottom() + egui::vec2(6.0, -6.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{name}  ·  drag: orbit  wheel: zoom  right-drag: pan"),
                    egui::FontId::proportional(crate::theme::SIZE_META),
                    crate::theme::TEXT_HIDDEN,
                );
            }
        });
        self.request = self.widget.take_request();
    }

    fn cached(&self) -> Option<&Cached> {
        self.key.as_ref().and_then(|k| self.cache.get(k))
    }

    fn switch(&mut self, source: &PreviewSource) {
        self.widget.reset();
        self.source = Some(source.clone());
        self.key = Some(CacheKey::of(source));
        self.stale = true;
        self.framed = false;
        self.live = false;
        self.next_stamp_check = Instant::now() + Duration::from_millis(500);
    }

    /// Called once per frame after the UI pass and before rendering. `allowed` is false while
    /// something else owns the model-preview target (the rig Configure window).
    pub fn frame(&mut self, renderer: &mut Renderer, allowed: bool) {
        if !std::mem::take(&mut self.shown) || !allowed {
            self.live = false;
            self.request = None;
            // Hidden: free the GPU copy once it has been unused for a while; this is the only work.
            if matches!(self.stage, Stage::Ready(_)) && self.last_shown.elapsed() > KEEP_HIDDEN {
                self.drop_gpu(renderer);
            }
            return;
        }
        if std::mem::take(&mut self.stale) {
            self.drop_gpu(renderer);
            self.stage = Stage::Idle;
            self.settle.reset();
        }
        let Some(req) = self.request.take() else { return };
        let Some(source) = self.source.clone() else { return };
        // Does the cached image still show what the camera sees?
        let cache_ok = self.cached().is_some_and(|c| c.view == req.view && c.req_size == req.size);
        if cache_ok && !matches!(self.stage, Stage::Loading(_)) {
            self.live = false;
            if matches!(self.stage, Stage::Ready(_)) && self.last_render.elapsed() > KEEP_HIDDEN {
                self.drop_gpu(renderer);
            }
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Idle) {
            Stage::Idle => {
                let (tx, rx) = std::sync::mpsc::channel();
                let src = source.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(load(&src));
                });
                self.stage = Stage::Loading(rx);
                self.repaint_soon();
            }
            Stage::Loading(rx) => match rx.try_recv() {
                Ok(Ok(loaded)) => match upload(renderer, loaded, false) {
                    Ok(gpu) => {
                        self.stage = Stage::Ready(gpu);
                        self.settle.reset();
                        self.frame_model(req.size);
                        self.repaint_soon();
                    }
                    Err(e) => self.stage = Stage::Failed(format!("{e:#}")),
                },
                Ok(Err(e)) => self.stage = Stage::Failed(e),
                Err(TryRecvError::Empty) => {
                    self.stage = Stage::Loading(rx);
                    self.repaint_soon();
                }
                Err(TryRecvError::Disconnected) => self.stage = Stage::Failed("the loader stopped".into()),
            },
            Stage::Ready(gpu) => {
                self.stage = Stage::Ready(gpu);
                self.render(renderer, req);
            }
            Stage::Failed(e) => self.stage = Stage::Failed(e),
        }
    }

    fn repaint_soon(&self) {
        if let Some(ctx) = &self.ctx {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    /// First framing of a freshly uploaded asset (when no cached framing exists).
    fn frame_model(&mut self, size: [u32; 2]) {
        let Stage::Ready(gpu) = &self.stage else { return };
        let aspect = size[0] as f32 / size[1].max(1) as f32;
        if let Some(home) = home_orbit(&gpu.points(), aspect)
            && !self.framed {
                self.widget.go_to(home);
                self.framed = true;
            }
    }

    fn render(&mut self, renderer: &mut Renderer, req: PreviewRequest) {
        let Stage::Ready(gpu) = &self.stage else { return };
        gpu.draws(&mut self.draws);
        let camera = zestengine_render::CameraData { view: req.view, proj: req.proj, position: req.position };
        let postfx = PostFx { bloom_enabled: false, grading_enabled: false, vignette_enabled: false, ca_enabled: false, ..PostFx::default() };
        renderer.set_model_only_preview(Some(&ModelOnlyPreview {
            camera,
            size: req.size,
            draws: &self.draws,
            background: BACKGROUND,
            light: zestengine_render::neutral_light(),
            postfx,
            want_depth: false,
        }));
        let was_live = self.live;
        self.live = true;
        self.last_render = Instant::now();
        self.repaint_soon();
        // The frame that rendered `req.view` is the one in flight; the texture is current
        // `SETTLE_FRAMES` frames after the camera stopped moving.
        if was_live && self.settle.step(req.view) {
            self.store(renderer, req);
        } else if !was_live {
            self.settle.reset();
            self.settle.step(req.view);
        }
    }

    /// Read the settled target back into the cache and let the renderer's target go.
    fn store(&mut self, renderer: &mut Renderer, req: PreviewRequest) {
        let (Some(ctx), Some(key)) = (self.ctx.clone(), self.key.clone()) else { return };
        match renderer.debug_read_model_preview() {
            Ok(Some((w, h, px))) if w * h * 4 == px.len() as u32 => {
                let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
                let texture = ctx.load_texture(format!("asset-preview:{}", key.source.path().display()), image, egui::TextureOptions::LINEAR);
                let home = self.widget.home();
                let cached = Cached { texture, orbit: self.widget.orbit(), home, view: req.view, size: [w, h], req_size: req.size };
                self.cache.insert(key.clone(), cached);
                self.cache_order.retain(|k| *k != key);
                self.cache_order.push(key);
                while self.cache_order.len() > CACHE_LIMIT {
                    let old = self.cache_order.remove(0);
                    self.cache.remove(&old);
                }
                self.live = false;
                renderer.set_model_only_preview(None);
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("reading the asset preview back: {e:#}");
                self.settle.reset();
            }
        }
    }

    fn drop_gpu(&mut self, renderer: &mut Renderer) {
        if let Stage::Ready(gpu) = std::mem::replace(&mut self.stage, Stage::Idle) {
            gpu.release(renderer);
        }
        self.live = false;
    }
}

// ------------------------------------------------------------------ avatar figure

/// Linear background behind the Configure window's figure (a lighter grey than the inspector's).
const FIGURE_BACKGROUND: [f32; 4] = [0.04, 0.042, 0.05, 1.0];

/// One skinned mesh of the figure: its upload and the bind-pose vertices it is skinned from.
struct FigureMesh {
    handle: MeshHandle,
    base: Vec<zestengine_render::Vertex>,
}

struct Figure {
    gpu: Gpu,
    skinned: Vec<FigureMesh>,
    skeleton: Option<zestengine_assets::Skeleton>,
    /// The pose the skinned meshes were last written for.
    posed: Vec<Mat4>,
}

enum FigureStage {
    Idle,
    Loading(Receiver<Result<Loaded, String>>),
    Ready(Box<Figure>),
    Failed,
}

/// The model of the avatar Configure window rendered on its own (no scene object needed), for the
/// window's figure: the model's meshes skinned on the CPU to the pose the window asks for, through
/// the same model-only pass as [`AssetPreview`]. Static meshes are drawn at their instance
/// transform, skinned ones in the skeleton's frame (the frame the bone dots are projected in).
pub struct AvatarFigure {
    path: Option<PathBuf>,
    stamp: u128,
    stage: FigureStage,
    next_stamp_check: Instant,
    draws: Vec<DrawCmd>,
    /// The model-only target is held (it is released when the window closes).
    live: bool,
}

impl Default for AvatarFigure {
    fn default() -> Self {
        Self { path: None, stamp: 0, stage: FigureStage::Idle, next_stamp_check: Instant::now(), draws: Vec::new(), live: false }
    }
}

impl AvatarFigure {
    /// Whether the figure can be shown (false once the model failed to load).
    pub fn usable(&self) -> bool {
        !matches!(self.stage, FigureStage::Failed)
    }

    /// Called once per frame after the UI pass. `model` is the open Configure window's model
    /// (`None` frees everything), `req` the camera the window's preview asked for and `pose` the
    /// joint locals it wants shown (the skeleton's rest pose when `None`).
    pub fn frame(&mut self, ctx: &egui::Context, renderer: &mut Renderer, model: Option<&Path>, req: Option<PreviewRequest>, pose: Option<&[Mat4]>) {
        let Some(model) = model else {
            self.release(renderer);
            self.path = None;
            return;
        };
        // A different model, or a changed file (a reimport after Apply), starts over.
        if self.path.as_deref() != Some(model) {
            self.release(renderer);
            self.path = Some(model.to_path_buf());
            self.stamp = stamp(model);
        } else if Instant::now() >= self.next_stamp_check {
            self.next_stamp_check = Instant::now() + Duration::from_millis(500);
            let now = stamp(model);
            if now != self.stamp {
                self.release(renderer);
                self.stamp = now;
            }
        }
        match std::mem::replace(&mut self.stage, FigureStage::Idle) {
            FigureStage::Idle => {
                let (tx, rx) = std::sync::mpsc::channel();
                let src = PreviewSource::Model(model.to_path_buf());
                std::thread::spawn(move || {
                    let _ = tx.send(load(&src));
                });
                self.stage = FigureStage::Loading(rx);
                ctx.request_repaint_after(Duration::from_millis(16));
            }
            FigureStage::Loading(rx) => match rx.try_recv() {
                Ok(Ok(loaded)) => match Self::build(renderer, loaded) {
                    Ok(figure) => {
                        self.stage = FigureStage::Ready(Box::new(figure));
                        ctx.request_repaint_after(Duration::from_millis(16));
                    }
                    Err(e) => {
                        tracing::warn!("avatar figure upload: {e:#}");
                        self.stage = FigureStage::Failed;
                    }
                },
                Ok(Err(e)) => {
                    tracing::warn!("avatar figure load: {e}");
                    self.stage = FigureStage::Failed;
                }
                Err(TryRecvError::Empty) => {
                    self.stage = FigureStage::Loading(rx);
                    ctx.request_repaint_after(Duration::from_millis(16));
                }
                Err(TryRecvError::Disconnected) => self.stage = FigureStage::Failed,
            },
            FigureStage::Ready(mut figure) => {
                if let Some(req) = req {
                    figure.pose(renderer, pose);
                    figure.gpu.draws(&mut self.draws);
                    let camera = zestengine_render::CameraData { view: req.view, proj: req.proj, position: req.position };
                    let postfx = PostFx { bloom_enabled: false, grading_enabled: false, vignette_enabled: false, ca_enabled: false, ..PostFx::default() };
                    renderer.set_model_only_preview(Some(&ModelOnlyPreview {
                        camera,
                        size: req.size,
                        draws: &self.draws,
                        background: FIGURE_BACKGROUND,
                        light: zestengine_render::neutral_light(),
                        postfx,
                        want_depth: false,
                    }));
                    self.live = true;
                    ctx.request_repaint_after(Duration::from_millis(16));
                }
                self.stage = FigureStage::Ready(figure);
            }
            FigureStage::Failed => self.stage = FigureStage::Failed,
        }
    }

    fn build(renderer: &mut Renderer, mut loaded: Loaded) -> anyhow::Result<Figure> {
        let Some((_, scene)) = loaded.scenes.first() else { anyhow::bail!("no model") };
        let skeleton = scene.skeletons.first().cloned();
        // Skinned meshes are drawn in the skeleton's frame (identity), like the scene draws an
        // animated object; the rest keep their instance transform.
        let weighted: Vec<bool> = scene.meshes.iter().map(|m| skeleton.is_some() && is_skinned(m)).collect();
        for it in &mut loaded.items {
            if weighted.get(it.mesh).copied().unwrap_or(false) {
                it.transform = Mat4::IDENTITY;
            }
        }
        let bases: Vec<Option<Vec<zestengine_render::Vertex>>> =
            scene.meshes.iter().zip(&weighted).map(|(m, &w)| w.then(|| m.vertices.clone())).collect();
        let mut gpu = upload(renderer, loaded, true)?;
        let model = gpu.models.first_mut().ok_or_else(|| anyhow::anyhow!("no model"))?;
        let mut skinned = Vec::new();
        for (k, base) in bases.into_iter().enumerate() {
            let Some(base) = base else { continue };
            // A posed limb leaves the bind-pose bounds: cull with a generous box instead.
            if let Some((lo, hi)) = &mut model.bounds[k] {
                let ext = *hi - *lo;
                *lo -= ext;
                *hi += ext;
            }
            skinned.push(FigureMesh { handle: model.meshes[k], base });
        }
        Ok(Figure { gpu, skinned, skeleton, posed: Vec::new() })
    }

    fn release(&mut self, renderer: &mut Renderer) {
        if let FigureStage::Ready(figure) = std::mem::replace(&mut self.stage, FigureStage::Idle) {
            figure.gpu.release(renderer);
        } else {
            self.stage = FigureStage::Idle;
        }
        if std::mem::take(&mut self.live) {
            renderer.set_model_only_preview(None);
        }
    }
}

impl Figure {
    /// Skin the meshes to `locals` (the skeleton's rest pose when `None` or of the wrong size),
    /// when it differs from the pose last written.
    fn pose(&mut self, renderer: &mut Renderer, locals: Option<&[Mat4]>) {
        let Some(skel) = &self.skeleton else { return };
        let locals: Vec<Mat4> = match locals {
            Some(l) if l.len() == skel.joints.len() => l.to_vec(),
            _ => skel.rest_locals(),
        };
        if locals == self.posed {
            return;
        }
        let palette = skel.palette(&locals);
        for m in &self.skinned {
            let verts: Vec<zestengine_render::Vertex> = m
                .base
                .iter()
                .map(|v| {
                    let mut o = *v;
                    o.position = zestengine_assets::skin_position(Vec3::from(v.position), v.joints, v.weights, &palette).to_array();
                    o.normal = zestengine_assets::skin_direction(Vec3::from(v.normal), v.joints, v.weights, &palette).to_array();
                    o
                })
                .collect();
            renderer.update_mesh_vertices(m.handle, &verts);
        }
        self.posed = locals;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::Instance;
    use zestengine_render::{MeshData, Vertex};

    fn tri(lo: Vec3, hi: Vec3) -> MeshData {
        let v = |p: Vec3| Vertex { position: p.to_array(), ..Default::default() };
        MeshData { vertices: vec![v(lo), v(hi), v((lo + hi) * 0.5)], indices: vec![0, 1, 2], has_lightmap_uv: false }
    }

    fn entry(source: ObjectSource, parent: Option<usize>, t: [f32; 3]) -> SceneEntry {
        SceneEntry {
            id: String::new(),
            name: "e".into(),
            source,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: zestengine_assets::MaterialRef::Model,
            extra_materials: vec![],
            renderer_split: false,
            parent,
            components: vec![],
            translation: t,
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

    fn model(path: &str, mesh: usize) -> ObjectSource {
        ObjectSource::Model { path: path.into(), mesh, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] }
    }

    #[test]
    fn model_plan_lists_every_slot_with_its_transform() {
        let mut scene = empty_scene();
        scene.meshes = vec![tri(Vec3::ZERO, Vec3::ONE), tri(Vec3::ZERO, Vec3::ONE)];
        scene.instances = vec![
            Instance::single("a", 0, 0, Mat4::from_translation(Vec3::X)),
            Instance { name: "b".into(), transform: Mat4::IDENTITY, slots: vec![zestengine_assets::MeshSlot { mesh: 1, material: 2 }, zestengine_assets::MeshSlot { mesh: 9, material: 0 }] },
        ];
        let plan = plan_model(&scene, 3);
        assert_eq!(plan.len(), 2, "a slot pointing at a missing mesh is dropped");
        assert_eq!((plan[0].model, plan[0].mesh, plan[0].transform), (3, 0, Mat4::from_translation(Vec3::X)));
        assert_eq!((plan[1].mesh, plan[1].material), (1, 2));
    }

    fn empty_scene() -> Scene {
        Scene {
            meshes: vec![],
            morphs: vec![],
            springs: Default::default(),
            textures: vec![],
            materials: vec![],
            instances: vec![],
            skeletons: vec![],
            animations: vec![],
            collision: vec![],
            identity: Default::default(),
        }
    }

    #[test]
    fn prefab_plan_applies_parents_and_skips_disabled_subtrees_and_non_models() {
        let mut off = entry(model("m.glb", 1), None, [0.0; 3]);
        off.enabled = false;
        let entries = vec![
            entry(ObjectSource::Empty, None, [10.0, 0.0, 0.0]),
            entry(model("a.glb", 0), Some(0), [0.0, 2.0, 0.0]),
            off,
            entry(model("b.glb", 4), Some(2), [0.0; 3]),
            entry(model("a.glb", 3), None, [0.0; 3]),
        ];
        let (paths, items) = plan_prefab(&entries);
        assert_eq!(paths, vec!["a.glb".to_owned()], "disabled subtree and the empty draw nothing");
        assert_eq!(items.len(), 2);
        let p = items[0].transform.transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(10.0, 2.0, 0.0)).length() < 1e-5, "child sits at parent + local: {p}");
        assert_eq!((items[1].model, items[1].mesh), (0, 3), "the second use of a model shares its slot");
    }

    #[test]
    fn a_parent_loop_draws_nothing_instead_of_hanging() {
        let mut a = entry(model("a.glb", 0), Some(1), [0.0; 3]);
        a.name = "a".into();
        let b = entry(ObjectSource::Empty, Some(0), [0.0; 3]);
        assert!(plan_prefab(&[a, b]).1.is_empty());
    }

    #[test]
    fn corners_follow_the_transform_and_the_plan_points_use_the_mesh_bounds() {
        let b = (Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
        let c = corners(b, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        assert_eq!(c.len(), 8);
        assert!(c.iter().all(|p| (4.0..=6.0).contains(&p.x)));
        let items = [PlanItem { model: 0, mesh: 0, material: 0, transform: Mat4::IDENTITY }, PlanItem { model: 0, mesh: 7, material: 0, transform: Mat4::IDENTITY }];
        let pts = plan_points(&items, &[vec![Some(b)]]);
        assert_eq!(pts.len(), 8, "an item whose mesh has no bounds adds no points");
    }

    #[test]
    fn home_orbit_frames_the_bounds_centre_from_a_three_quarter_view() {
        let pts = corners((Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.8, 0.5)), Mat4::IDENTITY);
        let o = home_orbit(&pts, 1.5).unwrap();
        assert!((o.target - Vec3::new(0.0, 0.9, 0.0)).length() < 1e-4);
        assert_eq!((o.yaw, o.pitch), (HOME_YAW, HOME_PITCH));
        // Every corner projects inside the view with the framing's camera.
        let (view, proj, _) = o.camera(Mat4::IDENTITY, 1.5);
        for p in pts {
            let c = proj * view * p.extend(1.0);
            let (x, y) = (c.x / c.w, c.y / c.w);
            assert!(x.abs() < 1.0 && y.abs() < 1.0 && c.w > 0.0, "corner {p} off screen: {x},{y}");
        }
        assert!(home_orbit(&[], 1.0).is_none());
    }

    #[test]
    fn home_orbit_fills_the_view_instead_of_the_bounding_sphere() {
        // A flat wide slab: the sphere framing would leave it tiny.
        let pts = corners((Vec3::new(-2.0, -0.1, -0.1), Vec3::new(2.0, 0.1, 0.1)), Mat4::IDENTITY);
        let sphere = model_preview::frame_points(&pts, HOME_YAW, HOME_PITCH, 1.6).unwrap();
        let o = home_orbit(&pts, 1.6).unwrap();
        assert!(o.dist < sphere.dist, "tighter than the sphere: {} vs {}", o.dist, sphere.dist);
        let (view, proj, _) = o.camera(Mat4::IDENTITY, 1.6);
        let ndc: Vec<(f32, f32)> = pts.iter().map(|p| { let c = proj * view * p.extend(1.0); (c.x / c.w, c.y / c.w) }).collect();
        let widest = ndc.iter().map(|(x, y)| x.abs().max(y.abs())).fold(0.0, f32::max);
        assert!(widest <= 1.0 && widest > 0.8, "fills 80-100 % of the view, not {widest}");
    }

    #[test]
    fn settle_needs_the_same_view_for_several_frames_and_restarts_on_a_move() {
        let a = Mat4::from_translation(Vec3::X);
        let mut s = Settle::default();
        let fired: Vec<bool> = (0..6).map(|_| s.step(a)).collect();
        assert_eq!(fired, [false, false, false, true, true, true], "the first frame sets the camera, then SETTLE_FRAMES identical ones");
        assert!(!s.step(Mat4::from_translation(Vec3::Y)), "a move restarts the count");
    }

    #[test]
    fn cache_keys_change_with_the_file_stamp() {
        let dir = std::env::temp_dir().join(format!("zest-ap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("m.glb");
        std::fs::write(&f, b"a").unwrap();
        let src = PreviewSource::Model(f.clone());
        let k1 = CacheKey::of(&src);
        assert_eq!(k1, CacheKey::of(&src));
        std::fs::write(f.with_extension("glb.meta"), b"x").unwrap();
        assert_ne!(k1, CacheKey::of(&src), "a new .meta changes the key");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn draw_cmd_carries_the_mesh_bounds_for_culling() {
        let d = draw_cmd(MeshHandle::NONE, MaterialHandle::NONE, Mat4::IDENTITY, Some((Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0))));
        assert_eq!(d.mesh_center, Vec3::new(1.0, 0.0, 0.0));
        assert!((d.bound_radius - 1.0).abs() < 1e-6);
        assert_eq!((d.lightmap_layer, d.planar_index, d.highlight), (-1, -1, 0.0));
    }
}

/// The pane on a real `Renderer` (hidden window): load, upload, frame, render, settle, read back
/// into the cache, then hide and free. Needs a display; skipped without one or with
/// `ZESTENGINE_TEST_GPU=0`. Saves the cached image to `docs/design/asset-preview-helmet.png`.
#[cfg(test)]
mod gpu {
    use super::*;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
    use winit::platform::x11::EventLoopBuilderExtX11;
    use winit::window::{Window, WindowId};
    use zestengine_render::{CameraData, FrameInput, LightData, MAX_REFLECTION_PROBES};

    fn frame_input<'a>() -> FrameInput<'a> {
        let camera = CameraData { view: Mat4::IDENTITY, proj: Mat4::IDENTITY, position: Vec3::ZERO };
        FrameInput {
            clear_color: [0.0; 4],
            camera,
            light: LightData::default(),
            lights: &[],
            capture_lights: &[],
            voxel_emitters: &[],
            camera_preview: None,
            model_preview: None,
            draw_skybox: false,
            shadow_pcf_texel: 0.0,
            shadow_bias_scale: 0.0,
            shadow_normal_offset: 0.0,
            shadow_distance: 60.0,
            shadow_cascades: 0,
            shadow_cascade_lambda: 0.0,
            time: 0.0,
            water_time: 0.0,
            water: &[],
            draws: &[],
            decals: &[],
            lightmap_preview: false,
            gi_debug: 0,
            render_viewport: false,
            viewport_extent: None,
            postfx: PostFx::default(),
            preview_postfx: PostFx::default(),
            reflection_probe: None,
            reflection_probes: [None; MAX_REFLECTION_PROBES],
            reflection_probe_b: None,
            reflection_primary: 0,
            fog: None,
            voxel_specular: false,
            planar: &[],
            camera_targets: &[],
            egui: None,
            world_ui: Vec::new(),
        }
    }

    #[derive(Default)]
    struct Probe {
        done: bool,
        error: Option<String>,
        /// (width, height, rgba) of the cached image.
        image: Option<(u32, u32, Vec<u8>)>,
        frames_to_cache: usize,
        live_after_cache: bool,
        freed_when_hidden: bool,
        cache_hit_reloads_nothing: bool,
        cache_entries: usize,
    }

    fn run(p: &mut Probe, event_loop: &ActiveEventLoop) -> anyhow::Result<()> {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(320.0, 240.0));
        let window = std::sync::Arc::new(event_loop.create_window(attrs)?);
        let mut r = Renderer::new(window)?;
        r.set_vsync(false);
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = root.join("models/DamagedHelmet.glb");
        let source = PreviewSource::Model(model.clone());
        let mut pane = AssetPreview::default();
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        crate::theme::install_fonts(&mut fonts);
        ctx.set_fonts(fonts);
        crate::theme::apply(&ctx);
        // Fonts are applied at the start of the next pass.
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let step = |pane: &mut AssetPreview, r: &mut Renderer, show: Option<&PreviewSource>| -> anyhow::Result<()> {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0))), ..Default::default() };
            pane.set_live_texture(r.model_preview_texture());
            let _ = ctx.run(raw, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    if let Some(s) = show {
                        pane.ui(ui, s);
                    }
                });
            });
            pane.frame(r, true);
            r.render(&frame_input())
        };
        // Visible: the model loads on a worker, then renders until it settles into the cache.
        for n in 0..400 {
            step(&mut pane, &mut r, Some(&source))?;
            if pane.cached().is_some() {
                p.frames_to_cache = n;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let Some(c) = pane.cached() else { anyhow::bail!("no cached image after the loop (stage failed: {})", matches!(pane.stage, Stage::Failed(_))) };
        let (w, h) = (c.size[0], c.size[1]);
        p.cache_entries = pane.cache.len();
        // After the cache exists the renderer's target is released and nothing renders.
        for _ in 0..3 {
            step(&mut pane, &mut r, Some(&source))?;
        }
        p.live_after_cache = pane.live;
        p.cache_hit_reloads_nothing = matches!(pane.stage, Stage::Ready(_));
        // Hidden past KEEP_HIDDEN: resources are freed with no ui at all.
        pane.last_shown -= KEEP_HIDDEN + Duration::from_secs(1);
        step(&mut pane, &mut r, None)?;
        p.freed_when_hidden = matches!(pane.stage, Stage::Idle);
        // Pixels for the self-check: the cached ColorImage lives in egui; re-render through the
        // low-level path to get bytes (same draws, same camera as the cache).
        let orbit = pane.cached().map(|c| c.orbit).unwrap();
        let loaded = load(&source).map_err(|e| anyhow::anyhow!(e))?;
        let gpu = upload(&mut r, loaded, false)?;
        let mut draws = Vec::new();
        gpu.draws(&mut draws);
        let (view, proj, position) = orbit.camera(Mat4::IDENTITY, w as f32 / h as f32);
        let postfx = PostFx { bloom_enabled: false, grading_enabled: false, vignette_enabled: false, ca_enabled: false, ..PostFx::default() };
        for _ in 0..6 {
            r.set_model_only_preview(Some(&ModelOnlyPreview {
                camera: CameraData { view, proj, position },
                size: [w, h],
                draws: &draws,
                background: BACKGROUND,
                light: zestengine_render::neutral_light(),
                postfx,
                want_depth: false,
            }));
            r.render(&frame_input())?;
        }
        p.image = r.debug_read_model_preview()?;
        gpu.release(&mut r);
        Ok(())
    }

    impl ApplicationHandler for Probe {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if !self.done {
                self.done = true;
                if let Err(e) = run(self, event_loop) {
                    self.error = Some(format!("{e:#}"));
                }
                event_loop.exit();
            }
        }
        fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
    }

    #[test]
    fn the_pane_loads_renders_settles_into_the_cache_and_frees_when_hidden() {
        if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
            eprintln!("[gpu test] no display / GPU disabled: skipped");
            return;
        }
        let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
            eprintln!("[gpu test] no X11 event loop: skipped");
            return;
        };
        let mut p = Probe::default();
        event_loop.run_app_on_demand(&mut p).unwrap();
        if let Some(e) = &p.error {
            panic!("{e}");
        }
        assert_eq!(p.cache_entries, 1, "one cached render");
        assert!(!p.live_after_cache, "no model-only request once the image is cached");
        assert!(p.cache_hit_reloads_nothing, "the loaded model stays until it has been idle");
        assert!(p.freed_when_hidden, "hidden past KEEP_HIDDEN frees the uploaded resources");
        let (w, h, px) = p.image.clone().expect("a readback of the preview target");
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/asset-preview-helmet.png");
        image::save_buffer(&out, &px, w, h, image::ColorType::Rgba8).expect("saving the capture");
        // Not blank: model pixels differ from the background and cover a sensible share.
        let bg = [px[0], px[1], px[2]];
        let differs = px.as_chunks::<4>().0.iter().filter(|c| c[..3].iter().zip(&bg).any(|(a, b)| (*a as i32 - *b as i32).abs() > 12)).count();
        let share = differs as f32 / (w * h) as f32;
        eprintln!("asset preview: {w}x{h}, model covers {:.0}% of the view, cached after {} frames", share * 100.0, p.frames_to_cache);
        assert!((0.08..0.9).contains(&share), "the model covers {share:.2} of the view");
        // Lit: more than one brightness level inside the model (not a flat silhouette).
        let lum: std::collections::HashSet<u8> = px.as_chunks::<4>().0.iter().filter(|c| c[..3].iter().zip(&bg).any(|(a, b)| (*a as i32 - *b as i32).abs() > 12)).map(|c| c[0] / 16).collect();
        assert!(lum.len() >= 4, "the model looks flat: {} brightness steps", lum.len());
    }
}
