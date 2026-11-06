//! Offscreen renders of a list of draws that do not belong to the scene frame:
//!
//! * the MODEL-ONLY preview ([`ModelOnlyPreview`], `Renderer::set_model_only_preview`):
//!   a given list of draws with its own camera, a neutral light set, no scene sky
//!   and a background colour, into the model-preview target; optionally with the
//!   depth buffer read back ([`ModelPreviewDepth`]) so overlay dots can be dimmed
//!   where the model hides them;
//! * the textured IMPOSTOR capture ([`ImpostorCaptureRequest`],
//!   `Renderer::start_impostor_capture`): every view of the octahedral atlas drawn
//!   with the engine's own pipelines (materials and textures as imported), the
//!   whole atlas in ONE pass per layer (a viewport per tile), albedo through the
//!   shader's unlit-albedo mode and normals through its world-normal mode, read
//!   back and packed into the `.zimp` layers. It runs inside the frame loop, one
//!   pass per rendered frame, into a target of its own (two frames of GPU work
//!   for an 8 x 8 x 64 atlas), so nothing blocks the main thread.
//!
//! The view maths here mirrors `zestengine_assets::impostor` (`cell_dir`,
//! `view_basis`, `dir_to_oct`); a test in the assets crate keeps them equal.

use glam::{Mat4, Vec3};

use anyhow::{Result, bail};
use ash::vk;

use crate::alloc::Buffer;
use crate::frame::FRAMES_IN_FLIGHT;
use crate::{
    CameraData, DrawCmd, FrameInput, GpuLight, LightData, LightInstance, LightKind, MAX_LIGHTS, MAX_REFLECTION_PROBES, MAX_SHADOW_VIEWS,
    PostFx, Renderer, frame_ubo, gpu_light, image_barrier,
};

/// A model-only preview request (set once per frame, like `FrameInput::model_preview`).
#[derive(Clone, Copy)]
pub struct ModelOnlyPreview<'a> {
    pub camera: CameraData,
    /// Pixel size of the target (re-created when it changes).
    pub size: [u32; 2],
    /// Exactly the draws rendered: mesh handles (already uploaded) and world transforms.
    pub draws: &'a [DrawCmd],
    /// Linear clear colour behind the model (alpha 0 = a transparent background in the texture).
    pub background: [f32; 4],
    /// Light set; see [`neutral_light`].
    pub light: LightData,
    pub postfx: PostFx,
    /// Also read the depth buffer back (a copy per frame, available a couple of frames later).
    pub want_depth: bool,
}

/// A neutral studio light: a soft key from up-left-front and a flat ambient, no sky reflections.
pub fn neutral_light() -> LightData {
    LightData { direction: Vec3::new(-0.35, -0.8, -0.5).normalize(), color: [1.0, 1.0, 1.0], intensity: 2.2, ambient: [0.42, 0.42, 0.45], env_intensity: 0.0 }
}

/// Bytes per texel of a depth-aspect-only copy of the scene depth.
fn depth_texel_bytes() -> usize {
    if crate::depth_format() == vk::Format::D16_UNORM_S8_UINT { 2 } else { 4 }
}

/// Decode a depth-aspect buffer copy of the scene depth to 0..1 floats: `D32` is
/// raw f32; `D24` is a 24-bit unorm in the low bits of a u32; `D16` a u16 unorm.
fn decode_depth(bytes: &[u8], texels: usize) -> Vec<f32> {
    match crate::depth_format() {
        vk::Format::D24_UNORM_S8_UINT => bytes[..texels * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| (u32::from_le_bytes([c[0], c[1], c[2], 0]) as f32) / 16_777_215.0)
            .collect(),
        vk::Format::D16_UNORM_S8_UINT => bytes[..texels * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65_535.0)
            .collect(),
        _ => bytemuck::pod_collect_to_vec(&bytes[..texels * 4]),
    }
}

/// The model preview's depth buffer (`D32`, 0 near .. 1 far, 1 = nothing drawn), row 0 at the top.
#[derive(Clone, Debug, Default)]
pub struct ModelPreviewDepth {
    pub size: [u32; 2],
    pub depth: Vec<f32>,
}

impl ModelPreviewDepth {
    /// Depth at pixel `(x, y)` (None outside).
    pub fn at(&self, x: i32, y: i32) -> Option<f32> {
        (x >= 0 && y >= 0 && (x as u32) < self.size[0] && (y as u32) < self.size[1]).then(|| self.depth[y as usize * self.size[0] as usize + x as usize])
    }

    /// Whether `world` is hidden by the model as seen from `camera` (the nearest surface at its
    /// pixel is nearer than the point by more than `bias` in depth units). A point outside the
    /// view is not occluded; so is one over empty background.
    pub fn occludes(&self, camera: &CameraData, world: Vec3, bias: f32) -> bool {
        let clip = camera.proj * camera.view * world.extend(1.0);
        if clip.w <= 0.0 {
            return false;
        }
        let ndc = clip.truncate() / clip.w;
        let (x, y) = ((ndc.x * 0.5 + 0.5) * self.size[0] as f32, (0.5 - ndc.y * 0.5) * self.size[1] as f32);
        self.at(x as i32, y as i32).is_some_and(|d| d < 1.0 && d + bias < ndc.z)
    }
}

// ---------------------------------------------------------------- impostor capture

/// A textured impostor capture: the model's bind pose as ready-to-draw meshes.
#[derive(Clone)]
pub struct ImpostorCaptureRequest {
    /// Views per side and texels per view side.
    pub grid: u32,
    pub tile: u32,
    /// Bounding sphere every view frames (model space).
    pub centre: [f32; 3],
    pub radius: f32,
    /// Bind-pose draws in the model's space: uploaded mesh handles + the material to draw them with.
    pub draws: Vec<DrawCmd>,
}

/// The two layers a capture produced (`(grid * tile)^2` RGBA8 texels, views row by row).
#[derive(Clone, Debug, PartialEq)]
pub struct CapturedImpostor {
    pub albedo: Vec<u8>,
    pub aux: Vec<u8>,
}

/// Names a queued capture (`Renderer::take_impostor_capture`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CaptureTicket(pub u64);

/// Largest atlas side in texels a capture target is made for.
pub const MAX_CAPTURE_SIDE: u32 = 4096;

/// Octahedral encoding of a unit direction into `[-1, 1]^2`.
pub fn dir_to_oct(d: [f32; 3]) -> [f32; 2] {
    let l1 = d[0].abs() + d[1].abs() + d[2].abs();
    let (mut x, mut y) = (d[0] / l1, d[1] / l1);
    if d[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    [x, y]
}

/// Inverse of [`dir_to_oct`].
pub fn oct_to_dir(e: [f32; 2]) -> [f32; 3] {
    let mut n = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
    if n[2] < 0.0 {
        let (ox, oy) = (n[0], n[1]);
        n[0] = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        n[1] = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
    [n[0] / l, n[1] / l, n[2] / l]
}

/// Direction (model -> camera) of atlas cell `(i, j)`.
pub fn cell_dir(grid: u32, i: u32, j: u32) -> [f32; 3] {
    oct_to_dir([(i as f32 + 0.5) / grid as f32 * 2.0 - 1.0, (j as f32 + 0.5) / grid as f32 * 2.0 - 1.0])
}

/// Right and up of the view looking at the model from `dir`.
pub fn view_basis(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let d = Vec3::from(dir).normalize_or_zero();
    let hint = if d.y.abs() > 0.999 { Vec3::Z } else { Vec3::Y };
    let right = hint.cross(d).normalize_or_zero();
    let up = d.cross(right);
    (right.to_array(), up.to_array())
}

/// Model-space -> view-space matrix of cell `(i, j)`: x right, y up, the camera
/// `radius` from the centre along the cell direction looking at it, so view
/// distance `0..2 radius` spans the sphere front to back.
pub fn view_matrix(grid: u32, i: u32, j: u32, centre: [f32; 3], radius: f32) -> Mat4 {
    let dir = Vec3::from(cell_dir(grid, i, j));
    let (_, up) = view_basis(dir.to_array());
    let c = Vec3::from(centre);
    Mat4::look_at_rh(c + dir * radius, c, Vec3::from(up))
}

/// The orthographic projection every tile shares.
pub fn tile_projection(radius: f32) -> Mat4 {
    Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, 2.0 * radius)
}

/// Pack a capture's two float readbacks into the `.zimp` layers.
///
/// `albedo_rgba`: the albedo pass's linear RGBA32F texels (alpha = coverage), `normal_rgba`: the
/// world-normal pass (`n * 0.5 + 0.5` in VIEW space, alpha = coverage), `depth`: the normal
/// pass's depth (0 at the eye .. 1 at `2 radius`, 1 = empty). Atlas layout: `grid * tile` square,
/// views row by row.
pub fn pack_atlas(grid: u32, tile: u32, albedo_rgba: &[f32], normal_rgba: &[f32], depth: &[f32]) -> CapturedImpostor {
    let size = (grid * tile) as usize;
    assert_eq!(albedo_rgba.len(), size * size * 4);
    assert_eq!(normal_rgba.len(), size * size * 4);
    assert_eq!(depth.len(), size * size);
    let mut albedo = vec![0u8; size * size * 4];
    let mut aux = vec![0u8; size * size * 4];
    let q8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    for j in 0..grid {
        for i in 0..grid {
            let dir = Vec3::from(cell_dir(grid, i, j));
            let (right, up) = view_basis(dir.to_array());
            let (right, up) = (Vec3::from(right), Vec3::from(up));
            for ty in 0..tile {
                for tx in 0..tile {
                    let texel = ((j * tile + ty) as usize) * size + (i * tile + tx) as usize;
                    let at = texel * 4;
                    if albedo_rgba[at + 3] > 0.5 {
                        albedo[at] = q8(albedo_rgba[at]);
                        albedo[at + 1] = q8(albedo_rgba[at + 1]);
                        albedo[at + 2] = q8(albedo_rgba[at + 2]);
                        albedo[at + 3] = 255;
                    }
                    if normal_rgba[at + 3] > 0.5 && depth[texel] < 1.0 {
                        let nv = Vec3::new(normal_rgba[at], normal_rgba[at + 1], normal_rgba[at + 2]) * 2.0 - Vec3::ONE;
                        let mut n = (right * nv.x + up * nv.y + dir * nv.z).normalize_or_zero();
                        if n == Vec3::ZERO {
                            n = dir;
                        }
                        let o = dir_to_oct(n.to_array());
                        aux[at] = q8(o[0] * 0.5 + 0.5);
                        aux[at + 1] = q8(o[1] * 0.5 + 0.5);
                        aux[at + 2] = q8(1.0 - depth[texel]);
                        aux[at + 3] = 255;
                    }
                }
            }
        }
    }
    CapturedImpostor { albedo, aux }
}

/// Which capture layer a pass renders (the shader's `gi_debug` mode for it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapturePass {
    /// Unlit albedo (`gi_debug` 3).
    Albedo,
    /// World normals (`gi_debug` 1) and depth.
    Normal,
}

impl CapturePass {
    pub(crate) fn gi_debug(self) -> u32 {
        match self {
            Self::Albedo => 3,
            Self::Normal => 1,
        }
    }
}

/// A capture being worked on (one pass per rendered frame).
pub(crate) struct CaptureJob {
    pub id: u64,
    pub grid: u32,
    pub tile: u32,
    /// `grid^2 * draws` view-space draws: view `k` owns `k * per_view .. (k + 1) * per_view`.
    pub draws: Vec<DrawCmd>,
    pub per_view: usize,
    pub radius: f32,
    pub next: Option<CapturePass>,
    pub albedo: Option<Vec<f32>>,
}

impl CaptureJob {
    pub fn new(id: u64, req: ImpostorCaptureRequest) -> Self {
        let per_view = req.draws.len();
        let mut draws = Vec::with_capacity(per_view * (req.grid * req.grid) as usize);
        for j in 0..req.grid {
            for i in 0..req.grid {
                let v = view_matrix(req.grid, i, j, req.centre, req.radius);
                draws.extend(req.draws.iter().map(|d| DrawCmd {
                    transform: v * d.transform,
                    bound_radius: 0.0,
                    highlight: 0.0,
                    planar_index: -1,
                    lightmap_layer: -1,
                    occluded: false,
                    ..*d
                }));
            }
        }
        Self { id, grid: req.grid, tile: req.tile, draws, per_view, radius: req.radius, next: Some(CapturePass::Albedo), albedo: None }
    }
}

/// What a finished device readback was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadbackKind {
    ModelDepth,
    CaptureAlbedo(u64),
    CaptureNormal(u64),
}

/// Decode `n` RGBA16F texels (little-endian f16) to floats.
pub(crate) fn rgba16f_to_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    bytes[..n * 8].as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

/// The model-only preview as the renderer keeps it between `set_model_only_preview` and the frame.
pub(crate) struct ModelOnlyState {
    camera: CameraData,
    size: [u32; 2],
    draws: Vec<DrawCmd>,
    background: [f32; 4],
    light: LightData,
    postfx: PostFx,
    want_depth: bool,
}

/// A device->host copy waiting for its frame slot's fence.
pub(crate) struct PendingReadback {
    slot: usize,
    kind: ReadbackKind,
    w: u32,
    h: u32,
    hdr: Option<Buffer>,
    depth: Option<Buffer>,
}

impl PendingReadback {
    pub(crate) fn destroy(&mut self, device: &ash::Device, alloc: &mut gpu_allocator::vulkan::Allocator) {
        if let Some(b) = self.hdr.as_mut() {
            b.destroy(device, alloc);
        }
        if let Some(b) = self.depth.as_mut() {
            b.destroy(device, alloc);
        }
    }
}

/// Post settings that leave the HDR pass untouched (the capture reads the HDR texels anyway).
fn capture_postfx() -> PostFx {
    PostFx { tonemap: 0, bloom_enabled: false, grading_enabled: false, vignette_enabled: false, ca_enabled: false, ..PostFx::default() }
}

/// A frame input for an offscreen pass: the given draws and light, no scene around them.
fn synthetic_input<'a>(camera: CameraData, draws: &'a [DrawCmd], light: LightData, postfx: PostFx, gi_debug: u32) -> FrameInput<'a> {
    FrameInput {
        clear_color: [0.0; 4],
        camera,
        light,
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
        draws,
        decals: &[],
        lightmap_preview: false,
        gi_debug,
        render_viewport: false,
        viewport_extent: None,
        postfx,
        preview_postfx: postfx,
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

impl Renderer {
    /// Ask for a MODEL-ONLY preview this frame: `req.draws` seen by `req.camera` over
    /// `req.background`, lit by `req.light` alone (no scene lights, sky, probes, GI or
    /// shadows), into the model-preview target (`model_preview_texture`, replaces
    /// `FrameInput::model_preview` while set). Call it every frame it should show;
    /// `None` (or not calling) lets the target go. The draws' meshes and materials must be
    /// uploaded; they need no instance in the scene. With `want_depth` the depth buffer is
    /// copied back each frame ([`Self::model_preview_depth`], two frames late).
    pub fn set_model_only_preview(&mut self, req: Option<&ModelOnlyPreview<'_>>) {
        let Some(r) = req else {
            self.model_only = None;
            return;
        };
        let mut draws = self.model_only.take().map(|m| m.draws).unwrap_or_else(|| std::mem::take(&mut self.model_only_spare));
        draws.clear();
        draws.extend_from_slice(r.draws);
        self.model_only = Some(ModelOnlyState {
            camera: r.camera,
            size: r.size,
            draws,
            background: r.background,
            light: r.light,
            postfx: r.postfx,
            want_depth: r.want_depth,
        });
        if !r.want_depth {
            self.model_only_depth = None;
        }
    }

    /// The model-only preview's depth buffer as of a couple of frames ago (None until the first
    /// copy lands, or without `want_depth`).
    pub fn model_preview_depth(&self) -> Option<&ModelPreviewDepth> {
        self.model_only_depth.as_ref()
    }

    /// Queue a textured impostor capture; it advances one pass per rendered frame (see the module
    /// docs) and is collected with [`Self::take_impostor_capture`]. The request's meshes and
    /// materials must stay alive until it is taken.
    pub fn start_impostor_capture(&mut self, req: ImpostorCaptureRequest) -> Result<CaptureTicket> {
        let side = req.grid.saturating_mul(req.tile);
        if !(1..=32).contains(&req.grid) || !(4..=512).contains(&req.tile) || side > MAX_CAPTURE_SIDE {
            bail!("impostor capture grid {} / tile {} out of range", req.grid, req.tile);
        }
        if req.draws.is_empty() || !(req.radius.is_finite() && req.radius > 0.0) || !req.centre.iter().all(|c| c.is_finite()) {
            bail!("impostor capture needs draws and finite bounds");
        }
        let id = self.capture_next_id;
        self.capture_next_id += 1;
        self.capture_jobs.push_back(CaptureJob::new(id, req));
        Ok(CaptureTicket(id))
    }

    /// The result of a queued capture once it is done (`None` while it is still running).
    pub fn take_impostor_capture(&mut self, ticket: CaptureTicket) -> Option<Result<CapturedImpostor, String>> {
        let at = self.capture_done.iter().position(|(id, _)| *id == ticket.0)?;
        Some(self.capture_done.swap_remove(at).1)
    }

    /// Captures queued or running.
    pub fn impostor_captures_pending(&self) -> usize {
        self.capture_jobs.len()
    }

    fn make_preview(&mut self, extent: vk::Extent2D) -> Result<crate::CameraPreview> {
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let mut preview = crate::CameraPreview::new(
            &self.ctx,
            &mut alloc,
            self.pipeline_cache.set0_layout,
            self.swapchain.format.format,
            self.sampler,
            self.shadow_atlas.array_view,
            self.shadow_atlas.sampler,
            self.probe_buffer.handle,
            self.lightmaps.view,
            self.lightmap_sampler,
            self.reflection_env.view,
            self.reflection_env_b.view,
            self.reflection_env_c.view,
            self.reflection_env_d.view,
            self.ripple_dummy.view,
            self.default_textures[3].view,
            self.default_textures[crate::ao::AO_DEFAULT_TEXTURE].view,
            self.post_pass.set_layout(),
            self.post_pass.sampler(),
            self.egui.as_mut().unwrap(),
            extent,
        )?;
        preview.bloom = Some(self.post_pass.new_bloom_chain(&self.ctx.device, &mut alloc, preview.extent)?);
        Ok(preview)
    }

    /// The frame UBO of an offscreen pass: one directional light from `light`, nothing of the scene.
    fn neutral_ubo(&self, camera: &CameraData, input: &FrameInput, extent: vk::Extent2D) -> crate::FrameUbo {
        let mut lights = [GpuLight::default(); MAX_LIGHTS];
        lights[0] = gpu_light(&LightInstance {
            kind: LightKind::Directional,
            position: glam::Vec3::ZERO,
            direction: input.light.direction,
            color: input.light.color,
            intensity: input.light.intensity,
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: false,
            soft_shadows: false, radius: 0.0,
            shadow_bias: 0.0,
            baked: false,
        });
        let mut ubo = frame_ubo(
            camera,
            input,
            lights,
            1,
            [[[0.0; 4]; 4]; MAX_SHADOW_VIEWS],
            [0.0; 4],
            &[],
            [0.5; 2],
            [1.0, 1.0],
            [extent.width as f32, extent.height as f32],
            glam::Mat4::IDENTITY.to_cols_array_2d(),
        );
        // Linear HDR out; the preview's post pass resolves it.
        ubo.debug[3] = 1.0;
        ubo
    }

    /// Create / resize the offscreen targets and write this frame's UBOs. Runs before the pass loop.
    pub(crate) fn prepare_offscreen(&mut self, model_only: Option<&ModelOnlyState>) -> Result<()> {
        if let Some(m) = model_only {
            let extent = vk::Extent2D { width: m.size[0].clamp(64, 2048), height: m.size[1].clamp(64, 2048) };
            if self.model_preview.as_ref().is_some_and(|p| p.extent != extent)
                && let Some(old) = self.model_preview.take()
            {
                self.retired_model_previews.push((old, FRAMES_IN_FLIGHT + 1));
            }
            if self.model_preview.is_none() {
                self.model_preview = Some(self.make_preview(extent)?);
            }
            let input = synthetic_input(m.camera, &[], m.light, m.postfx, 0);
            let ubo = self.neutral_ubo(&m.camera, &input, extent);
            let fi = self.frame_index;
            self.model_preview.as_mut().unwrap().ubos[fi].write(0, bytemuck::bytes_of(&ubo));
        }
        // The front capture, when a pass is due.
        let due = self.capture_jobs.front().and_then(|j| j.next.map(|p| (j.id, p, j.grid * j.tile, j.radius)));
        if let Some((id, pass, side, radius)) = due {
            let extent = vk::Extent2D { width: side, height: side };
            let made: Result<()> = (|| {
                if self.capture_target.as_ref().is_some_and(|p| p.extent != extent)
                    && let Some(old) = self.capture_target.take()
                {
                    self.retired_model_previews.push((old, FRAMES_IN_FLIGHT + 1));
                }
                if self.capture_target.is_none() {
                    self.capture_target = Some(self.make_preview(extent)?);
                }
                let camera = CameraData { view: glam::Mat4::IDENTITY, proj: tile_projection(radius), position: glam::Vec3::ZERO };
                let input = synthetic_input(camera, &[], neutral_light(), capture_postfx(), pass.gi_debug());
                let ubo = self.neutral_ubo(&camera, &input, extent);
                let fi = self.frame_index;
                self.capture_target.as_mut().unwrap().ubos[fi].write(0, bytemuck::bytes_of(&ubo));
                Ok(())
            })();
            if let Err(e) = made {
                self.fail_capture(id, format!("capture target: {e:#}"));
            }
        } else if self.capture_jobs.is_empty()
            && let Some(old) = self.capture_target.take()
        {
            self.retired_model_previews.push((old, FRAMES_IN_FLIGHT + 1));
        }
        Ok(())
    }

    fn fail_capture(&mut self, id: u64, why: String) {
        self.capture_jobs.retain(|j| j.id != id);
        self.capture_done.push((id, Err(why)));
    }

    /// Record the model-only preview and the due impostor capture pass into the frame's command buffer.
    pub(crate) fn record_offscreen(&mut self, device: &ash::Device, cb: vk::CommandBuffer, model_only: Option<ModelOnlyState>) -> Result<()> {
        if let Some(mut m) = model_only {
            let mut order = std::mem::take(&mut self.model_only_order);
            order.clear();
            order.extend(0..m.draws.len());
            let input = synthetic_input(m.camera, &m.draws, m.light, m.postfx, 0);
            std::mem::swap(&mut self.camera_preview, &mut self.model_preview);
            self.preview_keep_depth = m.want_depth;
            let mut res = self.record_camera_preview(device, cb, &order, &input, &m.camera, false, m.background, None);
            self.preview_keep_depth = false;
            if res.is_ok() && m.want_depth {
                res = self.record_readback(device, cb, ReadbackKind::ModelDepth, false, true);
            }
            std::mem::swap(&mut self.camera_preview, &mut self.model_preview);
            drop(input);
            self.model_only_order = order;
            self.model_only_spare = std::mem::take(&mut m.draws);
            res?;
        }
        if self.capture_jobs.front().is_some_and(|j| j.next.is_some()) && self.capture_target.is_some() {
            let mut job = self.capture_jobs.pop_front().unwrap();
            let pass = job.next.take().unwrap();
            let camera = CameraData { view: glam::Mat4::IDENTITY, proj: tile_projection(job.radius), position: glam::Vec3::ZERO };
            let input = synthetic_input(camera, &job.draws, neutral_light(), capture_postfx(), pass.gi_debug());
            let order: Vec<usize> = (0..job.draws.len()).collect();
            std::mem::swap(&mut self.camera_preview, &mut self.capture_target);
            self.preview_keep_depth = pass == CapturePass::Normal;
            let mut res = self.record_camera_preview(device, cb, &order, &input, &camera, false, [0.0; 4], Some((job.grid, job.tile, job.per_view)));
            self.preview_keep_depth = false;
            if res.is_ok() {
                let kind = if pass == CapturePass::Albedo { ReadbackKind::CaptureAlbedo(job.id) } else { ReadbackKind::CaptureNormal(job.id) };
                res = self.record_readback(device, cb, kind, true, pass == CapturePass::Normal);
            }
            std::mem::swap(&mut self.camera_preview, &mut self.capture_target);
            match res {
                Ok(()) => self.capture_jobs.push_front(job),
                Err(e) => self.capture_done.push((job.id, Err(format!("capture pass: {e:#}")))),
            }
        }
        Ok(())
    }

    /// Copy the HDR colour and / or depth of the preview target currently in `camera_preview` to
    /// host buffers (complete when this frame slot's fence is next waited).
    fn record_readback(&mut self, device: &ash::Device, cb: vk::CommandBuffer, kind: ReadbackKind, hdr: bool, depth: bool) -> Result<()> {
        let p = self.camera_preview.as_ref().expect("preview target swapped in");
        let (w, h, hdr_image, depth_image) = (p.extent.width, p.extent.height, p.hdr_image, p.depth);
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let mut rb = PendingReadback { slot: self.frame_index, kind, w, h, hdr: None, depth: None };
        let texels = w as u64 * h as u64;
        if hdr {
            rb.hdr = Some(Buffer::new(device, &mut alloc, texels * 8, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "offscreen hdr readback")?);
        }
        if depth {
            match Buffer::new(device, &mut alloc, texels * depth_texel_bytes() as u64, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "offscreen depth readback") {
                Ok(b) => rb.depth = Some(b),
                Err(e) => {
                    rb.destroy(device, &mut alloc);
                    return Err(e);
                }
            }
        }
        let region = |aspect: vk::ImageAspectFlags| {
            vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1))
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 })
        };
        unsafe {
            if let Some(b) = &rb.hdr {
                image_barrier(
                    device, cb, hdr_image, vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
                );
                device.cmd_copy_image_to_buffer(cb, hdr_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, b.handle, &[region(vk::ImageAspectFlags::COLOR)]);
            }
            if let Some(b) = &rb.depth {
                image_barrier(
                    device, cb, depth_image, crate::depth_aspect(),
                    crate::depth_attach_layout(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
                );
                device.cmd_copy_image_to_buffer(cb, depth_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, b.handle, &[region(vk::ImageAspectFlags::DEPTH)]);
            }
            // Make the host reads below see the copies once the fence signals.
            let host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&host));
        }
        self.readbacks.push(rb);
        Ok(())
    }

    /// Consume the readbacks recorded in the frame slot whose fence was just waited.
    pub(crate) fn complete_readbacks(&mut self) {
        if self.readbacks.is_empty() {
            return;
        }
        let slot = self.frame_index;
        let Some(allocator) = self.allocator.clone() else { return };
        let mut i = 0;
        while i < self.readbacks.len() {
            if self.readbacks[i].slot != slot {
                i += 1;
                continue;
            }
            let mut rb = self.readbacks.swap_remove(i);
            let texels = rb.w as usize * rb.h as usize;
            let hdr = rb.hdr.as_ref().and_then(|b| b.mapped()).map(|m| rgba16f_to_f32(m, texels));
            let depth: Option<Vec<f32>> = rb.depth.as_ref().and_then(|b| b.mapped()).map(|m| decode_depth(m, texels));
            rb.destroy(&self.ctx.device, &mut allocator.lock().unwrap());
            match rb.kind {
                ReadbackKind::ModelDepth => {
                    if let Some(depth) = depth {
                        self.model_only_depth = Some(ModelPreviewDepth { size: [rb.w, rb.h], depth });
                    }
                }
                ReadbackKind::CaptureAlbedo(id) => match (hdr, self.capture_jobs.iter_mut().find(|j| j.id == id)) {
                    (Some(a), Some(job)) => {
                        job.albedo = Some(a);
                        job.next = Some(CapturePass::Normal);
                    }
                    _ => self.fail_capture(id, "albedo readback missing".into()),
                },
                ReadbackKind::CaptureNormal(id) => {
                    let albedo = self.capture_jobs.iter_mut().find(|j| j.id == id).and_then(|j| j.albedo.take().map(|a| (j.grid, j.tile, a)));
                    match (hdr, depth, albedo) {
                        (Some(n), Some(d), Some((grid, tile, a))) => {
                            let packed = pack_atlas(grid, tile, &a, &n, &d);
                            self.capture_jobs.retain(|j| j.id != id);
                            self.capture_done.push((id, Ok(packed)));
                        }
                        _ => self.fail_capture(id, "normal readback missing".into()),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octahedral_roundtrip_and_cells_are_unit() {
        for j in 0..8 {
            for i in 0..8 {
                let d = Vec3::from(cell_dir(8, i, j));
                assert!((d.length() - 1.0).abs() < 1e-5);
                let back = Vec3::from(oct_to_dir(dir_to_oct(d.to_array())));
                assert!((back - d).length() < 1e-4, "{d:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn view_matrix_looks_down_the_cell_direction_with_basis_axes() {
        let (centre, r) = ([1.0, 2.0, 3.0], 2.0);
        for (i, j) in [(0, 0), (3, 4), (7, 7), (2, 6)] {
            let v = view_matrix(8, i, j, centre, r);
            let dir = Vec3::from(cell_dir(8, i, j));
            let (right, up) = view_basis(dir.to_array());
            // The centre sits straight ahead at distance `r`.
            let c = v.transform_point3(Vec3::from(centre));
            assert!(c.truncate().length() < 1e-4 && (c.z + r).abs() < 1e-4, "{c:?}");
            // View x / y are the basis right / up (the CPU rasteriser's frame).
            let x = v.transform_vector3(Vec3::from(right));
            let y = v.transform_vector3(Vec3::from(up));
            assert!((x - Vec3::X).length() < 1e-4 && (y - Vec3::Y).length() < 1e-4, "{x:?} {y:?}");
        }
    }

    #[test]
    fn packing_turns_view_normals_back_into_model_space_and_flips_depth() {
        let (grid, tile) = (2u32, 4u32);
        let size = (grid * tile) as usize;
        let mut albedo = vec![0.0f32; size * size * 4];
        let mut normal = vec![0.0f32; size * size * 4];
        let mut depth = vec![1.0f32; size * size];
        // Cell (1, 0), texel (1, 2): covered, view normal = +Z (towards the camera).
        let texel = 2 * size + 5;
        albedo[texel * 4..texel * 4 + 4].copy_from_slice(&[0.5, 0.25, 1.0, 1.0]);
        normal[texel * 4..texel * 4 + 4].copy_from_slice(&[0.5, 0.5, 1.0, 1.0]);
        depth[texel] = 0.25;
        let out = pack_atlas(grid, tile, &albedo, &normal, &depth);
        assert_eq!(&out.albedo[texel * 4..texel * 4 + 4], &[128, 64, 255, 255]);
        // Facing the camera: the stored model-space normal is the cell direction.
        let dir = Vec3::from(cell_dir(grid, 1, 0));
        let o = [out.aux[texel * 4] as f32 / 255.0 * 2.0 - 1.0, out.aux[texel * 4 + 1] as f32 / 255.0 * 2.0 - 1.0];
        assert!((Vec3::from(oct_to_dir(o)) - dir).length() < 0.02);
        assert_eq!(out.aux[texel * 4 + 2], (0.75f32 * 255.0 + 0.5) as u8);
        assert_eq!(out.aux[texel * 4 + 3], 255);
        // Everything else is empty.
        assert_eq!(out.albedo.iter().filter(|&&b| b != 0).count(), 4);
    }

    #[test]
    fn depth_occlusion_test_projects_and_compares() {
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 10.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let camera = CameraData { view, proj, position: Vec3::new(0.0, 0.0, 5.0) };
        let ndc_z = |p: Vec3| {
            let c = proj * view * p.extend(1.0);
            c.z / c.w
        };
        let mut d = ModelPreviewDepth { size: [8, 8], depth: vec![1.0; 64] };
        // A surface at z = 1 fills the middle; a point behind it (z = -1) is hidden, one in front (z = 2) is not.
        for y in 2..6 {
            for x in 2..6 {
                d.depth[y * 8 + x] = ndc_z(Vec3::new(0.0, 0.0, 1.0));
            }
        }
        assert!(d.occludes(&camera, Vec3::new(0.0, 0.0, -1.0), 1e-4));
        assert!(!d.occludes(&camera, Vec3::new(0.0, 0.0, 2.0), 1e-4));
        // Over empty background nothing hides it; a point off screen is never hidden.
        assert!(!d.occludes(&camera, Vec3::new(2.0, 0.0, -1.0), 1e-4));
        assert!(!d.occludes(&camera, Vec3::new(40.0, 0.0, -1.0), 1e-4));
    }
}
