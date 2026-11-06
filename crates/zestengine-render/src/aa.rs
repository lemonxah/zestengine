//! Anti-aliasing: temporal AA (TAA) and the renderer-side mode plumbing.
//!
//! * FXAA lives in the post pass (`post.frag`, `PostPass::set_aa`); it needs no
//!   targets of its own and is the default mode.
//! * TAA (`shaders/taa.frag`) resolves the HDR scene (after the SSR / RT / fog
//!   resolve, before bloom + post) against an RGBA16F history that ping-pongs per
//!   camera path (game, editor viewport). The scene pass is jittered with a
//!   Halton(2, 3) sequence and writes motion vectors + a reactive mask; the
//!   velocity target is cleared to a sentinel so pixels no pipeline wrote (sky,
//!   sprites, text) fall back to depth-based camera reprojection in the shader.
//!   The resolve writes the next history image, which is also the source set the
//!   post pass samples (so overlays drawn in or after post stay sharp).
//! * FSR3 (`Upscaler::Fsr3`) and TAA are exclusive: with the upscaler on, TAA
//!   does not run. The editor viewport has no FSR, so `AntiAliasing::Fsr` runs
//!   TAA there.
//! * MSAA 2x/4x is not available on the desktop renderer (see Features.md); the
//!   modes render as FXAA.

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use gpu_allocator::vulkan::Allocator;

use crate::frame::FRAMES_IN_FLIGHT;
use crate::screen_pass::{self, ScreenImage, read_info, sampled};

pub use crate::types::AntiAliasing;

const TAA_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/taa.frag.spv"));

/// History / output format: rgb = resolved colour, a = linear depth.
pub(crate) const HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Velocity value no pipeline wrote (the attachment is cleared to it while TAA
/// runs): the resolve reprojects those pixels from depth.
pub(crate) const VELOCITY_SENTINEL: f32 = 32768.0;
/// Weight of the new frame in the temporal blend (about 10 frames of history).
pub(crate) const TAA_BLEND: f32 = 0.1;
/// Variance-clip box size in standard deviations.
pub(crate) const CLIP_GAMMA: f32 = 1.25;
/// Halton jitter period (frames).
const JITTER_PERIOD: u32 = 16;

pub(crate) const TARGET_GAME: usize = 0;
pub(crate) const TARGET_VIEWPORT: usize = 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaPush {
    reproj: [[f32; 4]; 4],
    jit: [f32; 4],
    lin: [f32; 4],
}

/// What one TAA resolve needs from the frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AaFrame {
    /// This frame's view-projection as rendered (jittered).
    pub view_proj: Mat4,
    /// The previous frame's, as it was rendered (jittered).
    pub prev_view_proj: Mat4,
    /// The projection the depth was rendered with (only its depth row is used).
    pub proj: Mat4,
    /// This frame's jitter minus the previous frame's, in NDC.
    pub jitter_delta: [f32; 2],
    /// Weight of the new frame (0.1 normally).
    pub blend: f32,
    /// No usable history (first frame, cut, resize, mode change).
    pub reset: bool,
}

/// The scene images one resolve reads. All SHADER_READ_ONLY.
#[derive(Clone, Copy)]
pub(crate) struct AaInputs {
    pub color: vk::ImageView,
    pub depth: vk::ImageView,
    pub velocity: vk::ImageView,
    pub reactive: vk::ImageView,
}

/// Halton(2, 3) jitter sample for TAA frame `frame`, in [0, 1)^2 (0.5 = none).
pub(crate) fn jitter_for(frame: u32) -> [f32; 2] {
    crate::halton23(frame % JITTER_PERIOD + 1)
}

/// The NDC shift `jitter_proj` applies for `jitter` at `render_size`.
pub(crate) fn jitter_ndc(jitter: [f32; 2], render_size: [f32; 2]) -> [f32; 2] {
    [(jitter[0] - 0.5) * 2.0 / render_size[0].max(1.0), (jitter[1] - 0.5) * 2.0 / render_size[1].max(1.0)]
}

/// Shared pipeline, layout and samplers.
pub(crate) struct AaPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    point: vk::Sampler,
    linear: vk::Sampler,
}

impl AaPass {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let set_layout = screen_pass::samplers_layout(device, 5)?;
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(std::mem::size_of::<TaaPush>() as u32)];
        let layouts = [set_layout];
        let layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None)?
        };
        let pipeline = screen_pass::fullscreen_pipeline(device, layout, TAA_SPV, HISTORY_FORMAT, &[])?;
        Ok(Self {
            set_layout,
            layout,
            pipeline,
            point: screen_pass::clamp_sampler(device, false)?,
            linear: screen_pass::clamp_sampler(device, true)?,
        })
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.point, None);
            device.destroy_sampler(self.linear, None);
        }
    }

    /// Resolve `inputs` against `target`'s history. `slot` is the frame-in-flight
    /// index (selects the descriptor set, which must not be in use). Record
    /// outside a render pass. Returns the post-pass source set holding the
    /// resolved image (SHADER_READ_ONLY on return, readable by fragment and
    /// compute stages).
    pub fn record(&self, device: &ash::Device, cb: vk::CommandBuffer, target: &mut AaTarget, slot: usize, inputs: &AaInputs, f: &AaFrame) -> vk::DescriptorSet {
        target.init(device, cb);
        let cur = target.parity;
        let prev = cur ^ 1;
        let set = target.sets[slot];
        {
            let color = read_info(self.point, inputs.color);
            let hist = read_info(self.linear, target.hist[prev].view);
            let depth = read_info(self.point, inputs.depth);
            let vel = read_info(self.point, inputs.velocity);
            let react = read_info(self.point, inputs.reactive);
            let writes = [sampled(set, 0, &color), sampled(set, 1, &hist), sampled(set, 2, &depth), sampled(set, 3, &vel), sampled(set, 4, &react)];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let out = &target.hist[cur];
        crate::image_barrier(
            device, cb, out.image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        let reproj = f.prev_view_proj * f.view_proj.inverse();
        let ortho = crate::is_ortho_proj(&f.proj);
        let push = TaaPush {
            reproj: reproj.to_cols_array_2d(),
            jit: [f.jitter_delta[0], f.jitter_delta[1], f.blend, if f.reset || !target.history_valid { 1.0 } else { 0.0 }],
            lin: [f.proj.z_axis.z, f.proj.w_axis.z, if ortho { 1.0 } else { 0.0 }, CLIP_GAMMA],
        };
        screen_pass::draw_fullscreen(device, cb, out.view, target.extent, self.pipeline, self.layout, &[set], bytemuck::bytes_of(&push));
        crate::image_barrier(
            device, cb, out.image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        target.parity = prev;
        target.history_valid = true;
        target.post_sets[cur]
    }
}

/// One camera path's TAA images (the game path and the editor viewport each own
/// one, sized to their scene extent).
pub(crate) struct AaTarget {
    pub extent: vk::Extent2D,
    hist: [ScreenImage; 2],
    /// The viewport path's own motion vectors + reactive mask (the game path
    /// reuses the renderer's per-frame targets).
    pub motion: Option<(ScreenImage, ScreenImage)>,
    pool: vk::DescriptorPool,
    /// Resolve input sets, one per frame in flight.
    sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    /// Post-pass source sets, one per history image.
    post_sets: [vk::DescriptorSet; 2],
    /// Which history image the next resolve writes.
    parity: usize,
    history_valid: bool,
    initialized: bool,
}

impl AaTarget {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        pass: &AaPass,
        post_layout: vk::DescriptorSetLayout,
        post_sampler: vk::Sampler,
        extent: vk::Extent2D,
        with_motion: bool,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width: extent.width.max(2), height: extent.height.max(2) };
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        let hist = [
            ScreenImage::new(device, allocator, HISTORY_FORMAT, usage, extent, 1, "taa history 0")?,
            ScreenImage::new(device, allocator, HISTORY_FORMAT, usage, extent, 1, "taa history 1")?,
        ];
        let rt = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let motion = if with_motion {
            Some((
                ScreenImage::new(device, allocator, crate::pipeline::VELOCITY_FORMAT, rt, extent, 1, "taa motion vectors")?,
                ScreenImage::new(device, allocator, crate::pipeline::REACTIVE_FORMAT, rt, extent, 1, "taa reactive mask")?,
            ))
        } else {
            None
        };
        let n_sets = (FRAMES_IN_FLIGHT + 2) as u32;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(n_sets).pool_sizes(&[vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 5 * FRAMES_IN_FLIGHT as u32 + 2,
                }]),
                None,
            )?
        };
        let mut layouts = vec![pass.set_layout; FRAMES_IN_FLIGHT];
        layouts.extend([post_layout; 2]);
        let all = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let mut sets = [vk::DescriptorSet::null(); FRAMES_IN_FLIGHT];
        sets.copy_from_slice(&all[..FRAMES_IN_FLIGHT]);
        let post_sets = [all[FRAMES_IN_FLIGHT], all[FRAMES_IN_FLIGHT + 1]];
        for (i, &ps) in post_sets.iter().enumerate() {
            let info = read_info(post_sampler, hist[i].view);
            unsafe { device.update_descriptor_sets(&[sampled(ps, 0, &info)], &[]) };
        }
        Ok(Self { extent, hist, motion, pool, sets, post_sets, parity: 0, history_valid: false, initialized: false })
    }

    /// First use: the history images are in their read layout (the resolve
    /// binds the previous one even when it ignores it).
    fn init(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        for img in [self.hist[0].image, self.hist[1].image] {
            crate::image_barrier(
                device, cb, img, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
            );
            unsafe {
                device.cmd_clear_color_image(cb, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
            }
            crate::image_barrier(
                device, cb, img, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
        }
    }

    /// Forget the temporal history (camera cut, resize, mode change).
    pub fn reset_history(&mut self) {
        self.history_valid = false;
    }

    /// The image the last resolve wrote (SHADER_READ_ONLY once recorded).
    #[cfg(test)]
    pub fn last_output(&self) -> &ScreenImage {
        &self.hist[self.parity ^ 1]
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for h in &mut self.hist {
            h.destroy(device, allocator);
        }
        if let Some((mut v, mut r)) = self.motion.take() {
            v.destroy(device, allocator);
            r.destroy(device, allocator);
        }
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// This frame's decision, made before the jitter is chosen.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AaPlan {
    /// FXAA runs inside the post pass of the main camera.
    pub fxaa: bool,
    /// TAA runs for this path (`TARGET_GAME` / `TARGET_VIEWPORT`).
    pub taa: Option<usize>,
    /// CAS-like sharpen amount for the main camera's post pass (TAA / FSR).
    pub sharpen: f32,
}

/// The renderer's AA state.
#[derive(Default)]
pub(crate) struct AaState {
    /// The requested mode (Project Settings / camera override).
    pub mode: AntiAliasing,
    /// Output sharpening for TAA / FSR, 0..1.
    pub sharpen: f32,
    pub pass: Option<AaPass>,
    pub targets: [Option<AaTarget>; 2],
    retired: Vec<(AaTarget, usize)>,
    pub plan: AaPlan,
    /// TAA frames since the last reset (Halton index).
    pub frame: u32,
    /// The previous TAA frame's jitter (NDC); zero while TAA is idle.
    pub prev_jitter_ndc: [f32; 2],
    /// Camera of the previous TAA frame, for cut detection: (position, forward).
    last_camera: Option<([f32; 3], [f32; 3])>,
}

impl AaState {
    /// A fresh state running `mode`.
    pub fn new(mode: AntiAliasing) -> Self {
        Self { mode, ..Default::default() }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for t in self.targets.iter_mut().filter_map(Option::take) {
            let mut t = t;
            t.destroy(device, allocator);
        }
        for (mut t, _) in self.retired.drain(..) {
            t.destroy(device, allocator);
        }
        if let Some(mut p) = self.pass.take() {
            p.destroy(device);
        }
    }

    /// Free targets retired at least FRAMES_IN_FLIGHT frames ago.
    fn tick_retired(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        let mut i = 0;
        while i < self.retired.len() {
            if self.retired[i].1 == 0 {
                let (mut t, _) = self.retired.swap_remove(i);
                t.destroy(device, allocator);
            } else {
                self.retired[i].1 -= 1;
                i += 1;
            }
        }
    }

    fn reset_all(&mut self) {
        for t in self.targets.iter_mut().flatten() {
            t.reset_history();
        }
        self.last_camera = None;
    }
}

/// The mode a path actually runs: MSAA falls back to FXAA, and FSR runs TAA in
/// the editor viewport (no upscaler there).
pub(crate) fn effective_mode(mode: AntiAliasing, viewport: bool) -> AntiAliasing {
    match mode {
        AntiAliasing::Msaa2 | AntiAliasing::Msaa4 => AntiAliasing::Fxaa,
        AntiAliasing::Fsr if viewport => AntiAliasing::Taa,
        m => m,
    }
}

impl crate::Renderer {
    /// Set the anti-aliasing mode (Project Settings; a camera may override it).
    /// `Fsr` turns the FSR3 upscaler on (Quality preset unless it already is);
    /// leaving it, or choosing TAA, turns the upscaler off (they are exclusive).
    /// Cheap and idempotent: call it every frame.
    pub fn set_anti_aliasing(&mut self, mode: AntiAliasing) {
        if self.aa.mode == mode {
            return;
        }
        let was = self.aa.mode;
        self.aa.mode = mode;
        self.aa.reset_all();
        if mode == AntiAliasing::Fsr {
            if self.upscaler() == crate::Upscaler::Off {
                self.set_upscale_quality(crate::UpscaleQuality::Quality);
            }
        } else if (was == AntiAliasing::Fsr || mode == AntiAliasing::Taa) && self.upscaler() != crate::Upscaler::Off {
            self.set_upscale_quality(crate::UpscaleQuality::Native);
        }
    }

    pub fn anti_aliasing(&self) -> AntiAliasing {
        self.aa.mode
    }

    /// TAA / FSR output sharpening (CAS-like, 0..1; 0 = off).
    pub fn set_taa_sharpen(&mut self, amount: f32) {
        self.aa.sharpen = amount.clamp(0.0, 1.0);
    }

    /// Drop the temporal history (e.g. after teleporting the camera).
    pub fn reset_anti_aliasing_history(&mut self) {
        self.aa.reset_all();
    }

    /// Decide this frame's AA before the jitter is chosen: which path runs TAA,
    /// whether FXAA runs, create / resize the TAA target, and detect camera cuts.
    pub(crate) fn aa_prepare(&mut self, input: &crate::FrameInput, window: bool) {
        self.aa.plan = AaPlan::default();
        let device = self.ctx.device.clone();
        let allocator = self.allocator();
        {
            let mut a = allocator.lock().unwrap();
            self.aa.tick_retired(&device, &mut a);
        }
        let viewport = input.viewport_extent.is_some() && self.viewport_target.is_some();
        let game = input.render_viewport && window && !viewport;
        let mode = effective_mode(self.aa.mode, viewport);
        let upscaled = self.upscaler() != crate::Upscaler::Off;
        // The game path's selection outline wipes the scene depth: no TAA then.
        let outline_in_scene = game && input.draws.iter().any(|d| d.highlight > 0.0);
        let taa_path = match mode {
            AntiAliasing::Taa if viewport => Some((TARGET_VIEWPORT, self.viewport_target.as_ref().map_or(vk::Extent2D::default(), |t| t.extent))),
            AntiAliasing::Taa if game && !upscaled && !outline_in_scene => {
                Some((TARGET_GAME, vk::Extent2D { width: self.render_extent[0], height: self.render_extent[1] }))
            }
            _ => None,
        };
        self.aa.plan.fxaa = mode == AntiAliasing::Fxaa && (viewport || game);
        self.aa.plan.sharpen = if (viewport || game) && (taa_path.is_some() || (mode == AntiAliasing::Fsr && upscaled)) { self.aa.sharpen } else { 0.0 };
        let Some((target, extent)) = taa_path else {
            // Idle: the next TAA frame starts from scratch.
            self.aa.prev_jitter_ndc = [0.0; 2];
            self.aa.last_camera = None;
            for t in self.aa.targets.iter_mut().flatten() {
                t.reset_history();
            }
            return;
        };
        let res: Result<()> = (|| {
            if self.aa.pass.is_none() {
                self.aa.pass = Some(AaPass::new(&device)?);
            }
            let stale = self.aa.targets[target].as_ref().is_some_and(|t| t.extent != extent);
            if stale && let Some(old) = self.aa.targets[target].take() {
                self.aa.retired.push((old, FRAMES_IN_FLIGHT + 1));
                self.aa.last_camera = None;
            }
            if self.aa.targets[target].is_none() {
                let mut a = allocator.lock().unwrap();
                self.aa.targets[target] = Some(AaTarget::new(
                    &device,
                    &mut a,
                    self.aa.pass.as_ref().unwrap(),
                    self.post_pass.set_layout(),
                    self.post_pass.sampler(),
                    extent,
                    target == TARGET_VIEWPORT,
                )?);
            }
            Ok(())
        })();
        if let Err(e) = res {
            tracing::warn!("TAA targets: {e:#}");
            return;
        }
        // Camera cut: a teleport or a near-reversal of the view direction.
        let inv = input.camera.view.inverse();
        let cam = (inv.w_axis.truncate().to_array(), (-inv.z_axis.truncate()).normalize_or_zero().to_array());
        if let Some((p, f)) = self.aa.last_camera {
            let moved = (glam::Vec3::from(cam.0) - glam::Vec3::from(p)).length();
            let turned = glam::Vec3::from(cam.1).dot(glam::Vec3::from(f));
            if moved > 50.0 || turned < 0.2 {
                self.aa.targets[target].as_mut().unwrap().reset_history();
            }
        }
        self.aa.last_camera = Some(cam);
        self.aa.plan.taa = Some(target);
    }

    /// Turn FXAA / sharpen on for the main camera's post pass recorded next
    /// (camera previews and VR eyes never get them); pair with [`Self::aa_post_end`].
    pub(crate) fn aa_post_begin(&mut self) {
        self.post_pass.set_aa(self.aa.plan.fxaa, self.aa.plan.sharpen);
    }

    pub(crate) fn aa_post_end(&mut self) {
        self.post_pass.set_aa(false, 0.0);
    }

    /// The jitter for this frame: Halton while TAA runs (the upscaler picks its
    /// own), none otherwise.
    pub(crate) fn aa_jitter(&mut self) -> Option<[f32; 2]> {
        self.aa.plan.taa.map(|_| {
            self.aa.frame = self.aa.frame.wrapping_add(1);
            jitter_for(self.aa.frame)
        })
    }

    /// Record the TAA resolve for `target` if [`Self::aa_prepare`] planned it.
    /// `ubo` is this frame's UBO (jittered matrices); returns the post source
    /// set to use instead of the unresolved one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_taa(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        target: usize,
        ubo: &crate::FrameUbo,
        inputs: AaInputs,
    ) -> Option<vk::DescriptorSet> {
        if self.aa.plan.taa != Some(target) || self.aa.pass.is_none() || self.aa.targets[target].is_none() {
            return None;
        }
        let size = [self.render_extent[0] as f32, self.render_extent[1] as f32];
        let cur = jitter_ndc(self.jitter, size);
        let prev = self.aa.prev_jitter_ndc;
        self.aa.prev_jitter_ndc = cur;
        let f = AaFrame {
            view_proj: Mat4::from_cols_array_2d(&ubo.view_proj),
            prev_view_proj: Mat4::from_cols_array_2d(&ubo.prev_view_proj),
            proj: Mat4::from_cols_array_2d(&ubo.proj),
            jitter_delta: [cur[0] - prev[0], cur[1] - prev[1]],
            blend: TAA_BLEND,
            reset: false,
        };
        self.ctx.checkpoint(cb, c"taa");
        self.gpu_zone_begin(cb, crate::gpu_zone::AA);
        let slot = self.frame_index;
        let set = match (self.aa.pass.as_ref(), self.aa.targets[target].as_mut()) {
            (Some(pass), Some(t)) => Some(pass.record(device, cb, t, slot, &inputs, &f)),
            _ => None,
        };
        self.gpu_zone_end(cb, crate::gpu_zone::AA);
        set
    }
}
