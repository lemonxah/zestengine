//! Screen-space ambient occlusion.
//!
//! Runs after the camera's depth prepass and before the forward pass, which
//! multiplies ONLY its indirect terms by the result (indirect diffuse with an
//! optional multi-bounce colour fit, plus specular occlusion on the
//! environment reflection; direct lights are untouched). See `standard.frag`
//! (`u_ssao`, set 0 binding 13; enabled per pass by `frame.refl_blend2.w`).
//!
//! Passes (fullscreen fragment, `shaders/ao_*.frag`), all at half resolution
//! except the last:
//!   1. prep: depth buffer -> linear depth (closest of 2x2), then a 4-level
//!      pyramid (rotated-grid subsample) so wide kernels stay cache-friendly;
//!   2. main: [`AoQuality::Low`] = Scalable Ambient Obscurance (spiral taps
//!      over the pyramid), [`AoQuality::High`] = GTAO (horizon search over 2
//!      slices x 2 sides x 6 steps, analytic cosine-weighted visibility);
//!   3. denoise: 4x4 depth-aware box (one period of the 4x4 rotation pattern)
//!      plus, for High, temporal accumulation reprojected by the camera with
//!      depth-based history rejection and neighbourhood clamping;
//!   4. upsample: joint-bilateral 2x upsample to the full-resolution R8
//!      target the forward pass samples per pixel.
//!
//! Normals are reconstructed from depth (the G-buffer is written by the
//! forward pass itself, after AO). Bent normals are not produced. The XR
//! (multiview) pass does not use AO: its eye UBOs clear the enable flag.

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use gpu_allocator::vulkan::Allocator;

use crate::frame::FRAMES_IN_FLIGHT;
use crate::screen_pass::{self, ScreenImage, read_info, sampled};

const PREP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ao_prep.frag.spv"));
const MAIN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ao_main.frag.spv"));
const DENOISE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ao_denoise.frag.spv"));
const UPSAMPLE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ao_upsample.frag.spv"));

const LIN_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
const RAW_FORMAT: vk::Format = vk::Format::R8_UNORM;
const HIST_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
/// Format of the full-resolution result the forward pass samples.
pub(crate) const AO_FORMAT: vk::Format = vk::Format::R8_UNORM;
/// Depth pyramid levels (half res and three coarser).
const MAX_MIPS: u32 = 4;
/// Descriptor sets per frame slot: prep + (MAX_MIPS - 1) downsamples + main + denoise + upsample.
const SETS_PER_SLOT: usize = MAX_MIPS as usize + 3;
/// `Renderer::default_textures` entry bound to binding 13 while AO is off: the
/// 1x1 white (linear) opacity default, i.e. "no occlusion".
pub(crate) const AO_DEFAULT_TEXTURE: usize = 4;
/// Weight of the new frame in the temporal blend (about 10 frames of history).
const TEMPORAL_BLEND: f32 = 0.1;

pub use crate::types::{AoQuality, AoSettings, VIEW_AO};

/// What one AO dispatch needs from the frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AoFrame {
    /// The (jittered) projection the depth was rendered with.
    pub proj: Mat4,
    pub view: Mat4,
    pub settings: AoSettings,
    pub quality: AoQuality,
    pub frame: u32,
}

impl AoFrame {
    fn view_proj(&self) -> Mat4 {
        self.proj * self.view
    }
}

fn ortho(proj: &Mat4) -> bool {
    crate::is_ortho_proj(proj)
}

/// (P22, P32, ortho, mode).
fn lin_params(proj: &Mat4, mode: f32) -> [f32; 4] {
    [proj.z_axis.z, proj.w_axis.z, if ortho(proj) { 1.0 } else { 0.0 }, mode]
}

/// (P00, P11, offx, offy) for `ao_view_pos`.
fn view_params(proj: &Mat4) -> [f32; 4] {
    if ortho(proj) {
        [proj.x_axis.x, proj.y_axis.y, -proj.w_axis.x, -proj.w_axis.y]
    } else {
        [proj.x_axis.x, proj.y_axis.y, proj.z_axis.x, proj.z_axis.y]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrepPush {
    lin: [f32; 4],
    size: [i32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MainPush {
    view: [f32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenoisePush {
    reproj: [[f32; 4]; 4],
    view: [f32; 4],
    p0: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpsamplePush {
    lin: [f32; 4],
    size: [f32; 4],
}

/// Shared pipelines, layout and samplers.
pub(crate) struct AoPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    prep: vk::Pipeline,
    main_low: vk::Pipeline,
    main_high: vk::Pipeline,
    denoise: vk::Pipeline,
    upsample: vk::Pipeline,
    point: vk::Sampler,
    linear: vk::Sampler,
}

impl AoPass {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let set_layout = screen_pass::samplers_layout(device, 3)?;
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(128)];
        let layouts = [set_layout];
        let layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None)?
        };
        let prep = screen_pass::fullscreen_pipeline(device, layout, PREP_SPV, LIN_FORMAT, &[])?;
        let main_low = screen_pass::fullscreen_pipeline(device, layout, MAIN_SPV, RAW_FORMAT, &[(0, 0)])?;
        let main_high = screen_pass::fullscreen_pipeline(device, layout, MAIN_SPV, RAW_FORMAT, &[(0, 1)])?;
        let denoise = screen_pass::fullscreen_pipeline(device, layout, DENOISE_SPV, HIST_FORMAT, &[])?;
        let upsample = screen_pass::fullscreen_pipeline(device, layout, UPSAMPLE_SPV, AO_FORMAT, &[])?;
        Ok(Self {
            set_layout,
            layout,
            prep,
            main_low,
            main_high,
            denoise,
            upsample,
            point: screen_pass::clamp_sampler(device, false)?,
            linear: screen_pass::clamp_sampler(device, true)?,
        })
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for p in [self.prep, self.main_low, self.main_high, self.denoise, self.upsample] {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.point, None);
            device.destroy_sampler(self.linear, None);
        }
    }

    /// Record AO for `target` from `depth_view` (the full-resolution depth,
    /// SHADER_READ_ONLY, its writes made visible to fragment reads). `slot` is
    /// the frame-in-flight index (selects the descriptor sets, which must not
    /// be in use). On return the full-resolution result is SHADER_READ_ONLY for
    /// the forward pass's fragment shader.
    pub fn record(&self, device: &ash::Device, cb: vk::CommandBuffer, target: &mut AoTarget, slot: usize, depth_view: vk::ImageView, f: &AoFrame) {
        target.init(device, cb);
        let sets = target.sets[slot];
        let cur = target.parity;
        let prev = cur ^ 1;
        let mips = target.lin.mips;

        // Descriptors (this slot's sets are idle: its previous submission retired).
        {
            let depth = read_info(self.point, depth_view);
            let lin_levels: Vec<[vk::DescriptorImageInfo; 1]> = (0..mips).map(|l| read_info(self.point, target.lin.level(l))).collect();
            let lin_all = read_info(self.point, target.lin.view);
            let raw = read_info(self.point, target.raw.view);
            let hist_prev = read_info(self.linear, target.hist[prev].view);
            let hist_cur = read_info(self.point, target.hist[cur].view);
            let mut writes = vec![sampled(sets[0], 0, &depth)];
            for l in 1..mips {
                writes.push(sampled(sets[l as usize], 0, &lin_levels[l as usize - 1]));
            }
            let m = MAX_MIPS as usize;
            writes.push(sampled(sets[m], 0, &lin_all));
            writes.push(sampled(sets[m + 1], 0, &raw));
            writes.push(sampled(sets[m + 1], 1, &lin_levels[0]));
            writes.push(sampled(sets[m + 1], 2, &hist_prev));
            writes.push(sampled(sets[m + 2], 0, &hist_cur));
            writes.push(sampled(sets[m + 2], 1, &depth));
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        let to_target = |img: vk::Image, base: u32| {
            crate::image_barrier_mips(
                device, cb, img, vk::ImageAspectFlags::COLOR, base, 1,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
        };
        let to_read = |img: vk::Image, base: u32| {
            crate::image_barrier_mips(
                device, cb, img, vk::ImageAspectFlags::COLOR, base, 1,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
        };

        // 1. Linear depth pyramid.
        let full = target.extent;
        for l in 0..mips {
            let (src_size, mode) = if l == 0 { (full, 0.0) } else { (target.lin.level_extent(l - 1), 1.0) };
            let push = PrepPush { lin: lin_params(&f.proj, mode), size: [src_size.width as i32, src_size.height as i32, l as i32, 0] };
            to_target(target.lin.image, l);
            screen_pass::draw_fullscreen(device, cb, target.lin.level(l), target.lin.level_extent(l), self.prep, self.layout, &[sets[l as usize]], bytemuck::bytes_of(&push));
            to_read(target.lin.image, l);
        }

        // 2. AO.
        let half = target.half;
        let ortho = ortho(&f.proj);
        let s = f.settings;
        let temporal = f.quality == AoQuality::High;
        let flags = (if ortho { 1 } else { 0 }) | (if temporal { 2 } else { 0 });
        let push = MainPush {
            view: view_params(&f.proj),
            p0: [s.radius.max(0.01), s.intensity.max(0.0), s.falloff.clamp(0.05, 1.0), 0.5 * half.height as f32 * f.proj.y_axis.y.abs()],
            p1: [half.width as f32, half.height as f32, (f.frame % 1024) as f32, flags as f32],
            p2: [(0.25 * half.height as f32).max(32.0), mips as f32, 0.0, 0.0],
        };
        let main = if f.quality == AoQuality::Low { self.main_low } else { self.main_high };
        to_target(target.raw.image, 0);
        screen_pass::draw_fullscreen(device, cb, target.raw.view, half, main, self.layout, &[sets[MAX_MIPS as usize]], bytemuck::bytes_of(&push));
        to_read(target.raw.image, 0);

        // 3. Denoise (+ temporal) into this frame's history slot.
        let vp = f.view_proj();
        let history = temporal && target.history_valid && target.last_view_proj.is_some();
        let prev_vp = target.last_view_proj.unwrap_or(vp);
        let reproj = prev_vp * f.view.inverse();
        let dflags = flags | if history { 4 } else { 0 };
        let push = DenoisePush {
            reproj: reproj.to_cols_array_2d(),
            view: view_params(&f.proj),
            p0: [half.width as f32, half.height as f32, dflags as f32, TEMPORAL_BLEND],
        };
        to_target(target.hist[cur].image, 0);
        screen_pass::draw_fullscreen(device, cb, target.hist[cur].view, half, self.denoise, self.layout, &[sets[MAX_MIPS as usize + 1]], bytemuck::bytes_of(&push));
        to_read(target.hist[cur].image, 0);

        // 4. Full-resolution upsample. The previous frame's forward pass read
        //    this image: wait for its fragment reads before the layout change.
        let push = UpsamplePush {
            lin: lin_params(&f.proj, 0.0),
            size: [half.width as f32, half.height as f32, full.width as f32, full.height as f32],
        };
        crate::image_barrier(
            device, cb, target.full.image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        screen_pass::draw_fullscreen(device, cb, target.full.view, full, self.upsample, self.layout, &[sets[MAX_MIPS as usize + 2]], bytemuck::bytes_of(&push));
        to_read(target.full.image, 0);

        target.parity = prev;
        target.history_valid = true;
        target.last_view_proj = Some(vp);
    }
}

/// One camera target's AO images (the game path and the editor viewport each
/// own one, sized to their scene extent).
pub(crate) struct AoTarget {
    pub extent: vk::Extent2D,
    half: vk::Extent2D,
    lin: ScreenImage,
    raw: ScreenImage,
    hist: [ScreenImage; 2],
    /// Full-resolution visibility, sampled by the forward pass.
    pub full: ScreenImage,
    pool: vk::DescriptorPool,
    sets: [[vk::DescriptorSet; SETS_PER_SLOT]; FRAMES_IN_FLIGHT],
    parity: usize,
    history_valid: bool,
    last_view_proj: Option<Mat4>,
    initialized: bool,
}

impl AoTarget {
    pub fn new(device: &ash::Device, allocator: &mut Allocator, pass: &AoPass, extent: vk::Extent2D) -> Result<Self> {
        let extent = vk::Extent2D { width: extent.width.max(2), height: extent.height.max(2) };
        let half = vk::Extent2D { width: extent.width.div_ceil(2), height: extent.height.div_ceil(2) };
        let mips = (32 - half.width.min(half.height).leading_zeros()).clamp(1, MAX_MIPS);
        let rt = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let lin = ScreenImage::new(device, allocator, LIN_FORMAT, rt, half, mips, "ao linear depth")?;
        let raw = ScreenImage::new(device, allocator, RAW_FORMAT, rt, half, 1, "ao raw")?;
        let hist = [
            ScreenImage::new(device, allocator, HIST_FORMAT, rt | vk::ImageUsageFlags::TRANSFER_DST, half, 1, "ao history 0")?,
            ScreenImage::new(device, allocator, HIST_FORMAT, rt | vk::ImageUsageFlags::TRANSFER_DST, half, 1, "ao history 1")?,
        ];
        let full = ScreenImage::new(device, allocator, AO_FORMAT, rt | vk::ImageUsageFlags::TRANSFER_DST, extent, 1, "ao full")?;
        let n = (SETS_PER_SLOT * FRAMES_IN_FLIGHT) as u32;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n)
                    .pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 3 * n }]),
                None,
            )?
        };
        let layouts = vec![pass.set_layout; n as usize];
        let all = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let mut sets = [[vk::DescriptorSet::null(); SETS_PER_SLOT]; FRAMES_IN_FLIGHT];
        for (i, s) in all.into_iter().enumerate() {
            sets[i / SETS_PER_SLOT][i % SETS_PER_SLOT] = s;
        }
        Ok(Self { extent, half, lin, raw, hist, full, pool, sets, parity: 0, history_valid: false, last_view_proj: None, initialized: false })
    }

    /// First use: the result reads as "no occlusion" and the history images
    /// are in their read layout before anything samples them.
    fn init(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        for img in [self.full.image, self.hist[0].image, self.hist[1].image] {
            crate::image_barrier(
                device, cb, img, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
            );
            unsafe {
                device.cmd_clear_color_image(
                    cb, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] },
                    &[range],
                );
            }
            crate::image_barrier(
                device, cb, img, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
        }
    }

    /// Forget the temporal history (camera cut, settings change).
    pub fn reset_history(&mut self) {
        self.history_valid = false;
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.lin.destroy(device, allocator);
        self.raw.destroy(device, allocator);
        for h in &mut self.hist {
            h.destroy(device, allocator);
        }
        self.full.destroy(device, allocator);
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// Which AO target a frame uses.
pub(crate) const TARGET_GAME: usize = 0;
pub(crate) const TARGET_VIEWPORT: usize = 1;

/// This frame's AO decision (made before the frame UBO is written).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AoPlan {
    pub target: usize,
    pub frame: AoFrame,
}

/// The renderer's AO state: pass, per-path targets, retired targets waiting
/// for the frames that used them.
#[derive(Default)]
pub(crate) struct AoState {
    pub pass: Option<AoPass>,
    pub targets: [Option<AoTarget>; 2],
    retired: Vec<(AoTarget, usize)>,
    pub quality: AoQuality,
    pub frame: u32,
    pub plan: Option<AoPlan>,
    /// The last frame's quality / settings, to reset history on a change.
    last: Option<(AoQuality, AoSettings)>,
}

impl AoState {
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
}

impl crate::Renderer {
    /// Set the screen-space AO algorithm (Project Settings > Rendering).
    pub fn set_ao_quality(&mut self, quality: AoQuality) {
        if self.ao.quality != quality {
            self.ao.quality = quality;
            for t in self.ao.targets.iter_mut().flatten() {
                t.reset_history();
            }
        }
    }

    pub fn ao_quality(&self) -> AoQuality {
        self.ao.quality
    }

    /// Decide whether this frame computes AO and for which target; create /
    /// resize that target, point `frame_set` binding 13 at its result, and
    /// return the frame-UBO flag (`refl_blend2.w`: 0 off, 1 on, 2 on with
    /// multi-bounce). Call after the frame slot's fence wait, before the UBO
    /// is written. AO stays off when the frame records no 3D camera pass.
    pub(crate) fn ao_prepare(&mut self, input: &crate::FrameInput, frame_set: vk::DescriptorSet, proj: Mat4, window: bool) -> f32 {
        self.ao.plan = None;
        let debug_view = input.gi_debug == VIEW_AO;
        let mut settings = input.postfx.ao;
        if debug_view {
            settings.enabled = true;
        }
        let device = self.ctx.device.clone();
        let allocator = self.allocator();
        {
            let mut a = allocator.lock().unwrap();
            self.ao.tick_retired(&device, &mut a);
        }
        // Off: binding 13 goes back to the neutral white texture, so the set
        // never keeps a view of a target that is later retired.
        let white = read_info(self.lightmap_sampler, self.default_textures[AO_DEFAULT_TEXTURE].view);
        let off = |device: &ash::Device| {
            unsafe { device.update_descriptor_sets(&[sampled(frame_set, 13, &white)], &[]) };
            0.0
        };
        if !settings.enabled || settings.intensity <= 0.0 {
            return off(&device);
        }
        let (target, extent) = if input.viewport_extent.is_some() && let Some(vt) = self.viewport_target.as_ref() {
            (TARGET_VIEWPORT, vt.extent)
        } else if input.render_viewport && window {
            (TARGET_GAME, vk::Extent2D { width: self.render_extent[0], height: self.render_extent[1] })
        } else {
            return off(&device);
        };
        let res: Result<()> = (|| {
            if self.ao.pass.is_none() {
                self.ao.pass = Some(AoPass::new(&device)?);
            }
            let stale = self.ao.targets[target].as_ref().is_some_and(|t| t.extent != extent);
            if stale && let Some(old) = self.ao.targets[target].take() {
                self.ao.retired.push((old, FRAMES_IN_FLIGHT + 1));
            }
            if self.ao.targets[target].is_none() {
                let mut a = allocator.lock().unwrap();
                self.ao.targets[target] = Some(AoTarget::new(&device, &mut a, self.ao.pass.as_ref().unwrap(), extent)?);
            }
            Ok(())
        })();
        if let Err(e) = res {
            tracing::warn!("AO targets: {e:#}");
            return off(&device);
        }
        let quality = self.ao.quality;
        if self.ao.last != Some((quality, settings)) {
            self.ao.last = Some((quality, settings));
            for t in self.ao.targets.iter_mut().flatten() {
                t.reset_history();
            }
        }
        self.ao.frame = self.ao.frame.wrapping_add(1);
        let t = self.ao.targets[target].as_ref().unwrap();
        let info = read_info(self.lightmap_sampler, t.full.view);
        unsafe { device.update_descriptor_sets(&[sampled(frame_set, 13, &info)], &[]) };
        self.ao.plan = Some(AoPlan {
            target,
            frame: AoFrame { proj, view: input.camera.view, settings, quality, frame: self.ao.frame },
        });
        if settings.multi_bounce { 2.0 } else { 1.0 }
    }

    /// Record this frame's AO for `target` if [`Self::ao_prepare`] planned it.
    /// `depth_view` must hold this camera's depth prepass (SHADER_READ_ONLY).
    pub(crate) fn record_ao(&mut self, device: &ash::Device, cb: vk::CommandBuffer, target: usize, depth_view: vk::ImageView) {
        let Some(plan) = self.ao.plan.filter(|p| p.target == target) else {
            return;
        };
        if self.ao.pass.is_none() || self.ao.targets[target].is_none() {
            return;
        }
        self.ctx.checkpoint(cb, c"ambient occlusion");
        self.gpu_zone_begin(cb, crate::gpu_zone::AO);
        let slot = self.frame_index;
        if let (Some(pass), Some(t)) = (self.ao.pass.as_ref(), self.ao.targets[target].as_mut()) {
            pass.record(device, cb, t, slot, depth_view, &plan.frame);
        }
        self.gpu_zone_end(cb, crate::gpu_zone::AO);
    }
}
