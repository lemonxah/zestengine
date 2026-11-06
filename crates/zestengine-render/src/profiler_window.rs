//! A second OS window that renders only an egui UI (the profiler), sharing the
//! main `GpuContext` (instance/device/queue). It owns its own surface,
//! swapchain, egui renderer, and command buffer + sync for `FRAMES` frames in
//! flight.
//!
//! As a secondary window (the Profiler) it must never make the main loop wait
//! on a vblank: its swapchain is MAILBOX/IMMEDIATE (never FIFO unless the
//! surface offers nothing else), it acquires with a tiny bounded timeout and
//! skips the frame instead of blocking, and it is throttled to ~60 Hz. The
//! standalone-window users (`EguiWindow`, the Hub) are their own main window
//! and keep the caller's VSync choice.

use crate::hitch::WaitNamed as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::EguiDraw;
use crate::context::GpuContext;
use crate::image_barrier;
use crate::swapchain::Swapchain;

/// Frames in flight for this window (own command buffer, semaphore and fence
/// per slot, so the CPU can record N+1 while N is still on the GPU).
const FRAMES: usize = 2;
/// Secondary windows redraw at most this often.
const SECONDARY_INTERVAL: Duration = Duration::from_micros(16_000);

pub struct ProfilerWindow {
    allocator: Arc<Mutex<Allocator>>,
    surface: vk::SurfaceKHR,
    owns_surface: bool,
    swapchain: Swapchain,
    egui: egui_ash_renderer::Renderer,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    /// Frame slot the next submit uses.
    slot: usize,
    needs_resize: bool,
    /// egui texture frees from this window's `textures_delta.free` that have not
    /// been assigned to a submission yet.
    incoming_free: Vec<egui::TextureId>,
    /// Texture uploads not yet recorded into a command buffer (kept across
    /// skipped frames: egui sends each delta once).
    pending_uploads: Vec<(egui::TextureId, egui::epaint::ImageDelta)>,
    /// Frees assigned to the submission that used each slot. Released only
    /// after that slot's fence retires again, i.e. after that submission AND the
    /// one before it (which may still sample the texture) are done.
    pending_free: Vec<Vec<egui::TextureId>>,
    /// Secondary window: MAILBOX/IMMEDIATE, bounded acquire, throttled.
    secondary: bool,
    last_render: Option<Instant>,
    width: u32,
    height: u32,
    vsync: bool,
}

impl ProfilerWindow {
    pub fn new(
        ctx: &GpuContext,
        allocator: Arc<Mutex<Allocator>>,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self> {
        let surface = unsafe {
            ash_window::create_surface(&ctx.entry, &ctx.instance, display, window, None)
        }
        .context("creating profiler surface")?;
        Self::build(ctx, allocator, surface, true, width, height, vsync, true)
    }

    /// Like [`ProfilerWindow::new`] on an existing surface. `owns_surface`
    /// decides whether `destroy` frees it (false when the context owns it).
    pub fn with_surface(
        ctx: &GpuContext,
        allocator: Arc<Mutex<Allocator>>,
        surface: vk::SurfaceKHR,
        owns_surface: bool,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self> {
        Self::build(ctx, allocator, surface, owns_surface, width, height, vsync, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        ctx: &GpuContext,
        allocator: Arc<Mutex<Allocator>>,
        surface: vk::SurfaceKHR,
        owns_surface: bool,
        width: u32,
        height: u32,
        vsync: bool,
        secondary: bool,
    ) -> Result<Self> {
        let swapchain = if secondary {
            Swapchain::new_secondary(ctx, surface, width, height)?
        } else {
            Swapchain::new_for_surface(ctx, surface, width, height, vsync)?
        };

        let egui = egui_ash_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            ctx.device.clone(),
            egui_ash_renderer::DynamicRendering {
                color_attachment_format: swapchain.format.format,
                depth_attachment_format: None,
            },
            egui_ash_renderer::Options {
                in_flight_frames: FRAMES,
                enable_depth_test: false,
                enable_depth_write: false,
                srgb_framebuffer: true,
            },
        )
        .map_err(|e| anyhow::anyhow!("creating profiler egui renderer: {e}"))?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(ctx.queue_family);
        let command_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };
        let command_buffers = unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES as u32),
            )?
        };
        let mut image_available = Vec::with_capacity(FRAMES);
        let mut in_flight = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            image_available.push(unsafe {
                ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            });
            in_flight.push(unsafe {
                ctx.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?
            });
        }

        Ok(Self {
            allocator,
            surface,
            owns_surface,
            swapchain,
            egui,
            command_pool,
            command_buffers,
            image_available,
            in_flight,
            slot: 0,
            needs_resize: false,
            incoming_free: Vec::new(),
            pending_uploads: Vec::new(),
            pending_free: vec![Vec::new(); FRAMES],
            secondary,
            last_render: None,
            width,
            height,
            vsync,
        })
    }

    /// Drop the swapchain and surface so the OS window can be destroyed while
    /// the egui renderer (and its uploaded textures) stays alive.
    pub fn detach_surface(&mut self, ctx: &GpuContext) {
        unsafe {
            let _ = ctx.device.wait_idle("wait:device_wait_idle@detach_surface");
            self.swapchain.destroy(&ctx.device);
            if self.owns_surface {
                ctx.surface_loader.destroy_surface(self.surface, None);
            }
        }
        self.surface = vk::SurfaceKHR::null();
        self.owns_surface = false;
    }

    /// Render to a new window's surface after [`Self::detach_surface`].
    pub fn attach_surface(
        &mut self,
        ctx: &GpuContext,
        surface: vk::SurfaceKHR,
        owns_surface: bool,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.detach_surface(ctx);
        self.surface = surface;
        self.owns_surface = owns_surface;
        self.width = width.max(1);
        self.height = height.max(1);
        self.swapchain = self.make_swapchain(ctx, surface)?;
        self.needs_resize = false;
        Ok(())
    }

    fn make_swapchain(&self, ctx: &GpuContext, surface: vk::SurfaceKHR) -> Result<Swapchain> {
        if self.secondary {
            Swapchain::new_secondary(ctx, surface, self.width, self.height)
        } else {
            Swapchain::new_for_surface(ctx, surface, self.width, self.height, self.vsync)
        }
    }

    /// True when a throttled secondary window is due for another redraw. Always
    /// true for a primary (standalone) window, which paces itself on vblank.
    pub fn due(&self) -> bool {
        !self.secondary || self.last_render.is_none_or(|t| t.elapsed() >= SECONDARY_INTERVAL)
    }

    pub fn request_resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.needs_resize = true;
    }

    fn recreate(&mut self, ctx: &GpuContext) -> Result<()> {
        unsafe { ctx.device.wait_idle("wait:device_wait_idle@recreate")? };
        self.swapchain.destroy(&ctx.device);
        self.swapchain = self.make_swapchain(ctx, self.surface)?;
        self.needs_resize = false;
        Ok(())
    }

    /// Record + submit + present the egui UI to the profiler window. Skips
    /// silently on a zero-size window or an out-of-date swapchain (re-armed for
    /// the next frame).
    pub fn render(&mut self, ctx: &GpuContext, draw: &EguiDraw) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.surface == vk::SurfaceKHR::null() {
            return Ok(());
        }
        if self.needs_resize {
            self.recreate(ctx)?;
        }
        let device = &ctx.device;
        let slot = self.slot;

        unsafe { device.wait_fences("wait:fence@render", &[self.in_flight[slot]], true, u64::MAX)? };

        // This slot's previous submission retired (and, queue order, the one
        // before it), so the frees assigned to it can no longer be sampled.
        // Staging buffers / replaced textures of this slot's earlier uploads.
        let _ = self.egui.release_retired(slot);
        if !self.pending_free[slot].is_empty() {
            let ids = std::mem::take(&mut self.pending_free[slot]);
            let _ = self.egui.free_textures(&ids);
        }
        // egui never recycles a managed id, but if a `set` ever names an id that
        // is still waiting to be freed, fall back to a full idle and free
        // everything first rather than updating a texture a frame may sample.
        if draw.textures_delta.set.iter().any(|(id, _)| {
            self.incoming_free.contains(id) || self.pending_free.iter().any(|v| v.contains(id))
        }) {
            unsafe { device.wait_idle("wait:device_wait_idle@render")? };
            let mut ids = std::mem::take(&mut self.incoming_free);
            for v in &mut self.pending_free {
                ids.append(v);
            }
            let _ = self.egui.free_textures(&ids);
        }
        // Uploads are recorded into this frame's command buffer below (no
        // submit + queue wait); queued here so a skipped frame keeps them.
        self.pending_uploads.extend(draw.textures_delta.set.iter().cloned());

        // Frees from this frame's delta are not referenced by its primitives;
        // they are assigned to a submission once one is made (below).
        self.incoming_free
            .extend_from_slice(&draw.textures_delta.free);

        // Bounded acquire: a secondary window skips the frame rather than
        // waiting on a presentation engine that may be vblank-locked.
        let acquired = if self.secondary {
            self.swapchain
                .try_acquire(self.image_available[slot], 2_000_000)
        } else {
            self.swapchain.acquire(self.image_available[slot]).map(Some)
        };
        let image_index = match acquired {
            Ok(Some((index, suboptimal))) => {
                if suboptimal {
                    self.needs_resize = true;
                }
                index
            }
            Ok(None) => return Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_resize = true;
                return Ok(());
            }
            Err(e) => return Err(e).context("acquiring profiler swapchain image"),
        };

        unsafe { device.reset_fences(&[self.in_flight[slot]])? };
        self.last_render = Some(Instant::now());

        let cb = self.command_buffers[slot];
        let image = self.swapchain.images[image_index as usize];
        let view = self.swapchain.views[image_index as usize];
        let render_finished = self.swapchain.render_finished[image_index as usize];
        let extent = self.swapchain.extent;

        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            if !self.pending_uploads.is_empty() {
                let uploads = std::mem::take(&mut self.pending_uploads);
                // Not fatal: an early return here would leave the fence reset
                // and unsignalled, hanging the next wait on this slot.
                if let Err(e) = self.egui.record_textures(cb, slot, &uploads) {
                    tracing::warn!("profiler egui textures: {e}");
                }
            }

            image_barrier(
                device, cb, image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );

            let color = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.02, 0.02, 0.03, 1.0] },
                });
            let attachments = [color];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(&attachments);
            device.cmd_begin_rendering(cb, &info);
            self.egui
                .cmd_draw(cb, extent, draw.pixels_per_point, &draw.primitives)
                .map_err(|e| anyhow::anyhow!("profiler egui draw: {e}"))?;
            device.cmd_end_rendering(cb);

            image_barrier(
                device, cb, image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE, vk::AccessFlags2::empty(),
            );

            device.end_command_buffer(cb)?;

            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.image_available[slot])
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(render_finished)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let submit = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait)
                .command_buffer_infos(&cbs)
                .signal_semaphore_infos(&signal);
            device.queue_submit2(ctx.queue, &[submit], self.in_flight[slot])?;
        }

        match self.swapchain.present(ctx.queue, image_index, render_finished) {
            Ok(suboptimal) => {
                if suboptimal {
                    self.needs_resize = true;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_resize = true,
            Err(e) => return Err(e).context("presenting profiler swapchain"),
        }

        // The submission is in flight: park the pending frees on its slot.
        let parked = std::mem::take(&mut self.incoming_free);
        self.pending_free[slot].extend(parked);
        self.slot = (slot + 1) % FRAMES;
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            let _ = ctx.device.wait_idle("wait:device_wait_idle@destroy");
            let mut ids = std::mem::take(&mut self.incoming_free);
            for v in &mut self.pending_free {
                ids.append(v);
            }
            if !ids.is_empty() {
                let _ = self.egui.free_textures(&ids);
            }
            for s in self.image_available.drain(..) {
                ctx.device.destroy_semaphore(s, None);
            }
            for f in self.in_flight.drain(..) {
                ctx.device.destroy_fence(f, None);
            }
            self.swapchain.destroy(&ctx.device);
            ctx.device.destroy_command_pool(self.command_pool, None);
            if self.owns_surface {
                ctx.surface_loader.destroy_surface(self.surface, None);
            }
        }
        // egui renderer + allocator handle drop themselves.
        let _ = &self.allocator;
    }
}
