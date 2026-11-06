//! Swapchain creation, image views, and per-image present semaphores.

use anyhow::{Context as _, Result};
use ash::{khr, vk};

use crate::context::GpuContext;

pub struct Swapchain {
    loader: khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    /// One per swapchain image: signaled by the submit that renders to that
    /// image, waited on by present.
    pub render_finished: Vec<vk::Semaphore>,
    #[allow(dead_code)] // consumed by pipeline creation in the mesh milestone
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    /// Present mode actually chosen (read by the frame limiter policy).
    pub present_mode: vk::PresentModeKHR,
    /// The images can be a copy source (probe captures).
    pub can_copy_from: bool,
}

impl Swapchain {
    pub fn new(ctx: &GpuContext, width: u32, height: u32, vsync: bool) -> Result<Self> {
        Self::new_for_surface(ctx, ctx.surface, width, height, vsync)
    }

    /// Build a swapchain for an arbitrary surface (the main window uses
    /// `ctx.surface`; the profiler window passes its own). Shares the context's
    /// device/queue so no extra GPU setup is needed.
    pub fn new_for_surface(
        ctx: &GpuContext,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self> {
        Self::build(ctx, surface, width, height, vsync)
    }

    /// A swapchain for a window that must never wait on vblank (secondary
    /// windows such as the Profiler). Always MAILBOX, else IMMEDIATE, else FIFO
    /// when the surface offers neither (FIFO then needs the non-blocking
    /// [`Self::try_acquire`] so the main loop still never stalls on it).
    pub fn new_secondary(
        ctx: &GpuContext,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        Self::build(ctx, surface, width, height, false)
    }

    fn build(
        ctx: &GpuContext,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self> {
        let loader = khr::swapchain::Device::new(&ctx.instance, &ctx.device);

        let caps = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, surface)?
        };
        let formats = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_formats(ctx.physical_device, surface)?
        };
        let present_modes = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_present_modes(ctx.physical_device, surface)?
        };
        // FIFO (vsync) is always available. Without vsync prefer MAILBOX
        // (uncapped, no tearing), then IMMEDIATE (uncapped, may tear).
        // ZESTENGINE_PRESENT_MODE=fifo|fifo_relaxed|mailbox|immediate forces a
        // mode (when the surface offers it) to diagnose present-path stalls:
        // fifo_relaxed presents a late frame at once (may tear) instead of
        // waiting a whole extra refresh, mailbox never blocks.
        let forced = std::env::var("ZESTENGINE_PRESENT_MODE").ok().and_then(|v| {
            let mode = match v.trim().to_ascii_lowercase().as_str() {
                "fifo" => vk::PresentModeKHR::FIFO,
                "fifo_relaxed" | "relaxed" => vk::PresentModeKHR::FIFO_RELAXED,
                "mailbox" => vk::PresentModeKHR::MAILBOX,
                "immediate" => vk::PresentModeKHR::IMMEDIATE,
                "fifo_latest_ready" | "latest_ready" => crate::context::PRESENT_MODE_FIFO_LATEST_READY,
                _ => return None,
            };
            if mode == crate::context::PRESENT_MODE_FIFO_LATEST_READY && !ctx.caps.present_fifo_latest_ready {
                tracing::warn!("ZESTENGINE_PRESENT_MODE={v}: device extension/feature present_mode_fifo_latest_ready not enabled");
                None
            } else if present_modes.contains(&mode) {
                Some(mode)
            } else {
                tracing::warn!("ZESTENGINE_PRESENT_MODE={v}: not offered by this surface ({})", present_mode_list(&present_modes));
                None
            }
        });
        let present_mode = if let Some(mode) = forced {
            mode
        } else if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
                .into_iter()
                .find(|m| present_modes.contains(m))
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "present modes offered: [{}], chosen: {} (ZESTENGINE_PRESENT_MODE=fifo|fifo_relaxed|mailbox|immediate|fifo_latest_ready overrides)",
                present_mode_list(&present_modes),
                crate::context::present_mode_name(present_mode)
            );
        });

        let format = formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first().copied())
            .context("surface reports no formats")?;

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let env_images = std::env::var("ZESTENGINE_SWAPCHAIN_IMAGES").ok();
        let image_count = choose_image_count(caps.min_image_count, caps.max_image_count, env_images.as_deref());
        static LOGGED_IMAGES: std::sync::Once = std::sync::Once::new();
        LOGGED_IMAGES.call_once(|| {
            tracing::info!(
                "swapchain images: requested {image_count} (surface min {} max {}, ZESTENGINE_SWAPCHAIN_IMAGES=<n> overrides, default min+1)",
                caps.min_image_count,
                if caps.max_image_count == 0 { "unbounded".to_string() } else { caps.max_image_count.to_string() }
            );
        });

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | (caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC),
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);
        let handle =
            unsafe { loader.create_swapchain(&info, None) }.context("creating swapchain")?;

        let images = unsafe { loader.get_swapchain_images(handle)? };
        let views = images
            .iter()
            .map(|&image| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                unsafe { ctx.device.create_image_view(&info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let render_finished = images
            .iter()
            .map(|_| unsafe {
                ctx.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        tracing::debug!(
            ?extent,
            format = ?format.format,
            images = images.len(),
            ?present_mode,
            "swapchain created"
        );

        Ok(Self {
            loader,
            handle,
            images,
            views,
            render_finished,
            format,
            extent,
            present_mode,
            can_copy_from: caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_SRC),
        })
    }

    pub fn acquire(&self, signal: vk::Semaphore) -> ash::prelude::VkResult<(u32, bool)> {
        unsafe {
            self.loader
                .acquire_next_image(self.handle, u64::MAX, signal, vk::Fence::null())
        }
    }

    /// Bounded acquire: `Ok(None)` when no image is free within `timeout_ns`
    /// (keep it tiny). Used by secondary windows so a FIFO fallback can't stall
    /// the main loop.
    pub fn try_acquire(
        &self,
        signal: vk::Semaphore,
        timeout_ns: u64,
    ) -> ash::prelude::VkResult<Option<(u32, bool)>> {
        match unsafe {
            self.loader
                .acquire_next_image(self.handle, timeout_ns, signal, vk::Fence::null())
        } {
            Ok(r) => Ok(Some(r)),
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait: vk::Semaphore,
    ) -> ash::prelude::VkResult<bool> {
        let waits = [wait];
        let swapchains = [self.handle];
        let indices = [image_index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        unsafe { self.loader.queue_present(queue, &info) }
    }

    /// Caller must ensure the GPU is idle or no longer using these objects.
    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for &view in &self.views {
                device.destroy_image_view(view, None);
            }
            for &sem in &self.render_finished {
                device.destroy_semaphore(sem, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
        self.handle = vk::SwapchainKHR::null();
        self.views.clear();
        self.render_finished.clear();
        self.images.clear();
    }
}

/// `MAILBOX, FIFO_LATEST_READY, FIFO` style list for logs.
/// Swapchain image count: `min + 1` by default, or `ZESTENGINE_SWAPCHAIN_IMAGES`
/// (`env`), clamped to the surface's `[min, max]` (`max` 0 = unbounded).
pub fn choose_image_count(min: u32, max: u32, env: Option<&str>) -> u32 {
    let want = env.and_then(|v| v.trim().parse::<u32>().ok()).filter(|n| *n > 0).unwrap_or(min + 1);
    let want = want.max(min);
    if max > 0 { want.min(max) } else { want }
}

fn present_mode_list(modes: &[vk::PresentModeKHR]) -> String {
    modes.iter().map(|m| crate::context::present_mode_name(*m)).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod image_count_tests {
    use super::choose_image_count;

    #[test]
    fn default_is_min_plus_one_clamped() {
        assert_eq!(choose_image_count(2, 8, None), 3);
        assert_eq!(choose_image_count(2, 2, None), 2);
        assert_eq!(choose_image_count(2, 0, None), 3);
    }

    #[test]
    fn env_override_is_clamped_to_the_surface() {
        assert_eq!(choose_image_count(2, 8, Some("4")), 4);
        assert_eq!(choose_image_count(2, 8, Some(" 16 ")), 8);
        assert_eq!(choose_image_count(3, 8, Some("1")), 3);
        assert_eq!(choose_image_count(2, 0, Some("6")), 6);
        assert_eq!(choose_image_count(2, 8, Some("junk")), 3);
        assert_eq!(choose_image_count(2, 8, Some("0")), 3);
    }
}
