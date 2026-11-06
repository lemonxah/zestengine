//! A standalone window that draws only an egui UI: its own small Vulkan
//! context (instance, device, the window's surface) plus the profiler
//! window's swapchain + egui renderer. For tools with no 3D view, such as the
//! Project Hub, so they don't build the full scene renderer.

use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::EguiDraw;
use crate::context::GpuContext;
use crate::profiler_window::ProfilerWindow;

pub struct EguiWindow {
    // Declared before `ctx`: the swapchain, egui renderer and allocator must go
    // before the device they were made on.
    target: ProfilerWindow,
    ctx: GpuContext,
}

impl EguiWindow {
    pub fn new(window: &Window, app_name: &std::ffi::CStr, vsync: bool) -> Result<Self> {
        let display = window.display_handle()?.as_raw();
        let handle = window.window_handle()?.as_raw();
        let ctx = GpuContext::new(display, handle, app_name, None)?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance.clone(),
            device: ctx.device.clone(),
            physical_device: ctx.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: ctx.ray_tracing(),
            allocation_sizes: Default::default(),
        })
        .context("creating GPU allocator")?;
        let size = window.inner_size();
        let target = ProfilerWindow::with_surface(
            &ctx,
            Arc::new(Mutex::new(allocator)),
            ctx.surface,
            false,
            size.width.max(1),
            size.height.max(1),
            vsync,
        )?;
        Ok(Self { target, ctx })
    }

    /// Let go of `window` so it can be destroyed; the device and egui textures
    /// stay, ready for [`Self::attach`] to a new window. Wayland can't hide or
    /// un-minimise a window, so hiding to a tray means destroying it.
    pub fn detach(&mut self) {
        self.target.detach_surface(&self.ctx);
        unsafe { self.ctx.surface_loader.destroy_surface(self.ctx.surface, None) };
        self.ctx.surface = ash::vk::SurfaceKHR::null();
    }

    pub fn attach(&mut self, window: &Window) -> Result<()> {
        let display = window.display_handle()?.as_raw();
        let handle = window.window_handle()?.as_raw();
        let surface = unsafe {
            ash_window::create_surface(&self.ctx.entry, &self.ctx.instance, display, handle, None)
        }
        .context("creating window surface")?;
        let size = window.inner_size();
        self.target.attach_surface(&self.ctx, surface, true, size.width, size.height)
    }

    /// Recreate the swapchain at the new size before the next frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.target.request_resize(width, height);
    }

    pub fn render(&mut self, draw: &EguiDraw) -> Result<()> {
        self.target.render(&self.ctx, draw)
    }
}

impl Drop for EguiWindow {
    fn drop(&mut self) {
        self.target.destroy(&self.ctx);
    }
}
