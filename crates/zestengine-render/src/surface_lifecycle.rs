//! Window surface loss and recreation. On Android the activity's native
//! window is destroyed whenever the app pauses (home button, headset off, app
//! switch) and a new one arrives on resume; the `VkSurfaceKHR` and swapchain
//! built on the old window must go with it. The device and every other GPU
//! resource survive, so off-screen work (XR eyes, uploads) is unaffected.

use crate::hitch::WaitNamed as _;
use anyhow::{Context as _, Result, bail};
use ash::vk;

use crate::{Renderer, WindowHandles};

impl Renderer {
    /// The window's native surface is gone (winit `suspended`). Destroys the
    /// swapchain and surface; skip [`Renderer::render`] until
    /// [`Renderer::surface_recreated`].
    pub fn surface_lost(&mut self) {
        if self.ctx.surface == vk::SurfaceKHR::null() {
            return;
        }
        unsafe {
            let _ = self.ctx.device.wait_idle("wait:device_wait_idle@surface_lost");
        }
        self.swapchain.destroy(&self.ctx.device);
        self.swapchain.handle = vk::SwapchainKHR::null();
        unsafe { self.ctx.surface_loader.destroy_surface(self.ctx.surface, None) };
        self.ctx.surface = vk::SurfaceKHR::null();
    }

    /// A native window is back (winit `resumed` after `suspended`). Builds a
    /// surface for it; the next `render` recreates the swapchain.
    pub fn surface_recreated(&mut self) -> Result<()> {
        if self.ctx.surface != vk::SurfaceKHR::null() {
            return Ok(());
        }
        let handles = WindowHandles::of(&self.window)?;
        let surface = unsafe {
            ash_window::create_surface(&self.ctx.entry, &self.ctx.instance, handles.display, handles.window, None)
        }
        .context("recreating window surface")?;
        let present = unsafe {
            self.ctx.surface_loader.get_physical_device_surface_support(
                self.ctx.physical_device,
                self.ctx.queue_family,
                surface,
            )
        }
        .unwrap_or(false);
        if !present {
            unsafe { self.ctx.surface_loader.destroy_surface(surface, None) };
            bail!("the recreated window surface cannot be presented from the graphics queue");
        }
        self.ctx.surface = surface;
        self.resize_gate.force();
        Ok(())
    }

    /// False between [`Renderer::surface_lost`] and
    /// [`Renderer::surface_recreated`].
    pub fn has_surface(&self) -> bool {
        self.ctx.surface != vk::SurfaceKHR::null()
    }
}
