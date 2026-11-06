//! The hub's window: winit events, egui input and the egui-only Vulkan
//! window. Redraws only on input, egui's repaint requests and background
//! results, so an idle hub costs nothing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::hub::{Hub, HubRequests};
use crate::settings::MIN_SIZE;

pub fn run() -> Result<()> {
    let event_loop = EventLoop::<HubEvent>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        proxy: event_loop.create_proxy(),
        #[cfg(target_os = "linux")]
        tray: None,
        gpu: None,
        egui_state: None,
        window: None,
        egui_ctx: egui::Context::default(),
        hub: None,
        repaint_at: None,
        minimized: false,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

struct App {
    proxy: winit::event_loop::EventLoopProxy<HubEvent>,
    #[cfg(target_os = "linux")]
    tray: Option<ksni::blocking::Handle<crate::tray::HubTray>>,
    // Before `window`: the surface must go before the window it was made for.
    gpu: Option<zestengine_render::EguiWindow>,
    egui_state: Option<egui_winit::State>,
    window: Option<Arc<Window>>,
    egui_ctx: egui::Context,
    hub: Option<Hub>,
    /// When egui asked for its next repaint (animations, timed messages).
    repaint_at: Option<Instant>,
    minimized: bool,
    error: Option<anyhow::Error>,
}

/// Unwinding out of the event loop (a panic) skips `exiting`; dropping the
/// egui-winit state after the display connection closes would SIGSEGV in its
/// Wayland clipboard worker and hide the real panic. Leak it instead.
impl Drop for App {
    fn drop(&mut self) {
        if std::thread::panicking() {
            std::mem::forget(self.egui_state.take());
        }
    }
}

/// `Show` and `Toggle` carry the tray's XDG activation token, if the host sent one.
#[derive(Clone, Debug)]
pub(crate) enum HubEvent { Show(Option<String>), Toggle(Option<String>), Quit }

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        zestengine_editor::install_icon_font(&self.egui_ctx);
        zestengine_editor::theme::apply(&self.egui_ctx);
        if let Err(e) = zestengine_editor::icons::install_app_icon() {
            tracing::debug!("installing app icon: {e}");
        }
        let hub = Hub::new(&self.egui_ctx);
        let window = Self::create_window(event_loop, &hub, None)?;
        #[cfg(target_os = "linux")]
        { self.tray = crate::tray::start(self.proxy.clone()); }
        let gpu = zestengine_render::EguiWindow::new(&window, c"zestengine hub", true).context("starting Vulkan for the hub window")?;
        self.gpu = Some(gpu);
        self.hub = Some(hub);
        self.adopt_window(window);
        Ok(())
    }

    fn create_window(event_loop: &ActiveEventLoop, hub: &Hub, token: Option<String>) -> Result<Arc<Window>> {
        let [w, h] = hub.settings.window_size_clamped();
        let (icon_w, icon_h, icon_rgba) = zestengine_editor::icons::app_icon_rgba();
        let attrs = Window::default_attributes()
            .with_title("ZestEngine Project Hub")
            .with_inner_size(LogicalSize::new(w, h))
            .with_min_inner_size(LogicalSize::new(MIN_SIZE[0], MIN_SIZE[1]))
            .with_window_icon(winit::window::Icon::from_rgba(icon_rgba, icon_w, icon_h).ok());
        // Its own app id so the compositor matches the window to zestengine-hub.desktop (same icon).
        #[cfg(target_os = "linux")]
        let attrs = {
            use winit::platform::{wayland, x11};
            let attrs = wayland::WindowAttributesExtWayland::with_name(attrs, "zestengine-hub", "zestengine-hub");
            let attrs = x11::WindowAttributesExtX11::with_name(attrs, "zestengine-hub", "zestengine-hub");
            match token {
                Some(token) => winit::platform::startup_notify::WindowAttributesExtStartupNotify::with_activation_token(
                    attrs,
                    winit::window::ActivationToken::from_raw(token),
                ),
                None => attrs,
            }
        };
        #[cfg(not(target_os = "linux"))]
        let _ = token;
        Ok(Arc::new(event_loop.create_window(attrs).context("creating the hub window")?))
    }

    fn adopt_window(&mut self, window: Arc<Window>) {
        // Kept across reopened windows: a new State restarts its input clock at
        // zero, and egui panics when input time moves backwards.
        if self.egui_state.is_none() {
            self.egui_state = Some(egui_winit::State::new(
                self.egui_ctx.clone(),
                egui::ViewportId::ROOT,
                &window,
                Some(window.scale_factor() as f32),
                None,
                None,
            ));
        }
        window.request_redraw();
        self.window = Some(window);
        self.minimized = false;
    }

    /// Bring the hub back from the tray: a new window, since the old one was
    /// destroyed (see `minimize_to_tray`).
    fn show(&mut self, event_loop: &ActiveEventLoop, token: Option<String>) {
        if let Some(window) = &self.window {
            if self.minimized {
                window.set_minimized(false);
            }
            window.focus_window();
            self.wake();
            return;
        }
        let (Some(hub), Some(gpu)) = (self.hub.as_ref(), self.gpu.as_mut()) else { return };
        let window = match Self::create_window(event_loop, hub, token) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!("reopening the hub window: {e:#}");
                return;
            }
        };
        if let Err(e) = gpu.attach(&window) {
            tracing::error!("reopening the hub window: {e:#}");
            return;
        }
        self.adopt_window(window);
        if let Some(hub) = self.hub.as_mut() {
            hub.refresh();
        }
    }

    /// A tray click toggles: a shown window minimises to the tray, a hidden or
    /// minimised one is restored. On Wayland "hidden" means the window was
    /// destroyed by `minimize_to_tray`, so restoring makes a new one.
    fn toggle(&mut self, event_loop: &ActiveEventLoop, token: Option<String>) {
        if self.window.is_none() || self.minimized {
            self.show(event_loop, token);
        } else {
            self.minimize_to_tray();
        }
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(state), Some(hub), Some(gpu)) =
            (self.window.as_ref(), self.egui_state.as_mut(), self.hub.as_mut(), self.gpu.as_mut())
        else {
            return;
        };
        if self.minimized {
            return;
        }
        let raw = state.take_egui_input(window);
        let mut requests = HubRequests::default();
        let output = self.egui_ctx.run(raw, |ctx| requests = hub.ui(ctx));
        state.handle_platform_output(window, output.platform_output);
        let primitives = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);
        let draw = zestengine_render::EguiDraw {
            pixels_per_point: output.pixels_per_point,
            primitives,
            textures_delta: output.textures_delta,
        };
        if let Err(e) = gpu.render(&draw) {
            tracing::error!("hub frame: {e:#}");
        }
        let delay = output.viewport_output.get(&egui::ViewportId::ROOT).map_or(Duration::MAX, |v| v.repaint_delay);
        if delay.is_zero() {
            window.request_redraw();
        } else if delay < Duration::from_secs(3600) {
            self.repaint_at = Some(Instant::now() + delay);
        }
        if requests.close {
            tracing::info!("exiting: closing after opening a project");
            hub.save_settings();
            event_loop.exit();
        } else if requests.minimize {
            self.minimize_to_tray();
        }
    }
}

impl App {
    fn minimize_to_tray(&mut self) {
        let Some(window) = &self.window else { return };
        // Wayland can neither hide nor un-minimise a window, so the window is
        // destroyed and the tray's Open makes a new one. The device and egui
        // textures stay alive in `gpu`.
        #[cfg(target_os = "linux")]
        if self.tray.as_ref().is_some_and(|tray| !tray.is_closed())
            && let Some(gpu) = self.gpu.as_mut()
        {
            gpu.detach();
            self.window = None;
            self.repaint_at = None;
            return;
        }
        // Rendering pauses while minimised. Wayland never reports un-minimising,
        // so focus or input on the window (see `wake`) resumes it.
        window.set_minimized(true);
        self.minimized = true;
    }

    fn wake(&mut self) {
        if self.minimized {
            self.minimized = false;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

impl ApplicationHandler<HubEvent> for App {
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: HubEvent) {
        match event {
            HubEvent::Show(token) => self.show(event_loop, token),
            HubEvent::Toggle(token) => self.toggle(event_loop, token),
            HubEvent::Quit => {
                tracing::info!("exiting: Quit from the tray");
                event_loop.exit();
            }
        }
    }
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.hub.is_some() {
            return;
        }
        if let Err(e) = self.init(event_loop) {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(state)) = (self.window.clone(), self.egui_state.as_mut()) else {
            return;
        };
        if state.on_window_event(&window, &event).repaint {
            window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("exiting: window closed");
                if let Some(hub) = &self.hub {
                    hub.save_settings();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if !self.minimized {
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.resize(size.width, size.height);
                    }
                    if let Some(hub) = self.hub.as_mut() {
                        let logical = size.to_logical::<f32>(window.scale_factor());
                        hub.settings.window_size = [logical.width, logical.height];
                    }
                }
                window.request_redraw();
            }
            WindowEvent::Focused(true) => {
                self.wake();
                // The editor may have recorded a project, or been built, meanwhile.
                if let Some(hub) = self.hub.as_mut() {
                    hub.refresh();
                }
                window.request_redraw();
            }
            WindowEvent::CursorEntered { .. }
            | WindowEvent::MouseInput { .. }
            | WindowEvent::MouseWheel { .. }
            | WindowEvent::KeyboardInput { .. } => self.wake(),
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.minimized && self.window.as_ref().is_some_and(|w| w.is_visible() == Some(true) && w.is_minimized() == Some(true)) {
            self.minimize_to_tray();
        }
        match self.repaint_at {
            Some(at) if Instant::now() >= at => {
                self.repaint_at = None;
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(at) => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(hub) = &self.hub {
            hub.save_settings();
        }
        // GPU objects before the window they render to.
        self.gpu = None;
        // The egui-winit state while the display connection is still open: its
        // Wayland clipboard worker thread is joined on drop, and destroying
        // Wayland objects after the connection closes crashes (SIGSEGV in
        // `wl_proxy_destroy`).
        self.egui_state = None;
    }
}
