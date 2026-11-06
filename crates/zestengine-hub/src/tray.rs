//! Linux StatusNotifier tray, with the exact same orange icon as the window.
use std::time::Duration;

use ksni::blocking::TrayMethods;
use winit::event_loop::EventLoopProxy;

use crate::app::HubEvent;

pub struct HubTray {
    proxy: EventLoopProxy<HubEvent>,
    icon: Vec<u8>,
}

pub fn start(proxy: EventLoopProxy<HubEvent>) -> Option<ksni::blocking::Handle<HubTray>> {
    let (_, _, rgba) = zestengine_editor::icons::app_icon_rgba();
    // StatusNotifier pixmaps are network-order ARGB, rather than winit RGBA.
    let icon = rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[3], p[0], p[1], p[2]]).collect();
    match (HubTray { proxy, icon }).spawn() {
        Ok(handle) => Some(handle),
        Err(e) => {
            tracing::info!("Tray unavailable; keeping normal window minimization: {e}");
            None
        }
    }
}

impl HubTray {
    /// Wayland only lets a window take focus with the XDG activation token
    /// the tray host sends with the click, which KDE delivers just after it.
    fn activation_token() -> Option<String> {
        (0..100).find_map(|_| {
            let token = ksni::current_xdg_activation_token();
            if token.is_none() {
                std::thread::sleep(Duration::from_millis(1));
            }
            token
        })
    }

    fn show(&self) {
        let _ = self.proxy.send_event(HubEvent::Show(Self::activation_token()));
    }

    fn toggle(&self) {
        let _ = self.proxy.send_event(HubEvent::Toggle(Self::activation_token()));
    }
}

impl ksni::Tray for HubTray {
    fn id(&self) -> String { "zestengine-hub".into() }
    fn title(&self) -> String { "ZestEngine Project Hub".into() }
    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        vec![ksni::Icon { width: 256, height: 256, data: self.icon.clone() }]
    }
    fn activate(&mut self, _: i32, _: i32) {
        self.toggle();
    }
    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::StandardItem;
        vec![
            StandardItem { label: "Open ZestEngine Hub".into(), activate: Box::new(|tray: &mut Self| tray.show()), ..Default::default() }.into(),
            StandardItem { label: "Quit ZestEngine Hub".into(), activate: Box::new(|tray: &mut Self| { let _ = tray.proxy.send_event(HubEvent::Quit); }), ..Default::default() }.into(),
        ]
    }
}
