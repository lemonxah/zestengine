//! Engine-side input capture for the binding system (2C). Translates winit
//! keyboard/mouse events, `gilrs` gamepad state and the VR controllers into a
//! [`RawInput`], runs the live bindings (player overrides, auto scheme
//! switching, rebind capture; see [`ControlsRuntime`]), resolves the active
//! scheme and exposes the per-frame [`InputState`] that components read
//! through `ComponentCtx`.

use std::path::{Path, PathBuf};

use zestengine_core::{
    Bindings, CaptureResult, ControlScheme, ControlsInfo, ControlsRuntime, InputCapture, InputCommand, InputState, Key,
    MouseButton, PadAxis, PadButton, RawInput, resolve,
};
use winit::event::MouseButton as WMouse;
use winit::keyboard::KeyCode;

/// Owns gamepad polling + accumulated raw input + the resolved snapshot.
pub struct InputManager {
    gilrs: Option<gilrs::Gilrs>,
    controls: ControlsRuntime,
    raw: RawInput,
    vr_active: bool,
    state: InputState,
    /// Empty scheme used when bindings have none (keeps resolve total).
    fallback: ControlScheme,
    /// UI navigation inputs held last frame / pressed this frame.
    nav_held: crate::game_ui::UiNav,
    nav: crate::game_ui::UiNav,
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new(Bindings::default())
    }
}

impl InputManager {
    pub fn new(bindings: Bindings) -> Self {
        let gilrs = match gilrs::Gilrs::new() {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!("gamepad init failed (gamepads disabled): {e}");
                None
            }
        };
        Self {
            gilrs,
            controls: ControlsRuntime::new(bindings),
            raw: RawInput::default(),
            vr_active: false,
            state: InputState::default(),
            fallback: ControlScheme {
                name: "none".into(),
                actions: Vec::new(),
            },
            nav_held: Default::default(),
            nav: Default::default(),
        }
    }

    /// The resolved action snapshot for the current frame.
    pub fn state(&self) -> &InputState {
        &self.state
    }

        pub fn controls(&self) -> &ControlsRuntime {
        &self.controls
    }

    /// New project bindings (edited in the editor); player overrides stay on top.
        pub fn set_project_bindings(&mut self, bindings: Bindings) {
        self.controls.set_defaults(bindings);
    }

    /// Where player overrides are loaded from and saved to.
    pub fn set_overrides_path(&mut self, path: Option<PathBuf>) {
        self.controls.set_overrides_path(path);
    }

    /// Apply a gameplay rebind / reset / capture request.
    pub fn apply(&mut self, cmd: InputCommand) {
        if let Err(e) = self.controls.apply(cmd) {
            tracing::warn!("input bindings: {e}");
        }
    }

    /// What gameplay sees of the bindings this frame (`ScreenInfo::controls`).
    pub fn controls_info(&self) -> ControlsInfo {
        self.controls.info()
    }

    // --- winit event feed (accumulated through the frame) ---

    pub fn key(&mut self, code: KeyCode, pressed: bool) {
        let k = map_key(code);
        if pressed {
            self.raw.keys_down.insert(k);
        } else {
            self.raw.keys_down.remove(&k);
        }
    }

    pub fn mouse_button(&mut self, button: WMouse, pressed: bool) {
        let Some(b) = map_mouse(button) else { return };
        if pressed {
            self.raw.mouse_down.insert(b);
        } else {
            self.raw.mouse_down.remove(&b);
        }
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        self.raw.mouse_delta.x += dx as f32;
        self.raw.mouse_delta.y += dy as f32;
    }

    pub fn wheel(&mut self, dy: f32) {
        self.raw.wheel += dy;
    }

    /// This frame's VR controllers (call before [`Self::resolve_frame`]).
    pub fn feed_vr(&mut self, vr: &zestengine_core::VrFrame) {
        self.vr_active = vr.active;
        self.raw.set_vr(vr);
    }

    /// Clear keys/buttons (e.g. on focus loss) so nothing sticks "down".
    pub fn clear_held(&mut self) {
        self.raw.keys_down.clear();
        self.raw.mouse_down.clear();
    }

    /// Poll the gamepads and feed one editor-side capture (rebinding in the
    /// bindings editor, outside Play).
        pub fn poll_capture(&mut self, capture: &mut InputCapture) -> Option<CaptureResult> {
        self.poll_gamepads();
        let result = capture.poll(&self.raw);
        self.raw.mouse_delta = glam::Vec2::ZERO;
        self.raw.wheel = 0.0;
        result
    }

    /// Poll gamepads and resolve the accumulated raw input into the snapshot.
    /// Call once per frame before driving components. Returns the new snapshot.
    /// While a rebind capture runs, every action reads released.
    pub fn resolve_frame(&mut self) -> &InputState {
        self.poll_gamepads();
        self.controls.frame(&self.raw, self.vr_active);
        let capturing = self.controls.capturing();
        let scheme = self.controls.bindings().active_scheme().unwrap_or(&self.fallback);
        self.state = if capturing {
            resolve(&self.fallback, &RawInput::default(), &self.state)
        } else {
            resolve(scheme, &self.raw, &self.state)
        };
        self.update_nav(capturing);
        // Per-frame deltas reset after resolving; held sets persist.
        self.raw.mouse_delta = glam::Vec2::ZERO;
        self.raw.wheel = 0.0;
        &self.state
    }

    /// UI navigation edges this frame (arrows / D-pad / left stick, Enter or
    /// South to submit, Escape or East to cancel).
    pub fn ui_nav(&self) -> crate::game_ui::UiNav {
        self.nav
    }

    /// `suppress`: a capture has the input, so nothing navigates (and what
    /// is held now won't fire when it ends).
    fn update_nav(&mut self, suppress: bool) {
        use zestengine_core::{Key, PadAxis, PadButton};
        let r = &self.raw;
        let key = |k: Key| r.keys_down.contains(&k);
        let pad = |b: PadButton| r.pad_down.contains(&b);
        let axis = |a: PadAxis| r.pad_axes.get(&a).copied().unwrap_or(0.0);
        let held = crate::game_ui::UiNav {
            up: key(Key::Up) || pad(PadButton::DPadUp) || axis(PadAxis::LeftStickY) > 0.6,
            down: key(Key::Down) || pad(PadButton::DPadDown) || axis(PadAxis::LeftStickY) < -0.6,
            left: key(Key::Left) || pad(PadButton::DPadLeft) || axis(PadAxis::LeftStickX) < -0.6,
            right: key(Key::Right) || pad(PadButton::DPadRight) || axis(PadAxis::LeftStickX) > 0.6,
            submit: key(Key::Enter) || pad(PadButton::South),
            cancel: key(Key::Escape) || pad(PadButton::East),
        };
        let p = if suppress { held } else { self.nav_held };
        self.nav = crate::game_ui::UiNav {
            up: held.up && !p.up,
            down: held.down && !p.down,
            left: held.left && !p.left,
            right: held.right && !p.right,
            submit: held.submit && !p.submit,
            cancel: held.cancel && !p.cancel,
        };
        self.nav_held = held;
    }

    fn poll_gamepads(&mut self) {
        let Some(gilrs) = self.gilrs.as_mut() else {
            return;
        };
        // Pump the event queue so cached gamepad state is current, noting any
        // disconnect: a removed pad leaves gilrs's force-feedback thread holding
        // the now-deleted `/dev/input/eventN` fd, which then spams ENODEV every
        // ~50ms forever. Recreating the Gilrs context below drops that thread +
        // its stale fd (and re-enumerates the pads still plugged in).
        let mut disconnected = false;
        while let Some(ev) = gilrs.next_event() {
            if matches!(ev.event, gilrs::EventType::Disconnected) {
                disconnected = true;
            }
        }
        self.raw.pad_down.clear();
        self.raw.pad_axes.clear();
        // Merge all connected pads (local co-op shares one snapshot for now).
        for (_id, pad) in gilrs.gamepads() {
            for (gb, pb) in PAD_BUTTONS {
                if pad.is_pressed(gb) {
                    self.raw.pad_down.insert(pb);
                }
            }
            for (ga, pa) in PAD_AXES {
                let v = pad.value(ga);
                if v != 0.0 {
                    *self.raw.pad_axes.entry(pa).or_insert(0.0) += v;
                }
            }
        }
        if disconnected {
            // Drop the old context (joins its FF thread, closes the dead fd) and
            // rebuild. Keep the old one if rebuild fails so input still works.
            if let Ok(g) = gilrs::Gilrs::new() {
                self.gilrs = Some(g);
            }
        }
    }
}

const PAD_BUTTONS: [(gilrs::Button, PadButton); 14] = [
    (gilrs::Button::South, PadButton::South),
    (gilrs::Button::East, PadButton::East),
    (gilrs::Button::West, PadButton::West),
    (gilrs::Button::North, PadButton::North),
    (gilrs::Button::LeftTrigger, PadButton::LeftBumper),
    (gilrs::Button::RightTrigger, PadButton::RightBumper),
    (gilrs::Button::Select, PadButton::Select),
    (gilrs::Button::Start, PadButton::Start),
    (gilrs::Button::LeftThumb, PadButton::LeftThumb),
    (gilrs::Button::RightThumb, PadButton::RightThumb),
    (gilrs::Button::DPadUp, PadButton::DPadUp),
    (gilrs::Button::DPadDown, PadButton::DPadDown),
    (gilrs::Button::DPadLeft, PadButton::DPadLeft),
    (gilrs::Button::DPadRight, PadButton::DPadRight),
];

const PAD_AXES: [(gilrs::Axis, PadAxis); 6] = [
    (gilrs::Axis::LeftStickX, PadAxis::LeftStickX),
    (gilrs::Axis::LeftStickY, PadAxis::LeftStickY),
    (gilrs::Axis::RightStickX, PadAxis::RightStickX),
    (gilrs::Axis::RightStickY, PadAxis::RightStickY),
    (gilrs::Axis::LeftZ, PadAxis::LeftTrigger),
    (gilrs::Axis::RightZ, PadAxis::RightTrigger),
];

pub fn map_mouse(b: WMouse) -> Option<MouseButton> {
    match b {
        WMouse::Left => Some(MouseButton::Left),
        WMouse::Right => Some(MouseButton::Right),
        WMouse::Middle => Some(MouseButton::Middle),
        _ => None,
    }
}

pub fn map_key(code: KeyCode) -> Key {
    use KeyCode as C;
    match code {
        C::KeyA => Key::A, C::KeyB => Key::B, C::KeyC => Key::C, C::KeyD => Key::D,
        C::KeyE => Key::E, C::KeyF => Key::F, C::KeyG => Key::G, C::KeyH => Key::H,
        C::KeyI => Key::I, C::KeyJ => Key::J, C::KeyK => Key::K, C::KeyL => Key::L,
        C::KeyM => Key::M, C::KeyN => Key::N, C::KeyO => Key::O, C::KeyP => Key::P,
        C::KeyQ => Key::Q, C::KeyR => Key::R, C::KeyS => Key::S, C::KeyT => Key::T,
        C::KeyU => Key::U, C::KeyV => Key::V, C::KeyW => Key::W, C::KeyX => Key::X,
        C::KeyY => Key::Y, C::KeyZ => Key::Z,
        C::Digit0 => Key::Num0, C::Digit1 => Key::Num1, C::Digit2 => Key::Num2,
        C::Digit3 => Key::Num3, C::Digit4 => Key::Num4, C::Digit5 => Key::Num5,
        C::Digit6 => Key::Num6, C::Digit7 => Key::Num7, C::Digit8 => Key::Num8,
        C::Digit9 => Key::Num9,
        C::F1 => Key::F1, C::F2 => Key::F2, C::F3 => Key::F3, C::F4 => Key::F4,
        C::F5 => Key::F5, C::F6 => Key::F6, C::F7 => Key::F7, C::F8 => Key::F8,
        C::F9 => Key::F9, C::F10 => Key::F10, C::F11 => Key::F11, C::F12 => Key::F12,
        C::ArrowUp => Key::Up, C::ArrowDown => Key::Down,
        C::ArrowLeft => Key::Left, C::ArrowRight => Key::Right,
        C::Space => Key::Space, C::Enter => Key::Enter, C::Escape => Key::Escape,
        C::Tab => Key::Tab, C::Backspace => Key::Backspace, C::Delete => Key::Delete,
        C::ShiftLeft => Key::LShift, C::ShiftRight => Key::RShift,
        C::ControlLeft => Key::LCtrl, C::ControlRight => Key::RCtrl,
        C::AltLeft => Key::LAlt, C::AltRight => Key::RAlt,
        C::Minus => Key::Minus, C::Equal => Key::Equals,
        C::BracketLeft => Key::LBracket, C::BracketRight => Key::RBracket,
        C::Semicolon => Key::Semicolon, C::Comma => Key::Comma,
        C::Period => Key::Period, C::Slash => Key::Slash,
        C::Backslash => Key::Backslash, C::Backquote => Key::Grave,
        _ => Key::Unknown,
    }
}

static CONFIG_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// The config folder on platforms without a user one (Android: the app's
/// internal storage). Call once at startup.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub fn set_user_config_dir(dir: PathBuf) {
    let _ = CONFIG_DIR.set(dir);
}

/// The user's config folder (`%APPDATA%`, `~/Library/Application Support`,
/// `$XDG_CONFIG_HOME` or `~/.config`). `ZESTENGINE_CONFIG_DIR` replaces it
/// (platforms without one).
pub fn user_config_dir() -> Option<PathBuf> {
    std::env::var_os("ZESTENGINE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| CONFIG_DIR.get().cloned())
        .or_else(zestengine_assets::recent_projects::platform_config_dir)
}

/// The player's binding overrides for a shipped game: `input.ron` in the
/// user's config folder for `title` (see [`user_config_dir`]).
pub fn player_bindings_path(title: &str) -> PathBuf {
    let name: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    user_config_dir().unwrap_or_else(|| PathBuf::from(".")).join(name).join("input.ron")
}

/// Player overrides made in the editor's Play mode, kept in the project's
/// `.zestengine` folder so testing a remap menu never touches the shipped game's.
pub fn editor_bindings_path(project_root: &Path) -> PathBuf {
    project_root.join(".zestengine").join("input_overrides.ron")
}
