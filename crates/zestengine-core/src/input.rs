//! Input binding system (engine-agnostic). Defines abstract actions
//! (`Move`, `Jump`, `Look`, ...) and control schemes that map physical
//! inputs (keys, mouse, gamepad, VR controllers) to them, plus the per-frame [`InputState`]
//! snapshot a component reads through `ComponentCtx`.
//!
//! This crate is winit/gilrs-free, so physical inputs are mirrored as
//! plain enums ([`Key`], [`MouseButton`], [`PadButton`], [`PadAxis`], [`VrButton`],
//! [`VrAxis`]). Rebinding, player overrides and capture live in `input_remap`. The engine
//! translates real device events into [`RawInput`] and [`resolve`]s it against the
//! active [`ControlScheme`] into an [`InputState`].

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

/// A physical keyboard key (layout-independent, mirrors winit `KeyCode`). Only
/// the commonly-bound keys are enumerated; anything else maps to [`Key::Unknown`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Key {
    /// The A key.
    A,
    /// The B key.
    B,
    /// The C key.
    C,
    /// The D key.
    D,
    /// The E key.
    E,
    /// The F key.
    F,
    /// The G key.
    G,
    /// The H key.
    H,
    /// The I key.
    I,
    /// The J key.
    J,
    /// The K key.
    K,
    /// The L key.
    L,
    /// The M key.
    M,
    /// The N key.
    N,
    /// The O key.
    O,
    /// The P key.
    P,
    /// The Q key.
    Q,
    /// The R key.
    R,
    /// The S key.
    S,
    /// The T key.
    T,
    /// The U key.
    U,
    /// The V key.
    V,
    /// The W key.
    W,
    /// The X key.
    X,
    /// The Y key.
    Y,
    /// The Z key.
    Z,
    /// The number row 0 key.
    Num0,
    /// The number row 1 key.
    Num1,
    /// The number row 2 key.
    Num2,
    /// The number row 3 key.
    Num3,
    /// The number row 4 key.
    Num4,
    /// The number row 5 key.
    Num5,
    /// The number row 6 key.
    Num6,
    /// The number row 7 key.
    Num7,
    /// The number row 8 key.
    Num8,
    /// The number row 9 key.
    Num9,
    /// The F1 function key.
    F1,
    /// The F2 function key.
    F2,
    /// The F3 function key.
    F3,
    /// The F4 function key.
    F4,
    /// The F5 function key.
    F5,
    /// The F6 function key.
    F6,
    /// The F7 function key.
    F7,
    /// The F8 function key.
    F8,
    /// The F9 function key.
    F9,
    /// The F10 function key.
    F10,
    /// The F11 function key.
    F11,
    /// The F12 function key.
    F12,
    /// The up arrow key.
    Up,
    /// The down arrow key.
    Down,
    /// The left arrow key.
    Left,
    /// The right arrow key.
    Right,
    /// The space bar.
    Space,
    /// The enter / return key.
    Enter,
    /// The escape key.
    Escape,
    /// The tab key.
    Tab,
    /// The backspace key.
    Backspace,
    /// The delete key.
    Delete,
    /// The left shift key.
    LShift,
    /// The right shift key.
    RShift,
    /// The left control key.
    LCtrl,
    /// The right control key.
    RCtrl,
    /// The left alt key.
    LAlt,
    /// The right alt key.
    RAlt,
    /// The minus / hyphen key.
    Minus,
    /// The equals key.
    Equals,
    /// The left bracket key.
    LBracket,
    /// The right bracket key.
    RBracket,
    /// The semicolon key.
    Semicolon,
    /// The comma key.
    Comma,
    /// The period key.
    Period,
    /// The slash key.
    Slash,
    /// The backslash key.
    Backslash,
    /// The grave / backtick key.
    Grave,
    /// A key outside the commonly-bound set.
    Unknown,
}

/// A mouse button.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum MouseButton {
    /// The left mouse button.
    Left,
    /// The right mouse button.
    Right,
    /// The middle mouse button (wheel press).
    Middle,
}

/// A mouse analog axis (relative motion / wheel).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum MouseAxis {
    /// Horizontal mouse motion (pixels per frame).
    X,
    /// Vertical mouse motion (pixels per frame).
    Y,
    /// Mouse wheel motion (lines / notches).
    Wheel,
}

/// A gamepad button (Xbox naming; mirrors `gilrs::Button`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum PadButton {
    /// The bottom face button (A on Xbox).
    South,
    /// The right face button (B on Xbox).
    East,
    /// The left face button (X on Xbox).
    West,
    /// The top face button (Y on Xbox).
    North,
    /// The left shoulder button.
    LeftBumper,
    /// The right shoulder button.
    RightBumper,
    /// The select / back button.
    Select,
    /// The start button.
    Start,
    /// The left stick click.
    LeftThumb,
    /// The right stick click.
    RightThumb,
    /// The D-pad up.
    DPadUp,
    /// The D-pad down.
    DPadDown,
    /// The D-pad left.
    DPadLeft,
    /// The D-pad right.
    DPadRight,
}

/// A gamepad analog axis (sticks + triggers; mirrors `gilrs::Axis`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum PadAxis {
    /// Left stick horizontal, -1..1.
    LeftStickX,
    /// Left stick vertical, -1..1.
    LeftStickY,
    /// Right stick horizontal, -1..1.
    RightStickX,
    /// Right stick vertical, -1..1.
    RightStickY,
    /// Left trigger, 0..1.
    LeftTrigger,
    /// Right trigger, 0..1.
    RightTrigger,
}

/// A VR controller button. Triggers, grips and stick directions are the
/// thresholded versions of their analog values ([`VrAxis`]); `X` / `Y` sit on
/// the left controller and `A` / `B` on the right. `Menu` is the left menu
/// button (or `Y` on runtimes without one).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum VrButton {
    /// The left controller's trigger (thresholded).
    LeftTrigger,
    /// The right controller's trigger (thresholded).
    RightTrigger,
    /// The left controller's grip (thresholded).
    LeftGrip,
    /// The right controller's grip (thresholded).
    RightGrip,
    /// The A button (right controller).
    A,
    /// The B button (right controller).
    B,
    /// The X button (left controller).
    X,
    /// The Y button (left controller).
    Y,
    /// The left stick click.
    LeftStickClick,
    /// The right stick click.
    RightStickClick,
    /// The left menu button.
    Menu,
    /// The left stick pushed up (thresholded).
    LeftStickUp,
    /// The left stick pushed down (thresholded).
    LeftStickDown,
    /// The left stick pushed left (thresholded).
    LeftStickLeft,
    /// The left stick pushed right (thresholded).
    LeftStickRight,
    /// The right stick pushed up (thresholded).
    RightStickUp,
    /// The right stick pushed down (thresholded).
    RightStickDown,
    /// The right stick pushed left (thresholded).
    RightStickLeft,
    /// The right stick pushed right (thresholded).
    RightStickRight,
}

impl VrButton {
    /// Every button, in declaration order.
    pub const ALL: [VrButton; 19] = [
        VrButton::LeftTrigger, VrButton::RightTrigger, VrButton::LeftGrip, VrButton::RightGrip,
        VrButton::A, VrButton::B, VrButton::X, VrButton::Y,
        VrButton::LeftStickClick, VrButton::RightStickClick, VrButton::Menu,
        VrButton::LeftStickUp, VrButton::LeftStickDown, VrButton::LeftStickLeft, VrButton::LeftStickRight,
        VrButton::RightStickUp, VrButton::RightStickDown, VrButton::RightStickLeft, VrButton::RightStickRight,
    ];
}

/// A VR controller analog value: thumbsticks in [-1, 1], triggers and grips in [0, 1].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum VrAxis {
    /// Left thumbstick horizontal, -1..1.
    LeftStickX,
    /// Left thumbstick vertical, -1..1.
    LeftStickY,
    /// Right thumbstick horizontal, -1..1.
    RightStickX,
    /// Right thumbstick vertical, -1..1.
    RightStickY,
    /// Left trigger, 0..1.
    LeftTrigger,
    /// Right trigger, 0..1.
    RightTrigger,
    /// Left grip, 0..1.
    LeftGrip,
    /// Right grip, 0..1.
    RightGrip,
}

impl VrAxis {
    /// Every axis, in declaration order.
    pub const ALL: [VrAxis; 8] = [
        VrAxis::LeftStickX, VrAxis::LeftStickY, VrAxis::RightStickX, VrAxis::RightStickY,
        VrAxis::LeftTrigger, VrAxis::RightTrigger, VrAxis::LeftGrip, VrAxis::RightGrip,
    ];
}

/// One physical input that can drive an action.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum InputSource {
    /// A keyboard key.
    Key(Key),
    /// A mouse button.
    Mouse(MouseButton),
    /// A mouse analog axis.
    MouseAxis(MouseAxis),
    /// A gamepad button.
    PadButton(PadButton),
    /// A gamepad analog axis.
    PadAxis(PadAxis),
    /// A VR controller button.
    VrButton(VrButton),
    /// A VR controller analog value.
    VrAxis(VrAxis),
}

impl InputSource {
    /// Short human label for the editor binding UI.
    pub fn label(&self) -> String {
        match self {
            InputSource::Key(k) => format!("{k:?}"),
            InputSource::Mouse(b) => format!("Mouse{b:?}"),
            InputSource::MouseAxis(a) => format!("Mouse{a:?}"),
            InputSource::PadButton(b) => format!("Pad{b:?}"),
            InputSource::PadAxis(a) => format!("Pad{a:?}"),
            InputSource::VrButton(b) => format!("Vr{b:?}"),
            InputSource::VrAxis(a) => format!("Vr{a:?}"),
        }
    }
}

/// The shape of an action's value.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ActionKind {
    /// On/off (Jump, Fire, Interact).
    Button,
    /// One-dimensional axis in [-1, 1] (Throttle).
    Axis1,
    /// Two-dimensional axis, each component in [-1, 1] (Move, Look).
    Axis2,
}

/// Binding of one named action to physical inputs. Composite fields let a 2D axis
/// be built from four buttons (WASD) and/or an analog stick.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ActionBinding {
    /// Action name.
    pub name: String,
    /// Value shape (button, 1D or 2D axis).
    pub kind: ActionKind,
    /// Button action: pressed if ANY of these are down.
    #[serde(default)]
    pub buttons: Vec<InputSource>,
    /// Axis1/Axis2 X: positive / negative digital contributions.
    #[serde(default)]
    pub pos_x: Vec<InputSource>,
    /// Axis1 / Axis2 X: negative digital contributions.
    #[serde(default)]
    pub neg_x: Vec<InputSource>,
    /// Axis2 Y: positive / negative digital contributions.
    #[serde(default)]
    pub pos_y: Vec<InputSource>,
    /// Axis2 Y: negative digital contributions.
    #[serde(default)]
    pub neg_y: Vec<InputSource>,
    /// Analog axis for Axis1, or X for Axis2 (e.g. a stick / mouse axis).
    #[serde(default)]
    pub analog_x: Option<InputSource>,
    /// Analog Y for Axis2.
    #[serde(default)]
    pub analog_y: Option<InputSource>,
    /// Values below this magnitude (analog) read as zero.
    #[serde(default)]
    pub deadzone: f32,
    /// Output multiplier (also use negative to invert).
    #[serde(default = "one")]
    pub scale: f32,
    /// Flip the X (Axis1 value) / Y output, e.g. an "Invert look" setting.
    #[serde(default)]
    pub invert_x: bool,
    /// Flip the Y output (e.g. an "Invert look" setting).
    #[serde(default)]
    pub invert_y: bool,
}

fn one() -> f32 {
    1.0
}

fn yes() -> bool {
    true
}

impl ActionBinding {
    /// A button action bound to these inputs.
    pub fn button(name: &str, inputs: impl IntoIterator<Item = InputSource>) -> Self {
        Self {
            name: name.to_string(),
            kind: ActionKind::Button,
            buttons: inputs.into_iter().collect(),
            ..Self::empty(name, ActionKind::Button)
        }
    }

    /// An action of `kind` with nothing bound (deadzone 0.15, scale 1).
    pub fn empty(name: &str, kind: ActionKind) -> Self {
        Self {
            name: name.to_string(),
            kind,
            buttons: Vec::new(),
            pos_x: Vec::new(),
            neg_x: Vec::new(),
            pos_y: Vec::new(),
            neg_y: Vec::new(),
            analog_x: None,
            analog_y: None,
            deadzone: 0.15,
            scale: 1.0,
            invert_x: false,
            invert_y: false,
        }
    }

    /// An analog 2D action read from a stick-like pair of axes.
    pub fn stick(name: &str, x: InputSource, y: InputSource, deadzone: f32) -> Self {
        Self { analog_x: Some(x), analog_y: Some(y), deadzone, ..Self::empty(name, ActionKind::Axis2) }
    }
}

/// A named set of bindings (e.g. "KB+Mouse", "Gamepad").
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ControlScheme {
    /// Scheme name (e.g. "KB+Mouse").
    pub name: String,
    /// The bindings, in order.
    pub actions: Vec<ActionBinding>,
}

impl ControlScheme {
    /// The binding for `name`, if bound in this scheme.
    pub fn action_mut(&mut self, name: &str) -> Option<&mut ActionBinding> {
        self.actions.iter_mut().find(|a| a.name == name)
    }
}

/// All control schemes for a project + the active one. Serialized to
/// `project.zestengine` (so it survives reloads) and editable at runtime.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Bindings {
    /// All schemes.
    pub schemes: Vec<ControlScheme>,
    /// Index of the active scheme.
    #[serde(default)]
    pub active: usize,
    /// Follow the device in use: the VR scheme while a headset runs, else
    /// whichever of keyboard + mouse / gamepad was touched last.
    #[serde(default = "yes")]
    pub auto_switch: bool,
    /// Format revision; older files are upgraded by [`Bindings::upgrade`].
    #[serde(default)]
    pub version: u32,
}

impl Default for Bindings {
    fn default() -> Self {
        Self::from_schemes(vec![
            ControlScheme::default_kb_mouse(),
            ControlScheme::default_gamepad(),
            ControlScheme::default_vr(),
        ])
    }
}

impl Bindings {
    /// The current [`Bindings::version`].
    pub const VERSION: u32 = 1;

    /// These schemes, the first active, auto-switching on.
    pub fn from_schemes(schemes: Vec<ControlScheme>) -> Self {
        Self { schemes, active: 0, auto_switch: true, version: Self::VERSION }
    }

    /// Bring bindings saved by an older version up to date: projects from
    /// before VR bindings get the default VR Controllers scheme. A scheme the
    /// user deleted later stays deleted.
    pub fn upgrade(&mut self) {
        if self.version < 1 && !self.schemes.iter().any(|s| s.device() == Some(crate::InputDevice::Vr)) {
            self.schemes.push(ControlScheme::default_vr());
        }
        self.version = Self::VERSION;
        if self.active >= self.schemes.len() {
            self.active = 0;
        }
    }

    /// The active scheme, if any.
    pub fn active_scheme(&self) -> Option<&ControlScheme> {
        self.schemes.get(self.active)
    }
    /// Mutable access to the active scheme.
    pub fn active_scheme_mut(&mut self) -> Option<&mut ControlScheme> {
        self.schemes.get_mut(self.active)
    }
}

impl ControlScheme {
    /// The default keyboard + mouse scheme: WASD move, mouse look, common verbs.
    pub fn default_kb_mouse() -> Self {
        use InputSource::*;
        let mut move_act = ActionBinding::empty("Move", ActionKind::Axis2);
        move_act.pos_x = vec![Key(self::Key::D)];
        move_act.neg_x = vec![Key(self::Key::A)];
        move_act.pos_y = vec![Key(self::Key::W)];
        move_act.neg_y = vec![Key(self::Key::S)];
        move_act.deadzone = 0.0;

        let mut look = ActionBinding::empty("Look", ActionKind::Axis2);
        look.analog_x = Some(MouseAxis(self::MouseAxis::X));
        look.analog_y = Some(MouseAxis(self::MouseAxis::Y));
        look.deadzone = 0.0;

        Self {
            name: "KB+Mouse".to_string(),
            actions: vec![
                move_act,
                look,
                ActionBinding::button("Jump", [Key(self::Key::Space)]),
                ActionBinding::button("Sprint", [Key(self::Key::LShift)]),
                ActionBinding::button("Crouch", [Key(self::Key::LCtrl)]),
                ActionBinding::button("Fire", [Mouse(self::MouseButton::Left)]),
                ActionBinding::button("Interact", [Key(self::Key::E)]),
                ActionBinding::button("Grab", [Key(self::Key::F), Mouse(self::MouseButton::Right)]),
                ActionBinding::button("Voice", [Key(self::Key::V)]),
            ],
        }
    }

    /// The default gamepad scheme: left stick move, right stick look, A jump, etc.
    pub fn default_gamepad() -> Self {
        use InputSource::*;
        let mut move_act = ActionBinding::empty("Move", ActionKind::Axis2);
        move_act.analog_x = Some(PadAxis(self::PadAxis::LeftStickX));
        move_act.analog_y = Some(PadAxis(self::PadAxis::LeftStickY));

        let mut look = ActionBinding::empty("Look", ActionKind::Axis2);
        look.analog_x = Some(PadAxis(self::PadAxis::RightStickX));
        look.analog_y = Some(PadAxis(self::PadAxis::RightStickY));
        look.scale = 8.0;

        Self {
            name: "Gamepad".to_string(),
            actions: vec![
                move_act,
                look,
                ActionBinding::button("Jump", [PadButton(self::PadButton::South)]),
                ActionBinding::button("Sprint", [PadButton(self::PadButton::LeftThumb)]),
                ActionBinding::button("Crouch", [PadButton(self::PadButton::East)]),
                ActionBinding::button("Fire", [PadAxis(self::PadAxis::RightTrigger)]),
                ActionBinding::button("Interact", [PadButton(self::PadButton::North)]),
                ActionBinding::button("Grab", [PadButton(self::PadButton::West)]),
                ActionBinding::button("Voice", [PadButton(self::PadButton::LeftBumper)]),
            ],
        }
    }

    /// The default VR controller scheme (what the VR Player reads): left
    /// stick move, right stick turn (push forward to aim a teleport), grips
    /// grab and triggers use per hand, A jump, B crouch, left stick click
    /// sprint, X interact, menu button opens the menu.
    pub fn default_vr() -> Self {
        use InputSource::VrButton as B;
        let mut turn = ActionBinding::empty("Turn", ActionKind::Axis1);
        turn.analog_x = Some(InputSource::VrAxis(VrAxis::RightStickX));
        Self {
            name: "VR Controllers".to_string(),
            actions: vec![
                ActionBinding::stick("Move", InputSource::VrAxis(VrAxis::LeftStickX), InputSource::VrAxis(VrAxis::LeftStickY), 0.1),
                turn,
                ActionBinding::button("Teleport", [B(VrButton::RightStickUp)]),
                ActionBinding::button("Grab Left", [B(VrButton::LeftGrip)]),
                ActionBinding::button("Grab Right", [B(VrButton::RightGrip)]),
                ActionBinding::button("Use Left", [B(VrButton::LeftTrigger)]),
                ActionBinding::button("Use Right", [B(VrButton::RightTrigger)]),
                ActionBinding::button("Jump", [B(VrButton::A)]),
                ActionBinding::button("Crouch", [B(VrButton::B)]),
                ActionBinding::button("Sprint", [B(VrButton::LeftStickClick)]),
                ActionBinding::button("Interact", [B(VrButton::X)]),
                ActionBinding::button("Menu", [B(VrButton::Menu)]),
                ActionBinding::button("Voice", [B(VrButton::RightStickClick)]),
            ],
        }
    }
}

impl PadAxis {
}

/// Raw per-frame device state the engine fills before resolving. Digital sets are
/// "currently down"; the engine also tracks edges for just-pressed/released.
#[derive(Default, Clone)]
pub struct RawInput {
    /// Keyboard keys currently held.
    pub keys_down: std::collections::HashSet<Key>,
    /// Mouse buttons currently held.
    pub mouse_down: std::collections::HashSet<MouseButton>,
    /// Gamepad buttons currently held.
    pub pad_down: std::collections::HashSet<PadButton>,
    /// Gamepad analog axes (non-zero values only).
    pub pad_axes: HashMap<PadAxis, f32>,
    /// Mouse motion this frame (pixels).
    pub mouse_delta: Vec2,
    /// Mouse wheel this frame (lines/notches).
    pub wheel: f32,
    /// VR controller buttons held (see [`RawInput::set_vr`]).
    pub vr_down: std::collections::HashSet<VrButton>,
    /// VR controller analog values (non-zero values only).
    pub vr_axes: HashMap<VrAxis, f32>,
}

/// Grip / stick deflection past which the thresholded VR buttons read as held.
const VR_PRESS: f32 = 0.6;

impl RawInput {
    /// Mirror this frame's VR controller state (cleared when inactive).
    pub fn set_vr(&mut self, vr: &crate::VrFrame) {
        self.vr_down.clear();
        self.vr_axes.clear();
        if !vr.active {
            return;
        }
        let axes = [
            (VrAxis::LeftStickX, vr.left_stick.x),
            (VrAxis::LeftStickY, vr.left_stick.y),
            (VrAxis::RightStickX, vr.right_stick.x),
            (VrAxis::RightStickY, vr.right_stick.y),
            (VrAxis::LeftTrigger, vr.left_trigger_value.max(vr.left_trigger as i32 as f32)),
            (VrAxis::RightTrigger, vr.right_trigger_value.max(vr.right_trigger as i32 as f32)),
            (VrAxis::LeftGrip, vr.left_grip),
            (VrAxis::RightGrip, vr.right_grip),
        ];
        for (a, v) in axes {
            if v != 0.0 {
                self.vr_axes.insert(a, v);
            }
        }
        let (l, r) = (vr.left_stick, vr.right_stick);
        let held = [
            (VrButton::LeftTrigger, vr.left_trigger),
            (VrButton::RightTrigger, vr.right_trigger),
            (VrButton::LeftGrip, vr.left_grip > VR_PRESS),
            (VrButton::RightGrip, vr.right_grip > VR_PRESS),
            (VrButton::A, vr.a),
            (VrButton::B, vr.b),
            (VrButton::X, vr.x),
            (VrButton::Y, vr.y),
            (VrButton::LeftStickClick, vr.left_stick_click),
            (VrButton::RightStickClick, vr.right_stick_click),
            (VrButton::Menu, vr.menu),
            (VrButton::LeftStickUp, l.y > VR_PRESS),
            (VrButton::LeftStickDown, l.y < -VR_PRESS),
            (VrButton::LeftStickLeft, l.x < -VR_PRESS),
            (VrButton::LeftStickRight, l.x > VR_PRESS),
            (VrButton::RightStickUp, r.y > VR_PRESS),
            (VrButton::RightStickDown, r.y < -VR_PRESS),
            (VrButton::RightStickLeft, r.x < -VR_PRESS),
            (VrButton::RightStickRight, r.x > VR_PRESS),
        ];
        self.vr_down.extend(held.into_iter().filter(|(_, on)| *on).map(|(b, _)| b));
    }

    /// Current value of one source: 0/1 for buttons, the raw analog value otherwise.
    pub fn source_value(&self, src: &InputSource) -> f32 {
        match src {
            InputSource::Key(k) => self.keys_down.contains(k) as i32 as f32,
            InputSource::Mouse(b) => self.mouse_down.contains(b) as i32 as f32,
            InputSource::PadButton(b) => self.pad_down.contains(b) as i32 as f32,
            InputSource::MouseAxis(a) => match a {
                MouseAxis::X => self.mouse_delta.x,
                MouseAxis::Y => self.mouse_delta.y,
                MouseAxis::Wheel => self.wheel,
            },
            InputSource::PadAxis(a) => self.pad_axes.get(a).copied().unwrap_or(0.0),
            InputSource::VrButton(b) => self.vr_down.contains(b) as i32 as f32,
            InputSource::VrAxis(a) => self.vr_axes.get(a).copied().unwrap_or(0.0),
        }
    }

    fn any_down(&self, srcs: &[InputSource]) -> bool {
        srcs.iter().any(|s| self.source_value(s).abs() > 0.5)
    }
}

/// The resolved value of one action this frame.
#[derive(Clone, Copy, Default)]
pub struct ActionValue {
    /// Held this frame.
    pub down: bool,
    /// Pressed this frame (edge).
    pub just_pressed: bool,
    /// Released this frame (edge).
    pub just_released: bool,
    /// Scalar value: 0/1 for a button, -1..1 for a 1D axis.
    pub value: f32,
    /// 2D axis value.
    pub axis2: Vec2,
}

/// Per-frame snapshot of every action, read by components via `ComponentCtx`.
#[derive(Default, Clone)]
pub struct InputState {
    actions: HashMap<String, ActionValue>,
    /// Raw mouse delta this frame (for free look that bypasses a bound action).
    pub mouse_delta: Vec2,
}

impl InputState {
    fn get(&self, name: &str) -> ActionValue {
        self.actions.get(name).copied().unwrap_or_default()
    }
    /// True while a button action is held.
    pub fn down(&self, name: &str) -> bool {
        self.get(name).down
    }
    /// True the frame a button action was pressed.
    pub fn pressed(&self, name: &str) -> bool {
        self.get(name).just_pressed
    }
    /// True the frame a button action was released.
    pub fn released(&self, name: &str) -> bool {
        self.get(name).just_released
    }
    /// Scalar value of a 1D-axis action (or 0/1 for a button).
    pub fn axis(&self, name: &str) -> f32 {
        self.get(name).value
    }
    /// 2D value of an axis2 action.
    pub fn axis2(&self, name: &str) -> Vec2 {
        self.get(name).axis2
    }
    /// Every action that is not at rest this frame (what a networked input
    /// frame carries).
    pub fn active_actions(&self) -> impl Iterator<Item = (&str, ActionValue)> {
        self.actions.iter().filter(|(_, v)| v.is_active()).map(|(k, v)| (k.as_str(), *v))
    }
    /// An input frame rebuilt from action values (a remote player's input
    /// replayed on the host, or a scripted frame in tests).
    pub fn from_actions(actions: impl IntoIterator<Item = (String, ActionValue)>, mouse_delta: Vec2) -> Self {
        Self { actions: actions.into_iter().collect(), mouse_delta }
    }
}

impl ActionValue {
    /// Pressed, released, held or deflected.
    pub fn is_active(&self) -> bool {
        self.down || self.just_pressed || self.just_released || self.value != 0.0 || self.axis2 != Vec2::ZERO
    }
}

fn apply_deadzone(v: f32, dz: f32) -> f32 {
    if v.abs() < dz {
        0.0
    } else {
        v
    }
}

/// Resolve raw device state against a control scheme into the action snapshot.
/// `prev` carries last frame's down-state so edges (just-pressed/released) work.
pub fn resolve(scheme: &ControlScheme, raw: &RawInput, prev: &InputState) -> InputState {
    let mut out = InputState {
        actions: HashMap::with_capacity(scheme.actions.len()),
        mouse_delta: raw.mouse_delta,
    };
    for a in &scheme.actions {
        let mut v = ActionValue::default();
        match a.kind {
            ActionKind::Button => {
                v.down = raw.any_down(&a.buttons);
                v.value = v.down as i32 as f32;
            }
            ActionKind::Axis1 => {
                let mut x = (raw.any_down(&a.pos_x) as i32 - raw.any_down(&a.neg_x) as i32) as f32;
                if let Some(an) = &a.analog_x {
                    let av = raw.source_value(an);
                    if av.abs() > x.abs() {
                        x = av;
                    }
                }
                let flip = if a.invert_x { -1.0 } else { 1.0 };
                v.value = apply_deadzone(x, a.deadzone) * a.scale * flip;
                v.down = v.value.abs() > 0.5;
            }
            ActionKind::Axis2 => {
                let mut x = (raw.any_down(&a.pos_x) as i32 - raw.any_down(&a.neg_x) as i32) as f32;
                let mut y = (raw.any_down(&a.pos_y) as i32 - raw.any_down(&a.neg_y) as i32) as f32;
                if let Some(an) = &a.analog_x {
                    let av = raw.source_value(an);
                    if av.abs() > x.abs() {
                        x = av;
                    }
                }
                if let Some(an) = &a.analog_y {
                    let av = raw.source_value(an);
                    if av.abs() > y.abs() {
                        y = av;
                    }
                }
                let mut axis = Vec2::new(x, y);
                // Mouse deltas are pixels moved this frame, not a stick
                // deflection: clamping them to a unit vector caps look speed.
                let mouse = [&a.analog_x, &a.analog_y].iter().any(|s| matches!(s, Some(InputSource::MouseAxis(_))));
                if !mouse {
                    if axis.length() > 1.0 {
                        axis = axis.normalize();
                    }
                    if axis.length() < a.deadzone {
                        axis = Vec2::ZERO;
                    }
                }
                if a.invert_x {
                    axis.x = -axis.x;
                }
                if a.invert_y {
                    axis.y = -axis.y;
                }
                v.axis2 = axis * a.scale;
                v.value = v.axis2.length();
                v.down = v.value > 0.5;
            }
        }
        let was = prev.get(&a.name).down;
        v.just_pressed = v.down && !was;
        v.just_released = !v.down && was;
        out.actions.insert(a.name.clone(), v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasd_resolves_to_move_axis_and_jump_edges() {
        let scheme = ControlScheme::default_kb_mouse();
        let mut raw = RawInput::default();
        raw.keys_down.insert(Key::W);
        raw.keys_down.insert(Key::D);
        let s = resolve(&scheme, &raw, &InputState::default());
        let mv = s.axis2("Move");
        assert!(mv.x > 0.0 && mv.y > 0.0, "W+D should move +x/+y, got {mv:?}");
        assert!((mv.length() - 1.0).abs() < 1e-4, "diagonal move is normalized");

        // Jump edge detection across frames.
        let mut raw2 = RawInput::default();
        raw2.keys_down.insert(Key::Space);
        let s2 = resolve(&scheme, &raw2, &s);
        assert!(s2.pressed("Jump") && s2.down("Jump"), "Space → Jump just-pressed");
        let s3 = resolve(&scheme, &raw2, &s2);
        assert!(!s3.pressed("Jump") && s3.down("Jump"), "held Jump not re-pressed");
        let s4 = resolve(&scheme, &RawInput::default(), &s3);
        assert!(s4.released("Jump") && !s4.down("Jump"), "release Jump");
    }

    #[test]
    fn gamepad_stick_drives_move() {
        let scheme = ControlScheme::default_gamepad();
        let mut raw = RawInput::default();
        raw.pad_axes.insert(PadAxis::LeftStickX, 0.9);
        let s = resolve(&scheme, &raw, &InputState::default());
        assert!(s.axis2("Move").x > 0.5, "left stick X drives Move.x");
    }

    #[test]
    fn mouse_look_keeps_full_pixel_delta() {
        let scheme = ControlScheme::default_kb_mouse();
        let raw = RawInput { mouse_delta: Vec2::new(30.0, -12.0), ..Default::default() };
        let s = resolve(&scheme, &raw, &InputState::default());
        assert_eq!(s.axis2("Look"), Vec2::new(30.0, -12.0), "mouse look is not clamped like a stick");
    }
}
