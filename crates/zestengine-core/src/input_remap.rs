//! Rebinding on top of the binding system: device kinds, conflict detection,
//! player overrides stored apart from the project defaults, "press a key"
//! capture and automatic scheme switching. Pure data + logic; the engine owns
//! a [`ControlsRuntime`] and games reach it through `ComponentCtx`
//! (`ctx.rebind`, `ctx.capture_next_input`, `ctx.reset_bindings`, ...).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::input::{
    ActionBinding, ActionKind, Bindings, ControlScheme, InputSource, Key, MouseAxis, MouseButton, PadAxis, PadButton,
    RawInput, VrAxis, VrButton,
};
use crate::ObjectId;

/// The kind of device a source or scheme belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
pub enum InputDevice {
    /// Keyboard and mouse.
    #[default]
    KeyboardMouse,
    /// A gamepad.
    Gamepad,
    /// VR controllers.
    Vr,
}

impl InputDevice {
    /// Every device kind, in declaration order.
    pub const ALL: [InputDevice; 3] = [InputDevice::KeyboardMouse, InputDevice::Gamepad, InputDevice::Vr];

    /// Player-facing label of this device kind.
    pub fn label(self) -> &'static str {
        match self {
            InputDevice::KeyboardMouse => "Keyboard + Mouse",
            InputDevice::Gamepad => "Gamepad",
            InputDevice::Vr => "VR Controllers",
        }
    }

    fn slot(self) -> usize {
        self as usize
    }
}

impl InputSource {
    /// The device kind this source belongs to.
    pub fn device(&self) -> InputDevice {
        match self {
            InputSource::Key(_) | InputSource::Mouse(_) | InputSource::MouseAxis(_) => InputDevice::KeyboardMouse,
            InputSource::PadButton(_) | InputSource::PadAxis(_) => InputDevice::Gamepad,
            InputSource::VrButton(_) | InputSource::VrAxis(_) => InputDevice::Vr,
        }
    }

    /// Analog sources (sticks, triggers, mouse motion) as opposed to buttons.
    pub fn is_analog(&self) -> bool {
        matches!(self, InputSource::MouseAxis(_) | InputSource::PadAxis(_) | InputSource::VrAxis(_))
    }

    /// The other half of a stick-like pair (X <-> Y), so binding one stick
    /// axis to a 2D action binds the whole stick.
    pub fn stick_pair(&self) -> Option<InputSource> {
        use InputSource as S;
        Some(match *self {
            S::MouseAxis(MouseAxis::X) => S::MouseAxis(MouseAxis::Y),
            S::MouseAxis(MouseAxis::Y) => S::MouseAxis(MouseAxis::X),
            S::PadAxis(PadAxis::LeftStickX) => S::PadAxis(PadAxis::LeftStickY),
            S::PadAxis(PadAxis::LeftStickY) => S::PadAxis(PadAxis::LeftStickX),
            S::PadAxis(PadAxis::RightStickX) => S::PadAxis(PadAxis::RightStickY),
            S::PadAxis(PadAxis::RightStickY) => S::PadAxis(PadAxis::RightStickX),
            S::VrAxis(VrAxis::LeftStickX) => S::VrAxis(VrAxis::LeftStickY),
            S::VrAxis(VrAxis::LeftStickY) => S::VrAxis(VrAxis::LeftStickX),
            S::VrAxis(VrAxis::RightStickX) => S::VrAxis(VrAxis::RightStickY),
            S::VrAxis(VrAxis::RightStickY) => S::VrAxis(VrAxis::RightStickX),
            _ => return None,
        })
    }

    /// Player-facing name ("Space", "Left Mouse", "Right Trigger").
    pub fn display_name(&self) -> String {
        match self {
            InputSource::Key(k) => key_name(*k).to_string(),
            InputSource::Mouse(b) => match b {
                MouseButton::Left => "Left Mouse",
                MouseButton::Right => "Right Mouse",
                MouseButton::Middle => "Middle Mouse",
            }
            .to_string(),
            InputSource::MouseAxis(a) => match a {
                MouseAxis::X => "Mouse X",
                MouseAxis::Y => "Mouse Y",
                MouseAxis::Wheel => "Mouse Wheel",
            }
            .to_string(),
            InputSource::PadButton(b) => pad_button_name(*b).to_string(),
            InputSource::PadAxis(a) => match a {
                PadAxis::LeftStickX => "Left Stick X",
                PadAxis::LeftStickY => "Left Stick Y",
                PadAxis::RightStickX => "Right Stick X",
                PadAxis::RightStickY => "Right Stick Y",
                PadAxis::LeftTrigger => "Left Trigger",
                PadAxis::RightTrigger => "Right Trigger",
            }
            .to_string(),
            InputSource::VrButton(b) => vr_button_name(*b).to_string(),
            InputSource::VrAxis(a) => match a {
                VrAxis::LeftStickX => "Left Stick X",
                VrAxis::LeftStickY => "Left Stick Y",
                VrAxis::RightStickX => "Right Stick X",
                VrAxis::RightStickY => "Right Stick Y",
                VrAxis::LeftTrigger => "Left Trigger",
                VrAxis::RightTrigger => "Right Trigger",
                VrAxis::LeftGrip => "Left Grip",
                VrAxis::RightGrip => "Right Grip",
            }
            .to_string(),
        }
    }
}

fn key_name(k: Key) -> &'static str {
    use Key::*;
    match k {
        A => "A", B => "B", C => "C", D => "D", E => "E", F => "F", G => "G", H => "H", I => "I",
        J => "J", K => "K", L => "L", M => "M", N => "N", O => "O", P => "P", Q => "Q", R => "R",
        S => "S", T => "T", U => "U", V => "V", W => "W", X => "X", Y => "Y", Z => "Z",
        Num0 => "0", Num1 => "1", Num2 => "2", Num3 => "3", Num4 => "4",
        Num5 => "5", Num6 => "6", Num7 => "7", Num8 => "8", Num9 => "9",
        F1 => "F1", F2 => "F2", F3 => "F3", F4 => "F4", F5 => "F5", F6 => "F6",
        F7 => "F7", F8 => "F8", F9 => "F9", F10 => "F10", F11 => "F11", F12 => "F12",
        Up => "Up Arrow", Down => "Down Arrow", Left => "Left Arrow", Right => "Right Arrow",
        Space => "Space", Enter => "Enter", Escape => "Esc", Tab => "Tab", Backspace => "Backspace", Delete => "Delete",
        LShift => "Left Shift", RShift => "Right Shift", LCtrl => "Left Ctrl", RCtrl => "Right Ctrl",
        LAlt => "Left Alt", RAlt => "Right Alt",
        Minus => "-", Equals => "=", LBracket => "[", RBracket => "]", Semicolon => ";", Comma => ",",
        Period => ".", Slash => "/", Backslash => "\\", Grave => "`",
        Unknown => "?",
    }
}

fn pad_button_name(b: PadButton) -> &'static str {
    match b {
        PadButton::South => "A Button",
        PadButton::East => "B Button",
        PadButton::West => "X Button",
        PadButton::North => "Y Button",
        PadButton::LeftBumper => "Left Bumper",
        PadButton::RightBumper => "Right Bumper",
        PadButton::Select => "Select",
        PadButton::Start => "Start",
        PadButton::LeftThumb => "Left Stick Click",
        PadButton::RightThumb => "Right Stick Click",
        PadButton::DPadUp => "D-Pad Up",
        PadButton::DPadDown => "D-Pad Down",
        PadButton::DPadLeft => "D-Pad Left",
        PadButton::DPadRight => "D-Pad Right",
    }
}

fn vr_button_name(b: VrButton) -> &'static str {
    match b {
        VrButton::LeftTrigger => "Left Trigger",
        VrButton::RightTrigger => "Right Trigger",
        VrButton::LeftGrip => "Left Grip",
        VrButton::RightGrip => "Right Grip",
        VrButton::A => "A",
        VrButton::B => "B",
        VrButton::X => "X",
        VrButton::Y => "Y",
        VrButton::LeftStickClick => "Left Stick Click",
        VrButton::RightStickClick => "Right Stick Click",
        VrButton::Menu => "Menu",
        VrButton::LeftStickUp => "Left Stick Up",
        VrButton::LeftStickDown => "Left Stick Down",
        VrButton::LeftStickLeft => "Left Stick Left",
        VrButton::LeftStickRight => "Left Stick Right",
        VrButton::RightStickUp => "Right Stick Up",
        VrButton::RightStickDown => "Right Stick Down",
        VrButton::RightStickLeft => "Right Stick Left",
        VrButton::RightStickRight => "Right Stick Right",
    }
}

impl Key {
    /// Every bindable key (not `Unknown`), in keyboard-ish order.
    pub const ALL: [Key; 74] = {
        use Key::*;
        [
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
            Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
            F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
            Up, Down, Left, Right,
            Space, Enter, Escape, Tab, Backspace, Delete,
            LShift, RShift, LCtrl, RCtrl, LAlt, RAlt,
            Minus, Equals, LBracket, RBracket, Semicolon, Comma, Period, Slash, Backslash, Grave,
        ]
    };
}

impl PadButton {
    /// Every gamepad button, in declaration order.
    pub const ALL: [PadButton; 14] = [
        PadButton::South, PadButton::East, PadButton::West, PadButton::North,
        PadButton::LeftBumper, PadButton::RightBumper, PadButton::Select, PadButton::Start,
        PadButton::LeftThumb, PadButton::RightThumb,
        PadButton::DPadUp, PadButton::DPadDown, PadButton::DPadLeft, PadButton::DPadRight,
    ];
}

impl PadAxis {
    /// Every gamepad axis, in declaration order.
    pub const ALL: [PadAxis; 6] = [
        PadAxis::LeftStickX, PadAxis::LeftStickY, PadAxis::RightStickX, PadAxis::RightStickY,
        PadAxis::LeftTrigger, PadAxis::RightTrigger,
    ];
}

/// Every source of `device` a slot of `kind` can take, for pick lists:
/// buttons (and triggers) for button slots, axes for analog slots.
pub fn all_sources(device: InputDevice, kind: CaptureKind) -> Vec<InputSource> {
    use InputSource as S;
    match (device, kind) {
        (InputDevice::KeyboardMouse, CaptureKind::Button) => Key::ALL
            .iter()
            .map(|k| S::Key(*k))
            .chain([MouseButton::Left, MouseButton::Right, MouseButton::Middle].map(S::Mouse))
            .collect(),
        (InputDevice::KeyboardMouse, CaptureKind::Axis) => [MouseAxis::X, MouseAxis::Y, MouseAxis::Wheel].map(S::MouseAxis).to_vec(),
        (InputDevice::Gamepad, CaptureKind::Button) => PadButton::ALL
            .map(S::PadButton)
            .into_iter()
            .chain([S::PadAxis(PadAxis::LeftTrigger), S::PadAxis(PadAxis::RightTrigger)])
            .collect(),
        (InputDevice::Gamepad, CaptureKind::Axis) => PadAxis::ALL.map(S::PadAxis).to_vec(),
        (InputDevice::Vr, CaptureKind::Button) => VrButton::ALL.map(S::VrButton).to_vec(),
        (InputDevice::Vr, CaptureKind::Axis) => VrAxis::ALL.map(S::VrAxis).to_vec(),
    }
}

/// Name of a full stick bound as an X / Y pair ("Left Stick", "Mouse").
pub fn stick_name(x: &InputSource, y: &InputSource) -> Option<String> {
    if x.stick_pair() != Some(*y) {
        return None;
    }
    let name = x.display_name();
    Some(match name.strip_suffix(" X") {
        Some("Mouse") | None => "Mouse".to_string(),
        Some(stick) => stick.to_string(),
    })
}

// ------------------------------------------------------------------ slots

/// Which list of an [`ActionBinding`] a source goes in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum BindingSlot {
    /// A button action's inputs.
    Buttons,
    /// Composite axis parts (digital +X / -X / +Y / -Y).
    PosX,
    /// Composite -X inputs.
    NegX,
    /// Composite +Y inputs.
    PosY,
    /// Composite -Y inputs.
    NegY,
    /// The analog axis (Axis1, or X of an Axis2).
    AnalogX,
    /// The analog Y of an Axis2.
    AnalogY,
}

impl BindingSlot {
    /// Every slot, in declaration order.
    pub const ALL: [BindingSlot; 7] = [
        BindingSlot::Buttons,
        BindingSlot::PosX,
        BindingSlot::NegX,
        BindingSlot::PosY,
        BindingSlot::NegY,
        BindingSlot::AnalogX,
        BindingSlot::AnalogY,
    ];

    /// Inspector label of this slot.
    pub fn label(self) -> &'static str {
        match self {
            BindingSlot::Buttons => "Buttons",
            BindingSlot::PosX => "+X",
            BindingSlot::NegX => "-X",
            BindingSlot::PosY => "+Y",
            BindingSlot::NegY => "-Y",
            BindingSlot::AnalogX => "Analog X",
            BindingSlot::AnalogY => "Analog Y",
        }
    }

    /// The slots an action of `kind` reads.
    pub fn for_kind(kind: ActionKind) -> &'static [BindingSlot] {
        match kind {
            ActionKind::Button => &[BindingSlot::Buttons],
            ActionKind::Axis1 => &[BindingSlot::PosX, BindingSlot::NegX, BindingSlot::AnalogX],
            ActionKind::Axis2 => &[
                BindingSlot::PosY,
                BindingSlot::NegY,
                BindingSlot::NegX,
                BindingSlot::PosX,
                BindingSlot::AnalogX,
                BindingSlot::AnalogY,
            ],
        }
    }

    /// Analog slots hold one source.
    pub fn is_analog(self) -> bool {
        matches!(self, BindingSlot::AnalogX | BindingSlot::AnalogY)
    }

    /// What a capture for this slot should listen for.
    pub fn capture_kind(self) -> CaptureKind {
        if self.is_analog() { CaptureKind::Axis } else { CaptureKind::Button }
    }
}

impl ActionBinding {
    /// The sources in one slot (0 or 1 for an analog slot).
    pub fn slot(&self, slot: BindingSlot) -> Vec<InputSource> {
        match slot {
            BindingSlot::Buttons => self.buttons.clone(),
            BindingSlot::PosX => self.pos_x.clone(),
            BindingSlot::NegX => self.neg_x.clone(),
            BindingSlot::PosY => self.pos_y.clone(),
            BindingSlot::NegY => self.neg_y.clone(),
            BindingSlot::AnalogX => self.analog_x.into_iter().collect(),
            BindingSlot::AnalogY => self.analog_y.into_iter().collect(),
        }
    }

    fn list_mut(&mut self, slot: BindingSlot) -> Option<&mut Vec<InputSource>> {
        Some(match slot {
            BindingSlot::Buttons => &mut self.buttons,
            BindingSlot::PosX => &mut self.pos_x,
            BindingSlot::NegX => &mut self.neg_x,
            BindingSlot::PosY => &mut self.pos_y,
            BindingSlot::NegY => &mut self.neg_y,
            BindingSlot::AnalogX | BindingSlot::AnalogY => return None,
        })
    }

    fn analog_mut(&mut self, slot: BindingSlot) -> Option<&mut Option<InputSource>> {
        match slot {
            BindingSlot::AnalogX => Some(&mut self.analog_x),
            BindingSlot::AnalogY => Some(&mut self.analog_y),
            _ => None,
        }
    }

    /// Whether this action reads `slot`.
    pub fn uses_slot(&self, slot: BindingSlot) -> bool {
        BindingSlot::for_kind(self.kind).contains(&slot)
    }

    /// Put `source` at `index` of `slot`, replacing what was there (an index
    /// past the end appends). Returns the replaced source.
    pub fn set_in_slot(&mut self, slot: BindingSlot, index: usize, source: InputSource) -> Option<InputSource> {
        if let Some(a) = self.analog_mut(slot) {
            return a.replace(source);
        }
        let list = self.list_mut(slot)?;
        if let Some(cur) = list.get_mut(index) {
            return Some(std::mem::replace(cur, source));
        }
        if !list.contains(&source) {
            list.push(source);
        }
        None
    }

    /// Remove the source at `index` of `slot`.
    pub fn remove_from_slot(&mut self, slot: BindingSlot, index: usize) -> Option<InputSource> {
        if let Some(a) = self.analog_mut(slot) {
            return a.take();
        }
        let list = self.list_mut(slot)?;
        (index < list.len()).then(|| list.remove(index))
    }

    /// Every (slot, index, source) this action reads.
    pub fn bound(&self) -> Vec<(BindingSlot, usize, InputSource)> {
        BindingSlot::ALL
            .iter()
            .filter(|s| self.uses_slot(**s))
            .flat_map(|s| self.slot(*s).into_iter().enumerate().map(move |(i, src)| (*s, i, src)))
            .collect()
    }

    /// True when the action reads this source anywhere.
    pub fn contains(&self, source: &InputSource) -> bool {
        self.bound().iter().any(|(_, _, s)| s == source)
    }

    /// Replace `old` with `new` (or drop it when `new` is None) everywhere in
    /// this action except `skip`. True if anything changed.
    fn substitute(&mut self, old: InputSource, new: Option<InputSource>, skip: Option<(BindingSlot, usize)>) -> bool {
        let mut changed = false;
        for slot in BindingSlot::ALL {
            if let Some(a) = self.analog_mut(slot) {
                if *a == Some(old) && skip != Some((slot, 0)) {
                    *a = new;
                    changed = true;
                }
                continue;
            }
            let Some(list) = self.list_mut(slot) else { continue };
            let mut i = 0;
            while i < list.len() {
                if list[i] == old && skip != Some((slot, i)) {
                    changed = true;
                    match new {
                        Some(n) if !list.contains(&n) => list[i] = n,
                        _ => {
                            list.remove(i);
                            continue;
                        }
                    }
                }
                i += 1;
            }
        }
        changed
    }

    /// The binding text of an analog slot pair: a stick name when X / Y are
    /// one stick, else the X source.
    pub fn analog_label(&self) -> Option<String> {
        let x = self.analog_x?;
        Some(match &self.analog_y {
            Some(y) if self.kind == ActionKind::Axis2 => stick_name(&x, y).unwrap_or_else(|| x.display_name()),
            _ => x.display_name(),
        })
    }
}

// ------------------------------------------------------------------ schemes

/// Two or more actions of one scheme bound to the same input.
#[derive(Clone, PartialEq, Debug)]
pub struct BindingConflict {
    /// The input bound to more than one action.
    pub source: InputSource,
    /// The actions sharing it.
    pub actions: Vec<String>,
}

/// One rebindable row of a controls menu: an action, or one direction of a
/// composite axis.
#[derive(Clone, PartialEq, Debug)]
pub struct RemapEntry {
    /// Action name.
    pub action: String,
    /// Which list of the action this row rebinds.
    pub slot: BindingSlot,
    /// Row label shown in the menu.
    pub label: String,
}

impl ControlScheme {
    /// The binding for `name`, if bound in this scheme.
    pub fn action(&self, name: &str) -> Option<&ActionBinding> {
        self.actions.iter().find(|a| a.name == name)
    }

    /// The device this scheme is for, from what its actions are bound to
    /// (the device with the most sources). None for an empty scheme.
    pub fn device(&self) -> Option<InputDevice> {
        let mut counts = [0usize; 3];
        for a in &self.actions {
            for (_, _, s) in a.bound() {
                counts[s.device().slot()] += 1;
            }
        }
        let best = (0..3).max_by_key(|&i| (counts[i], 2 - i))?;
        (counts[best] > 0).then_some(InputDevice::ALL[best])
    }

    /// Actions that read `source`.
    pub fn actions_using(&self, source: &InputSource) -> Vec<String> {
        self.actions.iter().filter(|a| a.contains(source)).map(|a| a.name.clone()).collect()
    }

    /// Every input bound to more than one action.
    pub fn conflicts(&self) -> Vec<BindingConflict> {
        let mut seen: Vec<InputSource> = Vec::new();
        let mut out = Vec::new();
        for a in &self.actions {
            for (_, _, s) in a.bound() {
                if seen.contains(&s) {
                    continue;
                }
                seen.push(s);
                let actions = self.actions_using(&s);
                if actions.len() > 1 {
                    out.push(BindingConflict { source: s, actions });
                }
            }
        }
        out
    }

    /// The rows a controls menu lists: one per button action, one for an
    /// analog axis, and one per direction of a keyed (composite) axis.
    pub fn remap_entries(&self) -> Vec<RemapEntry> {
        let mut out = Vec::new();
        let mut push = |a: &ActionBinding, slot, label: String| out.push(RemapEntry { action: a.name.clone(), slot, label });
        for a in &self.actions {
            match a.kind {
                ActionKind::Button => push(a, BindingSlot::Buttons, a.name.clone()),
                ActionKind::Axis1 => {
                    if a.analog_x.is_some() {
                        push(a, BindingSlot::AnalogX, a.name.clone());
                    }
                    if a.analog_x.is_none() || !a.pos_x.is_empty() || !a.neg_x.is_empty() {
                        push(a, BindingSlot::PosX, format!("{} +", a.name));
                        push(a, BindingSlot::NegX, format!("{} -", a.name));
                    }
                }
                ActionKind::Axis2 => {
                    if a.analog_x.is_some() {
                        push(a, BindingSlot::AnalogX, a.name.clone());
                    }
                    let keyed = !(a.pos_x.is_empty() && a.neg_x.is_empty() && a.pos_y.is_empty() && a.neg_y.is_empty());
                    if a.analog_x.is_none() || keyed {
                        push(a, BindingSlot::PosY, format!("{} Up", a.name));
                        push(a, BindingSlot::NegY, format!("{} Down", a.name));
                        push(a, BindingSlot::NegX, format!("{} Left", a.name));
                        push(a, BindingSlot::PosX, format!("{} Right", a.name));
                    }
                }
            }
        }
        out
    }
}

/// What to do when a rebind picks an input another action already uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// Keep both (the input drives both actions).
    Allow,
    /// Take it away from the other actions.
    Steal,
    /// Give the other actions the input this slot had (or remove it if empty).
    #[default]
    Swap,
}

impl ConflictPolicy {
    /// Every policy, in declaration order.
    pub const ALL: [ConflictPolicy; 3] = [ConflictPolicy::Swap, ConflictPolicy::Steal, ConflictPolicy::Allow];

    /// Inspector label of this policy.
    pub fn label(self) -> &'static str {
        match self {
            ConflictPolicy::Allow => "Allow duplicates",
            ConflictPolicy::Steal => "Unbind the other action",
            ConflictPolicy::Swap => "Swap with the other action",
        }
    }
}

/// Why a rebind was rejected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RebindError {
    /// No scheme with this name (empty = the active one).
    NoScheme(String),
    /// No action with this name in the scheme.
    NoAction(String),
    /// The action's kind doesn't read that slot (Buttons on an axis, ...).
    WrongSlot(BindingSlot),
}

impl std::fmt::Display for RebindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebindError::NoScheme(s) => write!(f, "no control scheme {s:?}"),
            RebindError::NoAction(a) => write!(f, "no action {a:?}"),
            RebindError::WrongSlot(s) => write!(f, "the action has no {} slot", s.label()),
        }
    }
}

/// Analog shaping for an action; `None` fields are left as they are.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct ActionTuning {
    /// Set the deadzone (clamped to 0..1).
    pub deadzone: Option<f32>,
    /// Set the output scale.
    pub scale: Option<f32>,
    /// Set the X inversion.
    pub invert_x: Option<bool>,
    /// Set the Y inversion.
    pub invert_y: Option<bool>,
}

impl ActionTuning {
    /// Apply the set fields to a binding.
    pub fn apply(&self, a: &mut ActionBinding) {
        if let Some(d) = self.deadzone {
            a.deadzone = d.clamp(0.0, 1.0);
        }
        if let Some(s) = self.scale {
            a.scale = s;
        }
        if let Some(i) = self.invert_x {
            a.invert_x = i;
        }
        if let Some(i) = self.invert_y {
            a.invert_y = i;
        }
    }
}

impl Bindings {
    /// Index of a scheme by name; an empty name is the active scheme.
    pub fn scheme_index(&self, name: &str) -> Option<usize> {
        if name.is_empty() {
            return (self.active < self.schemes.len()).then_some(self.active);
        }
        self.schemes.iter().position(|s| s.name == name)
    }

    /// The scheme with this name (empty = the active one).
    pub fn scheme(&self, name: &str) -> Option<&ControlScheme> {
        self.scheme_index(name).map(|i| &self.schemes[i])
    }

    /// Mutable access to the scheme with this name (empty = the active one).
    pub fn scheme_mut(&mut self, name: &str) -> Option<&mut ControlScheme> {
        self.scheme_index(name).map(|i| &mut self.schemes[i])
    }

    /// Make the named scheme active. False if there is none.
    pub fn set_active_scheme(&mut self, name: &str) -> bool {
        match self.scheme_index(name) {
            Some(i) => {
                self.active = i;
                true
            }
            None => false,
        }
    }

    /// Bind `source` at `index` of `slot` on `action` in `scheme` ("" = the
    /// active one), replacing what was there. Binding a stick axis to the
    /// analog X of a 2D action also binds its Y partner. Other actions using
    /// the input are handled by `policy`; returns the actions it changed.
    pub fn rebind(
        &mut self,
        scheme: &str,
        action: &str,
        slot: BindingSlot,
        index: usize,
        source: InputSource,
        policy: ConflictPolicy,
    ) -> Result<Vec<String>, RebindError> {
        let si = self.scheme_index(scheme).ok_or_else(|| RebindError::NoScheme(scheme.to_string()))?;
        let sch = &mut self.schemes[si];
        let ai = sch.actions.iter().position(|a| a.name == action).ok_or_else(|| RebindError::NoAction(action.to_string()))?;
        if !sch.actions[ai].uses_slot(slot) {
            return Err(RebindError::WrongSlot(slot));
        }
        let mut writes = vec![(slot, index, source)];
        if slot == BindingSlot::AnalogX
            && sch.actions[ai].kind == ActionKind::Axis2
            && let Some(pair) = source.stick_pair()
        {
            writes.push((BindingSlot::AnalogY, 0, pair));
        }
        let mut changed = Vec::new();
        for (slot, index, source) in writes {
            let old = sch.actions[ai].slot(slot).get(index).copied();
            if old == Some(source) {
                continue;
            }
            if policy != ConflictPolicy::Allow {
                let replacement = if policy == ConflictPolicy::Swap { old } else { None };
                for (i, other) in sch.actions.iter_mut().enumerate() {
                    if i != ai && other.substitute(source, replacement, None) && !changed.contains(&other.name) {
                        changed.push(other.name.clone());
                    }
                }
                // The same input elsewhere on this action.
                let index = index.min(sch.actions[ai].slot(slot).len());
                sch.actions[ai].substitute(source, replacement, Some((slot, index)));
            }
            sch.actions[ai].set_in_slot(slot, index, source);
        }
        Ok(changed)
    }

    /// Remove the source at `index` of `slot`.
    pub fn unbind(&mut self, scheme: &str, action: &str, slot: BindingSlot, index: usize) -> Result<(), RebindError> {
        let a = self.action_mut_in(scheme, action)?;
        a.remove_from_slot(slot, index);
        Ok(())
    }

    /// Change an action's deadzone / scale / inversion.
    pub fn tune(&mut self, scheme: &str, action: &str, tuning: ActionTuning) -> Result<(), RebindError> {
        tuning.apply(self.action_mut_in(scheme, action)?);
        Ok(())
    }

    fn action_mut_in(&mut self, scheme: &str, action: &str) -> Result<&mut ActionBinding, RebindError> {
        let s = self.scheme_mut(scheme).ok_or_else(|| RebindError::NoScheme(scheme.to_string()))?;
        s.action_mut(action).ok_or_else(|| RebindError::NoAction(action.to_string()))
    }

    /// Put one action back to its binding in `defaults`.
    pub fn reset_action(&mut self, defaults: &Bindings, scheme: &str, action: &str) -> Result<(), RebindError> {
        let name = self.scheme(scheme).map(|s| s.name.clone()).ok_or_else(|| RebindError::NoScheme(scheme.to_string()))?;
        let def = defaults
            .schemes
            .iter()
            .find(|s| s.name == name)
            .and_then(|s| s.action(action))
            .cloned()
            .ok_or_else(|| RebindError::NoAction(action.to_string()))?;
        *self.action_mut_in(&name, action)? = def;
        Ok(())
    }

    /// Put one scheme back to its bindings in `defaults`.
    pub fn reset_scheme(&mut self, defaults: &Bindings, scheme: &str) -> Result<(), RebindError> {
        let si = self.scheme_index(scheme).ok_or_else(|| RebindError::NoScheme(scheme.to_string()))?;
        let def = defaults
            .schemes
            .iter()
            .find(|s| s.name == self.schemes[si].name)
            .cloned()
            .ok_or_else(|| RebindError::NoScheme(scheme.to_string()))?;
        self.schemes[si] = def;
        Ok(())
    }

    /// The scheme to switch to for `device` (None = keep the current one):
    /// the current scheme if it already fits, else the first that does.
    pub fn scheme_for(&self, device: InputDevice) -> Option<usize> {
        if self.active_scheme().and_then(|s| s.device()) == Some(device) {
            return None;
        }
        self.schemes.iter().position(|s| s.device() == Some(device))
    }
}

// ------------------------------------------------------------------ overrides

/// One action as the player rebound it.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ActionOverride {
    /// Scheme name.
    pub scheme: String,
    /// The action as the player rebound it.
    pub binding: ActionBinding,
}

/// The player's changes to the project's bindings, saved in a user config
/// file next to (not inside) the game. Only actions that differ from the
/// project defaults are stored, so actions a game update adds or changes
/// still come through.
#[derive(Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BindingOverrides {
    /// Rebound actions, in scheme order.
    pub actions: Vec<ActionOverride>,
    /// A scheme the player picked by hand.
    pub active_scheme: Option<String>,
    /// The player's auto-switch choice (None = the project's).
    pub auto_switch: Option<bool>,
}

impl BindingOverrides {
    /// The actions of `current` that differ from `defaults`, keeping this
    /// value's scheme / auto-switch choices.
    pub fn diff(&self, defaults: &Bindings, current: &Bindings) -> Self {
        let mut actions = Vec::new();
        for s in &current.schemes {
            let def = defaults.schemes.iter().find(|d| d.name == s.name);
            for a in &s.actions {
                if def.and_then(|d| d.action(&a.name)) != Some(a) {
                    actions.push(ActionOverride { scheme: s.name.clone(), binding: a.clone() });
                }
            }
        }
        Self { actions, active_scheme: self.active_scheme.clone(), auto_switch: self.auto_switch }
    }

    /// `defaults` with these overrides on top. Overrides for actions the
    /// project no longer has (or whose kind changed) are skipped.
    pub fn apply(&self, defaults: &Bindings) -> Bindings {
        let mut out = defaults.clone();
        for o in &self.actions {
            if let Some(a) = out
                .schemes
                .iter_mut()
                .find(|s| s.name == o.scheme)
                .and_then(|s| s.action_mut(&o.binding.name))
                && a.kind == o.binding.kind
            {
                *a = o.binding.clone();
            }
        }
        if let Some(name) = &self.active_scheme {
            out.set_active_scheme(name);
        }
        if let Some(on) = self.auto_switch {
            out.auto_switch = on;
        }
        out
    }

    /// True when nothing is overridden.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty() && self.active_scheme.is_none() && self.auto_switch.is_none()
    }

    /// Serialize to RON.
    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap_or_default()
    }

    /// Parse RON.
    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    /// Read from disk; a missing file is no overrides.
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(s) => Self::from_ron(&s).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// Write to disk (atomically); empty overrides remove the file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        if self.is_empty() {
            return match std::fs::remove_file(path) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(err(e)),
                _ => Ok(()),
            };
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(err)?;
        }
        let tmp = path.with_extension("ron.tmp");
        std::fs::write(&tmp, self.to_ron()).map_err(err)?;
        std::fs::rename(&tmp, path).map_err(err)
    }
}

// ------------------------------------------------------------------ capture

/// What a capture listens for: buttons (keys, mouse / pad / VR buttons,
/// triggers) or analog axes (sticks, triggers, mouse motion).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CaptureKind {
    /// Buttons: keys, mouse / pad / VR buttons, triggers.
    #[default]
    Button,
    /// Analog axes: sticks, triggers, mouse motion.
    Axis,
}

/// A request to capture the next input the player presses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CaptureRequest {
    /// What to listen for.
    pub kind: CaptureKind,
    /// Only accept inputs of this device (None = any).
    pub device: Option<InputDevice>,
}

/// The outcome of an input capture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CaptureResult {
    /// The player pressed this input.
    Captured(InputSource),
    /// Escape was pressed (or the capture was cancelled).
    Cancelled,
}

/// Mouse motion (pixels in one frame) that counts as moving a mouse axis.
const MOUSE_CAPTURE_PX: f32 = 12.0;

/// The inputs that are pressed / deflected in `raw`, as sources of `kind`.
pub fn active_sources(raw: &RawInput, kind: CaptureKind) -> Vec<InputSource> {
    let mut out = Vec::new();
    match kind {
        CaptureKind::Button => {
            out.extend(raw.keys_down.iter().filter(|k| **k != Key::Unknown).map(|k| InputSource::Key(*k)));
            out.extend(raw.mouse_down.iter().map(|b| InputSource::Mouse(*b)));
            out.extend(raw.pad_down.iter().map(|b| InputSource::PadButton(*b)));
            out.extend(raw.pad_axes.iter().filter(|(_, v)| v.abs() > 0.5).map(|(a, _)| InputSource::PadAxis(*a)));
            out.extend(raw.vr_down.iter().map(|b| InputSource::VrButton(*b)));
        }
        CaptureKind::Axis => {
            if raw.mouse_delta.x.abs() > MOUSE_CAPTURE_PX && raw.mouse_delta.x.abs() >= raw.mouse_delta.y.abs() {
                out.push(InputSource::MouseAxis(MouseAxis::X));
            } else if raw.mouse_delta.y.abs() > MOUSE_CAPTURE_PX {
                out.push(InputSource::MouseAxis(MouseAxis::Y));
            }
            if raw.wheel != 0.0 {
                out.push(InputSource::MouseAxis(MouseAxis::Wheel));
            }
            out.extend(raw.pad_axes.iter().filter(|(_, v)| v.abs() > 0.5).map(|(a, _)| InputSource::PadAxis(*a)));
            out.extend(raw.vr_axes.iter().filter(|(_, v)| v.abs() > 0.5).map(|(a, _)| InputSource::VrAxis(*a)));
        }
    }
    // Hash-set order varies; keep the pick stable.
    out.sort_by_key(|s| format!("{s:?}"));
    out
}

/// A running "press any input" capture. Inputs already held when it starts
/// (the click or button that opened it) are ignored until released.
#[derive(Clone, Debug)]
pub struct InputCapture {
    /// What the capture listens for.
    pub request: CaptureRequest,
    held: Option<HashSet<InputSource>>,
}

impl InputCapture {
    /// Start a capture.
    pub fn new(request: CaptureRequest) -> Self {
        Self { request, held: None }
    }

    /// Feed one frame of raw input; returns the result once there is one.
    pub fn poll(&mut self, raw: &RawInput) -> Option<CaptureResult> {
        let escape = InputSource::Key(Key::Escape);
        let now: Vec<InputSource> = active_sources(raw, self.request.kind)
            .into_iter()
            .chain(raw.keys_down.contains(&Key::Escape).then_some(escape))
            .filter(|s| *s == escape || self.request.device.is_none_or(|d| s.device() == d))
            .collect();
        let Some(held) = self.held.as_mut() else {
            self.held = Some(now.into_iter().collect());
            return None;
        };
        held.retain(|s| now.contains(s));
        let fresh = now.iter().find(|s| !held.contains(s))?;
        Some(if *fresh == escape { CaptureResult::Cancelled } else { CaptureResult::Captured(*fresh) })
    }
}

/// Which devices are in use in `raw` (indexed like [`InputDevice::ALL`]).
pub fn device_activity(raw: &RawInput) -> [bool; 3] {
    let kb = !raw.keys_down.is_empty() || !raw.mouse_down.is_empty() || raw.mouse_delta.length() > 3.0 || raw.wheel != 0.0;
    let pad = !raw.pad_down.is_empty() || raw.pad_axes.values().any(|v| v.abs() > 0.35);
    let vr = !raw.vr_down.is_empty() || raw.vr_axes.values().any(|v| v.abs() > 0.35);
    [kb, pad, vr]
}

// ------------------------------------------------------------------ runtime

/// A binding change requested by gameplay (`ComponentCommand::Input`).
/// Scheme names may be empty for the active scheme.
#[derive(Clone, Debug)]
pub enum InputCommand {
    /// Bind an input on an action.
    Rebind {
        /// Scheme name (empty = the active one).
        scheme: String,
        /// Action name.
        action: String,
        /// Which list of the action.
        slot: BindingSlot,
        /// Position in the list.
        index: usize,
        /// The new input.
        source: InputSource,
        /// How conflicts with other actions resolve.
        policy: ConflictPolicy,
    },
    /// Remove one bound input.
    Unbind {
        /// Scheme name (empty = the active one).
        scheme: String,
        /// Action name.
        action: String,
        /// Which list of the action.
        slot: BindingSlot,
        /// Position in the list.
        index: usize,
    },
    /// Change an action's deadzone / scale / inversion.
    Tune {
        /// Scheme name (empty = the active one).
        scheme: String,
        /// Action name.
        action: String,
        /// Which fields to change.
        tuning: ActionTuning,
    },
    /// Put one action back to its defaults.
    ResetAction {
        /// Scheme name (empty = the active one).
        scheme: String,
        /// Action name.
        action: String,
    },
    /// Put a whole scheme back to its defaults.
    ResetScheme(String),
    /// Drop every player override.
    ResetAll,
    /// Pick a scheme by hand (turns auto-switching off).
    SetScheme(String),
    /// Turn automatic scheme switching on or off.
    SetAutoSwitch(bool),
    /// Ask the player to press an input, delivered to `owner`.
    Capture {
        /// The object to deliver the result to.
        owner: ObjectId,
        /// What to listen for.
        request: CaptureRequest,
    },
    /// Cancel the object's pending capture.
    CancelCapture(ObjectId),
}

/// What gameplay sees of the live bindings this frame (`ctx.screen.controls`).
#[derive(Clone, Default, Debug)]
pub struct ControlsInfo {
    /// The bindings resolving this frame.
    pub bindings: Arc<Bindings>,
    /// The device used last (VR while a headset runs).
    pub device: InputDevice,
    /// The object whose capture is waiting for input.
    pub capturing: Option<ObjectId>,
    /// A capture that finished this frame, and who asked for it.
    pub captured: Option<(ObjectId, CaptureResult)>,
}

/// Live bindings for a running game: the project defaults, the player's
/// overrides on top (persisted to `overrides_path` on every change), the
/// last-used device with auto scheme switching, and the pending capture.
pub struct ControlsRuntime {
    defaults: Bindings,
    live: Arc<Bindings>,
    overrides: BindingOverrides,
    path: Option<PathBuf>,
    capture: Option<(ObjectId, InputCapture)>,
    captured: Option<(ObjectId, CaptureResult)>,
    /// Inputs held when a capture ended; gameplay waits for their release.
    settle: Vec<InputSource>,
    device: InputDevice,
    device_held: [bool; 3],
}

impl ControlsRuntime {
    /// Project bindings (upgraded if old), no overrides yet.
    pub fn new(mut defaults: Bindings) -> Self {
        defaults.upgrade();
        Self {
            live: Arc::new(defaults.clone()),
            defaults,
            overrides: BindingOverrides::default(),
            path: None,
            capture: None,
            captured: None,
            settle: Vec::new(),
            device: InputDevice::KeyboardMouse,
            device_held: [false; 3],
        }
    }

    /// Load (and from now on save) player overrides at `path`.
    pub fn set_overrides_path(&mut self, path: Option<PathBuf>) {
        self.path = path;
        self.overrides = match &self.path {
            Some(p) => BindingOverrides::load(p).unwrap_or_else(|e| {
                tracing::warn!("input overrides: {e}");
                BindingOverrides::default()
            }),
            None => BindingOverrides::default(),
        };
        self.rebuild();
    }

    /// Where player overrides are loaded from and saved to.
    pub fn overrides_path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Swap in new project defaults (the editor's bindings changed),
    /// keeping the player's overrides on top.
    pub fn set_defaults(&mut self, mut defaults: Bindings) {
        defaults.upgrade();
        self.defaults = defaults;
        self.rebuild();
    }

    fn rebuild(&mut self) {
        let mut live = self.overrides.apply(&self.defaults);
        live.active = if self.overrides.active_scheme.is_some() { live.active } else { self.live.active.min(live.schemes.len().saturating_sub(1)) };
        self.live = Arc::new(live);
    }

    /// The bindings that resolve this frame.
    pub fn bindings(&self) -> &Bindings {
        &self.live
    }

    /// The project's default bindings.
    pub fn defaults(&self) -> &Bindings {
        &self.defaults
    }

    /// The player's overrides.
    pub fn overrides(&self) -> &BindingOverrides {
        &self.overrides
    }

    /// The device used last (VR while a headset runs).
    pub fn device(&self) -> InputDevice {
        self.device
    }

    /// A capture is waiting for input, or the input that ended it is still
    /// held: gameplay should not see it.
    pub fn capturing(&self) -> bool {
        self.capture.is_some() || self.captured.is_some() || !self.settle.is_empty()
    }

    /// Per frame, before resolving: track the device in use, auto-switch the
    /// scheme, and run a pending capture.
    pub fn frame(&mut self, raw: &RawInput, vr_active: bool) {
        self.captured = None;
        let held = || {
            let mut v = active_sources(raw, CaptureKind::Button);
            v.extend(active_sources(raw, CaptureKind::Axis));
            v
        };
        if !self.settle.is_empty() {
            let now = held();
            self.settle.retain(|s| now.contains(s));
        }
        if let Some((owner, cap)) = self.capture.as_mut()
            && let Some(result) = cap.poll(raw)
        {
            self.captured = Some((*owner, result));
            self.capture = None;
            self.settle = held();
        }
        let active = device_activity(raw);
        for (i, on) in active.iter().enumerate() {
            if *on && !self.device_held[i] {
                self.device = InputDevice::ALL[i];
            }
        }
        self.device_held = active;
        if vr_active {
            self.device = InputDevice::Vr;
        } else if self.device == InputDevice::Vr {
            self.device = InputDevice::KeyboardMouse;
        }
        if self.live.auto_switch
            && let Some(i) = self.live.scheme_for(self.device)
        {
            Arc::make_mut(&mut self.live).active = i;
        }
    }

    /// Apply a gameplay request; binding changes are saved to the overrides file.
    pub fn apply(&mut self, cmd: InputCommand) -> Result<(), String> {
        let defaults = &self.defaults;
        let live = Arc::make_mut(&mut self.live);
        let changed = match cmd {
            InputCommand::Rebind { scheme, action, slot, index, source, policy } => {
                live.rebind(&scheme, &action, slot, index, source, policy).map_err(|e| e.to_string())?;
                true
            }
            InputCommand::Unbind { scheme, action, slot, index } => {
                live.unbind(&scheme, &action, slot, index).map_err(|e| e.to_string())?;
                true
            }
            InputCommand::Tune { scheme, action, tuning } => {
                live.tune(&scheme, &action, tuning).map_err(|e| e.to_string())?;
                true
            }
            InputCommand::ResetAction { scheme, action } => {
                live.reset_action(defaults, &scheme, &action).map_err(|e| e.to_string())?;
                true
            }
            InputCommand::ResetScheme(scheme) => {
                live.reset_scheme(defaults, &scheme).map_err(|e| e.to_string())?;
                true
            }
            InputCommand::ResetAll => {
                let active = live.active;
                *live = defaults.clone();
                live.active = active.min(live.schemes.len().saturating_sub(1));
                self.overrides = BindingOverrides::default();
                true
            }
            InputCommand::SetScheme(name) => {
                if !live.set_active_scheme(&name) {
                    return Err(RebindError::NoScheme(name).to_string());
                }
                live.auto_switch = false;
                self.overrides.active_scheme = live.active_scheme().map(|s| s.name.clone());
                self.overrides.auto_switch = Some(false);
                true
            }
            InputCommand::SetAutoSwitch(on) => {
                live.auto_switch = on;
                self.overrides.auto_switch = Some(on);
                if on {
                    self.overrides.active_scheme = None;
                }
                true
            }
            InputCommand::Capture { owner, request } => {
                self.capture = Some((owner, InputCapture::new(request)));
                false
            }
            InputCommand::CancelCapture(owner) => {
                if self.capture.as_ref().is_some_and(|(o, _)| *o == owner) {
                    self.capture = None;
                    self.captured = Some((owner, CaptureResult::Cancelled));
                }
                false
            }
        };
        if changed {
            self.overrides = self.overrides.diff(&self.defaults, &self.live);
            if let Some(p) = &self.path {
                self.overrides.save(p)?;
            }
        }
        Ok(())
    }

    /// The view gameplay reads this frame.
    pub fn info(&self) -> ControlsInfo {
        ControlsInfo {
            bindings: self.live.clone(),
            device: self.device,
            capturing: self.capture.as_ref().map(|(o, _)| *o),
            captured: self.captured,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{resolve, InputState};

    fn vr_frame() -> crate::VrFrame {
        crate::VrFrame { active: true, ..Default::default() }
    }

    #[test]
    fn vr_sources_drive_the_default_vr_scheme() {
        let scheme = ControlScheme::default_vr();
        assert_eq!(scheme.device(), Some(InputDevice::Vr));
        let mut vr = vr_frame();
        vr.left_stick = glam::Vec2::new(0.0, 0.8);
        vr.right_stick = glam::Vec2::new(0.9, 0.0);
        vr.right_grip = 0.9;
        vr.left_trigger = true;
        vr.left_trigger_value = 1.0;
        vr.a = true;
        let mut raw = RawInput::default();
        raw.set_vr(&vr);
        let s = resolve(&scheme, &raw, &InputState::default());
        assert!(s.axis2("Move").y > 0.7, "left stick moves");
        assert!(s.axis("Turn") > 0.8, "right stick X turns");
        assert!(s.down("Grab Right") && !s.down("Grab Left"), "grip past the threshold grabs");
        assert!(s.down("Use Left") && s.down("Jump"));
        assert!(!s.down("Teleport"), "stick sideways is not a teleport");
        vr.right_stick = glam::Vec2::new(0.0, 0.7);
        raw.set_vr(&vr);
        assert!(resolve(&scheme, &raw, &s).pressed("Teleport"), "right stick forward aims a teleport");
        raw.set_vr(&crate::VrFrame::default());
        assert!(raw.vr_down.is_empty() && raw.vr_axes.is_empty(), "inactive VR reads nothing");
    }

    #[test]
    fn rebind_replaces_and_reports_conflicts() {
        let mut b = Bindings::default();
        let kb = ControlScheme::default_kb_mouse().name;
        let space = InputSource::Key(Key::Space);
        let e = InputSource::Key(Key::E);
        b.rebind(&kb, "Interact", BindingSlot::Buttons, 0, space, ConflictPolicy::Allow).unwrap();
        let conflicts = b.scheme(&kb).unwrap().conflicts();
        assert_eq!(conflicts, vec![BindingConflict { source: space, actions: vec!["Jump".into(), "Interact".into()] }]);

        // Swap: Jump takes Interact's old key.
        let mut b = Bindings::default();
        let changed = b.rebind(&kb, "Interact", BindingSlot::Buttons, 0, space, ConflictPolicy::Swap).unwrap();
        assert_eq!(changed, vec!["Jump".to_string()]);
        let s = b.scheme(&kb).unwrap();
        assert_eq!(s.action("Jump").unwrap().buttons, vec![e]);
        assert_eq!(s.action("Interact").unwrap().buttons, vec![space]);
        assert!(s.conflicts().is_empty());

        // Steal: Jump loses it.
        let mut b = Bindings::default();
        b.rebind(&kb, "Interact", BindingSlot::Buttons, 0, space, ConflictPolicy::Steal).unwrap();
        assert!(b.scheme(&kb).unwrap().action("Jump").unwrap().buttons.is_empty());

        // Past the end appends a second binding; wrong slot / names fail.
        let mut b = Bindings::default();
        b.rebind("", "Jump", BindingSlot::Buttons, 5, InputSource::Key(Key::J), ConflictPolicy::Allow).unwrap();
        assert_eq!(b.active_scheme().unwrap().action("Jump").unwrap().buttons.len(), 2);
        assert_eq!(b.rebind("", "Move", BindingSlot::Buttons, 0, space, ConflictPolicy::Allow), Err(RebindError::WrongSlot(BindingSlot::Buttons)));
        assert!(matches!(b.rebind("", "Nope", BindingSlot::Buttons, 0, space, ConflictPolicy::Allow), Err(RebindError::NoAction(_))));
        assert!(matches!(b.rebind("Nope", "Jump", BindingSlot::Buttons, 0, space, ConflictPolicy::Allow), Err(RebindError::NoScheme(_))));
    }

    #[test]
    fn rebinding_a_stick_axis_binds_the_whole_stick_and_swaps() {
        let mut b = Bindings::default();
        let pad = "Gamepad";
        b.rebind(pad, "Move", BindingSlot::AnalogX, 0, InputSource::PadAxis(PadAxis::RightStickX), ConflictPolicy::Swap)
            .unwrap();
        let s = b.scheme(pad).unwrap();
        let mv = s.action("Move").unwrap();
        assert_eq!((mv.analog_x, mv.analog_y), (Some(InputSource::PadAxis(PadAxis::RightStickX)), Some(InputSource::PadAxis(PadAxis::RightStickY))));
        let look = s.action("Look").unwrap();
        assert_eq!(look.analog_label().as_deref(), Some("Left Stick"), "Look got the left stick");
        assert!(s.conflicts().is_empty());
    }

    #[test]
    fn overrides_round_trip_and_survive_project_changes() {
        let defaults = Bindings::default();
        let mut live = defaults.clone();
        live.rebind("KB+Mouse", "Jump", BindingSlot::Buttons, 0, InputSource::Key(Key::J), ConflictPolicy::Swap).unwrap();
        live.tune("Gamepad", "Look", ActionTuning { invert_y: Some(true), ..Default::default() }).unwrap();
        let o = BindingOverrides { active_scheme: Some("Gamepad".into()), ..Default::default() }.diff(&defaults, &live);
        assert_eq!(o.actions.len(), 2, "only changed actions are stored");

        let dir = std::env::temp_dir().join(format!("zestengine-input-overrides-{}", std::process::id()));
        let path = dir.join("input.ron");
        o.save(&path).unwrap();
        let back = BindingOverrides::load(&path).unwrap();
        assert_eq!(back, o);
        let applied = back.apply(&defaults);
        assert_eq!(applied.schemes, live.schemes);
        assert_eq!(applied.active_scheme().unwrap().name, "Gamepad");

        // A game update that adds an action keeps it; removed actions are skipped.
        let mut updated = defaults.clone();
        updated.schemes[0].actions.retain(|a| a.name != "Jump");
        updated.schemes[0].actions.push(ActionBinding::button("Dash", [InputSource::Key(Key::Q)]));
        let applied = back.apply(&updated);
        assert!(applied.schemes[0].action("Dash").is_some() && applied.schemes[0].action("Jump").is_none());

        BindingOverrides::default().save(&path).unwrap();
        assert!(!path.exists(), "no overrides, no file");
        assert_eq!(BindingOverrides::load(&path).unwrap(), BindingOverrides::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_persists_player_changes_apart_from_defaults() {
        let dir = std::env::temp_dir().join(format!("zestengine-input-runtime-{}", std::process::id()));
        let path = dir.join("input.ron");
        let mut rt = ControlsRuntime::new(Bindings::default());
        rt.set_overrides_path(Some(path.clone()));
        rt.apply(InputCommand::Rebind {
            scheme: String::new(),
            action: "Jump".into(),
            slot: BindingSlot::Buttons,
            index: 0,
            source: InputSource::Key(Key::J),
            policy: ConflictPolicy::Swap,
        })
        .unwrap();
        assert_eq!(rt.defaults().schemes[0].action("Jump").unwrap().buttons, vec![InputSource::Key(Key::Space)], "defaults untouched");

        let mut again = ControlsRuntime::new(Bindings::default());
        again.set_overrides_path(Some(path.clone()));
        assert_eq!(again.bindings().schemes[0].action("Jump").unwrap().buttons, vec![InputSource::Key(Key::J)], "loaded at start");
        again.apply(InputCommand::ResetAll).unwrap();
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_switch_follows_the_last_used_device() {
        let mut rt = ControlsRuntime::new(Bindings::default());
        let name = |rt: &ControlsRuntime| rt.bindings().active_scheme().unwrap().name.clone();
        let mut raw = RawInput::default();
        raw.pad_axes.insert(PadAxis::LeftStickX, 0.8);
        rt.frame(&raw, false);
        assert_eq!(name(&rt), "Gamepad");
        // Still holding the stick while a key goes down: the key is the newest.
        raw.keys_down.insert(Key::W);
        rt.frame(&raw, false);
        assert_eq!(name(&rt), "KB+Mouse");
        rt.frame(&raw, false);
        assert_eq!(name(&rt), "KB+Mouse", "no flapping while both are held");
        rt.frame(&RawInput::default(), true);
        assert_eq!((name(&rt), rt.device()), ("VR Controllers".to_string(), InputDevice::Vr));
        rt.frame(&RawInput::default(), false);
        assert_eq!(name(&rt), "KB+Mouse", "headset off returns to the desktop");

        // Picking a scheme by hand stops auto-switching.
        rt.apply(InputCommand::SetScheme("Gamepad".into())).unwrap();
        rt.frame(&raw, false);
        assert_eq!(name(&rt), "Gamepad");
        rt.apply(InputCommand::SetAutoSwitch(true)).unwrap();
        rt.frame(&RawInput::default(), false);
        let mut raw = RawInput::default();
        raw.mouse_down.insert(MouseButton::Left);
        rt.frame(&raw, false);
        assert_eq!(name(&rt), "KB+Mouse");
    }

    #[test]
    fn capture_ignores_held_inputs_and_escape_cancels() {
        let mut cap = InputCapture::new(CaptureRequest::default());
        let mut raw = RawInput::default();
        raw.mouse_down.insert(MouseButton::Left);
        assert_eq!(cap.poll(&raw), None, "first frame records what is held");
        assert_eq!(cap.poll(&raw), None, "the click that opened it is ignored");
        raw.mouse_down.clear();
        assert_eq!(cap.poll(&raw), None);
        raw.keys_down.insert(Key::G);
        assert_eq!(cap.poll(&raw), Some(CaptureResult::Captured(InputSource::Key(Key::G))));

        let mut cap = InputCapture::new(CaptureRequest { kind: CaptureKind::Button, device: Some(InputDevice::Vr) });
        let mut raw = RawInput::default();
        cap.poll(&raw);
        raw.keys_down.insert(Key::G);
        assert_eq!(cap.poll(&raw), None, "device filter");
        let mut vr = vr_frame();
        vr.right_grip = 1.0;
        raw.set_vr(&vr);
        assert_eq!(cap.poll(&raw), Some(CaptureResult::Captured(InputSource::VrButton(VrButton::RightGrip))));

        let mut cap = InputCapture::new(CaptureRequest { kind: CaptureKind::Axis, device: None });
        let mut raw = RawInput::default();
        cap.poll(&raw);
        raw.keys_down.insert(Key::Escape);
        assert_eq!(cap.poll(&raw), Some(CaptureResult::Cancelled));
    }

    #[test]
    fn runtime_capture_hands_the_result_to_its_owner() {
        let mut rt = ControlsRuntime::new(Bindings::default());
        let owner = ObjectId::new();
        rt.apply(InputCommand::Capture { owner, request: CaptureRequest::default() }).unwrap();
        rt.frame(&RawInput::default(), false);
        assert_eq!(rt.info().capturing, Some(owner));
        let mut raw = RawInput::default();
        raw.pad_down.insert(PadButton::North);
        rt.frame(&raw, false);
        assert!(rt.capturing(), "gameplay is held off the frame it lands");
        assert_eq!(rt.info().captured, Some((owner, CaptureResult::Captured(InputSource::PadButton(PadButton::North)))));
        rt.frame(&raw, false);
        assert_eq!(rt.info().captured, None, "the result lasts one frame");
        assert!(rt.capturing(), "until the button is let go");
        rt.frame(&RawInput::default(), false);
        assert!(!rt.capturing());
    }

    #[test]
    fn old_bindings_get_a_vr_scheme_once() {
        let old = "(schemes: [(name: \"KB+Mouse\", actions: [(name: \"Jump\", kind: Button, buttons: [Key(Space)])])], active: 0)";
        let mut b: Bindings = ron::from_str(old).unwrap();
        assert!(b.auto_switch && b.version == 0);
        b.upgrade();
        assert_eq!(b.schemes.len(), 2);
        assert_eq!(b.schemes[1].device(), Some(InputDevice::Vr));
        b.schemes.pop();
        b.upgrade();
        assert_eq!(b.schemes.len(), 1, "a deleted VR scheme stays deleted");
    }

    #[test]
    fn remap_entries_split_keyed_axes() {
        let kb = ControlScheme::default_kb_mouse();
        let rows = kb.remap_entries();
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(&labels[..6], &["Move Up", "Move Down", "Move Left", "Move Right", "Look", "Jump"]);
        let vr = ControlScheme::default_vr();
        let rows = vr.remap_entries();
        assert_eq!((rows[0].label.as_str(), rows[0].slot), ("Move", BindingSlot::AnalogX));
        assert_eq!(vr.action("Move").unwrap().analog_label().as_deref(), Some("Left Stick"));
    }
}
