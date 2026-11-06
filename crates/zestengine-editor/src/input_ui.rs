//! Input bindings editor, shared by Project Settings > Input and the Input
//! Bindings window: control schemes (add / rename / duplicate / delete),
//! actions (add / rename / delete / kind), per-slot bindings set by capture
//! ("press the input") or picked from a list, analog shaping, conflicts and
//! reset. Edits the project's default bindings; players' own changes are a
//! separate overrides file (see `zestengine_core::BindingOverrides`).

use zestengine_core::{
    ActionBinding, ActionKind, BindingSlot, Bindings, CaptureRequest, CaptureResult, ConflictPolicy, ControlScheme,
    InputCapture, InputDevice, InputSource, MouseButton, all_sources,
};
use egui::{DragValue, RichText, Ui};
use egui_phosphor::regular as ph;

use crate::{theme, widgets};

/// The slot a capture will fill.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RebindTarget {
    pub scheme: usize,
    pub action: usize,
    pub slot: BindingSlot,
    /// Index in the slot's list (past the end adds a binding).
    pub index: usize,
}

/// Editor-side state of the bindings editor.
pub struct BindingsEditor {
    /// Scheme being edited (not necessarily the starting one).
    pub scheme: usize,
    /// What a capture or pick does when another action has the input.
    pub policy: ConflictPolicy,
    rebinding: Option<(RebindTarget, InputCapture)>,
    renaming_scheme: Option<String>,
    /// The click that bound a mouse button must not re-arm the capture.
    swallow_click: bool,
    new_action: String,
    new_kind: ActionKind,
    filter: String,
    status: Option<String>,
}

impl Default for BindingsEditor {
    fn default() -> Self {
        Self {
            scheme: 0,
            policy: ConflictPolicy::Allow,
            rebinding: None,
            renaming_scheme: None,
            swallow_click: false,
            new_action: String::new(),
            new_kind: ActionKind::Button,
            filter: String::new(),
            status: None,
        }
    }
}

impl BindingsEditor {
    /// The slot waiting for an input, if any.
    pub fn rebinding(&self) -> Option<RebindTarget> {
        self.rebinding.as_ref().map(|(t, _)| *t)
    }

    /// The running capture, for the host to feed gamepad / VR input to.
    pub fn capture_mut(&mut self) -> Option<&mut InputCapture> {
        self.rebinding.as_mut().map(|(_, c)| c)
    }

    pub fn cancel_rebind(&mut self) {
        self.rebinding = None;
    }

    fn arm(&mut self, target: RebindTarget) {
        let request = CaptureRequest { kind: target.slot.capture_kind(), device: None };
        self.rebinding = Some((target, InputCapture::new(request)));
    }

    /// Finish the pending capture. Inputs that don't fit the slot (a key for
    /// an analog axis) are ignored and the capture keeps waiting. True if the
    /// bindings changed.
    pub fn finish_rebind(&mut self, bindings: &mut Bindings, result: CaptureResult) -> bool {
        let Some((target, _)) = self.rebinding else {
            return false;
        };
        let src = match result {
            CaptureResult::Cancelled => {
                self.rebinding = None;
                return false;
            }
            CaptureResult::Captured(src) => src,
        };
        if target.slot.is_analog() && !src.is_analog() {
            return false;
        }
        self.rebinding = None;
        self.assign(bindings, target, src)
    }

    fn assign(&mut self, bindings: &mut Bindings, target: RebindTarget, src: InputSource) -> bool {
        let Some(scheme) = bindings.schemes.get(target.scheme) else {
            return false;
        };
        let Some(action) = scheme.actions.get(target.action) else {
            return false;
        };
        let (scheme, action) = (scheme.name.clone(), action.name.clone());
        match bindings.rebind(&scheme, &action, target.slot, target.index, src, self.policy) {
            Ok(changed) => {
                self.status = (!changed.is_empty()).then(|| format!("{} also changed {}", src.display_name(), changed.join(", ")));
                true
            }
            Err(e) => {
                self.status = Some(e.to_string());
                false
            }
        }
    }
}

/// Draw the editor for `bindings`. `id` keeps two open copies apart.
/// Returns true when the bindings changed (save them).
pub fn bindings_editor_ui(ui: &mut Ui, id: &str, bindings: &mut Bindings, st: &mut BindingsEditor) -> bool {
    let mut changed = false;
    if bindings.schemes.is_empty() {
        ui.label(RichText::new("No control schemes.").color(theme::TEXT_DIM));
    }
    st.scheme = st.scheme.min(bindings.schemes.len().saturating_sub(1));
    changed |= scheme_bar(ui, id, bindings, st);
    let si = st.scheme;
    if si >= bindings.schemes.len() {
        return changed;
    }

    let device = bindings.schemes[si].device();
    widgets::property_row(ui, "Device", |ui| {
        ui.label(RichText::new(device.map(|d| d.label()).unwrap_or("Nothing bound yet")).color(theme::TEXT_SECONDARY));
    });
    widgets::property_row(ui, "Starting scheme", |ui| {
        if bindings.active == si {
            ui.label(RichText::new("Yes").color(theme::TEXT_SECONDARY));
        } else if ui.small_button("Start with this scheme").clicked() {
            bindings.active = si;
            changed = true;
        }
    });
    widgets::property_row(ui, "Auto-switch", |ui| {
        changed |= ui
            .checkbox(&mut bindings.auto_switch, "Follow the device in use")
            .on_hover_text("VR Controllers while a headset runs, else whichever of keyboard + mouse / gamepad was used last")
            .changed();
    });
    widgets::property_row(ui, "On conflict", |ui| {
        egui::ComboBox::from_id_salt((id, "policy")).selected_text(st.policy.label()).show_ui(ui, |ui| {
            for p in ConflictPolicy::ALL {
                ui.selectable_value(&mut st.policy, p, p.label());
            }
        });
    });

    let conflicts = bindings.schemes[si].conflicts();
    for c in &conflicts {
        ui.label(
            RichText::new(format!("{} {} is bound to {}", ph::WARNING, c.source.display_name(), c.actions.join(", ")))
                .size(theme::SIZE_META)
                .color(theme::STATUS_WARN_FG),
        );
    }
    if let Some(s) = &st.status {
        ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_MUTED));
    }
    if let Some(t) = st.rebinding()
        && t.scheme == si
    {
        let what = if t.slot.is_analog() { "Move a stick, trigger or the mouse" } else { "Press a key or button" };
        ui.label(RichText::new(format!("{what} (Esc cancels)")).color(theme::accent()));
    }

    ui.add_space(4.0);
    widgets::search_field(ui, &mut st.filter, "Filter actions", ui.available_width());
    ui.add_space(4.0);

    let filter = st.filter.to_lowercase();
    let mut remove = None;
    let count = bindings.schemes[si].actions.len();
    for ai in 0..count {
        let name = bindings.schemes[si].actions[ai].name.clone();
        if !filter.is_empty() && !name.to_lowercase().contains(&filter) {
            continue;
        }
        let (c, del) = action_ui(ui, id, bindings, st, si, ai);
        changed |= c;
        if del {
            remove = Some(ai);
        }
    }
    if let Some(ai) = remove {
        bindings.schemes[si].actions.remove(ai);
        st.rebinding = None;
        changed = true;
    }

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut st.new_action).hint_text("New action").desired_width(140.0));
        kind_combo(ui, (id, "new-kind"), &mut st.new_kind);
        let name = st.new_action.trim().to_string();
        let taken = bindings.schemes[si].action(&name).is_some();
        if ui
            .add_enabled(!name.is_empty() && !taken, egui::Button::new(format!("{} Add", ph::PLUS)))
            .on_hover_text("Add the action to every scheme that lacks it")
            .clicked()
        {
            for s in &mut bindings.schemes {
                if s.action(&name).is_none() {
                    s.actions.push(ActionBinding::empty(&name, st.new_kind));
                }
            }
            st.new_action.clear();
            changed = true;
        }
    });

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let preset = preset_for(&bindings.schemes[si].name);
        if ui
            .add_enabled(preset.is_some(), egui::Button::new("Reset scheme"))
            .on_hover_text("Back to the built-in bindings for this scheme name")
            .clicked()
            && let Some(p) = preset
        {
            bindings.schemes[si] = p;
            st.rebinding = None;
            changed = true;
        }
        if ui.button("Reset all to defaults").on_hover_text("Keyboard + Mouse, Gamepad and VR Controllers").clicked() {
            *bindings = Bindings::default();
            st.scheme = 0;
            st.rebinding = None;
            changed = true;
        }
    });
    changed
}

/// The built-in scheme a name stands for.
fn preset_for(name: &str) -> Option<ControlScheme> {
    [ControlScheme::default_kb_mouse(), ControlScheme::default_gamepad(), ControlScheme::default_vr()]
        .into_iter()
        .find(|p| p.name == name)
}

fn unique_name(bindings: &Bindings, base: &str) -> String {
    let mut name = base.to_string();
    let mut n = 2;
    while bindings.schemes.iter().any(|s| s.name == name) {
        name = format!("{base} {n}");
        n += 1;
    }
    name
}

fn scheme_bar(ui: &mut Ui, id: &str, bindings: &mut Bindings, st: &mut BindingsEditor) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        if let Some(name) = st.renaming_scheme.as_mut() {
            let r = ui.add(egui::TextEdit::singleline(name).desired_width(160.0));
            let trimmed = name.trim().to_string();
            let ok = !trimmed.is_empty()
                && !bindings.schemes.iter().enumerate().any(|(i, s)| i != st.scheme && s.name == trimmed);
            let commit = ui.add_enabled(ok, egui::Button::new(ph::CHECK)).clicked()
                || (ok && r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
            if commit {
                if let Some(s) = bindings.schemes.get_mut(st.scheme) {
                    s.name = trimmed;
                    changed = true;
                }
                st.renaming_scheme = None;
            } else if ui.button(ph::X).on_hover_text("Cancel").clicked() {
                st.renaming_scheme = None;
            }
            return;
        }
        let current = bindings.schemes.get(st.scheme).map(|s| s.name.clone()).unwrap_or_default();
        egui::ComboBox::from_id_salt((id, "scheme")).selected_text(current).show_ui(ui, |ui| {
            for (i, s) in bindings.schemes.iter().enumerate() {
                let label = if i == bindings.active { format!("{} (starting)", s.name) } else { s.name.clone() };
                if ui.selectable_label(st.scheme == i, label).clicked() {
                    st.scheme = i;
                    st.rebinding = None;
                }
            }
        });
        let has = st.scheme < bindings.schemes.len();
        if ui.add_enabled(has, egui::Button::new(ph::PENCIL_SIMPLE)).on_hover_text("Rename").clicked() {
            st.renaming_scheme = bindings.schemes.get(st.scheme).map(|s| s.name.clone());
        }
        if ui.add_enabled(has, egui::Button::new(ph::COPY)).on_hover_text("Duplicate").clicked() {
            let mut copy = bindings.schemes[st.scheme].clone();
            copy.name = unique_name(bindings, &format!("{} copy", copy.name));
            bindings.schemes.push(copy);
            st.scheme = bindings.schemes.len() - 1;
            changed = true;
        }
        if ui
            .add_enabled(bindings.schemes.len() > 1, egui::Button::new(ph::TRASH))
            .on_hover_text("Delete this scheme")
            .clicked()
        {
            bindings.schemes.remove(st.scheme);
            if bindings.active > st.scheme || bindings.active >= bindings.schemes.len() {
                bindings.active = bindings.active.saturating_sub(1);
            }
            st.scheme = st.scheme.saturating_sub(1);
            st.rebinding = None;
            changed = true;
        }
        let unbound_copy = bindings.schemes.get(st.scheme).filter(|s| !s.actions.is_empty()).map(|s| ControlScheme {
            name: format!("{} (unbound)", s.name),
            actions: s.actions.iter().map(|a| ActionBinding::empty(&a.name, a.kind)).collect(),
        });
        ui.menu_button(format!("{} Scheme", ph::PLUS), |ui| {
            crate::widgets::menu_frame(ui);
            let mut add = |ui: &mut Ui, label: &str, mut scheme: ControlScheme| {
                if ui.button(label).clicked() {
                    scheme.name = unique_name(bindings, &scheme.name);
                    bindings.schemes.push(scheme);
                    st.scheme = bindings.schemes.len() - 1;
                    st.rebinding = None;
                    changed = true;
                    ui.close();
                }
            };
            add(ui, "Empty", ControlScheme { name: "New Scheme".into(), actions: Vec::new() });
            add(ui, "Keyboard + Mouse preset", ControlScheme::default_kb_mouse());
            add(ui, "Gamepad preset", ControlScheme::default_gamepad());
            add(ui, "VR Controllers preset", ControlScheme::default_vr());
            if let Some(copy) = unbound_copy {
                add(ui, "Same actions, unbound", copy);
            }
        });
    });
    changed
}

fn kind_label(k: ActionKind) -> &'static str {
    match k {
        ActionKind::Button => "Button",
        ActionKind::Axis1 => "Axis (1D)",
        ActionKind::Axis2 => "Axis (2D)",
    }
}

fn kind_combo(ui: &mut Ui, id: impl std::hash::Hash, kind: &mut ActionKind) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(id).selected_text(kind_label(*kind)).width(90.0).show_ui(ui, |ui| {
        for k in [ActionKind::Button, ActionKind::Axis1, ActionKind::Axis2] {
            changed |= ui.selectable_value(kind, k, kind_label(k)).changed();
        }
    });
    changed
}

fn slot_label(kind: ActionKind, slot: BindingSlot) -> &'static str {
    match (kind, slot) {
        (_, BindingSlot::Buttons) => "Inputs",
        (ActionKind::Axis1, BindingSlot::PosX) => "Positive",
        (ActionKind::Axis1, BindingSlot::NegX) => "Negative",
        (ActionKind::Axis1, BindingSlot::AnalogX) => "Analog",
        (_, BindingSlot::PosY) => "Up",
        (_, BindingSlot::NegY) => "Down",
        (_, BindingSlot::NegX) => "Left",
        (_, BindingSlot::PosX) => "Right",
        (_, BindingSlot::AnalogX) => "Analog X",
        (_, BindingSlot::AnalogY) => "Analog Y",
    }
}

/// Short summary of an action's bindings for its header.
fn summary(a: &ActionBinding) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(l) = a.analog_label() {
        parts.push(l);
    }
    for (slot, _, src) in a.bound() {
        if !slot.is_analog() && parts.len() < 4 {
            parts.push(src.display_name());
        }
    }
    if parts.is_empty() { "unbound".into() } else { parts.join(", ") }
}

/// One action: name, kind, slots, shaping. Returns (changed, delete).
fn action_ui(ui: &mut Ui, id: &str, bindings: &mut Bindings, st: &mut BindingsEditor, si: usize, ai: usize) -> (bool, bool) {
    let mut changed = false;
    let mut delete = false;
    let a = &bindings.schemes[si].actions[ai];
    let header = RichText::new(format!("{}   {}", a.name, summary(a)));
    egui::CollapsingHeader::new(header).id_salt((id, "action", si, ai)).show(ui, |ui| {
        ui.horizontal(|ui| {
            let mut name = bindings.schemes[si].actions[ai].name.clone();
            let r = ui.add(egui::TextEdit::singleline(&mut name).desired_width(140.0));
            let trimmed = name.trim().to_string();
            if r.changed()
                && !trimmed.is_empty()
                && !bindings.schemes[si].actions.iter().enumerate().any(|(i, x)| i != ai && x.name == trimmed)
            {
                bindings.schemes[si].actions[ai].name = trimmed;
                changed = true;
            }
            let a = &mut bindings.schemes[si].actions[ai];
            if kind_combo(ui, (id, "kind", si, ai), &mut a.kind) {
                changed = true;
                st.rebinding = None;
            }
            if ui.button(ph::TRASH).on_hover_text("Delete this action from the scheme").clicked() {
                delete = true;
            }
        });
        let kind = bindings.schemes[si].actions[ai].kind;
        for &slot in BindingSlot::for_kind(kind) {
            changed |= slot_row(ui, bindings, st, RebindTarget { scheme: si, action: ai, slot, index: 0 }, slot_label(kind, slot));
        }
        if kind != ActionKind::Button {
            let a = &mut bindings.schemes[si].actions[ai];
            ui.horizontal(|ui| {
                ui.label(RichText::new("Deadzone").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                changed |= ui.add(DragValue::new(&mut a.deadzone).speed(0.01).range(0.0..=1.0)).changed();
                ui.label(RichText::new("Scale").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                changed |= ui.add(DragValue::new(&mut a.scale).speed(0.05)).changed();
                changed |= ui.checkbox(&mut a.invert_x, "Invert X").changed();
                if kind == ActionKind::Axis2 {
                    changed |= ui.checkbox(&mut a.invert_y, "Invert Y").changed();
                }
            });
        }
    });
    (changed, delete)
}

/// One slot: its inputs as removable chips, a capture button and a pick list.
fn slot_row(ui: &mut Ui, bindings: &mut Bindings, st: &mut BindingsEditor, target: RebindTarget, label: &str) -> bool {
    let mut changed = false;
    let sources = bindings.schemes[target.scheme].actions[target.action].slot(target.slot);
    ui.horizontal_wrapped(|ui| {
        ui.add_sized([64.0, theme::H_BUTTON_SMALL], egui::Label::new(RichText::new(label).size(theme::SIZE_META).color(theme::TEXT_MUTED)));
        let mut remove = None;
        for (i, src) in sources.iter().enumerate() {
            let r = ui
                .small_button(format!("{} {}", src.display_name(), ph::X))
                .on_hover_text(format!("{} ({}). Click to remove.", src.label(), src.device().label()));
            if r.clicked() {
                remove = Some(i);
            }
        }
        if let Some(i) = remove {
            bindings.schemes[target.scheme].actions[target.action].remove_from_slot(target.slot, i);
            st.rebinding = None;
            changed = true;
        }
        let full = target.slot.is_analog() && !sources.is_empty();
        let add_at = RebindTarget { index: if target.slot.is_analog() { 0 } else { sources.len() }, ..target };
        let waiting = st.rebinding().is_some_and(|t| t.scheme == target.scheme && t.action == target.action && t.slot == target.slot);
        let capture_label = if waiting {
            RichText::new(format!("{} Listening", ph::RECORD)).color(theme::accent())
        } else {
            RichText::new(if full { ph::ARROWS_CLOCKWISE } else { ph::PLUS })
        };
        let hint = if target.slot.is_analog() { "Capture: move a stick / trigger" } else { "Capture: press a key or button" };
        let listening = if target.slot.is_analog() {
            "Move a gamepad / VR stick or trigger"
        } else {
            "Press a key, gamepad or VR button, or click here with a mouse button"
        };
        let r = ui.small_button(capture_label).on_hover_text(if waiting { listening } else { hint });
        // Mouse buttons bind by clicking the listening button with them.
        let mouse = waiting && !target.slot.is_analog() && r.hovered();
        let pressed = ui.input(|i| {
            [
                (egui::PointerButton::Primary, MouseButton::Left),
                (egui::PointerButton::Secondary, MouseButton::Right),
                (egui::PointerButton::Middle, MouseButton::Middle),
            ]
            .into_iter()
            .find(|(b, _)| i.pointer.button_pressed(*b))
        });
        if let Some((_, b)) = pressed.filter(|_| mouse) {
            st.rebinding = None;
            st.swallow_click = true;
            changed |= st.assign(bindings, add_at, InputSource::Mouse(b));
        } else if (r.clicked() || r.secondary_clicked() || r.middle_clicked()) && !std::mem::take(&mut st.swallow_click) && !waiting {
            st.arm(add_at);
        }
        if waiting && ui.small_button(ph::X).on_hover_text("Stop listening (Esc)").clicked() {
            st.rebinding = None;
        }
        ui.menu_button(RichText::new(ph::LIST).size(theme::SIZE_META), |ui| {
            crate::widgets::menu_frame(ui);
            for device in InputDevice::ALL {
                ui.menu_button(device.label(), |ui| {
                    crate::widgets::menu_frame(ui);
                    egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                        for src in all_sources(device, target.slot.capture_kind()) {
                            if ui.button(src.display_name()).clicked() {
                                changed |= st.assign(bindings, add_at, src);
                                ui.close();
                            }
                        }
                    });
                });
            }
        })
        .response
        .on_hover_text("Pick an input from a list");
    });
    changed
}
