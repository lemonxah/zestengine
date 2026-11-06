//! In-game controls / remap screen. Fills a fixed set of UI rows (built by
//! the Controls UI template) with the actions of a control scheme and lets
//! the player rebind them: click a binding, press the new input. Changes go
//! through `ctx.rebind_with` and are saved to the player's overrides.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    CaptureResult, ComponentCtx, ConflictPolicy, ObjectId, RemapEntry, TypedComponent, UiButton, UiText,
};

/// Rebind menu driven by named UI objects. With prefix `P` it looks for:
/// `P Scheme` (button + text: cycles the scheme shown), `P Row 1..rows`
/// (shown / hidden per action) holding `P Row n Label` (text) and
/// `P Row n Bind 1..columns` (buttons + text), `P Prev` / `P Next` / `P Page`
/// for paging, `P Reset` (back to the game's defaults) and `P Status`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct ControlsMenu {
    /// Name prefix of the UI objects (see the component docs).
    pub prefix: String,
    /// Row objects in the layout.
    pub rows: u32,
    /// Bindings per row (1 or 2).
    pub columns: u32,
    /// When the new input is already used by another action.
    pub conflicts: ConflictPolicy,
    /// Actions not listed (camera look, push-to-talk, ...).
    pub hidden_actions: Vec<String>,
    /// Engine-filled state of the menu (not saved).
    #[serde(skip)]
    pub state: ControlsMenuState,
}

#[doc(hidden)]
#[derive(Default)]
pub struct ControlsMenuState {
    scheme: Option<String>,
    page: usize,
    /// The entry, slot index and column being rebound.
    listening: Option<(RemapEntry, usize, usize)>,
    /// Frames to wait for the host to start the capture.
    pending: u8,
    /// Seconds clicks are ignored after a capture (its mouse release).
    cooldown: f32,
    message: String,
    shown: HashMap<String, String>,
    active: HashMap<String, bool>,
}

impl Default for ControlsMenu {
    fn default() -> Self {
        Self {
            prefix: "Controls".into(),
            rows: 8,
            columns: 2,
            conflicts: ConflictPolicy::Swap,
            hidden_actions: Vec::new(),
            state: ControlsMenuState::default(),
        }
    }
}

impl ControlsMenu {
    fn id(&self, ctx: &ComponentCtx, name: &str) -> Option<ObjectId> {
        let full = format!("{} {name}", self.prefix);
        ctx.find_object(&full).and_then(|i| ctx.object_ids.get(i).copied())
    }

    fn clicked(&self, ctx: &ComponentCtx, name: &str) -> bool {
        self.id(ctx, name).is_some_and(|id| ctx.ui_clicked(id))
    }

    fn set_text(&mut self, ctx: &mut ComponentCtx, name: &str, text: String) {
        let Some(id) = self.id(ctx, name) else { return };
        if self.state.shown.get(name) == Some(&text) {
            return;
        }
        self.state.shown.insert(name.to_string(), text.clone());
        ctx.modify::<UiText>(id, move |t| t.text = text);
    }

    fn set_shown(&mut self, ctx: &mut ComponentCtx, name: &str, on: bool) {
        let Some(id) = self.id(ctx, name) else { return };
        if self.state.active.get(name) == Some(&on) {
            return;
        }
        self.state.active.insert(name.to_string(), on);
        ctx.set_active(id, on);
    }

    fn set_interactable(ctx: &mut ComponentCtx, id: Option<ObjectId>, on: bool) {
        if let Some(id) = id
            && ctx.get::<UiButton>(id).is_some_and(|b| b.interactable != on)
        {
            ctx.modify::<UiButton>(id, move |b| b.interactable = on);
        }
    }

    /// The text of binding `index` of an entry.
    fn binding_text(ctx: &ComponentCtx, scheme: &str, entry: &RemapEntry, index: usize) -> String {
        let Some(a) = ctx.input_bindings().scheme(scheme).and_then(|s| s.action(&entry.action)) else {
            return String::new();
        };
        if entry.slot.is_analog() {
            return if index == 0 { a.analog_label().unwrap_or_else(|| "-".into()) } else { String::new() };
        }
        a.slot(entry.slot).get(index).map(|s| s.display_name()).unwrap_or_else(|| "-".into())
    }
}

impl TypedComponent for ControlsMenu {
    const NAME: &'static str = "Controls Menu";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let bindings = std::sync::Arc::clone(&ctx.screen.controls.bindings);
        if bindings.schemes.is_empty() {
            return;
        }
        let active = bindings.active_scheme().map(|s| s.name.clone()).unwrap_or_default();
        let scheme_name = self.state.scheme.clone().filter(|n| bindings.scheme(n).is_some()).unwrap_or_else(|| active.clone());
        let Some(scheme) = bindings.scheme(&scheme_name) else { return };
        let entries: Vec<RemapEntry> =
            scheme.remap_entries().into_iter().filter(|e| !self.hidden_actions.contains(&e.action)).collect();
        let rows = self.rows.max(1) as usize;
        let columns = self.columns.clamp(1, 2) as usize;
        let pages = entries.len().div_ceil(rows).max(1);
        self.state.page = self.state.page.min(pages - 1);
        self.state.cooldown = (self.state.cooldown - ctx.unscaled_dt).max(0.0);

        // A capture in flight.
        if let Some((entry, index, _)) = self.state.listening.clone() {
            match ctx.captured_input() {
                Some(CaptureResult::Captured(src)) => {
                    ctx.rebind_with(&scheme_name, &entry.action, entry.slot, index, src, self.conflicts);
                    self.state.message = format!("{}: {}", entry.label, src.display_name());
                    self.state.listening = None;
                    self.state.cooldown = 0.3;
                }
                Some(CaptureResult::Cancelled) => {
                    self.state.message.clear();
                    self.state.listening = None;
                    self.state.cooldown = 0.3;
                }
                None if self.state.pending > 0 => self.state.pending -= 1,
                None if !ctx.is_capturing_input() => self.state.listening = None,
                None => {}
            }
        } else if self.state.cooldown <= 0.0 {
            for row in 0..rows {
                let Some(entry) = entries.get(self.state.page * rows + row) else { break };
                for col in 0..columns {
                    if (entry.slot.is_analog() && col > 0) || !self.clicked(ctx, &format!("Row {} Bind {}", row + 1, col + 1)) {
                        continue;
                    }
                    let device = scheme.device();
                    ctx.capture_next_input(entry.slot.capture_kind(), device);
                    let index = if entry.slot.is_analog() { 0 } else { col.min(scheme.action(&entry.action).map_or(0, |a| a.slot(entry.slot).len())) };
                    self.state.listening = Some((entry.clone(), index, col));
                    self.state.pending = 2;
                    self.state.message = if entry.slot.is_analog() {
                        format!("Move a stick for {} (Esc cancels)", entry.label)
                    } else {
                        format!("Press an input for {} (Esc cancels)", entry.label)
                    };
                }
            }
            if self.clicked(ctx, "Prev") {
                self.state.page = (self.state.page + pages - 1) % pages;
            }
            if self.clicked(ctx, "Next") {
                self.state.page = (self.state.page + 1) % pages;
            }
            if self.clicked(ctx, "Scheme") {
                let i = bindings.scheme_index(&scheme_name).unwrap_or(0);
                self.state.scheme = Some(bindings.schemes[(i + 1) % bindings.schemes.len()].name.clone());
                self.state.page = 0;
                self.state.message.clear();
            }
            if self.clicked(ctx, "Reset") {
                ctx.reset_scheme_bindings(&scheme_name);
                self.state.message = format!("{scheme_name} reset to defaults");
            }
        }

        // Show it.
        let listening = self.state.listening.clone();
        let tag = if scheme_name == active { " (in use)" } else { "" };
        self.set_text(ctx, "Scheme", format!("{scheme_name}{tag}"));
        self.set_text(ctx, "Page", format!("{} / {}", self.state.page + 1, pages));
        for row in 0..rows {
            let entry = entries.get(self.state.page * rows + row);
            self.set_shown(ctx, &format!("Row {}", row + 1), entry.is_some());
            let Some(entry) = entry else { continue };
            self.set_text(ctx, &format!("Row {} Label", row + 1), entry.label.clone());
            for col in 0..2 {
                let name = format!("Row {} Bind {}", row + 1, col + 1);
                let usable = col < columns && !(entry.slot.is_analog() && col > 0);
                self.set_shown(ctx, &name, usable);
                if !usable {
                    continue;
                }
                let waiting = listening.as_ref().is_some_and(|(e, _, c)| e == entry && *c == col);
                let text = if waiting { "...".to_string() } else { Self::binding_text(ctx, &scheme_name, entry, col) };
                self.set_text(ctx, &name, text);
            }
        }
        let conflicts = ctx.binding_conflicts(&scheme_name);
        let status = if self.state.message.is_empty() && !conflicts.is_empty() {
            let c = &conflicts[0];
            format!("{} is used by {}", c.source.display_name(), c.actions.join(" and "))
        } else {
            self.state.message.clone()
        };
        self.set_text(ctx, "Status", status);
        let id = self.id(ctx, "Reset");
        Self::set_interactable(ctx, id, listening.is_none());
    }
}
