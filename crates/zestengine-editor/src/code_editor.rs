//! Dedicated reusable code editor for dock tabs (Rust, GLSL, etc).
//!
//! Syntax highlighting via egui_extras (syntect), plus an LSP-driven
//! completion popup and hover tooltip. The widget renders the popups and
//! applies completion edits locally; the engine feeds it items/hover text and
//! relays completion/hover requests to the language server.

use std::path::Path;

use egui::text::{CCursor, CCursorRange};
use egui::{Event, Key, Modifiers, RichText, Ui};

use crate::inspector::CodeDiagnostic;
use crate::vim::{VimHistory, VimMode, VimOutcome, VimState};
use egui_phosphor::regular as ph;

/// One completion candidate from the language server.
pub struct CompletionItem {
    pub label: String,
    pub insert_text: String,
    pub detail: String,
    /// Tab stops of a snippet completion (char ranges into `insert_text`),
    /// ascending. Empty = a plain insert.
    pub stops: Vec<(usize, usize)>,
    /// Final caret offset (`${0}`) of a snippet, if the server gave one.
    pub exit: Option<usize>,
}

/// An active snippet session after accepting a call completion: the user is
/// filling in the parameters; Tab / Shift+Tab move through the stops.
#[derive(Clone, Default)]
pub struct ActiveSnippet {
    /// (start, end) char ranges of the remaining placeholders, in the text.
    pub stops: Vec<(usize, usize)>,
    /// Which stop is selected next.
    pub current: usize,
    /// Where the caret lands after the last stop.
    pub exit: usize,
    /// The buffer as of the last frame, to shift stops after edits.
    pub last_text: String,
}

/// Expand a VS Code-style snippet (`name(${1:arg}, ${2:arg})`) into its plain
/// text plus the tab stops: the char ranges of each `${N:…}` placeholder in
/// the expanded text, and the `${0}` / final caret offset when present.
pub fn expand_snippet(insert: &str) -> (String, Vec<(usize, usize)>, Option<usize>) {
    let mut out = String::new();
    let mut stops: Vec<(u32, usize, usize)> = Vec::new();
    let mut exit = None;
    let mut chars = insert.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '$' || chars.peek() != Some(&'{') {
            out.push(c);
            continue;
        }
        chars.next(); // '{'
        let mut num = String::new();
        while let Some(d) = chars.peek().filter(|d| d.is_ascii_digit()) {
            num.push(*d);
            chars.next();
        }
        let n: u32 = num.parse().unwrap_or(u32::MAX);
        match chars.next() {
            Some('}') => {
                if n == 0 {
                    exit = Some(out.chars().count());
                } else {
                    let at = out.chars().count();
                    stops.push((n, at, at));
                }
            }
            Some(':') => {
                let mut ph = String::new();
                let mut closed = false;
                for ch in chars.by_ref() {
                    if ch == '}' {
                        closed = true;
                        break;
                    }
                    ph.push(ch);
                }
                if !closed {
                    // Unterminated: emit what we consumed, literally.
                    out.push_str(&format!("${{{n}:{ph}"));
                    continue;
                }
                if n == 0 {
                    exit = Some(out.chars().count());
                    out.push_str(&ph);
                } else {
                    let s = out.chars().count();
                    out.push_str(&ph);
                    let e = out.chars().count();
                    stops.push((n, s, e));
                }
            }
            _ => {
                // Malformed token: emit literally.
                out.push('$');
                out.push('{');
                out.push_str(&num);
            }
        }
    }
    stops.sort_by_key(|(n, _, _)| *n);
    (out, stops.into_iter().map(|(_, s, e)| (s, e)).collect(), exit)
}

/// Shift snippet stop ranges and the exit offset after an edit between
/// `old` and `new` (the buffers of two consecutive frames). Stops inside the
/// edited region are dropped (their placeholder was replaced); stops after it
/// move by the size delta.
fn shift_after_edit(ranges: &mut Vec<(usize, usize)>, exit: &mut usize, old: &str, new: &str) {
    if old == new {
        return;
    }
    let p = old.chars().zip(new.chars()).take_while(|(a, b)| a == b).count();
    let s_old = old
        .chars()
        .rev()
        .zip(new.chars().rev())
        .take_while(|(a, b)| a == b)
        .count()
        .min(old.chars().count() - p)
        .min(new.chars().count() - p);
    let old_end = old.chars().count() - s_old;
    let new_end = new.chars().count() - s_old;
    let delta = new_end as isize - old_end as isize;
    ranges.retain_mut(|(s, e)| {
        if *s >= old_end {
            *s = (*s as isize + delta).max(0) as usize;
            *e = (*e as isize + delta).max(0) as usize;
            true
        } else {
            *e <= p
        }
    });
    if *exit >= old_end {
        *exit = (*exit as isize + delta).max(0) as usize;
    } else if *exit > p {
        *exit = p;
    }
}

/// Active completion popup state (owned by the engine per editor).
pub struct CompletionState {
    pub items: Vec<CompletionItem>,
    pub selected: usize,
    /// Char index where the current word started (filter + replace anchor).
    pub anchor_char: usize,
    /// Opened with Ctrl+Space: Enter accepts immediately. Auto-opened popups
    /// need navigation first (a bare Enter stays a newline).
    pub explicit: bool,
    /// The user moved the selection (arrow keys): Enter accepts.
    pub navigated: bool,
}

/// Hover info to display as a tooltip.
pub struct HoverState {
    pub text: String,
}

/// One reference location from `textDocument/references` (vim `gr`).
#[derive(Clone)]
pub struct ReferenceItem {
    pub path: std::path::PathBuf,
    /// 0-based line / utf-8 column (LSP convention).
    pub line: u32,
    pub col: u32,
    /// Display label, e.g. `file.rs:42`.
    pub label: String,
}

/// One quick fix / refactor from `textDocument/codeAction` (e.g. "Implement
/// missing members"). Positions are LSP: 0-based line, UTF-8 byte column (the
/// editor's convention, matching `char_to_line_col`).
pub struct CodeActionItem {
    pub title: String,
    /// (start line, start col, end line, end col, replacement text).
    pub edits: Vec<(u32, u32, u32, u32, String)>,
}

/// Response from the code editor UI.
#[derive(Default)]
pub struct CodeEditorResponse {
    pub text_changed: bool,
    pub save_requested: bool,
    /// Vim `:q` / `:wq` asked to close this tab.
    pub close_requested: bool,
    pub run_check_requested: bool,
    /// Completion requested (Ctrl+Space or auto-trigger) at this cursor char
    /// index. `completion_explicit` says whether Enter should accept
    /// immediately (true for Ctrl+Space, or a popup that already was).
    pub request_completion: Option<usize>,
    pub completion_explicit: bool,
    /// Hover requested at this char index (Ctrl + pointer over the text).
    pub request_hover: Option<usize>,
    /// Go-to-definition requested (Ctrl+Click or vim `gd`) at this char index.
    pub request_definition: Option<usize>,
    /// References requested (vim `gr`) at this char index.
    pub request_references: Option<usize>,
    /// Code actions requested (Ctrl+.) at this char index.
    pub request_code_action: Option<usize>,
    /// The header's Build & Reload button was clicked.
    pub build_requested: bool,
    /// The header's external-editor button was clicked (open the file in nvim).
    pub open_external: bool,
    /// Ctrl+scroll changed the shared editor zoom (the owner persists it).
    pub zoom_changed: bool,
    /// A reference was picked from the popup: open + jump here.
    pub goto_location: Option<(std::path::PathBuf, u32, u32)>,
}

pub struct CodeEditor;

impl CodeEditor {
    #[allow(clippy::too_many_arguments)]
    pub fn ui(
        &self,
        ui: &mut Ui,
        path: &Path,
        text: &mut String,
        language: &str,
        dirty: bool,
        diagnostics: &[CodeDiagnostic],
        checking: bool,
        completion: &mut Option<CompletionState>,
        hover: &mut Option<HoverState>,
        // Pending go-to-definition target (0-based line, utf-8 col) to jump to.
        goto: &mut Option<(u32, u32)>,
        // References popup (vim `gr`): list to show, cleared when one is picked.
        references: &mut Option<Vec<ReferenceItem>>,
        // Code-action popup (Ctrl+.): quick fixes to show, cleared when picked.
        actions: &mut Vec<CodeActionItem>,
        // Shared editor zoom (global prefs, Ctrl+scroll to change).
        zoom: &mut f32,
        // Vim mode on/off (persisted in project settings, toggled from Edit menu).
        vim_enabled: bool,
    ) -> CodeEditorResponse {
        let mut response = CodeEditorResponse::default();
        // Stable TextEdit id per file: lets us address its cursor state for
        // go-to-def and (when enabled) vim motions.
        let edit_id = ui.make_persistent_id(("zestengine-code-edit", path));

        // Per-file vim modal state (kept in egui memory). The enable flag is
        // passed in (project setting); the file name lives in the tab and the
        // bottom status line, so the header is just problem counts and a hint.
        let vstate_id = ui.make_persistent_id(("zestengine-vim-state", path));
        let mut vstate: VimState = ui.data(|d| d.get_temp(vstate_id).unwrap_or_default());

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".to_owned());

        ui.horizontal(|ui| {
            if checking {
                ui.spinner();
            }
            // Fixed-height problem counts (the full list lives in the
            // Inspector so the code area never shifts while typing).
            let errors = diagnostics.iter().filter(|d| d.level == "error").count();
            let warns = diagnostics.len() - errors;
            if errors > 0 {
                ui.label(
                    RichText::new(format!("{} {errors}", ph::X_CIRCLE))
                        .small()
                        .color(ui.visuals().error_fg_color),
                );
            }
            if warns > 0 {
                ui.label(
                    RichText::new(format!("{} {warns}", ph::WARNING))
                        .small()
                        .color(ui.visuals().warn_fg_color),
                );
            }
            if ui
                .button("nvim")
                .on_hover_text("Open this file in Neovim (external editor)")
                .clicked()
            {
                response.open_external = true;
            }
            if ui
                .button("Build & Reload")
                .on_hover_text("Build the plugin crates and hot-reload them")
                .clicked()
            {
                response.build_requested = true;
            }
        });
        ui.separator();

        // Ctrl+scroll (or pinch) zooms the editor text. The value lives in the
        // editor's GLOBAL preferences, so every code editor shares one zoom
        // that survives restarts. `zoom_delta` is egui's native ctrl+scroll/
        // pinch signal; the raw scroll delta is unreliable here because
        // ScrollAreas zero it as they consume it.
        let zoom_delta = ui.input(|i| i.zoom_delta());
        if ui.input(|i| i.modifiers.ctrl) {
            ui.input_mut(|i| i.smooth_scroll_delta = egui::Vec2::ZERO);
        }
        if (zoom_delta - 1.0).abs() > 1e-4 {
            *zoom = (*zoom * zoom_delta).clamp(0.6, 2.5);
            response.zoom_changed = true;
        }
        if *zoom != 1.0 {
            let base = egui::TextStyle::Monospace.resolve(ui.style()).size;
            ui.style_mut().text_styles.insert(
                egui::TextStyle::Monospace,
                egui::FontId::monospace(base * *zoom),
            );
        }

        // While the completion popup is open, steal navigation keys before the
        // TextEdit consumes them. Enter accepts (like every IDE: Escape
        // dismisses when you wanted a newline instead).
        let (mut nav_down, mut nav_up, mut accept, mut dismiss) = (false, false, false, false);
        if completion.is_some() {
            ui.input_mut(|i| {
                nav_down = i.consume_key(Modifiers::NONE, Key::ArrowDown);
                nav_up = i.consume_key(Modifiers::NONE, Key::ArrowUp);
                accept = i.consume_key(Modifiers::NONE, Key::Enter)
                    || i.consume_key(Modifiers::NONE, Key::Tab);
                // With vim on, Escape is left for the vim block below: one
                // press closes the popup AND ends Insert mode (vim never
                // deselects the editor).
                if !vim_enabled {
                    dismiss = i.consume_key(Modifiers::NONE, Key::Escape);
                }
            });
        }
        let trigger_completion = ui.input_mut(|i| i.consume_key(Modifiers::CTRL, Key::Space));

        // Snippet tab stops (after accepting a call completion): Tab /
        // Shift+Tab move through the params. Steal them before the TextEdit.
        let snippet_id = ui.make_persistent_id(("zestengine-snippet", path));
        let snippet_active = ui.data(|d| d.get_temp::<ActiveSnippet>(snippet_id)).is_some();
        let (mut snippet_tab, mut snippet_shift_tab) = (false, false);
        if snippet_active {
            ui.input_mut(|i| {
                snippet_tab = i.consume_key(Modifiers::NONE, Key::Tab);
                snippet_shift_tab = i.consume_key(Modifiers::SHIFT, Key::Tab);
            });
        }

        // Solid-black editor background + purple selection (ZestEngine Purple theme),
        // and no border/focus outline on the text box.
        {
            let v = ui.visuals_mut();
            v.extreme_bg_color = egui::Color32::BLACK;
            // Translucent so the selected glyphs stay visible through the
            // highlight (an opaque fill painted over the galley hides the text).
            v.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(130, 100, 215, 110);
            // egui recolours selected glyphs to `selection.stroke.color` (text_selection/visuals.rs):
            // `Stroke::NONE` is transparent, which erased every selected character.
            v.selection.stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(240));
            v.widgets.inactive.bg_stroke = egui::Stroke::NONE;
            v.widgets.hovered.bg_stroke = egui::Stroke::NONE;
            v.widgets.active.bg_stroke = egui::Stroke::NONE;
            // The focused (open) widget outline, which egui draws as a light
            // border around the TextEdit.
            v.widgets.open.bg_stroke = egui::Stroke::NONE;
        }

        let font = egui::TextStyle::Monospace.resolve(ui.style());
        let mut layouter = |ui: &Ui, buf: &dyn egui::TextBuffer, wrap_width: f32| {
            let mut job = highlight_code(buf.as_str(), language, &font);
            job.wrap.max_width = wrap_width;
            ui.fonts_mut(|f| f.layout_job(job))
        };

        // Vim: when enabled and the editor is focused, intercept keys before
        // the TextEdit. Normal/Visual modes consume all unmodified keys + text
        // (modified keys pass through so app shortcuts like Ctrl+S still work);
        // Insert mode steals only Escape and lets typing flow to the TextEdit.
        // Treat the editor as focused if it was focused last frame too: egui can
        // surrender a TextEdit's focus on the Escape press before we get to run,
        // and we still want to intercept that Escape (vim Insert -> Normal) and
        // re-grab focus so it doesn't deselect the buffer.
        let focus_id = ui.make_persistent_id(("zestengine-code-focus", path));
        let has_focus_now = ui.memory(|m| m.has_focus(edit_id));
        let was_focused: bool = ui.data(|d| d.get_temp(focus_id).unwrap_or(false));
        let editor_focused = has_focus_now || was_focused;
        let mut keep_focus = has_focus_now;
        let mut pending_vim: Option<VimOutcome> = None;
        // Live `:s`/`:%s` preview: highlight ranges (in char indices) + whether
        // a replacement is being shown (vs just matches).
        let mut preview_ranges: Vec<(usize, usize)> = Vec::new();
        let mut preview_replaced = false;
        // Cursor position for the status line (vim's computed cursor wins so it
        // isn't a frame stale after a motion).
        let mut shown_cursor: Option<usize> = None;
        // Keep the caret solid (no blink) for a moment after any vim keystroke
        // so it stays visible while moving. egui only resets the blink on its
        // own edits, so it won't reset on our programmatic cursor moves.
        let now = ui.input(|i| i.time);
        let active_id = ui.make_persistent_id(("zestengine-caret-active", path));
        if vim_enabled && editor_focused {
            let insert = vstate.mode() == VimMode::Insert;
            let cur = egui::text_edit::TextEditState::load(ui.ctx(), edit_id)
                .and_then(|s| s.cursor.char_range())
                .map(|r| r.primary.index)
                .unwrap_or(0);
            let mut captured: Vec<Event> = Vec::new();
            ui.input_mut(|i| {
                i.events.retain(|e| match e {
                    Event::Key {
                        key: Key::Escape,
                        pressed: true,
                        ..
                    } => {
                        captured.push(e.clone());
                        false
                    }
                    _ if insert => true,
                    Event::Text(_) => {
                        captured.push(e.clone());
                        false
                    }
                    Event::Key {
                        key,
                        pressed,
                        modifiers,
                        ..
                    } if *pressed && crate::vim::capture_key(*key, *modifiers) => {
                        captured.push(e.clone());
                        false
                    }
                    _ => true,
                });
            });
            if !captured.is_empty() {
                let mode_before = vstate.mode();
                // In command mode, run the command against the original text:
                // last frame's live preview is discarded and recomputed.
                if mode_before == VimMode::Command
                    && let Some(base) = &vstate.preview_base {
                        *text = base.clone();
                    }
                let before = text.clone();
                let hist_id = ui.make_persistent_id(("zestengine-vim-undo", path));
                let mut hist: VimHistory = ui.data(|d| d.get_temp(hist_id).unwrap_or_default());
                let outcome = crate::vim::handle(&mut vstate, text, cur, &captured, &mut hist);
                ui.data_mut(|d| d.insert_temp(hist_id, hist));
                // Stash the base on entering command mode; clear on leaving
                // (commit via Enter or cancel via Escape).
                if mode_before != VimMode::Command && vstate.mode() == VimMode::Command {
                    vstate.preview_base = Some(before.clone());
                } else if mode_before == VimMode::Command && vstate.mode() != VimMode::Command {
                    vstate.preview_base = None;
                }
                // An Escape captured here also closes the completion popup
                // (one press: popup + Insert -> Normal).
                if captured.iter().any(|e| matches!(e, Event::Key { key: Key::Escape, pressed: true, .. })) {
                    dismiss = true;
                }

                if outcome.text_changed {
                    response.text_changed = true;
                }
                shown_cursor = outcome.cursor;
                // Mark caret activity so it stays solid while moving.
                ui.data_mut(|d| d.insert_temp(active_id, now));
                pending_vim = Some(outcome);
            }

            // Live substitution preview, recomputed every frame a `:` command
            // is open so the highlights persist and the text reflects the
            // current cmdline. Not marked dirty — committed only on Enter,
            // reverted to the base on Escape (both handled above).
            if vstate.mode() == VimMode::Command
                && let Some(base) = vstate.preview_base.clone() {
                    match crate::vim::preview_substitute(&base, &vstate.cmdline(), cur) {
                        Some(prev) => {
                            *text = prev.text;
                            preview_ranges = prev.highlights;
                            preview_replaced = prev.replaced;
                        }
                        None => *text = base,
                    }
                }
        }

        // The editor scrolls so long files (and go-to-definition jumps) work.
        let goto_target = goto.take();
        // Escape never deselects the editor. With vim on, the block above
        // already captured it (Insert -> Normal) and re-grabbed focus; this
        // swallows any Escape left over while the editor is (or was) focused,
        // so the TextEdit can never see one and surrender its focus.
        if editor_focused {
            ui.input_mut(|i| {
                i.consume_key(Modifiers::NONE, Key::Escape);
            });
        }
        // PageUp / PageDown move the caret a screenful and scroll the code
        // view (vim command mode keeps them for the command line).
        let page_dy = if editor_focused && vstate.mode() != VimMode::Command {
            let h = ui.clip_rect().height().max(40.0);
            ui.input_mut(|i| {
                if i.consume_key(Modifiers::NONE, Key::PageDown) {
                    Some(0.85 * h)
                } else if i.consume_key(Modifiers::NONE, Key::PageUp) {
                    Some(-0.85 * h)
                } else {
                    None
                }
            })
        } else {
            None
        };
        // Enter = newline. The completion popup already consumed its own Enter
        // above (accept, or the auto-opened popup's plain newline); vim command
        // mode keeps Enter (it runs the command). Everywhere else Enter inserts
        // an auto-indented newline instead of the TextEdit's plain one.
        let enter_newline = ui.input_mut(|i| {
            editor_focused
                && completion.is_none()
                && vstate.mode() != VimMode::Command
                && i.consume_key(Modifiers::NONE, Key::Enter)
        });
        // Caret stays solid for ~0.6s after the last vim keystroke so it's
        // easy to follow while moving; egui resumes blinking once idle.
        let last_active: f64 = ui.data(|d| d.get_temp(active_id).unwrap_or(f64::NEG_INFINITY));
        if now - last_active < 0.6 {
            ui.style_mut().visuals.text_cursor.blink = false;
        }

        // Fill the dock even when the text is shorter than the viewport: size
        // the TextEdit to at least the visible height so the editable/clickable
        // area reaches the bottom (longer files just grow past it and scroll).
        let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
        // Reserve a row for the bottom status line (sized for ~14px text).
        let status_h = row_h.max(18.0) + 10.0;
        let avail_h = (ui.available_height() - status_h).max(0.0);
        let fill_rows = ((avail_h / row_h).floor() as usize).max(8);
        // Left gutter for line numbers, sized to the line count.
        let line_count = text.bytes().filter(|&b| b == b'\n').count() + 1;
        let digits = line_count.to_string().len().max(2);
        let mono = egui::TextStyle::Monospace.resolve(ui.style());
        let digit_w = ui.fonts_mut(|f| f.glyph_width(&mono, '0'));
        let gutter_w = digits as f32 * digit_w + 12.0;

        let output = egui::ScrollArea::vertical()
            .id_salt("code-scroll")
            .auto_shrink([false, false])
            // Cap the scroll viewport so the bottom status line stays on-screen
            // (without this it consumes the reserved row and the status line is
            // pushed out of view).
            .max_height(avail_h)
            .show(ui, |ui| {
                let mut out = None;
                ui.horizontal_top(|ui| {
                    let gutter_left = ui.cursor().left();
                    ui.add_space(gutter_w);
                    let output = egui::TextEdit::multiline(text)
                        .id(edit_id)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(fill_rows)
                        .layouter(&mut layouter)
                        .show(ui);
                    draw_line_numbers(ui, text, &output, gutter_left, gutter_w, row_h, &mono);
                    // Jump the cursor to a go-to-definition target.
                    if let Some((line, col)) = goto_target {
                        let target = line_col_to_char(text, line, col);
                        let cc = CCursor::new(target);
                        let mut st = output.state.clone();
                        st.cursor.set_char_range(Some(CCursorRange::one(cc)));
                        st.store(ui.ctx(), output.response.id);
                        output.response.request_focus();
                        let rect = output
                            .galley
                            .pos_from_cursor(cc)
                            .translate(output.galley_pos.to_vec2());
                        ui.scroll_to_rect(rect, Some(egui::Align::Center));
                    }
                    // PageUp / PageDown: move the caret a screenful; the
                    // margin scroll below brings the new position into view.
                    if let Some(dy) = page_dy {
                        let cc = output.cursor_range.map(|r| r.primary.index).unwrap_or(0);
                        let caret = output.galley.pos_from_cursor(CCursor::new(cc));
                        let local = caret.center() + egui::vec2(0.0, dy) - output.galley_pos;
                        let new_cc = output.galley.cursor_from_pos(local);
                        let mut st = output.state.clone();
                        st.cursor.set_char_range(Some(CCursorRange::one(new_cc)));
                        st.store(ui.ctx(), output.response.id);
                    }
                    // Keep the caret in view while navigating: when it is
                    // within 4 rows of the viewport edge, scroll the code
                    // view. Gated on the caret actually moving, so manual
                    // mouse scrolling is never fought.
                    let caret_id = ui.make_persistent_id(("zestengine-caret-last", path));
                    let last_caret: Option<usize> = ui.data(|d| d.get_temp(caret_id));
                    if let Some(cc) = output.cursor_range.map(|r| r.primary.index) {
                        if last_caret != Some(cc) {
                            let caret = output
                                .galley
                                .pos_from_cursor(CCursor::new(cc))
                                .translate(output.galley_pos.to_vec2());
                            let band = 4.0 * row_h;
                            let clip = ui.clip_rect();
                            if caret.top() < clip.top() + band || caret.bottom() > clip.bottom() - band {
                                ui.scroll_to_rect(
                                    egui::Rect::from_min_max(caret.min, caret.max).expand2(egui::vec2(0.0, band)),
                                    None,
                                );
                            }
                        }
                        ui.data_mut(|d| d.insert_temp(caret_id, cc));
                    }
                    out = Some(output);
                });
                out.unwrap()
            })
            .inner;

        response.text_changed |= output.response.changed();

        // Live `:s`/`:%s` preview highlights: green over replaced spans, yellow
        // over matches (pattern-only).
        if !preview_ranges.is_empty() {
            let color = if preview_replaced {
                egui::Color32::from_rgba_unmultiplied(80, 200, 120, 70)
            } else {
                egui::Color32::from_rgba_unmultiplied(230, 200, 80, 70)
            };
            draw_ranges(ui, &output, &preview_ranges, color);
        }

        // Apply vim's computed cursor / selection to the TextEdit, then persist
        // the modal state for next frame.
        if let Some(out) = pending_vim {
            response.save_requested |= out.save;
            response.close_requested |= out.close;
            if let Some(c) = out.cursor {
                if out.goto_def {
                    response.request_definition = Some(c);
                }
                if out.goto_refs {
                    response.request_references = Some(c);
                }
            }
            let range = if let Some((a, b)) = out.selection {
                // Primary = the ACTIVE cursor end (the TextEdit reports it back
                // as the cursor next frame, and the engine's motions then move
                // from there); secondary = the fixed anchor end. Using the lo
                // end as primary made visual j/k jump two rows and broke
                // moving back over the anchor.
                let cur = out.cursor.unwrap_or(b);
                let other = if cur <= a { b } else { a };
                Some(CCursorRange::two(CCursor::new(cur), CCursor::new(other)))
            } else {
                out.cursor.map(|c| CCursorRange::one(CCursor::new(c)))
            };
            if let Some(r) = range {
                let mut st = output.state.clone();
                st.cursor.set_char_range(Some(r));
                st.store(ui.ctx(), output.response.id);
            }
            // vim handles Escape (Insert -> Normal), so re-grab focus after any
            // vim keystroke to keep the buffer focused instead of letting egui
            // surrender it (which reads as deselecting the editor).
            output.response.request_focus();
            keep_focus = true;
        }
        let cursor_char = output.cursor_range.map(|r| r.primary.index);

        // Bottom status line. In vim command mode it folds in the `:` command
        // line; otherwise it shows mode / file / language / cursor position.
        let status_cursor = shown_cursor.or(cursor_char).unwrap_or(0);
        let (ln, col) = line_col(text, status_cursor);
        egui::Frame::new()
            .fill(ui.visuals().faint_bg_color)
            .inner_margin(egui::Margin::symmetric(6, 2))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.set_width(ui.available_width());
                    if vim_enabled && vstate.mode() == VimMode::Command {
                        ui.label(
                            RichText::new(format!("{}{}", vstate.prompt(), vstate.cmdline()))
                                .monospace()
                                .size(14.0)
                                .color(egui::Color32::from_rgb(200, 150, 220)),
                        );
                        return;
                    }
                    if vim_enabled {
                        let mc = match vstate.mode() {
                            VimMode::Insert => egui::Color32::from_rgb(120, 200, 120),
                            VimMode::Visual | VimMode::VisualLine => {
                                egui::Color32::from_rgb(200, 170, 90)
                            }
                            VimMode::Command => egui::Color32::from_rgb(200, 150, 220),
                            VimMode::Normal => egui::Color32::from_rgb(120, 170, 235),
                        };
                        ui.label(RichText::new(vstate.mode().label()).size(14.0).strong().color(mc));
                        ui.separator();
                    }
                    ui.label(RichText::new(&name).size(14.0));
                    ui.separator();
                    ui.label(RichText::new(language).size(14.0).weak());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("Ln {ln}, Col {col}")).size(14.0));
                        if dirty {
                            ui.separator();
                            ui.label(RichText::new("unsaved").size(14.0).weak());
                        }
                    });
                });
            });
        ui.data_mut(|d| d.insert_temp(vstate_id, vstate));
        // Remember focus for next frame so a focus surrender on Escape doesn't
        // stop us from intercepting it.
        ui.data_mut(|d| d.insert_temp(focus_id, keep_focus));

        // Inline diagnostics: squiggle under each problem line, with the
        // message on hover.
        draw_diagnostics(ui, text, &output, diagnostics);

        if trigger_completion && let Some(c) = cursor_char {
            response.request_completion = Some(c);
            response.completion_explicit = true;
        }

        // Ctrl+. : code actions (quick fixes / refactors) at the cursor.
        let trigger_action = ui.input_mut(|i| i.consume_key(Modifiers::CTRL, Key::Period));
        if trigger_action && let Some(c) = cursor_char {
            response.request_code_action = Some(c);
        }

        // Auto-trigger: completion requests on meaningful typing, so the
        // popup opens and refreshes while you type (Ctrl+Space still forces
        // it). Member access (`.` / `:`) requests immediately; mid-word
        // keystrokes re-open the popup; and while it is open, every inserted
        // character refreshes its items. Multi-char inserts (pastes, vim
        // edits) never auto-trigger, and neither do programmatic edits (no
        // input event). An empty response simply drops the popup. After `;`
        // or `{` (statement / scope boundaries) no suggestion is offered:
        // the open popup dismisses and no new one opens.
        if completion.is_none() {
            if let Some(c) = cursor_char {
                let typed = ui.input(|i| {
                    i.events.iter().find_map(|e| match e {
                        Event::Text(s) if !s.is_empty() => Some(s.clone()),
                        _ => None,
                    })
                });
                if let Some(t) = typed {
                    let one_char = t.chars().count() == 1;
                    let ch = t.chars().last();
                    let boundary = matches!(ch, Some(';') | Some('{'));
                    let access = matches!(ch, Some('.') | Some(':'));
                    let mid_word = one_char
                        && matches!(ch, Some(c2) if c2.is_alphanumeric() || c2 == '_')
                        && c > 0
                        && text[..c]
                            .chars()
                            .next_back()
                            .is_some_and(|p| p.is_alphanumeric() || p == '_');
                    if !boundary && (access || mid_word) {
                        response.request_completion = Some(c);
                    }
                }
            }
        } else if let Some(c) = cursor_char {
            // Popup open: keep its items tracking the word being typed - but
            // not across a statement / scope boundary: `;` or `{` dismisses.
            let typed: Option<String> = ui.input(|i| {
                i.events.iter().find_map(|e| match e {
                    Event::Text(s) if !s.is_empty() => Some(s.clone()),
                    _ => None,
                })
            });
            let boundary = typed
                .as_ref()
                .and_then(|t| t.chars().last())
                .is_some_and(|ch| matches!(ch, ';' | '{'));
            if boundary {
                *completion = None;
            } else if typed.is_some() {
                response.request_completion = Some(c);
                response.completion_explicit =
                    completion.as_ref().is_some_and(|s| s.explicit);
            }
        }

        // Ctrl + pointer over the text: hover info; Ctrl+Click: go to def.
        let ctrl = ui.input(|i| i.modifiers.ctrl);
        if let Some(pos) = output.response.hover_pos() {
            if ctrl {
                let local = pos - output.galley_pos;
                let cc = output.galley.cursor_from_pos(local);
                response.request_hover = Some(cc.index);
                if output.response.clicked() {
                    response.request_definition = Some(cc.index);
                }
            } else {
                *hover = None;
            }
        } else {
            *hover = None;
        }
        if let Some(h) = hover.as_ref()
            && !h.text.is_empty()
            && let Some(pos) = output.response.hover_pos()
        {
            egui::Area::new(egui::Id::new("lsp-hover"))
                .order(egui::Order::Tooltip)
                .fixed_pos(pos + egui::vec2(12.0, 16.0))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_max_width(480.0);
                        ui.label(RichText::new(&h.text).small());
                    });
                });
        }

        // Completion popup. Take the state out to avoid borrow conflicts.
        if let Some(mut state) = completion.take() {
            let keep = render_completion(
                ui,
                text,
                &output,
                &mut state,
                cursor_char,
                nav_down,
                nav_up,
                accept,
                dismiss,
                snippet_id,
                &mut response,
            );
            if keep {
                *completion = Some(state);
            }
        }

        // A plain Enter (no popup open) inserts an auto-indented newline
        // instead of the TextEdit's plain one.
        if enter_newline && let Some(c) = cursor_char {
            let indent = newline_indent(text, c);
            let at = byte_of_char(text, c);
            text.insert_str(at, &format!("\n{indent}"));
            let mut st = output.state.clone();
            st.cursor.set_char_range(Some(CCursorRange::one(CCursor::new(
                c + 1 + indent.chars().count(),
            ))));
            st.store(ui.ctx(), output.response.id);
            response.text_changed = true;
        }

        // Snippet session (call completions with params): track the buffer,
        // advance through the tab stops.
        if let Some(mut snip) = ui.data(|d| d.get_temp::<ActiveSnippet>(snippet_id)) {
            shift_after_edit(&mut snip.stops, &mut snip.exit, &snip.last_text, text);
            snip.last_text = text.clone();
            if snippet_shift_tab {
                snip.current = snip.current.saturating_sub(1);
            } else if snippet_tab {
                snip.current += 1;
            }
            let mut st = output.state.clone();
            if snip.current < snip.stops.len() {
                let len = text.chars().count();
                let (s, e) = snip.stops[snip.current];
                let s = s.min(len);
                let e = e.max(s).min(len);
                st.cursor.set_char_range(Some(CCursorRange::two(CCursor::new(s), CCursor::new(e))));
                st.store(ui.ctx(), output.response.id);
                ui.data_mut(|d| d.insert_temp(snippet_id, snip));
            } else {
                // Past the last stop: park the caret at the exit offset.
                st.cursor.set_char_range(Some(CCursorRange::one(CCursor::new(
                    snip.exit.min(text.chars().count()),
                ))));
                st.store(ui.ctx(), output.response.id);
            }
        }

        // References popup (vim `gr`): centered list; click or Enter to jump,
        // Escape to dismiss.
        if let Some(list) = references.take() {
            let mut keep = true;
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                keep = false;
            }
            egui::Window::new(format!("References ({})", list.len()))
                .id(egui::Id::new(("vim-refs", path)))
                .collapsible(false)
                .resizable(true)
                .default_width(560.0)
                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 60.0))
                .show(ui.ctx(), |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            for item in &list {
                                if ui
                                    .add(
                                        egui::Button::new(&item.label)
                                            .min_size(egui::vec2(ui.available_width(), 0.0)),
                                    )
                                    .clicked()
                                {
                                    response.goto_location =
                                        Some((item.path.clone(), item.line, item.col));
                                    keep = false;
                                }
                            }
                        });
                    if ui.button("Close").clicked() {
                        keep = false;
                    }
                });
            if keep && response.goto_location.is_none() {
                *references = Some(list);
            }
        }

        // Code-action popup (Ctrl+.): quick fixes / refactors from the
        // server. Click to apply (edits go in right-to-left so positions
        // don't shift); Escape dismisses.
        if !actions.is_empty() {
            let mut keep = true;
            let mut pick: Option<usize> = None;
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                keep = false;
            }
            egui::Window::new("Code actions")
                .id(egui::Id::new(("code-actions", path)))
                .collapsible(false)
                .resizable(true)
                .default_width(560.0)
                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 60.0))
                .show(ui.ctx(), |ui| {
                    for (i, action) in actions.iter().enumerate() {
                        if ui
                            .add(
                                egui::Button::new(&action.title)
                                    .min_size(egui::vec2(ui.available_width(), 0.0)),
                            )
                            .clicked()
                        {
                            pick = Some(i);
                        }
                    }
                });
            if !keep {
                actions.clear();
            } else if let Some(i) = pick {
                apply_code_action(text, &actions[i].edits, &output, ui.ctx());
                actions.clear();
                response.text_changed = true;
            }
        }

        response
    }
}

/// Apply one code action's edits to the buffer: LSP positions -> char
/// indices, applied right-to-left so earlier edits don't shift later ones.
/// The caret lands at the end of the first (lowest-position) edit.
fn apply_code_action(
    text: &mut String,
    edits: &[(u32, u32, u32, u32, String)],
    output: &egui::text_edit::TextEditOutput,
    ctx: &egui::Context,
) {
    let mut ranges: Vec<(usize, usize, &str)> = edits
        .iter()
        .map(|(sl, sc, el, ec, new_text)| {
            (
                line_col_to_char(text, *sl, *sc),
                line_col_to_char(text, *el, *ec),
                new_text.as_str(),
            )
        })
        .collect();
    ranges.sort_by_key(|(s, _, _)| std::cmp::Reverse(*s));
    let mut caret_end = None;
    let mut last_start = usize::MAX;
    for (s, e, new_text) in ranges {
        // Skip overlapping edits (rust-analyzer returns disjoint ones).
        if e > last_start {
            continue;
        }
        replace_range_chars(text, s, e, new_text);
        caret_end = Some(s + new_text.chars().count());
        last_start = s;
    }
    if let Some(end) = caret_end {
        let mut st = output.state.clone();
        st.cursor
            .set_char_range(Some(CCursorRange::one(CCursor::new(end))));
        st.store(ctx, output.response.id);
    }
}

/// Render the completion popup and apply navigation/acceptance. Returns true
/// to keep the popup open.
#[allow(clippy::too_many_arguments)]
fn render_completion(
    ui: &mut Ui,
    text: &mut String,
    output: &egui::text_edit::TextEditOutput,
    state: &mut CompletionState,
    cursor_char: Option<usize>,
    nav_down: bool,
    nav_up: bool,
    accept: bool,
    dismiss: bool,
    snippet_id: egui::Id,
    response: &mut CodeEditorResponse,
) -> bool {
    if dismiss {
        return false;
    }
    let Some(cur) = cursor_char else {
        return false;
    };
    if cur < state.anchor_char {
        return false; // moved before the word
    }

    let prefix: String = text
        .chars()
        .skip(state.anchor_char)
        .take(cur - state.anchor_char)
        .collect();
    let lower = prefix.to_lowercase();
    let mut filtered: Vec<usize> = (0..state.items.len())
        .filter(|&i| state.items[i].label.to_lowercase().starts_with(&lower))
        .collect();
    if filtered.is_empty() && !lower.is_empty() {
        filtered = (0..state.items.len())
            .filter(|&i| state.items[i].label.to_lowercase().contains(&lower))
            .collect();
    }
    if filtered.is_empty() {
        return false;
    }

    let n = filtered.len();
    let mut sel = state.selected.min(n - 1);
    if nav_down {
        sel = (sel + 1) % n;
    }
    if nav_up {
        sel = (sel + n - 1) % n;
    }
    state.selected = sel;

    // Render the popup and capture a click. Flip above the cursor when the
    // list would run off the bottom of the screen (the cursor is usually near
    // the editor's bottom edge), so it always shows a full page of items.
    let cursor_rect = output.galley.pos_from_cursor(CCursor::new(cur));
    let below = output.galley_pos + cursor_rect.left_bottom().to_vec2();
    let flip = below.y + 440.0 > ui.ctx().content_rect().height();
    let screen = if flip {
        output.galley_pos + cursor_rect.left_top().to_vec2() - egui::vec2(0.0, 6.0)
    } else {
        below
    };
    let mut clicked: Option<usize> = None;
    egui::Area::new(egui::Id::new("lsp-completion"))
        .order(egui::Order::Foreground)
        .pivot(if flip { egui::Align2::LEFT_BOTTOM } else { egui::Align2::LEFT_TOP })
        .fixed_pos(screen)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width(640.0);
                egui::ScrollArea::vertical()
                    .max_height(460.0)
                    .show(ui, |ui| {
                        for (vis, &item_i) in filtered.iter().enumerate() {
                            let it = &state.items[item_i];
                            let label = if it.detail.is_empty() {
                                it.label.clone()
                            } else {
                                format!("{}   {}", it.label, it.detail)
                            };
                            let item_resp = ui.selectable_label(vis == sel, label);
                            if item_resp.clicked() {
                                clicked = Some(vis);
                            }
                            // Keep the keyboard-selected item visible.
                            if vis == sel && (nav_down || nav_up) {
                                item_resp.scroll_to_me(Some(egui::Align::Center));
                            }
                        }
                    });
            });
        });

    let accept_vis = if accept { Some(sel) } else { clicked };
    if let Some(vis) = accept_vis {
        let item = &state.items[filtered[vis]];
        let insert = item.insert_text.clone();
        let new_cursor = replace_range_chars(text, state.anchor_char, cur, &insert);
        let mut st = output.state.clone();
        if !item.stops.is_empty() {
            // Call completion: select the first parameter and start a snippet
            // session so Tab / Shift+Tab walk through the rest.
            let (s, e) = item.stops[0];
            let start = state.anchor_char + s;
            let end = state.anchor_char + e;
            st.cursor.set_char_range(Some(CCursorRange::two(CCursor::new(start), CCursor::new(end))));
            st.store(ui.ctx(), output.response.id);
            let stops: Vec<(usize, usize)> = item
                .stops
                .iter()
                .map(|(s, e)| (state.anchor_char + s, state.anchor_char + e))
                .collect();
            let exit = item
                .exit
                .map(|x| state.anchor_char + x)
                .unwrap_or(new_cursor);
            ui.data_mut(|d| {
                d.insert_temp(
                    snippet_id,
                    ActiveSnippet {
                        stops,
                        current: 0,
                        exit,
                        last_text: text.clone(),
                    },
                );
            });
        } else {
            st.cursor.set_char_range(Some(CCursorRange::one(CCursor::new(new_cursor))));
            st.store(ui.ctx(), output.response.id);
        }
        response.text_changed = true;
        return false;
    }
    true
}

fn replace_range_chars(text: &mut String, start_char: usize, end_char: usize, repl: &str) -> usize {
    let start_byte = byte_of_char(text, start_char);
    let end_byte = byte_of_char(text, end_char);
    text.replace_range(start_byte..end_byte, repl);
    start_char + repl.chars().count()
}

fn byte_of_char(text: &str, char_idx: usize) -> usize {
    text.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(text.len())
}

/// The indentation for a new line opened at char index `cursor`: the current
/// line's leading whitespace, plus one indent level when the line opens a
/// scope (ends in `{`), or the OWNING scope's indent when the line closes one
/// (ends in `}` / `)` / `]`).
fn newline_indent(text: &str, cursor: usize) -> String {
    let at = byte_of_char(text, cursor);
    let up_to = &text[..at];
    let line_start = up_to.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line = &up_to[line_start..];
    let leading: String = line.chars().take_while(|c| *c == ' ' || *c == '\t').collect();
    let trimmed = line.trim_end();
    if trimmed.ends_with('{') {
        format!("{leading}    ")
    } else if trimmed.ends_with(['}', ']', ')']) {
        // The next line belongs to the scope that OWNS the opener: indent
        // like the opener's line (closing a nested scope un-indents).
        opener_indent(up_to, line_start + trimmed.chars().count() - 1).unwrap_or(leading)
    } else {
        leading
    }
}

/// The indentation of the line that OPENS the scope whose closing bracket
/// sits at char index `close_at` (inclusive): scan backward matching
/// same-kind brackets, then take the opener's line's leading whitespace.
fn opener_indent(text: &str, close_at: usize) -> Option<String> {
    let close = text[..byte_of_char(text, close_at + 1).min(text.len())].chars().next_back()?;
    let open = match close {
        '}' => '{',
        ')' => '(',
        ']' => '[',
        _ => return None,
    };
    let before: Vec<char> = text.chars().take(close_at).collect();
    let mut depth = 0usize;
    for (i, ch) in before.iter().enumerate().rev() {
        if *ch == close {
            depth += 1;
        } else if *ch == open {
            if depth == 0 {
                let line_start = before[..i].iter().rposition(|&c| c == '\n').map_or(0, |p| p + 1);
                let leading: String = before[line_start..i]
                    .iter()
                    .take_while(|c| **c == ' ' || **c == '\t')
                    .collect();
                return Some(leading);
            }
            depth -= 1;
        }
    }
    None
}

/// 1-based (line, column) for a char index, for the status line.
fn line_col(text: &str, char_idx: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for ch in text.chars().take(char_idx) {
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// (0-based line, utf-8 byte column) → char index in `text`.
fn line_col_to_char(text: &str, line: u32, col: u32) -> usize {
    let (start, _) = line_char_range(text, line as usize);
    let chars: Vec<char> = text.chars().collect();
    let mut bytes = 0u32;
    let mut idx = start;
    while idx < chars.len() && chars[idx] != '\n' && bytes < col {
        bytes += chars[idx].len_utf8() as u32;
        idx += 1;
    }
    idx
}

/// (start, end) char indices of the 0-based line (end excludes the newline).
fn line_char_range(text: &str, line0: usize) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0usize;
    if line0 > 0 {
        let mut line = 0usize;
        let mut found = false;
        for (i, &c) in chars.iter().enumerate() {
            if c == '\n' {
                line += 1;
                if line == line0 {
                    start = i + 1;
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return (chars.len(), chars.len());
        }
    }
    let mut end = start;
    while end < chars.len() && chars[end] != '\n' {
        end += 1;
    }
    (start, end)
}

/// Bundled syntax definitions (Rust, GLSL, TOML, …), loaded once.
fn syntaxes() -> &'static syntect::parsing::SyntaxSet {
    use std::sync::OnceLock;
    static SET: OnceLock<syntect::parsing::SyntaxSet> = OnceLock::new();
    SET.get_or_init(syntect::parsing::SyntaxSet::load_defaults_newlines)
}

/// The ZestEngine Purple theme (Solarized-style structure, purple-leaning, black bg).
fn purple_theme() -> &'static syntect::highlighting::Theme {
    use std::sync::OnceLock;
    static THEME: OnceLock<syntect::highlighting::Theme> = OnceLock::new();
    THEME.get_or_init(|| {
        let bytes = include_bytes!("../assets/zestengine-purple.tmTheme");
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        syntect::highlighting::ThemeSet::load_from_reader(&mut cursor)
            .expect("embedded zestengine-purple.tmTheme parses")
    })
}

/// Syntax-highlight `text` into a `LayoutJob` with the ZestEngine Purple theme.
/// Caches the last-built job (keyed by text/lang/font) so syntect doesn't run
/// every frame while the buffer is unchanged.
fn highlight_code(text: &str, language: &str, font: &egui::FontId) -> egui::text::LayoutJob {
    use std::cell::RefCell;
    use std::hash::{Hash, Hasher};

    thread_local! {
        static CACHE: RefCell<Option<(u64, egui::text::LayoutJob)>> = const { RefCell::new(None) };
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    language.hash(&mut hasher);
    font.size.to_bits().hash(&mut hasher);
    let key = hasher.finish();
    if let Some(job) = CACHE.with(|c| match &*c.borrow() {
        Some((k, job)) if *k == key => Some(job.clone()),
        _ => None,
    }) {
        return job;
    }

    let ps = syntaxes();
    let syntax = ps
        .find_syntax_by_extension(language)
        .or_else(|| ps.find_syntax_by_token(language))
        .unwrap_or_else(|| ps.find_syntax_plain_text());
    let mut h = syntect::easy::HighlightLines::new(syntax, purple_theme());

    let mut job = egui::text::LayoutJob::default();
    for line in syntect::util::LinesWithEndings::from(text) {
        match h.highlight_line(line, ps) {
            Ok(regions) => {
                for (style, piece) in regions {
                    let c = style.foreground;
                    job.append(
                        piece,
                        0.0,
                        egui::TextFormat {
                            font_id: font.clone(),
                            color: egui::Color32::from_rgb(c.r, c.g, c.b),
                            italics: style
                                .font_style
                                .contains(syntect::highlighting::FontStyle::ITALIC),
                            ..Default::default()
                        },
                    );
                }
            }
            Err(_) => job.append(
                line,
                0.0,
                egui::TextFormat {
                    font_id: font.clone(),
                    color: egui::Color32::from_rgb(0xC9, 0xC3, 0xDA),
                    ..Default::default()
                },
            ),
        }
    }
    CACHE.with(|c| *c.borrow_mut() = Some((key, job.clone())));
    job
}

/// Paint right-aligned line numbers in the left gutter. Works with wrapping:
/// a number is drawn at the visual row where each logical line begins (found
/// via the galley position of the line's first char), so wrapped continuation
/// rows get no number. Rows outside the viewport are skipped.
fn draw_line_numbers(
    ui: &Ui,
    text: &str,
    output: &egui::text_edit::TextEditOutput,
    gutter_left: f32,
    gutter_w: f32,
    row_h: f32,
    font: &egui::FontId,
) {
    let painter = ui.painter();
    let clip = ui.clip_rect();
    let color = ui.visuals().weak_text_color();
    let x = gutter_left + gutter_w - 5.0;

    let paint = |line: usize, char_start: usize| {
        let rect = output
            .galley
            .pos_from_cursor(CCursor::new(char_start))
            .translate(output.galley_pos.to_vec2());
        let y = rect.top();
        if y + row_h < clip.top() || y > clip.bottom() {
            return;
        }
        painter.text(
            egui::pos2(x, y),
            egui::Align2::RIGHT_TOP,
            line.to_string(),
            font.clone(),
            color,
        );
    };

    paint(1, 0);
    let mut line = 1usize;
    for (i, ch) in text.char_indices() {
        if ch == '\n' {
            line += 1;
            // char_start is the char index just after this newline byte.
            let char_start = text[..=i].chars().count();
            paint(line, char_start);
        }
    }
}

/// Paint translucent rectangles over char ranges (substitution preview). Each
/// range is assumed to sit on one visual row (matches are within a line).
fn draw_ranges(
    ui: &Ui,
    output: &egui::text_edit::TextEditOutput,
    ranges: &[(usize, usize)],
    color: egui::Color32,
) {
    let painter = ui.painter_at(ui.clip_rect());
    let off = output.galley_pos.to_vec2();
    for &(s, e) in ranges {
        let a = output.galley.pos_from_cursor(CCursor::new(s)).translate(off);
        let b = output.galley.pos_from_cursor(CCursor::new(e)).translate(off);
        let rect = if (a.top() - b.top()).abs() < 1.0 && b.right() > a.left() {
            egui::Rect::from_min_max(a.left_top(), b.right_bottom())
        } else {
            // Range wraps rows: just mark its start row to the line's end.
            egui::Rect::from_min_max(a.left_top(), egui::pos2(a.right() + 20.0, a.bottom()))
        };
        painter.rect_filled(rect, 2.0, color);
    }
}

/// Draw a wavy squiggle under each diagnostic line and show its message on
/// hover. Diagnostic lines are 1-based (engine convention).
fn draw_diagnostics(
    ui: &Ui,
    text: &str,
    output: &egui::text_edit::TextEditOutput,
    diagnostics: &[CodeDiagnostic],
) {
    if diagnostics.is_empty() {
        return;
    }
    let painter = ui.painter_at(output.response.rect);
    let hover_pos = output.response.hover_pos();
    let mut tip: Option<(egui::Pos2, String, bool)> = None;

    for d in diagnostics {
        let is_error = d.level == "error";
        let color = if is_error {
            egui::Color32::from_rgb(235, 90, 80)
        } else {
            egui::Color32::from_rgb(220, 180, 70)
        };
        let line0 = d.line.saturating_sub(1) as usize;
        let (s, e) = line_char_range(text, line0);
        let start_rect = output.galley.pos_from_cursor(CCursor::new(s));
        let end_rect = output.galley.pos_from_cursor(CCursor::new(e.max(s)));
        // Single visual row only (wrapped lines underline their first row).
        let same_row = (start_rect.top() - end_rect.top()).abs() < 1.0;
        let y = output.galley_pos.y + start_rect.bottom() - 1.0;
        let x0 = output.galley_pos.x + start_rect.left();
        let x1 = if same_row && end_rect.right() > start_rect.left() {
            output.galley_pos.x + end_rect.right()
        } else {
            x0 + 40.0
        };
        squiggle(&painter, x0, x1.max(x0 + 6.0), y, color);

        // Hover over the row shows a tooltip with the message.
        if let Some(p) = hover_pos {
            let row_top = output.galley_pos.y + start_rect.top();
            let row_bot = output.galley_pos.y + start_rect.bottom();
            if p.y >= row_top && p.y <= row_bot {
                tip = Some((p, d.message.clone(), is_error));
            }
        }
    }

    if let Some((p, msg, is_error)) = tip {
        egui::Area::new(egui::Id::new("diag-tip"))
            .order(egui::Order::Tooltip)
            .fixed_pos(p + egui::vec2(12.0, 16.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(480.0);
                    let color = if is_error {
                        ui.visuals().error_fg_color
                    } else {
                        ui.visuals().warn_fg_color
                    };
                    ui.label(RichText::new(msg).small().color(color));
                });
            });
    }
}

/// Wavy underline between x0 and x1 at baseline y.
fn squiggle(painter: &egui::Painter, x0: f32, x1: f32, y: f32, color: egui::Color32) {
    let step = 3.0;
    let amp = 1.5;
    let mut points = Vec::new();
    let mut x = x0;
    let mut up = true;
    while x <= x1 {
        points.push(egui::pos2(x, y + if up { 0.0 } else { amp }));
        x += step;
        up = !up;
    }
    if points.len() >= 2 {
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.0_f32, color)));
    }
}
