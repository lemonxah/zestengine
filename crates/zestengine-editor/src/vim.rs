//! Vim modal editing for the code editor, driven by the [`vimcore`] engine.
//!
//! The engine owns modes, motions, operators, text objects, registers,
//! search, marks and the ex command line. This module is the seam: it maps
//! egui input onto vimcore keys, adapts the `&mut String` buffer onto
//! [`VimBufferMut`], and funnels engine side effects (`:w`/`:q`, undo
//! groups) into the [`VimOutcome`] shape the code editor consumes.
//!
//! The code editor stores [`VimState`] in egui memory per file and, when
//! enabled and focused, feeds captured events through [`handle`] before the
//! `TextEdit` sees them. Insert-mode typing still flows to the `TextEdit`
//! (completion popup and IME keep working untouched); the engine only
//! watches for Escape there. Undo groups that open without an engine edit
//! (plain `i`/`a`/`A`/`I`/`R`/`gI`) are covered by an insert snapshot this
//! module flushes when the session ends, so `u` still steps back one whole
//! session.
//!
//! `gd`/`gr` (go to definition / references) are intercepted here because
//! the engine has no LSP motions: fed through, `g` would arm its command
//! prefix and `d` would arm the delete operator.

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use egui::Event;
use vimcore::buffer::VimBuffer;
use vimcore::host::VimHost;
use vimcore::key::{Key, KeyKind};
use vimcore::mode::{Mode, VisualKind};
use vimcore::state::{Ctx, VimState as Engine};

/// Single-threaded storage for the engine: egui temp memory requires
/// `Send + Sync`, but vimcore's ex-command registry holds `Rc` handlers.
/// This crate never registers ex commands (the registry stays empty), so
/// the engine is only ever accessed through the mutex on egui's one UI
/// thread.
struct EngineCell(Mutex<Engine>);

// SAFETY: the `Rc`s that make `Engine` !Send live only in the ex-command
// registry, which this crate never populates (`register_ex` is never
// called), so no `Rc` can ever be moved or aliased across threads; every
// access to the cell goes through the mutex on egui's single UI thread.
unsafe impl Send for EngineCell {}
unsafe impl Sync for EngineCell {}

impl EngineCell {
    fn new() -> Self {
        EngineCell(Mutex::new(Engine::new()))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Engine> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Editor-facing mode labels (a projection of [`vimcore::mode::Mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VimMode {
    #[default]
    Normal,
    Insert,
    Visual,
    VisualLine,
    /// `:` / `/` / `?` command-line entry (ex commands, search).
    Command,
}

impl VimMode {
    pub fn label(self) -> &'static str {
        match self {
            VimMode::Normal => "NORMAL",
            VimMode::Insert => "INSERT",
            VimMode::Visual => "VISUAL",
            VimMode::VisualLine => "V-LINE",
            VimMode::Command => "COMMAND",
        }
    }
}

fn map_mode(mode: Mode) -> VimMode {
    match mode {
        Mode::Normal => VimMode::Normal,
        Mode::Insert | Mode::Replace => VimMode::Insert,
        Mode::Visual {
            kind: VisualKind::Line,
        } => VimMode::VisualLine,
        Mode::Visual { .. } => VimMode::Visual,
        Mode::CommandLine { .. } => VimMode::Command,
    }
}

/// Per-file vim modal state, persisted in egui memory.
#[derive(Clone)]
pub struct VimState {
    engine: Arc<EngineCell>,
    /// Original text stashed while a `:` command is being typed, so the live
    /// substitution preview can be reverted on cancel and applied cleanly on
    /// commit. Some only during [`VimMode::Command`].
    pub preview_base: Option<String>,
    /// (text, char cursor) snapshot taken when insert mode was entered
    /// without an engine edit (`i`/`a`/`A`/`I`/`R`/`gI`): flushed as one
    /// undo entry when the session ends, if the buffer changed.
    insert_snapshot: Option<(String, usize)>,
    /// A `g` held for the next key, so `gd`/`gr` can be intercepted before
    /// the engine sees them.
    g_held: bool,
    /// Engine undo group whose `begin` was announced but not yet closed
    /// (the group stays open across a change/insert session).
    open_group: Option<(u64, String, usize)>,
}

impl Default for VimState {
    fn default() -> Self {
        let engine = Arc::new(EngineCell::new());
        {
            let mut e = engine.lock();
            // `o` / `O` (and the engine's own insert-mode newline) open the
            // line already indented to the context, like real vim.
            e.options.autoindent = true;
        }
        VimState {
            engine,
            preview_base: None,
            insert_snapshot: None,
            g_held: false,
            open_group: None,
        }
    }
}

impl VimState {
    /// The current mode as the editor renders it.
    pub fn mode(&self) -> VimMode {
        let engine = self.engine.lock();
        map_mode(engine.mode)
    }

    /// The command-line text after the prompt (`:` / `/` / `?`).
    pub fn cmdline(&self) -> String {
        let engine = self.engine.lock();
        engine.cmdline.buffer.clone()
    }

    /// The open prompt char, or `:` when no prompt is open.
    pub fn prompt(&self) -> char {
        let engine = self.engine.lock();
        match engine.mode {
            Mode::CommandLine { prompt } => prompt,
            _ => ':',
        }
    }
}

/// Per-file vim undo/redo snapshot stacks (text + char cursor), kept in egui
/// memory and driven by the engine's undo-group hooks inside [`handle`].
#[derive(Clone, Default)]
pub struct VimHistory {
    undo: Vec<(String, usize)>,
    redo: Vec<(String, usize)>,
}

#[derive(Default)]
pub struct VimOutcome {
    /// New primary cursor (char index), if it moved.
    pub cursor: Option<usize>,
    /// Visual selection (anchor, cursor) char indices.
    pub selection: Option<(usize, usize)>,
    pub text_changed: bool,
    /// `:w` requested a save.
    pub save: bool,
    /// `:q` requested the tab be closed.
    pub close: bool,
    /// `gd` — go to definition at the cursor.
    pub goto_def: bool,
    /// `gr` — list references at the cursor.
    pub goto_refs: bool,
}

/// Which egui key events the editor should steal for vim (the rest pass
/// through to the `TextEdit` / app shortcuts). Text events are always
/// captured outside insert mode; printable characters arrive as
/// `Event::Text`, so letter/digit `Key` events stay with egui.
pub fn capture_key(key: egui::Key, modifiers: egui::Modifiers) -> bool {
    // Ctrl+R is vim's redo in normal mode.
    if (modifiers.ctrl || modifiers.command) && key == egui::Key::R {
        return true;
    }
    if modifiers.ctrl || modifiers.command || modifiers.alt {
        return false;
    }
    matches!(
        key,
        egui::Key::Enter
            | egui::Key::Tab
            | egui::Key::Backspace
            | egui::Key::Delete
            | egui::Key::Insert
            | egui::Key::Home
            | egui::Key::End
            | egui::Key::PageUp
            | egui::Key::PageDown
            | egui::Key::ArrowLeft
            | egui::Key::ArrowRight
            | egui::Key::ArrowUp
            | egui::Key::ArrowDown
    )
}

/// Process captured events for one frame. `cursor` is the current primary
/// cursor char index. Mutates `text` for edits and `hist` for undo; returns
/// the new cursor / selection / flags to apply to the TextEdit.
pub fn handle(
    state: &mut VimState,
    text: &mut String,
    cursor: usize,
    events: &[Event],
    hist: &mut VimHistory,
) -> VimOutcome {
    let mut out = VimOutcome::default();
    let mut engine = state.engine.lock();
    let before = text.clone();
    let shared = Rc::new(RefCell::new(Shared {
        text: std::mem::take(text),
        hist: std::mem::take(hist),
        save: false,
        close: false,
    }));
    let mut buf = Buf {
        shared: shared.clone(),
    };
    let mut host = Host {
        shared: shared.clone(),
        open: &mut state.open_group,
        cur_hint: None,
        pushed: false,
    };

    // Re-sync the engine cursor with the TextEdit (mouse clicks, pastes and
    // external edits move it without any keys). No-op while a prompt is open
    // and safe in every mode: the engine floors and clamps internally.
    {
        let s = shared.borrow();
        engine.set_cursor_offset(&buf, byte_of_char(&s.text, cursor));
    }

    {
        let g_held = &mut state.g_held;
        let snap = &mut state.insert_snapshot;
        let mut one_key =
            |engine: &mut Engine, buf: &mut Buf, host: &mut Host, out: &mut VimOutcome, mut key: Key| {
                // gd / gr interception: the engine has no LSP motions, and a
                // bare `g` + `d` would arm the delete operator.
                let can_hold_g = matches!(engine.mode, Mode::Normal | Mode::Visual { .. });
                if can_hold_g {
                    if *g_held {
                        *g_held = false;
                        let plain = !key.modifiers.control && !key.modifiers.alt;
                        match (&key.kind, plain) {
                            (KeyKind::Char('d'), true) => {
                                out.goto_def = true;
                                return;
                            }
                            (KeyKind::Char('r'), true) => {
                                out.goto_refs = true;
                                return;
                            }
                            _ => feed(engine, buf, host, Key::char('g')),
                        }
                    } else if key == Key::char('g') {
                        *g_held = true;
                        return;
                    }
                }
                // vim maps <CR> to j outside the command line.
                if key == Key::enter() && !matches!(engine.mode, Mode::CommandLine { .. }) {
                    key = Key::named("down");
                }
                let mode_before = engine.mode;
                feed(engine, buf, host, key);
                let mode_after = engine.mode;
                let was_insert = matches!(mode_before, Mode::Insert | Mode::Replace);
                let is_insert = matches!(mode_after, Mode::Insert | Mode::Replace);
                if !was_insert && is_insert {
                    // Snapshot for sessions the engine never edits itself
                    // (typing goes through the TextEdit); c/o/s/S/C sessions
                    // are covered by the engine's undo group instead and the
                    // snapshot is discarded on exit.
                    let s = shared.borrow();
                    *snap = Some((s.text.clone(), char_of_byte(&s.text, engine.cursor.offset)));
                } else if was_insert && !is_insert {
                    if host.pushed {
                        *snap = None;
                    } else if let Some((snap_text, snap_cursor)) = snap.take() {
                        let mut s = shared.borrow_mut();
                        if s.text != snap_text {
                            s.hist.undo.push((snap_text, snap_cursor));
                            if s.hist.undo.len() > 200 {
                                s.hist.undo.remove(0);
                            }
                            s.hist.redo.clear();
                        }
                    }
                }
            };

        for ev in events {
            match ev {
                Event::Text(t) => {
                    for ch in t.chars() {
                        one_key(&mut engine, &mut buf, &mut host, &mut out, Key::char(ch));
                    }
                }
                Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    if let Some(key) = map_egui_key(*key, *modifiers) {
                        one_key(&mut engine, &mut buf, &mut host, &mut out, key);
                    }
                }
                _ => {}
            }
        }
    }

    drop(buf);
    drop(host);
    let shared = match Rc::try_unwrap(shared) {
        Ok(shared) => shared.into_inner(),
        Err(_) => unreachable!("vim buffer/host adapters dropped"),
    };
    *text = shared.text;
    *hist = shared.hist;
    out.save = shared.save;
    out.close = shared.close;
    out.text_changed = *text != before;

    let cur = char_of_byte(text, engine.cursor.offset).min(text.chars().count());
    out.cursor = Some(cur);
    if let Some((anchor, cur, kind)) = engine.visual_selection() {
        out.selection = Some(selection_range(text, anchor, cur, kind));
    }
    out
}

/// State shared between the buffer and host adapters for one [`handle`] call.
struct Shared {
    text: String,
    hist: VimHistory,
    save: bool,
    close: bool,
}

/// Buffer adapter over the edited `String`. All offsets are UTF-8 bytes, the
/// engine's coordinate space.
struct Buf {
    shared: Rc<RefCell<Shared>>,
}

impl VimBuffer for Buf {
    fn len(&self) -> usize {
        self.shared.borrow().text.len()
    }

    fn line_count(&self) -> usize {
        self.shared.borrow().text.bytes().filter(|&b| b == b'\n').count() + 1
    }

    fn char_at(&self, offset: usize) -> Option<char> {
        self.shared.borrow().text.get(offset..)?.chars().next()
    }

    fn prev_char_offset(&self, offset: usize) -> Option<usize> {
        let s = self.shared.borrow();
        let b = offset.min(s.text.len());
        if b == 0 {
            return None;
        }
        s.text[..b].char_indices().next_back().map(|(i, _)| i)
    }

    fn line_range(&self, line: usize) -> Range<usize> {
        let s = self.shared.borrow();
        let mut start = 0usize;
        let mut seen = 0usize;
        if line > 0 {
            for (i, c) in s.text.char_indices() {
                if c == '\n' {
                    seen += 1;
                    if seen == line {
                        start = i + 1;
                        break;
                    }
                }
            }
            if seen < line {
                return s.text.len()..s.text.len();
            }
        }
        let mut end = s.text.len();
        for (i, c) in s.text[start..].char_indices() {
            if c == '\n' {
                end = start + i + 1;
                break;
            }
        }
        start..end
    }

    fn offset_to_line(&self, offset: usize) -> usize {
        let s = self.shared.borrow();
        let b = offset.min(s.text.len());
        s.text[..b].bytes().filter(|&c| c == b'\n').count()
    }

    fn slice(&self, range: Range<usize>) -> String {
        let s = self.shared.borrow();
        let range = range.start.min(s.text.len())..range.end.min(s.text.len());
        s.text[range].to_string()
    }
}

impl vimcore::buffer::VimBufferMut for Buf {
    fn insert_text(&mut self, offset: usize, text: &str) -> bool {
        let mut s = self.shared.borrow_mut();
        if offset > s.text.len() || !s.text.is_char_boundary(offset) {
            return false;
        }
        s.text.insert_str(offset, text);
        true
    }

    fn delete_range(&mut self, range: Range<usize>) -> bool {
        let mut s = self.shared.borrow_mut();
        if range.start > range.end
            || range.end > s.text.len()
            || !s.text.is_char_boundary(range.start)
            || !s.text.is_char_boundary(range.end)
        {
            return false;
        }
        s.text.replace_range(range, "");
        true
    }

    fn replace_range(&mut self, range: Range<usize>, text: &str) -> bool {
        let mut s = self.shared.borrow_mut();
        if range.start > range.end
            || range.end > s.text.len()
            || !s.text.is_char_boundary(range.start)
            || !s.text.is_char_boundary(range.end)
        {
            return false;
        }
        s.text.replace_range(range, text);
        true
    }
}

/// Host adapter: collects side effects the engine reports and drives the
/// per-file undo stacks with the engine's undo-group hooks.
struct Host<'a> {
    shared: Rc<RefCell<Shared>>,
    /// Slot persisted on [`VimState`]: the undo group can stay open across
    /// several frames (a change/insert session).
    open: &'a mut Option<(u64, String, usize)>,
    /// Char cursor before the current key, for redo entries.
    cur_hint: Option<usize>,
    /// An engine undo group closed (an entry was pushed) during this call.
    pushed: bool,
}

impl VimHost for Host<'_> {
    fn viewport(&self) -> (usize, usize) {
        (0, 0)
    }

    fn scroll_to_line(&mut self, _line: usize) {}

    fn clipboard_write(&mut self, _text: &str) {}

    fn clipboard_read(&self) -> Option<String> {
        None
    }

    fn set_search_highlights(&mut self, _matches: &[Range<usize>], _current: Option<Range<usize>>) {}

    fn begin_undo_group(&mut self, id: u64, cursor_offset: usize) {
        let mut s = self.shared.borrow_mut();
        if self.open.as_ref().map(|(i, _, _)| *i) != Some(id) {
            // A fresh change invalidates redo (vim's undo tree).
            s.hist.redo.clear();
        }
        let cursor = char_of_byte(&s.text, cursor_offset);
        *self.open = Some((id, s.text.clone(), cursor));
    }

    fn end_undo_group(&mut self, id: u64, _cursor_offset: usize) {
        if let Some((open_id, before_text, before_cursor)) = self.open.take()
            && open_id == id
        {
            let mut s = self.shared.borrow_mut();
            s.hist.undo.push((before_text, before_cursor));
            if s.hist.undo.len() > 200 {
                s.hist.undo.remove(0);
            }
            self.pushed = true;
        }
    }

    fn undo(&mut self) -> Option<usize> {
        let mut s = self.shared.borrow_mut();
        let (prev_text, prev_cursor) = s.hist.undo.pop()?;
        let cur = s.text.clone();
        let hint = self.cur_hint.take().unwrap_or(prev_cursor);
        s.hist.redo.push((cur, hint));
        s.text = prev_text;
        self.cur_hint = Some(prev_cursor);
        Some(byte_of_char(&s.text, prev_cursor))
    }

    fn redo(&mut self) -> Option<usize> {
        let mut s = self.shared.borrow_mut();
        let (next_text, next_cursor) = s.hist.redo.pop()?;
        let cur = s.text.clone();
        let hint = self.cur_hint.take().unwrap_or(next_cursor);
        s.hist.undo.push((cur, hint));
        s.text = next_text;
        self.cur_hint = Some(next_cursor);
        Some(byte_of_char(&s.text, next_cursor))
    }

    fn save(&mut self) {
        self.shared.borrow_mut().save = true;
    }

    fn request_close(&mut self) {
        self.shared.borrow_mut().close = true;
    }
}

/// Feed one key to the engine, remembering the cursor before it so undo/redo
/// entries can restore to the right place.
fn feed(engine: &mut Engine, buf: &mut Buf, host: &mut Host<'_>, key: Key) {
    let cur = {
        let s = buf.shared.borrow();
        char_of_byte(&s.text, engine.cursor.offset)
    };
    host.cur_hint = Some(cur);
    engine.handle_key(
        &mut Ctx {
            buf,
            host,
        },
        key,
    );
}

/// egui key event -> vimcore key. Text events carry the printable characters
/// separately, so letter/digit keys never need mapping here.
fn map_egui_key(key: egui::Key, modifiers: egui::Modifiers) -> Option<Key> {
    if modifiers.ctrl || modifiers.command {
        return (key == egui::Key::R).then(|| Key::ctrl_char('r'));
    }
    if modifiers.alt {
        return None;
    }
    Some(match key {
        egui::Key::Escape => Key::escape(),
        egui::Key::Enter => Key::enter(),
        egui::Key::Tab => Key::tab(),
        egui::Key::Backspace => Key::backspace(),
        egui::Key::Delete => Key::named("delete"),
        egui::Key::Insert => Key::named("insert"),
        egui::Key::Home => Key::named("home"),
        egui::Key::End => Key::named("end"),
        egui::Key::ArrowLeft => Key::named("left"),
        egui::Key::ArrowRight => Key::named("right"),
        egui::Key::ArrowUp => Key::named("up"),
        egui::Key::ArrowDown => Key::named("down"),
        egui::Key::PageUp => Key::named("pageup"),
        egui::Key::PageDown => Key::named("pagedown"),
        _ => return None,
    })
}

/// Engine visual selection (byte offsets) -> the egui char range the old
/// editor produced: inclusive of the cursor char for charwise, extended over
/// the line for linewise.
fn selection_range(text: &str, anchor: usize, cursor: usize, kind: VisualKind) -> (usize, usize) {
    let len = text.chars().count();
    let a = char_of_byte(text, anchor.min(text.len()));
    let b = char_of_byte(text, cursor.min(text.len()));
    let (mut lo, mut hi) = (a.min(b), a.max(b));
    match kind {
        VisualKind::Line => {
            lo = line_start_char(text, lo);
            hi = line_end_char(text, hi);
            if hi < len {
                hi += 1;
            }
        }
        VisualKind::Char | VisualKind::Block => {
            hi = (hi + 1).min(len);
        }
    }
    (lo, hi)
}

fn char_of_byte(text: &str, byte: usize) -> usize {
    let mut b = byte.min(text.len());
    while b > 0 && !text.is_char_boundary(b) {
        b -= 1;
    }
    text[..b].chars().count()
}

fn byte_of_char(text: &str, idx: usize) -> usize {
    text.char_indices()
        .nth(idx)
        .map(|(b, _)| b)
        .unwrap_or(text.len())
}

fn line_start_char(text: &str, idx: usize) -> usize {
    let idx = idx.min(text.chars().count());
    let mut s = 0;
    for (i, c) in text.chars().enumerate() {
        if i == idx {
            break;
        }
        if c == '\n' {
            s = i + 1;
        }
    }
    s
}

fn line_end_char(text: &str, idx: usize) -> usize {
    let mut e = idx.min(text.chars().count());
    for c in text.chars().skip(e) {
        if c == '\n' {
            break;
        }
        e += 1;
    }
    e
}

/// Parsed `[%]s/pat/rep/[flags]` command.
struct Subst<'a> {
    whole: bool,
    pat: &'a str,
    rep: &'a str,
    has_rep: bool,
    global: bool,
}

/// Parse `[%]s<delim>pat<delim>rep<delim>flags`. The delimiter is the char
/// after `s`/`%s` and must be punctuation. Returns None if `rest` isn't a
/// substitution.
fn parse_subst(whole: bool, rest: &str) -> Option<Subst<'_>> {
    let delim = rest.chars().next()?;
    if delim.is_alphanumeric() || delim.is_whitespace() {
        return None;
    }
    let parts: Vec<&str> = rest[delim.len_utf8()..].split(delim).collect();
    let pat = parts.first().filter(|p| !p.is_empty())?;
    Some(Subst {
        whole,
        pat,
        rep: parts.get(1).copied().unwrap_or(""),
        has_rep: parts.len() >= 2,
        global: parts.get(2).is_some_and(|f| f.contains('g')),
    })
}

/// Result of previewing a substitution against a base text.
pub struct SubstPreview {
    /// The text with the substitution applied (or the base when only a pattern
    /// has been typed).
    pub text: String,
    /// Char ranges to highlight in `text` (replaced spans, or match spans when
    /// no replacement is given yet).
    pub highlights: Vec<(usize, usize)>,
    /// True once a replacement is being applied (vs just highlighting matches).
    pub replaced: bool,
}

/// Apply a regex substitution line by line, tracking the char ranges of the
/// replacements in the output (for highlighting). Mirrors what commit does.
fn apply_subst(
    base: &str,
    re: &regex::Regex,
    rep: &str,
    whole: bool,
    global: bool,
    cur_line: usize,
) -> (String, Vec<(usize, usize)>) {
    let mut out = String::new();
    let mut ranges = Vec::new();
    let mut off = 0usize; // chars written so far
    for (idx, line) in base.split('\n').enumerate() {
        if idx > 0 {
            out.push('\n');
            off += 1;
        }
        if !(whole || idx == cur_line) {
            out.push_str(line);
            off += line.chars().count();
            continue;
        }
        let mut last = 0usize; // byte cursor in `line`
        for (n, caps) in re.captures_iter(line).enumerate() {
            if !global && n >= 1 {
                break;
            }
            let m = caps.get(0).unwrap();
            let prefix = &line[last..m.start()];
            out.push_str(prefix);
            off += prefix.chars().count();
            let mut rep_str = String::new();
            caps.expand(rep, &mut rep_str);
            let start = off;
            out.push_str(&rep_str);
            off += rep_str.chars().count();
            ranges.push((start, off));
            last = m.end();
        }
        let tail = &line[last..];
        out.push_str(tail);
        off += tail.chars().count();
    }
    (out, ranges)
}

/// Char ranges of every match (for pattern-only highlighting).
fn match_ranges(base: &str, re: &regex::Regex, whole: bool, cur_line: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut off = 0usize;
    for (idx, line) in base.split('\n').enumerate() {
        if idx > 0 {
            off += 1;
        }
        let line_start = off;
        if whole || idx == cur_line {
            for m in re.find_iter(line) {
                let s = line_start + line[..m.start()].chars().count();
                let e = line_start + line[..m.end()].chars().count();
                ranges.push((s, e));
            }
        }
        off += line.chars().count();
    }
    ranges
}

fn cur_line_of(base: &str, cur_char: usize) -> usize {
    base.chars().take(cur_char).filter(|&c| c == '\n').count()
}

/// Compute a live preview of a substitution command against `base`. Returns
/// None if `cmd` isn't a substitution. On a bad regex, returns the base text
/// unchanged with no highlights. The commit itself runs in the engine (vim's
/// `:s`), so a preview may differ from the final text on exotic regexes.
pub fn preview_substitute(base: &str, cmd: &str, cur_char: usize) -> Option<SubstPreview> {
    let (whole, rest) = if let Some(r) = cmd.strip_prefix("%s") {
        (true, r)
    } else {
        let r = cmd.strip_prefix('s')?;
        (false, r)
    };
    let sub = parse_subst(whole, rest)?;
    let re = match regex::Regex::new(sub.pat) {
        Ok(re) => re,
        Err(_) => {
            return Some(SubstPreview {
                text: base.to_string(),
                highlights: Vec::new(),
                replaced: false,
            });
        }
    };
    let cur_line = cur_line_of(base, cur_char);
    if sub.has_rep {
        let (text, highlights) = apply_subst(base, &re, sub.rep, sub.whole, sub.global, cur_line);
        Some(SubstPreview {
            text,
            highlights,
            replaced: true,
        })
    } else {
        Some(SubstPreview {
            highlights: match_ranges(base, &re, sub.whole, cur_line),
            text: base.to_string(),
            replaced: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mini version of the editor's frame loop: feeds key strings into
    /// [`handle`] and applies the returned cursor, keeping the per-file
    /// history like egui memory does.
    struct Driver {
        state: VimState,
        hist: VimHistory,
        text: String,
        cursor: usize,
    }

    impl Driver {
        fn new(text: &str, cursor: usize) -> Driver {
            Driver {
                state: VimState::default(),
                hist: VimHistory::default(),
                text: text.to_string(),
                cursor,
            }
        }

        fn keys(&mut self, keys: &str) -> VimOutcome {
            let mut out = VimOutcome::default();
            for ch in keys.chars() {
                let ev = match ch {
                    '\u{1b}' => Some(key_event(egui::Key::Escape, egui::Modifiers::NONE)),
                    '\r' => Some(key_event(egui::Key::Enter, egui::Modifiers::NONE)),
                    '\u{12}' => Some(key_event(egui::Key::R, egui::Modifiers::CTRL)),
                    c if self.state.mode() == VimMode::Insert => {
                        // The editor leaves insert-mode typing to the TextEdit.
                        let at = byte_of_char(&self.text, self.cursor);
                        self.text.insert(at, c);
                        self.cursor += 1;
                        None
                    }
                    c => Some(Event::Text(c.to_string())),
                };
                if let Some(ev) = ev {
                    let captured = [ev];
                    out = handle(&mut self.state, &mut self.text, self.cursor, &captured, &mut self.hist);
                    if let Some(c) = out.cursor {
                        self.cursor = c;
                    }
                }
            }
            out
        }
    }

    fn key_event(key: egui::Key, modifiers: egui::Modifiers) -> Event {
        Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }

    #[test]
    fn diw_deletes_inner_word() {
        let mut d = Driver::new("foo bar baz", 4); // on the 'b' of "bar"
        let out = d.keys("diw");
        assert_eq!(d.text, "foo  baz");
        assert!(out.cursor.is_some());
        assert_eq!(d.state.mode(), VimMode::Normal);
    }

    #[test]
    fn ciw_changes_inner_word() {
        let mut d = Driver::new("foo bar baz", 4);
        let out = d.keys("ciw");
        assert_eq!(d.text, "foo  baz");
        assert!(out.cursor.is_some());
        assert_eq!(d.state.mode(), VimMode::Insert);
        d.keys("X\u{1b}");
        assert_eq!(d.text, "foo X baz");
        assert_eq!(d.state.mode(), VimMode::Normal);
    }

    #[test]
    fn ci_quote_changes_inside_quotes() {
        let mut d = Driver::new("say \"hello world\" ok", 5); // on the 'h'
        let out = d.keys("ci\"");
        assert_eq!(d.text, "say \"\" ok");
        assert!(out.cursor.is_some());
        assert_eq!(d.state.mode(), VimMode::Insert);
        d.keys("hi\u{1b}");
        assert_eq!(d.text, "say \"hi\" ok");
        assert_eq!(d.state.mode(), VimMode::Normal);
    }

    #[test]
    fn registers_yank_and_put() {
        let mut d = Driver::new("foo bar", 0);
        d.keys("yiw$p");
        assert_eq!(d.text, "foo barfoo");
    }

    #[test]
    fn undo_redo_normal_edit() {
        let mut d = Driver::new("foo", 0);
        d.keys("x");
        assert_eq!(d.text, "oo");
        d.keys("u");
        assert_eq!(d.text, "foo");
        d.keys("\u{12}");
        assert_eq!(d.text, "oo");
    }

    #[test]
    fn undo_insert_session_is_one_entry() {
        let mut d = Driver::new("foo", 0);
        d.keys("iX\u{1b}");
        assert_eq!(d.text, "Xfoo");
        assert_eq!(d.cursor, 0); // insert exit steps left
        d.keys("u");
        assert_eq!(d.text, "foo");
    }

    #[test]
    fn substitute_current_line_and_whole_buffer() {
        let mut d = Driver::new("one two one\n", 0);
        d.keys(":s/one/ONE/\r");
        assert_eq!(d.text, "ONE two one\n");
        d.keys(":%s/one/ONE/g\r");
        assert_eq!(d.text, "ONE two ONE\n");
    }

    #[test]
    fn ex_save_and_quit_flags() {
        let mut d = Driver::new("foo", 0);
        let out = d.keys(":w\r");
        assert!(out.save);
        assert!(!out.close);
        let out = d.keys(":q\r");
        assert!(out.close);
    }

    #[test]
    fn gd_gr_report_lsp_actions() {
        let mut d = Driver::new("foo", 0);
        let out = d.keys("gd");
        assert!(out.goto_def);
        assert!(!out.goto_refs);
        let out = d.keys("gr");
        assert!(out.goto_refs);
    }

    #[test]
    fn gg_and_g_motions() {
        let mut d = Driver::new("a\nb\nc", 0);
        let out = d.keys("G");
        assert_eq!(out.cursor, Some(4)); // on the 'c'
        let out = d.keys("gg");
        assert_eq!(out.cursor, Some(0));
    }

    #[test]
    fn visual_linewise_selection() {
        let mut d = Driver::new("foo\nbar", 0);
        let out = d.keys("Vj");
        assert_eq!(out.selection, Some((0, 7))); // both lines, incl. newline
    }

    #[test]
    fn preview_substitute_matches_and_replaces() {
        let prev = preview_substitute("foo bar", "s/foo", 0).unwrap();
        assert_eq!(prev.text, "foo bar");
        assert_eq!(prev.highlights, vec![(0, 3)]);
        assert!(!prev.replaced);
        let prev = preview_substitute("foo bar", "s/foo/X/", 0).unwrap();
        assert_eq!(prev.text, "X bar");
        assert!(prev.replaced);
    }
}
