//! Embedded Neovim: a real `nvim --embed` child process driven over its
//! msgpack-RPC UI protocol (the Neovide model), rendered into an egui dock
//! tab. Floats/popupmenus (`ext_multigrid`), clipboard provider registration,
//! and scrollbars are intentionally left for later stages.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use nvim_rs::compat::tokio::Compat;
use nvim_rs::create::tokio::new_child_cmd;
use nvim_rs::rpc::model::IntoVal;
use nvim_rs::{Handler, Neovim, UiAttachOptions, Value, call_args};
use tokio::process::ChildStdin;
use tokio::sync::mpsc::{Receiver, Sender, channel};

const FONT_SIZE: f32 = 16.0;
const MIN_COLS: u16 = 2;
const MAX_COLS: u16 = 400;
const MIN_ROWS: u16 = 2;
const MAX_ROWS: u16 = 300;
const DEFAULT_COLS: u16 = 80;
const DEFAULT_ROWS: u16 = 24;
const RESIZE_COOLDOWN: Duration = Duration::from_millis(300);
const RESIZE_RETRY_AFTER: Duration = Duration::from_millis(500);
const WHEEL_STEP: f32 = 50.0;

#[derive(Default, Clone, Debug)]
enum ProcState {
    #[default]
    Starting,
    Running,
    Exited(String),
    Failed(String),
}

#[derive(Default, Clone)]
struct Cell {
    text: String,
    hl: u64,
}

#[derive(Default, Clone)]
struct HlAttr {
    fg: Option<u32>,
    bg: Option<u32>,
    special: Option<u32>,
    reverse: bool,
    italic: bool,
    underline: bool,
    undercurl: bool,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum CursorShape {
    #[default]
    Block,
    Horizontal,
    Vertical,
}

#[derive(Default)]
struct UiState {
    proc: ProcState,
    width: u16,
    height: u16,
    cells: Vec<Cell>,
    hl: HashMap<u64, HlAttr>,
    default_fg: egui::Color32,
    default_bg: egui::Color32,
    default_sp: egui::Color32,
    cursor_row: u16,
    cursor_col: u16,
    cursor_shapes: HashMap<u64, CursorShape>,
    mode_idx: u64,
    mode: String,
    generation: u64,
    resize_retry_after: Option<Instant>,
}

impl UiState {
    fn set_cell(&mut self, row: u16, col: u16, cell: Cell) {
        let idx = row as usize * self.width as usize + col as usize;
        if idx < self.cells.len() {
            self.cells[idx] = cell;
        }
    }
}

enum NvimCmd {
    Input(String),
    Mouse {
        button: &'static str,
        action: &'static str,
        modifier: String,
        row: i64,
        col: i64,
    },
    Resize {
        width: i64,
        height: i64,
    },
}

#[derive(Clone)]
struct BridgeHandler {
    state: Arc<Mutex<UiState>>,
    wake: Arc<Condvar>,
}

#[async_trait::async_trait]
impl Handler for BridgeHandler {
    type Writer = Compat<ChildStdin>;

    async fn handle_notify(&self, name: String, args: Vec<Value>, _nvim: Neovim<Self::Writer>) {
        if name != "redraw" {
            return;
        }
        let mut s = self.state.lock().unwrap();
        let mut flushed = false;
        for batch in &args {
            let Some(events) = batch.as_array() else { continue };
            let Some(name) = events.first().and_then(Value::as_str) else { continue };
            for tuple in &events[1..] {
                let params: Vec<&Value> = match tuple {
                    Value::Array(inner) => inner.iter().collect(),
                    other => vec![other],
                };
                flushed |= apply_redraw(&mut s, name, &params);
            }
        }
        if flushed {
            s.generation += 1;
            drop(s);
            self.wake.notify_all();
        }
    }
}

fn apply_redraw(s: &mut UiState, name: &str, p: &[&Value]) -> bool {
    match name {
        "grid_resize" => grid_resize(s, p),
        "grid_line" => grid_line(s, p),
        "grid_clear" => {
            if arg_u64(p, 0) == Some(1) {
                s.cells.fill(Cell::default());
            }
            false
        }
        "grid_cursor_goto" => grid_cursor_goto(s, p),
        "grid_scroll" => grid_scroll(s, p),
        "hl_attr_define" => hl_attr_define(s, p),
        "default_colors_set" => default_colors_set(s, p),
        "mode_change" => mode_change(s, p),
        "mode_info_set" => mode_info_set(s, p),
        "flush" => true,
        _ => false,
    }
}

fn arg_u64(p: &[&Value], i: usize) -> Option<u64> {
    p.get(i).and_then(|v| v.as_u64())
}

fn arg_i64(p: &[&Value], i: usize) -> Option<i64> {
    p.get(i).and_then(|v| v.as_i64())
}

fn arg_bool(p: &[&Value], i: usize) -> Option<bool> {
    p.get(i).and_then(|v| v.as_bool())
}

fn grid_resize(s: &mut UiState, p: &[&Value]) -> bool {
    let (Some(grid), Some(w), Some(h)) = (arg_u64(p, 0), arg_u64(p, 1), arg_u64(p, 2)) else {
        return false;
    };
    if grid != 1 {
        return false;
    }
    s.width = w.min(MAX_COLS as u64) as u16;
    s.height = h.min(MAX_ROWS as u64) as u16;
    s.cells = vec![Cell::default(); s.width as usize * s.height as usize];
    false
}

fn grid_line(s: &mut UiState, p: &[&Value]) -> bool {
    let (Some(grid), Some(mut row), Some(mut col)) = (arg_u64(p, 0), arg_u64(p, 1), arg_u64(p, 2)) else {
        return false;
    };
    if grid != 1 || row as u16 >= s.height {
        return false;
    }
    let Some(cells) = p.get(3).and_then(|v| v.as_array()) else { return false };
    let wrap = arg_bool(p, 4).unwrap_or(false);
    for cell in cells {
        let Some(parts) = cell.as_array() else { continue };
        let text = parts.first().and_then(Value::as_str).unwrap_or("").to_owned();
        let hl = parts.get(1).and_then(Value::as_u64).unwrap_or(0);
        let repeat = parts.get(2).and_then(Value::as_u64).unwrap_or(1).max(1);
        for _ in 0..repeat {
            if col >= s.width as u64 {
                if wrap && row + 1 < s.height as u64 {
                    row += 1;
                    col = 0;
                } else {
                    break;
                }
            }
            s.set_cell(row as u16, col as u16, Cell { text: text.clone(), hl });
            col += 1;
        }
        if col >= s.width as u64 && (!wrap || row + 1 >= s.height as u64) {
            break;
        }
    }
    false
}

fn grid_cursor_goto(s: &mut UiState, p: &[&Value]) -> bool {
    let (Some(grid), Some(row), Some(col)) = (arg_u64(p, 0), arg_u64(p, 1), arg_u64(p, 2)) else {
        return false;
    };
    if grid != 1 {
        return false;
    }
    s.cursor_row = row as u16;
    s.cursor_col = col as u16;
    false
}

fn grid_scroll(s: &mut UiState, p: &[&Value]) -> bool {
    let (Some(grid), Some(top), Some(bot)) = (arg_u64(p, 0), arg_u64(p, 1), arg_u64(p, 2)) else {
        return false;
    };
    let rows = arg_i64(p, 5).unwrap_or(0);
    if grid != 1 || rows == 0 || s.height == 0 {
        return false;
    }
    let w = s.width as usize;
    let top = (top as usize).min(s.height as usize - 1);
    let bot = (bot as usize).min(s.height as usize - 1) + 1;
    if top >= bot {
        return false;
    }
    let n = rows.unsigned_abs() as usize;
    if n >= bot - top {
        for r in top..bot {
            blank_row(&mut s.cells, r, w);
        }
        return false;
    }
    if rows > 0 {
        for r in top..bot {
            let src = r + n;
            for c in 0..w {
                s.cells[r * w + c] = if src < bot { s.cells[src * w + c].clone() } else { Cell::default() };
            }
        }
    } else {
        for r in (top..(bot - n)).rev() {
            for c in 0..w {
                s.cells[(r + n) * w + c] = s.cells[r * w + c].clone();
            }
        }
        for r in top..(top + n) {
            blank_row(&mut s.cells, r, w);
        }
    }
    false
}

fn blank_row(cells: &mut [Cell], r: usize, w: usize) {
    cells[r * w..(r + 1) * w].fill(Cell::default());
}

fn hl_attr_define(s: &mut UiState, p: &[&Value]) -> bool {
    let Some(id) = arg_u64(p, 0) else { return false };
    let Some(rgb) = p.get(1).and_then(|v| v.as_map()) else { return false };
    let mut attr = HlAttr::default();
    for (key, val) in rgb {
        let Some(key) = key.as_u64() else { continue };
        match key {
            0 => attr.fg = val.as_u64().map(|c| c as u32),
            1 => attr.bg = val.as_u64().map(|c| c as u32),
            2 => attr.special = val.as_u64().map(|c| c as u32),
            3 => attr.reverse = val.as_bool().unwrap_or(false),
            4 => attr.italic = val.as_bool().unwrap_or(false),
            7 => attr.underline = val.as_bool().unwrap_or(false),
            8 => attr.undercurl = val.as_bool().unwrap_or(false),
            _ => {}
        }
    }
    s.hl.insert(id, attr);
    false
}

fn color32(color: Option<u64>) -> Option<egui::Color32> {
    color.map(|c| {
        let c = c as u32;
        egui::Color32::from_rgb(((c >> 16) & 0xff) as u8, ((c >> 8) & 0xff) as u8, (c & 0xff) as u8)
    })
}

fn default_colors_set(s: &mut UiState, p: &[&Value]) -> bool {
    s.default_fg = color32(arg_u64(p, 0)).unwrap_or(s.default_fg);
    s.default_bg = color32(arg_u64(p, 1)).unwrap_or(s.default_bg);
    s.default_sp = color32(arg_u64(p, 2)).unwrap_or(s.default_sp);
    false
}

fn mode_change(s: &mut UiState, p: &[&Value]) -> bool {
    if let Some(name) = p.first().and_then(|v| v.as_str()) {
        s.mode = name.to_owned();
    }
    s.mode_idx = arg_u64(p, 1).unwrap_or(0);
    false
}

fn mode_info_set(s: &mut UiState, p: &[&Value]) -> bool {
    let Some(infos) = p.get(1).and_then(|v| v.as_array()) else { return false };
    for info in infos {
        let Some(parts) = info.as_array() else { continue };
        let (Some(idx), Some(map)) = (
            parts.first().and_then(Value::as_u64),
            parts.get(1).and_then(Value::as_map),
        ) else {
            continue;
        };
        let mut shape = CursorShape::Block;
        for (key, val) in map {
            if key.as_str() == Some("cursor_shape") {
                shape = match val.as_str() {
                    Some("horizontal") => CursorShape::Horizontal,
                    Some("vertical") => CursorShape::Vertical,
                    _ => CursorShape::Block,
                };
            }
        }
        s.cursor_shapes.insert(idx, shape);
    }
    false
}

fn color32_u32(color: u32) -> egui::Color32 {
    egui::Color32::from_rgb(((color >> 16) & 0xff) as u8, ((color >> 8) & 0xff) as u8, (color & 0xff) as u8)
}

fn format_for(s: &UiState, hl: u64, font_id: &egui::FontId) -> egui::TextFormat {
    let mut fg = s.default_fg;
    let mut bg = s.default_bg;
    let mut italics = false;
    let mut underline = egui::Stroke::NONE;
    if let Some(attr) = s.hl.get(&hl) {
        fg = attr.fg.map(color32_u32).unwrap_or(fg);
        bg = attr.bg.map(color32_u32).unwrap_or(bg);
        if attr.reverse {
            std::mem::swap(&mut fg, &mut bg);
        }
        italics = attr.italic;
        if attr.underline || attr.undercurl {
            underline = egui::Stroke::new(1.0_f32, attr.special.map(color32_u32).unwrap_or(fg));
        }
    }
    egui::TextFormat {
        font_id: font_id.clone(),
        color: fg,
        background: bg,
        italics,
        underline,
        valign: egui::Align::Center,
        ..Default::default()
    }
}

fn build_job(s: &UiState, font_id: &egui::FontId) -> egui::text::LayoutJob {
    let w = s.width as usize;
    let h = s.height as usize;
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = f32::INFINITY;
    let mut formats: HashMap<u64, egui::TextFormat> = HashMap::new();
    let mut cur: Option<(u64, egui::TextFormat, usize)> = None;
    for r in 0..h {
        for c in 0..w {
            let cell = &s.cells[r * w + c];
            let fmt = formats
                .entry(cell.hl)
                .or_insert_with(|| format_for(s, cell.hl, font_id))
                .clone();
            let keep = matches!(&cur, Some((hl, f, _)) if *hl == cell.hl);
            if !keep {
                if let Some((_, f, start)) = cur.take() {
                    job.sections.push(egui::text::LayoutSection {
                        leading_space: 0.0,
                        byte_range: start..job.text.len(),
                        format: f,
                    });
                }
                cur = Some((cell.hl, fmt, job.text.len()));
            }
            job.text.push_str(if cell.text.is_empty() { " " } else { cell.text.as_str() });
        }
        if let Some((_, f, start)) = cur.take() {
            job.sections.push(egui::text::LayoutSection {
                leading_space: 0.0,
                byte_range: start..job.text.len(),
                format: f,
            });
        }
        if r + 1 < h {
            job.text.push('\n');
        }
    }
    job
}

fn escape_input(input: &str) -> String {
    input.replace('<', "<lt>")
}

fn ctrl_sequence(key: &egui::Key, modifiers: &egui::Modifiers) -> Option<String> {
    let byte = match key {
        egui::Key::A => 0x01,
        egui::Key::B => 0x02,
        egui::Key::C => 0x03,
        egui::Key::D => 0x04,
        egui::Key::E => 0x05,
        egui::Key::F => 0x06,
        egui::Key::G => 0x07,
        egui::Key::H => 0x08,
        egui::Key::I => 0x09,
        egui::Key::J => 0x0a,
        egui::Key::K => 0x0b,
        egui::Key::L => 0x0c,
        egui::Key::M => 0x0d,
        egui::Key::N => 0x0e,
        egui::Key::O => 0x0f,
        egui::Key::P => 0x10,
        egui::Key::Q => 0x11,
        egui::Key::R => 0x12,
        egui::Key::S => 0x13,
        egui::Key::T => 0x14,
        egui::Key::U => 0x15,
        egui::Key::V => 0x16,
        egui::Key::W => 0x17,
        egui::Key::X => 0x18,
        egui::Key::Y => 0x19,
        egui::Key::Z => 0x1a,
        egui::Key::Space => 0x00,
        egui::Key::OpenBracket => 0x1b,
        egui::Key::Backslash => 0x1c,
        egui::Key::CloseBracket => 0x1d,
        egui::Key::Minus | egui::Key::Slash => 0x1f,
        egui::Key::Num6 if modifiers.shift => 0x1e,
        _ => return None,
    };
    Some(char::from(byte).to_string())
}

fn alt_sequence(key: &egui::Key) -> Option<String> {
    if let Some(ch) = match key {
        egui::Key::A => Some('a'),
        egui::Key::B => Some('b'),
        egui::Key::C => Some('c'),
        egui::Key::D => Some('d'),
        egui::Key::E => Some('e'),
        egui::Key::F => Some('f'),
        egui::Key::G => Some('g'),
        egui::Key::H => Some('h'),
        egui::Key::I => Some('i'),
        egui::Key::J => Some('j'),
        egui::Key::K => Some('k'),
        egui::Key::L => Some('l'),
        egui::Key::M => Some('m'),
        egui::Key::N => Some('n'),
        egui::Key::O => Some('o'),
        egui::Key::P => Some('p'),
        egui::Key::Q => Some('q'),
        egui::Key::R => Some('r'),
        egui::Key::S => Some('s'),
        egui::Key::T => Some('t'),
        egui::Key::U => Some('u'),
        egui::Key::V => Some('v'),
        egui::Key::W => Some('w'),
        egui::Key::X => Some('x'),
        egui::Key::Y => Some('y'),
        egui::Key::Z => Some('z'),
        egui::Key::Num0 => Some('0'),
        egui::Key::Num1 => Some('1'),
        egui::Key::Num2 => Some('2'),
        egui::Key::Num3 => Some('3'),
        egui::Key::Num4 => Some('4'),
        egui::Key::Num5 => Some('5'),
        egui::Key::Num6 => Some('6'),
        egui::Key::Num7 => Some('7'),
        egui::Key::Num8 => Some('8'),
        egui::Key::Num9 => Some('9'),
        egui::Key::Minus => Some('-'),
        egui::Key::Comma => Some(','),
        egui::Key::Period => Some('.'),
        egui::Key::Slash => Some('/'),
        egui::Key::Backslash => Some('\\'),
        egui::Key::OpenBracket => Some('['),
        egui::Key::CloseBracket => Some(']'),
        egui::Key::Semicolon => Some(';'),
        egui::Key::Quote => Some('\''),
        egui::Key::Backtick => Some('`'),
        _ => None,
    } {
        return Some(format!("\u{1b}{ch}"));
    }
    match key {
        egui::Key::ArrowUp => Some("<M-Up>".to_owned()),
        egui::Key::ArrowDown => Some("<M-Down>".to_owned()),
        egui::Key::ArrowLeft => Some("<M-Left>".to_owned()),
        egui::Key::ArrowRight => Some("<M-Right>".to_owned()),
        _ => None,
    }
}

fn special_sequence(key: &egui::Key, shift: bool) -> Option<String> {
    Some(match key {
        egui::Key::Enter => "\r".to_owned(),
        egui::Key::Escape => "\u{1b}".to_owned(),
        egui::Key::Tab => {
            if shift { "<S-Tab>".to_owned() } else { "\t".to_owned() }
        }
        egui::Key::Backspace => "\u{7f}".to_owned(),
        egui::Key::Delete => "<Del>".to_owned(),
        egui::Key::Insert => "<Insert>".to_owned(),
        egui::Key::Home => "<Home>".to_owned(),
        egui::Key::End => "<End>".to_owned(),
        egui::Key::PageUp => "<PageUp>".to_owned(),
        egui::Key::PageDown => "<PageDown>".to_owned(),
        egui::Key::ArrowUp => {
            if shift { "<S-Up>".to_owned() } else { "<Up>".to_owned() }
        }
        egui::Key::ArrowDown => {
            if shift { "<S-Down>".to_owned() } else { "<Down>".to_owned() }
        }
        egui::Key::ArrowLeft => {
            if shift { "<S-Left>".to_owned() } else { "<Left>".to_owned() }
        }
        egui::Key::ArrowRight => {
            if shift { "<S-Right>".to_owned() } else { "<Right>".to_owned() }
        }
        egui::Key::F1 => "<F1>".to_owned(),
        egui::Key::F2 => "<F2>".to_owned(),
        egui::Key::F3 => "<F3>".to_owned(),
        egui::Key::F4 => "<F4>".to_owned(),
        egui::Key::F5 => "<F5>".to_owned(),
        egui::Key::F6 => "<F6>".to_owned(),
        egui::Key::F7 => "<F7>".to_owned(),
        egui::Key::F8 => "<F8>".to_owned(),
        egui::Key::F9 => "<F9>".to_owned(),
        egui::Key::F10 => "<F10>".to_owned(),
        egui::Key::F11 => "<F11>".to_owned(),
        egui::Key::F12 => "<F12>".to_owned(),
        _ => return None,
    })
}

fn note_proc(state: &Mutex<UiState>, wake: &Condvar, proc: ProcState) {
    let mut s = state.lock().unwrap();
    s.proc = proc;
    drop(s);
    wake.notify_all();
}

fn spawn_bridge(
    root: PathBuf,
    bin: Option<PathBuf>,
    extra_args: Vec<String>,
    state: Arc<Mutex<UiState>>,
    wake: Arc<Condvar>,
    rx: Receiver<NvimCmd>,
) -> (thread::JoinHandle<()>, tokio::runtime::Handle) {
    // The runtime's handle must come from INSIDE the spawned thread (only
    // there does a runtime context exist); `Handle::current()` on the caller
    // panics with "no reactor running".
    let (htx, hrx) = std::sync::mpsc::channel();
    let join = thread::Builder::new()
        .name("nvim-bridge".to_owned())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build nvim bridge runtime");
            let _ = htx.send(rt.handle().clone());
            rt.block_on(bridge_main(root, bin, extra_args, state, wake, rx));
        })
        .expect("spawn nvim bridge thread");
    let handle = hrx.recv().expect("bridge runtime handle");
    (join, handle)
}

async fn bridge_main(
    root: PathBuf,
    bin: Option<PathBuf>,
    extra_args: Vec<String>,
    state: Arc<Mutex<UiState>>,
    wake: Arc<Condvar>,
    mut rx: Receiver<NvimCmd>,
) {
    let mut cmd = tokio::process::Command::new(bin.as_deref().unwrap_or(Path::new("nvim")));
    cmd.arg("--embed").args(&extra_args).current_dir(&root);
    let handler = BridgeHandler { state: state.clone(), wake: wake.clone() };
    let (nvim, _io, mut child) = match new_child_cmd(&mut cmd, handler).await {
        Ok(v) => v,
        Err(e) => {
            note_proc(&state, &wake, ProcState::Failed(format!("failed to start `nvim --embed`: {e}")));
            return;
        }
    };
    let mut opts = UiAttachOptions::new();
    opts.set_rgb(true).set_linegrid_external(true);
    if let Err(e) = nvim.ui_attach(DEFAULT_COLS as i64, DEFAULT_ROWS as i64, &opts).await {
        let _ = child.kill().await;
        note_proc(&state, &wake, ProcState::Failed(format!("nvim_ui_attach failed: {e}")));
        return;
    }
    note_proc(&state, &wake, ProcState::Running);
    let pump = {
        let nvim = nvim.clone();
        let state = state.clone();
        let wake = wake.clone();
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    NvimCmd::Input(keys) => {
                        if nvim.input(&keys).await.is_err() {
                            break;
                        }
                    }
                    NvimCmd::Mouse { button, action, modifier, row, col } => {
                        if nvim.input_mouse(button, action, &modifier, 1, row, col).await.is_err() {
                            break;
                        }
                    }
                    NvimCmd::Resize { width, height } => {
                        let accepted = matches!(
                            nvim.call("nvim_ui_try_resize", call_args![width, height]).await,
                            Ok(Ok(Value::Boolean(true)))
                        );
                        let mut s = state.lock().unwrap();
                        s.resize_retry_after = if accepted { None } else { Some(Instant::now() + RESIZE_RETRY_AFTER) };
                        drop(s);
                        wake.notify_all();
                    }
                }
            }
            let _ = nvim.quit_no_save().await;
        })
    };
    let status = match child.wait().await {
        Ok(status) => status.to_string(),
        Err(e) => format!("wait failed: {e}"),
    };
    pump.abort();
    note_proc(&state, &wake, ProcState::Exited(status));
}

/// The embedded nvim dock tab: owns the child-process bridge and the cached
/// render state.
pub struct NvimEmbed {
    state: Arc<Mutex<UiState>>,
    wake: Arc<Condvar>,
    tx: Option<Sender<NvimCmd>>,
    rt: Option<tokio::runtime::Handle>,
    root: PathBuf,
    bin: Option<PathBuf>,
    extra_args: Vec<String>,
    grid_id: Option<egui::Id>,
    job: Option<egui::text::LayoutJob>,
    job_generation: u64,
    job_size: (u16, u16),
    last_sent_size: Option<(u16, u16)>,
    last_resize_sent: Option<Instant>,
    scroll_rem: f32,
}

impl NvimEmbed {
    /// Create the tab. nvim is spawned lazily the first time the tab is
    /// shown.
    pub fn new(root: PathBuf) -> Self {
        Self::with_bin(root, None)
    }

    fn with_bin(root: PathBuf, bin: Option<PathBuf>) -> Self {
        Self::with_bin_and_args(root, bin, Vec::new())
    }

    fn with_bin_and_args(root: PathBuf, bin: Option<PathBuf>, extra_args: Vec<String>) -> Self {
        let state = Arc::new(Mutex::new(UiState {
            default_fg: egui::Color32::from_gray(230),
            default_bg: egui::Color32::from_gray(18),
            default_sp: egui::Color32::from_gray(180),
            ..Default::default()
        }));
        Self {
            state,
            wake: Arc::new(Condvar::new()),
            tx: None,
            rt: None,
            root,
            bin,
            extra_args,
            grid_id: None,
            job: None,
            job_generation: u64::MAX,
            job_size: (0, 0),
            last_sent_size: None,
            last_resize_sent: None,
            scroll_rem: 0.0,
        }
    }

    /// Stop the child process (used when the tab closes; the next show
    /// restarts it).
    pub fn shutdown(&mut self) {
        self.tx = None;
        let mut s = self.state.lock().unwrap();
        s.proc = ProcState::Starting;
        s.generation += 1;
        self.job = None;
        self.last_sent_size = None;
        self.last_resize_sent = None;
    }

    fn ensure_started(&mut self, root: &Path) {
        if self.tx.is_some() && self.root == root {
            return;
        }
        self.shutdown();
        self.root = root.to_path_buf();
        // Open nvim in the project's COMPONENTS crate by default (the folder
        // the user edits most): a `plugins/<name>` crate root when there is
        // one, else the project root.
        let mut dir = root.to_path_buf();
        if let Ok(rd) = std::fs::read_dir(root.join("plugins"))
            && let Some(crate_dir) = rd.flatten().find(|e| e.path().join("Cargo.toml").is_file())
        {
            dir = crate_dir.path();
        }
        let (tx, rx) = channel(128);
        let (_join, rt) = spawn_bridge(
            dir,
            self.bin.clone(),
            self.extra_args.clone(),
            self.state.clone(),
            self.wake.clone(),
            rx,
        );
        self.rt = Some(rt);
        self.tx = Some(tx);
    }

    fn send_cmd(&self, cmd: NvimCmd) {
        // The UI thread has no tokio runtime context: `blocking_send` would
        // abort the process (SIGABRT on the editor's crash handler). Drive
        // the send through the bridge runtime's handle instead.
        if let (Some(tx), Some(rt)) = (&self.tx, &self.rt) {
            let _ = rt.block_on(tx.send(cmd));
        }
    }

    /// Draw the tab: grid rendering, input forwarding, and resize driving.
    pub fn show(&mut self, ui: &mut egui::Ui, root: &Path) {
        self.ensure_started(root);
        let (proc, mode, size) = {
            let s = self.state.lock().unwrap();
            (s.proc.clone(), s.mode.clone(), (s.width, s.height))
        };
        match &proc {
            ProcState::Failed(msg) => {
                ui.label(msg);
                if ui.button("Retry").clicked() {
                    self.shutdown();
                }
                return;
            }
            ProcState::Exited(status) => {
                ui.label(format!("Neovim exited ({status})."));
                if ui.button("Restart").clicked() {
                    self.shutdown();
                }
                return;
            }
            ProcState::Starting => {
                ui.spinner();
                ui.label("starting nvim\u{2026}");
                return;
            }
            ProcState::Running => {}
        }
        ui.horizontal(|ui| {
            let mode = if mode.is_empty() { "nvim" } else { mode.as_str() };
            ui.label(egui::RichText::new(mode).monospace().color(crate::theme::TEXT_MUTED));
            ui.label(
                egui::RichText::new(format!("{}\u{d7}{}", size.0, size.1))
                    .monospace()
                    .color(crate::theme::TEXT_DIM),
            );
        });
        let font_id = egui::FontId::monospace(FONT_SIZE);
        let (cell_w, row_h) = ui.fonts_mut(|f| (f.glyph_width(&font_id, 'M'), f.row_height(&font_id)));
        let avail = ui.available_size();
        let cols = ((avail.x / cell_w.max(1.0)).floor() as u16).clamp(MIN_COLS, MAX_COLS);
        let rows = ((avail.y / row_h.max(1.0)).floor() as u16).clamp(MIN_ROWS, MAX_ROWS);
        let rect = egui::Rect::from_min_size(
            ui.min_rect().min,
            egui::vec2(cols as f32 * cell_w, rows as f32 * row_h),
        );
        let grid_id = *self.grid_id.get_or_insert_with(|| ui.make_persistent_id("nvim-grid"));
        let resp = ui.interact(rect, grid_id, egui::Sense::click_and_drag());
        // Glue the focus to the tab: while the pointer is over it, it claims
        // (or re-claims) focus, so the arrow keys and friends stay inside
        // nvim instead of walking egui's focus between dock widgets.
        if resp.clicked() || resp.hovered() {
            resp.request_focus();
        }
        self.maybe_resize(cols, rows);
        let galley = self.galley(ui, &font_id);
        let default_bg = self.state.lock().unwrap().default_bg;
        let painter = ui.painter();
        painter.rect_filled(rect, egui::CornerRadius::ZERO, default_bg);
        painter.galley(rect.min, galley, egui::Color32::PLACEHOLDER);
        self.paint_cursor(painter, rect, cell_w, row_h, &font_id);
        if resp.has_focus() || resp.hovered() {
            self.forward_keys(ui);
        }
        self.forward_pointer(ui, &resp, rect, cell_w, row_h);
    }

    fn maybe_resize(&mut self, cols: u16, rows: u16) {
        let now = Instant::now();
        let cooldown_ok = self.last_resize_sent.is_none_or(|t| now.duration_since(t) >= RESIZE_COOLDOWN);
        let retry_due = self
            .state
            .lock()
            .unwrap()
            .resize_retry_after
            .is_some_and(|t| now >= t);
        if !cooldown_ok && !retry_due {
            return;
        }
        if self.last_sent_size == Some((cols, rows)) && !retry_due {
            return;
        }
        self.last_sent_size = Some((cols, rows));
        self.last_resize_sent = Some(now);
        self.send_cmd(NvimCmd::Resize { width: cols as i64, height: rows as i64 });
    }

    fn galley(&mut self, ui: &mut egui::Ui, font_id: &egui::FontId) -> Arc<egui::Galley> {
        let (generation, w, h) = {
            let s = self.state.lock().unwrap();
            (s.generation, s.width, s.height)
        };
        if self.job.is_none() || generation != self.job_generation || (w, h) != self.job_size {
            self.job = Some(build_job(&self.state.lock().unwrap(), font_id));
            self.job_generation = generation;
            self.job_size = (w, h);
        }
        ui.fonts_mut(|f| f.layout_job(self.job.as_ref().expect("job built above").clone()))
    }

    fn paint_cursor(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        cell_w: f32,
        row_h: f32,
        font_id: &egui::FontId,
    ) {
        let s = self.state.lock().unwrap();
        if s.cursor_row >= s.height || s.cursor_col >= s.width {
            return;
        }
        let cell = &s.cells[s.cursor_row as usize * s.width as usize + s.cursor_col as usize];
        let fmt = format_for(&s, cell.hl, font_id);
        let cell_rect = egui::Rect::from_min_size(
            rect.min + egui::vec2(s.cursor_col as f32 * cell_w, s.cursor_row as f32 * row_h),
            egui::vec2(cell_w, row_h),
        );
        match s.cursor_shapes.get(&s.mode_idx).copied().unwrap_or_default() {
            CursorShape::Block => {
                painter.rect_filled(cell_rect, egui::CornerRadius::ZERO, fmt.color);
                let text = if cell.text.is_empty() { " " } else { cell.text.as_str() };
                painter.text(cell_rect.min, egui::Align2::LEFT_TOP, text, font_id.clone(), fmt.background);
            }
            CursorShape::Horizontal => {
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(cell_rect.left(), cell_rect.bottom() - 2.0),
                        egui::vec2(cell_w, 2.0),
                    ),
                    egui::CornerRadius::ZERO,
                    fmt.color,
                );
            }
            CursorShape::Vertical => {
                painter.rect_filled(
                    egui::Rect::from_min_size(cell_rect.min, egui::vec2(2.0, row_h)),
                    egui::CornerRadius::ZERO,
                    fmt.color,
                );
            }
        }
    }

    fn forward_keys(&self, ui: &mut egui::Ui) {
        let events: Vec<egui::Event> = ui.input(|i| i.events.clone());
        let mut handled = false;
        for event in &events {
            match event {
                egui::Event::Copy => {
                    self.send_cmd(NvimCmd::Input("\u{3}".to_owned()));
                    handled = true;
                }
                egui::Event::Cut => {
                    self.send_cmd(NvimCmd::Input("\u{18}".to_owned()));
                    handled = true;
                }
                egui::Event::Paste(text) => {
                    self.send_cmd(NvimCmd::Input(escape_input(text)));
                    handled = true;
                }
                egui::Event::Key { key, pressed: true, modifiers, .. } => {
                    let seq = if modifiers.ctrl || modifiers.command || modifiers.mac_cmd {
                        ctrl_sequence(key, modifiers)
                    } else if modifiers.alt {
                        alt_sequence(key)
                    } else {
                        special_sequence(key, modifiers.shift)
                    };
                    if let Some(seq) = seq {
                        self.send_cmd(NvimCmd::Input(seq));
                        ui.input_mut(|i| {
                            i.consume_key(*modifiers, *key);
                        });
                        handled = true;
                    }
                }
                egui::Event::Text(text) if !ui.input(|i| i.modifiers.alt) => {
                    self.send_cmd(NvimCmd::Input(escape_input(text)));
                    handled = true;
                }
                _ => {}
            }
        }
        if handled {
            ui.input_mut(|i| {
                i.events.retain(|e| matches!(e, egui::Event::Text(_) | egui::Event::Copy | egui::Event::Cut | egui::Event::Paste(_)));
            });
        }
    }

    fn forward_pointer(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: egui::Rect, cell_w: f32, row_h: f32) {
        let (w, h) = {
            let s = self.state.lock().unwrap();
            (s.width as i64, s.height as i64)
        };
        let modifier = {
            let m = ui.input(|i| i.modifiers);
            let mut parts = Vec::new();
            if m.ctrl {
                parts.push("c");
            }
            if m.shift {
                parts.push("s");
            }
            if m.alt {
                parts.push("a");
            }
            parts.join("")
        };
        let cell_of = |pos: egui::Pos2| {
            let rel = pos - rect.min;
            let row = (rel.y / row_h.max(1.0)).floor().clamp(0.0, (h - 1).max(0) as f32) as i64;
            let col = (rel.x / cell_w.max(1.0)).floor().clamp(0.0, (w - 1).max(0) as f32) as i64;
            (row, col)
        };
        let tx = self.tx.clone();
        let rt = self.rt.clone();
        let mouse = move |button: &'static str, action: &'static str, pos: egui::Pos2| {
            let (row, col) = cell_of(pos);
            if let (Some(tx), Some(rt)) = (&tx, &rt) {
                let _ = rt.block_on(tx.send(NvimCmd::Mouse { button, action, modifier: modifier.clone(), row, col }));
            }
        };
        if let Some(pos) = resp.interact_pointer_pos() {
            if resp.clicked_by(egui::PointerButton::Primary) {
                mouse("left", "press", pos);
                mouse("left", "release", pos);
            }
            if resp.clicked_by(egui::PointerButton::Secondary) {
                mouse("right", "press", pos);
                mouse("right", "release", pos);
            }
            if resp.drag_started_by(egui::PointerButton::Primary) {
                mouse("left", "press", pos);
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                mouse("left", "drag", pos);
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) {
                mouse("left", "release", pos);
            }
        }
        if resp.hovered() {
            self.scroll_rem += ui.input(|i| i.raw_scroll_delta.y);
            while self.scroll_rem >= WHEEL_STEP {
                self.scroll_rem -= WHEEL_STEP;
                if let Some(pos) = resp.hover_pos() {
                    mouse("wheel", "down", pos);
                }
            }
            while self.scroll_rem <= -WHEEL_STEP {
                self.scroll_rem += WHEEL_STEP;
                if let Some(pos) = resp.hover_pos() {
                    mouse("wheel", "up", pos);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_line_places_cells_and_repeats() {
        let mut s = UiState::default();
        grid_resize(&mut s, &[&Value::from(1u64), &Value::from(4u64), &Value::from(2u64)]);
        let cells = Value::Array(vec![
            Value::Array(vec![Value::from("ab"), Value::from(7u64), Value::from(2u64)]),
            Value::Array(vec![Value::from("x"), Value::from(0u64), Value::from(1u64)]),
        ]);
        grid_line(&mut s, &[&Value::from(1u64), &Value::from(1u64), &Value::from(0u64), &cells, &Value::from(true)]);
        let cell = |r: usize, c: usize| &s.cells[r * s.width as usize + c];
        assert_eq!(s.width, 4);
        assert_eq!(s.height, 2);
        assert_eq!(cell(1, 0).text, "ab");
        assert_eq!(cell(1, 0).hl, 7);
        assert_eq!(cell(1, 1).text, "ab");
        assert_eq!(cell(1, 2).text, "x");
        assert_eq!(cell(1, 3).text, "");
    }

    #[test]
    fn grid_line_without_wrap_fills_the_whole_row() {
        let mut s = UiState::default();
        grid_resize(&mut s, &[&Value::from(1u64), &Value::from(4u64), &Value::from(2u64)]);
        let cells = Value::Array(
            ["N", "V", "I", "M", "X"]
                .iter()
                .map(|c| Value::Array(vec![Value::from(*c), Value::from(0u64), Value::from(1u64)]))
                .collect(),
        );
        grid_line(&mut s, &[&Value::from(1u64), &Value::from(0u64), &Value::from(0u64), &cells, &Value::from(false)]);
        let row: String = s.cells[..4].iter().map(|c| c.text.as_str()).collect();
        assert_eq!(row, "NVIM");
    }

    #[test]
    fn grid_line_wrapped_spills_to_next_row() {
        let mut s = UiState::default();
        grid_resize(&mut s, &[&Value::from(1u64), &Value::from(4u64), &Value::from(2u64)]);
        let cells = Value::Array(vec![Value::Array(vec![Value::from("a"), Value::from(3u64), Value::from(8u64)])]);
        grid_line(&mut s, &[&Value::from(1u64), &Value::from(0u64), &Value::from(0u64), &cells, &Value::from(true)]);
        let row = |r: usize| s.cells[r * 4..(r + 1) * 4].iter().map(|c| c.text.as_str()).collect::<String>();
        assert_eq!(row(0), "aaaa");
        assert_eq!(row(1), "aaaa");
    }

    #[test]
    fn grid_scroll_shifts_and_blanks_rows() {
        let mut s = UiState::default();
        grid_resize(&mut s, &[&Value::from(1u64), &Value::from(5u64), &Value::from(3u64)]);
        let mut line = |row: u64, text: &str| {
            let cells = Value::Array(vec![Value::Array(vec![Value::from(text), Value::from(0u64), Value::from(1u64)])]);
            grid_line(&mut s, &[&Value::from(1u64), &Value::from(row), &Value::from(0u64), &cells, &Value::from(true)]);
        };
        line(0, "a");
        line(1, "b");
        line(2, "c");
        grid_scroll(&mut s, &[&Value::from(1u64), &Value::from(0u64), &Value::from(2u64), &Value::from(0u64), &Value::from(0u64), &Value::from(1i64), &Value::from(0u64)]);
        let row = |r: usize| s.cells[r * s.width as usize].text.clone();
        assert_eq!(row(0), "b");
        assert_eq!(row(1), "c");
        assert_eq!(row(2), "");
    }

    #[test]
    fn hl_attrs_and_default_colors_resolve() {
        let mut s = UiState {
            default_fg: egui::Color32::WHITE,
            default_bg: egui::Color32::BLACK,
            ..Default::default()
        };
        let attr = Value::Map(vec![
            (Value::from(0u64), Value::from(0x00ff00u64)),
            (Value::from(1u64), Value::from(0x0000ffu64)),
            (Value::from(3u64), Value::from(true)),
        ]);
        hl_attr_define(&mut s, &[&Value::from(5u64), &attr]);
        default_colors_set(&mut s, &[&Value::from(0x010203u64), &Value::from(0x0a0b0cu64), &Value::from(0u64)]);
        assert_eq!(s.default_fg, egui::Color32::from_rgb(1, 2, 3));
        assert_eq!(s.default_bg, egui::Color32::from_rgb(10, 11, 12));
        let fmt = format_for(&s, 5, &egui::FontId::monospace(16.0));
        assert_eq!(fmt.color, egui::Color32::from_rgb(0, 0, 255));
        assert_eq!(fmt.background, egui::Color32::from_rgb(0, 255, 0));
    }

    #[test]
    fn mode_info_sets_cursor_shape() {
        let mut s = UiState::default();
        let info = Value::Array(vec![Value::Array(vec![
            Value::from(2u64),
            Value::Map(vec![(Value::from("cursor_shape"), Value::from("vertical"))]),
        ])]);
        mode_info_set(&mut s, &[&Value::from(true), &info]);
        mode_change(&mut s, &[&Value::from("insert"), &Value::from(2u64)]);
        assert_eq!(s.cursor_shapes.get(&2), Some(&CursorShape::Vertical));
        assert_eq!(s.mode, "insert");
    }

    #[test]
    fn key_sequences_map_to_nvim_input() {
        let ctrl = egui::Modifiers::CTRL;
        assert_eq!(ctrl_sequence(&egui::Key::Z, &ctrl), Some("\u{1a}".to_owned()));
        assert_eq!(ctrl_sequence(&egui::Key::Space, &ctrl), Some("\u{0}".to_owned()));
        assert_eq!(ctrl_sequence(&egui::Key::OpenBracket, &ctrl), Some("\u{1b}".to_owned()));
        assert_eq!(alt_sequence(&egui::Key::B), Some("\u{1b}b".to_owned()));
        assert_eq!(alt_sequence(&egui::Key::ArrowUp), Some("<M-Up>".to_owned()));
        assert_eq!(special_sequence(&egui::Key::Enter, false), Some("\r".to_owned()));
        assert_eq!(special_sequence(&egui::Key::Escape, false), Some("\u{1b}".to_owned()));
        assert_eq!(special_sequence(&egui::Key::ArrowUp, false), Some("<Up>".to_owned()));
        assert_eq!(special_sequence(&egui::Key::Tab, true), Some("<S-Tab>".to_owned()));
        assert_eq!(special_sequence(&egui::Key::F5, false), Some("<F5>".to_owned()));
        assert_eq!(special_sequence(&egui::Key::A, false), None);
        assert_eq!(ctrl_sequence(&egui::Key::Enter, &ctrl), None);
    }

    #[test]
    fn escape_input_quotes_angle_brackets() {
        assert_eq!(escape_input("<C-x>"), "<lt>C-x>");
        assert_eq!(escape_input("plain"), "plain");
    }

    #[test]
    fn multi_tuple_events_apply_every_line() {
        let mut s = UiState::default();
        grid_resize(&mut s, &[&Value::from(1u64), &Value::from(4u64), &Value::from(4u64)]);
        let line = |row: u64, text: &str| {
            Value::Array(vec![
                Value::from(1u64),
                Value::from(row),
                Value::from(0u64),
                Value::Array(vec![Value::Array(vec![Value::from(text), Value::from(0u64), Value::from(1u64)])]),
                Value::from(false),
            ])
        };
        let event = Value::Array(vec![Value::from("grid_line"), line(0, "a"), line(1, "b"), line(2, "c")]);
        let events = event.as_array().unwrap();
        for tuple in &events[1..] {
            let params: Vec<&Value> = tuple.as_array().unwrap().iter().collect();
            grid_line(&mut s, &params);
        }
        let row = |r: usize| s.cells[r * 4].text.clone();
        assert_eq!(row(0), "a");
        assert_eq!(row(1), "b");
        assert_eq!(row(2), "c");
    }

    fn wait_until(state: &Mutex<UiState>, wake: &Condvar, deadline: Instant, pred: impl Fn(&UiState) -> bool) -> bool {
        let mut s = state.lock().unwrap();
        loop {
            if pred(&s) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, _) = wake.wait_timeout(s, deadline - now).unwrap();
            s = guard;
        }
    }

    #[test]
    #[ignore] // needs a real nvim: NVIM_BIN=/path/to/nvim NVIM_SMOKE_ARGS=--clean cargo test --lib nvim_embed -- --ignored --nocapture
    fn real_nvim_bridge_roundtrip() {
        let bin = std::env::var_os("NVIM_BIN").map(PathBuf::from);
        let root = std::env::temp_dir();
        let extra_args = std::env::var("NVIM_SMOKE_ARGS").map(|s| s.split_whitespace().map(str::to_owned).collect()).unwrap_or_default();
        let mut embed = NvimEmbed::with_bin_and_args(root.clone(), bin, extra_args);
        embed.ensure_started(&root);
        let deadline = Instant::now() + Duration::from_secs(15);
        let ready = wait_until(&embed.state, &embed.wake, deadline, |s| {
            matches!(s.proc, ProcState::Failed(_))
                || (matches!(s.proc, ProcState::Running)
                    && s.generation > 0
                    && s.cells.iter().any(|c| !c.text.trim().is_empty()))
        });
        let proc = embed.state.lock().unwrap().proc.clone();
        let (w, h, generation) = {
            let s = embed.state.lock().unwrap();
            (s.width, s.height, s.generation)
        };
        assert!(ready && matches!(proc, ProcState::Running), "nvim did not start: {proc:?} grid={w}x{h} flushes={generation}");
        embed.send_cmd(NvimCmd::Input("iXYZ\u{1b}".to_owned()));
        let deadline = Instant::now() + Duration::from_secs(10);
        let typed = wait_until(&embed.state, &embed.wake, deadline, |s| {
            s.cells.iter().take(8).any(|c| matches!(c.text.as_str(), "X" | "Y" | "Z"))
        });
        assert!(typed, "typed input never reached nvim");
        embed.send_cmd(NvimCmd::Input(":q!\r".to_owned()));
        let deadline = Instant::now() + Duration::from_secs(15);
        let exited = wait_until(&embed.state, &embed.wake, deadline, |s| matches!(s.proc, ProcState::Exited(_)));
        assert!(exited, "nvim did not exit after :q!");
    }
}
