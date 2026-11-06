//! CPU-drawn splash window shown before the Vulkan renderer exists. Uses
//! softbuffer to present a scaled-down background (a static image, or an animated
//! WebP/GIF/APNG played by elapsed time) with a status line along the bottom, so
//! the user sees branding while the editor loads. The bottom band carries the
//! load phase, the editor version + project name and a step progress bar.
//! Editor only: a shipped game never shows it.
//!
//! Animation only visibly plays while the main thread is free to tick it; the
//! renderer build is backgrounded (see editor_app) so it animates during that.
//! The status line still updates across the later (main-thread) load phases.

use std::io::Cursor;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use anyhow::{Context as _, Result};
use image::AnimationDecoder;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId, WindowLevel};

/// Plex Sans Medium, the editor's UI face (docs/design).
const SPLASH_FONT: &[u8] = include_bytes!("../../zestengine-editor/assets/fonts/IBMPlexSans-Medium.ttf");
/// Embedded animated splash (looped). Default branding; an external
/// `zestengine.webp`/`splash.gif` in the project root still overrides it.
const SPLASH_WEBP: &[u8] = include_bytes!("../../zestengine-editor/assets/zestengine.webp");

/// Splash window size (source art is 16:9, scaled down to this).
const W: u32 = 960;
const H: u32 = 540;

/// One pre-scaled animation frame: W*H pixels (0x00RRGGBB) + cumulative end time
/// (ms) within the loop, so `tick` can pick the current frame by elapsed time.
struct Frame {
    px: Vec<u32>,
    end_ms: u32,
}

pub struct Splash {
    window: Arc<Window>,
    // Context must outlive the surface; kept alive here.
    _context: softbuffer::Context<Arc<Window>>,
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    frames: Vec<Frame>,
    loop_ms: u32,
    start: Instant,
    font: FontRef<'static>,
    status: String,
    /// Right side of the band: "ZestEngine Editor 0.1.0" and the project name.
    caption: String,
    /// Completed load steps out of the total; the bar eases towards it.
    progress: f32,
    shown_progress: f32,
    accent: u32,
    /// Animation frames streamed in from a worker thread as they decode (so the
    /// animation starts almost immediately instead of after a multi-second full
    /// decode). The placeholder shows until the first frame arrives.
    frames_rx: Option<mpsc::Receiver<Frame>>,
    /// True once the first streamed frame replaced the placeholder.
    streaming: bool,
}

impl Splash {
    /// Create the splash window. Loads an animated `zestengine.webp`/`splash.gif`
    /// from `<asset_dir>` if present (so it can be swapped without recompiling),
    /// otherwise the embedded static PNG.
    pub fn new(event_loop: &ActiveEventLoop, asset_dir: &Path) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("zestengine")
            .with_decorations(false)
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_inner_size(winit::dpi::LogicalSize::new(W, H));
        let window = Arc::new(event_loop.create_window(attrs)?);

        let context = softbuffer::Context::new(window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer context: {e}"))?;
        let mut surface = softbuffer::Surface::new(&context, window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer surface: {e}"))?;
        surface
            .resize(NonZeroU32::new(W).unwrap(), NonZeroU32::new(H).unwrap())
            .map_err(|e| anyhow::anyhow!("softbuffer resize: {e}"))?;

        // Placeholder = the webp's first frame (cheap to decode), so when the
        // streamed animation starts it continues from it with no visible jump.
        let frames = vec![first_placeholder(asset_dir)];
        let (tx, rx) = mpsc::channel();
        let dir = asset_dir.to_path_buf();
        std::thread::spawn(move || stream_frames(&dir, tx));

        let font = FontRef::try_from_slice(SPLASH_FONT).context("loading splash font")?;

        Ok(Self {
            window,
            _context: context,
            surface,
            frames,
            loop_ms: 0,
            start: Instant::now(),
            font,
            status: String::new(),
            caption: format!("ZestEngine Editor {}", env!("CARGO_PKG_VERSION")),
            progress: 0.0,
            shown_progress: 0.0,
            accent: 0xF5C84B,
            frames_rx: Some(rx),
            streaming: false,
        })
    }

    pub fn window_id(&self) -> WindowId {
        self.window.id()
    }

    /// Index of the frame to show right now.
    ///
    /// While frames are still streaming in we play strictly FORWARD and hold on
    /// the newest decoded frame — never wrapping, since `loop_ms` only covers the
    /// decoded-so-far frames and wrapping over that growing window jumps the
    /// playback backwards every tick (the stutter). Only once decoding finishes
    /// do we loop over the full clip.
    fn current_frame(&self) -> usize {
        let n = self.frames.len();
        if n <= 1 {
            return 0;
        }
        let elapsed = self.start.elapsed().as_millis() as u32;
        let done = self.frames_rx.is_none();
        let t = if done && self.loop_ms > 0 {
            elapsed % self.loop_ms
        } else {
            elapsed
        };
        self.frames
            .iter()
            .position(|f| t < f.end_ms)
            .unwrap_or(n - 1)
    }

    /// Name the project being opened (shown beside the version) and use its
    /// accent colour for the progress bar.
    pub fn set_project(&mut self, name: &str, accent: Option<[u8; 3]>) {
        self.caption = format!("{name}   ZestEngine Editor {}", env!("CARGO_PKG_VERSION"));
        if let Some([r, g, b]) = accent {
            self.accent = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
        }
    }

    /// Step `done` of `total` load steps has started; drives the progress bar.
    pub fn set_progress(&mut self, done: usize, total: usize) {
        self.progress = (done as f32 / total.max(1) as f32).clamp(0.0, 1.0);
    }

    /// Update the bottom status line and redraw. Returns whether it painted.
    pub fn set_status(&mut self, status: &str) -> bool {
        self.status = status.to_string();
        self.present()
    }

    /// Composite the current frame + a darkened bottom band + the status text,
    /// and present. Returns whether a frame was actually presented — false when
    /// the surface isn't configured yet (the caller retries next tick).
    pub fn present(&mut self) -> bool {
        // Drain any frames the decode worker has produced since last present.
        if let Some(rx) = self.frames_rx.as_ref() {
            let mut disconnected = false;
            loop {
                match rx.try_recv() {
                    Ok(frame) => {
                        if !self.streaming {
                            // First real frame: drop the placeholder, restart clock.
                            self.streaming = true;
                            self.frames.clear();
                            self.start = Instant::now();
                        }
                        self.loop_ms = frame.end_ms;
                        self.frames.push(frame);
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            if disconnected {
                self.frames_rx = None;
            }
        }
        let idx = self.current_frame();
        let Some(frame) = self.frames.get(idx) else {
            return false;
        };
        let Ok(mut buf) = self.surface.buffer_mut() else {
            return false;
        };
        buf.copy_from_slice(&frame.px);

        // Darkened bottom band (fades in) so the text reads over any background.
        let band = 64u32;
        for y in (H - band)..H {
            let t = (y - (H - band)) as f32 / band as f32;
            let d = (t * 0.72).clamp(0.0, 0.72);
            for x in 0..W {
                let i = (y * W + x) as usize;
                buf[i] = darken(buf[i], d);
            }
        }

        let text = if self.status.is_empty() {
            "Loading…"
        } else {
            self.status.as_str()
        };
        draw_text(&mut buf, &self.font, 18.0, 24, (H - 22) as i32, text, 0xF2F3F5);
        let caption_w = text_width(&self.font, 13.0, &self.caption);
        draw_text(&mut buf, &self.font, 13.0, W as i32 - 24 - caption_w as i32, (H - 22) as i32, &self.caption, 0xA3A6AF);

        // Step progress along the bottom edge: a dim track and the accent fill,
        // easing forward so a step that lands at once still reads as motion.
        self.shown_progress += (self.progress - self.shown_progress) * 0.18;
        let fill = (self.shown_progress * W as f32) as u32;
        for y in (H - 3)..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                buf[i] = if x < fill { self.accent } else { blend(buf[i], 0x26282D, 0.8) };
            }
        }

        buf.present().is_ok()
    }
}

/// Decode the splash frames: an external animated file if present, else the
/// embedded static PNG (a single frame). Always returns at least one frame.
/// Stream splash animation frames to `tx` as they decode. Prefers system ffmpeg
/// (SIMD-fast; handles mp4/webp/gif and pre-scales for us) so the animation
/// starts in ~0.1s; falls back to the pure-Rust `image` webp decoder when ffmpeg
/// is unavailable or produces nothing.
fn stream_frames(asset_dir: &Path, tx: mpsc::Sender<Frame>) {
    // mp4 needs ffmpeg (the only h264 decoder available); webp/gif decode with
    // the pure-Rust `image` path (ffmpeg can't read these animated webps, and at
    // splash resolution the Rust decoder is fast enough — streamed, so it shows
    // immediately). External overrides in the project root win over the embed.
    let mp4 = asset_dir.join("splash.mp4");
    if mp4.exists() && ffmpeg_stream(&mp4, &tx).is_ok() {
        return;
    }
    for name in ["zestengine.webp", "splash.gif"] {
        let p = asset_dir.join(name);
        if let Ok(bytes) = std::fs::read(&p)
            && pure_rust_bytes(&bytes, name, &tx) {
                return;
            }
    }
    pure_rust_bytes(SPLASH_WEBP, "zestengine.webp", &tx);
}

/// Pipe pre-scaled frames out of ffmpeg. Returns Err if ffmpeg can't run or
/// yields no frames (so the caller can fall back).
fn ffmpeg_stream(path: &Path, tx: &mpsc::Sender<Frame>) -> std::io::Result<()> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let fps = 25u32;
    let mut child = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("scale={W}:{H}:flags=bilinear"),
            "-r",
            &fps.to_string(),
            "-f",
            "rawvideo",
            // bytes B,G,R,0 → little-endian u32 = 0x00RRGGBB (softbuffer format).
            "-pix_fmt",
            "bgr0",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut out = child.stdout.take().unwrap();
    let frame_bytes = (W * H * 4) as usize;
    let mut buf = vec![0u8; frame_bytes];
    let delay = 1000 / fps;
    let mut cum = 0u32;
    let mut count = 0u32;
    loop {
        match out.read_exact(&mut buf) {
            Ok(()) => {
                let px: Vec<u32> = buf
                    .as_chunks::<4>().0.iter()
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], 0]))
                    .collect();
                cum += delay;
                count += 1;
                if tx.send(Frame { px, end_ms: cum }).is_err() {
                    break; // splash closed
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                let _ = child.kill();
                return Err(e);
            }
        }
    }
    let _ = child.wait();
    if count == 0 {
        return Err(std::io::Error::other("ffmpeg produced no frames"));
    }
    Ok(())
}

/// Pure-Rust streaming decode of a webp/gif animation; scales each frame to the
/// splash size and drops the raw buffer. Returns true if it produced any frame.
fn pure_rust_bytes(bytes: &[u8], name: &str, tx: &mpsc::Sender<Frame>) -> bool {
    let frames: image::Frames = if name.ends_with(".gif") {
        match image::codecs::gif::GifDecoder::new(Cursor::new(bytes)) {
            Ok(d) => d.into_frames(),
            Err(e) => {
                tracing::warn!("splash: gif decode: {e:#}");
                return false;
            }
        }
    } else {
        match image::codecs::webp::WebPDecoder::new(Cursor::new(bytes)) {
            Ok(d) => d.into_frames(),
            Err(e) => {
                tracing::warn!("splash: webp decode: {e:#}");
                return false;
            }
        }
    };
    let mut cum = 0u32;
    let mut any = false;
    for frame in frames {
        let Ok(frame) = frame else { break };
        let (num, den) = frame.delay().numer_denom_ms();
        let delay = num.checked_div(den).map_or(40, |d| d.max(16));
        cum += delay;
        let px = scale_to_bg(&frame.into_buffer());
        any = true;
        if tx.send(Frame { px, end_ms: cum }).is_err() {
            break;
        }
    }
    any
}

/// A solid-colour frame (0x00RRGGBB) — the last-resort placeholder.
fn solid_frame(rgb: u32) -> Frame {
    Frame {
        px: vec![rgb & 0x00FF_FFFF; (W * H) as usize],
        end_ms: 0,
    }
}

/// Placeholder shown instantly: the webp's FIRST frame (decoding one frame is
/// cheap — ~15ms at 540p), so the animation continues from it with no jump.
/// Falls back to a plain dark frame if the first frame can't be decoded.
fn first_placeholder(asset_dir: &Path) -> Frame {
    let external = std::fs::read(asset_dir.join("zestengine.webp")).ok();
    for bytes in [external.as_deref(), Some(SPLASH_WEBP)].into_iter().flatten() {
        if let Some(f) = first_webp_frame(bytes) {
            return f;
        }
    }
    solid_frame(0x0A0A12)
}

fn first_webp_frame(bytes: &[u8]) -> Option<Frame> {
    let decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes)).ok()?;
    let frame = decoder.into_frames().next()?.ok()?;
    Some(Frame {
        px: scale_to_bg(&frame.into_buffer()),
        end_ms: 0,
    })
}

/// Resize an RGBA frame to the window size and pack to 0x00RRGGBB.
fn scale_to_bg(rgba: &image::RgbaImage) -> Vec<u32> {
    let scaled = image::imageops::resize(rgba, W, H, image::imageops::FilterType::Triangle);
    scaled
        .pixels()
        .map(|p| {
            let [r, g, b, _] = p.0;
            ((r as u32) << 16) | ((g as u32) << 8) | b as u32
        })
        .collect()
}

fn darken(p: u32, f: f32) -> u32 {
    let r = ((p >> 16) & 0xFF) as f32 * (1.0 - f);
    let g = ((p >> 8) & 0xFF) as f32 * (1.0 - f);
    let b = (p & 0xFF) as f32 * (1.0 - f);
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn blend(dst: u32, color: u32, cov: f32) -> u32 {
    let chan = |shift: u32| {
        let cc = ((color >> shift) & 0xFF) as f32;
        let dd = ((dst >> shift) & 0xFF) as f32;
        (dd + (cc - dd) * cov) as u32
    };
    (chan(16) << 16) | (chan(8) << 8) | chan(0)
}

/// Advance width of `text` at `px`.
fn text_width(font: &FontRef, px: f32, text: &str) -> f32 {
    let sf = font.as_scaled(PxScale::from(px));
    text.chars().map(|c| sf.h_advance(sf.glyph_id(c))).sum()
}

/// Rasterize a left-aligned string (baseline at `baseline`) blended over the
/// buffer with anti-aliased coverage from ab_glyph.
fn draw_text(
    buf: &mut [u32],
    font: &FontRef,
    px: f32,
    x0: i32,
    baseline: i32,
    text: &str,
    color: u32,
) {
    let sf = font.as_scaled(PxScale::from(px));
    let mut caret_x = x0 as f32;
    for ch in text.chars() {
        let mut glyph = sf.scaled_glyph(ch);
        glyph.position = ab_glyph::point(caret_x, baseline as f32);
        caret_x += sf.h_advance(glyph.id);
        if let Some(outline) = font.outline_glyph(glyph) {
            let bb = outline.px_bounds();
            outline.draw(|gx, gy, cov| {
                let x = bb.min.x as i32 + gx as i32;
                let y = bb.min.y as i32 + gy as i32;
                if x >= 0 && x < W as i32 && y >= 0 && y < H as i32 {
                    let i = (y as u32 * W + x as u32) as usize;
                    buf[i] = blend(buf[i], color, cov.clamp(0.0, 1.0));
                }
            });
        }
    }
}
