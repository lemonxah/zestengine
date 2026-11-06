//! Headless egui screenshot harness (no GPU). Runs a real egui frame, then
//! rasterises the tessellated meshes (textures, clip rects, any
//! pixels_per_point) with a small software triangle filler.
//!
//! ```ignore
//! let mut shot = Shot::new(480.0, 600.0, 2.0);
//! let img = shot.run(|ctx| { egui::CentralPanel::default().show(ctx, |ui| { ... }); });
//! shot.save_named(&img, "blendshapes");   // only when ZESTENGINE_SHOT_DIR is set
//! ```
//! Fonts and theme are installed like the editor does at startup. Used by
//! the panel tests (`tests/panel_shots.rs` and the engine crate's) to look at
//! panels without a display; results are approximations of the wgpu path
//! (no MSAA, simple bilinear sampling).

use std::collections::HashMap;
use std::path::PathBuf;

use egui::epaint::{ImageDelta, Primitive};
use egui::{Event, Pos2, RawInput, Rect, TextureId, ViewportId, pos2, vec2};

/// An RGBA8 image.
#[derive(Clone)]
pub struct Shot8 {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[u8; 4]>,
}

impl Shot8 {
    pub fn save(&self, path: &std::path::Path) {
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let mut img = image::RgbaImage::new(self.w as u32, self.h as u32);
        for (i, p) in self.px.iter().enumerate() {
            img.put_pixel((i % self.w) as u32, (i / self.w) as u32, image::Rgba(*p));
        }
        img.save(path).expect("write png");
    }

    pub fn load(path: &std::path::Path) -> Option<Shot8> {
        let img = image::open(path).ok()?.to_rgba8();
        Some(Shot8 { w: img.width() as usize, h: img.height() as usize, px: img.pixels().map(|p| p.0).collect() })
    }

    /// Box-filter down by an integer factor.
    pub fn downscale(&self, n: usize) -> Shot8 {
        let (w, h) = (self.w / n, self.h / n);
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0u32; 4];
                for dy in 0..n {
                    for dx in 0..n {
                        let p = self.px[(y * n + dy) * self.w + x * n + dx];
                        for i in 0..4 {
                            acc[i] += p[i] as u32;
                        }
                    }
                }
                let d = (n * n) as u32;
                px.push([(acc[0] / d) as u8, (acc[1] / d) as u8, (acc[2] / d) as u8, (acc[3] / d) as u8]);
            }
        }
        Shot8 { w, h, px }
    }

    /// Number of pixels that differ from the top-left one (blank check).
    pub fn non_background(&self) -> usize {
        let bg = self.px[0];
        self.px.iter().filter(|p| p[0].abs_diff(bg[0]) > 12 || p[1].abs_diff(bg[1]) > 12 || p[2].abs_diff(bg[2]) > 12).count()
    }

    /// Lay several images out in a grid on a dark canvas with a gap.
    pub fn sheet(images: &[Shot8], cols: usize) -> Shot8 {
        let gap = 8;
        let cols = cols.max(1);
        let rows = images.len().div_ceil(cols);
        let cw = images.iter().map(|i| i.w).max().unwrap_or(1);
        let mut row_h = vec![0; rows];
        for (i, im) in images.iter().enumerate() {
            row_h[i / cols] = row_h[i / cols].max(im.h);
        }
        let w = cols * cw + (cols + 1) * gap;
        let h = row_h.iter().sum::<usize>() + (rows + 1) * gap;
        let mut out = Shot8 { w, h, px: vec![[10, 10, 12, 255]; w * h] };
        let mut y0 = gap;
        for (r, rh) in row_h.iter().enumerate() {
            for c in 0..cols {
                let Some(im) = images.get(r * cols + c) else { continue };
                let x0 = gap + c * (cw + gap);
                for y in 0..im.h {
                    for x in 0..im.w {
                        out.px[(y0 + y) * w + x0 + x] = im.px[y * im.w + x];
                    }
                }
            }
            y0 += rh + gap;
        }
        out
    }
}

struct Tex {
    size: [usize; 2],
    px: Vec<egui::Color32>,
}

/// Directory screenshots are written to (`ZESTENGINE_SHOT_DIR`), if set.
pub fn shot_dir() -> Option<PathBuf> {
    std::env::var_os("ZESTENGINE_SHOT_DIR").map(PathBuf::from)
}

pub struct Shot {
    pub ctx: egui::Context,
    pub size: egui::Vec2,
    pub ppp: f32,
    texs: HashMap<TextureId, Tex>,
    /// Pointer position to hover with (points), if any.
    pub pointer: Option<Pos2>,
    /// Frames to run before the capture (layout settles over 2-3).
    pub frames: usize,
    /// Click each point once (press on frame 1, release on frame 2, in order,
    /// one click per frame pair starting at frame 1).
    pub clicks: Vec<Pos2>,
    /// Real time to sleep after each frame (lets background loaders finish).
    pub frame_delay: std::time::Duration,
    time: f64,
}

impl Shot {
    pub fn new(w: f32, h: f32, ppp: f32) -> Self {
        let ctx = egui::Context::default();
        crate::install_icon_font(&ctx);
        crate::theme::apply(&ctx);
        Shot { ctx, size: vec2(w, h), ppp, texs: HashMap::new(), pointer: None, frames: 4, clicks: Vec::new(), frame_delay: std::time::Duration::ZERO, time: 0.0 }
    }

    /// Run the UI closure for `self.frames` frames and rasterise the last.
    pub fn run(&mut self, mut ui: impl FnMut(&egui::Context)) -> Shot8 {
        let mut last = None;
        for frame in 0..self.frames.max(1 + 2 * self.clicks.len() + 2) {
            let mut input = RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, self.size)),
                viewport_id: ViewportId::ROOT,
                time: Some(self.time),
                ..Default::default()
            };
            self.time += 0.05;
            input.viewports.entry(ViewportId::ROOT).or_default().native_pixels_per_point = Some(self.ppp);
            let click_frames = frame.saturating_sub(1) / 2;
            if frame >= 1 && click_frames < self.clicks.len() {
                let pos = self.clicks[click_frames];
                input.events.push(Event::PointerMoved(pos));
                input.events.push(Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: frame % 2 == 1, modifiers: Default::default() });
            } else if let Some(p) = self.pointer {
                input.events.push(Event::PointerMoved(p));
            } else if frame > 0 {
                input.events.push(Event::PointerGone);
            }
            let out = self.ctx.run(input, &mut ui);
            if !self.frame_delay.is_zero() {
                std::thread::sleep(self.frame_delay);
            }
            for (id, d) in &out.textures_delta.set {
                self.apply_delta(*id, d);
            }
            for id in &out.textures_delta.free {
                self.texs.remove(id);
            }
            last = Some((out.shapes, out.pixels_per_point));
        }
        let (shapes, ppp) = last.unwrap();
        let prims = self.ctx.tessellate(shapes, ppp);
        self.raster(ppp, &prims)
    }

    /// Draw `ui` in a panel-coloured central panel with the editor's side
    /// margin, then crop the image to the used height (plus a little).
    pub fn run_panel(&mut self, mut ui: impl FnMut(&mut egui::Ui)) -> Shot8 {
        let bottom = std::cell::Cell::new(self.size.y);
        let mut img = self.run(|ctx| {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(crate::theme::BG_PANEL).inner_margin(egui::Margin::symmetric(8, 6))).show(ctx, |u| {
                let r = u.scope(|u| ui(u));
                bottom.set(r.response.rect.bottom() + 8.0);
            });
        });
        let h = ((bottom.get() * self.ppp) as usize).min(img.h).max(16);
        img.px.truncate(h * img.w);
        img.h = h;
        img
    }

    /// Save to `$ZESTENGINE_SHOT_DIR/<name>.png` when the variable is set.
    pub fn save_named(&self, img: &Shot8, name: &str) {
        if let Some(d) = shot_dir() {
            img.save(&d.join(format!("{name}.png")));
        }
    }

    fn apply_delta(&mut self, id: TextureId, d: &ImageDelta) {
        let egui::ImageData::Color(img) = &d.image;
        match d.pos {
            None => {
                self.texs.insert(id, Tex { size: img.size, px: img.pixels.clone() });
            }
            Some([x, y]) => {
                let t = self.texs.get_mut(&id).expect("patch before full upload");
                for r in 0..img.size[1] {
                    for c in 0..img.size[0] {
                        t.px[(y + r) * t.size[0] + x + c] = img.pixels[r * img.size[0] + c];
                    }
                }
            }
        }
    }

    fn raster(&self, ppp: f32, prims: &[egui::ClippedPrimitive]) -> Shot8 {
        let (w, h) = ((self.size.x * ppp) as usize, (self.size.y * ppp) as usize);
        let mut fb = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];
        for cp in prims {
            let Primitive::Mesh(mesh) = &cp.primitive else { continue };
            let Some(tex) = self.texs.get(&mesh.texture_id) else { continue };
            let clip = cp.clip_rect;
            let cx0 = (clip.min.x * ppp).floor().max(0.0) as usize;
            let cy0 = (clip.min.y * ppp).floor().max(0.0) as usize;
            let cx1 = ((clip.max.x * ppp).ceil().max(0.0) as usize).min(w);
            let cy1 = ((clip.max.y * ppp).ceil().max(0.0) as usize).min(h);
            for tri in mesh.indices.as_chunks::<3>().0 {
                let v = [mesh.vertices[tri[0] as usize], mesh.vertices[tri[1] as usize], mesh.vertices[tri[2] as usize]];
                let p = [v[0].pos * ppp, v[1].pos * ppp, v[2].pos * ppp];
                let area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
                if area.abs() < 1e-6 {
                    continue;
                }
                let minx = p.iter().map(|q| q.x).fold(f32::MAX, f32::min).floor().max(cx0 as f32) as usize;
                let maxx = p.iter().map(|q| q.x).fold(f32::MIN, f32::max).ceil().min(cx1 as f32).max(0.0) as usize;
                let miny = p.iter().map(|q| q.y).fold(f32::MAX, f32::min).floor().max(cy0 as f32) as usize;
                let maxy = p.iter().map(|q| q.y).fold(f32::MIN, f32::max).ceil().min(cy1 as f32).max(0.0) as usize;
                for y in miny..maxy {
                    for x in minx..maxx {
                        let q = pos2(x as f32 + 0.5, y as f32 + 0.5);
                        let w0 = ((p[1].x - q.x) * (p[2].y - q.y) - (p[2].x - q.x) * (p[1].y - q.y)) / area;
                        let w1 = ((p[2].x - q.x) * (p[0].y - q.y) - (p[0].x - q.x) * (p[2].y - q.y)) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                            continue;
                        }
                        let u = w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x;
                        let vv = w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y;
                        let col = |f: fn(egui::Color32) -> u8| (w0 * f(v[0].color) as f32 + w1 * f(v[1].color) as f32 + w2 * f(v[2].color) as f32) / 255.0;
                        let (vr, vg, vb, va) = (col(|c| c.r()), col(|c| c.g()), col(|c| c.b()), col(|c| c.a()));
                        let t = sample(tex, u, vv);
                        let src = [vr * t[0], vg * t[1], vb * t[2], va * t[3]];
                        let dst = &mut fb[y * w + x];
                        for i in 0..3 {
                            dst[i] = src[i] + dst[i] * (1.0 - src[3]);
                        }
                    }
                }
            }
        }
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        Shot8 { w, h, px: fb.iter().map(|p| [q(p[0]), q(p[1]), q(p[2]), 255]).collect() }
    }
}

fn sample(t: &Tex, u: f32, v: f32) -> [f32; 4] {
    let x = (u * t.size[0] as f32 - 0.5).clamp(0.0, t.size[0] as f32 - 1.0);
    let y = (v * t.size[1] as f32 - 0.5).clamp(0.0, t.size[1] as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(t.size[0] - 1), (y0 + 1).min(t.size[1] - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |x: usize, y: usize| {
        let c = t.px[y * t.size[0] + x];
        [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, c.a() as f32 / 255.0]
    };
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut o = [0.0; 4];
    for i in 0..4 {
        o[i] = (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy;
    }
    o
}
