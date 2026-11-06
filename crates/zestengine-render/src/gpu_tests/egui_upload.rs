//! egui texture upload on a real device: `record_textures` (full uploads and
//! partial updates recorded into a caller's command buffer, no queue idle) and
//! `release_retired` (staging buffers and replaced textures freed once the
//! slot's fence passed), checked by drawing the textures and reading pixels.

use std::sync::{Arc, Mutex};

use ash::vk;
use egui::epaint::{ImageDelta, Mesh};
use egui::{ClippedPrimitive, Color32, ColorImage, Pos2, Rect, TextureId, TextureOptions, epaint::Primitive};

use super::{COLOR_OUT, Img, NONE, gpu, read_image, transition};
use crate::cluster_headless::HeadlessGpu;

const N: u32 = 8;
const SCALE: u32 = 32;
const W: u32 = N * SCALE;

fn quad(tex: TextureId) -> ClippedPrimitive {
    let mut mesh = Mesh::with_texture(tex);
    let r = Rect::from_min_max(Pos2::ZERO, Pos2::new(W as f32, W as f32));
    mesh.add_rect_with_uv(r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
    ClippedPrimitive { clip_rect: r, primitive: Primitive::Mesh(mesh) }
}

fn pattern(f: impl Fn(usize, usize) -> Color32) -> ColorImage {
    let mut img = ColorImage::filled([N as usize, N as usize], Color32::BLACK);
    for y in 0..N as usize {
        for x in 0..N as usize {
            img.pixels[y * N as usize + x] = f(x, y);
        }
    }
    img
}

fn base(x: usize, y: usize) -> Color32 {
    Color32::from_rgb((x * 30) as u8, (y * 30) as u8, ((x + y) * 15) as u8)
}

struct Fixture<'a> {
    g: &'a HeadlessGpu,
    alloc: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    renderer: egui_ash_renderer::Renderer,
}

impl<'a> Fixture<'a> {
    fn new(g: &'a HeadlessGpu, frames: usize) -> Self {
        let alloc = Arc::new(Mutex::new(g.allocator().unwrap()));
        let renderer = egui_ash_renderer::Renderer::with_gpu_allocator(
            alloc.clone(),
            g.device.clone(),
            egui_ash_renderer::DynamicRendering { color_attachment_format: vk::Format::R8G8B8A8_SRGB, depth_attachment_format: None },
            egui_ash_renderer::Options { in_flight_frames: frames, enable_depth_test: false, enable_depth_write: false, srgb_framebuffer: true },
        )
        .unwrap();
        Self { g, alloc, renderer }
    }

    /// Record `deltas` + a draw of `prims` into one command buffer for frame
    /// `slot`, submit, wait, release the slot's retired resources, read the
    /// RGBA8 target back.
    fn frame(&mut self, slot: usize, deltas: &[(TextureId, ImageDelta)], prims: &[ClippedPrimitive], w: u32, h: u32) -> Vec<u8> {
        let g = self.g;
        let mut target = {
            let mut a = self.alloc.lock().unwrap();
            Img::new(g, &mut a, w, h, 1, vk::Format::R8G8B8A8_SRGB, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap()
        };
        let r = &mut self.renderer;
        let device = &g.device;
        crate::alloc::one_time_submit(device, g.pool, g.queue, |cb| unsafe {
            r.record_textures(cb, slot, deltas).unwrap();
            transition(g, cb, &target, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(target.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att));
            r.cmd_draw(cb, area.extent, 1.0, prims).unwrap();
            device.cmd_end_rendering(cb);
        })
        .unwrap();
        // The submit above waited for its fence: this slot's resources can go.
        self.renderer.release_retired(slot).unwrap();
        let mut a = self.alloc.lock().unwrap();
        let bytes = read_image(g, &mut a, &target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT, 4).unwrap();
        target.destroy(g, &mut a);
        bytes
    }
}

/// Centre pixel of texel (x, y) of the scaled-up quad.
fn texel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let (cx, cy) = (x * SCALE + SCALE / 2 - 1, y * SCALE + SCALE / 2 - 1);
    let i = ((cy * W + cx) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn close(a: [u8; 4], c: Color32) -> bool {
    let e = c.to_array();
    (0..4).all(|k| (a[k] as i32 - e[k] as i32).abs() <= 12)
}

#[test]
fn managed_texture_full_upload_partial_update_replace_and_free() {
    let Some(g) = gpu() else { return };
    let mut fx = Fixture::new(&g, 2);
    let id = TextureId::Managed(7);
    let prims = [quad(id)];

    // 1. Full upload, drawn by nearest-sampled pixels.
    let full = ImageDelta::full(pattern(base), TextureOptions::NEAREST);
    let px = fx.frame(0, &[(id, full)], &prims, W, W);
    for y in 0..N {
        for x in 0..N {
            assert!(close(texel(&px, x, y), base(x as usize, y as usize)), "full upload texel ({x},{y}): {:?}", texel(&px, x, y));
        }
    }

    // 2. Partial update of a 3x2 rect at (2, 4): only those texels change.
    let patch = ColorImage::filled([3, 2], Color32::from_rgb(250, 10, 120));
    let partial = ImageDelta::partial([2, 4], patch, TextureOptions::NEAREST);
    let px = fx.frame(1, &[(id, partial)], &prims, W, W);
    for y in 0..N {
        for x in 0..N {
            let want = if (2..5).contains(&x) && (4..6).contains(&y) { Color32::from_rgb(250, 10, 120) } else { base(x as usize, y as usize) };
            assert!(close(texel(&px, x, y), want), "after partial update texel ({x},{y}): {:?} want {want:?}", texel(&px, x, y));
        }
    }

    // 3. A full re-upload under the same id replaces the texture (the old one is parked then freed).
    let flipped = ImageDelta::full(pattern(|x, y| base(y, x)), TextureOptions::NEAREST);
    let px = fx.frame(0, &[(id, flipped)], &prims, W, W);
    for y in 0..N {
        for x in 0..N {
            assert!(close(texel(&px, x, y), base(y as usize, x as usize)), "replaced texture texel ({x},{y})");
        }
    }

    // 4. Many frames cycling both slots with full + partial updates: nothing leaks or deadlocks.
    for i in 0..40usize {
        let d = if i % 3 == 0 {
            ImageDelta::full(pattern(|x, y| base((x + i) % 8, y)), TextureOptions::NEAREST)
        } else {
            ImageDelta::partial([i % 6, i % 7], ColorImage::filled([2, 1], Color32::from_gray(i as u8 * 5)), TextureOptions::NEAREST)
        };
        let px = fx.frame(i % 2, &[(id, d)], &prims, W, W);
        assert_eq!(px.len(), (W * W * 4) as usize);
    }

    // An update to a texture that never existed is an error, not a crash.
    let bogus = ImageDelta::partial([0, 0], ColorImage::filled([1, 1], Color32::WHITE), TextureOptions::NEAREST);
    let mut ok = false;
    let r = &mut fx.renderer;
    crate::alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| {
        ok = r.record_textures(cb, 0, &[(TextureId::Managed(999), bogus)]).is_err();
    })
    .unwrap();
    assert!(ok, "a partial update of an unknown texture must fail cleanly");
    fx.renderer.free_textures(&[id]).unwrap();
    drop(fx.renderer);
    drop(fx.alloc);
}

/// The real egui font atlas: frame 1 uploads it in full, frame 2 adds glyphs
/// through partial updates; the text drawn in frame 1 must look the same in both.
#[test]
fn egui_font_atlas_partial_updates_keep_earlier_text_intact() {
    let Some(g) = gpu() else { return };
    let mut fx = Fixture::new(&g, 2);
    let ctx = egui::Context::default();
    let (w, h) = (320u32, 120u32);
    let run = |ctx: &egui::Context, extra: bool| {
        let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(w as f32, h as f32))), ..Default::default() };
        let out = ctx.run(input, |ctx| {
            egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
                ui.label("Hello egui");
                if extra {
                    ui.label("Quartz jumping zebras vex fjord 0123456789");
                }
            });
        });
        let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
        (out.textures_delta, prims)
    };
    let (d1, p1) = run(&ctx, false);
    let deltas1: Vec<_> = d1.set.clone();
    let px1 = fx.frame(0, &deltas1, &p1, w, h);
    let (d2, p2) = run(&ctx, true);
    assert!(d2.set.iter().any(|(_, d)| d.pos.is_some()), "second frame should use a partial atlas update");
    let deltas2: Vec<_> = d2.set.clone();
    let px2 = fx.frame(1, &deltas2, &p2, w, h);
    let lit = px1.chunks(4).filter(|p| p[3] > 0).count();
    assert!(lit > 150, "first frame drew no text ({lit} px)");
    // The first label's rows are identical in both frames.
    let rows = 17usize; // the first label only; the second starts below
    let n = rows * w as usize * 4;
    let diff = px1[..n].iter().zip(&px2[..n]).filter(|(a, b)| (**a as i32 - **b as i32).abs() > 2).count();
    eprintln!("egui font atlas: {lit} lit px in frame 1; first-label diff after partial update: {diff} bytes");
    if diff != 0 {
        let mut shown = 0;
        for (i, (a, b)) in px1[..n].iter().zip(&px2[..n]).enumerate() {
            if (*a as i32 - *b as i32).abs() > 2 && shown < 12 {
                eprintln!("  differs at px ({}, {}) ch {}: {a} vs {b}", (i / 4) % w as usize, (i / 4) / w as usize, i % 4);
                shown += 1;
            }
        }
    }
    assert_eq!(diff, 0, "text from the first frame changed after a partial atlas update");
    let lit2 = px2.chunks(4).filter(|p| p[3] > 0).count();
    assert!(lit2 > lit * 2, "the second label did not draw (new glyphs missing?): {lit} -> {lit2}");
    fx.renderer.free_textures(&d2.free).unwrap();
}
