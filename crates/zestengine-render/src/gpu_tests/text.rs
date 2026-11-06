//! The text pipeline on a real device: every pipeline variant, atlas page
//! upload, and rendered coverage against CPU references of the shader's SDF
//! model (plus an analytic circle for the single-channel atlas).

use ash::vk;
use glam::{Mat4, Vec3};
use zestengine_text::atlas::{AtlasFormat, AtlasUpload};
use zestengine_text::{
    Billboard, Effects, GlyphInstance, LayoutParams, TextConfig, TextDraw, TextHandle, TextRequest, TextShared, TextSpace,
    TextSystem,
};

use super::{PipeKey, TextGpu};
use crate::gpu_tests::{COLOR_OUT, Img, NONE, gpu, read_image, transition};
use crate::alloc::Buffer;
use crate::cluster_headless::HeadlessGpu;
use crate::pipeline::PipelineCache;

fn plex() -> Vec<u8> {
    std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../zestengine-editor/assets/fonts/IBMPlexSans-Regular.ttf"))
        .expect("the editor's IBM Plex Sans font")
}

/// CPU copy of the atlas pages, built from the same uploads the GPU gets.
struct CpuAtlas {
    page: usize,
    pages: Vec<Vec<u8>>,
    bpp: usize,
}

impl CpuAtlas {
    fn new(page: u32, layers: u32, bpp: usize) -> Self {
        let page = page as usize;
        Self { page, pages: vec![vec![0; page * page * bpp]; layers as usize], bpp }
    }

    fn apply(&mut self, u: &AtlasUpload) {
        for row in 0..u.h as usize {
            let dst = ((u.y as usize + row) * self.page + u.x as usize) * self.bpp;
            let src = row * u.w as usize * self.bpp;
            self.pages[u.layer as usize][dst..dst + u.w as usize * self.bpp].copy_from_slice(&u.data[src..src + u.w as usize * self.bpp]);
        }
    }

    /// Bilinear sample (texel centres at +0.5, clamp to edge), channels 0..1.
    fn sample(&self, layer: usize, uv: [f32; 2]) -> [f32; 4] {
        let p = self.page as f32;
        let (x, y) = (uv[0] * p - 0.5, uv[1] * p - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xi: f32, yi: f32| -> [f32; 4] {
            let xi = (xi.max(0.0) as usize).min(self.page - 1);
            let yi = (yi.max(0.0) as usize).min(self.page - 1);
            let o = (yi * self.page + xi) * self.bpp;
            let mut t = [0.0; 4];
            for (c, v) in t.iter_mut().enumerate().take(self.bpp) {
                *v = self.pages[layer][o + c] as f32 / 255.0;
            }
            t
        };
        let mut out = [0.0; 4];
        let (a, b, c, d) = (at(x0, y0), at(x0 + 1.0, y0), at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
        for k in 0..4 {
            out[k] = (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy;
        }
        out
    }
}

/// text.frag's face coverage for one fragment (no outline, no supersample):
/// `sd` is the sampled distance, `tpp` texels per screen pixel.
fn shader_coverage(sd: f32, spread: f32, tpp: f32) -> f32 {
    let px_range = (2.0 * spread / tpp.max(1e-4)).max(1.0);
    ((sd - 0.5) * px_range + 0.5).clamp(0.0, 1.0)
}

fn median3(v: [f32; 3]) -> f32 {
    v[0].min(v[1]).max(v[0].max(v[1]).min(v[2]))
}

struct TextRig<'a> {
    g: &'a HeadlessGpu,
    alloc: gpu_allocator::vulkan::Allocator,
    cache: PipelineCache,
    handle: TextHandle,
    text: TextGpu,
    rgba: CpuAtlas,
    r8: CpuAtlas,
}

impl<'a> TextRig<'a> {
    fn new(g: &'a HeadlessGpu) -> Self {
        let mut alloc = g.allocator().unwrap();
        let cache = PipelineCache::new(&g.device, vk::Format::R16G16B16A16_SFLOAT, vk::Format::D32_SFLOAT, vk::Format::A2B10G10R10_UNORM_PACK32).unwrap();
        let cfg = TextConfig::default();
        let mut sys = TextSystem::new(cfg);
        sys.add_builtin_font("IBM Plex Sans", plex()).expect("font parses");
        let handle = TextShared::new(sys);
        let text = TextGpu::new(&g.device, &mut alloc, g.pool, g.queue, handle.clone(), cache.set0_layout).unwrap();
        Self {
            g,
            alloc,
            cache,
            handle,
            text,
            rgba: CpuAtlas::new(cfg.atlas_rgba.page_size, cfg.atlas_rgba.max_pages, 4),
            r8: CpuAtlas::new(cfg.atlas_r8.page_size, cfg.atlas_r8.max_pages, 1),
        }
    }

    /// Lay `text` out and rasterize every glyph (the per-frame budget may
    /// defer some; loop until complete). Uploads go to both the GPU queue and
    /// the CPU copy.
    fn block(&mut self, text: &str, size: f32) -> std::sync::Arc<zestengine_text::TextBlock> {
        let req = TextRequest {
            text,
            font: "",
            rich: false,
            params: LayoutParams { size, wrap: false, ..Default::default() },
            effects: Effects::default(),
            rtl: None,
        };
        for _ in 0..200 {
            let mut sys = self.handle.system.lock().unwrap();
            let b = sys.block(&req).expect("a font is registered");
            if b.complete {
                let ups = sys.take_uploads();
                drop(sys);
                self.upload(ups);
                return b;
            }
            sys.next_frame();
        }
        panic!("glyphs never completed");
    }

    fn upload(&mut self, ups: Vec<AtlasUpload>) {
        for u in &ups {
            match u.format {
                AtlasFormat::Rgba8 => self.rgba.apply(u),
                AtlasFormat::R8 => self.r8.apply(u),
            }
        }
        self.text.pending.extend(ups);
    }

    fn screen_draw(&self, instances: std::sync::Arc<[GlyphInstance]>, scale: f32, offset: [f32; 2]) -> TextDraw {
        let m = Mat4::from_translation(Vec3::new(offset[0], offset[1], 0.0)) * Mat4::from_scale(Vec3::new(scale, scale, 1.0));
        TextDraw {
            instances,
            transform: m.to_cols_array_2d(),
            space: TextSpace::Screen,
            billboard: Billboard::None,
            tint: [1.0, 1.0, 1.0, 1.0],
            local_offset: [0.0, 0.0],
            clip: None,
            supersample: false,
            order: 0,
        }
    }

    /// Render the queued screen draws into a fresh `w` x `h` target of `format`
    /// (cleared to transparent black) and return its RGBA8 / RGBA16F bytes.
    fn render_screen(&mut self, draws: Vec<TextDraw>, w: u32, h: u32, format: vk::Format, bpp: usize) -> Vec<u8> {
        for d in draws {
            self.handle.push(d);
        }
        let g = self.g;
        let mut target = Img::new(g, &mut self.alloc, w, h, 1, format, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap();
        let device = &g.device;
        let text = &mut self.text;
        crate::alloc::one_time_submit(device, g.pool, g.queue, |cb| unsafe {
            text.prepare(device, cb, 0);
            transition(g, cb, &target, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(target.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att));
            text.record_screen(device, cb, 0, area.extent, format);
            device.cmd_end_rendering(cb);
        })
        .unwrap();
        text.end_frame();
        let bytes = read_image(g, &mut self.alloc, &target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT, bpp).unwrap();
        target.destroy(g, &mut self.alloc);
        bytes
    }

    fn destroy(mut self) {
        self.text.destroy(&self.g.device, &mut self.alloc);
        self.cache.destroy(&self.g.device);
    }
}

/// Alpha channel of a pixel in an RGBA8 buffer.
fn alpha8(px: &[u8], w: u32, x: u32, y: u32) -> f32 {
    px[((y * w + x) * 4 + 3) as usize] as f32 / 255.0
}

/// The CPU model of text.frag for an axis-aligned `scale` screen draw at
/// `offset`: coverage of pixel (x, y) from the glyph instances and atlas.
fn model_alpha(rig: &TextRig, inst: &[GlyphInstance], scale: f32, offset: [f32; 2], x: u32, y: u32) -> f32 {
    let lp = [(x as f32 + 0.5 - offset[0]) / scale, (y as f32 + 0.5 - offset[1]) / scale];
    let mut best = 0.0f32;
    for g in inst {
        let kind = (g.meta >> 8) & 3;
        if kind == 3 {
            continue;
        }
        if lp[0] < g.pos[0] || lp[1] < g.pos[1] || lp[0] > g.pos[0] + g.size[0] || lp[1] > g.pos[1] + g.size[1] {
            continue;
        }
        let u = |i: usize| g.uv[i] as f32 / 65535.0;
        let c = [(lp[0] - g.pos[0]) / g.size[0], (lp[1] - g.pos[1]) / g.size[1]];
        let uv = [u(0) + (u(2) - u(0)) * c[0], u(1) + (u(3) - u(1)) * c[1]];
        let layer = (g.meta & 0xff) as usize;
        let spread = (((g.meta >> 16) & 0xff) as f32 / 8.0).max(1.0);
        let (atlas, mtsdf) = if kind == 0 { (&rig.rgba, true) } else { (&rig.r8, false) };
        let page = atlas.page as f32;
        // Texels per screen pixel from the quad's uv gradient.
        let dudx = (u(2) - u(0)) / g.size[0] / scale * page;
        let dvdy = (u(3) - u(1)) / g.size[1] / scale * page;
        let tpp = 0.5 * (dudx.abs() + dvdy.abs());
        let t = atlas.sample(layer, uv);
        let sd = if mtsdf { median3([t[0], t[1], t[2]]) } else { t[0] };
        best = best.max(shader_coverage(sd, spread, tpp));
    }
    best
}

#[test]
fn text_pipelines_exist_for_every_variant() {
    let Some(g) = gpu() else { return };
    let mut rig = TextRig::new(&g);
    let formats = [vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_SRGB, vk::Format::R16G16B16A16_SFLOAT, vk::Format::A2B10G10R10_UNORM_PACK32];
    let mut made = 0;
    for &color in &formats {
        let key = |world, gbuf, velocity, multiview, samples| PipeKey {
            world,
            color,
            depth: vk::Format::D32_SFLOAT,
            gbuf,
            gbuf_format: vk::Format::A2B10G10R10_UNORM_PACK32,
            velocity,
            multiview,
            samples,
        };
        // Screen: the swapchain pass, one sample.
        rig.text.pipeline(&g.device, key(false, false, false, false, 1)).unwrap();
        made += 1;
        // Scene pass: single and 4x MSAA, G-buffer / velocity attachments, multiview.
        for samples in [1u8, 4] {
            for (gbuf, velocity) in [(false, false), (true, false), (false, true), (true, true)] {
                rig.text.pipeline(&g.device, key(true, gbuf, velocity, false, samples)).unwrap();
                made += 1;
                if g.multiview {
                    rig.text.pipeline(&g.device, key(true, gbuf, velocity, true, samples)).unwrap();
                    made += 1;
                }
            }
        }
    }
    eprintln!("text: {made} pipeline variants created (multiview device support: {})", g.multiview);
    rig.destroy();
}

#[test]
fn rendered_string_matches_the_cpu_sdf_model() {
    let Some(g) = gpu() else { return };
    let mut rig = TextRig::new(&g);
    let b = rig.block("Hello, Zest 123 fi AVW", 32.0);
    assert!(b.instances.len() >= 15);
    // Atlas upload layout: the GPU atlas page must hold exactly the uploaded texels.
    let (w, h) = (((b.width + 16.0).ceil()) as u32, ((b.height + 16.0).ceil()) as u32);
    let offset = [8.0, 8.0];
    let draw = rig.screen_draw(b.instances.clone(), 1.0, offset);
    let px = rig.render_screen(vec![draw], w, h, vk::Format::R8G8B8A8_UNORM, 4);
    let mut worst = 0.0f32;
    let mut bad = 0;
    let mut covered = 0;
    let mut sum_err = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let got = alpha8(&px, w, x, y);
            let want = model_alpha(&rig, &b.instances, 1.0, offset, x, y);
            let e = (got - want).abs();
            worst = worst.max(e);
            sum_err += e as f64;
            if e > 8.0 / 255.0 {
                bad += 1;
            }
            if want > 0.5 {
                covered += 1;
            }
        }
    }
    eprintln!("text 1x: {w}x{h} px, {covered} covered, max |gpu-cpu| alpha = {worst:.4}, mean {:.6}, {bad} pixels off by > 8/255", sum_err / (w * h) as f64);
    assert!(covered > 300);
    assert_eq!(bad, 0, "pixels where the GPU coverage disagrees with the CPU model");
    assert!(worst < 0.05);
    // And the silhouette against the plain SDF threshold (median > 0.5): >= 99% of pixels agree.
    let mut agree = 0;
    for y in 0..h {
        for x in 0..w {
            let thr = model_alpha(&rig, &b.instances, 1.0, offset, x, y) > 0.5;
            if (alpha8(&px, w, x, y) > 0.5) == thr {
                agree += 1;
            }
        }
    }
    assert!(agree as f32 / (w * h) as f32 > 0.99);
    rig.destroy();
}

/// Width (px) of the anti-aliased ramp at the left edge of the first stem on
/// row `y`, estimated as 6 * sum(a * (1 - a)) (a unit-width linear ramp gives 1).
fn edge_width(px: &[u8], w: u32, y: u32) -> Option<f32> {
    let row: Vec<f32> = (0..w).map(|x| alpha8(px, w, x, y)).collect();
    let first = row.iter().position(|&a| a > 0.02)?;
    // The ramp ends where coverage reaches (almost) 1.
    let end = row[first..].iter().position(|&a| a > 0.98)? + first;
    Some(6.0 * row[first..=end].iter().map(|a| a * (1.0 - a)).sum::<f32>())
}

#[test]
fn edge_antialiasing_is_one_pixel_at_any_scale() {
    let Some(g) = gpu() else { return };
    let mut rig = TextRig::new(&g);
    // 'l' is a plain vertical stem: sample the ramp mid-height.
    let b = rig.block("l", 64.0);
    let gi = b.instances[0];
    let mid = gi.pos[1] + gi.size[1] * 0.5;
    let mut widths = Vec::new();
    for scale in [0.5f32, 1.0, 4.0, 20.0] {
        // Put the stem's left edge ~8 px inside the target and the row through its middle.
        let stem_left = gi.pos[0] + gi.size[0] * 0.25;
        let (w, h) = (96u32, 8u32);
        let mut measured = Vec::new();
        // Several sub-pixel phases so the estimate isn't tied to one alignment.
        for phase in 0..4 {
            let off = [8.0 - stem_left * scale + phase as f32 * 0.25, 4.0 - mid * scale];
            let draw = rig.screen_draw(b.instances.clone(), scale, off);
            let px = rig.render_screen(vec![draw], w, h, vk::Format::R8G8B8A8_UNORM, 4);
            if let Some(m) = edge_width(&px, w, 4) {
                measured.push(m);
            }
        }
        assert!(!measured.is_empty(), "no edge found at {scale}x");
        let mean = measured.iter().sum::<f32>() / measured.len() as f32;
        eprintln!("text edge ramp width at {scale}x: {mean:.2} px (phases {measured:.2?})");
        widths.push((scale, mean));
    }
    for (scale, wpx) in widths {
        assert!((0.6..=1.5).contains(&wpx), "edge ramp at {scale}x is {wpx:.2} px, expected ~1");
    }
    rig.destroy();
}

/// Single-channel SDF atlas: a synthetic exact circle distance field, so the
/// reference is analytic (not another copy of the shader formula).
#[test]
fn r8_sdf_atlas_renders_an_exact_circle() {
    let Some(g) = gpu() else { return };
    let mut rig = TextRig::new(&g);
    // A 64x64 field, spread 8 texels: sd = 0.5 + dist_to_edge / (2 * spread), circle radius 20 centred in the rect.
    let (n, spread, radius) = (64u32, 8.0f32, 20.0f32);
    let mut data = vec![0u8; (n * n) as usize];
    for y in 0..n {
        for x in 0..n {
            let d = radius - (((x as f32 + 0.5 - 32.0).powi(2) + (y as f32 + 0.5 - 32.0).powi(2)).sqrt());
            data[(y * n + x) as usize] = ((0.5 + d / (2.0 * spread)).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    // Place it at an odd position on layer 2 so a wrong layer/offset shows.
    let up = AtlasUpload { format: AtlasFormat::R8, layer: 2, x: 37, y: 101, w: n, h: n, data };
    rig.upload(vec![up]);
    let page = 1024.0f32;
    let pack = |v: u32| ((v as f32 / page) * 65535.0).round() as u16;
    let scale = 3.0f32; // quad of 64 layout px drawn 3x
    let inst = GlyphInstance {
        pos: [0.0, 0.0],
        size: [n as f32, n as f32],
        uv: [pack(37), pack(101), pack(37 + n), pack(101 + n)],
        color: u32::from_le_bytes([255, 255, 255, 255]),
        meta: zestengine_text::gpu::pack_meta(2, zestengine_text::stack::GlyphKind::Sdf, 0, spread),
        fx: 0,
        outline: 0,
    };
    let (w, h) = (n * 3, n * 3);
    let draw = rig.screen_draw(vec![inst].into(), scale, [0.0, 0.0]);
    let px = rig.render_screen(vec![draw], w, h, vk::Format::R8G8B8A8_UNORM, 4);
    // Exact coverage of the circle (radius 20 layout px = 60 screen px) by 4x4 supersampling.
    let mut worst = 0.0f32;
    let mut sum = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let mut cov = 0.0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let (fx, fy) = (x as f32 + (sx as f32 + 0.5) / 4.0, y as f32 + (sy as f32 + 0.5) / 4.0);
                    if ((fx - 96.0).powi(2) + (fy - 96.0).powi(2)).sqrt() <= radius * scale {
                        cov += 1.0 / 16.0;
                    }
                }
            }
            let e = (alpha8(&px, w, x, y) - cov).abs();
            worst = worst.max(e);
            sum += e as f64;
        }
    }
    eprintln!("r8 circle: max |alpha - exact coverage| = {worst:.3}, mean {:.5}", sum / (w * h) as f64);
    assert!(sum / ((w * h) as f64) < 0.002, "mean coverage error");
    assert!(worst < 0.35, "worst coverage error {worst}");
    rig.destroy();
}

/// Scene-pass text: depth tested, drawn through the real world pipeline and
/// frame UBO layout, plus the multiview variant (two eyes, one pass).
#[test]
fn world_text_is_depth_tested_and_multiview_draws_both_eyes() {
    let Some(g) = gpu() else { return };
    let mut rig = TextRig::new(&g);
    let b = rig.block("Zest", 48.0);
    let (w, h) = (256u32, 128u32);
    // Camera at the origin looking down -Z; text 3 m ahead, 0.02 m per layout px.
    let proj = Mat4::perspective_rh(1.0, w as f32 / h as f32, 0.1, 50.0);
    let view = Mat4::IDENTITY;
    let mut ubo: crate::FrameUbo = bytemuck::Zeroable::zeroed();
    ubo.view = view.to_cols_array_2d();
    ubo.proj = proj.to_cols_array_2d();
    ubo.view_proj = (proj * view).to_cols_array_2d();
    // Eye 1 is shifted 0.5 m to the left of eye 0 (the text moves right on screen).
    let eye = [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0))];
    for (e, eye_m) in eye.iter().enumerate() {
        ubo.eye_view[e] = eye_m.to_cols_array_2d();
        ubo.eye_view_proj[e] = (proj * eye_m).to_cols_array_2d();
    }
    let mut ubo_buf = Buffer::new(&g.device, &mut rig.alloc, std::mem::size_of::<crate::FrameUbo>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, gpu_allocator::MemoryLocation::CpuToGpu, "text-world-ubo").unwrap();
    ubo_buf.write(0, bytemuck::bytes_of(&ubo));
    let pool = unsafe {
        g.device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]),
                None,
            )
            .unwrap()
    };
    let layouts = [rig.cache.set0_layout];
    let set = unsafe { g.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).unwrap()[0] };
    let info = [vk::DescriptorBufferInfo::default().buffer(ubo_buf.handle).range(vk::WHOLE_SIZE)];
    unsafe {
        g.device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&info)], &[]);
    }
    let model = Mat4::from_translation(Vec3::new(-0.6, 0.3, -3.0)) * Mat4::from_scale(Vec3::new(0.02, -0.02, 0.02));
    let color_fmt = vk::Format::R8G8B8A8_UNORM;

    // Render once with the depth buffer cleared to `depth_clear`, `multiview` or not.
    let run = |rig: &mut TextRig, depth_clear: f32, multiview: bool| -> Vec<u8> {
        let draw = TextDraw {
            instances: b.instances.clone(),
            transform: model.to_cols_array_2d(),
            space: TextSpace::World,
            billboard: Billboard::None,
            tint: [1.0; 4],
            local_offset: [0.0; 2],
            clip: None,
            supersample: false,
            order: 0,
        };
        rig.handle.push(draw);
        let layers = if multiview { 2 } else { 1 };
        let mut color = Img::new(&g, &mut rig.alloc, w, h, layers, color_fmt, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap();
        let mut depth = Img::new(&g, &mut rig.alloc, w, h, layers, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::SampleCountFlags::TYPE_1).unwrap();
        let pass = crate::pipeline::PassFormats { color: color_fmt, depth: vk::Format::D32_SFLOAT, gbuf: vk::Format::A2B10G10R10_UNORM_PACK32, multiview, samples: 1 };
        let device = &g.device;
        let text = &mut rig.text;
        crate::alloc::one_time_submit(device, g.pool, g.queue, |cb| unsafe {
            text.prepare(device, cb, 0);
            transition(&g, cb, &color, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
            transition(
                &g,
                cb,
                &depth,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                NONE,
                (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ),
            );
            let catt = [vk::RenderingAttachmentInfo::default()
                .image_view(color.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            let datt = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: depth_clear, stencil: 0 } });
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } };
            let mut ri = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&catt).depth_attachment(&datt);
            if multiview {
                ri = ri.view_mask(0b11);
            }
            device.cmd_begin_rendering(cb, &ri);
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: w as f32, height: h as f32, min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cb, 0, &[area]);
            text.record_world(device, cb, 0, set, pass, false, false);
            device.cmd_end_rendering(cb);
        })
        .unwrap();
        text.end_frame();
        let bytes = read_image(&g, &mut rig.alloc, &color, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT, 4).unwrap();
        color.destroy(&g, &mut rig.alloc);
        depth.destroy(&g, &mut rig.alloc);
        bytes
    };
    let count = |px: &[u8]| px.chunks(4).filter(|p| p[3] > 128).count();
    let visible = run(&mut rig, 1.0, false);
    let n_visible = count(&visible);
    eprintln!("world text: {n_visible} covered pixels with an empty depth buffer");
    assert!(n_visible > 200, "world text did not render");
    let hidden = run(&mut rig, 0.0, false);
    assert_eq!(count(&hidden), 0, "text must be depth tested against nearer geometry");
    if g.multiview {
        let mv = run(&mut rig, 1.0, true);
        let per = (w * h * 4) as usize;
        let (e0, e1) = (&mv[..per], &mv[per..]);
        assert!(count(e0) > 200 && count(e1) > 200, "both eyes must draw");
        // Eye 0 matches the single-view render; eye 1 is the same text shifted right.
        let centroid = |px: &[u8]| {
            let (mut sx, mut n) = (0.0f64, 0.0f64);
            for (i, p) in px.chunks(4).enumerate() {
                if p[3] > 128 {
                    sx += (i as u32 % w) as f64;
                    n += 1.0;
                }
            }
            sx / n
        };
        let (c0, c1) = (centroid(e0), centroid(e1));
        assert!((c0 - centroid(&visible)).abs() < 0.01, "eye 0 differs from the single-view pass");
        // 0.5 m at 3 m with a 1.0 rad vertical fov and aspect 2: f = 1/tan(0.5) -> x px = 0.5/3 * f / aspect * (w/2).
        let expect = 0.5 / 3.0 * (1.0 / 0.5f64.tan()) / 2.0 * (w as f64 / 2.0);
        eprintln!("multiview eye shift: {:.2} px (expected {expect:.2})", c1 - c0);
        assert!(((c1 - c0) - expect).abs() < 1.0);
    }
    unsafe { g.device.destroy_descriptor_pool(pool, None) };
    ubo_buf.destroy(&g.device, &mut rig.alloc);
    rig.destroy();
}
