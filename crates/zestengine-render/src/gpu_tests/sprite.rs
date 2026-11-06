//! The 2D sprite pass on a real device (2D-3 phase 2; run and passing). Every
//! test skips cleanly when `gpu()` finds no device or
//! `ZESTENGINE_TEST_GPU=0`. Run with
//! `cargo test -p zestengine-render --lib gpu_tests::sprite -- --test-threads=1`.
//!
//! Covered: sorting (layer, order, group), the 9-slice vertex shader incl. a
//! target smaller than the border sum, Tiled expansion, batching counts
//! on a real frame, and premultiplied blending. This rig draws over plain
//! `D32_SFLOAT` (stencil off); sprite masks are covered over a depth-stencil
//! target in `gpu_tests/stencil.rs`.

use ash::vk;
use glam::Mat4;

use crate::alloc::Buffer;
use crate::cluster_headless::HeadlessGpu;
use crate::gpu_tests::{COLOR_OUT, Img, NONE, gpu, read_image, transition};
use crate::pipeline::{PassFormats, PipelineCache};
use crate::sprite2d::{BuildConfig, DrawMode, MeshRange, SortAxis, SpriteMeshPool, SpriteSource, push_animation};
use crate::sprite_gpu::{SpriteBlend, SpriteFrameInput, SpriteGpu, SpriteMaterialDesc, SpritePart};
use crate::texture::GpuTexture;
use crate::types::TextureData;

const W: u32 = 64;
/// World units covered by the target: -2..2 on both axes (16 px per unit).
const HALF: f32 = 2.0;
const PX_PER_UNIT: f32 = W as f32 / (2.0 * HALF);

struct Rig<'a> {
    g: &'a HeadlessGpu,
    alloc: gpu_allocator::vulkan::Allocator,
    cache: PipelineCache,
    sprites: SpriteGpu,
    pool: SpriteMeshPool,
    tex: GpuTexture,
    sampler: vk::Sampler,
    quad: MeshRange,
    ubo: Buffer,
    desc_pool: vk::DescriptorPool,
    frame_set: vk::DescriptorSet,
    /// A pixel rect whose depth is cleared to 0 (a stand-in for 3D geometry in front) before the sprites draw.
    occlude: Option<vk::Rect2D>,
}

impl<'a> Rig<'a> {
    fn new(g: &'a HeadlessGpu) -> Self {
        let mut alloc = g.allocator().unwrap();
        let cache = PipelineCache::new(&g.device, vk::Format::R8G8B8A8_UNORM, vk::Format::D32_SFLOAT, vk::Format::A2B10G10R10_UNORM_PACK32).unwrap();
        let mut sprites = SpriteGpu::new(&g.device, &mut alloc, cache.set0_layout).unwrap();
        let tex = GpuTexture::upload(&g.device, &mut alloc, g.pool, g.queue, &TextureData { width: 4, height: 4, pixels: vec![255; 4 * 4 * 4], srgb: false, hdr: false }).unwrap();
        let sampler = unsafe {
            g.device
                .create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST).address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE), None)
                .unwrap()
        };
        let id = sprites.register_texture(&g.device, tex.view, sampler, false).unwrap();
        assert_eq!(id, 0);
        sprites.register_material(SpriteMaterialDesc { blend: SpriteBlend::Alpha, alpha_cutoff: 0.0 });
        sprites.register_material(SpriteMaterialDesc { blend: SpriteBlend::Premultiplied, alpha_cutoff: 0.0 });
        sprites.register_material(SpriteMaterialDesc { blend: SpriteBlend::Additive, alpha_cutoff: 0.0 });
        sprites.register_material(SpriteMaterialDesc { blend: SpriteBlend::Multiply, alpha_cutoff: 0.0 });
        let mut pool = SpriteMeshPool::new(1024, 4096);
        let quad = pool
            .add(&[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], &[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]], &[0, 1, 2, 2, 1, 3])
            .unwrap();
        // Orthographic camera on the XY plane.
        let proj = Mat4::orthographic_rh(-HALF, HALF, -HALF, HALF, -10.0, 10.0);
        let mut ubo_data: crate::FrameUbo = bytemuck::Zeroable::zeroed();
        ubo_data.proj = proj.to_cols_array_2d();
        ubo_data.view_proj = proj.to_cols_array_2d();
        ubo_data.view = Mat4::IDENTITY.to_cols_array_2d();
        let mut ubo = Buffer::new(&g.device, &mut alloc, std::mem::size_of::<crate::FrameUbo>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, gpu_allocator::MemoryLocation::CpuToGpu, "sprite-test-ubo").unwrap();
        ubo.write(0, bytemuck::bytes_of(&ubo_data));
        let desc_pool = unsafe {
            g.device
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None)
                .unwrap()
        };
        let layouts = [cache.set0_layout];
        let frame_set = unsafe { g.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts)).unwrap()[0] };
        let info = [vk::DescriptorBufferInfo::default().buffer(ubo.handle).range(vk::WHOLE_SIZE)];
        unsafe {
            g.device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(frame_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&info)], &[]);
        }
        Self { g, alloc, cache, sprites, pool, tex, sampler, quad, ubo, desc_pool, frame_set, occlude: None }
    }

    /// Replace the camera (the frame UBO is host visible and rewritten between renders).
    fn set_view_proj(&mut self, vp: Mat4) {
        let mut ubo_data: crate::FrameUbo = bytemuck::Zeroable::zeroed();
        ubo_data.proj = vp.to_cols_array_2d();
        ubo_data.view_proj = vp.to_cols_array_2d();
        ubo_data.view = Mat4::IDENTITY.to_cols_array_2d();
        self.ubo.write(0, bytemuck::bytes_of(&ubo_data));
    }

    /// A unit white sprite at `pos` with a tint.
    fn sprite(&self, pos: [f32; 2], color: [f32; 4], layer: i16, order: i16) -> SpriteSource {
        SpriteSource {
            position: [pos[0], pos[1], 0.0],
            scale: [1.0, 1.0],
            sort_point: [pos[0], pos[1], 0.0],
            mesh: self.quad,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            sprite_size: [1.0, 1.0],
            pivot: [0.5, 0.5],
            color,
            adaptive_threshold: 0.5,
            size: [1.0, 1.0],
            layer,
            order,
            ..Default::default()
        }
    }

    /// Draw `input` into a fresh W x W RGBA8 target (cleared to transparent) and
    /// return its pixels.
    fn render(&mut self, input: &mut SpriteFrameInput) -> Vec<u8> {
        self.render_with(input, false)
    }

    /// As `render`; with `cut`, the sprites are recorded in two parts with a
    /// stand-in "3D transparent" (a blue clear-rect over the middle band) between
    /// them, the way the renderer cuts the scene draw order.
    fn render_with_behind(&mut self, input: &mut SpriteFrameInput, behind: bool) -> Vec<u8> {
        input.config = BuildConfig {
            axis: SortAxis { origin: [0.0; 3], axis: [0.0, 0.0, -1.0] },
            threads: 1,
            behind_layer_max: behind.then_some(0),
            ..Default::default()
        };
        self.render_cut(input, behind)
    }

    /// As `render` with a sorting axis (the default looks along -Z; a top-down camera along -Y).
    fn render_with_axis(&mut self, input: &mut SpriteFrameInput, axis: [f32; 3]) -> Vec<u8> {
        input.config = BuildConfig { axis: SortAxis { origin: [0.0; 3], axis }, threads: 1, ..Default::default() };
        self.render_cut(input, false)
    }

    fn render_with(&mut self, input: &mut SpriteFrameInput, cut: bool) -> Vec<u8> {
        input.config = BuildConfig { axis: SortAxis { origin: [0.0; 3], axis: [0.0, 0.0, -1.0] }, threads: 1, ..Default::default() };
        self.render_cut(input, cut)
    }

    fn render_cut(&mut self, input: &mut SpriteFrameInput, cut: bool) -> Vec<u8> {
        let g = self.g;
        let fmt = vk::Format::R8G8B8A8_UNORM;
        let mut target = Img::new(g, &mut self.alloc, W, W, 1, fmt, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap();
        let mut depth = Img::new(g, &mut self.alloc, W, W, 1, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::SampleCountFlags::TYPE_1).unwrap();
        let pass = PassFormats { color: fmt, depth: vk::Format::D32_SFLOAT, gbuf: vk::Format::A2B10G10R10_UNORM_PACK32, multiview: false, samples: 1 };
        self.sprites.prepare(input, &mut self.pool, 0);
        let device = &g.device;
        let (sprites, set, occlude) = (&mut self.sprites, self.frame_set, self.occlude);
        crate::alloc::one_time_submit(device, g.pool, g.queue, |cb| unsafe {
            transition(g, cb, &target, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
            transition(
                g,
                cb,
                &depth,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                NONE,
                (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ),
            );
            let catt = [vk::RenderingAttachmentInfo::default()
                .image_view(target.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            let datt = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: W } };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&catt).depth_attachment(&datt));
            // The renderer's scene pass uses a negative-height viewport (y up).
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: W as f32, width: W as f32, height: -(W as f32), min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cb, 0, &[area]);
            if let Some(rect) = occlude {
                let clear = vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::DEPTH, color_attachment: 0, clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } } };
                device.cmd_clear_attachments(cb, &[clear], &[vk::ClearRect { rect, base_array_layer: 0, layer_count: 1 }]);
            }
            if cut {
                sprites.record(device, cb, 0, set, pass, false, false, SpritePart::Behind);
                let rect = vk::ClearRect { rect: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 24 }, extent: vk::Extent2D { width: W, height: 16 } }, base_array_layer: 0, layer_count: 1 };
                let blue = vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::COLOR, color_attachment: 0, clear_value: vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 1.0, 1.0] } } };
                device.cmd_clear_attachments(cb, &[blue], &[rect]);
                sprites.record(device, cb, 0, set, pass, false, false, SpritePart::Front);
            } else {
                sprites.record(device, cb, 0, set, pass, false, false, SpritePart::All);
            }
            device.cmd_end_rendering(cb);
        })
        .unwrap();
        let bytes = read_image(g, &mut self.alloc, &target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT, 4).unwrap();
        target.destroy(g, &mut self.alloc);
        depth.destroy(g, &mut self.alloc);
        bytes
    }

    fn destroy(mut self) {
        self.sprites.destroy(&self.g.device, &mut self.alloc);
        self.cache.destroy(&self.g.device);
        self.ubo.destroy(&self.g.device, &mut self.alloc);
        unsafe {
            self.g.device.destroy_descriptor_pool(self.desc_pool, None);
            self.g.device.destroy_sampler(self.sampler, None);
            self.g.device.destroy_image_view(self.tex.view, None);
            self.g.device.destroy_image(self.tex.image, None);
        }
    }
}

fn px(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let o = ((y * W + x) * 4) as usize;
    [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
}

/// Bounding box (x0, y0, x1, y1 exclusive) and count of pixels with alpha > 128.
fn coverage(bytes: &[u8]) -> (u32, u32, u32, u32, usize) {
    let (mut x0, mut y0, mut x1, mut y1, mut n) = (W, W, 0, 0, 0usize);
    for y in 0..W {
        for x in 0..W {
            if px(bytes, x, y)[3] > 128 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
                n += 1;
            }
        }
    }
    (x0, y0, x1, y1, n)
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

#[test]
fn higher_order_draws_on_top_and_layer_beats_order() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput { sprites: vec![rig.sprite([0.0, 0.0], RED, 0, 0), rig.sprite([0.0, 0.0], GREEN, 0, 1)], ..Default::default() };
    let img = rig.render(&mut input);
    assert_eq!(px(&img, W / 2, W / 2), [0, 255, 0, 255]);
    // Submission order must not matter.
    input.sprites = vec![rig.sprite([0.0, 0.0], GREEN, 0, 1), rig.sprite([0.0, 0.0], RED, 0, 0)];
    assert_eq!(px(&rig.render(&mut input), W / 2, W / 2), [0, 255, 0, 255]);
    // A higher layer wins even with a lower order.
    input.sprites = vec![rig.sprite([0.0, 0.0], RED, 1, -100), rig.sprite([0.0, 0.0], GREEN, 0, 100)];
    assert_eq!(px(&rig.render(&mut input), W / 2, W / 2), [255, 0, 0, 255]);
    rig.destroy();
}

#[test]
fn simple_sprite_scale_covers_its_rect() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    // A 1x1 sprite with a centred pivot scaled 2x covers 2 x 2 units.
    let mut s = rig.sprite([0.0, 0.0], RED, 0, 0);
    s.scale = [2.0, 2.0];
    input.sprites = vec![s];
    let (x0, _, x1, _, n) = coverage(&rig.render(&mut input));
    assert!(((x1 - x0) as f32 - 2.0 * PX_PER_UNIT).abs() <= 1.5 && n > 900);
    rig.destroy();
}

#[test]
fn nine_slice_target_smaller_than_the_border_sum_still_fills_its_rect() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    // Border 0.6 on every side (sum 1.2) drawn at 0.8 x 0.8: borders shrink to 0.4 and meet.
    let mut s = rig.sprite([0.0, 0.0], RED, 0, 0);
    s.draw = DrawMode::Sliced;
    s.border = [0.6; 4];
    s.size = [0.8, 0.8];
    input.sprites = vec![s];
    let (x0, y0, x1, y1, n) = coverage(&rig.render(&mut input));
    let want = 0.8 * PX_PER_UNIT;
    assert!(((x1 - x0) as f32 - want).abs() <= 1.5 && ((y1 - y0) as f32 - want).abs() <= 1.5, "bbox {x0},{y0}..{x1},{y1}");
    assert!(n as u32 >= (x1 - x0) * (y1 - y0), "the rect is solid: covered {n} of bbox {}", (x1 - x0) * (y1 - y0));
    // A wide sliced sprite keeps its full height and stretches in width.
    let mut s = rig.sprite([0.0, 0.0], RED, 0, 0);
    s.draw = DrawMode::Sliced;
    s.border = [0.25; 4];
    s.size = [3.0, 1.0];
    input.sprites = vec![s];
    let (x0, y0, x1, y1, _) = coverage(&rig.render(&mut input));
    assert!(((x1 - x0) as f32 - 3.0 * PX_PER_UNIT).abs() <= 1.5 && ((y1 - y0) as f32 - PX_PER_UNIT).abs() <= 1.5);
    rig.destroy();
}

#[test]
fn tiled_sprite_expands_to_one_draw_covering_the_rect() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    let mut s = rig.sprite([0.0, 0.0], RED, 0, 0);
    s.draw = DrawMode::Tiled;
    s.size = [3.5, 1.0]; // 3 whole tiles and a half one
    input.sprites = vec![s];
    let img = rig.render(&mut input);
    let (x0, _, x1, _, _) = coverage(&img);
    assert!(((x1 - x0) as f32 - 3.5 * PX_PER_UNIT).abs() <= 1.5);
    let st = rig.sprites.stats();
    assert_eq!((st.instances, st.draws, st.batches), (4, 1, 1));
    rig.destroy();
}

#[test]
fn many_sprites_with_one_texture_are_one_batch_and_premultiplied_blends() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    for i in 0..200 {
        let x = -1.5 + (i % 10) as f32 * 0.3;
        let y = -1.5 + (i / 10) as f32 * 0.15;
        input.sprites.push(rig.sprite([x, y], RED, 0, (i % 7) as i16));
    }
    let _ = rig.render(&mut input);
    let st = rig.sprites.stats();
    assert_eq!((st.sprites, st.batches), (200, 1), "one material, one texture, no masks");
    // A half-transparent premultiplied tint over nothing: output = rgb*a, alpha a.
    let mut s = rig.sprite([0.0, 0.0], [1.0, 1.0, 1.0, 0.5], 0, 0);
    s.material = 2; // the premultiplied material registered in Rig::new
    input.sprites = vec![s];
    let img = rig.render(&mut input);
    let p = px(&img, W / 2, W / 2);
    assert!((p[3] as i32 - 128).abs() <= 2 && (p[0] as i32 - 128).abs() <= 2, "premultiplied pixel {p:?}");
    rig.destroy();
}

// ---------------------------------------------------------------------------
// Tilemap tiles: Quad draws, shader animation, rotated atlas content, array layers
// (2D-5, run on the GPU once the machine was free)
// ---------------------------------------------------------------------------

fn upload_rgba(rig: &mut Rig, w: u32, h: u32, pixels: Vec<u8>) -> GpuTexture {
    GpuTexture::upload(&rig.g.device, &mut rig.alloc, rig.g.pool, rig.g.queue, &TextureData { width: w, height: h, pixels, srgb: false, hdr: false }).unwrap()
}

fn free_tex(rig: &Rig, t: GpuTexture) {
    unsafe {
        rig.g.device.destroy_image_view(t.view, None);
        rig.g.device.destroy_image(t.image, None);
    }
}

fn tile(pos: [f32; 2], uv: [f32; 4], texture: u32) -> SpriteSource {
    SpriteSource {
        position: [pos[0], pos[1], 0.0],
        scale: [1.0, 1.0],
        sort_point: [pos[0], pos[1], 0.0],
        draw: DrawMode::Quad,
        rect: [-0.5, -0.5, 1.0, 1.0],
        uv_rect: uv,
        color: [1.0; 4],
        texture,
        ..Default::default()
    }
}

#[test]
fn quad_tiles_of_one_texture_are_one_batch_and_one_draw() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    for i in 0..64 {
        input.sprites.push(tile([-1.75 + (i % 8) as f32 * 0.5, -1.75 + (i / 8) as f32 * 0.5], [0.0, 0.0, 1.0, 1.0], 0));
        input.sprites.last_mut().unwrap().scale = [0.5, 0.5];
        input.sprites.last_mut().unwrap().color = RED;
    }
    let img = rig.render(&mut input);
    let st = rig.sprites.stats();
    assert_eq!((st.sprites, st.instances, st.batches, st.draws), (64, 64, 1, 1), "tiles share the quad template: one batch, one instanced draw");
    let (x0, y0, x1, y1, n) = coverage(&img);
    assert_eq!((x1 - x0, y1 - y0), (W, W));
    assert!(n as u32 >= W * W - 8, "the 8 x 8 tiles of 0.5 units tile the whole target: {n}");
    rig.destroy();
}

#[test]
fn animated_tile_frame_comes_from_the_frame_clock() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    // A 2 x 1 texture: red | green.
    let tex = upload_rgba(&mut rig, 2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255]);
    let id = rig.sprites.register_texture(&g.device, tex.view, rig.sampler, false).unwrap();
    let mut input = SpriteFrameInput::default();
    let frames = [([0.0, 0.0, 0.5, 1.0], [-0.5, -0.5, 1.0, 1.0], false), ([0.5, 0.0, 1.0, 1.0], [-0.5, -0.5, 1.0, 1.0], false)];
    let anim = push_animation(&mut input.anim, &frames, 1.0, false);
    let mut t = tile([0.0, 0.0], frames[0].0, id);
    t.scale = [2.0, 2.0];
    t.anim = anim;
    input.sprites = vec![t];
    for (time, want) in [(0.2f32, [255u8, 0, 0, 255]), (1.2, [0, 255, 0, 255]), (2.2, [255, 0, 0, 255])] {
        input.time = time;
        input.anim = input.anim.clone();
        assert_eq!(px(&rig.render(&mut input), W / 2, W / 2), want, "at t = {time}");
        input.sprites = vec![t];
    }
    // A per-instance phase shifts the clock (0.2 + 1.0 = frame 1).
    let mut t2 = t;
    t2.anim_phase = 1.0;
    input.sprites = vec![t2];
    input.time = 0.2;
    assert_eq!(px(&rig.render(&mut input), W / 2, W / 2), [0, 255, 0, 255]);
    free_tex(&rig, tex);
    rig.destroy();
}

#[test]
fn rotated_atlas_content_is_turned_back_in_the_vertex_shader() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    // The sprite is 2 x 1 (left red, right green) stored turned 90 degrees
    // clockwise in a 1 x 2 page: red on top, green below.
    let tex = upload_rgba(&mut rig, 1, 2, vec![255, 0, 0, 255, 0, 255, 0, 255]);
    let id = rig.sprites.register_texture(&g.device, tex.view, rig.sampler, false).unwrap();
    let mut input = SpriteFrameInput::default();
    let mut t = tile([0.0, 0.0], [0.0, 0.0, 1.0, 1.0], id);
    t.rect = [-1.0, -0.5, 2.0, 1.0];
    t.rotated = true;
    input.sprites = vec![t];
    let img = rig.render(&mut input);
    assert_eq!(px(&img, W / 2 - 8, W / 2), [255, 0, 0, 255], "the left half is the sprite's left (red)");
    assert_eq!(px(&img, W / 2 + 8, W / 2), [0, 255, 0, 255], "the right half is green");
    // The same data drawn unrotated would be red above green.
    t.rotated = false;
    input.sprites = vec![t];
    let img = rig.render(&mut input);
    assert_eq!(px(&img, W / 2, W / 2 - 4), [255, 0, 0, 255]);
    free_tex(&rig, tex);
    rig.destroy();
}

#[test]
fn array_layer_selects_the_atlas_page() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let layer = |c: [u8; 4]| crate::types::CompressedTexture { format: crate::types::TexFormat::RgbaUnorm, width: 1, height: 1, mips: vec![c.to_vec()] };
    let (red, green) = (layer([255, 0, 0, 255]), layer([0, 255, 0, 255]));
    let arr = GpuTexture::upload_compressed_layers(&g.device, &mut rig.alloc, g.pool, g.queue, &[&red, &green], true).unwrap();
    let id = rig.sprites.register_texture(&g.device, arr.view, rig.sampler, true).unwrap();
    let mut input = SpriteFrameInput::default();
    let mut a = tile([-1.0, 0.0], [0.0, 0.0, 1.0, 1.0], id);
    a.atlas_layer = 0;
    let mut b = tile([1.0, 0.0], [0.0, 0.0, 1.0, 1.0], id);
    b.atlas_layer = 1;
    input.sprites = vec![a, b];
    let img = rig.render(&mut input);
    assert_eq!(px(&img, W / 4, W / 2), [255, 0, 0, 255]);
    assert_eq!(px(&img, 3 * W / 4, W / 2), [0, 255, 0, 255]);
    assert_eq!(rig.sprites.stats().batches, 1, "the layer is per instance and never breaks a batch");
    free_tex(&rig, arr);
    rig.destroy();
}

#[test]
fn cpu_skinned_dynamic_mesh_follows_its_rewritten_positions_without_stale_slots() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let uvs = [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]];
    let d = rig.pool.add_dynamic(&uvs, &[0, 1, 2, 2, 1, 3]).unwrap();
    // A unit quad whose centre is at x.
    let quad_at = |x: f32| [[x - 0.5, -0.5], [x + 0.5, -0.5], [x - 0.5, 0.5], [x + 0.5, 0.5]];
    let mut input = SpriteFrameInput::default();
    let centre = |img: &[u8]| {
        let (x0, _, x1, _, n) = coverage(img);
        (n, (x0 + x1) as f32 / 2.0 / PX_PER_UNIT - HALF)
    };
    // Move it left, right, left again: every frame shows exactly the latest write
    // (three slots rotate, and a slot is reused only after the others).
    for (frame, x) in [-1.0f32, 1.0, -1.0, 0.5, -0.5].into_iter().enumerate() {
        let range = rig.pool.write_dynamic(d, &quad_at(x)).unwrap();
        let mut s = rig.sprite([0.0, 0.0], RED, 0, 0);
        s.mesh = range;
        input.sprites = vec![s];
        let (n, cx) = centre(&rig.render(&mut input));
        assert!((cx - x).abs() < 0.15, "frame {frame}: drawn at {cx}, written at {x}");
        assert!(n > 200 && n < 300, "one unit quad covers about {} px, got {n}", PX_PER_UNIT * PX_PER_UNIT);
    }
    // An unchanged pose keeps drawing the last slot with no write.
    let mut s = rig.sprite([0.0, 0.0], RED, 0, 0);
    s.mesh = rig.pool.dynamic_range(d).unwrap();
    input.sprites = vec![s];
    for _ in 0..3 {
        let (_, cx) = centre(&rig.render(&mut input));
        assert!((cx + 0.5).abs() < 0.15);
    }
    rig.destroy();
}

// ---------------------------------------------------------------------------
// Sprite Shapes (2D-8): real generator output drawn through rewritable pool meshes
// ---------------------------------------------------------------------------

mod shapes {
    use super::*;
    use crate::sprite2d::ShapeMesh;
    use zestengine_core::sprite_shape::{GenInput, GroupKind, MeshGroup, SpriteMetrics, generate_shape};
    use zestengine_core::sprite_shape::FillTexture;
    use zestengine_core::{AngleRange, EdgeFit, ShapePoint, SpriteShapeProfile, SpriteShapeRenderer};

    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    fn sampler(rig: &Rig, repeat: bool) -> vk::Sampler {
        let mode = if repeat { vk::SamplerAddressMode::REPEAT } else { vk::SamplerAddressMode::CLAMP_TO_EDGE };
        unsafe {
            rig.g
                .device
                .create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST).address_mode_u(mode).address_mode_v(mode), None)
                .unwrap()
        }
    }

    /// Upload and register a texture; returns (id, texture, sampler).
    fn texture(rig: &mut Rig, w: u32, h: u32, texels: &[[u8; 4]], repeat: bool) -> (u32, GpuTexture, vk::Sampler) {
        let t = upload_rgba(rig, w, h, texels.iter().flatten().copied().collect());
        let s = sampler(rig, repeat);
        let id = rig.sprites.register_texture(&rig.g.device, t.view, s, false).unwrap();
        (id, t, s)
    }

    /// Draw every group of a generated mesh through its own pool mesh.
    fn sources(rig: &Rig, mesh: &[(MeshGroup, MeshRange)]) -> Vec<SpriteSource> {
        mesh.iter()
            .map(|(g, range)| {
                let mut s = rig.sprite([0.0, 0.0], [1.0; 4], 0, 0);
                s.mesh = *range;
                s.texture = g.texture;
                s.group = if g.kind == GroupKind::Edge { 1 } else { 0 };
                s
            })
            .collect()
    }

    fn upload(rig: &mut Rig, handles: &mut Vec<ShapeMesh>, groups: &[MeshGroup]) -> Vec<(MeshGroup, MeshRange)> {
        groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                if handles.len() <= i {
                    handles.push(rig.pool.add_shape_mesh((g.positions.len() + 8).max(64), (g.indices.len() + 24).max(192)).unwrap());
                }
                (g.clone(), rig.pool.write_shape_mesh(handles[i], &g.positions, &g.uvs, &g.indices).unwrap())
            })
            .collect()
    }

    fn at_world(img: &[u8], x: f32, y: f32) -> [u8; 4] {
        px(img, ((x + HALF) * PX_PER_UNIT) as u32, ((HALF - y) * PX_PER_UNIT) as u32)
    }

    fn square(size: f32) -> SpriteShapeRenderer {
        let s = size / 2.0;
        SpriteShapeRenderer { points: vec![ShapePoint::at(-s, s), ShapePoint::at(s, s), ShapePoint::at(s, -s), ShapePoint::at(-s, -s)], closed: true, ..Default::default() }
    }

    #[test]
    fn a_closed_fill_covers_its_outline_and_tiles_the_texture_in_world_units() {
        let Some(g) = gpu() else { return };
        let mut rig = Rig::new(&g);
        // 2x2 texture: R G / B W, repeating every 1 unit (4 px per unit would be 2 px / 4... PPU 2 -> 1 unit).
        let (tex, t, s) = texture(&mut rig, 2, 2, &[[255, 0, 0, 255], [0, 255, 0, 255], BLUE, WHITE], true);
        let profile = SpriteShapeProfile { fill_pixels_per_unit: 2.0, ..Default::default() };
        let shape = square(3.0);
        let mesh = generate_shape(&GenInput { shape: &shape, profile: &profile, sprite: &|_| None, fill_texture: Some(FillTexture { texture: tex, size_px: [2, 2] }) });
        assert_eq!(mesh.groups.len(), 1);
        let mut handles = Vec::new();
        let drawn = upload(&mut rig, &mut handles, &mesh.groups);
        let mut input = SpriteFrameInput { sprites: sources(&rig, &drawn), ..Default::default() };
        let img = rig.render(&mut input);
        // 3 x 3 units of solid coverage, nothing outside.
        let (x0, y0, x1, y1, n) = coverage(&img);
        let want = (3.0 * PX_PER_UNIT) as u32;
        assert!((x1 - x0).abs_diff(want) <= 1 && (y1 - y0).abs_diff(want) <= 1, "bbox {x0},{y0}..{x1},{y1}");
        assert!(n as u32 >= (want - 2) * (want - 2), "solid fill: {n} px");
        assert_eq!(at_world(&img, 1.8, 0.0)[3], 0, "outside the outline");
        // Quadrants of the first tile (cell [0,1] x [0,1]): top-left R, top-right G, bottom-left B, bottom-right W.
        assert_eq!(at_world(&img, 0.25, 0.75), [255, 0, 0, 255]);
        assert_eq!(at_world(&img, 0.75, 0.75), [0, 255, 0, 255]);
        assert_eq!(at_world(&img, 0.25, 0.25), BLUE);
        assert_eq!(at_world(&img, 0.75, 0.25), WHITE);
        // It repeats one unit over and under the origin.
        assert_eq!(at_world(&img, 1.25, 0.75), [255, 0, 0, 255]);
        assert_eq!(at_world(&img, -0.25, -0.75), WHITE, "bottom-right of the tile below-left of the origin");
        assert_eq!(at_world(&img, -0.75, -0.25), [255, 0, 0, 255], "top-left of that tile");
        rig.destroy();
        unsafe {
            g.device.destroy_sampler(s, None);
        }
        free_tex_raw(&g, t);
    }

    fn free_tex_raw(g: &HeadlessGpu, t: GpuTexture) {
        unsafe {
            g.device.destroy_image_view(t.view, None);
            g.device.destroy_image(t.image, None);
        }
    }

    #[test]
    fn an_edge_strip_tiles_its_sprite_along_the_line_with_the_right_colours() {
        let Some(g) = gpu() else { return };
        let mut rig = Rig::new(&g);
        // Edge sprite texture: left half red, right half green (2 x 1 texels).
        let (tex, t, s) = texture(&mut rig, 2, 1, &[[255, 0, 0, 255], [0, 255, 0, 255]], false);
        let profile = SpriteShapeProfile { angle_ranges: vec![AngleRange { start: 0.0, end: 360.0, order: 0, sprites: vec!["edge".into()] }], ..Default::default() };
        let metrics = move |name: &str| (name == "edge").then_some(SpriteMetrics { texture: tex, uv_rect: [0.0, 0.0, 1.0, 1.0], size: [1.0, 0.5] });
        let shape = SpriteShapeRenderer { points: vec![ShapePoint::at(-1.5, 0.0), ShapePoint::at(1.5, 0.0)], fit: EdgeFit::Stretch, ..Default::default() };
        let mesh = generate_shape(&GenInput { shape: &shape, profile: &profile, sprite: &metrics, fill_texture: None });
        assert_eq!(mesh.stats.tiles, 3);
        let mut handles = Vec::new();
        let drawn = upload(&mut rig, &mut handles, &mesh.groups);
        let mut input = SpriteFrameInput { sprites: sources(&rig, &drawn), ..Default::default() };
        let img = rig.render(&mut input);
        // Three tiles of width 1: red on the left half of each, green on the right half.
        for (x, want) in [(-1.25, [255, 0, 0, 255]), (-0.75, [0, 255, 0, 255]), (-0.25, [255, 0, 0, 255]), (0.25, [0, 255, 0, 255]), (0.75, [255, 0, 0, 255]), (1.25, [0, 255, 0, 255])] {
            assert_eq!(at_world(&img, x, 0.0), want, "x = {x}");
        }
        // Thickness 0.5 (8 px), centred; nothing beyond the ends.
        let (x0, y0, x1, y1, _) = coverage(&img);
        assert!((x1 - x0).abs_diff((3.0 * PX_PER_UNIT) as u32) <= 1 && (y1 - y0).abs_diff((0.5 * PX_PER_UNIT) as u32) <= 1, "bbox {x0},{y0}..{x1},{y1}");
        assert_eq!(at_world(&img, 0.0, 0.4)[3], 0);
        assert_eq!(at_world(&img, 1.7, 0.0)[3], 0);
        rig.destroy();
        unsafe {
            g.device.destroy_sampler(s, None);
        }
        free_tex_raw(&g, t);
    }

    #[test]
    fn the_dynamic_shape_mesh_follows_the_spline_through_changes_of_size_and_topology() {
        let Some(g) = gpu() else { return };
        let mut rig = Rig::new(&g);
        let (tex, t, s) = texture(&mut rig, 1, 1, &[[255, 0, 0, 255]], true);
        let profile = SpriteShapeProfile::default();
        let ft = Some(FillTexture { texture: tex, size_px: [1, 1] });
        let mut handles = Vec::new();
        let mut input = SpriteFrameInput::default();
        let mut last_sig = None;
        // A square, a bigger square, a pentagon (more vertices and triangles), the first square again.
        let pentagon = |r: f32| SpriteShapeRenderer {
            points: (0..5).map(|i| {
                let a = std::f32::consts::FRAC_PI_2 - i as f32 * std::f32::consts::TAU / 5.0;
                ShapePoint::at(r * a.cos(), r * a.sin())
            }).collect(),
            closed: true,
            ..Default::default()
        };
        let steps = [(square(1.0), 1.0f32), (square(2.0), 2.0), (pentagon(1.5), 3.0), (square(1.0), 1.0)];
        for (frame, (shape, width)) in steps.iter().enumerate() {
            let mesh = generate_shape(&GenInput { shape, profile: &profile, sprite: &|_| None, fill_texture: ft });
            let drawn = upload(&mut rig, &mut handles, &mesh.groups);
            input.sprites = sources(&rig, &drawn);
            let img = rig.render(&mut input);
            let (x0, _, x1, _, n) = coverage(&img);
            let bbox = (x1 - x0) as f32 / PX_PER_UNIT;
            if frame < 2 || frame == 3 {
                assert!((bbox - width).abs() < 0.1, "frame {frame}: width {bbox}, want {width}");
            } else {
                // A regular pentagon of radius 1.5 is about 2.85 wide.
                assert!((bbox - 2.85).abs() < 0.15, "pentagon width {bbox}");
            }
            let area_units = n as f32 / (PX_PER_UNIT * PX_PER_UNIT);
            let want = zestengine_core::sprite_shape::spline::signed_area(
                &zestengine_core::sprite_shape::tessellate(&shape.points, true, shape.tess_mode(), 100).pts,
            )
            .abs();
            assert!((area_units - want).abs() < want * 0.08 + 0.03, "frame {frame}: covered {area_units} units, polygon {want}");
            let sig = (mesh.vertex_count(), mesh.triangle_count(), (bbox * 10.0).round() as i32);
            assert_ne!(Some(sig), last_sig, "each step really changes the topology or size");
            last_sig = Some(sig);
        }
        // Writing never needed a second mesh: one handle served all four shapes.
        assert_eq!(rig.pool.shape_mesh_count(), 1);
        rig.destroy();
        unsafe {
            g.device.destroy_sampler(s, None);
        }
        free_tex_raw(&g, t);
    }

    /// A grass-topped ground with a tiled fill, outer corner sprites and a rope: a
    /// picture for a human. Writes `<dir>/sprite-shape.rgba` (64x64 RGBA8) when
    /// `ZESTENGINE_SHAPE_PREVIEW=<dir>` is set; always checks that the layers drew.
    #[test]
    fn a_composed_shape_draws_fill_edges_corners_and_a_rope() {
        let Some(g) = gpu() else { return };
        let mut rig = Rig::new(&g);
        let brown = [120, 80, 40, 255];
        let dark = [90, 60, 30, 255];
        let (fill, t0, s0) = texture(&mut rig, 2, 2, &[brown, dark, dark, brown], true);
        let leaf = [60, 200, 70, 255];
        let tip = [220, 240, 90, 255];
        let (grass, t1, s1) = texture(&mut rig, 2, 2, &[tip, leaf, leaf, leaf], false);
        let (corner, t2, s2) = texture(&mut rig, 1, 1, &[[255, 140, 0, 255]], false);
        let (rope, t3, s3) = texture(&mut rig, 2, 1, &[[230, 220, 200, 255], [150, 130, 100, 255]], false);
        let metrics = move |n: &str| match n {
            "grass" => Some(SpriteMetrics { texture: grass, uv_rect: [0.0, 0.0, 1.0, 1.0], size: [0.5, 0.4] }),
            "corner" => Some(SpriteMetrics { texture: corner, uv_rect: [0.0, 0.0, 1.0, 1.0], size: [0.3, 0.3] }),
            "rope" => Some(SpriteMetrics { texture: rope, uv_rect: [0.0, 0.0, 1.0, 1.0], size: [0.25, 0.1] }),
            _ => None,
        };
        let mut profile = SpriteShapeProfile {
            angle_ranges: vec![AngleRange { start: -45.0, end: 45.0, order: 1, sprites: vec!["grass".into()] }, AngleRange { start: 45.0, end: 315.0, order: 0, sprites: vec!["grass".into()] }],
            ..Default::default()
        };
        profile.outer_corner = "corner".into();
        profile.fill_pixels_per_unit = 4.0; // a 2 px texture repeats every half unit
        profile.fill_edge_offset = -0.1;
        let mut ground = SpriteShapeRenderer {
            points: vec![ShapePoint::at(-1.6, 0.2), ShapePoint::at(0.0, 0.6), ShapePoint::at(1.6, 0.2), ShapePoint::at(1.6, -1.8), ShapePoint::at(-1.6, -1.8)],
            closed: true,
            ..Default::default()
        };
        ground.points[1].set_mode(zestengine_core::ShapeTangentMode::Continuous, [1.0, 0.0], 0.8);
        let mesh = generate_shape(&GenInput { shape: &ground, profile: &profile, sprite: &metrics, fill_texture: Some(FillTexture { texture: fill, size_px: [2, 2] }) });
        assert!(mesh.stats.corners >= 2 && mesh.stats.fill_triangles > 0 && mesh.stats.tiles > 4, "{:?}", mesh.stats);
        let rope_profile = SpriteShapeProfile { angle_ranges: vec![AngleRange { start: 0.0, end: 360.0, order: 0, sprites: vec!["rope".into()] }], ..Default::default() };
        let mut cable = SpriteShapeRenderer { points: vec![ShapePoint::at(-1.8, 1.7), ShapePoint::at(0.0, 0.95), ShapePoint::at(1.8, 1.7)], fit: EdgeFit::Tile, ..Default::default() };
        cable.points[1].set_mode(zestengine_core::ShapeTangentMode::Continuous, [1.0, 0.0], 0.9);
        let rope_mesh = generate_shape(&GenInput { shape: &cable, profile: &rope_profile, sprite: &metrics, fill_texture: None });
        let mut handles = Vec::new();
        let mut drawn = upload(&mut rig, &mut handles, &mesh.groups);
        let mut rope_handles = Vec::new();
        let rope_drawn = upload(&mut rig, &mut rope_handles, &rope_mesh.groups);
        drawn.extend(rope_drawn);
        let mut input = SpriteFrameInput { sprites: sources(&rig, &drawn), ..Default::default() };
        // Edge groups after the fill, corners (their own texture) after those.
        for (k, sp) in input.sprites.iter_mut().enumerate() {
            sp.group = k as u16;
        }
        let img = rig.render(&mut input);
        assert_eq!(at_world(&img, 0.0, -1.0)[3], 255, "fill");
        assert_eq!(at_world(&img, 0.0, 1.9)[3], 0, "empty sky");
        let rope_px = at_world(&img, 0.0, 0.95);
        assert!(rope_px[3] == 255 && rope_px[0] > 140, "the rope's sagging middle: {rope_px:?}");
        if let Ok(dir) = std::env::var("ZESTENGINE_SHAPE_PREVIEW") {
            std::fs::write(std::path::Path::new(&dir).join("sprite-shape.rgba"), &img).unwrap();
        }
        rig.destroy();
        for (s, t) in [(s0, t0), (s1, t1), (s2, t2), (s3, t3)] {
            unsafe {
                g.device.destroy_sampler(s, None);
            }
            free_tex_raw(&g, t);
        }
    }
}

#[test]
fn behind_layers_draw_before_the_cut_and_the_rest_after() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    // Red on layer -1 (behind), green on layer 1 (in front), side by side so each
    // overlaps the blue band (rows 24..40) but not the other.
    let tall = |mut s: SpriteSource| {
        s.scale = [1.0, 3.0];
        s
    };
    input.sprites = vec![tall(rig.sprite([-1.0, 0.0], RED, -1, 0)), tall(rig.sprite([1.0, 0.0], GREEN, 1, 0))];
    let img = rig.render_with_behind(&mut input, true);
    // Red sprite (left, x 8..24): the blue stand-in covers its middle rows, the rest is red.
    assert_eq!(px(&img, 16, 32), [0, 0, 255, 255], "behind sprite is under the 3D transparent");
    assert_eq!(px(&img, 16, 20), [255, 0, 0, 255], "and visible outside it");
    // Green sprite (right, x 40..56): drawn after the stand-in, so on top of it.
    assert_eq!(px(&img, 48, 32), [0, 255, 0, 255], "front sprite is over the 3D transparent");
    assert_eq!(px(&img, 32, 32), [0, 0, 255, 255], "blue band between the sprites");
    // Without the boundary everything is one part drawn last (over the band).
    let mut all = SpriteFrameInput { sprites: vec![tall(rig.sprite([-1.0, 0.0], RED, -1, 0)), tall(rig.sprite([1.0, 0.0], GREEN, 1, 0))], ..Default::default() };
    let img = rig.render_with_behind(&mut all, false);
    assert_eq!(px(&img, 16, 32), [255, 0, 0, 255], "no boundary: the red sprite is over the band too");
    assert_eq!(px(&img, 48, 32), [0, 255, 0, 255]);
    rig.destroy();
}

/// A top-down camera over the XZ plane: world +X to the right, world -Z up the image, so +Z (cell y)
/// runs down the picture; 16 px per unit like the XY rig.
fn top_down() -> Mat4 {
    use glam::Vec3;
    Mat4::orthographic_rh(-HALF, HALF, -HALF, HALF, -10.0, 10.0) * Mat4::look_at_rh(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, Vec3::NEG_Z)
}

/// An XZ sprite at world (x, height, z).
fn xz_sprite(rig: &Rig, x: f32, z: f32, height: f32, color: [f32; 4]) -> SpriteSource {
    let mut s = rig.sprite([x, z], color, 0, 0);
    s.position = [x, z, height];
    s.sort_point = [x, height, z];
    s.material = crate::sprite2d::SPRITE_MATERIAL_XZ | 1;
    s
}

#[test]
fn xz_sprites_lie_flat_in_the_ground_plane_and_obey_the_depth_test() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    rig.set_view_proj(top_down());
    let mut input = SpriteFrameInput::default();
    input.config.axis = SortAxis { origin: [0.0; 3], axis: [0.0, -1.0, 0.0] };
    // A unit tile at world (x 0.5, z 0.5): right of centre, and (z grows down the picture) below it.
    input.sprites = vec![xz_sprite(&rig, 0.5, 0.5, 0.0, RED)];
    let img = rig.render_with_axis(&mut input, [0.0, -1.0, 0.0]);
    assert_eq!(px(&img, 40, 40), [255, 0, 0, 255], "x 0.5 is 8 px right of the middle, z 0.5 is 8 px below it");
    assert_eq!(px(&img, 24, 40)[3], 0, "nothing on the other side in x");
    assert_eq!(px(&img, 40, 24)[3], 0, "nor above in the picture (negative z)");
    let (x0, y0, x1, y1, n) = coverage(&img);
    assert_eq!((x1 - x0, y1 - y0), (16, 16), "a unit tile is a 16 px square seen from above");
    assert!(n >= 250);
    // Rotated 90 degrees in its plane: a 2 x 1 rectangle whose long side lay along x now lies along z.
    let mut long = xz_sprite(&rig, 0.0, 0.0, 0.0, GREEN);
    long.scale = [2.0, 1.0];
    input.sprites = vec![long];
    let (x0, y0, x1, y1, _) = coverage(&rig.render_with_axis(&mut input, [0.0, -1.0, 0.0]));
    assert_eq!((x1 - x0, y1 - y0), (32, 16));
    long.rotation = std::f32::consts::FRAC_PI_2;
    input.sprites = vec![long];
    let (x0, y0, x1, y1, _) = coverage(&rig.render_with_axis(&mut input, [0.0, -1.0, 0.0]));
    assert_eq!((x1 - x0, y1 - y0), (16, 32), "the rotation turns the quad in the XZ plane");
    // Depth test: geometry in front (depth 0) of the right half hides the tile there, not on the left.
    let mut wide = xz_sprite(&rig, 0.0, 0.0, 0.0, RED);
    wide.scale = [3.0, 1.0];
    input.sprites = vec![wide];
    rig.occlude = Some(vk::Rect2D { offset: vk::Offset2D { x: 32, y: 0 }, extent: vk::Extent2D { width: 32, height: 64 } });
    let img = rig.render_with_axis(&mut input, [0.0, -1.0, 0.0]);
    assert_eq!(px(&img, 20, 32), [255, 0, 0, 255], "left of the occluder: visible");
    assert_eq!(px(&img, 44, 32)[3], 0, "behind 'geometry': hidden by the depth test");
    rig.occlude = None;
    // An overlay (ordinary) sprite ignores depth completely: it draws even over the occluder.
    let mut flat = rig.sprite([0.0, 0.0], GREEN, 0, 0);
    flat.scale = [3.0, 1.0];
    rig.set_view_proj(Mat4::orthographic_rh(-HALF, HALF, -HALF, HALF, -10.0, 10.0));
    rig.occlude = Some(vk::Rect2D { offset: vk::Offset2D { x: 32, y: 0 }, extent: vk::Extent2D { width: 32, height: 64 } });
    input.sprites = vec![flat];
    let img = rig.render(&mut input);
    assert_eq!(px(&img, 44, 32), [0, 255, 0, 255], "overlay sprites are not depth tested");
    rig.destroy();
}

#[test]
fn additive_and_multiply_materials_blend_as_documented() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut input = SpriteFrameInput::default();
    let grey = [0.5, 0.5, 0.5, 1.0];
    let mut bg = rig.sprite([0.0, 0.0], grey, 0, 0);
    bg.scale = [4.0, 4.0];
    let over = |rig: &Rig, material: u32, color: [f32; 4]| {
        let mut s = rig.sprite([0.0, 0.0], color, 0, 1);
        s.material = material;
        s
    };
    // Additive (material 3): grey + half red = (1.0, 0.5, 0.5) -> clamps at 255.
    input.sprites = vec![bg, over(&rig, 3, [0.5, 0.0, 0.0, 1.0])];
    let p = px(&rig.render(&mut input), 32, 32);
    assert!(p[0] >= 250 && (p[1] as i32 - 128).abs() <= 2 && (p[2] as i32 - 128).abs() <= 2, "{p:?}");
    // Additive with alpha 0.5 adds half the tint: grey + 0.25 red.
    input.sprites = vec![bg, over(&rig, 3, [0.5, 0.0, 0.0, 0.5])];
    let p = px(&rig.render(&mut input), 32, 32);
    assert!((p[0] as i32 - 192).abs() <= 3 && (p[1] as i32 - 128).abs() <= 2, "{p:?}");
    // Multiply (material 4): grey * (1.0, 0.5, 0.0) = (0.5, 0.25, 0).
    input.sprites = vec![bg, over(&rig, 4, [1.0, 0.5, 0.0, 1.0])];
    let p = px(&rig.render(&mut input), 32, 32);
    assert!((p[0] as i32 - 128).abs() <= 2 && (p[1] as i32 - 64).abs() <= 2 && p[2] <= 2, "{p:?}");
    // Multiply fades with alpha: alpha 0 leaves the background alone (the sprite is discarded).
    input.sprites = vec![bg, over(&rig, 4, [0.0, 0.0, 0.0, 0.0])];
    let p = px(&rig.render(&mut input), 32, 32);
    assert!((p[0] as i32 - 128).abs() <= 2, "{p:?}");
    rig.destroy();
}
