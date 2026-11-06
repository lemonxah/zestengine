//! Stencil on a real device (UNITY-2 / 2D-3): per-material stencil state through
//! the engine's own standard pipelines over a `D32_SFLOAT_S8_UINT` depth-stencil
//! attachment, with colour probes. Each case is a list of flat quads drawn in
//! order (writers first, readers after, exactly the render-queue rule); the
//! result is classified per pixel as black / red / green / blue.
//!
//! Also here: the sprite pass with `SpriteMask` (inside, outside, alpha cutoff,
//! sorting range) and the check that sprite masks and 3D stencil do not leak
//! into each other, plus the depth-stencil vs depth-only cost benchmark
//! (`--ignored`). Run with
//! `ZESTENGINE_TEST_GPU=1 cargo test -p zestengine-render --lib gpu_tests::stencil -- --test-threads=1`;
//! `ZESTENGINE_STENCIL_SHEET=<file.png>` also saves the contact sheet of the cases.

use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::shadow::{dummy_image, vert};
use super::{COLOR_OUT, Img, NONE, gpu, read_image, transition};
use crate::alloc::{self, Buffer};
use crate::cluster_headless::HeadlessGpu;
use crate::pipeline::{InstanceData, PipelineCache, PipelineKey};
use crate::sprite2d::{BuildConfig, MaskMode, MaskRange, MaskSource, MeshRange, SortAxis, SpriteMeshPool, SpriteSource};
use crate::sprite_gpu::{SpriteBlend, SpriteFrameInput, SpriteGpu, SpriteMaterialDesc};
use crate::texture::GpuTexture;
use crate::types::TextureData;
use crate::{
    AlphaMode, CameraData, FrameUbo, MAX_LIGHTS, MaterialFeatures, PushData, StencilCompare as Cmp, StencilFace, StencilOp as Op, StencilState, Vertex,
    frame_ubo, plan_shadows,
};

const W: u32 = 64;

/// Which way a quad faces the camera.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Facing {
    /// Counter-clockwise seen from the camera: a front face.
    Front,
    /// Clockwise seen from the camera: a back face.
    Back,
}

/// A flat quad, in pixel coordinates of the W x W target.
#[derive(Clone, Copy)]
struct Quad {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    /// View-space distance in front of the camera (smaller = nearer).
    dist: f32,
    color: [f32; 4],
    stencil: StencilState,
    facing: Facing,
}

fn quad(x0: u32, y0: u32, x1: u32, y1: u32, color: [f32; 4], stencil: StencilState) -> Quad {
    Quad { x0, y0, x1, y1, dist: 5.0, color, stencil, facing: Facing::Front }
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

fn face(compare: Cmp, pass_op: Op) -> StencilFace {
    StencilFace { compare, pass_op, fail_op: Op::Keep, zfail_op: Op::Keep }
}

/// A pure stencil writer: `Always`, `op` on pass, no colour.
fn writer(reference: u8, op: Op) -> StencilState {
    StencilState { color_mask: 0, ..StencilState::both(face(Cmp::Always, op), reference, 255, 255) }
}

/// A reader: draws colour where `compare` of `reference` against the stored value passes.
fn reader(reference: u8, compare: Cmp) -> StencilState {
    StencilState::both(face(compare, Op::Keep), reference, 255, 0)
}

/// Pixel classes of the probe colours.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum C {
    Black,
    Red,
    Green,
    Blue,
}

fn classify(p: [u8; 4]) -> C {
    let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
    let m = r.max(g).max(b);
    if m < 40 {
        C::Black
    } else if r == m && r > g + 40 && r > b + 40 {
        C::Red
    } else if g == m && g > r + 40 && g > b + 40 {
        C::Green
    } else if b == m && b > r + 40 && b > g + 40 {
        C::Blue
    } else {
        panic!("unclassifiable pixel {p:?}")
    }
}

struct Mesh {
    vbuf: Buffer,
    ibuf: Buffer,
}

/// The standard pipelines, a valid frame set / material set, a unit quad (both windings) and a sprite pass.
struct Rig<'a> {
    g: &'a HeadlessGpu,
    /// Square target size in pixels (the cases use 64).
    w: u32,
    /// Depth-only passes (the renderer's prepass variant: no colour attachment, no fragment work).
    depth_only: bool,
    alloc: Allocator,
    cache: PipelineCache,
    sets: Vec<vk::DescriptorSet>,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    dummies: Vec<(vk::Image, [vk::ImageView; 3], gpu_allocator::vulkan::Allocation)>,
    shadow: Img,
    shadow_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    sh_buf: Buffer,
    fx_buf: Buffer,
    ubo: Buffer,
    front: Mesh,
    back: Mesh,
}

fn fan(facing: Facing) -> (Vec<Vertex>, Vec<u32>) {
    let n = Vec3::Z;
    let v = vec![vert(Vec3::new(-0.5, -0.5, 0.0), n), vert(Vec3::new(0.5, -0.5, 0.0), n), vert(Vec3::new(0.5, 0.5, 0.0), n), vert(Vec3::new(-0.5, 0.5, 0.0), n)];
    let i = match facing {
        Facing::Front => vec![0, 1, 2, 0, 2, 3],
        Facing::Back => vec![0, 2, 1, 0, 3, 2],
    };
    (v, i)
}

/// Orthographic camera on the XY plane looking down -Z; world x/y in -1..1 fill the target.
fn camera() -> CameraData {
    CameraData { view: Mat4::IDENTITY, proj: Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 20.0), position: Vec3::ZERO }
}

impl<'a> Rig<'a> {
    fn new(g: &'a HeadlessGpu, depth: vk::Format) -> Self {
        Self::sized(g, depth, W)
    }

    fn sized(g: &'a HeadlessGpu, depth: vk::Format, size: u32) -> Self {
        let mut alloc = g.allocator().unwrap();
        let device = &g.device;
        let cache = PipelineCache::new(device, vk::Format::R8G8B8A8_UNORM, depth, vk::Format::A2B10G10R10_UNORM_PACK32).unwrap();
        let sampler = unsafe { device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR), None).unwrap() };
        let shadow = Img::new(g, &mut alloc, 4, 4, 1, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED, vk::SampleCountFlags::TYPE_1).unwrap();
        let shadow_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default().image(shadow.image).view_type(vk::ImageViewType::TYPE_2D_ARRAY).format(vk::Format::D32_SFLOAT).subresource_range(
                        vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(1),
                    ),
                    None,
                )
                .unwrap()
        };
        alloc::one_time_submit(device, g.pool, g.queue, |cb| {
            transition(g, cb, &shadow, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL, NONE, (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ));
        })
        .unwrap();
        let shadow_sampler = unsafe {
            device
                .create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR).compare_enable(true).compare_op(vk::CompareOp::LESS_OR_EQUAL), None)
                .unwrap()
        };
        let black = dummy_image(g, &mut alloc, 6, true, [0.0; 4]).unwrap();
        let white = dummy_image(g, &mut alloc, 1, false, [1.0; 4]).unwrap();
        let sh_buf = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "sh").unwrap();
        let mut fx_buf = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "fx").unwrap();
        fx_buf.write(0, &[0u8; 4096]);
        let ubo = Buffer::new(device, &mut alloc, std::mem::size_of::<FrameUbo>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "ubo").unwrap();
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&[
                        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(4),
                        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(64),
                        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4),
                    ]),
                    None,
                )
                .unwrap()
        };
        let layouts = [cache.set0_layout, cache.set1_layout];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).unwrap() };
        let mesh = |alloc: &mut Allocator, (v, i): (Vec<Vertex>, Vec<u32>)| Mesh {
            vbuf: alloc::upload_buffer(&g.device, alloc, g.pool, g.queue, bytemuck::cast_slice(&v), vk::BufferUsageFlags::VERTEX_BUFFER, "stencil-test vertices").unwrap(),
            ibuf: alloc::upload_buffer(&g.device, alloc, g.pool, g.queue, bytemuck::cast_slice(&i), vk::BufferUsageFlags::INDEX_BUFFER, "stencil-test indices").unwrap(),
        };
        let front = mesh(&mut alloc, fan(Facing::Front));
        let back = mesh(&mut alloc, fan(Facing::Back));
        let mut me = Self {
            g,
            w: size,
            depth_only: false,
            alloc,
            cache,
            sets,
            pool,
            sampler,
            dummies: vec![black, white],
            shadow,
            shadow_view,
            shadow_sampler,
            sh_buf,
            fx_buf,
            ubo,
            front,
            back,
        };
        me.write_sets();
        me
    }

    /// Frame UBO (ambient light only: a quad's colour is its albedo) and every descriptor valid.
    fn write_sets(&mut self) {
        let device = &self.g.device;
        let cam = camera();
        let mut fi = crate::cascade_tests::input(cam);
        fi.light = crate::LightData { ambient: [1.0; 3], env_intensity: 0.0, intensity: 0.0, ..Default::default() };
        let mut gl = [crate::GpuLight::default(); MAX_LIGHTS];
        let (_, shadow_vp, splits) = plan_shadows(&fi, &mut gl, 0, crate::CascadeCfg::resolve(&fi, 4, 2048));
        let vp = cam.proj * cam.view;
        let mut ubo = frame_ubo(&cam, &fi, gl, 0, shadow_vp, splits, &[], [0.5, 0.5], [W as f32, W as f32], [W as f32, W as f32], vp.to_cols_array_2d());
        ubo.debug[3] = 1.0; // linear output, no tonemap
        self.ubo.write(0, bytemuck::bytes_of(&ubo));
        let black = &self.dummies[0].1;
        let white = &self.dummies[1].1;
        let img = |view, sampler, layout| [vk::DescriptorImageInfo::default().image_view(view).sampler(sampler).image_layout(layout)];
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let shadow_info = img(self.shadow_view, self.shadow_sampler, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        let black_arr = img(black[1], self.sampler, ro);
        let black_2d = img(black[0], self.sampler, ro);
        let black_cube = img(black[2], self.sampler, ro);
        let water: Vec<vk::DescriptorImageInfo> = (0..crate::MAX_WATER_BODIES).map(|_| img(black[0], self.sampler, ro)[0]).collect();
        let ubo_info = [vk::DescriptorBufferInfo::default().buffer(self.ubo.handle).range(vk::WHOLE_SIZE)];
        let sh_info = [vk::DescriptorBufferInfo::default().buffer(self.sh_buf.handle).range(vk::WHOLE_SIZE)];
        let fx_info = [vk::DescriptorBufferInfo::default().buffer(self.fx_buf.handle).range(vk::WHOLE_SIZE)];
        let cs = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        let sets = &self.sets;
        let w = |set, binding, ty| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty);
        let mut writes = vec![
            w(sets[0], 0, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
            w(sets[0], 1, cs).image_info(&shadow_info),
            w(sets[0], 2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&sh_info),
            // No water in this fixture, but the shared layout still requires its buffer.
            w(sets[0], 14, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&sh_info),
            w(sets[0], 3, cs).image_info(&black_arr),
            w(sets[0], 4, cs).image_info(&black_2d),
            w(sets[0], 5, cs).image_info(&black_2d),
            w(sets[0], 6, cs).image_info(&black_2d),
            w(sets[0], 7, cs).image_info(&black_cube),
            w(sets[0], 8, cs).image_info(&black_arr),
            w(sets[0], 9, cs).image_info(&black_cube),
            w(sets[0], 11, cs).image_info(&black_cube),
            w(sets[0], 12, cs).image_info(&black_cube),
            w(sets[0], 10, cs).image_info(&water),
            w(sets[1], 4, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&fx_info),
        ];
        let white_info = img(white[0], self.sampler, ro);
        for b in crate::TEXTURE_BINDINGS {
            writes.push(w(sets[1], b, cs).image_info(&white_info));
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Per-quad instance streams and pipelines (one per distinct state).
    fn prepare(&mut self, quads: &[Quad]) -> (Vec<vk::Pipeline>, Vec<Buffer>) {
        let (g, w) = (self.g, self.w);
        let device = &g.device;
        let mut inst_bufs = Vec::new();
        for q in quads {
            let (cx, cy) = ((q.x0 + q.x1) as f32 / w as f32 - 1.0, 1.0 - (q.y0 + q.y1) as f32 / w as f32);
            let (sx, sy) = ((q.x1 - q.x0) as f32 * 2.0 / w as f32, (q.y1 - q.y0) as f32 * 2.0 / w as f32);
            let model = Mat4::from_translation(Vec3::new(cx, cy, -q.dist)) * Mat4::from_scale(Vec3::new(sx, sy, 1.0));
            let inst = InstanceData { model: model.to_cols_array_2d(), color: q.color, param: [0.0; 4] };
            inst_bufs.push(alloc::upload_buffer(device, &mut self.alloc, g.pool, g.queue, bytemuck::bytes_of(&inst), vk::BufferUsageFlags::VERTEX_BUFFER, "stencil-test instance").unwrap());
        }
        self.cache.set_target_views(false, 1);
        let pipes = quads
            .iter()
            .map(|q| {
                let mut key = PipelineKey::from_features(&MaterialFeatures { alpha_mode: AlphaMode::Opaque, double_sided: true, stencil: q.stencil, ..Default::default() });
                key.per_instance = false;
                key.prepass = self.depth_only;
                self.cache.get(device, key).unwrap()
            })
            .collect();
        (pipes, inst_bufs)
    }

    /// Record one pass of `quads` (and the sprite pass) into a begun command buffer: begin, draws, end.
    #[allow(clippy::too_many_arguments)]
    fn record_pass(
        &self,
        cb: vk::CommandBuffer,
        color: &Img,
        depth: &Img,
        quads: &[Quad],
        prepared: &(Vec<vk::Pipeline>, Vec<Buffer>),
        sprites: &mut Option<(&mut SpriteGpu, &mut SpriteFrameInput, &mut SpriteMeshPool)>,
    ) {
        let (g, w) = (self.g, self.w);
        let device = &g.device;
        let stencil = crate::sprite_gpu::has_stencil(depth.format);
        let attach = if stencil { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL };
        let layout = self.cache.layout;
        let both = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let pass = self.cache.pass_formats();
        unsafe {
            let catt = [vk::RenderingAttachmentInfo::default()
                .image_view(color.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            let datt = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(attach)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
            let satt = crate::stencil_attachment_for(&datt, stencil);
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: w } };
            let colors: &[vk::RenderingAttachmentInfo] = if self.depth_only { &[] } else { &catt };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(colors).depth_attachment(&datt).stencil_attachment(&satt));
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: w as f32, width: w as f32, height: -(w as f32), min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cb, 0, &[area]);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[self.sets[0], self.sets[1]], &[]);
            for ((q, pipe), inst) in quads.iter().zip(&prepared.0).zip(&prepared.1) {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, *pipe);
                let mut push: PushData = bytemuck::Zeroable::zeroed();
                push.model[0] = [f32::from_bits(u32::MAX); 4];
                push.base_color = q.color;
                push.params0 = [0.0, 0.8, 0.0, 0.0];
                push.params1 = [0.5, 1.0, 1.0, -1.0];
                device.cmd_push_constants(cb, layout, both, 0, bytemuck::bytes_of(&push));
                let m = if q.facing == Facing::Front { &self.front } else { &self.back };
                device.cmd_bind_vertex_buffers(cb, 0, &[m.vbuf.handle], &[0]);
                device.cmd_bind_vertex_buffers(cb, 1, &[inst.handle], &[0]);
                device.cmd_bind_index_buffer(cb, m.ibuf.handle, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cb, 6, 1, 0, 0, 0);
            }
            if let Some((gpu, _, _)) = sprites.as_mut() {
                gpu.record(device, cb, 0, self.sets[0], pass, false, false, crate::sprite_gpu::SpritePart::All);
            }
            device.cmd_end_rendering(cb);
        }
    }

    fn targets(&mut self, depth_fmt: vk::Format, depth_extra: vk::ImageUsageFlags) -> (Img, Img) {
        let g = self.g;
        let w = self.w;
        let color = Img::new(g, &mut self.alloc, w, w, 1, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap();
        let depth = Img::new(g, &mut self.alloc, w, w, 1, depth_fmt, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | depth_extra, vk::SampleCountFlags::TYPE_1).unwrap();
        (color, depth)
    }

    /// Draw `quads` in order into a cleared colour + depth-stencil target (optionally followed by the
    /// sprite pass) and return the colour pixels and the stencil plane (empty without a stencil aspect).
    fn render(&mut self, quads: &[Quad], mut sprites: Option<(&mut SpriteGpu, &mut SpriteFrameInput, &mut SpriteMeshPool)>) -> (Vec<u8>, Vec<u8>) {
        let g = self.g;
        let device = &g.device;
        let depth_fmt = self.cache.pass_formats().depth;
        let (mut color, mut depth) = self.targets(depth_fmt, vk::ImageUsageFlags::TRANSFER_SRC);
        let stencil = crate::sprite_gpu::has_stencil(depth_fmt);
        let attach = if stencil { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL };
        let mut prepared = self.prepare(quads);
        if let Some((gpu, input, pool)) = sprites.as_mut() {
            input.config = BuildConfig { axis: SortAxis { origin: [0.0; 3], axis: [0.0, 0.0, -1.0] }, threads: 1, ..Default::default() };
            gpu.prepare(input, pool, 0);
        }
        let ds = (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
        alloc::one_time_submit(device, g.pool, g.queue, |cb| {
            transition(g, cb, &color, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
            transition(g, cb, &depth, vk::ImageLayout::UNDEFINED, attach, NONE, ds);
            self.record_pass(cb, &color, &depth, quads, &prepared, &mut sprites);
        })
        .unwrap();
        let pixels = read_image(g, &mut self.alloc, &color, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT, 4).unwrap();
        let stencil_plane = if stencil { read_stencil(g, &mut self.alloc, &depth, attach) } else { Vec::new() };
        for b in prepared.1.iter_mut() {
            b.destroy(device, &mut self.alloc);
        }
        color.destroy(g, &mut self.alloc);
        depth.destroy(g, &mut self.alloc);
        (pixels, stencil_plane)
    }

    /// GPU time of `reps` passes of `quads` in one submission (no readback), in milliseconds per pass.
    fn time_passes(&mut self, quads: &[Quad], reps: u32) -> f64 {
        let g = self.g;
        let device = &g.device;
        let depth_fmt = self.cache.pass_formats().depth;
        let (mut color, mut depth) = self.targets(depth_fmt, vk::ImageUsageFlags::empty());
        let stencil = crate::sprite_gpu::has_stencil(depth_fmt);
        let attach = if stencil { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL };
        let mut prepared = self.prepare(quads);
        let ds = (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
        let run = |n: u32| {
            let t0 = std::time::Instant::now();
            alloc::one_time_submit(device, g.pool, g.queue, |cb| {
                transition(g, cb, &color, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
                transition(g, cb, &depth, vk::ImageLayout::UNDEFINED, attach, NONE, ds);
                for _ in 0..n {
                    self.record_pass(cb, &color, &depth, quads, &prepared, &mut None);
                    // Back to the pass's first layouts (contents are cleared by the next pass anyway).
                }
            })
            .unwrap();
            t0.elapsed().as_secs_f64() * 1000.0
        };
        let _warm = run(2);
        let ms = run(reps);
        for b in prepared.1.iter_mut() {
            b.destroy(device, &mut self.alloc);
        }
        color.destroy(g, &mut self.alloc);
        depth.destroy(g, &mut self.alloc);
        ms / reps as f64
    }

    fn destroy(mut self) {
        let device = &self.g.device;
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_image_view(self.shadow_view, None);
            for (image, views, _) in &self.dummies {
                for v in views {
                    if *v != vk::ImageView::null() {
                        device.destroy_image_view(*v, None);
                    }
                }
                device.destroy_image(*image, None);
            }
        }
        for m in [&mut self.front, &mut self.back] {
            m.vbuf.destroy(device, &mut self.alloc);
            m.ibuf.destroy(device, &mut self.alloc);
        }
        self.sh_buf.destroy(device, &mut self.alloc);
        self.fx_buf.destroy(device, &mut self.alloc);
        self.ubo.destroy(device, &mut self.alloc);
        self.shadow.destroy(self.g, &mut self.alloc);
        self.cache.destroy(device);
    }
}

/// The stencil plane of a depth-stencil image that is in `layout` (left there), one byte per texel.
fn read_stencil(g: &HeadlessGpu, alloc: &mut Allocator, img: &Img, layout: vk::ImageLayout) -> Vec<u8> {
    let size = (img.w * img.h) as usize;
    let mut host = Buffer::new(&g.device, alloc, size as u64, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "stencil readback").unwrap();
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        transition(
            g,
            cb,
            img,
            layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            (vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE),
            (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ),
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::STENCIL).layer_count(1))
            .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 });
        g.device.cmd_copy_image_to_buffer(cb, img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[copy]);
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        g.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
    })
    .unwrap();
    let out = host.mapped().map(|m| m[..size].to_vec()).unwrap_or_default();
    host.destroy(&g.device, alloc);
    out
}

fn px(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let o = ((y * W + x) * 4) as usize;
    [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
}

// ---------------------------------------------------------------------------
// Material stencil cases
// ---------------------------------------------------------------------------

/// A named case: draws plus (x, y, expected class) probes.
struct Case {
    name: &'static str,
    quads: Vec<Quad>,
    probes: Vec<(u32, u32, C)>,
}

#[allow(clippy::vec_init_then_push)] // one push per case keeps the table readable
fn cases() -> Vec<Case> {
    let full = |color, st| quad(0, 0, W, W, color, st);
    let left = |color, st| quad(0, 0, W / 2, W, color, st);
    let mut v = Vec::new();
    // Write ref = 1 on the left half, then draw only where stencil == 1.
    v.push(Case {
        name: "equal",
        quads: vec![left(RED, writer(1, Op::Replace)), full(RED, reader(1, Cmp::Equal))],
        probes: vec![(8, 32, C::Red), (24, 8, C::Red), (40, 32, C::Black), (56, 56, C::Black)],
    });
    // ... and only where stencil != 1.
    v.push(Case {
        name: "not-equal",
        quads: vec![left(GREEN, writer(1, Op::Replace)), full(GREEN, reader(1, Cmp::NotEqual))],
        probes: vec![(8, 32, C::Black), (40, 32, C::Green), (56, 8, C::Green)],
    });
    // Comparison ops against a stored 2: Less (1 < 2 yes: ref is the left operand), Greater, LessEqual, Never.
    v.push(Case {
        name: "compare-ops",
        quads: vec![
            full(RED, writer(2, Op::Replace)),
            quad(0, 0, W / 4, W, RED, reader(3, Cmp::Greater)),
            quad(W / 4, 0, W / 2, W, GREEN, reader(2, Cmp::GreaterEqual)),
            quad(W / 2, 0, 3 * W / 4, W, BLUE, reader(1, Cmp::Greater)),
            quad(3 * W / 4, 0, W, W, RED, reader(2, Cmp::Never)),
        ],
        probes: vec![(8, 32, C::Red), (24, 32, C::Green), (40, 32, C::Black), (56, 32, C::Black)],
    });
    // Increment twice, then Equal 2 shows; decrement once, Equal 1 shows; DecrementClamp at 0 stays 0.
    v.push(Case {
        name: "incr-decr",
        quads: vec![
            left(RED, writer(0, Op::IncrementClamp)),
            left(RED, writer(0, Op::IncrementClamp)),
            quad(0, 0, W / 2, W / 2, GREEN, reader(2, Cmp::Equal)),
            quad(0, 0, W / 2, W, RED, writer(0, Op::DecrementClamp)),
            quad(0, W / 2, W / 2, W, BLUE, reader(1, Cmp::Equal)),
            // the right half: DecrementClamp on 0 stays 0, so Equal 0 passes there
            quad(W / 2, 0, W, W, RED, writer(0, Op::DecrementClamp)),
            quad(W / 2, 0, W, W, GREEN, reader(0, Cmp::Equal)),
        ],
        // top-left: 2 then decremented to 1 -> the Equal-2 reader drew before the decrement
        probes: vec![(8, 8, C::Green), (8, 56, C::Blue), (48, 32, C::Green)],
    });
    // Invert: 0 -> 255, shown by Equal 255; inverted twice -> 0.
    v.push(Case {
        name: "invert",
        quads: vec![
            left(RED, writer(0, Op::Invert)),
            quad(0, 0, W / 2, W / 2, RED, writer(0, Op::Invert)),
            full(BLUE, reader(255, Cmp::Equal)),
        ],
        probes: vec![(8, 8, C::Black), (8, 56, C::Blue), (48, 32, C::Black)],
    });
    // IncrementWrap from 255 wraps to 0; DecrementWrap from 0 wraps to 255.
    v.push(Case {
        name: "wrap",
        quads: vec![
            left(RED, writer(0, Op::Invert)),
            left(RED, writer(0, Op::IncrementWrap)),
            quad(W / 2, 0, W, W, RED, writer(0, Op::DecrementWrap)),
            full(GREEN, reader(0, Cmp::Equal)),
            quad(W / 2, 0, W, W, BLUE, reader(255, Cmp::Equal)),
        ],
        probes: vec![(8, 32, C::Green), (48, 32, C::Blue)],
    });
    // Z-fail: an occluder in front writes depth; a farther writer with zfail = Replace marks only behind it.
    let mut zwriter = writer(1, Op::Keep);
    zwriter.front.zfail_op = Op::Replace;
    zwriter.back.zfail_op = Op::Replace;
    let mut occluder = quad(W / 4, W / 4, 3 * W / 4, 3 * W / 4, RED, writer(0, Op::Keep));
    occluder.stencil.color_mask = 15;
    occluder.dist = 3.0;
    let mut far = quad(0, 0, W, W, RED, zwriter);
    far.dist = 8.0;
    v.push(Case {
        name: "zfail",
        quads: vec![occluder, far, full(BLUE, reader(1, Cmp::Equal))],
        // The reader at dist 5 is behind the occluder (3): where it is occluded the reader's depth fails; so draw it
        // nearer than the occluder: handled below by a nearer reader quad instead.
        probes: vec![],
    });
    // Fix up the zfail reader: it must be nearer than the occluder to be visible there.
    {
        let c = v.last_mut().unwrap();
        c.quads[2].dist = 2.0;
        c.probes = vec![(32, 32, C::Blue), (4, 4, C::Black), (60, 60, C::Black)];
    }
    // Fail op: Never + FailOp Replace writes everywhere the quad covers, colour off.
    let mut failw = StencilState { color_mask: 0, ..StencilState::both(StencilFace { compare: Cmp::Never, pass_op: Op::Keep, fail_op: Op::Replace, zfail_op: Op::Keep }, 7, 255, 255) };
    failw.reference = 7;
    v.push(Case {
        name: "fail-op",
        quads: vec![left(RED, failw), full(GREEN, reader(7, Cmp::Equal))],
        probes: vec![(8, 32, C::Green), (40, 32, C::Black)],
    });
    // Write mask: the writer sets ref 0xFF through mask 0x0F, stored value 0x0F.
    let mut wm = writer(0xFF, Op::Replace);
    wm.write_mask = 0x0F;
    let mut rd_full = reader(0xFF, Cmp::Equal);
    rd_full.read_mask = 0xFF;
    let mut rd_low = reader(0xFF, Cmp::Equal);
    rd_low.read_mask = 0x0F;
    let mut rd_exact = reader(0x0F, Cmp::Equal);
    rd_exact.read_mask = 0xFF;
    v.push(Case {
        name: "masks",
        quads: vec![
            full(RED, wm),
            // stored 0x0F: ref 0xFF with full read mask fails ...
            quad(0, 0, W / 3, W, RED, rd_full),
            // ... with read mask 0x0F both sides compare 0x0F: passes ...
            quad(W / 3, 0, 2 * W / 3, W, GREEN, rd_low),
            // ... and ref 0x0F against the stored 0x0F passes
            quad(2 * W / 3, 0, W, W, BLUE, rd_exact),
        ],
        probes: vec![(4, 32, C::Black), (32, 32, C::Green), (60, 32, C::Blue)],
    });
    // Front vs back: a front-only writer marks only the front-facing quad.
    let mut front_only = writer(1, Op::Keep);
    front_only.front.pass_op = Op::Replace;
    let mut back_only = writer(2, Op::Keep);
    back_only.back.pass_op = Op::Replace;
    let mut fq = quad(0, 0, W / 2, W, RED, front_only);
    fq.facing = Facing::Front;
    let mut bq = quad(W / 2, 0, W, W, RED, front_only);
    bq.facing = Facing::Back;
    let mut fq2 = quad(0, 0, W / 2, W, RED, back_only);
    fq2.facing = Facing::Front;
    let mut bq2 = quad(W / 2, 0, W, W, RED, back_only);
    bq2.facing = Facing::Back;
    v.push(Case {
        name: "front-back",
        // front-only writer (ref 1) over a front quad (left) and a back quad (right);
        // back-only writer (ref 2) over the same. Readers: Equal 1 -> green, Equal 2 -> blue.
        quads: vec![fq, bq, fq2, bq2, full(GREEN, reader(1, Cmp::Equal)), full(BLUE, reader(2, Cmp::Equal))],
        probes: vec![(16, 32, C::Green), (48, 32, C::Blue)],
    });
    // ZWrite off: the same writer leaves the depth alone, so the quad behind it draws.
    let mut wz = quad(0, 0, W / 2, W, RED, StencilState { depth_write: false, ..writer(1, Op::Replace) });
    wz.dist = 3.0;
    let mut behind_z = quad(0, 0, W, W, GREEN, StencilState::default());
    behind_z.dist = 6.0;
    v.push(Case { name: "zwrite-off", quads: vec![wz, behind_z, full(BLUE, reader(1, Cmp::Equal))], probes: vec![(8, 32, C::Blue), (48, 32, C::Green)] });
    // A pure-writer draws no colour but (ZWrite on) still writes depth: a later quad behind it fails depth.
    let mut wd = quad(0, 0, W / 2, W, RED, writer(1, Op::Replace));
    wd.dist = 3.0;
    let mut behind = quad(0, 0, W, W, GREEN, StencilState::default());
    behind.dist = 6.0;
    v.push(Case { name: "writer-depth", quads: vec![wd, behind], probes: vec![(8, 32, C::Black), (48, 32, C::Green)] });
    v
}

fn run_case(rig: &mut Rig, case: &Case) -> Vec<u8> {
    let (img, _) = rig.render(&case.quads, None);
    for &(x, y, want) in &case.probes {
        let got = classify(px(&img, x, y));
        assert_eq!(got, want, "case '{}' at ({x},{y}): {:?}", case.name, px(&img, x, y));
    }
    img
}

#[test]
fn material_stencil_cases_match_the_reference_semantics() {
    let Some(g) = gpu() else { return };
    let depth = g.depth_stencil_format();
    if !crate::sprite_gpu::has_stencil(depth) {
        eprintln!("[stencil] no stencil depth format on this device ({depth:?}): skipped");
        return;
    }
    let mut rig = Rig::new(&g, depth);
    let mut tiles = Vec::new();
    for case in cases() {
        let img = run_case(&mut rig, &case);
        tiles.push((case.name, img));
    }
    if let Ok(path) = std::env::var("ZESTENGINE_STENCIL_SHEET") {
        save_sheet(&path, &tiles);
    }
    rig.destroy();
}

/// A material with the default state must leave the stencil untouched and draw like before.
#[test]
fn default_stencil_state_is_a_no_op_and_the_plain_depth_path_still_draws() {
    let Some(g) = gpu() else { return };
    // Plain D32: stencil is dropped from the pipeline (no stencil aspect) and the quads just draw.
    let mut rig = Rig::new(&g, vk::Format::D32_SFLOAT);
    let q = vec![quad(0, 0, W / 2, W, RED, reader(1, Cmp::Equal)), quad(W / 2, 0, W, W, GREEN, StencilState::default())];
    let (img, plane) = rig.render(&q, None);
    assert!(plane.is_empty());
    // Without a stencil aspect the reader's test is ignored: it draws everywhere it covers.
    assert_eq!(classify(px(&img, 8, 32)), C::Red);
    assert_eq!(classify(px(&img, 48, 32)), C::Green);
    rig.destroy();
    // D32S8: an inactive state leaves every stencil texel 0.
    let depth = g.depth_stencil_format();
    if crate::sprite_gpu::has_stencil(depth) {
        let mut rig = Rig::new(&g, depth);
        let (img, plane) = rig.render(&[quad(0, 0, W, W, GREEN, StencilState::default())], None);
        assert_eq!(classify(px(&img, 32, 32)), C::Green);
        assert!(plane.iter().all(|&s| s == 0), "an inactive state must not write the stencil");
        rig.destroy();
    }
}

// ---------------------------------------------------------------------------
// Contact sheet (a small PNG writer: stored deflate blocks, no extra crate)
// ---------------------------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

pub(super) fn png_rgba(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity((w * 4 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * w as usize * 4..(y + 1) * w as usize * 4]);
    }
    let mut z = vec![0x78, 0x01];
    for (i, chunk) in raw.chunks(65_535).enumerate() {
        let last = (i + 1) * 65_535 >= raw.len();
        z.push(last as u8);
        z.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        z.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |tag: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = tag.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

/// Tiles of W x W, 4 per row, upscaled 4x with a grey gutter.
fn save_sheet(path: &str, tiles: &[(&'static str, Vec<u8>)]) {
    const SCALE: u32 = 4;
    const GAP: u32 = 8;
    let cols = 4u32;
    let rows = (tiles.len() as u32).div_ceil(cols);
    let tw = W * SCALE;
    let (sw, sh) = (cols * (tw + GAP) + GAP, rows * (tw + GAP) + GAP);
    let mut sheet = vec![0u8; (sw * sh * 4) as usize];
    for p in sheet.as_chunks_mut::<4>().0 {
        p.copy_from_slice(&[60, 60, 60, 255]);
    }
    for (i, (_, img)) in tiles.iter().enumerate() {
        let (cx, cy) = (GAP + (i as u32 % cols) * (tw + GAP), GAP + (i as u32 / cols) * (tw + GAP));
        for y in 0..tw {
            for x in 0..tw {
                let s = px(img, x / SCALE, y / SCALE);
                let o = (((cy + y) * sw + cx + x) * 4) as usize;
                sheet[o..o + 4].copy_from_slice(&[s[0], s[1], s[2], 255]);
            }
        }
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, png_rgba(sw, sh, &sheet)).expect("writing the contact sheet");
    eprintln!("[stencil] contact sheet: {path} ({} tiles: {})", tiles.len(), tiles.iter().map(|t| t.0).collect::<Vec<_>>().join(", "));
}

// ---------------------------------------------------------------------------
// Sprite masks (2D-3)
// ---------------------------------------------------------------------------

/// Sprite pass fixtures on top of the same rig: a white 4x4 texture, a half-alpha texture, a unit quad mesh.
struct SpriteBits {
    gpu: SpriteGpu,
    pool: SpriteMeshPool,
    quad: MeshRange,
    white: GpuTexture,
    ramp: GpuTexture,
    sampler: vk::Sampler,
}

impl SpriteBits {
    fn new(rig: &mut Rig) -> Self {
        let g = rig.g;
        let mut gpu = SpriteGpu::new(&g.device, &mut rig.alloc, rig.cache.set0_layout).unwrap();
        let upload = |rig: &mut Rig, pixels: Vec<u8>| GpuTexture::upload(&g.device, &mut rig.alloc, g.pool, g.queue, &TextureData { width: 4, height: 4, pixels, srgb: false, hdr: false }).unwrap();
        let white = upload(rig, vec![255; 4 * 4 * 4]);
        // Alpha ramp: column 0..3 = 0, 85, 170, 255 (colour white).
        let mut px4 = Vec::new();
        for _ in 0..4 {
            for c in 0..4u8 {
                px4.extend_from_slice(&[255, 255, 255, c * 85]);
            }
        }
        let ramp = upload(rig, px4);
        let sampler = unsafe {
            g.device
                .create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST).address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE), None)
                .unwrap()
        };
        assert_eq!(gpu.register_texture(&g.device, white.view, sampler, false).unwrap(), 0);
        assert_eq!(gpu.register_texture(&g.device, ramp.view, sampler, false).unwrap(), 1);
        gpu.register_material(SpriteMaterialDesc { blend: SpriteBlend::Alpha, alpha_cutoff: 0.0 });
        let mut pool = SpriteMeshPool::new(1024, 4096);
        let quad = pool.add(&[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], &[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]], &[0, 1, 2, 2, 1, 3]).unwrap();
        Self { gpu, pool, quad, white, ramp, sampler }
    }

    /// A tinted sprite of `size` world units at `pos` (target spans -1..1: 32 px per unit).
    fn sprite(&self, pos: [f32; 2], size: f32, color: [f32; 4], layer: i16, order: i16, mask: MaskMode) -> SpriteSource {
        SpriteSource {
            position: [pos[0], pos[1], -5.0],
            scale: [size, size],
            sort_point: [pos[0], pos[1], -5.0],
            mesh: self.quad,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            sprite_size: [1.0, 1.0],
            pivot: [0.5, 0.5],
            color,
            adaptive_threshold: 0.5,
            size: [1.0, 1.0],
            layer,
            order,
            mask,
            ..Default::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mask(&self, pos: [f32; 2], size: f32, texture: u32, alpha_cutoff: f32, layer: i16, order: i16, range: MaskRange) -> MaskSource {
        MaskSource { position: [pos[0], pos[1], -5.0], scale: [size, size], mesh: self.quad, uv_rect: [0.0, 0.0, 1.0, 1.0], texture, alpha_cutoff, layer, order, range, ..Default::default() }
    }

    fn destroy(mut self, rig: &mut Rig) {
        let g = rig.g;
        self.gpu.destroy(&g.device, &mut rig.alloc);
        unsafe {
            g.device.destroy_sampler(self.sampler, None);
            for t in [&self.white, &self.ramp] {
                g.device.destroy_image_view(t.view, None);
                g.device.destroy_image(t.image, None);
            }
        }
    }
}

fn sprite_scene(g: &HeadlessGpu, f: impl FnOnce(&mut Rig, &mut SpriteBits)) {
    let depth = g.depth_stencil_format();
    if !crate::sprite_gpu::has_stencil(depth) {
        eprintln!("[stencil] no stencil depth format on this device ({depth:?}): skipped");
        return;
    }
    let mut rig = Rig::new(g, depth);
    let mut bits = SpriteBits::new(&mut rig);
    f(&mut rig, &mut bits);
    bits.destroy(&mut rig);
    rig.destroy();
}

fn render_sprites(rig: &mut Rig, bits: &mut SpriteBits, quads: &[Quad], input: &mut SpriteFrameInput) -> (Vec<u8>, Vec<u8>) {
    rig.render(quads, Some((&mut bits.gpu, input, &mut bits.pool)))
}

#[test]
fn sprite_mask_shows_inside_and_outside() {
    let Some(g) = gpu() else { return };
    sprite_scene(&g, |rig, bits| {
        // A 1 x 1 mask in the middle (px 16..48); a big red sprite behind-covering the whole target.
        let mut input = SpriteFrameInput {
            masks: vec![bits.mask([0.0, 0.0], 1.0, 0, 0.0, 0, 0, MaskRange::Auto)],
            sprites: vec![bits.sprite([0.0, 0.0], 2.0, RED, 0, 1, MaskMode::VisibleInside)],
            ..Default::default()
        };
        let (img, plane) = render_sprites(rig, bits, &[], &mut input);
        assert_eq!(classify(px(&img, 32, 32)), C::Red, "inside the mask");
        assert_eq!(classify(px(&img, 8, 8)), C::Black, "outside the mask is hidden");
        assert_eq!(classify(px(&img, 56, 40)), C::Black);
        assert!(plane.iter().all(|&s| s & 0x80 == 0), "the pass resets its stencil bit when it ends");
        // Visible outside: the mirror image.
        input.sprites = vec![bits.sprite([0.0, 0.0], 2.0, GREEN, 0, 1, MaskMode::VisibleOutside)];
        let (img, _) = render_sprites(rig, bits, &[], &mut input);
        assert_eq!(classify(px(&img, 32, 32)), C::Black, "inside the mask is hidden");
        assert_eq!(classify(px(&img, 8, 8)), C::Green);
        assert_eq!(classify(px(&img, 56, 56)), C::Green);
        // An unmasked sprite ignores the mask.
        input.sprites = vec![bits.sprite([0.0, 0.0], 2.0, BLUE, 0, 1, MaskMode::None)];
        let (img, _) = render_sprites(rig, bits, &[], &mut input);
        assert_eq!((classify(px(&img, 32, 32)), classify(px(&img, 8, 8))), (C::Blue, C::Blue));
    });
}

#[test]
fn sprite_mask_alpha_cutoff_keeps_only_opaque_texels() {
    let Some(g) = gpu() else { return };
    sprite_scene(&g, |rig, bits| {
        // The ramp texture has alpha 0, 85, 170, 255 per column over a 2 x 2 unit mask (px 0..64: 16 px per column).
        let mut input = SpriteFrameInput::default();
        let sprites = vec![bits.sprite([0.0, 0.0], 2.0, RED, 0, 1, MaskMode::VisibleInside)];
        for (cutoff, shown) in [(0.0f32, [false, true, true, true]), (0.5, [false, false, true, true]), (0.9, [false, false, false, true])] {
            input.masks = vec![bits.mask([0.0, 0.0], 2.0, 1, cutoff, 0, 0, MaskRange::Auto)];
            input.sprites = sprites.clone();
            let (img, _) = render_sprites(rig, bits, &[], &mut input);
            for (col, want) in shown.iter().enumerate() {
                let got = classify(px(&img, col as u32 * 16 + 8, 32));
                // cutoff 0 keeps alpha > 0 only through the shader's `discard` on a == 0 texels
                assert_eq!(got == C::Red, *want, "cutoff {cutoff}, column {col}: {:?}", px(&img, col as u32 * 16 + 8, 32));
            }
        }
    });
}

#[test]
fn sprite_mask_sorting_range_limits_which_sprites_are_masked() {
    let Some(g) = gpu() else { return };
    sprite_scene(&g, |rig, bits| {
        // Mask range: layer 0, orders 0..=5. A masked sprite at order 3 is inside the range; one at order 9 is not,
        // so it sees an empty stencil (VisibleInside hides it everywhere, VisibleOutside shows it everywhere).
        let mut input = SpriteFrameInput {
            masks: vec![bits.mask([0.0, -0.5], 1.0, 0, 0.0, 0, 0, MaskRange::Custom { back: (0, 0), front: (0, 5) })],
            sprites: vec![
            bits.sprite([-0.5, 0.5], 0.8, RED, 0, 3, MaskMode::VisibleInside),    // in range, outside the mask shape: hidden
            bits.sprite([0.0, -0.5], 0.4, GREEN, 0, 3, MaskMode::VisibleInside),  // in range, inside the mask: shown
            bits.sprite([0.5, 0.5], 0.8, BLUE, 0, 9, MaskMode::VisibleOutside),   // out of range: stencil empty, shown whole
            bits.sprite([0.5, -0.5], 0.4, RED, 0, 9, MaskMode::VisibleInside),    // out of range: hidden even over the mask area? no mask here
            ],
            ..Default::default()
        };
        let (img, plane) = render_sprites(rig, bits, &[], &mut input);
        let at = |w: [f32; 2]| px(&img, ((w[0] + 1.0) * 32.0) as u32, ((1.0 - w[1]) * 32.0) as u32);
        assert_eq!(classify(at([-0.5, 0.5])), C::Black, "in range, outside the mask: hidden");
        assert_eq!(classify(at([0.0, -0.5])), C::Green, "in range, inside the mask: shown");
        assert_eq!(classify(at([0.5, 0.5])), C::Blue, "out of range: VisibleOutside shows");
        assert_eq!(classify(at([0.5, -0.5])), C::Black, "out of range: VisibleInside sees no mask");
        assert!(plane.iter().all(|&s| s & 0x80 == 0));
    });
}

#[test]
fn sprite_masks_and_3d_stencil_do_not_leak_into_each_other() {
    let Some(g) = gpu() else { return };
    sprite_scene(&g, |rig, bits| {
        // 3D: a writer sets stencil = 5 on the left half (colour off) and a reader (Equal 5) paints green there.
        // The sprite pass then masks a red sprite with a mask over the RIGHT half only.
        let q3d = [quad(0, 0, W / 2, W, RED, writer(5, Op::Replace)), quad(0, 0, W / 2, W, GREEN, reader(5, Cmp::Equal))];
        let mut input = SpriteFrameInput {
            masks: vec![bits.mask([0.5, 0.0], 1.0, 0, 0.0, 0, 0, MaskRange::Auto)],
            sprites: vec![bits.sprite([0.0, 0.0], 2.0, RED, 0, 1, MaskMode::VisibleInside)],
            ..Default::default()
        };
        let (img, plane) = render_sprites(rig, bits, &q3d, &mut input);
        // The sprite is visible only in the mask (right half, px 32..64 x 16..48).
        assert_eq!(classify(px(&img, 48, 32)), C::Red, "sprite inside its mask");
        // The 3D reader's green survives on the left; the sprite did not appear there although the 3D stencil is non-zero.
        assert_eq!(classify(px(&img, 16, 32)), C::Green, "3D stencil reader intact; the sprite is not tested against 3D stencil values");
        // The stencil's low bits keep the 3D value 5 on the left; bit 7 is clear everywhere afterwards.
        assert_eq!(plane[(32 * W + 16) as usize], 5, "3D stencil bits are untouched by the sprite pass");
        assert!(plane.iter().all(|&s| s & 0x80 == 0), "bit 7 is reset after the sprite pass");
        // A 3D stencil value of 5 does not make a VisibleOutside sprite think it is inside a mask.
        input.masks.clear();
        input.sprites = vec![bits.sprite([-0.5, 0.0], 0.6, BLUE, 0, 1, MaskMode::VisibleOutside)];
        let (img, _) = render_sprites(rig, bits, &q3d, &mut input);
        assert_eq!(classify(px(&img, 16, 32)), C::Blue, "no mask: visible outside even over 3D stencil 5");
    });
}

// ---------------------------------------------------------------------------
// Cost of D32_SFLOAT_S8_UINT vs D32_SFLOAT (run with --ignored --nocapture)
// ---------------------------------------------------------------------------

/// Memory and fill cost of the depth-stencil format against plain depth, on the first device: image
/// memory at 1080p / 1440p / 4K and the time of a depth-heavy pass (clear + many overdrawn full-screen
/// quads, depth test + write, with and without an inactive stencil state).
#[test]
#[ignore = "benchmark: prints numbers, takes a few seconds"]
fn depth_stencil_cost_vs_depth_only() {
    let Some(g) = gpu() else { return };
    let ds = g.depth_stencil_format();
    println!("[bench] device {} depth-stencil format {ds:?}", g.device_name());
    let mut alloc = g.allocator().unwrap();
    for (w, h) in [(1920u32, 1080u32), (2560, 1440), (3840, 2160)] {
        let mut sizes = Vec::new();
        for f in [vk::Format::D32_SFLOAT, ds] {
            let mut img = Img::new(&g, &mut alloc, w, h, 1, f, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED, vk::SampleCountFlags::TYPE_1).unwrap();
            let req = unsafe { g.device.get_image_memory_requirements(img.image) };
            sizes.push(req.size);
            img.destroy(&g, &mut alloc);
        }
        println!("[bench] {w}x{h}: {:?} {:.2} MiB, {ds:?} {:.2} MiB (+{:.1}%)", vk::Format::D32_SFLOAT, sizes[0] as f64 / 1048576.0, sizes[1] as f64 / 1048576.0, (sizes[1] as f64 / sizes[0] as f64 - 1.0) * 100.0);
    }
    drop(alloc);
    // Timing: many full-screen quads at decreasing distance (depth test passes + writes every layer),
    // at 4K, with the target cleared every pass (the clear is where the stencil plane costs bandwidth).
    for (depth, label, only) in [
        (vk::Format::D32_SFLOAT, "D32", false),
        (ds, "D32S8", false),
        (ds, "D32S8 + active stencil test", false),
        (vk::Format::D32_SFLOAT, "D32 depth-only", true),
        (ds, "D32S8 depth-only", true),
    ] {
        let mut rig = Rig::sized(&g, depth, 3840);
        rig.depth_only = only;
        let w = rig.w;
        let st = if label.contains("active") { reader(0, Cmp::Equal) } else { StencilState::default() };
        let quads: Vec<Quad> = (0..if only { 128 } else { 32 })
            .map(|i| {
                let mut q = quad(0, 0, w, w, GREEN, st);
                q.dist = 15.0 - i as f32 * 0.3;
                q
            })
            .collect();
        let ms = rig.time_passes(&quads, 20);
        println!("[bench] {label:>28}: {ms:.3} ms / pass (3840x3840, {} opaque full-target quads, clear + depth test/write)", quads.len());
        rig.destroy();
    }
}
