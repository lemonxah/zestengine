//! Anti-aliasing on the device: FXAA through the real `PostPass`, and the TAA
//! resolve (`aa.rs`) driven with synthetic scene images (colour, depth, motion
//! vectors, reactive mask) the way the scene pass would write them.
//!
//! * FXAA: a hard diagonal edge, error against the analytic coverage image drops
//!   by more than 40 percent, flat regions are untouched.
//! * TAA: (a) a static, jittered scene converges (frame-to-frame delta below
//!   1/255 on a smooth scene; the hard-edged scene ends up closer to the
//!   supersampled reference than a single aliased frame), (b) a moving bright
//!   square leaves < 2 percent ghost energy after 4 frames, (c) a disocclusion
//!   shows no old-object colour after 1 frame.

use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;
use half::f16;

use super::gpu;
use crate::PostFx;
use crate::aa::{AaFrame, AaInputs, AaPass, AaTarget, TAA_BLEND, VELOCITY_SENTINEL, jitter_for, jitter_ndc};
use crate::alloc::{self, Buffer};
use crate::cluster_headless::HeadlessGpu;
use crate::post::PostPass;
use crate::screen_pass::{self, ScreenImage, read_info, sampled};

const W: u32 = 64;
const H: u32 = 64;

// ------------------------------------------------------------------ helpers

/// A sampled test image the host fills.
struct Tex {
    img: ScreenImage,
    bpp: usize,
    w: u32,
    h: u32,
    fresh: bool,
}

fn tex(g: &HeadlessGpu, a: &mut Allocator, format: vk::Format, bpp: usize, w: u32, h: u32) -> Tex {
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    let img = ScreenImage::new(&g.device, a, format, usage, vk::Extent2D { width: w, height: h }, 1, "aa test image").unwrap();
    Tex { img, bpp, w, h, fresh: true }
}

fn upload(g: &HeadlessGpu, a: &mut Allocator, t: &mut Tex, data: &[u8]) {
    assert_eq!(data.len(), (t.w * t.h) as usize * t.bpp);
    let mut staging = Buffer::new(&g.device, a, data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "aa test staging").unwrap();
    staging.write(0, data);
    let old = if t.fresh { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL };
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        crate::image_barrier(&g.device, cb, t.img.image, vk::ImageAspectFlags::COLOR, old, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::empty(), vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: t.w, height: t.h, depth: 1 });
        g.device.cmd_copy_buffer_to_image(cb, staging.handle, t.img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
        crate::image_barrier(&g.device, cb, t.img.image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
    })
    .unwrap();
    staging.destroy(&g.device, a);
    t.fresh = false;
}

fn f16s(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(|x| f16::from_f32(x).to_bits().to_le_bytes()).collect()
}

/// Gray HDR image -> RGBA16F bytes.
fn rgba16_gray(v: &[f32]) -> Vec<u8> {
    f16s(v.iter().flat_map(|&x| [x, x, x, 1.0]))
}

/// Read a SHADER_READ_ONLY RGBA16F image back as floats.
fn read_rgba16(g: &HeadlessGpu, a: &mut Allocator, image: vk::Image, w: u32, h: u32) -> Vec<[f32; 4]> {
    let size = (w * h * 8) as usize;
    let mut host = Buffer::new(&g.device, a, size as u64, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "aa readback").unwrap();
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        crate::image_barrier(&g.device, cb, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        g.device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[copy]);
        crate::image_barrier(&g.device, cb, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::empty(), vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::empty());
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        g.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
    })
    .unwrap();
    let bytes = host.mapped().unwrap();
    let out = (0..(w * h) as usize)
        .map(|i| {
            let px = |c: usize| f16::from_bits(u16::from_le_bytes([bytes[i * 8 + c * 2], bytes[i * 8 + c * 2 + 1]])).to_f32();
            [px(0), px(1), px(2), px(3)]
        })
        .collect();
    host.destroy(&g.device, a);
    out
}

fn proj() -> Mat4 {
    Mat4::perspective_rh(60f32.to_radians(), 1.0, 0.1, 100.0)
}

// ------------------------------------------------------------------ FXAA

/// A hard edge (an exact 45 degree line has no multi-pixel stairs for FXAA to
/// smooth, so the test uses shallower slopes): 0.9 above the line y = slope * x + c (rows grow downward), 0.1 below.
fn edge_value(x: f32, y: f32, slope: f32, c: f32) -> f32 {
    if y < slope * x + c { 0.9 } else { 0.1 }
}

/// Analytic pixel coverage of [edge_value] (8x8 supersampling).
fn ideal_edge(slope: f32, c: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut s = 0.0;
            for j in 0..8 {
                for i in 0..8 {
                    s += edge_value(x as f32 + (i as f32 + 0.5) / 8.0, y as f32 + (j as f32 + 0.5) / 8.0, slope, c);
                }
            }
            out.push(s / 64.0);
        }
    }
    out
}

/// Run the real post pass (no tonemap) over an HDR gray image into RGBA8 and
/// return the red channel in [0, 1].
fn run_post(g: &HeadlessGpu, a: &mut Allocator, src_pixels: &[f32], fxaa: bool, sharpen: f32) -> Vec<f32> {
    let extent = vk::Extent2D { width: W, height: H };
    let mut post = PostPass::new(&g.device, a, vk::Format::R8G8B8A8_UNORM, extent, 1, 1).unwrap();
    let mut src = tex(g, a, vk::Format::R16G16B16A16_SFLOAT, 8, W, H);
    upload(g, a, &mut src, &rgba16_gray(src_pixels));
    let pool = unsafe {
        g.device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 }]),
                None,
            )
            .unwrap()
    };
    let layouts = [post.set_layout()];
    let set = unsafe { g.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).unwrap()[0] };
    let info = read_info(post.sampler(), src.img.view);
    unsafe { g.device.update_descriptor_sets(&[sampled(set, 0, &info)], &[]) };
    let mut out = super::Img::new(g, a, W, H, 1, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap();
    let fx = PostFx { tonemap: 0, ..PostFx::default() };
    post.set_aa(fxaa, sharpen);
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        post.record_bloom(&g.device, cb, post.chain(), set, &fx);
        super::transition(g, cb, &out, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, super::NONE, super::COLOR_OUT);
        let att = [vk::RenderingAttachmentInfo::default()
            .image_view(out.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let info = vk::RenderingInfo::default().render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent }).layer_count(1).color_attachments(&att);
        g.device.cmd_begin_rendering(cb, &info);
        post.record_set(&g.device, cb, set, post.chain().output_set(), extent, &fx, None);
        g.device.cmd_end_rendering(cb);
    })
    .unwrap();
    let bytes = super::read_image(g, a, &out, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, super::COLOR_OUT, 4).unwrap();
    let px: Vec<f32> = bytes.chunks(4).map(|p| p[0] as f32 / 255.0).collect();
    out.destroy(g, a);
    unsafe { g.device.destroy_descriptor_pool(pool, None) };
    src.img.destroy(&g.device, a);
    post.destroy(&g.device, a);
    px
}

/// Mean absolute error against `ideal` over the pixels within 3 px of the edge.
fn edge_error(img: &[f32], ideal: &[f32], slope: f32, c: f32) -> f32 {
    let (mut sum, mut n) = (0.0, 0);
    for y in 0..H {
        for x in 0..W {
            let line = slope * x as f32 + c;
            if (y as f32 + 0.5 - line).abs() < 3.0 {
                sum += (img[(y * W + x) as usize] - ideal[(y * W + x) as usize]).abs();
                n += 1;
            }
        }
    }
    sum / n as f32
}

#[test]
fn fxaa_smooths_a_hard_diagonal_edge_and_leaves_flat_regions() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    for (slope, c) in [(0.15f32, 24.0f32), (0.3, 20.0), (0.6, 8.0)] {
        let src: Vec<f32> = (0..H).flat_map(|y| (0..W).map(move |x| edge_value(x as f32 + 0.5, y as f32 + 0.5, slope, c))).collect();
        let ideal = ideal_edge(slope, c);
        let off = run_post(&g, &mut a, &src, false, 0.0);
        let on = run_post(&g, &mut a, &src, true, 0.0);
        let (e_off, e_on) = (edge_error(&off, &ideal, slope, c), edge_error(&on, &ideal, slope, c));
        let reduction = 1.0 - e_on / e_off;
        // Flat regions: further than 3 px from the edge nothing changes.
        let mut flat_max = 0.0f32;
        for y in 0..H {
            for x in 0..W {
                if (y as f32 + 0.5 - (slope * x as f32 + c)).abs() > 4.0 {
                    flat_max = flat_max.max((on[(y * W + x) as usize] - off[(y * W + x) as usize]).abs());
                }
            }
        }
        // Distinct intermediate levels along the edge: FXAA creates them.
        let levels = |img: &[f32]| img.iter().filter(|&&v| v > 0.15 && v < 0.85).count();
        eprintln!(
            "[fxaa] slope {slope}: edge error vs coverage {e_off:.4} -> {e_on:.4} (-{:.0}%), blended pixels {} -> {}, max flat-region change {flat_max:.4}",
            reduction * 100.0,
            levels(&off),
            levels(&on)
        );
        super::dump_gray(&format!("fxaa_off_{slope}"), &off, W);
        super::dump_gray(&format!("fxaa_on_{slope}"), &on, W);
        assert!(reduction > 0.40, "slope {slope}: FXAA must cut the aliasing error by more than 40% ({e_off} -> {e_on})");
        assert!(flat_max < 1.5 / 255.0, "slope {slope}: flat regions must be unchanged (max change {flat_max})");
    }
}

#[test]
fn sharpen_only_touches_edges_and_never_overshoots() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let src: Vec<f32> = (0..H).flat_map(|y| (0..W).map(move |x| edge_value(x as f32 + 0.5, y as f32 + 0.5, 0.0, 32.0))).collect();
    let soft: Vec<f32> = (0..H * W).map(|i| { let y = (i / W) as f32; 0.1 + 0.8 * ((y - 28.0) / 8.0).clamp(0.0, 1.0) }).collect();
    let _ = src;
    let plain = run_post(&g, &mut a, &soft, false, 0.0);
    let sharp = run_post(&g, &mut a, &soft, false, 1.0);
    let (lo, hi) = (plain.iter().cloned().fold(1.0, f32::min), plain.iter().cloned().fold(0.0, f32::max));
    let overshoot = sharp.iter().map(|&v| (lo - v).max(v - hi).max(0.0)).fold(0.0, f32::max);
    let contrast = |img: &[f32]| (img[(36 * W + 5) as usize] - img[(28 * W + 5) as usize]).abs();
    let steep = |img: &[f32]| (0..H - 1).map(|y| (img[((y + 1) * W + 5) as usize] - img[(y * W + 5) as usize]).abs()).fold(0.0, f32::max);
    eprintln!("[sharpen] steepest step {:.4} -> {:.4}, overshoot {overshoot:.4}, contrast {:.3} -> {:.3}", steep(&plain), steep(&sharp), contrast(&plain), contrast(&sharp));
    assert!(overshoot < 1.5 / 255.0, "sharpening must stay inside the local range (overshoot {overshoot})");
    assert!(steep(&sharp) > steep(&plain), "sharpening steepens the ramp");
    // Flat area untouched.
    assert!((plain[(5 * W + 5) as usize] - sharp[(5 * W + 5) as usize]).abs() < 1.5 / 255.0);
}

// ------------------------------------------------------------------ TAA

struct Rig<'a> {
    g: &'a HeadlessGpu,
    pass: AaPass,
    target: AaTarget,
    color: Tex,
    depth: Tex,
    vel: Tex,
    react: Tex,
    post_layout: vk::DescriptorSetLayout,
    post_sampler: vk::Sampler,
    frame: u32,
    prev_jitter: [f32; 2],
}

/// One synthetic frame's scene inputs.
struct Frame {
    color: Vec<f32>,
    depth: Vec<f32>,
    /// Velocity in NDC (cur - prev, jitter delta NOT included), written as the
    /// scene pass does: motion + this frame's jitter delta; `None` = the
    /// sentinel (pixel no pipeline wrote).
    motion: Vec<Option<[f32; 2]>>,
    react: Vec<f32>,
}

impl<'a> Rig<'a> {
    fn new(g: &'a HeadlessGpu, a: &mut Allocator, w: u32, h: u32) -> Self {
        let pass = AaPass::new(&g.device).unwrap();
        let post_layout = screen_pass::samplers_layout(&g.device, 1).unwrap();
        let post_sampler = screen_pass::clamp_sampler(&g.device, true).unwrap();
        let target = AaTarget::new(&g.device, a, &pass, post_layout, post_sampler, vk::Extent2D { width: w, height: h }, false).unwrap();
        Self {
            g,
            pass,
            target,
            color: tex(g, a, vk::Format::R16G16B16A16_SFLOAT, 8, w, h),
            depth: tex(g, a, vk::Format::R32_SFLOAT, 4, w, h),
            vel: tex(g, a, crate::pipeline::VELOCITY_FORMAT, 4, w, h),
            react: tex(g, a, crate::pipeline::REACTIVE_FORMAT, 1, w, h),
            post_layout,
            post_sampler,
            frame: 0,
            prev_jitter: [0.0; 2],
        }
    }

    /// Resolve one frame rendered with `jitter` (the Halton sample, 0.5 = none).
    fn step(&mut self, a: &mut Allocator, f: &Frame, jitter: [f32; 2], slot: usize) -> Vec<[f32; 4]> {
        let (w, h) = (self.color.w, self.color.h);
        let cur = jitter_ndc(jitter, [w as f32, h as f32]);
        let delta = [cur[0] - self.prev_jitter[0], cur[1] - self.prev_jitter[1]];
        self.prev_jitter = cur;
        let vel: Vec<u8> = f16s(f.motion.iter().flat_map(|m| match m {
            Some(m) => [m[0] + delta[0], m[1] + delta[1]],
            None => [VELOCITY_SENTINEL, VELOCITY_SENTINEL],
        }));
        upload(self.g, a, &mut self.color, &rgba16_gray(&f.color));
        upload(self.g, a, &mut self.depth, bytemuck::cast_slice(&f.depth));
        upload(self.g, a, &mut self.vel, &vel);
        let react: Vec<u8> = f.react.iter().map(|&r| (r * 255.0) as u8).collect();
        upload(self.g, a, &mut self.react, &react);
        let fr = AaFrame {
            view_proj: Mat4::IDENTITY,
            prev_view_proj: Mat4::IDENTITY,
            proj: proj(),
            jitter_delta: delta,
            blend: TAA_BLEND,
            reset: false,
        };
        let inputs = AaInputs { color: self.color.img.view, depth: self.depth.img.view, velocity: self.vel.img.view, reactive: self.react.img.view };
        let g = self.g;
        alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| {
            self.pass.record(&g.device, cb, &mut self.target, slot, &inputs, &fr);
        })
        .unwrap();
        self.frame += 1;
        read_rgba16(self.g, a, self.target.last_output().image, w, h)
    }

    fn destroy(mut self, a: &mut Allocator) {
        let d = &self.g.device;
        self.target.destroy(d, a);
        for t in [&mut self.color, &mut self.depth, &mut self.vel, &mut self.react] {
            t.img.destroy(d, a);
        }
        self.pass.destroy(d);
        unsafe {
            d.destroy_descriptor_set_layout(self.post_layout, None);
            d.destroy_sampler(self.post_sampler, None);
        }
    }
}

fn gray(out: &[[f32; 4]]) -> Vec<f32> {
    out.iter().map(|p| p[0]).collect()
}

/// The scene shifts by (jitter - 0.5) px right / up in the rendered image
/// (`jitter_proj`): sampling it at the pixel centre moved the other way.
fn sample_at(f: impl Fn(f32, f32) -> f32, x: u32, y: u32, jitter: [f32; 2]) -> f32 {
    f(x as f32 + 0.5 - (jitter[0] - 0.5), y as f32 + 0.5 + (jitter[1] - 0.5))
}

fn static_frame(f: impl Fn(f32, f32) -> f32, jitter: [f32; 2]) -> Frame {
    let n = (W * H) as usize;
    Frame {
        color: (0..n as u32).map(|i| sample_at(&f, i % W, i / W, jitter)).collect(),
        depth: vec![0.5; n],
        motion: vec![Some([0.0, 0.0]); n],
        react: vec![0.0; n],
    }
}

#[test]
fn taa_converges_on_a_static_scene() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    // (1) Smooth scene (a soft-edged disc, 24 px transition): after 32 jittered
    // frames the output is stable to 1/255.
    let smooth = |x: f32, y: f32| {
        let d = ((x - 32.0).powi(2) + (y - 32.0).powi(2)).sqrt() - 18.0;
        let t = ((d + 12.0) / 24.0).clamp(0.0, 1.0);
        0.15 + 0.7 * t * t * (3.0 - 2.0 * t)
    };
    let mut rig = Rig::new(&g, &mut a, W, H);
    let mut prev = Vec::new();
    let mut last = Vec::new();
    let mut max_delta = 0.0f32;
    for i in 0..32u32 {
        let j = jitter_for(i + 1);
        let out = gray(&rig.step(&mut a, &static_frame(smooth, j), j, (i % 2) as usize));
        if i == 30 {
            prev = out.clone();
        }
        if i == 31 {
            max_delta = prev.iter().zip(&out).map(|(p, c)| (p - c).abs()).fold(0.0, f32::max);
        }
        last = out;
    }
    let truth: Vec<f32> = (0..W * H).map(|i| smooth((i % W) as f32 + 0.5, (i / W) as f32 + 0.5)).collect();
    let err = last.iter().zip(&truth).map(|(o, t)| (o - t).abs()).fold(0.0, f32::max);
    eprintln!("[taa static, smooth] frame-to-frame max delta after 32 frames {max_delta:.5} (1/255 = {:.5}), max error vs truth {err:.4}", 1.0 / 255.0);
    rig.destroy(&mut a);
    assert!(max_delta < 1.0 / 255.0, "a static scene must be stable after 32 frames (max delta {max_delta})");
    assert!(err < 0.02, "converged TAA must reproduce the smooth scene (max error {err})");

    // (2) Hard diagonal edge: the converged image is closer to the supersampled
    // reference than a single aliased frame.
    let (slope, c) = (0.37f32, 14.0f32);
    let hard = |x: f32, y: f32| edge_value(x, y, slope, c);
    let ideal = ideal_edge(slope, c);
    let mut rig = Rig::new(&g, &mut a, W, H);
    let mut out = Vec::new();
    for i in 0..32u32 {
        let j = jitter_for(i + 1);
        out = gray(&rig.step(&mut a, &static_frame(hard, j), j, (i % 2) as usize));
    }
    rig.destroy(&mut a);
    let aliased: Vec<f32> = (0..W * H).map(|i| hard((i % W) as f32 + 0.5, (i / W) as f32 + 0.5)).collect();
    let (e_alias, e_taa) = (edge_error(&aliased, &ideal, slope, c), edge_error(&out, &ideal, slope, c));
    super::dump_gray("taa_hard_converged", &out, W);
    super::dump_gray("taa_hard_aliased", &aliased, W);
    eprintln!("[taa static, hard edge] edge error vs 8x8 supersampled reference: aliased {e_alias:.4}, TAA {e_taa:.4} (-{:.0}%)", (1.0 - e_taa / e_alias) * 100.0);
    assert!(e_taa < 0.5 * e_alias, "TAA must anti-alias the edge ({e_alias} -> {e_taa})");
}

/// A bright square of side `s` whose left edge is at `x0`, over y in [20, 20 + s).
fn square(x0: f32, s: f32) -> impl Fn(f32, f32) -> f32 {
    move |x, y| if x >= x0 && x < x0 + s && y >= 20.0 && y < 20.0 + s { 8.0 } else { 0.05 }
}

#[test]
fn taa_leaves_no_ghost_behind_a_moving_square() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let mut rig = Rig::new(&g, &mut a, W, H);
    let speed = 3.0f32; // px per frame
    let s = 12.0f32;
    let ndc_motion = [speed * 2.0 / W as f32, 0.0];
    let mut x0 = 10.0f32;
    let mut energy_ghost = 0.0f32;
    let mut energy_square = 1.0f32;
    // 6 static frames to fill the history, then 6 moving frames (N = 4 checked).
    for i in 0..12u32 {
        if i >= 6 {
            x0 += speed;
        }
        let moving = i >= 6;
        let j = jitter_for(i + 1);
        let f = square(x0, s);
        let mut fr = static_frame(&f, j);
        if moving {
            for y in 0..H {
                for x in 0..W {
                    let c = (x as f32 + 0.5, y as f32 + 0.5);
                    if c.0 >= x0 - 0.5 && c.0 < x0 + s + 0.5 && c.1 >= 19.5 && c.1 < 20.5 + s {
                        fr.motion[(y * W + x) as usize] = Some(ndc_motion);
                    }
                }
            }
        }
        let out = gray(&rig.step(&mut a, &fr, j, (i % 2) as usize));
        if i == 9 {
            // 4 moving frames in. Ghost = excess energy over the dark background
            // away (>= 2 px) from the square; compare with the square's energy.
            energy_ghost = 0.0;
            energy_square = 0.0;
            for y in 0..H {
                for x in 0..W {
                    let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                    let dx = (x0 - cx).max(cx - (x0 + s)).max(0.0);
                    let dy = (20.0 - cy).max(cy - (20.0 + s)).max(0.0);
                    let dist = dx.max(dy);
                    let v = out[(y * W + x) as usize];
                    if dist == 0.0 {
                        energy_square += v - 0.05;
                    } else if dist >= 2.0 {
                        energy_ghost += (v - 0.05).max(0.0);
                    }
                }
            }
        }
    }
    rig.destroy(&mut a);
    let ratio = energy_ghost / energy_square;
    eprintln!("[taa moving square] ghost energy / square energy after 4 moving frames: {:.4}%", ratio * 100.0);
    assert!(ratio < 0.02, "trailing ghost energy {ratio} must be below 2%");
}

#[test]
fn taa_rejects_disoccluded_history() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let mut rig = Rig::new(&g, &mut a, W, H);
    // A very contrasty background (alternating 0.05 / 6.0) so the neighbourhood
    // clip alone cannot reject a bright old-object colour: only the depth test can.
    let bg = |x: u32, y: u32| if (x + y).is_multiple_of(2) { 0.05 } else { 6.0 };
    let n = (W * H) as usize;
    let obj = |x: u32| (20..40).contains(&x);
    // 10 frames with a near object (depth 0.3, colour 5.0) over x in 20..40.
    let near = Frame {
        color: (0..n as u32).map(|i| if obj(i % W) { 5.0 } else { bg(i % W, i / W) }).collect(),
        depth: (0..n as u32).map(|i| if obj(i % W) { 0.3 } else { 0.8 }).collect(),
        motion: vec![Some([0.0, 0.0]); n],
        react: vec![0.0; n],
    };
    for i in 0..10u32 {
        rig.step(&mut a, &near, [0.5, 0.5], (i % 2) as usize);
    }
    // The object vanishes: the revealed background is at depth 0.8.
    let gone = Frame {
        color: (0..n as u32).map(|i| bg(i % W, i / W)).collect(),
        depth: vec![0.8; n],
        motion: vec![Some([0.0, 0.0]); n],
        react: vec![0.0; n],
    };
    let out = gray(&rig.step(&mut a, &gone, [0.5, 0.5], 0));
    rig.destroy(&mut a);
    let mut worst = 0.0f32;
    for y in 0..H {
        for x in 22..38u32 {
            let i = (y * W + x) as usize;
            worst = worst.max((out[i] - gone.color[i]).abs() / gone.color[i].max(0.5));
        }
    }
    eprintln!("[taa disocclusion] worst relative deviation from the current frame in the revealed area after 1 frame: {:.4}%", worst * 100.0);
    assert!(worst < 0.02, "revealed background must show no old-object colour (deviation {worst})");
}

#[test]
fn taa_reprojects_sentinel_pixels_from_depth() {
    // A sky-like pixel (no velocity written) under camera motion: the resolve must
    // follow the camera, not smear. Camera pans by 4 px; the scene is a vertical
    // bar at the far plane, so its history lands 4 px away.
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let mut rig = Rig::new(&g, &mut a, W, H);
    let n = (W * H) as usize;
    let shift = 4.0f32;
    let bar = |x: f32| if (20.0..28.0).contains(&x) { 4.0 } else { 0.1 };
    let frame_at = |offset: f32| Frame {
        color: (0..n as u32).map(|i| bar((i % W) as f32 + 0.5 + offset)).collect(),
        depth: vec![1.0; n],
        motion: vec![None; n],
        react: vec![0.0; n],
    };
    // Camera yaw as a pure NDC translation of the view-projection: moving the
    // content left by `shift` px = -shift*2/W in NDC x.
    let mut out = Vec::new();
    for i in 0..10u32 {
        let off = shift * i as f32 * 0.25;
        let f = frame_at(off);
        let cur_t = Mat4::from_translation(glam::Vec3::new(-(off) * 2.0 / W as f32, 0.0, 0.0));
        let prev_t = Mat4::from_translation(glam::Vec3::new(-(off - shift * 0.25) * 2.0 / W as f32, 0.0, 0.0));
        out = step_with_matrices(&mut rig, &mut a, &f, cur_t, prev_t, (i % 2) as usize);
    }
    rig.destroy(&mut a);
    // Final content offset is 9 * 1 = 9 px: the bar spans x in [11, 19) in the image.
    let off = shift * 9.0 * 0.25;
    let mut worst = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let want = bar(x as f32 + 0.5 + off);
            worst = worst.max((out[(y * W + x) as usize] - want).abs() / want.max(0.5));
        }
    }
    eprintln!("[taa sentinel reprojection] worst relative deviation from the current frame while panning: {:.4}%", worst * 100.0);
    assert!(worst < 0.05, "sky pixels must reproject with the camera (deviation {worst})");
}

fn step_with_matrices(rig: &mut Rig, a: &mut Allocator, f: &Frame, cur: Mat4, prev: Mat4, slot: usize) -> Vec<f32> {
    let g = rig.g;
    let (w, h) = (rig.color.w, rig.color.h);
    upload(g, a, &mut rig.color, &rgba16_gray(&f.color));
    upload(g, a, &mut rig.depth, bytemuck::cast_slice(&f.depth));
    let vel = f16s(f.motion.iter().flat_map(|_| [VELOCITY_SENTINEL, VELOCITY_SENTINEL]));
    upload(g, a, &mut rig.vel, &vel);
    let react = vec![0u8; (w * h) as usize];
    upload(g, a, &mut rig.react, &react);
    let fr = AaFrame { view_proj: cur, prev_view_proj: prev, proj: proj(), jitter_delta: [0.0, 0.0], blend: TAA_BLEND, reset: false };
    let inputs = AaInputs { color: rig.color.img.view, depth: rig.depth.img.view, velocity: rig.vel.img.view, reactive: rig.react.img.view };
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| {
        rig.pass.record(&g.device, cb, &mut rig.target, slot, &inputs, &fr);
    })
    .unwrap();
    gray(&read_rgba16(g, a, rig.target.last_output().image, w, h))
}

#[test]
fn taa_cost_at_1440p() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let (w, h) = (2560u32, 1440u32);
    let mut rig = Rig::new(&g, &mut a, w, h);
    let n = (w * h) as usize;
    upload(&g, &mut a, &mut rig.color, &rgba16_gray(&vec![0.5; n]));
    upload(&g, &mut a, &mut rig.depth, bytemuck::cast_slice(&vec![0.5f32; n]));
    upload(&g, &mut a, &mut rig.vel, &f16s(std::iter::repeat_n(0.0, n * 2)));
    upload(&g, &mut a, &mut rig.react, &vec![0u8; n]);
    let pool = unsafe { g.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap() };
    let inputs = AaInputs { color: rig.color.img.view, depth: rig.depth.img.view, velocity: rig.vel.img.view, reactive: rig.react.img.view };
    let fr = AaFrame { view_proj: Mat4::IDENTITY, prev_view_proj: Mat4::IDENTITY, proj: proj(), jitter_delta: [0.0; 2], blend: TAA_BLEND, reset: false };
    let mut best = f32::MAX;
    for i in 0..12usize {
        alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
            g.device.cmd_reset_query_pool(cb, pool, 0, 2);
            g.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
            rig.pass.record(&g.device, cb, &mut rig.target, i % 2, &inputs, &fr);
            g.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
        })
        .unwrap();
        let mut res = [crate::TsResult::default(); 2];
        unsafe { g.device.get_query_pool_results(pool, 0, &mut res, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY).unwrap() };
        if i >= 2
            && let Some(ms) = crate::ts_zone_ms(res[0], res[1], g.timestamp_period as f64, crate::ts_valid_mask(g.timestamp_valid_bits))
        {
            best = best.min(ms);
        }
    }
    eprintln!("[taa] 2560x1440 resolve GPU time (best of 10): {best:.3} ms on {}", g.device_name());
    unsafe { g.device.destroy_query_pool(pool, None) };
    rig.destroy(&mut a);
    assert!(best < 1.0, "TAA resolve at 1440p took {best} ms");
}
