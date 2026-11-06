//! Screen-space ambient occlusion (`ao.rs`) on the device, both quality modes,
//! from a depth buffer ray-cast on the CPU (so the test checks the AO passes
//! alone): a floor point in the inner corner between two walls must be
//! occluded, a point on the open floor and every pixel of a lone ground plane
//! must not be (no self-occlusion on flat, grazing surfaces). Also times the
//! whole AO chain at 2560x1440.

use ash::vk;
use glam::{Mat4, Vec3, Vec4Swizzles};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::gpu;
use crate::alloc::{self, Buffer};
use crate::ao::{AoFrame, AoPass, AoQuality, AoSettings, AoTarget};
use crate::cluster_headless::HeadlessGpu;
use crate::screen_pass::ScreenImage;

/// Axis-aligned boxes standing on an (optional) ground plane y = 0.
struct Scene {
    ground: bool,
    boxes: Vec<(Vec3, Vec3)>,
}

impl Scene {
    fn hit(&self, o: Vec3, d: Vec3) -> Option<f32> {
        let mut best: Option<f32> = None;
        let mut take = |t: f32| {
            if t > 1e-4 && best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        };
        if self.ground && d.y.abs() > 1e-6 {
            let t = -o.y / d.y;
            let p = o + d * t;
            if p.x.abs() < 40.0 && p.z.abs() < 40.0 {
                take(t);
            }
        }
        for &(lo, hi) in &self.boxes {
            let inv = d.recip();
            let t0 = (lo - o) * inv;
            let t1 = (hi - o) * inv;
            let tmin = t0.min(t1).max_element();
            let tmax = t0.max(t1).min_element();
            if tmax >= tmin.max(0.0) {
                take(tmin);
            }
        }
        best
    }
}

struct Cam {
    view: Mat4,
    proj: Mat4,
    eye: Vec3,
}

/// Depth buffer values [0, 1] for `scene` (row 0 = top, as the scene pass renders).
fn raycast_depth(scene: &Scene, cam: &Cam, w: u32, h: u32) -> Vec<f32> {
    let inv_proj = cam.proj.inverse();
    let inv_view = cam.view.inverse();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let ndc = glam::Vec2::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0);
            let v = inv_proj * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
            let dir_view = (v.xyz() / v.w).normalize();
            let dir = inv_view.transform_vector3(dir_view).normalize();
            let d = match scene.hit(cam.eye, dir) {
                Some(t) => {
                    let p = cam.eye + dir * t;
                    let clip = cam.proj * cam.view * p.extend(1.0);
                    (clip.z / clip.w).clamp(0.0, 1.0)
                }
                None => 1.0,
            };
            out.push(d);
        }
    }
    out
}

/// Pixel (x, y) of a world point.
fn pixel_of(cam: &Cam, p: Vec3, w: u32, h: u32) -> (u32, u32) {
    let clip = cam.proj * cam.view * p.extend(1.0);
    let ndc = clip.xy() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * w as f32) as u32;
    let y = ((0.5 - ndc.y * 0.5) * h as f32) as u32;
    (x.min(w - 1), y.min(h - 1))
}

fn upload_depth(g: &HeadlessGpu, a: &mut Allocator, depth: &[f32], w: u32, h: u32) -> ScreenImage {
    let extent = vk::Extent2D { width: w, height: h };
    let img = ScreenImage::new(&g.device, a, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST, extent, 1, "test depth").unwrap();
    let mut staging = Buffer::new(&g.device, a, (depth.len() * 4) as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "test depth staging").unwrap();
    staging.write(0, bytemuck::cast_slice(depth));
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        crate::image_barrier(&g.device, cb, img.image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(), vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        g.device.cmd_copy_buffer_to_image(cb, staging.handle, img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
        crate::image_barrier(&g.device, cb, img.image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
    })
    .unwrap();
    staging.destroy(&g.device, a);
    img
}

/// Read back a SHADER_READ_ONLY single-channel R8 image as [0, 1] values.
pub(crate) fn read_r8(g: &HeadlessGpu, a: &mut Allocator, image: vk::Image, w: u32, h: u32) -> Vec<f32> {
    let mut host = Buffer::new(&g.device, a, (w * h) as u64, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "ao readback").unwrap();
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
    let out = host.mapped().unwrap()[..(w * h) as usize].iter().map(|&b| b as f32 / 255.0).collect();
    host.destroy(&g.device, a);
    out
}

fn settings() -> AoSettings {
    AoSettings { enabled: true, intensity: 1.0, radius: 0.6, falloff: 0.6, multi_bounce: true }
}

/// Run `frames` AO frames (static camera) and return the full-res result.
#[allow(clippy::too_many_arguments)]
fn run_ao(g: &HeadlessGpu, a: &mut Allocator, pass: &AoPass, scene: &Scene, cam: &Cam, w: u32, h: u32, quality: AoQuality, frames: u32) -> Vec<f32> {
    let depth = raycast_depth(scene, cam, w, h);
    let mut depth_img = upload_depth(g, a, &depth, w, h);
    let mut target = AoTarget::new(&g.device, a, pass, vk::Extent2D { width: w, height: h }).unwrap();
    for i in 0..frames {
        let f = AoFrame { proj: cam.proj, view: cam.view, settings: settings(), quality, frame: i };
        alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| pass.record(&g.device, cb, &mut target, (i % 2) as usize, depth_img.view, &f)).unwrap();
    }
    let out = read_r8(g, a, target.full.image, w, h);
    target.destroy(&g.device, a);
    depth_img.destroy(&g.device, a);
    out
}

fn corner_scene() -> Scene {
    Scene {
        ground: true,
        boxes: vec![
            // Back wall and left wall: an inner corner on the floor at (-1.6, 0, -1.0).
            (Vec3::new(-2.0, 0.0, -1.4), Vec3::new(2.0, 1.5, -1.0)),
            (Vec3::new(-2.0, 0.0, -1.4), Vec3::new(-1.6, 1.5, 1.5)),
        ],
    }
}

fn camera(w: u32, h: u32) -> Cam {
    let eye = Vec3::new(0.6, 2.2, 3.2);
    let view = Mat4::look_at_rh(eye, Vec3::new(-0.5, 0.0, -0.4), Vec3::Y);
    let proj = Mat4::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 0.1, 100.0);
    Cam { view, proj, eye }
}

#[test]
fn ao_darkens_the_corner_between_two_walls_and_not_the_open_floor() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let pass = AoPass::new(&g.device).unwrap();
    let (w, h) = (320u32, 240u32);
    let cam = camera(w, h);
    let scene = corner_scene();
    for quality in AoQuality::ALL {
        let frames = if quality == AoQuality::High { 8 } else { 1 };
        let ao = run_ao(&g, &mut a, &pass, &scene, &cam, w, h, quality, frames);
        super::dump_gray(&format!("ao_corner_{quality:?}"), &ao, w);
        let at = |p: Vec3| {
            let (x, y) = pixel_of(&cam, p, w, h);
            // 3x3 mean: robust to the exact pixel a point lands on.
            let mut s = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    s += ao[(yy * w + xx) as usize];
                }
            }
            s / 9.0
        };
        let corner = at(Vec3::new(-1.5, 0.0, -0.9));
        let edge = at(Vec3::new(0.6, 0.0, -0.9));
        let open = at(Vec3::new(1.2, 0.0, 1.4));
        let wall = at(Vec3::new(0.3, 1.0, -1.0));
        eprintln!("[ao {quality:?}] corner {corner:.3} wall-floor edge {edge:.3} open floor {open:.3} open wall {wall:.3}");
        assert!(corner < 0.7, "{quality:?}: the inner corner must be occluded, got {corner}");
        assert!(edge < 0.85 && edge > corner - 0.02, "{quality:?}: the wall/floor edge is occluded, less than the corner: {edge} vs {corner}");
        assert!(open > 0.95, "{quality:?}: the open floor must stay unoccluded, got {open}");
        assert!(wall > 0.9, "{quality:?}: the middle of the wall must stay unoccluded, got {wall}");
    }
    let mut pass = pass;
    pass.destroy(&g.device);
}

#[test]
fn ao_leaves_a_lone_ground_plane_unoccluded() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let pass = AoPass::new(&g.device).unwrap();
    let (w, h) = (320u32, 240u32);
    // A low, grazing view over a flat ground: the case where AO from
    // reconstructed normals tends to self-occlude.
    let eye = Vec3::new(0.0, 0.6, 3.0);
    let cam = Cam {
        view: Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, -6.0), Vec3::Y),
        proj: Mat4::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 0.1, 100.0),
        eye,
    };
    let scene = Scene { ground: true, boxes: Vec::new() };
    let depth = raycast_depth(&scene, &cam, w, h);
    for quality in AoQuality::ALL {
        let frames = if quality == AoQuality::High { 8 } else { 1 };
        let ao = run_ao(&g, &mut a, &pass, &scene, &cam, w, h, quality, frames);
        super::dump_gray(&format!("ao_plane_{quality:?}"), &ao, w);
        // Ground pixels up to ~20 m away (beyond that a pixel spans metres).
        let near_far = |d: f32| {
            let (p22, p32) = (cam.proj.z_axis.z, cam.proj.w_axis.z);
            p32 / (d + p22)
        };
        let ground: Vec<f32> = ao.iter().zip(&depth).filter(|&(_, &d)| d < 1.0 && near_far(d) < 20.0).map(|(&v, _)| v).collect();
        assert!(ground.len() > (w * h / 4) as usize, "the plane covers the lower image");
        let min = ground.iter().cloned().fold(1.0f32, f32::min);
        let mean = ground.iter().sum::<f32>() / ground.len() as f32;
        eprintln!("[ao {quality:?}] open plane: min {min:.3} mean {mean:.4} over {} px", ground.len());
        assert!(mean > 0.98, "{quality:?}: a flat plane must not occlude itself (mean {mean})");
        assert!(min > 0.85, "{quality:?}: no dark speckles on a flat plane (min {min})");
    }
    let mut pass = pass;
    pass.destroy(&g.device);
}

#[test]
fn ao_high_cost_at_1440p() {
    let Some(g) = gpu() else { return };
    let mut a = g.allocator().unwrap();
    let pass = AoPass::new(&g.device).unwrap();
    let (w, h) = (2560u32, 1440u32);
    let cam = camera(w, h);
    let depth = raycast_depth(&corner_scene(), &cam, w, h);
    let mut depth_img = upload_depth(&g, &mut a, &depth, w, h);
    let mut target = AoTarget::new(&g.device, &mut a, &pass, vk::Extent2D { width: w, height: h }).unwrap();
    let pool = unsafe {
        g.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap()
    };
    let mut best = [f32::MAX; 2];
    for (qi, quality) in AoQuality::ALL.into_iter().enumerate() {
        for i in 0..12u32 {
            let f = AoFrame { proj: cam.proj, view: cam.view, settings: settings(), quality, frame: i };
            alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
                g.device.cmd_reset_query_pool(cb, pool, 0, 2);
                g.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
                pass.record(&g.device, cb, &mut target, (i % 2) as usize, depth_img.view, &f);
                g.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
            })
            .unwrap();
            let mut res = [crate::TsResult::default(); 2];
            unsafe {
                g.device.get_query_pool_results(pool, 0, &mut res, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY).unwrap();
            }
            if i >= 2
                && let Some(ms) = crate::ts_zone_ms(res[0], res[1], g.timestamp_period as f64, crate::ts_valid_mask(g.timestamp_valid_bits))
            {
                best[qi] = best[qi].min(ms);
            }
        }
    }
    eprintln!("[ao] 2560x1440 GPU time (best of 10): Low (SAO) {:.3} ms, High (GTAO + temporal) {:.3} ms on {}", best[0], best[1], g.device_name());
    unsafe { g.device.destroy_query_pool(pool, None) };
    target.destroy(&g.device, &mut a);
    depth_img.destroy(&g.device, &mut a);
    let mut pass = pass;
    pass.destroy(&g.device);
    // Budget 0.3 ms on an RTX 4080 class GPU; generous bound against a shared GPU.
    assert!(best[1] < 1.0, "GTAO at 1440p took {} ms", best[1]);
}
