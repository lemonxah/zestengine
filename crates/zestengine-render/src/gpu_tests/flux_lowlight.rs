//! Diagnostic: the final gather must behave identically at every radiance
//! scale. The same scene is rendered with its only light scaled down; relative
//! temporal noise of the integrated and history-filtered GI is then compared.
use ash::vk;
use glam::{Mat4, Vec3};

use crate::{CameraData, FluxSettings, GpuGiMaterial, Renderer};
use super::planar::{cube, draw, lit};

/// Mean RGB luminance of a 5x5 block of an rgba16f image around a world point.
fn sample(r: &Renderer, image: vk::Image, size: vk::Extent2D, cam: CameraData, point: Vec3) -> anyhow::Result<f32> {
    let q = (cam.proj * cam.view).project_point3(point);
    let offset = vk::Offset3D { x: ((q.x * 0.5 + 0.5) * size.width as f32) as i32 - 2,
        y: ((0.5 - q.y * 0.5) * size.height as f32) as i32 - 2, z: 0 };
    let device = &r.ctx.device;
    let allocator = r.allocator();
    let mut alloc = allocator.lock().unwrap();
    let mut host = crate::alloc::Buffer::new(device, &mut alloc, 25 * 8, vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuToCpu, "Flux low light readback")?;
    crate::alloc::one_time_submit(device, r.command_pool, r.ctx.queue, |cb| unsafe {
        crate::image_barrier(device, cb, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_offset(offset).image_extent(vk::Extent3D { width: 5, height: 5, depth: 1 })]);
        crate::image_barrier(device, cb, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_READ);
    })?;
    let bytes = host.read();
    let half = |b: &[u8]| half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32();
    let mean = bytes[..200].as_chunks::<8>().0.iter()
        .map(|b| 0.2126 * half(&b[0..2]) + 0.7152 * half(&b[2..4]) + 0.0722 * half(&b[4..6])).sum::<f32>() / 25.0;
    host.destroy(device, &mut alloc);
    Ok(mean)
}

/// Mean, coefficient of variation, and mean absolute frame-to-frame step
/// relative to the mean.
fn stats(values: &[f32]) -> (f32, f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let step = values.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (n - 1.0);
    (mean, var.sqrt() / mean.max(1e-30), step / mean.max(1e-30))
}

/// Gather specializations compile in the background, and a view records
/// nothing (keeping its previous image) until its set is ready. Render until
/// this view has traced with no compilation outstanding.
pub(super) fn await_pipelines(r: &mut Renderer, input: &crate::FrameInput<'_>) -> anyhow::Result<f32> {
    let started = std::time::Instant::now();
    loop {
        let parity = r.sgi_parity;
        r.render(input)?;
        if r.sgi_parity != parity && !r.shader_tasks().iter().any(|t| t.result.is_none()) { break; }
        anyhow::ensure!(started.elapsed().as_secs() < 300, "Flux pipelines did not finish compiling");
    }
    Ok(started.elapsed().as_secs_f32())
}

pub(super) fn run(r: &mut Renderer) -> anyhow::Result<()> {
    let mesh = r.upload_mesh(&cube())?;
    let grey = lit(r, [0.7; 4], [0.0; 3], 1.0, 0.0);
    let red = lit(r, [0.8, 0.02, 0.02, 1.0], [0.0; 3], 1.0, 0.0);
    let boxes = [(Vec3::new(0.0, -0.1, 0.0), Vec3::new(4.0, 0.1, 4.0)),
        (Vec3::new(1.5, 1.5, 0.0), Vec3::new(0.1, 1.5, 3.0))];
    let draws: Vec<_> = boxes.iter().enumerate().map(|(i, (p, h))| {
        let mut d = draw(mesh, if i == 0 { grey } else { red },
            Mat4::from_scale_rotation_translation(*h * 2.0, glam::Quat::IDENTITY, *p), -1);
        d.shadow_casting = crate::ShadowCasting::On;
        d
    }).collect();
    let dims = [64, 32, 64];
    let (min, max) = (Vec3::new(-4.0, -1.0, -4.0), Vec3::new(4.0, 4.0, 4.0));
    let (mut dist, mut index) = (Vec::new(), Vec::new());
    for z in 0..dims[2] { for y in 0..dims[1] { for x in 0..dims[0] {
        let p = min + (max - min) * Vec3::new(x as f32 / 63.0, y as f32 / 31.0, z as f32 / 63.0);
        let (i, d) = boxes.iter().enumerate().map(|(i, (c, h))| {
            let q = (p - *c).abs() - *h;
            (i, q.max(Vec3::ZERO).length() + q.max_element().min(0.0))
        }).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        dist.push(d); index.push(i as u32);
    } } }
    r.gi_set_gdf(&dist, &index, dims, min.to_array(), max.to_array(), &[
        GpuGiMaterial { albedo: [0.7; 3], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
        GpuGiMaterial { albedo: [0.8, 0.02, 0.02], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
    ]);
    r.gi_set_emitters(&[]); r.gi_set_mesh_sdfs(&[])?; r.set_gi_enabled(true);
    // (lamp, auto exposure, radiance scale, ablation, light). Ablations:
    // 1 = single bounce (no surface radiosity), 2 = radiosity without its ray
    // limit and spatial filter (the paired reference).
    let light = |position: Vec3, range: f32, scale: f32| [crate::LightInstance { kind: crate::LightKind::Point, position,
        direction: Vec3::NEG_Y, color: [1.0; 3], intensity: 15.0 * scale, range, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
        cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false }];
    let mut configs = Vec::new();
    // A narrow lamp on the wall: most of the room is lit only by bounce.
    for (label, position, range) in [("broad", Vec3::new(0.0, 1.5, 0.0), 10.0f32), ("narrow", Vec3::new(1.0, 2.0, 0.0), 1.2)] {
        for (auto, scale) in [(false, 1.0f32), (true, 1e-2), (true, 1e-4)] {
            for ablation in [0u32, 2] { configs.push((label, auto, scale, ablation, light(position, range, scale))); }
        }
        configs.push((label, false, 1.0, 1, light(position, range, 1.0)));
    }
    let eye = Vec3::new(0.0, 2.5, 5.0);
    let camera = CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
        proj: Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0) };
    let mut input = crate::cascade_tests::input(camera);
    input.viewport_extent = Some([640, 360]);
    input.draws = &draws;
    input.light.ambient = [0.0; 3];
    input.light.intensity = 0.0;
    input.light.env_intensity = 0.0;
    input.postfx.bloom_enabled = false;
    // Two receivers on the floor: one beside the lit wall, one far from it.
    let points = [Vec3::new(0.5, 0.0, 0.0), Vec3::new(-2.5, 0.0, 1.0)];
    let mut clock = 100.0f64;
    for rt in [false, true] {
        if rt && !r.ctx.ray_tracing() { continue; }
        for (label, auto, scale, ablation, lights) in &configs {
            let (auto, scale, ablation) = (*auto, *scale, *ablation);
            r.set_flux_settings(FluxSettings { rt_trace: rt, reflection_mode: 0, ssr_enabled: false,
                bounces: if ablation == 1 { 1 } else { FluxSettings::default().bounces }, ..Default::default() });
            r.gpu_gi.as_mut().unwrap().radiosity_reference = ablation == 2;
            input.lights = lights;
            input.postfx.auto_exposure = crate::AutoExposure { enabled: auto, ..Default::default() };
            r.invalidate_screen_gi();
            let waited = await_pipelines(r, &input)?;
            let mut traced = 0;
            let mut series = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
            let mut stops = Vec::new();
            let mut radiosity = Vec::new();
            let mut timing = [0.0f32; 5];
            for frame in 0..540 {
                clock += 1.0 / 60.0;
                r.post_pass.exposure.time_override = Some(clock);
                let parity = r.sgi_parity;
                r.render(&input)?;
                if frame < 360 { continue; }
                traced += usize::from(r.sgi_parity != parity);
                let target = r.viewport_target.as_ref().unwrap();
                let (integrated, size) = (target.sgi.gi_final, target.extent);
                let history = target.sgi.full_history[(r.sgi_parity ^ 1) & 1].image;
                for (i, point) in points.iter().enumerate() {
                    series[i][0].push(sample(r, integrated, size, camera, *point)?);
                    series[i][1].push(sample(r, history, size, camera, *point)?);
                }
                stops.push(r.post_pass.exposure.debug_state(&r.ctx.device, 0)[0]);
                if ablation != 1 {
                    let allocator = r.allocator();
                    let cached = r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
                        &mut allocator.lock().unwrap(), r.command_pool, r.ctx.queue, grey.0, 2)?[1];
                    radiosity.push(0.2126 * cached[0] + 0.7152 * cached[1] + 0.0722 * cached[2]);
                }
                let s = r.stats();
                for (sum, value) in timing.iter_mut().zip([s.gpu_gi_ms, s.gpu_gi_gather_ms, s.gpu_gi_denoise_ms,
                    s.gpu_gi_integrate_ms, s.gpu_gi_surface_ms]) { *sum += value / 180.0; }
            }
            let exposure = if auto { stops.iter().sum::<f32>() / stops.len() as f32 } else { 0.0 };
            let cached = if radiosity.is_empty() { (0.0, 0.0, 0.0) } else { stats(&radiosity) };
            eprintln!("[Flux low light] RT={rt} lamp={label} auto={auto} scale={scale:e} ablation={ablation} floor radiosity texel: \
                mean/scale={:.6} cv={:.4} step={:.4} | GI ms total={:.3} gather={:.3} denoise={:.3} integrate={:.3} surface={:.3}",
                cached.0 / scale, cached.1, cached.2, timing[0], timing[1], timing[2], timing[3], timing[4]);
            for (i, name) in ["near", "far"].iter().enumerate() {
                let (gather, shown) = (stats(&series[i][0]), stats(&series[i][1]));
                eprintln!("[Flux low light] RT={rt} lamp={label} auto={auto} scale={scale:e} ablation={ablation} {name}: \
                    integrate mean/scale={:.6} cv={:.4} step={:.4} | history mean/scale={:.6} cv={:.4} step={:.4} | stops={exposure:.2} traced={traced}/180 compile={waited:.1}s",
                    gather.0 / scale, gather.1, gather.2, shown.0 / scale, shown.1, shown.2);
            }
        }
    }
    r.gpu_gi.as_mut().unwrap().radiosity_reference = false;
    // Cost at the resolution Lumen's published budgets refer to.
    input.viewport_extent = Some([1920, 1080]);
    input.postfx.auto_exposure = crate::AutoExposure::default();
    input.lights = &configs[0].4;
    for rt in [false, true] {
        if rt && !r.ctx.ray_tracing() { continue; }
        r.set_flux_settings(settings(rt, true));
        r.invalidate_screen_gi();
        let mut timing = [0.0f32; 5];
        for frame in 0..360 {
            r.render(&input)?;
            if frame < 240 { continue; }
            let s = r.stats();
            for (sum, value) in timing.iter_mut().zip([s.gpu_gi_ms, s.gpu_gi_gather_ms, s.gpu_gi_denoise_ms,
                s.gpu_gi_integrate_ms, s.gpu_gi_surface_ms]) { *sum += value / 120.0; }
        }
        eprintln!("[Flux low light] 1920x1080 RT={rt}: GI ms total={:.3} gather={:.3} denoise={:.3} integrate={:.3} surface={:.3}",
            timing[0], timing[1], timing[2], timing[3], timing[4]);
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-pixel temporal noise in a dark room lit only by a small emissive panel.
// ---------------------------------------------------------------------------

/// Luminance of a block of an rgba16f image: rect = [x, y, width, height].
fn block(r: &Renderer, image: vk::Image, rect: [u32; 4]) -> anyhow::Result<Vec<f32>> {
    let [x, y, w, h] = rect;
    let device = &r.ctx.device;
    let allocator = r.allocator();
    let mut alloc = allocator.lock().unwrap();
    let mut host = crate::alloc::Buffer::new(device, &mut alloc, u64::from(w * h * 8), vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuToCpu, "Flux noise readback")?;
    crate::alloc::one_time_submit(device, r.command_pool, r.ctx.queue, |cb| unsafe {
        crate::image_barrier(device, cb, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 }).image_extent(vk::Extent3D { width: w, height: h, depth: 1 })]);
        crate::image_barrier(device, cb, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_READ);
    })?;
    let bytes = host.read();
    let half = |b: &[u8]| half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32();
    let out = bytes[..(w * h * 8) as usize].as_chunks::<8>().0.iter()
        .map(|b| 0.2126 * half(&b[0..2]) + 0.7152 * half(&b[2..4]) + 0.0722 * half(&b[4..6])).collect();
    host.destroy(device, &mut alloc);
    Ok(out)
}

#[derive(Clone, Copy, Default)]
struct Noise { mean: f32, pixel_cv: f32, pixel_cv_dt: f32, step: f32, region_cv: f32, region_cv_dt: f32, trend: f32 }

/// frames[t][pixel] -> mean, mean per-pixel temporal std/mean (raw and after removing a
/// linear trend), mean per-pixel frame-to-frame step relative to the mean, std/mean of the
/// region mean (raw and detrended), and the region-mean drift over the window.
fn noise(frames: &[Vec<f32>]) -> Noise {
    let t = frames.len() as f32;
    let pixels = frames[0].len();
    let tm = (t - 1.0) / 2.0;
    let tv = (0..frames.len()).map(|i| (i as f32 - tm).powi(2)).sum::<f32>();
    let detrended = |series: &dyn Fn(usize) -> f32| -> (f32, f32, f32) {
        // returns (mean, std raw, std after linear detrend)
        let mu = (0..frames.len()).map(series).sum::<f32>() / t;
        let var = (0..frames.len()).map(|i| (series(i) - mu).powi(2)).sum::<f32>() / t;
        let slope = (0..frames.len()).map(|i| (i as f32 - tm) * (series(i) - mu)).sum::<f32>() / tv;
        let resid = (0..frames.len()).map(|i| (series(i) - mu - slope * (i as f32 - tm)).powi(2)).sum::<f32>() / t;
        (mu, var.sqrt(), resid.sqrt())
    };
    let (mut cv, mut cv_dt, mut step, mut mean, mut counted) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..pixels {
        let (mu, sd, sd_dt) = detrended(&|k| frames[k][i]);
        if mu <= 1e-12 { continue; }
        cv += sd / mu; cv_dt += sd_dt / mu;
        step += frames.windows(2).map(|w| (w[1][i] - w[0][i]).abs()).sum::<f32>() / (t - 1.0) / mu;
        mean += mu; counted += 1.0;
    }
    let region: Vec<f32> = frames.iter().map(|f| f.iter().sum::<f32>() / pixels as f32).collect();
    let (rm, rsd, rsd_dt) = detrended(&|k| region[k]);
    let slope = (0..frames.len()).map(|i| (i as f32 - tm) * (region[i] - rm)).sum::<f32>() / tv;
    let c = counted.max(1.0);
    Noise { mean: mean / c, pixel_cv: cv / c, pixel_cv_dt: cv_dt / c, step: step / c,
        region_cv: rsd / rm.max(1e-12), region_cv_dt: rsd_dt / rm.max(1e-12), trend: slope * t / rm.max(1e-12) }
}

fn ppm(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) {
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = write!(f, "P6\n{w} {h}\n255\n");
        let _ = f.write_all(rgb);
    }
}

fn room(r: &mut Renderer, emitter: [f32; 2], radiance: f32) -> anyhow::Result<(Vec<crate::DrawCmd>, CameraData)> {
    use glam::Quat;
    let mesh = r.upload_mesh(&cube())?;
    let floor = lit(r, [0.8; 4], [0.0; 3], 1.0, 0.0);
    let wall = lit(r, [0.8; 4], [0.0; 3], 1.0, 0.0);
    let emission = [radiance, radiance, radiance * 1.15];
    let panel = lit(r, [0.0, 0.0, 0.0, 1.0], emission, 1.0, 0.0);
    r.set_material_features(panel, crate::MaterialFeatures { emission: true, ..Default::default() });
    let place = |size: Vec3, at: Vec3| Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, at);
    let mut draws = vec![
        draw(mesh, floor, place(Vec3::new(12.0, 0.2, 12.0), Vec3::new(0.0, -0.1, 0.0)), -1),
        draw(mesh, wall, place(Vec3::new(12.0, 5.0, 0.2), Vec3::new(0.0, 2.5, -6.1)), -1),
        draw(mesh, panel, place(Vec3::new(emitter[0], emitter[1], 0.1), Vec3::new(0.0, 1.5, -5.95)), -1),
    ];
    for (i, d) in draws.iter_mut().enumerate() {
        d.gi_source = Some(crate::GiSource { object: 0xbd00 + i as u128, slot: 0 });
    }
    let volume = super::flux::cube_distance_field();
    let materials = [
        GpuGiMaterial { albedo: [0.8; 3], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
        GpuGiMaterial { albedo: [0.8; 3], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
        GpuGiMaterial { albedo: [0.0; 3], emission, metallic: 0.0, roughness: 1.0 },
    ];
    let fields: Vec<_> = draws.iter().zip(materials).map(|(d, material)| crate::GiMeshSdf {
        gi_source: d.gi_source, world_to_local: d.transform.inverse(), volume: volume.clone(),
        in_global_field: false, material }).collect();
    r.gi_set_mesh_sdfs(&fields)?;
    r.gi_set_gdf(&vec![100.0; 512], &vec![0; 512], [8; 3], [-30.0; 3], [30.0; 3], &[materials[0]]);
    r.gi_set_emitters(&[]);
    r.set_gi_enabled(true);
    let eye = Vec3::new(0.0, 1.7, 4.5);
    let camera = CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::new(0.0, 0.5, -2.0), Vec3::Y),
        proj: Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0) };
    Ok((draws, camera))
}

fn floor_rect(camera: CameraData, size: [u32; 2]) -> [u32; 4] {
    project_rect(camera, size, [Vec3::new(-1.2, 0.0, -1.5), Vec3::new(1.2, 0.0, -1.5), Vec3::new(-1.2, 0.0, 0.5), Vec3::new(1.2, 0.0, 0.5)])
}


/// FluxSettings for the diagnostics. FLUX_NOISE_BALANCED=1 uses the editor's default quality
/// (Balanced: 32 rays per probe, firefly clamp 4); otherwise FluxSettings::default() like the other tests.
/// FLUX_NOISE_PRESET=performance is the Performance preset (16 rays per probe on an 8 px grid, a quarter of
/// the radiosity rays at half the update rate) with four bounces, firefly clamp 0.5 and emissive boost 8,
/// as in the LongDark project. FLUX_NOISE_SPACING overrides the probe grid in pixels (2..=32).
fn settings(rt: bool, screen_traces: bool) -> FluxSettings {
    let balanced = std::env::var("FLUX_NOISE_BALANCED").is_ok_and(|v| v == "1");
    let mut settings = FluxSettings { rt_trace: rt, reflection_mode: 0, ssr_enabled: false, gi_screen_traces: screen_traces,
        samples: if balanced { 32 } else { FluxSettings::default().samples },
        firefly_clamp: if balanced { 4.0 } else { FluxSettings::default().firefly_clamp }, ..Default::default() };
    if std::env::var("FLUX_NOISE_PRESET").is_ok_and(|v| v == "performance") {
        settings = FluxSettings { samples: 16, probe_spacing: 8, scene_lighting_quality: 0.5, scene_lighting_update_speed: 0.5,
            radiance_cache_budget: 16, bounces: 4, firefly_clamp: 0.5, emissive_boost: 8.0, smoothing: 0.5, ..settings };
    }
    if let Some(spacing) = std::env::var("FLUX_NOISE_SPACING").ok().and_then(|v| v.parse::<u32>().ok()) { settings.probe_spacing = spacing.clamp(2, 32); }
    // FLUX_NOISE_LIGHTING_QUALITY: Scene Lighting Quality (0.5 = 4 radiosity rays per receiver and update, 1 = 16).
    if let Some(quality) = std::env::var("FLUX_NOISE_LIGHTING_QUALITY").ok().and_then(|v| v.parse::<f32>().ok()) { settings.scene_lighting_quality = quality.clamp(0.5, 8.0); }
    settings
}

/// Pixel bounding box of four world-space corners.
fn project_rect(camera: CameraData, size: [u32; 2], corners: [Vec3; 4]) -> [u32; 4] {
    let vp = camera.proj * camera.view;
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in corners {
        let q = vp.project_point3(p);
        let (x, y) = ((q.x * 0.5 + 0.5) * size[0] as f32, (0.5 - q.y * 0.5) * size[1] as f32);
        lo = [lo[0].min(x), lo[1].min(y)]; hi = [hi[0].max(x), hi[1].max(y)];
    }
    let (x, y) = (lo[0].max(0.0) as u32, lo[1].max(0.0) as u32);
    [x, y, (hi[0].min(size[0] as f32) as u32).saturating_sub(x).max(4), (hi[1].min(size[1] as f32) as u32).saturating_sub(y).max(4)]
}

/// A closed 8 x 3 x 8 m room: floor, ceiling, four walls, and an emissive panel on the back wall.
fn closed_room(r: &mut Renderer, panel: [f32; 2], radiance: f32) -> anyhow::Result<(Vec<crate::DrawCmd>, CameraData, crate::MaterialHandle)> {
    let (draws, camera, screen, _) = closed_room_with_prop(r, panel, radiance, false)?;
    Ok((draws, camera, screen))
}

fn closed_room_with_prop(r: &mut Renderer, panel: [f32; 2], radiance: f32, prop: bool) -> anyhow::Result<(Vec<crate::DrawCmd>, CameraData, crate::MaterialHandle, Vec<crate::GiMeshSdf>)> {
    use glam::Quat;
    let mesh = r.upload_mesh(&cube())?;
    let surface = lit(r, [0.75, 0.72, 0.68, 1.0], [0.0; 3], 1.0, 0.0);
    let emission = [radiance, radiance, radiance * 1.15];
    let screen = lit(r, [0.0, 0.0, 0.0, 1.0], emission, 1.0, 0.0);
    r.set_material_features(screen, crate::MaterialFeatures { emission: true, ..Default::default() });
    let place = |size: Vec3, at: Vec3| Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, at);
    let mut draws = vec![
        draw(mesh, surface, place(Vec3::new(8.4, 0.2, 8.4), Vec3::new(0.0, -0.1, 0.0)), -1),
        draw(mesh, surface, place(Vec3::new(8.4, 0.2, 8.4), Vec3::new(0.0, 3.1, 0.0)), -1),
        draw(mesh, surface, place(Vec3::new(8.4, 3.2, 0.2), Vec3::new(0.0, 1.5, -4.1)), -1),
        draw(mesh, surface, place(Vec3::new(8.4, 3.2, 0.2), Vec3::new(0.0, 1.5, 4.1)), -1),
        draw(mesh, surface, place(Vec3::new(0.2, 3.2, 8.4), Vec3::new(-4.1, 1.5, 0.0)), -1),
        draw(mesh, surface, place(Vec3::new(0.2, 3.2, 8.4), Vec3::new(4.1, 1.5, 0.0)), -1),
        draw(mesh, screen, place(Vec3::new(panel[0], panel[1], 0.1), Vec3::new(0.0, 1.4, -3.95)), -1),
    ];
    if prop { draws.push(draw(mesh, surface, place(Vec3::splat(0.4), Vec3::new(-3.0, 0.2, 3.0)), -1)); }
    for (i, d) in draws.iter_mut().enumerate() {
        d.gi_source = Some(crate::GiSource { object: 0xbe00 + i as u128, slot: 0 });
    }
    let volume = super::flux::cube_distance_field();
    let wall = GpuGiMaterial { albedo: [0.75, 0.72, 0.68], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 };
    let panel_material = GpuGiMaterial { albedo: [0.0; 3], emission, metallic: 0.0, roughness: 1.0 };
    let fields: Vec<_> = draws.iter().enumerate().map(|(i, d)| crate::GiMeshSdf {
        gi_source: d.gi_source, world_to_local: d.transform.inverse(), volume: volume.clone(), in_global_field: false,
        material: if i == 6 { panel_material } else { wall } }).collect();
    r.gi_set_mesh_sdfs(&fields)?;
    r.gi_set_gdf(&vec![100.0; 512], &vec![0; 512], [8; 3], [-30.0; 3], [30.0; 3], &[wall]);
    r.gi_set_emitters(&[]);
    r.set_gi_enabled(true);
    let eye = Vec3::new(0.0, 1.5, 3.5);
    let camera = CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::new(0.0, 1.1, -4.0), Vec3::Y),
        proj: Mat4::perspective_rh(1.1, 16.0 / 9.0, 0.1, 100.0) };
    Ok((draws, camera, screen, fields))
}

struct Measured { gi_history: Vec<Noise>, gi_integrate: Vec<Noise>, display: Vec<Noise>, stops: f32, stops_std: f32, gi_series: Vec<Vec<f32>> }

fn measure(r: &mut Renderer, input: &mut crate::FrameInput<'_>, rects: &[[u32; 4]], clock: &mut f64,
    tag: &str, dir: &Option<std::path::PathBuf>, warmup: usize) -> anyhow::Result<Measured> {
    measure_with(r, input, rects, clock, tag, dir, warmup, 150, &mut |_, _, _| {})
}

#[allow(clippy::too_many_arguments)]
fn measure_with(r: &mut Renderer, input: &mut crate::FrameInput<'_>, rects: &[[u32; 4]], clock: &mut f64,
    tag: &str, dir: &Option<std::path::PathBuf>, warmup: usize, frames: usize, animate: &mut dyn FnMut(&mut Renderer, usize, &mut crate::FrameInput<'_>)) -> anyhow::Result<Measured> {
    r.invalidate_screen_gi();
    await_pipelines(r, input)?;
    let kept = 60usize.min(frames);
    let regions = rects.len();
    let mut history: Vec<Vec<Vec<f32>>> = vec![Vec::new(); regions];
    let mut integrate = history.clone();
    let mut display = history.clone();
    let mut stops = Vec::new();
    let mut images: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    for frame in 0..warmup + frames {
        *clock += 1.0 / 60.0;
        r.post_pass.exposure.time_override = Some(*clock);
        animate(r, frame, input);
        r.render(input)?;
        if frame < warmup { continue; }
        let target = r.viewport_target.as_ref().unwrap();
        let (integrate_image, history_image) = (target.sgi.gi_final, target.sgi.full_history[(r.sgi_parity ^ 1) & 1].image);
        let image = r.debug_read_viewport()?.unwrap();
        for (k, rect) in rects.iter().enumerate() {
            history[k].push(block(r, history_image, *rect)?);
            integrate[k].push(block(r, integrate_image, *rect)?);
            let [x, y, w, h] = *rect;
            let mut shown = Vec::with_capacity((w * h) as usize);
            for yy in y..y + h { for xx in x..x + w {
                let i = ((yy * image.0 + xx) * 4) as usize;
                shown.push((0.2126 * image.2[i] as f32 + 0.7152 * image.2[i + 1] as f32 + 0.0722 * image.2[i + 2] as f32) / 255.0);
            } }
            display[k].push(shown);
        }
        stops.push(r.post_pass.exposure.debug_state(&r.ctx.device, 0)[0]);
        if frame >= warmup + frames - kept { images.push(image); }
    }
    if let Some(dir) = dir {
        let (w, h) = (images[0].0, images[0].1);
        let lum = |image: &(u32, u32, Vec<u8>), i: usize| (0.2126 * image.2[i * 4] as f32 + 0.7152 * image.2[i * 4 + 1] as f32
            + 0.0722 * image.2[i * 4 + 2] as f32) / 255.0;
        let n = (w * h) as usize;
        let (mut mean, mut std) = (vec![0.0f32; n], vec![0.0f32; n]);
        for image in &images { for (i, value) in mean.iter_mut().enumerate() { *value += lum(image, i) / images.len() as f32; } }
        for image in &images { for i in 0..n { std[i] += (lum(image, i) - mean[i]).powi(2) / images.len() as f32; } }
        let to_rgb = |f: &dyn Fn(usize) -> f32| -> Vec<u8> { (0..n).flat_map(|i| { let v = (f(i).clamp(0.0, 1.0) * 255.0) as u8; [v, v, v] }).collect() };
        ppm(&dir.join(format!("{tag}_mean.ppm")), w, h, &to_rgb(&|i| mean[i]));
        // Relative temporal std, so dark areas are not hidden: 100 percent = full white.
        ppm(&dir.join(format!("{tag}_relstd.ppm")), w, h, &to_rgb(&|i| std[i].sqrt() / mean[i].max(0.02)));
        for k in 0..4 {
            let image = &images[images.len() - 4 + k];
            let rgb: Vec<u8> = image.2.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
            ppm(&dir.join(format!("{tag}_frame{k}.ppm")), w, h, &rgb);
        }
    }
    let mean = stops.iter().sum::<f32>() / stops.len() as f32;
    let stops_std = (stops.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / stops.len() as f32).sqrt();
    Ok(Measured {
        gi_history: history.iter().map(|f| noise(f)).collect(),
        gi_integrate: integrate.iter().map(|f| noise(f)).collect(),
        display: display.iter().map(|f| noise(f)).collect(),
        stops: mean, stops_std,
        gi_series: history.iter().map(|f| f.iter().map(|px| px.iter().sum::<f32>() / px.len() as f32).collect()).collect(),
    })
}

fn report(tag: &str, names: &[&str], m: &Measured) {
    for (k, name) in names.iter().enumerate() {
        let (g, i, d) = (&m.gi_history[k], &m.gi_integrate[k], &m.display[k]);
        eprintln!("[Flux noise] {tag} [{name}]: GI final mean={:.5} pixel_cv={:.4} (detrended {:.4}) region_cv={:.4} (detrended {:.4}) trend={:+.4} step={:.4} | \
            integrate pixel_cv={:.4} | display mean={:.3} pixel_cv={:.4} region_cv={:.4} | stops={:.2}+-{:.3}",
            g.mean, g.pixel_cv, g.pixel_cv_dt, g.region_cv, g.region_cv_dt, g.trend, g.step,
            i.pixel_cv, d.mean, d.pixel_cv, d.region_cv, m.stops, m.stops_std);
    }
}

pub(super) fn noise_run(r: &mut Renderer) -> anyhow::Result<()> {
    let dir = std::env::var("FLUX_NOISE_DIR").ok().map(std::path::PathBuf::from);
    if let Some(dir) = &dir { std::fs::create_dir_all(dir)?; }
    let only = std::env::var("FLUX_NOISE_SCENE").ok();
    let mut clock = 500.0f64;
    let size = [640u32, 360];
    for scene in ["closed", "tv"] {
        if only.as_deref().is_some_and(|o| o != scene) { continue; }
        let (draws, camera, names, rects): (Vec<crate::DrawCmd>, CameraData, Vec<&str>, Vec<[u32; 4]>) = if scene == "closed" {
            let (draws, camera, _) = closed_room(r, [1.6, 0.9], 8.0)?;
            let rects = vec![
                project_rect(camera, size, [Vec3::new(-1.5, 0.0, -2.0), Vec3::new(1.5, 0.0, -2.0), Vec3::new(-1.5, 0.0, 1.0), Vec3::new(1.5, 0.0, 1.0)]),
                project_rect(camera, size, [Vec3::new(-4.0, 0.4, -1.0), Vec3::new(-4.0, 2.0, -1.0), Vec3::new(-4.0, 0.4, 1.5), Vec3::new(-4.0, 2.0, 1.5)]),
                project_rect(camera, size, [Vec3::new(-1.5, 3.0, -2.5), Vec3::new(1.5, 3.0, -2.5), Vec3::new(-1.5, 3.0, 0.0), Vec3::new(1.5, 3.0, 0.0)]),
            ];
            (draws, camera, vec!["floor", "left wall", "ceiling"], rects)
        } else {
            let (draws, camera) = room(r, [1.6, 0.9], 8.0)?;
            let rect = floor_rect(camera, [640, 360]);
            (draws, camera, vec!["floor"], vec![rect])
        };
        let lamp = [crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(0.0, 2.7, 0.0), direction: Vec3::NEG_Y,
            color: [1.0, 0.85, 0.6], intensity: if scene == "closed" { 0.6 } else { 0.0 }, range: 9.0, spot_inner_deg: 30.0,
            spot_outer_deg: 45.0, cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false }];
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([size[0], size[1]]);
        input.draws = &draws;
        if scene == "closed" { input.lights = &lamp; }
        input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
        input.draw_skybox = false;
        input.postfx.bloom_enabled = false;
        eprintln!("[Flux noise] scene={scene} regions={names:?} rects={rects:?}");
        for rt in [false, true] {
            if rt && !r.ctx.ray_tracing() { continue; }
            r.set_flux_settings(settings(rt, true));
            r.gpu_gi.as_mut().unwrap().temporal_override = None;
            r.gpu_gi.as_mut().unwrap().radiosity_reference = false;
            let backend = if rt { "hw" } else { "sw" };
            input.postfx.exposure = 0.0;
            input.postfx.auto_exposure = crate::AutoExposure { enabled: true, ..Default::default() };
            // Long first warm-up: caches (radiosity multi-bounce, radiance cache) and exposure converge.
            let auto = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_auto"), &dir, 3000)?;
            report(&format!("scene={scene} {backend} auto-exposure"), &names, &auto);
            let auto2 = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_auto2"), &None, 600)?;
            report(&format!("scene={scene} {backend} auto-exposure (repeat)"), &names, &auto2);
            // The same display gain, held fixed: no adaptation dynamics.
            input.postfx.auto_exposure = crate::AutoExposure::default();
            input.postfx.exposure = auto2.stops;
            let manual = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_manual"), &dir, 600)?;
            report(&format!("scene={scene} {backend} manual {:+.2} stops", auto2.stops), &names, &manual);
            input.postfx.exposure = 0.0;
            let none = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_none"), &dir, 600)?;
            report(&format!("scene={scene} {backend} no exposure"), &names, &none);
            // The radiosity clamp + filter versus the unclamped, unfiltered reference, same manual exposure.
            r.gpu_gi.as_mut().unwrap().radiosity_reference = true;
            input.postfx.exposure = auto2.stops;
            let reference = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_ref"), &dir, 1500)?;
            report(&format!("scene={scene} {backend} manual {:+.2} stops, radiosity reference (no clamp/filter)", auto2.stops), &names, &reference);
            r.gpu_gi.as_mut().unwrap().radiosity_reference = false;
            input.postfx.exposure = auto2.stops;
            let again = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_manual2"), &None, 1500)?;
            report(&format!("scene={scene} {backend} manual {:+.2} stops (repeat after reference)", auto2.stops), &names, &again);
            let auto = auto2;
            if scene == "closed" {
                input.postfx.auto_exposure = crate::AutoExposure::default();
                input.postfx.exposure = auto.stops;
                for (name, over) in [("final alpha 1.0 (no final history)", [0.06, 1.0, 1.0, 0.0]),
                    ("final alpha 0.25", [0.06, 8.0, 0.25, 0.0]), ("final alpha 0.015", [0.06, 128.0, 0.015, 0.0]),
                    ("probe alpha 1.0 (no probe history)", [1.0, 32.0, 0.06, 0.0]),
                    ("probe alpha 0.25", [0.25, 32.0, 0.06, 0.0]), ("probe alpha 0.02", [0.02, 32.0, 0.06, 0.0])] {
                    r.gpu_gi.as_mut().unwrap().temporal_override = Some(over);
                    let m = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_sweep"), &None, 300)?;
                    report(&format!("scene={scene} {backend} {name}"), &names, &m);
                }
                r.gpu_gi.as_mut().unwrap().temporal_override = None;
            }
            // Final-history neighbourhood clamp off (test switch bit 0), screen traces off, and the clamp at other alphas.
            input.postfx.auto_exposure = crate::AutoExposure::default();
            input.postfx.exposure = auto.stops;
            for (name, flags, over, screen_traces) in [
                ("history clamp ON, final alpha 0.25", 0u32, Some([0.06, 8.0, 0.25, 0.0]), true),
                ("history clamp ON, final alpha 0.015", 0, Some([0.06, 128.0, 0.015, 0.0]), true),
                ("history clamp OFF, default alphas", 1, None, true),
                ("history clamp OFF, final alpha 0.25", 1, Some([0.06, 8.0, 0.25, 0.0]), true),
                ("history clamp OFF, final alpha 0.015", 1, Some([0.06, 128.0, 0.015, 0.0]), true),
                ("screen traces OFF, default", 0, None, false)] {
                r.set_flux_settings(settings(rt, screen_traces));
                r.gpu_gi.as_mut().unwrap().debug_flags = flags;
                r.gpu_gi.as_mut().unwrap().temporal_override = over;
                let m = measure(r, &mut input, &rects, &mut clock, &format!("{scene}_{backend}_ablate"), &None, 600)?;
                report(&format!("scene={scene} {backend} {name}"), &names, &m);
            }
            r.set_flux_settings(settings(rt, true));
            r.gpu_gi.as_mut().unwrap().debug_flags = 0;
            r.gpu_gi.as_mut().unwrap().temporal_override = None;
        }
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

/// Video-like brightness: a new level every 6..14 frames (+-15 percent frame jitter), mean 1.
fn tv_signal(frames: usize) -> Vec<f32> {
    let mut state = 0x2545f491u32;
    let mut next = move || { state ^= state << 13; state ^= state >> 17; state ^= state << 5; (state >> 8) as f32 / (1u32 << 24) as f32 };
    let mut out = Vec::with_capacity(frames);
    while out.len() < frames {
        let (level, length) = (0.2 + 1.6 * next(), 6 + (next() * 9.0) as usize);
        for _ in 0..length { out.push(level * (0.85 + 0.3 * next())); }
    }
    out.truncate(frames);
    let mean = out.iter().sum::<f32>() / frames as f32;
    out.iter().map(|v| v / mean).collect()
}

/// Correlation of the emission with the room GI at lags 0..=60 frames (GI lagging the emission).
fn lag(emission: &[f32], gi: &[f32]) -> (usize, f32) {
    let norm = |x: &[f32]| { let m = x.iter().sum::<f32>() / x.len() as f32; let v = x.iter().map(|a| (a - m).powi(2)).sum::<f32>().sqrt(); x.iter().map(|a| (a - m) / v.max(1e-20)).collect::<Vec<_>>() };
    let mut best = (0usize, -2.0f32);
    for l in 0..=60usize {
        let n = emission.len() - l;
        let (a, b) = (norm(&emission[..n]), norm(&gi[l..l + n]));
        let c = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>();
        if c > best.1 { best = (l, c); }
    }
    best
}

pub(super) fn animated_run(r: &mut Renderer) -> anyhow::Result<()> {
    let mut clock = 900.0f64;
    let size = [640u32, 360];
    let (draws, camera, screen) = closed_room(r, [1.6, 0.9], 8.0)?;
    let rects = vec![
        project_rect(camera, size, [Vec3::new(-1.5, 0.0, -2.0), Vec3::new(1.5, 0.0, -2.0), Vec3::new(-1.5, 0.0, 1.0), Vec3::new(1.5, 0.0, 1.0)]),
        project_rect(camera, size, [Vec3::new(-4.0, 0.4, -1.0), Vec3::new(-4.0, 2.0, -1.0), Vec3::new(-4.0, 0.4, 1.5), Vec3::new(-4.0, 2.0, 1.5)]),
    ];
    let names = ["floor", "left wall"];
    let lamp = [crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(0.0, 2.7, 0.0), direction: Vec3::NEG_Y,
        color: [1.0, 0.85, 0.6], intensity: 0.0, range: 9.0, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
        cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false }];
    let mut input = crate::cascade_tests::input(camera);
    input.viewport_extent = Some([size[0], size[1]]);
    input.draws = &draws;
    input.lights = &lamp;
    input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
    input.draw_skybox = false;
    input.postfx.bloom_enabled = false;
    let (warmup, frames) = (900usize, 400usize);
    let signal = tv_signal(warmup + frames + 10);
    let emission_cv = { let w = &signal[warmup..warmup + frames]; let m = w.iter().sum::<f32>() / w.len() as f32;
        (w.iter().map(|v| (v - m).powi(2)).sum::<f32>() / w.len() as f32).sqrt() / m };
    eprintln!("[Flux anim] emission cv={emission_cv:.3}; white-noise transmission of one EMA(0.06) would be {:.3}", (0.06f32 / (2.0 - 0.06)).sqrt());
    for rt in [false, true] {
        if rt && !r.ctx.ray_tracing() { continue; }
        r.set_flux_settings(settings(rt, true));
        let backend = if rt { "hw" } else { "sw" };
        input.postfx.exposure = 3.0;
        input.postfx.auto_exposure = crate::AutoExposure::default();
        for (name, flags, reference, over) in [("default", 0u32, false, None), ("no history clamp", 1, false, None),
            ("radiosity reference", 0, true, None), ("final alpha 0.015 + probe alpha 0.02", 0, false, Some([0.02, 128.0, 0.015, 0.0])),
            ("no history clamp, alphas 0.02/0.015", 1, false, Some([0.02, 128.0, 0.015, 0.0]))] {
            r.gpu_gi.as_mut().unwrap().debug_flags = flags;
            r.gpu_gi.as_mut().unwrap().radiosity_reference = reference;
            r.gpu_gi.as_mut().unwrap().temporal_override = over;
            let sig = signal.clone();
            let m = measure_with(r, &mut input, &rects, &mut clock, &format!("anim_{backend}"), &None, warmup, frames, &mut |r, frame, _| {
                let v = sig[frame];
                r.material_params_mut(screen).emission_color = [8.0 * v, 8.0 * v, 9.2 * v];
            })?;
            for (k, region) in names.iter().enumerate() {
                let (best_lag, correlation) = lag(&signal[warmup..warmup + frames], &m.gi_series[k]);
                eprintln!("[Flux anim] {backend} {name} [{region}]: GI mean={:.5} region_cv(detrended)={:.4} transmission={:.3} lag={best_lag} frames (corr {correlation:.2}) | pixel_cv={:.4} | display region_cv={:.4}",
                    m.gi_history[k].mean, m.gi_history[k].region_cv_dt, m.gi_history[k].region_cv_dt / emission_cv,
                    m.gi_history[k].pixel_cv_dt, m.display[k].region_cv_dt);
            }
        }
        r.gpu_gi.as_mut().unwrap().debug_flags = 0;
        r.gpu_gi.as_mut().unwrap().radiosity_reference = false;
        r.gpu_gi.as_mut().unwrap().temporal_override = None;
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

/// Camera panning at ~0.5 m/s through a dim room: relative error of the moving GI image against a
/// converged render of the same pose (noise + history lag + disocclusion), per configuration.
pub(super) fn motion_run(r: &mut Renderer) -> anyhow::Result<()> {
    let mut clock = 1500.0f64;
    let size = [640u32, 360];
    let tv = std::env::var("FLUX_NOISE_SCENE").is_ok_and(|v| v == "tv");
    let (draws, camera) = if tv { room(r, [1.6, 0.9], 8.0)? } else { let (d, c, _) = closed_room(r, [1.6, 0.9], 8.0)?; (d, c) };
    let pose = |x: f32| {
        let (eye, target) = if tv { (Vec3::new(x, 1.7, 4.5), Vec3::new(x * 0.3, 0.5, -2.0)) } else { (Vec3::new(x, 1.5, 3.5), Vec3::new(x * 0.3, 1.1, -4.0)) };
        CameraData { position: eye, view: Mat4::look_at_rh(eye, target, Vec3::Y), ..camera }
    };
    let lamp = [crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(0.0, 2.7, 0.0), direction: Vec3::NEG_Y,
        color: [1.0, 0.85, 0.6], intensity: if tv { 0.0 } else { 0.6 }, range: 9.0, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
        cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false }];
    let mut input = crate::cascade_tests::input(camera);
    input.viewport_extent = Some([size[0], size[1]]);
    input.draws = &draws;
    input.lights = &lamp;
    input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
    input.draw_skybox = false;
    input.postfx.bloom_enabled = false;
    input.postfx.exposure = if tv { 5.0 } else { 1.5 };
    input.postfx.auto_exposure = crate::AutoExposure::default();
    let full = [0u32, 0, size[0], size[1]];
    let (start, per_pose, poses) = (-1.2f32, 48usize, [0usize, 1, 2, 3, 4]);
    let pose_x = |k: usize| start + 0.4 * (k as f32 + 1.0) * 1.0;
    for rt in [false, true] {
        if rt && !r.ctx.ray_tracing() { continue; }
        r.set_flux_settings(settings(rt, true));
        let backend = if rt { "hw" } else { "sw" };
        // Converged references, one per pose, rendered once per backend with the default configuration.
        let mut reference = Vec::new();
        for &k in &poses {
            input.camera = pose(pose_x(k));
            r.invalidate_screen_gi();
            await_pipelines(r, &input)?;
            for _ in 0..500 { clock += 1.0 / 60.0; r.post_pass.exposure.time_override = Some(clock); r.render(&input)?; }
            let target = r.viewport_target.as_ref().unwrap();
            reference.push(block(r, target.sgi.full_history[(r.sgi_parity ^ 1) & 1].image, full)?);
        }
        for (name, flags, over) in [("default", 0u32, None), ("history clamp OFF", 1, None),
            ("final alpha 0.03 / probe 0.03", 0, Some([0.03, 64.0, 0.03, 0.0])), ("final 0.015 / probe 0.02", 0, Some([0.02, 128.0, 0.015, 0.0])),
            ("clamp OFF + final 0.015 / probe 0.02", 1, Some([0.02, 128.0, 0.015, 0.0]))] {
            r.gpu_gi.as_mut().unwrap().debug_flags = flags;
            r.gpu_gi.as_mut().unwrap().temporal_override = over;
            input.camera = pose(start);
            r.invalidate_screen_gi();
            for _ in 0..600 { clock += 1.0 / 60.0; r.post_pass.exposure.time_override = Some(clock); r.render(&input)?; }
            let mut errors = Vec::new();
            for step in 1..=(per_pose * 5) {
                let x = start + 0.4 / per_pose as f32 * step as f32;
                input.camera = pose(x);
                clock += 1.0 / 60.0; r.post_pass.exposure.time_override = Some(clock);
                r.render(&input)?;
                if step % per_pose == 0 {
                    let k = step / per_pose - 1;
                    let target = r.viewport_target.as_ref().unwrap();
                    let image = block(r, target.sgi.full_history[(r.sgi_parity ^ 1) & 1].image, full)?;
                    let (mut sum, mut mean, mut n) = (0.0f32, 0.0f32, 0.0f32);
                    let reference_mean = reference[k].iter().sum::<f32>() / reference[k].len() as f32;
                    for (a, b) in image.iter().zip(&reference[k]) {
                        if *b > reference_mean * 0.3 { sum += (a - b).abs() / b; mean += a / b; n += 1.0; }
                    }
                    errors.push((sum / n, mean / n));
                }
            }
            let relative = errors.iter().map(|e| e.0).sum::<f32>() / errors.len() as f32;
            let bias = errors.iter().map(|e| e.1).sum::<f32>() / errors.len() as f32 - 1.0;
            eprintln!("[Flux motion] {backend} {name}: mean |error|/reference over visible surfaces = {relative:.4}, mean bias = {bias:+.4}, per pose {:?}",
                errors.iter().map(|e| (e.0 * 1e4).round() / 1e4).collect::<Vec<_>>());
        }
        r.gpu_gi.as_mut().unwrap().debug_flags = 0;
        r.gpu_gi.as_mut().unwrap().temporal_override = None;
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

/// The engine re-publishes every gather instance each frame. Does one object moving anywhere in
/// the scene (a character, a physics body, jitter of 0.1 mm) cost the whole screen its GI history?
pub(super) fn prop_run(r: &mut Renderer) -> anyhow::Result<()> {
    let mut clock = 2500.0f64;
    let size = [640u32, 360];
    for (label, emitter, radiance) in [("TV-sized emitter", [1.6f32, 0.9], 8.0f32), ("bulb-sized emitter", [0.25, 0.25], 8.0 * 1.6 * 0.9 / (0.25 * 0.25))] {
        let (mut draws, camera, _screen, mut fields) = closed_room_with_prop(r, emitter, radiance, true)?;
        let lamp = [crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(0.0, 2.7, 0.0), direction: Vec3::NEG_Y,
            color: [1.0, 0.85, 0.6], intensity: 0.6, range: 9.0, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
            cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false }];
        let rects = vec![
            project_rect(camera, size, [Vec3::new(-1.5, 0.0, -2.0), Vec3::new(1.5, 0.0, -2.0), Vec3::new(-1.5, 0.0, 1.0), Vec3::new(1.5, 0.0, 1.0)]),
            project_rect(camera, size, [Vec3::new(-4.0, 0.4, -1.0), Vec3::new(-4.0, 2.0, -1.0), Vec3::new(-4.0, 0.4, 1.5), Vec3::new(-4.0, 2.0, 1.5)]),
        ];
        let names = ["floor", "left wall"];
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([size[0], size[1]]);
        input.lights = &lamp;
        input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
        input.draw_skybox = false;
        input.postfx.bloom_enabled = false;
        input.postfx.exposure = 1.5;
        input.postfx.auto_exposure = crate::AutoExposure::default();
        let prop = draws.len() - 1;
        let home = Vec3::new(-3.0, 0.2, 3.0);
        for rt in [false, true] {
            if rt && !r.ctx.ray_tracing() { continue; }
            r.set_flux_settings(settings(rt, true));
            let backend = if rt { "hw" } else { "sw" };
            for (name, mode) in [("prop republished each frame, not moving", 0u32), ("prop moving 0.4 m circle", 1), ("prop jitter 0.1 mm", 2), ("nothing republished (static)", 3)] {
                let mut base_draws = draws.clone();
                let m = measure_with(r, &mut input, &rects, &mut clock, &format!("prop_{backend}"), &None, 600, 200, &mut |r, frame, input| {
                    let t = frame as f32;
                    let at = match mode {
                        1 => home + Vec3::new(0.4 * (t * 0.02).sin(), 0.0, 0.4 * (t * 0.02).cos()),
                        2 => home + Vec3::new(1e-4 * (t * 1.7).sin(), 0.0, 1e-4 * (t * 2.3).cos()),
                        _ => home,
                    };
                    base_draws[prop].transform = Mat4::from_scale_rotation_translation(Vec3::splat(0.4), glam::Quat::IDENTITY, at);
                    fields[prop].world_to_local = base_draws[prop].transform.inverse();
                    if mode != 3 { r.gi_set_mesh_sdfs(&fields).unwrap(); }
                    input.draws = Box::leak(base_draws.clone().into_boxed_slice());
                })?;
                draws[prop].transform = Mat4::from_scale_rotation_translation(Vec3::splat(0.4), glam::Quat::IDENTITY, home);
                fields[prop].world_to_local = draws[prop].transform.inverse();
                r.gi_set_mesh_sdfs(&fields)?;
                for (k, region) in names.iter().enumerate() {
                    eprintln!("[Flux prop] {label} {backend} {name} [{region}]: GI mean={:.5} pixel_cv(detrended)={:.4} step={:.4} | integrate pixel_cv={:.4} | display pixel_cv={:.4}",
                        m.gi_history[k].mean, m.gi_history[k].pixel_cv_dt, m.gi_history[k].step, m.gi_integrate[k].pixel_cv_dt, m.display[k].pixel_cv_dt);
                }
            }
        }
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

/// Means over `cell` x `cell` pixel blocks of each `w`-wide frame. Per-pixel grain averages out; what is
/// left is the blob-sized wobble of the indirect light (turbulence).
fn blobs(frames: &[Vec<f32>], w: u32, cell: u32) -> Vec<Vec<f32>> {
    let h = frames[0].len() as u32 / w;
    let (bw, bh) = ((w / cell).max(1), (h / cell).max(1));
    frames.iter().map(|f| {
        let (mut sum, mut n) = (vec![0.0f32; (bw * bh) as usize], vec![0.0f32; (bw * bh) as usize]);
        for y in 0..h.min(bh * cell) { for x in 0..w.min(bw * cell) {
            let at = ((y / cell) * bw + x / cell) as usize;
            sum[at] += f[(y * w + x) as usize]; n[at] += 1.0;
        } }
        sum.iter().zip(&n).map(|(s, n)| s / n.max(1.0)).collect()
    }).collect()
}

/// Mean lag-1 autocorrelation of the linearly detrended series: near 1 is a slow drift, near 0 is flicker.
fn persistence(frames: &[Vec<f32>]) -> f32 {
    let t = frames.len() as f32;
    let tm = (t - 1.0) / 2.0;
    let tv = (0..frames.len()).map(|i| (i as f32 - tm).powi(2)).sum::<f32>();
    let (mut sum, mut n) = (0.0f32, 0.0f32);
    for i in 0..frames[0].len() {
        let mu = frames.iter().map(|f| f[i]).sum::<f32>() / t;
        let slope = frames.iter().enumerate().map(|(k, f)| (k as f32 - tm) * (f[i] - mu)).sum::<f32>() / tv;
        let r: Vec<f32> = frames.iter().enumerate().map(|(k, f)| f[i] - mu - slope * (k as f32 - tm)).collect();
        let var = r.iter().map(|v| v * v).sum::<f32>();
        if var <= 1e-20 { continue; }
        sum += r.windows(2).map(|w| w[0] * w[1]).sum::<f32>() / var; n += 1.0;
    }
    sum / n.max(1.0)
}

/// The `rect` = [x, y, width, height] part of a `w`-wide frame.
fn crop(frame: &[f32], w: u32, rect: [u32; 4]) -> Vec<f32> {
    let [x, y, rw, rh] = rect;
    (y..y + rh).flat_map(|yy| frame[(yy * w + x) as usize..(yy * w + x + rw) as usize].iter().copied()).collect()
}

/// Per-pixel mean and temporal std (linear trend removed) of whole frames, accumulated without keeping them.
#[derive(Default)]
struct PixelMap { n: f32, sum: Vec<f32>, squares: Vec<f32>, ramp: Vec<f32> }
impl PixelMap {
    fn add(&mut self, frame: &[f32]) {
        if self.sum.is_empty() { self.sum = vec![0.0; frame.len()]; self.squares = self.sum.clone(); self.ramp = self.sum.clone(); }
        for (i, v) in frame.iter().enumerate() { self.sum[i] += v; self.squares[i] += v * v; self.ramp[i] += self.n * v; }
        self.n += 1.0;
    }
    /// Raw little-endian f32 planes `<tag>_mean.f32` and `<tag>_std.f32` for plotting.
    fn write(&self, dir: &std::path::Path, tag: &str) {
        let n = self.n;
        let (tm, tv) = ((n - 1.0) / 2.0, n * (n * n - 1.0) / 12.0);
        let mean: Vec<f32> = self.sum.iter().map(|s| s / n).collect();
        let std: Vec<f32> = (0..mean.len()).map(|i| {
            let slope = (self.ramp[i] - tm * self.sum[i]) / tv.max(1e-20);
            ((self.squares[i] / n - mean[i] * mean[i]) - slope * slope * tv / n).max(0.0).sqrt()
        }).collect();
        for (name, plane) in [("mean", &mean), ("std", &std)] {
            let _ = std::fs::write(dir.join(format!("{tag}_{name}.f32")), bytemuck::cast_slice::<f32, u8>(plane));
        }
    }
}

/// Where Flux turbulence comes from: slow, blob-sized wobble of the indirect light, measured with the
/// scene static and with one prop circling in a far corner. Every case removes one suspect.
/// FLUX_NOISE_SCENE=tv|bulb picks one emitter size, FLUX_NOISE_BACKEND=sw|hw one tracer,
/// FLUX_NOISE_CASE=<text>|<text> the cases whose name contains one of them, FLUX_NOISE_FRAMES the measured
/// window (the wobble has a memory of about 80 frames: short windows scatter by a third), FLUX_NOISE_DIR
/// receives whole-frame mean and temporal-std planes of the final GI per case, FLUX_NOISE_LAMP_RADIUS
/// gives the lamp a size in metres, so that its direct light is sampled and accumulated too, and
/// FLUX_NOISE_LAMP_INTENSITY its strength. See `settings` for the quality switches.
pub(super) fn turbulence_run(r: &mut Renderer) -> anyhow::Result<()> {
    #[derive(Clone, Copy)]
    struct Case { name: &'static str, moving: bool, temporal: Option<[f32; 4]>, clamp_off: bool, radiance_cache: bool,
        screen_traces: bool, fixed_seed: bool, surface: u32, keep_probes: bool, taa: bool, ignore_changes: bool, aloft: bool, rough_cache: bool }
    let base = Case { name: "", moving: false, temporal: None, clamp_off: false, radiance_cache: true, screen_traces: true,
        fixed_seed: false, surface: 0, keep_probes: false, taa: false, ignore_changes: false, aloft: false, rough_cache: false };
    // surface: 1 keeps the radiosity history beside changes, 2 keeps the direct history, 4 stops radiosity updates.
    // ignore_changes: no history anywhere reacts to the change regions (surface lighting and both screen histories).
    // aloft: the prop circles in mid-air between the lamp and the measured floor instead of in its corner.
    // rough_cache: the surface cache without its extra smoothing (independent light samples, nearest neighbours only).
    let cases = [
        Case { name: "static: default", ..base },
        Case { name: "static: one frame of rays (no temporal history)", temporal: Some([1.0, 1.0, 1.0, 0.0]), ..base },
        Case { name: "static: frozen gather seed (cache wobble only)", fixed_seed: true, ..base },
        Case { name: "static: frozen radiosity (gather noise only)", surface: 4, ..base },
        Case { name: "static: frozen seed and radiosity (floor)", fixed_seed: true, surface: 4, ..base },
        Case { name: "static: final history clamp off", clamp_off: true, ..base },
        Case { name: "static: radiance cache off", radiance_cache: false, ..base },
        Case { name: "static: screen traces off", screen_traces: false, ..base },
        Case { name: "static: TAA jitter", taa: true, ..base },
        Case { name: "static: surface cache without its smoothing", rough_cache: true, ..base },
        Case { name: "moving: default", moving: true, ..base },
        Case { name: "moving: histories ignore the change regions", moving: true, ignore_changes: true, ..base },
        Case { name: "moving: radiosity keeps its history beside the prop", moving: true, surface: 1, ..base },
        Case { name: "moving: direct light keeps its history beside the prop", moving: true, surface: 2, ..base },
        Case { name: "moving: radiance cache keeps its probes", moving: true, keep_probes: true, ..base },
        Case { name: "moving: frozen radiosity (gather noise only)", moving: true, surface: 4, ..base },
        Case { name: "moving: TAA jitter", moving: true, taa: true, ..base },
        Case { name: "aloft: prop circling between the lamp and the floor", moving: true, aloft: true, ..base },
        Case { name: "aloft: histories ignore the change regions", moving: true, aloft: true, ignore_changes: true, ..base },
    ];
    let mut clock = 4500.0f64;
    // FLUX_NOISE_WIDTH: view width in pixels (16:9), default 640.
    let width = std::env::var("FLUX_NOISE_WIDTH").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(640).clamp(320, 2560) / 16 * 16;
    let size = [width, width / 16 * 9];
    let only_scene = std::env::var("FLUX_NOISE_SCENE").ok();
    let only_backend = std::env::var("FLUX_NOISE_BACKEND").ok();
    let only_case = std::env::var("FLUX_NOISE_CASE").ok();
    let dir = std::env::var("FLUX_NOISE_DIR").ok().map(std::path::PathBuf::from);
    if let Some(dir) = &dir { std::fs::create_dir_all(dir)?; }
    let frames = std::env::var("FLUX_NOISE_FRAMES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(240).max(16);
    for (label, emitter, radiance) in [("tv", [1.6f32, 0.9], 8.0f32), ("bulb", [0.25, 0.25], 8.0 * 1.6 * 0.9 / (0.25 * 0.25))] {
        if only_scene.as_deref().is_some_and(|o| o != label) { continue; }
        let (mut draws, camera, _screen, mut fields) = closed_room_with_prop(r, emitter, radiance, true)?;
        // The fixture geometry casts no shadows by default. FLUX_NOISE_SHADOWS=1 makes every body a caster,
        // so that lights are occluded and a light with a size has penumbrae to resolve.
        if std::env::var("FLUX_NOISE_SHADOWS").is_ok_and(|v| v == "1") { for d in &mut draws { d.shadow_casting = crate::ShadowCasting::On; } }
        let lamp_radius = std::env::var("FLUX_NOISE_LAMP_RADIUS").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0).max(0.0);
        let lamp_intensity = std::env::var("FLUX_NOISE_LAMP_INTENSITY").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.6).max(0.0);
        let lamp = [crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(0.0, 2.7, 0.0), direction: Vec3::NEG_Y,
            color: [1.0, 0.85, 0.6], intensity: lamp_intensity, range: 9.0, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
            cast_shadows: false, soft_shadows: lamp_radius > 0.0, radius: lamp_radius, shadow_bias: 0.0, shadow_clip: None, baked: false }];
        let rects = [
            project_rect(camera, size, [Vec3::new(-1.5, 0.0, -2.0), Vec3::new(1.5, 0.0, -2.0), Vec3::new(-1.5, 0.0, 1.0), Vec3::new(1.5, 0.0, 1.0)]),
            project_rect(camera, size, [Vec3::new(-4.0, 0.4, -1.0), Vec3::new(-4.0, 2.0, -1.0), Vec3::new(-4.0, 0.4, 1.5), Vec3::new(-4.0, 2.0, 1.5)]),
        ];
        let names = ["floor", "left wall"];
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([size[0], size[1]]);
        input.draws = &draws;
        input.lights = &lamp;
        input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
        input.draw_skybox = false;
        input.postfx.bloom_enabled = false;
        input.postfx.exposure = 1.5;
        input.postfx.auto_exposure = crate::AutoExposure::default();
        let prop = draws.len() - 1;
        // FLUX_NOISE_PROP_AT=x,y,z rests the prop elsewhere, e.g. 0.6,1.5,0.3 to shade the measured floor.
        let home = std::env::var("FLUX_NOISE_PROP_AT").ok().and_then(|v| {
            let c: Vec<f32> = v.split(',').filter_map(|c| c.trim().parse().ok()).collect();
            (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
        }).unwrap_or(Vec3::new(-3.0, 0.2, 3.0));
        let place = |at: Vec3| Mat4::from_scale_rotation_translation(Vec3::splat(0.4), glam::Quat::IDENTITY, at);
        eprintln!("[Flux turbulence] scene={label} frames={frames} lamp radius={lamp_radius} rects={rects:?} (blob = 16 px block mean; cv = temporal std / mean, linear trend removed)");
        for rt in [false, true] {
            if rt && !r.ctx.ray_tracing() { continue; }
            let backend = if rt { "hw" } else { "sw" };
            if only_backend.as_deref().is_some_and(|o| o != backend) { continue; }
            let mut converged = false;
            for case in &cases {
                if only_case.as_deref().is_some_and(|o| !o.split('|').any(|part| case.name.contains(part))) { continue; }
                let mut current = draws.clone();
                current[prop].transform = place(home);
                fields[prop].world_to_local = place(home).inverse();
                r.gi_set_mesh_sdfs(&fields)?;
                r.set_anti_aliasing(if case.taa { crate::AntiAliasing::Taa } else { crate::AntiAliasing::None });
                r.set_flux_settings(settings(rt, case.screen_traces));
                let gi = r.gpu_gi.as_mut().unwrap();
                gi.temporal_override = case.temporal;
                gi.debug_flags = u32::from(case.clamp_off) | (u32::from(case.ignore_changes) * 2) | (u32::from(case.rough_cache) * 4);
                gi.gather_seed = case.fixed_seed.then_some(7);
                gi.radiance_cache.enabled_override = Some(case.radiance_cache);
                gi.radiance_cache.ignore_changes = case.keep_probes;
                // Radiosity is stopped only after the caches converged with it running.
                if let Some(surface) = r.flux_surface.as_mut() { surface.debug_keep_history = case.surface & 3; }
                r.invalidate_screen_gi();
                await_pipelines(r, &input)?;
                // Stages in pipeline order: traced probes with their directional history, the probe-space
                // spatial filter, the per-pixel integrate, the final full-resolution history.
                let stage_names = ["probe gather", "spatial filter", "integrate", "final history"];
                let mut stages: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); rects.len()]; 4];
                let mut radiosity: Vec<Vec<f32>> = Vec::new();
                let mut direct: Vec<Vec<f32>> = Vec::new();
                let mut map = PixelMap::default();
                let (mut far, mut ms) = ([0.0f64; 3], [0.0f32; 3]);
                // Pixels per texel of each stage: the probe grid follows the probe spacing.
                let mut stage_scale = [1u32; 4];
                // The first case of a backend also converges the world caches from empty.
                let warmup = if converged { 360 } else { 1500 };
                converged = true;
                for frame in 0..warmup + frames {
                    clock += 1.0 / 60.0;
                    r.post_pass.exposure.time_override = Some(clock);
                    if frame == warmup / 2 { r.flux_surface.as_mut().unwrap().debug_keep_history = case.surface; }
                    if case.moving {
                        let t = frame as f32;
                        let centre = if case.aloft { Vec3::new(0.6, 1.5, 0.3) } else { home };
                        current[prop].transform = place(centre + Vec3::new(0.4 * (t * 0.02).sin(), 0.0, 0.4 * (t * 0.02).cos()));
                        fields[prop].world_to_local = current[prop].transform.inverse();
                        r.gi_set_mesh_sdfs(&fields)?;
                    }
                    r.render(&crate::FrameInput { draws: &current, egui: None, world_ui: Vec::new(), ..input })?;
                    if frame < warmup { continue; }
                    let target = r.viewport_target.as_ref().unwrap();
                    let (written, probes) = ((r.sgi_parity ^ 1) & 1, target.sgi.probe_extent);
                    let images = [(target.sgi.gi_image[written], probes.width, probes.height), (target.sgi.gi_filtered, probes.width, probes.height),
                        (target.sgi.gi_final, size[0], size[1]), (target.sgi.full_history[written].image, size[0], size[1])];
                    for (stage, (image, w, h)) in images.into_iter().enumerate() {
                        let whole = block(r, image, [0, 0, w, h])?;
                        let scale = (size[0] / w).max(1);
                        stage_scale[stage] = scale;
                        for (k, rect) in rects.iter().enumerate() {
                            let (x, y) = ((rect[0] / scale).min(w - 1), (rect[1] / scale).min(h - 1));
                            stages[stage][k].push(crop(&whole, w, [x, y, (rect[2] / scale).clamp(1, w - x), (rect[3] / scale).clamp(1, h - y)]));
                        }
                        if stage == 3 { map.add(&whole); }
                    }
                    if case.radiance_cache {
                        let (_, work, _) = r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0);
                        let words: Vec<u32> = super::flux_radiance::read(r, work, 0, 48 * 4)?.as_chunks::<4>().0.iter()
                            .map(|b| u32::from_ne_bytes(*b)).collect();
                        for (sum, word) in far.iter_mut().zip([3usize, 4, 0]) { *sum += f64::from(words[word]) / frames as f64; }
                    }
                    let allocator = r.allocator();
                    let (_, page) = r.flux_surface.as_ref().unwrap().debug_lighting_page(&r.ctx.device, &mut allocator.lock().unwrap(),
                        r.command_pool, r.ctx.queue, draws[0].material.0, 2)?;
                    radiosity.push(page.iter().map(|t| 0.2126 * t[1][0] + 0.7152 * t[1][1] + 0.0722 * t[1][2]).collect());
                    direct.push(page.iter().map(|t| 0.2126 * t[0][0] + 0.7152 * t[0][1] + 0.0722 * t[0][2]).collect());
                    let s = r.stats();
                    for (sum, value) in ms.iter_mut().zip([s.gpu_gi_ms, s.gpu_gi_gather_ms, s.gpu_gi_surface_ms]) { *sum += value / frames as f32; }
                }
                if let Some(dir) = &dir {
                    let tag: String = format!("{label}_{backend}_{}", case.name).chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
                    map.write(dir, &tag);
                }
                let cached = noise(&radiosity);
                // What a mirror shows of the cache: each texel's temporal std against the floor's mean light,
                // averaged and for the noisiest tenth of the texels (penumbrae, surroundings of small emitters).
                let grain = |frames: &[Vec<f32>]| -> (f32, f32, f32) {
                    let n = frames.len() as f32;
                    let mut texels: Vec<(f32, f32)> = (0..frames[0].len()).map(|i| {
                        let mean = frames.iter().map(|f| f[i]).sum::<f32>() / n;
                        (mean, (frames.iter().map(|f| (f[i] - mean).powi(2)).sum::<f32>() / n).sqrt())
                    }).collect();
                    let level = texels.iter().map(|t| t.0).sum::<f32>() / texels.len() as f32;
                    texels.sort_by(|a, b| a.1.total_cmp(&b.1));
                    let worst = &texels[texels.len() * 9 / 10..];
                    (level, texels.iter().map(|t| t.1).sum::<f32>() / texels.len() as f32 / level.max(1e-12),
                        worst.iter().map(|t| t.1).sum::<f32>() / worst.len() as f32 / level.max(1e-12))
                };
                let (direct_grain, bounce_grain) = (grain(&direct), grain(&radiosity));
                eprintln!("[Flux turbulence] {label} {backend} {}: floor surface cache, temporal std / mean light: direct mean={:.5} all texels={:.4} noisiest tenth={:.4} | \
                    radiosity mean={:.5} all texels={:.4} noisiest tenth={:.4}", case.name, direct_grain.0, direct_grain.1, direct_grain.2,
                    bounce_grain.0, bounce_grain.1, bounce_grain.2);
                eprintln!("[Flux turbulence] {label} {backend} {}: radiance cache answered {:.0}% of far lookups ({:.0} hit, {:.0} traced instead, {:.0} probes refreshed per frame) | \
                    floor radiosity texel mean={:.5} cv={:.4} step={:.4} | GI ms total={:.3} gather={:.3} surface={:.3}", case.name,
                    100.0 * far[0] / (far[0] + far[1]).max(1.0), far[0], far[1], far[2], cached.mean, cached.pixel_cv_dt, cached.step, ms[0], ms[1], ms[2]);
                for (k, region) in names.iter().enumerate() {
                    let w = rects[k][2];
                    let shown = blobs(&stages[3][k], w, 16);
                    let (grain, blob, before) = (noise(&stages[3][k]), noise(&shown), noise(&blobs(&stages[2][k], w, 16)));
                    eprintln!("[Flux turbulence] {label} {backend} {} [{region}]: final GI mean={:.5} pixel cv={:.4} blob cv={:.4} blob step={:.5} blob persistence={:.3} | \
                        before final history: pixel cv={:.4} blob cv={:.4}", case.name, grain.mean, grain.pixel_cv_dt, blob.pixel_cv_dt, blob.step,
                        persistence(&shown), noise(&stages[2][k]).pixel_cv_dt, before.pixel_cv_dt);
                    // Probe stages are at probe resolution: a 16 px blob is 16 / spacing texels, at least one.
                    let chain: Vec<String> = (0..4).map(|stage| {
                        let (scale, frames) = (stage_scale[stage], &stages[stage][k]);
                        format!("{} texel cv={:.4} blob cv={:.4}", stage_names[stage], noise(frames).pixel_cv_dt,
                            noise(&blobs(frames, (rects[k][2] / scale).max(1), (16 / scale).max(1))).pixel_cv_dt)
                    }).collect();
                    eprintln!("[Flux turbulence stages] {label} {backend} {} [{region}]: {}", case.name, chain.join(" -> "));
                }
            }
            let gi = r.gpu_gi.as_mut().unwrap();
            gi.temporal_override = None; gi.debug_flags = 0; gi.gather_seed = None;
            gi.radiance_cache.enabled_override = None; gi.radiance_cache.ignore_changes = false;
            if let Some(surface) = r.flux_surface.as_mut() { surface.debug_keep_history = 0; }
            r.set_anti_aliasing(crate::AntiAliasing::None);
        }
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

/// The desktop game path must honour auto exposure in the Flux ray limit exactly like the editor views.
/// In a dark room lit by a small emitter the limit removes most of the emitter's energy at high gain, so a
/// path that ignores the meter measures an order of magnitude more physical GI than the editor.
pub(super) fn game_run(r: &mut Renderer) -> anyhow::Result<()> {
    let mut clock = 3500.0f64;
    let (draws, camera) = room(r, [1.6, 0.9], 8.0)?;
    let mut input = crate::cascade_tests::input(camera);
    input.draws = &draws;
    input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
    input.draw_skybox = false;
    input.postfx.bloom_enabled = false;
    input.postfx.exposure = 0.0;
    // Pinned adaptation range: both paths adapt to the same exposure, so only the wiring can differ.
    input.postfx.auto_exposure = crate::AutoExposure { enabled: true, min_ev100: -6.0, max_ev100: -6.0, ..Default::default() };
    for rt in [false, true] {
        if rt && !r.ctx.ray_tracing() { continue; }
        r.set_flux_settings(FluxSettings { rt_trace: rt, reflection_mode: 0, ssr_enabled: false, ..Default::default() });
        let mut editor = 0.0f32;
        for (path, extent) in [("editor Scene view", Some([640u32, 360])), ("desktop game path", None)] {
            input.viewport_extent = extent;
            input.camera_preview = None;
            r.invalidate_screen_gi();
            await_pipelines(r, &input)?;
            for _ in 0..600 { clock += 1.0 / 60.0; r.post_pass.exposure.time_override = Some(clock); r.render(&input)?; }
            let (image, size) = if extent.is_some() {
                let t = r.viewport_target.as_ref().unwrap();
                (t.sgi.full_history[(r.sgi_parity ^ 1) & 1].image, t.extent)
            } else {
                (r.sgi.full_history[(r.sgi_parity ^ 1) & 1].image, vk::Extent2D { width: r.render_extent[0], height: r.render_extent[1] })
            };
            let rect = floor_rect(camera, [size.width, size.height]);
            let mean = { let v = block(r, image, rect)?; v.iter().sum::<f32>() / v.len() as f32 };
            let stops = r.post_pass.exposure.debug_state(&r.ctx.device, 0)[0];
            eprintln!("[Flux game] RT={rt} {path}: adapted exposure {stops:+.2} stops, floor GI mean (physical) = {mean:.6}");
            anyhow::ensure!(stops > 3.0 && stops < 9.0, "{path}: auto exposure did not brighten the dark room ({stops})");
            if extent.is_some() { editor = mean; } else {
                let ratio = mean / editor;
                anyhow::ensure!((0.4..2.5).contains(&ratio),
                    "RT={rt}: the game path's Flux ray limit ignores auto exposure: game GI {mean} vs editor GI {editor} (x{ratio:.1})");
            }
        }
    }
    r.post_pass.exposure.time_override = None;
    Ok(())
}

/// Moving an existing gather instance (a character, a physics body, 0.1 mm of jitter) must cost only the
/// surroundings of the mover their lighting history, in the screen histories, the surface cache and the
/// radiance cache alike. Bounce light also follows beside the shadows the mover casts, however far away.
/// A surface that takes no part in the transport costs nothing at all; adding or removing an instance
/// still resets the screen history.
pub(super) fn history_run(r: &mut Renderer) -> anyhow::Result<()> {
    let (draws, camera, _screen, mut fields) = closed_room_with_prop(r, [1.6, 0.9], 8.0, true)?;
    let lamp = [crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(0.0, 2.7, 0.0), direction: Vec3::NEG_Y,
        color: [1.0, 0.85, 0.6], intensity: 0.6, range: 9.0, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
        cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false }];
    let mut input = crate::cascade_tests::input(camera);
    input.viewport_extent = Some([320, 180]);
    input.draws = &draws;
    input.lights = &lamp;
    input.light.ambient = [0.0; 3]; input.light.intensity = 0.0; input.light.env_intensity = 0.0;
    input.draw_skybox = false;
    input.postfx.bloom_enabled = false;
    let floor_counts = |r: &Renderer| -> anyhow::Result<Vec<f32>> {
        let allocator = r.allocator();
        let (_, page) = r.flux_surface.as_ref().unwrap().debug_lighting_page(&r.ctx.device, &mut allocator.lock().unwrap(),
            r.command_pool, r.ctx.queue, draws[0].material.0, 2)?;
        Ok(page.iter().filter(|t| t[1][3] > 0.5).map(|t| t[1][3]).collect())
    };
    // Far lookups the radiance cache answered and those traced instead, in the last frame.
    let far_lookups = |r: &Renderer| -> anyhow::Result<[f64; 2]> {
        let (_, work, _) = r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0);
        let words: Vec<u32> = super::flux_radiance::read(r, work, 0, 48 * 4)?.as_chunks::<4>().0.iter().map(|b| u32::from_ne_bytes(*b)).collect();
        Ok([f64::from(words[3]), f64::from(words[4])])
    };
    // A pane of glass: alpha blended without emission, so neither traced nor captured.
    let glass = lit(r, [0.6, 0.8, 0.9, 0.3], [0.0; 3], 0.1, 0.0);
    r.set_material_features(glass, crate::MaterialFeatures { alpha_mode: crate::AlphaMode::Blend, ..Default::default() });
    for rt in [false, true] {
        if rt && !r.ctx.ray_tracing() { continue; }
        r.set_flux_settings(FluxSettings { rt_trace: rt, reflection_mode: 0, ssr_enabled: false, ..Default::default() });
        r.invalidate_screen_gi();
        await_pipelines(r, &input)?;
        for _ in 0..40 { r.render(&input)?; }
        anyhow::ensure!(r.sgi_history_valid, "RT={rt}: no screen GI history after 40 static frames");
        let prop = fields.len() - 1;
        let home = Vec3::new(-3.0, 0.2, 3.0);
        let place = |at: Vec3| Mat4::from_scale_rotation_translation(Vec3::splat(0.4), glam::Quat::IDENTITY, at);
        for (i, at) in [home + Vec3::X * 0.3, home + Vec3::X * 0.3001, home + Vec3::Z * 0.2].into_iter().enumerate() {
            fields[prop].world_to_local = place(at).inverse();
            r.gi_set_mesh_sdfs(&fields)?;
            anyhow::ensure!(r.sgi_history_valid, "RT={rt}: moving an existing instance (step {i}) discarded the screen GI history");
            r.render(&input)?;
            // The move is known as a place with the reach of the prop, not as a scene-wide event.
            let packed = r.flux_changes.packed();
            let located = (0..packed.info[0] as usize).any(|k| {
                let (lo, hi) = (packed.bounds[k * 2], packed.bounds[k * 2 + 1]);
                lo[3] > 0.0 && lo[3] < 2.0 && hi[3] == 0.0
                    && Vec3::from_slice(&lo).cmple(at).all() && Vec3::from_slice(&hi).cmpge(at).all()
            });
            anyhow::ensure!(located, "RT={rt}: step {i} was not reported as a local geometry change: {:?}",
                &packed.bounds[..packed.info[0] as usize * 2]);
        }
        // The settled blend of the whole view is untouched; the gather shortens it beside the mover only.
        anyhow::ensure!((r.screen_history_alpha(true, 0.5) - 0.06).abs() < 1e-6, "instance motion shortened the whole view's history");
        // Surface-cache radiosity: let it settle, then circle the prop in its corner.
        fields[prop].world_to_local = place(home).inverse();
        r.gi_set_mesh_sdfs(&fields)?;
        let mut answered = [0.0f64; 2];
        for frame in 0..500 {
            r.render(&input)?;
            if frame >= 440 { for (sum, n) in answered.iter_mut().zip(far_lookups(r)?) { *sum += n; } }
        }
        let settled = floor_counts(r)?;
        let lowest = settled.iter().cloned().fold(f32::MAX, f32::min);
        anyhow::ensure!(!settled.is_empty() && lowest > 20.0, "RT={rt}: static radiosity history did not build up: shortest {lowest}");
        let still = answered[0] / (answered[0] + answered[1]).max(1.0);
        // Glass carried through the room changes no light that Flux transports: no region, no history lost.
        let epoch = r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0).2;
        let mut with_glass = draws.clone();
        with_glass.push(draw(draws[0].mesh, glass, Mat4::IDENTITY, -1));
        for frame in 0..40 {
            let t = frame as f32;
            *with_glass.last_mut().unwrap() = draw(draws[0].mesh, glass,
                Mat4::from_scale_rotation_translation(Vec3::new(1.0, 1.5, 0.02), glam::Quat::IDENTITY, Vec3::new(-1.0 + 0.05 * t, 1.2, 0.5)), -1);
            r.render(&crate::FrameInput { draws: &with_glass, egui: None, world_ui: Vec::new(), ..input })?;
            anyhow::ensure!(r.flux_changes.packed().info[0] == 0.0, "RT={rt}: moving glass (frame {frame}) was reported as a lighting change: {:?}",
                &r.flux_changes.packed().bounds[..2]);
        }
        anyhow::ensure!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0).2 == epoch, "RT={rt}: moving glass reset the radiance cache");
        anyhow::ensure!(r.sgi_history_valid, "RT={rt}: moving glass discarded the screen GI history");
        let beside_glass = floor_counts(r)?.iter().cloned().fold(f32::MAX, f32::min);
        anyhow::ensure!(beside_glass > 20.0, "RT={rt}: moving glass shortened the radiosity history to {beside_glass}");
        let mut answered = [0.0f64; 2];
        for frame in 0..240 {
            let t = frame as f32;
            fields[prop].world_to_local = place(home + Vec3::new(0.4 * (t * 0.02).sin(), 0.0, 0.4 * (t * 0.02).cos())).inverse();
            r.gi_set_mesh_sdfs(&fields)?;
            r.render(&input)?;
            if frame >= 180 { for (sum, n) in answered.iter_mut().zip(far_lookups(r)?) { *sum += n; } }
        }
        let moving = floor_counts(r)?;
        let (shortest, kept) = (moving.iter().cloned().fold(f32::MAX, f32::min),
            moving.iter().filter(|c| **c > 20.0).count() as f32 / moving.len() as f32);
        eprintln!("[Flux history] RT={rt}: floor radiosity history while a corner prop moves: shortest {shortest:.1} updates, {:.0}% of texels above 20 (settled shortest {lowest:.1})", kept * 100.0);
        anyhow::ensure!(shortest < 6.0, "RT={rt}: radiosity beside the moving prop kept a long history ({shortest}); it would trail the prop");
        anyhow::ensure!(kept > 0.6, "RT={rt}: a prop moving in one corner cost {:.0}% of the floor its radiosity history", (1.0 - kept) * 100.0);
        // The radiance cache drops the probes beside the prop, not every probe that could see it.
        let moved = answered[0] / (answered[0] + answered[1]).max(1.0);
        eprintln!("[Flux history] RT={rt}: radiance cache answered {:.0}% of far lookups at rest and {:.0}% while the corner prop moves", still * 100.0, moved * 100.0);
        anyhow::ensure!(still < 0.05 || moved > 0.5 * still, "RT={rt}: a prop moving in one corner emptied the radiance cache: {moved:.3} of far lookups answered, {still:.3} at rest");
        // Lifted between the lamp and the floor, the prop shades the floor two metres away, beyond its
        // own reach: bounce light in that shadow follows it at once, the rest of the floor keeps a
        // useful history. With the lamp off there is no shadow, and nothing to follow there.
        let high = Vec3::new(0.9, 1.5, 0.9);
        let mut shortest = [0.0f32; 2];
        for (k, lights) in [&lamp[..], &[]].into_iter().enumerate() {
            let lit = crate::FrameInput { lights, egui: None, world_ui: Vec::new(), ..input };
            fields[prop].world_to_local = place(home).inverse();
            r.gi_set_mesh_sdfs(&fields)?;
            for _ in 0..400 { r.render(&lit)?; }
            for frame in 0..240 {
                let t = frame as f32;
                fields[prop].world_to_local = place(high + Vec3::new(0.02 * (t * 0.3).sin(), 0.0, 0.02 * (t * 0.3).cos())).inverse();
                r.gi_set_mesh_sdfs(&fields)?;
                r.render(&lit)?;
            }
            let counts = floor_counts(r)?;
            shortest[k] = counts.iter().cloned().fold(f32::MAX, f32::min);
            let kept = counts.iter().filter(|c| **c > 8.0).count() as f32 / counts.len() as f32;
            eprintln!("[Flux history] RT={rt}: floor radiosity history with the prop moving 2 m above it, lamp {}: shortest {:.1} updates, {:.0}% of texels above 8",
                if k == 0 { "on" } else { "off" }, shortest[k], kept * 100.0);
            anyhow::ensure!(kept > 0.6, "RT={rt}: a prop in mid-air cost {:.0}% of the floor its radiosity history", (1.0 - kept) * 100.0);
        }
        anyhow::ensure!(shortest[0] * 2.0 < shortest[1], "RT={rt}: bounce light in the prop's shadow kept as long a history ({}) as without the lamp ({}); it would lag the shadow",
            shortest[0], shortest[1]);
        fields[prop].world_to_local = place(home).inverse();
        r.gi_set_mesh_sdfs(&fields)?;
        // Removing an instance is structural.
        let removed = fields.pop().unwrap();
        r.gi_set_mesh_sdfs(&fields)?;
        anyhow::ensure!(!r.sgi_history_valid, "RT={rt}: removing an instance kept stale screen GI history");
        fields.push(removed);
        r.gi_set_mesh_sdfs(&fields)?;
    }
    Ok(())
}
