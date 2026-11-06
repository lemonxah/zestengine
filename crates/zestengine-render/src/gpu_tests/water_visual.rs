//! Fixed scene captures of the production water material, without opening an
//! editor project. Run explicitly with --ignored; ZESTENGINE_TEST_DUMP_DIR
//! optionally overrides the output directory.
use std::sync::Arc;
use glam::{Mat4, Vec3};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ActiveEventLoop, EventLoop}, window::{Window, WindowId}};
use winit::platform::{run_on_demand::EventLoopExtRunOnDemand, x11::EventLoopBuilderExtX11};
use crate::{CameraData, GpuWater, MaterialDesc, MaterialFeatures, MeshData, Renderer, Vertex};
use super::super::planar::{cube, draw, lit};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 600;

fn surface() -> MeshData {
    surface_span(12.0)
}

fn surface_span(span: f32) -> MeshData {
    let n = 96u32;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for z in 0..=n { for x in 0..=n {
        let p = [(x as f32 / n as f32 - 0.5) * span, (z as f32 / n as f32 - 0.5) * span];
        vertices.push(Vertex { position: [p[0], 0.0, p[1]], normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0], uv: p, uv1: [span / n as f32; 2],
            color: [0.5, 0.5, 0.0, 1.0], ..Default::default() });
    }}
    for z in 0..n { for x in 0..n { let a = z * (n + 1) + x; indices.extend([a, a+n+2, a+1, a, a+n+1, a+n+2]); }}
    MeshData { vertices, indices, has_lightmap_uv: false }
}

fn capture(window: Arc<Window>) -> anyhow::Result<()> {
    let mut r = Renderer::new(window)?;
    r.set_vsync(false);
    r.set_gi_enabled(false);
    let cube = r.upload_mesh(&cube())?;
    let surface = r.upload_mesh(&surface())?;
    let pale = lit(&mut r, [0.65, 0.60, 0.48, 1.0], [0.0; 3], 0.8, 0.0);
    let dark = lit(&mut r, [0.10, 0.15, 0.17, 1.0], [0.0; 3], 0.8, 0.0);
    let red = lit(&mut r, [0.65, 0.04, 0.02, 1.0], [0.0; 3], 0.5, 0.0);
    let mirror = lit(&mut r, [0.95, 0.95, 0.95, 1.0], [0.0; 3], 0.02, 1.0);
    let water_mat = r.create_material(&MaterialDesc { name: "Water visual fixture".into(),
        features: MaterialFeatures { water: true, double_sided: true, alpha_mode: crate::AlphaMode::Blend, ..Default::default() }, ..Default::default() })?;
    r.set_material_water_slot(water_mat, 0);
    let mut draws = Vec::new();
    for z in 0..12 { for x in 0..12 {
        draws.push(draw(cube, if (x+z)%2 == 0 { pale } else { dark },
            Mat4::from_scale_rotation_translation(Vec3::new(1.0, 0.2, 1.0), glam::Quat::IDENTITY, Vec3::new(x as f32-5.5, -1.6, z as f32-5.5)), -1));
    }}
    for (p, scale) in [(Vec3::new(-2.5, -0.75, -1.0), Vec3::new(1.0, 1.5, 1.0)), (Vec3::new(2.0, 0.0, -2.5), Vec3::new(1.0, 3.0, 1.0))] {
        draws.push(draw(cube, red, Mat4::from_scale_rotation_translation(scale, glam::Quat::IDENTITY, p), -1));
    }
    draws.push(draw(surface, water_mat, Mat4::IDENTITY, -1));
    draws.push(draw(cube, mirror, Mat4::from_scale_rotation_translation(Vec3::new(0.1, 4.0, 12.0), glam::Quat::IDENTITY, Vec3::new(-6.2, 0.5, 0.0)), -1));
    let base = zestengine_core::Water::default();
    let mut water = GpuWater::default();
    water.to_local = Mat4::IDENTITY.to_cols_array_2d();
    water.shallow = [0.1, 0.45, 0.48, 4.0]; water.deep = [0.01, 0.07, 0.12, 1.0];
    water.look0 = [0.06, 0.15, 1.0, 0.2]; water.look1 = [0.2, 1.0, 0.0, 0.0];
    water.ripple[3] = 1.0; water.bounds = [6.0, 6.0, 0.0, 1.0];
    for (i, w) in base.resolved_waves().iter().enumerate() {
        water.wave_a[i] = [w.dir.x, w.dir.y, w.k, w.omega];
        water.wave_b[i] = [w.qa, w.amplitude, w.qa * w.k, 0.0];
        water.look1[2] += 1.0; water.look1[3] += w.amplitude;
    }
    let mut sun = crate::LightInstance { kind: crate::LightKind::Directional, position: Vec3::ZERO,
        direction: Vec3::new(0.5, -1.0, -0.4).normalize(), color: [1.0, 0.95, 0.85], intensity: 3.0,
        range: 100.0, spot_inner_deg: 0.0, spot_outer_deg: 0.0, cast_shadows: true, soft_shadows: false,
        radius: 0.0, shadow_bias: 0.0, baked: false };
    let optics = zestengine_core::WaterOptics::default();
    let mut reference = None::<Vec<u8>>;
    let mut mirror_reference = None::<f64>;
    for (name, eye, absorption, sun_gain, caustics) in [
        ("pool_clear", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_last_slot", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_last_wave", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_no_caustics", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_absorbing", Vec3::new(7.0, 4.5, 8.0), [1.2, 0.4, 0.12], 3.0, 1.0),
        ("pool_grazing", Vec3::new(7.0, 0.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_no_sun", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 0.0, 1.0),
        ("pool_underwater", Vec3::new(3.0, -0.65, 4.0), optics.absorption, 3.0, 1.0),
        ("pool_rt", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_absorbing", Vec3::new(7.0, 4.5, 8.0), [1.2, 0.4, 0.12], 3.0, 1.0),
    ] {
        let ray_traced = name.starts_with("pool_rt");
        if ray_traced && !r.ctx.ray_tracing() { continue; }
        r.set_flux_settings(crate::FluxSettings { reflection_mode: if ray_traced { 2 } else { 0 }, rt_samples: 1, rt_bounces: 2, ssr_enabled: false, ..Default::default() });
        water.set_optics([absorption[0], absorption[1], absorption[2], optics.ior],
            [optics.scattering[0], optics.scattering[1], optics.scattering[2], optics.anisotropy]);
        water.bounds[3] = caustics;
        sun.intensity = sun_gain;
        let camera = CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::new(0.0, -0.5, 0.0), Vec3::Y), proj: Mat4::perspective_rh(0.95, WIDTH as f32 / HEIGHT as f32, 0.05, 500.0) };
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([WIDTH, HEIGHT]); input.draws = &draws;
        let mut bodies = [GpuWater::default(); crate::MAX_WATER_BODIES];
        let slot = if name == "pool_last_slot" { crate::MAX_WATER_BODIES - 1 } else { 0 };
        bodies[slot] = water;
        if name == "pool_last_wave" {
            // Move the same waves to the end of the larger arrays. Preserve the
            // optical coefficients in the first eight w lanes.
            let count = water.look1[2] as usize;
            for i in 0..crate::MAX_WATER_WAVES {
                bodies[slot].wave_a[i] = [0.0; 4];
                bodies[slot].wave_b[i][..3].fill(0.0);
            }
            for i in 0..count {
                let dst = crate::MAX_WATER_WAVES - count + i;
                bodies[slot].wave_a[dst] = water.wave_a[i];
                bodies[slot].wave_b[dst][..3].copy_from_slice(&water.wave_b[i][..3]);
            }
            bodies[slot].look1[2] = crate::MAX_WATER_WAVES as f32;
        }
        r.set_material_water_slot(water_mat, slot as u32);
        input.water = &bodies[..=slot]; input.water_time = 1.25;
        input.lights = std::slice::from_ref(&sun);
        input.light.intensity = 0.0; input.light.ambient = [0.12, 0.15, 0.19];
        input.postfx.bloom_enabled = false; input.clear_color = [0.06, 0.09, 0.13, 1.0];
        if name == "pool_underwater" {
            input.postfx.underwater = Some(crate::UnderwaterFx { surface: 0.0, shallow_color: [0.1, 0.45, 0.48], deep_color: [0.01, 0.07, 0.12],
                density: 0.1, light_falloff: 0.05, distortion: 0.0, blur: 0.0, caustics: 1.0,
                water_to_world: Mat4::IDENTITY, body: water, rest_surface: 0.0, clock: 1.25 });
        }
        for _ in 0..12 { r.render(&input)?; }
        let (w, h, pixels) = r.debug_read_viewport()?.ok_or_else(|| anyhow::anyhow!("no viewport readback"))?;
        anyhow::ensure!((w,h) == (WIDTH,HEIGHT));
        anyhow::ensure!(pixels.chunks_exact(4).any(|p| p[0] > 30 && p[0] < 240), "water fixture is empty/clipped");
        if name == "pool_clear" { reference = Some(pixels.clone()); }
        if matches!(name, "pool_last_slot" | "pool_last_wave") {
            let delta = pixels.iter().zip(reference.as_ref().unwrap()).filter(|(a,b)| a.abs_diff(**b) > 3).count();
            anyhow::ensure!(delta < 2000, "{name}: moving identical water data changes the image ({delta} channels)");
        }
        if name == "pool_no_caustics" {
            let delta = pixels.iter().zip(reference.as_ref().unwrap()).filter(|(a,b)| a.abs_diff(**b) > 3).count();
            anyhow::ensure!(delta > 500, "caustics have no rendered effect: {delta} changed channels");
        }
        if name == "pool_absorbing" {
            let delta = pixels.iter().zip(reference.as_ref().unwrap()).filter(|(a,b)| a.abs_diff(**b) > 8).count();
            anyhow::ensure!(delta > 5000, "absorption has no meaningful rendered effect: {delta} changed channels");
        }
        if ray_traced {
            // Fixed crop wholly inside the mirror, below its reflected water
            // line. It must respond to the medium along reflected ray hits.
            let mut red = 0.0;
            for y in 180..220 { for x in 80..180 { red += pixels[((y * WIDTH + x) * 4) as usize] as f64; } }
            if name == "pool_rt" { mirror_reference = Some(red); }
            else { anyhow::ensure!(red < mirror_reference.unwrap() * 0.9, "reflected water ignores absorption: red sum {red} vs {:?}", mirror_reference); }
        }
        {
            let dir = std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp/zest-water-visual".into());
            std::fs::create_dir_all(&dir)?;
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            for px in pixels.chunks_exact(4) { ppm.extend_from_slice(&px[..3]); }
            std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")), ppm)?;
        }
    }
    // Exercise the real XR paths without a headset. The extreme body/wave
    // slots catch descriptor/layout differences from the desktop pass.
    use super::super::xr_cluster::{Target, W, H, FORMAT};
    let targets = [Target::new(&r, 1)?, Target::new(&r, 1)?];
    let stereo = if r.ctx.multiview { Some(Target::new(&r, 2)?) } else { None };
    let result = (|| -> anyhow::Result<()> {
        r.set_flux_settings(crate::FluxSettings { reflection_mode: 0, ssr_enabled: false, ..Default::default() });
        r.set_material_water_slot(water_mat, (crate::MAX_WATER_BODIES - 1) as u32);
        let eye = Vec3::new(7.0, 4.5, 8.0);
        let center = Mat4::look_at_rh(eye, Vec3::new(0.0, -0.5, 0.0), Vec3::Y);
        let proj = Mat4::perspective_rh(0.95, 1.0, 0.05, 500.0);
        let clustered = super::super::xr_cluster::Scene::new(&mut r)?;
        let mut xr_draws = draws.clone();
        xr_draws.extend(clustered.draws(false));
        r.set_cluster_settings(crate::ClusterSettings { mode: crate::ClusterMode::On, ..r.cluster_settings() });
        r.set_gpu_cull_settings(crate::GpuCullSettings { hiz_occlusion: true, ..r.gpu_cull_settings() });
        let mut input = crate::cascade_tests::input(CameraData { position: eye, view: center, proj });
        input.render_viewport = false; input.draws = &xr_draws; input.water_time = 1.25;
        input.lights = std::slice::from_ref(&sun); input.light.intensity = 0.0;
        input.light.ambient = [0.12, 0.15, 0.19]; input.postfx.bloom_enabled = false;
        let eyes = [-0.032f32, 0.032].map(|offset| {
            let mut asymmetric = proj;
            asymmetric.z_axis.x = offset.signum() * 0.16;
            crate::XrEyeCamera { view: Mat4::from_translation(Vec3::new(-offset, 0.0, 0.0)) * center,
                proj: asymmetric, postfx: input.postfx.clone() }
        });
        let cull_proj = Mat4::perspective_rh(1.25, 1.0, 0.05, 500.0);
        let mut references = Vec::new();
        let mut water_cases = [[GpuWater::default(); crate::MAX_WATER_BODIES]; 3];
        for (bodies, absorption) in water_cases.iter_mut().zip([optics.absorption, [1.2, 0.4, 0.12], optics.absorption]) {
            let mut body = water;
            body.set_optics([absorption[0], absorption[1], absorption[2], optics.ior],
                [optics.scattering[0], optics.scattering[1], optics.scattering[2], optics.anisotropy]);
            bodies[crate::MAX_WATER_BODIES - 1] = body;
        }
        water_cases[2][crate::MAX_WATER_BODIES - 1].look1[1] = 0.0;
        for (name, bodies) in ["clear", "absorbing", "unbent"].into_iter().zip(&water_cases) {
            input.water = bodies;
            let mut shots = Vec::new();
            for e in 0..2 {
                for _ in 0..3 {
                    r.render_xr_frame(&input)?;
                    r.render_xr_eye_indexed(e, targets[e].raw(), FORMAT.as_raw() as u32, W, H, eyes[e].view, eyes[e].proj, &input, None)?;
                    let pixels = targets[e].read(&r)?;
                    if let Some((copy, scene)) = r.xr_water_timing() { eprintln!("[XR water] eye {e} {name}: copy {copy:.4} ms, scene {scene:.4} ms at {W}x{H}"); }
                    if shots.len() <= e { shots.push((format!("xr_eye_{e}_{name}"), pixels)); }
                    else { shots[e].1 = pixels; }
                }
            }
            if let Some(stereo) = &stereo {
                let limits = unsafe { r.ctx.instance.get_physical_device_properties(r.ctx.physical_device) }.limits;
                let supports_four = (limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts).contains(ash::vk::SampleCountFlags::TYPE_4);
                for samples in [1, 4].into_iter().filter(|n| *n == 1 || supports_four) {
                    r.xr_msaa = samples;
                    let mut pixels = Vec::new();
                    for _ in 0..3 {
                        r.render_xr_frame(&input)?;
                        r.render_xr_stereo(stereo.raw(), FORMAT.as_raw() as u32, W, H, &eyes, (center, cull_proj), &input)?;
                        pixels = stereo.read(&r)?;
                        anyhow::ensure!(r.cluster.as_ref().is_some_and(|c| c.last_instances > 0 && c.last_batches > 0), "water/Hi-Z regression did not exercise clustered draws");
                        if let Some((copy, scene)) = r.xr_water_timing() { eprintln!("[XR water] multiview {samples}x {name}: copy {copy:.4} ms, scene {scene:.4} ms at {W}x{H} per eye"); }
                    }
                    let layer = (W * H * 4) as usize;
                    for e in 0..2 {
                        let image = pixels[e * layer..(e + 1) * layer].to_vec();
                        let error = image.iter().zip(&shots[e].1).map(|(a,b)| a.abs_diff(*b) as f64).sum::<f64>() / layer as f64;
                        eprintln!("[XR water] {name} eye {e}, multiview {samples}x vs single-eye 1x: mean byte error {error:.4}");
                        anyhow::ensure!(error < if samples == 1 { 1.5 } else { 3.0 }, "{name}: stereo eye {e} does not match independent eye (mean error {error})");
                        shots.push((format!("xr_multiview_{samples}x_{e}_{name}"), image));
                    }
                }
            }
            for (i, (label, pixels)) in shots.into_iter().enumerate() {
                if name == "clear" { references.push(pixels.clone()); }
                else {
                    let delta = pixels.iter().zip(&references[i]).filter(|(a,b)| a.abs_diff(**b) > 3).count();
                    anyhow::ensure!(delta > 100, "{label}: optical/refraction control does not reach this eye ({delta} changed channels)");
                }
                let dir = std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp/zest-water-visual".into());
                let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
                for px in pixels.chunks_exact(4) { ppm.extend_from_slice(&px[..3]); }
                std::fs::write(std::path::Path::new(&dir).join(format!("{label}.ppm")), ppm)?;
            }
        }
        // Fog and the moving waterline must also agree between independently
        // rendered eyes and multiview, including MSAA depth resolve.
        for (name, eye) in [("underwater", Vec3::new(3.0, -0.65, 4.0)), ("waterline", Vec3::new(3.0, 0.04, 4.0))] {
            let center = Mat4::look_at_rh(eye, Vec3::new(0.0, eye.y - 0.1, 0.0), Vec3::Y);
            let mut input = crate::cascade_tests::input(CameraData { position: eye, view: center, proj });
            input.render_viewport = false; input.draws = &xr_draws; input.water_time = 1.25;
            input.water = &water_cases[0]; input.lights = std::slice::from_ref(&sun);
            input.light.intensity = 0.0; input.light.ambient = [0.12, 0.15, 0.19];
            input.postfx.bloom_enabled = false;
            input.postfx.underwater = Some(crate::UnderwaterFx { surface: 0.0,
                shallow_color: [0.1, 0.45, 0.48], deep_color: [0.01, 0.07, 0.12], density: 0.1,
                light_falloff: 0.05, distortion: 0.0, blur: 0.0, caustics: 1.0,
                water_to_world: Mat4::IDENTITY, body: water_cases[0][crate::MAX_WATER_BODIES - 1], rest_surface: 0.0, clock: 1.25 });
            let eyes = [-0.032f32, 0.032].map(|offset| {
                let mut asymmetric = proj; asymmetric.z_axis.x = offset.signum() * 0.16;
                crate::XrEyeCamera { view: Mat4::from_translation(Vec3::new(-offset, 0.0, 0.0)) * center,
                    proj: asymmetric, postfx: input.postfx.clone() }
            });
            let mut per_eye = Vec::new();
            for e in 0..2 {
                r.render_xr_frame(&input)?;
                r.render_xr_eye_indexed(e, targets[e].raw(), FORMAT.as_raw() as u32, W, H, eyes[e].view, eyes[e].proj, &input, None)?;
                per_eye.push(targets[e].read(&r)?);
            }
            if let Some(stereo) = &stereo {
                let limits = unsafe { r.ctx.instance.get_physical_device_properties(r.ctx.physical_device) }.limits;
                let supports_four = (limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts).contains(ash::vk::SampleCountFlags::TYPE_4);
                for samples in [1, 4].into_iter().filter(|n| *n == 1 || supports_four) {
                r.xr_msaa = samples;
                r.render_xr_frame(&input)?;
                r.render_xr_stereo(stereo.raw(), FORMAT.as_raw() as u32, W, H, &eyes, (center, cull_proj), &input)?;
                let pixels = stereo.read(&r)?;
                let layer = (W * H * 4) as usize;
                for e in 0..2 {
                    let image = &pixels[e * layer..(e + 1) * layer];
                    let error = image.iter().zip(&per_eye[e]).map(|(a,b)| a.abs_diff(*b) as f64).sum::<f64>() / layer as f64;
                    eprintln!("[XR water] {name} eye {e}: {samples}x stereo vs independent mean byte error {error:.4}");
                    anyhow::ensure!(error < if samples == 1 { 1.5 } else { 3.0 }, "{name}: stereo waterline/fog differs in eye {e} ({error})");
                    let dir = std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp/zest-water-visual".into());
                    let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
                    for px in image.chunks_exact(4) { ppm.extend_from_slice(&px[..3]); }
                    std::fs::write(std::path::Path::new(&dir).join(format!("xr_{name}_{samples}x_{e}.ppm")), ppm)?;
                }
                }
            }
        }
        if stereo.is_some() {
            let large = Target::with_extent(&r, 2, 1024, 1024)?;
            let dry_draws: Vec<_> = xr_draws.iter().filter(|d| d.material != water_mat).cloned().collect();
            let benchmark = (|| -> anyhow::Result<()> {
                r.set_material_water_slot(water_mat, 0);
                for samples in [1, 4] {
                    let limits = unsafe { r.ctx.instance.get_physical_device_properties(r.ctx.physical_device) }.limits;
                    if !(limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts)
                        .contains(ash::vk::SampleCountFlags::from_raw(samples as u32)) { continue; }
                    r.xr_msaa = samples;
                    for (mode, draws) in [("water", &xr_draws), ("no water", &dry_draws)] {
                    input.draws = draws;
                    input.water = if mode == "water" { &water_cases[0][crate::MAX_WATER_BODIES - 1..] } else { &[] };
                    let mut copy_ms = Vec::new(); let mut scene_ms = Vec::new();
                    for frame in 0..8 {
                        r.render_xr_frame(&input)?;
                        r.render_xr_stereo(large.raw(), FORMAT.as_raw() as u32, 1024, 1024, &eyes, (center, cull_proj), &input)?;
                        unsafe { r.ctx.device.device_wait_idle()?; }
                        if frame >= 3 {
                            if let Some((copy, scene)) = r.xr_water_timing() { copy_ms.push(copy); scene_ms.push(scene); }
                        }
                    }
                    if r.gpu_ts_period > 0.0 {
                        anyhow::ensure!(copy_ms.len() == 5, "missing XR water benchmark GPU timestamps ({mode}, {samples}x)");
                        copy_ms.sort_by(f64::total_cmp); scene_ms.sort_by(f64::total_cmp);
                        eprintln!("[XR water benchmark] {mode}, 1024x1024 per eye, {samples}x MSAA, both eyes: median opaque copy {:.4} ms, scene draw {:.4} ms (5 warm samples)", copy_ms[copy_ms.len()/2], scene_ms[scene_ms.len()/2]);
                    }
                    }
                }
                Ok(())
            })();
            unsafe { r.ctx.device.device_wait_idle()?; }
            large.destroy(&r);
            benchmark?;
        }
        Ok(())
    })();
    unsafe { r.ctx.device.device_wait_idle()?; }
    for target in targets { target.destroy(&r); }
    if let Some(target) = stereo { target.destroy(&r); }
    result
}

// A sloped shore, irregular rocks and an HDR sky expose detail/foam/reflection
// changes that the optical checkerboard alone cannot judge. Camera/light/water
// settings are identical in before/after captures; no project files are edited.
fn appearance(window: Arc<Window>) -> anyhow::Result<()> {
    let mut r = Renderer::new(window)?;
    r.set_vsync(false);
    r.set_gi_enabled(false);
    let mut pixels = Vec::new();
    for y in 0..256 { for x in 0..512 {
        let phi = (x as f32 / 512.0 - 0.5) * std::f32::consts::TAU;
        let theta = y as f32 / 256.0 * std::f32::consts::PI;
        let d = Vec3::new(phi.cos()*theta.sin(), theta.cos(), phi.sin()*theta.sin());
        let cloud = ((phi * 3.0 + (theta * 11.0).sin()).sin() * (theta * 9.0).sin() - 0.12).max(0.0) * 0.75;
        let c = if d.y < 0.0 { Vec3::new(0.18,0.22,0.15) }
            else { Vec3::new(0.65,0.79,0.95).lerp(Vec3::new(0.08,0.27,0.65), d.y.sqrt()) + Vec3::splat(cloud) };
        for v in [c.x,c.y,c.z,1.0] { pixels.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()); }
    }}
    r.set_skybox(Some(&crate::TextureData { width:512, height:256, pixels, srgb:false, hdr:true }))?;
    let mut bed = surface_span(60.0);
    for v in &mut bed.vertices {
        let [x,_,z] = v.position;
        v.position[1] = (-0.38*x - 0.7 + 0.2*(z*0.8).sin()).max(-6.0);
        v.normal = Vec3::new(0.38,1.0,-0.16*(z*0.8).cos()).normalize().to_array();
    }
    let bed = r.upload_mesh(&bed)?;
    let water_mesh = r.upload_mesh(&surface_span(40.0))?;
    let sand = lit(&mut r, [0.47,0.36,0.19,1.0], [0.0;3], 0.85,0.0);
    let stone = lit(&mut r, [0.16,0.19,0.17,1.0], [0.0;3], 0.8,0.0);
    let mut rock = MeshData { vertices:Vec::new(), indices:Vec::new(), has_lightmap_uv:false };
    for y in 0..=12 { for x in 0..=20 {
        let a=x as f32/20.0*std::f32::consts::TAU; let b=y as f32/12.0*std::f32::consts::PI;
        let n=Vec3::new(a.cos()*b.sin(),b.cos(),a.sin()*b.sin());
        let p=n*(1.0+0.12*(a*3.0+b*4.0).sin()+0.07*(a*7.0-b*3.0).cos());
        rock.vertices.push(Vertex { position:p.to_array(), normal:n.to_array(), tangent:[1.0,0.0,0.0,1.0], color:[1.0;4], ..Default::default() });
    }}
    for y in 0..12 { for x in 0..20 {let a=y*21+x;rock.indices.extend([a,a+1,a+22,a,a+22,a+21]);}}
    let rock=r.upload_mesh(&rock)?;
    let mat=r.create_material(&MaterialDesc { name:"Water appearance".into(), features:MaterialFeatures {water:true,double_sided:true,alpha_mode:crate::AlphaMode::Blend,..Default::default()},..Default::default() })?;
    r.set_material_water_slot(mat,0);
    let mut draws=vec![draw(bed,sand,Mat4::IDENTITY,-1),draw(water_mesh,mat,Mat4::IDENTITY,-1)];
    for (p,s) in [(Vec3::new(-0.3,-0.2,-1.5),Vec3::new(1.0,0.8,1.4)),(Vec3::new(1.8,-0.6,1.5),Vec3::new(0.65,0.9,0.8)),(Vec3::new(-2.0,0.25,-5.0),Vec3::new(2.0,1.3,1.4))] {
        draws.push(draw(rock,stone,Mat4::from_scale_rotation_translation(s,glam::Quat::from_rotation_y(0.7),p),-1));
    }
    let sun=crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::new(-0.6,-0.7,-0.3).normalize(),color:[1.0,0.93,0.8],intensity:3.0,range:100.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false};
    let dir=std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_|"/tmp/zest-water-appearance".into());
    std::fs::create_dir_all(&dir)?;
    let planar = [crate::PlanarSurface {point:[0.0;3],normal:[0.0,1.0,0.0],max_roughness:0.6}];
    let mut shots = std::collections::HashMap::new();
    for (name,eye,time,foam,detail) in [
        ("shore",Vec3::new(8.0,4.2,10.0),1.25,1.0,0.35),
        ("shore_later",Vec3::new(8.0,4.2,10.0),2.25,1.0,0.35),
        ("grazing",Vec3::new(8.0,0.75,10.0),1.25,1.0,0.35),
        ("no_foam",Vec3::new(8.0,4.2,10.0),1.25,0.0,0.35),
        ("no_detail",Vec3::new(8.0,4.2,10.0),1.25,1.0,0.0),
        ("legacy_pool",Vec3::new(7.0,2.4,8.0),1.25,0.0,0.15),
        ("legacy_pool_flat",Vec3::new(7.0,2.4,8.0),1.25,0.0,0.0),
        ("planar_pool",Vec3::new(7.0,2.4,8.0),1.25,0.0,0.15),
        ("planar_pool_flat",Vec3::new(7.0,2.4,8.0),1.25,0.0,0.0),
    ] {
        let legacy=name.contains("pool");
        let is_planar=name.starts_with("planar");
        draws[1].planar_index=if is_planar {0} else {-1};
        let mut body=GpuWater::default();body.to_local=Mat4::IDENTITY.to_cols_array_2d();
        body.bounds=[20.0,20.0,0.0,0.65];body.shallow=[0.1,0.45,0.48,4.0];body.deep=[0.01,0.07,0.12,0.95];
        body.look0=[0.06,detail,1.0,0.35];body.look1=[0.3,1.0,0.0,0.0];body.ripple[3]=foam;
        let optics=if legacy {
            body.look0=[0.03,detail,0.28,0.35];body.look1[1]=0.71;
            zestengine_core::WaterOptics::from_legacy(&zestengine_core::WaterLook { optics:None,shallow_color:[0.35,0.75,0.85],deep_color:[0.02,0.25,0.42],absorption_depth:8.0,..Default::default() })
        } else { zestengine_core::WaterOptics::default() };
        body.set_optics([optics.absorption[0],optics.absorption[1],optics.absorption[2],optics.ior],[optics.scattering[0],optics.scattering[1],optics.scattering[2],optics.anisotropy]);
        if !legacy { for (i,w) in zestengine_core::Water::default().resolved_waves().iter().enumerate() {
            body.wave_a[i]=[w.dir.x,w.dir.y,w.k,w.omega];body.wave_b[i][..3].copy_from_slice(&[w.qa,w.amplitude,w.qa*w.k]);body.look1[2]+=1.0;body.look1[3]+=w.amplitude;
        }}
        let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::new(0.0,-0.35,-1.0),Vec3::Y),proj:Mat4::perspective_rh(0.95,WIDTH as f32/HEIGHT as f32,0.05,500.0)};
        let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([WIDTH,HEIGHT]);input.draws=&draws;input.water=std::slice::from_ref(&body);input.water_time=time;
        input.draw_skybox=true;input.light.intensity=0.0;input.light.ambient=[0.12,0.15,0.2];input.light.env_intensity=1.0;input.lights=std::slice::from_ref(&sun);input.postfx.bloom_enabled=false;
        if is_planar {input.planar=&planar;}
        for _ in 0..6 {r.render(&input)?;}
        let (w,h,pixels)=r.debug_read_viewport()?.unwrap();
        anyhow::ensure!(pixels.chunks_exact(4).filter(|p|p[0]>10&&p[0]<245).count()>10000,"empty appearance fixture");
        shots.insert(name,pixels.clone());
        let mut ppm=format!("P6\n{w} {h}\n255\n").into_bytes();for p in pixels.chunks_exact(4){ppm.extend_from_slice(&p[..3]);}
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")),ppm)?;
    }
    for (a,b) in [("shore","no_foam"),("shore","no_detail"),("shore","shore_later"),("legacy_pool","legacy_pool_flat"),("planar_pool","planar_pool_flat")] {
        let delta=shots[a].iter().zip(&shots[b]).filter(|(x,y)|x.abs_diff(**y)>3).count();
        anyhow::ensure!(delta>500,"appearance control has no visible effect: {a} vs {b}: {delta} channels");
        eprintln!("[water appearance] {a} vs {b}: {delta} channels changed");
    }
    Ok(())
}

#[derive(Default)]
struct App { done: bool, error: Option<String>, appearance: bool }
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.done { return; } self.done = true;
        let result = event_loop.create_window(Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT)))
            .map_err(anyhow::Error::from).and_then(|w| {
                let w=Arc::new(w);
                appearance(w.clone())?;
                if !self.appearance { capture(w)?; }
                Ok(())
            });
        self.error = result.err().map(|e| format!("{e:#}")); event_loop.exit();
    }
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}
#[test]
#[ignore = "requires GPU/X11; saves production water reference images under /tmp/zest-water-visual"]
fn capture_water_visual_reference() {
    let _lock = crate::cluster_headless::gpu_lock();
    let mut event_loop = EventLoop::builder().with_x11().with_any_thread(true).build().expect("water visual test requires X11");
    let mut app = App {appearance:std::env::var_os("ZESTENGINE_WATER_APPEARANCE_ONLY").is_some(),..Default::default()};
    event_loop.run_app_on_demand(&mut app).unwrap();
    assert!(app.done); assert!(app.error.is_none(), "{:?}", app.error);
}

