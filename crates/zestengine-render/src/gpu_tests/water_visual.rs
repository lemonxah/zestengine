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

fn recording_fixture()->zestengine_core::WaterSimulationCache {
    zestengine_core::WaterSimulationCache{n:32,origin:[-8.0;2],cell:0.5,frame_rate:1.0,
        frames:(0..3).map(|f|(0..1024).map(|i|{let x=(i%32)as f32*0.5;let z=(i/32)as f32*0.5;
            [0.18*(x*1.3+z*0.7+f as f32).sin(),0.3,0.2,0.25] }).collect()).collect()}
}

fn capture(window: Arc<Window>) -> anyhow::Result<()> {
    let mut r = Renderer::new(window)?;
    r.set_vsync(false);
    r.set_gi_enabled(false);
    let cube = r.upload_mesh(&cube())?;
    let surface = r.upload_mesh_skinned(&surface())?;
    let mesh_count=r.meshes.len();
    for span in [24.0,4.0,12.0] {
        let mut replacement=surface_span(span);
        // Exercise index/vertex capacity changes, not just rewriting positions.
        if span!=12.0 {replacement.indices.truncate(6);replacement.vertices.truncate(99);}
        r.replace_water_mesh(surface,&replacement)?;
        assert_eq!(r.meshes.len(),mesh_count,"LOD changes retain one mesh handle");
    }
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
    let mut water = GpuWater { to_local: Mat4::IDENTITY.to_cols_array_2d(), shallow: [0.1, 0.45, 0.48, 4.0], deep: [0.01, 0.07, 0.12, 1.0], look0: [0.06, 0.15, 1.0, 0.2], look1: [0.2, 1.0, 0.0, 0.0], ..Default::default() };
    water.ripple[3] = 1.0; water.bounds = [6.0, 6.0, 0.0, 1.0];
    for (i, w) in base.resolved_waves().iter().enumerate() {
        water.wave_a[i] = [w.dir.x, w.dir.y, w.k, w.omega];
        water.wave_b[i] = [w.qa, w.amplitude, w.qa * w.k, 0.0];
        water.look1[2] += 1.0; water.look1[3] += w.amplitude;
    }
    let mut sun = crate::LightInstance { kind: crate::LightKind::Directional, position: Vec3::ZERO,
        direction: Vec3::new(0.5, -1.0, -0.4).normalize(), color: [1.0, 0.95, 0.85], intensity: 3.0,
        range: 100.0, spot_inner_deg: 0.0, spot_outer_deg: 0.0, cast_shadows: true, soft_shadows: false,
        radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false };
    let optics = zestengine_core::WaterOptics::default();
    // Nonempty authored geometry forces both desktop and XR water storage
    // to grow, and checks that capture/eye descriptor sets follow the new buffer.
    let authored = zestengine_core::Water { depth: 20.0,
        shape: vec![[-6.0,-6.0],[6.0,-6.0],[6.0,6.0],[-6.0,6.0]], ..Default::default() };
    let mut coverage = Vec::new();
    crate::WaterCoverage::new(&authored).append(&authored, &mut water, &mut coverage);
    let mut reference = None::<Vec<u8>>;
    let mut mirror_reference = None::<f64>;
    let dry_draws: Vec<_> = draws.iter().filter(|d| d.material != water_mat).cloned().collect();
    let mut dry_references = std::collections::HashMap::new();
    let mut boundary_references = std::collections::HashMap::new();
    let mut boundary_unclipped = std::collections::HashMap::new();
    let mut reflected_material = std::collections::HashMap::new();
    let mut media_references = std::collections::HashMap::new();
    let mut connection_references = std::collections::HashMap::new();
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
        ("pool_removed", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_excluded", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_removed", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_excluded", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_boundary", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_boundary_reference", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_boundary", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_boundary_reference", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_detail_off", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_foam", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_foam_off", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_fft", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_fft", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_recording", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_recording", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 1.0),
        ("pool_media_reference", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 0.0),
        ("pool_media_overlap", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_media_reference", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_media_overlap", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 0.0),
        ("pool_low", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 1.0),
        ("pool_balanced", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_low", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 1.0),
        ("pool_rt_balanced", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 1.0),
        ("pool_lod", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_lod", Vec3::new(7.0,4.5,8.0), optics.absorption, 3.0, 0.0),
        ("pool_connection_reference", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_connection", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_connection_reference", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
        ("pool_rt_connection", Vec3::new(7.0, 4.5, 8.0), optics.absorption, 3.0, 0.0),
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
        if name.ends_with("low"){bodies[slot].quality=zestengine_core::WaterQuality::Low.shader_settings();}
        if name.ends_with("balanced"){bodies[slot].quality=zestengine_core::WaterQuality::Balanced.shader_settings();}
        if name=="pool_rt_detail_off" {bodies[slot].look0[1]=0.0;}
        if name.starts_with("pool_rt_foam") {bodies[slot].foam[0]=[0.0,0.0,20.0,1.0];}
        if name=="pool_rt_foam_off" {bodies[slot].ripple[3]=0.0;}
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
        let mut active_coverage = coverage.clone();
        if name.ends_with("fft") {
            let frame=zestengine_core::OceanCache::default().frame(&zestengine_core::OceanSpectrum{patch_size:16.0,significant_height:0.6,..Default::default()},1.25);
            bodies[slot].ocean=[active_coverage.len()as u32,frame.n as u32,frame.size.to_bits(),0];active_coverage.extend_from_slice(&frame.fields);
        }
        if name.ends_with("recording") {
            let cache=recording_fixture();bodies[slot].recording=[active_coverage.len()as u32,cache.n,0,0];
            bodies[slot].recording_window=[cache.origin[0],cache.origin[1],cache.cell,0.0];active_coverage.extend(cache.gpu_field(1.25,&zestengine_core::WaterCacheSettings{looping:false,..Default::default()}));
        }
        if name.ends_with("excluded") {
            bodies[slot].exclusions = [active_coverage.len() as u32, 1, 0, 0];
            active_coverage.extend_from_slice(&Mat4::from_scale(Vec3::splat(1.0/30.0)).to_cols_array_2d());
        }
        if name.contains("boundary") {
            if name.ends_with("reference") {
                let floor_air=zestengine_core::WaterExclusionVolume {half_extents:[100.0,49.6,100.0],..Default::default()}
                    .region(Mat4::from_translation(Vec3::new(0.0,-50.4,0.0))).unwrap();
                bodies[slot].exclusions=[active_coverage.len() as u32,1,0,0];
                active_coverage.extend_from_slice(&floor_air.to_unit.to_cols_array_2d());
            } else {bodies[slot].coverage[3]=0.8f32.to_bits();}
        }
        if name.ends_with("lod"){
            bodies[0].zone_to_water=Mat4::IDENTITY.to_cols_array_2d();bodies[0].zone_lod=[4.0,2.0,0.125,1.0];bodies[0].zone_camera=[0.0,0.85,0.0,0.0];
        }
        let body_end=if name.contains("media_"){
            bodies[0].look1[2]=0.0;bodies[0].look1[3]=0.0;
            if name.ends_with("overlap"){bodies[1]=bodies[0];bodies[0].ownership=[2.0,1.0,0.0,0.0];bodies[1].ownership=[1.0,1.0,0.0,0.0];1}else{0}
        }else if name.contains("connection") {
            let mut receiving=water;receiving.look0=[0.22,0.4,1.8,0.6];receiving.look1[0]=0.6;
            receiving.set_optics([0.7,0.18,0.08,1.4],[0.1,0.16,0.2,0.2]);
            for w in &mut receiving.wave_b{w[0]*=1.8;w[1]*=1.8;w[2]*=1.8;}
            receiving.look1[3]*=1.8;receiving.bounds[3]=0.0;
            // Saturated endpoint band: the root's appearance and displacement
            // must equal rendering the receiving body directly.
            if name.ends_with("reference"){bodies[0]=receiving;}else{bodies[1]=receiving;bodies[0].connections[0]=2;bodies[0].connection_lengths=[5.0,0.0,10.0,0.0];}
            1
        }else{slot};
        input.water = &bodies[..=body_end]; input.water_time = 1.25; input.water_coverage = &active_coverage;
        if name.ends_with("removed") {input.draws = &dry_draws; input.water = &[];}
        input.lights = std::slice::from_ref(&sun);
        input.light.intensity = 0.0; input.light.ambient = [0.12, 0.15, 0.19];
        input.postfx.bloom_enabled = false; input.clear_color = [0.06, 0.09, 0.13, 1.0];
        if name == "pool_underwater" {
            input.postfx.underwater = Some(crate::UnderwaterFx { inside_dry: false, surface: 0.0, shallow_color: [0.1, 0.45, 0.48], deep_color: [0.01, 0.07, 0.12],
                density: 0.1, light_falloff: 0.05, distortion: 0.0, blur: 0.0, caustics: 1.0,
                water_to_world: Mat4::IDENTITY, body: water, rest_surface: 0.0, clock: 1.25 });
        }
        for _ in 0..12 { r.render(&input)?; }
        let (w, h, pixels) = r.debug_read_viewport()?.ok_or_else(|| anyhow::anyhow!("no viewport readback"))?;
        anyhow::ensure!((w,h) == (WIDTH,HEIGHT));
        anyhow::ensure!(pixels.as_chunks::<4>().0.iter().any(|p| p[0] > 30 && p[0] < 240), "water fixture is empty/clipped");
        if name.ends_with("media_reference"){media_references.insert(ray_traced,pixels.clone());}
        if name.ends_with("media_overlap"){
            let baseline:&Vec<u8>=&media_references[&ray_traced];let error=pixels.iter().zip(baseline).map(|(a,b)|a.abs_diff(*b)as f64).sum::<f64>()/pixels.len()as f64;
            eprintln!("[water surface overlap] RT={ray_traced}: mean byte error {error:.5}");anyhow::ensure!(error<0.15,"surface counted overlap twice: {error}");
        }
        if name.ends_with("connection_reference"){connection_references.insert(ray_traced,pixels.clone());}
        if name.ends_with("connection"){
            let baseline:&Vec<u8>=&connection_references[&ray_traced];
            let error=pixels.iter().zip(baseline).map(|(a,b)|a.abs_diff(*b)as f64).sum::<f64>()/pixels.len()as f64;
            eprintln!("[water connection] {name}: endpoint vs receiving material/waves {error:.5}");
            anyhow::ensure!(error<0.5,"{name}: connected endpoint differs from its receiving body ({error})");
        }
        if name == "pool_clear" { reference = Some(pixels.clone()); }
        if name=="pool_rt" || name=="pool_rt_foam" {reflected_material.insert(name,pixels.clone());}
        if name.ends_with("fft") || name.ends_with("recording") {
            let baseline=if ray_traced {&reflected_material["pool_rt"]} else {reference.as_ref().unwrap()};
            let(xs,xe,ys,ye)=if ray_traced{(60,240,100,240)}else{(0,WIDTH,0,HEIGHT)};
            let mut changed=0;for y in ys..ye{for x in xs..xe{let i=((y*WIDTH+x)*4)as usize;if(0..3).any(|k|pixels[i+k].abs_diff(baseline[i+k])>4){changed+=1;}}}
            eprintln!("[water FFT] {name}: {changed} changed pixels");anyhow::ensure!(changed>100,"FFT field did not reach {name}");
        }
        if name=="pool_rt_detail_off" || name=="pool_rt_foam_off" {
            let baseline=&reflected_material[if name=="pool_rt_detail_off" {"pool_rt"} else {"pool_rt_foam"}];
            let mut changed=0;
            for y in 100..240 {for x in 60..240 {let i=((y*WIDTH+x)*4) as usize;
                if (0..3).any(|k|pixels[i+k].abs_diff(baseline[i+k])>4) {changed+=1;}
            }}
            eprintln!("[water reflected material] {name}: {changed} mirror pixels changed");
            anyhow::ensure!(changed>100,"{name}: reflected water ignores its material control");
        }
        if name=="pool_no_caustics" || name=="pool_rt" {boundary_unclipped.insert(ray_traced,pixels.clone());}
        if name.ends_with("boundary") {boundary_references.insert(ray_traced,pixels.clone());}
        if name.ends_with("boundary_reference") {
            let baseline:&Vec<u8>=&boundary_references[&ray_traced];
            let error=pixels.iter().zip(baseline).map(|(a,b)|a.abs_diff(*b) as f64).sum::<f64>()/pixels.len() as f64;
            eprintln!("[water boundary transmission] {name}: floor vs air-prism reference {error:.5}");
            anyhow::ensure!(error<0.3,"{name}: floor clipping differs from dry-volume reference");
            let unclipped:&Vec<u8>=&boundary_unclipped[&ray_traced];
            let mut response=0.0;
            // Mirror-only crop for ray hits, whole image for raster.
            let (xs,xe,ys,ye)=if ray_traced {(80,180,180,220)} else {(0,WIDTH,0,HEIGHT)};
            for y in ys..ye {for x in xs..xe {for k in 0..3 {let i=((y*WIDTH+x)*4+k) as usize;response+=baseline[i].abs_diff(unclipped[i]) as f64;}}}
            response/=((xe-xs)*(ye-ys)*3) as f64;
            anyhow::ensure!(response>0.1,"{name}: optical path fixture did not respond to its floor: {response}");
        }
        if name.ends_with("removed") {dry_references.insert(ray_traced,pixels.clone());}
        if name.ends_with("excluded") {
            let baseline: &Vec<u8> = &dry_references[&ray_traced];
            let error = pixels.iter().zip(baseline).map(|(a,b)|a.abs_diff(*b) as f64).sum::<f64>()/pixels.len() as f64;
            anyhow::ensure!(error<1.0,"{name}: excluded water differs from removed water by {error} bytes/channel");
            eprintln!("[water exclusion surface] {name}: excluded vs removed mean byte error {error:.5}");
        }
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
        if matches!(name, "pool_rt" | "pool_rt_absorbing") {
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
            for px in pixels.as_chunks::<4>().0 { ppm.extend_from_slice(&px[..3]); }
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
        r.set_cluster_settings(crate::ClusterSettings { mode: crate::ClusterMode::Forced, ..r.cluster_settings() });
        r.set_gpu_cull_settings(crate::GpuCullSettings { hiz_occlusion: true, ..r.gpu_cull_settings() });
        let mut input = crate::cascade_tests::input(CameraData { position: eye, view: center, proj });
        input.render_viewport = false; input.draws = &xr_draws; input.water_time = 1.25; input.water_coverage = &coverage;
        input.lights = std::slice::from_ref(&sun); input.light.intensity = 0.0;
        input.light.ambient = [0.12, 0.15, 0.19]; input.postfx.bloom_enabled = false;
        let eyes = [-0.032f32, 0.032].map(|offset| {
            let mut asymmetric = proj;
            asymmetric.z_axis.x = offset.signum() * 0.16;
            crate::XrEyeCamera { view: Mat4::from_translation(Vec3::new(-offset, 0.0, 0.0)) * center,
                proj: asymmetric, postfx: input.postfx }
        });
        let cull_proj = Mat4::perspective_rh(1.25, 1.0, 0.05, 500.0);
        let mut references = Vec::new();
        let mut water_cases = [[GpuWater::default(); crate::MAX_WATER_BODIES]; 8];
        for (bodies, absorption) in water_cases.iter_mut().zip([optics.absorption, [1.2, 0.4, 0.12], optics.absorption,optics.absorption,optics.absorption,optics.absorption,optics.absorption,optics.absorption]) {
            let mut body = water;
            body.set_optics([absorption[0], absorption[1], absorption[2], optics.ior],
                [optics.scattering[0], optics.scattering[1], optics.scattering[2], optics.anisotropy]);
            bodies[crate::MAX_WATER_BODIES - 1] = body;
        }
        water_cases[2][crate::MAX_WATER_BODIES - 1].look1[1] = 0.0;
        let fft=zestengine_core::OceanCache::default().frame(&zestengine_core::OceanSpectrum{patch_size:16.0,significant_height:0.6,..Default::default()},1.25);
        let mut fft_coverage=coverage.clone();water_cases[3][crate::MAX_WATER_BODIES-1].ocean=[fft_coverage.len()as u32,fft.n as u32,fft.size.to_bits(),0];fft_coverage.extend_from_slice(&fft.fields);
        let recorded=recording_fixture();let mut recorded_coverage=coverage.clone();
        water_cases[4][crate::MAX_WATER_BODIES-1].recording=[recorded_coverage.len()as u32,recorded.n,0,0];water_cases[4][crate::MAX_WATER_BODIES-1].recording_window=[recorded.origin[0],recorded.origin[1],recorded.cell,0.0];
        recorded_coverage.extend(recorded.gpu_field(1.25,&zestengine_core::WaterCacheSettings{looping:false,..Default::default()}));
        water_cases[5][crate::MAX_WATER_BODIES-1].quality=zestengine_core::WaterQuality::Low.shader_settings();
        water_cases[6][crate::MAX_WATER_BODIES-1].quality=zestengine_core::WaterQuality::Balanced.shader_settings();
        let lod=&mut water_cases[7][crate::MAX_WATER_BODIES-1];lod.zone_to_water=Mat4::IDENTITY.to_cols_array_2d();lod.zone_lod=[4.0,2.0,0.125,1.0];lod.zone_camera=[0.0,0.85,0.0,0.0];
        for (name, bodies) in ["clear", "absorbing", "unbent","fft","recording","low","balanced","lod"].into_iter().zip(&water_cases) {
            input.water_coverage=if name=="fft"{&fft_coverage}else if name=="recording"{&recorded_coverage}else{&coverage};
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
                else if ["absorbing","unbent","fft","recording"].contains(&name) {
                    let delta = pixels.iter().zip(&references[i]).filter(|(a,b)| a.abs_diff(**b) > 3).count();
                    anyhow::ensure!(delta > 100, "{label}: optical/refraction control does not reach this eye ({delta} changed channels)");
                }
                let dir = std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp/zest-water-visual".into());
                let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
                for px in pixels.as_chunks::<4>().0 { ppm.extend_from_slice(&px[..3]); }
                std::fs::write(std::path::Path::new(&dir).join(format!("{label}.ppm")), ppm)?;
            }
        }
        // Fog and the moving waterline must also agree between independently
        // rendered eyes and multiview, including MSAA depth resolve.
        for (name, eye) in [("underwater", Vec3::new(3.0, -0.65, 4.0)), ("waterline", Vec3::new(3.0, 0.04, 4.0)), ("dry_cabin", Vec3::new(3.0, -0.65, 4.0)),("media",Vec3::new(3.0,-0.65,4.0))] {
            let center = Mat4::look_at_rh(eye, Vec3::new(0.0, eye.y - 0.1, 0.0), Vec3::Y);
            let mut input = crate::cascade_tests::input(CameraData { position: eye, view: center, proj });
            input.render_viewport = false; input.draws = &xr_draws; input.water_time = 1.25; input.water_coverage = &coverage;
            let mut case_bodies = water_cases[0];
            let mut case_coverage = coverage.clone();
            let dry = name == "dry_cabin";
            if name=="media"{let root=crate::MAX_WATER_BODIES-1;
                case_bodies[root].ownership=[1.0,1.0,0.0,0.0];case_bodies[root].underwater=[0.1,0.05,1.0,0.0];
                case_bodies[root-1]=case_bodies[root];case_bodies[root-1].ownership[0]=2.0;case_bodies[root-1].bounds[1]=2.0;
                case_bodies[root-1].set_optics([0.9,0.7,0.3,1.333],[0.05,0.04,0.02,0.65]);
            }
            if dry {
                let body = &mut case_bodies[crate::MAX_WATER_BODIES-1];
                body.exclusions = [case_coverage.len() as u32,1,0,0];
                case_coverage.extend_from_slice(&(Mat4::from_scale(Vec3::new(1.5,1.0,1.5))*Mat4::from_translation(-eye)).to_cols_array_2d());
            }
            input.water = &case_bodies; input.water_coverage = &case_coverage; input.lights = std::slice::from_ref(&sun);
            input.light.intensity = 0.0; input.light.ambient = [0.12, 0.15, 0.19];
            input.postfx.bloom_enabled = false;
            input.postfx.underwater = Some(crate::UnderwaterFx { inside_dry: dry, surface: 0.0,
                shallow_color: [0.1, 0.45, 0.48], deep_color: [0.01, 0.07, 0.12], density: 0.1,
                light_falloff: 0.05, distortion: 0.0, blur: 0.0, caustics: 1.0,
                water_to_world: Mat4::IDENTITY, body: case_bodies[crate::MAX_WATER_BODIES - 1], rest_surface: 0.0, clock: 1.25 });
            let eyes = [-0.032f32, 0.032].map(|offset| {
                let mut asymmetric = proj; asymmetric.z_axis.x = offset.signum() * 0.16;
                crate::XrEyeCamera { view: Mat4::from_translation(Vec3::new(-offset, 0.0, 0.0)) * center,
                    proj: asymmetric, postfx: input.postfx }
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
                    for px in image.as_chunks::<4>().0 { ppm.extend_from_slice(&px[..3]); }
                    std::fs::write(std::path::Path::new(&dir).join(format!("xr_{name}_{samples}x_{e}.ppm")), ppm)?;
                }
                }
            }
        }
        if stereo.is_some() {
            let large = Target::with_extent(&r, 2, 1024, 1024)?;
            let dry_draws: Vec<_> = xr_draws.iter().filter(|d| d.material != water_mat).cloned().collect();
            let measured_waters=[zestengine_core::WaterQuality::High,zestengine_core::WaterQuality::Balanced,zestengine_core::WaterQuality::Low].map(|q|{let mut w=water_cases[0][crate::MAX_WATER_BODIES-1];w.quality=q.shader_settings();w});
            let benchmark = (|| -> anyhow::Result<()> {
                r.set_material_water_slot(water_mat, 0);
                for samples in [1, 4] {
                    let limits = unsafe { r.ctx.instance.get_physical_device_properties(r.ctx.physical_device) }.limits;
                    if !(limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts)
                        .contains(ash::vk::SampleCountFlags::from_raw(samples as u32)) { continue; }
                    r.xr_msaa = samples;
                    for (mode,quality) in [("high",Some(0)),("balanced",Some(1)),("low",Some(2)),("no water",None)] {
                    input.draws = if quality.is_some(){&xr_draws}else{&dry_draws};
                    input.water = if let Some(q)=quality{std::slice::from_ref(&measured_waters[q])}else{&[]};
                    let mut copy_ms = Vec::new(); let mut scene_ms = Vec::new();
                    for frame in 0..8 {
                        r.render_xr_frame(&input)?;
                        r.render_xr_stereo(large.raw(), FORMAT.as_raw() as u32, 1024, 1024, &eyes, (center, cull_proj), &input)?;
                        unsafe { r.ctx.device.device_wait_idle()?; }
                        if frame >= 3
                            && let Some((copy, scene)) = r.xr_water_timing() { copy_ms.push(copy); scene_ms.push(scene); }
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
    let sun=crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::new(-0.6,-0.7,-0.3).normalize(),color:[1.0,0.93,0.8],intensity:3.0,range:100.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false};
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
        let mut body = GpuWater { to_local: Mat4::IDENTITY.to_cols_array_2d(), bounds: [20.0,20.0,0.0,0.65], shallow: [0.1,0.45,0.48,4.0], deep: [0.01,0.07,0.12,0.95], look0: [0.06,detail,1.0,0.35], look1: [0.3,1.0,0.0,0.0], ..Default::default() };body.ripple[3]=foam;
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
        anyhow::ensure!(pixels.as_chunks::<4>().0.iter().filter(|p|p[0]>10&&p[0]<245).count()>10000,"empty appearance fixture");
        shots.insert(name,pixels.clone());
        let mut ppm=format!("P6\n{w} {h}\n255\n").into_bytes();for p in pixels.as_chunks::<4>().0{ppm.extend_from_slice(&p[..3]);}
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")),ppm)?;
    }
    for (a,b) in [("shore","no_foam"),("shore","no_detail"),("shore","shore_later"),("legacy_pool","legacy_pool_flat"),("planar_pool","planar_pool_flat")] {
        let delta=shots[a].iter().zip(&shots[b]).filter(|(x,y)|x.abs_diff(**y)>3).count();
        anyhow::ensure!(delta>500,"appearance control has no visible effect: {a} vs {b}: {delta} channels");
        eprintln!("[water appearance] {a} vs {b}: {delta} channels changed");
    }
    Ok(())
}

// Render actual receiver shading with the water surface hidden. A top-down
// camera and vertical sun make wet/dry classification independent of refraction
// and screen-space lookup. This catches rectangular caustic leakage, not just
// errors in the isolated coverage query.
fn coverage_receivers(window: Arc<Window>) -> anyhow::Result<()> {
    use zestengine_core::{Water, WaterBody, RiverPoint};
    let mut r = Renderer::new(window)?;
    r.set_vsync(false);
    r.set_gi_enabled(false);
    let mut floor = surface_span(16.0);
    for v in &mut floor.vertices { v.position[1] = -2.0; }
    let mesh = r.upload_mesh(&floor)?;
    let mat = lit(&mut r, [0.4,0.4,0.4,1.0], [0.0;3], 1.0, 0.0);
    let draws = [draw(mesh, mat, Mat4::IDENTITY, -1)];
    let camera = CameraData { position: Vec3::new(0.0,12.0,0.0),
        view: Mat4::look_at_rh(Vec3::new(0.0,12.0,0.0), Vec3::ZERO, Vec3::NEG_Z),
        proj: Mat4::orthographic_rh(-8.0,8.0,-5.0,5.0,0.05,50.0) };
    let sun = crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Y,
        color:[1.0;3],intensity:1.5,range:100.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:false,
        soft_shadows:false,radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false};
    let polygon = Water { shape:vec![[-6.0,-4.0],[6.0,-4.0],[6.0,-2.0],[-2.0,-2.0],[-2.0,4.0],[-6.0,4.0]], depth:10.0, ..Default::default() };
    let river = Water { body:WaterBody::River, river:vec![
        RiverPoint {position:[-6.0,2.0,-3.0],width:1.5,depth:10.0,..Default::default()},
        RiverPoint {position:[0.0,0.0,1.0],width:2.0,depth:10.0,..Default::default()},
        RiverPoint {position:[6.0,-1.0,-2.0],width:1.0,depth:10.0,..Default::default()},
    ], ..Default::default() };
    let custom=Water{body:WaterBody::Custom,depth:10.0,custom_mesh:Some(Arc::new(zestengine_core::WaterMesh::new(&[
        Vec3::new(-6.0,1.0,-4.0),Vec3::new(-1.0,2.0,-4.0),Vec3::new(-6.0,1.0,4.0),Vec3::new(1.0,3.0,-4.0),Vec3::new(6.0,1.0,-4.0),Vec3::new(6.0,1.0,4.0)],&[0,1,2,3,4,5]))),..Default::default()};
    let ocean=Water{body:WaterBody::Ocean,shape:vec![[-2.0,-3.0],[2.0,-3.0],[2.0,3.0],[-2.0,3.0]],depth:10.0,..Default::default()};
    for (name,water) in [("concave",polygon),("sloping_river",river),("custom_mesh",custom),("ocean_shore",ocean)] {
        let mut body = GpuWater { to_local: Mat4::IDENTITY.to_cols_array_2d(), bounds: [8.0,5.0,0.0,1.0], look0: [0.1,0.35,1.0,0.0], ..Default::default() };
        body.set_optics([0.05,0.02,0.01,1.333],[0.0;4]);
        let mut coverage = Vec::new();
        crate::WaterCoverage::new(&water).append(&water, &mut body, &mut coverage);
        let mut images = Vec::new();
        for strength in [0.0,1.0] {
            body.bounds[3] = strength;
            let mut input = crate::cascade_tests::input(camera);
            input.viewport_extent = Some([WIDTH,HEIGHT]);input.draws=&draws;
            input.water=std::slice::from_ref(&body);input.water_coverage=&coverage;input.water_time=1.25;
            input.lights=std::slice::from_ref(&sun);input.light.intensity=0.0;input.light.ambient=[0.0;3];
            input.light.env_intensity=0.0;input.postfx.bloom_enabled=false;
            for _ in 0..4 { r.render(&input)?; }
            images.push(r.debug_read_viewport()?.unwrap().2);
        }
        let inv = (camera.proj*camera.view).inverse();
        let (mut wet_changed,mut dry_changed) = (0usize,0usize);
        for y in 0..HEIGHT { for x in 0..WIDTH {
            let ndc=glam::Vec4::new((x as f32+0.5)/WIDTH as f32*2.0-1.0,1.0-(y as f32+0.5)/HEIGHT as f32*2.0,0.0,1.0);
            let p=inv*ndc;let p=p.truncate()/p.w;
            let i=((y*WIDTH+x)*4) as usize;
            if (0..3).any(|k|images[0][i+k].abs_diff(images[1][i+k])>2) {
                if water.column(glam::Vec2::new(p.x,p.z)).is_some() {wet_changed+=1;} else {dry_changed+=1;}
            }
        }}
        eprintln!("[water coverage receivers] {name}: {wet_changed} wet pixels changed, {dry_changed} dry pixels changed");
        anyhow::ensure!(wet_changed>500,"{name}: caustics missing from wet receivers");
        anyhow::ensure!(dry_changed<20,"{name}: caustics leaked outside the authored footprint ({dry_changed} pixels)");
        let dir=std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_|"/tmp/zest-water-visual".into());
        std::fs::create_dir_all(&dir)?;
        let mut ppm=format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
        for p in images[1].as_chunks::<4>().0 {ppm.extend_from_slice(&p[..3]);}
        std::fs::write(std::path::Path::new(&dir).join(format!("coverage_{name}.ppm")),ppm)?;
    }
    Ok(())
}

// A still pool as the pool preset makes it (no waves, ripple strength 0.15) at
// a wide ripple scale, 0.33 m deep under a ceiling lamp: the scene where no
// caustics showed. That little water bends light too little for the detail
// ripples to focus, so their focusing is lifted to WATER_CAUSTIC_FOCUS
// (water_caustics.glsl). Receiver shading only, like `coverage_receivers`:
// the floor from above, then a ceiling over a lamp under the surface. The
// "murky" cases are that scene's later fluid: a baked (flat) simulation cache
// on the surface and a dyed fluid that absorbs most light within the pool's
// depth but scatters little, which must dim the floor, not the pattern.
fn shallow_pool_caustics(window: Arc<Window>) -> anyhow::Result<()> {
    use zestengine_core::{Water, WaterQuality};
    let mut r = Renderer::new(window)?;
    r.set_vsync(false);
    r.set_gi_enabled(false);
    let mesh = r.upload_mesh(&surface_span(16.0))?;
    let mat = lit(&mut r, [0.4,0.4,0.4,1.0], [0.0;3], 1.0, 0.0);
    let mut water = Water { size:[12.0,12.0], depth:1.0, ..Default::default() };
    water.make_pool();
    let clear = zestengine_core::WaterOptics::from_legacy(&water.look).sanitized();
    let dyed = zestengine_core::WaterOptics { absorption:[4.6467953,2.614966,2.358805], scattering:[0.003204415,0.3850339,0.04119517],
        ior:1.146, anisotropy:0.25 }.sanitized();
    let still = zestengine_core::WaterSimulationCache { n:32, origin:[-8.0;2], cell:0.5, frame_rate:1.0,
        frames:vec![vec![[0.0,0.3,0.0,0.0];1024];2] };
    let lamp = |y: f32| crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(-0.8,y,0.0),direction:Vec3::NEG_Y,
        color:[1.0;3],intensity:40.0,range:8.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:false,
        soft_shadows:false,radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false};
    // Pixels lit well inside the 8-bit range: how many differ, and the RMS of
    // their brightness ratio.
    let compare = |a: &[u8], b: &[u8]| {
        let (mut changed,mut sum,mut sum2,mut n) = (0usize,0.0f64,0.0f64,0usize);
        for (a,b) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
            if a[0] < 60 || a[0] > 230 { continue; }
            let ratio = (b[0] as f64/a[0] as f64).powf(2.2);
            if a[0].abs_diff(b[0]) > 2 { changed += 1; }
            sum += ratio; sum2 += ratio*ratio; n += 1;
        }
        let mean = sum/n.max(1) as f64;
        (changed, n, (sum2/n.max(1) as f64-mean*mean).max(0.0).sqrt())
    };
    let dir=std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_|"/tmp/zest-water-visual".into());
    std::fs::create_dir_all(&dir)?;
    let save = |name: &str, image: &[u8]| {
        let mut ppm=format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
        for p in image.as_chunks::<4>().0 {ppm.extend_from_slice(&p[..3]);}
        std::fs::write(std::path::Path::new(&dir).join(format!("shallow_{name}.ppm")),ppm)
    };
    // (receiver plane, camera, lamp height, water level, projection strength)
    let floor = (Mat4::IDENTITY, CameraData { position: Vec3::new(0.0,6.0,0.0),
        view: Mat4::look_at_rh(Vec3::new(0.0,6.0,0.0), Vec3::ZERO, Vec3::NEG_Z),
        proj: Mat4::orthographic_rh(-2.0,2.0,-1.25,1.25,0.05,50.0) }, 2.124, 0.3341867, 0.0);
    let ceiling = (Mat4::from_translation(Vec3::new(0.0,2.0,0.0))*Mat4::from_rotation_x(std::f32::consts::PI),
        CameraData { position: Vec3::new(0.0,0.5,0.0),
        view: Mat4::look_at_rh(Vec3::new(0.0,0.5,0.0), Vec3::new(0.0,2.0,0.0), Vec3::Z),
        proj: Mat4::orthographic_rh(-2.0,2.0,-1.25,1.25,0.05,50.0) }, -0.3, 0.0, 1.0);
    for (name,(plane,camera,lamp_y,level,projection),ripples,scale,quality,murky,visible) in [
        ("floor", floor, 0.15, 0.28, WaterQuality::High, false, true),
        ("floor_scale1", floor, 0.15, 1.0, WaterQuality::High, false, true),
        ("floor_balanced", floor, 0.15, 0.28, WaterQuality::Balanced, false, true),
        ("floor_low", floor, 0.15, 0.28, WaterQuality::Low, false, false),
        ("floor_flat", floor, 0.0, 0.28, WaterQuality::High, false, false),
        ("floor_murky", floor, 0.23, 0.15, WaterQuality::High, true, true),
        ("floor_murky_flat", floor, 0.0, 0.15, WaterQuality::High, true, false),
        ("ceiling", ceiling, 0.15, 0.28, WaterQuality::High, false, true),
        ("ceiling_flat", ceiling, 0.0, 0.28, WaterQuality::High, false, false),
    ] {
        let draws = [draw(mesh, mat, plane, -1)];
        let light = lamp(lamp_y);
        let mut images = Vec::new();
        // Without the pattern: no caustics on the floor, no ripples under the
        // ceiling (the projected light itself stays).
        for pattern in [false,true] {
            let mut body = GpuWater { to_local: Mat4::from_translation(Vec3::new(0.0,level,0.0)).inverse().to_cols_array_2d(), bounds: [6.0,6.0,0.0,if pattern || projection > 0.0 {1.0} else {0.0}], look0: [0.03,if pattern || projection == 0.0 {ripples} else {0.0},scale,0.35], ..Default::default() };
            body.extra[0] = projection;
            body.quality = quality.shader_settings();
            let optics = if murky { dyed } else { clear };
            body.set_optics([optics.absorption[0],optics.absorption[1],optics.absorption[2],optics.ior],
                [optics.scattering[0],optics.scattering[1],optics.scattering[2],optics.anisotropy]);
            let mut coverage = Vec::new();
            // The murky pool is 0.4 m deep by its own setting, like the scene's.
            let pool = Water { depth: if murky {0.4} else {water.depth}, ..water.clone() };
            crate::WaterCoverage::new(&pool).append(&pool, &mut body, &mut coverage);
            if murky {
                body.recording = [coverage.len() as u32,still.n,0,0];
                body.recording_window = [still.origin[0],still.origin[1],still.cell,0.0];
                coverage.extend(still.gpu_field(1.25,&zestengine_core::WaterCacheSettings{looping:false,..Default::default()}));
            }
            let mut input = crate::cascade_tests::input(camera);
            input.viewport_extent = Some([WIDTH,HEIGHT]);input.draws=&draws;
            input.water=std::slice::from_ref(&body);input.water_coverage=&coverage;input.water_time=1.25;
            input.lights=std::slice::from_ref(&light);input.light.intensity=0.0;input.light.ambient=[0.0;3];
            input.light.env_intensity=0.0;input.postfx.bloom_enabled=false;input.postfx.tonemap=0;
            for _ in 0..4 { r.render(&input)?; }
            images.push(r.debug_read_viewport()?.unwrap().2);
        }
        let (changed,lit_pixels,contrast) = compare(&images[0],&images[1]);
        eprintln!("[water shallow caustics] {name}: {changed} of {lit_pixels} lit pixels changed, contrast {contrast:.3}");
        anyhow::ensure!(lit_pixels > 100_000, "{name}: receiver is not lit ({lit_pixels} pixels)");
        if visible {
            anyhow::ensure!(changed > lit_pixels/3 && contrast > 0.08, "{name}: caustics do not show in shallow water ({changed} of {lit_pixels} pixels, contrast {contrast:.3})");
        } else {
            anyhow::ensure!(changed < 20, "{name}: a surface without detail ripples still focuses light ({changed} pixels)");
        }
        save(name, &images[1])?;
    }
    Ok(())
}

fn mapped_pool_caustics(window: Arc<Window>) -> anyhow::Result<()> {
    use zestengine_core::{Water, WaterQuality};
    let mut r = Renderer::new(window)?;
    r.set_vsync(false);
    r.set_gi_enabled(false);
    let mesh = r.upload_mesh(&surface_span(16.0))?;
    let floor_mat = lit(&mut r, [0.4, 0.4, 0.4, 1.0], [0.0; 3], 1.0, 0.0);
    let water_mat = r.create_material(&MaterialDesc { name: "Mapped caustics water".into(),
        features: MaterialFeatures { water: true, double_sided: true, alpha_mode: crate::AlphaMode::Blend, ..Default::default() }, ..Default::default() })?;
    r.set_material_water_slot(water_mat, 0);
    let level = 0.3341867;
    let draws = [draw(mesh, floor_mat, Mat4::IDENTITY, -1), draw(mesh, water_mat, Mat4::from_translation(Vec3::new(0.0, level, 0.0)), 0)];
    let mut water = Water { size: [12.0, 12.0], depth: 1.0, ..Default::default() };
    water.make_pool();
    let light = crate::LightInstance { kind: crate::LightKind::Point, position: Vec3::new(-0.8, 2.124, 0.0), direction: Vec3::NEG_Y,
        color: [1.0; 3], intensity: 40.0, range: 8.0, spot_inner_deg: 0.0, spot_outer_deg: 0.0, cast_shadows: false,
        soft_shadows: false, radius: 0.0, shadow_bias: 0.0, shadow_clip: None, baked: false };
    let camera = CameraData { position: Vec3::new(0.0, 6.0, 0.0),
        view: Mat4::look_at_rh(Vec3::new(0.0, 6.0, 0.0), Vec3::ZERO, Vec3::NEG_Z),
        proj: Mat4::orthographic_rh(-2.0, 2.0, -1.25, 1.25, 0.05, 50.0) };
    let mut images = Vec::new();
    for mapped in [false, true] {
        let mut body = GpuWater { to_local: Mat4::from_translation(Vec3::new(0.0, level, 0.0)).inverse().to_cols_array_2d(),
            bounds: [6.0, 6.0, 0.0, 1.0], shallow: [0.85, 0.95, 0.95, 20.0], deep: [0.6, 0.8, 0.85, 1.0],
            look0: [0.04, 0.15, 0.28, 0.0], look1: [0.0, 1.0, 0.0, 0.0], ..Default::default() };
        body.quality = WaterQuality::High.shader_settings();
        let optics = zestengine_core::WaterOptics::from_legacy(&water.look).sanitized();
        body.set_optics([optics.absorption[0], optics.absorption[1], optics.absorption[2], optics.ior],
            [optics.scattering[0], optics.scattering[1], optics.scattering[2], optics.anisotropy]);
        if mapped {
            body.caustic_map[0] = crate::CAUSTIC_MAP_REQUESTED;
        }
        let mut coverage = Vec::new();
        crate::WaterCoverage::new(&water).append(&water, &mut body, &mut coverage);
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([WIDTH, HEIGHT]);
        input.draws = &draws;
        input.water = std::slice::from_ref(&body);
        input.water_coverage = &coverage;
        input.water_time = 1.25;
        input.lights = std::slice::from_ref(&light);
        input.light.intensity = 0.0;
        input.light.ambient = [0.02; 3];
        input.light.env_intensity = 0.0;
        input.postfx.bloom_enabled = false;
        input.postfx.tonemap = 0;
        for _ in 0..6 {
            r.render(&input)?;
        }
        if mapped {
            anyhow::ensure!(r.caustics.active(), "the mapped pool did not allocate or schedule caustic maps");
        }
        images.push(r.debug_read_viewport()?.unwrap().2);
    }
    let pixels = (images[0].len() / 4) as f64;
    let (mut sum, mut far, mut lit_pixels) = (0.0f64, 0usize, 0usize);
    for (a, b) in images[0].as_chunks::<4>().0.iter().zip(images[1].as_chunks::<4>().0) {
        let d = (0..3).map(|k| (a[k] as f64 - b[k] as f64).abs()).sum::<f64>() / 3.0;
        sum += d;
        if d > 24.0 { far += 1; }
        if a[0] > 40 { lit_pixels += 1; }
    }
    let mean = sum / pixels;
    eprintln!("[water mapped caustics] mean difference to exact {mean:.3} levels, {:.3}% of pixels off by more than 24, {lit_pixels} lit pixels", 100.0 * far as f64 / pixels);
    let dir = std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp/zest-water-visual".into());
    std::fs::create_dir_all(&dir)?;
    for (name, image) in ["exact", "mapped"].iter().zip(&images) {
        let mut ppm = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
        for p in image.as_chunks::<4>().0 { ppm.extend_from_slice(&p[..3]); }
        std::fs::write(std::path::Path::new(&dir).join(format!("caustics_{name}.ppm")), ppm)?;
    }
    anyhow::ensure!(lit_pixels > 50_000, "the floor is not lit ({lit_pixels} pixels)");
    anyhow::ensure!(mean > 0.005, "the mapped image is identical to the exact one: the map was not used");
    if r.ctx.multiview {
        use super::super::xr_cluster::{FORMAT, H as XH, Target, W as XW};
        let stereo = Target::new(&r, 2)?;
        let singles = [Target::new(&r, 1)?, Target::new(&r, 1)?];
        let eye = Vec3::new(0.0, 3.0, 0.9);
        let center = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, -0.2), Vec3::Y);
        let proj = Mat4::perspective_rh(1.2, 1.0, 0.05, 50.0);
        let eyes = [-0.032f32, 0.032].map(|offset| crate::XrEyeCamera {
            view: Mat4::from_translation(Vec3::new(-offset, 0.0, 0.0)) * center,
            proj,
            postfx: crate::PostFx { bloom_enabled: false, tonemap: 0, ..Default::default() },
        });
        let mut stereo_images = Vec::new();
        let mut eye_images: Vec<[Vec<u8>; 2]> = Vec::new();
        for mapped in [false, true] {
            let mut body = GpuWater { to_local: Mat4::from_translation(Vec3::new(0.0, level, 0.0)).inverse().to_cols_array_2d(),
                bounds: [6.0, 6.0, 0.0, 1.0], shallow: [0.85, 0.95, 0.95, 20.0], deep: [0.6, 0.8, 0.85, 1.0],
                look0: [0.04, 0.15, 0.28, 0.0], look1: [0.0, 1.0, 0.0, 0.0], ..Default::default() };
            body.quality = WaterQuality::High.shader_settings();
            if mapped {
                body.caustic_map[0] = crate::CAUSTIC_MAP_REQUESTED;
            }
            let mut coverage = Vec::new();
            crate::WaterCoverage::new(&water).append(&water, &mut body, &mut coverage);
            let mut input = crate::cascade_tests::input(CameraData { position: eye, view: center, proj });
            input.render_viewport = false;
            input.draws = &draws;
            input.water = std::slice::from_ref(&body);
            input.water_coverage = &coverage;
            input.water_time = 1.25;
            input.lights = std::slice::from_ref(&light);
            input.light.intensity = 0.0;
            input.light.ambient = [0.02; 3];
            input.light.env_intensity = 0.0;
            input.postfx.bloom_enabled = false;
            input.postfx.tonemap = 0;
            for _ in 0..6 {
                r.render_xr_frame(&input)?;
                r.render_xr_stereo(stereo.raw(), FORMAT.as_raw() as u32, XW, XH, &eyes, (center, proj), &input)?;
            }
            stereo_images.push(stereo.read(&r)?);
            for _ in 0..6 {
                r.render_xr_frame(&input)?;
                for e in 0..2 {
                    r.render_xr_eye_indexed(e, singles[e].raw(), FORMAT.as_raw() as u32, XW, XH, eyes[e].view, eyes[e].proj, &input, None)?;
                }
            }
            eye_images.push([singles[0].read(&r)?, singles[1].read(&r)?]);
        }
        let layer = (XW * XH * 4) as usize;
        for e in 0..2 {
            let (a, b) = (&stereo_images[0][e * layer..(e + 1) * layer], &stereo_images[1][e * layer..(e + 1) * layer]);
            let lit = a.as_chunks::<4>().0.iter().filter(|p| p[0] > 40).count();
            let diff = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as f64).sum::<f64>() / layer as f64;
            eprintln!("[water mapped caustics] VR eye {e}: mean byte difference to exact {diff:.4}, {lit} lit pixels");
            anyhow::ensure!(lit > 10_000, "VR eye {e}: the floor is not lit ({lit} pixels)");
            anyhow::ensure!(diff > 0.002, "VR eye {e}: the mapped image is identical to the exact one: the map was not used in VR");
            anyhow::ensure!(diff < 2.0, "VR eye {e}: mapped caustics differ from exact ones (mean {diff:.4})");
        }
        for e in 0..2 {
            let (a, b) = (&eye_images[0][e], &eye_images[1][e]);
            let diff = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as f64).sum::<f64>() / layer as f64;
            eprintln!("[water mapped caustics] VR single eye {e}: mean byte difference to exact {diff:.4}");
            anyhow::ensure!(diff > 0.002, "VR single eye {e}: the mapped image is identical to the exact one: the map was not used");
            anyhow::ensure!(diff < 2.0, "VR single eye {e}: mapped caustics differ from exact ones (mean {diff:.4})");
        }
        stereo.destroy(&r);
        for target in singles {
            target.destroy(&r);
        }
    }
    anyhow::ensure!(mean < 2.0 && (far as f64) < pixels * 0.01, "mapped caustics differ from exact ones: mean {mean:.3}, {far} pixels far off");
    Ok(())
}

fn dry_underwater_views(window: Arc<Window>) -> anyhow::Result<()> {
    let mut r=Renderer::new(window)?;
    r.set_vsync(false);r.set_gi_enabled(false);
    let mesh=r.upload_mesh(&cube())?;
    let mat=lit(&mut r,[0.65,0.12,0.04,1.0],[0.0;3],0.9,0.0);
    let wall=draw(mesh,mat,Mat4::from_scale_rotation_translation(Vec3::new(20.0,20.0,0.2),glam::Quat::IDENTITY,Vec3::new(0.0,-1.0,-2.0)),-1);
    let draws=[wall];
    let eye=Vec3::new(0.0,-1.0,4.0);
    let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::new(0.0,-1.0,0.0),Vec3::Y),proj:Mat4::perspective_rh(0.95,WIDTH as f32/HEIGHT as f32,0.05,50.0)};
    let mut shots=std::collections::HashMap::new();
    for name in ["clear_room","wet_room","dry_room","dry_window","finite_pool","pool_air_reference","concave_pool","concave_air_reference","media_reference","media_overlap","media_split"] {
        let mut authored=zestengine_core::Water {depth:20.0,..Default::default()};
        if name=="dry_room" || name=="dry_window" {
            let size=if name=="dry_room" {Vec3::splat(15.0)} else {Vec3::ONE};
            authored.exclusions.push(zestengine_core::WaterExclusionVolume {half_extents:size.to_array(),..Default::default()}
                .region(Mat4::from_translation(eye)).unwrap());
        }
        let mut body = GpuWater { to_local: Mat4::IDENTITY.to_cols_array_2d(), bounds: [10.0,10.0,0.0,0.0], ..Default::default() };
        if name=="finite_pool" || name=="concave_pool" {
            authored.depth=2.0;body.bounds=[4.0,5.0,0.0,0.0];
            if name=="concave_pool" {
                authored.shape=vec![[-4.0,-5.0],[4.0,-5.0],[4.0,5.0],[-4.0,5.0],[-4.0,1.0],[2.0,1.0],[2.0,-1.0],[-4.0,-1.0]];
            }
        }
        if name.ends_with("air_reference") {
            // An endless medium with air prisms removed must give the same
            // image as clipping to the authored basin, including re-entry.
            body.bounds=[0.0;4];
            for (lo,hi) in [([-100.0,-100.0,-100.0],[-4.0,100.0,100.0]),([4.0,-100.0,-100.0],[100.0,100.0,100.0]),
                ([-100.0,-100.0,-100.0],[100.0,100.0,-5.0]),([-100.0,-100.0,5.0],[100.0,100.0,100.0]),
                ([-100.0,-100.0,-100.0],[100.0,-2.0,100.0])] {
                let lo=Vec3::from(lo);let hi=Vec3::from(hi);
                authored.exclusions.push(zestengine_core::WaterExclusionVolume {half_extents:((hi-lo)*0.5).to_array(),..Default::default()}
                    .region(Mat4::from_translation((hi+lo)*0.5)).unwrap());
            }
            if name=="concave_air_reference" {
                authored.exclusions.push(zestengine_core::WaterExclusionVolume {half_extents:[3.0,100.0,1.0],..Default::default()}
                    .region(Mat4::from_translation(Vec3::new(-1.0,0.0,0.0))).unwrap());
            }
        }
        body.set_optics([0.32,0.065,0.025,1.333],[0.015,0.035,0.045,0.65]);
        let mut coverage=Vec::new();crate::WaterCoverage::new(&authored).append(&authored,&mut body,&mut coverage);
        let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([WIDTH,HEIGHT]);input.draws=&draws;
        let mut media=[body;2];
        if name.starts_with("media_"){
            body.underwater=[0.35,0.05,1.0,0.0];body.shallow=[0.1,0.45,0.48,1.0];body.deep=[0.01,0.07,0.12,1.0];
            media=[body;2];media[0].ownership=[2.0,1.0,0.0,0.0];media[1].ownership=[1.0,1.0,0.0,0.0];
            if name=="media_split"{media[1].ownership[0]=3.0;media[1].bounds[1]=2.0;media[1].set_optics([0.9,0.7,0.3,1.333],[0.05,0.04,0.02,0.65]);}
        }
        input.water=if name=="media_overlap"||name=="media_split"{&media}else{std::slice::from_ref(&body)};
        input.water_coverage=&coverage;input.light.ambient=[0.2;3];input.postfx.bloom_enabled=false;
        if name!="clear_room" {
            input.postfx.underwater=Some(crate::UnderwaterFx {inside_dry:name.starts_with("dry"),surface:0.0,
                shallow_color:[0.1,0.45,0.48],deep_color:[0.01,0.07,0.12],density:0.35,light_falloff:0.05,
                distortion:0.0,blur:0.0,caustics:0.0,water_to_world:Mat4::IDENTITY,body,rest_surface:0.0,clock:0.0});
        }
        for _ in 0..5 {r.render(&input)?;}
        shots.insert(name,r.debug_read_viewport()?.unwrap().2);
    }
    let delta=|a:&str,b:&str|shots[a].iter().zip(&shots[b]).map(|(x,y)|x.abs_diff(*y) as f64).sum::<f64>()/shots[a].len() as f64;
    let dry=delta("dry_room","clear_room");let wet=delta("wet_room","clear_room");let window=delta("dry_window","clear_room");
    eprintln!("[water exclusion fog] dry room vs clear {dry:.5}; wet vs clear {wet:.5}; dry window vs clear {window:.5}");
    anyhow::ensure!(dry<0.1,"fog or depth darkening leaked into a completely dry room: {dry}");
    anyhow::ensure!(wet>3.0,"underwater fog fixture has no medium response: {wet}");
    anyhow::ensure!(window>1.0 && window<wet && delta("dry_window","wet_room")>0.5,"dry window must retain water beyond its air segment");
    let pool_error=delta("finite_pool","pool_air_reference");let concave_error=delta("concave_pool","concave_air_reference");
    eprintln!("[water boundary fog] pool vs air-prism reference {pool_error:.5}; concave vs reference {concave_error:.5}");
    anyhow::ensure!(pool_error<0.1 && concave_error<0.1,"basin optical path or dry receiver differs from independent air-prism reference");
    anyhow::ensure!(delta("finite_pool","wet_room")>1.0 && delta("concave_pool","finite_pool")>1.0,"boundary fixture must exercise exits and re-entry");
    let overlap_error=delta("media_reference","media_overlap");let split_delta=delta("media_overlap","media_split");
    eprintln!("[water multiple media] identical overlap error {overlap_error:.5}; second material response {split_delta:.5}");
    anyhow::ensure!(overlap_error<0.1,"overlapping water counted the medium twice: {overlap_error}");
    anyhow::ensure!(split_delta>1.0,"view ray did not enter the second water material: {split_delta}");
    let dir=std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_|"/tmp/zest-water-visual".into());std::fs::create_dir_all(&dir)?;
    for (name,pixels) in shots {let mut ppm=format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();for p in pixels.as_chunks::<4>().0{ppm.extend_from_slice(&p[..3]);}
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")),ppm)?;}
    Ok(())
}

fn partial_exclusion_surface(window: Arc<Window>) -> anyhow::Result<()> {
    let mut r=Renderer::new(window)?;r.set_vsync(false);r.set_gi_enabled(false);
    let mut floor=surface_span(16.0);for v in &mut floor.vertices {v.position[1] = -2.0;}
    let floor=r.upload_mesh(&floor)?;let plane=r.upload_mesh(&surface_span(16.0))?;
    let ground=lit(&mut r,[0.5,0.18,0.08,1.0],[0.0;3],0.9,0.0);
    let material=r.create_material(&MaterialDesc {name:"dry box surface".into(),features:MaterialFeatures {water:true,double_sided:true,alpha_mode:crate::AlphaMode::Blend,..Default::default()},..Default::default()})?;
    r.set_material_water_slot(material,0);
    let draws=[draw(floor,ground,Mat4::IDENTITY,-1),draw(plane,material,Mat4::IDENTITY,-1)];
    let eye=Vec3::new(0.0,12.0,0.0);
    let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::NEG_Z),proj:Mat4::orthographic_rh(-8.0,8.0,-5.0,5.0,0.05,50.0)};
    let region=zestengine_core::WaterExclusionVolume {half_extents:[2.5,1.0,1.2],..Default::default()}
        .region(Mat4::from_rotation_translation(glam::Quat::from_rotation_y(0.55),Vec3::new(1.0,0.0,-0.5))).unwrap();
    let mut shots=Vec::new();
    let outline=vec![[-7.0,-4.0],[7.0,-4.0],[7.0,4.0],[3.0,4.0],[3.0,-1.0],[-3.0,-1.0],[-3.0,4.0],[-7.0,4.0]];
    for mode in 0..4 {
        let mut authored=zestengine_core::Water {size:[16.0,16.0],depth:10.0,..Default::default()};
        if mode==2 {authored.exclusions.push(region);}
        if mode==3 {authored.shape=outline.clone();}
        let mut body = GpuWater { to_local: Mat4::IDENTITY.to_cols_array_2d(), bounds: [8.0,8.0,0.0,0.0], shallow: [0.1,0.45,0.48,4.0], deep: [0.01,0.07,0.12,1.0], look0: [0.08,0.0,1.0,0.0], look1: [0.0,1.0,0.0,0.0], ..Default::default() };
        body.set_optics([0.32,0.065,0.025,1.333],[0.015,0.035,0.045,0.65]);
        let mut coverage=Vec::new();crate::WaterCoverage::new(&authored).append(&authored,&mut body,&mut coverage);
        let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([WIDTH,HEIGHT]);input.draws=if mode==0 {&draws[..1]} else {&draws};
        input.water=if mode==0{&[]}else{std::slice::from_ref(&body)};input.water_coverage=&coverage;input.postfx.bloom_enabled=false;
        for _ in 0..4 {r.render(&input)?;}
        shots.push(r.debug_read_viewport()?.unwrap().2);
    }
    let inv=(camera.proj*camera.view).inverse();let (mut bad,mut wet_changed,mut dry_count)=(0usize,0usize,0usize);
    for y in 0..HEIGHT {for x in 0..WIDTH {
        let p=inv.project_point3(Vec3::new((x as f32+0.5)/WIDTH as f32*2.0-1.0,1.0-(y as f32+0.5)/HEIGHT as f32*2.0,0.0));
        let p=Vec3::new(p.x,0.0,p.z);let uv=region.to_unit.transform_point3(p);
        if (uv.x.abs()-1.0).abs()<0.02 || (uv.z.abs()-1.0).abs()<0.02 {continue;}
        let inside=region.contains(p);let expected=if inside {0} else {1};let i=((y*WIDTH+x)*4) as usize;
        if (0..3).any(|k|shots[2][i+k].abs_diff(shots[expected][i+k])>2) {bad+=1;}
        if inside {dry_count+=1;} else if (0..3).any(|k|shots[1][i+k].abs_diff(shots[0][i+k])>4) {wet_changed+=1;}
    }}
    let shaped=zestengine_core::Water {shape:outline,..Default::default()};
    let(mut shape_bad,mut shape_wet,mut shape_dry,mut shape_response)=(0usize,0usize,0usize,0usize);
    for y in 0..HEIGHT {for x in 0..WIDTH {
        let p=inv.project_point3(Vec3::new((x as f32+0.5)/WIDTH as f32*2.0-1.0,1.0-(y as f32+0.5)/HEIGHT as f32*2.0,0.0));
        let inside=shaped.column(glam::Vec2::new(p.x,p.z)).is_some();let expected=if inside {1} else {0};let i=((y*WIDTH+x)*4) as usize;
        // Wet pixels may change near a bank because their refracted ray
        // exits sooner. Dry pixels must exactly recover the opaque scene.
        if !inside && (0..3).any(|k|shots[3][i+k].abs_diff(shots[expected][i+k])>2) {shape_bad+=1;}
        if inside {shape_wet+=1;if (0..3).any(|k|shots[3][i+k].abs_diff(shots[0][i+k])>4) {shape_response+=1;}}else{shape_dry+=1;}
    }}
    eprintln!("[water shoreline surface] concave mask: {shape_wet} wet, {shape_dry} dry, {shape_bad} mismatches");
    anyhow::ensure!(shape_bad<20 && shape_wet>1000 && shape_dry>1000 && shape_response>shape_wet*9/10,"surface footprint disagrees with gameplay");
    eprintln!("[water exclusion surface] rotated partial box: {dry_count} dry pixels, {wet_changed} visible wet pixels, {bad} mismatches");
    anyhow::ensure!(bad<20 && dry_count>1000 && wet_changed>10000,"partial exclusion failed");
    let dir=std::env::var("ZESTENGINE_TEST_DUMP_DIR").unwrap_or_else(|_|"/tmp/zest-water-visual".into());std::fs::create_dir_all(&dir)?;
    let mut ppm=format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();for p in shots[2].as_chunks::<4>().0{ppm.extend_from_slice(&p[..3]);}
    std::fs::write(std::path::Path::new(&dir).join("partial_dry_box.ppm"),ppm)?;
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
                shallow_pool_caustics(w.clone())?;
                mapped_pool_caustics(w.clone())?;
                partial_exclusion_surface(w.clone())?;
                dry_underwater_views(w.clone())?;
                coverage_receivers(w.clone())?;
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

