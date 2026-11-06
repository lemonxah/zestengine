//! The GPU water simulation (`water_sim.comp` through `GpuWaterSim`) on a
//! pool set up like the user's scenes: a 20 m pool 5 m deep by its own
//! depth but standing on a floor 0.31 m under its surface, a static sphere
//! through the surface and a static box standing on the floor through it.
//! The engine feeds such a pool its field twice: first from the water's own
//! depth (no geometry known yet), then, a frame later, with the floor and
//! walls under it. The surface must stay at rest through that (the field
//! change used to keep the columns' old heights, lifting the whole surface
//! by the 4.69 m difference), static objects must leave it flat, and a body
//! dragged through must raise ripples that die back to flat.
//!
//! `ZESTENGINE_TEST_GPU=1 cargo test -p zestengine-render --lib gpu_tests::water -- --test-threads=1 --nocapture`

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use glam::{Vec2, Vec3};
use zestengine_core::{EwaveParams, PipeParams, SimField, SimWindow, Water, WaterBody, WaterDisplacer};

use super::{Img, gpu, read_image};
use crate::texture::GpuTexture;
use crate::water_sim::{GpuWaterSim, WaterSimBody, WaterSimEwave};

#[path = "water_optics.rs"]
mod optics;

const KEY: u64 = 7;
const DT: f32 = 1.0 / 60.0;
/// Floor under the pool (local Y), and the static sphere and box.
const FLOOR: f32 = -0.31;
const SPHERE_R: f32 = 0.5;
const SPHERE_Y: f32 = -0.2;
const BOX_C: Vec3 = Vec3::new(-3.0, 0.0, 2.0);
const BOX_H: Vec3 = Vec3::new(0.5, 0.6, 0.5);

/// The Water of the user's vrfriends scene (its saved fields).
fn user_pool() -> Water {
    Water {
        body: WaterBody::Pool,
        size: [20.0, 20.0],
        depth: 5.0,
        waves: Vec::new(),
        ripple_area: 32.0,
        ripple_resolution: 256,
        ripple_damping: 0.8,
        gpu_sim: true,
        interactive: true,
        sim_wave_depth: 1.0,
        sim_displacement: 0.7,
        sim_momentum: 0.43,
        sim_dispersion: 1.0,
        sim_ripple_mix: 0.45,
        sim_chop_smoothing: 1.0,
        sim_wake_ripples: 0.2,
        sim_ripple_smoothing: 0.06,
        sim_fine_damping: 6.0,
        ..Default::default()
    }
}

/// Window cell centre `i` (row-major), water-local XZ.
fn centre(w: SimWindow, i: usize) -> Vec2 {
    let n = w.n as usize;
    w.lo() + (Vec2::new((i % n) as f32, (i / n) as f32) + 0.5) * w.cell
}

/// What `water_walls` bakes for the scene: obstacle cells (the box, and the
/// sphere inside its waterline when `sphere` is a wall at that XZ) and the
/// depth of the ground under every other cell (the floor, the sphere's top).
fn walls(w: SimWindow, sphere: Option<Vec2>) -> (Vec<bool>, Vec<f32>) {
    let cells = (w.n * w.n) as usize;
    let mut blocked = vec![false; cells];
    let mut bed = vec![-FLOOR; cells];
    for i in 0..cells {
        let p = centre(w, i);
        let b = p - Vec2::new(BOX_C.x, BOX_C.z);
        if b.x.abs() <= BOX_H.x && b.y.abs() <= BOX_H.z {
            blocked[i] = true;
        }
        if let Some(s) = sphere {
            let d2 = (p - s).length_squared();
            if d2 < SPHERE_R * SPHERE_R {
                let top = SPHERE_Y + (SPHERE_R * SPHERE_R - d2).sqrt();
                if top >= 0.0 {
                    blocked[i] = true;
                } else {
                    bed[i] = bed[i].min(-top);
                }
            }
        }
    }
    (blocked, bed)
}

/// The sphere as a body pushing the water (`EditorWater::feed`).
fn sphere_displacer(water: &Water, at: Vec2, velocity: Vec2, cell: f32) -> WaterDisplacer {
    let c = Vec3::new(at.x, SPHERE_Y, at.y);
    let corners: [Vec3; 8] = std::array::from_fn(|k| {
        c + Vec3::new(
            if k & 1 == 0 { -SPHERE_R } else { SPHERE_R },
            if k & 2 == 0 { -SPHERE_R } else { SPHERE_R },
            if k & 4 == 0 { -SPHERE_R } else { SPHERE_R },
        )
    });
    WaterDisplacer::from_corners(&corners, Vec3::X, 0.0, water.sim_wave_depth.min(water.depth), velocity, true).unwrap().fit_to_cell(cell)
}

/// One frame of input, as `WaterRenderer::gpu_ripples` builds it.
fn frame(water: &Water, field: &SimField, displacers: &[WaterDisplacer], reset: bool) -> WaterSimBody {
    let window = field.window;
    let params = PipeParams::for_water(water);
    let ewave = EwaveParams::for_water(water, window, field.max_depth).map(|e| WaterSimEwave {
        gain: e.gain,
        depth: e.depth,
        damping: e.damping,
        split: e.split,
        current: e.current.to_array(),
        body_gain: e.body_gain,
        smoothing: e.smoothing,
        grid_damping: e.grid_damping,
        mix: e.mix,
    });
    assert!(ewave.is_some(), "the dispersive layer runs, as in the user's scene");
    let depth = field.max_depth.min(params.wave_depth) + 0.05;
    let dt = DT.min(zestengine_core::max_sim_dt(window.cell, depth, 1.0));
    let substeps = zestengine_core::pipe_substeps(dt, window.cell, depth, 1.0);
    WaterSimBody {
        key: KEY,
        texture: crate::TextureHandle(0),
        n: window.n,
        cell: window.cell,
        origin: window.origin.to_array(),
        closed: window.closed,
        sponge: window.sponge(),
        reset,
        idle: false,
        dt,
        substeps,
        gravity: params.gravity,
        keep: params.keep(dt / substeps as f32),
        momentum: params.momentum,
        level_return: params.level_return,
        wave_depth: params.wave_depth,
        displace: params.displacement,
        viscosity: params.viscosity,
        eddy: params.eddy,
        ewave,
        field_version: field.version,
        field_origin: field.window.origin.to_array(),
        field: field.cells.clone(),
        flowing: field.flowing,
        default_depth: field.default_depth,
        displacers: displacers.iter().map(|d| d.to_gpu()).collect(),
    }
}

/// Largest |height offset| over the open cells of a readback / texture.
fn peak(field: &SimField, heights: impl Iterator<Item = f32>) -> f32 {
    heights.zip(field.cells.iter()).filter(|(_, c)| c[3] < 0.5).map(|(h, _)| h.abs()).fold(0.0, f32::max)
}

#[test]
fn static_objects_leave_the_pool_flat_and_moving_ones_ripple_and_settle() {
    let Some(g) = gpu() else { return };
    let water = user_pool();
    let window = zestengine_core::choose_window(&water, Vec2::ZERO, None);
    assert!(window.closed && window.n == 256, "{window:?}");
    let n = window.n;
    let levels = 32 - n.leading_zeros();
    let allocator = Arc::new(Mutex::new(g.allocator().unwrap()));
    let tex = GpuTexture::cleared_mips(
        &g.device,
        &mut allocator.lock().unwrap(),
        g.pool,
        g.queue,
        n,
        n,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        levels,
    )
    .unwrap();
    let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
    let base = unsafe {
        g.device
            .create_image_view(
                &vk::ImageViewCreateInfo::default().image(tex.image).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R16G16B16A16_SFLOAT).subresource_range(range),
                None,
            )
            .unwrap()
    };
    let mips: HashMap<usize, (vk::ImageView, u32)> = [(0usize, (base, levels))].into_iter().collect();
    let textures = vec![tex];
    let mut sim = GpuWaterSim::new(&g.device).unwrap();
    let mut slot = 0usize;
    let mut step = |sim: &mut GpuWaterSim, queued: &[WaterSimBody]| {
        crate::alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| {
            sim.record(&g.device, &allocator, cb, slot, queued, &textures, &mips).unwrap();
        })
        .unwrap();
        slot = (slot + 1) % crate::frame::FRAMES_IN_FLIGHT;
    };
    // The newest frame's readback (collected once its slot comes round).
    let latest = |sim: &mut GpuWaterSim, step: &mut dyn FnMut(&mut GpuWaterSim, &[WaterSimBody])| {
        for _ in 0..crate::frame::FRAMES_IN_FLIGHT {
            step(sim, &[]);
        }
        sim.readback(KEY).expect("a readback").cells.iter().map(|c| c[0]).collect::<Vec<f32>>()
    };
    let read_texture = |allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>| -> Vec<f32> {
        let img = Img { image: textures[0].image, view: textures[0].view, format: vk::Format::R16G16B16A16_SFLOAT, w: n, h: n, layers: 1, allocation: None };
        let readers = (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::SHADER_SAMPLED_READ);
        let bytes = read_image(&g, &mut allocator.lock().unwrap(), &img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, readers, 8).unwrap();
        bytes.as_chunks::<8>().0.iter().map(|p| half::f16::from_le_bytes([p[0], p[1]]).to_f32()).collect()
    };

    // Frame 0: the field from the water alone (5 m deep everywhere), as the
    // first frame has it before the walls are baked.
    let mut field = SimField::build(&water, window, None);
    field.version = 1;
    step(&mut sim, &[frame(&water, &field, &[], false)]);
    // From frame 1 on: the floor 0.31 m down, the box, the sphere.
    let sphere_at = Vec2::new(3.0, 0.0);
    let with_walls = |sphere: Option<Vec2>, version: u64| {
        let (blocked, bed) = walls(window, sphere);
        let mut f = SimField::build_with_bed(&water, window, Some(&blocked), Some(&bed));
        f.version = version;
        f
    };
    field = with_walls(Some(sphere_at), 2);
    let mut most = 0.0f32;
    for k in 0..300 {
        step(&mut sim, &[frame(&water, &field, &[], false)]);
        if k % 60 == 59 {
            let h = latest(&mut sim, &mut step);
            most = most.max(peak(&field, h.into_iter()));
        }
    }
    let rest = latest(&mut sim, &mut step);
    let rest_peak = peak(&field, rest.iter().copied());
    let drawn = read_texture(&allocator);
    let drawn_peak = peak(&field, drawn.iter().copied());
    eprintln!("[water] static: peak offset during {most:.5} m, after 5 s {rest_peak:.6} m, drawn {drawn_peak:.6} m");
    assert!(most < 2e-3, "the floor and walls arriving a frame late leave the surface at rest: {most} m");
    assert!(rest_peak < 1e-3, "still objects leave it flat: {rest_peak} m");
    assert!(drawn_peak < 1e-3, "the drawn surface sits at the water's own height (the gizmo's top): {drawn_peak} m");

    // Drag the sphere 2 m along X over 2 s: a body, no longer a wall.
    field = with_walls(None, 3);
    let mut moving = 0.0f32;
    for k in 0..120 {
        let at = sphere_at - Vec2::new(2.0 * k as f32 / 120.0, 0.0);
        let d = sphere_displacer(&water, at, Vec2::new(-1.0, 0.0), window.cell);
        step(&mut sim, &[frame(&water, &field, &[d], false)]);
        if k % 30 == 29 {
            let h = latest(&mut sim, &mut step);
            moving = moving.max(peak(&field, h.into_iter()));
        }
    }
    // Put down: a wall again where it stopped.
    let stop = sphere_at - Vec2::new(2.0, 0.0);
    field = with_walls(Some(stop), 4);
    let mut after = Vec::new();
    for k in 0..900 {
        step(&mut sim, &[frame(&water, &field, &[], false)]);
        if k % 150 == 149 {
            let h = latest(&mut sim, &mut step);
            after.push(peak(&field, h.into_iter()));
        }
    }
    let drawn_end = peak(&field, read_texture(&allocator).into_iter());
    eprintln!("[water] dragged: peak while moving {moving:.4} m; after it stopped, every 2.5 s: {after:?}; drawn at 15 s {drawn_end:.5} m");
    assert!(moving > 5e-3, "a body dragged through makes waves: {moving} m");
    assert!(after.last().copied().unwrap_or(1.0) < 2e-3, "they die back to flat: {after:?}");
    assert!(after.windows(2).skip(1).all(|w| w[1] <= w[0] * 1.5 + 1e-4), "and keep dying down: {after:?}");
    assert!(drawn_end < 2e-3, "the drawn surface too: {drawn_end}");

    unsafe { g.device.device_wait_idle().unwrap() };
    let mut alloc = allocator.lock().unwrap();
    sim.destroy(&g.device, &mut alloc);
    unsafe { g.device.destroy_image_view(base, None) };
    for mut t in textures {
        t.destroy(&g.device, &mut alloc);
    }
}

// ---------------------------------------------------------------------------
// The user's LongDark scene: a 20.25 x 20.83 m pool 1 m deep by its own depth
// at y 1.015 over a 20 x 20 m plane (top y 0; the body overhangs it 0.42 m at
// -x and 0.93 m at -z, where nothing is under it), a 0.5 m sphere whose crown
// is 6.8 cm under the surface. ripple_resolution 300 -> a 304-cell closed
// window (no dispersive layer: 304 is not a power of two). Stepped as the
// editor steps it: a 0.05 s first frame with the water's own field, the bed
// a frame later, then `dt` per frame; at 60 fps and at ~600 fps (the user
// runs at 791 fps).

fn longdark() -> Water {
    Water { size: [20.253569, 20.82951], depth: 1.0, ripple_area: 40.0, ripple_resolution: 300, ..user_pool() }
}

/// LongDark's bed over `w` (water-local): plane and sphere crown.
fn longdark_bed(w: SimWindow) -> (Vec<bool>, Vec<f32>) {
    let cells = (w.n * w.n) as usize;
    let off = Vec2::new(0.2923679, 0.5116773); // world origin in water-local XZ
    let level = 1.0150946f32;
    let sphere = Vec2::new(3.0768197, 0.17970873) + off;
    let sphere_y = 0.44685358 - level;
    let mut blocked = vec![false; cells];
    let mut bed = vec![f32::NAN; cells];
    for i in 0..cells {
        let p = centre(w, i) - off;
        if p.x.abs() <= 10.0 && p.y.abs() <= 10.0 {
            bed[i] = level;
        }
        let d2 = (centre(w, i) - sphere).length_squared();
        if d2 < 0.25 {
            let top = sphere_y + (0.25 - d2).sqrt();
            if top >= 0.0 {
                blocked[i] = true;
            } else {
                bed[i] = if bed[i].is_nan() { -top } else { bed[i].min(-top) };
            }
        }
    }
    (blocked, bed)
}

/// `frame` with an explicit frame time (the dispersive layer may be off).
fn frame_dt(water: &Water, field: &SimField, dt: f32) -> WaterSimBody {
    let window = field.window;
    let params = PipeParams::for_water(water);
    let ewave = EwaveParams::for_water(water, window, field.max_depth).map(|e| WaterSimEwave {
        gain: e.gain,
        depth: e.depth,
        damping: e.damping,
        split: e.split,
        current: e.current.to_array(),
        body_gain: e.body_gain,
        smoothing: e.smoothing,
        grid_damping: e.grid_damping,
        mix: e.mix,
    });
    let depth = field.max_depth.min(params.wave_depth) + 0.05;
    let dt = dt.min(zestengine_core::max_sim_dt(window.cell, depth, 1.0));
    let substeps = zestengine_core::pipe_substeps(dt, window.cell, depth, 1.0);
    WaterSimBody {
        key: KEY,
        texture: crate::TextureHandle(0),
        n: window.n,
        cell: window.cell,
        origin: window.origin.to_array(),
        closed: window.closed,
        sponge: window.sponge(),
        reset: false,
        idle: false,
        dt,
        substeps,
        gravity: params.gravity,
        keep: params.keep(dt / substeps as f32),
        momentum: params.momentum,
        level_return: params.level_return,
        wave_depth: params.wave_depth,
        displace: params.displacement,
        viscosity: params.viscosity,
        eddy: params.eddy,
        ewave,
        field_version: field.version,
        field_origin: field.window.origin.to_array(),
        field: field.cells.clone(),
        flowing: field.flowing,
        default_depth: field.default_depth,
        displacers: Vec::new(),
    }
}

/// Offsets split by the cell's place: open cells next to a shore cell (the
/// body's outline ring), open cells next to an obstacle, the rest.
fn regions(field: &SimField, heights: &[f32]) -> String {
    let n = field.window.n as usize;
    let kind = |x: i32, y: i32| -> f32 {
        if x < 0 || y < 0 || x >= n as i32 || y >= n as i32 {
            return 1.0;
        }
        field.cells[y as usize * n + x as usize][3]
    };
    let mut ring = (0usize, f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    let mut near = (0usize, f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    let mut rest = (0usize, f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for (i, cell) in field.cells.iter().enumerate() {
        if cell[3] > 0.5 {
            continue;
        }
        let (x, y) = ((i % n) as i32, (i / n) as i32);
        let ks = [kind(x - 1, y), kind(x + 1, y), kind(x, y - 1), kind(x, y + 1)];
        let r = if ks.iter().any(|k| *k > 1.5) {
            &mut near
        } else if ks.iter().any(|k| (*k - 1.0).abs() < 0.5) {
            &mut ring
        } else {
            &mut rest
        };
        r.0 += 1;
        r.1 = r.1.min(heights[i]);
        r.2 = r.2.max(heights[i]);
        r.3 += heights[i] as f64;
    }
    let f = |name: &str, r: &(usize, f32, f32, f64)| if r.0 == 0 { String::new() } else { format!(" {name}[n={} min {:+.4} max {:+.4} mean {:+.4}]", r.0, r.1, r.2, r.3 / r.0 as f64) };
    format!("{}{}{}", f("interior", &rest), f("outline-ring", &ring), f("near-obstacle", &near))
}

struct Rig {
    allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    textures: Vec<GpuTexture>,
    base: vk::ImageView,
    mips: HashMap<usize, (vk::ImageView, u32)>,
    sim: GpuWaterSim,
    slot: usize,
    n: u32,
}

impl Rig {
    fn new(g: &crate::cluster_headless::HeadlessGpu, n: u32) -> Self {
        let levels = 32 - n.leading_zeros();
        let allocator = Arc::new(Mutex::new(g.allocator().unwrap()));
        let tex = GpuTexture::cleared_mips(
            &g.device,
            &mut allocator.lock().unwrap(),
            g.pool,
            g.queue,
            n,
            n,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            levels,
        )
        .unwrap();
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let base = unsafe {
            g.device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default().image(tex.image).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R16G16B16A16_SFLOAT).subresource_range(range),
                    None,
                )
                .unwrap()
        };
        let mips: HashMap<usize, (vk::ImageView, u32)> = [(0usize, (base, levels))].into_iter().collect();
        Self { allocator, textures: vec![tex], base, mips, sim: GpuWaterSim::new(&g.device).unwrap(), slot: 0, n }
    }

    fn step(&mut self, g: &crate::cluster_headless::HeadlessGpu, queued: &[WaterSimBody]) {
        crate::alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| {
            self.sim.record(&g.device, &self.allocator, cb, self.slot, queued, &self.textures, &self.mips).unwrap();
        })
        .unwrap();
        self.slot = (self.slot + 1) % crate::frame::FRAMES_IN_FLIGHT;
    }

    /// The newest frame's readback heights (collected once its slot comes round).
    fn latest(&mut self, g: &crate::cluster_headless::HeadlessGpu) -> Vec<f32> {
        for _ in 0..crate::frame::FRAMES_IN_FLIGHT {
            self.step(g, &[]);
        }
        self.sim.readback(KEY).expect("a readback").cells.iter().map(|c| c[0]).collect()
    }

    fn read_texture(&self, g: &crate::cluster_headless::HeadlessGpu) -> Vec<f32> {
        let n = self.n;
        let img = Img { image: self.textures[0].image, view: self.textures[0].view, format: vk::Format::R16G16B16A16_SFLOAT, w: n, h: n, layers: 1, allocation: None };
        let readers = (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::SHADER_SAMPLED_READ);
        let bytes = read_image(g, &mut self.allocator.lock().unwrap(), &img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, readers, 8).unwrap();
        bytes.as_chunks::<8>().0.iter().map(|p| half::f16::from_le_bytes([p[0], p[1]]).to_f32()).collect()
    }

    fn destroy(mut self, g: &crate::cluster_headless::HeadlessGpu) {
        unsafe { g.device.device_wait_idle().unwrap() };
        let mut alloc = self.allocator.lock().unwrap();
        self.sim.destroy(&g.device, &mut alloc);
        unsafe { g.device.destroy_image_view(self.base, None) };
        for mut t in self.textures.drain(..) {
            t.destroy(&g.device, &mut alloc);
        }
    }
}

/// A 1 m box held half under in the user's pool, fed as a body every frame
/// (a rigidbody in the editor): the drawn texture shows no rise in the
/// cells it occupies, and the rim round it settles toward the pool's own
/// rise, no slab (water_pipe.rs `a_held_or_dragged_box_draws_no_raised_slab`).
#[test]
fn a_held_box_draws_no_slab_round_itself() {
    let Some(g) = gpu() else { return };
    let water = user_pool();
    let window = zestengine_core::choose_window(&water, Vec2::ZERO, None);
    let mut rig = Rig::new(&g, window.n);
    let mut field = SimField::build(&water, window, None);
    field.version = 1;
    let corners: [Vec3; 8] = std::array::from_fn(|k| Vec3::new(if k & 1 == 0 { -0.5 } else { 0.5 }, if k & 2 == 0 { -0.5 } else { 0.5 }, if k & 4 == 0 { -0.5 } else { 0.5 }));
    let axis = Vec3::new(0.96, 0.0, 0.28);
    let d = WaterDisplacer::from_corners(&corners, axis, 0.0, water.sim_wave_depth.min(water.depth), Vec2::ZERO, false).unwrap().fit_to_cell(window.cell);
    let prof = |i: usize| d.splat_profile(centre(window, i), window.cell);
    let split = |heights: &[f32]| {
        let (mut inside, mut rim, mut far, mut nfar) = (0.0f32, 0.0f32, 0.0f32, 0usize);
        for (i, &h) in heights.iter().enumerate() {
            let p = prof(i);
            if p >= 0.5 {
                inside = inside.max(h);
            } else if p > 1e-4 {
                rim = rim.max(h);
            } else if field.cells[i][3] < 0.5 {
                far += h;
                nfar += 1;
            }
        }
        (inside, rim, far / nfar.max(1) as f32)
    };
    // The pool first (RESET would otherwise start the columns level with
    // the body in them), the box from the next frame on.
    rig.step(&g, &[frame_dt(&water, &field, 1.0 / 60.0)]);
    let mut inside_peak = 0.0f32;
    let mut rim_at_1s = 0.0f32;
    for k in 0..180 {
        let mut q = frame_dt(&water, &field, 1.0 / 60.0);
        q.displacers = vec![d.to_gpu()];
        rig.step(&g, &[q]);
        if k % 20 == 19 {
            let (inside, rim, far) = split(&rig.read_texture(&g));
            inside_peak = inside_peak.max(inside);
            if k == 59 {
                rim_at_1s = rim;
            }
            eprintln!("[held box] t={:.2}s drawn: inside max {inside:+.4} rim max {rim:+.4} pool mean {far:+.4}", (k + 1) as f32 / 60.0);
        }
    }
    let (inside, rim, far) = split(&rig.read_texture(&g));
    let back = rig.latest(&g);
    let (bi, br, bf) = split(&back);
    let back_peak = back.iter().map(|h| h.abs()).fold(0.0f32, f32::max);
    eprintln!("[held box] readback: inside max {bi:+.4} rim max {br:+.4} pool mean {bf:+.5} |peak| {back_peak:.4}; displacer half {:?} thickness {:.3} cell {:.3}", d.half, d.thickness, window.cell);
    rig.destroy(&g);
    assert!(bf > 1e-4, "the body displaces water (the pool rises): {bf}");
    assert!(inside_peak < 1e-3, "the occupied cells never draw a rise: {inside_peak}");
    assert!(rim - far < 5e-3 && rim <= rim_at_1s + 1e-4, "the rim settles toward the pool's rise, no slab: rim {rim} vs pool {far} (at 1 s {rim_at_1s})");
    let _ = inside;
}

/// Run LongDark from load for `seconds` at `dt` per frame, logging the
/// regions at the sample times; returns the worst |offset| over open cells
/// after 0.1 s.
fn run_longdark(g: &crate::cluster_headless::HeadlessGpu, dt: f32, seconds: f32, first_frame_dt: f32, bed_first: bool) -> f32 {
    let water = longdark();
    let window = zestengine_core::choose_window(&water, Vec2::ZERO, None);
    assert!(window.closed && window.n == 304, "{window:?}");
    let mut rig = Rig::new(g, window.n);
    let (blocked, bed) = longdark_bed(window);
    let with_bed = |version: u64| {
        let mut f = SimField::build_with_bed(&water, window, Some(&blocked), Some(&bed));
        f.version = version;
        f
    };
    let mut field;
    if bed_first {
        field = with_bed(1);
        rig.step(g, &[frame_dt(&water, &field, first_frame_dt)]);
    } else {
        // Frame 1: the field from the water alone (no bed yet), the editor's
        // clamped first frame; frame 2: the bed arrives (REBASE).
        field = SimField::build(&water, window, None);
        field.version = 1;
        rig.step(g, &[frame_dt(&water, &field, first_frame_dt)]);
        field = with_bed(2);
    }
    let samples = [0.15f32, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut t = first_frame_dt;
    let mut next = 0;
    let mut worst = 0.0f32;
    let mut frames = 0u32;
    while t < seconds && next < samples.len() {
        let q = frame_dt(&water, &field, dt);
        let substeps = q.substeps;
        t += q.dt;
        frames += 1;
        rig.step(g, &[q]);
        if t >= samples[next] {
            let h = rig.latest(g);
            let p = peak(&field, h.iter().copied());
            if t > 0.1 {
                worst = worst.max(p);
            }
            eprintln!("[water longdark dt={dt:.4} (substeps {})] t={t:.2}s frame {frames}: peak {p:.4} m;{}", substeps, regions(&field, &h));
            next += 1;
        }
    }
    let drawn = rig.read_texture(g);
    eprintln!("[water longdark dt={dt:.4}] drawn texture peak {:.4} m", peak(&field, drawn.into_iter()));
    rig.destroy(g);
    worst
}

#[test]
fn longdark_pool_stays_level_at_60_and_600_fps() {
    let Some(g) = gpu() else { return };
    let at_60 = run_longdark(&g, 1.0 / 60.0, 10.0, 0.05, false);
    let at_600 = run_longdark(&g, 1.0 / 600.0, 5.0, 0.05, false);
    eprintln!("[water longdark] worst open-cell offset after 0.1 s: 60 fps {at_60:.4} m, 600 fps {at_600:.4} m");
    assert!(at_60 < 2e-3, "at 60 fps the pool stays at the authored level: {at_60} m");
    assert!(at_600 < 2e-3, "at 600 fps too: {at_600} m");
}

#[test]
fn longdark_pool_stays_level_when_the_bed_is_known_from_the_start() {
    let Some(g) = gpu() else { return };
    let w = run_longdark(&g, 1.0 / 600.0, 3.0, 1.0 / 600.0, true);
    eprintln!("[water longdark] bed known from frame 1, 600 fps: worst {w:.4} m");
    assert!(w < 2e-3, "{w} m");
}

/// Which ingredient of the LongDark setup makes the GPU drain the pool
/// through its outline ring (the CPU reference does not): the window size
/// (304 cells, not a power of two, so the dispersive layer is off), the
/// dispersive layer itself, the depth.
#[test]
fn longdark_variants_drain_or_not() {
    let Some(g) = gpu() else { return };
    let variants: Vec<(&str, Water)> = vec![
        ("as-is n=304 depth 1", longdark()),
        ("n=256 depth 1", Water { ripple_resolution: 256, ..longdark() }),
        ("n=256 depth 1 dispersion 0 (ewave off)", Water { ripple_resolution: 256, sim_dispersion: 0.0, ..longdark() }),
        ("n=304 depth 5", Water { depth: 5.0, ..longdark() }),
        ("n=288 depth 1", Water { ripple_resolution: 288, ..longdark() }),
        ("n=320 depth 1", Water { ripple_resolution: 320, ..longdark() }),
        ("n=512 depth 1", Water { ripple_resolution: 512, ..longdark() }),
        ("n=304 depth 1 viscosity 0 chop 0", Water { ripple_resolution: 300, viscosity: 0.0, sim_chop_smoothing: 0.0, ..longdark() }),
    ];
    for (name, water) in variants {
        let window = zestengine_core::choose_window(&water, Vec2::ZERO, None);
        let mut rig = Rig::new(&g, window.n);
        let (blocked, bed) = longdark_bed(window);
        let mut field = SimField::build_with_bed(&water, window, Some(&blocked), Some(&bed));
        field.version = 1;
        let mut t = 0.0f32;
        let mut line = format!("[water variant] {name}: window n={} cell={:.4} closed={} ewave={}:", window.n, window.cell, window.closed, EwaveParams::for_water(&water, window, field.max_depth).is_some());
        for _ in 0..60 {
            let q = frame_dt(&water, &field, 1.0 / 60.0);
            t += q.dt;
            rig.step(&g, &[q]);
        }
        let h = rig.latest(&g);
        line += &format!(" t={t:.2}s peak {:.4} m;{}", peak(&field, h.iter().copied()), regions(&field, &h));
        eprintln!("{line}");
        rig.destroy(&g);
    }
}

/// A field whose rest depth is unique per cell, all cells water: after a
/// dt-0 frame (RESET writes h = rest, OUTPUT reads it back) every cell
/// whose readback is off was overwritten through its toroidal storage slot
/// by the cell whose rest depth it now holds. Prints the collision map.
#[test]
fn storage_slots_are_unique() {
    let Some(g) = gpu() else { return };
    for res in [256u32, 304, 288] {
        let water = Water { ripple_resolution: res, ..longdark() };
        let window = zestengine_core::choose_window(&water, Vec2::ZERO, None);
        let n = window.n as usize;
        let mut field = SimField::build(&water, window, None);
        let cells: Vec<[f32; 4]> = (0..n * n).map(|i| [0.1 + i as f32 * 1e-5, 0.0, 0.0, 0.0]).collect();
        field.cells = Arc::new(cells);
        field.version = 1;
        let mut rig = Rig::new(&g, window.n);
        rig.step(&g, &[frame_dt(&water, &field, 0.0)]);
        let h = rig.latest(&g);
        let mut pairs = Vec::new();
        for (i, hv) in h.iter().enumerate() {
            if hv.abs() > 5e-6 {
                let other = ((hv + field.cells[i][0] - 0.1) / 1e-5).round() as i64;
                pairs.push((i, other));
            }
        }
        let o = window.origin;
        let show = |i: i64| -> String {
            let (x, y) = (i % n as i64, i / n as i64);
            format!("cell({x},{y}) g({},{})", x + o.x as i64, y + o.y as i64)
        };
        eprintln!("[water slots] n={n} origin={:?}: {} cells hold another cell's value after RESET", o, pairs.len());
        for (i, other) in pairs.iter().take(12) {
            eprintln!("[water slots]   {} holds the value of {}", show(*i as i64), show(*other));
        }
        // Distinct x and y offsets between the victim and the writer.
        let mut dx: Vec<i64> = pairs.iter().map(|(i, o)| (o % n as i64) - (*i as i64 % n as i64)).collect();
        let mut dy: Vec<i64> = pairs.iter().map(|(i, o)| (o / n as i64) - (*i as i64 / n as i64)).collect();
        dx.sort();
        dx.dedup();
        dy.sort();
        dy.dedup();
        eprintln!("[water slots]   writer - victim offsets: dx {dx:?} dy {dy:?}");
        rig.destroy(&g);
        assert!(pairs.is_empty(), "n={n}: {} cells share a storage slot with another cell (water_sim.comp wrapi: `%` on a negative operand is unsigned on this GPU, right only for a power-of-two n)", pairs.len());
    }
}

// ---------------------------------------------------------------------------
// Hypothesis "rebase": the rest-depth / REBASE logic (fields re-uploaded
// repeatedly, a rebase with the wrong sign or only on some frames, no-floor
// cells resting at a different depth from floored cells) makes the water
// flow toward the ground plane's edge. The tests below isolate each
// ingredient from the storage wrap (`storage_slots_are_unique`): the same
// LongDark pool at a positive window origin, a pool whose field is replaced
// and rebased every frame over a bed both above and below the level, a
// second rebase raising and lowering the floor, and an open window shifted
// mid-run with the engine's no-bed-then-bed sequence.

/// LongDark's pool as a rectangle polygon centred on `shift` (so that
/// `choose_window` puts the window's origin where `shift` says).
fn longdark_at(shift: Vec2) -> Water {
    let (hx, hz) = (20.253569 / 2.0, 20.82951 / 2.0);
    let shape = vec![[shift.x - hx, shift.y - hz], [shift.x + hx, shift.y - hz], [shift.x + hx, shift.y + hz], [shift.x - hx, shift.y + hz]];
    Water { shape, smooth_shape: false, ..longdark() }
}

/// `longdark_bed` for the pool centred on `shift`.
fn longdark_bed_at(w: SimWindow, shift: Vec2) -> (Vec<bool>, Vec<f32>) {
    let cells = (w.n * w.n) as usize;
    let off = Vec2::new(0.2923679, 0.5116773);
    let level = 1.0150946f32;
    let sphere = Vec2::new(3.0768197, 0.17970873) + off + shift;
    let sphere_y = 0.44685358 - level;
    let mut blocked = vec![false; cells];
    let mut bed = vec![f32::NAN; cells];
    for i in 0..cells {
        let p = centre(w, i) - shift - off;
        if p.x.abs() <= 10.0 && p.y.abs() <= 10.0 {
            bed[i] = level;
        }
        let d2 = (centre(w, i) - sphere).length_squared();
        if d2 < 0.25 {
            let top = sphere_y + (0.25 - d2).sqrt();
            if top >= 0.0 {
                blocked[i] = true;
            } else {
                bed[i] = if bed[i].is_nan() { -top } else { bed[i].min(-top) };
            }
        }
    }
    (blocked, bed)
}

/// Largest |offset| over the wet open cells (rest depth > 0; dry beach
/// cells are drawn sunk by design) and the (min, max) over the beach cells.
fn wet_peak(field: &SimField, heights: &[f32]) -> (f32, (f32, f32)) {
    let mut wet = 0.0f32;
    let mut beach = (f32::INFINITY, f32::NEG_INFINITY);
    for (h, c) in heights.iter().zip(field.cells.iter()) {
        if c[3] > 0.5 {
            continue;
        }
        if c[0] > 0.0 {
            wet = wet.max(h.abs());
        } else {
            beach = (beach.0.min(*h), beach.1.max(*h));
        }
    }
    (wet, beach)
}

/// LongDark from load (own field for a 0.05 s frame, the bed a frame
/// later, then 1/600 s frames for 1.5 s) with the pool centred on `shift`;
/// returns the window and the worst wet-cell |offset| after 0.1 s.
fn rebase_hyp_longdark(g: &crate::cluster_headless::HeadlessGpu, shift: Vec2) -> (SimWindow, f32) {
    let water = longdark_at(shift);
    let window = zestengine_core::choose_window(&water, shift, None);
    assert!(window.closed && window.n == 304, "{window:?}");
    let (blocked, bed) = longdark_bed_at(window, shift);
    let mut rig = Rig::new(g, window.n);
    let mut field = SimField::build(&water, window, None);
    field.version = 1;
    rig.step(g, &[frame_dt(&water, &field, 0.05)]);
    field = SimField::build_with_bed(&water, window, Some(&blocked), Some(&bed));
    field.version = 2;
    let open = field.cells.iter().filter(|c| c[3] < 0.5).count();
    let mut t = 0.05f32;
    let mut worst = 0.0f32;
    for k in 0..900 {
        let q = frame_dt(&water, &field, 1.0 / 600.0);
        t += q.dt;
        rig.step(g, &[q]);
        if k % 300 == 299 {
            let h = rig.latest(g);
            let (p, _) = wet_peak(&field, &h);
            worst = worst.max(p);
            eprintln!("[rebase_hyp longdark origin {:?} open cells {open}] t={t:.2}s: wet peak {p:.4} m;{}", window.origin, regions(&field, &h));
        }
    }
    let drawn = rig.read_texture(g);
    eprintln!("[rebase_hyp longdark origin {:?}] drawn texture wet peak {:.4} m", window.origin, wet_peak(&field, &drawn).0);
    rig.destroy(g);
    (window, worst)
}

#[test]
fn rebase_hyp_longdark_n304_positive_origin_stays_level() {
    let Some(g) = gpu() else { return };
    let (w, worst) = rebase_hyp_longdark(&g, Vec2::splat(13.0));
    assert!(w.origin.x >= 0 && w.origin.y >= 0, "{w:?}");
    eprintln!("[rebase_hyp] n=304 at origin {:?}: worst wet offset {worst:.4} m", w.origin);
    assert!(worst < 2e-3, "the same frames (own field, bed a frame later = REBASE) keep the pool level at a positive origin: {worst} m");
}

#[test]
fn rebase_hyp_longdark_n304_negative_origin_stays_level() {
    let Some(g) = gpu() else { return };
    let (w, worst) = rebase_hyp_longdark(&g, Vec2::ZERO);
    assert!(w.origin.x < 0 && w.origin.y < 0, "{w:?}");
    eprintln!("[rebase_hyp] n=304 at origin {:?}: worst wet offset {worst:.4} m", w.origin);
    assert!(worst < 2e-3, "the user's window (negative origin): {worst} m");
}

/// A 20 m pool 1 m deep whose bed has everything: a plane 0.9 m under the
/// level ending at x = 3 (nothing under it beyond: rest depth = the water's
/// own 1 m), a shelf 0.4 m under for z > 6, a beach rim 0.3 m ABOVE the
/// level around (-5, -5), and a box obstacle at (5, 5). `raised` moves the
/// floor both ways: up 0.4 m in R1, a new floor 0.6 m under in R2 (was
/// none), the shelf dropped to 0.8 m in R3.
fn mixed_pool() -> Water {
    Water { size: [20.0, 20.0], depth: 1.0, ripple_area: 40.0, ripple_resolution: 256, ..user_pool() }
}

fn mixed_bed(w: SimWindow, raised: bool) -> (Vec<bool>, Vec<f32>) {
    let cells = (w.n * w.n) as usize;
    let mut blocked = vec![false; cells];
    let mut bed = vec![f32::NAN; cells];
    for i in 0..cells {
        let p = centre(w, i);
        let mut d = if p.x <= 3.0 { 0.9 } else { f32::NAN };
        if p.y > 6.0 {
            d = 0.4;
        }
        let r = (p - Vec2::new(-5.0, -5.0)).length();
        if (4.0..5.0).contains(&r) {
            d = -0.3;
        }
        if raised {
            if (-9.0..=-3.0).contains(&p.x) && (-3.0..=3.0).contains(&p.y) {
                d = 0.5;
            }
            if (4.0..=9.0).contains(&p.x) && (-9.0..=-4.0).contains(&p.y) {
                d = 0.6;
            }
            if (-9.0..=-6.0).contains(&p.x) && (6.5..=9.5).contains(&p.y) {
                d = 0.8;
            }
        }
        bed[i] = d;
        if (p.x - 5.0).abs() <= 0.5 && (p.y - 5.0).abs() <= 0.5 {
            blocked[i] = true;
        }
    }
    (blocked, bed)
}

#[test]
fn rebase_hyp_mixed_bed_rebased_every_frame_and_both_ways_stays_level() {
    let Some(g) = gpu() else { return };
    let water = mixed_pool();
    let window = zestengine_core::choose_window(&water, Vec2::ZERO, None);
    assert!(window.closed && window.n == 256, "{window:?}");
    let mut rig = Rig::new(&g, window.n);
    let mut field = SimField::build(&water, window, None);
    field.version = 1;
    rig.step(&g, &[frame_dt(&water, &field, 0.05)]);
    let (blocked, bed) = mixed_bed(window, false);
    field = SimField::build_with_bed(&water, window, Some(&blocked), Some(&bed));
    let beach = field.cells.iter().filter(|c| c[3] < 0.5 && c[0] <= 0.0).count();
    let nofloor = field.cells.iter().filter(|c| c[3] < 0.5 && (c[0] - 1.0).abs() < 1e-6).count();
    eprintln!("[rebase_hyp mixed] {beach} beach cells, {nofloor} no-floor cells, {} obstacle cells", blocked.iter().filter(|b| **b).count());
    let mut t = 0.05f32;
    let mut worst = 0.0f32;
    // The bed, then the SAME field re-uploaded (new version -> upload +
    // REBASE) on every one of 300 frames.
    for k in 0..300 {
        field.version = 2 + k as u64;
        let q = frame_dt(&water, &field, 1.0 / 600.0);
        t += q.dt;
        rig.step(&g, &[q]);
        if k % 100 == 99 {
            let h = rig.latest(&g);
            let (p, b) = wet_peak(&field, &h);
            worst = worst.max(p);
            eprintln!("[rebase_hyp mixed, field v{} rebased every frame] t={t:.2}s: wet peak {p:.4} m, beach {b:.3?};{}", field.version, regions(&field, &h));
        }
    }
    // The floor moves both ways under the running pool.
    let (blocked, bed) = mixed_bed(window, true);
    field = SimField::build_with_bed(&water, window, Some(&blocked), Some(&bed));
    field.version = 400;
    for k in 0..600 {
        let q = frame_dt(&water, &field, 1.0 / 600.0);
        t += q.dt;
        rig.step(&g, &[q]);
        if k % 200 == 199 {
            let h = rig.latest(&g);
            let (p, b) = wet_peak(&field, &h);
            worst = worst.max(p);
            eprintln!("[rebase_hyp mixed, floor raised/lowered] t={t:.2}s: wet peak {p:.4} m, beach {b:.3?};{}", regions(&field, &h));
        }
    }
    let drawn = rig.read_texture(&g);
    let (dp, db) = wet_peak(&field, &drawn);
    eprintln!("[rebase_hyp mixed] drawn texture: wet peak {dp:.4} m, beach {db:.3?}");
    rig.destroy(&g);
    assert!(worst < 2e-3, "re-uploading and rebasing every frame over a bed above and below the level keeps the wet cells at the level: {worst} m");
    assert!(dp < 2e-3, "drawn too: {dp} m");
}

#[test]
fn rebase_hyp_open_window_moved_mid_run_stays_level() {
    let Some(g) = gpu() else { return };
    let water = Water { ripple_area: 10.0, ..mixed_pool() };
    let w1 = zestengine_core::choose_window(&water, Vec2::ZERO, None);
    assert!(!w1.closed && w1.n == 256, "{w1:?}");
    let mut rig = Rig::new(&g, w1.n);
    let mut field = SimField::build(&water, w1, None);
    field.version = 1;
    rig.step(&g, &[frame_dt(&water, &field, 0.05)]);
    let (blocked, bed) = mixed_bed(w1, false);
    field = SimField::build_with_bed(&water, w1, Some(&blocked), Some(&bed));
    field.version = 2;
    let mut t = 0.05f32;
    let mut worst = 0.0f32;
    for k in 0..300 {
        let q = frame_dt(&water, &field, 1.0 / 600.0);
        t += q.dt;
        rig.step(&g, &[q]);
        if k % 150 == 149 {
            let h = rig.latest(&g);
            let (p, b) = wet_peak(&field, &h);
            worst = worst.max(p);
            eprintln!("[rebase_hyp open window {:?}] t={t:.2}s: wet peak {p:.4} m, beach {b:.3?}", w1.origin);
        }
    }
    // The camera moves 3 m: a new window. The engine's first frame after the
    // move has no bed for it (blocked/bed keyed to the old window), the next
    // frame does.
    let w2 = zestengine_core::choose_window(&water, Vec2::new(3.0, 0.0), Some(w1));
    assert!(w2.origin != w1.origin && w2.n == w1.n, "{w1:?} -> {w2:?}");
    field = SimField::build(&water, w2, None);
    field.version = 3;
    let q = frame_dt(&water, &field, 1.0 / 600.0);
    t += q.dt;
    rig.step(&g, &[q]);
    let (blocked, bed) = mixed_bed(w2, false);
    field = SimField::build_with_bed(&water, w2, Some(&blocked), Some(&bed));
    field.version = 4;
    for k in 0..600 {
        let q = frame_dt(&water, &field, 1.0 / 600.0);
        t += q.dt;
        rig.step(&g, &[q]);
        if k % 200 == 199 {
            let h = rig.latest(&g);
            let (p, b) = wet_peak(&field, &h);
            worst = worst.max(p);
            eprintln!("[rebase_hyp open window moved to {:?}] t={t:.2}s: wet peak {p:.4} m, beach {b:.3?}", w2.origin);
        }
    }
    let drawn = rig.read_texture(&g);
    let (dp, _) = wet_peak(&field, &drawn);
    eprintln!("[rebase_hyp open window] drawn texture wet peak {dp:.4} m");
    rig.destroy(&g);
    assert!(worst < 2e-3, "a window moved mid-run (no bed for a frame, then the bed) keeps the wet cells level: {worst} m");
    assert!(dp < 2e-3, "drawn too: {dp} m");
}
