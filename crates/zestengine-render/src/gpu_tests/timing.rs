//! GPU timestamps on a real device: the tick -> ms conversion with the
//! device's own `timestampPeriod` / `timestampValidBits`, a two-timestamp
//! measurement of a known workload against wall time and scaling, and the
//! SKIN / MORPH zones the skinner stamps.

use ash::vk;

use super::gpu;
use crate::cluster_headless::HeadlessGpu;
use crate::{TsResult, gpu_zone, ts_valid_mask, ts_zone_ms};

/// Two timestamps around `bytes` of vkCmdFillBuffer (device-local memory
/// bandwidth); returns (gpu_ms from ts_zone_ms, wall_ms of the submit).
fn fill_ms(g: &HeadlessGpu, bytes: u64) -> Option<(f32, f64)> {
    let mut alloc = g.allocator().unwrap();
    // A shared GPU can be out of memory for a 2 GB buffer (other tests, a running editor): not a failure.
    let mut buf = crate::alloc::Buffer::new(&g.device, &mut alloc, bytes, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuOnly, "fill").ok()?;
    let pool = unsafe {
        g.device
            .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)
            .unwrap()
    };
    let started = std::time::Instant::now();
    crate::alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        g.device.cmd_reset_query_pool(cb, pool, 0, 2);
        g.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
        g.device.cmd_fill_buffer(cb, buf.handle, 0, bytes, 0x01020304);
        g.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
    })
    .unwrap();
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let mut res = [TsResult::default(); 2];
    unsafe {
        g.device
            .get_query_pool_results(pool, 0, &mut res, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY)
            .unwrap();
        g.device.destroy_query_pool(pool, None);
    }
    let mask = ts_valid_mask(g.timestamp_valid_bits);
    let ms = ts_zone_ms(res[0], res[1], g.timestamp_period as f64, mask).expect("both queries written");
    buf.destroy(&g.device, &mut alloc);
    Some((ms, wall))
}

#[test]
fn timestamp_conversion_matches_wall_time_and_scales_with_work() {
    let Some(g) = gpu() else { return };
    assert!(g.timestamp_valid_bits > 0, "this queue family cannot take timestamps");
    eprintln!(
        "timestampPeriod {} ns/tick, timestampValidBits {} (mask {:#x})",
        g.timestamp_period,
        g.timestamp_valid_bits,
        ts_valid_mask(g.timestamp_valid_bits)
    );
    // Warm up (first submit pays driver start-up costs), then measure 256 MB / 1 GB / 2 GB fills.
    // Other GPU users (other test binaries, a running editor) inflate individual runs, so each
    // size is repeated and judged by its minimum; a big min/median spread means the GPU was
    // shared during the measurement and the test skips instead of reporting a false failure.
    if fill_ms(&g, 1 << 26).is_none() {
        eprintln!("SKIP: cannot allocate the fill buffer");
        return;
    }
    let mut rows = Vec::new();
    for mb in [256u64, 1024, 2048] {
        let Some(mut runs) = (0..7).map(|_| fill_ms(&g, mb << 20)).collect::<Option<Vec<_>>>() else {
            eprintln!("SKIP: cannot allocate a {mb} MB fill buffer (the device is shared with other work)");
            return;
        };
        runs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (best, median) = (runs[0], runs[runs.len() / 2]);
        let spread = median.0 / best.0.max(1e-6);
        rows.push((mb, best.0, best.1));
        eprintln!(
            "fill {mb:>5} MB: gpu min {:.3} ms median {:.3} ms (spread {spread:.2}x, {:.0} GB/s), wall {:.3} ms",
            best.0,
            median.0,
            mb as f32 / 1024.0 / (best.0 / 1000.0),
            best.1
        );
        if spread > 1.6 {
            eprintln!("SKIP: GPU timing too noisy ({mb} MB spread {spread:.2}x): the device is shared with other work");
            return;
        }
    }
    for (mb, gpu_ms, wall_ms) in &rows {
        assert!(*gpu_ms > 0.0, "{mb} MB: zero GPU time");
        // The GPU interval sits inside the submit's wall time (plus fence/CPU overhead on top).
        assert!((*gpu_ms as f64) <= *wall_ms * 1.25 + 1.0, "{mb} MB: gpu {gpu_ms} ms exceeds wall {wall_ms} ms");
    }
    // Twice the bytes takes about twice as long (fill is bandwidth-bound). The two big sizes are
    // compared: the 256 MB fill can run partly out of the L2 and is only held to "not slower".
    let ratio = rows[2].1 / rows[1].1;
    assert!((1.3..=3.2).contains(&ratio), "2x the bytes took {ratio}x the GPU time");
    assert!(rows[2].1 >= rows[0].1 * 2.0, "8x the bytes took {}x the GPU time", rows[2].1 / rows[0].1);
    // A sane bandwidth for any GPU of this decade (guards a wrong period: e.g. ns vs ticks off by 1000x).
    let gbps = 2.0 / (rows[2].1 / 1000.0);
    assert!((20.0..=5000.0).contains(&gbps), "implausible fill bandwidth {gbps} GB/s");
}

#[test]
fn skin_and_morph_zones_are_stamped_and_nest_inside_the_frame_zone() {
    let Some(g) = gpu() else { return };
    let mask = ts_valid_mask(g.timestamp_valid_bits);
    let period = g.timestamp_period as f64;
    let zone = |res: &[TsResult], z: usize| ts_zone_ms(res[2 * z], res[2 * z + 1], period, mask);
    // Skinning only: SKIN written, MORPH unwritten (reads as unavailable -> 0).
    let (res, wall) = g.skin_timed(200_000, 8, false).unwrap();
    let frame_ms = zone(&res, gpu_zone::FRAME).expect("FRAME written");
    let skin_ms = zone(&res, gpu_zone::SKIN).expect("SKIN written");
    assert!(zone(&res, gpu_zone::MORPH).is_none(), "MORPH must stay unwritten without blendshapes");
    assert_eq!(res[2 * gpu_zone::MORPH].available, 0);
    assert!(skin_ms > 0.0 && skin_ms <= frame_ms + 1e-4, "SKIN {skin_ms} ms vs FRAME {frame_ms} ms");
    assert!((frame_ms as f64) <= wall * 1.25 + 1.0);
    // Every zone but FRAME and SKIN is unavailable.
    for z in 1..gpu_zone::COUNT {
        if z != gpu_zone::SKIN {
            assert!(zone(&res, z).is_none(), "zone {z} unexpectedly written");
        }
    }
    // Twice the meshes ~ twice the skin time (best of several runs: the first ones
    // run at idle GPU clocks).
    let best = |meshes: usize| {
        (0..6).map(|_| zone(&g.skin_timed(200_000, meshes, false).unwrap().0, gpu_zone::SKIN).unwrap()).fold(f32::INFINITY, f32::min)
    };
    let (skin1, skin2) = (best(8), best(16));
    eprintln!("skin: 8 meshes x 200k verts {skin1:.3} ms, 16 meshes {skin2:.3} ms (FRAME {frame_ms:.3} ms)");
    let ratio = skin2 / skin1;
    assert!((1.25..=3.2).contains(&ratio), "2x the meshes took {ratio}x");
    // With blendshapes: MORPH written too, ahead of SKIN, both inside FRAME.
    let (resm, _) = g.skin_timed(200_000, 8, true).unwrap();
    let morph_ms = zone(&resm, gpu_zone::MORPH).expect("MORPH written");
    let skin_m = zone(&resm, gpu_zone::SKIN).expect("SKIN written");
    let fm = zone(&resm, gpu_zone::FRAME).unwrap();
    let begin = |z: usize| resm[2 * z].value & mask;
    eprintln!("morph: 8 meshes x 200k verts morph {morph_ms:.3} ms, skin {skin_m:.3} ms, frame {fm:.3} ms");
    assert!(morph_ms > 0.0 && morph_ms + skin_m <= fm + 1e-3);
    assert!(begin(gpu_zone::MORPH) <= begin(gpu_zone::SKIN), "MORPH zone must start before SKIN");
}
