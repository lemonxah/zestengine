//! Run a scene with no window, GPU or audio device, for N ticks (NET-1).
//!
//!   cargo run -p zestengine --no-default-features --example headless_tick -- <project-dir> [scene] [ticks] [tick-rate-hz]
//!
//! Loads the scene CPU-side (collision geometry, skeletons, prefab instances,
//! components, physics) and runs gameplay + physics + networking at a fixed tick
//! rate. If the scene has a Network Manager set to auto-start, it hosts or joins.
//! NET-2's dedicated server binary builds on `zestengine_engine::server`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use zestengine_engine::server::{HeadlessConfig, HeadlessHost};

fn main() -> Result<()> {
    zestengine_engine::init_logging();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(args.first().context("usage: headless_tick <project-dir> [scene] [ticks] [tick-rate-hz]")?);
    let scene = args.get(1).filter(|s| s.as_str() != "-").cloned();
    let ticks: u64 = args.get(2).map_or(Ok(600), |s| s.parse()).context("ticks")?;
    let rate: f32 = args.get(3).map_or(Ok(60.0), |s| s.parse()).context("tick rate")?;
    let mut host = HeadlessHost::open(HeadlessConfig { assets_root: root, scene, physics: None, ..Default::default() }, |_| {})?;
    host.auto_start_net();
    println!("loaded {} ({} objects)", host.scene_path(), host.scene().objects.len());
    let dt = 1.0 / rate;
    let start = Instant::now();
    for n in 0..ticks {
        let tick_start = Instant::now();
        host.tick(dt);
        if host.quit_requested() {
            println!("quit requested at tick {n}");
            break;
        }
        if let Some(rest) = Duration::from_secs_f32(dt).checked_sub(tick_start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    println!("{} ticks in {:.2}s", host.ticks(), start.elapsed().as_secs_f32());
    Ok(())
}
