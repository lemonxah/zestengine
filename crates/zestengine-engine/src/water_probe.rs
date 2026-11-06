//! `ZESTENGINE_WATER_PROBE=<dir>`: a diagnostic for "the water surface is not
//! at the Water object's own height". Off unless the variable is set.
//!
//! Every frame of each GPU-simulated water, the newest readback of its height
//! field (`eta`, the surface offset from the authored level, after the dry-cell
//! sink, as the ripple texture holds it) is classified by the cell's role in the
//! sim field and summarised: `interior` (ordinary water), `edge` (water next to
//! a shore cell: the body outline or a bank), `near_obstacle` (water next to an
//! obstacle cell: a collider through the surface), `beach` (rest depth <= 0:
//! ground at or just above the level), `nofloor` (nothing baked under the cell:
//! past a plane edge, outside a room) and `solid` (shore / obstacle cells, whose
//! offset is forced to 0). Twice per simulated second a line is logged with the
//! extreme offset of each region and where it is (local XZ and world XYZ); once
//! per simulated second the whole field, the sim field and the bed are written
//! to `<dir>` as raw little-endian f32 (`<name>_t<sec>.eta`, `<name>_field_v<n>.f32`
//! (vec4 per cell), `<name>_bed.f32`, with `<name>_meta.txt` describing the
//! window and the water's world matrix) for offline plots and a mesh-sampled
//! estimate.
//!
//! `ZESTENGINE_WATER_PROBE_SHOTS=0.5,2,5,10` (seconds of simulated time, the
//! default): the viewport is captured at each and saved as `<name>_shot_<sec>.ppm`.
//! `ZESTENGINE_WATER_PROBE_SECONDS=<s>` (default 12): the process exits once
//! every water simulated that long (so a scripted run never leaves the editor
//! up). `ZESTENGINE_WATER_PROBE_FLAT=1` reports every frame on which a cell of
//! open water is more than 1 mm off the level, for the first 30 frames.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use glam::{Mat4, Vec2, Vec3};
use zestengine_core::{SimField, SimWindow};
use zestengine_render::Renderer;

const REGIONS: [&str; 6] = ["interior", "edge", "near_obstacle", "beach", "nofloor", "solid"];

struct State {
    name: String,
    sim_t: f32,
    frames: u64,
    started: Instant,
    serial: u64,
    readbacks: u64,
    last_log: f32,
    last_dump: f32,
    dumped_versions: Vec<u64>,
    shots: Vec<f32>,
    capturing: Option<f32>,
    done: bool,
    worst: f32,
    worst_at: String,
}

static STATES: Mutex<Option<HashMap<u64, State>>> = Mutex::new(None);

pub fn enabled() -> bool {
    std::env::var_os("ZESTENGINE_WATER_PROBE").is_some()
}

fn dir() -> PathBuf {
    PathBuf::from(std::env::var_os("ZESTENGINE_WATER_PROBE").unwrap_or_default())
}

fn shots() -> Vec<f32> {
    // A dedicated server can inspect simulation data but has no viewport.
    if !cfg!(feature = "client") { return Vec::new(); }
    let s = std::env::var("ZESTENGINE_WATER_PROBE_SHOTS").unwrap_or_else(|_| "0.5,2,5,10".into());
    let mut v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn seconds() -> f32 {
    std::env::var("ZESTENGINE_WATER_PROBE_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(12.0)
}

struct Region {
    count: usize,
    max: f32,
    max_i: usize,
    min: f32,
    min_i: usize,
    sum: f64,
}

impl Region {
    fn new() -> Self {
        Self { count: 0, max: f32::NEG_INFINITY, max_i: 0, min: f32::INFINITY, min_i: 0, sum: 0.0 }
    }
    fn add(&mut self, i: usize, e: f32) {
        self.count += 1;
        self.sum += e as f64;
        if e > self.max {
            self.max = e;
            self.max_i = i;
        }
        if e < self.min {
            self.min = e;
            self.min_i = i;
        }
    }
}

/// The region (index into `REGIONS`) of every cell of `field`.
fn classify(field: &SimField, bed: Option<&[f32]>) -> Vec<u8> {
    let n = field.window.n as usize;
    let cells = &field.cells;
    let kind = |i: usize| cells[i][3];
    (0..n * n)
        .map(|i| {
            if kind(i) > 0.5 {
                return 5;
            }
            if cells[i][0] <= 0.0 {
                return 3;
            }
            let (x, y) = (i % n, i / n);
            let mut shore = false;
            let mut obstacle = false;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= n as i32 || ny >= n as i32 {
                    shore = true;
                    continue;
                }
                let k = kind(ny as usize * n + nx as usize);
                shore |= (k - 1.0).abs() < 0.5;
                obstacle |= k > 1.5;
            }
            if obstacle {
                return 2;
            }
            if shore {
                return 1;
            }
            if bed.is_some_and(|b| b.get(i).is_some_and(|d| d.is_nan())) {
                return 4;
            }
            0
        })
        .collect()
}

fn centre(w: SimWindow, i: usize) -> Vec2 {
    let n = w.n as usize;
    w.lo() + (Vec2::new((i % n) as f32, (i / n) as f32) + 0.5) * w.cell
}

fn where_is(w: SimWindow, world: Mat4, i: usize, e: f32) -> String {
    let n = w.n as usize;
    let l = centre(w, i);
    let p = world.transform_point3(Vec3::new(l.x, e, l.y));
    format!("cell ({},{}) local ({:.2},{:.2}) world ({:.2},{:.3},{:.2})", i % n, i / n, l.x, l.y, p.x, p.y, p.z)
}

fn write_f32s(path: PathBuf, data: impl Iterator<Item = f32>) {
    if let Ok(mut f) = std::fs::File::create(path) {
        let mut buf = Vec::new();
        for v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&buf);
    }
}

/// Call once per frame per GPU-simulated water, after its field for this frame
/// is known and before its simulation is queued. `dt` is the sim time this
/// frame advances by.
#[allow(clippy::too_many_arguments)]
pub fn frame(renderer: &mut Renderer, key: u64, name: &str, world: Mat4, window: SimWindow, field: &SimField, bed: Option<&[f32]>, dt: f32) {
    if !enabled() {
        return;
    }
    let mut guard = STATES.lock().unwrap();
    let states = guard.get_or_insert_with(HashMap::new);
    let st = states.entry(key).or_insert_with(|| {
        let _ = std::fs::create_dir_all(dir());
        State {
            name: name.replace(|c: char| !c.is_alphanumeric(), "_"),
            sim_t: 0.0,
            frames: 0,
            started: Instant::now(),
            serial: 0,
            readbacks: 0,
            last_log: -1.0,
            last_dump: -1.0,
            dumped_versions: Vec::new(),
            shots: shots(),
            capturing: None,
            done: false,
            worst: 0.0,
            worst_at: String::new(),
        }
    });
    if st.done {
        return;
    }
    st.frames += 1;
    let n = window.n as usize;
    let wall = st.started.elapsed().as_secs_f32();

    // Field versions: dump each once.
    if !st.dumped_versions.contains(&field.version) {
        st.dumped_versions.push(field.version);
        let counts = classify(field, bed).iter().fold([0usize; 6], |mut c, &r| {
            c[r as usize] += 1;
            c
        });
        let beach_lo = field.cells.iter().filter(|c| c[3] < 0.5 && c[0] <= 0.0).map(|c| c[0]).fold(0.0f32, f32::min);
        tracing::info!(
            "water probe [{}]: frame {} t={:.3}s field v{} window n={} cell={:.4} origin={:?} closed={} default_depth={} max_depth={:.3}; cells interior={} edge={} near_obstacle={} beach={} nofloor={} solid={} (beach lowest rest {:.3} = ground up to {:.3} m above the level)",
            st.name, st.frames, st.sim_t, field.version, n, window.cell, window.origin, window.closed, field.default_depth, field.max_depth,
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], beach_lo, -beach_lo
        );
        write_f32s(dir().join(format!("{}_field_v{}.f32", st.name, field.version)), field.cells.iter().flat_map(|c| c.iter().copied()));
        if let Some(b) = bed {
            write_f32s(dir().join(format!("{}_bed_v{}.f32", st.name, field.version)), b.iter().copied());
        }
        let m = world.to_cols_array();
        let _ = std::fs::write(
            dir().join(format!("{}_meta.txt", st.name)),
            format!("n {n}\ncell {}\norigin {} {}\nclosed {}\nworld {:?}\nfield_version {}\n", window.cell, window.origin.x, window.origin.y, window.closed, m, field.version),
        );
    }

    // The newest readback.
    if let Some(rb) = renderer.water_sim_readback(key)
        && rb.serial != st.serial
    {
        st.serial = rb.serial;
        st.readbacks += 1;
        let same = rb.n as usize == n && rb.origin == window.origin.to_array();
        if !same {
            tracing::warn!("water probe [{}]: readback window n={} origin={:?} differs from the field's (n={} origin={:?})", st.name, rb.n, rb.origin, n, window.origin);
        } else {
            let roles = classify(field, bed);
            let mut regions: Vec<Region> = (0..6).map(|_| Region::new()).collect();
            for (i, c) in rb.cells.iter().enumerate() {
                regions[roles[i] as usize].add(i, c[0]);
            }
            let open_worst = regions[..5].iter().filter(|r| r.count > 0).map(|r| r.max.abs().max(r.min.abs())).fold(0.0f32, f32::max);
            if open_worst > st.worst {
                st.worst = open_worst;
                let r = regions[..5].iter().enumerate().filter(|(_, r)| r.count > 0).max_by(|a, b| a.1.max.abs().max(a.1.min.abs()).partial_cmp(&b.1.max.abs().max(b.1.min.abs())).unwrap()).unwrap();
                let (i, e) = if r.1.max.abs() >= r.1.min.abs() { (r.1.max_i, r.1.max) } else { (r.1.min_i, r.1.min) };
                st.worst_at = format!("t={:.3}s {} {:+.4} m at {}", st.sim_t, REGIONS[r.0], e, where_is(window, world, i, e));
            }
            let flat = std::env::var_os("ZESTENGINE_WATER_PROBE_FLAT").is_some() && st.frames <= 30 && open_worst > 1e-3;
            if st.sim_t - st.last_log >= 0.5 || st.last_log < 0.0 || flat {
                st.last_log = st.sim_t;
                let mut line = format!(
                    "water probe [{}]: t={:.3}s wall={:.2}s frame {} ({:.0} fps, dt {:.4}) readback #{} field v{}:",
                    st.name, st.sim_t, wall, st.frames, st.frames as f32 / wall.max(1e-3), dt, st.readbacks, field.version
                );
                for (k, r) in regions.iter().enumerate() {
                    if r.count == 0 {
                        continue;
                    }
                    let mean = r.sum / r.count as f64;
                    line += &format!(" | {} n={} mean {:+.4} max {:+.4} @{} min {:+.4} @{}", REGIONS[k], r.count, mean, r.max, where_is(window, world, r.max_i, r.max), r.min, where_is(window, world, r.min_i, r.min));
                }
                tracing::info!("{line}");
            }
            if st.sim_t - st.last_dump >= 1.0 || st.last_dump < 0.0 {
                st.last_dump = st.sim_t;
                write_f32s(dir().join(format!("{}_t{:06.2}.eta", st.name, st.sim_t)), rb.cells.iter().map(|c| c[0]));
            }
        }
    }

    // Viewport shots at the asked sim times: switch the capture on this frame,
    // take it next frame.
    #[cfg(feature = "client")]
    if let Some(t) = st.capturing.take() {
        match renderer.debug_take_viewport_frames() {
            Ok(frames) if !frames.is_empty() => {
                let (w, h, px) = frames.last().unwrap();
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                let path = dir().join(format!("{}_shot_{:05.2}.ppm", st.name, t));
                let _ = std::fs::write(&path, out);
                tracing::info!("water probe [{}]: shot at t={:.2}s -> {} ({w}x{h})", st.name, t, path.display());
                renderer.debug_capture_viewport_frames(false);
            }
            Ok(_) => st.capturing = Some(t),
            Err(e) => {
                tracing::error!("water probe: capture: {e:#}");
                renderer.debug_capture_viewport_frames(false);
            }
        }
    } else if st.shots.first().is_some_and(|&s| st.sim_t >= s) {
        let t = st.shots.remove(0);
        renderer.debug_capture_viewport_frames(true);
        st.capturing = Some(t);
    }

    st.sim_t += dt.max(0.0);
    if st.sim_t >= seconds() && st.shots.is_empty() && st.capturing.is_none() {
        st.done = true;
        tracing::info!("water probe [{}]: done after {:.2}s sim / {:.2}s wall / {} frames; worst open-water offset {:.4} m: {}", st.name, st.sim_t, wall, st.frames, st.worst, st.worst_at);
        if states.values().all(|s| s.done) {
            tracing::info!("water probe: all waters done, exiting");
            std::process::exit(0);
        }
    }
}
