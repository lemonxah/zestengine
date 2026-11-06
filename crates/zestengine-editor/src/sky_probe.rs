//! `ZESTENGINE_SKY_PROBE=1`: a diagnostic for "the sky flickers while I rotate the scene view".
//! After a warm-up the editor camera yaws a fixed step every frame (the procedural sky only
//! varies with pitch, so its pixels must not change at all), every frame's final viewport
//! colour is copied out at the end of its own command buffer (no stall, so frames overlap
//! exactly as in normal use), and a summary is logged. Off unless the variable is set.
//!
//! Modes (`ZESTENGINE_SKY_PROBE=<mode>`), all driving the camera on a triangle wave of
//! `PERIOD` frames (except `static`), so every pose recurs and same-pose frames can be compared:
//! - `1` (anything else): continuous yaw, top fifth of the viewport compared with frame 0.
//! - `static`: no motion; every consecutive pair must match.
//! - `pitch`: 8 degrees of pitch per frame, written straight into the camera.
//! - `dolly`: 0.6 m per frame along the view direction, written straight into the camera.
//!   The three above mutate `camera` in [`frame`], i.e. BEFORE the pre-egui matrices are read,
//!   so they never exercise the scroll / orbit input path.
//! - `action-dolly` / `action-dolly-small`: the probe queues a [`ProbeAction::Dolly`] (2.0 = one
//!   wheel notch, 1.2 m; small = 0.2) that `viewport_ui` takes with [`take_gestures`] and
//!   applies inside the egui pass through `dolly_now`, the helper the scroll wheel itself
//!   uses, before any viewport overlay is drawn: the real scroll-zoom path.
//! - `action-orbit`: likewise [`ProbeAction::Orbit`] (20 px of drag per frame) with an
//!   [`ProbeAction::OrbitEnd`] every time the wave returns to its start pose (one drag out
//!   and back, then release), so the pivot is re-locked only where a real drag would;
//!   applied through `orbit_now` / `orbit_end_now` where a left drag is.
//! - `action-dolly-stale`: the A/B control for the overlay lag: the wheel amount is returned
//!   from [`frame`] and pushed as an `EditorAction::Dolly` BEFORE the egui pass, so the camera
//!   moves in `process_actions` after the pass, as the wheel once did. The 3D image is still
//!   consistent (the render re-reads the camera) but the overlays were drawn a frame behind.
//! - `action-look`: the control: `FlyCamera::look` (what right-drag mouse-look does) applied
//!   in [`frame`], before the matrices are read, as `update_camera` does for a real drag.
//!
//! `ZESTENGINE_SKY_PROBE_SELECT=<object name>` selects that object on the first probe frame,
//! so the transform gizmo and the selection overlays are part of the composed frames.
//!
//! For the `action-*` modes the viewport frames are compared in three ways: consecutive
//! pairs (always differ while moving: a sanity check that the camera did move), same-pose
//! pairs `PERIOD` apart, and MIRROR pairs: the frame at step `m` of a period against the frame
//! at step `PERIOD - m`, which shows the same pose reached from the opposite direction. If a
//! frame is a pure function of the camera pose, mirror pairs are identical; if the render
//! mixes the previous frame's matrices with this frame's position (the stale-action bug),
//! mirror pairs differ while the plain same-pose pairs (same previous pose too) still agree.
//! The composed swapchain is compared the same way, inside the viewport (where the egui
//! overlays such as the gizmo and object icons are drawn, and which shows them a frame behind
//! if they use a camera the gesture has not yet reached) and outside it.
//! `ZESTENGINE_SKY_PROBE_DUMP=<dir>` writes the first `2 * PERIOD + 2` frames of each capture
//! there as PPMs (`vp###.ppm` viewport, `frame###.ppm` swapchain).
//! `ZESTENGINE_SKY_PROBE_WARMUP=<frames>` overrides the 90-frame default when
//! measuring a lighting cache that needs longer to converge.

use crate::camera::FlyCamera;
use zestengine_render::Renderer;

fn warmup() -> u32 {
    static FRAMES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FRAMES.get_or_init(|| std::env::var("ZESTENGINE_SKY_PROBE_WARMUP").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(90).min(10_000))
}
const FRAMES: u32 = 120;

static FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

const PERIOD: u32 = 20;

/// Pixels per frame of drag for `action-orbit` / `action-look`.
const DRAG_PX: f32 = 20.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Yaw,
    Static,
    Pitch,
    Dolly,
    /// Scroll-wheel dolly through the in-pass path (`dolly_now`); the wheel amount per frame.
    ActionDolly(f32),
    /// The same wheel amount pushed as an `EditorAction::Dolly` before the egui pass: the old
    /// post-pass path, kept to reproduce the overlay lag for an A/B comparison.
    ActionDollyStale(f32),
    ActionOrbit,
    ActionLook,
}

impl Mode {
    /// Compared with the action-mode (same-pose / mirror) summaries.
    fn is_action(self) -> bool {
        matches!(self, Mode::ActionDolly(_) | Mode::ActionDollyStale(_) | Mode::ActionOrbit | Mode::ActionLook)
    }
}

fn mode() -> Mode {
    match std::env::var("ZESTENGINE_SKY_PROBE").as_deref() {
        Ok("static") => Mode::Static,
        Ok("pitch") => Mode::Pitch,
        Ok("dolly") => Mode::Dolly,
        Ok("action-dolly") => Mode::ActionDolly(2.0),
        Ok("action-dolly-small") => Mode::ActionDolly(0.2),
        Ok("action-dolly-stale") => Mode::ActionDollyStale(2.0),
        Ok("action-orbit") => Mode::ActionOrbit,
        Ok("action-look") => Mode::ActionLook,
        _ => Mode::Yaw,
    }
}

/// A camera gesture the probe queues for this frame's egui pass; `viewport_ui` takes it with
/// [`take_gestures`] and applies it through the same in-pass helpers the wheel and a left
/// drag use (`dolly_now`, `orbit_now`, `orbit_end_now`), at the same point of the pass.
#[derive(Clone, Copy, Debug)]
pub enum ProbeAction {
    Dolly(f32),
    Orbit(f32, f32),
    OrbitEnd,
}

static DOLLY_BASE: std::sync::Mutex<Option<glam::Vec3>> = std::sync::Mutex::new(None);

/// Gestures queued by [`frame`] for this frame's `viewport_ui`.
static PENDING: std::sync::Mutex<Vec<ProbeAction>> = std::sync::Mutex::new(Vec::new());

static SELECTION_REQUESTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn enabled() -> bool {
    std::env::var_os("ZESTENGINE_SKY_PROBE").is_some()
}

/// The gestures queued for this frame (empty except in the `action-dolly*` / `action-orbit`
/// modes); `viewport_ui` calls this once per pass and applies them in place of real input.
pub fn take_gestures() -> Vec<ProbeAction> {
    if !enabled() {
        return Vec::new();
    }
    std::mem::take(&mut *PENDING.lock().unwrap())
}

/// `ZESTENGINE_SKY_PROBE_SELECT=<object name>`: the object the editor should select, returned
/// once, on the first probe frame (so the scene is loaded by then).
pub fn take_selection_request() -> Option<String> {
    if !enabled() || FRAME.load(std::sync::atomic::Ordering::Relaxed) <= warmup() {
        return None;
    }
    if SELECTION_REQUESTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    std::env::var("ZESTENGINE_SKY_PROBE_SELECT").ok().filter(|s| !s.is_empty())
}

/// The scene viewport's rect in window pixels `[x0, y0, x1, y1]` (latest frame).
static VIEWPORT_PX: std::sync::Mutex<[u32; 4]> = std::sync::Mutex::new([0; 4]);

/// Call once per editor frame, before the camera matrices are read. Queues this frame's
/// in-pass gestures for [`take_gestures`]; returns a wheel amount only in `action-dolly-stale`,
/// which the editor pushes as an `EditorAction::Dolly` (applied after the pass).
pub fn frame(camera: &mut FlyCamera, renderer: &mut Renderer, viewport_px: [u32; 4]) -> Option<f32> {
    if !enabled() {
        return None;
    }
    let mut out = Vec::new();
    let mut stale = None;
    *VIEWPORT_PX.lock().unwrap() = viewport_px;
    let n = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < warmup() {
        return None;
    }
    let mode = mode();
    if n == warmup() {
        tracing::info!("sky probe: mode {mode:?}, viewport {viewport_px:?}");
        renderer.debug_capture_viewport_frames(true);
    }
    if n < warmup() + FRAMES {
        let k = n - warmup();
        // Triangle wave: pose 0..=PERIOD/2 and back; `step` is this frame's change of pose.
        let t = pose_of(k as usize);
        let step = if k == 0 { 0 } else { t as i32 - pose_of(k as usize - 1) as i32 };
        match mode {
            Mode::Static => {}
            Mode::Dolly => {
                // 0.6 m per step (one wheel notch), triangle wave of PERIOD frames.
                let base = *DOLLY_BASE.lock().unwrap().get_or_insert(camera.position);
                camera.position = base + camera.forward() * (0.6 * t as f32);
            }
            Mode::Pitch => {
                // Fast up/down: 8 degrees per frame along a triangle wave, so each pose
                // recurs every PERIOD frames (and on the way back).
                camera.pitch = (-40.0f32 + 8.0 * t as f32).to_radians();
            }
            Mode::ActionDolly(amount) => {
                if step != 0 {
                    out.push(ProbeAction::Dolly(amount * step as f32));
                }
            }
            Mode::ActionDollyStale(amount) => {
                if step != 0 {
                    stale = Some(amount * step as f32);
                }
            }
            Mode::ActionOrbit => {
                if step != 0 {
                    out.push(ProbeAction::Orbit(DRAG_PX * step as f32, 0.0));
                }
                if k > 0 && t == 0 {
                    out.push(ProbeAction::OrbitEnd);
                }
            }
            Mode::ActionLook => {
                if step != 0 {
                    camera.look(DRAG_PX * step as f32, 0.0);
                }
            }
            Mode::Yaw => camera.yaw += 0.012,
        }
        *PENDING.lock().unwrap() = out;
        return stale;
    }
    if n == warmup() + FRAMES && mode == Mode::ActionOrbit {
        out.push(ProbeAction::OrbitEnd);
    }
    *PENDING.lock().unwrap() = out;
    if n == warmup() + FRAMES + 3 {
        match renderer.debug_take_viewport_frames() {
            Ok(frames) => {
                dump(&frames, "vp");
                match mode {
                    Mode::Static => summarise_static(&frames),
                    Mode::Pitch | Mode::Dolly => summarise_pitch(&frames),
                    m if m.is_action() => summarise_action(&frames),
                    _ => summarise(&frames),
                }
            }
            Err(e) => tracing::error!("sky probe: {e:#}"),
        }
        if mode != Mode::Yaw {
            match renderer.debug_take_swapchain_frames() {
                Ok(frames) => {
                    dump(&frames, "frame");
                    summarise_swapchain(&frames, mode == Mode::Static);
                    if mode.is_action() {
                        summarise_swapchain_viewport(&frames);
                    }
                }
                Err(e) => tracing::error!("sky probe (swapchain): {e:#}"),
            }
        }
        renderer.debug_capture_viewport_frames(false);
    }
    None
}

/// `ZESTENGINE_SKY_PROBE_DUMP=<dir>`: the first two periods of a capture as PPMs.
fn dump(frames: &[(u32, u32, Vec<u8>)], prefix: &str) {
    let Some(dir) = std::env::var_os("ZESTENGINE_SKY_PROBE_DUMP") else { return };
    let _ = std::fs::create_dir_all(&dir);
    for (i, (w, h, px)) in frames.iter().enumerate().take(2 * PERIOD as usize + 2) {
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{prefix}{i:03}.ppm")), out);
    }
}

fn summarise(frames: &[(u32, u32, Vec<u8>)]) {
    let Some((w, h, first)) = frames.first() else { return };
    // The upper fifth of the viewport: sky for the editor's default camera.
    let rows = (*h as usize / 5) * *w as usize * 4;
    let mut worst = 0i32;
    let mut bad = 0;
    for (i, (_, _, px)) in frames.iter().enumerate() {
        let d = px[..rows].iter().zip(&first[..rows]).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
        if d > 2 {
            bad += 1;
        }
        worst = worst.max(d);
        if i < 3 || d > 2 {
            tracing::info!("sky probe: frame {i}: max top-rows difference from frame 0 = {d}");
        }
    }
    tracing::info!("sky probe: {} frames captured at {w}x{h}, {bad} differ from frame 0 in the top fifth (worst {worst})", frames.len());
}

/// Frames `PERIOD` apart show the same pose: they must match everywhere (the dither is +-1).
fn summarise_pitch(frames: &[(u32, u32, Vec<u8>)]) {
    let p = PERIOD as usize;
    let mut worst = (0i32, 0usize, 0usize);
    let mut bad = 0;
    let mut compared = 0;
    // Only frames rendered during the sweep (the last few are after the camera stopped).
    for i in 0..(FRAMES as usize).min(frames.len()).saturating_sub(p) {
        let (a, b) = (&frames[i].2, &frames[i + p].2);
        let mut d = 0;
        let mut n = 0;
        for (x, y) in a.iter().zip(b) {
            let e = (*x as i32 - *y as i32).abs();
            d = d.max(e);
            n += (e > 2) as usize;
        }
        compared += 1;
        if d > 2 {
            bad += 1;
            if bad <= 8 {
                let (w, h, _) = &frames[i];
                let df = diff(*w, *h, a, b);
                let (m1, d1) = best_match(frames, i);
                let (m2, d2) = best_match(frames, i + p);
                tracing::info!(
                    "sky probe: frame {i} (pose {}) vs {} (pose {}): max difference {d}, {n} values over 2, bbox {:?}, grid {}; best match of {i}: frame {m1} (pose {}, max {d1}); of {}: frame {m2} (pose {}, max {d2})",
                    pose_of(i), i + p, pose_of(i + p), df.bbox, grid_str(&df.grid),
                    pose_of(m1), i + p, pose_of(m2)
                );
            }
        }
        if d > worst.0 {
            worst = (d, n, i);
        }
    }
    tracing::info!("sky probe (pitch): {compared} same-pose pairs, {bad} differ by more than 2 (worst {} with {} values over 2, at frame {})", worst.0, worst.1, worst.2);
}

/// The frame showing the same pose as `i` reached from the other direction of the triangle
/// wave (`None` at the turning points, which are their own mirror).
fn mirror_of(i: usize) -> Option<usize> {
    let p = PERIOD as usize;
    let m = i % p;
    if m == 0 || m == p / 2 {
        return None;
    }
    Some(i - m + (p - m))
}

/// Action modes: consecutive, same-pose (`PERIOD` apart) and mirror pairs, each with the
/// differing region (bbox, 8x8 grid, and how much of it is in the top fifth = sky for the
/// default camera). See the module docs for what each comparison tells.
fn summarise_action(frames: &[(u32, u32, Vec<u8>)]) {
    let n = (FRAMES as usize).min(frames.len());
    let Some((w, h, _)) = frames.first() else { return };
    let (w, h) = (*w, *h);
    let report = |name: &str, pairs: Vec<(usize, usize)>| {
        let (mut bad, mut worst, mut worst_pair, mut shown) = (0usize, 0i32, (0usize, 0usize), 0usize);
        let (mut sky_bad, mut geo_bad) = (0usize, 0usize);
        for (i, j) in &pairs {
            let d = diff(w, h, &frames[*i].2, &frames[*j].2);
            if d.max > 2 {
                bad += 1;
                sky_bad += (d.top > 0) as usize;
                geo_bad += (d.over > d.top) as usize;
                if shown < 6 {
                    shown += 1;
                    tracing::info!(
                        "sky probe ({name}): frame {i} (pose {}) vs {j} (pose {}): max {}, {} px over 2 ({} in the top fifth), bbox {:?}, grid {}",
                        pose_of(*i), pose_of(*j), d.max, d.over, d.top, d.bbox, grid_str(&d.grid)
                    );
                }
            }
            if d.max > worst {
                worst = d.max;
                worst_pair = (*i, *j);
            }
        }
        tracing::info!(
            "sky probe ({name}): {bad} / {} pairs differ by more than 2 (worst {worst} at {:?}); {sky_bad} differ in the top fifth (sky), {geo_bad} below it",
            pairs.len(), worst_pair
        );
    };
    report("action consecutive", (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect());
    report("action same-pose", (0..n.saturating_sub(PERIOD as usize)).map(|i| (i, i + PERIOD as usize)).collect());
    report("action mirror", (0..n).filter_map(|i| mirror_of(i).filter(|&j| j > i && j < n).map(|j| (i, j))).collect());
}

/// Where two frames differ: (max channel difference, channel values over 2, of which in the
/// top fifth, bounding box `[x0, y0, x1, y1]` of the pixels over 2, and an 8x8 occupancy grid).
struct Diff {
    max: i32,
    over: usize,
    top: usize,
    bbox: [u32; 4],
    grid: [[u32; 8]; 8],
}

fn diff(w: u32, h: u32, a: &[u8], b: &[u8]) -> Diff {
    let mut d = Diff { max: 0, over: 0, top: 0, bbox: [u32::MAX, u32::MAX, 0, 0], grid: [[0; 8]; 8] };
    for (i, (pa, pb)) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()).enumerate() {
        let e = (0..3).map(|c| (pa[c] as i32 - pb[c] as i32).abs()).max().unwrap_or(0);
        d.max = d.max.max(e);
        if e > 2 {
            d.over += 1;
            let (x, y) = (i as u32 % w, i as u32 / w);
            d.top += (y < h / 5) as usize;
            d.bbox = [d.bbox[0].min(x), d.bbox[1].min(y), d.bbox[2].max(x), d.bbox[3].max(y)];
            d.grid[(y * 8 / h) as usize][(x * 8 / w) as usize] += 1;
        }
    }
    d
}

fn grid_str(g: &[[u32; 8]; 8]) -> String {
    g.iter()
        .map(|r| r.iter().map(|&n| if n == 0 { '.' } else if n < 100 { 'o' } else { '#' }).collect::<String>())
        .collect::<Vec<_>>()
        .join("/")
}

/// Static camera: ALL consecutive pairs (and pairs 2 apart) compared, with where they differ.
fn summarise_static(frames: &[(u32, u32, Vec<u8>)]) {
    let n = (FRAMES as usize).min(frames.len());
    let mut bad = [0usize; 2];
    let mut worst = 0;
    for gap in 1..=2usize {
        for i in 0..n.saturating_sub(gap) {
            let (w, h, a) = &frames[i];
            let d = diff(*w, *h, a, &frames[i + gap].2);
            worst = worst.max(d.max);
            if d.max > 2 {
                bad[gap - 1] += 1;
                if bad[gap - 1] <= 6 {
                    tracing::info!("sky probe (static): frame {i} vs {}: max {} , {} px over 2, bbox {:?}, grid {}", i + gap, d.max, d.over, d.bbox, grid_str(&d.grid));
                }
            }
        }
    }
    if bad[0] + bad[1] > 0 {
        tracing::error!("sky probe (static) FAILED: a fixed camera produced differing viewport frames");
    }
    tracing::info!("sky probe (static): {n} frames; consecutive pairs differing {} / {}, 2-apart {} / {} (worst {worst})", bad[0], n.saturating_sub(1), bad[1], n.saturating_sub(2));
}

/// For a frame that disagrees with its same-pose twin: which other frame (within 3 either
/// side, or a same-pose frame) does it match best? A match with a neighbouring pose means a
/// stale frame was shown instead of this one.
fn best_match(frames: &[(u32, u32, Vec<u8>)], i: usize) -> (usize, i32) {
    let mut best = (i, i32::MAX);
    let lo = i.saturating_sub(3);
    let hi = (i + 3).min(frames.len() - 1);
    let mut cand: Vec<usize> = (lo..=hi).filter(|&j| j != i).collect();
    for k in [PERIOD as usize, 2 * PERIOD as usize] {
        cand.extend(i.checked_sub(k));
        if i + k < frames.len() {
            cand.push(i + k);
        }
    }
    for j in cand {
        let (w, h, a) = &frames[i];
        let d = diff(*w, *h, a, &frames[j].2).max;
        if d < best.1 {
            best = (j, d);
        }
    }
    best
}

fn pose_of(i: usize) -> u32 {
    let m = (i as u32) % PERIOD;
    if m <= PERIOD / 2 { m } else { PERIOD - m }
}

/// Max channel difference and count over 2 between two swapchain frames, restricted to the
/// pixels for which `keep(x, y)` holds, with bbox and grid.
fn diff_where(w: u32, h: u32, a: &[u8], b: &[u8], keep: impl Fn(u32, u32) -> bool) -> Diff {
    let mut d = Diff { max: 0, over: 0, top: 0, bbox: [u32::MAX, u32::MAX, 0, 0], grid: [[0; 8]; 8] };
    for (k, (pa, pb)) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()).enumerate() {
        let (x, y) = (k as u32 % w, k as u32 / w);
        if !keep(x, y) {
            continue;
        }
        let e = (0..3).map(|c| (pa[c] as i32 - pb[c] as i32).abs()).max().unwrap_or(0);
        d.max = d.max.max(e);
        if e > 2 {
            d.over += 1;
            d.bbox = [d.bbox[0].min(x), d.bbox[1].min(y), d.bbox[2].max(x), d.bbox[3].max(y)];
            d.grid[(y * 8 / h) as usize][(x * 8 / w) as usize] += 1;
        }
    }
    d
}

/// The composed window image (egui over the viewport) as it is presented. Static camera: all
/// consecutive pairs; pitch/dolly sweep: same-pose pairs `PERIOD` apart. Pixels inside the
/// scene viewport (plus a 2 px margin) are left out: only the egui UI around it is compared.
fn summarise_swapchain(frames: &[(u32, u32, Vec<u8>)], is_static: bool) {
    let gap = if is_static { 1 } else { PERIOD as usize };
    let n = (FRAMES as usize).min(frames.len());
    let vp = *VIEWPORT_PX.lock().unwrap();
    let in_vp = |x: u32, y: u32| x + 2 >= vp[0] && x <= vp[2] + 2 && y + 2 >= vp[1] && y <= vp[3] + 2;
    let (mut bad, mut cmp, mut worst) = (0, 0, 0);
    for i in 0..n.saturating_sub(gap) {
        let (w, h, a) = &frames[i];
        let d = diff_where(*w, *h, a, &frames[i + gap].2, |x, y| !in_vp(x, y));
        cmp += 1;
        worst = worst.max(d.max);
        if d.max > 2 {
            bad += 1;
            if bad <= 8 {
                tracing::info!("sky probe (swapchain, outside viewport {vp:?}): frame {i} vs {}: max {}, {} px over 2, bbox {:?}, grid {}", i + gap, d.max, d.over, d.bbox, grid_str(&d.grid));
            }
        }
    }
    tracing::info!("sky probe (swapchain, outside viewport {vp:?}): {} frames captured, {bad} / {cmp} pairs {gap} apart differ (worst {worst})", frames.len());
}

/// Action modes: the swapchain INSIDE the viewport (the 3D image plus the egui overlays drawn
/// over it: grid labels, gizmo, light / camera icons) on same-pose and mirror pairs. Overlays
/// drawn with a camera the gesture has not reached yet lag the 3D image by a frame, which
/// shows up here as mirror pairs that differ at the overlay positions while the viewport
/// capture agrees. The fps / frame-time pill in the viewport's bottom-left corner (its text
/// changes with the frame rate, not the camera) is left out: `FPS_PILL_PX` from that corner.
fn summarise_swapchain_viewport(frames: &[(u32, u32, Vec<u8>)]) {
    /// Width and height of the excluded bottom-left corner (the stats pill is ~110x35 px at
    /// 12 px from the corner at 1x scale).
    const FPS_PILL_PX: (u32, u32) = (150, 50);
    let n = (FRAMES as usize).min(frames.len());
    let vp = *VIEWPORT_PX.lock().unwrap();
    let in_vp = |x: u32, y: u32| {
        x >= vp[0] && x < vp[2] && y >= vp[1] && y < vp[3] && !(x < vp[0] + FPS_PILL_PX.0 && y + FPS_PILL_PX.1 >= vp[3])
    };
    let report = |name: &str, pairs: Vec<(usize, usize)>| {
        let (mut bad, mut worst, mut shown) = (0usize, 0i32, 0usize);
        for (i, j) in &pairs {
            let (w, h, a) = &frames[*i];
            let d = diff_where(*w, *h, a, &frames[*j].2, in_vp);
            worst = worst.max(d.max);
            if d.max > 2 {
                bad += 1;
                if shown < 4 {
                    shown += 1;
                    tracing::info!("sky probe (swapchain inside viewport, {name}): frame {i} (pose {}) vs {j} (pose {}): max {}, {} px over 2, bbox {:?}, grid {}", pose_of(*i), pose_of(*j), d.max, d.over, d.bbox, grid_str(&d.grid));
                }
            }
        }
        tracing::info!("sky probe (swapchain inside viewport {vp:?} minus the fps pill, {name}): {bad} / {} pairs differ (worst {worst})", pairs.len());
    };
    report("same-pose", (0..n.saturating_sub(PERIOD as usize)).map(|i| (i, i + PERIOD as usize)).collect());
    report("mirror", (0..n).filter_map(|i| mirror_of(i).filter(|&j| j > i && j < n).map(|j| (i, j))).collect());
}
