//! Hitch capture: names the cause of a single long frame.
//!
//! The frame log (`FrameTimeLog`) only says a frame was slow. This module splits
//! each frame-to-frame interval into named phases and scopes, keeps a ring of
//! recent event markers (focus, scene load, task start/finish, ...), and when an
//! interval exceeds the threshold (default 50 ms, `ZESTENGINE_HITCH_MS`) logs ONE
//! line under target `frame_log::hitch` at `warn`:
//!
//! ```text
//! hitch total_ms=412.3 top=wait:device_wait_idle@viewport_resize=380.2,render_record=20.1 \
//!   counters=tex_upload=4 events=-0.41s:focus_gained,-0.40s:task_start(focus_reimport)
//! ```
//!
//! Timings are EXCLUSIVE (a scope's time is subtracted from its parent), so the
//! top list names the leaf that actually blocked, not the phase around it. The
//! per-frame state is thread-local and only the thread that calls
//! [`frame_begin`] (the main thread) records; calls from worker threads are no-ops.
//! To catch MISSED VBLANKS under VSync (a 120 Hz frame whose work ran past 8.33 ms
//! waits for the next vblank, so it shows up as 16.7 ms), set
//! `ZESTENGINE_HITCH_MS=10`: every frame over 10 ms is then reported, with the
//! CPU work excluding the vblank wait (`cpu_ms`), the time blocked in acquire /
//! present, and the GPU time of the last completed frame (`gpu_ms`), so CPU-bound,
//! GPU-bound and vblank-quantised frames can be told apart. Lines are rate
//! limited (at most 5 per second; `suppressed=N` on the next line counts the rest).
//! Nothing allocates per frame: a fixed table and `Instant::now` at boundaries.
//! Top-level [`phase`] marks always run; [`scope`] (the deep, named blocking
//! calls) is on by default and can be switched off with `ZESTENGINE_HITCH_SCOPES=0`.
//! Events and the last hitch are process-wide so worker threads can post events
//! and the profiler window can read the breakdown.

use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Default hitch threshold (ms); `ZESTENGINE_HITCH_MS` overrides it.
pub const DEFAULT_HITCH_MS: f32 = 50.0;
/// Entries per frame table (phases + scopes, deduplicated by name).
const MAX_ENTRIES: usize = 48;
/// Distinct per-frame counters.
const MAX_COUNTERS: usize = 8;
/// Events kept in the ring.
pub const MAX_EVENTS: usize = 32;
/// Nesting depth of open scopes tracked for exclusive timing.
const MAX_DEPTH: usize = 8;
/// How far back the hitch line reports events.
const EVENT_WINDOW_US: u64 = 2_000_000;
/// Hitch lines logged per second at most.
const MAX_LINES_PER_SEC: u32 = 5;
/// Phases listed in the hitch line.
pub const TOP_N: usize = 5;

/// Fixed-size table of (name, summed exclusive time) for one frame.
#[derive(Clone, Copy)]
pub struct PhaseTable {
    names: [&'static str; MAX_ENTRIES],
    ns: [u64; MAX_ENTRIES],
    len: usize,
}

impl PhaseTable {
    pub const fn new() -> Self {
        Self { names: [""; MAX_ENTRIES], ns: [0; MAX_ENTRIES], len: 0 }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Add `d` under `name`, merging with an existing entry; drops the sample
    /// when the table is full (never allocates).
    pub fn add(&mut self, name: &'static str, d: Duration) {
        let ns = d.as_nanos().min(u64::MAX as u128) as u64;
        for i in 0..self.len {
            let n = self.names[i];
            if std::ptr::eq(n, name) || n == name {
                self.ns[i] = self.ns[i].saturating_add(ns);
                return;
            }
        }
        if self.len < MAX_ENTRIES {
            self.names[self.len] = name;
            self.ns[self.len] = ns;
            self.len += 1;
        }
    }

    /// Summed time under `name` (zero when absent).
    pub fn get(&self, name: &str) -> Duration {
        (0..self.len)
            .find(|&i| self.names[i] == name)
            .map_or(Duration::ZERO, |i| Duration::from_nanos(self.ns[i]))
    }

    pub fn total(&self) -> Duration {
        Duration::from_nanos(self.ns[..self.len].iter().fold(0u64, |a, &b| a.saturating_add(b)))
    }

    /// The `N` longest entries, longest first (ties keep insertion order).
    /// Returns the array and how many slots are filled.
    pub fn top<const N: usize>(&self) -> ([(&'static str, Duration); N], usize) {
        let mut out = [("", Duration::ZERO); N];
        let mut n = 0;
        for i in 0..self.len {
            let d = Duration::from_nanos(self.ns[i]);
            // Insertion point: after the last entry that is >= d.
            let mut pos = n;
            while pos > 0 && out[pos - 1].1 < d {
                pos -= 1;
            }
            if pos >= N {
                continue;
            }
            let end = n.min(N - 1);
            for j in (pos..end).rev() {
                out[j + 1] = out[j];
            }
            out[pos] = (self.names[i], d);
            n = (n + 1).min(N);
        }
        (out, n)
    }
}

impl Default for PhaseTable {
    fn default() -> Self {
        Self::new()
    }
}

/// One event marker.
#[derive(Clone, Copy)]
pub struct Event {
    /// Microseconds since the process epoch.
    pub at_us: u64,
    pub name: &'static str,
    /// Optional static detail (e.g. a task kind).
    pub tag: &'static str,
    /// Optional number (e.g. a count).
    pub val: u64,
}

/// Fixed ring of the most recent events.
pub struct EventRing {
    buf: [Event; MAX_EVENTS],
    head: usize,
    len: usize,
}

impl EventRing {
    pub const fn new() -> Self {
        Self {
            buf: [Event { at_us: 0, name: "", tag: "", val: 0 }; MAX_EVENTS],
            head: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, e: Event) {
        self.buf[self.head] = e;
        self.head = (self.head + 1) % MAX_EVENTS;
        self.len = (self.len + 1).min(MAX_EVENTS);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Events no older than `window_us` before `now_us`, oldest first.
    pub fn recent(&self, now_us: u64, window_us: u64) -> Vec<Event> {
        let start = (self.head + MAX_EVENTS - self.len) % MAX_EVENTS;
        (0..self.len)
            .map(|i| self.buf[(start + i) % MAX_EVENTS])
            .filter(|e| now_us.saturating_sub(e.at_us) <= window_us)
            .collect()
    }
}

impl Default for EventRing {
    fn default() -> Self {
        Self::new()
    }
}

/// True when an interval of `total_ms` counts as a hitch.
pub fn is_hitch(total_ms: f32, threshold_ms: f32) -> bool {
    total_ms > threshold_ms
}

/// Parse a `ZESTENGINE_HITCH_MS` value; anything non-positive or unparsable
/// falls back to the default.
pub fn parse_threshold(v: Option<&str>) -> f32 {
    v.and_then(|s| s.trim().parse::<f32>().ok()).filter(|t| *t > 0.0).unwrap_or(DEFAULT_HITCH_MS)
}

fn threshold_ms() -> f32 {
    static T: OnceLock<f32> = OnceLock::new();
    *T.get_or_init(|| parse_threshold(std::env::var("ZESTENGINE_HITCH_MS").ok().as_deref()))
}

fn scopes_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZESTENGINE_HITCH_SCOPES").map(|v| v != "0").unwrap_or(true))
}

fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

fn now_us() -> u64 {
    epoch().elapsed().as_micros() as u64
}

/// CPU work of an interval excluding the time blocked in acquire / present
/// (where the vblank wait happens).
pub fn cpu_work_ms(total_ms: f32, acquire_ms: f32, present_ms: f32) -> f32 {
    (total_ms - acquire_ms - present_ms).max(0.0)
}

/// Allows at most `limit` events per `window_us`; counts what it suppressed.
pub struct RateLimiter {
    limit: u32,
    window_us: u64,
    win_start: u64,
    used: u32,
    suppressed: u32,
}

impl RateLimiter {
    pub const fn new(limit: u32, window_us: u64) -> Self {
        Self { limit, window_us, win_start: 0, used: 0, suppressed: 0 }
    }

    /// `Some(n)` when an event at `now_us` may be logged (`n` = events suppressed
    /// since the last logged one), `None` when it is suppressed.
    pub fn try_pass(&mut self, now_us: u64) -> Option<u32> {
        if self.used == 0 || now_us.saturating_sub(self.win_start) >= self.window_us {
            self.win_start = now_us;
            self.used = 0;
        }
        if self.used < self.limit {
            self.used += 1;
            Some(std::mem::take(&mut self.suppressed))
        } else {
            self.suppressed = self.suppressed.saturating_add(1);
            None
        }
    }
}

/// Timings of one finished frame-to-frame interval.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Interval {
    pub total_ms: f32,
    /// `total` minus time blocked in acquire and present.
    pub cpu_ms: f32,
    pub acquire_ms: f32,
    pub present_ms: f32,
    /// Acquire returning -> present being called (the recorded + submitted part).
    pub acq_to_present_ms: Option<f32>,
    /// GPU frame time of the last completed frame, when timestamps exist.
    pub gpu_ms: Option<f32>,
}

/// Per-second aggregate of CPU work (excluding the vblank wait) and GPU time.
#[derive(Default, Clone, Copy)]
pub struct WorkAgg {
    n: u32,
    cpu_sum: f32,
    cpu_max: f32,
    gpu_n: u32,
    gpu_sum: f32,
    gpu_max: f32,
}

/// One finished second of [`WorkAgg`].
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct WorkSummary {
    pub cpu_mean: f32,
    pub cpu_max: f32,
    pub gpu_mean: Option<f32>,
    pub gpu_max: Option<f32>,
}

impl WorkAgg {
    pub fn record(&mut self, cpu_ms: f32, gpu_ms: Option<f32>) {
        self.n += 1;
        self.cpu_sum += cpu_ms;
        self.cpu_max = self.cpu_max.max(cpu_ms);
        if let Some(g) = gpu_ms {
            self.gpu_n += 1;
            self.gpu_sum += g;
            self.gpu_max = self.gpu_max.max(g);
        }
    }

    /// Summarise and reset.
    pub fn finish(&mut self) -> WorkSummary {
        let a = std::mem::take(self);
        WorkSummary {
            cpu_mean: a.cpu_sum / a.n.max(1) as f32,
            cpu_max: a.cpu_max,
            gpu_mean: (a.gpu_n > 0).then(|| a.gpu_sum / a.gpu_n as f32),
            gpu_max: (a.gpu_n > 0).then_some(a.gpu_max),
        }
    }
}

impl WorkSummary {
    /// ` cpu_work_mean=.. cpu_work_max=.. [gpu_mean=.. gpu_max=..]` for the per-second line.
    pub fn fields(&self) -> String {
        let mut s = format!(" cpu_work_mean={:.2}ms cpu_work_max={:.2}ms", self.cpu_mean, self.cpu_max);
        if let (Some(m), Some(x)) = (self.gpu_mean, self.gpu_max) {
            s.push_str(&format!(" gpu_mean={m:.2}ms gpu_max={x:.2}ms"));
        }
        s
    }
}

/// Breakdown of the most recent hitch, for the Profiler window.
#[derive(Clone, Debug, Default)]
pub struct HitchReport {
    pub total_ms: f32,
    pub interval: Interval,
    /// Hitches not logged (rate limit) since the previous logged line.
    pub suppressed: u32,
    /// Longest exclusive phases/scopes, longest first.
    pub top: Vec<(&'static str, f32)>,
    /// (seconds before the end of the hitch, name, detail) for the last 2 s.
    pub events: Vec<(f32, String)>,
    pub counters: Vec<(&'static str, u32)>,
}

impl HitchReport {
    /// The single copy-friendly log line (key=value).
    pub fn line(&self) -> String {
        let top = self.top.iter().map(|(n, ms)| format!("{n}={ms:.1}")).collect::<Vec<_>>().join(",");
        let counters = self.counters.iter().map(|(n, c)| format!("{n}={c}")).collect::<Vec<_>>().join(",");
        let events = self.events.iter().map(|(ago, s)| format!("-{ago:.2}s:{s}")).collect::<Vec<_>>().join(",");
        let i = &self.interval;
        let opt = |v: Option<f32>| v.map_or("na".to_string(), |v| format!("{v:.1}"));
        format!(
            "hitch total_ms={:.1} cpu_ms={:.1} acquire_ms={:.1} present_ms={:.1} acq_to_present_ms={} gpu_ms={} suppressed={} top={top} counters={counters} events={events}",
            self.total_ms, i.cpu_ms, i.acquire_ms, i.present_ms, opt(i.acq_to_present_ms), opt(i.gpu_ms), self.suppressed
        )
    }
}

fn fmt_event(e: &Event) -> String {
    let mut s = e.name.to_string();
    if !e.tag.is_empty() {
        s.push('(');
        s.push_str(e.tag);
        s.push(')');
    }
    if e.val != 0 {
        s.push_str(&format!("={}", e.val));
    }
    s
}

static EVENTS: Mutex<EventRing> = Mutex::new(EventRing::new());
static LAST: Mutex<Option<HitchReport>> = Mutex::new(None);

/// Post an event marker (any thread). `tag` is a static detail, `val` a number
/// (0 = none).
pub fn event(name: &'static str, tag: &'static str, val: u64) {
    if let Ok(mut r) = EVENTS.lock() {
        r.push(Event { at_us: now_us(), name, tag, val });
    }
}

/// The most recent hitch breakdown, if any.
pub fn last_hitch() -> Option<HitchReport> {
    LAST.lock().ok().and_then(|g| g.clone())
}

struct State {
    frame_start: Instant,
    table: PhaseTable,
    counters: [(&'static str, u32); MAX_COUNTERS],
    n_counters: usize,
    phase: &'static str,
    phase_start: Instant,
    /// Child time per open depth; `[0]` belongs to the current phase.
    child_ns: [u64; MAX_DEPTH + 1],
    depth: usize,
    acquired_at: Option<Instant>,
    /// Start of the event-loop sleep the frame limiter requested (`WaitUntil`).
    idle_start: Option<Instant>,
    present_at: Option<Instant>,
    gpu_ms: Option<f32>,
    prev: Option<Interval>,
    /// Phase table of the interval that just ended.
    prev_phases: PhaseTable,
    rate: RateLimiter,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| s.try_borrow_mut().ok().and_then(|mut g| g.as_mut().map(f)))
}

impl State {
    fn close_phase(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.phase_start).as_nanos() as u64;
        let own = elapsed.saturating_sub(self.child_ns[0]);
        self.table.add(self.phase, Duration::from_nanos(own));
        self.child_ns[0] = 0;
    }
}

/// Start a new frame-to-frame interval (call at the top of `redraw`). Closes
/// the previous interval, logging it if it was a hitch. The returned guard marks
/// the `between_frames` phase when `redraw` returns, so event-loop work (window
/// events, focus handlers) between redraws is attributed too.
pub fn frame_begin() -> FrameGuard {
    let now = Instant::now();
    STATE.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else { return };
        match slot.as_mut() {
            None => {
                *slot = Some(State {
                    frame_start: now,
                    table: PhaseTable::new(),
                    counters: [("", 0); MAX_COUNTERS],
                    n_counters: 0,
                    phase: "redraw_start",
                    phase_start: now,
                    child_ns: [0; MAX_DEPTH + 1],
                    depth: 0,
                    acquired_at: None,
                    idle_start: None,
                    present_at: None,
                    gpu_ms: None,
                    prev: None,
                    prev_phases: PhaseTable::new(),
                    rate: RateLimiter::new(MAX_LINES_PER_SEC, 1_000_000),
                });
            }
            Some(st) => {
                // Event-loop sleep requested by the frame limiter: attribute it
                // to `frame_limiter` instead of the `between_frames` phase.
                if let Some(t) = st.idle_start.take() {
                    let d = now.duration_since(t);
                    st.table.add("frame_limiter", d);
                    st.child_ns[0] = st.child_ns[0].saturating_add(d.as_nanos().min(u64::MAX as u128) as u64);
                }
                st.close_phase(now);
                let total_ms = now.duration_since(st.frame_start).as_secs_f32() * 1000.0;
                let acquire_ms = st.table.get("acquire_image").as_secs_f32() * 1000.0;
                let present_ms = st.table.get("present").as_secs_f32() * 1000.0;
                let limiter_ms = st.table.get("frame_limiter").as_secs_f32() * 1000.0;
                let iv = Interval {
                    total_ms,
                    // The frame limiter's deliberate wait is neither CPU work nor a vblank wait.
                    cpu_ms: cpu_work_ms(total_ms - limiter_ms, acquire_ms, present_ms),
                    acquire_ms,
                    present_ms,
                    acq_to_present_ms: match (st.acquired_at, st.present_at) {
                        (Some(a), Some(p)) if p >= a => Some(p.duration_since(a).as_secs_f32() * 1000.0),
                        _ => None,
                    },
                    gpu_ms: st.gpu_ms,
                };
                st.prev = Some(iv);
                if is_hitch(total_ms, threshold_ms()) {
                    report(st, iv);
                }
                st.acquired_at = None;
                st.present_at = None;
                st.prev_phases = st.table;
                st.table.clear();
                st.n_counters = 0;
                st.depth = 0;
                st.child_ns = [0; MAX_DEPTH + 1];
                st.frame_start = now;
                st.phase = "redraw_start";
                st.phase_start = now;
            }
        }
    });
    FrameGuard
}

/// The frame limiter is about to leave the event loop sleeping (`WaitUntil`);
/// the time until the next [`frame_begin`] is booked to the `frame_limiter`
/// scope (not `between_frames`, not CPU work). Idempotent until that frame begins.
pub fn limiter_idle_begin() {
    let now = Instant::now();
    with_state(|st| {
        if st.idle_start.is_none() {
            st.idle_start = Some(now);
        }
    });
}

/// Timings of the interval that just ended (valid right after [`frame_begin`]).
pub fn last_interval() -> Option<Interval> {
    with_state(|st| st.prev).flatten()
}

/// Phase times of the interval that just ended (valid right after [`frame_begin`]).
pub fn last_phases() -> Option<PhaseTable> {
    with_state(|st| st.prev_phases)
}

/// Per-second sum of the CPU phases (waits excluded), for the `phases:` field of the frame log.
#[derive(Default)]
pub struct PhaseAgg {
    names: Vec<&'static str>,
    ns: Vec<u64>,
    frames: u32,
}

impl PhaseAgg {
    pub fn record(&mut self, t: &PhaseTable) {
        self.frames += 1;
        for i in 0..t.len {
            let n = t.names[i];
            if matches!(n, "frame_limiter" | "acquire_image" | "present") {
                continue;
            }
            match self.names.iter().position(|x| *x == n) {
                Some(k) => self.ns[k] += t.ns[i],
                None => {
                    self.names.push(n);
                    self.ns.push(t.ns[i]);
                }
            }
        }
    }

    /// ` phases: a=9.8 b=4.1 c=3.0` (mean ms per frame, top 12 of at least 0.05 ms), and reset. Empty when nothing was recorded.
    pub fn finish_top3(&mut self) -> String {
        let frames = self.frames.max(1) as f64;
        let mut v: Vec<(&'static str, f64)> = self.names.iter().zip(&self.ns).map(|(n, ns)| (*n, *ns as f64 / frames / 1e6)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        *self = Self::default();
        if v.is_empty() {
            return String::new();
        }
        let list: Vec<String> = v.iter().take(12).filter(|(_, ms)| *ms >= 0.05).map(|(n, ms)| format!("{n}={ms:.1}")).collect();
        format!(" phases: {}", list.join(" "))
    }
}

/// Tell the hitch capture the GPU frame time of the last completed frame
/// (`None` when GPU timestamps are unavailable).
pub fn set_gpu_ms(ms: Option<f32>) {
    with_state(|st| st.gpu_ms = ms.filter(|m| *m > 0.0));
}

/// Run the main swapchain's image acquire (where a FIFO vblank wait can block)
/// inside the `acquire_image` scope.
pub fn acquire<T>(f: impl FnOnce() -> T) -> T {
    let r = {
        let _h = scope("acquire_image");
        f()
    };
    let now = Instant::now();
    with_state(|st| st.acquired_at = Some(now));
    r
}

/// Run the main swapchain's present (blocks on the vblank under some drivers)
/// inside the `present` scope.
pub fn present<T>(f: impl FnOnce() -> T) -> T {
    let now = Instant::now();
    with_state(|st| st.present_at = Some(now));
    let _h = scope("present");
    f()
}

/// Marks `between_frames` when dropped (end of `redraw`).
pub struct FrameGuard;

impl Drop for FrameGuard {
    fn drop(&mut self) {
        phase("between_frames");
    }
}

/// Mark the start of a top-level phase; the previous phase ends here.
pub fn phase(name: &'static str) {
    let now = Instant::now();
    with_state(|st| {
        st.close_phase(now);
        st.phase = name;
        st.phase_start = now;
    });
}

/// Count something this frame (e.g. `tex_upload`); shown in the hitch line.
pub fn count(name: &'static str, n: u32) {
    with_state(|st| {
        for i in 0..st.n_counters {
            if st.counters[i].0 == name {
                st.counters[i].1 += n;
                return;
            }
        }
        if st.n_counters < MAX_COUNTERS {
            st.counters[st.n_counters] = (name, n);
            st.n_counters += 1;
        }
    });
}

/// RAII scope for a named, possibly blocking call. Hold it across the call:
/// `let _h = hitch::scope("wait:device_wait_idle@viewport_resize");`.
#[must_use]
pub struct Scope {
    name: &'static str,
    start: Option<Instant>,
    depth: u8,
}

/// Open a named scope inside the current phase (no-op off the main thread or
/// with `ZESTENGINE_HITCH_SCOPES=0`).
pub fn scope(name: &'static str) -> Scope {
    let off = Scope { name, start: None, depth: 0 };
    if !scopes_on() {
        return off;
    }
    with_state(|st| {
        if st.depth >= MAX_DEPTH {
            return None;
        }
        st.depth += 1;
        st.child_ns[st.depth] = 0;
        Some(Scope { name, start: Some(Instant::now()), depth: st.depth as u8 })
    })
    .flatten()
    .unwrap_or(off)
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let elapsed = start.elapsed();
        let d = self.depth as usize;
        with_state(|st| {
            let ns = elapsed.as_nanos() as u64;
            st.table.add(self.name, Duration::from_nanos(ns.saturating_sub(st.child_ns[d])));
            st.child_ns[d - 1] = st.child_ns[d - 1].saturating_add(ns);
            st.depth = d - 1;
        });
    }
}

/// `device_wait_idle` / `wait_for_fences` with a named scope, so a stall is
/// attributed to its call site (`wait:device_wait_idle@viewport_resize`). Same
/// signatures (and `unsafe`) as the `ash::Device` methods they wrap.
pub trait WaitNamed {
    /// # Safety
    /// Same contract as `ash::Device::device_wait_idle`.
    unsafe fn wait_idle(&self, site: &'static str) -> ash::prelude::VkResult<()>;
    /// # Safety
    /// Same contract as `ash::Device::wait_for_fences`.
    unsafe fn wait_fences(&self, site: &'static str, fences: &[ash::vk::Fence], all: bool, timeout: u64) -> ash::prelude::VkResult<()>;
}

/// How many `wait_idle` calls this process has made (tests assert that a path
/// stalls on none).
static IDLE_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn idle_wait_count() -> u64 {
    IDLE_WAITS.load(std::sync::atomic::Ordering::Relaxed)
}

impl WaitNamed for ash::Device {
    unsafe fn wait_idle(&self, site: &'static str) -> ash::prelude::VkResult<()> {
        IDLE_WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _h = scope(site);
        unsafe { self.device_wait_idle() }
    }
    unsafe fn wait_fences(&self, site: &'static str, fences: &[ash::vk::Fence], all: bool, timeout: u64) -> ash::prelude::VkResult<()> {
        let _h = scope(site);
        unsafe { self.wait_for_fences(fences, all, timeout) }
    }
}

/// Build + log + store the report for an interval that crossed the threshold.
fn report(st: &mut State, iv: Interval) {
    let now = now_us();
    let Some(suppressed) = st.rate.try_pass(now) else { return };
    let total_ms = iv.total_ms;
    let (top, n) = st.table.top::<TOP_N>();
    let events = EVENTS.lock().map(|r| r.recent(now, EVENT_WINDOW_US)).unwrap_or_default();
    let rep = HitchReport {
        total_ms,
        interval: iv,
        suppressed,
        top: top[..n].iter().map(|(name, d)| (*name, d.as_secs_f32() * 1000.0)).collect(),
        events: events.iter().map(|e| (now.saturating_sub(e.at_us) as f32 / 1e6, fmt_event(e))).collect(),
        counters: st.counters[..st.n_counters].iter().filter(|c| c.1 > 0).copied().collect(),
    };
    tracing::warn!(target: "frame_log::hitch", "{}", rep.line());
    if let Ok(mut g) = LAST.lock() {
        *g = Some(rep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn top_n_picks_longest_in_order() {
        let mut t = PhaseTable::new();
        for (n, d) in [("a", 5), ("b", 40), ("c", 1), ("d", 22), ("e", 9), ("f", 30), ("g", 2)] {
            t.add(n, ms(d));
        }
        let (top, n) = t.top::<5>();
        let names: Vec<_> = top[..n].iter().map(|x| x.0).collect();
        assert_eq!(names, ["b", "f", "d", "e", "a"]);
    }

    #[test]
    fn top_n_with_fewer_entries_and_merge() {
        let mut t = PhaseTable::new();
        t.add("x", ms(3));
        t.add("y", ms(4));
        t.add("x", ms(3));
        let (top, n) = t.top::<5>();
        assert_eq!(n, 2);
        assert_eq!(top[0], ("x", ms(6)));
        assert_eq!(top[1], ("y", ms(4)));
        assert_eq!(t.total(), ms(10));
    }

    #[test]
    fn table_overflow_drops_instead_of_allocating() {
        let mut t = PhaseTable::new();
        let names: Vec<&'static str> = (0..60).map(|i| &*Box::leak(format!("p{i}").into_boxed_str())).collect();
        for n in &names {
            t.add(n, ms(1));
        }
        assert_eq!(t.total(), ms(MAX_ENTRIES as u64));
    }

    #[test]
    fn ring_keeps_last_32_and_filters_window() {
        let mut r = EventRing::new();
        for i in 0..40u64 {
            r.push(Event { at_us: i * 100_000, name: "e", tag: "", val: i });
        }
        assert_eq!(r.len(), MAX_EVENTS);
        let all = r.recent(4_000_000, u64::MAX);
        assert_eq!(all.len(), MAX_EVENTS);
        assert_eq!(all[0].val, 8);
        assert_eq!(all[MAX_EVENTS - 1].val, 39);
        // Last 2 s before t=3.9 s: events at >= 1.9 s => val 19..=39.
        let recent = r.recent(3_900_000, 2_000_000);
        assert_eq!(recent.first().map(|e| e.val), Some(19));
        assert_eq!(recent.len(), 21);
    }

    #[test]
    fn threshold_and_parse() {
        assert!(is_hitch(50.1, 50.0));
        assert!(!is_hitch(50.0, 50.0));
        assert_eq!(parse_threshold(None), DEFAULT_HITCH_MS);
        assert_eq!(parse_threshold(Some("120")), 120.0);
        assert_eq!(parse_threshold(Some("-3")), DEFAULT_HITCH_MS);
        assert_eq!(parse_threshold(Some("abc")), DEFAULT_HITCH_MS);
    }

    #[test]
    fn line_is_single_line_key_value() {
        let r = HitchReport {
            total_ms: 412.34,
            interval: Interval { total_ms: 412.34, cpu_ms: 30.0, acquire_ms: 1.0, present_ms: 381.3, acq_to_present_ms: Some(12.5), gpu_ms: None },
            suppressed: 2,
            top: vec![("wait:device_wait_idle@x", 380.2), ("render_record", 20.1)],
            events: vec![(0.41, "focus_gained".into())],
            counters: vec![("tex_upload", 4)],
        };
        let l = r.line();
        assert!(!l.contains('\n'));
        assert_eq!(
            l,
            "hitch total_ms=412.3 cpu_ms=30.0 acquire_ms=1.0 present_ms=381.3 acq_to_present_ms=12.5 gpu_ms=na suppressed=2 top=wait:device_wait_idle@x=380.2,render_record=20.1 counters=tex_upload=4 events=-0.41s:focus_gained"
        );
    }

    #[test]
    fn rate_limiter_caps_and_counts_suppressed() {
        let mut r = RateLimiter::new(5, 1_000_000);
        let passed: Vec<_> = (0..8).map(|i| r.try_pass(1_000 + i * 10_000)).collect();
        assert_eq!(passed.iter().filter(|p| p.is_some()).count(), 5);
        assert!(passed[5..].iter().all(|p| p.is_none()));
        // Next window: passes, reporting the 3 suppressed.
        assert_eq!(r.try_pass(1_200_000), Some(3));
        assert_eq!(r.try_pass(1_210_000), Some(0));
    }

    #[test]
    fn cpu_work_excludes_waits() {
        assert_eq!(cpu_work_ms(16.7, 0.2, 8.0), 8.5);
        assert_eq!(cpu_work_ms(5.0, 4.0, 4.0), 0.0);
        // The limiter wait is subtracted from the total before the vblank waits.
        assert!((cpu_work_ms(8.3 - 7.3, 0.0, 0.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn phase_agg_reports_the_top_phases_without_waits() {
        let mut t = PhaseTable::new();
        t.add("skinning_pose", Duration::from_millis(10));
        t.add("draw_prep", Duration::from_millis(3));
        t.add("present", Duration::from_millis(30));
        t.add("realtime_gi", Duration::from_millis(4));
        t.add("egui_run", Duration::from_millis(1));
        let mut a = PhaseAgg::default();
        a.record(&t);
        a.record(&t);
        assert_eq!(a.finish_top3(), " phases: skinning_pose=10.0 realtime_gi=4.0 draw_prep=3.0 egui_run=1.0");
        assert_eq!(a.finish_top3(), "");
    }

    #[test]
    fn work_agg_summary() {
        let mut a = WorkAgg::default();
        a.record(4.0, Some(2.0));
        a.record(6.0, Some(4.0));
        a.record(8.0, None);
        let s = a.finish();
        assert_eq!(s, WorkSummary { cpu_mean: 6.0, cpu_max: 8.0, gpu_mean: Some(3.0), gpu_max: Some(4.0) });
        assert_eq!(s.fields(), " cpu_work_mean=6.00ms cpu_work_max=8.00ms gpu_mean=3.00ms gpu_max=4.00ms");
        let s2 = a.finish();
        assert_eq!(s2.gpu_mean, None);
        assert_eq!(s2.fields(), " cpu_work_mean=0.00ms cpu_work_max=0.00ms");
    }

    #[test]
    fn exclusive_scope_time_and_phases() {
        // Runs on its own thread so the thread-local state is fresh.
        std::thread::spawn(|| {
            let _g = frame_begin();
            phase("p1");
            {
                let _s = scope("outer");
                std::thread::sleep(ms(6));
                {
                    let _i = scope("inner");
                    std::thread::sleep(ms(10));
                }
            }
            phase("p2");
            let (inner, outer, p1) = with_state(|st| {
                let get = |n: &str| (0..st.table.len).find(|&i| st.table.names[i] == n).map(|i| st.table.ns[i]).unwrap_or(0);
                (get("inner"), get("outer"), get("p1"))
            })
            .unwrap();
            assert!(inner >= 10_000_000);
            // outer excludes inner's 10 ms.
            assert!((6_000_000..10_000_000).contains(&outer), "outer={outer}");
            assert!(p1 < 5_000_000, "p1={p1}");
        })
        .join()
        .unwrap();
    }
}
