//! Frame-rate limiter for present modes that never block the CPU.
//!
//! MAILBOX and IMMEDIATE return from acquire/present at once, so the loop would
//! render at ~1000 fps. [`Pacer`] is the pure deadline math (a monotonic
//! deadline advancing one interval per frame, so no drift; a catch-up reset when
//! more than two intervals behind, so a long hitch is not followed by a burst of
//! uncapped frames). [`FrameLimiter`] adds the waiting (sleep for the bulk, spin
//! for the last [`SPIN_MARGIN`]) over an injectable [`Clock`]. [`resolve_policy`]
//! decides whether and at what rate to cap, and [`FramePacing`] ties it together
//! for the editor and the game loop (policy logging, monitor refresh, hitch scope).
//!
//! Environment: `ZESTENGINE_FPS_CAP=<number>` (0 = uncapped) applies in every
//! mode where the limiter can run; with `ZESTENGINE_PRESENT_MODE` forcing
//! MAILBOX / IMMEDIATE the default cap is the monitor refresh rate.

use std::time::{Duration, Instant};

use ash::vk;

/// The last part of a wait is spun instead of slept (OS sleeps overshoot).
pub const SPIN_MARGIN: Duration = Duration::from_millis(1);
/// Behind by more than this many intervals: drop the backlog and restart pacing.
const CATCH_UP_INTERVALS: u32 = 2;
/// Cap used when the monitor refresh rate is unknown.
pub const FALLBACK_HZ: f64 = 60.0;

/// Deadline bookkeeping; times are offsets from an arbitrary epoch.
#[derive(Debug, Clone)]
pub struct Pacer {
    interval: Duration,
    /// Earliest start of the next frame; `None` until the first frame.
    deadline: Option<Duration>,
}

impl Pacer {
    pub fn new(interval: Duration) -> Self {
        Self { interval, deadline: None }
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Change the interval; the pending deadline is kept, so the switch is
    /// seamless (the next deadline is the pending one plus the new interval).
    pub fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Forget the schedule (limiter was off, window hidden): the next frame
    /// starts immediately and pacing restarts from it.
    pub fn reset(&mut self) {
        self.deadline = None;
    }

    /// Time left until the next frame may start.
    pub fn remaining(&self, now: Duration) -> Duration {
        self.deadline.map_or(Duration::ZERO, |d| d.saturating_sub(now))
    }

    /// A frame starts at `now` (after waiting): advance the deadline one
    /// interval, or restart from `now` when more than two intervals behind.
    pub fn frame_started(&mut self, now: Duration) {
        let mut d = self.deadline.unwrap_or(now);
        if now > d + self.interval * CATCH_UP_INTERVALS {
            d = now;
        }
        self.deadline = Some(d + self.interval);
    }
}

/// Time source and sleeper, injectable for tests.
pub trait Clock {
    /// Monotonic time since an arbitrary epoch.
    fn now(&self) -> Duration;
    fn sleep(&self, d: Duration);
}

pub struct SystemClock {
    epoch: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self { epoch: Instant::now() }
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.epoch.elapsed()
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

pub struct FrameLimiter<C: Clock = SystemClock> {
    clock: C,
    pacer: Pacer,
}

impl<C: Clock> FrameLimiter<C> {
    pub fn with_clock(clock: C, interval: Duration) -> Self {
        Self { clock, pacer: Pacer::new(interval) }
    }

    pub fn set_interval(&mut self, interval: Duration) {
        self.pacer.set_interval(interval);
    }

    pub fn interval(&self) -> Duration {
        self.pacer.interval()
    }

    pub fn reset(&mut self) {
        self.pacer.reset();
    }

    pub fn remaining(&self) -> Duration {
        self.pacer.remaining(self.clock.now())
    }

    /// Block until the frame deadline (sleep the bulk, spin the last
    /// [`SPIN_MARGIN`]), then start the next interval. Returns the time waited.
    pub fn wait_until_deadline(&mut self) -> Duration {
        let start = self.clock.now();
        loop {
            let rem = self.pacer.remaining(self.clock.now());
            if rem.is_zero() {
                break;
            }
            if rem > SPIN_MARGIN {
                self.clock.sleep(rem - SPIN_MARGIN);
            } else {
                std::hint::spin_loop();
            }
        }
        let now = self.clock.now();
        self.pacer.frame_started(now);
        now.saturating_sub(start)
    }
}

/// Where the active cap came from (logged).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapSource {
    /// Present mode paces itself (FIFO family): limiter not running.
    Paced,
    /// `ZESTENGINE_FPS_CAP`.
    Env,
    /// Monitor refresh rate.
    MonitorRefresh,
    /// Monitor refresh rate unknown: [`FALLBACK_HZ`].
    Fallback,
    /// No cap by intent (vsync-off setting, or `ZESTENGINE_FPS_CAP=0`).
    Uncapped,
}

impl CapSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Paced => "present mode paces itself",
            Self::Env => "env ZESTENGINE_FPS_CAP",
            Self::MonitorRefresh => "monitor refresh",
            Self::Fallback => "fallback, monitor refresh unknown",
            Self::Uncapped => "uncapped",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// `None` = the limiter does not wait.
    pub cap_hz: Option<f64>,
    pub source: CapSource,
}

/// MAILBOX / IMMEDIATE: the modes that never block the CPU.
pub fn is_unpaced(mode: vk::PresentModeKHR) -> bool {
    matches!(mode, vk::PresentModeKHR::MAILBOX | vk::PresentModeKHR::IMMEDIATE)
}

/// Parse `ZESTENGINE_FPS_CAP`: `Some(0.0)` = uncapped, `None` = unset/invalid.
pub fn parse_fps_cap(v: Option<&str>) -> Option<f64> {
    v.and_then(|s| s.trim().parse::<f64>().ok()).filter(|x| x.is_finite() && *x >= 0.0)
}

/// Decide the cap. `mode_forced`: the mode was chosen by `ZESTENGINE_PRESENT_MODE`
/// (default cap = monitor refresh); otherwise it came from the vsync-off
/// setting, which stays uncapped unless `env_cap` is set.
pub fn resolve_policy(unpaced: bool, mode_forced: bool, env_cap: Option<f64>, monitor_hz: Option<f64>) -> Policy {
    if !unpaced {
        return Policy { cap_hz: None, source: CapSource::Paced };
    }
    match env_cap {
        Some(c) if c > 0.0 => Policy { cap_hz: Some(c), source: CapSource::Env },
        Some(_) => Policy { cap_hz: None, source: CapSource::Uncapped },
        None if mode_forced => match monitor_hz.filter(|h| h.is_finite() && *h > 0.0) {
            Some(h) => Policy { cap_hz: Some(h), source: CapSource::MonitorRefresh },
            None => Policy { cap_hz: Some(FALLBACK_HZ), source: CapSource::Fallback },
        },
        None => Policy { cap_hz: None, source: CapSource::Uncapped },
    }
}

/// Refresh rate (Hz) of the monitor the window is on, if the platform reports it.
pub fn monitor_refresh_hz(window: &winit::window::Window) -> Option<f64> {
    window.current_monitor().and_then(|m| m.refresh_rate_millihertz()).map(|mhz| mhz as f64 / 1000.0)
}

/// Limiter + policy for one main window. Call [`Self::update`] every frame
/// (cheap), [`Self::refresh_monitor`] when the window moves / rescales.
pub struct FramePacing {
    limiter: FrameLimiter,
    policy: Policy,
    mode: vk::PresentModeKHR,
    mode_forced: bool,
    env_cap: Option<f64>,
    monitor_hz: Option<f64>,
    monitor_known: bool,
    occluded: bool,
    suspended: bool,
    logged: Option<(vk::PresentModeKHR, Policy)>,
}

impl Default for FramePacing {
    fn default() -> Self {
        Self::new()
    }
}

impl FramePacing {
    pub fn new() -> Self {
        let env_cap = parse_fps_cap(std::env::var("ZESTENGINE_FPS_CAP").ok().as_deref());
        Self {
            limiter: FrameLimiter::with_clock(SystemClock::default(), Duration::from_secs_f64(1.0 / FALLBACK_HZ)),
            policy: Policy { cap_hz: None, source: CapSource::Paced },
            mode: vk::PresentModeKHR::FIFO,
            mode_forced: std::env::var_os("ZESTENGINE_PRESENT_MODE").is_some(),
            env_cap,
            monitor_hz: None,
            monitor_known: false,
            occluded: false,
            suspended: false,
            logged: None,
        }
    }

    /// The rate the frame time budget aims for: the limiter's cap, else the
    /// monitor refresh rate (None while neither is known).
    pub fn target_hz(&self) -> Option<f64> {
        self.policy.cap_hz.or(self.monitor_hz).filter(|h| h.is_finite() && *h > 1.0)
    }

    /// Re-read the monitor refresh rate (window moved, scale / monitor change).
    pub fn refresh_monitor(&mut self, window: &winit::window::Window) {
        self.monitor_hz = monitor_refresh_hz(window);
        self.monitor_known = true;
    }

    /// Window hidden (`WindowEvent::Occluded`): the limiter stops waiting.
    pub fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }

    /// Per-frame hold set by the app (minimised, blocking-job overlay, headset
    /// pacing): the limiter stops waiting while true.
    pub fn set_suspended(&mut self, suspended: bool) {
        self.suspended = suspended;
    }

    /// Recompute the policy for the swapchain's present mode; logs one INFO
    /// line when mode or policy changed (and at startup).
    pub fn update(&mut self, window: &winit::window::Window, mode: vk::PresentModeKHR) {
        if !self.monitor_known {
            self.refresh_monitor(window);
        }
        let policy = resolve_policy(is_unpaced(mode), self.mode_forced, self.env_cap, self.monitor_hz);
        if policy != self.policy {
            match policy.cap_hz {
                Some(hz) => self.limiter.set_interval(Duration::from_secs_f64(1.0 / hz)),
                None => self.limiter.reset(),
            }
            self.policy = policy;
        }
        self.mode = mode;
        if self.logged != Some((mode, policy)) {
            self.logged = Some((mode, policy));
            let cap = policy.cap_hz.map_or("none".to_string(), |h| format!("{h:.1} Hz"));
            tracing::info!(
                "frame limiter: present_mode={} cap={cap} source={}",
                crate::context::present_mode_name(mode),
                policy.source.label()
            );
        }
    }

    /// True when the limiter should pace frames right now.
    pub fn active(&self) -> bool {
        self.policy.cap_hz.is_some() && !self.occluded && !self.suspended
    }

    /// How long the event loop may sleep (`ControlFlow::WaitUntil`) before the
    /// next redraw is due, leaving [`SPIN_MARGIN`] for the precise wait in
    /// [`Self::wait`]. `None` = redraw now (or the limiter is inactive).
    pub fn sleep_before_redraw(&self) -> Option<Duration> {
        if !self.active() {
            return None;
        }
        self.limiter.remaining().checked_sub(SPIN_MARGIN).filter(|d| !d.is_zero())
    }

    /// Wait for the frame deadline inside the `frame_limiter` hitch scope. Call
    /// right after `hitch::frame_begin`, before acquiring the swapchain image.
    pub fn wait(&mut self) {
        if !self.active() {
            return;
        }
        let _h = crate::hitch::scope("frame_limiter");
        self.limiter.wait_until_deadline();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn steady_state_advances_one_interval_without_drift() {
        let mut p = Pacer::new(ms(10));
        p.frame_started(ms(0));
        assert_eq!(p.remaining(ms(3)), ms(7));
        // Frames start a little late each time; the schedule does not slip.
        for k in 1..=50u64 {
            p.frame_started(ms(k * 10) + ms(k % 3));
        }
        assert_eq!(p.remaining(ms(500)), ms(10));
    }

    #[test]
    fn no_wait_before_first_frame() {
        let p = Pacer::new(ms(10));
        assert_eq!(p.remaining(ms(123)), Duration::ZERO);
    }

    #[test]
    fn small_overrun_catches_up_with_short_frames() {
        let mut p = Pacer::new(ms(10));
        p.frame_started(ms(0)); // next deadline 10
        p.frame_started(ms(25)); // 1.5 intervals late: not reset
        assert_eq!(p.remaining(ms(25)), Duration::ZERO); // deadline 20 already passed
        p.frame_started(ms(25));
        assert_eq!(p.remaining(ms(25)), ms(5)); // back on the 10 ms grid (30)
    }

    #[test]
    fn long_hitch_resets_instead_of_bursting() {
        let mut p = Pacer::new(ms(10));
        p.frame_started(ms(0));
        p.frame_started(ms(500)); // 490 ms behind
        assert_eq!(p.remaining(ms(500)), ms(10));
    }

    #[test]
    fn interval_change_keeps_schedule() {
        let mut p = Pacer::new(ms(10));
        p.frame_started(ms(0)); // deadline 10
        p.set_interval(ms(20));
        assert_eq!(p.remaining(ms(0)), ms(10));
        p.frame_started(ms(10)); // deadline 30
        assert_eq!(p.remaining(ms(10)), ms(20));
        p.set_interval(ms(5));
        p.frame_started(ms(30)); // deadline 35
        assert_eq!(p.remaining(ms(30)), ms(5));
    }

    #[test]
    fn reset_restarts_pacing() {
        let mut p = Pacer::new(ms(10));
        p.frame_started(ms(0));
        p.reset();
        assert_eq!(p.remaining(ms(1)), Duration::ZERO);
    }

    /// Fake clock: sleeping advances time by the request plus an overshoot.
    struct Fake {
        t: Cell<Duration>,
        overshoot: Duration,
        sleeps: Cell<u32>,
    }
    impl Clock for &Fake {
        fn now(&self) -> Duration {
            // Spinning must make progress too.
            self.t.set(self.t.get() + Duration::from_micros(10));
            self.t.get()
        }
        fn sleep(&self, d: Duration) {
            self.sleeps.set(self.sleeps.get() + 1);
            self.t.set(self.t.get() + d + self.overshoot);
        }
    }

    #[test]
    fn wait_sleeps_bulk_then_spins_and_lands_on_the_deadline() {
        let fake = Fake { t: Cell::new(Duration::ZERO), overshoot: Duration::from_micros(100), sleeps: Cell::new(0) };
        let mut l = FrameLimiter::with_clock(&fake, ms(10));
        l.wait_until_deadline(); // first frame: no wait
        assert_eq!(fake.sleeps.get(), 0);
        let start = fake.t.get();
        let waited = l.wait_until_deadline();
        assert_eq!(fake.sleeps.get(), 1, "one coarse sleep, rest spun");
        assert!(waited >= ms(9) && waited <= ms(10), "{waited:?}");
        assert!(fake.t.get() >= start + ms(9));
        // Third frame lands one more interval later (no drift).
        let before = fake.t.get();
        l.wait_until_deadline();
        assert!(fake.t.get() - before <= ms(10));
    }

    #[test]
    fn policy_table() {
        let p = |u, f, e, m| resolve_policy(u, f, e, m);
        // FIFO family: never runs, even with an env cap.
        assert_eq!(p(false, true, Some(30.0), Some(120.0)).cap_hz, None);
        assert_eq!(p(false, false, None, None).source, CapSource::Paced);
        // Forced MAILBOX: monitor refresh, 60 Hz fallback, env override, 0 = off.
        assert_eq!(p(true, true, None, Some(119.88)), Policy { cap_hz: Some(119.88), source: CapSource::MonitorRefresh });
        assert_eq!(p(true, true, None, None), Policy { cap_hz: Some(60.0), source: CapSource::Fallback });
        assert_eq!(p(true, true, Some(144.0), Some(120.0)), Policy { cap_hz: Some(144.0), source: CapSource::Env });
        assert_eq!(p(true, true, Some(0.0), Some(120.0)), Policy { cap_hz: None, source: CapSource::Uncapped });
        // Vsync-off setting: uncapped unless the env says otherwise.
        assert_eq!(p(true, false, None, Some(120.0)), Policy { cap_hz: None, source: CapSource::Uncapped });
        assert_eq!(p(true, false, Some(90.0), None), Policy { cap_hz: Some(90.0), source: CapSource::Env });
    }

    #[test]
    fn unpaced_modes_and_env_parse() {
        assert!(is_unpaced(vk::PresentModeKHR::MAILBOX));
        assert!(is_unpaced(vk::PresentModeKHR::IMMEDIATE));
        assert!(!is_unpaced(vk::PresentModeKHR::FIFO));
        assert!(!is_unpaced(vk::PresentModeKHR::FIFO_RELAXED));
        assert!(!is_unpaced(crate::context::PRESENT_MODE_FIFO_LATEST_READY));
        assert_eq!(parse_fps_cap(Some(" 120 ")), Some(120.0));
        assert_eq!(parse_fps_cap(Some("0")), Some(0.0));
        assert_eq!(parse_fps_cap(Some("-5")), None);
        assert_eq!(parse_fps_cap(Some("x")), None);
        assert_eq!(parse_fps_cap(None), None);
    }
}
