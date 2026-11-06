//! Fixed-step pacing: a monotonic accumulator, a catch-up cap, and a wait that
//! sleeps the bulk and spins only the last stretch.

use std::time::{Duration, Instant};

/// The last stretch of a wait that is spun instead of slept (OS sleeps overshoot by about this much).
const SPIN: Duration = Duration::from_micros(1500);
/// Longest single sleep, so a stop request is noticed promptly even at a low tick rate.
const MAX_SLEEP: Duration = Duration::from_millis(50);

/// What the accumulator decided for one wake-up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Frame {
    /// Ticks to run now (never more than the catch-up cap).
    pub ticks: u32,
    /// Whole ticks of backlog thrown away because of the cap (the server is overloaded).
    pub dropped: u64,
}

/// Turns wall-clock time into whole fixed ticks.
#[derive(Debug)]
pub struct Pacer {
    tick: Duration,
    max_catch_up: u32,
    acc: Duration,
    last: Instant,
}

impl Pacer {
    pub fn new(rate_hz: f32, max_catch_up: u32, now: Instant) -> Self {
        Self { tick: Duration::from_secs_f64(1.0 / f64::from(rate_hz.max(0.001))), max_catch_up: max_catch_up.max(1), acc: Duration::ZERO, last: now }
    }

    pub fn tick_len(&self) -> Duration {
        self.tick
    }

    /// Fold in the time since the last call. Past `max_catch_up` due ticks the
    /// rest of the backlog is dropped rather than replayed, so one long stall
    /// cannot snowball into a spiral of ever-longer frames.
    pub fn advance(&mut self, now: Instant) -> Frame {
        let elapsed = now.saturating_duration_since(self.last);
        self.last = now;
        self.advance_by(elapsed)
    }

    /// [`advance`](Self::advance) with the elapsed time given (deterministic tests).
    pub fn advance_by(&mut self, elapsed: Duration) -> Frame {
        self.acc += elapsed;
        let due = (self.acc.as_nanos() / self.tick.as_nanos().max(1)) as u64;
        let run = due.min(u64::from(self.max_catch_up));
        // All due ticks are consumed from the accumulator, the capped ones are simply not run.
        self.acc -= Duration::from_nanos((self.tick.as_nanos() as u64).saturating_mul(due));
        Frame { ticks: run as u32, dropped: due - run }
    }

    /// Time until the next tick is due.
    pub fn time_to_next(&self) -> Duration {
        self.tick.saturating_sub(self.acc)
    }

    /// Wait for the next tick: sleep the bulk (in short slices), spin the end.
    /// `interrupted` is polled between slices; returns early when it is true.
    pub fn wait(&self, mut interrupted: impl FnMut() -> bool) {
        // The accumulator is only updated at `advance`, so measure from "now".
        let deadline = Instant::now() + self.time_to_next();
        loop {
            if interrupted() {
                return;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            if left > SPIN {
                std::thread::sleep((left - SPIN).min(MAX_SLEEP));
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

/// What the loop drives.
pub trait TickBody {
    /// Run one fixed step of `dt` seconds.
    fn tick(&mut self, dt: f32);
    /// Checked every wake-up; true ends the loop.
    fn should_stop(&mut self) -> bool;
    /// Called once per frame before ticking (admin requests, stats).
    fn pre_frame(&mut self) {}
    /// A frame dropped `dropped` ticks of backlog.
    fn overloaded(&mut self, _dropped: u64) {}
}

/// Totals of one [`run_fixed`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopReport {
    pub ticks: u64,
    pub overloads: u64,
    pub dropped_ticks: u64,
}

/// Run `body` at `rate_hz` until it asks to stop.
pub fn run_fixed(body: &mut impl TickBody, rate_hz: f32, max_catch_up: u32) -> LoopReport {
    let mut pacer = Pacer::new(rate_hz, max_catch_up, Instant::now());
    let dt = pacer.tick_len().as_secs_f32();
    let mut report = LoopReport::default();
    while !body.should_stop() {
        body.pre_frame();
        let frame = pacer.advance(Instant::now());
        if frame.dropped > 0 {
            report.overloads += 1;
            report.dropped_ticks += frame.dropped;
            body.overloaded(frame.dropped);
        }
        for _ in 0..frame.ticks {
            body.tick(dt);
            report.ticks += 1;
            if body.should_stop() {
                return report;
            }
        }
        // A tick that took longer than its slot has already put the next one
        // past due: wait() then returns at once and the next advance() catches up.
        let mut stop = false;
        pacer.wait(|| {
            stop = body.should_stop();
            stop
        });
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn accumulator_yields_whole_ticks_and_keeps_the_remainder() {
        let mut p = Pacer::new(100.0, 5, Instant::now()); // 10 ms
        assert_eq!(p.advance_by(ms(4)), Frame { ticks: 0, dropped: 0 });
        assert_eq!(p.advance_by(ms(7)), Frame { ticks: 1, dropped: 0 });
        assert_eq!(p.time_to_next(), ms(9)); // 1 ms carried
        assert_eq!(p.advance_by(ms(29)), Frame { ticks: 3, dropped: 0 });
    }

    #[test]
    fn catch_up_is_capped_and_the_backlog_dropped() {
        let mut p = Pacer::new(100.0, 3, Instant::now());
        // A 500 ms stall owes 50 ticks: run 3, drop 47, and do not owe them later.
        assert_eq!(p.advance_by(ms(500)), Frame { ticks: 3, dropped: 47 });
        assert_eq!(p.advance_by(ms(10)), Frame { ticks: 1, dropped: 0 });
    }

    #[test]
    fn steady_state_runs_the_configured_rate() {
        let mut p = Pacer::new(30.0, 5, Instant::now());
        let total: u32 = (0..300).map(|_| p.advance_by(Duration::from_secs_f64(1.0 / 300.0)).ticks).sum();
        assert!((29..=30).contains(&total), "{total}");
    }

    struct Body {
        ticks: u64,
        heavy: u64,
        overloads: u64,
        started: Instant,
        run_for: Duration,
    }
    impl TickBody for Body {
        fn tick(&mut self, _dt: f32) {
            self.ticks += 1;
            // The first `heavy` ticks take 4x their slot.
            if self.ticks <= self.heavy {
                std::thread::sleep(ms(40));
            }
        }
        fn should_stop(&mut self) -> bool {
            self.started.elapsed() >= self.run_for
        }
        fn overloaded(&mut self, _dropped: u64) {
            self.overloads += 1;
        }
    }

    #[test]
    fn heavy_ticks_do_not_spiral_and_the_loop_recovers() {
        let mut b = Body { ticks: 0, heavy: 5, overloads: 0, started: Instant::now(), run_for: ms(900) };
        let r = run_fixed(&mut b, 100.0, 3); // 10 ms slots
        assert!(r.overloads > 0 && b.overloads == r.overloads, "overload was reported: {r:?}");
        assert!(r.dropped_ticks > 0, "{r:?}");
        // Never more ticks than wall time allows plus one capped burst, and the
        // light ticks afterwards ran near the configured rate (not stuck).
        let max = (0.9 * 100.0) as u64 + 3;
        assert!(r.ticks <= max, "{r:?}");
        assert!(r.ticks >= 50, "recovered after the heavy stretch: {r:?}");
    }

    #[test]
    fn light_load_holds_the_rate_without_busy_waiting() {
        let mut b = Body { ticks: 0, heavy: 0, overloads: 0, started: Instant::now(), run_for: ms(500) };
        let cpu0 = cpu_time();
        let r = run_fixed(&mut b, 50.0, 5);
        let cpu = cpu_time().saturating_sub(cpu0);
        assert_eq!(r.overloads, 0);
        assert!((23..=26).contains(&r.ticks), "{r:?}");
        assert!(cpu < ms(200), "mostly sleeping, used {cpu:?} cpu in 500 ms");
    }

    #[cfg(unix)]
    fn cpu_time() -> Duration {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: plain libc call with a valid out pointer.
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
    }
    #[cfg(not(unix))]
    fn cpu_time() -> Duration {
        Duration::ZERO
    }
}
