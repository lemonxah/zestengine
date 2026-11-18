use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootStep {
    Holding { done: usize, total: usize },
    Watching,
    Released,
}

pub struct BootHold {
    started: Option<Instant>,
    base: usize,
    frames: u32,
    quiet: Option<(Instant, u32)>,
    progress: (usize, Instant),
    lazy: bool,
    engaged: bool,
}

impl BootHold {
    pub const QUIET_FRAMES: u32 = 12;
    pub const QUIET: Duration = Duration::from_millis(300);
    pub const STALLED: Duration = Duration::from_secs(15);
    pub const LIMIT: Duration = Duration::from_secs(90);

    pub fn from_env() -> Option<Self> {
        std::env::var_os("ZESTENGINE_NO_BOOT_HOLD").is_none().then(Self::new)
    }

    pub fn new() -> Self {
        let now = Instant::now();
        Self { started: None, base: 0, frames: 0, quiet: None, progress: (0, now), lazy: false, engaged: false }
    }

    pub fn scene_from_env() -> Option<Self> {
        Self::from_env().map(|hold| Self { lazy: true, ..hold })
    }

    pub fn for_scene() -> Self {
        Self { lazy: true, ..Self::new() }
    }

    pub fn step(&mut self, pending: usize, finished: usize, now: Instant) -> BootStep {
        let started = *self.started.get_or_insert_with(|| {
            self.base = finished;
            self.progress = (finished, now);
            now
        });
        self.frames += 1;
        if self.progress.0 != finished {
            self.progress = (finished, now);
        }
        let mut started = started;
        if pending > 0 && !self.engaged {
            self.engaged = true;
            if self.lazy {
                self.started = Some(now);
                self.base = finished;
                self.progress = (finished, now);
                started = now;
            }
        }
        let done = finished.saturating_sub(self.base);
        if pending > 0 {
            self.quiet = None;
            let stalled = now.duration_since(self.progress.1) >= Self::STALLED;
            let too_long = now.duration_since(started) >= Self::LIMIT;
            if stalled || too_long {
                tracing::warn!(pending, waited = ?now.duration_since(started), "boot shader hold released while shaders are still compiling");
                return BootStep::Released;
            }
            return BootStep::Holding { done, total: done + pending };
        }
        let (since, frames) = self.quiet.get_or_insert((now, 0));
        *frames += 1;
        if *frames >= Self::QUIET_FRAMES && now.duration_since(*since) >= Self::QUIET {
            if !self.lazy || self.engaged {
                tracing::info!(waited = ?now.duration_since(started), frames = self.frames, shaders = done, scene = self.lazy, "shader hold finished");
            }
            return BootStep::Released;
        }
        if self.lazy && !self.engaged {
            return BootStep::Watching;
        }
        BootStep::Holding { done, total: done }
    }
}

impl Default for BootHold {
    fn default() -> Self {
        Self::new()
    }
}

pub fn holding_label(done: usize, total: usize) -> String {
    if total == 0 { "Loading".to_string() } else { format!("Preparing shaders {done}/{total}") }
}

pub fn holding_title(title: &str, done: usize, total: usize) -> String {
    if total == 0 { format!("{title} - loading") } else { format!("{title} - preparing shaders {done}/{total}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_game_is_held_until_the_compiler_has_been_quiet_for_a_moment() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut hold = BootHold::new();
        assert_eq!(hold.step(0, 7, ms(0)), BootStep::Holding { done: 0, total: 0 }, "the first frames have asked for nothing yet");
        assert_eq!(hold.step(3, 7, ms(16)), BootStep::Holding { done: 0, total: 3 }, "shaders finished before the hold are not counted");
        assert_eq!(hold.step(1, 9, ms(4000)), BootStep::Holding { done: 2, total: 3 });
        let mut now = 9000;
        let mut last = hold.step(0, 10, ms(now));
        assert_eq!(last, BootStep::Holding { done: 3, total: 3 }, "one quiet frame is not enough");
        for _ in 0..BootHold::QUIET_FRAMES {
            now += 30;
            last = hold.step(0, 10, ms(now));
        }
        assert_eq!(last, BootStep::Released);
    }

    #[test]
    fn a_new_request_restarts_the_quiet_wait_and_a_warm_cache_releases_within_a_second() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut hold = BootHold::new();
        for frame in 0..8 {
            assert!(matches!(hold.step(0, 0, ms(frame * 16)), BootStep::Holding { .. }));
        }
        assert_eq!(hold.step(2, 0, ms(140)), BootStep::Holding { done: 0, total: 2 });
        let mut released_at = None;
        for frame in 0..60u64 {
            let now = 200 + frame * 16;
            if hold.step(0, 2, ms(now)) == BootStep::Released {
                released_at = Some(now);
                break;
            }
        }
        let released_at = released_at.expect("released");
        assert!((500..1000).contains(&released_at), "released {released_at} ms after start");
    }

    #[test]
    fn a_stuck_or_endless_compile_does_not_hold_the_game_for_ever() {
        let t0 = Instant::now();
        let at = |d: Duration| t0 + d;
        let mut stuck = BootHold::new();
        assert!(matches!(stuck.step(1, 0, at(Duration::ZERO)), BootStep::Holding { .. }));
        assert!(matches!(stuck.step(1, 0, at(BootHold::STALLED - Duration::from_millis(1))), BootStep::Holding { .. }));
        assert_eq!(stuck.step(1, 0, at(BootHold::STALLED)), BootStep::Released, "no shader finished for the stall time");
        let mut endless = BootHold::new();
        let mut finished = 0;
        let mut now = Duration::ZERO;
        let mut last = BootStep::Released;
        while now <= BootHold::LIMIT {
            last = endless.step(1, finished, at(now));
            finished += 1;
            now += Duration::from_secs(1);
        }
        assert_eq!(last, BootStep::Released, "steady progress still ends at the limit");
        assert_eq!(holding_title("Game", 2, 5), "Game - preparing shaders 2/5");
        assert_eq!(holding_title("Game", 0, 0), "Game - loading");
        assert_eq!(holding_label(2, 5), "Preparing shaders 2/5");
        assert_eq!(holding_label(0, 0), "Loading");
    }

    #[test]
    fn a_scene_hold_only_pauses_play_once_a_shader_is_actually_compiling() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut warm = BootHold::for_scene();
        let mut now = 0;
        let mut last = warm.step(0, 40, ms(now));
        assert_eq!(last, BootStep::Watching, "a warm scene keeps playing while the hold watches");
        while last == BootStep::Watching {
            now += 16;
            last = warm.step(0, 40, ms(now));
        }
        assert_eq!(last, BootStep::Released);
        assert!((300..400).contains(&now), "the watch ends after the quiet wait: {now} ms");

        let mut cold = BootHold::for_scene();
        assert_eq!(cold.step(0, 40, ms(0)), BootStep::Watching);
        assert_eq!(cold.step(0, 40, ms(100)), BootStep::Watching);
        assert_eq!(cold.step(2, 40, ms(116)), BootStep::Holding { done: 0, total: 2 }, "a request pauses play");
        assert_eq!(cold.step(1, 41, ms(132)), BootStep::Holding { done: 1, total: 2 });
        let mut now = 148;
        let mut last = cold.step(0, 42, ms(now));
        assert_eq!(last, BootStep::Holding { done: 2, total: 2 }, "finished shaders hold until the compiler is quiet");
        while last != BootStep::Released {
            now += 16;
            last = cold.step(0, 42, ms(now));
            assert!(now < 1000, "the scene hold never released");
        }
    }

    #[test]
    fn a_scene_hold_measures_its_limit_from_the_first_request() {
        let t0 = Instant::now();
        let at = |d: Duration| t0 + d;
        let mut hold = BootHold::for_scene();
        assert_eq!(hold.step(0, 0, at(Duration::ZERO)), BootStep::Watching);
        let first = Duration::from_millis(250);
        assert!(matches!(hold.step(1, 0, at(first)), BootStep::Holding { .. }));
        assert!(matches!(hold.step(1, 0, at(first + BootHold::STALLED - Duration::from_millis(1))), BootStep::Holding { .. }));
        assert_eq!(hold.step(1, 0, at(first + BootHold::STALLED)), BootStep::Released);
    }
}
