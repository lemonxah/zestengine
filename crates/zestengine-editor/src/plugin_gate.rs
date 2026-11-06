//! State machine for the background component-plugin build.
//!
//! `cargo build` of `plugins/` takes tens of seconds; it runs on a worker
//! (see `EngineApp::request_plugin_build`) while the UI stays alive. This is the
//! pure bookkeeping: at most one build runs at a time, a request made while one
//! runs is queued (the running build may have started before the latest source
//! edit, so one more build follows), and work that needs the plugin's components
//! (a scene install, entering Play) parks as a waiter until the last build
//! finished. No threads, no clock.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildState {
    Idle,
    Building,
    Ready,
    Failed,
}

/// What a finished build leaves to do.
#[derive(Debug, PartialEq)]
pub enum Finished<W> {
    /// A rebuild was queued during the build: start it, keep the waiters.
    Restart,
    /// Nothing queued: this build is the final one; run the waiters.
    Done { ok: bool, waiters: Vec<W> },
}

#[derive(Debug)]
pub struct BuildGate<W> {
    state: BuildState,
    queued: bool,
    waiters: Vec<W>,
}

impl<W> Default for BuildGate<W> {
    fn default() -> Self {
        Self { state: BuildState::Idle, queued: false, waiters: Vec::new() }
    }
}

impl<W> BuildGate<W> {
    #[allow(dead_code)] // state getter used by the gate UI paths
    pub fn state(&self) -> BuildState {
        self.state
    }

    pub fn is_building(&self) -> bool {
        self.state == BuildState::Building
    }

    /// A rebuild is queued behind the running build (its result is stale).
    pub fn rebuild_queued(&self) -> bool {
        self.queued
    }

    /// A build was asked for. True: the caller starts the worker now. False: one
    /// is already running; a rebuild is queued behind it.
    pub fn request(&mut self) -> bool {
        if self.is_building() {
            self.queued = true;
            false
        } else {
            self.state = BuildState::Building;
            true
        }
    }

    /// Work that needs the plugin components. Returns it back when no build is
    /// running (run it now), else parks it until the build finishes.
    pub fn wait_or_run(&mut self, work: W) -> Option<W> {
        if self.is_building() {
            self.waiters.push(work);
            None
        } else {
            Some(work)
        }
    }

    /// The worker finished (`ok` = built and loadable).
    pub fn finish(&mut self, ok: bool) -> Finished<W> {
        if self.queued {
            self.queued = false;
            self.state = BuildState::Building;
            return Finished::Restart;
        }
        self.state = if ok { BuildState::Ready } else { BuildState::Failed };
        Finished::Done { ok, waiters: std::mem::take(&mut self.waiters) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_to_building_to_ready() {
        let mut g = BuildGate::<u32>::default();
        assert_eq!(g.state(), BuildState::Idle);
        assert!(g.request());
        assert!(g.is_building());
        assert_eq!(g.finish(true), Finished::Done { ok: true, waiters: vec![] });
        assert_eq!(g.state(), BuildState::Ready);
    }

    #[test]
    fn failure_is_recorded_and_a_new_build_can_start() {
        let mut g = BuildGate::<u32>::default();
        g.request();
        assert_eq!(g.finish(false), Finished::Done { ok: false, waiters: vec![] });
        assert_eq!(g.state(), BuildState::Failed);
        assert!(g.request());
    }

    #[test]
    fn never_two_concurrent_builds() {
        let mut g = BuildGate::<u32>::default();
        assert!(g.request());
        assert!(!g.request());
        assert!(!g.request());
    }

    #[test]
    fn requests_during_a_build_queue_exactly_one_rebuild() {
        let mut g = BuildGate::<u32>::default();
        g.request();
        g.request();
        g.request();
        assert_eq!(g.finish(true), Finished::Restart);
        assert!(g.is_building());
        assert_eq!(g.finish(true), Finished::Done { ok: true, waiters: vec![] });
    }

    #[test]
    fn waiters_run_only_after_the_final_build() {
        let mut g = BuildGate::<&str>::default();
        assert_eq!(g.wait_or_run("free"), Some("free"));
        g.request();
        assert_eq!(g.wait_or_run("scene"), None);
        g.request(); // sources changed again
        assert_eq!(g.finish(true), Finished::Restart);
        assert_eq!(g.wait_or_run("play"), None);
        assert_eq!(g.finish(true), Finished::Done { ok: true, waiters: vec!["scene", "play"] });
        assert_eq!(g.wait_or_run("later"), Some("later"));
    }

    #[test]
    fn waiters_are_released_on_failure_too() {
        let mut g = BuildGate::<u8>::default();
        g.request();
        g.wait_or_run(1);
        assert_eq!(g.finish(false), Finished::Done { ok: false, waiters: vec![1] });
    }
}
