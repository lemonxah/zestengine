//! Debounce of the editor viewport render-target size.
//!
//! Dragging a dock border changes the viewport rect every frame; recreating the
//! HDR/depth/GI targets each time is a storm of allocations. While the pointer
//! is held the target keeps its old size (the image is shown stretched into the
//! new rect) and is recreated once the wanted size has been stable for
//! [`SETTLE`], or at once when the pointer is released. Pure logic, no clock:
//! the caller passes the time.

/// How long the wanted size must stay unchanged while dragging before the
/// target is recreated.
pub const SETTLE: f64 = 0.1;

#[derive(Default)]
pub struct Debounce {
    applied: Option<[u32; 2]>,
    pending: [u32; 2],
    since: f64,
}

impl Debounce {
    /// The size the target should have now. `now` is in seconds, `held` is
    /// true while a pointer button is down (a drag may be in progress).
    pub fn resolve(&mut self, now: f64, wanted: [u32; 2], held: bool) -> [u32; 2] {
        let Some(applied) = self.applied else {
            self.applied = Some(wanted);
            self.pending = wanted;
            return wanted;
        };
        if wanted == applied {
            self.pending = wanted;
            return applied;
        }
        if wanted != self.pending {
            self.pending = wanted;
            self.since = now;
        }
        if !held || now - self.since >= SETTLE {
            self.applied = Some(wanted);
            return wanted;
        }
        applied
    }

    /// The size currently applied (None before the first frame).
    #[cfg(test)]
    pub fn applied(&self) -> Option<[u32; 2]> {
        self.applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_size_applies_immediately() {
        let mut d = Debounce::default();
        assert_eq!(d.resolve(0.0, [800, 600], true), [800, 600]);
    }

    #[test]
    fn a_drag_keeps_the_old_size_until_it_settles() {
        let mut d = Debounce::default();
        d.resolve(0.0, [800, 600], false);
        // 30 frames of a border drag, each a new size, 10 ms apart.
        let mut recreated = 0;
        let mut last = [800, 600];
        for i in 0..30 {
            let now = 1.0 + i as f64 * 0.01;
            let got = d.resolve(now, [800 + i, 600], true);
            if got != last {
                recreated += 1;
                last = got;
            }
        }
        assert_eq!(recreated, 0, "no recreate while the size keeps changing");
        // Held still: settles after 100 ms.
        assert_eq!(d.resolve(1.29 + 0.05, [829, 600], true), [800, 600]);
        assert_eq!(d.resolve(1.29 + 0.11, [829, 600], true), [829, 600]);
    }

    #[test]
    fn releasing_the_pointer_applies_at_once() {
        let mut d = Debounce::default();
        d.resolve(0.0, [800, 600], false);
        assert_eq!(d.resolve(0.01, [900, 600], true), [800, 600]);
        assert_eq!(d.resolve(0.02, [905, 600], false), [905, 600]);
    }

    #[test]
    fn returning_to_the_applied_size_cancels_the_pending_change() {
        let mut d = Debounce::default();
        d.resolve(0.0, [800, 600], false);
        d.resolve(0.01, [900, 600], true);
        assert_eq!(d.resolve(0.5, [800, 600], true), [800, 600]);
        assert_eq!(d.applied(), Some([800, 600]));
    }
}
