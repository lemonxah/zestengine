//! Coalescing of window resizes into at most one swapchain rebuild per frame.
//!
//! A compositor animation / tiling reflow delivers dozens of `Resized` events
//! in a fraction of a second. Events only record that the size may have
//! changed; the frame compares the window's *current* size with the size the
//! swapchain was built for, so the latest size is applied once per frame and a
//! `Resized` that did not change the size (or a burst that ended where it
//! began) rebuilds nothing. Out-of-date / suboptimal / vsync changes force a
//! rebuild at the same size. Pure logic, no GPU.

#[derive(Debug, Default)]
pub struct ResizeGate {
    /// Size the swapchain was last built for.
    applied: (u32, u32),
    /// Rebuild needed at any size (out of date, suboptimal, present mode change).
    forced: bool,
    /// The swapchain was rebuilt in this frame's prep; the editor holds its
    /// viewport target back (it would resize in lock step otherwise).
    rebuilt_this_frame: bool,
}

impl ResizeGate {
    pub fn new(size: (u32, u32)) -> Self {
        Self { applied: size, forced: false, rebuilt_this_frame: false }
    }

    /// Rebuild at the next frame whatever the size.
    pub fn force(&mut self) {
        self.forced = true;
    }

    /// Start of a frame with the window at `size`: true when the swapchain must
    /// be rebuilt now (size differs from the built one, or forced).
    pub fn begin_frame(&mut self, size: (u32, u32)) -> bool {
        self.rebuilt_this_frame = false;
        self.forced || self.applied != size
    }

    /// The swapchain was rebuilt for `size`.
    pub fn rebuilt(&mut self, size: (u32, u32)) {
        self.applied = size;
        self.forced = false;
        self.rebuilt_this_frame = true;
    }

    pub fn rebuilt_this_frame(&self) -> bool {
        self.rebuilt_this_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_size_does_not_rebuild() {
        let mut g = ResizeGate::new((800, 600));
        assert!(!g.begin_frame((800, 600)));
    }

    #[test]
    fn a_burst_of_sizes_rebuilds_once_at_the_latest() {
        let mut g = ResizeGate::new((800, 600));
        // Many Resized events between frames change nothing here; only the
        // size seen at the frame start matters.
        assert!(g.begin_frame((1013, 600)));
        g.rebuilt((1013, 600));
        assert!(g.rebuilt_this_frame());
        assert!(!g.begin_frame((1013, 600)));
        assert!(!g.rebuilt_this_frame());
    }

    #[test]
    fn a_burst_that_ends_where_it_began_rebuilds_nothing() {
        let mut g = ResizeGate::new((800, 600));
        assert!(!g.begin_frame((800, 600)));
    }

    #[test]
    fn forced_rebuilds_once_at_the_same_size() {
        let mut g = ResizeGate::new((800, 600));
        g.force();
        g.force();
        assert!(g.begin_frame((800, 600)));
        g.rebuilt((800, 600));
        assert!(!g.begin_frame((800, 600)));
    }
}
