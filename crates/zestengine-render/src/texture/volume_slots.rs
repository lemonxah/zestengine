//! Rectangular atlas reservations. A cache entry and every GPU snapshot retain
//! the same lease. Evicting the cache entry cannot free storage still in flight.
//! Splits form a tree so retiring all children always recovers their parent,
//! without relying on an order-dependent greedy rectangle merge.
use std::sync::{Arc, Weak};

#[derive(Clone, Copy, Debug)]
struct Region { origin: [u32; 3], size: [u32; 3] }
impl Region {
    fn volume(self) -> u64 { self.size.iter().map(|&n| n as u64).product() }
    fn fits(self, size: [u32; 3]) -> bool { (0..3).all(|a| size[a] <= self.size[a]) }
}

#[derive(Debug)]
pub(crate) struct VolumeSlot { region: Region }
impl VolumeSlot {
    pub fn origin(&self) -> [u32; 3] { self.region.origin }
    pub fn size(&self) -> [u32; 3] { self.region.size }
}

enum State { Free, Reserved(Weak<VolumeSlot>), Split(Vec<Node>) }
struct Node { region: Region, state: State }
impl Node {
    fn collect(&mut self) {
        let free = match &mut self.state {
            State::Free => false,
            State::Reserved(lease) => lease.strong_count() == 0,
            State::Split(children) => {
                for child in children.iter_mut() { child.collect(); }
                children.iter().all(|c| matches!(c.state, State::Free))
            }
        };
        if free { self.state = State::Free; }
    }

    fn best(&self, size: [u32; 3], path: &mut Vec<usize>, best: &mut Option<(u64, Vec<usize>)>) {
        if !self.region.fits(size) { return; }
        match &self.state {
            State::Free => {
                let volume = self.region.volume();
                if best.as_ref().is_none_or(|(v, _)| volume < *v) { *best = Some((volume, path.clone())); }
            }
            State::Reserved(_) => (),
            State::Split(children) => {
                for (i, child) in children.iter().enumerate() {
                    path.push(i); child.best(size, path, best); path.pop();
                }
            }
        }
    }

    fn reserve(&mut self, path: &[usize], size: [u32; 3]) -> Arc<VolumeSlot> {
        if let Some((&i, rest)) = path.split_first() {
            let State::Split(children) = &mut self.state else { unreachable!() };
            return children[i].reserve(rest, size);
        }
        debug_assert!(matches!(self.state, State::Free));
        let lease = Arc::new(VolumeSlot { region: Region { origin: self.region.origin, size } });
        if size == self.region.size {
            self.state = State::Reserved(Arc::downgrade(&lease));
        } else {
            let mut children = vec![Node { region: lease.region, state: State::Reserved(Arc::downgrade(&lease)) }];
            let mut remaining = self.region;
            // Disjoint complement: each slab spans the unsplit dimensions,
            // while previously split axes are restricted to the requested size.
            for (axis, axis_value) in size.iter().enumerate() {
                if remaining.size[axis] > *axis_value {
                    let mut slab = remaining;
                    slab.origin[axis] += *axis_value; slab.size[axis] -= *axis_value;
                    children.push(Node { region: slab, state: State::Free });
                    remaining.size[axis] = *axis_value;
                }
            }
            self.state = State::Split(children);
        }
        lease
    }
}

pub(crate) struct VolumeSlots { root: Node }
impl VolumeSlots {
    /// Atlas coordinates are packed as ten bits per axis by mesh-field shaders.
    pub fn new(dims: [u32; 3]) -> anyhow::Result<Self> {
        anyhow::ensure!(dims.iter().all(|&n| n > 0 && n <= 1024), "invalid mesh atlas extent");
        Ok(Self { root: Node { region: Region { origin: [0; 3], size: dims }, state: State::Free } })
    }

    /// Includes the interpolation border in `size`. None means fragmentation
    /// or capacity pressure; callers can evict detail and retry after retirement.
    /// Never replaces a live reservation to satisfy a higher-priority request.
    pub fn reserve(&mut self, size: [u32; 3]) -> Option<Arc<VolumeSlot>> {
        if size.contains(&0) { return None; }
        self.root.collect();
        let mut best = None;
        self.root.best(size, &mut Vec::new(), &mut best);
        best.map(|(_, path)| self.root.reserve(&path, size))
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn page_eviction_waits_for_every_frame_lease() {
        let mut slots = VolumeSlots::new([8; 3]).unwrap();
        let cache = slots.reserve([8; 3]).unwrap();
        let scene = cache.clone(); let camera = cache.clone();
        drop(cache);
        assert!(slots.reserve([1; 3]).is_none());
        drop(scene);
        assert!(slots.reserve([1; 3]).is_none());
        drop(camera);
        let replacement = slots.reserve([8; 3]).unwrap();
        assert_eq!(replacement.origin(), [0; 3]);
        assert_eq!(replacement.size(), [8; 3]);
        assert!(slots.reserve([0, 1, 1]).is_none());
        assert!(VolumeSlots::new([1025, 1, 1]).is_err());
    }

    #[test] fn mixed_page_churn_preserves_live_regions_and_recovers_the_entire_atlas() {
        let dims = [31, 23, 19];
        let mut slots = VolumeSlots::new(dims).unwrap();
        let mut live: Vec<Arc<VolumeSlot>> = Vec::new(); let mut seed = 913u32;
        let mut next = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed };
        let mut accepted = 0;
        for _ in 0..2000 {
            if !live.is_empty() && next() % 3 == 0 { live.swap_remove(next() as usize % live.len()); }
            let size = std::array::from_fn(|_| next() % 7 + 1);
            if let Some(page) = slots.reserve(size) {
                assert!((0..3).all(|a| page.origin()[a] + page.size()[a] <= dims[a]));
                for old in &live {
                    assert!((0..3).any(|a| page.origin()[a] >= old.origin()[a] + old.size()[a]
                        || old.origin()[a] >= page.origin()[a] + page.size()[a]), "live atlas reservations overlap");
                }
                live.push(page); accepted += 1;
            }
        }
        assert!(accepted > 400, "churn fixture did not exercise enough allocations");
        drop(live);
        assert!(slots.reserve(dims).is_some(), "retired splits failed to coalesce");
    }
}
