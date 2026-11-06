//! Virtual shadow map page table and physical page pool (CPU side). A
//! directional light's shadow is addressed as a hierarchy of mip levels, each a
//! 2^mip × 2^mip grid of virtual pages covering a world-space square; a fixed
//! physical pool backs the resident pages with LRU eviction.
#![allow(dead_code)] // VSM page table/pool are built + tested but not yet wired into the renderer.

use std::collections::HashMap;

use glam::Vec2;

/// Virtual page table for a directional-light VSM.
pub struct VsmPageTable {
    mips: u32,
    center: Vec2,
    half: f32,
}

impl VsmPageTable {
    pub fn new(mips: u32, center: Vec2, half: f32) -> Self {
        Self { mips: mips.max(1), center, half: half.max(1e-4) }
    }

    /// Pages per axis at a mip level (2^mip).
    pub fn grid_size(&self, mip: u32) -> u32 {
        1u32 << mip.min(self.mips - 1)
    }

    fn cell(&self, mip: u32) -> f32 {
        (self.half * 2.0) / self.grid_size(mip) as f32
    }

    /// World-space XZ rect of page `(mip, x, y)`.
    pub fn page_rect(&self, mip: u32, x: u32, y: u32) -> (Vec2, Vec2) {
        let cell = self.cell(mip);
        let min = Vec2::new(
            self.center.x - self.half + x as f32 * cell,
            self.center.y - self.half + y as f32 * cell,
        );
        (min, min + Vec2::splat(cell))
    }

    /// Pages at `mip` overlapping a world XZ axis-aligned rect.
    pub fn pages_for_rect(&self, mip: u32, min: Vec2, max: Vec2) -> Vec<(u32, u32)> {
        let cell = self.cell(mip);
        let size = self.grid_size(mip);
        let lo = Vec2::new(self.center.x - self.half, self.center.y - self.half);
        let hi = lo + Vec2::splat(self.half * 2.0);
        let omin = min.max(lo);
        let omax = max.min(hi);
        if omin.x >= omax.x || omin.y >= omax.y {
            return Vec::new();
        }
        let x0 = ((omin.x - lo.x) / cell).floor().clamp(0.0, (size - 1) as f32) as u32;
        let x1 = ((omax.x - lo.x) / cell - 1e-4).floor().clamp(0.0, (size - 1) as f32) as u32;
        let y0 = ((omin.y - lo.y) / cell).floor().clamp(0.0, (size - 1) as f32) as u32;
        let y1 = ((omax.y - lo.y) / cell - 1e-4).floor().clamp(0.0, (size - 1) as f32) as u32;
        let mut out = Vec::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.push((x, y));
            }
        }
        out
    }

    /// Pages across all mips overlapping `rect`, coarsest first.
    pub fn pages_for_rect_all(&self, min: Vec2, max: Vec2) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::new();
        for mip in 0..self.mips {
            for (x, y) in self.pages_for_rect(mip, min, max) {
                out.push((mip, x, y));
            }
        }
        out
    }
}

/// Fixed-size physical page pool mapping virtual pages to slots, with LRU
/// eviction when full.
pub struct VsmPagePool {
    capacity: usize,
    mapping: HashMap<(u32, u32, u32), usize>,
    last_used: Vec<u64>,
    tick: u64,
}

impl VsmPagePool {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self { capacity, mapping: HashMap::new(), last_used: vec![0; capacity], tick: 0 }
    }

    /// Ensure `(mip, x, y)` is resident and return its physical slot, evicting
    /// the least-recently-used page when the pool is full.
    pub fn ensure(&mut self, mip: u32, x: u32, y: u32) -> usize {
        self.tick += 1;
        let key = (mip, x, y);
        if let Some(&slot) = self.mapping.get(&key) {
            self.last_used[slot] = self.tick;
            return slot;
        }
        let slot = if self.mapping.len() < self.capacity {
            self.mapping.len()
        } else {
            let (evict, _) = self.last_used.iter().enumerate().min_by_key(|&(_, &t)| t).unwrap();
            let key = self.mapping.iter().find(|&(_, &slot)| slot == evict).map(|(k, _)| *k).unwrap();
            self.mapping.remove(&key);
            evict
        };
        self.mapping.insert(key, slot);
        self.last_used[slot] = self.tick;
        slot
    }

    /// Number of resident pages.
    pub fn resident(&self) -> usize {
        self.mapping.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_rect_places_pages_on_a_grid() {
        let t = VsmPageTable::new(4, Vec2::ZERO, 100.0);
        let (min, max) = t.page_rect(0, 0, 0);
        assert_eq!((min.x, min.y, max.x, max.y), (-100.0, -100.0, 100.0, 100.0));
        let (min, max) = t.page_rect(2, 1, 2);
        assert_eq!((min.x, min.y, max.x, max.y), (-50.0, 0.0, 0.0, 50.0));
    }

    #[test]
    fn pages_for_rect_selects_overlapping_pages() {
        let t = VsmPageTable::new(4, Vec2::ZERO, 100.0);
        let center = t.pages_for_rect(2, Vec2::new(-25.0, -25.0), Vec2::new(25.0, 25.0));
        let mut want = vec![(1, 1), (2, 1), (1, 2), (2, 2)];
        want.sort();
        let mut got = center;
        got.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn pages_for_rect_returns_nothing_outside_the_region() {
        let t = VsmPageTable::new(4, Vec2::ZERO, 100.0);
        assert!(t.pages_for_rect(2, Vec2::new(200.0, 200.0), Vec2::new(300.0, 300.0)).is_empty());
        assert_eq!(t.pages_for_rect(0, Vec2::new(-100.0, -100.0), Vec2::new(100.0, 100.0)).len(), 1);
    }

    #[test]
    fn pool_dedups_and_evicts_lru() {
        let mut p = VsmPagePool::new(2);
        let a = p.ensure(0, 0, 0);
        assert_eq!(p.ensure(0, 0, 0), a, "resident page returns its slot");
        let b = p.ensure(1, 1, 1);
        assert_ne!(a, b);
        assert_eq!(p.resident(), 2);
        // Touch A so B becomes the LRU, then a third page evicts B.
        p.ensure(0, 0, 0);
        let c = p.ensure(2, 2, 2);
        assert_eq!(c, b, "the least-recently-used page is evicted");
        assert_eq!(p.resident(), 2);
    }
}
