//! Paged glyph atlas: etagere shelf allocation per page, per-rect uploads and
//! LRU eviction of glyphs that have not been drawn for a few frames.
//!
//! Pages are layers of one image array per pixel format (the renderer owns the
//! images); this module is the CPU bookkeeping: where each glyph lives, which
//! rects still need uploading, and which glyph to drop when every page is full.

use std::collections::HashMap;

use etagere::{AllocId, AtlasAllocator, size2};

/// Pixel format of an atlas (one image array each).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtlasFormat {
    /// One byte per texel: single-channel SDF.
    R8,
    /// Four bytes per texel: MTSDF or colour bitmaps.
    Rgba8,
}

impl AtlasFormat {
    pub fn bytes_per_texel(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::Rgba8 => 4,
        }
    }
}

/// What a cached glyph is: which face, which glyph, which raster mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    pub font: u32,
    pub glyph: u32,
    /// `GlyphKind` discriminant (separates MTSDF/SDF renders of one glyph).
    pub kind: u8,
}

/// Handle of a live glyph slot. Stale after the glyph is evicted: check
/// [`Atlas::epoch`] before reusing one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(pub u32);

/// Where a glyph's texels sit (padding included, gutter excluded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRect {
    pub layer: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// A rect of texels the GPU copy still needs.
#[derive(Clone, Debug)]
pub struct AtlasUpload {
    pub format: AtlasFormat,
    pub layer: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

struct Slot {
    key: GlyphKey,
    rect: SlotRect,
    alloc: AllocId,
    last_used: u64,
    pinned: bool,
    live: bool,
}

/// Sizing of an atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasConfig {
    /// Page edge in texels.
    pub page_size: u32,
    /// Most pages (image array layers) it may grow to.
    pub max_pages: u32,
    /// A glyph is only evictable once unused for this many frames, so the
    /// frames still in flight on the GPU never see their texels overwritten.
    pub guard_frames: u64,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self { page_size: 1024, max_pages: 4, guard_frames: 4 }
    }
}

/// Gutter between glyphs, so linear filtering never bleeds a neighbour in.
const GUTTER: u32 = 1;

pub struct Atlas {
    pub format: AtlasFormat,
    cfg: AtlasConfig,
    pages: Vec<AtlasAllocator>,
    slots: Vec<Slot>,
    free_slots: Vec<u32>,
    map: HashMap<GlyphKey, SlotId>,
    frame: u64,
    epoch: u64,
    uploads: Vec<AtlasUpload>,
    evictions: u64,
}

impl Atlas {
    pub fn new(format: AtlasFormat, cfg: AtlasConfig) -> Self {
        Self {
            format,
            cfg,
            pages: Vec::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            map: HashMap::new(),
            frame: 0,
            epoch: 0,
            uploads: Vec::new(),
            evictions: 0,
        }
    }

    pub fn config(&self) -> AtlasConfig {
        self.cfg
    }

    /// Advance the frame counter used for LRU ages.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Bumps whenever a glyph is evicted: instance lists built under an older
    /// epoch may point at texels that now hold something else.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn eviction_count(&self) -> u64 {
        self.evictions
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn live_glyphs(&self) -> usize {
        self.map.len()
    }

    /// Slot of `key`, marking it used this frame.
    pub fn get(&mut self, key: &GlyphKey) -> Option<(SlotId, SlotRect)> {
        let id = *self.map.get(key)?;
        let s = &mut self.slots[id.0 as usize];
        s.last_used = self.frame;
        Some((id, s.rect))
    }

    /// Mark a slot used this frame (cached instance lists call this each frame
    /// for the glyphs they reference, so they are never evicted from under it).
    pub fn touch(&mut self, id: SlotId) {
        if let Some(s) = self.slots.get_mut(id.0 as usize)
            && s.live
        {
            s.last_used = self.frame;
        }
    }

    pub fn contains(&self, key: &GlyphKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn rect(&self, id: SlotId) -> Option<SlotRect> {
        self.slots.get(id.0 as usize).filter(|s| s.live).map(|s| s.rect)
    }

    /// Pin a slot against eviction (imported/pre-baked glyphs).
    pub fn pin(&mut self, id: SlotId) {
        if let Some(s) = self.slots.get_mut(id.0 as usize) {
            s.pinned = true;
        }
    }

    /// Allocate `w * h` texels for `key` and queue `data` for upload. Grows by a
    /// page first, then evicts least-recently-used glyphs. `None` when the
    /// glyph is larger than a page, or every page is full of glyphs drawn this
    /// frame (the caller retries later).
    pub fn insert(&mut self, key: GlyphKey, w: u32, h: u32, data: Vec<u8>) -> Option<(SlotId, SlotRect)> {
        debug_assert_eq!(data.len(), (w * h) as usize * self.format.bytes_per_texel());
        if let Some(found) = self.get(&key) {
            return Some(found);
        }
        let (aw, ah) = (w + 2 * GUTTER, h + 2 * GUTTER);
        if aw > self.cfg.page_size || ah > self.cfg.page_size {
            return None;
        }
        let mut placed = self.try_alloc(aw, ah);
        if placed.is_none() && (self.pages.len() as u32) < self.cfg.max_pages {
            let size = self.cfg.page_size as i32;
            self.pages.push(AtlasAllocator::new(size2(size, size)));
            placed = self.try_alloc(aw, ah);
        }
        if placed.is_none() {
            placed = self.evict_until_fits(aw, ah);
        }
        let (layer, alloc, x, y) = placed?;
        let rect = SlotRect { layer, x: x + GUTTER, y: y + GUTTER, w, h };
        let slot = Slot { key, rect, alloc, last_used: self.frame, pinned: false, live: true };
        let id = match self.free_slots.pop() {
            Some(i) => {
                self.slots[i as usize] = slot;
                SlotId(i)
            }
            None => {
                self.slots.push(slot);
                SlotId(self.slots.len() as u32 - 1)
            }
        };
        self.map.insert(key, id);
        self.uploads.push(AtlasUpload { format: self.format, layer, x: rect.x, y: rect.y, w, h, data });
        Some((id, rect))
    }

    fn try_alloc(&mut self, w: u32, h: u32) -> Option<(u32, AllocId, u32, u32)> {
        for (layer, page) in self.pages.iter_mut().enumerate() {
            if let Some(a) = page.allocate(size2(w as i32, h as i32)) {
                return Some((layer as u32, a.id, a.rectangle.min.x as u32, a.rectangle.min.y as u32));
            }
        }
        None
    }

    fn evict_until_fits(&mut self, w: u32, h: u32) -> Option<(u32, AllocId, u32, u32)> {
        let limit = self.frame.saturating_sub(self.cfg.guard_frames);
        let mut candidates: Vec<(u64, u32)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.live && !s.pinned && s.last_used < limit)
            .map(|(i, s)| (s.last_used, i as u32))
            .collect();
        candidates.sort_unstable();
        for (_, i) in candidates {
            let s = &mut self.slots[i as usize];
            s.live = false;
            let (key, alloc, layer) = (s.key, s.alloc, s.rect.layer);
            self.map.remove(&key);
            self.pages[layer as usize].deallocate(alloc);
            self.free_slots.push(i);
            self.evictions += 1;
            self.epoch += 1;
            if let Some(p) = self.try_alloc(w, h) {
                return Some(p);
            }
        }
        self.try_alloc(w, h)
    }

    /// Rects waiting for the GPU copy, oldest first.
    pub fn pending_uploads(&self) -> usize {
        self.uploads.len()
    }

    /// Take every queued upload.
    pub fn take_uploads(&mut self) -> Vec<AtlasUpload> {
        std::mem::take(&mut self.uploads)
    }

    /// Drop everything (font set changed, device lost).
    pub fn clear(&mut self) {
        self.pages.clear();
        self.slots.clear();
        self.free_slots.clear();
        self.map.clear();
        self.uploads.clear();
        self.epoch += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(g: u32) -> GlyphKey {
        GlyphKey { font: 0, glyph: g, kind: 0 }
    }

    fn atlas(page: u32, pages: u32) -> Atlas {
        Atlas::new(AtlasFormat::R8, AtlasConfig { page_size: page, max_pages: pages, guard_frames: 2 })
    }

    fn put(a: &mut Atlas, g: u32, w: u32, h: u32) -> Option<(SlotId, SlotRect)> {
        a.insert(key(g), w, h, vec![7; (w * h) as usize])
    }

    #[test]
    fn packs_without_overlap_and_queues_uploads() {
        let mut a = atlas(64, 1);
        let mut rects = Vec::new();
        for g in 0..9 {
            let (_, r) = put(&mut a, g, 14, 14).expect("fits");
            rects.push(r);
        }
        for (i, r) in rects.iter().enumerate() {
            assert!(r.x + r.w <= 64 && r.y + r.h <= 64);
            for s in &rects[i + 1..] {
                let apart = r.x + r.w <= s.x || s.x + s.w <= r.x || r.y + r.h <= s.y || s.y + s.h <= r.y;
                assert!(apart, "{r:?} overlaps {s:?}");
            }
        }
        assert_eq!(a.pending_uploads(), 9);
        let ups = a.take_uploads();
        assert_eq!(ups.len(), 9);
        assert_eq!(ups[0].data.len(), 14 * 14);
        assert_eq!(a.pending_uploads(), 0);
    }

    #[test]
    fn reinsert_returns_the_same_slot() {
        let mut a = atlas(64, 1);
        let first = put(&mut a, 5, 10, 10).unwrap();
        let again = put(&mut a, 5, 10, 10).unwrap();
        assert_eq!(first, again);
        assert_eq!(a.pending_uploads(), 1);
    }

    #[test]
    fn grows_pages_before_evicting() {
        let mut a = atlas(32, 2);
        // 30x30 fills a 32x32 page (one per page with gutters).
        put(&mut a, 1, 30, 30).unwrap();
        let (_, r) = put(&mut a, 2, 30, 30).unwrap();
        assert_eq!(r.layer, 1);
        assert_eq!(a.page_count(), 2);
        assert_eq!(a.eviction_count(), 0);
        assert!(put(&mut a, 3, 30, 30).is_none(), "both pages busy this frame");
    }

    #[test]
    fn evicts_least_recently_used_after_guard() {
        let mut a = atlas(32, 1);
        let (s1, _) = put(&mut a, 1, 30, 30).unwrap();
        assert!(put(&mut a, 2, 30, 30).is_none(), "used this frame: not evictable");
        for _ in 0..5 {
            a.begin_frame();
        }
        let epoch = a.epoch();
        let (_, r) = put(&mut a, 2, 30, 30).expect("evicts glyph 1");
        assert_eq!(r.layer, 0);
        assert!(!a.contains(&key(1)));
        assert!(a.contains(&key(2)));
        assert!(a.epoch() > epoch);
        assert_eq!(a.eviction_count(), 1);
        assert!(a.rect(s1).is_some(), "slot index is reused by the new glyph");
    }

    #[test]
    fn touched_and_pinned_glyphs_survive() {
        let mut a = atlas(64, 1);
        let (keep, _) = put(&mut a, 1, 30, 30).unwrap();
        let (pin, _) = put(&mut a, 2, 30, 30).unwrap();
        a.pin(pin);
        put(&mut a, 3, 30, 30).unwrap();
        put(&mut a, 4, 30, 30).unwrap();
        for _ in 0..10 {
            a.begin_frame();
            a.touch(keep);
        }
        // Only glyphs 3 and 4 are old enough and unpinned.
        let (_, r) = put(&mut a, 5, 30, 30).expect("room after eviction");
        assert!(a.contains(&key(1)) && a.contains(&key(2)));
        assert!(a.contains(&key(5)));
        assert!(r.x + r.w <= 64);
        assert!(!a.contains(&key(3)) || !a.contains(&key(4)));
    }

    #[test]
    fn oversize_glyph_is_rejected() {
        let mut a = atlas(32, 1);
        assert!(put(&mut a, 1, 40, 8).is_none());
    }
}
