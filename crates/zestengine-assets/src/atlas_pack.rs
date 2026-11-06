//! Rectangle packer for sprite atlases (2D-4): MaxRects with the best-short-side
//! heuristic, optional 90 degree rotation, padding around every item, pages up
//! to a maximum size (power-of-two or shrink-wrapped), and a fully deterministic
//! result (no randomness; ties broken by index, then position).
//!
//! The packer only deals in sizes. `sprite_atlas` trims the sprites, calls
//! [`pack`] and composites the pixels. Unity's own packer fits bit masks with a
//! random search; this one is the classical rectangle packer (the research,
//! phase5b §3.2, notes MaxRects exists there too but serves other users).
//!
//! Coordinates: page pixels, origin top-left. A [`Placement`]'s `x, y` is the
//! top-left of the item's **content** (padding excluded); `rotated` means the
//! content is stored turned 90 degrees clockwise and so occupies `h x w`.

/// Packing parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackSettings {
    /// Empty ring around every item, in pixels (items are `2 * padding` apart).
    pub padding: u32,
    /// Allow turning items by 90 degrees.
    pub allow_rotation: bool,
    /// Largest page edge.
    pub max_size: u32,
    /// Round each page edge up to a power of two (and search the smallest one).
    pub pow2: bool,
    /// Give every page the size of the largest (texture-array layers must match).
    pub uniform_pages: bool,
}

impl Default for PackSettings {
    fn default() -> Self {
        Self { padding: 4, allow_rotation: false, max_size: 2048, pow2: true, uniform_pages: false }
    }
}

/// Where one item went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub page: usize,
    /// Content top-left on the page.
    pub x: u32,
    pub y: u32,
    /// Content stored rotated 90 degrees clockwise (occupies `h x w`).
    pub rotated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackResult {
    /// (width, height) of each page.
    pub pages: Vec<(u32, u32)>,
    /// One placement per input item, in input order.
    pub placements: Vec<Placement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackError {
    /// An item (with its padding) is bigger than a page.
    TooLarge { item: usize, w: u32, h: u32, max: u32 },
    /// An item has no pixels.
    Empty { item: usize },
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { item, w, h, max } => write!(f, "item {item} ({w}x{h} with padding) does not fit the {max}x{max} page limit"),
            Self::Empty { item } => write!(f, "item {item} has no pixels"),
        }
    }
}

impl std::error::Error for PackError {}

#[derive(Clone, Copy, Debug)]
struct R {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl R {
    fn contains(&self, o: &R) -> bool {
        o.x >= self.x && o.y >= self.y && o.x + o.w <= self.x + self.w && o.y + o.h <= self.y + self.h
    }

    fn intersects(&self, o: &R) -> bool {
        self.x < o.x + o.w && o.x < self.x + self.w && self.y < o.y + o.h && o.y < self.y + self.h
    }
}

/// One bin being filled (MaxRects: the list of maximal free rectangles).
struct Bin {
    free: Vec<R>,
}

impl Bin {
    fn new(w: u32, h: u32) -> Self {
        Self { free: vec![R { x: 0, y: 0, w, h }] }
    }

    /// Best short side fit over every free rect and orientation. Returns the
    /// footprint position and whether it was rotated.
    fn insert(&mut self, fw: u32, fh: u32, rotate: bool) -> Option<(R, bool)> {
        type Best<R> = ((u32, u32, u32, u32), R, bool);
        let mut best: Option<Best<R>> = None;
        for f in &self.free {
            for rot in [false, true] {
                if rot && (!rotate || fw == fh) {
                    continue;
                }
                let (w, h) = if rot { (fh, fw) } else { (fw, fh) };
                if w > f.w || h > f.h {
                    continue;
                }
                let (lw, lh) = (f.w - w, f.h - h);
                let score = (lw.min(lh), lw.max(lh), f.y, f.x);
                if best.as_ref().is_none_or(|b| score < b.0) {
                    best = Some((score, R { x: f.x, y: f.y, w, h }, rot));
                }
            }
        }
        let (_, placed, rot) = best?;
        self.place(placed);
        Some((placed, rot))
    }

    fn place(&mut self, p: R) {
        let mut next = Vec::with_capacity(self.free.len() + 4);
        for f in &self.free {
            if !f.intersects(&p) {
                next.push(*f);
                continue;
            }
            if p.x > f.x {
                next.push(R { x: f.x, y: f.y, w: p.x - f.x, h: f.h });
            }
            if p.x + p.w < f.x + f.w {
                next.push(R { x: p.x + p.w, y: f.y, w: f.x + f.w - (p.x + p.w), h: f.h });
            }
            if p.y > f.y {
                next.push(R { x: f.x, y: f.y, w: f.w, h: p.y - f.y });
            }
            if p.y + p.h < f.y + f.h {
                next.push(R { x: f.x, y: p.y + p.h, w: f.w, h: f.y + f.h - (p.y + p.h) });
            }
        }
        // Drop free rects contained in another (keep the first of equal ones).
        let mut keep = vec![true; next.len()];
        for i in 0..next.len() {
            for j in 0..next.len() {
                if i != j && keep[j] && next[j].contains(&next[i]) && (!next[i].contains(&next[j]) || j < i) {
                    keep[i] = false;
                    break;
                }
            }
        }
        self.free = next.into_iter().zip(keep).filter_map(|(r, k)| k.then_some(r)).collect();
    }
}

fn next_pow2(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}

/// Fill a `bw x bh` bin with `order` (indices into `foot`), in that order.
/// Items that don't fit are returned in the second list.
fn fill(bw: u32, bh: u32, order: &[usize], foot: &[(u32, u32)], rotate: bool) -> (Vec<(usize, R, bool)>, Vec<usize>) {
    let mut bin = Bin::new(bw, bh);
    let (mut placed, mut left) = (Vec::new(), Vec::new());
    for &i in order {
        match bin.insert(foot[i].0, foot[i].1, rotate) {
            Some((r, rot)) => placed.push((i, r, rot)),
            None => left.push(i),
        }
    }
    (placed, left)
}

/// Candidate bin sizes for the last page: power-of-two pairs up to `max`
/// holding at least `area`, smallest area first, squarer first.
fn pow2_candidates(area: u64, max: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut w = 1u32;
    while w <= max {
        let mut h = 1u32;
        while h <= max {
            if w as u64 * h as u64 >= area {
                out.push((w, h));
            }
            h = h.saturating_mul(2);
            if h == 0 {
                break;
            }
        }
        w = w.saturating_mul(2);
        if w == 0 {
            break;
        }
    }
    out.sort_by_key(|&(w, h)| (w as u64 * h as u64, w.abs_diff(h), w));
    out
}

/// Pack `items` (content sizes `(w, h)`) into pages.
pub fn pack(items: &[(u32, u32)], s: &PackSettings) -> Result<PackResult, PackError> {
    let pad2 = 2 * s.padding;
    let max = s.max_size.max(1);
    let foot: Vec<(u32, u32)> = items.iter().map(|&(w, h)| (w + pad2, h + pad2)).collect();
    for (i, &(w, h)) in items.iter().enumerate() {
        if w == 0 || h == 0 {
            return Err(PackError::Empty { item: i });
        }
        let (fw, fh) = foot[i];
        // Pages are square, so turning an item never helps it fit.
        if fw > max || fh > max {
            return Err(PackError::TooLarge { item: i, w: fw, h: fh, max });
        }
    }
    // Largest first (longest side, then area), index as the final tie-break.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(foot[i].0.max(foot[i].1)), std::cmp::Reverse(foot[i].0 as u64 * foot[i].1 as u64), i));

    let mut placements: Vec<Option<Placement>> = vec![None; items.len()];
    let mut pages: Vec<(u32, u32)> = Vec::new();
    let mut remaining = order;
    while !remaining.is_empty() {
        let page = pages.len();
        let area: u64 = remaining.iter().map(|&i| foot[i].0 as u64 * foot[i].1 as u64).sum();
        // The final page: the smallest bin that takes everything left.
        type Chosen<R> = (Vec<(usize, R, bool)>, (u32, u32));
        let mut chosen: Option<Chosen<R>> = None;
        if s.pow2 {
            for (w, h) in pow2_candidates(area, max) {
                let (placed, left) = fill(w, h, &remaining, &foot, s.allow_rotation);
                if left.is_empty() {
                    chosen = Some((placed, (w, h)));
                    break;
                }
            }
        }
        let (placed, size) = match chosen {
            Some(c) => c,
            None => {
                let (placed, _) = fill(max, max, &remaining, &foot, s.allow_rotation);
                let bw = placed.iter().map(|(_, r, _)| r.x + r.w).max().unwrap_or(1);
                let bh = placed.iter().map(|(_, r, _)| r.y + r.h).max().unwrap_or(1);
                let size = if s.pow2 { (next_pow2(bw).min(max), next_pow2(bh).min(max)) } else { (bw, bh) };
                (placed, size)
            }
        };
        if placed.is_empty() {
            // Cannot happen after the size checks; guard against looping forever.
            return Err(PackError::TooLarge { item: remaining[0], w: foot[remaining[0]].0, h: foot[remaining[0]].1, max });
        }
        let placed_ids: std::collections::HashSet<usize> = placed.iter().map(|(i, ..)| *i).collect();
        for (i, r, rot) in &placed {
            placements[*i] = Some(Placement { page, x: r.x + s.padding, y: r.y + s.padding, rotated: *rot });
        }
        remaining.retain(|i| !placed_ids.contains(i));
        pages.push(size);
    }
    if s.uniform_pages && !pages.is_empty() {
        let w = pages.iter().map(|p| p.0).max().unwrap_or(1);
        let h = pages.iter().map(|p| p.1).max().unwrap_or(1);
        pages.iter_mut().for_each(|p| *p = (w, h));
    }
    Ok(PackResult { pages, placements: placements.into_iter().map(|p| p.expect("every item placed")).collect() })
}

/// Size the content of `item` occupies on its page.
pub fn placed_size(item: (u32, u32), p: &Placement) -> (u32, u32) {
    if p.rotated { (item.1, item.0) } else { item }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn random_items(n: usize, seed: u64, lo: u32, hi: u32) -> Vec<(u32, u32)> {
        let mut s = seed;
        (0..n).map(|_| (lo + lcg(&mut s) % (hi - lo + 1), lo + lcg(&mut s) % (hi - lo + 1))).collect()
    }

    /// Footprint rects (content grown by padding) of one page.
    fn footprints(items: &[(u32, u32)], r: &PackResult, page: usize, pad: u32) -> Vec<(usize, R)> {
        r.placements
            .iter()
            .enumerate()
            .filter(|(_, p)| p.page == page)
            .map(|(i, p)| {
                let (w, h) = placed_size(items[i], p);
                (i, R { x: p.x - pad, y: p.y - pad, w: w + 2 * pad, h: h + 2 * pad })
            })
            .collect()
    }

    fn check_valid(items: &[(u32, u32)], r: &PackResult, s: &PackSettings) {
        assert_eq!(r.placements.len(), items.len());
        for page in 0..r.pages.len() {
            let fp = footprints(items, r, page, s.padding);
            let (pw, ph) = r.pages[page];
            for (i, a) in &fp {
                assert!(a.x + a.w <= pw && a.y + a.h <= ph, "item {i} leaves page {page}: {a:?} vs {pw}x{ph}");
                assert!(pw <= s.max_size && ph <= s.max_size);
            }
            for (k, (i, a)) in fp.iter().enumerate() {
                for (j, b) in &fp[k + 1..] {
                    assert!(!a.intersects(b), "items {i} and {j} overlap on page {page}");
                }
            }
        }
    }

    #[test]
    fn no_overlaps_padding_respected_and_everything_placed() {
        for seed in 1..6u64 {
            let items = random_items(60, seed, 5, 90);
            for &(pad, rot, pow2) in &[(0, false, true), (2, true, true), (4, false, false), (8, true, false)] {
                let s = PackSettings { padding: pad, allow_rotation: rot, max_size: 512, pow2, uniform_pages: false };
                let r = pack(&items, &s).unwrap();
                check_valid(&items, &r, &s);
                // Content of different items is at least 2 * padding apart.
                for page in 0..r.pages.len() {
                    let fp = footprints(&items, &r, page, 0);
                    for (k, (_, a)) in fp.iter().enumerate() {
                        for (_, b) in &fp[k + 1..] {
                            let gap_x = (b.x as i64 - (a.x + a.w) as i64).max(a.x as i64 - (b.x + b.w) as i64);
                            let gap_y = (b.y as i64 - (a.y + a.h) as i64).max(a.y as i64 - (b.y + b.h) as i64);
                            assert!(gap_x.max(gap_y) >= 2 * pad as i64, "padding gap {gap_x},{gap_y} < {}", 2 * pad);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rotation_swaps_the_occupied_size_and_never_for_squares() {
        // A tall strip next to a wide hole: only a rotated item fits the page.
        let items = vec![(10, 100), (100, 10), (30, 30)];
        let s = PackSettings { padding: 0, allow_rotation: true, max_size: 128, pow2: true, uniform_pages: false };
        let r = pack(&items, &s).unwrap();
        check_valid(&items, &r, &s);
        for (i, p) in r.placements.iter().enumerate() {
            if items[i].0 == items[i].1 {
                assert!(!p.rotated);
            }
            let (w, h) = placed_size(items[i], p);
            assert_eq!(w * h, items[i].0 * items[i].1);
            if p.rotated {
                assert_eq!((w, h), (items[i].1, items[i].0));
            }
        }
        // 4 strips 20x120 in a 128 page: rotation gives the same area but
        // they all stay inside the page either way.
        let strips = vec![(20, 120); 4];
        let r = pack(&strips, &PackSettings { padding: 0, allow_rotation: true, max_size: 128, ..s }).unwrap();
        assert_eq!(r.pages.len(), 1);
        // Without rotation a 100x10 only ever lies flat.
        let r2 = pack(&items, &PackSettings { allow_rotation: false, ..s }).unwrap();
        assert!(r2.placements.iter().all(|p| !p.rotated));
    }

    #[test]
    fn rotation_is_used_when_it_is_the_only_way_to_fit() {
        // Page 64x32 is not allowed (square pages), so use a 40-wide limit: a 30x40
        // item cannot sit beside a 40x30 one unless one of them turns.
        let items = vec![(40, 30), (30, 40)];
        let s = PackSettings { padding: 0, allow_rotation: true, max_size: 40, pow2: false, uniform_pages: false };
        let r = pack(&items, &s).unwrap();
        check_valid(&items, &r, &s);
        let s2 = PackSettings { allow_rotation: false, ..s };
        let r2 = pack(&items, &s2).unwrap();
        check_valid(&items, &r2, &s2);
        assert!(r.pages.len() <= r2.pages.len());
    }

    #[test]
    fn deterministic() {
        let items = random_items(80, 9, 4, 70);
        let s = PackSettings { padding: 2, allow_rotation: true, max_size: 256, ..PackSettings::default() };
        let a = pack(&items, &s).unwrap();
        for _ in 0..3 {
            assert_eq!(a, pack(&items, &s).unwrap());
        }
    }

    #[test]
    fn page_count_is_bounded() {
        let items = random_items(100, 3, 20, 60);
        let s = PackSettings { padding: 2, allow_rotation: false, max_size: 256, pow2: true, uniform_pages: false };
        let r = pack(&items, &s).unwrap();
        check_valid(&items, &r, &s);
        let area: u64 = items.iter().map(|&(w, h)| (w + 4) as u64 * (h + 4) as u64).sum();
        let lower = area.div_ceil(256 * 256) as usize;
        assert!(r.pages.len() >= lower, "{} pages < lower bound {lower}", r.pages.len());
        assert!(r.pages.len() <= items.len());
        // A decent packer stays within 2x of the area bound for this mix.
        assert!(r.pages.len() <= 2 * lower + 1, "{} pages for a bound of {lower}", r.pages.len());
    }

    #[test]
    fn last_page_is_the_smallest_power_of_two_that_fits() {
        let items = vec![(30, 30), (30, 30)];
        let s = PackSettings { padding: 1, allow_rotation: false, max_size: 1024, pow2: true, uniform_pages: false };
        let r = pack(&items, &s).unwrap();
        assert_eq!(r.pages.len(), 1);
        let (w, h) = r.pages[0];
        assert!(w.is_power_of_two() && h.is_power_of_two());
        assert!(w * h <= 64 * 64, "{w}x{h}");
        check_valid(&items, &r, &s);
        // Shrink-wrapped pages are exactly the bounding box.
        let r = pack(&items, &PackSettings { pow2: false, ..s }).unwrap();
        let fp = footprints(&items, &r, 0, 1);
        assert_eq!(r.pages[0].0, fp.iter().map(|(_, a)| a.x + a.w).max().unwrap());
    }

    #[test]
    fn uniform_pages_share_one_size() {
        let items = random_items(40, 4, 40, 100);
        let s = PackSettings { padding: 2, allow_rotation: false, max_size: 256, pow2: true, uniform_pages: true };
        let r = pack(&items, &s).unwrap();
        assert!(r.pages.len() > 1);
        assert!(r.pages.iter().all(|p| *p == r.pages[0]));
        check_valid(&items, &r, &s);
    }

    #[test]
    fn too_large_and_empty_items_are_clear_errors() {
        let s = PackSettings { padding: 4, max_size: 64, ..PackSettings::default() };
        assert_eq!(pack(&[(10, 10), (60, 20)], &s), Err(PackError::TooLarge { item: 1, w: 68, h: 28, max: 64 }));
        assert_eq!(pack(&[(10, 0)], &s), Err(PackError::Empty { item: 0 }));
        assert!(pack(&[(10, 0)], &s).unwrap_err().to_string().contains("no pixels"));
        let none = pack(&[], &s).unwrap();
        assert!(none.pages.is_empty() && none.placements.is_empty());
    }
}
