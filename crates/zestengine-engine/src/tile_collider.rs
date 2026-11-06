//! Pure geometry of tilemap colliders (2D-6 leftover): greedy rectangle
//! merging of solid cells and boundary tracing, on one 16 x 16 chunk mask.
//! The rapier side (compound per chunk, incremental updates) is in
//! `physics2d.rs`.

use glam::IVec2;

pub const N: usize = 16;

/// A rectangle of cells: `x, y` of the min cell and the size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// Merge the set cells of a `N x N` mask (index `y * 16 + x`) into rectangles:
/// scan rows bottom to top, grow each unused cell to the right as far as it
/// goes, then upward while the whole row segment is set. Area is preserved,
/// rectangles never overlap.
pub fn greedy_rects(mask: &[bool; N * N]) -> Vec<CellRect> {
    let mut used = [false; N * N];
    let mut out = Vec::new();
    for y in 0..N {
        for x in 0..N {
            let i = y * N + x;
            if !mask[i] || used[i] {
                continue;
            }
            let mut w = 1;
            while x + w < N && mask[y * N + x + w] && !used[y * N + x + w] {
                w += 1;
            }
            let mut h = 1;
            'grow: while y + h < N {
                for k in 0..w {
                    let j = (y + h) * N + x + k;
                    if !mask[j] || used[j] {
                        break 'grow;
                    }
                }
                h += 1;
            }
            for yy in y..y + h {
                for xx in x..x + w {
                    used[yy * N + xx] = true;
                }
            }
            out.push(CellRect { x: x as i32, y: y as i32, w: w as i32, h: h as i32 });
        }
    }
    out
}

/// One cell per rectangle (no merging): the reference for tests and the
/// `merge_cells = false` mode.
pub fn single_rects(mask: &[bool; N * N]) -> Vec<CellRect> {
    (0..N * N).filter(|&i| mask[i]).map(|i| CellRect { x: (i % N) as i32, y: (i / N) as i32, w: 1, h: 1 }).collect()
}

/// Closed boundary rings of the solid cells, in corner coordinates (a cell
/// `(x, y)` spans corners `(x, y)`..`(x+1, y+1)`). Solid regions wind
/// counter clockwise, holes clockwise; collinear points are dropped. Corners
/// where two cells touch only diagonally are split so every ring is simple.
pub fn trace_outlines(mask: &[bool; N * N]) -> Vec<Vec<IVec2>> {
    let solid = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < N && (y as usize) < N && mask[y as usize * N + x as usize];
    // Directed boundary edges with the solid on the left.
    let mut edges: Vec<(IVec2, IVec2)> = Vec::new();
    for y in 0..N as i32 {
        for x in 0..N as i32 {
            if !solid(x, y) {
                continue;
            }
            if !solid(x, y - 1) {
                edges.push((IVec2::new(x, y), IVec2::new(x + 1, y)));
            }
            if !solid(x + 1, y) {
                edges.push((IVec2::new(x + 1, y), IVec2::new(x + 1, y + 1)));
            }
            if !solid(x, y + 1) {
                edges.push((IVec2::new(x + 1, y + 1), IVec2::new(x, y + 1)));
            }
            if !solid(x - 1, y) {
                edges.push((IVec2::new(x, y + 1), IVec2::new(x, y)));
            }
        }
    }
    let mut taken = vec![false; edges.len()];
    let mut rings = Vec::new();
    for start in 0..edges.len() {
        if taken[start] {
            continue;
        }
        let mut ring = vec![edges[start].0];
        let mut cur = start;
        loop {
            taken[cur] = true;
            let (a, b) = edges[cur];
            if b == edges[start].0 {
                break;
            }
            ring.push(b);
            // Among the edges leaving `b`, prefer the sharpest left turn so
            // diagonal contacts split into separate rings.
            let dir = b - a;
            let mut best: Option<(i32, usize)> = None;
            for (j, e) in edges.iter().enumerate() {
                if taken[j] || e.0 != b {
                    continue;
                }
                let d = e.1 - e.0;
                let cross = dir.x * d.y - dir.y * d.x; // > 0 = left turn
                let dot = dir.x * d.x + dir.y * d.y;
                let score = if cross > 0 { 3 } else if dot > 0 { 2 } else if cross < 0 { 1 } else { 0 };
                if best.is_none_or(|(s, _)| score > s) {
                    best = Some((score, j));
                }
            }
            match best {
                Some((_, j)) => cur = j,
                None => break,
            }
        }
        // Drop collinear points.
        let n = ring.len();
        let mut simple = Vec::new();
        for i in 0..n {
            let (p, c, q) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
            let (d1, d2) = (c - p, q - c);
            if d1.x * d2.y - d1.y * d2.x != 0 {
                simple.push(c);
            }
        }
        if simple.len() >= 3 {
            rings.push(simple);
        }
    }
    rings
}

/// Signed area of a ring (positive = counter clockwise).
pub fn ring_area(r: &[IVec2]) -> i64 {
    let n = r.len();
    (0..n).map(|i| r[i].x as i64 * r[(i + 1) % n].y as i64 - r[(i + 1) % n].x as i64 * r[i].y as i64).sum::<i64>() / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_of(mut f: impl FnMut(usize, usize) -> bool) -> [bool; N * N] {
        let mut m = [false; N * N];
        for y in 0..N {
            for x in 0..N {
                m[y * N + x] = f(x, y);
            }
        }
        m
    }

    fn check(mask: &[bool; N * N], rects: &[CellRect]) {
        let mut cover = [0u8; N * N];
        for r in rects {
            for y in r.y..r.y + r.h {
                for x in r.x..r.x + r.w {
                    cover[y as usize * N + x as usize] += 1;
                }
            }
        }
        for i in 0..N * N {
            assert_eq!(cover[i], mask[i] as u8, "cell {i}: area preserved, no overlap");
        }
    }

    #[test]
    fn greedy_merge_preserves_area_without_overlap_and_is_compact() {
        let solid = mask_of(|_, _| true);
        let r = greedy_rects(&solid);
        check(&solid, &r);
        assert_eq!(r.len(), 1, "a full chunk is one rectangle");
        let floor = mask_of(|_, y| y < 2);
        assert_eq!(greedy_rects(&floor).len(), 1);
        let checker = mask_of(|x, y| (x + y) % 2 == 0);
        let r = greedy_rects(&checker);
        check(&checker, &r);
        assert_eq!(r.len(), 128, "no merging possible");
        let l = mask_of(|x, y| x < 3 || y < 3);
        let r = greedy_rects(&l);
        check(&l, &r);
        assert!(r.len() <= 2, "an L is two rectangles: {r:?}");
        // Random masks: always valid, never worse than one rect per cell.
        let mut s = 7u32;
        for _ in 0..50 {
            let m = mask_of(|_, _| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                !s.is_multiple_of(3)
            });
            let r = greedy_rects(&m);
            check(&m, &r);
            assert!(r.len() <= single_rects(&m).len());
        }
    }

    #[test]
    fn outlines_trace_islands_and_holes() {
        let m = mask_of(|x, y| (2..6).contains(&x) && (2..5).contains(&y));
        let rings = trace_outlines(&m);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 4, "a rectangle has four corners");
        assert_eq!(ring_area(&rings[0]), 12);
        // A ring with a hole: outer ccw, inner cw, areas subtract.
        let ring = mask_of(|x, y| (1..8).contains(&x) && (1..8).contains(&y) && !((3..6).contains(&x) && (3..6).contains(&y)));
        let rings = trace_outlines(&ring);
        assert_eq!(rings.len(), 2);
        let total: i64 = rings.iter().map(|r| ring_area(r)).sum();
        assert_eq!(total, 49 - 9);
        // Two islands.
        let two = mask_of(|x, y| y < 2 && (x < 2 || (5..8).contains(&x)));
        assert_eq!(trace_outlines(&two).len(), 2);
        // Diagonal contact splits into two simple rings.
        let diag = mask_of(|x, y| (x, y) == (0, 0) || (x, y) == (1, 1));
        let r = trace_outlines(&diag);
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|r| r.len() == 4 && ring_area(r) == 1));
        assert!(trace_outlines(&[false; N * N]).is_empty());
    }
}
