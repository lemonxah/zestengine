//! Graph partitioning for the cluster builder, without METIS: Morton-ordered
//! seeds, greedy region growing over the adjacency graph (so every region is
//! connected where the mesh is), a pass that folds fragments into neighbours
//! and a boundary refinement that moves dangling triangles to the region they
//! touch most. Everything is deterministic: ties break on indices.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use zestengine_render::Vertex;

use super::{CLUSTER_MAX_BONES, CLUSTER_MAX_TRIS, CLUSTER_MAX_VERTS};

fn spread_bits(mut x: u32) -> u32 {
    x &= 0x3ff;
    x = (x | (x << 16)) & 0x0300_00FF;
    x = (x | (x << 8)) & 0x0300_F00F;
    x = (x | (x << 4)) & 0x030C_30C3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// Indices of `points` sorted along a 3D Morton curve (ties by index).
pub fn morton_order(points: &[[f32; 3]]) -> Vec<u32> {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let inv: [f32; 3] = std::array::from_fn(|k| if hi[k] > lo[k] { 1023.0 / (hi[k] - lo[k]) } else { 0.0 });
    let mut keyed: Vec<(u32, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let q: [u32; 3] = std::array::from_fn(|k| ((p[k] - lo[k]) * inv[k]).clamp(0.0, 1023.0) as u32);
            (spread_bits(q[0]) | (spread_bits(q[1]) << 1) | (spread_bits(q[2]) << 2), i as u32)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}

/// Compressed sparse row adjacency with a weight per link.
pub struct Csr {
    pub start: Vec<u32>,
    pub adj: Vec<u32>,
    pub weight: Vec<f32>,
}

impl Csr {
    pub fn neighbours(&self, n: u32) -> impl Iterator<Item = (u32, f32)> + '_ {
        let (a, b) = (self.start[n as usize] as usize, self.start[n as usize + 1] as usize);
        self.adj[a..b].iter().copied().zip(self.weight[a..b].iter().copied())
    }

    /// From directed `(from, to, weight)` links; equal pairs are merged by summing.
    pub fn from_links(n: usize, mut links: Vec<(u32, u32, f32)>) -> Csr {
        links.sort_unstable_by_key(|l| (l.0, l.1));
        let mut start = vec![0u32; n + 1];
        let mut adj: Vec<u32> = Vec::with_capacity(links.len());
        let mut weight: Vec<f32> = Vec::with_capacity(links.len());
        let mut last = (u32::MAX, u32::MAX);
        for (a, b, w) in links {
            if (a, b) == last {
                *weight.last_mut().unwrap() += w;
                continue;
            }
            last = (a, b);
            adj.push(b);
            weight.push(w);
            start[a as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        Csr { start, adj, weight }
    }
}

/// Triangles that share an edge (by welded point), as a graph.
pub fn triangle_adjacency(tris: &[[u32; 3]], point: &[u32]) -> Csr {
    let mut edges: Vec<(u64, u32)> = Vec::with_capacity(tris.len() * 3);
    for (t, tri) in tris.iter().enumerate() {
        let p = tri.map(|v| point[v as usize]);
        for k in 0..3 {
            let (a, b) = (p[k].min(p[(k + 1) % 3]), p[k].max(p[(k + 1) % 3]));
            edges.push((((a as u64) << 32) | b as u64, t as u32));
        }
    }
    edges.sort_unstable();
    let mut links = Vec::with_capacity(tris.len() * 3);
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 {
            j += 1;
        }
        // Non-manifold fans link their first few members only.
        let run = &edges[i..j.min(i + 6)];
        for x in 0..run.len() {
            for y in 0..run.len() {
                if run[x].1 != run[y].1 {
                    links.push((run[x].1, run[y].1, 1.0));
                }
            }
        }
        i = j;
    }
    Csr::from_links(tris.len(), links)
}

/// Decides whether a node may join the region being grown.
pub trait Admit {
    fn reset(&mut self);
    /// True (and the node is counted) when `node` fits the region.
    fn admit(&mut self, node: u32) -> bool;
}

/// Parameters for [`region_grow`].
pub struct Grow<'a> {
    pub centroid: &'a [[f32; 3]],
    pub order: &'a [u32],
    pub adj: &'a Csr,
    /// Prefer nodes with the most link weight into the region (cluster
    /// grouping); otherwise grow by distance to the region centre (triangles).
    pub use_weight: bool,
    pub target: usize,
    /// A region that runs out of frontier before this size jumps to the next
    /// unassigned node on the Morton curve.
    pub min_fill: usize,
}

struct Region {
    members: Vec<u32>,
    sum: [f64; 3],
}

#[allow(clippy::too_many_arguments)]
fn add_node(
    node: u32,
    g: &Grow,
    assigned: &mut [bool],
    conn: &mut [f32],
    touched: &mut Vec<u32>,
    heap: &mut BinaryHeap<(u64, Reverse<u32>)>,
    region: &mut Region,
) {
    assigned[node as usize] = true;
    region.members.push(node);
    let c = g.centroid[node as usize];
    for (k, &ck) in c.iter().enumerate() {
        region.sum[k] += ck as f64;
    }
    let m = region.members.len() as f64;
    let centre = [(region.sum[0] / m) as f32, (region.sum[1] / m) as f32, (region.sum[2] / m) as f32];
    for (nb, w) in g.adj.neighbours(node) {
        if assigned[nb as usize] {
            continue;
        }
        if g.use_weight {
            conn[nb as usize] += w;
        }
        touched.push(nb);
        let p = g.centroid[nb as usize];
        let d2 = (0..3).map(|k| (p[k] - centre[k]) * (p[k] - centre[k])).sum::<f32>();
        let conn_q = (conn[nb as usize] * 16.0).min(1.0e9) as u32;
        heap.push(((u64::from(conn_q) << 32) | u64::from(u32::MAX - d2.to_bits().min(u32::MAX - 1)), Reverse(nb)));
    }
}

/// Greedy region growing: regions of about `target` nodes, in creation order.
pub fn region_grow(g: &Grow, admit: &mut dyn Admit) -> Vec<Vec<u32>> {
    let n = g.centroid.len();
    let mut assigned = vec![false; n];
    let mut conn = vec![0f32; n];
    let mut touched: Vec<u32> = Vec::new();
    let mut heap: BinaryHeap<(u64, Reverse<u32>)> = BinaryHeap::new();
    let mut out = Vec::new();
    let mut ptr = 0usize;
    loop {
        while ptr < n && assigned[g.order[ptr] as usize] {
            ptr += 1;
        }
        if ptr == n {
            break;
        }
        let seed = g.order[ptr];
        admit.reset();
        let admitted = admit.admit(seed);
        debug_assert!(admitted, "a single node always fits an empty region");
        heap.clear();
        let mut region = Region { members: Vec::new(), sum: [0.0; 3] };
        add_node(seed, g, &mut assigned, &mut conn, &mut touched, &mut heap, &mut region);
        let mut jump = ptr + 1;
        while region.members.len() < g.target {
            let Some((_, Reverse(node))) = heap.pop() else {
                if region.members.len() >= g.min_fill {
                    break;
                }
                let found = loop {
                    while jump < n && assigned[g.order[jump] as usize] {
                        jump += 1;
                    }
                    if jump == n {
                        break None;
                    }
                    let c = g.order[jump];
                    jump += 1;
                    if admit.admit(c) {
                        break Some(c);
                    }
                };
                match found {
                    Some(c) => {
                        add_node(c, g, &mut assigned, &mut conn, &mut touched, &mut heap, &mut region);
                        continue;
                    }
                    None => break,
                }
            };
            if assigned[node as usize] || !admit.admit(node) {
                continue;
            }
            add_node(node, g, &mut assigned, &mut conn, &mut touched, &mut heap, &mut region);
        }
        for t in touched.drain(..) {
            conn[t as usize] = 0.0;
        }
        out.push(region.members);
    }
    out
}

/// Accepts nodes always: cluster grouping has no per-region limits.
pub struct AdmitAll;

impl Admit for AdmitAll {
    fn reset(&mut self) {}
    fn admit(&mut self, _: u32) -> bool {
        true
    }
}

/// Limits of one cluster: triangles, distinct vertices and distinct bones.
pub struct TriAdmit<'a> {
    tris: &'a [[u32; 3]],
    verts: &'a [Vertex],
    skinned: bool,
    vstamp: Vec<u32>,
    bstamp: Vec<u32>,
    stamp: u32,
    vcount: usize,
    bcount: usize,
    tcount: usize,
    max_tris: usize,
}

impl<'a> TriAdmit<'a> {
    pub fn new(tris: &'a [[u32; 3]], verts: &'a [Vertex], skinned: bool, max_tris: usize) -> Self {
        let joints = if skinned { verts.iter().flat_map(|v| (0..4).filter(|&k| v.weights[k] > 0.0).map(|k| v.joints[k] as usize + 1)).max().unwrap_or(0) } else { 0 };
        Self { tris, verts, skinned, vstamp: vec![0; verts.len()], bstamp: vec![0; joints], stamp: 0, vcount: 0, bcount: 0, tcount: 0, max_tris }
    }

    fn bones_of(&self, v: u32, out: &mut Vec<u32>) {
        let vert = &self.verts[v as usize];
        for k in 0..4 {
            if vert.weights[k] > 0.0 && !out.contains(&vert.joints[k]) {
                out.push(vert.joints[k]);
            }
        }
    }
}

impl Admit for TriAdmit<'_> {
    fn reset(&mut self) {
        self.stamp += 1;
        self.vcount = 0;
        self.bcount = 0;
        self.tcount = 0;
    }

    fn admit(&mut self, t: u32) -> bool {
        if self.tcount >= self.max_tris {
            return false;
        }
        let tri = self.tris[t as usize];
        let mut new_v: Vec<u32> = Vec::with_capacity(3);
        for v in tri {
            if self.vstamp[v as usize] != self.stamp && !new_v.contains(&v) {
                new_v.push(v);
            }
        }
        let mut new_b: Vec<u32> = Vec::new();
        if self.skinned {
            let mut all = Vec::with_capacity(12);
            for v in tri {
                self.bones_of(v, &mut all);
            }
            new_b = all.into_iter().filter(|&b| self.bstamp[b as usize] != self.stamp).collect();
        }
        if self.vcount + new_v.len() > CLUSTER_MAX_VERTS || self.bcount + new_b.len() > CLUSTER_MAX_BONES {
            return false;
        }
        for v in &new_v {
            self.vstamp[*v as usize] = self.stamp;
        }
        for b in &new_b {
            self.bstamp[*b as usize] = self.stamp;
        }
        self.vcount += new_v.len();
        self.bcount += new_b.len();
        self.tcount += 1;
        true
    }
}

/// Partition triangles into clusters of at most 128 triangles, 256 vertices
/// and 16 bones, as compact and connected as the mesh allows. Returns the
/// triangle indices of each cluster.
pub fn partition_triangles(tris: &[[u32; 3]], verts: &[Vertex], point: &[u32], skinned: bool, target: usize) -> Vec<Vec<u32>> {
    let n = tris.len();
    if n == 0 {
        return Vec::new();
    }
    let target = target.clamp(1, CLUSTER_MAX_TRIS);
    let centroid: Vec<[f32; 3]> = tris
        .iter()
        .map(|t| {
            let p = t.map(|v| verts[v as usize].position);
            std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0)
        })
        .collect();
    let order = morton_order(&centroid);
    let adj = triangle_adjacency(tris, point);
    let regions_n = n.div_ceil(target);
    let size = n.div_ceil(regions_n).min(CLUSTER_MAX_TRIS);
    let mut admit = TriAdmit::new(tris, verts, skinned, size);
    let grow = Grow { centroid: &centroid, order: &order, adj: &adj, use_weight: false, target: size, min_fill: size / 2 };
    let regions = region_grow(&grow, &mut admit);

    let mut cluster_of = vec![0u32; n];
    let mut members: Vec<Vec<u32>> = regions;
    for (c, m) in members.iter().enumerate() {
        for &t in m {
            cluster_of[t as usize] = c as u32;
        }
    }

    // Fold fragments into the neighbour they share the most edges with.
    let mut check = TriAdmit::new(tris, verts, skinned, CLUSTER_MAX_TRIS);
    let mut by_size: Vec<u32> = (0..members.len() as u32).collect();
    by_size.sort_by_key(|&c| (members[c as usize].len(), c));
    for c in by_size {
        let c = c as usize;
        if members[c].is_empty() || members[c].len() >= size / 3 {
            continue;
        }
        let mut tally: Vec<(u32, u32)> = Vec::new();
        for &t in &members[c] {
            for (nb, _) in adj.neighbours(t) {
                let o = cluster_of[nb as usize];
                if o as usize == c {
                    continue;
                }
                match tally.iter_mut().find(|e| e.0 == o) {
                    Some(e) => e.1 += 1,
                    None => tally.push((o, 1)),
                }
            }
        }
        tally.sort_by_key(|&(o, k)| (Reverse(k), o));
        for (o, _) in tally {
            let o = o as usize;
            if members[o].len() + members[c].len() > CLUSTER_MAX_TRIS {
                continue;
            }
            check.reset();
            if members[o].iter().chain(&members[c]).all(|&t| check.admit(t)) {
                let moved = std::mem::take(&mut members[c]);
                for &t in &moved {
                    cluster_of[t as usize] = o as u32;
                }
                members[o].extend(moved);
                break;
            }
        }
    }

    // Boundary refinement: a triangle hanging on its cluster by one edge (or
    // none) moves to the neighbour it touches more. Skipped for skinned meshes
    // (the bone limit is not tracked incrementally).
    if !skinned {
        let mut sizes: Vec<usize> = members.iter().map(Vec::len).collect();
        let mut vcount: Vec<usize> = Vec::with_capacity(members.len());
        for m in &members {
            check.reset();
            for &t in m {
                check.admit(t);
            }
            vcount.push(check.vcount);
        }
        for _ in 0..3 {
            let mut moved = 0;
            for t in 0..n {
                let c = cluster_of[t] as usize;
                let (mut own, mut best, mut best_n) = (0u32, u32::MAX, 0u32);
                let mut tally: Vec<(u32, u32)> = Vec::new();
                for (nb, _) in adj.neighbours(t as u32) {
                    let o = cluster_of[nb as usize];
                    if o as usize == c {
                        own += 1;
                    } else {
                        match tally.iter_mut().find(|e| e.0 == o) {
                            Some(e) => e.1 += 1,
                            None => tally.push((o, 1)),
                        }
                    }
                }
                for (o, k) in tally {
                    if k > best_n || (k == best_n && o < best) {
                        best = o;
                        best_n = k;
                    }
                }
                if best == u32::MAX || sizes[c] <= 1 || own > 1 || best_n < own.max(1) + u32::from(own == 1) {
                    continue;
                }
                let b = best as usize;
                if sizes[b] >= CLUSTER_MAX_TRIS || vcount[b] + 3 > CLUSTER_MAX_VERTS {
                    continue;
                }
                cluster_of[t] = best;
                sizes[c] -= 1;
                sizes[b] += 1;
                vcount[b] += 3;
                moved += 1;
            }
            if moved == 0 {
                break;
            }
        }
        members = vec![Vec::new(); members.len()];
        for t in 0..n {
            members[cluster_of[t] as usize].push(t as u32);
        }
    } else {
        for m in &mut members {
            m.sort_unstable();
        }
    }

    // Safety net: whatever the passes did, every cluster obeys the limits.
    let mut out = Vec::with_capacity(members.len());
    for m in members {
        if m.is_empty() {
            continue;
        }
        check.reset();
        let mut cur: Vec<u32> = Vec::new();
        for t in m {
            if !check.admit(t) {
                out.push(std::mem::take(&mut cur));
                check.reset();
                check.admit(t);
            }
            cur.push(t);
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

/// Group `centroids.len()` clusters into groups of about `target`, linked by
/// shared edges (`adj` weights), at most `max` per group.
pub fn group_clusters(centroids: &[[f32; 3]], adj: &Csr, target: usize, min: usize, max: usize) -> Vec<Vec<u32>> {
    let n = centroids.len();
    if n == 0 {
        return Vec::new();
    }
    let groups_n = n.div_ceil(target.max(1));
    let size = n.div_ceil(groups_n).clamp(1, max);
    let order = morton_order(centroids);
    let grow = Grow { centroid: centroids, order: &order, adj, use_weight: true, target: size, min_fill: min.min(size) };
    region_grow(&grow, &mut AdmitAll)
}
