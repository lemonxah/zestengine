//! Mesh simplification for generated LODs: quadric-error edge collapse
//! (Garland & Heckbert). Open borders, which include UV and normal seams in
//! an unwelded mesh, are locked so neighbouring charts never crack apart, and
//! collapses that would flip a triangle are refused. The surviving vertex
//! keeps its attributes (UVs, skin weights), so skinned meshes stay skinned.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use zestengine_render::MeshData;
use glam::{DMat4, DVec3, DVec4, Vec3};

/// A fast multiply-rotate hasher (FxHash) for the builder's integer-keyed maps.
/// The default SipHash is the slowest part of a group's point numbering; nothing
/// here depends on map iteration order (keys are sorted before use), so the
/// hasher does not change any output.
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

impl std::hash::Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
    fn write_u16(&mut self, n: u16) {
        self.write_u64(u64::from(n));
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type FxBuild = std::hash::BuildHasherDefault<FxHasher>;
pub(crate) type FxMap<K, V> = HashMap<K, V, FxBuild>;

#[derive(Clone, Copy)]
struct Quadric(DMat4);

impl Default for Quadric {
    /// The empty quadric (no planes, zero error everywhere). Not derived: `DMat4::default()` is the
    /// IDENTITY, which put `|v|^2 + 1` into every vertex's error, so collapse costs measured distance
    /// from the origin instead of from the surface.
    fn default() -> Self {
        Quadric(DMat4::ZERO)
    }
}

impl Quadric {
    fn plane(n: DVec3, d: f64, weight: f64) -> Self {
        let p = DVec4::new(n.x, n.y, n.z, d);
        Quadric(DMat4::from_cols(p * p.x, p * p.y, p * p.z, p * p.w) * weight)
    }
    fn error(&self, v: DVec3) -> f64 {
        let p = DVec4::new(v.x, v.y, v.z, 1.0);
        p.dot(self.0 * p).max(0.0)
    }
    fn add(&self, o: &Quadric) -> Quadric {
        Quadric(self.0 + o.0)
    }
}

#[derive(PartialEq)]
struct Candidate {
    error: f64,
    a: u32,
    b: u32,
    /// Collapse target: 0 = a's position, 1 = b's, 2 = midpoint.
    at: u8,
    stamp: (u32, u32),
}

impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        o.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Simplify `mesh` down to about `keep` (0..1) of its triangles.
pub fn simplify(mesh: &MeshData, keep: f32) -> MeshData {
    simplify_with_provenance(mesh, keep).0
}

/// [`simplify`] plus, for every output vertex, the index of the source vertex
/// it is (a collapse keeps one of its two endpoints, so each survivor keeps its
/// identity). Per-vertex data that indexes the source mesh, such as blendshape
/// deltas, is carried over with [`remap_morph_deltas`].
pub fn simplify_with_provenance(mesh: &MeshData, keep: f32) -> (MeshData, Vec<u32>) {
    let tri_count = mesh.indices.len() / 3;
    let target = ((tri_count as f32 * keep.clamp(0.0, 1.0)).round() as usize).max(4);
    if tri_count <= target || mesh.vertices.is_empty() {
        let same = MeshData { vertices: mesh.vertices.clone(), indices: mesh.indices.clone(), has_lightmap_uv: mesh.has_lightmap_uv };
        return (same, (0..mesh.vertices.len() as u32).collect());
    }
    let mut pos: Vec<DVec3> = mesh.vertices.iter().map(|v| Vec3::from(v.position).as_dvec3()).collect();
    let mut tris: Vec<[u32; 3]> = mesh.indices.as_chunks::<3>().0.iter().map(|t| [t[0], t[1], t[2]]).collect();
    let mut alive = vec![true; tris.len()];
    let n = pos.len();

    // Border edges (one triangle) lock their vertices: in an unwelded mesh that
    // covers both real holes and attribute seams.
    let mut edge_use: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &tris {
        for k in 0..3 {
            let (a, b) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
            *edge_use.entry((a, b)).or_default() += 1;
        }
    }
    let mut locked = vec![false; n];
    for ((a, b), uses) in &edge_use {
        if *uses == 1 {
            locked[*a as usize] = true;
            locked[*b as usize] = true;
        }
    }
    let mut faces_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (f, t) in tris.iter().enumerate() {
        for &v in t {
            faces_of[v as usize].push(f as u32);
        }
    }
    let mut quad = vec![Quadric::default(); n];
    for t in &tris {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let cross = (b - a).cross(c - a);
        let area = cross.length();
        if area < 1e-18 {
            continue;
        }
        let nrm = cross / area;
        let q = Quadric::plane(nrm, -nrm.dot(a), area);
        for &v in t {
            quad[v as usize] = quad[v as usize].add(&q);
        }
    }
    let mut version = vec![0u32; n];
    let mut removed = vec![false; n];

    let candidate = |a: u32, b: u32, pos: &[DVec3], quad: &[Quadric], locked: &[bool], version: &[u32]| -> Option<Candidate> {
        let (la, lb) = (locked[a as usize], locked[b as usize]);
        if la && lb {
            return None;
        }
        let q = quad[a as usize].add(&quad[b as usize]);
        let (pa, pb) = (pos[a as usize], pos[b as usize]);
        let options: &[(u8, DVec3)] = if la {
            &[(0, pa)]
        } else if lb {
            &[(1, pb)]
        } else {
            &[(0, pa), (1, pb), (2, (pa + pb) * 0.5)]
        };
        let (at, err) = options.iter().map(|(k, p)| (*k, q.error(*p))).min_by(|x, y| x.1.total_cmp(&y.1))?;
        Some(Candidate { error: err, a, b, at, stamp: (version[a as usize], version[b as usize]) })
    };
    let mut heap = BinaryHeap::new();
    for (a, b) in edge_use.keys() {
        if let Some(c) = candidate(*a, *b, &pos, &quad, &locked, &version) {
            heap.push(c);
        }
    }

    let mut live = tri_count;
    while live > target {
        let Some(c) = heap.pop() else { break };
        let (a, b) = (c.a as usize, c.b as usize);
        if removed[a] || removed[b] || (version[a], version[b]) != c.stamp {
            continue;
        }
        // Keep `keep_v`, drop `gone`; the kept vertex moves to the target.
        let (keep_v, gone, target_pos) = match c.at {
            0 => (a, b, pos[a]),
            1 => (b, a, pos[b]),
            _ => (a, b, (pos[a] + pos[b]) * 0.5),
        };
        // Refuse collapses that flip or squash a surviving triangle.
        let mut flips = false;
        for &f in faces_of[keep_v].iter().chain(&faces_of[gone]) {
            let f = f as usize;
            if !alive[f] {
                continue;
            }
            let t = tris[f];
            if t.contains(&(keep_v as u32)) && t.contains(&(gone as u32)) {
                continue;
            }
            let before = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
            let moved: Vec<DVec3> = t
                .iter()
                .map(|&v| if v as usize == keep_v || v as usize == gone { target_pos } else { pos[v as usize] })
                .collect();
            let after = (moved[1] - moved[0]).cross(moved[2] - moved[0]);
            if before.dot(after) <= 0.0 || after.length_squared() < before.length_squared() * 1e-6 {
                flips = true;
                break;
            }
        }
        if flips {
            continue;
        }
        pos[keep_v] = target_pos;
        quad[keep_v] = quad[keep_v].add(&quad[gone]);
        removed[gone] = true;
        version[keep_v] += 1;
        let moved_faces = std::mem::take(&mut faces_of[gone]);
        for f in moved_faces {
            let fi = f as usize;
            if !alive[fi] {
                continue;
            }
            let t = &mut tris[fi];
            if t.contains(&(keep_v as u32)) {
                alive[fi] = false;
                live -= 1;
                continue;
            }
            for v in t.iter_mut() {
                if *v as usize == gone {
                    *v = keep_v as u32;
                }
            }
            faces_of[keep_v].push(f);
        }
        faces_of[keep_v].retain(|f| alive[*f as usize]);
        // Re-queue the kept vertex's edges.
        let mut neighbours: Vec<u32> = faces_of[keep_v].iter().flat_map(|f| tris[*f as usize]).filter(|v| *v as usize != keep_v).collect();
        neighbours.sort_unstable();
        neighbours.dedup();
        for nb in neighbours {
            if let Some(c) = candidate(keep_v as u32, nb, &pos, &quad, &locked, &version) {
                heap.push(c);
            }
        }
    }

    // Compact the surviving vertices and triangles.
    let mut remap = vec![u32::MAX; n];
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut provenance = Vec::new();
    for (f, t) in tris.iter().enumerate() {
        if !alive[f] {
            continue;
        }
        for &v in t {
            let v = v as usize;
            if remap[v] == u32::MAX {
                remap[v] = vertices.len() as u32;
                provenance.push(v as u32);
                let mut vert = mesh.vertices[v];
                vert.position = pos[v].as_vec3().to_array();
                vertices.push(vert);
            }
            indices.push(remap[v]);
        }
    }
    (MeshData { vertices, indices, has_lightmap_uv: mesh.has_lightmap_uv }, provenance)
}

/// Re-index sparse blendshape deltas (`(source vertex, d position, d normal)`)
/// onto a simplified mesh using the `provenance` from
/// [`simplify_with_provenance`]. Deltas of collapsed vertices are dropped.
pub fn remap_morph_deltas(deltas: &[(u32, [f32; 3], [f32; 3])], provenance: &[u32]) -> Vec<(u32, [f32; 3], [f32; 3])> {
    let mut new_of: HashMap<u32, u32> = HashMap::with_capacity(provenance.len());
    for (new, &old) in provenance.iter().enumerate() {
        new_of.insert(old, new as u32);
    }
    deltas.iter().filter_map(|&(v, dp, dn)| new_of.get(&v).map(|&n| (n, dp, dn))).collect()
}

// ---------------------------------------------------------------------------
// Group simplification with a pinned outer boundary (GEO-3 cluster DAG)
// ---------------------------------------------------------------------------

/// One group of triangles for [`simplify_group`]. Vertices are indexed in the
/// caller's (source) vertex array; `point` welds vertices that share a
/// position (UV and normal seams split a position into several vertices), so
/// the topology is that of the welded surface while every attribute stays
/// the source's.
pub struct GroupSimplify<'a> {
    pub vertices: &'a [zestengine_render::Vertex],
    /// Welded point id per vertex (equal positions share an id).
    pub point: &'a [u32],
    /// Triangles as source vertex indices.
    pub tris: &'a [[u32; 3]],
    /// Sorted point ids that must survive: the group's outer boundary.
    pub locked: &'a [u32],
    /// Triangle count to stop at.
    pub target_tris: usize,
    /// Per-point (not per-vertex) multiplier >= 1 on the cost of removing the
    /// point, to keep detail where blendshapes act.
    pub point_cost: Option<&'a [f32]>,
    /// Strength of the penalty on collapsing across a skin-weight
    /// discontinuity (0 = off); in units of edge length per unit weight change.
    pub skin_penalty: f32,
    /// Strength of the penalty on collapsing across a vertex-colour change (0 =
    /// off, the byte-identical behaviour for meshes without colours); in units of
    /// edge length per unit of the largest RGBA channel difference.
    pub colour_penalty: f32,
    /// Let a collapse proceed when a surviving corner at the removed point has no partner on the kept
    /// point (a UV / normal seam or a hard edge that does not continue onto the kept end): the corner is
    /// re-pointed at the most similar corner vertex of the kept point, trading an attribute jump (never a
    /// crack: positions are welded) for the ability to keep coarsening. Meshes of disconnected pieces,
    /// hair cards and faceted parts stall without it; the builder retries a group with it only when the
    /// strict pass cannot reach the target.
    pub relax_seams: bool,
    /// Sorted points of the group's outline (a subset of `locked`): whole pieces touching one are never
    /// pruned (a neighbouring group still draws them). Other locked points (non-manifold seams, which are
    /// only kept so the simplifier never reasons about them) do not stop a whole piece from being pruned.
    pub pinned: &'a [u32],
}

/// Result of [`simplify_group`].
pub struct GroupSimplified {
    /// Surviving triangles, as source vertex indices.
    pub tris: Vec<[u32; 3]>,
    /// Largest distance from a removed point to the simplified surface
    /// (object space, rounded up).
    pub error: f32,
}

/// Squared distance from `p` to triangle `abc` (closest-point method).
fn point_triangle_dist2(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length_squared();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length_squared();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length_squared();
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    (p - (a + ab * v + ac * w)).length_squared()
}

/// How far (L1) two vertices' skin weights are apart; 0 = same bones and weights.
/// Largest RGBA channel difference of two vertices' colours (0..=1).
pub(crate) fn colour_distance(a: &zestengine_render::Vertex, b: &zestengine_render::Vertex) -> f32 {
    (0..4).map(|k| (a.color[k] - b.color[k]).abs()).fold(0.0, f32::max)
}

fn skin_distance(a: &zestengine_render::Vertex, b: &zestengine_render::Vertex) -> f32 {
    let mut d = 0.0;
    for i in 0..4 {
        if a.weights[i] > 0.0 {
            let other = (0..4).find(|&j| b.weights[j] > 0.0 && b.joints[j] == a.joints[i]).map_or(0.0, |j| b.weights[j]);
            d += (a.weights[i] - other).abs();
        }
    }
    for j in 0..4 {
        if b.weights[j] > 0.0 && !(0..4).any(|i| a.weights[i] > 0.0 && a.joints[i] == b.joints[j]) {
            d += b.weights[j];
        }
    }
    d
}

#[derive(PartialEq)]
struct GroupCandidate {
    cost: f64,
    gone: u32,
    keep: u32,
    stamp: (u32, u32),
}

impl Eq for GroupCandidate {}
impl Ord for GroupCandidate {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap on cost; ties broken by indices so the result is deterministic.
        o.cost.total_cmp(&self.cost).then_with(|| (o.gone, o.keep).cmp(&(self.gone, self.keep)))
    }
}
impl PartialOrd for GroupCandidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Quadric-error simplification of one group of triangles whose outer
/// boundary must stay intact, for the cluster DAG builder.
///
/// Differences from [`simplify`]:
/// - Topology is that of the position-welded surface (`point`), so UV and
///   normal seams do not block collapses; a collapse runs along the seam on
///   both sides at once, each side keeping its own attribute set.
/// - Collapses move an edge's removed end onto its kept end (positions never
///   change), so every surviving vertex is a source vertex: attributes, skin
///   weights and blendshape deltas follow it unchanged.
/// - `locked` points are never removed, and with them every edge between two
///   locked points stays: neighbouring groups simplified separately (even at
///   different levels) still meet along the same edges, with no cracks.
/// - The link condition is enforced, so the surface stays manifold, and flips
///   are refused.
/// - Open mesh borders are not pinned: a border point may only slide along the
///   border onto another border point (never inward), and perpendicular penalty
///   planes make that cost what it costs the outline, so a plane can shrink to a
///   few triangles without eroding its edge.
/// - The reported error is the true largest distance of a removed point to the
///   result, not the quadric estimate.
pub fn simplify_group(g: &GroupSimplify) -> GroupSimplified {
    let target = g.target_tris.max(1);
    let keep_all = || GroupSimplified { tris: g.tris.to_vec(), error: 0.0 };
    if g.tris.len() <= target {
        return keep_all();
    }
    // Local point numbering in first-use order.
    let mut local_of: FxMap<u32, u32> = FxMap::with_capacity_and_hasher(g.tris.len(), FxBuild::default());
    let mut pts: Vec<u32> = Vec::new();
    let mut rep: Vec<u32> = Vec::new();
    let mut tp: Vec<[u32; 3]> = Vec::with_capacity(g.tris.len());
    for t in g.tris {
        let mut l = [0u32; 3];
        for k in 0..3 {
            let p = g.point[t[k] as usize];
            let next = pts.len() as u32;
            let id = *local_of.entry(p).or_insert(next);
            if id == next {
                pts.push(p);
                rep.push(t[k]);
            }
            l[k] = id;
        }
        tp.push(l);
    }
    let n = pts.len();
    let pos: Vec<DVec3> = rep.iter().map(|&v| Vec3::from(g.vertices[v as usize].position).as_dvec3()).collect();
    let locked: Vec<bool> = pts.iter().map(|p| g.locked.binary_search(p).is_ok()).collect();
    let mut tri: Vec<[u32; 3]> = g.tris.to_vec();
    let mut alive: Vec<bool> = tp.iter().map(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]).collect();
    let mut live = alive.iter().filter(|a| **a).count();
    let mut faces_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (f, t) in tp.iter().enumerate() {
        if alive[f] {
            for &p in t {
                faces_of[p as usize].push(f as u32);
            }
        }
    }

    // Connected pieces of the group (on the welded points), for the error of a piece that simplifies away.
    let comp: Vec<u32> = {
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn root(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        for (f, t) in tp.iter().enumerate() {
            if alive[f] {
                for k in 1..3 {
                    let (a, b) = (root(&mut parent, t[0]), root(&mut parent, t[k]));
                    if a != b {
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            }
        }
        (0..n as u32).map(|p| root(&mut parent, p)).collect()
    };

    // Quadrics (area weighted) and the area they sum, so cost is a mean squared distance.
    let mut quad = vec![Quadric::default(); n];
    let mut area = vec![0.0f64; n];
    let mut edge_use: FxMap<(u32, u32), u32> = FxMap::with_capacity_and_hasher(g.tris.len() * 2, FxBuild::default());
    for (f, t) in tp.iter().enumerate() {
        if !alive[f] {
            continue;
        }
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let cross = (b - a).cross(c - a);
        let ar = cross.length();
        for k in 0..3 {
            let (x, y) = (t[k], t[(k + 1) % 3]);
            *edge_use.entry((x.min(y), x.max(y))).or_default() += 1;
        }
        if ar < 1e-18 {
            continue;
        }
        let nrm = cross / ar;
        let q = Quadric::plane(nrm, -nrm.dot(a), ar);
        for &p in t {
            quad[p as usize] = quad[p as usize].add(&q);
            area[p as usize] += ar;
        }
    }
    // Points on an open border (a mesh edge, or the group's outline): a border
    // point may only slide along the border onto another border point, so the
    // outline of the surface never moves inward and the cuts of neighbouring
    // groups keep meeting at the same border points.
    let mut border_pt = vec![false; n];
    for (&(a, b), &uses) in &edge_use {
        if uses == 1 {
            border_pt[a as usize] = true;
            border_pt[b as usize] = true;
        }
    }
    // Mesh borders (edges with no neighbour and no pinned end pair) get penalty planes.
    for (f, t) in tp.iter().enumerate() {
        if !alive[f] {
            continue;
        }
        let nrm = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]).normalize_or_zero();
        for k in 0..3 {
            let (x, y) = (t[k], t[(k + 1) % 3]);
            if edge_use[&(x.min(y), x.max(y))] != 1 || (locked[x as usize] && locked[y as usize]) {
                continue;
            }
            let (pa, pb) = (pos[x as usize], pos[y as usize]);
            let m = (pb - pa).cross(nrm).normalize_or_zero();
            let w = (pb - pa).length_squared() * 2.0;
            if m == DVec3::ZERO || w <= 0.0 {
                continue;
            }
            let q = Quadric::plane(m, -m.dot(pa), w);
            for p in [x, y] {
                quad[p as usize] = quad[p as usize].add(&q);
                area[p as usize] += w;
            }
        }
    }

    let cost_of = |gone: usize, keep: usize, quad: &[Quadric], area: &[f64]| -> f64 {
        let q = quad[gone].add(&quad[keep]);
        let a = (area[gone] + area[keep]).max(1e-30);
        let mut c = q.error(pos[keep]) / a;
        if let Some(pc) = g.point_cost {
            let s = pc[pts[gone] as usize].max(pc[pts[keep] as usize]).max(1.0) as f64;
            c *= s;
        }
        if g.skin_penalty > 0.0 {
            let sd = skin_distance(&g.vertices[rep[gone] as usize], &g.vertices[rep[keep] as usize]) as f64;
            let p = g.skin_penalty as f64 * sd * (pos[gone] - pos[keep]).length();
            c += p * p;
        }
        if g.colour_penalty > 0.0 {
            let cd = colour_distance(&g.vertices[rep[gone] as usize], &g.vertices[rep[keep] as usize]) as f64;
            let p = g.colour_penalty as f64 * cd * (pos[gone] - pos[keep]).length();
            c += p * p;
        }
        c
    };
    let mut version = vec![0u32; n];
    let mut removed = vec![false; n];
    let mut heap = BinaryHeap::new();
    let mut edges: Vec<(u32, u32)> = edge_use.keys().copied().collect();
    edges.sort_unstable();
    for (a, b) in edges {
        for (gone, keep) in [(a, b), (b, a)] {
            if !locked[gone as usize] {
                heap.push(GroupCandidate { cost: cost_of(gone as usize, keep as usize, &quad, &area), gone, keep, stamp: (0, 0) });
            }
        }
    }

    let corner = |tri: &[[u32; 3]], tp: &[[u32; 3]], f: usize, p: u32| -> u32 {
        let i = tp[f].iter().position(|&x| x == p).unwrap();
        tri[f][i]
    };
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let (mut nb_g, mut nb_k): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    // Reused across collapses (one allocation per group, not per collapse).
    let mut around: Vec<u32> = Vec::new();
    while live > target {
        let Some(c) = heap.pop() else { break };
        let (gone, keep) = (c.gone as usize, c.keep as usize);
        if removed[gone] || removed[keep] || (version[gone], version[keep]) != c.stamp {
            continue;
        }
        // Corner pairs along the collapsing edge: each side of a seam maps its
        // vertex at `gone` to its vertex at `keep`.
        pairs.clear();
        let mut ok = true;
        let mut dying = 0usize;
        for &f in &faces_of[gone] {
            let f = f as usize;
            if !alive[f] || !tp[f].contains(&c.keep) {
                continue;
            }
            let (vg, vk) = (corner(&tri, &tp, f, c.gone), corner(&tri, &tp, f, c.keep));
            match pairs.iter().find(|p| p.0 == vg) {
                Some(p) if p.1 != vk => ok = false,
                Some(_) => {}
                None => pairs.push((vg, vk)),
            }
            dying += 1;
        }
        if !ok || dying == 0 {
            continue;
        }
        if border_pt[gone] && !(border_pt[keep] && dying == 1) {
            continue;
        }
        // Every surviving corner at `gone` needs a partner, or its attribute set would jump (unless the
        // caller allows the jump: then the corner follows the most similar corner vertex at `keep`).
        for &f in &faces_of[gone] {
            let f = f as usize;
            if !alive[f] || tp[f].contains(&c.keep) {
                continue;
            }
            let vg = corner(&tri, &tp, f, c.gone);
            if pairs.iter().any(|p| p.0 == vg) {
                continue;
            }
            if !g.relax_seams {
                ok = false;
                break;
            }
            let a = &g.vertices[vg as usize];
            let mut best: Option<(f32, u32)> = None;
            for &kf in &faces_of[keep] {
                let kf = kf as usize;
                if !alive[kf] {
                    continue;
                }
                let vk = corner(&tri, &tp, kf, c.keep);
                let b = &g.vertices[vk as usize];
                let nd: f32 = (0..3).map(|k| a.normal[k] * b.normal[k]).sum();
                let ud = (a.uv[0] - b.uv[0]).abs() + (a.uv[1] - b.uv[1]).abs();
                let score = nd - ud;
                if best.is_none_or(|(s, v)| score > s || (score == s && vk < v)) {
                    best = Some((score, vk));
                }
            }
            match best {
                Some((_, vk)) => pairs.push((vg, vk)),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        // Link condition: the neighbours the two ends share are exactly the
        // apexes of the triangles on the edge.
        nb_g.clear();
        nb_k.clear();
        for &f in &faces_of[gone] {
            if alive[f as usize] {
                nb_g.extend(tp[f as usize].iter().copied().filter(|&p| p != c.gone && p != c.keep));
            }
        }
        for &f in &faces_of[keep] {
            if alive[f as usize] {
                nb_k.extend(tp[f as usize].iter().copied().filter(|&p| p != c.gone && p != c.keep));
            }
        }
        nb_g.sort_unstable();
        nb_g.dedup();
        nb_k.sort_unstable();
        nb_k.dedup();
        let shared = nb_g.iter().filter(|p| nb_k.binary_search(p).is_ok()).count();
        if shared != dying {
            continue;
        }
        // Two pinned points may already be joined by an edge outside the group;
        // an edge made between them here would be a second one on the same pair.
        if locked[keep] && nb_g.iter().any(|&p| locked[p as usize] && nb_k.binary_search(&p).is_err()) {
            continue;
        }
        // Refuse flips and slivers among the triangles that move.
        let mut flips = false;
        for &f in &faces_of[gone] {
            let f = f as usize;
            if !alive[f] || tp[f].contains(&c.keep) {
                continue;
            }
            let t = tp[f];
            let at = |p: u32, moved: bool| if moved && p == c.gone { pos[keep] } else { pos[p as usize] };
            let before = (at(t[1], false) - at(t[0], false)).cross(at(t[2], false) - at(t[0], false));
            let after = (at(t[1], true) - at(t[0], true)).cross(at(t[2], true) - at(t[0], true));
            if before.dot(after) <= 0.0 || after.length_squared() < before.length_squared() * 1e-6 {
                flips = true;
                break;
            }
        }
        if flips {
            continue;
        }
        // Commit.
        let faces = std::mem::take(&mut faces_of[gone]);
        for f in faces {
            let fi = f as usize;
            if !alive[fi] {
                continue;
            }
            if tp[fi].contains(&c.keep) {
                alive[fi] = false;
                live -= 1;
                continue;
            }
            let i = tp[fi].iter().position(|&x| x == c.gone).unwrap();
            let vg = tri[fi][i];
            tri[fi][i] = pairs.iter().find(|p| p.0 == vg).unwrap().1;
            tp[fi][i] = c.keep;
            faces_of[keep].push(f);
        }
        faces_of[keep].retain(|f| alive[*f as usize]);
        quad[keep] = quad[keep].add(&quad[gone]);
        area[keep] += area[gone];
        removed[gone] = true;
        version[keep] += 1;
        around.clear();
        around.extend(faces_of[keep].iter().flat_map(|f| tp[*f as usize]).filter(|&p| p != c.keep));
        around.sort_unstable();
        around.dedup();
        for &nb in &around {
            for (gn, kp) in [(c.keep, nb), (nb, c.keep)] {
                if !locked[gn as usize] {
                    heap.push(GroupCandidate {
                        cost: cost_of(gn as usize, kp as usize, &quad, &area),
                        gone: gn,
                        keep: kp,
                        stamp: (version[gn as usize], version[kp as usize]),
                    });
                }
            }
        }
    }

    // Collapses ran dry above the target (small disconnected parts: lash and hair cards, buttons, each
    // already down to its few smallest triangles): drop whole pieces, smallest first, skipping any that
    // touch the group outline (a neighbour group still draws them, so removing them here would crack the
    // cut) and any whose removal would take the group below half its target. The error charged below is
    // the piece's own size, so a piece only disappears at distances where it is sub-pixel.
    if g.relax_seams && live > target {
        let mut stats: FxMap<u32, (DVec3, DVec3, usize, bool)> = FxMap::default();
        for i in 0..n {
            if removed[i] {
                continue;
            }
            let e = stats.entry(comp[i]).or_insert((DVec3::MAX, DVec3::MIN, 0, false));
            e.0 = e.0.min(pos[i]);
            e.1 = e.1.max(pos[i]);
            e.3 |= g.pinned.binary_search(&pts[i]).is_ok();
        }
        for (f, t) in tp.iter().enumerate() {
            if alive[f]
                && let Some(e) = stats.get_mut(&comp[t[0] as usize])
            {
                e.2 += 1;
            }
        }
        let mut order: Vec<(f64, u32, usize)> = stats.iter().filter(|(_, e)| !e.3 && e.2 > 0).map(|(&c, e)| ((e.1 - e.0).length_squared(), c, e.2)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, c, count) in order {
            if live <= target {
                break;
            }
            if live - count < target / 2 {
                continue;
            }
            for (f, t) in tp.iter().enumerate() {
                if alive[f] && comp[t[0] as usize] == c {
                    alive[f] = false;
                }
            }
            for i in 0..n {
                if comp[i] == c {
                    removed[i] = true;
                }
            }
            live -= count;
        }
    }
    let out: Vec<[u32; 3]> = tri.iter().enumerate().filter(|(f, _)| alive[*f]).map(|(_, t)| *t).collect();
    // True deviation: the farthest any removed point lies from what is left. A piece that simplified
    // away entirely (a small disconnected part: a button, a lash card) is judged by its own size, the
    // most its disappearance changes the picture, not by how far the next surviving surface happens to
    // be (which once made a 2 mm piece cost 30 cm of error and pinned its whole group to fine levels).
    let corners: Vec<[DVec3; 3]> = out.iter().map(|t| t.map(|v| Vec3::from(g.vertices[v as usize].position).as_dvec3())).collect();
    let boxes: Vec<(DVec3, DVec3)> = corners.iter().map(|c| (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]))).collect();
    let mut survives = vec![false; n];
    for (f, t) in tp.iter().enumerate() {
        if alive[f] {
            survives[comp[t[0] as usize] as usize] = true;
        }
    }
    let mut piece: Vec<(DVec3, DVec3)> = vec![(DVec3::MAX, DVec3::MIN); n];
    for (i, &c) in comp.iter().enumerate() {
        if !survives[c as usize] {
            let b = &mut piece[c as usize];
            *b = (b.0.min(pos[i]), b.1.max(pos[i]));
        }
    }
    let mut worst = 0.0f64;
    for (i, gone) in removed.iter().enumerate() {
        if !*gone {
            continue;
        }
        let c = comp[i] as usize;
        if !survives[c] {
            let (lo, hi) = piece[c];
            worst = worst.max(((hi - lo) * 0.5).length_squared());
            continue;
        }
        let p = pos[i];
        let mut best = f64::MAX;
        for (k, c) in corners.iter().enumerate() {
            let (lo, hi) = boxes[k];
            let d = (lo - p).max(DVec3::ZERO).max(p - hi).length_squared();
            if d >= best {
                continue;
            }
            best = best.min(point_triangle_dist2(p, c[0], c[1], c[2]));
        }
        worst = worst.max(best);
    }
    let error = if worst.is_finite() { (worst.sqrt() * (1.0 + 1e-6)) as f32 } else { 0.0 };
    GroupSimplified { tris: out, error }
}

// ---------------------------------------------------------------------------
// Skin-aware simplification (GEO-2 skinned auto LODs)
// ---------------------------------------------------------------------------

/// Triangles whose skin weights differ by more than this (half the L1 distance
/// of their averaged influence sets, 0..1) are never merged by
/// [`simplify_skinned_with_provenance`]: a hard weight step (a rigid part
/// meeting another, the face against the neck, a finger against the palm) keeps
/// its outline, while a smooth blend across a joint stays free to simplify.
pub const SKIN_REGION_MAX_STEP: f32 = 0.2;

/// Position welding key: the simplifier's border logic works on vertex ids, so
/// regions are found on welded positions (seams must not split a region).
fn weld_key(p: [f32; 3]) -> (i64, i64, i64) {
    let q = |x: f32| (x as f64 * 1.0e5).round() as i64;
    (q(p[0]), q(p[1]), q(p[2]))
}

/// A triangle's averaged skin influences as (joint, weight) pairs, sorted by
/// joint, weights summing to 1 (empty for an unweighted triangle).
fn tri_influence(mesh: &MeshData, t: &[u32]) -> Vec<(u32, f32)> {
    let mut acc: Vec<(u32, f32)> = Vec::with_capacity(12);
    for &v in t {
        let v = &mesh.vertices[v as usize];
        for (j, w) in v.joints.iter().zip(v.weights) {
            if w <= 0.0 {
                continue;
            }
            match acc.iter_mut().find(|(k, _)| k == j) {
                Some(e) => e.1 += w,
                None => acc.push((*j, w)),
            }
        }
    }
    let sum: f32 = acc.iter().map(|e| e.1).sum();
    if sum > 0.0 {
        acc.iter_mut().for_each(|e| e.1 /= sum);
    }
    acc.sort_by_key(|e| e.0);
    acc
}

/// Half the L1 distance between two influence sets (0 = identical, 1 = disjoint).
fn influence_distance(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut d) = (0, 0, 0.0f32);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x.0 == y.0 => {
                d += (x.1 - y.1).abs();
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x.0 < y.0 => {
                d += x.1;
                i += 1;
            }
            (Some(_), Some(y)) => {
                d += y.1;
                j += 1;
            }
            (Some(x), None) => {
                d += x.1;
                i += 1;
            }
            (None, Some(y)) => {
                d += y.1;
                j += 1;
            }
            (None, None) => break,
        }
    }
    d * 0.5
}

/// Region id per triangle: triangles sharing a welded edge are in the same
/// region unless their skin influences differ by more than `max_step`.
pub fn skin_regions(mesh: &MeshData, max_step: f32) -> Vec<u32> {
    let tris: Vec<&[u32]> = mesh.indices.as_chunks::<3>().0.iter().map(|t| t.as_slice()).collect();
    let infl: Vec<Vec<(u32, f32)>> = tris.iter().map(|t| tri_influence(mesh, t)).collect();
    let mut parent: Vec<u32> = (0..tris.len() as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    /// One weld key: the quantised positions of an edge's two ends.
    type EdgeKey = ((i64, i64, i64), (i64, i64, i64));
    let mut edge_owner: HashMap<EdgeKey, u32> = HashMap::new();
    for (f, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (weld_key(mesh.vertices[t[k] as usize].position), weld_key(mesh.vertices[t[(k + 1) % 3] as usize].position));
            let key = if a <= b { (a, b) } else { (b, a) };
            match edge_owner.get(&key) {
                Some(&g) => {
                    if influence_distance(&infl[f], &infl[g as usize]) <= max_step {
                        let (ra, rb) = (find(&mut parent, f as u32), find(&mut parent, g));
                        if ra != rb {
                            parent[ra as usize] = rb;
                        }
                    }
                }
                None => {
                    edge_owner.insert(key, f as u32);
                }
            }
        }
    }
    let mut ids: HashMap<u32, u32> = HashMap::new();
    (0..tris.len() as u32)
        .map(|f| {
            let root = find(&mut parent, f);
            let next = ids.len() as u32;
            *ids.entry(root).or_insert(next)
        })
        .collect()
}

/// Skin-aware [`simplify_with_provenance`]: the mesh is cut into the skin
/// regions of [`skin_regions`] and each region is simplified on its own to the
/// same `keep`. A region border is an open border to the simplifier, so its
/// vertices are locked and no edge collapse ever crosses a hard bone-weight
/// step; smooth blends stay in one region and simplify normally. Vertices on a
/// region border are duplicated across the regions (same position and
/// attributes, so there are no cracks), hence the returned provenance (output
/// vertex -> source vertex) can name one source vertex several times: carry
/// per-vertex data with [`remap_morph_deltas_multi`], not
/// [`remap_morph_deltas`]. Bones, weights and every other attribute of a
/// surviving vertex are those of its source vertex.
pub fn simplify_skinned_with_provenance(mesh: &MeshData, keep: f32, max_step: f32) -> (MeshData, Vec<u32>) {
    let regions = skin_regions(mesh, max_step);
    let region_count = regions.iter().copied().max().map_or(0, |m| m as usize + 1);
    if region_count <= 1 {
        return simplify_with_provenance(mesh, keep);
    }
    let mut parts: Vec<(MeshData, Vec<u32>)> = Vec::with_capacity(region_count);
    {
        let mut local: Vec<HashMap<u32, u32>> = vec![HashMap::new(); region_count];
        let mut sub: Vec<(Vec<zestengine_render::Vertex>, Vec<u32>, Vec<u32>)> = vec![(Vec::new(), Vec::new(), Vec::new()); region_count];
        for (f, t) in mesh.indices.as_chunks::<3>().0.iter().enumerate() {
            let r = regions[f] as usize;
            for &v in t {
                let (verts, idx, origin) = &mut sub[r];
                let id = *local[r].entry(v).or_insert_with(|| {
                    verts.push(mesh.vertices[v as usize]);
                    origin.push(v);
                    verts.len() as u32 - 1
                });
                idx.push(id);
            }
        }
        for (vertices, indices, origin) in sub {
            parts.push((MeshData { vertices, indices, has_lightmap_uv: mesh.has_lightmap_uv }, origin));
        }
    }
    let mut out = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: mesh.has_lightmap_uv };
    let mut provenance = Vec::new();
    for (part, origin) in parts {
        let (simple, prov) = simplify_with_provenance(&part, keep);
        let base = out.vertices.len() as u32;
        out.indices.extend(simple.indices.iter().map(|i| i + base));
        out.vertices.extend(simple.vertices);
        provenance.extend(prov.iter().map(|&p| origin[p as usize]));
    }
    (out, provenance)
}

/// [`remap_morph_deltas`] for a provenance that can repeat a source vertex
/// (see [`simplify_skinned_with_provenance`]): every output copy of a source
/// vertex gets its delta, so a duplicated border vertex moves identically on
/// both sides.
pub fn remap_morph_deltas_multi(deltas: &[(u32, [f32; 3], [f32; 3])], provenance: &[u32]) -> Vec<(u32, [f32; 3], [f32; 3])> {
    let mut news_of: HashMap<u32, Vec<u32>> = HashMap::new();
    for (new, &old) in provenance.iter().enumerate() {
        news_of.entry(old).or_default().push(new as u32);
    }
    let mut out: Vec<_> = deltas
        .iter()
        .flat_map(|&(v, dp, dn)| news_of.get(&v).into_iter().flatten().map(move |&n| (n, dp, dn)))
        .collect();
    out.sort_by_key(|d| d.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(m: &MeshData) -> (Vec3, Vec3) {
        m.vertices.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), v| (lo.min(Vec3::from(v.position)), hi.max(Vec3::from(v.position))))
    }

    #[test]
    fn provenance_maps_survivors_to_source_vertices_and_carries_morphs() {
        let sphere = crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere);
        let (low, prov) = simplify_with_provenance(&sphere, 0.5);
        assert_eq!(prov.len(), low.vertices.len());
        for (v, &src) in low.vertices.iter().zip(&prov) {
            let o = &sphere.vertices[src as usize];
            assert_eq!(v.uv, o.uv, "survivors keep their source attributes");
        }
        // A delta on every source vertex survives exactly once per kept vertex.
        let deltas: Vec<_> = (0..sphere.vertices.len() as u32).map(|i| (i, [i as f32, 0.0, 0.0], [0.0; 3])).collect();
        let kept = remap_morph_deltas(&deltas, &prov);
        assert_eq!(kept.len(), low.vertices.len());
        for (new, dp, _) in &kept {
            assert_eq!(dp[0], prov[*new as usize] as f32, "delta follows its vertex");
        }
        // No-op simplification is the identity mapping.
        let (_, ident) = simplify_with_provenance(&sphere, 1.0);
        assert!(ident.iter().enumerate().all(|(i, &p)| p as usize == i));
    }

    #[test]
    fn a_sphere_halves_and_keeps_its_shape() {
        let sphere = crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere);
        let tris = sphere.indices.len() / 3;
        let low = simplify(&sphere, 0.5);
        let low_tris = low.indices.len() / 3;
        assert!(low_tris <= tris * 6 / 10 && low_tris >= tris / 4, "{tris} -> {low_tris}");
        let (a, b) = (bounds(&sphere), bounds(&low));
        assert!((a.0 - b.0).length() < 0.05 && (a.1 - b.1).length() < 0.05, "silhouette kept");
        for t in low.indices.as_chunks::<3>().0 {
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2], "no degenerate triangles");
        }
    }

    #[test]
    fn flat_grids_collapse_hard_but_keep_their_border() {
        let n = 20u32;
        let mut vertices = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                vertices.push(zestengine_render::Vertex { position: [x as f32, 0.0, y as f32], ..Default::default() });
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let i = y * (n + 1) + x;
                indices.extend([i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
            }
        }
        let grid = MeshData { vertices, indices, has_lightmap_uv: false };
        let low = simplify(&grid, 0.1);
        assert!(low.indices.len() / 3 < 300, "a flat plane loses most of its triangles: {}", low.indices.len() / 3);
        let (lo, hi) = bounds(&low);
        assert_eq!((lo, hi), (Vec3::ZERO, Vec3::new(n as f32, 0.0, n as f32)), "border locked");
        assert!(low.vertices.iter().all(|v| v.position[1].abs() < 1e-6), "stays flat");
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;
    use zestengine_render::Vertex;

    fn grid_tris(n: u32) -> (Vec<Vertex>, Vec<[u32; 3]>) {
        let mut vertices = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                vertices.push(Vertex { position: [x as f32, 0.02 * ((x * 7 + y * 3) % 5) as f32, y as f32], ..Default::default() });
            }
        }
        let mut tris = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let i = y * (n + 1) + x;
                tris.push([i, i + n + 1, i + 1]);
                tris.push([i + 1, i + n + 1, i + n + 2]);
            }
        }
        (vertices, tris)
    }

    fn area(v: &[Vertex], tris: &[[u32; 3]]) -> f64 {
        tris.iter()
            .map(|t| {
                let p = t.map(|i| v[i as usize].position);
                0.5 * ((p[1][0] - p[0][0]) * (p[2][2] - p[0][2]) - (p[2][0] - p[0][0]) * (p[1][2] - p[0][2])) as f64
            })
            .sum()
    }

    #[test]
    fn a_pinned_outline_keeps_its_area_and_edges() {
        let n = 24;
        let (vertices, tris) = grid_tris(n);
        let point: Vec<u32> = (0..vertices.len() as u32).collect();
        let on_ring = |v: u32| {
            let (x, y) = (v % (n + 1), v / (n + 1));
            x == 0 || y == 0 || x == n || y == n
        };
        let locked: Vec<u32> = (0..vertices.len() as u32).filter(|&v| on_ring(v)).collect();
        let out = simplify_group(&GroupSimplify { vertices: &vertices, point: &point, tris: &tris, locked: &locked, target_tris: 60, point_cost: None, skin_penalty: 0.0, colour_penalty: 0.0, relax_seams: false, pinned: &locked });
        assert!(out.tris.len() < 200, "{}", out.tris.len());
        assert!((area(&vertices, &out.tris) - area(&vertices, &tris)).abs() < 1e-3, "area {} -> {}", area(&vertices, &tris), area(&vertices, &out.tris));
        // Every ring edge is still there, once.
        let mut dir = std::collections::HashSet::new();
        for t in &out.tris {
            for k in 0..3 {
                assert!(dir.insert((t[k], t[(k + 1) % 3])), "duplicate directed edge");
            }
        }
        for (a, b) in tris.iter().flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3]))).filter(|(a, b)| on_ring(*a) && on_ring(*b)) {
            if !tris.iter().any(|t| (0..3).any(|k| t[k] == b && t[(k + 1) % 3] == a)) {
                assert!(dir.contains(&(a, b)), "ring edge {a}->{b} lost");
            }
        }
    }

    #[test]
    fn neighbouring_groups_simplified_apart_still_meet_exactly() {
        let n = 48u32;
        let (vertices, tris) = grid_tris(n);
        let point: Vec<u32> = (0..vertices.len() as u32).collect();
        let group_of = |t: &[u32; 3]| {
            let c = t.map(|v| (v % (n + 1), v / (n + 1)));
            u32::from(c[0].0 >= n / 2) + 2 * u32::from(c[0].1 >= n / 2)
        };
        let mut dir_owner = std::collections::HashMap::new();
        for t in &tris {
            for k in 0..3 {
                dir_owner.insert((t[k], t[(k + 1) % 3]), group_of(t));
            }
        }
        let mut all = Vec::new();
        for g in 0..4u32 {
            let mine: Vec<[u32; 3]> = tris.iter().filter(|t| group_of(t) == g).copied().collect();
            let mut locked = Vec::new();
            for t in &mine {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    if dir_owner.get(&(b, a)).is_some_and(|&o| o != g) {
                        locked.push(a);
                        locked.push(b);
                    }
                }
            }
            locked.sort_unstable();
            locked.dedup();
            let out = simplify_group(&GroupSimplify { vertices: &vertices, point: &point, tris: &mine, locked: &locked, target_tris: mine.len() / 4, point_cost: None, skin_penalty: 0.0, colour_penalty: 0.0, relax_seams: false, pinned: &locked });
            eprintln!("group {g}: {} -> {} tris, area {:.2} -> {:.2}, error {}", mine.len(), out.tris.len(), area(&vertices, &mine), area(&vertices, &out.tris), out.error);
            all.extend(out.tris);
        }
        let mut dir = std::collections::HashMap::<(u32, u32), u32>::new();
        for t in &all {
            for k in 0..3 {
                *dir.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(dir.values().all(|&c| c == 1), "overlap");
        let open = dir.keys().filter(|(a, b)| !dir.contains_key(&(*b, *a))).count();
        let ring = dir.keys().filter(|(a, b)| !dir.contains_key(&(*b, *a)) && [a, b].iter().all(|&&v| v % (n + 1) == 0 || v / (n + 1) == 0 || v % (n + 1) == n || v / (n + 1) == n)).count();
        eprintln!("open edges {open}, of which on the outer ring {ring}");
        assert!(open <= ring + 8, "cracks between groups: {open} open edges, {ring} on the ring");
    }
}

