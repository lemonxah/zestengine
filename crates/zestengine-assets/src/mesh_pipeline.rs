//! The mesh import pipeline: one ordered chain of passes over an intermediate
//! mesh, shared by every model format (FBX, glTF, ...), so a setting means the
//! same thing whichever file the mesh came from.
//!
//! The intermediate [`ImportMesh`] is a polygon soup: one [`Vertex`] per
//! polygon corner, faces of 3 or 4 corners, blendshape deltas keyed by corner.
//! Importers only convert their format into it; everything else lives here.
//!
//! Pass order ([`STAGE_ORDER`]):
//!
//! 1. **Triangulate** (skipped with Keep Quads; quads split on the shorter diagonal)
//! 2. **Remove degenerates** (zero-area faces, repeated corners)
//! 3. **Normals** (imported, calculated with the crease angle and weighting, or
//!    none; missing ones are calculated)
//! 4. **Tangents** (imported, calculated with MikkTSpace, legacy, or left alone)
//! 5. **Quantise** (Mesh Compression snaps positions, UVs, normals and tangents
//!    to a grid, so the weld below merges what lands on the same values)
//! 6. **Weld** (corners sharing every attribute become one vertex; this is also
//!    where the soup becomes an indexed mesh)
//! 7. **Lightmap UV** (unwrap UV1 when asked for and the file has none)
//! 8. **Optimise** (vertex-cache triangle order, then fetch-order vertices)
//! 9. **LOD** (the artifact keeps full detail: the LOD chain is built from it at
//!    load time by `auto_lod`, which caches its own result)
//!
//! After the passes, a forced 16 bit index format splits a mesh that has more
//! than 65535 vertices into pieces that fit ([`split_16bit`]).
//!
//! Unity's own per-mesh order inside `GenerateMeshData` is still an open
//! research item; this order follows its documented outline (triangulate,
//! normals, tangents, weld, secondary UVs, optimise) with degenerate removal up
//! front, since normals and tangents are meaningless on zero-area faces, and
//! with normals before tangents because tangents are orthogonalised to them.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use glam::{Vec2, Vec3};
use zestengine_render::{MeshData, Vertex};

use crate::{MeshCompression, ModelImport, MorphTarget, NormalWeighting, NormalsMode, TangentMode, UV_ALL};

/// One pass of the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Triangulate,
    RemoveDegenerates,
    Normals,
    Tangents,
    Quantise,
    Weld,
    LightmapUv,
    Optimise,
    Lod,
}

/// The order passes run in.
pub const STAGE_ORDER: [Stage; 9] = [
    Stage::Triangulate,
    Stage::RemoveDegenerates,
    Stage::Normals,
    Stage::Tangents,
    Stage::Quantise,
    Stage::Weld,
    Stage::LightmapUv,
    Stage::Optimise,
    Stage::Lod,
];

/// Which passes run and how, taken from a model's [`ModelImport`].
#[derive(Clone, Debug)]
pub struct Settings {
    pub keep_quads: bool,
    pub weld: bool,
    pub normals: NormalsMode,
    pub weighting: NormalWeighting,
    pub smoothing_angle: f32,
    pub tangents: TangentMode,
    pub lightmap_uv: bool,
    pub optimize: bool,
    /// Split meshes with more than 65535 vertices (forced 16 bit indices).
    pub split_16bit: bool,
    /// UV sets kept ([`UV_ALL`]).
    pub uv_mask: u8,
    pub compression: MeshCompression,
}

impl Settings {
    pub fn from_import(m: &ModelImport) -> Self {
        Self {
            keep_quads: m.keep_quads,
            weld: m.weld_vertices,
            normals: m.normals_mode(),
            weighting: m.normal_weighting,
            smoothing_angle: m.smoothing_angle,
            tangents: m.tangents,
            lightmap_uv: m.generate_lightmap_uvs,
            optimize: m.optimize_mesh,
            split_16bit: m.index_format == crate::IndexFormat::UInt16,
            uv_mask: m.uv_mask,
            compression: m.mesh_compression,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::from_import(&ModelImport::default())
    }
}

/// A mesh as an importer hands it over: a polygon soup.
#[derive(Clone, Debug, Default)]
pub struct ImportMesh {
    /// One vertex per polygon corner.
    pub corners: Vec<Vertex>,
    /// Corner count of each polygon (3 or 4, more is fan-triangulated); the
    /// corners of face `f` follow those of face `f - 1`.
    pub faces: Vec<u8>,
    /// Blendshapes, with `deltas` keyed by corner index.
    pub morphs: Vec<MorphTarget>,
    /// The file carried normals (otherwise they are calculated).
    pub has_normals: bool,
    /// The file carried tangents.
    pub has_tangents: bool,
    /// `uv1` is a real lightmap UV set (otherwise it is a copy of `uv`).
    pub real_uv1: bool,
}

/// What the pipeline produces.
pub struct Imported {
    pub mesh: MeshData,
    /// Blendshapes, with `deltas` keyed by vertex index of `mesh`.
    pub morphs: Vec<MorphTarget>,
    /// With Keep Quads: the quads, as vertex indices in polygon order. A quad
    /// `[a, b, c, d]` is the triangles `(a, b, c)` and `(a, c, d)` of `mesh`.
    /// Empty otherwise.
    pub quads: Vec<[u32; 4]>,
}

impl ImportMesh {
    /// Append one polygon.
    pub fn push_face(&mut self, corners: &[Vertex]) {
        self.corners.extend_from_slice(corners);
        self.faces.push(corners.len().min(255) as u8);
    }

    /// An indexed triangle list (glTF) as a soup. `morphs` are keyed by vertex
    /// index and are re-keyed to the corners that use each vertex.
    pub fn from_indexed(vertices: &[Vertex], indices: &[u32], morphs: Vec<MorphTarget>) -> Self {
        let tris = indices.len() / 3;
        let used = &indices[..tris * 3];
        let corners: Vec<Vertex> = used.iter().map(|&i| vertices[i as usize]).collect();
        let mut mesh = Self { corners, faces: vec![3; tris], morphs: Vec::new(), has_normals: true, has_tangents: true, real_uv1: true };
        if !morphs.is_empty() {
            let by_vertex = Csr::build(vertices.len(), used.iter().enumerate().map(|(c, &v)| (v as usize, c as u32)));
            mesh.morphs = morphs
                .into_iter()
                .map(|t| {
                    let mut deltas = Vec::new();
                    for (v, p, n) in t.deltas {
                        for &c in by_vertex.get(v as usize) {
                            deltas.push((c, p, n));
                        }
                    }
                    deltas.sort_unstable_by_key(|d| d.0);
                    MorphTarget { name: t.name, deltas }
                })
                .collect();
        }
        mesh
    }

    /// Triangle list of the faces, as corner indices. A quad becomes
    /// (0,1,2),(0,2,3); larger polygons fan from corner 0.
    fn triangle_corners(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.corners.len());
        let mut at = 0u32;
        for &n in &self.faces {
            for k in 1..(n as u32).saturating_sub(1) {
                out.extend_from_slice(&[at, at + k, at + k + 1]);
            }
            at += n as u32;
        }
        out
    }

    /// Rebuild from a subset/expansion of the current corners: new corner `k`
    /// is a copy of old corner `src[k]`, with `faces` as the new face sizes.
    /// Blendshape deltas follow their corners (to every copy).
    fn gather(&mut self, src: &[u32], faces: Vec<u8>) {
        if !self.morphs.is_empty() {
            let copies = Csr::build(self.corners.len(), src.iter().enumerate().map(|(k, &old)| (old as usize, k as u32)));
            par_each(&mut self.morphs, |t| {
                let mut deltas = Vec::with_capacity(t.deltas.len());
                for &(c, p, n) in &t.deltas {
                    for &k in copies.get(c as usize) {
                        deltas.push((k, p, n));
                    }
                }
                deltas.sort_unstable_by_key(|d| d.0);
                t.deltas = deltas;
            });
        }
        self.corners = src.iter().map(|&c| self.corners[c as usize]).collect();
        self.faces = faces;
    }
}

/// Run `f` over every item, spread over the cores when there are enough of
/// them to pay for the threads (a model with hundreds of blendshapes). Items
/// are independent, so the result equals the sequential loop.
pub(crate) fn par_each<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    if items.len() < 16 || threads < 2 {
        items.iter_mut().for_each(f);
        return;
    }
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for part in items.chunks_mut(chunk) {
            let f = &f;
            scope.spawn(move || part.iter_mut().for_each(f));
        }
    });
}

/// [`par_each`] for a handful of heavy, uneven items (one per mesh part):
/// threads pull the next item as they finish, whatever the item count.
pub(crate) fn par_each_dyn<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
    if threads < 2 {
        items.iter_mut().for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    let cells: Vec<std::sync::Mutex<&mut T>> = items.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some(cell) = cells.get(next.fetch_add(1, Ordering::Relaxed)) {
                    f(&mut cell.lock().unwrap());
                }
            });
        }
    });
}

/// Compressed "bucket" list: `get(i)` is the values pushed for key `i`.
struct Csr {
    start: Vec<u32>,
    list: Vec<u32>,
}

impl Csr {
    fn build(keys: usize, pairs: impl Iterator<Item = (usize, u32)> + Clone) -> Self {
        let mut start = vec![0u32; keys + 1];
        for (k, _) in pairs.clone() {
            start[k + 1] += 1;
        }
        for i in 0..keys {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut list = vec![0u32; *start.last().unwrap() as usize];
        for (k, v) in pairs {
            list[fill[k] as usize] = v;
            fill[k] += 1;
        }
        Self { start, list }
    }

    fn get(&self, key: usize) -> &[u32] {
        match (self.start.get(key), self.start.get(key + 1)) {
            (Some(&a), Some(&b)) => &self.list[a as usize..b as usize],
            _ => &[],
        }
    }
}

// --- the passes -----------------------------------------------------------

fn pos(v: &Vertex) -> Vec3 {
    Vec3::from(v.position)
}

/// Split a polygon into triangles of local corner indices. A quad splits along
/// its shorter diagonal; anything larger fans from corner 0.
fn split_polygon(p: &[Vec3]) -> Vec<[usize; 3]> {
    match p.len() {
        0..=2 => Vec::new(),
        3 => vec![[0, 1, 2]],
        4 => {
            if (p[0] - p[2]).length_squared() <= (p[1] - p[3]).length_squared() {
                vec![[0, 1, 2], [0, 2, 3]]
            } else {
                vec![[1, 2, 3], [1, 3, 0]]
            }
        }
        n => (1..n - 1).map(|k| [0, k, k + 1]).collect(),
    }
}

/// Triangulate faces with more than `keep` corners (`keep` 3 = everything).
fn triangulate(mesh: &mut ImportMesh) {
    triangulate_above(mesh, 3);
}

fn triangulate_above(mesh: &mut ImportMesh, keep: usize) {
    if mesh.faces.iter().all(|&n| n as usize <= keep) {
        return;
    }
    let (mut src, mut faces) = (Vec::with_capacity(mesh.corners.len() + mesh.corners.len() / 2), Vec::new());
    let mut at = 0usize;
    for &n in &mesh.faces {
        let n = n as usize;
        if n <= keep {
            src.extend((at..at + n).map(|c| c as u32));
            faces.push(n as u8);
        } else {
            let p: Vec<Vec3> = mesh.corners[at..at + n].iter().map(pos).collect();
            for t in split_polygon(&p) {
                src.extend(t.iter().map(|k| (at + k) as u32));
                faces.push(3);
            }
        }
        at += n;
    }
    mesh.gather(&src, faces);
}

fn remove_degenerates(mesh: &mut ImportMesh) {
    let (mut src, mut faces) = (Vec::with_capacity(mesh.corners.len()), Vec::with_capacity(mesh.faces.len()));
    let mut at = 0usize;
    let mut changed = false;
    for &n in &mesh.faces {
        let n = n as usize;
        let p: Vec<Vec3> = mesh.corners[at..at + n].iter().map(pos).collect();
        // Corners that repeat the previous corner's position collapse (a quad
        // with two coincident corners is a triangle).
        let keep: Vec<usize> = (0..n).filter(|&k| p[k] != p[(k + n - 1) % n]).collect();
        let ok = keep.len() >= 3 && {
            let kp: Vec<Vec3> = keep.iter().map(|&k| p[k]).collect();
            let longest = (0..kp.len()).map(|k| (kp[(k + 1) % kp.len()] - kp[k]).length_squared()).fold(0.0f32, f32::max);
            let area2: f32 = (1..kp.len() - 1).map(|k| (kp[k] - kp[0]).cross(kp[k + 1] - kp[0]).length()).sum();
            area2.is_finite() && area2 > 1e-6 * longest
        };
        if ok {
            src.extend(keep.iter().map(|&k| (at + k) as u32));
            faces.push(keep.len() as u8);
            changed |= keep.len() != n;
        } else {
            changed = true;
        }
        at += n;
    }
    if changed {
        mesh.gather(&src, faces);
    }
}

fn normals(mesh: &mut ImportMesh, s: &Settings) {
    match s.normals {
        NormalsMode::None => {
            for c in &mut mesh.corners {
                c.normal = [0.0, 0.0, 1.0];
            }
            mesh.has_normals = true;
            return;
        }
        NormalsMode::Import if mesh.has_normals => return,
        _ => {}
    }
    // The crease-angle routine works on an indexed triangle list; the soup's
    // corner indices serve as the vertex indices. (A kept quad's two shared
    // corners take the second triangle's value: they differ only for non-planar quads.)
    let mut temp = MeshData { vertices: mesh.corners.clone(), indices: mesh.triangle_corners(), has_lightmap_uv: false };
    crate::mesh_normals::recalculate_normals_weighted(&mut temp, s.smoothing_angle, s.weighting);
    for (c, t) in mesh.corners.iter_mut().zip(&temp.vertices) {
        c.normal = t.normal;
    }
    mesh.has_normals = true;
}

fn bits3(v: [f32; 3]) -> [u32; 3] {
    v.map(|x| (x + 0.0).to_bits())
}

fn tangents(mesh: &mut ImportMesh, s: &Settings) {
    let calc = match s.tangents {
        TangentMode::None => false,
        TangentMode::Calculate | TangentMode::Legacy => true,
        TangentMode::Import => !mesh.has_tangents,
    };
    if !calc {
        return;
    }
    if s.tangents == TangentMode::Legacy {
        legacy_tangents(mesh);
    } else {
        mikktspace_tangents(mesh);
    }
    mesh.has_tangents = true;
}

/// Tangents by MikkTSpace (Morten Mikkelsen's algorithm, the one normal maps
/// are baked against). The soup's corners are the face-vertices it works on;
/// the V coordinate is flipped for it because models carry top-left UVs while
/// tangent frames follow +Y-up normal maps.
fn mikktspace_tangents(mesh: &mut ImportMesh) {
    // MikkTSpace takes triangles and quads.
    triangulate_above(mesh, 4);
    struct Geo<'a> {
        corners: &'a mut [Vertex],
        start: Vec<usize>,
        size: &'a [u8],
    }
    impl mikktspace::Geometry for Geo<'_> {
        fn num_faces(&self) -> usize {
            self.size.len()
        }
        fn num_vertices_of_face(&self, face: usize) -> usize {
            self.size[face] as usize
        }
        fn position(&self, face: usize, vert: usize) -> [f32; 3] {
            self.corners[self.start[face] + vert].position
        }
        fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
            self.corners[self.start[face] + vert].normal
        }
        fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
            let uv = self.corners[self.start[face] + vert].uv;
            [uv[0], 1.0 - uv[1]]
        }
        fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
            self.corners[self.start[face] + vert].tangent = tangent;
        }
    }
    let mut start = Vec::with_capacity(mesh.faces.len());
    let mut at = 0;
    for &n in &mesh.faces {
        start.push(at);
        at += n as usize;
    }
    let mut geo = Geo { corners: &mut mesh.corners, start, size: &mesh.faces };
    if !mikktspace::generate_tangents(&mut geo) {
        tracing::warn!("MikkTSpace could not generate tangents for a mesh; keeping its tangents");
    }
}

/// The accumulate-and-orthogonalise tangents imports used before MikkTSpace.
fn legacy_tangents(mesh: &mut ImportMesh) {
    // Per-triangle tangent (dP/du) and bitangent (-dP/dv: glTF's v runs down the
    // image, and normal maps are +Y up), summed over the corners that share a
    // position, normal and UV, then orthogonalised to the normal. Not
    // MikkTSpace, but the same idea for smooth surfaces.
    let key = |v: &Vertex| (bits3(v.position), bits3(v.normal), [(v.uv[0] + 0.0).to_bits(), (v.uv[1] + 0.0).to_bits()]);
    let mut sums: HashMap<_, (Vec3, Vec3)> = HashMap::new();
    for t in mesh.triangle_corners().as_chunks::<3>().0 {
        let v = [&mesh.corners[t[0] as usize], &mesh.corners[t[1] as usize], &mesh.corners[t[2] as usize]];
        let (e1, e2) = (pos(v[1]) - pos(v[0]), pos(v[2]) - pos(v[0]));
        let (d1, d2) = (Vec2::from(v[1].uv) - Vec2::from(v[0].uv), Vec2::from(v[2].uv) - Vec2::from(v[0].uv));
        let det = d1.x * d2.y - d2.x * d1.y;
        if det.abs() < 1e-20 {
            continue;
        }
        let r = 1.0 / det;
        let tan = (e1 * d2.y - e2 * d1.y) * r;
        let bit = -((e2 * d1.x - e1 * d2.x) * r);
        // Weight by the triangle's area so slivers do not dominate.
        let w = e1.cross(e2).length();
        for c in v {
            let e = sums.entry(key(c)).or_default();
            e.0 += tan.normalize_or_zero() * w;
            e.1 += bit.normalize_or_zero() * w;
        }
    }
    for c in &mut mesh.corners {
        let n = Vec3::from(c.normal).normalize_or_zero();
        let (t, b) = sums.get(&key(c)).copied().unwrap_or_default();
        let t = (t - n * n.dot(t)).normalize_or_zero();
        if t == Vec3::ZERO || n == Vec3::ZERO {
            continue;
        }
        let w = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };
        c.tangent = [t.x, t.y, t.z, w];
    }
}

/// FNV-1a: the weld map hashes millions of 112-byte keys, where SipHash is slow.
#[derive(Default)]
struct WordHasher(u64);

impl Hasher for WordHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }
}

type WeldMap = HashMap<[u32; 28], u32, BuildHasherDefault<WordHasher>>;

/// Clear the UV sets the mask drops. A dropped UV0 is zeroed; a dropped UV1 is
/// not real any more, so it falls back to a copy of UV0.
fn mask_uvs(mesh: &mut ImportMesh, mask: u8) {
    let mask = mask & UV_ALL;
    if mask == UV_ALL {
        return;
    }
    if mask & 1 == 0 {
        for c in &mut mesh.corners {
            c.uv = [0.0; 2];
        }
    }
    if mask & 2 == 0 {
        mesh.real_uv1 = false;
    }
    if !mesh.real_uv1 {
        for c in &mut mesh.corners {
            c.uv1 = c.uv;
        }
    }
}

/// Snap `x` (inside `lo..=hi`) to one of `2^bits` evenly spaced values.
fn snap(x: f32, lo: f32, hi: f32, bits: u32) -> f32 {
    let extent = hi - lo;
    if extent <= 0.0 || !extent.is_finite() {
        return x;
    }
    let steps = ((1u64 << bits) - 1) as f32;
    lo + ((x - lo) / extent * steps).round() / steps * extent
}

/// Mesh Compression: positions snap to a grid over the mesh's bounds, UVs over
/// their bounds, and normals and tangents over -1..1 (then are renormalised).
fn quantise(mesh: &mut ImportMesh, level: MeshCompression) {
    let Some((pos_bits, uv_bits, dir_bits)) = level.bits() else { return };
    if mesh.corners.is_empty() {
        return;
    }
    let mut pb = [(f32::MAX, f32::MIN); 3];
    let mut uvb = [(f32::MAX, f32::MIN); 4];
    for c in &mesh.corners {
        for (k, b) in pb.iter_mut().enumerate() {
            *b = (b.0.min(c.position[k]), b.1.max(c.position[k]));
        }
        for k in 0..2 {
            uvb[k] = (uvb[k].0.min(c.uv[k]), uvb[k].1.max(c.uv[k]));
            uvb[2 + k] = (uvb[2 + k].0.min(c.uv1[k]), uvb[2 + k].1.max(c.uv1[k]));
        }
    }
    let unit = |d: [f32; 3]| -> [f32; 3] { d.map(|x| snap(x.clamp(-1.0, 1.0), -1.0, 1.0, dir_bits)) };
    for c in &mut mesh.corners {
        for (k, b) in pb.iter().enumerate() {
            c.position[k] = snap(c.position[k], b.0, b.1, pos_bits);
        }
        for k in 0..2 {
            c.uv[k] = snap(c.uv[k], uvb[k].0, uvb[k].1, uv_bits);
            c.uv1[k] = snap(c.uv1[k], uvb[2 + k].0, uvb[2 + k].1, uv_bits);
        }
        let n = Vec3::from(unit(c.normal));
        c.normal = n.try_normalize().unwrap_or(Vec3::from(c.normal)).to_array();
        let t = Vec3::from(unit([c.tangent[0], c.tangent[1], c.tangent[2]]));
        let t = t.try_normalize().unwrap_or(Vec3::new(c.tangent[0], c.tangent[1], c.tangent[2]));
        c.tangent = [t.x, t.y, t.z, c.tangent[3]];
    }
}

/// Everything about a corner that must match for two corners to be one vertex.
fn weld_key(v: &Vertex, morph_sig: u64) -> [u32; 28] {
    let mut k = [0u32; 28];
    let floats = v.position.iter().chain(&v.normal).chain(&v.uv).chain(&v.color).chain(&v.tangent).chain(&v.uv1).chain(&v.weights);
    for (n, x) in floats.enumerate() {
        k[n] = (x + 0.0).to_bits();
    }
    k[22..26].copy_from_slice(&v.joints);
    k[26] = morph_sig as u32;
    k[27] = (morph_sig >> 32) as u32;
    k
}

/// Turn the soup into an indexed mesh, welding when asked. Blendshape deltas
/// move from corners to the vertices those corners became.
fn weld_and_index(mesh: ImportMesh, weld: bool) -> (MeshData, Vec<MorphTarget>, Vec<[u32; 4]>) {
    let tris = mesh.triangle_corners();
    // Corners with different blendshape deltas must stay different vertices.
    let mut sig = vec![0u64; if weld { mesh.corners.len() } else { 0 }];
    if weld {
        for (ti, t) in mesh.morphs.iter().enumerate() {
            for &(c, p, n) in &t.deltas {
                let mut h = sig[c as usize] ^ (ti as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                for x in p.iter().chain(&n) {
                    h = (h ^ x.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
                }
                sig[c as usize] = h;
            }
        }
    }
    let mut corner_vertex: Vec<u32> = Vec::with_capacity(mesh.corners.len());
    let mut vertices: Vec<Vertex> = Vec::new();
    if weld {
        let mut map = WeldMap::with_capacity_and_hasher(mesh.corners.len(), Default::default());
        for (c, v) in mesh.corners.iter().enumerate() {
            let next = vertices.len() as u32;
            let id = *map.entry(weld_key(v, sig[c])).or_insert(next);
            if id == next {
                vertices.push(*v);
            }
            corner_vertex.push(id);
        }
    } else {
        vertices = mesh.corners.clone();
        corner_vertex = (0..mesh.corners.len() as u32).collect();
    }
    let indices: Vec<u32> = tris.iter().map(|&c| corner_vertex[c as usize]).collect();
    // Kept quads, by the vertices their corners became.
    let mut quads = Vec::new();
    let mut at = 0usize;
    for &n in &mesh.faces {
        if n == 4 {
            let q = [0, 1, 2, 3].map(|k| corner_vertex[at + k]);
            // A quad whose corners welded together is a triangle already.
            if (0..4).all(|a| (a + 1..4).all(|b| q[a] != q[b])) {
                quads.push(q);
            }
        }
        at += n as usize;
    }
    // Per target, the first delta of each vertex its corners became.
    let mut morphs = mesh.morphs;
    par_each(&mut morphs, |t| {
        let mut deltas: Vec<(u32, [f32; 3], [f32; 3])> = t.deltas.iter().map(|&(c, p, n)| (corner_vertex[c as usize], p, n)).collect();
        deltas.sort_by_key(|d| d.0);
        deltas.dedup_by_key(|d| d.0);
        t.deltas = deltas;
    });
    (MeshData { vertices, indices, has_lightmap_uv: false }, morphs, quads)
}

/// Reorder triangles for the vertex cache (a greedy scorer in the spirit of
/// Forsyth's published algorithm: prefer triangles whose vertices were just
/// used and vertices with few triangles left), then renumber vertices in the
/// order they are first fetched. Returns the vertex remap (`old -> new`).
pub fn optimise(mesh: &mut MeshData) -> Vec<u32> {
    const CACHE: usize = 32;
    let (nv, nt) = (mesh.vertices.len(), mesh.indices.len() / 3);
    if nt < 2 {
        return (0..nv as u32).collect();
    }
    let adj = Csr::build(nv, mesh.indices.iter().enumerate().map(|(i, &v)| (v as usize, (i / 3) as u32)));
    let mut live: Vec<u32> = (0..nv).map(|v| adj.get(v).len() as u32).collect();
    let mut cache_pos = vec![u32::MAX; nv];
    let score = |pos: u32, live: u32| -> f32 {
        if live == 0 {
            return -1.0;
        }
        let cache = match pos {
            u32::MAX => 0.0,
            0..=2 => 0.75,
            p => (1.0 - (p - 3) as f32 / (CACHE - 3) as f32).powf(1.5),
        };
        cache + 2.0 * (live as f32).powf(-0.5)
    };
    let mut done = vec![false; nt];
    let mut order: Vec<u32> = Vec::with_capacity(nt);
    let mut cache: Vec<u32> = Vec::with_capacity(CACHE + 3);
    let mut cursor = 0usize;
    let mut next: Option<usize> = None;
    while order.len() < nt {
        let t = match next.take() {
            Some(t) => t,
            None => {
                while done[cursor] {
                    cursor += 1;
                }
                cursor
            }
        };
        done[t] = true;
        order.push(t as u32);
        let tri = [mesh.indices[t * 3] as usize, mesh.indices[t * 3 + 1] as usize, mesh.indices[t * 3 + 2] as usize];
        for &v in &tri {
            live[v] -= 1;
        }
        // Move the triangle's vertices to the front of the cache.
        let mut fresh: Vec<u32> = Vec::with_capacity(CACHE + 3);
        for &v in tri.iter().rev() {
            if !fresh.contains(&(v as u32)) {
                fresh.push(v as u32);
            }
        }
        fresh.extend(cache.iter().copied().filter(|v| !tri.contains(&(*v as usize))));
        for &v in &cache {
            cache_pos[v as usize] = u32::MAX;
        }
        fresh.truncate(CACHE);
        cache = fresh;
        for (p, &v) in cache.iter().enumerate() {
            cache_pos[v as usize] = p as u32;
        }
        // The best unemitted triangle touching the cache continues the strip.
        let mut best = (f32::NEG_INFINITY, None);
        for &v in &cache {
            for &c in adj.get(v as usize) {
                let c = c as usize;
                if done[c] {
                    continue;
                }
                let s: f32 = (0..3)
                    .map(|k| {
                        let u = mesh.indices[c * 3 + k] as usize;
                        score(cache_pos[u], live[u])
                    })
                    .sum();
                if s > best.0 {
                    best = (s, Some(c));
                }
            }
        }
        next = best.1;
    }
    // Triangles in the new order, vertices renumbered by first use.
    let mut remap = vec![u32::MAX; nv];
    let mut count = 0u32;
    let mut indices = Vec::with_capacity(mesh.indices.len());
    for &t in &order {
        for k in 0..3 {
            let v = mesh.indices[t as usize * 3 + k] as usize;
            if remap[v] == u32::MAX {
                remap[v] = count;
                count += 1;
            }
            indices.push(remap[v]);
        }
    }
    for r in remap.iter_mut().filter(|r| **r == u32::MAX) {
        *r = count; // unused vertices go last
        count += 1;
    }
    let mut vertices = mesh.vertices.clone();
    for (old, &new) in remap.iter().enumerate() {
        vertices[new as usize] = mesh.vertices[old];
    }
    mesh.vertices = vertices;
    mesh.indices = indices;
    remap
}

/// Average cache miss ratio of a triangle list on a FIFO cache of `size`
/// (lower is better; 3.0 is the worst case, ~0.5 the best).
pub fn acmr(indices: &[u32], size: usize) -> f32 {
    let mut cache: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let mut misses = 0usize;
    for &i in indices {
        if !cache.contains(&i) {
            misses += 1;
            cache.push_back(i);
            if cache.len() > size {
                cache.pop_front();
            }
        }
    }
    misses as f32 / (indices.len() / 3).max(1) as f32
}

/// Run the pipeline. `None` when no face survives. The mesh is whole: see
/// [`run_split`] for a forced 16 bit index format.
pub fn run(mesh: ImportMesh, s: &Settings) -> Option<Imported> {
    run_traced(mesh, s, &mut |_| {})
}

/// [`run`], then split into pieces of at most 65535 vertices when the settings
/// force 16 bit indices. One entry (the whole mesh) otherwise; empty when no
/// face survives.
pub fn run_split(mesh: ImportMesh, s: &Settings) -> Vec<Imported> {
    match run(mesh, s) {
        Some(m) if s.split_16bit && m.mesh.vertices.len() > MAX_16BIT_VERTICES => split_16bit(m),
        Some(m) => vec![m],
        None => Vec::new(),
    }
}

/// [`run`], reporting each pass as it starts (for tests and import logs).
pub fn run_traced(mut mesh: ImportMesh, s: &Settings, trace: &mut dyn FnMut(Stage)) -> Option<Imported> {
    mask_uvs(&mut mesh, s.uv_mask);
    if !mesh.real_uv1 {
        for c in &mut mesh.corners {
            c.uv1 = c.uv;
        }
    }
    for stage in STAGE_ORDER {
        trace(stage);
        match stage {
            Stage::Triangulate if !s.keep_quads => triangulate(&mut mesh),
            // Quads stay, but larger polygons do not (the quad table has four corners).
            Stage::Triangulate => triangulate_above(&mut mesh, 4),
            Stage::RemoveDegenerates => remove_degenerates(&mut mesh),
            Stage::Normals => normals(&mut mesh, s),
            Stage::Tangents => tangents(&mut mesh, s),
            Stage::Quantise => quantise(&mut mesh, s.compression),
            Stage::Weld => {
                if mesh.faces.is_empty() {
                    return None;
                }
                let real_uv1 = mesh.real_uv1;
                let (mut data, mut morphs, mut quads) = weld_and_index(std::mem::take(&mut mesh), s.weld);
                // The remaining passes work on the indexed mesh.
                data.has_lightmap_uv = real_uv1 || crate::lightmap_uv::uv0_is_lightmappable(&data.vertices, &data.indices);
                trace(Stage::LightmapUv);
                if s.lightmap_uv && !real_uv1 {
                    if !morphs.is_empty() {
                        tracing::warn!("not generating lightmap UVs: the mesh has blendshapes, which are keyed by vertex");
                    } else {
                        data = crate::lightmap_uv::generate_lightmap_uv(&data);
                        // The unwrap splits vertices: the quads' indices no longer apply.
                        quads.clear();
                    }
                }
                trace(Stage::Optimise);
                if s.optimize {
                    let remap = optimise(&mut data);
                    par_each(&mut morphs, |t| {
                        for d in &mut t.deltas {
                            d.0 = remap[d.0 as usize];
                        }
                        t.deltas.sort_unstable_by_key(|d| d.0);
                    });
                    for q in &mut quads {
                        *q = q.map(|v| remap[v as usize]);
                    }
                }
                trace(Stage::Lod);
                return Some(Imported { mesh: data, morphs, quads });
            }
            _ => {}
        }
    }
    None
}

/// Most vertices a mesh can have and still index with 16 bits.
pub const MAX_16BIT_VERTICES: usize = 65535;

/// Split a mesh into pieces of at most [`MAX_16BIT_VERTICES`] vertices, so each
/// can use 16 bit indices. Triangles keep their order and their winding and
/// every triangle lands in exactly one piece, so the union of the pieces' triangles
/// is the original set; a vertex used by triangles in two pieces is copied into both.
/// Blendshape deltas follow their vertices; a quad stays in the table of the
/// piece that holds all four of its vertices (a quad cut across two pieces is
/// two triangles there).
pub fn split_16bit(whole: Imported) -> Vec<Imported> {
    let Imported { mesh, morphs, quads } = whole;
    let mut pieces: Vec<(Vec<u32>, HashMap<u32, u32>)> = Vec::new(); // (old vertex per new vertex, old -> new)
    let mut tris: Vec<Vec<u32>> = Vec::new(); // per piece: indices into the piece
    for t in mesh.indices.as_chunks::<3>().0 {
        let needs_room = |p: &(Vec<u32>, HashMap<u32, u32>)| {
            let mut fresh = 0;
            for (k, v) in t.iter().enumerate() {
                if !p.1.contains_key(v) && !t[..k].contains(v) {
                    fresh += 1;
                }
            }
            p.0.len() + fresh <= MAX_16BIT_VERTICES
        };
        if pieces.last().is_none_or(|p| !needs_room(p)) {
            pieces.push((Vec::new(), HashMap::new()));
            tris.push(Vec::new());
        }
        let at = pieces.len() - 1;
        for &v in t {
            let (list, map) = &mut pieces[at];
            let n = *map.entry(v).or_insert_with(|| {
                list.push(v);
                list.len() as u32 - 1
            });
            tris[at].push(n);
        }
    }
    pieces
        .into_iter()
        .zip(tris)
        .map(|((old, map), indices)| {
            let vertices = old.iter().map(|&v| mesh.vertices[v as usize]).collect();
            let morphs = morphs
                .iter()
                .map(|t| MorphTarget { name: t.name.clone(), deltas: t.deltas.iter().filter_map(|&(v, p, n)| map.get(&v).map(|&nv| (nv, p, n))).collect() })
                .map(|mut t| {
                    t.deltas.sort_unstable_by_key(|d| d.0);
                    t
                })
                .collect::<Vec<_>>();
            // A quad stays a quad where both its triangles ended up.
            let have: std::collections::HashSet<[u32; 3]> = indices
                .as_chunks::<3>().0.iter()
                .map(|t| {
                    let mut k = [t[0], t[1], t[2]];
                    k.sort_unstable();
                    k
                })
                .collect();
            let tri_in = |a: u32, b: u32, c: u32| {
                let mut k = [a, b, c];
                k.sort_unstable();
                have.contains(&k)
            };
            let quads = quads
                .iter()
                .filter(|q| q.iter().all(|v| map.contains_key(v)))
                .map(|q| q.map(|v| map[&v]))
                .filter(|q| tri_in(q[0], q[1], q[2]) && tri_in(q[0], q[2], q[3]))
                .collect();
            Imported { mesh: MeshData { vertices, indices, has_lightmap_uv: mesh.has_lightmap_uv }, morphs, quads }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> Vertex {
        Vertex { position: p, normal: n, uv, color: [1.0; 4], tangent: [1.0, 0.0, 0.0, 1.0], uv1: uv, joints: [0; 4], weights: [1.0, 0.0, 0.0, 0.0] }
    }

    fn flat(p: [f32; 3]) -> Vertex {
        v(p, [0.0, 0.0, 1.0], [p[0], 1.0 - p[1]])
    }

    /// A unit quad in the XY plane facing +Z.
    fn quad_mesh() -> ImportMesh {
        let mut m = ImportMesh { has_normals: true, has_tangents: false, real_uv1: false, ..Default::default() };
        m.push_face(&[flat([0.0, 0.0, 0.0]), flat([1.0, 0.0, 0.0]), flat([1.0, 1.0, 0.0]), flat([0.0, 1.0, 0.0])]);
        m
    }

    #[test]
    fn stages_run_in_the_documented_order() {
        let mut seen = Vec::new();
        run_traced(quad_mesh(), &Settings::default(), &mut |s| seen.push(s)).unwrap();
        assert_eq!(seen, STAGE_ORDER.to_vec());
        assert_eq!(
            STAGE_ORDER,
            [Stage::Triangulate, Stage::RemoveDegenerates, Stage::Normals, Stage::Tangents, Stage::Quantise, Stage::Weld, Stage::LightmapUv, Stage::Optimise, Stage::Lod]
        );
    }

    #[test]
    fn triangulating_a_quad_gives_two_triangles_and_keep_quads_keeps_the_corners() {
        let out = run(quad_mesh(), &Settings { weld: true, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.indices.len(), 6);
        assert_eq!(out.mesh.vertices.len(), 4, "the two triangles share the diagonal's vertices");
        // Both triangles face +Z.
        for t in out.mesh.indices.chunks(3) {
            let p = |i: u32| Vec3::from(out.mesh.vertices[i as usize].position);
            assert!((p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).z > 0.0);
        }
        let kept = run(quad_mesh(), &Settings { keep_quads: true, ..Default::default() }).unwrap();
        assert_eq!(kept.mesh.indices.len(), 6, "emitted as a triangle pair");
        assert_eq!(kept.mesh.vertices.len(), 4);
    }

    #[test]
    fn a_degenerate_face_is_removed_before_normals_and_tangents_see_it() {
        let mut m = quad_mesh();
        m.push_face(&[flat([5.0, 5.0, 0.0]), flat([5.0, 5.0, 0.0]), flat([6.0, 5.0, 0.0])]); // repeated corner
        m.push_face(&[flat([7.0, 0.0, 0.0]), flat([8.0, 0.0, 0.0]), flat([9.0, 0.0, 0.0])]); // collinear
        let out = run(m, &Settings::default()).unwrap();
        assert_eq!(out.mesh.indices.len(), 6, "only the quad's two triangles remain");
        assert!(out.mesh.vertices.iter().all(|v| v.position[0] <= 1.0));
    }

    #[test]
    fn a_quad_with_coincident_corners_collapses_to_a_triangle() {
        let mut m = ImportMesh { has_normals: true, has_tangents: true, real_uv1: true, ..Default::default() };
        m.push_face(&[flat([0.0, 0.0, 0.0]), flat([0.0, 0.0, 0.0]), flat([1.0, 0.0, 0.0]), flat([0.0, 1.0, 0.0])]);
        let out = run(m, &Settings::default()).unwrap();
        assert_eq!(out.mesh.indices.len(), 3);
    }

    #[test]
    fn nothing_left_is_none() {
        let mut m = ImportMesh::default();
        m.push_face(&[flat([0.0; 3]), flat([0.0; 3]), flat([0.0; 3])]);
        assert!(run(m, &Settings::default()).is_none());
    }

    /// A cube as an un-indexed soup with flat normals and per-face UVs.
    fn cube_soup() -> ImportMesh {
        let mut m = ImportMesh { has_normals: true, has_tangents: false, real_uv1: false, ..Default::default() };
        let faces = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        for (n, u, w) in faces {
            let (n, u, w) = (Vec3::from(n), Vec3::from(u), Vec3::from(w));
            let p = |a: f32, b: f32| (n * 0.5 + u * a + w * b).to_array();
            let corner = |a: f32, b: f32| v(p(a, b), n.to_array(), [a + 0.5, b + 0.5]);
            m.push_face(&[corner(-0.5, -0.5), corner(0.5, -0.5), corner(0.5, 0.5), corner(-0.5, 0.5)]);
        }
        m
    }

    /// The cube with smooth radial normals and one shared UV, so only the
    /// position distinguishes corners.
    fn smooth_cube() -> ImportMesh {
        let mut m = cube_soup();
        for c in &mut m.corners {
            c.normal = Vec3::from(c.position).normalize().to_array();
            c.uv = [0.0; 2];
        }
        m
    }

    #[test]
    fn weld_merges_shared_corners_but_keeps_normal_and_uv_seams() {
        // 6 quads = 24 corners; flat normals + per-face UVs make every face
        // corner distinct (each cube corner has three different normals).
        let out = run(cube_soup(), &Settings { optimize: false, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.vertices.len(), 24, "hard edges (different normals) are not welded");
        assert_eq!(out.mesh.indices.len(), 36);

        let out = run(smooth_cube(), &Settings { optimize: false, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.vertices.len(), 8);

        // A UV seam alone (same position and normal, different UV) is kept.
        let mut seam = smooth_cube();
        seam.corners[0].uv = [0.25, 0.25];
        let out = run(seam, &Settings { optimize: false, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.vertices.len(), 9, "the seam splits exactly one vertex");

        // Weld off leaves every (triangulated) corner alone: 12 triangles x 3.
        let out = run(cube_soup(), &Settings { weld: false, optimize: false, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.vertices.len(), 36);
    }

    #[test]
    fn welding_keeps_the_geometry() {
        let area = |m: &MeshData| -> f32 {
            m.indices
                .chunks(3)
                .map(|t| {
                    let p = |i: u32| Vec3::from(m.vertices[i as usize].position);
                    (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5
                })
                .sum()
        };
        let welded = run(cube_soup(), &Settings::default()).unwrap();
        let raw = run(cube_soup(), &Settings { weld: false, optimize: false, ..Default::default() }).unwrap();
        assert!((area(&welded.mesh) - 6.0).abs() < 1e-4);
        assert!((area(&welded.mesh) - area(&raw.mesh)).abs() < 1e-4);
    }

    #[test]
    fn blendshape_deltas_follow_welded_and_reordered_vertices() {
        let mut m = smooth_cube();
        // Lift every corner at the (+,+,+) cube corner.
        let target: Vec<u32> = (0..m.corners.len() as u32).filter(|&c| m.corners[c as usize].position == [0.5, 0.5, 0.5]).collect();
        assert_eq!(target.len(), 3);
        m.morphs.push(MorphTarget { name: "up".into(), deltas: target.iter().map(|&c| (c, [0.0, 1.0, 0.0], [0.0; 3])).collect() });
        for optimize in [false, true] {
            let out = run(m.clone(), &Settings { optimize, ..Default::default() }).unwrap();
            assert_eq!(out.morphs[0].deltas.len(), 1, "three corners, one welded vertex, one delta");
            let (vi, dp, _) = out.morphs[0].deltas[0];
            assert_eq!(out.mesh.vertices[vi as usize].position, [0.5, 0.5, 0.5], "optimise={optimize}");
            assert_eq!(dp, [0.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn corners_with_different_blendshape_deltas_are_not_welded() {
        let mut m = smooth_cube();
        let target: Vec<u32> = (0..m.corners.len() as u32).filter(|&c| m.corners[c as usize].position == [0.5, 0.5, 0.5]).collect();
        m.morphs.push(MorphTarget { name: "split".into(), deltas: vec![(target[0], [0.0, 1.0, 0.0], [0.0; 3])] });
        let out = run(m, &Settings::default()).unwrap();
        assert_eq!(out.mesh.vertices.len(), 9, "one corner moves differently, so it keeps its own vertex");
    }

    #[test]
    fn missing_normals_are_calculated_and_import_keeps_the_files() {
        let mut m = quad_mesh();
        m.has_normals = false;
        for c in &mut m.corners {
            c.normal = [0.0, 1.0, 0.0]; // the placeholder a loader writes
        }
        let out = run(m, &Settings::default()).unwrap();
        assert!(out.mesh.vertices.iter().all(|v| (Vec3::from(v.normal) - Vec3::Z).length() < 1e-4));
        let mut with = quad_mesh();
        for c in &mut with.corners {
            c.normal = [1.0, 0.0, 0.0];
        }
        let out = run(with, &Settings::default()).unwrap();
        assert!(out.mesh.vertices.iter().all(|v| v.normal == [1.0, 0.0, 0.0]), "file normals are kept");
    }

    #[test]
    fn tangents_follow_the_uv_u_direction_with_the_gltf_handedness() {
        let out = run(quad_mesh(), &Settings::default()).unwrap();
        for vtx in &out.mesh.vertices {
            assert!((Vec3::new(vtx.tangent[0], vtx.tangent[1], vtx.tangent[2]) - Vec3::X).length() < 1e-4, "{:?}", vtx.tangent);
            assert_eq!(vtx.tangent[3], 1.0);
        }
        // Mirroring the UVs in U flips the tangent frame's handedness.
        let mut mirrored = quad_mesh();
        for c in &mut mirrored.corners {
            c.uv[0] = 1.0 - c.uv[0];
        }
        let out = run(mirrored, &Settings::default()).unwrap();
        assert!(out.mesh.vertices.iter().all(|v| v.tangent[3] == -1.0));
    }

    #[test]
    fn file_tangents_are_kept_unless_calculate_is_chosen() {
        let mut m = quad_mesh();
        m.has_tangents = true;
        for c in &mut m.corners {
            c.tangent = [0.0, 1.0, 0.0, 1.0];
        }
        let kept = run(m.clone(), &Settings::default()).unwrap();
        assert!(kept.mesh.vertices.iter().all(|v| v.tangent == [0.0, 1.0, 0.0, 1.0]));
        let calc = run(m.clone(), &Settings { tangents: TangentMode::Calculate, ..Default::default() }).unwrap();
        assert!(calc.mesh.vertices.iter().all(|v| v.tangent[0] > 0.99));
        let none = run(m, &Settings { tangents: TangentMode::None, ..Default::default() }).unwrap();
        assert!(none.mesh.vertices.iter().all(|v| v.tangent == [0.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn recalculated_normals_keep_hard_edges_under_the_crease_angle() {
        let mut m = cube_soup();
        for c in &mut m.corners {
            c.normal = [0.0, 1.0, 0.0];
        }
        let s = Settings { normals: NormalsMode::Calculate, smoothing_angle: 60.0, optimize: false, ..Default::default() };
        let out = run(m, &s).unwrap();
        // A cube's 90 degree edges exceed the 60 degree crease: flat faces.
        for vtx in &out.mesh.vertices {
            let n = Vec3::from(vtx.normal);
            assert!((n.abs().max_element() - 1.0).abs() < 1e-4, "axis-aligned normal, got {n:?}");
        }
        assert_eq!(out.mesh.vertices.len(), 24);
    }

    #[test]
    fn optimising_improves_the_cache_miss_ratio_and_keeps_every_triangle() {
        // A 40x40 grid with its triangles shuffled.
        let n = 40usize;
        let mut vertices = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                vertices.push(flat([x as f32, y as f32, 0.0]));
            }
        }
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let at = |x: usize, y: usize| (y * (n + 1) + x) as u32;
        for y in 0..n {
            for x in 0..n {
                tris.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
                tris.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        // Deterministic shuffle.
        let mut seed = 12345u64;
        for i in (1..tris.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            tris.swap(i, (seed >> 33) as usize % (i + 1));
        }
        let indices: Vec<u32> = tris.iter().flatten().copied().collect();
        let mut mesh = MeshData { vertices, indices, has_lightmap_uv: false };
        let as_set = |m: &MeshData| -> std::collections::BTreeSet<Vec<[u32; 3]>> {
            m.indices
                .chunks(3)
                .map(|t| {
                    let mut k: Vec<[u32; 3]> = t.iter().map(|&i| bits3(m.vertices[i as usize].position)).collect();
                    k.sort();
                    k
                })
                .collect()
        };
        let before_set = as_set(&mesh);
        let before = acmr(&mesh.indices, 32);
        let remap = optimise(&mut mesh);
        let after = acmr(&mesh.indices, 32);
        assert!(after < before * 0.6, "ACMR {before:.2} -> {after:.2}");
        assert!(after < 1.2, "a grid should reach roughly 0.5-1.0, got {after:.2}");
        assert_eq!(remap.len(), mesh.vertices.len());
        assert_eq!(before_set, as_set(&mesh), "the same triangles, reordered");
    }

    #[test]
    fn indexed_input_round_trips_through_the_soup() {
        // Two triangles sharing an edge, as glTF would hand them over.
        let verts = vec![flat([0.0, 0.0, 0.0]), flat([1.0, 0.0, 0.0]), flat([1.0, 1.0, 0.0]), flat([0.0, 1.0, 0.0])];
        let morph = MorphTarget { name: "m".into(), deltas: vec![(2, [0.0, 0.0, 1.0], [0.0; 3])] };
        let m = ImportMesh::from_indexed(&verts, &[0, 1, 2, 0, 2, 3], vec![morph]);
        assert_eq!(m.corners.len(), 6);
        assert_eq!(m.morphs[0].deltas.len(), 2, "vertex 2 is used by both triangles");
        let out = run(m, &Settings { optimize: false, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.vertices.len(), 4);
        assert_eq!(out.mesh.indices.len(), 6);
        assert_eq!(out.morphs[0].deltas.len(), 1);
        assert_eq!(out.mesh.vertices[out.morphs[0].deltas[0].0 as usize].position, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn lightmap_uvs_are_generated_when_asked_for_and_skipped_with_blendshapes() {
        let on = Settings { lightmap_uv: true, ..Default::default() };
        let out = run(cube_soup(), &on).unwrap();
        assert!(out.mesh.has_lightmap_uv, "a generated unwrap is bakeable");
        // Without the setting the cube's per-face UVs overlap nothing but UV1 is just a copy of UV0.
        let off = run(cube_soup(), &Settings::default()).unwrap();
        assert!(off.mesh.vertices.iter().all(|v| v.uv1 == v.uv));
        // A real UV1 is kept as is.
        let mut real = cube_soup();
        real.real_uv1 = true;
        for c in &mut real.corners {
            c.uv1 = [0.5, 0.25];
        }
        let kept = run(real, &on).unwrap();
        assert!(kept.mesh.vertices.iter().all(|v| v.uv1 == [0.5, 0.25]));
        // Blendshapes are keyed by vertex, so the unwrap (which splits vertices) is skipped.
        let mut morphed = cube_soup();
        morphed.morphs.push(MorphTarget { name: "m".into(), deltas: vec![(0, [0.0, 1.0, 0.0], [0.0; 3])] });
        let out = run(morphed, &on).unwrap();
        assert!(out.morphs[0].deltas.len() == 1);
    }

    // ---- 16 bit splitting ---------------------------------------------------

    /// An `n x n` grid of quads (n+1 squared vertices), as an indexed triangle mesh.
    fn grid(n: usize) -> ImportMesh {
        let mut verts = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                verts.push(flat([x as f32, y as f32, 0.0]));
            }
        }
        let at = |x: usize, y: usize| (y * (n + 1) + x) as u32;
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                idx.extend_from_slice(&[at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        ImportMesh::from_indexed(&verts, &idx, Vec::new())
    }

    /// Every triangle of a mesh as its three positions (winding kept), sorted.
    fn triangle_set(m: &MeshData) -> Vec<[[u32; 3]; 3]> {
        let mut out: Vec<_> = m
            .indices
            .as_chunks::<3>().0.iter()
            .map(|t| [0, 1, 2].map(|k| bits3(m.vertices[t[k] as usize].position)))
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn a_forced_16_bit_format_splits_a_big_mesh_and_keeps_every_triangle() {
        let n = 300; // 90601 vertices, 180000 triangles
        let whole = run(grid(n), &Settings::default()).unwrap();
        assert!(whole.mesh.vertices.len() > MAX_16BIT_VERTICES);
        let want = triangle_set(&whole.mesh);
        // Auto and 32 bit leave it whole.
        assert_eq!(run_split(grid(n), &Settings { split_16bit: false, ..Default::default() }).len(), 1);
        let pieces = run_split(grid(n), &Settings { split_16bit: true, ..Default::default() });
        assert!(pieces.len() >= 2, "{} pieces", pieces.len());
        let mut got = Vec::new();
        for p in &pieces {
            assert!(p.mesh.vertices.len() <= MAX_16BIT_VERTICES, "a piece has {} vertices", p.mesh.vertices.len());
            assert!(p.mesh.indices.iter().all(|&i| (i as usize) < p.mesh.vertices.len() && i <= u16::MAX as u32), "indices fit 16 bit");
            got.extend(triangle_set(&p.mesh));
        }
        got.sort_unstable();
        assert_eq!(got, want, "the same triangles, same winding, none lost or duplicated");
        // A mesh that fits is not touched.
        let small = run_split(grid(10), &Settings { split_16bit: true, ..Default::default() });
        assert_eq!(small.len(), 1);
    }

    #[test]
    fn par_each_matches_the_sequential_loop() {
        let mut a: Vec<u64> = (0..1000).collect();
        let mut b = a.clone();
        par_each(&mut a, |x| *x = *x * 3 + 1);
        b.iter_mut().for_each(|x| *x = *x * 3 + 1);
        assert_eq!(a, b);
        let mut c: Vec<u64> = (0..5).collect();
        par_each_dyn(&mut c, |x| *x += 10);
        assert_eq!(c, [10, 11, 12, 13, 14]);
    }

    #[test]
    fn a_face_with_many_blendshapes_keeps_every_delta_on_its_vertex() {
        // 60 targets (enough to take the threaded path): target k lifts the column x == k % 8.
        let n = 8;
        let mut soup = grid(n);
        for k in 0..60usize {
            let col = (k % n) as f32;
            let deltas: Vec<(u32, [f32; 3], [f32; 3])> =
                (0..soup.corners.len() as u32).filter(|&c| soup.corners[c as usize].position[0] == col).map(|c| (c, [0.0, 0.0, k as f32], [0.0; 3])).collect();
            soup.morphs.push(MorphTarget { name: format!("shape{k}"), deltas });
        }
        let out = run(soup, &Settings { weld: true, ..Default::default() }).unwrap();
        assert_eq!(out.morphs.len(), 60);
        for (k, t) in out.morphs.iter().enumerate() {
            assert_eq!(t.name, format!("shape{k}"), "targets keep their order");
            let col = (k % n) as f32;
            let want = out.mesh.vertices.iter().filter(|v| v.position[0] == col).count();
            assert_eq!(t.deltas.len(), want, "one delta per vertex of the column");
            assert!(t.deltas.windows(2).all(|w| w[0].0 < w[1].0), "sorted, no vertex twice");
            for d in &t.deltas {
                assert_eq!(out.mesh.vertices[d.0 as usize].position[0], col);
                assert_eq!(d.1, [0.0, 0.0, k as f32]);
            }
        }
    }

    #[test]
    fn splitting_carries_blendshapes_and_quads_with_their_vertices() {
        let n = 300;
        let mut soup = grid(n);
        // Lift the last vertex's corners; and keep quads.
        let last = [n as f32, n as f32, 0.0];
        let deltas: Vec<(u32, [f32; 3], [f32; 3])> =
            (0..soup.corners.len() as u32).filter(|&c| soup.corners[c as usize].position == last).map(|c| (c, [0.0, 0.0, 1.0], [0.0; 3])).collect();
        soup.morphs.push(MorphTarget { name: "lift".into(), deltas });
        let pieces = run_split(soup, &Settings { split_16bit: true, ..Default::default() });
        let mut lifted = 0;
        for p in &pieces {
            for d in &p.morphs[0].deltas {
                assert_eq!(p.mesh.vertices[d.0 as usize].position, last);
                lifted += 1;
            }
        }
        assert!(lifted >= 1, "the delta landed on the vertex in at least one piece");
        // Quads: a quad grid with Keep Quads keeps a quad table that holds up after the split.
        let mut quads = ImportMesh { has_normals: true, has_tangents: true, real_uv1: true, ..Default::default() };
        let at = |x: usize, y: usize| flat([x as f32, y as f32, 0.0]);
        for y in 0..n {
            for x in 0..n {
                quads.push_face(&[at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        let pieces = run_split(quads, &Settings { split_16bit: true, keep_quads: true, ..Default::default() });
        assert!(pieces.len() >= 2);
        let total: usize = pieces.iter().map(|p| p.quads.len()).sum();
        assert!(total > n * n * 9 / 10, "almost every quad survives whole ({total} of {})", n * n);
        for p in &pieces {
            for q in p.quads.iter().step_by(499) {
                let tri = |a: u32, b: u32, c: u32| p.mesh.indices.as_chunks::<3>().0.iter().any(|t| { let mut x = [t[0], t[1], t[2]]; let mut y = [a, b, c]; x.sort_unstable(); y.sort_unstable(); x == y });
                assert!(tri(q[0], q[1], q[2]) && tri(q[0], q[2], q[3]), "a quad is its two triangles");
            }
        }
    }

    // ---- quads --------------------------------------------------------------

    #[test]
    fn keep_quads_persists_a_quad_table_that_survives_welding_and_optimising() {
        let plain = run(quad_mesh(), &Settings::default()).unwrap();
        assert!(plain.quads.is_empty(), "no table unless Keep Quads is on");
        for optimize in [false, true] {
            let kept = run(quad_mesh(), &Settings { keep_quads: true, optimize, ..Default::default() }).unwrap();
            assert_eq!(kept.quads.len(), 1);
            let q = kept.quads[0];
            // The quad is the corners in order, so its first triangle is (q0, q1, q2) and its second (q0, q2, q3).
            let pos = |v: u32| kept.mesh.vertices[v as usize].position;
            assert_eq!([pos(q[0]), pos(q[1]), pos(q[2]), pos(q[3])], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], "optimise={optimize}");
            let has = |a: u32, b: u32, c: u32| kept.mesh.indices.as_chunks::<3>().0.iter().any(|t| *t == [a, b, c] || *t == [b, c, a] || *t == [c, a, b]);
            assert!(has(q[0], q[1], q[2]) && has(q[0], q[2], q[3]), "the table matches the triangles");
        }
        // Triangles in, no quads out.
        let mut tri = ImportMesh { has_normals: true, has_tangents: true, real_uv1: true, ..Default::default() };
        tri.push_face(&[flat([0.0; 3]), flat([1.0, 0.0, 0.0]), flat([0.0, 1.0, 0.0])]);
        assert!(run(tri, &Settings { keep_quads: true, ..Default::default() }).unwrap().quads.is_empty());
        // A six-sided face is cut into triangles even with Keep Quads (the table holds quads).
        let mut hex = ImportMesh { has_normals: true, has_tangents: true, real_uv1: true, ..Default::default() };
        let ring: Vec<Vertex> = (0..6).map(|k| { let a = k as f32 * std::f32::consts::TAU / 6.0; flat([a.cos(), a.sin(), 0.0]) }).collect();
        hex.push_face(&ring);
        let out = run(hex, &Settings { keep_quads: true, ..Default::default() }).unwrap();
        assert_eq!(out.mesh.indices.len(), 12);
        assert!(out.quads.is_empty());
    }

    // ---- MikkTSpace ---------------------------------------------------------

    /// A UV sphere (u around, v from the top) with smooth normals, as an indexed mesh.
    fn uv_sphere(rings: usize, segs: usize) -> ImportMesh {
        let mut verts = Vec::new();
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;
            for sgm in 0..=segs {
                let u = sgm as f32 / segs as f32;
                let th = u * std::f32::consts::TAU;
                let p = Vec3::new(phi.sin() * th.cos(), phi.cos(), phi.sin() * th.sin());
                verts.push(Vertex { position: p.to_array(), normal: p.to_array(), uv: [u, v], color: [1.0; 4], tangent: [1.0, 0.0, 0.0, 1.0], uv1: [u, v], joints: [0; 4], weights: [1.0, 0.0, 0.0, 0.0] });
            }
        }
        let at = |r: usize, s: usize| (r * (segs + 1) + s) as u32;
        let mut idx = Vec::new();
        for r in 0..rings {
            for sgm in 0..segs {
                // Counter-clockwise seen from outside (u runs the way the surface turns, v runs down).
                idx.extend_from_slice(&[at(r, sgm), at(r + 1, sgm + 1), at(r + 1, sgm)]);
                idx.extend_from_slice(&[at(r, sgm), at(r, sgm + 1), at(r + 1, sgm + 1)]);
            }
        }
        let mut m = ImportMesh::from_indexed(&verts, &idx, Vec::new());
        m.has_tangents = false;
        m
    }

    #[test]
    fn mikktspace_tangents_on_a_sphere_are_orthonormal_and_consistently_handed() {
        let out = run(uv_sphere(24, 48), &Settings { tangents: TangentMode::Calculate, optimize: false, ..Default::default() }).unwrap();
        let mut sign = None;
        let mut checked = 0;
        for v in &out.mesh.vertices {
            let (n, t) = (Vec3::from(v.normal), Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]));
            let p = Vec3::from(v.position);
            // Away from the poles, where u is degenerate.
            if p.y.abs() > 0.97 {
                continue;
            }
            assert!((t.length() - 1.0).abs() < 1e-3, "unit tangent, got {}", t.length());
            assert!(n.dot(t).abs() < 1e-3, "tangent is orthogonal to the normal, dot {}", n.dot(t));
            assert!(v.tangent[3] == 1.0 || v.tangent[3] == -1.0);
            // u grows with the angle theta: the tangent follows d/dtheta = (-sin th, 0, cos th) at (cos th, ., sin th).
            let along_u = Vec3::new(-p.z, 0.0, p.x).normalize();
            assert!(t.dot(along_u) > 0.99, "tangent follows the U direction, dot {}", t.dot(along_u));
            // The frame is right handed with respect to the +Y-up normal map: cross(n, t) * w points where v (flipped, so up) grows.
            let up_v = Vec3::new(0.0, 1.0, 0.0) - n * n.y; // dP/d(1 - v) is towards +Y on the surface
            let b = n.cross(t) * v.tangent[3];
            assert!(b.dot(up_v.normalize()) > 0.99, "bitangent points up the surface, dot {}", b.dot(up_v.normalize()));
            match sign {
                None => sign = Some(v.tangent[3]),
                Some(s) => assert_eq!(s, v.tangent[3], "one handedness over a non-mirrored surface"),
            }
            checked += 1;
        }
        assert!(checked > 500);
    }

    #[test]
    fn mikktspace_handles_kept_quads_and_the_legacy_mode_still_works() {
        // The unit quad with Keep Quads: the quad goes to MikkTSpace whole.
        let out = run(quad_mesh(), &Settings { keep_quads: true, tangents: TangentMode::Calculate, ..Default::default() }).unwrap();
        for v in &out.mesh.vertices {
            assert!((Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]) - Vec3::X).length() < 1e-4);
            assert_eq!(v.tangent[3], 1.0);
        }
        let legacy = run(quad_mesh(), &Settings { tangents: TangentMode::Legacy, ..Default::default() }).unwrap();
        assert!(legacy.mesh.vertices.iter().all(|v| v.tangent[0] > 0.99 && v.tangent[3] == 1.0));
    }

    // ---- normals mode, UV mask, compression -----------------------------------

    #[test]
    fn normals_none_writes_a_constant_and_calculate_ignores_the_files() {
        let mut m = quad_mesh();
        for c in &mut m.corners {
            c.normal = [1.0, 0.0, 0.0];
        }
        let none = run(m.clone(), &Settings { normals: NormalsMode::None, ..Default::default() }).unwrap();
        assert!(none.mesh.vertices.iter().all(|v| v.normal == [0.0, 0.0, 1.0]));
        let calc = run(m, &Settings { normals: NormalsMode::Calculate, ..Default::default() }).unwrap();
        assert!(calc.mesh.vertices.iter().all(|v| (Vec3::from(v.normal) - Vec3::Z).length() < 1e-4), "calculated from the geometry, not the file's +X");
    }

    #[test]
    fn the_uv_mask_drops_uv_sets() {
        let mut m = quad_mesh();
        m.real_uv1 = true;
        for c in &mut m.corners {
            c.uv1 = [0.25, 0.75];
        }
        let both = run(m.clone(), &Settings::default()).unwrap();
        assert!(both.mesh.vertices.iter().all(|v| v.uv1 == [0.25, 0.75]));
        let no1 = run(m.clone(), &Settings { uv_mask: 0b01, ..Default::default() }).unwrap();
        assert!(no1.mesh.vertices.iter().all(|v| v.uv1 == v.uv), "a dropped UV1 falls back to UV0");
        let no0 = run(m, &Settings { uv_mask: 0b10, ..Default::default() }).unwrap();
        assert!(no0.mesh.vertices.iter().all(|v| v.uv == [0.0, 0.0] && v.uv1 == [0.25, 0.75]));
    }

    #[test]
    fn mesh_compression_snaps_vertices_and_lets_the_weld_merge_them() {
        // Two triangles whose shared corner differs by far less than a High step.
        let mut m = ImportMesh { has_normals: true, has_tangents: true, real_uv1: true, ..Default::default() };
        m.push_face(&[flat([0.0, 0.0, 0.0]), flat([100.0, 0.0, 0.0]), flat([0.0, 100.0, 0.0])]);
        m.push_face(&[flat([100.0, 0.0, 0.0]), flat([100.0, 100.0, 0.0]), flat([0.0, 100.001, 0.0])]);
        let off = run(m.clone(), &Settings::default()).unwrap();
        assert_eq!(off.mesh.vertices.len(), 5, "the 0.001 offset keeps a vertex apart");
        let high = run(m.clone(), &Settings { compression: MeshCompression::High, ..Default::default() }).unwrap();
        assert_eq!(high.mesh.vertices.len(), 4, "quantising welds them");
        for v in &high.mesh.vertices {
            let step = 100.0 / 4095.0;
            // On the grid: a whole number of steps from the minimum.
            let k = v.position[0] / step;
            assert!((k - k.round()).abs() < 1e-2, "x={} is {k} steps", v.position[0]);
            assert!((Vec3::from(v.normal).length() - 1.0).abs() < 1e-4, "normals stay unit length");
        }
        let low = run(m.clone(), &Settings { compression: MeshCompression::Low, ..Default::default() }).unwrap();
        let err = |o: &Imported| o.mesh.vertices.iter().map(|v| (v.position[0] - (v.position[0] / 100.0).round() * 100.0).abs()).fold(0.0f32, f32::max);
        assert!(err(&low) <= err(&high) + 1e-3);
        assert_eq!(MeshCompression::Off.bits(), None);
    }

    #[test]
    fn model_import_settings_map_into_the_pipeline() {
        let m = ModelImport { index_format: crate::IndexFormat::UInt16, normals: NormalsMode::Calculate, normal_weighting: NormalWeighting::Area, uv_mask: 1, mesh_compression: MeshCompression::Medium, ..Default::default() };
        let s = Settings::from_import(&m);
        assert!(s.split_16bit && s.normals == NormalsMode::Calculate && s.weighting == NormalWeighting::Area && s.uv_mask == 1 && s.compression == MeshCompression::Medium);
        // The older toggle still means Calculate.
        let old = ModelImport { recalculate_normals: true, ..Default::default() };
        assert_eq!(Settings::from_import(&old).normals, NormalsMode::Calculate);
        assert!(!Settings::from_import(&ModelImport::default()).split_16bit);
    }
}
