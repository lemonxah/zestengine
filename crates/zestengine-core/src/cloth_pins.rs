//! Painted per-vertex Mesh Cloth data (pin weight and max distance) and the
//! brush math the editor's pin painter runs on it.
//!
//! Values are quantized to a byte per render vertex and saved as a short
//! run-length + base64 string. Each set is keyed by the vertex count and a
//! hash of the triangle list, and keeps quantized bind positions so a
//! reimported mesh with a different topology can be remapped by nearest
//! position.

use std::collections::HashMap;
use std::sync::OnceLock;

use glam::Vec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Bytes that serialize as a compact string: `r` + base64 of (run, value)
/// pairs when that is shorter, else `b` + base64 of the bytes. The encoded
/// string is cached until the bytes change.
#[derive(Default)]
pub struct PackedBytes {
    data: Vec<u8>,
    encoded: OnceLock<String>,
}

impl PackedBytes {
    /// Build from raw bytes.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, encoded: OnceLock::new() }
    }

    /// The raw bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Byte length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Mutable access; drops the cached encoding.
    pub fn make_mut(&mut self) -> &mut Vec<u8> {
        self.encoded.take();
        &mut self.data
    }

    /// Replace the bytes.
    pub fn set(&mut self, data: Vec<u8>) {
        *self = Self::new(data);
    }

    /// The serialized form (cached until the bytes change).
    pub fn encoded(&self) -> &str {
        self.encoded.get_or_init(|| pack(&self.data))
    }
}

impl Clone for PackedBytes {
    fn clone(&self) -> Self {
        Self::new(self.data.clone())
    }
}

impl PartialEq for PackedBytes {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl std::fmt::Debug for PackedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PackedBytes({} bytes)", self.data.len())
    }
}

impl Serialize for PackedBytes {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.encoded())
    }
}

impl<'de> Deserialize<'de> for PackedBytes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        match unpack(&s) {
            Some(data) => Ok(Self::new(data)),
            None => {
                tracing::warn!("cloth paint: unreadable packed data ({} chars), dropped", s.len());
                Ok(Self::default())
            }
        }
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = ((c[0] as u32) << 16) | ((*c.get(1).unwrap_or(&0) as u32) << 8) | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                out.push(B64[((n >> (18 - 6 * k)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn unbase64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for ch in s.bytes() {
        if ch == b'=' {
            break;
        }
        let v = B64.iter().position(|&b| b == ch)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

fn rle(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < bytes.len() {
        let v = bytes[k];
        let mut run = 1;
        while k + run < bytes.len() && bytes[k + run] == v && run < 255 {
            run += 1;
        }
        out.extend([run as u8, v]);
        k += run;
    }
    out
}

fn pack(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let runs = rle(bytes);
    if runs.len() < bytes.len() { format!("r{}", base64(&runs)) } else { format!("b{}", base64(bytes)) }
}

fn unpack(s: &str) -> Option<Vec<u8>> {
    if s.is_empty() {
        return Some(Vec::new());
    }
    let body = unbase64(&s[1..])?;
    match s.as_bytes()[0] {
        b'b' => Some(body),
        b'r' => {
            let mut out = Vec::new();
            for pair in body.as_chunks::<2>().0 {
                out.extend(std::iter::repeat_n(pair[1], pair[0] as usize));
            }
            Some(out)
        }
        _ => None,
    }
}

/// FNV-1a over the vertex count and triangle list: stable across runs and
/// builds, so saved paint recognises its mesh.
pub fn topology_hash(vertex_count: usize, indices: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for b in (vertex_count as u64).to_le_bytes() {
        eat(b);
    }
    for i in indices {
        for b in i.to_le_bytes() {
            eat(b);
        }
    }
    h
}

/// Which painted value a brush edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PaintChannel {
    /// 1 = follows the animation, 0 = free.
    #[default]
    Pin,
    /// Share of the cloth's max distance a vertex may stray from its animated
    /// position.
    MaxDistance,
}

/// How a brush fades from its centre to its rim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BrushFalloff {
    /// Cosine falloff (smoothstep).
    #[default]
    Smooth,
    /// Straight-line falloff.
    Linear,
    /// Full strength to the rim.
    Constant,
}

impl BrushFalloff {
    /// Every falloff, in declaration order.
    pub const ALL: [BrushFalloff; 3] = [BrushFalloff::Smooth, BrushFalloff::Linear, BrushFalloff::Constant];

    /// Inspector label of this falloff.
    pub fn label(self) -> &'static str {
        match self {
            BrushFalloff::Smooth => "Smooth",
            BrushFalloff::Linear => "Linear",
            BrushFalloff::Constant => "Constant",
        }
    }

    /// Brush weight at distance `d` from the centre (1 at the centre, 0 at
    /// and past `radius`).
    pub fn weight(self, d: f32, radius: f32) -> f32 {
        if radius <= 0.0 || d >= radius {
            return 0.0;
        }
        let t = (d / radius).max(0.0);
        match self {
            BrushFalloff::Smooth => 1.0 - t * t * (3.0 - 2.0 * t),
            BrushFalloff::Linear => 1.0 - t,
            BrushFalloff::Constant => 1.0,
        }
    }
}

/// What a dab does to the values under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BrushMode {
    /// Toward 1.
    #[default]
    Paint,
    /// Toward 0.
    Erase,
    /// Toward the neighbours' average.
    Smooth,
    /// Toward the brush value.
    Set,
}

impl BrushMode {
    /// Every mode, in declaration order.
    pub const ALL: [BrushMode; 4] = [BrushMode::Paint, BrushMode::Erase, BrushMode::Smooth, BrushMode::Set];

    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            BrushMode::Paint => "Paint",
            BrushMode::Erase => "Erase",
            BrushMode::Smooth => "Smooth",
            BrushMode::Set => "Set",
        }
    }
}

/// One dab: object-space centres (two with mirroring), radius in object units.
#[derive(Clone, Debug)]
pub struct Dab {
    /// Brush centres in object space (two with mirroring).
    pub centers: Vec<Vec3>,
    /// Brush radius (object units).
    pub radius: f32,
    /// How much a full dab applies (0..1).
    pub strength: f32,
    /// How the brush fades to its rim.
    pub falloff: BrushFalloff,
    /// What the dab does.
    pub mode: BrushMode,
    /// The target value (Set mode).
    pub value: f32,
}

/// Mirror a point across the mesh's local X (the YZ plane).
pub fn mirror_x(p: Vec3) -> Vec3 {
    Vec3::new(-p.x, p.y, p.z)
}

/// Render vertices welded by position (UV seam copies share a group), with
/// each group's edge neighbours: brushes and smoothing act per group so seams
/// never split.
#[derive(Clone, Debug, Default)]
pub struct WeldTopology {
    /// Vertex -> group.
    pub group: Vec<u32>,
    /// Group -> its vertices.
    pub members: Vec<Vec<u32>>,
    /// Group -> neighbouring groups.
    pub neighbours: Vec<Vec<u32>>,
    /// Group position (object space, bind pose).
    pub positions: Vec<Vec3>,
}

impl WeldTopology {
    /// Weld by position and link each group's edge neighbours.
    pub fn build(positions: &[Vec3], indices: &[u32]) -> Self {
        let cell = 1e-5;
        let key = |p: Vec3| ((p.x / cell).round() as i64, (p.y / cell).round() as i64, (p.z / cell).round() as i64);
        let mut by_key: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut group = Vec::with_capacity(positions.len());
        let mut members: Vec<Vec<u32>> = Vec::new();
        let mut group_pos = Vec::new();
        for (v, p) in positions.iter().enumerate() {
            let next = members.len() as u32;
            let g = *by_key.entry(key(*p)).or_insert(next);
            if g == next {
                members.push(Vec::new());
                group_pos.push(*p);
            }
            members[g as usize].push(v as u32);
            group.push(g);
        }
        let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); members.len()];
        for tri in indices.as_chunks::<3>().0 {
            if tri.iter().any(|&v| v as usize >= group.len()) {
                continue;
            }
            for k in 0..3 {
                let (a, b) = (group[tri[k] as usize], group[tri[(k + 1) % 3] as usize]);
                if a != b {
                    neighbours[a as usize].push(b);
                    neighbours[b as usize].push(a);
                }
            }
        }
        for n in &mut neighbours {
            n.sort_unstable();
            n.dedup();
        }
        Self { group, members, neighbours, positions: group_pos }
    }

    /// Render vertices in the welded mesh.
    pub fn vertex_count(&self) -> usize {
        self.group.len()
    }

    /// Apply a dab to per-vertex `values` (0 to 1). True when any changed.
    pub fn apply(&self, values: &mut [f32], dab: &Dab) -> bool {
        if values.len() != self.group.len() {
            return false;
        }
        let r2 = dab.radius * dab.radius;
        let hit: Vec<(usize, f32)> = self
            .positions
            .iter()
            .enumerate()
            .filter_map(|(g, p)| {
                let w = dab
                    .centers
                    .iter()
                    .filter(|c| (**c - *p).length_squared() < r2)
                    .map(|c| dab.falloff.weight((*c - *p).length(), dab.radius))
                    .fold(0.0f32, f32::max);
                (w > 0.0).then_some((g, w * dab.strength.clamp(0.0, 1.0)))
            })
            .collect();
        let value_of = |values: &[f32], g: usize| self.members[g].first().map_or(0.0, |&v| values[v as usize]);
        let targets: Vec<f32> = match dab.mode {
            BrushMode::Smooth => hit
                .iter()
                .map(|&(g, _)| {
                    let n = &self.neighbours[g];
                    if n.is_empty() {
                        value_of(values, g)
                    } else {
                        n.iter().map(|&o| value_of(values, o as usize)).sum::<f32>() / n.len() as f32
                    }
                })
                .collect(),
            BrushMode::Paint => vec![1.0; hit.len()],
            BrushMode::Erase => vec![0.0; hit.len()],
            BrushMode::Set => vec![dab.value.clamp(0.0, 1.0); hit.len()],
        };
        let mut changed = false;
        for (&(g, a), target) in hit.iter().zip(targets) {
            let old = value_of(values, g);
            let new = step_toward(old, target, a);
            if new != old {
                changed = true;
                for &v in &self.members[g] {
                    values[v as usize] = new;
                }
            }
        }
        changed
    }
}

/// `old` moved `amount` of the way to `target`, but by at least one byte
/// step when it moves at all, so light strokes still reach the target once
/// quantized.
fn step_toward(old: f32, target: f32, amount: f32) -> f32 {
    if amount <= 0.0 || (old - target).abs() < 0.5 / 255.0 {
        return old;
    }
    let new = old + (target - old) * amount.min(1.0);
    let step = 1.0 / 255.0;
    if (quantize(new) as i32 - quantize(old) as i32).abs() >= 1 {
        new
    } else if target > old {
        (old + step).min(target)
    } else {
        (old - step).max(target)
    }
}

pub fn quantize(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn dequantize(b: u8) -> f32 {
    b as f32 / 255.0
}

/// For each vertex, the vertex nearest its mirror across local X (within
/// `tolerance`), via a coarse grid.
pub fn mirror_map(positions: &[Vec3], tolerance: f32) -> Vec<Option<u32>> {
    let grid = Grid::new(positions, tolerance.max(1e-5));
    positions.iter().map(|p| grid.nearest(positions, mirror_x(*p), tolerance)).collect()
}

/// Uniform hash grid over points for nearest lookups.
struct Grid {
    cell: f32,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl Grid {
    fn new(points: &[Vec3], cell: f32) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (k, p) in points.iter().enumerate() {
            cells.entry(Self::key(cell, *p)).or_default().push(k as u32);
        }
        Self { cell, cells }
    }

    fn key(cell: f32, p: Vec3) -> (i32, i32, i32) {
        ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32)
    }

    /// Nearest point within `max` (any distance when `max` is infinite),
    /// searching outward ring by ring.
    fn nearest(&self, points: &[Vec3], q: Vec3, max: f32) -> Option<u32> {
        if points.is_empty() {
            return None;
        }
        let (cx, cy, cz) = Self::key(self.cell, q);
        let rings = if max.is_finite() { (max / self.cell).ceil() as i32 + 1 } else { 64 };
        let mut best: Option<(f32, u32)> = None;
        for ring in 0..=rings {
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -ring..=ring {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != ring {
                            continue;
                        }
                        let Some(list) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                        for &k in list {
                            let d = (points[k as usize] - q).length_squared();
                            if best.is_none_or(|(bd, _)| d < bd) {
                                best = Some((d, k));
                            }
                        }
                    }
                }
            }
            // Anything in a later ring is at least `ring * cell` away.
            if let Some((d, _)) = best
                && d.sqrt() <= ring as f32 * self.cell
            {
                break;
            }
        }
        if !max.is_finite() && best.is_none() {
            return points
                .iter()
                .enumerate()
                .min_by(|a, b| (*a.1 - q).length_squared().total_cmp(&(*b.1 - q).length_squared()))
                .map(|(k, _)| k as u32);
        }
        best.filter(|(d, _)| d.sqrt() <= max).map(|(_, k)| k)
    }
}

/// What [`ClothPaint::fit`] had to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintFit {
    /// Painted on this mesh.
    Same,
    /// Nothing painted yet: started empty.
    Fresh,
    /// The mesh changed: values were carried over by nearest position.
    Remapped,
}

/// Painted Mesh Cloth values for one submesh, keyed by its topology.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClothPaint {
    /// Render vertices the paint covers.
    pub vertex_count: u32,
    /// [`topology_hash`] of the mesh it was painted on.
    pub topology: u64,
    /// Pin weight per render vertex (255 = pinned).
    pub pins: PackedBytes,
    /// Max distance per render vertex as a share of the cloth's max distance;
    /// empty = not limited.
    pub max_distance: PackedBytes,
    /// Bind-pose positions, 3 x u16 per vertex across `bounds`, for remapping.
    pub positions: PackedBytes,
    /// Object-space bind bounds (min xyz, max xyz) the positions span.
    pub bounds: [f32; 6],
}

impl ClothPaint {
    /// True when nothing is painted.
    pub fn is_empty(&self) -> bool {
        self.vertex_count == 0
    }

    /// True when the paint was made for this mesh (same vertex count and
    /// topology).
    pub fn matches(&self, vertex_count: usize, topology: u64) -> bool {
        self.vertex_count as usize == vertex_count
            && self.topology == topology
            && self.pins.len() == vertex_count
            && (self.max_distance.is_empty() || self.max_distance.len() == vertex_count)
    }

    /// Fresh paint for a mesh from per-vertex pins (0 to 1).
    pub fn seed(bind: &[Vec3], indices: &[u32], pins: &[f32]) -> Self {
        let mut out = Self {
            vertex_count: bind.len() as u32,
            topology: topology_hash(bind.len(), indices),
            pins: PackedBytes::new((0..bind.len()).map(|v| quantize(pins.get(v).copied().unwrap_or(0.0))).collect()),
            ..Default::default()
        };
        out.store_positions(bind);
        out
    }

    fn store_positions(&mut self, bind: &[Vec3]) {
        let (lo, hi) = bind.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let (lo, hi) = if bind.is_empty() { (Vec3::ZERO, Vec3::ZERO) } else { (lo, hi) };
        self.bounds = [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z];
        let span = (hi - lo).max(Vec3::splat(1e-9));
        let mut bytes = Vec::with_capacity(bind.len() * 6);
        for p in bind {
            let t = ((*p - lo) / span).clamp(Vec3::ZERO, Vec3::ONE);
            for c in [t.x, t.y, t.z] {
                bytes.extend(((c * 65535.0).round() as u16).to_le_bytes());
            }
        }
        self.positions.set(bytes);
    }

    /// The saved bind positions, dequantized.
    pub fn stored_positions(&self) -> Vec<Vec3> {
        let lo = Vec3::new(self.bounds[0], self.bounds[1], self.bounds[2]);
        let hi = Vec3::new(self.bounds[3], self.bounds[4], self.bounds[5]);
        let span = hi - lo;
        self.positions
            .as_slice()
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| {
                let u = |k: usize| u16::from_le_bytes([c[k], c[k + 1]]) as f32 / 65535.0;
                lo + Vec3::new(u(0), u(2), u(4)) * span
            })
            .collect()
    }

    /// Make the paint fit this mesh: keep it, start it, or remap it by
    /// nearest stored position when the topology changed.
    pub fn fit(&mut self, bind: &[Vec3], indices: &[u32]) -> PaintFit {
        let topology = topology_hash(bind.len(), indices);
        if self.matches(bind.len(), topology) {
            return PaintFit::Same;
        }
        let old = self.stored_positions();
        if self.is_empty() || old.is_empty() || old.len() != self.pins.len() {
            *self = Self::seed(bind, indices, &[]);
            return PaintFit::Fresh;
        }
        let grid = Grid::new(&old, ((Vec3::from_slice(&self.bounds[3..]) - Vec3::from_slice(&self.bounds[..3])).max_element() / 64.0).max(1e-4));
        let nearest: Vec<usize> = bind.iter().map(|p| grid.nearest(&old, *p, f32::INFINITY).unwrap_or(0) as usize).collect();
        let pick = |src: &PackedBytes| -> Vec<u8> { nearest.iter().map(|&k| src.as_slice().get(k).copied().unwrap_or(0)).collect() };
        let pins = pick(&self.pins);
        let max_distance = if self.max_distance.len() == old.len() { pick(&self.max_distance) } else { Vec::new() };
        *self = Self::seed(bind, indices, &[]);
        self.pins.set(pins);
        self.max_distance.set(max_distance);
        PaintFit::Remapped
    }

    /// Per-vertex values (0 to 1) of a channel; an unpainted max distance
    /// reads as 1 (the full distance).
    pub fn values(&self, channel: PaintChannel) -> Vec<f32> {
        match channel {
            PaintChannel::Pin => self.pins.as_slice().iter().map(|&b| dequantize(b)).collect(),
            PaintChannel::MaxDistance if self.max_distance.is_empty() => vec![1.0; self.vertex_count as usize],
            PaintChannel::MaxDistance => self.max_distance.as_slice().iter().map(|&b| dequantize(b)).collect(),
        }
    }

    /// Write per-vertex values (0..1) of a channel.
    pub fn set_values(&mut self, channel: PaintChannel, values: &[f32]) {
        let bytes = values.iter().map(|&v| quantize(v)).collect();
        match channel {
            PaintChannel::Pin => self.pins.set(bytes),
            PaintChannel::MaxDistance => self.max_distance.set(bytes),
        }
    }

    /// Pin weight per vertex (0..1).
    pub fn pin_weights(&self) -> Vec<f32> {
        self.values(PaintChannel::Pin)
    }

    /// Per-vertex distance limits in metres (`full` = a value of 1), or None
    /// when the max distance isn't painted.
    pub fn limits(&self, full: f32) -> Option<Vec<f32>> {
        (!self.max_distance.is_empty() && full > 0.0)
            .then(|| self.max_distance.as_slice().iter().map(|&b| dequantize(b) * full).collect())
    }

    /// Copy each vertex's value from its mirror on the +X side to the -X side
    /// (`positive_to_negative`) or the other way round. Vertices without a
    /// mirror keep their value.
    pub fn symmetrize(&mut self, channel: PaintChannel, bind: &[Vec3], positive_to_negative: bool, tolerance: f32) {
        let mut values = self.values(channel);
        if values.len() != bind.len() {
            return;
        }
        let map = mirror_map(bind, tolerance);
        let src = values.clone();
        for (v, m) in map.iter().enumerate() {
            let take = if positive_to_negative { bind[v].x < 0.0 } else { bind[v].x > 0.0 };
            if let Some(m) = m
                && take
            {
                values[v] = src[*m as usize];
            }
        }
        self.set_values(channel, &values);
    }
}

/// Vertices whose bind height (local Y) is in the top (`above`) or bottom
/// `share` of the mesh.
pub fn select_by_height(bind: &[Vec3], share: f32, above: bool) -> Vec<bool> {
    let (lo, hi) = bind.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
    let share = share.clamp(0.0, 1.0);
    bind.iter()
        .map(|p| if above { p.y >= hi - (hi - lo) * share - 1e-6 } else { p.y <= lo + (hi - lo) * share + 1e-6 })
        .collect()
}

/// Nearest hit of a ray with a triangle list (both faces): (distance, triangle
/// index, barycentric weights of its three corners).
pub fn ray_triangles(origin: Vec3, dir: Vec3, positions: &[Vec3], indices: &[u32]) -> Option<(f32, usize, Vec3)> {
    let mut best: Option<(f32, usize, Vec3)> = None;
    for (t, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let (Some(a), Some(b), Some(c)) = (positions.get(tri[0] as usize), positions.get(tri[1] as usize), positions.get(tri[2] as usize)) else {
            continue;
        };
        let (e1, e2) = (*b - *a, *c - *a);
        let p = dir.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let s = origin - *a;
        let u = s.dot(p) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = dir.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let d = e2.dot(q) * inv;
        if d > 0.0 && best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, t, Vec3::new(1.0 - u - v, u, v)));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_mesh(n: usize) -> (Vec<Vec3>, Vec<u32>) {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.push(Vec3::new(x as f32 / n as f32 - 0.5, -(y as f32) / n as f32, 0.0));
            }
        }
        let w = n as u32 + 1;
        let mut idx = Vec::new();
        for y in 0..n as u32 {
            for x in 0..n as u32 {
                let i = y * w + x;
                idx.extend([i, i + w, i + 1, i + 1, i + w, i + w + 1]);
            }
        }
        (pos, idx)
    }

    #[test]
    fn falloff_shapes() {
        for f in BrushFalloff::ALL {
            assert_eq!(f.weight(0.0, 1.0), 1.0, "{f:?} centre");
            assert_eq!(f.weight(1.0, 1.0), 0.0, "{f:?} rim");
            assert_eq!(f.weight(2.0, 1.0), 0.0, "{f:?} outside");
            assert_eq!(f.weight(0.1, 0.0), 0.0, "{f:?} zero radius");
        }
        assert!((BrushFalloff::Linear.weight(0.25, 1.0) - 0.75).abs() < 1e-6);
        assert!((BrushFalloff::Smooth.weight(0.5, 1.0) - 0.5).abs() < 1e-6, "smoothstep midpoint");
        assert!(BrushFalloff::Smooth.weight(0.2, 1.0) > BrushFalloff::Linear.weight(0.2, 1.0), "smooth holds the centre");
        assert!(BrushFalloff::Smooth.weight(0.8, 1.0) < BrushFalloff::Linear.weight(0.8, 1.0), "and fades the rim");
        assert_eq!(BrushFalloff::Constant.weight(0.99, 1.0), 1.0);
        let mut prev = 1.0;
        for k in 1..=20 {
            let w = BrushFalloff::Smooth.weight(k as f32 / 20.0, 1.0);
            assert!(w <= prev, "monotonic");
            prev = w;
        }
    }

    #[test]
    fn packed_bytes_round_trip_and_stay_small() {
        for data in [vec![], vec![7u8], vec![0, 255, 3], (0..=255u8).collect::<Vec<_>>()] {
            let p = PackedBytes::new(data.clone());
            assert_eq!(unpack(p.encoded()).unwrap(), data);
        }
        let mut runs = vec![255u8; 3000];
        runs.extend(vec![0u8; 7000]);
        let p = PackedBytes::new(runs.clone());
        assert!(p.encoded().starts_with('r') && p.encoded().len() < 200, "{}", p.encoded().len());
        assert_eq!(unpack(p.encoded()).unwrap(), runs);
        let mut m = p.clone();
        m.make_mut()[0] = 1;
        assert_ne!(m.encoded(), p.encoded(), "edits drop the cached string");
        assert!(unpack("x123").is_none());
    }

    #[test]
    fn dab_paints_erases_sets_and_smooths() {
        let (pos, idx) = grid_mesh(10);
        let topo = WeldTopology::build(&pos, &idx);
        let mut values = vec![0.0; pos.len()];
        let center = pos[5];
        let mut dab = Dab { centers: vec![center], radius: 0.25, strength: 1.0, falloff: BrushFalloff::Constant, mode: BrushMode::Paint, value: 0.5 };
        assert!(topo.apply(&mut values, &dab));
        for (v, p) in pos.iter().enumerate() {
            let inside = (*p - center).length() < 0.25;
            assert_eq!(values[v], if inside { 1.0 } else { 0.0 }, "vertex {v}");
        }
        dab.mode = BrushMode::Erase;
        dab.strength = 0.5;
        dab.falloff = BrushFalloff::Linear;
        topo.apply(&mut values, &dab);
        assert!((values[5] - 0.5).abs() < 1e-6, "half strength at the centre: {}", values[5]);
        dab.mode = BrushMode::Set;
        dab.strength = 1.0;
        dab.falloff = BrushFalloff::Constant;
        topo.apply(&mut values, &dab);
        assert!((values[5] - 0.5).abs() < 1e-6);
        // Smoothing a spike pulls it toward its neighbours.
        let mut spike = vec![0.0; pos.len()];
        spike[60] = 1.0;
        let dab = Dab { centers: vec![pos[60]], radius: 0.01, strength: 1.0, falloff: BrushFalloff::Constant, mode: BrushMode::Smooth, value: 0.0 };
        topo.apply(&mut spike, &dab);
        assert_eq!(spike[60], 0.0);
        // Weak strokes still get there once quantized.
        let mut v = vec![0.99; pos.len()];
        let dab = Dab { centers: vec![pos[0]], radius: 0.01, strength: 0.01, falloff: BrushFalloff::Constant, mode: BrushMode::Paint, value: 0.0 };
        for _ in 0..10 {
            topo.apply(&mut v, &dab);
        }
        assert_eq!(quantize(v[0]), 255);
    }

    #[test]
    fn seams_paint_together() {
        let (mut pos, mut idx) = grid_mesh(4);
        pos.push(pos[12]);
        idx.extend([pos.len() as u32 - 1, 13, 7]);
        let topo = WeldTopology::build(&pos, &idx);
        assert_eq!(topo.members.len(), 25);
        let mut values = vec![0.0; pos.len()];
        let dab = Dab { centers: vec![pos[12]], radius: 0.01, strength: 1.0, falloff: BrushFalloff::Smooth, mode: BrushMode::Paint, value: 0.0 };
        topo.apply(&mut values, &dab);
        assert_eq!(values[12], 1.0);
        assert_eq!(values[pos.len() - 1], 1.0, "the seam copy follows");
    }

    #[test]
    fn mirror_maps_across_local_x() {
        let (pos, idx) = grid_mesh(10);
        let map = mirror_map(&pos, 1e-3);
        for (v, m) in map.iter().enumerate() {
            let m = m.expect("symmetric grid") as usize;
            assert!((pos[m] - mirror_x(pos[v])).length() < 1e-5);
            assert_eq!(map[m], Some(v as u32), "involution");
        }
        // A mirrored dab paints both sides alike.
        let topo = WeldTopology::build(&pos, &idx);
        let mut values = vec![0.0; pos.len()];
        let c = pos[13];
        let dab = Dab { centers: vec![c, mirror_x(c)], radius: 0.15, strength: 0.7, falloff: BrushFalloff::Linear, mode: BrushMode::Paint, value: 0.0 };
        topo.apply(&mut values, &dab);
        for (v, m) in map.iter().enumerate() {
            let (a, b) = (quantize(values[v]) as i32, quantize(values[m.unwrap() as usize]) as i32);
            assert!((a - b).abs() <= 1, "vertex {v}: {a} vs {b}");
        }
        assert!(values[13] > 0.0);
        // A mirror off the mesh finds nothing within tolerance.
        let lopsided = vec![Vec3::new(0.3, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)];
        assert_eq!(mirror_map(&lopsided, 0.01), vec![None, None]);
        let mut paint = ClothPaint::seed(&pos, &idx, &values);
        paint.set_values(PaintChannel::Pin, &(0..pos.len()).map(|v| if pos[v].x > 0.0 { 1.0 } else { 0.0 }).collect::<Vec<_>>());
        paint.symmetrize(PaintChannel::Pin, &pos, true, 1e-3);
        let w = paint.pin_weights();
        for (v, p) in pos.iter().enumerate() {
            assert_eq!(w[v], if p.x.abs() > 1e-6 { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn paint_remaps_when_the_mesh_changes() {
        let (pos, idx) = grid_mesh(8);
        let pins: Vec<f32> = pos.iter().map(|p| if p.y > -0.3 { 1.0 } else { 0.0 }).collect();
        let mut paint = ClothPaint::seed(&pos, &idx, &pins);
        paint.set_values(PaintChannel::MaxDistance, &pos.iter().map(|p| -p.y).collect::<Vec<_>>());
        assert_eq!(paint.clone().fit(&pos, &idx), PaintFit::Same);
        // A denser remesh of the same sheet.
        let (dense, dense_idx) = grid_mesh(16);
        let mut remapped = paint.clone();
        assert_eq!(remapped.fit(&dense, &dense_idx), PaintFit::Remapped);
        assert!(remapped.matches(dense.len(), topology_hash(dense.len(), &dense_idx)));
        let w = remapped.pin_weights();
        let d = remapped.values(PaintChannel::MaxDistance);
        for (v, p) in dense.iter().enumerate() {
            if (p.y + 0.3).abs() > 0.07 {
                assert_eq!(w[v], if p.y > -0.3 { 1.0 } else { 0.0 }, "vertex {v} at {p}");
            }
            assert!((d[v] + p.y).abs() < 0.08, "max distance follows position: {} vs {}", d[v], -p.y);
        }
        // Reordered vertices of the same mesh keep their values by position.
        let order: Vec<usize> = (0..pos.len()).rev().collect();
        let shuffled: Vec<Vec3> = order.iter().map(|&k| pos[k]).collect();
        let remap_idx: Vec<u32> = idx.iter().map(|&i| (pos.len() - 1 - i as usize) as u32).collect();
        let mut again = paint.clone();
        assert_eq!(again.fit(&shuffled, &remap_idx), PaintFit::Remapped);
        let w = again.pin_weights();
        for (v, &k) in order.iter().enumerate() {
            assert_eq!(w[v], pins[k]);
        }
        let mut empty = ClothPaint::default();
        assert_eq!(empty.fit(&pos, &idx), PaintFit::Fresh);
        assert!(empty.pin_weights().iter().all(|&w| w == 0.0) && empty.limits(1.0).is_none());
    }

    #[test]
    fn paint_serializes_compactly_and_round_trips() {
        let (pos, idx) = grid_mesh(30);
        let pins: Vec<f32> = pos.iter().map(|p| if p.y > -0.2 { 1.0 } else { 0.0 }).collect();
        let paint = ClothPaint::seed(&pos, &idx, &pins);
        let text = ron::to_string(&paint).unwrap();
        let back: ClothPaint = ron::from_str(&text).unwrap();
        assert_eq!(back, paint);
        assert!(back.pins.encoded().len() < 40, "runs pack small: {}", back.pins.encoded());
        let p = back.stored_positions();
        assert!(p.iter().zip(&pos).all(|(a, b)| (*a - *b).length() < 1e-4));
        let blank: ClothPaint = ron::from_str("()").unwrap();
        assert!(blank.is_empty());
    }

    #[test]
    fn height_selection_and_ray_hits() {
        let (pos, idx) = grid_mesh(10);
        let top = select_by_height(&pos, 0.2, true);
        assert_eq!(top.iter().filter(|&&t| t).count(), 33, "three rows of eleven");
        let bottom = select_by_height(&pos, 0.0, false);
        assert_eq!(bottom.iter().filter(|&&t| t).count(), 11);
        let (t, tri, bary) = ray_triangles(Vec3::new(0.01, -0.52, 5.0), Vec3::NEG_Z, &pos, &idx).unwrap();
        assert!((t - 5.0).abs() < 1e-5);
        let c = idx[tri * 3..tri * 3 + 3].iter().zip([bary.x, bary.y, bary.z]).map(|(&i, w)| pos[i as usize] * w).sum::<Vec3>();
        assert!((c - Vec3::new(0.01, -0.52, 0.0)).length() < 1e-5);
        assert!(ray_triangles(Vec3::new(0.0, -0.5, 5.0), Vec3::Z, &pos, &idx).is_none(), "pointing away");
        assert!(ray_triangles(Vec3::new(3.0, -0.5, 5.0), Vec3::NEG_Z, &pos, &idx).is_none(), "beside the sheet");
    }
}
